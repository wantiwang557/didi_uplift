# DFCL v6 统一样本构造方案

> 2026-10-23 更新：以下主体保留为旧方案背景记录，不能直接作为当前取数口径。当前最终 SQL 版本及准确口径见 [v6_dfcl_mt README](../../offline_bigdata/script/sqls/v6_dfcl_mt/README.md)，版本清单见 [SQL_INVENTORY](../../offline_bigdata/script/sqls/v6_dfcl_mt/SQL_INVENTORY.md)。当前 SQL 已通过本地逻辑测试，但尚未完成 Spark/Hive 集群验收。

## 1. 目标

本方案为 EUEN 与 DFCL 构造同一份基础样本。样本粒度不是“一个用户只保留一次”，而是：

> 一个用户 × 一个 7 天激励周期 = 一条基础样本。

同一个用户在 train、valid、test 各时间段内可以有多个相互独立的周期，所有合法周期都保留。

## 2. 已确认的业务口径

- 用户所属实验组由年初实验框架固定，后续入组仍进入同一个 treatment，不会在 treatment 之间切换。
- treatment 共 5 个：
  - 0：control，原始组名为 `DS空白 + control_group`
  - 1：`留存6张3元`
  - 2：`8元留存5张3`
  - 3：`留存6张5元`
  - 4：`10元留存5张3`
- 券有效期为 7 天。用户在同一个激励期内可能因为未完成 APP 订单、未成功触达而重复入组。
- 用户出组后，业务上约 28 天后才可能开始下一次激励周期。
- 完单先按“向前找 7 天入组记录”的归因逻辑归给某一条入组；`finish_cnt_7d` 是该入组记录被归因到的完单数。
- control 的 `cost_7d` 固定为 0。

## 3. 实验分流概率

使用已审核通过的整个窗口分流比例，并统一保留 4 位小数：

- treatment 0：0.4916
- treatment 1：0.1778
- treatment 2：0.1185
- treatment 3：0.0935
- treatment 4：0.1186

五个概率相加为 1.0000。SQL 不额外增加 `propensity` 输出列，以保持原 v6 特征 schema 不变；DFCL 训练时根据 `fake_treatment_id` 查上述常量。

## 4. 时间切分

- train：周期开始日 2026-07-16 至 2026-08-21
- valid：周期开始日 2026-08-22 至 2026-08-27
- test：周期开始日 2026-08-28 至 2026-09-10

每份 SQL 会额外向前读取 7 天数据，只用于识别跨切分边界的旧周期，最终不会把这些 lookback 周期输出。例如 valid 从 2026-08-15 开始读取，但只输出周期开始日在 2026-08-22 至 2026-08-27 的样本。

## 5. 基础过滤条件

三份 SQL 均沿用 v6 的特征表、标签表与连接键：

- 特征表：`dm.ut_alg_flowback_msg_outcall_feature_train_di`
- 标签表：`dm.ut_alg_flowback_msg_outcall_label_v2_di`
- 连接键：`flowtree_group_name + group_name_eve + dt + pid`

保留条件：

- `new_exp = '迭代流量'`
- `pid is not null`
- 仅保留 control 和 4 个目标 treatment
- 沿用原 v6 的异常日期排除规则
- train、valid、test 使用相同字段定义和相同周期去重规则

新 profile 明确删除原 train SQL 的 outcome 筛选：

```sql
(finish_all_cnt > 0 and finish_cnt_7d > 0)
or finish_all_cnt = 0
```

原因是新方案先构造完整周期，再在周期内选择代表记录；不能先删除部分负记录，否则会改变周期结构。EUEN 如需处理正负样本不均衡，应在模型训练阶段采样或加权。

## 6. 重复记录如何处理

### 6.1 同一天完全重复

不使用 `select distinct` 静默吞掉冲突。源数据应满足同一个 `pid + dt + treatment` 至多一条；如果出现多条，应先作为数据质量问题检查 join 是否膨胀、标签或特征是否冲突。

### 6.2 7 天内重复入组

先按用户和日期排序：

- 用户第一条记录开启新周期；
- 与上一条记录间隔小于 7 天，视为同一周期内重复入组；
- 间隔大于等于 7 天，开启新周期。

业务上重新入组通常间隔约 28 天，因此“大于等于 7 天”是识别新周期的保守边界。应在验收时检查单个周期的首末日期跨度不超过 6 天，避免异常连续记录被链式合并。

### 6.3 一个周期最终保留哪条

按 MT 确认的实用口径：

1. 周期内存在 `finish_cnt_7d > 0`：保留正样本记录。
2. 如果罕见地存在多条正记录：先保留 `finish_cnt_7d` 最大的；相同则保留日期较晚的。
3. 周期内全部为负：保留日期最晚的一条。
4. 被保留记录自身的特征、`finish_cnt_7d`、触达标记和 `cost_7d` 一起进入模型。
5. control 的 `cost_7d` 在最外层强制改为 0。

多正样本是已知但占比较小的边界情况。v1 不累加不同入组行上的 label/cost，避免把不同归因记录重复合并；上线前仍应输出多正周期数量和占比作为验收指标。

## 7. EUEN 与 DFCL 分别如何使用

### EUEN

- 使用统一基础样本中的特征、实际 treatment、`label = finish_cnt_7d` 和 `cost_7d`。
- train 阶段可以做负样本下采样、分层采样或 loss 加权。
- valid/test 不采样，保持自然分布。
- 如果 EUEN 与 DFCL 联合训练，不要在 SQL 层物理删样本；建议用 prediction loss 的 sample weight/mask，让 DFCL loss 仍看到完整样本。

### DFCL

- 使用同一基础样本，不再单独构造另一套日期或标签。
- 每个周期样本用实际 treatment 对应的 propensity：
  `p = [0.4916, 0.1778, 0.1185, 0.0935, 0.1186][fake_treatment_id]`。
- DFCL loss 使用完整、未经过 EUEN 正负采样的周期样本。
- control cost 为 0；其他 treatment 使用被保留入组行的 `cost_7d`。

因此，统一的是“基础样本及数据切分”；不同的是训练时的 loss 和权重：

- EUEN：prediction loss，可做训练期采样或加权。
- DFCL：decision-focused/IPW loss，使用完整基础样本和固定 propensity。

## 8. SQL 文件

- `offline_bigdata/script/sqls/v6_dfcl_mt/feature_v6_dfcl_mt_train.sql`
- `offline_bigdata/script/sqls/v6_dfcl_mt/feature_v6_dfcl_mt_valid.sql`
- `offline_bigdata/script/sqls/v6_dfcl_mt/feature_v6_dfcl_mt_test.sql`
- `offline_bigdata/script/sqls/v6_dfcl_mt/audit_multi_positive_success_summary.sql`

原 `offline_bigdata/script/sqls/v6/` 文件未修改，便于和旧 v6 结果做 A/B 对照。

## 9. 运行前验收

至少检查以下指标：

- 每个 `pid + label_dt + treatment` 是否唯一。
- 是否仍有用户跨 treatment。
- 每个 `pid + cycle_no` 最终是否恰好一条。
- 周期首末日期跨度是否小于 7 天。
- 多正周期数、多正周期用户数及其比例。
- train/valid/test 的周期开始日是否互斥。
- 五组样本数和 propensity 是否接近历史稳定比例。
- control 的 `cost_7d` 是否全部等于 0。
- EUEN 采样前后样本数、正负比例，以及 DFCL 输入样本数是否保持为采样前数量。

