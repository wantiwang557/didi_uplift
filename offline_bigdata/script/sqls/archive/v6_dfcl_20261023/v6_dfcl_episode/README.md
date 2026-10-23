# v6_dfcl_episode

这是根据最新业务确认建立的新样本 profile。原 `v6`、`v6_dfcl_touch` 不覆盖。

## 最终选样规则

在每个 split 内，先按 `pid` 的相邻记录间隔识别连续在组阶段；间隔 `>= 28` 天视为重新入组的新阶段。

每个阶段中：

1. 当天 `send_success_cnt > 0` 或 `outcall_success_cnt > 0`：保留该条成功触达记录。`finish_cnt_7d = 0` 时它就是成功发券但7天内无完单的负样本。
2. `finish_cnt_7d > 0`：保留上游归因到的正记录。若同一行也是成功触达，只保留一次；若后续完单被归到另一行，则另一行作为下一条归因样本继续保留。
3. 阶段内既没有成功触达，也没有任何正记录：只保留该阶段的末次入组负样本。
4. 普通失败重试记录不单独产生样本。

这套规则不再使用人为的 `failed_fixed7/control_fixed7` 分块，也不再使用原 v6 的 `finish_all_cnt` outcome 筛选。control 的最终 `cost_7d` 强制为 0。

## 文件

- `feature_v6_dfcl_episode_train.sql`：2026-07-16 ～ 2026-08-21
- `feature_v6_dfcl_episode_valid.sql`：2026-08-22 ～ 2026-08-27
- `feature_v6_dfcl_episode_test.sql`：2026-08-28 ～ 2026-09-10
- `episode_stats_query.py`：可直接放到 Jupyter 运行的样本量、选样原因、treatment 占比和多正阶段统计。
- `build_episode_sql.py`：从完整 v6 特征投影机械生成三份 SQL；修改规则时重新生成。

## 重要边界

- train/valid/test 独立计算“末次负样本”。一个跨越 split 边界、始终没有成功和完单的用户，可能在每个 split 各贡献一条末次负样本。这与旧数据按 split 独立抽取的方式一致，但不是全历史只保留一次。
- `finish_cnt_7d` 继续使用上游已经完成的订单归因结果，本 SQL 不从订单明细重算固定7天标签。
- `>=28` 天仅是根据已确认的重新入组规则推断阶段边界；若上游能提供真实 `episode_id`/出组时间，应替换该近似。
- 新样本分布变化后要检查 treatment 占比，但不能把经过成功/完单选样后的占比直接当 propensity。DFCL 仍应使用随机分流阶段估计的原始分流概率，除非重新定义并验证了决策单元。新 train 必须重新拟合特征配置，不能直接把旧 v6 配置当作最终配置。
- 当前只完成 SQL 和本地逻辑验证，尚未在 Spark/Hive 集群验收。
