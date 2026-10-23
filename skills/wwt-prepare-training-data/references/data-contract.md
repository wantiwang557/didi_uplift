# v6 数据契约与已知限制

核对基准：`offline_bigdata/script/sqls/v6/feature_v6_train.sql`、`feature_confs/v6/gen_feature_conf_v6.sh`。扩展两块来自 `sqls/baseline/feature_baseline_train.sql`，不继承 baseline 的旧 label 表。

## 主数据与口径

- 特征主表：`dm.ut_alg_flowback_msg_outcall_feature_train_di`。
- label 表：`dm.ut_alg_flowback_msg_outcall_label_v2_di`。注意早期文档曾使用不带 v2 的 label 表，本 profile 以 v6 SQL 为准。
- INNER JOIN 键：flowtree_group_name、group_name_eve、dt、pid。集群验收检查两侧唯一、非空及特征样本是否缺 label。
- 过滤：dt >= 2026-03-02、new_exp='迭代流量'；四个 treatment 加 DS空白 中的 control。
- label：finish_cnt_7d；exp_label：send_success_cnt_t 或 outcall_success_cnt > 0。
- cost_7d：send_cost、send_remind_cost、outcall_cost、remind_cost、msg_coupon_cost_7d、outcall_coupon_cost_7d、outcall_coupon2_cost_7d，NULL 按 0 累加后转 double。
- treatment：control=0；留存6张3元=1；8元留存5张3=2；留存6张5元=3；10元留存5张3=4。
- label_dt=t.dt；flag 由 split 决定，分别为 train/valid/test。
- 主表特征的直接来源已核实；上游生产时间、是否存在晚到回刷尚未核实。同日关联不能证明特征在干预前已可用。

## 重要：样本总体随窗口变化

保留原有 `sum(finish_cnt_7d) over(partition by pid)` 和最终过滤：

```sql
(finish_all_cnt > 0 AND finish_cnt_7d > 0) OR finish_all_cnt = 0
```

它使用当前 split 窗口内的 label 决定样本保留，因此既是 outcome 相关筛选，也可能使同一天在长短窗口里的样本不同。这是继承的业务口径，不能把生成结果宣称为未经筛选的全部实验样本，或仅靠日期切分就声称已排除所有泄漏。更改此规则需建立新 profile。

输出候选唯一键采用 pid、group_name_eve、label_dt；尚无线上 schema 证明，集群发现重复会停止，需分析真实粒度，不能自动 dropDuplicates。

## 特征块

| SQL 注释名 | ID | 字段数 | 直接来源 |
|---|---|---:|---|
| sirius画像特征 | sirius | 194 | 特征主表 |
| 补贴环境 | subsidy | 24 | 特征主表 |
| 双端特征 | dual_client | 252 | 特征主表 |
| 有端 | client_score | 2 | 特征主表 |
| 券敏感 | coupon_sensitivity | 1 | 特征主表 |
| 新版小天狼星用户订单信息特征 | xtlx_order | 93 | dm.xtlx_wchs_test_v1 |
| 新版小天狼星用户基础属性特征 | xtlx_user | 43 | dm.xtlx_user_test_v1 |

前五块合计 473；七块合计 609。详情及每个字段的表达式在 feature-catalog.json。

订单表以 passenger_id，用户表以 pid 关联；两者均为 t.dt=date_add(快照.dt,1)，LEFT JOIN。检查快照用户/日期唯一和有效快照日期覆盖。部分用户匹配不到保留 NULL，并报告缺失数；全空特征停止执行。

## 配置和模型输入

label 单列。token 按以下顺序共 15 列：

```
fin_order_wxmini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,exp_label,cost_7d,fake_treatment_id,flag,label_dt
```

含 exp_label，与旧的 14-token 流程不同。train/valid/test 共享 train 配置。仅支持已有 lisan 编码；按选择块顺序及块内字段顺序输出并检查 conf 依赖顺序。不同新任务不复用旧 slots；需要 slots 时基于本次 conf 重新生成。

分箱调用仓库原有 handler；其数值转换、NULL/NaN 处理以冻结的源代码为准。目录的 missing 字段只表示 SQL 保留 NULL，不保证旧 handler 所有阶段都不填充。生成器可能漏写常量特征，配置检查会报错，不自动删列掩盖问题。

## 日期与可复现

日期区间均含首尾，三个请求窗口按顺序且不重叠。业务历史排除保存在 date-exclusions.json，来源 v5/v6 SQL；故障原因尚待补充，不把历史排除泛化到其他 profile。

执行前用户需提供 label_available_through：实际完整产出的样本日期水位，不能仅取当前日期减 7 天。还会检查实际来源数据的日期覆盖与匹配。窗口是否还需要标签观察期 gap 属于实验定义，当前继承 v6 相邻窗口，不擅自新增 gap。

任务目录固定请求、模板、规则、运行源码和二进制 SHA256；运行保存样本快照、fit_sample 和最终 conf。分箱采样用逻辑样本键加 seed 做稳定哈希。原始 Hive 表如果回刷，重新生成快照可能不同；近似分位数跨 Spark 版本也可能不同。已有快照和保存的 conf 才是本次实验依据。
