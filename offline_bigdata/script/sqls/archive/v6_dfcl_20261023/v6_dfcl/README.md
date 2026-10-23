# v6_dfcl SQL

> 2026-10-22 状态：本目录的旧 SQL 按相邻记录间隔聚类，已不符合最新确认的激励轮次口径，暂勿用于正式训练。新的可查 case 草稿位于 `../v6_dfcl_touch_audit/`；下文保留为历史说明。

本目录是基于原 v6 新建的 EUEN/DFCL 统一周期样本 profile，不覆盖 `../v6/`。

- `feature_v6_dfcl_train.sql`：输出周期开始日在 2026-07-16 至 2026-08-21 的训练样本。
- `feature_v6_dfcl_valid.sql`：输出周期开始日在 2026-08-22 至 2026-08-27 的验证样本。
- `feature_v6_dfcl_test.sql`：输出周期开始日在 2026-08-28 至 2026-09-10 的测试样本。
- `audit_repeat_entry_finish_cnt_cases.sql`：抽查重复入组周期的 `finish_cnt_7d` 序列及最终保留记录。

核心规则：

- 一个用户的每个 7 天激励周期只输出一条。
- 有正归因记录时保留正记录；多正时按 label 最大、日期最新选择。
- 全负周期保留最后一条。
- 不使用原 train 的 `finish_all_cnt` outcome 筛选。
- control cost 强制为 0。
- 输出字段顺序沿用原 v6，因此可继续复用 train 生成的 parser/feature 配置。

完整设计说明见 `docs/project_background/dfcl_v6_sample_construction.md`。
