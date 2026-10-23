# v6 DFCL SQL 版本清单

更新：2026-10-23。

## 最终保留

正式取数为同一个 `v6_dfcl_mt` 版本下的三个数据集 SQL，三份文件只按原 v6 的日期范围区分：

- `feature_v6_dfcl_mt_train.sql`：train，2026-07-16 ～ 2026-08-21。
- `feature_v6_dfcl_mt_valid.sql`：valid，2026-08-22 ～ 2026-08-27。
- `feature_v6_dfcl_mt_test.sql`：test，2026-08-28 ～ 2026-09-10。

多正周期统计保留一份汇总 SQL：

- `audit_multi_positive_success_summary.sql`：按 split、treatment 汇总成功周期数、多正周期数、多正用户数、多正周期占比，并区分 eligible 口径。

当前状态为 `sql_generated_not_executed`：本地 21 个逻辑测试已通过，但尚未在 Spark/Hive 集群执行和验收。

## 建议暂留的生成与校验文件

这些文件不是另一套 SQL 版本，建议至少保留到首次集群验收完成：

- `build_profile.py`：从原 v6 完整字段投影生成三份最终 SQL 和审计 SQL。
- `selection_ctes.sql`：最终取样逻辑的公共 CTE 源文件。
- `test_profile.py`：本地边界用例测试。
- `profile_manifest.json`：日期范围、源 SQL 哈希、生成文件哈希和交付状态。
- `audit_samples.sql`：集群跑数后的样本量与分布验收。
- `audit_zero_success_then_next_positive.sql`：核对前一候选全负、后一候选转正的用户案例。
- `audit_multi_positive_success_details.sql`：多正周期明细；汇总 SQL 出现异常时用于下钻。

## 已归档的旧版本目录

以下目录均已被 `v6_dfcl_mt` 替代，并于 2026-10-23 移入 `../archive/v6_dfcl_20261023/`；文件仅移动，未删除：

- `../archive/v6_dfcl_20261023/v6_dfcl/`：最早的相邻记录间隔聚类版本，README 已标记不符合最新激励轮次口径。
- `../archive/v6_dfcl_20261023/v6_dfcl_touch_audit/`：按当天触达结果查 case 的草稿，不是正式训练取数版本。
- `../archive/v6_dfcl_20261023/v6_dfcl_touch/`：触达锚点加固定 7 天分块版本，README 已标记被后续版本替代。
- `../archive/v6_dfcl_20261023/v6_dfcl_episode/`：阶段版本；成功触达和归因正记录都可能产出样本，已被 MT 代表行口径替代。
- `../archive/v6_dfcl_20261023/v6_dfcl_cycle/`：中间讨论版本，README 已明确标记停用。

## 不应随上述目录归档

- `../v6/`：原 EUEN v6 基线 SQL，也是最终版本生成时的完整字段投影来源；负样本下采样实验仍依赖该基线口径，因此继续保留。

## 归档结果

五个旧版本目录已经归档。最终目录内的生成器、测试和三份辅助审计 SQL 暂时保留，待集群确认样本量、正负分布、多正周期比例及典型 case 后，再决定是否精简。
