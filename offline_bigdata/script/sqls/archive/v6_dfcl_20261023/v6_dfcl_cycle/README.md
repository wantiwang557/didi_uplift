# v6_dfcl_cycle（已停用）

该目录是中间讨论版本，不具备原 v6 的完整固定投影，请勿用于正式抽数。
正式版本请使用 `../v6_dfcl_mt/` 下的 train、valid、test SQL。

该版本严格按已确认业务口径构造样本：当天 `send_success_cnt > 0` 或
`outcall_success_cnt > 0` 即为发券成功日，成功周期固定为成功日起 7 个自然日
`[success_dt, success_dt + 6]`。

## 选样规则

1. 每条成功触达记录产生一条样本，样本特征、标签、成本均取成功当天记录。
2. 成功前的失败重试，以及成功后固定 7 天内的重复入组，不单独产生样本。
3. 同一次连续在组阶段内，最后一个成功周期结束后的未成功尾段只产生一条样本：
   有归因正记录时选正记录；全部为负时选末次入组。
4. 相邻入组记录间隔 `>= 28` 天只用于识别完单出组后的重新入组，不用于划分激励周期。
5. 全时间范围先统一构造样本，再按样本日划分 train/valid/test，避免每个 split
   人工制造一条“末次负样本”。

## 文件

- `feature_v6_dfcl_cycle_all.sql`：统一取数并按 `sample_dt` 产生 train/valid/test。
- `audit_success_cycle_multi_positive.sql`：统计成功 7 天周期内多条正记录的规模和占比。

## 必须验收的边界

- `zero_anchor_but_non_anchor_positive_cycle_cnt`：成功锚点自身为 0、但周期内其他入组行为正。
  这种情况不能简单丢掉，否则会把有完单的成功周期标成负样本；应先查 case。
- 同一用户跨 treatment、同日源表冲突、成功周期重叠，需要在正式落 TFRecord 前继续验收。
- 依据触达成功这一处理后变量选择样本，会使 treatment/control 的决策单元数量不同。
  因此本表可以作为 EUEN 的业务样本版本，但在直接用于 DFCL 的 IPW loss 前，必须验证
  选样后随机分流概率仍适用，或改成所有组统一的决策时点口径。
