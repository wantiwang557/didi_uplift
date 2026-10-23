# 可直接把整个文件复制到一个 Jupyter 单元格运行（已有 spark 会话）。
# 每类只展示5个完整候选组；不是全量训练样本。
import pandas as pd

query = r"""
-- 2026-10-22: 可查 case 的样本构造草稿，非已验收的训练 SQL。
-- 用户已确认：两个 success_cnt 仅记录当前日期；任一个 >0 表示当天成功。
-- 成功锚点：当天成功日起 [dt, dt+6]；自然日边界待与上游核对。
-- 当前没有真实 round_id：成功前最多6天的失败记录近似作为重试归入下一成功，
-- 已被此前成功窗口覆盖的记录优先归此前窗口；这只是候选匹配，不声称真实批次。
-- 其余失败/control：按用户在 history_start~history_end 内首次记录日起每7天分块。
-- 上述分块只用于去重；未成功样本观察窗口从最终所选入组日开始。
-- 不重算 finish_cnt_7d、不改变源表标签、不汇总多条记录成本。
-- history 范围同时覆盖三个 split；不要分别按 split 重置首次日期。
with params as (
    select
        to_date('2026-05-11') as history_start,
        to_date('2026-09-16') as history_end,
        to_date('2026-07-16') as sample_start,
        to_date('2026-09-10') as sample_end,
        to_date(cast(null as string)) as label_available_through,
        5 as examples_per_type
),
raw_source as (
    select
        to_date(t.dt) as dt,
        cast(t.pid as string) as pid,
        t.flowtree_group_name,
        t.group_name_eve,
        case
            when t.flowtree_group_name = 'DS空白'
             and t.group_name_eve = 'control_group' then 0
            when t.group_name_eve = '留存6张3元' then 1
            when t.group_name_eve = '8元留存5张3' then 2
            when t.group_name_eve = '留存6张5元' then 3
            when t.group_name_eve = '10元留存5张3' then 4
        end as treatment_id,
        t.finish_cnt_7d,
        coalesce(t.send_success_cnt, 0) as send_success_cnt,
        coalesce(t.outcall_success_cnt, 0) as outcall_success_cnt,
        cast(
            coalesce(t.send_cost, 0.0)
            + coalesce(t.send_remind_cost, 0.0)
            + coalesce(t.outcall_cost, 0.0)
            + coalesce(t.remind_cost, 0.0)
            + coalesce(t.msg_coupon_cost_7d, 0.0)
            + coalesce(t.outcall_coupon_cost_7d, 0.0)
            + coalesce(t.outcall_coupon2_cost_7d, 0.0)
            as double
        ) as original_row_cost_7d,
        -- 仅用于识别所用字段相同的重复行。冲突记录只展示，不作为合格样本。
        concat_ws('|',
            coalesce(cast(t.finish_cnt_7d as string), '__NULL__'),
            cast(coalesce(t.send_success_cnt, 0) as string),
            cast(coalesce(t.outcall_success_cnt, 0) as string),
            cast(coalesce(t.send_cost, 0.0) as string),
            cast(coalesce(t.send_remind_cost, 0.0) as string),
            cast(coalesce(t.outcall_cost, 0.0) as string),
            cast(coalesce(t.remind_cost, 0.0) as string),
            cast(coalesce(t.msg_coupon_cost_7d, 0.0) as string),
            cast(coalesce(t.outcall_coupon_cost_7d, 0.0) as string),
            cast(coalesce(t.outcall_coupon2_cost_7d, 0.0) as string)
        ) as source_signature
    from dm.ut_alg_flowback_msg_outcall_label_v2_di t
    cross join params p
    where t.dt between cast(p.history_start as string) and cast(p.history_end as string)
      and t.pid is not null
      and t.new_exp = '迭代流量'
      and (
          t.group_name_eve in ('留存6张3元', '8元留存5张3', '留存6张5元', '10元留存5张3')
          or (t.flowtree_group_name = 'DS空白' and t.group_name_eve = 'control_group')
      )
),
daily as (
    select
        pid, dt, flowtree_group_name, group_name_eve, treatment_id,
        max(finish_cnt_7d) as finish_cnt_7d,
        max(send_success_cnt) as send_success_cnt,
        max(outcall_success_cnt) as outcall_success_cnt,
        max(original_row_cost_7d) as original_row_cost_7d,
        count(*) as source_row_cnt,
        count(distinct source_signature) as source_variant_cnt,
        max(case when finish_cnt_7d is null or finish_cnt_7d < 0 then 1 else 0 end) as bad_label_flag
    from raw_source
    group by pid, dt, flowtree_group_name, group_name_eve, treatment_id
),
daily_flags as (
    select d.*,
        case when send_success_cnt > 0 or outcall_success_cnt > 0 then 1 else 0 end as touch_success,
        case when treatment_id = 0 then cast(0.0 as double) else original_row_cost_7d end as cost_7d,
        count(*) over (partition by pid, dt) as pid_day_key_cnt
    from daily d
),
user_group_counts as (
    select pid, count(distinct treatment_id) as user_treatment_cnt
    from daily_flags
    group by pid
),
success_dates as (
    select distinct pid, dt
    from daily_flags
    where treatment_id <> 0 and touch_success = 1
),
success_neighbors as (
    select pid, dt,
        lag(dt) over (partition by pid order by dt) as prev_success_dt,
        lead(dt) over (partition by pid order by dt) as next_success_dt
    from success_dates
),
success_audit as (
    select *,
        case when datediff(dt, prev_success_dt) < 7
                   or datediff(next_success_dt, dt) < 7 then 1 else 0 end as overlapping_success_flag
    from success_neighbors
),
neighbors as (
    select d.*,
        min(dt) over (partition by pid) as first_observed_dt,
        max(case when treatment_id <> 0 and touch_success = 1 then dt end) over (
            partition by pid order by dt
            rows between unbounded preceding and current row
        ) as previous_or_same_success_dt,
        min(case when treatment_id <> 0 and touch_success = 1 then dt end) over (
            partition by pid order by dt
            rows between current row and unbounded following
        ) as next_or_same_success_dt,
        max(case when source_variant_cnt > 1 or pid_day_key_cnt > 1 then 1 else 0 end)
            over (partition by pid) as user_source_conflict_flag
    from daily_flags d
),
matched as (
    select n.*,
        case
            when treatment_id = 0 then null
            when datediff(dt, previous_or_same_success_dt) between 0 and 6 then previous_or_same_success_dt
            when datediff(next_or_same_success_dt, dt) between 0 and 6 then next_or_same_success_dt
            else null
        end as success_anchor_dt,
        case
            when treatment_id = 0 then 'control_fixed7'
            when touch_success = 1 then 'success_day'
            when datediff(dt, previous_or_same_success_dt) between 0 and 6 then 'inside_success_window'
            when datediff(next_or_same_success_dt, dt) between 0 and 6 then 'retry_before_success_approx'
            else 'failed_fixed7'
        end as assignment_reason,
        date_add(first_observed_dt,
            cast(floor(datediff(dt, first_observed_dt) / 7.0) * 7 as int)
        ) as fallback_block_start_dt
    from neighbors n
),
mapped as (
    select m.*,
        case when success_anchor_dt is not null then 'success'
             when treatment_id = 0 then 'control_fixed7'
             else 'failed_fixed7' end as candidate_kind,
        coalesce(success_anchor_dt, fallback_block_start_dt) as candidate_start_dt
    from matched m
),
candidate_rows as (
    select m.*,
        g.user_treatment_cnt,
        coalesce(a.overlapping_success_flag, 0) as overlapping_success_flag,
        min(m.dt) over (partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt) as first_member_dt,
        max(m.dt) over (partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt) as last_member_dt,
        count(*) over (partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt) as member_cnt,
        sum(case when m.finish_cnt_7d > 0 then 1 else 0 end) over (
            partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt
        ) as positive_member_cnt,
        max(m.bad_label_flag) over (
            partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt
        ) as candidate_bad_label_flag,
        row_number() over (
            partition by m.pid, m.treatment_id, m.candidate_kind, m.candidate_start_dt
            order by
                case when m.finish_cnt_7d > 0 then 0 else 1 end,
                coalesce(m.finish_cnt_7d, -1) desc,
                -- 全负但曾成功：保留成功锚点，不能用其后的无动作行代替。
                case when m.candidate_kind = 'success' and m.dt = m.success_anchor_dt then 0 else 1 end,
                m.dt desc, m.flowtree_group_name, m.group_name_eve
        ) as keep_rank
    from mapped m
    join user_group_counts g on m.pid = g.pid
    left join success_audit a on m.pid = a.pid and m.success_anchor_dt = a.dt
),
representatives as (
    select r.*,
        case when candidate_kind = 'success' then success_anchor_dt else dt end as observation_start_dt,
        case when positive_member_cnt > 1 then 'max_positive_then_anchor_then_latest'
             when positive_member_cnt = 1 then 'only_positive'
             when candidate_kind = 'success' then 'successful_anchor_with_zero_label'
             else 'last_entry_with_zero_label' end as selection_reason,
        case
            when user_source_conflict_flag = 1 or user_treatment_cnt > 1
              or candidate_bad_label_flag = 1 then 'source_problem'
            when overlapping_success_flag = 1 then 'overlapping_success'
            when candidate_kind = 'success' and dt <> success_anchor_dt then 'positive_anchor_mismatch'
            when positive_member_cnt > 1 then 'multiple_positive'
            when candidate_kind = 'success' and finish_cnt_7d > 0 then 'success_positive'
            when candidate_kind = 'success' then 'success_zero'
            when treatment_id = 0 and finish_cnt_7d > 0 then 'control_positive'
            when treatment_id = 0 then 'control_zero'
            when finish_cnt_7d > 0 then 'failed_positive'
            else 'failed_zero'
        end as case_type,
        case when dt between to_date('2026-07-16') and to_date('2026-08-21') then 'train'
             when dt between to_date('2026-08-22') and to_date('2026-08-27') then 'valid'
             when dt between to_date('2026-08-28') and to_date('2026-09-10') then 'test'
             else 'outside_v6' end as split,
        case when dt in (to_date('2026-08-02'), to_date('2026-08-08'), to_date('2026-08-09'))
             then 1 else 0 end as selected_excluded_date_flag,
        case when candidate_kind = 'success' and dt <> success_anchor_dt then 1 else 0 end as label_anchor_mismatch_flag
    from candidate_rows r
    where keep_rank = 1
),
representative_timeline as (
    select r.*,
        max(date_add(observation_start_dt, 6)) over (
            partition by pid order by observation_start_dt, candidate_start_dt, candidate_kind, treatment_id
            rows between unbounded preceding and 1 preceding
        ) as prior_observation_end_dt
    from representatives r
),
eligible_cases as (
    select r.*,
        case when prior_observation_end_dt >= observation_start_dt then 1 else 0 end as observation_overlap_flag,
        case when p.label_available_through is null then 'unknown'
             when dt <= p.label_available_through then 'mature'
             else 'not_mature' end as label_maturity_status,
        row_number() over (partition by case_type order by pid, candidate_start_dt, treatment_id) as case_rank
    from representative_timeline r
    cross join params p
    where r.dt between p.sample_start and p.sample_end
),
case_keys as (
    select c.*
    from eligible_cases c
    cross join params p
    where p.examples_per_type = 0 or c.case_rank <= p.examples_per_type
)
select
    k.case_type, k.case_rank, k.split,
    r.pid, r.treatment_id, r.group_name_eve, r.flowtree_group_name,
    case r.treatment_id when 0 then 0.4916 when 1 then 0.1778
         when 2 then 0.1185 when 3 then 0.0935 when 4 then 0.1186 end as propensity,
    r.candidate_kind,
    r.candidate_start_dt as dedup_block_or_success_start_dt,
    k.observation_start_dt,
    date_add(k.observation_start_dt, 6) as observation_end_dt_assumed,
    r.first_member_dt, r.last_member_dt, r.member_cnt, r.positive_member_cnt,
    r.dt as entry_dt,
    r.send_success_cnt, r.outcall_success_cnt, r.touch_success,
    r.finish_cnt_7d, r.cost_7d,
    r.assignment_reason,
    case when r.keep_rank = 1 then 1 else 0 end as would_keep_flag,
    k.dt as selected_dt,
    k.finish_cnt_7d as selected_finish_cnt_7d,
    k.cost_7d as selected_row_cost_7d,
    k.selection_reason,
    r.source_row_cnt, r.source_variant_cnt,
    r.user_source_conflict_flag, r.user_treatment_cnt,
    r.candidate_bad_label_flag,
    r.overlapping_success_flag,
    k.label_anchor_mismatch_flag,
    k.observation_overlap_flag,
    k.selected_excluded_date_flag,
    k.label_maturity_status,
    case when (r.first_member_dt < to_date('2026-08-22') and r.last_member_dt >= to_date('2026-08-22'))
           or (r.first_member_dt < to_date('2026-08-28') and r.last_member_dt >= to_date('2026-08-28'))
         then 1 else 0 end as crosses_split_boundary_flag,
    'audit_only' as sample_status
from candidate_rows r
join case_keys k
    on r.pid = k.pid
   and r.treatment_id = k.treatment_id
   and r.candidate_kind = k.candidate_kind
   and r.candidate_start_dt = k.candidate_start_dt
order by k.case_type, k.case_rank, r.pid, r.candidate_start_dt, r.dt, r.flowtree_group_name
"""

case_pdf = spark.sql(query).toPandas()
selected_pdf = case_pdf.loc[case_pdf["would_keep_flag"] == 1].copy()

# 先看这些列；全部质量标记仍保存在 case_pdf 中。
case_columns = [
    "case_type", "pid", "treatment_id", "candidate_kind",
    "dedup_block_or_success_start_dt", "observation_start_dt",
    "observation_end_dt_assumed", "entry_dt",
    "send_success_cnt", "outcall_success_cnt", "touch_success",
    "finish_cnt_7d", "would_keep_flag", "selected_dt",
    "assignment_reason", "selection_reason",
    "positive_member_cnt", "label_anchor_mismatch_flag",
    "observation_overlap_flag",
]
with pd.option_context("display.max_rows", 300, "display.max_columns", None):
    display(case_pdf[case_columns])

# 需要只看拟保留样本时，在另一个单元格运行：
# display(selected_pdf)
# 例如只看“失败重试后成功且有完单”的完整case：
# display(case_pdf[case_pdf["case_type"] == "success_positive"])

