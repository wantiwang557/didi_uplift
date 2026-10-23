-- 统计与 v6_dfcl_touch 相同候选周期口径下，多条正记录的数量和比例。
-- 输出 ALL、success、failed_fixed7、control_fixed7 四行（没有候选的类型不会出现）。
with raw_source as (
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
        concat_ws(
            '|',
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
    where t.dt between '2026-05-11' and '2026-09-16'
      and t.dt not in (
          '2026-03-26',
          '2026-03-27',
          '2026-04-08',
          '2026-08-02',
          '2026-08-08',
          '2026-08-09'
      )
      and t.pid is not null
      and t.new_exp = '迭代流量'
      and (
          t.group_name_eve in (
              '留存6张3元',
              '8元留存5张3',
              '留存6张5元',
              '10元留存5张3'
          )
          or (
              t.flowtree_group_name = 'DS空白'
              and t.group_name_eve = 'control_group'
          )
      )
),

-- 同一天、同一用户、同一实验组的完全相同记录折叠；字段冲突留标记。
daily as (
    select
        pid,
        dt,
        flowtree_group_name,
        group_name_eve,
        treatment_id,
        max(finish_cnt_7d) as finish_cnt_7d,
        max(send_success_cnt) as send_success_cnt,
        max(outcall_success_cnt) as outcall_success_cnt,
        count(distinct source_signature) as source_variant_cnt,
        max(
            case
                when finish_cnt_7d is null or finish_cnt_7d < 0 then 1
                else 0
            end
        ) as bad_label_flag
    from raw_source
    group by
        pid,
        dt,
        flowtree_group_name,
        group_name_eve,
        treatment_id
),

daily_flags as (
    select
        d.*,
        case
            when send_success_cnt > 0 or outcall_success_cnt > 0 then 1
            else 0
        end as touch_success,
        count(*) over (
            partition by pid, dt
        ) as pid_day_key_cnt
    from daily d
),

user_group_counts as (
    select
        pid,
        count(distinct treatment_id) as user_treatment_cnt
    from daily_flags
    group by pid
),

neighbors as (
    select
        d.*,
        min(dt) over (
            partition by pid
        ) as first_observed_dt,
        max(
            case
                when treatment_id <> 0 and touch_success = 1 then dt
            end
        ) over (
            partition by pid
            order by dt
            rows between unbounded preceding and current row
        ) as previous_or_same_success_dt,
        min(
            case
                when treatment_id <> 0 and touch_success = 1 then dt
            end
        ) over (
            partition by pid
            order by dt
            rows between current row and unbounded following
        ) as next_or_same_success_dt
    from daily_flags d
),

mapped as (
    select
        n.*,
        case
            when treatment_id = 0 then null
            when datediff(dt, previous_or_same_success_dt) between 0 and 6
                then previous_or_same_success_dt
            when datediff(next_or_same_success_dt, dt) between 0 and 6
                then next_or_same_success_dt
            else null
        end as success_anchor_dt,
        date_add(
            first_observed_dt,
            cast(
                floor(datediff(dt, first_observed_dt) / 7.0) * 7
                as int
            )
        ) as fallback_block_start_dt
    from neighbors n
),

candidate_rows as (
    select
        m.*,
        g.user_treatment_cnt,
        case
            when success_anchor_dt is not null then 'success'
            when treatment_id = 0 then 'control_fixed7'
            else 'failed_fixed7'
        end as candidate_kind,
        coalesce(success_anchor_dt, fallback_block_start_dt) as candidate_start_dt
    from mapped m
    inner join user_group_counts g
        on m.pid = g.pid
),

candidate_ranked as (
    select
        c.*,
        sum(
            case when finish_cnt_7d > 0 then 1 else 0 end
        ) over (
            partition by
                pid,
                treatment_id,
                candidate_kind,
                candidate_start_dt
        ) as positive_member_cnt,
        max(bad_label_flag) over (
            partition by
                pid,
                treatment_id,
                candidate_kind,
                candidate_start_dt
        ) as candidate_bad_label_flag,
        max(
            case
                when source_variant_cnt > 1 or pid_day_key_cnt > 1 then 1
                else 0
            end
        ) over (
            partition by
                pid,
                treatment_id,
                candidate_kind,
                candidate_start_dt
        ) as candidate_source_conflict_flag,
        row_number() over (
            partition by
                pid,
                treatment_id,
                candidate_kind,
                candidate_start_dt
            order by
                case when finish_cnt_7d > 0 then 0 else 1 end,
                coalesce(finish_cnt_7d, -1) desc,
                case
                    when candidate_kind = 'success'
                     and dt = success_anchor_dt then 0
                    else 1
                end,
                dt desc,
                flowtree_group_name,
                group_name_eve
        ) as keep_rank
    from candidate_rows c
),

candidate_level as (
    select
        pid,
        treatment_id,
        candidate_kind,
        candidate_start_dt,
        max(positive_member_cnt) as positive_member_cnt,
        max(candidate_bad_label_flag) as candidate_bad_label_flag,
        max(candidate_source_conflict_flag) as candidate_source_conflict_flag,
        max(user_treatment_cnt) as user_treatment_cnt,
        max(case when keep_rank = 1 then dt end) as selected_dt,
        max(
            case
                when candidate_kind = 'success'
                 and dt = success_anchor_dt
                 and finish_cnt_7d > 0 then 1
                else 0
            end
        ) as success_anchor_positive_flag,
        max(
            case
                when candidate_kind = 'success'
                 and dt <> success_anchor_dt
                 and finish_cnt_7d > 0 then 1
                else 0
            end
        ) as non_anchor_positive_flag
    from candidate_ranked
    group by
        pid,
        treatment_id,
        candidate_kind,
        candidate_start_dt
),

eligible_candidates as (
    select *
    from candidate_level
    where candidate_bad_label_flag = 0
      and candidate_source_conflict_flag = 0
      and user_treatment_cnt = 1
      and selected_dt between to_date('2026-07-16') and to_date('2026-09-10')
),

summary_by_kind as (
    select
        candidate_kind,
        count(*) as candidate_cnt,
        sum(case when positive_member_cnt >= 1 then 1 else 0 end) as positive_candidate_cnt,
        sum(case when positive_member_cnt > 1 then 1 else 0 end) as multi_positive_candidate_cnt,
        sum(
            case
                when positive_member_cnt > 1
                 and success_anchor_positive_flag = 1
                 and non_anchor_positive_flag = 1 then 1
                else 0
            end
        ) as anchor_and_non_anchor_positive_cnt
    from eligible_candidates
    group by candidate_kind
),

summary_all as (
    select
        'ALL' as candidate_kind,
        count(*) as candidate_cnt,
        sum(case when positive_member_cnt >= 1 then 1 else 0 end) as positive_candidate_cnt,
        sum(case when positive_member_cnt > 1 then 1 else 0 end) as multi_positive_candidate_cnt,
        sum(
            case
                when positive_member_cnt > 1
                 and success_anchor_positive_flag = 1
                 and non_anchor_positive_flag = 1 then 1
                else 0
            end
        ) as anchor_and_non_anchor_positive_cnt
    from eligible_candidates
),

summary as (
    select * from summary_all
    union all
    select * from summary_by_kind
)

select
    candidate_kind,
    candidate_cnt,
    positive_candidate_cnt,
    multi_positive_candidate_cnt,
    anchor_and_non_anchor_positive_cnt,
    case
        when candidate_cnt > 0 then round(
            cast(multi_positive_candidate_cnt as double) / candidate_cnt,
            8
        )
        else null
    end as multi_positive_ratio_all,
    case
        when positive_candidate_cnt > 0 then round(
            cast(multi_positive_candidate_cnt as double) / positive_candidate_cnt,
            8
        )
        else null
    end as multi_positive_ratio_in_positive
from summary
order by
    case when candidate_kind = 'ALL' then 0 else 1 end,
    candidate_kind
