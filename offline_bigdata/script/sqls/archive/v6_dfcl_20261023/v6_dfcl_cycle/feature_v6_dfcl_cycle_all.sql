-- v6 DFCL / EUEN unified sample construction
--
-- Incentive-cycle definition:
--   1. A successful coupon delivery is observed on the current row when
--      send_success_cnt > 0 OR outcall_success_cnt > 0.
--   2. Its fixed seven-day cycle is [success_dt, success_dt + 6].
--   3. Every successful delivery row contributes exactly one sample.
--   4. Rows before a later successful delivery are retries and do not contribute
--      a separate failed sample.
--   5. After the last successful cycle in one in-group episode, select one
--      unsuccessful-tail sample: prefer an attributed positive row; if all rows
--      are negative, select the last entry row.
--   6. A >= 28-day gap only separates a post-exit re-entry episode. It is NOT an
--      incentive-cycle length.
--
-- The full 2026-05-11 .. 2026-09-10 range is used to reconstruct episodes before
-- assigning train/valid/test by sample_dt. This prevents creating an artificial
-- terminal negative at every split boundary.

with label_raw as (
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
        coalesce(t.finish_cnt_7d, 0) as finish_cnt_7d,
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
        ) as cost_7d,
        concat_ws('|',
            cast(coalesce(t.finish_cnt_7d, 0) as string),
            cast(coalesce(t.send_success_cnt, 0) as string),
            cast(coalesce(t.outcall_success_cnt, 0) as string)
        ) as source_signature
    from dm.ut_alg_flowback_msg_outcall_label_v2_di t
    where t.dt between '2026-05-11' and '2026-09-10'
      and t.dt not between '2026-06-18' and '2026-06-22'
      and t.dt not in ('2026-08-02', '2026-08-08', '2026-08-09')
      and t.new_exp = '迭代流量'
      and t.pid is not null
      and (
          t.group_name_eve in (
              '留存6张3元', '8元留存5张3',
              '留存6张5元', '10元留存5张3'
          )
          or (
              t.flowtree_group_name = 'DS空白'
              and t.group_name_eve = 'control_group'
          )
      )
),

-- The user confirmed there is only one entry per user per day. Keep this
-- aggregation as a defensive audit/deduplication layer.
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
        max(case when treatment_id = 0 then 0.0 else cost_7d end) as cost_7d,
        count(*) as source_row_cnt,
        count(distinct source_signature) as source_variant_cnt
    from label_raw
    group by
        pid, dt, flowtree_group_name, group_name_eve, treatment_id
),

daily_flags as (
    select
        d.*,
        case
            when send_success_cnt > 0 or outcall_success_cnt > 0 then 1
            else 0
        end as touch_success,
        case when finish_cnt_7d > 0 then 1 else 0 end as positive_flag,
        lag(dt) over (partition by pid order by dt, treatment_id) as previous_dt,
        min(treatment_id) over (partition by pid) as user_min_treatment_id,
        max(treatment_id) over (partition by pid) as user_max_treatment_id
    from daily d
),

-- 28 days is used only to recognize re-entry after exit.
episode_starts as (
    select
        d.*,
        case
            when previous_dt is null then 1
            when datediff(dt, previous_dt) >= 28 then 1
            else 0
        end as new_episode_flag
    from daily_flags d
),

episode_rows as (
    select
        e.*,
        sum(new_episode_flag) over (
            partition by pid
            order by dt
            rows between unbounded preceding and current row
        ) as episode_id
    from episode_starts e
    where user_min_treatment_id = user_max_treatment_id
),

episode_context as (
    select
        e.*,
        max(case when touch_success = 1 then dt end) over (
            partition by pid, episode_id
        ) as last_success_dt,
        max(dt) over (
            partition by pid, episode_id
        ) as episode_last_dt
    from episode_rows e
),

-- One row for every real successful-delivery cycle. The successful row itself
-- is the sample anchor and supplies X, label and cost.
success_samples as (
    select
        pid,
        episode_id,
        dt as sample_dt,
        date_add(dt, 6) as cycle_end_dt,
        flowtree_group_name,
        group_name_eve,
        treatment_id,
        finish_cnt_7d as sample_label,
        cost_7d as sample_cost,
        'touch_success_fixed_7d' as sample_kind
    from episode_context
    where touch_success = 1
),

-- Candidate rows for the unsuccessful tail after the LAST successful 7-day
-- cycle. Rows before a later success are retry records and are intentionally
-- excluded. If an episode never succeeds, the whole episode is its failed tail.
failed_tail_candidates as (
    select
        e.*
    from episode_context e
    where e.touch_success = 0
      and (
          e.last_success_dt is null
          or e.dt > date_add(e.last_success_dt, 6)
      )
),

-- In one failed tail: attributed positive first; all-negative tail uses its
-- last entry. Multiple positives are rare boundary cases and are reduced to one
-- by larger finish_cnt_7d, then later attribution date.
failed_tail_ranked as (
    select
        f.*,
        row_number() over (
            partition by pid, episode_id
            order by
                positive_flag desc,
                finish_cnt_7d desc,
                dt desc
        ) as failed_tail_rank
    from failed_tail_candidates f
),

failed_samples as (
    select
        pid,
        episode_id,
        dt as sample_dt,
        date_add(dt, 6) as cycle_end_dt,
        flowtree_group_name,
        group_name_eve,
        treatment_id,
        finish_cnt_7d as sample_label,
        cost_7d as sample_cost,
        case
            when positive_flag = 1 then 'no_touch_attributed_positive'
            else 'no_touch_last_negative'
        end as sample_kind
    from failed_tail_ranked
    where failed_tail_rank = 1
),

selected_samples as (
    select * from success_samples
    union all
    select * from failed_samples
),

samples_with_split as (
    select
        s.*,
        case
            when sample_dt between to_date('2026-07-16') and to_date('2026-08-21')
                then 'train'
            when sample_dt between to_date('2026-08-22') and to_date('2026-08-27')
                then 'valid'
            when sample_dt between to_date('2026-08-28') and to_date('2026-09-10')
                then 'test'
        end as data_split,
        case treatment_id
            when 0 then 0.4916
            when 1 then 0.1778
            when 2 then 0.1185
            when 3 then 0.0935
            when 4 then 0.1186
        end as propensity
    from selected_samples s
)

select
    f.*,
    s.episode_id,
    s.sample_dt,
    s.cycle_end_dt,
    s.sample_label,
    s.sample_cost,
    s.sample_kind,
    s.data_split,
    s.treatment_id as dfcl_treatment_id,
    s.propensity
from samples_with_split s
inner join dm.ut_alg_flowback_msg_outcall_feature_train_di f
    on cast(f.pid as string) = s.pid
   and to_date(f.dt) = s.sample_dt
   and f.flowtree_group_name = s.flowtree_group_name
   and f.group_name_eve = s.group_name_eve
where s.data_split is not null
  and f.new_exp = '迭代流量'
;
