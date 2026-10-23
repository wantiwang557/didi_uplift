-- Audit: how often one fixed seven-day SUCCESS cycle contains >= 2 positive
-- attributed entry rows. Membership is defined by entry dt falling in
-- [success_dt, success_dt + 6], which intentionally captures the user's
-- 7/3-success + 7/5-attributed-positive boundary case.

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
        coalesce(t.outcall_success_cnt, 0) as outcall_success_cnt
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

daily as (
    select
        pid, dt, flowtree_group_name, group_name_eve, treatment_id,
        max(finish_cnt_7d) as finish_cnt_7d,
        max(send_success_cnt) as send_success_cnt,
        max(outcall_success_cnt) as outcall_success_cnt
    from label_raw
    group by pid, dt, flowtree_group_name, group_name_eve, treatment_id
),

success_anchor as (
    select
        pid,
        treatment_id,
        dt as success_dt,
        date_add(dt, 6) as cycle_end_dt,
        finish_cnt_7d as anchor_finish_cnt_7d
    from daily
    where send_success_cnt > 0 or outcall_success_cnt > 0
),

cycle_stats as (
    select
        s.pid,
        s.treatment_id,
        s.success_dt,
        s.cycle_end_dt,
        s.anchor_finish_cnt_7d,
        count(d.dt) as entry_row_cnt,
        sum(case when d.finish_cnt_7d > 0 then 1 else 0 end) as positive_entry_cnt,
        sum(case
                when d.finish_cnt_7d > 0 and d.dt <> s.success_dt then 1
                else 0
            end) as non_anchor_positive_entry_cnt,
        max(case
                when d.finish_cnt_7d > 0 and d.dt <> s.success_dt then 1
                else 0
            end) as has_non_anchor_positive
    from success_anchor s
    inner join daily d
        on d.pid = s.pid
       and d.treatment_id = s.treatment_id
       and d.dt between s.success_dt and s.cycle_end_dt
    group by
        s.pid, s.treatment_id, s.success_dt, s.cycle_end_dt,
        s.anchor_finish_cnt_7d
),

summary as (
    select
        count(*) as success_cycle_cnt,
        sum(case when positive_entry_cnt > 0 then 1 else 0 end)
            as positive_success_cycle_cnt,
        sum(case when positive_entry_cnt >= 2 then 1 else 0 end)
            as multi_positive_success_cycle_cnt,
        sum(case
                when anchor_finish_cnt_7d > 0 and has_non_anchor_positive = 1
                then 1 else 0
            end) as anchor_and_non_anchor_positive_cycle_cnt,
        sum(case
                when anchor_finish_cnt_7d = 0 and has_non_anchor_positive = 1
                then 1 else 0
            end) as zero_anchor_but_non_anchor_positive_cycle_cnt
    from cycle_stats
)

select
    success_cycle_cnt,
    positive_success_cycle_cnt,
    multi_positive_success_cycle_cnt,
    anchor_and_non_anchor_positive_cycle_cnt,
    zero_anchor_but_non_anchor_positive_cycle_cnt,
    cast(multi_positive_success_cycle_cnt as double)
        / nullif(success_cycle_cnt, 0) as multi_positive_ratio_all,
    cast(multi_positive_success_cycle_cnt as double)
        / nullif(positive_success_cycle_cnt, 0) as multi_positive_ratio_in_positive
from summary
;

-- To inspect concrete cases, replace the final SELECT above with:
-- select *
-- from cycle_stats
-- where positive_entry_cnt >= 2
-- order by pid, success_dt
-- limit 200;
