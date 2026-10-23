-- 查看 v6 train 窗口内：同一用户、同一 7 天激励周期重复入组时，
-- 每条入组记录的 finish_cnt_7d，以及按 v6_dfcl 规则最终会保留哪一条。
--
-- 在 Jupyter 中执行：
-- repeat_case_pdf = spark.sql("""<本文件 SQL>""").toPandas()
--
-- case_type:
--   all_negative          周期内全部 finish_cnt_7d = 0
--   one_positive          周期内恰好一条 finish_cnt_7d > 0（通常是预期的 0/0/x）
--   multiple_positive     周期内多条 finish_cnt_7d > 0（需要重点抽查）

with base as (
    select
        dt,
        pid,
        case
            when flowtree_group_name = 'DS空白'
             and group_name_eve = 'control_group' then 0
            when group_name_eve = '留存6张3元' then 1
            when group_name_eve = '8元留存5张3' then 2
            when group_name_eve = '留存6张5元' then 3
            when group_name_eve = '10元留存5张3' then 4
        end as treatment_id,
        case
            when flowtree_group_name = 'DS空白'
             and group_name_eve = 'control_group' then 'DS空白'
            else group_name_eve
        end as treatment_name,
        coalesce(finish_cnt_7d, 0) as finish_cnt_7d,
        coalesce(send_success_cnt, 0) as send_success_cnt,
        coalesce(outcall_success_cnt, 0) as outcall_success_cnt,
        count(*) over (
            partition by pid, dt, flowtree_group_name, group_name_eve
        ) as same_day_row_cnt
    from dm.ut_alg_flowback_msg_outcall_label_v2_di
    where dt between '2026-07-09' and '2026-08-21'
      and dt not in (
          '2026-03-26',
          '2026-03-27',
          '2026-04-08',
          '2026-08-02',
          '2026-08-08',
          '2026-08-09'
      )
      and new_exp = '迭代流量'
      and pid is not null
      and (
          group_name_eve in (
              '留存6张3元',
              '8元留存5张3',
              '留存6张5元',
              '10元留存5张3'
          )
          or (
              flowtree_group_name = 'DS空白'
              and group_name_eve = 'control_group'
          )
      )
),
ordered_rows as (
    select
        b.*,
        lag(dt) over (
            partition by pid
            order by dt, treatment_id
        ) as previous_dt
    from base b
),
cycle_marked as (
    select
        b.*,
        case
            when previous_dt is null then 1
            when datediff(to_date(dt), to_date(previous_dt)) >= 7 then 1
            else 0
        end as is_new_cycle
    from ordered_rows b
),
cycle_numbered as (
    select
        b.*,
        sum(is_new_cycle) over (
            partition by pid
            order by dt, treatment_id
            rows between unbounded preceding and current row
        ) as cycle_no
    from cycle_marked b
),
cycle_stats as (
    select
        b.*,
        min(dt) over (
            partition by pid, cycle_no
        ) as cycle_start_dt,
        max(dt) over (
            partition by pid, cycle_no
        ) as cycle_end_dt,
        count(*) over (
            partition by pid, cycle_no
        ) as entry_cnt,
        sum(
            case when finish_cnt_7d > 0 then 1 else 0 end
        ) over (
            partition by pid, cycle_no
        ) as positive_entry_cnt
    from cycle_numbered b
),
ranked_rows as (
    select
        b.*,
        row_number() over (
            partition by pid, cycle_no
            order by
                case when finish_cnt_7d > 0 then 0 else 1 end,
                finish_cnt_7d desc,
                dt desc
        ) as keep_rank
    from cycle_stats b
)
select
    pid,
    cycle_no,
    cycle_start_dt,
    cycle_end_dt,
    treatment_id,
    treatment_name,
    entry_cnt,
    positive_entry_cnt,
    case
        when positive_entry_cnt = 0 then 'all_negative'
        when positive_entry_cnt = 1 then 'one_positive'
        else 'multiple_positive'
    end as case_type,
    dt as entry_dt,
    finish_cnt_7d,
    send_success_cnt,
    outcall_success_cnt,
    same_day_row_cnt,
    case when keep_rank = 1 then 1 else 0 end as would_keep_flag
from ranked_rows
where entry_cnt > 1
  and cycle_start_dt between '2026-07-16' and '2026-08-21'
order by
    case
        when positive_entry_cnt > 1 then 0
        when positive_entry_cnt = 1 then 1
        else 2
    end,
    pid,
    cycle_no,
    entry_dt
limit 500

