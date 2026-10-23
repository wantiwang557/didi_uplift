-- DFCL v5：RCT 分组重复进入与 propensity 审计
--
-- 口径：
--   1. 数据源使用未经过 outcome 筛选、未关联特征的原始 label 表。
--   2. 日期及异常日期过滤与 feature_v5_train.sql 保持一致。
--   3. 一个实验分配单元暂定义为 pid + dt + treatment_id。
--   4. 输出同时包含：
--        - 同一 pid 是否在不足 7 天内重复出现同一个 treatment；
--        - 上述同组重复的记录数和用户数占比；
--        - 同一 pid 跨整个窗口多次进组/跨组的参考指标；
--        - 同一 pid + dt 同时出现多个 treatment 的冲突数/比例；
--        - 每日各组分流概率；
--        - 整个 v5 窗口各组总体分流概率。
--
-- 注意：
--   如果业务随机化单元不是 pid + dt，而是 pid + experiment_id，
--   应用真实 experiment_id 替换 dt 后再计算。

with raw_assignment as (
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
            when group_name_eve = '留存6张3元' then '留存6张3元'
            when group_name_eve = '8元留存5张3' then '8元留存5张3'
            when group_name_eve = '留存6张5元' then '留存6张5元'
            when group_name_eve = '10元留存5张3' then '10元留存5张3'
        end as treatment_name
    from dm.ut_alg_flowback_msg_outcall_label_v2_di
    where dt between '2026-05-11' and '2026-08-21'
      and dt not between '2026-06-18' and '2026-06-22'
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
              '留存6张5元',
              '留存6张3元',
              '8元留存5张3',
              '10元留存5张3'
          )
          or (
              flowtree_group_name = 'DS空白'
              and group_name_eve = 'control_group'
          )
      )
),

-- 去掉源表中完全重复的 pid + dt + treatment 记录。
assignment as (
    select distinct
        dt,
        pid,
        treatment_id,
        treatment_name
    from raw_assignment
    where treatment_id is not null
),

user_summary as (
    select
        pid,
        count(distinct dt) as entry_day_cnt,
        count(distinct treatment_id) as treatment_cnt
    from assignment
    group by pid
),

user_day_summary as (
    select
        dt,
        pid,
        count(distinct treatment_id) as treatment_cnt
    from assignment
    group by dt, pid
),

-- 用户在同一个 treatment 下按日期排序。
-- 这里专门检查用户关心的“7 天内是否多次出现同组记录”。
same_treatment_ordered as (
    select
        dt,
        pid,
        treatment_id,
        treatment_name,
        lag(dt, 1) over (
            partition by pid, treatment_id
            order by dt
        ) as previous_same_treatment_dt
    from assignment
),

same_treatment_7d_flag as (
    select
        dt,
        pid,
        treatment_id,
        treatment_name,
        previous_same_treatment_dt,
        case
            when previous_same_treatment_dt is not null
             and datediff(to_date(dt), to_date(previous_same_treatment_dt)) < 7
                then 1
            else 0
        end as repeat_same_treatment_7d_flag
    from same_treatment_ordered
),

-- 理论规则：用户从某次进组日起，接下来的 6 个自然日内不应再次进组；
-- 即相邻两次进组日期 datediff < 7 时，视为落在同一个 7 天冷却窗口。
-- 先按用户和日期排序，只要存在任意两次不足 7 天，相邻记录一定也会满足该条件。
ordered_assignment as (
    select
        dt,
        pid,
        treatment_id,
        lag(dt, 1) over (
            partition by pid
            order by dt, treatment_id
        ) as previous_dt,
        lag(treatment_id, 1) over (
            partition by pid
            order by dt, treatment_id
        ) as previous_treatment_id
    from assignment
),

seven_day_user_summary as (
    select
        pid,
        max(
            case
                when previous_dt is not null
                 and datediff(to_date(dt), to_date(previous_dt)) < 7
                then 1 else 0
            end
        ) as repeat_entry_within_7d,
        max(
            case
                when previous_dt is not null
                 and datediff(to_date(dt), to_date(previous_dt)) < 7
                 and treatment_id <> previous_treatment_id
                then 1 else 0
            end
        ) as cross_treatment_within_7d
    from ordered_assignment
    group by pid
),

base_total as (
    select
        count(*) as user_cnt,
        sum(case when entry_day_cnt > 1 then 1 else 0 end) as repeat_entry_user_cnt,
        sum(case when treatment_cnt > 1 then 1 else 0 end) as cross_treatment_user_cnt
    from user_summary
),

seven_day_total as (
    select
        count(*) as user_cnt,
        sum(repeat_entry_within_7d) as repeat_entry_user_cnt,
        sum(cross_treatment_within_7d) as cross_treatment_user_cnt
    from seven_day_user_summary
),

same_treatment_7d_total as (
    select
        count(*) as user_day_arm_cnt,
        sum(repeat_same_treatment_7d_flag) as repeat_record_cnt,
        count(distinct pid) as user_cnt,
        count(distinct case
            when repeat_same_treatment_7d_flag = 1 then pid
        end) as repeat_user_cnt
    from same_treatment_7d_flag
),

user_day_total as (
    select
        count(*) as user_day_cnt,
        sum(case when treatment_cnt > 1 then 1 else 0 end) as conflict_user_day_cnt
    from user_day_summary
),

source_row_total as (
    select count(*) as raw_row_cnt
    from raw_assignment
),

dedup_row_total as (
    select count(*) as dedup_user_day_arm_cnt
    from assignment
),

daily_group_count as (
    select
        dt,
        treatment_id,
        treatment_name,
        count(*) as sample_cnt
    from assignment
    group by dt, treatment_id, treatment_name
),

daily_propensity as (
    select
        dt,
        treatment_id,
        treatment_name,
        sample_cnt,
        sum(sample_cnt) over (partition by dt) as total_cnt
    from daily_group_count
),

window_group_count as (
    select
        treatment_id,
        treatment_name,
        count(*) as sample_cnt
    from assignment
    group by treatment_id, treatment_name
),

window_propensity as (
    select
        treatment_id,
        treatment_name,
        sample_cnt,
        sum(sample_cnt) over () as total_cnt
    from window_group_count
),

audit_metrics as (
    -- 指标 1：有多少用户在不足 7 天内重复出现同一个 treatment。
    select
        '01_same_treatment_repeat_7d_user' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(repeat_user_cnt as bigint) as numerator,
        cast(user_cnt as bigint) as denominator,
        case
            when user_cnt > 0
                then cast(repeat_user_cnt as double) / user_cnt
        end as ratio,
        '存在相邻同组记录日期差<7天的pid / 窗口去重pid' as metric_desc
    from same_treatment_7d_total

    union all

    -- 指标 2：不足 7 天内重复出现的同组记录占全部用户日组记录的比例。
    select
        '02_same_treatment_repeat_7d_record' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(repeat_record_cnt as bigint) as numerator,
        cast(user_day_arm_cnt as bigint) as denominator,
        case
            when user_day_arm_cnt > 0
                then cast(repeat_record_cnt as double) / user_day_arm_cnt
        end as ratio,
        '与上一条同组记录日期差<7天的记录 / 全部去重用户日组记录' as metric_desc
    from same_treatment_7d_total

    union all

    -- 指标 3：整个窗口内跨日期重复进入实验，仅作为参考；间隔满 7 天可属正常。
    select
        '03_repeat_entry_in_full_window' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(repeat_entry_user_cnt as bigint) as numerator,
        cast(user_cnt as bigint) as denominator,
        case
            when user_cnt > 0
                then cast(repeat_entry_user_cnt as double) / user_cnt
        end as ratio,
        '整个窗口进入实验超过1天的pid / 窗口去重pid，仅供参考' as metric_desc
    from base_total

    union all

    -- 指标 4：整个窗口内进入过多个 treatment，仅作为参考。
    select
        '04_cross_treatment_in_full_window' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(cross_treatment_user_cnt as bigint) as numerator,
        cast(user_cnt as bigint) as denominator,
        case
            when user_cnt > 0
                then cast(cross_treatment_user_cnt as double) / user_cnt
        end as ratio,
        '整个窗口进入超过1个不同treatment的pid / 窗口去重pid，仅供参考' as metric_desc
    from base_total

    union all

    -- 指标 5：同一个用户在同一天是否同时出现多个 treatment。
    select
        '05_same_day_treatment_conflict' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(conflict_user_day_cnt as bigint) as numerator,
        cast(user_day_cnt as bigint) as denominator,
        case
            when user_day_cnt > 0
                then cast(conflict_user_day_cnt as double) / user_day_cnt
        end as ratio,
        '同一pid+dt出现多个treatment的用户日 / 全部去重用户日' as metric_desc
    from user_day_total

    union all

    -- 指标 6：原始表完全重复记录的规模。
    select
        '06_exact_duplicate_row' as metric_type,
        cast(null as string) as dt,
        cast(null as int) as treatment_id,
        cast(null as string) as treatment_name,
        cast(raw_row_cnt - dedup_user_day_arm_cnt as bigint) as numerator,
        cast(raw_row_cnt as bigint) as denominator,
        case
            when raw_row_cnt > 0
                then cast(raw_row_cnt - dedup_user_day_arm_cnt as double) / raw_row_cnt
        end as ratio,
        '被pid+dt+treatment去重掉的记录 / 原始记录' as metric_desc
    from source_row_total
    cross join dedup_row_total

    union all

    -- 指标 7：每日实验分流概率。
    select
        '07_daily_propensity' as metric_type,
        dt,
        treatment_id,
        treatment_name,
        cast(sample_cnt as bigint) as numerator,
        cast(total_cnt as bigint) as denominator,
        case
            when total_cnt > 0 then cast(sample_cnt as double) / total_cnt
        end as ratio,
        '当日该实验组用户日数 / 当日全部实验组用户日数' as metric_desc
    from daily_propensity

    union all

    -- 指标 8：整个 v5 训练窗口的总体实验分流概率。
    select
        '08_window_propensity' as metric_type,
        cast(null as string) as dt,
        treatment_id,
        treatment_name,
        cast(sample_cnt as bigint) as numerator,
        cast(total_cnt as bigint) as denominator,
        case
            when total_cnt > 0 then cast(sample_cnt as double) / total_cnt
        end as ratio,
        '窗口内该实验组用户日数 / 窗口内全部实验组用户日数' as metric_desc
    from window_propensity
)

select
    metric_type,
    dt,
    treatment_id,
    treatment_name,
    numerator,
    denominator,
    ratio,
    metric_desc
from audit_metrics
order by metric_type, dt, treatment_id;
