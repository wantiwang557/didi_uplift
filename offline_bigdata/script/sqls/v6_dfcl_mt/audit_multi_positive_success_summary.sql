-- 标签取数范围与原 v6 对应的 train、valid、test 保持一致，三个数据集独立构造。
-- 先读取实验组和对照组的入组记录、7天完单标签、触达结果和成本。
--
-- 临时表总览：
-- 1. label_raw：读取并统一 treatment、标签、触达和成本口径。
-- 2. daily：按“用户+日期+组别”形成用户日记录，对照组成本置0。
-- 3. daily_flags：生成当天是否触达成功的标记 touch_success。
-- 4. neighbors：寻找每条记录前后最近的触达成功日。
-- 5. mapped：为记录匹配成功周期锚点 success_anchor_dt。
-- 6. run_neighbors：取得上一入组日和上一行成功锚点。
-- 7. run_numbered：给未成功实验组/control 的连续在组段编号 failed_run_no。
-- 8. candidate_rows：生成候选类型、候选起点和观察窗口标记。
-- 9. ranked：统计候选信息，并给候选内记录生成保留顺序 keep_rank。
-- 10. representatives：每个候选保留 keep_rank=1 的唯一代表行。
-- 11. audited_representatives：补充跨候选观察区间审计字段。
-- 12. selected_keys：保留标签合法、且代表日期属于当前 split 的样本键。
--
-- 关键字段：
-- touch_success：短信或外呼任一成功记为1。
-- success_anchor_dt：成功周期的真实触达成功日；control 和未成功段为空。
-- candidate_kind：success_7d / failed_streak / control_streak。
-- candidate_start_dt：候选唯一标识中的周期/连续段起点。
-- in_observation_window：成功前重试为0，正式候选记录为1。
-- positive_member_cnt：正式候选内 finish_cnt_7d>0 的记录数，用于多正审计。
-- keep_rank：候选内代表行排序，最终只保留 keep_rank=1。
with label_raw as (
    select
        cast(t.pid as string) as pid,
        to_date(t.dt) as dt,
        t.flowtree_group_name,
        t.group_name_eve,
        case
            when t.flowtree_group_name = 'DS空白' and t.group_name_eve = 'control_group' then 0
            when t.group_name_eve = '留存6张3元' then 1
            when t.group_name_eve = '8元留存5张3' then 2
            when t.group_name_eve = '留存6张5元' then 3
            when t.group_name_eve = '10元留存5张3' then 4
        end as treatment_id,
        t.finish_cnt_7d,
        coalesce(t.send_success_cnt, 0) as send_success_cnt,
        coalesce(t.outcall_success_cnt, 0) as outcall_success_cnt,
        cast(coalesce(t.send_cost, 0.0) + coalesce(t.send_remind_cost, 0.0)
             + coalesce(t.outcall_cost, 0.0) + coalesce(t.remind_cost, 0.0)
             + coalesce(t.msg_coupon_cost_7d, 0.0) + coalesce(t.outcall_coupon_cost_7d, 0.0)
             + coalesce(t.outcall_coupon2_cost_7d, 0.0) as double) as row_cost_7d
    from dm.ut_alg_flowback_msg_outcall_label_v2_di t
    where t.dt between '2026-07-16' and '2026-09-10'
      and t.dt not in ('2026-03-26', '2026-03-27', '2026-04-08', '2026-08-02', '2026-08-08', '2026-08-09')
      and t.pid is not null
      and t.new_exp = '迭代流量'
      and (t.group_name_eve in ('留存6张3元', '8元留存5张3', '留存6张5元', '10元留存5张3')
           or (t.flowtree_group_name = 'DS空白' and t.group_name_eve = 'control_group'))
),
daily as (
    -- 业务已确认同一用户同一天只有一条入组记录，且不会跨 treatment。
    -- 此处按完整入组键聚合，仅防止完全重复源行放大样本；对照组成本强制为0。
    select pid, dt, flowtree_group_name, group_name_eve, treatment_id,
        max(finish_cnt_7d) as finish_cnt_7d,
        max(send_success_cnt) as send_success_cnt,
        max(outcall_success_cnt) as outcall_success_cnt,
        case when treatment_id = 0 then cast(0.0 as double)
             else max(row_cost_7d) end as cost_7d,
        max(case when finish_cnt_7d is null or finish_cnt_7d < 0 then 1 else 0 end) as bad_label_flag
    from label_raw
    group by pid, dt, flowtree_group_name, group_name_eve, treatment_id
),
daily_flags as (
    -- 短信或外呼任一成功，即认为当天发券触达成功。
    select d.*,
        case when send_success_cnt > 0 or outcall_success_cnt > 0 then 1 else 0 end as touch_success
    from daily d
),
neighbors as (
    -- 为每条记录寻找前后最近的成功日：前者用于固定成功后的7天窗口，
    -- 后者用于把成功前的失败重试并入后续成功候选，避免额外产生负样本。
    select d.*,
        max(case when treatment_id <> 0 and touch_success = 1 then dt end) over (
            partition by pid order by dt, flowtree_group_name, group_name_eve
            rows between unbounded preceding and current row
        ) as previous_or_same_success_dt,
        min(case when treatment_id <> 0 and touch_success = 1 then dt end) over (
            partition by pid order by dt, flowtree_group_name, group_name_eve
            rows between current row and unbounded following
        ) as next_or_same_success_dt
    from daily_flags d
),
mapped as (
    -- 成功日及其后0～6天归入该成功周期；成功前0～6天的失败重试也标记同一锚点，
    -- 但后续不会把成功前的记录放入正式观察窗口。对照组没有成功锚点。
    select n.*,
        case
            when treatment_id = 0 then null
            when datediff(dt, previous_or_same_success_dt) between 0 and 6
                then previous_or_same_success_dt
            when datediff(next_or_same_success_dt, dt) between 0 and 6
                then next_or_same_success_dt
            else null
        end as success_anchor_dt
    from neighbors n
),
run_neighbors as (
    -- 记录上一入组日及上一行的成功锚点，用于识别未成功连续段的起点。
    select m.*,
        lag(dt) over (partition by pid order by dt, flowtree_group_name, group_name_eve) as previous_entry_dt,
        lag(success_anchor_dt) over (partition by pid order by dt, flowtree_group_name, group_name_eve) as previous_row_anchor_dt
    from mapped m
),
run_numbered as (
    -- 未成功实验组和对照组不按每7个自然日切段。
    -- 从首条未成功记录、成功周期结束后的首条未成功记录，或相邻入组间隔达到28天时开启新段。
    -- 28天仅用于识别可能的重新入组，不代表激励周期长度。
    select n.*,
        sum(case when success_anchor_dt is null and (
                    previous_entry_dt is null or previous_row_anchor_dt is not null
                    or datediff(dt, previous_entry_dt) >= 28
                ) then 1 else 0 end) over (
            partition by pid order by dt, flowtree_group_name, group_name_eve
            rows between unbounded preceding and current row
        ) as failed_run_no
    from run_neighbors n
),
candidate_rows as (
    -- 候选分为三类：成功后固定7天、实验组未成功连续段、对照组连续段。
    -- 成功前失败重试不属于正式观察窗口；正记录按外呼、短信、普通归因标记优先级。
    select r.*,
        case when success_anchor_dt is not null then 'success_7d'
             when treatment_id = 0 then 'control_streak' else 'failed_streak' end as candidate_kind,
        coalesce(success_anchor_dt,
            min(case when success_anchor_dt is null then dt end) over (
                partition by r.pid, failed_run_no
            )) as candidate_start_dt,
        case when success_anchor_dt is null or dt >= success_anchor_dt then 1 else 0 end as in_observation_window,
        case when finish_cnt_7d > 0 and outcall_success_cnt > 0 then 1
             when finish_cnt_7d > 0 and send_success_cnt > 0 then 2
             when finish_cnt_7d > 0 then 3
             else null end as positive_priority_rank
    from run_numbered r
),
ranked as (
    -- 统计候选内的记录数、正记录数和日期范围，并为每个候选选择唯一代表行。
    select c.*,
        count(*) over (partition by pid, candidate_kind, candidate_start_dt) as member_cnt,
        sum(case when in_observation_window = 1 then 1 else 0 end) over (
            partition by pid, candidate_kind, candidate_start_dt
        ) as observation_member_cnt,
        sum(case when in_observation_window = 0 then 1 else 0 end) over (
            partition by pid, candidate_kind, candidate_start_dt
        ) as pre_touch_retry_cnt,
        min(dt) over (partition by pid, candidate_kind, candidate_start_dt) as first_member_dt,
        max(dt) over (partition by pid, candidate_kind, candidate_start_dt) as last_member_dt,
        sum(case when in_observation_window = 1 and finish_cnt_7d > 0 then 1 else 0 end) over (
            partition by pid, candidate_kind, candidate_start_dt
        ) as positive_member_cnt,
        sum(case when in_observation_window = 0 and finish_cnt_7d > 0 then 1 else 0 end) over (
            partition by pid, candidate_kind, candidate_start_dt
        ) as pre_touch_positive_cnt,
        max(case when in_observation_window = 1 then bad_label_flag else 0 end) over (
            partition by pid, candidate_kind, candidate_start_dt
        ) as candidate_bad_label_flag,
        row_number() over (
            partition by pid, candidate_kind, candidate_start_dt
            order by case when in_observation_window = 1 then 0 else 1 end,
                     -- 成功候选固定选择真实触达成功日；窗口内其他正记录只用于审计。
                     case when candidate_kind = 'success_7d' and touch_success = 1
                          then 0 else 1 end,
                     -- 未成功实验组和对照组：有正记录优先选正记录，否则选末次负记录。
                     case when candidate_kind <> 'success_7d' and finish_cnt_7d > 0
                          then 0 else 1 end,
                     coalesce(positive_priority_rank, 4),
                     dt desc, coalesce(finish_cnt_7d, -1) desc,
                     flowtree_group_name, group_name_eve
        ) as keep_rank
    from candidate_rows c
),
representatives as (
    -- 每个候选只保留排序第一的代表行，并记录选择原因和7天观察区间。
    select r.*,
        coalesce(success_anchor_dt, dt) as observation_start_dt,
        date_add(coalesce(success_anchor_dt, dt), 6) as observation_end_dt_assumed,
        case when success_anchor_dt is not null and dt <> success_anchor_dt then 1 else 0 end as label_anchor_mismatch_flag,
        case when finish_cnt_7d <= 0 and success_anchor_dt is not null then 'success_anchor_negative'
             when finish_cnt_7d <= 0 then 'latest_negative_entry'
             when positive_priority_rank = 1 then 'outcall_success_positive'
             when positive_priority_rank = 2 then 'sms_success_positive'
             else 'latest_positive_entry' end as selected_priority_reason,
        case when candidate_bad_label_flag <> 0 then 'invalid_label'
             else 'eligible' end as sample_status
    from ranked r
    where keep_rank = 1
),
audited_representatives as (
    -- 仅为审计计算该用户此前候选的最晚观察结束日，不参与代表行排序。
    select r.*,
        max(observation_end_dt_assumed) over (
            partition by pid order by observation_start_dt, candidate_kind, candidate_start_dt
            rows between unbounded preceding and 1 preceding
        ) as previous_max_observation_end_dt
    from representatives r
),
selected_keys as (
    -- 只保留标签有效且代表日期位于当前 split 的候选。
    select * from audited_representatives
    where sample_status = 'eligible'
      and dt between to_date('2026-07-16') and to_date('2026-09-10')
)
, success_candidates as (
    select
        case when success_anchor_dt between to_date('2026-07-16') and to_date('2026-08-21') then 'train'
             when success_anchor_dt between to_date('2026-08-22') and to_date('2026-08-27') then 'valid'
             when success_anchor_dt between to_date('2026-08-28') and to_date('2026-09-10') then 'test'
             else 'context_only' end as cycle_split,
        treatment_id, pid, positive_member_cnt, sample_status
    from audited_representatives
    where candidate_kind = 'success_7d'
      and success_anchor_dt between to_date('2026-07-16') and to_date('2026-09-10')
), aggregated as (
    select cycle_split, treatment_id,
        count(*) as success_cycle_cnt,
        count(distinct pid) as success_user_cnt,
        sum(case when sample_status = 'eligible' then 1 else 0 end) as eligible_success_cycle_cnt,
        sum(case when positive_member_cnt > 1 then 1 else 0 end) as multi_positive_cycle_cnt,
        count(distinct case when positive_member_cnt > 1 then pid end) as multi_positive_user_cnt,
        sum(case when sample_status = 'eligible' and positive_member_cnt > 1 then 1 else 0 end) as eligible_multi_positive_cycle_cnt,
        sum(case when sample_status = 'eligible' and positive_member_cnt = 0 then 1 else 0 end) as eligible_zero_positive_cycle_cnt,
        sum(case when sample_status = 'eligible' and positive_member_cnt = 1 then 1 else 0 end) as eligible_single_positive_cycle_cnt,
        max(positive_member_cnt) as max_positive_rows_in_cycle
    from success_candidates
    group by cycle_split, treatment_id
)
select *,
    case when success_cycle_cnt = 0 then cast(null as double)
         else cast(multi_positive_cycle_cnt as double) / success_cycle_cnt end as multi_positive_cycle_rate,
    case when eligible_success_cycle_cnt = 0 then cast(null as double)
         else cast(eligible_multi_positive_cycle_cnt as double) / eligible_success_cycle_cnt end as eligible_multi_positive_cycle_rate
from aggregated
order by cycle_split, treatment_id
