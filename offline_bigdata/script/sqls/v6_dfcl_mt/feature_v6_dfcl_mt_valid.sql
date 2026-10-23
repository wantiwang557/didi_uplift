-- v6_dfcl_mt valid：成功候选固定选择触达成功日；未成功实验组和对照组有正选正、全负选末次。
-- 未成功连续段不按每7个自然日切分，边界定义见 README.md。
-- 输出字段与原 v6 一致，代表行同时提供特征、标签、曝光和成本。
-- 本文件由 build_profile.py 生成，正式抽数前应先运行 audit_samples.sql。
-- 代表行去重不等于 DFCL/IPW 已满足无偏条件，分流概率和选择机制需要另行验证。
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
    where t.dt between '2026-08-22' and '2026-08-27'
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
      and dt between to_date('2026-08-22') and to_date('2026-08-27')
),
-- 读取当前 split 的原 v6 特征，并统计连接键对应的特征行数，防止连接放大样本。
feature_source as (
    select f.*,
        count(*) over (partition by pid, dt, flowtree_group_name, group_name_eve) as source_feature_row_cnt
    from dm.ut_alg_flowback_msg_outcall_feature_train_di f
    where dt between '2026-08-22' and '2026-08-27'
      and pid is not null and new_exp = '迭代流量'
      and (group_name_eve in ('留存6张3元', '8元留存5张3', '留存6张5元', '10元留存5张3')
           or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
),
model_input as (
    -- 将候选代表行与同日、同用户、同组特征连接；特征键必须唯一。
    select f.*, k.finish_cnt_7d, k.cost_7d,
        -- 周期级曝光标记：成功候选为1，未成功候选和对照组为0。
        case when k.success_anchor_dt is not null then 1
             else k.send_success_cnt end as send_success_cnt_t,
        k.outcall_success_cnt
    from feature_source f
    inner join selected_keys k
        on cast(f.pid as string) = k.pid and to_date(f.dt) = k.dt
       and f.flowtree_group_name = k.flowtree_group_name
       and f.group_name_eve = k.group_name_eve
    where f.source_feature_row_cnt = 1
)
select --/*+ REPARTITION(100) */
    fin_order_wxmini_28d
    ,group_name_eve
    ,t.pid, t.oneid
    ,send_cnt, send_success_cnt, send_cost, remind_cost, coupon_cost
    ,dau_cnt
    -- 设置label
    ,finish_cnt_7d as label
    ,if(nvl(send_success_cnt_t, 0) > 0 or nvl(outcall_success_cnt, 0) > 0, 1, 0) as exp_label
    ,cost_7d
    ,case
        when group_name_eve = '10元留存5张3' then 4
        when group_name_eve = '留存6张5元' then 3
        when group_name_eve = '8元留存5张3' then 2
        when group_name_eve = '留存6张3元' then 1
        when group_name_eve = 'control_group' then 0
        else null
    end as fake_treatment_id
    ,'valid' as flag
    ,t.dt as label_dt
    
    -- sirius画像特征
    ,sirius_city_id
    ,dm_user_base_oneid_info_didi_silent_days
    ,dm_user_base_oneid_info_didi_active_days
    ,cast(dm_user_base_oneid_info_last_use_htm as int) as dm_user_base_oneid_info_last_use_htm
    ,cast(dm_user_base_oneid_info_last_use_btls as int) as dm_user_base_oneid_info_last_use_btls
    ,dm_user_base_oneid_info_car_days
    ,case when dm_user_base_oneid_info_phone_os_type = 'ios' then 2
        when dm_user_base_oneid_info_phone_os_type = 'android' then 1
        else 0
    end as dm_user_base_oneid_info_phone_os_type
    ,cast(dm_user_base_oneid_info_phone_model as int) as dm_user_base_oneid_info_phone_model
    ,dm_user_base_oneid_info_oneid_create_days
    ,dm_user_base_oneid_info_oneid_openid_num
    ,dm_user_base_oneid_info_oneid_record_num
    ,cast(dm_user_base_oneid_info_oneid_has_rnid as int) as dm_user_base_oneid_info_oneid_has_rnid
    ,cast(dm_user_base_oneid_info_oneid_status as int) as dm_user_base_oneid_info_oneid_status
    ,cast(dm_user_vzt_pub_rfmpeat_flag as int) as dm_user_vzt_pub_rfmpeat_flag
    ,dm_user_vzt_pub_rfmpeat_r_score
    ,dm_user_vzt_pub_rfmpeat_f_score
    ,dm_user_vzt_pub_rfmpeat_m_score
    ,dm_user_vzt_pub_rfmpeat_p_score
    ,dm_user_vzt_pub_rfmpeat_e_score
    ,dm_user_vzt_pub_rfmpeat_a_score
    ,dm_user_vzt_pub_rfmpeat_t_score
    ,dm_user_vzt_pub_rfmpeat_synth_score
    ,dm_user_vzt_pub_rfmpeat_r_rank
    ,dm_user_vzt_pub_rfmpeat_f_rank
    ,dm_user_vzt_pub_rfmpeat_m_rank
    ,dm_user_vzt_pub_rfmpeat_p_rank
    ,dm_user_vzt_pub_rfmpeat_e_rank
    ,dm_user_vzt_pub_rfmpeat_a_rank
    ,dm_user_vzt_pub_rfmpeat_t_rank
    ,dm_user_vzt_pub_rfmpeat_ut_r
    ,dm_user_vzt_pub_rfmpeat_ut_f
    ,dm_user_vzt_pub_rfmpeat_ut_p
    ,dm_user_vzt_pub_rfmpeat_ut_a
    ,dm_user_vzt_pub_rfmpeat_ut_t
    ,dm_user_vzt_pub_rfmpeat_ut_synth
    ,dm_user_vzt_pub_rfmpeat_kflw_r
    ,dm_user_vzt_pub_rfmpeat_kflw_f
    ,dm_user_vzt_pub_rfmpeat_kflw_p
    ,dm_user_vzt_pub_rfmpeat_kflw_a
    ,dm_user_vzt_pub_rfmpeat_kflw_t
    ,dm_user_vzt_pub_rfmpeat_kflw_synth
    ,dm_user_vzt_pub_rfmpeat_taxi_r
    ,dm_user_vzt_pub_rfmpeat_taxi_f
    ,dm_user_vzt_pub_rfmpeat_taxi_m
    ,dm_user_vzt_pub_rfmpeat_taxi_e
    ,dm_user_vzt_pub_rfmpeat_taxi_a
    ,dm_user_vzt_pub_rfmpeat_taxi_t
    ,dm_user_vzt_pub_rfmpeat_taxi_synth
    ,dm_user_vzt_pub_rfmpeat_netcar_r
    ,dm_user_vzt_pub_rfmpeat_netcar_f
    ,dm_user_vzt_pub_rfmpeat_netcar_m
    ,dm_user_vzt_pub_rfmpeat_netcar_p
    ,dm_user_vzt_pub_rfmpeat_netcar_e
    ,dm_user_vzt_pub_rfmpeat_netcar_a
    ,dm_user_vzt_pub_rfmpeat_netcar_t
    ,dm_user_vzt_pub_rfmpeat_netcar_synth
    ,dm_user_vzt_pub_rfmpeat_btls_r
    ,dm_user_vzt_pub_rfmpeat_btls_f
    ,dm_user_vzt_pub_rfmpeat_btls_m
    ,dm_user_vzt_pub_rfmpeat_btls_p
    ,dm_user_vzt_pub_rfmpeat_btls_e
    ,dm_user_vzt_pub_rfmpeat_btls_a
    ,dm_user_vzt_pub_rfmpeat_btls_t
    ,dm_user_vzt_pub_rfmpeat_btls_synth
    ,dm_user_vzt_pub_rfmpeat_htm_r
    ,dm_user_vzt_pub_rfmpeat_htm_f
    ,dm_user_vzt_pub_rfmpeat_htm_p
    ,dm_user_vzt_pub_rfmpeat_htm_e
    ,dm_user_vzt_pub_rfmpeat_htm_a
    ,dm_user_vzt_pub_rfmpeat_htm_t
    ,dm_user_vzt_pub_rfmpeat_htm_synth
    ,dm_user_vzt_pub_rfmpeat_dj_r
    ,dm_user_vzt_pub_rfmpeat_dj_f
    ,dm_user_vzt_pub_rfmpeat_dj_m
    ,dm_user_vzt_pub_rfmpeat_dj_p
    ,dm_user_vzt_pub_rfmpeat_dj_e
    ,dm_user_vzt_pub_rfmpeat_dj_a
    ,dm_user_vzt_pub_rfmpeat_dj_t
    ,dm_user_vzt_pub_rfmpeat_dj_synth
    ,dm_user_vzt_pub_rfmpeat_other_r
    ,dm_user_vzt_pub_rfmpeat_other_p
    ,dm_user_vzt_pub_rfmpeat_other_a
    ,dm_user_vzt_pub_rfmpeat_other_t
    ,cast(dm_user_deal_pub_rfm_flag as int) as dm_user_deal_pub_rfm_flag
    ,dm_user_deal_pub_rfm_r_score
    ,dm_user_deal_pub_rfm_m_score
    ,dm_user_deal_pub_rfm_r_rank
    ,dm_user_deal_pub_rfm_f_rank
    ,case when dm_user_deal_pub_rfm_rfm_importance = '重要' then 2
        when dm_user_deal_pub_rfm_rfm_importance = '一般' then 1
        else 0 
    end as dm_user_deal_pub_rfm_rfm_importance
    ,case when dm_user_deal_pub_rfm_rfm_value = '高价值' then 5
        when dm_user_deal_pub_rfm_rfm_value = '发展' then 4
        when dm_user_deal_pub_rfm_rfm_value = '保持' then 3
        when dm_user_deal_pub_rfm_rfm_value = '挽留' then 2
        else 1
    end as dm_user_deal_pub_rfm_rfm_value
    ,dm_user_deal_pub_rfm_first_pref_name
    ,dm_user_deal_pub_rfm_first_pref_score
    ,dm_user_deal_pub_rfm_second_pref_score
    ,dm_user_deal_pub_rfm_netcar_f
    ,dm_user_deal_pub_rfm_netcar_m
    ,dm_user_deal_pub_rfm_netcar_rfm_score
    ,dm_user_deal_pub_rfm_netcar_score_rank
    ,dm_user_deal_pub_rfm_taxi_f
    ,dm_user_deal_pub_rfm_taxi_m
    ,dm_user_deal_pub_rfm_taxi_rfm_score
    ,dm_user_deal_pub_rfm_taxi_score_rank
    ,dm_user_deal_pub_rfm_btls_r
    ,dm_user_deal_pub_rfm_btls_f
    ,dm_user_deal_pub_rfm_btls_m
    ,dm_user_deal_pub_rfm_btls_rfm_score
    ,case when dm_user_deal_pub_rfm_btls_rfm_class = '重要高价值' then 9
        when dm_user_deal_pub_rfm_btls_rfm_class = '重要发展' then 8
        when dm_user_deal_pub_rfm_btls_rfm_class = '重要保持' then 7
        when dm_user_deal_pub_rfm_btls_rfm_class = '重要挽留' then 6
        when dm_user_deal_pub_rfm_btls_rfm_class = '一般高价值' then 5
        when dm_user_deal_pub_rfm_btls_rfm_class = '一般发展' then 4
        when dm_user_deal_pub_rfm_btls_rfm_class = '一般保持' then 3
        when dm_user_deal_pub_rfm_btls_rfm_class = '一般挽留' then 2
        else 1
    end as dm_user_deal_pub_rfm_btls_rfm_class
    ,dm_user_deal_pub_rfm_btls_score_rank
    ,dm_user_deal_pub_rfm_htm_f
    ,dm_user_deal_pub_rfm_htm_m
    ,dm_user_deal_pub_rfm_htm_rfm_score
    ,dm_user_deal_pub_rfm_bus_r
    ,dm_user_deal_pub_rfm_bus_f
    ,dm_user_deal_pub_rfm_bus_rfm_score
    ,case when dm_user_deal_pub_rfm_bus_rfm_class = '一般高价值' then 5
        when dm_user_deal_pub_rfm_bus_rfm_class = '一般发展' then 4
        when dm_user_deal_pub_rfm_bus_rfm_class = '一般保持' then 3
        when dm_user_deal_pub_rfm_bus_rfm_class = '一般挽留' then 2
        else 1
    end as dm_user_deal_pub_rfm_bus_rfm_class
    ,dm_user_deal_pub_rfm_bus_score_rank
    ,cast(dm_user_mkt_netcar_ordacc_flag as int) as dm_user_mkt_netcar_ordacc_flag
    ,dm_user_mkt_netcar_ordacc_jie_call_decay
    ,dm_user_mkt_netcar_ordacc_jie_finish_order_ratio
    ,dm_user_mkt_netcar_ordacc_jie_last_second_call_day
    ,dm_user_mkt_netcar_ordacc_jie_pay_finish_time_diff_max
    ,dm_user_mkt_netcar_ordacc_jie_pay_finish_time_diff_mean
    ,dm_user_mkt_netcar_ordacc_jie_spend_per_kilo_min
    ,dm_user_mkt_netcar_ordacc_jie_actual_amt_max
    ,dm_user_mkt_netcar_ordacc_jie_actual_amt_mean
    ,dm_user_mkt_netcar_ordacc_jie_fee_diff_max
    ,dm_user_mkt_netcar_ordacc_jie_fee_diff_mean
    ,dm_user_mkt_netcar_ordacc_jie_subsidy_c_order_ratio
    ,dm_user_mkt_netcar_ordacc_jie_subsidy_c_max
    ,dm_user_mkt_netcar_ordacc_jie_subsidy_c_mean
    ,dm_user_mkt_netcar_ordacc_jie_gmv_max
    ,dm_user_mkt_netcar_ordacc_jie_gmv_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_order_charge_dis_max
    ,dm_user_mkt_netcar_ordacc_jie_real_time_charge_dur_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_driver_wait_get_on_time_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_order_charge_dis_sub_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_start_distance_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_end_distance_mean
    ,dm_user_mkt_netcar_ordacc_jie_real_time_answer_dur_mean
    ,dm_user_mkt_netcar_ordacc_jie_interval_week_mean
    ,dm_user_mkt_netcar_ordacc_gcsv2_recored_days
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_with_subsidy_cnt
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_std
    ,dm_user_mkt_netcar_ordacc_gcsv2_cost_mean
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_with_subsidy_mean
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_with_subsidy_std
    ,dm_user_mkt_netcar_ordacc_gcsv2_subsidy_rate_mean
    ,dm_user_mkt_netcar_ordacc_gcsv2_subsidy_rate_std
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_with_no_subsidy_mean
    ,dm_user_mkt_netcar_ordacc_gcsv2_gmv_with_no_subsidy_std
    ,dm_user_mkt_netcar_ordacc_gcsv2_subsidy_rate
    ,dm_user_mkt_netcar_ordacc_gcsv2_wilson_left
    ,dm_user_mkt_netcar_ordacc_gcsv2_wilson_right
    ,cast(dm_user_mkt_netcar_ordacc_gcsv2_gcs_rank_level as int) as dm_user_mkt_netcar_ordacc_gcsv2_gcs_rank_level
    ,cast(dm_user_behv_netcar_htpp_flag as int) as dm_user_behv_netcar_htpp_flag
    ,dm_user_behv_netcar_htpp_pas2dri_prob
    ,dm_user_behv_netcar_htpp_car_ewma
    ,dm_user_behv_netcar_htpp_car_avg
    ,dm_user_behv_netcar_htpp_car_d4
    ,dm_user_behv_netcar_htpp_car_d3
    ,dm_user_behv_netcar_htpp_car_d2
    ,dm_user_behv_netcar_htpp_car_d1
    ,cast(dm_user_behv_netcar_htpp_car_is_churn as int) as dm_user_behv_netcar_htpp_car_is_churn
    ,dm_user_behv_netcar_htpp_taxi_score
    ,dm_user_behv_netcar_htpp_kflw_score
    ,dm_user_behv_netcar_htpp_fast_score
    ,dm_user_behv_netcar_htpp_aplus_score
    ,dm_user_behv_netcar_htpp_tehui_score
    ,dm_user_behv_netcar_htpp_pc_score
    ,dm_user_behv_netcar_htpp_uber_score
    ,dm_user_behv_netcar_htpp_gulf_score
    ,dm_user_behv_netcar_htpp_luxr_score
    ,dm_user_behv_netcar_htpp_yc_score
    ,dm_user_behv_netcar_htpp_first_pd_score
    ,dm_user_behv_netcar_htpp_second_pd_score
    ,dm_user_behv_netcar_htpp_low_price
    ,dm_user_behv_netcar_htpp_up_price
    ,case when dm_user_behv_netcar_htpp_d7_desc = 'climb' then 3
        when dm_user_behv_netcar_htpp_d7_desc = 'stable' then 2
        when dm_user_behv_netcar_htpp_d7_desc = 'drop' then 1
        else 0 
    end as dm_user_behv_netcar_htpp_d7_desc
    ,dm_user_behv_netcar_htpp_d30_call_rank
    ,dm_user_behv_netcar_htpp_d90_call_rank
    ,dm_user_behv_netcar_htpp_d180_call_rank
    ,dm_user_behv_netcar_htpp_d370_call_rank
    ,dm_user_behv_netcar_htpp_prefer_hour
    ,cast(dm_user_mkt_biz_abst_flag as int) as dm_user_mkt_biz_abst_flag
    ,dm_user_mkt_biz_abst_hy_amt_mean
    ,dm_user_mkt_biz_abst_hy_car_types_divg
    ,cast(dm_user_mkt_biz_abst_htm_hm_used as int) as dm_user_mkt_biz_abst_htm_hm_used
    ,dm_user_mkt_biz_abst_netcar_call_finish_ratio
    ,dm_user_mkt_biz_abst_netcar_call_time_divg
    ,dm_user_mkt_biz_abst_netcar_start_grid_divg
    ,dm_user_mkt_biz_abst_netcar_end_grid_divg
    ,dm_user_mkt_biz_abst_netcar_dist_types_divg
    ,dm_user_mkt_biz_abst_netcar_pd_ctgs_divg
    ,cast(dm_user_mkt_biz_abst_netcar_agency_used as int) as dm_user_mkt_biz_abst_netcar_agency_used
    ,cast(dm_user_mkt_biz_abst_netcar_appt_used as int) as dm_user_mkt_biz_abst_netcar_appt_used
        
    -- 补贴环境
    ,user_pref_weekend_prefer_score
    ,user_base_home_company_distance
    ,case when user_pref_prefer_end = 'native' then 3
        when user_pref_prefer_end = 'mini' then 2
        when user_pref_prefer_end = 'uncertain' then 1
        else 0
    end as user_pref_prefer_end
    ,user_behv_mixin_native_pct
    ,user_behv_mixin_mini_pct
    ,user_behv_native_pct
    ,user_behv_mini_pct
    ,user_behv_native_pct_91d
    ,user_behv_mini_pct_91d
    ,user_behv_g_cnt
    ,user_behv_g_native_cnt
    ,user_behv_g_mini_cnt
    ,user_behv_cnt_91d
    ,user_behv_native_cnt_91d
    ,user_behv_mini_cnt_91d
    ,user_behv_coupon_spend_cn
    ,user_behv_coupon_spend_ratio
    ,user_behv_coupon_coupon_cn
    ,user_behv_coupon_max_amount
    ,user_behv_coupon_amount_span
    ,user_behv_coupon_min_discount
    ,user_behv_coupon_discount_span
    ,user_behv_avg_bubble_dist
    ,user_behv_min_bubble_dist

    -- 双端特征
    ,app_dau_label_1d
    ,app_bubble_num_1d
    ,app_call_order_num_1d
    ,app_finish_order_num_1d
    ,app_bubble_dis_max_1d
    ,app_bubble_dis_min_1d
    ,app_bubble_dis_avg_1d
    ,app_order_dis_max_1d
    ,app_order_dis_min_1d
    ,app_order_dis_avg_1d
    ,app_coupon_count_1d
    ,app_coupon_amount_1d
    ,app_coupon_cost_1d
    ,app_bubble_rate_1d
    ,app_call_order_rate_1d
    ,app_finish_order_rate_1d
    ,app_coupon_order_rate_1d
    ,app_coupon_cost_order_rate_1d
    ,xcx_dau_label_1d
    ,xcx_bubble_num_1d
    ,xcx_call_order_num_1d
    ,xcx_finish_order_num_1d
    ,xcx_bubble_dis_max_1d
    ,xcx_bubble_dis_min_1d
    ,xcx_bubble_dis_avg_1d
    ,xcx_order_dis_max_1d
    ,xcx_order_dis_min_1d
    ,xcx_order_dis_avg_1d
    ,xcx_coupon_count_1d
    ,xcx_coupon_amount_1d
    ,xcx_coupon_cost_1d
    ,xcx_bubble_rate_1d
    ,xcx_call_order_rate_1d
    ,xcx_finish_order_rate_1d
    ,xcx_coupon_order_rate_1d
    ,xcx_coupon_cost_order_rate_1d
    ,app_dau_label_7d
    ,app_bubble_num_7d
    ,app_call_order_num_7d
    ,app_finish_order_num_7d
    ,app_bubble_dis_max_7d
    ,app_bubble_dis_min_7d
    ,app_bubble_dis_avg_7d
    ,app_order_dis_max_7d
    ,app_order_dis_min_7d
    ,app_order_dis_avg_7d
    ,app_coupon_count_7d
    ,app_coupon_amount_7d
    ,app_coupon_cost_7d
    ,app_bubble_rate_7d
    ,app_call_order_rate_7d
    ,app_finish_order_rate_7d
    ,app_coupon_order_rate_7d
    ,app_coupon_cost_order_rate_7d
    ,xcx_dau_label_7d
    ,xcx_bubble_num_7d
    ,xcx_call_order_num_7d
    ,xcx_finish_order_num_7d
    ,xcx_bubble_dis_max_7d
    ,xcx_bubble_dis_min_7d
    ,xcx_bubble_dis_avg_7d
    ,xcx_order_dis_max_7d
    ,xcx_order_dis_min_7d
    ,xcx_order_dis_avg_7d
    ,xcx_coupon_count_7d
    ,xcx_coupon_amount_7d
    ,xcx_coupon_cost_7d
    ,xcx_bubble_rate_7d
    ,xcx_call_order_rate_7d
    ,xcx_finish_order_rate_7d
    ,xcx_coupon_order_rate_7d
    ,xcx_coupon_cost_order_rate_7d
    ,app_dau_label_14d
    ,app_bubble_num_14d
    ,app_call_order_num_14d
    ,app_finish_order_num_14d
    ,app_bubble_dis_max_14d
    ,app_bubble_dis_min_14d
    ,app_bubble_dis_avg_14d
    ,app_order_dis_max_14d
    ,app_order_dis_min_14d
    ,app_order_dis_avg_14d
    ,app_coupon_count_14d
    ,app_coupon_amount_14d
    ,app_coupon_cost_14d
    ,app_bubble_rate_14d
    ,app_call_order_rate_14d
    ,app_finish_order_rate_14d
    ,app_coupon_order_rate_14d
    ,app_coupon_cost_order_rate_14d
    ,xcx_dau_label_14d
    ,xcx_bubble_num_14d
    ,xcx_call_order_num_14d
    ,xcx_finish_order_num_14d
    ,xcx_bubble_dis_max_14d
    ,xcx_bubble_dis_min_14d
    ,xcx_bubble_dis_avg_14d
    ,xcx_order_dis_max_14d
    ,xcx_order_dis_min_14d
    ,xcx_order_dis_avg_14d
    ,xcx_coupon_count_14d
    ,xcx_coupon_amount_14d
    ,xcx_coupon_cost_14d
    ,xcx_bubble_rate_14d
    ,xcx_call_order_rate_14d
    ,xcx_finish_order_rate_14d
    ,xcx_coupon_order_rate_14d
    ,xcx_coupon_cost_order_rate_14d
    ,app_dau_label_28d
    ,app_bubble_num_28d
    ,app_call_order_num_28d
    ,app_finish_order_num_28d
    ,app_bubble_dis_max_28d
    ,app_bubble_dis_min_28d
    ,app_bubble_dis_avg_28d
    ,app_order_dis_max_28d
    ,app_order_dis_min_28d
    ,app_order_dis_avg_28d
    ,app_coupon_count_28d
    ,app_coupon_amount_28d
    ,app_coupon_cost_28d
    ,app_bubble_rate_28d
    ,app_call_order_rate_28d
    ,app_finish_order_rate_28d
    ,app_coupon_order_rate_28d
    ,app_coupon_cost_order_rate_28d
    ,xcx_dau_label_28d
    ,xcx_bubble_num_28d
    ,xcx_call_order_num_28d
    ,xcx_finish_order_num_28d
    ,xcx_bubble_dis_max_28d
    ,xcx_bubble_dis_min_28d
    ,xcx_bubble_dis_avg_28d
    ,xcx_order_dis_max_28d
    ,xcx_order_dis_min_28d
    ,xcx_order_dis_avg_28d
    ,xcx_coupon_count_28d
    ,xcx_coupon_amount_28d
    ,xcx_coupon_cost_28d
    ,xcx_bubble_rate_28d
    ,xcx_call_order_rate_28d
    ,xcx_finish_order_rate_28d
    ,xcx_coupon_order_rate_28d
    ,xcx_coupon_cost_order_rate_28d
    ,app_dau_label_60d
    ,app_bubble_num_60d
    ,app_call_order_num_60d
    ,app_finish_order_num_60d
    ,app_bubble_dis_max_60d
    ,app_bubble_dis_min_60d
    ,app_bubble_dis_avg_60d
    ,app_order_dis_max_60d
    ,app_order_dis_min_60d
    ,app_order_dis_avg_60d
    ,app_coupon_count_60d
    ,app_coupon_amount_60d
    ,app_coupon_cost_60d
    ,app_bubble_rate_60d
    ,app_call_order_rate_60d
    ,app_finish_order_rate_60d
    ,app_coupon_order_rate_60d
    ,app_coupon_cost_order_rate_60d
    ,xcx_dau_label_60d
    ,xcx_bubble_num_60d
    ,xcx_call_order_num_60d
    ,xcx_finish_order_num_60d
    ,xcx_bubble_dis_max_60d
    ,xcx_bubble_dis_min_60d
    ,xcx_bubble_dis_avg_60d
    ,xcx_order_dis_max_60d
    ,xcx_order_dis_min_60d
    ,xcx_order_dis_avg_60d
    ,xcx_coupon_count_60d
    ,xcx_coupon_amount_60d
    ,xcx_coupon_cost_60d
    ,xcx_bubble_rate_60d
    ,xcx_call_order_rate_60d
    ,xcx_finish_order_rate_60d
    ,xcx_coupon_order_rate_60d
    ,xcx_coupon_cost_order_rate_60d
    ,app_dau_label_120d
    ,app_bubble_num_120d
    ,app_call_order_num_120d
    ,app_finish_order_num_120d
    ,app_bubble_dis_max_120d
    ,app_bubble_dis_min_120d
    ,app_bubble_dis_avg_120d
    ,app_order_dis_max_120d
    ,app_order_dis_min_120d
    ,app_order_dis_avg_120d
    ,app_coupon_count_120d
    ,app_coupon_amount_120d
    ,app_coupon_cost_120d
    ,app_bubble_rate_120d
    ,app_call_order_rate_120d
    ,app_finish_order_rate_120d
    ,app_coupon_order_rate_120d
    ,app_coupon_cost_order_rate_120d
    ,xcx_dau_label_120d
    ,xcx_bubble_num_120d
    ,xcx_call_order_num_120d
    ,xcx_finish_order_num_120d
    ,xcx_bubble_dis_max_120d
    ,xcx_bubble_dis_min_120d
    ,xcx_bubble_dis_avg_120d
    ,xcx_order_dis_max_120d
    ,xcx_order_dis_min_120d
    ,xcx_order_dis_avg_120d
    ,xcx_coupon_count_120d
    ,xcx_coupon_amount_120d
    ,xcx_coupon_cost_120d
    ,xcx_bubble_rate_120d
    ,xcx_call_order_rate_120d
    ,xcx_finish_order_rate_120d
    ,xcx_coupon_order_rate_120d
    ,xcx_coupon_cost_order_rate_120d
    ,app_dau_label_180d
    ,app_bubble_num_180d
    ,app_call_order_num_180d
    ,app_finish_order_num_180d
    ,app_bubble_dis_max_180d
    ,app_bubble_dis_min_180d
    ,app_bubble_dis_avg_180d
    ,app_order_dis_max_180d
    ,app_order_dis_min_180d
    ,app_order_dis_avg_180d
    ,app_coupon_count_180d
    ,app_coupon_amount_180d
    ,app_coupon_cost_180d
    ,app_bubble_rate_180d
    ,app_call_order_rate_180d
    ,app_finish_order_rate_180d
    ,app_coupon_order_rate_180d
    ,app_coupon_cost_order_rate_180d
    ,xcx_dau_label_180d
    ,xcx_bubble_num_180d
    ,xcx_call_order_num_180d
    ,xcx_finish_order_num_180d
    ,xcx_bubble_dis_max_180d
    ,xcx_bubble_dis_min_180d
    ,xcx_bubble_dis_avg_180d
    ,xcx_order_dis_max_180d
    ,xcx_order_dis_min_180d
    ,xcx_order_dis_avg_180d
    ,xcx_coupon_count_180d
    ,xcx_coupon_amount_180d
    ,xcx_coupon_cost_180d
    ,xcx_bubble_rate_180d
    ,xcx_call_order_rate_180d
    ,xcx_finish_order_rate_180d
    ,xcx_coupon_order_rate_180d
    ,xcx_coupon_cost_order_rate_180d

    -- 有端
    ,clue_score
    ,clue_score_rank
    
    -- 券敏感
    ,coupon_sensitive
    
from model_input t
