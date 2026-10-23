-- v6_dfcl_touch train: event-based sample construction.
-- One successful touch row is one fixed-7-day incentive sample.
-- A later positive attributed to another entry is retained as another sample.
-- Failed retries before a later event are not separate negatives.
-- A terminal segment with no success and no positive keeps only its last mature entry.
-- Continuous in-group episodes are approximated by gaps shorter than 28 days.
-- Propensity is looked up in the model:
-- 0=0.4916, 1=0.1778, 2=0.1185, 3=0.0935, 4=0.1186.
-- Original v6 SQL remains unchanged.

with
label_raw as (
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
    from label_raw
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
            when treatment_id <> 0
             and (send_success_cnt > 0 or outcall_success_cnt > 0) then 1
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
ordered as (
    select
        d.*,
        lag(dt) over (
            partition by pid
            order by dt
        ) as previous_dt
    from daily_flags d
),
episode_flags as (
    select
        o.*,
        case
            when previous_dt is null then 1
            when datediff(dt, previous_dt) >= 28 then 1
            else 0
        end as new_episode_flag
    from ordered o
),
episode_rows as (
    select
        e.*,
        sum(new_episode_flag) over (
            partition by pid
            order by dt
            rows between unbounded preceding and current row
        ) as episode_id
    from episode_flags e
),
row_context as (
    select
        e.*,
        max(
            case when touch_success = 1 or finish_cnt_7d > 0 then dt end
        ) over (
            partition by pid, episode_id
            order by dt
            rows between unbounded preceding and current row
        ) as previous_or_same_event_dt,
        min(
            case
                when touch_success = 1 or finish_cnt_7d > 0 then dt
            end
        ) over (
            partition by pid, episode_id
            order by dt
            rows between 1 following and unbounded following
        ) as next_event_dt,
        max(bad_label_flag) over (
            partition by pid, episode_id
        ) as episode_bad_label_flag,
        max(
            case
                when source_variant_cnt > 1 or pid_day_key_cnt > 1 then 1
                else 0
            end
        ) over (
            partition by pid, episode_id
        ) as episode_source_conflict_flag
    from episode_rows e
),
classified as (
    select
        r.*,
        case
            when previous_or_same_event_dt is not null
             and datediff(dt, previous_or_same_event_dt) between 0 and 6
                then 1
            else 0
        end as inside_resolved_event_window_flag,
        case
            when touch_success = 1 then 'success_touch'
            when finish_cnt_7d > 0 then 'attributed_positive_no_success'
            else 'zero_retry'
        end as row_kind
    from row_context r
),
terminal_ranked as (
    select
        c.*,
        case
            when touch_success = 0
             and coalesce(finish_cnt_7d, 0) = 0
             and inside_resolved_event_window_flag = 0
             and next_event_dt is null then 1
            else 0
        end as terminal_zero_eligible_flag,
        row_number() over (
            partition by pid, episode_id
            order by
                case
                    when touch_success = 0
                     and coalesce(finish_cnt_7d, 0) = 0
                     and inside_resolved_event_window_flag = 0
                     and next_event_dt is null then 0
                    else 1
                end,
                dt desc,
                flowtree_group_name,
                group_name_eve
        ) as terminal_rank
    from classified c
),
selected_keys as (
    select
        r.pid,
        r.flowtree_group_name,
        r.group_name_eve,
        r.dt
    from terminal_ranked r
    inner join user_group_counts g
        on r.pid = g.pid
    where (
            r.touch_success = 1
            or r.finish_cnt_7d > 0
            or (
                r.terminal_zero_eligible_flag = 1
                and r.terminal_rank = 1
            )
          )
      and r.episode_bad_label_flag = 0
      and r.episode_source_conflict_flag = 0
      and g.user_treatment_cnt = 1
      and r.dt between to_date('2026-07-16') and to_date('2026-08-21')
),
original_v6_rows as (
select --/*+ REPARTITION(100) */
    fin_order_wxmini_28d
    ,group_name_eve
    ,t.pid, t.oneid
    ,send_cnt, send_success_cnt, send_cost, remind_cost, coupon_cost
    ,dau_cnt
    -- 设置label
    ,finish_cnt_7d as label
    ,if(nvl(send_success_cnt_t, 0) > 0 or nvl(outcall_success_cnt, 0) > 0, 1, 0) as exp_label
    --,cast((nvl(remind_cost, 0.0) +  nvl(coupon_cost, 0.0)) as double) as cost
    ,case when group_name_eve = 'control_group' then cast(0.0 as double) else cost_7d end as cost_7d
    ,case
        when group_name_eve = '10元留存5张3' then 4
        when group_name_eve = '留存6张5元' then 3
        when group_name_eve = '8元留存5张3' then 2
        when group_name_eve = '留存6张3元' then 1
        when group_name_eve = 'control_group' then 0
        else null
    end as fake_treatment_id
    ,'train' as flag
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
    
from
(
    select t1.*, t.finish_cnt_7d, t.send_success_cnt as send_success_cnt_t, t.outcall_success_cnt, t.cost_7d
    from
    (
        select 
            *
        from dm.ut_alg_flowback_msg_outcall_feature_train_di
        where 
            dt between '2026-07-16' and '2026-08-21'
            and dt >= '2026-03-02'
            and dt not in ('2026-03-26','2026-03-27','2026-04-08','2026-08-02','2026-08-08','2026-08-09')
            and new_exp = '迭代流量'
            and pid is not null
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    ) t1
    join 
    (
        select distinct
            flowtree_group_name, group_name_eve, dt, pid, finish_cnt_7d, send_success_cnt, outcall_success_cnt
            ,cast((nvl(send_cost, 0.0) + nvl(send_remind_cost, 0.0)+ nvl(outcall_cost, 0.0)+ nvl(remind_cost, 0.0)+ nvl(msg_coupon_cost_7d, 0.0)+ nvl(outcall_coupon_cost_7d, 0.0)+ nvl(outcall_coupon2_cost_7d, 0.0)) as double) as cost_7d
        from dm.ut_alg_flowback_msg_outcall_label_v2_di
        where 
            dt between '2026-07-16' and '2026-08-21'
            and dt >= '2026-03-02'
            and dt not in ('2026-03-26','2026-03-27','2026-04-08','2026-08-02','2026-08-08','2026-08-09')
            and new_exp = '迭代流量'
            and pid is not null
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    )t
    on t1.flowtree_group_name = t.flowtree_group_name
    and t1.group_name_eve = t.group_name_eve
    and t1.dt = t.dt
    and t1.pid = t.pid
) t
)
select o.*
from original_v6_rows o
inner join selected_keys k
    on cast(o.pid as string) = k.pid
   and o.group_name_eve = k.group_name_eve
   and to_date(o.label_dt) = k.dt


