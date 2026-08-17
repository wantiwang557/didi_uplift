-- EUEN train raw SQL
-- Purpose: produce raw train samples for downstream feature_conf/parser_conf extraction.
-- Keep fake_treatment_id mapping as-is; align EUEN training code later.
-- Default window: 2026-05-11 ~ 2026-06-17

select --/*+ REPARTITION(100) */
    cast(0.0 as double) as fin_order_mini_28d
    ,group_name_eve
    ,pid, oneid
    ,send_cnt, send_success_cnt, send_cost, remind_cost, coupon_cost
    ,dau_cnt
    -- 设置label
    ,finish_cnt_7d as label
    -- 改动：直接使用预计算完整cost_7d，替代手动求和
    ,cast(nvl(t.cost_7d, 0.0) as double) as cost
    ,case
        when group_name_eve = '10元留存5张3' then 4
        when group_name_eve = '留存6张5元' then 3
        when group_name_eve = '8元留存5张3' then 2
        when group_name_eve = '留存6张3元' then 1
        when group_name_eve = 'control_group' then 0
        else null
    end as fake_treatment_id
    ,'train' as flag
    ,dt as label_dt
    
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
    
    -- 个推特征
    ,age_level
    ,gender
    ,has_car
    ,car_type
    ,career
    ,career_type
    ,muying
    ,shanglv
    ,student
    ,resident_city_id
    ,holiday_city_id
    ,phone_brand
    ,zufang
    ,pet
    ,game
    ,consume_level
    ,family_level
from
(
    select t1.*, finish_cnt_7d, t.cost_7d, sum(finish_cnt_7d) over(partition by t1.pid) as finish_all_cnt
    from
    (
        select 
            *
        from dm.ut_alg_flowback_msg_outcall_feature_train_di
        -- 训练集固定区间：5.11 ~ 6.17
        where 
            dt between '2026-05-11' and '2026-06-17'
            and dt >= '2026-03-02'
            and dt not in ('2026-03-26','2026-03-27','2026-04-08')
            and new_exp = '迭代流量'
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    ) t1
    join 
   
    (
        select 
            flowtree_group_name, group_name_eve, dt, pid, finish_cnt_7d
            ,cast((nvl(send_cost, 0.0) + nvl(send_remind_cost, 0.0)+ nvl(outcall_cost, 0.0)+ nvl(remind_cost, 0.0)+ nvl(msg_coupon_cost_7d, 0.0)+ nvl(outcall_coupon_cost_7d, 0.0)+ nvl(outcall_coupon2_cost_7d, 0.0)) as double) as cost_7d
        from dm.ut_alg_flowback_msg_outcall_label_di
        -- 和特征表同步日期
        where 
            dt between '2026-05-11' and '2026-06-17'
            and new_exp = '迭代流量'
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    )t
    on t1.flowtree_group_name = t.flowtree_group_name
    and t1.group_name_eve = t.group_name_eve
    and t1.dt = t.dt
    and t1.pid = t.pid
) t
where 
    (finish_all_cnt > 0 and finish_cnt_7d > 0) or (finish_all_cnt = 0)
