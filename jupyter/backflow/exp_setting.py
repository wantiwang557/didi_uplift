
from collections import OrderedDict

colname_mapping = {
    'group_name':'组名',
    'exp_name':'实验名称',
    'group_tag':'组名tag',
    'splited_group_name':'拆分实验组名',
    'group_describe':'组名描述',
    'fin_order_mini_28d_level':'小程序最近28d完单数',
    'ingroup_cnt':'入组人数',
    'outcall_success_uv':'外呼成功数',
    'outcall_uv':'外呼人数',
    'active_uv':'激活人数',
    'finish_ord_uv':'完单uv',
    'finish_ord_pv':'完单pv',
    'delta_finish_ord_uv':'delta完单uv',
    'delta_finish_ord_pv':'delta完单pv',
    'channel_cost':'渠道费',
    'coupon_cost':'券消耗',
    'pay_coupon_cost':'支付券消耗',
    'order_coupon_cost':'完单券消耗',
    'total_cost':'总消耗',
    'finish_ord_uv_cac':'完单uv_cac',
    'finish_ord_pv_cac':'完单pv_cac',
    'pay_coupon_uv_cac':'支付券uv_cac',
    'pay_coupon_pv_cac':'支付券pv_cac',
    'ord_coupon_pv_cac':'完单券pv_cac',
    'ord_coupon_uv_cac':'完单券uv_cac',
    'delta_finish_ord_uv_cac':'delta完单uv_cac',
    'delta_finish_ord_pv_cac':'delta完单pv_cac',
    'outcall_success2finish_ord_uv_ratio':'uv转化率(完单/外呼成功)',
    'outcall_success2finish_ord_pv_ratio':'pv转化率(完单/外呼成功)',
    'coupon_uv':'券核销量',
    'pay_coupon_uv':'支付券核销量',
    'ord_coupon_uv':'完单券核销量',
    'delta_outcall_success2finish_ord_uv_ratio':'delta_uv转化率',
    'delta_outcall_success2finish_ord_pv_ratio':'delta_pv转化率',
    'app_finish_active_tag':'NA网出完单生命周期',
    'avg_finish_ord':'人均完单',
    'ingroup_cvt_uv':'入组完单率uv',
    'ingroup_cvt_pv':'入组完单率pv',
    'outcall_success_ratio':'外呼成功率',
    'finish_ord_uv_wide':'入组完单uv',
    'finish_ord_pv_wide':'入组完单pv',
    'ingroup2outcall_success_ratio':'入组外呼成功率',
    'finish_ord_attr_uv':'归因完单uv',
    'finish_ord_attr_pv':'归因完单pv',
    'finish_ord_attr_uv_cac':'归因完单uv_cac',
    'finish_ord_attr_pv_cac':'归因完单pv_cac',
    'channel_cost_div_total_cost':'渠道占总消耗',
    'pay_coupon_cost_div_total_cost':'支付券占总消耗',
    'order_coupon_cost_div_total_cost':'发单券占总消耗',
    'finish_ord_pv_3d':'3日完单pv',
    'finish_ord_uv_3d':'3日完单uv'
}

colmapping = OrderedDict({
    'exp_name'  :'实验名',
    'group_name':'组名',
    'group_tag' :'实验标记',
    'dt'        :'入组日期',
    'group_describe':'组名描述',
    
    'ingroup_cnt'                  :'入组人数',
    'outcall_uv'                   :'外呼人数',
    'outcall_success_uv'           :'外呼成功人数',
    'outcall_success_ratio'        :'外呼成功率',
    'ingroup2outcall_success_ratio':'入组外呼成功率',

    'coupon_uv'    :'券核销总人数',
    'pay_coupon_uv':'支付券核销人数',
    'ord_coupon_uv':'发单券核销人数',
    'active_uv'    :'激活人数',

    'channel_cost'                      :'渠道费',
    'coupon_cost'                       :'券消耗',
    'pay_coupon_cost'                   :'支付券消耗',
    'order_coupon_cost'                 :'发单券消耗',
    'channel_cost_div_total_cost'       :'渠道费占总消耗',
    'pay_coupon_cost_div_total_cost'    :'支付券占总消耗',
    'order_coupon_cost_div_total_cost'  :'发单券占总消耗',
    'total_cost':'总消耗',

    'finish_ord_uv_7d'     :'7日完单uv',
    'finish_ord_pv_7d'     :'7日完单pv',
    'finish_ord_uv_3d'     :'3日完单uv',
    'finish_ord_pv_3d'     :'3日完单pv',
    'finish_ord_uv_7d_callsuccess'     :'7日完单uv(外呼成功)',
    'finish_ord_pv_7d_callsuccess'     :'7日完单pv(外呼成功)',
    'finish_ord_uv_3d_callsuccess'     :'3日完单uv(外呼成功)',
    'finish_ord_pv_3d_callsuccess'     :'3日完单pv(外呼成功)',
    
    'finish_ord_attr_uv_3d':'3日完单uv(归因)',
    'finish_ord_attr_pv_3d':'3日完单pv(归因)',

    'delta_finish_ord_uv_7d_callsuccess':'7日完单uv(外呼成功)',
    'delta_finish_ord_pv_7d_callsuccess':'7日完单pv(外呼成功)',
    'delta_finish_ord_uv_3d_callsuccess':'3日完单uv(外呼成功)',
    'delta_finish_ord_pv_3d_callsuccess':'3日完单pv(外呼成功)',
    

    'avg_finish_ord_3d'     :'3日人均完单',
    'avg_finish_ord_7d'     :'7日人均完单',
    'avg_finish_ord_3d_callsuccess':'3日人均完单(外呼成功)',
    'avg_finish_ord_7d_callsuccess':'7日人均完单(外呼成功)',
    
    'delta_finish_ord_uv_3d':'3日delta完单uv',
    'delta_finish_ord_pv_3d':'3日delta完单pv',
    'delta_finish_ord_uv_7d':'7日delta完单uv',
    'delta_finish_ord_pv_7d':'7日delta完单pv',

    'delta_finish_ord_uv_3d_callsuccess':'3日delta完单uv(外呼成功)',
    'delta_finish_ord_pv_3d_callsuccess':'3日delta完单pv(外呼成功)',
    'delta_finish_ord_uv_7d_callsuccess':'7日delta完单uv(外呼成功)',
    'delta_finish_ord_pv_7d_callsuccess':'7日delta完单pv(外呼成功)',
    


    'pay_coupon_uv_cac_3d':'3日支付券cac',
    'ord_coupon_uv_cac_3d':'3日发单券cac',

    'finish_ord_uv_cac_7d':'7日完单uv_cac',
    'finish_ord_pv_cac_7d':'7日完单pv_cac',
    'finish_ord_uv_cac_3d':'3日完单pv_cac',
    'finish_ord_pv_cac_3d':'3日完单uv_cac',

    'finish_ord_uv_cac_7d_callsuccess':'7日完单uv_cac(外呼成功)',
    'finish_ord_pv_cac_7d_callsuccess':'7日完单pv_cac(外呼成功)',
    'finish_ord_uv_cac_3d_callsuccess':'3日完单pv_cac(外呼成功)',
    'finish_ord_pv_cac_3d_callsuccess':'3日完单uv_cac(外呼成功)',
    
    
    'finish_ord_attr_uv_cac_3d':'7日完单uv_cac(归因)',
    'finish_ord_attr_pv_cac_3d':'7日完单pv_cac(归因)',
    'finish_ord_attr_uv_cac_7d':'3日完单uv_cac(归因)',
    'finish_ord_attr_pv_cac_7d':'3日完单pv_cac(归因)',
    
    
    'delta_uv_cac_7d':'7日delta完单uv_cac',
    'delta_pv_cac_7d':'7日delta完单pv_cac',
    'delta_uv_cac_3d':'3日delta完单uv_cac',
    'delta_pv_cac_3d':'3日delta完单pv_cac',
    'delta_uv_cac_7d_callsuccess':'7日delta完单uv_cac(外呼成功)',
    'delta_pv_cac_7d_callsuccess':'7日delta完单pv_cac(外呼成功)',
    'delta_uv_cac_3d_callsuccess':'3日delta完单uv_cac(外呼成功)',
    'delta_pv_cac_3d_callsuccess':'3日delta完单pv_cac(外呼成功)',

    'delta_uv_cvr_7d':'7日delta_uv_转化率',
    'delta_pv_cvr_7d':'7日delta_pv_转化率',
    'delta_uv_cvr_3d':'3日delta_uv_转化率',
    'delta_pv_cvr_3d':'3日delta_pv_转化率',
    'delta_uv_cvr_7d_callsuccess':'7日delta_uv_转化率(外呼成功)',
    'delta_pv_cvr_7d_callsuccess':'7日delta_pv_转化率(外呼成功)',
    'delta_uv_cvr_3d_callsuccess':'3日delta_uv_转化率(外呼成功)',
    'delta_pv_cvr_3d_callsuccess':'3日delta_pv_转化率(外呼成功)',

    
})


group_tag_mapping_v1 = {
    'tp0295_main_gys_call_03-base_random_group':'v1大盘通补',
    'tp0295_main_suanfa_call_exp_v2_03-control_group':'v1空白',
    'tp0295_main_suanfa_call_exp_v2_03-group_1':'实验组1:3张3元',
    'tp0295_main_suanfa_call_exp_v2_03-group_2':'实验组2:3张5元',
    'tp0295_main_suanfa_call_exp_v2_03-group_3':'实验组3:10元&2张5元',
    'tp0295_main_suanfa_call_exp_v2_03-group_4':'实验组4:3张8元',
    'tp0295_main_suanfa_call_exp_v2_03-group_5':'实验组5:3张10元',
}

group_tag_mapping_v1pro = {
    'tp0295_main_suanfa_call_exp_v2_03-backflow_v1pro_A':'算法组588',
    'tp0295_main_suanfa_call_exp_v2_03-backflow_v1pro_BCD':'算法组555',
    'tp0295_main_suanfa_call_exp_v2_03-backflow_v1pro_E':'算法组333',
    'tp0295_main_suanfa_call_exp_v2_03-group_1':'算法组miss',
    'tp0295_main_suanfa_call_exp_v2_03-group_2':'算法组通补555',
    'tp0295_main_gys_call_03-group_1':'大盘通补555',
    'tp0295_main_gys_call_03-group_2':'大盘通补588',
    'tp0295_main_gys_call_03-group_3':'大盘通补333',
    
}

group_tag_mapping_v1_pro_expand = {
    'tp0295_main_suanfa_call_exp_v2_03-grouo_3':'小流量空白',
    'tp0295_main_gys_call_03-group_1':'扩量流量',
    'tp0295_main_suanfa_call_exp_v2_03-group_1':'小流量算法组',
    'tp0295_main_suanfa_call_exp_v2_03-group_2':'小流量通补555',

    'tp0295_main_gys_call_03-backflow_v1pro_A_expand':'主流量588',
    'tp0295_main_gys_call_03-backflow_v1pro_nonA_expand':'主流量555',
    'tp0295_main_suanfa_call_exp_v3_03-group_1':'小流量通补588',
    'tp0295_main_suanfa_call_exp_v3_03-group_2':'小流量通补555',
    'tp0295_main_suanfa_call_exp_v3_03-group_3':'小流量通补51010',
    'tp0295_main_suanfa_call_exp_v3_03-control_group':'小流量空白',
    'tp0295_main_call_opening_exp_v3_03-group_1':'外呼在线扩量组'
}

group_tag_mapping_v_rsh_1228 = {
    'tp0295_main_app_coupon_baiying_treat_03-group_1':'主流量通补10',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_AB':'小流量88(含免)',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_C':'小流量通补5',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_D':'小流量通补8',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_E':'小流量通补10',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_F':'小流量通补55',
    'tp0295_main_app_coupon_baiying_treat_03-research_space_G':'小流量通补58',
    'tp0295_main_app_coupon_baiying_treat_03-control_group':'空白',
}

