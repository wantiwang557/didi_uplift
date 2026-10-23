SELECT
    fin_order_wxmini_28d
    ,group_name_eve
    ,t.pid, t.oneid
    ,send_cnt, send_success_cnt, send_cost, remind_cost, coupon_cost
    ,dau_cnt
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
    ,'{{split}}' as flag
    ,t.dt as label_dt
{{features}}

from
(
    select t1.*, t.finish_cnt_7d, t.send_success_cnt as send_success_cnt_t, t.outcall_success_cnt, t.cost_7d, sum(t.finish_cnt_7d) over(partition by t1.pid) as finish_all_cnt
    from
    (
        select 
            *
        from dm.ut_alg_flowback_msg_outcall_feature_train_di
        where 
            {{date_predicate}}
            and dt >= '2026-03-02'
            and new_exp = '迭代流量'
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    ) t1
    join 
    (
        select 
            flowtree_group_name, group_name_eve, dt, pid, finish_cnt_7d, send_success_cnt, outcall_success_cnt
            ,cast((nvl(send_cost, 0.0) + nvl(send_remind_cost, 0.0)+ nvl(outcall_cost, 0.0)+ nvl(remind_cost, 0.0)+ nvl(msg_coupon_cost_7d, 0.0)+ nvl(outcall_coupon_cost_7d, 0.0)+ nvl(outcall_coupon2_cost_7d, 0.0)) as double) as cost_7d
        from dm.ut_alg_flowback_msg_outcall_label_v2_di
        where 
            {{date_predicate}}
            and dt >= '2026-03-02'
            and new_exp = '迭代流量'
            and ((group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
        
    )t
    on t1.flowtree_group_name = t.flowtree_group_name
    and t1.group_name_eve = t.group_name_eve
    and t1.dt = t.dt
    and t1.pid = t.pid
) t
{{joins}}
WHERE
    (finish_all_cnt > 0 and finish_cnt_7d > 0) or (finish_all_cnt = 0)
