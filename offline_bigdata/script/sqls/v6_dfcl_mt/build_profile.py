"""Render the MT sample profile, preserving the original v6 output projection.

Local SQL generation only: no Hive queries, remote writes or fitted configs.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SPLITS = {
    'train': ('2026-07-16', '2026-08-21'),
    'valid': ('2026-08-22', '2026-08-27'),
    'test': ('2026-08-28', '2026-09-10'),
}
SETTINGS = {
    # 业务确认的重新入组边界；成功发券周期仍固定为7天，不能把28天当成激励周期。
    'failed_gap_days': 28,
    'excluded_dates': ', '.join("'{}'".format(d) for d in (
        '2026-03-26', '2026-03-27', '2026-04-08',
        '2026-08-02', '2026-08-08', '2026-08-09')),
}


def render(text, values):
    for key, value in values.items():
        text = text.replace('{{' + key + '}}', str(value))
    if '{{' in text or '}}' in text:
        raise ValueError('Unresolved SQL template parameter')
    return text


def selection(start, end):
    return render((HERE / 'selection_ctes.sql').read_text(), dict(
        SETTINGS,
        history_start=start,
        history_end=end,
        sample_start=start,
        sample_end=end,
    )).strip()


def feature_cte(start, end):
    return """,
-- 读取当前 split 的原 v6 特征，并统计连接键对应的特征行数，防止连接放大样本。
feature_source as (
    select f.*,
        count(*) over (partition by pid, dt, flowtree_group_name, group_name_eve) as source_feature_row_cnt
    from dm.ut_alg_flowback_msg_outcall_feature_train_di f
    where dt between '{start}' and '{end}'
      and pid is not null and new_exp = '迭代流量'
      and (group_name_eve in ('留存6张3元', '8元留存5张3', '留存6张5元', '10元留存5张3')
           or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
)""".format(start=start, end=end)


def main():
    source_hashes = {}
    for split, (start, end) in SPLITS.items():
        path = HERE.parent / 'v6' / ('feature_v6_' + split + '.sql')
        original = path.read_text()
        source_hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        projection = original[original.index('select --'):original.index('\nfrom\n(')]
        sql = selection(start, end) + feature_cte(start, end) + """,
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
""" + projection + '\nfrom model_input t\n'
        header = """-- v6_dfcl_mt {split}：成功候选固定选择触达成功日；未成功实验组和对照组有正选正、全负选末次。
-- 未成功连续段不按每7个自然日切分，边界定义见 README.md。
-- 输出字段与原 v6 一致，代表行同时提供特征、标签、曝光和成本。
-- 本文件由 build_profile.py 生成，正式抽数前应先运行 audit_samples.sql。
-- 代表行去重不等于 DFCL/IPW 已满足无偏条件，分流概率和选择机制需要另行验证。
""".format(split=split)
        (HERE / ('feature_v6_dfcl_mt_' + split + '.sql')).write_text(header + sql)

    start, end = SPLITS['train'][0], SPLITS['test'][1]
    audit = selection(start, end) + feature_cte(start, end) + """,
feature_keys as (
    select cast(pid as string) as pid, to_date(dt) as dt, flowtree_group_name, group_name_eve,
        max(source_feature_row_cnt) as feature_row_cnt
    from feature_source
    group by pid, dt, flowtree_group_name, group_name_eve
)
select r.pid, r.flowtree_group_name, r.group_name_eve,
    r.treatment_id, r.candidate_kind, r.candidate_start_dt,
    r.first_member_dt, r.last_member_dt, r.member_cnt, r.observation_member_cnt,
    r.pre_touch_retry_cnt, r.positive_member_cnt, r.pre_touch_positive_cnt,
    datediff(r.last_member_dt, r.first_member_dt) + 1 as member_span_days,
    r.dt as selected_dt, r.finish_cnt_7d as selected_label, r.cost_7d as selected_cost_7d,
    r.touch_success as selected_touch_success,
    r.positive_priority_rank, r.selected_priority_reason,
    r.success_anchor_dt, r.observation_start_dt, r.observation_end_dt_assumed,
    r.label_anchor_mismatch_flag,
    case when r.observation_start_dt <= r.previous_max_observation_end_dt then 1 else 0 end as observation_overlap_flag,
    case when r.first_member_dt = to_date('{{history_start}}') then 1 else 0 end as touches_history_start_flag,
    case when r.candidate_kind = 'success_7d' and date_add(r.success_anchor_dt, 6) > to_date('{{history_end}}') then 1
         when r.candidate_kind <> 'success_7d' and date_add(r.last_member_dt, {{failed_gap_days}} - 1) > to_date('{{history_end}}') then 1
         else 0 end as right_boundary_flag,
    r.candidate_bad_label_flag, r.sample_status,
    case when (r.first_member_dt <= to_date('2026-08-21') and r.last_member_dt >= to_date('2026-08-22'))
           or (r.first_member_dt <= to_date('2026-08-27') and r.last_member_dt >= to_date('2026-08-28'))
         then 1 else 0 end as crosses_split_boundary_flag,
    coalesce(f.feature_row_cnt, 0) as selected_feature_row_cnt,
    case when r.dt between to_date('2026-07-16') and to_date('2026-08-21') then 'train'
         when r.dt between to_date('2026-08-22') and to_date('2026-08-27') then 'valid'
         when r.dt between to_date('2026-08-28') and to_date('2026-09-10') then 'test'
         else 'context_only' end as selected_split,
    case when r.dt not between to_date('2026-07-16') and to_date('2026-09-10') then 'representative_outside_output'
         when r.sample_status <> 'eligible' then r.sample_status
         when f.feature_row_cnt is null then 'missing_selected_features'
         when f.feature_row_cnt <> 1 then 'duplicate_selected_features'
         else 'output' end as output_status
from audited_representatives r
left join feature_keys f
    on r.pid = f.pid and r.dt = f.dt
   and r.flowtree_group_name = f.flowtree_group_name and r.group_name_eve = f.group_name_eve
-- 审计覆盖与总时间范围相交的候选，即使其最终代表日期落在输出范围之外。
where r.first_member_dt <= to_date('2026-09-10') and r.last_member_dt >= to_date('2026-07-16')
"""
    (HERE / 'audit_samples.sql').write_text(render(audit, dict(
        SETTINGS,
        history_start=start,
        history_end=end,
    )))

    multi_positive_summary = selection(start, end) + """
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
"""
    (HERE / 'audit_multi_positive_success_summary.sql').write_text(multi_positive_summary)

    multi_positive_details = selection(start, end) + """
, multi_cycles as (
    select pid, candidate_kind, candidate_start_dt, selected_priority_reason, sample_status
    from audited_representatives
    where candidate_kind = 'success_7d'
      and positive_member_cnt > 1
      and success_anchor_dt between to_date('2026-07-16') and to_date('2026-09-10')
)
select r.pid, r.treatment_id, r.group_name_eve, r.success_anchor_dt,
    date_add(r.success_anchor_dt, 6) as success_cycle_end_dt,
    r.dt as entry_dt, r.in_observation_window, r.finish_cnt_7d,
    r.send_success_cnt, r.outcall_success_cnt, r.cost_7d,
    r.positive_priority_rank,
    r.positive_member_cnt, r.pre_touch_retry_cnt, r.pre_touch_positive_cnt,
    r.keep_rank, case when r.keep_rank = 1 then 1 else 0 end as selected_flag,
    m.selected_priority_reason, m.sample_status
from ranked r
inner join multi_cycles m
    on r.pid = m.pid and r.candidate_kind = m.candidate_kind
   and r.candidate_start_dt = m.candidate_start_dt
order by r.pid, r.success_anchor_dt, r.dt
"""
    (HERE / 'audit_multi_positive_success_details.sql').write_text(multi_positive_details)

    transition_audit = selection(start, end) + """
, ordered_candidates as (
    select r.*,
        lead(candidate_kind) over (partition by pid order by candidate_start_dt, dt) as next_candidate_kind,
        lead(candidate_start_dt) over (partition by pid order by candidate_start_dt, dt) as next_candidate_start_dt,
        lead(dt) over (partition by pid order by candidate_start_dt, dt) as next_selected_dt,
        lead(finish_cnt_7d) over (partition by pid order by candidate_start_dt, dt) as next_selected_label,
        lead(sample_status) over (partition by pid order by candidate_start_dt, dt) as next_sample_status
    from audited_representatives r
)
select pid, treatment_id,
    success_anchor_dt, date_add(success_anchor_dt, 6) as success_cycle_end_dt,
    dt as selected_negative_dt, finish_cnt_7d as selected_negative_label,
    next_candidate_kind, next_candidate_start_dt, next_selected_dt, next_selected_label,
    sample_status, next_sample_status
from ordered_candidates
where candidate_kind = 'success_7d'
  and finish_cnt_7d = 0
  and next_selected_label > 0
  and success_anchor_dt between to_date('2026-07-16') and to_date('2026-09-10')
order by pid, success_anchor_dt
"""
    (HERE / 'audit_zero_success_then_next_positive.sql').write_text(transition_audit)
    meta = {
        'profile': 'v6_dfcl_mt', 'status': 'sql_generated_not_executed',
        'splits': SPLITS,
        'settings': dict(SETTINGS, date_scope='same_as_each_v6_split'),
        'feature_count': 473, 'token_count': 15,
        'source_sql_sha256': source_hashes,
        'propensity': 'not embedded; verify experiment assignment probabilities separately',
        'generated_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(HERE.glob('*.sql'))},
    }
    (HERE / 'profile_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': meta['status'], 'files': list(meta['generated_sha256'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
