"""Generate the full v6 episode-profile SQL from the v6_dfcl_touch wrappers.

The large v6 feature projection stays byte-for-byte aligned with the existing
wrapper. Only the sample-selection CTE and profile names are replaced.
"""
from pathlib import Path


HERE = Path(__file__).parent
SOURCE_DIR = HERE.parent / "v6_dfcl_touch"

SPLITS = {
    "train": ("2026-07-16", "2026-08-21"),
    "valid": ("2026-08-22", "2026-08-27"),
    "test": ("2026-08-28", "2026-09-10"),
}


def selection_cte(split, begin, end):
    return f"""-- v6_dfcl_episode {split}
-- 选样口径：
-- 1) 当天短信/外呼成功：保留该行；finish_cnt_7d 可为 0 或正数。
-- 2) finish_cnt_7d > 0：保留上游归因到的正记录；若与成功行重合只保留一次。
-- 3) 同一连续在组阶段既无成功也无正记录：仅保留末次入组的成熟负记录。
-- 4) 相邻记录间隔 >= 28 天近似识别重新入组的新阶段。
-- 5) train/valid/test 独立取末次负记录；同一持续在组用户可能各 split 出现一次。
-- 6) 不使用原 v6 的 finish_all_cnt outcome 筛选；control cost 强制为 0。

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
    where t.dt between '{begin}' and '{end}'
      and t.dt not between '2026-06-18' and '2026-06-22'
      and t.dt not in (
          '2026-03-26', '2026-03-27', '2026-04-08',
          '2026-08-02', '2026-08-08', '2026-08-09'
      )
      and t.pid is not null
      and t.new_exp = '迭代流量'
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
        max(outcall_success_cnt) as outcall_success_cnt,
        count(distinct source_signature) as source_variant_cnt,
        max(case when finish_cnt_7d is null or finish_cnt_7d < 0 then 1 else 0 end)
            as bad_label_flag
    from label_raw
    group by pid, dt, flowtree_group_name, group_name_eve, treatment_id
),
daily_flags as (
    select
        d.*,
        case when send_success_cnt > 0 or outcall_success_cnt > 0 then 1 else 0 end
            as touch_success,
        case when finish_cnt_7d > 0 then 1 else 0 end as positive_flag,
        case
            when send_success_cnt > 0 or outcall_success_cnt > 0
              or finish_cnt_7d > 0 then 1
            else 0
        end as anchor_flag,
        count(*) over (partition by pid, dt) as pid_day_key_cnt,
        lag(dt) over (partition by pid order by dt) as previous_dt
    from daily d
),
episode_starts as (
    select
        d.*,
        case
            when previous_dt is null then 1
            when datediff(dt, previous_dt) >= 28 then 1
            else 0
        end as new_episode_flag
    from daily_flags d
),
episode_rows as (
    select
        e.*,
        sum(new_episode_flag) over (
            partition by pid
            order by dt
            rows between unbounded preceding and current row
        ) as episode_id
    from episode_starts e
),
user_quality as (
    select
        pid,
        count(distinct treatment_id) as user_treatment_cnt,
        max(case when source_variant_cnt > 1 or pid_day_key_cnt > 1 then 1 else 0 end)
            as user_source_conflict_flag,
        max(bad_label_flag) as user_bad_label_flag
    from episode_rows
    group by pid
),
episode_ranked as (
    select
        e.*,
        q.user_treatment_cnt,
        q.user_source_conflict_flag,
        q.user_bad_label_flag,
        sum(anchor_flag) over (
            partition by e.pid, e.episode_id
        ) as episode_anchor_cnt,
        row_number() over (
            partition by e.pid, e.episode_id
            order by e.dt desc, e.flowtree_group_name, e.group_name_eve
        ) as episode_last_rank
    from episode_rows e
    inner join user_quality q
        on e.pid = q.pid
),
selected_keys as (
    select
        pid,
        flowtree_group_name,
        group_name_eve,
        dt
    from episode_ranked
    where user_treatment_cnt = 1
      and user_source_conflict_flag = 0
      and user_bad_label_flag = 0
      and (
          anchor_flag = 1
          or (episode_anchor_cnt = 0 and episode_last_rank = 1)
      )
),
original_v6_rows as ("""


def build(split, begin, end):
    source = SOURCE_DIR / f"feature_v6_dfcl_touch_{split}.sql"
    sql = source.read_text()
    body_start = sql.index(",\noriginal_v6_rows as (")
    body_end = body_start + len(",\noriginal_v6_rows as (")
    body = sql[body_end:]
    output = selection_cte(split, begin, end) + body
    target = HERE / f"feature_v6_dfcl_episode_{split}.sql"
    target.write_text(output)


for split_name, (split_begin, split_end) in SPLITS.items():
    build(split_name, split_begin, split_end)
