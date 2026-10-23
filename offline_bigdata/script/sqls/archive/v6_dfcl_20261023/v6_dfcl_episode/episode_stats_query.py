"""Paste this file into Jupyter to audit the v6_dfcl_episode sample rule."""

from pyspark.sql import functions as F
from pyspark.sql.window import Window


query = r"""
with raw_source as (
    select
        case
            when t.dt between '2026-07-16' and '2026-08-21' then 'train'
            when t.dt between '2026-08-22' and '2026-08-27' then 'valid'
            when t.dt between '2026-08-28' and '2026-09-10' then 'test'
        end as split,
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
    where t.dt between '2026-07-16' and '2026-09-10'
      and t.dt not in (
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
        split, pid, dt, flowtree_group_name, group_name_eve, treatment_id,
        max(finish_cnt_7d) as finish_cnt_7d,
        max(send_success_cnt) as send_success_cnt,
        max(outcall_success_cnt) as outcall_success_cnt,
        count(distinct source_signature) as source_variant_cnt,
        max(case when finish_cnt_7d is null or finish_cnt_7d < 0 then 1 else 0 end)
            as bad_label_flag
    from raw_source
    group by
        split, pid, dt, flowtree_group_name, group_name_eve, treatment_id
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
        count(*) over (partition by split, pid, dt) as pid_day_key_cnt,
        lag(dt) over (partition by split, pid order by dt) as previous_dt
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
            partition by split, pid
            order by dt
            rows between unbounded preceding and current row
        ) as episode_id
    from episode_starts e
),
user_quality as (
    select
        split,
        pid,
        count(distinct treatment_id) as user_treatment_cnt,
        max(case when source_variant_cnt > 1 or pid_day_key_cnt > 1 then 1 else 0 end)
            as user_source_conflict_flag,
        max(bad_label_flag) as user_bad_label_flag
    from episode_rows
    group by split, pid
),
episode_ranked as (
    select
        e.*,
        q.user_treatment_cnt,
        q.user_source_conflict_flag,
        q.user_bad_label_flag,
        count(*) over (
            partition by e.split, e.pid, e.episode_id
        ) as episode_row_cnt,
        sum(e.anchor_flag) over (
            partition by e.split, e.pid, e.episode_id
        ) as episode_anchor_cnt,
        sum(e.positive_flag) over (
            partition by e.split, e.pid, e.episode_id
        ) as episode_positive_cnt,
        sum(e.touch_success) over (
            partition by e.split, e.pid, e.episode_id
        ) as episode_success_cnt,
        row_number() over (
            partition by e.split, e.pid, e.episode_id
            order by e.dt desc, e.flowtree_group_name, e.group_name_eve
        ) as episode_last_rank
    from episode_rows e
    inner join user_quality q
        on e.split = q.split
       and e.pid = q.pid
)
select
    *,
    case
        when user_treatment_cnt <> 1
          or user_source_conflict_flag <> 0
          or user_bad_label_flag <> 0 then 0
        when anchor_flag = 1 then 1
        when episode_anchor_cnt = 0 and episode_last_rank = 1 then 1
        else 0
    end as selected_flag,
    case
        when user_treatment_cnt <> 1
          or user_source_conflict_flag <> 0
          or user_bad_label_flag <> 0 then 'source_problem'
        when touch_success = 1 and positive_flag = 1 then 'success_positive'
        when touch_success = 1 then 'success_negative'
        when positive_flag = 1 then 'attributed_positive'
        when episode_anchor_cnt = 0 and episode_last_rank = 1 then 'terminal_negative'
        else 'dropped_retry'
    end as selection_reason
from episode_ranked
"""


audit_df = spark.sql(query).cache()
selected_df = audit_df.where("selected_flag = 1")

# 1. 每个 split 各类最终样本数量和占比
reason_window = Window.partitionBy("split")
sample_reason_pdf = (
    selected_df
    .groupBy("split", "selection_reason")
    .agg(F.count("*").alias("sample_cnt"))
    .withColumn("split_total_cnt", F.sum("sample_cnt").over(reason_window))
    .withColumn("sample_ratio", F.round(F.col("sample_cnt") / F.col("split_total_cnt"), 8))
    .orderBy("split", "selection_reason")
    .toPandas()
)

# 2. 每个 split、treatment 的最终样本占比，仅作选样偏移诊断。
# 注意：这是成功/完单选样后的占比，不能直接当作 DFCL propensity。
treatment_window = Window.partitionBy("split")
treatment_ratio_pdf = (
    selected_df
    .groupBy("split", "treatment_id", "group_name_eve")
    .agg(F.count("*").alias("sample_cnt"))
    .withColumn("split_total_cnt", F.sum("sample_cnt").over(treatment_window))
    .withColumn("sample_ratio", F.round(F.col("sample_cnt") / F.col("split_total_cnt"), 8))
    .orderBy("split", "treatment_id")
    .toPandas()
)

# 3. 连续在组阶段中的多正记录比例
episode_df = audit_df.where("episode_last_rank = 1")
multi_positive_pdf = (
    episode_df
    .groupBy("split")
    .agg(
        F.count("*").alias("episode_cnt"),
        F.sum((F.col("episode_positive_cnt") >= 1).cast("long")).alias("positive_episode_cnt"),
        F.sum((F.col("episode_positive_cnt") > 1).cast("long")).alias("multi_positive_episode_cnt"),
    )
    .withColumn(
        "multi_positive_ratio_all",
        F.round(F.col("multi_positive_episode_cnt") / F.col("episode_cnt"), 8),
    )
    .withColumn(
        "multi_positive_ratio_in_positive",
        F.round(F.col("multi_positive_episode_cnt") / F.col("positive_episode_cnt"), 8),
    )
    .orderBy("split")
    .toPandas()
)

# 4. 原始记录压缩成最终样本的比例
compression_pdf = (
    audit_df
    .groupBy("split")
    .agg(
        F.count("*").alias("raw_daily_row_cnt"),
        F.sum("selected_flag").alias("selected_sample_cnt"),
        F.sum((F.col("selection_reason") == "source_problem").cast("long"))
            .alias("source_problem_row_cnt"),
    )
    .withColumn(
        "selected_ratio",
        F.round(F.col("selected_sample_cnt") / F.col("raw_daily_row_cnt"), 8),
    )
    .orderBy("split")
    .toPandas()
)

display(sample_reason_pdf)
display(treatment_ratio_pdf)
display(multi_positive_pdf)
display(compression_pdf)
