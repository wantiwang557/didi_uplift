"""分桶运筹原函数快照，仅包含每日仿真所需依赖。

来源和校验值见 bucket_opt_legacy_source.json；原项目文件未修改。
保留原后验放缩、动作约束和求解逻辑；由bucket_opt_backtest负责日期及输出检查。
"""
import numpy as np
import pandas as pd

def split_data_no_cartesian(small_data, columns, all_columns, min_count):
    """Split rows by whether a staged column combination has enough samples."""
    missing = [col for col in list(columns) + list(all_columns) + ["pid"] if col not in small_data.columns]
    if missing:
        raise ValueError("missing columns for split_data_no_cartesian: %s" % missing)

    combination_counts = (
        small_data.groupby(columns, observed=True)["pid"]
        .count()
        .reset_index(name="counts")
    )
    large_combinations = combination_counts[combination_counts["counts"] >= min_count]
    small_combinations = combination_counts[combination_counts["counts"] < min_count]

    large_sample_data = small_data.merge(large_combinations[columns], on=columns, how="inner")
    small_sample_data = small_data.merge(small_combinations[columns], on=columns, how="inner")

    large_use_combinations = (
        large_sample_data.groupby(all_columns, observed=True)
        .size()
        .reset_index(name="counts")
    )
    small_use_combinations = (
        small_sample_data.groupby(all_columns, observed=True)
        .size()
        .reset_index(name="counts")
    )

    if not large_use_combinations.empty and not (large_use_combinations["counts"] > 0).all():
        raise AssertionError("large_use_combinations contains non-positive counts")
    if not small_use_combinations.empty and not (small_use_combinations["counts"] > 0).all():
        raise AssertionError("small_use_combinations contains non-positive counts")

    return (
        large_sample_data,
        small_sample_data,
        large_combinations,
        small_combinations,
        large_use_combinations,
        small_use_combinations,
    )


def generate_bucket_mapping_final(stage_combinations, all_columns, stage_columns=None):
    """Build bucket -> full-combination mapping from staged split results."""
    bucket_mapping = {}
    bucket_id = 0
    combination_to_bucket = {}
    small_combinations = None

    for large_combinations, small_combinations, columns in stage_combinations:
        if large_combinations is None or large_combinations.empty:
            continue

        for _, row in large_combinations.iterrows():
            combination = tuple(row[col] for col in all_columns)
            base_combination = tuple(row[col] for col in columns) + (8888,) * (
                len(all_columns) - len(columns)
            )

            if base_combination not in combination_to_bucket:
                combination_to_bucket[base_combination] = bucket_id
                bucket_id += 1

            assigned_bucket = combination_to_bucket[base_combination]
            bucket_mapping.setdefault(assigned_bucket, []).append(combination)

    if small_combinations is not None and not small_combinations.empty:
        final_bucket_id = bucket_id
        for _, row in small_combinations.iterrows():
            combination = tuple(row[col] for col in all_columns)
            bucket_mapping.setdefault(final_bucket_id, []).append(combination)

    return bucket_mapping


def assign_bucket_series(data, columns, reverse_mapping):
    """Vectorized equivalent of row-wise ``assign_bucket_num``.

    ``MultiIndex.map`` preserves the exact tuple lookup semantics while avoiding
    a Python ``DataFrame.apply(axis=1)`` over million-row simulation windows.
    """
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("missing bucket columns: %s" % missing)
    keys = pd.MultiIndex.from_frame(data[list(columns)])
    mapped = keys.map(dict(reverse_mapping or {}))
    return pd.Series(mapped, index=data.index).fillna(-1).astype(int)


def build_stage_bucket_mapping(train_df, all_columns, stages):
    """Run staged splitting and return ``(bucket_mapping, reverse_mapping)``."""
    if not stages:
        raise ValueError("未能匹配到传入分阶段划分参数组合")

    stage_results = []
    current_data = train_df.copy()

    for columns, min_count in stages:
        (
            _large_data,
            small_data,
            _large_combinations,
            _small_combinations,
            large_use_combinations,
            small_use_combinations,
        ) = split_data_no_cartesian(current_data, columns, all_columns, min_count)
        current_data = small_data
        stage_results.append((large_use_combinations, small_use_combinations, columns))

    large_use_combinations = stage_results[-1][0]
    small_use_combinations = stage_results[-1][1]
    stage_combinations = [(res[0], None, res[2]) for res in stage_results[:-1]]
    stage_combinations.append((large_use_combinations, small_use_combinations, stages[-1][0]))

    bucket_mapping = generate_bucket_mapping_final(
        stage_combinations,
        all_columns,
        [stage[0] for stage in stages],
    )
    reverse_mapping = {}
    for bucket_num, combinations in bucket_mapping.items():
        for combination in combinations:
            reverse_mapping[combination] = bucket_num

    return bucket_mapping, reverse_mapping


def safe_divide(numerator, denominator, default=0):
    return numerator / denominator if denominator not in (0, None) else default


def process_df(df, control_name, bucket_name="bucket_name"):
    """Scale RCT aggregate rows inside each bucket and compute deltas."""
    pieces = []
    for _, group in df.groupby(bucket_name, observed=True):
        group = group.copy()
        sample = group["in_uv"].sum()
        group["zoom_sample"] = sample
        group["zoom_first"] = group.apply(
            lambda row: safe_divide(row.zoom_sample, row.in_uv) * row.finish_uv,
            axis=1,
        )
        group["zoom_finish"] = group.apply(
            lambda row: safe_divide(row.zoom_sample, row.in_uv) * row.finish_pv,
            axis=1,
        )
        group["zoom_cost"] = group.apply(
            lambda row: safe_divide(row.zoom_sample, row.in_uv) * float(row.total_cost),
            axis=1,
        )
        group["cvr_pv"] = group.apply(
            lambda row: safe_divide(row.zoom_finish, row.zoom_sample),
            axis=1,
        )

        control_rows = group[group["exp_name"] == control_name]
        control_finish = control_rows["zoom_finish"].sum()
        control_first = control_rows["zoom_first"].sum()
        control_cost = control_rows["zoom_cost"].sum()

        group["zoom_delta_cost"] = group["zoom_cost"] - control_cost
        group["zoom_delta_finish"] = group["zoom_finish"] - control_finish
        group["zoom_delta_first"] = group["zoom_first"] - control_first
        pieces.append(group)

    if not pieces:
        return df.copy()
    return pd.concat(pieces, ignore_index=True)


def data_prepare(df, para, bucket_col, group_name, control_name):
    """Create objective and constraint arrays for one-hot coupon assignment."""
    discount_num = len(para)
    bucket_info = list(df[bucket_col].unique())
    bucket_num = len(bucket_info)
    n_vars = discount_num * bucket_num
    target = np.ones((1, n_vars))
    constraint = np.ones((1, n_vars))

    for bucket_idx, bucket_id in enumerate(bucket_info):
        part_df = df[df[bucket_col] == bucket_id]
        sub = bucket_idx * discount_num
        used = set()

        if part_df[part_df[group_name] == control_name].empty:
            raise ValueError("出现空白为空的情况")

        for _, row in part_df.iterrows():
            if row[group_name] not in para:
                continue

            col_idx = sub + para[row[group_name]]
            if row[group_name] == control_name:
                target[0][col_idx] = row.zoom_delta_finish
                constraint[0][col_idx] = 0
            else:
                finish_uv = row.zoom_delta_finish
                coupon_cost = row.zoom_cost
                if finish_uv == 0 and coupon_cost == 0:
                    finish_uv = -100
                    coupon_cost = 0
                target[0][col_idx] = finish_uv
                constraint[0][col_idx] = coupon_cost
            used.add(row[group_name])

        if len(used) != discount_num:
            raise ValueError("折扣数量不一致")

    return target.reshape(-1), constraint.reshape(-1), bucket_info


def _solver_forbidden_constraints(
    solver,
    variables,
    para,
    discount_num,
    forbidden_pairs=None,
    bucket_info=None,
):
    if not forbidden_pairs:
        return
    if bucket_info is None:
        raise ValueError("forbidden_pairs 需要同时传入 bucket_info")
    bucket_index = {bucket_id: idx for idx, bucket_id in enumerate(bucket_info)}
    forbidden_by_bucket = {}
    for bucket_id, coupon_name in forbidden_pairs:
        if bucket_id not in bucket_index or coupon_name not in para:
            continue
        bucket_idx = bucket_index[bucket_id]
        forbidden_by_bucket.setdefault(bucket_id, set()).add(coupon_name)
        var_idx = bucket_idx * discount_num + para[coupon_name]
        solver.Add(variables[var_idx] == 0)
    for bucket_id, coupon_names in forbidden_by_bucket.items():
        if len(coupon_names) >= discount_num:
            raise ValueError("候选动作过滤后 bucket=%s 无可选动作" % bucket_id)


def ortools_coupon(target, constraint, bucket_num, set_cost, para, forbidden_pairs=None, bucket_info=None):
    """Solve coupon assignment with OR-Tools SCIP backend."""
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ImportError("ortools is required for ortools_coupon") from exc

    discount_num = len(para)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is unavailable")

    variables = [solver.IntVar(0, 1, "x%s" % i) for i in range(len(target))]
    obj_exp = solver.Sum([variables[i] * float(target[i]) for i in range(len(target))])
    st_exp_cost = solver.Sum([variables[i] * float(constraint[i]) for i in range(len(constraint))])

    for bucket_idx in range(bucket_num):
        bucket_vars = variables[bucket_idx * discount_num : (bucket_idx + 1) * discount_num]
        solver.Add(solver.Sum(bucket_vars) == 1)

    _solver_forbidden_constraints(
        solver,
        variables,
        para,
        discount_num,
        forbidden_pairs=forbidden_pairs,
        bucket_info=bucket_info,
    )

    solver.Add(st_exp_cost <= set_cost * obj_exp)
    solver.Maximize(obj_exp)

    result_status = solver.Solve()
    res = []
    max_finish = 0
    if result_status == pywraplp.Solver.OPTIMAL:
        res = [var.solution_value() for var in variables]
        max_finish = solver.Objective().Value()

    return max_finish, res


def or_convert(res, bucket_info, para):
    """Convert solver vector into ``bucket -> coupon`` mapping."""
    new_para = dict((v, k) for k, v in para.items())
    discount_num = len(para)
    coupon_ans = {}

    for bucket_idx, bucket_id in enumerate(bucket_info):
        selected = False
        for j in range(bucket_idx * discount_num, (bucket_idx + 1) * discount_num):
            if res[j] == 1:
                coupon_ans[bucket_id] = new_para[j - bucket_idx * discount_num]
                selected = True
                break
        if not selected:
            if 0 not in new_para:
                raise ValueError("未设置默认发券情况")
            coupon_ans[bucket_id] = new_para[0]

    return coupon_ans


def mock_use_simulation(df, control_name):
    """Evaluate a simulated strategy where bucket value is the selected coupon."""
    treat_sample = treat_finish_uv = treat_finish_pv = treat_cost = 0
    control_sample = control_finish_uv = control_finish_pv = control_cost = 0

    for value in list(df["bucket_name"].unique()):
        treat_df = df[(df["bucket_name"] == value) & (df["exp_name"] == df["bucket_value"])]
        control_df = df[(df["bucket_name"] == value) & (df["exp_name"] == control_name)]
        if treat_df.shape[0] == 0 or control_df.shape[0] == 0:
            continue

        treat_sample += treat_df["zoom_sample"].values[0]
        treat_finish_uv += treat_df["zoom_first"].values[0]
        treat_finish_pv += treat_df["zoom_finish"].values[0]
        treat_cost += treat_df["zoom_cost"].values[0]
        control_sample += control_df["zoom_sample"].values[0]
        control_finish_uv += control_df["zoom_first"].values[0]
        control_finish_pv += control_df["zoom_finish"].values[0]
        control_cost += control_df["zoom_cost"].values[0]

    delta_finish_uv = treat_finish_uv - control_finish_uv
    delta_finish_pv = treat_finish_pv - control_finish_pv
    delta_cost = treat_cost - control_cost
    return [
        treat_sample,
        treat_finish_uv,
        treat_finish_pv,
        delta_finish_uv,
        delta_finish_pv,
        delta_cost,
        safe_divide(delta_finish_uv, treat_sample),
        safe_divide(delta_finish_pv, treat_sample),
        safe_divide(delta_cost, delta_finish_uv),
        safe_divide(delta_cost, delta_finish_pv),
    ]


def build_rct_pred_df(df, bucket_col, bucket_value_col=None, weight_col=None):
    """Aggregate user-level RCT rows to bucket/treatment rows."""
    if weight_col is not None and weight_col not in df.columns:
        raise ValueError("missing weight_col=%s" % weight_col)
    rows = []
    for name, group in df.groupby([bucket_col, "group_name_eve"]):
        bucket_id = name[0]
        exp_name = name[1]
        bucket_value = bucket_id if bucket_value_col is None else group[bucket_value_col].iloc[0]
        if weight_col is None:
            in_uv = group["pid"].count()
            finish_uv = group["uv_label_7d"].sum()
            finish_pv = group["pv_label_7d"].sum()
            total_cost = group["cost_7d"].sum()
        else:
            weights = group[weight_col].astype(float)
            in_uv = weights.sum()
            finish_uv = (weights * group["uv_label_7d"].astype(float)).sum()
            finish_pv = (weights * group["pv_label_7d"].astype(float)).sum()
            total_cost = (weights * group["cost_7d"].astype(float)).sum()
        rows.append(
            [
                bucket_id,
                bucket_value,
                exp_name,
                in_uv,
                finish_uv,
                finish_pv,
                total_cost,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "bucket_name",
            "bucket_value",
            "exp_name",
            "in_uv",
            "finish_uv",
            "finish_pv",
            "total_cost",
        ],
    )


def evaluate_strategy_df(df, control_name, discount_res=None, bucket_col="bucket_num"):
    """Evaluate one strategy mapping on a user-level DataFrame."""
    strategy_df = df.copy()
    if discount_res is not None:
        strategy_df["bucket_num_discount"] = strategy_df["bucket_num"].map(discount_res)

    processed_data_sim = process_df(
        build_rct_pred_df(strategy_df, bucket_col),
        control_name=control_name,
    )
    ans = mock_use_simulation(processed_data_sim, control_name=control_name)
    return strategy_df, processed_data_sim, ans
