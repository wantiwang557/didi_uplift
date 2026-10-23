"""固定分桶配置和target_cac，每日成熟7日RCT重解的独立离线仿真。

输入原始all_rct_data_pdf_v3/v4；不读写表，不调用在线运筹求解。
分位数函数来自用户2026-09-15补充；其余原求解函数见bucket_opt_legacy。
"""
import copy
import math

import numpy as np
import pandas as pd

if __package__:
    from . import bucket_opt_legacy as legacy
else:
    import bucket_opt_legacy as legacy


CONTROL = "DS空白"
TRAIN_GROUPS = [CONTROL, "留存6张3元", "8元留存5张3", "留存6张5元", "10元留存5张3"]
ACTIONS = [CONTROL, "留存6张3元", "留存6张5元", "10元留存5张3"]


def _quantiles_for_bin_count(bin_count):
    if not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer")
    if bin_count == 1:
        return np.asarray([])
    return np.linspace(1.0 / bin_count, (bin_count - 1.0) / bin_count, bin_count - 1)


def _bin_count_for_column(bin_count, col):
    if isinstance(bin_count, dict):
        if col not in bin_count:
            raise ValueError("bin_count dict missing column: {}".format(col))
        use_bin_count = bin_count[col]
    else:
        use_bin_count = bin_count
    if not isinstance(use_bin_count, int) or use_bin_count < 1:
        raise ValueError("bin_count for {} must be a positive integer".format(col))
    return use_bin_count


def _labels_for_bin_count(bin_count):
    return [round(float(x), 3) for x in np.linspace(1.0 / bin_count, 1.0, bin_count)]


def _unique_increasing_edges(edges):
    unique_edges = []
    for edge in edges:
        edge = float(edge)
        if not unique_edges or edge > unique_edges[-1]:
            unique_edges.append(edge)
    return unique_edges


def move_threshold_train_with_nan_list(train_data, group_name_list, columns, bin_count=5):
    """原样保留用户提供的分位数逻辑，包含NaN/999，常量列标记为1.0。"""
    train_data = train_data.copy()
    quantile_thresholds = {}
    for col in columns:
        if col not in train_data.columns:
            quantile_thresholds[col] = None
            continue
        col_bin_count = _bin_count_for_column(bin_count, col)
        quantiles = _quantiles_for_bin_count(col_bin_count)
        filled_data = train_data[col].fillna(999)
        if filled_data.nunique() > 1 and len(quantiles) > 0:
            quantile_thresholds[col] = filled_data.quantile(quantiles).values
        else:
            quantile_thresholds[col] = None
    for col in columns:
        if col not in train_data.columns:
            continue
        new_col_name = "{}_per".format(col)
        thresholds = quantile_thresholds.get(col)
        if thresholds is None:
            train_data[new_col_name] = 1.0
            continue
        bins = np.concatenate([[-np.inf], thresholds, [np.inf]])
        bins = np.asarray(_unique_increasing_edges(bins))
        labels = _labels_for_bin_count(len(bins) - 1)
        train_data[new_col_name] = pd.cut(train_data[col].fillna(999), bins=bins,
                                        labels=labels, include_lowest=True)
    train_df = train_data[train_data["group_name_eve"].isin(group_name_list)].copy()
    return quantile_thresholds, train_df


def apply_threshold_to_test(test_data, quantile_thresholds, columns):
    """只应用训练阈值；None对应1.0，与用户训练函数保持一致。"""
    result = test_data.copy()
    for col in columns:
        thresholds = quantile_thresholds[col]
        if thresholds is None:
            result[col + "_per"] = 1.0
        else:
            edges = _unique_increasing_edges(np.concatenate([[-np.inf], thresholds, [np.inf]]))
            result[col + "_per"] = pd.cut(result[col].fillna(999), bins=edges,
                labels=_labels_for_bin_count(len(edges) - 1), include_lowest=True)
    return result


def default_bucket_config(model_version="v3"):
    if model_version not in ("v3", "v4"):
        raise ValueError("model_version仅支持v3或v4")
    p = "msg_euen_uplift_cost_26_{}_final_".format(model_version)
    columns = [p + x for x in ["cost_3", "cost_4", "uplift_1", "uplift_diff_3_1", "uplift_diff_4_1"]]
    all_columns = ["fin_order_wxmini_28d_per"] + [c + "_per" for c in columns]
    f, c3, c4, u1, d31, d41 = all_columns
    return dict(columns=columns, bin_count_by_column=dict(zip(columns, [5, 5, 5, 3, 3])),
                all_columns=all_columns,
                stages=[([f, c3, c4, u1, d31, d41], 15000),
                        ([f, u1, d31, d41], 15000), ([f, u1, c3], 20000),
                        ([f, u1], 20000), ([f], 20000)])


def _require_dates(data, dates, name):
    missing = sorted(set(dates) - set(data.dt))
    if missing:
        raise ValueError(name + "缺少日期: " + str(missing))


def _map_day(users, policy):
    binned = apply_threshold_to_test(users, policy["quantile_thresholds"], policy["columns"])
    binned["bucket_num"] = legacy.assign_bucket_series(
        binned, policy["all_columns"], policy["reverse_mapping"])
    binned["bucket_num_discount"] = binned["bucket_num"].map(policy["discount_res"])
    return binned


def _evaluate_mapped(decisions):
    """只回收已冻结的推荐动作，不重新分桶或套用某一天的mapping。"""
    # 保留原回放：推荐动作聚合后，再用RCT均值放缩与对照相减。
    mapped, processed, ans = legacy.evaluate_strategy_df(
        decisions, CONTROL, bucket_col="bucket_num_discount")
    unsupported = []
    for action, part in mapped.groupby("bucket_num_discount"):
        groups = set(part.group_name_eve)
        if action not in groups or CONTROL not in groups:
            unsupported.append(action)
    n = len(decisions)
    complete = not unsupported and math.isclose(float(ans[0]), n, rel_tol=1e-10, abs_tol=1e-8)
    pv, cost = (float(ans[4]), float(ans[5])) if complete else (np.nan, np.nan)
    status = "missing_support" if not complete else (
        "historical_rct_replay" if pv > 0 else "nonpositive_delta_pv")
    metrics = dict(n=n, evaluated_n=float(ans[0]), status=status, unsupported_actions=unsupported,
                   delta_finish_pv_total=pv, delta_cost=cost,
                   delta_cvr_pv_ratio=pv / n, delta_pv_per_user=pv / n,
                   delta_cac=cost / pv if pv > 0 else np.nan,
                   no_send_share=float((mapped.bucket_num_discount == CONTROL).mean()),
                   unknown_bucket_share=float((mapped.bucket_num == -1).mean()),
                   action_share={a: float((mapped.bucket_num_discount == a).mean()) for a in ACTIONS})
    return metrics, dict(legacy_ans=ans, processed_data=processed)


def _evaluate_day(users, policy):
    return _evaluate_mapped(_map_day(users, policy))


def run_bucket_backtest(raw, target_cac, start_dt="2026-09-01", end_dt="2026-09-07",
                        model_version="v3", train_window_size=7, train_gap_days=6,
                        default_coupon="留存6张3元", bucket_config=None,
                        evaluation_mode="daily"):
    """独立执行每日分桶求解与效果回收，target_cac由调用者明确提供。

    固定五组训练人口（含8元）、四动作求解/评价；原求解约束使用zoom_cost，
    回放成本使用相对control的增量成本。默认分桶配置来自用户固定配置。
    成熟度依赖传入数据完整回补；本函数只检查日期覆盖，不核验历史快照。
    evaluation_mode=daily：逐日放缩后累加；pooled：合并每日冻结决策后统一放缩。
    pooled模式仍返回daily/summary_daily诊断，单日缺支持不阻止整期重新判断支持。
    """
    if evaluation_mode not in ("daily", "pooled"):
        raise ValueError("evaluation_mode仅支持'daily'或'pooled'")
    cfg = default_bucket_config(model_version) if bucket_config is None else copy.deepcopy(bucket_config)
    target_cac = float(target_cac)
    if not math.isfinite(target_cac) or target_cac < 0:
        raise ValueError("target_cac必须为有限非负数")
    if train_window_size < 1 or train_gap_days < 6:
        raise ValueError("训练窗至少1天；当前7日成熟口径要求train_gap_days>=6")
    if default_coupon not in ACTIONS:
        raise ValueError("default_coupon必须在三券+不发动作空间中")
    dates = pd.date_range(start_dt, end_dt).strftime("%Y-%m-%d").tolist()
    if not dates:
        raise ValueError("生效日期窗口为空")
    required = ["pid", "dt", "group_name_eve", "uv_label_7d", "pv_label_7d", "cost_7d",
                "fin_order_wxmini_28d_per"] + cfg["columns"]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError("输入缺少列: " + str(missing))
    schedule = []
    for day in dates:
        train_end = pd.Timestamp(day) - pd.Timedelta(days=train_gap_days + 1)
        train_dates = pd.date_range(end=train_end, periods=train_window_size).strftime("%Y-%m-%d").tolist()
        schedule.append((day, train_dates))
    needed = set(dates).union(*(set(x[1]) for x in schedule))
    data = raw[required].copy()
    data["dt"] = pd.to_datetime(data.dt).dt.strftime("%Y-%m-%d")
    data = data.loc[data.dt.isin(needed) & data.group_name_eve.isin(TRAIN_GROUPS)].copy()
    if data.pid.isna().any():
        raise ValueError("pid缺失会导致旧回放人口计数不完整")
    for col in ["uv_label_7d", "pv_label_7d", "cost_7d", "fin_order_wxmini_28d_per"] + cfg["columns"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if not np.isfinite(data[["uv_label_7d", "pv_label_7d", "cost_7d", "fin_order_wxmini_28d_per"]]).all().all():
        raise ValueError("标签或fin_order_wxmini_28d_per含缺失/非有限值")
    evaluation = data.loc[data.group_name_eve.isin(ACTIONS)].copy()
    _require_dates(evaluation, dates, "应用期")
    discount_para = {a: i for i, a in enumerate(ACTIONS)}
    policies, trajectory = {}, []
    # 先完成所有策略求解，应用期回放不参与选择。
    for day, train_dates in schedule:
        train_raw = data.loc[data.dt.isin(train_dates)].copy()
        _require_dates(train_raw, train_dates, "求解集")
        thresholds, train = move_threshold_train_with_nan_list(
            train_raw, TRAIN_GROUPS, cfg["columns"], bin_count=cfg["bin_count_by_column"])
        bucket_mapping, reverse = legacy.build_stage_bucket_mapping(train, cfg["all_columns"], cfg["stages"])
        train["bucket_num"] = legacy.assign_bucket_series(train, cfg["all_columns"], reverse)
        posterior = legacy.process_df(legacy.build_rct_pred_df(train, "bucket_num"), CONTROL)
        y, c, bucket_info = legacy.data_prepare(
            posterior, discount_para, bucket_col="bucket_name", group_name="exp_name", control_name=CONTROL)
        objective, solution = legacy.ortools_coupon(y, c, len(bucket_info), target_cac, discount_para)
        if not solution:
            raise RuntimeError(day + "分桶求解未返回OPTIMAL解；停止，不伪造该日收益")
        mapping = legacy.or_convert(solution, bucket_info, discount_para)
        mapping[-1] = default_coupon
        policies[day] = dict(quantile_thresholds=thresholds, columns=cfg["columns"],
                            all_columns=cfg["all_columns"], reverse_mapping=reverse,
                            bucket_mapping=bucket_mapping, discount_res=mapping)
        trajectory.append(dict(effective_dt=day, train_start=train_dates[0], train_end=train_dates[-1],
                               target_cac=target_cac, train_n=len(train), bucket_count=len(bucket_info),
                               status="SOLVED", train_objective=float(objective)))
    daily, replays, decision_frames = [], {}, []
    for day in dates:
        mapped = _map_day(evaluation.loc[evaluation.dt == day].copy(), policies[day])
        metrics, detail = _evaluate_mapped(mapped)
        if evaluation_mode == "pooled":
            decision_frames.append(mapped[["pid", "dt", "group_name_eve", "uv_label_7d",
                "pv_label_7d", "cost_7d", "bucket_num", "bucket_num_discount"]].copy())
        daily.append(dict(effective_dt=day, strategy="bucket_rolling", target_cac=target_cac, **metrics))
        replays[day] = detail
    daily = pd.DataFrame(daily)
    unsupported_dates = daily.loc[daily.status == "missing_support", "effective_dt"].tolist()
    n = int(daily.n.sum())
    pv = float(daily.delta_finish_pv_total.sum()) if not unsupported_dates else np.nan
    cost = float(daily.delta_cost.sum()) if not unsupported_dates else np.nan
    summary = pd.DataFrame([dict(strategy="bucket_rolling", n=n, days=len(dates), target_cac=target_cac,
        evaluation_mode="daily",
        status="missing_support" if unsupported_dates else (
            "historical_rct_replay" if pv > 0 else "nonpositive_delta_pv"),
        unsupported_dates=unsupported_dates, delta_finish_pv_total=pv, delta_cost=cost,
        delta_cvr_pv_ratio=pv / n, delta_pv_per_user=pv / n, delta_cac=cost / pv if pv > 0 else np.nan,
        no_send_share=float((daily.no_send_share * daily.n).sum() / n),
        unknown_bucket_share=float((daily.unknown_bucket_share * daily.n).sum() / n))])
    summary_daily = summary.copy(deep=True)
    decision_details, pooled_replay = None, None
    if evaluation_mode == "pooled":
        decision_details = pd.concat(decision_frames, ignore_index=True)
        metrics, pooled_replay = _evaluate_mapped(decision_details)
        summary = pd.DataFrame([dict(strategy="bucket_rolling", days=len(dates),
            target_cac=target_cac, evaluation_mode="pooled",
            daily_unsupported_dates=unsupported_dates, **metrics)])
    return dict(trajectory=pd.DataFrame(trajectory), daily=daily, summary=summary,
                summary_daily=summary_daily, decision_details=decision_details,
                pooled_replay=pooled_replay,
                policies=policies, replays=replays,
                config=dict(model_version=model_version, target_cac=target_cac,
                            evaluation_mode=evaluation_mode,
                            train_window_size=train_window_size, train_gap_days=train_gap_days,
                            default_coupon=default_coupon, train_groups=list(TRAIN_GROUPS),
                            evaluation_groups=list(ACTIONS), bucket_config=cfg,
                            solver_cost="zoom_cost", data_snapshot_verified=False))
