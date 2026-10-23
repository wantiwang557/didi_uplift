"""每日选参的纯 pandas 历史仿真；复用调度函数，不连接 Spark 或参数表。

输入是原始 all_rct_data_pdf_v4/v3。需要历史预测及成熟后验标签；本函数按日期
隔离选参与评价，但不能核验历史快照、模型/初始参数是否使用过未来信息。
"""
import copy

import numpy as np
import pandas as pd

if __package__:
    from . import online_opt_param_task as opt
else:
    import online_opt_param_task as opt


def summarize_daily(daily):
    """保留每日负增量；缺支持则整期不可评价，不跳日或平均每日CAC。"""
    rows = []
    for strategy, part in daily.groupby("strategy", sort=False):
        missing = part.loc[
            ~np.isfinite(part[["delta_finish_pv_total", "delta_cost"]]).all(axis=1),
            "effective_dt",
        ].tolist()
        n = int(part["n"].sum())
        pv = float(part["delta_finish_pv_total"].sum()) if not missing else np.nan
        cost = float(part["delta_cost"].sum()) if not missing else np.nan
        row = dict(strategy=strategy, n=n, days=len(part), unsupported_dates=missing,
                   status=("missing_support" if missing else
                           "historical_rct_replay" if pv > 0 else "nonpositive_delta_pv"),
                   delta_finish_pv_total=pv, delta_cost=cost,
                   delta_pv_per_user=pv / n if n else np.nan,
                   delta_cvr_pv_ratio=pv / n if n else np.nan,
                   delta_cac=cost / pv if pv > 0 else np.nan)
        for col in ["no_send_share"] + ["share_" + a for a in opt.ACTION_LIST]:
            row[col] = float((part[col] * part["n"]).sum() / n) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_backtest(raw, start_dt="2026-09-01", end_dt="2026-09-07",
                 initial_params=None, cost_list=None, train_days=7, val_days=7,
                 maturity_lag=6, model_version="v4", include_fixed_c_daily=False):
    """先冻结每日参数，再按生效日回放；返回轨迹、逐日效果、整期对比及明细。

    start_dt/end_dt 是实际生效日期，内部减一天传给 opt.windows。
    默认初始参数也用作全期不更新基准。n 是输入决策行数，不是跨日去重UV。
    假设每日任务成功且当日用户决策前生效，标签已完整回补。
    model_version控制输入预测字段及来源标注；不会重训练模型或修改生产配置。
    include_fixed_c_daily=True时增加第三套方案：固定初始C，在最新成熟
    train_days日预测上每天直接重解lambda；不以该窗口后验决定更新/保持。
    """
    model_sources = {
        "v4": (opt.PREFIX, opt.MODEL_TABLE),
        "v3": ("msg_euen_uplift_cost_26_v3_final_",
               "dm.ut_alg_flowback_outcall_euen_26_v3_model_inference_di"),
    }
    if model_version not in model_sources:
        raise ValueError("model_version 仅支持 'v3' 或 'v4'")
    prefix, model_table = model_sources[model_version]
    start, end = opt.iso_date(start_dt), opt.iso_date(end_dt)
    if start > end:
        raise ValueError("start_dt 不能晚于 end_dt")
    initial = opt.validate_params(opt.INITIAL_PARAMS if initial_params is None else initial_params)
    dates = pd.date_range(start, end).strftime("%Y-%m-%d").tolist()
    schedule, history_dates = [], set()
    for day in dates:
        biz = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        train_dates, val_dates = opt.windows(
            biz, train_days=train_days, val_days=val_days, maturity_lag=maturity_lag)
        schedule.append((day, biz, train_dates, val_dates))
        history_dates.update(train_dates + val_dates)
    # 只转换一次；保留历史模块三券人口、标签及模型头口径。
    history, application = opt.prepare_data(
        raw, sorted(history_dates), dates, prediction_prefix=prefix)

    current, trajectory, selections = dict(initial), [], {}
    fixed_c_trajectory = []
    for day, biz, train_dates, val_dates in schedule:
        if include_fixed_c_daily:
            # 与分桶基准一样截至生效日-(maturity_lag+1)，而非较早的候选求解窗。
            latest_dates = pd.date_range(end=val_dates[-1], periods=train_days).strftime("%Y-%m-%d").tolist()
            latest = history.loc[history.dt.isin(latest_dates)].copy()
            direct_lambda = opt.find_optimal_lambda(latest, initial["C"])
            fixed_c_trajectory.append(dict(
                effective_dt=day, biz_date=biz, strategy="fixed_c_daily",
                train_start=latest_dates[0], train_end=latest_dates[-1],
                selection_start=None, selection_end=None,
                C=initial["C"], **{"lambda": direct_lambda},
                status="DIRECT_RESOLVE",
                alpha=direct_lambda / (1 + direct_lambda * initial["C"])))
        incoming = dict(current)
        result = opt.evaluate_candidates(
            train=history.loc[history.dt.isin(train_dates)].copy(),
            val=history.loc[history.dt.isin(val_dates)].copy(),
            current=incoming, cost_list=cost_list)
        current = dict(result["chosen"])
        # 选优表中的实际胜出记录；无需重复求解/回放，也不读取应用期效果。
        selected = (next(x for x in result["candidates"]
                         if x["C"] == current["C"] and x["lambda"] == current["lambda"])
                    if result["status"] == "UPDATE" else result["old"])
        row = dict(effective_dt=day, biz_date=biz,
                   train_start=train_dates[0], train_end=train_dates[-1],
                   selection_start=val_dates[0], selection_end=val_dates[-1],
                   old_C=incoming["C"], old_lambda=incoming["lambda"],
                   C=current["C"], **{"lambda": current["lambda"]},
                   status=result["status"])
        for metric_prefix, metrics in [("old", result["old"]["validation"]),
                                ("chosen", selected["validation"])]:
            row[metric_prefix + "_selection_cac"] = metrics["delta_cac"]
            row[metric_prefix + "_selection_pv_per_user"] = metrics["delta_pv_per_user"]
        trajectory.append(row)
        selections[day] = result

    # 参数轨迹已经全部确定。应用期后验标签只在下面评价，不回传选参循环。
    daily, replays = [], {}
    direct_by_date = {row["effective_dt"]: row for row in fixed_c_trajectory}
    for row in trajectory:
        day = row["effective_dt"]
        users = application.loc[application.dt == day].copy()
        replays[day] = {}
        policies = [
            ("rolling", {"C": row["C"], "lambda": row["lambda"]}),
            ("fixed_initial", initial),
        ]
        if include_fixed_c_daily:
            direct = direct_by_date[day]
            policies.append(("fixed_c_daily", {"C": direct["C"], "lambda": direct["lambda"]}))
        for strategy, params in policies:
            metrics = opt.replay(users, params)
            replays[day][strategy] = metrics
            item = dict(effective_dt=day, strategy=strategy,
                        alpha=params["lambda"] / (1 + params["lambda"] * params["C"]), **params)
            for key in ["status", "n", "delta_finish_pv_total", "delta_cost",
                        "delta_cac", "delta_pv_per_user", "delta_cvr_pv_ratio",
                        "no_send_share"]:
                item[key] = metrics[key]
            item.update({"share_" + a: metrics["action_share"][a] for a in opt.ACTION_LIST})
            daily.append(item)
    daily = pd.DataFrame(daily)
    summary = summarize_daily(daily)
    indexed = summary.set_index("strategy")
    comparisons = []
    for metric in ["delta_finish_pv_total", "delta_cost", "delta_pv_per_user", "delta_cac"]:
        rolling, fixed = indexed.loc["rolling", metric], indexed.loc["fixed_initial", metric]
        comparison = dict(metric=metric, rolling=rolling, fixed_initial=fixed,
                          rolling_minus_fixed=rolling - fixed)
        if include_fixed_c_daily:
            direct = indexed.loc["fixed_c_daily", metric]
            comparison.update(fixed_c_daily=direct, fixed_c_daily_minus_fixed=direct - fixed,
                              rolling_minus_fixed_c_daily=rolling - direct)
        comparisons.append(comparison)
    strategy_trajectory = []
    for row in trajectory:
        strategy_trajectory.append(dict(row, strategy="rolling",
            alpha=row["lambda"] / (1 + row["lambda"] * row["C"])))
        strategy_trajectory.append(dict(effective_dt=row["effective_dt"], biz_date=row["biz_date"],
            strategy="fixed_initial", train_start=None, train_end=None,
            selection_start=None, selection_end=None, status="FIXED", **initial,
            alpha=initial["lambda"] / (1 + initial["lambda"] * initial["C"])))
    strategy_trajectory.extend(fixed_c_trajectory)
    return dict(trajectory=pd.DataFrame(trajectory), daily=daily, summary=summary,
                strategy_trajectory=pd.DataFrame(strategy_trajectory).sort_values(
                    ["effective_dt", "strategy"]).reset_index(drop=True),
                comparison=pd.DataFrame(comparisons), selections=selections, replays=replays,
                config=dict(start_dt=str(start), end_dt=str(end), initial_params=initial,
                            cost_list=copy.deepcopy(opt.DEFAULT_C_LIST if cost_list is None else cost_list),
                            train_days=train_days, val_days=val_days, maturity_lag=maturity_lag,
                            model_version=model_version, prediction_prefix=prefix,
                            include_fixed_c_daily=include_fixed_c_daily,
                            replacement_rule=opt.REPLACEMENT_RULE,
                            model_table=model_table, action_list=list(opt.ACTION_LIST),
                            evaluation="historical_rct_replay_point_estimate",
                            data_snapshot_verified=False))
