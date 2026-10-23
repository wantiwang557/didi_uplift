"""26_v4 定时选参：Spark SQL -> pandas bounded 求解 -> 同窗回放 -> 单向一换一更新。

可以独立复制到调度平台，不依赖项目模块。导入不会启动Spark或写表。
依赖：numpy / pandas / scipy；真正运行取数和写表时使用平台已有SparkSession。
"""
import argparse
import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


INITIAL_PARAMS = {"C": 5.6, "lambda": 0.1218002994}
ACTION_LIST = ["left_63", "C", "left_65", "left_1053"]
ACTION_HEADS = {"left_63": 1, "left_65": 3, "left_1053": 4}
DICT_COUPON = {
    "left_63": "留存6张3元", "C": "DS空白",
    "left_65": "留存6张5元", "left_1053": "10元留存5张3",
}
PREFIX = "msg_euen_uplift_cost_26_v4_final_"
LABEL_TABLE = "dm.ut_alg_flowback_msg_outcall_label_v2_di"
MODEL_TABLE = "dm.ut_alg_flowback_outcall_euen_26_v4_new_rct_new_feature_month_5_6_model_inference_di"
PARAM_KEY = "online_opt_param"
DEFAULT_C_LIST = [i / 10.0 for i in range(40, 60)]
REPLACEMENT_RULE = "pv_nondecrease_one_for_one_v1"

# 保留用户提供的原始取数SQL；8元组在prepare_data中按约定排除。
QUERY_ALL_FMT = """
with base as (
    select t1.*
    from
    ( select
        if(flowtree_group_name = 'DS空白', 'DS空白', group_name_eve) as group_name_eve,
        pid, oneid,
        flowtree_group_name,
        send_cnt, send_success_cnt, send_cost, send_remind_cost, outcall_cost, remind_cost, msg_coupon_cost as coupon_cost,outcall_coupon_cost,outcall_coupon2_cost, 
        cast(if(finish_cnt > 0, 1.0, 0.0) as double) as uv_label,
        finish_cnt as pv_label,
        cast(if(finish_cnt_7d > 0, 1.0, 0.0) as double) as uv_label_7d,
        finish_cnt_7d as pv_label_7d,
        cast((nvl(send_cost, 0.0) + nvl(send_remind_cost, 0.0)+ nvl(outcall_cost, 0.0)+ nvl(remind_cost, 0.0)+ nvl(msg_coupon_cost, 0.0)+ nvl(outcall_coupon_cost, 0.0)+ nvl(outcall_coupon2_cost, 0.0)) as double) as cost,
        cast((nvl(send_cost, 0.0) + nvl(send_remind_cost, 0.0)+ nvl(outcall_cost, 0.0)+ nvl(remind_cost, 0.0)+ nvl(msg_coupon_cost_7d, 0.0)+ nvl(outcall_coupon_cost_7d, 0.0)+ nvl(outcall_coupon2_cost_7d, 0.0)) as double) as cost_7d,
        dt, 
        cast(nvl(fin_order_mini_28d, 999) as double) as fin_order_wxmini_28d,
        cast(
            if(nvl(fin_order_mini_28d, 999) >= 9, 1.0, 0.0)
            as double
        ) as fin_order_wxmini_28d_per
    from {traindata_tbname_via_growstrategy}
    where dt in {seed_dts}
    and new_exp = '迭代流量'
    and oneid is not null 
    and oneid not in ('')
    and ((flowtree_group_name = '探索' and group_name_eve in ('8元留存5张3', '留存6张5元', '留存6张3元','10元留存5张3')) or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group'))
    )t1
),
prediction_agg as (
    select
        bs.*,
        nvl(mapping_data.left63, 999) as msg_euen_uplift_cost_26_v4_final_uplift_1,
        nvl(mapping_data.left853, 999) as msg_euen_uplift_cost_26_v4_final_uplift_2,
        nvl(mapping_data.left65, 999) as msg_euen_uplift_cost_26_v4_final_uplift_3,
        nvl(mapping_data.left1053, 999) as msg_euen_uplift_cost_26_v4_final_uplift_4,
        nvl(mapping_data.cost_1, 999) as msg_euen_uplift_cost_26_v4_final_cost_1,
        nvl(mapping_data.cost_2, 999) as msg_euen_uplift_cost_26_v4_final_cost_2,
        nvl(mapping_data.cost_3, 999) as msg_euen_uplift_cost_26_v4_final_cost_3,
        nvl(mapping_data.cost_4, 999) as msg_euen_uplift_cost_26_v4_final_cost_4,
        nvl(mapping_data.left853_minus_left63, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_2_1,
        nvl(mapping_data.left65_minus_left63, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_3_1,
        nvl(mapping_data.left1053_minus_left63, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_4_1,
        nvl(mapping_data.left65_minus_left853, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_3_2,
        nvl(mapping_data.left1053_minus_left853, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_4_2,
        nvl(mapping_data.left1053_minus_left65, 999) as msg_euen_uplift_cost_26_v4_final_uplift_diff_4_3,
        mapping_data.treatment as msg_euen_uplift_cost_26_v4_treatment
    from (
        select * from base
    ) bs
    left join (
        select 
            pid,
            dt,
            treatment,
            uplift_1 as left63,
            uplift_2 as left853,
            uplift_3 as left65,
            uplift_4 as left1053,
            cost_1,
            cost_2,
            cost_3,
            cost_4,
            uplift_2_1 as left853_minus_left63,
            uplift_3_1 as left65_minus_left63,
            uplift_4_1 as left1053_minus_left63,
            uplift_3_2 as left65_minus_left853,
            uplift_4_2 as left1053_minus_left853,
            uplift_4_3 as left1053_minus_left65
        from {mapping_data}
        where dt in {seed_dts}
    ) mapping_data on bs.pid = mapping_data.pid and bs.dt = mapping_data.dt
)

select *
from prediction_agg
"""


def iso_date(value):
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def table_name(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", value):
        raise ValueError("表名须为table或database.table")
    return value


def windows(exe_dt, mature_end_dt=None, train_days=7, val_days=7, maturity_lag=6):
    """exe_dt是BIZ_DATE_LINE（实际运行日-1）；默认结束于业务日-6，即运行日-7。"""
    day = iso_date(exe_dt)
    if train_days < 1 or val_days < 1 or maturity_lag < 6:
        raise ValueError("窗口长度须为正，至少回退6个业务日期日（实际运行日回退7天）")
    end = iso_date(mature_end_dt) if mature_end_dt else day - timedelta(days=maturity_lag)
    if (day + timedelta(days=1) - end).days < 7:
        raise ValueError("选优窗口包含未满7天的cohort")
    val = [(end - timedelta(days=i)).isoformat() for i in reversed(range(val_days))]
    train_end = end - timedelta(days=val_days)
    train = [(train_end - timedelta(days=i)).isoformat() for i in reversed(range(train_days))]
    return train, val


def build_query(dates):
    if not dates:
        raise ValueError("取数日期为空")
    dates = sorted({iso_date(d).isoformat() for d in dates})
    return QUERY_ALL_FMT.format(
        seed_dts="(" + ", ".join("'" + d + "'" for d in dates) + ")",
        traindata_tbname_via_growstrategy=LABEL_TABLE, mapping_data=MODEL_TABLE,
    )


def validate_params(params):
    result = {"C": float(params["C"]), "lambda": float(params["lambda"])}
    if not all(math.isfinite(v) and v >= 0 for v in result.values()):
        raise ValueError("C、lambda必须是有限非负数")
    return result


def prepare_data(raw, train_dates, val_dates, prediction_prefix=None):
    # 离线仿真可显式选择V3字段；生产任务未传参时仍使用V4。
    prefix = PREFIX if prediction_prefix is None else prediction_prefix
    # 与用户当前三券实验相同：同时排除8元候选和实际8元RCT样本。
    data = raw.loc[raw["group_name_eve"].isin(DICT_COUPON.values())].copy()
    data["dt"] = pd.to_datetime(data["dt"]).dt.strftime("%Y-%m-%d")
    data["finish_cnt_7d"] = pd.to_numeric(data["pv_label_7d"], errors="raise")
    data["cost_7d"] = pd.to_numeric(data["cost_7d"], errors="raise")
    for action, head in ACTION_HEADS.items():
        for target, source in [("delta_cvr_", "uplift_"), ("cost_", "cost_")]:
            data[target + action] = pd.to_numeric(data[prefix + source + str(head)], errors="raise")
    # 不额外处理999，不按触达成功、历史完单档位或实际成本过滤人口。
    train = data.loc[data.dt.isin(train_dates)].copy()
    val = data.loc[data.dt.isin(val_dates)].copy()
    for name, part, dates in [("求解集", train, train_dates), ("选优集", val, val_dates)]:
        if part.empty:
            raise ValueError(name + "为空，停止更新")
        missing = sorted(set(dates) - set(part.dt))
        if missing:
            raise ValueError(name + "缺少日期: " + str(missing))
        labels = part[["finish_cnt_7d", "cost_7d"]].to_numpy(dtype=float)
        if not np.isfinite(labels).all():
            raise ValueError(name + "标签缺失或非有限，停止更新")
    return train, val


def intervention_scores(df, lambda_val, cost):
    scores = pd.DataFrame(index=df.index)
    for action in ACTION_LIST:
        scores[action] = (0.0 if action == "C" else df["delta_cvr_" + action]
                          + lambda_val * (cost * df["delta_cvr_" + action] - df["cost_" + action]))
    return scores


def objective_function(lambda_val, df, cost):
    return float(intervention_scores(df, lambda_val, cost).max(axis=1).sum())


def find_optimal_lambda(df, cost, lamb_s=0.0, lamb_end=10.0):
    if df.empty or not 0 <= lamb_s < lamb_end:
        raise ValueError("求解样本为空或lambda区间无效")
    result = minimize_scalar(objective_function, args=(df, cost),
                             bounds=(lamb_s, lamb_end), method="bounded",
                             options={"xatol": 1e-6})
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("lambda求解失败: " + str(result.message))
    candidates = [float(lamb_s), float(result.x), float(lamb_end)]
    return min(candidates, key=lambda x: (objective_function(x, df, cost), x))


def replay(df, params):
    """推荐层RCT均值放缩；返回后验指标和预测诊断，两者不混用。"""
    params = validate_params(params)
    selected = df.copy()
    selected["best_intervention"] = intervention_scores(
        selected, params["lambda"], params["C"]).idxmax(axis=1)
    details = []
    pred_pv = pred_cost = 0.0
    for action, layer in selected.groupby("best_intervention", sort=False):
        treatment = layer.loc[layer.group_name_eve == DICT_COUPON[action]]
        control = layer.loc[layer.group_name_eve == DICT_COUPON["C"]]
        supported = action == "C" or (len(treatment) > 0 and len(control) > 0)
        if action == "C":
            dy = dc = 0.0
        elif supported:
            dy = len(layer) * (treatment.finish_cnt_7d.mean() - control.finish_cnt_7d.mean())
            dc = len(layer) * (treatment.cost_7d.mean() - control.cost_7d.mean())
        else:
            dy = dc = float("nan")
        if action != "C":
            pred_pv += float(layer["delta_cvr_" + action].sum())
            pred_cost += float(layer["cost_" + action].sum())
        details.append(dict(action=action, n=len(layer), n_t=len(treatment), n_c=len(control),
                            supported=supported, delta_pv=float(dy), delta_cost=float(dc)))
    supported = bool(details) and all(d["supported"] for d in details)
    dy = sum(d["delta_pv"] for d in details) if supported else float("nan")
    dc = sum(d["delta_cost"] for d in details) if supported else float("nan")
    status = ("empty_population" if not details else "missing_support" if not supported
              else "nonpositive_delta_pv" if dy <= 0 else "historical_rct_replay")
    ratio = dy / len(df) if len(df) else float("nan")
    residual = pred_cost - params["C"] * pred_pv
    return dict(status=status, n=len(df), delta_finish_pv_total=dy, delta_cost=dc,
                delta_cvr_pv_ratio=ratio, delta_pv_per_user=ratio,
                delta_cac=dc / dy if supported and dy > 0 else float("nan"),
                predicted_delta_pv=pred_pv, predicted_cost=pred_cost,
                predicted_delta_cac=pred_cost / pred_pv if pred_pv > 0 else float("nan"),
                constraint_residual=residual,
                predicted_feasible=bool(pred_pv > 0 and residual <= 1e-8),
                no_send_share=float((selected.best_intervention == "C").mean()),
                action_share={a: float((selected.best_intervention == a).mean()) for a in ACTION_LIST},
                details=details)


def valid_metrics(metrics):
    return (metrics["status"] == "historical_rct_replay"
            and math.isfinite(metrics["delta_cac"])
            and math.isfinite(metrics["delta_cvr_pv_ratio"])
            and metrics["delta_cvr_pv_ratio"] > 0)


def dominates(new, old):
    """兼容原函数名：拿量不降，允许CAC相对涨幅不超过拿量相对提升。

    相同效果保持；原有双指标改善仍接受。旧CAC<=0时不用相对涨幅，
    仅接受原有双指标改善，避免零分母或负基数反转比较方向。
    """
    if not valid_metrics(new) or not valid_metrics(old):
        return False
    new_pv, old_pv = new["delta_cvr_pv_ratio"], old["delta_cvr_pv_ratio"]
    new_cac, old_cac = new["delta_cac"], old["delta_cac"]
    if new_pv < old_pv:
        return False
    if new_cac <= old_cac:
        return new_pv > old_pv or new_cac < old_cac
    # old_pv>0已由valid_metrics保证。交叉相乘等价于两项相对涨幅比较。
    return (old_cac > 0 and new_pv > old_pv
            and new_cac * old_pv <= old_cac * new_pv)


def select_candidate(old_metrics, candidates):
    if not valid_metrics(old_metrics):
        return None, "KEEP_BASELINE_UNEVALUABLE"
    winners = [x for x in candidates if dominates(x["validation"], old_metrics)]
    if not winners:
        return None, "KEEP_NO_DOMINATING_CANDIDATE"
    best = min(winners, key=lambda x: (-x["validation"]["delta_cvr_pv_ratio"],
                                      x["validation"]["delta_cac"], x["C"], x["lambda"]))
    return best, "UPDATE"


def evaluate_candidates(train, val, current, cost_list=None):
    current = validate_params(current)
    old = {"C": current["C"], "lambda": current["lambda"],
           "train": replay(train, current), "validation": replay(val, current)}
    # 包含旧C用于新窗重新求解；但旧参数仍原样作为独立基准回放。
    costs = sorted(set(DEFAULT_C_LIST if cost_list is None else cost_list) | {current["C"]})
    candidates = []
    for cost in costs:
        if not math.isfinite(float(cost)) or cost < 0:
            raise ValueError("候选C必须为有限非负数")
        lam = find_optimal_lambda(train, float(cost))
        params = {"C": float(cost), "lambda": lam}
        candidates.append(dict(params, alpha=lam / (1 + lam * cost),
                               lambda_at_boundary=lam in (0.0, 10.0),
                               train=replay(train, params), validation=replay(val, params)))
    winner, status = select_candidate(old["validation"], candidates)
    chosen = {k: winner[k] for k in ("C", "lambda")} if winner else current
    return dict(status=status, old=old, candidates=candidates, chosen=chosen,
                replacement_rule=REPLACEMENT_RULE)


def read_current(spark, param_table, exe_dt):
    """读最新成功分区（包含同日重试）；查询异常必须上抛，不能回退初始参数。"""
    param_table = table_name(param_table)
    day = iso_date(exe_dt).isoformat()
    if not spark.catalog.tableExists(param_table):
        return None
    rows = spark.sql("SELECT param_value, dt FROM " + param_table
                     + " WHERE param_key = '" + PARAM_KEY + "' ORDER BY dt DESC LIMIT 2").collect()
    if not rows:
        return None
    if len(rows) > 1 and str(rows[0]["dt"]) == str(rows[1]["dt"]):
        raise ValueError("最新参数分区存在多条配置，停止更新")
    saved_dt = iso_date(rows[0]["dt"]).isoformat()
    if saved_dt > day:
        raise ValueError("已有更晚生效分区，禁止历史补跑覆盖当前参数")
    payload = json.loads(rows[0]["param_value"])
    validate_params(payload)
    if (payload.get("model_table") != MODEL_TABLE or payload.get("action_list") != ACTION_LIST
            or payload.get("action_heads") != ACTION_HEADS):
        raise ValueError("现用参数的模型或动作配置与本任务不一致")
    return dict(payload=payload, dt=saved_dt)


def build_payload(params, exe_dt, train_dates, val_dates, run_id, parameter_origin="rolling_search"):
    return dict(validate_params(params), schema_version=1, model_table=MODEL_TABLE,
                action_list=ACTION_LIST, action_heads=ACTION_HEADS, updated_dt=exe_dt,
                parameter_origin=parameter_origin,
                calibration_dates=train_dates if parameter_origin == "rolling_search" else None,
                comparison_train_dates=train_dates, validation_dates=val_dates, run_id=run_id)


def write_params(spark, param_table, exe_dt, payload, expected_current):
    """单任务串行调度。完整JSON单行写入；重读可检出多数陈旧更新，不是并发CAS。"""
    param_table = table_name(param_table)
    day = iso_date(exe_dt).isoformat()
    if read_current(spark, param_table, day) != expected_current:
        raise RuntimeError("计算期间生效参数发生变化，本轮停止覆盖")
    spark.sql("CREATE TABLE IF NOT EXISTS " + param_table
              + " (param_key STRING, param_value STRING) PARTITIONED BY (dt STRING) STORED AS PARQUET")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    view = "online_opt_param_write_" + uuid.uuid4().hex
    spark.createDataFrame([(PARAM_KEY, encoded)], "param_key string, param_value string").createOrReplaceTempView(view)
    try:
        # 不用str(dict)/引号替换；临时视图保留JSON全文与浮点精度。
        spark.sql("INSERT OVERWRITE TABLE " + param_table + " PARTITION(dt='" + day
                  + "') SELECT param_key, param_value FROM " + view)
    finally:
        spark.catalog.dropTempView(view)
    actual = read_current(spark, param_table, day)
    if actual != {"payload": payload, "dt": day}:
        raise RuntimeError("写表后读回不一致")


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_task(spark, exe_dt, param_table="online_opt_param", mature_end_dt=None,
             train_days=7, val_days=7, maturity_lag=6, cost_list=None,
             write=False, report_path=None):
    """exe_dt直接传调度业务日期；分区原样使用该日期，不再减一天。"""
    train_dates, val_dates = windows(exe_dt, mature_end_dt, train_days, val_days, maturity_lag)
    exe_dt = iso_date(exe_dt).isoformat()
    stored = read_current(spark, param_table, exe_dt)
    current = validate_params(stored["payload"] if stored else INITIAL_PARAMS)
    query = build_query(train_dates + val_dates)
    print("求解窗口:", train_dates, "选优窗口:", val_dates)
    print("现用参数:", current, "来源:", stored["dt"] if stored else "人工初始值")
    raw = spark.sql(query).toPandas()
    train, val = prepare_data(raw, train_dates, val_dates)
    result = evaluate_candidates(train, val, current, cost_list)
    run_id = exe_dt + "-" + uuid.uuid4().hex[:12]
    result.update(run_id=run_id, exe_dt=exe_dt, train_dates=train_dates, validation_dates=val_dates,
                  model_table=MODEL_TABLE, action_list=ACTION_LIST, source_param_dt=stored["dt"] if stored else None,
                  sql_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                  maturity_basis="upstream_complete_date" if mature_end_dt else "calendar_lag_requires_upstream_backfill",
                  publication="not_written")
    # 每次正常完成都产出BIZ_DATE_LINE分区；KEEP只保持参数，不跳过写表。
    result["write_reason"] = ("UPDATE" if result["status"] == "UPDATE" else
                              "INITIALIZE" if stored is None else "KEEP")
    if result["status"] != "UPDATE" and stored is not None:
        # 沿用参数及其真实求解来源，不能把每日续写冒充新训练结果。
        payload = dict(stored["payload"])
        payload.update(updated_dt=exe_dt, run_id=run_id,
                       comparison_train_dates=train_dates, validation_dates=val_dates)
    else:
        payload = build_payload(result["chosen"], exe_dt, train_dates, val_dates, run_id,
                                "rolling_search" if result["status"] == "UPDATE" else "manual_seed")
    result["proposed_payload"] = payload
    if write:
        write_params(spark, param_table, exe_dt, payload, stored)
        result["publication"] = "table_written_and_read_back_not_service_verified"
    else:
        result["publication"] = "dry_run"
    summary = [{"source": "incumbent", **current, **{
        k: result["old"]["validation"][k] for k in ("delta_cac", "delta_cvr_pv_ratio")}}]
    for candidate in result["candidates"]:
        summary.append(dict(source="candidate", C=candidate["C"], **{"lambda": candidate["lambda"]},
                            delta_cac=candidate["validation"]["delta_cac"],
                            delta_cvr_pv_ratio=candidate["validation"]["delta_cvr_pv_ratio"],
                            dominates_old=dominates(candidate["validation"], result["old"]["validation"])))
    print(pd.DataFrame(summary).to_string(index=False))
    print("决策:", result["status"], "参数:", result["chosen"], "写表:", result["publication"])
    if report_path:
        with open(report_path, "w", encoding="utf-8") as stream:
            json.dump(json_ready(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biz-date", default=os.environ.get("BIZ_DATE_LINE"),
                        help="平台业务日期，实际运行日减一天；直接用于dt分区")
    parser.add_argument("--param-table", default="online_opt_param")
    parser.add_argument("--mature-end-date")
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--maturity-lag", type=int, default=6,
                        help="相对业务日期回退天数；默认6，即实际运行日回退7天")
    parser.add_argument("--write", action="store_true", help="执行真实参数写表；默认只计算")
    parser.add_argument("--report-path")
    args = parser.parse_args()
    if not args.biz_date:
        parser.error("需要--biz-date或BIZ_DATE_LINE环境变量")
    iso_date(args.biz_date)  # 未替换的调度宏在启动Spark前报错。
    spark = globals().get("spark")
    if spark is None:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("msg_v4_auto_c_lambda").enableHiveSupport().getOrCreate()
    run_task(spark, args.biz_date, args.param_table, args.mature_end_date,
             args.train_days, args.val_days, args.maturity_lag,
             write=args.write, report_path=args.report_path)


if __name__ == "__main__":
    main()
