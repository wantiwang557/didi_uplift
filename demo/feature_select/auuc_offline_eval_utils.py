import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    import metrics  # causalml/backflow style metrics, if already on PYTHONPATH
except Exception:
    metrics = None


@dataclass
class TreatmentEvalSpec:
    treatment: int
    control: int
    uplift_col: str
    cost_col: Optional[str] = None
    name: str = ""


DEFAULT_EUEN_4T_SPECS = [
    TreatmentEvalSpec(1, 0, "predict_delta_cvr_1", "predict_cost_7d_1", "treatment_1"),
    TreatmentEvalSpec(2, 0, "predict_delta_cvr_2", "predict_cost_7d_2", "treatment_2"),
    TreatmentEvalSpec(3, 0, "predict_delta_cvr_3", "predict_cost_7d_3", "treatment_3"),
    TreatmentEvalSpec(4, 0, "predict_delta_cvr_4", "predict_cost_7d_4", "treatment_4"),
]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def ranking_auc(y_labels, y_scores) -> float:
    """Pairwise ranking AUC for numeric/multi-level labels."""
    labels = np.asarray(y_labels).reshape(-1)
    scores = np.asarray(y_scores).reshape(-1)
    mask = ~(pd.isna(labels) | pd.isna(scores))
    labels = labels[mask]
    scores = scores[mask]
    n = len(labels)
    if n <= 1:
        return np.nan

    total = 0
    correct = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            total += 1
            label_order = labels[i] > labels[j]
            score_order = scores[i] > scores[j]
            if score_order == label_order:
                correct += 1.0
            elif scores[i] == scores[j]:
                correct += 0.5
    return correct / total if total else np.nan


def draw_auuc_like_previous(
    df: pd.DataFrame,
    spec: TreatmentEvalSpec,
    label_col: str = "label",
    treatment_col: str = "real_treatment",
    show_curve: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute AUUC with the same causalml/backflow style used in euen_4t_feature.py."""
    subset = df[df[treatment_col].isin([spec.treatment, spec.control])].copy()
    subset = subset.dropna(subset=[spec.uplift_col, label_col, treatment_col])
    if subset.empty:
        return {"auuc": np.nan, "random": np.nan, "n_total": 0, "n_treatment": 0, "n_control": 0}

    subset["exp_group"] = (subset[treatment_col] != spec.control).astype(int)
    subset["label"] = subset[label_col]
    n_treatment = int((subset[treatment_col] == spec.treatment).sum())
    n_control = int((subset[treatment_col] == spec.control).sum())

    if n_treatment == 0 or n_control == 0:
        return {
            "auuc": np.nan,
            "random": np.nan,
            "n_total": len(subset),
            "n_treatment": n_treatment,
            "n_control": n_control,
        }

    if metrics is not None:
        lift = metrics.get_cumlift(
            subset[["exp_group", "label", spec.uplift_col]],
            outcome_col="label",
            treatment_col="exp_group",
            random_seed=4,
        )
        gain = lift.mul(lift.index.values, axis=0)
        last = gain.iloc[-1, :].replace(0, np.nan).abs()
        gain = gain.div(last, axis=1)
        res = gain.sum() / gain.shape[0]

        if show_curve or save_path:
            ax = gain.plot(title=f"{spec.name or spec.treatment} AUUC")
            ax.set_xlabel("Ite ranks")
            ax.set_ylabel("AUUC")
            if save_path:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            if show_curve:
                plt.show()
            else:
                plt.close()

        random_val = float(res["Random"]) if "Random" in res else np.nan
        auuc_val = float(res[spec.uplift_col]) if spec.uplift_col in res else np.nan
    else:
        curve = cumulative_uplift_curve(subset, spec, label_col="label", treatment_col=treatment_col)
        auuc_val = float(np.nanmean(curve["actual_uplift"]))
        random_val = 0.0

    return {
        "auuc": auuc_val,
        "random": random_val,
        "n_total": len(subset),
        "n_treatment": n_treatment,
        "n_control": n_control,
    }


def cumulative_uplift_curve(
    df: pd.DataFrame,
    spec: TreatmentEvalSpec,
    label_col: str = "label",
    treatment_col: str = "real_treatment",
) -> pd.DataFrame:
    subset = df[df[treatment_col].isin([spec.treatment, spec.control])].copy()
    subset = subset.dropna(subset=[spec.uplift_col, label_col, treatment_col])
    subset = subset.sort_values(spec.uplift_col, ascending=False).reset_index(drop=True)
    subset["is_treat"] = (subset[treatment_col] == spec.treatment).astype(int)
    subset["is_control"] = (subset[treatment_col] == spec.control).astype(int)
    subset["treat_y"] = subset[label_col] * subset["is_treat"]
    subset["control_y"] = subset[label_col] * subset["is_control"]

    cum_treat = subset["is_treat"].cumsum()
    cum_control = subset["is_control"].cumsum()
    treat_mean = subset["treat_y"].cumsum() / cum_treat.replace(0, np.nan)
    control_mean = subset["control_y"].cumsum() / cum_control.replace(0, np.nan)
    actual_uplift = treat_mean - control_mean
    ratio = cum_treat / (cum_treat + cum_control).replace(0, np.nan)

    return pd.DataFrame(
        {
            "rank": np.arange(1, len(subset) + 1),
            "actual_uplift": actual_uplift,
            "treatment_control_ratio": ratio,
        }
    )


def cost_auc_for_spec(
    df: pd.DataFrame,
    spec: TreatmentEvalSpec,
    cost_label_col: str = "label_cost_7d",
    treatment_col: str = "real_treatment",
) -> float:
    if not spec.cost_col or spec.cost_col not in df.columns or cost_label_col not in df.columns:
        return np.nan
    part = df[df[treatment_col] == spec.treatment].copy()
    part = part.dropna(subset=[cost_label_col, spec.cost_col])
    if len(part) <= 1:
        return np.nan
    return ranking_auc(part[cost_label_col], part[spec.cost_col])


def pcoc_and_cal_n_for_spec(
    df: pd.DataFrame,
    spec: TreatmentEvalSpec,
    label_col: str = "label",
    treatment_col: str = "real_treatment",
    n_bins: int = 10,
) -> Dict[str, float]:
    subset = df[df[treatment_col].isin([spec.treatment, spec.control])].copy()
    subset = subset.dropna(subset=[spec.uplift_col, label_col, treatment_col])
    treat = subset[subset[treatment_col] == spec.treatment]
    control = subset[subset[treatment_col] == spec.control]
    if treat.empty or control.empty:
        return {
            "pcoc": np.nan,
            "cal_n": np.nan,
            "predicted_uplift": np.nan,
            "actual_uplift": np.nan,
            "actual_treatment_cvr": np.nan,
            "actual_control_cvr": np.nan,
        }

    predicted_uplift = subset[spec.uplift_col].mean()
    actual_treatment_cvr = treat[label_col].mean()
    actual_control_cvr = control[label_col].mean()
    actual_uplift = actual_treatment_cvr - actual_control_cvr
    pcoc = predicted_uplift / actual_uplift if actual_uplift != 0 else np.nan

    decile = uplift_decile_summary(
        subset,
        spec,
        label_col=label_col,
        treatment_col=treatment_col,
        n_bins=n_bins,
    )
    errors = []
    for _, row in decile.dropna(subset=["predict_uplift", "actual_uplift"]).iterrows():
        if row["actual_uplift"] == 0:
            continue
        bucket_pcoc = row["predict_uplift"] / row["actual_uplift"]
        if bucket_pcoc > 1:
            errors.append(bucket_pcoc - 1)
        elif bucket_pcoc > 0:
            errors.append(1 / bucket_pcoc - 1)
    cal_n = float(np.sqrt(np.mean(np.square(errors)))) if errors else np.nan

    return {
        "pcoc": float(pcoc) if not pd.isna(pcoc) else np.nan,
        "cal_n": cal_n,
        "predicted_uplift": float(predicted_uplift),
        "actual_uplift": float(actual_uplift),
        "actual_treatment_cvr": float(actual_treatment_cvr),
        "actual_control_cvr": float(actual_control_cvr),
    }


def uplift_decile_summary(
    df: pd.DataFrame,
    spec: TreatmentEvalSpec,
    label_col: str = "label",
    treatment_col: str = "real_treatment",
    n_bins: int = 10,
) -> pd.DataFrame:
    subset = df[df[treatment_col].isin([spec.treatment, spec.control])].copy()
    subset = subset.dropna(subset=[spec.uplift_col, label_col, treatment_col])
    if subset.empty:
        return pd.DataFrame()

    subset = subset.sort_values(spec.uplift_col, ascending=False).reset_index(drop=True)
    subset["decile"] = pd.qcut(subset.index, q=n_bins, labels=False, duplicates="drop")

    rows = []
    for decile in sorted(subset["decile"].dropna().unique()):
        bucket = subset[subset["decile"] == decile]
        treat = bucket[bucket[treatment_col] == spec.treatment]
        control = bucket[bucket[treatment_col] == spec.control]
        predict_uplift = bucket[spec.uplift_col].mean()
        actual_uplift = np.nan
        if len(treat) > 0 and len(control) > 0:
            actual_uplift = treat[label_col].mean() - control[label_col].mean()
        rows.append(
            {
                "decile": int(decile) + 1,
                "rank_bucket": int(decile) + 1,
                "predict_uplift": predict_uplift,
                "actual_uplift": actual_uplift,
                "n_total": len(bucket),
                "n_treatment": len(treat),
                "n_control": len(control),
                "treatment_control_ratio": len(treat) / len(bucket) if len(bucket) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_uplift_decile(
    decile_df: pd.DataFrame,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    if decile_df.empty:
        return
    x = np.arange(len(decile_df))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, decile_df["predict_uplift"], width, label="predict", color="#ff6b73")
    plt.bar(x + width / 2, decile_df["actual_uplift"], width, label="actual", color="#6c63ff")
    plt.axhline(0, color="#888888", linewidth=1)
    plt.xticks(x, [str(v) for v in decile_df["rank_bucket"]], rotation=30)
    plt.xlabel("Ite ranks")
    plt.ylabel("Uplift")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_one_scheme(
    df: pd.DataFrame,
    scheme_name: str,
    specs: Iterable[TreatmentEvalSpec] = DEFAULT_EUEN_4T_SPECS,
    label_col: str = "label",
    cost_label_col: str = "label_cost_7d",
    treatment_col: str = "real_treatment",
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    summary_rows = []
    decile_frames = []
    curve_frames = []

    if output_dir:
        ensure_dir(output_dir)

    for spec in specs:
        spec_name = spec.name or f"treatment_{spec.treatment}"
        auuc_stats = draw_auuc_like_previous(
            df,
            spec,
            label_col=label_col,
            treatment_col=treatment_col,
            show_curve=False,
            save_path=(
                os.path.join(output_dir, f"{scheme_name}_{spec_name}_auuc.png")
                if output_dir and save_plots
                else None
            ),
        )
        cost_auc = cost_auc_for_spec(df, spec, cost_label_col=cost_label_col, treatment_col=treatment_col)
        calib = pcoc_and_cal_n_for_spec(df, spec, label_col=label_col, treatment_col=treatment_col)

        summary_rows.append(
            {
                "scheme_name": scheme_name,
                "treatment": spec.treatment,
                "control": spec.control,
                "treatment_name": spec_name,
                "uplift_col": spec.uplift_col,
                "cost_col": spec.cost_col,
                "auuc": auuc_stats["auuc"],
                "random_auuc": auuc_stats["random"],
                "cost_auc": cost_auc,
                **calib,
                "n_total": auuc_stats["n_total"],
                "n_treatment": auuc_stats["n_treatment"],
                "n_control": auuc_stats["n_control"],
            }
        )

        decile = uplift_decile_summary(df, spec, label_col=label_col, treatment_col=treatment_col)
        if not decile.empty:
            decile.insert(0, "scheme_name", scheme_name)
            decile.insert(1, "treatment", spec.treatment)
            decile.insert(2, "control", spec.control)
            decile.insert(3, "treatment_name", spec_name)
            decile.insert(4, "uplift_col", spec.uplift_col)
            decile_frames.append(decile)
            if output_dir and save_plots:
                plot_uplift_decile(
                    decile,
                    title=f"{scheme_name} {spec_name}",
                    save_path=os.path.join(output_dir, f"{scheme_name}_{spec_name}_decile.png"),
                    show=show_plots,
                )

        curve = cumulative_uplift_curve(df, spec, label_col=label_col, treatment_col=treatment_col)
        if not curve.empty:
            curve.insert(0, "scheme_name", scheme_name)
            curve.insert(1, "treatment", spec.treatment)
            curve.insert(2, "control", spec.control)
            curve.insert(3, "treatment_name", spec_name)
            curve.insert(4, "uplift_col", spec.uplift_col)
            curve_frames.append(curve)

    summary = pd.DataFrame(summary_rows)
    deciles = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()

    if output_dir:
        summary.to_csv(os.path.join(output_dir, f"{scheme_name}_offline_eval_summary.csv"), index=False)
        if not deciles.empty:
            deciles.to_csv(os.path.join(output_dir, f"{scheme_name}_uplift_deciles.csv"), index=False)
        if not curves.empty:
            curves.to_csv(os.path.join(output_dir, f"{scheme_name}_cumulative_curves.csv"), index=False)

    return {"summary": summary, "deciles": deciles, "curves": curves}


def evaluate_three_schemes(
    scheme_dfs: Dict[str, pd.DataFrame],
    specs: Iterable[TreatmentEvalSpec] = DEFAULT_EUEN_4T_SPECS,
    label_col: str = "label",
    cost_label_col: str = "label_cost_7d",
    treatment_col: str = "real_treatment",
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    all_summary = []
    all_deciles = []
    all_curves = []
    if output_dir:
        ensure_dir(output_dir)

    for scheme_name, df in scheme_dfs.items():
        scheme_dir = os.path.join(output_dir, scheme_name) if output_dir else None
        result = evaluate_one_scheme(
            df,
            scheme_name,
            specs=specs,
            label_col=label_col,
            cost_label_col=cost_label_col,
            treatment_col=treatment_col,
            output_dir=scheme_dir,
            save_plots=save_plots,
            show_plots=show_plots,
        )
        all_summary.append(result["summary"])
        if not result["deciles"].empty:
            all_deciles.append(result["deciles"])
        if not result["curves"].empty:
            all_curves.append(result["curves"])

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    deciles = pd.concat(all_deciles, ignore_index=True) if all_deciles else pd.DataFrame()
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()

    if output_dir:
        summary.to_csv(os.path.join(output_dir, "offline_eval_summary_3x4.csv"), index=False)
        if not deciles.empty:
            deciles.to_csv(os.path.join(output_dir, "uplift_deciles_3x4.csv"), index=False)
        if not curves.empty:
            curves.to_csv(os.path.join(output_dir, "cumulative_curves_3x4.csv"), index=False)

    return {"summary": summary, "deciles": deciles, "curves": curves}


def make_left_absolute_specs() -> List[TreatmentEvalSpec]:
    return [
        TreatmentEvalSpec(1, 0, "left63", "cost_1", "left63_vs_control"),
        TreatmentEvalSpec(2, 0, "left853", "cost_2", "left853_vs_control"),
        TreatmentEvalSpec(3, 0, "left65", "cost_3", "left65_vs_control"),
    ]


def make_left_residual_specs() -> List[TreatmentEvalSpec]:
    return [
        TreatmentEvalSpec(2, 1, "left853_minus_left63", "cost_2", "left853_vs_left63"),
        TreatmentEvalSpec(3, 1, "left65_minus_left63", "cost_3", "left65_vs_left63"),
    ]

