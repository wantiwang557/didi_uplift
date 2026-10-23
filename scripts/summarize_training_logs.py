#!/usr/bin/env python3
"""Extract per-epoch training metrics from ignored local log files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_PHASE_RE = re.compile(
    r"\[Epoch\s+(\d+)\s+(train_eval|train|val|valid|validation|test)\]",
    re.IGNORECASE,
)
EPOCH_START_RE = re.compile(r"={2,}\s*Epoch\s+(\d+)\s+", re.IGNORECASE)
TOTAL_LOSS_RE = re.compile(
    r"Epoch\s+(\d+)\s*,\s*Total Loss\s*:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)
BEST_UPDATE_RE = re.compile(r"Best model updated at epoch\s+(\d+)", re.IGNORECASE)
BEST_FINAL_RE = re.compile(r"Best epoch\s*:\s*(\d+)", re.IGNORECASE)
METRIC_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)\s*$",
    re.IGNORECASE,
)
METRIC_PREFIXES = (
    "auuc",
    "auuce",
    "qinie",
    "mean_",
    "average_auuc",
    "early_stop_",
    "final_train_",
    "exposure_",
    "loss1_mean",
    "outcome_loss_mean",
    "cost_loss_mean",
    "exposure_loss_mean",
)


@dataclass
class RunMetrics:
    number: int
    epochs: dict[int, dict[str, float]] = field(default_factory=dict)
    best_updates: list[int] = field(default_factory=list)
    explicit_best: int | None = None

    def epoch(self, number: int) -> dict[str, float]:
        return self.epochs.setdefault(number, {})

    def best_epoch(self) -> tuple[int | None, str]:
        if self.explicit_best is not None:
            return self.explicit_best, "explicit"
        if self.best_updates:
            return self.best_updates[-1], "last_best_update"
        return None, "not_found"


def parse_float(value: str) -> float:
    return float(value)


def is_metric_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.startswith(METRIC_PREFIXES)


def infer_phase(key: str, fallback: str | None) -> str | None:
    lowered = key.lower()
    for phase in ("train_eval", "train", "val", "valid", "validation", "test"):
        if f"_{phase}" in lowered or lowered.endswith(phase):
            return "val" if phase in {"valid", "validation"} else phase
    return fallback


def parse_log(path: Path) -> list[RunMetrics]:
    runs = [RunMetrics(number=1)]
    current_epoch: int | None = None
    current_phase: str | None = None
    epoch_one_starts = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).strip()
            if not line:
                continue

            start_match = EPOCH_START_RE.search(line)
            if start_match:
                epoch = int(start_match.group(1))
                if epoch == 1:
                    epoch_one_starts += 1
                    if epoch_one_starts > 1:
                        runs.append(RunMetrics(number=len(runs) + 1))
                current_epoch = epoch
                current_phase = None
                runs[-1].epoch(epoch)

            phase_match = EPOCH_PHASE_RE.search(line)
            if phase_match:
                current_epoch = int(phase_match.group(1))
                current_phase = phase_match.group(2).lower()
                if current_phase in {"valid", "validation"}:
                    current_phase = "val"
                runs[-1].epoch(current_epoch)

            loss_match = TOTAL_LOSS_RE.search(line)
            if loss_match:
                current_epoch = int(loss_match.group(1))
                runs[-1].epoch(current_epoch)["total_loss"] = parse_float(
                    loss_match.group(2)
                )

            update_match = BEST_UPDATE_RE.search(line)
            if update_match:
                epoch = int(update_match.group(1))
                runs[-1].best_updates.append(epoch)
                runs[-1].epoch(epoch)["best_model_updated"] = 1.0

            final_match = BEST_FINAL_RE.search(line)
            if final_match:
                runs[-1].explicit_best = int(final_match.group(1))

            metric_match = METRIC_RE.match(line)
            if metric_match and current_epoch is not None:
                key = metric_match.group(1)
                if is_metric_key(key):
                    value = parse_float(metric_match.group(2))
                    phase = infer_phase(key, current_phase)
                    stored_key = key
                    if phase and f"_{phase}" not in key.lower():
                        stored_key = f"{phase}_{key}"
                    runs[-1].epoch(current_epoch)[stored_key] = value

    return [run for run in runs if run.epochs or run.best_updates or run.explicit_best]


def first_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def canonical_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        "total_loss": metrics.get("total_loss"),
        "train_mean_auuc": first_metric(
            metrics,
            (
                "mean_auuc_train_eval",
                "mean_auuc_train",
                "train_mean_auuc_train",
                "train_average_auuc",
            ),
        ),
        "val_mean_auuc": first_metric(
            metrics,
            (
                "early_stop_mean_auuc",
                "mean_auuc_val",
                "val_mean_auuc_val",
                "val_average_auuc",
                "average_auuc",
            ),
        ),
        "test_mean_auuc": first_metric(
            metrics, ("mean_auuc_test", "test_mean_auuc_test", "test_average_auuc")
        ),
        "val_mean_qini": first_metric(
            metrics, ("mean_qinie_val", "val_mean_qinie_val")
        ),
        "test_mean_qini": first_metric(
            metrics, ("mean_qinie_test", "test_mean_qinie_test")
        ),
    }


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return str(value)
    return f"{value:.8g}"


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log_file",
        "run",
        "epoch",
        "is_best",
        "best_epoch_source",
        "total_loss",
        "train_mean_auuc",
        "val_mean_auuc",
        "test_mean_auuc",
        "val_mean_qini",
        "test_mean_qini",
        "metrics_json",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    parsed: list[tuple[Path, RunMetrics]], output: Path, root: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Epoch Summary",
        "",
        "Generated from local `*.log` files. Raw logs remain ignored by Git.",
        "A blank metric means that the source log did not report it for that epoch.",
        "",
    ]
    for path, run in parsed:
        relative = path.relative_to(root).as_posix()
        best_epoch, source = run.best_epoch()
        best_text = str(best_epoch) if best_epoch is not None else "not found"
        lines.extend(
            [
                f"## `{relative}` (run {run.number})",
                "",
                f"Best epoch: **{best_text}** (`{source}`)",
                "",
                "| Epoch | Best | Total loss | Train mean AUUC | Val mean AUUC | Test mean AUUC | Val mean QINI | Test mean QINI |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for epoch, metrics in sorted(run.epochs.items()):
            values = canonical_metrics(metrics)
            lines.append(
                "| {epoch} | {best} | {loss} | {train} | {val} | {test} | {vq} | {tq} |".format(
                    epoch=epoch,
                    best="yes" if epoch == best_epoch else "",
                    loss=format_number(values["total_loss"]),
                    train=format_number(values["train_mean_auuc"]),
                    val=format_number(values["val_mean_auuc"]),
                    test=format_number(values["test_mean_auuc"]),
                    vq=format_number(values["val_mean_qini"]),
                    tq=format_number(values["test_mean_qini"]),
                )
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--logs", type=Path, default=Path("demo"))
    parser.add_argument(
        "--csv", type=Path, default=Path("docs/training_results/epoch_metrics.csv")
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/training_results/training_summary.md"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    log_root = (root / args.logs).resolve()
    parsed: list[tuple[Path, RunMetrics]] = []
    rows: list[dict[str, object]] = []

    for path in sorted(log_root.rglob("*.log")):
        for run in parse_log(path):
            parsed.append((path, run))
            best_epoch, source = run.best_epoch()
            for epoch, metrics in sorted(run.epochs.items()):
                canonical = canonical_metrics(metrics)
                rows.append(
                    {
                        "log_file": path.relative_to(root).as_posix(),
                        "run": run.number,
                        "epoch": epoch,
                        "is_best": epoch == best_epoch,
                        "best_epoch_source": source,
                        **canonical,
                        "metrics_json": json.dumps(
                            metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )

    write_csv(rows, root / args.csv)
    write_markdown(parsed, root / args.markdown, root)
    print(
        f"Parsed {len(parsed)} log runs and {len(rows)} epoch rows from {log_root}."
    )


if __name__ == "__main__":
    main()
