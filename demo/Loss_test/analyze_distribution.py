
'''
Version: analyze_label_cost_distribution

用途：
1. 分析 EUEN v1/v2/v3 训练或测试 TFRecord 中 label 和 cost_7d 的分布。
2. 输出整体、control、各 treatment 的 0 值比例、正值比例、均值、正值均值、分位数和 max。
3. 保存 summary CSV、sample 明细 CSV 和分布图，辅助判断 cost/label 更适合 MSE、ZILN 还是 Tweedie。

启动示例：

建议先分别看 v1/v2/v3 的 train 分布，因为 loss 是在 train 上优化的；
再看对应 test 分布，用来判断线上评估区间是否存在明显分布漂移。

# v1 train: 0511-0617
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/train_0511_0617/part-*" \
--data_name v1_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v1_train.log 2>&1 &

# v1 test: 0623-0706
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/test_0623_0706/part-*" \
--data_name v1_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v1_test.log 2>&1 &

# v2 train: 0511-0617
nohup python analyze_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/train_0511_0617/part-*" \
--data_name v2_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v2_train.log 2>&1 &

# v2 test: 0723-0806 excluding 0802
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/test_0723_0806/part-*" \
--data_name v2_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v2_test.log 2>&1 &

# v3 train: 0511-0617 + 0623-0722
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/train_0511_0722/part-*" \
--data_name v3_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v3_train.log 2>&1 &

# v3 test: 0723-0806
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/test_0723_0806/part-*" \
--data_name v3_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v3_test.log 2>&1 &

# 手动指定任意 TFRecord 路径
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/train_0511_0722/part-*" \
--data_name my_data \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_my_data.log 2>&1 &
'''



import argparse
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


TOKEN_COLUMNS = [
    "fin_order_wxmini_28d",
    "group_name_eve",
    "pid",
    "oneid",
    "send_cnt",
    "send_success_cnt",
    "send_cost",
    "remind_cost",
    "coupon_cost",
    "dau_cnt",
    "exp_label",
    "cost_7d",
    "fake_treatment_id",
    "flag",
    "label_dt",
]

COST_INDEX = 11
TREATMENT_INDEX = 12
DEFAULT_QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


class RunningStats:
    def __init__(self):
        self.n = 0
        self.zero_n = 0
        self.positive_n = 0
        self.sum_value = 0.0
        self.sum_positive = 0.0
        self.min_value = np.inf
        self.max_value = -np.inf

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        positive_mask = values > 0
        self.n += values.size
        self.zero_n += int(np.sum(values == 0))
        self.positive_n += int(np.sum(positive_mask))
        self.sum_value += float(np.sum(values))
        self.sum_positive += float(np.sum(values[positive_mask]))
        self.min_value = min(self.min_value, float(np.min(values)))
        self.max_value = max(self.max_value, float(np.max(values)))

    def to_dict(self):
        mean = self.sum_value / self.n if self.n else np.nan
        positive_mean = self.sum_positive / self.positive_n if self.positive_n else np.nan
        return {
            "n": self.n,
            "zero_n": self.zero_n,
            "zero_rate": self.zero_n / self.n if self.n else np.nan,
            "positive_n": self.positive_n,
            "positive_rate": self.positive_n / self.n if self.n else np.nan,
            "mean": mean,
            "positive_mean": positive_mean,
            "min": self.min_value if self.n else np.nan,
            "max": self.max_value if self.n else np.nan,
        }


class ReservoirSampler:
    def __init__(self, limit, seed):
        self.limit = int(limit)
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.sample = []

    def update(self, labels, costs, treatments):
        for label, cost, treatment in zip(labels, costs, treatments):
            row = (float(label), float(cost), int(treatment))
            self.n_seen += 1
            if self.limit <= 0:
                continue
            if len(self.sample) < self.limit:
                self.sample.append(row)
                continue
            j = self.rng.integers(0, self.n_seen)
            if j < self.limit:
                self.sample[j] = row

    def to_frame(self):
        return pd.DataFrame(self.sample, columns=["label", "cost_7d", "treatment"])


def parse_batch(serialized_batch, token_size):
    feature_dict = {
        "label": tf.io.FixedLenFeature([1], tf.string),
        "feature": tf.io.FixedLenFeature([1], tf.string),
        "token": tf.io.FixedLenFeature([1], tf.string),
    }
    feature_map = tf.io.parse_example(serialized_batch, feature_dict)

    labels = tf.strings.split(feature_map["label"][:, 0], ",")
    labels = tf.strings.to_number(labels, tf.float32).to_tensor()
    labels = tf.reshape(labels, [-1])

    tokens = tf.strings.split(feature_map["token"][:, 0], ",")
    tokens = tokens.to_tensor(default_value="")
    tokens = tf.reshape(tokens, [-1, token_size])

    costs = tf.strings.to_number(tokens[:, COST_INDEX], tf.float32)
    treatments = tf.strings.to_number(tokens[:, TREATMENT_INDEX], tf.int32)
    return labels, costs, treatments


def build_dataset(tfr_path, batch_size, parallel, token_size):
    files = tf.io.gfile.glob(tfr_path)
    if not files:
        raise ValueError(f"No TFRecord files found: {tfr_path}")
    print(f"Found files: {len(files)}")
    print(f"First file: {files[0]}")
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=parallel)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.map(lambda x: parse_batch(x, token_size), num_parallel_calls=parallel)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def group_name(treatment):
    if treatment == -1:
        return "overall"
    if treatment == 0:
        return "control_t0"
    return f"treatment_{treatment}"


def update_stats(stats, labels, costs, treatments):
    stats[("overall", "label")].update(labels)
    stats[("overall", "cost_7d")].update(costs)
    for treatment in sorted(np.unique(treatments)):
        mask = treatments == treatment
        name = group_name(int(treatment))
        stats[(name, "label")].update(labels[mask])
        stats[(name, "cost_7d")].update(costs[mask])


def add_sample_quantiles(summary_df, sample_df):
    if sample_df.empty:
        return summary_df
    rows = []
    for _, row in summary_df.iterrows():
        group = row["group"]
        target = row["target"]
        if group == "overall":
            values = sample_df[target]
        elif group == "control_t0":
            values = sample_df.loc[sample_df["treatment"] == 0, target]
        elif group.startswith("treatment_"):
            treatment = int(group.split("_")[-1])
            values = sample_df.loc[sample_df["treatment"] == treatment, target]
        else:
            values = pd.Series(dtype=float)

        values = values[np.isfinite(values)]
        positive_values = values[values > 0]
        quantile_values = {
            f"sample_p{str(q).replace('.', '')}": values.quantile(q) if len(values) else np.nan
            for q in DEFAULT_QUANTILES
        }
        positive_quantile_values = {
            f"sample_pos_p{str(q).replace('.', '')}": positive_values.quantile(q) if len(positive_values) else np.nan
            for q in DEFAULT_QUANTILES
        }
        rows.append({**row.to_dict(), **quantile_values, **positive_quantile_values})
    return pd.DataFrame(rows)


def build_summary(stats, sample_df):
    rows = []
    group_order = ["overall", "control_t0", "treatment_1", "treatment_2", "treatment_3", "treatment_4"]
    for group in group_order:
        for target in ["label", "cost_7d"]:
            if (group, target) not in stats:
                continue
            rows.append({
                "group": group,
                "target": target,
                **stats[(group, target)].to_dict(),
            })
    summary_df = pd.DataFrame(rows)
    return add_sample_quantiles(summary_df, sample_df)


def plot_distribution(sample_df, target, out_dir, data_name):
    if sample_df.empty:
        return

    values = sample_df[target].dropna()
    if values.empty:
        return

    plt.figure(figsize=(10, 6))
    upper = values.quantile(0.995)
    clipped = values.clip(upper=upper)
    plt.hist(clipped, bins=100)
    plt.title(f"{data_name} {target} raw clipped at p99.5")
    plt.xlabel(target)
    plt.ylabel("sample_count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_raw_hist.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(values), bins=100)
    plt.title(f"{data_name} log1p({target})")
    plt.xlabel(f"log1p({target})")
    plt.ylabel("sample_count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_log1p_hist.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for treatment in sorted(sample_df["treatment"].dropna().unique()):
        group_values = sample_df.loc[sample_df["treatment"] == treatment, target].dropna()
        if len(group_values) == 0:
            continue
        plt.hist(np.log1p(group_values), bins=80, alpha=0.35, label=f"t{int(treatment)}")
    plt.title(f"{data_name} log1p({target}) by treatment")
    plt.xlabel(f"log1p({target})")
    plt.ylabel("sample_count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_log1p_by_treatment.png"))
    plt.close()


def print_loss_hint(summary_df):
    pivot = summary_df.pivot(index="group", columns="target", values="zero_rate")
    print("\nLoss choice hints:")
    for target in ["label", "cost_7d"]:
        zero_rate = float(pivot.loc["overall", target]) if target in pivot.columns else np.nan
        target_df = summary_df[(summary_df["group"] == "overall") & (summary_df["target"] == target)]
        pos_p99 = float(target_df["sample_pos_p099"].iloc[0]) if len(target_df) else np.nan
        pos_p50 = float(target_df["sample_pos_p05"].iloc[0]) if len(target_df) else np.nan
        tail_ratio = pos_p99 / pos_p50 if pos_p50 and pos_p50 > 0 else np.nan
        print(f"- {target}: zero_rate={zero_rate:.4f}, positive_p99/p50={tail_ratio:.2f}")
        if zero_rate > 0.5 and tail_ratio > 5:
            print("  Suggestion: strong zero inflation + long positive tail; ZILN/Tweedie are worth testing.")
        elif zero_rate > 0.2:
            print("  Suggestion: has visible zeros; Tweedie or ZILN may help, compare with MSE.")
        elif tail_ratio > 5:
            print("  Suggestion: long positive tail; log-space loss or Tweedie may be more stable than MSE.")
        else:
            print("  Suggestion: MSE may be a reasonable baseline; distribution loss may not add much.")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EUEN label and cost_7d distribution")
    parser.add_argument("--tfr_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--token_size", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--sample_limit", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tfr_path = args.tfr_path
    data_name = args.data_name or os.path.basename(os.path.dirname(tfr_path.rstrip("/")))
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "distribution_outputs",
        f"{data_name}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("Data name:", data_name)
    print("TFRecord path:", tfr_path)
    print("Output dir:", out_dir)
    print("Token columns:")
    print(f"  cost_7d index={COST_INDEX}, treatment index={TREATMENT_INDEX}, token_size={args.token_size}")

    dataset = build_dataset(tfr_path, args.batch_size, args.parallel, args.token_size)
    stats = defaultdict(RunningStats)
    sampler = ReservoirSampler(args.sample_limit, args.seed)

    batch_count = 0
    total_count = 0
    start_time = time.time()
    for labels, costs, treatments in dataset:
        label_np = labels.numpy().reshape(-1)
        cost_np = costs.numpy().reshape(-1)
        treatment_np = treatments.numpy().reshape(-1)
        update_stats(stats, label_np, cost_np, treatment_np)
        sampler.update(label_np, cost_np, treatment_np)

        batch_count += 1
        total_count += len(label_np)
        if batch_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed batches={batch_count}, rows={total_count}, speed={total_count / max(elapsed, 1):.0f} rows/sec")

    sample_df = sampler.to_frame()
    summary_df = build_summary(stats, sample_df)

    summary_path = os.path.join(out_dir, "label_cost_distribution_summary.csv")
    sample_path = os.path.join(out_dir, "label_cost_distribution_sample.csv")
    summary_df.to_csv(summary_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    for target in ["label", "cost_7d"]:
        plot_distribution(sample_df, target, out_dir, data_name)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)
    print("\nSummary:")
    print(summary_df)
    print_loss_hint(summary_df)
    print("\nSaved:")
    print("  summary:", summary_path)
    print("  sample:", sample_path)
    print("  plots:", out_dir)


if __name__ == "__main__":
    main()
