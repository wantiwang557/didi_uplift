import argparse
import os
import random
import sys
import warnings
from datetime import datetime, timedelta

import evesdk
import numpy as np
import pandas as pd
import tensorflow as tf
from dmlite.common import DataFrameIO
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)


# 保持你当前用于测试的 0706 版本路径。注意：这个 HDFS 日期目录是否对应 0623-0706，
# 取决于你 extract_test.sh 当时写入的 exec_date。
TEST_TFR_PATH = "hdfs://difed/user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/16/part-*"

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
    "cost_7d",
    "fake_treatment_id",
    "flag",
    "label_dt",
]
PID_INDEX = TOKEN_COLUMNS.index("pid")
TREATMENT_INDEX = TOKEN_COLUMNS.index("fake_treatment_id")
LABEL_DT_INDEX = TOKEN_COLUMNS.index("label_dt")


def setup_runtime():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


class LiteFeTFRecords(object):
    def __init__(self, hdfs_pattern, start_dt, end_dt, label_size=-1, token_size=-1, dt_fmt="%Y%m%d"):
        self.hdfs_pattern = hdfs_pattern.strip()
        self.label_size = label_size
        self.token_size = token_size
        self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
        self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)

    def _get_hdfs_path_list(self):
        tfr_path_list = []
        days = (self.end_dt - self.start_dt).days
        for i in range(days + 1):
            pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
            tfr_path_list.extend(tf.io.gfile.glob(pattern))
        return tfr_path_list

    def get_dataset(
        self,
        epoch=1,
        batch_size=1,
        parallel=1,
        shuffle_size=0,
        drop_remainder=True,
        return_token=False,
        token_dtype=str,
    ):
        feature_dict = {
            "label": tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token": tf.io.FixedLenFeature([1], tf.string),
        }
        token_to_str, token_dtype_tf = {
            int: (False, tf.int64),
            float: (False, tf.float32),
            str: (True, tf.string),
        }[token_dtype]

        def _parse_single_example(example):
            feature_map = tf.io.parse_single_example(example, feature_dict)

            labels = tf.strings.split(feature_map["label"], ",").values
            labels = tf.strings.to_number(labels, tf.float32)
            label = tf.reshape(labels, [self.label_size])

            libsvm_kv = tf.strings.split(feature_map["feature"], ",").values
            libsvm_kv = tf.strings.split(libsvm_kv, ":").values
            feature = tf.strings.to_number(libsvm_kv, tf.float32)

            if not return_token:
                return feature, label

            token = tf.strings.split(feature_map["token"], ",").values
            if not token_to_str:
                token = tf.strings.to_number(token, token_dtype_tf)
            token = tf.reshape(token, [self.token_size])
            return feature, label, token

        tfr_path_list = self._get_hdfs_path_list()
        print(f"[TFRecord] Found {len(tfr_path_list)} files from: {self.hdfs_pattern}")
        if not tfr_path_list:
            raise ValueError(f"No TFRecord files found: {self.hdfs_pattern}")

        dataset = tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel,
        ).map(
            map_func=_parse_single_example,
            num_parallel_calls=parallel,
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        )

        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)

        return dataset.repeat(epoch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def load_feature_slot_indices(feature_slot_file):
    if not feature_slot_file:
        return None

    slots = []
    with open(feature_slot_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            slots.append(int(line))

    if not slots:
        raise ValueError(f"feature_slot_file is empty: {feature_slot_file}")

    if len(slots) != len(set(slots)):
        raise ValueError(f"feature_slot_file contains duplicated slots: {feature_slot_file}")

    print(f"[FEATURE SLOT] loaded {len(slots)} slots from {feature_slot_file}")
    return slots


def select_feature_slots_dataset(dataset, feature_slot_indices, parallel):
    if feature_slot_indices is None:
        return dataset

    slot_tensor = tf.constant(feature_slot_indices, dtype=tf.int32)

    def _select_feature_slots(feature, label, token):
        # 原始全量 feature 是 [batch, 609 * 2]，每个 slot 对应 [index, value]。
        feature_pairs = tf.reshape(feature, [tf.shape(feature)[0], -1, 2])
        selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
        selected_feature = tf.reshape(selected_pairs, [tf.shape(feature)[0], -1])
        return selected_feature, label, token

    return dataset.map(_select_feature_slots, num_parallel_calls=parallel)


def load_from_tfr(args):
    reader = LiteFeTFRecords(
        hdfs_pattern=args.test_tfr_path,
        start_dt=args.test_dt,
        end_dt=args.test_dt,
        token_size=args.token_size,
        label_size=args.label_size,
    )
    dataset = reader.get_dataset(
        epoch=1,
        batch_size=args.BATCH_SIZE,
        parallel=args.parallel,
        token_dtype=str,
        drop_remainder=False,
        return_token=True,
    )
    dataset = select_feature_slots_dataset(dataset, args.feature_slot_indices, args.parallel)
    return dataset


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def save_to_hive(args):
    print(f"Loading model from: {args.hdfs_model_path}")
    model = tf.keras.models.load_model(args.hdfs_model_path)

    test_dataset = load_from_tfr(args)

    pid, treatment, label_dt = [], [], []
    prediction_1, prediction_2, prediction_3, prediction_4 = [], [], [], []
    scores_cost_1, scores_cost_2, scores_cost_3, scores_cost_4 = [], [], [], []

    for feature, label, token in tqdm(test_dataset, desc="预测数据", leave=False):
        inputs = feature
        is_treat = tf.strings.to_number(token[:, TREATMENT_INDEX], tf.float32)

        # 模型输出顺序:
        # [control, control_stop_grad, uplift_1, uplift_2, uplift_3, uplift_4,
        #  cost_1, cost_2, cost_3, cost_4]
        _, _, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)

        # 与训练/评估脚本保持一致：treatment 2/3/4 的 delta 是基于 treatment 1 的残差还原。
        prediction_1.append(u_tau_1)
        prediction_2.append(u_tau_1 + u_tau_2)
        prediction_3.append(u_tau_1 + u_tau_3)
        prediction_4.append(u_tau_1 + u_tau_4)

        scores_cost_1.append(cost_1)
        scores_cost_2.append(cost_2)
        scores_cost_3.append(cost_3)
        scores_cost_4.append(cost_4)

        pid.append(token[:, PID_INDEX].numpy())
        treatment.append(is_treat.numpy())
        label_dt.append(token[:, LABEL_DT_INDEX].numpy())

    if len(pid) == 0:
        raise ValueError("测试集为空，没有预测任何数据。请检查 HDFS 路径下是否有对应 part 文件。")

    pid = np.concatenate(pid, axis=0).squeeze()
    treatment = np.concatenate(treatment, axis=0).squeeze()
    label_dt = np.concatenate(label_dt, axis=0).squeeze()

    predict_1 = tf.concat(prediction_1, axis=0).numpy().squeeze()
    predict_2 = tf.concat(prediction_2, axis=0).numpy().squeeze()
    predict_3 = tf.concat(prediction_3, axis=0).numpy().squeeze()
    predict_4 = tf.concat(prediction_4, axis=0).numpy().squeeze()

    cost_1_exp = tf.concat(scores_cost_1, axis=0).numpy().squeeze()
    cost_2_exp = tf.concat(scores_cost_2, axis=0).numpy().squeeze()
    cost_3_exp = tf.concat(scores_cost_3, axis=0).numpy().squeeze()
    cost_4_exp = tf.concat(scores_cost_4, axis=0).numpy().squeeze()

    pred_df = pd.DataFrame({
        "pid": [decode_if_bytes(x) for x in pid],
        "treatment": treatment,
        "label_dt": [decode_if_bytes(x) for x in label_dt],
        "uplift_1": predict_1,
        "uplift_2": predict_2,
        "uplift_3": predict_3,
        "uplift_4": predict_4,
        "cost_1": cost_1_exp,
        "cost_2": cost_2_exp,
        "cost_3": cost_3_exp,
        "cost_4": cost_4_exp,
    })

    print(pred_df.head())
    print("预测样本数:", len(pred_df))
    print("预测dt分区为:", args.startDate)
    print("写入Hive表:", args.output_table)

    dfio = DataFrameIO()
    dfio.write_table(
        df=pred_df,
        table_name=args.output_table,
        partition_val=args.startDate,
        partition_col="dt",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="EUEN feature-select inference and Hive writer")
    parser.add_argument("--BATCH_SIZE", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--label_size", type=int, default=1)
    parser.add_argument("--token_size", type=int, default=14)
    parser.add_argument("--test_dt", type=str, default="20250910")
    parser.add_argument("--startDate", type=str, default="20260706")

    parser.add_argument(
        "--test_tfr_path",
        type=str,
        default=TEST_TFR_PATH,
        help="HDFS TFRecord path. Keep 2026/07/16 for the current 0706 test version.",
    )
    parser.add_argument(
        "--hdfs_model_path",
        type=str,
        default="hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/",
    )
    parser.add_argument(
        "--feature_slot_file",
        type=str,
        default="./baseline_baseline.slots",
        help="Must match the trained model: baseline/no_sirius/replace_order.",
    )
    parser.add_argument(
        "--output_table",
        type=str,
        default="growstrategy.ut_msg_outcall_feature_select_v0_month_5_6_model_inference_di",
    )

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    evesdk.spark_util.init(
        username="",
        eve_api_caller="alpha_growth",
        eve_api_token="b846350027e783fb9de3553ef130f8fe",
    )
    evesdk.new_context()

    setup_runtime()
    args = parse_args()
    args.feature_slot_indices = load_feature_slot_indices(args.feature_slot_file)
    if args.feature_slot_indices is not None:
        print(f"[FEATURE SLOT] selected feature input length: {len(args.feature_slot_indices) * 2}")

    print("Parsed arguments:")
    print(args)
    save_to_hive(args)
