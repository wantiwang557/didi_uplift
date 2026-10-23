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


# EVE 直接运行时，改这里即可切换 v2 / v3。
DEFAULT_MODEL_VERSION = "v2"

MODEL_CONFIGS = {
    "v2": {
        "test_tfr_path": "hdfs://difed/user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/part-*",
        "hdfs_model_path": "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/euen_test_feature_time/v2_train_0511_0617",
        "output_table": "growstrategy.ut_msg_outcall_euen_v2_0511_0617_inference_di",
        "version": "euen_v2_train_0511_0617_test_0723_0806",
    },
    "v3": {
        "test_tfr_path": "hdfs://difed/user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/part-*",
        "hdfs_model_path": "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/euen_test_feature_time/v3_train_0511_0722",
        "output_table": "growstrategy.ut_msg_outcall_euen_v3_0511_0722_inference_di",
        "version": "euen_v3_train_0511_0722_test_0723_0806",
    },
}

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
PID_INDEX = TOKEN_COLUMNS.index("pid")
EXP_INDEX = TOKEN_COLUMNS.index("exp_label")
TREATMENT_INDEX = TOKEN_COLUMNS.index("fake_treatment_id")
LABEL_DT_INDEX = TOKEN_COLUMNS.index("label_dt")


class IndexEmbedding(tf.keras.layers.Embedding):
    def __init__(
        self,
        dense_fea_dim,
        embedding_dim,
        sparse_fea_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        **kwargs
    ):
        super(IndexEmbedding, self).__init__(
            input_dim=dense_fea_dim,
            output_dim=embedding_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs
        )
        self.sparse_fea_dim = sparse_fea_dim
        self.out_dense_dim = sparse_fea_dim * embedding_dim

    def build(self, input_shape=None):
        name = "index_embedding_{}_{}".format(self.input_dim, self.output_dim)
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False,
            name=name,
        )
        self.built = True

    def call(self, feature, **kwargs):
        embedding = tf.nn.embedding_lookup(self.embeddings, feature)
        return tf.reshape(embedding, [-1, self.out_dense_dim])


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

    def get_dataset(self, epoch=1, batch_size=1, parallel=1, drop_remainder=True, return_token=False, token_dtype=str):
        feature_dict = {
            "label": tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token": tf.io.FixedLenFeature([1], tf.string),
        }
        token2str, token_tf_dtype = {
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

            if return_token:
                token = tf.strings.split(feature_map["token"], ",").values
                if not token2str:
                    token = tf.strings.to_number(token, token_tf_dtype)
                token = tf.reshape(token, [self.token_size])
                return feature, label, token

            return feature, label

        tfr_path_list = self._get_hdfs_path_list()
        print("TFRecord files count:", len(tfr_path_list))
        if len(tfr_path_list) == 0:
            raise ValueError("No TFRecord part files matched. Please check test_tfr_path.")

        return tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel,
        ).map(
            map_func=_parse_single_example,
            num_parallel_calls=parallel,
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        ).repeat(epoch).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )


def setup_runtime(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_from_tfr(args):
    reader = LiteFeTFRecords(
        hdfs_pattern=args.test_tfr_path,
        start_dt=args.test_dt,
        end_dt=args.test_dt,
        token_size=args.token_size,
        label_size=1,
    )
    return reader.get_dataset(
        epoch=1,
        batch_size=args.BATCH_SIZE,
        parallel=args.parallel,
        token_dtype=str,
        drop_remainder=False,
        return_token=True,
    )


def save_to_hive(args):
    print(f"Loading model from: {args.hdfs_model_path}")
    model = tf.keras.models.load_model(args.hdfs_model_path, compile=False)

    test_dataset = load_from_tfr(args)

    pid, treatment, label_dt, exp_label = [], [], [], []
    prediction_1, prediction_2, prediction_3, prediction_4 = [], [], [], []
    scores_cost_1, scores_cost_2, scores_cost_3, scores_cost_4 = [], [], [], []

    first_batch = True
    for feature, _, token in tqdm(test_dataset, desc="predict", leave=False):
        if first_batch:
            raw_token = [decode_if_bytes(x) for x in token[0].numpy()]
            print("First token sample:")
            for idx, val in enumerate(raw_token):
                print(idx, TOKEN_COLUMNS[idx] if idx < len(TOKEN_COLUMNS) else "UNKNOWN", val)
            first_batch = False

        inputs = feature
        real_treatment = tf.strings.to_number(token[:, TREATMENT_INDEX], tf.float32)
        batch_exp_label = tf.strings.to_number(token[:, EXP_INDEX], tf.float32)

        # EUEN outputs:
        # [control, control_logit_fix, uplift_1, uplift_2, uplift_3, uplift_4,
        #  cost_1, cost_2, cost_3, cost_4]
        _, _, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)

        # Treatment 1 is the base uplift. Treatment 2/3/4 are residuals on T1.
        prediction_1.append(u_tau_1)
        prediction_2.append(u_tau_1 + u_tau_2)
        prediction_3.append(u_tau_1 + u_tau_3)
        prediction_4.append(u_tau_1 + u_tau_4)

        scores_cost_1.append(cost_1)
        scores_cost_2.append(cost_2)
        scores_cost_3.append(cost_3)
        scores_cost_4.append(cost_4)

        pid.append(token[:, PID_INDEX].numpy())
        treatment.append(real_treatment.numpy())
        label_dt.append(token[:, LABEL_DT_INDEX].numpy())
        exp_label.append(batch_exp_label.numpy())

    if len(pid) == 0:
        raise ValueError("Empty prediction result. Please check input TFRecord path.")

    pred_df = pd.DataFrame({
        "pid": [decode_if_bytes(x) for x in np.concatenate(pid, axis=0).squeeze()],
        "treatment": np.concatenate(treatment, axis=0).squeeze().astype(int),
        "label_dt": [decode_if_bytes(x) for x in np.concatenate(label_dt, axis=0).squeeze()],
        "exp_label": np.concatenate(exp_label, axis=0).squeeze().astype(int),
        "uplift_1": tf.concat(prediction_1, axis=0).numpy().squeeze(),
        "uplift_2": tf.concat(prediction_2, axis=0).numpy().squeeze(),
        "uplift_3": tf.concat(prediction_3, axis=0).numpy().squeeze(),
        "uplift_4": tf.concat(prediction_4, axis=0).numpy().squeeze(),
        "cost_1": tf.concat(scores_cost_1, axis=0).numpy().squeeze(),
        "cost_2": tf.concat(scores_cost_2, axis=0).numpy().squeeze(),
        "cost_3": tf.concat(scores_cost_3, axis=0).numpy().squeeze(),
        "cost_4": tf.concat(scores_cost_4, axis=0).numpy().squeeze(),
    })

    print(pred_df.head())
    print("prediction rows:", len(pred_df))
    print("output table:", args.output_table)
    print("output partition dt:", args.startDate)

    dfio = DataFrameIO()
    dfio.write_table(
        df=pred_df,
        table_name=args.output_table,
        partition_val=args.startDate,
        partition_col="dt",
    )


def parse_args():
    config = MODEL_CONFIGS[DEFAULT_MODEL_VERSION]
    parser = argparse.ArgumentParser(description="EUEN v2/v3 inference and Hive writer")
    parser.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION, choices=["v2", "v3"])
    parser.add_argument("--test_tfr_path", type=str, default=config["test_tfr_path"])
    parser.add_argument("--hdfs_model_path", type=str, default=config["hdfs_model_path"])
    parser.add_argument("--output_table", type=str, default=config["output_table"])
    parser.add_argument("--version", type=str, default=config["version"])
    parser.add_argument("--test_dt", type=str, default="20260813")
    parser.add_argument("--startDate", type=str, default="20260813")
    parser.add_argument("--token_size", type=int, default=15)
    parser.add_argument("--BATCH_SIZE", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")

    args, _ = parser.parse_known_args()

    # If the caller passes --model_version v3 in a command/task arg, refresh defaults.
    cfg = MODEL_CONFIGS[args.model_version]
    if args.test_tfr_path == config["test_tfr_path"]:
        args.test_tfr_path = cfg["test_tfr_path"]
    if args.hdfs_model_path == config["hdfs_model_path"]:
        args.hdfs_model_path = cfg["hdfs_model_path"]
    if args.output_table == config["output_table"]:
        args.output_table = cfg["output_table"]
    return args


if __name__ == "__main__":
    evesdk.spark_util.init(
        username="",
        eve_api_caller="alpha_growth",
        eve_api_token="b846350027e783fb9de3553ef130f8fe",
    )
    evesdk.new_context()

    args = parse_args()
    setup_runtime(args)

    print("Parsed arguments:")
    print(args)
    save_to_hive(args)
