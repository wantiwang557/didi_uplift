import pandas as pd
import tensorflow as tf
import os
import argparse
import sys
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from dmlite.common import DataFrameIO
import evesdk
from tensorflow.keras.layers import BatchNormalization, Activation

# =========================================================
# 只改这里即可切换 v1 / v2 / v3
# 当前文件对应 EEUEN_4t_02：共享 exposure head。
# =========================================================
MODEL_VERSION = "v1"

MODEL_CONFIG = {
    "v1": {
        "test_rta_tfr_path": "hdfs://difed/user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13/part-*",
        "hdfs_model_path": "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/eeuen_4t_02/v1_train_0511_0617_test_0623_0629_epoch2",
        "table_name": "growstrategy.ut_msg_outcall_eeuen_exposure_v1_0511_0617_inference_di",
        "version": "eeuen_v1_train_0511_0617_test_0623_0706_02",
        "train_start_dt": "20260511",
        "train_end_dt": "20260617",
        "test_window": "2026-06-23~2026-07-06",
        "best_epoch": 2,
    },
    "v2": {
        "test_rta_tfr_path": "hdfs://difed/user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/part-*",
        "hdfs_model_path": "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/eeuen_4t_02/v2_train_0511_0617_test_0723_0806_epoch3",
        "table_name": "growstrategy.ut_msg_outcall_eeuen_exposure_v2_0511_0617_inference_di",
        "version": "eeuen_v2_train_0511_0617_test_0723_0806_02",
        "train_start_dt": "20260511",
        "train_end_dt": "20260617",
        "test_window": "2026-07-23~2026-08-06, excluding 2026-08-02",
        "best_epoch": 3,
    },
    "v3": {
        "test_rta_tfr_path": "hdfs://difed/user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/part-*",
        "hdfs_model_path": "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/eeuen_4t_02/v3_train_0511_0722_test_0723_0806_epoch2",
        "table_name": "growstrategy.ut_msg_outcall_eeuen_exposure_v3_0511_0722_inference_di",
        "version": "eeuen_v3_train_0511_0722_test_0723_0806_02",
        "train_start_dt": "20260511",
        "train_end_dt": "20260722",
        "test_window": "2026-07-23~2026-08-06, excluding 2026-08-02",
        "best_epoch": 2,
    },
}

if MODEL_VERSION not in MODEL_CONFIG:
    raise ValueError("MODEL_VERSION 只能是 v1 / v2 / v3")

CFG = MODEL_CONFIG[MODEL_VERSION]
test_rta_tfr_path = CFG["test_rta_tfr_path"]

fextract_path = os.path.abspath("/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/zyr_workspace/offline_bigdata/python")
sys.path.append(fextract_path)

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
            name=name
        )
        self.built = True

    def call(self, feature, **kwargs):
        _embedding = tf.nn.embedding_lookup(self.embeddings, feature)
        embedding = tf.reshape(_embedding, [-1, self.out_dense_dim])
        return embedding


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
            _pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
            tfr_path_list.extend(tf.io.gfile.glob(_pattern))
        return tfr_path_list

    def get_dataset(self, epoch=1, batch_size=1, parallel=1, shuffle_size=0, drop_remainder=True, return_token=False, token_dtype=str):
        feature_dict = {
            "label": tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token": tf.io.FixedLenFeature([1], tf.string),
        }
        _token2str, _token_dtype = {
            int: (False, tf.int64),
            float: (False, tf.float32),
            str: (True, tf.string)
        }[token_dtype]

        def _parse_single_example(example):
            feature_map = tf.io.parse_single_example(example, feature_dict)
            _labels = tf.strings.split(feature_map["label"], ",").values
            _labels = tf.strings.to_number(_labels, tf.float32)
            label = tf.reshape(_labels, [self.label_size])

            libsvm_kv = tf.strings.split(feature_map["feature"], ",").values
            libsvm_kv = tf.strings.split(libsvm_kv, ":").values
            feature = tf.strings.to_number(libsvm_kv, tf.float32)

            if return_token:
                _token = tf.strings.split(feature_map["token"], ",").values
                if not _token2str:
                    _token = tf.strings.to_number(_token, _token_dtype)
                token = tf.reshape(_token, [self.token_size])
                return feature, label, token

            return feature, label

        tfr_path_list = self._get_hdfs_path_list()
        print("TFRecord files count:", len(tfr_path_list))
        if len(tfr_path_list) == 0:
            raise ValueError("测试集为空：没有匹配到任何 TFRecord part 文件，请检查 test_rta_tfr_path。")

        dataset = tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel
        ).map(
            map_func=_parse_single_example,
            num_parallel_calls=parallel
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ).repeat(epoch).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)

        return dataset


def load_from_tfr(args):
    reader = LiteFeTFRecords(
        hdfs_pattern=test_rta_tfr_path,
        start_dt=args.test_dt,
        end_dt=args.test_dt,
        token_size=args.token_size,
        label_size=1
    )
    return reader.get_dataset(
        epoch=1,
        batch_size=args.BATCH_SIZE,
        parallel=4,
        token_dtype=str,
        drop_remainder=False,
        return_token=True
    )


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def concat_tensor_list(values):
    return tf.concat(values, axis=0).numpy().squeeze()


def save_to_hive(args):
    print("MODEL_VERSION:", MODEL_VERSION)
    print("Best epoch:", CFG["best_epoch"])
    print("Test window:", CFG["test_window"])
    print("Loading model from:", CFG["hdfs_model_path"])
    print("Writing table:", CFG["table_name"])

    model = tf.keras.models.load_model(CFG["hdfs_model_path"], compile=False)
    test_dataset = load_from_tfr(args)

    pid, treatment, label_dt, exp_label = [], [], [], []
    prediction_1, prediction_2, prediction_3, prediction_4 = [], [], [], []
    exposure_uplift_1, exposure_uplift_2, exposure_uplift_3, exposure_uplift_4 = [], [], [], []
    exposure_prob_list = []
    scores_cost_1, scores_cost_2, scores_cost_3, scores_cost_4 = [], [], [], []
    first_batch = True

    for feature, _label, token in tqdm(test_dataset, desc="预测数据", leave=False):
        if first_batch:
            raw_token = [decode_if_bytes(x) for x in token[0].numpy()]
            print("First token sample:")
            for idx, val in enumerate(raw_token):
                print(idx, TOKEN_COLUMNS[idx] if idx < len(TOKEN_COLUMNS) else "UNKNOWN", val)
            first_batch = False

        is_treat = tf.strings.to_number(token[:, TREATMENT_INDEX], tf.float32)
        batch_exp_label = tf.strings.to_number(token[:, EXP_INDEX], tf.float32)

        (
            _uc, _c_logit_fix,
            u_tau_1, u_tau_2, u_tau_3, u_tau_4,
            exposure_prob,
            cost_1, cost_2, cost_3, cost_4
        ) = model(feature)

        tau_e_1 = u_tau_1
        tau_e_2 = u_tau_1 + u_tau_2
        tau_e_3 = u_tau_1 + u_tau_3
        tau_e_4 = u_tau_1 + u_tau_4

        prediction_1.append(exposure_prob * tau_e_1)
        prediction_2.append(exposure_prob * tau_e_2)
        prediction_3.append(exposure_prob * tau_e_3)
        prediction_4.append(exposure_prob * tau_e_4)
        exposure_prob_list.append(exposure_prob)
        exposure_uplift_1.append(tau_e_1)
        exposure_uplift_2.append(tau_e_2)
        exposure_uplift_3.append(tau_e_3)
        exposure_uplift_4.append(tau_e_4)
        scores_cost_1.append(cost_1)
        scores_cost_2.append(cost_2)
        scores_cost_3.append(cost_3)
        scores_cost_4.append(cost_4)

        pid.append(token[:, PID_INDEX].numpy())
        treatment.append(is_treat.numpy())
        label_dt.append(token[:, LABEL_DT_INDEX].numpy())
        exp_label.append(batch_exp_label.numpy())

    if len(pid) == 0:
        raise ValueError("测试集为空，没有预测任何数据！请检查 HDFS 路径。")

    exposure_prob_arr = concat_tensor_list(exposure_prob_list)

    pred_df = pd.DataFrame({
        "pid": [decode_if_bytes(x) for x in np.concatenate(pid, axis=0).squeeze()],
        "treatment": np.concatenate(treatment, axis=0).squeeze().astype(int),
        "label_dt": [decode_if_bytes(x) for x in np.concatenate(label_dt, axis=0).squeeze()],
        "exp_label": np.concatenate(exp_label, axis=0).squeeze().astype(int),
        "uplift_1": concat_tensor_list(prediction_1),
        "uplift_2": concat_tensor_list(prediction_2),
        "uplift_3": concat_tensor_list(prediction_3),
        "uplift_4": concat_tensor_list(prediction_4),
        "exposure_prob": exposure_prob_arr,
        "exposure_uplift_1": concat_tensor_list(exposure_uplift_1),
        "exposure_uplift_2": concat_tensor_list(exposure_uplift_2),
        "exposure_uplift_3": concat_tensor_list(exposure_uplift_3),
        "exposure_uplift_4": concat_tensor_list(exposure_uplift_4),
        "cost_1": concat_tensor_list(scores_cost_1),
        "cost_2": concat_tensor_list(scores_cost_2),
        "cost_3": concat_tensor_list(scores_cost_3),
        "cost_4": concat_tensor_list(scores_cost_4),
    })

    print(pred_df.head())
    print("预测样本数:", len(pred_df))
    print("预测dt分区为:", args.startDate)

    DataFrameIO().write_table(
        df=pred_df,
        table_name=CFG["table_name"],
        partition_val=args.startDate,
        partition_col="dt"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="EEUEN inference and Hive writer")
    parser.add_argument("--PADDING_SIZE", type=int, default=473)
    parser.add_argument("--DENSE_FEA_SIZE", type=int, default=3500)
    parser.add_argument("--EMBEDDING_SIZE", type=int, default=8)
    parser.add_argument("--BATCH_SIZE", type=int, default=2048)
    parser.add_argument("--train_start_dt", type=str, default=CFG["train_start_dt"])
    parser.add_argument("--train_end_dt", type=str, default=CFG["train_end_dt"])
    parser.add_argument("--test_dt", type=str, default="20260813")
    parser.add_argument("--version", type=str, default=CFG["version"])
    parser.add_argument("--hc_dim", type=int, default=256)
    parser.add_argument("--hu_dim", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=2048)
    parser.add_argument("--treats_num", type=int, default=4)
    parser.add_argument("--startDate", type=str, default="20260813")
    parser.add_argument("--token_size", type=int, default=15)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    evesdk.spark_util.init(
        username="",
        eve_api_caller="alpha_growth",
        eve_api_token="b846350027e783fb9de3553ef130f8fe"
    )
    _eve_ctx = evesdk.new_context()
    args = parse_args()
    print("Parsed arguments:")
    print(args)
    save_to_hive(args)
