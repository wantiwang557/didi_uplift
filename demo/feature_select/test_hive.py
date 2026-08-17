import pandas as pd
import tensorflow as tf
import os
import json
import argparse
import sys
import datetime
import math
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from dmlite.common import DataFrameIO, init_spark_session
from dmlite.tensorflow import save_model, load_model, TFRecordReader
import evesdk
from matplotlib import pyplot as plt
from functools import reduce

# If you have fextract in your path
fextract_path = os.path.abspath("/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/zyr_workspace/offline_bigdata/python")
sys.path.append(fextract_path)
from tensorflow.keras.layers import BatchNormalization, Activation

# 设置可使用的 GPU 序号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置随机种子复现相同的结果
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

try:
    import metrics
    print("load metric func from causalml...")
except:
    import metrics
    print("load metric func from local...")

# 注意：请确保这些路径下有真实数据
# test_rta_tfr_path = "hdfs://difed/user/prod_growstrategy/zyr/tfr_data/26v0417/val2/2026/04/20/part-*"
# test_rta_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/winterfu/26_v2/test/2026/07/16/part-*"
test_rta_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/16/part-*"
now = datetime.now()
time_str = now.strftime('%Y%m%d')

# =========================================================================
# 模型结构与数据管道
# =========================================================================
class IndexEmbedding(tf.keras.layers.Embedding):
    def __init__(self, dense_fea_dim, embedding_dim, sparse_fea_dim,
                 embeddings_initializer="uniform", embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None,
                 mask_zero=False, input_length=None, **kwargs):
        super(IndexEmbedding, self).__init__(
            input_dim=dense_fea_dim, output_dim=embedding_dim,
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero, input_length=input_length, **kwargs)
        self.sparse_fea_dim = sparse_fea_dim
        self.out_dense_dim = sparse_fea_dim * embedding_dim

    def build(self, input_shape=None):
        name = "index_embedding_{}_{}".format(self.input_dim, self.output_dim)
        self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer, regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint, experimental_autocast=False, name=name)
        self.built = True

    def call(self, feature, **kwargs):
        _embedding = tf.nn.embedding_lookup(self.embeddings, feature)
        embedding = tf.reshape(_embedding, [-1, self.out_dense_dim])
        return embedding

class LiteFeTFRecords(object):
    def __init__(self, hdfs_pattern, start_dt, end_dt, label_size=-1, token_size=-1, dt_fmt="%Y%m%d"):
        _hdfs_pattern = hdfs_pattern.strip()
        # if not _hdfs_pattern.startswith("hdfs://"): _hdfs_pattern = "hdfs://" + _hdfs_pattern
        self.hdfs_pattern = _hdfs_pattern
        self.label_size = label_size
        self.token_size = token_size
        self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
        self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)

    def _get_hdfs_path_list(self):
        tfr_path_list = []
        days = (self.end_dt - self.start_dt).days
        for i in range(days + 1):
            _pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
            tfr_path = tf.io.gfile.glob(_pattern)
            tfr_path_list.extend(tfr_path)
        return tfr_path_list

    def get_dataset(self, epoch=1, batch_size=1, parallel=1, shuffle_size=0, drop_remainder=True, return_token=False, token_dtype=str):
        feature_dict= {
            "label":   tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token":   tf.io.FixedLenFeature([1], tf.string),
        }
        _token2str, _token_dtype = { int: (False, tf.int64), float: (False, tf.float32), str: (True, tf.string) }[token_dtype]

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
                if not _token2str: _token = tf.strings.to_number(_token, _token_dtype)
                token = tf.reshape(_token, [self.token_size])
                res = (feature, label, token)
            else:
                res = (feature, label)
            return res

        tfr_path_list = self._get_hdfs_path_list()
        dataset = tf.data.TFRecordDataset(filenames=tfr_path_list, num_parallel_reads=parallel) \
            .map(map_func=_parse_single_example, num_parallel_calls=parallel) \
            .batch(batch_size=batch_size, drop_remainder=drop_remainder) \
            .repeat(epoch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        if shuffle_size > 0: dataset = dataset.shuffle(shuffle_size)
        return dataset

# =========================================================================
# 任务调度与执行方法
# =========================================================================

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

def load_from_tfr(data_type, args):
    if data_type == "test":
        reader = LiteFeTFRecords(hdfs_pattern=test_rta_tfr_path, start_dt=args.test_dt, end_dt=args.test_dt, token_size=14, label_size=1)
    else:
        print("never into here...")
        reader = LiteFeTFRecords(hdfs_pattern=test_rta_tfr_path, start_dt=args.test_dt, end_dt=args.test_dt, token_size=14, label_size=1)
    dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=4, token_dtype=str, drop_remainder=False, return_token=True)

    if args.feature_slot_indices is not None:
        slot_tensor = tf.constant(args.feature_slot_indices, dtype=tf.int32)
        def _select_feature_slots(feature, label, token):
            feature_pairs = tf.reshape(feature, [tf.shape(feature)[0], -1, 2])
            selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
            selected_feature = tf.reshape(selected_pairs, [tf.shape(feature)[0], -1])
            return selected_feature, label, token
        dataset = dataset.map(_select_feature_slots, num_parallel_calls=4)

    return dataset

def model_path(args):
    # 设定实际存储的路径
    hdfs_model_path = "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/"
    args.hdfs_model_path = hdfs_model_path
    if not hasattr(args, 'nfs_model_path'):
        args.nfs_model_path = args.hdfs_model_path
    return args

def save_to_hive(args):
    print(f"Loading model from: {args.hdfs_model_path}")
    model = tf.keras.models.load_model(args.hdfs_model_path)

    test_dataset = load_from_tfr("test", args)
    #print(test_dataset.element_spec)

    pid, treatment, label_dt = [], [], []
    prediction_1, prediction_2, prediction_3, prediction_4 = [], [], [], []
    scores_cost_1, scores_cost_2, scores_cost_3, scores_cost_4 = [], [], [], []

    for feature, label, token in tqdm(test_dataset, desc="预测数据", leave=False):
        inputs = feature
        #print('get feature........')
        #print("inputs shape is: ",inputs.shape)
        t = token
        #print('get totken........')
        #print("token shape is: ", t.shape)
        is_treat = tf.strings.to_number(t[:, 11], tf.float32)

        # 当前 EUEN 模型输出顺序:
        # [control, control_stop_grad, uplift_1, uplift_2, uplift_3, uplift_4, cost_1, cost_2, cost_3, cost_4]
        # 其中 uplift_1 表示 T1-control, uplift_2 表示 T2-T1, uplift_3 表示 T3-T1, uplift_4 表示 T4-T1
        uc, c_logit_fix, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)

        prediction_1.append(u_tau_1)
        prediction_2.append(u_tau_1 + u_tau_2)
        prediction_3.append(u_tau_1 + u_tau_3)
        prediction_4.append(u_tau_1 + u_tau_4)
        scores_cost_1.append(cost_1)
        scores_cost_2.append(cost_2)
        scores_cost_3.append(cost_3)
        scores_cost_4.append(cost_4)

        pid.append(token[:, 2].numpy())
        treatment.append(is_treat.numpy())
        label_dt.append(token[:, 13].numpy())

    if len(pid) == 0:
        print("错误：测试集为空，没有预测任何数据！请检查 HDFS 路径下是否有对应日期的 part 文件。")
        return

    pid = tf.concat(pid, axis=0).numpy().squeeze()
    treatment = tf.concat(treatment, axis=0).numpy().squeeze()
    label_dt = tf.concat(label_dt, axis=0).numpy().squeeze()

    predict_1 = tf.concat(prediction_1, axis=0).numpy().squeeze()
    predict_2 = tf.concat(prediction_2, axis=0).numpy().squeeze()
    predict_3 = tf.concat(prediction_3, axis=0).numpy().squeeze()
    predict_4 = tf.concat(prediction_4, axis=0).numpy().squeeze()

    cost_1_exp = tf.concat(scores_cost_1, axis=0).numpy().squeeze()
    cost_2_exp = tf.concat(scores_cost_2, axis=0).numpy().squeeze()
    cost_3_exp = tf.concat(scores_cost_3, axis=0).numpy().squeeze()
    cost_4_exp = tf.concat(scores_cost_4, axis=0).numpy().squeeze()

    pred_df = pd.DataFrame({
        'pid': pid,
        'treatment': treatment,
        'label_dt': label_dt,
        'uplift_1': predict_1,
        'uplift_2': predict_2,
        'uplift_3': predict_3,
        'uplift_4': predict_4,
        'cost_1': cost_1_exp,
        'cost_2': cost_2_exp,
        'cost_3': cost_3_exp,
        'cost_4': cost_4_exp
    })

    pred_df['pid'] = pred_df['pid'].apply(lambda x: x.decode('utf-8'))
    pred_df['label_dt'] = pred_df['label_dt'].apply(lambda x: x.decode('utf-8'))
    print(pred_df)
    print('预测dt分区为:', args.startDate)

    dfio = DataFrameIO()
    dfio.write_table(
        df = pred_df,
       # table_name = 'growstrategy.ut_msg_outcall_euen_26_v2_month_5_6_model_inference_di',
        table_name = 'growstrategy.ut_msg_outcall_feature_select_v0_month_5_6_model_inference_di',
        partition_val = args.startDate,
        partition_col = 'dt'
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Uplift modeling parameters")
    parser.add_argument('--PADDING_SIZE', type=int, default=446)
    parser.add_argument('--DENSE_FEA_SIZE', type=int, default=3500)
    parser.add_argument('--EMBEDDING_SIZE', type=int, default=8)
    parser.add_argument('--BATCH_SIZE', type=int, default=2048)
    parser.add_argument('--train_start_dt', type=str, default="20250624")
    parser.add_argument('--train_end_dt', type=str, default="20250909")
    parser.add_argument('--test_dt', type=str, default="20260716")
    parser.add_argument('--version', type=str, default="multi_EUEN_uplift_cost_merge")
    parser.add_argument('--hc_dim', type=int, default=256)
    parser.add_argument('--hu_dim', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=2048)
    parser.add_argument('--treats_num', type=int, default=4)
    parser.add_argument('--startDate', type=str, default="20260706")
    parser.add_argument('--feature_slot_file', type=str, default="./baseline_baseline.slots")
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    evesdk.spark_util.init(username='', eve_api_caller="alpha_growth", eve_api_token="b846350027e783fb9de3553ef130f8fe")
    eve_ctx = evesdk.new_context()

    args = parse_args()
    args = model_path(args)
    args.feature_slot_indices = load_feature_slot_indices(args.feature_slot_file)
    if args.feature_slot_indices is not None:
        args.PADDING_SIZE = len(args.feature_slot_indices)

    print("Parsed arguments:")
    print(args)
    save_to_hive(args)
