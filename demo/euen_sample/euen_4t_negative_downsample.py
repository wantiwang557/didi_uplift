'''
Version: euen_4t_negative_downsample (standalone)

基于 euen_4t_02.py 的完整单文件训练代码，不导入 euen_4t_02。

采样口径：
1. 仅训练流对全负用户做 pid 级稳定下采样，正样本始终保留。
2. train_eval、valid、test 保持全量，不改变评估口径。
3. --negative_keep_probs 支持一次输入多个保留比例；1.0 表示不下采样。
4. 当前 SQL 已保证“有正有负用户只留正样本，全负用户保留全部记录”。

v6 示例：依次训练保留 25%、50%、75%、100% 全负用户的四组模型

nohup python euen_4t_negative_downsample.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v6/train_0716_0821/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v6/valid_0822_0827/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v6/test_0828_0910/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/euen_4t_negative_downsample/v6" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_4t_v6" \
--negative_keep_probs 0.25 0.5 0.75 1.0 \
--negative_sample_seed 2026 \
--DENSE_FEA_SIZE 3500 \
--PADDING_SIZE 473 \
--token_size 15 \
--gpu 2 > ./run_logs/euen_4t_v6_negative_downsample.log 2>&1 &

查看日志：
tail -f ./run_logs/euen_4t_v6_negative_downsample.log
'''
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation
from functools import reduce
from matplotlib import pyplot as plt
import datetime
from datetime import datetime, timedelta
import os
import json
import argparse
import sys
import time
sys.stdout.reconfigure(line_buffering=True)


def setup_gpu(gpu_id="0"):
    """设置可使用的 GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpu_id}")
    print(f"Num GPUs Available: {len(physical_gpus)}")

# 设置随机种子复现相同的结果
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wh/demo/thirdparty/backflow/backflow/visualization')
try:
    import metrics
    print("load metric func from causalml...")
except:
    import metrics
    print("load metric func from local...")

from pcoc_evaluation import PCOCEvaluator
import pytz

# 北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

def get_beijing_time_str():
    """获取当前北京时间字符串"""
    return datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

# 获取当前时间
now = datetime.now()
time_str = now.strftime('%Y%m%d_%H%M%S')


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
HASH_BUCKETS = 1_000_000


class IndexEmbedding(tf.keras.layers.Embedding):
    def __init__(self,
                 dense_fea_dim,
                 embedding_dim,
                 sparse_fea_dim,
                 embeddings_initializer="uniform", # 均匀分布初始化
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None, 
                 mask_zero=False, # 为什么不mask padding的0占位？
                 input_length=None,
                 **kwargs):

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
    """TFRecord 数据读取器"""
    def __init__(self,
                 hdfs_pattern,
                 start_dt,
                 end_dt,
                 label_size=-1,
                 token_size=-1,
                 dt_fmt="%Y%m%d",
                 parse_mode='batch_then_map'):
        self.hdfs_pattern = hdfs_pattern.strip()
        self.label_size = label_size
        self.token_size = token_size
        self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
        self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)
        self.parse_mode = parse_mode

    def _get_hdfs_path_list(self):
        tfr_path_list = []
        days = (self.end_dt - self.start_dt).days
        for i in range(days + 1):
            _pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
            tfr_path = tf.io.gfile.glob(_pattern)
            tfr_path_list.extend(tfr_path)
        return tfr_path_list
    
    # 创建dataset，模型训练时会按照batch从这个dataset里拿数据
    def get_dataset(self,
                    epoch=1,
                    batch_size=1,
                    parallel=1,
                    shuffle_size=0,
                    shuffle_seed=None,
                    drop_remainder=True,
                    return_token=False,
                    token_dtype=str,
                    negative_keep_prob=1.0,
                    negative_sample_seed=2026):
        # 这个字段固定长度是 1，是字符串类型。
        feature_dict = {
            "label":   tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token":   tf.io.FixedLenFeature([1], tf.string),
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
                return (feature, label, token)
            return (feature, label)

        # 解析一个 batch 的 TFRecord，改成tensor.
        def _parse_batch_example(serialized_batch):
            feature_map = tf.io.parse_example(serialized_batch, feature_dict)
            
            _labels = tf.strings.split(feature_map["label"][:, 0], ",")
            _labels = tf.strings.to_number(_labels, tf.float32)
            label = _labels.to_tensor()
            label = tf.reshape(label, [-1, self.label_size])
            # 根据样本1 feature = "12:1,45:1"的格式对feature进行解析，最后reshape回batch的形状。
            libsvm_kv = tf.strings.split(feature_map["feature"][:, 0], ",")
            libsvm_kv = tf.strings.split(libsvm_kv, ":")
            libsvm_kv = libsvm_kv.flat_values
            feature = tf.strings.to_number(libsvm_kv, tf.float32)
            feature = tf.reshape(feature, [-1, tf.shape(feature)[0] // tf.shape(label)[0]])

            if return_token:
                _token = tf.strings.split(feature_map["token"][:, 0], ",")
                if not _token2str:
                    _token = tf.strings.to_number(_token, _token_dtype)
                    token = _token.to_tensor()
                else:
                    token = _token.to_tensor(default_value="")
                token = tf.reshape(token, [-1, self.token_size])
                return (feature, label, token)
            else:
                return (feature, label)

        tfr_path_list = self._get_hdfs_path_list()
        print(f"[LiteFeTFRecords] Found {len(tfr_path_list)} files, parse_mode={self.parse_mode}")
        
        dataset = tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel
        )
        
        if shuffle_size > 0:
            dataset = dataset.shuffle(
                buffer_size=shuffle_size,
                seed=shuffle_seed,
                reshuffle_each_iteration=True
            )
        
        if self.parse_mode == 'batch_then_map':
            dataset = dataset.batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder
            ).map(
                map_func=_parse_batch_example,
                num_parallel_calls=parallel
            )
        else:
            dataset = dataset.map(
                map_func=_parse_single_example,
                num_parallel_calls=parallel
            ).batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        
        # 上游 SQL 已保证：有正样本用户只剩正样本，label==0 的记录均来自
        # 全负用户。这里按 pid 做稳定哈希，确保同一用户的负样本一起保留或剔除。
        if negative_keep_prob < 1.0:
            threshold = int(negative_keep_prob * HASH_BUCKETS)
            hash_key = [
                int(negative_sample_seed) & 0xFFFFFFFF,
                (int(negative_sample_seed) ^ 0x9E3779B9) & 0xFFFFFFFF,
            ]

            def keep_sample(_feature, label, token):
                label_value = tf.reshape(label, [-1])[0]
                pid = token[PID_INDEX]
                user_bucket = tf.strings.to_hash_bucket_strong(
                    pid,
                    num_buckets=HASH_BUCKETS,
                    key=hash_key,
                )
                keep_negative = user_bucket < tf.cast(threshold, tf.int64)
                return tf.logical_or(label_value > 0.0, keep_negative)

            dataset = (
                dataset.unbatch()
                .filter(keep_sample)
                .batch(batch_size, drop_remainder=drop_remainder)
            )

        dataset = dataset.repeat(epoch).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        
        return dataset


def simple_gap_delta_cvr(df, prob_col, label_col):
    df['{}_freq_bucket'.format(prob_col)] = pd.qcut(df[prob_col], q=10, labels=False)
    stats = {}
    for i in range(10):
        group_df = df[df['{}_freq_bucket'.format(prob_col)] == i]
        count_888 = group_df[group_df['real_treatment'] == 1].shape[0]
        count_555 = group_df[group_df['real_treatment'] == 0].shape[0]
        sum_label_888 = group_df[group_df['real_treatment'] == 1][label_col].sum()
        sum_label_555 = group_df[group_df['real_treatment'] == 0][label_col].sum()
        ingroup_cnt = len(group_df)
        cvt = group_df[label_col].sum()
        tCVR = group_df[label_col].mean()
        mean_label_888 = sum_label_888 / count_888 if count_888 > 0 else 0
        mean_label_555 = sum_label_555 / count_555 if count_555 > 0 else 0
        mean_true_label_diff = mean_label_888 - mean_label_555

        stats[i] = {
            '{}_freq_bucket'.format(prob_col): 10-i,
            'count_888': count_888,
            'count_555': count_555,
            'sum_label_888': sum_label_888,
            'sum_label_555': sum_label_555,
            'ingroup_cnt': ingroup_cnt,
            'cvt': cvt,
            'tCVR': tCVR,
            'mean_label_888': mean_label_888,
            'mean_label_555': mean_label_555,
            'mean_true_label_diff': mean_true_label_diff,
            prob_col: group_df[prob_col].mean()
        }
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index(drop=True)
    return stats_df

# uplift-curve
def draw_auuc(path, prediction_pdf, treatments, ite_names, label_col='pv_label', 
              real_treatment='real_treatment', control_group_name='control_group', show_curve=True, verbose=True):
    metric_results = []
    
    for i, (treatment, ite_name, control_group) in enumerate(zip(treatments, ite_names, control_group_name)):
        subset = prediction_pdf[prediction_pdf[real_treatment].isin([treatment, control_group])]
        subset['exp_group'] = subset[real_treatment].apply(lambda x: 1 if x != control_group else 0)
        subset['label'] = subset[label_col]
    
    # 按照ite_name(ite_rank)排序，计算累计uplift:
    # topK 人群真实 uplift = topK 中 treatment 组平均 finish_cnt_7d - topK 中 control 组平均 finish_cnt_7d
        lift = metrics.get_cumlift(subset[['exp_group', 'label', ite_name]],
                                   outcome_col='label', treatment_col='exp_group', random_seed=4)
        gain = lift.mul(lift.index.values, axis=0)
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)
        if show_curve:
            real_path = "{base}_{num}.png".format(base=path, num=i)
            gain.plot()
            plt.savefig(real_path)
        res = gain.sum() / gain.shape[0]
        metric_results.append(res)
    
    return metric_results

import math

class InversePairsCalc:
    def InversePairs(self, data):
        if not data: return 0
        if len(data) == 1: return 0
        
        def merge(tuple_fir, tuple_sec):
            array_before = tuple_fir[0]
            cnt_before = tuple_fir[1]
            array_after = tuple_sec[0]
            cnt_after = tuple_sec[1]
            cnt = cnt_before + cnt_after
            flag = len(array_after) - 1
            array_merge = []
            for i in range(len(array_before) - 1, -1, -1):
                while flag >= 0 and array_before[i] < array_after[flag]:
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1:
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag + 1)
            
            if flag == -1:
                for j in range(i, -1, -1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag, -1, -1):
                    array_merge.append(array_after[j])
            return array_merge[::-1], cnt
        
        def mergesort(array):
            if len(array) == 1:
                return (array, 0)
            cut = math.floor(len(array) / 2)
            tuple_fir = mergesort(array[:cut])
            tuple_sec = mergesort(array[cut:])
            return merge(tuple_fir, tuple_sec)
        return mergesort(data)[1]

def xauc_score(y_labels, y_scores, verbose_name=None):
    if hasattr(y_labels, 'values'):
        y_labels = y_labels.values
    if hasattr(y_scores, 'values'):
        y_scores = y_scores.values
    
    y_labels = np.array(y_labels).reshape(-1)
    y_scores = np.array(y_scores).reshape(-1)
    n = len(y_labels)
    if n <= 1: return 0.0
    
    label_preds = list(zip(y_labels, y_scores))
    sorted_label_preds = sorted(label_preds, key=lambda lc: lc[1], reverse=True)
    labels_sort = [ele[0] for ele in sorted_label_preds]
    
    S = InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    pairs_cnt = n * (n - 1) / 2
    return total_positive / pairs_cnt if pairs_cnt > 0 else 0.0

def load_from_tfr(train_type, args):
    shuffle_seed = getattr(args, 'shuffle_seed', None)
    parse_mode = getattr(args, 'parse_mode', 'batch_then_map')
    
    if train_type == "train":
        reader = LiteFeTFRecords(
            hdfs_pattern=args.train_tfr_path, start_dt=args.train_start_dt, end_dt=args.train_end_dt, # tfrecord文件的路径，起止时间，token,label_size
            token_size=args.token_size, label_size=args.label_size, parse_mode=parse_mode
        )
        dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=args.parallel, 
                                     shuffle_size=args.shuffle_size, shuffle_seed=shuffle_seed, 
                                     token_dtype=str, drop_remainder=False, return_token=True,
                                     negative_keep_prob=args.negative_keep_prob,
                                     negative_sample_seed=args.negative_sample_seed)
        print(
            "[NEGATIVE DOWNSAMPLE] train only: "
            f"keep_prob={args.negative_keep_prob}, "
            f"seed={args.negative_sample_seed}, pid_token_index={PID_INDEX}"
        )
    elif train_type == "train_eval":
        reader = LiteFeTFRecords(
            hdfs_pattern=args.train_tfr_path, start_dt=args.train_start_dt, end_dt=args.train_end_dt,
            token_size=args.token_size, label_size=args.label_size, parse_mode=parse_mode
        )
        dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=args.parallel,
                                     shuffle_size=0, shuffle_seed=None,
                                     token_dtype=str, drop_remainder=False, return_token=True)
    elif train_type == "test":
        reader = LiteFeTFRecords(
            hdfs_pattern=args.test_tfr_path, start_dt=args.test_dt, end_dt=args.test_dt, 
            token_size=args.token_size, label_size=args.label_size, parse_mode=parse_mode
        )
        dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=args.parallel, 
                                     token_dtype=str, drop_remainder=False, return_token=True)
    else:
        reader = LiteFeTFRecords(
            hdfs_pattern=args.valid_tfr_path, start_dt=args.test_dt, end_dt=args.test_dt, 
            token_size=args.token_size, label_size=args.label_size, parse_mode=parse_mode
        )
        dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=args.parallel, 
                                     token_dtype=str, drop_remainder=False, return_token=True)

    if args.debug_mode:
        dataset = dataset.take(args.debug_batches)
        print(f"[DEBUG MODE] {train_type} dataset limited to {args.debug_batches} batches")
    return dataset


class EUEN(tf.keras.Model):
    def __init__(self, args):
        super(EUEN, self).__init__()
        # V2: 4个Treatment (1, 2, 3, 4)
        self.treats_num = args.treats_num 
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) #
        self.bias_initializer = tf.keras.initializers.Constant(value=0)
        self.kernel_regularizer = tf.keras.regularizers.l2(args.l2_reg)

        # DENSE_FEA_SIZE：全局特征 ID 数量，也就是 embedding 表行数
        # EMBEDDING_SIZE：每个特征 ID 映射成多少维向量
        # PADDING_SIZE：每条样本固定有多少个特征槽位
        self.embedding_layer = IndexEmbedding(dense_fea_dim=args.DENSE_FEA_SIZE, embedding_dim=args.EMBEDDING_SIZE, sparse_fea_dim=args.PADDING_SIZE)
        
        self.activation = [args.act_type, args.act_type, args.act_type]
        # shared_bottomt通过三层MLP输出为emb
        self.share_bottom = [tf.keras.layers.Dense(units=unit, activation=act, kernel_initializer=self.kernel_initializer, 
                             bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                             for unit, act in zip([args.emb_dim, args.emb_dim // 2, args.emb_dim // 4], self.activation)]

        self.c_head = [tf.keras.layers.Dense(units=hc_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                       bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                       for hc_dim, act in zip([args.hc_dim, args.hc_dim//2, args.hc_dim//4, 1], self.activation + [None])]

        # 生成 4 个 U-Head 和 4 个 Cost-Head
        self.u_head = [[tf.keras.layers.Dense(units=hu_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                        bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                        for hu_dim, act in zip([args.hu_dim, args.hu_dim//2, args.hu_dim//4, 1], self.activation + [None])] 
                       for _ in range(self.treats_num)]

        self.cost_head = [[tf.keras.layers.Dense(units=hu_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                           bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                           for hu_dim, act in zip([args.hu_dim, args.hu_dim//2, args.hu_dim//4, 1], self.activation + [None])] 
                          for _ in range(self.treats_num)]

    def call(self, inputs):
        input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1 #取出所有特征索引 index. :样本batch维度，取出所有样本，0从下标0开始取,::2步长为2， 取 0、2、4、6…… 偶数下标）
        input_value = tf.cast(inputs[:, 1::2], tf.float32) #1::2：从下标 1 开始，步长 2，取 1、3、5、7…… 奇数下标
        inputs = self.embedding_layer(input_index[:, :])

        inputs = reduce(lambda x, layer: layer(x), self.share_bottom, inputs)

        c_head = reduce(lambda x, c_task: c_task(x), self.c_head, inputs)
        u_head = [reduce(lambda x, u_task: u_task(x), self.u_head[i], inputs) for i in range(self.treats_num)]
        cost_head = [reduce(lambda x, cost: cost(x), self.cost_head[i], inputs) for i in range(self.treats_num)]

        c_head_logit_fix = tf.stop_gradient(c_head)
        
        # 返回 Control, Fix-Logit, 及 1,2,3,4 的 uplift 和 cost
        return c_head, c_head_logit_fix, u_head[0], u_head[1], u_head[2], u_head[3], cost_head[0], cost_head[1], cost_head[2], cost_head[3]

# 训练模型，预测结果，计算AUUC指标
def predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False,
                               valid_distribution=False, epoch_num=None, collect_predictions=True):
    # 为 V2 准备的数据列表 (Treatment: 1, 2, 3, 4)
    uplift_scores_list_1 = []
    uplift_scores_list_2 = []
    uplift_scores_list_3 = []
    uplift_scores_list_4 = []
    scores_cost_7d_1_list = []
    scores_cost_7d_2_list = []
    scores_cost_7d_3_list = []
    scores_cost_7d_4_list = []
    label_cost_7d_list = []
    label_list = [] #7天完单量
    real_treatment_list = []
    batch_count = 0
    loss_values = []
    collect_predictions = collect_predictions or (not train)
    
    mode_str = 'train' if train else test
    epoch_str = f"Epoch {epoch_num}" if epoch_num is not None else ""
    log_interval = getattr(args, 'log_interval', 100)
    debug_token_check = getattr(args, 'debug_token_check', False)
    epoch_start_time = time.time()
    
    TREAT_INDEX = -3
    COST_INDEX = -4

    for x, y, t in test_dataset:
        inputs = x
        

        try:
            is_treat_tensor = t[:, TREAT_INDEX]
            is_treat = tf.strings.to_number(is_treat_tensor, tf.float32)
            
            cost_tensor = t[:, COST_INDEX]
            label_cost_7d = tf.strings.to_number(cost_tensor, tf.float32)

            if debug_token_check and batch_count == 0:
                raw_token = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in t[0].numpy()]
                print("First token sample:")
                for idx, val in enumerate(raw_token):
                    print(f"   -> 索引 {idx} 的值是: {val}")
                print("unique treat:", np.unique(is_treat.numpy()))
                print("unique cost:", np.unique(label_cost_7d.numpy()))
        except Exception as e:
            print("\n" + "="*80)
            print("智能拦截：数据索引 (Index) 发生错位！")
            print("代码正试图把文本(例如日期或名称)转换成数字，说明cost_7d 或 treatment 字段不在代码预设的位置。")
            try:
                raw_token = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in t[0].numpy()]
                for idx, val in enumerate(raw_token):
                    print(f"   -> 索引 {idx} 的值是:  {val}")
            except:
                pass
            print("\n【解决办法 - 仅需2步】：")
            print(f"1. 从上方打印的数据中，找到代表 '0/1/2/3/4' (treatment) 的索引数字。")
            print(f"2. 找到代表 '花费/cost' (如 0.0, 1.5) 的索引数字。")
            print(f"3. 返回代码大概第 500 行，把 TREAT_INDEX 和 COST_INDEX 的值改成你找到的真实数字，然后重新运行即可一通到底！")
            print("="*80 + "\n")
            sys.exit(1) # 拦截并安全退出
        # =========================================================

        label = tf.squeeze(y)
        label_cost_7d = tf.squeeze(label_cost_7d)

        if train:
            with tf.GradientTape() as tape:
                uc, c_logit_fix, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)
                
                # 在 v2 中，以 treatment 1 作为 base 计算增量学习的残差
                t_logit_1_fix = tf.stop_gradient(u_tau_1) #为了不让其他treatement更新control组和t1组的参数
                is_treat = tf.reshape(is_treat, [-1, 1])
                label = tf.reshape(label, [-1, 1])
                label_cost_7d = tf.reshape(label_cost_7d, [-1, 1])
                
                # 只保留 [0, 1, 2, 3, 4] 五个组
                is_list = [tf.cast((is_treat == treat), tf.float32) for treat in [0, 1, 2, 3, 4]]

                # ut_1 为基底增量，ut_2、ut_3和ut_4 学习基于 1 的残差
                ut_1 = (c_logit_fix + u_tau_1)
                ut = [(c_logit_fix + t_logit_1_fix + tau) for tau in [u_tau_2, u_tau_3, u_tau_4]]
                
                loss1 = tf.reduce_mean(tf.square(is_list[0] * uc + is_list[1] * ut_1 + is_list[2] * ut[0] + is_list[3] * ut[1] + is_list[4] * ut[2] - label)
                                       + tf.square(is_list[1] * cost_1 + is_list[2] * cost_2 + is_list[3] * cost_3 + is_list[4] * cost_4 - label_cost_7d))
                
                total_loss = loss1
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_values.append(float(loss1.numpy()))
        else:
            _, _, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)

        # 还原：真正的 delta cvr
        delta_cvr_1 = u_tau_1
        delta_cvr_2 = u_tau_2 + u_tau_1
        delta_cvr_3 = u_tau_3 + u_tau_1
        delta_cvr_4 = u_tau_4 + u_tau_1

        if collect_predictions:
            lists = [uplift_scores_list_1, uplift_scores_list_2, uplift_scores_list_3, uplift_scores_list_4,
                     scores_cost_7d_1_list, scores_cost_7d_2_list, scores_cost_7d_3_list, scores_cost_7d_4_list,
                     label_cost_7d_list, label_list, real_treatment_list]
            outs = [delta_cvr_1, delta_cvr_2, delta_cvr_3, delta_cvr_4, cost_1, cost_2, cost_3, cost_4,
                    label_cost_7d, label, is_treat]
            for list_name, value_name in zip(lists, outs):
                list_name.append(value_name.numpy())
        
        batch_count += 1
        
        if batch_count % log_interval == 0:
            elapsed = time.time() - epoch_start_time
            samples_processed = batch_count * args.BATCH_SIZE
            throughput = samples_processed / elapsed if elapsed > 0 else 0
            if train:
                print(f'[{epoch_str} {mode_str}] Batch {batch_count}: {samples_processed} samples, {throughput:.0f} samples/sec | loss={loss1.numpy():.6f}')
            else:
                print(f'[{epoch_str} {mode_str}] Batch {batch_count}: {samples_processed} samples, {throughput:.0f} samples/sec')
    
    total_epoch_time = time.time() - epoch_start_time
    throughput = batch_count * args.BATCH_SIZE / total_epoch_time if total_epoch_time > 0 else 0
    
    print(f'\n{"="*70}')
    print(f'[{epoch_str} {mode_str}] Epoch Summary')
    print(f'{"="*70}')
    print(f'Total batches: {batch_count}, Total time: {total_epoch_time:.2f}s')
    print(f'Throughput: {throughput:.0f} samples/sec')
    print(f'{"="*70}\n')
    
    mean_loss = float(np.mean(loss_values)) if loss_values else 0.0
    if train:
        print(f'loss1_mean: {mean_loss}')

    if train and not collect_predictions:
        return None, 0.0, mean_loss

    out_lists = [uplift_scores_list_1, uplift_scores_list_2, uplift_scores_list_3, uplift_scores_list_4,
                 scores_cost_7d_1_list, scores_cost_7d_2_list, scores_cost_7d_3_list, scores_cost_7d_4_list, 
                 label_cost_7d_list, label_list, real_treatment_list]
    print([len(lst) for lst in out_lists])
    output_col = [np.concatenate(list_name, axis=0) for list_name in out_lists]

    data = {
        'predict_delta_cvr_1': output_col[0].flatten(),
        'predict_delta_cvr_2': output_col[1].flatten(),
        'predict_delta_cvr_3': output_col[2].flatten(),
        'predict_delta_cvr_4': output_col[3].flatten(),
        'predict_cost_7d_1':   output_col[4].flatten(),
        'predict_cost_7d_2':   output_col[5].flatten(),
        'predict_cost_7d_3':   output_col[6].flatten(),
        'predict_cost_7d_4':   output_col[7].flatten(),
        'label_cost_7d':       output_col[8].flatten(),
        'label':               output_col[9].flatten(),
        'real_treatment':      output_col[-1].flatten()
    }
    df = pd.DataFrame(data)

    if not valid_distribution:
        ite_columns = ['predict_delta_cvr_1', 'predict_delta_cvr_2', 'predict_delta_cvr_3', 'predict_delta_cvr_4']
        print(f'\n{"="*70}\n[{epoch_str} {mode_str}] 指标评估\n{"="*70}')
        
        print(f'\n[AUUC 计算]')
        auuc = draw_auuc('', df,
            treatments=[1, 2, 3, 4],
            ite_names=ite_columns,
            label_col='label',
            real_treatment='real_treatment',
            control_group_name=[0, 0, 0, 0],
            show_curve=0,
            verbose=True
        )

        print(f'\n[AUUC 结果]')
        auuc_values = []
        for i, item in enumerate(auuc):
            key = ite_columns[i]
            auuc_value = item[key]
            auuc_values.append(auuc_value)
            print(f'auuc_{test}_{key}: {auuc_value}')
        mean_auuc = sum(auuc_values) / len(auuc_values) if auuc_values else 0
        print(f'mean_auuc_{test}: {mean_auuc}')

        skip_auc = getattr(args, 'skip_auc', False)
        if not skip_auc:
            print(f'\n[AUC 计算 (Cost)] - 使用 xAUC (归并排序, O(n log n))')
            auc = []
            cost_columns = ['predict_cost_7d_1', 'predict_cost_7d_2', 'predict_cost_7d_3', 'predict_cost_7d_4']
            for i, predict_cost in zip([1, 2, 3, 4], cost_columns):
                treat_i = df[df['real_treatment'] == i]
                auc_i = xauc_score(treat_i['label_cost_7d'], treat_i[predict_cost])
                auc.append(auc_i)
                print(f'  auc_{test}_{predict_cost}: {auc_i}')
        else:
            print(f'\n[AUC 计算] 已跳过 (--skip_auc)')
        
        print(f'[{get_beijing_time_str()}] [{epoch_str} {mode_str}] 指标评估完成')
        print(f'{"="*70}\n')

    if train:
        return df, mean_auuc, loss1
    if valid_distribution:
        return df
    if not valid_distribution and not train:
        return df, mean_auuc


def evaluate_pcoc_and_cal_n(df, args):
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_log', args.version)
    os.makedirs(plot_dir, exist_ok=True)
    evaluator = PCOCEvaluator()
    treatment_control_pairs = [(1, 0, 'predict_delta_cvr_1'), (2, 0, 'predict_delta_cvr_2'), (3, 0, 'predict_delta_cvr_3'), (4, 0, 'predict_delta_cvr_4')]
    pcoc_results, cal_n_results = [], []

    for treatment, control, pred_col in treatment_control_pairs:
        subset = df[df['real_treatment'].isin([treatment, control])].copy()
        if len(subset) == 0: continue
        try:
            pcoc_stats = evaluator.calculate_pcoc(pdf=subset, pred_col=pred_col, label_col='label', group_col='real_treatment', treatment_group=treatment, control_group=control)
            pcoc_results.append({'pred_col': pred_col, 'treatment': treatment, 'control': control, **pcoc_stats})
        except Exception: pass
        try:
            cal_n, binned_df = evaluator.calculate_cal_n(pdf=subset, pred_col=pred_col, label_col='label', group_col='real_treatment', treatment_group=treatment, control_group=control, n_bins=10)
            cal_n_results.append({'pred_col': pred_col, 'treatment': treatment, 'control': control, 'cal_n': cal_n})
        except Exception: pass

    return pcoc_results, cal_n_results

def valid_distribution(args):
    model = tf.keras.models.load_model(args.model_save_path)
    test_dataset = load_from_tfr("test", args)
    df = predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=True)
    pcoc_results, cal_n_results = evaluate_pcoc_and_cal_n(df, args)

def evaluation_train(args):
    model = tf.keras.models.load_model(args.model_save_path)
    test_dataset = load_from_tfr("test", args)
    df, mean_auuc = predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=False)
    print('test_mean_auuc:', mean_auuc)
    ite_columns = ['predict_delta_cvr_1', 'predict_delta_cvr_2', 'predict_delta_cvr_3', 'predict_delta_cvr_4']
    for ite_column in ite_columns:
        stats_new75 = simple_gap_delta_cvr(df, prob_col=ite_column, label_col='label')


def parse_args():
    parser = argparse.ArgumentParser(description="Uplift modeling parameters (V2)")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--PADDING_SIZE', type=int, default=446) # 稀疏特征槽位，一条样本最多放多少个特征
    parser.add_argument('--DENSE_FEA_SIZE', type=int, default=3075) #全局一共
    parser.add_argument('--EMBEDDING_SIZE', type=int, default=8) #每个embedding的维度
    parser.add_argument('--BATCH_SIZE', type=int, default=2048)
    parser.add_argument('--LEARNING_RATE', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--train_start_dt', type=str, default="20250106")
    parser.add_argument('--train_end_dt', type=str, default="20250106")
    parser.add_argument('--test_dt', type=str, default="20250118")
    parser.add_argument('--version', type=str, default="euen_4t_negative_downsample")
    parser.add_argument('--hc_dim', type=int, default=256)
    parser.add_argument('--hu_dim', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=2048)
    parser.add_argument('--treats_num', type=int, default=4) # V2 固定为 4 个实验组 (1, 2, 3, 4)
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--act_type', type=str, default='relu')
    parser.add_argument('--skip_eval', default=False, action='store_true', help='跳过验证集和测试集评估')
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--saveDir', type=str)
    parser.add_argument('--fromPretrained', default=False, action='store_true')
    parser.add_argument('--scheduleDt', type=str, default='')
    parser.add_argument('--startDate', type=str, default="20250102")
    parser.add_argument('--shuffle_size', type=int, default=20480) #把数据随机打乱顺序
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--label_size', type=int, default=1)
    parser.add_argument('--token_size', type=int, default=15) # v5/v6 token 包含 exp_label，共 15 列
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--debug_mode', default=False, action='store_true')
    parser.add_argument('--debug_batches', type=int, default=10)
    parser.add_argument('--debug_token_check', default=False, action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--skip_auc', default=False, action='store_true')
    parser.add_argument('--train_tfr_path', type=str, required=True)
    parser.add_argument('--valid_tfr_path', type=str, required=True)
    parser.add_argument('--test_tfr_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--parse_mode', type=str, default='batch_then_map', choices=['batch_then_map', 'map_then_batch'])
    parser.add_argument(
        '--negative_keep_probs',
        type=float,
        nargs='+',
        default=[1.0],
        help='依次训练的全负用户保留比例列表，例如 0.25 0.5 0.75 1.0。',
    )
    parser.add_argument(
        '--negative_sample_seed',
        type=int,
        default=2026,
        help='与 pid 一起用于稳定用户级抽样的随机种子。',
    )

    args = parser.parse_args()
    for keep_prob in args.negative_keep_probs:
        if not 0.0 <= keep_prob <= 1.0:
            parser.error('--negative_keep_probs values must be in [0, 1]')
    if args.negative_sample_seed < 0:
        parser.error('--negative_sample_seed must be non-negative')
    if args.token_size <= PID_INDEX:
        parser.error('--token_size does not contain the pid token')
    return args


def keep_prob_suffix(keep_prob):
    percent = keep_prob * 100
    percent_text = (
        str(int(round(percent)))
        if abs(percent - round(percent)) < 1e-9
        else f"{percent:.4f}".rstrip('0').rstrip('.').replace('.', 'p')
    )
    return f"neg{percent_text}"


def reset_training_seed():
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def run_one_experiment(args):
    global optimizer
    reset_training_seed()
    print("\n" + "=" * 80)
    print(
        f"Start experiment: keep_prob={args.negative_keep_prob}, "
        f"version={args.version}, model_path={args.model_save_path}"
    )
    print("=" * 80)

    train_dataset = load_from_tfr("train", args)
    print("Train dataset loaded with configured negative-user sampling")
    train_eval_dataset = load_from_tfr("train_eval", args)
    print("Train eval dataset loaded (fixed order, no shuffle)")
    
    if not args.skip_eval:
        valid_dataset = load_from_tfr("valid", args)
        print("Valid dataset loaded")
    else:
        valid_dataset = None
        print("Valid dataset loading skipped (--skip_eval 模式)")

    # 模型构建
    model = EUEN(args)
    input_len = args.PADDING_SIZE * 2  # 自动算出物理长度 (448 * 2 = 896)
    model.build(input_shape=[None, input_len])
    inputs = tf.keras.Input(shape=(input_len,), name='input_feature')
    outputs = model(inputs)
    
    # 变更为 4 个 uplift 和 4 个 cost (对应 1, 2, 3, 4)
    output_names = ['control', 'control_logit_fix', 'uplift_1', 'uplift_2', 'uplift_3', 'uplift_4', 'cost_1', 'cost_2', 'cost_3', 'cost_4']
    named_outputs = []
    for i, output in enumerate(outputs):
        named_output = tf.keras.layers.Lambda(lambda x: x, name=output_names[i])(output)
        named_outputs.append(named_output)
        
    model = tf.keras.Model(inputs=inputs, outputs=named_outputs)
    print("Model built")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.LEARNING_RATE)
    num_epochs = args.num_epochs
    best_mean_auuc = -float('inf')
    best_epoch = 0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        print(f'\n[{get_beijing_time_str()}] ========== Epoch {epoch+1} 开始 ==========')
        
        _, _, total_loss = predict_or_train_data_auuc(
            args, model, train_dataset, test='train', train=True,
            valid_distribution=False, epoch_num=epoch+1, collect_predictions=False
        )
        
        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 训练完成')
        print(f'Epoch {epoch+1}, Total Loss: {total_loss}')

        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 训练集最终模型评估开始')
        df_train_eval, mean_auuc_train_eval = predict_or_train_data_auuc(
            args, model, train_eval_dataset, test='train_eval', train=False,
            valid_distribution=False, epoch_num=epoch+1
        )
        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 训练集最终模型评估完成')
        print('final_train_mean_auuc:', mean_auuc_train_eval)
        del df_train_eval

        if not args.skip_eval:
            df_val, mean_auuc_val = predict_or_train_data_auuc(args, model, valid_dataset, test='val', train=False, valid_distribution=False, epoch_num=epoch+1)
            print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 验证完成')
            mean_auuc_for_stop = mean_auuc_val
            print('early_stop_mean_auuc:', mean_auuc_for_stop)
        else:
            mean_auuc_for_stop = float(epoch)
            print(f'[{get_beijing_time_str()}] 验证集评估阶段已跳过 (--skip_eval 模式)')

        epoch_model_path = f"{args.model_save_path}_epoch{epoch+1}"
        model.save(epoch_model_path)
        print(f"Epoch {epoch+1} model saved to: {epoch_model_path}")

        if mean_auuc_for_stop > best_mean_auuc:
            best_mean_auuc = mean_auuc_for_stop
            best_epoch = epoch + 1
            no_improvement_count = 0
            model.save(args.model_save_path)
            print(f"Best model updated at epoch {epoch+1}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.patience:
                print(f'Early stopping at Epoch {epoch + 1} with best_mean_AUUC: {best_mean_auuc}')
                break

    print(f"All epoch models saved to: {args.model_save_path}_epochX")
    print(f"Best model saved to: {args.model_save_path}")
    
    # 🌟 修改部分在这里：补上最终的 Best 判断指标总结打印
    print(f'Best mean AUUC_val: {best_mean_auuc}')
    print(f'Best epoch: {best_epoch}')
    print(f'Now epoch: {epoch + 1}')

    if not args.skip_eval:
        evaluation_train(args)
        valid_distribution(args)
    else:
        print("\n" + "="*80)
        print("因为使用了 --skip_eval 参数，测试集流程已跳过。")
        print("="*80 + "\n")

    print(
        f"Finished experiment: keep_prob={args.negative_keep_prob}, "
        f"best_epoch={best_epoch}, best_mean_auuc={best_mean_auuc}"
    )


if __name__ == "__main__":
    base_args = parse_args()
    setup_gpu(base_args.gpu)
    base_model_save_path = base_args.model_save_path.rstrip('/')
    base_version = base_args.version

    print("Parsed arguments:")
    print(base_args)
    print(f"Negative keep probabilities: {base_args.negative_keep_probs}")

    for keep_prob in base_args.negative_keep_probs:
        args = argparse.Namespace(**vars(base_args))
        args.negative_keep_prob = keep_prob
        suffix = keep_prob_suffix(keep_prob)
        args.model_save_path = f"{base_model_save_path}_{suffix}"
        args.version = f"{base_version}_{suffix}"
        run_one_experiment(args)
        tf.keras.backend.clear_session()