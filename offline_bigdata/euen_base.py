

"""
nohup python euen_base.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/train_20250601_20250831/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/test_20260401_20260412/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/test_20260401_20260412/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wh/demo/multi_model/euen/model2/base" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_model_base2"  \
--DENSE_FEA_SIZE 3100 \
--shuffle_size 0 \
--gpu 2 > ./run_logs/euen_model_base2.log 2>&1 &

"""
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

"""
执行代码示例1：
nohup python euen_zlin.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/train_20250106_20250302/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/valid_20251101/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/valid_20251101/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wh/multi_model/euen" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_data_expand"  \
--shuffle_size 0 \
--gpu 2 > ./run_logs/euen_data_expand.log 2>&1 &
"""


fextract_path = os.path.abspath("/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wh/demo/offline_bigdata/python")
sys.path.append(fextract_path)
from fextract import LiteFeTFRecords
from fextract import IndexEmbedding




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


class IndexEmbedding(tf.keras.layers.Embedding):
    def __init__(self,
                 dense_fea_dim,
                 embedding_dim,
                 sparse_fea_dim,
                 embeddings_initializer="uniform",
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
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
    """
    TFRecord 数据读取器
    
    解析模式 (parse_mode):
        - 'batch_then_map': 先 batch 再解析（推荐，更高效）
        - 'map_then_batch': 先解析再 batch（传统方式）
    """
    def __init__(self,
                 hdfs_pattern,
                 start_dt,
                 end_dt,
                 label_size=-1,
                 token_size=-1,
                 dt_fmt="%Y%m%d",
                 parse_mode='batch_then_map'):
        """
        Args:
            hdfs_pattern: 文件路径模式（支持本地路径和HDFS路径）
            start_dt: 开始日期
            end_dt: 结束日期
            label_size: label 维度
            token_size: token 维度
            dt_fmt: 日期格式
            parse_mode: 解析模式 ('batch_then_map' 或 'map_then_batch')
        """
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

    def get_dataset(self,
                    epoch=1,
                    batch_size=1,
                    parallel=1,
                    shuffle_size=0,
                    shuffle_seed=None,
                    drop_remainder=True,
                    return_token=False,
                    token_dtype=str):

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
            """Parse a single TFRecord example (map-then-batch pattern)"""
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

        def _parse_batch_example(serialized_batch):
            """Parse a batch of serialized TFRecord examples (batch-then-map pattern)"""
            feature_map = tf.io.parse_example(serialized_batch, feature_dict)
            
            # Parse labels - handle RaggedTensor from batch string split
            _labels = tf.strings.split(feature_map["label"][:, 0], ",")
            _labels = tf.strings.to_number(_labels, tf.float32)
            label = _labels.to_tensor()
            label = tf.reshape(label, [-1, self.label_size])

            # Parse features - handle RaggedTensor from batch string split
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
        
        # Step 1: 创建原始数据集（顺序读取）
        dataset = tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel
        )
        
        # Step 2: Shuffle（在 batch 之前，实现样本级别的打散）
        if shuffle_size > 0:
            dataset = dataset.shuffle(
                buffer_size=shuffle_size,
                seed=shuffle_seed,
                reshuffle_each_iteration=True
            )
        
        # Step 3: 根据 parse_mode 选择解析方式
        if self.parse_mode == 'batch_then_map':
            # 先 batch 再解析（更高效）
            dataset = dataset.batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder
            ).map(
                map_func=_parse_batch_example,
                num_parallel_calls=parallel
            )
        else:
            # 先解析再 batch（传统方式）
            dataset = dataset.map(
                map_func=_parse_single_example,
                num_parallel_calls=parallel
            ).batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        
        # Step 4: repeat 和 prefetch
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
        mean_label_888 = sum_label_888 / count_888
        mean_label_555 = sum_label_555 / count_555
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


def draw_auuc(path, prediction_pdf, treatments, ite_names, label_col='pv_label', 
              real_treatment='real_treatment', control_group_name='control_group', show_curve=False, verbose=True):
    """
    计算 AUUC 指标
    """
    metric_results = []
    
    for i, (treatment, ite_name, control_group) in enumerate(zip(treatments, ite_names, control_group_name)):
        # 数据过滤
        subset = prediction_pdf[prediction_pdf[real_treatment].isin([treatment, control_group])]
        subset['exp_group'] = subset[real_treatment].apply(lambda x: 1 if x != control_group else 0)
        subset['label'] = subset[label_col]

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

# caculate_auc 计算的是：
# AUC = (concordant pairs) / (所有不同标签的样本对)

# xauc_score 计算的是：
# xAUC = (逆序对数) / (所有样本对数)
# xauc_score 的分母包含了 相同标签 的样本对，而 caculate_auc 排除了这些
# 两者计算的是 互补关系：caculate_auc ≈ 1 - xauc_score（在忽略相同标签对的情况下）

import math

class InversePairsCalc:
    """使用归并排序计算逆序对数，时间复杂度 O(n log n)"""
    
    def InversePairs(self, data):
        if not data:
            return 0
        if len(data) == 1:
            return 0
        
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
    """
    使用归并排序计算多分类 AUC (xAUC)
    时间复杂度: O(n log n)
    
    Args:
        y_labels: 真实标签 (可以是 list, numpy array, 或 pandas Series)
        y_scores: 预测分数 (可以是 list, numpy array, 或 pandas Series)
        verbose_name: 如果提供，会输出详细的耗时信息
    
    Returns:
        xAUC 分数 (逆序对比例)
    """
    # 转换为 numpy array
    if hasattr(y_labels, 'values'):
        y_labels = y_labels.values
    if hasattr(y_scores, 'values'):
        y_scores = y_scores.values
    
    y_labels = np.array(y_labels).reshape(-1)
    y_scores = np.array(y_scores).reshape(-1)
    
    n = len(y_labels)
    if n <= 1:
        return 0.0
    
    # Step 1: 按 score 降序排序
    label_preds = list(zip(y_labels, y_scores))
    sorted_label_preds = sorted(label_preds, key=lambda lc: lc[1], reverse=True)
    labels_sort = [ele[0] for ele in sorted_label_preds]
    
    # Step 2: 计算逆序对数
    S = InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    
    # Step 3: 计算总对数
    pairs_cnt = n * (n - 1) / 2
    
    # 计算 xAUC
    xauc = total_positive / pairs_cnt if pairs_cnt > 0 else 0.0
    
    return xauc




def load_from_tfr(train_type, args):
    shuffle_seed = getattr(args, 'shuffle_seed', None)
    parse_mode = getattr(args, 'parse_mode', 'batch_then_map')
    
    if train_type == "train":
        reader = LiteFeTFRecords(
            hdfs_pattern=args.train_tfr_path, 
            start_dt=args.train_start_dt, 
            end_dt=args.train_end_dt, 
            token_size=args.token_size, 
            label_size=args.label_size,
            parse_mode=parse_mode
        )
        dataset = reader.get_dataset(
            epoch=1, 
            batch_size=args.BATCH_SIZE, 
            parallel=args.parallel, 
            shuffle_size=args.shuffle_size, 
            shuffle_seed=shuffle_seed, 
            token_dtype=str, 
            drop_remainder=False, 
            return_token=True
        )
    elif train_type == "test":
        reader = LiteFeTFRecords(
            hdfs_pattern=args.test_tfr_path, 
            start_dt=args.test_dt, 
            end_dt=args.test_dt, 
            token_size=args.token_size, 
            label_size=args.label_size,
            parse_mode=parse_mode
        )
        dataset = reader.get_dataset(
            epoch=1, 
            batch_size=args.BATCH_SIZE, 
            parallel=args.parallel, 
            token_dtype=str, 
            drop_remainder=False, 
            return_token=True
        )
    else:
        reader = LiteFeTFRecords(
            hdfs_pattern=args.valid_tfr_path, 
            start_dt=args.test_dt, 
            end_dt=args.test_dt, 
            token_size=args.token_size, 
            label_size=args.label_size,
            parse_mode=parse_mode
        )
        dataset = reader.get_dataset(
            epoch=1, 
            batch_size=args.BATCH_SIZE, 
            parallel=args.parallel, 
            token_dtype=str, 
            drop_remainder=False, 
            return_token=True
        )

    # Debug mode: 限制数据量
    if args.debug_mode:
        dataset = dataset.take(args.debug_batches)
        print(f"[DEBUG MODE] {train_type} dataset limited to {args.debug_batches} batches")

    return dataset


class EUEN(tf.keras.Model):
    def __init__(self, args):
        super(EUEN, self).__init__()
        self.treats_num = args.treats_num
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.bias_initializer = tf.keras.initializers.Constant(value=0)
        self.kernel_regularizer = tf.keras.regularizers.l2(args.l2_reg)

        self.embedding_layer = IndexEmbedding(dense_fea_dim=args.DENSE_FEA_SIZE, embedding_dim=args.EMBEDDING_SIZE, sparse_fea_dim=args.PADDING_SIZE)
        self.activation = [args.act_type, args.act_type, args.act_type]

        self.share_bottom = [tf.keras.layers.Dense(units=unit, activation=act, kernel_initializer=self.kernel_initializer, 
                             bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                             for unit, act in zip([args.emb_dim, args.emb_dim // 2, args.emb_dim // 4], self.activation)]

        self.c_head = [tf.keras.layers.Dense(units=hc_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                       bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                       for hc_dim, act in zip([args.hc_dim, args.hc_dim//2, args.hc_dim//4, 1], self.activation + [None])]

        self.u_head = [[tf.keras.layers.Dense(units=hu_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                        bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                        for hu_dim, act in zip([args.hu_dim, args.hu_dim//2, args.hu_dim//4, 1], self.activation + [None])] 
                       for _ in range(self.treats_num)]

        self.cost_head = [[tf.keras.layers.Dense(units=hu_dim, activation=act, kernel_initializer=self.kernel_initializer, 
                           bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                           for hu_dim, act in zip([args.hu_dim, args.hu_dim//2, args.hu_dim//4, 1], self.activation + [None])] 
                          for _ in range(self.treats_num)]

    def call(self, inputs):
        input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
        input_value = tf.cast(inputs[:, 1::2], tf.float32)
        inputs = self.embedding_layer(input_index[:, :])

        inputs = reduce(lambda x, layer: layer(x), self.share_bottom, inputs)

        c_head = reduce(lambda x, c_task: c_task(x), self.c_head, inputs)
        u_head = [reduce(lambda x, u_task: u_task(x), self.u_head[i], inputs) for i in range(self.treats_num)]
        cost_head = [reduce(lambda x, cost: cost(x), self.cost_head[i], inputs) for i in range(self.treats_num)]

        c_head_logit_fix = tf.stop_gradient(c_head)
        return c_head, c_head_logit_fix, u_head[0], u_head[1], u_head[2], u_head[3], cost_head[0], cost_head[1], cost_head[2], cost_head[3]



def predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=False, epoch_num=None):
    uplift_scores_list_1 = []
    uplift_scores_list_4 = []
    uplift_scores_list_5 = []
    uplift_scores_list_6 = []
    scores_cost_7d_1_list = []
    scores_cost_7d_4_list = []
    scores_cost_7d_5_list = []
    scores_cost_7d_6_list = []
    label_cost_7d_list = []
    label_list = []
    real_treatment_list = []
    batch_count = 0
    
    mode_str = 'train' if train else test
    epoch_str = f"Epoch {epoch_num}" if epoch_num is not None else ""
    
    # 进度日志配置
    log_interval = getattr(args, 'log_interval', 100)  # 每 N 个 batch 打印一次进度
    epoch_start_time = time.time()
    
    # Debug mode: 打印样本信息的配置
    debug_print_batches = 3  # 打印前几个 batch
    debug_print_samples = 5  # 每个 batch 打印前几个样本

    for x, y, t in test_dataset:
        inputs = x
        is_treat = t[:, -5]
        is_treat = tf.strings.to_number(is_treat, tf.float32)
        label = tf.squeeze(y)
        label_cost_7d = t[:, -1]
        label_cost_7d = tf.strings.to_number(label_cost_7d, tf.float32)
        label_cost_7d = tf.squeeze(label_cost_7d)
        
        # Debug mode: 打印前几个 batch 的样本信息
        if args.debug_mode and batch_count < debug_print_batches:
            print(f"\n[DEBUG] {epoch_str} {mode_str} Batch {batch_count}:")
            for j in range(min(debug_print_samples, t.shape[0])):
                token_vals = t[j].numpy()
                token_str = ",".join([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in token_vals])
                label_val = y[j].numpy()[0] if len(y[j].numpy()) > 0 else 'N/A'
                print(f"  Sample {j}: label={label_val:.0f}, token=[{token_str}]")

        if train:
            with tf.GradientTape() as tape:
                uc, c_logit_fix, u_tau_1, u_tau_4, u_tau_5, u_tau_6, cost_1, cost_4, cost_5, cost_6 = model(inputs)
                t_logit_6_fix = tf.stop_gradient(u_tau_6)
                is_treat = tf.reshape(is_treat, [-1, 1])
                label = tf.reshape(label, [-1, 1])
                label_cost_7d = tf.reshape(label_cost_7d, [-1, 1])
                is_list = [tf.cast((is_treat == treat), tf.float32) for treat in [0, 1, 4, 5, 6]]

                ut = [(c_logit_fix + t_logit_6_fix + tau) for tau in [u_tau_1, u_tau_4, u_tau_5]]
                ut_6 = (c_logit_fix + u_tau_6)
                loss1 = tf.reduce_mean(tf.square(is_list[0] * uc + is_list[1] * ut[0] + is_list[2] * ut[1] + is_list[3] * ut[2] + is_list[4] * ut_6 - label)
                                       + tf.square(is_list[1] * cost_1 + is_list[2] * cost_4 + is_list[3] * cost_5 + is_list[4] * cost_6 - label_cost_7d))
                total_loss = loss1
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            _, _, u_tau_1, u_tau_4, u_tau_5, u_tau_6, cost_1, cost_4, cost_5, cost_6 = model(inputs)

        # 复原真正的 delta_cvr：将相对于 treatment=6 的增量加上 u_tau_6
        delta_cvr_1 = u_tau_1 + u_tau_6
        delta_cvr_4 = u_tau_4 + u_tau_6
        delta_cvr_5 = u_tau_5 + u_tau_6
        delta_cvr_6 = u_tau_6

        lists = [uplift_scores_list_1, uplift_scores_list_4, uplift_scores_list_5, uplift_scores_list_6,
                 scores_cost_7d_1_list, scores_cost_7d_4_list, scores_cost_7d_5_list, scores_cost_7d_6_list, 
                 label_cost_7d_list, label_list, real_treatment_list]
        outs = [delta_cvr_1, delta_cvr_4, delta_cvr_5, delta_cvr_6, cost_1, cost_4, cost_5, cost_6, 
                label_cost_7d, label, is_treat]
        for list_name, value_name in zip(lists, outs):
            list_name.append(value_name.numpy())
        
        batch_count += 1
        
        # 每隔 log_interval 个 batch 打印进度
        if batch_count % log_interval == 0:
            elapsed = time.time() - epoch_start_time
            samples_processed = batch_count * args.BATCH_SIZE
            throughput = samples_processed / elapsed if elapsed > 0 else 0
            
            if train:
                print(f'[{epoch_str} {mode_str}] Batch {batch_count}: {samples_processed} samples, '
                      f'{throughput:.0f} samples/sec | loss={loss1.numpy():.6f}')
            else:
                print(f'[{epoch_str} {mode_str}] Batch {batch_count}: {samples_processed} samples, '
                      f'{throughput:.0f} samples/sec')
    
    # 打印 epoch 级别统计
    total_epoch_time = time.time() - epoch_start_time
    throughput = batch_count * args.BATCH_SIZE / total_epoch_time if total_epoch_time > 0 else 0
    
    print(f'\n{"="*70}')
    print(f'[{epoch_str} {mode_str}] Epoch Summary')
    print(f'{"="*70}')
    print(f'Total batches: {batch_count}, Total time: {total_epoch_time:.2f}s')
    print(f'Throughput: {throughput:.0f} samples/sec')
    print(f'{"="*70}\n')
    
    if train:
        print('loss1:', loss1)

    out_lists = [uplift_scores_list_1, uplift_scores_list_4, uplift_scores_list_5, uplift_scores_list_6,
                 scores_cost_7d_1_list, scores_cost_7d_4_list, scores_cost_7d_5_list, scores_cost_7d_6_list, 
                 label_cost_7d_list, label_list, real_treatment_list]
    output_col = [np.concatenate(list_name, axis=0) for list_name in out_lists]

    data = {
        'predict_delta_cvr_1': output_col[0].flatten(),
        'predict_delta_cvr_4': output_col[1].flatten(),
        'predict_delta_cvr_5': output_col[2].flatten(),
        'predict_delta_cvr_6': output_col[3].flatten(),
        'predict_cost_7d_1':   output_col[4].flatten(),
        'predict_cost_7d_4':   output_col[5].flatten(),
        'predict_cost_7d_5':   output_col[6].flatten(),
        'predict_cost_7d_6':   output_col[7].flatten(),
        'label_cost_7d':       output_col[8].flatten(),
        'label':               output_col[9].flatten(),
        'real_treatment':      output_col[-1].flatten()
    }
    df = pd.DataFrame(data)

    if not valid_distribution:
        # 修改：所有treatment都与control组(treatment=0)比较
        ite_columns = ['predict_delta_cvr_1', 'predict_delta_cvr_4', 'predict_delta_cvr_5', 'predict_delta_cvr_6']

        print(f'\n{"="*70}')
        print(f'[{epoch_str} {mode_str}] 指标评估')
        print(f'{"="*70}')
        
        # AUUC 计算
        print(f'\n[AUUC 计算]')
        auuc = draw_auuc('', df,
            treatments=[1, 4, 5, 6],
            ite_names=ite_columns,
            label_col='label',
            real_treatment='real_treatment',
            control_group_name=[0, 0, 0, 0],  # 修改：所有treatment都与control组(treatment=0)比较
            show_curve=0,
            verbose=True
        )

        print(f'\n[AUUC 结果]')
        for i, item in enumerate(auuc):
            key = ite_columns[i]
            print(f'auuc_{test}_{key}: {item[key]}')
            if key == 'predict_delta_cvr_6':
                auuc_val_left63 = item[key]

        # AUC 计算（可通过 --skip_auc 跳过）
        skip_auc = getattr(args, 'skip_auc', False)
        if not skip_auc:
            print(f'\n[AUC 计算 (Cost)] - 使用 xAUC (归并排序, O(n log n))')
            auc = []
            cost_columns = ['predict_cost_7d_1', 'predict_cost_7d_4', 'predict_cost_7d_5', 'predict_cost_7d_6']
            for i, predict_cost in zip([1, 4, 5, 6], cost_columns):
                treat_i = df[df['real_treatment'] == i]
                auc_i = xauc_score(treat_i['label_cost_7d'], treat_i[predict_cost])
                auc.append(auc_i)
                print(f'  auc_{test}_{predict_cost}: {auc_i}')
        else:
            print(f'\n[AUC 计算] 已跳过 (--skip_auc)')
        
        print(f'[{get_beijing_time_str()}] [{epoch_str} {mode_str}] 指标评估完成')
        print(f'{"="*70}\n')

    if train:
        return df, auuc_val_left63, loss1
    if valid_distribution:
        return df
    if not valid_distribution and not train:
        return df, auuc_val_left63


def evaluate_pcoc_and_cal_n(df, args):
    """计算PCOC和Cal-N指标，并保存可视化图片"""
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_log', args.version)
    os.makedirs(plot_dir, exist_ok=True)

    evaluator = PCOCEvaluator()

    # 定义treatment和control的对应关系（所有treatment都与control组0比较）
    treatment_control_pairs = [
        (1, 0, 'predict_delta_cvr_1'),
        (4, 0, 'predict_delta_cvr_4'),
        (5, 0, 'predict_delta_cvr_5'),
        (6, 0, 'predict_delta_cvr_6'),
    ]

    pcoc_results = []
    cal_n_results = []

    for treatment, control, pred_col in treatment_control_pairs:
        subset = df[df['real_treatment'].isin([treatment, control])].copy()

        if len(subset) == 0:
            print(f"[WARNING] {pred_col}: 无数据，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"评估 {pred_col} (Treatment={treatment}, Control={control})")
        print(f"{'='*60}")

        # 计算PCOC
        try:
            pcoc_stats = evaluator.calculate_pcoc(
                pdf=subset,
                pred_col=pred_col,
                label_col='label',
                group_col='real_treatment',
                treatment_group=treatment,
                control_group=control
            )
            pcoc_results.append({
                'pred_col': pred_col,
                'treatment': treatment,
                'control': control,
                **pcoc_stats
            })
        except Exception as e:
            print(f"[ERROR] PCOC计算失败: {e}")

        # 计算Cal-N
        try:
            cal_n, binned_df = evaluator.calculate_cal_n(
                pdf=subset,
                pred_col=pred_col,
                label_col='label',
                group_col='real_treatment',
                treatment_group=treatment,
                control_group=control,
                n_bins=10
            )
            cal_n_results.append({
                'pred_col': pred_col,
                'treatment': treatment,
                'control': control,
                'cal_n': cal_n
            })

            if binned_df is not None and len(binned_df) > 0:
                plot_pcoc_by_bin(binned_df, pred_col, treatment, control, plot_dir, args.test_dt)

        except Exception as e:
            print(f"[ERROR] Cal-N计算失败: {e}")

    print_summary(pcoc_results, cal_n_results)
    plot_summary(pcoc_results, cal_n_results, plot_dir, args.test_dt)

    return pcoc_results, cal_n_results


def plot_pcoc_by_bin(binned_df, pred_col, treatment, control, plot_dir, test_dt):
    """绘制分桶PCOC图"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1: Predicted vs Actual Uplift
    ax1 = axes[0]
    x = binned_df['bin'].values
    ax1.bar(x - 0.2, binned_df['predicted_uplift'], width=0.4, label='Predicted Uplift', alpha=0.8)
    ax1.bar(x + 0.2, binned_df['actual_uplift'], width=0.4, label='Actual Uplift', alpha=0.8)
    ax1.set_xlabel('Bin')
    ax1.set_ylabel('Uplift')
    ax1.set_title(f'Predicted vs Actual Uplift\n{pred_col}')
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 图2: PCOC by Bin
    ax2 = axes[1]
    pcoc_values = binned_df['pcoc'].replace([np.inf, -np.inf], np.nan)
    ax2.bar(x, pcoc_values, alpha=0.8, color='green')
    ax2.axhline(y=1, color='red', linestyle='--', label='PCOC=1 (理想值)')
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('PCOC')
    ax2.set_title(f'PCOC by Bin\n{pred_col}')
    ax2.legend()

    # 图3: Symmetric Error by Bin
    ax3 = axes[2]
    sym_errors = binned_df['symmetric_error'].replace([np.inf, -np.inf], np.nan)
    ax3.bar(x, sym_errors, alpha=0.8, color='orange')
    ax3.set_xlabel('Bin')
    ax3.set_ylabel('Symmetric Error')
    ax3.set_title(f'Symmetric Error by Bin\n{pred_col}')

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'pcoc_bin_{pred_col}_t{treatment}_c{control}_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 分桶PCOC图已保存: {save_path}")


def plot_summary(pcoc_results, cal_n_results, plot_dir, test_dt):
    """绘制汇总图"""
    if not pcoc_results:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: PCOC汇总
    ax1 = axes[0]
    labels = [r['pred_col'].replace('predict_delta_cvr_', 'T') for r in pcoc_results]
    pcoc_values = [r['pcoc'] for r in pcoc_results]
    colors = ['green' if 0.8 <= p <= 1.2 else 'orange' if 0.5 <= p <= 1.5 else 'red' for p in pcoc_values]
    bars = ax1.bar(labels, pcoc_values, color=colors, alpha=0.8)
    ax1.axhline(y=1, color='red', linestyle='--', label='PCOC=1')
    ax1.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(y=1.2, color='orange', linestyle=':', alpha=0.7)
    ax1.set_ylabel('PCOC')
    ax1.set_title(f'PCOC Summary ({test_dt})')
    ax1.legend()
    for bar, val in zip(bars, pcoc_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 图2: Cal-N汇总
    ax2 = axes[1]
    labels = [r['pred_col'].replace('predict_delta_cvr_', 'T') for r in cal_n_results]
    cal_n_values = [r['cal_n'] for r in cal_n_results]
    colors = ['green' if c < 0.3 else 'orange' if c < 0.5 else 'red' for c in cal_n_values]
    bars = ax2.bar(labels, cal_n_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Cal-N')
    ax2.set_title(f'Cal-N Summary ({test_dt})')
    for bar, val in zip(bars, cal_n_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'pcoc_cal_n_summary_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 汇总图已保存: {save_path}")


def print_summary(pcoc_results, cal_n_results):
    """打印汇总结果"""
    print(f"\n{'='*80}")
    print("PCOC & Cal-N 评估汇总")
    print(f"{'='*80}")

    print("\n[PCOC Results]")
    print(f"{'Pred Column':<25} {'Treatment':<10} {'Control':<10} {'PCOC':<10} {'偏离度':<10}")
    print("-" * 65)
    for r in pcoc_results:
        deviation = abs(1 - r['pcoc'])
        print(f"{r['pred_col']:<25} {r['treatment']:<10} {r['control']:<10} {r['pcoc']:<10.4f} {deviation:<10.6f}")

    print("\n[Cal-N Results]")
    print(f"{'Pred Column':<25} {'Treatment':<10} {'Control':<10} {'Cal-N':<10}")
    print("-" * 55)
    for r in cal_n_results:
        print(f"{r['pred_col']:<25} {r['treatment']:<10} {r['control']:<10} {r['cal_n']:<10.6f}")

    print(f"\n{'='*80}\n")


def valid_distribution(args):
    model = tf.keras.models.load_model(args.model_save_path)

    test_dataset = load_from_tfr("test", args)
    df = predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=True)

    # PCOC & Cal-N 评估
    print("\n" + "="*80)
    print("开始 PCOC & Cal-N 评估...")
    print("="*80)
    pcoc_results, cal_n_results = evaluate_pcoc_and_cal_n(df, args)

    # 原有的分布分析
    ite_columns = ['predict_delta_cvr_1', 'predict_delta_cvr_4', 'predict_delta_cvr_5', 'predict_delta_cvr_6', 'predict_delta_cvr_6']
    for i, ite_column in enumerate(ite_columns):
        ite_column_binned = ite_column + '_binned'
        df[ite_column_binned] = pd.qcut(df[ite_column], q=10, labels=False, duplicates='drop')
        bucket_summary_df = pd.DataFrame()
        for bin in range(10):
            if i == 4:
                bin_data = df[(df['predict_delta_cvr_6_binned'] == bin) & (df['real_treatment'].isin([0, 6]))]
            else:
                bin_data = df[df[ite_column_binned] == bin]
            treatment_counts = bin_data['real_treatment'].value_counts().sort_index()
            proportions = treatment_counts / len(bin_data)
            proportions_df = proportions.rename('proportion').reset_index()
            proportions_df.columns = ['real_treatment', f'bin_{bin}']
            if bucket_summary_df.empty:
                bucket_summary_df = proportions_df
            else:
                bucket_summary_df = pd.merge(bucket_summary_df, proportions_df, on='real_treatment', how='outer')
        if i == 4:
            print('predict_delta_cvr_6:0,6')
        else:
            print(f'{ite_column}:')
        print(bucket_summary_df)

    df_with_index = bucket_summary_df.set_index('real_treatment')
    max_values = df_with_index.max(axis=1).reset_index()
    min_values = df_with_index.min(axis=1).reset_index()
    max_values.columns = ['real_treatment', 'max_proportion']
    min_values.columns = ['real_treatment', 'min_proportion']

    print('Max proportions for each treatment:')
    print(max_values)
    print('Min proportions for each treatment:')
    print(min_values)


def evaluation_train(args):
    model = tf.keras.models.load_model(args.model_save_path)

    plot_path = "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wh/demo/euen_part_code/plot_logs/{version}_at_{time_str}_gap".format(version=args.version, time_str=time_str)
    gain_path = "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wh/demo/euen_part_code/plot_logs/{version}_at_{time_str}_auuc".format(version=args.version, time_str=time_str)
    test_dataset = load_from_tfr("test", args)

    df, auuc_val_left63 = predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=False)

    average_auuc = auuc_val_left63
    print('average_auuc:', average_auuc)

    ite_columns = ['predict_delta_cvr_1', 'predict_delta_cvr_4', 'predict_delta_cvr_5', 'predict_delta_cvr_6']
    for ite_column, i in zip(ite_columns, [1, 4, 5, 6]):
        stats_new75 = simple_gap_delta_cvr(df, prob_col=ite_column, label_col='label')


def parse_args():
    parser = argparse.ArgumentParser(description="Uplift modeling parameters")

    parser.add_argument('--gpu', type=str, default="0", help='GPU device id to use, e.g., "0", "1", "2", "3" or "0,1" for multi-gpu')
    parser.add_argument('--PADDING_SIZE', type=int, default=446, help='表示稀疏特征的维度sparse_fea_dim')
    parser.add_argument('--DENSE_FEA_SIZE', type=int, default=3075, help='Input dimension')
    parser.add_argument('--EMBEDDING_SIZE', type=int, default=8, help='Dimension of embedding for each feature')
    parser.add_argument('--BATCH_SIZE', type=int, default=2048, help='Batch size for training')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--train_start_dt', type=str, default="20250106", help='Training start date')
    parser.add_argument('--train_end_dt', type=str, default="20250106", help='Training end date')
    parser.add_argument('--test_dt', type=str, default="20250118", help='Testing date')
    parser.add_argument('--version', type=str, default="multi_EUEN_uplift_cost_v213_010203", help='Model version')
    parser.add_argument('--hc_dim', type=int, default=256)
    parser.add_argument('--hu_dim', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=2048)
    parser.add_argument('--treats_num', type=int, default=4)
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--act_type', type=str, default='relu')
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--saveDir', type=str, help='output model saved directory, this name required')
    parser.add_argument('--fromPretrained', default=False, action='store_true', help='load model weights from EVE')
    parser.add_argument('--scheduleDt', type=str, required=False, default='', help='scheduleDt uses to access input table partition')
    parser.add_argument('--startDate', type=str, default="20250102", help='predict start date')
    parser.add_argument('--shuffle_size', type=int, default=20480, help='Shuffle buffer size for training data (0 to disable)')
    parser.add_argument('--shuffle_seed', type=int, default=0, help='Random seed for shuffle (None for random, set value for reproducibility)')
    parser.add_argument('--label_size', type=int, default=1, help='Label 维度')
    parser.add_argument('--token_size', type=int, default=7, help='Token 维度')
    parser.add_argument('--parallel', type=int, default=4, help='并行读取数')
    parser.add_argument('--debug_mode', default=False, action='store_true', help='Enable debug mode with reduced data')
    parser.add_argument('--debug_batches', type=int, default=10, help='Number of batches to use in debug mode')
    parser.add_argument('--log_interval', type=int, default=100, help='Print progress every N batches')
    parser.add_argument('--skip_auc', default=False, action='store_true', help='跳过AUC指标评估（AUC计算耗时较长，可用于加速训练验证）')
    # 路径相关参数（必需）
    parser.add_argument('--train_tfr_path', type=str, required=True, help='Path pattern for training TFRecords')
    parser.add_argument('--valid_tfr_path', type=str, required=True, help='Path pattern for validation TFRecords')
    parser.add_argument('--test_tfr_path', type=str, required=True, help='Path pattern for test TFRecords')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save/load model')

    # 数据解析模式参数
    parser.add_argument('--parse_mode', type=str, default='batch_then_map', choices=['batch_then_map', 'map_then_batch'],
                        help='解析模式: batch_then_map(先batch再解析,更高效) 或 map_then_batch(先解析再batch)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # 设置 GPU
    setup_gpu(args.gpu)

    print("Parsed arguments:")
    print(args)

    # 数据加载
    train_dataset = load_from_tfr("train", args)
    print("Train dataset loaded")
    
    valid_dataset = load_from_tfr("valid", args)
    print("Valid dataset loaded")

    # 模型构建
    model = EUEN(args)
    model.build(input_shape=[None, 892])
    inputs = tf.keras.Input(shape=(892,), name='input_feature')
    outputs = model(inputs)
    output_names = ['control', 'control_logit_fix', 'uplift_1', 'uplift_4', 'uplift_5', 'uplift_6', 'cost_1', 'cost_4', 'cost_5', 'cost_6']
    named_outputs = []
    for i, output in enumerate(outputs):
        named_output = tf.keras.layers.Lambda(lambda x: x, name=output_names[i])(output)
        named_outputs.append(named_output)
    model = tf.keras.Model(inputs=inputs, outputs=named_outputs)
    print("Model built")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.LEARNING_RATE)
    num_epochs = args.num_epochs

    best_average_auuc = -float('inf')
    best_epoch = 0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        # 每轮开始前输出北京时间
        print(f'\n[{get_beijing_time_str()}] ========== Epoch {epoch+1} 开始 ==========')
        
        df, _, total_loss = predict_or_train_data_auuc(args, model, train_dataset, test='train', train=True, valid_distribution=False, epoch_num=epoch+1)
        
        # train 数据集训练结束后输出北京时间
        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 训练完成')
        print(f'Epoch {epoch+1}, Total Loss: {total_loss}')
        negative_ratio_train = (df < 0).mean()
        print('negative_ratio_train:')
        print(negative_ratio_train)

        df_val, auuc_val_left63 = predict_or_train_data_auuc(args, model, valid_dataset, test='val', train=False, valid_distribution=False, epoch_num=epoch+1)
        
        # val 数据集评估结束后输出北京时间
        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 验证完成')
        negative_ratio = (df_val < 0).mean()
        print('negative_ratio_val:')
        print(negative_ratio)

        average_auuc = auuc_val_left63
        print('average_auuc:', average_auuc)

        # 每个 epoch 都保存模型，命名包含 epoch 编号
        epoch_model_path = f"{args.model_save_path}_epoch{epoch+1}"
        model.save(epoch_model_path)
        print(f"Epoch {epoch+1} model saved to: {epoch_model_path}")

        # 同时记录最优模型
        if average_auuc > best_average_auuc:
            best_average_auuc = average_auuc
            best_epoch = epoch + 1
            no_improvement_count = 0
            # 保存最优模型（不带 epoch 后缀）
            model.save(args.model_save_path)
            print(f"Best model updated at epoch {epoch+1}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.patience:
                print(f'Early stopping at Epoch {epoch + 1} with best_average_AUUC: {best_average_auuc}')
                break

    print(f"All epoch models saved to: {args.model_save_path}_epochX")
    print(f"Best model saved to: {args.model_save_path}")
    print(f'Best AUUC_val: {best_average_auuc}')
    print(f'Best epoch: {best_epoch}')
    print(f'Now epoch: {epoch + 1}')

    model = tf.keras.models.load_model(args.model_save_path)
    print('---------模型的签名如下--------')
    print(list(model.signatures.keys()))
    input_names = model.input_names
    if 'input_feature' in input_names:
        print("模型在输入层定义了 name='input_feature'")
    output_names = model.output_names
    for name in output_names:
        print(f"模型在输出层定义了: {name}")
    print(model.summary())

    evaluation_train(args)
    valid_distribution(args)