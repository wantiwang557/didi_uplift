
'''
Basic 4-treatment S-learner for WWT feature-selection experiments.

Example:

nohup python s_learner_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/train_20260511_20260617/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/s_learner_baseline_0623_0706" \
--feature_slot_file "./baseline_baseline.slots" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "s_learner_baseline_0623_0706" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--treatment_emb_dim 16 \
--gpu 2 > ./run_logs/s_learner_baseline_0623_0706.log 2>&1 &
'''
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import tensorflow as tf
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
            feature_map = tf.io.parse_example(serialized_batch, feature_dict)
            
            _labels = tf.strings.split(feature_map["label"][:, 0], ",")
            _labels = tf.strings.to_number(_labels, tf.float32)
            label = _labels.to_tensor()
            label = tf.reshape(label, [-1, self.label_size])

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
            hdfs_pattern=args.train_tfr_path, start_dt=args.train_start_dt, end_dt=args.train_end_dt, 
            token_size=args.token_size, label_size=args.label_size, parse_mode=parse_mode
        )
        dataset = reader.get_dataset(epoch=1, batch_size=args.BATCH_SIZE, parallel=args.parallel, 
                                     shuffle_size=args.shuffle_size, shuffle_seed=shuffle_seed, 
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

    # 从全量 TFRecord 特征中按 slot 选出当前方案需要的特征。
    feature_slot_indices = getattr(args, 'feature_slot_indices', None)
    if feature_slot_indices is not None:
        slot_tensor = tf.constant(feature_slot_indices, dtype=tf.int32)

        def _select_feature_slots(feature, label, token):
            feature_pairs = tf.reshape(feature, [tf.shape(feature)[0], -1, 2])
            selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
            selected_feature = tf.reshape(selected_pairs, [tf.shape(feature)[0], -1])
            return selected_feature, label, token

        dataset = dataset.map(_select_feature_slots, num_parallel_calls=args.parallel)

    return dataset

# 读取 .slots 文件：某个特征方案对应的 slot 编号。
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

    print(f"[FEATURE SLOT] loaded {len(slots)} slots from {feature_slot_file}") # 当前方案所用的特征数量，baseline:472,删除画像特这个：279；替换为订单信息：372
    # print(f"[FEATURE SLOT] first 20 slots: {slots[:20]}")
    return slots


class SLearner(tf.keras.Model):
    def __init__(self, args):
        super(SLearner, self).__init__()
        # Basic S-learner: one shared response function f(x, t).
        # It computes y_hat(control) and y_hat(T1~T4), then exports uplift_k = y_hat(Tk) - y_hat(control).
        self.treats_num = args.treats_num 
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) #
        self.bias_initializer = tf.keras.initializers.Constant(value=0)
        self.kernel_regularizer = tf.keras.regularizers.l2(args.l2_reg)

        # DENSE_FEA_SIZE：全局特征 ID 数量，也就是 embedding 表行数
        # EMBEDDING_SIZE：每个特征 ID 映射成多少维向量
        # PADDING_SIZE：每条样本固定有多少个特征槽位
        self.embedding_layer = IndexEmbedding(dense_fea_dim=args.DENSE_FEA_SIZE, embedding_dim=args.EMBEDDING_SIZE, sparse_fea_dim=args.PADDING_SIZE)
        self.treatment_embedding = tf.keras.layers.Embedding(
            input_dim=args.treats_num + 1,
            output_dim=args.treatment_emb_dim,
            embeddings_initializer=self.kernel_initializer,
            embeddings_regularizer=self.kernel_regularizer,
            name="treatment_id_embedding"
        )
        
        self.activation = [args.act_type, args.act_type, args.act_type]
        # shared_bottomt通过三层MLP输出为emb
        self.share_bottom = [tf.keras.layers.Dense(units=unit, activation=act, kernel_initializer=self.kernel_initializer, 
                             bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                             for unit, act in zip([args.emb_dim, args.emb_dim // 2, args.emb_dim // 4], self.activation)]

        # Shared outcome tower f_y(x, t): reused by control and all treatments.
        self.outcome_tower = [tf.keras.layers.Dense(units=hc_dim, activation=act, kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                              for hc_dim, act in zip([args.hc_dim, args.hc_dim//2, args.hc_dim//4, 1], self.activation + [None])]

        # Shared cost tower f_cost(x, t): only supervised on treatment samples.
        self.cost_tower = [tf.keras.layers.Dense(units=hu_dim, activation=act, kernel_initializer=self.kernel_initializer,
                           bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer)
                           for hu_dim, act in zip([args.hu_dim, args.hu_dim//2, args.hu_dim//4, 1], self.activation + [None])]

    def _predict_with_treatment(self, user_repr, treatment_id, tower):
        batch_size = tf.shape(user_repr)[0]
        treatment_ids = tf.fill([batch_size], tf.cast(treatment_id, tf.int32))
        treatment_repr = self.treatment_embedding(treatment_ids)
        tower_input = tf.concat([user_repr, treatment_repr], axis=1)
        return reduce(lambda x, layer: layer(x), tower, tower_input)

    def call(self, inputs):
        input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
        user_repr = self.embedding_layer(input_index[:, :])

        user_repr = reduce(lambda x, layer: layer(x), self.share_bottom, user_repr)

        y_control = self._predict_with_treatment(user_repr, 0, self.outcome_tower)
        y_treat = [self._predict_with_treatment(user_repr, i, self.outcome_tower) for i in range(1, self.treats_num + 1)]
        cost_head = [self._predict_with_treatment(user_repr, i, self.cost_tower) for i in range(1, self.treats_num + 1)]

        # 返回 control response、T1~T4 response 和 cost；uplift 在评估/落表阶段由 y_treat - y_control 得到。
        return y_control, y_treat[0], y_treat[1], y_treat[2], y_treat[3], cost_head[0], cost_head[1], cost_head[2], cost_head[3]

# 训练模型，预测结果，计算AUUC指标
def predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, valid_distribution=False, epoch_num=None):
    # Four treatment outputs: T1, T2, T3, T4.
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
    
    mode_str = 'train' if train else test
    epoch_str = f"Epoch {epoch_num}" if epoch_num is not None else ""
    log_interval = getattr(args, 'log_interval', 100)
    epoch_start_time = time.time()
    
    TREAT_INDEX = -3
    COST_INDEX = -4

    for x, y, t in test_dataset:
        inputs = x
        

        try:
            is_treat_tensor = t[:, TREAT_INDEX]
            print(t[0])
            is_treat = tf.strings.to_number(is_treat_tensor, tf.float32)
            print("treat is: ",is_treat)
            print("max treat is: ",max(is_treat))
            print("min treat is: ",min(is_treat))
            print("unique treat is:", tf.constant(np.unique(is_treat.numpy())))
            
            cost_tensor = t[:, COST_INDEX]
            label_cost_7d = tf.strings.to_number(cost_tensor, tf.float32)
            print("cost is: ",label_cost_7d)
            print("max cost is: ",max(label_cost_7d))
            print("min cost is: ",min(label_cost_7d))
            print("unique cost is:", tf.constant(np.unique(label_cost_7d.numpy())))
            

            _ = is_treat.numpy()
            _ = label_cost_7d.numpy()
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
                y_control, y_t1, y_t2, y_t3, y_t4, cost_1, cost_2, cost_3, cost_4 = model(inputs)
                
                is_treat = tf.reshape(is_treat, [-1, 1])
                label = tf.reshape(label, [-1, 1])
                label_cost_7d = tf.reshape(label_cost_7d, [-1, 1])
                
                # 只保留 [0, 1, 2, 3, 4] 五个组
                is_list = [tf.cast((is_treat == treat), tf.float32) for treat in [0, 1, 2, 3, 4]]

                # Basic S-learner: 训练阶段直接拟合观测 treatment 下的 response label。
                y_treat = [y_t1, y_t2, y_t3, y_t4]
                
                loss1 = tf.reduce_mean(tf.square(is_list[0] * y_control + is_list[1] * y_treat[0] + is_list[2] * y_treat[1] + is_list[3] * y_treat[2] + is_list[4] * y_treat[3] - label)
                                       + tf.square(is_list[1] * cost_1 + is_list[2] * cost_2 + is_list[3] * cost_3 + is_list[4] * cost_4 - label_cost_7d))
                
                total_loss = loss1
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            y_control, y_t1, y_t2, y_t3, y_t4, cost_1, cost_2, cost_3, cost_4 = model(inputs)

        # S-learner 先预测各 treatment 的 response，再相减得到四个 treatment 相对 control 的 uplift。
        delta_cvr_1 = y_t1 - y_control
        delta_cvr_2 = y_t2 - y_control
        delta_cvr_3 = y_t3 - y_control
        delta_cvr_4 = y_t4 - y_control

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
    
    if train: print('loss1:', loss1)

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
    parser = argparse.ArgumentParser(description="Basic 4-treatment S-learner parameters")
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
    parser.add_argument('--version', type=str, default="s_learner_4t_model")
    parser.add_argument('--hc_dim', type=int, default=256)
    parser.add_argument('--hu_dim', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=2048)
    parser.add_argument('--treatment_emb_dim', type=int, default=16, help='S-learner treatment_id embedding dimension')
    parser.add_argument('--treats_num', type=int, default=4) # 固定为 4 个实验组 (1, 2, 3, 4)
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--act_type', type=str, default='relu')
    parser.add_argument('--skip_eval', default=False, action='store_true', help='跳过验证集和测试集评估')
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--saveDir', type=str)
    parser.add_argument('--fromPretrained', default=False, action='store_true')
    parser.add_argument('--scheduleDt', type=str, default='')
    parser.add_argument('--startDate', type=str, default="20250102")
    parser.add_argument('--shuffle_size', type=int, default=20480)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--label_size', type=int, default=1)
    parser.add_argument('--token_size', type=int, default=14) #已修改，token_size =14
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--debug_mode', default=False, action='store_true')
    parser.add_argument('--debug_batches', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--skip_auc', default=False, action='store_true')
    parser.add_argument('--train_tfr_path', type=str, required=True)
    parser.add_argument('--valid_tfr_path', type=str, required=True)
    parser.add_argument('--test_tfr_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--parse_mode', type=str, default='batch_then_map', choices=['batch_then_map', 'map_then_batch'])
    parser.add_argument('--feature_slot_file', type=str, default='', help='0-based feature_conf slot list for feature selection')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_gpu(args.gpu)
    print("Parsed arguments:"); print(args)

    # 读取 .slots 文件里的 slot 编号，并根据方案自动修改 PADDING_SIZE。
    args.feature_slot_indices = load_feature_slot_indices(args.feature_slot_file)
    if args.feature_slot_indices is not None:
        args.PADDING_SIZE = len(args.feature_slot_indices)
        print(f"[FEATURE SLOT] override PADDING_SIZE to {args.PADDING_SIZE}")

    train_dataset = load_from_tfr("train", args)
    print("Train dataset loaded")
    
    if not args.skip_eval:
        valid_dataset = load_from_tfr("valid", args)
        print("Valid dataset loaded")
    else:
        valid_dataset = None
        print("Valid dataset loading skipped (--skip_eval 模式)")

    # 模型构建
    model = SLearner(args)
    input_len = args.PADDING_SIZE * 2  # 自动算出物理长度 (448 * 2 = 896)
    model.build(input_shape=[None, input_len])
    inputs = tf.keras.Input(shape=(input_len,), name='input_feature')
    outputs = model(inputs)
    
    # S-learner 模型输出 response；评估/落表阶段再计算 uplift_k = response_k - control。
    output_names = ['control', 'response_1', 'response_2', 'response_3', 'response_4', 'cost_1', 'cost_2', 'cost_3', 'cost_4']
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
        
        df, _, total_loss = predict_or_train_data_auuc(args, model, train_dataset, test='train', train=True, valid_distribution=False, epoch_num=epoch+1)
        
        print(f'[{get_beijing_time_str()}] Epoch {epoch+1} 训练完成')
        print(f'Epoch {epoch+1}, Total Loss: {total_loss}')

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
