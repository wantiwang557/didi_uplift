# test_hive.py 改动与参数说明

本文档说明 `test_hive.py` 相比原始 Hive 推理脚本做了哪些最小必要改动，以及后续切换三套特征方案或模型版本时需要改哪些参数。

目标脚本：

```text
/home/xiaoju/data9/hnb/user_code/3/wangwanti_i/task/multi_treatment_feature_select_0/main.py
```

本地参考文件：

```text
/Users/didi/wwt_local/wwt/demo/feature_select/test_hive.py
```

## 1. 为什么需要改

原始脚本直接读取全量 TFRecord 后喂给模型：

```python
inputs = feature
model(inputs)
```

但当前 EUEN 模型是按不同特征方案训练出来的：

```text
baseline      473 slots -> 输入长度 946
no_sirius     279 slots -> 输入长度 558
replace_order 372 slots -> 输入长度 744
```

全量 TFRecord 是 609 slots：

```text
609 slots -> 输入长度 1218
```

所以如果不按 `.slots` 筛选特征，会报错：

```text
expected shape=(None, 946), found shape=(20480, 1218)
```

## 2. 本次最小改动点

### 2.1 新增读取 slot 文件函数

新增：

```python
def load_feature_slot_indices(feature_slot_file):
    ...
```

作用：

```text
读取 baseline_baseline.slots 等文件中的行号，得到当前方案要保留的 feature slot。
```

### 2.2 在 load_from_tfr 中筛选特征

新增逻辑：

```python
if args.feature_slot_indices is not None:
    slot_tensor = tf.constant(args.feature_slot_indices, dtype=tf.int32)

    def _select_feature_slots(feature, label, token):
        feature_pairs = tf.reshape(feature, [tf.shape(feature)[0], -1, 2])
        selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
        selected_feature = tf.reshape(selected_pairs, [tf.shape(feature)[0], -1])
        return selected_feature, label, token

    dataset = dataset.map(_select_feature_slots, num_parallel_calls=4)
```

作用：

```text
全量 609 slots TFRecord -> 当前方案需要的 473 / 279 / 372 slots。
```

### 2.3 token_size 改为 14

原始脚本里 reader 使用：

```python
token_size=-1
```

当前 token 固定是 14 个字段，因此改成：

```python
token_size=14
```

否则 token reshape 可能不稳定，也不方便按下标取字段。

### 2.4 修正 token 下标

当前 token 顺序是：

```text
0  fin_order_wxmini_28d
1  group_name_eve
2  pid
3  oneid
4  send_cnt
5  send_success_cnt
6  send_cost
7  remind_cost
8  coupon_cost
9  dau_cnt
10 cost_7d
11 fake_treatment_id
12 flag
13 label_dt
```

所以推理脚本中应使用：

```python
is_treat = tf.strings.to_number(t[:, 11], tf.float32)
pid.append(token[:, 2].numpy())
label_dt.append(token[:, 13].numpy())
```

原始写法中 `t[:, 10]` 实际是 `cost_7d`，不是 treatment；`token[:, 1]` 实际是 `group_name_eve`，不是 pid。

### 2.5 新增 feature_slot_file 参数

新增：

```python
parser.add_argument('--feature_slot_file', type=str, default="./baseline_baseline.slots")
```

如果任务无法访问 `/nfs/...`，最稳做法是把 `.slots` 文件和 `main.py` 放同一个目录：

```text
/home/xiaoju/data9/hnb/user_code/3/wangwanti_i/task/multi_treatment_feature_select_0/main.py
/home/xiaoju/data9/hnb/user_code/3/wangwanti_i/task/multi_treatment_feature_select_0/baseline_baseline.slots
```

## 3. 当前代码需要注意的问题

你刚才贴的最新版中有两处需要修正。

### 3.1 model_path 不能写错

错误写法：

```python
_path = hdfs_model"hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/"
args.hdfs_model_path = hdfs_model_path
```

正确写法：

```python
def model_path(args):
    hdfs_model_path = "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/"
    args.hdfs_model_path = hdfs_model_path
    if not hasattr(args, 'nfs_model_path'):
        args.nfs_model_path = args.hdfs_model_path
    return args
```

### 3.2 test_dt 不能直接注释掉

`load_from_tfr()` 中仍然使用：

```python
start_dt=args.test_dt
end_dt=args.test_dt
```

所以 `parse_args()` 里要保留：

```python
parser.add_argument('--test_dt', type=str, default="20260716")
```

虽然当前 `test_rta_tfr_path` 是写死的，`test_dt` 不决定真实读取路径，但代码仍需要这个参数存在。

## 4. 三套方案参数

以下参数需要随方案同步修改。

### 4.1 baseline

模型路径：

```python
hdfs_model_path = "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/"
```

参数：

```python
parser.add_argument('--PADDING_SIZE', type=int, default=473)
parser.add_argument('--DENSE_FEA_SIZE', type=int, default=5000)
parser.add_argument('--EMBEDDING_SIZE', type=int, default=8)
parser.add_argument('--BATCH_SIZE', type=int, default=2048)
parser.add_argument('--test_dt', type=str, default="20260716")
parser.add_argument('--version', type=str, default="euen_baseline_0623_0629")
parser.add_argument('--hc_dim', type=int, default=256)
parser.add_argument('--hu_dim', type=int, default=256)
parser.add_argument('--emb_dim', type=int, default=2048)
parser.add_argument('--treats_num', type=int, default=4)
parser.add_argument('--startDate', type=str, default="20260706")
parser.add_argument('--feature_slot_file', type=str, default="./baseline_baseline.slots")
```

### 4.2 no_sirius

模型路径按实际训练产物修改，例如：

```python
hdfs_model_path = "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/no_sirius_0623_0706"
```

参数：

```python
parser.add_argument('--PADDING_SIZE', type=int, default=279)
parser.add_argument('--DENSE_FEA_SIZE', type=int, default=5000)
parser.add_argument('--EMBEDDING_SIZE', type=int, default=8)
parser.add_argument('--BATCH_SIZE', type=int, default=2048)
parser.add_argument('--test_dt', type=str, default="20260716")
parser.add_argument('--version', type=str, default="euen_no_sirius_0623_0706")
parser.add_argument('--hc_dim', type=int, default=256)
parser.add_argument('--hu_dim', type=int, default=256)
parser.add_argument('--emb_dim', type=int, default=2048)
parser.add_argument('--treats_num', type=int, default=4)
parser.add_argument('--startDate', type=str, default="20260706")
parser.add_argument('--feature_slot_file', type=str, default="./baseline_no_sirius.slots")
```

### 4.3 replace_order

模型路径按实际训练产物修改，例如：

```python
hdfs_model_path = "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/replace_order_0623_0706"
```

参数：

```python
parser.add_argument('--PADDING_SIZE', type=int, default=372)
parser.add_argument('--DENSE_FEA_SIZE', type=int, default=5000)
parser.add_argument('--EMBEDDING_SIZE', type=int, default=8)
parser.add_argument('--BATCH_SIZE', type=int, default=2048)
parser.add_argument('--test_dt', type=str, default="20260716")
parser.add_argument('--version', type=str, default="euen_replace_order_0623_0706")
parser.add_argument('--hc_dim', type=int, default=256)
parser.add_argument('--hu_dim', type=int, default=256)
parser.add_argument('--emb_dim', type=int, default=2048)
parser.add_argument('--treats_num', type=int, default=4)
parser.add_argument('--startDate', type=str, default="20260706")
parser.add_argument('--feature_slot_file', type=str, default="./baseline_replace_sirius_with_xtlx_order.slots")
```

## 5. 参数含义和是否必须修改

### 必须随方案修改

```text
hdfs_model_path
PADDING_SIZE
feature_slot_file
version
```

其中 `PADDING_SIZE` 后面会被：

```python
args.PADDING_SIZE = len(args.feature_slot_indices)
```

覆盖，但默认值仍建议写对，方便排查日志。

### 必须和训练保持一致

```text
DENSE_FEA_SIZE = 5000
EMBEDDING_SIZE = 8
hc_dim = 256
hu_dim = 256
emb_dim = 2048
treats_num = 4
```

如果后续训练模型改了这些网络结构参数，Hive 推理脚本也要同步。

### 当前可以保持原脚本默认

```text
BATCH_SIZE = 2048
```

如果平台自动提交时传 `--BATCH_SIZE=20480`，会覆盖默认值。这个参数只影响推理 batch 大小，不影响模型结构。

### test_dt 的作用

当前路径写死：

```python
test_rta_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/16/part-*"
```

因此真实读取的数据由 `test_rta_tfr_path` 决定，不由 `test_dt` 决定。

但 `test_dt` 仍被 reader 使用，不能删：

```python
parser.add_argument('--test_dt', type=str, default="20260716")
```

### startDate 的作用

`startDate` 是 Hive 写表分区：

```python
partition_val = args.startDate
partition_col = 'dt'
```

当前代码只会写一个分区，例如：

```text
dt=20260706
```

如果想按每条样本的 `label_dt` 写多个分区，需要另外改成按 `label_dt` groupby 后循环写表。

## 6. 后续模型迭代时检查清单

每次换模型或换方案时，至少检查：

```text
1. hdfs_model_path 是否指向正确模型
2. feature_slot_file 是否和模型训练时的方案一致
3. .slots 文件是否在任务运行目录可访问
4. PADDING_SIZE 是否等于 slot 文件行数
5. DENSE_FEA_SIZE / EMBEDDING_SIZE / hc_dim / hu_dim / emb_dim 是否和训练一致
6. token 顺序是否没有变化
7. test_rta_tfr_path 是否指向本次要推理的 TFRecord
8. startDate 是否是本次想写入的 Hive 分区
9. output table_name 是否是目标表
```

如果 token 列顺序变化，需要同步修改：

```python
is_treat = tf.strings.to_number(t[:, 11], tf.float32)
pid.append(token[:, 2].numpy())
label_dt.append(token[:, 13].numpy())
```

如果 feature_conf/slot 方案变化，需要重新生成 `.slots` 文件，并同步替换 `feature_slot_file`。
