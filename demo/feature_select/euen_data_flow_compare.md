# EUEN 原版与特征筛选版数据链路对比

本文档说明两个文件从 TFRecord 读取数据、构造模型输入、进入 `EUEN.call()`、完成训练的完整流程。

- 原版：`/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EUEN/euen_base_4t.py`
- 特征筛选版：`/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/euen_4t_feature.py`

核心区别：

```text
原版：TFRecord 里解析出来多少特征，模型就全部使用。
新版：TFRecord 仍然是全量特征，但模型训练前按 .slots 文件只保留当前方案需要的特征。
```

## 1. TFRecord 样本长什么样

离线抽数流程会把 SQL 结果经过 `fea_extract` 转成 TFRecord。每条 TFRecord 样本包含三个字段：

```python
feature_dict = {
    "label":   tf.io.FixedLenFeature([1], tf.string),
    "feature": tf.io.FixedLenFeature([1], tf.string),
    "token":   tf.io.FixedLenFeature([1], tf.string),
}
```

可以理解成：

```text
label:   "1"
feature: "12:1,305:1,98:1,2044:1,..."
token:   "fin_order_wxmini_28d,group_name_eve,pid,oneid,..."
```

其中：

- `label`：训练目标，比如 7 天完单量。
- `feature`：真正输入模型的特征，格式是 `index:value,index:value,...`。
- `token`：辅助字段，不直接作为模型特征，主要用于取 treatment、cost、pid、日期等。

注意：TFRecord 里的 `feature` 已经没有原始字段名，只有按 `feature_conf` 顺序排列的 `index:value`。

## 2. slot 是什么

`slot` 是 `feature_conf` 中的特征位置，也可以理解为第几个入模特征。

例如全量 `feature_baseline_train.conf` 有 609 行，那么一条样本会有 609 个 slot：

```text
slot 0   -> feature_conf 第 0 行对应的特征
slot 1   -> feature_conf 第 1 行对应的特征
slot 2   -> feature_conf 第 2 行对应的特征
...
slot 608 -> feature_conf 第 608 行对应的特征
```

每个 slot 在 TFRecord 里对应一组 `[index, value]`。

对于当前的 `S_slice_binning` 离散分箱特征，一个原始特征会根据取值落入某个分箱，然后得到一个 `index`。所以：

```text
一个 slot = 一个原始特征的位置
一个 slot 在每条样本里 = 一组 index:value
同一个 slot 在不同样本里，index 可能不同，因为落入的分箱不同
```

`.slots` 文件里只有行号，是因为模型筛选特征时需要知道“保留第几个 slot”，而不是知道原始特征名。

## 3. 原版 euen_base_4t.py 的数据链路

原版整体流程：

```text
TFRecord 文件
  ↓
LiteFeTFRecords.get_dataset()
  ↓
load_from_tfr()
  ↓
predict_or_train_data_auuc()
  ↓
model(inputs)
  ↓
EUEN.call()
  ↓
embedding + shared_bottom + heads
  ↓
loss 计算 + 梯度更新
```

### 3.1 初始化 TFRecord reader

相关函数：

```python
class LiteFeTFRecords(object):
    def __init__(self,
                 hdfs_pattern,
                 start_dt,
                 end_dt,
                 label_size=-1,
                 token_size=-1,
                 dt_fmt="%Y%m%d",
                 parse_mode='batch_then_map'):
```

作用：

- `hdfs_pattern`：TFRecord 文件路径，可以是 `part-*`。
- `start_dt`、`end_dt`：日期范围，用于按天拼路径。
- `label_size`：label 的长度。当前一般是 1。
- `token_size`：token 字段数量。当前是 14。
- `parse_mode`：解析模式，默认是 `batch_then_map`。

### 3.2 找到 TFRecord 文件

相关函数：

```python
def _get_hdfs_path_list(self):
    tfr_path_list = []
    days = (self.end_dt - self.start_dt).days
    for i in range(days + 1):
        _pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
        tfr_path = tf.io.gfile.glob(_pattern)
        tfr_path_list.extend(tfr_path)
    return tfr_path_list
```

作用：根据传入的路径 pattern 找到真实的 TFRecord 文件列表。

如果路径是：

```text
/path/to/train/part-*
```

它会通过：

```python
tf.io.gfile.glob(_pattern)
```

找到所有匹配的文件。

### 3.3 解析 TFRecord

相关函数：

```python
def get_dataset(...):
```

核心代码：

```python
dataset = tf.data.TFRecordDataset(
    filenames=tfr_path_list,
    num_parallel_reads=parallel
)
```

然后根据 `parse_mode` 选择解析方式：

```python
dataset = dataset.batch(
    batch_size=batch_size,
    drop_remainder=drop_remainder
).map(
    map_func=_parse_batch_example,
    num_parallel_calls=parallel
)
```

当前默认是 `batch_then_map`，也就是：

```text
先把样本组成 batch，再对整个 batch 做解析。
```

### 3.4 _parse_batch_example 做了什么

核心代码：

```python
feature_map = tf.io.parse_example(serialized_batch, feature_dict)
```

这一步把 TFRecord 里的三列字符串取出来：

```text
label
feature
token
```

解析 label：

```python
_labels = tf.strings.split(feature_map["label"][:, 0], ",")
_labels = tf.strings.to_number(_labels, tf.float32)
label = _labels.to_tensor()
label = tf.reshape(label, [-1, self.label_size])
```

解析 feature：

```python
libsvm_kv = tf.strings.split(feature_map["feature"][:, 0], ",")
libsvm_kv = tf.strings.split(libsvm_kv, ":")
libsvm_kv = libsvm_kv.flat_values
feature = tf.strings.to_number(libsvm_kv, tf.float32)
feature = tf.reshape(feature, [-1, tf.shape(feature)[0] // tf.shape(label)[0]])
```

假设一条样本有 448 个 slot，那么每个 slot 是 `[index, value]`，所以 feature 长度是：

```text
448 * 2 = 896
```

解析后的 batch 形状是：

```text
feature: [batch_size, 896]
label:   [batch_size, 1]
token:   [batch_size, 14]
```

### 3.5 load_from_tfr 返回 dataset

相关函数：

```python
def load_from_tfr(train_type, args):
```

根据 `train_type` 选择路径：

```python
if train_type == "train":
    hdfs_pattern = args.train_tfr_path
elif train_type == "test":
    hdfs_pattern = args.test_tfr_path
else:
    hdfs_pattern = args.valid_tfr_path
```

最后返回：

```python
dataset = reader.get_dataset(..., return_token=True)
return dataset
```

因此训练循环里拿到的是：

```python
for x, y, t in dataset:
```

含义：

```text
x = feature，模型输入特征
y = label，训练目标
t = token，辅助字段
```

### 3.6 构建模型输入

原版主函数中：

```python
model = EUEN(args)
input_len = args.PADDING_SIZE * 2
model.build(input_shape=[None, input_len])
inputs = tf.keras.Input(shape=(input_len,), name='input_feature')
outputs = model(inputs)
```

如果 `PADDING_SIZE=448`：

```text
input_len = 448 * 2 = 896
```

这里的 `tf.keras.Input` 只是定义模型输入形状，不负责读取数据。

### 3.7 EUEN.call 如何使用输入特征

相关函数：

```python
def call(self, inputs):
    input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
    input_value = tf.cast(inputs[:, 1::2], tf.float32)
    inputs = self.embedding_layer(input_index[:, :])
```

输入格式是：

```text
[index1, value1, index2, value2, index3, value3, ...]
```

所以：

```python
inputs[:, 0::2]
```

取偶数位，也就是所有 `index`。

```python
inputs[:, 1::2]
```

取奇数位，也就是所有 `value`。

当前代码中实际进入 embedding 的是：

```python
input_index
```

`input_value` 被取出来了，但后续没有参与计算。

### 3.8 模型结构

`EUEN.__init__()` 中定义：

```python
self.embedding_layer = IndexEmbedding(...)
self.share_bottom = [...]
self.c_head = [...]
self.u_head = [...]
self.cost_head = [...]
```

含义：

- `embedding_layer`：把离散 index 转成 embedding。
- `share_bottom`：共享底层 MLP。
- `c_head`：control 组预测。
- `u_head`：4 个 treatment 的 uplift head。
- `cost_head`：4 个 treatment 的 cost head。

输出：

```python
return c_head, c_head_logit_fix, u_head[0], u_head[1], u_head[2], u_head[3], cost_head[0], cost_head[1], cost_head[2], cost_head[3]
```

也就是：

```text
control
control_logit_fix
uplift_1
uplift_2
uplift_3
uplift_4
cost_1
cost_2
cost_3
cost_4
```

### 3.9 训练函数

相关函数：

```python
def predict_or_train_data_auuc(args, model, test_dataset, test='test', train=False, ...):
```

虽然参数名叫 `test_dataset`，但它不一定是 test 数据。实际传什么就是什么：

```python
predict_or_train_data_auuc(args, model, train_dataset, test='train', train=True)
predict_or_train_data_auuc(args, model, valid_dataset, test='val', train=False)
```

训练时：

```python
for x, y, t in test_dataset:
    inputs = x
    with tf.GradientTape() as tape:
        uc, c_logit_fix, u_tau_1, ... = model(inputs)
        loss1 = ...
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

真正进入模型的是：

```python
inputs = x
model(inputs)
```

## 4. 特征筛选版 euen_4t_feature.py 的数据链路

新版整体流程：

```text
全量 TFRecord 文件，609 个 slot
  ↓
LiteFeTFRecords.get_dataset()
  ↓
load_from_tfr()
  ↓
按 .slots 文件筛选 feature
  ↓
predict_or_train_data_auuc()
  ↓
model(inputs)
  ↓
EUEN.call()
  ↓
embedding + shared_bottom + heads
  ↓
loss 计算 + 梯度更新
```

### 4.1 新增参数 feature_slot_file

新增命令行参数：

```python
parser.add_argument(
    '--feature_slot_file',
    type=str,
    default='',
    help='0-based feature_conf slot list for feature selection'
)
```

运行时传入：

```bash
--feature_slot_file ./baseline_no_sirius.slots
```

意思是：本次训练只使用这个 `.slots` 文件指定的特征。

### 4.2 读取 .slots 文件

新增函数：

```python
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

    return slots
```

例如 `baseline_no_sirius.slots` 里有 279 行，那么返回：

```text
[194, 195, 196, ..., 472]
```

具体数字取决于 `feature_conf` 中对应特征的位置。

### 4.3 自动修改 PADDING_SIZE

新版主函数中：

```python
args.feature_slot_indices = load_feature_slot_indices(args.feature_slot_file)
if args.feature_slot_indices is not None:
    args.PADDING_SIZE = len(args.feature_slot_indices)
```

这样不同方案会自动得到不同的输入 slot 数：

```text
baseline: 473
no_sirius: 279
replace_sirius_with_xtlx_order: 372
```

因此新版运行时不需要再手动传旧的：

```bash
--PADDING_SIZE 448
```

### 4.4 在 load_from_tfr 中筛选特征

新版核心改动在 `load_from_tfr()` 末尾：

```python
feature_slot_indices = getattr(args, 'feature_slot_indices', None)
if feature_slot_indices is not None:
    slot_tensor = tf.constant(feature_slot_indices, dtype=tf.int32)

    def _select_feature_slots(feature, label, token):
        feature_pairs = tf.reshape(feature, [tf.shape(feature)[0], -1, 2])
        selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
        selected_feature = tf.reshape(selected_pairs, [tf.shape(feature)[0], -1])
        return selected_feature, label, token

    dataset = dataset.map(_select_feature_slots, num_parallel_calls=args.parallel)
```

假设全量 TFRecord 是 609 个 slot：

```text
feature 原始形状: [batch_size, 609 * 2]
```

先 reshape：

```text
[batch_size, 609, 2]
```

变成：

```text
slot 0:   [index, value]
slot 1:   [index, value]
slot 2:   [index, value]
...
slot 608: [index, value]
```

然后：

```python
tf.gather(feature_pairs, slot_tensor, axis=1)
```

表示按 `.slots` 文件保留指定 slot。

如果当前方案是 279 个 slot：

```text
selected_pairs: [batch_size, 279, 2]
```

最后 flatten 回：

```text
selected_feature: [batch_size, 279 * 2]
```

也就是：

```text
[batch_size, 558]
```

### 4.5 新版 Input 和 call 为什么不用大改

因为筛选前后，输入格式不变。

原版输入：

```text
[index, value, index, value, index, value, ...]
```

新版输入：

```text
[index, value, index, value, index, value, ...]
```

只是长度从全量变成了当前方案长度。

所以 `EUEN.call()` 仍然可以使用：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
input_value = tf.cast(inputs[:, 1::2], tf.float32)
inputs = self.embedding_layer(input_index[:, :])
```

区别只在于：

```text
方案1: input_len = 473 * 2 = 946
方案2: input_len = 279 * 2 = 558
方案3: input_len = 372 * 2 = 744
```

## 5. 原版与新版关键差异汇总

| 环节 | 原版 euen_base_4t.py | 新版 euen_4t_feature.py |
|---|---|---|
| TFRecord | 使用抽好的特征 | 使用同一份全量 609 特征 |
| 特征选择位置 | 抽数阶段决定 | 训练读取 TFRecord 后决定 |
| 是否需要多次抽数 | 不同方案通常要抽多份 | 只需要抽一次全量 |
| 控制方案的方式 | 换 TFRecord 或换 feature_conf 后重抽 | 换 `--feature_slot_file` |
| `PADDING_SIZE` | 命令行手动传 | 根据 `.slots` 行数自动覆盖 |
| `Input` | `PADDING_SIZE * 2` | `len(slots) * 2` |
| `EUEN.call()` | 取 index 做 embedding | 不变，仍然取 index 做 embedding |
| 网络结构 | embedding + shared_bottom + heads | 不变 |
| loss | 不变 | 不变 |

## 6. 新版完整执行顺序

```text
1. parse_args()
   读取命令行参数，包括 --feature_slot_file

2. load_feature_slot_indices()
   读取 .slots 文件，得到当前方案要保留的 slot 编号

3. args.PADDING_SIZE = len(args.feature_slot_indices)
   自动设置当前模型输入 slot 数

4. load_from_tfr("train", args)
   读取全量 train TFRecord

5. LiteFeTFRecords.get_dataset()
   把 label、feature、token 从字符串解析成 tensor

6. _select_feature_slots()
   把全量 609 个 slot 筛成当前方案的 473、279 或 372 个 slot

7. model = EUEN(args)
   根据新的 PADDING_SIZE 初始化 embedding 层和网络

8. inputs = tf.keras.Input(shape=(args.PADDING_SIZE * 2,))
   建立 Keras 输入形状

9. predict_or_train_data_auuc()
   遍历 dataset，执行 for x, y, t in dataset

10. inputs = x
    把当前 batch 的 feature 作为模型输入

11. model(inputs)
    调用 EUEN.call()

12. EUEN.call()
    从输入中取 index，查 embedding，经过 shared_bottom、uplift head、cost head

13. loss = ...
    使用 label、treatment、cost 计算损失

14. tape.gradient() + optimizer.apply_gradients()
    更新模型参数
```

## 7. 三个方案的输入变化

三套方案共用同一份全量 TFRecord，只换 `.slots` 文件：

| 方案 | `.slots` 文件 | slot 数 | 模型输入长度 |
|---|---|---:|---:|
| baseline | `baseline_baseline.slots` | 473 | 946 |
| 删除 sirius | `baseline_no_sirius.slots` | 279 | 558 |
| sirius 替换为小天狼星订单 | `baseline_replace_sirius_with_xtlx_order.slots` | 372 | 744 |

运行时只需要替换：

```bash
--feature_slot_file ./baseline_baseline.slots
```

或：

```bash
--feature_slot_file ./baseline_no_sirius.slots
```

或：

```bash
--feature_slot_file ./baseline_replace_sirius_with_xtlx_order.slots
```

## 8. 一句话总结

原版 EUEN 的特征选择发生在抽数之前，模型读到什么就训练什么。

新版 `euen_4t_feature.py` 的特征选择发生在模型读取 TFRecord 之后、进入网络之前。全量 TFRecord 不变，通过 `.slots` 文件决定当前实验真正进入模型的特征集合。
