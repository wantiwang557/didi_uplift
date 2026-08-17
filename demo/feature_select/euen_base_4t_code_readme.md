# euen_base_4t.py 代码解读

本文说明原始 `demo/EUEN/euen_base_4t.py` 的数据链路、模型结构、训练过程和评估逻辑。

## 1. 整体流程

原始 EUEN 训练链路是：

```text
Hive SQL + feature_conf/parser_conf
  -> Spark extract 生成 TFRecord
  -> LiteFeTFRecords 读取 TFRecord
  -> 解析 label / feature / token
  -> EUEN.call 提取 index 特征并做 embedding
  -> shared bottom + control/uplift/cost heads
  -> 按 treatment 计算 loss
  -> 每轮计算 train/valid AUUC/AUC
  -> 训练结束后在 test 上计算 AUUC/PCOC/Cal-N
```

原始模型要求传入固定长度特征，例如旧命令中：

```bash
--PADDING_SIZE 448
--DENSE_FEA_SIZE 3300
--token_size 14
```

含义：

- `PADDING_SIZE=448`：每条样本有 448 个特征 slot。
- 每个 slot 在 TFRecord 中是一个 `index:value` 对。
- 模型实际输入长度是 `PADDING_SIZE * 2 = 896`。
- `DENSE_FEA_SIZE=3300`：embedding 表大小，也就是全局离散分箱 id 空间。
- `token_size=14`：token 字段包含 14 个辅助字段。

## 2. TFRecord 样本格式

上游 `run_offline_tans2tfr.py` 写出的 TFRecord 有三个 string 字段：

```text
label
feature
token
```

一条样本可以理解成：

```text
label = "1.0"
feature = "12:1,45:1,103:1,899:1,..."
token = "fin_order_wxmini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,cost_7d,fake_treatment_id,flag,label_dt"
```

其中：

- `label`：当前代码中是 `finish_cnt_7d`，训练 uplift 的 outcome。
- `feature`：libsvm 风格的离散特征串，格式是 `index:value,index:value,...`。
- `token`：额外信息，不进入 embedding；代码从这里取 `cost_7d` 和 `fake_treatment_id`。

当前项目的 `feature_conf` 都是 `S_slice_binning`，即离散分箱。一个原始特征可能有多个 bucket index，但每条样本只会落到其中一个 bucket，因此每个特征 slot 对每条样本只产生一个 `index:value`。

## 3. TFRecord 读取函数

### `LiteFeTFRecords.__init__`

负责记录 TFRecord 路径、起止日期、label/token 长度和解析模式：

```python
self.hdfs_pattern = hdfs_pattern.strip()
self.label_size = label_size
self.token_size = token_size
self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)
self.parse_mode = parse_mode
```

### `LiteFeTFRecords._get_hdfs_path_list`

根据 `hdfs_pattern` 和日期范围展开文件列表：

```python
_pattern = (self.start_dt + timedelta(days=i)).strftime(self.hdfs_pattern)
tfr_path = tf.io.gfile.glob(_pattern)
```

如果命令里传的是：

```bash
--train_tfr_path ".../part-*"
```

且路径里没有 `%Y%m%d` 这类格式占位符，日期展开不会改变路径，实际就是 glob 这批 `part-*` 文件。

### `LiteFeTFRecords.get_dataset`

定义 TFRecord schema：

```python
feature_dict = {
    "label":   tf.io.FixedLenFeature([1], tf.string),
    "feature": tf.io.FixedLenFeature([1], tf.string),
    "token":   tf.io.FixedLenFeature([1], tf.string),
}
```

默认 `parse_mode='batch_then_map'`，所以主要使用 `_parse_batch_example`。

关键解析逻辑：

```python
feature_map = tf.io.parse_example(serialized_batch, feature_dict)

_labels = tf.strings.split(feature_map["label"][:, 0], ",")
_labels = tf.strings.to_number(_labels, tf.float32)
label = tf.reshape(label, [-1, self.label_size])

libsvm_kv = tf.strings.split(feature_map["feature"][:, 0], ",")
libsvm_kv = tf.strings.split(libsvm_kv, ":")
libsvm_kv = libsvm_kv.flat_values
feature = tf.strings.to_number(libsvm_kv, tf.float32)
feature = tf.reshape(feature, [-1, tf.shape(feature)[0] // tf.shape(label)[0]])
```

假设一条样本的 `feature` 是：

```text
12:1,45:1,103:1
```

解析后会变成：

```text
[12, 1, 45, 1, 103, 1]
```

也就是：

```text
[index0, value0, index1, value1, index2, value2, ...]
```

token 解析：

```python
_token = tf.strings.split(feature_map["token"][:, 0], ",")
token = _token.to_tensor(default_value="")
token = tf.reshape(token, [-1, self.token_size])
```

返回：

```python
return (feature, label, token)
```

## 4. 数据集选择函数

### `load_from_tfr(train_type, args)`

根据 `train_type` 选择不同路径：

```python
if train_type == "train":
    hdfs_pattern = args.train_tfr_path
elif train_type == "test":
    hdfs_pattern = args.test_tfr_path
else:
    hdfs_pattern = args.valid_tfr_path
```

因此：

- `train_dataset` 来自 `--train_tfr_path`
- `valid_dataset` 来自 `--valid_tfr_path`
- `test_dataset` 来自 `--test_tfr_path`

代码本身不会自动从 train 中切 validation。

## 5. 模型结构

### `IndexEmbedding`

`IndexEmbedding` 是对离散 index 做 embedding lookup：

```python
_embedding = tf.nn.embedding_lookup(self.embeddings, feature)
embedding = tf.reshape(_embedding, [-1, self.out_dense_dim])
```

其中：

```python
self.out_dense_dim = sparse_fea_dim * embedding_dim
```

如果：

```text
PADDING_SIZE = 448
EMBEDDING_SIZE = 8
```

则 embedding 展平后维度是：

```text
448 * 8 = 3584
```

### `EUEN.__init__`

主要模块：

```python
self.embedding_layer = IndexEmbedding(
    dense_fea_dim=args.DENSE_FEA_SIZE,
    embedding_dim=args.EMBEDDING_SIZE,
    sparse_fea_dim=args.PADDING_SIZE
)
```

shared bottom：

```python
self.share_bottom = [
    Dense(args.emb_dim),
    Dense(args.emb_dim // 2),
    Dense(args.emb_dim // 4)
]
```

control head：

```python
self.c_head = [
    Dense(args.hc_dim),
    Dense(args.hc_dim // 2),
    Dense(args.hc_dim // 4),
    Dense(1)
]
```

uplift head：4 个 treatment 各一个 uplift head。

```python
self.u_head = [[Dense(...), ..., Dense(1)] for _ in range(self.treats_num)]
```

cost head：4 个 treatment 各一个 cost head。

```python
self.cost_head = [[Dense(...), ..., Dense(1)] for _ in range(self.treats_num)]
```

### `EUEN.call`

输入格式：

```text
[index0,value0,index1,value1,...]
```

拆 index 和 value：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
input_value = tf.cast(inputs[:, 1::2], tf.float32)
```

当前代码只使用 `input_index`：

```python
inputs = self.embedding_layer(input_index[:, :])
```

`input_value` 被解析出来，但没有参与计算。这和当前 baseline 逻辑一致。

网络前向：

```python
inputs = reduce(lambda x, layer: layer(x), self.share_bottom, inputs)
c_head = reduce(lambda x, c_task: c_task(x), self.c_head, inputs)
u_head = [reduce(lambda x, u_task: u_task(x), self.u_head[i], inputs) for i in range(self.treats_num)]
cost_head = [reduce(lambda x, cost: cost(x), self.cost_head[i], inputs) for i in range(self.treats_num)]
```

返回：

```python
return c_head, c_head_logit_fix, u_head[0], u_head[1], u_head[2], u_head[3], cost_head[0], cost_head[1], cost_head[2], cost_head[3]
```

## 6. 训练与预测函数

### `predict_or_train_data_auuc`

这个函数同时承担训练、预测和指标计算。

循环 dataset：

```python
for x, y, t in test_dataset:
    inputs = x
```

其中：

- `x`：feature tensor
- `y`：label
- `t`：token

从 token 中取 treatment 和 cost：

```python
TREAT_INDEX = -3
COST_INDEX = -4

is_treat_tensor = t[:, TREAT_INDEX]
is_treat = tf.strings.to_number(is_treat_tensor, tf.float32)

cost_tensor = t[:, COST_INDEX]
label_cost_7d = tf.strings.to_number(cost_tensor, tf.float32)
```

根据当前 token 顺序：

```text
..., dau_cnt, cost_7d, fake_treatment_id, flag, label_dt
```

所以：

- `-4` 是 `cost_7d`
- `-3` 是 `fake_treatment_id`

训练分支：

```python
with tf.GradientTape() as tape:
    uc, c_logit_fix, u_tau_1, ..., cost_4 = model(inputs)
    ...
    total_loss = loss1

gradients = tape.gradient(total_loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

预测分支：

```python
_, _, u_tau_1, u_tau_2, u_tau_3, u_tau_4, cost_1, cost_2, cost_3, cost_4 = model(inputs)
```

## 7. 损失函数

代码中保留五类组：

```python
is_list = [tf.cast((is_treat == treat), tf.float32) for treat in [0, 1, 2, 3, 4]]
```

control 预测：

```python
uc
```

treatment 1：

```python
ut_1 = c_logit_fix + u_tau_1
```

treatment 2/3/4：

```python
ut = [
    c_logit_fix + t_logit_1_fix + u_tau_2,
    c_logit_fix + t_logit_1_fix + u_tau_3,
    c_logit_fix + t_logit_1_fix + u_tau_4
]
```

最终 loss：

```python
loss1 = mean(
    square(predicted_label - label)
    + square(predicted_cost - label_cost_7d)
)
```

其中 `predicted_label` 根据真实 treatment 选择 control 或对应 treatment 预测；`predicted_cost` 根据 treatment 选择对应 cost head。

## 8. 指标计算

### AUUC

`draw_auuc` 调用 causalml 风格的 cumulative lift：

```python
lift = metrics.get_cumlift(
    subset[['exp_group', 'label', ite_name]],
    outcome_col='label',
    treatment_col='exp_group',
    random_seed=4
)
```

`predict_or_train_data_auuc` 中每次会计算：

```text
predict_delta_cvr_1
predict_delta_cvr_2
predict_delta_cvr_3
predict_delta_cvr_4
```

### Cost AUC

如果没有传 `--skip_auc`，会计算 cost 预测和真实 cost 的 xAUC：

```python
auc_i = xauc_score(treat_i['label_cost_7d'], treat_i[predict_cost])
```

### PCOC / Cal-N

训练结束后执行：

```python
valid_distribution(args)
```

它会加载 best model，在 `test_tfr_path` 上预测，然后调用：

```python
evaluate_pcoc_and_cal_n(df, args)
```

内部使用：

```python
PCOCEvaluator.calculate_pcoc(...)
PCOCEvaluator.calculate_cal_n(...)
```

注意：当前代码中 treatment 4 的 PCOC 配置有一个可疑点：

```python
(4, 0, 'predict_delta_cvr_3')
```

更合理的是：

```python
(4, 0, 'predict_delta_cvr_4')
```

## 9. 训练主流程

主流程在 `if __name__ == "__main__"` 中：

```python
train_dataset = load_from_tfr("train", args)
valid_dataset = load_from_tfr("valid", args)
```

构建模型：

```python
model = EUEN(args)
input_len = args.PADDING_SIZE * 2
inputs = tf.keras.Input(shape=(input_len,), name='input_feature')
outputs = model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=named_outputs)
```

epoch 循环：

```python
for epoch in range(num_epochs):
    train
    valid
    save epoch model
    update best model by validation AUUC
```

训练结束：

```python
evaluation_train(args)
valid_distribution(args)
```

即在 test 集上做最终 AUUC/AUC 和 PCOC/Cal-N。

## 10. 没有 valid 时怎么办

代码不会自动划分 validation。可选方案：

1. 最推荐：从训练窗口中单独切出一段 valid，生成单独 TFRecord，然后传给 `--valid_tfr_path`。
2. 临时对比实验：让 `--valid_tfr_path` 和 `--test_tfr_path` 指向同一份数据。这样能跑通，但 best model 选择会使用 test，评估会有泄漏。
3. 使用 `--skip_eval` 可以跳过验证，但当前代码训练结束后的 test 流程也会跳过，不适合最终比较三套方案。

如果要严谨比较三套特征方案，建议使用同一套 train/valid/test 切分，并让三套模型完全复用这些 TFRecord。
