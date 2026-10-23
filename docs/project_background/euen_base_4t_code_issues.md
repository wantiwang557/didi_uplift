# euen_base_4t.py 基础问题清单

本文先罗列 `demo/euen_base_4t.py` 这份代码的基础问题，重点关注：运行稳定性、数据字段一致性、训练逻辑、评估可信度和可维护性。

## 1. token_size 与上游 token 数量不一致

位置：

- `demo/euen_base_4t.py:15`
- `demo/euen_base_4t.py:30`
- `demo/euen_base_4t.py:740`
- `offline_bigdata/script/feature_confs/v0/parser_v0_train.conf:5`

问题：

上游 `parser_v0_train.conf` 的 `add_token` 一共有 14 个字段：

```text
fin_order_mini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,cost,fake_treatment_id,flag,label_dt
```

但 `demo/euen_base_4t.py` 顶部示例命令中写的是 `--token_size 13`，参数默认值还是 `default=7`。

影响：

如果实际 TFRecord 是按当前 `parser_v0_train.conf` 生成的，运行时 `tf.reshape(token, [-1, self.token_size])` 可能 reshape 失败，或者字段位置错位，导致 treatment/cost 读取错误。

建议：

将示例命令中的 `--token_size 13` 改为 `--token_size 14`，并将默认值改为：

```python
parser.add_argument('--token_size', type=int, default=14)
```

## 2. treatment/cost 字段索引用负数硬编码

位置：

- `demo/euen_base_4t.py:493`
- `demo/euen_base_4t.py:494`

当前代码：

```python
TREAT_INDEX = -3
COST_INDEX = -4
```

问题：

这依赖 token 字段顺序固定不变。当前 14 token 时：

- `-3` 等价于 index 11，即 `fake_treatment_id`
- `-4` 等价于 index 10，即 `cost`

但只要上游 `add_token` 增删字段或调整顺序，代码就会读错字段。

影响：

轻则训练直接报错，重则不报错但用错误字段训练，模型指标会失真。

建议：

至少将索引改为显式正索引并加注释：

```python
COST_INDEX = 10
TREAT_INDEX = 11
```

更好的方式是定义 token schema，通过字段名查索引。

## 3. 读取了 input_value，但模型完全没有使用特征值

位置：

- `demo/euen_base_4t.py:457`
- `demo/euen_base_4t.py:458`
- `demo/euen_base_4t.py:459`

当前代码：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
input_value = tf.cast(inputs[:, 1::2], tf.float32)
inputs = self.embedding_layer(input_index[:, :])
```

问题：

TFRecord 的 feature 是 libsvm 风格 `index:value`，代码拆出了 index 和 value，但只把 index 送入 embedding，`input_value` 完全没参与计算。

影响：

如果特征值不是全 1，例如分桶特征、连续值特征、权重特征，模型会丢失 value 信息，只知道“哪个特征出现了”，不知道“特征值是多少”。

建议：

根据特征工程语义确认：

- 如果所有特征都是 one-hot/index 特征，可以删除 `input_value`，避免误导。
- 如果 value 有意义，应将 embedding 与 value 相乘，例如 `embedding * input_value` 后再 flatten。

## 4. mask_zero=False，但 index 会减 1，padding 可能被误用

位置：

- `demo/euen_base_4t.py:107`
- `demo/euen_base_4t.py:457`

问题：

模型将输入 index 做了 `-1`：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
```

如果原始 padding index 是 0，那么减 1 后会变成 -1。`tf.nn.embedding_lookup` 对负索引通常不是合法的 padding 语义，可能报错，或在某些场景造成不可控行为。

同时 `mask_zero=False`，即使 padding 是 0，也不会被自动 mask。

影响：

padding 特征可能污染 embedding；如果出现 -1 index，可能导致训练失败。

建议：

确认上游 `fea_extract` 输出的 padding index 到底是什么。如果 padding 用 0，应避免直接 `-1`，或者对 padding 单独处理。

## 5. 导入了 fextract 里的类，但随后又本地重定义同名类

位置：

- `demo/euen_base_4t.py:56`
- `demo/euen_base_4t.py:57`
- `demo/euen_base_4t.py:98`
- `demo/euen_base_4t.py:143`

问题：

代码先导入：

```python
from fextract import LiteFeTFRecords
from fextract import IndexEmbedding
```

后面又在同一个文件里重新定义了 `IndexEmbedding` 和 `LiteFeTFRecords`，覆盖了前面的导入。

影响：

读代码的人容易误以为使用的是 `offline_bigdata/python/fextract/` 里的实现，但实际运行使用的是本文件内的实现。

建议：

二选一：

- 删除无效导入，只保留本文件定义。
- 删除本文件重复定义，统一使用 `fextract` 模块。

## 6. setup_gpu 设置 CUDA_VISIBLE_DEVICES 的时机偏晚

位置：

- `demo/euen_base_4t.py:40`
- `demo/euen_base_4t.py:59`
- `demo/euen_base_4t.py:758`

问题：

代码在 import TensorFlow 后才调用 `setup_gpu(args.gpu)` 设置 `CUDA_VISIBLE_DEVICES`。通常更推荐在 import TensorFlow 之前设置环境变量。

影响：

在某些环境下，TensorFlow 可能已经初始化 GPU 可见设备，导致 `CUDA_VISIBLE_DEVICES` 不完全生效。

建议：

将 GPU 环境变量设置提前到 import TensorFlow 前，或在程序入口尽早处理。

## 7. 训练函数依赖全局 optimizer

位置：

- `demo/euen_base_4t.py:565`
- `demo/euen_base_4t.py:788`

问题：

`predict_or_train_data_auuc` 函数内部直接使用了全局变量 `optimizer`：

```python
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

但函数签名里没有传入 optimizer。

影响：

函数可复用性差，也不利于单测。只要调用顺序变化，或者外部想复用该函数，就可能出现 `NameError` 或使用错误 optimizer。

建议：

将 optimizer 作为参数传入：

```python
def predict_or_train_data_auuc(..., optimizer=None):
```

训练时显式传入。

## 8. 每个 batch 都打印完整 token/treatment/cost，日志量过大

位置：

- `demo/euen_base_4t.py:502`
- `demo/euen_base_4t.py:504`
- `demo/euen_base_4t.py:511`

问题：

这些调试打印在每个 batch 都执行：

```python
print(t[0])
print("treat is: ", is_treat)
print("cost is: ", label_cost_7d)
```

影响：

大规模训练时日志会非常大，影响训练速度，也会让有效日志被淹没。

建议：

只在 debug 模式或第一个 batch 打印。例如：

```python
if args.debug_mode and batch_count == 0:
    ...
```

## 9. AUUC 早停只看 treatment 1

位置：

- `demo/euen_base_4t.py:648`
- `demo/euen_base_4t.py:805`

问题：

代码计算了 4 个 treatment 的 AUUC，但 early stopping 使用的是 `predict_delta_cvr_1`：

```python
if key == 'predict_delta_cvr_1':
    auuc_val_base = item[key]
```

影响：

如果 treatment 1 表现提升，但 treatment 2/3/4 下降，早停仍然可能认为模型变好。对于 4 treatment 模型，这个 best model 选择标准偏窄。

建议：

根据业务目标改成：

- 4 个 AUUC 的平均值
- 加权平均 AUUC
- 关键 treatment 的 AUUC
- 或同时约束 cost 指标

## 10. treatment 4 的 PCOC 配错了预测列

位置：

- `demo/euen_base_4t.py:679`

当前代码：

```python
treatment_control_pairs = [
    (1, 0, 'predict_delta_cvr_1'),
    (2, 0, 'predict_delta_cvr_2'),
    (3, 0, 'predict_delta_cvr_3'),
    (4, 0, 'predict_delta_cvr_3')
]
```

问题：

treatment 4 使用了 `predict_delta_cvr_3`，应该是 `predict_delta_cvr_4`。

影响：

treatment 4 的 PCOC 和 Cal-N 评估结果错误。

建议：

改为：

```python
(4, 0, 'predict_delta_cvr_4')
```

## 11. PCOC/Cal-N 结果计算后没有打印或保存

位置：

- `demo/euen_base_4t.py:696`
- `demo/euen_base_4t.py:700`

问题：

`valid_distribution` 中计算了：

```python
pcoc_results, cal_n_results = evaluate_pcoc_and_cal_n(df, args)
```

但没有打印、保存或返回。

影响：

这部分评估等于静默执行，用户无法看到结果，也无法追踪模型校准表现。

建议：

至少打印结果，或保存到 `plot_log/{version}` 下的 json/csv。

## 12. evaluation_train 中分桶统计结果没有使用

位置：

- `demo/euen_base_4t.py:706`
- `demo/euen_base_4t.py:708`

问题：

代码生成了 `stats_new75`：

```python
stats_new75 = simple_gap_delta_cvr(df, prob_col=ite_column, label_col='label')
```

但没有打印、保存或返回。

影响：

这段计算没有实际产出，浪费计算资源。

建议：

打印、保存 CSV，或者删除这段逻辑。

## 13. valid/test 日期参数共用 test_dt，无法表达验证集区间

位置：

- `demo/euen_base_4t.py:403`
- `demo/euen_base_4t.py:410`
- `demo/euen_base_4t.py:720`
- `demo/euen_base_4t.py:721`
- `demo/euen_base_4t.py:722`

问题：

train 支持 `train_start_dt` 和 `train_end_dt`，但 valid/test 都只使用 `test_dt`，且 valid 分支也使用 `args.test_dt`。

影响：

如果验证集或测试集是多天路径，当前参数表达能力不足；如果路径中没有日期格式占位，日期参数又变成无效配置。

建议：

增加：

```python
--valid_start_dt
--valid_end_dt
--test_start_dt
--test_end_dt
```

或者明确约束 valid/test 只能单日。

## 14. batch_then_map 解析 feature 时假设每条样本 feature 长度一致

位置：

- `demo/euen_base_4t.py:217`
- `demo/euen_base_4t.py:221`

问题：

代码将 batch 中全部 feature 拆平后，用总长度除以 batch size 再 reshape：

```python
feature = tf.reshape(feature, [-1, tf.shape(feature)[0] // tf.shape(label)[0]])
```

这隐含假设：batch 内每条样本 feature 长度完全相等。

影响：

如果上游 padding 没做好，或者部分样本 feature 数量不同，会 reshape 失败，或者产生错位。

建议：

在上游保证固定 `PADDING_SIZE * 2`，并在这里增加 shape/assert 检查。

## 15. 空数据集时 np.concatenate 会直接报错

位置：

- `demo/euen_base_4t.py:611`

问题：

如果 TFRecord 路径没有匹配到文件，或者 dataset 为空，`out_lists` 中各列表为空，执行：

```python
np.concatenate(list_name, axis=0)
```

会报错。

影响：

报错信息离真正原因较远，不容易定位是路径为空还是数据为空。

建议：

在 `_get_hdfs_path_list` 后检查文件数，在 batch 循环后检查 `batch_count`。

## 16. 没有检查 TFRecord 文件匹配数量是否为 0

位置：

- `demo/euen_base_4t.py:235`
- `demo/euen_base_4t.py:238`

问题：

代码只是打印：

```python
print(f"[LiteFeTFRecords] Found {len(tfr_path_list)} files")
```

但如果是 0，仍然继续构建 dataset。

影响：

后续会在训练/拼接阶段以不直观的方式失败。

建议：

如果 `len(tfr_path_list) == 0`，直接 `raise FileNotFoundError`。

## 17. cost loss 对 control 样本的处理可能不合理

位置：

- `demo/euen_base_4t.py:557`
- `demo/euen_base_4t.py:558`

当前 cost loss：

```python
tf.square(is_list[1] * cost_1 + is_list[2] * cost_2 + is_list[3] * cost_3 + is_list[4] * cost_4 - label_cost_7d)
```

问题：

control 样本的 `is_list[1..4]` 都是 0，所以预测 cost 为 0，然后仍然和 `label_cost_7d` 做差。

如果 control 的真实 cost 恒为 0，这没问题；但如果 control 也存在非零成本，loss 会强迫 control cost 为 0。更关键的是，control 样本也会参与 cost loss，可能改变 cost head 的整体优化权重。

建议：

确认 control cost 业务定义。如果 control 不训练 cost，可对 cost loss 加 treatment mask，仅在 treatment 样本上计算。

## 18. loss 没有区分 label loss 和 cost loss 权重

位置：

- `demo/euen_base_4t.py:557`
- `demo/euen_base_4t.py:558`

问题：

label loss 和 cost loss 直接相加，权重都是 1。

影响：

如果 label 和 cost 的量纲差异很大，某一个任务会主导训练。比如 cost 数值远大于 label，模型可能主要优化 cost。

建议：

增加参数：

```python
--label_loss_weight
--cost_loss_weight
```

并打印两个 loss 的单独值。

## 19. 保存模型时每个 epoch 都保存完整模型，存储开销大

位置：

- `demo/euen_base_4t.py:811`
- `demo/euen_base_4t.py:819`

问题：

每个 epoch 都保存一个完整模型：

```python
model.save(epoch_model_path)
```

同时 best model 也保存。

影响：

HDFS 存储开销大，训练时间也会被保存拖慢。

建议：

按需保存：

- 只保存 best
- 或每 N 个 epoch 保存
- 或增加 `--save_all_epochs` 参数

## 20. skip_eval 模式下 best model 选择逻辑不是真实指标

位置：

- `demo/euen_base_4t.py:807`
- `demo/euen_base_4t.py:808`

当前代码：

```python
average_auuc = float(epoch)
```

问题：

跳过评估时用 epoch 序号伪造指标，因此每个 epoch 都会比前一个 epoch 更好。

影响：

best model 最终一定是最后一个 epoch，不再代表最佳模型。

建议：

如果 `--skip_eval`，明确命名为 `last_model`，或者不执行 best AUUC 逻辑。

## 21. 部分参数未被使用

位置：

- `demo/euen_base_4t.py:732`
- `demo/euen_base_4t.py:733`
- `demo/euen_base_4t.py:734`
- `demo/euen_base_4t.py:735`
- `demo/euen_base_4t.py:736`

问题：

这些参数目前没有实际参与训练逻辑：

- `percentage`
- `saveDir`
- `fromPretrained`
- `scheduleDt`
- `startDate`

影响：

增加理解成本，容易让使用者误以为这些参数有效。

建议：

删除无用参数，或补齐对应功能。

## 22. 导入项较多未使用

位置：

- `demo/euen_base_4t.py:41`
- `demo/euen_base_4t.py:44`
- `demo/euen_base_4t.py:47`

问题：

例如 `BatchNormalization`、`Activation`、`datetime` 模块、`json` 等没有实际使用。

影响：

不影响运行，但降低代码清晰度。

建议：

清理无用 import。

## 23. draw_auuc 中对 DataFrame 切片直接赋值

位置：

- `demo/euen_base_4t.py:312`
- `demo/euen_base_4t.py:313`
- `demo/euen_base_4t.py:314`

问题：

`subset = prediction_pdf[...]` 后直接赋值：

```python
subset['exp_group'] = ...
subset['label'] = ...
```

这可能触发 pandas 的 `SettingWithCopyWarning`。虽然文件开头忽略了 warning，但潜在行为仍不够清晰。

建议：

改为：

```python
subset = prediction_pdf[prediction_pdf[real_treatment].isin([treatment, control_group])].copy()
```

## 24. simple_gap_delta_cvr 只按 treatment 1/control 统计

位置：

- `demo/euen_base_4t.py:279`
- `demo/euen_base_4t.py:280`

问题：

该函数里变量名仍是 `count_888/count_555`，并且只比较 `real_treatment == 1` 和 `real_treatment == 0`。

影响：

对 4 treatment 模型来说，这个函数不适合直接复用到 treatment 2/3/4，否则统计口径错误。

建议：

增加 treatment 参数：

```python
def simple_gap_delta_cvr(df, prob_col, label_col, treatment=1, control=0):
```

## 25. 运行示例中 train/valid/test 路径和日期参数容易混淆

位置：

- `demo/euen_base_4t.py:4`
- `demo/euen_base_4t.py:15`
- `demo/euen_base_4t.py:720`
- `demo/euen_base_4t.py:722`

问题：

示例命令只传了路径，没有传 `train_start_dt/train_end_dt/test_dt`。但 `load_from_tfr` 会使用这些默认日期格式化路径。

如果路径中没有 `%Y%m%d` 等 strftime 占位，日期参数不会改变路径；如果路径中有日期占位，默认日期又可能不符合实际数据。

建议：

在示例命令中补齐日期参数，或者在代码注释中说明：当路径不含日期占位时日期参数无效。

## 建议优先修复顺序

1. 修复 `token_size`：统一为 14。
2. 修复 treatment 4 的 PCOC 预测列：`predict_delta_cvr_4`。
3. 确认 `input_value` 是否应该参与模型计算。
4. 确认 padding/index 减 1 的逻辑是否安全。
5. 删除每 batch 的大日志打印。
6. 给空 TFRecord 路径和空 dataset 加明确报错。
7. 将 early stopping 指标从 treatment 1 改成更合理的 4 treatment 综合指标。
8. 拆分 label loss 和 cost loss，并加入权重。

