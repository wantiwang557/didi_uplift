# euen_4t_feature.py 特征筛选版代码解读

本文说明 `demo/feature_select/euen_4t_feature.py` 相对原始 EUEN 的改动、运行方式，以及现在如何从全量 TFRecord 中选择不同方案的特征。

## 1. 目标

你现在希望比较三套特征方案：

```text
方案1 baseline:
sirius画像 + 补贴环境 + 双端 + 有端 + 券敏感 = 473

方案2 no_sirius:
补贴环境 + 双端 + 有端 + 券敏感 = 279

方案3 replace_sirius_with_xtlx_order:
补贴环境 + 双端 + 有端 + 券敏感 + 新版小天狼星订单信息 = 372
```

为了避免三套方案各自拉数，当前推荐流程是：

```text
先抽一次 609 全量特征 TFRecord
  -> 模型读取全量 feature
  -> 根据 feature_slot_file 选择其中 473 / 279 / 372 个 slot
  -> 使用同一套 EUEN 网络训练
```

这样可以保证三套方案的样本、label、treatment、cost 完全一致，只比较特征集合差异。

## 2. 什么是 slot

`slot` 是 `feature_conf` 中的行号，也就是一条样本中第几个特征位置。

例如：

```text
feature_baseline_train.conf 第 0 行:
name=i_lisan_sirius_city_id;depend=sirius_city_id;...

feature_baseline_train.conf 第 1 行:
name=i_lisan_dm_user_base_oneid_info_didi_silent_days;depend=dm_user_base_oneid_info_didi_silent_days;...
```

则：

```text
slot 0 = sirius_city_id
slot 1 = dm_user_base_oneid_info_didi_silent_days
```

当前 609 个特征全部是 `S_slice_binning`。一个特征分箱后会有多个可能 index，但每条样本只会落到其中一个 index。

例如：

```text
slot 0 = sirius_city_id
样本 A 可能是 12:1
样本 B 可能是 15:1
样本 C 可能是 18:1
```

所以：

```text
一个 slot = 一个原始/变换特征位置
一个 slot 在一条样本里 = 一个 index:value
一个 slot 在不同样本里 = 可能出现不同 index
```

因此我们按 slot 筛选，而不是按 index 筛选。

## 3. slot 文件生成

脚本路径：

```text
offline_bigdata/python/feature_select/build_feature_slot.py
```

推荐在项目根目录运行：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt

python offline_bigdata/python/feature_select/build_feature_slot.py \
  --feature_conf offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf \
  --sql offline_bigdata/script/sqls/baseline/feature_baseline_train.sql \
  --out_dir demo/feature_select \
  --prefix baseline_
```

如果希望更整洁，也可以输出到子目录：

```bash
--out_dir demo/feature_select/feature_slots/baseline
```

如果直接输出到 `demo/feature_select`，会得到：

```text
demo/feature_select/baseline_baseline.slots
demo/feature_select/baseline_no_sirius.slots
demo/feature_select/baseline_replace_sirius_with_xtlx_order.slots
demo/feature_select/baseline_all_superset.slots
demo/feature_select/baseline_summary.json
```

检查行数：

```bash
wc -l demo/feature_select/*.slots
```

期望：

```text
473 baseline_baseline.slots
279 baseline_no_sirius.slots
372 baseline_replace_sirius_with_xtlx_order.slots
609 baseline_all_superset.slots
```

## 4. 数据准备流程

特征筛选版模型要求 TFRecord 是全量 609 特征。

完整流程：

```text
1. 生成全量 feature_conf / parser_conf
2. 用 baseline 全量 parser/conf/sql extract TFRecord
3. 生成三套 slot 文件
4. 分别运行 euen_4t_feature.py
```

注意：旧路径下的 `part-*` 不一定是全量特征。它是不是全量，取决于当时 extract 使用的 `parser.conf` 和 `feature.conf`。

本次应使用：

```text
offline_bigdata/script/feature_confs/baseline/parser_baseline_train.conf
offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf
offline_bigdata/script/sqls/baseline/feature_baseline_train.sql
```

生成新的全量 TFRecord 后，再传给模型。

## 5. euen_4t_feature.py 的新增参数

新增：

```python
parser.add_argument(
    '--feature_slot_file',
    type=str,
    default='',
    help='0-based feature_conf slot list for feature selection'
)
```

运行模型时传入：

```bash
--feature_slot_file "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/baseline_no_sirius.slots"
```

如果不传 `--feature_slot_file`，模型退化为原始逻辑，吃 TFRecord 中全部特征。

## 6. 新增函数：load_feature_slot_indices

代码：

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
        raise ValueError(...)

    if len(slots) != len(set(slots)):
        raise ValueError(...)

    print(...)
    return slots
```

作用：

```text
读取 .slots 文件
  -> 得到需要保留的 slot 行号
  -> 检查空文件和重复 slot
  -> 返回整数列表
```

例如 `baseline_no_sirius.slots` 可能包含：

```text
194
195
196
...
```

表示保留从补贴环境开始的那些特征 slot。

## 7. 自动覆盖 PADDING_SIZE

main 中新增：

```python
args.feature_slot_indices = load_feature_slot_indices(args.feature_slot_file)
if args.feature_slot_indices is not None:
    args.PADDING_SIZE = len(args.feature_slot_indices)
```

作用：

```text
baseline slot 文件 473 行 -> PADDING_SIZE = 473
no_sirius slot 文件 279 行 -> PADDING_SIZE = 279
replace slot 文件 372 行 -> PADDING_SIZE = 372
```

所以运行特征筛选版模型时，不建议再传旧的：

```bash
--PADDING_SIZE 448
```

即使传了，也会被 slot 文件行数覆盖。

## 8. dataloader 中的特征选择

改动在 `load_from_tfr` 中：

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

原始 TFRecord 全量 feature：

```text
[idx0,value0, idx1,value1, idx2,value2, ..., idx608,value608]
```

reshape 后：

```text
slot0 = [idx0, value0]
slot1 = [idx1, value1]
slot2 = [idx2, value2]
...
slot608 = [idx608, value608]
```

假设 `.slots` 文件是方案2，包含 279 个 slot：

```python
selected_pairs = tf.gather(feature_pairs, slot_tensor, axis=1)
```

会只保留这 279 个 slot。

最后 reshape 回模型原来需要的格式：

```text
[idx194,value194, idx195,value195, ...]
```

因此后续 `EUEN.call` 不需要改。

## 9. 为什么 EUEN.call 不改

原始 `EUEN.call` 期望输入格式是：

```text
index,value,index,value,...
```

特征筛选后，格式仍然是：

```text
index,value,index,value,...
```

只是长度变了。

原始：

```text
448 个 slot -> input_len = 896
```

现在：

```text
473 个 slot -> input_len = 946
279 个 slot -> input_len = 558
372 个 slot -> input_len = 744
```

`EUEN.call` 仍然可以这样拆：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
input_value = tf.cast(inputs[:, 1::2], tf.float32)
```

然后使用 index 做 embedding：

```python
inputs = self.embedding_layer(input_index[:, :])
```

当前代码仍然不使用 `input_value`，这和原始 EUEN 保持一致，保证三套方案只比较特征集合差异。

## 10. 网络结构是否改变

网络结构不改变。

仍然是：

```text
IndexEmbedding
  -> shared bottom
  -> control head
  -> 4 个 uplift head
  -> 4 个 cost head
```

变化的是输入 slot 数：

```text
PADDING_SIZE = 473 / 279 / 372
```

因此 embedding flatten 后维度会随方案变化：

```text
473 * EMBEDDING_SIZE
279 * EMBEDDING_SIZE
372 * EMBEDDING_SIZE
```

后面的 Dense 层会根据实际输入维度自动 build。

## 11. DENSE_FEA_SIZE 设置

`DENSE_FEA_SIZE` 不是特征数量，不应该设为 473 / 279 / 372。

它表示全局离散分箱 index 的 embedding 表大小。

当前全量 `feature_conf` 粗略估计需要大于旧的 3300，建议先用：

```bash
--DENSE_FEA_SIZE 5000
```

如果日志出现 embedding index out of range，再继续调大。

## 12. 运行命令

在 feature_select 目录运行：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select
mkdir -p run_logs models
```

方案1 baseline：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_train_xxx/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_valid_xxx/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_test_xxx/part-*" \
--feature_slot_file "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/baseline_baseline.slots" \
--model_save_path "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/models/euen_baseline" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_baseline" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 2 > ./run_logs/euen_baseline.log 2>&1 &
```

方案2 no_sirius：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_train_xxx/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_valid_xxx/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_test_xxx/part-*" \
--feature_slot_file "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/baseline_no_sirius.slots" \
--model_save_path "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/models/euen_no_sirius" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_no_sirius" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 2 > ./run_logs/euen_no_sirius.log 2>&1 &
```

方案3 replace_sirius_with_xtlx_order：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_train_xxx/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_valid_xxx/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline_test_xxx/part-*" \
--feature_slot_file "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/baseline_replace_sirius_with_xtlx_order.slots" \
--model_save_path "/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select/models/euen_replace_sirius_with_xtlx_order" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_replace_sirius_with_xtlx_order" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 2 > ./run_logs/euen_replace_sirius_with_xtlx_order.log 2>&1 &
```

如果暂时没有 valid，可以把 `--valid_tfr_path` 指向 test，但这会让 best model 选择使用 test 数据，不适合严谨最终结论。

## 13. 指标计算

特征筛选版沿用原始指标逻辑：

- 每个 epoch 在 train 上计算 train AUUC/AUC。
- 每个 epoch 在 valid 上计算 val AUUC/AUC，并用 val AUUC 选择 best model。
- 训练结束后在 test 上计算最终 AUUC/AUC。
- 训练结束后在 test 上计算 PCOC/Cal-N。

如果传 `--skip_eval`，会跳过 valid，同时当前代码也会跳过训练结束后的 test 流程。

## 14. 注意事项

1. 三套方案必须使用同一份全量 TFRecord。
2. 三套方案只换 `--feature_slot_file`、`--version`、`--model_save_path`、日志文件名。
3. 不要再传旧的 `--PADDING_SIZE 448`。
4. `DENSE_FEA_SIZE` 建议先用 5000。
5. `token_size` 仍然是 14。
6. `TREAT_INDEX=-3`、`COST_INDEX=-4` 依赖 token 顺序，确保 `add_token` 是一行且顺序不变。
7. 当前 PCOC 中 treatment 4 的预测列建议确认并修正为 `predict_delta_cvr_4`。
