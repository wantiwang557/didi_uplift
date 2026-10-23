# EUEN ZILN Loss Experiments

本目录用于在 `demo/euen_test_feature_time/euen_4t_02.py` 基础上做 ZILN loss 实验。

## 文件

- `loss_iteration_memo.md`
  - 记录原版 EUEN、`euen_4t_ziln_01.py`、`euen_4t_ziln_01_batch.py`、`euen_4t_ziln_02.py`、`euen_4t_ziln_02_batch.py`、`euen_4t_ziln_03.py`、`euen_4t_ziln_03_batch.py`、`euen_4t_ziln_04.py`、`euen_4t_ziln_04_batch.py` 的 loss 公式、分母口径、代码实现和差异总结。

- `ziln_loss_experiment_doc.md`
  - ZILN loss 沉淀文档，包含设计思想、loss/还原公式、01/02/03/04 四个主实验说明，以及当前已有日志下的 AUUC 和 cost AUC 效果对比。

- `analyze_label_cost_distribution.py`
  - 分析 v1/v2/v3 train/test TFRecord 中 label 和 cost_7d 的分布。
  - 输出整体、control、各 treatment 的 0 值比例、正值比例、均值、正值均值、分位数和 max。
  - 保存 summary CSV、sample CSV 和 raw/log1p 直方图。
  - 数据路径不在脚本中维护，统一通过启动命令的 `--tfr_path` 传入。

- `euen_4t_ziln_01.py`
  - label/outcome 仍使用原 MSE。
  - cost head 输出 `zero_logit, mu, sigma_raw`。
  - cost loss 使用 ZILN NLL，只在 `treatment > 0` 样本内平均，没有被 control 样本稀释。
  - `predict_cost_7d_k` 使用 ZILN 期望：
    `P(cost > 0) * exp(mu + sigma^2 / 2)`。

- `euen_4t_ziln_01_batch.py`
  - label/outcome 仍使用原 MSE。
  - cost head 输出 `zero_logit, mu, sigma_raw`。
  - cost loss 使用 ZILN NLL，按原 EUEN cost MSE 口径做全 batch 平均，control 样本 cost loss 用 mask 置 0。
  - `predict_cost_7d_k` 使用 ZILN 期望：
    `P(cost > 0) * exp(mu + sigma^2 / 2)`。

- `euen_4t_ziln_02.py`
  - label/outcome 和 cost 都使用 ZILN NLL。
  - control/uplift head 输出 ZILN 参数残差，按原 EUEN residual 方式组合 factual outcome 参数。
  - AUUC 使用各 treatment outcome 期望与 control outcome 期望的差值：
    `E[y_k] - E[y_0]`。
  - `predict_cost_7d_k` 同样使用 ZILN 期望。
  - 当前 cost loss 只在 `treatment > 0` 样本内平均，没有被 control 样本稀释。

- `euen_4t_ziln_02_batch.py`
  - 模型结构、outcome ZILN 参数 residual、还原方式都和 `euen_4t_ziln_02.py` 一致。
  - 唯一核心差异：cost loss 改为全 batch 平均，control 样本 cost loss 用 mask 置 0，但仍进入 batch 分母。

- `euen_4t_ziln_03.py`
  - 保留原 EUEN 的 control + uplift residual 结构。
  - control/uplift/cost head 都输出 ZILN 参数。
  - outcome 侧先把 control/uplift 参数还原成期望标量，再按原 residual 结构组合 factual outcome，因此 label/outcome loss 使用 MSE。
  - AUUC 和落表输出直接使用还原后的 uplift residual 相加，和原版 EUEN 的 `predict_delta_cvr_k` 口径一致：
    `uplift_1 = E[u_tau_1]`，`uplift_k = E[u_tau_1] + E[u_tau_k]`。
  - `predict_delta_cvr_k` 是还原并相加后的标量 uplift。
  - `predict_cost_7d_k` 同样使用 ZILN 期望。
  - 当前 cost loss 只在 `treatment > 0` 样本内平均，没有被 control 样本稀释。

- `euen_4t_ziln_03_batch.py`
  - 模型结构、期望空间 residual 相加、还原方式都和 `euen_4t_ziln_03.py` 一致。
  - 唯一核心差异：cost loss 改为全 batch 平均，control 样本 cost loss 用 mask 置 0，但仍进入 batch 分母。

- `euen_4t_ziln_04.py`
  - 这是之前 `03` 的直接 outcome 版本。
  - label/outcome 和 cost 都使用 ZILN NLL。
  - 不保留原 EUEN 的 uplift residual 结构。
  - control 和 treatment 1/2/3/4 outcome head 各自直接输出 ZILN 参数，训练时按真实 treatment 选择 factual outcome 参数算 ZILN NLL。
  - AUUC 使用各 treatment outcome 期望与 control outcome 期望的差值：
    `E[y_k] - E[y_0]`。
  - `predict_cost_7d_k` 同样使用 ZILN 期望。
  - 当前 cost loss 只在 `treatment > 0` 样本内平均，没有被 control 样本稀释。

- `euen_4t_ziln_04_batch.py`
  - 模型结构、直接 outcome ZILN NLL、还原方式都和 `euen_4t_ziln_04.py` 一致。
  - 唯一核心差异：cost loss 改为全 batch 平均，control 样本 cost loss 用 mask 置 0，但仍进入 batch 分母。

## 注意

这些 ZILN 脚本保存出来的模型里，带 `_ziln_params` 的输出是 ZILN 参数，不是直接标量。如果后续要写 Hive 推理表，需要让推理脚本同样先把参数还原成期望值。`03` 的 uplift 需要还原 uplift residual 后直接相加，`04` 的 uplift 则是 treatment 期望减 control 期望。

## 分布分析启动命令

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/Loss_test
mkdir -p ./run_logs
```

```bash
nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/train_0511_0617/part-*" \
--data_name v1_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v1_train.log 2>&1 &

nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/test_0623_0706/part-*" \
--data_name v1_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v1_test.log 2>&1 &

nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/train_0511_0617/part-*" \
--data_name v2_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v2_train.log 2>&1 &

nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/test_0723_0806/part-*" \
--data_name v2_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v2_test.log 2>&1 &

nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/train_0511_0722/part-*" \
--data_name v3_train \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v3_train.log 2>&1 &

nohup python analyze_label_cost_distribution.py \
--tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/test_0723_0806/part-*" \
--data_name v3_test \
--token_size 15 \
--batch_size 8192 \
> ./run_logs/analyze_cost_v3_test.log 2>&1 &
```

```bash
tail -f ./run_logs/analyze_cost_v1_train.log
ps -ef | grep "analyze_label_cost_distribution.py" | grep -v grep
```

## 迭代规范

每次新增模型迭代脚本时，需要同步修改：

- 文件名和脚本顶部 `Version`。
- 脚本顶部改动点。
- 示例启动命令里的脚本名、`--version`、`--model_save_path` 和日志文件名。
- 本 README 或项目 memo 中的模型说明。
