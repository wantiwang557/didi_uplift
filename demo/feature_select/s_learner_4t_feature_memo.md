# WWT 4-Treatment S-Learner Memo

本文用于说明 `s_learner_4t_feature.py` 和 `test_hive_s_learner.py` 的当前实现口径，方便后续和 EUEN、EFIN 实验做公平对比。

## 1. 实验定位

当前脚本实现的是最基础的 4-treatment S-learner：

```text
f(x, t) -> response label
uplift_k = f(x, T_k) - f(x, control)
```

也就是说，模型训练阶段预测的是各 treatment 下的 response/label，不直接训练 uplift label；评估和落表阶段再用 treatment response 减 control response 得到 uplift。

本版本只加入 `treatment_id` embedding，还没有加入张数、面额、门槛、总面额等显式 treatment 属性，也没有 EFIN 的 treatment-aware attention。

## 2. 数据输入

数据链路继续复用当前 EUEN/特征筛选项目的 TFRecord：

```text
label:   finish_cnt_7d
feature: index:value,index:value,...
token:   fin_order_wxmini_28d,group_name_eve,pid,oneid,...,cost_7d,fake_treatment_id,flag,label_dt
```

训练脚本中关键 token 下标仍是：

```text
cost_7d           = token[:, -4] = token[:, 10]
fake_treatment_id = token[:, -3] = token[:, 11]
```

`.slots` 文件仍然用于从全量 TFRecord 中选择某套特征方案，例如 baseline/no_sirius/replace_order。若传入 `--feature_slot_file`，脚本会自动把 `PADDING_SIZE` 改成 slot 文件行数。

## 3. Treatment 编码

S-learner 中 control 和 4 个 treatment 都会编码：

```text
0 = control_group
1 = 留存6张3元
2 = 8元留存5张3
3 = 留存6张5元
4 = 10元留存5张3
```

代码中有一张独立 treatment embedding 表：

```text
shape = [treats_num + 1, treatment_emb_dim]
默认 = [5, 16]
```

它没有放入原来的大 `IndexEmbedding`，原因是 treatment 是干预动作，不是普通用户/上下文特征；并且推理时需要对同一个用户枚举 `control/T1/T2/T3/T4`，单独 treatment encoder 更方便后续扩展到显式 treatment 属性和 EFIN 交互。

## 4. 模型结构

核心类是 `SLearner`。

用户特征链路：

```text
feature index
  -> IndexEmbedding
  -> flatten
  -> shared_bottom
  -> user_repr
```

默认维度示例：

```text
PADDING_SIZE = 473
EMBEDDING_SIZE = 8
flatten 后为 473 * 8 = 3784
emb_dim = 2048
shared_bottom: 3784 -> 2048 -> 1024 -> 512
```

response 预测链路：

```text
user_repr + treatment_embedding(t)
  -> outcome_tower
  -> response_t
```

一次 forward 会枚举并输出：

```text
control
response_1
response_2
response_3
response_4
cost_1
cost_2
cost_3
cost_4
```

注意：模型保存的输出是 9 个字段，没有 `control_logit_fix`，也没有 EUEN 残差 head。

## 5. Loss 口径

每条样本只用自己真实 treatment 对应的 response 拟合真实 label。

```text
control 样本 -> y_pred = control
T1 样本      -> y_pred = response_1
T2 样本      -> y_pred = response_2
T3 样本      -> y_pred = response_3
T4 样本      -> y_pred = response_4
```

outcome loss：

```text
mean((y_pred - finish_cnt_7d)^2)
```

cost loss 使用另一个共享 S-learner tower：

```text
user_repr + treatment_embedding(t)
  -> cost_tower
  -> cost_t
```

cost 只在 treatment 样本上训练：

```text
T1 -> cost_1 拟合 cost_7d
T2 -> cost_2 拟合 cost_7d
T3 -> cost_3 拟合 cost_7d
T4 -> cost_4 拟合 cost_7d
control 不训练 cost
```

因此当前模型可以理解为：

```text
共享用户表征 + 两个 S-learner tower
1. outcome_tower: f_y(x, t) -> response label
2. cost_tower:    f_cost(x, t) -> cost_7d
```

## 6. Uplift 输出口径

训练和模型 forward 输出的是 response，评估/落表阶段才计算 uplift：

```text
predict_delta_cvr_1 = response_1 - control
predict_delta_cvr_2 = response_2 - control
predict_delta_cvr_3 = response_3 - control
predict_delta_cvr_4 = response_4 - control
```

`test_hive_s_learner.py` 也按同样口径写 Hive：

```text
uplift_1 = response_1 - control
uplift_2 = response_2 - control
uplift_3 = response_3 - control
uplift_4 = response_4 - control
```

不要用旧的 `test_hive.py` 加载该模型，因为旧 EUEN 推理脚本会按残差口径处理输出。

## 7. 指标和早停

AUUC 分别计算 4 个 treatment 相对 control 的 uplift 排序能力：

```text
predict_delta_cvr_1 vs T1/control
predict_delta_cvr_2 vs T2/control
predict_delta_cvr_3 vs T3/control
predict_delta_cvr_4 vs T4/control
```

当前早停指标已经改为 4 个 AUUC 的平均值：

```text
mean_auuc = mean(AUUC_T1, AUUC_T2, AUUC_T3, AUUC_T4)
```

每个 epoch 验证集评估后，如果 `mean_auuc` 超过历史最好值，就保存 best model；如果连续 `patience` 次没有提升，则 early stop。

除了 AUUC，脚本仍保留：

```text
cost xAUC: 对 cost_1~4 在对应 treatment 样本上的排序能力
PCOC: 平均预测 uplift / 真实 uplift，越接近 1 越好
Cal-N: 分桶 PCOC 的对称误差 RMS，越接近 0 越好
simple_gap_delta_cvr: 分桶查看真实 treatment-control label gap
```

## 8. 与 EUEN 的核心差异

EUEN 口径：

```text
x -> shared_bottom -> control head + 多个 uplift/cost head
treatment 主要是输出分支
```

当前 S-learner 口径：

```text
x + treatment_id -> 共享 response 函数
treatment 是模型输入
response 相减得到 uplift
```

这意味着其他 treatment 样本不会把自己的 label 直接当作 control label 训练，但会更新共享的用户表征和 outcome/cost tower，因此会间接影响 `f(x, control)`。这是 S-learner 的共享学习特性。

## 9. 后续改进思路

建议按以下顺序迭代，便于拆清楚收益来源：

```text
1. 当前基础 S-learner: 只用 treatment_id embedding
2. S-learner + treatment 显式属性: 张数、面额、门槛、总面额、是否 control
3. EFIN-lite: treatment 表征与 slot embedding 做 treatment-aware attention
4. EFIN-residual: 在 EFIN-lite 上尝试 T1 基底 + T2/T3/T4 残差
5. Intervention constraint: 如果数据不是严格 RCT，再加入分布约束/去偏模块
```

与 EUEN 公平比较时，应保持同一份 train/test TFRecord、同一份 `.slots`、同一套 label/treatment/cost 口径，只切换模型脚本和模型保存路径。
