# EFIN 多 Treatment Uplift 模型迭代文档

## 1. 背景与动机

当前业务目标是在 RCT 券包实验数据上，对 4 个 treatment 相对 control 的增量效果进行排序建模，核心评价指标是 AUUC。已有 S-learner / EUEN 方案在效果上较强，但 treatment 信息的使用方式仍然偏“隐式”：模型主要通过 `treatment_id` 或不同 treatment tower 学习组间差异，而没有显式建模券包本身的结构化属性，例如券总数量、首张券面额、留存券面额、券包总金额等。

这会带来两个问题：

1. 模型能区分不同 treatment，但较难理解 treatment 之间为什么不同。比如 `8+5张3` 和 `10+5张3` 共享了留存券面额和券数量，但首张券面额不同；`6张5` 和 `留存6张3元` 券数量相同，但面额结构不同。只用 treatment_id 时，这些关系需要完全靠数据学习。
2. 模型缺少显式的 treatment-aware 特征交互。即使 EUEN/S-learner 有较强 shared-bottom，它们更像是在学习 `x` 或 `(x,t)` 到 outcome/uplift 的映射，没有单独回答“哪些用户特征对当前 treatment 更敏感”。

因此本轮迭代尝试引入 EFIN 思路：对 treatment 做显式表征，并利用 treatment-aware attention 建模 treatment 特征和非 treatment 用户特征之间的交互，得到当前 treatment 条件下的敏感用户表征，再用于 uplift 预测。

换句话说，EFIN 要填补的是：

- 从只用 treatment id，升级为使用 treatment 结构化属性。
- 从普通 shared-bottom / tower 结构，升级为显式建模 treatment 与用户特征交互。
- 从只追求预测效果，进一步获得可解释的 treatment-sensitive feature representation。

## 2. 当前结论

从当前实验结果看，原始 EFIN 思路本身有效，但直接套用论文结构在当前业务数据上存在瓶颈。主要瓶颈不是 attention 公式，而是 tau 分支的用户表征能力不足，以及 cost_loss 与 treatment-aware attention 的共享方式可能干扰 uplift 排序。

关键结论如下：

1. 显式 treatment 表征是必要的。共享 tau tower 如果不显式看到 treatment_repr，区分不同券包的能力不足。
2. 只依赖 treatment-aware attention 压缩后的 32 维用户向量不够，tau 分支需要更强的全局用户表征。
3. 直接把 self-attention/control 分支表征拼给 tau 会明显伤害效果，说明 natural response 表征和 uplift 排序表征不能粗暴混用。
4. 引入 raw embedding 的 MLP 用户底座可以提升 EFIN，但独立 tau_user_bottom 仍不如强 shared-bottom。
5. 当前最有效方向是 EFIN05：使用 shared_bottom 共享用户表征，同时保留 treatment-aware attention 和显式 treatment_repr。
6. cost_loss 可以回归为与 S-learner/EUEN 一致的 `label_loss + cost_loss`，但 cost 分支不直接使用 treatment_aware_repr，而是使用 `shared_user_repr + treatment_repr`，避免 cost 目标直接拉偏 treatment-aware attention。

当前 EFIN05 仍在运行中，已观察到 epoch1 validation AUUC 明显提升：

```text
T1: 0.824749
T2: 0.834198
T3: 0.842793
T4: 0.827426
mean_auuc_val: 0.832292
```

## 3. 论文支撑：EFIN 原始模型

EFIN 的核心思想是显式建模 feature interaction，并将自然响应和 treatment 增量响应拆成两个模块。

### 3.1 Feature Encoder

EFIN 不只编码用户/context 特征 `x_i`，也编码 treatment 特征 `t_i`。

对于 treatment 中的连续特征，论文使用全连接层编码：

```text
e_ij^t = W_j * t_ij + b_j
```

对于稀疏特征，使用 embedding lookup：

```text
e_ij^t = lookup(E_j, e_ij)
```

在当前券包业务里，我们把 treatment 显式拆成：

```text
treatment_id
券总数量
首张券面额
留存券面额
券包总金额
```

并分别 embedding 后 concat 成 `treatment_repr`。

### 3.2 Self-interaction Module

Self-interaction 模块只使用非 treatment 特征，目标是预测自然响应，也就是 control / natural response：

```text
slot_embedding
-> self-attention
-> control_tower
-> y_control
```

论文动机是：在不引入 treatment 信息时，学习用户在自然状态下的响应能力。

### 3.3 Treatment-aware Interaction Module

Treatment-aware attention 使用 treatment 表征作为条件，给用户特征 slot 分配注意力权重：

```text
alpha_j^i = Softmax(W_t0^T ReLU(W_t1 e_i^t + W_t2 e_ij^x + b_t2))
```

然后加权得到当前 treatment 条件下的用户表征：

```text
e_i^{xt} = sum_j alpha_j^i * e_ij^x
```

再用 MLP 预测 treatment 对用户的增量：

```text
tau_k(x_i) = MLP(e_i^{xt})
```

### 3.4 Loss

论文整体 loss 可以写成：

```text
L_EFIN = L_S + L_T + L_C + regularization
```

当前 RCT 数据中，confounding constraint 不是重点，因此主要关注：

```text
control样本:
y_control -> label

treatment样本:
stop_gradient(y_control) + tau_k -> label
```

以及 cost 辅助任务：

```text
cost_k -> label_cost_7d
```

## 4. Baseline 与版本迭代

### 4.1 S-learner baseline

代码：`demo/EFIN/s_learner_4t.py`

结构：

```text
slot_embedding flatten
-> shared_bottom: 2048 -> 1024 -> 512
-> concat treatment_id embedding
-> outcome_tower
-> y(t)

uplift_k = y(k) - y(0)
```

cost 分支：

```text
shared_user_repr + treatment_id embedding
-> cost_tower
-> cost_k
```

loss：

```text
loss = label_loss + cost_loss
```

结果：

```text
best_epoch: 3
best mean_auuc_val: 0.846012
mean_auuc_test: 0.846012
```

结论：S-learner 是强 baseline，说明 shared-bottom 的全局用户表征能力很强。

### 4.2 原始 EFIN

代码：`demo/EFIN/efin_4t.py`

结构：

```text
slot_embedding
-> feature_projection
-> self-attention
-> control_tower
-> y_control
```

```text
treatment_id + treatment属性
-> embedding
-> mean pooling
-> treatment_repr
```

```text
slot_repr + treatment_repr
-> treatment-aware attention
-> treatment_aware_repr
-> uplift_tower
-> tau_k
```

cost 分支：

```text
concat(treatment_aware_repr, treatment_repr)
-> cost_tower
-> cost_k
```

loss：

```text
loss = outcome_loss + cost_loss
```

结果：

```text
best_epoch: 4
best mean_auuc_val: 0.805389
mean_auuc_test: 0.805389
```

迭代原因：

- treatment 属性用 mean pooling，会压缩不同 treatment 属性之间的差异。
- tau 只使用 `treatment_aware_repr`，用户表征较窄。
- cost_loss 直接更新 treatment-aware attention，可能让 attention 更偏向成本目标，而不是 uplift 敏感性。

### 4.3 EFIN01

代码：`demo/EFIN/EFIN_4t_01.py`

主要改动：

1. treatment 表征从 pooling 改成 concat。
2. self-attention dim 与 treatment-aware attention dim 解耦。
3. cost_loss_weight 置 0，先隔离 cost 对 AUUC 的影响。

结构变化：

```text
treatment_repr = concat(
  treatment_id_embedding,
  coupon_count_embedding,
  first_coupon_face_embedding,
  retention_coupon_face_embedding,
  package_total_amount_embedding
)
```

```text
self-attention dim = 8
treatment-aware dim = 32
```

结果：

```text
best_epoch: 6
best mean_auuc_val: 0.814507
mean_auuc_test: 0.814507
```

结论：显式 treatment 属性 concat 和 dim 解耦有效，相比原始 EFIN 提升约 0.0091。

### 4.4 EFIN02

代码：`demo/EFIN/EFIN_4t_02.py`

主要改动：

尝试把 control/self-attention 分支的用户表征加入 tau：

```text
self-attention output
-> flatten
-> stop_gradient
-> Dense 256
-> self_user_repr
```

```text
tau_input = concat(
  self_user_repr,
  treatment_aware_repr,
  treatment_repr
)
```

结果：

```text
epoch1 mean_auuc_train: 0.657593
未跑出有效 val/test 汇总
```

结论：

效果大幅下降。原因判断是 self-attention/control 分支主要学习 natural response 或基础转化倾向，把它直接拼给 tau 后，tau tower 容易用基础响应信号解释 factual label，从而损伤 uplift 排序。

### 4.5 EFIN03

代码：`demo/EFIN/EFIN_4t_03.py`

主要改动：

去掉 EFIN02 的 self_user_repr，只保留显式 treatment 表征进入 tau：

```text
tau_input = concat(
  treatment_aware_repr,
  treatment_repr
)
```

结果：

```text
运行参数: BATCH_SIZE=512
best_epoch: 7
best mean_auuc_val: 0.814826
mean_auuc_test: 0.814826
```

结论：

显式 treatment_repr 对共享 tau tower 有帮助，但只依赖 32 维 `treatment_aware_repr` 的用户信息仍然有限。

### 4.6 EFIN04

代码：`demo/EFIN/EFIN_4t_04.py`

主要改动：

新增 tau 分支独立用户底座 `tau_user_bottom`：

```text
slot_embedding
-> flatten
-> 1024 -> 512 -> 256
-> tau_user_repr
```

tau 输入：

```text
tau_input = concat(
  tau_user_repr,
  treatment_aware_repr,
  treatment_repr
)
```

cost 分支仍然是：

```text
cost_input = concat(treatment_aware_repr, treatment_repr)
```

loss：

```text
loss = outcome_loss
cost_loss_weight = 0
```

结果：

```text
best_epoch: 2
best mean_auuc_val: 0.819310
mean_auuc_test: 0.819310
```

结论：

补充全局用户表征有效，说明 EFIN tau 分支的主要瓶颈之一是用户排序信息不足。但独立 tau_user_bottom 相比 S-learner/EUEN 的 shared-bottom 仍有差距。

### 4.7 EFIN05

代码：`demo/EFIN/EFIN_4t_05.py`

主要改动：

1. 引入方案 A：control / tau / cost 三分支共享 `shared_bottom`。
2. `shared_bottom` 对齐 S-learner/EUEN：

```text
slot_embedding
-> flatten
-> 2048 -> 1024 -> 512
-> shared_user_repr
```

3. control 分支：

```text
shared_user_repr
-> control_tower
-> y_control
```

4. treatment-aware 分支：

```text
slot_embedding
-> treatment_feature_projection
-> treatment-aware attention with treatment_repr
-> treatment_aware_repr
```

5. tau 分支：

```text
tau_input = concat(
  shared_user_repr,
  treatment_aware_repr,
  treatment_repr
)
-> uplift_tower
-> tau_k
```

6. cost 分支：

```text
cost_input = concat(
  shared_user_repr,
  treatment_repr
)
-> cost_tower
-> cost_k
```

7. treatment 样本中固定：

```python
y_control_for_treatment = tf.stop_gradient(y_control)
y_treat = y_control_for_treatment + tau_k
```

8. loss 对齐 S-learner/EUEN：

```python
loss = outcome_loss + cost_loss
```

当前结果：

```text
epoch1 mean_auuc_train: 0.798012
epoch1 mean_auuc_val: 0.832292  # 用户当前观测，训练仍在运行
```

epoch1 validation 明细：

```text
T1: 0.824749
T2: 0.834198
T3: 0.842793
T4: 0.827426
mean: 0.832292
```

结论：

EFIN05 是目前最有效的 EFIN 方向。它保留了 EFIN 的 treatment-aware interaction，又补上了 S-learner/EUEN 中强 shared-bottom 用户表征的优势，同时避免 cost_loss 直接训练 treatment-aware attention。

## 5. 实验结果汇总

| 模型 | 关键结构 | Loss | best / 当前 val AUUC | test AUUC | 结论 |
|---|---|---|---:|---:|---|
| S-learner | shared_bottom + treatment_id embedding + outcome tower | label + cost | 0.846012 | 0.846012 | 强 baseline |
| 原始 EFIN | self-attention + treatment-aware attention + cost共享attention | outcome + cost | 0.805389 | 0.805389 | 原始结构直接迁移效果不足 |
| EFIN01 | treatment concat + dim解耦 + 去cost_loss | outcome | 0.814507 | 0.814507 | 显式 treatment 与 dim 解耦有效 |
| EFIN02 | 加入 self-attention 用户表征到 tau | outcome | 无有效 val | 无 | natural response 表征污染 tau |
| EFIN03 | tau concat treatment_repr | outcome | 0.814826 | 0.814826 | treatment 条件有效，但用户表征仍弱 |
| EFIN04 | tau_user_bottom + treatment-aware + treatment_repr | outcome | 0.819310 | 0.819310 | 全局用户表征有效 |
| EFIN05 | shared_bottom + treatment-aware + cost_loss | outcome + cost | 0.832292 当前 epoch1 | 运行中 | 当前最优方向 |

## 6. 后续计划

1. 等待 EFIN05 完整收敛，确认 best val/test AUUC 是否持续超过 EFIN04，并观察是否接近 S-learner。
2. 增加 control 分支诊断指标：

```text
control组 y_control AUC / MSE / MAE / calibration
```

3. 对 EFIN05 做 cost_loss 消融：

```text
loss = outcome + cost
loss = outcome + 0.1 * cost
loss = outcome + 0.01 * cost
```

4. 尝试每个 treatment 独立 tau tower，靠近 EUEN 多 treatment head 结构。
5. 尝试 shared_bottom 加 L2 / dropout，防止过拟合。
6. 进一步分析 treatment-aware attention 的可解释性，输出不同 treatment 下 attention 较高的用户特征 slot。

