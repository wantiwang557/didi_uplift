# EUEN Loss 迭代 Memo

本文用于记录 EUEN 损失函数迭代实验，重点说明原版 EUEN 的 loss 口径，以及当前 ZILN loss 脚本的差异、公式和代码实现位置。

## 1. 原版 EUEN Loss

参考代码：

```text
demo/euen_test_feature_time/euen_4t_02.py
```

原版 EUEN 的模型结构包括：

- 1 个 control head：预测 control 组 outcome。
- 4 个 uplift head：预测 treatment 1/2/3/4 相对 control 的 uplift 残差。
- 4 个 cost head：预测 treatment 1/2/3/4 下的 cost 标量。

其中 `control_logit_fix` 不是额外 head，而是 `tf.stop_gradient(c_head)` 得到的 stop-gradient control 输出。

训练时，每个 batch 会同时包含 control 和各 treatment 样本。模型会对每条样本输出所有 head，然后用真实 treatment 的 mask 选择该样本应该参与 factual loss 的分支。

原版 loss 代码：

```python
loss1 = tf.reduce_mean(
    tf.square(
        is_list[0] * uc
        + is_list[1] * ut_1
        + is_list[2] * ut[0]
        + is_list[3] * ut[1]
        + is_list[4] * ut[2]
        - label
    )
    + tf.square(
        is_list[1] * cost_1
        + is_list[2] * cost_2
        + is_list[3] * cost_3
        + is_list[4] * cost_4
        - label_cost_7d
    )
)
```

公式上等价于：

> **L<sub>origin</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_mse</sup> + (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>cost_mse</sup>**

其中：

- outcome loss 对全 batch 平均，control 和 treatment 都参与。
- cost loss 也是对全 batch 平均。
- control 样本没有 cost head，cost 预测项为 0。
- 如果 control 的真实 `cost_7d = 0`，则 control 的 cost loss 数值为 0，但这个 0 仍进入全 batch 的平均分母，因此会稀释 cost loss 的整体权重。

## 2. ZILN Loss 原理

ZILN 是 Zero-Inflated Lognormal，适用于大量 0 加正值长尾的非负目标，例如 cost、LTV、消费金额等。

每个 ZILN head 输出 3 个参数：

> **(zero_logit, μ, sigma_raw)**

代码中：

```python
zero_logit, mu, sigma_raw = tf.split(params, 3, axis=-1)
sigma = tf.nn.softplus(sigma_raw) + eps
```

其中：

> **p<sub>0</sub> = P(y = 0) = sigmoid(zero_logit)**

> **p = P(y > 0) = 1 - p<sub>0</sub> = sigmoid(-zero_logit)**

> **σ = softplus(sigma_raw) + ε**

ZILN NLL 公式。这里用 **p = P(y > 0)**，也就是非零概率：

> **L<sub>ZILN</sub>(y) = -𝟙(y = 0) log(1 - p) - 𝟙(y > 0) log(p) + 𝟙(y > 0) · [ log(yσ√(2π)) + ((log y - μ)<sup>2</sup> / 2σ<sup>2</sup>) ]**

预测、评估和落表时，需要把 ZILN 参数还原为期望：

> **ŷ = E[y] = p · exp(μ + 0.5σ<sup>2</sup>)**

代码实现：

```python
def ziln_expected_value(params, eps=1e-6):
    zero_logit, mu, sigma_raw = tf.split(params, 3, axis=-1)
    prob_non_zero = tf.nn.sigmoid(-zero_logit)
    sigma = tf.nn.softplus(sigma_raw) + eps
    return prob_non_zero * tf.exp(mu + 0.5 * tf.square(sigma))
```

注意：训练时主 loss 直接用 ZILN NLL，不用还原值和 label 再算 MSE。还原值主要用于 AUUC、cost AUC、DataFrame 输出和后续 test Hive 落表。

## 3. 当前 ZILN 实验

### 3.1 `euen_4t_ziln_01.py`

目标：只把 cost 从 MSE 替换成 ZILN，label/outcome 仍保持原版 EUEN 的 MSE。

主要结构：

- control head：输出 1 维标量，仍用于 outcome MSE。
- uplift head：输出 1 维标量，仍用于原 EUEN uplift residual。
- cost head：输出 3 维 ZILN 参数。

loss 代码：

```python
outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

其中：

```python
def weighted_mean(values, weights, eps=1e-8):
    values = tf.reshape(values, [-1, 1])
    weights = tf.reshape(tf.cast(weights, tf.float32), [-1, 1])
    return tf.reduce_sum(values * weights) / (tf.reduce_sum(weights) + eps)
```

公式：

> **L<sub>01</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_mse</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: tᵢ > 0</sub> L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome loss 全 batch 平均。
- cost loss 只在当前 batch 的 treatment 样本内平均。
- control 不进 cost loss 分子，也不进 cost loss 分母。
- cost loss 不会被 control 样本稀释。

输出还原：

```python
outs = [
    delta_cvr_1, delta_cvr_2, delta_cvr_3, delta_cvr_4,
    ziln_expected_value(cost_1), ziln_expected_value(cost_2),
    ziln_expected_value(cost_3), ziln_expected_value(cost_4),
    label_cost_7d, label, is_treat
]
```

所以 DataFrame 中的 `predict_cost_7d_1/2/3/4` 已经是还原后的期望 cost。

### 3.2 `euen_4t_ziln_01_batch.py`

目标：只把 cost 从 MSE 替换成 ZILN，同时尽量保持原版 EUEN 的 cost loss 分母口径。

主要结构和 `euen_4t_ziln_01.py` 一致：

- control head：输出 1 维标量。
- uplift head：输出 1 维标量。
- cost head：输出 3 维 ZILN 参数。

loss 代码：

```python
outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = tf.reduce_mean(ziln_nll(label_cost_7d, cost_params) * treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>01_batch</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_mse</sup> + (1 / N<sub>batch</sub>) Σ<sub>i</sub> m<sub>i</sub>L<sub>i</sub><sup>cost_ziln</sup>**

其中：

> **m<sub>i</sub> = 𝟙(t<sub>i</sub> > 0)**

含义：

- outcome loss 全 batch 平均。
- cost loss 也全 batch 平均。
- control 样本的 cost loss 被 mask 成 0，但仍进入 batch 平均分母。
- 这个版本更接近原版 EUEN cost MSE 的 loss 权重口径。

输出还原和 `euen_4t_ziln_01.py` 一致，`predict_cost_7d_1/2/3/4` 已经是还原后的期望 cost。

### 3.3 `euen_4t_ziln_02.py`

目标：label/outcome 和 cost 都使用 ZILN NLL。

主要结构：

- control head：输出 3 维 ZILN 参数。
- uplift head：输出 3 维 ZILN 参数残差。
- cost head：输出 3 维 ZILN 参数。

这个版本仍保留 EUEN 的 residual 结构，但 residual 发生在 ZILN 参数空间。训练时先按真实 treatment 组合出 factual outcome 的 ZILN 参数，再对真实 label 算 ZILN NLL。

loss 代码：

```python
outcome_params = (
    is_list[0] * uc
    + is_list[1] * ut_1
    + is_list[2] * ut[0]
    + is_list[3] * ut[1]
    + is_list[4] * ut[2]
)
cost_params = select_treatment_output(is_list, [cost_1, cost_2, cost_3, cost_4])

outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>02</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_ziln</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: tᵢ > 0</sub> L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome loss 使用 ZILN，对全 batch 平均。
- cost loss 使用 ZILN，只在当前 batch 的 treatment 样本内平均。
- control 样本参与 outcome ZILN loss，不参与 cost ZILN loss。

uplift 还原：

```python
outcome_control = ziln_expected_value(uc)
outcome_1 = ziln_expected_value(c_logit_fix + u_tau_1)
outcome_2 = ziln_expected_value(c_logit_fix + t_logit_1_fix_pred + u_tau_2)
outcome_3 = ziln_expected_value(c_logit_fix + t_logit_1_fix_pred + u_tau_3)
outcome_4 = ziln_expected_value(c_logit_fix + t_logit_1_fix_pred + u_tau_4)

delta_cvr_1 = outcome_1 - outcome_control
delta_cvr_2 = outcome_2 - outcome_control
delta_cvr_3 = outcome_3 - outcome_control
delta_cvr_4 = outcome_4 - outcome_control
```

cost 还原：

```python
ziln_expected_value(cost_1)
ziln_expected_value(cost_2)
ziln_expected_value(cost_3)
ziln_expected_value(cost_4)
```

所以 DataFrame 中的 `predict_delta_cvr_1/2/3/4` 和 `predict_cost_7d_1/2/3/4` 都已经是还原后的标量。

### 3.3.1 `euen_4t_ziln_02_batch.py`

目标：和 `euen_4t_ziln_02.py` 保持同样的模型结构、outcome ZILN 参数 residual、还原方式，只把 cost loss 从 treatment 内平均改成全 batch 平均。

loss 差异代码：

```python
outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = tf.reduce_mean(ziln_nll(label_cost_7d, cost_params) * treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>02_batch</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_ziln</sup> + (1 / N<sub>batch</sub>) Σ<sub>i</sub> m<sub>i</sub>L<sub>i</sub><sup>cost_ziln</sup>**

其中：

> **m<sub>i</sub> = 1(t<sub>i</sub> > 0)**

含义：

- outcome 部分和 `02` 完全一致。
- cost 部分仍只让 treatment 样本产生非零 loss。
- control 样本 cost loss 被 mask 成 0，但进入全 batch 分母。
- 这个版本用于观察 `02` 中 cost loss 权重从 treatment 内平均改成全 batch 平均后，指标是否变化。

### 3.4 `euen_4t_ziln_03.py`

目标：保留原 EUEN 的 residual uplift 结构，但避免 `02` 里的 ZILN 参数空间相加。做法是 control/uplift head 仍输出 ZILN 参数，训练时先还原成期望标量再组合 factual outcome；评估和落表时直接使用还原后的 uplift residual 相加，和原版 EUEN 的 `predict_delta_cvr_k` 口径一致。

主要结构：

- control head：输出 3 维 ZILN 参数。
- uplift head：输出 3 维 ZILN 参数，但语义是“正向 residual 分量”的参数。
- cost head：输出 3 维 ZILN 参数。

这个版本不再做 `c_logit_fix + u_tau_k` 这种参数空间相加。它会先把每个 head 的参数还原为期望，再组合出 factual outcome 标量。

loss 代码：

```python
control_pred = ziln_expected_value(uc)
control_fix_pred = ziln_expected_value(c_logit_fix)
uplift_1 = ziln_expected_value(u_tau_1)
uplift_1_fix = tf.stop_gradient(uplift_1)
uplift_2_res = ziln_expected_value(u_tau_2)
uplift_3_res = ziln_expected_value(u_tau_3)
uplift_4_res = ziln_expected_value(u_tau_4)

outcome_t1 = control_fix_pred + uplift_1
outcome_t2 = control_fix_pred + uplift_1_fix + uplift_2_res
outcome_t3 = control_fix_pred + uplift_1_fix + uplift_3_res
outcome_t4 = control_fix_pred + uplift_1_fix + uplift_4_res
outcome_pred = (
    is_list[0] * control_pred
    + is_list[1] * outcome_t1
    + is_list[2] * outcome_t2
    + is_list[3] * outcome_t3
    + is_list[4] * outcome_t4
)
cost_params = select_treatment_output(is_list, [cost_1, cost_2, cost_3, cost_4])

outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>03</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> (ŷ<sub>i</sub><sup>factual</sup> - y<sub>i</sub>)<sup>2</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome head 输出的是 ZILN 参数，但 outcome loss 不再是严格 ZILN NLL，而是还原后的 factual outcome 标量对真实 label 做 MSE。
- control 样本用 `E[uc]` 拟合 label。
- treatment 1 样本用 `E[c_logit_fix] + E[u_tau_1]` 拟合 label。
- treatment 2/3/4 样本用 `E[c_logit_fix] + stop_gradient(E[u_tau_1]) + E[u_tau_k]` 拟合 label。
- cost loss 使用 ZILN，只在当前 batch 的 treatment 样本内平均。
- 相比 `02`，03 不再把 ZILN 参数相加，因此避免了“参数空间 residual 是否合理”的额外假设。
- 由于 `ziln_expected_value(...)` 是非负的，这个版本会让 uplift residual 分量非负；因此 treatment 1 相对 control 偏正，treatment 2/3/4 相对 treatment 1 的残差也偏正。这一点比“每个 treatment 相对 control 为正”更强，需要结合业务假设看是否合适。

uplift 还原和 `predict_delta_cvr_k` 输出：

```python
uplift_1 = ziln_expected_value(u_tau_1)
uplift_2_res = ziln_expected_value(u_tau_2)
uplift_3_res = ziln_expected_value(u_tau_3)
uplift_4_res = ziln_expected_value(u_tau_4)

delta_cvr_1 = uplift_1
delta_cvr_2 = uplift_1 + uplift_2_res
delta_cvr_3 = uplift_1 + uplift_3_res
delta_cvr_4 = uplift_1 + uplift_4_res
```

训练 loss 里的 factual outcome 仍然需要先构造出来：

```python
outcome_t1 = control_fix_pred + uplift_1
outcome_t2 = control_fix_pred + uplift_1_fix + uplift_2_res
outcome_t3 = control_fix_pred + uplift_1_fix + uplift_3_res
outcome_t4 = control_fix_pred + uplift_1_fix + uplift_4_res
```

注意：如果严格说“先把 residual 参数还原成期望再相加”，组合后的 `outcome_tk` 已经是一个标量期望，不再是一组三维 ZILN 参数，所以不能再直接对它写标准 ZILN NLL。03 因此是“ZILN 参数化的非负 residual + outcome MSE”，不是严格的 outcome ZILN NLL。

### 3.4.1 `euen_4t_ziln_03_batch.py`

目标：和 `euen_4t_ziln_03.py` 保持同样的模型结构、期望空间 residual 相加、还原方式，只把 cost loss 从 treatment 内平均改成全 batch 平均。

loss 差异代码：

```python
outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = tf.reduce_mean(ziln_nll(label_cost_7d, cost_params) * treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>03_batch</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> (ŷ<sub>i</sub><sup>factual</sup> - y<sub>i</sub>)<sup>2</sup> + (1 / N<sub>batch</sub>) Σ<sub>i</sub> m<sub>i</sub>L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome 部分和 `03` 完全一致。
- `predict_delta_cvr_k` 仍由还原后的 uplift residual 直接相加得到。
- cost 部分仍只让 treatment 样本产生非零 loss。
- control 样本 cost loss 被 mask 成 0，但进入全 batch 分母。

### 3.5 `euen_4t_ziln_04.py`

目标：保留之前直接 outcome 的版本。label/outcome 和 cost 都使用 ZILN NLL，但不保留原 EUEN 的 uplift residual 结构。

主要结构：

- control outcome head：输出 3 维 ZILN 参数。
- treatment 1/2/3/4 outcome head：各自直接输出 3 维 ZILN 参数。
- cost head：输出 3 维 ZILN 参数。

这个版本不再计算 `c_logit_fix + u_tau_k`。训练时直接按真实 treatment 选择 factual outcome 参数，再对真实 label 算 ZILN NLL。

loss 代码：

```python
outcome_params = (
    is_list[0] * uc
    + select_treatment_output(is_list, [out_t1, out_t2, out_t3, out_t4])
)
cost_params = select_treatment_output(is_list, [cost_1, cost_2, cost_3, cost_4])

outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>04</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_ziln_direct</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome loss 使用 ZILN，对全 batch 平均。
- control 样本用 control outcome head 的 ZILN 参数算 label NLL。
- treatment 样本用对应 treatment outcome head 的 ZILN 参数算 label NLL。
- cost loss 使用 ZILN，只在当前 batch 的 treatment 样本内平均。
- 相比 `02`，04 不做参数空间 residual。
- 相比 `03`，04 不做期望空间 residual，而是让每个 treatment 直接学习自己的 outcome 分布。

uplift 还原：

```python
outcome_control = ziln_expected_value(uc)
outcome_1 = ziln_expected_value(out_t1)
outcome_2 = ziln_expected_value(out_t2)
outcome_3 = ziln_expected_value(out_t3)
outcome_4 = ziln_expected_value(out_t4)

delta_cvr_1 = outcome_1 - outcome_control
delta_cvr_2 = outcome_2 - outcome_control
delta_cvr_3 = outcome_3 - outcome_control
delta_cvr_4 = outcome_4 - outcome_control
```

### 3.5.1 `euen_4t_ziln_04_batch.py`

目标：和 `euen_4t_ziln_04.py` 保持同样的直接 outcome ZILN 结构、loss 和还原方式，只把 cost loss 从 treatment 内平均改成全 batch 平均。

loss 差异代码：

```python
outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = tf.reduce_mean(ziln_nll(label_cost_7d, cost_params) * treat_mask)
loss1 = outcome_loss + cost_loss
```

公式：

> **L<sub>04_batch</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>i</sub><sup>outcome_ziln_direct</sup> + (1 / N<sub>batch</sub>) Σ<sub>i</sub> m<sub>i</sub>L<sub>i</sub><sup>cost_ziln</sup>**

含义：

- outcome 部分和 `04` 完全一致。
- `predict_delta_cvr_k` 仍由 `E[y_k] - E[y_0]` 得到。
- cost 部分仍只让 treatment 样本产生非零 loss。
- control 样本 cost loss 被 mask 成 0，但进入全 batch 分母。

## 4. 举例说明 Loss 分母差异

假设一个 batch 有 10 条样本：

```text
control 样本数: 4
treatment 样本数: 6
```

6 个 treatment 样本的 cost 单样本 loss 分别是：

```text
1, 2, 3, 4, 5, 6
```

control 样本没有 cost head。按 mask 后，control 的 cost loss 数值为 0。

### 原版 EUEN

cost loss 是全 batch 平均：

> **L<sub>cost_origin</sub> = (0 + 0 + 0 + 0 + 1 + 2 + 3 + 4 + 5 + 6) / 10 = 2.1**

### `euen_4t_ziln_01.py`

cost loss 是 treatment 样本内平均：

> **L<sub>cost_01</sub> = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5**

这等价于先算每个 treatment 的组内均值，再按 treatment 样本量加权平均：

> **L<sub>cost_01</sub> = (n<sub>1</sub>L<sub>1</sub> + n<sub>2</sub>L<sub>2</sub> + n<sub>3</sub>L<sub>3</sub> + n<sub>4</sub>L<sub>4</sub>) / (n<sub>1</sub> + n<sub>2</sub> + n<sub>3</sub> + n<sub>4</sub>)**

### `euen_4t_ziln_01_batch.py`

cost loss 是全 batch 平均：

> **L<sub>cost_01_batch</sub> = (0 + 0 + 0 + 0 + 1 + 2 + 3 + 4 + 5 + 6) / 10 = 2.1**

这个分母口径更接近原版 EUEN。

### `euen_4t_ziln_02.py`

cost loss 和 `01` 一样，是 treatment 样本内平均：

> **L<sub>cost_02</sub> = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5**

区别在于 `02` 的 outcome loss 也从 MSE 换成了 ZILN。

### `euen_4t_ziln_02_batch.py`

cost loss 是全 batch 平均：

> **L<sub>cost_02_batch</sub> = (0 + 0 + 0 + 0 + 1 + 2 + 3 + 4 + 5 + 6) / 10 = 2.1**

区别在于 `02_batch` 的 outcome 结构和 `02` 一样，但 cost loss 权重更接近原版 EUEN。

### `euen_4t_ziln_03.py`

cost loss 和 `01`、`02` 一样，是 treatment 样本内平均：

> **L<sub>cost_03</sub> = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5**

区别在于 `03` 的 outcome 侧仍保留 EUEN residual 结构，但 residual 发生在期望标量空间：训练时先把 control/uplift 参数还原为期望，再做 `control + uplift` 并用 MSE 拟合 label；评估和落表时直接按还原后的 uplift residual 相加输出 `predict_delta_cvr_k`。

### `euen_4t_ziln_03_batch.py`

cost loss 是全 batch 平均：

> **L<sub>cost_03_batch</sub> = (0 + 0 + 0 + 0 + 1 + 2 + 3 + 4 + 5 + 6) / 10 = 2.1**

区别在于 `03_batch` 的 outcome 结构和 `03` 一样，但 cost loss 权重更接近原版 EUEN。

### `euen_4t_ziln_04.py`

cost loss 和 `01`、`02`、`03` 一样，是 treatment 样本内平均：

> **L<sub>cost_04</sub> = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5**

区别在于 `04` 的 outcome loss 使用直接 treatment outcome ZILN 参数，不再保留 EUEN residual 结构。

### `euen_4t_ziln_04_batch.py`

cost loss 是全 batch 平均：

> **L<sub>cost_04_batch</sub> = (0 + 0 + 0 + 0 + 1 + 2 + 3 + 4 + 5 + 6) / 10 = 2.1**

区别在于 `04_batch` 的 outcome 结构和 `04` 一样，但 cost loss 权重更接近原版 EUEN。

## 5. 总结表

| 版本 | outcome loss | cost loss | cost 分母 | control 是否稀释 cost | 输出还原 |
|---|---|---|---|---|---|
| 原版 EUEN | MSE | MSE | 全 batch | 是 | uplift 和 cost 都是原标量输出 |
| `euen_4t_ziln_01.py` | MSE | ZILN NLL | batch 内 treatment 样本数 | 否 | cost 还原为期望 |
| `euen_4t_ziln_01_batch.py` | MSE | ZILN NLL | 全 batch | 是 | cost 还原为期望 |
| `euen_4t_ziln_02.py` | ZILN NLL，outcome 参数 residual 相加 | ZILN NLL | batch 内 treatment 样本数 | 否 | 参数相加后还原 outcome，再求 uplift；cost 还原为期望 |
| `euen_4t_ziln_02_batch.py` | ZILN NLL，outcome 参数 residual 相加 | ZILN NLL | 全 batch | 是 | 参数相加后还原 outcome，再求 uplift；cost 还原为期望 |
| `euen_4t_ziln_03.py` | MSE，先还原 control/uplift 期望再做 residual 相加 | ZILN NLL | batch 内 treatment 样本数 | 否 | `predict_delta_cvr_k` 由还原后的 uplift residual 直接相加得到；cost 还原为期望 |
| `euen_4t_ziln_03_batch.py` | MSE，先还原 control/uplift 期望再做 residual 相加 | ZILN NLL | 全 batch | 是 | `predict_delta_cvr_k` 由还原后的 uplift residual 直接相加得到；cost 还原为期望 |
| `euen_4t_ziln_04.py` | ZILN NLL，直接 treatment outcome 参数 | ZILN NLL | batch 内 treatment 样本数 | 否 | control/treatment 分别还原 outcome，再求 uplift；cost 还原为期望 |
| `euen_4t_ziln_04_batch.py` | ZILN NLL，直接 treatment outcome 参数 | ZILN NLL | 全 batch | 是 | control/treatment 分别还原 outcome，再求 uplift；cost 还原为期望 |

一句话总结：

```text
euen_4t_ziln_01.py 同时改变了 cost loss 形式和 cost loss 权重；
euen_4t_ziln_01_batch.py 更适合做“只替换 cost MSE 为 ZILN”的干净对比；
euen_4t_ziln_02.py 是更激进的版本，把 outcome/label 和 cost 都改成 ZILN 分布建模，并在 ZILN 参数空间保留 residual 相加；
euen_4t_ziln_03.py 保留原 EUEN residual 结构，但先把 ZILN 参数还原为期望标量，再在期望空间做 residual 相加，所以 outcome loss 回到 MSE；
euen_4t_ziln_04.py 保留之前直接 outcome 的版本，每个 treatment 直接学习自己的 outcome ZILN 分布，再在期望空间做 uplift 差值；
euen_4t_ziln_02_batch.py、euen_4t_ziln_03_batch.py、euen_4t_ziln_04_batch.py 分别对应 02、03、04 的全 batch cost 分母版本。
```

后续做 test Hive 落表时需要注意：ZILN 模型保存后的原始输出是参数，不是标量；Hive writer 需要复用 `ziln_expected_value` 逻辑，把参数还原后再写原有字段。
