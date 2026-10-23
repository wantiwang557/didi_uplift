# EUEN ZILN Loss 实验沉淀

本文沉淀 EUEN 上 ZILN loss 的设计、四个主实验版本，以及当前已有日志下的实验效果对比。

实验效果部分使用的主对比口径：

```text
train: v2 train_0511_0617
test:  v2 test_0723_0806
```

对比原版 EUEN 日志：

```text
demo/euen_test_feature_time/run_logs/euen_v2_train_0511_0617_test_0723_0806_02.log
```

ZILN 实验日志：

```text
demo/Loss_test/run_logs/euen_4t_ziln_01_v2_train_0511_0617_test_0723_0806.log
demo/Loss_test/run_logs/euen_4t_ziln_02_v2_train_0511_0617_test_0723_0806.log
demo/Loss_test/run_logs/euen_4t_ziln_03_v2_train_0511_0617_test_0723_0806.log
```

当前本地暂未看到 `euen_4t_ziln_04_v2_train_0511_0617_test_0723_0806.log`，所以 04 的实验效果先标记为待补充。

## 1. ZILN 设计思想、Loss 和还原公式

### 1.1 设计思想

ZILN 是 Zero-Inflated Lognormal，适合建模非负、零值很多、正值部分长尾的目标。

在当前业务里，`cost_7d` 很适合这种假设：

- 很多样本 cost 为 0。
- 非 0 cost 是连续非负值。
- 正值 cost 往往右偏、长尾。

ZILN 把目标拆成两部分：

- 是否为 0：用二分类建模。
- 如果大于 0，正值大小服从 LogNormal 分布。

因此一个 ZILN head 不再直接输出一个预测标量，而是输出 3 个参数：

```text
params = (zero_logit, mu, sigma_raw)
```

其中：

> **p<sub>0</sub> = P(y = 0) = sigmoid(zero_logit)**

> **p = P(y > 0) = 1 - p<sub>0</sub> = sigmoid(-zero_logit)**

> **sigma = softplus(sigma_raw) + eps**

这里 `zero_logit` 越大，模型越倾向判断 y 为 0；`mu` 和 `sigma` 描述 y 大于 0 时 `log(y)` 的正态分布。

### 1.2 ZILN Loss

单样本 ZILN 负对数似然可以写成：

> **L<sub>ZILN</sub>(y) = 1(y = 0) × [-log(1 - p)] + 1(y > 0) × [-log(p) + log(y × sigma × sqrt(2π)) + (log(y) - mu)<sup>2</sup> / (2 × sigma<sup>2</sup>)]**

其中：

> **p = P(y > 0) = sigmoid(-zero_logit)**

拆开看：

- `1(y = 0) × [-log(1 - p)]`：零值样本的二分类损失。
- `1(y > 0) × [-log(p)]`：正值样本的二分类损失。
- `1(y > 0) × LogNormal_NLL(y)`：正值样本的 lognormal 回归损失。

对应代码：

```python
def ziln_nll(y_true, params, eps=1e-6):
    y_true = tf.reshape(tf.maximum(y_true, 0.0), [-1, 1])
    zero_logit, mu, sigma_raw = tf.split(params, 3, axis=-1)
    sigma = tf.nn.softplus(sigma_raw) + eps
    is_zero = tf.cast(y_true <= eps, tf.float32)
    zero_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=is_zero, logits=zero_logit)
    safe_y = tf.maximum(y_true, eps)
    log_y = tf.math.log(safe_y)
    lognormal_loss = (
        tf.math.log(safe_y)
        + tf.math.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
        + tf.square(log_y - mu) / (2.0 * tf.square(sigma))
    )
    return zero_loss + (1.0 - is_zero) * lognormal_loss
```

### 1.3 ZILN 还原公式

训练时可以直接使用 ZILN NLL；但评估、AUUC、AUC、落表时，需要把 ZILN 参数还原成一个可排序、可落表的标量期望。

还原公式：

> **E[y] = P(y > 0) × exp(mu + 0.5 × sigma<sup>2</sup>)**

也就是：

> **E[y] = sigmoid(-zero_logit) × exp(mu + 0.5 × sigma<sup>2</sup>)**

对应代码：

```python
def ziln_expected_value(params, eps=1e-6):
    zero_logit, mu, sigma_raw = tf.split(params, 3, axis=-1)
    prob_non_zero = tf.nn.sigmoid(-zero_logit)
    sigma = tf.nn.softplus(sigma_raw) + eps
    return prob_non_zero * tf.exp(mu + 0.5 * tf.square(sigma))
```

## 2. 四个实验详细介绍

### 2.1 `euen_4t_ziln_01.py`

代码：

```text
demo/Loss_test/euen_4t_ziln_01.py
```

目标：只把 cost loss 从 MSE 替换成 ZILN NLL，outcome/label 仍保持原 EUEN 的 MSE。

结构：

- 模型主体沿用原版 EUEN。
- control head 输出 1 维标量 outcome。
- uplift head 输出 1 维标量 uplift residual。
- cost head 输出 3 维 ZILN 参数：`zero_logit, mu, sigma_raw`。

loss：

> **L<sub>01</sub> = outcome_mse_batch_mean + cost_ziln_treatment_mean**

展开为：

> **L<sub>01</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> (ŷ<sub>i</sub><sup>outcome</sup> - y<sub>i</sub>)<sup>2</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>ZILN</sub>(cost<sub>i</sub>)**

代码口径：

```python
outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

还原方式：

```text
predict_delta_cvr_1 = u_tau_1
predict_delta_cvr_2 = u_tau_1 + u_tau_2
predict_delta_cvr_3 = u_tau_1 + u_tau_3
predict_delta_cvr_4 = u_tau_1 + u_tau_4

predict_cost_7d_k = E[cost_k]
```

小结：01 是最保守的 ZILN 版本，只改 cost 建模，不改变 uplift 主结构和 outcome loss。

### 2.2 `euen_4t_ziln_02.py`

代码：

```text
demo/Loss_test/euen_4t_ziln_02.py
```

目标：label/outcome 和 cost 都使用 ZILN NLL，并保留原 EUEN residual 结构。但这里的 residual 发生在 ZILN 参数空间。

结构：

- control head 输出 3 维 ZILN 参数。
- uplift head 输出 3 维 ZILN 参数 residual。
- cost head 输出 3 维 ZILN 参数。
- treatment outcome 参数先按原 EUEN residual 结构相加，再作为 factual outcome 参数参与 ZILN NLL。

loss：

> **L<sub>02</sub> = outcome_ziln_batch_mean + cost_ziln_treatment_mean**

展开为：

> **L<sub>02</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>ZILN</sub>(label<sub>i</sub>) + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>ZILN</sub>(cost<sub>i</sub>)**

代码口径：

```python
ut_1 = c_logit_fix + u_tau_1
ut = [c_logit_fix + t_logit_1_fix + tau for tau in [u_tau_2, u_tau_3, u_tau_4]]
outcome_params = (
    is_list[0] * uc
    + is_list[1] * ut_1
    + is_list[2] * ut[0]
    + is_list[3] * ut[1]
    + is_list[4] * ut[2]
)

outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

还原方式：

02 是先在参数空间相加，再还原 outcome，最后 outcome 减 control：

```text
predict_delta_cvr_1 = E[c_logit_fix + u_tau_1] - E[uc]
predict_delta_cvr_2 = E[c_logit_fix + stop_gradient(u_tau_1) + u_tau_2] - E[uc]
predict_delta_cvr_3 = E[c_logit_fix + stop_gradient(u_tau_1) + u_tau_3] - E[uc]
predict_delta_cvr_4 = E[c_logit_fix + stop_gradient(u_tau_1) + u_tau_4] - E[uc]

predict_cost_7d_k = E[cost_k]
```

小结：02 是更激进的版本，outcome 和 cost 都做 ZILN 分布建模，但引入了一个额外假设：ZILN 参数空间的 residual 相加是合理的。

### 2.3 `euen_4t_ziln_03.py`

代码：

```text
demo/Loss_test/euen_4t_ziln_03.py
```

目标：保留原 EUEN residual uplift 结构，但避免 02 的参数空间 residual 相加。做法是先把 control/uplift 参数还原为期望，再在期望空间做 residual 相加。

结构：

- control head 输出 3 维 ZILN 参数。
- uplift head 输出 3 维 ZILN 参数，但语义是 uplift residual 的分布参数。
- cost head 输出 3 维 ZILN 参数。
- 训练时先还原 control/uplift 参数，组合 factual outcome 标量。
- 评估和落表时，`predict_delta_cvr_k` 直接由还原后的 uplift residual 相加得到。

loss：

> **L<sub>03</sub> = outcome_mse_batch_mean + cost_ziln_treatment_mean**

展开为：

> **L<sub>03</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> (ŷ<sub>i</sub><sup>factual</sup> - y<sub>i</sub>)<sup>2</sup> + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>ZILN</sub>(cost<sub>i</sub>)**

代码口径：

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

outcome_loss = tf.reduce_mean(tf.square(outcome_pred - label))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

还原方式：

03 是先分别还原 uplift residual，再直接相加：

```text
predict_delta_cvr_1 = E[u_tau_1]
predict_delta_cvr_2 = E[u_tau_1] + E[u_tau_2]
predict_delta_cvr_3 = E[u_tau_1] + E[u_tau_3]
predict_delta_cvr_4 = E[u_tau_1] + E[u_tau_4]

predict_cost_7d_k = E[cost_k]
```

注意：03 的 outcome loss 不是 ZILN NLL。因为 residual 参数还原并相加后得到的是标量期望，不再是一组三维 ZILN 参数，所以只能对组合后的 factual outcome 用 MSE。

小结：03 的输出口径最接近原版 EUEN，只是把 uplift residual 和 cost 从直接标量改成 ZILN 参数，再还原成期望。

### 2.4 `euen_4t_ziln_04.py`

代码：

```text
demo/Loss_test/euen_4t_ziln_04.py
```

目标：直接建模每个 treatment 下的 outcome 分布，不保留 uplift residual 结构。

结构：

- control outcome head 输出 3 维 ZILN 参数。
- treatment 1/2/3/4 outcome head 各自输出 3 维 ZILN 参数。
- cost head 输出 3 维 ZILN 参数。
- 不再输出 residual uplift。

loss：

> **L<sub>04</sub> = outcome_ziln_direct_batch_mean + cost_ziln_treatment_mean**

展开为：

> **L<sub>04</sub> = (1 / N<sub>batch</sub>) Σ<sub>i</sub> L<sub>ZILN</sub>(label<sub>i</sub>) + (1 / N<sub>treat</sub>) Σ<sub>i: t<sub>i</sub> > 0</sub> L<sub>ZILN</sub>(cost<sub>i</sub>)**

代码口径：

```python
outcome_params = (
    is_list[0] * uc
    + select_treatment_output(is_list, [out_t1, out_t2, out_t3, out_t4])
)

outcome_loss = tf.reduce_mean(ziln_nll(label, outcome_params))
cost_loss = weighted_mean(ziln_nll(label_cost_7d, cost_params), treat_mask)
loss1 = outcome_loss + cost_loss
```

还原方式：

04 是 control 和各 treatment outcome 先分别还原，再做差：

```text
predict_delta_cvr_1 = E[y_1] - E[y_0]
predict_delta_cvr_2 = E[y_2] - E[y_0]
predict_delta_cvr_3 = E[y_3] - E[y_0]
predict_delta_cvr_4 = E[y_4] - E[y_0]

predict_cost_7d_k = E[cost_k]
```

小结：04 最像多 outcome head 的直接分布建模版本，结构更直观，但不再利用原 EUEN 的 residual uplift 约束。

### 2.5 四个版本差异总结

| 版本 | outcome 结构 | outcome loss | cost loss | predict_delta_cvr_k |
|---|---|---|---|---|
| 01 | 原 EUEN 标量 residual | MSE | ZILN NLL，treatment 内平均 | 原标量 tau 相加 |
| 02 | ZILN 参数空间 residual | ZILN NLL | ZILN NLL，treatment 内平均 | 参数相加后还原 outcome，再减 control |
| 03 | ZILN 期望空间 residual | MSE | ZILN NLL，treatment 内平均 | 还原后的 uplift residual 直接相加 |
| 04 | 直接 treatment outcome 参数 | ZILN NLL | ZILN NLL，treatment 内平均 | treatment outcome 期望减 control 期望 |

补充：`02_batch`、`03_batch`、`04_batch` 是对应版本的 cost loss 全 batch 分母版本。它们不改变模型结构和还原方式，只把 cost loss 从 treatment 内平均改为全 batch 平均。

## 3. 实验效果

### 3.1 AUUC 对比

| 版本 | best epoch | AUUC T1 | AUUC T2 | AUUC T3 | AUUC T4 | mean AUUC | 相对原版 mean AUUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原版 EUEN | 2 | 0.974215 | 0.934367 | 0.874519 | 0.910687 | 0.923447 | 0.000000 |
| 01 | 4 | 1.047244 | 1.007583 | 0.960197 | 0.928703 | 0.985932 | +0.062485 |
| 02 | 3 | 1.053168 | 1.021017 | 0.966212 | 0.942099 | 0.995624 | +0.072177 |
| 03 | 4 | 1.018648 | 0.977417 | 0.907030 | 0.917636 | 0.955183 | +0.031736 |
| 04 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

从 AUUC 看：

- 01、02、03 都高于原版 EUEN 的 mean AUUC。
- 当前已有日志里，02 的 mean AUUC 最高，为 `0.995624`。
- 01 也有明显提升，mean AUUC 为 `0.985932`。
- 03 提升幅度较小，mean AUUC 为 `0.955183`，但它的结构解释性更接近原版 EUEN。
- 04 当前本地没有对应日志，暂时不能判断效果。

### 3.2 Cost AUC 对比

| 版本 | AUC cost T1 | AUC cost T2 | AUC cost T3 | AUC cost T4 | mean cost AUC | 相对原版 mean AUC |
|---|---:|---:|---:|---:|---:|---:|
| 原版 EUEN | 0.885891 | 0.883792 | 0.836879 | 0.849819 | 0.864095 | 0.000000 |
| 01 | 0.906649 | 0.898847 | 0.867285 | 0.863253 | 0.884008 | +0.019913 |
| 02 | 0.906530 | 0.902119 | 0.871427 | 0.869720 | 0.887449 | +0.023354 |
| 03 | 0.906039 | 0.897794 | 0.867264 | 0.863083 | 0.883545 | +0.019449 |
| 04 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

从 cost AUC 看：

- 01、02、03 都比原版 EUEN 高。
- 当前已有日志里，02 的 mean cost AUC 最高，为 `0.887449`。
- 01 和 03 的 mean cost AUC 接近，分别为 `0.884008` 和 `0.883545`。
- 这说明把 cost 从 MSE 换成 ZILN 后，对 cost 排序能力有稳定收益。

### 3.3 当前结论

在当前 v2 日志下：

- 如果只看效果，02 当前最好：mean AUUC 和 mean cost AUC 都是已有版本里最高。
- 如果看最小改动，01 最稳：只改 cost loss，不改变 outcome/uplift 主结构。
- 如果看结构解释性，03 更接近原版 EUEN：最终 `predict_delta_cvr_k` 仍然是 residual 相加，只是 residual 先由 ZILN 参数还原为期望。
- 04 是直接 outcome 分布建模版本，目前需要跑完日志后再和 01/02/03 做完整比较。

后续建议：

- 继续补跑 `euen_4t_ziln_04.py` 的 v2 实验。
- 同步跑 `02_batch`、`03_batch`、`04_batch`，观察 cost loss 分母从 treatment 内平均改成全 batch 平均后，对 AUUC 和 cost AUC 的影响。
- 如果 02 持续最好，需要重点关注它的参数空间 residual 假设是否会带来校准风险；如果 03 接近 02，则 03 可能更适合作为结构解释性更强的版本。
