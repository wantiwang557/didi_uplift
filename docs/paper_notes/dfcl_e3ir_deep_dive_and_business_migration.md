# DFCL 与 E3IR 原论文精读及 WWT 补贴业务迁移分析

> 调研日期：2026-10-12  
> 业务上下文：RCT、多 treatment、EUEN/EEUEN、用户分数分桶、桶内 RCT 后验、增量 CAC 约束 OR  
> 原论文：DFCL（KDD 2024，美团）；E3IR / E³IR（RecSys 2024，人大/腾讯）

## 0. 先给结论

### 0.1 两篇论文分别解决了什么

- **DFCL** 解决的是“预测目标与最终分配价值不一致”。它保留 outcome/cost 预测和 OR，但用 RCT 的事实结果构造无偏或等价的策略价值损失，让最终决策质量反向训练预测模型。
- **E3IR** 同时改造“uplift 模型结构”和“预测到 ILP 的梯度通路”。它用相邻 treatment 的正增量累加保证单调，用共享参数和 Lipschitz 正则控制平滑，再为离散 ILP 手工构造 backward 方向。

两者都不是 LLM，也都没有真正取消约束求解：

```text
DFCL：训练时用对偶分解/策略价值，部署时仍按预算求解
E3IR：直接把 ILP 放进训练图，部署时仍运行 ILP
```

因此，如果“end-to-end”严格指模型直接输出最终动作、完全跳出“预估＋OR”，这两篇都不满足；它们属于 **decision-focused predict-and-optimize**。

### 0.2 对当前业务的推荐顺序

```text
第一步：EUEN/EEUEN + DFCL-PL-CAC
第二步：加入 E3IR 的单调/平滑 treatment head
第三步：若前两步证明 decision loss 有增益，再做 DFCL-IFD-CAC
第四步：不建议直接复刻 E3IR 全量可微 ILP；先做经过修正的 allocation value loss
```

核心原因：

1. 当前已有 RCT、多 treatment、真实 outcome/cost，正好满足 DFCL 的数据要求。
2. 当前 CAC 约束可以线性化，能够自然改造成 DFCL 的对偶净价值。
3. DFCL-PL 只改训练 loss，风险和工程量远低于重写 ILP backward。
4. E3IR 的单调/平滑 head 与当前 EUEN residual 结构相近，适合拆出来单独验证。
5. E3IR 原文公式 (19) 的 allocation loss 是实验 treatment 标签与推荐动作之间的交叉熵，并未直接使用事实收益；在 RCT 中 treatment 本身是随机的，因此不宜不加修改地照搬。

---

## 1. 当前业务问题的数学化

当前用户级动作集合记为：

```text
a ∈ {control, 券1, 券2, 券3}
```

对用户 `i` 和动作 `a`：

- `Δy_ia`：相对 control 的增量完单 PV；
- `Δc_ia`：相对 control 的增量成本；
- `z_ia ∈ {0,1}`：是否给用户 `i` 分配动作 `a`。

现有 OR 的目标与约束可写为：

```text
max  Σ_i Σ_a z_ia Δy_ia

s.t. Σ_i Σ_a z_ia Δc_ia
     ≤ κ Σ_i Σ_a z_ia Δy_ia

     Σ_a z_ia = 1
```

其中 `κ` 是目标增量 CAC。将约束移项：

```text
Σ_i Σ_a z_ia (Δc_ia - κΔy_ia) ≤ 0
```

定义 CAC slack：

```text
g_ia(κ) = Δc_ia - κΔy_ia
```

则固定 `κ` 下的拉格朗日目标为：

```text
max_z Σ_i Σ_a z_ia [Δy_ia - λg_ia(κ)]

= max_z Σ_i Σ_a z_ia [(1 + λκ)Δy_ia - λΔc_ia]
```

个体动作分数因此可以写成：

```text
score_ia(κ, λ)
= Δy_hat_ia - λ(Δc_hat_ia - κΔy_hat_ia)
= (1 + λκ)Δy_hat_ia - λΔc_hat_ia
```

这是把 DFCL 从“总预算”迁移到“增量 CAC”的关键。它说明不需要先把 CAC 约束近似成固定预算；只要在训练和部署中使用同一 `g_ia(κ)` 即可。

需要注意：`g_ia` 可能为负，因此它不是经典“所有 item cost 都非负”的背包重量。但对偶后的个体动作选择仍然成立；最终硬约束是否满足，需要用真实 OR 或单独的 `λ` 搜索/校准验证。

---

## 2. DFCL 原论文方法精读

论文全名：*Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization*。

### 2.1 原始问题

论文考虑 `N` 个个体、`M` 个 treatment：

```text
max_z Σ_i Σ_j z_ij r_ij

s.t. Σ_i Σ_j z_ij c_ij ≤ B
     Σ_j z_ij = 1
     z_ij ∈ {0,1}
```

`r_ij`、`c_ij` 分别是个体 `i` 接受 treatment `j` 时的收入与成本。模型在决策时只能提供 `r_hat_ij`、`c_hat_ij`。

### 2.2 总 loss

```text
L_DFCL = α L_PL + L_DL
```

- `L_PL`：事实 treatment 上的 outcome/cost 预测损失；
- `L_DL`：预测产生的分配策略，在真实潜在结果上的负价值；
- `α`：预测精度与决策质量的权衡。

原文在 RCT 各组样本数为 `N_j` 时，对事实样本使用 `N/N_ti` 加权。其本质是按实验 propensity 做逆概率校正。若当前实验并非各 treatment 等概率分流，应使用真实 `p(t|x)`，不能简单套 `N/N_t`。

### 2.3 为什么先从预算 B 转为影子价格 λ

原问题的预算 `B` 会变化，而且训练时反复跑 MCKP 成本高。DFCL 对预算约束做拉格朗日松弛：

```text
H(z, λ) = Σ_i Σ_j (r_ij - λc_ij) z_ij + λB
```

给定 `λ`，常数 `λB` 不影响动作选择，因此每个用户独立选择：

```text
j*(i, λ) = argmax_j (r_hat_ij - λc_hat_ij)
```

论文进一步利用：

- 较大的 `B` 对应较小的最优 `λ*`；
- 不同 `λ` 可以覆盖不同预算压力；
- 大规模营销下，对偶解与原 MCKP 的目标差相对很小。

训练多个 `λ`，等价于让模型同时学习多个预算水平，而非只服务单一预算点。

### 2.4 反事实问题怎样解决

一个用户只观察到实际 treatment `t_i` 下的 `r_i,t_i`、`c_i,t_i`。DFCL 不能直接计算完整：

```text
Σ_j (r_ij - λc_ij) π(j|x_i)
```

因此对 RCT 事实结果做 IPW：

```text
V_hat(π; λ)
= (1/N) Σ_i
  [ I(策略动作 = t_i) / p(t_i|x_i) ]
  × (r_i - λc_i)
```

对软策略则把 indicator 换成 `π(t_i|x_i)`。随机实验和 positivity 是其可识别性的基础。

### 2.5 三种梯度方案

#### A. DFCL-PL：Policy Learning Loss

把硬 argmax 平滑成 softmax：

```text
π_ij(λ)
= softmax_j(r_hat_ij - λc_hat_ij)
```

策略损失为：

```text
L_PLL
= - Σ_λ Σ_i
    [1 / p(t_i|x_i)]
    (r_i - λc_i)
    π_i,t_i(λ)
```

直觉：

- 如果某用户事实 treatment 的真实净价值高，就提高模型选择它的概率；
- 如果真实净价值低，就降低选择它的概率；
- 只有会改变动作概率的预测误差收到强梯度。

优点：实现最简单、不需要训练时调用完整 OR、适合先验证核心假设。缺点：softmax 策略与部署时硬策略存在温度/平滑偏差。

#### B. DFCL-MER：Maximum Entropy Regularized Loss

论文将离散 `z` 放松为 `[0,1]`，并在对偶目标中加入最大熵正则，闭式解仍是带温度 `τ` 的 softmax：

```text
z_ij
= exp[(r_hat_ij - λc_hat_ij)/τ]
  / Σ_k exp[(r_hat_ik - λc_hat_ik)/τ]
```

然后同样使用事实 outcome/cost 和 IPW 构造 surrogate。

与 PL 相比，MER 明确从熵正则优化问题推导出温度 softmax。`τ` 越小越接近硬决策，但梯度更尖锐；`τ` 越大更稳定，但决策偏差更大。

#### C. DFCL-IFD：Improved Finite Difference

IFD 保留硬动作，用 EOM/IPW 评价当前策略，再扰动预测，估计 decision loss 对预测的梯度：

```text
∂L/∂r_hat_ij
≈ [L(r_hat + h e_ij, c_hat) - L(r_hat, c_hat)] / h
```

朴素做法对每个样本、每个 treatment 扰动后重新求解，无法扩展。论文利用对偶分解后的个体独立性，只计算让当前最优 treatment 切换所需的最小扰动，并增量更新 EOM；还对扰动做截断和 softmax 平滑。

优点：最接近硬决策，论文实验表现最好。缺点：实现细节多、数值稳定性敏感、没有公开官方实现可直接复用。

### 2.6 论文证据强度

论文实验包含：

- Criteo-Uplift v2：1,390 万样本，二元 treatment；
- 美团营销数据：280 万样本、107 特征、5 档折扣；
- 线上 A/B：31 万商家、4 周、5 档折扣。

关键结果：

- Criteo AUCC：DFCL-IFD 相对 TSM-SL 提升 3.94%；
- 美团离线不同预算下，DFCL-IFD 平均 improvement 为 2.85%；
- 线上相对 TSM-SL 的订单平均提升 2.17%，相对 DPM 提升 0.85%。

这些结果说明“decision loss 能带来独立于 MSE 的业务增益”，但不能直接外推到当前场景：论文使用总预算，当前使用增量 CAC；论文直接做用户级分配，当前还有分桶后验层。

### 2.7 对 DFCL 的关键评价

1. **它没有取消 OR。** 它解决的是训练目标对齐，不是把 OR 变没。
2. **它最有价值的是 RCT 策略价值＋多 λ 训练。** 这两点可直接迁移。
3. **论文的对偶近似依赖大规模加总。** 用户量足够大时单个 item 导致的积分间隙相对小；小桶或强配额下未必成立。
4. **原文只处理一个总成本约束。** 当前若同时有 CAC、券库存、触达配额、渠道配比，单一 `λ` 不够，需要向量乘子或保留完整 OR。
5. **PL 是最合适的 MVP，IFD 是上限实验。** 不应一开始就投入 IFD 复刻。

---

## 3. E3IR 原论文方法精读

论文全名：*End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling*。

### 3.1 原始问题

E3IR 直接定义相对 control 的 response uplift 与 cost uplift：

```text
τ^r_i,k = E[y^r_i(k) - y^r_i(0) | x_i]
τ^c_i,k = E[y^c_i(k) - y^c_i(0) | x_i]
```

多 treatment 分配写成：

```text
max_z Σ_i Σ_k τ^r_i,k z_i,k

s.t. Σ_i Σ_k τ^c_i,k z_i,k ≤ B
     Σ_k z_i,k = 1
     z_i,k ∈ {0,1}
```

原文问题定义的一处文字把最大化对象误写成了 `τ^c`，但紧邻公式 (3) 明确目标是 `τ^r`、约束是 `τ^c`，应按公式理解。

### 3.2 Uplift Prediction Module

#### 相邻档位增量累加

模型先预测 control outcome `y_hat_0`，再预测相邻 treatment 之间的增量 `v_hat_k`：

```text
y_hat_t
= y_hat_0 + Σ_{k=1}^t v_hat_k
```

response 和 cost 各自有一套 head。通过对原始输出平方或取指数，强制：

```text
v_hat_k ≥ 0
```

从而使 response/cost 随 treatment 强度单调不降。

这与当前 EUEN 结构并不完全相同：

```text
当前 EUEN：
τ_1 = head_1
τ_2 = stop_grad(τ_1) + residual_2
τ_3 = stop_grad(τ_1) + residual_3
τ_4 = stop_grad(τ_1) + residual_4

E3IR：
τ_1 = v_1
τ_2 = v_1 + v_2
τ_3 = v_1 + v_2 + v_3
...
```

E3IR 表达的是严格有序的相邻剂量曲线；当前 EUEN 的 2/3/4 都是相对 treatment 1 的残差，并不保证 treatment 2→3→4 的顺序或单调性。

#### treatment embedding 与共享 head

E3IR 不是每档完全独立一套参数，而是把：

```text
[用户共享表示 Φ；treatment embedding δ_k]
```

输入共享的 response/cost incremental head。这样相邻档位共享统计强度，也便于给 treatment 强度施加平滑约束。

#### Lipschitz 平滑

论文希望 treatment embedding 小幅变化时，预测 outcome 不会突变。它对网络各层的 Lipschitz 上界做正则：

```text
L_Lip = Π_l softplus(c_l)
L_uplift = L_prediction + α L_Lip
```

这里的 `L_Lip` 更接近“控制整个 head 的敏感度上界”，不是直接对相邻档预测差做二阶平滑。因此工程上可以先用更简单、可解释的相邻差分正则作为近似基线。

### 3.3 Differentiable Allocation Module

前向：把预测的 `τ_hat^r`、`τ_hat^c` 和预算 `B` 送入 ILP，得到离散 allocation matrix `z`。

难点：`z` 是整数解，普通 autograd 无法得到：

```text
∂L/∂τ^r, ∂L/∂τ^c
```

论文借鉴 CombOptNet，把下游给到 `z` 的梯度：

```text
dz = ∂L/∂z
```

分解成一组可实现的整数方向：

```text
dz = Σ_i λ_i Δ_i
```

其中 `Δ_i ∈ {-1,0,1}^n`，代表从当前整数解走向邻近整数解的方向。随后：

- 对 cost uplift，构造当前解/候选解到预算超平面的分段仿射 mismatch，反推 `dτ^c`；
- 对 response uplift，构造让候选解相对当前解更优的目标差，反推 `dτ^r`；
- 对多个 basis 方向的梯度按 `λ_i` 线性组合。

这并不是 ILP 的“真实导数”，而是一个让更新后 ILP 解尽量沿负梯度方向变化的 surrogate backward。

### 3.4 Allocation loss 的关键疑点

原文公式 (19)：

```text
L_allocation
= (1/N) Σ_i CE(t_i, z_i)
```

其中：

- `t_i` 是数据中实际被随机分配的实验 treatment；
- `z_i` 是 ILP 根据预测 uplift/cost 给出的推荐动作。

总 loss：

```text
L_E3IR = L_predict + β L_allocation
```

这带来一个需要严肃对待的问题：如果 `t_i` 来自 RCT，它是随机动作，不是最优动作。让 `z_i` 拟合 `t_i` 并不等于最大化事实收益，也不等于最小化 policy regret。

论文称该 loss 受 EOM 启发，但公式中没有：

- 事实 response `y^r_i`；
- 事实 cost `y^c_i`；
- propensity/IPW；
- 净价值或 regret 权重。

因此更稳妥的判断是：

> E3IR 的“可微 ILP backward”有方法价值，但原文 allocation supervision 与 RCT 策略价值之间的连接不如 DFCL 严谨。迁移时应保留其可微求解思想，替换其 allocation loss，而不是原样照搬 CE(t,z)。

如果作者实现中存在论文未写清的 label 构造，公开论文和当前可检索资料不足以确认。复现前必须以代码或作者说明为准；本轮未找到作者公开的官方仓库。

### 3.5 实验证据

E3IR 使用：

- Hillstrom 公共数据：约 6.4 万消费者；
- 生产数据：800 万以上用户、108 特征、3 档视频锐化 treatment＋control；
- 5 个随机种子；
- PyTorch 1.10、JAX 0.4.23、最多 30 iterations；
- 用 QINI 选超参数。

多 treatment 结果：

```text
Hillstrom EOM：
Multi-TPM-SL 21.9767
DRM          26.5744
E3IR         29.9221

Production EOM：
Multi-TPM-SL 27.6576
DRM          36.6544
E3IR         44.5087
```

论文还报告去掉 monotonic、smooth 或 differentiable allocation 都会退化。

但证据有几个限制：

1. 没有线上 A/B；
2. 生产数据场景是视频锐化与带宽成本，不是券补贴；
3. 没有把“结构增益”和“可微 ILP 增益”做更细的统计显著性拆分；
4. 以 QINI 选所有模型的超参，可能偏向 uplift 排序而不是最终 EOM；
5. allocation loss 的因果解释不够完整；
6. 作者也把降低 differentiable allocation 时间复杂度列为未来工作。

---

## 4. 两篇论文对当前业务的逐项对比

| 维度 | DFCL | E3IR | 当前 WWT 业务判断 |
|---|---|---|---|
| 数据 | RCT 事实 outcome/cost | RCT outcome/cost | 都匹配 |
| Treatment | 多 treatment | 多 treatment、有序强度 | 当前券档是否严格有序需确认 |
| 预测对象 | 各 treatment 绝对 revenue/cost | 相对 control 的 response/cost uplift | 当前 EUEN 更接近 E3IR 输出 |
| 约束 | 总预算 `Σc≤B` | 增量成本预算 `ΣΔc≤B` | 当前是 `ΣΔc≤κΣΔy`，都需改 |
| 决策梯度 | IPW 策略价值；softmax/熵/有限差分 | 手工可微 ILP backward | DFCL 因果监督更扎实 |
| 跨预算 | 多 `λ` 联合训练 | 论文按给定 B 进入 ILP | DFCL 更自然 |
| 结构先验 | 基本不限定宿主模型 | 单调、平滑、相邻增量 | E3IR 结构可作为独立模块 |
| 训练开销 | PL/MER 低，IFD 中等 | 高，batch 内 ILP＋custom backward | DFCL-PL 更适合 MVP |
| 部署 | 仍需预算求解 | 仍需 ILP | 都没跳出 OR |
| 证据 | 离线＋4 周线上 A/B | 离线公共＋生产数据 | DFCL 证据更强 |
| 公开代码 | 本轮未找到官方仓库 | 本轮未找到官方仓库 | 都要自主复现 |

---

## 5. 迁移方案 A：DFCL-PL-CAC（首选 MVP）

### 5.1 模型输出

尽量保留现有 EUEN/EEUEN：

```text
Δy_hat_i,a = 当前 uplift 输出
Δc_hat_i,a = treatment cost 输出 - control cost
```

当前 cost head 只对 treatment 预测、control 在 OR 中近似设为 0。若真实 control 也有非零成本，建议显式增加 control cost head，否则“增量成本”口径会偏。

### 5.2 CAC-aware score

对一组训练 CAC 目标 `K = {κ_1, κ_2, ...}` 和乘子 `Λ = {λ_1, λ_2, ...}`：

```text
s_i,a(κ,λ)
= (1 + λκ)Δy_hat_i,a - λΔc_hat_i,a

π_i,a(κ,λ)
= softmax_a(s_i,a / τ)
```

control 设置：

```text
Δy_hat_i,0 = 0
Δc_hat_i,0 = 0
```

这样用户永远可以选择“不干预”。

### 5.3 RCT 策略价值 loss

事实动作 `t_i`、事实增量净价值不能逐用户直接观察，因为 control 反事实缺失。可以采用两种实现。

#### 实现 1：绝对 outcome/cost 的 IPW（更贴近 DFCL）

让模型同时输出各 action 的绝对 `y_hat`、`c_hat`，用事实：

```text
q_i(κ,λ)
= (1 + λκ)y_i - λc_i

L_decision
= - mean_{κ,λ,i}
   [π_i,t_i(κ,λ) q_i(κ,λ) / p_i,t_i]
```

control 的 baseline 会出现在所有策略价值中，在比较策略时原则上可抵消。优点是 estimator 直接、与原文一致。

#### 实现 2：DR/AIPW value（推荐中期使用）

纯 IPW 方差可能较大，可用 outcome model 做 doubly robust：

```text
V_DR(π)
= mean_i [
    Σ_a π_i,a q_hat_i,a
    + π_i,t_i / p_i,t_i × (q_i - q_hat_i,t_i)
  ]
```

它同时利用全 treatment 预测和 RCT 残差修正。只要 propensity 或 outcome model 任一正确，策略价值仍一致；在多 treatment、组间样本不均衡时通常比纯 IPW 稳定。

### 5.4 总 loss

```text
L_total
= L_outcome
  + γ_c L_cost
  + γ_e L_exposure        # EEUEN 才有
  + β L_decision_CAC
```

训练策略：

```text
阶段 1：只训练原 prediction loss，得到稳定 warm start
阶段 2：逐步把 β 从 0 增加到目标值
阶段 3：τ 从较大值退火到较小值
```

不要一开始就把 decision loss 权重开得很大，否则模型可能牺牲 calibration，只学会在当前 κ/λ 网格上排序。

### 5.5 与当前分桶后验的关系

有两种路线：

#### A1. 保留分桶后验

训练时 decision loss 仍使用用户级软策略；部署继续：

```text
模型分数 → 分桶 → 桶内 RCT 后验 → 当前 OR
```

优点是上线风险小。缺点是训练 surrogate 与部署链路仍不完全一致：模型学的是用户级策略，实际用的是分桶后的历史后验。

#### A2. 取消分桶后验

部署直接用用户级 `Δy_hat/Δc_hat` 进入 CAC OR。它最符合论文，但需要解决：

- 用户级 CATE calibration；
- cost 预测尾部风险；
- 约束违例；
- 跨时间漂移；
- 无分桶后验的可解释性和稳定性。

建议 MVP 先走 A1，确认 decision loss 是否能改善现有仿真曲线；若有效，再用 A2 做增量实验，而不是同时更换训练和部署两端。

---

## 6. 迁移方案 B：E3IR-style 单调/平滑 EUEN

### 6.1 先确认 treatment 是否真有序

E3IR 的单调结构只适用于 treatment 存在明确强度顺序，且业务上合理相信：

```text
treatment 强度增加
⇒ 期望完单不下降
⇒ 期望成本不下降
```

当前券名：

```text
留存6张3元
8元留存5张3
留存6张5元
10元留存5张3
```

券包同时改变门槛、张数、面额，未必能排成单一剂量轴。因此必须先用 RCT 做：

1. 全局 treatment 平均 outcome/cost 顺序；
2. 核心人群分层后的顺序；
3. 相邻 treatment dominance 比例；
4. 是否存在大量“高档券反而低转化”的真实异质性。

如果只有当前 OR 使用的三档券能形成稳定顺序，就只在这三档上用单调 head；不要为了套论文把所有券硬排序。

### 6.2 推荐的结构改造

若 treatment 顺序成立：

```text
shared user representation Φ
        +
treatment intensity embedding δ_k
        ↓
shared incremental head
        ↓
softplus(raw_increment_k) ≥ 0
        ↓
累计得到 τ_1, τ_2, τ_3
```

建议用 `softplus`，而不是平方或 exp：

- 平方在 0 附近梯度为 0；
- exp 容易产生极大值；
- softplus 更稳定。

EEUEN 仍可保留：

```text
最终 uplift_k
= exposure_prob_k × exposure_uplift_k
```

但单调性应该约束在哪一层需谨慎：

- 若券强度只影响“曝光后的效果”，可约束 `exposure_uplift_k`；
- 若不同渠道/券导致曝光概率也变化，最终 `exposure_prob_k × uplift_k` 未必单调；
- 不应强制 exposure probability 本身随券档单调。

### 6.3 平滑正则的简单可复现版本

先不用论文复杂的 Lipschitz 上界，可做：

```text
L_smooth_1
= mean_i,k (τ_i,k+1 - τ_i,k)^2

或二阶差分：

L_smooth_2
= mean_i,k (τ_i,k+1 - 2τ_i,k + τ_i,k-1)^2
```

如果 treatment 间距不同，应按实际补贴强度差归一化，而不能把各档当等距。

### 6.4 不建议原样使用 E3IR allocation CE

可替换为：

```text
方案 B1：DFCL 的 IPW/DR 策略价值 loss
方案 B2：hard ILP + IFD/black-box gradient
方案 B3：连续松弛后的 CAC violation + value loss
```

即：**保留 E3IR 的 predictor，使用 DFCL 的因果决策监督**。这是两篇论文最自然、也最适合当前业务的组合。

---

## 7. 如果一定要做“可微 CAC-OR 层”

前向问题：

```text
max_z Σ z Δy_hat

s.t. Σ z(Δc_hat - κΔy_hat) ≤ 0
     one action per user
```

可考虑三种 backward：

### 7.1 对偶 softmax（首选）

直接用：

```text
π = softmax(((1+λκ)Δy_hat - λΔc_hat)/τ)
```

训练时软策略、部署时硬 OR。实现最稳，等价于 DFCL-PL-CAC。

### 7.2 连续松弛＋隐式微分

把 `z∈{0,1}` 放松到 simplex，加熵或二次正则，使用 KKT/凸优化层求梯度。它比 E3IR 的整数 basis backward 容易验证，但存在 relaxation gap。

### 7.3 E3IR/CombOptNet 式整数 backward

保留硬 ILP，手工构造可行邻域方向。只有在软策略已证明显著、但 hard/soft gap 成为主要瓶颈时才值得做。

当前场景还有 bucket、配额、时间窗等业务逻辑，整数 backward 的每个新约束都要重新检查 mismatch 函数与可行方向，维护成本较高。

---

## 8. 实验设计

### 8.1 最小消融矩阵

| 组别 | Predictor | 决策 loss | 部署链路 | 目的 |
|---|---|---|---|---|
| T0 | 当前 EUEN/EEUEN | 无 | 分桶＋当前 OR | 基线 |
| T1 | 当前 EUEN/EEUEN | DFCL-PL-CAC | 分桶＋当前 OR | 只验证目标对齐 |
| T2 | 单调/平滑 EUEN | 无 | 分桶＋当前 OR | 只验证 E3IR 结构先验 |
| T3 | 单调/平滑 EUEN | DFCL-PL-CAC | 分桶＋当前 OR | 两者组合 |
| T4 | T3 | DFCL-PL-CAC | 用户级 CAC OR | 验证取消分桶 |
| T5 | T3 | DFCL-IFD-CAC | 用户级 CAC OR | 验证 hard decision 上限 |

### 8.2 训练/调参/评估切分

不能用同一段 RCT 同时：

- 拟合 predictor；
- 调 `β/τ/κ/λ`；
- 计算最终 EOM。

建议至少做三段时间切分：

```text
train：模型参数
calibration/validation：温度、loss 权重、λ 与 CAC 落点
test：冻结后只做策略价值与约束评估
```

若样本允许，使用按日期滚动的多折验证，避免单周偶然性。

### 8.3 指标优先级

一级指标：

1. 测试集实际增量完单 PV；
2. 测试集实际增量 CAC；
3. 多个目标 `κ` 下的完整 value-CAC frontier；
4. CAC 约束违例概率和违例幅度；
5. IPW/DR 策略价值及置信区间。

二级指标：

1. 各 treatment AUUC/QINI；
2. cost calibration 与 tail error；
3. 策略动作占比、相对基线换券率；
4. 不同人群、渠道、日期的稳定性；
5. 有效样本量 `ESS = (Σw)^2 / Σw^2`。

不能只看 AUUC，也不能只看某个 CAC 单点。最终应比较整条 frontier，以及业务目标 CAC 附近的置信区间。

### 8.4 必须做的稳健性检查

- propensity 是否准确、每个动作是否有足够覆盖；
- IPW 权重截断前后结论是否一致；
- negative uplift / negative incremental cost 如何处理；
- 小样本桶是否被 decision loss 放大噪声；
- treatment 顺序假设是否被 RCT 数据支持；
- 不同 `κ`、`λ` 网格是否过拟合测试预算点；
- soft policy 的提升是否能转移到 hard OR；
- constraint calibration 在跨周是否漂移。

---

## 9. 实现工作量与风险判断

| 方案 | 预计改动 | 技术风险 | 推荐度 |
|---|---|---:|---:|
| DFCL-PL-CAC | 在现有训练循环增加 factual policy value loss | 中 | 最高 |
| E3IR 单调 head | 改 treatment head 与正则 | 中 | 高，但先验要验证 |
| DR value loss | 增加 propensity/outcome correction | 中高 | 高 |
| 用户级 CAC OR | 改仿真与部署输入、加强校准 | 高 | 中 |
| DFCL-IFD-CAC | 自研最小切换扰动与增量 EOM | 高 | 中后期 |
| E3IR 完整 ILP backward | JAX/PyTorch custom layer 或 TF 自定义梯度 | 很高 | 低 |

### Go / No-Go 标准

MVP 可以进入下一阶段，至少要满足：

```text
1. 在两个以上独立时间窗口，目标 CAC 附近的增量完单均提升；
2. DR/IPW 结论方向一致，且 ESS 不过低；
3. hard OR 复算后增益仍存在；
4. CAC 违例不高于当前基线，或可通过 λ 校准消除；
5. 动作分布没有异常坍缩到单一券档；
6. 提升不是仅来自改变分桶参数或测试泄漏。
```

若 decision loss 只提升训练内 EOM、跨周不稳定，优先排查估计方差和 propensity，而不是继续加大模型容量。

---

## 10. 对“更大革新 / 大模型方向”的判断

DFCL 和 E3IR 可以证明“训练目标与决策对齐”值得做，但它们不是 MT 所说的“直接 end-to-end 决策模型”终局：

```text
它们仍然显式建模 uplift/cost
→ 仍然依赖约束求解或其 surrogate
→ 仍然主要解决单期静态分配
```

对于当前只有 3～4 个固定动作的业务，LLM 不会天然改善求解；真正值得大模型化的部分是：

1. 长历史、多轮触达、补贴疲劳和延迟反馈的序列状态表征；
2. 新券/新活动的语义表示和少样本迁移；
3. 将预算/CAC/库存等约束作为条件输入，学习跨约束策略；
4. 直接生成动作或计划，但仍保留安全约束投影和离线因果评估。

因此合理的研究路线是：

```text
先用 DFCL-PL-CAC 证明“决策信号反传”本身有增益
→ 再验证单调/平滑结构
→ 再做用户级直接策略
→ 最后才比较 Transformer/生成式策略是否比小型策略网络有独立增益
```

否则一开始上大模型，无法区分收益究竟来自：模型容量、序列特征、decision loss，还是策略评估噪声。

---

## 11. 最终选型建议

### 建议立项名称

> 面向多券增量 CAC 约束的决策导向因果学习

### 第一版技术定义

```text
宿主模型：当前 EUEN/EEUEN
数据：现有 RCT
动作：control + 当前 OR 的 3 个 treatment
约束：增量 CAC
训练：prediction loss + DFCL-PL-CAC
估值：DR 为主、IPW 为辅
部署：先保留分桶＋当前 OR
评估：多 CAC frontier + 跨周验证 + hard OR 复算
```

### E3IR 的使用方式

```text
采用：相邻增量、共享 treatment head、可选平滑正则
暂不采用：原文 CE(t,z) allocation loss
暂不采用：完整整数 ILP custom backward
```

一句话结论：

> DFCL 更适合作为当前业务的第一条端到端主线；E3IR 更适合作为 predictor 结构增强。最佳组合不是二选一，而是“E3IR-style uplift head + DFCL-style causal decision loss + 当前 CAC OR”。

---

## 参考原文

1. Hao Zhou et al. *Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization*. KDD 2024. https://arxiv.org/abs/2407.13664
2. Zexu Sun et al. *End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling*. RecSys 2024. https://arxiv.org/abs/2408.11623
3. ACM DOI for E3IR: https://doi.org/10.1145/3640457.3688147
