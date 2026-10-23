# 多 Treatment Uplift＋成本约束的端到端决策论文调研

> 检索日期：2026-10-12  
> 业务范围：RCT、多 treatment、用户级增量效果与成本、预算/CAC 约束、营销资源分配

## 一、检索结论

严格符合“因果效果预测＋成本预测＋全局资源分配＋决策信号反向训练预测模型”的论文并不多。与当前项目最接近的研究主线是：

```text
2023  DHC：直接学习 OR 真正使用的决策因子
  ↓
2023  MEO：将预算下的真实营销目标作为训练正则
  ↓
2024  DFCL：系统处理反事实、预算变化和离散梯度
  ↓
2024  E3IR：将 ILP 作为可微分配层嵌入 uplift 网络
  ↓
2025  Bi-DFCL：用双层优化联合观察数据与 RCT
```

这五篇构成最值得精读的核心文献。它们均不是 LLM 方法，主要使用普通深度因果模型、策略价值估计和可微/近似可微优化。严格使用语言大模型解决这一完整问题的成熟工业论文仍然很少。

---

## 二、核心论文对比

| 论文 | 年份/出处 | 核心对齐方式 | 是否多 treatment | 是否显式预算 | 与当前项目贴合度 |
|---|---|---|---:|---:|---:|
| DHC | AAAI 2023 | 直接学习 ROI/边际效用等决策因子 | 是 | 是 | 很高 |
| MEO | arXiv 2023，快手 | 将预算下 EOM 作为正则，用黑盒梯度训练 | 是 | 是 | 很高 |
| DFCL | KDD 2024，美团 | RCT 策略价值＋对偶决策损失＋有限差分 | 是 | 是 | 最高 |
| E3IR | RecSys 2024，腾讯/人大 | 单调 uplift 网络＋可微 ILP 分配层 | 是 | 是 | 最高 |
| Bi-DFCL | NeurIPS 2025，美团 | RCT 外层决策损失＋观察数据内层预测损失 | 是 | 是 | 很高 |

---

## 三、最值得继续精读的论文

### 3.1 DHC：直接学习运筹需要的决策因子

论文：[Direct Heterogeneous Causal Learning for Resource Allocation Problems in Marketing](https://arxiv.org/abs/2211.15728)，AAAI 2023。

DHC 不再要求先分别精确预测收益 uplift 和成本 uplift，再组合成决策指标。它先分析 OR 最终依赖什么排序量，再直接学习这个“决策因子”。

二元 treatment 下，预算分配近似按 ROI 排序：

```text
ROI_i = 收益 uplift_i / 成本 uplift_i
```

多 treatment 下，它学习相邻 treatment 升档的边际效用：

```text
边际效用_i,j
= (收益_i,j+1 − 收益_i,j)
  / (成本_i,j+1 − 成本_i,j)
```

论文基于 RCT 设计定制损失，使模型直接学习这些决策因子的排序，再通过比较和二分搜索得到资源分配结果。

与当前项目的关系：

- 很接近“模型不必把所有潜在结果都预测到绝对准确，只需学对最终分配需要的收益—成本顺序”；
- 比 DFCL 更简单，适合做直接 ROI/边际 ROI 基线；
- 对当前增量 CAC 约束具有直接启发；
- 不属于“求解器梯度反传”，而是通过选择正确的学习目标实现预测—决策对齐。

### 3.2 MEO：把真实营销目标作为模型正则

论文：[An End-to-End Framework for Marketing Effectiveness Optimization under Budget Constraint](https://arxiv.org/abs/2302.04477)，2023，快手。

该方法先用 S-Learner 预测每个用户在各 treatment 下的响应和成本，再执行预算分配。它利用 RCT 的 Expected Outcome Metric 构造：

```text
Q(预测收益矩阵, 预测成本矩阵)
= 给定预算下，当前分配策略的实验估计收益
```

训练目标为：

```text
L = L_prediction − λ × Q
```

由于 `Q` 中包含拉格朗日搜索、动作选择和离散分配，无法直接求导，论文将它视为黑盒，使用：

- 有限差分；
- Natural Evolution Strategies（NES）；

估计 `Q` 对预测收益和成本的梯度，再反向更新 S-Learner。

与当前项目的关系：

- 和“预测 uplift/cost → 运筹 → 用实验结果评估策略”链路非常接近；
- 它把最终预算下的策略效果直接接回模型训练；
- 可以看作 DFCL 之前的一种黑盒 decision-focused 实现；
- 主要问题是梯度估计开销大、噪声高，而且针对固定预算训练时跨预算泛化有限。

论文称已在快手的大规模现金激励分配中部署。检索到的 arXiv 版本写明代码将公开，但本轮未找到对应的官方代码仓库。

### 3.3 E3IR：把整数规划作为可微网络层

论文：[End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling](https://arxiv.org/abs/2408.11623)，RecSys 2024，人大/腾讯。

E3IR 由两个模块组成：

```text
Uplift Prediction Module
        ↓
Differentiable Allocation Module
```

第一部分同时预测 response uplift 和 cost uplift，并利用营销领域知识约束多档 treatment：

- 用相邻档位的增量 head 累加得到各 treatment 结果；
- 强制相邻增量为非负，保证激励响应曲线单调；
- 共享各档位 head 参数并加入 Lipschitz 正则，保证曲线平滑。

第二部分把多选背包 ILP 作为网络中的可微层。前向过程求预算分配；反向过程构造可实现的下降方向，将分配损失对动作的梯度转换为对 response uplift 和 cost uplift 的梯度。

总损失为：

```text
L_E3IR = L_predict + β × L_allocation
```

与当前项目的关系：

- 多 treatment、uplift、cost、预算约束与当前场景高度一致；
- “相邻档位增量累加”与 EUEN 当前 residual treatment 结构非常接近；
- 可以研究把当前 EUEN 宿主替换为 E3IR 式单调/平滑 head，再接可微分配层；
- 论文使用总预算约束，迁移到增量 CAC 仍需重新定义分配层；
- 它是除 DFCL 外最值得下一篇详细拆解的论文。

### 3.4 Bi-DFCL：用双层优化融合观察数据和 RCT

论文：[Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](https://arxiv.org/abs/2510.19517)，NeurIPS 2025，美团。

Bi-DFCL 进一步处理工业数据中的偏差—方差矛盾：

```text
观察数据：规模大，但存在历史策略、选择和位置偏差
RCT 数据：相对无偏，但昂贵且样本少
```

它将两类目标放进双层优化：

```text
内层：用大规模观察数据学习预测模型
外层：用小规模 RCT 的无偏决策质量校正训练方向
```

外层使用 RCT 构造 OR 决策质量估计和 surrogate decision loss，内层训练观察数据预测目标，两层通过隐式微分连接。这样既利用观察数据降低方差，又让 RCT 决策信号纠正观察数据偏差。

与当前项目的关系：

- 如果历史自然策略数据很多、随机实验样本有限，它比纯 RCT 的 DFCL 更适合；
- 如果当前已有充足且动作覆盖良好的 RCT，第一阶段未必需要立刻上双层框架；
- 工程复杂度明显高于 DFCL，应放在 DFCL 基线验证有效之后。

---

## 四、与当前项目相关，但不属于同一种端到端训练

### 4.1 实时系统端到端，训练仍是两阶段

[Data-Driven Real-time Coupon Allocation in the Online Platform](https://arxiv.org/abs/2406.05987)，2024，美团。

它把 CVR 预测、等距回归校准、拉格朗日实时选券和预算反馈控制组成完整线上系统，支持随机流量下的实时预算 pacing。但预测目标没有由最终 OR 决策损失反向训练，因此属于“系统链路端到端”，不是 DFCL 意义下的“训练端到端”。

这篇对线上部署、动态预算和实时控制很有价值。

### 4.2 连续 treatment 的 predict-then-optimize

[Uplift Modeling with Continuous Treatments: A Predict-then-Optimize Approach](https://arxiv.org/abs/2412.09232)，2024/2025。

它先估计连续剂量响应，再使用 ILP 进行剂量分配，并支持公平性、个体成本和效用约束。框架仍明确属于 predict-then-optimize，决策质量不反向训练因果模型。

代码：[SimonDeVos/UMCT](https://github.com/SimonDeVos/UMCT)。

### 4.3 不确定成本下的优先级学习

[Treatment Allocation under Uncertain Costs](https://arxiv.org/abs/2103.11066)。

它证明在成本随协变量变化时，预算约束下的最优分配可表示为基于优先级分数的阈值规则，并研究如何从随机实验学习该分数。它是 DHC/DFCL 的重要统计基础，但不是神经网络＋OR 的端到端系统。

---

## 五、2026 年更前沿的直接策略与序列决策

### 5.1 GOAL：约束条件化的生成式激励策略

[Generative Optimization for Incentivized Advertising with Global Level Constraints](https://arxiv.org/abs/2608.04421)。

GOAL 不先独立预测每档 uplift，而是把用户历史和全局约束压力输入生成式策略，直接生成连续激励金额，并用约束策略优化适配不同 ROI 水平。它还显式处理延迟反馈和营销疲劳。

它更接近“模型直接输出决策”，但属于生成式序列模型，并不是使用通用语言知识的 LLM。

### 5.2 SACO：面向长期收益的序列优惠券决策

[SACO: Sequence-Aware Constrained Optimization Framework for Coupon Distribution in E-commerce](https://doi.org/10.1609/aaai.v40i17.38525)，AAAI 2026。

SACO 面向平台和用户反复交互的连续发券问题，直接学习长期收益下的优惠券策略。它适合研究疲劳、等待优惠和多轮干预，问题范围超出当前一次性静态 uplift 分配。

### 5.3 ReAlloc：跨渠道预算的因果策略学习

[Multi-channel Uplift Policy Learning](https://arxiv.org/abs/2607.28182)，2026，淘宝场景。

ReAlloc 处理多个营销渠道之间的连续预算构成，使用正交 teacher 从短期日志提取较无偏的局部梯度，再蒸馏到长期结构化策略中。它更接近“渠道预算组合”，而不是当前“用户 × 券型”的离散分配。

### 5.4 BCCB：边学习 uplift、边在线花预算

[Budget-Constrained Causal Bandits](https://arxiv.org/abs/2604.26169)，2026。

该工作将 uplift 学习、探索和预算 pacing 合并为在线因果 bandit，针对历史实验数据不足和冷启动场景。当前若已有充足离线 RCT，它不是最优先方向；若新券、新人群冷启动严重，则值得关注。

---

## 六、不同“端到端”路线的本质区别

| 路线 | 代表论文 | 模型学习什么 | 部署时如何决策 |
|---|---|---|---|
| 决策因子学习 | DHC | ROI或边际效用排序 | 排序、比较、二分搜索 |
| 业务目标正则 | MEO | 收益与成本，同时最大化 EOM | MCKP/对偶分配 |
| 决策损失学习 | DFCL | 收益与成本，接受策略价值监督 | 对偶分配或 OR |
| 可微优化层 | E3IR | uplift/cost，并穿过 ILP 反传 | ILP 分配 |
| 双层因果决策 | Bi-DFCL | 观察数据预测＋RCT 决策校正 | OR/对偶分配 |
| 直接策略生成 | GOAL/SACO | 直接输出激励动作或序列 | 模型策略＋约束控制 |

---

## 七、针对当前项目的阅读与实验顺序

### 第一优先级：E3IR

原因：它与当前 EUEN 的多 treatment residual 结构、uplift/cost 双输出和后续预算分配最接近。建议重点看：

```text
1. 相邻 treatment 的单调增量 head
2. smooth/Lipschitz 约束
3. ILP 层如何产生 response/cost 梯度
4. allocation loss 如何利用 RCT treatment
```

### 第二优先级：MEO

原因：它最直观地展示“最终预算下的实验策略价值如何作为正则反传”，也能帮助理解 DFCL 相比早期黑盒梯度方法解决了什么问题。

### 第三优先级：DHC

原因：实现最简单，可以先验证直接学习增量 ROI 或相邻券边际效用是否优于“分别预测 uplift/cost＋分桶后验”。它也是增量 CAC 场景很自然的基线。

### 第四优先级：Bi-DFCL

原因：只有在确认观察数据规模远大于 RCT、且历史策略偏差确实是关键限制时，双层优化的复杂度才更有价值。

### 长期方向：GOAL 或 SACO

当研究目标从单次静态券分配扩展到多轮发券、长期活跃、补贴疲劳和跨期预算时，再考虑生成式策略或序列约束优化。

---

## 八、对当前研究问题的直接判断

现有论文尚未直接覆盖当前完整链路：

```text
用户级多 treatment EUEN
→ 预测分数分桶
→ 桶内 RCT 后验 uplift/cost
→ 增量 CAC 约束 OR
```

主流端到端论文通常选择取消分桶，直接让用户级预测进入分配层，并通过策略价值或可微优化梯度训练模型。因此，当前项目可以比较两条路线：

```text
路线 A：保留分桶后验，设计桶级 decision loss

路线 B：取消分桶后验，让用户级 uplift/cost 直接进入
        DFCL、E3IR 或直接决策因子框架
```

此外，大多数核心论文使用总预算约束：

```text
总成本 ≤ B
```

当前业务使用增量 CAC：

```text
总增量成本 ≤ κ × 总增量完单
```

因此，真正可能形成项目贡献的部分，是将 decision-focused causal learning 系统地改造成：

> 面向多券、桶内 RCT 后验和增量 CAC 约束的端到端因果决策框架。
