# 大模型、因果效应预估与补贴决策：WWT 探索建议

调研日期：2026-09-24。检索范围：2023-09-24 至 2026-09-24，重点相关成果集中于 2024—2026 年。不讨论因果发现。本轮为选题调研，未复现实验；论文结果属于作者报告，不能直接外推到本项目。下文区分已有工作和针对 WWT 提出的实验设计。

**建议主线：先建立 RCT 驱动、成本约束条件化的直接策略学习基线，再研究预训练用户表征对该策略的提升。** 若希望近期验证严格意义上的语言大模型价值，增加一个独立的语义表征小实验；不要同时改表征、策略目标和评估口径。

## 1. 项目背景如何影响选题

已阅读的主要依据：

- [EUEN/EEUEN 全流程](../project_background/wwt_eeuen_model_iter_full_flow_memo.md)：短信/电话外呼、触达成功后发券，4 个 treatment＋control，曝光与 uplift 分解，成本预估和运筹仿真。
- [特征与运筹全流程](../project_background/wwt_euen_feature_select_full_flow_memo.md)：模型分数用于分桶，桶内统计实验效果，OR-Tools 在 CAC 约束下分配券型，再在后续日期上验证。
- [EFIN 迭代](/Users/didi/wwt_local/wwt/demo/EFIN/efin_model_iteration_doc.md)：明确说明 RCT 券包数据；已引入券包结构属性、treatment-aware attention；记录了自然响应与 uplift 表征粗暴共享的问题。
- [Loss 迭代](/Users/didi/wwt_local/wwt/demo/Loss_test/loss_iteration_memo.md)：已探索零膨胀、长尾 outcome/cost 的 ZILN，以及 cost loss 权重问题。

因此，本项目有四个区别于泛泛 LLM＋因果推断的特点：

1. **存在 RCT 数据。** 若随机分流机制有效、无事后选择，分配动作的混杂不是首要矛盾。更直接的问题是有限样本下的异质性、跨期漂移、成本校准和策略价值。文档中的 RCT 描述还需用分流概率、随机化单位、过滤逻辑验证。
2. **当前动作空间很小。** 4 券＋control；现有 OR 常只纳入 3 券＋control。直接策略可先用小型网络输出动作概率，没有必要为 4 个动作引入复杂金额 tokenizer。
3. **现有 OR 不只是把个体预估送进背包求解。** 模型分数承担分桶职责，桶内实验统计也承担效果估计与降噪职责。去掉分桶和 OR 时，需要用独立策略评估、正则化及不确定性控制接替这些功能。
4. **业务目标是增量完单与实际增量 CAC。** 文档中的“增量 CVR”包含人均增量完单 PV 口径，不能直接当作二分类转化率。AUUC、cost AUC、曝光 AUC 应是诊断指标，策略价值才是主指标。

## 2. 将 MT 的思路拆成三个问题

| 问题 | 真正希望改变什么 | 对应路线 |
|---|---|---|
| 大模型能否帮助效应预估？ | 用更好的用户/券表征，提高 CATE 的样本效率与迁移能力 | LLM 语义增强、行为预训练、因果基础模型 |
| 一二阶段能否一致？ | 让训练关注错误决策的代价 | Decision-Focused Learning、DFCL/Bi-DFCL |
| 能否直接输出策略？ | 从用户状态和成本约束直接决定发哪张券 | 约束策略学习、序列策略、生成式激励分配 |

“直接效应”也需明确：直接回归 CATE 与中介分析中的 natural/controlled direct effect 是不同问题。当前发券优化首先需要分配策略的总效应；若要拆分触达、领取、核销路径，才进入中介效应识别。

端到端至少有三种含义：

- 数据处理端到端：原始文本→因果效应，例如 NATURAL。
- 训练端到端：用决策损失训练预估模型，但部署时仍有优化器，例如 DFCL。
- 决策输出端到端：状态＋约束→动作，例如直接策略网络。

第三种也可以有预训练、奖励模型和离线校准。衡量框架升级时，应明确是否移除了线上逐批组合求解，不能要求训练必须只有一个步骤。全局约束仍需要协调机制；是否保留一个成本控制器，与是否保留原有分桶＋OR 链路是两件事。

## 3. 值得阅读的论文与工业材料

### 3.1 与框架升级最相关

| 工作、时间及来源 | 核心贡献 | 对 WWT 的意义与边界 |
|---|---|---|
| **GOAL: Generative Optimization for Incentivized Advertising with Global Level Constraints**，快手/电子科大，KDD 2026；2026-08 公开。[原文](https://arxiv.org/html/2608.04421v1) | 历史序列与 λ 条件化的激励生成，SCPO 策略优化 | 与 MT 提及的 next coin 高度接近，但不能确认就是同一篇。可借鉴约束条件化；它属于受 LLM 启发的生成式决策模型，不能据此视为通用语言模型迁移成功 |
| **SACO: Sequence-Aware Constrained Optimization Framework for Coupon Distribution in E-commerce**，人大/字节等，AAAI 2026。[原文](https://ojs.aaai.org/index.php/AAAI/article/view/38525) | 面向重复到访、多用户多轮交互，直接学习长期收益的券分配策略 | 动作类型比连续金币更接近本项目。需要真实重复决策轨迹；当前汇总特征不是完整轨迹 |
| **Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization**，美团/南京大学，KDD 2024。[论文](https://arxiv.org/abs/2407.13664)、[美团官方解读](https://tech.meituan.com/2024/07/26/KDD-2024.html) | 下游决策目标进入因果预估训练，处理反事实与优化梯度问题 | 非 LLM，是验证“目标对齐到底值多少”的强基线；训练端到端不等于部署免 OR |
| **Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization: Bridging Observational and Experimental Data**，美团等，NeurIPS 2025。[论文](https://arxiv.org/abs/2510.19517)、[会议全文](https://papers.neurips.cc/paper_files/paper/2025/file/93427bc08aef8f9d6d5366d70284f571-Paper-Conference.pdf) | 用实验数据评估决策质量，并通过双层优化纠正观察数据训练方向 | 若存在大规模历史策略数据＋较小 RCT，优先级很高；纯 RCT 阶段不必复刻全部双层架构 |
| **Practical Marketplace Optimization at Uber Using Causally-Informed Machine Learning**，Uber，2024-07。[原文](https://arxiv.org/abs/2407.19078) | 因果响应估计与城市预算优化联动，使用 S-learner、样条与优化算法 | 非 LLM；提醒我们：业务整体端到端系统仍可保留优化器，不能仅凭标题判断架构 |

GOAL 精读后的边界：其 causal encoder 使用时序因果卷积，不等于识别了干预效应；仍训练奖励模型，推理前仍校准 λ。其 ROI 与我们的增量 CAC 不同，需要重写目标。论文报告线上 A/B，但正文“每组 65%”的流量描述有歧义，不宜照搬该实验设计。[方法及附录](https://arxiv.org/html/2608.04421v1)

### 3.2 严格意义上的语言大模型＋效应估计

| 工作与时间 | 做了什么 | 适用性 |
|---|---|---|
| **Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation（CURL）**，2026-07 预印本。[原文](https://arxiv.org/abs/2607.26599) | 冻结 LLM，以 assignment/heterogeneity 两类提示构建表征，并按估计不确定性分配语义增强 | 适合用作轻量插件实验；公开 benchmark 证据，不是 WWT 规模工业证明。RCT 中 assignment 分支不应被机械当作必需 |
| **LLM-Driven Treatment Effect Estimation Under Inference Time Text Confounding**，NeurIPS 2025。[会议原文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0f3b62a67900418cbb8583839129ffc6-Abstract-Conference.html) | 处理训练时结构信息完整、推理时只剩不完整文本的落差，结合 LLM 与定制 DR learner | 适合未来接入文本画像或跨渠道缺失信息；与当前训练推理同构的表格链路并非完全对位 |
| **Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments**，2024-10 首稿；作者页后续题名为 Causal Inference with Generative Artificial Intelligence。[首稿记录](https://arxiv.org/abs/2410.00903)、[作者页面](https://imai.fas.harvard.edu/research/llm/) | 利用生成模型内部表征研究文本 treatment 属性的效应及识别条件 | 与“LLM 表征、解耦”相关；特别适合文案成为 treatment 的新问题，不等于任意用户 embedding 都能去混杂 |
| **End-To-End Causal Effect Estimation from Unstructured Natural Language Data（NATURAL）**，多伦多/Vector/Meta AI，NeurIPS 2024。[会议原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8d2aad53fa148791ffd004a7c9cce5ad-Abstract-Conference.html) | 从非结构文本提取信息并结合经典因果估计 | “端到端”终点是效应估计，不是预算约束下直接发券；当前已有结构化数据，优先级较低 |

### 3.3 基础模型范式：不一定使用自然语言

| 工作与时间 | 做了什么 | 建议用途 |
|---|---|---|
| **CURE: A deep learning framework pre-trained on large-scale patient data for treatment effect estimation**，Patterns 2024。[原文](https://www.sciencedirect.com/science/article/pii/S2666389924000813) | 大规模无标签纵向数据预训练，再针对效应估计微调 | 借鉴“全量行为数据预训练→RCT 微调”，不要直接迁移医疗领域权重 |
| **CausalPFN: Amortized Causal Effect Estimation via In-Context Learning**，2025，NeurIPS 2025。[原文](https://arxiv.org/abs/2506.07918) | 在满足可忽略性假设的模拟数据生成机制上训练，再对新数据做 in-context 效应估计 | 适合先看少样本实验任务；需验证上下文规模、连续 outcome、多 treatment 的支持，不能默认适配现有全量样本 |
| **Foundation Models for Causal Inference via Prior-Data Fitted Networks（CausalFM）**，2025 首稿，2026-02 修订。[原文](https://arxiv.org/abs/2506.10914) | 显式设计因果先验并训练 PFN；当前版本覆盖多种识别设置 | 可研究补贴业务先验下的跨实验预训练；先验有效性与业务迁移是关键，不是加大参数就能解决 |
| **Causal Foundation Models with Continuous Treatments**，2026-05 首稿，2026-07 更新，预印本。[原文](https://arxiv.org/abs/2605.15133) | 通过合成机制预训练，用上下文学习重建剂量响应曲线 | 适合未来连续补贴强度；当前券包有数量、门槛、首券等多维属性，不能直接压成“总金额” |

补充：如要严格建模历史 treatment 影响后续状态，再影响后续决策，可读 [G-Transformer: Counterfactual Outcome Prediction under Dynamic and Time-varying Treatment Regimes](https://pmc.ncbi.nlm.nih.gov/articles/PMC12113242/)。它将 Transformer 与 g-computation 结合，属于序列反事实建模，并非语言大模型。

## 4. 推荐选题排序

### A. 成本约束条件化的直接策略学习：首选主线

**研究问题：在同等实际增量 CAC 下，直接优化策略价值能否超过“uplift 分桶＋桶内估计＋OR”？**

先用现有静态特征与 RCT，学习 π(a|x,λ)。a 包含不发券和现有券型。λ 表示成本惩罚强度；提高 λ 应倾向于节约成本，但有限样本训练出的网络不保证严格单调，需要验证。

第一版不需要 LLM。用现有 shared-bottom 或小型 MLP 输出 4/5 类动作概率，就能先检验架构价值。随后加入预训练表征，才能证明大模型提供了额外收益。

**针对 WWT 的建议目标（本项目设计，不是上述论文的逐式复现）：**

令 Y 为统一窗口的完单量，C 为同窗口成本，a=0 为不干预，定义

τY(a,x)=E[Y(a)−Y(0)|x]，τC(a,x)=E[C(a)−C(0)|x]。

如果主约束为增量 CAC≤κ，在策略总增量完单为正的情况下可写为：

```
maximize    E[Σa π(a|x) τY(a,x)]
subject to  E[Σa π(a|x) (τC(a,x) − κ τY(a,x))] ≤ 0
```

引入 μ≥0 后，策略价值项为 (1+μκ)τY−μτC。除以正数后是 τY−λτC，其中 λ=μ/(1+μκ)。若另有总预算 B，还应加独立预算约束，不能把 CAC 与预算视为同一件事。无正增量时 CAC 比值无定义，不能仅依据线性化约束宣称策略有效。

用 cross-fitting 估计 outcome/cost nuisance models，并构造多 treatment DR 分数。对 Y：

```
ΓY_i(a) = mY_a(x_i)
          + 1(A_i=a)/e_a(x_i) · (Y_i − mY_a(x_i))

ΔΓY_i(a) = ΓY_i(a) − ΓY_i(0)
```

C 同理。RCT 的 e_a 优先取真实分流概率，可随分层/日期变化，不能默认各组均分。建议直接最大化 Σa πθ(a|x,λ)[ΔΓY(a)−λΔΓC(a)]，配合正则化、独立验证与约束校准。

这仍使用辅助效应模型，但最终策略自身输出动作，不必保留原有分桶＋组合 OR。推理时若保留 λ 校准或预算控制，要如实描述为“直接策略＋全局约束控制”。有限样本下期望约束不保证每个批次、每次实现都满足硬预算。

对于小动作空间，上述动作求和可以精确计算。无须为了借鉴 LLM 后训练而首先使用高方差采样式 GRPO。若未来扩展到大规模组合券或生成文案，再考虑生成式策略的必要性。

**可能形成研究贡献的地方：**曝光不完全执行＋增量 CAC 约束＋小样本 RCT＋跨期变化。单纯“网络输入 λ”并非新颖贡献。

### B. 行为预训练表征＋RCT 效应/策略微调：最符合当前数据形态的大模型路线

**研究问题：大量非实验行为数据能否改善稀疏实验数据的表征，而不把历史策略偏好当成因果结论？**

若可取得历史事件流，使用补贴、触达、领取、核销、完单及时间间隔等构建用户历史。用大规模历史数据做行为预训练，在 RCT 上训练效应 head 或 A 的策略 head。可参考 CURE 的迁移范式；这属于行为基础模型，不必自然语言化。

先比较：相同容量随机初始化、行为预训练、聚合特征 MLP。观察小样本 RCT、跨时间、新客群上的差异，再决定是否扩大模型。否则“预训练收益”容易与容量、样本数量或新特征收益混淆。

如果只有当前汇总 slot，可先用现有 embedding 训练 A；不要把 slot self-attention 误认为用户时间序列建模。

### C. 冻结 LLM 的语义表征插件：严格 LLM 路线的最小实验

**研究问题：语义先验是否能提升低样本 treatment/人群的效应估计？**

可参考 CURL。保留结构化数值分支，将决策前画像摘要和券规则送入冻结文本编码器，得到语义表示后接到 EFIN/效应 head。语义模块离线计算并缓存，先验证收益再考虑在线成本。

必须包含三组对照：原始结构化特征；人工构造的等价语义标签；LLM 表征。再加入“相同信息、去掉业务字段含义”或打乱语义的消融，判断收益来自先验还是简单特征重编码。

这里有一个项目特有风险：当前 libsvm 全局 feature id 和离散桶值未必带原始业务语义。需要稳定的字段字典、单位、时间窗口与决策前原始值；不能把 feature id 直接包装成自然语言后期待 LLM 理解。

只有几个固定券包时，结构化属性通常已经表达大部分规则；LLM 更可能在复杂适用范围、文案、用户文本信息上体现价值。新券“零样本预测”属于待验证泛化，不是因果识别保证；需要保留真实新券实验作评估。

### D. 历史条件化的长期策略：第二阶段框架升级

**研究问题：历史触达和补贴引起的疲劳、等待优惠等效应，是否使单次 7 天 uplift 策略长期次优？**

当重复决策数据具备后，策略输入改为 Ht 与约束状态。用序列编码器刻画历史，再以长期回报训练策略；SACO、GOAL 提供值得借鉴的方向。

“观测状态不满足马尔科夫性”并不排除强化学习，可以使用完整历史或 POMDP 的近似状态表示。Transformer 有助于压缩历史，但不会自动解决未观测混杂、动作覆盖不足、历史压缩不充分等问题。

尤其要避免把多个重叠的 7 天完单标签逐日求和当作长期 reward；同一订单可能被重复计入。应定义非重叠的逐期奖励、成熟的观测窗口和留存终点。

训练行为模型模仿历史发券动作，只能得到历史策略的近似。要优化它，需要可信的反事实策略价值信号、奖励模型或交互证据。不能把 next-action prediction 的准确率当作政策改善证据。

### E. 因果基础模型／跨实验元学习：研究性更强的备选

用 CausalPFN/CausalFM 做小规模基准，或把历史活动/城市/渠道实验视为任务进行跨实验预训练。研究重点放在“新实验少样本适配”，比直接替换当前全量 EUEN 更清晰。

先选单个 treatment 对 control，验证数据类型、上下文规模和效应校准，再推进多 treatment。仅有单一业务的一次大实验时，跨任务优势不易验证；也不建议第一步从零训练通用因果基础模型。

## 5. 曝光解耦与混杂需要单独澄清

现有 EEUEN 使用 exposure_prob×exposure_uplift。它是有用的结构假设，但模型乘法本身不会证明得到正确的因果分解。

应区分分配 A、实际触达 E、用户结果 Y。即使 A 随机，E 也可能受用户活跃度影响；只选触达成功者训练或评估可能破坏随机化。当前分配决策首先对齐 intention-to-treat 总效应。

如果要研究“触达之后券的效应”，需明确处理的是不依从、可达用户效应还是中介直接/间接效应。工具变量的排除限制、单调性或中介识别假设都需要额外论证。短信/电话自身可能影响出行行为，不能直接假设它们只通过领券产生作用。

因此，近期可以把曝光作为辅助预测任务，但用分配动作的 RCT 策略价值验收；不必先声称已识别曝光后的纯净效应。所有政策输入必须在决策时可用；本次发送成功、核销等只可作为后续反馈或辅助标签。

## 6. 最小实验矩阵与验收方式

| 实验 | 表征 | 训练目标 | 部署方式 | 回答的问题 |
|---|---|---|---|---|
| E0 | 当前 EUEN/EFIN 最佳版本 | 当前 outcome/cost loss | 原分桶＋OR | 基线 |
| E1 | 与 E0 对齐 | 决策感知 loss | 保留 OR | 一二阶段对齐的价值 |
| E2 | 小型静态网络 | DR 策略价值＋约束条件化 | 直接动作＋λ 校准 | 去掉原链路是否有收益 |
| E3 | E2＋预训练行为表征 | 同 E2 | 同 E2 | 预训练的额外价值 |
| E4 | E0 或 E2＋冻结 LLM 语义 | 与宿主完全相同 | 与宿主相同 | 严格 LLM 的额外价值 |
| E5 | 历史序列模型 | 长期约束策略目标 | 序列策略 | 长期决策是否优于单次策略 |

E0—E4 首轮必须用同一可选券集合；如果 OR 只有 3 券＋control，策略也先保持一致。E5 属于新 estimand，不能只拿它与 E0 的单次 AUUC 比较。

评估约定：

1. **拆开模型训练、策略/λ 选择、最终评估。** 现有文档中的 14 天 test 还被拆成 OR train/val 并用于 stage 选择；如果已据此挑过方案，需要新的 untouched 时间窗作最终检验。
2. **独立评估策略。** 使用留出 RCT 的 IPW/DR policy value，对增量 Y、增量 C 分别估计，再计算比值。不能只汇总“推荐券与实际券一致”的人群而忽略分流概率。
3. **锁定策略后再估计置信区间。** 重复出现的用户用用户聚类 bootstrap；跨日期漂移额外做时间块稳定性检查。训练评估按用户与时间合理隔离，并确保标签窗口成熟。
4. **策略主指标：**相同实际 CAC 下增量完单、目标 CAC 下约束违例、增量成本；有总预算时额外看预算违例。CAC 分母接近零时报告区间与不稳定性。
5. **辅助指标：**分 treatment AUUC、效果/成本校准、策略动作占比、有效样本量、多个随机种子、推理成本。不可只凭平均 AUUC 早停选择直接策略模型。
6. **新增动作和序列另行验收。** 旧 RCT 对未出现的新券没有直接支持；序列策略还需要逐步 propensity/覆盖条件或可信的序列评估，单步 DR 不能直接宣称验证完整长期策略。

建议的停走条件：E2 若无法在独立时间窗稳定超过 E0，不应靠扩大语言模型掩盖目标或评估问题；E4 如果仅在训练期或普通 outcome AUC 上提升，不能认定对因果决策有帮助；E3 若只有全量样本收益而少样本/跨期无改善，应检查是否只是模型容量收益。

## 7. 立项前只需补齐的关键信息

- 随机化单位是用户、用户日还是触达机会？分流概率能否按日期、渠道、实验版本恢复？
- 是否保留完整行为时间戳、历史动作、候选券集合、行为策略版本？能否获得 28 天等更长成熟标签？
- 现有筛选、join、曝光条件是否改变了入组总体？cost_7d 是否包含短信/外呼及全部补贴成本？control 是否有非零成本？
- 最终硬约束是增量 CAC、总预算，还是二者兼有？是否还有券型配额或城市约束？
- 是否具备有业务含义的原始字段字典，以及多个历史 RCT 任务或决策前文本？

这些信息分别决定 A、B、C、D、E 的实施边界，并不妨碍先用现有数据规范评估和构建 E2。

## 8. 建议对 MT 的表述

“近期已有两条不同的进展：一条利用语义或因果基础模型增强效应估计，另一条通过约束条件化策略直接输出补贴动作。结合当前 RCT、多券和 CAC 运筹链路，我们先验证直接策略学习的价值，再叠加行为预训练或 LLM 表征。第一阶段以相同实际 CAC 下的增量完单为验收标准；如果数据支持重复决策，再推进长期序列策略。拟研究的重点是曝光不完全执行与跨期变化下的增量价值和约束可靠性，而不是仅把模型换成 Transformer。”
