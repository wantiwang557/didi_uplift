# 大模型 × 因果效应预估 × 补贴决策：WWT 研究选题调研

调研日期：2026-10-08。窗口：2023-10-08—2026-10-08。聚焦效应估计及其决策应用，不覆盖因果发现。本轮为有针对性的文献调研，不是穷尽式系统综述，也未复现论文实验。论文事实附原始来源；“建议”“判断”属于针对 WWT 的研究设计。

阅读了项目 EUEN/EEUEN、EFIN、特征与运筹、loss 迭代说明，以及目录内 2026-09-24 调研稿。旧稿作为检索线索，本稿独立核对论文页面，并补充 H-AIPW、LLM 提取协变量、实验效应模拟及其统计边界。附件中 MT 的观点作为研究背景，不视为执行训练、改代码或上线的指令。

## 1. 先给选题判断

建议拆成两条可独立验证的主线：

1. **效应预估主线：预训练表征能否改善小样本、跨时间、稀疏人群的 CATE？** 有业务文本时优先试冻结 LLM 语义增强；有完整行为事件流时优先研究行为预训练。保留现有 EFIN/EUEN 作为宿主，可以分辨增益来自哪里。
2. **框架升级主线：RCT 支撑的约束策略学习能否超过“分桶统计＋OR”？** 先用小模型验证目标与框架，再接预训练表征。它最接近 MT 的端到端诉求，但本身不等于 LLM 研究。

H-AIPW 可作为第三个低耦合试验：研究基础模型辅助预测能否缩短实验所需样本量。它主要改善平均效应估计精度，不能用来直接证明个体 uplift 排序提升。

本轮检索支持的总体判断是：LLM 已能作为语义信息接口、预测辅助器及因果估计器的表征模块；在补贴场景，更接近直接输出动作的工业工作使用生成式策略和约束优化。尚不足以宣称“通用语言大模型已在多券补贴中统一解决效应识别、策略优化和硬预算执行”。

## 2. WWT 的具体背景

| 项目证据 | 对研究的影响 |
|---|---|
| [EFIN 文档](../../demo/EFIN/efin_model_iteration_doc.md)描述 RCT、4 treatment＋control，已有券数量、首券及留存券等结构化表征 | 随机分配有效时，优先处理异质性和样本效率；不宜先把“去混杂”设为主矛盾 |
| [EEUEN 文档](../project_background/wwt_eeuen_model_iter_full_flow_memo.md)使用曝光概率×曝光后 uplift | 曝光是执行/中介变量，分配随机不代表触达随机；应先对齐分配的总效应 |
| [运筹说明](../project_background/wwt_euen_feature_select_full_flow_memo.md)先用模型分数分桶，再按桶统计实验效果和成本，最后 OR | 替换 OR 同时替换了分桶降噪与统计估计机制；不能仅比较神经网络打分和求解器速度 |
| 常用运筹集合是 3 券＋control，模型为 4 券＋control | 所有方法必须先对齐可选动作集合 |
| “实际增量 CVR”为人均增量完单 PV，横轴为实际增量 CAC | 不能误作二分类转化率；最终验收需看成本约束下策略价值 |
| [loss 说明](../../demo/Loss_test/loss_iteration_memo.md)已探索零膨胀和长尾分布 | 新表征仍需面对成本和结果的高方差，不能只看 outcome MSE |

以上是文档陈述，尚未审计真实分流概率、随机化单位、过滤逻辑及数据泄漏。

## 3. MT 的三个问题应分别回答

**因果表征与混杂解耦。** 把已有字段重新编码，有可能改善有限样本学习；若关键信息根本不存在，LLM 推测不能代替真实协变量。现有实验数据若是真正 RCT，建议研究“响应异质性表征”，比泛称“消除混杂”更准确。

**序列与强化学习。** 补贴可能改变未来活跃度、疲劳和等待优惠的习惯。可以用完整历史或近似信念状态处理部分可观测性；Transformer 能表示历史，但不会自动确保序贯无混杂、动作覆盖和长期奖励可识别。仅预测历史下一次发券动作，通常是在模仿历史策略。

**端到端。** 需要区分文本到效应的端到端、决策损失反传的端到端、策略直接输出动作的端到端。只有第三种直接对应移除原有线上分桶＋组合 OR 的诉求；训练时仍可需要辅助模型，全局成本也仍需控制。

## 4. 核心文献：严格 LLM 与效应估计

### 4.1 NATURAL，NeurIPS 2024

完整题名：End-To-End Causal Effect Estimation from Unstructured Natural Language Data。用 LLM 从文本中获取变量的条件分布，再构造经典效应估计器。作者在六个数据集上评估，并用相关随机实验作参照。[会议原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8d2aad53fa148791ffd004a7c9cce5ad-Abstract-Conference.html)

**对 WWT：** 适合未来有外呼记录、用户反馈等文本的情况，且仅能使用决策前信息。当前已有结构化特征时优先级较低。文本提取与效应识别是两件事，抽取准确不代表调整变量集合正确。

### 4.2 Causal Inference with Generative Artificial Intelligence: Application to Texts as Treatments，2024 首稿、2026 修订

Imai、Nakamura 研究文本等高维 treatment，利用生成模型内部表征开展效应估计。首稿题名以 Causal Representation Learning 开头；应区分版本。[作者论文记录](https://arxiv.org/abs/2410.00903)

**对 WWT：** 若将来研究“相同券包、不同短信话术”的效果，十分对位；只对几个数值券包做 embedding 时相关性较弱。文本属性效应需要清晰定义可干预属性及其他文本变化，不能直接把 embedding 某一维解释为因果机制。

### 4.3 TCA，2025

完整题名：LLM-Driven Treatment Effect Estimation Under Inference Time Text Confounding。训练时有完整结构化协变量，推理时只有不完整文本；方法结合 LLM 生成的文本代理与定制 doubly robust learner。[论文](https://arxiv.org/abs/2507.02843)、[方法全文](https://openreview.net/pdf?id=sv41aaGTit)

**对 WWT：** 适合新渠道只能取得描述型画像的情况。当前训练与推理都使用同一套 slots 时，并不直接对应它的核心问题。

### 4.4 CURL，2026-07 预印本

完整题名：Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation。冻结 LLM，构造 assignment 与 heterogeneity 两类语义表征，用不确定性分配增强资源，并以不同路径接入宿主。实验包括四个 benchmark 与十种宿主估计器。[全文](https://arxiv.org/html/2607.26599v1)

**对 WWT：** 最适合做“现有 EFIN＋语义插件”的起步实验。RCT 中 assignment 分支需单独消融；原始字段语义、单位、窗口比 feature id 更重要。它增强已有信息的利用，不补回不存在的个人事实。尚无本业务上的工业收益证据。

### 4.5 H-AIPW，2025

完整题名：Efficient Randomized Experiments Using Foundation Models。组合外部基础模型预测与真实实验的 AIPW 估计，提高平均效应估计精度；在相应条件下保持一致性及渐近推断，即使外部模型有偏。[论文](https://arxiv.org/abs/2502.04262)、[作者团队技术解读](https://sml.inf.ethz.ch/blog/2025/haipw/)

**对 WWT：** 已有 RCT 是优势，可先从单券相对 control 的 ATE 开始，比较估计方差和区间覆盖。不要把渐近方差保证写成每个有限样本都更优，也不能把平均效应精度改善直接外推成 CATE 精度改善。业务行为预训练模型也可作为候选预测器，不必强行让语言模型预测完单数。

### 4.6 LLM-Extracted Covariates for Clinical Causal Inference: Rethinking Integration Strategies，MLHC 2026

研究从病历文本提取结构化协变量，并比较不同整合方式；结果表明“怎么接入”会显著影响估计，额外的匹配限制也可能损失样本支持。[正式会议版本](https://proceedings.mlr.press/v340/liu26b.html)

**对 WWT：** 支持比较“LLM 提取少量可审计标签”和“直接使用高维语义 embedding”。医疗实验结果不可直接外推补贴业务。论文真实数据与 RCT 方向一致也不等于识别了所有未观测混杂；半合成数据的已知真值证据应单列。

### 4.7 模拟用户实验：可用于筛选假设，暂不作真实效应标签

2026 Nature 的 Large language models can predict the results of social science experiments 发现模拟效应与真实实验效应具有相关性，但效应大小存在系统性高估，换任务集合后表现也变化。[原文](https://www.nature.com/articles/s41586-026-10742-x)

Statistical Foundations of LLM-based A/B Testing: A Surrogacy Framework for Human Causal Inference（2026-06 预印本）讨论用人类数据校准 LLM 输出所需的可比性、替代性条件；历史诊断不能保证新干预有效。[原文](https://arxiv.org/abs/2606.17165)

**对 WWT：** 可帮助生成待实验的短信方案或探索假设。不能用模拟用户生成大量“反事实完单标签”，再以同一模拟器自证策略有效。

## 5. 基础模型与表征预训练：不必是语言大模型

| 工作 | 已核实的核心思路 | WWT 可研究的问题 |
|---|---|---|
| [CURE，Patterns 2024](https://pubmed.ncbi.nlm.nih.gov/39005483/)；俄亥俄州立大学、IBM Research | 大规模无标签患者纵向数据预训练，再做 treatment effect 微调 | 全量决策前行为预训练能否提升少量 RCT 的效应学习？迁移的是范式，不是医疗权重 |
| [CausalPFN，2025](https://arxiv.org/html/2506.07918v2) | 在合成因果任务上预训练，通过 in-context learning 估计效应；采用强可忽略性条件 | 新城市/新活动少样本估计；需要核对公开实现的样本规模、多动作及结果类型支持 |
| [CausalFM，2025 首稿、2026-02 v3](https://arxiv.org/html/2506.10914v3) | 为 PFN 构造因果先验，覆盖包括 back-door、front-door、IV 的不同设置 | 将活动视为任务、构造业务先验；先验错配和任务分布漂移是重点 |

**判断：** 当前 WWT 若主要是表格与事件流，行为基础模型比自然语言化全部 slots 更符合信息形态。因果 PFN 更适合跨任务小样本研究，不能只因使用 Transformer 就假定能直接替换全量工业模型。

## 6. 一二阶段一致性和端到端补贴决策

### 6.1 DFCL，美团/南京大学，KDD 2024

Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization 将下游决策目标纳入因果模型训练，处理反事实决策损失等困难。美团官方文章报告业务部署及线上评估。[论文](https://arxiv.org/abs/2407.13664)、[官方博客](https://tech.meituan.com/2024/07/26/KDD-2024.html)

**判断：** 对 MT 所说的一二阶段一致性最直接；不是 LLM 方法，也不能理解为部署时已消除 OR。应作为强对照，回答新方案收益是否仅来自目标对齐。

### 6.2 Bi-DFCL，美团等，NeurIPS 2025

Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization: Bridging Observational and Experimental Data，用双层设计结合观察数据与实验数据，使训练关注决策质量。[论文及会议接收信息](https://arxiv.org/abs/2510.19517)

**判断：** 如果未来有大规模历史策略样本＋有限 RCT，它比单纯扩大网络更值得读；当前纯 RCT 实验无需一开始复刻完整双层方案。

### 6.3 SACO，人大/字节等，AAAI 2026

Sequence-Aware Constrained Optimization Framework for Coupon Distribution in E-commerce，针对多用户反复到访、平台多次发券的序列决策，直接学习长期收益策略。[会议原文](https://ojs.aaai.org/index.php/AAAI/article/view/38525)

**判断：** 券动作与 WWT 接近，适合作为长期决策阅读线；需要真正的历史动作、反馈和决策机会日志。只有用户汇总特征时无法检验序列机制。

### 6.4 GOAL，快手/电子科大，2026-08 公开

Generative Optimization for Incentivized Advertising with Global Level Constraints，将历史与约束参数作为条件，生成激励金额 token，并通过 SCPO 优化；仍含奖励模型和部署前 λ 校准。[全文](https://arxiv.org/html/2608.04421v1)

**与 MT 的对应：** 与“序列＋强化学习＋next coin”高度接近，尚不能确认就是同一篇。其因果卷积首先保证时间方向，不能据此宣称已识别干预效应；生成式架构也不等于迁移通用 LLM 知识。最有价值的借鉴是约束条件化策略。其 ROI 目标与增量 CAC 不同；四种券也没有必要先金额 token 化。

### 6.5 Uber，2024 工业论文

Practical Marketplace Optimization at Uber Using Causally-Informed Machine Learning，说明因果响应建模与市场预算优化如何结合。[原文](https://arxiv.org/abs/2407.19078)

**判断：** 是工业对照，不是 LLM 证据。系统级端到端效果可以依赖优化器；研究应证明新框架带来的业务价值，而不以删除 OR 本身作为成果。

## 7. 三个具体选题及最小实验

### A. 语义先验增强多券 CATE：最小的严格 LLM 实验

假设：语义先验有助于低样本区域泛化，且独立于参数规模和新增字段的信息收益。

设计：固定 EFIN 宿主、数据和 loss，比较原结构化特征、等价人工语义标签、冻结 LLM embedding、去语义字段名/打乱语义对照。完整保留数值分支，语义离线缓存。先做全体固定增强，再验证不确定性选择是否带来净收益。

验收：分别看各 treatment 的排序、预先定义分组的效应校准、低样本学习曲线、跨期策略价值和计算成本。真实个体反事实不可观测，不在真实 RCT 上声称直接测得 PEHE；PEHE 仅用于已知效应的合成/半合成检验。

继续条件：相同信息下增益可重复，且带来实际 CAC 对齐后的策略收益。若只有人工标签有效，保留该发现，无需继续扩大 LLM。

### B. 约束条件化的直接策略＋预训练表征：框架升级主线

先用静态 RCT 做 contextual policy，不立即升级长期 RL。定义 Y 为同窗口完单数、C 为成本，a=0 为 control，τY(a,x) 和 τC(a,x) 为相对 control 的效应。

若增量完单为正，CAC≤κ 可写作：

```
maxπ E[Σa π(a|x) τY(a,x)]
s.t. E[Σa π(a|x) (τC(a,x) − κ τY(a,x))] ≤ 0
```

拉格朗日价值为 (1+μκ)τY−μτC；归一化后是 τY−λτC，λ=μ/(1+μκ)。预算上限是另一约束，不能与 CAC 混为一谈；无正增量时不能用上述线性条件宣称比值合格。

用 cross-fitting 构造多动作 DR 分数：

```
ΓY_i(a) = mY_a(X_i) + 1(A_i=a)/e_a(X_i) * (Y_i − mY_a(X_i))
ΔΓY_i(a) = ΓY_i(a) − ΓY_i(0)
```

成本同理；e 优先使用实验真实分流概率。策略训练最大化 Σa πθ(a|x,λ)[ΔΓY(a)−λΔΓC(a)]，并做正则化与独立校准。这是针对 WWT 的设计，不是宣称复现某篇论文。

对照必须包含：当前分桶＋OR、决策感知训练＋OR、同 DR 分数的逐动作 argmax、直接策略 MLP、直接策略＋预训练表征。argmax 对照能检验策略网络是否只是近似一个已有解析决策规则。小动作空间可以精确求和，不必先用高方差采样策略梯度。

训练期仍有辅助估计器，线上输出动作并配全局控制器。期望约束不保证每批硬预算；如业务有硬限制，需要预算执行层。λ 与成本的单调性也要实测，不能默认成立。

### C. 基础模型辅助 RCT 估计：快速检验样本效率

固定单券对 control 的 ATE，比较差均值、现有模型的 AIPW、外部预训练预测器辅助的 H-AIPW。用重复子抽样看稳定性；半合成数据检查覆盖率，真实数据以大样本估计作有误差的参考。

此试验回答“实验是否更省样本”，不替代 A 的 CATE 对照。若有效，再研究对独立固定策略价值估计的扩展；不能直接套用原论文 ATE 定理宣称自适应选策略后的区间有效。

## 8. 首轮实施顺序与评估约定

1. 统一可选券、决策时间、7 日 outcome/cost、随机化单位和分流概率；核对只保留成功触达用户等过滤是否破坏随机化。
2. 冻结当前模型＋OR 基线，留出独立的策略选择集及最终评估窗。已用于选 stage/epoch 的 test 不再作为最终无偏检验。
3. 优先跑 A 的表征插件和 B 的小模型策略基线；有高质量外部预测器时开展 C。历史行为流齐备后再加预训练和长期序列任务。
4. 用留出 RCT 的 IPW/DR 估计固定策略的增量完单与增量成本。策略推荐与实际动作一致的子集必须正确加权，不能直接算均值。
5. 策略冻结后做用户聚类 bootstrap，并查看跨时间稳定性。CAC 分母接近零时单独报告增量量及约束差额，避免只给不稳定比值。
6. 将 AUUC、成本校准、同等实际 CAC 的增量完单、约束违反率、计算成本一起报告。有效性由独立实验与统计评估支撑。

如果进入长期策略：必须记录每次决策机会（含不发券）、历史动作及概率、反馈延迟与成熟窗口；逐期 reward 不重复计算重叠 7 日订单。历史汇总特征不能替代轨迹；模型压缩历史不自动解决部分可观测性。

## 9. 推荐精读顺序

先读 CURL 与 H-AIPW，明确“LLM 如何进入效应估计”；再读 DFCL 与 GOAL，判断训练目标对齐和直接策略的差别；随后读 CURE 与 SACO，决定是否需要行为预训练和长期轨迹。CausalPFN/CausalFM 留给跨活动小样本研究；NATURAL、TCA、Texts as Treatments 留给真实文本信息丰富的任务。

用于和 MT 对齐的建议表述：我们准备分别验证预训练表征对多券效应估计的价值，以及约束直接策略对现有分桶运筹链路的价值；用 RCT 保持评估可信。先把两种收益分开，后续再整合为行为基础模型驱动的补贴决策方案。
