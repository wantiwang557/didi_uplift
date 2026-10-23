# 基于当天触达结果的 v6 样本 case 草稿

更新：2026-10-22。本目录取代旧 `v6_dfcl` 的日期间隔聚类方案，用于查 case，尚未用于集群训练。

## 已确认的口径

- 标签表的 `send_success_cnt`、`outcall_success_cnt` 是当天对应入组的触达结果，不在后续日期沿用历史成功状态。任一个大于0，记 `touch_success=1`。
- 成功发券：从发券成功日开始观察7天。本草稿以当天成功字段对应的 `dt` 为成功日。
- 未成功：从按约定选中的代表入组日期开始观察7天。
- `finish_cnt_7d` 已由上游按订单向前寻找入组记录、按渠道优先级归因；本 SQL 不重做订单归因，不移动或累加现有 label。
- 用户已接受：没有成功锚点可关联的失败用户/control，先从完整取数范围内首次入组日开始，每7个自然日分一个去重块。分块不是实际发券批次，也不是最终 label 的观察起点。
- 保留所有时段的候选轮次，不因用户后来有正标签删除此前的负轮次。

## 文件与运行

- `sample_cases.sql`：完整 Spark SQL。
- `sample_cases_query.py`：同一 SQL 已放入 `query = r"""..."""`，整个文件可复制到 Jupyter 的一个单元格运行；要求已有 `spark` 会话。
- `test_sample_cases.py`：模拟数据行为测试，用 SQLite 窗口函数及日期 UDF 验证算法；不是 Spark/Hive 执行验收。

默认只抽取每种 case 类型5个候选组，并展示这些组的全部明细行。先选组再取明细，避免 `limit 500` 截断同一个用户的案例。

SQL 最前面的参数：

- `history_start=2026-05-11`、`history_end=2026-09-16`：三个 split 共同读取的完整历史范围，前后上下文用于候选关联。改变历史起点会影响失败/control分块，正式任务必须冻结它。
- `sample_start=2026-07-16`、`sample_end=2026-09-10`：按代表记录日期输出的 v6 范围。
- `label_available_through=NULL`：最后一个已完整产出归因标签的入组日期；目前未知，输出 `label_maturity_status=unknown`，不自动假设标签成熟。
- `examples_per_type=5`：每类抽查数量；设0输出全部候选明细，可能非常大，不建议直接全量 `toPandas()`。

## 候选组如何构造

对每个用户按日期排序，只有 treatment 非0的成功记录作为成功锚点：

1. 当天成功：归到当天的成功锚点。
2. 未成功，但处于之前最近一次成功日到成功日+6天之内：归到该成功锚点。
3. 不在前一成功窗口内，但未来6天内有成功记录：近似视为下一次成功前的失败重试，归到最近的下一成功锚点。
4. 其余失败记录及control：归到从首次观测日开始的固定7天去重块。

第3步是为了让“1号失败、2号失败、3号成功”成为一个候选组的工程近似；并非已验证的业务任务ID。不能证明更早失败记录一定属于哪个成功轮次。相邻成功日相差不足7天会标记 `overlapping_success_flag`，不悄悄合并。

第4步是在已明确接受的固定7天分块上操作；一部分记录被成功组吸收后，原块中剩余记录仍可能成为独立的候选组。它们的观察期可能与其他组重叠，SQL 会标记 `observation_overlap_flag`。因此这些分块不能直接宣称为真实独立激励轮次。

## 每个候选组怎么选择代表行

- 有正归因行：优先正行；多正时取最大 `finish_cnt_7d`，同值优先成功锚点，再取较晚日期。多正行数量同时展示，即使正标签数值完全相同也会被发现。
- 没有正归因行，但存在成功锚点：保留成功日，即使之后还有0标签的无动作入组行。
- 全程未成功、全部为0：保留该候选组最后一条。
- NULL或负label不被悄悄当成有效0标签，会标记 `candidate_bad_label_flag`。

本阶段遵循用户与MT讨论的代表记录口径，特征连接键将是所选原始行的 `pid + dt + flowtree_group_name + group_name_eve`；尚未在本 SQL 中连接473个特征。这个按结果选代表行的方案不自动满足因果识别条件，不应只凭去重宣称DFCL损失无偏。

## 两个不同的日期必须区分

- `dedup_block_or_success_start_dt`：固定去重块起点，或成功锚点。用于给记录分组。
- `selected_dt`：最终保留的原始入组记录日期。其 `finish_cnt_7d` 仍保留原含义。
- `observation_start_dt`：成功候选取成功日；未成功候选取 `selected_dt`。
- `observation_end_dt_assumed`：起点+6天，仅作为当前自然日边界假设展示；上游是否含第7天/按168小时仍需核对。

例：失败用户1～7号全部为0，固定块起点为1号，选择7号。观察起点为7号，展示结束日为13号；不会把原始7号标签重新解释为1～7号结果。

例：1、2号失败，3号成功且归因label=1。选择3号，观察起点3号。

例：3号成功但label=0，8号另有归因正记录且被映射到同一个成功窗口。SQL会选择该正记录供展示，但标记 `label_anchor_mismatch_flag=1`；不能把8号label冒充从3号开始的结果，该类需要查真实归因和轮次。

## 输出如何看

先看 `case_type`、`pid`、`entry_dt`、两个成功次数、`finish_cnt_7d`、`would_keep_flag`。

- `would_keep_flag=1`：本候选组拟保留的代表记录。
- `assignment_reason`：该条记录为什么归到此组。
- `selection_reason`：为什么选择这条代表记录。
- `source_row_cnt`：同一原始键有几条源记录。所用字段完全相同才可在本审计中折叠。
- `source_variant_cnt>1` 或 `user_source_conflict_flag=1`：所用字段发生冲突；用于展示的max聚合不等于冲突已解决。
- `user_treatment_cnt>1`：用户跨组异常。
- `positive_member_cnt>1`：同候选组有多条正行。
- `label_anchor_mismatch_flag=1`：所选正记录日期不同于成功窗口起点。
- `observation_overlap_flag=1`：当前代表窗口与该用户较早候选窗口重叠。
- `selected_excluded_date_flag=1`：所选日期是原v6排除的8月2/8/9日。本阶段保留展示，不能直接用于训练。
- `crosses_split_boundary_flag=1`：同一候选组的明细跨越原v6 split边界。
- `sample_status=audit_only`：所有结果都是审计候选，不是已验收训练样本。

train/valid/test 仍按代表记录日期分别标记为7月16日～8月21日、8月22日～27日、8月28日～9月10日。分组先在共同历史范围内完成，不在每个split内重置。

## 成本与概率

`selected_row_cost_7d` 仍是所选原始行的原v6成本公式；control强制0。这个值不声称包括前面所有失败重试成本。不把重复记录的现有cost直接相加，核销和触达成本归属需另行核对。

`propensity` 按已接受的常量映射：0=0.4916，1=0.1778，2=0.1185，3=0.0935，4=0.1186。不会使用 `touch_success=1` 过滤整体样本。

## 本地验证

```bash
python3 -m unittest discover -s offline_bigdata/script/sqls/v6_dfcl_touch_audit -p 'test_*.py' -v
```

场景覆盖：失败重试后成功、成功零完单、持续失败/control、失败自然完单、前负后正、成功窗口重叠、同值多正、标签锚点冲突、源表重复/冲突、空label、跨组、7天边界、split边界与排除日期、整组抽样。

尚未连接集群运行，也未改正式TFRecord配置或训练流程。case审核后才能决定真实轮次来源、边界、跨split处理及成本汇总口径，再产出正式三份特征SQL。
