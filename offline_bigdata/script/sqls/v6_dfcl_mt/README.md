# v6_dfcl_mt：按 MT 代表行口径构造 EUEN / DFCL 候选样本

更新：2026-10-23。交付为可在 Spark SQL 中查询的三份完整特征 SQL；尚未连接集群，未产出实际样本或 TFRecord。

## 当前最终保留文件

正式取数使用同一版本的三份 split SQL：

- `feature_v6_dfcl_mt_train.sql`
- `feature_v6_dfcl_mt_valid.sql`
- `feature_v6_dfcl_mt_test.sql`

统计成功周期内“多条正记录”的规模和占比，使用：

- `audit_multi_positive_success_summary.sql`

旧版本及辅助文件的保留/归档建议见 `SQL_INVENTORY.md`。这里的“最终”表示当前已确认口径的 SQL 版本；当前状态仍为 `sql_generated_not_executed`，需要在 Spark/Hive 跑数并完成样本验收后才能标记为生产验收完成。

## 本次采用的口径

用户明确要求：成功候选始终保留触达成功日；只有未成功/control 连续段才是有归因完单保留正记录、全负保留最后一条。连续未成功负记录只保留一条，不应每隔 7 天再留一条。

因此，**未成功用户和 control 不采用“首次入组日起每 7 天固定分块”**。本版本区分成功窗口与连续未成功段；“连续段”可能长于 7 天，不将它描述成 7 天真实发券周期。

1. 成功窗口：沿用已有触达版字段解释，标签表 `send_success_cnt > 0 OR outcall_success_cnt > 0` 表示当天成功；成功日到 +6 个自然日作为候选窗口。现有字段没有提供时分秒，本版不是精确 168 小时计算。
2. 未处于前一成功窗口、但距离下一成功日不超过 6 天的失败记录，沿用已有触达版的近似规则，并入下一成功候选，只用于消除成功前的失败重试。它们不属于“成功后固定7天”的观察窗口，其正标签不会进入该成功周期的多正统计，也不会被选为成功周期代表。
3. 其余未成功记录/control 按日期形成连续段，**不会因为段持续超过 7 天就拆段**。遇到成功候选，或者相邻入组日期间隔至少 28 天，开始另一段。28 天只用于识别完单出组后的重新入组，不是激励周期长度；频控导致的长间隔仍需在审计中核对。
4. 成功候选无条件保留触达成功当天；周期内其他正记录只进入多正审计，不能替代成功锚点。只有未成功/control 连续段才执行“有正保留正、全负保留末次入组”；多正时按日期较晚选择，`finish_cnt_7d` 仅作为同日期冲突时的确定性兜底。
5. 代表行的特征、label、触达标签、成本一起保留。label 保留订单数，不改为 0/1；UV 指的是一名用户在同一候选中只有一条记录，不是改成 UV 完单标签。
6. 不执行旧 v6 的跨整个 split 的 `finish_all_cnt` 筛选。一个用户在早期候选全负、较晚另一候选有正时，两条都可以保留。

## 给 MT 确认的简版口径

样本粒度是“用户 × 候选周期”，每个候选最终只留一行：

- 实验组触达成功：成功日作为锚点，建立成功日起固定 7 天候选；最终固定保留成功日。
- 实验组未触达成功：连续入组形成一个未成功段；段内有正记录选一条正记录，全部为负选末次入组。
- 对照组：没有成功锚点，按连续在组段处理；段内有正记录选一条正记录，全部为负选末次入组。
- 成功前 6 天内的失败触达记录并入后续成功候选，只作为重试审计，不单独生成负样本。
- 相邻入组间隔达到 28 天时，未成功实验组或对照组开启新段。28 天是重新入组边界，不是 7 天激励周期。

每行样本提供：代表日特征 `X`、实际组别 `T`、7天完单数 `Y=finish_cnt_7d`、7天成本 `C=cost_7d`、触达标记 `exp_label` 和代表日期 `label_dt`。

EUEN 使用方式：以 `X、T、Y` 训练各 treatment 相对 control 的响应/uplift；如需正负采样，只能在这份候选样本生成之后对 train 做，并记录采样率或样本权重，valid/test 保持原分布。`exp_label` 是动作执行后的结果，只能用于掩码或分析，不能作为线上决策前特征。

DFCL 使用方式：使用未做 outcome 采样的完整候选样本，将实际 treatment、对应分流概率 `p(T)`、实际 `Y` 和 `C` 输入 DFCL loss；control 成本为 0。已计算的 propensity 应按 treatment 查表，不能把触达成功率当作随机分流概率。

需要 MT 明确确认的关键边界：

1. 成功组按“成功日固定7天”，但未成功组/control 按“整段选代表”，三者的候选起点并不完全对称，是否符合希望优化的线上决策时点。
2. 未成功组/control 的“有正选正、全负选末次”使用了结果变量。它可作为 EUEN 的业务采样口径，但若 DFCL loss 依赖随机实验/IPW 无偏性，是否允许这种 outcome-dependent selection；如不允许，应改成不看未来标签的事前锚点规则。
3. 成功锚点本身 `finish_cnt_7d=0`、但同一7天窗口其他行为正时，当前仍保留成功锚点的 0，其他正记录只审计；是否应改用周期聚合标签，需要 MT 确认。
4. 同一用户可在不同候选周期重复出现；年度固定分组使 treatment 不变。这些行不是相互独立的新随机分流，应确认 DFCL 是否需要按用户聚类、去重或设置周期权重。
5. 代表日特征必须是该动作发生前可获得的特征；若成功当天特征包含触达后或完单后的信息，会造成训练泄漏，需要核查特征生产时点。

示例：

- 7/1～7/8 每天未成功、全部无完单：连续段只有一条代表，选 7/8。
- 7/1～7/20 每天未成功、全部无完单：只选 7/20，不选 7/7、7/14。
- 7/1 入组、7/2 失败、7/3 触达成功且归因 1 单、7/5 无成功触达但又有归因 1 单：成功周期为 7/3～7/9，统计为一个含两条正记录的周期；按优先级保留触达成功的 7/3。即使 7/5 的 `finish_cnt_7d` 更大，也不覆盖 7/3。
- 同一成功候选只保留真实成功锚点；成功周期内其他非成功正记录不能替代成功锚点。
- 7/16、7/17 未成功，7/18 成功且归因 2 单，7/20 为 0：同一成功候选选 7/18，直接保留其 2 单和该行成本。
- 7/16 成功，7/19 又有一条零完单记录，全期为 0：保留成功锚点 7/16，label=0；7/19 不再单独产生样本。
- 7/1 成功且 7/1～7/7 无完单，7/9 再入组且后来有归因完单：保留 7/1 的成功周期负样本，同时保留 7/9 所在下一候选的正样本；两条属于不同激励周期。
- 7/16、7/17 未成功，7/24 再出现：相邻间隔只有 7 天，仍属于同一未成功连续段，只选 7/24。
- 7/16、7/17 未成功，8/14 再出现：相邻间隔达到 28 天，按重新入组边界分成两段，分别选 7/17、8/14。

上游已经按“外呼成功 > 短信成功 > 末次入组”做订单归因；这里不重新归因、不累加多条 `finish_cnt_7d`。成功候选固定选择成功锚点；未成功/control 连续段才从正记录中选择唯一代表。

## 文件与日期

- `feature_v6_dfcl_mt_train.sql`：代表行日期 2026-07-16～2026-08-21。
- `feature_v6_dfcl_mt_valid.sql`：代表行日期 2026-08-22～2026-08-27。
- `feature_v6_dfcl_mt_test.sql`：代表行日期 2026-08-28～2026-09-10。
- `audit_samples.sql`：按候选输出所选记录、周期信息、异常和最终是否进入样本，供真实数据验收。
- `audit_multi_positive_success_summary.sql`：按 split、treatment 汇总成功周期数、多正周期数、用户数和占比；分母同时提供原始候选和合格候选两种口径。
- `audit_multi_positive_success_details.sql`：列出所有多正成功周期的逐条入组记录、label、成本和最终选择行，便于抽查 7/3、7/5 这类 case。
- `audit_zero_success_then_next_positive.sql`：列出“前一成功周期选中0标签、下一候选为正”的用户，用于核对 7/1 与7/9应同时保留的 case。
- `selection_ctes.sql`：共同选样模板；含参数，不能作为完整查询直接执行。
- `build_profile.py`：生成上述四份完整 SQL；从原 v6 提取投影，保留列定义与顺序。
- `test_profile.py`：本地合成案例测试。
- `profile_manifest.json`：日期、口径参数、源 SQL 和产物哈希，状态为 `sql_generated_not_executed`。

三份 SQL 严格使用原 v6 的 split 日期：train 为 2026-07-16～2026-08-21，valid 为 2026-08-22～2026-08-27，test 为 2026-08-28～2026-09-10；标签表和特征表使用相同范围，不读取 valid/test 来决定 train 样本。

因此三个 split 独立构造候选。跨 split 持续在组的用户可能在相邻 split 各出现一次，这是保持原 v6 日期隔离的代价；不能同时宣称“完全不跨 split 重复”和“每份 SQL 只读取本 split”。如需评估边界影响，应使用覆盖 train～test 的审计 SQL 专门统计，不应把未来记录输入 train SQL。

原 v6 排除日期保持在进入选样前直接删除：3/26、3/27、4/8、8/2、8/8、8/9。它们不会参与候选识别。

## 与 v6 对齐的内容

特征表、标签表和连接键沿用 v6：

- `dm.ut_alg_flowback_msg_outcall_feature_train_di`
- `dm.ut_alg_flowback_msg_outcall_label_v2_di`
- `pid + dt + flowtree_group_name + group_name_eve`

输出为原 v6 的 **473 个特征 + 15 个 token + 1 个 label = 489 列**。保留五块：sirius画像、补贴环境、双端、有端、券敏感。

token 顺序不变：

```text
fin_order_wxmini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,exp_label,cost_7d,fake_treatment_id,flag,label_dt
```

四种券映射为 1/2/3/4，control 为 0；control 成本按之前确认的口径强制为 0。其他组使用代表行原 v6 的七项成本公式。**这不等于汇总全部失败重试成本**。

`exp_label` 是候选周期级曝光标记：成功候选固定为 1，未成功候选为 0。成功候选始终选择成功锚点，因此特征、label、成本和触达标记全部来自成功当天；成功窗口内其他正记录只用于审计。

## 数据质量处理

- 业务已确认：同一用户同一天只有一条入组记录，且用户不会跨 treatment。本 SQL 直接采用这两个业务不变量，不再计算同日变体数、多 treatment 标记，也不再据此整用户剔除。
- 业务已确认：一次触达成功后的固定 7 天内不会再次触达成功；下一次触达最早从上一轮 7 天结束后开始。因此不再计算成功窗口重叠标记，也不据此剔除样本。
- `daily` 中仍按完整入组键聚合一次，仅用于保证查询对完全重复源行幂等；不会生成冲突标记或拉黑用户。若上游业务口径以后改变，应单独做源表质量审计，而不是在训练 SQL 中静默删除整位用户。
- 某候选有 null/负 label：该候选标记 `invalid_label`，不把缺失当 0。
- 最终代表没有特征或特征键重复：不输出，也不自动换一条代表。审计分别显示 `missing_selected_features` / `duplicate_selected_features`。

上述候选和特征质量处理可能改变样本分布，必须看审计统计；不能静默视作天然完整 RCT。

## 在集群执行

先将本目录同步到集群。在已有 `spark` 会话的 Jupyter 中可以运行：

```python
from pathlib import Path
base = Path('/你的仓库路径/offline_bigdata/script/sqls/v6_dfcl_mt')
audit = spark.sql((base / 'audit_samples.sql').read_text())
audit.groupBy('output_status', 'candidate_kind').count().show(100, truncate=False)
audit.groupBy('selected_split', 'treatment_id').count().show(100, truncate=False)
audit.filter('positive_member_cnt > 1 OR observation_overlap_flag = 1 OR label_anchor_mismatch_flag = 1').show(50, truncate=False)

multi_summary = spark.sql((base / 'audit_multi_positive_success_summary.sql').read_text())
multi_summary.show(100, truncate=False)

multi_details = spark.sql((base / 'audit_multi_positive_success_details.sql').read_text())
multi_details.show(100, truncate=False)

zero_then_positive = spark.sql((base / 'audit_zero_success_then_next_positive.sql').read_text())
zero_then_positive.show(100, truncate=False)

train = spark.sql((base / 'feature_v6_dfcl_mt_train.sql').read_text())
valid = spark.sql((base / 'feature_v6_dfcl_mt_valid.sql').read_text())
test = spark.sql((base / 'feature_v6_dfcl_mt_test.sql').read_text())
# 真实样本与路径仅在你执行并物化后产生；本地未创建 Hive/HDFS 结果。
```

验收至少确认：每个候选唯一代表、实际结果键唯一、样本数与审计 `output` 数一致、各组正负比例、长负段跨度及跨 split/区间外代表数量、多正候选及其占比、成功负周期之后的下一正候选、标签锚点不一致、观察窗重叠、空/负标签和缺失/重复特征数量。候选唯一并不等于所有样本观察窗完全不重叠。

标签水位必须保证每个 split 范围内的 `finish_cnt_7d` 已完整产出；`right_boundary_flag` 表示入组记录在查询右边界可能仍在延续，不表示 label 已成熟。

如后续转 TFRecord，应从新版 train 重新拟合 feature/parser 配置，valid/test 复用该份 train 配置；token_size 为 15。不要直接运行原 skill 的默认 `euen_v6` build 来代替本目录，它目前仍使用原有 `finish_all_cnt` 规则，尚未接入本 profile。

## 本地验证与方法边界

```bash
python3 offline_bigdata/script/sqls/v6_dfcl_mt/build_profile.py
python3 -m unittest discover -s offline_bigdata/script/sqls/v6_dfcl_mt -p 'test_*.py' -v
```

测试执行完整投影和选样查询，覆盖连续 8/20 天负段、成功全负固定取成功锚点、成功周期内其他正记录不能替代锚点、未成功段有正取正/全负取末次、成功前正标签不混入成功周期、成功负周期后下一候选为正、7 天边界、两个成功周期、跨 split、排除日期、完全重复源行、空标签、缺失/重复特征和输出 schema。使用 SQLite + 日期函数模拟；不等同于 Spark/Hive 编译和真实数据分布验收。

本次交付实现取样规则，不宣称“UV去重后DFCL无偏”。按结果选择特征日期、固定年组下后续入组的选择效应仍存在。SQL 不写入未经核实的固定 propensity；真实实验分配概率须另外核对。连续段保留一条也不代表每用户全历史永远只有一条，不同候选仍可各有一条。
