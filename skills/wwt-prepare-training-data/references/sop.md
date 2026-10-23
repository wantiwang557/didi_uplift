# 从零学习并使用这份 Skill

这份 skill 把三个层次分开：你用自然语言描述需求；SKILL.md 告诉 Codex 如何解析和选择资料；Python 程序生成、执行并校验任务。理解这三层，就能逐步独立维护它。

## 1. 先认识已经实现的文件

| 文件 | 你应该在什么时候看 |
|---|---|
| ../SKILL.md | 理解 skill 如何接收任务、调用脚本、汇报状态 |
| data-contract.md | 核实样本、label、cost、token 和时间口径 |
| feature-catalog.json | 按 SQL 注释名选择或新增特征块，追查直接来源 |
| date-exclusions.json | 查询和登记不可用日期 |
| ../assets/base.sql | 查看 v6 主样本、标签关联及总体筛选 |
| ../assets/blocks/ | 查看特征块各自的 JOIN 模板 |
| ../scripts/dataset.py | 统一输入、SQL 生成、任务记录和顺序执行 |
| ../scripts/fit_config.py | 仅用 train 快照生成分箱；隔离旧的共享缓存 |
| ../scripts/cluster_worker.py | 集群样本快照、抽取、验收 |
| ../tests/test_dataset.py | 查看容易出错的规则如何被测试 |

第一版沿用 TFRecord/HDFS 交付，同时保留 HDFS Parquet 样本快照。尚未实现 Hive 表注册和 NFS 自动下载。没有依赖 agent SDK、向量库或额外模型服务；由 Codex 调用这些文件即可。

## 2. 第一次只生成计划

在仓库根目录使用下面的命令。这里的日期来自 v6 示例，只用于学习；生产任务请先修改请求副本。

```bash
python3 skills/wwt-prepare-training-data/scripts/dataset.py catalog
cp skills/wwt-prepare-training-data/assets/examples/v6-request.json /tmp/wwt-request.json
python3 skills/wwt-prepare-training-data/scripts/dataset.py build \
  --request /tmp/wwt-request.json \
  --out /tmp/wwt-v6-demo \
  --repo "$PWD"
```

--out 必须是不存在的新目录。不要覆盖历史任务。JSON 是第一版的实际格式，方便在没有 PyYAML 的集群上运行；你仍然只需自然语言描述，由 Codex 生成 JSON。

建议先看三个结果：

1. resolved_request.json：名称解析后的特征、字段、实际日期、排除原因。
2. sql/train.sql、valid.sql、test.sql：字段、JOIN、日期与 flag。
3. manifest.json：此时 status=planned，表示尚未执行。

SQL、代码和定义已经冻结到该目录。后续即使仓库原文件变化，本次执行仍使用冻结版本。修改请求需要重新生成新目录。

## 3. 接通集群后执行

需要 Linux 集群客户端有 Spark、Hive、HDFS 凭据，以及可导入 pyspark/findspark/toad/dmdeepl 的 Python。旧 handler 内仍有 dmdeepl 的 NFS 查找路径，需在目标环境验证。抽取二进制需要与执行节点兼容。

把 assets/examples/cluster-environment.json 复制为独立环境配置，并核实：

- hdfs_root 为当前身份允许写入的根目录；每个任务写入独立 dataset_id 子目录。
- queue、archives、jars、python_executor 来自现有仓库 common_spark32.sh，属于待验证默认值。
- python 为集群可用 Python 3.7+ 命令；driver/executor 需使用兼容的 Python 版本。
- label_available_through 必须填真实完整产出水位。示例为 null，故意不提供未经验证的日期。
- user_name 用于记录预期身份，不会自动切换账号或授予权限。

将整个任务目录复制到服务器，包括 runtime、definition、sql 等。根据你的真实连接方式操作，不从示例推断 SSH 主机。

```bash
python3 /实际任务目录/runtime/dataset.py verify --run-dir /实际任务目录
python3 /实际任务目录/runtime/dataset.py execute \
  --run-dir /实际任务目录 \
  --environment /实际环境配置.json
```

运行程序会启动：

```
snapshot_train → snapshot_valid → snapshot_test
→ fit（仅 train）
→ extract_train → extract_valid → extract_test
```

每个阶段日志在 logs，验收结果在 reports，manifest 保存阶段状态、Yarn ID、命令和配置哈希。valid/test 不再次拟合配置。

生成目录可从 Mac 移到 Linux，执行路径根据任务目录重新计算。不要仅复制 SQL 后改用旧 gen.sh，否则任务清单无法反映真正的执行过程。

## 4. 怎么判断真的完成了

manifest 的 planned、running、failed、validated 含义不同。只有 validated 表示本执行器完成验收。成功时输出路径结构：

```
<hdfs_root>/<dataset_id>/samples/train|valid|test
<hdfs_root>/<dataset_id>/fit_sample
<hdfs_root>/<dataset_id>/tfrecord/train|valid|test
```

验收包括来源日期覆盖、关联键唯一、缺失 label、候选样本键、label/cost/treatment、特征全空、物化读回行数、配置字段/token、转换前后行数、TFRecord 读回行数及最多 1000 条格式检查。reports 还记录缺失数和分组/日期分布；部分缺失率目前只报告，不自动判定业务可接受阈值。

训练时使用本次 conf 和对应数据路径，token_size=15，特征数由 manifest 得到。实际模型是否接受 15-token 以及是否需要重新生成 slots，要核对所使用的模型入口，不能盲目沿用旧命令。

## 5. 增加一个特征块的 SOP

1. 从真实 SQL 或表结构查证来源；记录上游业务含义、主键、时间口径及可用范围。未知的部分明确注明。
2. 在 feature-catalog.json 增加唯一 id、沿用注释的 name、可选 aliases、source_table、source_evidence、snapshot_offset_days 和 fields。
3. 每个 field 明确 name、expression、encoding、missing。表达式写清来源别名，输出使用 `AS 字段名`；字段名不能与 token/label 或其他块重复。
4. 如需 JOIN，在 assets/blocks 增加完整 LEFT JOIN 片段，登记 join_template、source_key、source_date、join_cardinality=many_to_one。时间模板可用 source_start/source_end（已应用偏移）和 sample_start/sample_end。仅把已有明确定义的业务 JOIN 加入模板。
5. 当前通用检查适用于单表、单用户键、一个快照日期的多对一 JOIN；更复杂的多表聚合/多键 JOIN，需要同步扩展来源校验适配器。不能只改模板却谎称已有校验覆盖复杂业务。
6. 用只包含该块和必要基础块的请求生成 SQL，检查字段与日期，添加能验证关联结果的 fixture 测试。
7. 在有代表性的小窗口运行，确认关联不会放大样本、缺失率可接受、分箱与下游读取一致，然后再用于正式窗口。

当前两块小天狼星已完整登记，作为增加 JOIN 的真实参考。第一个练习可以在 v6 请求里加入“新版小天狼星用户订单信息特征”，生成后应该有 566 个特征并出现 t-1 JOIN。

## 6. 登记历史故障的 SOP

sample_date 规则表示样本本身不可用；feature_snapshot 表示某一特征源快照不可用，须指定 block_id，程序会按偏移映射到受影响样本日期。规则包含 profile、splits、起止日期、原因和 evidence。

新增日期先确定故障影响范围，再登记原因及来源。历史 SQL 已排除但原因不明的，保留“原因待补充”；不能把它解释为已确认的具体事故。单次任务的额外排除可以写 additional_exclusions，每项是 date 和 reason。

## 7. 失败时如何处理

先读具体 stage 的日志和 Yarn ID。当前实现不自动重试失败任务，也不删除已有 HDFS 数据：

- completed 阶段可在验证报告及 _SUCCESS 后跳过。
- failed/running 或残留锁表示需要核查。进程仍在执行时不要清锁重跑。
- 确认失败后修改原因，使用新 dataset_id、新目录重新生成。
- 如果仅完成阶段之间退出，确认没有运行中的作业后，可以清理残留本地锁并继续；不能把 failed 手工改成 completed。

这版先保证不会误覆盖历史数据，后续可基于真实故障再实现更精细的阶段恢复。

## 8. 维护和验证

```bash
python3 -m unittest discover -s skills/wwt-prepare-training-data/tests -v
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/wwt-prepare-training-data
```

本地 SQL fixture 使用 SQLite 验证组合、JOIN、日期和字段行为，不等同于真实 Hive/Spark 集成测试。集群运行通过前，报告必须注明“执行适配器已实现，集群尚未验收”。

建议学习顺序：先改请求 → 增加一个已知块 → 登记一条测试日期规则 → 阅读 build → 阅读 runner → 最后扩展新数据口径。之后做完整 agent 时，这些脚本和 manifest 可以继续作为工具接口使用。
