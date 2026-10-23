---
name: wwt-prepare-training-data
description: 根据 WWT 已登记的 SQL 特征块和日期口径，生成及执行 train、valid、test 数据准备任务，交付共享 train 配置的 TFRecord、样本快照和验收清单。适用于抽数与训练数据准备，不包含模型训练或预测落表。
---

# WWT 训练数据准备

先读 [数据契约](references/data-contract.md)。使用 [操作与学习 SOP](references/sop.md) 执行或向用户解释各环节；增加特征块时读该文档的扩展流程。

## 需求解析

从用户需求提取任务 ID、特征块、三个完整日期窗口及临时排除日期。参考 [请求样例](assets/examples/v6-request.json)，生成 JSON 请求；当前 CLI 使用标准库 JSON，不接受 YAML。用户可用自然语言，无需亲自维护 JSON。

用 [特征目录](references/feature-catalog.json) 中的 name、id 或 aliases 精确解析 SQL 注释名称。未指定特征时可明确说明继承 v6 的五块；日期缺失且无法从当前请求确定时询问，不把样例日期当作实际生产请求。未知块先查 SQL/表来源并登记，不猜测字段和 JOIN。

历史排除日期使用 [日期登记](references/date-exclusions.json)。已登记的排除原因有的尚未查明，不编造线上故障原因。额外排除必须保留原因。特征快照故障按偏移转换成样本日期。

## 生成和执行

以下 SKILL_DIR 为本文件所在目录，REPO 为用户的 wwt 仓库，RUN 为新任务绝对目录；先解析实际路径再调用。

```bash
python3 "$SKILL_DIR/scripts/dataset.py" build --request "$REQUEST" --out "$RUN" --repo "$REPO"
python3 "$RUN/runtime/dataset.py" verify --run-dir "$RUN"
```

build 会冻结请求、日期规则、特征定义、SQL、运行代码和抽取二进制。不得编辑生成目录来修任务；修改请求或定义并创建新任务。

需要执行时读取 [环境示例](assets/examples/cluster-environment.json) 和 SOP。确认可用执行入口、HDFS 根路径、集群 Python 及真实 label 完整产出日期；环境中缺少的关键信息问用户。示例环境不是已验证连接。用户要求执行且信息完整时按其授权执行，不逐阶段重复确认。

```bash
python3 "$RUN/runtime/dataset.py" execute --run-dir "$RUN" --environment "$ENVIRONMENT"
```

必须在可用的 Linux 集群客户端运行；不能将本机没有 Spark 的情况报告为任务执行成功。跨机器时同步整个 RUN（不只 SQL），再在远端执行其 runtime 入口。

执行器依次物化三份样本 Parquet，使用 train 快照拟合一次配置，再对三个 split 抽取 TFRecord。valid/test 必须复用同一份 feature.conf/parser.conf；v5/v6 目录中的旧 conf 不能直接冒充新任务配置。第一版没有 Hive 注册表和 NFS 自动下载，不应宣称已完成这些操作。

## 结果与恢复

读取 manifest.json 和 reports；返回真实状态、选定块、有效日期与排除说明、特征/token 数、样本/TFRecord 路径、配置哈希和校验限制。planned 只表示生成完成，validated 才表示脚本完成了集群验收；格式检查目前为最多 1000 条抽样。

已成功阶段可校验后跳过。failed/running 状态、残留锁、输出冲突不得盲目重试或删除；先核实 Yarn 与路径。当前失败恢复采用新任务 ID，避免误覆盖。仅当旧进程已经结束且任务状态明确，才处理残留本地锁。用户未要求持续监控时不另建定时任务。

样本筛选沿用 v6 的 finish_all_cnt/finish_cnt_7d 规则，它依赖 outcome 且随窗口变化。解释数据量变化时必须考虑该规则；不要擅自改变。若用户需要无此筛选的新口径，应新增 profile 和相应验收。
