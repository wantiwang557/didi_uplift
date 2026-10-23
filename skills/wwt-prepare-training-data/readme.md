# WWT 拉数 Skill 使用与实现说明

这份 skill 把你原来的操作串起来：**编辑 SQL → 配置特征并运行 gen → 修正 parser 路径 → extract 抽取 train/valid/test → 检查分片 → 下载到 NFS**。

你提供特征块、日期、版本和执行日期，Codex 依据这里的规则生成文件，再按你的要求执行。它负责准备训练数据，不负责模型训练，也不会自动注册 Hive 结果表。

当前状态：目录生成、执行入口和检查逻辑已实现，12 项本地测试通过；最新修改尚未同步服务器，真实集群全流程尚未验收。本文描述的是当前代码行为，不代表已经成功产出一份线上数据。

## 1. 怎么调用

### 1.1 在 Codex 对话里调用（推荐）

复制下面的描述，替换成你的实际需求：

> 使用 $wwt-prepare-training-data 创建 v7。  
> 任务 ID：euen_v7_20260921。  
> 特征块：sirius画像特征、补贴环境、双端特征、有端、券敏感。  
> train：2026-05-11 至 2026-08-21。  
> valid：2026-08-22 至 2026-08-27。  
> test：2026-08-28 至 2026-09-10。  
> exec_date：2026-09-21。  
> 使用已登记的历史不可用日期，每份数据 100 个分片。  
> 先生成文件，告诉我生成路径，暂不运行集群任务。

如果希望执行，最后一句改成：

> 生成后在服务器运行 gen、extract，并检查分片、下载到 NFS 的 v7 目录。

上面的日期仅是示例。首次执行还需要确认集群配置和标签实际完整产出的日期。Codex 可以帮你整理请求 JSON 和环境 JSON，不需要你每次手写。

### 1.2 必须区分的参数

| 参数 | 示例 | 用途 |
|---|---|---|
| version | v7 | 决定 SQL、配置、脚本和 NFS 的版本目录 |
| dataset_id | euen_v7_20260921 | 区分一次具体任务，决定 HDFS 任务目录 |
| exec_date | 2026-09-21 | gen 和三份 extract 共用的执行日期，用于本次输出日期目录 |
| splits | 三组起止日期 | 决定三个 SQL 实际查询哪段数据，包含首尾 |
| feature_blocks | SQL 注释名组成的列表 | 决定输出哪些特征和需要哪些 JOIN |
| additional_exclusions | 日期及原因 | 本次额外排除的日期；已有历史排除自动应用 |
| partitions | 100 | 环境配置里的每份 TFRecord 目标分片数，也是检查标准 |
| label_available_through | 实际已完整产出标签的样本日期 | 必须覆盖 test 末日，不能仅根据今天减 7 天推算 |

**exec_date 必须一致。** 当前 gen 和 extract 不各自保存一个日期，而是读取同一份任务配置。未提供时 build 固定当天日期；正式使用建议明确填写。执行日期不会改变 SQL 中的 train/valid/test 窗口。

### 1.3 自己用命令生成文件

以下命令在服务器仓库根目录执行。`/绝对路径/request.json` 和 `/绝对路径/environment.json` 是你或 Codex 准备的实际文件，不是现成文件。

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt
python3 skills/wwt-prepare-training-data/scripts/dataset.py build \
  --request /绝对路径/request.json \
  --repo "$PWD"
```

请求格式可参考 [v5-request.json](assets/examples/v5-request.json)。复制后修改 version 和 dataset_id，不能直接用已有 v5 创建新任务。示例中的 `profile=euen_v6` 是目前程序支持的数据口径名称，不是输出目录版本；设置 version=v7 不需要把 profile 改成 euen_v7。

环境格式参考 [cluster-environment.json](assets/examples/cluster-environment.json)。其中 label_available_through 默认为空，需要补真实值；其他集群参数也需核实，不能把示例视为已验证配置。

默认不传 `--out`，会生成下一节的版本目录。`--out /新目录` 仅用于独立任务或临时测试，不会创建那三个仓库版本目录。

## 2. 调用后生成什么，在哪里

以下以 version=v7、dataset_id=euen_v7_20260921 为例。仓库路径为：

```text
/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt
```

### 2.1 仓库内的文件

```text
offline_bigdata/script/
├── sqls/v7/
│   ├── feature_v7_train.sql
│   ├── feature_v7_valid.sql
│   └── feature_v7_test.sql
├── feature_confs/v7/
│   ├── gen_feature_conf_v7.sh
│   ├── feature_v7_train.conf
│   ├── parser_v7_train.conf
│   ├── feature_conf_v7.log
│   ├── snapshot_train.log
│   ├── snapshot_valid.log
│   ├── snapshot_test.log
│   └── task/
└── extract_feature_scripts/v7/
    ├── extract_v7.sh
    ├── download_v7.sh
    ├── extract_v7_train.log
    ├── extract_v7_valid.log
    └── extract_v7_test.log
```

| 文件 | 什么时候生成 | 对应你的人工操作 |
|---|---|---|
| 三个 feature_v7_*.sql | build 时 | 分别编辑 train、valid、test SQL 的特征和日期 |
| gen_feature_conf_v7.sh | build 时 | 准备生成特征配置的入口 |
| feature_v7_train.conf | gen 执行后 | 原 gen 运行后得到的特征编码配置 |
| parser_v7_train.conf | gen 执行后 | 字段顺序、label、token、feature_list 等 parser 配置 |
| feature_conf_v7.log | gen 中生成配置时 | 原 feature_conf_v5.log |
| snapshot_train/valid/test.log | gen 执行三个 SQL 时 | SQL 读取数据的日志；新实现把这一步单独记日志，文件名暂沿用代码命名 |
| extract_v7.sh | build 时 | 一次依次抽取 train、valid、test 的入口 |
| 三个 extract_v7_*.log | 对应 extract 运行时 | 原三个 extract 日志，全部放在当前 v7 目录 |
| download_v7.sh | build 时 | 原检查分片和执行 hdfs dfs -get 的步骤 |
| task/ | build 时，运行中补充记录 | 保存本次请求、代码、配置、日志索引和检查结果 |

公开 SQL/conf 文件使用相对软链接：即两个路径指向同一份文件，不维护两份内容。SQL 的实际文件在 task/sql；conf 的实际文件在 task/conf。gen 前 conf 链接已经存在，但还没有有效配置文件。抽取日志的实际文件在 extract_feature_scripts/v7，task/logs 只链接到它们。

三个 version 目录中任意一个已存在，build 都会停止，避免覆盖旧版本。当前生成后会检查文件是否被修改，因此不要再手改已生成的 SQL；需要修改时更新请求或模板，再创建新版本。

### 2.2 task 目录里的文件

| 相对 task/ 的路径 | 谁生成 | 内容及用途 |
|---|---|---|
| request.json | build | 原始需求：版本、特征块、日期等 |
| resolved_request.json | build | 解析后的字段清单、来源表、统一 exec_date、实际可用日期及排除原因；执行时读它 |
| sql/train.sql、valid.sql、test.sql | build | 实际运行的三个 SQL，与公开 SQL 路径为同一份文件 |
| conf/feature.conf、parser.conf | gen | 本次 train 生成、三份抽取共享的实际配置 |
| runtime/ | build | 复制本次要用的 Python 程序和 fea_extract 二进制，避免以后仓库改代码影响本次任务 |
| definition/references/、definition/assets/ | build | 复制本次特征登记、日期规则、业务说明及 SQL 模板 |
| manifest.json | build 和 execute | 记录文件哈希、阶段状态、命令、Yarn ID、输出路径、配置哈希 |
| execution.json | 首次 execute | 保存本次使用的集群环境配置；后续不能悄悄换环境 |
| reports/snapshot_*.json | gen 中执行 SQL 后 | 记录各份查询结果行数、日期分布、关联及空值检查 |
| reports/fit.json | gen 成功后 | 记录 train 配置已生成及其哈希 |
| reports/extract_*.json | 各份 extract 成功后 | 记录输出位置、行数、分片文件及格式检查数量 |
| reports/delivery.json | 下载成功后 | 记录三份 NFS 目录、分片数和字节数 |
| logs/ | 运行时 | 指向版本目录中实际日志的链接 |
| gen.sh、extract.sh、download.sh | build | task 内部入口，与外层版本 shell 调用同一执行器 |
| .runner.lock/ | 执行期间 | 阻止同一任务被同时提交，正常结束后移除 |

文件哈希是用来判断内容有没有发生变化的校验值，不是新的业务配置。

### 2.3 HDFS 上的数据

使用环境样例的 hdfs_root 时，本次根目录为：

```text
/user/prod_growstrategy/tmp/wwt/skill_datasets/euen_v7_20260921/
├── samples/train/             # train SQL 查询出来的原始行
├── samples/valid/             # valid SQL 查询出来的原始行
├── samples/test/              # test SQL 查询出来的原始行
├── fit_sample/                # 从 train 中选出、用于生成特征配置的行
└── tfrecord/
    ├── train/2026/09/21/
    ├── valid/2026/09/21/
    └── test/2026/09/21/
```

samples 和 fit_sample 是程序额外保存的查询结果，使用 Parquet 文件格式；训练使用最终的 TFRecord。它们对应原流程里“gen/extract 执行 SQL 读取数据”的部分，只是现在读取后先保存下来，后续步骤读同一份结果。

HDFS 目录没有照搬旧的 `/tmp/wwt/v5/train/...`，而是按 dataset_id 隔离，以 manifest 中实际记录为准。三份 TFRecord 日期目录都来自同一个 exec_date。

### 2.4 NFS 上供训练使用的数据

使用环境样例的 nfs_root 时：

```text
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v7/
├── train_0511_0821/
├── valid_0822_0827/
└── test_0828_0910/
```

每个目录包含下载的 `part-*` 文件和 `delivery.json` 下载记录。这里的目录日期取 SQL 请求窗口的首尾，不是 exec_date，也不因排除几天而改变。重复实验或跨年导致名称相同时，应使用新 version。

## 3. 与原来手工步骤逐一对应

| 你原来的操作 | skill 现在做什么 | 调用程序 → 生成结果 |
|---|---|---|
| 复制一个版本并编辑 SQL | 根据块名组合字段、JOIN 和日期条件，生成三个 SQL | dataset.py 的 build_version/build/resolve → 三个 version 目录、SQL、请求记录 |
| 修改 gen 中 lisan_fea 等特征列表 | 从与 SQL 相同的字段定义得到特征列表，不再手写第二份长列表 | resolved_request.json 的 fields → fit_config.py 的 lisan_fea_list |
| 运行 gen.sh | 先执行三份 SQL，再只用 train 生成配置 | cluster_worker.py → 查询结果；fit_config.py → feature.conf/parser.conf |
| gen 后手动改 parser 的 feature_list | 自动改为 ./feature.conf，然后检查字段、token 和配置顺序 | dataset.py 的 check_conf → 修正后的 parser 和配置哈希 |
| 分别运行三份 extract | 依次读取三份已保存的查询结果，用同一 train 配置转换 | cluster_worker.py + 原 get_map_func + fea_extract → 三份 TFRecord |
| 看日志、数 part-* | 自动分开记录日志，并检查实际文件数与 partitions 一致 | shard_inventory → 分片清单；reports/extract_*.json |
| hdfs dfs -get 下载 | 下载到临时目录，核对文件名/数量/大小后改为最终目录 | dataset.py 的 deliver → NFS 三个目录及下载记录 |

**与人工流程的实际差别：** 原 gen/extract 各自读取 Hive；现在 gen 阶段先把三份 SQL 都执行并保存结果，然后生成配置和抽取都读这些结果。因此 gen 阶段会读取 valid/test，但它们不参与特征配置生成。这个安排是新增实现，不应把它理解成原 v5 shell 完全原样运行。

## 4. 实际怎么运行，日志怎么看

在服务器仓库根目录依次执行：

```bash
bash offline_bigdata/script/feature_confs/v7/gen_feature_conf_v7.sh /绝对路径/environment.json
bash offline_bigdata/script/extract_feature_scripts/v7/extract_v7.sh /绝对路径/environment.json
bash offline_bigdata/script/extract_feature_scripts/v7/download_v7.sh /绝对路径/environment.json
```

这些入口不依赖当前工作目录；上例使用相对命令路径，所以先进入仓库。第一个参数是环境 JSON，**不是 train/valid/test**。extract_v7.sh 一次顺序运行三份数据。

查看日志：

```bash
tail -f offline_bigdata/script/feature_confs/v7/feature_conf_v7.log
tail -f offline_bigdata/script/extract_feature_scripts/v7/extract_v7_train.log
```

程序自动写三个 extract 日志。若要后台运行，可把入口本身的输出也留在当前版本目录：

```bash
nohup bash offline_bigdata/script/extract_feature_scripts/v7/extract_v7.sh \
  /绝对路径/environment.json \
  > offline_bigdata/script/extract_feature_scripts/v7/extract_runner.log 2>&1 &
```

也可以由统一入口完成全部步骤：

```bash
python3 offline_bigdata/script/feature_confs/v7/task/runtime/dataset.py execute \
  --run-dir offline_bigdata/script/feature_confs/v7/task \
  --environment /绝对路径/environment.json
```

环境填写 nfs_root 时，上述 execute 会在抽取完成后自动下载。未填时只完成集群抽取；单独 download 必须填写 nfs_root。

## 5. Skill 是怎样封装的：逐文件说明

Skill 源码位于仓库 `skills/wwt-prepare-training-data/`。本地 Codex 的技能目录链接到这里，所以 Codex 可以识别技能名。把文件复制到服务器只提供执行程序；不会自动启动一个后台 agent。

### 5.1 给 Codex 看的说明和业务资料

| 文件 | 用来做什么 | 被谁读取、影响什么 |
|---|---|---|
| SKILL.md | 技能入口说明：适用范围、如何解析需求、调用程序、如何判断完成 | Codex 读它后选择参考文件和执行命令；它本身不运行 SQL |
| readme.md（本文） | 解释调用、产物路径、人工步骤与代码的对应关系 | 用户和 Codex 阅读 |
| agents/openai.yaml | 技能在 Codex 中的显示名称和调用提示 | Codex 展示/发现技能使用，不配置集群 |
| references/data-contract.md | 说明主表、label、cost、treatment、token、样本筛选等业务规则 | Codex 改 SQL/解释结果时参考；不是动态执行的 SQL |
| references/feature-catalog.json | 每个块的 id、注释名、别名、字段表达式、来源表、编码、JOIN 文件和日期偏移 | dataset.py 解析块名并生成字段和 JOIN；cluster_worker.py 用于来源检查 |
| references/date-exclusions.json | 不可用日期、影响 split、原因、证据；部分历史原因尚待补充 | dataset.py 从请求窗口中排除受影响日期，并记录原因 |

### 5.2 用来生成文件的模板和样例

| 文件 | 用来做什么 | 输入 → 输出 |
|---|---|---|
| assets/base.sql | 保存现有主表、标签关联、label/token 和样本筛选逻辑，预留字段、日期、JOIN 位置 | 选定特征和窗口 → 三个完整 SQL |
| assets/blocks/xtlx_order.sql | 新版小天狼星订单块的 JOIN SQL | 该块日期及偏移 → SQL 中对应 JOIN |
| assets/blocks/xtlx_user.sql | 新版小天狼星用户块的 JOIN SQL | 该块日期及偏移 → SQL 中对应 JOIN |
| assets/examples/v5-request.json | v5 窗口和特征的请求格式示例 | 复制并修改后作为 build 输入，不自动提交 |
| assets/examples/v6-request.json | 另一组日期的请求格式示例 | 同上 |
| assets/examples/cluster-environment.json | Python、队列、HDFS、NFS、分片等环境字段示例 | 核实并补齐后作为 execute 输入 |

### 5.3 三个实际执行程序

**scripts/dataset.py：总入口和流程控制。**

- `resolve`：检查块名、重复字段、日期窗口，应用排除规则，固定 exec_date。
- `build`：生成 SQL，复制本次代码和定义，写请求及 manifest。
- `build_version`：在三个仓库目录下创建 version，生成 shell 和相对链接，把日志位置安排到当前版本。
- `verify`：检查本次 SQL、代码、配置是否被改动。
- `execute`：依次启动下面两个程序，保存日志、状态和 Yarn ID。
- `check_conf`：确认生成配置与选定字段一致、token 正确，并修正 feature_list 路径。
- `shard_inventory`：调用 HDFS 命令检查 _SUCCESS 和实际分片数、零字节文件。
- `deliver`：调用 hdfs dfs -get，核对下载前后的文件名和大小，生成 NFS 下载记录。

**scripts/fit_config.py：对应原来运行 gen 生成特征配置。**

它加载本次复制的 `feature_list_handler.py`，继承原来的 `FeatureHandler`，调用其 `.run()` 生成配置。保留原分箱和配置写入逻辑，替换读取数据的方法 `_get_orig_df`：改为读取本次已经查出的 train 数据，并按样本键和 seed 选取用于分箱的行，写入本任务的 fit_sample。

特征列表来自 resolved_request.json 的 fields；label 固定为 label，token 使用已登记的 15 列，分箱数和采样比例来自请求。输出为 task/conf/feature.conf 和 parser.conf，公开的版本 conf 路径指向这两个文件。

**scripts/cluster_worker.py：对应原来的 SQL 读数和 extract 转换。**

- gen 阶段：通过 Spark 运行 SQL；检查来源日期、关联键、label、特征空值等，再保存查询结果到 HDFS samples/train、valid、test。
- extract 阶段：加载本次复制的 `run_offline_tans2tfr.py`，调用其中的 `get_map_func`。该函数再执行原 `fea_extract` 二进制，将原始列按 parser/feature 配置转换。结果写为 TFRecord。
- 检查转换前后及写入读回的行数、分片数量；数据格式最多抽查 1000 条，不是逐条格式验收。

### 5.4 从原项目复用哪些程序

| 原项目文件 | 如何复用 | 是否直接执行原入口 |
|---|---|---|
| offline_bigdata/python/feature_list_handler.py | build 复制到 task/runtime，fit_config.py 加载 FeatureHandler | 复用类和分箱逻辑，不执行原文件命令行入口 |
| offline_bigdata/python/run_offline_tans2tfr.py | build 复制到 task/runtime，cluster_worker.py 加载 get_map_func | 复用转换函数，不执行原文件命令行入口 |
| offline_bigdata/bin/fea_extract | build 复制到 task/runtime，由 Spark 分发给执行节点 | 转换函数实际调用此二进制 |
| 原 gen_feature_conf_v5.sh | 参考其特征和参数设置 | 不直接调用 |
| 原 extract_v5.sh / run_trans2tfr_spark32.sh | 参考其抽取流程和环境参数 | 不直接调用；新执行器组织 spark-submit |

调用关系如下：

```text
你的需求
  → Codex 阅读 SKILL.md，准备请求和环境 JSON
  → dataset.py build
      → SQL / 版本 shell / task 记录

版本 gen shell
  → task/runtime/dataset.py execute --phase gen
      → spark-submit cluster_worker.py：分别执行三个 SQL 并保存查询结果
      → Python fit_config.py
          → 原 FeatureHandler.run()：仅用 train 生成配置
      → check_conf：检查字段并恢复相对路径

版本 extract shell
  → task/runtime/dataset.py execute --phase extract
      → spark-submit cluster_worker.py：依次处理三份数据
          → 原 get_map_func → fea_extract → TFRecord
      → 分片和数据检查、各份日志及报告

版本 download shell
  → task/runtime/dataset.py execute --phase download
      → HDFS 检查 → hdfs dfs -get → NFS 文件核对
```

### 5.5 其他文件

| 文件 | 用途 |
|---|---|
| tests/test_dataset.py | 检查特征顺序、SQL 组合、日期排除、配置、执行顺序、分片下载、目录迁移及日志归属；集群调用部分使用模拟，不等于真实 Spark 验收 |
| .gitignore | 排除生成任务或缓存等文件，避免作为技能源码提交 |
| sync-manifest.json（如存在） | 之前同步时生成的文件清单；不是运行配置，也不能证明最新修改已同步 |
| runs/（如存在） | 之前生成的演示任务；不会随源码更新自动改变，不作为新版本的运行入口 |

## 6. 以后如何增加特征或修改规则

- **选已有块、改日期**：只改请求 JSON（也可让 Codex 从自然语言生成），无需改 Python。
- **替换主表/标签表**：请求使用 source_tables.feature / source_tables.label；替代表必须具有现有业务规则所需字段。SQL 和来源检查使用同一映射。
- **新增特征块**：先确认来源、字段、用户键、日期关系；在 feature-catalog.json 登记块名、字段表达式和编码。需要新 JOIN 时增加 assets/blocks SQL 文件并登记 join_template。目前通用检查只覆盖单表、单用户键、单日期的多对一 JOIN；复杂聚合或多键关联还需要改检查代码。
- **登记不可用日期**：修改 date-exclusions.json，写明影响范围、原因、依据。若坏的是前一天的特征数据，需要根据该块的日期偏移换算成受影响的样本日期。原因不清楚时明确注明，不能编造。
- **修改 label、token 或样本筛选**：同时核对 data-contract.md、base.sql、Python 中相关字段和检查逻辑。当前仅支持已登记的 lisan 编码，不自动支持任意新编码。

生成新版本前完成必要的 SQL 和行为测试。历史任务保存了自己的程序和定义，不会自动采用新规则。

## 7. 如何判断完成，失败后怎么做

manifest.json 中：planned=文件已生成；configured=gen 已完成；running=执行中；failed=某阶段失败；validated=集群抽取及检查完成；delivered=又完成 NFS 下载检查。

失败时先看对应日志和 Yarn 状态，不要仅看到目录存在就认为成功。已有失败输出不自动覆盖，当前通常修正原因后使用新 version 和 dataset_id。不要在原任务还运行时删除 .runner.lock，也不要手动把 failed 改为 completed。

下载先写 .partial 目录，检查通过才改为最终目录。下载中断会保留现场，需检查后处理；文件验收比较数量、名称、大小，不是远端与本地逐字节哈希比对。

跨机器复制版本任务时，需要保留相对软链接，并一起复制 sqls/<version>、feature_confs/<version>、extract_feature_scripts/<version>，保持相对目录关系。仅在显式 --out 独立任务模式下，才只需复制整个任务目录。
