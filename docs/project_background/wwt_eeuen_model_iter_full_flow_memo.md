# WWT EUEN / EEUEN 模型迭代全流程 Memo

本文用于让后续对话或项目参与者快速恢复本轮 EUEN / EEUEN 迭代上下文。重点记录项目背景、数据窗口、取数和特征生成、模型训练、test Hive 落表、模型评估与运筹仿真。后续如果继续迭代其他 uplift 模型，可以先读这份 memo，再按本文路径检查对应代码。

上一轮特征筛选流程可参考：

- `docs/project_background/wwt_euen_feature_select_full_flow_memo.md`

本轮不是做特征筛选，特征先沿用旧特征集合：`sirius画像特征 + 补贴环境 + 双端特征 + 有端 + 券敏感`。核心目标是验证：

1. `EEUEN` 相比 `EUEN` 是否带来收益。
2. 扩大训练样本窗口是否带来收益。

## 1. 项目背景

当前业务场景是短信/电话外呼触达用户，并在触达成功后发券。原 EUEN 模型直接学习不同 treatment 的 uplift 和成本，用于后续运筹分桶和券策略分配。EEUEN 的动机是把“是否成功曝光/触达”从转化 uplift 中拆出来：模型单独预测曝光概率，再学习曝光后的 uplift，最终排序分数使用：

```text
delta_cvr_k = exposure_prob * exposure_uplift_k
```

这样做的直觉是：用户最终是否被策略影响，既取决于被触达成功的概率，也取决于触达成功后券本身带来的转化提升。

本轮实现过两个 EEUEN 版本：

- `EEUEN_4t_02.py`：共享一个 exposure head，所有 treatment 共用一个曝光概率。
- `EEUEN_4t_03.py`：四个 exposure head，每个 treatment 一个曝光概率。

目前主流程建议先沿用 `EEUEN_4t_02.py` 作为稳定版本；`03` 可作为结构实验补充。原因是 `03` 虽然 exposure AUC 有提升，但最终 AUUC 不一定更好，因为 AUUC 看的是最终排序分数 `exposure_prob_k * exposure_uplift_k`，曝光概率更准不等于 uplift 排序更准。

## 2. 实验设计

### 2.1 数据窗口

当前有三套训练窗口，简称 `v1/v2/v3`：

| version | train | test | 说明 |
|---|---|---|---|
| `v1` | `2026-05-11` ~ `2026-06-17` | `2026-06-23` ~ `2026-07-06` | 对齐更早一期 EUEN baseline test |
| `v2` | `2026-05-11` ~ `2026-06-17` | `2026-07-23` ~ `2026-08-06`，排除 `2026-08-02` | 同训练窗口，换最新 test |
| `v3` | `2026-05-11` ~ `2026-06-17` + `2026-06-23` ~ `2026-07-22` | `2026-07-23` ~ `2026-08-06`，排除 `2026-08-02` | 扩大训练样本，看增量收益 |

注意：

- `v2` 和 `v3` 的 test 理论上应该完全一致，区别只在 train。
- `v1` 的 test 当前以 `2026-06-23~2026-07-06` 为准。历史上部分日志/模型路径里可能出现 `0623_0629`，不要被路径名误导；后续评估和运筹以实际 TFRecord / Hive 表数据日期为准。
- `v2/v3` 的 test 排除了 `2026-08-02`，最终仍凑够 14 天：`2026-07-23~2026-08-01` 加 `2026-08-03~2026-08-06`。

### 2.2 Treatment 映射

模型和 SQL 中需要保持 treatment 映射一致：

| fake_treatment_id | group_name_eve |
|---:|---|
| 0 | `control_group`，且 `flowtree_group_name = 'DS空白'` |
| 1 | `留存6张3元` |
| 2 | `8元留存5张3` |
| 3 | `留存6张5元` |
| 4 | `10元留存5张3` |

运筹仿真里当前主要使用 3 个 treatment 加 control：

```python
opt_group_name_list = [
    "DS空白",
    "留存6张3元",
    "8元留存5张3",
    "10元留存5张3",
]

discount_para = {
    "DS空白": 0,
    "留存6张3元": 1,
    "8元留存5张3": 2,
    "10元留存5张3": 3,
}
```

如果后续要把 `留存6张5元` 也纳入运筹，需要同步改 `opt_group_name_list`、`discount_para`、运筹函数里的组别逻辑和画图展示。

### 2.3 曝光 Label

EEUEN 需要额外曝光标签，本轮定义：

```sql
if(nvl(send_success_cnt, 0) > 0 or nvl(outcall_success_cnt, 0) > 0, 1, 0) as exp_label
```

含义：短信发券成功或电话外呼成功任一成立，则认为用户被成功曝光。

`exp_label` 必须同步出现在：

- SQL 输出字段。
- `parser_*.conf` 的 token 配置。
- `gen_feature_conf_*.sh` 生成逻辑。
- 模型训练脚本的 token 下标。
- test Hive 落表代码中，方便后续评估 exposure AUC / AUUCE。

## 3. 目录地图

### 3.1 SQL

```text
/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v2/feature_v2_train.sql
/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v2/feature_v2_test.sql

/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v3/feature_v3_train.sql
/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v3/feature_v3_test.sql

/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v4/feature_v4_train.sql
/Users/didi/wwt_local/wwt/offline_bigdata/script/sqls/v4/feature_v4_test.sql
```

其中：

- `v2`：train `0511-0617`，test `0723-0806 excluding 0802`。
- `v3`：train `0511-0617 + 0623-0722`，test `0723-0806 excluding 0802`。
- `v4`：主要用于补 `v1` 的 test，test `0623-0706`；train 可沿用 `v2` 窗口，方便命令统一。

### 3.2 Feature Conf

```text
/Users/didi/wwt_local/wwt/offline_bigdata/script/feature_confs/v2/
/Users/didi/wwt_local/wwt/offline_bigdata/script/feature_confs/v3/
/Users/didi/wwt_local/wwt/offline_bigdata/script/feature_confs/v4/
```

关键文件：

```text
gen_feature_conf_v2.sh
parser_v2_train.conf
feature_v2_train.conf

gen_feature_conf_v3.sh
parser_v3_train.conf
feature_v3_train.conf

parser_v4_train.conf
feature_v4_train.conf
```

### 3.3 Extract

```text
/Users/didi/wwt_local/wwt/offline_bigdata/script/extract_feature_scripts/v2/
/Users/didi/wwt_local/wwt/offline_bigdata/script/extract_feature_scripts/v3/
/Users/didi/wwt_local/wwt/offline_bigdata/script/extract_feature_scripts/v4/
```

常用脚本：

```text
extract_train_v4.sh
extract_test_v4.sh
```

### 3.4 模型训练代码

EUEN：

```text
/Users/didi/wwt_local/wwt/demo/euen_test_feature_time/euen_4t_02.py
```

EEUEN：

```text
/Users/didi/wwt_local/wwt/demo/EEUEN/EEUEN_4t_02.py
/Users/didi/wwt_local/wwt/demo/EEUEN/EEUEN_4t_03.py
```

参考原始 EUEN：

```text
/Users/didi/wwt_local/wwt/demo/EUEN/euen_base_4t.py
```

### 3.5 Test Hive 落表代码

EUEN 多版本切换：

```text
/Users/didi/wwt_local/wwt/demo/euen_test_feature_time/euen_test_hive_writer_switch.py
```

EEUEN 多版本切换：

```text
/Users/didi/wwt_local/wwt/demo/EEUEN/eeuen_test_hive_writer_switch.py
```

这两份代码都设计成“只改开头 `MODEL_VERSION = "v1" / "v2" / "v3"` 即可切换版本”。如果后续只有一个训练窗口，也可以只保留一个 config，其他逻辑不需要改。

### 3.6 运筹仿真 Notebook

当前主 notebook：

```text
/Users/didi/wwt_local/wwt/jupyter/EEUEN/EEUEN_运筹仿真-2.ipynb
```

历史参考：

```text
/Users/didi/wwt_local/wwt/jupyter/EEUEN/EEUEN_运筹仿真.ipynb
/Users/didi/wwt_local/wwt/jupyter/运筹仿真_self.ipynb
```

### 3.7 常用终端命令

```text
/Users/didi/wwt_local/wwt/offline_bigdata/script/euen_terminal_commands.md
```

里面记录了查日志、查 HDFS 文件、下载 TFRecord、查询 app id、kill Yarn 任务和 kill Python 训练进程的命令。

## 4. 拉数据流程

### 4.1 SQL 取数

SQL 主要做三件事：

1. 从 feature 表取用户特征。
2. 从 label 表取转化、成本、曝光、treatment 等 token / label。
3. 按 pid、dt、实验组关联后输出给后续特征转换。

label 表当前使用：

```text
dm.ut_alg_flowback_msg_outcall_label_v2_di
```

常用过滤条件：

```sql
and new_exp = '迭代流量'
and (
    group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')
    or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group')
)
```

`v2/v3` test 记得排除：

```sql
and dt not in ('2026-08-02')
```

如果 SQL 中还有历史排除日期，例如 `2026-04-03~2026-04-06`、`2026-04-08`，需要确认是否来自旧实验异常日逻辑。后续新实验不要机械照抄，要和业务确认是否仍需要排除。

### 4.2 Token 顺序

当前 token 顺序固定为：

```text
0  fin_order_wxmini_28d
1  group_name_eve
2  pid
3  oneid
4  send_cnt
5  send_success_cnt
6  send_cost
7  remind_cost
8  coupon_cost
9  dau_cnt
10 exp_label
11 cost_7d
12 fake_treatment_id
13 flag
14 label_dt
```

因此训练和落表代码里关键下标为：

```python
PID_INDEX = 2
EXP_INDEX = 10
COST_INDEX = 11
TREATMENT_INDEX = 12
LABEL_DT_INDEX = 14
```

更稳的写法是维护 `TOKEN_COLUMNS`，然后通过 `.index()` 获取下标：

```python
TOKEN_COLUMNS = [
    "fin_order_wxmini_28d",
    "group_name_eve",
    "pid",
    "oneid",
    "send_cnt",
    "send_success_cnt",
    "send_cost",
    "remind_cost",
    "coupon_cost",
    "dau_cnt",
    "exp_label",
    "cost_7d",
    "fake_treatment_id",
    "flag",
    "label_dt",
]

PID_INDEX = TOKEN_COLUMNS.index("pid")
EXP_INDEX = TOKEN_COLUMNS.index("exp_label")
COST_INDEX = TOKEN_COLUMNS.index("cost_7d")
TREATMENT_INDEX = TOKEN_COLUMNS.index("fake_treatment_id")
LABEL_DT_INDEX = TOKEN_COLUMNS.index("label_dt")
```

### 4.3 Feature Conf 生成

进入对应目录后运行：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/v2
nohup ./gen_feature_conf_v2.sh > feature_conf_v2.log 2>&1 &
tail -f feature_conf_v2.log
```

`gen_feature_conf_*.sh` 主要作用：

- 读取 SQL / parser 配置。
- 生成 `feature_v*_train.conf`。
- 将 token、label、feature slot 信息整理成 extract 可用配置。

注意事项：

- 如果报 `ModuleNotFoundError: No module named 'dmdeepl'`，优先检查 `gen.sh` 和 baseline 版本的环境变量、python 路径是否一致。
- `feature.conf` 可能在任务结束或关键步骤完成后才写入，日志还在跑时不要贸然判断文件为空。
- 如果修改了 token 字段、parser 字段、feature 列表，必须重新跑 gen。
- 如果只是改 extract 输出路径，一般不需要重新跑 gen。

### 4.4 Parser 配置

`parser_*.conf` 要和 SQL 输出字段严格对齐。重点检查：

- label 字段是否仍是训练脚本需要的 label。
- token 字段顺序是否和训练代码的 `TOKEN_COLUMNS` 一致。
- 新增 `exp_label` 后 `token_size` 是否同步为 `15`。
- 最后一行不要出现异常空字段、脏字符或多余分隔符，否则可能导致特征转换后行数为 0。

### 4.5 Extract 生成 TFRecord

extract 会把 SQL 结果和 feature.conf 转成 TFRecord，并写到 HDFS。

HDFS 当前常见输出：

```text
hdfs://difed/user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13/part-*
hdfs://difed/user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/part-*

hdfs://difed/user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13/part-*
hdfs://difed/user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/part-*

hdfs://difed/user/prod_growstrategy/tmp/wwt/v4/train/2026/08/13/part-*
hdfs://difed/user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13/part-*
```

本地训练缓存：

```text
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/train_0511_0617
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/test_0723_0806

/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/train_0511_0722
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v3/test_0723_0806

/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/train_0511_0617
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v4/test_0623_0706
```

下载示例：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203

mkdir -p wwt/v2/train_0511_0617
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13/* wwt/v2/train_0511_0617/

mkdir -p wwt/v3/train_0511_0722
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13/* wwt/v3/train_0511_0722/

mkdir -p wwt/v4/test_0623_0706
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13/* wwt/v4/test_0623_0706/
```

检查分片：

```bash
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13/part-* | wc -l
ls /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/v2/train_0511_0617/part-* | wc -l
```

如果 HDFS 只有 1 个 part，通常是 extract 脚本中的 repartition / 输出版本 / feature_version / stage 参数没有对齐，需要检查 extract 日志和 SQL stage count。

## 5. 模型训练

### 5.1 训练脚本

EUEN：

```text
/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/euen_test_feature_time/euen_4t_02.py
```

EEUEN：

```text
/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EEUEN/EEUEN_4t_02.py
/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EEUEN/EEUEN_4t_03.py
```

当前推荐：

- EUEN 对比使用 `euen_4t_02.py`。
- EEUEN 主对比使用 `EEUEN_4t_02.py`。
- `EEUEN_4t_03.py` 用于四 exposure head 的结构实验。

### 5.2 训练参数

当前 v2/v3/v4 数据对应参数：

```text
PADDING_SIZE = 473
DENSE_FEA_SIZE = 3500
token_size = 15
```

说明：

- `PADDING_SIZE` 是 feature slot 数，不包含 token。
- `token_size=15` 是 token 字段数。
- 如果 feature.conf / slots 文件变了，必须重新统计 `PADDING_SIZE`。
- 如果 parser token 变了，必须重新检查 `token_size` 和 token 下标。

### 5.3 训练评估方式

这次发现过一个关键问题：训练集 AUUC 和测试集 AUUC 的评估方式要对齐。

现在统一约定：

1. train 数据流用于训练，保持 shuffle。
2. 每个 epoch 训练结束后，用当前 epoch 最后参数，对 `train_eval_dataset` 完整推理一遍，再计算 train AUUC。
3. valid/test 也用同一份当前模型参数完整推理，再计算 AUUC。
4. `train_eval_dataset` 固定顺序、无 shuffle，避免额外遍历训练集时改变下一轮训练 shuffle 顺序。

这比旧逻辑更公平。旧逻辑可能是在训练过程中边训练边收集 batch 预测，再拼起来算 train AUUC；这样 train AUUC 其实混合了一个 epoch 内不同参数状态下的预测，不适合和 test AUUC 对比。

### 5.4 日志输出原则

训练日志需要保留：

- 每 `log_interval=100` 个 batch 的训练进度。
- epoch 粒度 loss。
- `outcome_loss / cost_loss / exposure_loss` 明细，方便看量级。
- 四个 treatment 的 AUUC 明细和 mean AUUC。
- cost AUC 明细。
- EEUEN 的 exposure AUC。
- EEUEN 的 AUUCE / Qinie，可收敛展示 mean，也可保留明细。

训练日志不要打印：

- 每个 batch 的完整 token。
- 每个 batch 的 treatment tensor。
- 每个 batch 的 cost tensor。
- 每个 batch 的 exp_label tensor。
- 每个 batch 的 unique 值。

这些调试输出会显著放大日志、影响 IO，并让排查真正问题变困难。只有在定位 token 下标或 parser 错位时，才通过 `--debug_token_check` 这类开关临时打开。

### 5.5 Early Stop

当前 early stop 暂时按：

```text
mean_auuc_val
```

也就是四个 treatment AUUC 的平均值。不要只看 T1，否则会偏向 base treatment。

## 6. EEUEN 模型结构

简化结构：

```text
Input TFRecord
    |
    |-- sparse feature ids
    |-- token: pid / treatment / exp_label / cost / label_dt
    |
Shared Bottom / Embedding
    |
    |-- Control Head: 预测 control response
    |
    |-- Uplift Head: 预测 exposure 后 uplift
    |       tau_e_1 = head_1
    |       tau_e_2 = head_1 + residual_2
    |       tau_e_3 = head_1 + residual_3
    |       tau_e_4 = head_1 + residual_4
    |
    |-- Exposure Head:
    |       02: 共享 exposure_prob
    |       03: exposure_prob_1 / 2 / 3 / 4
    |
    |-- Cost Head: cost_1 / cost_2 / cost_3 / cost_4
```

最终落表和运筹使用：

```text
uplift_k = exposure_prob * tau_e_k
```

对于 `03`：

```text
uplift_k = exposure_prob_k * tau_e_k
```

loss 当前为：

```text
loss = outcome_loss + cost_loss + exposure_loss
```

其中：

- `outcome_loss`：转化/完单相关监督。
- `cost_loss`：成本预测监督。
- `exposure_loss`：曝光二分类 BCE。

## 7. Test Hive 落表

训练完成后，需要对 test TFRecord 做推理，并落 Hive 表，供模型评估和运筹仿真使用。

### 7.1 EUEN 落表

路径：

```text
/Users/didi/wwt_local/wwt/demo/euen_test_feature_time/euen_test_hive_writer_switch.py
```

只改开头：

```python
MODEL_VERSION = "v1"  # v1 / v2 / v3
```

当前表名：

| version | table |
|---|---|
| `v1` | `growstrategy.ut_msg_outcall_euen_timetest_v1_0511_0617_inference_di` |
| `v2` | `growstrategy.ut_msg_outcall_euen_timetest_v2_0511_0617_inference_di` |
| `v3` | `growstrategy.ut_msg_outcall_euen_timetest_v3_0511_0722_inference_di` |

输出字段：

```text
pid
treatment
label_dt
exp_label
uplift_1, uplift_2, uplift_3, uplift_4
cost_1, cost_2, cost_3, cost_4
```

### 7.2 EEUEN 落表

路径：

```text
/Users/didi/wwt_local/wwt/demo/EEUEN/eeuen_test_hive_writer_switch.py
```

只改开头：

```python
MODEL_VERSION = "v1"  # v1 / v2 / v3
```

当前表名：

| version | table |
|---|---|
| `v1` | `growstrategy.ut_msg_outcall_eeuen_exposure_v1_0511_0617_inference_di` |
| `v2` | `growstrategy.ut_msg_outcall_eeuen_exposure_v2_0511_0617_inference_di` |
| `v3` | `growstrategy.ut_msg_outcall_eeuen_exposure_v3_0511_0722_inference_di` |

输出字段：

```text
pid
treatment
label_dt
exp_label
uplift_1, uplift_2, uplift_3, uplift_4
exposure_prob
exposure_uplift_1, exposure_uplift_2, exposure_uplift_3, exposure_uplift_4
cost_1, cost_2, cost_3, cost_4
```

注意：

- 表名本身已经区分 v1/v2/v3，不需要额外传 `inference_partition`。
- `best_epoch` 只是 config 里的说明字段，真正决定加载哪个模型的是 `hdfs_model_path`。
- 如果后续模型保存路径变化，只改 `MODEL_CONFIG[version]["hdfs_model_path"]`。

## 8. 模型评估

模型评估 notebook：

```text
/Users/didi/wwt_local/wwt/jupyter/EEUEN/EEUEN_模型评估.ipynb
```

评估要看：

1. AUUC：每个 treatment 一条，另算 mean AUUC。
2. Cost AUC：每个 treatment 的 cost 排序能力。
3. Exposure AUC：EEUEN 额外看曝光概率预测能力。
4. AUUCE / Qinie：EEUEN 的 exposure-aware uplift 效果。

注意：

- AUUC 是 uplift 排序指标，不是普通 AUC。
- Exposure AUC 更高不代表 AUUC 一定更高，因为最终排序分数还要乘 `tau_e_k`。
- 做 EUEN vs EEUEN 对比时，要保证 test 日期、base label 表、treatment 映射完全一致。

## 9. 运筹仿真

主 notebook：

```text
/Users/didi/wwt_local/wwt/jupyter/EEUEN/EEUEN_运筹仿真-2.ipynb
```

每次模型迭代后，运筹仿真需要重新做四步：

1. 拼接 SQL 表，生成每个模型对应的明细 dataframe。
2. 生成日期，并划分前 7 天 train、后 7 天 val。
3. 搜索 stage 参数。
4. 画图并定义 best stage，再做模型间对比。

### 9.1 拼接 SQL

当前需要拼六个模型：

```python
scheme_tables = {
    "euen_v1": {
        "table": "growstrategy.ut_msg_outcall_euen_timetest_v1_0511_0617_inference_di",
        "has_exposure": False,
        "seed_dts": V1_DTS,
    },
    "eeuen_v1": {
        "table": "growstrategy.ut_msg_outcall_eeuen_exposure_v1_0511_0617_inference_di",
        "has_exposure": True,
        "seed_dts": V1_DTS,
    },
    "euen_v2": {
        "table": "growstrategy.ut_msg_outcall_euen_timetest_v2_0511_0617_inference_di",
        "has_exposure": False,
        "seed_dts": V23_DTS,
    },
    "eeuen_v2": {
        "table": "growstrategy.ut_msg_outcall_eeuen_exposure_v2_0511_0617_inference_di",
        "has_exposure": True,
        "seed_dts": V23_DTS,
    },
    "euen_v3": {
        "table": "growstrategy.ut_msg_outcall_euen_timetest_v3_0511_0722_inference_di",
        "has_exposure": False,
        "seed_dts": V23_DTS,
    },
    "eeuen_v3": {
        "table": "growstrategy.ut_msg_outcall_eeuen_exposure_v3_0511_0722_inference_di",
        "has_exposure": True,
        "seed_dts": V23_DTS,
    },
}
```

日期不要再用一个全局 `SEED_DTS` 套所有模型。v1 和 v2/v3 的 test 日期不同，必须每个 scheme 自带 `seed_dts`。

日期定义：

```python
V1_DTS = tuple(pd.date_range("2026-06-23", "2026-07-06").strftime("%Y-%m-%d").tolist())

V23_DTS = tuple([
    dt for dt in pd.date_range("2026-07-23", "2026-08-06").strftime("%Y-%m-%d").tolist()
    if dt != "2026-08-02"
])
```

### 9.2 Train / Val 划分

运筹仿真里使用 test 窗口的前 7 天学习分桶/策略，后 7 天验证策略效果：

```python
def get_search_dates(seed_dts):
    seed_dts = list(seed_dts)
    train_dts = seed_dts[:7]
    val_dts = seed_dts[7:]

    train_dates = [train_dts[-1]] * len(val_dts)
    val_dates = val_dts

    return train_dts, val_dts, train_dates, val_dates
```

对于 `v2/v3`：

```text
train_dts = 2026-07-23 ~ 2026-07-29
val_dts   = 2026-07-30, 2026-07-31, 2026-08-01, 2026-08-03, 2026-08-04, 2026-08-05, 2026-08-06
```

对于 `v1`：

```text
train_dts = 2026-06-23 ~ 2026-06-29
val_dts   = 2026-06-30 ~ 2026-07-06
```

### 9.3 搜索 Stage 参数

EUEN 和 EEUEN 分开搜，因为 EEUEN 可以使用 `exposure_prob_per`，EUEN 没有这个字段。

```python
euen_scheme_dfs = {
    "euen_v1": scheme_dfs["euen_v1"],
    "euen_v2": scheme_dfs["euen_v2"],
    "euen_v3": scheme_dfs["euen_v3"],
}

eeuen_scheme_dfs = {
    "eeuen_v1": scheme_dfs["eeuen_v1"],
    "eeuen_v2": scheme_dfs["eeuen_v2"],
    "eeuen_v3": scheme_dfs["eeuen_v3"],
}
```

搜索函数里不要使用全局 `train_dates / val_dates`，必须按 `scheme_name` 取自己的日期：

```python
seed_dts = scheme_tables[scheme_name]["seed_dts"]
train_dts, val_dts, train_dates, val_dates = get_search_dates(seed_dts)
```

如果 stage 里包含某字段，但 SQL 没 select 出来，会在 `assign_bucket_num` 或 `move_threshold_main` 阶段报错。建议保留：

```python
check_stage_columns(data_df, columns, scheme_name, stage_name)
```

### 9.4 画图 1：一个模型一张图

目的：看每个模型不同 stage 参数下的验证集仿真曲线。

需要画六张：

```text
euen_v1
euen_v2
euen_v3
eeuen_v1
eeuen_v2
eeuen_v3
```

横轴：

```text
验证集实际增量 PV CAC
```

纵轴：

```text
验证集人均增量完单 PV
```

曲线排序建议按：

```python
sort_values("约束cac")
```

也就是说，策略搜索是按“约束 CAC”扫参数，画图时 x 轴展示实际落点。

### 9.5 定义 Best Stage

当前推荐人工看曲线后手动定义：

```python
best_stage_map = {
    "euen_v1": "s5",
    "eeuen_v1": "s3",
    "euen_v2": "s5",
    "eeuen_v2": "s3",
    "euen_v3": "s5",
    "eeuen_v3": "s3",
}
```

不要机械按单点最大值选。需要结合：

- 曲线整体是否稳定。
- 在目标约束 CAC 附近是否更优。
- 是否出现局部尖峰但整体不可控。
- 实际增量 CAC 是否落在业务可接受范围。

### 9.6 画图 2：不同模型最佳参数对比

主要画三张图：

```text
v1: euen_v1 vs eeuen_v1
v2: euen_v2 vs eeuen_v2
v3: euen_v3 vs eeuen_v3
```

用于回答：

1. 同一个训练窗口下，EEUEN 是否优于 EUEN。
2. 训练窗口扩大后，EUEN / EEUEN 是否各自提升。

如果需要看样本量收益，再补两张：

```text
euen_v2 vs euen_v3
eeuen_v2 vs eeuen_v3
```

## 10. 后续迭代模型时的检查清单

### 10.1 改数据窗口

需要同步检查：

- SQL train/test 日期。
- 是否排除异常日期。
- HDFS 输出路径。
- 本地下载目录。
- 模型启动命令里的 `train_tfr_path / valid_tfr_path / test_tfr_path`。
- `version` 字符串。
- `model_save_path`。
- test Hive 落表 config。
- 运筹仿真的 `scheme_tables` 和 `seed_dts`。

### 10.2 改特征

需要同步检查：

- SQL select 字段。
- parser.conf。
- gen_feature_conf 脚本。
- feature.conf。
- `PADDING_SIZE`。
- `DENSE_FEA_SIZE`。
- 是否需要重新 extract。
- 本地 TFRecord 是否重新下载。

### 10.3 改 token

需要同步检查：

- SQL token 字段顺序。
- parser token 顺序。
- `token_size`。
- 训练代码 `TOKEN_COLUMNS`。
- test Hive 落表代码 `TOKEN_COLUMNS`。
- 运筹 SQL 是否需要新增字段。

### 10.4 改模型结构

需要同步检查：

- 模型输出顺序。
- loss 函数。
- train / val / test 评估函数。
- test Hive 落表模型输出解包逻辑。
- Hive 输出字段名。
- 评估 notebook 和运筹 notebook 中 mapping_data 字段。
- 启动命令注释，注明本版本改动点。
- 每次复制/新建模型脚本后，同步修改脚本顶部 `Version`、改动点、示例命令、`--version`、`--model_save_path` 和日志文件名。

### 10.5 改训练评估

统一按当前方式：

- train 用 shuffle。
- train_eval 固定顺序、无 shuffle。
- 每个 epoch 后用当前模型完整推理 train_eval / val / test。
- early stop 看 `mean_auuc_val`。
- 保留 AUUC 明细，去掉 token/cost/treatment 的大批量 debug 日志。

## 11. 常见问题

### 11.1 为什么 AUUC 可能大于 1？

当前 AUUC 实现是业务归一化/面积类指标，不一定严格限制在 `[0, 1]`。要重点看同一口径下的相对变化，不要按普通 AUC 的范围理解。

### 11.2 为什么 exposure AUC 更高但 AUUC 更低？

EEUEN 的最终排序不是只看曝光概率，而是：

```text
delta_cvr_k = exposure_prob_k * exposure_uplift_k
```

exposure AUC 更高只说明“是否触达成功”的排序更准，但 AUUC 看的是 treatment uplift 的最终排序。如果 exposure head 改变了不同 treatment 的 score 尺度，可能反而影响 uplift 排序。

### 11.3 为什么 train 和 test AUUC gap 很大？

可能原因：

- train/test 数据分布不同。
- 老代码中 train AUUC 和 test AUUC 计算方式不一致。
- train AUUC 如果混合了训练过程中不同 batch 参数，会和 test 口径不一致。
- shuffle 被额外 train_eval 遍历影响，导致后续 epoch 训练顺序变了。

当前已统一为 train_eval 固定顺序完整推理，和 test 评估口径对齐。

### 11.4 v2/v3 test 是否应该完全一样？

是的。v2/v3 的 test 都是：

```text
2026-07-23 ~ 2026-08-06，排除 2026-08-02
```

如果评估结果差异来自 test 样本本身，需要检查：

- SQL 是否一致。
- label 表是否一致。
- extract 的 feature_version / output path 是否一致。
- 下载的本地目录是否指向同一批 HDFS 数据。
- Hive 落表时 model table 是否混用了旧模型或旧路径。

### 11.5 v1 test 到底是 0623-0629 还是 0623-0706？

当前流程以 `0623-0706` 为准。部分历史命令或模型路径可能包含 `0623_0629` 字样，但这只是旧命名残留。后续文档、运筹和评估统一写：

```text
v1 test = 2026-06-23 ~ 2026-07-06
```

## 12. 最短复用路径

后续如果重新迭代一个模型，建议按这个顺序走：

1. 复制或修改 SQL，确认 train/test 日期、label 表、treatment、exp_label。
2. 检查 parser.conf token 顺序。
3. 跑 gen，生成 feature.conf。
4. 跑 extract，生成 HDFS TFRecord。
5. 下载 TFRecord 到本地训练目录。
6. 修改模型训练脚本和启动命令，确认 `PADDING_SIZE=473`、`DENSE_FEA_SIZE=3500`、`token_size=15` 是否仍适用。
7. 训练模型，确认日志中 AUUC、loss、exposure AUC 等正常。
8. 用 test Hive writer 落表。
9. 在 `EEUEN_模型评估.ipynb` 做指标评估。
10. 在 `EEUEN_运筹仿真-2.ipynb` 拼六张/多张模型表，重新搜 stage 参数并画对比图。

这份 memo 的作用是恢复项目上下文；具体代码以对应目录中的最新脚本为准。
