# WWT EUEN 特征筛选完整流程 Memo

本文用于让后续 Codex 或项目参与者快速理解当前项目如何从原始 SQL 跑通到 EUEN 训练、Hive 推理表、离线 AUUC 评估和运筹仿真。

参考会话：

- `019f6acc-2714-7231-b172-b69a7f529cb4`：特征筛选项目、三套方案、EUEN 训练、推理落 Hive、日期和字段口径。
- `019f7ef0-3070-7ad1-afb5-291f011ed26c`：离线模型评估、AUUC 图、训练日志 AUUC 与 Jupyter AUUC 差异。
- `019f7edd-48dc-75a2-bc62-56eb355c5547`：离线运筹、stage 参数、实际 CAC 区间选方案。

## 1. 项目目标

当前项目背景是原始特征过多，希望做特征筛选，同时模型效果不能变差。核心评估指标包括：

- EUEN 训练日志里的 4 treatment AUUC / cost xAUC。
- Jupyter 离线验模里的 AUUC / Qini / uplift decile。
- 运筹仿真里的实际增量 CAC、实际增量 CVR、实际增量首单 CVR、留存频次。

当前这一轮主要比较三套特征方案：

| 方案 | 含义 | slot 数 |
|---|---|---:|
| `baseline` | sirius画像 + 补贴环境 + 双端 + 有端 + 券敏感 | 473 |
| `no_sirius` | 删除 sirius画像，只保留补贴环境 + 双端 + 有端 + 券敏感 | 279 |
| `replace_order` / `replace_sirius_with_xtlx_order` | 用新版小天狼星订单信息替换 sirius画像 | 372 |
| `all_superset` | baseline + 新版小天狼星订单信息 + 新版小天狼星基础属性 | 609 |

本项目的关键设计是：只抽一次全量 609-slot TFRecord，再通过 `.slots` 文件在模型读取阶段筛选出 473 / 279 / 372 个 slot，避免三套方案重复拉数。

以后不一定固定是“三套方案 x EUEN”。比较对象可以泛化成：

```text
1. 一套特征方案 + 多个深度模型结构
2. 多套特征方案 + 同一个深度模型
3. 多套特征方案 + 多个深度模型
```

通用原则是：只要想做公平比较，就尽量让 train/test 样本、label、treatment、cost 口径一致；每次只改变一个主要因素，例如只换特征、只换模型结构、只换 epoch/版本。

## 2. 关键文件地图

主要代码和文档：

- `demo/feature_select/euen_4t_feature.py`：特征筛选版 EUEN 训练脚本。
- `demo/feature_select/test_hive.py`：加载已训练模型，对 test TFRecord 推理并写 Hive 表。
- `demo/feature_select/test_hive_change_guide.md`：推理写 Hive 的参数和改动说明。
- `demo/feature_select/euen_4t_feature_code_readme.md`：特征筛选版 EUEN 代码解读。
- `demo/feature_select/euen_data_flow_compare.md`：原版 EUEN 与特征筛选版数据链路对比。
- `offline_bigdata/python/feature_select/build_feature_slot.py`：根据 SQL 和 feature_conf 生成 `.slots` 文件。
- `offline_bigdata/script/sqls/baseline/feature_baseline_train.sql`：训练集抽数 SQL。
- `offline_bigdata/script/sqls/baseline/feature_baseline_test.sql`：测试/验证集抽数 SQL。
- `offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf`：全量特征配置。
- `offline_bigdata/script/feature_confs/baseline/parser_baseline_train.conf`：抽 TFRecord 的 parser 配置。
- `offline_bigdata/script/feature_confs/baseline/gen_feature_conf_baseline.sh`：根据 SQL + 特征列表生成 `feature.conf` 和 `parser.conf`。
- `offline_bigdata/script/extract_feature_scripts/extract.sh`：用 train SQL/config 生成 train TFRecord 到 HDFS。
- `offline_bigdata/script/extract_feature_scripts/extract_test.sh`：用 test SQL/config 生成 test TFRecord 到 HDFS。
- `jupyter/模型评估_self.ipynb`：三套方案 Hive 推理表拼 label 表做 AUUC 评估。
- `jupyter/运筹仿真_self.ipynb`：三套方案和多 stage 的运筹仿真。
- `jupyter/短信外呼2026EUEN策略运筹-所有输出都保留.ipynb`：较完整的运筹参考 notebook。

已经生成的 slot 文件：

- `demo/feature_select/baseline_baseline.slots`：473 行。
- `demo/feature_select/baseline_no_sirius.slots`：279 行。
- `demo/feature_select/baseline_replace_sirius_with_xtlx_order.slots`：372 行。
- `demo/feature_select/baseline_all_superset.slots`：609 行。
- `demo/feature_select/baseline_summary.json`：slot 分组汇总。

## 3. 完整抽数链路

抽数不是一步完成的，实际链路是：

```text
写/改 SQL
  -> 配置 parser.conf / 特征列表
  -> 运行 gen_feature_conf_baseline.sh 生成 feature.conf 和 parser.conf
  -> 运行 extract.sh / extract_test.sh 转 TFRecord 并落 HDFS
  -> 从 HDFS get 到本地 NFS 训练目录
  -> euen_4t_feature.py 使用本地 train/test 路径训练
```

### 3.1 第一步：写 SQL

训练和测试 SQL 分开维护：

```text
offline_bigdata/script/sqls/baseline/feature_baseline_train.sql
offline_bigdata/script/sqls/baseline/feature_baseline_test.sql
```

这里需要确认：

```text
1. train/test 日期范围。
2. 实验组过滤条件。
3. label 口径，例如 finish_cnt_7d as label。
4. cost_7d 口径。
5. fake_treatment_id 映射。
6. label_dt 是否代表样本/label 日期。
7. 外部快照特征是否需要 t-1，例如小天狼星。
```

### 3.2 第二步：配置 parser / feature 生成脚本

入口脚本：

```text
offline_bigdata/script/feature_confs/baseline/gen_feature_conf_baseline.sh
```

这个脚本里最关键的是：

```bash
exec_dt="2026-07-16"
feature_version='baseline'

FEATURE_CONF_PATH=".../feature_baseline_train.conf"
PARSER_CONF_PATH=".../parser_baseline_train.conf"
SQL_PATH=".../feature_baseline_train.sql"

label="label"
token="fin_order_wxmini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,cost_7d,fake_treatment_id,flag,label_dt"
lisan_fea="..."
```

`lisan_fea` 是本次所有要离散化的特征列表。当前全量特征是 609 个，来自这个脚本里的 `lisan_fea` 和 SQL 输出字段。

运行方式：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/baseline
chmod +x gen_feature_conf_baseline.sh
nohup ./gen_feature_conf_baseline.sh > feature_conf_baseline.log 2>&1 &
tail -f feature_conf_baseline.log
```

输出：

```text
offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf
offline_bigdata/script/feature_confs/baseline/parser_baseline_train.conf
```

注意：`parser_baseline_train.conf` 里有一句注释很重要：

```text
跑完 gen.sh 之后，parse.conf 的路径变成绝对路径，必须修改
```

所以每次生成后，要检查 `feature_list=...` 是否指向任务环境能访问的 `feature_baseline_train.conf`。

### 3.3 第三步：extract 生成 HDFS TFRecord

train 抽数脚本：

```text
offline_bigdata/script/extract_feature_scripts/extract.sh
```

关键参数：

```bash
exec_date='20260716'
feature_verson='baseline'
stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${stage}.conf"
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100
```

运行方式：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
chmod +x extract.sh
nohup ./extract.sh > extract_train_baseline.log 2>&1 &
tail -f extract_train_baseline.log
```

train HDFS 结果示例：

```text
/user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16/part-*
```

test 抽数脚本：

```text
offline_bigdata/script/extract_feature_scripts/extract_test.sh
```

它的特点是：

```bash
stage='train'  # parser / feature 配置继续用 train 版
stage='test'   # SQL 和写入目录用 test
```

也就是说 test TFRecord 仍然使用同一套 train feature/parser 配置，保证 train/test 特征空间一致；但 SQL 使用 `feature_baseline_test.sql`，输出到 test 目录。

运行方式：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
chmod +x extract_test.sh
nohup ./extract_test.sh > extract_test_baseline.log 2>&1 &
tail -f extract_test_baseline.log
```

test HDFS 结果示例：

```text
/user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22/part-*
```

检查 HDFS：

```bash
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22/part-* | wc -l
hdfs dfs -du -s -h /user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22
```

### 3.4 第四步：HDFS 拉到本地 NFS 训练路径

EUEN 训练命令里的路径一般读本地/NFS 缓存，例如：

```text
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/train_20260511_20260617/part-*
/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*
```

从 HDFS 拉取示例：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203

mkdir -p wwt/baseline/train_20260511_20260617
mkdir -p wwt/baseline/test_20260623_20260706

hdfs dfs -get /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16/* wwt/baseline/train_20260511_20260617/
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22/* wwt/baseline/test_20260623_20260706/

ls wwt/baseline/train_20260511_20260617/part-* | wc -l
ls wwt/baseline/test_20260623_20260706/part-* | wc -l
```

最终 `euen_4t_feature.py` 的 `--train_tfr_path`、`--valid_tfr_path`、`--test_tfr_path` 就指向这些本地/NFS `part-*` 路径。

## 4. 抽数 SQL 和日期口径

当前 baseline SQL 会从两类主表拼样本：

- `dm.ut_alg_flowback_msg_outcall_feature_train_di`：实验样本和当时可用的主特征。
- `dm.ut_alg_flowback_msg_outcall_label_di`：真实 label 和实际成本。

两张主表同日 join：

```sql
t1.dt = t.dt
```

含义是对齐同一批实验样本，不是说所有特征都使用当天完整行为。

关键 label 和成本：

```sql
finish_cnt_7d as label
cast((
    nvl(send_cost, 0.0)
    + nvl(send_remind_cost, 0.0)
    + nvl(outcall_cost, 0.0)
    + nvl(remind_cost, 0.0)
    + nvl(msg_coupon_cost_7d, 0.0)
    + nvl(outcall_coupon_cost_7d, 0.0)
    + nvl(outcall_coupon2_cost_7d, 0.0)
) as double) as cost_7d
```

四组 treatment 映射：

```sql
case
    when group_name_eve = '10元留存5张3' then 4
    when group_name_eve = '留存6张5元' then 3
    when group_name_eve = '8元留存5张3' then 2
    when group_name_eve = '留存6张3元' then 1
    when group_name_eve = 'control_group' then 0
    else null
end as fake_treatment_id
```

当前 `label_dt` 直接取主样本/label 日期：

```sql
t.dt as label_dt
```

所以当前这套 0623-0706 SQL 生成的推理表，后续评估时应直接：

```sql
bs.dt = mapping_data.label_dt
```

不要再 `date_add(label_dt, 1)`。

小天狼星表单独用 `t-1` 快照：

```sql
on t.pid = tt.passenger_id and t.dt = date_add(tt.dt, 1)
on t.pid = ttt.pid and t.dt = date_add(ttt.dt, 1)
```

原因是小天狼星是额外离线画像/订单特征，预测 t 日样本时只能使用 t-1 已产出的快照，避免数据穿越。

## 5. TFRecord 和 token 顺序

上游 `fea_extract` 会把 SQL 结果转成 TFRecord，样本包含：

```text
label
feature
token
```

`feature` 是 `index:value,index:value,...`，每个 slot 对应一对 `[index, value]`。所以模型输入长度是：

```text
input_len = PADDING_SIZE * 2
```

当前 token 固定 14 个字段，顺序是：

| 下标 | 字段 |
|---:|---|
| 0 | `fin_order_wxmini_28d` |
| 1 | `group_name_eve` |
| 2 | `pid` |
| 3 | `oneid` |
| 4 | `send_cnt` |
| 5 | `send_success_cnt` |
| 6 | `send_cost` |
| 7 | `remind_cost` |
| 8 | `coupon_cost` |
| 9 | `dau_cnt` |
| 10 | `cost_7d` |
| 11 | `fake_treatment_id` |
| 12 | `flag` |
| 13 | `label_dt` |

训练脚本中用负索引：

```python
TREAT_INDEX = -3
COST_INDEX = -4
```

等价于：

```text
treatment = token[:, 11]
cost_7d = token[:, 10]
```

推理写 Hive 的 `test_hive.py` 必须使用：

```python
is_treat = tf.strings.to_number(t[:, 11], tf.float32)
pid.append(token[:, 2].numpy())
label_dt.append(token[:, 13].numpy())
```

如果 token 顺序变化，训练和推理脚本都要同步改。

## 6. 生成 slot 文件

slot 是 `feature_conf` 里的特征行号，不是离散后的全局 index。`.slots` 文件保存的是“保留第几个原始特征位置”。

如果以后只有一套特征方案，也仍然可以使用 `.slots` 来显式记录入模特征；如果模型就吃全量 609 个 slot，可以传 `baseline_all_superset.slots`，或者在脚本支持的情况下不传 `--feature_slot_file`。关键是模型训练和推理时看到的输入维度必须一致。

生成命令参考：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt

python offline_bigdata/python/feature_select/build_feature_slot.py \
  --feature_conf offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf \
  --sql offline_bigdata/script/sqls/baseline/feature_baseline_train.sql \
  --out_dir demo/feature_select \
  --prefix baseline_
```

检查结果：

```bash
wc -l demo/feature_select/*.slots
```

期望：

```text
609 baseline_all_superset.slots
473 baseline_baseline.slots
279 baseline_no_sirius.slots
372 baseline_replace_sirius_with_xtlx_order.slots
```

## 7. 训练模型：可比较一套或多套方案

训练入口：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select
mkdir -p run_logs
```

当前这一轮是三套特征方案都使用同一份全量 TFRecord，只换：

```text
--feature_slot_file
--model_save_path
--version
日志文件名
```

共同注意：

- `--DENSE_FEA_SIZE 5000`：这是 embedding 全局 index 空间，不是 slot 数。
- `--token_size 14`。
- 不要传旧的 `--PADDING_SIZE 448`；脚本会用 `.slots` 行数覆盖 `args.PADDING_SIZE`。
- 当前 valid 和 test 都指向 `test_20260623_20260706`，因此 best epoch 的选择带有“用 test 当 valid”的性质，做最终严谨结论时要重新确认。

baseline：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/train_20260511_20260617/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_0623_0706" \
--feature_slot_file "./baseline_baseline.slots" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_baseline_0623_0706" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 2 > ./run_logs/euen_baseline_0623_0706.log 2>&1 &
```

no_sirius：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/train_20260511_20260617/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/no_sirius_0623_0706" \
--feature_slot_file "./baseline_no_sirius.slots" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_no_sirius_0623_0706" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 0 > ./run_logs/euen_no_sirius_0623_0706.log 2>&1 &
```

replace_order：

```bash
nohup python euen_4t_feature.py \
--train_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/train_20260511_20260617/part-*" \
--valid_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--test_tfr_path "/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt/baseline/test_20260623_20260706/part-*" \
--model_save_path "hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/replace_order_0623_0706" \
--feature_slot_file "./baseline_replace_sirius_with_xtlx_order.slots" \
--log_interval 100 \
--shuffle_seed 0 \
--parallel 4 \
--version "euen_replace_order_0623_0706" \
--DENSE_FEA_SIZE 5000 \
--token_size 14 \
--gpu 0 > ./run_logs/euen_replace_order_0623_0706.log 2>&1 &
```

注意运行目录。如果从 `demo` 目录跑，相对路径 `./baseline_baseline.slots` 和 `./run_logs/...` 会错。最稳是进入：

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/feature_select
```

如果以后只比较一套特征方案，则可以只跑一条训练命令；如果以后比较不同深度模型，则需要保持同一份 train/test TFRecord、同一份 `.slots`、同一套 label/treatment/cost 口径，然后只切换模型脚本和模型保存路径。

建议未来每次实验记录一张对照表：

| 实验名 | 特征方案 | 模型脚本 | train 路径 | test 路径 | slot 文件 | model_save_path | 日志 |
|---|---|---|---|---|---|---|---|
| `euen_baseline_0623_0706` | baseline | `euen_4t_feature.py` | `.../train_.../part-*` | `.../test_.../part-*` | `baseline_baseline.slots` | `hdfs://.../baseline_0623_0706` | `run_logs/...log` |

## 8. EUEN 输出含义

模型原始输出：

```text
u_tau_1 = T1 - control
u_tau_2 = T2 - T1
u_tau_3 = T3 - T1
u_tau_4 = T4 - T1
```

训练评估和 Hive 落表前都会还原为各 treatment 相对 control：

```python
delta_cvr_1 = u_tau_1
delta_cvr_2 = u_tau_2 + u_tau_1
delta_cvr_3 = u_tau_3 + u_tau_1
delta_cvr_4 = u_tau_4 + u_tau_1
```

因此 Hive 表里的：

```text
uplift_1 = T1 - control = 留存6张3元 vs control
uplift_2 = T2 - control = 8元留存5张3 vs control
uplift_3 = T3 - control = 留存6张5元 vs control
uplift_4 = T4 - control = 10元留存5张3 vs control
```

不要在评估 SQL 里再做：

```text
left853 = left63 + uplift_2
```

正确是：

```sql
uplift_1 as left63,
uplift_2 as left853,
uplift_3 as left65,
uplift_4 as left1053
```

然后如果需要 treatment 间差值，再后处理：

```sql
left853_minus_left63 = left853 - left63
left65_minus_left63 = left65 - left63
left1053_minus_left63 = left1053 - left63
left1053_minus_left853 = left1053 - left853
left65_minus_left853 = left65 - left853
left1053_minus_left65 = left1053 - left65
```

如果要评估 `留存6张5元` 相比 `10元留存5张3` 的升级效果，则应补：

```sql
left65_minus_left1053 = left65 - left1053
```

## 9. cost_1 到 cost_4 的含义

`cost_1~cost_4` 是 EUEN 模型的 cost head 输出，含义是：

```text
cost_i = 如果给用户 treatment i，模型预测的实际 7 日成本期望
```

它拟合的是 token 中的 `cost_7d`，也就是实际发生成本，不是券面额。因此 cost 数值很小通常是正常的，因为它已经隐含触达率、领取率、使用率、核销率和实际发生成本概率。

四组券型的大致业务顺序：

```text
1: 留存6张3元
2: 8元留存5张3
4: 10元留存5张3
3: 留存6张5元
```

如果做单调性检查，成本更合理的顺序一般是：

```python
cost_1 < cost_2 < cost_4 < cost_3
```

uplift 业务顺序也可按：

```python
left63 < left853 < left1053 < left65
```

但这只是业务合理性检查，不是强约束。

## 10. 训练脚本的 epoch 选择注意

当前 `euen_4t_feature.py` 里变量名 `average_auuc` 容易误导。代码实际把 `predict_delta_cvr_1` 的 AUUC 当作 best model 和 early stopping 指标：

```python
if key == 'predict_delta_cvr_1':
    auuc_val_base = item[key]
average_auuc = auuc_val_base
```

所以日志里自动保存的 best model 是“treatment 1 AUUC 最优”，早停也是基于 treatment 1，而不是 4 个 treatment AUUC 平均最优。

真实选 epoch 时，不要只依赖脚本自动保存的 best。需要打开训练日志，人工横向比较每个 epoch 的 4 个 treatment AUUC。当前经验上通常重点看 epoch 1 到 epoch 3，后面 epoch 即使 T1 更高，也可能出现某些 treatment 下滑或过拟合。

如果目标是四组 treatment 综合效果，建议人工计算或改代码用四组平均：

```python
auuc_values = [item[ite_columns[i]] for i, item in enumerate(auuc)]
average_auuc = sum(auuc_values) / len(auuc_values)
```

历史讨论中的判断：

- baseline 截图里，从 AUUC 综合看更偏向 Epoch 2；如果当前实际使用 `baseline_epoch3/`，需要以本轮日志和 Hive 推理分区为准。
- no_sirius 截图里，代码会因 T1 AUUC 选 Epoch 6，但四组平均更建议 Epoch 3。
- replace_order 截图里，代码会因 T1 AUUC 选 Epoch 5，但四组平均更建议 Epoch 3。

最终报告里应明确：选 epoch 的依据是 T1 AUUC、四组平均 AUUC、关键 treatment AUUC，还是后续 Jupyter 离线 AUUC。

## 11. 模型训练后推理写 Hive 表

入口脚本：

```text
demo/feature_select/test_hive.py
```

它做的事情：

```text
加载已训练模型
读取同一份全量 test TFRecord
按对应 .slots 文件筛特征
模型预测 uplift_1~4 和 cost_1~4
取 pid / treatment / label_dt
写 Hive 推理表
```

注意：训练脚本本身主要产出 SavedModel/checkpoint 到 `model_save_path`；模型结果要变成 Hive 推理表，需要再用 `test_hive.py` 或对应模型的推理落表脚本加载模型，对 test TFRecord 打分后写入 Hive。

当前输出表：

```text
growstrategy.ut_msg_outcall_feature_select_v0_month_5_6_model_inference_di
```

如果以后换成其他深度模型，也建议沉淀一份对应的“推理落 Hive 脚本”，并尽量保持最终 Hive 表字段一致。这样后续 `模型评估_self.ipynb` 和运筹 notebook 可以少改。

输出字段：

```text
pid
treatment
label_dt
uplift_1
uplift_2
uplift_3
uplift_4
cost_1
cost_2
cost_3
cost_4
dt 作为 Hive 分区字段
```

`startDate` 是写 Hive 的单个分区：

```python
partition_val = args.startDate
partition_col = 'dt'
```

当前代码不会按每条样本的 `label_dt` 写多个分区。如果 `startDate=20260706`，整批样本都会写入：

```text
dt=20260706
```

`test_dt` 当前基本只是 reader 的占位参数。如果 `test_rta_tfr_path` 是固定路径：

```python
test_rta_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/16/part-*"
```

那么真实读取哪批数据由该路径决定，不由 `test_dt` 决定。

三套方案推理时，模型路径和 slot 必须成对：

| 方案 | 模型路径示例 | slot 文件 |
|---|---|---|
| baseline | `hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/baseline_epoch3/` 或实际 best 路径 | `./baseline_baseline.slots` |
| no_sirius | `hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/no_sirius_0623_0706` | `./baseline_no_sirius.slots` |
| replace_order | `hdfs://difed/user/prod_growstrategy/wwt/demo/feature_select/replace_order_0623_0706` | `./baseline_replace_sirius_with_xtlx_order.slots` |

常见维度报错：

```text
expected shape=(None, 946), found shape=(20480, 1218)
```

含义是 473-slot 模型收到了 609-slot 全量输入。解决办法是确认推理脚本也读取并应用 `.slots` 文件。

## 12. Jupyter 离线 AUUC 验模

核心 notebook：

```text
jupyter/模型评估_self.ipynb
```

目的：把 Hive 推理表和真实 label 表重新拼成评估宽表 `BASE_FMT_v1/v2/v3`。

为什么要重新拼 label 表：

```text
推理表只有 pid、label_dt、uplift_1~4、cost_1~4 等预测结果。
AUUC / Qini / decile 需要真实 treatment、真实 label、真实 cost、group_name_eve。
这些真实字段来自 dm.ut_alg_flowback_msg_outcall_label_di。
```

推荐 SQL 口径：

```sql
with base as (
    select *,
        finish_cnt_7d as label,
        cast((
            nvl(send_cost, 0.0)
            + nvl(send_remind_cost, 0.0)
            + nvl(outcall_cost, 0.0)
            + nvl(remind_cost, 0.0)
            + nvl(msg_coupon_cost_7d, 0.0)
            + nvl(outcall_coupon_cost_7d, 0.0)
            + nvl(outcall_coupon2_cost_7d, 0.0)
        ) as double) as cost_7d
    from dm.ut_alg_flowback_msg_outcall_label_di
    where dt between '2026-06-23' and '2026-07-06'
      and new_exp = '迭代流量'
      and (
          group_name_eve in ('留存6张5元', '留存6张3元', '8元留存5张3', '10元留存5张3')
          or (flowtree_group_name = 'DS空白' and group_name_eve = 'control_group')
      )
),
mapping_data as (
    select
        pid,
        label_dt as dt,
        treatment as model_treatment,
        uplift_1 as left63,
        uplift_2 as left853,
        uplift_3 as left65,
        uplift_4 as left1053,
        cost_1,
        cost_2,
        cost_3,
        cost_4
    from growstrategy.ut_msg_outcall_feature_select_v0_month_5_6_model_inference_di
    where dt = '20260706'
      and label_dt between '2026-06-23' and '2026-07-06'
)
select ...
from base bs
join mapping_data
  on bs.pid = mapping_data.pid
 and bs.dt = mapping_data.dt
```

建议先用 `left join` 检查缺失，再正式用 `join`：

```sql
count(*) as base_cnt,
count(mapping_data.pid) as matched_cnt
```

离线 AUUC：

```python
vis.draw_auuc(
    BASE_FMT_v1,
    treatments=[1, 2, 3, 4],
    ite_names=['left63', 'left853', 'left65', 'left1053'],
    label_col='label',
    curve_type='auuc',
    real_treatment='real_treatment',
    control_group_name=0,
    show_curve=1
)
```

升级/残差 AUUC 示例：

```python
vis.draw_auuc(
    BASE_FMT_v1,
    treatments=[2],
    ite_names=['left853_minus_left63'],
    label_col='label',
    curve_type='auuc',
    real_treatment='real_treatment',
    control_group_name=1,
    show_curve=1
)
```

训练日志 AUUC 和 Jupyter AUUC 不一致时，优先排查：

```text
1. TFRecord test 与 Hive label 表是否逐样本一致。
2. Jupyter SQL join 是否丢样本。
3. join key 是 pid 还是 oneid，dt 是否和 label_dt 对齐。
4. label 是否都是 finish_cnt_7d。
5. uplift_2/3/4 是否被重复加 left63。
6. Hive 推理表是否真的是同一个 checkpoint/epoch 推出来的。
7. AUUC 函数版本是否一致。
```

Jupyter 快速检查：

```python
print(BASE_FMT_v1.shape)
print(BASE_FMT_v1['real_treatment'].value_counts(dropna=False).sort_index())
print(BASE_FMT_v1.groupby('real_treatment')['label'].agg(['count', 'mean', 'sum']))
print(BASE_FMT_v1[['left63', 'left853', 'left65', 'left1053']].describe())
print((BASE_FMT_v1[['left63', 'left853', 'left65', 'left1053']] == 999).sum())
```

## 13. 运筹仿真

核心 notebook：

```text
jupyter/运筹仿真_self.ipynb
```

输入通常是离线评估宽表。当前这一轮是三份：

```python
scheme_dfs = {
    'baseline': v1,
    'no_sirius': v2,
    'replace_order': v3,
}
```

核心函数：

- `ortools_coupon(...)`：调用 OR-Tools，在成本约束下选择每个 bucket 的券型。
- `simulate_and_plot_all(...)`：循环 `cost_list`，对 train 求解策略，对 test/val 仿真效果，并画 CAC-CVR 曲线。
- `main_try(...)`：负责分桶、调用求解、组织结果。

整体流程：

```text
1. 准备每套模型/特征方案的评估宽表，例如 v1/v2/v3。
2. 选定参与运筹的券型 group_name_list 和 control_name。
3. 选定用于分桶的模型分数字段 columns / all_columns。
4. 定义 stage_configs：每个 stage 是一套分桶字段 + min_count 阈值组合。
5. 对每个 scheme + stage 调 main_try。
6. main_try 内部先在 train_dates 上分桶，并统计每个桶下各券型的真实增量效果和成本。
7. 对每个 约束cac 调 ortools_coupon，在成本约束下求每个桶应选哪个券型。
8. 把 train 上求出的策略迁移到 val/test 日期上仿真。
9. 汇总成 stage_summary_df。
10. 按实际增量 CAC 区间和实际增量 CVR 选择方案、stage、策略点。
```

当前常用 treatment 组：

```python
group_name_list = ['DS空白', '留存6张3元', '8元留存5张3', '10元留存5张3']
discount_para = {'DS空白': 0, '留存6张3元': 1, '8元留存5张3': 2, '10元留存5张3': 3}
cost_list = np.arange(6.5, 9.0, 0.1)
```

如果运筹不纳入 `留存6张5元`，它仍然可以作为分桶特征参与，例如 `left65` 或 `left65_minus_left63` 可用于分桶判断，但不一定作为最终可选券型。

主要可调参数：

| 参数 | 位置/变量 | 作用 |
|---|---|---|
| 参与券型 | `group_name_list` | 决定 OR-Tools 可以选择哪些 treatment。 |
| 对照组 | `control_name` | 通常是 `DS空白`。 |
| 券型编号 | `discount_para` | 把券型名称映射成优化器内部编号。 |
| CAC 约束序列 | `cost_list` | 每个约束 CAC 下求一套策略点。 |
| 训练日期 | `train_dates` | 用于求解策略的人群和真实效果。 |
| 验证日期 | `val_dates` | 用于仿真策略泛化效果。 |
| 分桶原始字段 | `columns` | 需要先转成 `_per` 排名/分位字段的模型分数。 |
| 组合分桶字段 | `all_columns` | 真正参与组合桶的 `_per` 字段。 |
| stage 配置 | `stage_configs` | 多套分桶字段和 `min_count` 阈值。 |
| 桶样本阈值 | `min_count` | 阈值越低，第一阶段圈出的桶越多；阈值越高，桶更稳但更粗。 |
| 默认券 | `default_coupon` | 未被策略覆盖或兜底时使用的券型。 |
| 是否仿真 | `simulation_flag` | 控制是否走完整策略仿真。 |

stage 参数的直觉：

```text
字段越多 -> 桶更细 -> 单桶样本更少 -> 可能更精细，也更容易不稳。
字段越少 -> 桶更粗 -> 单桶样本更多 -> 更稳，但策略区分度下降。
min_count 越低 -> 能圈出的组合桶越多。
min_count 越高 -> 只保留大样本桶，泛化更稳。
```

`stage_summary_df` 是核心结果表，字段：

```text
scheme
stage_name
data_type
约束cac
实际增量cac
实际增量cvr
实际增量首单cvr
留存频次
```

主要看这些指标：

| 指标 | 含义 | 怎么用 |
|---|---|---|
| `约束cac` | OR-Tools 输入约束 | 看策略路径，不直接作为最终成本。 |
| `实际增量cac` | 验证集增量成本 / 增量完单 | 最终业务约束优先看它。 |
| `实际增量cvr` | 人均增量完单 PV | 同等实际 CAC 下越高越好。 |
| `实际增量首单cvr` | 人均增量首单/UV 转化 | 如果业务更关注用户首单，要重点看。 |
| `留存频次` | 增量 PV / 增量 UV 类似频次指标 | 判断策略带来的订单质量和频次。 |

选方案时推荐顺序：

```text
1. 先确定业务可接受的实际增量 CAC 区间或上限。
2. 在这个实际 CAC 区间内比较实际增量 CVR。
3. 同时看实际增量首单 CVR 和留存频次，避免只优化 PV。
4. 对 scheme/stage 不看单点最高，优先看区间平均表现和多个 CAC 档位胜率。
5. 最后检查 train/test gap 和桶质量，防止某个 stage 只是偶然高。
```

`约束cac` 与 `实际增量cac` 的区别：

- `约束cac` 是给 OR-Tools 的输入，回答“我要求求解器按多严格的 CAC 约束求策略”。
- `实际增量cac` 是验证集仿真结果，回答“这套策略在验证集真实跑出来的增量成本 / 增量完单是多少”。

主函数画图时，很多时候是按 `约束cac` 的循环顺序 append 点再 `plt.plot`，所以连线顺序是约束 CAC 顺序。若按图横轴实际 CAC 做业务比较，应使用 `实际增量cac` 区间筛选，而不是只看 `约束cac`。

按实际 CAC 区间选方案：

```python
cac_min = 7.0
cac_max = 7.3
stage_name = 's1'

actual_range_df = stage_summary_df[
    (stage_summary_df['data_type'] == 'test') &
    (stage_summary_df['stage_name'] == stage_name) &
    (stage_summary_df['实际增量cac'] >= cac_min) &
    (stage_summary_df['实际增量cac'] <= cac_max)
].copy()

actual_range_df.sort_values('实际增量cvr', ascending=False)[
    ['scheme', 'stage_name', '约束cac', '实际增量cac', '实际增量cvr', '实际增量首单cvr', '留存频次']
]
```

每套方案各自选区间内最优点：

```python
best_each_scheme = (
    actual_range_df
    .sort_values(['scheme', '实际增量cvr'], ascending=[True, False])
    .groupby('scheme')
    .head(1)
    .sort_values('实际增量cvr', ascending=False)
)
```

如果要选 stage，不要只看最高单点，建议综合：

```text
1. 目标实际 CAC 区间内 avg_delta_cvr。
2. 多个 target_cac 档位下的 win_count。
3. train/test gap。
4. bucket 质量：bucket 数量、val bucket_num=-1 占比、桶样本量。
```

历史判断：

- `s1` 是主推 stage，收益高。
- `s4` 是稳健备选，曲线更平滑。
- 如果业务目标实际 CAC 在 `7.0~7.3`，从截图和讨论看 `replace_order` 通常更有性价比。
- `no_sirius` 在已有图里整体偏弱，但仍应以目标 CAC 区间表格为准。

## 14. 常见问题和排查清单

每次换模型或方案，至少检查：

```text
1. SQL 的日期、实验组、label、cost_7d、fake_treatment_id 是否正确。
2. gen_feature_conf_baseline.sh 的 lisan_fea、label、token 是否和 SQL 输出一致。
3. parser_baseline_train.conf 里的 feature_list 路径是否可访问。
4. extract.sh / extract_test.sh 的 exec_date、feature_verson、stage、hdfs_work_space 是否正确。
5. HDFS 结果是否存在，part 文件数量和大小是否正常。
6. HDFS 是否已经 get 到 EUEN 训练命令使用的本地/NFS train/test 路径。
7. 是否在 demo/feature_select 目录运行训练，避免相对路径错。
8. train/valid/test TFRecord 是否是同一份 609-slot 全量特征空间。
9. feature_slot_file 是否和模型训练方案一致。
10. hdfs_model_path / model_save_path 是否指向正确 checkpoint。
11. DENSE_FEA_SIZE / EMBEDDING_SIZE / hc_dim / hu_dim / emb_dim 是否和训练一致。
12. token_size 是否是 14。
13. token 下标是否仍是 pid=2、cost=10、treatment=11、label_dt=13。
14. Hive 推理脚本是否已经应用 slot 筛选。
15. startDate 是否是想写入的 Hive 分区。
16. 评估 SQL 是否直接使用 uplift_1~4，而不是重复加 left63。
17. label_dt 是否直接对齐 label 表 dt；当前 0623-0706 版本不需要 date_add。
18. Jupyter join 是否丢样本。
19. 成本小是否只是实际成本期望小，不要拿券面额直接对比。
20. 选 epoch 时确认是按 T1 AUUC、四组平均 AUUC、关键 treatment AUUC，还是 Jupyter 离线 AUUC。
```

模型/slot 不匹配的典型输入长度：

```text
baseline: 473 slots -> 946 输入长度
no_sirius: 279 slots -> 558 输入长度
replace_order: 372 slots -> 744 输入长度
all_superset: 609 slots -> 1218 输入长度
```

如果看到：

```text
expected shape=(None, 946), found shape=(20480, 1218)
```

就是 baseline 473-slot 模型吃到了全量 609-slot 输入。

## 15. 以后我应该如何快速恢复上下文

建议按这个顺序读：

1. 本文件：`wwt_euen_feature_select_full_flow_memo.md`。
2. `demo/feature_select/euen_4t_feature_code_readme.md`：理解训练脚本。
3. `demo/feature_select/test_hive_change_guide.md`：理解推理写 Hive。
4. `offline_bigdata/script/sqls/baseline/feature_baseline_test.sql`：确认最新数据口径。
5. `jupyter/模型评估_self.ipynb`：确认 Hive 拼 label 和 AUUC 评估。
6. `jupyter/运筹仿真_self.ipynb`：确认最终策略选择和 stage 对比。

一句话主线：

```text
写 SQL -> 配 parser/token/lisan_fea -> gen.sh 生成 feature.conf/parser.conf -> extract.sh/extract_test.sh 落 HDFS TFRecord -> hdfs dfs -get 到本地/NFS train/test 路径 -> build_feature_slot 生成 slots -> euen_4t_feature 训练一套或多套模型 -> test_hive 按模型+slot 推理写 Hive -> 模型评估 notebook 拼 label 算 AUUC -> 运筹 notebook 按实际 CAC 区间选 scheme/stage/策略点。
```
