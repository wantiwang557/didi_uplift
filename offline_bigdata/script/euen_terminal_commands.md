# EUEN/EEUEN 常用终端命令

本文档用于记录 EUEN/EEUEN 数据生成、特征抽取、训练任务排查时常用的终端命令。

## 常用路径

```bash
# 代码主目录
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt

# 生成 feature.conf / parser.conf
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/v2
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/v3

# 抽取 TFRecord
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts

# 本地 TFRecord 缓存目录
cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203

# EUEN 训练目录
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/euen_test_feature_time

# EEUEN 训练目录
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EEUEN
```

## 查询日志

```bash
# 实时查看 feature_conf 生成日志
tail -f feature_conf_v2.log
tail -f feature_conf_v3.log

# 实时查看 extract 日志
tail -f extract_train_v2.log
tail -f extract_train_v3.log
tail -f extract_test_v2.log
tail -f extract_test_v3.log

# 实时查看 EUEN 训练日志
tail -f ./run_logs/euen_v2_train_0511_0617.log
tail -f ./run_logs/euen_v3_train_0511_0722.log

# 实时查看 EEUEN 训练日志
tail -f ./run_logs/eeuen_v2_train_0511_0617.log
tail -f ./run_logs/eeuen_v3_train_0511_0722.log

# 看日志最后 100 行
tail -100 ./run_logs/euen_v3_train_0511_0722.log

# 看日志开头 50 行
head -50 ./run_logs/euen_v3_train_0511_0722.log

# 搜索报错
grep -n "ERROR\\|Exception\\|Traceback\\|Permission denied\\|ValueError" ./run_logs/euen_v3_train_0511_0722.log

# 搜索关键指标
grep -n "mean_auuc\\|auuc_\\|Best mean\\|early_stop" ./run_logs/eeuen_v3_train_0511_0722.log
grep -n "outcome_loss_mean\\|cost_loss_mean\\|exposure_loss_mean" ./run_logs/eeuen_v3_train_0511_0722.log
grep -n "exposure_auc\\|mean_auuce\\|mean_qinie" ./run_logs/eeuen_v3_train_0511_0722.log
```

## 检查 HDFS 文件是否存在

```bash
# v2 train: 0511-0617
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13

# v3 train: 0511-0617 + 0623-0722
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13

# v2 test: 0723-0806, 排除 0802
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13

# v3 test: 0723-0806, 排除 0802
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13

# 检查 _SUCCESS
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13/_SUCCESS
```

## 查询 HDFS 文件数量

```bash
# 统计 part 分片数量
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13/part-* | wc -l
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13/part-* | wc -l
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/part-* | wc -l
hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/part-* | wc -l

# 查看文件大小
hdfs dfs -du -h /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13
```

## 下载文件到本地

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203

# 创建本地目录
mkdir -p wwt/v2/train_0511_0617
mkdir -p wwt/v3/train_0511_0722
mkdir -p wwt/v2/test_0723_0806
mkdir -p wwt/v3/test_0723_0806

# 下载 v2 train
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v2/train/2026/08/13/* wwt/v2/train_0511_0617/

# 下载 v3 train
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13/* wwt/v3/train_0511_0722/

# 下载 v2 test
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/* wwt/v2/test_0723_0806/

# 下载 v3 test
hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/* wwt/v3/test_0723_0806/
```

如果本地目录已经存在同名文件，`hdfs dfs -get` 可能失败。可先换新目录，或确认无误后再清理旧目录。

```bash
# 谨慎执行：删除本地旧缓存
rm -rf wwt/v3/train_0511_0722
mkdir -p wwt/v3/train_0511_0722
```

## 本地统计分片

```bash
cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203

ls wwt/v2/train_0511_0617/part-* | wc -l
ls wwt/v3/train_0511_0722/part-* | wc -l
ls wwt/v2/test_0723_0806/part-* | wc -l
ls wwt/v3/test_0723_0806/part-* | wc -l

# 查看本地目录大小
du -sh wwt/v3/train_0511_0722

# 查看前几个文件
ls -lh wwt/v3/train_0511_0722 | head
```

## 查询 Yarn App ID

适用于 `gen_feature_conf_*.sh` 和 `extract_*.sh` 这类 Spark/Yarn 任务。

```bash
# feature_conf 任务
grep -o "application_[0-9_]*" feature_conf_v2.log | tail -1
grep -o "application_[0-9_]*" feature_conf_v3.log | tail -1

# extract 任务
grep -o "application_[0-9_]*" extract_train_v2.log | tail -1
grep -o "application_[0-9_]*" extract_train_v3.log | tail -1
grep -o "application_[0-9_]*" extract_test_v2.log | tail -1
grep -o "application_[0-9_]*" extract_test_v3.log | tail -1

# 查询 Yarn 任务状态
yarn application -status application_xxx_xxx
```

如果训练日志里没有 app id，通常说明它是普通 Python/TensorFlow 进程，不是 Yarn application。

## 杀死 Yarn 任务

```bash
# 手动 kill
yarn application -kill application_xxx_xxx

# 从日志提取最后一个 app id 并 kill
yarn application -kill $(grep -o "application_[0-9_]*" extract_test_v3.log | tail -1)

# feature_conf v3
yarn application -kill $(grep -o "application_[0-9_]*" feature_conf_v3.log | tail -1)
```

如果提示 `Application with id ... doesn't exist in RM`，通常表示任务已经结束、失败退出，或 app id 不在当前 RM 上。

## 查询 Python 训练任务

适用于 `python euen_4t.py`、`python EEUEN_4t.py` 这类非 Yarn 训练任务。

```bash
# 查询所有 EUEN/EEUEN Python 训练进程
ps -ef | grep "euen_4t.py\\|EEUEN_4t.py" | grep -v grep

# 查询 v2 EUEN
ps -ef | grep "euen_4t.py" | grep "v2_train_0511_0617" | grep -v grep

# 查询 v3 EUEN
ps -ef | grep "euen_4t.py" | grep "v3_train_0511_0722" | grep -v grep

# 查询 v2 EEUEN
ps -ef | grep "EEUEN_4t.py" | grep "v2_train_0511_0617" | grep -v grep

# 查询 v3 EEUEN
ps -ef | grep "EEUEN_4t.py" | grep "v3_train_0511_0722" | grep -v grep
```

## 杀死 Python 训练任务

先查 PID：

```bash
ps -ef | grep "euen_4t.py" | grep "v3_train_0511_0722" | grep -v grep
```

输出第二列是 PID，例如：

```text
luban 2518904 ... python euen_4t.py ...
```

杀进程：

```bash
kill 2518904
```

确认是否已退出：

```bash
ps -ef | grep "euen_4t.py" | grep -v grep
```

如果确认是目标任务但普通 `kill` 没有停，再用：

```bash
kill -9 2518904
```

也可以一行 kill 指定版本任务。执行前建议先运行不带 `awk/xargs kill` 的查询命令确认匹配结果。

```bash
ps -ef | grep "euen_4t.py" | grep "v3_train_0511_0722" | grep -v grep | awk '{print $2}' | xargs kill
```

## 启动脚本参考

```bash
# 生成 v2 feature.conf / parser.conf
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/v2
chmod +x gen_feature_conf_v2.sh
nohup ./gen_feature_conf_v2.sh > feature_conf_v2.log 2>&1 &
tail -f feature_conf_v2.log

# 生成 v3 feature.conf / parser.conf
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/feature_confs/v3
chmod +x gen_feature_conf_v3.sh
nohup ./gen_feature_conf_v3.sh > feature_conf_v3.log 2>&1 &
tail -f feature_conf_v3.log

# 抽取 train
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
chmod +x extract.sh
nohup ./extract.sh > extract_train_v3.log 2>&1 &
tail -f extract_train_v3.log

# 抽取 test
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
chmod +x extract_test.sh
nohup ./extract_test.sh > extract_test_v3.log 2>&1 &
tail -f extract_test_v3.log
```

## 清理 HDFS 临时目录

谨慎使用，只在确认要重跑并覆盖旧结果时执行。

```bash
# 删除某个结果目录
hdfs dfs -rm -r /user/prod_growstrategy/tmp/wwt/v3/train/2026/08/13

# 只删除某版本 train 下的内容
hdfs dfs -rm -r /user/prod_growstrategy/tmp/wwt/v3/train/*
```
