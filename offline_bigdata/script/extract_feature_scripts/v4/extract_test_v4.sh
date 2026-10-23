#!/bin/bash
# 注意：运行需要在 extract_feature_scripts 目录下。
# 用途：抽取 v4 test TFRecord。
# test 窗口：2026-06-23 ~ 2026-07-06。
# feature/parser：使用 v4 train 配置，保证 train/test 特征空间一致。

exec_date='20260813'
run_mode='client'
feature_verson='v4'

# parser / feature_list 固定使用 train 版配置。
stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${stage}.conf"

# SQL 和写入目录使用 test。
stage='test'
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

cd ../

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}

echo "clean up..."
rm -f *.conf

# 运行方式：
# cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
# chmod +x v4/extract_test_v4.sh
# nohup ./v4/extract_test_v4.sh > extract_test_v4.log 2>&1 &
# tail -f extract_test_v4.log

# 最终结果写入：
# /user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13

# 检查文件是否存在：
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13

# 检查文件数量：
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13/part-* | wc -l

# 打开本地存储目录：
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/

# 下载数据集到目标本地文件夹：
# mkdir -p wwt/v4/test_0623_0706
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v4/test/2026/08/13/* wwt/v4/test_0623_0706/

# 本地统计 part 分片：
# ls wwt/v4/test_0623_0706/part-* | wc -l

# 查询 app_id：
# grep -o "application_[0-9_]*" extract_test_v4.log | tail -1

# 杀死 Yarn 任务：
# yarn application -kill application_xxx
