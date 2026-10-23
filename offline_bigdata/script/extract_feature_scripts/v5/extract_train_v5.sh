#!/bin/bash
# 注意：运行需要在 extract_feature_scripts 目录下。
# 用途：抽取 v5 train TFRecord。
#
# train 窗口：2026-05-11 ~ 2026-08-21。
# 日期排除：2026-06-18 ~ 2026-06-22，2026-08-02，2026-08-08，2026-08-09。
# exec_date 和 gen_feature_conf_v5.sh 的 exec_dt 保持一致。

exec_date='20260920'
run_mode='client'
feature_verson='v5'

# parser / feature 使用 v5 train 配置。
stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${stage}.conf"

# SQL 和写入目录使用 v5 train。
stage='train'
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

cd ../

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}

echo "clean up..."
rm -f *.conf

# 运行方式：
# cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
# chmod +x v5/extract_train_v5.sh
# nohup ./v5/extract_train_v5.sh > extract_train_v5.log 2>&1 &
# tail -f extract_train_v5.log
#
# 输出路径：
# /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20
#
# 检查：
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20/part-* | wc -l
#
# 下载：
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/
# mkdir -p wwt/v5/train_0511_0821
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20/* wwt/v5/train_0511_0821/
