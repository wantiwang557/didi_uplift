#!/bin/bash
# 注意：运行需要在 extract_feature_scripts 目录下。
# 用途：抽取 v5 train/valid/test TFRecord。
#
# v5 数据窗口：
#   train: 2026-05-11 ~ 2026-08-21
#   valid: 2026-08-22 ~ 2026-08-27
#   test : 2026-08-28 ~ 2026-09-10
#
# 日期排除：
#   2026-06-18 ~ 2026-06-22
#   2026-08-02, 2026-08-08, 2026-08-09
#
# feature/parser：固定使用 v5 train 配置，保证 train/valid/test 特征空间一致。
# exec_date：和 gen_feature_conf_v5.sh 的 exec_dt 保持一致，用于输出 HDFS 分区和 BIZ_DATE_LINE 参数；
#            不按 train/valid/test 的日期窗口分别变化。
#
# 运行方式：
#   cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
#   chmod +x v5/extract_v5.sh
#   nohup ./v5/extract_v5.sh train > extract_v5_train.log 2>&1 &
#   nohup ./v5/extract_v5.sh valid > extract_v5_valid.log 2>&1 &
#   nohup ./v5/extract_v5.sh test  > extract_v5_test.log 2>&1 &

exec_date='20260920'
run_mode='client'
feature_verson='v5'
stage=${1:-train}

if [ "${stage}" != "train" ] && [ "${stage}" != "valid" ] && [ "${stage}" != "test" ]; then
    echo "Usage: $0 [train|valid|test]"
    exit 1
fi

# parser / feature_list 固定使用 train 版配置。
parser_stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${parser_stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${parser_stage}.conf"

# SQL 和写入目录按 stage 区分。
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

cd ../

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}

echo "clean up..."
rm -f *.conf

# 输出路径：
#   train: /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20
#   valid: /user/prod_growstrategy/tmp/wwt/v5/valid/2026/09/20
#   test : /user/prod_growstrategy/tmp/wwt/v5/test/2026/09/20
#
# 检查示例：
#   hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/${stage}/$(date --date="${exec_date}" +%Y/%m/%d)/part-* | wc -l
#   hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20/part-* | wc -l
#   hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/test/2026/09/20/part-* | wc -l
#   hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v5/valid/2026/09/20/part-* | wc -l

#
# 下载示例：
#   cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/
#   mkdir -p wwt/v5/train_0511_0821
#   mkdir -p wwt/v5/valid_0822_0827
#   mkdir -p wwt/v5/test_0828_0910

# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v5/${stage}/$(date --date="${exec_date}" +%Y/%m/%d)/* wwt/v5/${stage}/
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v5/train/2026/09/20/* wwt/v5/train_0511_0821
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v5/valid/2026/09/20/* wwt/v5/valid_0822_0827
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v5/test/2026/09/20/* wwt/v5/test_0828_0910

#
# 查询 app_id：
#   grep -o "application_[0-9_]*" extract_v5_${stage}.log | tail -1
#
# 杀死 Yarn 任务：
#   yarn application -kill application_xxx