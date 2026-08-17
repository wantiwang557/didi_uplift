# 注意：运行需要在 extract_feature_scripts 目录下
# 用途：拉取 v2 的第二份训练集，SQL 为 feature_v2_2_train.sql
# train 窗口：2026-05-11 ~ 2026-06-17 + 2026-06-23 ~ 2026-07-22
# 依赖：需要先运行 feature_confs/v2/gen_feature_conf_v2_2.sh，
#      生成 feature_confs/v2/feature_v2_2_train.conf 和 parser_v2_2_train.conf。

exec_date='20260813'
run_mode='client'
feature_verson='v2'
sufix='_2'
stage='train'

# parser / feature 配置使用 v2_2 train 生成的配置。
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}${sufix}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}${sufix}_${stage}.conf"

sql_file="./sqls/${feature_verson}/feature_${feature_verson}${sufix}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/v2_2_${stage}"
partition=100

cd ../

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}

echo "clean up..."
rm -f *.conf

# 运行方式：
# cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/extract_feature_scripts
# chmod +x extract_v2_2_train.sh
# nohup ./extract_v2_2_train.sh > extract_v2_2_train.log 2>&1 &
# tail -f extract_v2_2_train.log

# 最终结果写入：
# /user/prod_growstrategy/tmp/wwt/v2/v2_2_train/2026/08/13

# 打开存储目录：
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/

# 检查文件是否存在：
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/v2_2_train/2026/08/13

# 检查文件数量：
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/v2_2_train/2026/08/13/part-* | wc -l

# 下载数据集到目标本地文件夹：
# mkdir -p wwt/v2/v2_2_train
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v2/v2_2_train/2026/08/13/* wwt/v2/v2_2_train/

# 本地统计 part 分片：
# ls wwt/v2/v2_2_train/part-* | wc -l
