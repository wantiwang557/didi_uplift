# 注意：运行需要在 extract_feature_scripts 目录下
#exec_date='20250714'
exec_date='20260813' # 和gen.sh保持一致。
#exec_date='20250709'
run_mode='client'

y=$(date -d "$exec_date" +%Y)
m=$(date -d "$exec_date" +%m)
d=$(date -d "$exec_date" +%d)

# 注意版本
#feature_verson='25_v4'
#feature_verson='v0'
feature_verson='v2'

# ------------------------------------------------------------------------------------------------#
# feature parser stage
# parser stage 如果没有额外生成不需要修改
stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${stage}.conf"

# ------------------------------------------------------------------------------------------------#
# 从 hive 表拉取数据
# stage 需要修改
# partition 数据量较大时可以修改

stage='train'
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"

hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

# 回到 script 目录下
cd ../

# 注释目的：
#cp -r ./feature_confs/${feature_verson}/*.conf ./

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}


echo "clean up..."
#rm *.conf

# 最终结果将写入 /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16
# 打开存储目录
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/

# 检查文件是否存在
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16
#检查文件数量
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16/part-* | wc -l
# 打开存储目录
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/
# # 下载测试集到目标本地文件夹
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/baseline/train/2026/07/16/* wwt/baseline/train/
# 本地统计part分片  ls wwt/baseline/train/part-* | wc -l







# 创建训练和测试文件夹
# mkdir 文件夹名字 (可以按照别人的设计，分别是train 和valid两个相关文件夹)
# cd 对应目录
# 拉取hdfsyarn application -status 
# hdfs dfs -get /user/prod_growstrategy/wwt/v1/train/2025/10/17/* 
# hdfs dfs -get /user/prod_growstrategy/wwt/v1/test/2026/07/16/*
# hdfs dfs -rm -r /user/prod_growstrategy/tmp/wwt/26_v2/train/*  只删除该目录下的文件


# chmod +x extract.sh
# nohup ./extract.sh > extract_train_baseline.log 2>&1 &
# nohup ./extract.sh > extract_train_v2.log 2>&1 &
# tail -f extract_train_v2.log
