

#exec_date='20250714'
#exec_date='20260716'
exec_date='20260813' # 新的testl路径
run_mode='client'

y=$(date -d "$exec_date" +%Y)
m=$(date -d "$exec_date" +%m)
d=$(date -d "$exec_date" +%d)

#feature_verson='25_v4'
feature_verson='v3'
#feature_verson='v216'
stage='train'   # 用的是train的配置信息


parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${stage}.conf"

stage='test'   # sql和写入的目录用test的
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

cd ../

#cp -r ./feature_confs/${feature_verson}/*.conf ./

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}


echo "clean up..."
rm *.conf
# yarn application -kill application_xxxx_xxxx
# jobs -l
# cd wwt/offline_bigdata/script/extract_feature_scripts/
# chmod +x extract_test.sh
# nohup ./extract_test.sh > extract_test_v3.log 2>&1 &
# tail -f extract_test_v3.log
# grep -o "application_[0-9_]*" extract_test_v0.log | tail -1
# yarn application -status application_xxx



# 最终结果将写入 /user/prod_growstrategy/tmp/wwt/baseline/test/2026/07/22
# 检查文件是否存在
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13
# 检查文件数量
# hdfs dfs -ls /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/part-* | wc -l
# 打开存储目录
# cd /nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/
# # 下载测试集到目标本地文件夹
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v2/test/2026/08/13/* wwt/v2/test_20260623_20260706
# hdfs dfs -get /user/prod_growstrategy/tmp/wwt/v3/test/2026/08/13/* wwt/v3/test_0723_0806
# 本地统计part分片  ls wwt/v3/test_0723_0806/part-* | wc -l