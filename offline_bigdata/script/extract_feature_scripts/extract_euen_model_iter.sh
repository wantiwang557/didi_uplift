# 注意：运行需要在 extract_feature_scripts 目录下
# Usage:
#   ./extract_euen_model_iter.sh train
#   ./extract_euen_model_iter.sh test

exec_date_train='20260722'
exec_date_test='20260805'
run_mode='client'
feature_verson='euen_model_iter'
stage=${1:-train}

if [ "${stage}" != "train" ] && [ "${stage}" != "test" ]; then
    echo "Usage: $0 [train|test]"
    exit 1
fi

if [ "${stage}" = "train" ]; then
    exec_date=${exec_date_train}
else
    exec_date=${exec_date_test}
fi

# parser / feature 配置固定使用 train 版，保证 train/test 特征空间一致。
parser_stage='train'
parser_file="./feature_confs/${feature_verson}/parser_${feature_verson}_${parser_stage}.conf"
feature_list_file="./feature_confs/${feature_verson}/feature_${feature_verson}_${parser_stage}.conf"
sql_file="./sqls/${feature_verson}/feature_${feature_verson}_${stage}.sql"
hdfs_work_space="/user/prod_growstrategy/tmp/wwt/${feature_verson}/${stage}"
partition=100

cd ../

sh ./run_trans2tfr_spark32.sh ${exec_date} ${run_mode} ${parser_file} ${feature_list_file} ${sql_file} ${hdfs_work_space} ${partition}

echo "clean up..."
rm -f *.conf

# Train output:
# /user/prod_growstrategy/tmp/wwt/euen_model_iter/train/2026/07/22
# Test output:
# /user/prod_growstrategy/tmp/wwt/euen_model_iter/test/2026/08/05
