#!/bin/bash


source ../common/common_spark32.sh

echo "PYTHON_ARCHIVES:" $PYTHON_ARCHIVES

exec_date=$1
run_mode=$2

# 运行所需的文件
EXEC_PY_FIEL="${BASE_DIR_PYTHON}/run_offline_tans2tfr.py"




BIN_EXTRACTOR_FILE="fea_extract"                            
PARSER_FILE=$3
FEATURE_LIST_FILE=$4
FEATURE_SQL_FILE=$5

HDFS_ONLINE_DEPLOY=$6
PARTITIONS=$7

# PARSER_FILE="./feature_confs/v0/parser_v0_train.conf"
# FEATURE_LIST_FILE='./feature_confs/v0/feature_v0_train.conf'
# FEATURE_SQL_FILE="./sqls/v0/feature_v0_train.sql"



echo "exec_date: " $exec_date
echo "run_mode: " $run_mode
echo "PARSER_FILE: " $PARSER_FILE
echo "FEATURE_LIST_FILE: " $FEATURE_LIST_FILE
echo "FEATURE_SQL_FILE: " $FEATURE_SQL_FILE
echo "PARTITIONS: " $PARTITIONS
echo "HDFS_ONLINE_DEPLOY: " $HDFS_ONLINE_DEPLOY
echo "EXEC_PY_FIEL: " $EXEC_PY_FIEL
echo "PARTITIONS: " $PARTITIONS


# spark运行模式
if [ -z "${exec_date}" ]; then
    exec_date=$(date --date="-1 day" +%Y%m%d)
    echo "Param [exec_date] not given, using default value: '${exec_date}'"
fi

if [ "${run_mode}" != "cluster" ]; then
    run_mode="client"
    echo "Param [run_mode] not given, using default value: '${run_mode}'"
fi


# 输出目录
SAVE_DIR=$(date --date="${exec_date}" +${HDFS_ONLINE_DEPLOY}/%Y/%m/%d)
echo "save dir: " $SAVE_DIR

# SQL 参数
sql_run_dt=$(date --date="${exec_date}" +%Y-%m-%d)

# Spark App Name
random_numver=$((RANDOM%9000+1000))
APP_NAME="lite_fea_extra_offline_bigdata_tans2tfr.${exec_date}.${random_numver}"
echo "APP NAME": $APP_NAME

# ==== ==== ==== ==== client 模式 ==== ==== ==== ==== #
function run_with_client_mode() {

    spark-submit \
        --master "yarn" \
        --queue "${YARN_QUEUE}" \
        --name "${APP_NAME}" \
        --driver-memory 4g \
        --executor-memory 20g \
        --archives "${PYTHON_ARCHIVES}" \
        --jars "${DEPEND_JARS}" \
        --conf "spark.submit.deployMode=client" \
        --conf "spark.dynamicAllocation.enabled=true" \
        --conf "spark.dynamicAllocation.minExecutors=260" \
        --conf "spark.dynamicAllocation.maxExecutors=500" \
        --conf "spark.executor.cores=4" \
        --conf "spark.yarn.executor.memoryOverhead=5g" \
        --conf "spark.sql.shuffle.partitions=2000" \
        --conf "spark.default.parallelism=4000" \
        --files "${BASE_DIR_BIN}/${BIN_EXTRACTOR_FILE},${BASE_DIR_SCRIPT}/${PARSER_FILE},${BASE_DIR_SCRIPT}/${FEATURE_LIST_FILE},${BASE_DIR_SCRIPT}/${FEATURE_SQL_FILE}" \
        ${EXEC_PY_FIEL} \
        --config "${BASE_DIR_SCRIPT}/${PARSER_FILE}" \
        --python_env "${PYTHON_SPARK_ENV}" \
        --partition "${PARTITIONS}" \
        --sql_file "${BASE_DIR_SCRIPT}/${FEATURE_SQL_FILE}" \
        --sql_param "BIZ_DATE_LINE=${sql_run_dt}" \
        --output "${SAVE_DIR}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] spark_submit job failed."
        exit -1
    fi
}
# ==== ==== ==== ==== ==== ==== ==== ==== #


# ==== ==== ==== ==== cluster 模式 ==== ==== ==== ==== #
function run_with_cluster_model() {

    spark-submit \
        --master "yarn" \
        --queue "${YARN_QUEUE}" \
        --name "${APP_NAME}" \
        --driver-memory 2g \
        --executor-memory 6g \
        --archives "${PYTHON_ARCHIVES}" \
        --jars "${DEPEND_JARS}" \
        --conf "spark.yarn.dist.archives=${PYTHON_ARCHIVES}" \
        --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYTHON_SPARK_ENV}" \
        --conf "spark.yarn.appaMasterEnv.PYSPARK_DRIVER_PYTHON=${PYTHON_SPARK_ENV}" \
        --conf "spark.executorEnv.PYSPARK_PYTHON=${PYTHON_SPARK_ENV}" \
        --conf "spark.submit.deployMode=cluster" \
        --conf "spark.dynamicAllocation.enabled=true" \
        --conf "spark.dynamicAllocation.minExecutors=100" \
        --conf "spark.dynamicAllocation.maxExecutors=200" \
        --conf "spark.executor.cores=2" \
        --conf "spark.sql.shuffle.partitions=200" \
        --conf "spark.default.parallelism=200" \
        --files "${BASE_DIR_BIN}/${BIN_EXTRACTOR_FILE},${BASE_DIR_SCRIPT}/${PARSER_FILE},${BASE_DIR_SCRIPT}/${FEATURE_LIST_FILE},${BASE_DIR_SCRIPT}/${FEATURE_SQL_FILE}" \
        ${EXEC_PY_FIEL} \
        --config "${PARSER_FILE}" \
        --partition "10" \
        --sql_file "${FEATURE_SQL_FILE}" \
        --sql_param "exec_dt=${sql_run_dt}" \
        --output "${SAVE_DIR}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] spark_submit job failed."
        exit -1
    fi
}
# ==== ==== ==== ==== ==== ==== ==== ==== #


if [ "${run_mode}" == "client" ]; then
    echo "Running with client mode."
    run_with_client_mode

elif [ "${run_mode}" == "cluster" ]; then
    echo "Running with cluster mode."
    run_with_cluster_model

else
    echo "Unknow parameter '${run_mode}'!"
fi