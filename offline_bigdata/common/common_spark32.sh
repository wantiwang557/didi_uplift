#!/bin/bash

# 项目根目录名
PROJECT_NAME="offline_bigdata"

# 向上确定绝对路径
BASE_DIR="$(pwd)"
dir_name=""
while [ "${dir_name}" != "/" ]
    do
        dir_name=$(basename "${BASE_DIR}")

        if [[ "${dir_name}" == "${PROJECT_NAME}" ]]; then
            break
        fi

        BASE_DIR=$(dirname "${BASE_DIR}")
    done

# 基于绝对路径确必要目录
BASE_DIR_BIN="${BASE_DIR}/bin"
BASE_DIR_SCRIPT="${BASE_DIR}/script"
BASE_DIR_CONFIG="${BASE_DIR}/config"
BASE_DIR_PYTHON="${BASE_DIR}/python"

# 设置运行的Python环境
PYTHON_ARCHIVES="hdfs://difed/user/dm/ENV/dlflow/v2/py3.7.15_for_tfcluster.tgz"
PYTHON_SPARK_ENV="./py3.7.15_for_tfcluster.tgz/py3.7.15_for_tfcluster/bin/python3"

# 设置运行队列
YARN_QUEUE="root.growstrategy_group_prod"

# 必要的jar包依赖，以支持spark读写 tfrecords
DEPEND_JARS="hdfs://difed/user/dm/ENV/jars/spark_tfrecord/compile/spark-tfrecord-0.5.1_scala2.12-spark3.2.0-tfhp1.15.0.jar"

# 数梦调度任务部署目录
HDFS_ONLINE_DEPLOY="/user/prod_growstrategy/tmp/lite_fea_extra_demo/online/offline_bigdata"