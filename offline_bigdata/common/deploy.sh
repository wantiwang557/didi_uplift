#!/bin/bash

is_overwrite=$1

source ./common.sh

# base deploy directory
DEPLOY_DIR="${BASE_DIR}/deploy"
if [ -d "${DEPLOY_DIR}" ]; then
    echo "Deploy directory already exists, it will be overwritten!"
    echo "Overwriting ${DEPLOY_DIR}"
    rm -rf "${DEPLOY_DIR}"
fi

DEPLOY_DIR="${DEPLOY_DIR}/${PROJECT_NAME}"
mkdir -p "${DEPLOY_DIR}"


# === ===  === ===
# 离线任务部署需要的模块
sub_modules="
bin
python
script
common
"

#
for sub_module in ${sub_modules};
do
  echo "Copying '${sub_module}' to ${DEPLOY_DIR}"
  cp -r "${BASE_DIR}/${sub_module}" "${DEPLOY_DIR}"
done

# 删除临时文件和目录
function delete()
{
    obj=$1
    echo "[DELETE] ${obj}"
    rm -rf "${obj}"
}
xargs_delete_cmd="echo '[DELETE] {}'; rm -rf {}"

delete "${DEPLOY_DIR}/common/deploy.sh"
find "${DEPLOY_DIR}" -name "__pycache__" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "*.pyc" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "*.zip" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "*.out" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "*.log" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "*.tmp_data" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "debug" | xargs -i sh -c "${xargs_delete_cmd}"
find "${DEPLOY_DIR}" -name "local_run_*.sh" | xargs -i sh -c "${xargs_delete_cmd}"


# 替换反引号
find "${DEPLOY_DIR}" -name "*.sh" | xargs sed -i 's/`/\\`/g'
find "${DEPLOY_DIR}" -name "*.sql" | xargs sed -i 's/\\`/\\\\\\`/g'
# ===========================================================================
# 由于数梦任务先启动一个docker，然后再spark-submit到am上，因此特殊字符过滤略微复杂
# 对于反引号 ` ，大致两种过滤规则如下：
# 1. 单引号字符串中反引号 : foo_str='whatever `'
#    * 本地无需转义
#    * 数梦执行加两个反斜杠转义，写成 foo_str='whatever \\`'
# 2. 双引号字符串中反引号 : foo_str="whatever `"
#    * 本地执行需要写成 foo_str="whatever \`"
#    * 数梦执行需要写成 foo_str="whatever \\\`"
# ===========================================================================


hdfs_online_deploy="${HDFS_ONLINE_DEPLOY}"
function deploy2HDFS()
{
    # 检查部署目录是否已经存在
    hadoop fs -test -e "${hdfs_online_deploy}"
    if [ $? -eq 0 ]; then
        echo -e "\033[32m [WARNING] dest hdfs dir exists: ${hdfs_online_deploy} \033[0m "
        if [ $is_overwrite = "--overwrite" ]
        then
            echo -e "\033[31m [INFO] using --overwrite, it will be overwrite \033[0m "
            hadoop fs -rm -r "${hdfs_online_deploy}"
            hadoop fs -mkdir -p "${hdfs_online_deploy}"
            hadoop fs -put ${DEPLOY_DIR}/* "${hdfs_online_deploy}"
        else
            echo "[INFO] only update local deploy_temp, do not update HDFS of capricornus"
        fi
    else
        echo "[INFO] deploy to HDFS of capricornus first time!"
        hadoop fs -mkdir -p "${hdfs_online_deploy}"
        hadoop fs -put ${DEPLOY_DIR}/* "${hdfs_online_deploy}"
    fi
}

deploy2HDFS

echo "DONE."
