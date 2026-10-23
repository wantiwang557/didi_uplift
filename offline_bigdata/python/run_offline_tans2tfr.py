# -*- coding: utf-8 -*-
from string import Template
from pathlib import Path
import argparse
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)


def init_spark(args):
    """
    加载Spark模块，支持直接加载和findspark两种方式
    """

    try:
        import pyspark
        load_spark_flag = "pyspark"

    except ModuleNotFoundError:

        try:
            import findspark
            findspark.init()
            load_spark_flag = "findspark"

        except ModuleNotFoundError:
            load_spark_flag = None

    if load_spark_flag is not None:
        # 加载成功
        logging.info(
            "Load pyspark success from '{}'.".format(load_spark_flag))
    else:
        # 加载失败报错
        raise ModuleNotFoundError(
            "Can not find model 'pysaprk' or 'findspark'! GG!")

    from pyspark.sql import SparkSession

    # 设置 spark client 模式下的 python环境，防止用错
    if args.python_env is not None:
        os.environ["PYSPARK_PYTHON"] = args.python_env

    # spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark=SparkSession.builder.master('yarn') \
    .config("spark.diver.bindAddress",os.environ.get('SPARK_LOCAL_HOSTNAME')) \
    .enableHiveSupport() \
    .getOrCreate()
    return spark


# def get_map_func(total_columns_bc, parser_file_name_bc):

#     def map_func(rdd):

#         from uuid import uuid4
#         import subprocess

#         # 0. 生成临时文件标识
#         lite_fe_id = uuid4().hex
#         input_file = "input_liteFeExtra.{}.tmp".format(lite_fe_id)
#         output_file = "output_liteFeExtra.{}.tmp".format(lite_fe_id)

#         # 1. 数据格式转换 -> \t分割字符串
#         res = []
#         for row in rdd:

#             val_list = []
#             for col in total_columns_bc.value:

#                 _value = row[col]

#                 if _value is None:
#                     value = "_NIL"

#                 else:
#                     value = str(_value)

#                 val_list.append(value)

#             _res = "{}\n".format("\t".join(val_list))
#             res.append(_res)

#         # 2. 写入worker本地文件
#         with open(input_file, "w+") as f:
#             f.writelines(res)

#          ## 看res是否有数据
#         if res is not None:
#             res_len = len(res) if hasattr(res, '__len__') else "unknown (iterator)"
#             print(f"DEBUG: res type: {type(res)}")
#             print(f"DEBUG: res count: {res_len}")
            
#             # 如果是列表等可切片对象，打印前3条看看样子
#             if hasattr(res, '__getitem__') and res_len != 0:
#                 print(f"DEBUG: res sample (first 3): {res[:3]}")
#             elif res_len == 0:
#                 print("DEBUG: res is EMPTY (length 0)")
#         else:
#             print("DEBUG: res is None")

#         del res # 删除变量，释放一部分内存

#         # 3. 处理特征，结果写入worker本地
#         cmd = "cat {input_file} | ./fea_extract {cfg_file} > {output_file}".format(
#             input_file = input_file,
#             output_file = output_file,
#             cfg_file = parser_file_name_bc.value
#         )
#         rc, out = subprocess.getstatusoutput(cmd)

#         # 4. 处理结果检测 return code != 0 -> 报错
#         with open(output_file, "r") as f:
#             libsvm_list = f.read().strip().split("\n")  # 读出来去掉多写入的一个\n && 分割

#         if rc != 0:
#             # raise ValueError("GG!\n{}".format(libsvm_list))
#             # 把 res 的长度和样例直接塞进报错信息里
#             debug_info = "res_len: {}, sample: {}".format(len(res), res[:3] if res else "empty")
#             raise ValueError("GG! cmd_rc: {}\nDebug: {}\nOutput: {}".format(rc, debug_info, libsvm_list))

#         # 5. 输出结果整合
#         ret_items = []
#         for libsvm in libsvm_list:

#             label_feature_token = libsvm.strip().split("#")

#             _label_fe = label_feature_token[0]
#             label_fe = _label_fe.strip().split(" ")
#             label = label_fe[0].strip()
#             feature = ",".join(label_fe[1:])

#             if len(label_feature_token) == 1:
#                 token = []
#             else:
#                 token = label_feature_token[1].strip().replace("\t", ",")

#             ret_items.append([label, feature, token])

#         return ret_items

#     return map_func
def get_map_func(total_columns_bc, parser_file_name_bc):

    def map_func(rdd):
        from uuid import uuid4
        import subprocess
        import os

        # 0. 生成临时文件标识
        lite_fe_id = uuid4().hex
        input_file = "input_liteFeExtra.{}.tmp".format(lite_fe_id)
        output_file = "output_liteFeExtra.{}.tmp".format(lite_fe_id)
        
        # 用于调试的计数器和样本
        count = 0
        sample_data = []

        try:
            # 1 & 2. 数据格式转换并直接流式写入文件 (优化内存)
            with open(input_file, "w") as f:
                for row in rdd:
                    val_list = []
                    for col in total_columns_bc.value:
                        _value = row[col]
                        value = "_NIL" if _value is None else str(_value)
                        val_list.append(value)
                    
                    line = "{}\n".format("\t".join(val_list))
                    f.write(line)
                    
                    # 记录调试信息
                    count += 1
                    if count <= 3:
                        sample_data.append(line.strip())

            # 打印调试信息
            print(f"DEBUG: Processed {count} rows for feature extraction.")
            if count == 0:
                print("DEBUG: Warning - res is EMPTY (no data in this partition)")

            # 3. 处理特征，结果写入worker本地
            # 注意：确保 ./fea_extract 有执行权限
            cmd = "cat {input_file} | ./fea_extract {cfg_file} > {output_file}".format(
                input_file = input_file,
                output_file = output_file,
                cfg_file = parser_file_name_bc.value
            )
            rc, out = subprocess.getstatusoutput(cmd)

            # 4. 处理结果检测
            if not os.path.exists(output_file):
                raise ValueError(f"GG! Output file not found. cmd_rc: {rc}, out: {out}")

            with open(output_file, "r") as f:
                content = f.read().strip()
                libsvm_list = content.split("\n") if content else []

            if rc != 0:
                # 此时 res 不存在了，我们改用记录好的 count 和 sample_data
                debug_info = "count: {}, sample: {}".format(count, sample_data)
                raise ValueError("GG! cmd_rc: {}\nDebug: {}\nError_Out: {}".format(rc, debug_info, out))
            if count > 0 and not libsvm_list:
                parser_path = parser_file_name_bc.value
                parser_head = ""
                feature_list_path = ""
                feature_list_exists = False
                feature_list_head = ""
                if os.path.exists(parser_path):
                    with open(parser_path, "r") as parser_f:
                        parser_lines = parser_f.read().splitlines()
                    parser_head = "\n".join(parser_lines[:5])
                    for parser_line in parser_lines:
                        if parser_line.startswith("feature_list="):
                            feature_list_path = parser_line.split("=", 1)[1].strip()
                            break
                if feature_list_path:
                    feature_list_exists = os.path.exists(feature_list_path)
                    if feature_list_exists:
                        with open(feature_list_path, "r") as feature_f:
                            feature_list_head = "\n".join(feature_f.read().splitlines()[:3])
                debug_info = {
                    "cwd": os.getcwd(),
                    "files": sorted(os.listdir("."))[:30],
                    "input_count": count,
                    "input_sample_prefix": sample_data[:1],
                    "cmd": cmd,
                    "cmd_rc": rc,
                    "cmd_out": out,
                    "parser_exists": os.path.exists(parser_path),
                    "parser_head": parser_head,
                    "feature_list_path": feature_list_path,
                    "feature_list_exists": feature_list_exists,
                    "feature_list_head": feature_list_head,
                }
                raise ValueError("fea_extract输出为空，但输入partition非空: {}".format(debug_info))

            # 5. 输出结果整合
            ret_items = []
            for libsvm in libsvm_list:
                if not libsvm.strip(): continue
                
                label_feature_token = libsvm.strip().split("#")
                _label_fe = label_feature_token[0]
                label_fe = _label_fe.strip().split(" ")
                label = label_fe[0].strip()
                feature = ",".join(label_fe[1:])

                if len(label_feature_token) == 1:
                    token = "" # 建议用空字符串而不是空列表，保持类型一致
                else:
                    token = label_feature_token[1].strip().replace("\t", ",")

                ret_items.append([label, feature, token])

            return ret_items

        finally:
            # 6. 清理临时文件 (非常重要！)
            for f_path in [input_file, output_file]:
                if os.path.exists(f_path):
                    os.remove(f_path)

    return map_func



def get_sql(args):

    sql_params = {}

    if args.sql_param is not None:
        _param_list = [
            param.split("=") for param in args.sql_param.strip().split(";")
        ]

        sql_params = {
            k.strip(): v.strip() for (k, v) in _param_list
        }

    sql = Template(
        Path(args.sql_file).read_text()
    ).substitute(
        sql_params
    )

    return sql


def parser_args():

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--sql_file",   type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--sql_param",  type=str, required=False, default=None)
    parser.add_argument("--partition",  type=int, required=False, default=100)
    parser.add_argument("--python_env", type=str, required=False, default=None)


    return parser.parse_args()


def run():

    # 1. parser feature columns
    args = parser_args()
    spark = init_spark(args)

    parser_file = Path(args.config)
    parser_file_name = parser_file.name

    for line in parser_file.read_text().split("\n"):
        check_line = line.split("=")
        if check_line[0].strip() == "fea_cols":
            columns = [
                col.strip()
                for col in check_line[1].strip().split(",")
            ]
            break
    else:
        # for 正常执行完的话报错
        raise ValueError("配置错误，无配置项'fea_cols'")

    # 2. 执行SQL + 检查
    sql = get_sql(args)

    logging.info("执行特征SQL:\n{}".format(sql))
    logging.info("最终结果将写入 {}".format(args.output))

    df = spark.sql(sql)
    raw_count = df.count()
    logging.info("原始SQL结果行数: {}".format(raw_count))
    if raw_count == 0:
        raise ValueError("原始SQL结果为空，请先检查SQL过滤条件、label表数据和join条件。")

    tfrecord_fmat = 'tfrecords'
    spark_version = spark.version
    if spark_version.startswith('2'):
        logging.info("Spark版本: {}, using tfrecords format: {}".format(spark_version, tfrecord_fmat))
    else:
        tfrecord_fmat = 'tfrecord'
        logging.info("Spark版本: {}, using tfrecords format: {}".format(spark_version, tfrecord_fmat))
        

    # 检查列名
    check_set = set(columns) - set(df.columns)
    assert not check_set, "如下特征列不在目标DataFrame中:\n > {}".format(
        "\n > ".join([i for i in check_set])
    )

    # 3. 特征转换
    total_columns_bc =  spark.sparkContext.broadcast(columns)
    parser_file_name_bc = spark.sparkContext.broadcast(parser_file_name)
    map_func = get_map_func(total_columns_bc, parser_file_name_bc)

    logging.info("全部特征列: {}".format(total_columns_bc.value))
    logging.info("特征配置文件: {}".format(parser_file_name_bc.value))

    from pyspark.sql.types import StructType, StructField
    from pyspark.sql.types import StringType

    schema = StructType([
        StructField("label",   StringType()),
        StructField("feature", StringType()),
        StructField("token",   StringType())
    ])

    res_df = df.repartition(
        args.partition
    ).rdd.mapPartitions(
        map_func
    ).toDF(
        schema
    ).cache()

    res_df.printSchema()
    converted_count = res_df.count()
    logging.info("特征转换后行数: {}".format(converted_count))
    if converted_count == 0:
        raise ValueError("特征转换后结果为空，请检查fea_extract、parser.conf和feature_list配置。")

    # 4. 落盘
    # py4j.protocol.Py4JJavaError: An error occurred while calling o161.save.
    res_df.repartition(
        args.partition
    ).write.mode(
        "overwrite"
    ).format(
        tfrecord_fmat
    ).save(
        args.output
    )

    logging.info("All is Done.")


if __name__ == "__main__":
    run()