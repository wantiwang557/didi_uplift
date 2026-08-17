# 这段代码实现了一个功能，主要用于处理特征工程的流程，包括从数据中提取特征，执行特征转换并将结果写入目标文件中。
# 这段代码是一个完整的特征工程处理流程，包括从数据中提取特征、执行特征转换和将结果写入目标文件等功能。

from string import Template
from pathlib import Path
import argparse
import os
import logging

# 初始化日志配置，设置日志级别为INFO级别，定义日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

# init_spark函数用于初始化Spark会话，支持直接加载或使用findspark模块加载Spark
# 返回的是创建的Spark会话对象
def init_spark(args):
    """
    加载Spark模块，支持直接加载和findspark两种方式
    """

    # 尝试导入pyspark模块。如果成功，将load_spark_flag设置为"pyspark"，表示通过直接导入pyspark来加载Spark。
    try:
        import pyspark
        load_spark_flag = "pyspark"

    except ModuleNotFoundError: # 如果pyspark模块找不到，将执行以下try块中的代码。

        try:
            import findspark # 尝试导入findspark模块，并使用其init方法来初始化Spark。如果成功，将load_spark_flag设置为"findspark"，表示通过findspark来加载Spark
            findspark.init()
            load_spark_flag = "findspark"

        except ModuleNotFoundError:
            load_spark_flag = None # 如果findspark模块也找不到，将load_spark_flag设置为None。

    if load_spark_flag is not None: # 如果load_spark_flag不是None（即成功加载了Spark），则使用logging.info记录一条消息，说明是通过哪种方式加载的Spark
        # 加载成功
        logging.info(
            "Load pyspark success from '{}'.".format(load_spark_flag))
    else: # 如果load_spark_flag是None（即没有成功加载Spark），则抛出一个ModuleNotFoundError异常，并附带一条错误消息。注意这里有一个小错误，'pysaprk'应该是'pyspark'。
        # 加载失败报错 
        raise ModuleNotFoundError(
            "Can not find model 'pysaprk' or 'findspark'! GG!")

    from pyspark.sql import SparkSession # 导入SparkSession类，这是用于创建Spark会话的主要类

    # 设置 spark client 模式下的 python环境，防止用错
    if args.python_env is not None: # 如果args对象中的python_env属性不是None，则设置环境变量PYSPARK_PYTHON为该属性的值。这允许您指定Spark客户端模式下要使用的Python解释器
        os.environ["PYSPARK_PYTHON"] = args.python_env

    # 使用SparkSession.builder创建一个新的Spark会话构建器，并启用对Hive的支持（如果Hive可用）。
    # 然后，使用getOrCreate方法获取或创建一个新的Spark会话，并将其赋值给变量spark。
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    return spark # 返回创建的Spark会话对象


# 用于在 Spark 的 RDD（弹性分布式数据集）上进行转换操作。这个函数的主要目的是对一个 RDD 中的每一行数据进行处理，调用外部的特征提取脚本，并将结果转换为特定格式。
# total_columns_bc 可能包含要处理的 RDD 中每一行的所有列名，而 parser_file_name_bc 可能包含特征提取脚本的配置文件路径。
# 当您调用 get_map_func 时，它会返回 map_func 函数，然后您可以在 Spark RDD 上调用这个 map_func 函数来处理数据。
# map_func 会处理 RDD 中的每一行，并返回一个包含标签、特征和令牌的列表的 RDD。
def get_map_func(total_columns_bc, parser_file_name_bc):

    def map_func(rdd): # 这个函数接受一个 RDD 作为输入，并对其进行处理

        from uuid import uuid4
        import subprocess

        # 0. 生成临时文件标识
        lite_fe_id = uuid4().hex # 使用 uuid4().hex 生成一个唯一的 ID，用于创建临时输入和输出文件的名称
        input_file = "input_liteFeExtra.{}.tmp".format(lite_fe_id)
        output_file = "output_liteFeExtra.{}.tmp".format(lite_fe_id)

        # 1. 数据格式转换 -> \t分割字符串
        res = []
        for row in rdd: # 遍历 RDD 中的每一行，并根据 total_columns_bc.value 中的列名提取值

            val_list = []
            for col in total_columns_bc.value:

                _value = row[col]

                if _value is None: # 如果值为 None，则替换为 "_NIL"。
                    value = "_NIL"

                else:
                    value = str(_value)

                val_list.append(value) # 将所有值转换为字符串，并使用制表符（\t）分隔

            _res = "{}\n".format("\t".join(val_list)) # 然后添加到结果列表 res 中
            res.append(_res)

        # 2. 写入worker本地文件  将 res 列表中的字符串写入到临时输入文件 input_file 中
        with open(input_file, "w+") as f:
            f.writelines(res)
        ## 看res是否有数据
        if res is not None:
            res_len = len(res) if hasattr(res, '__len__') else "unknown (iterator)"
            print(f"DEBUG: res type: {type(res)}")
            print(f"DEBUG: res count: {res_len}")
            
            # 如果是列表等可切片对象，打印前3条看看样子
            if hasattr(res, '__getitem__') and res_len != 0:
                print(f"DEBUG: res sample (first 3): {res[:3]}")
            elif res_len == 0:
                print("DEBUG: res is EMPTY (length 0)")
        else:
            print("DEBUG: res is None")

        del res # 删除变量，释放一部分内存

        # 3. 处理特征，结果写入worker本地
        # 使用 subprocess.getstatusoutput(cmd) 调用外部命令 fea_extract 来处理 input_file 文件，并将结果写入到 output_file。
        cmd = "cat {input_file} | ./fea_extract {cfg_file} > {output_file}".format(
            input_file = input_file,
            output_file = output_file,
            cfg_file = parser_file_name_bc.value
        )
        rc, out = subprocess.getstatusoutput(cmd)

        # 4. 处理结果检测 return code != 0 -> 报错
        # 读取 output_file 的内容，并分割为多个 libsvm 格式的行
        with open(output_file, "r") as f:
            libsvm_list = f.read().strip().split("\n")  # 读出来去掉多写入的一个\n && 分割

        if rc != 0: # 如果 fea_extract 命令的返回码 rc 不为 0，则抛出一个 ValueError 异常，并将 output_file 的内容作为异常消息的一部分
            raise ValueError("GG!\n{}".format(libsvm_list))

        # 5. 输出结果整合
        ret_items = []
        for libsvm in libsvm_list: # 遍历 output_file 中的每一行，按 "#" 分割标签和特征

            label_feature_token = libsvm.strip().split("#")

            # 提取标签和特征，并将特征用逗号连接
            _label_fe = label_feature_token[0]
            label_fe = _label_fe.strip().split(" ")
            label = label_fe[0].strip()
            feature = ",".join(label_fe[1:])

            if len(label_feature_token) == 1:
                token = []
            else:
                token = label_feature_token[1].strip().replace("\t", ",")

            ret_items.append([label, feature, token])

        return ret_items

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

#  run 函数是一个用于处理 Spark 任务的流程，包括读取配置文件、执行 SQL 查询、特征转换，并将结果保存为 TFRecords 格式。
def run():

    # 1. parser feature columns
    # 这段代码的主要目的是从一个配置文件中读取特定的特征列名称，并存储在columns列表中。如果配置文件中没有包含"fea_cols"配置项，则会抛出一个异常。
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

    # 这行代码调用了Spark会话的sql方法，并传入了之前获取的SQL查询语句sql。Spark会执行这个SQL查询，
    # 并将结果作为一个DataFrame对象返回，这个对象被存储在df变量中。DataFrame是Spark中用于存储和操作结构化数据的核心数据结构。
    df = spark.sql(sql) # 执行sql查询

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

    # 使用df.repartition(args.partition)对原始DataFrame进行重新分区，以便在后续处理中更好地利用集群资源。
    # 然后，将DataFrame转换为RDD（弹性分布式数据集），并使用mapPartitions方法对每个分区应用之前定义的映射函数map_func。映射函数可能会对每个分区中的数据进行特征转换。
    res_df = df.repartition(
        args.partition
    ).rdd.mapPartitions(
        map_func
    ).toDF( # 使用toDF(schema)方法将转换后的RDD转换为一个新的DataFrame，该DataFrame的schema由之前定义的schema指定。
        schema
    )

    res_df.printSchema() # 使用res_df.printSchema()打印结果DataFrame的schema，以便验证其结构是否符合预期

    # 4. 落盘 使用 repartition 调整 DataFrame 的分区数，并将其保存为 TFRecords 格式
    res_df.repartition( # 使用repartition方法重新调整res_df DataFrame的分区数
        args.partition
    ).write.mode( # 使用write.mode("overwrite")设置写入模式为“overwrite”。
        "overwrite"
    ).format( # 使用format("tfrecords")指定输出格式为TFRecords。TFRecords是TensorFlow中用于存储大量序列化的Tensor数据的文件格式。它非常适合用于存储训练数据和模型参数。
        "tfrecords"
    ).save( # 使用save(args.output)将DataFrame保存为TFRecords格式到指定的输出路径。
        args.output
    )

    logging.info("All is Done.")


if __name__ == "__main__":
    run()
