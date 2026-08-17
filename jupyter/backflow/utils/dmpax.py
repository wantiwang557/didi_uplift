#!/usr/bin/python

"""
DiaMond PickAXe - 钻石镐 (diamond pickaxe)

游戏《我的世界》中，最高级的挖矿工具，耐久度较高且可以挖掘稀有矿石。

这个工具主要是希望帮助开发 PySpark 提效，里面封装了常用的操作，具体包括：
  1. 启动一个Spark。
  2. HDFS操作。文件操做（exists，get，set，delete等），降级，压缩保存 等。
  3. Hive操作。写表，降级。
  4. 常用计算。wilson区间，Tricky Rank，日期计算。

如何使用：

# 导入模块
from dmpax import SparkBaseApp

# Spark配置项
spark_conf = {
    "master": "yarn",
    "spark.submit.deployMode": "client",
    "driver-memory": "2g",
    "spark.executor.cores": "2",
    "spark.executor.memory": "4g",
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "10",
    "spark.dynamicAllocation.maxExecutors": "20",
    "spark.default.parallelism": "200",
    "spark.sql.shuffle.partitions": "200",
}

# 启动Spark
spark_app = SparkBaseApp()
spark_app.initialize(app_name="my_spark_app", spark_conf=spark_conf)

spark = spark_app.spark
hdfs = spark_app.hdfs
hive = spark_app.hive

# HDSF降级
read_path = "/hdfs/some_dir/%Y/%m/%d"
valid_path, _, _ = hdfs.degrade("20220202", read_path)
df = spark.read.parquet(valid_path).repartition(200)

# Hive降级
_, valid_date, _ = hive.degrade("20220202", "db.table", "year='%Y', month='%m', day='%d'")
df = spark.sql("select * from db.table where concat(year, month, day) ='{}'".format(valid_date))

# 压缩存储
save_path = "/hdfs/new_dir/2022/02/02"
spark_app.save_compress(df, save_path)

# 写表
spark_app.load_hive(
    table="db.table",
    partition="year='2022', month='02', day='02'",
    location=save_path
)
"""

# version
__version__ = "0.1.1"


# Python StdLib
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from random import sample
from uuid import uuid4
from time import time

import re
import os

# 优先使用 absl的 logging，没有的话再用默认的
try:
    from absl import logging

except ModuleNotFoundError:
    import logging

# 依赖第三方库
from scipy.special import erfinv

import numpy as np
import pandas as pd



# --- --- 小工具 --- ---

def std_dt_tans(dt,
                days=0,
                dt_fmt="%Y%m%d",
                out_fmt="%Y%m%d"):
    """
    日期格式化，可基于天数进行简单的日期变化；也可以用来补全路径模板日期分区

    :param dt: str      - 日期字符串，eg. "20220101"
    :param days: int    - 偏移的天数，默认为 0
    :param dt_fmt: int  - 输入日期 dt 的日期格式
    :param out_fmt: int - 输出日期的格式

    :return res_dt: str - 经过 out_fmt 格式化后的日期字符串
    """

    now_dt = datetime.strptime(dt, dt_fmt)
    res_dt = (now_dt + timedelta(days=days)).strftime(out_fmt)

    return res_dt


def get_format_schema(df):
    info = [
        "{:>16} : {}".format("<{}>".format(i.dataType), i.name)
        for i in df.schema
    ]

    return "\n".join(info)


# --- --- 数值计算工具 --- ---

def wilson(c, n, z=1.96):
    """
    wilson 置信区间，可以作为置信度除法使用

    :param c: float - 某种行为的发生数量，如 点击数
    :param n: float - 某种行为的前置行为发生数量，如 曝光数
    :param z: float - 置信度，默认 1.96，95%的置信区间

    :return res: tuple(float, float) - wilson区间
    """

    res = (0.0, 0.0)

    if n > 0:
        p_hat = c / n if c < n else 1
        denorm = 1 + z**2 / n
        anchor = p_hat + z**2 / (2 * n)
        delta = z * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))**0.5
        res = ((anchor - delta) / denorm, (anchor + delta) / denorm)

    return res


def _np_sort_core(pdf, col):
    """
    Tricky Rank 排序核心方法，只处理asc排序，desc由上层转换成asc排序

    :param pdf: pandas.DataFrame - 待排表格
    :param col: str - 待排列 列名

    :return rank_asc: np.array - 排序后的结果
    """

    align_gaps = np.concatenate(
        [np.array([0]), pdf[col].values[:-1]]
    ).cumsum()

    step = 1 / align_gaps[-1]
    rank_asc = align_gaps * step

    return rank_asc


def partition_rank(pdf, partition_cols, sort_col):
    """
    分组rank，相当于 partition by <partition_cols> order by <sort_col>

    :param pdf: pandas.DataFrame - 待排表格
    :param partition_cols: str / list[str] - 分区字段
    :param sort_col: str - 排序列
    :return: np.array - 排序后的数组
    """

    rank_asc_list = []

    for col_values in pdf[partition_cols].drop_duplicates().values:

        # 过滤器，用于寻找目标 partition
        filt, desc = None, []

        for _col, _val in zip(partition_cols, col_values):

            # update filter
            _f = pdf[_col] == _val
            filt = _f if filt is None else (filt & _f)

            desc.append("{}={}".format(_col, _val))

        # partition ranking
        sub_pdf = pdf[filt]
        rank_asc = _np_sort_core(sub_pdf, sort_col)
        rank_asc_list.append(rank_asc)

        logging.info(
            "Partition ranking [{}]: rank result.\n  {}".format(
                ", ".join(desc), rank_asc))

    return np.concatenate(rank_asc_list)


def tricky_rank_core(df,
                     rank_seed,
                     partition=None,
                     out_col="rank",
                     ascending=True,
                     tmp_col="_tricky_rank_count"):

    """
    Trick Ranking - 利用排序数据本身的特点进行排序。

    多数情况下，待排序数据表现为 有限不重复数据集合 ，数据总量可能有10亿，但是数据内容可能只有10几万的不重复数据，
    这个时候仅需对不重复集合进行排序，然后join到源数据上，即可完成ranking。

    Ranking 的时候需要确定两个关键内容：
        1. 需要确定排序rank值的最小量，即排完一个人往后走的step长度
        2. 需要记录 有限不重复集合 中每个数据出现的次数

    排序核心算法见 _np_sort_core()

    :param df: pyspark.sql.dataframe.DataFrame - 待排DataFrame
    :param rank_seed: str - 排序列
    :param partition: str/list[str] - 排序分区(partition by)，None表示不使用
    :param out_col: str - 输出列
    :param ascending: bool - 正序排列，Fasle表示倒序
    :param tmp_col: str - 排序使用的临时列，无特殊情况不建议修改

    :return: pandas.DataFrame - 排序好的Pandas
    """

    from pyspark.sql import functions as func

    # 检查分区参数，是否进行分区Rank
    if partition is None:
        care_cols = [rank_seed]

    elif isinstance(partition, str):
        care_cols = [partition, rank_seed]

    elif isinstance(partition, (list, tuple, set)):
        care_cols = list(partition)
        care_cols.append(rank_seed)

    else:
        raise TypeError("Can not access type of peremeter 'partition'")

    # 聚合得到带排序列的 有限不重复数据集合
    pdf = df.where(
        "{} is not null".format(rank_seed)
    ).groupby(
        *care_cols
    ).agg(
        func.count(rank_seed).alias(tmp_col)
    ).toPandas(
    ).sort_values(
        care_cols, ascending=ascending
    )

    # ranking
    if partition is None:
        ranking = _np_sort_core(pdf, tmp_col)

    else:
        partition_cols = care_cols[:-1]
        ranking = partition_rank(pdf, partition_cols, tmp_col)

    logging.info(
        "Ranking by: {}. Rank size: {}. Rank Result:\n  {}".format(
            care_cols, len(pdf), ranking))

    pdf[out_col] = ranking

    return pdf.drop(tmp_col, axis=1)


# --- --- Spark / HDFS / Hive 工具 --- ---

def init_spark():
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


class SingletonMeta(type):
    """
    单例类的 metaclass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = \
                super(SingletonMeta, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class _UDChecker(object):
    """
    用户自定义数值检查
    固定 比较运算符 和 一个数值，检查新数值是否满足对比条件
    """

    methods = {
        ">" : (lambda x, y: x > y, "<="),
        "<" : (lambda x, y: x < y, ">="),
        "=" : (lambda x, y: x == y, "!="),
        "==" : (lambda x, y: x == y, "!="),
        "!=" : (lambda x, y: x != y, "="),
        "<>" : (lambda x, y: x != y, "="),
        ">=" : (lambda x, y: x >= y, "<"),
        "<=" : (lambda x, y: x <= y, ">"),
    }
    pattern = re.compile(r"^ *([=!><]+) *(\d+\.?\d*) *$")

    def __init__(self, expr):
        """
        :param expr: str - 固定模式表达式，如 "> 100" 表示初始化一个 大于100 检测器
        """

        _op, _val = _UDChecker.pattern.findall(expr)[0]

        self.op = _op
        self.val = int(_val)
        self.method, self.opps = _UDChecker.methods[_op]

    def check(self, real):
        """
        :param real: int - 检查数值

        :return check_res: bool - 检查结果，True通过，False未通过
        :return check_desc: str - 检查结果的描述字符串
        """

        check_res = self.method(real, self.val)

        if check_res:
            check_desc = "Pass - Real[{}] {} Expect[{}]".format(
                real, self.op, self.val
            )

        else:
            check_desc = "Reject - Real[{}] {} Expect[{}]".format(
                real, self.opps, self.val
            )

        return check_res, check_desc


class _SparkWapper(metaclass=SingletonMeta):
    """
    包装SparkSession和SparkContext，便于子类调用
    """

    __spark = None   # SparkSession
    __sc = None      # SparkContext
    __status = False # 状态
    __uuid = None    # 标识

    @property
    def spark(self):
        """
        :return: object - SparkSession
        """

        return self.__spark

    @property
    def sc(self):
        """
        :return: object - SparkContext
        """

        return self.__sc

    @property
    def status(self):
        """
        :return: bool - Spark的有效状态
        """

        return self.__status

    @property
    def uuid(self):
        """
        :return: str - 伴随Spark的UUID，进程唯一
        """

        return self.__uuid

    def _set_spark(self, spark):
        """
        :param spark: object - 可以接受 <SparkSession> 或 <SparkContext> 两种类型.
        """

        if self.status:
            # SparkWapper单例已经初始化，不再重复执行
            return

        from pyspark.sql import SparkSession
        from pyspark.context import SparkContext

        if isinstance(spark, SparkSession):
            _spark_session = spark
            _spark_context = spark.sparkContext

        elif isinstance(spark, SparkContext):
            _spark_session = SparkSession.Builder().getOrCreate()
            _spark_context = spark

        else:
            raise TypeError((
                "Got wrong type for parameter 'spark'! "
                "Expected type <{t1}> or <{t2}>, but got {tp}"
            ).format(
                t1=str(SparkContext), t2=str(SparkSession), tp=type(spark)
            ))

        _SparkWapper.__spark = _spark_session
        _SparkWapper.__sc = _spark_context
        _SparkWapper.__uuid = "".join(sample(uuid4().hex, 8))
        _SparkWapper.__status = True

    def close(self):

        if self.status:
            self.spark.stop()
            _SparkWapper.__status = False

        else:
            logging.info("Already Stoped.")

    def initialize(self, spark):
        self._set_spark(spark)


class HDFS(_SparkWapper):
    """
    HDFS封装，通过jvm提供的HDFS handler操作HDFS
    """

    def __init__(self, spark=None):
        """
        :param spark: object - 可以接受 <SparkSession> 或 <SparkContext> 两种类型.
        """

        super(HDFS, self).__init__()

        self.__active = False

        if spark is not None:
            self.initialize(spark)

    @property
    def fs(self):
        """
        :return: object - jvm HDFS 文件对象
        """

        return self.__fs

    @property
    def name(self):
        """
        :return: str - HDFS协议名 + 域名
        """

        return self.__name

    def initialize(self, spark):
        """
        HDFS初始化

        :param spark: object - 可以接受 <SparkSession> 或 <SparkContext> 两种类型.
        """

        if self.__active:
            return

        self._set_spark(spark)

        # 检查 spark context 状态
        if self.sc._jsc is None:
            raise RuntimeError("SparkContext has already closed!")

        hadoop_conf = self.sc._jsc.hadoopConfiguration()

        # HDFS 协议+域名
        self.__name = hadoop_conf.get("fs.default.name")

        # HDFS FS Handler
        self.__fs = self.sc._jvm.org.apache.hadoop.fs.FileSystem.get(
            hadoop_conf)

        self.__active = True

    def path(self, path):
        """
        封装成 java hdfs 标准路径

        :param path: str - posix格式路径
        :return jvm_hdfs_path: object - 封装后的路径对象
        """

        if not isinstance(path, (str, Path)):
            raise TypeError(
                "Can not accept type <{}> for 'path'!".format(type(path)))

        # 通过 Pathlib 兼容多种路径格式
        path = Path(path).as_posix()
        jvm_hdfs_path = self.sc._jvm.org.apache.hadoop.fs.Path(path)

        return jvm_hdfs_path

    def exists(self, path):
        """
        判断 目录/文件 是否存在

        :param path: str - posix格式路径
        :return: bool - True表示存在，False表示不存在
        """

        _path = self.path(path)
        return self.fs.exists(_path)

    def delete(self, path):
        """
        删除 目录/文件

        :param path: str - posix格式路径
        :return: bool - True删除成功，False表示删除失败，失败可能是因为path不存在
        """

        _path = self.path(path)
        return self.fs.delete(_path)

    def mkdirs(self, path):
        """
        递归创建目录，等同于 hadoop fs -mkdir -p <target>

        :param path: str - posix格式路径
        :return: bool - True创建成功 或 已经被创建
        """

        _path = self.path(path)
        return self.fs.mkdirs(_path)

    def get(self, src, dst):
        """
        将HDFS源get到本地，等同于 hadoop fs -get <src> <dst>

        :param src: str - HDFS源，posix格式路径
        :param dst: str - 本地路径，posix格式路径
        :return:
        """

        _src = self.path(src)
        _dst = self.path(dst)
        self.fs.copyToLocalFile(_src, _dst)

    def put(self, src, dst):
        """
        将本地源put到HDFS，等同于 hadoop fs -put <src> <dst>

        :param src: str - 本地源源，posix格式路径
        :param dst: str - HDFS路径，posix格式路径
        :return:
        """

        _src = self.path(src)
        _dst = self.path(dst)
        self.fs.copyFromLocalFile(_src, _dst)

    def isdir(self, path):
        """
        判断目标是否为目录

        :param path: str - posix格式路径
        :return: bool - True是目录，False表示非目录
        """

        _path = self.path(path)
        return self.fs.isDirectory(_path)

    def isfile(self, path):
        """
        判断目标是否为文件

        :param path: str - posix格式路径
        :return: bool - True是文件，False表示非文件
        """

        _path = self.path(path)
        return self.fs.isFile(_path)

    def count(self, path):
        """
        获取目标的统计信息，等同于 hadoop fs -count <target>

        :param path: str - posix格式路径
        :return (dir_cnt, file_cnt, byte_cnt): int,int,int
            dir_cnt  - 目录数量（包括自己）
            file_cnt - 文件数（包括自己）
            byte_cnt - 总大小（单位是 Byte)
        """
        _path = self.path(path)
        _summary = self.fs.getContentSummary(_path)
        dir_cnt = _summary.getDirectoryCount()
        file_cnt = _summary.getFileCount()
        byte_cnt = _summary.getLength()

        return dir_cnt, file_cnt, byte_cnt

    def degrade(self,
                dt,
                path_fmt,
                max_degrade=15,
                expect_dirs="=0",
                expect_files=">=1",
                expect_size=">=1000",
                in_dt_fmt="%Y%m%d",
                out_dt_fmt="%Y%m%d"):
        """
        HDFS 降级找可用的分区
        例如：时间T0执行的任务依赖某个T-1的结果，当T-1结果不存在，自动向前所寻可用结果，直到T-n

        :param dt: str           - 开始日期
        :param path_fmt: str     - 检测路径格式(可以带时间占位符，如 %Y/%m/%d)
        :param max_degrade: int  - 最大降级天数
        :param expect_dirs: int  - 目录数量阈值，检测数 等于 该阈值才能通过
        :param expect_files: int - 文件数量阈值，检测数 大于等于 该阈值才能通过
        :param expect_size: int  - 存储量（Byte）阈值，检测数 大于等于 该阈值才能通过
        :param in_dt_fmt: str    - 输入开始日期的格式
        :param out_dt_fmt: str   - 降级后日期输出格式

        :return valid_path: str   - 最近的一个有效路径
        :return valid_date: str   - 最近的一个有效的日期
        :return degrade_days: int - 降级的天数
        """

        valid_path, valid_date, degrade_days = None, None, None
        last_path = None

        dir_checker = _UDChecker(expect_dirs)
        file_checker = _UDChecker(expect_files)
        size_checker = _UDChecker(expect_size)

        for i in range(max_degrade + 1):

            # 待检测的path 和 日期
            check_path = std_dt_tans(
                dt, days=-i, dt_fmt=in_dt_fmt, out_fmt=path_fmt)

            if (last_path is not None) and (check_path == last_path):
                # 固定分区无需多降级检测
                break
            else:
                last_path = check_path

            # 1. 存在检测
            if not self.exists(check_path):
                logging.info(
                    "DEGRADE - Not Exists: {path}".format(path=check_path)
                )
                continue

            res_dirs, res_files, res_size = self.count(check_path)
            res_dirs -= 1

            # 2. 目录数量检测
            check_res, desc = dir_checker.check(res_dirs)
            if not check_res:
                logging.info("DEGRADE[dirs] {}: {}".format(desc, check_path))
                continue

            # 3. 文件数检测
            check_res, desc = file_checker.check(res_files)
            if not check_res:
                logging.info("DEGRADE[files] {}: {}".format(desc, check_path))
                continue

            # 4. 存储量检测
            check_res, desc = size_checker.check(res_size)
            if not check_res:
                logging.info("DEGRADE[size] {}: {}".format(desc, check_path))
                continue

            # 全部检测通过
            valid_date = std_dt_tans(
                dt, days=-i, dt_fmt=in_dt_fmt, out_fmt=out_dt_fmt)
            valid_path = check_path
            degrade_days = i
            break

        # 打印降级结果
        if valid_date is not None:
            logging.info("Find valid partition at {dt}: {path}".format(
                dt=valid_date,
                path=valid_path))

        else:
            logging.warning("Can not find any valid partition.")

        return valid_path, valid_date, degrade_days

    def save_compress(self, df, save_path):
        """
        压缩存储

        :param df: pyspark.sql.dataframe.DataFrame - 待压缩保存的DataFrame
        :param save_path: str - 输出目录
        :return:
        """

        start_time = time()

        # 1. 临时落盘
        tmp_save_path = ".".join(
            [Path(save_path).as_posix(), "_OptimizeCompress"])

        if self.exists(tmp_save_path):
            logging.warning(
                "Optimize Compress Directory already exists, "
                "it will be overwritten: {}".format(tmp_save_path))
            self.delete(tmp_save_path)

        df.write.save(tmp_save_path)

        # 2. 计算存储分区数, 尽量压缩到200个partition以内
        megabyte = self.count(tmp_save_path)[2] / 1048576  # # B -> MB: 1024*024

        opt_part_num = int(megabyte / 512) + 1

        if opt_part_num > 200:
            opt_part_num = int(megabyte / 1024) + 1

        if opt_part_num > 200:
            opt_part_num = int(megabyte / 2048) + 1

        # 3. 保存最终结果
        if self.exists(save_path):
            logging.warning(
                "Output Directory already exists, "
                "it will be overwritten: {}".format(save_path))
            self.delete(save_path)

        self.spark.read.parquet(
            tmp_save_path
        ).repartition(
            opt_part_num
        ).write.save(
            save_path
        )

        logging.info(
            "Compress save success with {} parts. "
            "Time spend: {:>.0f}s".format(opt_part_num, time() - start_time))

        # 删除临时目录
        if self.delete(tmp_save_path):
            logging.info("Delete success.")
            logging.warning("Delete tmp dir [SUCCESS]. - {}".format(tmp_save_path))

        else:
            logging.error("Delete tmp dir [ERROR]! - {}".format(tmp_save_path))


class Hive(_SparkWapper):
    """
    Hive封装，提供常用的 Hive操作
    """

    def __init__(self, spark=None):
        """
        :param spark: object - 可以接受 <SparkSession> 或 <SparkContext> 两种类型.
        """

        super(Hive, self).__init__()

        if spark is not None:
            self.initialize(spark)

    def load_hive(self, table, partition, location):
        """
        加载Hive外表，封装了添加分区的SQL

        :param table: str - Hive表名
        :param partition: str - 分区
        :param location: str - 路径
        :return:
        """

        logging.info((
            "\n=== HIVE TABLE ALTERING ==="
            "\n    Table: {table}"
            "\nPartition: {partition}"
            "\n Location: {location}"
        ).format(
            table=table,
            partition=partition,
            location=location
        ))

        self.spark.sql((
            "ALTER TABLE {table} DROP IF EXISTS PARTITION "
            "({partition})"
        ).format(
            table=table,
            partition=partition
        ))

        self.spark.sql((
            "ALTER TABLE {table} ADD IF NOT EXISTS PARTITION "
            "({partition}) "
            "LOCATION '{location}'"
        ).format(
            table=table,
            partition=partition,
            location=location
        ))

    def degrade(self,
                dt,
                table,
                partition_fmt,
                max_degrade=15,
                expect_parts="=1",
                expect_count=">=1",
                using_count_check=False,
                in_dt_fmt="%Y%m%d",
                out_dt_fmt="%Y%m%d"):
        """
        Hive 降级找可用的分区
        例如：时间T0执行的任务依赖某个T-1的结果，当T-1结果不存在，自动向前所寻可用结果，直到T-n

        :param dt: str            - 开始日期
        :param table: str         - hive表名
        :param partition_fmt: str - 日期分区格式(如 "year='%Y',month='%m',day='%d'") ## Spark3 对数据类型要求严格，分区数据类型为字符串必须加引号 ##
        :param max_degrade: int   - 最大降级天数
        :param expect_parts: str  - 分区数 判定条件+阈值
        :param expect_count: str  - row条数 判定条件+阈值【需读表统计，运行较慢】
        :param in_dt_fmt: str     - 输入开始日期的格式
        :param out_dt_fmt: str    - 降级后日期输出格式
        :param using_count_check: bool - 是否使用count检测，配合 expect_count 使用

        :return valid_part: str   - 最近的一个有分区
        :return valid_date: str   - 最近的一个有效的日期
        :return degrade_days: int - 降级的天数
        """

        valid_part, valid_date, degrade_days = None, None, None
        last_part = None

        parts_checker = _UDChecker(expect_parts)
        count_checker = _UDChecker(expect_count)

        for i in range(max_degrade + 1):

            check_part = std_dt_tans(
                dt, days=-i, dt_fmt=in_dt_fmt, out_fmt=partition_fmt)

            if (last_part is not None) and (check_part == last_part):
                # 固定分区无需多降级检测
                break
            else:
                last_part = check_part

            # 1. 分区数检测
            parts_cnt = len(
                self.spark.sql((
                    "show partitions {table} partition({partition})"
                ).format(
                    table=table, partition=check_part
                )).take(
                    parts_checker.val
                ))

            check_res, desc = parts_checker.check(parts_cnt)
            if not check_res:
                logging.info("DEGRADE[parts] {}: {}".format(desc, check_part))
                continue

            # 2. 数据量检测
            if using_count_check:
                _condition = " and ".join(
                    [s.strip() for s in check_part.split(",")])

                row_cnt = self.spark.sql(
                    "select count(1) from {table} where {partition}".format(
                        table=table, partition=_condition)
                ).take(1)[0][0]

                check_res, desc = count_checker.check(row_cnt)
                if not check_res:
                    logging.info(
                        "DEGRADE[count] {}: {}".format(desc, check_part))
                    continue

            # 检测通过
            valid_date = std_dt_tans(
                dt, days=-i, dt_fmt=in_dt_fmt, out_fmt=out_dt_fmt)
            valid_part = check_part
            degrade_days = i
            break

        if valid_part is not None:
            logging.info("Find valid partition at {d}: {t}[{p}]".format(
                d=valid_date, t=table, p=valid_part))

        else:
            logging.warning("Can not find any valid partition.")

        return valid_part, valid_date, degrade_days


class SparkBaseApp(_SparkWapper):
    """
    SparkApp 封装
    """

    @property
    def hdfs(self):
        return self.__hdfs

    @property
    def hive(self):
        return self.__hive

    def __init__(self):

        super(SparkBaseApp, self).__init__()

        self.spark_conf = None
        self.app_name = None
        self.__active = False

    def initialize(self, app_name=None, spark_conf=None, spark=None):
        """
        初始化SparkSession

        :param app_name: str - Spark App Name
        :param spark_conf: dict - Spark配置项
        :param spark: object - 可以接受 <SparkSession> 或 <SparkContext> 两种类型. 若提供，则直接从提供的Spark进行初始化

        """

        if self.__active:
            return

        if (not self.status) and (spark is None):

            # 1. 加载 pyspark
            init_spark()

            # 2. 构建SparkSession
            from pyspark.sql import SparkSession

            _spark = SparkSession.builder

            if app_name is not None:
                self.app_name = app_name

                _spark = _spark.appName(app_name)
                logging.info("Initializing Spark App: {}".format(app_name))

            # spark config
            if isinstance(spark_conf, dict):
                self.spark_conf = spark_conf.copy()
                _conf = spark_conf.copy()

                _fmt_conf_str = "{\n"
                for k, v in _conf.items():
                    _fmt_conf_str += "  {}: {}\n".format(k, v)
                _fmt_conf_str += "}\n"
                logging.info("Using spark config:\n{}".format(_fmt_conf_str))

                if "spark.executorEnv.PYSPARK_PYTHON" in _conf:
                    os.environ["PYSPARK_PYTHON"] = \
                        _conf["spark.executorEnv.PYSPARK_PYTHON"]

                if "master" in _conf:
                    _spark = _spark.master(_conf["master"])
                    _ = _conf.pop("master")

                for conf_key, conf_val in _conf.items():
                    _spark = _spark.config(conf_key, conf_val)

            _spark = _spark.enableHiveSupport().getOrCreate()

        else:
            _spark = spark

        # 3. 设置 Spark
        self._set_spark(_spark)
        logging.info("Spark Version: {}".format(self.spark.version))

        # 初始化 HDFS 和 Hive
        self.__hdfs = HDFS()
        self.__hdfs.initialize(self.spark)

        self.__hive = Hive()
        self.__hive.initialize(self.spark)

        self.__active = True

    def save_compress(self, df, save_path):
        """
        压缩存储 - 实际调用 HDFS实例的 save_compress()方法

        :param df: pyspark.sql.dataframe.DataFrame - 待压缩保存的DataFrame
        :param save_path: str - 输出目录
        :return:
        """

        self.__hdfs.save_compress(df, save_path)

    def load_hive(self, table, partition, location):
        """
        加载Hive外表 - 实际调用 Hive实例的 load_hive()方法

        :param table: str - Hive表名
        :param partition: str - 分区
        :param location: str - 路径
        :return:
        """

        self.__hive.load_hive(table, partition, location)




# =================================== 如下是 Tricky Rank 部分 ==========================================


# --- rank core methods ---
# 排序核心方法，只处理asc排序，desc由上层转换成asc排序

def _asc_sort_rank(arr, **kwargs):
    """
    常规rank
    :param arr: np.array/list - 待排数据
    :param kwargs: dict - 排序参数
    :return sort_rank: np.array - 排序后的结果
    """

    align_gaps = np.concatenate(
        [np.array([0]), np.array(arr[:-1])]
    ).cumsum()

    step = 1 / align_gaps[-1] if len(align_gaps) > 1 else 1.0
    sort_rank = align_gaps * step

    return sort_rank

def _asc_scal_rank(arr, **kwargs):
    """
    缩放rank
    :param arr: np.array/list - 待排数据
    :param kwargs: dict - 排序参数
    :return sort_rank: np.array - 排序后的结果
    """

    align_gaps = np.array(arr).cumsum()
    vmin, vmax = align_gaps.min(), align_gaps.max()

    if vmin != vmax:
        scal_rank = (align_gaps - vmin) / (vmax - vmin)

    else:
        # all zero
        scal_rank = np.zeros_like(align_gaps, dtype=float)

    return sacl_rank


def _asc_gaus_rank(arr, **kwargs):
    """
    高斯rank
    :param arr: np.array/list - 待排数据
    :param kwargs: dict - 排序参数
    :return sort_rank: np.array - 排序后的结果
    """

    arr = np.array(arr, dtype=float)

    if len(arr) > 1:
        epsilon = kwargs.get("epsilon")
        if epsilon is None or (epsilon > 1):
            epsilon = 1 / arr.sum()

        # 1. 排序
        sort_ary = _asc_sort_rank(arr)

        # 2. 归一到 (-1, 1)
        norm_ary = (sort_ary - 0.5) * 2
        clip_ary = np.clip(norm_ary, epsilon - 1, 1 - epsilon)

        # 3. err返函数算rank
        gaus_rank = erfinv(clip_ary)

    else:
        gaus_rank = np.zeros_like(arr)

    # 4. 缩放
    rscale = kwargs.get("rscale", False)
    if rscale:
        gmin, gmax = gaus_rank.min(), gaus_rank.max()
        diff = gmax - gmin if gmax > gmin else 1
        gaus_rank = (gaus_rank - gmin) / diff

    return gaus_rank


def coalesce(*args):

    for e in args:
        if e is not None:
            break
    return e


class TrickyRank(object):

    _methods_map = {
        "uniform": _asc_sort_rank,
        "scale": _asc_scal_rank,
        "gaussian": _asc_gaus_rank
    }

    def __init__(self,
                 sensibility=100000,
                 normalize=False,
                 ascending=True,
                 fillna=None,
                 using_cache=True):

        self._default_sensibility = sensibility
        self._default_normalize = normalize
        self._default_ascending = ascending
        self._default_fillna=fillna
        self._tmp_col = "_ttr_cnt_"

        self.df = None
        self.rank_col = None
        self.partitions = None
        self.group_keys = None
        self.sensibility = None
        self.methods = None
        self.normalize = None
        self.ascending = None
        self.fillna = None

        self.epsilon = None
        self.rscale = None

        self.raw_df = None
        self.raw_pdf = None

        self.using_cache = using_cache
        self.cache = {}

        from pyspark.sql import SparkSession
        from pyspark.sql import functions as func

        self._spark = SparkSession.Builder().getOrCreate()
        self._func = func

    def _parse_parameter(self,
                         df,
                         rank_col,
                         partition,
                         sensibility,
                         methods,
                         normalize,
                         epsilon,
                         rscale,
                         ascending,
                         fillna):
        """
        单独处理参数
        """

        # --- --- 1. 数据类型检查 --- ---
        from pyspark.sql.types import IntegerType, LongType
        from pyspark.sql.types import FloatType, DoubleType

        schema = df.select(rank_col).schema[0]
        check_res = isinstance(
            schema.dataType,
            (FloatType, DoubleType, IntegerType, LongType)
        )

        # --- --- 2. 检查分区参数，是否进行分区Rank，设置group by的 keys --- ---
        if not isinstance(rank_col, str):
            raise TypeError(
                "'rank_col' can not accept '{}' type.".format(type(rank_col)))

        # rank_column and  rank_seed
        self.rank_col = rank_col
        self.rank_seed = "_rank_seed_{}".format(rank_col)

        _partition = []
        if isinstance(partition, str):
            # 保序去重
            _partition = list(dict.fromkeys(
                [p.strip() for p in partition.split(",")]))

        elif isinstance(partition, (list, tuple, set)):
            _partition = list(dict.fromkeys(
                [p.strip() for p in partition]))

        else:
            logging.info("Not use partition rank.")

        # group keys / partitions
        self.group_keys = _partition + [self.rank_seed]
        self.partitions = _partition

        # --- --- 3. Method 设置 --- ---
        if isinstance(methods, str):
            _methods = set([m.strip() for m in methods.split(",")])

        elif isinstance(methods, (list, tuple, set)):
            _methods = set([m.strip() for m in methods])

        else:
            raise TypeError(
                "'methods' can not accept '{}' type.".format(type(methods)))

        accept_method = list(TrickyRank._methods_map.keys())
        accept_check = [m for m in _methods if m not in accept_method]
        assert not accept_check, (
            "{} method can not accept. support method {}.".format(
                accept_check, accept_method))

        self.methods = list(_methods)

        # --- --- 4. fillna --- ---

        fillna = coalesce(fillna, self._default_fillna)

        if isinstance(fillna, (int, float, type(None))):
            _fillna = fillna if fillna is not None else "null"

        else:
            raise TypeError(
                "'fillna' can not accept '{}' type.".format(type(fillna)))

        self.fillna = _fillna

        # --- --- 5. 其他 --- ---

        # normalize
        self.normalize = bool(coalesce(normalize, self._default_normalize, False))

        # sensibility
        self.sensibility = int(coalesce(sensibility, self._default_sensibility, 1))

        # ascending
        self.ascending = bool(coalesce(ascending, self._default_ascending, True))

        # df
        self.df = df

        # rank 参数 epsilon
        self.epsilon = epsilon
        self.rscale = rscale

        # 处理数据的uuid 暂时无用，未来用于缓存机制
        self.proc_uuid = "{df}|{gk}|{nm}|{sn}|{na}".format(
            df=id(self.df),
            gk=",".join(self.group_keys),
            nm=self.normalize,
            sn=self.sensibility,
            na=self.fillna
        ).lower()

    def _gen_raw_df(self):

        # fillna
        # _fillna = "null" if self.fillna is None else self.fillna
        _fillna = self.fillna

        if self.normalize:
            # 使用 normalize 对情况下

            agg_row = self.df.agg(
                self._func.min(self.rank_col).alias("min"),
                self._func.max(self.rank_col).alias("max")
            ).take(1)[0]

            vmin = agg_row["min"] if _fillna == "null" else min(agg_row["min"], _fillna)
            vmax = agg_row["max"] if _fillna == "null" else max(agg_row["max"], _fillna)
            diff = vmax - vmin

            rank_seed_sql = (
                "cast((nvl({col}, {fna}) - {vmin}) / {diff} * {sens} as bigint) as {ncol}"
            ).format(
                col=self.rank_col,
                fna=_fillna,
                vmin=vmin,
                diff=diff,
                sens=self.sensibility,
                ncol=self.rank_seed
            )

        else:
            rank_seed_sql = (
                "cast(nvl({col}, {fna}) * {sen} as bigint) as {ncol}"
            ).format(
                col=self.rank_col,
                fna=_fillna,
                sen=self.sensibility,
                ncol=self.rank_seed
            )

        logging.info(rank_seed_sql)

        self.raw_df = self.df.selectExpr("*", rank_seed_sql)
        self.raw_pdf = self.raw_df.where(
            "{} is not null".format(self.rank_seed)
        ).groupby(
            *self.group_keys
        ).agg(
            self._func.count(self.rank_seed).alias(self._tmp_col)
        ).toPandas()

    def _partition_rank(self, pdf, rank_method, rank_args):
        """
        分组rank，相当于 partition by <partition_cols> order by <sort_col>

        :param pdf: pandas.DataFrame - 待排表格
        :param partitions: str / list[str] - 分区字段
        :param sort_col: str - 排序列
        :return: np.array - 排序后的数组
        """

        ranking_list = []

        for col_values in pdf[self.partitions].drop_duplicates().values:

            # 过滤器，用于寻找目标 partition
            filt, desc = None, []

            for _col, _val in zip(self.partitions, col_values):

                # update filter
                _f = pdf[_col] == _val
                filt = _f if filt is None else (filt & _f)

                desc.append("{}={}".format(_col, _val))

            # partition ranking
            sub_pdf = pdf[filt]
            ranking = rank_method(sub_pdf[self._tmp_col].values, **rank_args)
            ranking_list.append(ranking)

            logging.info(
                "Partition ranking [{}]: rank result.\n  {}".format(
                    ", ".join(desc), ranking))

        return np.concatenate(ranking_list)

    def tricky_rank(self,
                    df,
                    rank_col,
                    methods,
                    out_col="rank",
                    partition=None,
                    sensibility=None,
                    epsilon=None,
                    rscale=True,
                    normalize=None,
                    ascending=None,
                    fillna=None):
        """
        :param df: spark.sql.dataframe.DataFrame - 输入dataframe
        :param rank_col: str - 待排列名

        :param out_col: str - rank输出列名；默认 'rank'
        :param partition: str/list[str] - 分区rank的分区字段；默认 None，即不使用分区rank
        :param methods: str/list[str] - rank方法，支持多方法同时rank，每种方法输出一列
        :param sensibility: int - rank敏感度（分桶大小）；默认 100000
        :param normalize: bool - 是否开启数据归一化，开启会优化rank效果，但会增加计算开销；默认 False
        :param ascending: bool - rank顺序，True=从小到达，False=由大到小；默认 True
        :param fillna: int/float - 填充缺失值；默认 None 即不填充

        :param epsilon: float - 高斯rank专用参数，截断误差；默认 None，为自适应截断
        :param rscale: bool - 高斯rank专用参数，是否缩放到[0~1]输出；默认 True
        """

        # 1. 参数解析
        self._parse_parameter(
            df=df,
            rank_col=rank_col,
            partition=partition,
            sensibility=sensibility,
            methods=methods,
            normalize=normalize,
            epsilon=epsilon,
            rscale=rscale,
            ascending=ascending,
            fillna=fillna
        )

        # 2. 生成目标dataframe
        self._gen_raw_df()

        rank_pdf = self.raw_pdf.sort_values(
            self.group_keys, ascending=self.ascending
        ).reset_index(
            drop=True
        )

        # 3. 根据method进行ranking
        rank_args = {"epsilon": self.epsilon, "rscale": self.rscale}

        for _method in self.methods:

            rank_method = TrickyRank._methods_map[_method]

            # ranking
            if self.partitions:
                # partition rank
                ranking = self._partition_rank(rank_pdf, rank_method, rank_args)

            else:
                ranking = rank_method(rank_pdf[self._tmp_col].values, **rank_args)
                # normal rank

            out_col_name = (
                out_col if len(self.methods) == 1 else "{}_{}".format(out_col, _method)
            )
            rank_pdf[out_col_name] = ranking

            logging.info(
                "Ranking by: {}. Rank size: {}.".format(self.group_keys, len(rank_pdf)))

        # 4. 结果组装
        rank_pdf = rank_pdf.drop(self._tmp_col, axis=1)
        _rank_df = self._spark.createDataFrame(rank_pdf)

        cooked_df = self.raw_df.join(
            self._func.broadcast(_rank_df), self.group_keys, "left"
        ).drop(
            self.rank_seed
        )

        return cooked_df

    def uniform(self,
                df,
                rank_col,
                out_col="rank",
                partition=None,
                sensibility=None,
                normalize=None,
                ascending=None,
                fillna=None):
        """
        :param df: spark.sql.dataframe.DataFrame - 输入dataframe
        :param rank_col: str - 待排列名

        :param out_col: str - rank输出列名；默认 'rank'
        :param partition: str/list[str] - 分区rank的分区字段；默认 None，即不使用分区rank
        :param sensibility: int - rank敏感度（分桶大小）；默认 100000
        :param normalize: bool - 是否开启数据归一化，开启会优化rank效果，但会增加计算开销；默认 False
        :param ascending: bool - rank顺序，True=从小到达，False=由大到小；默认 True
        :param fillna: int/float - 填充缺失值；默认 None 即不填充
        """

        return self.tricky_rank(
            df=df,
            rank_col=rank_col,
            methods="uniform",
            out_col=out_col,
            partition=partition,
            sensibility=sensibility,
            normalize=normalize,
            ascending=ascending,
            fillna=fillna
        )

    def gaussian(self,
                 df,
                 rank_col,
                 out_col="rank",
                 partition=None,
                 sensibility=None,
                 epsilon=None,
                 rscale=True,
                 normalize=None,
                 ascending=None,
                 fillna=None):
        """
        :param df: spark.sql.dataframe.DataFrame - 输入dataframe
        :param rank_col: str - 待排列名

        :param out_col: str - rank输出列名；默认 'rank'
        :param partition: str/list[str] - 分区rank的分区字段；默认 None，即不使用分区rank
        :param sensibility: int - rank敏感度（分桶大小）；默认 100000
        :param normalize: bool - 是否开启数据归一化，开启会优化rank效果，但会增加计算开销；默认 False
        :param ascending: bool - rank顺序，True=从小到达，False=由大到小；默认 True
        :param fillna: int/float - 填充缺失值；默认 None 即不填充
        :param epsilon: float - 高斯rank专用参数，截断误差；默认 None，为自适应截断
        :param rscale: bool - 高斯rank专用参数，是否缩放到[0~1]输出；默认 True
        """

        return self.tricky_rank(
            df=df,
            rank_col=rank_col,
            methods="gaussian",
            out_col=out_col,
            partition=partition,
            epsilon=epsilon,
            rscale=rscale,
            sensibility=sensibility,
            normalize=normalize,
            ascending=ascending,
            fillna=fillna
        )
