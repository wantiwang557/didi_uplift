# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import logging
import argparse
import pprint
from string import Template
import random

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import QuantileDiscretizer
from toad.merge import KMeansMerge
from pyspark.sql import functions as F
import os

import sys
sys.path.append('/nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/autoUpdate/my_workspace/dm-deeplearning/')
from dmdeepl.feature import buckets


# TODO 全局变量，根据项目酌情修改
SPARK_PREFIX = "lite_feature_handler_{}".format(random.randint(1, 10000))  # spark 任务前缀
HDFS_PREFIX = ""  # hdfs工作目录前缀（暂时没用）

DEBUG = False
PROD = True

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %a %H:%M:%S',
                    stream=sys.stdout)


class Config:
    def __init__(self, queue, user_name, min_executors, max_executors, executor_memory, cores):
        # spark 配置项
        self.cores = int(cores)
        self.minExecutors = int(min_executors)
        self.maxExecutors = int(max_executors)
        self.executorMemory = executor_memory
        self.parallelism = int(3 * self.cores * self.maxExecutors)

        # 项目配置项
        self.APP_PREFIX = SPARK_PREFIX
        self.HDFS_WORKSPACE = HDFS_PREFIX
        self.YARN_QUEUE = queue
        self.HADOOP_USER = user_name

        # check
        # assert os.path.exists(self.JAR), "JAR not exists: " + self.JAR

    def __repr__(self):
        return """MLflow Runner Config (
        -------------  Project Config  ------------
        YARN_QUEUE: {yarn_queue}
        HADOOP_USER: {hadoop_user}
        HDFS_WORKSPACE: {hdfs_workspace}
        -------------  Spark Runtime Config  ------------
        MinExecutors: {min_executors}
        MaxExecutors: {max_executors}
        ExecutorMemory: {executor_memory}
        parallelism: {parallelism}
        cores: {cores}
        )
        """.format(
            yarn_queue=self.YARN_QUEUE,
            hadoop_user=self.HADOOP_USER,
            hdfs_workspace=self.HDFS_WORKSPACE,
            min_executors=self.minExecutors,
            max_executors=self.maxExecutors,
            executor_memory=self.executorMemory,
            parallelism=self.parallelism,
            cores=self.cores)



class FeatureHandler():
    def __init__(self,
            spark,
            sql_file,
            sql_param,
            is_sample,
            feature_conf_file,
            parser_conf_file,
            onehot_fea_list,
            enum_fea_list,
            lisan_fea_list,
            days_fea_list,
            fix_ts_col,
            minmax_fea_list,
            direct_fea_list,
            bucket_type,
            bucket_num,
            label_col,
            token_col,
            float_calculate_default
        ):
        self.spark = spark

        self.feat = self._get_orig_df(sql_file, sql_param, is_sample)
        self.feature_conf_file = feature_conf_file
        self.parser_conf_file = parser_conf_file
        self.onehot_fea_list = onehot_fea_list
        self.enum_fea_list = enum_fea_list
        self.lisan_fea_list =lisan_fea_list
        self.days_fea_list = days_fea_list
        self.fix_ts_col = fix_ts_col
        self.minmax_fea_list = minmax_fea_list
        self.direct_fea_list = direct_fea_list
        self.bucket_type = bucket_type
        self.bucket_num = int(bucket_num)
        self.label_col = label_col
        self.token_col = token_col
        self.float_calculate_default = int(float_calculate_default)
        

    def _get_orig_df(self, sql_file, sql_param, is_sample):
        hdfs_cache_dir = "/user/prod_growstrategy/tmp.yyl/gen_conf/"
        #df = self.spark.read.parquet(hdfs_cache_dir).sample(fraction=0.5)
        #print("directly return...")
        #return df
        
        sql_params = {}

        if sql_param:
            _param_list = [
                param.split("=") for param in sql_param.strip().split(";")
            ]

            sql_params = {
                k.strip(): v.strip() for (k, v) in _param_list
            }
        with open(sql_file,'r') as f:
            sql = Template(f.read()).substitute(sql_params)
        logging.info(sql)
        
        print("sample rate: ", is_sample)
        df = self.spark.sql(sql).sample(fraction=float(is_sample))
        
        '''
        if is_sample == '1' or is_sample == 1:
            cnt = df.count()
            logging.info("data size: {cnt}, sample rate {rate}".format(cnt = cnt, rate = min(1, 100000/cnt)))
            if (cnt > 1000000):
                df = df.sample(100000/cnt)
        '''

        df.write.mode("overwrite").parquet(hdfs_cache_dir)
        df = self.spark.read.parquet(hdfs_cache_dir)
                
        return df

    def _get_float_subdivision(self, float_col):
        '''
        对float类型字段进一步划分-直接输出、离散化、正则化。划分条件：
        当取值 0~1则直接输出(3-direct)
        当最大值 > 100 则做分桶(2-lisan)
        其他情况使用4-正则化 (@todo. rfmpeat现在有负值 目前算子对负值的处理不太合理正在迭代)
        '''

        direct_float_cols = []
        lisan_float_cols = []
        norm_float_cols = []

        if float_col:
            float_min_max = self.feat.agg(
                *[F.max(c).alias("max_{}".format(c)) for c in float_col],
                *[F.min(c).alias("min_{}".format(c)) for c in float_col]
            ).collect()

            if float_min_max:
                float_min_max_result = float_min_max[0].asDict()
                for col in float_col:
                    min_col_value = float_min_max_result["min_{}".format(col)]
                    max_col_value = float_min_max_result["max_{}".format(col)]
                    if min_col_value is not None and max_col_value is not None and min_col_value < max_col_value:
                        if min_col_value >= 0 and max_col_value <= 1:
                            direct_float_cols.append(col)
                        elif max_col_value > 100:
                            lisan_float_cols.append(col)
                        else:
                            norm_float_cols.append(';'.join([col, str(min_col_value), str(max_col_value)])) # 正则化记录最大最小值 避免二次计算
                    else:
                        logging.info("{} has no min or max value".format(col))

        return (
                    direct_float_cols,
                    lisan_float_cols,
                    norm_float_cols
                )

    def _get_bucket_cutter(self, columns, wf):
        ### 等频分桶方法. 输入原始spark dataframe和需要分桶的特征list
        ### pyspark3的QuantileDiscretizer支持inputCols 同时对多列进行分桶; pyspark2不行
        # 2023.3.16 + + +
        #   abs_lisan\slice_binning\fillna_binning
        for col in columns:
            _col = col.split(";")[0]
            
                # 如果列的数据类型不是数值类型，尝试转换为浮点数
                
            # 如果列的数据类型不是数值类型，尝试转换为浮点数
            from pyspark.sql.types import StringType, FloatType
            from pyspark.sql.functions import col as spark_col, isnan, isnull
            if isinstance(self.feat.schema[_col].dataType, StringType):
                self.feat = self.feat.withColumn(_col, spark_col(_col).cast(FloatType()))
            
            
            discretizer = QuantileDiscretizer(
                numBuckets=self.bucket_num,
                inputCol=_col,
                outputCol=_col+"buk",
                relativeError=1e-5,
                handleInvalid='keep'
            )

            binning_suffix = ''
            if ';' in col: # 3.16加 判定最终使用哪个分桶算子
                binning_suffix = col.split(";")[1]
            if not binning_suffix: # 空值不考虑在分桶的计算中
                tmp_feat = self.feat.select(_col)
                operator_type = 'S_slice_binning'

            elif binning_suffix == 'abs': # 算子会将特征值取绝对值 所以这里分桶计算也转为绝对值
                tmp_feat = self.feat.selectExpr("abs({col}) as {col}".format(col = _col))
                operator_type = 'S_abs_lisan'

            else: # 缺失值填充分桶
                tmp_feat = self.feat.selectExpr("coalesce({col}, {nd}) as {col}".format(col = _col, nd = float(binning_suffix)))
                operator_type = 'S_fillna_binning'
                
                
                # 检查转换后的列是否存在空值或非数字值
            count_nan_or_null = self.feat.filter((isnan(_col)) | isnull(_col)).count()
            if count_nan_or_null > 0:
                fill_value = 0.0 if binning_suffix == '' else float(binning_suffix)
                print("Column {} contains null or NaN values. Filling with value: {}".format(_col, fill_value))
                self.feat = self.feat.fillna({ _col: fill_value })
                tmp_feat = self.feat.select(_col)
                
            print(tmp_feat,"*"*20)
            bucketizer = discretizer.fit(tmp_feat)
            splits = bucketizer.getSplits()
            logging.info('{} original splits: {}'.format(col, ', '.join([str(ele) for ele in splits])) )
            if len(splits) > 2:
                splits = [ ele for ele in splits[1:-1] ]
                split_args = ','.join( [str(ele) for ele in splits] )
                content = "name=i_lisan_{col};class={oper};depend={col};args={args}".format(
                    col = _col, oper = operator_type, args = split_args)
                if operator_type == 'S_fillna_binning':
                    content += ";default:{}\n".format(str(float(binning_suffix)))
                else:
                    content += "\n"
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
                wf.write( content )
            else:
                logging.info("{} bucket cutter not valid".format(col))
        return
        
    '''
    def _get_bucket_cutter(self, columns, wf):
        ### 等频分桶方法. 输入原始spark dataframe和需要分桶的特征list
        ### pyspark3的QuantileDiscretizer支持inputCols 同时对多列进行分桶; pyspark2不行
        # 2023.3.16 + + +
        #   abs_lisan\slice_binning\fillna_binning
        ops_meta = []
        transform_meta = []
        for col in columns:
            _col = col.split(";")[0]

            binning_suffix = ''
            if ';' in col: # 3.16加 判定最终使用哪个分桶算子
                binning_suffix = col.split(";")[1]

            if not binning_suffix: # 空值不考虑在分桶的计算中
                operator_type = 'S_slice_binning'
                exe_cmd = F.col(_col)

            elif binning_suffix == 'abs': # 算子会将特征值取绝对值 所以这里分桶计算也转为绝对值
                exe_cmd = F.abs(F.col(_col)).alias(_col)
                operator_type = 'S_abs_lisan'

            else: # 缺失值填充分桶
                #tmp_feat = self.feat.selectExpr("coalesce({col}, {nd}) as {col}".format(col = _col, nd = float(binning_suffix)))
                exe_cmd = F.coalesce(F.col(_col), F.lit(float(binning_suffix))).alias(_col)
                operator_type = 'S_fillna_binning'

            ops_meta.append(operator_type)
            transform_meta.append(exe_cmd)
        input_cols = [col.split(';')[0] for col in columns]

        logging.info('ops_meta: {}'.format(ops_meta))
        logging.info('transform_meta: {}'.format(transform_meta))
        logging.info('input_cols: {}'.format(input_cols))

        transformed_df = self.feat.select(*transform_meta)
        
        discretizer = QuantileDiscretizer(
                numBuckets=self.bucket_num,
                relativeError=1e-5,
                handleInvalid='keep'
            ).setInputCols(input_cols) \
             .setOutputCols(['{}buk'.format(x) for x in input_cols]) \
             .fit(transformed_df)

        splits_mat = discretizer.getSplitsArray()
        for idx, _col in enumerate(input_cols):
            splits = splits_mat[idx]
            operator_type = ops_meta[idx]

            logging.info('{} original splits: {}'.format(col, ', '.join([str(ele) for ele in splits])) )
            if len(splits) > 2:
                splits = [ ele for ele in splits[1:-1] ]
                split_args = ','.join( [str(ele) for ele in splits] )
                content = "name=i_lisan_{col};class={oper};depend={col};args={args}".format(
                    col = _col, oper = operator_type, args = split_args)
                if operator_type == 'S_fillna_binning':
                    content += ";default:{}\n".format(str(float(binning_suffix)))
                else:
                    content += "\n"
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
                wf.write( content )
            else:
                logging.info("{} bucket cutter not valid".format(col))
        return
        '''

    def _get_kmeans_bucket_cutter(self, columns, wf):
        ### 聚类分桶方法. 输入原始spark dataframe和需要分桶的特征list
        ### 因为需要pandas dataframe, 所以对输入数据预先做采样
        # 2023.3.16 + + + 
        #   abs_lisan\slice_binning\fillna_binning

        ops_meta = []
        transofrm_meta = []

        for col in columns:
            _col = col.split(";")[0]
            
            binning_suffix = ''
            if ';' in col: # 3.16加 判定最终使用哪个分桶算子
                binning_suffix = col.split(";")[1]
            
            if not binning_suffix: # 空值不考虑在分桶的计算中
                #splits = KMeansMerge(feature=x, target=y, n_bins=self.bucket_num) 
                operator_type = 'S_slice_binning'
                ops_meta.append(operator_type)
                transofrm_meta.append(F.col(_col).alias(_col))

            elif binning_suffix == 'abs': # 算子会将特征值取绝对值 所以这里分桶计算也转为绝对值
                #splits = KMeansMerge(feature=x.abs(), target=y, n_bins=self.bucket_num)
                operator_type = 'S_abs_lisan'
                ops_meta.append(operator_type)
                transofrm_meta.append(F.abs(F.col(_col)).alias(_col))

            else: # 缺失值填充分桶
                #splits = KMeansMerge(feature=x.fillna(float(binning_suffix)), target=y, n_bins=self.bucket_num)
                operator_type = 'S_fillna_binning'
                ops_meta.append(operator_type)
                transofrm_meta.append(F.when(F.col(_col).isNull(), 
                                      float(binning_suffix)).otherwise(F.col(_col)).alias(_col))
        
        __transormed_df = self.feat.select(transofrm_meta)

        if self.opt.hdfs_cache_dir != "":
            logging.info("write  to cache path: {}".format(self.opt.hdfs_cache_dir))
            __transormed_df.write.mode("overwrite").parquet(__wpath)
            __transormed_df = spark.read.parquet(__wpath)

        bucketer = buckets.KmeansBuckets(num_buckets=self.bucket_num, 
                                         max_iter=10).fit(__transormed_df)
        splits = bucketer.get_splits()

        for col, operator_type in zip(columns, ops_meta):
            _col = col.split(";")[0]
            _split = splits[_col]

            if len(_split) >= 1:
                split_args = ','.join( [str(ele) for ele in _split] )
                content = "name=i_lisan_{col};class={oper};depend={col};args={args}\n".format(
                    col = _col, oper = operator_type, args = split_args)
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
                wf.write(content)
            else:
                logging.info("{} bucket cutter not valid".format(_col))
                    
        return

    def get_onehot_fea(self, onehot_fea_list, wf):
        ### onehot方法. 默认用户不需要自定义维度(若自定义可以在生成配置文件之后手动改)
        logging.info(">>> hash features: {}<<<".format(",".join(onehot_fea_list)))
        count_distinct = self.feat.agg(*[F.countDistinct(c).alias(c) for c in onehot_fea_list]).collect() # count distinct会抛弃空值. 现在onehot使用murmurhash有问题，尽量多分
        if count_distinct:
            count_distinct_result = count_distinct[0].asDict()
            for col in onehot_fea_list:
                col_arg = (count_distinct_result[col] // 5 + 1) * 5
                content = "name=i_onehot_{col};class=S_hash_binning;depend={col};args={arg}\n".format(col = col, arg = col_arg)
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
                wf.write(content)
        return

    def get_enum_fea(self, enum_fea_list, wf):
        ### 2023.3.16补充
        logging.info(">>> enum features: {}<<<".format(",".join(enum_fea_list)))
        enum_collect = self.feat.agg(*[F.collect_set(c).alias(c) for c in enum_fea_list]).collect()
        if enum_collect:
            enum_collect_result = enum_collect[0].asDict()
            for col in enum_fea_list:
                col_arg = ','.join([str(ele) for ele in enum_collect_result[col]])
                content = "name=i_enum_{col};class=S_enum_binning;depend={col};args={arg}\n".format(col = col, arg = col_arg)
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
                wf.write(content)
        return

    def get_lisan_fea(self, lisan_fea_list, wf):
        # bucket_type: cluster - 聚类分桶 ; 其他 - 等频分桶(默认)
        # 2023.3.16 + + +
        #   lisan_fea_list格式：fea1;null_default,fea2;fea3;abs
        #   ;第二位为'abs'时使用原来的abs_lisan，否则为缺失值填充值选择使用fillna_binning，未指定该位置则默数值离散化slice_binning

        # 分桶的计算效率低 @todo

        logging.info(">>>lisan features: {}<<<".format(",".join(lisan_fea_list)))
        if self.bucket_type == "cluster":
            self._get_kmeans_bucket_cutter(lisan_fea_list, wf)
        else:
            self._get_bucket_cutter(lisan_fea_list, wf)
        return


    def get_direct_fea(self, direct_fea_list, wf):
        for col in direct_fea_list:
            content = "name=i_direct_{col};class=S_direct;depend={col}\n".format(col = col)
            wf.write(content)
            logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
        return

    def _get_const_fea(self, arg_name, arg_value, wf):
        # constant 中间计算用特征 不输出
        content = "name=m_cons_{name};class=S_const;args={value};hide=1;addcol=1\n".format(name=arg_name, value=arg_value)
        logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
        wf.write(content)

    def get_days_span_fea(self, days_fea_list, fix_ts_col, wf):
        ### rta常用的【距今天数】类特征; 其中 feature_list 为时间戳类的字段; fix_feature 为当前日期时间戳
        ### 默认做分桶; 在使用过程中通常是算天数后分桶, 之前的计算都为中间不输出

        self._get_const_fea("second", 86400, wf)

        for col in days_fea_list:
            content = "name=m_sub_{col};class=S_sub;depend={fix},{col};hide=1;addcol=1\n".format(col=col, fix=fix_ts_col)
            wf.write(content)
            logging.info('>>>>>>>>>>>>>>>>{}'.format(content))
            content = "name=m_days_{col};class=S_divide;depend=m_sub_{col},m_cons_second;hide=1;addcol=1\n".format(col = col)
            wf.write(content)
            logging.info('>>>>>>>>>>>>>>>>{}'.format(content))

        ts_feature_list = ["m_days_{col}".format(col=ele) for ele in days_fea_list]
        self.get_lisan_fea(ts_feature_list, wf)
        # print("name=i_lisan_days_{col};class=S_abs_lisan;depend=m_days_{col};args=5,10,20,50,100".format(col = col))
        # @todo 这里不好写 日期的分桶无法从原始sql拿到特征来计算 需要单独算一下 —— 先写死

    def get_minmax_fea(self, minmax_fea_list, wf, float_calculate_flag = 0):
        ### 归一化计算. 从采样的df算出当前特征的最大最小值, (x-min)/(max-min)

        if float_calculate_flag == 0: # 只传特征列表
            min_max = self.feat.selectExpr(*minmax_fea_list).agg(
                *[(F.max(c) - F.min(c)).alias("cons_{}".format(c)) for c in minmax_fea_list],
                *[F.min(c).alias("min_{}".format(c)) for c in minmax_fea_list]
            ).collect()

            if min_max:
                min_max_result = min_max[0].asDict()
            else:
                return
        else: # 特征列表中带最小最大值 - subdivision中计算
            minmax_fea_list = [item.split(";")[0] for item in minmax_fea_list]
            fea_min_arr = [float(item.split(";")[2]) for item in minmax_fea_list]
            fea_max_arr = [float(item.split(";")[2]) for item in minmax_fea_list]
            min_max_result = {}
            for i in range(minmax_fea_list):
                min_max_result["min_{}".format(minmax_fea_list[i])] = fea_min_arr[i]
                min_max_result["cons_{}".format(minmax_fea_list[i])] = fea_max_arr[i] - fea_min_arr[i]

        for col in minmax_fea_list:
            current_min = min_max_result["min_{}".format(col)]
            current_cons = min_max_result["cons_{}".format(col)]
            if current_cons > 0:
                self._get_const_fea("{col}_min".format(col=col), current_min, wf)
                self._get_const_fea("{col}_deno".format(col=col),  current_cons, wf)

                content = "name=m_sub_{col}_min;class=S_sub;depend={col},m_cons_{col}_min;hide=1;addcol=1\n".format(col = col)
                wf.write(content)
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))

                content = "name=i_norm_{col};class=S_divide;depend=m_sub_{col}_min,m_cons_{col}_deno\n".format(col = col)
                wf.write(content)
                logging.info('>>>>>>>>>>>>>>>>{}'.format(content))

    def run(self):
        logging.info("writing into {}...".format( self.parser_conf_file))
        with open(self.parser_conf_file, 'w') as writer:
            writer.write("fea_cols={}\n".format(','.join(self.feat.columns)))
            writer.write("feature_list={}\n".format(self.feature_conf_file))
            writer.write("label={}\n".format(self.label_col))
            writer.write("add_token={}\n".format(self.token_col))
            writer.write("debug=0\n")

        # 踢掉只有一个值的特征 后面不再做计算和输出
        #count_distinct = self.feat.agg(*[F.countDistinct(F.expr("coalesce({}, -9999.0)".format(c))).alias(c) for c in self.feat.columns]).collect()[0].asDict() # count distinct会抛弃空值 填一下
        #remove_cols = [k for k, v in count_distinct.items() if v <= 1]
        remove_cols = []
        logging.info(f">>> these columns only got one value: <<<".format(','.join(remove_cols)))

        onehot_fea_list = [item for item in self.onehot_fea_list.split(",") if item not in remove_cols + ['']]
        enum_fea_list = [item for item in self.enum_fea_list.split(",") if item not in remove_cols + ['']]
        lisan_fea_list = [item for item in self.lisan_fea_list.split(",") if item.split(";")[0] not in remove_cols + ['']]
        direct_fea_list = [item for item in self.direct_fea_list.split(",") if item not in remove_cols + ['']]
        minmax_fea_list = [item for item in self.minmax_fea_list.split(",") if item not in remove_cols + ['']]
        days_fea_list = [item for item in self.days_fea_list.split(",") if item not in remove_cols + ['']]

        float_calculate_flag = 0 # 0表示用户传 1表示自己分割计算
        if not lisan_fea_list and not direct_fea_list and not minmax_fea_list and self.float_calculate_default == 1: # 用户未输入情况下进行计算 - 已经剔除只有一个值的特征情况下
            logging.info("calculating float type features subdivision...")
            remove_cols = remove_cols \
                + onehot_fea_list \
                + enum_fea_list \
                + days_fea_list \
                + self.label_col.split(",") \
                + self.token_col.split(",") \
                + [self.fix_ts_col]

            subdivision_cols = [col[0] for col in self.feat.dtypes if col[0] not in remove_cols and col[1] != 'string']
            direct_fea_list, lisan_fea_list, minmax_fea_list = self._get_float_subdivision(subdivision_cols)
            float_calculate_flag = 1

        logging.info("writing into {}...".format( self.feature_conf_file ))
        with open(self.feature_conf_file, 'w') as writer:
            if onehot_fea_list: self.get_onehot_fea(onehot_fea_list, writer)
            if enum_fea_list: self.get_enum_fea(enum_fea_list, writer)
            if lisan_fea_list: self.get_lisan_fea(lisan_fea_list, writer)
            if direct_fea_list: self.get_direct_fea(direct_fea_list, writer)
            if minmax_fea_list: self.get_minmax_fea(minmax_fea_list, writer, float_calculate_flag)
            if days_fea_list: self.get_days_span_fea(days_fea_list, self.fix_ts_col, writer)


if __name__ == '__main__':

    TF211_PY_ENV="hdfs://difed/user/dm/ENV/dlflow/v2/py3.7.15_for_tfcluster.tgz"
    TF211_PY_BIN="./py3.7.15_for_tfcluster.tgz/py3.7.15_for_tfcluster/bin/python3"
    TFRECORD_CONNENV="hdfs://difed/user/dm/ENV/jars/spark_tfrecord/compile/spark-tfrecord-0.5.1_scala2.12-spark3.2.0-tfhp1.15.0.jar"

    print("[RUN]", ' '.join(sys.argv))

    parser = argparse.ArgumentParser(description='Lite-Fea Param Handler')
    # 输入参数
    parser.add_argument('-f', '--sql_file', help='sql文件名(脚本目录下面的相对路径)', type=str, required=True)
    parser.add_argument('--sql_param', help="sql需要传入的参数;分割", default='')
    parser.add_argument('--is_sample', help='3.16补充 是否对结果进行采样. 聚类分桶建议采样', default=1)
    parser.add_argument('--conf_wf', help='写入的feature_list.conf文件名', default='feature_list.conf')
    parser.add_argument('--parser_wf', help='写入的parser.conf文件名', default='parser.conf')
    parser.add_argument('--onehot_fea', help='需要做onehot处理的特征列表,分割 3.16修改算子S_hash_binnning可选xxhash/wyhash', default='')
    parser.add_argument('--enum_fea', help='3.16补充 字符串类型特征 根据枚举值进行分箱', default='')

    parser.add_argument('--lisan_fea', help='需要做分桶处理的特征列表,分割 3.16补充:特征名;后面设置控制默认值或者abs', default='')
    parser.add_argument('--days_fea', help='需要计算日期距今天数并分桶的特征列表', default='')
    parser.add_argument('--fix_ts_col', help='计算天数固定的日期ts字段', default='')
    parser.add_argument('--minmax_fea', help='需要做归一化处理的特征列表', default='')
    parser.add_argument('--direct_fea', help='需要直接输出的特征列表', default='')
    parser.add_argument('--bucket', help='分桶方式 cluster或者其他', default='freq')
    parser.add_argument('--bucket_num', help='分桶数 等频分桶需要指定', default=20)
    parser.add_argument('--label', help='label列 支持多label', default='label')
    parser.add_argument('--token', help='token列 支持多token',default='pid')
    parser.add_argument('--float_calculate_default', help='是否支持自定义float类型特征的计算', default=0)

    parser.add_argument('--log4j', help='spark --files后面的log文件地址', default='log4j.properties')
    parser.add_argument('--mode', help='spark是用什么模式在运行,默认client模式', type=str,
                        choices=["cluster", "client"], default='client')
    parser.add_argument('--app_name', help='spark app name', type=str, default='gen_conf_v1')
    parser.add_argument('--queue', help="yarn queue name", type=str, required=True)
    parser.add_argument('--user_name', help="hadoop user name for hdfs path", type=str, required=True)
    parser.add_argument('--cores', help="spark core fot each executor", type=str, default=2)
    parser.add_argument('--minExecutors', help="spark max executors", type=str, default=10)
    parser.add_argument('--maxExecutors', help="spark max executors", type=str, default=40)
    parser.add_argument('--executorMemory', help="spark executors memory", type=str, default="2g")
    parser.add_argument('--hdfs_cache_dir', help="hdfs cache tmp data", type=str, default="")

    # 开关
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
    # parser.add_argument("-d", "--debug", help="debug mode ()", action="store_true")

    opt = parser.parse_args()

    logging.info("输入的CLI参数如下:")
    pprint.pprint(vars(opt))

    # ShowSparkVersion.run()

    if opt.app_name != 'default':
        SPARK_PREFIX = opt.app_name

    config = Config(
        queue=opt.queue,
        user_name=opt.user_name,
        min_executors=opt.minExecutors,
        max_executors=opt.maxExecutors,
        executor_memory=opt.executorMemory,
        cores=opt.cores
    )
    logging.info(config)

    spark = SparkSession \
        .builder \
        .appName(opt.app_name) \
        .master("yarn") \
        .config("spark.submit.deployMode", "client") \
        .config("spark.driver.memory", "4g") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", f"{config.minExecutors}") \
        .config("spark.dynamicAllocation.maxExecutors", f"{config.maxExecutors}") \
        .config("spark.executor.cores", f"{config.cores}") \
        .config("spark.executor.memory", f"{config.executorMemory}") \
        .config("spark.default.parallelism", f"{config.parallelism}") \
        .config("spark.sql.shuffle.partitions", "1000") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .config("spark.yarn.queue", f"{config.YARN_QUEUE}") \
        .config("spark.driver.extraJavaOptions", "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS") \
        .config("spark.executor.extraJavaOptions", "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS") \
        .config("spark.yarn.dist.archives",
                f"hdfs://difed/user/{config.HADOOP_USER}/ENV/1.0.1/py3.7.5_for_tfcluster.tgz") \
        .config("spark.jars",
                f"hdfs://difed/user/{config.HADOOP_USER}/ENV/jars/spark-tensorflow-connector_2.11-1.14.0.jar,hdfs://difed/user/{config.HADOOP_USER}/ENV/jars/tensorflow-hadoop-1.14.0.jar") \
        .config("spark.executorEnv.PYSPARK_PYTHON", "./py3.7.5_for_tfcluster.tgz/py3.7.5_for_tfcluster/bin/python3") \
        .enableHiveSupport() \
        .getOrCreate()

        #.config("spark.yarn.dist.archives", f"{TF211_PY_ENV}") \
        #.config("spark.jars", f"{TFRECORD_CONNENV}") \
        #.config("spark.executorEnv.PYSPARK_PYTHON", f"{TF211_PY_BIN}") \
        #.enableHiveSupport() \
        #.getOrCreate()

    print("[Info] Spark Version : {}".format(spark.version))

    # 准备执行Runner Spark任务
    FeatureHandler(
        spark,
        sql_file=opt.sql_file,
        sql_param=opt.sql_param,
        is_sample=opt.is_sample,
        feature_conf_file=opt.conf_wf,
        parser_conf_file=opt.parser_wf,
        onehot_fea_list=opt.onehot_fea,
        enum_fea_list=opt.enum_fea,
        lisan_fea_list=opt.lisan_fea,
        days_fea_list=opt.days_fea,
        fix_ts_col=opt.fix_ts_col,
        minmax_fea_list=opt.minmax_fea,
        direct_fea_list=opt.direct_fea,
        bucket_type=opt.bucket,
        bucket_num=opt.bucket_num,
        label_col=opt.label,
        token_col=opt.token,
        float_calculate_default=opt.float_calculate_default
        ).run()