from pyspark.sql   import functions as func
from pyspark.sql   import DataFrame
from pyspark.sql.window import Window
from functools import reduce


from . import pgt
from ..calib import stats as calib_stats
from ..calib import metrics as calib_metric
from . import pgt


def logging(info):
    print(info)


def water_simulation(std_pdf, enhance_pdf, top_level, drop_level,
                    replace_buckets=None):
    """
    
    """
    
    print("省钱券(主券):{}".format(std_pdf['fix_group_name'].values[0]))
    print("花钱券(副券):{}".format(enhance_pdf['fix_group_name'].values[0]))
    
    
    joint_key = ['bucket']
    replace_cols = ['cvr','avg_cac','delta_cvr','pv_cvr','pv_delta_cvr']
    
    if replace_buckets is not None:
        print("使用跳跃选桶")
        enhancer = enhance_pdf[enhance_pdf['bucket'].isin(replace_buckets)].copy()
    else:
        enhancer = enhance_pdf[enhance_pdf['bucket']>top_level].copy()
        
    enhancer = enhancer[joint_key+replace_cols]
    print("待兑入数据分桶:{}".format(enhancer['bucket'].tolist()))
    
    exchange_droped = std_pdf[std_pdf['bucket']>=drop_level]
    exchange_left   = std_pdf[std_pdf['bucket']<drop_level]
    print("原始数据集丢弃桶:{}".format(exchange_droped['bucket'].tolist()))
    print("原始数据集保留桶:{}".format(exchange_left['bucket'].tolist()))
    
    
    natual_ratio      = exchange_droped['c_finish_ord_uv_3d'].sum() / exchange_droped['ct_ingroup_cnt'].sum()
    natual_pv_ratio   = exchange_droped['c_finish_ord_pv_3d'].sum() / exchange_droped['ct_ingroup_cnt'].sum()
    if str(natual_ratio) == 'nan':
        natual_ratio = 0
        natual_pv_ratio = 0

    natual_finish_ord = natual_ratio * exchange_droped['ingroup_cnt'].sum()
    natual_finish_ord_pv = natual_pv_ratio * exchange_droped['ingroup_cnt'].sum()
    print("丢弃桶自然完单数:{}".format(natual_finish_ord))
    
    
    #开始兑水
    if replace_buckets is not None:
        exchange_upper = exchange_left[~exchange_left['bucket'].isin(replace_buckets)]
        exchange_lower = exchange_left[exchange_left['bucket'].isin(replace_buckets)]
        
    else:
        exchange_upper = exchange_left[exchange_left['bucket']<=top_level]
        exchange_lower = exchange_left[exchange_left['bucket']>top_level]
        
    print('原始数据集替换桶:', exchange_lower['bucket'].tolist())
    
    exchange_lower = exchange_lower.drop(replace_cols, axis=1) \
                                  .merge(enhancer, on=joint_key, how='left')

    print("原始数据集shape:{}\t兑入数据集shape:{}".format(exchange_upper.shape, exchange_lower.shape))
    
    exchange = exchange_upper.append(exchange_lower, ignore_index=True)

    exchange['finish_ord_uv_3d']       = exchange['cvr'].astype(float) * exchange['ingroup_cnt'].astype(float)
    exchange['finish_ord_pv_3d']       = exchange['pv_cvr'].astype(float) * exchange['ingroup_cnt'].astype(float)

    exchange['total_cost']             = exchange['avg_cac'].astype(float) * exchange['ingroup_cnt'].astype(float)

    exchange['delta_finish_ord_uv_3d'] = exchange['delta_cvr'].astype(float) * exchange['ingroup_cnt'].astype(float)
    exchange['delta_finish_ord_pv_3d'] = exchange['pv_delta_cvr'].astype(float) * exchange['ingroup_cnt'].astype(float)
    
    exchange['delta_cac']    = exchange['total_cost'] / exchange['delta_finish_ord_uv_3d']
    exchange['pv_delta_cac'] = exchange['total_cost'] / exchange['delta_finish_ord_pv_3d']
    exchange['cac']          = exchange['total_cost'] / exchange['finish_ord_uv_3d']
    exchange['pv_cac']       = exchange['total_cost'] / exchange['finish_ord_pv_3d']
    
    print('\n')
    
    return exchange, natual_finish_ord, natual_finish_ord_pv
    
def call_result(pdf, natual_finish_ord=0, natual_finish_ord_pv=0,
                expected_ratio=0.03
               ):
    total_cost           = pdf['total_cost'].astype(float).sum()
    finish_ord_uv        = pdf['finish_ord_uv_3d'].astype(float).sum()
    finish_ord_pv        = pdf['finish_ord_pv_3d'].astype(float).sum()

    delta_finish_ord_uv  = pdf['delta_finish_ord_uv_3d'].sum()
    delta_finish_ord_pv  = pdf['delta_finish_ord_pv_3d'].sum()

    total_ords    = finish_ord_uv  + natual_finish_ord
    total_pv_ords = finish_ord_pv + natual_finish_ord_pv
    
    delta_cac    = total_cost / (delta_finish_ord_uv)
    pv_delta_cac = total_cost / (delta_finish_ord_pv)
    
    
    total_ords    = round(total_ords, 0)
    total_pv_ords = round(total_pv_ords, 0)
    total_cost    = round(total_cost,2)
    delta_cac     = round(delta_cac, 2)
    pv_delta_cac  = round(pv_delta_cac,2)

    return [total_ords, total_pv_ords,total_cost,delta_cac, pv_delta_cac]

def explain_metric(water_result, normal_result, title, rank_method='', log=True):
    if log:
        print("排序公式:")
        print(rank_method)
        print('\n')
        print("*"*20, title, "*"*20)

        print('完单uv 提升:{:.2f}'.format(water_result[0] / normal_result[0]))
        print('完单pv 提升:{:.2f}'.format(water_result[1] / normal_result[1]))
        print('总cost提升:{:.2f}'.format(water_result[2] / normal_result[2]))
        print('delta cac 提升:{:.2f}'.format(water_result[3] / normal_result[3]))
        print('pv delta cac 提升:{:.2f}'.format(water_result[4] / normal_result[4]))

        print("*"*40)
        print('\n')
    
    return [water_result[0] / normal_result[0], 
           water_result[1] / normal_result[1], 
           water_result[2] / normal_result[2], 
           water_result[3] / normal_result[3], 
           water_result[4] / normal_result[4]]


class CausalSimMetrics(object):
    def __init__(self, valdts, verbose=False, threshold_window_size=7, include_td=False,
                num_buckets=10, freeze_inference_dt=None,
                hdfs=None,
                spark=None,
                param_obj=None,
                tag='unique_tag',
                overwrite=True,
                bkt_metrci_sqlcmd='',
                pre_bkt_filter_cmd='app_valid_tag=0',
                drop_buckets = [],
                keep_ckpt=True,
                ignore_dts=['2024-04-04','2024-04-05','2024-04-06'],
                metric_ignore_dts=[],
                accelerate=False

                ):
        self.verbose               = verbose
        self.threshold_window_size = threshold_window_size
        self.include_td            = include_td
        self.num_buckets           = num_buckets
        self.valdts                = valdts
        self.freeze_inference_dt   = freeze_inference_dt
        self.accelerate            = accelerate

        self.hdfs                  = hdfs
        self.spark                 = spark
        self.param_obj             = param_obj
        self.tag                   = tag
        self.overwrite             = overwrite
        self.bkt_metrci_sqlcmd     = bkt_metrci_sqlcmd
        self.pre_bkt_filter_cmd    = pre_bkt_filter_cmd
        self.drop_buckets          = drop_buckets
        self.keep_ckpt = keep_ckpt
        self.ignore_dts = ignore_dts
        self.metric_ignore_dts = metric_ignore_dts
    
    def bucket_with_threshold(self, decide_df, rank_method, threshold_colname, val_df=None):
        """
        """
        if self.accelerate and self.include_td:
            print("using accelerate mode...")
            bktd_df = pgt.bucketize_freq(df=decide_df, 
                                          models=[threshold_colname], 
                                          N=self.num_buckets, 
                                    order_method=rank_method) \
                          .withColumnRenamed("{}_freq_bucket".format(threshold_colname), 'bucket')

            return bktd_df

        bucketed_dfs = []
        for dt in self.valdts:
            __threshold_dt = dt
            
            if self.freeze_inference_dt is not None:
                __threshold_dt = self.freeze_inference_dt
                
            __threshold = calib_stats.decide_threshold(df=decide_df, 
                                                   threshold_colname=threshold_colname,
                                                   inference_dt=__threshold_dt,
                                                   verbose=self.verbose,
                                                   rank_method=rank_method,
                                                   include_td=self.include_td,
                                                   window_size=self.threshold_window_size,
                                                   num_buckets=self.num_buckets,
                                                   ignore_dts=self.ignore_dts
                                                )
                
            __valset = val_df.filter("dt = '{}'".format(dt)) 
            __valset = pgt.apply_group_threshold(__valset, threshold_colname, 
                                                __threshold, group_name='bucket')
            
            if self.verbose:
                logging(calib_stats.stats_group_ratio(__valset, group_cols=['bucket']))
            
            bucketed_dfs.append(__valset)
            
        bktd_df = reduce(DataFrame.unionAll, bucketed_dfs)
        return bktd_df
    
    def bucket_metrics(self, bktd_df, fix_group_names, threshold_colname,
                          add_addition_cmd, add_addition_cmd_cols,
                          c_fix_group_name='control_group',
                          tag='bktd'):
        """
        """

        if self.verbose:
            print("using {} as control group...".format(c_fix_group_name))
            
        __bktd_tbname = pgt.register_buck_tb(self.hdfs, 
                                             self.spark, 
                                             self.param_obj, 
                                             dts=self.valdts,
                                             tag=tag,
                                             bucked_df=bktd_df,
                                             assign_group=False,
                                             group_permutations_map=None,
                                             overwrite=self.overwrite
                        )


        __uplift_cmd = self.bkt_metrci_sqlcmd.format(
                    bucketed_tbname=__bktd_tbname,
                    valdts=tuple(dt for dt in self.valdts if dt not in self.metric_ignore_dts),
                    c_fix_group_name=c_fix_group_name,
                    rank_prob_name  = threshold_colname,
                    fix_group_names = fix_group_names,
                    add_addition_cmd=add_addition_cmd,
                    add_addition_cmd_cols=add_addition_cmd_cols,
                )

        if self.verbose:
            logging(__uplift_cmd)

        level_bk_simulation = self.spark.sql(__uplift_cmd).toPandas()
        sim_pdf = level_bk_simulation.copy()

        sim_pdf['渠道占总消耗'] = sim_pdf['channel_cost'].astype(float) / sim_pdf['total_cost'].astype(float)
        sim_pdf['券占总消耗']   = sim_pdf['coupon_cost'].astype(float) / sim_pdf['total_cost'].astype(float)

        return sim_pdf, __bktd_tbname, threshold_colname
    

class DoubleRankSimulator(CausalSimMetrics):
    def __init__(self, valdts, first_rk_cmd, second_rk_cmd, rankcolname_first='rank_prob',
                 rankcolname_second='second_rk_prob',
                 first_rk_method='asc', second_rk_method='desc', **kwargs):
        
        super(DoubleRankSimulator, self).__init__(valdts=valdts, **kwargs)
        
        self.first_rk_method    = first_rk_method
        self.second_rk_method   = second_rk_method
        self.first_rk_cmd       = first_rk_cmd
        self.second_rk_cmd      = second_rk_cmd
        self.rankcolname_first  = rankcolname_first
        self.rankcolname_second = rankcolname_second
        
   
    def double_rank(self, decide_df, valdf):
        """
        
        """
        
        hdfs_wpath = self.param_obj.get_hdfs_path(self.valdts, 'first_bktd')
        if not self.overwrite and self.hdfs.exists(hdfs_wpath):
            logging("bktd data exists, skip...")
            return self.spark.read.parquet(self.param_obj.get_hdfs_path(self.valdts, 'first_bktd')), \
                  self.spark.read.parquet(self.param_obj.get_hdfs_path(self.valdts, 'second_bktd'))
        
        bucketed_dfs = []
        to_bkt_df = decide_df
        if self.pre_bkt_filter_cmd is not None:
            logging("executing filter cmd:{}".format(self.pre_bkt_filter_cmd))
            to_bkt_df = to_bkt_df.filter(self.pre_bkt_filter_cmd)
        
        
        first_bktd = self.bucket_with_threshold(to_bkt_df, self.first_rk_method, 
                                                self.rankcolname_first, val_df=valdf)
        second_bktd = first_bktd.filter("bucket not in {}".format(tuple(bk for bk in self.drop_buckets)))
        second_bktd = second_bktd.drop(*['bucket'])
        
        to_bkt_df = decide_df
        if self.pre_bkt_filter_cmd is not None:
            logging("executing filter cmd:{}".format(self.pre_bkt_filter_cmd))
            to_bkt_df = to_bkt_df.filter(self.pre_bkt_filter_cmd)
        
        second_bktd = self.bucket_with_threshold(to_bkt_df,self.second_rk_method,
                                                 self.rankcolname_second, val_df=second_bktd)
        
        if self.keep_ckpt:
            first_bktd.write.mode("overwrite").parquet(self.param_obj.get_hdfs_path(self.valdts, 'first_bktd'))
            second_bktd.write.mode("overwrite").parquet(self.param_obj.get_hdfs_path(self.valdts, 'second_bktd'))
            
        
        return first_bktd, second_bktd
    
    def double_rank_metric(self, decide_df, valdf=None, 
                           add_addition_cmd='',
                           add_addition_cmd_cols='',
                           fix_group_names=('research_space_I','research_space_K'),
                           c_fix_group_name_2th="research_space_I"
                          ):
        if valdf is None:
            valdf = decide_df
            
        if self.verbose:
            logging("rk1:{}".format(self.first_rk_cmd))
            logging("rk2:{}".format(self.second_rk_cmd))
            
        decide_df = decide_df.selectExpr("*", self.first_rk_cmd, self.second_rk_cmd)
        valdf     = valdf.selectExpr("*", self.first_rk_cmd, self.second_rk_cmd)
        
        first_bktd, second_bktd = self.double_rank(decide_df, valdf)    
        
        add_addition_cmd_1th      = add_addition_cmd + ',' + 'avg(second_rk_prob) as second_rk_prob, avg(rank_prob) as first_rk_prob'
        add_addition_cmd_cols_1th = add_addition_cmd_cols +',' + "second_rk_prob, first_rk_prob"
        first_bktd_metric, first_bk_tbname, first_rk_probname = self.bucket_metrics(
            bktd_df=first_bktd, 
            fix_group_names=fix_group_names,
            tag='first_bktd', 
            add_addition_cmd=add_addition_cmd_1th,
            add_addition_cmd_cols=add_addition_cmd_cols_1th,
            threshold_colname=self.rankcolname_first
        )
        
        add_addition_cmd_2th      = add_addition_cmd + ',' + 'avg(rank_prob) as first_rk_prob'
        add_addition_cmd_cols_2th = add_addition_cmd_cols +',' + "first_rk_prob"
        second_bktd_metric, second_bk_tbname, second_rk_probname = self.bucket_metrics(
            bktd_df=second_bktd, 
            fix_group_names=fix_group_names,
            tag='second_bktd', 
            add_addition_cmd=add_addition_cmd_2th,
            add_addition_cmd_cols=add_addition_cmd_cols_2th,
            threshold_colname=self.rankcolname_second,
            c_fix_group_name=c_fix_group_name_2th
        )
        
        return first_bktd_metric, second_bktd_metric


class SingleRankSimylator(CausalSimMetrics):
    def __init__(self, valdts, first_rk_cmd, rankcolname_first='rank_prob',
                 first_rk_method='asc', 
                 save_tag='single_first_bktd',
                  **kwargs):
        
        super(SingleRankSimylator, self).__init__(valdts=valdts, **kwargs)
        
        self.first_rk_method    = first_rk_method
        self.first_rk_cmd       = first_rk_cmd
        self.rankcolname_first  = rankcolname_first
        self.save_tag           = save_tag
        

    def single_rank(self, decide_df, valdf):
        hdfs_wpath = self.param_obj.get_hdfs_path(self.valdts, self.save_tag)

        if not self.overwrite and self.hdfs.exists(hdfs_wpath):
            logging("bktd data exists, skip...")
            return self.spark.read.parquet(self.param_obj.get_hdfs_path(self.valdts, self.save_tag))
        
        bucketed_dfs = []
        to_bkt_df = decide_df
        if self.pre_bkt_filter_cmd is not None:
            logging("executing filter cmd:{}".format(self.pre_bkt_filter_cmd))
            to_bkt_df = to_bkt_df.filter(self.pre_bkt_filter_cmd)
        
        
        first_bktd = self.bucket_with_threshold(to_bkt_df, self.first_rk_method, 
                                                self.rankcolname_first, val_df=valdf)

        if self.keep_ckpt:
            first_bktd.write.mode("overwrite").parquet(self.param_obj.get_hdfs_path(self.valdts, self.save_tag))

        return first_bktd

    def single_rank_metric(self, decide_df, valdf=None, 
                           add_addition_cmd='',
                           add_addition_cmd_cols='',
                           c_fix_group_name='research_space_I',
                           fix_group_names=('research_space_I','research_space_K'),
                           return_bktd=False   
                        ):
        
        if valdf is None:
            valdf = decide_df
            
        if self.verbose:
            logging("rk1:{}".format(self.first_rk_cmd))
            
        decide_df = decide_df.selectExpr("*", self.first_rk_cmd)
        valdf     = valdf.selectExpr("*", self.first_rk_cmd)
        
        first_bktd = self.single_rank(decide_df, valdf)    
        
        first_bktd_metric, first_bk_tbname, first_rk_probname = self.bucket_metrics(
            bktd_df=first_bktd, 
            fix_group_names=fix_group_names,
            tag=self.save_tag,
            threshold_colname=self.rankcolname_first,
            add_addition_cmd=add_addition_cmd,
            add_addition_cmd_cols=add_addition_cmd_cols,
            c_fix_group_name=c_fix_group_name
        )

        if return_bktd:
            return first_bktd_metric, first_bktd
        return first_bktd_metric