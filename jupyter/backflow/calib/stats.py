import pyspark.sql.types as T
from pyspark.sql import functions as func

from ..programming import pgt
from ..utils import tools

def simple_gap(df, prob_col, label_col, add_group_cols=None, add_agg=None):
    buck = pgt.bucketize_freq(df=df.filter("{} is not null".format(label_col)), 
                              models=[prob_col])
    
    agg_cmd = [
        func.count("*").alias("ingroup_cnt"),
        func.sum(label_col).alias("cvt"),
        (func.sum(label_col) / func.count("*")).alias("tCVR"),
        func.mean(prob_col).alias(prob_col),
    ]
    
    if add_agg is not None:
        agg_cmd += add_agg
    
    __group_cols = ["{}_freq_bucket".format(prob_col)]
    if add_group_cols is not None:
        __group_cols += add_group_cols
        
    stats = buck.groupBy(*__group_cols).agg(*agg_cmd).toPandas()
    
    stats['prob_col'] = prob_col
    return stats

def decide_threshold(df, threshold_colname, inference_dt, window_size=7, 
                    include_td=False,
                    verbose=True,
                     num_buckets=10,
                     rank_method='desc',
                     ignore_dts=[],
                    **kwargs
                    ):
    decide_threshold_dts = tools.smart_generate_dt_list(start_dt=inference_dt, 
                                                        degree_days=window_size+1, 
                                                        flag=-1,
                                                        ignore_dts=ignore_dts
                                                        )        
    if not include_td:
        decide_threshold_dts = decide_threshold_dts[:-1]
        
    if len(decide_threshold_dts)==1:
        decide_threshold_dts.append("unk")
    
    if verbose:
        print("*"*40)
        print("using threshold name:{}".format(threshold_colname))
        print("ignore dts:{}".format(ignore_dts))
        print("using threshold val dt:{}".format(inference_dt))
        print("threshold dt:{}".format(decide_threshold_dts))
        print("using rank method:{}".format(rank_method))
        

    __decide_set = df.filter("dt in {}".format(tuple(dt for dt in decide_threshold_dts))) \
                     .filter("{} is not null".format(threshold_colname))

    if verbose and 'group_check_cols' in kwargs:
        print("using group_check_cols:{}".format(kwargs['group_check_cols']))
        __decide_set.groupBy(kwargs['group_check_cols']).count().show()

    __level_bk_final = pgt.bucketize_freq(df=__decide_set, 
                                          models=[threshold_colname], 
                                          N=num_buckets, 
                                    order_method=rank_method) \
                          .withColumnRenamed("{}_freq_bucket".format(threshold_colname), 'bucket')

    stats = __level_bk_final.groupBy('bucket').agg(
                *[func.min(func.col(threshold_colname)).alias('min_threshold'),
                  func.max(func.col(threshold_colname)).alias('max_threshold')
                 ]
        ).toPandas()

    __threshold_map = {}
    for value in stats.values:
        __threshold_map[int(value[0])] = [float(value[1]), float(value[2])]
    
    if verbose:
        print(__threshold_map)
    
    return __threshold_map


def stats_group_ratio(df, group_cols):
    __bkt_dist = df.groupBy(*group_cols).count().toPandas()
    __bkt_dist['ratio'] = __bkt_dist['count'] / __bkt_dist['count'].sum()
    __bkt_dist = __bkt_dist.sort_values(by=group_cols[0])
    
    return __bkt_dist
    