
from pyspark.files import SparkFiles
from pyspark.sql import functions as func
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

from functools import reduce

import pyspark.sql.types as T
import pandas as pd
import numpy as np
import pyspark
import os
import itertools
import random
import math
import sys
import json

from ortools.linear_solver import pywraplp


def bucketize_freq(df, models=['bid_price'], 
                   order_method='desc', 
                   N=10,
                   additional_bucket_col=None, 
                   verbose=False,
              ):
    stay_fields = df.columns
    
    total_cnt = df.count()
    rn_sqls = []
    bucket_sqls = []
    
    for model in models:
        _rn_sql = "row_number() over (order by {m} {a}, rand(1) {a}) as {m}_rn".format(
            m=model, a=order_method)
        rn_sqls.append(_rn_sql)
    
    for model in models:
        _buck_sql = "cast(ceil({m}_rn / {total} * {N}) as bigint) as {m}_freq_bucket".format(
            m=model, N=N, total=total_cnt)
    
        bucket_sqls.append(_buck_sql)
    
    if verbose:
        print(bucket_sqls)
    

    eval_df = df.selectExpr(
        *stay_fields,
        *rn_sqls
    ).selectExpr(
        *stay_fields,
        *bucket_sqls
    )
    
    return eval_df


def bucketize_freq_nitl(df, models=['bid_price'], 
                   order_method='desc', 
                   N=10,
                   additional_bucket_col=None, 
                   verbose=False,
              ):
    stay_fields = df.columns
    
    total_cnt = df.count()
    rn_sqls = []
    bucket_sqls = []
    
    for model in models:
        __cmd = "NTILE({N}) OVER (order by {m} {a}) as {m}_freq_bucket".format(N=N, m=model, a=order_method)
        bucket_sqls.append(__cmd)

    if verbose:
        print(bucket_sqls)
    
    eval_df = df.selectExpr(
        *stay_fields,
        *bucket_sqls
    )
    
    return eval_df


def apply_group_threshold(df, ite_name, t_group_threshold_mapping, 
                        group_name=None):
    __max_group_id = max(t_group_threshold_mapping.keys())
    if group_name is None:
        group_name = '{}_bucket'.format(ite_name)

    def __assignx(row):
        if row is None:
            return __max_group_id

        for k, v in t_group_threshold_mapping.items():
            __min = min(v[0], v[1])
            __max = max(v[0], v[1])
            if row>= __min and row<__max:
                return k
                
        return __max_group_id
            
    assignx_udf = func.udf(lambda z: __assignx(z), T.IntegerType())
    
    tmpname = '{}_tmp'.format(ite_name)
    df = df.withColumn(tmpname, assignx_udf(func.col(ite_name))) \
                        .withColumnRenamed(tmpname, group_name)
    
    return df
    
    
def buckt_by_mutitreatment(ite_names, df, cvr_names=None,N=10, group_threshold_mapping=None):
    """
    group_threshold_mapping:{'T':{'G':[left_threshold, right_threshold]}}
    """

    dfs = []
    threshold_stats = {}
    for idx, ite_name in enumerate(ite_names):
        print('bucktinization:{}'.format(ite_name))
        
        if group_threshold_mapping is not None:
            print("using group_threshold_mapping:{}".format(group_threshold_mapping[ite_name]))
            buck_df = apply_group_threshold(df, ite_name, group_threshold_mapping[ite_name])
        
        else:
            buck_df = bucketize_freq(df, models=[ite_name],N=N)
            bucketed_col = '{}_freq_bucket'.format(ite_name)
            
            stats = buck_df.groupBy(bucketed_col).agg(
                    *[func.min(func.col(ite_name)).alias('min_threshold'),
                      func.max(func.col(ite_name)).alias('max_threshold')
                     ]
            ).toPandas()
            
            __threshold_map = {}
            for value in stats.values:
                __threshold_map[int(value[0])] = [float(value[1]), float(value[2])]
            threshold_stats[ite_name] = __threshold_map
            
            buck_df = buck_df.drop(ite_name).withColumnRenamed(bucketed_col, ite_name)
            
            
        use_cols = ['oneid', ite_name,'label_dt'] if cvr_names is None else ['oneid', ite_name, cvr_names[idx]] 
        buck_df = buck_df.select(*use_cols)
        
        
        
        dfs.append(buck_df)
    
    base_df = dfs[0]
    base_sampels = base_df.count()
    
    print('base samples:{}'.format(base_sampels))
    
    for buck_df in dfs[1:]:
        base_df = base_df.join(buck_df, ['oneid','label_dt'], "left")
    
    base_df = base_df.withColumn("combination", 
                    func.concat_ws("-", *ite_names)
            )
    
    if group_threshold_mapping is None:
        group_threshold_mapping = threshold_stats
    
    return base_df, group_threshold_mapping

def permutation_groups(numgroups, numtreatments):
    total_groups = math.pow(numgroups, numtreatments)
    combinations = list(itertools.product(range(1,numgroups+1), repeat=numtreatments))
    
    combinations = ["-".join([str(i) for i in x]) for x in combinations]
    group_permutations_map = {}
    for code, item in enumerate(combinations):
        group_permutations_map[item] = code
    
    
    return group_permutations_map

def assign_groups(group_permutations_map, 
                  bucketed_df:DataFrame, 
                  combination_col="combination", 
                  assign_group_col='assign_group'):
    """
    @group_permutations_map:{combination:group_id}
    @bucketed_df:[oneid, treatment1_bucket_id,...,treatmentn_bucket_id, combination]
    """
    
    def __assign(row):
        return group_permutations_map[row]
        
    assign_udf = func.udf(lambda z: __assign(z), T.IntegerType())
    bucketed_df = bucketed_df.withColumn(assign_group_col, assign_udf(func.col(combination_col)))
    
    return bucketed_df

def padding_cost(cost_pdf, group_permutations_map, cac_col='delta_cac_uv'):
    group_names = cost_pdf['group_name'].unique()
    assign_groups = sorted(group_permutations_map.values())
    
    adjusted_pdf = []
    columns = cost_pdf.columns
    
    padding_cnt = 0
    for as_g in assign_groups:
        for g in group_names:
            __cac = 0
            subset = cost_pdf[(cost_pdf['assign_group']==as_g) & (cost_pdf['group_name']==g)]
            
            if g == 'control_group':
                __cac = 0

            elif subset.shape[0] !=0:
                if subset.shape[0]>1:
                    assert "group={}\tassign_group={} error ".format(g, as_g)
                __cac = subset[cac_col].values[0]
            else:
                __cac      = None
                padding_cnt += 1

            adjusted_pdf.append([as_g, g, __cac])
                
    adjusted_pdf = pd.DataFrame(adjusted_pdf, 
                                columns=['assign_group','group_name',cac_col]
                            )
    
    adjusted_pdf = adjusted_pdf.fillna(method='pad')
    __ratio = padding_cnt / len(assign_groups) * len(group_names)
    print("total {}({:.2f} %) group&treatment padded...".format(padding_cnt, __ratio))
    
    return adjusted_pdf

def build_package_matrix(groups, threatments, cac_record_pdf,
                        est_cvr_pdf,
                        cvr_names,
                        cac_col='delta_cac_uv',
                        divide_attr=False):
    est_costs = []
    est_ords = []
    
    for group in groups:
        for threatment, cvr_name  in zip(threatments,cvr_names):
            
            if threatment == 'control_group':
                stats_cost = 0
            else:
                stats_cost = cac_record_pdf[(cac_record_pdf['assign_group']==group) & \
                                         (cac_record_pdf['group_name']==threatment)][cac_col].values[0]
            
            est_ord = est_cvr_pdf[est_cvr_pdf['assign_group']==group][cvr_name].values[0]
            gsize   = est_cvr_pdf[est_cvr_pdf['assign_group']==group]['gsize'].values[0]
            
            
            if threatment != 'control_group':
                cost = stats_cost * est_ord
                
                if cost<0:
                    cost = max(abs(cost), 1000000)
                    
                if divide_attr:
                    cost /= 0.7
                
            else:
                cost = 0
                
            est_costs.append(int(cost))
            est_ords.append(int(est_ord))
    
    return est_costs, est_ords

def ortools_coupon(est_costs, est_ords, discounts_mapping, num_g,set_cac,max_sel_num=1):
    """
    est_costs: 1xGxT
    est_ords:  1xGxT
    """
    
    discount_size = len(discounts_mapping)
    n = len(est_costs)

    st_exp_cost = None
    st_exp_dis  = []
    obj_exp     = None
    
    optimization_problem_type = 'SCIP'
    solver = pywraplp.Solver.CreateSolver(optimization_problem_type)
    
    x = [solver.IntVar(0, 1, 'x'+str(i)) for i in range(n)]
    
    # 是否发放折扣的变量
    for g in range(num_g):
        exp = None
        for t in range(g*discount_size, (g+1)*discount_size):
            if exp is None:
                exp = x[t] * 1
            else:
                exp += x[t] * 1
        st_exp_dis.append(exp)
    
    # cost 表达式
    for i in range(n):
        if st_exp_cost is None:
            st_exp_cost  = x[i] * est_costs[i]
        else:
            st_exp_cost += x[i] * est_costs[i]
    
    #目标函数表达式
    for i in range(n):
        if obj_exp is None:
            obj_exp  = x[i]*est_ords[i]
        else:
            obj_exp += x[i]*est_ords[i]
    
    
    for exp in st_exp_dis:
        solver.Add(exp == max_sel_num)
    
    # cac约束：st_exp_cost/obj_exp <= set_cac
    # 总的消耗/预计转化人数 = cac <= 约束cac
    solver.Add(st_exp_cost <= set_cac*obj_exp)
    
    solver.Maximize(obj_exp)
    result_status = solver.Solve()
    
    res = []
    max_finish = 0
    if result_status == pywraplp.Solver.OPTIMAL:
        for i in range(len(x)):
            res.append(x[i].solution_value())
            
        max_finish = solver.Objective().Value()

    return (max_finish, res)

def or_convert(res, group_indexs, original_groups, para):
    para_revc = dict((v,k) for k,v in para.items())
    
    dis_size = len(para)
    loss_res = {}
    gsize = len(group_indexs)
    
    pdf = []
    for i in range(gsize):
        tag = 0
        for j in range(i*dis_size,(i+1)*dis_size):
            
            if res[j] == 1:
                index = j - i*dis_size
                loss_res[original_groups[group_indexs[i]]] = para_revc[index]
                pdf.append([original_groups[group_indexs[i]], para_revc[index]])
                tag = 1
                break
                
        if tag == 0:
            loss_res[original_groups[group_indexs[i]]] = '不发'
            pdf.append([original_groups[group_indexs[i]], '不发'])
            
    pdf = pd.DataFrame(pdf, columns=['assign_group','programming_treatment'])
    
    return loss_res, pdf


def random_tbname(prefix='tb'):
    tbname = "{}_{}".format(prefix, random.randint(1,10000))
    return tbname

def call_threatment_dist(spark, integer_programming_pdf_tbname, estimate_assign_group_ite_tbname):
    
    fmt = """
    select
        a.*
        ,programming_treatment
    from (
        select * from {estimate_assign_group_ite_tbname}
    )a left join (
        select /*+ BAOADCAST(b) */ 
            * 
        from {integer_programming_pdf_tbname}
    )b on a.assign_group = b.assign_group
    """.format(integer_programming_pdf_tbname=integer_programming_pdf_tbname,
              estimate_assign_group_ite_tbname=estimate_assign_group_ite_tbname
              )
    
    stats = spark.sql(fmt).groupBy("programming_treatment").count().toPandas()
    total = stats['count'].sum()
    stats['ratio'] = stats['count'] / total
    
    return stats
    

class Parameters(object):
    def __init__(self, hdfs_space_fmt, local_space_fmt, model_version, dset_name, 
                 bucket_method, prob_type, exe_dt,num_groups,
                 treatments     = ['5','8','10','55','58','control_group'],
                 prg_train_pred_tbname = 'dm.dm_ut_mkt_backflow_outcall_trainval_inference',
                 prg_val_pred_tbname   = 'dm.dm_ut_mkt_backflow_outcall_trainval_inference',
                 prg_inference_tbname = 'dm.dm_ut_mkt_backflow_outcall_manymodel_inference_di',
                 label_name='uv_label_3d'
                 
                ):
        self.hdfs_space_fmt  = hdfs_space_fmt
        self.local_space_fmt = local_space_fmt
        self.model_version  = model_version
        self.bucket_method  = bucket_method
        self.prob_type      = prob_type
        self.exe_dt         = exe_dt
        self.dset_name      = dset_name
        self.treatments     = treatments
        self.num_treatments = len(treatments)
        self.num_groups     = num_groups
        self.label_name     = label_name
        
        self.prg_train_pred_tbname = prg_train_pred_tbname
        self.prg_val_pred_tbname   = prg_val_pred_tbname
        self.prg_infe_pred_tbname  = prg_inference_tbname
        
        self.add_restriction = "and label_dt in {label_dts} and dset = '{dset_name}'"
        self.add_restriction_infe = "and label_dt in {label_dts} and model = '{model_version}'"
        
        self.prg_tr_dts            = None
        self.prg_val_dts           = None
        self.prg_infe_dts          = [exe_dt]
        
        self.bucket_method_tag = "{}.{}.{}".format(
                                            bucket_method, 
                                            self.num_groups, 
                                            self.num_treatments
                                    )
    
    def set_validation_dts(self, prg_tr_dts, prg_val_dts, stage='simulation', skip_check=False):
        if stage == 'simulation':
            if not skip_check:
                assert max(prg_tr_dts) < max(prg_val_dts)
                assert len(set(prg_tr_dts).intersection(set(prg_val_dts)))== 0
            self.prg_val_dts = prg_val_dts
        
        
        self.prg_tr_dts  = prg_tr_dts
        
        return self
    
    def get_fields(self, stage='stats'):
        ite_names       = ['{}_delta_cvr_{}'.format(self.model_version, t) for t in self.treatments if t != 'control_group']
        ite_stair_names = ['{}_delta_cvr_stair_{}'.format(self.model_version, t) for t in self.treatments if t != 'control_group']

        cvr_names             = ['{}_cvr_{}'.format(self.model_version, t) for t in self.treatments]
        delta_cvr_names       = ['{}_delta_cvr_{}'.format(self.model_version, t) for t in  self.treatments]
        delta_cvr_stair_names = ['{}_delta_cvr_stair_{}'.format(self.model_version, t) for t in  self.treatments]
        
        if stage == 'simulation':
            return delta_cvr_names
        
        if stage == 'stats':
            if self.prob_type == 'TdivC':
                return ite_names
            elif self.prob_type == 'TdivT':
                return ite_stair_names
            elif self.prob_type == 'default':
                return ite_names
            
    
    def check_parameters(self):
        assert self.bucket_method in ['eqfreq','cluster']
        assert self.prob_type     in ['TdivC','TdivT','default']
        assert self.label_name    in ['uv_label_3d']
        
        return self
    
    @staticmethod
    def print_parameters(obj):
        for var_name, var_value in vars(obj).items():
            print(f"{var_name}: {var_value}\n")
    
    def get_hdfs_path(self, dts, tag):
        #assert tag in ['prg_train_deltas','prg_val_deltas', 'vector_model','cluster_model',
        #              'prg_train_buck','prg_val_buck', 'prg_infe_buck'], "tag={} unaccepted..".format(tag)
        
        hdfs_path = self.hdfs_space_fmt.format(
            model_version=self.model_version,
            dataspace=self.dset_name,
            bucket_method=self.bucket_method_tag,
            prob_type=self.prob_type,
            sdt=dts[0].replace('-',''),
            edt=dts[-1].replace('-',''),
            dt=self.exe_dt,
            tag=tag
        )
        
        print("get path for tag={}->{}".format(tag, hdfs_path))
        
        return hdfs_path
    
    def to_excel(self, pdf, tag, dts, only_path=False, **kwargs):
        local_path = self.local_space_fmt.format(
            model_version=self.model_version,
            dataspace=self.dset_name,
            bucket_method=self.bucket_method_tag,
            prob_type=self.prob_type,
            sdt=dts[0].replace('-',''),
            edt=dts[-1].replace('-',''),
            dt=self.exe_dt,
            tag=tag
        )
        
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            
        local_path = os.path.join(local_path, '{}.xlsx'.format(tag))
        
        print("get local path for tag={}->{}".format(tag, local_path))
        
        if only_path:
            return local_path
        
        
        pdf.to_excel(local_path, index=False)
    
    def read_excel(self, tag, dts):
        return pd.read_excel(self.to_excel(None, tag, dts, True))
        
class StageChecker(object):
    def __init__(self):
        pass
    
    def check_dt_valid(self, adts, bdts):
        assert len(adts) == len(bdts)
        assert len(set(adts).difference(set(bdts))) == 0
    
    def check_treatment_dist(self, df, agg_field='fake_treatment',
                            treatment_ranges=['5','8','10','55','58','control_group']):
        stats = df.groupBy(agg_field).count().toPandas()
        print(stats)

        assert len(set(stats[agg_field].tolist()).difference(set(treatment_ranges))) == 0


def register_buck_tb(hdfs,spark, param_obj:Parameters, dts, tag, group_permutations_map, bucked_df=None, 
                     overwrite=False,partition_size=100, 
                     assign_group=True):
    
    hdfs_wpath = param_obj.get_hdfs_path(dts, tag)
    if overwrite or not hdfs.exists(hdfs_wpath):
        if param_obj.bucket_method in ['eqfreq'] and assign_group:
            print("using bucket mehod:{}".format(param_obj.bucket_method))
            
            bucked_df = assign_groups(group_permutations_map=group_permutations_map, 
                                      bucketed_df=bucked_df)
        
        print('write bucketed df to:{}'.format(hdfs_wpath))
        bucked_df.repartition(partition_size).write \
                 .mode("overwrite").parquet(hdfs_wpath)
    
    bucked_df = spark.read.parquet(hdfs_wpath)
    tbname    = random_tbname(tag)
    bucked_df.registerTempTable(tbname)
    
    return tbname

def get_buck_dist(spark, tbname, tbtag):
    buck_dist_stats = spark.sql("""
    select 
        assign_group
        ,count(*) as samples
    from {tbname}
    group by assign_group
    """.format(tbname=tbname)).toPandas()
    
    buck_dist_stats['tbtag'] = tbtag
    return buck_dist_stats.sort_values(by='samples')

def transform_stast_cvr(pdf, model_version):
    """
    @pdf: 存储每个(G,T)的 后验delta_cvr
    
    计算后验下，每个桶每个T的delta转化=> F(G,T) = DeltaCVR(G, T) * |G| 
    """
    __assign_groups = pdf['assign_group'].unique()
    __treatments   = pdf['group_name'].unique()
    
    records = []
    for assign_group in __assign_groups:
        subset = pdf[pdf['assign_group']==assign_group]
        __records = []
        
        for treatment in __treatments:
            _subset = subset[subset['group_name']==treatment]
            _delta_cac = 0
            _delta_cvr = 0
            if _subset.shape[0] > 0:
                _delta_cac = _subset['delta_cac_uv'].values[0]
                _delta_cvr = _subset['delta_cvr_uv'].values[0]
            
            if treatment == 'control_group':
                _delta_cac = 0
                _delta_cvr = 0
            
            if _delta_cvr is None or str(_delta_cvr)=='nan':
                _delta_cvr = 0
            
            __records.append(_delta_cvr)
        __records = [assign_group] + __records + [subset['ingroup_cnt'].sum()]
        records.append(__records)
    
    __delta_cvr_names = ['{}_delta_cvr_{}'.format(model_version, treatment) for treatment in __treatments]
    columns = ['assign_group'] \
            + __delta_cvr_names \
            + ['gsize']
    
    records = pd.DataFrame(records, columns=columns)
    for _delta_cvr_name in __delta_cvr_names:
        records[_delta_cvr_name] = records[_delta_cvr_name] * records['gsize']
    
    return records



class BucketByClustering():
    def __init__(self, parameters, hdfs):
        
        self.parameters      = parameters
        self.hdfs     = hdfs
        self.vector_model = None
        self.cluster_model = None
        self.fnames = None
        
        self.cluser_model_wpath = parameters.get_hdfs_path(tag='cluster_model', 
                                                        dts=parameters.prg_tr_dts)
        
        self.vector_wpath = parameters.get_hdfs_path(tag='vector_model', 
                                                     dts=parameters.prg_tr_dts)
        
    def fit(self, df, fnames, overwrite=False):
        if not self.hdfs.exists(self.cluser_model_wpath) or overwrite:
            print('fit model, using fnames:{}'.format(fnames))
            assembler = VectorAssembler(inputCols=fnames, outputCol="features")

            _df = assembler.transform(df)
            kmeans = KMeans().setK(self.parameters.num_groups).setSeed(1)
            model = kmeans.fit(_df)

            self.vector_model = assembler
            self.cluster_model = model
            
            self.save_model()
        
        self.load_model()
        
        return self
    
    def transform(self, df):
        df = self.cluster_model.transform(self.vector_model.transform(df))
        
        return df
    
    def save_model(self):
        print("save vector model to:{}".format(self.vector_wpath))
        print("save cluster model to:{}".format(self.cluser_model_wpath))
        
        self.cluster_model.save(self.cluser_model_wpath)
        self.vector_model.save(self.vector_wpath)
        
        print("model saved...")
        
        return self
    
    def load_model(self):
        
        
        print("read vector model from:{}".format(self.vector_wpath))
        print("read cluster model from:{}".format(self.cluser_model_wpath))
        
        self.cluster_model = KMeansModel().load(self.cluser_model_wpath)
        self.vector_model  = VectorAssembler().load(self.vector_wpath)
        
        print("model loaded...")
        
        return self


