#!/usr/bin/env python3
"""Fit legacy feature encodings on the task's frozen TRAIN snapshot only."""
import importlib.util
import sys
from pathlib import Path
from dataset import read, require


def main():
    run=Path(sys.argv[1]); target=sys.argv[2]
    r=read(run/'resolved_request.json'); env=read(run/'execution.json')
    spec=importlib.util.spec_from_file_location('legacy_handler',run/'runtime/feature_list_handler.py')
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    from pyspark.sql import SparkSession, functions as F
    spark=(SparkSession.builder.master('yarn').appName(r['dataset_id']+'_fit')
           .config('spark.yarn.queue',env['queue'])
           .config('spark.yarn.dist.archives',env['python_archives'])
           .config('spark.executorEnv.PYSPARK_PYTHON',env['python_executor'])
           .enableHiveSupport().getOrCreate())
    # Override only the data-loading adapter; preserve legacy binning and conf writer.
    # The original loader writes to a global shared cache and samples without a seed.
    class TaskFeatureHandler(legacy.FeatureHandler):
        def _get_orig_df(self, sql_file, sql_param, is_sample):
            frame=self.spark.read.parquet(target+'/samples/train')
            # Stable hash sampling of the logical sample key, independent of partitioning.
            threshold=int(float(is_sample)*1000000)
            sample=frame.where(F.pmod(F.xxhash64('pid','group_name_eve','label_dt',F.lit(r['seed'])),F.lit(1000000)) < threshold)
            require(sample.limit(1).count()>0,'train 采样为空；增加采样比例并创建新任务')
            cache=target+'/fit_sample'
            sample.write.mode('errorifexists').parquet(cache)
            return self.spark.read.parquet(cache)
    try:
        TaskFeatureHandler(spark=spark,sql_file='',sql_param='',is_sample=r['sample_fraction'],
            feature_conf_file=str(run/'conf/feature.conf'),parser_conf_file=str(run/'conf/parser.conf'),
            onehot_fea_list='',enum_fea_list='',lisan_fea_list=','.join(r['fields']),days_fea_list='',fix_ts_col='',
            minmax_fea_list='',direct_fea_list='',bucket_type='其他',bucket_num=r['bucket_num'],label_col='label',
            token_col=','.join(r['tokens']),float_calculate_default=0).run()
    finally:
        spark.stop()

if __name__=='__main__': main()
