#!/usr/bin/env python3
"""One Spark client stage; every output is new and checked before completion."""
import datetime as dt
import importlib.util
import math
import sys
from pathlib import Path
from dataset import read, write, require, date, TOKENS


def unique(df, keys, label):
    from pyspark.sql import functions as F
    require(df.where(' OR '.join(k+' IS NULL' for k in keys)).limit(1).count()==0,label+' 关联键有 NULL')
    require(df.groupBy(*keys).count().where(F.col('count')>1).limit(1).count()==0,label+' 关联键不唯一，可能放大样本')


def validate_record(row, feature_count):
    try:
        label=float(row['label']); require(math.isfinite(label),'label 非有限数')
        tokens=row['token'].split(','); require(len(tokens)==len(TOKENS),'token 数量错误')
        features=row['feature'].split(','); require(len(features)==feature_count,'feature slot 数量错误')
        for item in features:
            index,value=item.split(':')
            require(int(index)>=1 and math.isfinite(float(value)),'非法 feature index/value')
        require(tokens[TOKENS.index('fake_treatment_id')] in ('0','1','2','3','4'),'treatment 非法')
        require(math.isfinite(float(tokens[TOKENS.index('cost_7d')])),'cost 非有限数')
        return True
    except (ValueError,TypeError,AttributeError):
        return False


def main():
    run=Path(sys.argv[1]); target=sys.argv[2]; stage=sys.argv[3]
    action,split=stage.split('_',1)
    r=read(run/'resolved_request.json'); env=read(run/'execution.json')
    from pyspark.sql import SparkSession, functions as F
    spark=SparkSession.builder.appName(r['dataset_id']+'_'+stage).enableHiveSupport().getOrCreate()
    report={'status':'running','stage':stage,'spark_version':spark.version,'application_id':spark.sparkContext.applicationId}
    try:
        if action=='snapshot':
            effective=r['date_report'][split]['effective_dates']
            keys=['flowtree_group_name','group_name_eve','dt','pid']
            experiment="new_exp = '迭代流量' AND (group_name_eve IN ('留存6张5元','留存6张3元','8元留存5张3','10元留存5张3') OR (flowtree_group_name = 'DS空白' AND group_name_eve = 'control_group'))"
            sources=[]
            for table in ('dm.ut_alg_flowback_msg_outcall_feature_train_di','dm.ut_alg_flowback_msg_outcall_label_v2_di'):
                frame=spark.table(table).where(F.col('dt').isin(effective)).where(experiment)
                unique(frame,keys,table)
                found={str(row['dt']) for row in frame.select('dt').distinct().collect()}
                require(found==set(effective),table+' 缺少有效日期: '+str(set(effective)-found))
                sources.append(frame)
            unmatched=sources[0].join(sources[1].select(*keys),keys,'left_anti').count()
            require(unmatched==0,'主样本缺少匹配 label: '+str(unmatched))
            # External joins have one record per user/date. Their physical snapshot
            # dates are shifted from sample dates, not from output-directory dates.
            for b in r['blocks']:
                if not b.get('join_template'): continue
                snapshot_days=[(date(d)+dt.timedelta(days=b['snapshot_offset_days'])).isoformat() for d in effective]
                extra=spark.table(b['source_table']).where(F.col(b['source_date']).isin(snapshot_days))
                unique(extra,[b['source_key'],b['source_date']],b['name'])
                found={str(row[b['source_date']]) for row in extra.select(b['source_date']).distinct().collect()}
                require(found==set(snapshot_days),b['name']+' 缺少快照日期')
            frame=spark.sql((run/'sql'/(split+'.sql')).read_text()).cache()
            count=frame.count(); require(count>0,split+' 样本为空')
            require(set(frame.columns)==set(r['fields']+TOKENS+['label']),'样本字段与声明不一致')
            unique(frame,['pid','group_name_eve','label_dt'],'最终样本候选唯一键')
            observed={str(row['label_dt']) for row in frame.select('label_dt').distinct().collect()}
            require(observed==set(effective),'最终样本日期覆盖不符合请求')
            bad=frame.where(F.col('label').isNull() | F.isnan('label') | (F.abs(F.col('label'))==float('inf')) |
                            F.col('cost_7d').isNull() | F.isnan('cost_7d') | (F.abs(F.col('cost_7d'))==float('inf')) |
                            F.col('fake_treatment_id').isNull() | (~F.col('fake_treatment_id').isin(0,1,2,3,4)))
            require(bad.limit(1).count()==0,'label/cost/treatment 有非法值')
            nulls=frame.agg(*[F.sum(F.col(c).isNull().cast('long')).alias(c) for c in r['fields']]).first().asDict()
            report.update(row_count=count,source_feature_rows=sources[0].count(),missing_labels=unmatched,
                          date_counts=[row.asDict() for row in frame.groupBy('label_dt').count().collect()],
                          treatment_counts=[row.asDict() for row in frame.groupBy('fake_treatment_id').count().collect()],
                          feature_null_counts=nulls)
            require(not any(n==count for n in nulls.values()),'存在全空特征；请检查特征来源或 JOIN')
            output=target+'/samples/'+split
            frame.write.mode('errorifexists').parquet(output)
            require(spark.read.parquet(output).count()==count,'样本快照读回行数不一致')
        elif action=='extract':
            spec=importlib.util.spec_from_file_location('legacy_extract',run/'runtime/run_offline_tans2tfr.py')
            legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
            values=dict(l.split('=',1) for l in (run/'conf/parser.conf').read_text().splitlines() if '=' in l)
            frame=spark.read.parquet(target+'/samples/'+split)
            expected=read(run/'reports'/('snapshot_'+split+'.json'))['row_count']
            require(frame.count()==expected,'物化样本行数发生变化')
            fn=legacy.get_map_func(spark.sparkContext.broadcast(values['fea_cols'].split(',')),spark.sparkContext.broadcast('parser.conf'))
            converted=frame.repartition(int(env.get('partitions',100))).rdd.mapPartitions(fn).toDF('label string, feature string, token string').cache()
            count=converted.count(); require(count==expected,'特征转换丢失或新增样本')
            feature_count=len(r['fields'])
            # Validate inside the driver over a bounded sample; full readback count below.
            for row in converted.limit(1000).collect(): require(validate_record(row,feature_count),'TFRecord 样本格式校验失败')
            output=target+'/tfrecord/'+split
            fmt='tfrecords' if spark.version.startswith('2') else 'tfrecord'
            converted.write.mode('errorifexists').format(fmt).save(output)
            back=spark.read.format(fmt).load(output)
            require(back.count()==count,'TFRecord 读回行数不一致')
            # Connector may expose scalar strings as singleton arrays on readback.
            for row in back.limit(1000).collect():
                value={k:(v[0] if isinstance(v,(list,tuple)) and len(v)==1 else v) for k,v in row.asDict().items()}
                value={k:(v.decode() if isinstance(v,(bytes,bytearray)) else v) for k,v in value.items()}
                require(validate_record(value,feature_count),'TFRecord 读回格式校验失败')
            report.update(row_count=count,format_checked_rows=min(count,1000),fit_split='train')
        else: raise ValueError('未知 stage')
        report.update(status='completed',output=output)
        write(run/'reports'/(stage+'.json'),report)
    finally:
        spark.stop()

if __name__=='__main__': main()
