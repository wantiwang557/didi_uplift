#!/usr/bin/env python3
"""Build portable, immutable WWT data tasks. Standard library only; JSON input."""
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'valid', 'test')
TOKENS = 'fin_order_wxmini_28d,group_name_eve,pid,oneid,send_cnt,send_success_cnt,send_cost,remind_cost,coupon_cost,dau_cnt,exp_label,cost_7d,fake_treatment_id,flag,label_dt'.split(',')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def date(value):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), '日期格式必须是 YYYY-MM-DD')
    return dt.date.fromisoformat(value)


def days(start, end):
    a, b = date(start), date(end)
    require(a <= b, '起始日期晚于结束日期')
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b-a).days+1)]


def render(template, values):
    for key, value in values.items():
        template = template.replace('{{'+key+'}}', str(value))
    require('{{' not in template or '}}' not in template, 'SQL 模板有未解析的变量')
    return template


def resolve(request, catalog, policy):
    allowed = {'dataset_id','profile','feature_blocks','splits','additional_exclusions','seed','sample_fraction','bucket_num'}
    require(not set(request)-allowed, '未知请求字段: '+str(set(request)-allowed))
    require(re.fullmatch(r'[a-z][a-z0-9_]{0,62}', request['dataset_id']), 'dataset_id 只能使用小写字母、数字、下划线')
    require(request['profile'] == catalog['profile'], '不支持此 profile')
    aliases = {}
    for block in catalog['blocks']:
        for name in [block['id'],block['name']] + block.get('aliases',[]):
            require(name not in aliases, '特征块名称/别名冲突: '+name)
            aliases[name] = block
    selected=[]
    for name in request['feature_blocks']:
        require(name in aliases, '未知特征块: '+name)
        block=aliases[name]
        require(block not in selected, '重复特征块: '+name)
        selected.append(block)
    require(selected, '至少选择一个特征块')
    fields=[f['name'] for b in selected for f in b['fields']]
    require(len(fields)==len(set(fields)), '不同特征块有重复输出字段')
    require(not set(fields).intersection(TOKENS+['label']), '特征与 label/token 重名')
    require(all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',f) for f in fields), '非法字段名')
    require(all(f['encoding']=='lisan' for b in selected for f in b['fields']), '第一版仅支持现有 lisan 分箱，请先扩展编码适配器')
    require(set(request['splits'])==set(SPLITS), '必须提供 train、valid、test')
    result=dict(request, blocks=selected, fields=fields, tokens=TOKENS, date_report={})
    previous=None
    for split in SPLITS:
        window=request['splits'][split]
        requested=days(window['start'],window['end'])
        require(requested[0]>='2026-03-02', '当前 v6 口径只支持 2026-03-02 及以后')
        require(previous is None or requested[0]>previous, 'train、valid、test 必须按时间顺序且不重叠')
        previous=requested[-1]
        exclusions={}
        for rule in policy['rules']:
            if rule['profile']!=request['profile'] or split not in rule['splits']:
                continue
            if rule['basis']=='sample_date':
                affected=days(rule['start'],rule['end'])
            elif rule['basis']=='feature_snapshot':
                matches=[b for b in selected if b['id']==rule['block_id']]
                if not matches:
                    continue
                offset=matches[0]['snapshot_offset_days']
                affected=[(date(d)-dt.timedelta(days=offset)).isoformat() for d in days(rule['start'],rule['end'])]
            else:
                raise ValueError('未知日期规则 basis')
            for day in affected:
                exclusions.setdefault(day,[]).append({'id':rule['id'],'reason':rule['reason'],'evidence':rule['evidence']})
        for item in request.get('additional_exclusions',[]):
            date(item['date']); require(item.get('reason'), '额外排除日期必须说明原因')
            exclusions.setdefault(item['date'],[]).append({'id':'user','reason':item['reason']})
        excluded={d:exclusions[d] for d in requested if d in exclusions}
        effective=[d for d in requested if d not in exclusions]
        require(effective, split+' 排除后没有可用日期')
        result['date_report'][split]={'requested':window,'excluded':excluded,'effective_dates':effective}
    result['seed']=request.get('seed',2026)
    result['sample_fraction']=request.get('sample_fraction',0.1)
    result['bucket_num']=request.get('bucket_num',10)
    require(type(result['seed']) is int, 'seed 必须为整数')
    require(0<float(result['sample_fraction'])<=1, 'sample_fraction 必须在 (0,1]')
    require(type(result['bucket_num']) is int and result['bucket_num']>=2, 'bucket_num 至少为 2')
    return result


def build(request_path, out, repo):
    out,repo=Path(out).resolve(),Path(repo).resolve()
    require(not out.exists(), '任务目录已存在；请使用新 dataset_id/目录，不覆盖历史任务')
    request=read(request_path)
    resolved=resolve(request,read(SKILL/'references/feature-catalog.json'),read(SKILL/'references/date-exclusions.json'))
    sqls={}
    for split in SPLITS:
        report=resolved['date_report'][split]; window=report['requested']
        predicate="dt BETWEEN '{}' AND '{}'".format(window['start'],window['end'])
        if report['excluded']:
            predicate+=" AND dt NOT IN ("+','.join("'"+d+"'" for d in report['excluded'])+")"
        joins=[]; features=[]
        for block in resolved['blocks']:
            features.append('    -- '+block['name'])
            features.extend('    ,'+f['expression'] for f in block['fields'])
            if block.get('join_template'):
                offset=dt.timedelta(days=block['snapshot_offset_days'])
                joins.append(render((SKILL/block['join_template']).read_text(),{
                    'source_start':date(window['start'])+offset,'source_end':date(window['end'])+offset,
                    'sample_start':window['start'],'sample_end':window['end']}))
        sqls[split]=render((SKILL/'assets/base.sql').read_text(),{'split':split,'features':'\n'.join(features),'date_predicate':predicate,'joins':'\n'.join(joins)})
    for name in ('feature_list_handler.py','run_offline_tans2tfr.py'):
        require((repo/'offline_bigdata/python'/name).is_file(), '仓库缺少 '+name)
    for folder in ('sql','conf','logs','reports','runtime','definition'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    write(out/'request.json',request); write(out/'resolved_request.json',resolved)
    for split,sql in sqls.items(): (out/'sql'/ (split+'.sql')).write_text(sql)
    shutil.copytree(SKILL/'references',out/'definition/references')
    shutil.copytree(SKILL/'assets',out/'definition/assets')
    for name in ('feature_list_handler.py','run_offline_tans2tfr.py'):
        shutil.copy2(repo/'offline_bigdata/python'/name,out/'runtime'/name)
    for name in ('dataset.py','cluster_worker.py','fit_config.py'):
        shutil.copy2(SKILL/'scripts'/name,out/'runtime'/name)
    shutil.copy2(repo/'offline_bigdata/bin/fea_extract',out/'runtime/fea_extract')
    hashes={str(p.relative_to(out)):digest(p) for folder in ('sql','runtime','definition') for p in (out/folder).rglob('*') if p.is_file()}
    hashes.update({name:digest(out/name) for name in ('request.json','resolved_request.json')})
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
    manifest={'schema_version':1,'dataset_id':request['dataset_id'],'status':'planned','git_head':revision.stdout.strip(),
              'artifact_sha256':hashes,'stages':{},'feature_count':len(resolved['fields']),'token_count':len(TOKENS),
              'limitations':['上游表可回刷；保存物化样本后才固定本次数据内容','同一 seed 不保证跨 Spark 版本或分区布局的分箱逐字一致；以保存的 conf 为准']}
    write(out/'manifest.json',manifest)
    return manifest


def verify(run):
    m=read(run/'manifest.json')
    for name,expected in m['artifact_sha256'].items():
        require((run/name).is_file() and digest(run/name)==expected, '任务文件发生变化: '+name)
    for name,expected in m.get('config_sha256',{}).items():
        require(digest(run/name)==expected,'train 配置发生变化: '+name)
    return m


def check_conf(run):
    r=read(run/'resolved_request.json'); path=run/'conf/feature.conf'
    rows=[l for l in path.read_text().splitlines() if l and not l.startswith('#')]
    deps=[re.search(r'(?:^|;)depend=([^;]+)',l).group(1) for l in rows]
    require(deps==r['fields'],'生成配置字段/顺序不匹配（常量或全空特征可能未被旧生成器写出），请检查 fit 日志')
    parser=run/'conf/parser.conf'
    values=dict(l.split('=',1) for l in parser.read_text().splitlines() if '=' in l and not l.startswith('#'))
    require(values.get('label')=='label' and values.get('add_token')==','.join(TOKENS),'label/token 不匹配')
    require(set(values['fea_cols'].split(','))==set(r['fields']+TOKENS+['label']), 'parser 字段集合不匹配')
    values['feature_list']='./feature.conf'
    parser.write_text(''.join(k+'='+v+'\n' for k,v in values.items()))
    return {str(p.relative_to(run)):digest(p) for p in (path,parser)}


def execute(run, env_path):
    run=Path(run).resolve(); m=verify(run); env=read(env_path); r=read(run/'resolved_request.json')
    required={'hdfs_root','queue','user_name','python','python_archives','python_executor','jars','label_available_through'}
    require(required<=set(env),'执行环境缺少: '+str(required-set(env)))
    require(date(env['label_available_through'])>=date(r['splits']['test']['end']), 'label 完整产出日期未覆盖 test 末日')
    require(re.fullmatch(r'/[A-Za-z0-9_./-]+',env['hdfs_root']) and '..' not in env['hdfs_root'].split('/'), '非法 HDFS 根路径')
    for cmd in ('spark-submit','hdfs',env['python']): require(shutil.which(cmd), '当前环境缺少命令: '+cmd)
    target=env['hdfs_root'].rstrip('/')+'/'+r['dataset_id']
    lock=run/'.runner.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError('执行锁存在；核实原任务/Yarn 状态后再人工恢复，不能重复提交')
    try:
        if (run/'execution.json').exists():
            require(read(run/'execution.json')==env,'不能为同一任务更换执行环境')
        else: write(run/'execution.json',env)
        m['hdfs_dataset_root']=target
        stages=['snapshot_'+s for s in SPLITS]+['fit']+['extract_'+s for s in SPLITS]
        for stage in stages:
            old=m['stages'].get(stage,{})
            if old.get('status')=='completed':
                report=run/'reports'/(stage+'.json')
                require(report.exists() and digest(report)==old['report_sha256'],'已完成阶段报告丢失/变化')
                if stage!='fit':
                    output=read(report)['output']
                    subprocess.run(['hdfs','dfs','-test','-e',output+'/_SUCCESS'],check=True)
                continue
            require(not old, '阶段 '+stage+' 已失败/状态不明；检查日志与 Yarn，使用新任务 ID 重试，避免覆盖残留数据')
            if stage=='fit':
                cmd=[env['python'],str(run/'runtime/fit_config.py'),str(run),target]
            else:
                cmd=['spark-submit','--master','yarn','--deploy-mode','client','--queue',env['queue'],
                     '--name',r['dataset_id']+'_'+stage,'--archives',env['python_archives'],'--jars',env['jars'],
                     '--conf','spark.executorEnv.PYSPARK_PYTHON='+env['python_executor']]
                if stage.startswith('extract'):
                    cmd+=['--files',','.join(str(run/p) for p in ('runtime/fea_extract','conf/parser.conf','conf/feature.conf'))]
                cmd += [str(run/'runtime/cluster_worker.py'),str(run),target,stage]
            m['status']='running'; m['stages'][stage]={'status':'running','command':cmd}
            write(run/'manifest.json',m)
            log=run/'logs'/(stage+'.log')
            try:
                with log.open('w') as f: subprocess.run(cmd,cwd=run,stdout=f,stderr=subprocess.STDOUT,check=True)
                if stage=='fit':
                    m['config_sha256']=check_conf(run)
                    write(run/'reports/fit.json',{'status':'completed','fit_split':'train','config_sha256':m['config_sha256']})
                report=run/'reports'/(stage+'.json')
                require(read(report)['status']=='completed','阶段没有成功验收报告')
                m['stages'][stage].update(status='completed',report_sha256=digest(report))
            except BaseException as exc:
                m['status']='failed'; m['stages'][stage].update(status='failed',error=str(exc)); write(run/'manifest.json',m)
                raise
            finally:
                if log.exists():
                    m['stages'][stage]['yarn_applications']=sorted(set(re.findall(r'application_\d+_\d+',log.read_text(errors='replace'))))
                    write(run/'manifest.json',m)
        m['status']='validated'; m['outputs']={s:target+'/tfrecord/'+s for s in SPLITS}; write(run/'manifest.json',m)
    finally:
        lock.rmdir()
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('build'); p.add_argument('--request',required=True); p.add_argument('--out',required=True); p.add_argument('--repo',required=True)
    p=sub.add_parser('execute'); p.add_argument('--run-dir',required=True); p.add_argument('--environment',required=True)
    p=sub.add_parser('verify'); p.add_argument('--run-dir',required=True)
    sub.add_parser('catalog')
    a=parser.parse_args()
    if a.action=='build': result=build(a.request,a.out,a.repo)
    elif a.action=='execute': result=execute(a.run_dir,a.environment)
    elif a.action=='verify': result=verify(Path(a.run_dir).resolve())
    else: result=[{'id':b['id'],'name':b['name'],'fields':len(b['fields']),'source':b['source_table']} for b in read(SKILL/'references/feature-catalog.json')['blocks']]
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
