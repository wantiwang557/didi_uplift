import copy
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import dataset as d
from cluster_worker import validate_record

ROOT=Path(__file__).resolve().parents[3]

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)
        self.req=d.read(d.SKILL/'assets/examples/v6-request.json')
        self.cat=d.read(d.SKILL/'references/feature-catalog.json')
        self.pol=d.read(d.SKILL/'references/date-exclusions.json')

    def build(self,req=None):
        p=self.folder/'request.json'; d.write(p,req or self.req)
        run=self.folder/'run'; d.build(p,run,ROOT); return run

    def test_default_matches_legacy_feature_order(self):
        r=d.resolve(self.req,self.cat,self.pol)
        script=(ROOT/'offline_bigdata/script/feature_confs/v6/gen_feature_conf_v6.sh').read_text()
        expected=re.search(r'lisan_fea="([^"]+)"',script).group(1).split(',')
        self.assertEqual(r['fields'],expected)
        self.assertEqual(len(r['tokens']),15)
        self.assertNotIn('2026-08-02',r['date_report']['train']['effective_dates'])

    def test_unknown_duplicate_overlap_empty_rejected(self):
        for mutate in (
            lambda r:r.update(feature_blocks=['不存在']),
            lambda r:r.update(feature_blocks=['sirius','sirius画像特征']),
            lambda r:r['splits']['valid'].update(start='2026-08-21'),
            lambda r:r['splits']['train'].update(start='2026-08-02',end='2026-08-02'),
            lambda r:r.update(typo=True),
        ):
            req=copy.deepcopy(self.req); mutate(req)
            with self.assertRaises(ValueError): d.resolve(req,self.cat,self.pol)

    def test_snapshot_exclusion_shift_and_scope(self):
        rule={'id':'snapshot_bad','profile':'euen_v6','splits':['test'],'basis':'feature_snapshot','block_id':'xtlx_order',
              'start':'2026-08-28','end':'2026-08-28','reason':'test','evidence':'fixture'}
        self.pol['rules'].append(rule)
        r=d.resolve(self.req,self.cat,self.pol)
        self.assertIn('2026-08-29',r['date_report']['test']['effective_dates'])
        self.req['feature_blocks'].append('xtlx_order')
        r=d.resolve(self.req,self.cat,self.pol)
        self.assertNotIn('2026-08-29',r['date_report']['test']['effective_dates'])
        self.assertIn('2026-08-28',r['date_report']['test']['effective_dates'])

    def test_artifact_tamper_and_existing_run_rejected(self):
        run=self.build(); d.verify(run)
        with self.assertRaises(ValueError): self.build()
        (run/'sql/test.sql').write_text('SELECT 1')
        with self.assertRaises(ValueError): d.verify(run)

    def test_config_contract(self):
        run=self.build(); r=d.read(run/'resolved_request.json')
        (run/'conf/feature.conf').write_text('\n'.join('name=i_'+f+';class=S_slice_binning;depend='+f+';args=1' for f in r['fields']))
        (run/'conf/parser.conf').write_text('fea_cols='+','.join(r['fields']+d.TOKENS+['label'])+'\nfeature_list=/tmp/old.conf\nlabel=label\nadd_token='+','.join(d.TOKENS)+'\ndebug=0\n')
        d.check_conf(run)
        self.assertIn('feature_list=./feature.conf',(run/'conf/parser.conf').read_text())
        (run/'conf/feature.conf').write_text('name=i;depend=wrong;args=1')
        with self.assertRaises(ValueError): d.check_conf(run)

    def test_generated_queries_execute_with_optional_joins(self):
        self.req['splits']={'train':{'start':'2026-08-01','end':'2026-08-03'},'valid':{'start':'2026-08-04','end':'2026-08-04'},'test':{'start':'2026-08-05','end':'2026-08-05'}}
        self.req['feature_blocks']=[b['name'] for b in self.cat['blocks']]
        run=self.build()
        db=sqlite3.connect(':memory:'); self.addCleanup(db.close)
        db.execute("ATTACH DATABASE ':memory:' AS dm")
        db.create_function('nvl',2,lambda x,y:y if x is None else x)
        db.create_function('if',3,lambda x,y,z:y if x else z)
        db.create_function('date_add',2,lambda x,n:(d.date(x)+d.dt.timedelta(days=n)).isoformat())
        base_fields=[f['name'] for b in self.cat['blocks'][:5] for f in b['fields']]
        cols=list(dict.fromkeys(base_fields+['fin_order_wxmini_28d','group_name_eve','pid','oneid','send_cnt','send_success_cnt','send_cost','remind_cost','coupon_cost','dau_cnt','dt','new_exp','flowtree_group_name']))
        db.execute('CREATE TABLE dm.ut_alg_flowback_msg_outcall_feature_train_di ('+','.join(c+' TEXT' for c in cols)+')')
        label_cols=['flowtree_group_name','group_name_eve','dt','pid','finish_cnt_7d','send_success_cnt','outcall_success_cnt','send_cost','send_remind_cost','outcall_cost','remind_cost','msg_coupon_cost_7d','outcall_coupon_cost_7d','outcall_coupon2_cost_7d','new_exp']
        db.execute('CREATE TABLE dm.ut_alg_flowback_msg_outcall_label_v2_di ('+','.join(c+' TEXT' for c in label_cols)+')')
        for block in self.cat['blocks'][5:]:
            extra_cols=list(dict.fromkeys([f['name'] for f in block['fields']]+[block['source_key'],'dt']))
            db.execute('CREATE TABLE '+block['source_table']+' ('+','.join(c+' TEXT' for c in extra_cols)+')')
        for day in d.days('2026-08-01','2026-08-05'):
            values={'dt':day,'pid':'123','oneid':'456','group_name_eve':'control_group','flowtree_group_name':'DS空白','new_exp':'迭代流量'}
            for table,columns in [('dm.ut_alg_flowback_msg_outcall_feature_train_di',cols),('dm.ut_alg_flowback_msg_outcall_label_v2_di',label_cols)]:
                db.execute('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in columns)+')',[values.get(c,'1') for c in columns])
            for block in self.cat['blocks'][5:]:
                columns=list(dict.fromkeys([f['name'] for f in block['fields']]+[block['source_key'],'dt']))
                values2={block['source_key']:'123','dt':(d.date(day)-d.dt.timedelta(days=1)).isoformat()}
                db.execute('INSERT INTO '+block['source_table']+' VALUES ('+','.join('?' for _ in columns)+')',[values2.get(c,'7') for c in columns])
        for split in d.SPLITS:
            cursor=db.execute((run/'sql'/(split+'.sql')).read_text()); names=[x[0] for x in cursor.description]; rows=cursor.fetchall()
            self.assertEqual(len(rows),2 if split=='train' else 1)
            self.assertEqual(len(names),609+16)
            for row in rows:
                value=dict(zip(names,row)); self.assertEqual(value['flag'],split)
                self.assertEqual(value['call_cnt'],'7'); self.assertEqual(value['gender'],'7')
                self.assertNotEqual(value['label_dt'],'2026-08-02')
        # Fixture executes SQL syntax and joins; real Hive/Spark types need cluster verification.

    def fake_cluster(self, run, fail_stage=None):
        calls=[]
        def invoke(cmd, **kwargs):
            if cmd[0]=='hdfs': return d.subprocess.CompletedProcess(cmd,0)
            stage='fit' if 'fit_config.py' in cmd[1] else cmd[-1]
            calls.append(stage)
            if stage==fail_stage: raise d.subprocess.CalledProcessError(1,cmd)
            if stage=='fit':
                r=d.read(run/'resolved_request.json')
                (run/'conf/feature.conf').write_text('\n'.join('name=i_'+f+';class=S_slice_binning;depend='+f+';args=1' for f in r['fields']))
                (run/'conf/parser.conf').write_text('fea_cols='+','.join(r['fields']+d.TOKENS+['label'])+'\nlabel=label\nadd_token='+','.join(d.TOKENS)+'\nfeature_list=old\n')
            else:
                d.write(run/'reports'/(stage+'.json'),{'status':'completed','output':'/fixture/'+stage})
            return d.subprocess.CompletedProcess(cmd,0)
        return calls,invoke

    def test_runner_sequence_and_successful_resume(self):
        run=self.build(); env=d.read(d.SKILL/'assets/examples/cluster-environment.json')
        env['label_available_through']='2026-09-10'; path=self.folder/'env.json'; d.write(path,env)
        calls,invoke=self.fake_cluster(run)
        with mock.patch.object(d.shutil,'which',return_value='/fixture'), mock.patch.object(d.subprocess,'run',side_effect=invoke):
            result=d.execute(run,path)
            self.assertEqual(result['status'],'validated')
            self.assertEqual(calls,['snapshot_train','snapshot_valid','snapshot_test','fit','extract_train','extract_valid','extract_test'])
            d.execute(run,path)
            self.assertEqual(len(calls),7)
        self.assertFalse((run/'.runner.lock').exists())

    def test_runner_stops_after_failure(self):
        run=self.build(); env=d.read(d.SKILL/'assets/examples/cluster-environment.json')
        env['label_available_through']='2026-09-10'; path=self.folder/'env.json'; d.write(path,env)
        calls,invoke=self.fake_cluster(run,'snapshot_valid')
        with mock.patch.object(d.shutil,'which',return_value='/fixture'), mock.patch.object(d.subprocess,'run',side_effect=invoke):
            with self.assertRaises(d.subprocess.CalledProcessError): d.execute(run,path)
            self.assertEqual(calls,['snapshot_train','snapshot_valid'])
            with self.assertRaises(ValueError): d.execute(run,path)
            self.assertEqual(len(calls),2)
        self.assertEqual(d.read(run/'manifest.json')['status'],'failed')

    def test_record_validation(self):
        token=['1']*15; token[12]='0'
        row={'label':'1','feature':'1:1,2:1','token':','.join(token)}
        self.assertTrue(validate_record(row,2))
        self.assertFalse(validate_record(row,3))
        row['feature']='0:1,2:1'; self.assertFalse(validate_record(row,2))
        row['label']='nan'; self.assertFalse(validate_record(row,2))

if __name__=='__main__': unittest.main()
