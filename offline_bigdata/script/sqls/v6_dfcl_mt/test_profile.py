"""Behavior fixtures for SQL, including full v6 projection and real join shape.

SQLite date/UDF emulation is not a Spark/Hive integration test.
"""
import datetime as dt
import json
import re
import sqlite3
import unittest
from pathlib import Path

import build_profile as profile

HERE = Path(__file__).resolve().parent


def date_add(value, days):
    return None if value is None else (dt.date.fromisoformat(value) + dt.timedelta(days=int(days))).isoformat()


def datediff(a, b):
    return None if a is None or b is None else (dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days


def sqlite_sql(sql):
    return sql.replace(' as string)', ' as text)')


class SampleProfileTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.execute("attach database ':memory:' as dm")
        self.db.create_function('to_date', 1, lambda x: x)
        self.db.create_function('date_add', 2, date_add)
        self.db.create_function('datediff', 2, datediff)
        self.db.create_function('concat_ws', -1, lambda sep, *xs: sep.join(str(x) for x in xs if x is not None))
        self.db.create_function('nvl', 2, lambda a, b: b if a is None else a)
        self.db.create_function('if', 3, lambda p, a, b: a if p else b)
        self.db.execute('''create table dm.ut_alg_flowback_msg_outcall_label_v2_di (
            dt text, pid text, flowtree_group_name text, group_name_eve text,
            new_exp text, finish_cnt_7d real, send_success_cnt integer, outcall_success_cnt integer,
            send_cost real, send_remind_cost real, outcall_cost real, remind_cost real,
            msg_coupon_cost_7d real, outcall_coupon_cost_7d real, outcall_coupon2_cost_7d real
        )''')
        catalog = json.loads((profile.REPO / 'skills/wwt-prepare-training-data/references/feature-catalog.json').read_text())
        self.features = [f['name'] for b in catalog['blocks'][:5] for f in b['fields']]
        self.feature_columns = list(dict.fromkeys(self.features + [
            'fin_order_wxmini_28d', 'group_name_eve', 'pid', 'oneid', 'send_cnt', 'send_success_cnt',
            'send_cost', 'remind_cost', 'coupon_cost', 'dau_cnt', 'dt', 'new_exp', 'flowtree_group_name']))
        self.db.execute('create table dm.ut_alg_flowback_msg_outcall_feature_train_di (' +
                        ','.join(c + ' TEXT' for c in self.feature_columns) + ')')

    def add(self, pid, day, label=0, sms=0, call=0, cost=0, control=False, group=None,
            feature=True, feature_value='1'):
        flow = 'DS空白' if control else '实验'
        group = group or ('control_group' if control else '留存6张3元')
        self.db.execute('insert into dm.ut_alg_flowback_msg_outcall_label_v2_di values (' + ','.join('?' * 15) + ')',
                        (day, pid, flow, group, '迭代流量', label, sms, call, cost, 0, 0, 0, 0, 0, 0))
        if feature:
            values = dict(dt=day, pid=pid, flowtree_group_name=flow, group_name_eve=group,
                          new_exp='迭代流量', oneid=pid)
            self.db.execute('insert into dm.ut_alg_flowback_msg_outcall_feature_train_di values (' +
                            ','.join('?' * len(self.feature_columns)) + ')',
                            [values.get(c, feature_value) for c in self.feature_columns])

    def add_feature(self, pid, day, control=False, group=None, feature_value='1'):
        flow = 'DS空白' if control else '实验'
        group = group or ('control_group' if control else '留存6张3元')
        values = dict(dt=day, pid=pid, flowtree_group_name=flow, group_name_eve=group,
                      new_exp='迭代流量', oneid=pid)
        self.db.execute('insert into dm.ut_alg_flowback_msg_outcall_feature_train_di values (' +
                        ','.join('?' * len(self.feature_columns)) + ')',
                        [values.get(c, feature_value) for c in self.feature_columns])

    def selected(self, split='train'):
        sql = profile.selection(*profile.SPLITS[split]) + '\nselect * from selected_keys'
        return [dict(r) for r in self.db.execute(sqlite_sql(sql))]

    def full(self, split='train'):
        sql = (HERE / ('feature_v6_dfcl_mt_' + split + '.sql')).read_text()
        return [dict(r) for r in self.db.execute(sqlite_sql(sql))]

    def audit(self):
        return [dict(r) for r in self.db.execute(sqlite_sql((HERE / 'audit_samples.sql').read_text()))]

    def audit_file(self, name):
        return [dict(r) for r in self.db.execute(sqlite_sql((HERE / name).read_text()))]

    def test_eight_and_twenty_continuous_negatives_keep_only_last(self):
        for pid, size in [('eight', 8), ('twenty', 20)]:
            for offset in range(size):
                self.add(pid, date_add('2026-07-01', offset))
        rows = self.selected()
        self.assertEqual([(r['pid'], r['dt'], r['member_cnt']) for r in rows],
                         [('twenty', '2026-07-20', 5)])
        # Expand the query window to include the eight-day user's last date.
        q = profile.selection('2026-07-01', '2026-07-31') + '\nselect * from selected_keys order by pid'
        self.assertEqual([(r['pid'], r['dt']) for r in self.db.execute(sqlite_sql(q))],
                         [('eight', '2026-07-08'), ('twenty', '2026-07-20')])

    def test_success_positive_keeps_selected_row_features_and_cost(self):
        self.add('u', '2026-07-16', cost=0.1, feature_value='11')
        self.add('u', '2026-07-18', label=2, call=1, cost=3.2, feature_value='18')
        self.add('u', '2026-07-20', cost=0.1, feature_value='20')
        row, = self.full()
        self.assertEqual((row['label_dt'], row['label'], row['cost_7d'], row['exp_label']),
                         ('2026-07-18', 2, 3.2, 1))
        self.assertEqual(row['sirius_city_id'], '18')
        self.assertEqual(len(row), 473 + 15 + 1)

    def test_success_all_zero_keeps_success_anchor(self):
        self.add('u', '2026-07-16', sms=1, cost=2)
        self.add('u', '2026-07-19', cost=0.1)
        row, = self.full()
        self.assertEqual((row['label_dt'], row['cost_7d'], row['exp_label']), ('2026-07-16', 2, 1))
        audit, = self.audit()
        self.assertEqual(audit['success_anchor_dt'], '2026-07-16')
        self.assertEqual(audit['label_anchor_mismatch_flag'], 0)

    def test_sms_success_positive_beats_larger_later_attributed_rows(self):
        self.add('u', '2026-07-16', label=2, sms=1)
        self.add('u', '2026-07-17', label=3, cost=7)
        self.add('u', '2026-07-18', label=3, cost=8)
        self.add('u', '2026-07-19')
        row, = self.full()
        self.assertEqual((row['label_dt'], row['label'], row['cost_7d']), ('2026-07-16', 2, 0))
        self.assertEqual(self.audit()[0]['positive_member_cnt'], 3)

    def test_non_anchor_positive_never_replaces_success_anchor(self):
        self.add('u', '2026-07-16', sms=1, label=0, cost=2)
        self.add('u', '2026-07-18', label=1, cost=3)
        row, = self.full()
        self.assertEqual((row['label_dt'], row['label'], row['exp_label']),
                         ('2026-07-16', 0, 1))
        audit, = self.audit()
        self.assertEqual((audit['success_anchor_dt'], audit['label_anchor_mismatch_flag']),
                         ('2026-07-16', 0))

    def test_outcall_success_positive_beats_larger_later_attributed_row(self):
        self.add('u', '2026-07-16', label=1, call=1)
        self.add('u', '2026-07-18', label=8)
        row, = self.full()
        self.assertEqual((row['label_dt'], row['label']), ('2026-07-16', 1))
        audit, = self.audit()
        self.assertEqual((audit['positive_priority_rank'], audit['selected_priority_reason']),
                         (1, 'outcall_success_positive'))

    def test_user_case_multiple_positive_rows_after_success(self):
        self.add('u', '2026-07-16')
        self.add('u', '2026-07-17')
        self.add('u', '2026-07-18', label=2, call=1)
        self.add('u', '2026-07-20', label=1)
        rows = self.selected()
        self.assertEqual([(r['success_anchor_dt'], r['dt'], r['finish_cnt_7d'],
                           r['positive_member_cnt']) for r in rows],
                         [('2026-07-18', '2026-07-18', 2, 2)])
        self.assertEqual((rows[0]['pre_touch_retry_cnt'], rows[0]['pre_touch_positive_cnt']), (2, 0))

    def test_positive_before_success_is_not_counted_or_selected_in_success_window(self):
        self.add('u', '2026-07-16', label=9)
        self.add('u', '2026-07-18', label=0, sms=1)
        row, = self.selected()
        self.assertEqual((row['dt'], row['positive_member_cnt'], row['pre_touch_positive_cnt']),
                         ('2026-07-18', 0, 1))

    def test_user_case_zero_success_cycle_and_later_positive_cycle_both_remain(self):
        self.add('u', '2026-07-16', sms=1)
        self.add('u', '2026-07-24', label=1)
        rows = self.selected()
        self.assertEqual([(r['candidate_kind'], r['dt'], r['finish_cnt_7d']) for r in rows],
                         [('success_7d', '2026-07-16', 0), ('failed_streak', '2026-07-24', 1)])

    def test_new_audit_queries_cover_both_user_cases(self):
        self.add('multi', '2026-07-16', sms=1, label=1)
        self.add('multi', '2026-07-18', label=1)
        self.add('transition', '2026-07-16', sms=1)
        self.add('transition', '2026-07-24', label=1)
        summary = self.audit_file('audit_multi_positive_success_summary.sql')
        self.assertEqual(sum(r['multi_positive_cycle_cnt'] for r in summary), 1)
        details = self.audit_file('audit_multi_positive_success_details.sql')
        self.assertEqual([(r['pid'], r['entry_dt'], r['selected_flag']) for r in details],
                         [('multi', '2026-07-16', 1), ('multi', '2026-07-18', 0)])
        self.assertTrue(all(r['selected_priority_reason'] == 'sms_success_positive' for r in details))
        transitions = self.audit_file('audit_zero_success_then_next_positive.sql')
        self.assertEqual([(r['pid'], r['selected_negative_dt'], r['next_selected_dt']) for r in transitions],
                         [('transition', '2026-07-16', '2026-07-24')])

    def test_gap_seven_does_not_start_new_unsuccessful_streak(self):
        self.add('u', '2026-07-16')
        self.add('u', '2026-07-17')
        self.add('u', '2026-07-24')
        self.assertEqual([r['dt'] for r in self.selected()], ['2026-07-24'])

    def test_gap_twenty_eight_starts_reentry_streak(self):
        self.add('u', '2026-07-16')
        self.add('u', '2026-07-17')
        self.add('u', '2026-08-14')
        self.assertEqual([r['dt'] for r in self.selected()], ['2026-07-17', '2026-08-14'])

    def test_two_successes_seven_days_apart_are_two_candidates(self):
        for day in ['2026-07-16', '2026-07-23']:
            self.add('u', day, label=1, sms=1)
        self.assertEqual(len(self.selected()), 2)

    def test_no_whole_user_outcome_filter_and_control_cost_zero(self):
        self.add('u', '2026-07-16', cost=2, control=True)
        self.add('u', '2026-07-17', cost=3, control=True)
        self.add('u', '2026-08-15', label=2, cost=4, control=True)
        rows = self.full()
        self.assertEqual([(r['label_dt'], r['label'], r['cost_7d']) for r in rows],
                         [('2026-07-17', 0, 0), ('2026-08-15', 2, 0)])

    def test_identical_label_duplicates_do_not_reinflate_output(self):
        self.add('u', '2026-07-16', label=2, cost=3)
        self.add('u', '2026-07-16', label=2, cost=3, feature=False)
        self.assertEqual(len(self.full()), 1)

    def test_missing_label_is_reported(self):
        self.add('null', '2026-07-16', label=None)
        self.assertEqual(self.full(), [])
        self.assertEqual([r['sample_status'] for r in self.audit()], ['invalid_label'])

    def test_feature_duplicate_or_missing_never_selects_another_representative(self):
        self.add('duplicate', '2026-07-16')
        self.add_feature('duplicate', '2026-07-16')
        self.add('missing', '2026-07-16')
        self.add('missing', '2026-07-17', label=1, feature=False)
        self.assertEqual(self.full(), [])
        self.assertEqual({r['output_status'] for r in self.audit()},
                         {'duplicate_selected_features', 'missing_selected_features'})

    def test_splits_use_their_own_v6_date_ranges(self):
        self.add('u', '2026-08-20', sms=1)
        self.add('u', '2026-08-22', label=2)
        self.add('u', '2026-08-24')
        self.assertEqual(self.full('train')[0]['label_dt'], '2026-08-20')
        self.assertEqual(self.full('test'), [])
        self.assertEqual(self.full('valid')[0]['label_dt'], '2026-08-22')

    def test_excluded_date_is_removed_before_selection_like_v6(self):
        self.add('u', '2026-08-01')
        self.add('u', '2026-08-02', label=2, call=1)
        self.add('u', '2026-08-03')
        row, = self.full()
        self.assertEqual((row['label_dt'], row['label'], row['exp_label']),
                         ('2026-08-03', 0, 0))

    def test_test_split_stops_at_v6_end_date(self):
        for offset in range(15):
            self.add('u', date_add('2026-09-02', offset))
        self.assertEqual(self.full('test')[0]['label_dt'], '2026-09-10')
        row, = self.audit()
        self.assertEqual(row['selected_dt'], '2026-09-10')
        self.assertEqual(row['output_status'], 'output')
        self.assertEqual(row['right_boundary_flag'], 1)

    def test_all_splits_match_v6_column_names_and_order(self):
        for split, (start, _) in profile.SPLITS.items():
            self.add(split, start)
            new = self.full(split)
            original = (HERE.parent / 'v6' / ('feature_v6_' + split + '.sql')).read_text()
            original = re.sub(r'--[^\n]*', '', original)
            legacy_cursor = self.db.execute(original)
            self.assertEqual(list(new[0]), [c[0] for c in legacy_cursor.description])
            self.assertEqual(new[0]['flag'], split)


if __name__ == '__main__':
    unittest.main()
