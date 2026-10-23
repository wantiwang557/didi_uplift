"""Run the audit SQL against small synthetic fixtures using SQLite.

SQLite implements the SQL windows/joins; UDFs emulate the date/string functions.
This checks selection behavior, not Spark parser/type compatibility or Hive data.
"""
import ast
import datetime as dt
from pathlib import Path
import sqlite3
import unittest


def date_add(value, days):
    if value is None or days is None:
        return None
    return (dt.date.fromisoformat(value) + dt.timedelta(days=int(days))).isoformat()


def datediff(left, right):
    if left is None or right is None:
        return None
    return (dt.date.fromisoformat(left) - dt.date.fromisoformat(right)).days


class SampleCasesTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.execute("attach database ':memory:' as dm")
        self.db.create_function('to_date', 1, lambda value: value)
        self.db.create_function('date_add', 2, date_add)
        self.db.create_function('datediff', 2, datediff)
        self.db.create_function('floor', 1, lambda value: int(value // 1) if value is not None else None)
        self.db.create_function('concat_ws', -1, lambda sep, *values: sep.join(str(v) for v in values if v is not None))
        self.db.execute('''create table dm.ut_alg_flowback_msg_outcall_label_v2_di (
            dt text, pid text, flowtree_group_name text, group_name_eve text,
            new_exp text, finish_cnt_7d real, send_success_cnt integer,
            outcall_success_cnt integer, send_cost real, send_remind_cost real,
            outcall_cost real, remind_cost real, msg_coupon_cost_7d real,
            outcall_coupon_cost_7d real, outcall_coupon2_cost_7d real
        )''')
        self.query = Path(__file__).with_name('sample_cases.sql').read_text()
        # SQLite calls the string type TEXT. Spark uses STRING.
        self.query = self.query.replace(' as string)', ' as text)')

    def tearDown(self):
        self.db.close()

    def add(self, pid, day, label=0, sms=0, call=0, control=False, cost=0, group=None):
        flow = 'DS空白' if control else '实验'
        group = group or ('control_group' if control else '留存6张3元')
        self.db.execute('''insert into dm.ut_alg_flowback_msg_outcall_label_v2_di
            values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (day, pid, flow, group, '迭代流量', label, sms, call, cost, 0, 0, 0, 0, 0, 0))

    def rows(self):
        return [dict(row) for row in self.db.execute(self.query)]

    def kept(self):
        return [row for row in self.rows() if row['would_keep_flag'] == 1]

    def test_retries_then_positive_success(self):
        self.add('a', '2026-07-16')
        self.add('a', '2026-07-17')
        self.add('a', '2026-07-18', label=1, call=1)
        self.add('a', '2026-07-19')
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        selected = self.kept()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['selected_dt'], '2026-07-18')
        self.assertEqual(selected[0]['observation_end_dt_assumed'], '2026-07-24')
        self.assertEqual(selected[0]['selected_finish_cnt_7d'], 1)

    def test_success_without_orders_keeps_success_day(self):
        for day in ('2026-07-16', '2026-07-17', '2026-07-19', '2026-07-22'):
            self.add('a', day)
        self.add('a', '2026-07-18', sms=1)
        selected = self.kept()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['selected_dt'], '2026-07-18')
        self.assertEqual(selected[0]['selection_reason'], 'successful_anchor_with_zero_label')

    def test_failed_and_control_continuous_rows_are_not_one_long_cycle(self):
        for pid, control in (('a', False), ('b', True)):
            for offset in range(14):
                self.add(pid, date_add('2026-07-16', offset), control=control, cost=3)
        selected = self.kept()
        for pid in ('a', 'b'):
            rows = [r for r in selected if r['pid'] == pid]
            self.assertEqual([r['selected_dt'] for r in rows], ['2026-07-22', '2026-07-29'])
            self.assertEqual([r['observation_start_dt'] for r in rows], ['2026-07-22', '2026-07-29'])
        self.assertTrue(all(r['cost_7d'] == 0 for r in selected if r['pid'] == 'b'))

    def test_failed_touch_can_have_positive_label(self):
        self.add('a', '2026-07-16')
        self.add('a', '2026-07-17', label=2)
        self.add('a', '2026-07-18')
        row = self.kept()[0]
        self.assertEqual(row['selected_dt'], '2026-07-17')
        self.assertEqual(row['touch_success'], 0)
        self.assertEqual(row['case_type'], 'failed_positive')
        self.assertEqual(row['observation_start_dt'], '2026-07-17')

    def test_earlier_negative_round_survives_later_positive(self):
        self.add('a', '2026-07-16', call=1)
        self.add('a', '2026-07-20')
        self.add('a', '2026-07-29')
        self.add('a', '2026-07-30', call=1, label=2)
        selected = sorted(self.kept(), key=lambda r: r['selected_dt'])
        self.assertEqual([r['selected_dt'] for r in selected], ['2026-07-16', '2026-07-30'])
        self.assertEqual([r['selected_finish_cnt_7d'] for r in selected], [0, 2])

    def test_success_overlap_is_visible(self):
        self.add('a', '2026-07-16', call=1)
        self.add('a', '2026-07-19', sms=1)
        self.assertEqual(len(self.kept()), 2)
        self.assertTrue(all(r['overlapping_success_flag'] == 1 for r in self.kept()))

    def test_same_valued_multiple_positive_rows_are_detected(self):
        self.add('a', '2026-07-16', label=1)
        self.add('a', '2026-07-17', label=1)
        row = self.kept()[0]
        self.assertEqual(row['positive_member_cnt'], 2)
        self.assertEqual(row['case_type'], 'multiple_positive')
        self.assertEqual(row['selected_dt'], '2026-07-17')

    def test_positive_not_at_success_anchor_is_not_silently_accepted(self):
        self.add('a', '2026-07-17', label=1)
        self.add('a', '2026-07-18', call=1)
        row = self.kept()[0]
        self.assertEqual(row['selected_dt'], '2026-07-17')
        self.assertEqual(row['label_anchor_mismatch_flag'], 1)
        self.assertEqual(row['observation_start_dt'], '2026-07-18')

    def test_exact_duplicates_collapse_but_conflicts_are_flagged(self):
        self.add('exact', '2026-07-16', label=1)
        self.add('exact', '2026-07-16', label=1)
        self.add('conflict', '2026-07-16', label=0)
        self.add('conflict', '2026-07-16', label=1)
        selected = {r['pid']: r for r in self.kept()}
        self.assertEqual(selected['exact']['source_row_cnt'], 2)
        self.assertEqual(selected['exact']['source_variant_cnt'], 1)
        self.assertEqual(selected['exact']['user_source_conflict_flag'], 0)
        self.assertEqual(selected['conflict']['source_variant_cnt'], 2)
        self.assertEqual(selected['conflict']['case_type'], 'source_problem')

    def test_null_label_and_cross_group_are_not_hidden(self):
        self.add('null', '2026-07-16', label=None)
        self.add('cross', '2026-07-16')
        self.add('cross', '2026-07-17', group='留存6张5元')
        selected = self.kept()
        self.assertTrue(all(r['case_type'] == 'source_problem' for r in selected))
        self.assertTrue(all(r['label_maturity_status'] == 'unknown' for r in selected))

    def test_plus_six_included_plus_seven_not_in_success_window(self):
        self.add('a', '2026-07-16', call=1)
        self.add('a', '2026-07-22')
        self.add('a', '2026-07-23')
        rows = {r['entry_dt']: r for r in self.rows()}
        self.assertEqual(rows['2026-07-22']['candidate_kind'], 'success')
        self.assertEqual(rows['2026-07-23']['candidate_kind'], 'failed_fixed7')

    def test_split_boundary_and_exclusion_are_reported(self):
        for day in ('2026-08-20', '2026-08-21', '2026-08-22', '2026-08-26'):
            self.add('a', day, control=True)
        self.add('excluded', '2026-08-02', call=1)
        rows = {r['pid']: r for r in self.kept()}
        self.assertEqual(rows['a']['split'], 'valid')
        self.assertEqual(rows['a']['crosses_split_boundary_flag'], 1)
        self.assertEqual(rows['excluded']['selected_excluded_date_flag'], 1)

    def test_case_sampling_preserves_whole_candidate(self):
        for user in range(7):
            for offset in range(3):
                self.add(str(user), date_add('2026-07-16', offset))
        rows = self.rows()
        self.assertEqual(len(rows), 15)  # 5 candidates, all 3 members each
        self.assertEqual(len(self.kept()), 5)

    def test_late_order_attributed_to_new_record_does_not_delete_old_zero(self):
        self.add('a', '2026-07-16')
        self.add('a', '2026-07-17')
        self.add('a', '2026-07-18', call=1, label=0)
        self.add('a', '2026-07-25', label=1)
        rows = sorted(self.kept(), key=lambda r: r['selected_dt'])
        self.assertEqual([r['selected_dt'] for r in rows], ['2026-07-18', '2026-07-25'])
        self.assertEqual([r['selected_finish_cnt_7d'] for r in rows], [0, 1])

    def test_actual_reentry_38_days_later_is_preserved(self):
        self.add('a', '2026-07-16', call=1)
        self.add('a', date_add('2026-07-16', 38), call=1, label=1)
        rows = sorted(self.kept(), key=lambda r: r['selected_dt'])
        self.assertEqual([r['selected_dt'] for r in rows], ['2026-07-16', '2026-08-23'])
        self.assertEqual([r['split'] for r in rows], ['train', 'valid'])

    def test_fallback_observation_windows_can_overlap_and_are_flagged(self):
        self.add('a', '2026-07-16')
        self.add('a', '2026-07-20', label=1)
        self.add('a', '2026-07-24', label=1)
        rows = sorted(self.kept(), key=lambda r: r['selected_dt'])
        self.assertEqual([r['observation_overlap_flag'] for r in rows], [0, 1])

    def test_notebook_query_is_identical_to_sql_file(self):
        notebook = Path(__file__).with_name('sample_cases_query.py').read_text()
        module = ast.parse(notebook)
        query_values = [node.value.value for node in module.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'query' for t in node.targets)]
        self.assertEqual(len(query_values), 1)
        self.assertEqual(query_values[0].strip(), Path(__file__).with_name('sample_cases.sql').read_text().strip())


if __name__ == '__main__':
    unittest.main()
