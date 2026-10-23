"""Small regression tests for the production sample-selection CTE.

These tests validate selection behavior with SQLite. They do not replace a
Spark/Hive parser check or a real-table distribution audit.
"""
import datetime as dt
from pathlib import Path
import re
import sqlite3
import unittest


HERE = Path(__file__).parent
PROFILE_FILES = {
    "train": HERE / "feature_v6_dfcl_touch_train.sql",
    "valid": HERE / "feature_v6_dfcl_touch_valid.sql",
    "test": HERE / "feature_v6_dfcl_touch_test.sql",
}


def date_add(value, days):
    if value is None or days is None:
        return None
    return (dt.date.fromisoformat(value) + dt.timedelta(days=int(days))).isoformat()


def datediff(left, right):
    if left is None or right is None:
        return None
    return (dt.date.fromisoformat(left) - dt.date.fromisoformat(right)).days


def selection_query(path):
    sql = path.read_text()
    start = sql.index("with\nlabel_raw")
    end = sql.index(",\noriginal_v6_rows as (")
    query = sql[start:end] + "\nselect * from selected_keys"
    return query.replace(" as string)", " as text)")


class ProductionSelectionTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("attach database ':memory:' as dm")
        self.db.create_function("to_date", 1, lambda value: value)
        self.db.create_function("date_add", 2, date_add)
        self.db.create_function("datediff", 2, datediff)
        self.db.create_function("floor", 1, lambda value: int(value // 1) if value is not None else None)
        self.db.create_function("concat_ws", -1, lambda sep, *values: sep.join(str(v) for v in values if v is not None))
        self.db.execute("""create table dm.ut_alg_flowback_msg_outcall_label_v2_di (
            dt text, pid text, flowtree_group_name text, group_name_eve text,
            new_exp text, finish_cnt_7d real, send_success_cnt integer,
            outcall_success_cnt integer, send_cost real, send_remind_cost real,
            outcall_cost real, remind_cost real, msg_coupon_cost_7d real,
            outcall_coupon_cost_7d real, outcall_coupon2_cost_7d real
        )""")

    def tearDown(self):
        self.db.close()

    def add(self, pid, day, label=0, sms=0, call=0, control=False, group=None, cost=0):
        flow = "DS空白" if control else "实验"
        group = group or ("control_group" if control else "留存6张3元")
        self.db.execute(
            "insert into dm.ut_alg_flowback_msg_outcall_label_v2_di values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (day, pid, flow, group, "迭代流量", label, sms, call, cost, 0, 0, 0, 0, 0, 0),
        )

    def selected(self, split="train"):
        return [dict(row) for row in self.db.execute(selection_query(PROFILE_FILES[split]))]

    def test_success_candidate_keeps_attributed_positive(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17")
        self.add("u", "2026-07-18", label=2, call=1)
        self.add("u", "2026-07-19")
        rows = self.selected()
        self.assertEqual([(r["pid"], r["dt"]) for r in rows], [("u", "2026-07-18")])

    def test_success_candidate_all_zero_keeps_success_day(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17")
        self.add("u", "2026-07-18", sms=1)
        self.add("u", "2026-07-20")
        self.assertEqual(self.selected()[0]["dt"], "2026-07-18")

    def test_no_success_uses_fixed_seven_day_blocks_and_last_negative(self):
        for offset in range(14):
            self.add("u", date_add("2026-07-16", offset))
        self.assertEqual([r["dt"] for r in self.selected()], ["2026-07-22", "2026-07-29"])

    def test_positive_beats_last_negative_in_failed_block(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17", label=1)
        self.add("u", "2026-07-18")
        self.assertEqual(self.selected()[0]["dt"], "2026-07-17")

    def test_conflicting_source_or_cross_treatment_is_excluded(self):
        self.add("conflict", "2026-07-16", label=0)
        self.add("conflict", "2026-07-16", label=1)
        self.add("cross", "2026-07-16")
        self.add("cross", "2026-07-17", group="留存6张5元")
        self.assertEqual(self.selected(), [])

    def test_split_dates_and_removed_outcome_filter_are_static(self):
        expected = {
            "train": ("2026-07-16", "2026-08-21"),
            "valid": ("2026-08-22", "2026-08-27"),
            "test": ("2026-08-28", "2026-09-10"),
        }
        for split, path in PROFILE_FILES.items():
            sql = path.read_text()
            begin, end = expected[split]
            self.assertIn(f"and dt between to_date('{begin}') and to_date('{end}')", sql)
            self.assertNotIn("finish_all_cnt", sql)
            self.assertIn("case when group_name_eve = 'control_group' then cast(0.0 as double)", sql)
            self.assertRegex(sql, r"select o\.\*\s+from original_v6_rows o")

    def test_selection_cte_is_same_across_splits_except_date_filter(self):
        normalized = []
        for path in PROFILE_FILES.values():
            query = selection_query(path)
            query = re.sub(
                r"and dt between to_date\('[0-9-]+'\) and to_date\('[0-9-]+'\)",
                "and dt between SPLIT_BEGIN and SPLIT_END",
                query,
            )
            normalized.append(query)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])


if __name__ == "__main__":
    unittest.main()
