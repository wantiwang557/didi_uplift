"""Synthetic regression tests for the v6_dfcl_episode selection CTE."""
import datetime as dt
from pathlib import Path
import sqlite3
import unittest


HERE = Path(__file__).parent


def datediff(left, right):
    if left is None or right is None:
        return None
    return (dt.date.fromisoformat(left) - dt.date.fromisoformat(right)).days


def selection_query(split="train"):
    path = HERE / f"feature_v6_dfcl_episode_{split}.sql"
    sql = path.read_text()
    start = sql.index("with\nlabel_raw")
    end = sql.index(",\noriginal_v6_rows as (")
    return (sql[start:end] + "\nselect * from selected_keys").replace(
        " as string)", " as text)"
    )


class EpisodeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("attach database ':memory:' as dm")
        self.db.create_function("to_date", 1, lambda value: value)
        self.db.create_function("datediff", 2, datediff)
        self.db.create_function(
            "concat_ws", -1,
            lambda sep, *values: sep.join(str(v) for v in values if v is not None),
        )
        self.db.execute("""create table dm.ut_alg_flowback_msg_outcall_label_v2_di (
            dt text, pid text, flowtree_group_name text, group_name_eve text,
            new_exp text, finish_cnt_7d real, send_success_cnt integer,
            outcall_success_cnt integer, send_cost real, send_remind_cost real,
            outcall_cost real, remind_cost real, msg_coupon_cost_7d real,
            outcall_coupon_cost_7d real, outcall_coupon2_cost_7d real
        )""")

    def tearDown(self):
        self.db.close()

    def add(self, pid, day, label=0, sms=0, call=0, control=False, group=None):
        flow = "DS空白" if control else "实验"
        group = group or ("control_group" if control else "留存6张3元")
        self.db.execute(
            "insert into dm.ut_alg_flowback_msg_outcall_label_v2_di values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (day, pid, flow, group, "迭代流量", label, sms, call, 0, 0, 0, 0, 0, 0, 0),
        )

    def selected_dates(self):
        return sorted(row["dt"] for row in self.db.execute(selection_query()))

    def test_success_without_order_is_one_negative_anchor(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17", call=1)
        self.add("u", "2026-07-18")
        self.add("u", "2026-07-25")
        self.assertEqual(self.selected_dates(), ["2026-07-17"])

    def test_later_attributed_positive_is_another_sample(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17", call=1, label=0)
        self.add("u", "2026-07-18")
        self.add("u", "2026-07-20", label=1)
        self.assertEqual(self.selected_dates(), ["2026-07-17", "2026-07-20"])

    def test_success_positive_row_is_not_duplicated(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17", call=1, label=2)
        self.add("u", "2026-07-18")
        self.assertEqual(self.selected_dates(), ["2026-07-17"])

    def test_never_success_all_negative_keeps_last(self):
        for day in ("2026-07-16", "2026-07-17", "2026-07-20"):
            self.add("u", day)
        self.assertEqual(self.selected_dates(), ["2026-07-20"])

    def test_no_success_with_positive_keeps_positive_not_last_zero(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17", label=1)
        self.add("u", "2026-07-18")
        self.assertEqual(self.selected_dates(), ["2026-07-17"])

    def test_gap_of_28_days_starts_new_episode(self):
        self.add("u", "2026-07-16")
        self.add("u", "2026-07-17")
        self.add("u", "2026-08-14")
        self.add("u", "2026-08-15")
        self.assertEqual(self.selected_dates(), ["2026-07-17", "2026-08-15"])

    def test_conflicting_same_day_source_is_excluded(self):
        self.add("u", "2026-07-16", label=0)
        self.add("u", "2026-07-16", label=1)
        self.assertEqual(self.selected_dates(), [])


if __name__ == "__main__":
    unittest.main()
