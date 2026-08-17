#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


GROUPS = [
    "sirius画像特征",
    "补贴环境",
    "双端特征",
    "有端",
    "券敏感",
    "新版小天狼星用户订单信息特征",
    "新版小天狼星用户基础属性特征",
]
# 定义三套方案
'''
baseline: sirius画像 + 补贴环境 + 双端 + 有端 + 券敏感
no_sirius: 补贴环境 + 双端 + 有端 + 券敏感
replace_sirius_with_xtlx_order:补贴环境 + 双端 + 有端 + 券敏感 + 新版小天狼星订单信息
'''

SCHEMES = {
    "baseline": GROUPS[:5],
    "no_sirius": GROUPS[1:5],
    "replace_sirius_with_xtlx_order": GROUPS[1:5] + [GROUPS[5]],
    "all_superset": GROUPS,
}


def strip_inline_comment(line):
    return line.split("--", 1)[0].strip()


def select_alias(expr):
    expr = expr.rstrip(",").strip()
    match = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", expr, re.I)
    if match:
        return match.group(1)
    token = expr.split(",")[-1].strip()
    token = token.split()[-1]
    return token.split(".")[-1]


def parse_sql_groups(sql_path):
    group_cols = {group: [] for group in GROUPS}
    current_group = None
    expr_buffer = []

    for raw_line in Path(sql_path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("--"):
            comment = line[2:].strip()
            if comment in group_cols:
                current_group = comment
            continue

        if current_group is None:
            continue

        if re.match(r"^from\b", line, re.I):
            break

        expr = strip_inline_comment(line)
        if not expr:
            continue

        if expr_buffer:
            expr_buffer.append(expr)
            joined_expr = " ".join(expr_buffer)
            if re.search(r"\bend\s+as\s+[A-Za-z_][A-Za-z0-9_]*", joined_expr, re.I):
                group_cols[current_group].append(select_alias(joined_expr))
                expr_buffer = []
            continue

        if expr.startswith(","):
            expr = expr[1:].strip()
            if re.match(r"case\b", expr, re.I) and not re.search(r"\bend\s+as\s+", expr, re.I):
                expr_buffer = [expr]
            elif expr:
                group_cols[current_group].append(select_alias(expr))

    return group_cols


def parse_feature_conf(feature_conf_path):
    features = []
    for slot, raw_line in enumerate(Path(feature_conf_path).read_text().splitlines()):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        meta = {}
        for part in line.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                meta[key.strip()] = value.strip()
        depend = meta.get("depend", "")
        depends = [item.strip() for item in depend.split(",") if item.strip()]
        features.append({
            "slot": slot,
            "name": meta.get("name", ""),
            "depend": depend,
            "depends": depends,
            "raw": line,
        })
    return features


def slots_for_columns(features, columns):
    selected = set(columns)
    slots = []
    matched_depends = set()
    for feature in features:
        if any(dep in selected for dep in feature["depends"]):
            slots.append(feature["slot"])
            matched_depends.update(dep for dep in feature["depends"] if dep in selected)
    return slots, matched_depends


def write_slot_file(path, slots):
    path.write_text("\n".join(str(slot) for slot in slots) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build EUEN feature slot files from feature_conf and SQL groups.")
    parser.add_argument("--feature_conf", required=True)
    parser.add_argument("--sql", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_cols = parse_sql_groups(args.sql)
    features = parse_feature_conf(args.feature_conf)

    summary = {
        "feature_conf": str(Path(args.feature_conf).resolve()),
        "sql": str(Path(args.sql).resolve()),
        "feature_conf_slots": len(features),
        "groups": {},
        "schemes": {},
    }

    for group, cols in group_cols.items():
        slots, matched = slots_for_columns(features, cols)
        summary["groups"][group] = {
            "sql_columns": len(cols),
            "matched_feature_slots": len(slots),
            "missing_in_feature_conf": sorted(set(cols) - matched),
        }

    for scheme, groups in SCHEMES.items():
        cols = []
        for group in groups:
            cols.extend(group_cols[group])
        slots, matched = slots_for_columns(features, cols)
        file_name = f"{args.prefix}{scheme}.slots"
        slot_path = out_dir / file_name
        write_slot_file(slot_path, slots)
        summary["schemes"][scheme] = {
            "groups": groups,
            "sql_columns": len(cols),
            "matched_feature_slots": len(slots),
            "slot_file": str(slot_path),
            "missing_in_feature_conf": sorted(set(cols) - matched),
        }

    summary_path = out_dir / f"{args.prefix}summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(summary["schemes"], ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()



'''
cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt

python offline_bigdata/python/feature_select/build_feature_slot.py \
  --feature_conf offline_bigdata/script/feature_confs/baseline/feature_baseline_train.conf \
  --sql offline_bigdata/script/sqls/baseline/feature_baseline_train.sql \
  --out_dir demo/feature_select \
  --prefix baseline_
'''