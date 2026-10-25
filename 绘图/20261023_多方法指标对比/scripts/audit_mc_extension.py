"""Read-only numeric checks for the MC extension; never generates endpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import runpy
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
DATA = PROJECT / "data/mc_extension_v2"


def audit(partial=False, self_test=False, data_dir=DATA):
    api = runpy.run_path(str(PROJECT / "scripts/plot_metrics_comparison.py"), run_name="read_only")
    protocol, points, lookup, report = api["load_and_check"](data_dir, partial)
    old = [json.loads(s) for s in (PROJECT / "data/assumed_intervals.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    assert points[:84] == old and len(old) == 84, "Historical point records changed"
    mc = points[84:]
    assert all(row["metric"] == "MC" for row in mc)
    for index, row in enumerate(mc, 1):
        assert row["id"] == f"MC{index:03d}", "MC batch order changed"
    means = {}
    for method in protocol["methods"]:
        means[method["id"]] = {}
        for definition in protocol["metrics"]:
            values = [lookup.get((method["id"], scenario["id"], definition["id"])) for scenario in protocol["scenarios"]]
            if all(values):
                means[method["id"]][definition["id"]] = round(sum(row["estimate"] for row in values) / len(values), definition.get("decimals", 1))
    table_checked = 0
    if not partial:
        text = (REPO / "CCDC/latex/1020/main.tex").read_text(encoding="utf-8")
        block = text.split(r"\label{tab:main_results}")[1].split(r"\end{table}")[0]
        labels = {"PPO_beta_L":r"PPO$_{\beta_L}$", "PPO_beta_H":r"PPO$_{\beta_H}$", "CPPO_beta_L":r"CPPO$_{\beta_L}$", "CPPO_beta_H":r"CPPO$_{\beta_H}$"}
        for method in protocol["methods"]:
            label = labels.get(method["id"], method["id"])
            rows = [line.split(" & ") for line in block.splitlines() if " & " in line]
            cells = next(c for c in rows if re.sub(r"\\textbf\{([^}]+)\}", r"\1", c[0]) == label)
            actual = [float(re.search(r"[0-9]+(?:\.[0-9]+)?", c)[0]) for c in cells[1:]]
            expected = [means[method["id"]][metric] for metric in ("SR", "Cost", "CR", "BVR", "MC")]
            assert actual == expected, (method["id"], actual, expected)
            table_checked += 5
    rejected = []
    if self_test:
        cases = ["negative_distance", "inverted_interval", "duplicate_id", "changed_center", "incomplete_full_data", "mc_percent_unit", "mc_insufficient_precision"]
        for case in cases:
            p, rows = copy.deepcopy(protocol), copy.deepcopy(points)
            partial_case = True
            if case == "negative_distance": rows[0]["lower"] = -0.1
            if case == "inverted_interval": rows[0]["lower"] = rows[0]["upper"] + 1
            if case == "duplicate_id": rows.append(copy.deepcopy(rows[0]))
            if case == "changed_center": rows[0]["estimate"] += 0.001
            if case == "incomplete_full_data": rows = rows[:-1]; partial_case = False
            if case == "mc_percent_unit": p["metrics"][-1]["unit"] = "%"
            if case == "mc_insufficient_precision": p["metrics"][-1]["decimals"] = 1
            with tempfile.TemporaryDirectory(prefix="mc_schema_check_") as directory:
                path = Path(directory)
                (path / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
                (path / "assumed_intervals.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                try:
                    api["load_and_check"](path, partial_case)
                except (ValueError, KeyError):
                    rejected.append(case)
                else:
                    raise AssertionError(f"Invalid fixture accepted: {case}")
    return {"status":"passed", "points":len(points), "mc_points":len(mc), "endpoints":2*len(points),
            "legacy_84_records_unchanged":True, "table_values_checked":table_checked,
            "means":means, "negative_cases_rejected":rejected,
            "points_sha256":hashlib.sha256((data_dir / "assumed_intervals.jsonl").read_bytes()).hexdigest(),
            "data_status":"hypothetical", "empirical_validation":False,
            "renderer_checks":report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    result = audit(args.partial, args.self_test, args.data_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
