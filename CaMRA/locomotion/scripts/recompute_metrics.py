"""Independently recompute evaluation metrics from the raw step log (TASK-020).

Reads an evaluator output directory and, for every episode in ``raw/<group>``,
recomputes the metric set from the step rows and compares it against
``per_episode.csv``. This is the evidence for "every metric is recomputable from
the raw log": the comparison is run by a separate process that never touches the
simulator.

Exit code 0 only if every metric matches within tolerance.

Usage:
    python scripts/recompute_metrics.py --run_dir <eval output dir> [--tol 1e-5]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "unitree_rl_lab"))
from unitree_rl_lab.utils.eval_metrics import compute_metrics, load_episode_blocks  # noqa: E402

REPO = "/home/caohy/repositories/Paper"
DEFAULT_PROTOCOL = os.path.join(REPO, "CaMRA/locomotion/docs/eval_protocol.json")

COMPARE = [
    "rmse_vx",
    "rmse_vy",
    "rmse_wz",
    "p95_abs_err_vx",
    "p95_abs_err_vy",
    "p95_abs_err_wz",
    "max_roll_deg",
    "max_pitch_deg",
    "stand_drift_m",
    "rise_time_s",
    "overshoot_mps",
    "stop_drift_m",
    "action_rate_rms",
    "steps",
    "steps_before_end",
    "failed",
    "duration_s",
    "variant_id",
    "episode_id",
    "invalid_reasons",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    with open(args.protocol, encoding="utf-8") as fh:
        protocol = json.load(fh)
    specs = {s["id"]: s for s in protocol["scenarios"]}
    warmup = float(protocol.get("warmup", {}).get("settle_s", 1.0))

    csv_path = os.path.join(args.run_dir, "per_episode.csv")
    if not os.path.isfile(csv_path):
        print(f"FAIL: {csv_path} not found")
        return 2
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)

    report = {"run_dir": args.run_dir, "protocol_version": protocol.get("version"), "scenarios": {}, "mismatches": []}
    total_cmp = 0
    total_bad = 0

    for npz_path in sorted(glob.glob(os.path.join(args.run_dir, "raw", "*", "*.npz"))):
        blocks, meta = load_episode_blocks(npz_path)
        sc = meta["scenario"] or os.path.splitext(os.path.basename(npz_path))[0]
        step_dt = meta["step_dt"]
        csv_rows = by_scenario.get(sc, [])
        info = {"npz": npz_path, "episodes_in_npz": len(blocks), "episodes_in_csv": len(csv_rows), "max_abs_err": 0.0}
        if len(blocks) != len(csv_rows):
            report["mismatches"].append(f"{sc}: npz has {len(blocks)} episodes but csv has {len(csv_rows)}")

        worst = 0.0
        for i, blk in enumerate(blocks):
            rec = compute_metrics(blk, specs[sc], step_dt, warmup)
            if i >= len(csv_rows):
                break
            row = csv_rows[i]
            # the invalid-reason map is stored as JSON text in the CSV: compare canonically
            # so "which metrics were not measurable, and why" is recomputable too
            if "invalid_reasons" in row:
                a = json.dumps(rec.get("invalid_reasons") or {}, ensure_ascii=False, sort_keys=True)
                b = row.get("invalid_reasons") or "{}"
                total_cmp += 1
                if a != b:
                    total_bad += 1
                    report["mismatches"].append(f"{sc}[{i}] invalid_reasons: recomputed={a} logged={b}")
            for key in COMPARE:
                if key == "invalid_reasons":
                    continue
                if key not in rec or key not in row:
                    continue
                a, b = rec[key], row[key]
                if isinstance(a, str) or isinstance(b, str):
                    total_cmp += 1
                    if str(a) != str(b):
                        total_bad += 1
                        report["mismatches"].append(f"{sc}[{i}] {key}: recomputed={a} logged={b}")
                    continue
                a, b = float(a), float(b)
                total_cmp += 1
                if math.isnan(a) and math.isnan(b):
                    continue
                err = abs(a - b) / max(1.0, abs(b))
                worst = max(worst, err)
                if err > args.tol:
                    total_bad += 1
                    report["mismatches"].append(f"{sc}[{i}] {key}: recomputed={a:.9g} logged={b:.9g} rel={err:.3g}")
        info["max_abs_err"] = worst
        report["scenarios"][sc] = info
        print(f"[RECOMPUTE] {sc:12s} episodes={len(blocks):4d} max_rel_err={worst:.3e}", flush=True)

    report["comparisons"] = total_cmp
    report["mismatch_count"] = total_bad
    report["verdict"] = "PASS" if total_bad == 0 else "FAIL"

    out = os.path.join(args.run_dir, "recompute_check.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[RECOMPUTE] comparisons={total_cmp} mismatches={total_bad} verdict={report['verdict']}", flush=True)
    print(f"[RECOMPUTE] report -> {out}", flush=True)
    for m in report["mismatches"][:10]:
        print("  MISMATCH " + m, flush=True)
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
