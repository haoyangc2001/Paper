#!/usr/bin/env python3
"""Print masked-vs-unmasked policy-layer metrics for NAV-064 round 12.

Reads train_log.jsonl of each run, keeps only update records, and reports the
windowed mean of the instrumentation fields written by policy_grad_decomp plus
the mask retention fraction. Read-only.
"""
import json
import statistics as st
import sys

R = "/pub/data/caohy/work/camra_navigation"
RUNS = [
    "full_s0", "full_s0_nomask",
    "uncalibrated_s0", "uncalibrated_s0_nomask",
]
KEYS = [
    "act_fwd_mean", "act_steer_absmean", "adv_std", "adv_mean",
    "policy_mask_kept_fraction", "reach_mask_kept_fraction",
    "grad_ent_share", "grad_adv_norm", "grad_ent_norm", "grad_adv_ent_cos",
]
WINDOWS = [(0, 100), (100, 200), (250, 350), (450, 550), (500, 600)]


def load(run):
    p = "%s/nav064_r12_%s/train_log.jsonl" % (R, run)
    out = []
    with open(p) as fh:
        for line in fh:
            if '"kind": "update"' not in line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def seg(rows, lo, hi, key):
    v = [r[key] for r in rows if lo <= r["update"] < hi and r.get(key) is not None]
    return (st.mean(v) if v else float("nan")), len(v)


def main():
    print("%-26s %s" % ("metric", "  ".join("u%d-%d" % (a, b - 1) for a, b in WINDOWS)))
    for run in RUNS:
        rows = load(run)
        if not rows:
            print("%-26s NO DATA" % run)
            continue
        print("== %s  (updates=%d)" % (run, len(rows)))
        for key in KEYS:
            if key not in rows[0]:
                continue
            cells = []
            for a, b in WINDOWS:
                m, n = seg(rows, a, b, key)
                cells.append("%8.4f" % m if n else "     n/a")
            print("  %-24s %s" % (key, "  ".join(cells)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
