#!/usr/bin/env python3
"""Final-window numbers for the NAV-064 r12 acceptance criteria.

Criterion 1: deterministic forward command of the full variant at u600.
Criterion 2: advantage std on the effective (safety_y<=0.5) samples. When the
             dead-step mask is on, ppo.normalize_advantage computes adv_std on the
             masked samples only, so the logged field *is* the effective-segment std.
Read-only.
"""
import json
import statistics as st

R = "/pub/data/caohy/work/camra_navigation"
RUNS = [
    ("full", "full_s0"), ("full", "full_s1"), ("full", "full_s2"),
    ("full_ctl", "full_s0_nomask"),
    ("uncal", "uncalibrated_s0"), ("uncal", "uncalibrated_s1"), ("uncal", "uncalibrated_s2"),
    ("uncal_ctl", "uncalibrated_s0_nomask"),
]


def load(run):
    p = "%s/nav064_r12_%s/train_log.jsonl" % (R, run)
    out = []
    with open(p) as fh:
        for line in fh:
            if '"kind": "update"' in line:
                out.append(json.loads(line))
    return out


def win(rows, lo, hi, key):
    v = [r[key] for r in rows if lo <= r["update"] <= hi and r.get(key) is not None]
    return st.mean(v) if v else float("nan")


print("%-18s %-22s %8s %8s %8s %8s %8s" % (
    "arm", "run", "fwd@600", "fwd550+", "adv550+", "kept550+", "steer550+"))
for arm, run in RUNS:
    rows = load(run)
    last = rows[-1]
    print("%-18s %-22s %8.4f %8.4f %8.4f %8.4f %8.4f" % (
        arm, run,
        last.get("act_fwd_mean", float("nan")),
        win(rows, 550, 599, "act_fwd_mean"),
        win(rows, 550, 599, "adv_std"),
        win(rows, 550, 599, "policy_mask_kept_fraction"),
        win(rows, 550, 599, "act_steer_absmean"),
    ))
