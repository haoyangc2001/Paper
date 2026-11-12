"""Assemble reports/training_round_r1.json + the cumulative rounds ledger (NAV-064).

Input: the JSON written by scripts/eval_training_round.py.  This script only adds
the parts the evaluator cannot know: the attempt history (crashed rounds), the
mechanism statistics read from the training logs, and the round decision.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

NAV = os.getcwd()
WORK = "/pub/data/caohy/work/camra_navigation"


def load_log(run_id: str) -> list[dict]:
    p = os.path.join(WORK, run_id, "train_log.jsonl")
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def mechanism(run_ids: list[str]) -> dict:
    out = {}
    for rid in run_ids:
        rows = load_log(rid)
        ups = [r for r in rows if r.get("kind") == "update"]
        end = next((r for r in rows if r.get("kind") == "run_end"), {})
        start = next((r for r in rows if r.get("kind") == "run_start"), {})
        series = lambda k: [u[k] for u in ups if k in u]
        out[rid] = {
            "updates": len(ups),
            "wall_hours": end.get("wall_hours"),
            "budget_rule": start.get("budget_rule"),
            "obs_version": start.get("obs_version"),
            "schema": start.get("schema"),
            "clip_fraction_mean": round(statistics.fmean(series("clipped_ratio_fraction")), 4)
            if series("clipped_ratio_fraction") else None,
            "clip_fraction_last50": round(statistics.fmean(series("clipped_ratio_fraction")[-50:]), 4)
            if series("clipped_ratio_fraction") else None,
            "entropy_first10": round(statistics.fmean(series("entropy")[:10]), 3)
            if series("entropy") else None,
            "entropy_last10": round(statistics.fmean(series("entropy")[-10:]), 3)
            if series("entropy") else None,
            "grad_steps_total": end.get("updates") and None,
            "label_counts": end.get("episode_label_counts"),
        }
    return out


def main() -> int:
    eval_path = os.path.join(NAV, "reports", "training_round_r1.json")
    ev = json.load(open(eval_path, encoding="utf-8"))
    run_ids = [r["run_id"] for r in ev["runs"]]
    attempts = [
        {"run_prefix": "nav064_r1_", "outcome": "crashed",
         "cause": "storage guard rejected in-segment overspend |z_norm|=1.53 (NAV-062 defect)",
         "updates_before_crash": "8-21", "kept": True},
        {"run_prefix": "nav064_r1b_", "outcome": "crashed/stopped",
         "cause": "same guard, second form: z[0] of a window-sliced segment is already negative (-1.62)",
         "updates_before_crash": "8-78", "kept": True},
        {"run_prefix": "nav064_r1c_", "outcome": "completed", "cause": "-", "kept": True},
    ]
    mech = mechanism(run_ids)

    report = dict(ev)
    report.update({
        "attempt_history": attempts,
        "mechanism": mech,
        "crashes": {
            "n": 2,
            "kind": "implementation defect (not business failure)",
            "fix": "fe24047: in-segment overspend is not an error; initial-budget range enforced in BudgetSampler; obj_z_saturated records saturation",
            "re_run_rule": "new run ids (r1c), attempts kept",
        },
        "metric_note": (
            "SR = label == goal_reached（与 G4 同定义）；sr_within_budget 是『到达且 z_end>=0』。"
            "预算可行性本体属于 closing_tier 的第二条件，本循环只在 development 上做预算响应诊断，"
            "不在 NAV-064 使用 budget_selection（acceptance 明确禁止）"
        ),
    })

    out = os.path.join(NAV, "reports", "training_round_r1.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)

    # cumulative ledger
    ledger_path = os.path.join(NAV, "reports", "training_loop_rounds.json")
    if os.path.exists(ledger_path):
        ledger = json.load(open(ledger_path, encoding="utf-8"))
    else:
        ledger = {"schema": "camra-training-loop-ledger-v1", "task": "NAV-064",
                  "protocol": "config/training_loop_protocol_v1_1.json", "rounds": []}
    ledger["rounds"] = [r for r in ledger["rounds"] if r.get("round") != report["round"]]
    ledger["rounds"].append({
        "round": report["round"],
        "prereg": report["prereg"],
        "closing_tier": report["closing_tier"]["by_method"],
        "closing_tier_met": report["closing_tier"]["closing_tier_met"],
        "report": "reports/training_round_r1.json",
        "attempts": [a["run_prefix"] + a["outcome"] for a in attempts],
    })
    ledger["rounds"].sort(key=lambda r: r["round"])
    with open(ledger_path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"round": report["round"],
                      "closing_tier_met": report["closing_tier"]["closing_tier_met"],
                      "by_method": report["closing_tier"]["by_method"],
                      "written": [out, ledger_path]}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
