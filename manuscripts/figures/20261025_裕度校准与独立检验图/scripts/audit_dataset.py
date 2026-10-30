"""Audit authored data and publish reviewed rows without generating samples."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from compute_statistics import ROOT, read_rows, summarize, short_summary, quantile
from validate_data import validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_check():
    freeze = json.loads((ROOT / "data/calibration_freeze.json").read_text(encoding="utf-8"))
    return {
        "calibration_unchanged": digest(ROOT / "data/calibration_frozen.csv") == freeze["calibration_sha256"],
        "protocol_unchanged": digest(ROOT / "data/protocol.json") == freeze["protocol_sha256"],
        "held_out_absent_at_freeze": freeze["held_out_exists"] is False,
    }


def limits_from_calibration(cal):
    return {k: quantile([r[k] for r in cal]) for k in ("cost_overrun", "max_deviation_m")}


def correlation(rows):
    if len(rows) < 2:
        return None
    x = [r["cost_overrun"] for r in rows]
    y = [r["max_deviation_m"] for r in rows]
    return statistics.correlation(x, y) if statistics.pstdev(x)*statistics.pstdev(y) else None


def held_out(cal, test):
    audit = validate(ROOT / "data/held_out_draft.csv")
    audit["freeze"] = freeze_check()
    audit["condition_counts"] = dict(Counter(r["condition_id"] for r in test))
    audit["complete"] = len(test) == 400 and len(audit["condition_counts"]) == 20 and set(audit["condition_counts"].values()) == {20}
    stats = summarize(test, limits_from_calibration(cal))
    audit["percentage_step"] = 100 / len(test)
    audit["coverage_integer_counts"] = {k: {"covered": v["covered_n"], "total": len(test), "percentage": v["coverage"]*100} for k, v in stats["metrics"].items()}
    assert audit["valid"] and audit["complete"] and all(audit["freeze"].values()), audit
    save(ROOT / "data/held_out_statistics.json", stats)
    save(ROOT / "reviews/held_out_final_audit.json", audit)
    print(json.dumps(short_summary(stats)))


def global_audit(cal, test):
    rows = cal + test
    reviews = [json.loads(line) for line in (ROOT / "reviews/point_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    review_map = {r["pair_id"]: r for r in reviews}
    source_errors = []
    for row in rows:
        r = review_map[row["pair_id"]]
        line = (ROOT / r["source_file"]).read_text(encoding="utf-8").splitlines()[r["source_line"]-1]
        j, cost_pct, geom_mm, cost_note, geom_note = line.split("|")
        if not all((math.isclose(float(j), row["J_nom"]), math.isclose(float(cost_pct)/100, row["cost_overrun"]), math.isclose(float(geom_mm)/1000, row["max_deviation_m"]), cost_note == r["cost_assessment"], geom_note == r["geometry_assessment"])):
            source_errors.append(row["pair_id"])
    fingerprints = defaultdict(list)
    numeric = defaultdict(list)
    for r in rows:
        fingerprints[(r["condition_id"], r["J_nom"], r["J_exec"], r["max_deviation_m"])].append(r["pair_id"])
        numeric[(r["cost_overrun"], r["max_deviation_m"])].append(r["pair_id"])
    thresholds = limits_from_calibration(cal)
    per_condition = {}
    for split, group in (("calibration", cal), ("held_out", test)):
        per_condition[split] = {cond: {**short_summary(summarize([r for r in group if r["condition_id"] == cond], thresholds)), "cost_geometry_correlation": correlation([r for r in group if r["condition_id"] == cond])} for cond in sorted({r["condition_id"] for r in group})}
    batches = {f"{split}_{i+1:04d}_{min(i+100,len(group)):04d}": short_summary(summarize(group[i:i+100], thresholds)) for split, group in (("calibration", cal), ("held_out", test)) for i in range(0, len(group), 100)}
    audit = {
        "data_status": "hypothetical", "records": len(rows), "error_values": len(rows)*2,
        "reviews": len(reviews), "unique_reviewed_ids": len(review_map), "authored_source_mismatches": source_errors,
        "checks": {"calibration": validate(ROOT / "data/calibration_frozen.csv"), "held_out": validate(ROOT / "data/held_out_draft.csv")},
        "freeze": freeze_check(), "identical_numeric_condition_tuples": [ids for ids in fingerprints.values() if len(ids)>1],
        "duplicate_command_ids": [key for key, n in Counter(r["command_id"] for r in rows).items() if n > 1],
        "tied_error_pairs": [ids for ids in numeric.values() if len(ids)>1],
        "unique_cost_assessments": len({r["cost_assessment"] for r in reviews}),
        "unique_geometry_assessments": len({r["geometry_assessment"] for r in reviews}),
        "overall_cost_geometry_correlation": correlation(rows), "conditions": per_condition, "batches": batches,
        "semantics": ["Storage order cycles conditions; not a physical time series.", "Numeric ties are allowed; a tied pair is not an extra observed rollout.", "Population quantiles do not follow from single trajectory examples.", "Marginal coverage differs from joint coverage and SR/CR/BVR.", "Illustrative absolute costs are not main-table Cost units.", "Held-out hypothetical construction is not independent empirical validation."]
    }
    assert len(rows) == len(review_map) == 1200 and len(reviews) == 1200
    assert not source_errors and all(audit["freeze"].values()) and all(v["valid"] for v in audit["checks"].values())
    save(ROOT / "reviews/cross_figure_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("conditions", "batches", "tied_error_pairs")}, ensure_ascii=False))


def release(cal, test):
    path = ROOT / "data/paired_rollouts.csv"
    if path.exists():
        raise FileExistsError("Canonical release already exists; use a new version for changed records")
    rows = cal + test
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = validate(path, final=True)
    assert audit["valid"], audit
    audit.update({"source_sha256": digest(path), "frozen_calibration_unchanged": freeze_check(), "release_method": "Concatenate frozen calibration and reviewed held-out; no new samples."})
    save(ROOT / "reviews/data_release_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("held-out", "global", "release"))
    args = parser.parse_args()
    calibration = read_rows(ROOT / "data/calibration_frozen.csv")
    held = read_rows(ROOT / "data/held_out_draft.csv")
    {"held-out": held_out, "global": global_audit, "release": release}[args.phase](calibration, held)
