"""Deterministic statistics from explicitly authored paired-rollout records."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ("duration_s", "J_nom", "J_exec", "cost_overrun", "max_deviation_m")


def read_rows(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    for row in rows:
        for name in NUMERIC:
            row[name] = float(row[name])
    return rows


def quantile(values, p=0.9):
    if not values or not 0 < p <= 1:
        raise ValueError("nonempty sample and 0<p<=1 required")
    return sorted(values)[math.ceil(len(values) * p) - 1]


def ecdf(values):
    counts = Counter(values)
    total = 0
    result = []
    for value in sorted(counts):
        total += counts[value]
        result.append({"x": value, "count_le": total, "fraction": total / len(values)})
    return result


def validate_rows(rows, expected_n=None):
    errors = []
    if not rows:
        errors.append("empty input")
    if expected_n is not None and len(rows) != expected_n:
        errors.append(f"count {len(rows)} != {expected_n}")
    ids = [r["pair_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate pair IDs")
    for r in rows:
        rid = r["pair_id"]
        if r["split"] not in ("calibration", "held_out"):
            errors.append(rid + ": invalid split")
        if any(not math.isfinite(float(r[n])) for n in NUMERIC):
            errors.append(rid + ": nonfinite numeric")
        if r["J_nom"] <= 0 or r["J_exec"] <= 0:
            errors.append(rid + ": costs must be positive")
        elif not math.isclose((r["J_exec"]-r["J_nom"])/r["J_nom"], r["cost_overrun"], rel_tol=1e-10, abs_tol=1e-10):
            errors.append(rid + ": cost equation mismatch")
        if r["duration_s"] <= 0 or r["max_deviation_m"] < 0:
            errors.append(rid + ": invalid duration or geometry")
        if r["data_status"] != "hypothetical":
            errors.append(rid + ": this illustration requires hypothetical provenance")
    return errors


def summarize(rows, thresholds=None):
    thresholds = thresholds or {"cost_overrun": 0.18, "max_deviation_m": 0.12}
    output = {"n": len(rows), "data_status": "hypothetical", "metrics": {}, "conditions": dict(Counter(r["condition_id"] for r in rows))}
    for metric, limit in thresholds.items():
        v = [r[metric] for r in rows]
        hits = sum(x <= limit for x in v)
        output["metrics"][metric] = {"min": min(v), "max": max(v), "median": statistics.median(v), "q90": quantile(v), "threshold": limit, "covered_n": hits, "coverage": hits/len(v), "exceed_n": len(v)-hits, "negative_n": sum(x < 0 for x in v), "equal_threshold_n": sum(x == limit for x in v), "ecdf": ecdf(v)}
    joint = sum(all(r[k] <= t for k,t in thresholds.items()) for r in rows)
    output["joint_covered_n"] = joint
    output["joint_coverage"] = joint/len(rows)
    output["joint_note"] = "Internal consistency only; not a navigation success or safety guarantee."
    return output


def short_summary(stats):
    return {"n": stats["n"], "metrics": {k: {a:b for a,b in v.items() if a != "ecdf"} for k,v in stats["metrics"].items()}, "joint_covered_n": stats["joint_covered_n"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"data/paired_rollouts.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"data/statistics.json")
    args = parser.parse_args()
    rows = read_rows(args.data)
    errors = validate_rows(rows)
    if errors:
        raise ValueError(errors)
    calibration = [r for r in rows if r["split"] == "calibration"]
    limits = {key: quantile([r[key] for r in calibration]) for key in ("cost_overrun", "max_deviation_m")}
    result = {"data_status": "hypothetical", "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "quantile_method": "inverted_cdf", "thresholds": limits, "groups": {s: summarize([r for r in rows if r["split"] == s], limits) for s in ("calibration", "held_out") if any(r["split"] == s for r in rows)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s:short_summary(v) for s,v in result["groups"].items()}))


if __name__ == "__main__":
    main()
