"""Audit records, split isolation, authored reviews, and the frozen calibration."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

from compute_statistics import ROOT, read_rows, validate_rows, quantile


def validate(path, final=False):
    rows = read_rows(path)
    errors = validate_rows(rows, 1200 if final else None)
    manifest = {r["pair_id"]:r for r in csv.DictReader((ROOT/"data/split_manifest.csv").open(encoding="utf-8"))}
    reviews_path = ROOT/"reviews/point_reviews.jsonl"
    reviews = [json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines()] if reviews_path.exists() else []
    latest = {r["pair_id"]:r for r in reviews}
    for row in rows:
        rid = row["pair_id"]
        if rid not in manifest:
            errors.append(rid + ": missing manifest")
            continue
        for field in ("split", "condition_id", "scenario_id", "perturbation_profile_id", "command_id"):
            if row[field] != manifest[rid][field]:
                errors.append(rid + ": manifest mismatch " + field)
        if row["duration_s"] != float(manifest[rid]["duration_s"]):
            errors.append(rid + ": duration mismatch")
        if rid not in latest:
            errors.append(rid + ": missing authored assessment")
        else:
            review = latest[rid]
            for field in ("cost_assessment", "geometry_assessment"):
                if len(review[field]) < 4:
                    errors.append(rid + ": insufficient " + field)
            for key in ("J_nom", "cost_overrun", "max_deviation_m"):
                if review["values"][key] != row[key]:
                    errors.append(rid + ": review value mismatch " + key)
    if final:
        counts = Counter(r["split"] for r in rows)
        if counts != {"calibration":800,"held_out":400}:
            errors.append("final 800/400 mismatch")
        for split, expected in [("calibration",40),("held_out",20)]:
            tally = Counter(r["condition_id"] for r in rows if r["split"]==split)
            if len(tally)!=20 or any(n != expected for n in tally.values()):
                errors.append("condition count mismatch " + split)
        frozen = ROOT/"data/calibration_frozen.csv"
        freeze = json.loads((ROOT/"data/calibration_freeze.json").read_text(encoding="utf-8"))
        if hashlib.sha256(frozen.read_bytes()).hexdigest()!=freeze["calibration_sha256"]:
            errors.append("calibration freeze hash mismatch")
        if [r for r in rows if r["split"]=="calibration"] != read_rows(frozen):
            errors.append("final calibration rows differ from frozen")
        if hashlib.sha256((ROOT/"data/protocol.json").read_bytes()).hexdigest()!=freeze["protocol_sha256"]:
            errors.append("protocol changed after freeze")
    return {"valid": not errors, "rows": len(rows), "errors": errors, "reviewed_rows": sum(r["pair_id"] in latest for r in rows)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"data/paired_rollouts.csv")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.data,args.final)
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
