"""Expand explicit human-readable rows; never sample or infer an error value."""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

from compute_statistics import ROOT, read_rows, validate_rows, summarize, short_summary


def ingest(source, split, start):
    source = source.resolve()
    manifest = {r["pair_id"]:r for r in csv.DictReader((ROOT/"data/split_manifest.csv").open(encoding="utf-8"))}
    prefix = "CAL" if split=="calibration" else "TEST"
    if split=="held_out" and not (ROOT/"data/calibration_freeze.json").exists():
        raise ValueError("Freeze calibration before authoring held-out records")
    draft = ROOT/"data"/("calibration_draft.csv" if split=="calibration" else "held_out_draft.csv")
    old = read_rows(draft) if draft.exists() else []
    if len(old) != start-1:
        raise ValueError(f"Expected {start-1} existing rows; got {len(old)}")
    records, reviews, checkpoints = [], [], []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        if line.startswith("#"):
            checkpoints.append({"after_rows":len(records),"assessment":line.lstrip("# ")})
            continue
        values = line.split("|")
        if len(values)!=5:
            raise ValueError(f"{source}:{line_no}: expected 5 authored fields")
        j,percent,mm,cost_note,geom_note = values
        pair_id = f"{prefix}-{start+len(records):04d}"
        m = manifest[pair_id]
        overrun = Decimal(percent)/100
        deviation = Decimal(mm)/1000
        row = {**m,"controller_id":"assumed_frozen_velocity_tracker_v1","J_nom":float(Decimal(j)),"J_exec":float(Decimal(j)*(1+overrun)),"cost_overrun":float(overrun),"max_deviation_m":float(deviation),"data_status":"hypothetical"}
        row["duration_s"] = float(row["duration_s"])
        related = [r["pair_id"] for r in old+records if r["condition_id"]==m["condition_id"]][-4:]
        related += [r["pair_id"] for r in (old+records)[-5:] if r["scenario_id"]==m["scenario_id"]]
        review = {"pair_id":pair_id,"revision":1,"source_file":str(source.relative_to(ROOT)),"source_line":line_no,"condition_id":m["condition_id"],"values":{k:row[k] for k in ["J_nom","cost_overrun","max_deviation_m"]},"cost_assessment":cost_note,"geometry_assessment":geom_note,"related_ids":list(dict.fromkeys(related)),"decision":"accept as explicitly hypothetical case","data_status":"hypothetical"}
        records.append(row)
        reviews.append(review)
    all_rows = old+records
    errors = validate_rows(all_rows)
    if errors:
        raise ValueError(errors)
    if any(len(r[key])<4 for r in reviews for key in ["cost_assessment","geometry_assessment"]):
        raise ValueError("Missing individually authored assessment")
    expected_checkpoints = set(range(20,len(records)+1,20))
    if not expected_checkpoints.issubset({c["after_rows"] for c in checkpoints}):
        raise ValueError("Missing 20-row authored review checkpoints")
    with draft.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    with (ROOT/"reviews/point_reviews.jsonl").open("a",encoding="utf-8") as f:
        for review in reviews:
            f.write(json.dumps(review,ensure_ascii=False)+"\n")
    report={"source":str(source.relative_to(ROOT)),"split":split,"start":start,"end":start+len(records)-1,"n_added":len(records),"authored_checkpoints":checkpoints,"checks":errors,"cumulative":short_summary(summarize(all_rows)),"timestamp":datetime.now(timezone.utc).isoformat()}
    out=ROOT/"reviews/batches"/(source.stem+".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report["cumulative"],ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("--split",choices=["calibration","held_out"],required=True)
    parser.add_argument("--start",type=int,required=True)
    args=parser.parse_args()
    ingest(args.source,args.split,args.start)
