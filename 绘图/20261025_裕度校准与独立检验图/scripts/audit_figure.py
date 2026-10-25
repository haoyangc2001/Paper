"""Independently check ECDF render coordinates against every source value."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from compute_statistics import ROOT, read_rows


def audit(folder):
    m = json.loads((folder / "render_manifest.json").read_text(encoding="utf-8"))
    source = Path(m["data_path"])
    rows = read_rows(source)
    errors = []
    checks = []
    if hashlib.sha256(source.read_bytes()).hexdigest() != m["data_sha256"]:
        errors.append("input hash differs")
    for panel in m["panels"]:
        metric, scale = panel["metric"], panel["scale"]
        calibration = sorted(r[metric] for r in rows if r["split"] == "calibration")
        threshold = calibration[math.ceil(.9*len(calibration))-1]
        if not math.isclose(threshold*scale, panel["threshold"]):
            errors.append(metric + ": threshold not calibration Q90")
        for split, curve in panel["curves"].items():
            values = [r[metric] for r in rows if r["split"] == split]
            xs = sorted(set(values))
            expected_x = [panel["xlim"][0]] + [x*scale for x in xs] + [panel["xlim"][1]]
            expected_y = [0] + [100*sum(v <= x for v in values)/len(values) for x in xs] + [100]
            if expected_x != curve["x"] or len(expected_y) != len(curve["percent"]) or any(not math.isclose(a, b, abs_tol=1e-10) for a,b in zip(expected_y,curve["percent"])):
                errors.append(metric + "/" + split + ": ECDF mismatch")
            count = sum(v <= threshold for v in values)
            if curve["covered_n"] != count or not math.isclose(curve["threshold_point"][1], 100*count/len(values)):
                errors.append(metric + "/" + split + ": marker/count mismatch")
            if not (panel["xlim"][0] <= min(values)*scale and max(values)*scale <= panel["xlim"][1]):
                errors.append(metric + ": cropped data")
            checks.append({"metric":metric, "split":split, "records_checked":len(values), "distinct_steps":len(xs), "covered":count, "coverage_percent":100*count/len(values)})
    outside = []
    for item in m["text_bounds"]:
        x,y,w,h = item["bounds_fraction"]
        if min(x,y) < -0.005 or x+w > 1.005 or y+h > 1.005:
            outside.append(item)
    if outside:
        errors.append("Text extends beyond figure bounds")
    if min(t["font_pt"] for t in m["text_bounds"]) < 6.5:
        errors.append("Font smaller than 6.5pt")
    report = {"valid":not errors,"errors":errors,"mapping_checks":checks,"out_of_bounds_text":outside,"width_mm":m["width_mm"],"height_mm":m["height_mm"],"source_sha256":m["data_sha256"]}
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.folder)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result))
    if not result["valid"]:
        raise SystemExit(1)
