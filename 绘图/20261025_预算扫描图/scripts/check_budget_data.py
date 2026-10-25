"""Deterministic checks; does not infer, smooth, or generate observations."""
from pathlib import Path
import argparse
import copy
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_points(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return read_json(path)


def feasible(row, protocol):
    t, eps = protocol["thresholds"], protocol["tolerance"]
    return (row["SR"] >= t["SR_min"]-eps and row["CR"] <= t["CR_max"]+eps
            and row["BVR"] <= t["BVR_max"]+eps)


def check(points, protocol):
    if protocol["data_status"] != "hypothetical":
        raise ValueError("This protocol only describes the explicitly assumed draft data")
    ids = [p["point_id"] for p in points]
    assert len(ids) == 96 and len(set(ids)) == 96, "96 unique IDs required"
    expected = [(b,m,k) for b in protocol["budget_ratios"] for m in protocol["methods"] for k in protocol["metrics"]]
    actual = [(p["budget_ratio"],p["method"],p["metric"]) for p in points]
    assert actual == expected, "Missing, reordered or duplicated data key"
    assert ids == [f"B{i:03d}" for i in range(1,97)], "ID mapping changed"
    eps = protocol["tolerance"]
    lookup = {}
    for p in points:
        v = p["value"]
        assert math.isfinite(v) and v >= 0, "Negative or nonfinite value"
        assert p["unit"] == ("draft_cost_units" if p["metric"] == "Cost" else "fraction"), "Unit mismatch"
        if p["metric"] != "Cost":
            assert v <= 1, "Rate must be a fraction"
        lookup[p["budget_ratio"],p["method"],p["metric"]] = v
    selected = {}
    for m in protocol["methods"]:
        for b in protocol["budget_ratios"]:
            row = {k:lookup[b,m,k] for k in protocol["metrics"]}
            assert row["SR"]+row["CR"] <= 1+eps, "SR/CR incompatible with collision-free success"
            if feasible(row,protocol) and m not in selected:
                selected[m] = {"budget_ratio":b, **row}
    for k in ["SR","Cost","CR"]:
        vals = [lookup[b,"RA",k] for b in protocol["budget_ratios"]]
        assert max(vals)-min(vals) <= eps, "Fixed RA trajectory metric drift"
    vals = [lookup[b,"RA","BVR"] for b in protocol["budget_ratios"]]
    assert all(y <= x+eps for x,y in zip(vals,vals[1:])), "RA exceedance must not rise with threshold"
    for a in protocol["reference_anchors"]:
        assert abs(lookup[1,a["method"],a["metric"]]-a["mean"]) <= eps, "Reference anchor mismatch"
    return {"passed":True,"points":96,"reference_anchors":12,"minimum_feasible":selected,
            "scope":"Internal consistency of assumed aggregate values; not empirical validation"}


def review_check(points, records, required=96):
    latest = {}
    for r in records:
        old = latest.get(r["point_id"])
        assert r["revision"] == (old["revision"]+1 if old else 1)
        assert r["supersedes_revision"] == (old["revision"] if old else None)
        latest[r["point_id"]] = r
    assert len(latest) == required
    for p in points[:required]:
        r = latest[p["point_id"]]
        for key in ["budget_ratio","method","metric","source_file","source_sheet","source_cell","unit"]:
            assert r[key] == p[key]
        assert r["original_value"] == p["value"]
        assert r["decision"] in ["retain","deterministic_correction","user_confirmed_revision"]
        assert r["adopted_value"] is not None and len(r["reasoning"]) > 35
        assert r["related_point_ids"] and r["decision_basis"]
    return {"reviewed_unique_points":required,"records":len(records),"passed":True}


def boundary_checks(points, protocol):
    result = []
    mutations = [
        ("duplicate_id", lambda a:a[1].update(point_id=a[0]["point_id"])),
        ("percent_as_fraction", lambda a:a[0].update(value=60.8)),
        ("RA_metric_drift", lambda a:a[8].update(value=.601)),
        ("anchor_mismatch", lambda a:a[49].update(value=88.4)),
        ("RA_BVR_increase", lambda a:a[22].update(value=.36)),
    ]
    for name, mutate in mutations:
        a = copy.deepcopy(points); mutate(a)
        try:
            check(a,protocol)
        except (AssertionError, ValueError):
            result.append({"case":name,"detected":True})
        else:
            raise AssertionError(name+" was not detected")
    assert feasible({"SR":.60,"CR":.20,"BVR":.15},protocol)
    assert not feasible({"SR":.59,"CR":.19,"BVR":.14},protocol)
    assert not feasible({"SR":.61,"CR":.21,"BVR":.14},protocol)
    p = copy.deepcopy(protocol); p["thresholds"]["BVR_max"] = 0
    assert not check(points,p)["minimum_feasible"]
    result.extend({"case":n,"detected":True} for n in ["inclusive_boundary","SR_fails_despite_BVR_pass","CR_fails_despite_BVR_pass","no_feasible_candidate"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=ROOT/"data/baseline_points.json")
    ap.add_argument("--protocol",type=Path,default=ROOT/"data/review_protocol.json")
    ap.add_argument("--review-count",type=int)
    ap.add_argument("--self-check",action="store_true")
    args = ap.parse_args()
    points, protocol = read_points(args.data), read_json(args.protocol)
    out = check(points,protocol)
    if args.self_check:
        out["boundary_checks"] = boundary_checks(points,protocol)
    if args.review_count is not None:
        out["review"] = review_check(points,read_points(ROOT/"reviews/point_review.jsonl"),args.review_count)
    print(json.dumps(out,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
