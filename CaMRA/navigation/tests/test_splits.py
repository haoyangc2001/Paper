"""Split manifest checks for NAV-020.

The manifest must be reproducible from the generator, the splits must be mutually
exclusive, every declared map must be admissible, and the final splits must not
have been evaluated yet.

Run:

    <frozen python> CaMRA/navigation/tests/test_splits.py \
        --json CaMRA/navigation/reports/split_manifest_check.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, os.path.join(NAV_ROOT, "scripts"))

from camra_calibration.power import PowerModel  # noqa: E402

from build_split_manifest import (  # noqa: E402
    DESIGN,
    FINAL_SPLITS,
    LEVELS,
    SPLIT_SPEC,
    byte_sha256,
    build_manifest,
    canonical,
)
from camra_navigation.costs import VARIANTS  # noqa: E402
from camra_navigation.envs.nominal import LABELS  # noqa: E402
from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

MANIFEST = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
REPORTS = os.path.join(NAV_ROOT, "reports")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--resolution", type=float, default=0.15)
    args = ap.parse_args()

    committed = json.load(open(MANIFEST, encoding="utf-8"))
    rebuilt = build_manifest()

    # --- 1. the file is exactly what the generator produces ----------------
    same = canonical(committed) == canonical(rebuilt)
    check(
        "manifest_matches_generator",
        same,
        {
            "committed_hash": committed["integrity"]["manifest_hash"],
            "rebuilt_hash": rebuilt["integrity"]["manifest_hash"],
            "identical": same,
            "note": "手改 JSON 会立刻暴露：清单是代码加设计文件的函数",
        },
    )

    # --- 2. hash rule reproduces -------------------------------------------
    payload = {k: v for k, v in committed.items() if k != "integrity"}
    recomputed = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    check(
        "manifest_hash_reproducible",
        recomputed == committed["integrity"]["manifest_hash"],
        {
            "recomputed": recomputed,
            "declared": committed["integrity"]["manifest_hash"],
            "rule": committed["integrity"]["hash_rule"],
        },
    )

    # --- 3. design binding -------------------------------------------------
    design = M.load_design(DESIGN)
    check(
        "design_binding_matches",
        committed["generator"]["design_sha256"] == byte_sha256(DESIGN)
        and committed["generator"]["design_version"] == str(design["version"]),
        {
            "declared_version": committed["generator"]["design_version"],
            "design_version": design["version"],
            "declared_sha256": committed["generator"]["design_sha256"][:16],
            "file_sha256": byte_sha256(DESIGN)[:16],
        },
    )

    # --- 4. mutual exclusivity --------------------------------------------
    pairs: dict[tuple[str, int], str] = {}
    dupes = []
    for split, body in committed["splits"].items():
        for m in body["maps"]:
            key = (m["level"], int(m["seed"]))
            if key in pairs:
                dupes.append({"pair": list(key), "splits": [pairs[key], split]})
            pairs[key] = split
    check(
        "pairs_unique_across_splits",
        not dupes and len(pairs) == committed["exclusivity"]["n_pairs"],
        {
            "n_pairs": len(pairs),
            "declared": committed["exclusivity"]["n_pairs"],
            "duplicates": dupes,
            "declared_holds": committed["exclusivity"]["holds"],
        },
    )

    # --- 5. declared sizes are the real sizes ------------------------------
    sizes = {name: body["n_maps"] for name, body in committed["splits"].items()}
    actual = {name: len(body["maps"]) for name, body in committed["splits"].items()}
    expect = {name: len(LEVELS) * len(spec["seeds"]) for name, spec in SPLIT_SPEC.items()}
    check(
        "split_sizes_match",
        sizes == actual == expect
        and sum(sizes.values()) == committed["exclusivity"]["n_maps_declared"],
        {"declared": sizes, "actual": actual, "expected": expect,
         "total": sum(sizes.values())},
    )

    # --- 6. every declared map rebuilds to the same identity --------------
    mismatched = []
    for split, body in committed["splits"].items():
        for m in body["maps"]:
            spec = M.build(m["level"], int(m["seed"]), design)
            if (
                spec.map_id != m["map_id"]
                or spec.geometry_hash() != m["geometry_hash"]
                or spec.design_version != committed["generator"]["design_version"]
            ):
                mismatched.append({"split": split, "map": m["map_id"]})
    check(
        "map_identity_reproducible",
        not mismatched,
        {
            "maps_checked": sum(sizes.values()),
            "mismatched": mismatched[:5],
            "note": "map_id、geometry_hash 与 design_version 必须与清单逐项一致",
        },
    )

    # --- 7. no two maps are the same instance -----------------------------
    geo = [m["geometry_hash"] for body in committed["splits"].values() for m in body["maps"]]
    check(
        "all_geometry_hashes_distinct",
        len(set(geo)) == len(geo),
        {"n_maps": len(geo), "n_distinct": len(set(geo))},
    )

    # --- 8. near-duplicate layouts, declared not hidden -------------------
    layouts = {lv: set() for lv in LEVELS}
    per_level = {lv: 0 for lv in LEVELS}
    for body in committed["splits"].values():
        for m in body["maps"]:
            layouts[m["level"]].add(m["layout_hash"])
            per_level[m["level"]] += 1
    declared = committed["near_duplicates"]["distinct_layouts_per_level"]
    check(
        "near_duplicate_layouts_declared",
        {lv: len(v) for lv, v in layouts.items()} == declared
        and all(declared[lv] == 1 for lv in LEVELS)
        and committed["near_duplicates"]["maps_per_level"] == per_level,
        {
            "distinct_layouts": {lv: len(v) for lv, v in layouts.items()},
            "declared": declared,
            "maps_per_level": per_level,
            "consequence": committed["near_duplicates"]["consequence"],
        },
    )

    # --- 9. seed blocks are disjoint and respected ------------------------
    blocks = {name: (min(spec["seeds"]), max(spec["seeds"])) for name, spec in SPLIT_SPEC.items()}
    overlaps = []
    names = list(blocks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = blocks[names[i]], blocks[names[j]]
            if not (a[1] < b[0] or b[1] < a[0]):
                overlaps.append([names[i], names[j]])
    used = [s for spec in SPLIT_SPEC.values() for s in spec["seeds"]]
    check(
        "seed_blocks_disjoint",
        not overlaps and len(set(used)) == len(used),
        {
            "blocks": {k: list(v) for k, v in blocks.items()},
            "overlaps": overlaps,
            "n_seeds": len(used),
            "n_distinct": len(set(used)),
        },
    )

    # --- 10. the selection set is not an evaluation set -------------------
    def keys_of(name: str) -> set:
        return {(m["level"], int(m["seed"])) for m in committed["splits"][name]["maps"]}

    sel = keys_of("budget_selection")
    finals = set().union(*(keys_of(s) for s in FINAL_SPLITS))
    dev = keys_of("development")
    train = keys_of("train")
    check(
        "budget_selection_disjoint_from_evaluation",
        not (sel & finals) and not (sel & dev) and not (sel & train),
        {
            "budget_selection": len(sel),
            "finals": len(finals),
            "development": len(dev),
            "overlaps": {
                "finals": sorted(sel & finals),
                "development": sorted(sel & dev),
                "train": sorted(sel & train),
            },
        },
    )

    # --- 11. every declared map is admissible -----------------------------
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    rejected = []
    times = []
    for split, body in committed["splits"].items():
        for m in body["maps"]:
            spec = M.build(m["level"], int(m["seed"]), design)
            rec = F.check_map(spec, geometry, model, resolution=args.resolution)
            if not rec["admitted"]:
                rejected.append({"split": split, "map": m["map_id"], "reasons": rec["reasons"]})
            else:
                times.append(rec["route"]["t_min_s"])
    check(
        "all_declared_maps_admissible",
        not rejected,
        {
            "maps_checked": len(times) + len(rejected),
            "rejected": rejected[:5],
            "t_min_range_s": [min(times), max(times)] if times else None,
            "deadline_s": 30.0,
            "resolution_m": args.resolution,
            "note": "准入是几何事实，在任何策略存在之前就已确定，因此不构成按结果挑图",
        },
    )

    # --- 12. finals not evaluated yet -------------------------------------
    final_files = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(REPORTS, "final_*"))
        if os.path.isfile(p)
    )
    check(
        "final_splits_not_evaluated",
        committed["locks"]["final_evaluated"] is False and not final_files,
        {
            "final_evaluated_flag": committed["locks"]["final_evaluated"],
            "final_result_files": final_files,
            "unlock_condition": committed["locks"]["unlock_condition"],
        },
    )

    # --- 13. method pairing covers the frozen variants --------------------
    check(
        "method_pairing_covers_variants",
        sorted(committed["method_pairing"]["variants"]) == sorted(VARIANTS),
        {
            "declared": committed["method_pairing"]["variants"],
            "frozen_variants": list(VARIANTS),
            "z0_owner": committed["method_pairing"]["z0_owner"],
        },
    )

    # --- 14. failure handling covers every label --------------------------
    fh = committed["failure_handling"]
    covered = set(fh["counted_as_success"]) | set(fh["counted_as_failure"]) | set(
        fh["counted_separately"]
    ) | set(fh["not_a_label"])
    check(
        "failure_handling_covers_all_labels",
        set(LABELS) <= covered and "internal_violation" in fh["not_a_label"],
        {
            "labels": list(LABELS),
            "covered": sorted(covered),
            "missing": sorted(set(LABELS) - covered),
            "no_resampling": fh["no_resampling"],
        },
    )

    # --- 15. rebuild is stable --------------------------------------------
    again = build_manifest()
    check(
        "rebuild_is_stable",
        canonical(again) == canonical(rebuilt),
        {
            "first": rebuilt["integrity"]["manifest_hash"],
            "second": again["integrity"]["manifest_hash"],
            "note": "同一代码同一设计文件两次生成必须逐字节相同",
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-020",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "manifest_hash": committed["integrity"]["manifest_hash"],
        "n_maps": committed["exclusivity"]["n_maps_declared"],
        "checks": RESULTS,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:400])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh2:
            json.dump(report, fh2, ensure_ascii=False, indent=2)
            fh2.write("\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
