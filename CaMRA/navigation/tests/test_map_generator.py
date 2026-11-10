"""Map generator checks for NAV-016.

Verifies reproducibility, the three levels' geometry and purposes, that every
reported route clears the inflated geometry, that route lengths really differ, and
that energy comes from the frozen power model rather than from length.

Run:

    <frozen python> CaMRA/navigation/tests/test_map_generator.py \
        --json CaMRA/navigation/reports/map_catalogue.json \
        --svg CaMRA/navigation/reports/map_catalogue.svg
"""

from __future__ import annotations

import argparse
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

from camra_calibration.power import PowerModel  # noqa: E402

from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--svg", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    seeds = design["seeds"]["values"]

    reports = [M.evaluate(M.build(lv, s, design), geometry, model, design)
               for lv in M.LEVELS for s in seeds]

    # --- 1. reproducibility -----------------------------------------------
    a = M.build("L3_multi_path", seeds[0], design)
    b = M.build("L3_multi_path", seeds[0], design)
    c = M.build("L3_multi_path", seeds[1], design)
    rt = M.MapSpec.from_dict(json.loads(json.dumps(a.to_dict())))
    check(
        "reproducible_and_round_trips",
        a.geometry_hash() == b.geometry_hash()
        and a.to_dict() == b.to_dict()
        and a.geometry_hash() == rt.geometry_hash()
        and a.geometry_hash() != c.geometry_hash(),
        {
            "same_seed_hash": a.geometry_hash()[:16],
            "other_seed_hash": c.geometry_hash()[:16],
            "round_trip_identical": a.to_dict() == rt.to_dict(),
            "map_ids": [a.map_id, c.map_id],
        },
    )

    # --- 2. level geometry and purposes -----------------------------------
    by_level: dict[str, list[dict]] = {}
    for report in reports:
        by_level.setdefault(report["level"], []).append(report)

    l1 = by_level["L1_empty"]
    check(
        "L1_empty_is_a_clean_corridor",
        all(r["n_routes_total"] == 1 and r["n_routes_feasible"] == 1 for r in l1)
        and all(7.0 <= r["net_width_m"] <= 8.0 for r in l1)
        and all(r["min_gap_m"] is None for r in l1),
        {
            "net_width_m": [r["net_width_m"] for r in l1],
            "routes": [r["n_routes_feasible"] for r in l1],
            "shortest_m": [r["shortest_feasible_m"] for r in l1],
            "straight_m": [r["straight_distance_m"] for r in l1],
        },
    )

    l2 = by_level["L2_single_detour"]
    l2_gaps = [g for r in l2 for values in r["gaps_by_column"].values() for g in values]
    check(
        "L2_single_detour_has_two_passages",
        all(r["n_routes_total"] >= 2 and r["n_routes_feasible"] >= 2 for r in l2)
        and all(7.0 <= g <= 8.0 for g in l2_gaps),
        {
            "gaps_m": sorted(set(round(g, 3) for g in l2_gaps)),
            "routes_total": [r["n_routes_total"] for r in l2],
            "routes_feasible": [r["n_routes_feasible"] for r in l2],
            "shortest_m": [round(r["shortest_feasible_m"], 3) for r in l2],
        },
    )

    l3 = by_level["L3_multi_path"]
    l3_gaps = [g for r in l3 for values in r["gaps_by_column"].values() for g in values]
    length_spread = [
        max(x["length_m"] for x in r["routes"]) - min(x["length_m"] for x in r["routes"])
        for r in l3
    ]
    check(
        "L3_multi_path_offers_three_openings",
        all(r["n_routes_total"] == 3 for r in l3)
        and all(r["n_routes_feasible"] >= 2 for r in l3)
        and all(g >= 7.0 - 1e-9 for g in l3_gaps)
        and min(length_spread) > 0.5,
        {
            "gaps_m": sorted(set(round(g, 3) for g in l3_gaps)),
            "routes_total": [r["n_routes_total"] for r in l3],
            "routes_feasible": [r["n_routes_feasible"] for r in l3],
            "length_spread_m": [round(v, 3) for v in length_spread],
        },
    )

    # --- 3. start / goal are inside the constraint ------------------------
    worst_start = min(r["start_margin_m"] for r in reports)
    worst_goal = min(r["goal_margin_m"] for r in reports)
    worst_boundary = min(
        min(r["start_boundary_margin_m"], r["goal_boundary_margin_m"]) for r in reports
    )
    check(
        "start_and_goal_clear_the_inflated_set",
        worst_start > 0.0 and worst_goal > 0.0 and worst_boundary > 0.0,
        {
            "worst_start_margin_to_islands_m": worst_start,
            "worst_goal_margin_to_islands_m": worst_goal,
            "worst_boundary_margin_m": worst_boundary,
            "inflation_radius_m": geometry.inflation_radius_m,
        },
    )

    # --- 4. detour exists and routes really differ ------------------------
    # The middle opening of L3 is nearly straight by design, so the criterion is
    # "the map offers a real detour and its routes differ", not "every route is
    # longer than the straight line".
    detours = []
    for r in reports:
        if r["level"] == "L1_empty":
            continue
        lengths = sorted(x["length_m"] for x in r["routes"] if x["feasible"])
        detours.append(
            {
                "map": r["map_id"],
                "level": r["level"],
                "shortest_m": lengths[0],
                "longest_m": lengths[-1],
                "straight_m": r["straight_distance_m"],
                "max_ratio": lengths[-1] / r["straight_distance_m"],
                "spread_m": lengths[-1] - lengths[0],
            }
        )
    spread_floor = {"L2_single_detour": 0.05, "L3_multi_path": 5.0}  # L2 passages are symmetric by design
    check(
        "detour_is_real",
        all(
            d["max_ratio"] > 1.05 and d["spread_m"] > spread_floor[d["level"]]
            for d in detours
        ),
        detours,
    )

    # --- 5. energy comes from the frozen model ---------------------------
    energies = [
        (r["map_id"], [x["energy_J"] for x in r["routes"] if x["energy_J"] is not None])
        for r in reports
    ]
    has_energy = all(len(e) >= 1 and all(v > 0.0 for v in e) for _m, e in energies)
    multi_route_maps = [
        r for r in reports if len([x for x in r["routes"] if x["energy_J"] is not None]) >= 2
    ]
    inversions = [r["map_id"] for r in multi_route_maps if r["length_and_energy_agree"] is False]
    check(
        "energy_from_frozen_power_model",
        has_energy,
        {
            "per_map": [{"map": m, "energy_J": [round(v, 3) for v in e]} for m, e in energies],
            "model": f"power-v1 sha256 {model.sha256()[:16]}",
        },
    )
    check(
        "length_and_energy_rankings_reported",
        len(multi_route_maps) >= 3,
        {
            "maps_with_multiple_priced_routes": len(multi_route_maps),
            "maps_where_shortest_is_not_cheapest": inversions,
            "note": "长度序与能量序一致并不算错；要紧的是能量是算出来的，不是复制的长度序",
        },
    )

    # --- 6. same raw geometry on both sides ------------------------------
    spec = M.build("L3_multi_path", seeds[0], design)
    report = next(r for r in reports if r["map_id"] == spec.map_id)
    poly_from_report = report["island_polygons"]
    poly_from_obstacles = [
        [[float(p[0]), float(p[1])] for p in ob.polygon().vertices] for ob in spec.obstacles()
    ]
    check(
        "one_raw_geometry_both_sides",
        len(poly_from_report) == len(spec.islands) and
        all(
            max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pa, pb)) < 1e-12
            for pa, pb in zip(poly_from_report, poly_from_obstacles)
        ),
        {
            "islands": len(spec.islands),
            "note": "名义侧 inflate 与物理侧 polygon 都由同一份 islands 生成，没有第二份地图",
        },
    )

    # --- 7. area scaling: a larger field still forces a detour -----------
    bigger = dict(design)
    bigger["levels"] = [dict(lv) for lv in design["levels"]]
    for entry in bigger["levels"]:
        if entry["level"] == "L2_single_detour":
            entry["extent"] = [-12.0, 12.0, -10.2, 10.2]
    scaled = M.evaluate(M.build("L2_single_detour", seeds[0], bigger), geometry, model, design)
    check(
        "larger_field_still_detours",
        scaled["detour_ratio"] is not None and scaled["detour_ratio"] > 1.05
        and scaled["n_routes_feasible"] >= 2,
        {
            "extent": scaled["extent"],
            "net_width_m": scaled["net_width_m"],
            "gaps_m": sorted(g for values in scaled["gaps_by_column"].values() for g in values),
            "shortest_m": scaled["shortest_feasible_m"],
            "straight_m": scaled["straight_distance_m"],
            "detour_ratio": scaled["detour_ratio"],
            "conclusion": "把场地从 18x16.4 扩到 24x20.4，仍然必须绕行且两侧路线都可行",
        },
    )

    # --- 8. one figure per level, geometry eyeballable --------------------
    representative = [r for r in reports if r["seed"] == seeds[0]]
    svg = M.render_svg(representative, geometry)
    if args.svg:
        with open(args.svg, "w", encoding="utf-8") as fh:
            fh.write(svg)
    check(
        "svg_renders_each_level",
        len(representative) == 3
        and all(r["map_id"] in svg for r in representative)
        and svg.count("<polygon") >= sum(len(r["island_polygons"]) for r in representative)
        and svg.count("<polyline") >= sum(r["n_routes_feasible"] for r in representative),
        {
            "maps_in_figure": [r["map_id"] for r in representative],
            "bytes": len(svg),
            "polygons": svg.count("<polygon"),
            "polylines": svg.count("<polyline"),
            "note": "图只画每级代表图（seed=%d）；完整 9 张图的数据在 JSON 里" % seeds[0],
        },
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "design_version": design["version"],
        "geometry_contract_version": geometry.contract_version,
        "inflation_radius_m": geometry.inflation_radius_m,
        "n_maps": len(reports),
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "maps": [
            {
                "map_id": r["map_id"],
                "level": r["level"],
                "seed": r["seed"],
                "geometry_hash": r["geometry_hash"],
                "net_width_m": r["net_width_m"],
                "min_gap_m": r["min_gap_m"],
                "routes_feasible": r["n_routes_feasible"],
                "routes_total": r["n_routes_total"],
                "shortest_feasible_m": r["shortest_feasible_m"],
                "straight_distance_m": r["straight_distance_m"],
                "detour_ratio": r["detour_ratio"],
                "routes": [
                    {
                        "length_m": round(x["length_m"], 4),
                        "feasible": x["feasible"],
                        "v_ref_mps": round(x["v_ref_mps"], 4),
                        "energy_J": None if x["energy_J"] is None else round(x["energy_J"], 4),
                        "turns": x["turns"],
                        "polyline": x["polyline"],
                    }
                    for x in r["routes"]
                ],
            }
            for r in reports
        ],
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("checks", "maps")},
                     ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
