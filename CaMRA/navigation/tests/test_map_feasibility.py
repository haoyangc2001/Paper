"""Feasibility checks for NAV-017.

Covers the admission record for the nine catalogue maps plus the four rejection
counterexamples the task names: a blocked corridor, an infeasible turn, a goal
inside the inflated region and a route that cannot finish in time.  Also shows
that the continuous verification is active by feeding it a straight line the
grid would happily accept.

Run:

    <frozen python> CaMRA/navigation/tests/test_map_feasibility.py \
        --json CaMRA/navigation/reports/map_feasibility.json
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

from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

RESOLUTION = 0.15
RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def with_level(design: dict, level: str, **overrides) -> dict:
    """Copy of the design with one level's fields replaced (for counterexamples)."""
    clone = json.loads(json.dumps(design))
    for entry in clone["levels"]:
        if entry["level"] == level:
            entry.update(overrides)
    return clone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    seeds = design["seeds"]["values"]

    specs = [M.build(lv, s, design) for lv in M.LEVELS for s in seeds]
    records = F.check_catalogue(specs, geometry, model, resolution=RESOLUTION)
    routes = [r["route"] for r in records if r["route"]]

    # --- 1. every catalogue map is admitted -------------------------------
    check(
        "catalogue_all_admitted",
        all(r["admitted"] for r in records) and len(records) == 9,
        [
            {"map": r["map_id"], "admitted": r["admitted"], "reasons": r["reasons"],
             "grid_connected": r["grid_connected"]}
            for r in records
        ],
    )

    # --- 2. the admission record has the required fields ------------------
    required = {
        "polyline", "length_m", "min_nominal_margin_m", "min_physical_clearance_m",
        "sweep_collisions", "n_commands", "predicted_energy_deadline_J",
        "predicted_energy_fast_J", "t_min_s", "v_cap_mps", "v_deadline_mps",
        "physical_status", "reasons", "feasible_in_time",
    }
    missing = [
        {"map": r["map_id"], "missing": sorted(required - set(r["route"] or {}))}
        for r in records if r["route"] is None or (required - set(r["route"]))
    ]
    check("admission_record_complete", not missing, missing or "all fields present")

    with_obstacles = [r for r in routes if r["min_nominal_margin_m"] is not None]
    check(
        "clearance_and_sweep_clean",
        all(x["min_nominal_margin_m"] > 0.0 for x in with_obstacles)
        and all(x["min_physical_clearance_m"] > 0.0 for x in with_obstacles)
        and all(x["sweep_collisions"] == 0 for x in routes),
        {
            "maps_with_obstacles": len(with_obstacles),
            "min_nominal_margin_m": round(min(x["min_nominal_margin_m"] for x in with_obstacles), 6),
            "min_physical_clearance_m": round(
                min(x["min_physical_clearance_m"] for x in with_obstacles), 6
            ),
            "total_sweep_collisions": sum(x["sweep_collisions"] for x in routes),
            "obstacle_free_maps_report_null": [
                r["map_id"] for r in records
                if r["route"] and r["route"]["min_nominal_margin_m"] is None
            ],
        },
    )

    check(
        "deadline_satisfied",
        all(x["feasible_in_time"] and x["t_min_s"] <= 30.0 for x in routes),
        [
            {"map": r["map_id"], "length_m": round(r["route"]["length_m"], 3),
             "v_cap": round(r["route"]["v_cap_mps"], 4),
             "v_deadline": round(r["route"]["v_deadline_mps"], 4),
             "t_min_s": round(r["route"]["t_min_s"], 3),
             "t_at_vmax_s": round(r["route"]["t_at_vmax_s"], 3)}
            for r in records if r["route"] is not None
        ],
    )

    check(
        "commands_and_two_energy_bounds",
        all(x["n_commands"] > 0
            and x["predicted_energy_fast_J"] is not None
            and x["predicted_energy_deadline_J"] is not None
            and x["predicted_energy_deadline_J"] <= x["predicted_energy_fast_J"]
            for x in routes),
        [
            {"map": r["map_id"], "n_commands": r["route"]["n_commands"],
             "energy_deadline_J": round(r["route"]["predicted_energy_deadline_J"], 2),
             "energy_fast_J": round(r["route"]["predicted_energy_fast_J"], 2)}
            for r in records if r["route"]
        ],
    )

    check(
        "nothing_claimed_as_measured",
        all(r["physical_status"] == "not_measured"
            and r["route"]["physical_status"] == "not_measured" for r in records),
        "every record carries physical_status=not_measured; NAV-028/059 own the physical evidence",
    )

    # --- 3. continuous verification is active -----------------------------
    spec_l2 = M.build("L2_single_detour", seeds[0], design)
    straight = [spec_l2.start, spec_l2.goal]
    straight_result = F.evaluate_path(spec_l2, geometry, straight, model)
    fs = F.build_free_space(spec_l2, geometry, RESOLUTION)
    check(
        "continuous_check_rejects_corner_clip",
        (not straight_result["admitted"])
        and "corner_clip_after_shortcut" in straight_result["reasons"]
        and fs.is_free(*spec_l2.start),
        {
            "reasons": straight_result["reasons"],
            "min_nominal_margin_m": round(straight_result["min_nominal_margin_m"], 4),
            "note": "网格连通不代表直线可行；0.1 m 连续采样把它挡下",
        },
    )

    # --- 4. counterexample: blocked corridor ------------------------------
    blocked = with_level(
        design, "L2_single_detour",
        islands=[{"x": 0.0, "y": 0.0, "width": 2.0, "depth": 40.0, "yaw": 0.0}],
    )
    blocked_rec = F.check_map(M.build("L2_single_detour", seeds[0], blocked), geometry, model,
                              resolution=RESOLUTION)
    check(
        "rejects_blocked_corridor",
        (not blocked_rec["admitted"]) and "no_connected_path" in blocked_rec["reasons"],
        {"reasons": blocked_rec["reasons"],
         "free_fraction": round(blocked_rec["free_fraction"], 4),
         "grid_connected": blocked_rec["grid_connected"]},
    )

    # --- 5. counterexample: goal inside the inflated region ---------------
    inside = with_level(design, "L2_single_detour", goal=[0.0, 3.0])
    inside_rec = F.check_map(M.build("L2_single_detour", seeds[0], inside), geometry, model,
                             resolution=RESOLUTION)
    check(
        "rejects_goal_inside_inflated_region",
        (not inside_rec["admitted"]) and "goal_outside_free" in inside_rec["reasons"],
        {"reasons": inside_rec["reasons"],
         "goal_margin_m": round(inside_rec["goal_margin_m"], 4)},
    )

    # --- 6. counterexample: route cannot finish in time -------------------
    long_extent = with_level(
        design, "L2_single_detour",
        extent=[-20.0, 20.0, -8.2, 8.2], start=[-16.0, 0.0], goal=[16.0, 0.0],
    )
    long_rec = F.check_map(M.build("L2_single_detour", seeds[0], long_extent), geometry, model,
                           resolution=RESOLUTION)
    check(
        "rejects_timeout",
        (not long_rec["admitted"]) and "timeout" in long_rec["reasons"],
        {"reasons": long_rec["reasons"],
         "length_m": round(long_rec["route"]["length_m"], 3),
         "t_at_vmax_s": round(long_rec["route"]["t_at_vmax_s"], 3),
         "note": "起终点相距 32 m，即使按 1.0 m/s 也要 32 s，超过 30 s 时限"},
    )

    # --- 7. counterexample: the turn rate, not the length, is binding -----
    # Synthetic route on a widened L2 (y = +-12) so the square wave sits between
    # the inflated island (|y| <= 4.12) and the inset wall (|y| <= 8.88).  Twenty
    # 1 m legs with 90 deg turns: 20 m at v_max takes 20 s, but each turn over a
    # 1 m span caps the speed at wz_max/kappa = 0.64 m/s, pushing the trip past
    # 30 s.  The real delta is kept; nothing is "fixed" by shrinking it.
    wide = with_level(design, "L2_single_detour", extent=[-11.0, 11.0, -12.0, 12.0])
    spec_wide = M.build("L2_single_detour", seeds[0], wide)
    square = [(-5.0, 6.0)]
    x = -5.0
    for _ in range(5):
        square.append((x, 7.0))
        square.append((x + 1.0, 7.0))
        square.append((x + 1.0, 6.0))
        square.append((x + 2.0, 6.0))
        x += 2.0
    turn_result = F.evaluate_path(spec_wide, geometry, square, model)
    check(
        "rejects_turn_limited_route",
        (not turn_result["admitted"])
        and "turn_limited" in turn_result["reasons"]
        and turn_result["t_at_vmax_s"] <= 30.0 < turn_result["t_min_s"]
        and turn_result["min_nominal_margin_m"] > 0.0,
        {
            "length_m": round(turn_result["length_m"], 3),
            "kappa_max_per_m": round(turn_result["kappa_max_per_m"], 3),
            "v_cap_mps": round(turn_result["v_cap_mps"], 4),
            "t_at_vmax_s": round(turn_result["t_at_vmax_s"], 3),
            "t_min_s": round(turn_result["t_min_s"], 3),
            "min_nominal_margin_m": round(turn_result["min_nominal_margin_m"], 4),
            "reasons": turn_result["reasons"],
            "note": "几何上能走完、按最大速度也够快，但 90 度弯把速度上限压到 v_cap，于是超时",
        },
    )
    check(
        "turning_not_binding_for_real_delta",
        all(abs(x["t_min_s"] - x["t_at_vmax_s"]) < 1e-9 for x in routes),
        {
            "pairs": [{"map": r["map_id"], "v_cap": round(r["route"]["v_cap_mps"], 4),
                       "kappa_max": round(r["route"]["kappa_max_per_m"], 4)}
                      for r in records if r["route"]],
            "conclusion": "delta=3.12 m 下所有正式地图的 v_cap 都顶到 1.0 m/s，转弯不构成限制；时限才是约束",
        },
    )

    # --- 8. compared with NAV-016's design polyline -----------------------
    design_routes = [M.evaluate(s, geometry, model, design) for s in specs]
    pairs = []
    for rec, des in zip(records, design_routes):
        best = min((x["length_m"] for x in des["routes"] if x["feasible"]), default=None)
        if rec["route"] and best is not None:
            pairs.append({"map": rec["map_id"], "design_m": best,
                          "searched_m": rec["route"]["length_m"]})
    check(
        "search_not_much_longer_than_design_polyline",
        all(p["searched_m"] <= p["design_m"] * 1.02 for p in pairs),
        {
            "pairs": [{"map": p["map"], "design_m": round(p["design_m"], 3),
                       "searched_m": round(p["searched_m"], 3),
                       "ratio": round(p["searched_m"] / p["design_m"], 4)} for p in pairs],
            "worst_ratio": round(max(p["searched_m"] / p["design_m"] for p in pairs), 4),
            "conclusion": "搜索路径可显著短于设计折线（L2 约 -1.8 m）；略长处仅为网格分辨率量级",
        },
    )

    # --- 9. tightest admitted margin is reported, not hidden --------------
    tightest = min(with_obstacles, key=lambda x: x["min_nominal_margin_m"])
    check(
        "tightest_margin_reported",
        tightest["min_nominal_margin_m"] > 0.0,
        {
            "min_nominal_margin_m": round(tightest["min_nominal_margin_m"], 4),
            "min_physical_clearance_m": round(tightest["min_physical_clearance_m"], 4),
            "note": "最短路径会贴着膨胀边界走，只剩厘米级名义余量；准入判据是 >0，"
                    "若要更大余量需另加 clearance 权重（留给 NAV-041/042）",
        },
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "geometry_contract_version": geometry.contract_version,
        "inflation_radius_m": geometry.inflation_radius_m,
        "grid_resolution_m": RESOLUTION,
        "n_maps": len(records),
        "n_admitted": sum(1 for r in records if r["admitted"]),
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "selection_rule": "admission happens before any training; maps are not re-selected by policy results",
        "physical_evidence_pending": "NAV-028 / NAV-059",
        "maps": [
            {
                "map_id": r["map_id"],
                "level": r["level"],
                "seed": r["seed"],
                "admitted": r["admitted"],
                "reasons": r["reasons"],
                "grid_connected": r["grid_connected"],
                "free_fraction": round(r["free_fraction"], 5),
                "route": None if r["route"] is None else {
                    "length_m": round(r["route"]["length_m"], 4),
                    "v_cap_mps": round(r["route"]["v_cap_mps"], 4),
                    "v_deadline_mps": round(r["route"]["v_deadline_mps"], 4),
                    "t_min_s": round(r["route"]["t_min_s"], 4),
                    "kappa_max_per_m": round(r["route"]["kappa_max_per_m"], 5),
                    "min_nominal_margin_m": None if r["route"]["min_nominal_margin_m"] is None
                    else round(r["route"]["min_nominal_margin_m"], 5),
                    "min_physical_clearance_m": None
                    if r["route"]["min_physical_clearance_m"] is None
                    else round(r["route"]["min_physical_clearance_m"], 5),
                    "sweep_collisions": r["route"]["sweep_collisions"],
                    "n_commands": r["route"]["n_commands"],
                    "predicted_energy_fast_J": round(r["route"]["predicted_energy_fast_J"], 3)
                    if r["route"]["predicted_energy_fast_J"] is not None else None,
                    "predicted_energy_deadline_J": round(r["route"]["predicted_energy_deadline_J"], 3)
                    if r["route"]["predicted_energy_deadline_J"] is not None else None,
                    "physical_status": r["route"]["physical_status"],
                    "polyline": r["route"]["polyline"],
                },
            }
            for r in records
        ],
        "counterexamples": {
            "corner_clip": straight_result["reasons"],
            "blocked": blocked_rec["reasons"],
            "goal_inside": inside_rec["reasons"],
            "timeout": long_rec["reasons"],
            "turn_limited": turn_result["reasons"],
        },
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("checks", "maps")},
                     ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
