"""Safety-geometry checks for NAV-015.

Every counterexample the task names is constructed here: zero and double
inflation, boundary contact, edge crossing without vertex containment,
cross-step tunnelling, and a rotating sweep that the endpoint check misses.
The frozen ``polygon_distance`` is also exercised, because its vertex-only
intersection test is not sound for thin crossings.

Run:

    <frozen python> CaMRA/navigation/tests/test_safety_geometry.py \
        --json CaMRA/navigation/reports/safety_geometry_check.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402

from camra_calibration.geometry import Polygon, minkowski_sum  # noqa: E402
from camra_calibration.obstacles import Obstacle, polygon_distance  # noqa: E402

from camra_navigation.maps import geometry as G  # noqa: E402

FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
DATA_LOCK = os.path.join(CALIB, "config", "data_lock_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def rect(cx: float, cy: float, w: float, d: float, yaw: float = 0.0) -> Polygon:
    return Obstacle(cx, cy, w, d, yaw).polygon()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    geom = G.SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, CONTRACT)
    lock = json.load(open(DATA_LOCK, encoding="utf-8"))

    # --- 1. geometry identity and metadata --------------------------------
    meta = geom.metadata()
    check(
        "geometry_from_frozen",
        abs(geom.r_fp_m - 0.5055013879783843) < 1e-12
        and abs(geom.inflation_radius_m - (geom.r_fp_m + geom.delta_m)) < 1e-12
        and geom.footprint_sha256 == lock["footprint"]["sha256"]
        and meta["delta_applications"] == 1
        and meta["not_a_safety_certificate"] is True
        and "quantile" in meta["delta_semantics"],
        {
            "r_fp_m": geom.r_fp_m,
            "delta_m": geom.delta_m,
            "inflation_radius_m": geom.inflation_radius_m,
            "contract_version": geom.contract_version,
            "metadata_keys": sorted(meta.keys()),
        },
    )

    # A point exactly at the inflation radius is the boundary of the constraint.
    obstacle = rect(6.0, 0.0, 0.6, 0.6)
    r_inf = geom.inflation_radius_m
    d_center = float(obstacle.distance(np.array([[6.0, 0.0]]))[0])
    near_edge = np.array([6.0 - 0.3 - r_inf, 0.0])
    outside = np.array([6.0 - 0.3 - r_inf - 1e-6, 0.0])
    inside = np.array([6.0 - 0.3 - r_inf + 1e-6, 0.0])
    check(
        "inflation_applied_exactly_once",
        abs(d_center - (-0.3)) < 1e-9
        and abs(G.nominal_margin(near_edge, obstacle, geom)) < 1e-9
        and G.nominal_violation(near_edge, obstacle, geom, tol=1e-9)
        and not G.nominal_violation(outside, obstacle, geom)
        and G.nominal_violation(inside, obstacle, geom),
        {
            "signed_distance_at_centre_m": d_center,
            "inflation_radius_m": r_inf,
            "margin_at_boundary_m": G.nominal_margin(near_edge, obstacle, geom),
            "violation_at_boundary": G.nominal_violation(near_edge, obstacle, geom, tol=1e-9),
            "violation_1mm_outside": G.nominal_violation(outside, obstacle, geom),
            "violation_1mm_inside": G.nominal_violation(inside, obstacle, geom),
            "note": "边界计为违反，对应 d <= delta 的定义",
        },
    )

    # --- 2. zero and double inflation -------------------------------------
    # Doubling delta makes the forbidden set *larger*, so the two probe distances
    # below separate 0x / 1x / 2x into a unique pattern.
    no_margin = dataclasses.replace(geom, delta_m=0.0)
    double = dataclasses.replace(geom, delta_m=2.0 * geom.delta_m)
    probe_a = np.array([6.0 - 0.3 - (geom.r_fp_m + 0.5 * geom.delta_m), 0.0])
    probe_b = np.array([6.0 - 0.3 - (geom.r_fp_m + 1.5 * geom.delta_m), 0.0])
    pattern = {
        label: (
            G.nominal_violation(probe_a, obstacle, g),
            G.nominal_violation(probe_b, obstacle, g),
        )
        for label, g in (("delta_0x", no_margin), ("delta_1x", geom), ("delta_2x", double))
    }
    check(
        "zero_and_double_inflation_distinguishable",
        pattern["delta_0x"] == (False, False)
        and pattern["delta_1x"] == (True, False)
        and pattern["delta_2x"] == (True, True),
        {
            "probe_a_distance_m": geom.r_fp_m + 0.5 * geom.delta_m,
            "probe_b_distance_m": geom.r_fp_m + 1.5 * geom.delta_m,
            "violation_pattern_a_b": pattern,
            "conclusion": (
                "两个探针把 0x/1x/2x 分成唯一的三组结果；二次膨胀会多出 %.3f m 的假违反带"
                % geom.delta_m
            ),
        },
    )

    # --- 3. boundary walls, including the corner --------------------------
    extent = (-12.0, 12.0, -9.0, 9.0)
    r = geom.inflation_radius_m
    safe_wall = (-12.0 + r + 1e-3, 0.0)
    bad_wall = (-12.0 + r - 1e-3, 0.0)
    corner_safe = (-12.0 + r + 1e-3, -9.0 + r + 1e-3)
    corner_bad = (-12.0 + r - 1e-3, -9.0 + r - 1e-3)
    check(
        "boundary_same_treatment",
        not G.boundary_violation(safe_wall, extent, geom)
        and G.boundary_violation(bad_wall, extent, geom)
        and not G.boundary_violation(corner_safe, extent, geom)
        and G.boundary_violation(corner_bad, extent, geom),
        {
            "r_plus_delta_m": r,
            "margin_safe_m": G.boundary_margin(safe_wall, extent, geom),
            "margin_bad_m": G.boundary_margin(bad_wall, extent, geom),
            "corner_safe": corner_safe,
            "corner_bad_violation": G.boundary_violation(corner_bad, extent, geom),
            "note": "四边半平面取 min 等价于矩形按 disc 腐蚀（内切矩形，方角），两轴同时越界即违反",
        },
    )

    # --- 4. edge crossing with no vertex containment ----------------------
    horizontal = rect(0.0, 0.0, 20.0, 0.2)
    vertical = rect(0.0, 0.0, 0.2, 20.0)
    frozen = polygon_distance(horizontal, vertical)
    sat = G.convex_polygons_intersect(horizontal, vertical)
    exact = G.polygon_distance_exact(horizontal, vertical)
    h_in_v = vertical.contains(horizontal.vertices).any()
    v_in_h = horizontal.contains(vertical.vertices).any()
    check(
        "edge_crossing_counterexample",
        (not h_in_v) and (not v_in_h) and sat and exact == 0.0,
        {
            "horizontal_vertices_inside_vertical": bool(h_in_v),
            "vertical_vertices_inside_horizontal": bool(v_in_h),
            "frozen_polygon_distance_m": frozen,
            "sat_intersect": sat,
            "exact_distance_m": exact,
            "conclusion": (
                "两个 20x0.2 的薄矩形十字交叉：任一方都没有顶点落在对方内部，"
                "冻结 polygon_distance 因此返回 %.3f m（漏判），SAT 与边-边距离判为相交。"
                "NAV-015 不能用该函数做碰撞判据。" % frozen
            ),
        },
    )

    # The same failure mode with the real footprint against a thin wall: the wall
    # lies entirely between the footprint's vertices, so the frozen vertex-based
    # test reports a positive distance for a real overlap.
    thin_wall = rect(0.0, 0.0, 4.0, 0.05)
    fp_hit = G.physical_collision((0.0, 0.0), 0.0, thin_wall, geom)
    frozen_fp = polygon_distance(G.footprint_at(geom, (0.0, 0.0), 0.0), thin_wall)
    check(
        "thin_wall_through_footprint",
        fp_hit and frozen_fp > 0.01,
        {
            "sat_collision": fp_hit,
            "frozen_polygon_distance_m": frozen_fp,
            "note": (
                "0.05 m 薄墙整条落在足迹顶点之间，冻结函数因此报 %.3f m 的非零距离；"
                "机器人穿过薄墙这一最常见情形就会被漏判" % frozen_fp
            ),
        },
    )

    # --- 5. cross-step tunnelling -----------------------------------------
    # At task scale it cannot happen: the forbidden band is 2*(r_fp + delta) across
    # while the longest possible step is v_max * 0.1 s.
    p0 = np.array([-1.0, 0.0])
    p1 = np.array([-0.9, 0.0])
    wall_between = rect(-0.95, 0.0, 0.02, 4.0)
    endpoint_only = G.nominal_violation(p0, wall_between, geom) or G.nominal_violation(
        p1, wall_between, geom
    )
    step_test = G.nominal_violation_during_step(p0, p1, wall_between, geom)
    check(
        "real_delta_endpoints_already_violate",
        endpoint_only and step_test,
        {
            "inflated_thickness_m": 2.0 * r + 0.02,
            "step_length_m": float(np.linalg.norm(p1 - p0)),
            "endpoint_violation": endpoint_only,
            "segment_violation": step_test,
            "conclusion": "delta=3.12 m 远大于 0.1 m 步长，端点就已经违反，不存在漏判空间",
        },
    )
    max_step = 1.0 * 0.1          # v_max * dt_nav
    band_zero_delta = 2.0 * geom.r_fp_m
    check(
        "tunnelling_impossible_at_task_scale",
        max_step < band_zero_delta / 2.0,
        {
            "max_step_m": max_step,
            "half_band_with_delta_m": geom.inflation_radius_m,
            "half_band_without_delta_m": geom.r_fp_m,
            "conclusion": (
                "即使把 delta 取 0，禁入带仍有 %.3f m 厚，是 0.1 m 步长的 %.1f 倍；"
                "本任务尺度下跨步穿透不可能发生" % (band_zero_delta, band_zero_delta / max_step)
            ),
        },
    )
    # The predicate is still correct where the assumption would break, so the
    # guard is real rather than decorative: a tiny footprint with delta=0 gives a
    # 0.057 m band, thinner than an 0.8 m step.
    tiny = Polygon(np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]]))
    tiny_geom = G.SafetyGeometry(
        footprint=tiny,
        r_fp_m=tiny.circumscribed_radius,
        delta_m=0.0,
        contract_version=geom.contract_version,
        footprint_sha256="synthetic",
        margins_sha256="synthetic",
    )
    wall_thin = rect(0.0, 0.0, 0.02, 4.0)
    q0 = np.array([-0.4, 0.0])
    q1 = np.array([0.4, 0.0])
    ep0 = G.nominal_violation(q0, wall_thin, tiny_geom)
    ep1 = G.nominal_violation(q1, wall_thin, tiny_geom)
    segv = G.nominal_violation_during_step(q0, q1, wall_thin, tiny_geom)
    check(
        "segment_test_catches_thin_band_tunnelling",
        (not ep0) and (not ep1) and segv,
        {
            "band_thickness_m": 2.0 * tiny_geom.inflation_radius_m,
            "step_length_m": float(np.linalg.norm(q1 - q0)),
            "endpoint_violation_start": ep0,
            "endpoint_violation_end": ep1,
            "segment_violation": segv,
            "conclusion": "带宽 0.057 m 小于步长时，只查端点会漏；段判据能抓到，证明该判断不是摆设",
        },
    )

    # --- 6. rotation sweep: a real case the endpoint check misses ---------
    thin = Polygon(np.array([[-1.0, -0.1], [1.0, -0.1], [1.0, 0.1], [-1.0, 0.1]]))
    thin_geom = G.SafetyGeometry(
        footprint=thin,
        r_fp_m=thin.circumscribed_radius,
        delta_m=0.0,
        contract_version=geom.contract_version,
        footprint_sha256="synthetic",
        margins_sha256="synthetic",
    )
    # At 45 deg the tip arc bulges well outside the hull of the 0/90 deg poses.
    probe = rect(0.70, 0.70, 0.05, 0.05)
    hit0 = G.physical_collision((0.0, 0.0), 0.0, probe, thin_geom)
    hit90 = G.physical_collision((0.0, 0.0), math.pi / 2.0, probe, thin_geom)
    swept = G.physical_collision_during_step(
        (0.0, 0.0), 0.0, (0.0, 0.0), math.pi / 2.0, probe, thin_geom
    )
    # Sub-sampling confirms the probe really is swept, so the bound is not vacuous.
    subsample = any(
        G.physical_collision((0.0, 0.0), k * (math.pi / 2.0) / 32.0, probe, thin_geom)
        for k in range(33)
    )
    hull_contains = thin.union(thin.rotate(math.pi / 2.0)).contains(
        np.array([[0.70, 0.70]])
    )[0]
    check(
        "rotation_sweep_endpoint_miss",
        (not hit0) and (not hit90) and swept and subsample and not hull_contains,
        {
            "probe_xy": [0.70, 0.70],
            "collision_at_0deg": hit0,
            "collision_at_90deg": hit90,
            "endpoint_hull_contains_probe": bool(hull_contains),
            "conservative_sweep_collision": swept,
            "subsample_finds_collision": subsample,
            "r_fp_m": thin_geom.r_fp_m,
            "bound_extra_radius_m": thin_geom.r_fp_m * (math.pi / 2.0) / 2.0,
            "conclusion": (
                "长条在 0->90 deg 的一步内扫过探针，但两个端点姿态都不碰、两端点凸包也不含探针；"
                "保守扫掠界与 32 细分采样都能判定碰撞"
            ),
        },
    )

    # --- 7. the three fields are not aliases ------------------------------
    far_enough_to_touch = np.array([8.0, 0.0])
    nom_v = G.nominal_violation(far_enough_to_touch, obstacle, geom)
    phys_c = G.physical_collision(far_enough_to_touch, 0.0, obstacle, geom)
    step = G.StepSafety(
        physical_collision=phys_c, nominal_violation=nom_v, safety_y=G.update_safety_history(0.0, nom_v)
    )
    check(
        "three_fields_separate",
        nom_v and (not phys_c) and step.safety_y == 1.0,
        {
            **step.as_dict(),
            "clearance_m": G.physical_clearance(far_enough_to_touch, 0.0, obstacle, geom),
            "nominal_margin_m": G.nominal_margin(far_enough_to_touch, obstacle, geom),
            "conclusion": "位置在 delta 带内（名义违反）但没有真实接触；三个字段必须各记各的",
        },
    )
    contact_pos = np.array([6.0 - 0.3 - 0.3990897413625509 + 1e-6, 0.0])
    check(
        "physical_contact_reported_when_touching",
        G.physical_collision(contact_pos, 0.0, obstacle, geom)
        and G.physical_clearance(contact_pos, 0.0, obstacle, geom) == 0.0,
        {
            "clearance_m": G.physical_clearance(contact_pos, 0.0, obstacle, geom),
            "note": "足迹多边形前缘（+x 极值 0.3991 m）压入障碍 1 um 即判接触；注意这用的是真实足迹而非圆盘",
        },
    )

    # --- 8. irreversible history and legacy avoid mapping ------------------
    seq = [0.0]
    for violation in (False, True, False, False):
        seq.append(G.update_safety_history(seq[-1], violation))
    avoid_map = {"avoid=-1 -> y": G.legacy_avoid_to_y(-1.0), "avoid=+1 -> y": G.legacy_avoid_to_y(1.0)}
    round_trip = all(
        G.y_to_legacy_avoid(G.legacy_avoid_to_y(a)) == a for a in (-1.0, 1.0)
    )
    check(
        "history_irreversible_and_sign_mapping",
        seq == [0.0, 0.0, 1.0, 1.0, 1.0] and avoid_map["avoid=-1 -> y"] == 1.0
        and avoid_map["avoid=+1 -> y"] == 0.0 and round_trip,
        {
            "history_sequence": seq,
            "avoid_mapping": avoid_map,
            "round_trip_ok": round_trip,
            "reference": "RC-PPO avoid_value = where(is_avoid, -1, 1)：-1 不安全、+1 安全",
        },
    )

    # --- 9. obstacle list order does not matter ---------------------------
    obs_list = [rect(5.0, 1.0, 0.5, 0.5), rect(5.0, -1.0, 0.5, 0.5)]
    probe_pos = np.array([3.0, 0.0])
    margin_fwd = min(G.nominal_margin(probe_pos, o, geom) for o in obs_list)
    margin_rev = min(G.nominal_margin(probe_pos, o, geom) for o in reversed(obs_list))
    check(
        "order_independent",
        abs(margin_fwd - margin_rev) < 1e-15,
        {"margin_forward_m": margin_fwd, "margin_reverse_m": margin_rev},
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "contract_version": geom.contract_version,
        "r_fp_m": geom.r_fp_m,
        "delta_m": geom.delta_m,
        "inflation_radius_m": geom.inflation_radius_m,
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
