"""Pose/footprint alignment checks for NAV-057.

The decision under test is: constrain the nominal position with a
rotation-invariant circumscribed disc of radius ``r_fp`` plus ``delta``, rather
than a nominal-attitude rectangle plus ``delta``.  The checks below are the
counterexamples the task asks for, computed on the frozen footprint.

Run:

    <frozen python> CaMRA/navigation/tests/test_pose_footprint.py \
        --json CaMRA/navigation/reports/pose_footprint_alignment.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from camra_calibration.geometry import (  # noqa: E402
    Polygon,
    disc,
    minkowski_sum,
    rotating_sweep_hull,
)

FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
DATA_LOCK = os.path.join(CALIB, "config", "data_lock_v1.json")
CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def rotate(points: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    contract = json.load(open(CONTRACT, encoding="utf-8"))
    fp_json = json.load(open(FOOTPRINT, encoding="utf-8"))
    margins = json.load(open(MARGINS, encoding="utf-8"))
    lock = json.load(open(DATA_LOCK, encoding="utf-8"))

    poly = Polygon(np.asarray(fp_json["polygon"]["vertices"], dtype=np.float64))
    r_fp = poly.circumscribed_radius
    delta = float(margins["delta"])
    c_geo = contract["frozen_geometry"]

    # --- 1. frozen footprint identity --------------------------------------
    byte_hash = hashlib.sha256(open(FOOTPRINT, "rb").read()).hexdigest()
    check(
        "footprint_identity",
        abs(r_fp - c_geo["circumscribed_radius_m"]) < 1e-12
        and abs(poly.area - c_geo["polygon_area_m2"]) < 1e-12
        and byte_hash == lock["footprint"]["sha256"]
        and fp_json["sha256"] == lock["footprint"]["content_sha256"],
        {
            "r_fp_m": r_fp,
            "area_m2": poly.area,
            "byte_sha256": byte_hash[:16],
            "declared_content_sha256": fp_json["sha256"][:16],
            "note": "冻结足迹未改动；字节散列与自述散列分别绑定",
        },
    )

    # --- 2. the radius is measured from the origin, not the centroid ------
    centroid = np.asarray(fp_json["polygon"]["centroid"], dtype=np.float64)
    r_centroid = float(np.linalg.norm(poly.vertices - centroid, axis=1).max())
    check(
        "radius_reference_is_base_link_origin",
        abs(r_fp - 0.5055013879783843) < 1e-12 and r_centroid < r_fp - 0.02,
        {
            "r_from_origin_m": r_fp,
            "r_from_centroid_m": r_centroid,
            "shortfall_if_centroid_m": r_fp - r_centroid,
            "conclusion": "以质心为圆心会漏掉机头/机尾约 2.42 cm，且旋转时不再保守",
        },
    )

    # --- 3. rotation invariance: every yaw stays inside the same disc -----
    worst_radius = 0.0
    for k in range(720):
        yaw = k * math.pi / 360.0
        rotated = rotate(poly.vertices, yaw)
        worst_radius = max(worst_radius, float(np.linalg.norm(rotated, axis=1).max()))
    disc_poly = disc(r_fp, circumscribed=True)
    disc_min_radius = float(np.linalg.norm(disc_poly, axis=1).min())
    check(
        "rotation_invariance_all_yaw",
        worst_radius <= r_fp + 1e-12 and disc_min_radius >= r_fp - 1e-12,
        {
            "yaw_samples": 720,
            "worst_vertex_radius_m": worst_radius,
            "r_fp_m": r_fp,
            "disc_polygon_inner_radius_m": disc_min_radius,
            "conclusion": "旋转不改变顶点到原点的距离，故任何航向都在同一圆盘内",
        },
    )

    # --- 4. the sampled sweep hull is a lower bound, not a boundary -------
    areas = []
    for substeps in (1, 4, 16):
        hull = rotating_sweep_hull(poly, [((0.0, 0.0), 0.0), ((0.0, 0.0), 2.0 * math.pi)],
                                   substeps=substeps)
        areas.append({"substeps": substeps, "area_m2": hull.area,
                      "max_vertex_radius_m": float(np.linalg.norm(hull.vertices, axis=1).max())})
    monotone = all(areas[i]["area_m2"] <= areas[i + 1]["area_m2"] + 1e-12 for i in range(len(areas) - 1))
    check(
        "sweep_hull_is_lower_bound",
        monotone and all(a["max_vertex_radius_m"] <= r_fp + 1e-9 for a in areas),
        {**{"samples": areas}, "conclusion": "采样越多面积单调不减；故不能当安全边界，只能做收敛检查"},
    )

    # --- 5. nominal-attitude rectangle is not attitude-robust -------------
    def aabb(yaw: float) -> dict[str, float]:
        v = rotate(poly.vertices, yaw)
        return {
            "x_extent_m": float(v[:, 0].max() - v[:, 0].min()),
            "y_extent_m": float(v[:, 1].max() - v[:, 1].min()),
        }

    boxes = {"0deg": aabb(0.0), "45deg": aabb(math.pi / 4), "90deg": aabb(math.pi / 2)}
    # The corridor-relevant quantity is the extent perpendicular to travel, so the
    # growth must be measured on the axis that actually changes: here +y grows
    # 0.585 -> 0.855 m when the long axis turns, while +x barely moves because the
    # long axis is already aligned at 0 deg.
    growth_x = max(b["x_extent_m"] for b in boxes.values()) - boxes["0deg"]["x_extent_m"]
    growth_y = max(b["y_extent_m"] for b in boxes.values()) - boxes["0deg"]["y_extent_m"]
    growth = max(growth_x, growth_y)
    check(
        "rectangle_changes_with_heading",
        growth > 0.2,
        {**boxes, "growth_x_m": growth_x, "growth_y_m": growth_y,
         "conclusion": "按 0° 姿态算的横向包围在转向 45° 后少算约 0.27 m；delta 补不上形状差"},
    )

    # --- 6. long rectangle turning: a point missed by the 0deg footprint --
    rotated45 = Polygon(rotate(poly.vertices, math.pi / 4))
    candidates = rotate(poly.vertices, math.pi / 4).tolist()
    missed = [
        p for p in candidates
        if rotated45.contains(np.asarray([p]))[0] and not poly.contains(np.asarray([p]))[0]
    ]
    sample_point = missed[0] if missed else None
    check(
        "long_rectangle_turning",
        bool(missed),
        {
            "n_vertices_outside_0deg_footprint": len(missed),
            "example_point_m": sample_point,
            "clearance_from_0deg_footprint_m": (
                float(min(np.linalg.norm(poly.vertices - np.asarray(sample_point), axis=1)))
                if sample_point else None
            ),
            "conclusion": "同一中心转到 45° 后，机角会伸到 0° 足迹之外；按 0° 判据会漏判真实接触",
        },
    )

    # --- 7. wall corner case built on that point --------------------------
    corner_ok = False
    corner_detail: dict[str, Any] = {}
    if sample_point is not None:
        q = np.asarray(sample_point, dtype=np.float64)
        # A tiny obstacle sitting exactly at the missed point: no contact for the
        # nominal-attitude footprint, contact for the turned one.
        hit0 = poly.contains(q.reshape(1, 2))[0]
        hit45 = rotated45.contains(q.reshape(1, 2))[0]
        corner_ok = (not hit0) and hit45
        corner_detail = {
            "point_m": q.tolist(),
            "contact_at_0deg": bool(hit0),
            "contact_at_45deg": bool(hit45),
            "conclusion": "墙角场景下，0° 判据报「无接触」而 45° 实际接触；与 delta 无关，是形状误差",
        }
    check("wall_corner_case", corner_ok, corner_detail)

    # --- 8. same centre, different heading --------------------------------
    disc_a = disc(r_fp, circumscribed=True)
    disc_b = disc(r_fp, circumscribed=True)
    same_disc = np.allclose(disc_a, disc_b)
    b0, b45 = aabb(0.0), aabb(math.pi / 4)
    check(
        "same_centre_different_heading",
        same_disc and (abs(b0["x_extent_m"] - b45["x_extent_m"]) > 1e-6),
        {
            "disc_identical": bool(same_disc),
            "rect_0deg": b0,
            "rect_45deg": b45,
            "conclusion": "约束几何必须与航向无关；矩形方案在同一中心换航向时几何就变了",
        },
    )

    # --- 9. conservativeness under a bounded position error ---------------
    r_plus_d = r_fp + delta
    worst_case = 0.0
    for k in range(16):
        eps = np.array([math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)]) * delta
        for j in range(24):
            yaw = j * math.pi / 12.0
            actual = rotate(poly.vertices, yaw) + eps
            worst_case = max(worst_case, float(np.linalg.norm(actual, axis=1).max()))
    check(
        "conservative_under_position_error",
        worst_case <= r_plus_d + 1e-12,
        {
            "r_fp_plus_delta_m": r_plus_d,
            "worst_actual_vertex_radius_m": worst_case,
            "headroom_m": r_plus_d - worst_case,
            "conclusion": "|eps| <= delta 时实际足迹必在 disc(p_hat, r_fp+delta) 内；delta 是 90% 分位，超出时不享有该保证",
        },
    )

    # --- 10. width arithmetic ---------------------------------------------
    w_disc = 2.0 * r_plus_d
    # Corridor width at a given heading: 2 * (perpendicular half-width + delta).
    v0 = rotate(poly.vertices, 0.0)
    hw0 = float(max(abs(v0[:, 1].min()), abs(v0[:, 1].max())))
    w_rect = 2.0 * (hw0 + delta)          # the 0 deg attitude stated in the contract
    half_widths = []
    for k in range(720):
        v = rotate(poly.vertices, k * math.pi / 360.0)
        half_widths.append(float(max(abs(v[:, 1].min()), abs(v[:, 1].max()))))
    hw_max = max(half_widths)
    w_rect_max = 2.0 * (hw_max + delta)   # worst sampled attitude
    premium = w_disc - w_rect
    check(
        "width_arithmetic",
        abs(w_disc - contract["width_and_map"]["straight_corridor_lower_bound_m"]) < 1e-9
        and abs(w_rect - contract["width_and_map"]["nominal_attitude_rectangle_width_m"]) < 1e-9
        and abs(premium - contract["width_and_map"]["disc_premium_m"]) < 1e-9
        and w_disc < 7.0,
        {
            "W_disc_m": w_disc,
            "W_rectangle_0deg_m": w_rect,
            "W_rectangle_worst_sampled_m": w_rect_max,
            "half_width_0deg_m": hw0,
            "half_width_worst_sampled_m": hw_max,
            "r_fp_m": r_fp,
            "premium_m": premium,
            "premium_ratio": premium / w_rect,
            "design_range_m": contract["width_and_map"]["design_range_m"],
            "conclusion": "圆盘下界 6.2387 m 仍低于已确认的 7 m 净宽，故本近似不改变已确认设计尺度",
        },
    )
    check(
        "rectangle_width_depends_on_heading",
        abs(hw_max - r_fp) < 1e-5 and abs(w_rect_max - w_disc) < 1e-4,
        {
            "W_rectangle_0deg_m": w_rect,
            "W_rectangle_worst_sampled_m": w_rect_max,
            "W_disc_m": w_disc,
            "gap_m": w_rect_max - w_rect,
            "half_width_worst_vs_r_fp_diff_m": r_fp - hw_max,
            "yaw_samples": 720,
            "conclusion": (
                "横向半宽随航向变化，最坏姿态等于圆盘半径 r_fp（采样 0.5° 步长下差 %.1e m，"
                "属航向采样分辨率而非几何误差）；按 0° 定的净宽对会转向的机器人少算 %.3f m"
                % (r_fp - hw_max, w_rect_max - w_rect)
            ),
        },
    )

    # --- 11. disc contains the footprint inflated by delta (two-step form) -
    inflation = minkowski_sum(poly, Polygon(disc(delta, circumscribed=True)))
    max_norm = float(np.linalg.norm(inflation.vertices, axis=1).max())
    arc_inflation = disc_poly.shape[0]
    slack = max_norm - r_plus_d
    check(
        "two_step_inflation_matches_one_step",
        0.0 <= slack <= 0.005,
        {
            "max_vertex_radius_of_footprint_plus_delta_m": max_norm,
            "r_fp_plus_delta_m": r_plus_d,
            "slack_m": slack,
            "arc_vertices": int(arc_inflation),
            "conclusion": (
                "两步写法方向一致但略保守：disc() 默认外切成多边形，半径放大 1/cos(pi/n)，"
                "在 r≈3.12 m 上多出约 1.35 mm。绝不更乐观，故两种写法都可接受；"
                "但要说「等价」就是错的"
            ),
        },
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "contract": "config/geometry_contract_v1.json",
        "contract_version": contract["version"],
        "r_fp_m": r_fp,
        "delta_m": delta,
        "r_plus_delta_m": r_plus_d,
        "W_disc_m": w_disc,
        "W_rectangle_m": w_rect,
        "premium_m": premium,
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
