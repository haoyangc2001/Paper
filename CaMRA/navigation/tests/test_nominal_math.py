"""Independent validation of the nominal environment's maths (NAV-019).

Nothing here reuses the environment's own helpers to check the environment.  The
references are:

* **analytic** — straight line, pure rotation, and rotation of the start point
  about the instantaneous centre, computed from geometry rather than from the
  frozen closed form;
* **RK4** — a numerical integration of the same ODE ``dp/dt = R(yaw) u``,
  ``dyaw/dt = wz``, stepping internally at ``dt/400``;
* **hand-built polynomial** — the power model rebuilt from ``weights`` and
  ``feature_names`` in this file, without importing ``PowerModel``;
* **counterexamples** — configurations where the wrong convention would give a
  different answer, so the check cannot pass by accident.

Tolerances are pre-registered below, not tuned after seeing the results.

Run:

    <frozen python> CaMRA/navigation/tests/test_nominal_math.py \
        --json CaMRA/navigation/reports/nominal_math_validation.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_calibration.geometry import Polygon  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.costs import DT_NAV_S, JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv, integrate  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import (  # noqa: E402
    SafetyGeometry,
    nominal_violation,
    nominal_violation_during_step,
)

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

#: Pre-registered tolerances.  ``ANALYTIC`` is pure floating-point round-off in
#: closed-form algebra; ``RK4`` is the O(h^3) global error at 400 substeps;
#: ``COST`` is a relative bound on a sum of ten products.
TOL_ANALYTIC = 1e-12
TOL_RK4 = 1e-9
TOL_COST_REL = 1e-12
TOL_SUM = 1e-9

RESULTS: list[dict[str, Any]] = []
SEED = 11


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


# ------------------------------------------------------------- references
def analytic_step(px, py, yaw, vx, vy, wz, dt):
    """Rotate the start point about the instantaneous centre, from geometry.

    Deliberately does not use the closed-form ``(sin(t1)-sin(t0))/w`` identity
    that the frozen integrator uses: this reference composes two rotations, so a
    sign slip in either implementation shows up as a disagreement.
    """
    if abs(wz) < 1e-15:
        c, s = math.cos(yaw), math.sin(yaw)
        return (px + (vx * c - vy * s) * dt, py + (vx * s + vy * c) * dt, yaw)
    speed = math.hypot(vx, vy)
    if speed == 0.0:
        return (px, py, yaw + wz * dt)
    phi = yaw + math.atan2(vy, vx)
    radius = speed / wz
    cx = px + radius * (-math.sin(phi))
    cy = py + radius * math.cos(phi)
    theta = wz * dt
    dx, dy = px - cx, py - cy
    nx = cx + dx * math.cos(theta) - dy * math.sin(theta)
    ny = cy + dx * math.sin(theta) + dy * math.cos(theta)
    return (nx, ny, yaw + theta)


def rk4_step(px, py, yaw, vx, vy, wz, dt, n=400):
    """Independent numerical integration of the same ODE."""

    def f(state):
        _x, _y, th = state
        return (
            vx * math.cos(th) - vy * math.sin(th),
            vx * math.sin(th) + vy * math.cos(th),
            wz,
        )

    h = dt / n
    s = [px, py, yaw]
    for _ in range(n):
        k1 = f(s)
        k2 = f([s[i] + 0.5 * h * k1[i] for i in range(3)])
        k3 = f([s[i] + 0.5 * h * k2[i] for i in range(3)])
        k4 = f([s[i] + h * k3[i] for i in range(3)])
        s = [s[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0 for i in range(3)]
    return tuple(s)


EXPECTED_FEATURES = [
    "bias", "abs_vx", "abs_vy", "abs_wz", "vx2", "vy2", "wz2",
    "abs_vx_wz", "abs_vy_wz", "abs_vx_vy",
]


def power_by_hand(weights, names, cmd) -> float:
    """Rebuild the frozen model from its coefficients, with no PowerModel call."""
    if names != EXPECTED_FEATURES:
        raise AssertionError(
            f"feature order changed: {names} != {EXPECTED_FEATURES}; the hand "
            "computation below would silently use the wrong terms"
        )
    vx, vy, wz = cmd
    table = {
        "bias": 1.0,
        "abs_vx": abs(vx),
        "abs_vy": abs(vy),
        "abs_wz": abs(wz),
        "vx2": vx * vx,
        "vy2": vy * vy,
        "wz2": wz * wz,
        "abs_vx_wz": abs(vx) * wz,
        "abs_vy_wz": abs(vy) * wz,
        "abs_vx_vy": abs(vx) * vy,
    }
    return sum(w * table[n] for w, n in zip(weights, names))


def build_env(spec, geom, model, *, z0_J: float) -> NominalNavEnv:
    return NominalNavEnv(spec, geom, model, z0_J=z0_J)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model_full = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    payload = json.load(open(POWER, encoding="utf-8"))
    weights = [float(w) for w in payload["weights"]]
    names = list(payload["feature_names"])

    # --- 1. straight line --------------------------------------------------
    cases = [
        (0.0, 0.0, 0.0, (0.6, 0.0, 0.0)),
        (2.0, -1.0, math.pi / 2, (0.4, 0.0, 0.0)),
        (-3.0, 2.5, -2.1, (1.0, 0.0, 0.0)),
    ]
    worst = 0.0
    for px, py, yaw, cmd in cases:
        got = integrate(px, py, yaw, cmd, DT_NAV_S)
        ref = analytic_step(px, py, yaw, *cmd, DT_NAV_S)
        worst = max(worst, abs(got.px - ref[0]), abs(got.py - ref[1]), abs(got.yaw - ref[2]))
    check(
        "straight_matches_analytic",
        worst <= TOL_ANALYTIC,
        {"max_abs_diff_m": worst, "tolerance": TOL_ANALYTIC, "cases": len(cases)},
    )

    # --- 2. pure rotation --------------------------------------------------
    got = integrate(1.5, -0.5, 0.3, (0.0, 0.0, 0.8), DT_NAV_S)
    check(
        "pure_rotation_matches_analytic",
        got.px == 1.5 and got.py == -0.5 and abs(got.yaw - (0.3 + 0.08)) <= TOL_ANALYTIC,
        {
            "pose_after": [got.px, got.py, got.yaw],
            "expected": [1.5, -0.5, 0.38],
            "note": "原地转向不应产生位移；位移若出现说明闭式解在 v=0 分支写错",
        },
    )

    # --- 3/5. arc, and a case with lateral velocity -----------------------
    arc_cases = [
        (0.0, 0.0, 0.0, (0.5, 0.0, 0.5)),
        (1.0, 2.0, 0.7, (0.5, 0.0, -1.0)),
        (-1.0, 0.5, 1.3, (0.3, 0.2, 0.9)),
    ]
    worst_arc = 0.0
    for px, py, yaw, cmd in arc_cases:
        got = integrate(px, py, yaw, cmd, DT_NAV_S)
        ref = analytic_step(px, py, yaw, *cmd, DT_NAV_S)
        worst_arc = max(worst_arc, abs(got.px - ref[0]), abs(got.py - ref[1]),
                        abs(got.yaw - ref[2]))
    # radius sanity: |u|/|w| = 1.0 for the first case
    r0 = math.hypot(*arc_cases[0][3][:2]) / abs(arc_cases[0][3][2])
    check(
        "arc_matches_geometric_reference",
        worst_arc <= TOL_ANALYTIC,
        {
            "max_abs_diff_m": worst_arc,
            "tolerance": TOL_ANALYTIC,
            "radius_first_case_m": r0,
            "reference": "绕瞬时中心的刚体旋转，与闭式解 sin/cos 恒等式不同源",
        },
    )

    # --- 4. RK4 ------------------------------------------------------------
    worst_rk4 = 0.0
    for px, py, yaw, cmd in arc_cases + cases:
        got = integrate(px, py, yaw, cmd, DT_NAV_S)
        ref = rk4_step(px, py, yaw, *cmd, DT_NAV_S)
        worst_rk4 = max(worst_rk4, abs(got.px - ref[0]), abs(got.py - ref[1]),
                        abs(got.yaw - ref[2]))
    check(
        "closed_form_matches_rk4",
        worst_rk4 <= TOL_RK4,
        {
            "max_abs_diff_m": worst_rk4,
            "tolerance": TOL_RK4,
            "substeps": 400,
            "note": "RK4 全局误差 O(h^3)，h=dt/400=2.5e-4，故 1e-9 足够严",
        },
    )

    # --- 6. yaw fold -------------------------------------------------------
    env = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=10000.0)
    yaw0 = env.pose[2]
    total = 0.0
    ys = []
    for _ in range(200):
        if env.done:
            break
        rec = env.step((0.3, 0.0, 1.0))
        total += 1.0 * DT_NAV_S
        ys.append(rec.pose_after[2])
    in_range = all(-math.pi < y < math.pi for y in ys)
    sin_err = max(abs(math.sin(y) - math.sin(yaw0 + total)) for y in ys[-1:])
    check(
        "yaw_folded_into_contract_range",
        in_range and abs(ys[-1] - math.remainder(yaw0 + total, 2 * math.pi)) < 1e-9,
        {
            "steps": len(ys),
            "unwrapped_total_rad": yaw0 + total,
            "final_yaw_rad": ys[-1],
            "all_within_open_pi_interval": in_range,
            "sin_residual": sin_err,
            "note": "冻结积分器不折回 yaw；状态边界处折回，否则长回合会漂到 5*pi",
        },
    )

    # --- 7. zero command ---------------------------------------------------
    env = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=10000.0)
    before = env.pose
    rec = env.step((0.0, 0.0, 0.0))
    idle_power = power_by_hand(weights, names, (0.0, 0.0, 0.0))
    check(
        "zero_command_holds_still_but_charges",
        rec.pose_after == before
        and rec.cost.cost_model_J > 0.0
        and abs(rec.cost.power_W - idle_power) <= TOL_COST_REL * max(1.0, idle_power),
        {
            "pose_unchanged": rec.pose_after == before,
            "power_W": rec.cost.power_W,
            "idle_power_W": idle_power,
            "cost_J": rec.cost.cost_model_J,
            "note": "静止仍耗能；若为 0 说明功率模型被当成纯运动项",
        },
    )

    # --- 8. extreme commands ----------------------------------------------
    env = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=10000.0)
    corner = env.step((1.0, 1.0, 1.0))
    env2 = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=10000.0)
    over = env2.step((1.5, -2.0, 0.5))
    expected_power = power_by_hand(weights, names, (1.0, 0.4, 1.0))
    check(
        "domain_corners_reachable_and_clipped",
        corner.command.applied == (1.0, 0.4, 1.0)
        and not corner.command.clipped
        and over.command.requested == (1.0, -0.4, 0.5)
        and over.command.clipped
        and abs(corner.cost.power_W - expected_power) <= TOL_COST_REL * expected_power,
        {
            "corner_applied": list(corner.command.applied),
            "corner_clipped": corner.command.clipped,
            "corner_power_W": corner.cost.power_W,
            "hand_power_W": expected_power,
            "over_requested": list(over.command.requested),
            "over_clipped": over.command.clipped,
        },
    )

    # --- 9. non-finite action ---------------------------------------------
    env = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=10000.0)
    bad = env.step((float("nan"), float("inf"), 0.3))
    ob = env.observation()
    check(
        "nonfinite_action_sanitised",
        bad.command.invalid
        and bad.command.applied[0] == 0.0
        and bad.command.applied[1] == 0.0
        and all(math.isfinite(v) for v in ob)
        and all(math.isfinite(v) for v in bad.pose_after),
        {
            "invalid_flag": bad.command.invalid,
            "applied": list(bad.command.applied),
            "observation_finite": all(math.isfinite(v) for v in ob),
            "note": "NaN/Inf 归零并置标志，绝不进入状态或观测",
        },
    )

    # --- 10. power polynomial ---------------------------------------------
    env = build_env(M.build("L3_multi_path", SEED, design), geom, model_full, z0_J=5000.0)
    cmds = [(0.6, 0.0, 0.0), (0.2, 0.1, -0.3), (-1.0, 0.4, 1.0), (0.0, 0.0, 0.0)]
    worst_rel = 0.0
    rows = []
    for cmd in cmds:
        env.reset()
        rec = env.step(cmd)
        hand = power_by_hand(weights, names, rec.command.applied)
        rel = abs(rec.cost.power_W - hand) / max(1e-12, abs(hand))
        worst_rel = max(worst_rel, rel)
        rows.append(
            {"applied": list(rec.command.applied), "env_W": rec.cost.power_W, "hand_W": hand}
        )
    check(
        "power_matches_independent_polynomial",
        worst_rel <= TOL_COST_REL,
        {
            "worst_relative_error": worst_rel,
            "tolerance": TOL_COST_REL,
            "feature_names": names,
            "rows": rows,
            "note": "本文件用 weights×features 重算，不导入 PowerModel",
        },
    )

    # --- 11. budget arithmetic --------------------------------------------
    env = build_env(M.build("L2_single_detour", SEED, design), geom, model_full, z0_J=5000.0)
    seq = [(0.5, 0.0, 0.0)] * 10 + [(0.2, 0.1, -0.3)] * 10 + [(0.8, 0.0, 0.2)] * 10
    manual = 0.0
    for a in seq:
        if env.done:
            break
        rec = env.step(a)
        manual += (1.0 + model_full.alpha) * power_by_hand(weights, names, rec.command.applied) * DT_NAV_S
    check(
        "budget_equals_independent_sum",
        abs((env.z0_J - env.budget.z_J) - manual) <= TOL_SUM,
        {
            "charged_J": env.z0_J - env.budget.z_J,
            "hand_sum_J": manual,
            "abs_diff_J": abs((env.z0_J - env.budget.z_J) - manual),
            "steps": env.index,
            "alpha": model_full.alpha,
        },
    )

    # --- 12. safety history counterexample --------------------------------
    env = build_env(M.build("L2_single_detour", SEED, design), geom, model_full, z0_J=5000.0)
    first_y1 = None
    for i in range(80):
        if env.done:
            break
        rec = env.step((0.6, 0.0, 0.0))
        if rec.safety_y >= 0.5 and first_y1 is None:
            first_y1 = i
    # retreat: leave the band and confirm y does not clear
    retreat = []
    for _ in range(20):
        if env.done:
            break
        rec = env.step((-0.6, 0.0, 0.0))
        retreat.append(rec.safety_y)
    check(
        "safety_history_counterexample",
        first_y1 is not None and retreat and all(v >= 0.5 for v in retreat)
        and env.budget.z_J > 0.0,
        {
            "first_step_with_y1": first_y1,
            "retreat_steps": len(retreat),
            "retreat_y_values": retreat[:5],
            "label": env.label,
            "note": "退出 delta 带后 y 仍为 1；若回到 0 说明历史被写成了瞬时标志",
        },
    )

    # --- 13. interval vs endpoint agreement -------------------------------
    env = build_env(M.build("L2_single_detour", SEED, design), geom, model_full, z0_J=1e9)
    polys = tuple(ob.polygon() for ob in env.spec.obstacles())
    rng = np.random.default_rng(20261110)
    disagree = 0
    tested = 0
    max_step = 0.0
    for _ in range(200):
        px = float(rng.uniform(-6.0, 6.0))
        py = float(rng.uniform(-6.0, 6.0))
        yaw = float(rng.uniform(-math.pi, math.pi))
        cmd = (
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(-1.0, 1.0)),
        )
        nxt = integrate(px, py, yaw, cmd, DT_NAV_S)
        max_step = max(max_step, math.hypot(float(nxt.px) - px, float(nxt.py) - py))
        for poly in polys:
            endpoint = nominal_violation((px, py), poly, geom) or nominal_violation(
                (float(nxt.px), float(nxt.py)), poly, geom
            )
            segment = nominal_violation_during_step(
                (px, py), (float(nxt.px), float(nxt.py)), poly, geom
            )
            tested += 1
            if endpoint != segment:
                disagree += 1
    check(
        "interval_and_endpoint_agree_at_task_scale",
        disagree == 0 and tested > 0,
        {
            "comparisons": tested,
            "disagreements": disagree,
            "max_step_m": max_step,
            "band_thickness_m": 2.0 * geom.inflation_radius_m,
            "ratio_band_over_step": (2.0 * geom.inflation_radius_m) / max(max_step, 1e-12),
            "note": ("禁入带厚 2*(r_fp+delta) 远大于单步位移，故逐步端点判据与段判据在本任务尺度等价；"
                     "段判据仍保留在实现里作为保险"),
        },
    )

    # --- 14. vertex-free overlap ------------------------------------------
    # A long thin wall through the footprint: no footprint vertex lies inside the
    # wall and no wall vertex lies inside the footprint, yet they overlap.  This is
    # the case the frozen polygon_distance misses.
    wall_island = (0.0, 0.0, 4.0, 0.02, 0.0)
    spec = M.MapSpec(
        map_id="synthetic-vertex-free",
        level="synthetic",
        seed=0,
        extent=(-11.0, 11.0, -6.0, 6.0),
        start=(0.0, 0.0),
        start_yaw=0.0,
        goal=(8.0, 0.0),
        islands=(wall_island,),
        design_version="synthetic",
    )
    geom_zero = dataclasses.replace(geom, delta_m=0.0)
    env = build_env(spec, geom_zero, model_full, z0_J=5000.0)
    rec = env.step((0.1, 0.0, 0.0))
    check(
        "vertex_free_overlap_labelled_collision",
        rec.label == "collision" and rec.min_physical_clearance_m == 0.0,
        {
            "label": rec.label,
            "clearance_m": rec.min_physical_clearance_m,
            "island": {"width": wall_island[2], "depth": wall_island[3]},
            "note": "机身跨在 4x0.02 m 薄墙上，两多边形互不含对方顶点，仍被正确判为碰撞",
        },
    )

    # --- 15. episode isolation --------------------------------------------
    env = build_env(M.build("L2_single_detour", SEED, design), geom, model_full, z0_J=200.0)
    for _ in range(200):
        if env.done:
            break
        env.step((0.6, 0.0, 0.0))
    trace1 = json.dumps(env.trace(), sort_keys=True)
    state1 = (env.pose, env.budget.z_J, env.safety_y, env.index, env.label)
    env.reset()
    fresh = (env.pose, env.budget.z_J, env.safety_y, env.index, env.label)
    for _ in range(5):
        env.step((0.3, 0.0, 0.0))
    trace2 = json.dumps(env.trace(), sort_keys=True)
    check(
        "episode_isolation",
        trace1 != trace2
        and fresh[1] == env.z0_J
        and fresh[2] == 0.0
        and fresh[3] == 0
        and fresh[4] is None
        and "l2_single_detour" in trace2,
        {
            "before_reset": {
                "label": state1[4],
                "z_J": state1[1],
                "safety_y": state1[2],
                "steps": state1[3],
            },
            "after_reset": {"z_J": fresh[1], "safety_y": fresh[2], "steps": fresh[3],
                            "label": fresh[4]},
            "new_episode_steps": env.index,
            "note": "上一回合的账本字符串在下一回合运行后不变，且新回合从 index=0 起",
        },
    )

    # --- 16. truncation ----------------------------------------------------
    env = build_env(M.build("L1_empty", SEED, design), geom, model_full, z0_J=1e9)
    for _ in range(400):
        if env.done:
            break
        env.step((0.3, 0.0, 0.5))
    check(
        "truncation_exact_at_horizon",
        env.label == "timeout" and env.index == 300 and abs(env.t_s - 30.0) < 1e-9,
        {
            "label": env.label,
            "steps": env.index,
            "t_s": env.t_s,
            "note": "300 x 0.1 s 恰好到 30 s；不应多走一步也不应提前截断",
        },
    )

    # --- 17. world -> body -------------------------------------------------
    worst_rot = 0.0
    for yaw in (0.0, 0.3, -1.1, math.pi - 1e-6, -math.pi + 1e-6, 2.5):
        for dx, dy in ((1.0, 0.0), (0.0, 1.0), (-0.4, 0.7)):
            bx, by = obs.world_to_body_vector(yaw, dx, dy)
            c, s = math.cos(yaw), math.sin(yaw)
            ref_x = c * dx + s * dy
            ref_y = -s * dx + c * dy
            worst_rot = max(worst_rot, abs(bx - ref_x), abs(by - ref_y))
    check(
        "world_to_body_matches_rotation_matrix",
        worst_rot <= TOL_ANALYTIC,
        {
            "max_abs_diff": worst_rot,
            "tolerance": TOL_ANALYTIC,
            "note": "用显式 R^T 复核，含接近 ±pi 的航向",
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-019",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "tolerances": {
            "analytic": TOL_ANALYTIC,
            "rk4": TOL_RK4,
            "cost_relative": TOL_COST_REL,
            "sum": TOL_SUM,
        },
        "references": [
            "analytic: geometric rotation about the instantaneous centre",
            "rk4: 400-substep numerical integration of the same ODE",
            "hand polynomial: weights x feature_names, no PowerModel import",
        ],
        "checks": RESULTS,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:400])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
