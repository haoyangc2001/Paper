"""Nominal environment checks for NAV-018.

The environment must be deterministic, must reuse the frozen integrator, must
charge the frozen power model exactly once per interval, must label each interval
at most once following the frozen priority order, and must never reset itself.

Run:

    <frozen python> CaMRA/navigation/tests/test_nominal_env.py \
        --json CaMRA/navigation/reports/nominal_env_check.json
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

from camra_calibration.nominal import PlanarState, step_exact  # noqa: E402
from camra_calibration.power import PowerModel  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.costs import DT_NAV_S, JCostModel  # noqa: E402
from camra_navigation.envs.nominal import (  # noqa: E402
    EpisodeOverError,
    NominalNavEnv,
    integrate,
    preset_actions,
)
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

RESULTS: list[dict[str, Any]] = []
SEED = 11


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def build_env(level: str, *, z0_J: float, variant: str = "full", seed: int = SEED, design=None,
              spec_override=None, jitter: float = 0.0) -> NominalNavEnv:
    design = design or M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = JCostModel.from_frozen(POWER, MARGINS, variant=variant)
    spec = spec_override or M.build(level, seed, design)
    return NominalNavEnv(
        spec, geom, model, z0_J=z0_J, initial_pose_jitter_m=jitter
    )


def run(env: NominalNavEnv, actions: list[tuple[float, float, float]]) -> list[dict]:
    out = []
    for a in actions:
        if env.done:
            break
        out.append(env.step(a).as_dict())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model_full = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    power = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))

    # --- 1. observation version and dimension -----------------------------
    check(
        "obs_version_widened_to_v2",
        obs.OBS_VERSION == "obs-v2" and obs.OBS_DIM == 41,
        {
            "obs_version": obs.OBS_VERSION,
            "dim": obs.OBS_DIM,
            "why": "NAV-016 froze rectangular islands; the contract forbids "
                   "encoding a polygon through a radius, so the primitive set grew",
            "unchanged": "layout, dimension and every constant",
        },
    )

    # --- 2. the frozen integrator is the one used -------------------------
    cases = [
        (0.0, 0.0, 0.0, (0.6, 0.0, 0.0)),
        (1.0, -2.0, 0.3, (0.4, 0.1, -0.7)),
        (-3.5, 0.5, math.pi / 2, (0.0, -0.4, 0.0)),
        (2.0, 2.0, -1.2, (1.0, 0.0, 0.9)),
    ]
    worst = 0.0
    for px, py, yaw, cmd in cases:
        got = integrate(px, py, yaw, cmd, DT_NAV_S)
        ref = step_exact(PlanarState(px, py, yaw), cmd, DT_NAV_S)
        worst = max(worst, abs(got.px - ref.px), abs(got.py - ref.py), abs(got.yaw - ref.yaw))
    straight = integrate(0.0, 0.0, 0.0, (0.6, 0.0, 0.0), DT_NAV_S)
    check(
        "uses_frozen_integrator",
        worst == 0.0 and abs(straight.px - 0.06) < 1e-15,
        {
            "max_abs_diff_vs_step_exact": worst,
            "straight_0p6_for_0p1s_dx_m": straight.px,
            "note": "逐位相同；直线退化情形 dx = vx*dt = 0.06 m",
        },
    )

    # --- 3. determinism: same seed and actions -> identical trace ---------
    env_a = build_env("L2_single_detour", z0_J=5000.0)
    env_b = build_env("L2_single_detour", z0_J=5000.0)
    actions = preset_actions("mixed", 40)
    run(env_a, actions)
    run(env_b, actions)
    ta = json.dumps(env_a.trace(), sort_keys=True, ensure_ascii=False)
    tb = json.dumps(env_b.trace(), sort_keys=True, ensure_ascii=False)
    check(
        "deterministic_replay",
        ta == tb and env_a.index == env_b.index and env_a.label == env_b.label,
        {
            "steps": env_a.index,
            "label": env_a.label,
            "trace_bytes": len(ta),
            "identical": ta == tb,
        },
    )

    # --- 4. seed controls the initial pose when jitter is enabled ---------
    e1 = build_env("L1_empty", z0_J=5000.0, jitter=0.05)
    e2 = build_env("L1_empty", z0_J=5000.0, jitter=0.05)
    e1.reset(seed=7)
    p7a = e1.pose
    e1.reset(seed=7)
    p7b = e1.pose
    e1.reset(seed=8)
    p8 = e1.pose
    plain = build_env("L1_empty", z0_J=5000.0)
    check(
        "seed_reproduces_initial_pose",
        p7a == p7b and p7a != p8 and plain.pose == (
            float(plain.spec.start[0]), float(plain.spec.start[1]), float(plain.spec.start_yaw)
        ),
        {
            "seed7_first": list(p7a),
            "seed7_second": list(p7b),
            "seed8": list(p8),
            "jitter_disabled_equals_spec_start": list(plain.pose),
        },
    )

    # --- 5/6. cost is the frozen power model, charged exactly once --------
    env = build_env("L1_empty", z0_J=5000.0)
    rec = env.step((0.6, 0.0, 0.0))
    expected_power = float(power.step_power([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])[0])
    expense = env.result()
    check(
        "cost_equals_frozen_power_once",
        abs(rec.cost.power_W - expected_power) < 1e-12
        and abs(rec.cost.cost_nom_J - expected_power * DT_NAV_S) < 1e-12
        and abs(rec.cost.cost_model_J - (1.0 + model_full.alpha) * rec.cost.cost_nom_J) < 1e-12,
        {
            "applied": list(rec.command.applied),
            "power_W": rec.cost.power_W,
            "frozen_power_W": expected_power,
            "cost_nom_J": rec.cost.cost_nom_J,
            "cost_model_J": rec.cost.cost_model_J,
            "alpha": model_full.alpha,
        },
    )
    env2 = build_env("L1_empty", z0_J=5000.0)
    run(env2, preset_actions("mixed", 30))
    res = env2.result()
    charged = env2.z0_J - env2.budget.z_J
    check(
        "budget_matches_step_costs",
        abs(charged - res.cost_model_J) < 1e-9
        and abs(res.cost_model_J - sum(r.cost.cost_model_J for r in env2.steps)) < 1e-9,
        {
            "z0_J": env2.z0_J,
            "z_J": env2.budget.z_J,
            "charged_J": charged,
            "sum_step_costs_J": res.cost_model_J,
            "steps": res.n_steps,
        },
    )

    # --- 7. safety history is irreversible --------------------------------
    env3 = build_env("L2_single_detour", z0_J=5000.0)
    ys = []
    for _ in range(60):
        if env3.done:
            break
        s = env3.step((0.6, 0.0, 0.0))
        ys.append(s.safety_y)
    check(
        "safety_history_irreversible",
        all(b >= a for a, b in zip(ys, ys[1:])) and ys[-1] >= 0.0,
        {
            "first_index_with_y1": next((i for i, v in enumerate(ys) if v >= 0.5), None),
            "n_steps": len(ys),
            "monotone": all(b >= a for a, b in zip(ys, ys[1:])),
            "label": env3.label,
        },
    )

    # --- 8. internal_violation does not become a label --------------------
    env4 = build_env("L2_single_detour", z0_J=5000.0)
    flagged = []
    for _ in range(60):
        if env4.done:
            break
        s = env4.step((0.6, 0.0, 0.0))
        if s.danger_violation:
            flagged.append((s.index, s.label, s.danger_violation, s.internal_violation))
    first = flagged[0] if flagged else None
    check(
        "internal_violation_is_orthogonal",
        first is not None and first[1] is None and first[2] and first[3],
        {
            "first_flagged_step": first,
            "note": "进入 delta 带时只置标志与 safety_y，不改 label；此处随后才因碰撞结束",
            "label_at_end": env4.label,
            "flags_raise_no_label": all(f[1] is None for f in flagged[:-1]) if len(flagged) > 1 else None,
        },
    )

    # --- 9. no automatic reset after the episode ends ---------------------
    env5 = build_env("L2_single_detour", z0_J=5000.0)
    for _ in range(200):
        if env5.done:
            break
        env5.step((0.6, 0.0, 0.0))
    pose_at_end = env5.pose
    raised = False
    try:
        env5.step((0.6, 0.0, 0.0))
    except EpisodeOverError:
        raised = True
    check(
        "no_auto_reset_after_done",
        env5.done and raised and env5.pose == pose_at_end,
        {
            "label": env5.label,
            "steps": env5.index,
            "raised": raised,
            "pose_unchanged": env5.pose == pose_at_end,
        },
    )

    # --- 10-15. each reachable label, and only under its own condition ----
    # collision: straight into the central island of L2
    e = build_env("L2_single_detour", z0_J=5000.0)
    for _ in range(300):
        if e.done:
            break
        e.step((0.6, 0.0, 0.0))
    collision = {"label": e.label, "steps": e.index, "pose": list(e.pose),
                 "clearance_to_island_m": e.steps[-1].min_physical_clearance_m}

    # goal_reached: start 0.3 m from the goal
    base = M.build("L1_empty", SEED, design)
    near = dataclasses.replace(base, start=(base.goal[0] - 0.3, base.goal[1]), start_yaw=0.0)
    e = build_env("L1_empty", z0_J=5000.0, spec_override=near)
    e.step((0.6, 0.0, 0.0))
    goal = {"label": e.label, "steps": e.index, "goal_distance_m": e.steps[-1].goal_distance_m}

    # budget_exhausted: 1 J cannot pay one 0.6 m/s interval (~2 J at alpha=0.19)
    e = build_env("L1_empty", z0_J=1.0)
    e.step((0.6, 0.0, 0.0))
    budget_lbl = {"label": e.label, "z_J": e.budget.z_J, "overshoot_J": -e.budget.z_J,
                  "cost_model_J": e.steps[-1].cost.cost_model_J}

    # stagnation: zero command, plenty of budget, 3.0 s window
    e = build_env("L1_empty", z0_J=10000.0)
    for _ in range(60):
        if e.done:
            break
        e.step((0.0, 0.0, 0.0))
    stagn = {"label": e.label, "t_s": e.t_s, "travelled_m": e.travelled_m}

    # timeout: circling in open space for the full 30 s
    e = build_env("L1_empty", z0_J=10000.0)
    for _ in range(400):
        if e.done:
            break
        e.step((0.5, 0.0, 0.5))
    timeout = {"label": e.label, "t_s": e.t_s, "steps": e.index}

    # out_of_bounds: spawned beyond the wall band, driving away
    far = dataclasses.replace(base, start=(base.extent[1] + 5.0, 0.0), start_yaw=0.0)
    e = build_env("L1_empty", z0_J=5000.0, spec_override=far)
    e.step((0.6, 0.0, 0.0))
    oob = {"label": e.label, "pose": list(e.pose), "extent": list(far.extent)}

    check(
        "labels_match_frozen_priority",
        collision["label"] == "collision"
        and goal["label"] == "goal_reached"
        and budget_lbl["label"] == "budget_exhausted"
        and stagn["label"] == "stagnation"
        and timeout["label"] == "timeout"
        and oob["label"] == "out_of_bounds",
        {
            "collision": collision,
            "goal_reached": goal,
            "budget_exhausted": budget_lbl,
            "stagnation": stagn,
            "timeout": timeout,
            "out_of_bounds": oob,
        },
    )

    # --- 16. goal beats budget on the same interval -----------------------
    tiny = dataclasses.replace(base, start=(base.goal[0] - 0.4, base.goal[1]), start_yaw=0.0)
    e = build_env("L1_empty", z0_J=2.0, spec_override=tiny)
    s = e.step((0.6, 0.0, 0.0))
    check(
        "goal_priority_over_budget",
        s.label == "goal_reached" and s.z_J <= 0.0 and s.goal_distance_m <= e.r_goal_m,
        {
            "label": s.label,
            "z_J_after": s.z_J,
            "goal_distance_m": s.goal_distance_m,
            "r_goal_m": e.r_goal_m,
            "note": "同一区间既到达又超预算时按优先级取 goal_reached",
        },
    )

    # --- 17. observation equals the frozen encoder ------------------------
    e = build_env("L3_multi_path", z0_J=5000.0)
    run(e, preset_actions("mixed", 12))
    got = e.observation()
    ref = obs.encode_observation(e.ego, e._obstacle_obs, e._extent, e.task)
    check(
        "observation_matches_frozen_encoder",
        got == ref and len(got) == obs.OBS_DIM and all(math.isfinite(v) for v in got),
        {
            "dim": len(got),
            "same_object": got == ref,
            "z_ratio": got[obs.field_slices()["z_ratio"]][0],
            "safety_y": got[obs.field_slices()["safety_y"]][0],
        },
    )

    # --- 18. polygon primitive distance and bearing -----------------------
    island = M.build("L2_single_detour", SEED, design).obstacles()[0]
    poly_obs = obs.obstacle_from_polygon(island.polygon().vertices)
    ego = obs.EgoState(px=0.0, py=4.0, yaw=-math.pi / 2, vx_body=0.0, vy_body=0.0, wz_body=0.0)
    surf, bearing = obs.surface_distance(ego, poly_obs)
    exact = float(island.polygon().distance([[0.0, 4.0]])[0])
    circle = obs.Obstacle(0.0, 0.0, 1.0)  # same nearest face point (0, 1) as the rectangle
    surf_c, bearing_c = obs.surface_distance(ego, circle)
    check(
        "polygon_primitive_distance_and_bearing",
        abs(surf - 3.0) < 1e-9 and abs(surf - exact) < 1e-9 and abs(bearing) < 1e-9
        and abs(surf_c - surf) < 1e-9 and abs(bearing_c - bearing) < 1e-9,
        {
            "surface_distance_m": surf,
            "frozen_point_distance_m": exact,
            "bearing_rad": bearing,
            "circle_equivalent_surface_m": surf_c,
            "circle_equivalent_bearing_rad": bearing_c,
            "note": "矩形最近面在正前方，方位角为 0；圆与矩形在该姿态下取到同一最近点",
        },
    )

    # --- 19. presets run on all three levels ------------------------------
    per_level = {}
    ok = True
    for level in ("L1_empty", "L2_single_detour", "L3_multi_path"):
        e = build_env(level, z0_J=5000.0)
        steps = run(e, preset_actions("forward", 20))
        per_level[level] = {
            "steps": len(steps),
            "t_s": e.t_s,
            "pose": list(e.pose),
            "cost_model_J": e.result().cost_model_J,
            "label": e.label,
            "obs_dim": len(e.observation()),
        }
        ok = ok and len(steps) == 20 and len(e.observation()) == obs.OBS_DIM
    check("presets_run_on_all_levels", ok, per_level)

    # --- 20. trace is strict JSON ------------------------------------------
    strict_ok = True
    err = ""
    try:
        json.dumps(env_a.trace(), allow_nan=False)
    except ValueError as exc:  # NaN / Infinity would break downstream readers
        strict_ok = False
        err = str(exc)
    check(
        "trace_is_strict_json",
        strict_ok and env_a.result().physical_status == "not_measured",
        {
            "strict_json": strict_ok,
            "error": err,
            "physical_status": env_a.result().physical_status,
            "note": "无障碍地图的净空写 null，不写 Infinity",
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-018",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "obs_version": obs.OBS_VERSION,
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
