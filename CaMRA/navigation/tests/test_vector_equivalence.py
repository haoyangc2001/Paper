"""Bitwise equivalence: VectorNominalNavEnv vs NominalNavEnv (NAV-036).

Acceptance clause of the preregistered vectorised environment
(config/experiment_prereg_v1.json): on the same (level, seed) and the same
command sequence, the vectorised env must reproduce the scalar env's poses,
costs, labels, z, safety history and the full observation vector **bit for
bit**, step by step.  No tolerance is applied anywhere in this file: floats
are compared with ``==``.

Also measures aggregate throughput (env-steps/s) against the authorised
training quota's requirement (47.4 steps/s = 1,024,000 steps / 6 h).

Run:

    <frozen python> CaMRA/navigation/tests/test_vector_equivalence.py \
        --json CaMRA/navigation/reports/vector_equivalence_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402

from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.envs.vector import VectorNominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


# ------------------------------------------------------------ command plans
def plan_random(seed: int, n: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    return [tuple(float(v) for v in rng.uniform(-1.0, 1.0, size=3)) for _ in range(n)]


def plan_idle_then_mixed(n_idle: int, n_mixed: int) -> list[tuple[float, float, float]]:
    out = [(0.0, 0.0, 0.0)] * n_idle
    for i in range(n_mixed):
        if i % 4 == 0:
            out.append((0.8, 0.0, 0.0))
        elif i % 4 == 1:
            out.append((0.5, 0.1, -0.2))
        elif i % 4 == 2:
            out.append((0.2, 0.0, 0.0))
        else:
            out.append((0.0, 0.0, 0.4))
    return out


def plan_turn_drift(n: int) -> list[tuple[float, float, float]]:
    return [(0.4, 0.0, 0.3)] * n


PLANS = {
    "random": lambda seed, cap: plan_random(seed, cap),
    "idle_then_mixed": lambda seed, cap: plan_idle_then_mixed(10, max(0, cap - 10)),
    "turn_drift": lambda seed, cap: plan_turn_drift(cap),
    "pure_idle": lambda seed, cap: [(0.0, 0.0, 0.0)] * cap,
}


# ------------------------------------------------------- scalar reference run
def run_scalar(spec, plan, *, seed: int, z0_J: float, geom, cm,
               t_max_s: float) -> dict[str, Any]:
    env = NominalNavEnv(spec, geom, cm, z0_J=z0_J, t_max_s=t_max_s)
    obs0 = env.reset(seed=seed)
    steps = []
    obs_trace = [list(obs0)]
    for action in plan:
        rec = env.step(action)
        steps.append(rec)
        obs_trace.append(env.observation())
        if env.done:
            break
    return {
        "steps": steps,
        "obs": obs_trace,
        "label": env.label,
        "n_steps": env.index,
        "result": env.result(),
    }


# ------------------------------------------------------- vector run + compare
def run_vector_and_compare(
    name: str, level: str, spec, seeds: list[int], plan_names: list[str],
    *, z0_J: float, geom, cm, t_max_s: float, cap: int,
) -> bool:
    """Drive ``N`` lockstep vector episodes against per-env scalar runs."""
    n = len(seeds)
    plan_lists = [PLANS[plan_names[i]](seeds[i], cap) for i in range(n)]
    scalar = {
        i: run_scalar(spec, plan_lists[i], seed=seeds[i],
                      z0_J=z0_J, geom=geom, cm=cm, t_max_s=t_max_s)
        for i in range(n)
    }
    vec = VectorNominalNavEnv(
        spec, geom, cm, n_envs=n, z0_J=z0_J, t_max_s=t_max_s,
        compute_margins=True,
    )
    vec.reset(seeds=seeds)

    problems: list[str] = []
    counts = {"fields": 0, "obs": 0, "degenerate_steps": 0}

    def same(tag: str, i: int, t: int, a, b) -> None:
        counts["fields"] += 1
        if a != b:
            problems.append(f"{tag}[env {i} step {t}]: scalar={a!r} vector={b!r}")

    # reset observations
    for i in range(n):
        s0 = scalar[i]["obs"][0]
        v0 = vec.observation[i]
        for d in range(len(s0)):
            same("obs_reset", i, 0, float(s0[d]), float(v0[d]))
        counts["obs"] += len(s0)

    n_scalar_steps = [len(scalar[i]["steps"]) for i in range(n)]
    max_steps = max(n_scalar_steps)
    for t in range(max_steps):
        acts = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            if t < len(plan_lists[i]):
                acts[i] = plan_lists[i][t]
        rec = vec.step(acts)
        for i in range(n):
            if t >= n_scalar_steps[i]:
                continue  # scalar episode already ended
            if not rec.active[i]:
                problems.append(f"active[env {i} step {t}]: scalar still running "
                                "but vector env reports done")
                continue
            s = scalar[i]["steps"][t]
            if (s.pose_after[0] == s.pose_before[0]
                    and s.pose_after[1] == s.pose_before[1]):
                counts["degenerate_steps"] += 1
            same("index", i, t, s.index, int(rec.index[i]))
            same("t_s", i, t, s.t_s, float(rec.t_s[i]))
            for c in range(3):
                same("requested", i, t, s.command.requested[c],
                     float(rec.requested[i, c]))
                same("applied", i, t, s.command.applied[c], float(rec.applied[i, c]))
            same("clipped", i, t, s.command.clipped, bool(rec.clipped[i]))
            same("invalid", i, t, s.command.invalid, bool(rec.invalid[i]))
            for c, fname in enumerate(("px", "py", "yaw")):
                same(f"pose_after.{fname}", i, t, s.pose_after[c],
                     float(rec.pose_after[i, c]))
            same("power_W", i, t, s.cost.power_W, float(rec.power_W[i]))
            same("cost_nom_J", i, t, s.cost.cost_nom_J, float(rec.cost_nom_J[i]))
            same("cost_model_J", i, t, s.cost.cost_model_J, float(rec.cost_model_J[i]))
            same("cost_t_s", i, t, s.cost.t_s, float(rec.cost_t_s[i]))
            same("z_J", i, t, s.z_J, float(rec.z_J[i]))
            same("z_ratio", i, t, s.z_ratio, float(rec.z_ratio[i]))
            same("safety_y", i, t, s.safety_y, float(rec.safety_y[i]))
            same("danger", i, t, s.danger_violation, bool(rec.danger_violation[i]))
            same("internal", i, t, s.internal_violation,
                 bool(rec.internal_violation[i]))
            same("goal_distance_m", i, t, s.goal_distance_m,
                 float(rec.goal_distance_m[i]))
            same("dropped_obstacles", i, t, s.dropped_obstacles, rec.dropped_obstacles)
            same("min_nominal_margin_m", i, t, s.min_nominal_margin_m,
                 None if math.isnan(rec.min_nominal_margin_m[i])
                 else float(rec.min_nominal_margin_m[i]))
            same("min_physical_clearance_m", i, t, s.min_physical_clearance_m,
                 None if math.isnan(rec.min_physical_clearance_m[i])
                 else float(rec.min_physical_clearance_m[i]))
            same("label", i, t, s.label, rec.label[i])
            same("done", i, t, s.done, bool(rec.done[i]))
            s_obs = scalar[i]["obs"][t + 1]
            v_obs = rec.obs[i]
            for d in range(len(s_obs)):
                same("obs", i, t, float(s_obs[d]), float(v_obs[d]))
            counts["obs"] += len(s_obs)

    # episode tails: label, step count and result fields must match
    tails: dict[str, Any] = {}
    for i in range(n):
        rs, rv = scalar[i]["result"], vec.result(i)
        if scalar[i]["label"] != rv.label:
            problems.append(f"final_label[env {i}]: scalar={scalar[i]['label']!r} "
                            f"vector={rv.label!r}")
        if scalar[i]["n_steps"] != rv.n_steps:
            problems.append(f"n_steps[env {i}]: scalar={scalar[i]['n_steps']} "
                            f"vector={rv.n_steps}")
        for f in ("travelled_m", "t_s", "danger_steps", "internal_violation_steps",
                  "cost_model_J", "z_J", "safety_y"):
            if getattr(rs, f) != getattr(rv, f):
                problems.append(f"result.{f}[env {i}]: scalar={getattr(rs, f)!r} "
                                f"vector={getattr(rv, f)!r}")
        tails[f"env{i}"] = {"label": scalar[i]["label"], "n_steps": scalar[i]["n_steps"]}

    ok = not problems
    check(f"bitwise_equivalence::{name}", ok, {
        "level": level, "seeds": seeds, "plans": plan_names, "z0_J": z0_J,
        "compared_fields": counts["fields"], "compared_obs_values": counts["obs"],
        "degenerate_steps_covered": counts["degenerate_steps"],
        "tails": tails, "n_problems": len(problems),
        "first_problems": problems[:6],
    })
    return ok


def throughput_check() -> None:
    """Aggregate env-steps/s at N=16 on the hardest level vs the 47.4 quota."""
    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    spec = M.build("L3_multi_path", 1001, design)
    n = 16
    vec = VectorNominalNavEnv(spec, geom, cm, n_envs=n, z0_J=550.0)
    vec.reset(seeds=list(range(1001, 1001 + n)))
    rng = np.random.default_rng(7)
    wall_cap = 25.0
    calls = 0
    t0 = time.time()
    while time.time() - t0 < wall_cap and not vec.done.all():
        vec.step(rng.uniform(-1.0, 1.0, size=(n, 3)))
        calls += 1
    elapsed = time.time() - t0
    env_steps = calls * n
    rate = env_steps / elapsed
    check(
        "throughput_aggregate_ge_47.4",
        rate >= 47.4,
        {"level": "L3_multi_path", "n_envs": n, "lockstep_calls": calls,
         "env_steps": env_steps, "elapsed_s": round(elapsed, 2),
         "aggregate_steps_per_s": round(rate, 1),
         "requirement": ">= 47.4 (1,024,000 steps / 6 h authorized quota)"},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="vector-vs-scalar bitwise equivalence")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    cm_uncal = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    cm_full = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    z0 = 550.0
    cap = 130  # enough for stagnation (30 idle steps) and long wander

    cases = [
        ("L1_random6", "L1_empty", [11, 1001, 2026, 3407, 777, 4242],
         ["random"] * 6, cm_uncal),
        ("L2_mixed_plans", "L2_single_detour", [11, 1001, 2026, 3407],
         ["idle_then_mixed", "random", "turn_drift", "random"], cm_uncal),
        ("L3_mixed_plans", "L3_multi_path", [11, 1001, 2026, 3407, 777],
         ["random", "idle_then_mixed", "turn_drift", "random", "idle_then_mixed"],
         cm_uncal),
        ("L1_full_variant", "L1_empty", [11, 2026], ["random", "turn_drift"],
         cm_full),
        ("L2_pure_idle", "L2_single_detour", [11, 1001],
         ["pure_idle", "pure_idle"], cm_uncal),
    ]
    all_ok = True
    for name, level, seeds, plans, cm in cases:
        spec = M.build(level, seeds[0], design)
        all_ok &= run_vector_and_compare(name, level, spec, seeds, plans,
                                         z0_J=z0, geom=geom, cm=cm,
                                         t_max_s=30.0, cap=cap)

    throughput_check()

    verdict = "PASS" if all_ok else "FAIL"
    report = {
        "task": "NAV-036",
        "check": "vector_nominal_env_bitwise_equivalence",
        "verdict": verdict,
        "n_checks": len(RESULTS),
        "n_passed": sum(1 for r in RESULTS if r["ok"]),
        "checks": RESULTS,
        "note": "All float comparisons are exact equality (==), no tolerance. "
                "Equivalence clause: same (level, seed) + same command sequence "
                "=> identical poses, costs, labels, z, safety and observations.",
    }
    print(json.dumps({"verdict": verdict, "n_checks": len(RESULTS),
                      "n_passed": report["n_passed"]}))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
