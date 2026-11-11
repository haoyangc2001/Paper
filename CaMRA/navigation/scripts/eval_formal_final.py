"""Final-checkpoint diagnostic evaluation for the formal runs (NAV-036).

Two things, per the prereg's ``evaluation_of_this_task`` and its OI-1 exit
condition:

1. **Development diagnostic** (reporting-only, never a final number): the
   final checkpoint acts deterministically (mean action) on the 6
   development maps, budget z0 = 1.0 x E_ref(map) from the frozen
   recomputation report.
2. **OI-1 command-distribution re-test**: the same deterministic rollouts on
   the 18 train maps, pooled applied commands scored with the *line-identical*
   ``command_stats``/``deadband_profile`` definitions used for the NAV-035
   candidate-vs-rule-controller comparison (change rate, hold duration, low-
   speed share, yaw-rate dead-band buckets, |vy|).

Run per seed:

    <frozen python> CaMRA/navigation/scripts/eval_formal_final.py \
        --run-id formal_uncal_s0 --json CaMRA/navigation/reports/formal_uncal_s0_eval.json
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, NAV_ROOT)  # for importing scripts.check_candidate_commands

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.vector import VectorNominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

NAV_CONTRACT = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
SPLIT = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
E_REF_REPORT = os.path.join(NAV_ROOT, "reports", "train_map_reference_energy.json")


def load_state(run_dir: str, variant: str) -> dict:
    path = os.path.join(run_dir, "state.pt")
    if not os.path.exists(path):
        raise SystemExit(f"missing final checkpoint {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="final diagnostic evaluation")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", default="uncalibrated")
    ap.add_argument("--work-root", default="/pub/data/caohy/work/camra_navigation")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    run_dir = os.path.join(os.path.abspath(args.work_root), args.run_id)
    state = load_state(run_dir, args.variant)

    torch.set_num_threads(1)
    nav_contract = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    net = NW.build(nav_contract)
    net.load_state_dict(state["model"])
    net.eval()

    import dataclasses
    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    if args.variant == "uncalibrated":
        geometry = dataclasses.replace(geometry, delta_m=0.0)
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    else:
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")

    refs = json.load(open(E_REF_REPORT, encoding="utf-8"))
    assert refs["verdict"] == "PASS"
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}

    split = json.load(open(SPLIT, encoding="utf-8"))

    def rollout(spec, z0: float) -> dict[str, Any]:
        env = VectorNominalNavEnv(spec, geometry, cm, n_envs=1, z0_J=z0)
        env.reset()
        cmds: list[list[float]] = []
        min_d = math.inf
        cost_J = 0.0
        steps = 0
        while not env._done[0] and steps < 300:
            o = torch.as_tensor(env.observation, dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = net.act(o, deterministic=True)
            rec = env.step(a.numpy())
            cmds.append(rec.applied[0].tolist())
            min_d = min(min_d, float(rec.goal_distance_m[0]))
            cost_J += float(rec.cost_model_J[0])
            steps += 1
        label = env._label[0] or "cap_300"
        res = env.result(0)
        return {
            "map_id": spec.map_id, "level": spec.level, "seed": spec.seed,
            "z0_J": round(z0, 2), "label": label, "n_steps": steps,
            "final_goal_distance_m": round(res.goal_distance_m, 3),
            "min_goal_distance_m": round(min_d, 3),
            "cost_model_J": round(cost_J, 2),
            "commands": cmds,
        }

    dev_rows: list[dict[str, Any]] = []
    for m in split["splits"]["development"]["maps"]:
        spec = M.build(m["level"], m["seed"], design)
        r = rollout(spec, e_ref[m["map_id"]])
        dev_rows.append({k: v for k, v in r.items() if k != "commands"})

    train_rows: list[dict[str, Any]] = []
    train_cmds: list[list[float]] = []
    for m in split["splits"]["train"]["maps"]:
        spec = M.build(m["level"], m["seed"], design)
        r = rollout(spec, e_ref[m["map_id"]])
        train_cmds.extend(r.pop("commands"))
        train_rows.append(r)

    # OI-1: pooled applied commands from the train-map deterministic rollouts
    from scripts.check_candidate_commands import command_stats, deadband_profile  # noqa: E402
    pooled = np.asarray(train_cmds, dtype=np.float64)
    oi1 = {
        "definitions_source": "scripts/check_candidate_commands.py (line-identical "
                              "to check_calibration_applicability.command_stats)",
        "mode": "deterministic mean action, nominal env, 18 train maps",
        "pooled": command_stats(pooled),
        "pooled_deadband": deadband_profile(pooled),
    }

    dev_sr = sum(1 for r in dev_rows if r["label"] == "goal_reached") / len(dev_rows)
    train_sr = sum(1 for r in train_rows if r["label"] == "goal_reached") / len(train_rows)
    report = {
        "task": "NAV-036",
        "run_id": args.run_id,
        "variant": args.variant,
        "seed": state.get("seed"),
        "prereg_sha256": state.get("prereg_sha256"),
        "development_diagnostic": {
            "reporting": "diagnostic only; never a final number (experiment_prereg)",
            "success_rate": dev_sr,
            "episodes": dev_rows,
        },
        "train_maps_deterministic": {
            "success_rate": train_sr,
            "episodes": train_rows,
        },
        "oi1_command_distribution": oi1,
    }
    print(json.dumps({
        "run_id": args.run_id,
        "dev_success_rate": dev_sr,
        "train_success_rate": train_sr,
        "dev_min_distance_range": [min(r["min_goal_distance_m"] for r in dev_rows),
                                   max(r["min_goal_distance_m"] for r in dev_rows)],
        "change_rate_mean": oi1["pooled"].get("change_rate_mean"),
        "hold_mean_s": oi1["pooled"].get("hold_mean_s"),
        "mean_abs_vy": oi1["pooled"]["mean_abs"].get("vy"),
        "low_speed_share": oi1["pooled"].get("low_speed_share"),
    }, ensure_ascii=False))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


def rollout_cmds_train(split, design, geometry, cm, e_ref):
    """Yield per-map applied-command lists from the deterministic rollouts."""
    import dataclasses
    geom = geometry
    for m in split["splits"]["train"]["maps"]:
        spec = M.build(m["level"], m["seed"], design)
        env = VectorNominalNavEnv(spec, geom, cm, n_envs=1, z0_J=e_ref[m["map_id"]])
        env.reset()
        cmds: list[list[float]] = []
        steps = 0
        while not env._done[0] and steps < 300:
            o = torch.as_tensor(env.observation, dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = net.act(o, deterministic=True)
            rec = env.step(a.numpy())
            cmds.append(rec.applied[0].tolist())
            steps += 1
        yield cmds


if __name__ == "__main__":
    sys.exit(main())
