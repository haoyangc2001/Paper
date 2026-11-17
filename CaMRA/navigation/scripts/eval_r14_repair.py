"""Development-only evaluation of the repaired r14 final checkpoints.

Explicit observation, checkpoint and preregistration identities are checked
before any rollout. All three registered budget points share the same six
development maps; only k=1.15 is used for the closing decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

import train_formal as TRAIN
import eval_training_round as E
import contract_margin_eval as CM
from camra_navigation.algorithm.validation import file_sha
from camra_navigation.budget.accounting import consistent_e_ref


def build_registered_network(nav, prereg):
    init = prereg['action_init_fix']
    return E.NW.build(nav, action_bounded=prereg['action_bounded'],
        initial_forward_speed=init['initial_forward_speed'],
        log_std_min=init['log_std_min'], log_std_init=init['log_std_init'],
        steering_gain=prereg['steering_prior']['gain'],
        avoidance_prior=prereg.get('avoidance_prior'))


def rollout(net, spec, ctx, z0, prereg):
    stag = prereg["stagnation"]
    sev = prereg["safety_continuity"]
    env = E.VectorNominalNavEnv(
        spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0,
        t_max_s=prereg["environment"]["t_max_s"],
        r_goal_m=prereg["environment"]["r_goal_m"],
        observation_version=prereg["implementation_versions"]["observation"],
        stagnation_window_s=stag["window_s"],
        stagnation_min_displacement_m=stag["min_displacement_m"],
        safety_severity_enabled=sev["enabled"],
        safety_severity_floor=sev["severity_floor"],
        safety_severity_ref_depth_m=sev["severity_ref_depth_m"])
    trace = []
    for step in range(math.ceil(env.t_max_s / .1) + 5):
        if env._done[0]:
            break
        observation = env.observation.copy()
        with torch.no_grad():
            action, _, _ = net.act(torch.as_tensor(observation, dtype=torch.float32), deterministic=True)
        rec = env.step(action.numpy())
        trace.append({"step": step, "observation": observation[0].tolist(),
                      "action": action[0].tolist(), "z_J": float(rec.z_J[0]),
                      "safety_y": float(rec.safety_y[0]),
                      "danger_violation": bool(rec.danger_violation[0]),
                      "goal_distance_m": float(rec.goal_distance_m[0])})
    result = env.result(0)
    dangers = sum(t["danger_violation"] for t in trace)
    clean = all(t["safety_y"] == 0. for t in trace)
    label = env._label[0] or "cap_step"
    return {"label": label, "n_steps": len(trace), "t_s": float(result.t_s),
            "cost_model_J": float(result.cost_model_J), "z0_J": z0,
            "z_J_end": float(result.z_J), "danger_steps": dangers,
            "ratio_cost_to_budget": float(result.cost_model_J) / z0,
            "feasible": bool(label == "goal_reached" and clean and result.z_J >= 0.),
            "stagnation_context": result.stagnation_context, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default="training_loop_prereg_r14_repair_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-root", default=E.WORK_ROOT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    prereg, pin = TRAIN.verify_prereg(args.prereg)
    root = Path(E.NAV)
    split = json.loads(Path(E.SPLIT).read_bytes())
    maps = split["splits"]["development"]["maps"]
    train_ids = {m["map_id"] for m in split["splits"]["train"]["maps"]}
    assert len(maps) == 6 and not train_ids.intersection(m["map_id"] for m in maps)
    nav = json.loads((root / prereg["binding"]["navigation_contract"]["path"]).read_bytes())
    cache = Path(args.out).with_suffix(".references.json")
    if cache.exists() or Path(args.out).exists():
        raise SystemExit("new evaluation output and cache paths are required")
    refs = CM.refs_at(prereg["environment"]["t_max_s"], maps, str(cache))
    rows = []
    for variant in prereg["method"]["variants"]:
        ctx = E.build_rollout_ctx(variant)
        for seed in prereg["method"]["seeds"]:
            rid = TRAIN.registered_run_id(prereg, variant, seed)
            run = Path(args.work_root) / rid
            state = torch.load(run / "state.pt", map_location="cpu", weights_only=False)
            logs = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
            start = next(x for x in logs if x["kind"] == "run_start")
            end = next(x for x in logs if x["kind"] == "run_end")
            updates = [x for x in logs if x["kind"] == "update"]
            assert state["prereg_sha256"] == start["prereg_sha256"] == pin
            assert state["implementation_versions"] == prereg["implementation_versions"]
            assert state["seed"] == seed and state["variant"] == variant
            assert state["update_idx"] + 1 == end["updates"] == len(updates) == prereg["window_semantics"]["max_updates"]
            assert start["data_access"]["split_used"] == "train" and set(start["data_access"]["map_ids"]) == train_ids
            assert start["obs_version"] == prereg["implementation_versions"]["observation"]
            assert not start["smoke"] and start["calls_per_window"] == prereg["window_semantics"]["calls_per_update"]
            assert start.get("collection", "window_v1") == prereg["window_semantics"].get("collection", "window_v1")
            assert start.get('avoidance_prior') == prereg.get('avoidance_prior')
            net = build_registered_network(nav, prereg)
            net.load_state_dict(state["model"])
            net.eval()
            budget_rows = []
            for k in prereg["round_acceptance"]["budget_response_on_dev"]["grid_k"]:
                for m in maps:
                    assert refs[m["map_id"]]["admitted"]
                    e_ref = consistent_e_ref(refs[m["map_id"]]["e_ref_J"], variant)
                    spec = E.M.build(m["level"], m["seed"], ctx["design"])
                    r = rollout(net, spec, ctx, k * e_ref, prereg)
                    budget_rows.append({"k": k, "map_id": m["map_id"], **r})
            primary = [r for r in budget_rows if r["k"] == 1.15]
            row = {"run_id": rid, "variant": variant, "seed": seed,
                   "state_sha256": file_sha(run / "state.pt"),
                   "sr_feasible": sum(r["feasible"] for r in primary) / 6,
                   "energy_ratio": statistics.median(r["ratio_cost_to_budget"] for r in primary),
                   "danger_steps": sum(r["danger_steps"] for r in primary),
                   "episodes": budget_rows}
            rows.append(row)
            print(rid, row["sr_feasible"], row["energy_ratio"], row["danger_steps"], flush=True)
    summary = {}
    thresholds = prereg["round_acceptance"]
    for variant in prereg["method"]["variants"]:
        group = [r for r in rows if r["variant"] == variant]
        med = {k: statistics.median(r[k] for r in group) for k in ("sr_feasible", "energy_ratio", "danger_steps")}
        med["performance_pass"] = (med["sr_feasible"] >= thresholds["sr_feasible_median"]
                                   and med["energy_ratio"] <= thresholds["energy_ratio_median"]
                                   and (variant != "full" or med["danger_steps"] <= thresholds["danger_steps_median_full"]))
        summary[variant] = med
    report = {"schema": "camra-r14-repair-development-v1", "owner": "NAV-064",
              "prereg_sha256": pin, "summary": summary, "runs": rows,
              "performance_pass": all(r["performance_pass"] for r in summary.values()),
              "closure": "requires separate stability, mechanism and full-store audit",
              "data_access": {"rollout_split": "development", "maps": [m["map_id"] for m in maps],
                              "paths": {p: file_sha(Path(p)) for p in (E.SPLIT, E.DESIGN, E.POWER, E.MARGINS, E.FOOTPRINT, E.GEOM)},
                              "training_map_ids": sorted(train_ids)},
              "aggregation": "within-seed median energy/budget across six maps, then three-seed median; danger is per-seed total, then three-seed median; primary k=1.15"}
    with Path(args.out).open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
