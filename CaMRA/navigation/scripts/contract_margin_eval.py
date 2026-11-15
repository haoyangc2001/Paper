#!/usr/bin/env python3
"""Contract-margin evaluation: what the existing checkpoints do when the task has
some slack, without retraining anything.

Why: S1 measured that ``E_ref`` equals the energy of the reference route driven at
exactly the 30 s deadline speed, and that ``v_deadline`` (0.465-0.571 m/s) is 50 %
above the energy-optimal 0.31 m/s.  Budget and time are therefore simultaneous
zeros, so "arrived AND within budget" is a single trajectory and any 3-19 % error
fails.  This script re-evaluates the *same* checkpoints under a variant contract

    deadline  T = 45 s        (so 0.31 m/s becomes feasible)
    budget    k = 1.15        (absorbs path/steering error)
    E_ref     recomputed at T (v_deadline = L/T, so the budget is honest)

and reports both contracts side by side.  Nothing is trained, nothing frozen is
overwritten: this is a labelled evaluation variant, not the new contract.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import eval_training_round as E  # noqa: E402
from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_calibration.power import PowerModel  # noqa: E402

DESIGN = json.load(open(E.DESIGN, encoding="utf-8"))
NAV = json.load(open(E.NAV_CONTRACT, encoding="utf-8"))


def build_net(prereg):
    init_fix = prereg.get("action_init_fix", {})
    init_on = bool(init_fix.get("enabled", False))
    steer = prereg.get("steering_prior", {})
    gain = float(steer.get("gain", 0.0)) if bool(steer.get("enabled", False)) else 0.0
    return E.NW.build(
        NAV,
        action_bounded=bool(prereg.get("action_bounded", False)),
        initial_forward_speed=float(init_fix.get("initial_forward_speed", 0.0)) if init_on else 0.0,
        log_std_min=float(init_fix["log_std_min"]) if init_on and "log_std_min" in init_fix else None,
        log_std_init=float(init_fix["log_std_init"]) if init_on and "log_std_init" in init_fix else None,
        steering_gain=gain,
    )


def refs_at(t_deadline, dev_maps, cache_path):
    """E_ref per dev map recomputed at the given deadline (cached on disk)."""
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        if float(cached.get("t_deadline_s")) == float(t_deadline):
            return cached["maps"]
    model = PowerModel.from_dict(json.load(open(E.POWER, encoding="utf-8")))
    geom = E.SafetyGeometry.from_frozen(E.FOOTPRINT, E.MARGINS, E.GEOM)
    out = {}
    for m in dev_maps:
        spec = E.M.build(m["level"], m["seed"], DESIGN)
        rec = F.check_map(spec, geom, model, resolution=0.1, t_deadline_s=float(t_deadline))
        route = rec.get("route") or {}
        out[m["map_id"]] = {
            "route_length_m": route.get("length_m"),
            "v_deadline_mps": route.get("v_deadline_mps"),
            "e_ref_J": route.get("predicted_energy_deadline_J"),
            "admitted": bool(rec.get("admitted")),
            "reasons": rec.get("reasons", []),
        }
    json.dump({"t_deadline_s": t_deadline, "maps": out}, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return out


def run_episode(net, spec, ctx, z0, t_max_s, cap, env_kwargs=None):
    env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0,
                                t_max_s=t_max_s, **(env_kwargs or {}))
    env.reset()
    steps = 0
    min_d = math.inf
    dangers = 0
    while not env._done[0] and steps < cap:
        o = torch.as_tensor(env.observation, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(o, deterministic=True)
        rec = env.step(a.numpy())
        steps += 1
        min_d = min(min_d, float(rec.goal_distance_m[0]))
        dangers += int(bool(rec.danger_violation[0]))
    res = env.result(0)
    return {
        "label": env._label[0] or "cap_step", "n_steps": steps,
        "t_s": round(float(res.t_s), 2), "min_goal_distance_m": round(float(min_d), 3),
        "cost_model_J": round(float(res.cost_model_J), 1), "z0_J": round(float(z0), 1),
        "z_J_end": round(float(res.z_J), 1), "within_budget": bool(res.within_budget),
        "danger_steps": dangers,
        # r14 诊断：分级严重度与停滞触发现场（无触发为 None）
        "max_safety_severity": round(float(res.max_safety_severity), 4),
        "stagnation_context": res.stagnation_context,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=float, default=45.0)
    ap.add_argument("--k", type=float, default=1.15)
    ap.add_argument("--runs", required=True, help="JSON 文件：{run_id: prereg_name}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="reports/contract_margin_refs.json")
    args = ap.parse_args()

    split = json.load(open(E.SPLIT, encoding="utf-8"))
    dev_maps = split["splits"]["development"]["maps"]
    refs = refs_at(args.t_max, dev_maps, args.cache)
    run_map = json.load(open(args.runs, encoding="utf-8"))

    from camra_navigation.budget import accounting as ACC
    preregs = {}
    for prereg_name in set(run_map.values()):
        p = os.path.join(E.NAV, "config", prereg_name)
        pinned = open(p + ".sha256", encoding="utf-8").read().strip()
        if E.sha(p) != pinned:
            raise SystemExit("prereg %s hash mismatch" % prereg_name)
        preregs[prereg_name] = json.load(open(p, encoding="utf-8"))

    results = []
    for rid, prereg_name in run_map.items():
        prereg = preregs[prereg_name]
        variant = "uncalibrated" if "uncalibrated" in rid else "full"
        acct = str(prereg.get("budget_accounting", "nominal"))
        # r14：评估环境的停滞/安全严重度参数一律读该 run 冻结的预登记——
        # 评估契约与训练契约由同一份文件保证，不再从命令行另传一套。
        stag = prereg.get("stagnation", {}) or {}
        sev = prereg.get("safety_continuity", {}) or {}
        env_kwargs = {
            "stagnation_window_s": float(stag.get("window_s", 3.0)),
            "stagnation_min_displacement_m": float(stag.get("min_displacement_m", 0.05)),
            "safety_severity_enabled": bool(sev.get("enabled", False)),
            "safety_severity_floor": float(sev.get("severity_floor", 0.2)),
            "safety_severity_ref_depth_m": float(sev.get("severity_ref_depth_m", 0.25)),
        }
        state = torch.load(os.path.join(E.WORK_ROOT, rid, "state.pt"),
                           map_location="cpu", weights_only=False)
        net = build_net(prereg)
        net.load_state_dict(state["model"])
        net.eval()
        ctx = E.build_rollout_ctx(variant)
        rows = []
        print("== %s (%s)" % (rid, variant))
        for m in dev_maps:
            spec = E.M.build(m["level"], m["seed"], DESIGN)
            e_nom = refs[m["map_id"]]["e_ref_J"]
            mult = ACC.consistent_e_ref if acct == "consistent" else (lambda e, v: e)
            z0 = args.k * mult(e_nom, variant)
            r = run_episode(net, spec, ctx, z0, args.t_max,
                            int(args.t_max / 0.1) + 50, env_kwargs=env_kwargs)
            feasible = (r["label"] == "goal_reached" and r["danger_steps"] == 0
                        and r["within_budget"])
            r.update({"map_id": m["map_id"], "feasible": bool(feasible),
                      "ratio_cost_to_budget": round(r["cost_model_J"] / z0, 3)})
            rows.append(r)
            print("  %-24s %-12s t=%5.1fs 距%5.2fm 耗%6.1fJ/%6.1fJ=%5.0f%% 危险%3d 可行=%s"
                  % (m["map_id"], r["label"], r["t_s"], r["min_goal_distance_m"],
                     r["cost_model_J"], z0, 100 * r["cost_model_J"] / z0,
                     r["danger_steps"], feasible))
        sr = sum(1 for r in rows if r["label"] == "goal_reached") / len(rows)
        feas = sum(1 for r in rows if r["feasible"]) / len(rows)
        results.append({"run_id": rid, "variant": variant, "sr": round(sr, 3),
                        "sr_feasible": round(feas, 3),
                        "env_kwargs": env_kwargs, "prereg": prereg_name,
                        "episodes": rows})
        print("   -> SR %.2f  可行到达 %.2f\n" % (sr, feas))

    payload = {"schema": "camra-contract-margin-eval-v1", "task": "NAV-064",
               "variant_contract": {"t_max_s": args.t_max, "k": args.k,
                                    "e_ref": "recomputed at the new deadline",
                                    "note": "评估变体，不是新契约；冻结契约与历史报告未改动"},
               "runs": results}
    json.dump(payload, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
