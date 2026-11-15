#!/usr/bin/env python3
"""S1: measure, per step, what a checkpoint actually does on the development maps.

The reports only keep the end-of-episode summary (min distance, cost, label), so
"it failed 1.3 m short" cannot be told apart from "it was still closing in but ran
out of the 30 s" or "it stalled and burned the budget in place". This script
replays episodes deterministically (mean action, same as the evaluation) and
records the trace:

  t_s, goal_distance_m, actual body speed (from pose deltas), commanded vx,
  cumulative energy, danger flag

and derives, per episode:
  path_len_m          distance actually travelled
  mean_speed_mps      path_len / t
  j_per_m             energy / path_len   (compare with the power model below)
  progress_last_5s_m  how much closer it still got in the final 5 s
  need_s_at_last_rate remaining distance / last-5s closing rate
  energy_to_finish_J  remaining distance * j_per_m

Read-only: loads checkpoints, touches no training artefact.

Power model reference (config/power_model_v1.json via NAV-059 analytic):
  P(v) = 3.0284 + 4.7226 v + 31.5420 v^2  [W]
  J/m at v* = 0.3099 m/s -> 24.27 ; at deadline speed 0.4654 m/s -> 25.91
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import eval_training_round as E  # noqa: E402  (reuse frozen loading path)

POW_A, POW_B, POW_C = 3.0284321184264877, 4.722608162350005, 31.5420191257139


def j_per_m_at(v):
    if v <= 0:
        return float("inf")
    return (POW_A + POW_B * v + POW_C * v * v) / v


def build_net(prereg, nav):
    init_fix = prereg.get("action_init_fix", {})
    init_on = bool(init_fix.get("enabled", False))
    steer = prereg.get("steering_prior", {})
    gain = float(steer.get("gain", 0.0)) if bool(steer.get("enabled", False)) else 0.0
    return E.NW.build(
        nav,
        action_bounded=bool(prereg.get("action_bounded", False)),
        initial_forward_speed=float(init_fix.get("initial_forward_speed", 0.0)) if init_on else 0.0,
        log_std_min=float(init_fix["log_std_min"]) if init_on and "log_std_min" in init_fix else None,
        log_std_init=float(init_fix["log_std_init"]) if init_on and "log_std_init" in init_fix else None,
        steering_gain=gain,
    )


def trace_episode(net, spec, ctx, z0, cap=400):
    env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
    env.reset()
    rows = []
    prev = np.array(env.pose[0], dtype=np.float64)
    energy = 0.0
    steps = 0
    while not env._done[0] and steps < cap:
        o = torch.as_tensor(env.observation, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(o, deterministic=True)
        rec = env.step(a.numpy())
        steps += 1
        pa = rec.pose_after[0]
        moved = math.hypot(pa[0] - prev[0], pa[1] - prev[1])
        prev = pa.copy()
        spent = z0 - float(rec.z_J[0])
        rows.append({
            "t": float(rec.t_s[0]),
            "dist": float(rec.goal_distance_m[0]),
            "speed": moved / 0.1,
            "cmd_vx": float(rec.applied[0][0]),
            "energy": spent,
            "step_energy": spent - energy,
            "danger": bool(rec.danger_violation[0]),
            "y": float(rec.safety_y[0]),
        })
        energy = spent
    label = env._label[0] or "cap_step"
    res = env.result(0)
    return rows, label, float(res.z_J)


def summarise(rows, z0, z_end):
    t = rows[-1]["t"] if rows else 0.0
    path = sum(r["speed"] * 0.1 for r in rows)
    energy = z0 - z_end
    span = [r for r in rows if r["t"] >= max(0.0, t - 5.0)]
    if len(span) >= 2:
        progress5 = span[0]["dist"] - span[-1]["dist"]
    else:
        progress5 = 0.0
    rate5 = progress5 / 5.0
    remaining = rows[-1]["dist"] if rows else float("nan")
    need_s = (remaining / rate5) if rate5 > 1e-6 else float("inf")
    jpm = (energy / path) if path > 1e-6 else float("inf")
    return {
        "t_s": round(t, 2),
        "path_len_m": round(path, 2),
        "mean_speed_mps": round(path / t, 3) if t > 0 else 0.0,
        "cmd_vx_mean": round(sum(r["cmd_vx"] for r in rows) / max(len(rows), 1), 3),
        "energy_J": round(energy, 1),
        "j_per_m": round(jpm, 2),
        "model_jpm_at_mean": round(j_per_m_at(path / t), 2) if t > 0 and path > 0 else None,
        "end_dist_m": round(remaining, 2),
        "progress_last_5s_m": round(progress5, 3),
        "closing_rate_last_5s": round(rate5, 3),
        "need_s_at_last_rate": (round(need_s, 1) if need_s != float("inf") else None),
        "energy_to_finish_J": round(remaining * jpm, 1) if path > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--run-ids", required=True)
    ap.add_argument("--work-root", default=E.WORK_ROOT)
    ap.add_argument("--maps", default=None, help="逗号分隔的 map_id 子集")
    ap.add_argument("--ckpt-name", default="state.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prereg_path = os.path.join(E.NAV, "config", args.prereg)
    pinned = open(prereg_path + ".sha256", encoding="utf-8").read().strip()
    if E.sha(prereg_path) != pinned:
        raise SystemExit("prereg hash mismatch; refusing to evaluate")
    prereg = json.load(open(prereg_path, encoding="utf-8"))
    nav = json.load(open(E.NAV_CONTRACT, encoding="utf-8"))
    split = json.load(open(E.SPLIT, encoding="utf-8"))
    refs = json.load(open(E.E_REF, encoding="utf-8"))
    e_ref_nominal = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
    # 与 eval_training_round.py 完全同一口径：budget_accounting=consistent 时
    # full 变体的预算 = 名义 E_ref x (1+alpha)。漏了这一步会同时把 z0 与观测里的
    # 预算比改错，策略行为也会跟着变（S1 首跑就在这上面错过一次）。
    acct = str(prereg.get("budget_accounting", "nominal"))
    if acct == "consistent":
        from camra_navigation.budget import accounting as ACC
        e_ref_full = {m: ACC.consistent_e_ref(e, "full") for m, e in e_ref_nominal.items()}
    elif acct == "nominal":
        e_ref_full = dict(e_ref_nominal)
    else:
        raise SystemExit(f"unknown budget_accounting={acct!r}")
    dev_maps = split["splits"]["development"]["maps"]
    if args.maps:
        keep = set(args.maps.split(","))
        dev_maps = [m for m in dev_maps if m["map_id"] in keep]

    out = {"prereg": args.prereg, "maps": [m["map_id"] for m in dev_maps], "runs": []}
    for rid in args.run_ids.split(","):
        variant = "uncalibrated" if "uncalibrated" in rid else "full"
        state = torch.load(os.path.join(args.work_root, rid, args.ckpt_name),
                           map_location="cpu", weights_only=False)
        net = build_net(prereg, nav)
        net.load_state_dict(state["model"])
        net.eval()
        ctx = E.build_rollout_ctx(variant)
        e_ref = e_ref_full if (acct == "consistent" and variant == "full") else e_ref_nominal
        run = {"run_id": rid, "variant": variant, "budget_accounting": acct, "episodes": []}
        print("== %s ==" % rid)
        for m in dev_maps:
            spec = E.M.build(m["level"], m["seed"], ctx["design"])
            z0 = e_ref[m["map_id"]]
            rows, label, z_end = trace_episode(net, spec, ctx, z0)
            s = summarise(rows, z0, z_end)
            s.update({"map_id": m["map_id"], "label": label, "z0_J": round(z0, 1)})
            s["trace"] = rows
            run["episodes"].append(s)
            print("  %-24s %-12s t=%5.1fs path=%5.1fm v=%4.2fm/s 指令vx=%4.2f "
                  "能耗%6.1fJ(%4.1fJ/m, 模型%4.1f) 终点距%5.2fm 末5s进度%5.2fm 还需%5.1fs"
                  % (m["map_id"], label, s["t_s"], s["path_len_m"], s["mean_speed_mps"],
                     s["cmd_vx_mean"], s["energy_J"], s["j_per_m"],
                     s["model_jpm_at_mean"] or -1, s["end_dist_m"],
                     s["progress_last_5s_m"], s["need_s_at_last_rate"] or -1))
        out["runs"].append(run)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
