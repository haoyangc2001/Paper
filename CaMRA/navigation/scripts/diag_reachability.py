"""Read-only feasibility check: at the reference speed, can a simple
goal-seeking controller reach the goal on the development maps?

This decides whether "give the initial policy a usable starting behaviour"
can break the no-arrival self-lock. No file is modified.
"""
import json
import math
import os
import sys

import numpy as np

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "scripts"))
import eval_training_round as E  # noqa: E402

nav = None
split = json.load(open(E.SPLIT, encoding="utf-8"))
refs = json.load(open(E.E_REF, encoding="utf-8"))
e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
dev_maps = split["splits"]["development"]["maps"]
V = 0.465  # m/s, reference speed from the frozen power model


def run(variant, m, z0, mode):
    ctx = E.build_rollout_ctx(variant)
    spec = E.M.build(m["level"], m["seed"], ctx["design"])
    env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
    env.reset()
    steps = 0
    while not env._done[0] and steps < 300:
        if mode == "straight":
            a = np.array([[V, 0.0, 0.0]], dtype=np.float32)
        else:
            px, py, yaw = float(env._pose[0, 0]), float(env._pose[0, 1]), float(env._pose[0, 2])
            gx, gy = float(env.spec.goal[0]), float(env.spec.goal[1])
            theta = math.atan2(gy - py, gx - px) - yaw
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            a = np.array([[V, 0.0, max(-1.0, min(1.0, 1.2 * theta))]], dtype=np.float32)
        env.step(a)
        steps += 1
    res = env.result(0)
    return env._label[0] or "cap", steps, round(float(res.cost_model_J), 1), round(float(res.z_J), 1)


for mode in ("straight", "seek"):
    print("=" * 10, "mode =", mode, "=" * 10)
    for variant in ("uncalibrated", "full"):
        labels, steps, costs, zends = [], [], [], []
        for m in dev_maps:
            z0 = e_ref[m["map_id"]]
            if variant == "full":
                from camra_navigation.budget import accounting as ACC
                z0 = ACC.consistent_e_ref(z0, "full")
            lab, st, cost, ze = run(variant, m, z0, mode)
            labels.append(f"{m['map_id']}:{lab}")
            steps.append(st)
            costs.append(cost)
            zends.append(ze)
        n_goal = sum(1 for x in labels if x.endswith("goal_reached"))
        print(f"{variant}: goal={n_goal}/6 steps_mean={np.mean(steps):.0f} "
              f"cost_mean={np.mean(costs):.1f}J z_end_mean={np.mean(zends):.1f}J")
        for x in labels:
            print("   ", x)
print("reference speed:", V, "m/s; 14 m route needs ~30 s")
