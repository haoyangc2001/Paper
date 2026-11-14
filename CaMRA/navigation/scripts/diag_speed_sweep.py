"""Read-only sweep: command speed vs arrival, cost and remaining budget on the dev split.

The fresh init-fix policy holds a constant forward speed, so this measures whether a
higher initial speed buys the arrival margin needed by the 30 s clock without
overspending the frozen reference energy. Deterministic actions, both variants.
"""
import json
import os
import statistics
import sys

import torch

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "scripts"))
sys.path.insert(0, os.path.join(NAV, "src"))
import eval_training_round as E  # noqa: E402
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.budget import accounting as ACC  # noqa: E402

nav = json.load(open(E.NAV_CONTRACT, encoding="utf-8"))
split = json.load(open(E.SPLIT, encoding="utf-8"))
refs = json.load(open(E.E_REF, encoding="utf-8"))
e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
dev = split["splits"]["development"]["maps"]

for v0 in (0.465, 0.520):
    for variant in ("uncalibrated", "full"):
        torch.manual_seed(0)
        net = NW.build(nav, action_bounded=True, initial_forward_speed=v0,
                       log_std_min=-5.0, log_std_init=-3.0)
        net.eval()
        labels, costs, zends, dmin = [], [], [], []
        for m in dev:
            ctx = E.build_rollout_ctx(variant)
            spec = E.M.build(m["level"], m["seed"], ctx["design"])
            z0 = e_ref[m["map_id"]]
            if variant == "full":
                z0 = ACC.consistent_e_ref(z0, "full")
            env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
            env.reset()
            st = 0
            while not env._done[0] and st < 300:
                ob = torch.as_tensor(env.observation, dtype=torch.float32)
                with torch.no_grad():
                    a, _, _ = net.act(ob, deterministic=True)
                env.step(a.numpy())
                st += 1
            res = env.result(0)
            labels.append(env._label[0] or "cap")
            costs.append(float(res.cost_model_J))
            zends.append(float(env.z_J(0)))
            dmin.append(float(res.goal_distance_m))
        n_goal = sum(1 for x in labels if x == "goal_reached")
        n_wb = sum(1 for lab, z in zip(labels, zends) if lab == "goal_reached" and z >= 0.0)
        bd = {}
        for x in labels:
            bd[x] = bd.get(x, 0) + 1
        print(f"v0={v0:.3f} {variant:13s} goal={n_goal}/6 within_budget={n_wb}/6 "
              f"cost_mean={statistics.mean(costs):.1f}J z_end_min={min(zends):.1f}J "
              f"min_dist_mean={statistics.mean(dmin):.2f} {bd}")
        for m, lab, c, z, d in zip(dev, labels, costs, zends, dmin):
            print(f"     {m['map_id']:24s} {lab:12s} cost={c:6.1f} z_end={z:7.1f} min_dist={d:5.2f}")
