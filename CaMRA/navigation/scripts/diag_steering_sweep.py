"""Offline acceptance for the steering prior: sampled actions on the dev split.

Training samples actions, so the acceptance must too (round-8 lesson). Sweeps the
prior gain and reports arrival, within-budget arrival and cost for both variants.
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
V0, STD_INIT = 0.465, -3.0


def rollout(net, variant, m, sample: bool):
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
            a, _, _ = net.act(ob, deterministic=not sample)
        env.step(a.numpy())
        st += 1
    res = env.result(0)
    return (env._label[0] or "cap", float(res.cost_model_J), float(env.z_J(0)),
            float(res.goal_distance_m), st)


for gain in (0.10, 0.15, 0.20, 0.30):
    for variant in ("uncalibrated", "full"):
        for sample in (True, False):
            torch.manual_seed(0)
            net = NW.build(nav, action_bounded=True, initial_forward_speed=V0,
                           log_std_min=-5.0, log_std_init=STD_INIT,
                           steering_gain=gain)
            net.eval()
            out = [rollout(net, variant, m, sample) for m in dev]
            lab = [o[0] for o in out]
            n_goal = sum(1 for x in lab if x == "goal_reached")
            n_wb = sum(1 for o in out if o[0] == "goal_reached" and o[2] >= 0.0)
            bd = {}
            for x in lab:
                bd[x] = bd.get(x, 0) + 1
            tag = "sampled" if sample else "determ "
            print(f"gain={gain:.1f} {variant:13s} {tag} goal={n_goal}/6 "
                  f"within_budget={n_wb}/6 cost={statistics.mean(o[1] for o in out):6.1f}J "
                  f"z_min={min(o[2] for o in out):7.1f}J dmin={statistics.mean(o[3] for o in out):5.2f} {bd}")
