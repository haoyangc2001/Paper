"""Read-only: how does the trained r10 policy actually move on development maps?

Compares (a) a freshly built init-fix policy and (b) the trained r10 checkpoint,
recording per-step forward speed and the trajectory shape.
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
dev = split["splits"]["development"]["maps"][:2]
ROOT = "/pub/data/caohy/work/camra_navigation"


def probe(net, variant, tag):
    for m in dev:
        ctx = E.build_rollout_ctx(variant)
        spec = E.M.build(m["level"], m["seed"], ctx["design"])
        z0 = e_ref[m["map_id"]]
        if variant == "full":
            z0 = ACC.consistent_e_ref(z0, "full")
        env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
        env.reset()
        steps, vxs, dists = 0, [], []
        while not env._done[0] and steps < 300:
            ob = torch.as_tensor(env.observation, dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = net.act(ob, deterministic=True)
            vxs.append(float(a[0, 0]))
            dists.append(float(env._goal_distance_now() if hasattr(env, "_goal_distance_now")
                               else 0.0))
            env.step(a.numpy())
            steps += 1
        res = env.result(0)
        seg = [round(statistics.mean(vxs[i:i + 50]), 3) for i in range(0, len(vxs), 50)]
        print(f"{tag} {m['map_id']}: label={env._label[0]} steps={steps} "
              f"cost={res.cost_model_J:.0f}J min_dist={res.goal_distance_m:.1f} "
              f"vx_by_50={seg}")


for variant in ("uncalibrated",):
    torch.manual_seed(0)
    fresh = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
                     log_std_min=-5.0, log_std_init=-3.0)
    fresh.eval()
    probe(fresh, variant, f"FRESH[{variant}]")
    st = torch.load(f"{ROOT}/nav064_r10_{variant}_s0/state.pt", map_location="cpu",
                    weights_only=False)
    tr = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
                  log_std_min=-5.0, log_std_init=-3.0)
    tr.load_state_dict(st["model"])
    tr.eval()
    probe(tr, variant, f"TRAINED_r10_s0[{variant}]")
