"""Read-only: is the trained r10 decay driven by the budget/time inputs?

Take a mid-episode state from a trained rollout, then intervene on the two task
inputs that could teach deceleration (remaining budget z_J, remaining time) and
measure the change in the commanded forward speed. If the policy is insensitive
to z_J, the residual decay is not a budget-channel effect.
"""
import dataclasses
import json
import math
import os
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
ROOT = "/pub/data/caohy/work/camra_navigation"
par = json.load(open(os.path.join(NAV, "config", "action_init_fix_prereg_v1.json"),
                     encoding="utf-8"))["change"]["parameters"]

m = next(x for x in split["splits"]["development"]["maps"] if x["map_id"] == "l1_empty-s2001")
variant = "uncalibrated"
ctx = E.build_rollout_ctx(variant)
spec = E.M.build(m["level"], m["seed"], ctx["design"])
z0 = e_ref[m["map_id"]]
env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
env.reset()

tr = NW.build(nav, action_bounded=True, initial_forward_speed=par["initial_forward_speed"],
              log_std_min=par["log_std_min"], log_std_init=par["log_std_init"])
tr.load_state_dict(torch.load(f"{ROOT}/nav064_r10_{variant}_s0/state.pt",
                             map_location="cpu", weights_only=False)["model"])
tr.eval()


def fwd(obs):
    with torch.no_grad():
        a, _, _ = tr.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
    return float(a[0, 0])


# walk 100 steps so the state is mid-episode, then probe
for _ in range(100):
    ob = torch.as_tensor(env.observation, dtype=torch.float32)
    with torch.no_grad():
        a, _, _ = tr.act(ob, deterministic=True)
    env.step(a.numpy())

base_obs = env.observation.copy()
base = fwd(base_obs)
print(f"state=step100 pose=({env._pose[0,0]:.2f},{env._pose[0,1]:.2f}) "
      f"z_J={env.z_J(0):.1f} t_s={float(env._t_s[0]):.1f} baseline_forward={base:.3f}")

cases = []
spent0 = float(env._spent_J[0])
t0 = float(env._t_s[0])

# 1) pretend the budget is untouched (fresh episode energy)
env._spent_J[0] = 0.0
env._refresh_obs()
cases.append(("z_J=max(full budget)", fwd(env.observation)))

# 2) pretend the budget is nearly gone
env._spent_J[0] = z0 * 0.99
env._refresh_obs()
cases.append(("z_J=1% of budget", fwd(env.observation)))

# 3) restore budget, vary remaining time
env._spent_J[0] = spent0
env._t_s[0] = 0.0
env._refresh_obs()
cases.append(("time_remaining=30s", fwd(env.observation)))

env._t_s[0] = float(env.t_max_s)
env._refresh_obs()
cases.append(("time_remaining=0s", fwd(env.observation)))

env._t_s[0] = t0
env._refresh_obs()

for tag, v in cases:
    print(f"  {tag:24s} forward={v:.3f}  delta={v - base:+.3f}")

# 4) vary the goal distance (task input) and the ego speed feedback
env._spent_J[0] = spent0
env._t_s[0] = t0
gx, gy = float(env.spec.goal[0]), float(env.spec.goal[1])
px, py = float(env._pose[0, 0]), float(env._pose[0, 1])
for tag, dist in (("goal 2 m ahead", 2.0), ("goal 20 m ahead", 20.0)):
    ang = 0.0
    env.spec = dataclasses.replace(env.spec, goal=(px + dist * math.cos(ang), py + dist * math.sin(ang)))
    env._refresh_obs()
    print(f"  {tag:24s} forward={fwd(env.observation):.3f}")
env.spec = dataclasses.replace(env.spec, goal=(gx, gy))

for tag, v in (("ego_vx=0", 0.0), ("ego_vx=0.8", 0.8)):
    old = float(env._applied_last[0, 0])
    env._applied_last[0, 0] = v
    env._refresh_obs()
    print(f"  {tag:24s} forward={fwd(env.observation):.3f}")
    env._applied_last[0, 0] = old
env._refresh_obs()
