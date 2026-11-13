"""Offline acceptance for action_init_fix (AI2/AI3/AI4/AI5)."""
import json
import math
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
dev_maps = split["splits"]["development"]["maps"]

# ---- AI2: bias calibration -------------------------------------------------
torch.manual_seed(0)
on = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
              log_std_min=-1.5)
off = NW.build(nav, action_bounded=True)
b_on = float(on.policy.head.bias.data[0])
b_off = float(off.policy.head.bias.data[0])
print(f"AI2 bias_on={b_on:.8f} tanh(bias_on)={math.tanh(b_on):.8f} "
      f"(target 0.465) | bias_off={b_off:.8f}")
assert abs(math.tanh(b_on) - 0.465) < 1e-6, "AI2 bias calibration"
assert abs(b_off) < 1e-9, "AI2 default must be unchanged"
print("AI2 PASS (bias calibrated; default unchanged)")

# ---- AI3: exploration floor ------------------------------------------------
with torch.no_grad():
    on.policy.log_std.fill_(-4.0)
on.policy.clamp_log_std()
floored = float(on.policy.log_std.min())
print(f"AI3 log_std after clamp from -4.0: min={floored:.4f} (floor -1.5)")
assert floored >= -1.5 - 1e-6
with torch.no_grad():
    off.policy.log_std.fill_(-4.0)
off.policy.clamp_log_std()
print(f"AI3 default floor: min={float(off.policy.log_std.min()):.4f} (expected -4.0)")
assert float(off.policy.log_std.min()) < -1.5
print("AI3 PASS")

# ---- AI4: distribution self-consistency ------------------------------------
torch.manual_seed(0)
on2 = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
               log_std_min=-1.5)
o = torch.zeros(1, on2.policy.cfg.obs_dim)
dist = on2.policy.forward(o)
smp = dist.sample((4000,))
lp = dist.log_prob(smp)
finite = bool(torch.isfinite(lp).all())
strict = bool((smp.abs() < 1.0).all())
ent = float(dist.entropy().mean())
print(f"AI4 log_prob finite={finite} |a|<1={strict} entropy={ent:.4f} "
      f"|a| mean={float(smp.abs().mean()):.4f}")
assert finite and strict and math.isfinite(ent)
print("AI4 PASS")

# ---- AI5: does a fresh policy produce arrival examples? --------------------
def rollout(net, variant, m, z0):
    ctx = E.build_rollout_ctx(variant)
    spec = E.M.build(m["level"], m["seed"], ctx["design"])
    env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
    env.reset()
    steps, vx = 0, []
    while not env._done[0] and steps < 300:
        ob = torch.as_tensor(env.observation, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(ob, deterministic=True)
        vx.append(abs(float(a[0, 0])))
        env.step(a.numpy())
        steps += 1
    res = env.result(0)
    return env._label[0] or "cap", steps, float(res.cost_model_J), statistics.mean(vx)


for label, kwargs in (("OFF", {}), ("ON", dict(initial_forward_speed=0.465,
                                              log_std_min=-1.5))):
    for variant in ("uncalibrated", "full"):
        torch.manual_seed(0)
        net = NW.build(nav, action_bounded=True, **kwargs)
        net.eval()
        labels, speeds = [], []
        for m in dev_maps:
            z0 = e_ref[m["map_id"]]
            if variant == "full":
                z0 = ACC.consistent_e_ref(z0, "full")
            lab, st, cost, vmean = rollout(net, variant, m, z0)
            labels.append(f"{m['map_id']}:{lab}")
            speeds.append(vmean)
        n_goal = sum(1 for x in labels if x.endswith("goal_reached"))
        print(f"AI5 {label} {variant}: goal={n_goal}/6 "
              f"mean|vx|={statistics.mean(speeds):.4f} labels={labels}")
print("AI5 target: ON >= 2/6 arrivals (pure-straight baseline was 2/6)")
