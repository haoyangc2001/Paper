"""Deterministic acceptance for the steering prior (SC1-SC5).

The curriculum half is execution-tested through the trainer smoke log
(``[CURRICULUM] update=... joint_cost_weight=...``), because the schedule lives
inline in the trainer's update loop.
"""
import json
import math
import os
import sys

import torch

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.observation import field_slices  # noqa: E402

nav = json.load(open(os.path.join(NAV, "config", "navigation_contract_v1.json"),
                     encoding="utf-8"))
GAIN = 0.2
V0, STD = 0.465, -3.0
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
base = NW.build(nav, action_bounded=True, initial_forward_speed=V0,
                log_std_min=-5.0, log_std_init=STD)
torch.manual_seed(0)  # 同一随机源重建，才能逐位比较“开关关闭”与旧路径
off = NW.build(nav, action_bounded=True, initial_forward_speed=V0,
               log_std_min=-5.0, log_std_init=STD, steering_gain=0.0)
torch.manual_seed(0)
on = NW.build(nav, action_bounded=True, initial_forward_speed=V0,
              log_std_min=-5.0, log_std_init=STD, steering_gain=GAIN)
on.load_state_dict(base.state_dict())

x = torch.randn(8, base.cfg.obs_dim)

# SC1: gain=0.0 must be bit-identical to the pre-existing path
m_base = base.policy.head(base.policy.body(x))
m_off = off.policy.head(off.policy.body(x))
check("SC1_default_gain_bitexact", torch.equal(m_base, m_off))

# SC2: the field index is taken from the frozen layout, not hardcoded
idx = base.policy.layout.slice_of("goal_dir_y_body").start
check("SC2_index_from_layout", idx == field_slices()["goal_dir_y_body"].start
      and idx == 8, f"idx={idx}")

# SC3: the prior adds exactly gain*goal_dir_y to the wz component of the loc only
loc0 = base.policy.forward(x).base.mean
locg = on.policy.forward(x).base.mean
d_expected = GAIN * x[:, idx]
check("SC3_loc_wz_increment", torch.allclose(locg[:, 2], loc0[:, 2] + d_expected, atol=1e-7))
check("SC3_loc_vx_vy_untouched", torch.allclose(locg[:, :2], loc0[:, :2], atol=0.0))
check("SC3_compare_uses_same_params",
      torch.allclose(base.policy.forward(x).base.mean[:, 2], loc0[:, 2], atol=0.0))

# SC4: distribution stays well formed (log_prob finite, action strictly inside (-1,1))
dist = on.policy.forward(x)
s = dist.sample()
lp = dist.log_prob(s)
check("SC4_log_prob_finite", torch.isfinite(lp).all().item())
check("SC4_action_strictly_bounded", bool((s.abs() < 1.0).all()))

# SC5: sign convention — goal on the left (goal_dir_y>0) must increase wz
xl = x.clone()
xl[:, idx] = 1.0
xr = x.clone()
xr[:, idx] = -1.0
wz_l = on.policy.forward(xl).base.mean[:, 2]
wz_r = on.policy.forward(xr).base.mean[:, 2]
check("SC5_sign_left_positive", bool((wz_l > wz_r).all()))

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
