"""NAV-064 策略层仪器化（policy_grad_decomp）的确定性验收 PD1–PD6。

开关的承诺只有两句：
  1) **关闭时逐位不变**（不新增字段、不改参数更新）；
  2) **打开时只读**——多算两个梯度范数，参数更新必须与关闭时逐位相同。

  PD1 关闭时不新增字段
  PD2 打开时字段齐全且取值范围合法
  PD3 打开时参数更新与关闭时**逐位相同**（只读，不改更新）
  PD4 ent_coef=0 时熵项梯度恰为 0、占比恰为 0（恒等式自检）
  PD5 掩码全 1 与不开掩码逐位等价
  PD6 行为侧记录（均值前进命令、转向幅度）与批内动作精确一致
"""
import dataclasses
import json
import os
import sys

import torch

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402

nav = json.load(open(os.path.join(NAV, "config", "navigation_contract_v1.json"),
                     encoding="utf-8"))
cfg = P.PPOConfig.from_contract()
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


torch.manual_seed(11)
n = 16
obs_dim = int(nav["observation_spec"]["total_dim"])
obs = torch.randn(n, obs_dim)
action = torch.rand(n, 3) * 1.6 - 0.8
lp_old = -1.0 * torch.ones(n)
adv = torch.cat([torch.tensor([0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.14, 0.07]),
                 torch.full((8,), 0.02)])
live = torch.cat([torch.ones(8), torch.zeros(8)]).float()
ones = torch.ones(n)
v = torch.cat([torch.full((8,), 0.50), torch.full((8,), 0.99)])
tg = torch.cat([torch.full((8,), 0.47), torch.ones(8)])
batch = {"obs": obs, "action": action, "log_prob_old": lp_old, "adv_total": adv,
         "value_cost_old": v, "target_cost": tg,
         "value_reach_old": v, "target_reach": tg}


def make_pair(ent_coef=None):
    c = cfg if ent_coef is None else dataclasses.replace(cfg, ent_coef=ent_coef)
    torch.manual_seed(3)
    a = NW.build(nav)
    torch.manual_seed(3)
    b = NW.build(nav)
    b.load_state_dict(a.state_dict())
    return (P.MinimizationUpdater(a, c, lr_policy=1e-3, lr_critic=1e-3),
            P.MinimizationUpdater(b, c, lr_policy=1e-3, lr_critic=1e-3,
                                  grad_decomp=True), a, b)


# ---- PD1 / PD2 / PD3 -------------------------------------------------
up_off, up_on, net_off, net_on = make_pair()
s_off = up_off.update(dict(batch))
s_on = up_on.update(dict(batch))
new_keys = {"grad_adv_norm", "grad_ent_norm", "grad_ent_share",
            "grad_adv_ent_cos", "act_fwd_mean", "act_steer_absmean"}
check("PD1_off_adds_no_field", not (new_keys & set(s_off)), str(sorted(set(s_off))))
check("PD2_fields_present", new_keys <= set(s_on),
      f"adv={s_on.get('grad_adv_norm'):.4f} ent={s_on.get('grad_ent_norm'):.4f} "
      f"share={s_on.get('grad_ent_share'):.4f}")
check("PD2_ranges", s_on["grad_adv_norm"] > 0.0 and s_on["grad_ent_norm"] >= 0.0
      and 0.0 <= s_on["grad_ent_share"] <= 1.0
      and -1.0 <= s_on["grad_adv_ent_cos"] <= 1.0)

same_params = all(torch.equal(p_off, p_on) for p_off, p_on
                  in zip(net_off.state_dict().values(), net_on.state_dict().values()))
common_off = {k: v for k, v in s_off.items()}
check("PD3_params_bit_identical", same_params,
      "打开仪器化后参数更新逐位相同" if same_params else "参数被改动")
check("PD3_old_stats_bit_identical",
      all(s_on[k] == v for k, v in common_off.items()),
      f"Δloss_policy={abs(s_on['loss_policy'] - s_off['loss_policy']):.3e}")

# ---- PD4 ent_coef=0 -------------------------------------------------
_, up0, _, _ = make_pair(ent_coef=0.0)
s0 = up0.update({**batch})
check("PD4_zero_ent_coef_zero_ent_grad",
      s0["grad_ent_norm"] == 0.0 and s0["grad_ent_share"] == 0.0,
      f"ent_norm={s0['grad_ent_norm']:.3e} share={s0['grad_ent_share']:.3e}")

# ---- PD5 掩码全 1 == 不开掩码 ---------------------------------------
pr = P.policy_grad_terms
net = NW.build(nav)
a_n, _ = P.normalize_advantage(adv)
d_none = pr(net, obs, action, lp_old, a_n, cfg)
d_ones = pr(net, obs, action, lp_old, a_n, cfg, mask=ones)
d_live = pr(net, obs, action, lp_old, P.normalize_advantage(adv, mask=live)[0],
            cfg, mask=live)
check("PD5_all_ones_equals_none",
      d_none == d_ones, f"Δadv={abs(d_none['grad_adv_norm'] - d_ones['grad_adv_norm']):.3e}")
check("PD5_mask_changes_signal",
      abs(d_live["grad_adv_norm"] - d_none["grad_adv_norm"]) > 1e-4,
      f"adv {d_none['grad_adv_norm']:.4f} -> {d_live['grad_adv_norm']:.4f}")

# ---- PD6 行为侧记录 -------------------------------------------------
check("PD6_action_stats_exact",
      abs(s_on["act_fwd_mean"] - float(action[:, 0].mean())) < 1e-9
      and abs(s_on["act_steer_absmean"] - float(action[:, 2].abs().mean())) < 1e-9,
      f"fwd={s_on['act_fwd_mean']:.6f} steer={s_on['act_steer_absmean']:.6f}")

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
