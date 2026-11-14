"""NAV-064 盖章段隔离（dead_step_mask）的确定性验收 DM1–DM7。

开关的核心承诺只有一句：**关闭时逐位不变，打开时只改「哪些样本参与损失」**，
不改 W / τ* / 可行性判据 / 目标定义。这里逐条验：

  DM1 默认路径：mask=None 与 mask=全 1 逐位等价
  DM2 标准化的统计量只取有效样本
  DM3 策略损失 == 只在有效子集上算的损失（逐位）
  DM4 reach 价值损失 == 只在有效子集上算的损失（逐位）
  DM5 反例：被遮蔽样本不参与统计时，它们的常数偏置不再进入梯度
  DM6 更新器：全 1 掩码与不开掩码逐位等价（不含新增的占比字段）
  DM7 守卫：有效样本不足 2 个时拒绝更新（不静默训练）
"""
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


torch.manual_seed(7)
n = 12
adv = torch.tensor([0.10, 0.12, 0.09, 0.11, 0.13, 0.08,
                    0.02, 0.02, 0.02, 0.02, 0.02, 0.02])       # 后 6 步＝盖章段
live = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
ones = torch.ones(n)

# ---- DM1 / DM2 ------------------------------------------------------
a_none, s_none = P.normalize_advantage(adv)
a_ones, s_ones = P.normalize_advantage(adv, mask=ones)
check("DM1_default_bitexact", torch.equal(a_none, a_ones)
      and s_none == s_ones)

a_mask, s_mask = P.normalize_advantage(adv, mask=live)
exp_mean = adv[live > 0.5].mean()
exp_std = adv[live > 0.5].std(unbiased=False)
check("DM2_stats_from_live_only",
      abs(s_mask["adv_mean"] - float(exp_mean)) < 1e-9
      and abs(s_mask["adv_std"] - float(exp_std)) < 1e-9,
      f"mean={s_mask['adv_mean']:.6f} std={s_mask['adv_std']:.6f}")

# ---- DM3 ------------------------------------------------------------
logp = torch.tensor([-0.30, -0.32, -0.28, -0.31, -0.29, -0.33,
                     -0.30, -0.31, -0.29, -0.32, -0.30, -0.31])
logp_old = logp + torch.tensor([0.01, -0.01, 0.0, 0.02, -0.02, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ent = torch.tensor([1.00, 1.01, 0.99, 1.02, 0.98, 1.00,
                    2.50, 2.50, 2.50, 2.50, 2.50, 2.50])        # 盖章段熵故意很高
l_all, st_all = P.policy_loss(logp, logp_old, a_mask, ent, cfg)
sel = live > 0.5
l_live, _ = P.policy_loss(logp[sel], logp_old[sel], a_mask[sel], ent[sel], cfg)
l_masked, st_masked = P.policy_loss(logp, logp_old, a_mask, ent, cfg, mask=live)
check("DM3_policy_loss_equals_live_subset",
      torch.allclose(l_masked, l_live, atol=1e-7),
      f"masked={float(l_masked):.9f} subset={float(l_live):.9f}")
check("DM3_entropy_not_dragged_by_dead",
      abs(float(st_masked["entropy_mean"]) - 1.0) < 0.02,
      f"entropy_mean={float(st_masked['entropy_mean']):.4f} (全样本口径 "
      f"{float(st_all['entropy_mean']):.4f})")

# ---- DM4 ------------------------------------------------------------
v = torch.tensor([0.50, 0.52, 0.49, 0.51, 0.50, 0.48,
                  0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
v_old = v - 0.01
tg = torch.tensor([0.45, 0.50, 0.47, 0.49, 0.52, 0.46,
                   1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
l4_masked, _ = P.value_loss(v, v_old, tg, cfg, mask=live)
l4_live, _ = P.value_loss(v[sel], v_old[sel], tg[sel], cfg)
check("DM4_value_loss_equals_live_subset",
      torch.allclose(l4_masked, l4_live, atol=1e-8),
      f"masked={float(l4_masked):.9f} subset={float(l4_live):.9f}")

# ---- DM5 反例：常数偏置 ---------------------------------------------
bias_all = float(((adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8))[~sel].mean())
frac_dead_nonzero = float(((a_none[~sel]).abs() > 1e-6).float().mean())
check("DM5_dead_get_constant_nonzero_adv_unmasked",
      abs(bias_all) > 0.1 and frac_dead_nonzero == 1.0,
      f"未屏蔽时盖章段的标准化优势恒为 {bias_all:+.4f}")
check("DM5_masked_excludes_them_from_loss",
      torch.allclose(l_masked, l_live, atol=1e-7)
      and float((a_mask[~sel] - a_mask[~sel][0]).abs().max()) == 0.0,
      "遮蔽后它们只影响不到损失（占比字段另记）")

# ---- DM6 更新器：全 1 掩码 == 不开掩码 -------------------------------
obs_dim = int(nav["observation_spec"]["total_dim"])
torch.manual_seed(3)
net_a = NW.build(nav)
torch.manual_seed(3)
net_b = NW.build(nav)
net_b.load_state_dict(net_a.state_dict())
batch = {
    "obs": torch.randn(n, obs_dim),
    "action": torch.rand(n, 3) * 1.6 - 0.8,
    "log_prob_old": -1.0 * torch.ones(n),
    "adv_total": adv,
    "value_cost_old": v,
    "target_cost": tg,
    "value_reach_old": v,
    "target_reach": tg,
}
up = P.MinimizationUpdater(net_a, cfg, lr_policy=1e-3, lr_critic=1e-3)
up_b = P.MinimizationUpdater(net_b, cfg, lr_policy=1e-3, lr_critic=1e-3)
s_plain = up.update(dict(batch))
s_ones = up_b.update({**batch, "live_policy": ones, "live_reach": ones})
common = {k: s_plain[k] for k in s_plain}
check("DM6_updater_all_ones_equivalent",
      all(abs(s_plain[k] - s_ones[k]) <= 1e-6 for k in s_plain),
      f"Δloss_policy={abs(s_plain['loss_policy'] - s_ones['loss_policy']):.3e}")
check("DM6_fraction_stats_present",
      s_ones.get("policy_mask_kept_fraction") == 1.0
      and s_ones.get("reach_mask_kept_fraction") == 1.0
      and "policy_mask_kept_fraction" not in s_plain)

# ---- DM7 守卫 -------------------------------------------------------
only_one = torch.zeros(n, dtype=torch.float32)
only_one[0] = 1.0
try:
    up.update({**batch, "live_policy": only_one})
    check("DM7_guard_rejects_tiny_live_set", False, "没有抛错")
except ValueError as exc:
    check("DM7_guard_rejects_tiny_live_set", "保留样本不足" in str(exc), str(exc)[:40])

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
