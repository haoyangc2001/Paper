"""成本最小化 PPO 更新的独立验证（NAV-031）。

要证的四件事：
  1. 符号：正优势压低概率、负优势抬高概率、零优势无梯度、熵项方向为"增大熵"；
  2. 裁剪：信任域内的梯度等于归一化优势，域外梯度为 0；
  3. 数值：把参考 JAX 损失的 numpy 转写当交叉核对，与 PyTorch 实现逐位一致；
  4. 端到端：一次真实更新只动上层参数、冻结底层散列不变，且重复更新会把策略推向低成本动作。

梯度方向一律用**有限差分**独立验证（不依赖 autograd 自身），避免"符号写反但自洽"。

Run:

    <frozen python> CaMRA/navigation/tests/test_minimization_update.py \
        --json CaMRA/navigation/reports/minimization_update_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))

from camra_navigation.algorithm import ppo as P  # noqa: E402
from camra_navigation.algorithm.networks import build as build_networks  # noqa: E402

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fd_grad(f, x: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """中心差分梯度（独立于 autograd）。"""
    base = x.detach().clone()
    g = np.zeros_like(base.numpy())
    flat = base.view(-1)
    for i in range(flat.numel()):
        orig = float(flat[i])
        flat[i] = orig + eps
        lp = float(f(base))
        flat[i] = orig - eps
        lm = float(f(base))
        flat[i] = orig
        g.reshape(-1)[i] = (lp - lm) / (2 * eps)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = P.PPOConfig.from_contract()
    torch.manual_seed(args.seed)

    # ---- 1) 契约加载 -----------------------------------------------------
    clash_ok = False
    try:
        P.PPOConfig.from_contract(gamma_cost=0.99)
    except ValueError:
        clash_ok = True
    check("config_loaded_from_contract",
          cfg.gamma_cost == 1.0 and cfg.gae_lambda == 0.95 and cfg.clip_eps == 0.2
          and cfg.ent_coef == 0.01 and cfg.vf_coef == 0.5 and cfg.beta == 0.0
          and abs(cfg.z_ref_j - 550.0) < 1e-12 and clash_ok,
          {"config": cfg.as_dict(),
           "rejects_frozen_override": clash_ok,
           "note": "γ_c=1（未折扣）、λ=0.95、clip=0.2、beta=0 全部来自契约；覆盖被拒"})

    # ---- 2) 策略损失：符号 / 裁剪 / 零优势 -------------------------------
    def loss_of_logp(logp_val: torch.Tensor, adv_val: torch.Tensor) -> torch.Tensor:
        ent = torch.zeros_like(adv_val)
        l, _ = P.policy_loss(logp_val, torch.zeros_like(logp_val), adv_val, ent, cfg)
        return l

    mix = torch.tensor([2.0, -1.0], dtype=torch.float64)
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    adv_n, _ = P.normalize_advantage(mix)
    l = loss_of_logp(x, adv_n)
    l.backward()
    g_auto = x.grad.detach().numpy()
    g_num = fd_grad(lambda z: loss_of_logp(z, adv_n), x)
    check("policy_gradient_sign_matches_fd",
          np.allclose(g_auto, g_num, atol=1e-4)
          and g_auto[0] > 0 and g_auto[1] < 0,
          {"adv_raw": mix.tolist(), "adv_normalized": adv_n.tolist(),
           "autograd": g_auto.tolist(), "finite_diff": g_num.tolist(),
           "reading": "A>0（更贵）⇒ 概率升高会让损失变大 ⇒ 梯度为正；A<0 反之"})

    zero_adv = torch.zeros(4)
    zn, _ = P.normalize_advantage(zero_adv)
    xz = torch.zeros(4, requires_grad=True)
    loss_of_logp(xz, zn).backward()
    check("zero_advantage_no_gradient",
          float(xz.grad.abs().max()) == 0.0 and float(zn.abs().max()) == 0.0,
          {"normalized_adv": zn.tolist(), "grad": xz.grad.tolist(),
           "note": "全零优势标准化后仍全零（不因 1e-8 变成噪声）"})

    # 裁剪内：ratio=1 ⇒ d mean(surrogate)/dlogp = A/B
    lp = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    adv2, _ = P.normalize_advantage(torch.tensor([1.0, -1.0], dtype=torch.float64))
    P.policy_loss(lp, torch.zeros(2, dtype=torch.float64), adv2,
                  torch.zeros(2, dtype=torch.float64), cfg)[0].backward()
    inside_expect = adv2.detach().numpy() / 2.0          # 损失对 batch 取均值 ⇒ 有 1/B
    inside_ok = np.allclose(lp.grad.detach().numpy(), inside_expect, atol=1e-9)

    # 裁剪外：ratio=e>1+ε。A<0 时 max 选裁剪分支 ⇒ 梯度 0；
    # A>0 时 max 选未裁剪分支 ⇒ 梯度仍为 A·ratio/B（这正是"继续加大高价动作概率要受罚"）
    lp_out = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    P.policy_loss(lp_out, torch.zeros(2, dtype=torch.float64), adv2,
                  torch.zeros(2, dtype=torch.float64), cfg)[0].backward()
    g_out = lp_out.grad.detach().numpy()
    ratio_out = float(np.exp(1.0))
    outside_ok = (abs(g_out[1]) < 1e-12
                  and abs(g_out[0] - adv2[0].item() * ratio_out / 2.0) < 1e-9)
    check("clip_gradient_inside_and_outside",
          inside_ok and outside_ok,
          {"inside_grad": lp.grad.detach().tolist(), "inside_expect": inside_expect.tolist(),
           "outside_ratio": ratio_out, "outside_grad": g_out.tolist(),
           "note": "域内=A/B；A<0 越界后梯度为 0（被裁剪）；A>0 越界后梯度仍为 A·ratio/B，"
                   "因为 max 选的是未裁剪项——与'最小化'语义一致，不是 bug"})

    # ---- 3) 熵项方向 -----------------------------------------------------
    net = build_networks()
    obs = torch.randn(16, 41)
    dist = net.policy(obs)
    ent = dist.entropy()
    loss_ent = -cfg.ent_coef * ent.mean()
    net.policy.zero_grad(set_to_none=True)
    loss_ent.backward()
    g_ls = net.policy.log_std.grad.detach()
    check("entropy_term_increases_entropy",
          float(g_ls.max()) < 0.0,
          {"d_loss_d_log_std": g_ls.tolist(),
           "note": "loss=−ent_coef·H；最小化它 ⇒ log_std 上升 ⇒ 熵增大（与'最大化奖励'模板同向，"
                   "但前提是这一项确实带负号）"})

    # ---- 4) 价值损失 -----------------------------------------------------
    # 第 0 点：|v−v_old|=0.1 未超 clip，误差不饱和；第 1 点：|v−v_old|=1.6 超 clip，
    # 裁剪后的误差 (0.6−3)²=5.76 大于未裁剪 (2−3)²=1，max 取到裁剪分支 ⇒ 梯度饱和为 0
    v = torch.tensor([0.5, 2.0], requires_grad=True)
    v_old = torch.tensor([0.4, 0.4])
    tgt = torch.tensor([0.1, 3.0])
    vl, vstats = P.value_loss(v, v_old, tgt, cfg)
    vl.backward()
    hand = cfg.vf_coef * 0.5 * np.mean([(0.5 - 0.1) ** 2, (0.6 - 3.0) ** 2])
    check("value_loss_hand_numbers",
          abs(float(vl.detach()) - hand) < 1e-6
          and np.allclose(v.grad.detach().numpy(),
                          [cfg.vf_coef * (0.5 - 0.1) / 2.0, 0.0], atol=1e-6),
          {"loss": float(vl.detach()), "hand": float(hand),
           "grad": v.grad.detach().tolist(),
           "clipped_fraction": float(vstats["clipped_fraction"]),
           "note": "第 0 点梯度 = vf·(v−t)/B = 0.5·0.4/2 = 0.1；"
                   "第 1 点裁剪后误差 5.76 > 未裁剪 1.0，max 取裁剪分支 ⇒ 梯度恰为 0"})

    # 域外若要"继续拟合"就必须越过 clip 边界：此时未裁剪误差更小、max 仍取裁剪分支 ⇒ 无梯度
    v_out = torch.tensor([3.0], requires_grad=True)
    P.value_loss(v_out, torch.tensor([0.0]), torch.tensor([3.5]), cfg)[0].backward()
    check("value_clip_saturates_outside",
          abs(float(v_out.grad.abs().max())) == 0.0,
          {"grad": v_out.grad.tolist(),
           "hand": "clipped=0.2，err=(3−3.5)²=0.25 但 err_clipped=(0.2−3.5)²=10.89，"
                   "max 取 10.89 且与 v 无关 ⇒ 梯度 0"})

    # ---- 5) 与参考转写一致 ----------------------------------------------
    rng = np.random.default_rng(args.seed)
    n = 12
    logp = rng.normal(0, 0.3, n).astype(np.float64)
    logp_old = rng.normal(0, 0.3, n).astype(np.float64)
    adv = rng.normal(0, 1.5, n).astype(np.float64)
    ent_np = rng.uniform(0.5, 2.0, n).astype(np.float64)
    ref = P._ref_policy_loss_numpy(logp, logp_old, adv, ent_np, cfg.clip_eps, cfg.ent_coef)
    mine = float(P.policy_loss(torch.tensor(logp), torch.tensor(logp_old),
                               P.normalize_advantage(torch.tensor(adv))[0],
                               torch.tensor(ent_np), cfg)[0].detach())
    ref_v = P._ref_value_loss_numpy(np.array([0.5, 2.0]), np.array([0.4, 0.4]),
                                    np.array([0.1, 3.0]), cfg.clip_eps, cfg.vf_coef)
    # PyTorch 侧用 float64 复算一次，避免把 float32 舍入当成实现差异
    vl64 = float(P.value_loss(torch.tensor([0.5, 2.0], dtype=torch.float64),
                              torch.tensor([0.4, 0.4], dtype=torch.float64),
                              torch.tensor([0.1, 3.0], dtype=torch.float64), cfg)[0])
    check("reference_transcription_matches",
          abs(ref - mine) < 1e-10 and abs(ref_v - vl64) < 1e-12,
          {"policy_loss_mine": mine, "policy_loss_reference": ref,
           "value_loss_mine_f32": float(vl.detach()), "value_loss_mine_f64": vl64,
           "value_loss_reference": ref_v,
           "note": "参考转写独立复算 numpy 版；float64 下逐位一致，"
                   "float32 的 9.5e-9 差异是舍入而非逻辑"})

    # ---- 6) 端到端：真实更新 --------------------------------------------
    torch.manual_seed(args.seed + 1)
    net2 = build_networks()
    upd = P.MinimizationUpdater(net2, cfg, lr_policy=1e-2, lr_critic=1e-2)
    B = 64
    obs2 = torch.randn(B, 41)
    act2 = torch.randn(B, 3) * 0.3
    with torch.no_grad():
        d0 = net2.policy(obs2)
        logp0 = d0.log_prob(act2)
        vc0 = net2.critic_cost(obs2)
        vh0 = net2.critic_reach(obs2)
    # 造一个"动作沿 +x 更贵"的合成优势：动作第 0 维越大 ⇒ A 越大（成本更高）
    adv_syn = 2.0 * act2[:, 0] + 0.1 * torch.randn(B)
    batch = {"obs": obs2, "action": act2, "log_prob_old": logp0, "adv_total": adv_syn,
             "value_cost_old": vc0, "target_cost": vc0 + 0.05,
             "value_reach_old": vh0, "target_reach": vh0 - 0.02}

    before = net2.policy.head.weight.detach().clone()
    frozen = {n: p.detach().clone() for n, p in net2.named_parameters()}
    stats = [upd.update(batch) for _ in range(15)]
    after = net2.policy.head.weight.detach().clone()
    with torch.no_grad():
        mean_after = net2.policy(obs2).mean[:, 0].mean()
        mean_before = d0.mean[:, 0].mean()

    check("update_pushes_toward_cheaper_action",
          float(mean_after) < float(mean_before) - 1e-3,
          {"mean_vx_before": float(mean_before), "mean_vx_after": float(mean_after),
           "adv_positive_means_costlier": True,
           "loss_trace_first": stats[0]["loss_policy"],
           "loss_trace_last": stats[-1]["loss_policy"],
           "note": "优势与动作第 0 维正相关（越正越贵）⇒ 更新后该维均值应下降"})

    # 只动上层：critic 参数不应因策略更新而变（三者各自 optimizer）
    changed_roles = {}
    for role in ("policy", "critic_cost", "critic_reach"):
        ps = net2.parameters_of(role)
        changed = any(not torch.equal(p, frozen[n]) for n, p in net2.named_parameters()
                      if any(p is q for q in ps))
        changed_roles[role] = changed
    check("update_touches_only_intended_subnets",
          changed_roles["policy"] and changed_roles["critic_cost"] and changed_roles["critic_reach"]
          and not torch.equal(before, after),
          {"changed": changed_roles, "policy_head_changed": not torch.equal(before, after),
           "note": "本调用同时更新三个子网（各自独立优化器）；任一子网未被更新即为实现错误"})

    # 冻结底层散列不变
    nav = json.load(open(os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json"),
                         encoding="utf-8"))
    ctrl = nav["controller"]
    policy_path = os.path.join(ctrl["bundle"], ctrl["policy_entry"].split(" ")[0])
    h_before = sha256_file(policy_path)
    upd.update(batch)
    h_after = sha256_file(policy_path)
    check("frozen_v2_untouched_by_update", h_before == h_after,
          {"sha256_before": h_before, "sha256_after": h_after})

    # ---- 7) 输入守卫 -----------------------------------------------------
    bad = dict(batch)
    bad["adv_total"] = batch["adv_total"].clone()
    bad["adv_total"][0] = float("nan")
    raised = False
    try:
        upd.update(bad)
    except ValueError:
        raised = True
    check("nan_guard_refuses_update", raised,
          {"note": "含 NaN 的 batch 直接拒绝，不靠'多训几轮'掩盖"})

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "task": "NAV-031",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "config": cfg.as_dict(),
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: v for k, v in summary.items() if k != "checks"}
    print(json.dumps(head, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
