"""NAV-064 盖章段隔离的**离线一步反事实**：不训练，只算一次策略梯度。

做法：取第 11 轮最终检查点的策略头，在冻结 rollout 的真实样本上算三次梯度
（同一批样本、同一组权重、同一熵系数），只改「哪些样本参与损失」：

  g_all      全样本（第 11 轮实际做的事）
  g_masked   屏蔽 y=1 段（本次改动）
  g_live     只在 y=0 段上算（理想目标，用于对照 g_masked 是否等价）
  g_dead     只在 y=1 段上算（封印这部分样本被拿掉的"份量"）

输出：各自范数、两两余弦、以及 g_dead 与 g_live 的夹角。若 g_dead 与 g_live
方向相反且份量可观，说明盖章段不是"没信号"，而是"反着推"。

只读：不写 checkpoint、不训练、不改任何项目代码。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402

WORK = "/pub/data/caohy/work/camra_navigation"
OUT = os.path.join(NAV, "reports", "training_round_r11_grad_counterfactual.json")
RUNS = [("uncalibrated", "nav064_r11_uncalibrated_s0"),
        ("full", "nav064_r11_full_s0")]
CKPT = "state_u600.pt"
N_CHUNKS = 24          # 取每个 env 末尾的 chunk（与 u600 权重最接近）
TAIL = True


def load_samples(run_id: str) -> dict[str, torch.Tensor]:
    obs, act, lp, adv, y, tgt, vr = [], [], [], [], [], [], []
    for env_dir in sorted(glob.glob(os.path.join(WORK, run_id, "rollout_env*"))):
        fs = sorted(glob.glob(os.path.join(env_dir, "rollout", "chunk_*.npz")))
        if not fs:
            continue
        use = fs[-N_CHUNKS:] if TAIL else fs[:N_CHUNKS]
        for p in use:
            with np.load(p) as z:
                obs.append(np.asarray(z["obs"], dtype=np.float32))
                act.append(np.asarray(z["action_raw"], dtype=np.float32))
                lp.append(np.asarray(z["log_prob"], dtype=np.float32))
                adv.append(np.asarray(z["obj_adv_total"], dtype=np.float32))
                y.append(np.asarray(z["safety_y"], dtype=np.float32))
                tgt.append(np.asarray(z["obj_target_reach"], dtype=np.float32))
                vr.append(np.asarray(z["value_reach"], dtype=np.float32))
    return {"obs": torch.as_tensor(np.concatenate(obs)),
            "action": torch.as_tensor(np.concatenate(act)),
            "log_prob_old": torch.as_tensor(np.concatenate(lp)),
            "adv_total": torch.as_tensor(np.concatenate(adv)),
            "target_reach": torch.as_tensor(np.concatenate(tgt)),
            "value_reach_old": torch.as_tensor(np.concatenate(vr)),
            "live": torch.as_tensor((np.concatenate(y) <= 0.5).astype(np.float32))}


def grads(net: NW, cfg: P.PPOConfig, data: dict, *, mask: torch.Tensor | None,
          subset: torch.Tensor | None = None) -> np.ndarray:
    """在（可选）子集上算策略损失对策略参数的梯度，返回展平向量。"""
    obs = data["obs"] if subset is None else data["obs"][subset > 0.5]
    act = data["action"] if subset is None else data["action"][subset > 0.5]
    lp_old = data["log_prob_old"] if subset is None else data["log_prob_old"][subset > 0.5]
    adv = data["adv_total"] if subset is None else data["adv_total"][subset > 0.5]
    m = mask if subset is None else (None if mask is None else mask[subset > 0.5])
    adv_n, _ = P.normalize_advantage(adv, mask=m)
    for p in net.parameters_of("policy"):
        p.grad = None
    dist = net.policy(obs)
    loss, _ = P.policy_loss(dist.log_prob(act), lp_old, adv_n, dist.entropy(), cfg, mask=m)
    loss.backward()
    g = np.concatenate([p.grad.detach().reshape(-1).numpy()
                        for p in net.parameters_of("policy") if p.grad is not None])
    for p in net.parameters_of("policy"):
        p.grad = None
    return g


def reach_diag(net: NW, cfg: P.PPOConfig, data: dict) -> dict:
    """reach 评价网络的回归损失里，盖章段占多少份量（决定 mask_reach 是否有用）。"""
    obs, live = data["obs"], data["live"]
    vh = net.critic_reach(obs)
    err = (vh - data["target_reach"]).square()
    with torch.no_grad():
        tot = float(err.sum())
        dead_mass = float(err[live < 0.5].sum())
    for p in net.parameters_of("critic_reach"):
        p.grad = None
    loss_m, _ = P.value_loss(vh, data["value_reach_old"], data["target_reach"], cfg,
                             mask=live)
    loss_m.backward()
    g_mask = np.concatenate([p.grad.detach().reshape(-1).numpy()
                             for p in net.parameters_of("critic_reach")
                             if p.grad is not None])
    for p in net.parameters_of("critic_reach"):
        p.grad = None
    vh2 = net.critic_reach(obs)
    loss_a, _ = P.value_loss(vh2, data["value_reach_old"], data["target_reach"], cfg)
    loss_a.backward()
    g_all = np.concatenate([p.grad.detach().reshape(-1).numpy()
                            for p in net.parameters_of("critic_reach")
                            if p.grad is not None])
    for p in net.parameters_of("critic_reach"):
        p.grad = None
    return {
        "err_mass_share_dead": round(dead_mass / max(1e-12, tot), 4),
        "err_mean_live": round(float(err[live > 0.5].mean()), 6),
        "err_mean_dead": round(float(err[live < 0.5].mean()), 6) if bool((live < 0.5).any()) else None,
        "loss_masked": round(float(loss_m), 6), "loss_all": round(float(loss_a), 6),
        "grad_norm_masked": round(float(np.linalg.norm(g_mask)), 6),
        "grad_norm_all": round(float(np.linalg.norm(g_all)), 6),
        "cos_masked_all": round(cos(g_mask, g_all), 4),
    }


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    nav = json.load(open(os.path.join(NAV, "config", "navigation_contract_v1.json"),
                        encoding="utf-8"))
    cfg = P.PPOConfig.from_contract()
    out = {"task": "NAV-064", "kind": "policy_gradient_counterfactual",
           "checkpoint": CKPT, "chunks_per_env": N_CHUNKS, "runs": []}
    for variant, run_id in RUNS:
        st = torch.load(os.path.join(WORK, run_id, CKPT), map_location="cpu",
                        weights_only=False)
        net = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
                       log_std_min=-5.0, log_std_init=-3.0, steering_gain=0.2)
        net.load_state_dict(st["model"])
        net.eval()
        data = load_samples(run_id)
        live, dead = data["live"], 1.0 - data["live"]
        n = int(data["obs"].shape[0])
        g_all = grads(net, cfg, data, mask=None)
        g_masked = grads(net, cfg, data, mask=live)
        g_live = grads(net, cfg, data, mask=None, subset=live)
        g_dead = grads(net, cfg, data, mask=None, subset=dead)
        rec = {
            "variant": variant, "run": run_id, "samples": n,
            "dead_fraction": round(float(dead.mean()), 4),
            "reach": reach_diag(net, cfg, data),
            "norms": {"all": round(float(np.linalg.norm(g_all)), 6),
                      "masked": round(float(np.linalg.norm(g_masked)), 6),
                      "live_only": round(float(np.linalg.norm(g_live)), 6),
                      "dead_only": round(float(np.linalg.norm(g_dead)), 6)},
            "cosine": {"all_vs_masked": round(cos(g_all, g_masked), 4),
                       "masked_vs_live_only": round(cos(g_masked, g_live), 4),
                       "live_vs_dead": round(cos(g_live, g_dead), 4),
                       "all_vs_dead": round(cos(g_all, g_dead), 4)},
            "dead_share_of_norm": round(float(np.linalg.norm(g_dead) /
                                             max(1e-12, np.linalg.norm(g_all))), 4),
            "note": ("cosine 越接近 −1 表示两组样本把策略往相反方向推；"
                     "dead_share_of_norm 是只算盖章段时的梯度范数占全样本的比例"),
        }
        out["runs"].append(rec)
        print(f"{variant}: n={n} dead={rec['dead_fraction']:.2%} "
              f"‖g_all‖={rec['norms']['all']:.4f} ‖g_masked‖={rec['norms']['masked']:.4f} "
              f"cos(all,masked)={rec['cosine']['all_vs_masked']:+.3f} "
              f"cos(live,dead)={rec['cosine']['live_vs_dead']:+.3f} "
              f"dead_share={rec['dead_share_of_norm']:.2f}")
        rch = rec["reach"]
        print(f"  reach 回归：盖章段占误差质量 {rch['err_mass_share_dead']:.2%}，"
              f"误差均值 live={rch['err_mean_live']} dead={rch['err_mean_dead']}；"
              f"‖g‖ masked={rch['grad_norm_masked']:.4f} all={rch['grad_norm_all']:.4f} "
              f"cos={rch['cos_masked_all']:+.3f}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("SAVED", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
