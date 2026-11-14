"""成本最小化 PPO 更新（NAV-031）。

迁移自只读参考 ``RC-PPO/rl/EFPPO_utils.py`` 的 ``_ecefppo_update``。参考做的是**最小化**
（成本越低越好），因此三处符号与常规"奖励最大化"模板相反，任何一处照抄都会把目标反过来：

  · 策略损失用 ``max(ratio·A, clip(ratio)·A)``（**不是** min）
  · 熵项是 ``− ent_coef · H``（最小化它等价于最大化熵）
  · 损失整体 ``.backward()`` 后 ``optimizer.step()`` 直接下降，**不做梯度上升**

梯度方向的自洽性：A > 0 表示该动作比平均更贵，最小化 ``max(ratio·A, …)`` 会压低其概率；
A < 0 则抬高。这条在 ``tests/test_minimization_update.py`` 里用有限差分独立验过。

价值损失（两个 critic 同式）：对"相对旧价值的改动量"做裁剪，取两段误差的较大者
（悲观界），乘 ``vf_coef`` 后取 0.5·mean。

成本折扣/尺度/停止规则**从契约显式读取**（``PPOConfig.from_contract()``），不在这里另设默认：
优化符号正确不等于目标语义正确——归一化只标准化优势，不改 W 的单位与判据。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

_THIS = os.path.abspath(__file__)
NAV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
VALUE_CONTRACT_PATH = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")


# ---------------------------------------------------------------- 有效样本掩码
def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """按 0/1 掩码取子集（掩码为 None 时**原样返回同一个张量**，保证逐位等价）。"""
    if mask is None:
        return x
    return x[mask > 0.5]


def mask_kept_fraction(mask: torch.Tensor | None) -> float | None:
    """被保留的样本占比；``mask is None`` 时返回 ``None``（旧路径不记录该字段）。"""
    if mask is None or mask.numel() == 0:
        return None
    return float((mask > 0.5).float().mean())



# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class PPOConfig:
    """更新超参。契约里冻结的量从这里读，默认值仅作占位并会在校验时被覆盖。"""

    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma_cost: float = 1.0
    gae_lambda: float = 0.95
    gamma_h_init: float = 0.999
    gamma_h_final: float = 0.99999
    beta: float = 0.0
    z_ref_j: float = 550.0
    contract_version: str = ""

    @classmethod
    def from_contract(cls, path: str = VALUE_CONTRACT_PATH, **overrides: Any) -> "PPOConfig":
        """按契约构造；契约冻结的量不允许被 overrides 覆盖。"""
        with open(path, encoding="utf-8") as fh:
            c = json.load(fh)
        est = c["estimator"]
        q = c["quantities"]
        frozen = {
            "clip_eps": float(est["clip_eps"]),
            "ent_coef": float(est["ent_coef"]),
            "vf_coef": float(est["vf_coef"]),
            "max_grad_norm": float(est["max_grad_norm"]),
            "gamma_cost": float(est["gamma_cost"]),
            "gae_lambda": float(est["gae_lambda"]),
            "gamma_h_init": float(est["gamma_h_init"]),
            "gamma_h_final": float(est["gamma_h_final"]),
            "beta": float(q["beta"]["value"]),
            "z_ref_j": float(q["Z_REF_J"]["value"]),
            "contract_version": str(c.get("version", "")),
        }
        clash = {k: overrides[k] for k in overrides if k in frozen and overrides[k] != frozen[k]}
        if clash:
            raise ValueError(f"这些量由契约冻结，不允许覆盖：{clash}")
        frozen.update({k: v for k, v in overrides.items() if k not in frozen})
        return cls(**frozen)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------- 损失
def normalize_advantage(adv: torch.Tensor, eps: float = 1e-8,
                        mask: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, dict[str, float]]:
    """按 minibatch 标准化优势：(A − mean)/(std + 1e-8)。

    注意：这是对**优势**做标量重标定，不改残差 W 的定义、单位或 beta 判据。
    标准化后均值 0、标准差约 1；全零优势会保持全零（不会因 eps 爆炸）。

    ``mask``（NAV-064 盖章段隔离）非空时，mean/std 只在被保留的样本上算，
    但仍对**全部**样本做同一变换。原因：不可逆安全历史触发后的步，其目标被
    ``h≡1`` 钉死，优势与「离终点多远」无关；若让它们参与统计，它们会被整体
    推成 ``(adv_dead − mean)/std`` 的常数偏置（r11 实测 −0.28～−0.74），
    等于对「进入警戒区之后的动作」统一判「便宜」，而最小化更新会抬高这些
    动作的概率。这不是标准化尺度问题，是混入了一组不该参与统计的样本。
    """
    if adv.numel() == 0:
        raise ValueError("优势为空")
    sample = _apply_mask(adv, mask)
    if sample.numel() == 0:
        raise ValueError("掩码把全部样本都排除了，无法标准化优势")
    mean = sample.mean()
    std = sample.std(unbiased=False)
    out = (adv - mean) / (std + eps)
    return out, {"adv_mean": float(mean.detach()), "adv_std": float(std.detach())}


def policy_loss(log_prob: torch.Tensor, log_prob_old: torch.Tensor,
                adv_normalized: torch.Tensor, entropy: torch.Tensor,
                cfg: PPOConfig, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """成本最小化的策略损失：``mean(max(ratio·A, clip(ratio)·A)) − ent_coef·mean(H)``。

    用 max 而不是 min 是"最小化"语义的必然结果：取两段中较大的那个，即对成本上界悲观。
    ``log_prob`` 与 ``log_prob_old`` 必须是**同一分布族、同一动作变换**下的值
    （本仓库：归一化动作上的对角高斯，裁剪在 applied 处发生，不在这里）。

    ``mask`` 非空时，代理目标与熵项都只在被保留的样本上取均值（NAV-064 盖章段
    隔离）。熵项一并遮蔽是必要的：盖章段的优势被抹平后，唯一还在推梯度的就是
    熵奖励，那会把「已经进了警戒区之后」的策略单独推向高熵，即更随机、更不敢走。
    """
    ratio = torch.exp(log_prob - log_prob_old)
    lo, hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
    unclipped = ratio * adv_normalized
    clipped = torch.clamp(ratio, lo, hi) * adv_normalized
    surrogate = torch.maximum(unclipped, clipped)
    entropy_mean = _apply_mask(entropy, mask).mean()
    loss = _apply_mask(surrogate, mask).mean() - cfg.ent_coef * entropy_mean
    return loss, {"ratio": ratio.detach(), "surrogate": surrogate.detach(),
                  "entropy_mean": entropy_mean.detach(),
                  "clipped_fraction": ((ratio < lo) | (ratio > hi)).float().mean().detach()}


def value_loss(value: torch.Tensor, value_old: torch.Tensor, target: torch.Tensor,
               cfg: PPOConfig, mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """截断价值损失：对"相对旧价值的改动量"裁剪，取两段误差较大者（悲观界）。

    ``mask`` 非空时只在被保留的样本上取均值（NAV-064 只作用于 reach 评价网络：
    盖章段的目标恒等于 1，回归它们既无信息，又会把「接近终点」这一批特征往
    常数 1 上拉）。成本评价网络不遮蔽——它的目标在盖章段依然有定义。
    """
    clipped_value = value_old + torch.clamp(value - value_old, -cfg.clip_eps, cfg.clip_eps)
    err = (value - target) ** 2
    err_clipped = (clipped_value - target) ** 2
    worst = torch.maximum(err, err_clipped)
    worst_kept = _apply_mask(worst, mask)
    loss = cfg.vf_coef * 0.5 * worst_kept.mean()
    return loss, {"value_loss_raw": (0.5 * worst_kept.mean()).detach(),
                  "clipped_fraction": (torch.abs(value - value_old) > cfg.clip_eps)
                  .float().mean().detach()}


# ---------------------------------------------------------------- 更新器
class MinimizationUpdater:
    """三网各自的优化器 + 独立反传；只碰上层参数。"""

    ROLES = ("policy", "critic_cost", "critic_reach")

    def __init__(self, networks: nn.Module, cfg: PPOConfig | None = None,
                 lr_policy: float = 3e-4, lr_critic: float = 1e-3) -> None:
        self.cfg = cfg or PPOConfig.from_contract()
        self.net = networks
        groups = networks.param_groups()
        self.optimizers = {
            "policy": torch.optim.Adam(groups["policy"], lr=lr_policy),
            "critic_cost": torch.optim.Adam(groups["critic_cost"], lr=lr_critic),
            "critic_reach": torch.optim.Adam(groups["critic_reach"], lr=lr_critic),
        }

    def _step(self, role: str, loss: torch.Tensor) -> float:
        opt = self.optimizers[role]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters_of(role), self.cfg.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(grad_norm)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """一次 minibatch 更新。``batch`` 需含：

        obs, action, log_prob_old, adv_total,
        value_cost_old, target_cost, value_reach_old, target_reach

        可选（NAV-064 盖章段隔离，缺省 = 旧路径逐位不变）：

          live_policy  策略侧有效样本掩码（1/0）。同时作用于优势标准化的统计量、
                       代理目标与熵项。
          live_reach   reach 评价网络的有效样本掩码；成本评价网络永远不遮蔽。
        """
        need = ("obs", "action", "log_prob_old", "adv_total",
                "value_cost_old", "target_cost", "value_reach_old", "target_reach")
        missing = [k for k in need if k not in batch]
        if missing:
            raise KeyError(f"batch 缺少字段：{missing}")
        for k in need:
            if not torch.isfinite(batch[k]).all():
                raise ValueError(f"batch[{k!r}] 含非有限值，拒绝更新（不以扩大训练绕过）")

        live_policy = batch.get("live_policy")
        live_reach = batch.get("live_reach")
        for name, m in (("live_policy", live_policy), ("live_reach", live_reach)):
            if m is not None:
                if m.shape[0] != batch["obs"].shape[0]:
                    raise ValueError(f"{name} 长度 {m.shape[0]} ≠ 样本数 "
                                     f"{batch['obs'].shape[0]}")
                if float((m > 0.5).sum()) < 2:
                    raise ValueError(f"{name} 保留样本不足 2 个，拒绝更新")

        obs = batch["obs"]
        adv_n, adv_stats = normalize_advantage(batch["adv_total"], mask=live_policy)

        dist = self.net.policy(obs)
        log_prob = dist.log_prob(batch["action"])
        entropy = dist.entropy()
        loss_pi, pi_stats = policy_loss(log_prob, batch["log_prob_old"], adv_n, entropy,
                                        self.cfg, mask=live_policy)
        gn_pi = self._step("policy", loss_pi)

        vc = self.net.critic_cost(obs)
        loss_c, c_stats = value_loss(vc, batch["value_cost_old"], batch["target_cost"],
                                     self.cfg)
        gn_c = self._step("critic_cost", loss_c)

        vh = self.net.critic_reach(obs)
        loss_h, h_stats = value_loss(vh, batch["value_reach_old"], batch["target_reach"],
                                     self.cfg, mask=live_reach)
        gn_h = self._step("critic_reach", loss_h)

        out = {
            "loss_policy": float(loss_pi.detach()),
            "loss_cost": float(loss_c.detach()),
            "loss_reach": float(loss_h.detach()),
            "grad_norm_policy": gn_pi, "grad_norm_cost": gn_c, "grad_norm_reach": gn_h,
            "entropy": float(pi_stats["entropy_mean"]),
            "clipped_ratio_fraction": float(pi_stats["clipped_fraction"]),
            "value_cost_clipped_fraction": float(c_stats["clipped_fraction"]),
            **adv_stats,
        }
        kept_pi = mask_kept_fraction(live_policy)
        kept_h = mask_kept_fraction(live_reach)
        if kept_pi is not None:
            out["policy_mask_kept_fraction"] = kept_pi
        if kept_h is not None:
            out["reach_mask_kept_fraction"] = kept_h
        return out


# ---------------------------------------------------------------- 参考转写
def _ref_policy_loss_numpy(logp, logp_old, adv, entropy, clip_eps, ent_coef):
    """``_loss_fn_policy`` 的 numpy 转写（只做交叉核对）。"""
    import numpy as np

    ratio = np.exp(logp - logp_old)
    a = (adv - adv.mean()) / (adv.std() + 1e-8)
    surrogate = np.maximum(ratio * a, np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a)
    return float(surrogate.mean() - ent_coef * entropy.mean())


def _ref_value_loss_numpy(value, value_old, target, clip_eps, vf_coef):
    import numpy as np

    clipped = value_old + np.clip(value - value_old, -clip_eps, clip_eps)
    worst = np.maximum((value - target) ** 2, (clipped - target) ** 2)
    return float(vf_coef * 0.5 * worst.mean())
