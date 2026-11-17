"""预算条件策略与两个评价网络（NAV-029）。

契约来源（全部只读，不在此重新选择任何尺度或编码）：
  * obs-v2（41 维）与 cmd-v1（3 维动作）：``config/navigation_contract_v1.json``（NAV-013）
  * 价值目标、V_c/V_h 尺度、共同停止规则、估计器参数：``config/value_objective_contract_v1.json``
    （NAV-056）
  * 参考架构（只读）：``RC-PPO/model/actorcritic.py`` 的 ``Policy_Network`` / ``Value_Network``。
    该参考是 JAX/Flax 实现；本仓库上层框架已由 NAV-008 冻结为 PyTorch（jax 在环境中缺失），
    因此这里按**同一架构**重写，不引入 jax 依赖、不改变层数/宽度/初始化增益。

三个网络严格分开、各自独立参数（不把单 critic 直接拆成两半）：
  ``policy``        obs -> Independent(Normal(mean, std))，作用于**归一化动作** a ∈ R^3
  ``critic_cost``   obs -> V_c，单位 J / Z_REF_J（无量纲、非负）
  ``critic_reach``  obs -> V_h，无量纲，<= 0 表示可安全到达（min 型 Bellman）

动作映射（cmd-v1，仿射而非 tanh）::

    u_req = clip(a, -1, 1) * (vx_max, vy_max, wz_max) = clip(a, -1, 1) * (1.0, 0.4, 1.0)

契约明文要求用仿射而非 tanh：tanh 会系统性压低指令值，而低速段（0.2–0.31 m/s）必须精确可达。
裁剪只在 applied 处发生一次，本模块只做映射，不写日志、不做重采样。

预算条件怎么进来的：obs-v2 已含 ``z_ratio`` / ``safety_y`` / ``time_remaining_n``
（layout 第 39/40/41 维），因此"预算条件策略"是**通过观测**实现的，不另建旁路输入通道——
这样训练与观测不会各算一套（value contract quantities.z_ratio.note）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.distributions import Independent, Normal

_THIS = os.path.abspath(__file__)
NAV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
CONTRACT_PATH = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
VALUE_CONTRACT_PATH = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")

#: 输出量的单位说明（契约要求"有单位说明"；不在此处做任何换算）
UNIT_TABLE: dict[str, dict[str, str]] = {
    "action_normalized": {"unit": "无量纲", "range": "R^3（未裁剪，可超出 ±1）",
                          "note": "策略分布作用在归一化动作上；裁剪只发生在 applied"},
    "command": {"unit": "m/s, m/s, rad/s", "range": "(1.0, 0.4, 1.0) 乘子后的机身系速度",
                "note": "cmd-v1：u_req = clip(a,-1,1) * (1.0, 0.4, 1.0)"},
    "V_c": {"unit": "J / Z_REF_J（无量纲）", "range": ">= 0",
            "note": "剩余期望模型成本除以场景常数 Z_REF_J=550.0 J（NAV-035 由 225.6069 重定）"},
    "V_h": {"unit": "无量纲", "range": "<= 0 表示可安全到达",
            "note": "min_tau max(g_norm, C*y)；C=1.0，g_norm 以 L_REF=15 m 归一化"},
}


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    """读取导航契约与价值目标契约（只读）。"""
    with open(CONTRACT_PATH, encoding="utf-8") as fh:
        nav = json.load(fh)
    with open(VALUE_CONTRACT_PATH, encoding="utf-8") as fh:
        val = json.load(fh)
    return nav, val


# ---------------------------------------------------------------- 观测布局
@dataclass(frozen=True)
class ObservationLayout:
    """obs-v2 的具名切片；只按契约给出的顺序与维度构造，不做任何推断。"""

    names: tuple[str, ...]
    dims: tuple[int, ...]

    @property
    def total_dim(self) -> int:
        return sum(self.dims)

    @classmethod
    def from_contract(cls, nav_contract: dict[str, Any]) -> "ObservationLayout":
        layout = nav_contract["observation_spec"]["layout"]
        names = tuple(str(item["name"]) for item in layout)
        dims = tuple(int(item["dim"]) for item in layout)
        total = int(nav_contract["observation_spec"]["total_dim"])
        if sum(dims) != total:
            raise ValueError(f"layout 维度之和 {sum(dims)} ≠ total_dim {total}")
        return cls(names=names, dims=dims)

    def slice_of(self, name: str) -> slice:
        if name not in self.names:
            raise KeyError(f"obs-v2 中没有字段 {name!r}；现有 {self.names}")
        start = sum(self.dims[: self.names.index(name)])
        return slice(start, start + self.dims[self.names.index(name)])

    #: 预算/时间条件字段（"预算条件策略"通过这三项进入网络）
    BUDGET_FIELDS: tuple[str, ...] = ("z_ratio", "safety_y", "time_remaining_n")
    #: 地图编码字段（16 方位 bin / 4 面墙距 / 8 容量掩码）
    MAP_FIELDS: tuple[str, ...] = ("obstacle_bins", "wall_dist_n", "obstacle_mask")

    def budget_slices(self) -> dict[str, slice]:
        return {name: self.slice_of(name) for name in self.BUDGET_FIELDS}

    def map_slices(self) -> dict[str, slice]:
        fields = self.MAP_FIELDS + tuple(n for n in ("safety_margin_bins", "wall_margin_n")
                                        if n in self.names)
        return {name: self.slice_of(name) for name in fields}

    def index_of(self, name: str) -> int:
        sl = self.slice_of(name)
        if sl.stop - sl.start != 1:
            raise ValueError(f"字段 {name!r} 维度为 {sl.stop - sl.start}，不是标量")
        return sl.start


# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class NetworkConfig:
    """网络结构配置。默认值取自 RC-PPO 参考实现（只读参考，不是重新设计）。"""

    obs_dim: int = 41
    action_dim: int = 3
    hidden: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    log_std_init: float = 0.0
    hidden_gain: float = math.sqrt(2.0)
    policy_head_gain: float = 0.01
    value_head_gain: float = 1.0
    action_scale: tuple[float, float, float] = (1.0, 0.4, 1.0)

    def validate_against(self, nav_contract: dict[str, Any]) -> dict[str, Any]:
        """与契约逐项核对；不匹配就是实现错误，不能靠调参绕过。"""
        problems: list[str] = []
        obs_spec = nav_contract["observation_spec"]
        act_spec = nav_contract["action_spec"]
        layout = ObservationLayout.from_contract(nav_contract)

        if self.obs_dim != int(obs_spec["total_dim"]):
            problems.append(f"obs_dim {self.obs_dim} ≠ 契约 total_dim {obs_spec['total_dim']}")
        if layout.total_dim != self.obs_dim:
            problems.append(f"layout 求和 {layout.total_dim} ≠ obs_dim {self.obs_dim}")
        if self.action_dim != int(act_spec["action_dim"]):
            problems.append(f"action_dim {self.action_dim} ≠ 契约 {act_spec['action_dim']}")
        if list(act_spec["action_order"]) != ["vx", "vy", "wz"]:
            problems.append(f"动作顺序与契约不符：{act_spec['action_order']}")
        scale_text = "(" + ", ".join(repr(float(v)) for v in self.action_scale) + ")"
        if scale_text not in str(act_spec["mapping"]):
            problems.append(f"动作尺度 {scale_text} 未出现在契约映射文本中：{act_spec['mapping']}")
        if self.activation not in ("tanh", "relu"):
            problems.append(f"未知激活 {self.activation!r}")

        # 预算条件必须来自观测，而不是旁路输入
        for required in ObservationLayout.BUDGET_FIELDS + ObservationLayout.MAP_FIELDS:
            if required not in layout.names:
                problems.append(f"obs-v2 缺少预算条件字段 {required!r}")

        return {
            "ok": not problems,
            "problems": problems,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_scale": list(self.action_scale),
            "obs_layout": {"names": list(layout.names), "dims": list(layout.dims)},
            "budget_fields": {k: layout.index_of(k)
                              for k in ("z_ratio", "safety_y", "time_remaining_n")
                              if k in layout.names},
        }


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"未知激活 {name!r}")


def _mlp(in_dim: int, hidden: Sequence[int], act: str, gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        linear = nn.Linear(last, width)
        nn.init.orthogonal_(linear.weight, gain)
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        layers.append(_activation(act))
        last = width
    return nn.Sequential(*layers)


def _head(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.constant_(linear.bias, 0.0)
    return linear


# ---------------------------------------------------------------- 网络
#: rcppo_alignment_contract_v1.amend2（v1.1.0，2026-11-13 用户批准）：log_std
#: 的可学习范围。上界 1.0（std≈2.72）阻断四轮训练实测的失效模式——优势≈0 的
#: 样本上损失只剩 −ent_coef·H，普通最小化单调推高 std（实测熵 4.7→32、run 死
#: 亡）；下界 −5.0（std≈0.007）保底防止探索完全塌缩。初值 0（std=1）在范围内。
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class SquashedNormal(torch.distributions.Distribution):
    """网络层有界动作分布（NAV-064 修正 B，默认不启用）。

    ``u ~ Normal(mu, std)``，``a = tanh(u)``，所以动作天然落在 (-1, 1)，
    不再依赖 ``action_to_requested`` 的硬裁剪。密度按变量替换修正：

        ``log pi(a) = log mu(u) - sum log(1 - a^2 + eps)``,  ``u = atanh(a)``

    契约 ``action_probability`` 的 ``known_defect`` 登记了"裁剪处梯度为 0、
    策略长期顶格"；``observation.action_to_requested`` 的注释也明确写了
    "有界化是网络层职责"。本类只在该层解决，不改映射、不改功率模型、
    不改 W / 判据。

    与 SAC 的标准做法一致；熵用 ``H(mu) + E[log(1 - tanh(u)^2)]`` 的
    单样本可导估计（squashed 熵 <= 原高斯熵）。
    """
    arg_constraints: dict = {}
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__(batch_shape=loc.shape[:-1], event_shape=loc.shape[-1:])
        self.base = torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), 1)
        self.eps = float(eps)

    def _atanh(self, a: torch.Tensor) -> torch.Tensor:
        lim = 1.0 - self.eps
        c = a.clamp(-lim, lim)
        return 0.5 * (torch.log1p(c) - torch.log1p(-c))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        u = self._atanh(value)
        corr = torch.log1p(-value.square() + self.eps).sum(-1)
        return self.base.log_prob(u) - torch.clamp(corr, min=-50.0)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return torch.tanh(self.base.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base.rsample(sample_shape))

    def entropy(self) -> torch.Tensor:
        u = self.base.rsample()
        return self.base.entropy() + torch.log1p(-torch.tanh(u).square() + self.eps).sum(-1)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.base.base_dist.loc)

    @property
    def stddev(self) -> torch.Tensor:
        """变换前的尺度（诊断用；变换后无闭式 std）。"""
        return self.base.base_dist.scale


class PolicyNetwork(nn.Module):
    """obs -> Independent(Normal(mean, std))，作用在归一化动作上。"""

    def __init__(self, cfg: NetworkConfig, layout: ObservationLayout | None = None,
                 action_bounded: bool = False,
                 initial_forward_speed: float = 0.0,
                 log_std_min: float | None = None,
                 log_std_init: float | None = None,
                 steering_gain: float = 0.0,
                 avoidance_prior: dict | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        #: NAV-064 修正 B 的开关；默认 False => 与改动前逐位等价。
        self.action_bounded = bool(action_bounded)
        #: NAV-064 初始化修正（action_init_fix，默认关闭）：只在该模式下给
        #: 第一维动作一个能持续前进的初值，并抬高探索尺度下限。
        self.initial_forward_speed = float(initial_forward_speed)
        self.log_std_min = float(LOG_STD_MIN if log_std_min is None else log_std_min)
        self.log_std_init = float(
            cfg.log_std_init if log_std_init is None else log_std_init)
        self.body = _mlp(cfg.obs_dim, cfg.hidden, cfg.activation, cfg.hidden_gain)
        self.head = _head(cfg.hidden[-1], cfg.action_dim, cfg.policy_head_gain)
        # 与参考实现一致：log_std 是独立参数、初值 0（std=1），不随状态变化。
        # 可学习范围见上方 LOG_STD_MIN/LOG_STD_MAX（amend2）。
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), self.log_std_init))
        #: NAV-064 转向先验（steering_prior，默认 0.0 = 关闭）：在动作**均值**上
        #: 叠加 ``gain * sin(目标方位角)``（obs 的 ``goal_dir_y_body``）。只改分布的
        #: loc，不改分布形态、``log_prob`` 雅可比或熵公式；网络学到的部分可反向
        #: 抵消该先验，因此它是先验而不是硬约束。
        self.steering_gain = float(steering_gain)
        self.avoidance_prior = None
        if avoidance_prior is not None:
            if not self.action_bounded:
                raise ValueError("avoidance prior requires bounded action distribution")
            from .policy_prior import ClearanceTangentPrior
            self.avoidance_prior = ClearanceTangentPrior(
                avoidance_prior, layout, self.initial_forward_speed, self.steering_gain)
        self._goal_dir_y_idx: int | None = None
        if self.steering_gain != 0.0:
            if layout is None:
                raise ValueError("steering_gain 非零时需要 layout 才能定位 goal_dir_y_body")
            if cfg.action_dim < 3:
                raise ValueError(f"steering_gain 需要 3 维动作，得到 {cfg.action_dim}")
            _sl = layout.slice_of("goal_dir_y_body")
            if _sl.stop - _sl.start != 1:
                raise ValueError("goal_dir_y_body 必须为 1 维")
            self._goal_dir_y_idx = int(_sl.start)
        if self.initial_forward_speed > 0.0 and self.action_bounded:
            # a = tanh(loc)：令零化 body 输出时的确定性前进速度等于参考速度。
            # 只改 bias 的一个分量；不改分布形态、log_prob 雅可比或熵公式。
            v = min(max(self.initial_forward_speed, 0.0), 1.0 - 1e-6)
            with torch.no_grad():
                self.head.bias.data[0] = math.atanh(v)

    def clamp_log_std(self) -> None:
        """把 log_std 参数钳回契约范围；训练器在每个梯度步后调用。"""
        with torch.no_grad():
            self.log_std.clamp_(self.log_std_min, LOG_STD_MAX)

    def forward(self, obs: torch.Tensor) -> Independent:
        if obs.shape[-1] != self.cfg.obs_dim:
            raise ValueError(f"obs 最后一维应为 {self.cfg.obs_dim}，得到 {tuple(obs.shape)}")
        mean = self.head(self.body(obs))
        if self.steering_gain != 0.0:
            # 只改均值第三分量（wz）；原地写 clone 以保持自动求导图完整。
            mean = mean.clone()
            mean[:, 2] = mean[:, 2] + self.steering_gain * obs[:, self._goal_dir_y_idx]
        # forward 侧防御性钳位（与 clamp_log_std 双保险；边界处梯度为 0）
        if self.avoidance_prior is not None:
            mean = mean + self.avoidance_prior(obs)
        std = self.log_std.clamp(self.log_std_min, LOG_STD_MAX).exp().expand_as(mean)
        if self.action_bounded:
            return SquashedNormal(mean, std)
        return Independent(Normal(mean, std), 1)

    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (动作, log_prob, 分布均值)。动作是**未裁剪**的归一化动作。"""
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.mean


class ValueNetwork(nn.Module):
    """obs -> 标量价值。V_c 与 V_h 各建一个独立实例，不共享参数。"""

    def __init__(self, cfg: NetworkConfig, layout: ObservationLayout | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        self.body = _mlp(cfg.obs_dim, cfg.hidden, cfg.activation, cfg.hidden_gain)
        self.head = _head(cfg.hidden[-1], 1, cfg.value_head_gain)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.cfg.obs_dim:
            raise ValueError(f"obs 最后一维应为 {self.cfg.obs_dim}，得到 {tuple(obs.shape)}")
        return self.head(self.body(obs)).squeeze(-1)


class BudgetConditionedNetworks(nn.Module):
    """策略 + 两个 critic 的容器；三个子网参数互不共享。"""

    ROLES = ("policy", "critic_cost", "critic_reach")

    def __init__(self, cfg: NetworkConfig, layout: ObservationLayout | None = None,
                 action_bounded: bool = False,
                 initial_forward_speed: float = 0.0,
                 log_std_min: float | None = None,
                 log_std_init: float | None = None,
                 steering_gain: float = 0.0,
                 avoidance_prior: dict | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        self.policy = PolicyNetwork(cfg, layout, action_bounded=action_bounded,
                                    initial_forward_speed=initial_forward_speed,
                                    log_std_min=log_std_min,
                                    log_std_init=log_std_init,
                                    steering_gain=steering_gain,
                                    avoidance_prior=avoidance_prior)
        self.critic_cost = ValueNetwork(cfg, layout)
        self.critic_reach = ValueNetwork(cfg, layout)

    # ---- 参数分组：优化器只允许拿到这些 ----
    def parameters_of(self, role: str) -> list[nn.Parameter]:
        if role not in self.ROLES:
            raise KeyError(f"未知角色 {role!r}；可选 {self.ROLES}")
        return list(getattr(self, role).parameters())

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {role: self.parameters_of(role) for role in self.ROLES}

    def n_parameters(self) -> dict[str, int]:
        return {role: sum(p.numel() for p in self.parameters_of(role)) for role in self.ROLES}

    # ---- 前向便捷入口 ----
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        return self.policy.act(obs, deterministic=deterministic)

    def values(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (V_c, V_h)。"""
        return self.critic_cost(obs), self.critic_reach(obs)

    def units(self) -> dict[str, dict[str, str]]:
        return dict(UNIT_TABLE)


# ---------------------------------------------------------------- 映射
def normalized_to_command(action: torch.Tensor,
                          scale: Sequence[float] = (1.0, 0.4, 1.0)) -> torch.Tensor:
    """cmd-v1 的仿射映射：``clip(a,-1,1) * scale``。

    刻意不是 tanh：契约要求目标域精确可达（低速段必须能精确下发 0.2–0.31 m/s）。
    这里只做映射；记录 requested/applied、单次裁剪与非法值置零属于 cmd-v1 的执行侧。
    """
    if action.shape[-1] != len(scale):
        raise ValueError(f"动作最后一维 {action.shape[-1]} ≠ scale 长度 {len(scale)}")
    scale_t = torch.as_tensor(scale, dtype=action.dtype, device=action.device)
    return torch.clamp(action, -1.0, 1.0) * scale_t


def build(nav_contract: dict[str, Any] | None = None,
          cfg: NetworkConfig | None = None,
          action_bounded: bool = False,
          initial_forward_speed: float = 0.0,
          log_std_min: float | None = None,
          log_std_init: float | None = None,
          steering_gain: float = 0.0,
          avoidance_prior: dict | None = None) -> BudgetConditionedNetworks:
    """按契约构造三个网络，并在构造时校验配置。"""
    if nav_contract is None:
        nav_contract, _ = load_contracts()
    cfg = cfg or NetworkConfig(obs_dim=int(nav_contract["observation_spec"]["total_dim"]))
    report = cfg.validate_against(nav_contract)
    if not report["ok"]:
        raise ValueError("网络配置与契约不符：" + "; ".join(report["problems"]))
    layout = ObservationLayout.from_contract(nav_contract)
    return BudgetConditionedNetworks(cfg, layout, action_bounded=action_bounded,
                                     initial_forward_speed=initial_forward_speed,
                                     log_std_min=log_std_min,
                                     log_std_init=log_std_init,
                                     steering_gain=steering_gain,
                                     avoidance_prior=avoidance_prior)
