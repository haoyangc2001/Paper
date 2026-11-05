"""动作映射与每物理子步 PD 力矩（HIRA-014）。

源 legged_gym 的控制链（``_compute_torques`` + ``step``）：

```
每 4 个物理子步（decimation=4）:
    actions = clip(actor_out, -100, 100)            # 只在策略步裁剪一次
    q_target = q_default + 0.25 * actions           # action_scale = 0.25
    for 4 个子步:
        τ = kp * (q_target - q) - kd * qdot         # q/qdot 是**该子步**的新鲜值
        τ = clip(τ, ±torque_limit)                  # 每关节一次
        write(τ)
```

三个"只能发生一次"的点：

1. **动作裁剪**：只在策略步入口裁一次。第二次裁剪在数值上无害，但会让
   "上一动作"与"实际执行动作"在 ±100 边界上不一致，所以本模块直接拒绝超界输入。
2. **PD**：`q_target` 在 4 个子步里保持不变，但力矩必须用**每个子步的新鲜 q/qdot** 重算。
   这是"显式 PD"与"隐式 PD"最本质的差别，也是本任务要证明的东西。
3. **力矩限幅**：`τ_clipped = clip(τ_PD, ±effort)`，只此一处；写进仿真的就是这个值。

关节顺序仍按名字双射（HIRA-012 实测 Lab 顺序与 URDF 不同），
所以 PD 在**源顺序**里算，写入前再映射回 Lab 顺序。
"""

from __future__ import annotations

import torch

from .observation import JointOrder

ACTION_DIM = 12
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0
STIFFNESS = 20.0   # GO2RoughCfg.control.stiffness（不是 v2 的 25）
DAMPING = 0.5      # GO2RoughCfg.control.damping
DECIMATION = 4


class ActionTorqueAdapter:
    """把策略动作变成关节目标，并在每个物理子步算出要写入的力矩。"""

    def __init__(
        self,
        order: JointOrder,
        default_joint_pos_source: torch.Tensor,
        effort_limit_source: torch.Tensor,
        stiffness: float = STIFFNESS,
        damping: float = DAMPING,
        action_scale: float = ACTION_SCALE,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.order = order
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_joint_pos_source = default_joint_pos_source.to(self.device, dtype).reshape(1, ACTION_DIM).clone()
        self.effort_limit_source = effort_limit_source.to(self.device, dtype).reshape(1, ACTION_DIM).clone()
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.action_scale = float(action_scale)

        self._q_target_source: torch.Tensor | None = None
        self._clipped_action_source: torch.Tensor | None = None
        self.substep_index = 0
        self.n_saturated = 0
        self.last_trace: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def set_policy_action(self, action_source: torch.Tensor) -> torch.Tensor:
        """策略步入口：接收**已裁剪**的动作（源顺序），算 q_target。

        传进来的动作必须已经裁到 ±100。超界说明调用方漏裁，或者想在这里再裁一次；
        两种情况都拒绝——重复裁剪会让"观测里的上一动作"和"实际执行动作"分叉。
        """
        action_source = action_source.to(self.device, self.dtype)
        if action_source.shape[-1] != ACTION_DIM:
            raise ValueError(f"动作维度应为 {ACTION_DIM}，实际 {tuple(action_source.shape)}")
        if bool((action_source.abs() > CLIP_ACTIONS + 1e-6).any()):
            raise ValueError(
                f"动作超出 ±{CLIP_ACTIONS}，说明上游没有裁剪；本模块不重复裁剪（HIRA-014 约束）"
            )
        q_target = self.default_joint_pos_source + self.action_scale * action_source
        self._clipped_action_source = action_source.clone()
        self._q_target_source = q_target
        self.substep_index = 0
        self.n_saturated = 0
        return q_target

    # ------------------------------------------------------------------ #
    def compute_substep(self, joint_pos_lab: torch.Tensor, joint_vel_lab: torch.Tensor) -> dict:
        """一个物理子步：用新鲜 q/qdot 重算 PD，裁剪一次，返回要写入的力矩（Lab 顺序）。"""
        if self._q_target_source is None:
            raise RuntimeError("还没调用 set_policy_action，没有关节目标可跟踪")

        pos_source = self.order.target_to_source(joint_pos_lab.to(self.device, self.dtype))
        vel_source = self.order.target_to_source(joint_vel_lab.to(self.device, self.dtype))

        tau_pd = self.stiffness * (self._q_target_source - pos_source) - self.damping * vel_source
        tau_clipped = torch.clamp(tau_pd, -self.effort_limit_source, self.effort_limit_source)
        saturated = (tau_pd.abs() > self.effort_limit_source).sum().item()

        tau_lab = self.order.source_to_target(tau_clipped)
        q_target_lab = self.order.source_to_target(self._q_target_source)

        self.n_saturated += int(saturated)
        self.substep_index += 1
        self.last_trace = {
            "substep_index": self.substep_index,
            "clipped_action_source": self._clipped_action_source.clone(),
            "q_target_source": self._q_target_source.clone(),
            "q_target_lab": q_target_lab.clone(),
            "joint_pos_source": pos_source.clone(),
            "joint_vel_source": vel_source.clone(),
            "tau_pd_source": tau_pd.clone(),
            "tau_clipped_source": tau_clipped.clone(),
            "effort_written_lab": tau_lab.clone(),
            "n_saturated_joints": int(saturated),
        }
        return self.last_trace

    def reset(self) -> None:
        self._q_target_source = None
        self._clipped_action_source = None
        self.substep_index = 0
        self.n_saturated = 0
        self.last_trace = {}


def source_pd_torque(
    q_target_source: torch.Tensor,
    joint_pos_source: torch.Tensor,
    joint_vel_source: torch.Tensor,
    effort_limit_source: torch.Tensor,
    stiffness: float = STIFFNESS,
    damping: float = DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """独立公式：返回 ``(τ_PD, τ_clipped)``。测试的 oracle 用它，适配器自身不调用。"""
    tau = stiffness * (q_target_source - joint_pos_source) - damping * joint_vel_source
    return tau, torch.clamp(tau, -effort_limit_source, effort_limit_source)


def source_q_target(
    action_source: torch.Tensor,
    default_joint_pos_source: torch.Tensor,
    action_scale: float = ACTION_SCALE,
) -> torch.Tensor:
    """独立公式：``q_target = q_default + 0.25 * a``。"""
    return default_joint_pos_source + action_scale * action_source
