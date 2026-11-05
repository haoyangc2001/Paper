"""45 维观测适配器（HIRA-013）。

源 ``go2_env.compute_observations`` 逐字语义：

```
obs = cat( base_ang_vel * 0.25,
           projected_gravity,
           commands[:, :3] * [2, 2, 0.25],
           (q - q_default) * 1.0,
           qdot * 0.05,
           prev_clipped_action )
# 然后按 noise_scale_vec 给 9:21、21:33 加均匀噪声，最后 clip 到 ±100
```

三个容易做错的地方，这里都用结构固定住：

1. **关节顺序**：源顺序（逐腿）与 Lab 的 DOF 顺序（按关节类型分组）不同，
   靠 :class:`JointOrder` 按名字建双射，绝不按位置对齐。
2. **上一动作的语义**：存的是**裁剪后、未乘 action_scale** 的动作
   （源里 ``self.actions`` 就是 ``clip(a, ±100)``；0.25 只在算力矩时乘）。
3. **缩放只做一次**：命令进来时已经是 m/s 与 rad/s，本模块只乘 ``[2,2,0.25]``；
   观测裁剪也在本模块末尾做一次。

本模块只用 torch，不 import isaaclab/isaacsim，因此可以在 CPU 上单测。
"""

from __future__ import annotations

import torch

OBS_DIM = 45
ACTION_DIM = 12

#: 六块顺序与切片，来自 ``go2_env.compute_observations`` 的 cat 顺序。
BLOCK_SLICES: tuple[tuple[str, slice], ...] = (
    ("ang_vel", slice(0, 3)),
    ("projected_gravity", slice(3, 6)),
    ("command", slice(6, 9)),
    ("dof_pos", slice(9, 21)),
    ("dof_vel", slice(21, 33)),
    ("prev_action", slice(33, 45)),
)

ANG_VEL_SCALE = 0.25          # obs_scales.ang_vel
GRAVITY_SCALE = 1.0           # 未缩放
COMMAND_SCALE = (2.0, 2.0, 0.25)  # lin_vel 2.0 / lin_vel 2.0 / ang_vel 0.25
DOF_POS_SCALE = 1.0           # obs_scales.dof_pos
DOF_VEL_SCALE = 0.05          # obs_scales.dof_vel
PREV_ACTION_SCALE = 1.0

CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0

#: 源的噪声只加在这两块上（33:45 永不加噪）。
NOISED_BLOCKS = ("dof_pos", "dof_vel")


class JointOrder:
    """按名字建立的双向置换。

    存两份索引：``source_index_of_target``（把 Lab 顺序的量换成源顺序）与
    ``target_index_of_source``（反向）。两个方向都做一致性断言，避免"只对一半"。
    """

    def __init__(self, source_order: list[str], target_order: list[str]):
        self.source_order = list(source_order)
        self.target_order = list(target_order)
        if len(set(self.source_order)) != len(self.source_order):
            raise ValueError("源关节名有重复")
        if len(set(self.target_order)) != len(self.target_order):
            raise ValueError("目标关节名有重复")
        missing = sorted(set(self.source_order) - set(self.target_order))
        extra = sorted(set(self.target_order) - set(self.source_order))
        if missing or extra:
            raise ValueError(f"关节名无法建立双射：缺少 {missing}，多出 {extra}")
        self.source_index_of_target = [self.source_order.index(name) for name in self.target_order]
        self.target_index_of_source = [self.target_order.index(name) for name in self.source_order]
        # 反向校验：两次置换必须回到原位
        for index, source_index in enumerate(self.target_index_of_source):
            if self.source_index_of_target[source_index] != index:
                raise ValueError("置换与其逆不一致")

    def target_to_source(self, tensor: torch.Tensor) -> torch.Tensor:
        """目标顺序 → 源顺序：``out[k] = in[源第 k 个关节在目标里的位置]``。"""
        return tensor[..., self.target_index_of_source]

    def source_to_target(self, tensor: torch.Tensor) -> torch.Tensor:
        """源顺序 → 目标顺序：``out[i] = in[目标第 i 个关节在源里的位置]``。"""
        return tensor[..., self.source_index_of_target]

    def describe(self) -> dict:
        return {
            "source_order": self.source_order,
            "target_order": self.target_order,
            "source_index_of_target": self.source_index_of_target,
            "target_index_of_source": self.target_index_of_source,
        }


class ObservationAdapter:
    """把 Lab 侧物理量拼成源语义的 45 维观测，并维护逐 env 的上一动作。"""

    def __init__(
        self,
        order: JointOrder,
        default_joint_pos_source: torch.Tensor,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        noise_scale: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        if len(order.source_order) != ACTION_DIM:
            raise ValueError(f"源关节数必须是 {ACTION_DIM}，实际 {len(order.source_order)}")
        self.order = order
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_joint_pos_source = default_joint_pos_source.to(self.device, dtype).reshape(1, ACTION_DIM).clone()
        self._prev_action = torch.zeros((num_envs, ACTION_DIM), device=self.device, dtype=dtype)
        # 噪声向量：与观测同宽，仅为 9:33 非零；None 表示完全关闭（clean 跑法）
        self.noise_scale = None if noise_scale is None else noise_scale.to(self.device, dtype).reshape(1, OBS_DIM)
        self._generator = generator
        self.last_blocks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    @property
    def prev_action(self) -> torch.Tensor:
        """上一策略步的裁剪后动作（源顺序）。"""
        return self._prev_action.clone()

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """逐 env 清零上一动作；不给 env_ids 就全清。

        源的 ``_reset_idx`` 只把对应 env 的 ``actions`` 归零，**不碰其他 env**——
        批量训练里这是必须的，否则一个 env 摔倒会污染别人的观测。
        """
        if env_ids is None:
            self._prev_action.zero_()
            return
        index = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self._prev_action[index] = 0.0

    def apply_action(self, raw_action_source: torch.Tensor) -> torch.Tensor:
        """裁剪并记录动作。返回裁剪后的动作（源顺序），供力矩计算使用。

        注意存的是**裁剪后未缩放**的值：源里 ``self.actions`` 就是这个量，
        ``action_scale`` 只在 ``_compute_torques`` 里乘。
        """
        if raw_action_source.shape[-1] != ACTION_DIM:
            raise ValueError(f"动作维度应为 {ACTION_DIM}，实际 {tuple(raw_action_source.shape)}")
        clipped = torch.clamp(raw_action_source.to(self.device, self.dtype), -CLIP_ACTIONS, CLIP_ACTIONS)
        self._prev_action = clipped.clone()
        return clipped

    # ------------------------------------------------------------------ #
    def build(
        self,
        ang_vel_body_target: torch.Tensor,
        projected_gravity_target: torch.Tensor,
        command: torch.Tensor,
        joint_pos_target: torch.Tensor,
        joint_vel_target: torch.Tensor,
        include_noise: bool = False,
    ) -> dict:
        """拼观测。所有输入都是 Lab 侧量（目标关节顺序），内部换成源顺序。

        返回 ``{"obs", "blocks", "scaled_blocks"}``，其中 ``blocks`` 是原始块，
        ``scaled_blocks`` 是乘过 scale 但未裁剪的块——两者都留档，便于和 oracle 对照。
        """
        ang_vel = self._to_source_rows(ang_vel_body_target, "ang_vel", 3)
        gravity = self._to_source_rows(projected_gravity_target, "projected_gravity", 3)
        cmd = self._to_source_rows(command, "command", 3)
        pos = self.order.target_to_source(self._to_rows(joint_pos_target, "joint_pos", ACTION_DIM))
        vel = self.order.target_to_source(self._to_rows(joint_vel_target, "joint_vel", ACTION_DIM))

        command_scale = torch.tensor(COMMAND_SCALE, device=self.device, dtype=self.dtype)
        scaled = {
            "ang_vel": ang_vel * ANG_VEL_SCALE,
            "projected_gravity": gravity * GRAVITY_SCALE,
            "command": cmd * command_scale,
            "dof_pos": (pos - self.default_joint_pos_source) * DOF_POS_SCALE,
            "dof_vel": vel * DOF_VEL_SCALE,
            "prev_action": self._prev_action * PREV_ACTION_SCALE,
        }
        blocks = {
            "ang_vel": ang_vel,
            "projected_gravity": gravity,
            "command": cmd,
            "dof_pos": pos - self.default_joint_pos_source,
            "dof_vel": vel,
            "prev_action": self._prev_action.clone(),
        }

        pieces = [scaled[name] for name, _ in BLOCK_SLICES]
        obs = torch.cat(pieces, dim=-1)

        if include_noise:
            if self.noise_scale is None:
                raise ValueError("include_noise=True 但没有提供 noise_scale；clean 跑法不应加噪")
            noise = (2.0 * torch.rand(obs.shape, device=self.device, dtype=self.dtype, generator=self._generator) - 1.0)
            obs = obs + noise * self.noise_scale

        obs = torch.clamp(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)
        if obs.shape[-1] != OBS_DIM:
            raise AssertionError(f"观测维度 {obs.shape[-1]} != {OBS_DIM}")

        self.last_blocks = {"raw": blocks, "scaled": scaled, "final": obs.clone()}
        return {"obs": obs, "blocks": blocks, "scaled_blocks": scaled}

    # ------------------------------------------------------------------ #
    def _to_rows(self, tensor: torch.Tensor, name: str, width: int) -> torch.Tensor:
        tensor = tensor.to(self.device, self.dtype)
        if tensor.dim() != 2:
            raise ValueError(f"{name} 应为 (num_envs, {width})，实际 {tuple(tensor.shape)}")
        if tensor.shape[-1] != width:
            raise ValueError(f"{name} 宽度应为 {width}，实际 {tensor.shape[-1]}")
        if tensor.shape[0] != self.num_envs:
            raise ValueError(f"{name} 的 env 数 {tensor.shape[0]} 与适配器的 {self.num_envs} 不一致")
        return tensor

    def _to_source_rows(self, tensor: torch.Tensor, name: str, width: int) -> torch.Tensor:
        """前三个块（角速度/重力/命令）在源里不按关节排列，只校验宽度。"""
        return self._to_rows(tensor, name, width)


def build_noise_vector(dof_pos_scale: float = 0.01, dof_vel_scale: float = 1.5, noise_level: float = 1.0) -> torch.Tensor:
    """按源的 ``noise_scale_vec`` 生成噪声幅度；只覆盖 9:21 与 21:33。

    源里 ``noise_scale_vec`` 对每个块用 ``noise_scales`` 乘 ``noise_level``，
    其余块为 0。33:45（上一动作）永不加噪——这是源的既定行为，不是我们的选择。
    """
    vector = torch.zeros(OBS_DIM)
    vector[9:21] = dof_pos_scale * noise_level
    vector[21:33] = dof_vel_scale * noise_level
    return vector


# --------------------------------------------------------------------------- #
# 姿态与角速度的约定（HIRA-013 第 1 步）
# --------------------------------------------------------------------------- #
#: 本模块统一使用 **wxyz**（Isaac Lab 约定）。源 isaacgym 的四元数是 **xyzw**，
#: 直接喂进来会得到一个"看起来也像旋转"的错误结果，所以这里把顺序写进函数名。
def quat_rotate_inverse_wxyz(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """把世界系向量旋到机体系：``v_b = R(q)^T v_w``（Isaac Lab 的 ``quat_rotate_inverse`` 同式）。"""
    q_w = quat_wxyz[..., 0:1]
    q_vec = quat_wxyz[..., 1:4]
    a = vector * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, vector, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * vector, dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_rotate_wxyz(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """机体系向量旋到世界系：``v_w = R(q) v_b``。"""
    return quat_rotate_inverse_wxyz(torch.cat([quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]], dim=-1), vector)


def projected_gravity_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """源 ``projected_gravity``：把世界重力方向 ``(0,0,-1)`` 旋到机体系。

    源在 ``_compute_observations`` 里就是 ``quat_rotate_inverse(base_quat, gravity_vec)``，
    其中 ``gravity_vec = (0,0,-1)``；本函数与之等价，不需要额外的 z 轴翻转。
    """
    gravity = torch.zeros_like(quat_wxyz[..., :3])
    gravity[..., 2] = -1.0
    return quat_rotate_inverse_wxyz(quat_wxyz, gravity)


def body_angular_velocity_wxyz(quat_wxyz: torch.Tensor, world_ang_vel: torch.Tensor) -> torch.Tensor:
    """世界系角速度 → 机体系角速度。

    源的 ``base_ang_vel`` 是 **机体系**量（isaacgym 直接给 root 的 body-frame 角速度）；
    Lab 侧拿到的是世界系，必须显式转一次。世界 yaw_rate 与机体系 z 分量不是一回事。
    """
    return quat_rotate_inverse_wxyz(quat_wxyz, world_ang_vel)


def quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """isaacgym/ROS 的 xyzw → Isaac Lab 的 wxyz。排错时最省事的定位手段。"""
    return torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)
