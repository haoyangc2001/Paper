"""把冻结的近障地图作为运动学刚体放进 HiRA 的 Isaac Lab 场景（HIRA-034）。

与 v2 的 `CaMRA/calibration/sim/obstacles_sim.py` 同一套语义，差别只在挂载方式：
v2 走 `gym.make` 的 `env_cfg.scene`，HiRA 走 `InteractiveSceneCfg` 的**类属性**。

设计约束（两条都是踩过的坑，别改）：

* 一个进程内 N 个 env 共享同一份场景几何，所以障碍必须**按 env 克隆**，
  每批用 :func:`set_slots` 逐 env 改写位姿；
* 槽位数固定为地图里出现过的最大障碍数（7），不用的槽位停到 :data:`PARK_XY`
  （远在场地外，机器人跑不到，也不参与任何判定）；
* 障碍是 **kinematic** 刚体：PhysX 仍把它当实体碰撞体拦住机器人，但可以逐批改写位姿
  而不重建场景；
* **绝不给障碍加自定义材质**。每挂一个 `PreviewSurfaceCfg` 会触发一次材质/着色器编译，
  本机实测 ~85 s/槽位，把场景创建从 4 s 拖到 >10 min（v2 CAL-028 的真凶）。
  默认材质足够，本轮无视觉需求；
* 场地边界墙**不建模**：每例 arena 尺寸都不同，逐例缩放静态体不划算。越界一律按
  "用冻结足迹做几何检查"判定并如实上报（与 v2 同一处理，协议也这么要求）。

地图坐标是**场地坐标系**，写进仿真时要加上该 env 的原点偏移。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

#: 障碍尺寸（与地图生成规则一致：0.3×0.3 的柱，高度取 0.6 保证能挡住机身）
OBS_SIZE: float = 0.3
OBS_HEIGHT: float = 0.6
#: 未使用槽位的停放点（场地最大 16×15 m，y=30 已在场外十余米）
PARK_XY: tuple[float, float] = (0.0, 30.0)
#: 槽位数 = 地图中出现过的最大障碍数（near_obstacle_maps_v1 的 n_obstacles_hist 最大值）
N_SLOTS: int = 7


def obstacle_slot_cfg(index: int):
    """第 ``index`` 个槽位的 ``RigidObjectCfg``（kinematic、无重力、默认材质）。

    延迟导入 isaaclab，保证 CPU 侧单测不需要仿真依赖。
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obs_{index}",
        spawn=sim_utils.CuboidCfg(
            size=(OBS_SIZE, OBS_SIZE, OBS_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PARK_XY[0], PARK_XY[1], OBS_HEIGHT / 2.0)),
    )


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """绕 z 的偏航 → Isaac Lab 的四元数约定 ``(w, x, y, z)``。"""
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def slot_pose(origin_xy: Sequence[float], x: float, y: float, yaw: float,
              z_center: float = OBS_HEIGHT / 2.0) -> list[float]:
    """场地坐标 → 世界位姿 ``[x, y, z, qw, qx, qy, qz]``。"""
    qw, qx, qy, qz = yaw_to_quat(yaw)
    return [float(origin_xy[0]) + float(x), float(origin_xy[1]) + float(y),
            float(z_center), qw, qx, qy, qz]


def build_batch_poses(cases: Sequence[dict | None], origins: torch.Tensor,
                      n_env: int, n_slots: int = N_SLOTS) -> torch.Tensor:
    """组装一批的槽位位姿 ``(n_env, n_slots, 7)``。

    ``cases`` 长度 ≤ n_env，顺序与 env 索引一致；``None`` 表示该 env 本批不用障碍
    （例如补位用的 scratch 槽位），整行停到 :data:`PARK_XY`。
    """
    dev = origins.device
    poses = torch.zeros(n_env, n_slots, 7, dtype=torch.float32, device=dev)
    for i in range(n_env):
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        row: list[list[float]] = []
        case = cases[i] if i < len(cases) else None
        if case is not None:
            for obs in case.get("obstacles", []):
                row.append(slot_pose((ox, oy), obs["x"], obs["y"], float(obs.get("yaw", 0.0))))
        while len(row) < n_slots:
            row.append(slot_pose((ox, oy), PARK_XY[0], PARK_XY[1], 0.0))
        poses[i] = torch.tensor(row[:n_slots], dtype=torch.float32, device=dev)
    return poses


def set_slots(scene, names: Sequence[str], poses: torch.Tensor) -> None:
    """把位姿写给每个槽位（``poses`` 形状 ``(n_env, len(names), 7)``）。"""
    for k, name in enumerate(names):
        scene[name].write_root_pose_to_sim(poses[:, k, :])


def read_back_slots(scene, names: Sequence[str]) -> torch.Tensor:
    """读回每个槽位当前位姿，用于"障碍确实注入"的逐 env 证据。"""
    rows = [scene[name].data.root_state_w[:, :7].clone() for name in names]
    return torch.stack(rows, dim=1)
