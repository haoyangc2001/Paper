"""把冻结地图里的障碍物作为运动学刚体放进场景（CAL-028）。

设计约束：

* 一个进程内 32 个 env 共享同一份场景几何，所以障碍物必须**按 env 克隆**，
  每批次用 :func:`set_slots` 逐 env 改写位姿；
* 槽位数量固定为地图里出现过的最大障碍数（7），不用的槽位停到 ``PARK_XY``
  （远在场地外，不影响任何 env）；
* 障碍物用 **kinematic** 刚体：既能被 PhysX 当作实体碰撞体拦住机器人，
  又能逐批次改写位姿而无需重建场景；
* 场地边界墙**不建模**（每例 arena 尺寸不同，逐例缩放静态体不划算）：
  改为离线用冻结足迹做"是否越过场地边界"的几何检查并如实上报。

地图坐标系是场地坐标系（arena frame），仿真里要加上该 env 的原点偏移。
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

#: 障碍物边长/高度（与地图一致：0.3×0.3 的柱子）
OBS_SIZE: float = 0.3
OBS_HEIGHT: float = 0.6
#: 未使用槽位的停放位置（场地最大 16×15 m，停到 y=30 已在场外十余米，机器人跑不到）。
#: 注：曾怀疑停放太远会撑爆 broadphase 包围盒——实测与停放距离无关（见 §30），
#: 真正的元凶是每个槽位的自定义材质，已移除。
PARK_XY: tuple[float, float] = (0.0, 30.0)
#: 槽位数量 = 地图中最大障碍数
N_SLOTS: int = 7


def add_obstacle_slots(env_cfg, n_slots: int = N_SLOTS) -> list[str]:
    """往 ``env_cfg.scene`` 里加 ``obs_0..obs_{n-1}`` 槽位，返回名字列表。

    必须在 ``gym.make`` 之前调用（场景 prim 在环境创建时生成）。
    """
    from isaaclab.assets import RigidObjectCfg   # 延迟导入：CPU 单测不需要
    import isaaclab.sim as sim_utils             # 本版本 sim_utils 是 sim 包的别名

    names: list[str] = []
    for i in range(n_slots):
        name = f"obs_{i}"
        cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(OBS_SIZE, OBS_SIZE, OBS_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # 不指定自定义材质：每个 PreviewSurface 会触发一次材质/着色器编译，
                # 在本机（渲染器告警、libXt/libGLU 缺失）实测每槽位约 85 s，直接把
                # gym.make 从 4 s 拖到 >10 min。默认材质足够，视觉无需求。
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(PARK_XY[0], PARK_XY[1], OBS_HEIGHT / 2.0)),
        )
        setattr(env_cfg.scene, name, cfg)
        names.append(name)
    return names


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """绕 z 的偏航 → IsaacLab 的四元数约定 (w, x, y, z)。"""
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def slot_pose(origin_xy: Sequence[float], x: float, y: float, yaw: float,
              z_center: float = OBS_HEIGHT / 2.0) -> list[float]:
    """把场地坐标下的障碍位姿换算成仿真世界位姿 ``[x,y,z,qw,qx,qy,qz]``。"""
    qw, qx, qy, qz = yaw_to_quat(yaw)
    return [float(origin_xy[0]) + float(x), float(origin_xy[1]) + float(y),
            float(z_center), qw, qx, qy, qz]


def build_batch_poses(cases: Sequence[dict], origins: torch.Tensor,
                      n_env: int, n_slots: int = N_SLOTS) -> torch.Tensor:
    """组装一个批次的槽位位姿张量 ``(n_env, n_slots, 7)``。

    ``cases`` 是长度 ≤ n_env 的地图条目（顺序与 env 索引一致）；多出来的 env
    与多出来的槽位一起停到 ``PARK_XY``。
    """
    dev = origins.device
    poses = torch.zeros(n_env, n_slots, 7, dtype=torch.float32, device=dev)
    for i in range(n_env):
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        row = []
        if i < len(cases):
            for obs in cases[i].get("obstacles", []):
                row.append(slot_pose((ox, oy), obs["x"], obs["y"], float(obs.get("yaw", 0.0))))
        while len(row) < n_slots:
            row.append(slot_pose((ox, oy), PARK_XY[0], PARK_XY[1], 0.0))
        poses[i] = torch.tensor(row[:n_slots], dtype=torch.float32, device=dev)
    return poses


def set_slots(env, names: Sequence[str], poses: torch.Tensor) -> None:
    """把位姿写给每个槽位的每个 env（``poses`` 形状 ``(n_env, len(names), 7)``）。"""
    scene = env.unwrapped.scene
    for k, name in enumerate(names):
        scene[name].write_root_pose_to_sim(poses[:, k, :])
