"""五类物理执行条件的仿真侧施加（CAL-013）。

设计：事件管理器里的隐藏随机化全部显式移除或退化为确定值，物理旋钮由本模块
**逐 env 精确写入**（每个 env 对应一条独立 trial，摩擦/质量/初态/推扰各不相同）：

* 摩擦  ``root_physx_view.get_material_properties() / set_material_properties()``，
  逐 env × 逐 shape 写 (static, dynamic, restitution)；
* 质量  ``root_physx_view.get_masses() / set_masses()``，base 刚体加增量，
  并按 IsaacLab 官方 randomize_rigid_body_mass 的方式以质量比缩放默认惯量；
* 初态  ``robot.write_root_pose_to_sim()`` 逐 env 写 trial 的平面位姿
  （z 取默认根高，四元数由 yaw 生成）；
* 推扰  到点后 ``robot.write_root_velocity_to_sim(当前速度 + [vx, vy, 0, 0, 0, 0])``，
  与官方 push_by_setting_velocity 同机制，但时刻与幅度来自 trial 的采样表；
* 观测噪声  cfg 级 ``observations.policy.enable_corruption``（combined 保持训练
  现有值，其余禁用）。

这样"只有指定旋钮变化"由两部分证明：
  1) cfg 级 diff（本模块 :func:`configure_env_for_profile` 的返回记录）；
  2) PhysX 读回校验（:func:`apply_friction` / :func:`apply_mass` 返回 read-back）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from camra_calibration.profiles import AppliedProfile, TRAINING_OBS_NOISE


def _env_ids_tensor(env_ids: Sequence[int], device: str = "cpu") -> torch.Tensor:
    """PhysX 视图 set_*（材料/质量/惯量）收 CPU int64 索引；write_root_* 收 GPU 索引。"""
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(dtype=torch.int64, device=device)
    return torch.as_tensor(list(env_ids), dtype=torch.int64, device=device)


@dataclass
class ProfileConfigReport:
    """cfg 级改动的完整记录（进入逐 trial 的追溯文件）。"""

    profile_id: str
    obs_corruption_enabled: bool
    obs_noise: dict[str, list[float]] | None
    removed_event_terms: list[str] = field(default_factory=list)
    reset_base_pose_range: dict[str, list[float]] = field(default_factory=dict)
    reset_base_velocity_range: dict[str, list[float]] = field(default_factory=dict)
    reset_joint_scale_range: list[float] = field(default_factory=list)
    reset_joint_velocity_range: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "obs_corruption_enabled": self.obs_corruption_enabled,
            "obs_noise": self.obs_noise,
            "removed_event_terms": self.removed_event_terms,
            "reset_base_pose_range": self.reset_base_pose_range,
            "reset_base_velocity_range": self.reset_base_velocity_range,
            "reset_joint_scale_range": self.reset_joint_scale_range,
            "reset_joint_velocity_range": self.reset_joint_velocity_range,
        }


def configure_env_for_profile(env_cfg, applied: AppliedProfile) -> ProfileConfigReport:
    """把 env cfg 改成给定 profile 的确定性形态（在 gym.make 之前调用）。

    事件管理器里所有隐藏随机化一律移除或退化：
      * ``physics_material`` / ``add_base_mass`` 移除（逐 env 直写替代）；
      * ``base_external_force_torque`` 移除（训练配置本就是 0，移除以保 diff 干净）；
      * ``push_robot`` 移除（按 trial 时刻表推，替代 interval 随机调度）；
      * ``reset_base`` 位姿区间退化为 0（平面位姿由 trial 指定后逐 env 直写）、
        速度区间保持 0（训练配置已是 0，显式写明）；
      * ``reset_robot_joints`` 关节速度区间退化为 0（禁用隐藏随机化），
        位置缩放已是 (1.0, 1.0) no-op。
    """
    noise_enabled = applied.obs_noise is not None
    env_cfg.observations.policy.enable_corruption = noise_enabled

    removed: list[str] = []
    for name in ("physics_material", "add_base_mass", "base_external_force_torque", "push_robot"):
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
            removed.append(name)

    rb = env_cfg.events.reset_base.params
    for key in ("x", "y", "yaw"):
        rb["pose_range"][key] = (0.0, 0.0)
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        rb["velocity_range"][key] = (0.0, 0.0)

    rj = env_cfg.events.reset_robot_joints.params
    rj["velocity_range"] = (0.0, 0.0)

    noise_rec = None
    if noise_enabled:
        noise_rec = {}
        group = env_cfg.observations.policy
        for term_name, rng in TRAINING_OBS_NOISE.items():
            term = getattr(group, term_name)
            n = getattr(term, "noise", None)
            noise_rec[term_name] = [float(n.n_min), float(n.n_max)] if n is not None else None
            if noise_rec[term_name] != [float(rng[0]), float(rng[1])]:
                raise RuntimeError(f"观测噪声 {term_name} 与训练配置不一致: {noise_rec[term_name]} vs {list(rng)}")

    return ProfileConfigReport(
        profile_id=applied.profile_id,
        obs_corruption_enabled=noise_enabled,
        obs_noise=noise_rec,
        removed_event_terms=removed,
        reset_base_pose_range={k: list(v) for k, v in rb["pose_range"].items()},
        reset_base_velocity_range={k: list(v) for k, v in rb["velocity_range"].items()},
        reset_joint_scale_range=list(rj["position_range"]),
        reset_joint_velocity_range=list(rj["velocity_range"]),
    )


def _yaw_quat_wxyz(yaw: float):
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def apply_friction(robot, env_ids: Sequence[int], friction: float,
                   restitution: float = 0.0) -> dict:
    """逐 env 写摩擦（static == dynamic == friction），并读回校验。"""
    if not env_ids:
        return {"checked": 0}
    env_ids = _env_ids_tensor(env_ids)
    mats = robot.root_physx_view.get_material_properties()  # (num_envs, max_shapes, 3)
    max_shapes = mats.shape[1]
    mats[env_ids] = mats[env_ids] * 0.0
    mats[env_ids, :, 0] = friction
    mats[env_ids, :, 1] = friction
    mats[env_ids, :, 2] = restitution
    robot.root_physx_view.set_material_properties(mats, env_ids)
    # 读回
    back = robot.root_physx_view.get_material_properties()
    sel = back[env_ids]
    max_abs = float((sel - mats[env_ids]).abs().max()) if sel.numel() else 0.0
    static_match = bool((sel[:, :, 0] == friction).all())
    dynamic_match = bool((sel[:, :, 1] == friction).all())
    rest_match = bool((sel[:, :, 2] == restitution).all())
    return {
        "checked_envs": len(env_ids), "shapes_per_env": int(max_shapes),
        "readback_max_abs_diff": max_abs,
        "static_match": static_match, "dynamic_match": dynamic_match,
        "restitution_match": rest_match, "ok": static_match and dynamic_match and rest_match,
    }


def apply_mass(robot, env_ids: Sequence[int], delta_kg: float,
               body_name: str = "base") -> dict:
    """逐 env 给 body 加质量增量，并按官方实现以质量比缩放默认惯量；读回校验。"""
    if not env_ids:
        return {"checked": 0}
    env_ids = _env_ids_tensor(env_ids)
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"body {body_name!r} 解析出 {len(body_ids)} 个: {body_names}")
    b = body_ids[0]
    masses = robot.root_physx_view.get_masses()          # (num_envs, num_bodies)
    default_mass = robot.data.default_mass.clone()
    masses[env_ids, b] = default_mass[env_ids, b] + delta_kg
    robot.root_physx_view.set_masses(masses, env_ids)

    inertias = robot.root_physx_view.get_inertias()       # (num_envs, num_bodies, 9)
    ratios = masses[env_ids, b] / default_mass[env_ids, b]
    inertias[env_ids, b] = robot.data.default_inertia[env_ids, b] * ratios[:, None]
    robot.root_physx_view.set_inertias(inertias, env_ids)

    back_m = robot.root_physx_view.get_masses()
    back_i = robot.root_physx_view.get_inertias()
    expect_m = default_mass[env_ids, b] + delta_kg
    m_max = float((back_m[env_ids, b] - expect_m).abs().max())
    # 质量 ~7 kg 时 float32 的 1 ULP ≈ 5e-7；用绝对容差而非逐位相等
    m_ok = bool(m_max <= 1e-5)
    i_expect = robot.data.default_inertia[env_ids, b] * ratios[:, None]
    # PhysX 在质量变化后会重算惯量，float32 舍入 ~1e-9；用绝对容差而非逐位相等
    i_max = float((back_i[env_ids, b] - i_expect).abs().max())
    i_ok = bool(i_max <= 1e-6)
    return {
        "checked_envs": len(env_ids), "body": body_name, "body_id": int(b),
        "default_mass_kg": float(default_mass[env_ids[0], b]),
        "delta_kg": delta_kg,
        "readback_mass_max_abs_diff": m_max, "mass_match": m_ok,
        "readback_inertia_max_abs_diff": i_max, "inertia_match": i_ok,
        "ok": m_ok and i_ok,
    }


def apply_initial_pose(robot, env_ids: Sequence[int], px: Sequence[float],
                       py: Sequence[float], yaw: Sequence[float]) -> dict:
    """逐 env 写 trial 的平面初始位姿（z 用默认根高）。"""
    import torch
    if not env_ids:
        return {"checked": 0}
    env_ids = _env_ids_tensor(env_ids)
    dev = robot.device
    default_root = robot.data.default_root_state[env_ids].clone()  # (n, 13): pos3 quat4 vel6
    pos = default_root[:, :3].clone()
    pos[:, 0] = torch.tensor(px, device=dev, dtype=pos.dtype)
    pos[:, 1] = torch.tensor(py, device=dev, dtype=pos.dtype)
    q = torch.tensor([_yaw_quat_wxyz(a) for a in yaw], device=dev, dtype=pos.dtype)
    root_pose = torch.cat([pos, q], dim=-1)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids.to(device=robot.device))
    back = robot.data.root_pos_w[env_ids].clone()  # may be stale until update; read via physx view
    view = robot.root_physx_view.get_root_transforms()  # (num_envs, 7) pos3 quat(qx,qy,qz,qw)
    pos_err = float((view[env_ids, :2] - root_pose[:, :2]).abs().max())
    yaw_ok = True
    for i, e in enumerate(env_ids):
        qx, qy, qz, qw = float(view[e, 3]), float(view[e, 4]), float(view[e, 5]), float(view[e, 6])
        yaw_back = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        d = abs((yaw_back - yaw[i] + math.pi) % (2 * math.pi) - math.pi)
        yaw_ok = yaw_ok and d < 1e-5
    _ = back
    return {"checked_envs": len(env_ids), "pos_max_abs_diff": pos_err,
            "yaw_match": yaw_ok, "ok": pos_err < 1e-5 and yaw_ok}


def apply_push(robot, env_id: int, vx: float, vy: float) -> dict:
    """按 push_by_setting_velocity 同机制施加一次速度冲击（只改 x/y）。"""
    import torch
    env_id = int(env_id)
    vel_w = robot.data.root_vel_w[env_id].clone()
    vel_w[0] += vx
    vel_w[1] += vy
    robot.write_root_velocity_to_sim(
        vel_w[None, :], env_ids=torch.tensor([env_id], dtype=torch.int64, device=robot.device))
    back = robot.root_physx_view.get_root_velocities()[env_id].clone()
    dvx = float(back[0] - vel_w[0]); dvy = float(back[1] - vel_w[1])
    return {"env_id": env_id, "applied": [vx, vy],
            "readback_delta": [dvx, dvy], "ok": abs(dvx) < 1e-4 and abs(dvy) < 1e-4}


class PushExecutor:
    """按 trial 时刻表逐 env 施加推扰。

    时刻以"测量窗口秒"计；由采集循环每个策略步调用 :meth:`tick`，
    传入当前窗口时间与步长，落在 ``(t, t+dt]`` 的事件被触发一次。
    实际触发时刻与幅度记录进 ``self.applied``（追溯记录）。
    """

    def __init__(self, per_env: dict[int, tuple[tuple[float, float, float], ...]], dt: float):
        self.dt = float(dt)
        self.pending: dict[int, list[tuple[float, float, float]]] = {
            e: list(ev) for e, ev in per_env.items()}
        self.applied: list[dict] = []

    def due(self, env_id: int, window_t: float) -> tuple[float, float, float] | None:
        pend = self.pending.get(env_id)
        if not pend:
            return None
        if pend[0][0] <= window_t + 0.5 * self.dt:  # 事件时刻落在当前步中点之前
            return pend.pop(0)
        return None

    def tick(self, robot, env_id: int, window_t: float) -> dict | None:
        ev = self.due(env_id, window_t)
        if ev is None:
            return None
        _t, vx, vy = ev
        rec = apply_push(robot, env_id, vx, vy)
        rec.update({"scheduled_t": ev[0], "triggered_window_t": window_t, "env_id": env_id})
        self.applied.append(rec)
        return rec
