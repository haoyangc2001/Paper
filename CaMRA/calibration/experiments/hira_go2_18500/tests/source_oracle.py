"""独立来源 oracle（HIRA-015）。

**故意不 import ``hira_go2`` 的任何模块**，也不读 URDF：只吃 fixture 里的数字，
按源 `go2_env.compute_observations` / `legged_robot._compute_torques` 的公式重算一遍。

两处刻意的实现差异，防止"两边同一个 bug 抵消"：

| 项 | 适配器（被测） | 本 oracle |
|---|---|---|
| 四元数旋转 | 叉乘展开式 ``v - 2w(q×v) + 2q×(q×v)`` | 先构造旋转矩阵，再做 ``Rᵀv`` |
| 关节顺序 | 索引置换数组 | 一律按**名字**取字典 |
| 数据类型 | float32/float64 张量 | Python float（双精度） |
| 裁剪 | ``torch.clamp`` | 逐元素 ``min/max`` |
"""

from __future__ import annotations

import math

OBS_DIM = 45
ACTION_DIM = 12
CLIP_OBS = 100.0
CLIP_ACTION = 100.0
ACTION_SCALE = 0.25
STIFFNESS = 20.0
DAMPING = 0.5

BLOCKS = {
    "ang_vel": (0, 3),
    "projected_gravity": (3, 6),
    "command": (6, 9),
    "dof_pos": (9, 21),
    "dof_vel": (21, 33),
    "prev_action": (33, 45),
}


def quat_to_matrix(quat_wxyz: list[float]) -> list[list[float]]:
    """wxyz → 旋转矩阵（行主序）。"""
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"四元数未归一化：norm={norm}")
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rotate_inverse(quat_wxyz: list[float], vector: list[float]) -> list[float]:
    """Rᵀ·v：世界系 → 机体系。"""
    rotation = quat_to_matrix(quat_wxyz)
    return [sum(rotation[row][col] * vector[row] for row in range(3)) for col in range(3)]


def clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def observation(fixture: dict, spec: dict) -> dict:
    """按源公式拼 45 维观测，返回逐块结果（都是源关节顺序的 list）。"""
    ang_vel_body = rotate_inverse(fixture["quat_wxyz"], fixture["world_ang_vel"])
    gravity_body = rotate_inverse(fixture["quat_wxyz"], [0.0, 0.0, -1.0])
    source_order = spec["source_order"]
    defaults = spec["joint_defaults"]

    command = fixture["command"]
    blocks = {
        "ang_vel": [value * 0.25 for value in ang_vel_body],
        "projected_gravity": list(gravity_body),
        "command": [command[0] * 2.0, command[1] * 2.0, command[2] * 0.25],
        "dof_pos": [(fixture["joint_pos"][name] - defaults[name]) * 1.0 for name in source_order],
        "dof_vel": [fixture["joint_vel"][name] * 0.05 for name in source_order],
        "prev_action": [fixture["prev_action"][name] * 1.0 for name in source_order],
    }
    obs: list[float] = []
    for name in ("ang_vel", "projected_gravity", "command", "dof_pos", "dof_vel", "prev_action"):
        obs.extend(blocks[name])
    obs = [clip(value, CLIP_OBS) for value in obs]
    if len(obs) != OBS_DIM:
        raise AssertionError("观测维度不对")
    return {"obs": obs, "blocks": blocks}


def q_target(fixture: dict, spec: dict) -> dict[str, float]:
    """``q_target = q_default + 0.25 * clip(a, ±100)``（源顺序字典）。"""
    return {
        name: spec["joint_defaults"][name] + ACTION_SCALE * clip(fixture["raw_action"][name], CLIP_ACTION)
        for name in spec["source_order"]
    }


def torque(fixture: dict, spec: dict, targets: dict[str, float]) -> dict:
    """``τ = kp(q_target − q) − kd·q̇``，按每关节 effort 裁一次。"""
    tau_pd = {}
    tau_clipped = {}
    for name in spec["source_order"]:
        value = STIFFNESS * (targets[name] - fixture["joint_pos"][name]) - DAMPING * fixture["joint_vel"][name]
        tau_pd[name] = value
        tau_clipped[name] = clip(value, spec["joint_effort"][name])
    return {"tau_pd": tau_pd, "tau_clipped": tau_clipped}


def run(fixture: dict, spec: dict) -> dict:
    """一条 fixture 的完整来源侧结果。"""
    obs = observation(fixture, spec)
    targets = q_target(fixture, spec)
    taus = torque(fixture, spec, targets)
    return {
        "obs": obs["obs"],
        "blocks": obs["blocks"],
        "q_target": targets,
        "tau_pd": taus["tau_pd"],
        "tau_clipped": taus["tau_clipped"],
    }
