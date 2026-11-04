"""五类物理执行条件的采样与追溯记录（CAL-013）。

协议（config/calibration_protocol_v1.0.0.json 的 profiles 段）定义了五个 profile：

============  ===========  ============  ====================  ====================
profile       friction     added_mass    push                  obs_noise
============  ===========  ============  ====================  ====================
clean         1.0          0.0           禁用                  禁用
friction_only U(0.3,1.25)  0.0           禁用                  禁用
mass_only     1.0          U(-1.0,3.0)   禁用                  禁用
push_only     1.0          0.0           每 5-10 s，±0.5 m/s   禁用
combined      U(0.3,1.25)  U(-1.0,3.0)   每 5-10 s，±0.5 m/s   训练配置现有值
============  ===========  ============  ====================  ====================

采样规则（协议 profile_rules.sampling_unit）：摩擦与质量按 trial 采样一次并在整条
轨迹内保持不变；push 事件在测量窗口内按 interval 范围触发。随机流与命令程序独立
（盐不同：``<seed>:profile:<profile_id>``）。

``sample_applied`` 只做采样与记录，不做任何仿真写入；仿真侧的逐 env 施加在
``CaMRA/calibration/sim/profiles_sim.py``（需要 torch / isaaclab，不进本包）。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Tuple

PROFILE_IDS: tuple[str, ...] = ("clean", "friction_only", "mass_only", "push_only", "combined")

#: 协议取值（config/calibration_protocol_v1.0.0.json）
FRICTION_RANGE: tuple[float, float] = (0.3, 1.25)
MASS_RANGE: tuple[float, float] = (-1.0, 3.0)
PUSH_VEL_RANGE: tuple[float, float] = (-0.5, 0.5)
PUSH_INTERVAL_RANGE: tuple[float, float] = (5.0, 10.0)

#: clean 的固定值（协议 profiles[0]）
CLEAN_FRICTION: float = 1.0
CLEAN_RESTITUTION: float = 0.0
CLEAN_MASS: float = 0.0

#: 训练配置现有观测噪声（velocity_env_cfg.py PolicyCfg 的 Unoise，camra 未覆盖）
TRAINING_OBS_NOISE: dict[str, tuple[float, float]] = {
    "base_ang_vel": (-0.2, 0.2),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos_rel": (-0.01, 0.01),
    "joint_vel_rel": (-1.5, 1.5),
}

#: 各 profile 相对 clean 变化的旋钮（协议 profiles 段的字面含义）
PROFILE_KNOBS: dict[str, frozenset[str]] = {
    "clean": frozenset(),
    "friction_only": frozenset({"friction"}),
    "mass_only": frozenset({"added_mass"}),
    "push_only": frozenset({"push_events"}),
    "combined": frozenset({"friction", "added_mass", "push_events", "obs_noise"}),
}

KNOB_ORDER: tuple[str, ...] = ("friction", "added_mass", "push_events", "obs_noise", "restitution")


@dataclass(frozen=True)
class AppliedProfile:
    """一条 trial 实际生效的执行条件（可追溯记录的最小完整集）。"""

    profile_id: str
    seed: int
    duration_s: float
    #: static == dynamic 的单一摩擦系数；clean 固定 1.0
    friction: float
    #: 相对默认质心的增量，kg；clean 固定 0.0
    added_mass_kg: float
    #: (t_s, vx, vy)，t 相对测量窗口起点；clean 为空
    push_events: tuple[tuple[float, float, float], ...]
    #: None 表示禁用观测噪声；combined 为训练现有值
    obs_noise: dict[str, tuple[float, float]] | None
    restitution: float = CLEAN_RESTITUTION

    # ---- 不变量（分组标签与实际配置一致） ----
    def knobs_changed(self) -> frozenset[str]:
        changed: set[str] = set()
        if abs(self.friction - CLEAN_FRICTION) > 1e-12:
            changed.add("friction")
        if abs(self.added_mass_kg - CLEAN_MASS) > 1e-12:
            changed.add("added_mass")
        if self.push_events:
            changed.add("push_events")
        if self.obs_noise is not None:
            changed.add("obs_noise")
        if abs(self.restitution - CLEAN_RESTITUTION) > 1e-12:
            changed.add("restitution")
        return frozenset(changed)

    def check_invariants(self) -> list[str]:
        """返回问题列表；空表示与协议 profile 定义一致。"""
        problems: list[str] = []
        expect = PROFILE_KNOBS[self.profile_id]
        got = self.knobs_changed()
        if got != expect:
            problems.append(f"旋钮集合不符: 期望 {sorted(expect)}, 实际 {sorted(got)}")
        if self.profile_id in ("clean", "mass_only", "push_only") and self.friction != CLEAN_FRICTION:
            problems.append(f"friction 应为 {CLEAN_FRICTION}, 得到 {self.friction}")
        if self.profile_id in ("clean", "friction_only", "push_only") and self.added_mass_kg != CLEAN_MASS:
            problems.append(f"added_mass_kg 应为 {CLEAN_MASS}, 得到 {self.added_mass_kg}")
        if not self.push_events and "push_events" in expect:
            problems.append("push 事件为空")
        if self.push_events and "push_events" not in expect:
            problems.append("该 profile 不应有 push 事件")
        if (self.obs_noise is not None) != ("obs_noise" in expect):
            problems.append("obs_noise 开关与 profile 定义不符")
        if self.obs_noise is not None and dict(self.obs_noise) != TRAINING_OBS_NOISE:
            problems.append("obs_noise 必须等于训练配置现有值")
        for t, vx, vy in self.push_events:
            if not (0.0 < t <= self.duration_s):
                problems.append(f"push 时刻 {t} 超出 (0, {self.duration_s}]")
            if not (PUSH_VEL_RANGE[0] <= vx <= PUSH_VEL_RANGE[1]):
                problems.append(f"push vx {vx} 超出 {PUSH_VEL_RANGE}")
            if not (PUSH_VEL_RANGE[0] <= vy <= PUSH_VEL_RANGE[1]):
                problems.append(f"push vy {vy} 超出 {PUSH_VEL_RANGE}")
        if self.profile_id in ("friction_only", "combined"):
            if not (FRICTION_RANGE[0] <= self.friction <= FRICTION_RANGE[1]):
                problems.append(f"friction {self.friction} 超出 {FRICTION_RANGE}")
        if self.profile_id in ("mass_only", "combined"):
            if not (MASS_RANGE[0] <= self.added_mass_kg <= MASS_RANGE[1]):
                problems.append(f"added_mass_kg {self.added_mass_kg} 超出 {MASS_RANGE}")
        return problems

    # ---- 序列化 ----
    def to_manifest_record(self) -> dict:
        """写入 trial 清单 ``profile_applied`` 字段的记录。"""
        return {
            "friction": self.friction,
            "added_mass_kg": self.added_mass_kg,
            "push_events": [[t, vx, vy] for t, vx, vy in self.push_events],
            "obs_noise": ("disabled" if self.obs_noise is None else
                          {k: list(v) for k, v in self.obs_noise.items()}),
        }

    def to_dense_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "friction": self.friction,
            "restitution": self.restitution,
            "added_mass_kg": self.added_mass_kg,
            "push_events": [[t, vx, vy] for t, vx, vy in self.push_events],
            "obs_noise": (None if self.obs_noise is None else
                          {k: list(v) for k, v in self.obs_noise.items()}),
            "knobs_changed": sorted(self.knobs_changed()),
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dense_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def sample_applied(profile_id: str, seed: int, duration_s: float = 30.0) -> AppliedProfile:
    """按 trial seed 确定性地采样一个 profile 的实际配置。

    随机流盐为 ``<seed>:profile:<profile_id>``，与命令程序的 ``<seed>:<family>``
    互不重叠；同一 (profile, seed) 两次调用结果逐位一致。
    """
    if profile_id not in PROFILE_IDS:
        raise ValueError(f"未知 profile {profile_id!r}（允许 {PROFILE_IDS}）")
    if duration_s <= 0:
        raise ValueError("duration_s 必须 > 0")
    rng = random.Random(f"{seed}:profile:{profile_id}")
    use_friction = profile_id in ("friction_only", "combined")
    use_mass = profile_id in ("mass_only", "combined")
    use_push = profile_id in ("push_only", "combined")
    noise = TRAINING_OBS_NOISE if profile_id == "combined" else None

    friction = rng.uniform(*FRICTION_RANGE) if use_friction else CLEAN_FRICTION
    mass = rng.uniform(*MASS_RANGE) if use_mass else CLEAN_MASS

    events: list[tuple[float, float, float]] = []
    if use_push:
        t = rng.uniform(*PUSH_INTERVAL_RANGE)
        while t <= duration_s:
            events.append((round(t, 6), rng.uniform(*PUSH_VEL_RANGE), rng.uniform(*PUSH_VEL_RANGE)))
            t += rng.uniform(*PUSH_INTERVAL_RANGE)

    applied = AppliedProfile(
        profile_id=profile_id, seed=seed, duration_s=duration_s,
        friction=friction, added_mass_kg=mass,
        push_events=tuple(events), obs_noise=noise,
    )
    problems = applied.check_invariants()
    if problems:
        raise AssertionError(f"采样结果违反协议不变量: {problems}")
    return applied


#: 协议 initial_state.plane_pose_sampling
PLANE_POSE_RANGE: dict[str, tuple[float, float]] = {
    "px_m": (-1.0, 1.0),
    "py_m": (-1.0, 1.0),
    "yaw_rad": (-3.141592653589793, 3.141592653589793),
}


def sample_initial_pose(seed: int) -> dict[str, float]:
    """按 trial seed 确定性采样平面初始位姿（协议 initial_state 规则）。

    随机流盐为 ``<seed>:initial``，与命令/ profile 流互不重叠；
    名义侧与执行侧共享同一位姿（同一 pair）。
    """
    rng = random.Random(f"{seed}:initial")
    return {
        "px": rng.uniform(*PLANE_POSE_RANGE["px_m"]),
        "py": rng.uniform(*PLANE_POSE_RANGE["py_m"]),
        "yaw": rng.uniform(*PLANE_POSE_RANGE["yaw_rad"]),
    }


def knob_diff() -> dict[str, dict[str, str]]:
    """逐 profile × 旋钮的 diff 矩阵，证明只有指定旋钮变化。"""
    out: dict[str, dict[str, str]] = {}
    for pid in PROFILE_IDS:
        out[pid] = {k: ("varied" if k in PROFILE_KNOBS[pid] else "unchanged")
                    for k in KNOB_ORDER}
    return out
