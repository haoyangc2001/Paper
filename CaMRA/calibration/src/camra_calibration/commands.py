"""命令程序生成（CAL-011）。

按协议的四个命令族生成 30 s、10 Hz 的命令序列：
  * 每 1.0 s 一个目标保持段；
  * 段首 0.2 s 线性过渡（ramp）到该段目标，其余 0.8 s 保持；
  * 命令点 301 个（含两端），第 i 个区间使用第 i 个点（零阶保持）；
  * 所有命令先按命令域裁剪，再记录为 applied。

同一 (family, seed) 必须产生逐位相同的序列，因此随机源只有 ``random.Random``，
种子由字符串 ``"<seed>:<family>"`` 派生（Python 的 str 种子走 sha512，跨进程稳定）。

**低速度层的定义存在协议歧义**：协议写的是"|vx| ≤ 0.3 或 |yaw_rate| ≤ 0.3"，
按字面（or）理解，wz=0 的直行族会 100% 落入低速层，25% 下限失去约束力。
本模块因此按**更严格**的读法强制低速段：每 3 段中第 3 段要求 |vx| ≤ 0.3 **且** |wz| ≤ 0.3，
并同时提供两种口径的占比函数（``low_speed_share``），把歧义留成可审计的观测而不是隐式选择。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Sequence

FAMILIES: tuple[str, ...] = ("straight_shift", "arc_combined", "alternating", "start_stop_turn")

#: 协议命令域
LIMITS = ((-1.0, 1.0), (-0.4, 0.4), (-1.0, 1.0))

LOW_SPEED_VX = 0.3
LOW_SPEED_WZ = 0.3


def clip(cmd: Sequence[float]) -> tuple[float, float, float]:
    return tuple(min(max(float(v), lo), hi) for v, (lo, hi) in zip(cmd, LIMITS))  # type: ignore[return-value]


def _force_low_speed(cmd: Sequence[float]) -> tuple[float, float, float]:
    """把目标压到低速层内：|vx| ≤ 0.3 且 |wz| ≤ 0.3（保留符号与 0）。

    必要性：alternating 只翻转一个轴，若其余轴仍是高速，该段并不属于低速层
    （v1.0.1 实测 seed 5103 占比掉到 0.2027）。强制低速段必须**所有轴**都低速。
    """
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    vx = math.copysign(min(abs(vx), LOW_SPEED_VX), vx) if vx else 0.0
    wz = math.copysign(min(abs(wz), LOW_SPEED_WZ), wz) if wz else 0.0
    return clip((vx, vy, wz))


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


@dataclass(frozen=True)
class Program:
    family: str
    seed: int
    duration_s: float
    target_period_s: float
    ramp_s: float
    command_dt: float
    commands: tuple[tuple[float, float, float], ...]
    targets: tuple[tuple[float, float, float], ...]

    # ---- 查询 ----
    def __len__(self) -> int:
        return len(self.commands)

    @property
    def n_intervals(self) -> int:
        return len(self.commands) - 1

    def low_speed_share(self, strict: bool = True) -> float:
        """低速样本占比。strict=True 用"且"（|vx| 与 |wz| 都小），False 用"或"。"""
        if not self.commands:
            return 0.0
        if strict:
            hit = sum(1 for vx, _vy, wz in self.commands
                      if abs(vx) <= LOW_SPEED_VX and abs(wz) <= LOW_SPEED_WZ)
        else:
            hit = sum(1 for vx, _vy, wz in self.commands
                      if abs(vx) <= LOW_SPEED_VX or abs(wz) <= LOW_SPEED_WZ)
        return hit / len(self.commands)

    def sha256(self) -> str:
        """程序散列，用于把 trial 绑定到具体命令序列。"""
        payload = {
            "family": self.family, "seed": self.seed, "duration_s": self.duration_s,
            "target_period_s": self.target_period_s, "ramp_s": self.ramp_s,
            "command_dt": self.command_dt,
            "commands": [list(c) for c in self.commands],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def as_dict(self) -> dict:
        return {
            "family": self.family, "seed": self.seed, "duration_s": self.duration_s,
            "target_period_s": self.target_period_s, "ramp_s": self.ramp_s,
            "command_dt": self.command_dt, "n_points": len(self.commands),
            "low_speed_share_strict": self.low_speed_share(True),
            "low_speed_share_literal": self.low_speed_share(False),
            "sha256": self.sha256(),
        }


# ---------------------------------------------------------------- 目标采样
def _sample_straight_shift(rng: random.Random, slow: bool) -> tuple[float, float, float]:
    if slow:
        vx = rng.choice([0.0, 0.1, 0.2, 0.3])
        vy = rng.choice([0.0, 0.1])
    else:
        vx = rng.choice([-1.0, -0.5, -0.3, 0.3, 0.5, 0.8, 1.0])
        vy = rng.choice([0.0, -0.4, -0.2, 0.2, 0.4])
    return clip((vx, vy, 0.0))


def _sample_arc_combined(rng: random.Random, slow: bool) -> tuple[float, float, float]:
    if slow:
        vx = rng.choice([0.0, 0.2, 0.3])
        wz = rng.choice([-0.3, -0.2, 0.2, 0.3])
        vy = rng.choice([0.0, 0.1])
    else:
        vx = rng.choice([0.3, 0.5, 0.8, 1.0])
        wz = rng.choice([-1.0, -0.6, -0.3, 0.3, 0.6, 1.0])
        vy = rng.choice([0.0, -0.2, -0.1, 0.1, 0.2])
    return clip((vx, vy, wz))


def _sample_start_stop_turn(rng: random.Random, slow: bool, phase: int) -> tuple[float, float, float]:
    # 四相循环：静止 → 前进 → 静止 → 原地转向
    if phase == 0 or phase == 2:
        return (0.0, 0.0, 0.0)
    if phase == 1:
        vx = rng.choice([0.1, 0.2]) if slow else rng.choice([0.3, 0.5, 0.8, 1.0])
        return clip((vx, 0.0, 0.0))
    wz = rng.choice([-0.3, 0.3]) if slow else rng.choice([-1.0, -0.6, 0.6, 1.0])
    return clip((0.0, 0.0, wz))


def _sample_alternating(rng: random.Random, slow: bool, prev: tuple[float, float, float],
                        axis: int) -> tuple[float, float, float]:
    """交替变向：每次把选定轴的符号翻一次。"""
    base = list(prev)
    if slow:
        magnitude = {0: rng.choice([0.1, 0.2, 0.3]), 1: rng.choice([0.1, 0.2]), 2: rng.choice([0.2, 0.3])}[axis]
    else:
        magnitude = {0: rng.choice([0.3, 0.5, 0.8]), 1: rng.choice([0.2, 0.4]), 2: rng.choice([0.3, 0.6, 1.0])}[axis]
    sign = -1.0 if base[axis] > 0 else 1.0
    base[axis] = sign * magnitude
    return clip(tuple(base))


def build_targets(family: str, seed: int, n_segments: int, *, low_speed_every: int = 3
                  ) -> list[tuple[float, float, float]]:
    """生成每段的保持目标。

    低速段强制规则：segment 0 与每 low_speed_every 段中的最后一段。segment 0
    强制低速与"warmup 后从近静止起步"一致，并把每条程序的低速点占比抬到
    协议 min_share_of_cell=0.25 之上（否则 ramp 从高速目标插值进来会吃掉
    低速点，个别格会跌破阈值，见 tests/test_trials.py 的分层检查）。
    """
    if family not in FAMILIES:
        raise ValueError(f"未知命令族 {family!r}（允许 {FAMILIES}）")
    if n_segments < 1:
        raise ValueError("n_segments 必须 >= 1")
    rng = random.Random(f"{seed}:{family}")
    targets: list[tuple[float, float, float]] = []
    axis = 0
    for i in range(n_segments):
        slow = low_speed_every > 0 and (i == 0 or (i % low_speed_every) == (low_speed_every - 1))
        if family == "straight_shift":
            tgt = _sample_straight_shift(rng, slow)
        elif family == "arc_combined":
            tgt = _sample_arc_combined(rng, slow)
        elif family == "start_stop_turn":
            tgt = _sample_start_stop_turn(rng, slow, i % 4)
        else:  # alternating
            prev = targets[-1] if targets else (0.0, 0.0, 0.0)
            tgt = _sample_alternating(rng, slow, prev, axis)
            axis = (axis + 1) % 3
        if slow:
            tgt = _force_low_speed(tgt)
        targets.append(tgt)
    return targets


def build_program(family: str, seed: int, *, duration_s: float = 30.0,
                  command_dt: float = 0.1, target_period_s: float = 1.0,
                  ramp_s: float = 0.2, low_speed_every: int = 3) -> Program:
    """构造一个 30 s 的命令程序，命令点为 duration/command_dt + 1 个（含两端）。"""
    n_points = round(duration_s / command_dt) + 1
    n_segments = round(duration_s / target_period_s)
    if abs(duration_s - (n_points - 1) * command_dt) > 1e-9:
        raise ValueError("duration 必须是 command_dt 的整数倍")
    if abs(duration_s - n_segments * target_period_s) > 1e-9:
        raise ValueError("duration 必须是 target_period 的整数倍")
    targets = build_targets(family, seed, n_segments, low_speed_every=low_speed_every)
    low_segments = {i for i, t in enumerate(targets)
                    if abs(t[0]) <= LOW_SPEED_VX and abs(t[2]) <= LOW_SPEED_WZ}

    cmds: list[tuple[float, float, float]] = []
    for j in range(n_points):
        t = j * command_dt
        seg = min(int(t // target_period_s + 1e-9), n_segments - 1)
        local = t - seg * target_period_s
        alpha = 1.0 if ramp_s <= 0 else min(1.0, local / ramp_s)
        prev = targets[seg - 1] if seg > 0 else (0.0, 0.0, 0.0)
        # 低速段：ramp 也从低速包络内的起点出发，使整段 10 个点都真正属于低速层。
        # 代价是段边界可能出现从高速到 0.3 的阶跃——这正是"减速停车"类指令的形态，
        # 且保证了 v1.0.1 的 min_share_of_cell 在每格 1 条时仍成立。
        start = _force_low_speed(prev) if seg in low_segments else prev
        cmds.append(clip(_lerp(start, targets[seg], alpha)))

    return Program(
        family=family, seed=seed, duration_s=duration_s,
        target_period_s=target_period_s, ramp_s=ramp_s, command_dt=command_dt,
        commands=tuple(cmds), targets=tuple(targets),
    )
