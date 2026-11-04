"""200 Hz 关节机械功计量（CAL-014）。

严格按 D-ENERGY-001（§10）实现：

    J = Σ_k Σ_j max(τ_j[k] · q̇_j[k], 0) · dt        k=1..N（N=6000），dt=0.005 s

三条不可协商的约束：

1. **逐物理步**积分：每个 200 Hz 子步用自己的力矩与该子步的关节速度；
   用"每 4 子步抽 1 点"或"50 Hz 速度配 200 Hz 力矩"都不成立
   （后者会把正功低估约 20%，见 §9.3）。
2. **逐关节**截断再求和：先对关节求和再截断会让关节间的正负功互相抵消，
   与协议语义不符（:func:`reference_naive_sum_then_clip` 仅用于对照证明差异）。
3. 只观察，不改动力学：本模块不做任何 sim 写入。

``EnergyMeter`` 服务在线累计；``integrate_series`` 是**独立实现**的离线复算
（向量化 + float64），两者对同一份原始数据必须给出相同结果。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_DT = 0.005
DEFAULT_JOINTS = 12


def steps_for(duration_s: float, dt: float = DEFAULT_DT) -> int:
    """时长 → 物理步数；不是 dt 的整数倍则抛错（禁止静默取整）。"""
    if dt <= 0:
        raise ValueError(f"dt 必须 > 0，得到 {dt}")
    n = round(duration_s / dt)
    if n <= 0 or abs(n * dt - duration_s) > 1e-9:
        raise ValueError(f"{duration_s} s 不是 dt={dt} 的整数倍（{n} 步覆盖 {n * dt} s）")
    return n


def positive_joint_work(torque, qdot, torque_scale: float = 1.0,
                        velocity_scale: float = 1.0) -> list[float]:
    """逐关节正功（单步，未乘 dt）。torque/qdot 为同一时刻的 12 维序列。"""
    if len(torque) != len(qdot):
        raise ValueError(f"力矩与速度维数不一致: {len(torque)} vs {len(qdot)}")
    return [max(t * torque_scale * q * velocity_scale, 0.0) for t, q in zip(torque, qdot)]


def step_energy(torque, qdot, dt: float = DEFAULT_DT,
                torque_scale: float = 1.0, velocity_scale: float = 1.0) -> float:
    """单个物理步的正关节机械功（J）。"""
    return sum(positive_joint_work(torque, qdot, torque_scale, velocity_scale)) * dt


def integrate_series(torque, qdot, dt: float = DEFAULT_DT,
                     torque_scale: float = 1.0, velocity_scale: float = 1.0) -> float:
    """离线复算（独立实现）：torque/qdot 形状 (N, J) 或 (N,)；返回总焦耳。

    用 numpy 向量化 + float64，与在线计量器逐级累加互为对照。
    """
    import numpy as np

    tau = np.asarray(torque, dtype=np.float64)
    qd = np.asarray(qdot, dtype=np.float64)
    if tau.shape != qd.shape:
        raise ValueError(f"形状不一致: {tau.shape} vs {qd.shape}")
    if tau.ndim not in (1, 2):
        raise ValueError(f"期望 (N,) 或 (N, J)，得到 {tau.shape}")
    power = np.maximum(tau * qd * torque_scale * velocity_scale, 0.0)
    return float(power.sum(axis=-1).sum() * dt)


def reference_naive_sum_then_clip(torque, qdot, dt: float = DEFAULT_DT) -> float:
    """**错误**口径的对照实现：先对关节求和再截断。

    仅用于在测试里证明它与契约口径不同（关节间正负功会抵消）。
    任何生产代码不得调用。
    """
    import numpy as np

    tau = np.asarray(torque, dtype=np.float64)
    qd = np.asarray(qdot, dtype=np.float64)
    # 先把关节功求和、再整体截断 —— 这正是契约禁止的做法
    return float(np.maximum((tau * qd).sum(axis=-1), 0.0).sum() * dt)


@dataclass
class EnergyMeter:
    """在线逐物理步累计（只观察，不写仿真）。"""

    dt: float = DEFAULT_DT
    n_joints: int = DEFAULT_JOINTS
    expected_steps: int | None = None
    torque_scale: float = 1.0
    velocity_scale: float = 1.0
    label: str = ""
    steps: int = 0
    total_j: float = 0.0
    per_joint_j: list[float] = field(default_factory=lambda: [0.0] * 0)
    per_step_j: list[float] = field(default_factory=list)
    _started: bool = False
    _stopped: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt 必须 > 0，得到 {self.dt}")
        if self.n_joints <= 0:
            raise ValueError("n_joints 必须 > 0")
        if self.torque_scale <= 0 or self.velocity_scale <= 0:
            raise ValueError("scale 必须 > 0")
        self.per_joint_j = [0.0] * self.n_joints

    # ---- 生命周期 ----
    def start(self) -> None:
        """开始（或重新开始）一个测量窗口；清空累加器。"""
        self.steps = 0
        self.total_j = 0.0
        self.per_joint_j = [0.0] * self.n_joints
        self.per_step_j = []
        self._started = True
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def add_step(self, torque: Sequence[float], qdot: Sequence[float]) -> float:
        """累加一个物理步；返回该步能量（J）。窗口外/未开始/已停止均抛错。"""
        if not self._started:
            raise RuntimeError("计量器未 start（禁止把 warmup 或复位步混入窗口）")
        if self._stopped:
            raise RuntimeError("计量器已 stop，不能再累加")
        if len(torque) != self.n_joints or len(qdot) != self.n_joints:
            raise ValueError(f"期望 {self.n_joints} 维，得到 {len(torque)} / {len(qdot)}")
        for v in list(torque) + list(qdot):
            if not math.isfinite(v):
                raise ValueError(f"输入出现非有限值: {v}")
        contrib = positive_joint_work(torque, qdot, self.torque_scale, self.velocity_scale)
        e = sum(contrib) * self.dt
        self.total_j += e
        self.per_step_j.append(e)
        for j, c in enumerate(contrib):
            self.per_joint_j[j] += c * self.dt
        self.steps += 1
        return e

    # ---- 结果 ----
    @property
    def elapsed_s(self) -> float:
        return self.steps * self.dt

    def verify_complete(self) -> list[str]:
        problems: list[str] = []
        if self.expected_steps is not None and self.steps != self.expected_steps:
            problems.append(f"步数 {self.steps} ≠ 期望 {self.expected_steps}")
        if not math.isfinite(self.total_j) or self.total_j < 0:
            problems.append(f"总能量异常: {self.total_j}")
        if len(self.per_step_j) != self.steps:
            problems.append(f"逐步记录长度 {len(self.per_step_j)} ≠ 步数 {self.steps}")
        if abs(sum(self.per_step_j) - self.total_j) > 1e-9 * max(1.0, abs(self.total_j)):
            problems.append("逐步记录求和与总量不一致")
        if abs(sum(self.per_joint_j) - self.total_j) > 1e-9 * max(1.0, abs(self.total_j)):
            problems.append("逐关节求和与总量不一致")
        return problems

    def summary(self) -> dict:
        return {
            "label": self.label,
            "dt_s": self.dt,
            "n_joints": self.n_joints,
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "expected_steps": self.expected_steps,
            "total_j": self.total_j,
            "per_joint_j": list(self.per_joint_j),
            "unit": "J",
            "metric_id": "D-ENERGY-001",
            "complete": not self.verify_complete(),
        }
