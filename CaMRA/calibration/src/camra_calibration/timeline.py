"""共享命令时间轴（CAL-011）。

把协议里的四个频率换算成同一 30 s 窗口上的**区间数**与**采样点数**，并保证在
t=0 与 t=30 s 两端严格对齐：所有速率下的区间数乘以周期必须精确等于测量窗口长度。

用词约定（避免 off-by-one 争议）：
  * interval（区间）：一个采样周期，共 N 个；
  * point（点）：含两端的采样位置，共 N+1 个。

  物理  200 Hz：6000 区间，步末采样 6000 个（t=0 的初态单独给出，不占步末采样）
  策略   50 Hz：1500 区间，1501 点
  命令   10 Hz：300 区间，301 点
  配对   10 Hz：300 区间，301 点

warmup 与测量窗口是两段互不重叠的区间：warmup 的 1 s 只用于稳定，任何记录都从测量窗口的
第 0 个点开始，因此 warmup 不会被混进正式数据。
"""
from __future__ import annotations

from dataclasses import dataclass

from .protocol import Protocol

TOL = 1e-9


@dataclass(frozen=True)
class Timeline:
    sim_dt: float
    policy_dt: float
    command_dt: float
    pairing_dt: float
    warmup_s: float
    measured_window_s: float

    # ---- 由频率与窗口派生 ----
    @property
    def physics_intervals(self) -> int:
        return _count(self.measured_window_s, self.sim_dt, "physics")

    @property
    def policy_intervals(self) -> int:
        return _count(self.measured_window_s, self.policy_dt, "policy")

    @property
    def command_intervals(self) -> int:
        return _count(self.measured_window_s, self.command_dt, "command")

    @property
    def pairing_intervals(self) -> int:
        return _count(self.measured_window_s, self.pairing_dt, "pairing")

    @property
    def pairing_points(self) -> int:
        return self.pairing_intervals + 1

    @property
    def warmup_intervals(self) -> int:
        return _count(self.warmup_s, self.sim_dt, "warmup")

    # ---- 时间 ↔ 索引 ----
    def physics_time(self, step_index: int) -> float:
        """第 k 个物理区间结束时刻（k=1..N）。k=0 表示初态。"""
        if not (0 <= step_index <= self.physics_intervals):
            raise IndexError(f"physics index {step_index} 超出 [0, {self.physics_intervals}]")
        return step_index * self.sim_dt

    def command_time(self, point_index: int) -> float:
        if not (0 <= point_index < self.pairing_points):
            raise IndexError(f"command point {point_index} 超出 [0, {self.pairing_points})")
        return point_index * self.command_dt

    def policy_time(self, point_index: int) -> float:
        if not (0 <= point_index <= self.policy_intervals):
            raise IndexError(f"policy point {point_index} 超出 [0, {self.policy_intervals}]")
        return point_index * self.policy_dt

    # ---- 一致性 ----
    def ratios(self) -> dict[str, int]:
        """整数倍关系：(200/50, 50/10, 200/10)。非整数则抛错。"""
        p2s = self.physics_intervals / self.policy_intervals
        p2c = self.policy_intervals / self.command_intervals
        if abs(p2s - round(p2s)) > TOL or abs(p2c - round(p2c)) > TOL:
            raise AssertionError(f"频率比不是整数：physics/policy={p2s}, policy/command={p2c}")
        return {"physics_per_policy": int(round(p2s)), "policy_per_command": int(round(p2c)),
                "physics_per_command": int(round(p2s * p2c))}

    def check(self) -> list[str]:
        """返回问题列表；空表示全部对齐。"""
        problems: list[str] = []
        for name, n, dt in (
            ("physics", self.physics_intervals, self.sim_dt),
            ("policy", self.policy_intervals, self.policy_dt),
            ("command", self.command_intervals, self.command_dt),
            ("pairing", self.pairing_intervals, self.pairing_dt),
        ):
            span = n * dt
            if abs(span - self.measured_window_s) > TOL:
                problems.append(f"{name}: {n} × {dt} = {span} ≠ {self.measured_window_s}")
        if abs(self.warmup_intervals * self.sim_dt - self.warmup_s) > TOL:
            problems.append(f"warmup: {self.warmup_intervals} × {self.sim_dt} ≠ {self.warmup_s}")
        try:
            r = self.ratios()
            if r["physics_per_policy"] != round(self.policy_dt / self.sim_dt):
                problems.append("physics_per_policy 与 decimation 不符")
        except AssertionError as exc:
            problems.append(str(exc))
        if self.warmup_intervals <= 0:
            problems.append("warmup 必须为正")
        return problems

    @classmethod
    def from_protocol(cls, protocol: Protocol | None = None) -> "Timeline":
        p = protocol or Protocol.load()
        k = p.payload["clocks"]
        return cls(
            sim_dt=float(k["sim_dt_s"]),
            policy_dt=float(k["policy_dt_s"]),
            command_dt=float(k["command_dt_s"]),
            pairing_dt=float(k["pairing_dt_s"]),
            warmup_s=float(k["warmup_s"]),
            measured_window_s=float(k["measured_window_s"]),
        )


def _count(span: float, dt: float, name: str) -> int:
    n = round(span / dt)
    if n <= 0 or abs(n * dt - span) > TOL:
        raise ValueError(f"{name}: {span} s 不能被 {dt} s 整除（得到 {n} 个区间，跨度 {n * dt} s）")
    return n
