"""确定性平面名义运动（CAL-011）。

状态只有平面位姿 (px, py, yaw)。给定机身坐标系速度指令 u=[vx, vy, wz]，
在航向旋转后积分：

    yaw(t+dt) = yaw(t) + wz·dt
    p(t+dt)   = p(t) + ∫_0^dt R(yaw(t)+wz·s)·[vx, vy]ᵀ ds

对分段常量命令有闭式解（见 :func:`step_exact`），因此本模块不依赖任何数值积分器。
另提供两个独立对照：
  * :func:`integrate_rk4`：四阶精度，用于与闭式解做数值一致性检查（容差 :data:`CROSSCHECK_TOL`）；
  * :func:`integrate_euler`：一阶精度，用于验证误差随步长线性收敛（阶数检查）。

本函数只接受命令与初态，不接受 profile、扰动或执行侧量，所以"扰动不改变名义轨迹"
是构造性保证，而不是事后断言。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

CROSSCHECK_TOL = 1e-9
_EPS = 1e-12


@dataclass(frozen=True)
class PlanarState:
    px: float
    py: float
    yaw: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.px, self.py, self.yaw)


def wrap_angle(a: float) -> float:
    """把角度折到 (-pi, pi]，便于比较但不改变轨迹。"""
    two_pi = 2.0 * math.pi
    a = math.fmod(a + math.pi, two_pi)
    if a <= 0:
        a += two_pi
    return a - math.pi


def step_exact(state: PlanarState, cmd: Sequence[float], dt: float) -> PlanarState:
    """分段常量命令下的精确推进（圆弧闭式解，wz=0 时退化为直线）。"""
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    th0 = state.yaw
    dyaw = wz * dt

    if abs(wz) < _EPS:
        # 直线：航向不变
        c, s = math.cos(th0), math.sin(th0)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
    else:
        th1 = th0 + dyaw
        # ∫R(θ0+wz·s)v ds 的闭式解
        sx = (math.sin(th1) - math.sin(th0)) / wz
        cx = (math.cos(th1) - math.cos(th0)) / wz
        dx = vx * sx + vy * cx
        dy = -vx * cx + vy * sx
    return PlanarState(state.px + dx, state.py + dy, th0 + dyaw)


def integrate(initial: PlanarState, commands: Sequence[Sequence[float]], dt: float) -> list[PlanarState]:
    """按命令点序列推进。commands 有 N+1 个点（含两端），返回 N+1 个状态。

    第 i 个区间使用 commands[i]（零阶保持），与"命令在区间起点生效"一致。
    """
    if len(commands) < 2:
        raise ValueError("至少需要两个命令点（起点与终点）")
    out = [initial]
    cur = initial
    for i in range(len(commands) - 1):
        cur = step_exact(cur, commands[i], dt)
        out.append(cur)
    return out


def integrate_euler(initial: PlanarState, commands: Sequence[Sequence[float]], dt: float,
                    substeps: int = 200) -> list[PlanarState]:
    """密采样 Euler 积分，作为 exact 的独立对照（不参与生产路径）。"""
    if substeps < 1:
        raise ValueError("substeps 必须 >= 1")
    h = dt / substeps
    out = [initial]
    cur = initial
    for i in range(len(commands) - 1):
        vx, vy, wz = (float(commands[i][0]), float(commands[i][1]), float(commands[i][2]))
        px, py, th = cur.px, cur.py, cur.yaw
        for _ in range(substeps):
            c, s = math.cos(th), math.sin(th)
            px += (vx * c - vy * s) * h
            py += (vx * s + vy * c) * h
            th += wz * h
        cur = PlanarState(px, py, th)
        out.append(cur)
    return out


def _f(vx: float, vy: float, wz: float):
    def rhs(px: float, py: float, th: float) -> tuple[float, float, float]:
        c, s = math.cos(th), math.sin(th)
        return (vx * c - vy * s, vx * s + vy * c, wz)
    return rhs


def integrate_rk4(initial: PlanarState, commands: Sequence[Sequence[float]], dt: float,
                  substeps: int = 50) -> list[PlanarState]:
    """四阶 Runge-Kutta 积分，作为闭式解的独立数值对照。"""
    if substeps < 1:
        raise ValueError("substeps 必须 >= 1")
    h = dt / substeps
    out = [initial]
    cur = initial
    for i in range(len(commands) - 1):
        f = _f(float(commands[i][0]), float(commands[i][1]), float(commands[i][2]))
        px, py, th = cur.px, cur.py, cur.yaw
        for _ in range(substeps):
            k1 = f(px, py, th)
            k2 = f(px + 0.5 * h * k1[0], py + 0.5 * h * k1[1], th + 0.5 * h * k1[2])
            k3 = f(px + 0.5 * h * k2[0], py + 0.5 * h * k2[1], th + 0.5 * h * k2[2])
            k4 = f(px + h * k3[0], py + h * k3[1], th + h * k3[2])
            px += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            py += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            th += h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        cur = PlanarState(px, py, th)
        out.append(cur)
    return out


def max_planar_deviation(a: Sequence[PlanarState], b: Sequence[PlanarState]) -> float:
    """两条轨迹在同一时间网格上的最大平面距离（不做时间插值）。"""
    if len(a) != len(b):
        raise ValueError(f"轨迹长度不一致: {len(a)} vs {len(b)}")
    worst = 0.0
    for sa, sb in zip(a, b):
        d = math.hypot(sa.px - sb.px, sa.py - sb.py)
        worst = max(worst, d)
    return worst


def positions(traj: Iterable[PlanarState]) -> list[tuple[float, float]]:
    return [(s.px, s.py) for s in traj]
