"""平面凸多边形几何原语（CAL-021）。

只依赖 numpy，不启动任何仿真：本模块服务"占地轮廓"的构造与独立几何测试，
包括旋转、窄通道、擦边与连续扫掠四类检查。

约定：

* 多边形顶点按**逆时针（CCW）** 排列，且为凸多边形；内部判据是
  "对每条边都在左侧"（叉积 ≥ 0）。
* 所有长度单位为米，角度为弧度；坐标一律是**机身坐标系**（x 前、y 左）。
* :meth:`Polygon.inflate` 用"多边形 ⊕ 圆盘"的 Minkowski 和实现，
  圆盘取**外切**正多边形（半径 r / cos(π/n)），因此膨胀结果是保守的
  （只会略大，绝不会漏掉真实距离 ≤ r 的点）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

DEFAULT_ARC_VERTICES = 96
_EPS = 1e-12


# ---------------------------------------------------------------- 凸包
def convex_hull_2d(points) -> np.ndarray:
    """Andrew 单调链：返回 CCW 的凸包顶点 (m,2)，共线点被剔除。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"点集形状应为 (N,2)，得到 {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("空点集没有凸包")
    if pts.shape[0] > 5000:
        hull = _hull_scipy(pts)
        if hull is not None:
            return hull
    pts = np.unique(pts, axis=0)
    if pts.shape[0] == 1:
        return pts.copy()
    if pts.shape[0] == 2:
        return pts.copy()

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:                     # 全部共线：退化为线段两端点
        return np.array([pts[0], pts[-1]])
    return _ensure_ccw(hull)


def _hull_scipy(pts: np.ndarray) -> np.ndarray | None:
    """大点集走 scipy 的 C++ 实现；不可用时返回 None 交由纯 Python 路径。"""
    try:
        from scipy.spatial import ConvexHull
    except ModuleNotFoundError:      # pragma: no cover - 环境问题
        return None
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    if poly.shape[0] < 3:
        return None
    return _ensure_ccw(poly)


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if _signed_area(poly) >= 0 else poly[::-1].copy()


def disc(radius: float, n: int = DEFAULT_ARC_VERTICES, circumscribed: bool = True) -> np.ndarray:
    """正多边形近似圆盘；默认取外切（半径放大 1/cos(π/n)）以保证膨胀保守。"""
    if radius <= 0:
        raise ValueError("radius 必须 > 0")
    if n < 8:
        raise ValueError("n 太小，至少 8")
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    r = radius / math.cos(math.pi / n) if circumscribed else radius
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)


# ---------------------------------------------------------------- 多边形
@dataclass(frozen=True)
class Polygon:
    """凸多边形（CCW）。构造时会重新取凸包，因此传入冗余点也安全。"""

    vertices: np.ndarray

    def __post_init__(self) -> None:
        poly = convex_hull_2d(np.asarray(self.vertices, dtype=np.float64))
        object.__setattr__(self, "vertices", _ensure_ccw(poly))

    # ---- 基本量 ----
    @property
    def n(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def area(self) -> float:
        return abs(_signed_area(self.vertices))

    @property
    def perimeter(self) -> float:
        v = self.vertices
        return float(np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1).sum())

    @property
    def circumscribed_radius(self) -> float:
        return float(np.linalg.norm(self.vertices, axis=1).max())

    @property
    def centroid(self) -> np.ndarray:
        v = self.vertices
        cross = v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]
        a = cross.sum()
        if abs(a) < _EPS:
            return v.mean(axis=0)
        cx = ((v[:, 0] + np.roll(v[:, 0], -1)) * cross).sum() / (3.0 * a)
        cy = ((v[:, 1] + np.roll(v[:, 1], -1)) * cross).sum() / (3.0 * a)
        return np.array([cx, cy])

    def bounds(self) -> tuple[float, float, float, float]:
        v = self.vertices
        return (float(v[:, 0].min()), float(v[:, 0].max()),
                float(v[:, 1].min()), float(v[:, 1].max()))

    # ---- 变换 ----
    def rotate(self, theta: float) -> "Polygon":
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        return Polygon(self.vertices @ rot.T)

    def translate(self, delta: Sequence[float]) -> "Polygon":
        return Polygon(self.vertices + np.asarray(delta, dtype=np.float64))

    # ---- 判据 ----
    def contains(self, points, eps: float = 0.0) -> np.ndarray:
        """逐点判断是否在内部（含边界）；eps>0 时把边界向外放宽 eps。

        逐边循环、对点向量化：不会因 (点数 × 边数) 的中间张量撑爆内存。
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.shape[1] != 2:
            raise ValueError(f"点集形状应为 (N,2)，得到 {pts.shape}")
        v = self.vertices
        nxt = np.roll(v, -1, axis=0)
        inside = np.ones(pts.shape[0], dtype=bool)
        for (vx, vy), (nx, ny) in zip(v, nxt):
            ex, ey = nx - vx, ny - vy
            length = math.hypot(ex, ey)
            if length <= _EPS:
                continue
            signed = (ex * (pts[:, 1] - vy) - ey * (pts[:, 0] - vx)) / length
            inside &= signed >= -eps
            if not inside.any():
                break
        return inside

    def distance(self, points) -> np.ndarray:
        """到多边形的有符号距离：内部为负、边界为 0、外部为正。"""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.shape[1] != 2:
            raise ValueError(f"点集形状应为 (N,2)，得到 {pts.shape}")
        v = self.vertices
        nxt = np.roll(v, -1, axis=0)
        best = np.full(pts.shape[0], np.inf)
        inside = np.ones(pts.shape[0], dtype=bool)
        for (vx, vy), (nx, ny) in zip(v, nxt):
            ex, ey = nx - vx, ny - vy
            l2 = ex * ex + ey * ey
            length = math.sqrt(l2)
            if length <= _EPS:
                continue
            relx, rely = pts[:, 0] - vx, pts[:, 1] - vy
            signed = (ex * rely - ey * relx) / length
            inside &= signed >= 0.0
            t = np.clip((relx * ex + rely * ey) / l2, 0.0, 1.0)
            dx = relx - t * ex
            dy = rely - t * ey
            np.minimum(best, np.hypot(dx, dy), out=best)
        return np.where(inside, -best, best)

    def min_width(self) -> float:
        """最小宽度（= 各边法向支撑宽度的最小值）：窄通道通过性的直接判据。

        退化边（长度为 0，顶点重复）会被跳过，否则会得到虚假的 0 宽度。
        """
        v = self.vertices
        nxt = np.roll(v, -1, axis=0)
        edge = nxt - v
        norm = np.linalg.norm(edge, axis=1)
        keep = norm > 1e-12
        if not keep.any():
            raise ValueError("多边形所有边都退化，无法计算宽度")
        normals = np.stack([-edge[keep, 1], edge[keep, 0]], axis=1) / norm[keep][:, None]
        proj = v @ normals.T                       # (m, k)
        return float((proj.max(axis=0) - proj.min(axis=0)).min())

    def width_along(self, direction: Sequence[float]) -> float:
        """沿给定方向的宽度（支撑函数之和）：窄通道通过性用旋转后的该值判断。"""
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n <= _EPS:
            raise ValueError("方向向量不能为零")
        d = d / n
        proj = self.vertices @ d
        return float(proj.max() - proj.min())

    def support(self, direction: Sequence[float]) -> float:
        d = np.asarray(direction, dtype=np.float64)
        d = d / max(np.linalg.norm(d), _EPS)
        return float((self.vertices @ d).max())

    # ---- 膨胀与并集 ----
    def inflate(self, radius: float, arc_vertices: int = DEFAULT_ARC_VERTICES) -> "Polygon":
        """Minkowski 和 F ⊕ disc(r)；radius=0 直接返回自身。"""
        if radius < 0:
            raise ValueError("radius 必须 ≥ 0")
        if radius == 0:
            return self
        d = disc(radius, arc_vertices, circumscribed=True)
        sums = (self.vertices[:, None, :] + d[None, :, :]).reshape(-1, 2)
        return Polygon(sums)

    def union(self, other: "Polygon") -> "Polygon":
        """两个凸多边形的凸包并（保守外包）。"""
        return Polygon(np.vstack([self.vertices, other.vertices]))

    # ---- 序列化 ----
    def as_dict(self) -> dict:
        return {
            "vertices": [[float(x), float(y)] for x, y in self.vertices],
            "area": self.area,
            "perimeter": self.perimeter,
            "circumscribed_radius": self.circumscribed_radius,
            "min_width": self.min_width(),
            "bounds_xyxy": list(self.bounds()),
            "centroid": [float(v) for v in self.centroid],
        }

    @classmethod
    def from_dict(cls, payload: dict | list) -> "Polygon":
        verts = payload["vertices"] if isinstance(payload, dict) else payload
        return cls(np.asarray(verts, dtype=np.float64))


# ---------------------------------------------------------------- 连续扫掠
def minkowski_sum(a: "Polygon", b: "Polygon") -> "Polygon":
    """两个凸多边形的 Minkowski 和（= 凸包下的逐点相加）。"""
    sums = (a.vertices[:, None, :] + b.vertices[None, :, :]).reshape(-1, 2)
    return Polygon(sums)


def swept_collision(polygon: Polygon, p0: Sequence[float], p1: Sequence[float],
                    obstacle: Sequence[float]) -> bool:
    """纯平移扫掠的**精确**碰撞判据：obstacle ∈ (线段 p0→p1) ⊕ F。

    等价于把起点与终点的多边形取凸包后做包含判断——不需要对时间采样，
    因此细长障碍物也不会像"逐帧检测"那样被漏掉。
    """
    a = polygon.translate(p0)
    b = polygon.translate(p1)
    swept = a.union(b)
    return bool(swept.contains(np.asarray(obstacle, dtype=np.float64).reshape(1, 2))[0])


def rotating_sweep_hull(polygon: Polygon, poses: Iterable[tuple[Sequence[float], float]],
                        substeps: int = 8) -> Polygon:
    """带旋转的扫掠**采样凸包**：在相邻位姿之间线性插值位置与角度后取凸包。

    语义必须说清楚：采样凸包是连续扫掠区域的**下界**（采样越多只会变大），
    因此 ``substeps`` 增大时面积单调不减并收敛到真值附近——它适合做收敛性检查
    与粗略外包，**不能**当作安全边界使用。真实碰撞判定用 :func:`swept_collision`
    （纯平移下的精确判据）。
    """
    poses = [(np.asarray(p, dtype=np.float64), float(th)) for p, th in poses]
    if not poses:
        raise ValueError("空位姿序列")
    if substeps < 1:
        raise ValueError("substeps 必须 ≥ 1")
    samples: list[np.ndarray] = []
    for (pa, ta), (pb, tb) in zip(poses[:-1], poses[1:]):
        for s in range(substeps + 1):
            u = s / (substeps + 1)
            p = pa + (pb - pa) * u
            th = ta + (tb - ta) * u
            samples.append(polygon.rotate(th).translate(p).vertices)
    samples.append(poses[-1][0][None, :] + polygon.rotate(poses[-1][1]).vertices)
    return Polygon(np.vstack(samples))
