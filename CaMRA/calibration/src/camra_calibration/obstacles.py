"""近障地图与名义路径净距（CAL-023）。

用途：在**正式采集之前**预先冻结 200 条"地图 + 命令"设计。设计只依赖已冻结的
占地轮廓 ``footprint_v1.json`` 与名义平面运动，**不读取**任何正式执行误差——
δ 要等 CAL-025 才有，因此每条 case 记录"名义路径上的最小净距"，
将来只需检查 ``δ ≤ 该净距`` 就能判断该 case 是否仍然安全。

场地尺寸（协议 v1.0.2）
----------------------
每例自适应：``arena = 名义路径包围盒 + 2×(轮廓法向半宽 + 目标净距 + 1.0 m 安全边)``，
向上取到 0.5 m，且不小于 8×8 m；边界墙计入地图（距任何障碍与路径 ≥ 1.0 m，
因此不会干扰近障判据）。固定 8×8 m 装不下 arc_combined/straight_shift 的 30 s 名义路径
（实测包围盒最长 11.8×11.5 / 10.1×3.6 m）。

坐标与单位
----------
* 世界系 x 向东、y 向北，原点在场地左下角，范围 [0,W]×[0,H]；
* 长度单位米，角度弧度；障碍为凸多边形（矩形按 yaw 旋转）；
* 净距定义：名义路径每个位姿下，``footprint``（按该位姿 yaw 旋转 + 平移）到障碍
  多边形的最近距离，再对所有位姿取最小值。凸多边形间的最近距离等于
  "一方顶点到另一方多边形"的最小值，实现里两边都算一遍取最小。

只做设计，不做仿真：障碍如何进入 Isaac 场景属于 CAL-028 的实现范围。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Polygon
from .nominal import PlanarState, integrate

CLEARANCE_TIERS = (0.05, 0.10, 0.20, 0.35)      # 目标净距（m），决定门洞/走廊宽度
SAFETY_EDGE = 1.0                                # 场地边界到路径/障碍的最小安全边（m）
MIN_ARENA = 8.0                                  # 场地下限（m），与 v1.0.1 的 8×8 一致
ARENA_STEP = 0.5                                 # 场地尺寸取整步长
GATE_FRACTION = 0.35                             # 双侧"门"至少占障碍数量的比例
OBSTACLE_SIZES = ((0.30, 0.30), (0.50, 0.20), (0.80, 0.15))
MIN_DESIGN_CLEARANCE = 0.02                      # 设计下限：任何位姿都不得比这更近
YAW_CANDIDATES = tuple(math.radians(a) for a in range(0, 180, 5))


class ObstacleError(RuntimeError):
    """地图/净距参数不合法。"""


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float          # 沿局部 x（切线方向）
    depth: float          # 沿局部 y（法向）
    yaw: float

    def polygon(self) -> Polygon:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hw, hd = self.width / 2.0, self.depth / 2.0
        local = np.array([[hw, hd], [-hw, hd], [-hw, -hd], [hw, -hd]])
        rot = np.array([[c, -s], [s, c]])
        return Polygon(local @ rot.T + np.array([self.x, self.y]))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "depth": self.depth,
                "yaw": self.yaw}

    @classmethod
    def from_dict(cls, d: dict) -> "Obstacle":
        return cls(x=float(d["x"]), y=float(d["y"]), width=float(d["width"]),
                   depth=float(d["depth"]), yaw=float(d["yaw"]))


def polygon_distance(a: Polygon, b: Polygon) -> float:
    """两个凸多边形的最近距离（相交时为 0），单对调用版。"""
    va = a.vertices[None, :, :]
    return float(distances_to_obstacle(va, b)[0])


def _edge_normals(poly: np.ndarray) -> np.ndarray:
    """多边形（(V,2) 或 (F,V,2)）各边的单位外法向。"""
    if poly.ndim == 2:
        edge = np.roll(poly, -1, axis=0) - poly
        norm = np.linalg.norm(edge, axis=1)
        return np.stack([edge[:, 1], -edge[:, 0]], axis=1) / np.maximum(norm, 1e-12)[:, None]
    edge = np.roll(poly, -1, axis=1) - poly
    norm = np.linalg.norm(edge, axis=2)
    return np.stack([edge[:, :, 1], -edge[:, :, 0]], axis=2) / np.maximum(norm, 1e-12)[:, :, None]


def _signed_points_to_polygons(pts: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """一组点 (P,2) 对一批凸多边形 (F,V,2) 的有符号距离 (F,P)：内部为负、边界 0。

    逐边向量化（张量规模 F×P×V，本用途下很小），不依赖逐帧 Python 循环。
    """
    pts = np.asarray(pts, dtype=np.float64)
    polys = np.asarray(polys, dtype=np.float64)
    if polys.ndim != 3:
        raise ObstacleError(f"多边形批次应为 (F,V,2)，得到 {polys.shape}")
    a = polys                                   # (F,V,2)
    edge = np.roll(a, -1, axis=1) - a           # (F,V,2)
    l2 = (edge ** 2).sum(axis=2)                # (F,V)
    length = np.sqrt(np.maximum(l2, 1e-24))
    rel = pts[None, :, None, :] - a[:, None, :, :]      # (F,P,V,2)
    relx, rely = rel[..., 0], rel[..., 1]
    ex, ey = edge[:, None, :, 0], edge[:, None, :, 1]   # (F,1,V)
    signed = (ex * rely - ey * relx) / length[:, None, :]
    inside = (signed >= 0.0).all(axis=2)                # (F,P)
    t = np.clip((relx * ex + rely * ey) / l2[:, None, :], 0.0, 1.0)
    dist = np.hypot(relx - t * ex, rely - t * ey).min(axis=2)   # (F,P)
    return np.where(inside, -dist, dist)


def distances_to_obstacle(fp_vertices: np.ndarray, obstacle: Polygon) -> np.ndarray:
    """一批轮廓多边形（F,V,2）到同一障碍的最近距离（F,）。相交返回 0。

    双向取最小：障碍各顶点到轮廓、轮廓各顶点到障碍；任一方向的点落进对方内部即判相交。
    凸多边形之间的最近距离一定由某一方的顶点到另一方决定，因此双向取最小是精确的。
    """
    fp = np.asarray(fp_vertices, dtype=np.float64)
    if fp.ndim != 3:
        raise ObstacleError(f"轮廓顶点应为 (F,V,2)，得到 {fp.shape}")
    obs = obstacle.vertices
    d_obs = _signed_points_to_polygons(obs, fp)                     # (F,M)
    d_fp = _signed_points_to_polygons(fp.reshape(-1, 2), obs[None, :, :])[0]
    d_fp = d_fp.reshape(fp.shape[0], fp.shape[1])                    # (F,V)
    inside = (d_obs <= 0.0).any(axis=1) | (d_fp <= 0.0).any(axis=1)
    dist = np.minimum(np.abs(d_obs).min(axis=1), np.abs(d_fp).min(axis=1))
    return np.where(inside, 0.0, dist)


def footprint_vertices_batch(footprint: Polygon, pos: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """把轮廓按每一帧的 (pos, yaw) 变换，返回 (F,V,2)。"""
    v = footprint.vertices
    c, s = np.cos(yaw), np.sin(yaw)
    x = v[None, :, 0] * c[:, None] - v[None, :, 1] * s[:, None] + pos[:, 0:1]
    y = v[None, :, 0] * s[:, None] + v[None, :, 1] * c[:, None] + pos[:, 1:2]
    return np.stack([x, y], axis=2)


def nominal_path(commands, start: Sequence[float], yaw0: float,
                 dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """由命令序列积分出名义平面路径，返回 (positions (N,2), yaws (N,))。"""
    states = integrate(PlanarState(float(start[0]), float(start[1]), float(yaw0)),
                       np.asarray(commands, dtype=np.float64), dt)
    pos = np.array([[s.px, s.py] for s in states], dtype=np.float64)
    yaw = np.array([s.yaw for s in states], dtype=np.float64)
    return pos, yaw


def footprint_at(footprint: Polygon, pos: Sequence[float], yaw: float) -> Polygon:
    return footprint.rotate(yaw).translate(pos)


def lateral_half_width(footprint: Polygon, yaw: float, normal: Sequence[float]) -> float:
    """给定朝向下，轮廓在法向上的半宽（支撑函数）。"""
    n = np.asarray(normal, dtype=np.float64)
    return float((footprint.rotate(yaw).vertices @ n).max())


def path_half_width(footprint: Polygon, pos: np.ndarray, yaw: np.ndarray,
                    stride: int = 5) -> float:
    """路径上各点"轮廓到自身中心线的最大法向半宽"，用于确定场地安全边。"""
    best = 0.0
    for i in range(0, pos.shape[0], stride):
        tangent = pos[min(i + 1, pos.shape[0] - 1)] - pos[max(i - 1, 0)]
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        best = max(best, lateral_half_width(footprint, yaw[i], normal),
                   lateral_half_width(footprint, yaw[i], -normal))
    return best


def min_path_clearance(footprint: Polygon, obstacles: Sequence[Obstacle],
                       pos: np.ndarray, yaw: np.ndarray,
                       stride: int = 1, fp_verts: np.ndarray | None = None
                       ) -> tuple[float, dict]:
    """名义路径上的最小净距，并返回最紧的那一帧/障碍索引。"""
    idx = np.arange(0, pos.shape[0], stride)
    fv = (footprint_vertices_batch(footprint, pos, yaw) if fp_verts is None else fp_verts)[idx]
    best = float("inf")
    worst = {"frame": -1, "obstacle": -1}
    for j, obs in enumerate(obstacles):
        d = distances_to_obstacle(fv, obs.polygon())
        k = int(np.argmin(d))
        if d[k] < best:
            best = float(d[k])
            worst = {"frame": int(idx[k]), "obstacle": j}
    return best, worst


def clearance_to_gate_width(clearance: float, half_width: float) -> float:
    """双侧门洞宽度 = 2·(轮廓半宽 + 净距)：把净距档位翻译成通道宽度。"""
    return 2.0 * (half_width + clearance)


def fit_arena(footprint: Polygon, commands, clearance: float) -> dict:
    """选一个朝向并定场地尺寸，使路径、障碍与边界都留足安全边。

    在 :data:`YAW_CANDIDATES` 里挑"场地面积最小"的朝向；场地最小 8×8 m，
    尺寸按 0.5 m 取整。
    """
    arr = np.asarray(commands, dtype=np.float64)
    best: dict | None = None
    for yaw0 in YAW_CANDIDATES:
        pos, yaw = nominal_path(arr, (0.0, 0.0), yaw0)
        x0, x1 = float(pos[:, 0].min()), float(pos[:, 0].max())
        y0, y1 = float(pos[:, 1].min()), float(pos[:, 1].max())
        half = path_half_width(footprint, pos, yaw)
        edge = half + clearance + SAFETY_EDGE
        w = _round_up(max(MIN_ARENA, (x1 - x0) + 2 * edge))
        h = _round_up(max(MIN_ARENA, (y1 - y0) + 2 * edge))
        cand = {"yaw0": yaw0, "arena_w": w, "arena_h": h, "area": w * h,
                "path_bbox": (x1 - x0, y1 - y0), "edge": edge,
                "half_width": half, "pos0": pos, "yaw_path": yaw}
        if best is None or cand["area"] < best["area"] - 1e-9:
            best = cand
    if best is None:
        raise ObstacleError("没有可行的场地朝向")
    pos = best.pop("pos0")
    yaw = best.pop("yaw_path")
    # 把路径包围盒平移到场地中心
    cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
    dx, dy = best["arena_w"] / 2.0 - cx, best["arena_h"] / 2.0 - cy
    best["shift"] = (dx, dy)
    best["pos"] = pos + np.array([dx, dy])
    best["yaw"] = yaw
    best["start"] = (float(pos[0, 0] + dx), float(pos[0, 1] + dy))
    return best


def _round_up(value: float) -> float:
    return float(math.ceil(value / ARENA_STEP) * ARENA_STEP)


# ---------------------------------------------------------------- 设计器
def _pick_anchors(pos: np.ndarray, arc: np.ndarray, total: float, n_obstacles: int,
                  n_candidates: int = 60, window: int = 5) -> np.ndarray:
    """在路径上挑"局部最直"的锚点：弯得厉害的地方横向净距没有意义。

    候选点均匀取在 12%–88% 弧长处，按 ±window 帧内的转向角绝对值排序，取前
    ``n_obstacles`` 个，再按弧长顺序返回。
    """
    n = pos.shape[0]
    cand = np.linspace(0.12, 0.88, n_candidates) * total
    scored = []
    for s in cand:
        i = min(max(int(np.searchsorted(arc, s)), window), n - window - 1)
        t0 = pos[i] - pos[i - window]
        t1 = pos[i + window] - pos[i]
        n0, n1 = np.linalg.norm(t0), np.linalg.norm(t1)
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(t0, t1) / (n0 * n1), -1.0, 1.0))
        scored.append((abs(math.acos(cosang)), float(s)))
    if len(scored) < n_obstacles:
        raise ObstacleError("可用直路段不足")
    scored.sort()
    picked = sorted(s for _ang, s in scored[:n_obstacles])
    return np.array(picked)


def _push_until_safe(footprint: Polygon, obs: Obstacle, pos: np.ndarray, yaw: np.ndarray,
                     target: float = MIN_DESIGN_CLEARANCE, max_push: float = 2.5,
                     step: float = 0.05, fp_verts: np.ndarray | None = None
                     ) -> tuple[Obstacle, float, bool]:
    """沿"当前最近路径帧 → 障碍"的方向外推，直到整条路径的最小净距达标。

    只看单一方向会失败（U 形路径上外推可能靠近另一段），所以每次都用**当前**最近帧
    重新定方向；超过 ``max_push`` 仍不达标就放弃该障碍。
    """
    fv = fp_verts if fp_verts is not None else footprint_vertices_batch(footprint, pos, yaw)
    pushed = 0.0
    while pushed <= max_push:
        dist = distances_to_obstacle(fv, obs.polygon())
        k = int(np.argmin(dist))
        d = float(dist[k])
        worst = {"frame": int(k), "obstacle": 0}
        if d >= target:
            return obs, pushed, True
        i = max(worst["frame"], 0)
        v = np.array([obs.x, obs.y]) - pos[i]
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            v, n = np.array([0.0, 1.0]), 1.0
        obs = Obstacle(x=obs.x + v[0] / n * step, y=obs.y + v[1] / n * step,
                       width=obs.width, depth=obs.depth, yaw=obs.yaw)
        pushed += step
    return obs, pushed, False


def design_case(footprint: Polygon, commands, clearance_target: float,
                rng: np.random.Generator, n_obstacles: int = 6) -> dict:
    """为一条命令序列生成地图：障碍贴着名义路径排布，净距按目标档位设定。

    沿路径按弧长取 ``n_obstacles`` 个锚点（跳过首尾 12%），在锚点处量出轮廓的法向
    半宽 ``h``，把障碍放在距中心线 ``h + clearance_target`` 处；双侧"门"按
    ``GATE_FRACTION`` 比例成对出现。放完后**逐帧复核**整条路径的最小净距，
    低于 :data:`MIN_DESIGN_CLEARANCE` 就把该障碍沿法向持续外推直到安全。
    """
    if clearance_target <= 0:
        raise ObstacleError("clearance_target 必须 > 0")
    fit = fit_arena(footprint, commands, clearance_target)
    pos, yaw = fit["pos"], fit["yaw"]
    n = pos.shape[0]
    if n < 20:
        raise ObstacleError("路径太短，无法布障")
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1.0:
        raise ObstacleError(f"名义路径总长仅 {total:.2f} m，不适合近障设计")

    anchors = _pick_anchors(pos, arc, total, n_obstacles)
    n_gate = max(1, int(round(n_obstacles * GATE_FRACTION / 2)))
    obstacles: list[Obstacle] = []
    for k, s in enumerate(anchors):
        i = min(max(int(np.searchsorted(arc, s)), 1), n - 2)
        tangent = pos[i + 1] - pos[i - 1]
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])      # 左法向
        heading = math.atan2(tangent[1], tangent[0])
        h = lateral_half_width(footprint, yaw[i], normal)
        w, d = OBSTACLE_SIZES[int(rng.integers(len(OBSTACLE_SIZES)))]
        sides = (1.0, -1.0) if k < n_gate else ((1.0,) if k % 2 == 0 else (-1.0,))
        for side in sides:
            off = h + clearance_target
            obstacles.append(Obstacle(x=float(pos[i][0] + side * normal[0] * off),
                                      y=float(pos[i][1] + side * normal[1] * off),
                                      width=float(w), depth=float(d), yaw=float(heading)))

    # 逐个障碍"沿最近路径法向外推"直到安全；推不动（超过预算）就撤掉该障碍。
    kept: list[Obstacle] = []
    pushed_total = 0.0
    dropped = 0
    fp_verts = footprint_vertices_batch(footprint, pos, yaw)
    for obs in obstacles:
        safe_obs, pushed, ok = _push_until_safe(footprint, obs, pos, yaw,
                                                target=clearance_target, max_push=3.0,
                                                fp_verts=fp_verts)
        pushed_total += pushed
        if ok:
            kept.append(safe_obs)
        else:
            dropped += 1
    if len(kept) < 3:
        raise ObstacleError(f"可用障碍不足（{len(kept)} 个），请增大净距档位或换朝向")
    obstacles = kept

    achieved, worst = min_path_clearance(footprint, obstacles, pos, yaw, stride=1)
    aw, ah = fit["arena_w"], fit["arena_h"]
    in_arena = all(0.0 <= o.x <= aw and 0.0 <= o.y <= ah for o in obstacles)
    path_in_arena = bool((pos[:, 0] >= 0).all() and (pos[:, 0] <= aw).all()
                         and (pos[:, 1] >= 0).all() and (pos[:, 1] <= ah).all())
    # 边界到障碍/路径的余量（墙计入地图但不参与近障判据）
    wall_margin = min(min(o.x, aw - o.x, o.y, ah - o.y) for o in obstacles)
    return {
        "arena": {"width_m": aw, "height_m": ah, "boundary": "walls"},
        "start_pose": {"px": fit["start"][0], "py": fit["start"][1], "yaw": fit["yaw0"]},
        "clearance_target_m": float(clearance_target),
        "clearance_achieved_m": float(achieved),
        "gate_width_m": clearance_to_gate_width(clearance_target, fit["half_width"]),
        "obstacles": [o.as_dict() for o in obstacles],
        "n_obstacles": len(obstacles),
        "n_pushed": int(round(pushed_total / 0.05)), "push_total_m": float(pushed_total),
        "n_dropped": int(dropped),
        "path_length_m": total,
        "path_bbox_m": [float(v) for v in fit["path_bbox"]],
        "path_in_arena": path_in_arena,
        "obstacles_in_arena": in_arena,
        "wall_margin_m": float(wall_margin),
        "tightest": worst,
    }
