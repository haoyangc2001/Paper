"""近障试验的几何检查（CAL-029）。

对照"冻结足迹 + 记录位姿"与"冻结地图障碍物/场地边界"，给出：

* 每一步的**最小几何净距**（足迹多边形到障碍多边形的最近距离，相交为 0）；
* 是否发生**几何重叠**（碰撞的下界证据）；
* 到**场地边界**的最小余量（墙未建模，用几何检查代替）；
* 按**目标净距档位**分层统计。

只依赖 camra_calibration 的纯几何模块，不碰仿真。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Polygon
from .obstacles import Obstacle, distances_to_obstacle, footprint_vertices_batch


@dataclass(frozen=True)
class TrialGeometry:
    trial_id: str
    cell: str
    tier_m: float
    n_steps: int
    min_clearance_m: float
    collision_free: bool
    min_wall_margin_m: float
    inside_arena: bool


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 偏航角（rad）。"""
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"四元数形状应为 (N,4)，得到 {q.shape}")
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def footprint_from_json(path) -> Polygon:
    """从冻结 footprint JSON 取轮廓（只读，不再膨胀）。"""
    import json
    payload = json.loads(open(path, encoding="utf-8").read())
    return Polygon(np.asarray(payload["polygon"]["vertices"], dtype=np.float64))


def obstacles_from_case(case: dict) -> list[Obstacle]:
    return [Obstacle(x=float(o["x"]), y=float(o["y"]), width=float(o["width"]),
                     depth=float(o["depth"]), yaw=float(o.get("yaw", 0.0)))
            for o in case.get("obstacles", [])]


def trial_geometry(trial_id: str, cell: str, tier_m: float, case: dict,
                   footprint: Polygon, pos_xy: np.ndarray, yaw: np.ndarray) -> TrialGeometry:
    """单条 trial 的几何检查（pos/yaw 为 50 Hz 逐帧）。"""
    pos = np.asarray(pos_xy, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    if pos.shape[0] != yaw.shape[0]:
        raise ValueError(f"位姿与朝向帧数不一致：{pos.shape[0]} vs {yaw.shape[0]}")
    fv = footprint_vertices_batch(footprint, pos, yaw)     # (F,V,2)
    obs = obstacles_from_case(case)
    if obs:
        d = np.min(np.stack([distances_to_obstacle(fv, o.polygon()) for o in obs], axis=0), axis=0)
        d_min = float(d.min())
    else:                                                   # pragma: no cover - 地图不会空
        d_min = float("inf")
    arena_w = float(case["arena"]["width_m"])
    arena_h = float(case["arena"]["height_m"])
    xs, ys = fv[:, :, 0], fv[:, :, 1]
    wall = np.minimum(np.minimum(xs, arena_w - xs), np.minimum(ys, arena_h - ys))
    wall_min = float(wall.min())
    return TrialGeometry(
        trial_id=trial_id, cell=cell, tier_m=float(tier_m), n_steps=int(pos.shape[0]),
        min_clearance_m=d_min, collision_free=bool(d_min > 0.0),
        min_wall_margin_m=wall_min, inside_arena=bool(wall_min >= 0.0))


def by_tier(rows: Sequence[TrialGeometry]) -> dict:
    """按目标净距档位分层：每档的条数、碰撞数、最小/中位净距、越界数。"""
    groups: dict[float, list[TrialGeometry]] = defaultdict(list)
    for r in rows:
        groups[round(r.tier_m, 4)].append(r)
    out = {}
    for tier, items in sorted(groups.items()):
        d = np.asarray([r.min_clearance_m for r in items], dtype=np.float64)
        out[f"{tier:.2f}"] = {
            "n": len(items),
            "n_collision": int(sum(1 for r in items if not r.collision_free)),
            "min_clearance_m": float(d.min()),
            "median_clearance_m": float(np.median(d)),
            "max_clearance_m": float(d.max()),
            "n_outside_arena": int(sum(1 for r in items if not r.inside_arena)),
            "min_wall_margin_m": float(min(r.min_wall_margin_m for r in items)),
        }
    return out


def clearance_verdict(rows: Sequence[TrialGeometry], delta: float,
                      by_tier_stats: dict) -> dict:
    """近障可用性判定（只陈述可复算的事实，不改任何冻结值）。

    * 碰撞：先按几何重叠统计（0 次才叫"无碰撞"）；
    * δ 覆盖：目标净距 ≤ δ 的档位，理论上 δ 足以把危险集扩张到覆盖全部轨迹
      位置误差；这里报告"实际最小净距 ≥ 0"的条数与 δ 的关系，供结论引用；
    * 档位分层必须全部有样本且每档 50 条（预登记配额）。
    """
    n = len(rows)
    n_coll = sum(1 for r in rows if not r.collision_free)
    n_out = sum(1 for r in rows if not r.inside_arena)
    tier_counts = {k: v["n"] for k, v in by_tier_stats.items()}
    balanced = bool(tier_counts) and len(set(tier_counts.values())) == 1
    strict = [r for r in rows if r.tier_m <= delta]
    return {
        "n": n, "n_collision": n_coll, "n_outside_arena": n_out,
        "delta_m": float(delta),
        "n_tier_le_delta": len(strict),
        "n_collision_within_delta": sum(1 for r in strict if not r.collision_free),
        "tier_counts": tier_counts, "tiers_balanced": balanced,
        "status": ("clean" if n_coll == 0 and n_out == 0 else "collisions_present"),
    }
