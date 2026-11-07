"""HIRA-035 前置：冻结 δ 在近障地图上的**区域**可行性（只读）。

第一版按"名义路径到障碍的最小距离"判可行，结论是连 v2 的 δ=2.6139 都不可行——这说明判据本身错了：
这些地图是按**名义足迹的 0.05–0.35 m 间隙**造的，任何 δ 超过间隙都会把名义路径圈进危险集，
那不是 δ 的问题，是判据把"路径间隙"当成了"δ 上限"。

正确的问题：δ 扩张后的危险集还剩多少**可通行区域**，起点/终点是否还在自由空间里。
本脚本算这个（栅格 0.02 m，numpy 向量化）：

* 危险区 = 障碍矩形 ⊕ disc(δ + r_f)，r_f 取 footprint 的**最小宽度一半**（0.3798 m，
  = 机器人最窄方向的半宽，比外接圆保守得少、又不会低估侧向占位）；同时给出外接圆 0.553 m 的对照；
* 指标：自由面积占比、起点姿态是否自由、自由区是否连成一片（从起点出发的连通面积）。

    python tests/assess_delta_feasibility.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--footprint", default=str(EXP / "config/footprint_hira_v1.json"))
parser.add_argument("--out", default=str(EXP / "reports/delta_feasibility.json"))
parser.add_argument("--step", type=float, default=0.02)
parser.add_argument("--deltas", default="0,0.5,1,2.6139,4,6,9.0322")
args = parser.parse_args()


def corners(ob):
    hw, hd = ob["width"] / 2.0, ob["depth"] / 2.0
    c, s = math.cos(ob["yaw"]), math.sin(ob["yaw"])
    return np.array([(ob["x"] + c * px - s * py, ob["y"] + s * px + c * py)
                     for px, py in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))])


def rect_distance_field(X, Y, quad):
    """点到带朝向矩形的欧氏距离（矩形内为 0），向量化。"""
    inside = np.ones(X.shape, dtype=bool)
    d = np.full(X.shape, np.inf)
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        d = np.minimum(d, np.hypot(wx - t * ex, wy - t * ey))
        # inside 判定（凸矩形，同向绕行）
        cross = ex * wy - ey * wx
        inside &= cross <= 0
    return np.where(inside, 0.0, d)


def analyse(case, delta, r_f, step):
    W, H = case["arena"]["width_m"], case["arena"]["height_m"]
    xs = np.arange(0.0, W + step / 2, step)
    ys = np.arange(0.0, H + step / 2, step)
    X, Y = np.meshgrid(xs, ys)
    danger = np.zeros(X.shape, dtype=bool)
    for ob in case["obstacles"]:
        danger |= rect_distance_field(X, Y, corners(ob)) <= (delta + r_f)
    free = ~danger
    free_share = float(free.mean())

    pose = case["start_pose"]
    ix = int(round(pose["px"] / step))
    iy = int(round(pose["py"] / step))
    start_free = bool(free[min(iy, free.shape[0] - 1), min(ix, free.shape[1] - 1)])

    # 从起点出发的连通自由区面积
    if start_free:
        seen = np.zeros(X.shape, dtype=bool)
        q = deque([(iy, ix)])
        seen[iy, ix] = True
        while q:
            cy, cx = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nY, nX = cy + dy, cx + dx
                if 0 <= nY < free.shape[0] and 0 <= nX < free.shape[1] and free[nY, nX] and not seen[nY, nX]:
                    seen[nY, nX] = True
                    q.append((nY, nX))
        reach_share = float(seen.mean())
    else:
        reach_share = 0.0

    # 名义路径终点
    program = build_program(case["family"], case["seed"], duration_s=case["duration_s"],
                            command_dt=case["command_dt"])
    cmds = [[float(v) for v in c] for c in program.commands]
    path = integrate(PlanarState(px=pose["px"], py=pose["py"], yaw=pose["yaw"]), cmds, 0.1)
    gx = int(round(np.clip(path[-1].px, 0, W) / step))
    gy = int(round(np.clip(path[-1].py, 0, H) / step))
    goal_free = bool(free[gy, gx])
    return {"delta_m": delta, "free_area_share": round(free_share, 5),
            "start_free": start_free, "goal_free": goal_free,
            "reachable_from_start_share": round(reach_share, 5)}


def main() -> int:
    maps = json.loads(Path(args.maps).read_text(encoding="utf-8"))
    margins = json.loads(Path(args.margins).read_text(encoding="utf-8"))
    fp = json.loads(Path(args.footprint).read_text(encoding="utf-8"))
    r_min = float(fp["polygon"]["min_width"]) / 2.0
    r_circ = float(fp["polygon"]["circumscribed_radius"])
    frozen = float(margins["delta"])
    deltas = sorted({float(v) for v in args.deltas.split(",") if v.strip()} | {frozen})
    print(f"[feas] r_f: min_width/2={r_min:.4f} m, circumscribed={r_circ:.4f} m；冻结 δ={frozen:.4f} m")

    # 每格抽 2 张地图做扫描（20 格 × 2 = 40 张），控制耗时
    sample = maps["cases"][::max(1, len(maps["cases"]) // 40)][:40]
    print(f"[feas] 采样 {len(sample)}/{len(maps['cases'])} 张地图，栅格 {args.step} m")

    sweep = {}
    for delta in deltas:
        rows = [analyse(c, delta, r_min, args.step) for c in sample]
        sweep[f"{delta:.4f}"] = {
            "delta_m": delta,
            "median_free_area_share": float(np.median([r["free_area_share"] for r in rows])),
            "min_free_area_share": min(r["free_area_share"] for r in rows),
            "max_free_area_share": max(r["free_area_share"] for r in rows),
            "start_free_share": round(sum(r["start_free"] for r in rows) / len(rows), 4),
            "goal_free_share": round(sum(r["goal_free"] for r in rows) / len(rows), 4),
            "median_reachable_share": float(np.median([r["reachable_from_start_share"] for r in rows])),
        }
        s = sweep[f"{delta:.4f}"]
        print(f"[feas] δ={delta:7.4f} → 自由面积中位 {s['median_free_area_share']:.4f}"
              f"（{s['min_free_area_share']:.4f}–{s['max_free_area_share']:.4f}），"
              f"起点自由 {s['start_free_share']:.2f}，终点自由 {s['goal_free_share']:.2f}", flush=True)

    frozen_row = sweep[f"{frozen:.4f}"]
    usable = (frozen_row["start_free_share"] == 1.0
              and frozen_row["goal_free_share"] == 1.0
              and frozen_row["median_free_area_share"] >= 0.10)
    report = {
        "report": "delta_feasibility",
        "task": "HIRA-035-precheck",
        "maps": {"path": str(args.maps), "n_cases": len(maps["cases"]), "n_sampled": len(sample),
                 "arena_m": [maps["cases"][0]["arena"]["width_m"], maps["cases"][0]["arena"]["height_m"]],
                 "gate_width_m_range": [round(min(c["gate_width_m"] for c in maps["cases"]), 4),
                                        round(max(c["gate_width_m"] for c in maps["cases"]), 4)],
                 "clearance_tiers_m": maps.get("clearance_tiers_m")},
        "footprint": {"min_width_half_m": round(r_min, 6), "circumscribed_radius_m": round(r_circ, 6),
                      "note": "危险区按 δ + min_width/2 计；外接圆 0.553 m 的版本会再保守 0.17 m"},
        "rule": "危险区 = obstacle ⊕ footprint ⊕ disc(δ)，δ 只施加一次",
        "criterion": "起点/终点仍在自由空间 且 中位自由面积占比 ≥ 0.10 视为几何可用",
        "sweep": sweep,
        "verdict": {
            "near_obstacle_geometrically_usable": bool(usable),
            "frozen_delta_m": frozen,
            "reference_v2_delta_m": 2.6139,
            "reason": ("冻结 δ 下起点/终点与自由面积仍可用"
                       if usable else
                       "冻结 δ 下自由面积或起终点被危险集吃掉，近障任务几何不可做"),
            "protocol_action": "记录并询问；不得通过改分位缩小 δ 来凑可行性",
            "first_attempt_note": ("第一版按'名义路径到障碍最小距离'判据，连 v2 的 2.6139 也判不可行，"
                                   "说明该判据把地图设计间隙（0.05–0.35 m）误当 δ 上限，已弃用；"
                                   "本报告改用区域面积判据"),
        },
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[feas] 结论：usable={usable} → {report['verdict']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
