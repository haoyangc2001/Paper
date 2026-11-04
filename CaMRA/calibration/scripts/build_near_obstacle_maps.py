"""生成并冻结 200 条近障地图与命令绑定（CAL-023）。

只读输入：``trial_lists_v1.0.2/near_obstacle.jsonl``（200 条，seed 8000–8199）与
``config/footprint_v1.json``（冻结轮廓）。**不读取**任何执行误差/正式结果。

档位分配：20 格（4 族 × 5 profile）× 10 条，格内第 j 条取
``CLEARANCE_TIERS[j % 4]``，于是四档各 50 条。

用法：
    python scripts/build_near_obstacle_maps.py \
        --manifest-dir /pub/data/caohy/logs/camra/trial_lists_v1.0.2 \
        --out config/near_obstacle_maps_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.footprint import Footprint  # noqa: E402
from camra_calibration.obstacles import (  # noqa: E402
    CLEARANCE_TIERS, MIN_DESIGN_CLEARANCE, ObstacleError, design_case,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-023 近障地图冻结")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="调试用：只做前 N 条")
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    fp_p = Path(args.footprint) if args.footprint else cal / "config" / "footprint_v1.json"
    fp = Footprint.load(fp_p)
    list_p = Path(args.manifest_dir) / "near_obstacle.jsonl"
    specs = [json.loads(l) for l in list_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        specs = specs[:args.limit]

    per_cell = defaultdict(int)
    cases: list[dict] = []
    tiers_used: Counter = Counter()
    cell_order: list[str] = []          # 单元格首次出现顺序，用于档位轮转偏移
    for spec in specs:
        prog = build_program(spec["family"], spec["seed"],
                             duration_s=spec["duration_s"], command_dt=spec["command_dt"])
        if prog.sha256() != spec["program_sha256"]:
            raise SystemExit(f"FATAL: {spec['trial_id']} 的 program 散列与清单不一致")
        cell = f"{spec['family']}|{spec['profile']}"
        if cell not in per_cell:
            cell_order.append(cell)
        j = per_cell[cell]
        per_cell[cell] += 1
        # 档位按"格序 + 格内序号"轮转：每格 10 条本身是 3/3/2/2，
        # 但 20 格的起点错开后，四档总数正好各 50 条。
        tier = CLEARANCE_TIERS[(cell_order.index(cell) + j) % len(CLEARANCE_TIERS)]
        tiers_used[tier] += 1
        rng = np.random.default_rng(spec["seed"])     # 布局随机性只由 seed 决定
        try:
            design = design_case(fp.polygon, prog.commands, tier, rng)
        except ObstacleError as exc:
            raise SystemExit(f"FATAL: {spec['trial_id']} 设计失败: {exc}")
        if not (design["path_in_arena"] and design["obstacles_in_arena"]):
            raise SystemExit(f"FATAL: {spec['trial_id']} 越界 {design['path_in_arena']=} "
                             f"{design['obstacles_in_arena']=}")
        # 目标净距必须真正达到（允许 2 mm 数值裕度），否则该档位就是假的
        if design["clearance_achieved_m"] < tier - 2e-3:
            raise SystemExit(f"FATAL: {spec['trial_id']} 净距 {design['clearance_achieved_m']:.4f} "
                             f"< 目标 {tier}")
        cases.append({
            "trial_id": spec["trial_id"], "split": "near_obstacle",
            "family": spec["family"], "profile": spec["profile"], "seed": spec["seed"],
            "cell": cell, "cell_index": j,
            "program_sha256": spec["program_sha256"],
            "command_dt": spec["command_dt"], "duration_s": spec["duration_s"],
            **design,
        })

    totals = {
        "n_cases": len(cases),
        "clearance_tiers": {str(k): v for k, v in sorted(tiers_used.items())},
        "arena_w_min": min(c["arena"]["width_m"] for c in cases),
        "arena_w_max": max(c["arena"]["width_m"] for c in cases),
        "arena_h_min": min(c["arena"]["height_m"] for c in cases),
        "arena_h_max": max(c["arena"]["height_m"] for c in cases),
        "clearance_achieved_min": min(c["clearance_achieved_m"] for c in cases),
        "clearance_achieved_max": max(c["clearance_achieved_m"] for c in cases),
        "n_obstacles_hist": dict(Counter(c["n_obstacles"] for c in cases)),
        "wall_margin_min": min(c["wall_margin_m"] for c in cases),
        "n_pushed_total": sum(c["n_pushed"] for c in cases),
        "cells": len(per_cell),
    }
    payload = {
        "schema": "camra-near-obstacle-maps-v1",
        "cal_task": "CAL-023",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_version": "1.0.2",
        "design_rule": (
            "障碍贴着名义路径排布：沿弧长取锚点，在锚点处量出轮廓法向半宽 h，"
            "把矩形障碍放在距中心线 h+目标净距处；35% 的锚点做成双侧门；"
            "放完逐帧复核整条路径的最小净距，低于 0.02 m 就外推。"
            "场地 = 路径包围盒 + 2×(h+净距+1.0 m)，0.5 m 取整，下限 8×8。"),
        "clearance_tiers_m": list(CLEARANCE_TIERS),
        "footprint": {"path": str(fp_p), "sha256": fp.sha256(),
                      "min_width_m": fp.polygon.min_width()},
        "trial_list": {"path": str(list_p), "sha256": sha256_file(list_p),
                       "n_trials": len(specs)},
        "totals": totals,
        "cases": cases,
        "notes": [
            "本文件只做设计：障碍如何进入 Isaac 场景属 CAL-028 实现范围。",
            "净距按已冻结轮廓计算；δ 由 CAL-025 给出后，需检查 δ ≤ clearance_achieved_m。",
            "边界墙计入地图且距路径/障碍 ≥ 安全边，不参与近障判据。",
            "生成过程不读取任何执行误差或正式结果。",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    side = out.with_suffix(out.suffix + ".sha256")
    side.write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")

    print(f"cases {totals['n_cases']}  cells {totals['cells']}  tiers {totals['clearance_tiers']}")
    print(f"arena w {totals['arena_w_min']}–{totals['arena_w_max']}  "
          f"h {totals['arena_h_min']}–{totals['arena_h_max']}")
    print(f"clearance achieved {totals['clearance_achieved_min']:.3f}–"
          f"{totals['clearance_achieved_max']:.3f} m  wall margin ≥ {totals['wall_margin_min']:.2f}")
    print(f"obstacles per case {totals['n_obstacles_hist']}  pushed {totals['n_pushed_total']}")
    print(f"wrote {out} sha {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
