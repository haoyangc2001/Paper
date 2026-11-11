#!/usr/bin/env python
"""学习候选的指令分布统计（NAV-035 的替代性证据）。

为什么有这个东西：计划要求 G3 包含「NAV-034 实际学习候选复查」，而那一项是按 NAV-060
口径跑真物理仿真。当前候选是 17 分钟试跑产物、即将重训，所以物理复查顺延；但「新司机
说话会不会太小声」这个实质问题可以在**名义侧、用已经落盘的轨迹**回答——不需要重训、
不需要 Isaac、不需要改那份已过门禁的脚本。

做法：用与 NAV-060 pilot **逐行同一定义**的 `command_stats`（同一组常数、同一个保持时长
判据、同一个低速判据），只是把指令来源从规则控制器换成学习候选。

必须写明的边界：
  * 这是**名义环境**的指令分布，不是物理闭环的指令分布；名义模型假设指令被理想跟踪。
  * 命名侧回合长度不一致（学习候选约 69 步、规则控制器 300 步），所以比的是分布形状
    与占比，不是绝对时间长度。
  * 它不能替代物理复查：指令能发出来 ≠ 底层能执行。它只能回答「会不会大量落进已知死区」
    这个必要条件。

Run::

    <frozen python> CaMRA/navigation/scripts/check_candidate_commands.py \
        --run-id nav034_smoke_02 --json CaMRA/navigation/reports/candidate_command_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(NAV_ROOT), "calibration", "src"))

from camra_navigation.algorithm import storage as S  # noqa: E402

PILOT = os.path.join(NAV_ROOT, "reports", "calibration_applicability_pilot.json")
DEADBAND = os.path.join(NAV_ROOT, "reports", "yawrate_deadband.json")

# --- 与 check_calibration_applicability.py 完全相同的判据常数 ---
NAV_DT = 0.1
EPISODE_SECONDS = 30.0
LOW_SPEED_VX = 0.3
LOW_SPEED_WZ = 0.3

#: NAV-060 实测的角速率死区（|wz| 分档的执行比例），用于把占比翻译成"能执行多少"
WZ_BUCKETS = ((0.0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def command_stats(cmds: np.ndarray) -> dict[str, Any]:
    """逐行照抄 check_calibration_applicability.command_stats 的定义（去掉域检查）。"""
    arr = np.asarray(cmds, dtype=np.float64)
    if arr.size == 0:
        return {}
    du = np.abs(np.diff(arr, axis=0)).max(axis=1) if len(arr) > 1 else np.array([0.0])
    keep = du <= 0.02
    runs, cur = [], 1
    for k in keep:
        cur += 1 if k else 0
        if not k:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    sign_flip = {}
    for i, name in enumerate(("vx", "vy", "wz")):
        v = arr[:, i]
        nz = np.sign(v[v != 0.0])
        sign_flip[name] = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    low = float(np.mean((np.abs(arr[:, 0]) <= LOW_SPEED_VX) & (np.abs(arr[:, 2]) <= LOW_SPEED_WZ)))
    return {
        "n_points": int(len(arr)),
        "mean_abs": {k: float(np.mean(np.abs(arr[:, i])))
                     for i, k in enumerate(("vx", "vy", "wz"))},
        "max_abs": {k: float(np.max(np.abs(arr[:, i])))
                    for i, k in enumerate(("vx", "vy", "wz"))},
        "change_rate_mean": float(np.mean(du)),
        "change_rate_max": float(np.max(du)),
        "switch_per_s": {k: v / EPISODE_SECONDS for k, v in sign_flip.items()},
        "switch_count": sign_flip,
        "hold_mean_s": float(np.mean(runs)) * NAV_DT,
        "hold_max_s": float(np.max(runs)) * NAV_DT,
        "low_speed_share": low,
    }


def deadband_profile(arr: np.ndarray) -> dict[str, Any]:
    """把 |wz| 落到 NAV-060 实测的死区分档里，看有多少步落在"基本不执行"的区间。"""
    wz = np.abs(np.asarray(arr[:, 2], dtype=np.float64))
    out: dict[str, Any] = {"n_points": int(wz.size), "buckets": []}
    for lo, hi in WZ_BUCKETS:
        mask = (wz >= lo) & (wz < hi) if lo > 0 else (wz < hi)
        out["buckets"].append({"wz_range": [lo, hi], "share": float(mask.mean())})
    vx = np.abs(np.asarray(arr[:, 0], dtype=np.float64))
    out["vx_below_0p2_share"] = float((vx < 0.2).mean())
    out["vx_below_0p31_share"] = float((vx < 0.31).mean())
    return out


def mean_over(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r["stats"][key] for r in rows if r.get("stats") and key in r["stats"]]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="/pub/data/caohy/work/camra_navigation")
    ap.add_argument("--run-id", default="nav034_smoke_02")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    store = S.RolloutStore(args.run_root, args.run_id, obs_dim=41)
    mine: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    for idx in range(store.n_chunks):
        a = store.load_chunk(idx)
        cmds = np.asarray(a["action_applied"], dtype=np.float64)
        pooled.append(cmds)
        mine.append({
            "chunk": idx, "level": store.entries[idx].get("level"),
            "event": store.entries[idx].get("event"),
            "stats": command_stats(cmds),
            "deadband": deadband_profile(cmds),
        })
    allc = np.concatenate(pooled, axis=0) if pooled else np.zeros((0, 3))

    # 规则控制器（NAV-060 pilot）的对照：同一套 command_stats，取 new 侧全部回合
    ref_rows: list[dict[str, Any]] = []
    if os.path.exists(PILOT):
        pilot = json.load(open(PILOT, encoding="utf-8"))
        for ep in pilot.get("episodes", []):
            if ep.get("kind") == "new" and ep.get("command_stats"):
                ref_rows.append({"level": ep.get("level"), "stats": ep["command_stats"]})

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        out: dict[str, Any] = {
            "n_episodes": len(rows),
            "mean_abs": {k: float(np.mean([r["stats"]["mean_abs"][k] for r in rows]))
                         for k in ("vx", "vy", "wz")},
            "change_rate_mean": mean_over(rows, "change_rate_mean"),
            "change_rate_max": max(r["stats"]["change_rate_max"] for r in rows),
            "hold_mean_s": mean_over(rows, "hold_mean_s"),
            "hold_max_s": max(r["stats"]["hold_max_s"] for r in rows),
            "low_speed_share": mean_over(rows, "low_speed_share"),
        }
        return out

    mine_summary = summarize(mine)
    mine_summary["mean_abs"] = {k: float(np.mean(np.abs(allc[:, i])))
                                for i, k in enumerate(("vx", "vy", "wz"))}
    mine_summary["pooled_deadband"] = deadband_profile(allc)
    ref_summary = summarize(ref_rows)

    report = {
        "task": "NAV-035",
        "part": "candidate_command_stats",
        "scope": "名义环境、指令分布；不是物理闭环复查，不能替代 NAV-060 同口径检查",
        "candidate": {"run_id": args.run_id, "n_chunks": store.n_chunks,
                      "n_samples": int(allc.shape[0]), **mine_summary},
        "reference_rule_controller": {"source": os.path.relpath(PILOT, os.path.dirname(NAV_ROOT)),
                                      "kind": "new（闭环规则控制器）", **ref_summary},
        "definitions": {
            "same_yardstick": "command_stats 的定义、常数（NAV_DT=0.1、LOW_SPEED_VX/WZ=0.3、"
                              "保持判据 du<=0.02）与 check_calibration_applicability.py 逐行一致",
            "low_speed_share": "同时满足 |vx|<=0.3 且 |wz|<=0.3 的步占比",
            "hold_mean_s": "指令变化 <=0.02 的连续段平均长度 × 0.1 s",
            "deadband_buckets": "按 NAV-060 实测的角速率死区分档统计步占比",
        },
        "caveats": [
            "名义回合约 69 步、规则控制器 300 步：比的是占比与形状，不是绝对时长",
            "名义模型假设指令被理想跟踪，因此这里看不到执行的衰减",
        ],
    }
    if os.path.exists(DEADBAND):
        report["deadband_reference"] = json.load(open(DEADBAND, encoding="utf-8"))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({
        "candidate": {k: report["candidate"][k] for k in
                      ("n_samples", "n_episodes", "mean_abs", "change_rate_mean",
                       "hold_mean_s", "low_speed_share")},
        "pooled_deadband": report["candidate"]["pooled_deadband"],
        "rule": {k: ref_summary.get(k) for k in
                 ("n_episodes", "mean_abs", "change_rate_mean", "hold_mean_s",
                  "low_speed_share")},
    }, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
