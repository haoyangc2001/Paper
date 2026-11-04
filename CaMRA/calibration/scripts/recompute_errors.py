#!/usr/bin/env python3
"""CAL-016：从采集产物独立复算 ε_c / ε_d / 50 Hz 敏感性，并给失败分类与分母。

不启动仿真。逐 trial 读 pairing_10hz / state_50hz / physics_200hz / energy / meta：
  * 校验 10 Hz 网格（起点 0、等间隔、单调、有限）与位置形状；
  * ε_d = max_t ‖p_exec − p_nom‖；ε_d(50 Hz) 用名义轨迹线性插值到 50 Hz 时刻后取最大；
  * ε_c = (J_exec − J_nom)/J_nom（仅完整窗口且 J_nom ≥ J_min），j_nom 缺失记 non_computable 原因；
  * 失败分类 business_failure / data_invalid，缺分片、NaN、乱时钟、重复 trial、来源错误各有明确 reason；
  * 写 <trial>/errors.json 与 <split>/errors_report.json（含 all/complete/failed/invalid 分母与排除原因）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_calibration.pairing import (  # noqa: E402
    Issue, aggregate, compute_trial_errors, detect_duplicates,
)


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception:  # 损坏的分片按 data_invalid 处理
        return "CORRUPT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--j-min", type=float, default=1.0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in
                (Path(args.manifest_dir) / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    if args.limit:
        manifest = manifest[:args.limit]
    base = Path(args.out_root) / args.split

    rows, problems = [], []
    for tr in manifest:
        tid = tr["trial_id"]
        d = base / tid
        if not d.is_dir():
            rows.append(compute_trial_errors(
                trial_id=tid, outcome="missing", t_pair=[], exec_pos=[], nom_pos=[],
                j_exec=None, j_nom=None, source=str(d)).__class__(
                trial_id=tid, outcome="missing", complete=False, official=False,
                issues=[Issue("missing_shard", "trial 目录不存在", str(d))]))
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        pair = _load(d / "pairing_10hz.npz")
        state = _load(d / "state_50hz.npz")
        energy = json.loads((d / "energy.json").read_text(encoding="utf-8"))
        outcome = meta.get("outcome", "unknown")

        if pair == "CORRUPT" or pair is None or state == "CORRUPT" or state is None:
            rows.append(compute_trial_errors(
                trial_id=tid, outcome=outcome, t_pair=[], exec_pos=[], nom_pos=[],
                j_exec=None, j_nom=None, source=str(d)).__class__(
                trial_id=tid, outcome=outcome, complete=(outcome == "completed"),
                official=False,
                issues=[Issue("missing_shard", "pairing_10hz/state_50hz 缺失或损坏", str(d))]))
            continue

        j_exec = energy.get("j_exec")
        j_nom = energy.get("j_nom")
        r = compute_trial_errors(
            trial_id=tid, outcome=outcome,
            t_pair=pair["t"].numpy(), exec_pos=pair["exec_pos"].numpy(),
            nom_pos=pair["nominal_pos"].numpy(),
            j_exec=j_exec, j_nom=j_nom, j_min=args.j_min,
            cmd=pair["cmd"].numpy() if "cmd" in pair else None,
            t50=(np.arange(state["root_pos"].shape[0]) + 1) * 0.02 if "root_pos" in state else None,
            exec50_pos=state["root_pos"][:, :2].numpy() if "root_pos" in state else None,
            steps_executed=meta.get("steps_executed"),
            expected_steps=meta.get("steps_expected", 6000),
            source=str(d),
        )
        # 额外：50 Hz 只对完整窗口有意义；前缀诊断时补一条说明
        if not r.complete and r.epsilon_d is not None:
            r.notes.append("prefix 诊断：仅前 %d 个配对点" % int(np.asarray(pair["t"]).size))
        (d / "errors.json").write_text(
            json.dumps(r.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(r)

    # 重复 trial 检测（同一 trial_id 出现多个 attempt 目录）
    dup = detect_duplicates([p.name.split("__a")[0] for p in base.iterdir() if p.is_dir()])
    if dup:
        problems.append(f"重复 attempt 目录: {dup}")

    report = aggregate(rows, expected_total=len(manifest))
    report["split"] = args.split
    report["duplicate_dirs"] = dup
    report["infra_notes"] = problems
    out = Path(args.report) if args.report else base / "errors_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("denominators", "exclusions", "duplicate_dirs")},
                     ensure_ascii=False, indent=1))
    print("epsilon_d:", json.dumps(report["epsilon_d"]))
    print("epsilon_d_50hz:", json.dumps(report["epsilon_d_50hz"]))
    print("epsilon_c:", json.dumps({k: v for k, v in report["epsilon_c"].items() if k != "stats"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
