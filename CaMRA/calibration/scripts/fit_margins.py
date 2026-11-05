#!/usr/bin/env python3
"""CAL-025：用正式校准集计算并冻结 α / δ。

严格按协议 ``statistics`` 的口径：

* ε_c = (J_exec − J_nom) / J_nom（只有完整且 J_nom ≥ J_min 的记录才可算）；
* ε_d = 配对偏差在 30 s 窗口上的最大值（m）；
* ``alpha = max(0, Q0.90(ε_c))``、``delta = Q0.90(ε_d)``，分位一律 nearest-rank；
* 区间用 bootstrap（整条轨迹为最小单位，10000 次、95%），另按 cell 簇重采样一份；
* 只读 ``calibration`` split，**不打开** heldout / near_obstacle，也不引用任何旧值。

产物：``config/margins_v1.json``（+ .sha256）与一份完整报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
for _p in (str(CAL / "src"), str(CAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camra_calibration.margins import (     # noqa: E402
    BOOTSTRAP_REPLICATES, MIN_COMPLETE_SHARE_PER_CELL, QUANTILE_LEVEL,
    Margins, build_margins, calibration_verdict, group_stats, nearest_rank,
)
from camra_calibration.pairing import (    # noqa: E402
    aggregate, compute_trial_errors, detect_duplicates,
)
from camra_calibration.power import PowerModel      # noqa: E402
from camra_calibration.protocol import Protocol     # noqa: E402

PAIR_DT = 0.1
STATE_DT = 0.02
EXPECTED_STEPS = 6000
#: 低速层切分：命令在低速包络内的时间占比 ≥ 该值即归入"低速层"
LOW_SPEED_SPLIT = 0.5


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def find_done_dir(base: Path, trial_id: str) -> Path | None:
    """优先正式目录，否则取 attempt 最大的 ``__aN``（都必须有 DONE）。"""
    d = base / trial_id
    if (d / "DONE").is_file():
        return d
    best: tuple[int, Path] | None = None
    for p in base.glob(f"{trial_id}__a*"):
        if not (p / "DONE").is_file():
            continue
        try:
            n = int(p.name.split("__a")[-1])
        except ValueError:
            n = 0
        if best is None or n > best[0]:
            best = (n, p)
    return None if best is None else best[1]


def load_npz(path: Path):
    if not path.is_file():
        return None
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def collect_rows(manifest_dir: Path, out_root: Path, split: str,
                 model: PowerModel) -> tuple[list, list[dict], list[str]]:
    """逐条复算 ε_c / ε_d；返回 (TrialErrors, 缺件清单, 重复 id)。"""
    specs = read_jsonl(Path(manifest_dir) / f"{split}.jsonl")
    dups = detect_duplicates([s["trial_id"] for s in specs])
    rows, missing = [], []
    for spec in specs:
        tid = spec["trial_id"]
        d = find_done_dir(Path(out_root) / split, tid)
        if d is None:
            missing.append(tid)
            continue
        meta_p, energy_p = d / "meta.json", d / "energy.json"
        pair = load_npz(d / "pairing_10hz.npz")
        state = load_npz(d / "state_50hz.npz")
        if not meta_p.is_file() or not energy_p.is_file() or pair is None:
            missing.append(tid)
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        energy = json.loads(energy_p.read_text(encoding="utf-8"))
        j_exec = float(energy.get("j_exec", float("nan")))
        cmd = np.asarray(pair["cmd"], dtype=np.float64)
        # J_nom 一律用冻结模型从命令重算：不接受产物里可能过期的字段
        j_nom = model.j_nom(cmd) if np.isfinite(j_exec) else None

        t_pair = np.asarray(pair["t"], dtype=np.float64)
        exec_pos = np.asarray(pair["exec_pos"], dtype=np.float64)
        nom_pos = np.asarray(pair["nominal_pos"], dtype=np.float64)
        t50 = exec50 = None
        if state is not None and "root_pos" in state:
            n50 = int(state["root_pos"].shape[0])
            t50 = np.arange(n50, dtype=np.float64) * STATE_DT
            exec50 = np.asarray(state["root_pos"], dtype=np.float64)[:, :2]

        rows.append(compute_trial_errors(
            trial_id=tid, outcome=str(meta.get("outcome", "missing")),
            t_pair=t_pair, exec_pos=exec_pos, nom_pos=nom_pos,
            j_exec=j_exec, j_nom=j_nom,
            steps_executed=meta.get("steps_executed"),
            t50=t50, exec50_pos=exec50, cmd=cmd,
            expected_steps=EXPECTED_STEPS, source=str(d)))
    return rows, missing, sorted(dups)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-025 计算并冻结 α/δ")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--model", default=None, help="默认 config/power_model_v1.json")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", default=None, help="默认 config/margins_v1.json")
    ap.add_argument("--report", required=True)
    ap.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args(argv)

    if args.split != "calibration":
        raise SystemExit(f"REFUSED: 只允许用 calibration 计算裕度，得到 {args.split!r}")

    cal_dir = Path(__file__).resolve().parent.parent
    proto_p = Path(args.protocol) if args.protocol else cal_dir / "config" / "calibration_protocol_v1.0.2.json"
    proto = Protocol.load(proto_p)
    model_p = Path(args.model) if args.model else cal_dir / "config" / "power_model_v1.json"
    model = PowerModel.load(model_p)          # 默认校验内容散列

    rows, missing, dups = collect_rows(Path(args.manifest_dir), Path(args.out_root),
                                       args.split, model)
    agg = aggregate(rows, expected_total=len(rows) + len(missing))

    # ---- 分组与分母 ----
    per_cell: dict[str, dict] = {}
    by_cell = defaultdict(list)
    for r in rows:
        fam, prof = r.trial_id.split("-")[1], r.trial_id.split("-")[2]
        by_cell[f"{fam}|{prof}"].append(r)
    expected_per_cell = 40                     # 20 格 × 40（协议配额）
    for cell, items in sorted(by_cell.items()):
        complete = [r for r in items if r.complete and not r.issues]
        per_cell[cell] = {
            "n_observed": len(items),
            "n_expected": expected_per_cell,
            "n_complete": len(complete),
            "complete_share": len(complete) / expected_per_cell,
            "n_non_computable": sum(1 for r in items if r.non_computable),
            "eps_c": group_stats([r.epsilon_c for r in complete]),
            "eps_d": group_stats([r.epsilon_d for r in complete]),
        }

    def split_by(key_fn) -> dict:
        groups = defaultdict(list)
        for r in rows:
            if r.complete and r.epsilon_c is not None:
                groups[key_fn(r)].append(r)
        return {k: {"n": len(v),
                    "eps_c": group_stats([r.epsilon_c for r in v]),
                    "eps_d": group_stats([r.epsilon_d for r in v])}
                for k, v in sorted(groups.items())}

    official = [r for r in rows if r.official]
    official_c = [r for r in official if r.epsilon_c is not None]
    non_computable = [r for r in rows if r.non_computable]
    n_all = len(rows) + len(missing)

    stats = {
        "denominators": {
            "expected": n_all,
            "observed": len(rows),
            "missing": len(missing),
            "complete": len(official),
            "business_failed": agg["denominators"]["business_failed"],
            "data_invalid": agg["denominators"]["data_invalid"],
            "non_computable": len(non_computable),
            "eps_c_usable": len(official_c),
        },
        "shares": {
            "complete_share_overall": len(official) / n_all if n_all else 0.0,
            "non_computable_share": len(non_computable) / n_all if n_all else 0.0,
        },
        "eps_c_overall": group_stats([r.epsilon_c for r in official_c]),
        "eps_d_overall": group_stats([r.epsilon_d for r in official]),
        "eps_d_50hz": group_stats([r.epsilon_d_50hz for r in official
                                   if r.epsilon_d_50hz is not None]),
        "per_cell": per_cell,
        "by_profile": split_by(lambda r: r.trial_id.split("-")[2]),
        "by_family": split_by(lambda r: r.trial_id.split("-")[1]),
        "by_low_speed": split_by(lambda r: "low_speed" if (r.low_speed_share or 0.0) >= LOW_SPEED_SPLIT
                                 else "high_speed"),
        "low_speed_split": LOW_SPEED_SPLIT,
        "exclusions": agg["exclusions"],
        "duplicates": dups,
        "missing_trial_ids": missing[:20],
    }

    if not official_c:
        raise SystemExit("没有可用样本（ε_c 全不可算），无法形成裕度")

    # ---- 簇重采样：同一 cell 共用同一命令模板 ----
    cells_c = [r.trial_id.rsplit("-", 1)[0] for r in official_c]
    cells_d = [r.trial_id.rsplit("-", 1)[0] for r in official]

    data = {
        "split": args.split,
        "manifest": f"{args.manifest_dir}/{args.split}.jsonl",
        "n_expected": n_all, "n_complete": len(official),
        "protocol_version": proto.version, "protocol_sha256": proto.sha256,
        "power_model_sha256": model.sha256(),
        "power_model_file_sha256": _sha256_file(model_p),
        "seeds": {"min": min(int(r.trial_id[-4:]) for r in rows) if rows else None},
        "note": "α/δ 只来自 calibration split；heldout 与 near_obstacle 未被读取",
    }
    applicability = {
        "command_domain": "与 data_lock 绑定的 30 s 命令程序域（|vx|≤2.0, |vy|≤1.0, |wz|≤2.0）",
        "j_min": 1.0,
        "requires_complete_window": True,
        "alpha_use": "成本预算 J_budget = (1+alpha)·J_nom",
        "delta_use": "几何危险集扩张 obstacle ⊕ footprint ⊕ disc(delta)，δ 只膨胀一次",
        "not_valid_for": ["J_nom < 1.0 J（non_computable）", "未完成的窗口", "校准命令域之外"],
    }
    margins, detail = build_margins(
        eps_c=[r.epsilon_c for r in official_c],
        eps_d=[r.epsilon_d for r in official],
        stats=stats, data=data, applicability=applicability,
        version="1.0", replicates=args.replicates, seed=args.seed,
        clusters_c=cells_c, clusters_d=cells_d,
        notes=("ε_c 的负超支保留原值参与分位，α 只在形成裕度时取 max(0,·)",
               "区间为整条轨迹重采样的 bootstrap 分位（nearest-rank 口径）"))

    verdict = calibration_verdict(per_cell, stats["shares"]["non_computable_share"],
                                  MIN_COMPLETE_SHARE_PER_CELL)
    detail["ties"] = {
        "n_at_or_above_alpha": int(sum(1 for r in official_c
                                       if r.epsilon_c is not None
                                       and abs(r.epsilon_c - margins.alpha) < 1e-12)),
        "n_at_or_above_delta": int(sum(1 for r in official
                                       if r.epsilon_d is not None
                                       and abs(r.epsilon_d - margins.delta) < 1e-12)),
        "rule": "等于阈值视为覆盖（协议 statistics.quantile_method）",
    }

    out_p = Path(args.out) if args.out else cal_dir / "config" / "margins_v1.json"
    saved, side = margins.save(out_p)
    report = {
        "version": margins.version, "sha256": margins.sha256(),
        "out": str(saved), "sidecar": str(side),
        "stats": stats, "detail": detail, "verdict": verdict,
        "margins": margins.to_dict(),
        "recomputable": {
            "alpha": margins.alpha,
            "alpha_from_q90": max(0.0, margins.q90_eps_c),
            "delta": margins.delta,
            "delta_from_q90": margins.q90_eps_d,
            "ok": (abs(margins.alpha - max(0.0, margins.q90_eps_c)) < 1e-12
                   and abs(margins.delta - margins.q90_eps_d) < 1e-12),
        },
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_expected": n_all, "n_complete": len(official),
        "missing": len(missing), "non_computable": len(non_computable),
        "q90_eps_c": margins.q90_eps_c, "alpha": margins.alpha,
        "alpha_ci": margins.alpha_interval,
        "q90_eps_d": margins.q90_eps_d, "delta": margins.delta,
        "delta_ci": margins.delta_interval,
        "verdict": verdict["verdict"],
        "cells_below_share": list(verdict["cells_below_share"]),
        "sha256": margins.sha256(),
    }, ensure_ascii=False, indent=1))
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
