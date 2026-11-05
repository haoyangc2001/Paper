#!/usr/bin/env python3
"""CAL-027：检验集的独立覆盖统计与可用性判定。

只读 ``heldout`` split；α/δ 与功率模型一律从冻结产物加载（**不重算、不调整**）。
输出：边际/联合覆盖、四套分母、20 格 + 低速层 + 方向子组、trial 级与 cell 簇级
95% 区间，以及按预登记规则的可用性判定。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
for _p in (str(CAL / "src"), str(CAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camra_calibration.coverage import (      # noqa: E402
    HeldoutRow, bootstrap_rate, cost_covered, coverage_verdict, denominators,
    direction_of, per_cell, position_covered, rate, rate_with_ci, subgroup,
)
from camra_calibration.margins import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, Margins  # noqa: E402
from camra_calibration.pairing import compute_trial_errors   # noqa: E402
from camra_calibration.power import PowerModel               # noqa: E402
from camra_calibration.protocol import Protocol              # noqa: E402

STATE_DT = 0.02
EXPECTED_STEPS = 6000
LOW_SPEED_SPLIT = 0.5


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def find_done_dir(base: Path, trial_id: str) -> Path | None:
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


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-027 检验集覆盖统计")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--margins", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--report", required=True)
    ap.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args(argv)

    if args.split != "heldout":
        raise SystemExit(f"REFUSED: 本脚本只评 heldout，得到 {args.split!r}")

    cal_dir = Path(__file__).resolve().parent.parent
    proto = Protocol.load(Path(args.protocol) if args.protocol
                          else cal_dir / "config" / "calibration_protocol_v1.0.2.json")
    model_p = Path(args.model) if args.model else cal_dir / "config" / "power_model_v1.json"
    model = PowerModel.load(model_p)
    mg_p = Path(args.margins) if args.margins else cal_dir / "config" / "margins_v1.json"
    margins = Margins.load(mg_p)
    alpha, delta = margins.alpha, margins.delta

    specs = read_jsonl(Path(args.manifest_dir) / f"{args.split}.jsonl")
    rows: list[HeldoutRow] = []
    missing: list[str] = []
    for spec in specs:
        tid = spec["trial_id"]
        d = find_done_dir(Path(args.out_root) / args.split, tid)
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
        j_nom = model.j_nom(cmd) if np.isfinite(j_exec) else None

        t50 = exec50 = None
        if state is not None and "root_pos" in state:
            n50 = int(state["root_pos"].shape[0])
            t50 = np.arange(n50, dtype=np.float64) * STATE_DT
            exec50 = np.asarray(state["root_pos"], dtype=np.float64)[:, :2]

        err = compute_trial_errors(
            trial_id=tid, outcome=str(meta.get("outcome", "missing")),
            t_pair=np.asarray(pair["t"], dtype=np.float64),
            exec_pos=np.asarray(pair["exec_pos"], dtype=np.float64),
            nom_pos=np.asarray(pair["nominal_pos"], dtype=np.float64),
            j_exec=j_exec, j_nom=j_nom, steps_executed=meta.get("steps_executed"),
            t50=t50, exec50_pos=exec50, cmd=cmd,
            expected_steps=EXPECTED_STEPS, source=str(d))
        rows.append(HeldoutRow(
            trial_id=tid, family=spec["family"], profile=spec["profile"],
            cell=f"{spec['family']}|{spec['profile']}",
            direction=direction_of(cmd),
            low_speed=bool((err.low_speed_share or 0.0) >= LOW_SPEED_SPLIT),
            outcome=str(meta.get("outcome", "missing")),
            complete=bool(err.complete), official=bool(err.official),
            non_computable=bool(err.non_computable),
            data_invalid=bool(err.issues) and err.complete,
            eps_c=err.epsilon_c, eps_d=err.epsilon_d))

    expected = len(specs)
    per = 20 if expected == 400 else max(1, expected // max(1, len({r.cell for r in rows})))
    den = denominators(rows, expected)
    den["missing"] = len(missing)
    den["missing_trial_ids"] = missing[:10]

    official = [r for r in rows if r.official]
    cells_flags = [r.cell for r in official]
    c_flags = [cost_covered(r.eps_c, alpha) for r in official]
    d_flags = [position_covered(r.eps_d, delta) for r in official]
    joint_pairs = [(a, b) for a, b in zip(c_flags, d_flags) if a is not None and b is not None]
    joint_flags = [bool(a) and bool(b) for a, b in joint_pairs]

    overall = {
        "cost": rate_with_ci(c_flags, cells_flags, args.replicates, args.seed),
        "position": rate_with_ci(d_flags, cells_flags, args.replicates, args.seed),
        "joint": rate_with_ci(joint_flags, cells_flags, args.replicates, args.seed),
    }
    cells = per_cell(rows, alpha, delta, per)
    groups = {
        "by_profile": subgroup(rows, lambda r: r.profile, alpha, delta),
        "by_family": subgroup(rows, lambda r: r.family, alpha, delta),
        "by_low_speed": subgroup(rows, lambda r: "low_speed" if r.low_speed else "high_speed",
                                 alpha, delta),
        "by_direction": subgroup(rows, lambda r: r.direction, alpha, delta),
    }
    verdict = coverage_verdict(den, overall, cells)

    # 未覆盖清单：便于定位是否某个条件系统性失败
    uncovered = [{"trial_id": r.trial_id, "cell": r.cell, "eps_c": r.eps_c, "eps_d": r.eps_d,
                  "cost_covered": cost_covered(r.eps_c, alpha),
                  "position_covered": position_covered(r.eps_d, delta)}
                 for r in official
                 if (cost_covered(r.eps_c, alpha) is False
                     or position_covered(r.eps_d, delta) is False)]

    report = {
        "split": args.split, "n_manifest": expected, "n_observed": len(rows),
        "alpha": alpha, "delta": delta,
        "margins": {"file": str(mg_p), "sha256": margins.sha256(),
                    "file_sha256": sha256_file(mg_p)},
        "power_model": {"file": str(model_p), "sha256": model.sha256()},
        "protocol": {"version": proto.version, "sha256": proto.sha256},
        "denominators": den,
        "outcomes": dict(Counter(r.outcome for r in rows)),
        "overall": overall,
        "per_cell": cells,
        "groups": groups,
        "uncovered": {"n": len(uncovered), "rows": uncovered[:50]},
        "verdict": verdict,
        "notes": [
            "覆盖判定：等于阈值视为覆盖（协议 statistics.quantile_method）",
            "联合覆盖单独报告，不用两项边际推联合",
            "区间以整条轨迹为最小重采样单位，另给 cell 整簇口径；不按 50 Hz 点计数",
            "单一固定 v2 控制器，不声称训练随机种子方差",
            "本报告不用于反向调整 α/δ 或功率模型（协议 statistical_acceptance.heldout.may_not）",
        ],
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_observed": len(rows), "n_official": len(official),
        "alpha": round(alpha, 5), "delta": round(delta, 5),
        "cost": {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in overall["cost"].items() if k in ("n", "n_covered", "rate")},
        "position": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in overall["position"].items() if k in ("n", "n_covered", "rate")},
        "joint": {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in overall["joint"].items() if k in ("n", "n_covered", "rate")},
        "joint_ci": [round(overall["joint"]["ci_trial"]["lo"], 4),
                     round(overall["joint"]["ci_trial"]["hi"], 4)],
        "verdict": verdict["status"],
        "below_target": verdict["below_target"],
        "uncovered": len(uncovered),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
