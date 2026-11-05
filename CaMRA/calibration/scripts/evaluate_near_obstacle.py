#!/usr/bin/env python3
"""CAL-029：近障试验的覆盖与几何评估。

只读 ``near_obstacle`` split；α/δ、功率模型、足迹、地图全部从冻结产物加载。
输出：成本/位置/联合覆盖（总体与按净距档位分层）、几何碰撞与最小净距、
场地边界余量、按 cell/档位/族的分组，以及可用性判定。
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

from camra_calibration.coverage import (       # noqa: E402
    HeldoutRow, cost_covered, coverage_verdict, denominators, direction_of,
    per_cell, position_covered, rate, rate_with_ci, subgroup,
)
from camra_calibration.margins import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, Margins  # noqa: E402
from camra_calibration.near_obstacle import (  # noqa: E402
    by_tier, clearance_verdict, footprint_from_json, trial_geometry, yaw_from_quat,
)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-029 近障评估")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="near_obstacle")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--margins", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--report", required=True)
    ap.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args(argv)

    if args.split != "near_obstacle":
        raise SystemExit(f"REFUSED: 本脚本只评 near_obstacle，得到 {args.split!r}")

    cal = Path(__file__).resolve().parent.parent
    proto = Protocol.load()
    model = PowerModel.load(Path(args.model) if args.model else cal / "config" / "power_model_v1.json")
    margins = Margins.load(Path(args.margins) if args.margins else cal / "config" / "margins_v1.json")
    fp = footprint_from_json(Path(args.footprint) if args.footprint
                             else cal / "config" / "footprint_v1.json")
    maps_p = Path(args.maps) if args.maps else cal / "config" / "near_obstacle_maps_v1.json"
    maps = {c["trial_id"]: c for c in json.loads(maps_p.read_text(encoding="utf-8"))["cases"]}

    specs = read_jsonl(Path(args.manifest_dir) / f"{args.split}.jsonl")
    rows: list[HeldoutRow] = []
    geo = []
    missing = []
    for spec in specs:
        tid = spec["trial_id"]
        d = find_done_dir(Path(args.out_root) / args.split, tid)
        if d is None:
            missing.append(tid)
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((d / "energy.json").read_text(encoding="utf-8"))
        pair = load_npz(d / "pairing_10hz.npz")
        state = load_npz(d / "state_50hz.npz")
        if pair is None or state is None:
            missing.append(tid)
            continue
        j_exec = float(energy.get("j_exec", float("nan")))
        cmd = np.asarray(pair["cmd"], dtype=np.float64)
        j_nom = model.j_nom(cmd) if np.isfinite(j_exec) else None
        t50 = np.arange(int(state["root_pos"].shape[0]), dtype=np.float64) * STATE_DT
        exec50 = np.asarray(state["root_pos"], dtype=np.float64)[:, :2]
        err = compute_trial_errors(
            trial_id=tid, outcome=str(meta.get("outcome", "missing")),
            t_pair=np.asarray(pair["t"], dtype=np.float64),
            exec_pos=np.asarray(pair["exec_pos"], dtype=np.float64),
            nom_pos=np.asarray(pair["nominal_pos"], dtype=np.float64),
            j_exec=j_exec, j_nom=j_nom, steps_executed=meta.get("steps_executed"),
            t50=t50, exec50_pos=exec50, cmd=cmd,
            expected_steps=EXPECTED_STEPS, source=str(d))
        case = maps[tid]
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
        # 几何：用 50 Hz 记录的位姿 + 冻结足迹对冻结障碍物
        yaw = yaw_from_quat(np.asarray(state["root_quat"], dtype=np.float64))
        geo.append(trial_geometry(tid, rows[-1].cell, case["clearance_target_m"],
                                  case, fp, exec50, yaw))

    expected = len(specs)
    den = denominators(rows, expected)
    den["missing"] = len(missing)
    alpha, delta = margins.alpha, margins.delta
    official = [r for r in rows if r.official]
    cells = [r.cell for r in official]
    c_flags = [cost_covered(r.eps_c, alpha) for r in official]
    d_flags = [position_covered(r.eps_d, delta) for r in official]
    joint = [bool(a) and bool(b) for a, b in zip(c_flags, d_flags)
             if a is not None and b is not None]
    overall = {
        "cost": rate_with_ci(c_flags, cells, args.replicates, args.seed),
        "position": rate_with_ci(d_flags, cells, args.replicates, args.seed),
        "joint": rate_with_ci(joint, cells, args.replicates, args.seed),
    }
    per = 10 if expected == 200 else max(1, expected // max(1, len({r.cell for r in rows})))
    tier_stats = by_tier(geo)
    # 档位分层的覆盖：把几何行与覆盖行按 trial_id 对齐
    by_id = {r.trial_id: r for r in official}
    tier_cover: dict[str, dict] = {}
    for g in geo:
        key = f"{g.tier_m:.2f}"
        row = by_id.get(g.trial_id)
        if row is None:
            continue
        t = tier_cover.setdefault(key, {"n": 0, "cost": 0, "position": 0, "joint": 0})
        t["n"] += 1
        cc, pc = cost_covered(row.eps_c, alpha), position_covered(row.eps_d, delta)
        t["cost"] += bool(cc)
        t["position"] += bool(pc)
        t["joint"] += bool(cc) and bool(pc)
    for v in tier_cover.values():
        v["cost_rate"] = v["cost"] / v["n"]
        v["position_rate"] = v["position"] / v["n"]
        v["joint_rate"] = v["joint"] / v["n"]

    verdict_geom = clearance_verdict(geo, delta, tier_stats)
    verdict_cov = coverage_verdict(den, overall, per_cell(rows, alpha, delta, per))
    report = {
        "split": args.split, "n_manifest": expected, "n_observed": len(rows),
        "alpha": alpha, "delta": delta,
        "margins_sha256": margins.sha256(),
        "model_sha256": model.sha256(),
        "protocol": {"version": proto.version, "sha256": proto.sha256},
        "maps_file": str(maps_p),
        "denominators": den,
        "outcomes": dict(Counter(r.outcome for r in rows)),
        "overall": overall,
        "per_cell": per_cell(rows, alpha, delta, per),
        "groups": {
            "by_profile": subgroup(rows, lambda r: r.profile, alpha, delta),
            "by_family": subgroup(rows, lambda r: r.family, alpha, delta),
            "by_low_speed": subgroup(rows, lambda r: "low_speed" if r.low_speed else "high_speed",
                                     alpha, delta),
            "by_direction": subgroup(rows, lambda r: r.direction, alpha, delta),
        },
        "geometry": {
            "by_tier": tier_stats,
            "tier_coverage": tier_cover,
            "verdict": verdict_geom,
            "min_clearance_m": float(min(g.min_clearance_m for g in geo)) if geo else None,
            "n_collision": int(sum(1 for g in geo if not g.collision_free)),
            "n_outside_arena": int(sum(1 for g in geo if not g.inside_arena)),
        },
        "verdict": verdict_cov,
        "notes": [
            "碰撞判据：冻结足迹多边形与冻结障碍多边形几何重叠（等于阈值算覆盖，重叠即 d=0）",
            "场地边界墙未建模，越界用几何余量检查代替并如实上报",
            "覆盖判定沿用 CAL-027 的含等号口径；联合单独报告",
            "本报告不用于反向调整 α/δ、功率模型或地图（协议 near_obstacle.fail_action）",
        ],
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "n": len(rows), "official": len(official), "missing": len(missing),
        "cost": round(overall["cost"]["rate"], 4),
        "position": round(overall["position"]["rate"], 4),
        "joint": round(overall["joint"]["rate"], 4),
        "n_collision": report["geometry"]["n_collision"],
        "min_clearance_m": round(report["geometry"]["min_clearance_m"], 4),
        "n_outside_arena": report["geometry"]["n_outside_arena"],
        "geom_status": verdict_geom["status"], "cov_status": verdict_cov["status"],
        "tiers": verdict_geom["tier_counts"], "tiers_balanced": verdict_geom["tiers_balanced"],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
