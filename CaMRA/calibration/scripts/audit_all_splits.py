#!/usr/bin/env python3
"""CAL-029：全数据分片独立审计。

刻意**不复用**采集侧的在线函数：能量用一段最简单的 numpy 重算
（逐关节 max(τ·q̇,0)·dt 再求和），与 ``energy.json`` 里的设备端累加对照；
配额、seed 块、散列、步数、配对点数全部独立核对。

审计对象：baseline_{fit,dev,validate} + pilot + calibration + heldout + near_obstacle。
"""
from __future__ import annotations

import argparse
import hashlib
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

TOL = 1e-5           # 协议 tolerances.energy_recompute_rel
SIM_DT = 0.005
EXPECTED_STEPS = 6000
EXPECTED_PAIRS = 301


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def energy_fresh(tau: np.ndarray, qdot: np.ndarray, dt: float = SIM_DT) -> tuple[float, np.ndarray]:
    """独立实现：逐子步逐关节取正功功率，再对时间求和。"""
    power = tau * qdot
    power = np.where(power > 0.0, power, 0.0)
    per_joint = power.sum(axis=0) * dt          # (J,)
    return float(per_joint.sum()), per_joint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-029 全数据分片独立审计")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--splits", default="baseline_fit,baseline_dev,baseline_validate,pilot,"
                                        "calibration,heldout,near_obstacle")
    args = ap.parse_args(argv)

    import torch

    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    report: dict = {"splits": {}, "problems": []}
    seed_blocks: dict[str, set[int]] = {}

    for split in [s for s in args.splits.split(",") if s]:
        man_p = Path(args.manifest_dir) / f"{split}.jsonl"
        if not man_p.is_file():
            report["problems"].append(f"{split}: 清单缺失 {man_p}")
            continue
        specs = read_jsonl(man_p)
        base = Path(args.out_root) / split
        seen, missing, duplicate = set(), [], []
        n_complete = n_term = 0
        worst_rel = 0.0
        worst_tid = None
        steps_bad, pairs_bad, energy_bad = [], [], []
        for spec in specs:
            tid = spec["trial_id"]
            if tid in seen:
                duplicate.append(tid)
            seen.add(tid)
            d = find_done_dir(base, tid)
            if d is None:
                missing.append(tid)
                continue
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            outcome = meta.get("outcome")
            if outcome == "completed":
                n_complete += 1
            else:
                n_term += 1
            pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
            n_pairs = int(pair["t"].shape[0])
            if outcome == "completed" and n_pairs != EXPECTED_PAIRS:
                pairs_bad.append((tid, n_pairs))
            phys = torch.load(d / "physics_200hz.npz", map_location="cpu")
            tau = np.asarray(phys["tau"], dtype=np.float64)
            qdot = np.asarray(phys["qdot"], dtype=np.float64)
            steps = int(tau.shape[0])
            if outcome == "completed" and steps != EXPECTED_STEPS:
                steps_bad.append((tid, steps))
            if meta.get("steps_executed") != steps:
                steps_bad.append((tid, f"meta {meta.get('steps_executed')} vs arrays {steps}"))
            j_fresh, per_joint = energy_fresh(tau, qdot)
            j_rec = float(json.loads((d / "energy.json").read_text(encoding="utf-8"))["j_exec"])
            rel = abs(j_fresh - j_rec) / max(abs(j_rec), 1e-9)
            if rel > worst_rel:
                worst_rel, worst_tid = rel, tid
            if rel > TOL:
                energy_bad.append((tid, round(rel, 12)))
        seed_blocks[split] = {int(s["seed"]) for s in specs}
        report["splits"][split] = {
            "n_manifest": len(specs), "n_missing": len(missing),
            "n_duplicate": len(duplicate),
            "n_complete": n_complete, "n_other_outcome": n_term,
            "manifest_sha256": sha256_file(man_p),
            "worst_energy_rel_err": worst_rel, "worst_energy_trial": worst_tid,
            "n_energy_violations": len(energy_bad),
            "energy_violations": energy_bad[:5],
            "n_steps_violations": len(steps_bad), "steps_violations": steps_bad[:5],
            "n_pairs_violations": len(pairs_bad), "pairs_violations": pairs_bad[:5],
            "missing": missing[:5], "duplicate": duplicate[:5],
        }
        if missing:
            report["problems"].append(f"{split}: {len(missing)} 条缺失")
        if duplicate:
            report["problems"].append(f"{split}: {len(duplicate)} 条重复")
        if energy_bad:
            report["problems"].append(f"{split}: {len(energy_bad)} 条能量复算超差")
        if steps_bad or pairs_bad:
            report["problems"].append(f"{split}: 步数/配对点异常")

    # ---- split 互斥 ----
    names = sorted(seed_blocks)
    overlaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = seed_blocks[a] & seed_blocks[b]
            if inter:
                overlaps.append({"a": a, "b": b, "n": len(inter), "sample": sorted(inter)[:5]})
    if overlaps:
        report["problems"].append(f"{len(overlaps)} 组 split 的 seed 块相交")
    report["seed_overlaps"] = overlaps
    report["seed_ranges"] = {k: [min(v), max(v), len(v)] for k, v in seed_blocks.items() if v}

    # ---- 清单散列与锁一致 ----
    lock_lists = lock["trial_lists"]["files"]
    list_mismatch = []
    for split, info in report["splits"].items():
        if split not in lock_lists:
            continue
        if info["manifest_sha256"] != lock_lists[split]["sha256"]:
            list_mismatch.append(split)
    if list_mismatch:
        report["problems"].append(f"清单散列与锁不符: {list_mismatch}")
    report["list_hash_mismatch"] = list_mismatch
    report["lock_bound"] = {
        "protocol": lock["protocol"]["sha256"][:16],
        "power_model": lock["power_model"]["content_sha256"][:16],
        "footprint": lock["footprint"]["content_sha256"][:16],
        "margins": (lock.get("margins") or {}).get("content_sha256", "")[:16],
    }
    report["ok"] = not report["problems"]

    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"], "problems": report["problems"],
        "splits": {k: {"n": v["n_manifest"], "missing": v["n_missing"],
                       "worst_rel": float(f"{v['worst_energy_rel_err']:.3e}")}
                   for k, v in report["splits"].items()},
        "seed_ranges": report["seed_ranges"], "overlaps": len(overlaps),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
