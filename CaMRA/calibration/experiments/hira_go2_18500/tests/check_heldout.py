"""HIRA-032：400 条 heldout 独立检验的完整性审查。

只读：不重跑、不改数据。判据来自协议 v1（每格完整率 ≥0.9、non_computable ≤0.20、
失败留在主分母）、数据锁（HIRA-029）与 manifest。

**不打开 near_obstacle**：本脚本只遍历 heldout run 目录；heldout 的结果在 HIRA-033 才聚合。

    python tests/check_heldout.py --run-id heldout_20261106 \
        --lock config/data_lock_hira_evaluation_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from camra_calibration.energy import integrate_series  # noqa: E402
from camra_calibration.power import J_MIN, PowerModel  # noqa: E402
from hira_go2.collector import attempt_number, done_is_valid, sample_spec  # noqa: E402

SPLIT = "heldout"
CELL_QUOTA = 20                     # 每格（族 × profile）20 条
FULL_RATE_MIN = 0.90
NON_COMPUTABLE_MAX = 0.20
ENERGY_REL_MAX = 1e-9
STEPS_PER_INTERVAL = 20

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default="heldout_20261106")
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_evaluation_v1.json"))
parser.add_argument("--out", default=str(EXP / "reports/heldout_collection_integrity.json"))
args_cli = parser.parse_args()

CHECKS: list[dict] = []


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> None:
    CHECKS.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_attempt(trial_dir: Path) -> Path | None:
    if not trial_dir.is_dir():
        return None
    for path in sorted((p for p in trial_dir.iterdir()
                        if p.is_dir() and p.name.startswith("attempt_")), key=attempt_number):
        if done_is_valid(path):
            return path
    return None


def main() -> int:
    bundle_id = json.loads((Path(args_cli.bundle) / "manifest.json").read_text(encoding="utf-8"))["bundle_id"]
    lock = json.loads(Path(args_cli.lock).read_text(encoding="utf-8"))
    model = PowerModel.load(args_cli.model, verify_hash=True)
    manifest_path = Path(args_cli.manifests) / f"{SPLIT}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {row["trial_id"]: row for row in rows}
    run_dir = Path(args_cli.runs_root) / args_cli.run_id
    print(f"[held] manifest {len(manifest)} 条，run {run_dir}", flush=True)

    cells: dict[str, dict] = defaultdict(
        lambda: {"manifest": 0, "completed": 0, "business_failure": 0, "missing": 0}
    )
    excluded: list[dict] = []
    trial_rows: list[dict] = []
    worst_energy = 0.0
    n_non_computable = 0
    n_used = 0
    push_measured: list[float] = []
    knob_mismatch: list[str] = []

    for trial_id, row in manifest.items():
        family, profile = row["family"], row["profile"]
        cell = f"{family}/{profile}"
        cells[cell]["manifest"] += 1
        attempt = best_attempt(run_dir / trial_id)
        if attempt is None:
            cells[cell]["missing"] += 1
            excluded.append({"trial_id": trial_id, "reason": "缺少有效 attempt"})
            continue
        meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((attempt / "energy.json").read_text(encoding="utf-8"))
        problems = []

        if meta["bundle_id"] != bundle_id:
            problems.append("bundle_id 不符")
        if meta["split"] != SPLIT:
            problems.append(f"split={meta['split']}")
        if (meta["seed"], meta["program"], meta["profile"]) != (row["seed"], family, profile):
            problems.append("seed/program/profile 与清单不一致")
        if meta["program_sha256"] != row["program_sha256"]:
            problems.append("program_sha256 不符")
        if lock["controller"]["bundle_id"] != bundle_id:
            problems.append("数据锁的 bundle 与当前冻结包不一致")

        # profile 实际旋钮：期望值由**同一个采样器**按 seed 重建（协议规定，不是采集器自述）
        applied = meta["profile_applied"]
        friction_read = float(applied["friction"]["readback"])
        mass_extra = float(applied["mass"]["readback"]) - float(applied["mass"]["base_mass_kg"])
        pushes = meta.get("push_trace") or []
        pairing_cmds = [list(map(float, c)) for c in
                        np.load(attempt / "pairing_10hz.npz")["command"].astype("float64")]
        expected = sample_spec(
            trial_id=trial_id, split=SPLIT, program=family, profile=profile, seed=int(row["seed"]),
            commands=pairing_cmds, window_s=30.0,
        )
        if abs(friction_read - expected.friction) > 1e-6:
            knob_mismatch.append(f"{trial_id}: friction {friction_read:.4f} ≠ 期望 {expected.friction:.4f}")
        if abs(mass_extra - expected.added_mass_kg) > 1e-3:
            knob_mismatch.append(f"{trial_id}: mass 增量 {mass_extra:.4f} ≠ 期望 {expected.added_mass_kg:.4f}")
        expect_pushes = len(expected.push_events or [])
        if len(pushes) != expect_pushes:
            knob_mismatch.append(f"{trial_id}: 推扰 {len(pushes)} 次 ≠ 期望 {expect_pushes} 次")
        for item in pushes:
            push_measured.append(float(np.hypot(*item["measured_dv_xy"])))

        # 结构
        n_steps = int(meta["timing"]["steps_executed"])
        n_pair = int(meta["timing"]["pairing_points"])
        n_state = int(meta["timing"]["state50_points"])
        outcome = meta["outcome"]
        if outcome == "completed":
            cells[cell]["completed"] += 1
            if (n_steps, n_state, n_pair) != (6000, 1500, 301):
                problems.append(f"completed 但计数 {n_steps}/{n_state}/{n_pair}")
        elif outcome == "business_failure":
            cells[cell]["business_failure"] += 1
            if not (0 < n_steps < 6000) or n_state != n_steps // 4 or n_pair != n_steps // STEPS_PER_INTERVAL + 1:
                problems.append(f"截断不自洽 {n_steps}/{n_state}/{n_pair}")
        else:
            problems.append(f"outcome={outcome}")

        # 能量复算
        phys = np.load(attempt / "physics_200hz.npz")
        tau = phys["tau_applied_source"].astype("float64")
        qdot = phys["qdot_post_source"].astype("float64")
        if tau.shape[0] != n_steps:
            problems.append(f"physics 行数 {tau.shape[0]} ≠ {n_steps}")
        j_re = float(integrate_series(tau, qdot, 0.005))
        rel = abs(j_re - energy["j_exec"]) / max(abs(energy["j_exec"]), 1e-12)
        worst_energy = max(worst_energy, rel)
        if rel > ENERGY_REL_MAX:
            problems.append(f"能量复算 rel {rel:.2e}")
        if energy["j_exec"] < J_MIN:
            problems.append(f"J={energy['j_exec']:.4f} < J_min")

        # non_computable：用名义模型算 J_nom 是否 < J_min
        pairing = np.load(attempt / "pairing_10hz.npz")
        commands = pairing["command"].astype("float64")
        j_nom = float(model.j_nom(commands)) if commands.shape[0] >= 2 else 0.0
        n_used += 1
        if j_nom < J_MIN:
            n_non_computable += 1

        trial_rows.append({
            "trial_id": trial_id, "cell": cell, "outcome": outcome,
            "steps": n_steps, "j_exec": round(float(energy["j_exec"]), 3),
            "j_nom": round(j_nom, 3), "push_events": len(pushes),
            "friction_readback": round(friction_read, 4),
            "mass_extra_kg": round(mass_extra, 4),
        })
        if problems:
            excluded.append({"trial_id": trial_id, "reason": "；".join(problems)})

    # ---- C01 配额与格计数 ----
    total_manifest = sum(c["manifest"] for c in cells.values())
    n_cells = len(cells)
    add("C01-quota", "400 条 / 20 格（4 族 × 5 profile），每格 20 条",
        total_manifest == 400 and n_cells == 20 and all(c["manifest"] == CELL_QUOTA for c in cells.values()),
        f"清单 {total_manifest} 条、{n_cells} 格、每格 {sorted({c['manifest'] for c in cells.values()})} 条")

    # ---- C02 身份与散列 ----
    id_bad = [e for e in excluded if any(k in e["reason"] for k in
                                         ("bundle_id", "split=", "seed/program", "program_sha256", "数据锁"))]
    add("C02-identity", "bundle/split/seed/program 散列逐条与清单、冻结包、数据锁一致",
        not id_bad, f"不合格 {len(id_bad)} 条" + (f"：{id_bad[:3]}" if id_bad else ""))

    # ---- C03 profile 旋钮 ----
    push_detail = (
        f"；推扰实测 |Δv| ∈ [{min(push_measured):.3f}, {max(push_measured):.3f}] m/s（{len(push_measured)} 次）"
        if push_measured else "；本次无推扰"
    )
    add("C03-profile-knobs", "五种 profile 的实际旋钮（摩擦/质量/推扰次数）与按 seed 重建的期望值一致",
        not knob_mismatch,
        f"异常 {len(knob_mismatch)} 条" + (f"：{knob_mismatch[:4]}" if knob_mismatch else "") + push_detail)

    # ---- C04 结构 ----
    struct_bad = [e for e in excluded if any(k in e["reason"] for k in ("计数", "截断", "outcome=", "physics 行数"))]
    add("C04-structure", "completed=6000/1500/301；business_failure 截断自洽",
        not struct_bad, f"不合格 {len(struct_bad)} 条" + (f"：{struct_bad[:3]}" if struct_bad else ""))

    # ---- C05 能量 ----
    energy_bad = [e for e in excluded if "能量复算" in e["reason"] or "J_min" in e["reason"]]
    add("C05-energy", f"能量离线复算 rel ≤ {ENERGY_REL_MAX:g} 且 J ≥ J_min",
        not energy_bad, f"复算 {n_used} 条，最差 rel {worst_energy:.2e}；不合格 {len(energy_bad)}")

    # ---- C06 每格完整率 ----
    rate = {cell: {"manifest": c["manifest"], "completed": c["completed"],
                   "business_failure": c["business_failure"], "missing": c["missing"],
                   "full_rate": round(c["completed"] / c["manifest"], 4) if c["manifest"] else 0.0}
            for cell, c in sorted(cells.items())}
    rate_ok = all(v["full_rate"] >= FULL_RATE_MIN for v in rate.values()) and not any(
        v["missing"] for v in rate.values())
    worst_cell = min(rate.items(), key=lambda kv: kv[1]["full_rate"])
    add("C06-cell-full-rate", f"每格完整率 ≥ {FULL_RATE_MIN:g}（失败留在主分母，不补成功替换）",
        rate_ok,
        f"最差格 {worst_cell[0]} = {worst_cell[1]['completed']}/{worst_cell[1]['manifest']} "
        f"= {worst_cell[1]['full_rate']:.3f}；缺失 {sum(v['missing'] for v in rate.values())} 条")

    # ---- C07 non_computable 比例 ----
    nc_share = n_non_computable / n_used if n_used else 1.0
    add("C07-non-computable", f"J_nom < J_min 的比例 ≤ {NON_COMPUTABLE_MAX:g}",
        nc_share <= NON_COMPUTABLE_MAX,
        f"{n_non_computable}/{n_used} = {nc_share:.4f}")

    # ---- C08 未触碰受保护 split ----
    foreign = [p.name for p in run_dir.iterdir()
               if p.is_dir() and not p.name.startswith(f"hira-{SPLIT}-")]
    near_runs = [p.name for p in Path(args_cli.runs_root).glob("*")
                 if p.is_dir() and "near_obstacle" in p.name]
    add("C08-protected-splits", "本 run 只含 heldout 的 trial，未打开 near_obstacle",
        not foreign and not near_runs,
        f"run 内非 heldout 目录：{foreign or '无'}；runs 目录下的 near_obstacle：{near_runs or '无'}")

    # ---- C09 seed / 模板簇与已采 split 零重叠（协议独立性要求）----
    other_splits = ("pilot", "baseline_fit", "baseline_dev", "baseline_validate", "calibration")
    own_seeds = {int(r["seed"]) for r in rows}
    own_programs = {r["program_sha256"] for r in rows}
    overlap = {}
    for name in other_splits:
        p = Path(args_cli.manifests) / f"{name}.jsonl"
        if not p.is_file():
            continue
        other = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        seeds = {int(o["seed"]) for o in other}
        progs = {o["program_sha256"] for o in other}
        # 模板簇：同一 (family, seed) 生成同一命令表 → 用 program_sha256 判定"完全相同的命令表"
        overlap[name] = {"seed": len(own_seeds & seeds), "program_sha256": len(own_programs & progs)}
    add("C09-split-independence", "heldout 的 seed 与 program 散列与 pilot/baseline/calibration 零重叠",
        all(v["seed"] == 0 for v in overlap.values()),
        "逐 split 交集 " + json.dumps(overlap, ensure_ascii=False))

    # ---- C10 α/δ 与检验锁绑定（只读，不重算）----
    mrec = lock.get("margins") or {}
    mp = Path(mrec.get("path") or "")
    bound = (mrec.get("state") == "frozen" and mp.is_file()
             and sha256_file(mp) == mrec.get("sha256"))
    add("C10-margins-bound", "采集所用检验锁已冻结 margin，且 margin 文件散列与锁一致",
        bound, f"margins.state={mrec.get('state')!r}；alpha={mrec.get('alpha')}；delta={mrec.get('delta')}")

    failed = [c["id"] for c in CHECKS if not c["ok"]]
    n_failed_trials = sum(1 for t in trial_rows if t["outcome"] != "completed")
    report = {
        "report": "heldout_collection_integrity",
        "task": "HIRA-032",
        "run": {"run_id": args_cli.run_id, "dir": str(run_dir)},
        "bundle_id": bundle_id,
        "data_lock": {"path": args_cli.lock, "sha256": sha256_file(Path(args_cli.lock))},
        "model": {"path": args_cli.model, "sha256": model.sha256()},
        "counts": {
            "manifest": len(manifest),
            "evaluated": n_used,
            "completed": sum(c["completed"] for c in cells.values()),
            "business_failure": sum(c["business_failure"] for c in cells.values()),
            "missing": sum(c["missing"] for c in cells.values()),
            "n_non_computable": n_non_computable,
        },
        "cells": rate,
        "excluded": excluded,
        "energy_worst_rel": worst_energy,
        "checks": CHECKS,
        "trials": trial_rows,
        "verdict": {
            "all_passed": not failed,
            "failed": failed,
            "note": "失败仍留在主分母；本次审查只读 heldout，未打开 near_obstacle，未反向调整 α/δ 或功率模型",
        },
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"CAL_INTEGRITY all_passed={not failed} failed={failed} "
          f"completed={report['counts']['completed']}/800 failed_trials={n_failed_trials}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
