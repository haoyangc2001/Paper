"""HIRA-034：200 条近障检查的完整性审查（只读，不重跑、不改数据）。

与 `check_heldout.py` 同一骨架，额外加了近障专属的三件事：

* **N03 障碍注入**：逐条核对 meta 里的 `obstacles` 段——非空、`n_obstacles` 与冻结地图
  一致、读回位姿偏差 ≤ 1e-3 m、起点与该 case 的 `start_pose` 一致。这是"障碍确实注入且
  逐 env 位置正确"的直接证据，没有它，这批数据和空场跑没区别。
* **N09 地图未改**：地图散列必须等于检验锁里登记的值（协议 no_redisign_rule）。
* 每格配额是 **10** 条（不是 heldout 的 20），完整率门槛仍 0.9。

判据来自协议 v1（每格完整率 ≥0.9、non_computable ≤0.20、失败留在主分母）与检验锁 v2。
本脚本**不评估覆盖率**——那是 HIRA-035；这里只回答"这 200 条数据能不能用"。

    python tests/check_near_obstacle.py --run-id near_obstacle_20261107 \\
        --lock config/data_lock_hira_evaluation_v2.json
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

SPLIT = "near_obstacle"
CELL_QUOTA = 10                     # 每格（族 × profile）10 条
FULL_RATE_MIN = 0.90
NON_COMPUTABLE_MAX = 0.20
ENERGY_REL_MAX = 1e-9
STEPS_PER_INTERVAL = 20
POSE_TOL_M = 1e-3                   # 障碍位姿读回偏差上限（float32 写入，实测 ~2e-7）

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default="near_obstacle_20261107")
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_evaluation_v2.json"))
parser.add_argument("--out", default=str(EXP / "reports/near_obstacle_collection_integrity.json"))
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
    maps_path = Path(args_cli.maps)
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    maps_sha = sha256_file(maps_path)
    case_by_trial = {str(c["trial_id"]): c for c in maps.get("cases", [])}

    manifest_path = Path(args_cli.manifests) / f"{SPLIT}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {row["trial_id"]: row for row in rows}
    run_dir = Path(args_cli.runs_root) / args_cli.run_id
    print(f"[near] manifest {len(manifest)} 条，run {run_dir}，maps sha={maps_sha[:16]}", flush=True)

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
    obstacle_bad: list[str] = []
    pose_devs: list[float] = []
    n_obstacles_total = 0

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
        problems: list[str] = []

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

        # ---- 障碍注入（本任务的核心证据）----
        case = case_by_trial.get(trial_id[5:] if trial_id.startswith("hira-") else trial_id)
        obs = meta.get("obstacles")
        if case is None:
            obstacle_bad.append(f"{trial_id}: 冻结地图里没有对应 case")
        elif not obs:
            obstacle_bad.append(f"{trial_id}: meta.obstacles 为空（未注入）")
        else:
            n_expect = int(case.get("n_obstacles", len(case.get("obstacles", []))))
            if int(obs.get("n_obstacles", -1)) != n_expect:
                obstacle_bad.append(f"{trial_id}: n_obstacles {obs.get('n_obstacles')} != 地图 {n_expect}")
            if obs.get("maps_sha256") != maps_sha:
                obstacle_bad.append(f"{trial_id}: 地图散列 {str(obs.get('maps_sha256'))[:12]} != {maps_sha[:12]}")
            dev = float(obs.get("max_pos_dev_m", 1e9))
            pose_devs.append(dev)
            if dev > POSE_TOL_M:
                obstacle_bad.append(f"{trial_id}: 障碍位姿读回偏差 {dev:.2e} > {POSE_TOL_M:g}")
            sp_meta = obs.get("start_pose") or {}
            sp_case = case.get("start_pose") or {}
            for key in ("px", "py", "yaw"):
                if abs(float(sp_meta.get(key, 1e9)) - float(sp_case.get(key, 0.0))) > 1e-9:
                    obstacle_bad.append(f"{trial_id}: start_pose.{key} 与地图不一致")
                    break
            n_obstacles_total += n_expect

        # ---- profile 实际旋钮（期望值由同一个采样器按 seed 重建）----
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
        # 推扰按**策略步**（0.02 s）调度，所以：
        #   1) 期望时刻若落在最后一个策略步之后（窗口末 <1 步），物理上调度不到，不能算缺失；
        #   2) 终止的 trial 之后不再施加，实际次数只会是"可调度"的前缀。
        # 两条都要显式写出来，否则会把调度边界当成采集缺陷（200 条里 2 条踩到，期望时刻在 29.996/29.998 s）。
        exp_p = list(expected.push_events or [])
        sched_hi = 30.0 - 0.02 + 1e-9
        sched = [p for p in exp_p if float(p["t_s"]) <= sched_hi]
        got_t = [float(p["t_s"]) for p in pushes]
        count_ok = (len(got_t) == len(sched)) if meta["outcome"] == "completed" else (len(got_t) <= len(sched))
        prefix_ok = all(abs(g - float(e["t_s"])) <= 0.02 + 1e-6 for g, e in zip(got_t, sched))
        if not (count_ok and prefix_ok):
            knob_mismatch.append(
                f"{trial_id}: 推扰 {len(got_t)} 次 vs 可调度 {len(sched)} 次"
                f"（期望总 {len(exp_p)}，末步后不可达 {len(exp_p) - len(sched)}）")
        for item in pushes:
            push_measured.append(float(np.hypot(*item["measured_dv_xy"])))

        # ---- 结构 ----
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

        # ---- 能量 ----
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

        commands = np.load(attempt / "pairing_10hz.npz")["command"].astype("float64")
        j_nom = float(model.j_nom(commands)) if commands.shape[0] >= 2 else 0.0
        n_used += 1
        if j_nom < J_MIN:
            n_non_computable += 1

        trial_rows.append({
            "trial_id": trial_id, "cell": cell, "outcome": outcome,
            "steps": n_steps, "j_exec": round(float(energy["j_exec"]), 3),
            "j_nom": round(j_nom, 3), "push_events": len(pushes),
            "n_obstacles": (obs or {}).get("n_obstacles"),
            "obstacle_pose_dev_m": round(float((obs or {}).get("max_pos_dev_m", -1)), 9),
            "start_pose": (obs or {}).get("start_pose"),
        })
        if problems:
            excluded.append({"trial_id": trial_id, "reason": "；".join(problems)})

    # ---- N01 配额 ----
    total_manifest = sum(c["manifest"] for c in cells.values())
    n_cells = len(cells)
    add("N01-quota", f"200 条 / 20 格（4 族 × 5 profile），每格 {CELL_QUOTA} 条",
        total_manifest == 200 and n_cells == 20
        and all(c["manifest"] == CELL_QUOTA for c in cells.values()),
        f"清单 {total_manifest} 条、{n_cells} 格、每格 {sorted({c['manifest'] for c in cells.values()})} 条")

    # ---- N02 身份 ----
    id_bad = [e for e in excluded if any(k in e["reason"] for k in
                                         ("bundle_id", "split=", "seed/program", "program_sha256", "数据锁"))]
    add("N02-identity", "bundle/split/seed/program 散列逐条与清单、冻结包、数据锁一致",
        not id_bad, f"不合格 {len(id_bad)} 条" + (f"：{id_bad[:3]}" if id_bad else ""))

    # ---- N03 障碍注入 ----
    add("N03-obstacles-injected",
        "逐条核对障碍注入：非空、数量与冻结地图一致、位姿读回偏差 ≤ 1e-3 m、起点等于地图 start_pose",
        not obstacle_bad,
        (f"{n_used} 条全部注入，累计障碍 {n_obstacles_total} 个；"
         f"位姿读回最大偏差 {max(pose_devs):.2e} m" if pose_devs and not obstacle_bad
         else f"异常 {len(obstacle_bad)} 条：{obstacle_bad[:4]}"))

    # ---- N04 profile 旋钮 ----
    push_detail = (
        f"；推扰实测 |Δv| ∈ [{min(push_measured):.3f}, {max(push_measured):.3f}] m/s（{len(push_measured)} 次）"
        if push_measured else "；本次无推扰"
    )
    add("N04-profile-knobs", "五种 profile 的实际旋钮（摩擦/质量/推扰次数）与按 seed 重建的期望值一致",
        not knob_mismatch,
        f"异常 {len(knob_mismatch)} 条" + (f"：{knob_mismatch[:4]}" if knob_mismatch else "") + push_detail)

    # ---- N05 结构 ----
    struct_bad = [e for e in excluded if any(k in e["reason"] for k in ("计数", "截断", "outcome=", "physics 行数"))]
    add("N05-structure", "completed=6000/1500/301；business_failure 截断自洽",
        not struct_bad, f"不合格 {len(struct_bad)} 条" + (f"：{struct_bad[:3]}" if struct_bad else ""))

    # ---- N06 能量 ----
    energy_bad = [e for e in excluded if "能量复算" in e["reason"] or "J_min" in e["reason"]]
    add("N06-energy", f"能量离线复算 rel ≤ {ENERGY_REL_MAX:g} 且 J ≥ J_min",
        not energy_bad, f"复算 {n_used} 条，最差 rel {worst_energy:.2e}；不合格 {len(energy_bad)}")

    # ---- N07 每格完整率 ----
    rate = {cell: {"manifest": c["manifest"], "completed": c["completed"],
                   "business_failure": c["business_failure"], "missing": c["missing"],
                   "full_rate": round(c["completed"] / c["manifest"], 4) if c["manifest"] else 0.0}
            for cell, c in sorted(cells.items())}
    rate_ok = all(v["full_rate"] >= FULL_RATE_MIN for v in rate.values()) and not any(
        v["missing"] for v in rate.values())
    worst_cell = min(rate.items(), key=lambda kv: kv[1]["full_rate"])
    add("N07-cell-full-rate", f"每格完整率 ≥ {FULL_RATE_MIN:g}（失败留在主分母，不补成功替换）",
        rate_ok,
        f"最差格 {worst_cell[0]} = {worst_cell[1]['completed']}/{worst_cell[1]['manifest']} "
        f"= {worst_cell[1]['full_rate']:.3f}；缺失 {sum(v['missing'] for v in rate.values())} 条")

    # ---- N08 non_computable ----
    add("N08-non-computable", f"J_nom < J_min 的比例 ≤ {NON_COMPUTABLE_MAX:g}",
        (n_non_computable / max(n_used, 1)) <= NON_COMPUTABLE_MAX,
        f"{n_non_computable}/{n_used} = {n_non_computable / max(n_used, 1):.4f}")

    # ---- N09 地图未改 ----
    lock_maps = lock.get("near_obstacle_maps") or {}
    add("N09-maps-unchanged", "地图散列 = 检验锁登记值（协议 no_redisign_rule：不得按新 δ 重设计）",
        lock_maps.get("sha256") in (None, maps_sha),
        f"现场 {maps_sha[:16]}；锁 {str(lock_maps.get('sha256'))[:16]}；cases={len(case_by_trial)}")

    # ---- N10 margin 绑定 ----
    mrec = lock.get("margins") or {}
    mp = Path(mrec.get("path") or "")
    bound = (mrec.get("state") == "frozen" and mp.is_file() and sha256_file(mp) == mrec.get("sha256"))
    add("N10-margins-bound", "采集所用检验锁已冻结 margin，且 margin 文件散列与锁一致",
        bound, f"margins.state={mrec.get('state')!r}；alpha={mrec.get('alpha')}；delta={mrec.get('delta')}")

    # ---- N11 未触碰其他 split ----
    others = [p.name for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("hira-near_obstacle")]
    add("N11-protected-splits", "本 run 只含 near_obstacle 的 trial，未混入其他 split",
        not others, f"run 内非 near_obstacle 目录：{others[:5] if others else '无'}")

    failed = [c["id"] for c in CHECKS if not c["ok"]]
    n_failed_trials = sum(1 for t in trial_rows if t["outcome"] != "completed")
    report = {
        "report": "near_obstacle_collection_integrity",
        "task": "HIRA-034",
        "run": {"run_id": args_cli.run_id, "dir": str(run_dir)},
        "bundle_id": bundle_id,
        "data_lock": {"path": args_cli.lock, "sha256": sha256_file(Path(args_cli.lock)),
                      "stage": lock.get("stage")},
        "maps": {"path": str(maps_path), "sha256": maps_sha, "n_cases": len(case_by_trial)},
        "model": {"path": args_cli.model, "sha256": model.sha256()},
        "counts": {
            "manifest": len(manifest),
            "evaluated": n_used,
            "completed": sum(c["completed"] for c in cells.values()),
            "business_failure": sum(c["business_failure"] for c in cells.values()),
            "missing": sum(c["missing"] for c in cells.values()),
            "n_non_computable": n_non_computable,
            "n_obstacles_total": n_obstacles_total,
        },
        "cells": rate,
        "excluded": excluded,
        "energy_worst_rel": worst_energy,
        "obstacle_pose_dev_max_m": max(pose_devs) if pose_devs else None,
        "checks": CHECKS,
        "trials": trial_rows,
        "verdict": {
            "all_passed": not failed,
            "failed": failed,
            "note": "失败仍留在主分母；本审查只读 near_obstacle，不评估覆盖率（那是 HIRA-035），"
                    "未反向调整 α/δ、功率模型或地图",
        },
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"NEAR_INTEGRITY all_passed={not failed} failed={failed} "
          f"completed={report['counts']['completed']}/{len(manifest)} "
          f"failed_trials={n_failed_trials}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
