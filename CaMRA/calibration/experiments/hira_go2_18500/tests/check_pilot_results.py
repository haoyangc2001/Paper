"""HIRA-022：20 条五条件 pilot 的独立验证与资源估算（无需仿真）。

对 run 目录里每条 trial 重新核验（不信任采集器的自述）：

* DONE 完整、身份与 G1 冻结包一致（这是"正式 runner 跑的是冻结包"的证据）；
* 配对点 301、50 Hz 1500 点、6000 物理步、命令列与程序生成器逐元素一致；
* 能量离线复算；profile 读回与 spec 一致；推扰实测 ≈ 指令；
* 名义/执行配对误差（`camra_calibration.nominal` 精确积分）；
* 摔倒按业务结果真实计数；低速段占比、near_zero 段存在性；
* 资源：每条 wall、每条存储 → 外推 400/800/200 条的估算。

    python tests/check_pilot_results.py --phase merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
CAL_SRC = REPO / "CaMRA" / "calibration" / "src"
for path in (str(ADAPTER), str(CAL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("merge",), default="merge")
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default=None, help="缺省取 runs 下最新的 pilot_*")
parser.add_argument("--out-dir", default=str(EXP / "reports"))
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
args_cli = parser.parse_args()

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402
from hira_go2.collector import done_is_valid, plan_attempts, sample_spec  # noqa: E402

FAMILIES = ("straight_shift", "arc_combined", "alternating", "start_stop_turn")
PROFILES = ("clean", "friction_only", "mass_only", "push_only", "combined")
CHECKS: list[dict] = []


def add(cid, name, ok, detail, **extra):
    record = {"id": cid, "name": name, "ok": bool(ok), "detail": detail}
    record.update(extra)
    CHECKS.append(record)
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name} :: {detail}", flush=True)
    return record


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> int:
    out_dir = Path(args_cli.out_dir)
    runs_root = Path(args_cli.runs_root)
    if args_cli.run_id:
        run_dir = runs_root / args_cli.run_id
    else:
        candidates = sorted(p for p in runs_root.glob("pilot_*") if p.is_dir())
        if not candidates:
            raise SystemExit(f"runs 下没有 pilot_* 目录：{runs_root}")
        run_dir = candidates[-1]
    stats = json.loads((run_dir / "run_stats.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "collect_config.json").read_text(encoding="utf-8"))
    gate = json.loads((out_dir / "integration_gate.json").read_text(encoding="utf-8"))
    bundle_id = gate["bundle"]["bundle_id"]
    print(f"run_dir={run_dir} bundle={bundle_id[:16]}", flush=True)

    # ---- 重建 20 条 spec（同一生成器 → 命令表可复现）----
    specs = []
    index = 0
    for family in FAMILIES:
        for profile in PROFILES:
            seed = config["seed_range_pilot"][0] + index
            program = build_program(family, seed, duration_s=config["window_s"])
            specs.append(
                (
                    sample_spec(
                        trial_id=f"hira-pilot-{family}-{profile}-0000",
                        split="pilot", program=family, profile=profile, seed=seed,
                        commands=[list(c) for c in program.commands],
                        window_s=config["window_s"],
                    ),
                    program,
                )
            )
            index += 1

    # ---- P1/P2 全量 DONE 与身份 ----
    plan = plan_attempts(run_dir, [s for s, _ in specs])
    add("P01-done", "20/20 DONE 完整且散列一致",
        all(item["action"] == "skip" for item in plan),
        f"{[item['action'] for item in plan].count('skip')}/20 skip；"
        f"其余 {[item['action'] for item in plan if item['action'] != 'skip']}")

    rows = []
    for spec, program in specs:
        attempt_dir = run_dir / spec.trial_id / "attempt_01"
        meta = json.loads((attempt_dir / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((attempt_dir / "energy.json").read_text(encoding="utf-8"))
        pairing = np.load(attempt_dir / "pairing_10hz.npz")
        state = np.load(attempt_dir / "state_50hz.npz")
        phys = np.load(attempt_dir / "physics_200hz.npz")
        rows.append({"spec": spec, "program": program, "dir": attempt_dir,
                     "meta": meta, "energy": energy, "pairing": pairing,
                     "state": state, "phys": phys})

    identity_ok = all(
        row["meta"]["bundle_id"] == bundle_id
        and row["meta"]["controller_id"] == gate["bundle"]["controller_id"]
        and row["meta"]["experiment_id"] == stats["experiment_id"]
        and config["seed_range_pilot"][0] <= row["meta"]["seed"] <= config["seed_range_pilot"][1]
        and row["meta"]["attempt"] == 1
        for row in rows
    )
    add("P02-identity", "身份与 G1 冻结包一致（正式 runner 跑的就是冻结包）", identity_ok,
        f"20 条 meta 的 bundle_id 均为 {bundle_id[:16]}…，controller/experiment/seed 区间一致")

    # ---- P3 配对点与命令复现 ----
    p3_ok, p3_detail = True, []
    for row in rows:
        z = row["pairing"]
        points = len(z["t_s"])
        dt = float(np.diff(z["t_s"]).min())
        expected = np.array(
            [row["spec"].commands[i % len(row["spec"].commands)] for i in range(points)],
            dtype=np.float32,
        )
        cmd_ok = bool(np.allclose(z["command"], expected, atol=1e-6))
        p3_ok = p3_ok and points == 301 and abs(dt - 0.1) < 1e-9 and cmd_ok
    add("P03-pairing", "配对 301 点、0.1 s 间距、命令与程序生成器逐元素一致", p3_ok,
        "20 条全部 301 点 / Δt=0.1 / 命令一致" if p3_ok else "存在不一致")

    # ---- P4 状态与步数 ----
    p4_ok = all(
        len(row["state"]["t_s"]) == 1500 and row["meta"]["timing"]["steps_executed"] == 6000
        for row in rows
    )
    add("P04-state-steps", "50 Hz 1500 点、6000 物理步", p4_ok, "20 条全部一致")

    # ---- P5 能量复算 ----
    worst = 0.0
    total_j = []
    for row in rows:
        j = float(
            np.maximum(
                row["phys"]["tau_applied_source"].astype("float64")
                * row["phys"]["qdot_post_source"].astype("float64"),
                0,
            ).sum() * 0.005
        )
        rel = abs(j - row["energy"]["j_exec"]) / max(abs(row["energy"]["j_exec"]), 1e-12)
        worst = max(worst, rel)
        total_j.append(row["energy"]["j_exec"])
    add("P05-energy", "执行能量离线复算", worst <= 1e-9,
        f"20 条复算最大 rel={worst:.2e}（判据 1e-9）；J 范围 {min(total_j):.1f}–{max(total_j):.1f} J",
        worst_rel=worst, j_range=[min(total_j), max(total_j)])

    # ---- P6 profile 读回 ----
    p6_ok, p6_detail = True, []
    base_mass_seen = set()
    for row in rows:
        spec, meta = row["spec"], row["meta"]
        applied = meta["profile_applied"]
        # 旋钮无条件写（绝对式、幂等）：readback 必须等于 spec 携带的值，
        # 与 profile 无关——这防止"mass_only 批继承 friction_only 批的摩擦"一类跨批污染。
        # base_mass_kg 是 base 刚体（躯干）的质量（≈4.5 kg），不是整车 15.019 kg
        base_mass_seen.add(round(applied["mass"]["base_mass_kg"], 3))
        ok = abs(applied["friction"]["readback"] - spec.friction) < 1e-6
        ok = ok and abs(applied["mass"]["readback"] - (applied["mass"]["base_mass_kg"] + spec.added_mass_kg)) < 1e-3
        p6_ok = p6_ok and ok
        if len(p6_detail) < 3:
            p6_detail.append(f"{spec.trial_id}: friction {applied['friction']['readback']}（spec {spec.friction}），"
                             f"base {applied['mass']['base_mass_kg']:.3f}+{spec.added_mass_kg} → {applied['mass']['readback']:.3f}")
    add("P06-profile-readback", "profile 旋钮读回与 spec 一致（绝对式、幂等，无跨批污染）", p6_ok,
        ("；".join(p6_detail) + f"；base 刚体质量集合 {sorted(base_mass_seen)} kg")
        if p6_ok else "存在不一致")

    # ---- P7 推扰 ----
    p7_ok, n_push = True, 0
    for row in rows:
        spec, trace = row["spec"], row["meta"]["push_trace"]
        for event, measured in zip(spec.push_events or [], trace):
            n_push += 1
            dv = measured["measured_dv_xy"]
            p7_ok = p7_ok and abs(dv[0] - event["dv_xy"][0]) < 0.05 and abs(dv[1] - event["dv_xy"][1]) < 0.05
    add("P07-push", "推扰实测 Δv 与指令一致", p7_ok, f"共 {n_push} 次推扰")

    # ---- P8 名义/执行配对误差 ----
    errors = []
    for row in rows:
        spec, program, meta = row["spec"], row["program"], row["meta"]
        pose = meta["profile_applied"]["initial_pose"] or {"px_m": 0.0, "py_m": 0.0, "yaw_rad": 0.0}
        initial = PlanarState(px=float(pose["px_m"]), py=float(pose["py_m"]), yaw=float(pose["yaw_rad"]))
        nominal = integrate(initial, program.commands[:300], 0.1)
        z = row["pairing"]
        n = min(len(z["t_s"]), len(nominal))
        exec_xy = z["pos_xy"][:n].astype("float64")
        nom_xy = np.array([[s.px, s.py] for s in nominal[:n]])
        errors.extend(float(v) for v in np.linalg.norm(exec_xy - nom_xy, axis=-1))
    errors_arr = np.array(errors)
    # 开环跟踪的固有漂移：机器人没有航向闭环，30 s 内与名义轨迹差到米级是预期行为；
    # 此项是"配对管线可用 + 逐组统计"的报告口径，不是判据（is_criterion=False）
    group_err = {}
    for row in rows:
        spec, program, meta = row["spec"], row["program"], row["meta"]
        pose = meta["profile_applied"]["initial_pose"] or {"px_m": 0.0, "py_m": 0.0, "yaw_rad": 0.0}
        initial = PlanarState(px=float(pose["px_m"]), py=float(pose["py_m"]), yaw=float(pose["yaw_rad"]))
        nominal = integrate(initial, program.commands[:300], 0.1)
        z = row["pairing"]
        n = min(len(z["t_s"]), len(nominal))
        exec_xy = z["pos_xy"][:n].astype("float64")
        nom_xy = np.array([[s2.px, s2.py] for s2 in nominal[:n]])
        group_err.setdefault(spec.profile, []).extend(
            float(v) for v in np.linalg.norm(exec_xy - nom_xy, axis=-1)
        )
    group_summary = {
        profile: {"median_m": round(float(np.median(values)), 3),
                  "max_m": round(float(np.max(values)), 3)}
        for profile, values in group_err.items()
    }
    all_err = np.array([v for values in group_err.values() for v in values])
    finite_ok = bool(np.isfinite(all_err).all())
    add("P08-pairing-error", "名义/执行配对误差分布（开环跟踪漂移，报告口径非判据）",
        finite_ok and float(np.median(all_err)) < 5.0,
        f"{len(all_err)} 个配对点：全体中位 {np.median(all_err):.3f} m，最大 {all_err.max():.3f} m；"
        f"分组中位 {group_summary}",
        is_criterion=False, per_group=group_summary,
        median_m=float(np.median(all_err)), max_m=float(all_err.max()))

    # ---- P9 摔倒与低速/near_zero 计数 ----
    fell = [row["spec"].trial_id for row in rows if row["meta"]["outcome"] == "terminated_early"]
    completed = [row["spec"].trial_id for row in rows if row["meta"]["outcome"] == "completed"]
    low_share = []
    has_near_zero = 0
    for row in rows:
        targets = row["program"].targets
        low = [t for t in targets if abs(t[0]) <= 0.3 and abs(t[2]) <= 0.3]
        low_share.append(len(low) / len(targets))
        if any(abs(t[0]) < 1e-9 and abs(t[1]) < 1e-9 and abs(t[2]) < 1e-9 for t in targets):
            has_near_zero += 1
    add("P09-outcomes", "业务结果真实计数（摔倒不隐身、低速与 near_zero 在报告中可见）",
        len(completed) + len(fell) == 20,
        f"完成 {len(completed)} 条，提前终止 {len(fell)} 条（{fell or '无'}）；"
        f"程序低速段占比 {min(low_share):.0%}–{max(low_share):.0%}（协议下限 25%），"
        f"{has_near_zero}/20 条含 near_zero 段",
        fell=fell, n_completed=len(completed),
        low_speed_share=[round(min(low_share), 2), round(max(low_share), 2)])

    # ---- P10 资源估算 ----
    per_trial_s = stats["wall_total_s"] / 20
    per_trial_mb = stats["total_dir_bytes"] / 20 / 1e6
    estimates = {
        "measured": {
            "wall_per_trial_s": round(per_trial_s, 1),
            "dir_mb_per_trial": round(per_trial_mb, 2),
            "rss_max_mb": stats["rss_max_mb"],
            "total_run_min": round(stats["wall_total_s"] / 60, 1),
        },
        "extrapolated": {
            "pilot20_done": True,
            "baseline400_min": round(400 * per_trial_s / 60, 1),
            "calibration800_min": round(800 * per_trial_s / 60, 1),
            "heldout400_min": round(400 * per_trial_s / 60, 1),
            "near_obstacle200_min": round(200 * per_trial_s / 60, 1),
            "storage_all_GB": round((400 + 800 + 400 + 200) * per_trial_mb / 1024, 1),
        },
        "note": "外推按单 trial 均摊（含批次开销）；推扰/摩擦/噪声批与 clean 同批大小，批间启动已摊入",
    }
    add("P10-resources", "本机实测吞吐与存储 → 更新资源估算", True,
        f"实测 {per_trial_s:.1f} s/条、{per_trial_mb:.2f} MB/条、峰值 RSS {stats['rss_max_mb']:.0f} MB；"
        f"外推 400/800/400/200 条 ≈ {estimates['extrapolated']['baseline400_min']}/"
        f"{estimates['extrapolated']['calibration800_min']}/"
        f"{estimates['extrapolated']['heldout400_min']}/"
        f"{estimates['extrapolated']['near_obstacle200_min']} 分钟，"
        f"总存储 {estimates['extrapolated']['storage_all_GB']} GB",
        **estimates)

    failed = [item["id"] for item in CHECKS if not item["ok"]]
    report = {
        "report": "pilot_validation",
        "task": "HIRA-022",
        "run_id": stats["run_id"],
        "run_dir": str(run_dir),
        "bundle_id": bundle_id,
        "seed_range": config["seed_range_pilot"],
        "checks": CHECKS,
        "resources": estimates,
        "outcomes": {"completed": len(completed), "terminated_early": len(fell), "fell_ids": fell},
        "verdict": {"n_checks": len(CHECKS), "failed": failed, "all_passed": not failed},
    }
    out = out_dir / "pilot_validation.json"
    out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"PILOT_{'PASSED' if not failed else 'BLOCKED'} failed={failed} -> {out}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"PILOT_VALIDATION_ERROR {exc}", flush=True)
        code = 2
    finally:
        import os

        os._exit(code)
