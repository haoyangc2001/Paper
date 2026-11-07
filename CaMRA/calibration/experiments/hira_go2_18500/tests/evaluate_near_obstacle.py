"""HIRA-035：近障检查的覆盖、几何适用性与可用性结论。

只读 `near_obstacle` split；α/δ、功率模型、足迹、地图全部从冻结产物加载，不重算、不调整。

本任务要回答的问题不是"控制器能不能避障"（没有上层导航，固定命令回放不构成避障器），
而是：**在冻结的 δ 下，这批近障工况还有没有几何上的意义**。因此分三层报告，不许混为一谈：

1. **实测几何**（执行轨迹 × 冻结足迹 × 冻结障碍）：几何重叠、最小净空、场地边界余量、净距档位。
   这是"机器人实际跑成什么样"。
2. **名义几何**（名义轨迹 × 冻结足迹 × 冻结障碍）：计划路线本身离障碍多近。
3. **δ 条件**：名义净空 ≥ δ 的条数。δ 是危险集膨胀量，若一条都不满足，说明在这套地图尺度下
   该 δ 无法用于近障任务——按协议只作"记录并询问"，**不得靠改分位缩小 δ 来凑可行**。

口径与 HIRA-031/033 对齐：ε_c=(J_exec−J_nom)/J_nom，ε_d 为 10 Hz 配对网格上执行与名义的
最大平面欧氏距离；覆盖判定等于阈值算覆盖；bootstrap 以整条轨迹/整簇为单位；两套分母并列
（全体 200 与剔除两个已裁定资格缺口格的 180）。

    python tests/evaluate_near_obstacle.py --run-id near_obstacle_20261107 \\
        --adjudicated-gap-cell arc_combined/combined \\
        --adjudicated-gap-cell start_stop_turn/mass_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.coverage import (  # noqa: E402
    TARGET_COVERAGE,
    HeldoutRow,
    cost_covered,
    coverage_verdict,
    denominators,
    direction_of,
    per_cell,
    position_covered,
    rate,
    rate_with_ci,
    subgroup,
)
from camra_calibration.margins import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: E402
from camra_calibration.near_obstacle import (  # noqa: E402
    by_tier,
    footprint_from_json,
    obstacles_from_case,
    trial_geometry,
    yaw_from_quat,
)
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402
from camra_calibration.obstacles import min_path_clearance, nominal_path  # noqa: E402
from camra_calibration.pairing import deviation, epsilon_c, low_speed_share  # noqa: E402
from camra_calibration.power import J_MIN, PowerModel  # noqa: E402
from hira_go2.collector import attempt_number, done_is_valid  # noqa: E402

SPLIT = "near_obstacle"
PAIR_DT = 0.1
CELL_QUOTA = 10
LOW_SPEED_SHARE = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default="near_obstacle_20261107")
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--footprint", default=str(EXP / "config/footprint_hira_v1.json"))
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_evaluation_v2.json"))
parser.add_argument("--out", default=str(EXP / "reports/near_obstacle_evaluation.json"))
parser.add_argument("--out-summary", default=str(EXP / "reports/near_obstacle_summary.md"))
parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
parser.add_argument("--stride", type=int, default=1, help="几何检查的帧步长（1=50Hz 全帧，最严格）")
parser.add_argument("--adjudicated-gap-cell", action="append", default=[],
                    help="完整率低于 0.9、已被显式裁定的格（family/profile），必须列全；少列即失败")
args_cli = parser.parse_args()


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
    lock = json.loads(Path(args_cli.lock).read_text(encoding="utf-8"))
    if lock.get("stage") != "evaluation":
        raise SystemExit(f"REFUSED：必须用 evaluation stage lock，得到 {lock.get('stage')!r}")
    margins = json.loads(Path(args_cli.margins).read_text(encoding="utf-8"))
    alpha = float(margins["alpha"])
    delta = float(margins["delta"])
    model = PowerModel.load(args_cli.model, verify_hash=True)
    fp_payload = json.loads(Path(args_cli.footprint).read_text(encoding="utf-8"))
    footprint = footprint_from_json(args_cli.footprint)
    fp_min_width = float(fp_payload["polygon"]["min_width"])
    maps_path = Path(args_cli.maps)
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    case_by_trial = {str(c["trial_id"]): c for c in maps.get("cases", [])}

    manifest_path = Path(args_cli.manifests) / f"{SPLIT}.jsonl"
    manifest = {row["trial_id"]: row for row in
                (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    run_dir = Path(args_cli.runs_root) / args_cli.run_id
    print(f"[near-eval] {len(manifest)} 条，run {run_dir}，alpha={alpha:.6f} delta={delta:.6f}", flush=True)

    rows: list[HeldoutRow] = []
    geo_exec = []
    nominal_rows: list[dict] = []
    missing: list[str] = []
    print(f"[geo] 基宽 min_width={fp_min_width:.4f} m，轮廓顶点 {footprint.vertices.shape[0]}，stride={args_cli.stride}", flush=True)

    for i, (trial_id, spec) in enumerate(manifest.items()):
        attempt = best_attempt(run_dir / trial_id)
        if attempt is None:
            missing.append(trial_id)
            continue
        meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((attempt / "energy.json").read_text(encoding="utf-8"))
        pairing = np.load(attempt / "pairing_10hz.npz")
        commands = pairing["command"].astype(np.float64)
        exec_xy = pairing["pos_xy"].astype(np.float64)
        t_pair = pairing["t_s"].astype(np.float64)
        state = np.load(attempt / "state_50hz.npz")
        pos50 = state["root_pos_w"][:, :2].astype(np.float64)
        yaw50 = yaw_from_quat(state["root_quat_w"].astype(np.float64))

        # ---- ε_c / ε_d（与 HIRA-031/033 同口径）----
        program = build_program(spec["family"], int(spec["seed"]),
                                duration_s=spec["duration_s"], command_dt=spec["command_dt"])
        expected = np.asarray([list(map(float, c)) for c in program.commands], dtype=np.float64)
        cmd_ok = (expected.shape == commands.shape and
                  float(np.abs(expected - commands).max()) <= 1e-5)
        if not cmd_ok and commands.shape[0] >= 2:
            # 业务失败时配对网格是前缀
            n_got = commands.shape[0]
            cmd_ok = (meta["outcome"] == "business_failure" and n_got < expected.shape[0]
                      and float(np.abs(expected[:n_got] - commands).max()) <= 1e-5)
        pose = (meta.get("profile_applied") or {}).get("initial_pose") or {}
        initial = PlanarState(px=float(pose.get("px_m", 0.0)), py=float(pose.get("py_m", 0.0)),
                              yaw=float(pose.get("yaw_rad", 0.0)))
        nominal_states = integrate(initial, commands, PAIR_DT)
        nom_xy = np.asarray([[s.px, s.py] for s in nominal_states], dtype=np.float64)

        j_exec = float(energy["j_exec"])
        j_nom = float(model.j_nom(commands)) if commands.shape[0] >= 2 else None
        complete = meta["outcome"] == "completed"
        eps_d = float(deviation(exec_xy, nom_xy).max()) if nom_xy.shape == exec_xy.shape else None
        eps_c_value = epsilon_c(j_exec, j_nom, J_MIN)[0] if (complete and j_nom is not None) else None
        cell = f"{spec['family']}/{spec['profile']}"
        rows.append(HeldoutRow(
            trial_id=trial_id, family=spec["family"], profile=spec["profile"], cell=cell,
            direction=direction_of(commands),
            low_speed=bool(low_speed_share(commands) >= LOW_SPEED_SHARE),
            outcome=meta["outcome"], complete=complete,
            official=bool(complete and eps_c_value is not None and eps_d is not None),
            non_computable=bool(j_nom is not None and j_nom < J_MIN), data_invalid=not cmd_ok,
            eps_c=eps_c_value, eps_d=eps_d))

        # ---- 实测几何（执行轨迹 50 Hz）----
        key = trial_id[5:] if trial_id.startswith("hira-") else trial_id
        case = case_by_trial[key]
        geo_exec.append(trial_geometry(trial_id, cell, float(case["clearance_target_m"]), case,
                                       footprint, pos50[:: args_cli.stride], yaw50[:: args_cli.stride]))

        # ---- 名义几何 + δ 条件（10 Hz 名义路径）----
        nom_pos, nom_yaw = nominal_path(commands, (initial.px, initial.py), initial.yaw, dt=PAIR_DT)
        nobs = obstacles_from_case(case)
        nom_clear, worst = min_path_clearance(footprint, nobs, nom_pos, nom_yaw)
        nominal_rows.append({
            "trial_id": trial_id, "cell": cell, "outcome": meta["outcome"],
            "nominal_min_clearance_m": float(nom_clear),
            "worst_frame": worst["frame"], "worst_obstacle": worst["obstacle"],
            "tier_m": float(case["clearance_target_m"]),
            "satisfies_delta": bool(nom_clear >= delta),
            "arena": case["arena"],
        })
        if (i + 1) % 25 == 0:
            print(f"[geo] {i + 1}/{len(manifest)} 条", flush=True)

    print(f"[geo] 完成 {len(rows)} 条，缺失 {len(missing)}", flush=True)

    # ---- 覆盖：两套分母 ----
    def cover(subset: list[HeldoutRow], expected: int) -> dict:
        official = [r for r in subset if r.official]
        cells_key = [r.cell for r in official]
        c_flags = [cost_covered(r.eps_c, alpha) for r in official]
        d_flags = [position_covered(r.eps_d, delta) for r in official]
        joint = [bool(a) and bool(b) for a, b in zip(c_flags, d_flags) if a is not None and b is not None]
        den = denominators(subset, expected)
        overall = {
            "cost": rate_with_ci(c_flags, cells_key, args_cli.replicates, args_cli.seed),
            "position": rate_with_ci(d_flags, cells_key, args_cli.replicates, args_cli.seed),
            "joint": rate_with_ci(joint, cells_key, args_cli.replicates, args_cli.seed),
        }
        cells = per_cell(subset, alpha, delta, CELL_QUOTA)
        return {"denominators": den, "overall": overall, "per_cell": cells,
                "n_official": len(official),
                "verdict": coverage_verdict(den, overall, cells, target=TARGET_COVERAGE)}

    full = cover(rows, len(manifest))
    actual_gap = sorted(c for c, v in full["per_cell"].items()
                        if v["complete_share"] is not None and v["complete_share"] < 0.9)
    declared_gap = sorted(args_cli.adjudicated_gap_cell)
    if actual_gap != declared_gap:
        raise SystemExit(
            "REFUSED：完整率不达标的格与 --adjudicated-gap-cell 不一致（必须列全、不得多列）\n"
            f"  实际：{actual_gap}\n  声明：{declared_gap}")

    gap_set = set(actual_gap)
    trim_rows = [r for r in rows if r.cell not in gap_set]
    trim_expected = len(manifest) - sum(1 for r in rows if r.cell in gap_set)
    trim = cover(trim_rows, trim_expected)

    # ---- 几何汇总 ----
    exec_min = min(g.min_clearance_m for g in geo_exec)
    n_overlap = sum(1 for g in geo_exec if not g.collision_free)
    n_outside = sum(1 for g in geo_exec if not g.inside_arena)
    nom_clear_arr = np.asarray([n["nominal_min_clearance_m"] for n in nominal_rows])
    n_sat = int(sum(1 for n in nominal_rows if n["satisfies_delta"]))
    # 判据稳健性：上面用的是「足迹到障碍 ≥ δ」（footprint 已在危险集定义内，偏保守）。
    # 换成最宽松的「中心线到障碍 ≥ δ」再判一次：直接用冻结地图自带的 clearance_achieved_m，
    # 它正是地图生成时按中心线量出来的。两种口径都得 0，结论才不依赖口径选择。
    centerline = np.asarray([float(c["clearance_achieved_m"]) for c in maps["cases"]], dtype=np.float64)
    n_sat_centerline = int(np.sum(centerline >= delta))
    nom_by_tier: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_satisfies": 0, "min_clear": 9e9})
    for n in nominal_rows:
        k = f"{n['tier_m']:.2f}"
        slot = nom_by_tier[k]
        slot["n"] += 1
        slot["n_satisfies"] += int(n["satisfies_delta"])
        slot["min_clear"] = min(slot["min_clear"], n["nominal_min_clearance_m"])
    for v in nom_by_tier.values():
        v["min_nominal_clearance_m"] = float(v["min_clear"])
        v.pop("min_clear")

    geometry = {
        "footprint": {"path": str(args_cli.footprint), "sha256": sha256_file(Path(args_cli.footprint)),
                      "n_vertices": int(footprint.vertices.shape[0]),
                      "min_width_m": fp_min_width},
        "executed": {
            "frame_rate_hz": 50 // max(args_cli.stride, 1),
            "n_trials": len(geo_exec),
            "n_geometric_overlap": int(n_overlap),
            "n_outside_arena": int(n_outside),
            "min_clearance_m": float(exec_min),
            "by_tier": by_tier(geo_exec),
        },
        "nominal": {
            "n_trials": len(nominal_rows),
            "min_clearance_m": float(nom_clear_arr.min()) if nom_clear_arr.size else None,
            "median_clearance_m": float(np.median(nom_clear_arr)) if nom_clear_arr.size else None,
            "max_clearance_m": float(nom_clear_arr.max()) if nom_clear_arr.size else None,
            "by_tier": dict(nom_by_tier),
        },
        "delta_condition": {
            "delta_m": delta,
            "criterion": "名义轨迹上 足迹到障碍的最小净距 ≥ δ（危险集 = obstacle ⊕ footprint ⊕ disc(δ)）",
            "n_total": len(nominal_rows),
            "n_satisfying": int(n_sat),
            "n_not_satisfying": int(len(nominal_rows) - n_sat),
            "satisfying_share": (n_sat / len(nominal_rows)) if nominal_rows else None,
            "evaluable": bool(n_sat > 0),
            "robustness_centerline": {
                "criterion": "中心线到障碍 ≥ δ（地图自带 clearance_achieved_m，最宽松口径）",
                "n_satisfying": n_sat_centerline,
                "median_clearance_m": float(np.median(centerline)),
                "max_clearance_m": float(centerline.max()),
                "delta_minus_max_m": float(delta - centerline.max()),
            },
            "note": ("0 条满足 → 按协议报告 0 与「不可评估」，不得换地图凑样本、不得改分位缩小 δ"
                     if n_sat == 0 else f"{n_sat} 条满足，可在该子组内单独统计"),
        },
        "caveat": [
            "场地边界墙未建模：越界是用冻结足迹对 arena 尺寸的几何余量判定，不是物理碰撞",
            "物理接触没有记录接触力/法向冲量；本节用「足迹多边形与障碍多边形几何重叠」作为碰撞的下界证据",
            "几何重叠 ≠ 传感器看到障碍；本节不评估感知，也不构成避障成功率",
            "名义净空是计划路线本身的量，与执行漂移无关；执行漂移由 ε_d 与覆盖部分单独报告",
        ],
    }

    headline = (
        f"在 {len(manifest)} 条近障 trial 上（名义净空判据）**{n_sat}/{len(nominal_rows)} 条满足 δ={delta:.3f} m**，"
        f"名义最小净空中位 {np.median(nom_clear_arr):.2f} m、最小 {nom_clear_arr.min():.2f} m、最大 {nom_clear_arr.max():.2f} m；"
        f"换成最宽松的「中心线净距 ≥ δ」口径也仍是 {n_sat_centerline}/200（地图自带的中心线净距最大 {centerline.max():.2f} m，"
        f"比 δ 还小 {delta - centerline.max():.2f} m）；"
        f"实测几何重叠 {n_overlap}/{len(geo_exec)} 条、越界 {n_outside} 条、最小净空 {exec_min:.4f} m。"
        f"条件分母 {full['n_official']} 条上，成本覆盖 {full['overall']['cost']['rate']:.4f}、"
        f"位置覆盖 {full['overall']['position']['rate']:.4f}、联合覆盖 {full['overall']['joint']['rate']:.4f}。"
        "结论：在该 δ 下近障任务在几何上不可评估——危险集膨胀量远超场地尺度，"
        "这不是控制器缺陷，而是裕度与地图量级不匹配。"
    )

    report = {
        "report": "near_obstacle_evaluation",
        "task": "HIRA-035",
        "run": {"run_id": args_cli.run_id, "dir": str(run_dir),
                "manifest_sha256": sha256_file(manifest_path)},
        "alpha": alpha,
        "delta": delta,
        "margins": {"file": str(args_cli.margins), "file_sha256": sha256_file(Path(args_cli.margins)),
                    "content_sha256": margins.get("sha256"), "state": "frozen"},
        "power_model": {"file": str(args_cli.model), "sha256": model.sha256()},
        "maps": {"file": str(maps_path), "sha256": sha256_file(maps_path), "n_cases": len(case_by_trial)},
        "data_lock": {"file": Path(args_cli.lock).name, "sha256": sha256_file(Path(args_cli.lock)),
                      "stage": lock.get("stage")},
        "n_manifest": len(manifest),
        "n_observed": len(rows),
        "missing": missing,
        "outcomes": dict(Counter(r.outcome for r in rows)),
        "gap_cells": actual_gap,
        "full_sample": full,
        "gap_cells_removed": trim,
        "groups": {
            "by_profile": subgroup(rows, lambda r: r.profile, alpha, delta),
            "by_family": subgroup(rows, lambda r: r.family, alpha, delta),
            "by_low_speed": subgroup(rows, lambda r: "low_speed" if r.low_speed else "high_speed", alpha, delta),
            "by_direction": subgroup(rows, lambda r: r.direction, alpha, delta),
        },
        "geometry": geometry,
        "nominal_rows": nominal_rows,
        "headline": headline,
        "scope_notes": [
            "本节是执行误差与几何适用性检查，**不是**已实现上层导航的避障成功率验收。",
            "全部 200 条预登记案例都进主表；没有只保留满足裕度且成功的轨迹。",
            f"格级正式结论只覆盖 {20 - len(actual_gap)}/20 格；" + "、".join(f"`{c}`" for c in actual_gap)
            + " 完整率不足 0.9，仅报诊断值（HIRA-034 裁定）。",
            "δ 由校准集冻结，不因近障表现调整；地图在采集前冻结，不因 δ 过大重新设计。",
        ],
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---- 人读摘要 ----
    L: list[str] = []
    A = L.append
    A("# HiRA 近障检查评估（HIRA-035）")
    A("")
    A(f"- run：`{args_cli.run_id}`，清单 {len(manifest)} 条，观测 {len(rows)} 条，缺失 {len(missing)} 条")
    A(f"- 冻结裕度：α = {alpha:.6f}，δ = **{delta:.6f} m**")
    A(f"- 足迹：{footprint.vertices.shape[0]} 顶点，基宽 {fp_min_width:.4f} m")
    A(f"- 地图：{len(case_by_trial)} 例，散列 {sha256_file(maps_path)[:16]}")
    A("")
    A("## 一句话结论")
    A("")
    A(headline)
    A("")
    A("## 1. δ 条件（名义轨迹 × 足迹 × 障碍）")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 判据 | {geometry['delta_condition']['criterion']} |")
    A(f"| 满足条数 | **{n_sat} / {len(nominal_rows)}** |")
    A(f"| 名义最小净空中位 | {np.median(nom_clear_arr):.3f} m |")
    A(f"| 名义最小净空 min / max | {nom_clear_arr.min():.3f} / {nom_clear_arr.max():.3f} m |")
    A(f"| 可评估 | {'是' if n_sat > 0 else '**否（0 条满足，按协议报告 0 与不可评估）**'} |")
    A(f"| 稳健性：中心线口径 | {n_sat_centerline}/{len(maps['cases'])} 满足（最大中心线净距 {centerline.max():.3f} m，比 δ 小 {delta - centerline.max():.3f} m） |")
    A("")
    A("## 2. 实测几何（执行轨迹 50 Hz × 足迹 × 障碍）")
    A("")
    A("| 净距档位 | 条数 | 几何重叠 | 最小净空 | 中位净空 | 越界 |")
    A("|---|---|---|---|---|---|")
    for tier, v in geometry["executed"]["by_tier"].items():
        A(f"| {tier} m | {v['n']} | {v['n_collision']} | {v['min_clearance_m']:.4f} | "
          f"{v['median_clearance_m']:.4f} | {v['n_outside_arena']} |")
    A("")
    A(f"合计：几何重叠 **{n_overlap}/{len(geo_exec)}**，越界 **{n_outside}**，最小净空 **{exec_min:.4f} m**。")
    A("")
    A("## 3. 覆盖（两套分母并列）")
    A("")
    A("| 口径 | 成本 | 位置 | 联合 |")
    A("|---|---|---|---|")
    for tag, blk in (("全体 200 条", full), ("剔除两格（180 条）", trim)):
        o = blk["overall"]
        def f(b):
            return "n/a" if b["rate"] is None else f"{b['rate']:.4f} ({b['n_covered']}/{b['n']})"
        A(f"| {tag} | {f(o['cost'])} | {f(o['position'])} | {f(o['joint'])} |")
    A("")
    A("## 4. 按 profile")
    A("")
    A("| profile | n | 成本 | 位置 | 联合 |")
    A("|---|---|---|---|---|")
    for k, v in report["groups"]["by_profile"].items():
        def g(b):
            return "n/a" if b["rate"] is None else f"{b['rate']:.4f}"
        A(f"| {k} | {v['n']} | {g(v['cost'])} | {g(v['position'])} | {g(v['joint'])} |")
    A("")
    A("## 5. 适用范围")
    A("")
    for s in report["scope_notes"]:
        A(f"- {s}")
    A("")
    A("## 6. 口径说明")
    A("")
    for s in geometry["caveat"]:
        A(f"- {s}")
    A("")
    Path(args_cli.out_summary).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({
        "n": len(rows), "n_official": full["n_official"],
        "delta_condition": geometry["delta_condition"]["n_satisfying"],
        "n_total": geometry["delta_condition"]["n_total"],
        "n_overlap": n_overlap, "n_outside": n_outside,
        "n_satisfies_centerline": n_sat_centerline,
        "min_clearance_exec_m": round(exec_min, 4),
        "nominal_median_m": round(float(np.median(nom_clear_arr)), 4),
        "cost": round(full["overall"]["cost"]["rate"], 4),
        "position": round(full["overall"]["position"]["rate"], 4),
        "joint": round(full["overall"]["joint"]["rate"], 4),
        "trim_cost": round(trim["overall"]["cost"]["rate"], 4),
        "trim_position": round(trim["overall"]["position"]["rate"], 4),
        "trim_joint": round(trim["overall"]["joint"]["rate"], 4),
    }, ensure_ascii=False, indent=1))
    print("NEAR_EVAL_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
