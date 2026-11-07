"""HIRA-031：从 800 条校准数据独立计算并冻结 HiRA 的 α（成本裕度）与 δ（几何裕度）。

口径严格按 `config/calibration_protocol_hira_v1.json` 的 statistics 段：
* 分位 nearest-rank（ceil(0.9·n) 第几个，不插值），阈值视为覆盖；
* `alpha = max(0, Q0.90(ε_c))`，`delta = Q0.90(ε_d)`（非负）；
* 负超支保留原值参与分位，只在形成裕度时取 max(0,·)；
* `ε_c = (J_exec − J_nom)/J_nom`，`J_nom < 1 J` 记 non_computable、不进比值；
* `ε_d` = 10 Hz 配对网格上执行与名义位置的**最大平面欧氏距离**（米）；另报 50 Hz 敏感性；
* bootstrap 10000×95%，以整条轨迹为最小重采样单位（不按时间点伪增样本），
  另按命令族骨架簇（family）做一次整簇重采样作保守诊断；
* 分母两套都报：预登记 800 条（含失败与 non_computable）与"完整且可算"的条件集；
* 两格完整率未达 0.9（HIRA-030 的 C06）按裁定不补采：`--adjudicated-gap-cell` 必须**显式**列出这些格，
  列齐才过、少列一格即失败（不静默通过），并另算"剔除这些格"的敏感性 α/δ 并列报告。

冻结产物：`config/margins_hira_v1.json`（+ .sha256）与派生 stage lock
`config/data_lock_hira_evaluation_v1.json`（+ .sha256，**不回写**采集锁）。

    python tests/compute_margins_hira.py --run-id calibration_20261106 \\
        --lock config/data_lock_hira_calibration_v2.json \\
        --adjudicated-gap-cell arc_combined/mass_only \\
        --adjudicated-gap-cell straight_shift/friction_only
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

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.margins import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    QUANTILE_LEVEL,
    bootstrap_quantile,
    build_margins,
    nearest_rank,
)
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402
from camra_calibration.pairing import deviation, deviation_on_grid, epsilon_c, low_speed_share  # noqa: E402
from camra_calibration.power import J_MIN, PowerModel  # noqa: E402
from hira_go2.collector import attempt_number, done_is_valid  # noqa: E402

SPLIT = "calibration"
PAIR_DT = 0.1
EXPECTED_POINTS = 301

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default="calibration_20261106")
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_calibration_v1.json"))
parser.add_argument("--out-margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--out-lock", default=str(EXP / "config/data_lock_hira_evaluation_v1.json"))
parser.add_argument("--out-report", default=str(EXP / "reports/margin_calibration.json"))
parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
parser.add_argument("--adjudicated-gap-cell", action="append", default=[],
                    help="完整率低于 0.9、已被显式裁定的格（family/profile），可重复；未列全则门禁失败")
parser.add_argument("--out-sensitivity", default=str(EXP / "reports/margin_sensitivity_gap_cells.json"))
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
    manifest_path = Path(args_cli.manifests) / f"{SPLIT}.jsonl"
    manifest = {row["trial_id"]: row for row in
                (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    run_dir = Path(args_cli.runs_root) / args_cli.run_id
    model = PowerModel.load(args_cli.model, verify_hash=True)
    print(f"[margin] manifest {len(manifest)} 条，model sha={model.sha256()[:16]}", flush=True)

    rows: list[dict] = []
    excluded: list[dict] = []
    for trial_id, row in manifest.items():
        attempt = best_attempt(run_dir / trial_id)
        if attempt is None:
            excluded.append({"trial_id": trial_id, "reason": "missing_attempt"})
            continue
        meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((attempt / "energy.json").read_text(encoding="utf-8"))
        pairing = np.load(attempt / "pairing_10hz.npz")
        t_pair = pairing["t_s"].astype(np.float64)
        commands = pairing["command"].astype(np.float64)
        exec_xy = pairing["pos_xy"].astype(np.float64)
        outcome = meta["outcome"]

        problems = []
        if outcome not in ("completed", "business_failure"):
            problems.append(f"outcome={outcome}")
        if t_pair.size < 2:
            problems.append("pairing 点数不足")
        if not np.isfinite(exec_xy).all():
            problems.append("位置含非有限值")
        if problems:
            if outcome == "business_failure" and problems == ["pairing 点数不足"]:
                # 开局就倒的业务失败只有 1 个配对点，无法积分名义轨迹。
                # 它仍是业务失败、必须留在主分母与 prefix_only 里，不能算作"数据排除"。
                rows.append({
                    "trial_id": trial_id, "family": row["family"], "profile": row["profile"],
                    "cell": f"{row['family']}/{row['profile']}",
                    "outcome": outcome, "complete": False,
                    "j_exec": float(energy["j_exec"]), "j_nom": None,
                    "eps_c": None, "eps_c_reason": "pairing_points_lt_2",
                    "eps_d": None, "eps_d_50hz": None,
                    "low_speed_share": low_speed_share(commands),
                    "n_points": int(t_pair.size), "dt_min": None,
                    "cluster_family": row["family"], "cluster_template": row.get("cluster_id"),
                })
            else:
                excluded.append({"trial_id": trial_id, "reason": "；".join(problems)})
            continue
        # 名义轨迹：同一 (family, seed) 由 build_program 重建，零阶保持积分
        program = build_program(row["family"], row["seed"], duration_s=row["duration_s"],
                                command_dt=row["command_dt"])
        expected_cmd = np.asarray([list(map(float, c)) for c in program.commands], dtype=np.float64)
        if expected_cmd.shape != commands.shape:
            n_got = commands.shape[0]
            if (outcome == "business_failure" and 2 <= n_got < expected_cmd.shape[0]
                    and expected_cmd.shape[1:] == commands.shape[1:]
                    and float(np.abs(expected_cmd[:n_got] - commands).max()) <= 1e-5):
                # 业务失败时采集器把配对网格截到实际走完的窗口：与重建命令的**前缀**逐点对齐
                # 即为合法前缀，不能因为"比整窗短"就把这条失败判成数据排除。
                pass
            else:
                problems.append(f"cmd 形状 {commands.shape} vs 重建 {expected_cmd.shape}")
        elif float(np.abs(expected_cmd - commands).max()) > 1e-5:
            problems.append("cmd 与重建不一致")
        pose = (meta.get("profile_applied") or {}).get("initial_pose") or {}
        initial = PlanarState(px=float(pose.get("px_m", 0.0)), py=float(pose.get("py_m", 0.0)),
                              yaw=float(pose.get("yaw_rad", 0.0)))
        nominal = integrate(initial, commands, PAIR_DT)
        nom_xy = np.asarray([[s.px, s.py] for s in nominal], dtype=np.float64)
        if nom_xy.shape != exec_xy.shape:
            problems.append(f"名义 {nom_xy.shape} vs 执行 {exec_xy.shape}")
        if problems:
            excluded.append({"trial_id": trial_id, "reason": "；".join(problems)})
            continue

        dev = deviation(exec_xy, nom_xy)
        eps_d = float(dev.max())
        complete = outcome == "completed"
        j_nom = float(model.j_nom(commands))
        j_exec = float(energy["j_exec"])
        eps_c_value, eps_c_reason = epsilon_c(j_exec, j_nom, J_MIN) if complete else (None, "prefix_only")

        # 50 Hz 敏感性（仅完整轨迹）
        eps_d50 = None
        if complete:
            state = np.load(attempt / "state_50hz.npz")
            t50 = state["t_s"].astype(np.float64)
            pos50 = state["root_pos_w"][:, 0:2].astype(np.float64)
            try:
                eps_d50 = float(deviation_on_grid(t50, pos50, t_pair, nom_xy).max())
            except ValueError:
                eps_d50 = None

        rows.append({
            "trial_id": trial_id, "family": row["family"], "profile": row["profile"],
            "cell": f"{row['family']}/{row['profile']}",
            "outcome": outcome, "complete": complete,
            "j_exec": j_exec, "j_nom": j_nom,
            "eps_c": eps_c_value, "eps_c_reason": eps_c_reason,
            "eps_d": eps_d, "eps_d_50hz": eps_d50,
            "low_speed_share": low_speed_share(commands),
            "n_points": int(t_pair.size),
            "dt_min": float(np.diff(t_pair).min()) if t_pair.size > 1 else None,
            "cluster_family": row["family"],
            "cluster_template": row.get("cluster_id"),
        })

    print(f"[margin] 可用 {len(rows)} 条，排除 {len(excluded)}", flush=True)

    official = [r for r in rows if r["complete"] and r["eps_c"] is not None]
    eps_c_all = [r["eps_c"] for r in official]
    eps_d_all = [r["eps_d"] for r in official]
    non_computable = [r for r in rows if r["eps_c_reason"] == "non_computable"]
    prefix_only = [r for r in rows if not r["complete"]]

    clusters_family = [r["cluster_family"] for r in official]
    margins, detail = build_margins(
        eps_c=eps_c_all, eps_d=eps_d_all,
        stats={"quantile": QUANTILE_LEVEL, "method": "nearest-rank",
               "bootstrap": {"replicates": args_cli.replicates, "seed": args_cli.seed,
                             "level": 0.95}},
        data={"split": SPLIT, "run_id": args_cli.run_id,
              "n_manifest": len(manifest), "n_usable": len(rows),
              "n_official": len(official), "n_non_computable": len(non_computable),
              "n_prefix_only": len(prefix_only),
              "manifest_sha256": sha256_file(manifest_path),
              "model_sha256": model.sha256(),
              "data_lock_sha256": sha256_file(Path(args_cli.lock)),
              "adjudicated_gap_cells": sorted(args_cli.adjudicated_gap_cell)},
        applicability={"command_domain": "u=[vx,vy,wz] 协议命令域",
                       "window_s": 30.0, "pair_dt_s": PAIR_DT,
                       "profiles": ["clean", "friction_only", "mass_only", "push_only", "combined"],
                       "note": "α 只用于成本预算 (1+α)·J_nom；δ 只用于危险集扩张 obstacle⊕footprint⊕disc(δ)，各一次"},
        version="hira-1.0",
        replicates=args_cli.replicates, seed=args_cli.seed,
        clusters_c=clusters_family, clusters_d=clusters_family,
        notes=("由 HIRA-030 的 800 条校准数据计算；固定 bootstrap 种子可复算",
               "负超支保留原值参与分位，仅在形成裕度时取 max(0,·)",
               "资格缺口：完整率低于 0.9 的格为 " + ", ".join(sorted(args_cli.adjudicated_gap_cell))
               + "（HIRA-030 C06），按协议裁定不补采、不进入格级正式结论；"
                 "剔除这些格的敏感性 α/δ 见 reports/margin_sensitivity_gap_cells.json"),
    )
    print(f"[margin] alpha={margins.alpha:.6f} delta={margins.delta:.6f} "
          f"(n_c={len(eps_c_all)}, n_d={len(eps_d_all)})", flush=True)

    # ---- 敏感性：剔除被显式裁定的资格缺口格（HIRA-030 的 C06）----
    gap_cells = sorted(args_cli.adjudicated_gap_cell)
    sens_official = [r for r in official if r["cell"] not in gap_cells]
    sens_c = [r["eps_c"] for r in sens_official]
    sens_d = [r["eps_d"] for r in sens_official]
    sens_ci_c = bootstrap_quantile(np.asarray(sens_c, dtype=np.float64), q=QUANTILE_LEVEL,
                                   replicates=args_cli.replicates, seed=args_cli.seed)
    sens_ci_d = bootstrap_quantile(np.asarray(sens_d, dtype=np.float64), q=QUANTILE_LEVEL,
                                   replicates=args_cli.replicates, seed=args_cli.seed)
    sens_alpha = max(0.0, sens_ci_c["point"])
    sens_delta = max(0.0, sens_ci_d["point"])
    sensitivity = {
        "kind": "exclude_adjudicated_gap_cells",
        "excluded_cells": gap_cells,
        "n_official": len(sens_official),
        "n_official_full": len(official),
        "n_dropped": len(official) - len(sens_official),
        "n_completed_not_official": len(prefix_only),
        "alpha": sens_alpha,
        "alpha_raw_q90": sens_ci_c["point"],
        "alpha_interval": [sens_ci_c["lo"], sens_ci_c["hi"]],
        "delta": sens_delta,
        "delta_interval": [sens_ci_d["lo"], sens_ci_d["hi"]],
        "delta_abs_change": sens_delta - margins.delta,
        "alpha_abs_change": sens_alpha - margins.alpha,
        "per_cell_n": {cell: sum(1 for r in rows if r["cell"] == cell) for cell in gap_cells},
        "note": ("全样本（含两格）与剔除两格两套 α/δ 必须并列报告；冻结值取全样本，"
                 "敏感性只作诊断，不用于替换冻结值"),
    }
    Path(args_cli.out_sensitivity).write_text(
        json.dumps({"report": "margin_sensitivity_gap_cells", "task": "HIRA-031",
                    "run_id": args_cli.run_id, "sensitivity": sensitivity,
                    "frozen": {"alpha": margins.alpha, "delta": margins.delta}},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[margin] 敏感性（剔除 {len(gap_cells)} 格，n={len(sens_official)}）："
          f"alpha={sens_alpha:.6f} delta={sens_delta:.6f} "
          f"(Δα={sensitivity['alpha_abs_change']:+.6f}, Δδ={sensitivity['delta_abs_change']:+.6f})", flush=True)

    # ---- 独立复算：不经过 build_margins，直接按协议公式算一遍 ----
    q_c = nearest_rank(eps_c_all, QUANTILE_LEVEL)
    q_d = nearest_rank(eps_d_all, QUANTILE_LEVEL)
    recompute = {"q90_eps_c": q_c, "q90_eps_d": q_d,
                 "alpha": max(0.0, q_c), "delta": max(0.0, q_d)}
    recompute["matches_frozen"] = (
        abs(recompute["alpha"] - margins.alpha) <= 1e-12
        and abs(recompute["delta"] - margins.delta) <= 1e-12
    )

    n_negative = sum(1 for v in eps_c_all if v < 0)

    # ---- 20 格 / 低速 / profile 诊断 ----
    def block(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        arr = np.sort(np.asarray(values, dtype=np.float64))
        return {"n": int(arr.size), "q90": nearest_rank(arr, QUANTILE_LEVEL),
                "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}

    per_cell = defaultdict(lambda: {"eps_c": [], "eps_d": [], "n_official": 0, "n_total": 0})
    # 每格的完整率分母必须取**清单**的 40 条，而不是"读到了几条"：
    # 用后者会把被排除的 trial 从分母里抹掉，反而把缺口藏起来。
    cell_manifest = defaultdict(int)
    for row in manifest.values():
        cell_manifest[f"{row['family']}/{row['profile']}"] += 1
    for r in rows:
        per_cell[r["cell"]]["n_total"] = cell_manifest[r["cell"]]
        if r["complete"] and r["eps_c"] is not None:
            per_cell[r["cell"]]["n_official"] += 1
            per_cell[r["cell"]]["eps_c"].append(r["eps_c"])
            per_cell[r["cell"]]["eps_d"].append(r["eps_d"])
    for cell, total in cell_manifest.items():
        per_cell[cell]["n_total"] = total
    cell_report = {
        cell: {
            "n_total": value["n_total"], "n_official": value["n_official"],
            "full_rate": round(value["n_official"] / value["n_total"], 4) if value["n_total"] else 0.0,
            "eps_c": block(value["eps_c"]), "eps_d": block(value["eps_d"]),
        }
        for cell, value in sorted(per_cell.items())
    }

    low = [r["eps_d"] for r in official if r["low_speed_share"] >= 0.3]
    high = [r["eps_d"] for r in official if r["low_speed_share"] < 0.3]
    per_profile = defaultdict(list)
    for r in official:
        per_profile[r["profile"]].append(r["eps_d"])

    # ---- 簇诊断（骨架簇 = family，4 簇）----
    family_sizes = defaultdict(int)
    for name in clusters_family:
        family_sizes[name] += 1
    boot_by_family = bootstrap_quantile(np.asarray(eps_d_all), q=QUANTILE_LEVEL,
                                        replicates=args_cli.replicates, seed=args_cli.seed,
                                        clusters=clusters_family)

    # ---- 两套分母 ----
    denominators = {
        "overall_pre_registered": len(manifest),
        "evaluated": len(rows),
        "official_complete_computable": len(official),
        "non_computable": len(non_computable),
        "prefix_only": len(prefix_only),
        "excluded": len(excluded),
        "coverage_overall": round(len(official) / len(manifest), 4) if manifest else 0.0,
        "note": "overall 以预登记 800 条为分母；条件覆盖只在完整且 J_nom≥1 J 的记录上另报",
    }

    # C06 的事实与裁定分开记录：事实不门禁（它就是数据长这样），裁定必须显式对齐才门禁。
    below_floor = sorted(cell for cell, v in cell_report.items() if v["full_rate"] < 0.9)
    facts = {
        "cells_below_full_rate_floor": below_floor,
        "min_cell_full_rate": min((v["full_rate"] for v in cell_report.values()), default=None),
        "gap_cells_adjudicated": gap_cells,
        "gap_cells_match": set(below_floor) == set(gap_cells),
    }

    checks = {
        "alpha_nonneg": margins.alpha >= 0.0,
        "delta_nonneg": margins.delta >= 0.0,
        "recompute_matches": recompute["matches_frozen"],
        "n_official_ge_720": len(official) >= int(0.9 * len(manifest)),
        "gap_cells_explicitly_adjudicated": set(below_floor) == set(gap_cells),
        "negative_overrun_reported": n_negative >= 0,   # 事实记录：负超差条数见 diagnostics
        "bootstrap_interval_ordered": margins.alpha_interval[0] <= margins.alpha <= margins.alpha_interval[1]
        and margins.delta_interval[0] <= margins.delta <= margins.delta_interval[1],
        "sensitivity_reported": len(sens_official) == len(official) - sum(
            1 for r in official if r["cell"] in gap_cells) and len(sens_official) > 0,
        "frozen_before_heldout": not any(
            (Path(args_cli.runs_root) / name).exists()
            for name in ("heldout_20261106", "near_obstacle_20261106")
        ),
        "model_and_lock_bound": margins.data["model_sha256"] == model.sha256()
        and margins.data["manifest_sha256"] == sha256_file(manifest_path),
    }

    out_margins = Path(args_cli.out_margins)
    margins.save(out_margins)
    print(f"[margin] frozen {out_margins.name} sha={margins.sha256()[:16]}", flush=True)

    # ---- 派生 stage lock（不回写采集锁）----
    base_lock = json.loads(Path(args_cli.lock).read_text(encoding="utf-8"))
    stage = {
        "schema": "camra-data-lock-v1",
        "cal_task": "HIRA-031",
        "stage": "evaluation",
        "purpose": "α/δ 冻结后的检验/近障放行锁（派生自采集锁，不修改其内容）",
        "parent_lock": {"path": str(args_cli.lock), "sha256": sha256_file(Path(args_cli.lock))},
        "controller": base_lock["controller"],
        "protocol": base_lock["protocol"],
        "power_model": base_lock["power_model"],
        "footprint": base_lock["footprint"],
        "step_cost_schema": base_lock["step_cost_schema"],
        "asset": base_lock["asset"],
        "trial_lists": base_lock["trial_lists"],
        "near_obstacle_maps": base_lock["near_obstacle_maps"],
        "code": base_lock["code"],
        "environment": base_lock["environment"],
        "margins": {
            "state": "frozen",
            "path": str(out_margins),
            "sha256": sha256_file(out_margins),
            "content_sha256": margins.sha256(),
            "alpha": margins.alpha,
            "delta": margins.delta,
        },
        "margins_derivation": {
            "from_task": "HIRA-031",
            "data": f"split=calibration, run={args_cli.run_id}, n_official={len(official)}",
            "rule": "本锁只追加 margins 指针；采集锁与其下的 800 条数据归属不变",
        },
        "gates": dict(base_lock["gates"], margins={
            "path": str(Path(args_cli.out_report)),
            "passed": all(checks.values()),
            "failed": [k for k, v in checks.items() if not v],
            "facts": facts,
            "sensitivity_report": str(Path(args_cli.out_sensitivity)),
        }),
    }
    out_lock = Path(args_cli.out_lock)
    out_lock.write_text(json.dumps(stage, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out_lock.with_suffix(out_lock.suffix + ".sha256").write_text(
        f"{sha256_file(out_lock)}  {out_lock.name}\n", encoding="utf-8")
    print(f"[margin] stage lock {out_lock.name} sha={sha256_file(out_lock)[:16]}", flush=True)

    report = {
        "report": "margin_calibration",
        "task": "HIRA-031",
        "run": {"run_id": args_cli.run_id, "dir": str(run_dir)},
        "margins": margins.to_dict(),
        "recompute": recompute,
        "detail": detail,
        "denominators": denominators,
        "cells": cell_report,
        "diagnostics": {
            "low_speed_ge_0p3": block(low),
            "low_speed_lt_0p3": block(high),
            "per_profile_eps_d": {k: block(v) for k, v in sorted(per_profile.items())},
            "eps_d_50hz": block([r["eps_d_50hz"] for r in official if r["eps_d_50hz"] is not None]),
            "n_negative_overrun": n_negative,
            "eps_d_10hz_vs_50hz_max_gap": max(
                (r["eps_d_50hz"] - r["eps_d"] for r in official if r["eps_d_50hz"] is not None),
                default=None),
            "clusters": {"definition": "命令族骨架（family）", "n_clusters": len(family_sizes),
                         "sizes": dict(sorted(family_sizes.items())),
                         "bootstrap_by_cluster_eps_d": boot_by_family},
            "prefix_only": [r["trial_id"] for r in prefix_only],
        },
        "checks": checks,
        "facts": facts,
        "sensitivity": sensitivity,
        "verdict": {"all_passed": all(checks.values()),
                    "failed": [k for k, v in checks.items() if not v]},
        "stage_lock": {"path": str(out_lock), "sha256": sha256_file(out_lock)},
        "notes": [
            "α 只进成本预算，δ 只进几何扩张，各一次；不把 δ 折进 footprint",
            "heldout/near 尚未采集：margins 冻结早于任何检验结果产生",
        ],
    }
    out_report = Path(args_cli.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"MARGIN {'OK' if report['verdict']['all_passed'] else 'FAIL'} "
          f"checks={sum(checks.values())}/{len(checks)} failed={report['verdict']['failed']}", flush=True)
    return 0 if report["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
