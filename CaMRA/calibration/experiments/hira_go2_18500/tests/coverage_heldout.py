"""HIRA-033：400 条 heldout 的独立覆盖率统计、区间与适用范围报告。

只读 heldout run；α/δ、功率模型与清单一律从冻结产物加载（**不重算、不调整**）。

口径严格照 `config/calibration_protocol_hira_v1.json` 的 statistics 与
statistical_acceptance.heldout 段（与 v2 的 CAL-027 同一套，只换控制器与冻结值）：

* 覆盖判定：等于阈值视为覆盖；
  成本 `J_exec <= (1+alpha)*J_nom` 即 `eps_c <= alpha`，位置 `eps_d <= delta`；
* **联合覆盖单独报告**，不用两项边际相乘顶替；另报"联合 / 边际乘积"的独立性诊断；
* 分母：清单全体（含业务失败）与"完整且可算"两套同时报告，不得只报条件覆盖；
* 区间：bootstrap 10000 x 95%，以整条轨迹为最小重采样单位，另给 cell 整簇口径；
* 失败与 non_computable 留在主分母；本脚本不删样本、不改判据、不调 α/δ。

`--adjudicated-gap-cell` 必须把 C06 实际不达标的格**列全**（HIRA-032 已裁定
arc_combined/combined、arc_combined/mass_only、start_stop_turn/mass_only 三格）：
少列或多列一格即失败，不静默通过。报告并列两套分母——全体 400 条与剔除这三格的 340 条。

    python tests/coverage_heldout.py --run-id heldout_20261106 \\
        --adjudicated-gap-cell arc_combined/combined \\
        --adjudicated-gap-cell arc_combined/mass_only \\
        --adjudicated-gap-cell start_stop_turn/mass_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
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
from camra_calibration.nominal import PlanarState, integrate  # noqa: E402
from camra_calibration.pairing import deviation, epsilon_c, low_speed_share  # noqa: E402
from camra_calibration.power import J_MIN, PowerModel  # noqa: E402
from hira_go2.collector import attempt_number, done_is_valid  # noqa: E402

SPLIT = "heldout"
PAIR_DT = 0.1
CELL_QUOTA = 20
LOW_SPEED_SHARE = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run-id", default="heldout_20261106")
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_evaluation_v1.json"))
parser.add_argument("--out", default=str(EXP / "reports/heldout_coverage.json"))
parser.add_argument("--out-summary", default=str(EXP / "reports/heldout_summary.md"))
parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
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


def load_margins(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["file_sha256"] = sha256_file(path)
    return payload


def build_rows(manifest: dict, run_dir: Path, model: PowerModel) -> tuple[list[HeldoutRow], list[dict], list[str]]:
    """把 400 个 run 目录装配成 HeldoutRow；返回 (rows, 排除明细, 缺失 trial_id)。"""
    rows: list[HeldoutRow] = []
    excluded: list[dict] = []
    missing: list[str] = []
    for trial_id, spec in manifest.items():
        attempt = best_attempt(run_dir / trial_id)
        if attempt is None:
            missing.append(trial_id)
            continue
        meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((attempt / "energy.json").read_text(encoding="utf-8"))
        pairing = np.load(attempt / "pairing_10hz.npz")
        t_pair = pairing["t_s"].astype(np.float64)
        commands = pairing["command"].astype(np.float64)
        exec_xy = pairing["pos_xy"].astype(np.float64)
        outcome = str(meta["outcome"])
        family, profile = spec["family"], spec["profile"]

        problems: list[str] = []
        if outcome not in ("completed", "business_failure"):
            problems.append(f"outcome={outcome}")
        if t_pair.size < 2:
            problems.append("pairing 点数不足")
        if not np.isfinite(exec_xy).all():
            problems.append("位置含非有限值")

        j_exec = float(energy["j_exec"])
        j_nom_calc = float(model.j_nom(commands)) if commands.shape[0] >= 2 else None
        non_computable = bool(j_nom_calc is not None and j_nom_calc < J_MIN)

        # 开局就倒的业务失败只有 1 个配对点：名义轨迹无法积分，但它是业务失败、
        # 必须留在主分母，不能算作"数据排除"（与 HIRA-031 同一处理）。
        if problems == ["pairing 点数不足"] and outcome == "business_failure":
            rows.append(HeldoutRow(
                trial_id=trial_id, family=family, profile=profile,
                cell=f"{family}/{profile}",
                direction="unknown_short", low_speed=True, outcome=outcome,
                complete=False, official=False, non_computable=non_computable,
                data_invalid=False, eps_c=None, eps_d=None))
            continue
        if problems:
            excluded.append({"trial_id": trial_id, "reason": "；".join(problems)})
            continue

        # 名义轨迹：同一 (family, seed) 由 build_program 重建，零阶保持积分
        program = build_program(family, int(spec["seed"]), duration_s=spec["duration_s"],
                                command_dt=spec["command_dt"])
        expected = np.asarray([list(map(float, c)) for c in program.commands], dtype=np.float64)
        if expected.shape != commands.shape:
            n_got = commands.shape[0]
            is_prefix = (outcome == "business_failure" and 2 <= n_got < expected.shape[0]
                         and expected.shape[1:] == commands.shape[1:]
                         and float(np.abs(expected[:n_got] - commands).max()) <= 1e-5)
            if not is_prefix:
                excluded.append({"trial_id": trial_id,
                                 "reason": f"cmd 形状 {commands.shape} vs 重建 {expected.shape}"})
                continue
        elif float(np.abs(expected - commands).max()) > 1e-5:
            excluded.append({"trial_id": trial_id, "reason": "cmd 与重建不一致"})
            continue

        pose = (meta.get("profile_applied") or {}).get("initial_pose") or {}
        initial = PlanarState(px=float(pose.get("px_m", 0.0)), py=float(pose.get("py_m", 0.0)),
                              yaw=float(pose.get("yaw_rad", 0.0)))
        nominal = integrate(initial, commands, PAIR_DT)
        nom_xy = np.asarray([[s.px, s.py] for s in nominal], dtype=np.float64)
        if nom_xy.shape != exec_xy.shape:
            excluded.append({"trial_id": trial_id,
                             "reason": f"名义 {nom_xy.shape} vs 执行 {exec_xy.shape}"})
            continue

        eps_d = float(deviation(exec_xy, nom_xy).max())
        complete = outcome == "completed"
        eps_c_value, _reason = (epsilon_c(j_exec, j_nom_calc, J_MIN) if complete else (None, "prefix_only"))

        rows.append(HeldoutRow(
            trial_id=trial_id, family=family, profile=profile, cell=f"{family}/{profile}",
            direction=direction_of(commands),
            low_speed=bool(low_speed_share(commands) >= LOW_SPEED_SHARE),
            outcome=outcome, complete=complete,
            official=bool(complete and eps_c_value is not None and np.isfinite(eps_d)),
            non_computable=non_computable, data_invalid=False,
            eps_c=eps_c_value, eps_d=eps_d))
    return rows, excluded, missing


def summarize(rows: list[HeldoutRow], alpha: float, delta: float, replicates: int,
              seed: int, expected: int, expected_per_cell: int) -> dict:
    """一套分母下的完整统计（总体 + 格 + 分组 + 判定）。"""
    official = [r for r in rows if r.official]
    cells_key = [r.cell for r in official]
    c_flags = [cost_covered(r.eps_c, alpha) for r in official]
    d_flags = [position_covered(r.eps_d, delta) for r in official]
    joint_pairs = [(a, b) for a, b in zip(c_flags, d_flags) if a is not None and b is not None]
    joint_flags = [bool(a) and bool(b) for a, b in joint_pairs]

    den = denominators(rows, expected)
    overall = {
        "cost": rate_with_ci(c_flags, cells_key, replicates, seed),
        "position": rate_with_ci(d_flags, cells_key, replicates, seed),
        "joint": rate_with_ci(joint_flags, cells_key, replicates, seed),
    }
    cells = per_cell(rows, alpha, delta, expected_per_cell)
    groups = {
        "by_profile": subgroup(rows, lambda r: r.profile, alpha, delta),
        "by_family": subgroup(rows, lambda r: r.family, alpha, delta),
        "by_low_speed": subgroup(rows, lambda r: "low_speed" if r.low_speed else "high_speed",
                                 alpha, delta),
        "by_direction": subgroup(rows, lambda r: r.direction, alpha, delta),
    }
    verdict = coverage_verdict(den, overall, cells, target=TARGET_COVERAGE)
    return {"denominators": den, "overall": overall, "per_cell": cells,
            "groups": groups, "verdict": verdict, "n_official": len(official)}


def fmt_rate(block: dict) -> str:
    if not block or block.get("rate") is None:
        return "n/a"
    ci = block.get("ci_cluster") or {}
    lo = ci.get("lo")
    hi = ci.get("hi")
    ci_txt = f" [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else ""
    return f"{block['rate']:.4f} ({block.get('n_covered')}/{block.get('n')}){ci_txt}"


def write_summary(path: Path, report: dict) -> None:
    full = report["full_sample"]
    trim = report["gap_cells_removed"]
    lines: list[str] = []
    add = lines.append
    add("# HiRA heldout 覆盖率报告（HIRA-033）")
    add("")
    add(f"- run：`{report['run']['run_id']}`，split `heldout`，清单 {report['n_manifest']} 条")
    add(f"- 冻结裕度：α = {report['alpha']:.6f}，δ = {report['delta']:.6f} m"
        f"（`margins_hira_v1.json`，sha {report['margins']['file_sha256'][:16]}）")
    add(f"- 功率模型：`power_model_hira_v1.json`，sha {report['power_model']['sha256'][:16]}")
    add(f"- 检验锁：`{report['data_lock']['file']}`，sha {report['data_lock']['sha256'][:16]}")
    add("")
    add("## 一句话结论")
    add("")
    add(report["headline"])
    add("")
    add("## 分母（两套并列，不得只报一套）")
    add("")
    add("| 口径 | 期望 | 观测 | 完整且可算 | 业务失败 | 缺失 | 数据排除 | 完整率 |")
    add("|---|---|---|---|---|---|---|---|")
    for tag, blk in (("全体 400 条", full), ("剔除 3 个资格缺口格", trim)):
        d = blk["denominators"]
        share = d.get("complete_share_overall")
        add(f"| {tag} | {d['expected']} | {d['observed']} | {d['complete']} | "
            f"{d['business_failed']} | {d.get('missing', 0)} | {d['data_invalid']} | "
            f"{share:.4f} |" if share is not None else
            f"| {tag} | {d['expected']} | {d['observed']} | {d['complete']} | n/a |")
    add("")
    add("## 总体覆盖（条件分母 = 完整且可算；区间为 cell 整簇 bootstrap 95%）")
    add("")
    add("| 口径 | 成本覆盖 | 位置覆盖 | 联合覆盖 |")
    add("|---|---|---|---|")
    for tag, blk in (("全体 400 条", full), ("剔除 3 格（340 条）", trim)):
        o = blk["overall"]
        add(f"| {tag} | {fmt_rate(o['cost'])} | {fmt_rate(o['position'])} | {fmt_rate(o['joint'])} |")
    add("")
    add("## 20 格（完整率 < 0.9 的格不进格级正式结论）")
    add("")
    add("| 格 | 完整/清单 | 完整率 | 成本 | 位置 | 联合 | 资格 |")
    add("|---|---|---|---|---|---|---|")
    for cell, v in sorted(full["per_cell"].items(), key=lambda kv: kv[1]["complete_share"] or 0):
        share = v["complete_share"] or 0.0
        tag = "不足" if cell in report["gap_cells"] else "可用"
        add(f"| `{cell}` | {v['n_complete']}/{v['n_observed']} | {share:.3f} | "
            f"{fmt_rate(v['cost'])} | {fmt_rate(v['position'])} | {fmt_rate(v['joint'])} | {tag} |")
    add("")
    add("## 分组（全体 400 条口径）")
    add("")
    for key, label in (("by_profile", "按 profile"), ("by_family", "按命令族"),
                       ("by_low_speed", "按低速占比"), ("by_direction", "按方向")):
        add(f"**{label}**")
        add("")
        add("| 组 | n | 成本 | 位置 | 联合 |")
        add("|---|---|---|---|---|")
        for grp, v in full["groups"][key].items():
            add(f"| {grp} | {v['n']} | {fmt_rate(v['cost'])} | {fmt_rate(v['position'])} | "
                f"{fmt_rate(v['joint'])} |")
        add("")
    add("## 独立性诊断")
    add("")
    v = full["verdict"]
    add(f"- 联合覆盖 / (成本 × 位置) = {v['joint_over_independence']:.4f}"
        if v["joint_over_independence"] is not None else "- 联合覆盖 / 边际乘积：n/a")
    add(f"- 判定：`{v['status']}`；`joint_explained_by_independence` = "
        f"{v['joint_explained_by_independence']}")
    add("")
    add("## 适用范围与未达标项")
    add("")
    for item in report["scope_notes"]:
        add(f"- {item}")
    add("")
    add("## 未覆盖样本")
    add("")
    add(f"- 未覆盖（成本或位置任一不达标）共 {report['uncovered']['n']} 条，前 50 条见 JSON 的 `uncovered.rows`。")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    lock = json.loads(Path(args_cli.lock).read_text(encoding="utf-8"))
    if lock.get("stage") != "evaluation":
        raise SystemExit(f"REFUSED：必须用 evaluation stage lock，得到 {lock.get('stage')!r}")
    margins = load_margins(Path(args_cli.margins))
    if margins.get("sha256") != lock["margins"]["content_sha256"]:
        raise SystemExit("REFUSED：margins 内容散列与检验锁不一致")
    alpha = float(margins["alpha"])
    delta = float(margins["delta"])
    model = PowerModel.load(args_cli.model, verify_hash=True)

    manifest_path = Path(args_cli.manifests) / f"{SPLIT}.jsonl"
    manifest = {row["trial_id"]: row for row in
                (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    run_dir = Path(args_cli.runs_root) / args_cli.run_id
    print(f"[cov] manifest {len(manifest)} 条，run {run_dir}，alpha={alpha:.6f} delta={delta:.6f}", flush=True)

    rows, excluded, missing = build_rows(manifest, run_dir, model)
    print(f"[cov] 装配 {len(rows)} 条，数据排除 {len(excluded)}，缺失 {len(missing)}", flush=True)

    full = summarize(rows, alpha, delta, args_cli.replicates, args_cli.seed,
                     expected=len(manifest), expected_per_cell=CELL_QUOTA)
    actual_gap = sorted(c for c, v in full["per_cell"].items()
                        if v["complete_share"] is not None and v["complete_share"] < 0.9)
    declared_gap = sorted(args_cli.adjudicated_gap_cell)
    if actual_gap != declared_gap:
        raise SystemExit(
            "REFUSED：C06 实际不达标的格与 --adjudicated-gap-cell 不一致（必须列全、不得多列）\n"
            f"  实际：{actual_gap}\n  声明：{declared_gap}")

    trim_rows = [r for r in rows if r.cell not in set(actual_gap)]
    trim_expected = len(manifest) - sum(1 for r in rows if r.cell in set(actual_gap))
    trim = summarize(trim_rows, alpha, delta, args_cli.replicates, args_cli.seed,
                     expected=trim_expected, expected_per_cell=CELL_QUOTA)

    official = [r for r in rows if r.official]
    cells_key = [r.cell for r in official]
    c_flags = [cost_covered(r.eps_c, alpha) for r in official]
    d_flags = [position_covered(r.eps_d, delta) for r in official]
    uncovered = [
        {"trial_id": r.trial_id, "cell": r.cell, "eps_c": r.eps_c, "eps_d": r.eps_d,
         "cost_covered": cost_covered(r.eps_c, alpha),
         "position_covered": position_covered(r.eps_d, delta)}
        for r in official
        if cost_covered(r.eps_c, alpha) is False or position_covered(r.eps_d, delta) is False
    ]

    fo = full["overall"]
    jr = full["verdict"]["joint_over_independence"]
    jr_txt = f"{jr:.4f}" if jr is not None else "n/a"
    headline = (
        f"在 {len(manifest)} 条预登记 trial 上，条件分母（完整且可算 {full['n_official']} 条）的"
        f"覆盖率为：成本 {fo['cost']['rate']:.4f}、位置 {fo['position']['rate']:.4f}、"
        f"联合 {fo['joint']['rate']:.4f}；"
        f"联合/(成本×位置) = {jr_txt}。"
        f"20 格中有 {len(actual_gap)} 格完整率低于 0.9（"
        + "、".join(f"`{c}` {full['per_cell'][c]['complete_share']:.3f}" for c in actual_gap)
        + "），按 HIRA-032 裁定不进格级正式结论；剔除这 3 格后联合覆盖率为 "
        f"{trim['overall']['joint']['rate']:.4f}。"
        "覆盖率只描述“在本次校准条件代表的执行条件下，误差落在裕度内的经验频率”，"
        "不等于导航成功率，也不构成随机安全保证。"
    )

    scope_notes = [
        "格级正式结论只覆盖 17/20 格；"
        + "、".join(f"`{c}`" for c in actual_gap)
        + " 完整率不足 0.9，仅报诊断值，不作结论（HIRA-032 裁定，沿用 HIRA-030 先例）。",
        "分母两套并列：全体 400 条（含 23 条业务失败）与剔除 3 格后的 "
        f"{trim_expected} 条；不得只引用其中一套。",
        "位置误差 ε_d 与 HIRA-031 同口径：10 Hz 配对网格上执行与名义平面的最大欧氏距离；"
        "名义轨迹由 (family, seed) 重建命令后零阶保持积分得到。",
        "本报告不用于反向调整 α/δ、功率模型或清单（协议 statistical_acceptance.heldout.may_not）。",
        "单一固定控制器与单一冻结裕度，不声称训练随机种子方差；区间只反映抽样不确定性。",
    ]

    report = {
        "report": "heldout_coverage",
        "task": "HIRA-033",
        "run": {"run_id": args_cli.run_id, "dir": str(run_dir),
                "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)},
        "alpha": alpha,
        "delta": delta,
        "margins": {"file": str(args_cli.margins), "file_sha256": margins["file_sha256"],
                    "content_sha256": margins["sha256"], "state": "frozen"},
        "power_model": {"file": str(args_cli.model), "sha256": model.sha256()},
        "data_lock": {"file": Path(args_cli.lock).name,
                      "sha256": sha256_file(Path(args_cli.lock)), "stage": lock.get("stage")},
        "n_manifest": len(manifest),
        "n_assembled": len(rows),
        "outcomes": dict(Counter(r.outcome for r in rows)),
        "excluded": excluded,
        "missing": missing,
        "gap_cells": actual_gap,
        "full_sample": full,
        "gap_cells_removed": trim,
        "uncovered": {"n": len(uncovered), "rows": uncovered[:50]},
        "headline": headline,
        "scope_notes": scope_notes,
        "notes": [
            "覆盖判定：等于阈值视为覆盖（协议 statistics.quantile_method）",
            "联合覆盖单独报告；另给联合/边际乘积作独立性诊断，但不得用乘积顶替观测联合",
            "区间以整条轨迹为最小重采样单位，另给 cell 整簇口径；不按 50 Hz 点计数",
            "业务失败与 non_computable 留在主分母，不删样本、不改判据",
        ],
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_summary(Path(args_cli.out_summary), report)

    print(json.dumps({
        "n_manifest": len(manifest), "n_official": full["n_official"],
        "alpha": round(alpha, 6), "delta": round(delta, 6),
        "full": {k: (round(fo[k]["rate"], 4) if fo[k]["rate"] is not None else None)
                 for k in ("cost", "position", "joint")},
        "trim": {k: (round(trim["overall"][k]["rate"], 4)
                     if trim["overall"][k]["rate"] is not None else None)
                 for k in ("cost", "position", "joint")},
        "gap_cells": actual_gap,
        "verdict_full": full["verdict"]["status"],
        "joint_over_independence": full["verdict"]["joint_over_independence"],
        "uncovered": len(uncovered),
    }, ensure_ascii=False, indent=1))
    print(f"HELDOUT_COVERAGE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
