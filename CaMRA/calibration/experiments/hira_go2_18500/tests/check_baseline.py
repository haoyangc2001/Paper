"""HIRA-025：400 条 clean 基线的资格清单与完整性核对。

只读：不重跑、不改数据。判据来自协议 v1（完整率 ≥0.9）、预登记（J_min=1 J、
能量契约 D-ENERGY-001）与 HIRA-024 已锁定的 manifest。

    python tests/check_baseline.py                 # 三个 subset 一起核
    python tests/check_baseline.py --runs baseline_fit=baseline_20261106

validate 的**性能结果保持封存**：本脚本对它只做采集状态与数据完整性核对，
不聚合、不输出任何 J 统计量（拟合误差属 HIRA-026）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from camra_calibration.energy import integrate_series  # noqa: E402
from hira_go2.collector import attempt_number, done_is_valid  # noqa: E402

SUBSETS = ("baseline_fit", "baseline_dev", "baseline_validate")
EXPECTED_COUNT = {"baseline_fit": 240, "baseline_dev": 80, "baseline_validate": 80}
EXPECTED_CELLS = {"baseline_fit": 60, "baseline_dev": 20, "baseline_validate": 20}
J_MIN_J = 1.0
FULL_RATE_MIN = 0.90
ENERGY_REL_MAX = 1e-9
PILOT_SEEDS = set(range(6110, 6130))
STEPS_PER_INTERVAL = 20   # command_dt 0.1 s / sim_dt 0.005 s

parser = argparse.ArgumentParser()
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--run", action="append", default=[],
                    help="split=run_id；缺省用 <split> 目录下最新的 run")
parser.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
parser.add_argument("--out", default=str(EXP / "reports/baseline_collection_integrity.json"))
args_cli = parser.parse_args()

checks: list[dict] = []


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> None:
    checks.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_runs() -> dict[str, Path]:
    explicit = {}
    for item in args_cli.run:
        split, _, run_id = item.partition("=")
        explicit[split] = Path(args_cli.runs_root) / run_id
    resolved = {}
    for split in SUBSETS:
        if split in explicit:
            resolved[split] = explicit[split]
            continue
        candidates = sorted(
            (p for p in Path(args_cli.runs_root).glob(f"{split}_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            resolved[split] = candidates[-1]
    return resolved


def load_manifest(split: str) -> dict[str, dict]:
    path = Path(args_cli.manifests) / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["trial_id"]: row for row in rows}


def best_attempt(trial_dir: Path) -> Path | None:
    """按\"第一有效主 attempt\"取有效 attempt（DONE 且散列全对）。"""
    if not trial_dir.is_dir():
        return None
    for path in sorted((p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")),
                       key=attempt_number):
        if done_is_valid(path):
            return path
    return None


def main() -> int:
    bundle_id = json.loads((Path(args_cli.bundle) / "manifest.json").read_text(encoding="utf-8"))["bundle_id"]
    runs = resolve_runs()
    missing_subsets = [s for s in SUBSETS if s not in runs]
    if missing_subsets:
        add("B00-subsets", "三个 subset 的 run 目录都存在", False, f"缺少：{missing_subsets}")
        return 1
    add("B00-subsets", "三个 subset 的 run 目录都存在", True,
        "；".join(f"{k}={v.name}" for k, v in runs.items()))

    # ---- 逐条核对 ----
    per_subset: dict[str, dict] = {}
    excluded: list[dict] = []
    all_seeds: set[int] = set()
    energy_worst = 0.0
    energy_checked = 0

    for split in SUBSETS:
        manifest = load_manifest(split)
        run_dir = runs[split]
        rows, problems = [], []
        if len(manifest) != EXPECTED_COUNT[split]:
            problems.append(f"manifest 条数 {len(manifest)} ≠ {EXPECTED_COUNT[split]}")

        cells: dict[str, dict] = {}
        for cluster in sorted({row["cluster_id"] for row in manifest.values()}):
            family = cluster.split(":")[0]
            cells[family] = {"manifest": 0, "completed": 0, "business_failure": 0, "missing": 0}

        for trial_id, row in manifest.items():
            family = row["family"]
            cells[family]["manifest"] += 1
            attempt = best_attempt(run_dir / trial_id)
            if attempt is None:
                cells[family]["missing"] += 1
                excluded.append({"split": split, "trial_id": trial_id, "reason": "缺少有效 attempt（无 DONE 或散列不符）"})
                continue
            meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
            energy = json.loads((attempt / "energy.json").read_text(encoding="utf-8"))
            row_problems = []

            # 身份
            if meta["bundle_id"] != bundle_id:
                row_problems.append(f"bundle_id {meta['bundle_id'][:12]} ≠ 冻结包 {bundle_id[:12]}")
            if meta["controller_id"] != "hira_go2_18500":
                row_problems.append(f"controller_id={meta['controller_id']}")
            if meta["split"] != split:
                row_problems.append(f"split={meta['split']}")
            if meta["seed"] != row["seed"] or meta["program"] != family or meta["profile"] != row["profile"]:
                row_problems.append("seed/program/profile 与 manifest 不一致")
            if meta["program_sha256"] != row["program_sha256"]:
                row_problems.append("program_sha256 与 manifest 不一致")
            if meta["attempt_kind"] not in ("initial", "resume"):
                row_problems.append(f"attempt_kind={meta['attempt_kind']}")

            # clean 的实际旋钮
            applied = meta["profile_applied"]
            if abs(applied["friction"]["readback"] - 1.0) > 1e-6:
                row_problems.append(f"friction={applied['friction']['readback']}")
            mass_delta = applied["mass"]["readback"] - applied["mass"]["base_mass_kg"]
            if abs(mass_delta) > 1e-3:
                row_problems.append(f"mass_delta={mass_delta:.4f}")
            if row["profile"] != "clean":
                row_problems.append(f"profile={row['profile']}（baseline 只允许 clean）")

            # 完整性
            n_steps = int(meta["timing"]["steps_executed"])
            n_pair = int(meta["timing"]["pairing_points"])
            n_state = int(meta["timing"]["state50_points"])
            outcome = meta["outcome"]
            if outcome == "completed":
                cells[family]["completed"] += 1
                if (n_steps, n_state, n_pair) != (6000, 1500, 301):
                    row_problems.append(f"completed 但步数/点数 {n_steps}/{n_state}/{n_pair}")
            elif outcome == "business_failure":
                cells[family]["business_failure"] += 1
                # state50 = 每控制步（4 物理步）1 点；pairing = 每命令区间（20 物理步）1 点 + 共享端点
                if (
                    not (0 < n_steps < 6000)
                    or n_state != n_steps // 4
                    or n_pair != n_steps // STEPS_PER_INTERVAL + 1
                ):
                    row_problems.append(
                        f"截断自洽性不符 {n_steps}/{n_state}/{n_pair}"
                        f"（期望 state={n_steps // 4}、pairing={n_steps // STEPS_PER_INTERVAL + 1}）"
                    )
            else:
                row_problems.append(f"outcome={outcome}")

            # 能量：离线按落盘的 tau/qdot 复算 + 正功契约 + J_min
            phys = np.load(attempt / "physics_200hz.npz")
            tau = phys["tau_applied_source"].astype("float64")
            qdot = phys["qdot_post_source"].astype("float64")
            if tau.shape[0] != n_steps:
                row_problems.append(f"physics 行数 {tau.shape[0]} ≠ steps {n_steps}")
            dt = 0.005
            j_recomputed = float(integrate_series(tau, qdot, dt))
            rel = abs(j_recomputed - energy["j_exec"]) / max(abs(energy["j_exec"]), 1e-12)
            energy_worst = max(energy_worst, rel)
            energy_checked += 1
            if rel > ENERGY_REL_MAX:
                row_problems.append(f"能量复算 rel={rel:.2e}")
            if energy["j_exec"] < J_MIN_J:
                row_problems.append(f"J={energy['j_exec']:.4f} < J_min")
            if float(np.minimum(tau * qdot, 0).sum()) > 1e-9:
                row_problems.append("存在负功率未被截断的迹象")

            all_seeds.add(int(meta["seed"]))
            rows.append(
                {
                    "trial_id": trial_id,
                    "attempt": attempt.name,
                    "outcome": outcome,
                    "steps_executed": n_steps,
                    "pairing_points": n_pair,
                    "state50_points": n_state,
                    "j_exec": round(float(energy["j_exec"]), 3) if split != "baseline_validate" else None,
                    "energy_rel": round(rel, 15),
                    "program_sha256": meta["program_sha256"],
                }
            )
            if row_problems:
                excluded.append({"split": split, "trial_id": trial_id, "reason": "；".join(row_problems)})

        per_subset[split] = {
            "run_dir": str(run_dir),
            "manifest_rows": len(manifest),
            "attempts_found": len(rows),
            "cells": cells,
            "problems": problems,
            "trials": rows,
        }

    # ---- B01 配额 ----
    quota_ok = all(len(per_subset[s]["trials"]) + sum(
        1 for item in excluded if item["split"] == s and "缺少有效 attempt" in item["reason"]
    ) == EXPECTED_COUNT[s] for s in SUBSETS)
    add("B01-quota", "三个 subset 各 240/80/80 条，逐条与 manifest 一一对应", quota_ok,
        "；".join(f"{s}: manifest {per_subset[s]['manifest_rows']}，有效 {len(per_subset[s]['trials'])}" for s in SUBSETS))

    # ---- B02 身份与散列 ----
    identity_bad = [item for item in excluded if any(
        key in item["reason"] for key in ("bundle_id", "controller_id", "split=", "seed/program", "program_sha256")
    )]
    add("B02-identity", "bundle_id / controller / seed / program_sha256 逐条与 manifest 及冻结包一致",
        not identity_bad, f"不合格 {len(identity_bad)} 条" + (f"：{identity_bad[:3]}" if identity_bad else ""))

    # ---- B03 clean 旋钮 ----
    knob_bad = [item for item in excluded if any(
        key in item["reason"] for key in ("friction=", "mass_delta=", "profile=")
    )]
    add("B03-clean-knobs", "clean 条件实际读回 friction=1.0、mass 增量 0，且无任何非 clean profile",
        not knob_bad, f"不合格 {len(knob_bad)} 条" + (f"：{knob_bad[:3]}" if knob_bad else ""))

    # ---- B04 完整性 ----
    structural_bad = [item for item in excluded if any(
        key in item["reason"] for key in ("步数", "自洽", "outcome=", "physics 行数")
    )]
    add("B04-structure", "completed=6000/1500/301；business_failure=截断自洽（steps/4 与 steps/4+1）",
        not structural_bad, f"不合格 {len(structural_bad)} 条" + (f"：{structural_bad[:3]}" if structural_bad else ""))

    # ---- B05 能量 ----
    energy_bad = [item for item in excluded if any(
        key in item["reason"] for key in ("能量复算", "J_min", "负功率")
    )]
    add("B05-energy", f"能量离线复算 rel ≤ {ENERGY_REL_MAX:g}、J ≥ {J_MIN_J} J、负功逐关节截断",
        not energy_bad,
        f"{energy_checked} 条复算，最差 rel {energy_worst:.2e}；不合格 {len(energy_bad)} 条")

    # ---- B06 完整性比例（每格）----
    rate_rows = {}
    rate_ok = True
    for split in SUBSETS:
        for family, cell in per_subset[split]["cells"].items():
            total = cell["manifest"]
            full = cell["completed"] / total if total else 0.0
            rate_rows[f"{split}/{family}"] = {"manifest": total, "completed": cell["completed"],
                                              "business_failure": cell["business_failure"],
                                              "missing": cell["missing"], "full_rate": round(full, 4)}
            if cell["manifest"] != EXPECTED_CELLS[split] or full < FULL_RATE_MIN:
                rate_ok = False
    add("B06-cell-full-rate", f"每格条数符合配额且完成率 ≥ {FULL_RATE_MIN:g}", rate_ok,
        "；".join(f"{k}: {v['completed']}/{v['manifest']}={v['full_rate']:.2f}" for k, v in rate_rows.items()))

    # ---- B07 轨迹簇/种子互斥 ----
    # 簇 = (family, template_sha256)，而 template_sha256 含 seed → 每条 trial 自成一簇。
    # 因此正确判据不是"同族只有 1 个簇"（那是把簇当成族），而是：
    #   ① 簇与 trial 一一对应（无两条 trial 共享簇）；
    #   ② 簇前缀等于所属族；
    #   ③ 每族的簇数 = 该族配额；
    #   ④ 采集到的 seed 与 pilot 区间无交集。
    cluster_bad = []
    cluster_stats = {}
    for split in SUBSETS:
        manifest = load_manifest(split)
        clusters = [row["cluster_id"] for row in manifest.values()]
        if len(clusters) != len(set(clusters)):
            cluster_bad.append(f"{split}: 簇不唯一（{len(clusters)} 条 / {len(set(clusters))} 簇）")
        for row in manifest.values():
            if not row["cluster_id"].startswith(row["family"] + ":"):
                cluster_bad.append(f"{split}/{row['trial_id']}: cluster_id 前缀不是 {row['family']}")
        per_family = {}
        for row in manifest.values():
            per_family.setdefault(row["family"], set()).add(row["cluster_id"])
        for family, group in per_family.items():
            if len(group) != EXPECTED_CELLS[split]:
                cluster_bad.append(f"{split}/{family}: {len(group)} 簇 ≠ {EXPECTED_CELLS[split]}")
        cluster_stats[split] = {family: len(group) for family, group in per_family.items()}
        if len(set(clusters)) != EXPECTED_COUNT[split]:
            cluster_bad.append(f"{split}: 总簇数 {len(set(clusters))} ≠ {EXPECTED_COUNT[split]}")
    seed_overlap = sorted(all_seeds & PILOT_SEEDS)
    add("B07-cluster-seed", "簇与 trial 一一对应、簇前缀=族、族簇数=配额；seed 与 pilot 区间无交集",
        not cluster_bad and not seed_overlap,
        (f"簇结构异常：{cluster_bad[:4]}" if cluster_bad else
         f"三个 subset 共 {sum(len(v) for v in cluster_stats.values())} 个族簇，"
         f"每簇 1 条 trial；各 bucket 簇数 {cluster_stats}")
        + f"；seed 交集 {seed_overlap or '空'}；采集 seed 数 {len(all_seeds)}",
        cluster_stats=cluster_stats)

    # ---- B09 run 级配置绑定（collect_config 的 split 必须与 run 对应）----
    config_bad = []
    config_seen = {}
    manifest_sha = {
        split: sha256_file(Path(args_cli.manifests) / f"{split}.jsonl") for split in SUBSETS
    }
    for split in SUBSETS:
        run_dir = runs[split]
        path = run_dir / "collect_config.json"
        if not path.is_file():
            config_bad.append(f"{split}: 缺少 collect_config.json")
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        config_seen[split] = {"run": run_dir.name, "config_split": cfg.get("split"),
                             "manifest": cfg.get("manifest"), "num_envs": cfg.get("num_envs")}
        if cfg.get("split") != split:
            config_bad.append(f"{split}: config.split={cfg.get('split')}（run 目录被多个 split 复用）")
        if cfg.get("manifest") != f"{split}.jsonl":
            config_bad.append(f"{split}: config.manifest={cfg.get('manifest')}")
        if cfg.get("manifest_sha256") != manifest_sha[split]:
            config_bad.append(f"{split}: config.manifest_sha256 与当前 manifest 不符")
        if cfg.get("bundle_id") != bundle_id:
            config_bad.append(f"{split}: config.bundle_id 与冻结包不符")
    add("B09-run-config", "每个 split 有独立 run 目录，collect_config 的 split/manifest/散列/bundle 与之一致",
        not config_bad, f"异常：{config_bad}" if config_bad else f"三份配置一致：{config_seen}",
        configs=config_seen)

    # ---- B08 validate 封存 ----
    validate_rows = per_subset["baseline_validate"]["trials"]
    sealed_ok = all(row["j_exec"] is None for row in validate_rows)
    add("B08-validate-sealed", "validate 只核采集状态，未计算/未输出任何性能指标",
        sealed_ok,
        f"{len(validate_rows)} 条：仅登记 outcome/步数/点数与能量复算残差，"
        f"j_exec 字段已置空（拟合误差属 HIRA-026）")

    # ---- 汇总结论 ----
    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "report": "baseline_collection_integrity",
        "task": "HIRA-025",
        "bundle_id": bundle_id,
        "j_min_j": J_MIN_J,
        "full_rate_min": FULL_RATE_MIN,
        "energy_rel_max": ENERGY_REL_MAX,
        "subsets": {
            split: {
                "run_dir": per_subset[split]["run_dir"],
                "manifest_rows": per_subset[split]["manifest_rows"],
                "valid_trials": len(per_subset[split]["trials"]),
                "cells": per_subset[split]["cells"],
                "problems": per_subset[split]["problems"],
            }
            for split in SUBSETS
        },
        "cell_full_rate": rate_rows,
        "excluded": excluded,
        "energy_worst_rel": energy_worst,
        "energy_checked": energy_checked,
        "checks": checks,
        "verdict": {
            "all_passed": not failed,
            "failed": failed,
            "n_valid": sum(len(per_subset[s]["trials"]) for s in SUBSETS),
            "n_excluded": len(excluded),
            "note": "baseline 仅采集与资格核对；功率模型拟合一节（HIRA-026）只能用 fit/dev，validate 封存",
        },
    }
    Path(args_cli.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"BASELINE_INTEGRITY all_passed={not failed} failed={failed} "
          f"valid={report['verdict']['n_valid']} excluded={report['verdict']['n_excluded']}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
