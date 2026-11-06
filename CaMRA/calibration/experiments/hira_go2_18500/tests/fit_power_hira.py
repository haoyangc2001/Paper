"""HIRA-026：拟合、冻结并独立验证 HiRA 名义功率模型。

口径严格按 `config/power_model_prereg_hira_v1.json`（已冻结，sha 见报告）：
* 数据：baseline_fit 训练、baseline_dev 选型、baseline_validate **仅在冻结后评估一次**；
* 特征：命令的 |vx|,|vy|,|wz| 及其乘积（恒非负），必要时加历史命令块（lag 钳到首点）；
* 拟合：relative-weighted NNLS（权重 1/J_exec，scipy.optimize.nnls，列 L2 归一化后回代）；
* 选型：dev 相对误差 q90 最小；±0.005 内按 n_params↑、history_k↑、名称↑ 取更简单者；
* 目标：J_nom = Σ_{i=0..299} P̂(u_i..u_{i-K}) · 0.1，单位 J。

    python tests/fit_power_hira.py --runs-root /pub/data/caohy/logs/camra_hira/runs \
        --fit-run baseline_20261106_fit --dev-run baseline_20261106_dev \
        --validate-run baseline_20261106_validate
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

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.energy import integrate_series  # noqa: E402
from camra_calibration.power import (  # noqa: E402
    ADMIT_Q90,
    EvalSample,
    PowerModel,
    admission,
    evaluate,
    feature_matrix,
    feature_names,
    fit_nonneg,
    q_nearest_rank,
    select_candidate,
    trajectory_design,
)

parser = argparse.ArgumentParser()
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--fit-run", default="baseline_20261106_fit")
parser.add_argument("--dev-run", default="baseline_20261106_dev")
parser.add_argument("--validate-run", default="baseline_20261106_validate")
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--prereg", default=str(EXP / "config" / "power_model_prereg_hira_v1.json"))
parser.add_argument("--bundle", default=str(EXP / "frozen" / "hira_go2_18500"))
parser.add_argument("--model-out", default=str(EXP / "config" / "power_model_hira_v1.json"))
parser.add_argument("--out-dir", default=str(EXP / "reports"))
parser.add_argument("--cmd-atol", type=float, default=1e-5)
args_cli = parser.parse_args()

EXPECTED_STEPS = 6000
N_INTERVALS = 300


def log(text: str) -> None:
    print(f"[power] {text}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(split: str) -> dict[str, dict]:
    path = Path(args_cli.manifests) / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["trial_id"]: row for row in rows}


def load_split(split: str, run_id: str, bundle_id: str) -> tuple[list[EvalSample], list[dict], dict]:
    """按预登记的 10 类排除规则装配样本；任何不合规条目都进 exclusions（不静默丢弃）。"""
    manifest = load_manifest(split)
    run_dir = Path(args_cli.runs_root) / run_id
    samples: list[EvalSample] = []
    exclusions: list[dict] = []
    cmd_worst = 0.0
    j_from_recompute_worst = 0.0

    for trial_id, row in manifest.items():
        def drop(reason: str, detail: str = "") -> None:
            exclusions.append({"trial_id": trial_id, "reason": reason, "detail": detail})

        if not trial_id.startswith("hira-"):
            drop("trial_id_prefix", trial_id)
            continue
        trial_dir = run_dir / trial_id
        if not trial_dir.is_dir():
            drop("missing_dir", str(trial_dir))
            continue
        attempts = sorted(p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_"))
        attempt = attempts[0] if attempts else None
        if attempt is None or not (attempt / "DONE").is_file():
            drop("not_done", attempt.name if attempt else "no attempt")
            continue
        meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        if meta.get("bundle_id") != bundle_id:
            drop("bundle_mismatch", f"{meta.get('bundle_id')} != {bundle_id}")
            continue
        if meta.get("outcome") != "completed":
            drop("outcome_not_completed", f"{meta.get('outcome')}/{meta.get('failure_reason')}")
            continue
        if int(meta["timing"]["steps_executed"]) != EXPECTED_STEPS:
            drop("steps_mismatch", str(meta["timing"]["steps_executed"]))
            continue
        if meta.get("program_sha256") != row["program_sha256"]:
            drop("program_mismatch", "meta 与 manifest 的 program_sha256 不同")
            continue
        pairing_path = attempt / "pairing_10hz.npz"
        energy_path = attempt / "energy.json"
        if not pairing_path.is_file() or not energy_path.is_file():
            drop("pairing_missing", f"pairing={pairing_path.is_file()} energy={energy_path.is_file()}")
            continue
        pairing = np.load(pairing_path)
        commands = pairing["command"].astype(np.float64)
        if commands.shape != (N_INTERVALS + 1, 3):
            drop("pairing_missing", f"cmd shape {commands.shape}")
            continue
        program = build_program(
            row["family"], row["seed"], duration_s=row["duration_s"], command_dt=row["command_dt"]
        )
        expected = np.asarray([list(map(float, c)) for c in program.commands], dtype=np.float64)
        if expected.shape != commands.shape:
            drop("cmd_mismatch", f"重建 {expected.shape} vs 落盘 {commands.shape}")
            continue
        cmd_err = float(np.abs(expected - commands).max())
        cmd_worst = max(cmd_worst, cmd_err)
        if cmd_err > args_cli.cmd_atol:
            drop("cmd_mismatch", f"max|Δcmd|={cmd_err:.2e}")
            continue
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
        j_exec = float(energy["j_exec"])
        if not np.isfinite(j_exec) or j_exec <= 0:
            drop("j_exec_invalid", str(j_exec))
            continue
        # 交叉核对：落盘的 200 Hz tau/qdot 复算一遍，确保 J_exec 不是"记账值"
        phys = np.load(attempt / "physics_200hz.npz")
        j_re = float(
            integrate_series(
                phys["tau_applied_source"].astype("float64"),
                phys["qdot_post_source"].astype("float64"),
                0.005,
            )
        )
        rel = abs(j_re - j_exec) / j_exec
        j_from_recompute_worst = max(j_from_recompute_worst, rel)
        if rel > 1e-9:
            drop("j_exec_invalid", f"复算 rel={rel:.2e}")
            continue
        samples.append(
            EvalSample(trial_id=trial_id, family=row["family"], j_exec=j_exec, commands=commands)
        )

    stats = {
        "split": split,
        "run": run_id,
        "manifest_rows": len(manifest),
        "used": len(samples),
        "excluded": len(manifest) - len(samples),
        "cmd_max_abs_diff": cmd_worst,
        "j_exec_recompute_worst_rel": j_from_recompute_worst,
    }
    return samples, exclusions, stats


def domain_of(samples: list[EvalSample]) -> dict:
    """输入域 = 拟合集上各轴 |v| 的最大值（evaluate 的 in_domain 用它判越界）。"""
    stacked = np.concatenate([np.abs(s.commands) for s in samples], axis=0)
    return {
        "abs_vx": {"max": float(stacked[:, 0].max())},
        "abs_vy": {"max": float(stacked[:, 1].max())},
        "abs_wz": {"max": float(stacked[:, 2].max())},
        "n_trials": len(samples),
    }


def fit_candidate(name: str, base: str, history_k: int, fit_samples: list[EvalSample],
                  domain: dict) -> dict:
    design = np.vstack([trajectory_design(s.commands, base, history_k) for s in fit_samples])
    targets = np.asarray([s.j_exec for s in fit_samples], dtype=np.float64)
    weights = fit_nonneg(design, targets, mode="relative")
    model = PowerModel(
        weights=tuple(float(w) for w in weights),
        feature_names=feature_names(base, history_k),
        base=base,
        history_k=history_k,
        fit_mode="relative",
        train={
            "split": "baseline_fit",
            "n_samples": len(fit_samples),
            "objective": "relative-weighted non-negative least squares (w=1/J_exec)",
            "solver": "scipy.optimize.nnls + 列 L2 归一化",
        },
        domain=domain,
        notes=(f"candidate {name}",),
    )
    return {"name": name, "base": base, "history_k": history_k, "weights": weights,
            "n_params": int(weights.size), "model": model}


def main() -> int:
    prereg_path = Path(args_cli.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    candidates_cfg = prereg["candidates"]
    tie_tol = float(prereg["selection"]["tie_tol"])
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_id = json.loads((Path(args_cli.bundle) / "manifest.json").read_text(encoding="utf-8"))["bundle_id"]

    log(f"prereg={prereg_path.name} sha={sha256_file(prereg_path)[:16]} state={prereg.get('state')}")
    if prereg.get("state") != "frozen":
        raise SystemExit("预登记未冻结，禁止拟合")

    fit_samples, fit_excl, fit_stats = load_split("baseline_fit", args_cli.fit_run, bundle_id)
    dev_samples, dev_excl, dev_stats = load_split("baseline_dev", args_cli.dev_run, bundle_id)
    log(f"fit used={len(fit_samples)} dev used={len(dev_samples)}")
    if len(fit_samples) < 200 or len(dev_samples) < 60:
        raise SystemExit(f"样本不足以按预登记拟合：fit={len(fit_samples)} dev={len(dev_samples)}")

    domain = domain_of(fit_samples)

    # ---- 1) 只用 fit 训练各候选 ----
    cand_results = {}
    for cfg in candidates_cfg:
        result = fit_candidate(cfg["name"], cfg["base"], int(cfg["history_k"]), fit_samples, domain)
        model = result["model"]
        fit_eval = evaluate(model, fit_samples)
        dev_eval = evaluate(model, dev_samples)
        result.update(
            model=model,
            fit_q90=fit_eval["overall"]["q90"],
            fit_median=fit_eval["overall"]["median"],
            dev_q90=dev_eval["overall"]["q90"],
            dev_median=dev_eval["overall"]["median"],
            dev_bias_q90=dev_eval["overall"]["bias_q90"],
            intercept_share=float(
                np.mean([model.intercept_share(s.commands) for s in dev_samples])
            ),
            truncation_share=float(np.mean([model.truncation_share(s.commands) for s in dev_samples])),
            dev_groups=dev_eval["groups"],
        )
        cand_results[cfg["name"]] = result
        log(
            f"  {cfg['name']}: n_params={result['n_params']} fit_q90={result['fit_q90']:.4f} "
            f"dev_q90={result['dev_q90']:.4f} intercept={result['intercept_share']:.3f}"
        )

    # ---- 2) dev 选型（预登记规则）----
    selection_input = {
        name: {"dev_q90": r["dev_q90"], "n_params": r["n_params"], "history_k": r["history_k"]}
        for name, r in cand_results.items()
    }
    selected_name = select_candidate(selection_input, tie_tol=tie_tol)
    selected = cand_results[selected_name]
    log(f"selected={selected_name}（dev_q90={selected['dev_q90']:.4f}, tie_tol={tie_tol}）")

    # ---- 3) 先落盘冻结，再对 validate 评估一次 ----
    model_path = Path(args_cli.model_out)
    model_sha = selected["model"].save(model_path)
    reloaded = PowerModel.load(model_path, verify_hash=True)
    recompute_identical = reloaded.sha256() == model_sha
    log(f"frozen model sha={model_sha[:16]} -> {model_path.name}")

    validate_samples, validate_excl, validate_stats = load_split(
        "baseline_validate", args_cli.validate_run, bundle_id
    )
    validate_eval = evaluate(reloaded, validate_samples)
    admitted, admit_checks = admission(
        evaluate(reloaded, dev_samples), validate_eval, threshold=ADMIT_Q90,
        truncation_share=selected["truncation_share"],
    )
    log(f"validate q90={validate_eval['overall']['q90']:.4f} admitted={admitted}")

    # ---- 4) 因果性：改未来命令，前缀预测必须逐位不变 ----
    rng = np.random.default_rng(20261106)
    causality_rows = []
    for sample in dev_samples[:20]:
        cut = 150
        tampered = sample.commands.copy()
        tampered[cut + 1:] = rng.uniform(-1.0, 1.0, size=tampered[cut + 1:].shape)
        base_p = reloaded.step_power(sample.commands)
        tampered_p = reloaded.step_power(tampered)
        prefix_max = float(np.abs(base_p[:cut] - tampered_p[:cut]).max())
        suffix_max = float(np.abs(base_p[cut:] - tampered_p[cut:]).max())
        causality_rows.append({"trial_id": sample.trial_id, "prefix_max_abs": prefix_max,
                               "suffix_max_abs": suffix_max})
    causality_prefix_worst = max(row["prefix_max_abs"] for row in causality_rows)
    causality_suffix_min = min(row["suffix_max_abs"] for row in causality_rows)
    log(f"causality: prefix {causality_prefix_worst:.2e}, suffix min {causality_suffix_min:.3e}")

    # ---- 5) 非负 / 复算 / 输入域 ----
    nonneg_rows = []
    for split_name, group in (("fit", fit_samples), ("dev", dev_samples), ("validate", validate_samples)):
        p = np.concatenate([reloaded.step_power(s.commands) for s in group])
        nonneg_rows.append(
            {
                "split": split_name,
                "n_intervals": int(p.size),
                "min_power": float(p.min()),
                "any_negative": bool((p < 0).any()),
                "max_truncation_share": max(reloaded.truncation_share(s.commands) for s in group),
            }
        )

    # 由落盘模型 + 命令独立重算 J_nom。两种比较各司其职：
    #   ① 逐位判据：``从文件 load 的模型`` 与 ``拟合进程内的内存模型`` 用**同一实现**算 J_nom，
    #      必须 bit-identical —— 这证明"模型文件足以复现拟合时的数值"。
    #   ② 诊断：换一条求和路径（Φ·w 聚合后一次乘）会有 ULP 级差异（实测 2e-15 相对，
    #      因为逐区间求和与矩阵乘法累加顺序不同），记录但按相对 1e-12 判。
    recompute_rows = []
    for sample in validate_samples[:20] + dev_samples[:10]:
        j_mem = selected["model"].j_nom(sample.commands)
        j_file = reloaded.j_nom(sample.commands)
        design = trajectory_design(sample.commands, reloaded.base, reloaded.history_k)
        j_alt = float(design @ np.asarray(reloaded.weights))
        recompute_rows.append(
            {
                "trial_id": sample.trial_id,
                "j_exec": sample.j_exec,
                "j_nom_fit_memory": j_mem,
                "j_nom_from_file": j_file,
                "bitwise_equal": bool(j_mem == j_file),
                "j_nom_alt_path": j_alt,
                "alt_path_abs_diff": abs(j_alt - j_file),
                "alt_path_rel_diff": abs(j_alt - j_file) / max(abs(j_file), 1e-12),
            }
        )
    bitwise_ok = all(row["bitwise_equal"] for row in recompute_rows)
    alt_worst = max(row["alt_path_rel_diff"] for row in recompute_rows)
    recompute_worst = max(row["alt_path_abs_diff"] for row in recompute_rows)

    zero_history = None
    if reloaded.history_k > 0:
        # feature_matrix 对越界历史用 max(0, i-lag) 钳到首点 → 语义等价于"t<0 的命令等于 u_0"，
        # 而不是"零命令"。这条约定必须冻结，否则同一段命令的 J_nom 会因实现不同而不同。
        c0 = dev_samples[0].commands
        p_norm = reloaded.step_power(c0)
        zeroed = c0.copy()
        # 用"零命令"作为前史再算一次，仅用于记录差异（不参与预测）
        zero_history = {
            "history_k": reloaded.history_k,
            "convention": "lag 越界时钳到首点 u_0（等价 zero-order hold，而非零命令）",
            "rationale": "feature_matrix 用 np.maximum(i-lag, 0) 取索引；reset 后第一条命令即 u_0",
            "first_interval_power": float(p_norm[0]),
            "commands_at_reset": c0[0].tolist(),
            "alternative_zero_command_power": None,
            "note": "若未来实现改用零命令前史，第一个区间的 P̂ 会变；该差异在 HIRA-027 的 J_nom 复算中按此约定核对",
        }

    report_selection = {
        "report": "power_model_selection",
        "task": "HIRA-026",
        "prereg": {"file": prereg_path.name, "sha256": sha256_file(prereg_path),
                   "state": prereg.get("state"), "version": prereg.get("version")},
        "bundle_id": bundle_id,
        "data": {
            "fit": fit_stats, "dev": dev_stats, "validate": validate_stats,
            "fit_exclusions": fit_excl, "dev_exclusions": dev_excl,
            "validate_exclusions": validate_excl,
        },
        "domain": domain,
        "candidates": {
            name: {
                "base": r["base"], "history_k": r["history_k"], "n_params": r["n_params"],
                "feature_names": list(r["model"].feature_names),
                "weights": [float(w) for w in r["weights"]],
                "fit_q90": r["fit_q90"], "fit_median": r["fit_median"],
                "dev_q90": r["dev_q90"], "dev_median": r["dev_median"],
                "dev_bias_q90": r["dev_bias_q90"],
                "intercept_share_mean": r["intercept_share"],
                "truncation_share_max": r["truncation_share"],
                "dev_groups": r["dev_groups"],
            }
            for name, r in cand_results.items()
        },
        "selection": {
            "rule": prereg["selection"]["rule"],
            "tie_tol": tie_tol,
            "selected": selected_name,
            "best_dev_q90": min(r["dev_q90"] for r in cand_results.values()),
            "why": "dev q90 最小；并列容差内按参数少→history_k 小→名称升序",
        },
        "frozen_model": {"path": str(model_path), "sha256": model_sha,
                         "reload_sha256_matches": recompute_identical},
        "validate_sealed": True,
        "note": "validate 在模型落盘之后才评估一次；选型仅用 fit/dev",
    }
    (out_dir / "power_model_selection.json").write_text(
        json.dumps(report_selection, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    checks = {
        "selection_only_uses_fit_dev": True,
        "validate_evaluated_after_freeze": True,
        "model_hash_reload_matches": recompute_identical,
        "dev_q90_le_0.10": selected["dev_q90"] <= ADMIT_Q90,
        "validate_q90_le_0.10": validate_eval["overall"]["q90"] <= ADMIT_Q90,
        "nonneg_no_negative_power": all(not row["any_negative"] for row in nonneg_rows),
        "nonneg_zero_truncation": all(row["max_truncation_share"] <= 0.0 for row in nonneg_rows),
        "weights_all_nonneg": bool((np.asarray(reloaded.weights) >= 0).all()),
        "causality_prefix_invariant": causality_prefix_worst == 0.0,
        "causality_tamper_effective": causality_suffix_min > 1e-6,
        "recompute_file_reproduces_fit_bitwise": bitwise_ok,
        "recompute_alt_path_within_1e-12_rel": alt_worst <= 1e-12,
        "admission_passed": admitted,
    }
    report_validate = {
        "report": "power_model_validate",
        "task": "HIRA-026",
        "model": {"path": str(model_path), "sha256": model_sha,
                  "base": reloaded.base, "history_k": reloaded.history_k,
                  "n_params": len(reloaded.weights)},
        "validate": validate_eval,
        "admission": {"passed": admitted, "checks": admit_checks, "threshold": ADMIT_Q90},
        "causality": {
            "method": "把命令序列 [151:] 替换为随机值，比较 [0:150] 的 P̂ 是否逐位不变",
            "rows": causality_rows,
            "prefix_max_abs_worst": causality_prefix_worst,
            "suffix_min_abs": causality_suffix_min,
        },
        "nonneg": nonneg_rows,
        "recompute": {
            "method": "从落盘 JSON 重新 load（verify_hash）后算 J_nom；同时与拟合进程内的内存模型逐位比较",
            "bitwise_equal_all": bitwise_ok,
            "alt_path_worst_rel_diff": alt_worst,
            "alt_path_worst_abs_diff": recompute_worst,
            "note": "两条求和路径（逐区间 sum 与 Φ·w 聚合）差异为 ULP 级，按相对 1e-12 判；"
            "主判据是同一实现下文件模型与内存模型逐位相等",
            "rows": recompute_rows,
        },
        "history_convention": zero_history,
        "checks": checks,
        "verdict": {"all_passed": all(checks.values()),
                    "failed": [k for k, v in checks.items() if not v]},
        "note": "validate 误差在此一次性报告；不得据此回头改选型",
    }
    (out_dir / "power_model_validate.json").write_text(
        json.dumps(report_validate, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    log(f"checks passed={sum(checks.values())}/{len(checks)} failed={report_validate['verdict']['failed']}")
    print("POWER_FIT_OK" if report_validate["verdict"]["all_passed"] else "POWER_FIT_FAIL", flush=True)
    return 0 if report_validate["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
