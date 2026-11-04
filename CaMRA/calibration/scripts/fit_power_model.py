"""CAL-020：拟合、选型、冻结并验证非负名义功率模型。

严格按 ``config/power_model_prereg_v1.json``（预登记，先于数据冻结）执行：

1. 只读 ``baseline_fit`` 拟合候选模型；``baseline_dev`` 只用于按预登记规则选型；
2. 选型完成后立刻把权重、特征 schema、输入域与散列写入模型文件（"冻结"），
   并在报告里记录冻结时刻与散列——**在**读取 ``baseline_validate`` 之前；
3. 冻结后才评估一次 ``baseline_validate``，按 §10.5 判定准入；
4. 逐条记录排除原因，不静默丢样本。

退出码：0 准入通过；2 预登记/数据不合法；3 未达准入；4 数据缺失。

用法：
    python scripts/fit_power_model.py \
        --manifest-dir /pub/data/caohy/logs/camra/trial_lists_v1.0.1 \
        --out-root /pub/data/caohy/logs/camra/trials \
        --prereg config/power_model_prereg_v1.json \
        --out-model config/power_model_v1.json \
        --report /pub/data/caohy/logs/camra/power_fit/power_fit_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.power import (  # noqa: E402
    ADMIT_Q90, EvalSample, PowerModel, PowerModelError, admission,
    evaluate, fit_nonneg, feature_names, select_candidate, trajectory_design,
)

REASONS = ("missing_dir", "not_done", "outcome_not_completed", "steps_mismatch",
           "program_mismatch", "cmd_mismatch", "j_exec_invalid", "pairing_missing")
CMD_ATOL = 1e-5
JEXEC_RTOL = 1e-6


# ---------------------------------------------------------------- 读取
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prereg(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise SystemExit(f"FATAL: 预登记文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "frozen":
        raise SystemExit(f"FATAL: 预登记未处于 frozen 状态: {payload.get('state')!r}")
    digest = sha256_file(path)
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.is_file():
        raise SystemExit(f"FATAL: 缺少预登记散列侧车: {side}")
    if side.read_text(encoding="utf-8").split()[0] != digest:
        raise SystemExit("FATAL: 预登记散列不匹配（文件被改动）")
    return payload, digest


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"FATAL: 清单不存在: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_done_dir(base: Path, trial_id: str) -> Path | None:
    """优先正式目录；否则取 attempt 最大的 ``__aN`` 目录（两者都必须有 DONE）。"""
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
        import torch  # 延迟导入：仅读 trial 产物时需要
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def load_split(manifest_dir: Path, out_root: Path, split: str
               ) -> tuple[list[EvalSample], list[dict]]:
    specs = read_jsonl(manifest_dir / f"{split}.jsonl")
    samples: list[EvalSample] = []
    excluded: list[dict] = []

    def drop(spec: dict, reason: str, detail: str = "") -> None:
        assert reason in REASONS, f"未预登记的排除原因: {reason}"
        excluded.append({"trial_id": spec["trial_id"], "reason": reason, "detail": detail})

    for spec in specs:
        d = find_done_dir(out_root / split, spec["trial_id"])
        if d is None:
            drop(spec, "missing_dir", str(out_root / split / spec["trial_id"]))
            continue
        meta_p = d / "meta.json"
        if not meta_p.is_file():
            drop(spec, "not_done", f"{d}/meta.json 缺失")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("outcome") != "completed":
            drop(spec, "outcome_not_completed", str(meta.get("outcome")))
            continue
        if (meta.get("steps_executed") != 6000
                or meta.get("steps_expected", 6000) != 6000):
            drop(spec, "steps_mismatch",
                 f"{meta.get('steps_executed')}/{meta.get('steps_expected')}")
            continue

        # 三重绑定：清单散列 / meta 散列 / 由 (family,seed) 重建的 Program 散列
        prog = build_program(spec["family"], spec["seed"],
                             duration_s=spec["duration_s"], command_dt=spec["command_dt"])
        rebuilt = prog.sha256()
        if not (rebuilt == spec["program_sha256"] == meta.get("program_sha256")):
            drop(spec, "program_mismatch",
                 f"rebuilt={rebuilt[:12]} list={spec['program_sha256'][:12]} "
                 f"meta={str(meta.get('program_sha256'))[:12]}")
            continue

        pair = load_npz(d / "pairing_10hz.npz")
        if pair is None or "cmd" not in pair:
            drop(spec, "pairing_missing", str(d / "pairing_10hz.npz"))
            continue
        cmd = np.asarray(pair["cmd"], dtype=np.float64)
        ref = np.asarray(prog.commands, dtype=np.float64)
        if cmd.shape != ref.shape or not np.allclose(cmd, ref, atol=CMD_ATOL, rtol=0.0):
            drop(spec, "cmd_mismatch", f"shape={cmd.shape} ref={ref.shape}")
            continue

        energy_p = d / "energy.json"
        if not energy_p.is_file():
            drop(spec, "j_exec_invalid", "energy.json 缺失")
            continue
        j_exec = json.loads(energy_p.read_text(encoding="utf-8")).get("j_exec")
        meta_j = (meta.get("energy") or {}).get("j_exec")
        try:
            j_exec = float(j_exec)
        except (TypeError, ValueError):
            drop(spec, "j_exec_invalid", f"j_exec={j_exec!r}")
            continue
        if not np.isfinite(j_exec) or j_exec <= 0:
            drop(spec, "j_exec_invalid", f"j_exec={j_exec}")
            continue
        if meta_j is not None and abs(float(meta_j) - j_exec) > JEXEC_RTOL * j_exec:
            drop(spec, "j_exec_invalid", f"energy.json 与 meta.json 不一致: {j_exec} vs {meta_j}")
            continue

        samples.append(EvalSample(trial_id=spec["trial_id"], family=spec["family"],
                                  j_exec=j_exec, commands=cmd))
    return samples, excluded


# ---------------------------------------------------------------- 组装
def build_design(samples: list[EvalSample], base: str, k: int) -> np.ndarray:
    return np.stack([trajectory_design(s.commands, base, k) for s in samples], axis=0)


def domain_from(samples: list[EvalSample]) -> dict:
    cmd = np.concatenate([s.commands for s in samples], axis=0)
    return {axis: {"min": 0.0, "max": float(np.abs(cmd[:, j]).max()), "n_points": int(cmd.shape[0])}
            for j, axis in enumerate(("abs_vx", "abs_vy", "abs_wz"))}


def causality_check(model: PowerModel, sample: EvalSample, cut: int = 150) -> dict:
    """前缀不变性：改动 cut 之后的命令，`< cut` 区间的 P̂ 必须逐位不变。"""
    cmd = np.array(sample.commands, dtype=np.float64, copy=True)
    base = model.step_power(cmd)
    cmd[cut + 1:] = -cmd[cut + 1:] * 0.5 + 0.25
    after = model.step_power(cmd)
    delta = float(np.abs(base[:cut] - after[:cut]).max())
    return {"trial_id": sample.trial_id, "cut": cut, "max_prefix_delta": delta,
            "ok": delta == 0.0}


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-020 名义功率模型拟合与验证")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--out-model", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--threshold", type=float, default=ADMIT_Q90)
    ap.add_argument("--check-split", default=None,
                    help="只做数据装配检查（不拟合）：载入该 split 并打印排除原因后退出")
    args = ap.parse_args(argv)

    cal_dir = Path(__file__).resolve().parent.parent
    prereg_p = Path(args.prereg) if args.prereg else cal_dir / "config" / "power_model_prereg_v1.json"
    prereg, prereg_sha = load_prereg(prereg_p)

    manifest_dir, out_root = Path(args.manifest_dir), Path(args.out_root)
    if args.check_split:
        s, ex = load_split(manifest_dir, out_root, args.check_split)
        hist: dict[str, int] = {}
        for e in ex:
            hist[e["reason"]] = hist.get(e["reason"], 0) + 1
        print(f"split={args.check_split} n_used={len(s)} n_excluded={len(ex)}")
        print(f"  exclusions: {hist or '{}'}")
        for r in ex[:5]:
            print(f"  DROP {r['trial_id']}: {r['reason']} {r['detail'][:80]}")
        if s:
            j = [x.j_exec for x in s]
            print(f"  J_exec min {min(j):.3f} max {max(j):.3f} mean {sum(j)/len(j):.3f}  "
                  f"cmd_len={sorted({len(x.commands) for x in s})}")
            # 不拟合任何系数；只证明特征管线在真实命令上可算（Φ 前 4 项 = bias,|vx|,|vy|,|wz|）
            phi = trajectory_design(s[0].commands, "linear", 0)
            print(f"  design Phi(linear,k=0) 前4项={np.round(phi, 3).tolist()}  "
                  f"共{phi.size}项（未拟合，无 J_nom 结论）")
        print("VERDICT: loader_ok" if s else "VERDICT: loader_empty")
        return 0 if s else 4

    if not (args.out_model and args.report):
        ap.error("拟合模式必须给出 --out-model 与 --report")

    fit_s, fit_ex = load_split(manifest_dir, out_root, "baseline_fit")
    dev_s, dev_ex = load_split(manifest_dir, out_root, "baseline_dev")
    print(f"fit n={len(fit_s)} (excluded {len(fit_ex)})  "
          f"dev n={len(dev_s)} (excluded {len(dev_ex)})")
    if len(fit_s) < 100 or len(dev_s) < 20:
        print("FATAL: 可用样本不足（fit < 100 或 dev < 20），确认采集是否完成")
        return 4

    # ---- 1) 只用 fit 拟合 ----
    results: dict[str, dict] = {}
    designs: dict[str, np.ndarray] = {}
    for cand in prereg["candidates"]:
        name, base, k = cand["name"], cand["base"], int(cand["history_k"])
        design = build_design(fit_s, base, k)
        designs[name] = design
        w = fit_nonneg(design, np.array([s.j_exec for s in fit_s]), mode="relative")
        model = PowerModel(weights=tuple(w), feature_names=feature_names(base, k),
                           base=base, history_k=k, fit_mode="relative",
                           train={"split": "baseline_fit", "n_trials": len(fit_s)},
                           domain=domain_from(fit_s))
        fit_eval = evaluate(model, fit_s)
        dev_eval = evaluate(model, dev_s)
        results[name] = {"n_params": int((w > 0).sum()), "n_params_total": int(w.size),
                         "history_k": k, "base": base, "weights": [float(x) for x in w],
                         "fit": fit_eval, "dev": dev_eval,
                         "dev_q90": dev_eval["overall"]["q90"],
                         "fit_q90": fit_eval["overall"]["q90"]}
        print(f"  {name:<12} params {results[name]['n_params']}/{w.size}  "
              f"fit q90 {fit_eval['overall']['q90']:.4f}  dev q90 {dev_eval['overall']['q90']:.4f}")

    # ---- 2) 按预登记规则选型 ----
    chosen = select_candidate(results)
    print(f"selected: {chosen}")

    # ---- 3) 冻结：落盘权重 + schema + 域 + 散列（在 validate 之前）----
    r = results[chosen]
    model = PowerModel(
        weights=tuple(r["weights"]),
        feature_names=feature_names(r["base"], int(r["history_k"])),
        base=r["base"], history_k=int(r["history_k"]), fit_mode="relative",
        train={
            "split": "baseline_fit", "n_trials": len(fit_s),
            "list_sha256": sha256_file(manifest_dir / "baseline_fit.jsonl"),
            "protocol_version": "1.0.1",
            "prereg_sha256": prereg_sha,
            "selection": "dev q90 nearest-rank, tie_tol 0.005, prefer fewer params",
            "candidate": chosen,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        domain=domain_from(fit_s),
        notes=("J_nom = 0.1 * Σ P̂_nom(i)；输入仅当前与历史命令（K=%d）" % int(r["history_k"]),
               "权重由 scipy NNLS 求解，构造性非负"),
    )
    model_sha = model.sha256()
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out_model)
    print(f"frozen model sha256: {model_sha}")

    # ---- 4) 冻结后才看 validate ----
    val_s, val_ex = load_split(manifest_dir, out_root, "baseline_validate")
    print(f"validate n={len(val_s)} (excluded {len(val_ex)})")
    if len(val_s) < 20:
        print("FATAL: validate 可用样本不足 20")
        return 4
    dev_eval = results[chosen]["dev"]
    val_eval = evaluate(model, val_s)
    trunc = max(model.truncation_share(s.commands) for s in val_s)
    ok, checks = admission(dev_eval, val_eval, threshold=args.threshold,
                           truncation_share=trunc)
    causal = [causality_check(model, s) for s in val_s[:3]]

    # 可复算：从落盘文件重新加载并复算，必须逐位一致
    reloaded = PowerModel.load(args.out_model)
    recompute = []
    for s in val_s[:5]:
        a, b = model.j_nom(s.commands), reloaded.j_nom(s.commands)
        recompute.append({"trial_id": s.trial_id, "j_nom_in_memory": a,
                          "j_nom_from_file": b, "identical": a == b})
    recompute_ok = all(x["identical"] for x in recompute)

    intercept = summarize_intercept(model, val_s)
    report = {
        "cal_task": "CAL-020",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prereg": {"path": str(prereg_p), "sha256": prereg_sha},
        "splits": {
            "fit": {"manifest": str(manifest_dir / "baseline_fit.jsonl"),
                    "n_used": len(fit_s), "excluded": fit_ex},
            "dev": {"manifest": str(manifest_dir / "baseline_dev.jsonl"),
                    "n_used": len(dev_s), "excluded": dev_ex},
            "validate": {"manifest": str(manifest_dir / "baseline_validate.jsonl"),
                         "n_used": len(val_s), "excluded": val_ex},
        },
        "candidates": {n: {k: v for k, v in c.items() if k not in ("fit", "dev")}
                       for n, c in results.items()},
        "selected": chosen,
        "frozen_model": {"path": str(args.out_model), "sha256": model_sha,
                         "schema_sha256": model.schema_sha256(),
                         "frozen_at": model.train["frozen_at"]},
        "dev": {"overall": dev_eval["overall"], "groups": dev_eval["groups"]},
        "validate": {"overall": val_eval["overall"], "groups": val_eval["groups"],
                     "n_non_computable": val_eval["n_non_computable"],
                     "n_out_of_domain": val_eval["n_out_of_domain"],
                     "rows": val_eval["rows"]},
        "nonneg": {"truncation_share_max": trunc,
                   "all_weights_nonneg": all(w >= 0 for w in model.weights)},
        "causality": causal,
        "recomputability": {"rows": recompute, "ok": recompute_ok},
        "intercept": intercept,
        "admission": {"threshold": args.threshold, "checks": checks, "pass": bool(ok)},
        "verdict": "admitted" if ok else "not_admitted",
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    rep_sha = sha256_file(rep_p)
    rep_p.with_suffix(rep_p.suffix + ".sha256").write_text(f"{rep_sha}  {rep_p.name}\n",
                                                           encoding="utf-8")

    # ---- 5) 摘要 ----
    o, g = val_eval["overall"], val_eval["groups"]
    print()
    print(f"dev      q90 {dev_eval['overall']['q90']:.4f}  median "
          f"{dev_eval['overall']['median']:.4f}")
    print(f"validate q90 {o['q90']:.4f}  median {o['median']:.4f}  max {o['max']:.4f}")
    for f, b in g["family"].items():
        print(f"  family {f:<16} n={b['n']:<3} q90 {b.get('q90', float('nan')):.4f}")
    for name in ("low_speed", "transition"):
        for k, b in g[name].items():
            print(f"  {name}:{k:<6} n={b.get('n', 0):<3} q90 {b.get('q90', float('nan')):.4f}")
    print(f"  intercept share (median) {intercept['median']:.3f}")
    print(f"  truncation {trunc:.1e}  out_of_domain {val_eval['n_out_of_domain']}  "
          f"non_computable {val_eval['n_non_computable']}")
    for c in causal:
        print(f"  causality {c['trial_id']}: prefix delta {c['max_prefix_delta']:.1e} ok={c['ok']}")
    print(f"  recompute-from-file ok={recompute_ok}")
    print(f"model  {args.out_model}  sha {model_sha}")
    print(f"report {rep_p}  sha {rep_sha}")
    print(f"VERDICT: {report['verdict']}")
    return 0 if ok else 3


def summarize_intercept(model: PowerModel, samples: list[EvalSample]) -> dict:
    vals = [model.intercept_share(s.commands) for s in samples]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return {"median": float("nan"), "max": float("nan"), "n": 0}
    arr = np.asarray(vals)
    return {"median": float(np.median(arr)), "max": float(arr.max()), "n": int(arr.size)}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PowerModelError as exc:
        print(f"MODEL ERROR: {exc}")
        sys.exit(2)
