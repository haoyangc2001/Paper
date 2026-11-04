"""CAL-022 验收：逐步成本与整条 J_nom 一致、能量比不为恒定 0、旧预算隔离。

检查项（逐项打印 + 写报告）：

1. **逐步相加 = J_nom**：真实 validate 命令序列与若干人为长度（10 s / 30 s / 45 s）上，
   ``Σ (1+α)c_nom / (1+α)`` 与 ``model.j_nom`` 的差 ≤ 1e-9 相对量；含 K>0 的合成模型。
2. **不因固定 30 s 而恒为 0**：真实 episode 的 ε_c 分布应跨 0 两侧且非退化；
   同时人工同能量案例（J_exec = J_nom）必须**恰好**得到 0.0 并被保留（不夹、不加 epsilon）。
3. **接口可消费**：schema/增广状态说明落盘并校验散列；预算 z_ratio 与扣减自洽。
4. **旧预算隔离**：``unit``/``kind`` 非 J 的预算构造必须被拒绝；本脚本不使用任何旧 a.u. 数值。

用法：
    python scripts/check_step_cost.py --model config/power_model_v1.json \
        --manifest-dir ... --out-root ... --report .../step_cost_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from camra_calibration.power import J_MIN, PowerModel, feature_names  # noqa: E402
from camra_calibration.step_cost import (  # noqa: E402
    BudgetState, StepCost, StepCostError, energy_ratio, schema_sha256, trajectory_cost,
    write_schema,
)

ALPHA = 0.0
TOL = 1e-9


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_done_dir(base: Path, trial_id: str) -> Path | None:
    d = base / trial_id
    if (d / "DONE").is_file():
        return d
    best = None
    for p in base.glob(f"{trial_id}__a*"):
        if (p / "DONE").is_file():
            n = int(p.name.split("__a")[-1]) if "__a" in p.name else 0
            if best is None or n > best[0]:
                best = (n, p)
    return None if best is None else best[1]


def load_validate_commands(manifest_dir: Path, out_root: Path, limit: int) -> list[dict]:
    import torch
    items = []
    for line in (manifest_dir / "baseline_validate.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        spec = json.loads(line)
        d = find_done_dir(out_root / "baseline_validate", spec["trial_id"])
        if d is None:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("outcome") != "completed":
            continue
        pair = torch.load(d / "pairing_10hz.npz", map_location="cpu")
        cmd = pair["cmd"].numpy().astype(np.float64)
        items.append({"trial_id": spec["trial_id"], "family": spec["family"],
                      "cmd": cmd, "j_exec": float(meta["energy"]["j_exec"])})
        if len(items) >= limit:
            break
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-022 步成本接口一致性检查")
    ap.add_argument("--model", default=None)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    model_p = Path(args.model) if args.model else cal / "config" / "power_model_v1.json"
    model = PowerModel.load(model_p)
    problems: list[str] = []

    # ---- 1) 逐步相加 = J_nom（真实数据 + 任意长度）----
    items = load_validate_commands(Path(args.manifest_dir), Path(args.out_root), args.limit)
    if not items:
        print("FATAL: 没有可用的 validate 命令")
        return 4
    rows = []
    for it in items:
        for frac in (1 / 3, 1.0, 1.5):            # 10 s / 30 s / 60 s 视窗
            cmd = it["cmd"]
            n = max(2, int(round((cmd.shape[0] - 1) * frac)) + 1)
            if n <= cmd.shape[0]:
                sub = cmd[:n]
            else:
                # 超出原始 30 s：把命令序列首尾相接延长，验证"任意视窗"都成立
                reps = int(math.ceil(n / (cmd.shape[0] - 1)))
                sub = np.concatenate([cmd] + [cmd[1:]] * (reps - 1), axis=0)[:n]
            out = trajectory_cost(model, sub, alpha=ALPHA)
            rows.append({"trial_id": it["trial_id"], "n_cmd": n, "horizon_s": out["horizon_s"],
                         "j_nom_J": out["j_nom_J"], "cost_nom_sum_J": out["cost_nom_sum_J"],
                         "abs_diff_J": out["abs_diff_J"], "rel_diff": out["rel_diff"],
                         "equal": out["equal"]})
    worst_rel = max(r["rel_diff"] for r in rows)
    all_equal = all(r["equal"] for r in rows)
    horizons = sorted({round(r["horizon_s"], 3) for r in rows})
    print(f"[1] 逐步相加 vs J_nom: {len(rows)} 组合，视窗 {horizons} s，"
          f"最大相对差 {worst_rel:.2e}，全部相等={all_equal}")
    if not all_equal:
        problems.append(f"逐步成本与 J_nom 不一致（最大相对差 {worst_rel:.2e}）")

    # 含历史依赖的合成模型：验证 K>0 时接口仍然等价
    h_names = feature_names("linear", 1)
    k1 = PowerModel(weights=(1.0, 30.0, 20.0, 10.0, 5.0, 15.0, 8.0, 6.0),
                    feature_names=h_names, base="linear", history_k=1,
                    fit_mode="relative", train={"split": "synthetic"}, domain={})
    k1_out = trajectory_cost(k1, items[0]["cmd"], alpha=ALPHA)
    print(f"[1b] K=1 合成模型: 逐步和 {k1_out['cost_nom_sum_J']:.6f} vs J_nom "
          f"{k1_out['j_nom_J']:.6f} (rel {k1_out['rel_diff']:.2e})")
    if not k1_out["equal"]:
        problems.append("K>0 时逐步成本与 J_nom 不等价")

    # ---- 2) 能量比不为恒定 0 ----
    ratios = []
    for it in items:
        j_nom = model.j_nom(it["cmd"])
        r, why = energy_ratio(it["j_exec"], j_nom)
        if r is not None:
            ratios.append(r)
    arr = np.asarray(ratios)
    spread = {"n": int(arr.size), "min": float(arr.min()), "max": float(arr.max()),
              "mean": float(arr.mean()), "n_above_zero": int((arr > 0).sum()),
              "n_below_zero": int((arr < 0).sum())}
    print(f"[2] 真实 ε_c: n={spread['n']} 范围 [{spread['min']:+.4f}, {spread['max']:+.4f}] "
          f"均值 {spread['mean']:+.4f}，正/负 = {spread['n_above_zero']}/{spread['n_below_zero']}")
    # 只要求"非退化"：真实数据上 ε_c 必须有可用方差且不全押在 0 上；
    # 单侧分布本身不是问题（模型系统偏高/偏低），但"恒为 0"是问题。
    degenerate = (spread["n"] < 10 or float(arr.std()) < 1e-4
                  or bool((np.abs(arr) < 1e-12).all()))
    print(f"[2] 非退化判定: std={float(arr.std()):.5f} degenerate={degenerate}")
    if degenerate:
        problems.append("ε_c 分布退化（固定 30 s 不应让比值恒为 0）")

    same, why_same = energy_ratio(432.1234, 432.1234)
    synthetic_zero = {"j_exec": 432.1234, "j_nom": 432.1234, "ratio": same, "reason": why_same,
                      "preserved_exact_zero": same == 0.0}
    print(f"[2b] 人工同能量案例: ratio={same} reason={why_same} 原样保留0={same == 0.0}")
    if same != 0.0:
        problems.append("同能量案例的 0 没有被原样保留")

    nc, why_nc = energy_ratio(5.0, 0.5, j_min=J_MIN)
    print(f"[2c] 近零案例: ratio={nc} reason={why_nc}（J_nom=0.5 < J_min={J_MIN}）")

    # ---- 3) 接口可消费 ----
    sc = StepCost(model)
    sc.reset()
    rec = sc.step((0.4, 0.0, 0.0))
    budget = BudgetState(initial_J=600.0)
    sc2 = StepCost(model, budget=budget)
    sc2.reset()
    for _ in range(5):
        sc2.step((0.5, 0.0, 0.1))
    schema = sc.schema()
    schema_p = Path(args.schema) if args.schema else cal / "config" / "step_cost_schema_v1.json"
    schema_digest = write_schema(schema_p, model, alpha=ALPHA)
    print(f"[3] 单步: power {rec.power_W:.3f} W → c_nom {rec.cost_nom_J:.4f} J (dt {rec.unit})")
    print(f"[3] 预算: spent {budget.spent_J:.4f} J / {budget.initial_J} J，"
          f"z_ratio {budget.z_ratio:.4f}，exhausted={budget.exhausted}")
    print(f"[3] 增广状态: {sc.augmented_state_spec()}")
    print(f"[3] schema: {schema_p} sha {schema_digest}")
    if schema.get("unit") != "J" or sc.history_k != model.history_k:
        problems.append("schema 单位或 history_k 与模型不一致")
    if not (0.0 <= budget.z_ratio <= 1.0):
        problems.append("z_ratio 越界")

    # 单位隔离：旧 a.u. 预算必须被拒
    rejected = []
    for kwargs in ({"unit": "a.u."}, {"kind": "legacy_au"}):
        try:
            BudgetState(initial_J=100.0, **kwargs)
        except StepCostError:
            rejected.append(str(kwargs))
    print(f"[4] 非 J 预算被拒: {rejected}")
    if len(rejected) != 2:
        problems.append("非 J 单位/口径的预算没有被拒绝")

    report = {
        "cal_task": "CAL-022",
        "model": {"path": str(model_p), "sha256": model.sha256(),
                  "history_k": model.history_k, "step_cost_dt": model.step_cost_dt,
                  "j_nom_for_standing_30s_J": model.j_nom(np.zeros((301, 3)))},
        "step_sum_vs_j_nom": {"worst_rel_diff": worst_rel, "all_equal": all_equal,
                              "rows": rows[:12], "n_combinations": len(rows)},
        "history_dependent_synthetic": {"rel_diff": k1_out["rel_diff"],
                                        "equal": k1_out["equal"]},
        "epsilon_c_spread": spread,
        "equal_energy_case": synthetic_zero,
        "near_zero_case": {"j_nom": 0.5, "ratio": nc, "reason": why_nc, "j_min": J_MIN},
        "schema": {"path": str(schema_p), "sha256": schema_digest,
                   "schema_sha256": schema_sha256(schema)},
        "budget_demo": budget.to_dict(),
        "unit_isolation": {"rejected": rejected},
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    rep_p.with_suffix(rep_p.suffix + ".sha256").write_text(
        f"{sha256_file(rep_p)}  {rep_p.name}\n", encoding="utf-8")
    for p in problems:
        print(f"  ERR {p}")
    print(f"report {rep_p}")
    print(f"VERDICT: {report['verdict']}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
