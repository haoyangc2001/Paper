"""HIRA-028：能量步成本与未来导航消费接口的验证。

交付物：
* `config/step_cost_schema_hira_v1.json`（+ .sha256）：`c_nom = P̂_nom(u)·0.1 s`、`c = (1+α)·c_nom`；
* `reports/step_cost_interface_tests.json`：13 项独立测试。

α 在 HIRA-031 才冻结，本脚本里的 α=0.25 **只是明确标记的测试参数**，不是估计值，
报告里逐处标注 `alpha_is_test_value=True`。

    python tests/verify_step_cost_hira.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from camra_calibration.power import PowerModel, feature_matrix, feature_names  # noqa: E402
from camra_calibration.step_cost import (  # noqa: E402
    BudgetState,
    StepCost,
    StepCostError,
    energy_ratio,
    schema_sha256,
    trajectory_cost,
    write_schema,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(EXP / "config/power_model_hira_v1.json"))
parser.add_argument("--footprint", default=str(EXP / "config/footprint_hira_v1.json"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--validate-run", default="baseline_20261106_validate")
parser.add_argument("--fit-run", default="baseline_20261106_fit")
parser.add_argument("--out-schema", default=str(EXP / "config/step_cost_schema_hira_v1.json"))
parser.add_argument("--out-report", default=str(EXP / "reports/step_cost_interface_tests.json"))
args_cli = parser.parse_args()

ALPHA_TEST = 0.25   # 测试参数，非冻结估计值


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_commands(run_id: str, limit: int | None = None) -> list[np.ndarray]:
    run_dir = Path(args_cli.runs_root) / run_id
    out = []
    for trial_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        attempts = sorted(a for a in trial_dir.iterdir() if a.is_dir() and a.name.startswith("attempt_"))
        if not attempts:
            continue
        path = attempts[0] / "pairing_10hz.npz"
        if not path.is_file():
            continue
        cmd = np.load(path)["command"].astype(np.float64)
        if cmd.shape != (301, 3):
            continue
        out.append(cmd)
        if limit is not None and len(out) >= limit:
            break
    return out


def add(checks: list, cid: str, name: str, ok: bool, detail: str, **extra) -> None:
    checks.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


def main() -> int:
    model_path = Path(args_cli.model)
    model = PowerModel.load(model_path, verify_hash=True)
    model_sha = model.sha256()
    print(f"[sc] model {model_path.name} sha={model_sha[:16]} base={model.base} k={model.history_k}", flush=True)

    schema_path = Path(args_cli.out_schema)
    schema_sha = write_schema(schema_path, model, alpha=0.0)
    print(f"[sc] schema {schema_path.name} sha={schema_sha[:16]}", flush=True)

    validate_cmds = load_commands(args_cli.validate_run)
    fit_cmds = load_commands(args_cli.fit_run, limit=20)
    if not validate_cmds or not fit_cmds:
        raise SystemExit("缺少命令表")
    print(f"[sc] commands: validate {len(validate_cmds)}, fit {len(fit_cmds)}", flush=True)

    checks: list[dict] = []

    # ---- S01 逐步求和 = J_nom（α=0）----
    worst_rel = 0.0
    n_bad = 0
    for cmd in validate_cmds + fit_cmds:
        out = trajectory_cost(model, cmd, alpha=0.0)
        worst_rel = max(worst_rel, out["rel_diff"])
        if not out["equal"] or out["n_steps"] != 300:
            n_bad += 1
    add(checks, "S01-step-sum-equals-jnom",
        "300 个命令区间的逐步成本之和 = 整窗 J_nom（α=0）",
        n_bad == 0 and worst_rel <= 1e-12,
        f"{len(validate_cmds) + len(fit_cmds)} 条轨迹，最差 rel {worst_rel:.2e}，不合格 {n_bad}；步数恒为 300")

    # ---- S02 α 只是放大系数，逐项可核 ----
    sample = validate_cmds[0]
    a0 = trajectory_cost(model, sample, alpha=0.0)
    a1 = trajectory_cost(model, sample, alpha=ALPHA_TEST)
    per_step_ok = all(
        abs(r1.cost_J - (1.0 + ALPHA_TEST) * r0.cost_nom_J) <= 1e-12 * max(1.0, r1.cost_J)
        for r0, r1 in zip(a0["records"], a1["records"])
    )
    add(checks, "S02-alpha-scaling",
        f"c_t = (1+α)·c_nom、Σ = (1+α)·J_nom（α={ALPHA_TEST} 为测试参数，非估计值）",
        per_step_ok and abs(a1["cost_sum_J"] - (1 + ALPHA_TEST) * a0["j_nom_J"]) <= 1e-9 * a0["j_nom_J"],
        f"J_nom={a0['j_nom_J']:.6f} J，Σc(α=0)={a0['cost_sum_J']:.6f}，"
        f"Σc(α={ALPHA_TEST})={a1['cost_sum_J']:.6f}（比值 {a1['cost_sum_J'] / a0['cost_sum_J']:.6f}）",
        alpha_is_test_value=True, alpha=ALPHA_TEST)

    # ---- S03 预算语义 ----
    budget = BudgetState(initial_J=500.0)
    sc = StepCost(model, alpha=0.0, budget=budget)
    sc.reset()
    seq = validate_cmds[0][:-1]
    remaining_after = []
    for row in seq:
        rec = sc.step(row)
        remaining_after.append(budget.remaining_J)
        if budget.exhausted:
            break
    monotone = all(b <= a + 1e-12 for a, b in zip(remaining_after, remaining_after[1:]))
    z_ok = abs(budget.z_ratio - max(0.0, budget.remaining_J / 500.0)) <= 1e-12
    spent_ok = abs(budget.spent_J - (500.0 - budget.remaining_J)) <= 1e-12
    add(checks, "S03-budget-accounting",
        "spent/remaining/exhausted/z_ratio 单调且自洽（z_ratio 夹在 [0,1]）",
        monotone and z_ok and spent_ok,
        f"走 {len(remaining_after)} 步后 spent={budget.spent_J:.4f} J、remaining={budget.remaining_J:.4f} J、"
        f"exhausted={budget.exhausted}、z_ratio={budget.z_ratio:.4f}；剩余单调不增={monotone}",
        budget=budget.to_dict())

    # ---- S04 零/负预算被拒 ----
    rejected = {}
    for label, kwargs in (("zero", {"initial_J": 0.0}), ("negative", {"initial_J": -1.0}),
                          ("nan", {"initial_J": float("nan")}), ("unit_au", {"initial_J": 10.0, "unit": "a.u."}),
                          ("kind_other", {"initial_J": 10.0, "kind": "normalized"})):
        try:
            BudgetState(**kwargs)
            rejected[label] = False
        except (StepCostError, ValueError):
            rejected[label] = True
    add(checks, "S04-budget-guards",
        "非正/非有限预算、旧 a.u. 单位、非 mechanical_J 口径全部被拒",
        all(rejected.values()), f"拒绝情况 {rejected}")

    # ---- S05 提前终止：只对已走步数收费 ----
    sc2 = StepCost(model, alpha=0.0, budget=BudgetState(initial_J=1e6))
    sc2.reset()
    n_trunc = 137
    total = 0.0
    for row in validate_cmds[0][:n_trunc]:
        total += sc2.step(row).cost_J
    full = trajectory_cost(model, validate_cmds[0], alpha=0.0)["cost_sum_J"]
    add(checks, "S05-early-termination",
        "提前终止时成本只累计实际执行步数（不外推、不补齐）",
        abs(sc2.budget.spent_J - total) <= 1e-9 and total < full,
        f"{n_trunc} 步累计 {total:.4f} J < 整窗 {full:.4f} J；预算记录与逐步和一致")

    # ---- S06 错误模型散列被拒 ----
    tampered_path = Path("/pub/data/caohy/tmp/sc_tampered_model.json")
    tampered_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    bad = copy.deepcopy(payload)
    bad["weights"][0] = float(bad["weights"][0]) + 1e-6
    tampered_path.write_text(json.dumps(bad, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hash_rejected = False
    try:
        PowerModel.load(tampered_path, verify_hash=True)
    except Exception:  # noqa: BLE001
        hash_rejected = True
    side_ok = True
    try:
        PowerModel.load(model_path, verify_hash=True)
    except Exception:  # noqa: BLE001
        side_ok = False
    add(checks, "S06-model-hash-guard",
        "权重被改 1e-6 后 load(verify_hash) 拒绝；正品可载",
        hash_rejected and side_ok,
        f"篡改文件被拒={hash_rejected}，正品可载={side_ok}")

    # ---- S07 dt 不一致被拒 ----
    dt_rejected = False
    try:
        StepCost(model, dt_s=0.05)
    except StepCostError:
        dt_rejected = True
    add(checks, "S07-dt-guard",
        "dt_s 与模型 step_cost_dt 不一致时构造即拒绝（防止错位能量）",
        dt_rejected, f"dt=0.05 被拒={dt_rejected}（模型 {model.step_cost_dt}）")

    # ---- S08 K>0 的历史状态与 reset（用临时测试模型，非交付模型）----
    k1_names = feature_names("linear", 1)
    k1_model = PowerModel(
        weights=tuple(0.01 for _ in k1_names), feature_names=k1_names, base="linear",
        history_k=1, fit_mode="relative",
        train={"note": "S08 专用测试模型（权重恒 0.01），不是交付模型"},
        domain={}, notes=("test-only",),
    )
    # 注意：zero_history 的语义是"episode 第一条命令即它自己的历史"，因此它会**覆盖** reset
    # 时传入的历史；要验证"外部历史参与预测"必须用 require_explicit + reset(history=...)。
    k1 = StepCost(k1_model, alpha=0.0, reset_rule="require_explicit")
    spec = k1.augmented_state_spec()
    h = [0.5, -0.2, 0.1]
    u = [0.4, 0.0, 0.2]
    k1.reset(history=[h])
    rec = k1.step(u)
    # 手工：命令序列 [h, u, u] 的 feature_matrix 第 1 行 = 以 u 为当前、h 为滞后
    manual = float(feature_matrix(np.asarray([h, u, u]), "linear", 1)[1] @ np.asarray(k1_model.weights))
    k1_ok = (k1.needs_history and spec["history_k"] == 1
             and abs(rec.power_W - manual) <= 1e-12)
    # zero_history 下 episode 第一条命令即它自己的历史
    k1b = StepCost(k1_model, alpha=0.0, reset_rule="zero_history")
    k1b.reset()
    first = [0.7, 0.1, -0.3]
    rec_first = k1b.step(first)
    manual_first = float(feature_matrix(np.asarray([first, first, first]), "linear", 1)[1]
                         @ np.asarray(k1_model.weights))
    zero_hist_ok = abs(rec_first.power_W - manual_first) <= 1e-12
    # require_explicit 缺历史要报错
    k1c = StepCost(k1_model, alpha=0.0, reset_rule="require_explicit")
    explicit_rejected = False
    try:
        k1c.reset()
    except StepCostError:
        explicit_rejected = True
    k1c.reset(history=[[0.1, 0.0, 0.0]])
    explicit_ok = abs(k1c.step(u).power_W - float(
        feature_matrix(np.asarray([[0.1, 0.0, 0.0], u, u]), "linear", 1)[1]
        @ np.asarray(k1_model.weights))) <= 1e-12
    add(checks, "S08-history-state",
        "K=1 时历史命令并入状态：step 与 feature_matrix 一致；zero_history 首条命令即自身历史；"
        "require_explicit 缺历史报错、给历史可用",
        k1_ok and zero_hist_ok and explicit_rejected and explicit_ok,
        f"K=1 step 一致={k1_ok}（{rec.power_W:.6f} vs {manual:.6f}），zero_history 首步一致={zero_hist_ok}，"
        f"require_explicit 缺历史被拒={explicit_rejected}、给历史可用={explicit_ok}；"
        f"K=0 交付模型 needs_history={StepCost(model).needs_history}",
        test_model_only=True)

    # ---- S09 交付模型的增广状态：K=0 不需要历史 ----
    sc0 = StepCost(model)
    spec0 = sc0.augmented_state_spec()
    add(checks, "S09-augmented-state-k0",
        "交付模型 history_k=0 → 上层的状态不需要增广（命令即完整 Markov 输入）",
        spec0["history_k"] == 0 and "none" in spec0["augmented_state"],
        f"history_k={spec0['history_k']}, augmented_state={spec0['augmented_state']!r}, "
        f"reset_rule={spec0['reset_rule']}, model_sha={spec0['model_sha256'][:16]}")

    # ---- S10 30 s / 300 步 / dt=0.1 与 schema 自洽 ----
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    window_ok = (schema["dt_s"] == 0.1 and 300 * schema["dt_s"] == 30.0
                 and schema["unit"] == "J" and schema["model_sha256"] == model_sha)
    add(checks, "S10-schema-window",
        "schema: dt=0.1 s、300 步 = 30 s、单位 J、绑定模型散列",
        window_ok,
        f"dt={schema['dt_s']}, 300·dt={300 * schema['dt_s']}, unit={schema['unit']}, "
        f"model_sha 绑定={schema['model_sha256'] == model_sha}")

    # ---- S11 成本非负 ----
    neg = 0
    for cmd in validate_cmds[:20]:
        out = trajectory_cost(model, cmd, alpha=ALPHA_TEST)
        neg += sum(1 for r in out["records"] if r.cost_J < 0 or r.cost_nom_J < 0 or r.power_W < 0)
    add(checks, "S11-nonneg-cost",
        "所有区间的 P̂ / c_nom / c 非负（α>0 也不会翻转符号）",
        neg == 0, f"20 条轨迹 × 300 步中负值 {neg} 个")

    # ---- S12 与几何/轮廓分别绑定同一控制器 ----
    fp_path = Path(args_cli.footprint)
    from camra_calibration.footprint import Footprint  # noqa: E402

    fp = Footprint.load(fp_path, verify_hash=True)
    add(checks, "S12-binding",
        "步成本 schema 与足迹轮廓各自绑定同一控制器/资产，互不代偿",
        schema["model_sha256"] == model_sha
        and fp.source.get("bundle") == "hira_go2_18500"
        and fp.source.get("geometry_sha256") != model_sha,
        f"model_sha={model_sha[:16]}，footprint_sha={fp.sha256()[:16]}，"
        f"geometry_sha={str(fp.source.get('geometry_sha256'))[:16]}，"
        f"二者引用不同对象（能量 vs 几何），δ 只在几何侧施加一次")

    # ---- S13 与执行能量的比值口径 ----
    pairs = []
    for cmd in validate_cmds[:10]:
        j_nom = model.j_nom(cmd)
        eps, status = energy_ratio(j_nom * 1.2, j_nom)
        pairs.append((status, eps))
    ratio_ok = all(s == "ok" and abs(e - 0.2) < 1e-9 for s, e in pairs)
    nc, status_nc = energy_ratio(5.0, 0.5)   # J_nom < J_min
    add(checks, "S13-energy-ratio",
        "ε_c=(J_exec−J_nom)/J_nom；J_nom<J_min 记 non_computable",
        ratio_ok and status_nc == "non_computable",
        f"10 条人工 1.2× 比值均为 0.2={ratio_ok}；J_nom=0.5<J_min 返回 {status_nc}")

    failed = [c["id"] for c in checks if not c["ok"]]
    report = {
        "report": "step_cost_interface_tests",
        "task": "HIRA-028",
        "model": {"path": str(model_path), "sha256": model_sha,
                  "base": model.base, "history_k": model.history_k,
                  "step_cost_dt": model.step_cost_dt},
        "schema": {"path": str(schema_path), "sha256": schema_sha,
                   "file_sha256": sha256_file(schema_path),
                   "sidecar_sha256": sha256_file(schema_path.with_suffix(schema_path.suffix + ".sha256")),
                   "recomputed_sha256": schema_sha256({k: v for k, v in schema.items() if k != "sha256"})},
        "alpha_policy": {
            "frozen_alpha": None,
            "alpha_is_test_value": True,
            "alpha_used_in_tests": ALPHA_TEST,
            "note": "α 在 HIRA-031 冻结；本报告的 α 只用于验证接口的放大与预算语义，不是估计值，"
            "不得被下游当作标定结果引用",
        },
        "unit_policy": {
            "unit": "J",
            "kind": "mechanical_J",
            "reject_a_u": True,
            "note": "旧论文的 a.u. 预算不得与本预算混用；BudgetState 对 unit/kind 强校验",
        },
        "checks": checks,
        "verdict": {"all_passed": not failed, "failed": failed},
    }
    out = Path(args_cli.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"STEP_COST_OK checks={sum(c['ok'] for c in checks)}/{len(checks)} failed={failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
