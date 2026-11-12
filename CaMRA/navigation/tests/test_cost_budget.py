"""Cost and budget adapter checks (NAV-014).

Covers the five things that can silently go wrong here: a double-applied alpha, a
step-wise sum that does not equal the whole-segment energy, a budget normalised by
the episode's own initial value, a joule/ratio unit mix-up, and an execution-energy
ledger that leaks into the model budget.

Run:

    <frozen python> CaMRA/navigation/tests/test_cost_budget.py \
        --json CaMRA/navigation/reports/cost_budget_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation import costs as C  # noqa: E402
from camra_navigation.observation import COMMAND_MAX, Z_REF_J  # noqa: E402

POWER_MODEL = os.path.join(CALIB, "config", "power_model_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
DATA_LOCK = os.path.join(CALIB, "config", "data_lock_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def raises(fn, *a, **kw) -> tuple[bool, str]:
    try:
        fn(*a, **kw)
        return False, "no exception"
    except Exception as exc:  # noqa: BLE001 - the type is part of the assertion
        return True, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # --- 1. margins are loaded, not typed ---------------------------------
    margins = C.load_margins(MARGINS)
    lock = json.load(open(DATA_LOCK, encoding="utf-8"))
    lock_byte = lock["margins"]["sha256"]
    lock_content = lock["margins"]["content_sha256"]
    check(
        "margins_loaded_exactly",
        abs(margins.alpha - 0.19048279965649906) < 1e-18
        and abs(margins.delta_m - 2.6138664115280967) < 1e-12,
        {"alpha": margins.alpha, "delta_m": margins.delta_m, "n": margins.n},
    )
    check(
        "margins_two_hashes_bind_correctly",
        margins.source_sha256 == lock_byte
        and margins.declared_content_sha256 == lock_content
        and margins.source_sha256 != margins.declared_content_sha256,
        {
            "byte_sha256": margins.source_sha256[:16],
            "data_lock_sha256": lock_byte[:16],
            "declared_content_sha256": margins.declared_content_sha256[:16],
            "data_lock_content_sha256": lock_content[:16],
            "note": "字节散列与文件自述散列是两个数，混用会造成 NAV-009 那类误判",
        },
    )

    # --- 2. alpha must be chosen, not inherited ---------------------------
    full = C.JCostModel.from_frozen(POWER_MODEL, MARGINS, variant="full")
    base = C.JCostModel.from_frozen(POWER_MODEL, MARGINS, variant="uncalibrated")
    check(
        "variants_carry_their_own_alpha",
        abs(full.alpha - margins.alpha) < 1e-18 and base.alpha == 0.0
        and full.variant == "full" and base.variant == "uncalibrated",
        {"full_alpha": full.alpha, "baseline_alpha": base.alpha},
    )
    ok1, msg1 = raises(C.JCostModel, full.model, 0.0, variant="full", alpha_source="x")
    ok2, msg2 = raises(C.JCostModel, full.model, 0.19, variant="uncalibrated", alpha_source="x")
    check(
        "alpha_zero_cannot_pose_as_full_method",
        ok1 and ok2,
        {"full_with_alpha_0": msg1, "uncalibrated_with_alpha_0.19": msg2},
    )

    # --- 3. alpha multiplies exactly once ---------------------------------
    rng = random.Random(20261110)
    cmds = [(1.0, 0.0, 0.0)] + [
        (rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(-0.5, 0.5))
        for _ in range(50)
    ]
    budget = C.SignedBudget(z0_J=400.0)
    seg = full.segment_cost(cmds, budget=budget)
    ratios = []
    full.reset()
    for c in cmds[:-1]:
        rec = full.step(c)
        ratios.append(rec.cost_model_J / rec.cost_nom_J)
    worst_ratio = max(abs(r - (1.0 + full.alpha)) for r in ratios)
    check(
        "alpha_applied_once_per_step",
        worst_ratio < 1e-12,
        {"1+alpha": 1.0 + full.alpha, "max_abs_dev": worst_ratio,
         "conclusion": "若叠乘两次，比值会是 (1+alpha)^2 = %.6f" % ((1 + full.alpha) ** 2)},
    )
    segment_ratio = seg["cost_model_J_whole"] / seg["cost_nom_J"]
    check(
        "alpha_applied_once_per_segment",
        abs(segment_ratio - (1.0 + full.alpha)) < 1e-12,
        {"segment_ratio": segment_ratio, "1+alpha": 1.0 + full.alpha},
    )

    # --- 4. step-wise == whole segment ------------------------------------
    check(
        "stepwise_equals_whole",
        seg["abs_diff_J"] < 1e-9 and seg["n_intervals"] == len(cmds) - 1,
        {"n_intervals": seg["n_intervals"], "stepwise_J": seg["cost_model_J_stepwise"],
         "whole_J": seg["cost_model_J_whole"], "abs_diff_J": seg["abs_diff_J"]},
    )

    # --- 5. budget recomputable step by step ------------------------------
    full.reset()
    b = C.SignedBudget(z0_J=100.0)
    manual = 0.0
    worst = 0.0
    for c in cmds[:-1]:
        rec = full.step(c, budget=b)
        manual += rec.cost_model_J
        worst = max(worst, abs(b.z_J - (100.0 - manual)))
    check(
        "budget_stepwise_recomputable",
        worst < 1e-9,
        {"max_abs_dev_J": worst, "spent_J": b.spent_J, "z_J": b.z_J,
         "z_ratio": b.z_ratio, "charges": b.charges},
    )
    check(
        "z_ratio_uses_scene_scale_not_initial_budget",
        abs(b.z_ratio - b.z_J / Z_REF_J) < 1e-15
        and abs(b.z_ratio - b.z_J / b.z0_J) > 0.05,
        {"z_ratio": b.z_ratio, "z_J/Z_REF_J": b.z_J / Z_REF_J,
         "z_J/z0_J_if_wrong": b.z_J / b.z0_J,
         "conclusion": "BudgetState.z_ratio 的 initial_J 分母在这里会得到完全不同的数"},
    )

    # --- 6. over-budget stays signed and visible --------------------------
    over = C.SignedBudget(z0_J=50.0)
    over.charge(35.0)
    over.charge(25.0)
    check(
        "negative_budget_preserved",
        abs(over.z_J + 10.0) < 1e-12 and over.exhausted and not over.affordable
        and abs(over.over_budget_J - 10.0) < 1e-12
        and abs(over.z_ratio - (-10.0 / Z_REF_J)) < 1e-15,
        {**over.to_dict()},
    )
    big = C.SignedBudget(z0_J=10.0)
    big.charge(1e6)
    clip = C.z_ratio_clip()
    check(
        "z_ratio_clipped_but_z_J_not",
        big.z_ratio == -clip and big.z_J < -1e5,
        {"z_ratio": big.z_ratio, "clip": clip, "z_J": big.z_J,
         "note": "ratio 截到 -clip 供观测（obs-v3 为 -1.5）；z_J 保留真实超支"},
    )

    # --- 7. reversible normalisation --------------------------------------
    rnd = C.SignedBudget(z0_J=200.0)
    lim = C.z_ratio_clip()
    zs = [-lim * Z_REF_J, -50.0, 0.0, 75.0, lim * Z_REF_J]
    rt = [rnd.from_ratio(rnd.to_ratio(z)) for z in zs]
    worst_rt = max(abs(a - z) for a, z in zip(rt, zs))
    ok3, msg3 = raises(rnd.from_ratio, lim + 0.25)
    check(
        "normalisation_invertible_inside_range",
        worst_rt < 1e-9 and ok3,
        {"z_values": zs, "round_trip": rt, "max_abs_dev": worst_rt,
         "clip": lim, "outside_range": msg3},
    )

    # --- 8. unit mix-up is detectable by magnitude ------------------------
    budget_ok = C.SignedBudget(z0_J=100.0)
    budget_ok.charge(0.5 * Z_REF_J)      # a ratio of 0.5, converted to joules
    budget_bad = C.SignedBudget(z0_J=100.0)
    budget_bad.charge(0.5)               # the ratio passed straight in as joules
    factor = budget_ok.spent_J / budget_bad.spent_J
    check(
        "unit_mixup_detectable",
        abs(factor - Z_REF_J) < 1e-6 and budget_bad.spent_J < 1.0,
        {"correct_spend_J": budget_ok.spent_J, "wrong_spend_J": budget_bad.spent_J,
         "factor": factor, "Z_REF_J": Z_REF_J,
         "conclusion": "把归一化值当焦耳会把支出低估 225.6 倍，预算永远不会耗尽"},
    )

    # --- 9. illegal costs rejected ----------------------------------------
    bad = C.SignedBudget(z0_J=100.0)
    okn, msgn = raises(bad.charge, -1.0)
    oki, msgi = raises(bad.charge, float("nan"))
    okinf, msginf = raises(bad.charge, float("inf"))
    check(
        "negative_and_nonfinite_cost_rejected",
        okn and oki and okinf and bad.spent_J == 0.0,
        {"negative": msgn, "nan": msgi, "inf": msginf},
    )
    ok_dom, msg_dom = raises(full.step, (1.5, 0.0, 0.0))
    check(
        "out_of_domain_command_rejected",
        ok_dom,
        {"msg": msg_dom[:120], "domain": COMMAND_MAX},
    )

    # --- 10. reset clears the episode, never the model --------------------
    reset_b = C.SignedBudget(z0_J=80.0)
    full.reset()
    for c in cmds[:5]:
        full.step(c, budget=reset_b)
    spent_before = reset_b.spent_J
    reset_b.reset()
    full.reset()
    rec2 = full.step((0.0, 0.0, 0.0), budget=reset_b)
    check(
        "reset_clears_budget_and_clock",
        spent_before > 0.0 and reset_b.charges == 1 and abs(rec2.t_s) < 1e-12
        and abs(reset_b.spent_J - rec2.cost_model_J) < 1e-12,
        {"spent_before_J": spent_before, "spent_after_reset_J": reset_b.spent_J,
         "t_s_after_reset": rec2.t_s, "history_k": full.model.history_k},
    )
    check(
        "history_k_zero_needs_no_augmentation",
        full.model.history_k == 0
        and not any(str(n).startswith("lag") for n in full.model.feature_names),
        {"history_k": full.model.history_k,
         "n_features": len(full.model.feature_names),
         "note": "K=0：当前命令即完整 Markov 输入，观测不需要额外历史字段"},
    )

    # --- 11. execution energy is a separate ledger ------------------------
    exec_e = C.ExecutionEnergy()
    model_b = C.SignedBudget(z0_J=100.0)
    full.reset()
    rec = full.step((0.4, 0.0, 0.2), budget=model_b)
    spent_model = model_b.spent_J
    tau = [8.0, 6.0, 9.0, 7.0] * 3
    qdot = [0.5, -0.4, 0.6, 0.2] * 3
    for _ in range(20):
        exec_e.add_substep(tau, qdot)
    check(
        "execution_ledger_isolated",
        abs(model_b.spent_J - spent_model) < 1e-15 and exec_e.total_J > 0.0
        and abs(model_b.spent_J - rec.cost_model_J) < 1e-12,
        {"model_spent_J": model_b.spent_J, "execution_J": exec_e.total_J,
         "substeps": exec_e.substeps,
         "conclusion": "执行能量只做审计，不参与预算扣减，否则 alpha 失去意义"},
    )
    # negative work must not be credited back
    e2 = C.ExecutionEnergy()
    e2.add_substep([5.0], [0.4])     # +2.0 W
    e2.add_substep([5.0], [-0.4])    # -2.0 W, must be ignored
    check(
        "negative_work_not_credited",
        abs(e2.total_J - (2.0 * 0.005)) < 1e-15,
        {"total_J": e2.total_J, "expected_J": 0.01},
    )

    # --- 12. baseline vs full differ only by alpha ------------------------
    full.reset()
    base.reset()
    same_cmds = [(0.5, 0.1, 0.2)] * 3
    f_seg = full.segment_cost(same_cmds)
    b_seg = base.segment_cost(same_cmds)
    check(
        "baseline_equals_nominal_energy",
        abs(b_seg["cost_model_J_stepwise"] - b_seg["cost_nom_J"]) < 1e-12
        and abs(f_seg["cost_model_J_stepwise"] / b_seg["cost_model_J_stepwise"]
                - (1.0 + margins.alpha)) < 1e-12,
        {"full_J": f_seg["cost_model_J_stepwise"], "baseline_J": b_seg["cost_model_J_stepwise"],
         "ratio": f_seg["cost_model_J_stepwise"] / b_seg["cost_model_J_stepwise"]},
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "unit": C.COST_UNIT,
        "alpha_full": full.alpha,
        "alpha_baseline": base.alpha,
        "z_ref_J": Z_REF_J,
        "model_sha256": full.model.sha256(),
        "margins_sha256": margins.source_sha256,
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:300])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
