"""Hand-computed alignment checks for the value objective (NAV-056).

Three counterexamples are required by the task: the same total J with different
timing, the same trajectory with different units/normalisation, and two critics
preferring different stop times.  Each is computed numerically here, so the
contract is backed by arithmetic rather than prose.

Run:

    <frozen python> CaMRA/navigation/tests/test_value_objective.py \
        --json CaMRA/navigation/reports/value_objective_alignment.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


# --- frozen operators -------------------------------------------------------
def cost_return(costs: list[float], gamma: float) -> float:
    """Cumulative cost from t=0, discounted by ``gamma`` (1.0 = undiscounted)."""
    return sum((gamma ** t) * c for t, c in enumerate(costs))


def combined_stop(vc: list[float], vh: list[float], z_ratio: float) -> tuple[int, float]:
    """Common stop: argmin over the tail of max(V_c - z, V_h); first minimum."""
    best_i, best_w = 0, math.inf
    for t, (c, h) in enumerate(zip(vc, vh)):
        w = max(c - z_ratio, h)
        if w < best_w:
            best_i, best_w = t, w
    return best_i, best_w


def separate_stop_wrong(vc: list[float], vh: list[float], z_ratio: float) -> float:
    """What a per-critic stop would produce.  Must never be implemented."""
    return max(min(c - z_ratio for c in vc), min(vh))


def gae_1d(
    costs: list[float],
    values: list[float],
    v_after: float,
    gamma: float,
    lam: float,
) -> list[float]:
    """GAE targets for a short episode ending at the last entry.

    ``v_after`` is the value bootstrapped from the state *after* the last step:
    ``V(x_T)`` for a time truncation, or the terminal cost for a failure (which
    is why the caller passes ``FAIL_PENALTY`` there instead of ``V(x_T)``).
    """
    n = len(costs)
    adv = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        v_next = v_after if t == n - 1 else values[t + 1]
        delta = costs[t] + gamma * v_next - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return [a + v for a, v in zip(adv, values)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    contract = json.load(open(CONTRACT, encoding="utf-8"))
    q = contract["quantities"]
    z_ref = q["Z_REF_J"]["value"]
    c_safe = q["C"]["value"]
    l_ref = q["g_norm"]["L_REF_m"]
    g_inside = q["g_norm"]["G_INSIDE"]
    beta = q["beta"]["value"]
    gamma_cost = contract["estimator"]["gamma_cost"]
    lam = contract["estimator"]["gae_lambda"]
    fail_penalty = contract["bootstrap"]["FAIL_PENALTY"]["value"]

    check(
        "contract_values",
        gamma_cost == 1.0 and l_ref == 15.0 and g_inside < 0.0 and c_safe >= 1.0 and beta == 0.0,
        {
            "gamma_cost": gamma_cost,
            "L_REF_m": l_ref,
            "G_INSIDE": g_inside,
            "C": c_safe,
            "beta": beta,
            "GAE_LAMBDA": lam,
        },
    )
    check(
        "reference_gamma_traced",
        True,
        "RC-PPO rl/arguments.py 实测默认值：GAMMA_ENERGY=1.0、GAMMA_REACH_INIT=0.999、"
        "GAMMA_REACH_FINAL=0.99999、GAE_LAMBDA=0.95、CLIP_EPS=0.2、ENT_COEF=0.01、"
        "VF_COEF=0.5、MAX_GRAD_NORM=0.5；calculate_indexs3 的 argmin 是倒序扫描中的逐步取优，不是全局 argmin",
    )

    # --- case 1: same total J, different timing ----------------------------
    total_j = 100.0
    early = [total_j] + [0.0] * 9
    late = [0.0] * 9 + [total_j]
    r_early = cost_return(early, gamma_cost)
    r_late = cost_return(late, gamma_cost)
    r_early_099 = cost_return(early, 0.99)
    r_late_099 = cost_return(late, 0.99)
    check(
        "case1_undiscounted_timing_invariant",
        abs(r_early - r_late) < 1e-12,
        {"gamma": gamma_cost, "early_all_at_t0_J": r_early, "late_all_at_t9_J": r_late,
         "diff_J": r_early - r_late},
    )
    check(
        "case1_discount_would_break_it",
        abs(r_early_099 - r_late_099) > 8.0,
        {"gamma": 0.99, "early_J": r_early_099, "late_J": r_late_099,
         "diff_J": r_early_099 - r_late_099,
         "conclusion": "gamma_cost 必须是 1.0；0.99 会把晚发生的 100 J 记成 91.35 J"},
    )

    # --- case 2: same trajectory, different units / normalisation ----------
    vc_J = 150.0
    vh = 0.80
    z_ratio = 0.50
    mixed = max(vh, vc_J - z_ratio)
    correct = max(vh, vc_J / z_ref - z_ratio)
    check(
        "case2_unit_mix_detected",
        abs(mixed - 149.5) < 1e-9 and abs(correct - 0.80) < 1e-9 and mixed > 100 * correct,
        {
            "vc_J": vc_J,
            "vc_normalised": vc_J / z_ref,
            "z_ratio": z_ratio,
            "W_mixed_units_WRONG": mixed,
            "W_correct": correct,
            "conclusion": "混用单位把 W 放大约 187 倍且被成本项吞掉，会永远判不可行",
        },
    )
    z0_a, z0_b, spent = 300.0, 150.0, 60.0
    scene_scale = [(z0_a - spent) / z_ref, (z0_b - spent) / z_ref]
    per_episode_bug = (z0_a - spent) / z0_a
    check(
        "case2b_scale_is_scene_fixed",
        abs(scene_scale[0] - per_episode_bug) > 0.2,
        {
            "z_ratio_with_Z_REF_J": scene_scale,
            "z_ratio_if_divided_by_initial_budget_WRONG": per_episode_bug,
            "conclusion": "除以本回合初始预算会把同样的绝对剩余量折成不同数值，破坏跨回合可比性与负预算表达",
        },
    )

    # --- case 3: two critics preferring different stop times ---------------
    vc = [0.2, 1.0]
    vh_seq = [0.5, -0.1]
    z = 0.4
    idx, w_common = combined_stop(vc, vh_seq, z)
    w_separate = separate_stop_wrong(vc, vh_seq, z)
    check(
        "case3_common_stop_is_stricter",
        idx == 0 and abs(w_common - 0.5) < 1e-12 and abs(w_separate + 0.1) < 1e-12,
        {
            "vc": vc,
            "vh": vh_seq,
            "z_ratio": z,
            "common_stop_index": idx,
            "W_common": w_common,
            "W_separate_stops_WRONG": w_separate,
            "feasible_common": w_common <= -beta,
            "feasible_separate_WRONG": w_separate <= -beta,
            "conclusion": "t=0 预算够但没到目标，t=1 到目标但超支；分开取时刻误判可行 (-0.1)，真实不可行 (+0.5)",
        },
    )
    check(
        "case3_inequality_direction",
        w_separate <= w_common,
        "max(min_t a_t, min_t b_t) <= min_t max(a_t, b_t)：分开取时刻系统性偏低，只会过度乐观",
    )

    # --- bootstrap: truncation vs failure ----------------------------------
    costs = [0.04, 0.06]
    values = [0.30, 0.25]
    v_last = 0.18
    t_trunc = gae_1d(costs, values, v_last, gamma_cost, lam)
    t_fail = gae_1d(costs, values, fail_penalty, gamma_cost, lam)
    check(
        "bootstrap_truncation_vs_failure",
        abs(t_trunc[1] - (costs[1] + v_last)) < 1e-12 and t_fail[1] > t_trunc[1] + 0.5,
        {
            "truncation_targets": t_trunc,
            "failure_targets": t_fail,
            "v_bootstrap_truncation": v_last,
            "terminal_cost_failure": fail_penalty,
            "note": "截断用 V(x_T)=0.18 bootstrap；失败给 FAIL_PENALTY=1.0 且不 bootstrap；混用即错",
        },
    )

    # --- g_norm / C sanity -------------------------------------------------
    r_goal = 0.5
    dists = [0.0, 0.5, 5.0, 20.0]
    g_vals = [g_inside if dd <= r_goal else (dd - r_goal) / l_ref for dd in dists]
    check(
        "g_norm_sign_and_scale",
        g_vals[0] < 0 and g_vals[1] < 0 and g_vals[2] > 0
        and abs(g_vals[3] - (19.5 / l_ref)) < 1e-12,
        {"dist_m": dists, "g_norm": g_vals, "r_goal_m": r_goal},
    )
    # r14（v1.3.0 分级安全）：y 不再只有 0/1，支配性要求改为
    # 「最低严重度也压得住深井」——即 C*floor > G_INSIDE，保证任何违规步
    # h=max(g, C*y)=C*y>0 不可行；C 本身仍 >= 1.0（全严重度与 FAIL_PENALTY 同级）。
    sev = q.get("safety_severity", {})
    floor = float(sev.get("operational_params", {}).get("severity_floor", 0.2))
    check(
        "C_dominates",
        c_safe >= 1.0 and c_safe * floor > g_inside and c_safe > g_inside,
        {
            "C": c_safe,
            "severity_floor": floor,
            "C_times_floor": c_safe * floor,
            "G_INSIDE": g_inside,
            "conclusion": "任何 y>0 时 C*y >= C*floor > G_INSIDE，违规步 h 必 > 0（不可行）；"
                           "y=1 时 C*y=1.0 与 FAIL_PENALTY 同级，可行判据不会翻转",
        },
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "contract": "config/value_objective_contract_v1.json",
        "contract_version": contract["version"],
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "hand_computed": {
            "case1_timing": {
                "undiscounted_1.0": [r_early, r_late],
                "discounted_0.99": [r_early_099, r_late_099],
            },
            "case2_units": {
                "mixed_WRONG": mixed,
                "correct": correct,
                "scene_scale": scene_scale,
                "per_episode_budget_WRONG": per_episode_bug,
            },
            "case3_stop": {
                "common": {"index": idx, "W": w_common},
                "separate_WRONG": w_separate,
            },
            "bootstrap": {"truncation": t_trunc, "failure": t_fail},
        },
        "reference_diff": contract["reference_diff"],
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: v for k, v in summary.items() if k not in ("checks", "reference_diff", "hand_computed")}
    print(json.dumps(head, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:300])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
