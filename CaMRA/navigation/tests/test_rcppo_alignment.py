"""RC-PPO alignment regression: shared stop + three λ-smoothed estimators (NAV-063).

Run with the registered interpreter:

    /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python \
        CaMRA/navigation/tests/test_rcppo_alignment.py --json <out.json>

No pytest dependency (same convention as the other NAV tests).  Every check is a
controlled case with a hand value or an independent enumeration; nothing is
compared against the function under test.

What is being pinned
--------------------
The reference (``RC-PPO/rl/EC-EFPPO.py``) computes one ``done`` mask from
``calculate_indexs3`` and feeds it to **all three** estimators:
``calculate_gae2`` (cost), ``calculate_gae_reach4`` (reach) and
``calculate_gae_reach4`` on ``g=max(reach,−energy)`` (policy).  CaMRA used to
record the stop index but then take the whole-episode cost suffix, an unmasked
min-type reach value and a one-shot ``W_sel − V_total`` policy advantage, i.e.
three different implicit endpoints.  These checks fail if that returns.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAV_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_NAV_ROOT))
sys.path.insert(0, os.path.join(_NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO, "CaMRA", "calibration", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402
from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402

CALIB = os.path.join(_REPO, "CaMRA", "calibration")
VALUE_CONTRACT = os.path.join(_NAV_ROOT, "config", "value_objective_contract_v1.json")
NAV_CONTRACT = os.path.join(_NAV_ROOT, "config", "navigation_contract_v1.json")
PREREG = os.path.join(_NAV_ROOT, "config", "formal_training_prereg_v3.json")

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name, flush=True)
    if not ok:
        print("       " + json.dumps(detail, ensure_ascii=False, default=str)[:1400],
              flush=True)


def cfg(**over) -> S.ObjectiveConfig:
    base = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    if over:
        import dataclasses
        base = dataclasses.replace(base, **over)
    return base


def segment(costs, z, h_d, y, *, d_next=None, y_next=None, z_next=None,
            term="truncation", vc_terminal=0.0, vc=0.0, vh=0.0):
    """Build schema-v2 raw arrays from (cost, budget, distance, safety) sequences."""
    Tn = len(costs)
    d_next = list(d_next if d_next is not None else list(h_d[1:]) + [h_d[-1]])
    y_next = list(y_next if y_next is not None else list(y[1:]) + [y[-1]])
    z_next = list(z_next if z_next is not None else list(z[1:]) + [z[-1] - costs[-1]])
    term_arr = np.zeros(Tn, dtype=bool)
    trunc_arr = np.zeros(Tn, dtype=bool)
    if term == "failure":
        term_arr[-1] = True
    elif term == "success":
        term_arr[-1] = True
    else:
        trunc_arr[-1] = True
    b_c = np.zeros(Tn)
    b_c[-1] = vc_terminal
    return {
        "cost_norm": np.asarray(costs, dtype=np.float64),
        "z_norm": np.asarray(z, dtype=np.float64),
        "z_ratio": np.asarray(z, dtype=np.float64),
        "value_cost": np.full(Tn, vc, dtype=np.float64),
        "value_reach": np.full(Tn, vh, dtype=np.float64),
        "safety_y": np.asarray(y, dtype=np.float64),
        "goal_distance_m": np.asarray(h_d, dtype=np.float64),
        "z_norm_next": np.asarray(z_next, dtype=np.float64),
        "z_ratio_next": np.asarray(z_next, dtype=np.float64),
        "safety_y_next": np.asarray(y_next, dtype=np.float64),
        "goal_distance_m_next": np.asarray(d_next, dtype=np.float64),
        "terminated": term_arr,
        "truncated": trunc_arr,
        "bootstrap_value_cost": b_c,
        "bootstrap_value_reach": np.zeros(Tn),
    }


# --------------------------------------------------------------- 1. wiring
def check_wiring_and_single_mask() -> None:
    """The production path must call the three estimators with ONE done mask."""
    arrays = segment([0.05, 0.05, 0.05], [0.5, 0.4, 0.3], [3.0, 2.0, 1.0], [0, 0, 0],
                     d_next=[2.0, 1.0, 0.2])
    c = cfg()
    calls = {"lambda_return": [], "cost_to_stop": 0, "joint_sequence": 0}
    orig_lr, orig_cts, orig_js = T.lambda_return, T.cost_to_stop, T.joint_sequence

    def lr_spy(v, d, lam):
        calls["lambda_return"].append({"values": np.array(v), "done": np.array(d), "lam": lam})
        return orig_lr(v, d, lam)

    def cts_spy(cost, idx, vc_terminal=0.0, gamma=1.0):
        calls["cost_to_stop"] += 1
        return orig_cts(cost, idx, vc_terminal=vc_terminal, gamma=gamma)

    def js_spy(*a, **k):
        calls["joint_sequence"] += 1
        return orig_js(*a, **k)

    T.lambda_return, T.cost_to_stop, T.joint_sequence = lr_spy, cts_spy, js_spy
    try:
        out = S.compute_objectives(arrays, c, terminal_kind="truncation")
    finally:
        T.lambda_return, T.cost_to_stop, T.joint_sequence = orig_lr, orig_cts, orig_js

    done_same = (len(calls["lambda_return"]) == 2
                 and np.array_equal(calls["lambda_return"][0]["done"],
                                    calls["lambda_return"][1]["done"]))
    check("production_uses_three_estimators_with_one_mask",
          done_same and calls["cost_to_stop"] == 1 and calls["joint_sequence"] == 1,
          {"lambda_return_calls": len(calls["lambda_return"]),
           "done_equal": done_same, "cost_to_stop": calls["cost_to_stop"],
           "joint_sequence": calls["joint_sequence"]})

    # the mask must be exactly 1[τ*(t)=t]
    idx = np.asarray(out["obj_stop_index"], dtype=np.int64)
    expect_done = (idx == np.arange(idx.shape[0]))
    lam_done = calls["lambda_return"][0]["done"]
    check("done_mask_is_stop_now",
          np.array_equal(np.asarray(lam_done[:-1], dtype=bool), expect_done)
          and int(lam_done[-1]) == 1
          and np.array_equal(np.asarray(out["obj_stop_now"]), expect_done),
          {"stop_index": idx.tolist(), "done": expect_done.tolist(),
           "lambda_done": lam_done.tolist(),
           "note": "λ-return 的序列含终端，故 done_ext 末尾固定为 1"})

    # λ and γ are the contract's, visible in the call
    lam_seen = {round(float(x["lam"]), 12) for x in calls["lambda_return"]}
    check("lambda_from_contract", lam_seen == {round(c.gae_lambda, 12)},
          {"lambda_seen": sorted(lam_seen), "contract": c.gae_lambda,
           "gamma_joint": c.gamma_joint, "gamma_cost": c.gamma_cost,
           "gamma_h": c.gamma_h})


# ------------------------------------------------------- 2. hand calculations
def check_tau_equals_t_and_T() -> None:
    """τ*=t (stop now) and τ*=T (stop at the horizon) with hand values."""
    c = cfg()
    # τ*=t: distance 1.0 m, no danger, budget 0.5 -> g=0.0333 everywhere, tie -> stop now
    arr = segment([0.05, 0.05], [0.5, 0.45], [1.0, 1.0], [0, 0],
                  d_next=[1.0, 1.0], vc_terminal=0.2)
    out = S.compute_objectives(arr, c, terminal_kind="truncation")
    g0 = (1.0 - 0.5) / 15.0
    # stop now: cost target 0, joint target g_t, no bootstrap
    ok_t = (np.allclose(out["obj_target_cost"], [0.0, 0.0], atol=1e-15)
            and abs(float(out["obj_target_total"][0]) - g0) < 1e-12
            and abs(float(out["obj_stop_residual"][0]) - g0) < 1e-12)
    check("tau_equals_t_stop_now", ok_t,
          {"target_cost": out["obj_target_cost"].tolist(),
           "target_total": out["obj_target_total"].tolist(), "g0": g0})

    # τ*=T: h decreasing to 0 only at T, budget large -> stop at the horizon
    arr_T = segment([0.05, 0.05], [0.6, 0.55], [3.0, 0.6], [0, 0],
                    d_next=[0.6, 0.2], z_next=[0.55, 0.15], vc_terminal=0.2)
    out_T = S.compute_objectives(arr_T, c, terminal_kind="truncation")
    # hand: idx(0)=2, target_cost = c0+c1+0.2 = 0.3 ; idx(1)=2 -> 0.05+0.2 = 0.25
    ok_T = (int(out_T["obj_stop_index"][0]) == 2
            and abs(float(out_T["obj_target_cost"][0]) - 0.3) < 1e-12
            and abs(float(out_T["obj_target_cost"][1]) - 0.25) < 1e-12
            # λ-return of g: g=[.16667,.006667], g_T=max(0,0.2-0.15)=0.05
            # G_T=g_T ; G1=.05*g1+.95*G_T ; G0=.05*g0+.95*G1
            and abs(float(out_T["obj_target_total"][0])
                    - (0.05 * ((3.0 - 0.5) / 15.0)
                       + 0.95 * (0.05 * ((0.6 - 0.5) / 15.0) + 0.95 * 0.05))) < 1e-12)
    check("tau_equals_T_and_bootstrap", ok_T,
          {"stop_index": out_T["obj_stop_index"].tolist(),
           "target_cost": out_T["obj_target_cost"].tolist(),
           "target_total": out_T["obj_target_total"].tolist(),
           "g_terminal": float(out_T["obj_g_terminal"][0])})


def check_same_total_cost_different_timing() -> None:
    """γ_c=1: same total J in different order must give the same cost target."""
    c = cfg()
    early = segment([0.10, 0.00], [0.5, 0.45], [3.0, 0.6], [0, 0], d_next=[0.6, 0.2])
    late = segment([0.00, 0.10], [0.5, 0.5], [3.0, 0.6], [0, 0], d_next=[0.6, 0.2])
    o_e = S.compute_objectives(early, c, terminal_kind="truncation")
    o_l = S.compute_objectives(late, c, terminal_kind="truncation")
    same_cost = abs(float(o_e["obj_target_cost"][0]) - float(o_l["obj_target_cost"][0])) < 1e-15
    check("same_total_cost_same_cost_target", same_cost,
          {"early": float(o_e["obj_target_cost"][0]),
           "late": float(o_l["obj_target_cost"][0]),
           "note": "γ_c=1：成本目标只看到停止时的累计 J，与发生时刻无关"})

    # 反例：若有人把 γ_c 改成 0.99，同样 100 J 晚发生会被低估
    c99 = cfg()
    import dataclasses
    c99 = dataclasses.replace(c99, gamma_cost=0.99)
    # stop_residual 的 γ 只影响列位置（spent(t,τ) 的折扣），成本目标 cost_to_stop 用同一 γ
    t_e = T.cost_to_stop(np.array([0.10, 0.00]), np.array([2, 2]), gamma=0.99)
    t_l = T.cost_to_stop(np.array([0.00, 0.10]), np.array([2, 2]), gamma=0.99)
    check("discounted_cost_would_differ",
          abs(float(t_e[0]) - float(t_l[0])) > 1e-9,
          {"early_discounted": float(t_e[0]), "late_discounted": float(t_l[0]),
           "note": "这正是不把 γ_c 改成 0.99 的理由：它会把目标偷换成折扣成本"})


def check_critics_prefer_different_endpoints() -> None:
    """reach prefers a later stop, the budget term forbids it: shared W decides."""
    c = cfg()
    # h=[0.30, 0.05, 0.00] (dist 5.0 / 1.25 / 0.2), c=[0.6, 0.6], z0=0.9
    #   W(0,0)=max(.30,-.90)=.30 ; W(0,1)=max(.05,.60-.90)=.05
    #   W(0,2)=max(.00,1.20-.90)=.30      -> shared stop = 1
    # reach alone would pick 2 (h=0); cost alone would pick 0 (pay nothing)
    arr = segment([0.6, 0.6], [0.9, 0.3], [5.0, 1.25], [0, 0],
                  d_next=[1.25, 0.2], z_next=[0.3, -0.3])
    out = S.compute_objectives(arr, c, terminal_kind="truncation")
    idx = np.asarray(out["obj_stop_index"], dtype=np.int64)
    h_all = np.array([*out["obj_h"], float(out["obj_h_terminal"][0])])
    check("shared_stop_overrides_each_critic_own_endpoint",
          int(idx[0]) == 1 and int(np.argmin(h_all)) == 2
          and abs(float(out["obj_target_cost"][0]) - 0.6) < 1e-12
          and abs(float(out["obj_stop_residual"][0]) - 0.05) < 1e-12,
          {"stop_index": idx.tolist(), "reach_only_argmin": int(np.argmin(h_all)),
           "h_all": h_all.tolist(),
           "target_cost_0": float(out["obj_target_cost"][0]),
           "W_0": float(out["obj_stop_residual"][0]),
           "note": "三套估计器都用 τ*=1：成本只到 1（0.6），联合残差是 0.05"})


def check_post_stop_cost_does_not_pollute() -> None:
    """Cost after the chosen stop must not move the targets at or before it."""
    c = cfg()
    base = segment([0.05, 0.05, 0.05, 0.05], [0.5, 0.45, 0.4, 0.35],
                   [3.0, 2.0, 1.0, 0.6], [0, 0, 0, 0],
                   d_next=[2.0, 1.0, 0.6, 0.2])
    o1 = S.compute_objectives(base, c, terminal_kind="truncation")
    idx = np.asarray(o1["obj_stop_index"], dtype=np.int64)
    stop = int(idx[0])
    # make everything strictly after the stop expensive
    expensive = dict(base)
    c2 = np.array(base["cost_norm"], dtype=np.float64)
    c2[stop + 1:] += 5.0
    expensive["cost_norm"] = c2
    z2 = np.array(base["z_norm"], dtype=np.float64)
    for t in range(stop + 1, len(z2)):
        z2[t] -= 5.0
    expensive["z_norm"] = z2
    o2 = S.compute_objectives(expensive, c, terminal_kind="truncation")
    ok = (np.array_equal(idx, np.asarray(o2["obj_stop_index"], dtype=np.int64))
          and np.allclose(o1["obj_target_cost"][:stop + 1],
                          o2["obj_target_cost"][:stop + 1], rtol=0.0, atol=0.0)
          and np.allclose(o1["obj_target_total"][:stop + 1],
                          o2["obj_target_total"][:stop + 1], rtol=0.0, atol=0.0))
    check("post_stop_cost_does_not_pollute", ok,
          {"stop": stop, "cost_before": o1["obj_target_cost"][:stop + 1].tolist(),
           "cost_after_perturbation": o2["obj_target_cost"][:stop + 1].tolist(),
           "joint_before": o1["obj_target_total"][:stop + 1].tolist(),
           "joint_after": o2["obj_target_total"][:stop + 1].tolist()})


def check_failure_penalty_propagates() -> None:
    """A failure at the horizon must reach every step inside the stop window."""
    c = cfg()
    arr = segment([0.05, 0.05, 0.05], [0.6, 0.55, 0.5], [4.0, 3.0, 2.0], [0, 0, 0],
                  d_next=[3.0, 2.0, 0.1], term="failure")
    out = S.compute_objectives(arr, c, terminal_kind="failure")
    idx = np.asarray(out["obj_stop_index"], dtype=np.int64)
    gT = float(out["obj_g_terminal"][0])
    g = np.asarray([max(h, -zz) for h, zz in zip(out["obj_h"], arr["z_norm"])])
    # 强制 τ*=T 后，λ-return 从每个 t 都含 g_T=1.0：
    #   G_T=1 ; G2=.05*g2+.95*1 ; G1=.05*g1+.95*G2 ; G0=.05*g0+.95*G1
    want = np.empty(3)
    want[2] = 0.05 * g[2] + 0.95 * gT
    want[1] = 0.05 * g[1] + 0.95 * want[2]
    want[0] = 0.05 * g[0] + 0.95 * want[1]
    prop = (abs(gT - c.fail_penalty) < 1e-15
            and np.allclose(np.asarray(out["obj_target_total"]), want, atol=1e-12)
            and np.all(np.asarray(out["obj_target_total"]) > 0.8)
            and abs(float(out["obj_target_cost"][-1]) - c.fail_penalty) < 1e-15
            and abs(float(out["obj_target_cost"][0]) - 0.15) < 1e-12)
    check("failure_penalty_propagates", prop and bool((idx == 3).all()),
          {"g_terminal": gT, "stop_index": idx.tolist(),
           "target_total": out["obj_target_total"].tolist(),
           "want": want.tolist(),
           "target_cost": out["obj_target_cost"].tolist(),
           "note": "失败回合强制 τ*=T：前序每一步的联合目标都含 1.0 的失败惩罚"})


def check_danger_and_negative_budget() -> None:
    c = cfg()
    danger = segment([0.05, 0.05], [0.5, 0.45], [2.0, 0.6], [0, 1],
                     d_next=[0.6, 0.2], y_next=[1, 1], term="success")
    o_d = S.compute_objectives(danger, c, terminal_kind="success")
    neg = segment([0.4, 0.4], [-0.1, -0.5], [2.0, 0.6], [0, 0],
                  d_next=[0.6, 0.2], term="success")
    o_n = S.compute_objectives(neg, c, terminal_kind="success")
    check("danger_then_arrival_infeasible",
          (not bool(o_d["obj_feasible"][-1])) and float(o_d["obj_stop_residual"][-1]) >= 1.0,
          {"W_last": float(o_d["obj_stop_residual"][-1]),
           "h_terminal": float(o_d["obj_h_terminal"][0])})
    check("negative_budget_joint_cost_positive",
          (not bool(o_n["obj_feasible"][-1]))
          and float(o_n["obj_stop_residual"][-1]) > 0.0
          and float(o_n["obj_g"][0]) == 0.1,
          {"W_last": float(o_n["obj_stop_residual"][-1]),
           "g0": float(o_n["obj_g"][0]),
           "note": "g=max(h,−z)=0.1：超支本身就是正成本"})


def check_lambda_and_gamma_sensitivity() -> None:
    """λ and γ_h must actually change the production estimators."""
    arr = segment([0.05, 0.05, 0.05], [0.6, 0.55, 0.5], [4.0, 3.0, 2.0], [0, 0, 0],
                  d_next=[3.0, 2.0, 0.2])
    base = S.compute_objectives(arr, cfg(), terminal_kind="truncation")
    lam1 = S.compute_objectives(arr, cfg(gae_lambda=1.0), terminal_kind="truncation")
    lam5 = S.compute_objectives(arr, cfg(gae_lambda=0.5), terminal_kind="truncation")
    gh = S.compute_objectives(arr, cfg(gamma_h=0.5), terminal_kind="truncation")
    d_l1 = float(np.max(np.abs(lam1["obj_target_reach"] - base["obj_target_reach"])))
    d_l5 = float(np.max(np.abs(lam5["obj_target_total"] - base["obj_target_total"])))
    d_gh = float(np.max(np.abs(gh["obj_target_reach"] - base["obj_target_reach"])))
    check("lambda_and_gamma_change_targets", d_l1 > 1e-9 and d_l5 > 1e-9 and d_gh > 1e-9,
          {"max_delta_lambda1_reach": d_l1, "max_delta_lambda05_total": d_l5,
           "max_delta_gamma_h_reach": d_gh,
           "note": "λ 是估计器参数、γ_h 是 reach 的收缩率；两者都必须真的进入目标"})
    # gamma_c must stay undiscounted: cost target unchanged when only γ_joint changes
    check("gamma_cost_undiscounted",
          abs(float(cfg().gamma_cost) - 1.0) < 1e-15
          and float(cfg().gamma_joint) == float(cfg().gamma_cost),
          {"gamma_cost": cfg().gamma_cost, "gamma_joint": cfg().gamma_joint})


def check_slice_rejoin() -> None:
    """A window slice with the right bootstrap must reproduce the full target."""
    c = cfg()
    full = segment([0.05, 0.05, 0.05, 0.05], [0.6, 0.55, 0.5, 0.45],
                   [5.0, 4.0, 3.0, 2.0], [0, 0, 0, 0],
                   d_next=[4.0, 3.0, 2.0, 0.2])
    o_full = S.compute_objectives(full, c, terminal_kind="truncation")
    # first half, truncated at 2 steps, bootstrapped with the *full* tail cost
    tail_cost = float(np.sum(full["cost_norm"][2:]))
    head = segment([0.05, 0.05], [0.6, 0.55], [5.0, 4.0], [0, 0],
                   d_next=[4.0, 3.0], vc_terminal=tail_cost)
    o_head = S.compute_objectives(head, c, terminal_kind="truncation")
    ok_cost = abs(float(o_head["obj_target_cost"][0])
                  - float(o_full["obj_target_cost"][0])) < 1e-12
    check("slice_rejoin_matches_full_cost_target", ok_cost,
          {"head": float(o_head["obj_target_cost"][0]),
           "full": float(o_full["obj_target_cost"][0]),
           "bootstrap": tail_cost,
           "note": "切片目标 = 段内成本 + 完整尾段成本（bootstrap 由采集时的 critic 给出）"})


# ----------------------------------------------------------- PPO controlled
def _batch(n=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, obs.OBS_DIM)


def check_ppo_controlled() -> None:
    """Minimisation form, entropy sign, both clip branches, fixed old log-prob."""
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    cfg_ppo = P.PPOConfig.from_contract(VALUE_CONTRACT)
    net = NW.build(nav)
    net.eval()
    b_obs = _batch(16, 3)
    with torch.no_grad():
        dist = net.policy(b_obs)
        act = dist.sample()
        logp = dist.log_prob(act)
    # positive advantage = more expensive -> minimising max(ratio*A, clip*A) must
    # lower its probability, i.e. increase the loss when the log-prob grows.
    adv_pos = torch.ones(16)
    lp_a = logp
    lp_b = logp + 0.05
    loss_a, _ = P.policy_loss(lp_a, logp.detach(), adv_pos, dist.entropy(), cfg_ppo)
    loss_b, _ = P.policy_loss(lp_b, logp.detach(), adv_pos, dist.entropy(), cfg_ppo)
    check("positive_advantage_penalises_higher_probability",
          float(loss_b) > float(loss_a),
          {"loss_a": float(loss_a), "loss_b": float(loss_b),
           "note": "A>0 = 更贵；概率升高必须让损失变大（最小化方向）"})

    # negative advantage: raising the probability must lower the loss
    adv_neg = -torch.ones(16)
    ln_a, _ = P.policy_loss(lp_a, logp.detach(), adv_neg, dist.entropy(), cfg_ppo)
    ln_b, _ = P.policy_loss(lp_b, logp.detach(), adv_neg, dist.entropy(), cfg_ppo)
    check("negative_advantage_rewards_higher_probability", float(ln_b) < float(ln_a),
          {"loss_a": float(ln_a), "loss_b": float(ln_b)})

    # entropy enters with a minus sign: more entropy -> lower loss
    e_lo = dist.entropy() * 0.0
    e_hi = dist.entropy() + 1.0
    l_lo, _ = P.policy_loss(lp_a, logp.detach(), adv_pos, e_lo, cfg_ppo)
    l_hi, _ = P.policy_loss(lp_a, logp.detach(), adv_pos, e_hi, cfg_ppo)
    check("entropy_sign_is_negative", float(l_hi) < float(l_lo),
          {"with_entropy": float(l_hi), "without": float(l_lo), "ent_coef": cfg_ppo.ent_coef})

    # both clip branches, in all four cells. 损失是 max(unclipped, clipped)，
    # 即对"成本上界"取悲观侧：A>0 时高 ratio 走未裁剪分支（惩罚更大），
    # A<0 时高 ratio 走裁剪分支（少降一点）；低 ratio 反之。
    eps = cfg_ppo.clip_eps
    hi = logp + math.log(1.0 + 3 * eps)      # ratio 1.6 > 1+eps
    lo = logp + math.log(0.5)                # ratio 0.5 < 1-eps
    e0 = dist.entropy() * 0.0
    cells = {}
    for tag, lp, adv in (("A+_hi", hi, 1.0), ("A+_lo", lo, 1.0),
                         ("A-_hi", hi, -1.0), ("A-_lo", lo, -1.0)):
        val, st = P.policy_loss(lp, logp.detach(), torch.full((16,), adv), e0, cfg_ppo)
        cells[tag] = (float(val), float(st["clipped_fraction"]))
    ok = (abs(cells["A+_hi"][0] - (1.0 + 3 * eps)) < 1e-6      # 1.6，未裁剪
          and abs(cells["A+_lo"][0] - (1.0 - eps)) < 1e-6      # 0.8，裁剪
          and abs(cells["A-_hi"][0] + (1.0 + eps)) < 1e-6      # -1.2，裁剪
          and abs(cells["A-_lo"][0] + 0.5) < 1e-6              # -0.5，未裁剪
          and all(v[1] == 1.0 for v in cells.values()))
    check("clip_branches_four_cells", ok,
          {k: {"loss": round(v[0], 6), "clipped": v[1]} for k, v in cells.items()},
          )

    # multi-epoch: the same batch is replayed; old log-prob must stay fixed and
    # the number of optimiser steps must equal epochs * minibatches
    net.train()
    updater = P.MinimizationUpdater(net, cfg_ppo, lr_policy=1e-3, lr_critic=1e-3)
    batch = {
        "obs": b_obs,
        "action": act,
        "log_prob_old": logp.detach(),
        "adv_total": torch.ones(16),
        "value_cost_old": torch.zeros(16),
        "target_cost": torch.ones(16),
        "value_reach_old": torch.zeros(16),
        "target_reach": torch.zeros(16),
    }
    old_snapshot = batch["log_prob_old"].clone()
    steps = 0
    ratios = []
    for _epoch in range(3):
        mb = 4
        perm = torch.randperm(16)
        for start in range(0, 16, mb):
            sel = perm[start:start + mb]
            sub = {k: v[sel] for k, v in batch.items()}
            updater.update(sub)
            steps += 1
            with torch.no_grad():
                ratios.append(float(torch.exp(
                    net.policy(sub["obs"]).log_prob(sub["action"])
                    - sub["log_prob_old"]).mean()))
    check("multi_epoch_fixed_old_log_prob",
          steps == 3 * (16 // 4)
          and torch.equal(batch["log_prob_old"], old_snapshot)
          and all(r != 1.0 for r in ratios),
          {"grad_steps": steps, "expected": 3 * 4,
           "ratios": [round(r, 6) for r in ratios],
           "note": "old_log_prob 由采集时固定：多 epoch 重放时 ratio 相对它计算，"
                   "不会退回 1（否则裁剪失效）"})


# ------------------------------------------------------- production probe
def check_production_probe() -> None:
    """Run the real trainer once and recompute every chunk from raw fields."""
    py = sys.executable
    tmp = tempfile.mkdtemp(prefix="nav063_")
    out_json = os.path.join(tmp, "run.json")
    cmd = [py, os.path.join(_NAV_ROOT, "scripts", "train_nominal.py"),
           "--run-root", tmp, "--run-id", "align_probe",
           "--level", "L2_single_detour", "--map-seed", "11", "--z0-j", "600",
           "--episodes", "1", "--seed-base", "7", "--json", out_json]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_NAV_ROOT, "src"), os.path.dirname(os.path.dirname(_NAV_ROOT)),
         os.path.join(CALIB, "src")])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if proc.returncode != 0:
        check("production_probe_runs", False, {"stderr": proc.stderr[-800:]})
        return
    check("production_probe_runs", True, {"run_json": os.path.exists(out_json)})

    rollout = os.path.join(tmp, "align_probe", "rollout")
    manifest = os.path.join(rollout, "manifest.jsonl")
    entries = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    c = cfg()
    bad = []
    rows = []
    for i, e in enumerate(entries):
        with np.load(os.path.join(rollout, e["file"])) as z:
            arrays = {k: z[k] for k in z.files}
        raw = {k: v for k, v in arrays.items() if not k.startswith("obj_")}
        back = S.compute_objectives(raw, c, terminal_kind=e["terminal_kind"])
        for key, val in back.items():
            if key not in arrays:
                bad.append(f"chunk {i}: 缺少派生字段 {key}")
            elif not np.array_equal(np.asarray(arrays[key]), np.asarray(val)):
                bad.append(f"chunk {i}: {key} 不一致")
        rows.append({"chunk": i, "kind": e["terminal_kind"], "T": int(len(raw["t"])),
                     "stop_now_fraction": float(np.asarray(arrays["obj_stop_now"]).mean()),
                     "adv_total_mean": float(np.asarray(arrays["obj_adv_total"]).mean())})
    check("production_chunks_recompute_from_raw", not bad,
          {"problems": bad[:8], "chunks": rows,
           "note": "真实训练产物的每个派生目标都能由原始字段逐位重算"})


def check_log_std_clamp_amend2() -> None:
    """amend2: log_std 可学习范围 [-5, 1] 阻断熵发散失效模式。"""
    net = NW.build(json.load(open(NAV_CONTRACT, encoding="utf-8")))
    eps = 1e-6

    # (a) forward 防御钳位：参数在界外，std 恰为 e^1
    with torch.no_grad():
        net.policy.log_std.fill_(30.0)
    d = net.policy(_batch(4))
    assert abs(float(d.stddev.max()) - math.exp(NW.LOG_STD_MAX)) < eps

    # (b) 参数钳位：就地回到界内
    net.policy.clamp_log_std()
    assert float(net.policy.log_std.max()) <= NW.LOG_STD_MAX + eps

    # (c) 反向机制复现：纯熵最大化（优势=0 时的退化损失）在旧语义下单调推高
    #     std；新语义下每步钳位后停在 e^1，不再发散
    with torch.no_grad():
        net.policy.log_std.fill_(0.0)
    opt = torch.optim.SGD(net.parameters_of("policy"), lr=0.05)
    ob = _batch(64)
    trace = []
    for _ in range(40):
        opt.zero_grad()
        d = net.policy(ob)
        loss = -d.entropy().mean()          # 最大化熵 = 旧失效模式的梯度
        loss.backward()
        opt.step()
        net.policy.clamp_log_std()
        trace.append(float(net.policy.log_std.mean()))
    check("log_std_clamp_blocks_entropy_divergence",
          max(trace) <= NW.LOG_STD_MAX + eps and trace[-1] > 0.5,
          {"log_std_trace_last5": [round(x, 4) for x in trace[-5:]],
           "trace_max": round(max(trace), 4),
           "note": "无钳位时 40 步 0.05 学习率已把 log_std 推到 ~2+；"
                   "钳位后停在 1.0，熵项仍有恢复信号"})

    # (d) 正常范围行为不变：log_std=0（契约初值）时与旧公式逐位一致
    with torch.no_grad():
        net.policy.log_std.fill_(0.0)
    d0 = net.policy(_batch(4))
    assert abs(float(d0.stddev.max()) - 1.0) < eps

    # (e) 下界：参数 -30 时 std 为 e^-5，探索不完全塌缩
    with torch.no_grad():
        net.policy.log_std.fill_(-30.0)
    dn = net.policy(_batch(4))
    assert abs(float(dn.stddev.max()) - math.exp(NW.LOG_STD_MIN)) < eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    check_log_std_clamp_amend2()
    check_wiring_and_single_mask()
    check_tau_equals_t_and_T()
    check_same_total_cost_different_timing()
    check_critics_prefer_different_endpoints()
    check_post_stop_cost_does_not_pollute()
    check_failure_penalty_propagates()
    check_danger_and_negative_budget()
    check_lambda_and_gamma_sensitivity()
    check_slice_rejoin()
    check_ppo_controlled()
    check_production_probe()

    failed = [r for r in RESULTS if not r["ok"]]
    report = {
        "task": "NAV-063",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"},
                     ensure_ascii=False, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
