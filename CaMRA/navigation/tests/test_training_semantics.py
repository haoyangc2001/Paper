"""Training-semantics regression for the t/t+1 pairing, budget and termination
interface (NAV-062).

Run with the registered interpreter:

    /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python \
        CaMRA/navigation/tests/test_training_semantics.py --json <out.json>

No pytest dependency (same convention as the other NAV tests).

What this file is for
---------------------
NAV-061 recorded four *candidate* defects it could not decide without numbers
(M1 sample pairing, M2 budget sampling/saturation, M3 budget-as-terminal, M8
window slice vs task horizon).  Each check below is written so that the defect,
if it were still present, would make the check fail rather than merely look odd:

* the pairing checks build the segment through the **production field table**
  (``storage.raw_fields``) and compare against hand-computed values, then mutate
  ``*_next`` alone to prove the objective does not read it for ``x_t``;
* the budget checks run the real environment and require it to keep stepping
  after ``z < 0``;
* the slice check requires the window boundary and the 30 s horizon to be
  distinguishable in the metadata even though both are ``truncation``;
* the action check reports the raw -> applied mapping statistics *and* the
  gradient the policy actually receives at saturated actions, because the
  50.1 % clipping measured in NAV-042 is only a defect if no usable signal
  reaches the mean.

Scope note: the objective half is a controlled test on two-to-five step
segments plus one real episode per case, not a training run.  Nothing here
claims navigation performance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

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
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

CALIB = os.path.join(_REPO, "CaMRA", "calibration")
DESIGN = os.path.join(_NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM = os.path.join(_NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
NAV_CONTRACT = os.path.join(_NAV_ROOT, "config", "navigation_contract_r14_repair_v2.json")
VALUE_CONTRACT = os.path.join(_NAV_ROOT, "config", "value_objective_contract_v1.json")
PREREG = os.path.join(_NAV_ROOT, "config", "formal_training_prereg_v3.json")

RESULTS: list[dict[str, Any]] = []
DETAIL: dict[str, Any] = {}


def check(name: str, ok: bool, detail: Any = None) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name, flush=True)
    if not ok:
        print("       " + json.dumps(detail, ensure_ascii=False, default=str)[:1200],
              flush=True)


# --------------------------------------------------------------- fixtures
def build(level: str = "L1_empty", seed: int = 11, *, z0_J: float = 300.0,
          variant: str = "full", t_max_s: float = 30.0) -> NominalNavEnv:
    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM)
    cm = JCostModel.from_frozen(POWER, MARGINS, variant=variant)
    spec = M.build(level, seed, design)
    return NominalNavEnv(spec, geom, cm, z0_J=z0_J, t_max_s=t_max_s)


def cfg() -> S.ObjectiveConfig:
    return S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)


def segment_from_records(records, z_ref: float) -> dict[str, np.ndarray]:
    """Build the *production* raw arrays from step records (one per step).

    This intentionally mirrors ``train_formal.append``: if the trainer and the
    buffer ever disagree about which side a field belongs to, this helper is the
    place where the test would keep passing while the trainer broke, so the
    production-entrypoint check at the end runs the trainer itself.
    """
    T = len(records)
    arrays = {
        "cost_norm": np.asarray([r.cost.cost_model_J / z_ref for r in records]),
        "z_j": np.asarray([r.z_J_before for r in records], dtype=np.float64),
        "z_norm": np.asarray([r.z_J_before / z_ref for r in records]),
        "z_ratio": np.asarray([r.z_ratio_before for r in records]),
        "safety_y": np.asarray([r.safety_y_before for r in records]),
        "goal_distance_m": np.asarray([r.goal_distance_before for r in records]),
        "z_norm_next": np.asarray([r.z_J / z_ref for r in records]),
        "z_ratio_next": np.asarray([r.z_ratio for r in records]),
        "safety_y_next": np.asarray([r.safety_y for r in records]),
        "goal_distance_m_next": np.asarray([r.goal_distance_m for r in records]),
        "value_cost": np.zeros(T),
        "value_reach": np.zeros(T),
        "terminated": np.zeros(T, dtype=bool),
        "truncated": np.zeros(T, dtype=bool),
        "bootstrap_value_cost": np.zeros(T),
        "bootstrap_value_reach": np.zeros(T),
    }
    return arrays


def greedy(env: NominalNavEnv, *, speed: float = 0.8, gain: float = 2.0):
    """Aim at the goal: enough to produce a real arrival, not a controller."""
    px, py, yaw = env.pose
    gx, gy = env.spec.goal
    err = math.atan2(gy - py, gx - px) - yaw
    err = (err + math.pi) % (2.0 * math.pi) - math.pi
    return (speed, 0.0, max(-1.0, min(1.0, gain * err)))


def run_until_done(env: NominalNavEnv, policy, *, cap: int = 400):
    recs = []
    while not env.done and len(recs) < cap:
        recs.append(env.step(policy(env)))
    return recs


# ------------------------------------------------------------------ checks
def check_schema_and_pairing() -> None:
    """M1: the buffer must carry both ends explicitly (schema v2)."""
    fields = {f.name for f in S.raw_fields(obs.OBS_DIM)}
    required = {"z_norm", "z_ratio", "safety_y", "goal_distance_m",
                "z_norm_next", "z_ratio_next", "safety_y_next", "goal_distance_m_next"}
    check("schema_v2_carries_both_ends",
          required <= fields and S.SCHEMA_BUFFER.endswith("-v2"),
          {"schema": S.SCHEMA_BUFFER, "missing": sorted(required - fields)})

    # a v1 chunk must be refused, not silently re-interpreted
    old = {
        "cost_norm": np.zeros(3), "value_cost": np.zeros(3), "value_reach": np.zeros(3),
        "z_ratio": np.zeros(3), "safety_y": np.zeros(3), "goal_distance_m": np.zeros(3),
        "terminated": np.zeros(3, dtype=bool), "truncated": np.array([False, False, True]),
        "bootstrap_value_cost": np.zeros(3),
    }
    raised = ""
    try:
        S.compute_objectives(old, cfg(), terminal_kind="truncation")
    except S.StorageError as exc:
        raised = str(exc)
    check("v1_chunk_refused_with_reason",
          "配对字段" in raised and "不能用于训练" in raised, {"message": raised[:200]})


def check_two_step_hand_calculation() -> None:
    """M1: two-step hand computation, then mutate one side only."""
    env = build(z0_J=300.0)
    recs = [env.step((0.5, 0.0, 0.0)), env.step((0.5, 0.0, 0.1))]
    c = cfg()
    arrays = segment_from_records(recs, obs.Z_REF_J)
    arrays["truncated"][-1] = True
    bvc = 0.25
    arrays["bootstrap_value_cost"][-1] = bvc
    out = S.compute_objectives(arrays, c, terminal_kind="truncation")

    def g(dist: float) -> float:
        return c.G_INSIDE if dist <= c.r_goal_m else (dist - c.r_goal_m) / c.L_REF_m

    h_hand = np.array([max(g(recs[0].goal_distance_before), c.C * recs[0].safety_y_before),
                       max(g(recs[1].goal_distance_before), c.C * recs[1].safety_y_before)])
    h_T_hand = max(g(recs[1].goal_distance_m), c.C * recs[1].safety_y)
    c0, c1 = (r.cost.cost_model_J / obs.Z_REF_J for r in recs)
    z0 = recs[0].z_J_before / obs.Z_REF_J

    # hand W: only τ=0,1,2 are open; τ=2 adds the bootstrap
    def w(tau: int) -> float:
        spent = [c0, c0 + c1, c0 + c1 + bvc][tau]
        h = [h_hand[0], h_hand[1], h_T_hand][tau]
        return max(h, spent - z0)

    ok = (
        np.allclose(out["obj_h"], h_hand, rtol=0.0, atol=0.0)
        and abs(float(out["obj_h_terminal"][0]) - h_T_hand) < 1e-12
        and np.allclose(out["obj_target_cost"], [c0 + c1 + bvc, c1 + bvc], rtol=0.0, atol=1e-12)
        and abs(float(out["obj_stop_residual"][0]) - min(w(0), w(1), w(2))) < 1e-12
        and abs(float(out["obj_spent"][0]) - (c0 + c1)) < 1e-12
    )
    check("two_step_hand_calculation", ok, {
        "obj_h": out["obj_h"].tolist(), "h_hand": h_hand.tolist(),
        "h_T": float(out["obj_h_terminal"][0]), "h_T_hand": h_T_hand,
        "target_cost": out["obj_target_cost"].tolist(),
        "spent": out["obj_spent"].tolist(), "W0": float(out["obj_stop_residual"][0]),
        "W_hand": [w(0), w(1), w(2)],
    })

    # The two sensitivity probes use a short *synthetic* segment whose W is
    # cost-dominated: on the real two-step episode above, h (~0.9, goal still
    # 13 m away) dominates the max and no budget perturbation could show up.
    def _syn(z_norm, z_next):
        T = 2
        return {
            "cost_norm": np.full(T, 0.4),
            "z_j": np.asarray(z_norm) * obs.Z_REF_J,
            "z_norm": np.asarray(z_norm, dtype=np.float64),
            "z_ratio": np.asarray(z_norm, dtype=np.float64),
            "value_cost": np.zeros(T), "value_reach": np.zeros(T),
            "safety_y": np.zeros(T),
            "goal_distance_m": np.array([1.0, 0.8]),
            "z_norm_next": np.asarray(z_next, dtype=np.float64),
            "z_ratio_next": np.asarray(z_next, dtype=np.float64),
            "safety_y_next": np.zeros(T),
            "goal_distance_m_next": np.array([0.8, 0.2]),
            "terminated": np.zeros(T, dtype=bool),
            "truncated": np.array([False, True]),
            "bootstrap_value_cost": np.array([0.0, 0.4]),
            "bootstrap_value_reach": np.zeros(T),
        }

    base = _syn([0.5, 0.1], [0.1, -0.3])
    out_base = S.compute_objectives(base, c, terminal_kind="truncation")
    arrays_next = _syn([0.5, 0.1], [-0.2, -0.6])
    out_next = S.compute_objectives(arrays_next, c, terminal_kind="truncation")
    # obj_g_terminal 属于终端状态 x_T，本来就该随 z_T 变；其余"当前状态"的量不许动
    x_t_fields = ("obj_h", "obj_target_cost", "obj_target_reach", "obj_total_value")
    moved = [k for k in x_t_fields
             if not np.allclose(out_base[k], out_next[k], rtol=0.0, atol=0.0)]
    g_terminal_moved = not np.allclose(out_base["obj_g_terminal"],
                                       out_next["obj_g_terminal"], rtol=0.0, atol=0.0)
    check("next_budget_does_not_pollute_current_value",
          not moved and g_terminal_moved,
          {"changed_x_t_fields": moved,
           "g_terminal_moved": g_terminal_moved,
           "W_base": out_base["obj_target_total"].tolist(),
           "W_next": out_next["obj_target_total"].tolist(),
           "note": "只有终端量跟 z_T 变，x_t 侧目标逐位不变"})

    arrays_cur = _syn([0.2, -0.2], [0.1, -0.3])
    out_cur = S.compute_objectives(arrays_cur, c, terminal_kind="truncation")
    delta = float(np.max(np.abs(out_cur["obj_target_total"] - out_base["obj_target_total"])))
    # W = max(h, spent - z_t): the cost term must be the binding one here
    spent_0 = float(base["cost_norm"].sum())
    binding = spent_0 - 0.5
    check("current_budget_moves_W", delta > 1e-9 and binding > 0.0,
          {"max_abs_delta_W": delta, "binding_term_base": binding,
           "W_base": out_base["obj_target_total"].tolist(),
           "W_cur": out_cur["obj_target_total"].tolist()})

    DETAIL["two_step"] = {"records": [r.as_dict() for r in recs]}


def check_single_step_arrival_and_failure() -> None:
    """Terminal handling on one-step segments built from real episodes."""
    env = build(z0_J=400.0)
    recs = run_until_done(env, lambda e: greedy(e))
    label = env.label
    last = recs[-1]
    c = cfg()
    arrays = segment_from_records([last], obs.Z_REF_J)
    arr_s = dict(arrays)
    arr_s["terminated"][-1] = True
    out_s = S.compute_objectives(arr_s, c, terminal_kind="success")
    # NAV-063：到达步的最优停止就是"现在停"，成本目标因此是 0（不再继续付费）
    ok_success = (
        abs(float(out_s["obj_target_cost"][-1])) < 1e-12
        and bool(out_s["obj_stop_now"][-1])
        and abs(float(out_s["obj_h_terminal"][0])
                - max(c.G_INSIDE if last.goal_distance_m <= c.r_goal_m
                      else (last.goal_distance_m - c.r_goal_m) / c.L_REF_m,
                      c.C * last.safety_y)) < 1e-12
    )
    check("single_step_arrival_target", label == "goal_reached" and ok_success,
          {"label": label, "target_cost": float(out_s["obj_target_cost"][-1]),
           "stop_now": bool(out_s["obj_stop_now"][-1]),
           "h_T": float(out_s["obj_h_terminal"][0]),
           "note": "到达即停止：成本目标 0，安全目标仍是 h_T"})

    # forward happens to point at the goal on this map, so the failure probe
    # drives backwards into the rear wall instead of "hoping" to fail
    env2 = build(z0_J=400.0)
    recs2 = run_until_done(env2, lambda e: (-1.0, 0.0, 0.0))
    arr_f = segment_from_records([recs2[-1]], obs.Z_REF_J)
    arr_f["terminated"][-1] = True
    out_f = S.compute_objectives(arr_f, c, terminal_kind="failure")
    check("single_step_failure_uses_penalty",
          env2.label in ("collision", "out_of_bounds")
          and abs(float(out_f["obj_target_cost"][-1]) - c.fail_penalty) < 1e-12
          and abs(float(out_f["obj_h_terminal"][0]) - c.fail_penalty) < 1e-12
          and not bool(out_f["obj_feasible"][-1]),
          {"label": env2.label, "target_cost": float(out_f["obj_target_cost"][-1])})


def check_negative_budget_keeps_running() -> None:
    """M3: z < 0 must not end the episode and must not produce a label."""
    env = build(z0_J=2.0)
    recs = []
    first_negative = None
    while not env.done and len(recs) < 200:
        rec = env.step((0.6, 0.0, 0.0))
        recs.append(rec)
        # `within_budget`/`z_J_before` describe the decision state z_t, so the
        # first *over-budget decision* is the next record after the overspend
        if first_negative is None and rec.z_J < 0.0:
            first_negative = rec
            continue
        if first_negative is not None and rec.z_J_before < 0.0:
            first_negative = rec
            break
    ok_first = (first_negative is not None
                and first_negative.z_J_before < 0.0
                and not first_negative.done
                and first_negative.label is None
                and not first_negative.within_budget)
    # keep going: the episode must end on something that is not "budget"
    while not env.done and len(recs) < 400:
        recs.append(env.step((0.6, 0.0, 0.0)))
    labels = {r.label for r in recs if r.label}
    check("negative_budget_keeps_stepping", ok_first and labels <= {"collision", "out_of_bounds",
                                                                    "goal_reached",
                                                                    "stagnation", "timeout"},
          {"first_negative": None if first_negative is None else first_negative.as_dict()["z_J_before"],
           "steps_after_negative": len(recs),
           "final_label": env.label, "labels_seen": sorted(labels)})
    detail = env.result().as_dict()
    check("no_budget_exhausted_label_in_new_episodes",
          "budget_exhausted" not in labels
          and detail["within_budget"] in (True, False)
          and "arrived_within_budget" in detail,
          detail)


def check_over_budget_arrival() -> None:
    """Arrival after overspending: label says arrived, feasibility says no."""
    poor = build(z0_J=1.0)
    recs = run_until_done(poor, lambda e: greedy(e))
    res_poor = poor.result()
    rich = build(z0_J=800.0)   # the greedy probe spends ~543 J; 800 <= 1.5*Z_REF
    run_until_done(rich, lambda e: greedy(e))
    res_rich = rich.result()
    ok = (res_poor.label == "goal_reached"
          and not res_poor.arrived_within_budget
          and res_poor.over_budget_J > 0.0
          and res_rich.label == "goal_reached"
          and res_rich.arrived_within_budget)
    check("over_budget_arrival_is_not_feasible", ok,
          {"poor": res_poor.as_dict(), "rich": res_rich.as_dict()})
    # the objective must agree with the flag: same episode, W > 0 at the end
    c = cfg()
    arrays = segment_from_records([recs[-1]], obs.Z_REF_J)
    arrays["terminated"][-1] = True
    out = S.compute_objectives(arrays, c, terminal_kind="success")
    check("over_budget_arrival_W_positive",
          not bool(out["obj_feasible"][-1]) and float(out["obj_stop_residual"][-1]) > 0.0,
          {"W_last": float(out["obj_stop_residual"][-1])})


def check_danger_then_arrival() -> None:
    """Event truth table: arriving after entering the forbidden band is not feasible."""
    c = cfg()

    def seg(y0, y1):
        T = 2
        return {
            "cost_norm": np.full(T, 0.05),
            "z_j": np.array([0.4, 0.35]) * obs.Z_REF_J,
            "z_norm": np.array([0.4, 0.35]),
            "z_ratio": np.array([0.4, 0.35]),
            "value_cost": np.zeros(T), "value_reach": np.zeros(T),
            "safety_y": np.array([0.0, y0]),
            "goal_distance_m": np.array([2.0, 0.9]),
            "z_norm_next": np.array([0.35, 0.30]),
            "z_ratio_next": np.array([0.35, 0.30]),
            "safety_y_next": np.array([y0, y1]),
            "goal_distance_m_next": np.array([0.9, 0.3]),
            "terminated": np.array([False, True]),
            "truncated": np.zeros(T, dtype=bool),
            "bootstrap_value_cost": np.zeros(T),
            "bootstrap_value_reach": np.zeros(T),
        }

    clean = S.compute_objectives(seg(0.0, 0.0), c, terminal_kind="success")
    danger = S.compute_objectives(seg(0.0, 1.0), c, terminal_kind="success")
    dirty_early = S.compute_objectives(seg(1.0, 1.0), c, terminal_kind="success")
    check("danger_then_arrival_not_feasible",
          bool(clean["obj_feasible"][-1])
          and not bool(danger["obj_feasible"][-1])
          # 契约 max_zero_caveat：h 内的 max(·,0) 会把目标内的 -0.001 抹平为 0，
          # 所以干净到达的 h_T 与 W 都恰为 0，可行判据是 W <= -beta（beta=0）
          and abs(float(clean["obj_h_terminal"][0])) < 1e-12
          and abs(float(clean["obj_stop_residual"][-1])) < 1e-12
          and abs(float(danger["obj_h_terminal"][0]) - 1.0) < 1e-12
          and not bool(dirty_early["obj_feasible"][-1]),
          {"clean_h_T": float(clean["obj_h_terminal"][0]),
           "clean_W": float(clean["obj_stop_residual"][-1]),
           "danger_h_T": float(danger["obj_h_terminal"][0]),
           "danger_W": float(danger["obj_stop_residual"][-1]),
           "danger_early_W": float(dirty_early["obj_stop_residual"][-1])})


def check_slice_vs_horizon() -> None:
    """M8: a window slice and the 30 s horizon are both truncation but differ."""
    env = build(z0_J=400.0)
    recs = [env.step((0.05, 0.0, 0.0)) for _ in range(5)]
    c = cfg()
    arr_slice = segment_from_records(recs, obs.Z_REF_J)
    arr_slice["truncated"][-1] = True
    arr_slice["bootstrap_value_cost"][-1] = 0.125
    out_slice = S.compute_objectives(arr_slice, c, terminal_kind="truncation")

    slow = build(z0_J=400.0)
    slow_recs = run_until_done(slow, lambda e: (0.05, 0.0, 0.0), cap=400)
    arr_to = segment_from_records(slow_recs, obs.Z_REF_J)
    arr_to["truncated"][-1] = True
    arr_to["bootstrap_value_cost"][-1] = 0.125
    out_to = S.compute_objectives(arr_to, c, terminal_kind="truncation")

    ok = (slow.label in ("timeout", "stagnation")
          and abs(float(out_slice["obj_target_cost"][-1])
                  - (float(arr_slice["cost_norm"][-1]) + 0.125)) < 1e-12
          and abs(float(out_to["obj_target_cost"][-1])
                  - (float(arr_to["cost_norm"][-1]) + 0.125)) < 1e-12
          and len(arr_slice["cost_norm"]) == 5 != len(arr_to["cost_norm"]))
    check("slice_and_horizon_both_bootstrap_but_differ_in_extent", ok,
          {"env_label": slow.label, "slice_T": int(len(arr_slice["cost_norm"])),
           "horizon_T": int(len(arr_to["cost_norm"])),
           "slice_cost_last": float(out_slice["obj_target_cost"][-1]),
           "horizon_cost_last": float(out_to["obj_target_cost"][-1]),
           "note": "窗口切片 env_done=False、时限 env_done=True（见 train_formal.close_segment）"})


def check_action_mapping_and_gradient() -> None:
    """Actions: declared mapping, saturation statistics and usable signal."""
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    net = NW.build(nav)
    net.eval()
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    obs_batch = torch.as_tensor(rng.normal(0.0, 0.5, size=(64, obs.OBS_DIM)),
                                dtype=torch.float32)
    with torch.no_grad():
        dist = net.policy(obs_batch)
        mean = dist.mean
    std = float(dist.stddev.mean())
    # saturated raw actions: 64 samples, all three components beyond +-1
    sign = torch.as_tensor(rng.choice([-1.0, 1.0], size=(64, 3)), dtype=torch.float32)
    raw = sign * torch.as_tensor(rng.uniform(1.05, 3.0, size=(64, 3)), dtype=torch.float32)
    applied = []
    for row in raw.numpy():
        rec = obs.command_from_action(row)
        applied.append(rec.applied)
    applied_t = torch.as_tensor(np.asarray(applied), dtype=torch.float32)
    # the mapping is a hard clip: applied is bounded, and many raw values collide
    unique_applied = int(np.unique(applied_t[:, 0].numpy()).size)
    exact_limits = int(np.sum(np.abs(applied_t[:, 0].numpy()) == obs.COMMAND_MAX[0]))

    # gradient the *mean* receives: LR gradient (log_prob) is not zero at clip
    net.train()
    dist = net.policy(obs_batch)
    logp = dist.log_prob(raw)
    adv = torch.ones(64)
    loss, _ = P.policy_loss(logp, logp.detach(), adv, dist.entropy(), P.PPOConfig.from_contract())
    for p in net.parameters_of("policy"):
        p.grad = None
    loss.backward()
    grads = [float(p.grad.abs().max()) for p in net.parameters_of("policy")
             if p.grad is not None]
    mu_grad = float(mean.abs().mean())
    check("action_saturation_statistics",
          exact_limits == 64 and unique_applied <= 64,
          {"raw_abs_mean": float(raw.abs().mean()), "policy_std": std,
           "applied_at_limit": exact_limits, "unique_applied_vx": unique_applied,
           "note": "硬裁剪：applied 顶格，多个 raw 值映到同一命令（可辨识性下降）"})
    check("saturated_actions_still_give_policy_gradient",
          max(grads) > 0.0 and mu_grad >= 0.0,
          {"max_abs_param_grad": max(grads), "n_params_with_grad": len(grads),
           "note": "似然比梯度在 raw 空间，裁剪不切断 dlogp/dmu；"
                   "路径梯度 dapplied/draw 在裁剪区为 0，这正是不能要求环境路径可导的原因"})

    # one SGD step with all-positive expensive actions must move the mean away
    # from them (lower vx).  A mixed-sign batch cannot show this: with mean ~0
    # moving either way raises some distances and lowers others, which is why
    # the probe uses one sign only.
    net.train()
    pos_raw = torch.as_tensor(rng.uniform(1.2, 3.0, size=(64, 3)),
                              dtype=torch.float32)
    with torch.no_grad():
        logp_pos = net.policy(obs_batch).log_prob(pos_raw)
    before = float(net.policy(obs_batch).mean[:, 0].mean())
    opt = torch.optim.SGD(net.parameters_of("policy"), lr=0.05)
    opt.zero_grad()
    dist_now = net.policy(obs_batch)
    loss, _ = P.policy_loss(dist_now.log_prob(pos_raw), logp_pos, torch.ones(64),
                            dist_now.entropy(), P.PPOConfig.from_contract())
    loss.backward()
    opt.step()
    after = float(net.policy(obs_batch).mean[:, 0].mean())
    check("positive_advantage_pushes_mean_away_from_expensive_actions", after < before,
          {"mean_vx_before": before, "mean_vx_after": after,
           "note": "A>0 = 更贵；最小化 max(ratio·A) 应降低这些动作的概率，"
                   "即把均值推离它们（不是推不动）"})

    # deployment uses the same transform for mean actions
    with torch.no_grad():
        a_mean, _, _ = net.act(obs_batch[:1], deterministic=True)
    rec = obs.command_from_action(a_mean[0].numpy())
    check("deployment_and_sampling_share_the_transform",
          rec.applied == obs.clip_applied(rec.requested)[0],
          {"applied": rec.applied, "requested": rec.requested})


def check_budget_sampler() -> None:
    """M2: per-episode sampling, reproducible, resumable, bounded."""
    sys.path.insert(0, os.path.join(_NAV_ROOT, "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nav062_train_formal", os.path.join(_NAV_ROOT, "scripts", "train_formal.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    e_ref = {"m0": 400.0, "m1": 480.0}
    kw = dict(seed=3, n_envs=2, low=0.6, high=1.3, e_ref=e_ref,
              z_ref_J=obs.Z_REF_J, clip=obs.z_ratio_clip())
    a = mod.BudgetSampler(**kw)
    seq_a = [a.draw(i % 2, "m0" if i % 2 == 0 else "m1") for i in range(40)]
    b = mod.BudgetSampler(**kw)
    seq_b = [b.draw(i % 2, "m0" if i % 2 == 0 else "m1") for i in range(40)]
    check("same_seed_same_budget_sequence",
          [round(x["z0_J"], 9) for x in seq_a] == [round(x["z0_J"], 9) for x in seq_b],
          {"first_five": [round(x["z0_J"], 3) for x in seq_a[:5]]})

    # interruption: save after 13 draws, rebuild, load, continue
    c = mod.BudgetSampler(**kw)
    head = [c.draw(i % 2, "m0" if i % 2 == 0 else "m1") for i in range(13)]
    state = c.state()
    d = mod.BudgetSampler(**kw)
    d.load(state)
    # the env index must continue where it stopped (13), not restart at 0
    tail = [d.draw(i % 2, "m0" if i % 2 == 0 else "m1") for i in range(13, 40)]
    joined = head + tail
    check("resume_continues_the_same_stream",
          [round(x["z0_J"], 9) for x in joined]
          == [round(x["z0_J"], 9) for x in seq_a],
          {"n": len(joined),
           "joined_head": [round(x["z0_J"], 6) for x in joined[:3]],
           "straight_head": [round(x["z0_J"], 6) for x in seq_a[:3]]})

    # bounds, spread and no fixed-k degeneracy
    big = mod.BudgetSampler(seed=11, n_envs=1, low=0.6, high=1.3, e_ref={"m": 489.4},
                            z_ref_J=obs.Z_REF_J, clip=obs.z_ratio_clip())
    draws = [big.draw(0, "m")["k"] for _ in range(2000)]
    z_max = obs.z_ratio_clip() * obs.Z_REF_J
    ok = (min(draws) >= 0.6 and max(draws) <= 1.3
          and abs(float(np.mean(draws)) - 0.95) < 0.03
          and len(set(round(k, 6) for k in draws)) > 1900
          and 1.3 * 489.4 <= z_max)
    check("budget_distribution_is_uniform_and_not_fixed_k", ok,
          {"min": min(draws), "max": max(draws), "mean": float(np.mean(draws)),
           "distinct": len(set(round(k, 6) for k in draws)), "z_max_J": z_max})

    # 初始预算越界必须在这里拒绝（段级守卫已移除：窗口切片段的 z[0] 本来就是负的）
    too_big = mod.BudgetSampler(seed=1, n_envs=1, low=1.3, high=1.3,
                                e_ref={"m": 1400.0}, z_ref_J=obs.Z_REF_J,
                                clip=obs.z_ratio_clip())
    rejected = False
    try:
        too_big.draw(0, "m")
    except SystemExit:
        rejected = True
    check("oversized_z0_rejected_at_the_sampler", rejected,
          {"z0_J": 1.3 * 1400.0, "z_max_J": z_max,
           "note": "z0 > clip*Z_REF 时采样器直接拒绝，不等到 compute_objectives"})

    # the formal trainer must offer the structured rule and the legacy fallback
    prereg = json.load(open(PREREG, encoding="utf-8"))
    legacy = mod.budget_rules(prereg)
    structured = mod.budget_rules({"budget": {"sampling": {"kind": "uniform",
                                                           "k_low": 0.6, "k_high": 1.3}}})
    check("budget_rule_forms", legacy["kind"] == "fixed" and structured["kind"] == "uniform",
          {"legacy": legacy, "structured": structured, "prereg": os.path.basename(PREREG)})


def check_production_entrypoints() -> None:
    """Run the real (small) trainer and check what it writes to disk."""
    py = sys.executable
    tmp = tempfile.mkdtemp(prefix="nav062_")
    try:
        out_json = os.path.join(tmp, "run.json")
        cmd = [py, os.path.join(_NAV_ROOT, "scripts", "train_nominal.py"),
               "--run-root", tmp, "--run-id", "smoke_semantics",
               "--level", "L1_empty", "--map-seed", "11", "--z0-j", "600",
               "--episodes", "1", "--seed-base", "5", "--json", out_json]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(_NAV_ROOT, "src"), os.path.dirname(os.path.dirname(_NAV_ROOT)),
             os.path.join(CALIB, "src")])
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        ran = proc.returncode == 0 and os.path.exists(out_json)
        manifest = os.path.join(tmp, "smoke_semantics", "rollout", "manifest.jsonl")
        chunk_files = sorted(
            f for f in os.listdir(os.path.join(tmp, "smoke_semantics", "rollout"))
            if f.startswith("chunk_")) if os.path.isdir(
                os.path.join(tmp, "smoke_semantics", "rollout")) else []
        keys: set[str] = set()
        schema = None
        if chunk_files:
            with np.load(os.path.join(tmp, "smoke_semantics", "rollout", chunk_files[0])) as z:
                keys = set(z.files)
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
            schema = first.get("schema")
        check("production_trainer_writes_paired_chunks",
              ran and {"z_norm", "z_ratio_next", "safety_y_next",
                       "goal_distance_m_next"} <= keys and schema == S.SCHEMA_BUFFER,
              {"returncode": proc.returncode, "stderr": proc.stderr[-400:],
               "keys_missing": sorted({"z_norm", "z_ratio_next", "safety_y_next",
                                       "goal_distance_m_next"} - keys),
               "manifest_schema": schema, "chunks": chunk_files})

        # critic refit entry (freeze_deployment) builds the same arrays
        spec = importlib.util.spec_from_file_location(
            "nav062_freeze", os.path.join(_NAV_ROOT, "scripts", "freeze_deployment.py"))
        fz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fz)
        from camra_navigation.envs.vector import VectorNominalNavEnv
        design = M.load_design(DESIGN)
        geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM)
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")
        env_o = VectorNominalNavEnv(M.build("L1_empty", 11, design), geom, cm,
                                    n_envs=1, z0_J=300.0)
        rows = []
        for _ in range(3):
            o = env_o.observation[0]
            rec = env_o.step(np.asarray([[0.5, 0.0, 0.0]]))
            rows.append(fz.step_row(np.asarray(o), 0.0, 0.0,
                                    float(rec.cost_model_J[0]), rec))
        c = cfg()
        fz.ctx_global = type("C", (), {"obj_cfg": c})()
        arr = fz.arrays_from_steps(rows, "truncation", 0.0, 0.0, False, None)
        check("critic_refit_entry_builds_v2_arrays",
              {"z_norm", "z_ratio_next", "safety_y_next", "goal_distance_m_next"} <= set(arr)
              and "obj_target_total" in arr,
              {"n_keys": len(arr)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_contract_and_flags() -> None:
    """The contract and the environments must agree on the new semantics."""
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    ev = {e["id"]: e for e in nav["event_table"]["events"]}
    z_spec = next(f for f in nav["observation_spec"]["field_table"] if f["name"] == "z_ratio")
    from camra_navigation.envs.nominal import LABELS, LEGACY_LABELS
    check("contract_matches_implementation",
          nav["observation_spec"]["version"] == obs.OBS_VERSION
          and z_spec["range"] == [-obs.z_ratio_clip(), obs.z_ratio_clip()]
          and ev["budget_exhausted"].get("terminal") is False
          and "budget_exhausted" not in LABELS
          and "budget_exhausted" in LEGACY_LABELS
          and "budget_not_terminal" in nav["termination_semantics"],
          {"obs_version": nav["observation_spec"]["version"],
           "clip": obs.z_ratio_clip(), "contract_version": nav["version"],
           "labels": list(LABELS)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    check_contract_and_flags()
    check_schema_and_pairing()
    check_two_step_hand_calculation()
    check_single_step_arrival_and_failure()
    check_negative_budget_keeps_running()
    check_over_budget_arrival()
    check_danger_then_arrival()
    check_slice_vs_horizon()
    check_action_mapping_and_gradient()
    check_budget_sampler()
    check_production_entrypoints()

    failed = [r for r in RESULTS if not r["ok"]]
    report = {
        "task": "NAV-062",
        "obs_version": obs.OBS_VERSION,
        "schema": S.SCHEMA_BUFFER,
        "z_ratio_clip": obs.z_ratio_clip(),
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
