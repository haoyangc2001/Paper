#!/usr/bin/env python3
"""r14 目标函数语义 v2（深井 + 分级安全）的单元测试。

覆盖（用户 2026-11-15T17:32 授权的四项对齐改动 + 并入的两项独立问题）：
  HS1  safety_step_v2 真值表：安全态 h=g（深井保留）；违规态 h=max(g,C·y)
  HS2  v1 回归保护：违规步与目标集外安全步逐位等价，仅目标集内安全态不同
  HS3  分级安全历史：不可逆、单调不减、合法域校验
  HS4  严重度公式：边界擦碰=floor、线性上升、超参考深度封顶 1.0
  HS5  compute_objectives v2：干净到达 h_T=G_INSIDE=-3；违规到达 h_T=C·y
  HS6  min 型递推开+深井：到达回退的目标严格低于差 1 m 停下的目标（方向性）
  HS7  标量环境分级盖章：二值模式 y=1.0 不变；分级模式 y=max(步级严重度)
  HS8  停滞触发现场捕获：字段齐全；reset 后清空
  HS9  obs-v4 编码：±1650 J 恰好饱和在 ±3.0，v2/v3 仍可重建
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402
from camra_navigation.maps.geometry import safety_severity_from_depth  # noqa: E402
from camra_navigation.maps.geometry import update_safety_history_graded  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")

SEED = 20261115
RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any = "") -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("  OK   " if ok else "  FAIL ") + name + (" " + json.dumps(detail, ensure_ascii=False, default=str)[:220] if not ok else ""))


def build_cfg(h_semantics: str = "v2_deep_well", **kw) -> S.ObjectiveConfig:
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    return dataclasses.replace(cfg, joint_cost_weight=0.0, joint_min_recursion=True,
                               h_semantics=h_semantics, **kw)


def synth_episode(final_d: float, y_hist: float, terminal_kind: str = "success"):
    """4 步合成回合：goal_distance 单调下降到 final_d，y 用 y_hist。"""
    dists = np.array([1.5, 0.9, 0.45, max(final_d, 0.05)])
    return {
        "cost_norm": np.full(4, 0.01),
        "z_norm": np.linspace(0.5, 0.4, 4),
        "z_ratio": np.linspace(0.5, 0.4, 4),
        "value_cost": np.zeros(4),
        "value_reach": np.zeros(4),
        "safety_y": np.full(4, y_hist),
        "goal_distance_m": dists,
        "z_norm_next": np.linspace(0.49, 0.39, 4),
        "z_ratio_next": np.linspace(0.49, 0.39, 4),
        "safety_y_next": np.full(4, y_hist),
        "goal_distance_m_next": np.concatenate([dists[1:], [final_d]]),
        "terminated": np.array([False, False, False, terminal_kind != "truncation"]),
        "truncated": np.array([False, False, False, terminal_kind == "truncation"]),
        "bootstrap_value_cost": np.array([0.0, 0.0, 0.0, 0.7]),
    }


def build_env(level: str, *, z0_J: float, seed: int = SEED, severity: bool = False,
              spec_override=None):
    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    spec = spec_override or M.build(level, seed, design)
    return NominalNavEnv(
        spec, geom, model, z0_J=z0_J, safety_severity_enabled=severity,
        safety_severity_floor=0.2, safety_severity_ref_depth_m=0.25,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfg = build_cfg()
    C = cfg.C
    floor = 0.2

    # --- HS1: safety_step_v2 真值表 ----------------------------------------
    g = np.array([-3.0, -0.5, 0.0, 0.3, 1.2])
    y_safe = np.zeros(5)
    h2 = T.safety_step_v2(g, y_safe, C)
    check("hs1_safe_passes_g_through", np.allclose(h2, g),
          {"h": h2.tolist(), "g": g.tolist()})
    y_viol = np.array([floor, 0.5, 0.8, 1.0, 1.0])
    h2v = T.safety_step_v2(g, y_viol, C)
    check("hs1_violated_is_max", np.allclose(h2v, np.maximum(g, C * y_viol)),
          {"h": h2v.tolist()})

    # --- HS2: v1 回归保护 ----------------------------------------------------
    h1 = T.safety_step(g, y_viol, C)
    check("hs2_v1_equivalent_on_violations", np.allclose(h1, h2v),
          {"v1": h1.tolist(), "v2": h2v.tolist()})
    h1_safe = T.safety_step(g, y_safe, C)
    differs_inside = np.where(h1_safe != h2)[0].tolist()
    # v2 只在 g<0（目标集内）与 v1 不同：g=[-3,-0.5,0,0.3,1.2] → 差在 idx 0,1
    check("hs2_differs_only_inside_goal", differs_inside == [0, 1] and h2[0] == -3.0,
          {"differs_at": differs_inside, "v1": h1_safe.tolist(), "v2": h2.tolist()})

    # --- HS3: 分级安全历史 ----------------------------------------------------
    ok = (update_safety_history_graded(0.0, 0.3) == 0.3
          and update_safety_history_graded(0.3, 0.1) == 0.3
          and update_safety_history_graded(0.3, 0.9) == 0.9
          and update_safety_history_graded(0.9, 0.0) == 0.9
          and update_safety_history_graded(0.5, 1.0) == 1.0)
    check("hs3_graded_history_irreversible", ok, "max-history, only grows")
    try:
        update_safety_history_graded(0.0, 1.5)
        bad = True
    except ValueError:
        bad = False
    check("hs3_rejects_out_of_range", not bad, "severity > 1 rejected")

    # --- HS4: 严重度公式 ------------------------------------------------------
    sevs = [safety_severity_from_depth(d, floor=floor, ref_depth_m=0.25)
            for d in (0.0, 0.05, 0.125, 0.25, 1.0, -0.1)]
    ok = (abs(sevs[0] - floor) < 1e-12
          and abs(sevs[1] - 0.36) < 1e-12
          and abs(sevs[2] - 0.6) < 1e-12
          and sevs[3] == 1.0 and sevs[4] == 1.0
          and abs(sevs[5] - floor) < 1e-12)
    check("hs4_severity_formula", ok, {"severities": [round(s, 4) for s in sevs]})

    # --- HS5: compute_objectives v2 的 h_T -----------------------------------
    clean = synth_episode(0.2, 0.0)
    out_clean = S.compute_objectives(clean, cfg, terminal_kind="success")
    cfg_v1 = build_cfg("v1")
    out_clean_v1 = S.compute_objectives(clean, cfg_v1, terminal_kind="success")
    check("hs5_clean_arrival_h_T_is_well",
          abs(float(out_clean["obj_h_terminal"][0]) - cfg.G_INSIDE) < 1e-12
          and abs(float(out_clean_v1["obj_h_terminal"][0])) < 1e-12,
          {"v2": float(out_clean["obj_h_terminal"][0]),
           "v1": float(out_clean_v1["obj_h_terminal"][0])})
    viol = synth_episode(0.2, 0.35)
    out_viol = S.compute_objectives(viol, cfg, terminal_kind="success")
    check("hs5_violated_arrival_h_T_is_Cy",
          abs(float(out_viol["obj_h_terminal"][0]) - C * 0.35) < 1e-12,
          {"h_T": float(out_viol["obj_h_terminal"][0])})
    check("hs5_arrival_after_violation_infeasible",
          not bool(out_viol["obj_feasible"][-1]),
          {"W_last": float(out_viol["obj_stop_residual"][-1])})
    check("hs5_clean_arrival_feasible",
          bool(out_clean["obj_feasible"][-1]),
          {"W_last": float(out_clean["obj_stop_residual"][-1])})

    # --- HS6: 深井 + min 递推的方向性 ----------------------------------------
    # 终端联合值 = max(G_INSIDE, −z_T) = −min(3, 剩余预算)：
    # 到达（负）严格优于差 1 m 停下（正）；剩得越多值越深（省钱放大深井）。
    stop1m = synth_episode(1.2, 0.0, terminal_kind="truncation")
    out_stop = S.compute_objectives(stop1m, cfg, terminal_kind="truncation")
    tt_arrive = float(out_clean["obj_target_total"][-1])
    tt_stop = float(out_stop["obj_target_total"][-1])
    check("hs6_arrival_beats_stopping_1m_short",
          tt_arrive < tt_stop - 0.1,
          {"target_total_arrival_T": round(tt_arrive, 4),
           "target_total_stop_T": round(tt_stop, 4),
           "note": "到达=−min(3,z_T)（负），差 1 m 停下=h_T>0（正）；v1 下两者都≈0，无方向"})
    # 到达时剩余预算 2.5 → 终端 ≈ −2.5：省下的钱直接放大深井
    rich = synth_episode(0.2, 0.0)
    rich["z_norm"] = np.linspace(2.6, 2.5, 4)
    rich["z_ratio"] = rich["z_norm"]
    rich["z_norm_next"] = np.linspace(2.59, 2.49, 4)
    rich["z_ratio_next"] = rich["z_norm_next"]
    rich["bootstrap_value_cost"] = np.array([0.0, 0.0, 0.0, 2.5])
    out_rich = S.compute_objectives(rich, cfg, terminal_kind="success")
    tt_rich = float(out_rich["obj_target_total"][-1])
    check("hs6_leftover_amplifies_the_well",
          abs(tt_rich + 2.5) < 0.05 and tt_rich < tt_arrive - 1.0,
          {"target_total_arrival_rich": round(tt_rich, 4),
           "target_total_arrival_lean": round(tt_arrive, 4)})
    # 对照：v1 语义下两者几乎无差别（到达被抹平的实证）
    out_stop_v1 = S.compute_objectives(stop1m, cfg_v1, terminal_kind="truncation")
    out_arr_v1 = S.compute_objectives(clean, cfg_v1, terminal_kind="success")
    tt_stop_v1 = float(out_stop_v1["obj_target_total"][-1])
    tt_arr_v1 = float(out_arr_v1["obj_target_total"][-1])
    check("hs6_v1_has_no_direction", abs(tt_arr_v1 - tt_stop_v1) < 0.05,
          {"v1_arrival": round(tt_arr_v1, 4), "v1_stop": round(tt_stop_v1, 4)})

    # --- HS7: 环境分级盖章 ----------------------------------------------------
    design = M.load_design(DESIGN)
    e_bin = build_env("L2_single_detour", z0_J=5000.0)
    y_bin_max, bin_sevs = 0.0, []
    for _ in range(300):
        if e_bin.done:
            break
        r = e_bin.step((0.6, 0.0, 0.0))
        y_bin_max = max(y_bin_max, r.safety_y)
        if r.safety_severity > 0.0:
            bin_sevs.append(round(r.safety_severity, 4))
    # 二值模式不变式：y 只能是 0/1，且每步严重度也只能是 0/1
    check("hs7_binary_mode_stamp_is_one",
          y_bin_max in (0.0, 1.0) and all(s == 1.0 for s in bin_sevs)
          and e_bin.label == "collision",
          {"y_max": y_bin_max, "label": e_bin.label, "severities": bin_sevs[:5]})
    e_gr = build_env("L2_single_detour", z0_J=5000.0, severity=True)
    y_gr, sev_list = 0.0, []
    for _ in range(300):
        if e_gr.done:
            break
        r = e_gr.step((0.6, 0.0, 0.0))
        y_gr = max(y_gr, r.safety_y)
        if r.safety_severity > 0.0:
            sev_list.append(round(r.safety_severity, 4))
    ok = (0.0 < y_gr <= 1.0
          and abs(y_gr - e_gr.max_safety_severity) < 1e-12
          and all(floor <= s <= 1.0 for s in sev_list)
          and len(sev_list) > 0
          and e_gr.label == "collision")
    check("hs7_graded_mode_severity_tracked", ok,
          {"y_max": round(y_gr, 4), "n_viol_steps": len(sev_list),
           "severities_head": sev_list[:6], "label": e_gr.label})

    # --- HS8: 停滞触发现场 ----------------------------------------------------
    e_st = build_env("L1_empty", z0_J=10000.0)
    for _ in range(60):
        if e_st.done:
            break
        e_st.step((0.0, 0.0, 0.0))
    ctx = e_st.stagnation_context
    ok = (e_st.label == "stagnation" and ctx is not None
          and set(ctx) >= {"t_s", "pose", "goal_distance_m", "z_J",
                           "window_displacement_m", "min_displacement_m",
                           "window_s", "time_remaining_s"}
          and ctx["window_displacement_m"] < ctx["min_displacement_m"])
    check("hs8_stagnation_context_captured", ok, ctx)
    res_st = e_st.result()
    check("hs8_result_carries_context", res_st.stagnation_context == ctx
          and res_st.max_safety_severity == 0.0, "fields exposed via result()")
    e_st.reset()
    check("hs8_reset_clears_context", e_st.stagnation_context is None,
          "cleared on reset")

    # --- HS9: obs-v4 编码 -----------------------------------------------------
    ego = obs.EgoState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    extent = obs.MapExtent(-12.0, 12.0, -9.0, 9.0)
    cases = {}
    for z in (-1650.0, -1600.0, 0.0, 1600.0, 1650.0):
        vec = obs.encode_observation(ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, z, 10.0), version="obs-v4")
        cases[round(z, 1)] = round(vec[obs.field_slices()["z_ratio"].start], 6)
    ok = (obs.z_ratio_clip("obs-v4") == 3.0
          and cases[-1650.0] == -3.0 and cases[1650.0] == 3.0
          and abs(cases[-1600.0] + 1600.0 / 550.0) < 1e-6
          and "obs-v4" in obs.SUPPORTED_OBS_VERSIONS)
    check("hs9_obs_v4_encoding", ok, cases)

    # --- 汇总 ---------------------------------------------------------------
    n_failed = sum(1 for r in RESULTS if not r["ok"])
    payload = {"task": "NAV-064-r14", "test": "h_semantics_v2",
               "n_checks": len(RESULTS), "n_failed": n_failed,
               "verdict": "PASS" if n_failed == 0 else "FAIL",
               "h_semantics": "v2_deep_well", "G_INSIDE": cfg.G_INSIDE,
               "obs_version": obs.OBS_VERSION,
               "checks": RESULTS}
    if args.json:
        json.dump(payload, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: payload[k] for k in ("n_checks", "n_failed", "verdict")}))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
