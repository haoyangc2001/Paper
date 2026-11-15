"""Contract test for ``obs-v1`` / ``cmd-v1`` (NAV-013).

Run with the registered interpreter:

    /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python \
        CaMRA/navigation/tests/test_observation_contract.py --json <out.json>

No pytest dependency: the script is its own runner so it can be launched from a
tmux session on tiger11 with nothing but the frozen interpreter.  Exit code 0
means every check passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAV_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_NAV_ROOT))
sys.path.insert(0, os.path.join(_NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO, "CaMRA", "calibration", "src"))

from camra_navigation import frames  # noqa: E402
from camra_navigation import observation as obs  # noqa: E402

CONTRACT = os.path.join(_NAV_ROOT, "config", "navigation_contract_r14_repair_v2.json")
POWER_MODEL = os.path.join(_REPO, "CaMRA", "calibration", "config", "power_model_v1.json")

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any = None) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def approx_equal(a: list[float], b: list[float], tol: float) -> tuple[bool, float]:
    worst = 0.0
    for x, y in zip(a, b):
        worst = max(worst, abs(x - y))
    return (worst <= tol, worst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    slices = obs.field_slices()
    extent = obs.MapExtent(-12.0, 12.0, -9.0, 9.0)

    # 1. Layout -------------------------------------------------------------
    check("layout_total_dim", obs.OBS_DIM == sum(d for _n, d in obs.LAYOUT), obs.OBS_DIM)
    covered: list[int] = []
    for name, sl in slices.items():
        covered.extend(range(sl.start, sl.stop))
    check(
        "layout_contiguous_no_overlap",
        sorted(covered) == list(range(obs.OBS_DIM)),
        {"n_slices": len(slices), "covered": len(covered)},
    )
    expected_dim = 7 + 3 + obs.N_OBSTACLE_BINS + obs.N_WALL_DISTS + obs.OBSTACLE_CAPACITY + 3
    check("layout_arithmetic", obs.OBS_DIM == expected_dim, {"got": obs.OBS_DIM, "want": expected_dim})

    # 2. Constants vs frozen contract ---------------------------------------
    contract = json.load(open(CONTRACT, encoding="utf-8"))
    spec = contract["observation_spec"]
    consts = spec["constants"]
    pairs = {
        "obs_version": (obs.OBS_VERSION, contract.get("observation_version")),
        "n_obstacle_bins": (obs.N_OBSTACLE_BINS, consts["n_obstacle_bins"]),
        "obstacle_capacity": (obs.OBSTACLE_CAPACITY, consts["obstacle_capacity"]),
        "n_wall_dists": (obs.N_WALL_DISTS, consts["n_wall_dists"]),
        "r_sense_m": (obs.R_SENSE_M, consts["r_sense_m"]),
        "map_half_m": (obs.MAP_HALF_M, consts["map_half_m"]),
        "t_max_s": (obs.T_MAX_S, consts["t_max_s"]),
        "v_ref_mps": (obs.V_REF_MPS, consts["v_ref_mps"]),
        "w_ref_radps": (obs.W_REF_RADPS, consts["w_ref_radps"]),
        "v_body_clip": (obs.V_BODY_CLIP, consts["v_body_clip"]),
        "z_ref_J": (obs.Z_REF_J, consts["z_ref_J"]),
    }
    mismatched = {k: v for k, v in pairs.items() if v[0] != v[1]}
    check("constants_match_contract", not mismatched, mismatched or "all match")
    dom = contract["validated_command_domain"]
    check(
        "command_max_matches_contract",
        abs(dom["vx_mps"][1]) == obs.COMMAND_MAX[0]
        and abs(dom["vy_mps"][1]) == obs.COMMAND_MAX[1]
        and abs(dom["wz_radps"][1]) == obs.COMMAND_MAX[2],
        {"contract": [dom["vx_mps"], dom["vy_mps"], dom["wz_radps"]], "module": obs.COMMAND_MAX},
    )
    check(
        "layout_matches_contract",
        [{"name": n, "dim": d} for n, d in obs.LAYOUT] == spec["layout"],
        "field order and dims",
    )
    check("obs_dim_matches_contract", obs.OBS_DIM == spec["total_dim"], spec["total_dim"])

    # 功率模型的权重后面还要用来手算成本，这里一并取出（原出处算式已随 NAV-035 移出）
    pm = json.load(open(POWER_MODEL, encoding="utf-8"))
    w = pm["weights"]

    # z_ref_J 的出处。NAV-035 把尺度从"解析省能速度的 30 s 能量"（225.6069 J）改成
    # "场景参考能量的 1.05 倍"，所以这里不再校验某个算式，而是校验**尺度必须满足的性质**：
    #   (a) 覆盖全部场景的参考路线能量（否则 z0 <= Z_REF 时任何策略都不可行）；
    #   (b) 余量有界（否则等于把归一化常数随便放大，V_c 全被压扁、网络学不到差别）；
    #   (c) 归一化后的参考成本仍 < FAIL_PENALTY=1.0（否则"成功"比"失败惩罚"更贵）。
    # 参考能量取自 NAV-058 的 reports/budget_fairness_check.json（54 张图，与策略无关）。
    REFERENCE_ENERGY_UPPER_J = {
        "L1_empty": 363.8930116886081,
        "L2_single_detour": 489.397680362936,
        "L3_multi_path": 371.3348172252241,
    }
    fail_penalty = json.load(
        open(os.path.join(_NAV_ROOT, "config", "value_objective_contract_v1.json"),
             encoding="utf-8"))["bootstrap"]["FAIL_PENALTY"]["value"]
    max_ref = max(REFERENCE_ENERGY_UPPER_J.values())
    normalized = {k: v / obs.Z_REF_J for k, v in REFERENCE_ENERGY_UPPER_J.items()}
    check(
        "z_ref_provenance",
        1.05 * max_ref <= obs.Z_REF_J <= 1.20 * max_ref
        and all(c < fail_penalty for c in normalized.values()),
        {"z_ref_J": obs.Z_REF_J, "max_reference_energy_J": max_ref,
         "ratio_to_max": obs.Z_REF_J / max_ref,
         "normalized_reference_cost": normalized, "fail_penalty": fail_penalty,
         "previous_value": 225.6069087767049,
         "note": "旧尺度 225.6 J 只有场景最大参考能量的 0.46 倍，三条性质全不满足"},
    )

    # 3. Determinism --------------------------------------------------------
    ego = obs.EgoState(1.5, -0.75, 0.4, 0.28, -0.05, 0.12)
    obstacles = [obs.Obstacle(3.0, 1.0, 0.35), obs.Obstacle(-1.0, 2.0, 0.5)]
    task = obs.TaskState(6.0, 3.0, 0.0, 120.0, 18.0)
    a = obs.encode_observation(ego, obstacles, extent, task)
    b = obs.encode_observation(ego, obstacles, extent, task)
    check("deterministic", a == b, "identical on repeat")
    check("all_finite", all(math.isfinite(x) for x in a), obs.OBS_DIM)

    # 4. Nominal vs physical: bitwise identical on the same artificial state --
    nominal_vs_physical: list[dict[str, Any]] = []
    for yaw in (0.0, 0.3, -2.9, math.pi, -math.pi + 1e-3, 1.0):
        ego_n = obs.EgoState(2.0, -3.0, yaw, 0.31, 0.07, -0.2)
        q = frames.quat_from_yaw(yaw)
        phys = obs.encode_from_physical(
            base_pos_xy=(2.0, -3.0),
            base_quat_wxyz=q,
            lin_vel_body=(0.31, 0.07, 0.0),
            ang_vel_body=(0.0, 0.0, -0.2),
            obstacles=obstacles,
            extent=extent,
            task=task,
        )
        nom = obs.encode_observation(ego_n, obstacles, extent, task)
        eq = nom == phys
        nominal_vs_physical.append({"yaw": yaw, "bitwise_equal": bool(eq)})
        if not eq:
            _, worst = approx_equal(nom, phys, 0.0)
            nominal_vs_physical[-1]["max_abs_diff"] = worst
    check(
        "nominal_equals_physical",
        all(item["bitwise_equal"] for item in nominal_vs_physical),
        nominal_vs_physical,
    )

    # Physical path recovers yaw from a tilted (roll/pitch) attitude: the
    # projected heading must still be well defined and continuous.
    tilt = 0.15
    q_tilt = (
        math.cos(tilt / 2) * math.cos(0.4 / 2),
        math.sin(tilt / 2) * math.sin(0.4 / 2),
        math.sin(tilt / 2) * math.cos(0.4 / 2),
        math.cos(tilt / 2) * math.sin(0.4 / 2),
    )
    yaw_tilt = frames.yaw_from_quat(q_tilt)
    c, s = frames.heading_from_quat(q_tilt)
    check(
        "tilted_heading_defined",
        abs(math.hypot(c, s) - 1.0) < 1e-12 and math.isfinite(yaw_tilt),
        {"yaw": yaw_tilt},
    )
    try:
        # +90 deg about y sends body +x onto the world z axis: no planar heading.
        half = math.sqrt(0.5)
        frames.heading_from_quat((half, 0.0, half, 0.0))
        check("heading_undefined_raises", False, "no exception")
    except frames.HeadingUndefined:
        check("heading_undefined_raises", True, "internal_violation path")

    # 5. Yaw wrap continuity ------------------------------------------------
    e_pi = obs.encode_observation(
        obs.EgoState(0.0, 0.0, math.pi, 0.0, 0.0, 0.0), [], extent, task
    )
    e_minus_pi = obs.encode_observation(
        obs.EgoState(0.0, 0.0, -math.pi, 0.0, 0.0, 0.0), [], extent, task
    )
    ok, worst = approx_equal(e_pi, e_minus_pi, 1e-12)
    check("yaw_wrap_continuity", ok, {"max_abs_diff": worst})

    # 6. Obstacle counts, capacity and order invariance ---------------------
    rng = random.Random(20261110)
    pool = [obs.Obstacle(2.0 + 0.4 * i, -1.5 + 0.7 * i, 0.25 + 0.02 * i) for i in range(12)]
    count_cases: list[dict[str, Any]] = []
    for n in (0, 1, 3, 8, 12):
        sub = pool[:n]
        vec = obs.encode_observation(ego, sub, extent, task)
        mask = vec[slices["obstacle_mask"]]
        bins = vec[slices["obstacle_bins"]]
        shuffled = list(sub)
        rng.shuffle(shuffled)
        vec_shuffled = obs.encode_observation(ego, shuffled, extent, task)
        count_cases.append(
            {
                "n": n,
                "mask_sum": sum(mask),
                "expected_mask_sum": min(n, obs.OBSTACLE_CAPACITY),
                "bins_in_range": all(0.0 <= x <= 1.0 for x in bins),
                "order_invariant": vec == vec_shuffled,
            }
        )
    check(
        "obstacle_mask_counts",
        all(c["mask_sum"] == c["expected_mask_sum"] for c in count_cases),
        count_cases,
    )
    check("obstacle_bins_range", all(c["bins_in_range"] for c in count_cases), "in [0,1]")
    check(
        "obstacle_order_invariance",
        all(c["order_invariant"] for c in count_cases),
        "shuffle does not change encoding",
    )

    # Zero obstacles must produce an all-zero mask and all-zero bins.
    empty = obs.encode_observation(ego, [], extent, task)
    check(
        "empty_map_silent",
        all(x == 0.0 for x in empty[slices["obstacle_bins"]])
        and all(x == 0.0 for x in empty[slices["obstacle_mask"]]),
        "no phantom obstacles",
    )

    # 7. Negative budget and remaining time ---------------------------------
    # NAV-062：obs-v3 曾把上限从 ±1.0 放宽到 ±1.5。r14（D9 修复，用户 2026-11-15T17:32
    # 授权）obs-v4 再放宽到 ±3.0：所有场景 E_ref>Z_REF，±1.5 下成本通道对 critic 饱和。
    # v2/v3 的重建断言原样保留，v4 断言新的 ±3.0。
    lim = obs.z_ratio_clip()
    z_cases = []
    for z in (-1650.0, -50.0, 0.0, 120.0, 1650.0):
        vec = obs.encode_observation(
            ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, z, 10.0)
        )
        z_cases.append({"z_J": z, "z_ratio": vec[slices["z_ratio"].start]})
    check(
        "z_ratio_signed_and_bounded",
        lim == 3.0
        and all(-lim <= c["z_ratio"] <= lim for c in z_cases)
        and z_cases[0]["z_ratio"] == -lim
        and z_cases[3]["z_ratio"] > 0.0
        and z_cases[4]["z_ratio"] == lim,
        {"clip": lim, "cases": z_cases},
    )
    v2_cases = []
    for z in (-1650.0, -50.0, 0.0, 120.0, 1650.0):
        vec = obs.encode_observation(
            ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, z, 10.0), version="obs-v2"
        )
        v2_cases.append({"z_J": z, "z_ratio": vec[slices["z_ratio"].start]})
    v3_vec = obs.encode_observation(ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, 700.0, 10.0))
    v2_vec = obs.encode_observation(
        ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, 700.0, 10.0), version="obs-v2")
    check(
        "obs_v2_still_reconstructible",
        all(-1.0 <= c["z_ratio"] <= 1.0 for c in v2_cases)
        and v2_cases[0]["z_ratio"] == -1.0
        and v2_cases[4]["z_ratio"] == 1.0
        # 700 J: v2 饱和在 1.0，v3 仍然看得见富余（1.2727）
        and v2_vec[slices["z_ratio"].start] == 1.0
        and abs(v3_vec[slices["z_ratio"].start] - 700.0 / obs.Z_REF_J) < 1e-12,
        {"v2_cases": v2_cases, "v2_at_700": v2_vec[slices["z_ratio"].start],
         "v3_at_700": v3_vec[slices["z_ratio"].start]},
    )
    t_cases = []
    for t in (-5.0, 0.0, 15.0, 30.0, 45.0):
        vec = obs.encode_observation(
            ego, [], extent, obs.TaskState(1.0, 1.0, 0.0, 100.0, t)
        )
        t_cases.append({"t_s": t, "t_n": vec[slices["time_remaining_n"].start]})
    check(
        "time_remaining_bounded_monotone",
        all(0.0 <= c["t_n"] <= 1.0 for c in t_cases)
        and [c["t_n"] for c in t_cases] == sorted(c["t_n"] for c in t_cases),
        t_cases,
    )
    safe = obs.encode_observation(ego, [], extent, obs.TaskState(1.0, 1.0, 1.0, 100.0, 10.0))
    check("safety_y_binary", safe[slices["safety_y"].start] == 1.0, "y in {0,1}")

    # 8. Action pipeline ----------------------------------------------------
    rec = obs.command_from_action((1.5, -2.0, 0.5))
    check(
        "action_clipped_to_domain",
        rec.requested == (1.0, -0.4, 0.5) and rec.applied == rec.requested and rec.clipped,
        {"requested": rec.requested, "applied": rec.applied, "clipped": rec.clipped},
    )
    rec_finite = obs.command_from_action((0.5, -0.25, 0.1))
    check(
        "action_within_domain_not_clipped",
        not rec_finite.clipped
        and rec_finite.applied == (0.5, -0.1, 0.1),
        {"applied": rec_finite.applied, "clipped": rec_finite.clipped},
    )
    rec_nan = obs.command_from_action((float("nan"), 0.2, float("inf")))
    check(
        "nonfinite_request_zeroed",
        rec_nan.applied == (0.0, 0.2 * 0.4, 0.0) and rec_nan.invalid and rec_nan.clipped,
        {"applied": rec_nan.applied, "invalid": rec_nan.invalid},
    )
    rec_reach = obs.command_from_action((1.0, 1.0, 1.0))
    check(
        "domain_fully_reachable",
        rec_reach.applied == obs.COMMAND_MAX and not rec_reach.clipped,
        {"applied": rec_reach.applied},
    )

    # 9. Single applied command drives both motion and cost ------------------
    from camra_calibration.nominal import PlanarState, step_exact  # noqa: E402

    req = (2.0, 0.0, 0.0)  # out of domain, as a buggy caller might send
    fixed = obs.clip_applied(req)
    applied = fixed[0]
    p0 = PlanarState(0.0, 0.0, 0.0)
    p_applied = step_exact(p0, applied, 0.1)
    p_requested = step_exact(p0, req, 0.1)
    cost_applied = w[0] + w[1] * abs(applied[0]) + w[4] * applied[0] ** 2
    cost_requested = w[0] + w[1] * abs(req[0]) + w[4] * req[0] ** 2
    check(
        "single_applied_command_matters",
        abs(p_applied.px - p_requested.px) > 0.05 and abs(cost_applied - cost_requested) > 1.0,
        {
            "applied": applied,
            "dx_applied_m": p_applied.px,
            "dx_requested_m": p_requested.px,
            "P_applied_W": cost_applied,
            "P_requested_W": cost_requested,
        },
    )

    # 10. Field ranges ------------------------------------------------------
    ranges = {
        "sin_yaw": (-1.0, 1.0),
        "cos_yaw": (-1.0, 1.0),
        "vx_body_n": (-obs.V_BODY_CLIP, obs.V_BODY_CLIP),
        "vy_body_n": (-obs.V_BODY_CLIP, obs.V_BODY_CLIP),
        "wz_body_n": (-obs.V_BODY_CLIP, obs.V_BODY_CLIP),
        "px_n": (-1.0, 1.0),
        "py_n": (-1.0, 1.0),
        "goal_dir_x_body": (-1.0, 1.0),
        "goal_dir_y_body": (-1.0, 1.0),
        "goal_dist_n": (0.0, 1.5),
        "wall_dist_n": (0.0, 1.0),
        "safety_y": (0.0, 1.0),
        "z_ratio": (-1.0, 1.0),
        "time_remaining_n": (0.0, 1.0),
    }
    worst_field = None
    for name, (lo, hi) in ranges.items():
        sl = slices[name]
        for x in a[sl]:
            if not (lo - 1e-12 <= x <= hi + 1e-12):
                worst_field = {"field": name, "value": x, "range": [lo, hi]}
                break
        if worst_field:
            break
    check("all_fields_in_declared_range", worst_field is None, worst_field or "ok")

    # Body origin is base_link, not the footprint centroid.
    centroid = (-0.04563529179509698, -0.010971589494882339)
    shifted = obs.encode_observation(
        obs.EgoState(ego.px + centroid[0], ego.py + centroid[1], ego.yaw, 0, 0, 0),
        [],
        extent,
        task,
    )
    check(
        "position_is_base_link_not_centroid",
        shifted[slices["px_n"].start] != a[slices["px_n"].start],
        "centroid offset would change px_n",
    )

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "obs_version": obs.OBS_VERSION,
        "command_version": obs.COMMAND_VERSION,
        "obs_dim": obs.OBS_DIM,
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
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
