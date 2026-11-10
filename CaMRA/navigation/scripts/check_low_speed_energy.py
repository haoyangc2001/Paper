"""NAV-059: analytic low-speed optimum vs actual v2 executability (frozen chain).

Pre-registered (experiment_prereg_v1.json / diagnostics/nav_059_low_speed):

    speeds_mps      [0.05, 0.1, 0.15, 0.2, 0.3]   (straight line, body-frame vx)
    levels          3  (integration_pilot maps: L1_empty / L2_single_detour /
                        L3_multi_path, geometry frozen at seed 101)
    seeds           3 per (level, speed), perturbing start/goal/start_yaw only
                    -> declared here as [101, 202, 303] (first run; frozen)
    episodes        45 straight-line + 6 diagnostic combos (side/turn), capped
    tolerance       tracking error <= 0.05 m (lateral, vs the straight-line
                    reference) and <= 0.05 rad/s (yaw rate), frozen here
    usability       per speed: >= 90% of episodes with no fall AND tracking ok
                    -> the nominal energy-saving speed is executable; a miss is
                    recorded as a known v2 low-speed limitation (no retraining,
                    no disabling low speeds)
    wall clock      <= 2 h total

Analytic part (no Isaac): the frozen power model (power-v1, quadratic, K=0)
    P(v) = w_bias + w_vx * v + w_vx2 * v^2          [W, straight line]
    J/m  = P(v)/v  minimized at v* = sqrt(w_bias / w_vx2)  (~0.30987 m/s)
Recomputed here from the exact frozen coefficients; the pre-registered speed
list has no 0.3099 entry, so the optimum is reported as a *prediction row*
and contrasted against the measured 0.3 tier (no list extension without user
approval).

Measured part (Isaac, frozen explicit-PD chain from NAV-027): constant body
command (v, 0, 0) for 30 s per episode on the frozen map start pose; energy
from NavEnergyRecorder (200 Hz); a J/m value is computable ONLY when the
longitudinal progress exceeds 0.01 m (otherwise null, never guessed).

Run (after the NAV-027 cache exists):
    <frozen python> CaMRA/navigation/scripts/check_low_speed_energy.py \
        --json CaMRA/navigation/reports/low_speed_energy_executability.json \
        [--pilot]   # one speed tier, one episode, wall-clock probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

# ---- pre-registered constants (frozen before the first run) ---------------
SPEEDS_MPS = [0.05, 0.1, 0.15, 0.2, 0.3]
LEVELS = ["L1_empty", "L2_single_detour", "L3_multi_path"]
MAP_SEED = 101                 # integration_pilot geometry seed (frozen)
EPISODE_SEEDS = [101, 202, 303]  # start/goal/start_yaw perturbation (declared)
EPISODE_SECONDS = 30.0
SUBSTEP_DT = 0.005
WARMUP_SECONDS = 1.0   # 校准协议 v1.0.2 warmup_rule：1.0 s 全零命令
TOL_LATERAL_M = 0.05
TOL_YAWRATE_RAD = 0.05
USABLE_RATIO = 0.90
PROGRESS_MIN_M = 0.01          # below this, J/m is not computable -> null
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0
# diagnostic combos (side/turn), outside the main usability denominator
DIAG_COMBOS = [(0.3, 0.1, 0.2), (0.1, 0.05, 0.1)]
DIAG_LEVEL = "L1_empty"
DIAG_SEEDS = [101, 202, 303]
WALL_CLOCK_CAP_H = 2.0
PREREG_SPEEDS_JSON = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"


def progress(msg: str) -> None:
    print(f"PROGRESS {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Phase A: analytic prediction from the frozen power model
# --------------------------------------------------------------------------- #
def load_power_model() -> dict:
    cfg_path = os.path.join(CALIB, "config", "power_model_v1.json")
    sha_path = cfg_path + ".sha256"
    raw = open(cfg_path, "rb").read()
    d = json.loads(raw)
    want = open(sha_path).read().strip().split()[0]
    declared = d.get("sha256", "")
    # 权威校验：PowerModel 官方 sha256()（= 冻结声明口径），字节哈希仅留档审计
    from camra_calibration.power import PowerModel
    pm = PowerModel.load(cfg_path)
    official = pm.sha256()
    ok = (want == declared == official)
    return {"model": d, "sha_file": want, "sha_bytes": hashlib.sha256(raw).hexdigest(),
            "sha_declared": declared, "sha_official": official, "sha_ok": bool(ok),
            "power_model": pm}


def analytic_prediction(model: dict, speeds: list[float], power_model=None) -> dict:
    """P(v)/v from the exact frozen coefficients (straight line, vy=wz=0)."""
    names = model["feature_names"]
    w = dict(zip(names, model["weights"]))
    w_bias, w_vx, w_vx2 = w["bias"], w["abs_vx"], w["vx2"]
    if w_vx2 <= 0 or w_bias <= 0:
        raise RuntimeError("frozen model lacks the bias/vx2 terms; cannot form v*")
    v_star = math.sqrt(w_bias / w_vx2)
    rows = []
    for v in speeds:
        p = w_bias + w_vx * v + w_vx2 * v * v          # W
        # 独立对照：冻结 PowerModel.step_power（同一权重，另一实现路径）
        p_ref = None
        if power_model is not None:
            try:
                import numpy as np
                p_ref = float(power_model.step_power(
                    np.array([[v, 0.0, 0.0], [v, 0.0, 0.0]]))[0])
            except Exception:
                p_ref = None
        rows.append({"speed_mps": v, "power_w": p,
                     "power_w_powermodel": p_ref,
                     "j_per_m": p / v, "j_30s": p * EPISODE_SECONDS})
    p_star = w_bias + w_vx * v_star + w_vx2 * v_star * v_star
    return {
        "v_star_mps": v_star,
        "j_per_m_at_v_star": p_star / v_star,
        "coefficients": {"bias_w": w_bias, "abs_vx_w_per_mps": w_vx,
                         "vx2_w_per_mps2": w_vx2},
        "per_speed": rows,
        "note": "解析最优只针对冻结模型匀速直线 P(v)/v；不含有限时限、转弯/启停，"
                "不能称实测最优。0.30987 不在预登记速度清单内，仅作预测对照行。",
    }


# --------------------------------------------------------------------------- #
# Phase B: frozen-chain execution
# --------------------------------------------------------------------------- #
def run_isaac(args, analytic: dict, design: dict) -> dict:
    """Phase B: frozen v2 via the COMPLETE ManagerBasedRLEnv pipeline.

    NAV-059 diagnosis (2026-11-10, 11 Isaac rounds): v2 is a Lab-trained
    policy; its frozen semantics are the FULL env.step() pipeline (implicit
    PD 25/0.5 from the actuator cfg, JointPositionAction, Lab joint order,
    manager-computed observations).  A control experiment with raw.step()
    walks stably (tilt 2.6 deg, 0.3 m/s, 5 s) while every hand-driven
    variant (explicit PD 20/25, per-substep writes) collapsed.

    Energy recording stays at 200 Hz by patching sim.step (the v2
    TrialRecorder pattern): read applied_torque -> robot.update(dt) ->
    read joint_vel -> jointwise-clamped float64 accumulation.
    """
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    sim_app = launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute
    from camra_navigation.maps.generator import build as build_map

    def progress(msg: str) -> None:
        print(f"PROGRESS {msg}", flush=True)

    BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
    policy = torch.jit.load(os.path.join(BUNDLE, "exported", "policy.pt")).eval()

    device = args.device
    env_cfg = parse_env_cfg(
        "CaMRA-Go2-Velocity", device=device, num_envs=args.batch, use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = 100
    env_cfg.observations.policy.enable_corruption = False
    # 恒速核验：命令由外部固定注入，禁用命令重采样（resample 会覆写 vel_command_b）
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    # episode 时长必须覆盖 warmup+窗口：默认 20 s 到期会触发 ManagerBasedRLEnv
    # 内部自动 reset（实测 ctrl1000 位置跳回 origin，能量/轨迹被污染）
    env_cfg.episode_length_s = WARMUP_SECONDS + EPISODE_SECONDS + 0.2
    progress("gym_make_start")
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = raw.unwrapped
    robot = env.scene["robot"]
    sim = env.sim
    sim_dt = sim.get_physics_dt()
    N = args.batch
    raw.reset()
    progress("gym_make_done")

    env_origins_np = env.scene.env_origins.detach().cpu().numpy()  # (N, 3)

    substeps_total = int(round(EPISODE_SECONDS / SUBSTEP_DT))       # 6000
    ctrl_steps_total = substeps_total // 4                          # 1500 policy steps
    warmup_ctrl = int(round(WARMUP_SECONDS / 0.02))                 # 50

    # ---- episode plan (pre-registered) ------------------------------------
    maps = {lv: build_map(lv, MAP_SEED, design) for lv in LEVELS}
    plan = []
    for v in SPEEDS_MPS:
        for lv in LEVELS:
            for s in EPISODE_SEEDS:
                plan.append({"speed": v, "level": lv, "seed": s, "cmd": (v, 0.0, 0.0)})
    for c in DIAG_COMBOS:
        for s in DIAG_SEEDS:
            plan.append({"speed": None, "level": DIAG_LEVEL, "seed": s,
                         "cmd": c, "diag": True})
    progress(f"episodes_planned={len(plan)}")

    def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))

    def run_batch(batch_plan: list[dict]) -> tuple[list[dict], dict]:
        B = len(batch_plan)
        assert B <= N
        recorder = NavEnergyRecorder(
            capacity_steps=substeps_total + warmup_ctrl * 4 + 40,
            num_envs=N, device=device, label="nav059",
            expected_steps=substeps_total)

        # ---- patched sim.step: 200 Hz recording (v2 TrialRecorder order) --
        state = {"rec": None}
        orig_step = sim.step

        def patched_step(*a, **kw):
            out = orig_step(*a, **kw)
            rec = state["rec"]
            if rec is not None:
                tau = robot.data.applied_torque.detach()
                robot.update(sim_dt)   # joint_vel 时间戳门控（v2 TrialRecorder 同款）
                rec.record(
                    tau_applied_lab=tau,
                    qdot_post_lab=robot.data.joint_vel.detach(),
                    root_pos_lab=robot.data.root_pos_w[:, :3].detach(),
                )
            return out

        sim.step = patched_step
        state["rec"] = recorder
        recorder.begin_warmup()

        # ---- reset every env to its own episode start ----------------------
        obs, _ = raw.reset()
        for e, ep in enumerate(batch_plan):
            spec = maps[ep["level"]]
            rng = torch.Generator().manual_seed(ep["seed"] + 5000)
            yaw_jit = float(torch.empty(1).uniform_(-0.15, 0.15, generator=rng))
            yaw = float(spec.start_yaw) + yaw_jit
            ox, oy = float(env_origins_np[e, 0]), float(env_origins_np[e, 1])
            ep["_yaw0"] = yaw
            ep["_origin"] = (ox, oy)
            ep["_start"] = (float(spec.start[0]), float(spec.start[1]))
            qw, qx, qy, qz = quat_wxyz_from_yaw(yaw)
            root = torch.zeros(1, 7, device=device)
            root[0, 0] = ox + spec.start[0]
            root[0, 1] = oy + spec.start[1]
            root[0, 2] = 0.4                      # v2 契约 init_state z
            root[0, 3], root[0, 4], root[0, 5], root[0, 6] = qw, qx, qy, qz
            robot.write_root_pose_to_sim(root, env_ids=torch.tensor([e], device=device))
            robot.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=device), env_ids=torch.tensor([e], device=device))
            djp = robot.data.default_joint_pos.detach()[e:e + 1]
            robot.write_joint_state_to_sim(
                djp, torch.zeros_like(djp), env_ids=torch.tensor([e], device=device))
        # 未分配 env（pilot）保持 raw.reset 的默认状态（各自 subplot，互不影响）
        robot.update(sim_dt)
        am = env.action_manager
        prev = getattr(am, "prev_action", getattr(am, "_prev_action"))
        prev.zero_()
        robot.update(sim_dt)

        # ---- warmup：v2 闭环跟踪零速度（命令通道 0）1.0 s ------------------
        low_cmd = torch.zeros(N, 3, device=device)

        def write_cmd(rows):
            term = env.command_manager.get_term("base_velocity")
            arr = torch.tensor([list(r) for r in rows], dtype=torch.float32, device=device)
            term.vel_command_b[:, :3] = arr

        write_cmd([[0.0, 0.0, 0.0]] * N)
        for _ in range(warmup_ctrl):
            act = policy(obs["policy"])
            obs, _, _, _, _ = raw.step(act)
        warmup_calls = recorder.n_record_calls
        recorder.begin_window()

        # ---- 30 s measurement window ----------------------------------------
        rows = [list(ep["cmd"]) for ep in batch_plan]
        while len(rows) < N:
            rows.append([0.0, 0.0, 0.0])
        write_cmd(rows)
        yawrate_max = [0.0] * B
        fell = [False] * B
        fall_step = [None] * B
        for i in range(ctrl_steps_total):
            act = policy(obs["policy"])
            obs, _, te, tr, _ = raw.step(act)
            q = robot.data.root_quat_w.detach().cpu().numpy()
            gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            tilt = np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0)))
            h = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
            wz_b = robot.data.root_ang_vel_w.detach().cpu().numpy()[:, 2]
            for e, ep in enumerate(batch_plan):
                yawrate_max[e] = max(yawrate_max[e],
                                     abs(float(wz_b[e]) - ep["cmd"][2]))
                if (not fell[e]) and (h[e] < FALL_HEIGHT_M or tilt[e] > FALL_TILT_DEG):
                    fell[e] = True
                    fall_step[e] = i
            if args.debug and i % 100 == 0:
                p0 = robot.data.root_pos_w[0].detach().cpu().numpy()
                print(f"DBG ctrl{i:04d} h={h[0]:.3f} tilt={tilt[0]:.1f} "
                      f"pos=({p0[0]:.3f},{p0[1]:.3f})", flush=True)
        recorder.mark_event(range(N), "timeout", detail="30 s 路线到时")
        result = recorder.finish()
        problems = recorder.problems(result)
        assert not problems, problems
        calls_in_window = recorder.n_record_calls - warmup_calls
        state["rec"] = None
        sim.step = orig_step

        # ---- per-env metrics（世界系 -> 本地系）------------------------------
        out = []
        for e, ep in enumerate(batch_plan):
            ox, oy = ep["_origin"]
            valid = result["valid_mask"][:, e]
            total_j = float(result["total_j"][e])
            pos = result["root_pos_w"][:, e, :2] - np.array([ox, oy])
            yaw0 = ep["_yaw0"]
            sx, sy = ep["_start"]
            dirx, diry = math.cos(yaw0), math.sin(yaw0)
            t_axis = np.arange(valid.shape[0], dtype=np.float64) * SUBSTEP_DT
            straight = not ep.get("diag", False)
            if straight:
                ref = np.stack([sx + dirx * ep["cmd"][0] * t_axis,
                                sy + diry * ep["cmd"][0] * t_axis], axis=1)
                diff = pos - ref
                lateral = np.abs(-dirx * diff[:, 1] + diry * diff[:, 0])
                lat_max = float(lateral.max())
                progress_m = float((pos[-1] - np.array([sx, sy])) @ np.array([dirx, diry]))
            else:
                lat_max = None
                progress_m = float(np.linalg.norm(pos[-1] - pos[0]))
            j_per_m = None
            if straight and progress_m > PROGRESS_MIN_M:
                j_per_m = total_j / progress_m
            tracking_ok = None
            if straight:
                tracking_ok = (lat_max <= TOL_LATERAL_M
                               and yawrate_max[e] <= TOL_YAWRATE_RAD)
            out.append({
                "env": e, "speed_mps": ep["speed"], "level": ep["level"],
                "seed": ep["seed"], "cmd": list(ep["cmd"]), "diag": not straight,
                "yaw0_rad": yaw0, "start": [sx, sy],
                "total_j": total_j, "substeps_recorded": int(valid.sum()),
                "progress_m": progress_m, "lateral_max_m": lat_max,
                "yawrate_err_max_rad_s": yawrate_max[e],
                "fell": fell[e], "fall_at_ctrl_step": fall_step[e],
                "stagnated": bool(straight and progress_m <= PROGRESS_MIN_M),
                "j_per_m": j_per_m, "tracking_ok": tracking_ok,
                "usable": bool(straight and (not fell[e]) and tracking_ok),
            })
        return out, {"steps": result["steps"], "calls_in_window": calls_in_window,
                     "warmup_calls": warmup_calls}

    # ---- execute batches ----------------------------------------------------
    attempts = []
    t_start = time.time()
    tiers = []
    for v in SPEEDS_MPS:
        tiers.append((f"speed_{v}", [ep for ep in plan
                                     if not ep.get("diag") and ep["speed"] == v]))
    tiers.append(("diag_combos", [ep for ep in plan if ep.get("diag")]))
    recon_note = None
    for tier_name, tier in tiers:
        if args.pilot and not tier_name.startswith("speed_0.3"):
            continue
        if args.pilot:
            tier = tier[:1]
        t0 = time.time()
        progress(f"batch_{tier_name}_n={len(tier)}")
        eps, meta = run_batch(tier)
        dt = time.time() - t0
        progress(f"batch_{tier_name}_done wallclock={dt:.1f}s")
        for ep_out in eps:
            ep_out["tier"] = tier_name
            ep_out["batch_wallclock_s"] = round(dt, 1)
        attempts.extend(eps)
        recon_note = meta
        if args.pilot:
            break
    elapsed_h = (time.time() - t_start) / 3600.0

    # ---- Phase C: aggregate + usability verdict -----------------------------
    per_speed = {}
    for v in SPEEDS_MPS:
        eps = [a for a in attempts if a["speed_mps"] == v and not a["diag"]]
        n = len(eps)
        if args.pilot and n == 0:
            continue
        usable = sum(1 for a in eps if a["usable"])
        per_speed[str(v)] = {
            "episodes": n,
            "fell": sum(1 for a in eps if a["fell"]),
            "stagnated": sum(1 for a in eps if a["stagnated"]),
            "tracking_ok": sum(1 for a in eps if a["tracking_ok"]),
            "usable": usable,
            "usable_ratio": usable / n if n else None,
            "j_per_m_measured": [a["j_per_m"] for a in eps if a["j_per_m"] is not None],
            "progress_m": [round(a["progress_m"], 4) for a in eps],
            "verdict": ("executable" if n and usable / n >= USABLE_RATIO
                        else "not_executable_under_preregistered_rule"),
        }
    diag_out = [a for a in attempts if a["diag"]]

    report = {
        "task": "NAV-059",
        "verdict": None,
        "execution_chain": "ManagerBasedRLEnv full pipeline (raw.step): implicit PD 25/0.5 "
                           "from actuator cfg, JointPositionAction, manager observations, "
                           "resampling disabled; 200 Hz energy via patched sim.step "
                           "(v2 TrialRecorder order, NavEnergyRecorder)",
        "preregistration": {
            "speeds_mps": SPEEDS_MPS, "levels": LEVELS, "map_seed": MAP_SEED,
            "episode_seeds": EPISODE_SEEDS, "episodes_total": len(plan),
            "episode_seconds": EPISODE_SECONDS,
            "tolerance": {"lateral_m": TOL_LATERAL_M, "yaw_rate_rad_s": TOL_YAWRATE_RAD,
                          "note": "首次运行前冻结"},
            "usability_criterion": f"每档速度 >= {USABLE_RATIO:.0%} 回合无跌倒且跟踪达标；"
                                   "不达标记录为 v2 低速局限，不重训、不禁用低速",
            "fall_rule": {"height_m": FALL_HEIGHT_M, "tilt_deg": FALL_TILT_DEG,
                          "source": "HiRA 口径（校准一致）"},
            "wall_clock_cap_hours": WALL_CLOCK_CAP_H,
        },
        "analytic": analytic,
        "reconciliation": recon_note,
        "wall_clock": {"hours_used": elapsed_h, "cap_hours": WALL_CLOCK_CAP_H,
                       "pilot_mode": bool(args.pilot)},
        "attempts": attempts,
        "per_speed_summary": per_speed,
        "diagnostic_combos": diag_out,
    }

    if args.pilot:
        report["verdict"] = "PILOT"
    else:
        star = analytic["v_star_mps"]
        nearest = min(SPEEDS_MPS, key=lambda v: abs(v - star))
        tier = per_speed.get(str(nearest), {})
        ok = tier.get("verdict") == "executable"
        report["verdict"] = "PASS" if ok else "FAIL"
        report["conclusion"] = {
            "nearest_tier_to_analytic_optimum": nearest,
            "analytic_optimum_mps": star,
            "nearest_tier_usable_ratio": tier.get("usable_ratio"),
            "statement": (
                f"名义省能速度 v*={star:.5f} m/s 的最近预登记档位 {nearest} m/s "
                + ("按预登记规则可执行。" if ok else
                   "未达到预登记可用性判据——记录为 v2 低速局限，需用户裁定"
                   "（禁用低速/加滤波/改时限/重训），不得以模型预测代替执行证据。"))
        }
    progress("writing_report")
    print(json.dumps({"task": report["task"], "verdict": report["verdict"]},
                     ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
    print("PROGRESS report_written", flush=True)
    print("PROGRESS hard_exit_skip_close", flush=True)
    os._exit(0)


def _projected_gravity(quat_wxyz: "np.ndarray") -> "np.ndarray":
    """body 系重力投影：R^T @ (0,0,-1)，输入 (N,4) wxyz。"""
    import numpy as np
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # R^T 的第三行乘 -1（重力向下单位向量取 -z_world 投到 body）
    # R^T @ g = -[R 第三列]^T？直接用四元数旋转公式对 g=(0,0,-1)：
    # v' = q* g q；g=(0,0,-1)。分量式（重力投影三行）：
    gx = -2.0 * (x * z + w * y)
    gy = -2.0 * (y * z - w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.stack([gx, gy, gz], axis=1)


def _body_ang_vel_z(quat_wxyz: "np.ndarray", ang_vel_w: "np.ndarray") -> "np.ndarray":
    """世界角速度 z 投到 body 系：w_body = R^T @ w_world 的 z 分量。"""
    import numpy as np
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # R^T 各行（四元数旋转矩阵转置）：
    r00, r10, r20 = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)
    r01, r11, r21 = 2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)
    r02, r12, r22 = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    wx, wy, wz_ = ang_vel_w[:, 0], ang_vel_w[:, 1], ang_vel_w[:, 2]
    return r02 * wx + r12 * wy + r22 * wz_


def _effort_limits(urdf_path: str) -> dict:
    import re
    text = open(urdf_path, "r", encoding="utf-8").read()
    limits = {}
    for m in re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S):
        e = re.search(r'effort="([0-9.]+)"', m.group(2))
        if e:
            limits[m.group(1)] = float(e.group(1))
    return limits


def _re_finditer_joint(text):
    import re
    return re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', text, re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pilot", action="store_true",
                    help="只跑 0.3 档 1 个回合，测墙钟")
    ap.add_argument("--debug", action="store_true", help="每 20 nav 步打印 env0 状态")
    ap.add_argument("--flat-start", action="store_true",
                    help="对照实验：原点 identity 起步（NAV-027 已验证条件），无地图/无 yaw 扰动")
    args = ap.parse_args()

    pm = load_power_model()
    if not pm["sha_ok"]:
        print(json.dumps({"error": "power model sha mismatch",
                          "sha_file": pm["sha_file"], "sha_bytes": pm["sha_bytes"],
                          "sha_declared": pm["sha_declared"]}, ensure_ascii=False))
        return 2
    design_path = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
    design = json.load(open(design_path))
    analytic = analytic_prediction(pm["model"], SPEEDS_MPS, power_model=pm.get("power_model"))
    print(json.dumps({"analytic_v_star": analytic["v_star_mps"],
                      "j_per_m_at_v_star": analytic["j_per_m_at_v_star"]},
                     ensure_ascii=False))
    run_isaac(args, analytic, design)
    return 0


if __name__ == "__main__":
    main()
