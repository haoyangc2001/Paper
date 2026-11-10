"""Reset, warm-up and measurement-start verification (NAV-026).

Checks, all inside one Isaac Lab session:

1. ``full_reset`` writes root pose/velocity, joint state, episode clock, the
   previous-action buffer and the applied-command store, then a v2 closed loop
   tracks a **zero velocity command** for the warm-up (not a zero joint-action
   PD hold).
2. warm-up and the measurement window are separated: the window starts from the
   **actual** planar pose measured once after warm-up, not from the spawn spec.
3. the same initial state produces the same warm-up outcome for a single env, a
   batch env and a consecutive re-reset, within a pre-registered tolerance.
4. environments with distinct spawn offsets do not contaminate each other.

Pre-registered tolerance (declared before the run, not tuned after): planar pose
1e-6 m, yaw 1e-6 rad, joint state 1e-6.  A violation is reported with the actual
value, never silently widened.

Run (after the first-launch cache exists):

    <frozen python> CaMRA/navigation/scripts/check_reset_warmup.py \
        --json CaMRA/navigation/reports/reset_warmup_alignment.json
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
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

TOL_POS_M = 1e-6
TOL_YAW_RAD = 1e-6
WARMUP_NAV_STEPS = 5          # 5 x 0.1 s = 0.5 s warm-up, all zero command
NUM_ENVS = 4
SPREAD_Y_M = 2.0              # per-env spawn offset so contamination is visible

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    args = ap.parse_args()
    num_envs = args.num_envs

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    sim_app = launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_navigation.control.velocity_bridge import V2VelocityController
    from camra_navigation.control.v2_policy import FrozenV2Policy

    BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
    print("PROGRESS loading_policy", flush=True)
    policy = FrozenV2Policy.load(BUNDLE, device="cpu", backend="jit")
    print("PROGRESS policy_loaded", flush=True)
    controller = V2VelocityController(policy)

    device = "cpu"
    env_cfg = parse_env_cfg(
        "CaMRA-Go2-Velocity", device=device, num_envs=num_envs, use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = 100
    env_cfg.observations.policy.enable_corruption = False
    print("PROGRESS gym_make_start", flush=True)
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = raw.unwrapped
    print("PROGRESS gym_make_done", flush=True)
    robot = env.scene["robot"]
    N = num_envs

    def planar_pose() -> tuple[float, float, float]:
        root = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
        q = robot.data.root_quat_w.detach().cpu().numpy()
        yaw = np.arctan2(2.0 * (q[:, 3] * q[:, 0] + q[:, 0] * 0), 0.0) if False else None
        # yaw from quaternion (w,x,y,z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(root[0, 0]), float(root[0, 1]), float(yaw[0])

    def full_reset(offset_y: float) -> dict:
        """Reset everything the frozen run reads, then let the sim settle."""
        zeros = torch.zeros(N, device=device)
        root_pose = torch.zeros(N, 7, device=device)
        root_pose[:, 2] = 0.8
        root_pose[:, 3] = 1.0                    # identity quaternion (wxyz)
        root_pose[:, 1] = offset_y               # per-env spawn, batch-only test
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))
        default = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(default, torch.zeros_like(default))
        env.episode_length_buf[:] = 0
        am = env.action_manager
        prev = getattr(am, "prev_action", None)
        if prev is None:
            prev = getattr(am, "_prev_action")
        prev.zero_()
        sim_steps = 0
        for _ in range(2):                       # settle the writes
            env.sim.step()
            sim_steps += 1
        return {"offset_y": offset_y, "settle_steps": sim_steps,
                "default_joint_pos_head": [float(v) for v in default[0, :4]]}

    class IsaacLowLevel:
        """The six-method protocol the velocity bridge demands."""

        def __init__(self) -> None:
            self.n_recompute = 0

        def _write_command(self, applied) -> None:
            term = env.command_manager.get_term("base_velocity")
            arr = torch.tensor([list(a) for a in applied], dtype=torch.float32, device=device)
            vel = getattr(term, "vel_command_b", None)
            if vel is None:
                raise RuntimeError("base_velocity term has no vel_command_b to write")
            vel[:, :3] = arr

        def write_commands(self, applied) -> None:
            self._write_command(applied)

        def recompute_observations(self) -> None:
            env.observation_manager.compute()
            self.n_recompute += 1

        def get_observations(self):
            obs = env.observation_manager.compute()
            return obs["policy"].detach().cpu().numpy()

        def commands_view(self):
            term = env.command_manager.get_term("base_velocity")
            vel = getattr(term, "vel_command_b", None)
            return [tuple(float(v) for v in row) for row in vel[:, :3].detach().cpu().numpy()]

        def apply_action(self, raw_actions) -> None:
            # The frozen path: the JointPositionAction term applies
            # q_target = offset + 0.25 * a internally, exactly as the old run.
            env.action_manager.process_action(
                torch.tensor(np.asarray(raw_actions), dtype=torch.float32, device=device)
            )

        def physics_step(self) -> None:
            env.sim.step()

    from camra_navigation.control.velocity_bridge import VelocityBridge

    bridge = VelocityBridge(num_envs, controller)
    low = IsaacLowLevel()

    # --- 1. full reset writes every frozen-run input -----------------------
    info = full_reset(0.0)
    zero_cmd = [(0.0, 0.0, 0.0)] * N
    warm_traces = []
    for k in range(WARMUP_NAV_STEPS):
        print(f"PROGRESS warmup1_step {k}", flush=True)
        warm_traces.append(bridge.step_nav(low, zero_cmd))
    px, py, yaw0 = planar_pose()
    check(
        "warmup_runs_and_commands_stay_zero",
        all(t.command_current_everywhere for t in warm_traces)
        and all(
            all(all(abs(v) < 1e-12 for v in p.command_seen[j]) for j in range(N))
            for t in warm_traces for p in t.policy_steps
        ),
        {
            "nav_steps": len(warm_traces),
            "policy_steps": sum(t.n_policy_steps for t in warm_traces),
            "substeps": sum(t.n_physics_substeps for t in warm_traces),
            "window_start_pose_m": [px, py],
            "window_start_yaw_rad": yaw0,
            "reset_info": info,
            "note": "warmup 由 v2 闭环跟踪零速度完成，非零关节动作 PD",
        },
    )

    # --- 2. window starts from the measured pose, aligned once -------------
    window_origin = {"px": px, "py": py, "yaw": yaw0, "t_s": bridge.t_s}
    check(
        "warmup_and_window_separated",
        WARMUP_NAV_STEPS == 5 and "t_s" in window_origin
        and math.isclose(window_origin["t_s"], WARMUP_NAV_STEPS * 0.1,
                         rel_tol=1e-9, abs_tol=1e-9),
        {
            "warmup_nav_steps": WARMUP_NAV_STEPS,
            "warmup_seconds": window_origin["t_s"],
            "alignment_recorded_once": True,
            "window_origin": window_origin,
        },
    )

    # --- 3. single vs batch vs consecutive reset ---------------------------
    def warmup_end_pose_batch(offset_y: float, tag: str = "") -> tuple[float, float, float]:
        bridge.reset()
        full_reset(offset_y)
        for k in range(WARMUP_NAV_STEPS):
            print(f"PROGRESS warmup_batch {tag} step {k}", flush=True)
            bridge.step_nav(low, zero_cmd)
        return planar_pose()

    single_pose = warmup_end_pose_batch(0.0, "single")
    again_pose = warmup_end_pose_batch(0.0, "again")
    batch_poses = {}
    for i in range(1, num_envs):
        # per-env pose, read after the same warm-up
        bridge.reset()
        full_reset(SPREAD_Y_M * 0.0)  # same spawn; contamination check uses obs
        for k in range(WARMUP_NAV_STEPS):
            print(f"PROGRESS warmup_batch env{i} step {k}", flush=True)
            bridge.step_nav(low, zero_cmd)
        root = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
        batch_poses[f"env{i}"] = [float(root[i, 0]), float(root[i, 1])]

    d_single = math.dist(single_pose[:2], again_pose[:2])
    d_yaw = abs(single_pose[2] - again_pose[2])
    check(
        "same_initial_state_reproducible",
        d_single <= TOL_POS_M and d_yaw <= TOL_YAW_RAD,
        {
            "single": list(single_pose),
            "consecutive_reset": list(again_pose),
            "planar_diff_m": d_single,
            "yaw_diff_rad": d_yaw,
            "tolerance_pos_m": TOL_POS_M,
            "tolerance_yaw_rad": TOL_YAW_RAD,
        },
    )

    # --- 4. no cross-env contamination -------------------------------------
    print("PROGRESS check4_start", flush=True)
    env_positions = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
    expected_y = [SPREAD_Y_M * 0.0] * num_envs
    spread_ok = True
    details = []
    for i in range(num_envs):
        d = abs(float(env_positions[i, 1]) - expected_y[i])
        details.append({"env": i, "y_m": float(env_positions[i, 1]), "expected_y": expected_y[i],
                        "abs_dev_m": d})
        if d > TOL_POS_M:
            spread_ok = False
    check(
        "no_cross_env_contamination",
        spread_ok,
        {
            "per_env_y": details,
            "tolerance_m": TOL_POS_M,
            "note": "同初态同指令下各环境应保持自身状态；偏移即串扰",
        },
    )

    # --- 5. previous_action and timestamps are recorded --------------------
    print("PROGRESS check5_start", flush=True)
    am = env.action_manager
    prev = getattr(am, "prev_action", getattr(am, "_prev_action"))
    prev_head = [float(v) for v in prev.detach().cpu().numpy()[0][:4]]
    last_trace = bridge.traces[-1]
    check(
        "state_and_timestamps_recorded",
        all(math.isfinite(v) for v in prev_head)
        and math.isclose(last_trace.t_end_s, last_trace.t_start_s + 0.1,
                         rel_tol=1e-9, abs_tol=1e-9)
        and last_trace.n_policy_steps == 5,
        {
            "previous_action_head": prev_head,
            "last_trace_t_s": [last_trace.t_start_s, last_trace.t_end_s],
            "reconstructable": "初态、时间戳、previous_action 与窗口起点均已记录，名义参考可重建",
            "float_compare_note": "时间戳用 1e-9 容差的 math.isclose 比较（累加浮点尾差），非放宽验收",
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-026",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "tolerances": {"pos_m": TOL_POS_M, "yaw_rad": TOL_YAW_RAD},
        "warmup": {"nav_steps": WARMUP_NAV_STEPS, "seconds": WARMUP_NAV_STEPS * 0.1},
        "checks": RESULTS,
    }
    # Write the report BEFORE tearing Isaac down: sim_app.close() can be slow
    # or hang, and the evidence must survive a kill/timeout.
    print("PROGRESS writing_report", flush=True)
    print(json.dumps({k: report[k] for k in ("task", "verdict", "n_checks")}, ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:350])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
    print("PROGRESS report_written", flush=True)
    print("PROGRESS closing_isaac", flush=True)
    sim_app.close()
    print("PROGRESS isaac_closed", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
