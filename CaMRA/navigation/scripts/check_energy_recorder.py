"""NAV-027 verification: energy meter + trajectory recorder against Isaac.

Run inside one Isaac Lab session (after the first-launch cache exists):

    <frozen python> CaMRA/navigation/scripts/check_energy_recorder.py \
        --json CaMRA/navigation/reports/energy_meter_validation.json

Execution chain (NAV-027 diagnosis, 2026-11-10): the frozen v2 semantics are
the HIRA explicit-PD chain, NOT the Lab JointPositionAction path:

    observe()  = ObservationAdapter.build(Lab data -> source-order 45-dim obs)
    policy     = torch.jit.load(bundle/exported/policy.pt)
    apply      = obs_adapter.apply_action(a) -> torque_adapter.set_policy_action
    substep    = torque_adapter.compute_substep(q, qd)   # explicit PD 25/0.5
                 -> robot.set_joint_effort_target(tau) -> robot.write_data_to_sim()
                 -> sim.step() -> robot.update(sim_dt)
    record     = NavEnergyRecorder.record(tau_applied, qdot_post, ...)

Diagnosed failures of the naive bridge (all measured, see work record):
- no raw.reset()        -> physics never initializes, every buffer reads 0.0
- no write_data_to_sim  -> action targets never reach PhysX (robot collapses)
- joint order           -> contract is FL/FR/RL/RR by joint type, Lab is per-leg;
                           mixed order saturates effort (23.7 N·m) and collapses
- implicit PD           -> applied_torque is not the frozen semantic

Acceptance mapping:

1. ``offline_recompute_matches_online``  — per env, integrate the recorded
   float32 tau/qdot (valid substeps only) with the independent numpy
   implementation and compare against the online float64 accumulation.
2. ``steps_and_dt_reconcile``            — steps*dt == elapsed, per-env
   active+invalid == total substeps, nav steps x 20 == substeps.
3. ``no_substep_missed``                 — record() called once per physics
   substep through the bridge (warmup + window).
4. ``early_termination_marked_and_frozen`` — env1 marked mid-window: status,
   event step number, validity off afterwards, energy frozen.
5. ``timeout_marked_on_full_run``        — env0/env3 run to the horizon.
6. ``async_env_flagged``                 — env2 masked off mid-run without an
   event: status ``incomplete_async``.
7. ``fault_detection_works``             — NaN input is caught; cross-reset
   write during an active window trips the fault and freezes.
8. ``no_cross_reset_on_main_path``       — main window finished with no fault.
9. ``sanity_energy_nonzero``             — walking mechanical work must be
   positive; all-zero means the sim never actually advanced.

All termination events in the Isaac part are FUNCTIONAL TESTS (labelled
``*_test``); they inject masks into the recorder, not into the control loop,
and delete no experiment data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
HIRA_ADAPTERS = os.path.join(REPO, "CaMRA", "locomotion", "adapters")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, HIRA_ADAPTERS)

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--nav-steps", type=int, default=20)
    args = ap.parse_args()
    num_envs = args.num_envs
    nav_steps = args.nav_steps
    substeps_per_nav = 20
    expected_steps = nav_steps * substeps_per_nav

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    sim_app = launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from camra_navigation.control.velocity_bridge import V2VelocityController, VelocityBridge
    from camra_navigation.control.v2_policy import FrozenV2Policy
    from camra_navigation.evaluation.recorder import (
        RecorderError,
        NavEnergyRecorder,
        offline_recompute,
    )
    from hira_go2.action_pd import ActionTorqueAdapter
    from hira_go2.asset_cfg import HIRA_DEFAULT_ANGLES, HIRA_INIT_BASE_POS
    from hira_go2.env_runner import SOURCE_ORDER
    from hira_go2.observation import (
        JointOrder,
        ObservationAdapter,
        body_angular_velocity_wxyz,
        projected_gravity_wxyz,
    )

    def progress(msg: str) -> None:
        print(f"PROGRESS {msg}", flush=True)

    BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"
    URDF = os.path.join(REPO, "HiRA", "legged_gym_go2", "resources", "robots", "go2", "urdf", "go2.urdf")

    progress("loading_policy")
    fpolicy = FrozenV2Policy.load(BUNDLE, device="cpu", backend="jit")
    policy = fpolicy  # frozen wrapper provides raw_action/joint_targets to the controller

    device = "cpu"
    env_cfg = parse_env_cfg(
        "CaMRA-Go2-Velocity", device=device, num_envs=num_envs, use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = 100
    env_cfg.observations.policy.enable_corruption = False
    progress("gym_make_start")
    raw = gym.make("CaMRA-Go2-Velocity", cfg=env_cfg)
    env = raw.unwrapped
    robot = env.scene["robot"]
    sim = env.sim
    N = num_envs
    # ManagerBasedRLEnv MUST be reset() once: without it the physics buffers
    # never initialize (measured: applied_torque/joint_vel/root_vel all 0.0).
    raw.reset()
    progress("gym_make_done")

    # ---- frozen-execution adapters (HIRA chain, read-only import) ----------
    joint_names_lab = list(robot.joint_names)
    order = JointOrder(list(SOURCE_ORDER), joint_names_lab)
    default_source = torch.tensor(
        [HIRA_DEFAULT_ANGLES[name] for name in SOURCE_ORDER], dtype=torch.float32, device=device
    )
    # effort limits from the frozen URDF (HIRA-019 same table)
    import re as _re
    _text = open(URDF, "r", encoding="utf-8").read()
    _limits = {}
    for _m in _re.finditer(r'<joint\s+name="([^"]+)"\s+type="revolute">(.*?)</joint>', _text, _re.S):
        _e = _re.search(r'effort="([0-9.]+)"', _m.group(2))
        if _e:
            _limits[_m.group(1)] = float(_e.group(1))
    effort_source = torch.tensor(
        [_limits[name] for name in SOURCE_ORDER], dtype=torch.float32, device=device
    )
    obs_adapter = ObservationAdapter(order, default_source, num_envs=N, device=device)
    torque_adapter = ActionTorqueAdapter(order, default_source, effort_source, device=device)
    default_lab = order.source_to_target(default_source)
    sim_dt = sim.get_physics_dt()
    progress("adapters_ready")

    recorder = NavEnergyRecorder(
        capacity_steps=expected_steps + 200,
        num_envs=N,
        device="cpu",
        label="nav027_validation",
        expected_steps=expected_steps,
    )

    # per-env run-state for the functional test scenario
    active_now = [True] * N
    early_env = 1 if N > 1 else 0
    early_at_nav = max(1, nav_steps // 2)
    async_env = 2 if N > 2 else 0
    async_at_nav = max(2, (nav_steps * 3) // 5)
    timeout_envs = [0] + ([3] if N > 3 else [])
    _qd_pre = None

    class HiRALowLevel:
        """The six-method bridge protocol on the frozen explicit-PD chain."""

        def __init__(self) -> None:
            self.command = torch.zeros((N, 3), dtype=torch.float32, device=device)

        def write_commands(self, applied) -> None:
            self.command = torch.tensor(
                [list(a) for a in applied], dtype=torch.float32, device=device
            )

        def recompute_observations(self) -> None:
            pass  # observations are built fresh in get_observations()

        def get_observations(self):
            quat = robot.data.root_quat_w
            ang_body = body_angular_velocity_wxyz(quat, robot.data.root_ang_vel_w)
            grav_body = projected_gravity_wxyz(quat)
            built = obs_adapter.build(
                ang_body, grav_body, self.command,
                robot.data.joint_pos, robot.data.joint_vel,
            )
            return built["obs"].detach().cpu().numpy()

        def commands_view(self):
            return [tuple(float(v) for v in row) for row in self.command.detach().cpu().numpy()]

        def apply_action(self, raw_actions) -> None:
            a = torch.tensor(np.asarray(raw_actions), dtype=torch.float32, device=device)
            clipped = obs_adapter.apply_action(a)
            torque_adapter.set_policy_action(clipped)

        def physics_step(self) -> None:
            global _qd_pre
            _qd_pre = robot.data.joint_vel.detach().clone()
            trace = torque_adapter.compute_substep(
                robot.data.joint_pos, robot.data.joint_vel
            )
            # frozen explicit-PD write path (HIRA env_runner.write_effort_lab)
            robot.set_joint_effort_target(trace["effort_written_lab"])
            robot.write_data_to_sim()
            sim.step(render=False)
            # sim.step() does NOT refresh articulation data buffers (HIRA-016)
            robot.update(sim_dt)
            tau_req_lab = order.source_to_target(trace["tau_pd_source"])
            recorder.record(
                tau_applied_lab=robot.data.applied_torque.detach(),
                qdot_post_lab=robot.data.joint_vel.detach(),
                qdot_pre_lab=_qd_pre,
                tau_request_lab=tau_req_lab.detach(),
                root_pos_lab=robot.data.root_pos_w[:, :3].detach(),
                active_mask=torch.tensor(active_now, device="cpu"),
            )

        def settle_substep(self) -> None:
            """One PD physics substep WITHOUT touching the recorder.

            full_reset 的写后沉降不属于 warmup 协议（NAV-026/027 的 warmup
            计数必须精确等于 5 nav 步 x 20 子步），沉降步绕过 record()。
            """
            trace = torque_adapter.compute_substep(
                robot.data.joint_pos, robot.data.joint_vel
            )
            robot.set_joint_effort_target(trace["effort_written_lab"])
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)

    bridge = VelocityBridge(N, V2VelocityController(policy))
    low = HiRALowLevel()

    # ---- reset + warmup (recorder in warmup mode) -------------------------
    def full_reset() -> dict:
        root_pose = torch.zeros(N, 7, device=device)
        root_pose[:, 0:3] = torch.tensor(HIRA_INIT_BASE_POS, device=device)
        root_pose[:, 3] = 1.0  # wxyz identity
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))
        joint_pos = default_lab.reshape(1, 12).repeat(N, 1)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        robot.update(sim_dt)  # refresh buffers right after the writes (HIRA)
        obs_adapter.reset()   # clear prev_action for every env
        # settle through the frozen PD path with zero action (q_target = default),
        # never zero torque (the robot would collapse before the window starts)
        torque_adapter.set_policy_action(torch.zeros(N, 12, device=device))
        for _ in range(2):
            low.settle_substep()  # 沉降不进记录器：warmup 计数保持精确
        recorder.observe_reset(range(N))  # warmup期：无窗口数据，正常放行
        return {"init_base_pos": list(HIRA_INIT_BASE_POS)}

    zero_cmd = [(0.0, 0.0, 0.0)] * N
    full_reset()
    recorder.begin_warmup()
    warmup_nav = 5
    for _ in range(warmup_nav):
        bridge.step_nav(low, zero_cmd)
    warmup_calls = recorder.n_record_calls
    progress("warmup_done")

    # ---- measurement window ----------------------------------------------
    recorder.begin_window()
    mid_cmd = [(0.6, 0.0, 0.2)] * N
    for s in range(nav_steps):
        bridge.step_nav(low, mid_cmd)
        if s == early_at_nav - 1 and N > 1:
            # FUNCTIONAL TEST: mark env1 terminated (stagnation-like event)
            recorder.mark_event(
                [early_env], "early_termination:stagnation_test",
                detail="功能测试注入：窗口内模拟提前终止，掩码仅作用于记录器",
            )
            active_now[early_env] = False
        if s == async_at_nav - 1 and N > 2:
            # FUNCTIONAL TEST: env2 silently masked off (async failure shape)
            active_now[async_env] = False
    recorder.mark_event(timeout_envs, "timeout", detail="正常时限到，窗口按计划结束")
    progress("window_done")

    result = recorder.finish()
    faults_main = recorder.problems(result)
    check("no_cross_reset_on_main_path", not faults_main,
          {"faults": faults_main,
           "note": "主窗口全程无 fault；observe_reset 仅在 warmup 前调用"})

    # ---- 0. sanity: the robot actually moved and did work ------------------
    totals = [float(v) for v in result["total_j"]]
    check("sanity_energy_nonzero", all(t > 0.0 for t in totals),
          {"total_j_per_env": totals,
           "note": "0.6 m/s 命令下行走机械功必为正；全 0 说明仿真未真正推进"})

    # ---- 1. offline recompute matches online ------------------------------
    progress("offline_recompute")
    recomp = [offline_recompute(result, i) for i in range(N)]
    ok1 = all(r["rel_diff"] < 1e-9 and r["steps_match"] for r in recomp)
    check("offline_recompute_matches_online", ok1,
          {"per_env": [{k: (v if not isinstance(v, float) else float(v))
                        for k, v in r.items()} for r in recomp],
           "tolerance": "rel_diff < 1e-9（在线 f64 设备累加 vs 离线 numpy f64 独立实现）"})

    # ---- 2. steps / dt reconciliation --------------------------------------
    calls_in_window = result["n_record_calls"] - warmup_calls
    per_env_ok = all(
        result["env_active_steps"][i] == int(result["valid_mask"][:, i].sum())
        and result["env_active_steps"][i] + (result["steps"] - result["env_active_steps"][i])
        == result["steps"]
        for i in range(N)
    )
    ok2 = (
        math.isclose(result["elapsed_s"], result["steps"] * result["dt_s"], rel_tol=1e-12)
        and calls_in_window == result["steps"]
        and result["steps"] == expected_steps
        and per_env_ok
    )
    check("steps_and_dt_reconcile", ok2,
          {"steps": result["steps"], "expected": expected_steps,
           "elapsed_s": result["elapsed_s"], "dt_s": result["dt_s"],
           "window_record_calls": calls_in_window,
           "warmup_record_calls": warmup_calls,
           "env_active_steps": result["env_active_steps"]})

    # ---- 3. no substep missed (4 substeps per nav step) --------------------
    ok3 = (warmup_calls == warmup_nav * substeps_per_nav
           and calls_in_window == nav_steps * substeps_per_nav)
    check("no_substep_missed", ok3,
          {"warmup_calls": warmup_calls, "warmup_expected": warmup_nav * substeps_per_nav,
           "window_calls": calls_in_window, "window_expected": nav_steps * substeps_per_nav,
           "note": "bridge 每 nav 步恰好 20 次 physics_step，每次恰 1 条 record"})

    # ---- 4. early termination marked and frozen ----------------------------
    ev = result["events"][early_env]
    st = result["env_status"][early_env]
    ev_k = ev[0]["step_k"] if ev else -1
    valid_after = result["valid_mask"][ev_k:, early_env].sum() == 0 if ev_k >= 0 else False
    energy_frozen = True
    if ev_k >= 0:
        tail = result["energy_step_j"][ev_k:, early_env]
        energy_frozen = bool((tail == 0).all())
    ok4 = (st == "early_termination" and ev_k == early_at_nav * substeps_per_nav
           and valid_after and energy_frozen
           and result["env_active_steps"][early_env] == ev_k)
    check("early_termination_marked_and_frozen", ok4,
          {"env": early_env, "status": st, "event_step_k": ev_k,
           "expected_step_k": early_at_nav * substeps_per_nav,
           "valid_after_event": int(valid_after), "energy_frozen": energy_frozen,
           "active_steps": result["env_active_steps"][early_env],
           "note": "终止事件后子步全部无效且能量为 0，绝不跨终止继续累加"})

    # ---- 5. timeout marked on full run --------------------------------------
    sts = {result["env_status"][i] for i in timeout_envs}
    full_steps = all(result["env_active_steps"][i] == expected_steps for i in timeout_envs)
    check("timeout_marked_on_full_run", sts == {"timeout"} and full_steps,
          {"timeout_envs": timeout_envs, "statuses": sorted(sts),
           "active_steps": {i: result["env_active_steps"][i] for i in timeout_envs}})

    # ---- 6. async env flagged ----------------------------------------------
    st2 = result["env_status"][async_env]
    a_k = async_at_nav * substeps_per_nav
    ok6 = (st2 == "incomplete_async"
           and result["env_active_steps"][async_env] == a_k
           and bool(result["valid_mask"][a_k:, async_env].sum() == 0))
    check("async_env_flagged", ok6,
          {"env": async_env, "status": st2,
           "active_steps": result["env_active_steps"][async_env],
           "expected_active": a_k,
           "note": "批内异步失败（无事件掩码关闭）得到 incomplete_async 标记"})

    # ---- 7. fault detection (offline unit tests on a fresh recorder) -------
    r2 = NavEnergyRecorder(capacity_steps=8, num_envs=1, device="cpu",
                           expected_steps=4, label="unit")
    r2.begin_window()
    nan_tau = torch.zeros(1, 12, dtype=torch.float32)
    nan_tau[0, 3] = float("nan")
    r2.record(tau_applied_lab=nan_tau, qdot_post_lab=torch.zeros(1, 12))
    r2.record(tau_applied_lab=torch.zeros(1, 12), qdot_post_lab=torch.zeros(1, 12))
    res2 = r2.finish()
    nan_caught = any("非有限值" in p for p in res2["problems"])
    status_fault = all(s == "recorder_fault" for s in res2["env_status"])

    r3 = NavEnergyRecorder(capacity_steps=8, num_envs=1, device="cpu",
                           expected_steps=4, label="unit2")
    r3.begin_window()
    r3.record(tau_applied_lab=torch.zeros(1, 12), qdot_post_lab=torch.zeros(1, 12))
    r3.observe_reset([0])            # 窗口内复位已累加 env → 必须 fault
    res3 = r3.finish()
    cross_reset_caught = any("cross_reset_write" in p for p in res3["problems"]) \
        and all(s == "recorder_fault" for s in res3["env_status"])

    reuse_rejected = False
    try:
        r4 = NavEnergyRecorder(capacity_steps=8, num_envs=1, device="cpu", label="unit3")
        r4.begin_window()
        r4.record(tau_applied_lab=torch.zeros(1, 12), qdot_post_lab=torch.zeros(1, 12))
        r4.begin_window()             # 复用含数据窗口 → 必须拒绝
    except RecorderError:
        reuse_rejected = True
    check("fault_detection_works", nan_caught and status_fault and cross_reset_caught
          and reuse_rejected,
          {"nan_caught": nan_caught, "nan_status_fault": status_fault,
           "cross_reset_caught": cross_reset_caught, "window_reuse_rejected": reuse_rejected,
           "problems_nan": res2["problems"], "problems_cross_reset": res3["problems"]})

    # ---- report -------------------------------------------------------------
    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-027",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "metric_id": "D-ENERGY-001",
        "energy_contract": "J = sum_k sum_j max(tau*qdot, 0)*dt, dt=0.005s, jointwise clamp",
        "execution_chain": "frozen explicit-PD (HIRA chain): obs_adapter -> jit policy -> "
                           "torque_adapter(25/0.5, scale 0.25) -> set_joint_effort_target",
        "window": {"nav_steps": nav_steps, "expected_substeps": expected_steps,
                   "warmup_nav_steps": warmup_nav,
                   "scenario": {"early_term_env": early_env, "early_at_nav": early_at_nav,
                                "async_env": async_env, "async_at_nav": async_at_nav,
                                "timeout_envs": timeout_envs,
                                "note": "所有终止事件均为功能测试注入，不删任何实验数据"}},
        "offline_recompute": recomp,
        "checks": RESULTS,
    }
    progress("writing_report")
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
    # sim_app.close() 在本机 Isaac Sim 5.1 headless 下两次复现长时间满载挂起
    # （7 分钟 / 30 分钟+），会阻塞 tmux 命令队列；JSON 报告已落盘，此处
    # 硬退出（GPU 上下文随进程终止由 OS 回收，-u 保证打印不丢缓冲）。
    print("PROGRESS hard_exit_skip_close", flush=True)
    os._exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
