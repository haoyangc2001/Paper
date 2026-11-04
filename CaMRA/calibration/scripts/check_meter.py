#!/usr/bin/env python3
"""CAL-014 计量验证：一次 31 s 单回合（1 s warmup + 30 s 窗口）。

``--mode on``  打开 200 Hz 记录与能量计量，落盘 tau/qdot 原始数据 + 50 Hz 状态；
``--mode off`` 完全关闭记录（不调用 robot.update、不计量），只落 50 Hz 状态。

两次运行使用同一 seed / profile / 初始位姿 / 命令程序，由
``scripts/verify_meter_runs.py`` 比对动作与状态是否被记录行为改变。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
for p in (str(CAL / "src"), str(CAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.energy import steps_for  # noqa: E402
from camra_calibration.profiles import sample_applied, sample_initial_pose  # noqa: E402


class StateOnlyRecorder:
    """关闭计量时的 50 Hz 记录（不 import simulator 内部，不调用 robot.update）。"""

    def __init__(self, robot, num_envs: int):
        self.robot = robot
        self.num_envs = num_envs
        self.warmup_steps = 0
        self._in_window = False
        self._action, self._cmd, self._pos, self._quat = [], [], [], []
        self._lin, self._ang, self._jpos, self._jvel = [], [], [], []

    def begin_warmup(self) -> None:
        self.warmup_steps = 0

    def count_warmup_step(self) -> None:
        self.warmup_steps += 1

    def begin_window(self) -> None:
        self._in_window = True

    def after_physics_step(self) -> None:
        self.count_warmup_step()

    def after_policy_step(self, action, cmd) -> None:
        d = self.robot.data
        self._action.append(action.detach().to("cpu", torch.float32).clone())
        self._cmd.append(cmd.detach().to("cpu", torch.float32).clone())
        self._pos.append(d.root_pos_w.detach().to("cpu", torch.float32).clone())
        self._quat.append(d.root_quat_w.detach().to("cpu", torch.float32).clone())
        self._lin.append(d.root_lin_vel_b.detach().to("cpu", torch.float32).clone())
        self._ang.append(d.root_ang_vel_b.detach().to("cpu", torch.float32).clone())
        self._jpos.append(d.joint_pos.detach().to("cpu", torch.float32).clone())
        self._jvel.append(d.joint_vel.detach().to("cpu", torch.float32).clone())

    def finish(self, outcome: str = "completed") -> dict:
        self.outcome = outcome
        return {"label": "off", "outcome": outcome, "record_physics": False}

    def save(self, out_dir: Path, stem: str):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_50hz": torch.stack(self._action),
            "cmd_50hz": torch.stack(self._cmd),
            "root_pos_50hz": torch.stack(self._pos),
            "root_quat_50hz": torch.stack(self._quat),
            "root_lin_vel_b_50hz": torch.stack(self._lin),
            "root_ang_vel_b_50hz": torch.stack(self._ang),
            "joint_pos_50hz": torch.stack(self._jpos),
            "joint_vel_50hz": torch.stack(self._jvel),
        }
        npz = out_dir / f"{stem}.npz"
        torch.save(payload, npz)
        meta = out_dir / f"{stem}.json"
        meta.write_text(json.dumps({
            "label": "off", "measure": False, "warmup_steps": self.warmup_steps,
            "control_steps_50hz": len(self._action),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return npz, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("on", "off"), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", default="CaMRA-Go2-Velocity")
    ap.add_argument("--num_envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--trial_seed", type=int, default=6000)
    ap.add_argument("--profile", default="clean")
    ap.add_argument("--family", default="straight_shift")
    ap.add_argument("--warmup_s", type=float, default=1.0)
    ap.add_argument("--window_s", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device)
    sim_app = launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401,F403
    import unitree_rl_lab.tasks  # noqa: F401,F403
    from sim.profiles_sim import (  # noqa: F401
        apply_friction, apply_initial_pose, apply_mass, apply_push,
        configure_env_for_profile,
    )
    from sim.recorder import TrialRecorder
    from unitree_rl_lab.utils.external_command import (
        DEFAULT_LIMITS, ExternalCommandWrapper, StepProgram,
    )
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    warmup_steps = steps_for(args.warmup_s)          # 200
    window_steps = steps_for(args.window_s)          # 6000
    policy_dt = 0.02
    warmup_ctrl = round(args.warmup_s / policy_dt)   # 50
    window_ctrl = round(args.window_s / policy_dt)   # 1500

    applied = sample_applied(args.profile, args.trial_seed, duration_s=args.window_s)
    pose = sample_initial_pose(args.trial_seed)
    prog = build_program(args.family, args.trial_seed, duration_s=args.window_s, command_dt=0.1)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = args.seed
    # 单回合必须容纳 1 s warmup + 30 s 窗口；训练默认 20 s 会提前截断。
    # 另留 0.2 s（10 个控制步）余量，避免窗口最后一步正好撞上 time_out 被记成 truncated。
    episode_margin_s = 0.2
    env_cfg.episode_length_s = args.warmup_s + args.window_s + episode_margin_s
    report_cfg = configure_env_for_profile(env_cfg, applied)

    raw = gym.make(args.task, cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    u = raw.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    sim_dt = float(sim.get_physics_dt())
    step_dt = float(u.step_dt)
    dec = int(round(step_dt / sim_dt))
    assert dec == 4, dec
    assert abs(step_dt - policy_dt) < 1e-12
    max_episode_steps = int(round(u.max_episode_length))
    required_ctrl = warmup_ctrl + window_ctrl
    if max_episode_steps <= required_ctrl:
        raise RuntimeError(
            f"trial 时限不足：max_episode_length={max_episode_steps} 步 ≤ 需要的 {required_ctrl} 步")

    # 命令程序：warmup 段全零 + 300 个 0.1 s 段（10 Hz 零阶保持）
    segments = [{"duration": args.warmup_s, "vel": [0.0, 0.0, 0.0]}]
    segments += [{"duration": 0.1, "vel": list(prog.commands[i])} for i in range(window_ctrl // 5)]
    env.set_program(StepProgram(segments))

    obs, _ = env.reset()
    obs = obs["policy"]
    env_ids = list(range(args.num_envs))
    apply_friction(robot, [i for i in env_ids], applied.friction, restitution=applied.restitution)
    for i in env_ids:
        apply_mass(robot, [i], applied.added_mass_kg)
        apply_initial_pose(robot, [i], [pose["px"]], [pose["py"]], [pose["yaw"]])

    recorder = (TrialRecorder(robot=robot, num_envs=args.num_envs, dt=sim_dt,
                              expected_steps=window_steps, n_joints=robot.num_joints,
                              record_physics=True, label=f"{args.mode}/{args.profile}")
                if args.mode == "on" else
                StateOnlyRecorder(robot, args.num_envs))

    orig_step = sim.step

    def patched_step(*a, **kw):
        out = orig_step(*a, **kw)
        recorder.after_physics_step()
        return out

    if args.mode == "on":
        sim.step = patched_step

    recorder.begin_warmup()
    outcome = "completed"
    steps_executed = 0
    push_log: list[dict] = []
    policy = None
    from pathlib import Path as _P
    bundle = _P("/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2")
    policy = torch.jit.load(str(bundle / "exported" / "policy.pt"), map_location=args.device).eval()

    with torch.no_grad():
        # warmup：全零命令，只推进不记录
        for _ in range(warmup_ctrl):
            act = policy(obs)
            obs, _r, _term, _trunc, _info = env.step(act)
            obs = obs["policy"]
        recorder.begin_window()

        for k in range(window_ctrl):
            act = policy(obs)
            obs, _r, term, trunc, _info = env.step(act)
            obs = obs["policy"]
            cmd = u.command_manager.get_command("base_velocity").detach().clone()
            recorder.after_policy_step(act, cmd)
            steps_executed = recorder.window_steps if args.mode == "on" else (k + 1) * dec
            if bool(term.any()):
                outcome = "terminated"
                break
            if bool(trunc.any()):
                outcome = "truncated"
                break

    if args.mode == "on":
        sim.step = orig_step
    summary = recorder.finish(outcome)
    summary.update({
        "mode": args.mode,
        "profile": args.profile,
        "family": args.family,
        "trial_seed": args.trial_seed,
        "env_seed": args.seed,
        "warmup_s": args.warmup_s,
        "window_s": args.window_s,
        "expected_window_steps": window_steps,
        "expected_control_steps": window_ctrl,
        "sim_dt": sim_dt,
        "policy_dt": step_dt,
        "decimation": dec,
        "max_episode_length_steps": max_episode_steps,
        "episode_length_s": env_cfg.episode_length_s,
        "steps_executed": steps_executed,
        "applied_profile": applied.to_dense_dict(),
        "initial_pose": pose,
        "program_sha256": prog.sha256(),
        "cfg_report": report_cfg.to_dict(),
    })
    out_dir = Path(args.out_dir)
    npz, meta = recorder.save(out_dir, f"meter_{args.mode}")
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("mode", "outcome", "warmup_steps", "window_steps", "control_steps_50hz",
                       "energy_total_j", "ok", "problems") if k in summary},
                     ensure_ascii=False))
    print("SAVED", npz.name, meta.name)
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
