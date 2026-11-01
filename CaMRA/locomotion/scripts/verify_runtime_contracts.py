"""Runtime contract tests on the target device (TASK-021).

Verifies, inside a real Isaac Sim session, the things unit tests cannot:

  A. rate chain: one 0.1 s high-level interval == 5 policy steps == 20 physics steps
  B. timestamps: sim_time advances by exactly step_dt, obs command == applied command
  C. external commands are not overwritten by the internal curriculum resampler
  D. episode boundaries: counters advance, command is cleared, old trajectory is not
     stitched onto the new one
  E. joint mapping: the 12 joint indices are a permutation, and sim order != SDK order
  F. limits / invalid input on a live env

Synthetic inputs are used only to provoke specific states; the pass/fail criteria
are counting and equality checks, not performance numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Runtime contract tests for the CaMRA locomotion stack.")
parser.add_argument("--task", type=str, default="CaMRA-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out_json", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from isaaclab.utils.string import resolve_matching_names

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.external_command import (
    DEFAULT_LIMITS,
    ExternalCommandWrapper,
    FixedProgram,
    clip_to_limits,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

RESULTS: dict = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} | {detail}", flush=True)
    return bool(ok)


def main() -> None:
    torch.manual_seed(0)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=True, entry_point_key="env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapper = ExternalCommandWrapper(env, program=FixedProgram([0.6, 0.0, 0.0]), on_reset="zero")
    bridge = wrapper.bridge
    u = wrapper.unwrapped

    step_dt = float(u.step_dt)
    physics_dt = float(u.cfg.sim.dt)
    decimation = int(u.cfg.decimation)
    action_dim = int(u.action_manager.total_action_dim)
    n = bridge.num_envs
    high_level_dt = 0.1

    print(f"[VERIFY] step_dt={step_dt} physics_dt={physics_dt} decimation={decimation} envs={n}", flush=True)

    # ---------------- A. rate chain ----------------
    wrapper.reset()
    actions = torch.zeros(n, action_dim, device=bridge.device)
    sim0 = int(u._sim_step_counter)
    t0 = float(u.common_step_counter) * step_dt
    samples = []
    n_steps = int(round(0.5 / step_dt))  # 0.5 s
    for _ in range(n_steps):
        sim_before = int(u._sim_step_counter)
        _, _, term, trunc, info = wrapper.step(actions)
        sim_after = int(u._sim_step_counter)
        samples.append(
            {
                "sim_time": float(info["external_command"].sim_time[0]),
                "physics_delta": sim_after - sim_before,
                "episode_step": float(info["external_command"].step_index[0]),
            }
        )

    phys_deltas = {s["physics_delta"] for s in samples}
    a_ok = phys_deltas == {decimation} and abs(step_dt - physics_dt * decimation) < 1e-12
    check(
        "A_rate_200_50",
        a_ok,
        f"physics per policy step={sorted(phys_deltas)} expected {{{decimation}}}; "
        f"step_dt={step_dt} == {physics_dt}*{decimation}",
    )

    # 0.1 s interval must contain exactly 5 policy steps
    times = [s["sim_time"] for s in samples]
    per_interval = []
    for i in range(len(times) - 5):
        if abs((times[i + 5] - times[i]) - high_level_dt) < 1e-6:
            per_interval.append(5)
            break
    steps_per_100ms = int(round(high_level_dt / step_dt))
    check(
        "A_rate_10Hz_interval",
        steps_per_100ms == 5 and bool(per_interval),
        f"policy steps per 0.1 s = {steps_per_100ms} (expected 5); measured window found={bool(per_interval)}",
    )

    # ---------------- B. timestamp alignment + obs command ----------------
    deltas = [round(times[i + 1] - times[i], 9) for i in range(len(times) - 1)]
    mono = all(d >= 0 for d in deltas) and all(abs(d - step_dt) < 1e-6 for d in deltas)
    check(
        "B_sim_time_advances_by_step_dt",
        mono,
        f"unique deltas={sorted(set(deltas))[:4]} expected {step_dt}",
    )

    # obs command channel must equal the applied command on the same step.
    # The wrapper's program rewrites the buffer every step, so drop it first.
    wrapper.program = None
    bridge.set_command([0.42, -0.13, 0.27])
    obs, _, _, _, info = wrapper.step(actions)
    rec = info["external_command"]
    op = obs["policy"] if isinstance(obs, dict) else obs
    obs_cmd = [round(float(x), 6) for x in op[0, 6:9]]
    app_cmd = [round(float(x), 6) for x in rec.applied[0]]
    check(
        "B_obs_command_matches_applied",
        obs_cmd == app_cmd and app_cmd == [0.42, -0.13, 0.27],
        f"obs[6:9]={obs_cmd} applied={app_cmd}",
    )

    # ---------------- C. no internal resampling over 15 s ----------------
    bridge.set_command([0.6, 0.0, 0.0])
    target = torch.tensor([0.6, 0.0, 0.0], device=bridge.device)
    rs0 = bridge.stats["internal_resample_count"]
    max_dev = 0.0
    for _ in range(int(round(15.0 / step_dt))):
        _, _, _, _, info = wrapper.step(actions)
        rec = info["external_command"]
        max_dev = max(max_dev, float((rec.applied - target).abs().max().item()))
    rs_delta = bridge.stats["internal_resample_count"] - rs0
    check(
        "C_external_command_survives_15s",
        max_dev <= 1e-6 and rs_delta == 0,
        f"15 s / {int(round(15.0 / step_dt))} steps max_dev={max_dev:.2e} timer_resamples={rs_delta}",
    )

    # ---------------- D. episode boundary does not stitch trajectories ----------------
    bridge.set_command([0.5, 0.0, 0.0])
    u.episode_length_buf.fill_(int(u.max_episode_length) - 1)
    ep_before = int(bridge._episode_index[0].item())
    saw = False
    for _ in range(5):
        _, _, term, trunc, info = wrapper.step(actions)
        rec = info["external_command"]
        if bool(rec.new_episode[0]):
            saw = True
            break
    ep_after = int(bridge._episode_index[0].item())
    print(
        f"[D-DEBUG] raw term={term[0].item()} trunc={trunc[0].item()} | "
        f"rec.term={rec.terminated[0].item()} rec.trunc={rec.truncated[0].item()} "
        f"rec.new={rec.new_episode[0].item()} rec.ep={rec.episode_index[0].item()}",
        flush=True,
    )
    # the closing step reports how many steps the OLD episode ran and is excluded
    # from trajectory metrics; the very next step must start a fresh counter
    closed_steps = int(rec.step_index[0].item())
    internal_step_after = int(bridge._step_index[0].item())
    _, _, _, _, info2 = wrapper.step(actions)
    rec2 = info2["external_command"]
    parts = {
        "saw_new_episode": bool(saw),
        "ended_by_term_or_trunc": bool(rec.truncated[0].item() or rec.terminated[0].item()),
        "command_cleared": float(rec.applied[0].abs().max().item()) < 1e-9,
        "excluded_from_metrics": not bool(rec.trajectory_valid_mask()[0]),
        "internal_step_reset": internal_step_after == 0,
        "next_record_step_zero": int(rec2.step_index[0].item()) == 0,
        "episode_counter_bumped": ep_after == ep_before + 1,
    }
    d_ok = all(parts.values())
    check(
        "D_episode_boundary_isolated",
        d_ok,
        f"new_episode={saw} truncated={bool(rec.truncated[0])} closed_after_steps={closed_steps} "
        f"applied_after_reset={float(rec.applied[0].abs().max().item()):.1e} "
        f"internal_step_index={internal_step_after} next_record_step_index={int(rec2.step_index[0].item())} "
        f"episode {ep_before}->{ep_after} parts={parts}",
    )

    # ---------------- E. joint mapping ----------------
    # Sim order comes from the USD (grouped by joint type), SDK order from the
    # Unitree config (grouped by leg). The deploy path must reorder via joint_ids_map.
    from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG

    robot = u.scene["robot"]
    sim_names = list(robot.data.joint_names)
    sdk_names = list(UNITREE_GO2_CFG.joint_sdk_names)
    joint_ids_map, _ = resolve_matching_names(sim_names, sdk_names, preserve_order=True)
    joint_ids_map = list(joint_ids_map)
    perm_ok = sorted(joint_ids_map) == list(range(len(sim_names))) and len(sim_names) == 12
    check(
        "E_joint_map_is_permutation",
        perm_ok,
        f"sim={len(sim_names)} sdk={len(sdk_names)} map={joint_ids_map}",
    )

    identity = list(range(len(sim_names)))
    check(
        "E_sim_order_differs_from_sdk",
        joint_ids_map != identity and len(sdk_names) == len(sim_names),
        f"sim order starts {sim_names[:4]} ; sdk order starts {sdk_names[:4]} ; "
        f"reorder_needed={joint_ids_map != identity}",
    )

    # ---------------- F. limits and invalid input on the live env ----------------
    wrapper.program = None
    bridge.set_command([2.0, -2.0, 5.0])
    _, _, _, _, info = wrapper.step(actions)
    rec_lim = info["external_command"]
    applied_lim = [round(float(x), 3) for x in rec_lim.applied[0]]
    lim_ok = applied_lim == [1.0, -0.4, 1.0] and bool(rec_lim.clipped.all())
    bridge.set_command([0.5, float("nan"), 0.0])
    _, _, _, _, info = wrapper.step(actions)
    rec_nan = info["external_command"]
    nan_ok = float(rec_nan.applied[0].abs().max().item()) < 1e-9 and bool(rec_nan.invalid.all())
    check(
        "F_limits_and_invalid_on_env",
        lim_ok and nan_ok,
        f"request=[2.0,-2.0,5.0] applied={applied_lim} clipped={bool(rec_lim.clipped.all())}; "
        f"nan_to_zero={nan_ok} invalid={bool(rec_nan.invalid.all())}",
    )

    RESULTS["counts"] = {
        "physics_per_policy": decimation,
        "policy_hz": 1.0 / step_dt,
        "physics_hz": 1.0 / physics_dt,
        "policy_steps_per_high_level": steps_per_100ms,
        "high_level_hz": 1.0 / high_level_dt,
    }
    RESULTS["all_pass"] = all(v["pass"] for v in RESULTS.values() if isinstance(v, dict) and "pass" in v)
    print("[TASK021_RESULT] " + json.dumps(RESULTS, ensure_ascii=False), flush=True)

    out = args_cli.out_json or os.path.abspath("logs/camra/TASK-021/runtime_contracts.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2)
    print(f"[VERIFY] json -> {out}", flush=True)

    wrapper.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
