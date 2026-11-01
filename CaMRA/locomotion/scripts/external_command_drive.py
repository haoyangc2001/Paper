"""Drive the Go2 locomotion policy from an external velocity-command program.

This is the reusable entry point requested by TASK-018: it takes batched 3-D
velocity commands from a named program (fixed / step / ramp / sine / combo),
applies them through :class:`ExternalCommandWrapper`, and writes a CSV of the
*actually applied* command together with the simulation clock.

Example::

    python scripts/external_command_drive.py --headless --num_envs 4 \
        --program step --episode_seconds 12

The command contract is ``CaMRA/locomotion/docs/command_interface.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive Go2 with an external velocity-command program.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_seconds", type=float, default=12.0, help="How long to drive the policy.")
parser.add_argument(
    "--program",
    type=str,
    default="fixed",
    choices=["fixed", "step", "ramp", "sine", "combo"],
)
parser.add_argument("--vel", type=float, nargs=3, default=[0.8, 0.0, 0.0], help="fixed: vx vy wz")
parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="ramp: start command")
parser.add_argument("--end", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="ramp: end command")
parser.add_argument("--ramp_seconds", type=float, default=5.0, help="ramp: ramp duration")
parser.add_argument("--axis", type=int, default=0, help="sine: channel index (0=vx,1=vy,2=wz)")
parser.add_argument("--amplitude", type=float, default=0.5, help="sine: amplitude")
parser.add_argument("--frequency", type=float, default=0.2, help="sine: frequency in Hz")
parser.add_argument(
    "--program_json",
    type=str,
    default=None,
    help="JSON file with {'type': ..., 'segments': [...]} for step/combo programs.",
)
parser.add_argument("--on_reset", type=str, default="zero", choices=["zero", "hold"])
parser.add_argument("--action", type=str, default="zero", choices=["zero", "random"], help="Action source.")
parser.add_argument("--csv", type=str, default=None, help="CSV output path (default under logs/).")
parser.add_argument("--quiet_every", type=int, default=50, help="Print a progress line every N steps.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.external_command import (
    DEFAULT_LIMITS,
    ComboProgram,
    ExternalCommandWrapper,
    FixedProgram,
    RampProgram,
    SineProgram,
    StepProgram,
    build_program,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _build_program():
    """Resolve the requested program from CLI arguments."""
    kind = args_cli.program
    if args_cli.program_json:
        with open(args_cli.program_json, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        kind = spec.get("type", kind)
        if kind in ("step", "combo"):
            return build_program(kind, segments=spec["segments"])
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return build_program(kind, **kwargs)

    if kind == "fixed":
        return FixedProgram(args_cli.vel)
    if kind == "step":
        return StepProgram(
            [
                {"duration": 3.0, "vel": [0.0, 0.0, 0.0]},
                {"duration": 3.0, "vel": [0.6, 0.0, 0.0]},
                {"duration": 3.0, "vel": [1.0, 0.0, 0.0]},
                {"duration": 3.0, "vel": [0.5, 0.2, 0.0]},
                {"duration": 3.0, "vel": [0.5, 0.0, 0.5]},
                {"duration": 3.0, "vel": [0.0, 0.0, 0.0]},
            ]
        )
    if kind == "ramp":
        return RampProgram(args_cli.start, args_cli.end, args_cli.ramp_seconds)
    if kind == "sine":
        return SineProgram(args_cli.axis, args_cli.amplitude, args_cli.frequency)
    if kind == "combo":
        return ComboProgram(
            [
                {"duration": 3.0, "type": "fixed", "kwargs": {"vel": [0.0, 0.0, 0.0]}},
                {"duration": 3.0, "type": "ramp", "kwargs": {"start": [0.0, 0.0, 0.0], "end": [1.0, 0.0, 0.0], "duration": 3.0}},
                {"duration": 3.0, "type": "sine", "kwargs": {"axis": 2, "amplitude": 0.5, "frequency": 0.25, "base": [0.8, 0.0, 0.0]}},
                {"duration": 3.0, "type": "fixed", "kwargs": {"vel": [0.0, 0.0, 0.0]}},
            ]
        )
    raise ValueError(f"unsupported program {kind}")


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    # headless driving must never touch the command-visualisation USD
    # (it lives on S3 and the cluster has no route to it -> 300 s stall)
    env_cfg.commands.base_velocity.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapper = ExternalCommandWrapper(
        env,
        program=_build_program(),
        limits=DEFAULT_LIMITS,
        enforce_limits=True,
        on_reset=args_cli.on_reset,
    )
    bridge = wrapper.bridge

    unwrapped = wrapper.unwrapped
    sim_dt = float(unwrapped.step_dt)
    action_dim = int(unwrapped.action_manager.total_action_dim)
    num_steps = max(1, int(round(args_cli.episode_seconds / sim_dt)))

    print(f"[DRIVE] task={args_cli.task} num_envs={bridge.num_envs} device={bridge.device}", flush=True)
    print(f"[DRIVE] step_dt={sim_dt:.4f}s -> {num_steps} steps ({args_cli.episode_seconds:.2f}s)", flush=True)
    print(f"[DRIVE] program={wrapper.program.describe()}", flush=True)
    print(f"[DRIVE] bridge: {bridge.describe()}", flush=True)

    wrapper.reset()

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.abspath(f"logs/external_command/drive_{args_cli.program}_{stamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(
        [
            "step",
            "env_id",
            "sim_time",
            "episode_index",
            "step_index",
            "req_vx",
            "req_vy",
            "req_wz",
            "app_vx",
            "app_vy",
            "app_wz",
            "clipped",
            "invalid",
            "terminated",
            "truncated",
            "new_episode",
        ]
    )

    total_terminated = 0
    total_truncated = 0
    clipped_steps = 0
    t0 = time.time()

    for step in range(num_steps):
        if args_cli.action == "zero":
            actions = torch.zeros(bridge.num_envs, action_dim, device=bridge.device)
        else:
            actions = 2.0 * torch.rand(bridge.num_envs, action_dim, device=bridge.device) - 1.0

        _, _, terminated, truncated, info = wrapper.step(actions)
        rec = info["external_command"]

        req = rec.requested.detach().cpu()
        app = rec.applied.detach().cpu()
        for env_id in range(bridge.num_envs):
            writer.writerow(
                [
                    step,
                    env_id,
                    f"{float(rec.sim_time[env_id]):.4f}",
                    int(rec.episode_index[env_id]),
                    int(rec.step_index[env_id]),
                    f"{float(req[env_id, 0]):.4f}",
                    f"{float(req[env_id, 1]):.4f}",
                    f"{float(req[env_id, 2]):.4f}",
                    f"{float(app[env_id, 0]):.4f}",
                    f"{float(app[env_id, 1]):.4f}",
                    f"{float(app[env_id, 2]):.4f}",
                    int(bool(rec.clipped[env_id])),
                    int(bool(rec.invalid[env_id])),
                    int(bool(rec.terminated[env_id])),
                    int(bool(rec.truncated[env_id])),
                    int(bool(rec.new_episode[env_id])),
                ]
            )

        total_terminated += int(rec.terminated.sum().item())
        total_truncated += int(rec.truncated.sum().item())
        clipped_steps += int(rec.clipped.sum().item())

        if args_cli.quiet_every and step % args_cli.quiet_every == 0:
            print(
                f"[DRIVE] step={step:5d} t={float(rec.sim_time[0]):6.2f}s "
                f"cmd={[round(float(x), 3) for x in rec.applied[0]]} "
                f"ep={int(rec.episode_index[0])} term={total_terminated} trunc={total_truncated}",
                flush=True,
            )

    fh.close()
    wall = time.time() - t0

    summary = {
        "task": args_cli.task,
        "num_envs": bridge.num_envs,
        "program": wrapper.program.describe(),
        "steps": num_steps,
        "sim_seconds": num_steps * sim_dt,
        "wall_seconds": round(wall, 2),
        "csv": csv_path,
        "terminated": total_terminated,
        "truncated": total_truncated,
        "clipped_env_steps": clipped_steps,
        "final_episode_index": bridge._episode_index.detach().cpu().tolist(),
        "bridge_stats": dict(bridge.stats),
        "applied_final": bridge.term.vel_command_b.detach().cpu().tolist(),
        "internal_resampling_range": list(bridge.term.cfg.resampling_time_range),
        "debug_vis": bool(bridge.term.cfg.debug_vis),
    }
    print("[DRIVE_SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)

    wrapper.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
