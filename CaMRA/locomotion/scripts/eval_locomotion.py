"""Reproducible locomotion evaluation and logging (TASK-020).

Runs a checkpoint (or an untrained policy, for tool self-tests) through the
scenarios fixed in ``docs/eval_protocol.json`` with deterministic actor
inference, and writes:

  <out>/run_info.json          versions, hashes, protocol copy, data origin
  <out>/per_episode.csv        one row per episode with all metrics
  <out>/summary.json           per-scenario / per-group aggregation + pass flags
  <out>/raw/<group>/<sc>.npz   step-level records (every metric is recomputable)
  <out>/snippets/...csv|.png   a few representative replays for eyeballing

Design notes required by the protocol:

* **Deterministic inference**: actions come from ``ActorCritic.act_inference``
  (mean action, no sampling noise), never ``act``.
* **No cherry-picking**: every episode is logged, including early terminations.
  ``failure_rate`` counts ``base_contact`` / ``bad_orientation`` terminations;
  terminated episodes still contribute their pre-termination samples.
* **Variant coverage** (TASK-031): every command variant has its own episode
  quota and a fixed slice of environments; the loop stops only when *each*
  variant is full. See ``unitree_rl_lab.utils.variant_scheduler``.
* **Warm-up**: the first ``warmup.settle_s`` (1.0 s) after a reset is excluded
  from tracking metrics, per protocol.
* **Error frame**: velocities are body-frame (``root_lin_vel_b`` /
  ``root_ang_vel_b``), commands are body-frame, both in m/s and rad/s.
* **Sampling**: one record per policy step (50 Hz); ``step_dt`` is recorded.
* **This is simulation data.** ``run_info.json`` states it explicitly; nothing
  here may be labelled as real-robot measurement.
* alpha/delta calibration is explicitly **out of scope** for this task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Go2 locomotion policy against the fixed protocol.")
parser.add_argument("--task", type=str, default="CaMRA-Go2-Velocity")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model_*.pt; omit for an untrained self-test.")
parser.add_argument("--protocol", type=str, default=None, help="eval_protocol.json (default: repo docs).")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=200, help="Env seed; protocol final_test seeds are 200-209.")
parser.add_argument("--group", type=str, default="nominal", choices=["nominal", "perturbed"])
parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenario ids, or 'all'.")
parser.add_argument(
    "--episodes_per_variant",
    type=int,
    default=None,
    help=(
        "Episodes required per command variant (sign direction / sign combination). "
        "This is the quota the loop actually enforces: multi-direction scenarios must satisfy it "
        "separately for every variant. Protocol v1.0.0 supplement D-2 uses 30 per variant per seed."
    ),
)
parser.add_argument(
    "--episodes_per_scenario",
    type=int,
    default=300,
    help=(
        "Legacy total-per-scenario budget. Only used when --episodes_per_variant is omitted: "
        "the per-variant quota becomes episodes_per_scenario // num_variants (so the total stays comparable)."
    ),
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="policy",
    choices=["policy", "random"],
    help="policy = deterministic actor; random = uniform actions (self-test only, provokes falls)",
)
parser.add_argument("--replay_snippets", type=int, default=1, help="Representative replays to export per scenario (0=off).")
parser.add_argument("--max_steps_per_episode", type=int, default=1200, help="Safety cap (24 s at 50 Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.utils.external_command import (
    DEFAULT_LIMITS,
    ComboProgram,
    ExternalCommandWrapper,
    FixedProgram,
    RampProgram,
    StepProgram,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.variant_scheduler import VariantScheduler, assign_variants

REPO = "/home/caohy/repositories/Paper"
DEFAULT_PROTOCOL = os.path.join(REPO, "CaMRA/locomotion/docs/eval_protocol.json")

from unitree_rl_lab.utils.eval_metrics import (  # noqa: E402  (after AppLauncher)
    FIELDS,
    I,
    NF,
    REASONS,
    aggregate,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# protocol helpers
# ---------------------------------------------------------------------------


def load_protocol(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def scenario_specs(protocol: dict, wanted: str) -> list[dict]:
    specs = protocol["scenarios"]
    if wanted == "all":
        return specs
    ids = {s.strip() for s in wanted.split(",") if s.strip()}
    return [s for s in specs if s["id"] in ids]


def program_variants(spec: dict) -> list:
    """Build the (possibly sign-flipped) command programs for a scenario."""
    prog = spec["command_program"]
    kind = prog["type"]
    duration = float(spec["duration_s"])

    if kind == "constant":
        vx, vy, wz = float(prog.get("vx", 0.0)), float(prog.get("vy", 0.0)), float(prog.get("wz", 0.0))
        if prog.get("sign_combinations"):
            n = int(prog["sign_combinations"])
            combos = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)][:n]
            return [FixedProgram([vx * a, vy * b, wz * c]) for a, b, c in combos]
        variants = [(1, 1, 1)]
        if prog.get("vy_sign") == "both":
            variants = [(1, 1, 1), (1, -1, 1)]
        elif prog.get("wz_sign") == "both":
            variants = [(1, 1, 1), (1, 1, -1)]
        return [FixedProgram([vx * a, vy * b, wz * c]) for a, b, c in variants]

    if kind == "step":
        # profile is [[value, until_t], ...] -> piecewise-constant segments
        profile = prog["profile"]
        segments, prev_t = [], 0.0
        for value, until in profile:
            dt = float(until) - prev_t
            if dt > 0:
                segments.append({"duration": dt, "vel": [float(value), 0.0, 0.0]})
            prev_t = float(until)
        if prev_t < duration:
            segments.append({"duration": duration - prev_t, "vel": [0.0, 0.0, 0.0]})
        return [StepProgram(segments)]

    if kind == "ramp":
        t_up0, t_up1 = prog["t_up"]
        t_dn0, t_dn1 = prog["t_down"]
        v0, v1 = float(prog["vx_from"]), float(prog["vx_to"])
        segs = []
        if t_up0 > 0:
            segs.append({"duration": t_up0, "program": FixedProgram([v0, 0.0, 0.0])})
        segs.append({"duration": t_up1 - t_up0, "program": RampProgram([v0, 0, 0], [v1, 0, 0], t_up1 - t_up0)})
        if t_dn0 > t_up1:
            segs.append({"duration": t_dn0 - t_up1, "program": FixedProgram([v1, 0.0, 0.0])})
        segs.append({"duration": t_dn1 - t_dn0, "program": RampProgram([v1, 0, 0], [v0, 0, 0], t_dn1 - t_dn0)})
        if duration > t_dn1:
            segs.append({"duration": duration - t_dn1, "program": FixedProgram([v0, 0.0, 0.0])})
        return [ComboProgram(segs)]

    raise ValueError(f"unsupported command program type: {kind}")


def apply_group_overrides(env_cfg, group: str, protocol: dict) -> dict:
    """Configure the perturbation knobs for nominal / perturbed. Returns the applied diff."""
    applied = {}
    if group == "nominal":
        # friction fixed at 1.0
        ev = env_cfg.events.physics_material
        ev.params["static_friction_range"] = (1.0, 1.0)
        ev.params["dynamic_friction_range"] = (1.0, 1.0)
        ev.params["restitution_range"] = (0.0, 0.0)
        # no added base mass
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        # no push (the term only declares x/y)
        env_cfg.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        # observation noise off
        env_cfg.observations.policy.enable_corruption = False
        applied = {
            "friction": [1.0, 1.0],
            "added_base_mass_kg": [0.0, 0.0],
            "push_velocity_mps": [0.0, 0.0],
            "obs_noise": "disabled",
        }
    else:
        spec = protocol["groups"]["perturbed"]["implemented_knobs_used"]
        fr = spec["friction_static_dynamic"]
        env_cfg.events.physics_material.params["static_friction_range"] = tuple(fr)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = tuple(fr)
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = tuple(spec["added_base_mass_kg"])
        pv = float(spec["push_velocity_mps"][1])
        env_cfg.events.push_robot.params["velocity_range"] = {"x": (-pv, pv), "y": (-pv, pv)}
        env_cfg.observations.policy.enable_corruption = True
        applied = {
            "friction": list(fr),
            "added_base_mass_kg": list(spec["added_base_mass_kg"]),
            "push_velocity_mps": list(spec["push_velocity_mps"]),
            "obs_noise": "enabled (training values)",
        }
    return applied


# ---------------------------------------------------------------------------
# step store
# ---------------------------------------------------------------------------


class StepStore:
    """Per-env rolling record of step rows, sliced into episodes on completion."""

    def __init__(self, num_envs: int, num_fields: int, capacity: int, device):
        self.num_envs = num_envs
        self.device = device
        self.capacity = capacity
        self.buf = torch.zeros(capacity, num_envs, num_fields, device=device)
        self.t = 0
        self.start = torch.zeros(num_envs, dtype=torch.long, device=device)

    def push(self, row: torch.Tensor) -> None:
        if self.t >= self.capacity:
            extra = torch.zeros(self.capacity, self.num_envs, self.buf.shape[2], device=self.device)
            self.buf = torch.cat([self.buf, extra], dim=0)
            self.capacity = self.buf.shape[0]
        self.buf[self.t] = row.detach()
        self.t += 1

    def episode(self, env_id: int) -> torch.Tensor:
        s = int(self.start[env_id].item())
        return self.buf[s : self.t, env_id].clone().detach()

    def restart(self, env_ids: torch.Tensor) -> None:
        self.start[env_ids] = self.t


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_agent_cfg(checkpoint: str | None, task: str) -> dict:
    """Prefer the agent.yaml stored next to the checkpoint; else the task entry point."""
    if checkpoint:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
        yml = os.path.join(run_dir, "params", "agent.yaml")
        if os.path.isfile(yml):
            with open(yml, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        print(f"[WARN] agent.yaml not found at {yml}; falling back to the task entry point", flush=True)
    spec = gym.spec(task)
    mod_name, cls_name = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    import importlib

    cfg = getattr(importlib.import_module(mod_name), cls_name)()
    return cfg.to_dict()


class _RunnerShim:
    """Minimal VecEnv surface needed by OnPolicyRunner construction."""

    def __init__(self, wrapper: ExternalCommandWrapper):
        u = wrapper.unwrapped
        self._wrapper = wrapper
        self.unwrapped = u
        self.num_envs = int(wrapper.bridge.num_envs)
        self.device = wrapper.bridge.device
        self.num_actions = int(u.action_manager.total_action_dim)
        self.max_episode_length = int(u.max_episode_length)
        self._obs = None

    def set_obs(self, obs):
        self._obs = obs

    def get_observations(self):
        return self._obs


def _rpy_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=1)


def _write_snippet(out_dir: str, name: str, ep: np.ndarray, step_dt: float) -> None:
    # Write one episode's step rows as CSV plus a stacked PNG (auxiliary check only).
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for row in ep:
            w.writerow([f"{float(v):.6g}" for v in row])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] snippet plot skipped: {exc}", flush=True)
        return

    t = np.arange(ep.shape[0]) * step_dt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    pairs = [
        ("vx_body", "cmd_vx", "vx [m/s]"),
        ("vy_body", "cmd_vy", "vy [m/s]"),
        ("wz_body", "cmd_wz", "wz [rad/s]"),
    ]
    for ax, (actual, cmd, label) in zip(axes, pairs):
        ax.plot(t, ep[:, I[actual]], label="actual", lw=1.2)
        ax.plot(t, ep[:, I[cmd]], label="command", lw=1.0, ls="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(f"{name} (simulation, deterministic actor)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    protocol_path = args_cli.protocol or DEFAULT_PROTOCOL
    protocol = load_protocol(protocol_path)
    specs = scenario_specs(protocol, args_cli.scenarios)
    if not specs:
        raise SystemExit("no scenarios selected")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args_cli.out_dir or os.path.join("/pub/data/caohy/logs/camra/TASK-020", f"eval_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw", args_cli.group)
    snip_dir = os.path.join(out_dir, "snippets", args_cli.group)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(snip_dir, exist_ok=True)

    # ---- environment ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
                            entry_point_key="env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    applied_group = apply_group_overrides(env_cfg, args_cli.group, protocol)

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapper = ExternalCommandWrapper(env, program=None, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    bridge = wrapper.bridge
    u = wrapper.unwrapped
    step_dt = float(u.step_dt)
    warmup_s = float(protocol.get("warmup", {}).get("settle_s", 1.0))
    limits = [list(map(float, x)) for x in DEFAULT_LIMITS]

    # ---- policy ----
    agent_cfg = load_agent_cfg(args_cli.checkpoint, args_cli.task)
    og = agent_cfg.get("obs_groups") or {}
    if "policy" not in og or "critic" not in og:
        agent_cfg["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    shim = _RunnerShim(wrapper)
    obs, _ = wrapper.reset()
    shim.set_obs(obs)
    obs = shim.get_observations()
    runner = OnPolicyRunner(shim, agent_cfg, log_dir=None, device=str(bridge.device))
    policy_source = "untrained_init"
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint, load_optimizer=False, map_location=str(bridge.device))
        policy_source = "checkpoint"
    policy = runner.get_inference_policy(device=str(bridge.device))

    # ---- version / provenance ----
    run_info = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None,
        "checkpoint_sha256": sha256_file(args_cli.checkpoint) if args_cli.checkpoint else None,
        "policy_source": policy_source,
        "action_mode": args_cli.action_mode,
        "protocol_path": protocol_path,
        "protocol_version": protocol.get("version"),
        "protocol_sha256": sha256_file(protocol_path),
        "git_commit": git_commit(),
        "env_seed": args_cli.seed,
        "group": args_cli.group,
        "group_applied": applied_group,
        "num_envs": bridge.num_envs,
        "scenarios": [s["id"] for s in specs],
        "episodes_per_scenario": args_cli.episodes_per_scenario,
        "episodes_per_variant": args_cli.episodes_per_variant,
        "variant_coverage_rule": (
            "per-variant quota enforced by VariantScheduler; env -> variant is a fixed slot "
            "(i % num_variants), never an episode counter that starts at 0"
        ),
        "step_dt": step_dt,
        "policy_hz": 1.0 / step_dt,
        "warmup_s": warmup_s,
        "error_frame": "body frame (+x forward, +y left, +z up), m/s and rad/s",
        "sampling": "one record per policy step (50 Hz); metrics computed on that grid",
        "data_origin": "isaac_sim_simulation",
        "disclaimer": "SIMULATION DATA ONLY - not real-robot measurements. alpha/delta calibration NOT computed by this task.",
        "early_termination_handling": (
            "terminated episodes are kept and counted in failure_rate; their pre-termination samples are used for all metrics"
        ),
        "started_at": stamp,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "protocol_used.json"), "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "agent_cfg.json"), "w", encoding="utf-8") as fh:
        json.dump(agent_cfg, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(out_dir, "env_cfg.json"), "w", encoding="utf-8") as fh:
        json.dump(env_cfg.to_dict(), fh, ensure_ascii=False, indent=2, default=str)

    print(f"[EVAL] out={out_dir} group={args_cli.group} envs={bridge.num_envs} step_dt={step_dt}", flush=True)
    print(f"[EVAL] policy_source={policy_source} scenarios={[s['id'] for s in specs]}", flush=True)

    store = StepStore(bridge.num_envs, NF, int(args_cli.max_steps_per_episode) + 64, bridge.device)
    episode_rows: list[dict] = []
    robot = u.scene["robot"]
    term_names = list(u.termination_manager.active_terms)
    action_dim = int(u.action_manager.total_action_dim)
    scenario_coverage: dict = {}

    for spec in specs:
        variants = program_variants(spec)
        duration = float(spec["duration_s"])
        dur_steps = int(round(duration / step_dt))
        # Per-variant quota (protocol supplement D-2). The previous version asked for a
        # scenario-wide total and stopped as soon as that total was reached; because every
        # environment started at ep_counter == 0, the first cohort was all variant 0 and the
        # remaining variants were never executed (TASK-029 finding).
        if args_cli.episodes_per_variant is not None:
            quota = int(args_cli.episodes_per_variant)
        else:
            quota = max(1, int(args_cli.episodes_per_scenario) // len(variants))
            print(
                f"[EVAL] {spec['id']}: --episodes_per_variant not given, derived quota={quota} from "
                f"episodes_per_scenario={args_cli.episodes_per_scenario} and {len(variants)} variant(s)",
                flush=True,
            )
        variant_of_env = assign_variants(bridge.num_envs, len(variants))
        variant_ids = torch.tensor(variant_of_env, dtype=torch.long, device=bridge.device)
        sched = VariantScheduler(variant_of_env, quota, num_variants=len(variants))
        done_rows: list[dict] = []
        ep_steps: list[torch.Tensor] = []
        ep_variant: list[int] = []
        ep_episode_id: list[int] = []
        ep_counter = torch.zeros(bridge.num_envs, dtype=torch.long, device=bridge.device)
        store.t = 0
        store.start.zero_()

        obs, _ = wrapper.reset()
        shim.set_obs(obs)
        store.start.zero_()

        t0 = time.time()
        guard = 0
        guard_limit = dur_steps * 8 + 200
        guard_hit = False
        while not sched.complete():
            guard += 1
            if guard > guard_limit:
                guard_hit = True
                print(
                    f"[WARN] {spec['id']}: step guard hit after {guard} steps with "
                    f"{sched.total_accepted}/{sched.target_total} episodes, missing {sched.missing()} "
                    "-> run marked INCOMPLETE",
                    flush=True,
                )
                break

            # per-env command from its own episode clock and its fixed variant slot
            ep_t = (u.episode_length_buf.to(torch.float32) * step_dt).unsqueeze(1)
            # map env -> variant by a fixed slot, not by an episode counter that starts at 0
            cmds = torch.zeros(bridge.num_envs, 3, device=bridge.device)
            for vi, prog in enumerate(variants):
                sel = variant_ids == vi
                if bool(sel.any()):
                    cmds[sel] = prog.at(ep_t[sel].squeeze(1))
            bridge.set_command_batch(cmds)

            # act_inference -> get_actor_obs expects the full obs dict (it selects groups itself)
            if args_cli.action_mode == "random":
                # self-test only: uniform raw actions drive the robot into the ground so the
                # termination path (and its logging) can be exercised
                actions = 8.0 * torch.rand(bridge.num_envs, action_dim, device=bridge.device) - 4.0
            else:
                # act_inference -> get_actor_obs expects the full obs dict (it selects groups itself)
                with torch.no_grad():
                    actions = policy(obs)
            obs, _, term, trunc, info = wrapper.step(actions)

            # ---- record one step for every env ----
            row = torch.zeros(bridge.num_envs, NF, device=bridge.device)
            lin_b = robot.data.root_lin_vel_b
            ang_b = robot.data.root_ang_vel_b
            rpy = _rpy_from_quat(robot.data.root_quat_w)
            row[:, I["sim_time"]] = float(u.common_step_counter) * step_dt
            row[:, I["episode_step"]] = u.episode_length_buf.to(torch.float32)
            row[:, I["cmd_vx"] : I["cmd_vx"] + 3] = cmds
            row[:, I["app_vx"] : I["app_vx"] + 3] = bridge.term.vel_command_b
            row[:, I["vx_body"]] = lin_b[:, 0]
            row[:, I["vy_body"]] = lin_b[:, 1]
            row[:, I["wz_body"]] = ang_b[:, 2]
            row[:, I["roll"] : I["roll"] + 3] = rpy
            row[:, I["pos_x"] : I["pos_x"] + 3] = robot.data.root_pos_w
            row[:, I["joint_pos_0"] : I["joint_pos_0"] + 12] = robot.data.joint_pos
            row[:, I["joint_vel_0"] : I["joint_vel_0"] + 12] = robot.data.joint_vel
            row[:, I["action_0"] : I["action_0"] + 12] = u.action_manager.action.detach()
            row[:, I["terminated"]] = term.float()
            row[:, I["truncated"]] = trunc.float()

            reason = torch.zeros(bridge.num_envs, device=bridge.device)
            if bool(u.termination_manager.get_term("base_contact").any()):
                reason[u.termination_manager.get_term("base_contact")] = 1.0
            if bool(u.termination_manager.get_term("bad_orientation").any()):
                mask = u.termination_manager.get_term("bad_orientation")
                reason[mask] = 2.0
            if bool(u.termination_manager.get_term("time_out").any()):
                mask = u.termination_manager.get_term("time_out")
                reason[(reason == 0) & mask] = 3.0
            row[:, I["reason_id"]] = reason
            store.push(row)

            # ---- close episodes ----
            reached = u.episode_length_buf >= dur_steps
            close = term | trunc | reached
            if bool(close.any()):
                ids = close.nonzero(as_tuple=False).flatten()
                for env_id in ids.tolist():
                    ep = store.episode(env_id)
                    # count the episode first: a rejected surplus episode must still be
                    # visible in the coverage report, never dropped without a trace
                    rec = sched.record(env_id)
                    if ep.shape[0] < 3 or not rec["accepted"]:
                        continue
                    # overwrite the closing reason when the scenario simply ran out
                    if float(ep[-1, I["reason_id"]]) == 0.0:
                        ep[-1, I["reason_id"]] = 4.0 if bool(reached[env_id]) else 0.0
                    m = compute_metrics(ep.detach().cpu().numpy(), spec, step_dt, warmup_s)
                    m["env_id"] = int(env_id)
                    m["episode_index"] = int(ep_counter[env_id].item())
                    m["variant_id"] = int(rec["variant"])
                    m["episode_id"] = int(rec["episode_id"])
                    m["env_seed"] = int(args_cli.seed)
                    # requested command of this episode, read back from the step log
                    m["cmd_vx_first"], m["cmd_vy_first"], m["cmd_wz_first"] = [
                        float(x) for x in ep[0, I["cmd_vx"] : I["cmd_vx"] + 3].tolist()
                    ]
                    done_rows.append(m)
                    ep_steps.append(ep.cpu())
                    ep_variant.append(int(rec["variant"]))
                    ep_episode_id.append(int(rec["episode_id"]))
                # reset the closed envs (IsaacLab already reset terminated ones)
                manual = (reached & ~term & ~trunc).nonzero(as_tuple=False).flatten()
                if manual.numel():
                    u._reset_idx(manual)
                bridge.notify_reset(ids)
                store.restart(ids)
                ep_counter[ids] += 1

        wall = time.time() - t0
        cov = sched.coverage()
        cov["step_guard_hit"] = bool(guard_hit)
        cov["guard_limit_steps"] = int(guard_limit)
        cov["wall_s"] = round(wall, 1)
        scenario_coverage[spec["id"]] = cov
        print(
            f"[EVAL] {spec['id']:12s} quota/variant={quota} accepted={cov['accepted_per_variant']} "
            f"rejected={cov['rejected_per_variant']} wall={wall:.1f}s complete={cov['complete']}",
            flush=True,
        )
        if not cov["complete"]:
            print(f"[EVAL][INCOMPLETE] {spec['id']}: missing per variant {cov['missing']}", flush=True)
        print(
            f"[EVAL] {spec['id']:12s} episodes={len(done_rows)} "
            f"failure_rate={sum(r['failed'] for r in done_rows) / max(1, len(done_rows)):.3f}",
            flush=True,
        )

        # ---- step-level npz: every per-episode metric is recomputable from here ----
        # float64 on disk: the recompute check must reproduce the logged metrics exactly,
        # and float32 rounding alone is enough to shift rmse in the 1e-4 range
        steps_cat = (
            torch.cat(ep_steps, 0).numpy().astype(np.float64) if ep_steps else np.zeros((0, NF), np.float64)
        )
        lengths = np.array([int(e.shape[0]) for e in ep_steps], dtype=np.int32)
        npz_path = os.path.join(raw_dir, f"{spec['id']}.npz")
        np.savez_compressed(
            npz_path,
            steps=steps_cat,
            lengths=lengths,
            fields=np.array(FIELDS),
            step_dt=np.array([step_dt], dtype=np.float64),
            group=np.array([args_cli.group]),
            scenario=np.array([spec["id"]]),
            variant_ids=np.array(ep_variant, dtype=np.int32),
            episode_ids=np.array(ep_episode_id, dtype=np.int32),
            note=np.array(
                [
                    "steps concatenated per episode; slice with lengths to recover each episode; "
                    "variant_ids/episode_ids are parallel to lengths"
                ]
            ),
        )

        # ---- representative replay snippets (auxiliary; metrics remain the evidence) ----
        n_snip = min(int(args_cli.replay_snippets), len(ep_steps))
        for k in range(n_snip):
            _write_snippet(snip_dir, f"{spec['id']}_ep{k:03d}", ep_steps[k].numpy(), step_dt)

        print(f"[EVAL] {spec['id']:12s} raw -> {npz_path} ({steps_cat.shape[0]} step rows)", flush=True)
        episode_rows.extend(done_rows)

    # ---- per-episode CSV ----
    csv_path = os.path.join(out_dir, "per_episode.csv")
    keys = sorted({k for r in episode_rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in episode_rows:
            w.writerow(r)

    # ---- summary ----
    # Verdict is per variant (protocol supplement D-4): a scenario passes only when every
    # variant passes and the coverage is complete. The scenario-wide aggregate is still
    # reported (key "combined") for comparison, but it never decides the verdict.
    summary = {"run_info": run_info, "scenarios": [], "coverage": scenario_coverage}
    for spec in specs:
        rows = [r for r in episode_rows if r["scenario"] == spec["id"]]
        cov = scenario_coverage.get(spec["id"], {"complete": False, "step_guard_hit": True})
        complete = bool(cov.get("complete")) and not bool(cov.get("step_guard_hit"))
        per_variant = []
        for vi in range(len(program_variants(spec))):
            vrows = [r for r in rows if r.get("variant_id") == vi]
            if not vrows:
                per_variant.append({
                    "scenario": f"{spec['id']}#v{vi}",
                    "variant_id": vi,
                    "episodes": 0,
                    "pass": False,
                    "uncovered": True,
                })
                continue
            agg_v = aggregate(vrows, {**spec, "id": f"{spec['id']}#v{vi}"}, args_cli.group)
            agg_v["variant_id"] = vi
            per_variant.append(agg_v)
        combined = aggregate(rows, spec, args_cli.group) if rows else {"scenario": spec["id"], "episodes": 0, "pass": False}
        summary["scenarios"].append({
            "scenario": spec["id"],
            "group": args_cli.group,
            "episodes": len(rows),
            "quota_per_variant": cov.get("quota_per_variant"),
            "coverage": cov,
            "variants": per_variant,
            "pass": bool(complete and all(v.get("pass") for v in per_variant)),
            "combined": combined,
            "failure_rate": combined.get("failure_rate"),
            "incomplete_reason": None if complete else ("step_guard_hit" if cov.get("step_guard_hit") else "quota_not_met"),
        })
    summary["coverage_complete"] = all(
        s["coverage"].get("complete") and not s["coverage"].get("step_guard_hit") for s in summary["scenarios"]
    )
    summary["overall_pass"] = bool(summary["coverage_complete"] and all(s["pass"] for s in summary["scenarios"]))
    summary["finished_at"] = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("[EVAL_SUMMARY] " + json.dumps(
        {
            s["scenario"]: {
                "pass": s["pass"],
                "failure_rate": None if s["failure_rate"] is None else round(s["failure_rate"], 4),
                "coverage_complete": bool(s["coverage"].get("complete")) and not bool(s["coverage"].get("step_guard_hit")),
                "per_variant_pass": [bool(v.get("pass")) for v in s["variants"]],
            }
            for s in summary["scenarios"]
        },
        ensure_ascii=False), flush=True)
    print(f"[EVAL] coverage_complete={summary['coverage_complete']} overall_pass={summary['overall_pass']}", flush=True)
    print(f"[EVAL] per_episode.csv -> {csv_path}", flush=True)
    print(f"[EVAL] summary.json  -> {os.path.join(out_dir, 'summary.json')}", flush=True)

    wrapper.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
