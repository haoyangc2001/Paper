"""Reload the exported controller in a fresh process and close the loop (TASK-026).

This script deliberately does **not** touch the trainer for the ``jit`` and ``onnx``
backends: no rsl-rl runner, no ActorCritic, no checkpoint. It rebuilds inference
from the candidate bundle only -- the exported graph plus the contract recorded in
``params/export_manifest.json`` -- and feeds the action straight into the Gym env,
which applies the documented ``q_target = default_joint_pos + scale * a`` rule.

``--backend trainer`` exists only as the reference arm: it loads the checkpoint
through rsl-rl the way the evaluation entry point does, so the two can be compared
on identical scenarios and seeds.

The command interface is the TASK-018 one (``ExternalCommandWrapper`` with a 10 Hz
zero-order-hold program), i.e. the same path the deployment uses, rather than
writing into the command term from the outside.

Reported per scenario: tracking RMSE against the applied command, attitude, failure
count, and for start_stop the displacement after the stop edge. Inference cost is
measured per call but is labelled as compute cost on this GPU -- headless Isaac runs
much faster than wall clock, so nothing here is evidence of real-time capability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Closed-loop rollout of an exported Go2 policy.")
parser.add_argument("--task", type=str, default="CaMRA-Go2-Velocity")
parser.add_argument("--bundle", required=True, help="Candidate bundle with exported/ and params/.")
parser.add_argument("--backend", required=True, choices=["jit", "onnx", "trainer"])
parser.add_argument("--checkpoint", default=None, help="Only for --backend trainer.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--scenarios", type=str, default="stand,start_stop,turn,ramp")
parser.add_argument("--warmup_s", type=float, default=1.0)
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
    ExternalCommandWrapper,
    RampProgram,
    StepProgram,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

SCENARIOS = {
    "stand": lambda: (StepProgram([{"duration": 8.0, "vel": [0.0, 0.0, 0.0]}]), 8.0),
    "start_stop": lambda: (
        StepProgram([{"duration": 4.0, "vel": [0.8, 0.0, 0.0]}, {"duration": 6.0, "vel": [0.0, 0.0, 0.0]}]),
        10.0,
    ),
    "turn": lambda: (StepProgram([{"duration": 8.0, "vel": [0.0, 0.0, 0.6]}]), 8.0),
    "ramp": lambda: (RampProgram([0.1, 0.0, 0.0], [0.9, 0.0, 0.0], duration=8.0), 8.0),
}


class OnnxPolicy:
    """Thin wrapper so ONNX looks like the other backends (single obs in, action out)."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.batch_dim = self.sess.get_inputs()[0].shape[0]
        self.output_dim = int(self.sess.get_outputs()[0].shape[-1])

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.detach().cpu().numpy().astype(np.float32)
        if isinstance(self.batch_dim, int) and self.batch_dim > 0 and self.batch_dim != x.shape[0]:
            out = np.concatenate([self.sess.run(None, {self.input_name: x[i : i + 1]})[0] for i in range(x.shape[0])])
        else:
            out = self.sess.run(None, {self.input_name: x})[0]
        return torch.from_numpy(out).to(obs.device)


def build_policy(bundle: str, backend: str, checkpoint: str | None, device: str, unwrapped=None):
    """Return (policy_callable, meta). Only the trainer arm imports rsl-rl."""
    man = json.load(open(os.path.join(bundle, "params", "export_manifest.json"), encoding="utf-8"))
    if backend == "jit":
        jit = torch.jit.load(os.path.join(bundle, "exported", "policy.pt"), map_location=device).eval()
        return (lambda obs: jit(obs)), {"kind": "torchscript", "path": "exported/policy.pt",
                                        "sha256": man["artifacts"]["exported/policy.pt"]["sha256"]}
    if backend == "onnx":
        pol = OnnxPolicy(os.path.join(bundle, "exported", "policy.onnx"))
        return pol, {"kind": "onnxruntime", "path": "exported/policy.onnx",
                     "sha256": man["artifacts"]["exported/policy.onnx"]["sha256"],
                     "batch_mode": "looped-batch1" if pol.batch_dim == 1 else "single-call"}
    # reference arm: trainer-style load, used only for the comparison
    import yaml
    from rsl_rl.runners import OnPolicyRunner

    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    agent_cfg = yaml.safe_load(open(os.path.join(run_dir, "params", "agent.yaml"), encoding="utf-8"))
    if "policy" not in (agent_cfg.get("obs_groups") or {}):
        agent_cfg["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}

    class _Shim:
        """Minimal VecEnv surface: OnPolicyRunner reads the observation shapes once."""

        def __init__(self, num_envs, dev, num_actions, max_len, u):
            self.num_envs, self.device, self.num_actions = num_envs, dev, num_actions
            self.max_episode_length, self.unwrapped = max_len, u

        def get_observations(self):
            return self.unwrapped.observation_manager.compute()

    shim = _Shim(args_cli.num_envs, device, 12, 1000, unwrapped)
    runner = OnPolicyRunner(shim, agent_cfg, log_dir=None, device=device)
    runner.load(checkpoint, load_optimizer=False, map_location=device)
    fn = runner.get_inference_policy(device=device)
    # rsl-rl >=3 indexes observations by group name, so adapt the flat tensor used here
    return (lambda obs: fn({"policy": obs})), {"kind": "rsl_rl_checkpoint", "path": checkpoint}


def main() -> int:
    device = args_cli.device
    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(args_cli.bundle, "params", "export_manifest.json"), encoding="utf-8"))
    obs_dim_exp = int(man["observation_contract"]["total_dim"])

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=True,
                            entry_point_key="env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False   # same deterministic obs as the export path
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges

    raw = gym.make(args_cli.task, cfg=env_cfg)
    env = ExternalCommandWrapper(raw, limits=DEFAULT_LIMITS, enforce_limits=True, on_reset="zero")
    u = raw.unwrapped
    robot = u.scene["robot"]
    step_dt = float(u.step_dt)

    if args_cli.backend == "trainer" and not args_cli.checkpoint:
        raise SystemExit("--backend trainer needs --checkpoint")
    policy, policy_meta = build_policy(
        args_cli.bundle, args_cli.backend, args_cli.checkpoint, device, unwrapped=u
    )
    # runtime proof that the exported arms carry no trainer dependency: a static grep can
    # be defeated by a transitive import, the module table cannot
    trainer_mods = sorted(m for m in sys.modules if m == "rsl_rl" or m.startswith("rsl_rl."))

    # ---- interface checks that do not depend on the policy at all ----
    default_pos = robot.data.default_joint_pos[0].detach().clone()
    checks: dict[str, object] = {}
    checks["no_trainer_dependency"] = {
        "rsl_rl_modules_in_sys_modules": trainer_mods,
        "pass": (len(trainer_mods) == 0) if args_cli.backend != "trainer" else None,
        "informational": args_cli.backend == "trainer",
        "note": "the trainer reference arm is expected to load rsl-rl; the exported arms must not",
    }
    obs0, _ = env.reset()
    obs = obs0["policy"]
    checks["obs_dim"] = {"observed": int(obs.shape[1]), "contract": obs_dim_exp,
                         "pass": int(obs.shape[1]) == obs_dim_exp}
    # Purity of the observation pipeline, plus the one-tick artefact the wrapper creates.
    # ExternalCommandWrapper.reset() installs the program command *after* the inner env has
    # already computed its observation, so the tensor reset() hands back still carries the
    # freshly sampled command while a re-compute already sees the program command. Only the
    # velocity_commands slice may therefore differ on that tick; anything else differing
    # would be a genuine pipeline inconsistency, which is what this gate checks.
    obs_re = u.observation_manager.compute()["policy"]
    d = (obs_re - obs0["policy"]).abs()
    start, cmd_slice = 0, slice(0, 0)
    for term in man["observation_contract"]["terms"]:
        if term["name"] == "velocity_commands":
            cmd_slice = slice(start, start + int(term["dim"]))
        start += int(term["dim"])
    mask = torch.ones(d.shape[1], dtype=torch.bool, device=d.device)
    mask[cmd_slice] = False
    checks["obs_pipeline_reproducible"] = {
        "max_abs_error_outside_command_slice": float(d[:, mask].max()),
        "max_abs_error_command_slice": float(d[:, cmd_slice].max()),
        "max_abs_error_total": float(d.max()),
        "command_slice": [cmd_slice.start, cmd_slice.stop],
        "pass": float(d[:, mask].max()) == 0.0,
        "note": "observation_manager.compute() re-run on the same state; the command dims are "
                "excluded because the wrapper writes the program command after the inner env "
                "computed its observation (every later tick is checked per step)",
    }
    probe_action = torch.zeros(int(u.action_manager.total_action_dim), device=device).unsqueeze(0).repeat(
        int(args_cli.num_envs), 1
    )
    probe_action[:, 0] = 0.5  # a value the scale/offset rule turns into a visible joint offset
    u.action_manager.process_action(probe_action)
    # processed_actions lives on the term, not on the manager
    act_term = u.action_manager.get_term(u.action_manager.active_terms[0])
    processed = act_term.processed_actions
    expected = default_pos.unsqueeze(0) + 0.25 * probe_action
    scale_err = float((processed - expected).abs().max())
    checks["action_scale_offset"] = {
        "max_abs_error_vs_manifest_rule": scale_err,
        "pass": scale_err < 1e-5,
        "note": "processed joint target compared with default_joint_pos + 0.25 * a",
    }
    # Two consecutive resets are *expected* to differ: the reset event re-samples the
    # initial state (domain randomisation), so equality would be the surprising result.
    # The number is recorded anyway because it must come out bit-identical in a fresh
    # process -- that is the real evidence that seed 100 reproduces the same draw stream,
    # and it is cross-checked between the three backends in the aggregation step.
    obs_r2, _ = env.reset()
    reset_err = float((obs_r2["policy"] - obs0["policy"]).abs().max())
    checks["reset_sampling_delta"] = {
        "max_abs_error": reset_err,
        "pass": None,
        "informational": True,
        "note": "reset #2 vs reset #1; inequality is by design, cross-backend equality is the signal",
    }

    results = {}
    for name in args_cli.scenarios.split(","):
        name = name.strip()
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name}")
        program, duration = SCENARIOS[name]()
        steps = int(round(duration / step_dt))
        env.set_program(program)
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]

        cmd_hist, vx_hist, vy_hist, wz_hist, roll_hist, pos_hist, act_hist = [], [], [], [], [], [], []
        lat_ms, falls, warm = [], 0, int(round(args_cli.warmup_s / step_dt))
        env_resets = 0
        layout_max = 0.0
        # torch.no_grad(), deliberately not inference_mode(): inference_mode propagates
        # through every tensor the env writes during step()/auto-reset, so the simulator
        # buffers become inference tensors and the *next* in-place write outside the mode
        # raises "Inplace update to inference tensor outside InferenceMode is not allowed"
        # (observed at the scenario-to-scenario reset). That is a simulator bookkeeping
        # artefact, not a policy defect. Autograd is off in both cases and the timing
        # basis below is unchanged.
        with torch.no_grad():
            for t in range(steps):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                action = policy(obs)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                lat_ms.append((time.perf_counter() - t0) * 1e3)
                # obs was built from the state we are about to step, so the documented
                # layout (first 3 dims = base_ang_vel * 0.2) is checked at every single
                # control tick, not only at the reset state where ang_vel is zeroed
                layout_max = max(layout_max, float(
                    (obs[:, :3] - robot.data.root_ang_vel_b[:, :3] * 0.2).abs().max()
                ))
                obs_dict, _rew, term, trunc, info = env.step(action)
                # a mid-scenario reset (fall or timeout) would silently re-initialise the
                # robot and pollute the tracking curves, so it is counted and reported
                env_resets += int((u.episode_length_buf == 0).sum().item())
                rec = info["external_command"]
                cmd_hist.append(rec.applied.detach().clone())
                vx_hist.append(robot.data.root_lin_vel_b[:, 0].detach().clone())
                vy_hist.append(robot.data.root_lin_vel_b[:, 1].detach().clone())
                wz_hist.append(robot.data.root_ang_vel_b[:, 2].detach().clone())
                q = robot.data.root_quat_w
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                roll_hist.append(roll.detach().clone())
                pos_hist.append(robot.data.root_pos_w[:, :2].detach().clone())
                act_hist.append(action.detach().clone())
                falls += int(rec.terminated.sum().item())
                obs = obs_dict["policy"]

        cmd = torch.stack(cmd_hist).cpu().numpy()      # (T, N, 3)
        vx = torch.stack(vx_hist).cpu().numpy()
        vy = torch.stack(vy_hist).cpu().numpy()
        wz = torch.stack(wz_hist).cpu().numpy()
        roll = torch.stack(roll_hist).cpu().numpy()
        pos = torch.stack(pos_hist).cpu().numpy()
        acts = torch.stack(act_hist).cpu().numpy()

        track = slice(warm, None)
        err = {
            "rmse_vx": float(np.sqrt(np.mean((vx[track] - cmd[track, :, 0]) ** 2))),
            "rmse_vy": float(np.sqrt(np.mean((vy[track] - cmd[track, :, 1]) ** 2))),
            "rmse_wz": float(np.sqrt(np.mean((wz[track] - cmd[track, :, 2]) ** 2))),
        }
        res = {
            "program": program.describe(),
            "steps": steps,
            "sim_seconds": steps * step_dt,
            "tracking": err,
            "max_abs_roll_deg": float(np.degrees(np.abs(roll).max())),
            "failures": falls,
            "env_resets": env_resets,
            "obs_layout_max_err_ang_vel": layout_max,
            "obs_layout_pass": layout_max < 1e-5,
            "action_abs_mean": float(np.abs(acts).mean()),
            "action_abs_max": float(np.abs(acts).max()),
            "latency_ms": {
                "median": float(np.median(lat_ms)),
                "p95": float(np.quantile(lat_ms, 0.95)),
                "max": float(np.max(lat_ms)),
                "basis": "policy call only, GPU sync bracketed; headless sim, not a real-time claim",
            },
        }
        if name == "start_stop":
            edge = int(round(4.0 / step_dt))  # command goes 0.8 -> 0 here
            p0 = pos[edge]
            disp = np.linalg.norm(pos[edge:] - p0, axis=2)   # (T-edge, N)
            after2 = int(round(2.0 / step_dt))
            res["stop"] = {
                "drift_at_2s_m": float(disp[min(after2, disp.shape[0] - 1)].mean()),
                "drift_max_m": float(disp.max(axis=0).mean()),
                "time_to_0p1_mps_s": float(
                    np.mean([
                        np.argmax(np.abs(vx[edge:, i]) < 0.1) * step_dt
                        if np.any(np.abs(vx[edge:, i]) < 0.1) else np.nan
                        for i in range(vx.shape[1])
                    ])
                ),
            }
        results[name] = res

        np.savez_compressed(
            os.path.join(out_dir, f"trace_{args_cli.backend}_{name}.npz"),
            cmd=cmd, vx=vx, vy=vy, wz=wz, roll=roll, pos=pos, actions=acts,
        )
        print(f"[ROLLOUT] {args_cli.backend} {name}: {json.dumps(res['tracking'])} "
              f"fails={falls} resets={env_resets} roll_max={res['max_abs_roll_deg']:.2f} "
              f"obs_layout_err={layout_max:.1e}", flush=True)

    report = {
        "backend": args_cli.backend,
        "policy": policy_meta,
        "bundle": os.path.abspath(args_cli.bundle),
        "manifest_sha256": {
            "policy.pt": man["artifacts"]["exported/policy.pt"]["sha256"],
            "policy.onnx": man["artifacts"]["exported/policy.onnx"]["sha256"],
        },
        "num_envs": int(args_cli.num_envs),
        "seed": int(args_cli.seed),
        "step_dt": step_dt,
        "device": device,
        "grad_mode": "torch.no_grad()",
        "interface_checks": checks,
        "scenarios": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = os.path.join(out_dir, f"rollout_{args_cli.backend}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[ROLLOUT] wrote {path}", flush=True)
    print(f"[ROLLOUT] interface checks: "
          f"{ {k: v['pass'] for k, v in checks.items()} }", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
