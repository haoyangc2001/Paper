"""CaMRA Go2 flat-ground locomotion configuration (TASK-019).

This module is an **override layer**, not a fork: it subclasses the upstream
``RobotEnvCfg`` and changes exactly the items listed in :data:`CAMRA_OVERRIDES`.
The upstream file is left untouched, so the diff between the baseline and the
CaMRA configuration is the table below -- nothing else.

Why an override layer instead of editing ``velocity_env_cfg.py``:

* the baseline stays runnable for every other robot task under this package;
* every deviation carries a written reason plus an impact statement;
* :func:`assert_overrides_applied` turns the table into a runtime assertion, so
  the configuration cannot silently drift away from the documentation.
"""

from __future__ import annotations

import os
from typing import Any

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from unitree_rl_lab.tasks.locomotion import mdp

#: Command resampling interval for CaMRA training, in seconds.
#:
#: The upstream value is a fixed 10 s, which yields only two command changes per
#: 20 s episode.  The evaluation protocol (``docs/eval_protocol.json``) contains
#: ``start_stop`` and ``combined`` scenarios that require the policy to have seen
#: command *transitions*, so the interval is shortened to 4-8 s: 3-5 switches per
#: episode.  This is still two orders of magnitude slower than the 10 Hz external
#: command interface, i.e. it is a piecewise-constant curriculum, not per-step
#: white noise on the command channel.
#:
#: Set back to ``(10.0, 10.0)`` to reproduce the upstream schedule exactly.
CAMRA_COMMAND_RESAMPLING: tuple[float, float] = (4.0, 8.0)

#: Every deviation from the upstream baseline, with reason / impact / rollback.
#: ``assert_overrides_applied`` checks the live config against this table.
CAMRA_OVERRIDES: dict[str, dict[str, Any]] = {
    "commands.base_velocity.debug_vis": {
        "baseline": True,
        "camra": False,
        "reason": (
            "With debug_vis=True the command term asks isaaclab.utils.assets to resolve "
            "arrow_x.usd on S3 to draw command arrows. The cluster has no route to "
            "aws s3, so every env construction stalls on a 300 s timeout (measured in "
            "TASK-013: gym.make took >300 s with 0% GPU utilisation)."
        ),
        "impact": "Visualisation markers only. Observations, actions, rewards and terminations are untouched.",
        "rollback": "Set debug_vis=True (only sensible on a machine with internet access and a GUI).",
    },
    "commands.base_velocity.resampling_time_range": {
        "baseline": (10.0, 10.0),
        "camra": CAMRA_COMMAND_RESAMPLING,
        "reason": (
            "20 s episode with a fixed 10 s interval gives only two command steps, which does "
            "not cover the start_stop / combined scenarios fixed in eval_protocol.json. "
            "4-8 s yields 3-5 transitions per episode while remaining far slower than the "
            "10 Hz external interface (so: piecewise-constant, not high-frequency noise)."
        ),
        "impact": "Command *schedule* only. The command domain stays (+-1.0, +-0.4, +-1.0), rewards and randomisation are unchanged.",
        "rollback": "Set CAMRA_COMMAND_RESAMPLING = (10.0, 10.0).",
    },
    "rewards.track_lin_vel_step": {
        "baseline": None,
        "camra": {"term": "track_lin_vel_step", "weight": 1.0, "std": 0.15, "step_window_s": 0.6},
        "reason": (
            "TASK-023 tuning round 1. The baseline reward set has no term sensitive to a command "
            "step: track_lin_vel_xy_exp averages over the episode, so a 0.55 s transient on a 20 s "
            "episode is ~2.5 % of the return. Measured on dev_it2999, doubling the start-up speed "
            "raises track_lin_vel_xy by +0.009 per step while action_rate_l2 costs more than that, "
            "so a slow start is optimal under the baseline reward -- which is why start_stop failed "
            "on all six checkpoints (rise_time 0.552 s vs 0.5 s). This term applies the same "
            "exponential kernel (std 0.15 instead of 0.5) only during the 0.6 s after a change."
        ),
        "impact": (
            "Reward only, and only inside the transient window (~10 % of steps at a 4-8 s "
            "resampling interval). Observations, actions, terminations, domain randomisation and "
            "the 16 baseline weights are untouched. Steady-state rmse may trade up slightly "
            "(0.05 -> ~0.06 at worst, still well inside the 0.15 threshold)."
        ),
        "rollback": "Set the term to None in __post_init__; the baseline reward set is restored exactly.",
    },
    "rewards.lin_vel_xy_at_standstill": {
        "baseline": None,
        "camra": None,
        "reason": (
            "TASK-023 tuning rounds 1-3 used this term (weight -5.0, gate narrowed 0.1 -> 0.05 m/s); "
            "round 4 REMOVES it and replaces it with rewards.stop_drift. It was firing on the right "
            "events but its whole-episode contribution was only 0.0036 against action_rate_l2's 0.1796, "
            "and because it applies whenever the command is zero it also covered the stillness just "
            "before a standing start -- the most plausible cause of the round 3 rise_time regression "
            "(0.225 s in round 1 -> 0.533 s). Round 3 measured stop_drift 0.115-0.120 m against the "
            "0.10 m threshold, i.e. the velocity form of the penalty had reached its limit."
        ),
        "impact": "Term disabled; the reward set loses one always-on penalty. Behavioural effect is carried by rewards.stop_drift.",
        "rollback": "Re-create the term in __post_init__ exactly as rounds 1-3 had it (weight -5.0, threshold 0.05) and set rewards.stop_drift to None.",
    },
    "rewards.stop_drift": {
        "baseline": None,
        "camra": {"term": "stop_drift", "weight": -5.0, "target_m": 0.1, "window_s": 0.5, "max_ratio": 4.0},
        "reason": (
            "TASK-023 tuning round 4. Frame-level analysis of dev_it2500_r3/raw/nominal/start_stop.npz "
            "(30 episodes) shows the policy reacts within 2-4 control steps and peaks at 4.1-4.5 m/s^2, "
            "yet still drifts 0.120 m: 85 % of the displacement (0.111 m) is travelled in the first "
            "0.20 s because the braking ramps up slowly, and the backward rebound (-0.07 m/s) then pulls "
            "the 2 s value back to 0.120 m. The graded metric is the planar displacement, so the term "
            "charges that displacement directly, normalised by the 0.10 m threshold, and only inside the "
            "0.5 s window that follows a falling edge (never before a start)."
        ),
        "impact": (
            "Reward only, active only in the ~2 % of steps inside a post-stop window. Observations, "
            "actions, terminations, domain randomisation and the 16 baseline weights are untouched. "
            "Watch that rmse_vx/rmse_vy/rmse_wz and the dev failure rate stay inside their thresholds; "
            "a hard-braking policy could otherwise trade steady-state tracking for stop quality."
        ),
        "rollback": "Set the term to None in __post_init__ and restore rewards.lin_vel_xy_at_standstill.",
    },
    "commands.base_velocity.stop_event_prob": {
        "baseline": 0.0,
        "camra": 0.25,
        "reason": (
            "TASK-023 tuning round 4. The training command is a uniform resample of lin_vel_x (-1, 1), "
            "so a hold of exactly zero is a near-measure-zero event, and the rel_standing_envs share "
            "stands from the first step and never brakes. 'Brake from 0.8 m/s to a full stop, then "
            "restart' -- precisely what the start_stop scenario grades -- was therefore never trained. "
            "0.25 of resamples now become a full-stop hold, i.e. 2-3 stop/restart events per 20 s "
            "episode at the 4-8 s schedule. The probability is ramped in over the first 300 "
            "control steps (= 12.5 PPO iterations at num_steps_per_env=24, = 6.0 s of simulated "
            "time; see TASK-033): enabling it from step 0 keeps a third of the batch standing while "
            "the gait is still forming, and the round 4 smoke run then sat at ~16-step episodes for "
            "60 iterations where round 3 had already recovered to 562."
        ),
        "impact": (
            "Command distribution only: ranges, heading logic, standing share and the curriculum are "
            "unchanged, and the zero command is already inside the (+-1.0, +-0.4, +-1.0) domain, so the "
            "eval protocol still describes the same task. The play/eval config sets the probability back "
            "to 0.0 because the evaluation feeds its own command program. Ramp length is 300 control "
            "steps (stop_event_ramp_iters), i.e. only the first ~6 s of simulated experience differ from "
            "round 3, not the first 300 iterations. The value was deliberately left unchanged so the "
            "delivered model keeps the training behaviour it was actually trained with."
        ),
        "rollback": "Set commands.base_velocity.stop_event_prob = 0.0 (upstream behaviour exactly).",
    },
}

#: Items deliberately **not** changed, recorded so the omission is auditable.
CAMRA_UNCHANGED: dict[str, str] = {
    "scene.num_envs": (
        "4096 upstream; TASK-013 measured ~8.1 GB VRAM at 4096 envs on an L40 (46 GB), "
        "so no down-scaling is needed. Highest transition throughput of the four tested sizes."
    ),
    "scene.terrain": (
        "Only the 'flat' sub-terrain is enabled upstream (all other sub_terrains are commented "
        "out), so the task is already flat-ground only; nothing to restrict."
    ),
    "commands.base_velocity.limit_ranges": (
        "(-1.0, 1.0) / (-0.4, 0.4) / (-1.0, 1.0) kept per the user decision on 2026-10-31."
    ),
    "rewards": (
        "The 16 baseline reward terms and their weights are kept identical to the baseline "
        "(track_lin_vel_xy 1.5, track_ang_vel_z 0.75, ...). Two terms are ADDED by tuning rounds 1-4, "
        "not modified: see the rewards.track_lin_vel_step and rewards.stop_drift entries of "
        "CAMRA_OVERRIDES. rewards.lin_vel_xy_at_standstill (rounds 1-3) is removed in round 4."
    ),
    "events": "All domain-randomisation terms kept identical (friction 0.3-1.2, added mass -1..3 kg, push 0.5 m/s every 5-10 s).",
    "terminations": (
        "Termination set kept identical. The base-height termination (<0.18 m) and the command "
        "delay randomisation (0-40 ms) present in the 1020 draft are still absent; adding them "
        "would change the failure-rate definition fixed in eval_protocol.json, so they remain "
        "explicit non-goals (see handoff note section 17.3)."
    ),
    "sim.dt / decimation": "0.005 s / 4 -> 200 Hz physics, 50 Hz policy. Matches the contract.",
}


#: Diagnostic A/B switch (TASK-036 follow-up). A 16-point velocity sweep found a
#: ~0.2 m/s take-off deadband: the policy does not move at all for small commands.
#: Modes: "no-standing" removes both knobs, "no-standing-envs" removes only the
#: always-standing share and keeps the stop events that train braking,
#: "stop-low" thins the stop events to 10% and keeps everything else delivered,
#: "track-boost" doubles the tracking reward weight and keeps the sampling delivered. Unset means the delivered configuration is used
#: byte for byte - this exists only so an experiment does not need a code fork.
CMD_SAMPLING_ENV = "CAMRA_EXP_CMD_SAMPLING"
#: Override keys the experiment is allowed to move away from CAMRA_OVERRIDES.
EXPERIMENT_KEYS = {
    "commands.base_velocity.stop_event_prob",
    "rewards.track_lin_vel_step",
}


def experimental_cmd_sampling() -> str:
    """Name of the active command-sampling experiment, or "" for none."""
    return os.environ.get(CMD_SAMPLING_ENV, "").strip()


@configclass
class CaMRAGo2VelocityEnvCfg(RobotEnvCfg):
    """Upstream Go2 velocity task with the CaMRA overrides applied."""

    def __post_init__(self):
        super().__post_init__()
        # override 1: never touch the S3-hosted command-visualisation USD
        self.commands.base_velocity.debug_vis = CAMRA_OVERRIDES["commands.base_velocity.debug_vis"]["camra"]
        # override 2: denser command schedule (piecewise constant, 4-8 s holds)
        self.commands.base_velocity.resampling_time_range = CAMRA_COMMAND_RESAMPLING
        # override 3: 25 % of the holds become a full stop, so braking is trained
        self.commands.base_velocity.stop_event_prob = CAMRA_OVERRIDES["commands.base_velocity.stop_event_prob"]["camra"]
        # ... ramped in over the first 300 CONTROL STEPS (= 12.5 PPO iterations at
        # num_steps_per_env=24, = 6.0 s simulated), not 300 iterations (TASK-033)
        self.commands.base_velocity.stop_event_ramp_iters = 300
        # override 4 / 5: two reward terms that are sensitive to a command edge
        self.rewards.track_lin_vel_step = RewTerm(
            func=mdp.track_lin_vel_xy_step_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.15, "step_window_s": 0.6},
        )
        self.rewards.lin_vel_xy_at_standstill = None  # removed in round 4
        self.rewards.stop_drift = RewTerm(
            func=mdp.stop_drift_l2,
            weight=-5.0,
            params={"command_name": "base_velocity", "target_m": 0.10, "window_s": 0.5, "max_ratio": 4.0},
        )
        # experimental override last, so it wins over everything above and cannot
        # be silently shadowed by a later assignment
        _exp = experimental_cmd_sampling()
        if _exp == "no-standing":
            self.commands.base_velocity.rel_standing_envs = 0.0
            self.commands.base_velocity.stop_event_prob = 0.0
        elif _exp == "no-standing-envs":
            # only the always-standing share. stop_event_prob stays at the delivered
            # 0.25 because that is the knob that trains braking (chapter 44)
            self.commands.base_velocity.rel_standing_envs = 0.0
        elif _exp == "stop-low":
            # option 1: thin the stop events to 10% instead of 25%. Standing share
            # stays at the delivered 0.1 - chapter 45 showed it is not the deadband
            self.commands.base_velocity.stop_event_prob = 0.10
        elif _exp == "track-boost":
            # option 2: sampling stays delivered; the deadband is a reward-balance
            # problem, so double the tracking term to outbid the action costs of
            # walking slowly. One knob only, so behaviour changes are attributable
            self.rewards.track_lin_vel_step.weight = 2.0
        elif _exp:
            raise ValueError(f"{CMD_SAMPLING_ENV}={_exp!r} is not a known experiment")


@configclass
class CaMRAGo2VelocityPlayEnvCfg(CaMRAGo2VelocityEnvCfg):
    """Evaluation/play variant: fewer envs, full command range, no curriculum warm-up."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        # play with the full command domain immediately instead of the curriculum start
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # the evaluation drives its own command program: no injected stop holds
        self.commands.base_velocity.stop_event_prob = 0.0


def assert_overrides_applied(cfg: RobotEnvCfg) -> dict[str, Any]:
    """Check a live config against :data:`CAMRA_OVERRIDES`.

    Returns a report dict; raises ``AssertionError`` on any mismatch so a silent
    drift between code and documentation cannot survive a training start.
    """
    term = cfg.commands.base_velocity
    actual = {
        "commands.base_velocity.debug_vis": bool(term.debug_vis),
        "commands.base_velocity.resampling_time_range": tuple(term.resampling_time_range),
        "commands.base_velocity.stop_event_prob": float(term.stop_event_prob),
    }
    for _name in ("track_lin_vel_step", "stop_drift", "lin_vel_xy_at_standstill"):
        _term = getattr(cfg.rewards, _name, None)
        actual["rewards." + _name] = (
            None
            if _term is None
            else {
                "term": _name,
                "weight": float(_term.weight),
                **{k: float(v) for k, v in (_term.params or {}).items() if isinstance(v, (int, float))},
            }
        )
    report: dict[str, Any] = {}
    _exp = experimental_cmd_sampling()
    for key, spec in CAMRA_OVERRIDES.items():
        want = spec["camra"]
        got = actual[key]
        ok = got == (tuple(want) if isinstance(want, tuple) else want)
        entry = {"expected": want, "actual": got, "pass": ok}
        if not ok and _exp and key in EXPERIMENT_KEYS:
            # deliberate: record it so the deviation stays auditable, but do not
            # fail - the experiment is the reason the value differs
            entry["experimental_override"] = _exp
        report[key] = entry
        if not ok and "experimental_override" not in entry:
            raise AssertionError(f"override '{key}' not applied: expected {want}, got {got}")
    report["num_envs"] = cfg.scene.num_envs
    report["epoch"] = "camra_override_layer"
    return report
