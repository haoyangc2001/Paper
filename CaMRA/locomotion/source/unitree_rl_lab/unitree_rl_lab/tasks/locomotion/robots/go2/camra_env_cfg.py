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
        "camra": {"term": "lin_vel_xy_at_standstill", "weight": -5.0, "threshold": 0.1},
        "reason": (
            "TASK-023 tuning round 1. The baseline base_linear_velocity penalty is lin_vel_z_l2, "
            "i.e. the vertical axis only; nothing in the baseline discourages horizontal coasting "
            "after a stop command, and stop_drift_m failed on all six checkpoints (0.129 m vs "
            "0.10 m, with 85 % of it travelled in the first 0.20 s). "
            "Round 2 raises the weight to -5.0: round 1 moved stop_drift only 0.129 -> 0.125 m, "
            "so the term was firing on the right events but was far too weak against action_rate_l2 "
            "(0.0012 against 0.1826)."
        ),
        "impact": (
            "Reward only, active while the command is below 0.1 m/s in the xy plane (10 % of envs "
            "stand by rel_standing_envs, plus every stop transient). May make standing stiffer; "
            "feet_air_time and action_rate still bound the resulting behaviour."
        ),
        "rollback": "Set the term to None in __post_init__; the baseline reward set is restored exactly.",
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
        "(track_lin_vel_xy 1.5, track_ang_vel_z 0.75, ...). Two terms are ADDED by tuning round 1, "
        "not modified: see the rewards.track_lin_vel_step and rewards.lin_vel_xy_at_standstill "
        "entries of CAMRA_OVERRIDES."
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


@configclass
class CaMRAGo2VelocityEnvCfg(RobotEnvCfg):
    """Upstream Go2 velocity task with the two CaMRA overrides applied."""

    def __post_init__(self):
        super().__post_init__()
        # override 1: never touch the S3-hosted command-visualisation USD
        self.commands.base_velocity.debug_vis = CAMRA_OVERRIDES["commands.base_velocity.debug_vis"]["camra"]
        # override 2: denser command schedule (piecewise constant, 4-8 s holds)
        self.commands.base_velocity.resampling_time_range = CAMRA_COMMAND_RESAMPLING
        # override 3 / 4: two reward terms that are sensitive to a command step
        self.rewards.track_lin_vel_step = RewTerm(
            func=mdp.track_lin_vel_xy_step_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.15, "step_window_s": 0.6},
        )
        self.rewards.lin_vel_xy_at_standstill = RewTerm(
            func=mdp.lin_vel_xy_at_standstill_l2,
            weight=-5.0,
            params={"command_name": "base_velocity", "threshold": 0.1},
        )


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


def assert_overrides_applied(cfg: RobotEnvCfg) -> dict[str, Any]:
    """Check a live config against :data:`CAMRA_OVERRIDES`.

    Returns a report dict; raises ``AssertionError`` on any mismatch so a silent
    drift between code and documentation cannot survive a training start.
    """
    term = cfg.commands.base_velocity
    actual = {
        "commands.base_velocity.debug_vis": bool(term.debug_vis),
        "commands.base_velocity.resampling_time_range": tuple(term.resampling_time_range),
    }
    for _name in ("track_lin_vel_step", "lin_vel_xy_at_standstill"):
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
    for key, spec in CAMRA_OVERRIDES.items():
        want = spec["camra"]
        got = actual[key]
        ok = got == (tuple(want) if isinstance(want, tuple) else want)
        report[key] = {"expected": want, "actual": got, "pass": ok}
        if not ok:
            raise AssertionError(f"override '{key}' not applied: expected {want}, got {got}")
    report["num_envs"] = cfg.scene.num_envs
    report["epoch"] = "camra_override_layer"
    return report
