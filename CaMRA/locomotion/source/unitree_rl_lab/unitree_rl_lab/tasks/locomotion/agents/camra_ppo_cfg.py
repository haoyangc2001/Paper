"""CaMRA Go2 PPO runner configuration (TASK-019).

Baseline: ``BasePPORunnerCfg`` (upstream).  Only the fields listed in
:data:`CAMRA_AGENT_OVERRIDES` change; the network shape, the PPO hyper-parameters
and the observation normalisation behaviour are inherited untouched so that any
difference in training outcome is attributable to a documented decision.
"""

from __future__ import annotations

from typing import Any

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg

#: Training iteration ceiling, fixed before training by the evaluation protocol.
CAMRA_MAX_ITERATIONS = 3000

#: Checkpoint cadence in iterations.
CAMRA_SAVE_INTERVAL = 100

#: Log root name; ``train.py`` writes to ``logs/rsl_rl/<experiment_name>``.
CAMRA_EXPERIMENT_NAME = "camra_go2_locomotion"

#: Every deviation from ``BasePPORunnerCfg``, with reason and rollback.
CAMRA_AGENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "max_iterations": {
        "baseline": 50000,
        "camra": CAMRA_MAX_ITERATIONS,
        "reason": (
            "TASK-013 measured 27,970 transitions/s at 4096 envs, so one iteration (4096 x 24 "
            "transitions) costs ~3.5 s of collection; 50000 iterations would need ~48.8 h of "
            "collection per seed and cannot finish on this machine. 3000 iterations is the "
            "ceiling fixed in eval_protocol.json (6 h wall clock per seed, early stop after "
            "500 non-improving evaluations)."
        ),
        "impact": "Shorter schedule. Learning-rate schedule is 'adaptive', so the policy adapts to the budget instead of following a fixed decay.",
        "rollback": "Set max_iterations back to 50000 (only advisable with a dedicated multi-day allocation).",
    },
    "experiment_name": {
        "baseline": "",
        "camra": CAMRA_EXPERIMENT_NAME,
        "reason": (
            "The baseline leaves the name empty, which makes train.py fall back to the task name "
            "and mixes robots/experiments in one log directory. A fixed name keeps the three "
            "seeds and all evaluation artefacts under one auditable root."
        ),
        "impact": "Log path only: logs/rsl_rl/camra_go2_locomotion/.",
        "rollback": "Set back to an empty string.",
    },
    "seed": {
        "baseline": None,
        "camra": 0,
        "reason": (
            "eval_protocol.json fixes the training seeds to {0, 1, 2}. Defaulting to 0 makes the "
            "first run reproducible without extra flags; the other two seeds are passed with "
            "--seed 1 / --seed 2, which overrides this value."
        ),
        "impact": "Reproducibility only; --seed on the command line still wins.",
        "rollback": "Pass --seed explicitly.",
    },
}


@configclass
class CaMRAGo2PPORunnerCfg(BasePPORunnerCfg):
    """PPO runner for the CaMRA Go2 velocity task."""

    seed: int = 0
    max_iterations: int = CAMRA_MAX_ITERATIONS
    save_interval: int = CAMRA_SAVE_INTERVAL
    experiment_name: str = CAMRA_EXPERIMENT_NAME
