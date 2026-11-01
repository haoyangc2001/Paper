from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#: A command range may not grow more often than this, in environment steps, so the
#: curriculum cannot race ahead of the policy.
_CMD_LEVEL_MIN_INTERVAL_STEPS = 1000
#: ...and it must grow at least this often. The deadline is what makes the
#: curriculum unable to stall: without it a policy that is merely *good enough for
#: the current, still easy command range* freezes the task forever.
_CMD_LEVEL_MAX_INTERVAL_STEPS = 1500
#: Growth applied to each bound of the command range per bump.
_CMD_LEVEL_DELTA = 0.1


def _bump(env: ManagerBasedRLEnv, ranges, limit_ranges, fields: Sequence[str]) -> None:
    """Widen ``ranges`` by one step, clamped to ``limit_ranges``."""
    delta = torch.tensor([-_CMD_LEVEL_DELTA, _CMD_LEVEL_DELTA], device=env.device)
    for name in fields:
        current = torch.tensor(getattr(ranges, name), device=env.device)
        lo, hi = getattr(limit_ranges, name)
        widened = torch.clamp(current + delta, lo, hi)
        setattr(ranges, name, widened.tolist())


def _level_should_bump(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    key: str,
) -> bool:
    """Return True when the command range may (or must) grow at this step.

    Upstream only grows the range when ``common_step_counter % max_episode_length
    == 0`` *and* the mean tracking reward of the environments that just reset
    exceeds ``0.8 * weight``. Both halves can fail together and the curriculum then
    stalls for good.

    Measured on TASK-023 round 2: ``Curriculum/lin_vel_cmd_levels`` stayed at 0.1
    for 981 iterations (24 % of the run at a time when rounds 0 and 1 had already
    advanced 9 and 8 times) while the mean tracking reward sat at 1.46 of a 1.5
    ceiling, i.e. the reward test was "happy" and the range never grew. The task
    stayed at +-0.1 m/s on flat ground, so the policy never saw the speeds the
    evaluation commands.

    So: keep the reward test as the fast path, rate-limit it, and add a deadline
    that guarantees progress. The deadline is the only behavioural difference from
    upstream in the normal case -- round 0 and 1 would have advanced on schedule.
    """
    step = int(env.common_step_counter)
    last = getattr(env, key, None)
    if last is None:
        setattr(env, key, step)
        return False

    elapsed = step - last
    if elapsed < _CMD_LEVEL_MIN_INTERVAL_STEPS:
        return False

    reward_ok = False
    if len(env_ids) > 0:
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        reward = reward / env.max_episode_length_s
        reward_ok = bool(torch.isfinite(reward) and reward > reward_term.weight * 0.8)

    if not reward_ok and elapsed < _CMD_LEVEL_MAX_INTERVAL_STEPS:
        return False

    if not reward_ok:
        print(
            f"[curriculum] {reward_term_name}: no reward-based bump within "
            f"{elapsed} steps, widening the command range on the deadline",
            flush=True,
        )
    setattr(env, key, step)
    return True


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    if _level_should_bump(env, env_ids, reward_term_name, "_camra_lin_vel_cmd_level_step"):
        _bump(env, ranges, command_term.cfg.limit_ranges, ("lin_vel_x", "lin_vel_y"))
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    if _level_should_bump(env, env_ids, reward_term_name, "_camra_ang_vel_cmd_level_step"):
        _bump(env, ranges, command_term.cfg.limit_ranges, ("ang_vel_z",))
    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
