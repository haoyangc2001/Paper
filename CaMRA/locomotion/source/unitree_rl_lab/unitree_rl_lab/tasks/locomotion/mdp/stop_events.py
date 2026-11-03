"""Pure-torch helpers shared by the stop-event command and the stop shaping term.

They live here rather than inside ``rewards.py`` / ``commands.py`` so the state
machine can be unit-tested on CPU without importing isaaclab (which needs
``omni``, i.e. a running simulator).

Background (TASK-023 tuning round 4): the ``start_stop`` scenario asks for a
hard 0.8 m/s -> 0 stop, but the training command is resampled uniformly from
+-1.0 m/s and therefore almost never lands on exactly zero, while the
``rel_standing_envs`` share stands from the first second and never brakes.  The
policy is only ever taught to *track* a velocity, never to *terminate* one, so
it coasts for 0.12-0.13 m where the protocol allows 0.10 m.
:func:`stop_event_mask` injects real stop holds into the command resampling;
:func:`advance_stop_window` turns the resulting falling edge into the window in
which the shaping term charges for the displacement that actually travels.
"""

from __future__ import annotations

import torch


def stop_event_mask(uniform_sample: torch.Tensor, prob: float) -> torch.Tensor:
    """Boolean mask of the envs whose next command hold must be a full stop.

    Args:
        uniform_sample: (N,) uniform samples in [0, 1).
        prob: per-resample probability of a full stop hold; <= 0 disables it.

    Returns:
        (N,) bool tensor.
    """
    if prob <= 0.0:
        return torch.zeros_like(uniform_sample, dtype=torch.bool)
    return uniform_sample <= prob


def ramped_probability(p_max: float, iteration: int, ramp_iters: int) -> float:
    """Linear warm-up of the stop-event probability.

    Enabling stops from the very first control step keeps a third of the batch
    standing while the gait itself is still being learned, which stalls the early
    phase: measured on the round 4 smoke run, episode length sat at ~16 steps for
    60 iterations where the round 3 run (no stop events) recovered to 562 by
    iteration 90.

    **Units (TASK-033, verified against the installed IsaacLab):** the second
    argument is ``env.common_step_counter``, which ``ManagerBasedRLEnv.step``
    increments **once per control step**, not per PPO iteration
    (``isaaclab/envs/manager_based_rl_env.py``, "total step (common for all envs)").
    With ``decimation=4`` and ``sim.dt=0.005`` a control step is 0.02 s and one PPO
    iteration is ``num_steps_per_env = 24`` control steps. So the configured 300 is
    **300 control steps = 12.5 iterations = 6.0 s of simulated experience**, not
    300 iterations. The earlier wording ("ramp in over ~300 iterations") was off by
    a factor of 24; the value itself was left unchanged so the delivered round-4
    model keeps behaving exactly as it was trained.

    Args:
        p_max: target probability.
        iteration: current control-step counter (``env.common_step_counter``).
        ramp_iters: warm-up length in **control steps**; <= 0 means "no ramp".

    Returns:
        Probability to use for this iteration, in ``[0, p_max]``.
    """
    if p_max <= 0.0:
        return 0.0
    if ramp_iters <= 0:
        return p_max
    return p_max * min(1.0, max(0.0, float(iteration)) / float(ramp_iters))


def advance_stop_window(
    command: torch.Tensor,
    prev_command: torch.Tensor,
    age: torch.Tensor,
    anchor: torch.Tensor,
    pos: torch.Tensor,
    window_steps: int,
    onset_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the falling-edge latch by one control step.

    Args:
        command: ``(N, >=2)`` current command, base frame.
        prev_command: same shape, command of the previous control step.
        age: ``(N,)`` int64 steps left in the active window; 0 means inactive.
        anchor: ``(N, 2)`` planar base position recorded at the falling edge.
        pos: ``(N, 2)`` current planar base position.
        window_steps: window length in control steps.
        onset_tol: a command with ``|xy| <= onset_tol`` counts as stopped.

    Returns:
        ``(age, anchor, drift2, active)`` -- updated latch, squared displacement
        from the anchor and the boolean window mask.
    """
    cur_speed = torch.norm(command[:, :2], dim=1)
    prev_speed = torch.norm(prev_command[:, :2], dim=1)
    falling = (prev_speed > onset_tol) & (cur_speed <= onset_tol)
    moving = cur_speed > onset_tol

    anchor = torch.where(falling.unsqueeze(1), pos, anchor)
    age = torch.where(
        falling,
        torch.full_like(age, int(window_steps)),
        torch.clamp(age - 1, min=0),
    )
    # a new non-zero command ends the window immediately, so a latch can never
    # survive into the next hold
    age = torch.where(moving, torch.zeros_like(age), age)

    drift2 = torch.sum(torch.square(pos - anchor), dim=1)
    return age, anchor, drift2, age > 0


def relatch_on_episode_start(
    age: torch.Tensor,
    anchor: torch.Tensor,
    pos: torch.Tensor,
    episode_length_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the latch for the envs that were respawned on the previous step.

    In this IsaacLab version ``ManagerBasedRLEnv.step`` computes the reward
    *before* ``_reset_idx`` (line order: ``episode_length_buf += 1`` -> reward ->
    reset), so the contaminated step is the one **after** a reset: the counter was
    zeroed by ``_reset_idx`` and then incremented to 1, while the anchor still
    belongs to the finished episode.  The two positions can be metres apart,
    which turns one step into ``-5 * (3 / 0.1)**2 = -4500`` -- invisible in the
    logged mean (a handful of envs out of 4096) but enough to destroy the critic:
    the first round 4 smoke run showed value loss climbing 0.05 -> 16.9 within 40
    iterations while the same run with the term disabled converged to 0.006, and
    guarding on ``== 0`` instead of ``<= 1`` kept charging a fake penalty at every
    respawn and stalled the early phase (episode length stuck at ~16 steps).

    Args:
        age: ``(N,)`` window counter.
        anchor: ``(N, 2)`` anchor position.
        pos: ``(N, 2)`` current position.
        episode_length_buf: ``(N,)`` env step counter; ``<= 1`` means the step
            right after a reset (or the very first step of the run).

    Returns:
        ``(age, anchor, drift2, active)`` with respawned envs cleared.
    """
    fresh = episode_length_buf <= 1
    if not bool(fresh.any()):
        drift2 = torch.sum(torch.square(pos - anchor), dim=1)
        return age, anchor, drift2, age > 0

    anchor = torch.where(fresh.unsqueeze(1), pos, anchor)
    age = torch.where(fresh, torch.zeros_like(age), age)
    drift2 = torch.sum(torch.square(pos - anchor), dim=1)
    return age, anchor, drift2, age > 0
