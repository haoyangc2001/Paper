from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from ..stop_events import ramped_probability, stop_event_mask


class StopEventVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` that turns a share of resamples into a full stop hold.

    Tuning round 4. ``start_stop`` grades a hard 0.8 m/s -> 0 stop, but a
    uniform resample of ``lin_vel_x = (-1, 1)`` almost never lands on exactly
    zero, and the ``rel_standing_envs`` share stands from the first step without
    ever braking -- so the graded transition was effectively untrained. With
    ``stop_event_prob > 0`` a resampled hold is zeroed out instead, which makes
    "brake from speed, then restart" a routine event (2-3 times per 20 s episode
    at 0.25 with the 4-8 s schedule).

    Everything else -- ranges, heading logic, standing envs, curriculum -- is
    inherited unchanged; with ``stop_event_prob = 0`` the behaviour is identical
    to the upstream class.
    """

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        prob = ramped_probability(
            float(getattr(self.cfg, "stop_event_prob", 0.0)),
            int(getattr(self._env, "common_step_counter", 0)),
            int(getattr(self.cfg, "stop_event_ramp_iters", 0)),
        )
        if prob <= 0.0 or len(env_ids) == 0:
            return
        sample = torch.empty(len(env_ids), device=self.device)
        # draw from the same RNG stream the parent used, so seeding stays honest
        sample.uniform_(0.0, 1.0)
        stop_ids = stop_event_mask(sample, prob)
        if not bool(stop_ids.any()):
            return
        selected = env_ids[stop_ids]
        self.vel_command_b[selected, :] = 0.0
        # a stopped env must not get its yaw command rebuilt from the heading target
        self.is_heading_env[selected] = False


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    #: Probability that a resampled hold is forced to a full stop (tuning round 4).
    #: 0.0 reproduces the upstream behaviour exactly.
    stop_event_prob: float = 0.0

    #: Iterations over which ``stop_event_prob`` is ramped in linearly. 0 = no ramp.
    #: See :func:`~.stop_events.ramped_probability`.
    stop_event_ramp_iters: int = 0

    class_type: type = StopEventVelocityCommand
