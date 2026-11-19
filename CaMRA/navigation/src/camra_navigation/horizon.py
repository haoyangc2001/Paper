"""Candidate 90-second task contract, isolated from frozen 45-second models.

This module supplies clocks and encoding, not evidence of model or calibration
validity. A model must explicitly bind to this profile before a future launcher
may use it. Equal tensor shapes do not imply compatible time semantics.
"""
from dataclasses import dataclass, replace
import math

from . import observation as O


@dataclass(frozen=True)
class Horizon90:
    profile_id: str = 'navigation-t90-v1'
    observation_version: str = 'obs-v8-t90'
    t_max_s: float = 90.

    def __post_init__(self):
        if (self.profile_id, self.observation_version, self.t_max_s) != (
                'navigation-t90-v1', 'obs-v8-t90', 90.):
            raise ValueError('unregistered horizon profile')

    def remaining(self, elapsed_s):
        if not math.isfinite(elapsed_s) or not 0. <= elapsed_s <= self.t_max_s + 1.e-8:
            raise ValueError('elapsed time outside the registered task')
        return max(0., self.t_max_s - elapsed_s)

    def step_cap(self, dt_s):
        if dt_s not in (.1, .02, .005):
            raise ValueError('only frozen 10/50/200 Hz clocks are allowed')
        return round(self.t_max_s / dt_s)

    def require_binding(self, binding):
        expected = {'time_profile': self.profile_id, 't_max_s': self.t_max_s,
                    'observation_version': self.observation_version}
        if any(binding.get(k) != v for k, v in expected.items()):
            raise ValueError('model/search/evaluation time binding mismatch; legacy models require revalidation')


PROFILE = Horizon90()


def encode_observation(ego, obstacles, extent, task, *, inflation_radius_m):
    """Keep v7 geometry and budget; encode remaining seconds divided by 90.

    Only the temporary encoder argument is converted to equivalent v7 units.
    Environment task state and recorded times remain real seconds throughout.
    """
    remaining = task.time_remaining_s
    if not math.isfinite(remaining) or not 0. <= remaining <= PROFILE.t_max_s:
        raise ValueError('remaining time outside the registered task')
    argument = replace(task, time_remaining_s=45. * remaining / PROFILE.t_max_s)
    return O.encode_observation(ego, obstacles, extent, argument, version='obs-v7',
                                inflation_radius_m=inflation_radius_m)
