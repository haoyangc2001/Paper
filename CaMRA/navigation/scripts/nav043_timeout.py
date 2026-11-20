"""Execution-only 90s diagnostic for the frozen 45s actor; no new task model."""
import math

from camra_navigation import observation as O
from camra_navigation.envs.physical_v2 import SubstepNavigation


class TimeoutNavigation(SubstepNavigation):
    """Keep obs-v7's 45s clock; change only the physical stopping deadline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_horizon = self.horizon
        self.horizon = 90.

    def observation(self, i):
        if not self.active[i]:
            raise ValueError('cannot query a terminated episode')
        s = self.states[i]
        return O.encode_observation(
            O.EgoState(s.x, s.y, s.yaw, s.vx, s.vy, s.wz),
            self._obs_obstacles, self._extent,
            O.TaskState(float(self.spec.goal[0]), float(self.spec.goal[1]),
                        self.y[i], self.budgets[i].z_J,
                        max(0., self.actor_horizon-self.times[i])),
            version=self.obs_version, inflation_radius_m=self.geom.inflation_radius_m)


def diagnostic_outcome(label, safety_y, t_s, measured_J, internal_J, budget_J):
    if not all(math.isfinite(x) for x in
               (safety_y, t_s, measured_J, internal_J, budget_J)) or budget_J <= 0:
        raise ValueError('invalid physical outcome units')
    reached = label == 'goal_reached' and t_s <= 90.+1e-9
    safe = safety_y == 0. and label not in ('collision', 'fall', 'out_of_bounds')
    return {'timely_goal': reached, 'safe': safe,
            'safe_timely_measured_budget_success': reached and safe and measured_J <= budget_J,
            'internal_safe_budget_success': reached and safe and internal_J <= budget_J,
            'measured_budget_utilization': measured_J/budget_J,
            'internal_budget_utilization': internal_J/budget_J,
            'measured_overspend_J': max(0., measured_J-budget_J),
            'measured_overspend_fraction': max(0., measured_J/budget_J-1.),
            'execution_horizon_s': 90., 'actor_observation_horizon_s': 45.,
            'old_45s_timely_goal': reached and t_s <= 45.+1e-9,
            'performance_evidence': False}
