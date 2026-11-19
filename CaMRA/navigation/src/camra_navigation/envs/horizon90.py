"""Candidate adapters sharing a single 90-second clock and observation version.

These reuse frozen motion, cost and event implementations. They do not change
old launchers or validate old actors/critics for the new task. Physical inputs
are actual states supplied by a caller, never generated simulator evidence.
"""
import copy

import numpy as np

from camra_calibration.nominal import PlanarState
from .. import observation as O
from ..costs import SignedBudget
from ..horizon import PROFILE, encode_observation
from .nominal import NominalNavEnv, wall_polygons
from .vector import VectorNominalNavEnv
from .physical_v2 import SubstepNavigation


def _options(options):
    options = dict(options)
    if options.pop('t_max_s', PROFILE.t_max_s) != PROFILE.t_max_s:
        raise ValueError('candidate requires the registered 90-second horizon')
    if options.pop('observation_version', PROFILE.observation_version) != PROFILE.observation_version:
        raise ValueError('candidate requires obs-v8-t90')
    return options


class NominalHorizon90(NominalNavEnv):
    def __init__(self, spec, geometry, cost_model, **options):
        super().__init__(spec, geometry, cost_model, t_max_s=PROFILE.t_max_s,
                         observation_version='obs-v7', **_options(options))
        self.observation_version = PROFILE.observation_version

    def observation(self):
        return encode_observation(self.ego, self._obstacle_obs, self._extent,
                                  self.task, inflation_radius_m=self.geometry.inflation_radius_m)


class VectorHorizon90(VectorNominalNavEnv):
    def __init__(self, spec, geometry, cost_model, **options):
        super().__init__(spec, geometry, cost_model, t_max_s=PROFILE.t_max_s,
                         observation_version='obs-v7', **_options(options))
        self.observation_version = PROFILE.observation_version

    def _refresh_obs(self):
        for i in range(self.n_envs):
            ego = O.EgoState(*self._pose[i], *self._applied_last[i])
            task = O.TaskState(float(self.spec.goal[0]), float(self.spec.goal[1]),
                               float(self._safety_y[i]), float(self._z0[i]-self._spent_J[i]),
                               PROFILE.remaining(float(self._t_s[i])))
            self._obs[i] = encode_observation(ego, self._obstacle_obs, self._extent, task,
                                             inflation_radius_m=self.geometry.inflation_radius_m)


class PhysicalHorizon90(SubstepNavigation):
    """90-second initializer; unchanged inherited 200 Hz event/accounting path.

    Initialization is explicit rather than disguising a 90-second registration
    as the frozen 45-second preregistration accepted by the old initializer.
    """
    def __init__(self, spec, geometry, cost_model, initial_states, z0_J, prereg):
        PROFILE.require_binding(prereg['time_binding'])
        if (prereg['environment']['t_max_s'] != PROFILE.t_max_s or
                prereg['implementation_versions']['observation'] != PROFILE.observation_version):
            raise ValueError('physical registration disagrees with its time binding')
        self.spec, self.geom, self.cm = spec, geometry, cost_model
        self.prereg = copy.deepcopy(prereg)
        self.horizon = PROFILE.t_max_s
        self.goal_radius = float(prereg['environment']['r_goal_m'])
        self.obs_version = PROFILE.observation_version
        self.states = list(initial_states)
        for state in self.states:
            state.validate()
        self.n = len(self.states)
        if self.n < 1 or len(z0_J) != self.n or not np.isfinite(z0_J).all() or any(z <= 0 for z in z0_J):
            raise ValueError('one positive finite J budget per environment required')
        self.budgets = [SignedBudget(z) for z in z0_J]
        self.active, self.labels, self.y, self.times = [True]*self.n, [None]*self.n, [0.]*self.n, [0.]*self.n
        self.commands, self.powers, self.cost_nom = [(0.,0.,0.)]*self.n, [0.]*self.n, [0.]*self.n
        self.reference = [PlanarState(s.x,s.y,s.yaw) for s in self.states]
        self.history = [[(0.,s.x,s.y)] for s in self.states]
        self.terminal, self.rows, self.tick = [None]*self.n, [], 0
        self._obstacles = tuple(o.polygon() for o in spec.obstacles())
        self._raw = self._obstacles + wall_polygons(spec.extent)
        self._inflated = tuple(p.inflate(geometry.inflation_radius_m) for p in self._obstacles)
        self._obs_obstacles = tuple(O.obstacle_from_polygon(p.vertices,label=f'island-{i}')
                                    for i,p in enumerate(self._obstacles))
        self._extent = O.MapExtent(*spec.extent)

    def observation(self, i):
        if not self.active[i]:
            raise ValueError('cannot query a terminated episode')
        s = self.states[i]
        task = O.TaskState(float(self.spec.goal[0]),float(self.spec.goal[1]),
                           self.y[i],self.budgets[i].z_J,PROFILE.remaining(self.times[i]))
        return encode_observation(O.EgoState(s.x,s.y,s.yaw,s.vx,s.vy,s.wz),
                                  self._obs_obstacles,self._extent,task,
                                  inflation_radius_m=self.geom.inflation_radius_m)
