"""NAV-067 substep state, event and budget accounting.

This module is isolated from the historical NAV-042 loop. Isaac passes an
actual state after every 0.005 s step; no simulator or automatic reset is
hidden here. The event step is charged, subsequent steps are masked out.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from camra_calibration.nominal import PlanarState, step_exact
from .. import observation as O
from ..costs import SignedBudget
from ..maps import geometry as G
from .nominal import LABELS, wall_polygons

DT = 0.005


@dataclass(frozen=True)
class PhysicalState:
    # Map coordinates, body velocities, actual height and wxyz quaternion.
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    height: float
    quat: tuple[float, float, float, float]

    def validate(self):
        if not all(math.isfinite(float(v)) for v in
                   (self.x, self.y, self.yaw, self.vx, self.vy, self.wz,
                    self.height, *self.quat)):
            raise ValueError('nonfinite actual physics state')
        if abs(sum(v*v for v in self.quat) - 1.) > 1e-4:
            raise ValueError('actual quaternion must be normalized wxyz')

    @property
    def tilt_deg(self):
        _, x, y, _ = self.quat
        return math.degrees(math.acos(np.clip(1.-2.*(x*x+y*y), -1., 1.)))


class SubstepNavigation:
    """One map, possibly asynchronous environments, no implicit reset.

    Observation and cost definitions are the selected model's frozen ones.
    Full 0.1 s intervals equal frozen StepCost exactly; a terminal partial
    interval uses the same frozen power times its actual active duration.
    Physical collision uses the raw map, danger uses footprint+delta once.
    """

    def __init__(self, spec, geometry, cost_model, initial_states, z0_J, prereg):
        self.spec, self.geom, self.cm = spec, geometry, cost_model
        self.prereg = prereg
        self.horizon = float(prereg['environment']['t_max_s'])
        self.goal_radius = float(prereg['environment']['r_goal_m'])
        self.obs_version = prereg['implementation_versions']['observation']
        if self.horizon != 45. or self.obs_version != 'obs-v7':
            raise ValueError('requires the selected candidates frozen 45s/obs-v7')
        self.states = list(initial_states)
        for state in self.states:
            state.validate()
        self.n = len(self.states)
        if self.n < 1 or len(z0_J) != self.n:
            raise ValueError('one positive J budget per environment required')
        self.budgets = [SignedBudget(z) for z in z0_J]
        self.active = [True] * self.n
        self.labels = [None] * self.n
        self.y = [0.] * self.n
        self.times = [0.] * self.n
        self.commands = [(0., 0., 0.)] * self.n
        self.powers = [0.] * self.n
        self.cost_nom = [0.] * self.n
        self.reference = [PlanarState(s.x, s.y, s.yaw) for s in self.states]
        self.history = [[(0., s.x, s.y)] for s in self.states]
        self.terminal = [None] * self.n
        self.rows = []
        self.tick = 0
        self._obstacles = tuple(o.polygon() for o in spec.obstacles())
        self._raw = self._obstacles + wall_polygons(spec.extent)
        self._inflated = tuple(p.inflate(geometry.inflation_radius_m) for p in self._obstacles)
        self._obs_obstacles = tuple(O.obstacle_from_polygon(p.vertices, label=f'island-{i}')
                                    for i, p in enumerate(self._obstacles))
        self._extent = O.MapExtent(*spec.extent)

    def observation(self, i):
        if not self.active[i]:
            raise ValueError('cannot query a terminated episode')
        s = self.states[i]
        return O.encode_observation(O.EgoState(s.x, s.y, s.yaw, s.vx, s.vy, s.wz),
            self._obs_obstacles, self._extent,
            O.TaskState(float(self.spec.goal[0]), float(self.spec.goal[1]),
                        self.y[i], self.budgets[i].z_J,
                        max(0., self.horizon-self.times[i])),
            version=self.obs_version, inflation_radius_m=self.geom.inflation_radius_m)

    def set_commands(self, commands):
        if len(commands) != self.n:
            raise ValueError('command batch shape mismatch')
        for i, row in enumerate(commands):
            cmd = tuple(float(v) for v in row)
            if len(cmd) != 3 or not all(math.isfinite(v) for v in cmd):
                raise ValueError('invalid applied command')
            if self.active[i]:
                self.commands[i] = cmd
                self.powers[i] = self.cm.step(cmd).power_W

    def advance(self, states, *, physics_tick):
        """Call once after scene.update(0.005), before any reset or next step."""
        if physics_tick != self.tick + 1 or len(states) != self.n:
            raise ValueError('missing, repeated or out-of-order physics evidence')
        for state in states:
            state.validate()
        before = list(self.active)
        ended = []
        for i, (old, new) in enumerate(zip(self.states, states)):
            if not before[i]:
                continue  # terminal state, time and budget remain frozen
            p0, p1 = (old.x, old.y), (new.x, new.y)
            collision = any(G.physical_collision_during_step(p0, old.yaw, p1,
                             new.yaw, p, self.geom) for p in self._raw)
            boundary_margin = min(G.boundary_margin(p0,self.spec.extent,self.geom),
                                  G.boundary_margin(p1,self.spec.extent,self.geom))
            danger = boundary_margin <= 0 or any(
                G.segment_intersects_polygon(p0,p1,p) for p in self._inflated)
            if danger:
                margin = min([boundary_margin] + [G.nominal_margin(p1,p,self.geom)
                                                   for p in self._obstacles])
                sev = self.prereg['safety_continuity']
                severity = G.safety_severity_from_depth(-min(0.,margin),
                    floor=sev['severity_floor'],ref_depth_m=sev['severity_ref_depth_m']) if sev['enabled'] else 1.
                self.y[i] = max(self.y[i],severity)
            vertices = G.footprint_at(self.geom,p1,new.yaw).vertices
            x0,x1,y0,y1 = self.spec.extent
            outside = bool(np.any(vertices[:,0]<x0) or np.any(vertices[:,0]>x1)
                           or np.any(vertices[:,1]<y0) or np.any(vertices[:,1]>y1))
            self.times[i] += DT
            nominal_j = self.powers[i] * DT
            self.cost_nom[i] += nominal_j
            self.budgets[i].charge((1.+self.cm.alpha)*nominal_j)
            ref = step_exact(self.reference[i], self.commands[i], dt=DT)
            self.reference[i] = ref
            self.states[i] = new
            t = self.times[i]
            self.history[i].append((t,new.x,new.y))
            stag = self.prereg['stagnation']
            while len(self.history[i])>1 and self.history[i][1][0] <= t-stag['window_s']+1e-9:
                self.history[i].pop(0)
            anchor = self.history[i][0]
            stagnated = t >= stag['window_s']-1e-9 and math.hypot(new.x-anchor[1],new.y-anchor[2]) < stag['min_displacement_m']
            distance = math.hypot(new.x-self.spec.goal[0],new.y-self.spec.goal[1])
            events = {'fall':new.height<.15 or new.tilt_deg>60.,
                      'collision':collision,'out_of_bounds':outside,
                      'goal_reached':distance<=self.goal_radius,
                      'stagnation':stagnated,'timeout':t>=self.horizon-1e-9}
            label = next((k for k in LABELS if events[k]),None)
            row = {'physics_tick':physics_tick,'env_id':i,'t_s':t,
                   'state':[new.x,new.y,new.yaw,new.vx,new.vy,new.wz,new.height,*new.quat],
                   'applied':list(self.commands[i]),'danger':bool(danger),
                   'safety_y':self.y[i],'events':events,'label':label,
                   'z_J':self.budgets[i].z_J,'cost_nom_J':nominal_j,
                   'goal_distance_m':distance,'ref_pose':[ref.px,ref.py,ref.yaw],
                   'deviation_m':math.hypot(new.x-ref.px,new.y-ref.py)}
            self.rows.append(row)
            if label:
                self.active[i]=False
                self.labels[i]=label
                self.terminal[i]=row
                ended.append(i)
        self.tick = physics_tick
        return {'active_before':before,'ended':ended,'active_after':list(self.active)}


def require_command_evidence(written, seen, *, tick_before, tick_after, substeps):
    """No fallback to requested command or constructed simulator clocks."""
    w,s = np.asarray(written,dtype=float),np.asarray(seen,dtype=float)
    if w.ndim!=2 or w.shape[1]!=3 or s.shape!=w.shape or not np.isfinite(s).all():
        raise ValueError('missing or malformed actual observation command evidence')
    error = float(np.max(np.abs(w-s)))
    if error>1e-6 or not 1<=substeps<=4 or tick_after-tick_before!=substeps:
        raise ValueError('stale command or incorrect actual physics clock')
    return {'command_written':w.tolist(),'command_seen':s.tolist(),
            'max_error':error,'tick_before':tick_before,'tick_after':tick_after,
            'substeps':substeps}
