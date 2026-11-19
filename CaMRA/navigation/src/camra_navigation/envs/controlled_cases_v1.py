"""NAV-067 synthetic fixtures and actual-readback assertions, without a simulator.

Pose schedules are test inputs, never substituted for measured states. The
existing production monitor and low-level adapter remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from camra_calibration.geometry import Polygon
from ..maps import geometry as G
from ..maps.generator import MapSpec
from .physical_v2 import DT, PhysicalState

CASE_IDS = tuple(f'C{i:02}' for i in range(1, 7))


def _box(cx, cy, sx, sy):
    return Polygon(np.array([[cx-sx/2, cy-sy/2], [cx+sx/2, cy-sy/2],
                             [cx+sx/2, cy+sy/2], [cx-sx/2, cy+sy/2]]))


def build_fixture(case_id, geom):
    if case_id not in CASE_IDS:
        raise ValueError('unknown controlled case')
    r, delta = geom.r_fp_m, geom.delta_m
    bound = max(30., 6.*geom.inflation_radius_m+6.)
    far = 5.*r+5.
    start = (-2.*r-2., 0., 0.)
    goal = (2.*r+3., far)
    companion = (-2.*r-2., far, 0.)
    islands = [(0., 0., .25, 2., 0.)]
    initial = [start, companion]
    targets = {
        1: [start, companion],
        2: [(0., 0., 0.), companion],
        3: [(2.*r+2., 0., 0.), companion],
        4: [(2.*r+2., 0., 0.), (*goal, 0.)],
    }
    max_ticks = 4
    if case_id == 'C02':
        if delta <= 1.e-4:
            raise ValueError('frozen full geometry has no usable danger-only band')
        y = .1+r+delta/2.
        safe = (-geom.inflation_radius_m-2., y, 0.)
        leave = (geom.inflation_radius_m+2., y, 0.)
        islands = [(0., 0., .2, .2, 0.)]
        initial = [safe]
        targets = {1: [safe], 2: [(0., y, 0.)], 3: [leave], 4: [leave]}
    elif case_id == 'C03':
        # Choose the largest endpoint clearance using frozen geometry only.
        # This is not selected using actual simulation outcomes.
        choices = []
        for yaw in (math.pi/8., math.pi/4., 3.*math.pi/8.):
            for vertex in G.footprint_at(geom, (0., 0.), yaw).vertices:
                cx, cy = .97*vertex
                obstacle = _box(float(cx), float(cy), .02, .02)
                clear = min(G.polygon_distance_exact(G.footprint_at(geom, (0., 0.), a), obstacle)
                            for a in (0., math.pi/2.))
                if clear > .015 and G.physical_collision((0., 0.), yaw, obstacle, geom):
                    choices.append((clear, float(cx), float(cy), yaw))
        if not choices:
            raise ValueError('no turning fixture with safe endpoints for frozen footprint')
        _, cx, cy, _ = max(choices)
        islands = [(cx, cy, .02, .02, 0.)]
        initial = [(0., 0., 0.)]
        targets = {1: [(0., 0., 0.)], 2: [(0., 0., math.pi/2.)]}
        max_ticks = 2
    elif case_id == 'C04':
        goal = (0., 0.)
        initial = [start]
        targets = {1: [start], 2: [(0., 0., 0.)]}
        max_ticks = 2
    elif case_id == 'C05':
        islands = []
        goal = (2., 0.)
        initial = [(-2., 0., 0.), (-2., 0., 0.)]
        targets = {1: initial, 2: [(1.9, 0., 0.), (-1., 0., 0.)],
                   3: [(1.9, 0., 0.), (1.9, 0., 0.)]}
        max_ticks = 3
    elif case_id == 'C06':
        islands = [(0., 0., .5, 2., 0.)]
        # One initial fixture placement; subsequent native physics is free.
        targets = {1: [(-.1, 0., 0.), companion]}
        max_ticks = 200
    spec = MapSpec(map_id='nav067-'+case_id, level='CONTROLLED', seed=3001,
                   extent=(-bound, bound, -bound, bound), start=initial[0][:2],
                   start_yaw=initial[0][2], goal=goal, islands=tuple(islands),
                   design_version='nav067-controlled-fixture-v1')
    return {'case_id': case_id, 'spec': spec.to_dict(), 'num_envs': len(initial),
            'initial': [list(p) for p in initial],
            'targets': {str(k): [list(p) for p in v] for k, v in targets.items()},
            'measurement_tick_cap': max_ticks, 'warmup_policy_calls': 50,
            'commands': [[0., 0., 0.]]*len(initial),
            'evidence_kind': 'native-contact-response-after-initial-placement' if case_id == 'C06'
                             else 'deliberately-written-fixture-with-actual-readback',
            'expected_task_horizon_s': 45., 'performance_evidence': False}


def fixture_spec(fixture):
    return MapSpec.from_dict(fixture['spec'])


class ScheduledMonitor:
    """Feed real post-step states to production, then write the next test input."""
    def __init__(self, monitor, fixture, write_poses, after_actual_tick=None):
        self.monitor, self.fixture = monitor, fixture
        self.write_poses = write_poses
        self.after_actual_tick = after_actual_tick
        self.raw = []
        self.requested_writes = []

    def __getattr__(self, key):
        return getattr(self.monitor, key)

    def prepare_tick(self, tick):
        target = self.fixture['targets'].get(str(tick))
        if target is not None:
            self.write_poses(target)
            self.requested_writes.append({'before_measurement_tick': tick, 'requested_map_poses': deepcopy(target)})

    def advance(self, states, *, physics_tick):
        if physics_tick > self.fixture['measurement_tick_cap']:
            raise ValueError('controlled measurement tick cap exceeded')
        actual = [vars(state).copy() for state in states]
        event = self.monitor.advance(states, physics_tick=physics_tick)
        contact = self.after_actual_tick() if self.after_actual_tick else None
        self.raw.append({'physics_tick': physics_tick, 'actual_states': actual,
                         'active_before': list(event['active_before']),
                         'active_after': list(event['active_after']),
                         'monitor_times_s': list(self.monitor.times),
                         'monitor_spent_J': [b.spent_J for b in self.monitor.budgets],
                         'monitor_safety_y': list(self.monitor.y),
                         'monitor_labels': list(self.monitor.labels),
                         'contact': contact})
        if any(self.active) and physics_tick < self.fixture['measurement_tick_cap']:
            self.prepare_tick(physics_tick+1)
        return event


def _state(row):
    return PhysicalState(**row)


def review_actual_case(fixture, geom, scheduled, energy, commands):
    """Strict actual-evidence checks; never derive measured states from targets."""
    m, raw = scheduled.monitor, scheduled.raw
    case_id = fixture['case_id']
    if not raw or [r['physics_tick'] for r in raw] != list(range(1, len(raw)+1)):
        raise ValueError('missing actual controlled ticks')
    if len(raw) > fixture['measurement_tick_cap'] or not commands:
        raise ValueError('controlled evidence exceeds cap or has no actual command evidence')
    if energy['steps'] != len(raw) or energy['problems']:
        raise ValueError('controlled recorder completeness failure')
    before = np.asarray([r['active_before'] for r in raw], dtype=bool)
    if not np.array_equal(energy['valid_mask'], before):
        raise ValueError('actual energy mask differs from pre-event active mask')
    if not np.array_equal(np.sum(before, axis=0), energy['env_active_steps']):
        raise ValueError('asynchronous active-step accounting mismatch')
    if any(c['max_error'] > 1.e-6 for c in commands):
        raise ValueError('stale actual observation command')
    # Every inactive episode must keep its measured terminal time and budget.
    for eid in range(m.n):
        frozen = None
        for row in raw:
            value = (row['monitor_times_s'][eid], row['monitor_spent_J'][eid])
            if frozen is not None and value != frozen:
                raise ValueError('terminal episode was charged after its event')
            if not row['active_after'][eid]:
                frozen = value
    spec = fixture_spec(fixture)
    obstacles = tuple(o.polygon() for o in spec.obstacles())
    first = [r for r in m.rows if r['env_id'] == 0]
    if case_id == 'C01':
        if m.labels[0] != 'collision' or m.terminal[0]['physics_tick'] != 2 or len(raw) != 4:
            raise ValueError('inner-substep collision was absent or recorded at the wrong tick')
        tail = _state(raw[-1]['actual_states'][0])
        if any(G.physical_collision((tail.x, tail.y), tail.yaw, o, geom) for o in obstacles):
            raise ValueError('actual raw pose never exited after collision')
    elif case_id == 'C02':
        if len(raw) != 4 or not any(r['danger'] for r in first) or first[-1]['danger']:
            raise ValueError('actual danger entry and exit are missing')
        if m.y[0] <= 0 or m.labels[0] is not None or any(r['events']['collision'] for r in first):
            raise ValueError('danger history or danger-only termination semantics failed')
    elif case_id == 'C03':
        a = _state(raw[0]['actual_states'][0])
        b = _state(raw[-1]['actual_states'][0])
        if m.labels[0] != 'collision' or len(raw) != 2:
            raise ValueError('actual turning sweep was not detected')
        if any(G.physical_collision((s.x, s.y), s.yaw, o, geom) for s in (a, b) for o in obstacles):
            raise ValueError('turning fixture actual endpoints were not clear')
        if any(G.segment_intersects_polygon((a.x, a.y), (b.x, b.y), o) for o in obstacles):
            raise ValueError('turning fixture center chord intersected the obstacle')
    elif case_id == 'C04':
        terminal = m.terminal[0]
        if not terminal or not terminal['events']['collision'] or not terminal['events']['goal_reached'] or m.labels[0] != 'collision':
            raise ValueError('actual simultaneous events or existing priority failed')
    elif case_id == 'C05':
        if m.labels != ['goal_reached', 'goal_reached'] or energy['env_active_steps'] != [2, 3]:
            raise ValueError('actual asynchronous termination did not occur on ticks 2 and 3')
        if len(raw) != 3 or commands[-1]['substeps'] != 3 or abs(m.times[0]-2.*DT) > 1.e-12 or abs(m.times[1]-3.*DT) > 1.e-12:
            raise ValueError('terminal partial cycle time/mask mismatch')
    elif case_id == 'C06':
        contact = [r['contact'] for r in raw if r['contact'] is not None]
        if not contact or max(c['robot_obstacle_pair_force_norm_N'] for c in contact) <= 1.e-6:
            raise ValueError('no native robot-obstacle filtered contact evidence')
    return {'case_id': case_id, 'actual_evidence_assertions_passed': True,
            'physics_steps': len(raw), 'env_active_steps': energy['env_active_steps'],
            'labels': list(m.labels), 'evidence_kind': fixture['evidence_kind'],
            'performance_evidence': False, 'synthetic_fixtures_are_navigation_successes': False}
