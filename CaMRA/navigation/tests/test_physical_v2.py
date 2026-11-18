"""Basic NAV-067 counterexamples; these are not real Isaac evidence."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camra_calibration.geometry import Polygon
from camra_navigation.envs.physical_v2 import (
    DT, PhysicalState, SubstepNavigation, require_command_evidence)
from camra_navigation.maps.geometry import SafetyGeometry

NAV = Path(__file__).resolve().parents[1]
PREREG = json.loads((NAV/'config/training_loop_prereg_terminal_tail_v1.json').read_bytes())


def state(x=0.,y=0.,yaw=0.):
    return PhysicalState(x,y,yaw,0.,0.,0.,.35,(math.cos(yaw/2),0.,0.,math.sin(yaw/2)))


def monitor(*,obstacle=None,delta=0.,goal=(9.,9.),initial=None):
    fp = Polygon(np.array([[-.3,-.1],[.3,-.1],[.3,.1],[-.3,.1]]))
    geom = SafetyGeometry(fp,fp.circumscribed_radius,delta,'test','','')
    polygons = [] if obstacle is None else [Polygon(np.asarray(obstacle,dtype=float))]
    spec = SimpleNamespace(extent=(-20.,20.,-20.,20.),goal=goal,
        obstacles=lambda:[SimpleNamespace(polygon=lambda p=p:p) for p in polygons])
    cm = SimpleNamespace(alpha=.2,step=lambda cmd:SimpleNamespace(power_W=10.))
    return SubstepNavigation(spec,geom,cm,initial or [state()],[100.] * len(initial or [0]),copy.deepcopy(PREREG))


def test_substep_hit_then_exit_is_latched_and_terminal_frozen():
    m = monitor(obstacle=[[.5,-1.],[.6,-1.],[.6,1.],[.5,1.]])
    m.set_commands([(1.,0.,0.)])
    event = m.advance([state(1.)],physics_tick=1)
    assert event['ended']==[0] and m.labels==['collision']
    assert m.y[0]>0 and m.terminal[0]['events']['collision']
    saved = (m.states[0],m.times[0],m.budgets[0].z_J,len(m.rows))
    m.advance([state(2.)],physics_tick=2)
    assert saved == (m.states[0],m.times[0],m.budgets[0].z_J,len(m.rows))
    assert m.budgets[0].spent_J == pytest.approx(10.*1.2*DT)


def test_danger_history_survives_exit_without_collision():
    m = monitor(obstacle=[[.5,.8],[.6,.8],[.6,1.],[.5,1.]],delta=1.)
    m.advance([state(.5)],physics_tick=1)
    assert m.active[0] and m.y[0]>0
    y = m.y[0]
    m.advance([state(.5,-2.)],physics_tick=2)
    m.advance([state(.5,-3.)],physics_tick=3)
    assert m.active[0] and m.y[0]==y and not m.rows[-1]['danger']


def test_rotation_sweep_detects_contact_between_safe_endpoints():
    m = monitor(obstacle=[[.18,.18],[.22,.18],[.22,.22],[.18,.22]])
    m.advance([state(yaw=math.pi/2)],physics_tick=1)
    assert m.labels[0]=='collision'


def test_collision_wins_over_simultaneous_arrival_both_events_retained():
    m = monitor(obstacle=[[.5,-1],[.6,-1],[.6,1],[.5,1]],goal=(1.,0.))
    m.advance([state(1.)],physics_tick=1)
    assert m.labels[0]=='collision'
    assert m.rows[-1]['events']['goal_reached']


def test_async_termination_does_not_stop_other_env_or_charge_dead_env():
    m = monitor(goal=(1.,0.),initial=[state(),state(-2.)])
    m.set_commands([(1.,0.,0.),(1.,0.,0.)])
    m.advance([state(.6),state(-1.9)],physics_tick=1)
    assert m.active==[False,True]
    m.advance([state(10.),state(-1.8)],physics_tick=2)
    assert m.times==pytest.approx([DT,2*DT])
    assert [b.spent_J for b in m.budgets]==pytest.approx([.06,.12])
    assert m.states[0].x==.6 and m.states[1].x==-1.8


@pytest.mark.parametrize('tick',[0,2])
def test_physics_evidence_missing_or_repeated_is_rejected(tick):
    with pytest.raises(ValueError,match='out-of-order'):
        monitor().advance([state()],physics_tick=tick)


def test_no_budget_terminal_and_model_cost_composes_for_full_nav_interval():
    m = monitor()
    m.budgets[0].z0_J=.001
    m.set_commands([(1.,0.,0.)])
    for k in range(20):
        m.advance([state(k*.005)],physics_tick=k+1)
    assert m.active[0] and m.budgets[0].z_J<0
    assert m.cost_nom[0]==pytest.approx(1.)
    assert m.budgets[0].spent_J==pytest.approx(1.2)
    assert m.reference[0].px==pytest.approx(.1)


@pytest.mark.parametrize('kind',['missing','stale','clock','nonfinite'])
def test_bad_actual_command_evidence_is_rejected(kind):
    seen = [[1.,0.,0.]]
    if kind=='missing': seen=[]
    if kind=='stale': seen=[[0.,0.,0.]]
    if kind=='nonfinite': seen=[[float('nan'),0.,0.]]
    with pytest.raises(ValueError):
        require_command_evidence([[1.,0.,0.]],seen,tick_before=0,
                                 tick_after=3 if kind=='clock' else 4,substeps=4)


def test_partial_cycle_evidence_preserves_actual_duration():
    r = require_command_evidence([[1.,0.,0.]],[[1.,0.,0.]],tick_before=7,tick_after=9,substeps=2)
    assert r['substeps']==2 and r['max_error']==0


def test_observation_is_explicit_v7_actual_state_and_unclipped_budget():
    m = monitor(initial=[state(1.,2.,.3)])
    m.budgets[0].z0_J=1100.
    from camra_navigation.observation import field_slices
    row = np.asarray(m.observation(0))
    sl = field_slices('obs-v7')
    assert row[sl['z_ratio']].item()==2.
    assert len(row)==sum(s.stop-s.start for s in sl.values())


@pytest.mark.parametrize('field',['height','x'])
def test_nonfinite_actual_state_rejected(field):
    with pytest.raises(ValueError,match='nonfinite'):
        monitor().advance([replace(state(),**{field:float('nan')})],physics_tick=1)
