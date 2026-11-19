"""Clock/interface counterexamples; no simulator, trained policy or W queries."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from camra_navigation import observation as O
from camra_navigation.costs import JCostModel
from camra_navigation.envs.horizon90 import NominalHorizon90, VectorHorizon90, PhysicalHorizon90
from camra_navigation.envs.physical_v2 import PhysicalState
from camra_navigation.horizon import PROFILE, Horizon90, encode_observation
from camra_navigation.maps import generator as M
from camra_navigation.maps.geometry import SafetyGeometry

ROOT = Path(__file__).resolve().parents[3]
NAV = ROOT / 'CaMRA/navigation'
CAL = ROOT / 'CaMRA/calibration/config'


def binding():
    return dict(time_profile=PROFILE.profile_id, t_max_s=90., observation_version='obs-v8-t90')


def prereg():
    p = json.loads((NAV/'config/training_loop_prereg_terminal_tail_v1.json').read_bytes())
    p['environment']['t_max_s'] = 90.
    p['implementation_versions']['observation'] = 'obs-v8-t90'
    p['time_binding'] = binding()
    return p


def components():
    spec = M.build('L1_empty',101,M.load_design(str(NAV/'config/map_design_v1.json')))
    geom = SafetyGeometry.from_frozen(str(CAL/'footprint_v1.json'),str(CAL/'margins_v1.json'),
                                      str(NAV/'config/geometry_contract_v1.json'))
    cm = JCostModel.from_frozen(str(CAL/'power_model_v1.json'),str(CAL/'margins_v1.json'),variant='full')
    return spec,geom,cm


def actual(spec, x=None):
    yaw = spec.start_yaw
    return PhysicalState(spec.start[0] if x is None else x,spec.start[1],yaw,0.,0.,0.,.35,
                         (math.cos(yaw/2),0.,0.,math.sin(yaw/2)))


def test_all_900_decision_times_are_distinguishable_without_initial_plateau():
    task = O.TaskState(7.,0.,0.,550.,90.)
    ego = O.EgoState(-7.,0.,0.,0.,0.,0.)
    values = [encode_observation(ego,(),O.MapExtent(-20.,20.,-20.,20.),
              replace(task,time_remaining_s=PROFILE.remaining(i*.1)),inflation_radius_m=1.)[-1]
              for i in range(901)]
    np.testing.assert_allclose(values,np.linspace(1.,0.,901),atol=1.e-12)
    assert np.all(np.diff(values)<0)
    assert [PROFILE.step_cap(dt) for dt in (.1,.02,.005)] == [900,4500,18000]


@pytest.mark.parametrize('seconds',[float('nan'),float('inf'),-.1,90.1])
def test_invalid_time_rejected(seconds):
    with pytest.raises(ValueError):
        PROFILE.remaining(seconds)
    with pytest.raises(ValueError):
        encode_observation(O.EgoState(0.,0.,0.,0.,0.,0.),(),O.MapExtent(-20.,20.,-20.,20.),
                           O.TaskState(7.,0.,0.,550.,seconds),inflation_radius_m=1.)


def test_other_observation_fields_and_legacy_behavior_are_preserved():
    ego,extent = O.EgoState(1.,2.,.4,.3,.1,.2),O.MapExtent(-20.,20.,-20.,20.)
    new = O.TaskState(7.,0.,.5,1100.,60.)
    row = encode_observation(ego,(),extent,new,inflation_radius_m=1.)
    expected = O.encode_observation(ego,(),extent,replace(new,time_remaining_s=30.),
                                    version='obs-v7',inflation_radius_m=1.)
    np.testing.assert_array_equal(row,expected)
    assert row[-2] == 2. and row[-1] == pytest.approx(2/3)
    with pytest.raises(ValueError,match='45 seconds'):
        O.encode_observation(ego,(),extent,new,version='obs-v7',inflation_radius_m=1.)


@pytest.mark.parametrize('key,value',[('t_max_s',45.),('observation_version','obs-v7'),
                                     ('time_profile','legacy'),('time_profile',None)])
def test_same_shape_legacy_model_or_search_binding_rejected(key,value):
    b = binding()
    b[key] = value
    with pytest.raises(ValueError,match='binding mismatch'):
        PROFILE.require_binding(b)


def test_profile_cannot_be_changed_by_unregistered_overrides():
    with pytest.raises(ValueError):
        Horizon90(t_max_s=120.)
    with pytest.raises(ValueError):
        PROFILE.step_cap(.01)
    spec,g,cm = components()
    with pytest.raises(ValueError):
        NominalHorizon90(spec,g,cm,z0_J=550.,t_max_s=45.)


def test_three_interfaces_share_real_seconds_budget_and_encoding():
    spec,g,cm = components()
    s = NominalHorizon90(spec,g,cm,z0_J=550.)
    v = VectorHorizon90(spec,g,cm,z0_J=550.,n_envs=1)
    p = PhysicalHorizon90(spec,g,cm,[actual(spec)],[550.],prereg())
    for elapsed in (0.,.1,44.9,45.,89.9):
        s.t_s = elapsed
        v._t_s[:] = elapsed
        v._refresh_obs()
        p.times[0] = elapsed
        np.testing.assert_allclose(s.observation(),v.observation[0],atol=1.e-12)
        np.testing.assert_allclose(s.observation(),p.observation(0),atol=1.e-12)
        assert s.task.time_remaining_s == pytest.approx(90.-elapsed)
    assert s.z0_J == v.z0_J == p.budgets[0].z0_J == 550.


@pytest.mark.parametrize('arrive',[False,True])
def test_nominal_boundary_timeout_and_goal_priority_match_scalar_vector(arrive):
    spec,g,cm = components()
    if arrive:
        spec = replace(spec,goal=spec.start)
    s = NominalHorizon90(spec,g,cm,z0_J=550.)
    v = VectorHorizon90(spec,g,cm,z0_J=550.,n_envs=1)
    s.t_s = 89.9
    v._t_s[:] = 89.9
    # Boundary fixtures: preserve recent progress so stagnation is not triggered.
    s._history = [(89.8,s.pose[0]-1.,s.pose[1])]
    v._history[0] = [(89.8,s.pose[0]-1.,s.pose[1])]
    a,b = s.step((0.,0.,0.)),v.step(np.zeros((1,3)))
    expected = 'goal_reached' if arrive else 'timeout'
    assert s.label == v._label[0] == expected
    assert s.t_s == pytest.approx(90.)
    assert s.budget.spent_J > 0 and v._spent_J[0] == pytest.approx(s.budget.spent_J)


@pytest.mark.parametrize('arrive',[False,True])
def test_physical_boundary_retains_200hz_events_cost_and_inactive_mask(arrive):
    spec,g,cm = components()
    if arrive:
        spec = replace(spec,goal=spec.start)
    p = PhysicalHorizon90(spec,g,cm,[actual(spec)],[550.],prereg())
    p.times[0],p.tick = 89.995,17999
    p.history[0] = [(89.9,spec.start[0]-1.,spec.start[1])]
    p.set_commands([(0.,0.,0.)])
    p.advance([actual(spec)],physics_tick=18000)
    assert p.labels == ['goal_reached' if arrive else 'timeout']
    assert p.rows[-1]['events']['timeout']
    charged = p.budgets[0].spent_J
    assert charged == pytest.approx(cm.step((0.,0.,0.)).power_W*.005*(1.+cm.alpha))
    p.advance([actual(spec)],physics_tick=18001)
    assert p.budgets[0].spent_J == charged and p.times[0] == pytest.approx(90.)
    with pytest.raises(ValueError,match='terminated'):
        p.observation(0)


def test_physical_registration_rejects_inconsistent_clock_and_version():
    spec,g,cm = components()
    for key in ('time','observation'):
        p = copy.deepcopy(prereg())
        if key=='time': p['environment']['t_max_s'] = 45.
        else: p['implementation_versions']['observation'] = 'obs-v7'
        with pytest.raises(ValueError):
            PhysicalHorizon90(spec,g,cm,[actual(spec)],[550.],p)
