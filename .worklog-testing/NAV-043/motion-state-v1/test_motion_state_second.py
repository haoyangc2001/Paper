"""Basic causal-state checks with synthetic poses; no fitting or simulator."""
import math

import numpy as np
import pytest

from camra_calibration.nominal import PlanarState, step_exact
from camra_navigation import observation as O
from camra_navigation.horizon import PROFILE, encode_observation
from camra_navigation.motion_state import (
    CompletedInterval, CompletedMotionTracker, OBSERVATION_VERSION, PoseSample,
    STATE_VERSION, encode_completed_observation, require_model_binding)


def motion(q=(.3, -.1, .4), *, initial=(1., 2., 3.13), tick=180):
    end = step_exact(PlanarState(*initial), q, .1)
    return CompletedInterval(PoseSample(tick, initial), PoseSample(tick+20,
        (end.px, end.py, math.atan2(math.sin(end.yaw), math.cos(end.yaw)))))


@pytest.mark.parametrize('q',[(0.,0.,0.),(.3,0.,0.),(.3,-.1,.4),(-.2,.1,-.5),(.3,.1,1.e-8)])
def test_se2_log_recovers_known_frozen_arc_including_wrapped_heading(q):
    np.testing.assert_allclose(motion(q).q, q, atol=1.e-12, rtol=0.)


@pytest.mark.parametrize('count',[0,1,19,21])
def test_partial_or_wrong_length_interval_is_rejected(count):
    with pytest.raises(ValueError,match='completed twenty-tick'):
        CompletedInterval(PoseSample(180,(0.,0.,0.)),PoseSample(180+count,(0.,0.,0.)))


def test_nonfinite_pose_bad_tick_and_ambiguous_turn_rejected():
    for pose in ((0.,float('nan'),0.),(0.,0.),(0.,0.,float('inf'))):
        with pytest.raises(ValueError):
            PoseSample(180,pose)
    for tick in (-1,180.5,True):
        with pytest.raises(ValueError):
            PoseSample(tick,(0.,0.,0.))
    with pytest.raises(ValueError,match='ambiguous'):
        CompletedInterval(PoseSample(180,(0.,0.,0.)),PoseSample(200,(0.,0.,math.pi/2)))


def test_tracker_never_uses_future_or_partial_interval_as_current_q():
    warmup = motion()
    tracker = CompletedMotionTracker(warmup.start,warmup.end)
    assert tracker.for_decision() == warmup
    pose = warmup.end.pose
    for tick in range(201,221):
        end = step_exact(PlanarState(*pose),(.2,.1,-.3),.005)
        pose = end.px,end.py,end.yaw
        tracker.advance(PoseSample(tick,pose))
        if tick < 220:
            assert tracker.last_completed == warmup
            with pytest.raises(ValueError,match='completed'):
                tracker.for_decision()
    np.testing.assert_allclose(tracker.for_decision().q,(.2,.1,-.3),atol=1.e-12)


def test_tracker_rejects_gaps_and_keeps_terminal_partial_tail_without_padding():
    warmup = motion()
    tracker = CompletedMotionTracker(warmup.start,warmup.end)
    for tick in (200,202):
        with pytest.raises(ValueError,match='physics sample'):
            tracker.advance(PoseSample(tick,warmup.end.pose))
    tracker.advance(PoseSample(201,warmup.end.pose),terminal=True)
    assert tracker.partial_ticks == 1 and tracker.last_completed == warmup and tracker.closed
    with pytest.raises(ValueError):
        tracker.for_decision()
    with pytest.raises(ValueError,match='terminated'):
        tracker.advance(PoseSample(202,warmup.end.pose))


def test_no_policy_decision_at_ninety_seconds_even_if_caller_omits_terminal_flag():
    warmup = motion((0.,0.,0.))
    tracker = CompletedMotionTracker(warmup.start,warmup.end)
    for tick in range(201,18201):
        tracker.advance(PoseSample(tick,warmup.end.pose))
    with pytest.raises(ValueError,match='live completed'):
        tracker.for_decision()
    with pytest.raises(ValueError,match='ninety-second'):
        tracker.advance(PoseSample(18201,warmup.end.pose))


def encode(interval, **overrides):
    arguments = dict(decision_tick=200,window_start_tick=200,obstacles=(),
                     extent=O.MapExtent(-20.,20.,-20.,20.),
                     task=O.TaskState(7.,0.,0.,1100.,90.),inflation_radius_m=1.)
    arguments.update(overrides)
    return encode_completed_observation(interval,**arguments)


def test_candidate_observation_encodes_completed_q_and_preserves_map_budget_fields():
    interval = motion()
    row = encode(interval)
    expected = encode_observation(O.EgoState(*interval.end.pose,*interval.q),(),
        O.MapExtent(-20.,20.,-20.,20.),O.TaskState(7.,0.,0.,1100.,90.),inflation_radius_m=1.)
    np.testing.assert_array_equal(row,expected)
    np.testing.assert_allclose(row[2:5],interval.q,atol=1.e-12)
    assert len(row) == 61 and row[-2:] == [2.,1.]
    assert O.SUPPORTED_OBS_VERSIONS[-1] == 'obs-v7' and PROFILE.observation_version == 'obs-v8-t90'


@pytest.mark.parametrize('overrides',[
    {'decision_tick':199},{'decision_tick':201},{'window_start_tick':201},
    {'window_start_tick':0},{'task':O.TaskState(7.,0.,0.,550.,45.)}])
def test_candidate_observation_rejects_future_stale_or_inconsistent_clock(overrides):
    with pytest.raises(ValueError):
        encode(motion(),**overrides)


def test_same_shape_old_checkpoint_and_changed_motion_identity_are_rejected():
    digest = '0'*64  # synthetic identity, not a deployment model
    binding = dict(time_profile=PROFILE.profile_id,t_max_s=90.,
        observation_version=OBSERVATION_VERSION,motion_state_version=STATE_VERSION,
        motion_model_sha256=digest)
    require_model_binding(binding,motion_model_sha256=digest)
    for key,value in (('observation_version','obs-v8-t90'),('observation_version','obs-v7'),
                      ('motion_state_version','instantaneous'),('motion_model_sha256','1'*64),
                      ('t_max_s',45.)):
        with pytest.raises(ValueError,match='mismatch'):
            require_model_binding({**binding,key:value},motion_model_sha256=digest)
    with pytest.raises(ValueError,match='identity'):
        require_model_binding(binding,motion_model_sha256='')
