"""Synthetic validation only; no fit on any of the eighteen real traces."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import nav043_affine_response as m


def test_affine_bias_is_identifiable_separately_from_gain():
    x = np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[-1.,2.,1.]])
    result = m.fit_axis(x,x@np.array([.5,.3,-.02]))
    assert result['valid']
    np.testing.assert_allclose([result[k] for k in ('a','b','d')],[.5,.3,-.02],atol=1.e-12)


@pytest.mark.parametrize('theta',[[1.1,.2,0.],[-.1,.2,0.],[.5,-.2,0.]])
def test_bad_dynamics_are_rejected_instead_of_clipped(theta):
    x = np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[-1.,2.,1.]])
    assert not m.fit_axis(x,x@theta)['valid']


def test_unidentified_model_rejected():
    assert not m.fit_axis(np.ones((10,3)),np.ones(10))['valid']


def test_zero_intercept_exactly_preserves_v1_replay_including_partial_tail():
    trace = SimpleNamespace(initial=np.array([1.,2.,.3]),ticks=45,
        interval_commands=np.array([[.4,.1,.2],[.5,-.1,-.2],[.6,0.,.1]]))
    parameters = np.array([[.5,.4],[.2,.6],[.4,.3]])
    old = m.base.free_replay(trace,parameters)
    new = m.free_replay(trace,np.column_stack((parameters,np.zeros(3))))
    for a,b in zip(old,new):
        np.testing.assert_array_equal(a,b)
    assert len(new[0])==45


def test_free_replay_does_not_read_truth():
    trace = SimpleNamespace(initial=np.zeros(3),ticks=21,
        interval_commands=np.zeros((2,3)))
    poses,responses = m.free_replay(trace,np.array([[.5,.3,.01]]*3))
    assert poses.shape==(21,3) and responses.shape==(2,3)
    np.testing.assert_allclose(responses[0],[.01]*3)


def test_real_execution_requires_new_approval_before_reading_data():
    with patch.object(m,'APPROVAL',Path('/does-not-exist/nav043-affine-approval')):
        with pytest.raises(ValueError,match='separate approved'):
            m.approval()


def fake_gate_rows(position_ratios,heading_ratio=.5):
    old = {'gate':{'all_heldout_response_RMSE_m':2.},'rows':[]}
    new = []
    for level,ratio in zip(m.base.LEVELS,position_ratios):
        metric = {'ticks':20,'position_squared_error_sum':80.,'heading_terminal_rad':1.}
        old['rows'].append({'level':level,'terminal':{'response':metric}})
        new.append({'level':level,'metrics':{**metric,
            'position_squared_error_sum':80.*ratio**2,'heading_terminal_rad':heading_ratio}})
    return new,old


def test_good_aggregate_cannot_hide_one_worse_map():
    result = m.gate(*fake_gate_rows([.1,.1,1.01]))
    assert result['checks']['total_position_RMSE_at_least_15pct_lower']
    assert not result['checks']['no_heldout_level_worse']
    assert not result['diagnostic_pass']


def test_position_improvement_cannot_hide_heading_failure():
    result = m.gate(*fake_gate_rows([.5]*3,heading_ratio=.9))
    assert result['checks']['no_heldout_level_worse']
    assert not result['diagnostic_pass']


def test_diagnostic_pass_never_marks_calibration_or_G5_pass():
    result = m.gate(*fake_gate_rows([.5]*3))
    assert result['diagnostic_pass']
    assert not any(result[k] for k in ('calibration_pass','NAV043_passed','G5_passed'))
