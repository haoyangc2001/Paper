"""Counterexamples for saved residual accounting, without real fits/rollouts."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_nav043_response_residuals as m


def test_identical_curved_poses_have_zero_accounting():
    initial = np.array([2.,-1.,.3])
    actual = np.array([m.m.step(initial,[.4,.2,.5],t) for t in (.005,.01,.015)])
    h,b,error,closure = m.decompose(initial,actual,actual)
    np.testing.assert_allclose([h,b,error],0.,atol=1.e-14)
    assert closure<1.e-14


def test_straight_speed_error_has_only_body_term():
    initial = np.zeros(3)
    actual = np.array([[1.,0.,0.],[2.,0.,0.]])
    predicted = np.array([[2.,0.,0.],[4.,0.,0.]])
    h,b,error,_ = m.decompose(initial,actual,predicted)
    np.testing.assert_allclose(h,0.)
    np.testing.assert_allclose(b,error)


def test_yaw_and_speed_terms_can_cancel_so_norms_are_not_shares():
    initial = np.zeros(3)
    actual = np.array([[1.,0.,0.],[2.,0.,0.]])
    predicted = np.array([[1.,0.,np.pi/2],[2.,0.,np.pi/2]])
    h,b,error,closure = m.decompose(initial,actual,predicted)
    np.testing.assert_allclose(h[-1],-b[-1],atol=1.e-14)
    assert np.linalg.norm(h[-1])>1.
    assert np.linalg.norm(error[-1])==0.
    metrics = m.accounting_metrics(h,b,error,closure)
    assert metrics['terminal_cross_term_m2']<0.


def test_wrapped_actual_yaw_does_not_break_world_identity():
    initial = np.array([4.,-2.,np.pi-.01])
    actual = np.array([m.m.step(initial,[.4,.1,.8],t) for t in (.1,.2,.3)])
    predicted = np.array([m.m.step(initial,[.5,-.1,.3],t) for t in (.1,.2,.3)])
    actual[:,2] = m.m.wrap(actual[:,2])
    h,b,error,closure = m.decompose(initial,actual,predicted)
    np.testing.assert_allclose(h+b,error,atol=1.e-12)
    assert closure<1.e-12


def test_prefix_accounting_cannot_use_future_poses():
    initial = np.zeros(3)
    actual = np.array([[.1,.2,.3],[.4,.5,.6]])
    predicted = np.array([[.5,.4,.3],[.2,.1,0.]])
    a = m.decompose(initial,actual[:1],predicted[:1])
    b = m.decompose(initial,actual,predicted)
    for x,y in zip(a[:3],b[:3]):
        np.testing.assert_array_equal(x,y[:1])


@pytest.mark.parametrize('actual,predicted',[(np.empty((0,3)),np.empty((0,3))),
    (np.zeros((2,3)),np.zeros((1,3))),(np.full((1,3),np.nan),np.zeros((1,3)))])
def test_bad_or_missing_states_are_rejected(actual,predicted):
    with pytest.raises(ValueError):
        m.decompose(np.zeros(3),actual,predicted)
