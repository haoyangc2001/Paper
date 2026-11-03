import math

import pytest

from camra_calibration.protocol import Protocol
from camra_calibration.timeline import Timeline


def tl() -> Timeline:
    return Timeline.from_protocol(Protocol.load())


def test_counts_match_protocol():
    t = tl()
    assert t.physics_intervals == 6000
    assert t.policy_intervals == 1500
    assert t.command_intervals == 300
    assert t.pairing_intervals == 300
    assert t.pairing_points == 301
    assert t.warmup_intervals == 200


def test_all_rates_share_the_same_window():
    t = tl()
    assert math.isclose(t.physics_intervals * t.sim_dt, 30.0, abs_tol=1e-12)
    assert math.isclose(t.policy_intervals * t.policy_dt, 30.0, abs_tol=1e-12)
    assert math.isclose(t.command_intervals * t.command_dt, 30.0, abs_tol=1e-12)
    assert math.isclose(t.warmup_intervals * t.sim_dt, 1.0, abs_tol=1e-12)
    assert t.check() == []


def test_integer_ratios():
    assert tl().ratios() == {"physics_per_policy": 4, "policy_per_command": 5, "physics_per_command": 20}


def test_endpoints_are_exact():
    t = tl()
    assert t.physics_time(6000) == pytest.approx(30.0, abs=1e-12)
    assert t.policy_time(1500) == pytest.approx(30.0, abs=1e-12)
    assert t.command_time(300) == pytest.approx(30.0, abs=1e-12)
    assert t.command_time(0) == 0.0
    assert t.physics_time(0) == 0.0


def test_index_bounds_enforced():
    t = tl()
    with pytest.raises(IndexError):
        t.command_time(301)
    with pytest.raises(IndexError):
        t.physics_time(6001)
    with pytest.raises(IndexError):
        t.policy_time(-1)


def test_non_divisible_window_rejected():
    t = Timeline(sim_dt=0.007, policy_dt=0.02, command_dt=0.1, pairing_dt=0.1,
                 warmup_s=1.0, measured_window_s=30.0)
    with pytest.raises(ValueError):
        _ = t.physics_intervals
