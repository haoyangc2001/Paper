import math

import pytest

from camra_calibration.commands import build_program
from camra_calibration.nominal import (
    CROSSCHECK_TOL,
    PlanarState,
    integrate,
    integrate_euler,
    integrate_rk4,
    max_planar_deviation,
    step_exact,
    wrap_angle,
)


def constant(cmd, n_points):
    return [tuple(cmd)] * n_points


def test_straight_line_forward():
    traj = integrate(PlanarState(0, 0, 0), constant((0.5, 0, 0), 301), 0.1)
    assert traj[-1].px == pytest.approx(15.0, abs=1e-12)
    assert traj[-1].py == pytest.approx(0.0, abs=1e-12)
    assert traj[-1].yaw == 0.0


def test_straight_line_lateral():
    traj = integrate(PlanarState(0, 0, 0), constant((0, 0.3, 0), 301), 0.1)
    assert traj[-1].px == pytest.approx(0.0, abs=1e-12)
    assert traj[-1].py == pytest.approx(9.0, abs=1e-12)


def test_zero_command_keeps_state():
    traj = integrate(PlanarState(1.0, -2.0, 0.7), constant((0, 0, 0), 301), 0.1)
    for s in traj:
        assert (s.px, s.py, s.yaw) == (1.0, -2.0, 0.7)


def test_non_zero_initial_heading_rotates_velocity():
    traj = integrate(PlanarState(0, 0, math.pi / 2), constant((0.5, 0, 0), 301), 0.1)
    assert traj[-1].px == pytest.approx(0.0, abs=1e-12)
    assert traj[-1].py == pytest.approx(15.0, abs=1e-12)


def test_circle_closes_after_full_turn():
    dt = math.pi / 10
    traj = integrate(PlanarState(0, 0, 0), constant((1.0, 0, 1.0), 21), dt)
    assert traj[-1].px == pytest.approx(0.0, abs=1e-12)
    assert traj[-1].py == pytest.approx(0.0, abs=1e-12)
    assert traj[-1].yaw == pytest.approx(2 * math.pi, abs=1e-12)
    # vx=1、wz=1 时圆心在机身左侧 (0, 1)，半径 = vx/wz = 1
    center = (0.0, 1.0)
    for s in traj:
        assert math.hypot(s.px - center[0], s.py - center[1]) == pytest.approx(1.0, abs=1e-12)
    quarter = traj[5]  # yaw = pi/2 时到达 (1, 1)
    assert (quarter.px, quarter.py) == (pytest.approx(1.0, abs=1e-12), pytest.approx(1.0, abs=1e-12))


def test_exact_matches_rk4_on_a_real_program():
    p = build_program("arc_combined", 6000)
    init = PlanarState(0.3, -0.4, 0.5)
    exact = integrate(init, p.commands, 0.1)
    rk4 = integrate_rk4(init, p.commands, 0.1, substeps=50)
    assert max_planar_deviation(exact, rk4) < CROSSCHECK_TOL


def test_euler_error_converges_first_order():
    """一阶 Euler 的误差应随步长减半而减半（阶数 ≈ 1），证明闭式解没有被某个固定偏置污染。"""
    p = build_program("arc_combined", 6000)
    init = PlanarState(0.0, 0.0, 0.0)
    exact = integrate(init, p.commands, 0.1)
    coarse = max_planar_deviation(exact, integrate_euler(init, p.commands, 0.1, substeps=50))
    fine = max_planar_deviation(exact, integrate_euler(init, p.commands, 0.1, substeps=100))
    assert coarse > fine > 0.0
    ratio = coarse / fine
    assert 1.7 < ratio < 2.3, ratio


def test_perturbation_cannot_change_nominal():
    # 名义轨迹只接受命令与初态；同命令必然同轨迹（扰动参数根本不在签名里）
    p = build_program("straight_shift", 7000)
    a = integrate(PlanarState(0, 0, 0), p.commands, 0.1)
    b = integrate(PlanarState(0, 0, 0), p.commands, 0.1)
    assert [s.as_tuple() for s in a] == [s.as_tuple() for s in b]


def test_step_exact_is_associative_over_two_steps():
    s0 = PlanarState(0.2, 0.1, -0.3)
    cmd = (0.7, -0.2, 0.6)
    one = step_exact(step_exact(s0, cmd, 0.1), cmd, 0.1)
    two = integrate(s0, [cmd, cmd, cmd], 0.1)[-1]
    assert one.px == pytest.approx(two.px, abs=1e-12)
    assert one.py == pytest.approx(two.py, abs=1e-12)
    assert one.yaw == pytest.approx(two.yaw, abs=1e-12)


def test_wrap_angle():
    assert wrap_angle(0.0) == 0.0
    assert wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-12)


def test_integrate_requires_two_points():
    with pytest.raises(ValueError):
        integrate(PlanarState(0, 0, 0), [(0, 0, 0)], 0.1)


def test_deviation_length_mismatch_rejected():
    with pytest.raises(ValueError):
        max_planar_deviation([PlanarState(0, 0, 0)], [])
