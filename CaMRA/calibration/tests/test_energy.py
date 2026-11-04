import math

import numpy as np
import pytest

from camra_calibration.energy import (
    DEFAULT_DT,
    EnergyMeter,
    integrate_series,
    positive_joint_work,
    reference_naive_sum_then_clip,
    step_energy,
    steps_for,
)

P = 2.0  # 单关节功率 W


def const_tau_qdot(n_steps: int, tau: float = 1.0, qdot: float = 2.0, n_joints: int = 12):
    tau_arr = np.full((n_steps, n_joints), tau, dtype=np.float64)
    qdot_arr = np.full((n_steps, n_joints), qdot, dtype=np.float64)
    return tau_arr, qdot_arr


# ---------------- 独立解析参考 ----------------
def test_constant_power_matches_analytic():
    n, j = 100, 12
    tau, qdot = const_tau_qdot(n, tau=1.0, qdot=2.0, n_joints=j)
    expect = n * j * 1.0 * 2.0 * DEFAULT_DT
    assert integrate_series(tau, qdot) == pytest.approx(expect, rel=1e-15)
    meter = EnergyMeter(expected_steps=n)
    meter.start()
    for k in range(n):
        meter.add_step(tau[k], qdot[k])
    assert meter.total_j == pytest.approx(expect, rel=1e-12)


def test_alternating_sign_counts_only_positive_half():
    n = 100
    tau = np.full((n, 1), 1.0)
    qdot = np.array([[2.0 if k % 2 == 0 else -2.0] for k in range(n)])
    expect = (n // 2) * 1.0 * 2.0 * DEFAULT_DT
    assert integrate_series(tau, qdot) == pytest.approx(expect, rel=1e-15)
    # 正负交替只剩一半步数计功，因此恰为"全正"情形的一半
    assert integrate_series(tau, qdot) == pytest.approx(
        0.5 * integrate_series(tau, np.abs(qdot)), rel=1e-15)


def test_inter_joint_cancellation_is_not_netting():
    # 关节 0 出力 +2 W，关节 1 吸收 -2 W（同一时刻）
    tau = np.array([[1.0, 1.0]])
    qdot = np.array([[2.0, -2.0]])
    clipped = integrate_series(tau, qdot)
    naive = reference_naive_sum_then_clip(tau, qdot)
    assert clipped == pytest.approx(2.0 * DEFAULT_DT, rel=1e-15)  # 只计输出功
    assert naive == pytest.approx(0.0, abs=1e-18)                 # 先求和会被抵消
    assert clipped > naive


def test_negative_work_never_subtracts():
    tau = np.full((10, 4), 5.0)
    qdot = np.full((10, 4), -1.0)
    assert integrate_series(tau, qdot) == 0.0
    meter = EnergyMeter(n_joints=4)
    meter.start()
    for _ in range(10):
        assert meter.add_step(tau[0], qdot[0]) == 0.0
    assert meter.total_j == 0.0


def test_non_integer_window_is_rejected_or_exact():
    # 2.5 s 在 dt=0.006 下不是整数倍 → 必须报错，不能静默取整
    with pytest.raises(ValueError):
        steps_for(2.5, dt=0.006)
    # dt=0.006 且步数取整后，能量必须用实际步数×dt
    meter = EnergyMeter(dt=0.006, n_joints=1)
    meter.start()
    for _ in range(417):
        meter.add_step([1.0], [1.0])
    assert meter.steps == 417
    assert meter.total_j == pytest.approx(417 * 0.006, rel=1e-15)
    assert meter.elapsed_s == pytest.approx(2.502, rel=1e-15)


def test_unit_conversion_scales_energy():
    # 力矩用 mN·m 给出（×1e-3 换算到 N·m），结果应等于 J 口径
    tau_nm, qdot = const_tau_qdot(10, tau=1.0, qdot=2.0, n_joints=3)
    tau_mnm = tau_nm * 1000.0
    j_nm = integrate_series(tau_nm, qdot)
    j_mnm = integrate_series(tau_mnm, qdot, torque_scale=1e-3)
    assert j_mnm == pytest.approx(j_nm, rel=1e-12)
    assert j_nm == pytest.approx(10 * 3 * 2.0 * DEFAULT_DT, rel=1e-15)


def test_velocity_unit_conversion():
    tau, qdot = const_tau_qdot(5, tau=2.0, qdot=1.0, n_joints=2)
    # 速度用 deg/s 给出（π/180 换算到 rad/s）
    deg = qdot * (180.0 / math.pi)
    assert integrate_series(tau, deg, velocity_scale=math.pi / 180.0) == pytest.approx(
        integrate_series(tau, qdot), rel=1e-12)


# ---------------- 计量器行为 ----------------
def test_meter_requires_start():
    meter = EnergyMeter()
    with pytest.raises(RuntimeError):
        meter.add_step([0.0] * 12, [0.0] * 12)


def test_meter_rejects_steps_after_stop():
    meter = EnergyMeter()
    meter.start()
    meter.add_step([1.0] * 12, [1.0] * 12)
    meter.stop()
    with pytest.raises(RuntimeError):
        meter.add_step([1.0] * 12, [1.0] * 12)


def test_meter_rejects_wrong_dimension_and_non_finite():
    meter = EnergyMeter(n_joints=12)
    meter.start()
    with pytest.raises(ValueError):
        meter.add_step([1.0] * 11, [1.0] * 12)
    with pytest.raises(ValueError):
        meter.add_step([float("nan")] * 12, [1.0] * 12)
    with pytest.raises(ValueError):
        meter.add_step([float("inf")] * 12, [1.0] * 12)


def test_meter_start_resets_state():
    meter = EnergyMeter(n_joints=2)
    meter.start()
    meter.add_step([1.0, 1.0], [1.0, 1.0])
    first = meter.total_j
    assert first > 0
    meter.start()
    assert meter.total_j == 0.0 and meter.steps == 0


def test_meter_matches_offline_recompute_bitwise():
    rng = np.random.default_rng(0)
    n = 500
    tau = rng.normal(0, 8, size=(n, 12))
    qdot = rng.normal(0, 10, size=(n, 12))
    meter = EnergyMeter(expected_steps=n)
    meter.start()
    for k in range(n):
        meter.add_step(tau[k], qdot[k])
    offline = integrate_series(tau, qdot)
    assert meter.total_j == pytest.approx(offline, rel=1e-12, abs=1e-12)
    assert meter.verify_complete() == []


def test_meter_detects_incomplete_window():
    meter = EnergyMeter(expected_steps=6000)
    meter.start()
    for _ in range(10):
        meter.add_step([1.0] * 12, [1.0] * 12)
    problems = meter.verify_complete()
    assert problems and "6000" in problems[0]
    assert meter.summary()["complete"] is False


def test_full_window_semantics_6000_steps():
    meter = EnergyMeter(expected_steps=6000)
    meter.start()
    for _ in range(6000):
        meter.add_step([0.5] * 12, [1.0] * 12)
    assert meter.steps == 6000
    assert meter.elapsed_s == pytest.approx(30.0, rel=1e-15)
    assert meter.total_j == pytest.approx(6000 * 12 * 0.5 * 0.005 * 1.0, rel=1e-12)
    assert meter.verify_complete() == []


def test_positive_joint_work_shape_and_clipping():
    w = positive_joint_work([1.0, -1.0, 2.0], [2.0, 2.0, -1.0])
    assert w == [2.0, 0.0, 0.0]
    assert step_energy([1.0, -1.0], [2.0, 2.0], dt=0.5) == pytest.approx(1.0)
