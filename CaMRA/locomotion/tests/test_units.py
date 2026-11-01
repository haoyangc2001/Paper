"""Unit tests for the TASK-018/019/020 contracts (TASK-021).

CPU-only and simulator-free: these tests use synthetic fixtures and verify the
*contracts* (axes, units, clipping, invalid input, metric arithmetic, threshold
judgement). They are explicitly NOT controller results -- see
``SYNTHETIC_FIXTURE_NOTE``.

Run:
    python -m pytest tests/test_units.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "unitree_rl_lab"))

from unitree_rl_lab.utils.eval_metrics import (  # noqa: E402
    FIELDS,
    I,
    NF,
    aggregate,
    compute_metrics,
)
from unitree_rl_lab.utils.external_command import (  # noqa: E402
    DEFAULT_LIMITS,
    FixedProgram,
    RampProgram,
    SineProgram,
    StepProgram,
    clip_to_limits,
)

SYNTHETIC_FIXTURE_NOTE = (
    "All fixtures here are synthetic. They verify code behaviour, not controller performance; "
    "no number produced by this file may be reported as a locomotion result."
)

T = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731


# ---------------------------------------------------------------------------
# axes / units
# ---------------------------------------------------------------------------


def test_fixed_program_axis_order_and_units():
    """Channel order is [vx, vy, wz]; values pass through unchanged (m/s, rad/s)."""
    prog = FixedProgram([0.5, -0.2, 0.3])
    out = prog.at(T([0.0, 1.0, 2.0]))
    assert out.shape == (3, 3)
    assert torch.allclose(out[:, 0], T([0.5, 0.5, 0.5]))
    assert torch.allclose(out[:, 1], T([-0.2, -0.2, -0.2]))
    assert torch.allclose(out[:, 2], T([0.3, 0.3, 0.3]))
    assert out.dtype == torch.float32


def test_step_program_switches_on_time_grid():
    prog = StepProgram(
        [
            {"duration": 1.0, "vel": [0.0, 0.0, 0.0]},
            {"duration": 1.0, "vel": [0.4, 0.0, 0.0]},
            {"duration": 1.0, "vel": [0.9, 0.0, 0.0]},
        ]
    )
    # last segment holds beyond its end time
    out = prog.at(T([0.0, 0.99, 1.0, 1.99, 2.0, 5.0]))
    assert [round(float(v), 3) for v in out[:, 0]] == [0.0, 0.0, 0.4, 0.4, 0.9, 0.9]


def test_ramp_program_endpoints_and_slope():
    prog = RampProgram([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], duration=2.0)
    out = prog.at(T([0.0, 1.0, 2.0, 3.0]))
    assert [round(float(v), 4) for v in out[:, 0]] == [0.0, 0.5, 1.0, 1.0]
    # slope = 0.5 per second on the vx channel only
    assert float(out[1, 1]) == 0.0 and float(out[1, 2]) == 0.0


def test_sine_program_peaks_and_base():
    prog = SineProgram(axis=2, amplitude=0.25, frequency=1.0, base=[0.6, 0.0, 0.0])
    out = prog.at(T([0.0, 0.25, 0.75]))
    assert float(out[0, 0]) == pytest.approx(0.6)  # base untouched on vx
    assert float(out[1, 2]) == pytest.approx(0.25, abs=1e-6)  # +peak at quarter period
    assert float(out[2, 2]) == pytest.approx(-0.25, abs=1e-6)  # -peak at three quarters


def test_command_dimension_is_three():
    assert len(DEFAULT_LIMITS) == 3
    assert [list(x) for x in DEFAULT_LIMITS] == [[-1.0, 1.0], [-0.4, 0.4], [-1.0, 1.0]]


# ---------------------------------------------------------------------------
# clipping / invalid input
# ---------------------------------------------------------------------------


def test_clip_to_limits_clips_each_axis_independently():
    clipped, changed, invalid = clip_to_limits(T([[2.0, -2.0, 5.0]]), DEFAULT_LIMITS)
    assert clipped[0].tolist() == pytest.approx([1.0, -0.4, 1.0], abs=1e-6)
    assert bool(changed[0]) and not bool(invalid[0])


def test_clip_to_limits_leaves_in_range_values_untouched():
    cmd = T([[0.5, -0.2, 0.3], [1.0, 0.4, -1.0]])  # second row sits exactly on the boundary
    clipped, changed, invalid = clip_to_limits(cmd, DEFAULT_LIMITS)
    assert torch.equal(clipped, cmd)
    assert not bool(changed.any()) and not bool(invalid.any())


def test_non_finite_command_becomes_zero_and_is_flagged():
    cmd = T([[0.5, float("nan"), 0.0], [float("inf"), 0.0, 0.0]])
    clipped, changed, invalid = clip_to_limits(cmd, DEFAULT_LIMITS)
    assert clipped.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert bool(invalid.all())
    assert not bool(changed.any())  # invalid is reported separately from "clipped"


def test_wrong_command_width_rejected():
    with pytest.raises(ValueError):
        clip_to_limits(T([[0.1, 0.2]]), DEFAULT_LIMITS)
    with pytest.raises(ValueError):
        FixedProgram([1.0, 2.0])


def test_empty_or_invalid_program_definitions_rejected():
    with pytest.raises(ValueError):
        StepProgram([])
    with pytest.raises(ValueError):
        SineProgram(axis=5, amplitude=0.1, frequency=1.0)
    with pytest.raises(ValueError):
        RampProgram([0, 0, 0], [1, 0, 0], duration=0.0)


# ---------------------------------------------------------------------------
# metric arithmetic on known error sequences
# ---------------------------------------------------------------------------


def make_episode(n: int, vx_err: float = 0.1, cmd_vx: float = 0.6, terminate_at: int | None = None) -> np.ndarray:
    """Synthetic step block: actual = cmd + constant error, optional termination."""
    ep = np.zeros((n, NF), dtype=np.float64)
    dt = 0.02
    for k in range(n):
        ep[k, I["sim_time"]] = 10.0 + k * dt
        ep[k, I["episode_step"]] = k + 1
        ep[k, I["cmd_vx"]] = cmd_vx
        ep[k, I["app_vx"]] = cmd_vx
        ep[k, I["vx_body"]] = cmd_vx + vx_err
        ep[k, I["pos_x"]] = 0.01 * k
    if terminate_at is not None:
        ep[terminate_at, I["terminated"]] = 1.0
        ep[terminate_at, I["reason_id"]] = 2.0  # bad_orientation
    else:
        ep[-1, I["reason_id"]] = 4.0  # scenario_end
    return ep


SPEC = {
    "id": "unit",
    "duration_s": 1.0,
    "metrics": ["rmse_vx", "p95_abs_err_vx", "failure_rate"],
    "thresholds": {"rmse_vx": 0.15, "p95_abs_err_vx": 0.25},
}


def test_rmse_and_p95_on_constant_error():
    """actual - cmd == 0.1 for every step -> rmse == 0.1 and p95 == 0.1 exactly."""
    ep = make_episode(100, vx_err=0.1)
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=0.0)
    assert m["rmse_vx"] == pytest.approx(0.1, abs=1e-12)
    assert m["p95_abs_err_vx"] == pytest.approx(0.1, abs=1e-12)


def test_rmse_matches_analytic_value_for_two_levels():
    """Half the steps at +0.1, half at -0.3 -> rmse = sqrt((0.01+0.09)/2) = 0.2236."""
    n = 100
    ep = make_episode(n, vx_err=0.0)
    ep[: n // 2, I["vx_body"]] = 0.6 + 0.1
    ep[n // 2 :, I["vx_body"]] = 0.6 - 0.3
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=0.0)
    assert m["rmse_vx"] == pytest.approx(np.sqrt((0.1**2 + 0.3**2) / 2), abs=1e-12)


def test_p95_ignores_the_isolated_spike():
    """One 5.0 outlier in 100 steps stays above p95 -> p95 remains near the bulk error."""
    ep = make_episode(100, vx_err=0.1)
    ep[50, I["vx_body"]] = 0.6 + 5.0
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=0.0)
    assert m["p95_abs_err_vx"] == pytest.approx(0.1, abs=1e-9)
    assert m["rmse_vx"] > 0.4  # rmse is NOT robust: it feels the spike


def test_warmup_window_is_excluded():
    """A large error confined to the first 1.0 s must not enter the metrics."""
    ep = make_episode(200, vx_err=0.1)
    ep[:50, I["vx_body"]] = 5.0  # samples 1..50 == the first 1.0 s, all excluded
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=1.0)
    assert m["rmse_vx"] == pytest.approx(0.1, abs=1e-9)


def test_early_termination_statistics():
    ep = make_episode(100, terminate_at=40)
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=0.0)
    assert m["failed"] == 1
    assert m["ended_by"] == "bad_orientation"
    assert m["steps_before_end"] == 41  # argmax+1


def test_scenario_end_is_not_a_failure():
    ep = make_episode(100)
    m = compute_metrics(ep, SPEC, 0.02, warmup_s=0.0)
    assert m["failed"] == 0
    assert m["ended_by"] == "scenario_end"


# ---------------------------------------------------------------------------
# aggregation: single vs batch, relaxed thresholds
# ---------------------------------------------------------------------------


def test_aggregate_single_episode_equals_its_own_value():
    rows = [compute_metrics(make_episode(100, vx_err=0.1), SPEC, 0.02, 0.0)]
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["episodes"] == 1
    assert agg["rmse_vx"] == pytest.approx(rows[0]["rmse_vx"])
    assert agg["pass"] is True


def test_aggregate_batch_mean_and_pass_flag():
    rows = [compute_metrics(make_episode(100, vx_err=e), SPEC, 0.02, 0.0) for e in (0.05, 0.10, 0.30)]
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["episodes"] == 3
    assert agg["rmse_vx"] == pytest.approx(np.mean([0.05, 0.10, 0.30]))
    # mean lands exactly on the 0.15 threshold; "<= threshold" must pass
    assert agg["rmse_vx"] == pytest.approx(0.15)
    assert agg["pass"] is True
    # one step over the line must fail
    rows_over = [compute_metrics(make_episode(100, vx_err=e), SPEC, 0.02, 0.0) for e in (0.10, 0.30)]
    assert aggregate(rows_over, SPEC, "nominal")["pass"] is False


def test_perturbed_relaxes_only_rmse_thresholds():
    """rmse limits get x1.2; p95 and failure-rate limits do not."""
    rows = [compute_metrics(make_episode(100, vx_err=0.17), SPEC, 0.02, 0.0)]  # 0.17 > 0.15
    nominal = aggregate(rows, SPEC, "nominal")
    perturbed = aggregate(rows, SPEC, "perturbed")
    assert nominal["checks"]["rmse_vx"]["pass"] is False
    assert perturbed["checks"]["rmse_vx"]["pass"] is True  # 0.15*1.2 = 0.18
    assert perturbed["checks"]["rmse_vx"]["threshold"] == pytest.approx(0.18)
    assert perturbed["checks"]["p95_abs_err_vx"]["threshold"] == pytest.approx(0.25)  # not relaxed


def test_failure_rate_thresholds_per_group():
    failing = [compute_metrics(make_episode(100, terminate_at=20), SPEC, 0.02, 0.0) for _ in range(10)]
    nominal = aggregate(failing, SPEC, "nominal")
    perturbed = aggregate(failing, SPEC, "perturbed")
    assert nominal["failure_rate"] == pytest.approx(1.0)
    assert nominal["checks"]["failure_rate"]["threshold"] == pytest.approx(0.05)
    assert perturbed["checks"]["failure_rate"]["threshold"] == pytest.approx(0.10)


def test_field_layout_is_stable():
    """Guards against silently reordering the step record (npz/CSV compatibility)."""
    assert NF == 56
    assert FIELDS[0] == "sim_time"
    assert FIELDS[I["cmd_vx"] : I["cmd_vx"] + 3] == ["cmd_vx", "cmd_vy", "cmd_wz"]
    assert FIELDS[I["joint_pos_0"] : I["joint_pos_0"] + 12] == [f"joint_pos_{i}" for i in range(12)]
    assert FIELDS[-3:] == ["terminated", "truncated", "reason_id"]
