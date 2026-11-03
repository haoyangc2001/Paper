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


# ---------------------------------------------------------------------------
# TASK-023 tuning round 4: stop-event command and stop-drift shaping
# ---------------------------------------------------------------------------
#
# ``stop_events.py`` is deliberately free of isaaclab imports so the state
# machine below can be exercised on CPU without ``omni``.  It is loaded from
# file so that importing the ``mdp`` package (which pulls in isaaclab) is not
# required.

import importlib.util  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_stop_events():
    path = os.path.join(
        _HERE, "..", "source", "unitree_rl_lab", "unitree_rl_lab", "tasks", "locomotion", "mdp", "stop_events.py"
    )
    spec = importlib.util.spec_from_file_location("camra_stop_events", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SE = _load_stop_events()


def test_stop_event_mask_probability_zero_is_identity():
    """prob=0 must reproduce the upstream command exactly (no injected stops)."""
    sample = T([0.0, 0.24, 0.26, 0.99])
    mask = SE.stop_event_mask(sample, 0.0)
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


def test_stop_event_mask_thresholds_on_probability():
    """prob=0.25 selects exactly the samples at or below 0.25."""
    sample = T([0.0, 0.2, 0.25, 0.26, 0.9])
    mask = SE.stop_event_mask(sample, 0.25)
    assert mask.tolist() == [True, True, True, False, False]


def _window_state(mod, command, prev, age, anchor, pos, window_steps=25, tol=0.02):
    return mod.advance_stop_window(
        command=T(command),
        prev_command=T(prev),
        age=torch.tensor(age, dtype=torch.long),
        anchor=T(anchor),
        pos=T(pos),
        window_steps=window_steps,
        onset_tol=tol,
    )


def test_falling_edge_opens_window_and_latches_anchor():
    """A 0.8 -> 0 edge must start the window at the current position."""
    age, anchor, drift2, active = _window_state(
        SE,
        command=[[0.0, 0.0, 0.0]],
        prev=[[0.8, 0.0, 0.0]],
        age=[0],
        anchor=[[9.0, 9.0]],
        pos=[[1.5, 0.25]],
    )
    assert int(age[0]) == 25
    assert anchor.tolist() == [[1.5, 0.25]]
    assert float(drift2[0]) == pytest.approx(0.0)
    assert bool(active[0])


def test_window_counts_down_and_closes_after_window_steps():
    """The window must close on its own, so a standing robot stops being charged."""
    age = torch.tensor([2], dtype=torch.long)
    cmd, prev, pos = [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.2, 0.0]]
    anchor = [[0.0, 0.0]]
    age, anchor, drift2, active = _window_state(SE, cmd, prev, age, anchor, pos)
    assert int(age[0]) == 1 and bool(active[0])
    age, anchor, drift2, active = _window_state(SE, cmd, prev, age, anchor, pos)
    assert int(age[0]) == 0 and not bool(active[0])


def test_stale_latch_is_cleared_by_a_new_moving_command():
    """A fresh non-zero hold must not inherit the previous anchor."""
    age, anchor, drift2, active = _window_state(
        SE,
        command=[[0.6, 0.0, 0.0]],
        prev=[[0.0, 0.0, 0.0]],
        age=[10],
        anchor=[[5.0, 5.0]],
        pos=[[0.0, 0.0]],
    )
    assert int(age[0]) == 0
    assert not bool(active[0])


def test_drift2_is_squared_planar_displacement_from_the_edge():
    """Only the xy plane counts, and the value is squared (metric shape)."""
    age, anchor, drift2, active = _window_state(
        SE,
        command=[[0.0, 0.0, 0.0]],
        prev=[[0.0, 0.0, 0.0]],
        age=[5],
        anchor=[[1.0, 1.0]],
        pos=[[1.03, 1.04]],  # 0.03 east, 0.04 north -> 0.05 m
    )
    assert float(drift2[0]) == pytest.approx(0.0025, abs=1e-7)
    assert float(drift2[0]) ** 0.5 == pytest.approx(0.05, abs=1e-6)


def test_normalisation_matches_protocol_threshold():
    """0.12 m of drift against a 0.10 m target is 1.44, i.e. 44 % over budget."""
    assert pytest.approx((0.12 / 0.10) ** 2) == 1.44


def test_per_env_independence():
    """Three envs in different phases must not share latch state."""
    age, anchor, drift2, active = _window_state(
        SE,
        command=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        prev=[[0.8, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        age=[0, 7, 0],
        anchor=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        pos=[[0.4, 0.0], [0.4, 0.0], [0.4, 0.0]],
        window_steps=25,
    )
    assert int(age[0]) == 25 and bool(active[0])  # just stopped
    assert int(age[1]) == 0 and not bool(active[1])  # still moving: window closed
    assert int(age[2]) == 0 and not bool(active[2])  # standing, no edge
    assert float(drift2[0]) == pytest.approx(0.0)  # anchor re-latched at current pos
    assert float(drift2[2]) == pytest.approx(0.16)  # 0.4 m from its old anchor


def test_reset_step_clears_the_latch_instead_of_charging_a_teleport():
    """A window still open when the episode ends must not charge the spawn jump.

    Rewards are computed after ``_reset_idx``, so without this guard a stop
    window that survives the reset compares the spawn position against the
    previous episode's anchor: metres of "drift" in one step, which is what
    destroyed the critic in the first round 4 smoke run.
    """
    age, anchor, drift2, active = SE.relatch_on_episode_start(
        age=torch.tensor([18], dtype=torch.long),
        anchor=T([[0.0, 0.0]]),
        pos=T([[12.0, 8.0]]),  # new episode spawn, 14.4 m away
        episode_length_buf=torch.tensor([1], dtype=torch.long),  # step right after _reset_idx
    )
    assert int(age[0]) == 0
    assert not bool(active[0])
    assert float(drift2[0]) == pytest.approx(0.0)
    assert anchor.tolist() == [[12.0, 8.0]]


def test_non_reset_step_keeps_the_latch():
    """The guard must only touch the envs that actually reset."""
    age, anchor, drift2, active = SE.relatch_on_episode_start(
        age=torch.tensor([18, 18], dtype=torch.long),
        anchor=T([[0.0, 0.0], [0.0, 0.0]]),
        pos=T([[0.06, 0.08], [12.0, 8.0]]),
        episode_length_buf=torch.tensor([37, 1], dtype=torch.long),
    )
    assert int(age[0]) == 18 and bool(active[0])
    assert float(drift2[0]) == pytest.approx(0.01, abs=1e-7)  # 0.10 m travelled
    assert int(age[1]) == 0 and not bool(active[1])


def test_clamp_saturates_a_pushed_stop():
    """Beyond max_ratio the penalty saturates instead of growing without bound."""
    target, max_ratio = 0.10, 4.0
    for drift in (0.05, 0.10, 0.20, 3.0):
        ratio = min((drift / target) ** 2, max_ratio)
        assert ratio <= max_ratio
    assert pytest.approx(min((0.20 / target) ** 2, max_ratio)) == 4.0
    assert pytest.approx(min((3.0 / target) ** 2, max_ratio)) == 4.0  # 900 would have been the raw value


def test_stop_probability_ramps_in_linearly():
    """Stop events must be off at iteration 0 and reach p_max at the ramp length."""
    assert pytest.approx(SE.ramped_probability(0.25, 0, 300)) == 0.0
    assert pytest.approx(SE.ramped_probability(0.25, 150, 300)) == 0.125
    assert pytest.approx(SE.ramped_probability(0.25, 300, 300)) == 0.25
    assert pytest.approx(SE.ramped_probability(0.25, 5000, 300)) == 0.25  # saturated


def test_stop_probability_without_ramp_is_immediate():
    """ramp_iters=0 must reproduce the un-ramped behaviour (and 0 stays 0)."""
    assert pytest.approx(SE.ramped_probability(0.25, 0, 0)) == 0.25
    assert pytest.approx(SE.ramped_probability(0.0, 100, 300)) == 0.0


def test_ramp_length_is_control_steps_not_ppo_iterations():
    """TASK-033: ``ramp_iters`` counts control steps, not PPO iterations.

    The counter passed in is ``env.common_step_counter``, which
    ``ManagerBasedRLEnv.step`` increments once per control step. One PPO iteration
    samples ``num_steps_per_env = 24`` control steps, so the configured 300 is
    12.5 iterations (6.0 s at 50 Hz) - the config comments used to claim
    "300 iterations". This test pins the unit so the wording cannot drift back.
    """
    num_steps_per_env = 24
    step_dt = 0.02  # 200 Hz physics / decimation 4
    ramp = 300
    assert ramp / num_steps_per_env == pytest.approx(12.5)
    assert ramp * step_dt == pytest.approx(6.0)
    assert pytest.approx(SE.ramped_probability(0.25, ramp, ramp)) == 0.25       # saturated at the end
    assert SE.ramped_probability(0.25, ramp - 1, ramp) < 0.25                  # not before
    # after 12 whole iterations (288 control steps) the ramp is not finished yet
    assert SE.ramped_probability(0.25, 12 * num_steps_per_env, ramp) == pytest.approx(0.24)
