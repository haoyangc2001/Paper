"""Validity and completeness gating for the aggregation step (TASK-032).

Pure Python/NumPy, no Isaac. Run with::

    python -m pytest CaMRA/locomotion/tests/test_validity_gating.py -q

The point of these tests is the failure mode found in TASK-029: the summary
dropped every non-finite value and then judged the mean of whatever was left,
so "1 valid value + 29 missing" produced a PASS. Each test below pins one way
that can go wrong.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

LOCOMOTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCOMOTION / "source" / "unitree_rl_lab"))

from unitree_rl_lab.utils.eval_metrics import (  # noqa: E402
    FIELDS,
    I,
    MIN_VALID_RATIO,
    aggregate,
    compute_metrics,
)

SPEC = {
    "id": "unit",
    "duration_s": 1.0,
    "metrics": ["rmse_vx", "rise_time_s", "failure_rate"],
    "thresholds": {"rmse_vx": 0.15, "rise_time_s": 0.5},
}


def row(**kw) -> dict:
    base = {
        "scenario": "unit",
        "failed": 0,
        "rmse_vx": 0.10,
        "rise_time_s": 0.20,
        "warmup_short": False,
        "invalid_reasons": "{}",
    }
    base.update(kw)
    return base


def rows_with(n_valid: int, n_total: int, reason: str = "not_reached") -> list[dict]:
    out = []
    for i in range(n_total):
        if i < n_valid:
            out.append(row())
        else:
            out.append(row(rise_time_s=float("nan"), invalid_reasons=json.dumps({"rise_time_s": reason})))
    return out


# ---------------------------------------------------------------------------
# the exact TASK-029 failure mode
# ---------------------------------------------------------------------------


def test_one_valid_plus_29_missing_cannot_pass():
    agg = aggregate(rows_with(1, 30), SPEC, "nominal")
    check = agg["checks"]["rise_time_s"]
    assert check["expected_count"] == 30
    assert check["valid_count"] == 1
    assert check["invalid_count"] == 29
    assert check["valid_ratio"] == pytest.approx(1 / 30)
    assert check["status"] == "insufficient_evidence"
    assert check["pass"] is False
    assert agg["pass"] is False
    assert agg["complete"] is False
    assert agg["incomplete_reason"] == "insufficient_evidence"
    # the failure is classified as missing evidence, not as an out-of-threshold value
    assert check["invalid_reasons"] == {"not_reached": 29}


def test_missing_values_are_not_silently_dropped_from_the_mean():
    agg = aggregate(rows_with(1, 10), SPEC, "nominal")
    assert agg["rise_time_s"] == pytest.approx(0.20)  # mean of what exists ...
    assert agg["checks"]["rise_time_s"]["valid_count"] == 1  # ... and the count is reported


def test_all_non_finite_cannot_pass():
    rows = [row(rise_time_s=float("nan"), invalid_reasons=json.dumps({"rise_time_s": "not_reached"})) for _ in range(20)]
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["checks"]["rise_time_s"]["status"] == "insufficient_evidence"
    assert math.isnan(agg["rise_time_s"])
    assert agg["pass"] is False


def test_missing_key_is_counted_as_missing_evidence():
    rows = [row() for _ in range(20)]
    for r in rows:
        r.pop("rise_time_s")
    agg = aggregate(rows, SPEC, "nominal")
    check = agg["checks"]["rise_time_s"]
    assert check["valid_count"] == 0
    assert check["invalid_reasons"] == {"missing_key": 20}
    assert check["status"] == "insufficient_evidence"


def test_empty_input_cannot_pass():
    agg = aggregate([], SPEC, "nominal")
    assert agg["episodes"] == 0
    assert agg["pass"] is False
    assert agg["complete"] is False
    assert agg["incomplete_reason"] == "no_episodes"
    assert agg["checks"]["failure_rate"]["status"] == "insufficient_evidence"


def test_warmup_short_samples_do_not_count_as_measurements():
    rows = [row(warmup_short=True) for _ in range(10)]
    agg = aggregate(rows, SPEC, "nominal")
    check = agg["checks"]["rmse_vx"]
    assert check["valid_count"] == 0
    assert check["invalid_reasons"] == {"warmup_not_reached": 10}
    assert agg["pass"] is False


# ---------------------------------------------------------------------------
# legitimate N/A keeps its own rule
# ---------------------------------------------------------------------------


def test_undefined_program_is_excluded_from_the_denominator():
    # 1 measured + 29 genuinely inapplicable -> denominator is 1, ratio is 1.0
    rows = rows_with(1, 30, reason="undefined_program")
    agg = aggregate(rows, SPEC, "nominal")
    check = agg["checks"]["rise_time_s"]
    assert check["na_count"] == 29
    assert check["valid_ratio"] == pytest.approx(1.0)
    assert check["status"] == "pass"
    assert agg["complete"] is True


def test_all_inapplicable_gated_metric_is_insufficient_not_pass():
    rows = rows_with(0, 10, reason="undefined_program")
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["checks"]["rise_time_s"]["na_count"] == 10
    assert agg["checks"]["rise_time_s"]["status"] == "insufficient_evidence"


def test_reported_only_metric_does_not_gate_but_still_reports_counts():
    spec = {"id": "unit", "metrics": ["rmse_vx", "action_rate_rms", "failure_rate"], "thresholds": {"rmse_vx": 0.15}}
    rows = [row() for _ in range(10)]
    for r in rows:
        r["action_rate_rms"] = float("nan")
        r["invalid_reasons"] = json.dumps({"action_rate_rms": "short_trajectory"})
    agg = aggregate(rows, spec, "nominal")
    chk = agg["checks"]["action_rate_rms"]
    assert chk["status"] == "reported_only"
    assert chk["valid_count"] == 0
    assert chk["invalid_reasons"] == {"short_trajectory": 10}
    assert agg["pass"] is True  # reported-only metrics never decide the verdict


# ---------------------------------------------------------------------------
# the 90 % boundary and the fail / insufficient distinction
# ---------------------------------------------------------------------------


def test_validity_boundary_is_the_declared_ratio():
    assert MIN_VALID_RATIO == 0.90
    at = aggregate(rows_with(9, 10), SPEC, "nominal")["checks"]["rise_time_s"]
    below = aggregate(rows_with(8, 10), SPEC, "nominal")["checks"]["rise_time_s"]
    assert at["valid_ratio"] == pytest.approx(0.9)
    assert at["status"] == "pass"
    assert below["valid_ratio"] == pytest.approx(0.8)
    assert below["status"] == "insufficient_evidence"


def test_out_of_threshold_with_full_validity_is_a_failure_not_missing_evidence():
    rows = [row(rmse_vx=0.30) for _ in range(10)]
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["checks"]["rmse_vx"]["status"] == "fail"
    assert agg["checks"]["rmse_vx"]["valid_ratio"] == pytest.approx(1.0)
    assert agg["complete"] is True  # evidence is complete, the controller simply misses
    assert agg["pass"] is False


def test_failure_rate_still_uses_every_episode():
    rows = [row(failed=1 if i < 3 else 0) for i in range(10)]
    agg = aggregate(rows, SPEC, "nominal")
    assert agg["failure_rate"] == pytest.approx(0.3)
    assert agg["checks"]["failure_rate"]["valid_count"] == 10
    assert agg["checks"]["failure_rate"]["status"] == "fail"  # 0.3 > 0.05


# ---------------------------------------------------------------------------
# compute_metrics must explain why a value is missing
# ---------------------------------------------------------------------------


def make_standing_episode(n: int = 100) -> np.ndarray:
    """Constant zero command: rise time / overshoot / stop drift are undefined."""
    ep = np.zeros((n, len(FIELDS)), dtype=np.float64)
    ep[:, I["episode_step"]] = np.arange(1, n + 1)
    ep[:, I["sim_time"]] = np.arange(n) * 0.02
    ep[:, I["cmd_vx"] : I["cmd_wz"] + 1] = 0.0
    return ep


def make_moving_episode(n: int = 100, target: float = 0.3, reached: float = 0.30) -> np.ndarray:
    """Constant 0.3 m/s command; ``reached`` says how fast the base actually goes."""
    ep = make_standing_episode(n)
    ep[:, I["cmd_vx"]] = target
    ep[:, I["vx_body"]] = reached
    return ep


def test_standing_command_marks_rise_time_undefined():
    m = compute_metrics(make_standing_episode(), SPEC, 0.02, warmup_s=0.0)
    assert math.isnan(m["rise_time_s"])
    assert m["invalid_reasons"]["rise_time_s"] == "undefined_program"
    assert m["invalid_reasons"]["stop_drift_m"] == "undefined_program"


def test_command_without_response_is_not_reached_not_undefined():
    m = compute_metrics(make_moving_episode(reached=0.05), SPEC, 0.02, warmup_s=0.0)
    assert math.isnan(m["rise_time_s"])
    assert m["invalid_reasons"]["rise_time_s"] == "not_reached"


def test_reached_command_has_no_invalid_reason_for_rise_time():
    m = compute_metrics(make_moving_episode(reached=0.30), SPEC, 0.02, warmup_s=0.0)
    assert m["rise_time_s"] == pytest.approx(0.0)
    assert "rise_time_s" not in m["invalid_reasons"]


def test_one_step_episode_is_flagged_as_short_trajectory():
    m = compute_metrics(make_moving_episode(n=1), SPEC, 0.02, warmup_s=0.0)
    assert m["invalid_reasons"]["action_rate_rms"] == "short_trajectory"


def test_warmup_short_flag_is_recorded():
    m = compute_metrics(make_moving_episode(n=10), SPEC, 0.02, warmup_s=1.0)  # 0.2 s < 1.0 s
    assert m["warmup_short"] is True
    assert m["warmup_samples"] < 2


def test_reasons_survive_the_csv_round_trip():
    """per_episode.csv stores the map as JSON text; the aggregator must parse it."""
    m = compute_metrics(make_standing_episode(), SPEC, 0.02, warmup_s=0.0)
    as_csv = dict(m)
    as_csv["invalid_reasons"] = json.dumps(m["invalid_reasons"], sort_keys=True)
    agg = aggregate([as_csv] * 10, SPEC, "nominal")
    chk = agg["checks"]["rise_time_s"]
    assert chk["na_count"] == 10
    assert chk["invalid_reasons"] == {"undefined_program": 10}
