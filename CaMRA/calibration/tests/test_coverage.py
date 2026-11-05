"""检验覆盖统计的单元测试（CAL-027）。

覆盖：阈值含等号、方向分层、四套分母、cell 完整率、簇重采样、三种判定状态。
"""
from __future__ import annotations

import numpy as np
import pytest

from camra_calibration.coverage import (
    CoverageError, HeldoutRow, bootstrap_rate, cost_covered, coverage_verdict,
    denominators, direction_of, per_cell, position_covered, rate, rate_with_ci,
)


def row(tid: str, *, cell="f|clean", direction="forward", low_speed=False,
        outcome="completed", official=True, non_computable=False,
        data_invalid=False, eps_c=0.05, eps_d=0.5) -> HeldoutRow:
    fam, prof = cell.split("|")
    return HeldoutRow(trial_id=tid, family=fam, profile=prof, cell=cell,
                      direction=direction, low_speed=low_speed, outcome=outcome,
                      complete=official, official=official,
                      non_computable=non_computable, data_invalid=data_invalid,
                      eps_c=eps_c, eps_d=eps_d)


# ---------------------------------------------------------------- 阈值
def test_threshold_inclusive():
    assert cost_covered(0.19, 0.19) is True          # 等于阈值算覆盖
    assert cost_covered(0.1900001, 0.19) is False
    assert position_covered(2.0, 2.0) is True
    assert position_covered(2.0001, 2.0) is False
    assert cost_covered(None, 0.19) is None
    assert position_covered(float("nan"), 1.0) is None


# ---------------------------------------------------------------- 方向
def test_direction_layers():
    fwd = np.array([[0.8, 0.0, 0.0]] * 10)
    bwd = np.array([[-0.8, 0.0, 0.0]] * 10)
    left = np.array([[0.0, 0.5, 0.0]] * 10)
    right = np.array([[0.0, -0.5, 0.0]] * 10)
    rot = np.array([[0.0, 0.0, 1.0]] * 10)
    assert [direction_of(x) for x in (fwd, bwd, left, right, rot)] == [
        "forward", "backward", "left", "right", "rotation_only"]
    with pytest.raises(CoverageError):
        direction_of(np.zeros((10, 1)))


# ---------------------------------------------------------------- 比率
def test_rate_skips_none_and_counts():
    out = rate([True, False, None, True])
    assert out["n"] == 3 and out["n_covered"] == 2
    assert out["rate"] == pytest.approx(2 / 3)
    assert rate([None, None])["rate"] is None


def test_bootstrap_rate_is_reproducible_and_clustered():
    flags = [True] * 18 + [False] * 2
    a = bootstrap_rate(flags, replicates=300, seed=11)
    b = bootstrap_rate(flags, replicates=300, seed=11)
    assert a == b
    assert a["point"] == pytest.approx(0.9)
    assert a["lo"] <= 0.9 <= a["hi"]
    cl = bootstrap_rate(flags, replicates=300, seed=11,
                        clusters=["A"] * 10 + ["B"] * 10)
    assert cl["unit"] == "cluster"
    with pytest.raises(CoverageError):
        bootstrap_rate(flags, replicates=10, clusters=["A"])
    with pytest.raises(CoverageError):
        bootstrap_rate([], replicates=10)


def test_rate_with_ci_reports_both_units():
    rows = [row(f"t{i}", eps_c=0.01 if i % 4 else 0.9) for i in range(40)]
    flags = [cost_covered(r.eps_c, 0.2) for r in rows]
    out = rate_with_ci(flags, [r.cell for r in rows], 200, 5)
    assert out["ci_trial"]["unit"] == "trial"
    assert out["ci_cluster"]["unit"] == "cluster"
    assert out["rate"] == pytest.approx(30 / 40)


# ---------------------------------------------------------------- 分母
def test_denominators_four_sets():
    rows = [row("a"), row("b", official=False, outcome="terminated"),
            row("c", non_computable=True, eps_c=None), row("d", data_invalid=True)]
    den = denominators(rows, 400)
    assert den["expected"] == 400 and den["observed"] == 4
    assert den["complete"] == 3          # c 仍 complete，只是 ε_c 不可算
    assert den["business_failed"] == 1
    assert den["data_invalid"] == 1
    assert den["non_computable"] == 1
    assert den["eps_c_usable"] == 2
    assert den["complete_share_overall"] == pytest.approx(3 / 400)


# ---------------------------------------------------------------- 分组
def test_per_cell_share_and_coverage():
    rows = [row(f"t{i}", cell="a|clean", eps_c=0.5 if i < 2 else 0.01) for i in range(10)]
    rows += [row("t9", cell="b|combined", official=False, outcome="terminated")]
    cells = per_cell(rows, alpha=0.2, delta=1.0, expected_per_cell=20)
    assert cells["a|clean"]["complete_share"] == pytest.approx(10 / 20)
    assert cells["a|clean"]["cost"]["n_covered"] == 8
    assert cells["b|combined"]["complete_share"] == 0.0


# ---------------------------------------------------------------- 判定
def _overall(rate_value: float) -> dict:
    ci = {"point": rate_value, "lo": rate_value - 0.02, "hi": min(1.0, rate_value + 0.02),
          "unit": "trial"}
    blk = {"n": 100, "n_covered": int(rate_value * 100), "rate": rate_value,
           "n_not_computable": 0, "ci_trial": ci, "ci_cluster": ci}
    return {"cost": blk, "position": blk, "joint": blk}


def test_verdict_three_states():
    cells_ok = {"a|clean": {"complete_share": 1.0}, "b|push_only": {"complete_share": 0.95}}
    den = {"expected": 400, "observed": 400}

    ok = coverage_verdict(den, _overall(0.93), cells_ok)
    assert ok["status"] == "meets_target"
    assert ok["joint_ci_lo"] == pytest.approx(0.91)

    low = coverage_verdict(den, _overall(0.85), cells_ok)
    assert low["status"] == "below_target"
    assert low["below_target"]["joint"] is True
    assert "禁止据此反向调整" in low["note"]

    weak = coverage_verdict(den, _overall(0.93),
                            {**cells_ok, "c|combined": {"complete_share": 0.5}})
    assert weak["status"] == "insufficient_cells"
    assert list(weak["cells_below_share"]) == ["c|combined"]
