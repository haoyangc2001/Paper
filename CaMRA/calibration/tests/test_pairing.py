"""CAL-017 数值与分类测试。

本文件所有数据都是 **test_fixture**（人工构造的解析场或合成误差），
用于验证口径与错误分支，不作为任何仿真证据。
"""
import numpy as np
import pytest

from camra_calibration.pairing import (
    Issue,
    aggregate,
    check_positions,
    check_time_grid,
    compute_trial_errors,
    detect_duplicates,
    deviation,
    deviation_on_grid,
    epsilon_c,
    low_speed_share,
)

DT = 0.1
N = 301


def grid(n=N, dt=DT, dtype=np.float64):
    return (np.arange(n) * dt).astype(dtype)


def zeros(n=N):
    return np.zeros((n, 2))


# ---------------- ε_c 数值（test_fixture） ----------------
def test_epsilon_c_exact_and_negative_overrun_kept():
    assert epsilon_c(12.0, 10.0) == (pytest.approx(0.2), "ok")
    val, reason = epsilon_c(9.0, 10.0)
    assert reason == "ok" and val == pytest.approx(-0.1)  # 负超支保留原值


def test_epsilon_c_missing_and_near_zero():
    assert epsilon_c(12.0, None) == (None, "j_nom_missing")
    assert epsilon_c(0.2, 0.5) == (None, "non_computable")
    assert epsilon_c(0.2, 0.0) == (None, "non_computable")
    assert epsilon_c(float("nan"), 10.0) == (None, "nan")


# ---------------- ε_d 几何（test_fixture） ----------------
def test_epsilon_d_is_max_not_mean():
    nom = zeros()
    ex = zeros()
    ex[:, 0] = np.linspace(0, 3.0, N)          # 线性增长偏差
    d = deviation(ex, nom)
    assert d.max() == pytest.approx(3.0)
    assert d.mean() == pytest.approx(1.5)


def test_deviation_requires_same_shape():
    with pytest.raises(ValueError):
        deviation(zeros(10), zeros(11))


def test_deviation_on_grid_exact_at_grid_points():
    t = grid()
    nom = np.stack([np.linspace(0, 10, N), np.linspace(0, -5, N)], axis=1)
    d = deviation_on_grid(t, nom, t, nom)
    assert np.allclose(d, 0.0)


def test_deviation_on_grid_linear_interpolation():
    t = grid()
    nom = np.stack([t, np.zeros_like(t)], axis=1)      # x = t（直线）
    tq = np.array([0.05, 0.15, 0.25])
    ex = np.stack([tq, np.zeros_like(tq)], axis=1)     # 恰好落在直线上
    d = deviation_on_grid(tq, ex, t, nom)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_deviation_on_grid_rejects_extrapolation():
    t = grid()
    nom = zeros()
    with pytest.raises(ValueError):
        deviation_on_grid(np.array([30.05]), zeros(1), t, nom)


# ---------------- 网格校验（test_fixture） ----------------
def test_grid_ok_float64():
    assert check_time_grid(grid()) == []


def test_grid_tolerates_float32_storage():
    issues = check_time_grid(grid(dtype=np.float32))
    assert issues == []


def test_grid_detects_start_offset_and_jump():
    bad = grid().copy()
    bad += 0.01
    assert any(i.reason == "time_jump" for i in check_time_grid(bad))
    jump = grid().copy()
    jump[100] += 0.05
    assert any(i.reason in ("time_jump", "non_monotonic") for i in check_time_grid(jump))


def test_grid_detects_nan_and_short():
    nan = grid()
    nan[5] = np.nan
    assert any(i.reason == "nan" for i in check_time_grid(nan))
    assert any(i.reason == "zero_window" for i in check_time_grid([0.0]))


def test_positions_shape_and_nan():
    assert check_positions(zeros(), zeros()) == []
    assert any(i.reason == "shape_mismatch" for i in check_positions(zeros(3), zeros(4)))
    bad = zeros()
    bad[0, 0] = np.inf
    assert any(i.reason == "nan" for i in check_positions(bad, zeros()))


# ---------------- 分类（test_fixture） ----------------
def complete_case(**over):
    kw = dict(trial_id="t", outcome="completed", t_pair=grid(), exec_pos=zeros(), nom_pos=zeros(),
              j_exec=12.0, j_nom=10.0, cmd=np.tile([[0.5, 0.0, 0.0]], (N, 1)), steps_executed=6000)
    kw.update(over)
    return compute_trial_errors(**kw)


def test_complete_trial_is_official():
    r = complete_case()
    assert r.ok_official() if hasattr(r, "ok_official") else True
    assert r.official and r.complete and r.error_class == "ok"
    assert r.epsilon_c == pytest.approx(0.2)


def test_partial_window_is_diagnostic_only():
    r = complete_case(outcome="terminated", steps_executed=5176)
    assert not r.official and r.prefix_only and not r.complete
    assert r.epsilon_c is None            # 部分窗口不产正式 ε_c
    assert r.epsilon_d is not None        # 前缀诊断仍给出
    assert r.error_class == "business_failure"


def test_nan_data_is_invalid_not_business():
    ex = zeros()
    ex[10, 1] = np.nan
    r = complete_case(exec_pos=ex)
    assert r.error_class == "data_invalid"
    assert not r.official


def test_completed_with_short_window_is_invalid():
    r = complete_case(steps_executed=5999)
    assert r.error_class == "data_invalid"
    assert any(i.reason == "shape_mismatch" for i in r.issues)


def test_unknown_outcome_is_wrong_source():
    r = complete_case(outcome="mystery", steps_executed=100)
    assert any(i.reason == "wrong_source" for i in r.issues)
    assert r.error_class == "data_invalid"


def test_j_nom_missing_is_non_computable_not_failure():
    r = complete_case(j_nom=None)
    assert r.epsilon_c is None and r.non_computable is False
    assert r.error_class == "ok"           # 数据本身没问题，只是模型未就绪
    assert any("j_nom" in n for n in r.notes)


def test_near_zero_denominator_flagged():
    r = complete_case(j_nom=0.5)
    assert r.non_computable and r.epsilon_c is None


# ---------------- 汇总与分母（test_fixture） ----------------
def test_aggregate_denominators_and_exclusions():
    rows = [complete_case(trial_id="ok1"), complete_case(trial_id="ok2"),
            complete_case(trial_id="term", outcome="interrupted", steps_executed=5000),
            complete_case(trial_id="bad", exec_pos=np.full((N, 2), np.nan))]
    rep = aggregate(rows, expected_total=5)
    d = rep["denominators"]
    assert d["all"] == 5 and d["observed"] == 4
    assert d["complete"] == 2 and d["business_failed"] == 1 and d["data_invalid"] == 1
    assert "data_invalid:nan" in rep["exclusions"]
    assert rep["epsilon_c"]["n_available"] == 2
    assert rep["epsilon_d"]["stats"]["n"] == 2
    assert rep["epsilon_d"]["prefix_diagnostic"]["n"] == 1


def test_aggregate_skips_none_values():
    r = complete_case(j_nom=None)
    rep = aggregate([r], expected_total=1)
    assert rep["epsilon_c"]["stats"] is None
    assert rep["epsilon_d"]["stats"]["n"] == 1


def test_low_speed_share_uses_and_semantics():
    cmd = np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.1, 0.0, 0.9]])
    assert low_speed_share(cmd) == pytest.approx(1 / 3)


def test_50hz_sensitivity_matches_when_exec_follows_nominal():
    t10 = grid()
    nom = np.stack([t10, np.zeros_like(t10)], axis=1)
    t50 = (np.arange(1500) + 1) * 0.02
    exec50 = np.stack([t50, np.zeros_like(t50)], axis=1)   # 与名义直线一致
    r = compute_trial_errors(trial_id="t", outcome="completed", t_pair=t10,
                             exec_pos=nom.copy(), nom_pos=nom,
                             j_exec=10.0, j_nom=10.0, t50=t50, exec50_pos=exec50,
                             steps_executed=6000)
    assert r.epsilon_d == pytest.approx(0.0)
    assert r.epsilon_d_50hz == pytest.approx(0.0, abs=1e-9)


def test_duplicate_trial_detection():
    assert detect_duplicates(["a", "b", "a", "c", "a"]) == {"a": 3}
    assert detect_duplicates(["a", "b"]) == {}
    assert detect_duplicates([]) == {}
