"""CAL-022 步成本接口测试（纯 CPU）。

覆盖：逐步相加 = J_nom（含 K>0 历史依赖）、任意视窗、预算扣减与 z_ratio、
单位隔离（拒绝 a.u.）、reset 语义、schema 散列、能量比 0 的原样保留。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from camra_calibration.power import N_INTERVALS, STEP_COST_DT, PowerModel, feature_names
from camra_calibration.step_cost import (
    BudgetState, StepCost, StepCostError, energy_ratio, schema_sha256, trajectory_cost,
    write_schema,
)

# 与冻结 power-v1 同构的权重（W），仅用于测试
WEIGHTS = (3.0284, 4.7226, 4.8385, 0.0, 31.542, 48.3139, 2.3904, 0.0, 0.0, 0.0)


def frozen_like_model() -> PowerModel:
    names = feature_names("quadratic", 0)
    return PowerModel(weights=WEIGHTS, feature_names=names, base="quadratic", history_k=0,
                      fit_mode="relative", train={"split": "test"},
                      domain={"abs_vx": {"max": 1.2}, "abs_vy": {"max": 0.6},
                              "abs_wz": {"max": 1.2}})


def history_model() -> PowerModel:
    names = feature_names("linear", 1)
    return PowerModel(weights=(1.0, 30.0, 20.0, 10.0, 5.0, 15.0, 8.0, 6.0),
                      feature_names=names, base="linear", history_k=1,
                      fit_mode="relative", train={"split": "test"}, domain={})


def commands(seed: int = 0, n: int = N_INTERVALS + 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seg = rng.uniform(-1.0, 1.0, size=((n - 1) // 10 + 1, 3))
    out = np.zeros((n, 3))
    for j in range(n):
        s = min(j // 10, seg.shape[0] - 1)
        out[j] = seg[s] * min(1.0, (j % 10) / 10.0 + 0.2)
    return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------- 逐步相加
@pytest.mark.parametrize("n", [11, 101, 301, 601])
def test_step_sum_equals_j_nom(n):
    m = frozen_like_model()
    cmd = commands(1, n)
    out = trajectory_cost(m, cmd)
    assert out["n_steps"] == n - 1
    assert out["horizon_s"] == pytest.approx((n - 1) * STEP_COST_DT)
    assert out["equal"], out["rel_diff"]
    assert out["abs_diff_J"] < 1e-9


def test_step_sum_equals_j_nom_with_history_dependency():
    m = history_model()
    cmd = commands(2, 51)
    out = trajectory_cost(m, cmd)
    assert out["equal"], out["rel_diff"]
    assert out["j_nom_J"] == pytest.approx(out["cost_nom_sum_J"], rel=1e-12)


def test_alpha_scales_cost_but_not_power():
    m = frozen_like_model()
    cmd = commands(3, 31)
    base = trajectory_cost(m, cmd, alpha=0.0)
    scaled = trajectory_cost(m, cmd, alpha=0.25)
    assert scaled["cost_sum_J"] == pytest.approx(1.25 * base["cost_nom_sum_J"], rel=1e-12)
    assert scaled["cost_nom_sum_J"] == pytest.approx(base["cost_nom_sum_J"], rel=1e-12)


# ---------------------------------------------------------------- 单步与预算
def test_single_step_matches_trajectory_first_step():
    m = frozen_like_model()
    cmd = commands(4, 21)
    sc = StepCost(m)
    sc.reset()
    rec = sc.step(cmd[0])
    whole = trajectory_cost(m, cmd)
    assert rec.cost_nom_J == pytest.approx(whole["records"][0].cost_nom_J, rel=1e-12)
    assert rec.t_s == 0.0
    assert rec.unit == "J"


def test_budget_charges_and_z_ratio():
    m = frozen_like_model()
    b = BudgetState(initial_J=100.0)
    sc = StepCost(m, budget=b)
    sc.reset()
    total = 0.0
    for row in commands(5, 41)[:-1]:
        total += sc.step(row).cost_J
    assert b.spent_J == pytest.approx(total, rel=1e-12)
    assert b.remaining_J == pytest.approx(100.0 - total, rel=1e-12)
    assert 0.0 <= b.z_ratio <= 1.0
    assert b.exhausted == (b.spent_J >= b.initial_J)


def test_budget_rejects_legacy_units():
    with pytest.raises(StepCostError):
        BudgetState(initial_J=10.0, unit="a.u.")
    with pytest.raises(StepCostError):
        BudgetState(initial_J=10.0, kind="legacy_au")
    with pytest.raises(StepCostError):
        BudgetState(initial_J=0.0)
    with pytest.raises(StepCostError):
        BudgetState(initial_J=float("nan"))


def test_charge_rejects_negative_cost():
    b = BudgetState(initial_J=10.0)
    with pytest.raises(StepCostError):
        b.charge(-1.0)
    with pytest.raises(StepCostError):
        b.charge(float("inf"))


# ---------------------------------------------------------------- 状态与重置
def test_reset_restarts_time_and_history():
    m = history_model()
    sc = StepCost(m)
    sc.reset()
    sc.step((0.5, 0.0, 0.0))
    sc.step((0.6, 0.0, 0.0))
    r = sc.step((0.7, 0.0, 0.0))
    assert r.t_s == pytest.approx(0.2)
    sc.reset()
    r2 = sc.step((0.7, 0.0, 0.0))
    assert r2.t_s == 0.0
    # 重置后首步等价于"命令自己做历史"，与 feature_matrix 的钳位约定一致
    assert r2.cost_nom_J == pytest.approx(trajectory_cost(m, np.array([[0.7, 0.0, 0.0]] * 2),
                                                          reset=True)["cost_nom_sum_J"], rel=1e-12)


def test_require_explicit_reset_needs_history():
    sc = StepCost(history_model(), reset_rule="require_explicit")
    with pytest.raises(StepCostError):
        sc.reset()
    sc.reset(history=[(0.1, 0.0, 0.0)])
    assert sc.history_k == 1


def test_dt_must_match_model():
    m = frozen_like_model()
    with pytest.raises(StepCostError):
        StepCost(m, dt_s=0.05)
    StepCost(m, dt_s=STEP_COST_DT)          # 一致即可


def test_augmented_state_spec_reflects_history():
    m0 = frozen_like_model()
    spec0 = StepCost(m0).augmented_state_spec()
    assert spec0["history_k"] == 0
    assert "none" in spec0["augmented_state"]
    spec1 = StepCost(history_model()).augmented_state_spec()
    assert spec1["history_k"] == 1
    assert "last 1" in spec1["augmented_state"]


def test_step_rejects_bad_command():
    sc = StepCost(frozen_like_model())
    sc.reset()
    with pytest.raises(StepCostError):
        sc.step((0.1, 0.2))
    with pytest.raises(StepCostError):
        sc.step((0.1, float("nan"), 0.0))


# ---------------------------------------------------------------- 能量比
def test_energy_ratio_equal_case_preserves_exact_zero():
    r, why = energy_ratio(432.1234, 432.1234)
    assert why == "ok"
    assert r == 0.0                        # 不夹、不加 epsilon


def test_energy_ratio_both_signs():
    up, _ = energy_ratio(110.0, 100.0)
    down, _ = energy_ratio(90.0, 100.0)
    assert up == pytest.approx(0.1)
    assert down == pytest.approx(-0.1)


def test_energy_ratio_non_computable_and_nan():
    r, why = energy_ratio(100.0, 0.5, j_min=1.0)
    assert r is None and why == "non_computable"
    r2, why2 = energy_ratio(float("nan"), 100.0)
    assert r2 is None and why2 == "nan"


# ---------------------------------------------------------------- schema
def test_schema_roundtrip_and_hash(tmp_path):
    m = frozen_like_model()
    p = tmp_path / "schema.json"
    digest = write_schema(p, m)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["sha256"] == digest
    assert payload["unit"] == "J"
    assert payload["model_sha256"] == m.sha256()
    body = {k: v for k, v in payload.items() if k != "sha256"}
    assert schema_sha256(body) == digest
    side = tmp_path / "schema.json.sha256"
    assert side.read_text(encoding="utf-8").split()[0] == digest


def test_schema_lists_required_fields():
    m = frozen_like_model()
    s = StepCost(m).schema()
    for key in ("dt_s", "unit", "history_k", "augmented_state_spec", "step", "budget"):
        assert key in s
    assert "cost_nom_J" in s["step"] and "remaining_J" in s["budget"]


def test_zero_command_cost_is_positive_but_small():
    """站立不是零成本：截距项给出 ~3 W；同时 J_nom 远大于 J_min，不会被判不可算。"""
    m = frozen_like_model()
    still = np.zeros((N_INTERVALS + 1, 3))
    j = m.j_nom(still)
    assert 50.0 < j < 200.0
    assert j > 1.0
