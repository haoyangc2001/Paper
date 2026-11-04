"""CAL-020 名义功率模型的纯逻辑测试（无 torch / 无仿真依赖）。"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from camra_calibration.power import (
    FEATURE_SETS, J_MIN, N_INTERVALS, STEP_COST_DT, EvalSample, PowerModel,
    PowerModelError, admission, as_commands, evaluate, feature_matrix,
    feature_names, fit_nonneg, low_speed_share, q_nearest_rank, select_candidate,
    summarize, trajectory_design, transition_share,
)


# ---------------------------------------------------------------- 工具
def make_commands(seed: int, n: int = N_INTERVALS + 1) -> np.ndarray:
    """随机但"像命令"的序列：每 10 点一段，段内线性 ramp，幅值在包络内。"""
    rng = np.random.default_rng(seed)
    n_seg = (n - 1) // 10
    cmds = np.zeros((n, 3), dtype=np.float64)
    targets = rng.uniform(-1.0, 1.0, size=(n_seg, 3))
    targets[::4] *= 0.2                       # 掺入低速段
    for j in range(n):
        seg = min(j // 10, n_seg - 1)
        local = (j % 10) / 10.0
        prev = targets[seg - 1] if seg > 0 else np.zeros(3)
        cmds[j] = prev + (targets[seg] - prev) * local
    return np.clip(cmds, -1.0, 1.0)


TRUE_WEIGHTS = {
    "bias": 55.0, "abs_vx": 90.0, "abs_vy": 60.0, "abs_wz": 25.0,
    "vx2": 40.0, "vy2": 30.0, "wz2": 15.0,
    "abs_vx_wz": 20.0, "abs_vy_wz": 10.0, "abs_vx_vy": 12.0,
}


def true_model() -> PowerModel:
    base = "quadratic"
    names = feature_names(base, 0)
    return PowerModel(weights=tuple(TRUE_WEIGHTS[n] for n in names),
                      feature_names=names, base=base, history_k=0,
                      fit_mode="relative", train={"split": "synthetic"},
                      domain={"abs_vx": {"max": 1.0}, "abs_vy": {"max": 1.0},
                              "abs_wz": {"max": 1.0}})


def synthetic_samples(n: int, seed0: int = 0, model: PowerModel | None = None
                      ) -> list[EvalSample]:
    m = model or true_model()
    out = []
    for i in range(n):
        cmd = make_commands(seed0 + i)
        out.append(EvalSample(trial_id=f"syn-{i:03d}",
                              family=("straight_shift", "arc_combined",
                                      "alternating", "start_stop_turn")[i % 4],
                              j_exec=m.j_nom(cmd), commands=cmd))
    return out


# ---------------------------------------------------------------- 特征
def test_feature_matrix_nonnegative_and_ordered():
    cmd = make_commands(1)
    for base in FEATURE_SETS:
        for k in (0, 1, 2):
            phi = feature_matrix(cmd, base, k)
            assert phi.shape == (N_INTERVALS, len(feature_names(base, k)))
            assert (phi >= 0).all(), f"{base} k={k} 出现负特征"
            assert np.isfinite(phi).all()


def test_feature_names_lag_prefix():
    names = feature_names("linear", 2)
    assert names[:4] == FEATURE_SETS["linear"]
    assert "lag1:abs_vx" in names and "lag2:abs_wz" in names
    assert len(names) == 12


def test_feature_causality_prefix_invariance():
    """改动未来命令不得改变已过去区间的特征（构造性因果）。"""
    cmd = make_commands(2).copy()
    base = "quadratic"
    phi = feature_matrix(cmd, base, 1)
    cut = 150
    cmd2 = cmd.copy()
    cmd2[cut + 1:] = -cmd2[cut + 1:] * 0.5 + 0.3
    phi2 = feature_matrix(cmd2, base, 1)
    assert np.array_equal(phi[:cut], phi2[:cut])
    assert not np.array_equal(phi[cut + 1:], phi2[cut + 1:])


def test_feature_history_clamped_at_start():
    """t=0 起就用首点补齐历史，越界不得抛错或改变起点。"""
    cmd = make_commands(3)
    phi0 = feature_matrix(cmd, "linear", 0)
    phi2 = feature_matrix(cmd, "linear", 2)
    # 首区间的两个历史块都等于当期块（u_{-1}, u_{-2} 都钳到 u_0）
    assert np.array_equal(phi2[0, 4:8], phi0[0])
    assert np.array_equal(phi2[0, 8:12], phi0[0])


def test_as_commands_rejects_bad_input():
    with pytest.raises(PowerModelError):
        as_commands(np.zeros((10, 2)))
    with pytest.raises(PowerModelError):
        as_commands(np.zeros((1, 3)))
    with pytest.raises(PowerModelError):
        as_commands(np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------- 预测
def test_j_nom_matches_step_cost_and_scale():
    m = true_model()
    cmd = make_commands(4)
    p = m.step_power(cmd)
    assert p.shape == (N_INTERVALS,)
    assert (p >= 0).all()
    assert math.isclose(m.j_nom(cmd), p.sum() * STEP_COST_DT, rel_tol=1e-12)
    assert m.j_nom(cmd) > 100.0            # 30 s 机械功量级应在百焦以上


def test_bias_gives_positive_standing_power():
    m = true_model()
    stand = np.zeros((N_INTERVALS + 1, 3))
    assert m.j_nom(stand) == pytest.approx(TRUE_WEIGHTS["bias"] * 30.0, rel=1e-12)


def test_truncation_share_is_zero_by_construction():
    m = true_model()
    for seed in range(5):
        assert m.truncation_share(make_commands(seed)) == 0.0


def test_intercept_share_bounded():
    m = true_model()
    share = m.intercept_share(make_commands(5))
    assert 0.0 < share < 1.0


def test_in_domain_detection():
    m = true_model()
    ok, bad = m.in_domain(make_commands(6))
    assert ok and bad == {}
    over = np.zeros((11, 3))
    over[:, 0] = 1.5
    ok2, bad2 = m.in_domain(over)
    assert not ok2 and "abs_vx" in bad2 and bad2["abs_vx"]["observed_max"] == 1.5


# ---------------------------------------------------------------- 拟合
def test_fit_recovers_true_weights():
    truth = true_model()
    samples = synthetic_samples(80)
    design = np.stack([trajectory_design(s.commands, "quadratic", 0) for s in samples])
    y = np.array([s.j_exec for s in samples])
    w = fit_nonneg(design, y, mode="relative")
    expected = np.array([TRUE_WEIGHTS[n] for n in feature_names("quadratic", 0)])
    assert np.allclose(w, expected, rtol=1e-6, atol=1e-6), (w, expected)
    model = PowerModel(weights=tuple(w), feature_names=feature_names("quadratic", 0),
                       base="quadratic", history_k=0, fit_mode="relative",
                       train={}, domain={})
    ev = evaluate(model, samples)
    assert ev["overall"]["q90"] < 1e-9
    assert ev["n_non_computable"] == 0


def test_fit_rejects_nonpositive_target():
    design = np.ones((5, 4))
    with pytest.raises(PowerModelError):
        fit_nonneg(design, np.array([10.0, 0.0, 5.0, 3.0, 2.0]))


def test_fit_is_deterministic():
    samples = synthetic_samples(30)
    design = np.stack([trajectory_design(s.commands, "quadratic", 0) for s in samples])
    y = np.array([s.j_exec for s in samples])
    a = fit_nonneg(design, y)
    b = fit_nonneg(design, y)
    assert np.array_equal(a, b)


def test_power_model_rejects_negative_weight():
    names = feature_names("linear", 0)
    with pytest.raises(PowerModelError):
        PowerModel(weights=(1.0, -1.0, 0.0, 0.0), feature_names=names, base="linear",
                   history_k=0, fit_mode="relative", train={}, domain={})
    with pytest.raises(PowerModelError):
        PowerModel(weights=(1.0, 2.0), feature_names=names, base="linear",
                   history_k=0, fit_mode="relative", train={}, domain={})
    with pytest.raises(PowerModelError):
        PowerModel(weights=(1.0, 2.0, 3.0, 4.0), feature_names=names, base="quadratic",
                   history_k=0, fit_mode="relative", train={}, domain={})


# ---------------------------------------------------------------- 序列化
def test_save_load_roundtrip(tmp_path):
    m = true_model()
    p = tmp_path / "m.json"
    digest = m.save(p)
    assert digest == m.sha256()
    side = tmp_path / "m.json.sha256"
    assert side.read_text(encoding="utf-8").split()[0] == digest
    back = PowerModel.load(p)
    assert back.sha256() == digest
    cmd = make_commands(7)
    assert back.j_nom(cmd) == m.j_nom(cmd)


def test_save_load_detects_tampering(tmp_path):
    m = true_model()
    p = tmp_path / "m.json"
    m.save(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["weights"][0] = payload["weights"][0] + 1.0
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PowerModelError):
        PowerModel.load(p)


def test_hash_ignores_provenance_metadata():
    """重跑拟合时 train/frozen_at 会变，但模型身份必须不变。"""
    m = true_model()
    a = PowerModel(weights=m.weights, feature_names=m.feature_names, base=m.base,
                   history_k=0, fit_mode="relative",
                   train={"split": "x", "frozen_at": "2026-01-01T00:00:00Z",
                          "candidate": "C2_quad_k0"}, domain=m.domain)
    b = PowerModel(weights=m.weights, feature_names=m.feature_names, base=m.base,
                   history_k=0, fit_mode="relative",
                   train={"split": "x", "frozen_at": "2026-12-31T23:59:59Z",
                          "candidate": "C2_quad_k0"}, domain=m.domain)
    assert a.sha256() == b.sha256()
    assert a.sha256() == m.sha256()
    assert a.to_dict()["train"] != b.to_dict()["train"]   # 溯源仍各自保留


def test_hash_changes_with_domain():
    m = true_model()
    m2 = PowerModel(weights=m.weights, feature_names=m.feature_names, base=m.base,
                    history_k=0, fit_mode="relative", train={},
                    domain={"abs_vx": {"max": 2.0}, "abs_vy": {"max": 1.0},
                            "abs_wz": {"max": 1.0}})
    assert m.sha256() != m2.sha256()


def test_hash_depends_on_schema_and_weights():
    m = true_model()
    names_lin = feature_names("linear", 0)
    m2 = PowerModel(weights=(1.0, 2.0, 3.0, 4.0), feature_names=names_lin, base="linear",
                    history_k=0, fit_mode="relative", train={}, domain={})
    assert m.sha256() != m2.sha256()
    assert m.schema_sha256() != m2.schema_sha256()


# ---------------------------------------------------------------- 统计与选型
def test_q_nearest_rank():
    vals = list(range(1, 11))
    assert q_nearest_rank(vals, 0.9) == 9.0
    assert q_nearest_rank(vals, 1.0) == 10.0
    assert q_nearest_rank([5.0], 0.9) == 5.0
    with pytest.raises(PowerModelError):
        q_nearest_rank([], 0.9)


def test_summarize_shapes():
    s = summarize([0.1, 0.2, 0.3])
    assert s["n"] == 3 and math.isclose(s["median"], 0.2)
    assert summarize([]) == {"n": 0}


def test_select_candidate_prefers_simpler_within_tie():
    cands = {
        "complex": {"dev_q90": 0.0500, "n_params": 10, "history_k": 0},
        "simple": {"dev_q90": 0.0540, "n_params": 4, "history_k": 0},
        "worse": {"dev_q90": 0.0900, "n_params": 2, "history_k": 0},
    }
    assert select_candidate(cands, tie_tol=0.005) == "simple"
    cands["simple"]["dev_q90"] = 0.0700          # 超出 tie 容差 → 只看最小
    assert select_candidate(cands, tie_tol=0.005) == "complex"
    cands["complex"]["dev_q90"] = 0.0400
    cands["simple"]["dev_q90"] = 0.0403          # 0.3 pp 内 → 参数少的赢
    assert select_candidate(cands, tie_tol=0.005) == "simple"


def test_select_candidate_name_breaks_exact_ties():
    cands = {
        "b": {"dev_q90": 0.05, "n_params": 4, "history_k": 0},
        "a": {"dev_q90": 0.05, "n_params": 4, "history_k": 0},
    }
    assert select_candidate(cands) == "a"


# ---------------------------------------------------------------- 评估
def test_evaluate_groups_and_non_computable():
    m = true_model()
    samples = synthetic_samples(12)
    # 构造两条近零样本（命令全 0 → J_nom = bias*30 ≫ 1 J，改用极小截距模型）
    tiny = PowerModel(weights=(0.0001, 0.0, 0.0, 0.0), feature_names=feature_names("linear", 0),
                      base="linear", history_k=0, fit_mode="relative", train={}, domain={})
    ev = evaluate(tiny, samples)
    assert ev["n"] == 12 and ev["n_non_computable"] == 12
    assert set(ev["groups"]) == {"family", "low_speed", "transition"}
    assert set(ev["groups"]["low_speed"]) == {"yes", "no"}

    ev2 = evaluate(m, samples)
    assert ev2["n_non_computable"] == 0
    assert ev2["overall"]["n"] == 12
    assert ev2["overall"]["q90"] < 1e-9
    fams = {r["family"] for r in ev2["rows"]}
    assert fams == set(ev2["groups"]["family"])


def test_admission_thresholds():
    dev = {"overall": {"q90": 0.05}, "n": 80, "n_non_computable": 0}
    val = {"overall": {"q90": 0.08}, "n": 80, "n_non_computable": 0}
    ok, checks = admission(dev, val, threshold=0.10, truncation_share=0.0)
    assert ok and all(checks.values())
    val2 = {"overall": {"q90": 0.12}, "n": 80, "n_non_computable": 0}
    ok2, checks2 = admission(dev, val2, threshold=0.10, truncation_share=0.0)
    assert not ok2 and not checks2["validate_q90_le_threshold"]
    ok3, checks3 = admission(dev, val, threshold=0.10, truncation_share=1e-6)
    assert not ok3 and not checks3["nonneg_constructive"]


def test_low_speed_and_transition_shares():
    still = np.zeros((N_INTERVALS + 1, 3))
    assert low_speed_share(still) == 1.0
    assert transition_share(still) == 0.0
    jumpy = np.tile(np.array([0.0, 0.0, 0.0]), (N_INTERVALS + 1, 1))
    jumpy[::2] = np.array([0.8, 0.0, 0.0])
    assert transition_share(jumpy) > 0.5
    assert low_speed_share(jumpy) < 0.6


def test_eval_sample_rejects_bad_jexec():
    cmd = make_commands(8)
    with pytest.raises(PowerModelError):
        EvalSample(trial_id="x", family="f", j_exec=0.0, commands=cmd)
    with pytest.raises(PowerModelError):
        EvalSample(trial_id="x", family="f", j_exec=float("nan"), commands=cmd)
