"""α/δ 统计与冻结的单元测试（CAL-025）。

覆盖：nearest-rank 口径、负超支保留、bootstrap 的可复现与簇重采样、
α=max(0,·) 的截断、冻结产物散列、预算换算、cell 完整率门槛。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from camra_calibration.margins import (
    BOOTSTRAP_SEED, MarginError, Margins, bootstrap_quantile, build_margins,
    calibration_verdict, group_stats, nearest_rank, percentile_interval,
)


# ---------------------------------------------------------------- 分位
def test_nearest_rank_takes_ceil_index_not_interpolation():
    vals = list(range(1, 11))                    # 1..10
    # ceil(0.9*10)=9 → 第 9 个（升序 9）
    assert nearest_rank(vals, 0.9) == 9.0
    # 与 numpy 的线性插值不同：np.percentile(vals, 90) = 9.1
    assert abs(float(np.percentile(vals, 90)) - 9.1) < 1e-12
    assert nearest_rank(vals, 0.5) == 5.0


def test_nearest_rank_boundary_and_single_sample():
    assert nearest_rank([3.0], 0.9) == 3.0
    assert nearest_rank([1.0, 2.0], 0.5) == 1.0      # ceil(1)=1 → 第 1 个
    assert nearest_rank([5.0, 1.0, 3.0], 1.0) == 5.0
    with pytest.raises(MarginError):
        nearest_rank([], 0.9)
    with pytest.raises(MarginError):
        nearest_rank([1.0, 2.0], 0.0)


def test_nearest_rank_keeps_negative_values():
    vals = [-0.2, -0.1, 0.05, 0.3]
    # ceil(0.9*4)=4 → 最大 0.3；0.5 → 第 2 个 = -0.1（负值参与排序，不被截断）
    assert nearest_rank(vals, 0.9) == 0.3
    assert nearest_rank(vals, 0.5) == -0.1


def test_percentile_interval_uses_same_rule():
    boot = np.arange(1, 101, dtype=float)        # 1..100
    lo, hi = percentile_interval(boot, 0.95)
    assert lo == 3.0 and hi == 98.0              # ceil(0.025*100)=3, ceil(0.975*100)=98


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_is_reproducible_with_same_seed():
    vals = [0.01, 0.02, 0.03, 0.4, -0.05, 0.11, 0.07, 0.02]
    a = bootstrap_quantile(vals, 0.9, replicates=200, seed=BOOTSTRAP_SEED)
    b = bootstrap_quantile(vals, 0.9, replicates=200, seed=BOOTSTRAP_SEED)
    assert a == b
    c = bootstrap_quantile(vals, 0.9, replicates=200, seed=BOOTSTRAP_SEED + 1)
    assert (c["lo"], c["hi"]) != (a["lo"], a["hi"])


def test_bootstrap_interval_brackets_point_and_is_trial_level():
    rng = np.random.default_rng(0)
    vals = list(rng.normal(0.05, 0.02, size=200))
    out = bootstrap_quantile(vals, 0.9, replicates=400, seed=1)
    assert out["unit"] == "trial"
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["n"] == 200


def test_cluster_bootstrap_resamples_whole_blocks():
    """簇重采样：结果只能由"整簇组成"产生，不能把簇拆散。

    两个簇 A=[0,0,0]、B=[5,5,5]，抽 2 个簇只有三种组成：
    A,A → 全 0 → Q90=0；B,B 或 A,B → 含 5 → Q90=5。
    所以 bootstrap 分位只能取 {0.0, 5.0}，区间两端必为这两个值。
    """
    vals = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    clusters = ["A", "A", "A", "B", "B", "B"]
    out = bootstrap_quantile(vals, 0.9, replicates=300, seed=7, clusters=clusters)
    assert out["unit"] == "cluster"
    assert out["lo"] == 0.0 and out["hi"] == 5.0
    # 单轨迹重采样可以把两簇的值混在一起，Q90 也可能取到 5，但区间端点
    # 不必落在整簇组成的集合里——这里只断言两种口径都围绕点估计。
    by_trial = bootstrap_quantile(vals, 0.9, replicates=300, seed=7)
    assert by_trial["lo"] <= 5.0 <= by_trial["hi"] or by_trial["hi"] >= 5.0
    with pytest.raises(MarginError):
        bootstrap_quantile(vals, 0.9, replicates=10, clusters=["A"])


# ---------------------------------------------------------------- α / δ
def test_alpha_clipped_to_zero_but_raw_quantile_kept():
    eps_c = [-0.4, -0.3, -0.2, -0.1, -0.05]          # 全部负超支
    eps_d = [0.2, 0.3, 0.4, 0.5, 0.6]
    m, detail = build_margins(eps_c=eps_c, eps_d=eps_d, stats={}, data={},
                              applicability={}, replicates=50)
    assert m.q90_eps_c == -0.05                      # 原值保留
    assert m.alpha == 0.0                            # 形成裕度时取 max(0,·)
    assert m.alpha_interval[0] == 0.0
    assert detail["eps_c"]["q90_raw"] == -0.05


def test_delta_is_q90_of_eps_d():
    eps_c = [0.01] * 10
    eps_d = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.40]
    m, _ = build_margins(eps_c=eps_c, eps_d=eps_d, stats={}, data={},
                         applicability={}, replicates=50)
    assert m.delta == nearest_rank(eps_d, 0.9) == 0.13
    assert m.q90_eps_d == 0.13
    assert m.delta_interval[0] <= 0.13 <= m.delta_interval[1]


def test_build_margins_rejects_empty_sample():
    with pytest.raises(MarginError):
        build_margins(eps_c=[], eps_d=[0.1], stats={}, data={}, applicability={})


def test_cost_budget_uses_alpha():
    m = Margins(version="1.0", alpha=0.12, delta=0.08,
                alpha_interval=(0.10, 0.15), delta_interval=(0.07, 0.09),
                q90_eps_c=0.12, q90_eps_d=0.08, n_eps_c=10, n_eps_d=10,
                data={}, stats={}, applicability={})
    assert m.cost_budget(1000.0) == pytest.approx(1120.0)
    assert m.geometric_margin() == 0.08
    with pytest.raises(MarginError):
        m.cost_budget(-1.0)


# ---------------------------------------------------------------- 冻结与散列
def test_save_load_roundtrip_and_tamper_detection(tmp_path):
    m, _ = build_margins(eps_c=[0.01, 0.02, 0.03], eps_d=[0.1, 0.2, 0.3],
                         stats={"a": 1}, data={"d": 2}, applicability={"x": 1},
                         replicates=20)
    path, side = m.save(tmp_path / "margins_v1.json")
    assert side.is_file()
    loaded = Margins.load(path)
    assert loaded.alpha == m.alpha and loaded.delta == m.delta
    assert loaded.sha256() == m.sha256()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["alpha"] = 0.9                   # 篡改
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MarginError):
        Margins.load(path)


def test_hash_ignores_provenance_but_tracks_numbers():
    """散列只覆盖数值与适用域：换数据来源不改身份，改 α 就改身份。"""
    eps_c, eps_d = [0.02] * 5, [0.1] * 5
    a, _ = build_margins(eps_c=eps_c, eps_d=eps_d, stats={}, data={"run": 1},
                         applicability={"x": 1}, replicates=20)
    b, _ = build_margins(eps_c=eps_c, eps_d=eps_d, stats={"other": 2}, data={"run": 2},
                         applicability={"x": 1}, replicates=20)
    assert a.sha256() == b.sha256()
    c, _ = build_margins(eps_c=[0.5] * 5, eps_d=eps_d, stats={}, data={},
                         applicability={"x": 1}, replicates=20)
    assert c.sha256() != a.sha256()


# ---------------------------------------------------------------- 分组与准入
def test_group_stats_reports_negative_count():
    out = group_stats([-0.1, 0.0, 0.2, None])
    assert out["n"] == 3 and out["n_negative"] == 1
    assert out["min"] == -0.1 and out["max"] == 0.2
    assert out["q90"] == 0.2


def test_verdict_blocks_when_a_cell_has_low_complete_share():
    per_cell = {"a|clean": {"complete_share": 1.0},
                "b|push_only": {"complete_share": 0.5}}
    out = calibration_verdict(per_cell, non_computable_share=0.01)
    assert out["verdict"] == "blocked"
    assert list(out["cells_below_share"]) == ["b|push_only"]

    ok = calibration_verdict({"a|clean": {"complete_share": 0.95}}, 0.01)
    assert ok["verdict"] == "admitted"
    assert ok["non_computable_share"] == 0.01
