"""Synthetic-only tests for NAV-065 deployment acceptance aggregation."""

import sys
from pathlib import Path

import pytest

NAV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV / "scripts"))
from evaluate_nav065_deployment import EPISODE_GRID, read_registration, review_documents, summarize


def fixture():
    maps = [f"dev{i}" for i in range(6)]
    reg = {"grid_k": list(EPISODE_GRID), "primary_k": 1.15,
           "criteria": {"max_false_feasible": 0, "min_primary_true_feasible": 6,
                        "rel_mae_max": .35, "pearson_min": .7, "variance_ratio_min": .5}}
    rows = []
    for k in EPISODE_GRID:
        for m in maps:
            feasible = k == 1.15
            rows.append({
                "k": k, "map_id": m, "actual_feasible": feasible,
                "predicted_feasible": feasible, "start_budget_clipped": False,
                "clipped_budget_rows": 1 if k == .7 else 0,
                "rows": {"vc_norm": [.2, .1], "vc_target_norm": [.21, .09],
                         "vh": [.2, .1], "vh_target": [.21, .09],
                         "W": [.2, .1], "path_W": [.21, .09]},
            })
    return rows, maps, reg


def test_gate_passes_only_with_conservative_budget_decisions_and_metric_baseline():
    rows, maps, reg = fixture()
    report = summarize(rows, maps, reg)
    assert report["pass"]
    assert report["by_budget"]["1.15"]["true_feasible"] == 6
    assert report["by_budget"]["0.7"]["false_feasible"] == 0
    assert report["clipped_budget_rows"] == 6
    assert report["clipped_initial_budget_inputs"] == 0


def test_any_false_feasible_start_fails_the_candidate():
    rows, maps, reg = fixture()
    rows[0]["predicted_feasible"] = True
    report = summarize(rows, maps, reg)
    assert not report["pass"]
    assert "false feasible budget at episode start" in report["failures"]


def test_primary_feasible_coverage_prevents_always_reject_policy():
    rows, maps, reg = fixture()
    for row in rows:
        row["predicted_feasible"] = False
    report = summarize(rows, maps, reg)
    assert not report["pass"]
    assert "known feasible primary starts were not accepted" in report["failures"]


def test_metric_gate_keeps_pearson_variance_conditional():
    rows, maps, reg = fixture()
    for row in rows:
        row["rows"]["vc_norm"] = [1.0, 1.0]
        row["rows"]["vc_target_norm"] = [1.0, 1.0]
    report = summarize(rows, maps, reg)
    assert report["pass"]
    assert not report["metrics"]["vc_norm"]["pearson_required"]
    for row in rows:
        row["rows"]["vc_norm"] = [1.0, .1]
        row["rows"]["vc_target_norm"] = [.1, 1.0]
    report = summarize(rows, maps, reg)
    assert not report["pass"]
    assert "vc_norm Pearson failed or undefined" in report["failures"]


def test_a_clipped_initial_budget_is_rejected_but_later_clipping_is_diagnostic():
    rows, maps, reg = fixture()
    rows[0]["start_budget_clipped"] = True
    report = summarize(rows, maps, reg)
    assert not report["pass"]
    assert "initial budget input clipped" in report["failures"]


def test_missing_or_duplicate_episode_cannot_shrink_gate_denominator():
    rows, maps, reg = fixture()
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        summarize(rows[:-1], maps, reg)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        summarize(rows[:-1] + [rows[0]], maps, reg)


def test_refuses_to_open_evaluation_without_registration(tmp_path):
    with pytest.raises(ValueError, match="approved registration"):
        read_registration(tmp_path / "unapproved.json")


def test_gate_only_prepares_bundle_with_six_passing_candidates_and_full_coverage():
    rows, maps, reg = fixture()
    candidates = [{"run_id": f"mock{idx}"} for idx in range(6)]
    score = {
        "candidates": [{**row, **summarize(rows, maps, reg)} for row in candidates],
        "new_episodes": 72, "train_pin": "synthetic-train", "eval_pin": "synthetic-eval",
    }
    evidence = {"NAV-063": {}, "NAV-064": {}, "contracts": {}}

    def review():
        return review_documents(score, candidates, evidence, "synthetic-result", {})

    gate, bundle = review()
    assert gate["verdict"] == "PASS"
    assert bundle["release_state"] == "attempt_review"
    assert bundle["n_candidates"] == 6
    score["candidates"][-1]["pass"] = False
    gate, bundle = review()
    assert gate["verdict"] == "FAIL"
    assert bundle is None
    score["candidates"][-1]["pass"] = True
    score["new_episodes"] = 71
    gate, bundle = review()
    assert not gate["checks"]["evaluation_coverage"]
    assert bundle is None
