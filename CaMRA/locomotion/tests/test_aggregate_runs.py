"""Cross-seed aggregation and completeness gating (TASK-032).

Pure Python: builds a throw-away run tree (``<root>/<group>/s<seed>/per_episode.csv``)
and checks that missing seeds / groups / variants are reported as incomplete
instead of silently disappearing into an average.

Run with::

    python -m pytest CaMRA/locomotion/tests/test_aggregate_runs.py -q
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

LOCOMOTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCOMOTION / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(LOCOMOTION / "scripts"))

_spec = importlib.util.spec_from_file_location("aggregate_eval_runs", LOCOMOTION / "scripts" / "aggregate_eval_runs.py")
agg_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg_runs)

PROTOCOL = {
    "version": "test",
    "scenarios": [
        {
            "id": "lateral",
            "duration_s": 10.0,
            "command_program": {"type": "constant", "vx": 0.0, "vy": 0.3, "wz": 0.0, "vy_sign": "both"},
            "metrics": ["rmse_vy", "p95_abs_err_vy", "failure_rate"],
            "thresholds": {"rmse_vy": 0.1, "p95_abs_err_vy": 0.18},
        }
    ],
}

FIELDS = ["scenario", "variant_id", "env_seed", "failed", "rmse_vy", "p95_abs_err_vy",
          "warmup_short", "invalid_reasons", "steps"]


def write_run(root: Path, group: str, seed: int, *, rmse: float = 0.05, variants=(0, 1), failed=0) -> None:
    d = root / group / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for vi in variants:
        for _ in range(5):
            rows.append({
                "scenario": "lateral",
                "variant_id": str(vi),
                "env_seed": str(seed),
                "failed": str(failed),
                "rmse_vy": str(rmse),
                "p95_abs_err_vy": "0.10",
                "warmup_short": "False",
                "invalid_reasons": "{}",
                "steps": "500",
            })
    with open(d / "per_episode.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def run(tmp_path: Path, seeds: str = "300-301", expected: int | None = None) -> tuple[int, dict]:
    out = tmp_path / "aggregate.json"
    proto = tmp_path / "protocol.json"
    proto.write_text(json.dumps(PROTOCOL), encoding="utf-8")
    argv = [
        "aggregate_eval_runs.py",
        "--runs_root", str(tmp_path / "runs"),
        "--seeds", seeds,
        "--groups", "nominal",
        "--protocol", str(proto),
        "--out", str(out),
    ]
    if expected is not None:
        argv += ["--expected_per_variant", str(expected)]
    old = sys.argv
    sys.argv = argv
    try:
        rc = agg_runs.main()
    finally:
        sys.argv = old
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_complete_runs_aggregate_across_seeds(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "nominal", 300)
    write_run(root, "nominal", 301)
    rc, report = run(tmp_path)
    assert rc == 0
    assert report["complete"] is True and report["overall_pass"] is True
    lat = report["scenarios"][0]
    assert lat["episodes"] == 20  # 2 seeds x 2 variants x 5 episodes
    assert [v["episodes"] for v in lat["variants"]] == [10, 10]
    assert all(v["seeds_present"] == [300, 301] for v in lat["variants"])


def test_missing_seed_is_reported_not_averaged(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "nominal", 300)  # seed 301 never ran
    rc, report = run(tmp_path)
    assert rc == 2
    assert report["complete"] is False
    assert report["overall_pass"] is False
    assert report["missing_runs"] == [{"group": "nominal", "seed": 301, "path": str(root / "nominal" / "s301")}]


def test_missing_variant_blocks_the_scenario(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "nominal", 300, variants=(0,))
    write_run(root, "nominal", 301, variants=(0,))
    rc, report = run(tmp_path)
    assert rc == 2
    lat = report["scenarios"][0]
    assert lat["complete"] is False
    missing = [v for v in lat["variants"] if v.get("status") == "missing"]
    assert [v["variant_id"] for v in missing] == [1]


def test_threshold_failure_returns_one_not_two(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "nominal", 300, rmse=0.30)
    write_run(root, "nominal", 301, rmse=0.30)
    rc, report = run(tmp_path)
    assert rc == 1  # complete but failing
    assert report["complete"] is True
    assert report["overall_pass"] is False
    statuses = {c["status"] for c in report["scenarios"][0]["variants"][0]["checks"].values()}
    assert "fail" in statuses


def test_short_of_quota_is_incomplete(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "nominal", 300)
    write_run(root, "nominal", 301)
    rc, report = run(tmp_path, expected=100)
    assert rc == 2
    assert all(v["status"] == "short_of_quota" for v in report["scenarios"][0]["variants"])


def test_csv_text_cells_do_not_break_the_maths(tmp_path):
    """'failed' arrives as text: '0' must not be read as truthy, '1' must count."""
    root = tmp_path / "runs"
    write_run(root, "nominal", 300, failed=1)
    write_run(root, "nominal", 301, failed=0)
    rc, report = run(tmp_path)
    lat = report["scenarios"][0]
    assert lat["variants"][0]["failure_rate"] == pytest.approx(0.5)  # 5 of 10 episodes
    assert lat["variants"][0]["checks"]["failure_rate"]["status"] == "fail"
    assert rc == 1
