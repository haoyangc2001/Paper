"""Regression tests for the reproduction entry points (TASK-035).

Every test here pins a defect that was actually shipped once, or a judging rule
whose arithmetic must not drift. The old behaviour is described in the test body
so a future reader can tell whether the assertion still means anything.

Covered defects
    1. ``config/repro.env`` resolution order (env > file > default). If this
       inverts, a run silently uses another machine's asset root.
    2. ``dev_it*<suffix>`` glob also matched the 10-seed sweeps
       (``dev_it2999_s100_r4``), and the selection then died parsing the
       iteration number (found while replaying TASK-023).
    3. ``int(row["failed"])`` raised on ``"False"`` from a CSV, and a flag
       parser that maps unknown text to False would silently read "True" as
       "did not fail" -- that is a 0 % failure rate for a run where everything
       fell over.
    4. A tree hash that ignores added/removed files cannot detect a swapped
       file set.
    5. Readiness must never report a missing checkpoint as a controller result,
       and an unimportable torch outranks a missing resource.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

LOCOMOTION = Path(__file__).resolve().parents[1]
SCRIPTS = LOCOMOTION / "scripts"
sys.path.insert(0, str(SCRIPTS))

import artifact_index  # noqa: E402
import camra_config  # noqa: E402
import check_repro_readiness as readiness  # noqa: E402
from camra_config import Config, ConfigError  # noqa: E402

PROTOCOL = LOCOMOTION / "docs" / "eval_protocol.json"


# --------------------------------------------------------------------------- 1
def test_config_environment_wins_over_the_file(tmp_path, monkeypatch):
    env_file = tmp_path / "repro.env"
    env_file.write_text("CAMRA_LOG_ROOT=/from/file\n", encoding="utf-8")
    monkeypatch.setenv("CAMRA_LOG_ROOT", "/from/env")
    config = Config(env_path=env_file)
    assert str(config.path("CAMRA_LOG_ROOT")) == "/from/env"
    assert config.source("CAMRA_LOG_ROOT") == "env"


def test_config_file_wins_over_the_builtin_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMRA_LOG_ROOT", raising=False)
    env_file = tmp_path / "repro.env"
    env_file.write_text("# comment\n\nCAMRA_LOG_ROOT=/from/file\n", encoding="utf-8")
    config = Config(env_path=env_file)
    assert str(config.path("CAMRA_LOG_ROOT")) == "/from/file"
    assert config.source("CAMRA_LOG_ROOT") == "file"


def test_config_falls_back_to_the_delivery_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMRA_ASSET_ROOT", raising=False)
    config = Config(env_path=tmp_path / "does-not-exist.env")
    assert config.source("CAMRA_ASSET_ROOT") == "default"
    assert str(config.path("CAMRA_ASSET_ROOT")) == camra_config.DEFAULTS["CAMRA_ASSET_ROOT"]


def test_config_refuses_to_invent_a_value(tmp_path):
    config = Config(env_path=tmp_path / "does-not-exist.env")
    with pytest.raises(ConfigError):
        config.raw("CAMRA_KEY_THAT_DOES_NOT_EXIST")


def test_go2_paths_follow_the_configured_asset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMRA_ASSET_ROOT", str(tmp_path))
    config = Config(env_path=tmp_path / "does-not-exist.env")
    # the asset tree name must stay "Go2": the bundle copies exactly this subtree
    assert config.asset_go2().parts[-1] == "Go2"
    assert config.go2_usd().parts[-3:] == ("Go2", "usd", "go2.usd")


# --------------------------------------------------------------------------- 2
def _write_dev_summary(path: Path, scenarios: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({"scenarios": scenarios}), encoding="utf-8"
    )


def _scenario(sid: str, rmse_vx: float, failure: float = 0.0) -> dict:
    return {"scenario": sid, "pass": True, "failure_rate": failure,
            "rmse_vx": rmse_vx, "rmse_vy": rmse_vx, "rmse_wz": rmse_vx}


def test_selection_ignores_per_seed_sweep_directories(tmp_path):
    """``dev_it2999_s100_r4`` is a 10-seed dev check, not iteration "2999_s100"."""
    dev_root = tmp_path / "devsweep"
    _write_dev_summary(dev_root / "dev_it500_r4", [_scenario("fwd_low", 0.30)])
    _write_dev_summary(dev_root / "dev_it2999_r4", [_scenario("fwd_low", 0.05)])
    for seed in (100, 101):
        _write_dev_summary(dev_root / f"dev_it2999_s{seed}_r4", [_scenario("fwd_low", 0.01)])
    (dev_root / "dev_it2999_r4.log").write_text("not a directory\n", encoding="utf-8")

    out = tmp_path / "selection.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "repro" / "select_checkpoint.py"),
         "--dev-root", str(dev_root), "--suffix", "_r4",
         "--protocol", str(PROTOCOL), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selected_iteration"] == 2999
    assert [c["iteration"] for c in payload["candidates"]] == [500, 2999]
    # the skipped directories must be visible, not silently dropped
    assert "dev_it2999_s100_r4" in result.stdout
    assert "not considered" in result.stdout


def test_selection_reports_when_no_candidate_meets_the_constraint(tmp_path):
    dev_root = tmp_path / "devsweep"
    _write_dev_summary(dev_root / "dev_it500_r0", [_scenario("fwd_low", 0.05, failure=0.30)])
    out = tmp_path / "selection.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "repro" / "select_checkpoint.py"),
         "--dev-root", str(dev_root), "--suffix", "_r0",
         "--protocol", str(PROTOCOL), "--out", str(out)],
        capture_output=True, text=True,
    )
    # exit 4 = selected but the acceptance constraint failed; never a silent pick
    assert result.returncode == 4
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == \
        "no_candidate_meets_failure_constraint"


# --------------------------------------------------------------------------- 3
def test_failed_flag_survives_text_forms(tmp_path):
    """``0``/``1`` and ``False``/``True`` must mean the same thing, not 0 % failures.

    The fixture deliberately stays off the exact threshold: the judge compares with
    ``value <= limit`` in binary floating point, and ``0.1+0.2+0.1+0.2`` is
    ``0.15000000000000002``, i.e. one ulp above 0.15. That is a property of the
    judge (unchanged since the protocol was fixed), not something to paper over
    with a tolerance here.
    """
    root = tmp_path / "finaltest"
    seed = root / "nominal" / "s200"
    seed.mkdir(parents=True)
    rows = [
        {"scenario": "fwd_low", "failed": "0", "rmse_vx": 0.10, "p95_abs_err_vx": 0.20},
        {"scenario": "fwd_low", "failed": "1", "rmse_vx": 0.19, "p95_abs_err_vx": 0.20},
        {"scenario": "fwd_low", "failed": "False", "rmse_vx": 0.10, "p95_abs_err_vx": 0.20},
        {"scenario": "fwd_low", "failed": "True", "rmse_vx": 0.19, "p95_abs_err_vx": 0.20},
    ]
    with (seed / "per_episode.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "aggregate_final_test.py"),
         "--root", str(root), "--groups", "nominal",
         "--protocol", str(PROTOCOL), "--json-out", str(tmp_path / "agg.json")],
        capture_output=True, text=True,
    )
    # the fixture holds one scenario only, so the tool must refuse a clean PASS (exit 3);
    # the arithmetic below is still computed and recorded, which is what this test checks
    assert result.returncode == 3
    assert "INCOMPLETE" in result.stderr
    payload = json.loads((tmp_path / "agg.json").read_text(encoding="utf-8"))
    check = payload["groups"]["nominal"]["scenarios"]["fwd_low"]["checks"]
    # 2 of 4 episodes flagged: 0.50, computed by hand, not "0 of 4"
    assert check["failure_rate"]["value"] == pytest.approx(0.50)
    assert check["failure_rate"]["pass"] is False
    # hand-computed mean of 0.10, 0.19, 0.10, 0.19 = 0.145, below the 0.15 limit
    assert check["rmse_vx"]["value"] == pytest.approx(0.145)
    assert check["rmse_vx"]["limit"] == pytest.approx(0.15)
    assert check["rmse_vx"]["pass"] is True


def test_missing_group_is_incomplete_not_a_pass(tmp_path):
    runs_root = tmp_path / "runs"
    (runs_root / "nominal" / "s300").mkdir(parents=True)
    (runs_root / "nominal" / "s300" / "per_episode.csv").write_text(
        "scenario,variant_id,failed,rmse_vx\nfwd_low,0,0,0.05\n", encoding="utf-8"
    )
    out = tmp_path / "agg.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "aggregate_eval_runs.py"),
         "--runs_root", str(runs_root), "--seeds", "300",
         "--groups", "nominal,perturbed", "--protocol", str(PROTOCOL),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    # exit 2 = incomplete: the perturbed group has no runs at all
    assert result.returncode == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert "perturbed" in json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- 4
def test_tree_hash_sees_added_removed_and_changed_files(tmp_path):
    tree = tmp_path / "bundle"
    (tree / "params").mkdir(parents=True)
    (tree / "params" / "deploy.yaml").write_text("kp: 25.0\n", encoding="utf-8")
    baseline = artifact_index.hash_tree(tree)

    assert artifact_index.hash_tree(tree) == baseline  # stable across calls

    (tree / "params" / "deploy.yaml").write_text("kp: 30.0\n", encoding="utf-8")
    changed = artifact_index.hash_tree(tree)
    assert changed["tree_sha256"] != baseline["tree_sha256"]

    (tree / "extra.txt").write_text("x\n", encoding="utf-8")
    added = artifact_index.hash_tree(tree)
    assert added["tree_sha256"] != changed["tree_sha256"]
    assert added["files"] == changed["files"] + 1


def test_index_verify_detects_a_swapped_file(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"entries": [{
        "id": "protocol",
        "kind": "file",
        "required": True,
        "note": "tampered on purpose",
        "path": str(PROTOCOL),
        "exists": True,
        "sha256": "0" * 64,
        "bytes": 1,
    }]}), encoding="utf-8")
    assert artifact_index.verify(index) == 4


def test_index_verify_refuses_a_missing_index(tmp_path):
    assert artifact_index.verify(tmp_path / "nope.json") == 2


# --------------------------------------------------------------------------- 5
def test_missing_resource_is_infra_blocked_not_a_verdict():
    findings = [
        {"check": "checkpoint", "ok": False, "blocking": True},
        {"check": "asset_root", "ok": True, "blocking": True},
    ]
    assert readiness.classify(findings) == ("infra_blocked", 2, ["checkpoint"])


def test_broken_environment_outranks_a_missing_resource():
    findings = [
        {"check": "checkpoint", "ok": False, "blocking": True},
        {"check": "torch_import", "ok": False, "blocking": True},
    ]
    verdict, code, blocking = readiness.classify(findings)
    assert (verdict, code) == ("env_incompatible", 3)
    assert blocking == ["checkpoint", "torch_import"]


def test_advisory_findings_never_block():
    findings = [
        {"check": "checkpoint", "ok": True, "blocking": True},
        {"check": "tmux", "ok": False, "blocking": False},
    ]
    assert readiness.classify(findings) == ("ready", 0, [])


# --------------------------------------------------------------------------- 6
def test_repro_scripts_do_not_assign_bash_reserved_variables():
    """``GROUPS`` is a bash special array holding the caller's group ids.

    Assignment to it is silently ignored, so ``finaltest.sh`` passed
    ``groups='39400005'`` to the evaluator and every run died with
    ``argument --group: invalid choice: '39400005'`` -- after the checkpoint hash
    and the provenance stamp had already been written, which makes it look like a
    simulator problem. Neither bash -n nor shellcheck flags it.
    """
    reserved = {
        "GROUPS", "UID", "EUID", "PPID", "RANDOM", "SECONDS", "LINENO",
        "FUNCNAME", "BASH_SOURCE", "PIPESTATUS", "REPLY", "HOSTNAME",
    }
    pattern = re.compile(
        r"^\s*(?:(?:local|export|readonly)\s+)?(" + "|".join(sorted(reserved)) + r")="
    )
    offenders = []
    for script in sorted((SCRIPTS / "repro").glob("*.sh")):
        for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = pattern.match(line)
            if match:
                offenders.append(f"{script.name}:{number}: {stripped}")
    assert offenders == [], "reserved bash names assigned (assignment is ignored):\n" + \
        "\n".join(offenders)
