"""Boundary-loss arithmetic and saved-only refit fail-closed checks."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

NAV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV / "scripts"))
import refit_nav065_saved as S


def test_proposal_cannot_authorize_extra_updates(tmp_path):
    path = tmp_path / "proposal.json"
    data = json.loads(S.PROPOSAL.read_text())
    path.write_text(json.dumps(data))
    path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="explicit approval"):
        S.approved_registration(path)
    data.update(approved=True, status="APPROVED", authorization_reference="fixture")
    path.write_text(json.dumps(data))
    path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="implementation binding"):
        S.approved_registration(path)


def test_boundary_term_uses_true_W_and_unchanged_two_critic_targets():
    def trajectory(cost, reach, residual):
        return SimpleNamespace(observations=np.zeros((2, 1)), cost_target=np.asarray(cost),
                               reach_target=np.asarray(reach), path_residual=np.asarray(residual))

    class ZeroModel:
        def values(self, obs):
            return torch.zeros(len(obs)), torch.zeros(len(obs))

    # start MSE=(2 + 8)/2=5, interior MSE=(.5+4.5)/2=2.5,
    # boundary MSE=2, hence (5+2.5+2)/3. No predicted sign selects samples.
    records = [(('a', 1), trajectory([2, 1], [0, 0], [.01, .01])),
               (('b', 1), trajectory([0, 0], [-4, -3], [-.2, -.2]))]
    settings = {"interior_samples_per_episode": 1, "boundary_absolute_W_max": .05}
    loss, n = S.regression_loss(ZeroModel(), records, settings, np.random.default_rng(0))
    assert n == 1
    assert loss.item() == pytest.approx((5 + 2.5 + 2) / 3)
    settings["boundary_absolute_W_max"] = .001
    loss, n = S.regression_loss(ZeroModel(), records, settings, np.random.default_rng(0))
    assert n == 0
    assert loss.item() == pytest.approx((5 + 2.5) / 2)


def test_saved_train_grid_rejects_unknown_duplicate_and_missing(tmp_path):
    inputs = SimpleNamespace(candidates=[{"run_id": "r"}])
    maps = [{"map_id": "train-a"}]
    path = tmp_path / "train_episodes.jsonl"
    row = {"kind": "rollout_result", "run_id": "r", "map_id": "train-a", "k": 1, "episode": {}}

    def read(rows):
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        manifest = {"train_journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return S.saved_train(inputs, maps, [1], tmp_path, manifest)

    assert len(read([row])["r"]) == 1
    with pytest.raises(ValueError, match="duplicate or unknown"):
        read([row, row])
    with pytest.raises(ValueError, match="duplicate or unknown"):
        read([{**row, "map_id": "development-a"}])
    with pytest.raises(ValueError, match="incomplete"):
        read([])


def test_failed_train_attempt_never_scores_development(tmp_path, monkeypatch):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(1, 1)

    candidates = [{"run_id": f"r{i}", "seed": i, "variant": "full"} for i in range(6)]
    inputs = SimpleNamespace(candidates=candidates,
                             value={"quantities": {"Z_REF_J": {"value": 550}}},
                             train_pin="fixture", prereg={"implementation_versions": {}})
    manifest = {"candidates": [{**c, "checkpoint": "fixture", "checkpoint_sha256": "fixture"}
                               for c in candidates], "train_episodes": 6, "train_steps": 6}
    raw = {c["run_id"]: {("train-a", 1): {}} for c in candidates}
    monkeypatch.setattr(S, "saved_train", lambda *a: raw)
    monkeypatch.setattr(S.R.E.M, "load_design", lambda *a: {})
    monkeypatch.setattr(S.R.E.M, "build", lambda *a: None)
    monkeypatch.setattr(S.R, "load_network", lambda *a: ToyModel())
    monkeypatch.setattr(S.R, "digests", lambda *a: {"policy": "fixture"})
    monkeypatch.setattr(S.R, "reconstruct", lambda *a: SimpleNamespace(budget_feature_clipped=np.array([False])))
    monkeypatch.setattr(S, "fit", lambda *a: (300, {"pass": False}))

    def forbidden(*args):
        pytest.fail("development was accessed after failed train gates")

    monkeypatch.setattr(S.R, "evaluate", forbidden)
    out = tmp_path / "attempt"
    S.run({}, "pin", {}, out, inputs, [{"map_id": "train-a", "level": "l1", "seed": 1}],
          {"grid_k": [1]}, tmp_path, manifest)
    assert (out / "HELD_TRAIN_GATE.json").exists()
    assert not (out / "development_scores.jsonl").exists()
