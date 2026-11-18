"""Focused verification of projected directions and finite-step safeguards."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refit_nav065_guarded as G


def test_conflicting_regression_cannot_cancel_sign_descent():
    pc, ph = torch.nn.Parameter(torch.tensor([1., 1.])), torch.nn.Parameter(torch.tensor([1., 1.]))
    regression = -pc[0] + pc[1] - ph[0] + ph[1]
    signs = (pc[0], ph[0])
    result = G.balanced_gradients(regression, signs, {"critic_cost": [pc], "critic_reach": [ph]})
    assert all(x["gradient_cosine"] < 0 for x in result.values())
    assert pc.grad[0] > 0 and ph.grad[0] > 0
    assert pc.grad[1] == 1 and ph.grad[1] == 1


def test_zero_sign_gradient_preserves_regression_direction():
    pc, ph = torch.nn.Parameter(torch.tensor([2.])), torch.nn.Parameter(torch.tensor([3.]))
    G.balanced_gradients(pc.square().sum() + ph.square().sum(), (pc.sum() * 0, ph.sum() * 0),
                         {"critic_cost": [pc], "critic_reach": [ph]})
    assert pc.grad.item() == 4 and ph.grad.item() == 6


@pytest.mark.parametrize("failure", ["vh relative MAE failed or undefined", "vc_norm Pearson failed or undefined",
                                    "known feasible primary starts were not accepted"])
def test_trial_rejects_numerical_or_primary_regression(failure):
    p = np.array([True, False])
    assert not G.trial_allowed({"failures": [failure]}, p, p, np.ones(2, dtype=bool))[0]


def test_locked_decision_cannot_be_broken_but_unknown_can_be_fixed():
    actual = np.array([True, False, False])
    locked = np.array([True, True, False])
    report = {"failures": ["false feasible budget at episode start"]}
    assert not G.trial_allowed(report, np.array([True, True, False]), actual, locked)[0]
    assert G.trial_allowed(report, actual, actual, locked)[0]


class Toy(torch.nn.Module):
    ROLES = ("policy", "critic_cost", "critic_reach")

    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Linear(1, 1)
        self.critic_cost = torch.nn.Linear(1, 1)
        self.critic_reach = torch.nn.Linear(1, 1)

    def parameters_of(self, role):
        return list(getattr(self, role).parameters())

    def values(self, x):
        return self.critic_cost(x).squeeze(-1), self.critic_reach(x).squeeze(-1)


def test_exhausted_line_search_restores_all_tensors(tmp_path, monkeypatch):
    model = Toy()
    before = G.R.digests(model)
    records = [(("map", 1.), SimpleNamespace(observations=np.array([[1.]], dtype=np.float32),
               budget_J=np.array([550.]), cost_target=np.array([1.01]), reach_target=np.array([.01]),
               path_residual=np.array([.01]), feasible=False))]
    settings = {"learning_rate": 1e-4, "sampling_seed": 1, "backtracks_max": 2,
                "backtrack_factor": .5, "interior_samples_per_episode": 1,
                "boundary_absolute_W_max": .05, "sign_boundary_absolute_max": .05,
                "sign_margin_cap": .01, "sign_normalization": .01}
    protocol = {"fit": settings, "resource_limit": {"optimizer_step_trials_max": 2}}
    report = {"pass": False, "failures": ["false feasible budget at episode start"]}
    monkeypatch.setattr(G.B, "train_summary", lambda *a: report)
    monkeypatch.setattr(G, "trial_allowed", lambda *a: (False, "forced_guard_rejection"))
    with (tmp_path / "journal").open("x") as journal:
        n, final = G.fit(model, records, protocol, {}, 550., {"run_id": G.TARGET, "seed": 1}, journal)
    assert n == 0 and not final["pass"]
    assert G.R.digests(model) == before
    assert (tmp_path / "journal").read_text().count('"kind": "trial"') == 2


def test_passing_source_does_not_create_optimizer(tmp_path, monkeypatch):
    model = Toy()
    before = G.R.digests(model)
    monkeypatch.setattr(G.B, "train_summary", lambda *a: {"pass": True})
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("passing source updated"))
    with (tmp_path / "journal").open("x") as journal:
        assert G.fit(model, [], {}, {}, 550., {"run_id": "passing"}, journal)[0] == 0
    assert G.R.digests(model) == before


def test_registration_cannot_be_an_unapproved_or_unbound_proposal(tmp_path):
    import json
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({"schema": "camra-nav065-guarded-refit-v4", "approved": False}))
    path.with_suffix(".json.sha256").write_text(G.R.file_sha(path))
    with pytest.raises(ValueError, match="approved new trial quota"):
        G.approved_registration(path)
