"""Small checks for adaptive collection, unchanged targets, guards and G4 wiring."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refit_nav065_adaptive as A


def test_crossing_priority_and_repeatable_lower_tie():
    k, bracket = A.next_budget({.7: False, .85: False, 1.: False, 1.15: True, 1.3: True})
    assert k == pytest.approx(1.075) and bracket["crossing"]
    k, bracket = A.next_budget({.7: False, .85: False, 1.: False, 1.15: False, 1.3: False})
    assert k == pytest.approx(.775) and not bracket["crossing"]


def test_reversal_is_kept_and_sampler_does_not_claim_monotonicity():
    intervals = A.adjacent_intervals({.7: False, .85: True, 1.: False, 1.15: True, 1.3: True})
    assert sum(x["reversal"] for x in intervals) == 1
    assert sum(x["crossing"] for x in intervals) == 3
    points = {.7: True, .85: False, 1.: True, 1.15: False, 1.3: True}
    for _ in range(4):
        k, _ = A.next_budget(points)
        assert k not in points and .7 < k < 1.3
        points[k] = k > 1.1


@pytest.mark.parametrize("points", [{.6: False, 1.: True}, {.7: False}, {.7: False, 1.3: 1}])
def test_invalid_sampling_points_stop(points):
    with pytest.raises(ValueError):
        A.next_budget(points)


def truth(k=1., n=3):
    return NS(observations=np.array([[k], [k + .01], [k + .02]], dtype=np.float32)[:n],
              budget_J=np.array([550., 549., 548.])[:n],
              cost_target=np.array([1.001, 1., .99])[:n],
              reach_target=np.array([.001, -.001, -3.])[:n],
              path_residual=np.array([.001, .002, -3.])[:n],
              stop_index=np.full(n, n, dtype=np.int64),
              budget_feature_clipped=np.zeros(n, dtype=bool), feasible=False)


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


def settings():
    return A.R.load(A.PROPOSAL)["fit"]


def test_boundary_rows_include_start_and_interior_without_changing_targets():
    t = truth()
    before = copy.deepcopy(t)
    batch = A.selected_rows([(("map", 1.), t)], settings())
    assert batch[0].shape == (2, 1)
    np.testing.assert_array_equal(batch[2], before.cost_target[:2])
    np.testing.assert_array_equal(t.reach_target, before.reach_target)
    np.testing.assert_array_equal(t.budget_J, before.budget_J)


def test_exact_targets_have_zero_auxiliary_loss_and_false_joint_has_gradient():
    batch = (np.array([[1.]], dtype=np.float32), np.array([550.]), np.array([1.001]), np.array([.001]))
    model = Toy()
    for role, value in (("critic_cost", 1.001), ("critic_reach", .001)):
        getattr(model, role).weight.data.zero_()
        getattr(model, role).bias.data.fill_(value)
    assert sum(A.sign_losses(model, batch, settings(), 550.)).item() == pytest.approx(0., abs=1e-8)
    model.critic_cost.bias.data.fill_(.999)
    model.critic_reach.bias.data.fill_(-.001)
    losses = A.sign_losses(model, batch, settings(), 550.)
    model.zero_grad()
    sum(losses).backward()
    assert model.critic_reach.bias.grad.item() < 0
    assert model.critic_cost.bias.grad.item() < 0
    assert all(p.grad is None for p in model.policy.parameters())


def test_unapproved_proposal_is_rejected_before_preflight_or_output():
    with pytest.raises(ValueError, match="explicit approval"):
        A.approved_registration(A.PROPOSAL)


def test_coverage_report_is_json_serializable_with_numpy_targets():
    records = [(("map", k), truth(k)) for k in (.7, .85, 1., 1.15, 1.3)]
    report = A.coverage([({"run_id": "test"}, None, records, {"pass": True})], 550.)
    assert json.loads(json.dumps(report, allow_nan=False))[0]["maps"][0]["starts_abs_W_le_0_01"] == 5


def test_additional_good_rows_cannot_hide_failure_on_original_grid(monkeypatch):
    old_gate = {"pass": False, "failures": ["vh relative MAE failed or undefined"]}
    monkeypatch.setattr(A.B, "train_summary", lambda *args: old_gate)
    monkeypatch.setattr(A.B.S, "predict", lambda model, obs: (np.full(len(obs), 1.001), np.full(len(obs), .001)))
    t = truth(n=1)
    result = A.train_gate(Toy(), [(("map", 1.), t)], [(("map", 1.075), t)],
        {"criteria": {"variance_ratio_min": .5, "rel_mae_max": .35, "pearson_min": .7}}, settings(), 550.)
    assert not result["pass"] and old_gate["failures"][0] in result["failures"]
    assert result["additional_metrics"]["vc_norm"]["relative_mae"] == 0


def test_changed_registration_settings_rejected_even_with_valid_self_pin(tmp_path):
    p = A.R.load(A.PROPOSAL)
    p.update(approved=True, status="APPROVED", authorization_reference="synthetic test only")
    p["resource_limit"]["new_train_episodes_max"] += 1
    path = tmp_path / "reg.json"
    path.write_text(json.dumps(p))
    path.with_suffix(".json.sha256").write_text(A.R.file_sha(path))
    with pytest.raises(ValueError, match="differs from reviewed"):
        A.approved_registration(path)


def test_collection_reexecutes_each_changed_budget_and_counts_attempts(tmp_path, monkeypatch):
    model, calls = Toy(), []
    before = A.R.digests(model)
    candidate = {"run_id": "test", "variant": "full"}
    old = [(("train", k), NS(feasible=k >= 1.15)) for k in (.7, .85, 1., 1.15, 1.3)]
    maps = [{"map_id": "train", "level": "L1", "seed": 1}]
    inputs = NS(prereg={"environment": {"t_max_s": .3}})
    p = A.R.load(A.PROPOSAL)
    monkeypatch.setattr(A.R.E, "build_rollout_ctx", lambda method: {"design": {}})
    monkeypatch.setattr(A.R.E.M, "build", lambda *a: "spec")
    monkeypatch.setattr(A.R, "consistent_e_ref", lambda e, v: e)

    def rollout(net, spec, ctx, budget, prereg):
        calls.append(budget)
        return {"n_steps": 3, "budget": budget}

    def reconstruct(ep, *args):
        t = truth(ep["budget"] / 550)
        t.feasible = ep["budget"] >= 600
        return t

    monkeypatch.setattr(A.R, "rollout", rollout)
    monkeypatch.setattr(A.R, "reconstruct", reconstruct)
    total = {"episodes": 0, "steps": 0}
    with (tmp_path / "journal").open("x") as journal:
        new = A.collect(model, candidate, old, maps, {"train": 550.}, inputs, p, journal, total)
    assert len(calls) == len(new) == 4 and len(set(calls)) == 4
    assert total == {"episodes": 4, "steps": 12}
    assert A.R.digests(model) == before
    p["resource_limit"]["new_train_episodes_max"] = 4
    with (tmp_path / "quota").open("x") as journal:
        with pytest.raises(ValueError, match="quota exhausted"):
            A.collect(model, candidate, old, maps, {"train": 550.}, inputs, p, journal, total)
    assert len(calls) == 4


def gate(passed=False):
    return {"pass": passed, "failures": [] if passed else ["selected boundary row signs"],
            "original_grid": {"failures": []}, "additional_false_feasible": 0,
            "additional_metrics": {}, "selected_sign_mismatches": 0 if passed else 1}


def test_rejected_trials_restore_both_critics_and_count_all_steps(tmp_path, monkeypatch):
    model = Toy()
    before = A.R.digests(model)
    monkeypatch.setattr(A, "train_gate", lambda *args: gate())
    monkeypatch.setattr(A, "allowed_trial", lambda *args: False)
    p = A.R.load(A.PROPOSAL)
    p["resource_limit"]["optimizer_trials_per_candidate_max"] = 2
    with (tmp_path / "journal").open("x") as journal:
        n, updates, result = A.fit(model, [(("map", 1.), truth())], [], p, {}, 550., 0, journal)
    assert n == 2 and updates == 0 and not result["pass"]
    assert A.R.digests(model) == before
    assert (tmp_path / "journal").read_text().count('"kind": "trial"') == 2


def test_passing_model_keeps_all_tensors_and_creates_no_optimizer(tmp_path, monkeypatch):
    model = Toy()
    before = A.R.digests(model)
    monkeypatch.setattr(A, "train_gate", lambda *args: gate(True))
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("unexpected optimizer"))
    with (tmp_path / "journal").open("x") as journal:
        assert A.fit(model, [], [], A.R.load(A.PROPOSAL), {}, 550., 0, journal)[:2] == (0, 0)
    assert A.R.digests(model) == before


def test_trial_preserves_old_numerical_gate_locked_starts_and_boundary_progress():
    p = np.array([True, False])
    reg = {"criteria": {"rel_mae_max": .35, "pearson_min": .7}}
    assert A.allowed_trial(gate(), gate(True), p, p, np.ones(2, bool), reg)
    trial = gate(True)
    trial["original_grid"]["failures"] = ["vh relative MAE failed or undefined"]
    assert not A.allowed_trial(gate(), trial, p, p, np.ones(2, bool), reg)
    assert not A.allowed_trial(gate(), gate(True), ~p, p, np.ones(2, bool), reg)


@pytest.mark.parametrize("passed", [False, True])
def test_actual_run_blocks_or_calls_original_g4_only_after_frozen_models(tmp_path, monkeypatch, passed):
    model, calls = Toy(), []
    p = A.R.load(A.PROPOSAL)
    p["resource_limit"]["new_train_episodes_max"] = 1
    monkeypatch.setattr(A.R, "WORK", tmp_path)
    candidate = {"run_id": "test", "seed": 0, "variant": "full", "checkpoint": "parent", "checkpoint_sha256": "abc"}

    def collect(*args):
        args[-1].update(episodes=1, steps=3)
        return []

    def evaluate(out, *args):
        assert (out / "fitted_candidates.json").is_file()
        assert (out / "train_gate.json").is_file()
        calls.append("original G4")

    monkeypatch.setattr(A, "collect", collect)
    monkeypatch.setattr(A, "fit", lambda *args: (0, 0, gate(passed)))
    monkeypatch.setattr(A.R, "evaluate", evaluate)
    monkeypatch.setattr(A.R, "load_network", lambda *args: model)
    inputs = NS(train_pin="pin", prereg={"implementation_versions": {}},
                value={"quantities": {"Z_REF_J": {"value": 550.}}})
    A.run(p, "pin", {}, {}, inputs, {"train_episodes": 540, "train_steps": 207298}, [], {},
          [(candidate, model, [], {})])
    assert calls == (["original G4"] if passed else [])
    assert (tmp_path / p["output_name"] / "HELD_TRAIN_GATE.json").exists() != passed
