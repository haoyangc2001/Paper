"""Focused v3 checks by the executing agent; no scientific acceptance implied."""

import copy
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
import refit_nav065_boundary as B


@pytest.fixture
def protocol():
    return json.loads(B.PROPOSAL.read_text())


def test_registration_requires_new_quota_approval_and_exact_source_binding(tmp_path, protocol):
    path = tmp_path / "registration.json"

    def write(data):
        path.write_text(json.dumps(data))
        path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())

    write(protocol)
    with pytest.raises(ValueError, match="explicit approval"):
        B.approved_registration(path)
    approved = {**protocol, "approved": True, "status": "APPROVED", "authorization_reference": "fixture"}
    write(approved)
    with pytest.raises(ValueError, match="implementation binding"):
        B.approved_registration(path)
    approved["implementation_hashes"] = {name: B.R.file_sha(NAV / name) for name in B.BOUND_INPUTS}
    write(approved)
    assert B.approved_registration(path)[0] == approved
    approved["resource_limit"]["additional_updates_total_max"] += 1
    write(approved)
    with pytest.raises(ValueError, match="settings differ"):
        B.approved_registration(path)


@pytest.mark.parametrize("target,prediction,gradient_sign", [
    (.003, -.5, -1), (-.003, .5, 1), (0., .5, 1), (.003, 0., -1),
])
def test_sign_gradient_repairs_both_sides_of_zero(protocol, target, prediction, gradient_sign):
    p = torch.tensor([prediction], requires_grad=True)
    loss = B.sign_consistency(p, torch.tensor([target]), protocol["fit"])
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert p.grad.item() * gradient_sign > 0


@pytest.mark.parametrize("values", [[.003, -.003, 0.], [.1, -3., 2.]])
def test_unchanged_targets_have_zero_sign_penalty(protocol, values):
    t = torch.tensor(values)
    prediction = t.clone().requires_grad_(True)
    loss = B.sign_consistency(prediction, t, protocol["fit"])
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_sign_cost_uses_J_budget_divided_by_fixed_scale(protocol, monkeypatch):
    class FixedModel:
        def values(self, obs):
            return torch.full((len(obs),), .998), torch.full((len(obs),), -.02)

    truth = SimpleNamespace(observations=np.zeros((1, 2)), budget_J=np.array([550.]),
                            cost_target=np.array([1.003]), reach_target=np.array([.005]))
    monkeypatch.setattr(B.S, "regression_loss", lambda *a: (torch.tensor(2.), 1))
    loss, terms = B.boundary_loss(FixedModel(), [(("toy", 1), truth)], protocol["fit"],
                                  550., np.random.default_rng(1))
    # Cost residuals: prediction=-.002, target=.003. Reach: -.02 versus .005.
    expected_sign = ((.005 / .01) ** 2 + (.025 / .01) ** 2) / 2
    assert terms["sign_loss"] == pytest.approx(expected_sign, abs=1e-5)
    assert loss.item() == pytest.approx(2. + .1 * expected_sign, abs=1e-5)


def rows(actual_feasible_count=17):
    records = []
    for i in range(18):
        feasible = i < actual_feasible_count
        reach = -1. if feasible else 1.
        records.append({"map_id": str(i), "k": 1.15, "actual_feasible": feasible,
                        "predicted_feasible": feasible, "clipped_budget_rows": 0,
                        "start_budget_clipped": False,
                        "rows": {"W": [reach], "path_W": [reach], "vc_norm": [2.],
                                 "vc_target_norm": [2.], "vh": [reach], "vh_target": [reach]}})
    return records


def train_reg():
    reg = json.loads((NAV / "config/nav065_deployment_prereg_v1.json").read_text())
    reg["grid_k"] = [1.15]
    return reg


def test_perfect_recognition_accepts_17_and_rejects_actual_timeout():
    report = B.recognition_summary(rows(), {str(i) for i in range(18)}, train_reg())
    assert report["pass"]
    assert report["primary_actual_feasible"] == 17
    assert report["primary_total"] == 18
    assert report["by_budget"]["1.15"]["true_infeasible"] == 1


@pytest.mark.parametrize("failure", ["false_feasible", "false_infeasible", "reject_all"])
def test_corrected_train_gate_cannot_hide_wrong_decisions(failure):
    records = rows()
    if failure == "false_feasible":
        records[-1]["predicted_feasible"] = True
    elif failure == "false_infeasible":
        records[0]["predicted_feasible"] = False
    else:
        for r in records:
            r["predicted_feasible"] = False
    report = B.recognition_summary(records, {str(i) for i in range(18)}, train_reg())
    assert not report["pass"]
    assert any("feasible" in reason for reason in report["failures"])


def test_nonprimary_false_infeasible_remains_a_diagnostic():
    records = rows(18)
    for r in copy.deepcopy(records):
        r["k"] = 1.
        if r["map_id"] == "0":
            r["predicted_feasible"] = False
        records.append(r)
    reg = train_reg()
    reg["grid_k"] = [1., 1.15]
    report = B.recognition_summary(records, {str(i) for i in range(18)}, reg)
    assert report["pass"]
    assert report["by_budget"]["1.0"]["false_infeasible"] == 1


class ToyModel(torch.nn.Module):
    ROLES = ("policy", "critic_cost", "critic_reach")

    def __init__(self):
        super().__init__()
        for role in self.ROLES:
            setattr(self, role, torch.nn.Linear(2, 1))

    def values(self, obs):
        return self.critic_cost(obs).squeeze(-1), self.critic_reach(obs).squeeze(-1)

    def parameters_of(self, role):
        return list(getattr(self, role).parameters())


def test_combined_loss_and_one_synthetic_update_freeze_actor(protocol, monkeypatch, tmp_path):
    torch.manual_seed(7)
    model = ToyModel()
    truth = SimpleNamespace(observations=np.asarray([[1., 2.], [2., 3.]]),
                            cost_target=np.asarray([1., .5]), reach_target=np.asarray([.003, -.5]),
                            path_residual=np.asarray([.003, -.1]), budget_J=np.asarray([549., 270.]))
    records = [(("toy", 1.15), truth)]
    before = B.R.digests(model)
    protocol["resource_limit"]["additional_updates_per_candidate_max"] = 1
    protocol["fit"]["gate_interval"] = 1
    reports = iter([{"pass": False}, {"pass": True}])
    monkeypatch.setattr(B, "train_summary", lambda *a: next(reports))
    with (tmp_path / "synthetic.jsonl").open("x") as journal:
        updates, report = B.fit(model, records, protocol, {}, 550.,
                                {"run_id": "nav064_terminal_tail_v1_full_s1", "seed": 1}, journal)
    after = B.R.digests(model)
    assert updates == 1 and report["pass"]
    assert after["policy"] == before["policy"]
    assert all(after[role] != before[role] for role in B.R.SCORE_ROLES)
    assert all(not p.requires_grad for p in model.policy.parameters())


def test_passing_source_has_zero_updates_and_no_optimizer(protocol, monkeypatch, tmp_path):
    model = ToyModel()
    before = B.R.digests(model)
    monkeypatch.setattr(B, "train_summary", lambda *a: {"pass": True})

    def forbidden(*args, **kwargs):
        pytest.fail("passing source must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with (tmp_path / "preserved.jsonl").open("x") as journal:
        assert B.fit(model, [], protocol, {}, 550., {"run_id": "passing"}, journal)[0] == 0
    assert before == B.R.digests(model)


def test_unapproved_failed_candidate_cannot_update(protocol, monkeypatch, tmp_path):
    monkeypatch.setattr(B, "train_summary", lambda *a: {"pass": False})
    with (tmp_path / "denied.jsonl").open("x") as journal:
        with pytest.raises(ValueError, match="outside approved"):
            B.fit(ToyModel(), [], protocol, {}, 550., {"run_id": "unexpected"}, journal)


@pytest.mark.parametrize("all_pass", [False, True])
def test_actual_run_blocks_development_until_all_six_pass(tmp_path, protocol, monkeypatch, all_pass):
    model = ToyModel()
    prepared = [({"run_id": f"toy{i}", "checkpoint": "source", "checkpoint_sha256": "sha",
                  "variant": "full", "seed": i}, model, [], {}) for i in range(6)]
    inputs = SimpleNamespace(value={"quantities": {"Z_REF_J": {"value": 550}}},
                             train_pin="fixture", prereg={"implementation_versions": {}})
    monkeypatch.setattr(B, "fit", lambda *a: (0, {"pass": all_pass}))
    monkeypatch.setattr(B.R, "load_network", lambda *a: model)
    calls = []
    monkeypatch.setattr(B.R, "evaluate", lambda *a: calls.append(a))
    train = {"train_episodes": 540, "train_steps": 207298}
    upstream = {"base_evaluation_pin": "original"}
    out = tmp_path / "attempt"
    B.run(protocol, "pin", upstream, out, {}, inputs, train, prepared)
    assert len(calls) == int(all_pass)
    assert (out / "HELD_TRAIN_GATE.json").exists() != all_pass
    if all_pass:
        assert calls[0][1] is upstream
        assert calls[0][-2:] == (540, 207298)
