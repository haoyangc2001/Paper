"""Main-agent basic tests: fixed features, sign precision, approval and isolation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_nav065_head as H


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Tanh())
        self.head = torch.nn.Linear(3, 1)

    def forward(self, obs):
        return self.head(self.body(obs)).squeeze(-1)


class Model(torch.nn.Module):
    ROLES = ('policy', 'critic_cost', 'critic_reach')

    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Linear(1, 1)
        self.critic_cost = Critic()
        self.critic_reach = Critic()

    def values(self, obs):
        return self.critic_cost(obs), self.critic_reach(obs)


def test_readonly_spectrum_detects_unreachable_conflicting_labels_without_solving(monkeypatch):
    monkeypatch.setattr(np.linalg, 'lstsq', lambda *a, **kw: pytest.fail('no fitting in diagnosis'))
    a = np.array([[1., 1.], [1., 1.]])
    y = np.array([-3., .001])
    a_before, y_before = a.copy(), y.copy()
    report = H.spectrum(a, y, np.zeros(2), 1e-12)
    assert report['rank_at_solve_rcond'] == 1
    assert report['target_residual_outside_retained_span_linf'] > 1
    assert not report['candidate_weights_solved'] and report['candidate_parameters_assigned'] == 0
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(y, y_before)


def test_full_row_rank_expressibility_is_separate_from_float32_condition():
    a = np.array([[1., 0.], [1., 1e-8]])
    report = H.spectrum(a, np.array([-3., .001]), np.zeros(2), 1e-12)
    assert report['rank_at_solve_rcond'] == 2
    assert report['rank_at_float32_tolerance'] == 1
    assert report['minimum_correction_norm_for_retained_projection'] > 1e8


@pytest.mark.parametrize('target,rcond', [(np.array([np.nan, 1.]), 1e-12), (np.ones(3), 1e-12), (np.ones(2), 0.)])
def test_invalid_spectral_inputs_are_rejected(target, rcond):
    with pytest.raises(ValueError, match='spectral'):
        H.spectrum(np.eye(2), target, np.zeros(2), rcond)


def test_one_solve_fits_positive_tiny_and_deep_negative_labels_without_changing_body():
    torch.manual_seed(65)
    model = Critic()
    before = copy.deepcopy(model.state_dict())
    x = np.array([[0.], [.01]], np.float32)
    target = np.array([.0021, -3.])
    report = H.solve_head(model, x, target, 1e-12)
    pred = model(torch.as_tensor(x)).detach().numpy()
    assert pred[0] > 0 and pred[1] < 0
    np.testing.assert_allclose(pred, target, atol=2e-5)
    assert report['head_solves'] == 1
    for name, value in before.items():
        if name.startswith('body.'):
            assert torch.equal(value, model.state_dict()[name])


def test_rank_deficient_solve_reports_error_instead_of_making_a_false_exact_claim():
    model = Critic()
    report = H.solve_head(model, np.zeros((2, 1)), np.array([-3., .001]), 1e-12)
    assert report['rank'] == 1
    assert report['production_float32_residual_linf'] > 1.


def test_nonfinite_target_stops_before_head_change():
    model = Critic()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='invalid head'):
        H.solve_head(model, np.zeros((2, 1)), np.array([np.nan, 1.]), 1e-12)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_proposal_is_rejected_before_source_loading_or_output(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['probe', '--registration', str(H.PROPOSAL)])
    monkeypatch.setattr(H.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setattr(H.L, 'sources', lambda: pytest.fail('unapproved candidate loaded'))
    with pytest.raises(ValueError, match='explicit approval'):
        H.main()


def test_approved_registration_checks_settings_and_source_hashes(tmp_path, monkeypatch):
    proposal = H.R.load(H.PROPOSAL)
    source = tmp_path / 'input'
    source.write_text('fixed')
    monkeypatch.setattr(H.R, 'NAV', tmp_path)
    monkeypatch.setattr(H, 'BOUND_INPUTS', ('input',))
    proposal_path = tmp_path / 'proposal.json'
    proposal_path.write_text(json.dumps(proposal))
    monkeypatch.setattr(H, 'PROPOSAL', proposal_path)
    protocol = {**proposal, 'approved': True, 'status': 'APPROVED', 'authorization_reference': 'synthetic only',
        'implementation_hashes': {'input': hashlib.sha256(source.read_bytes()).hexdigest()}}
    path = tmp_path / 'registration.json'

    def write():
        path.write_text(json.dumps(protocol))
        path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())

    write()
    assert H.approved_registration(path)[0] == protocol
    protocol['solve']['rcond'] = 1e-4
    write()
    with pytest.raises(ValueError, match='differs'):
        H.approved_registration(path)
    protocol['solve']['rcond'] = 1e-12
    write()
    source.write_text('modified')
    with pytest.raises(ValueError):
        H.approved_registration(path)


def prepared(tmp_path, monkeypatch):
    torch.manual_seed(65)
    source = Model()
    truth = NS(observations=np.array([[0.], [1.]], np.float32), budget_J=np.array([550., 550.]),
        cost_target=np.array([.9, .8]), reach_target=np.array([-3., -3.]))
    c = {'run_id': H.L.P.TARGET}
    rows = [(c, source, [(('map', 1.), truth)], [])]
    diagnostic = {'local_keys': [['map', 1.]], 'source_full_train_gate': {'pass': False}}
    monkeypatch.setattr(H.R, 'WORK', tmp_path)
    monkeypatch.setattr(H, 'diagnose', lambda *a: diagnostic)
    monkeypatch.setattr(H.R, 'evaluate', lambda *a: pytest.fail('development forbidden'))
    monkeypatch.setattr(H.L.P.A, 'train_gate', lambda *a: {'pass': False, 'regression_reported': True})
    def reload(*args):
        net = copy.deepcopy(source)
        net.load_state_dict(torch.load(args[-1], weights_only=False)['model'])
        return net
    monkeypatch.setattr(H.R, 'load_network', reload)
    return source, rows, NS(value={'quantities': {'Z_REF_J': {'value': 550.}}})


def test_actual_small_probe_reports_full_gate_and_does_not_deploy_even_if_local_passes(tmp_path, monkeypatch):
    source, rows, inputs = prepared(tmp_path, monkeypatch)
    before = copy.deepcopy(source.state_dict())
    settings = H.R.load(H.PROPOSAL)
    reg = {'criteria': {'rel_mae_max': .35, 'pearson_min': .7, 'variance_ratio_min': .01}}
    H.run(settings, 'synthetic', reg, inputs, rows)
    out = tmp_path / settings['output_name']
    result = H.R.load(out / 'HEAD_FINISHED.json')
    assert result['local']['local_pass']
    assert not result['full_train_after']['pass'] and result['full_train_after']['regression_reported']
    assert not result['scientific_nav065_pass'] and not result['deployment_allowed']
    assert result['development_scoring_rounds'] == result['optimizer_steps'] == 0
    assert result['actor_and_bodies_unchanged'] and not result['automatic_expansion_allowed']
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in before.items())
    assert len((out / 'attempts.jsonl').read_text().splitlines()) == 2
    with pytest.raises(ValueError, match='output exists'):
        H.run(settings, 'synthetic', reg, inputs, rows)


def test_partial_solve_failure_is_preserved_without_retry(tmp_path, monkeypatch):
    source, rows, inputs = prepared(tmp_path, monkeypatch)
    solver = H.solve_head
    calls = []
    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('synthetic failure')
        return solver(*args)
    monkeypatch.setattr(H, 'solve_head', fail_second)
    settings = H.R.load(H.PROPOSAL)
    with pytest.raises(ValueError, match='synthetic failure'):
        H.run(settings, 'synthetic', None, inputs, rows)
    out = tmp_path / settings['output_name']
    assert len(calls) == 2 and (out / 'failed-head.pt').is_file()
    assert not H.R.load(out / 'STOPPED.json')['automatic_restart_allowed']
    assert not (out / 'HEAD_FINISHED.json').exists()


@pytest.mark.parametrize('field,value', [('update_candidates', ['other']), ('head_linear_solves_total_max', 3), ('optimizer_steps_total_max', 1), ('new_train_episodes_max', 1), ('development_scoring_rounds_max', 1)])
def test_scope_expansion_fails_before_loading_or_outputs(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(H.R, 'WORK', tmp_path)
    monkeypatch.setattr(H, 'diagnose', lambda *a: pytest.fail('expanded run not allowed'))
    protocol = H.R.load(H.PROPOSAL)
    protocol['resource_limit'][field] = value
    with pytest.raises(ValueError, match='scope or quota'):
        H.run(protocol, 'synthetic', None, None, [])
    assert not list(tmp_path.iterdir())
