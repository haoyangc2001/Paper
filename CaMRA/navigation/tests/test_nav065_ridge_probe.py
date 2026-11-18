"""Main-agent basic verification of global data, ridge stability and run boundaries."""
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
import probe_nav065_ridge as G


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Identity()
        self.head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(.1)
            self.head.bias.zero_()

    def forward(self, obs):
        return self.head(self.body(obs)).squeeze(-1)


class Model(torch.nn.Module):
    ROLES = ('policy', 'critic_cost', 'critic_reach')

    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Linear(1, 1)
        self.critic_cost, self.critic_reach = Critic(), Critic()

    def values(self, obs):
        return self.critic_cost(obs), self.critic_reach(obs)


def truth(x, reach):
    n = len(x)
    return NS(observations=np.asarray(x, np.float32).reshape(-1, 1), budget_J=np.full(n, 550.),
              cost_target=np.linspace(.8, .2, n), reach_target=np.asarray(reach, np.float64),
              path_residual=np.linspace(-.005, .005, n), feasible=True)


def settings():
    return copy.deepcopy(G.R.load(G.PROPOSAL)['solve'])


def test_full_batch_keeps_every_row_and_balances_groups_and_sign_strata():
    old = [(('old', 1.), truth([0, 1, 2], [-3, -3, .002]))]
    new = [(('new', 1.), truth([3, 4], [-3, .001]))]
    before = [t.reach_target.copy() for _, t in old + new]
    batch, weights, groups = G.training_batch(old, new,
        {'boundary_absolute_W_max': .01, 'interior_samples_per_episode': 2})
    np.testing.assert_array_equal(batch[0][:5, 0], [0, 1, 2, 3, 4])
    assert sum(g['rows'] for g in groups) == len(weights)
    assert [g['name'] for g in groups] == list(G.GROUPS)
    assert groups[2]['rows'] == 2
    assert weights.sum() == pytest.approx(1.)
    for g in groups:
        s = slice(g['start'], g['stop'])
        assert weights[s].sum() == pytest.approx(.25)
        if 0 < g['reach_positive_rows'] < g['rows']:
            assert weights[s][batch[3][s] > 0].sum() == pytest.approx(.125)
    for (_, t), target in zip(old + new, before):
        np.testing.assert_array_equal(t.reach_target, target)


def test_empty_or_duplicate_train_keys_fail():
    row = (('same', 1.), truth([0, 1], [-3, .01]))
    with pytest.raises(ValueError, match='unique'):
        G.training_batch([row], [row], {})
    with pytest.raises(ValueError, match='nonempty'):
        G.training_batch([row], [], {})


def test_readonly_design_never_solves_or_changes_parameters(monkeypatch):
    model = Critic()
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(np.linalg, 'lstsq', lambda *a, **kw: pytest.fail('readonly fitted a head'))
    report = G.weighted_design(model, [[0.], [1.]], [-3., .001], [.5, .5], settings())[-1]
    assert not report['candidate_weights_solved'] and report['candidate_parameters_assigned'] == 0
    assert report['correction_norm_bound_for_objective_no_worse_than_source'] > 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_regularized_solution_matches_small_independent_quadratic_and_limits_displacement():
    model = Critic()
    x, y, p, s = np.array([[-1.], [0.], [1.]]), np.array([.001, -3., .001]), np.full(3, 1/3), settings()
    a, y, p, w, d = G.weighted_design(model, x, y, p, s)
    q = p / d['target_rms'] ** 2
    expected_delta = np.linalg.solve(a.T @ (q[:, None] * a) + d['ridge_coefficient'] * np.eye(len(w)),
                                     a.T @ (q * (y - a @ w)))
    result = G.solve_head(model, x, y, p, s)
    np.testing.assert_allclose(G.H.head_vector(model), w + expected_delta, atol=1e-6)
    assert result['head_solves'] == 1 and result['objective_guard_pass']
    assert result['regularized_objective_after_float32_cast'] <= result['source_objective'] + 1e-6
    assert result['actual_float32_correction_norm'] <= result['correction_norm_bound_for_objective_no_worse_than_source'] + 1e-5
    # Contradictory linear shape cannot be made exact; never report exact interpolation.
    assert np.abs(model(torch.as_tensor(x, dtype=torch.float32)).detach().numpy() - y).max() > .1


@pytest.mark.parametrize('field,value', [('ridge_strength', 0.), ('ridge_strength', np.nan),
    ('target_rms_floor', -1.), ('head_norm_floor', 0.), ('feature_batch_rows', 0), ('rcond', 0.)])
def test_bad_ridge_settings_fail_before_head_change(field, value):
    model, s = Critic(), settings()
    before = copy.deepcopy(model.state_dict())
    s[field] = value
    with pytest.raises(ValueError, match='invalid'):
        G.solve_head(model, [[0.], [1.]], [-3., .001], [.5, .5], s)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize('target,weights', [([np.nan, 1.], [.5, .5]), ([-3., .001], [0., 1.]),
                                          ([-3., .001], [.5, .6])])
def test_invalid_targets_or_weights_fail(target, weights):
    with pytest.raises(ValueError, match='invalid'):
        G.solve_head(Critic(), [[0.], [1.]], target, weights, settings())


def test_unapproved_run_fails_before_loading_candidates(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['probe', '--registration', str(G.PROPOSAL)])
    monkeypatch.setattr(G.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setattr(G.L, 'sources', lambda: pytest.fail('unapproved candidate loaded'))
    with pytest.raises(ValueError, match='explicit approval'):
        G.main()


def test_approval_requires_exact_proposal_and_all_bound_inputs(tmp_path, monkeypatch):
    proposal = G.R.load(G.PROPOSAL)
    source = tmp_path / 'input'
    source.write_text('fixed')
    monkeypatch.setattr(G.R, 'NAV', tmp_path)
    monkeypatch.setattr(G, 'BOUND_INPUTS', ('input',))
    proposal_path = tmp_path / 'proposal.json'
    proposal_path.write_text(json.dumps(proposal))
    monkeypatch.setattr(G, 'PROPOSAL', proposal_path)
    protocol = {**proposal, 'approved': True, 'status': 'APPROVED', 'authorization_reference': 'synthetic only',
        'implementation_hashes': {'input': hashlib.sha256(source.read_bytes()).hexdigest()}}
    path = tmp_path / 'registration.json'
    def write():
        path.write_text(json.dumps(protocol))
        path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    write()
    assert G.approved_registration(path)[0] == protocol
    protocol['solve']['ridge_strength'] *= 2
    write()
    with pytest.raises(ValueError, match='differs'):
        G.approved_registration(path)
    protocol['solve']['ridge_strength'] /= 2
    write()
    source.write_text('different')
    with pytest.raises(ValueError):
        G.approved_registration(path)


def prepared(tmp_path, monkeypatch):
    source = Model()
    t = truth([0, 1, 2], [-3, -3, .001])
    proposal = G.R.load(G.PROPOSAL)
    c = {'run_id': G.L.P.TARGET, 'checkpoint_sha256': proposal['source_checkpoint_sha256']}
    rows = [(c, source, [(('old', 1.15), t)], [(('new', 1.), t)])]
    proposal['expected_train_episodes'] = {'original': 1, 'additional': 1, 'total_steps': 6}
    original_load = G.R.load
    monkeypatch.setattr(G.R, 'load', lambda p: copy.deepcopy(proposal) if p == G.PROPOSAL else original_load(p))
    monkeypatch.setattr(G.R, 'WORK', tmp_path)
    monkeypatch.setattr(G.R, 'evaluate', lambda *a: pytest.fail('development forbidden'))
    monkeypatch.setattr(G.L.P.S, 'start_snapshot', lambda *a: [{'classification': 'false_feasible'}])
    monkeypatch.setattr(G.L.P.A, 'train_gate', lambda *a: {'pass': False, 'full_production_gate_called': True})
    monkeypatch.setattr(G.L, 'diagnose', lambda *a: ({'full_train_gate': {'pass': False},
        'local_source_evaluation': {'local_pass': False}}, [('old', 1.15)]))
    def reload(*args):
        net = copy.deepcopy(source)
        net.load_state_dict(torch.load(args[-1], weights_only=False)['model'])
        return net
    monkeypatch.setattr(G.R, 'load_network', reload)
    return source, rows, NS(value={'quantities': {'Z_REF_J': {'value': 550.}}}), proposal


def test_small_real_probe_checks_global_gate_and_preserves_source_and_no_deployment(tmp_path, monkeypatch):
    source, rows, inputs, proposal = prepared(tmp_path, monkeypatch)
    before = copy.deepcopy(source.state_dict())
    reg = {'criteria': {'rel_mae_max': .35, 'pearson_min': .7, 'variance_ratio_min': .01}, 'primary_k': 1.15}
    G.run(proposal, 'synthetic', reg, inputs, rows)
    out = tmp_path / proposal['output_name']
    report = G.R.load(out / 'RIDGE_FINISHED.json')
    assert report['full_train_after']['full_production_gate_called'] and not report['full_train_after']['pass']
    assert len(report['groups_after']) == 4
    assert not report['scientific_nav065_pass'] and not report['deployment_allowed'] and not report['automatic_expansion_allowed']
    assert report['optimizer_steps'] == report['new_rollouts'] == report['development_scoring_rounds'] == 0
    assert report['actor_and_bodies_unchanged'] and all(torch.equal(v, source.state_dict()[k]) for k, v in before.items())
    assert len((out / 'attempts.jsonl').read_text().splitlines()) == 2
    with pytest.raises(ValueError, match='output exists'):
        G.run(proposal, 'synthetic', reg, inputs, rows)


def test_second_solve_failure_preserves_partial_artifacts_without_retry(tmp_path, monkeypatch):
    _, rows, inputs, proposal = prepared(tmp_path, monkeypatch)
    solver, calls = G.solve_head, []
    def fail_second(*a):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('synthetic second solve failure')
        return solver(*a)
    monkeypatch.setattr(G, 'solve_head', fail_second)
    with pytest.raises(ValueError, match='second solve'):
        G.run(proposal, 'synthetic', None, inputs, rows)
    out = tmp_path / proposal['output_name']
    assert len(calls) == 2 and (out / 'failed-ridge.pt').is_file()
    assert not G.R.load(out / 'STOPPED.json')['automatic_restart_allowed']
    assert not (out / 'RIDGE_FINISHED.json').exists()


@pytest.mark.parametrize('field,value', [('head_linear_solves_total_max', 3), ('optimizer_steps_total_max', 1),
                                      ('new_train_episodes_max', 1), ('development_scoring_rounds_max', 1)])
def test_quota_expansion_fails_before_any_source_or_output(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(G.R, 'WORK', tmp_path)
    monkeypatch.setattr(G, 'diagnose', lambda *a: pytest.fail('scope guard too late'))
    proposal = G.R.load(G.PROPOSAL)
    proposal['resource_limit'][field] = value
    with pytest.raises(ValueError, match='scope, quota'):
        G.run(proposal, 'synthetic', None, None, [])
    assert not list(tmp_path.iterdir())
