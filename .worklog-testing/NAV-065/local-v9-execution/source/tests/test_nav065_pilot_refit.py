"""Main-agent basic checks of v8's regression direction and pilot boundaries."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refit_nav065_pilot as P
from test_nav065_block_refit import Toy, truth
from test_nav065_staged_refit import gate, REG


def protocol():
    p = P.R.load(P.PROPOSAL)
    p['fit']['numeric_steps_max'] = p['fit']['boundary_steps_max'] = 2
    p['fit']['numeric_review_interval'] = 1
    p['resource_limit']['optimizer_trials_total_max'] = 4
    return p


def records():
    return [(('map', 1.15), truth(False)), (('map', 1.0), truth(True))]


def patch_gates(monkeypatch, reports):
    values = iter(reports)
    monkeypatch.setattr(P.A, 'train_gate', lambda *a: copy.deepcopy(next(values)))


@pytest.mark.parametrize('sign', [[-1., 0.], [-100., 50.], [1e-9, -1e-9], [0., 0.], [1., 3.]])
def test_direction_protects_regression_and_never_amplifies_small_sign_gradient(sign):
    r, s = torch.tensor([1., 2.], dtype=torch.float64), torch.tensor(sign, dtype=torch.float64)
    d, stats = P.compose_direction(r, s, .25)
    assert torch.dot(r, d) >= torch.dot(r, r) - 1e-12
    assert (d - r).norm() <= .25 * r.norm() + 1e-12
    assert 0 <= stats['sign_multiplier'] <= 1
    assert (d - r).norm() <= s.norm() + 1e-12


def test_antiparallel_sign_cannot_reverse_numeric_descent():
    r = torch.tensor([1., 2.], dtype=torch.float64)
    d, _ = P.compose_direction(r, -r, .25)
    torch.testing.assert_close(d, r)


def test_zero_regression_has_no_unbounded_sign_only_step():
    d, _ = P.compose_direction(torch.zeros(2), torch.ones(2), .25)
    assert torch.count_nonzero(d) == 0


@pytest.mark.parametrize('fraction', [-.1, 1.1, float('nan')])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match='invalid'):
        P.compose_direction(torch.ones(2), torch.ones(2), fraction)


def test_nonfinite_direction_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        P.compose_direction(torch.ones(2), torch.tensor([float('nan'), 1.]), .25)


def test_actual_sgd_step_reduces_conflicting_quadratic_regression():
    pc, ph = torch.nn.Parameter(torch.tensor([1., 2.])), torch.nn.Parameter(torch.tensor([1., 2.]))
    regression = pc.square().sum() + ph.square().sum()
    before = float(regression.detach())
    stats = P.regression_first_gradients(regression, (-pc.sum(), -ph.sum()),
        {'critic_cost': [pc], 'critic_reach': [ph]}, .25)
    torch.optim.SGD([pc, ph], lr=.001, momentum=0.).step()
    assert float(pc.square().sum() + ph.square().sum()) < before
    assert all(s['gradient_cosine'] < 0 and s['regression_dot_direction'] > 0 for s in stats.values())


def test_exact_float32_aliases_detect_opposite_targets_without_changing_truth():
    data = records()
    saved = copy.deepcopy(data)
    report = P.observation_aliases(data)
    assert report['rows'] == 2 and report['unique_inputs'] == 1
    assert report['reach_target']['opposite_sign_groups'] == 1
    assert report['path_residual']['opposite_sign_groups'] == 1
    for (_, a), (_, b) in zip(data, saved):
        np.testing.assert_array_equal(a.observations, b.observations)
        np.testing.assert_array_equal(a.reach_target, b.reach_target)
    data[1][1].observations[0, 0] += .01
    assert P.observation_aliases(data)['reach_target']['opposite_sign_groups'] == 0


def test_numeric_cap_does_not_force_boundary_or_transfer_quota(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.8), gate(.7), gate(.6)])
    monkeypatch.setattr(torch.optim, 'SGD', lambda *a, **kw: pytest.fail('boundary must not start'))
    model = Toy()
    actor = P.R.digests(model)['policy']
    with (tmp_path / 'log').open('x') as stream:
        trials, result, reason = P.fit(model, records(), [], protocol(), REG, 550., 0, stream)
    assert trials == 2 and reason == 'numeric_gate_not_met' and not result['pass']
    assert P.R.digests(model)['policy'] == actor


def test_numeric_gate_then_sgd_and_first_complete_pass(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.8), gate(.3), gate(.2, True)])
    recorded = []
    sgd = torch.optim.SGD
    def optimizer(*a, **kw):
        recorded.append(kw)
        return sgd(*a, **kw)
    monkeypatch.setattr(torch.optim, 'SGD', optimizer)
    with (tmp_path / 'log').open('x') as stream:
        trials, result, reason = P.fit(Toy(), records(), [], protocol(), REG, 550., 0, stream)
    assert trials == 2 and result['pass'] and reason == 'train_pass'
    assert recorded[0]['momentum'] == 0
    rows = [json.loads(x) for x in (tmp_path / 'log').read_text().splitlines()]
    steps = [r for r in rows if r['kind'] == 'trial']
    assert [r['stage'] for r in steps] == ['numeric', 'boundary']
    assert steps[-1]['gradient_diagnostics'] is not None and len(steps[-1]['starts']) == 2


def test_boundary_numeric_regression_holds_without_rollback_or_retry(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.3), gate(.4)])
    model = Toy()
    before = P.R.digests(model)
    with (tmp_path / 'log').open('x') as stream:
        trials, result, reason = P.fit(model, records(), [], protocol(), REG, 550., 0, stream)
    assert trials == 1 and reason == 'boundary_numeric_regression' and not result['pass']
    assert P.R.digests(model) != before
    assert P.R.digests(model)['policy'] == before['policy']


def test_complete_source_pass_does_not_create_an_optimizer(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.2, True)])
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **kw: pytest.fail('unexpected optimizer'))
    monkeypatch.setattr(torch.optim, 'SGD', lambda *a, **kw: pytest.fail('unexpected optimizer'))
    with (tmp_path / 'log').open('x') as stream:
        assert P.fit(Toy(), records(), [], protocol(), REG, 550., 0, stream)[0] == 0


def test_nonfinite_gradient_stops_before_optimizer_attempt(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.8)])
    model = Toy()
    model.critic_reach.bias.register_hook(lambda g: g * float('nan'))
    before = P.R.digests(model)
    with (tmp_path / 'log').open('x') as stream, pytest.raises(ValueError, match='gradient'):
        P.fit(model, records(), [], protocol(), REG, 550., 0, stream)
    assert P.R.digests(model) == before
    assert 'optimizer_attempt' not in (tmp_path / 'log').read_text()


def test_unapproved_entry_stops_before_loading_sources(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['pilot', '--registration', str(P.PROPOSAL)])
    monkeypatch.setattr(P.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setattr(P, 'sources', lambda: pytest.fail('source access before approval'))
    with pytest.raises(ValueError, match='explicit approval'):
        P.main()


@pytest.mark.parametrize('passed', [False, True])
def test_pilot_never_scores_development_or_changes_other_candidates(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(P.R, 'WORK', tmp_path)
    monkeypatch.setattr(P.R, 'evaluate', lambda *a: pytest.fail('pilot cannot score development'))
    prepared = []
    for rid in [P.TARGET, 'other']:
        model = Toy()
        c = {'run_id': rid, 'seed': 0, 'checkpoint': 'source', 'state_dict_sha256': P.R.digests(model)}
        prepared.append((c, model, [], []))
    monkeypatch.setattr(P, 'fit', lambda *a: (0, gate(.2, passed), 'train_pass' if passed else 'numeric_gate_not_met'))
    monkeypatch.setattr(P.R, 'load_network', lambda *a: prepared[0][1])
    inputs = NS(value={'quantities': {'Z_REF_J': {'value': 550.}}})
    P.run(protocol(), 'pin', {}, REG, inputs, prepared)
    result = json.loads((tmp_path / protocol()['output_name'] / 'PILOT_FINISHED.json').read_text())
    assert result['train_gate']['pass'] == passed and not result['scientific_nav065_pass']
    assert result['development_scoring_rounds'] == 0 and result['untouched_candidates'] == ['other']
    assert not result['automatic_expansion_allowed']
    with pytest.raises(FileExistsError):
        P.run(protocol(), 'pin', {}, REG, inputs, prepared)


def test_pilot_scope_cannot_expand_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(P.R, 'WORK', tmp_path)
    p = protocol()
    p['resource_limit']['update_candidates'].append('other')
    with pytest.raises(ValueError, match='exactly'):
        P.run(p, 'pin', {}, REG, None, [])
    assert not list(tmp_path.iterdir())
