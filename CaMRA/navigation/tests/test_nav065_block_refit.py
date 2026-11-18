"""Basic verification of v6's distinct working and final criteria."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refit_nav065_blocks as V


class Toy(torch.nn.Module):
    ROLES = ('policy', 'critic_cost', 'critic_reach')

    def __init__(self):
        super().__init__()
        for name in self.ROLES:
            setattr(self, name, torch.nn.Linear(1, 1))

    def parameters_of(self, role):
        return list(getattr(self, role).parameters())

    def values(self, x):
        return self.critic_cost(x).squeeze(-1), self.critic_reach(x).squeeze(-1)


def protocol():
    return V.R.load(V.PROPOSAL)


def truth(positive=True):
    h = .001 if positive else -3.
    return NS(observations=np.ones((1, 1), np.float32), cost_target=np.array([.999]),
        reach_target=np.array([h]), budget_J=np.array([550.]), stop_index=np.array([1]),
        path_residual=np.array([h]), feasible=not positive)


def gate(passed=False, mismatches=10, mae=.64):
    return {'pass': passed, 'original_grid': {'pass': True, 'failures': []},
        'additional_false_feasible': 0, 'selected_sign_mismatches': 0 if passed else mismatches,
        'additional_metrics': {name: {'relative_mae': value, 'pearson': .4, 'pearson_required': True}
                               for name, value in (('vc_norm', .02), ('vh', mae))}}


REG = {'criteria': {'rel_mae_max': .35, 'pearson_min': .7}}


def test_strata_balancing_does_not_turn_repeated_one_side_into_more_weight():
    model = Toy()
    records = [(('map', 1.), truth(True)), (('map', 1.01), truth(False))]
    saved = copy.deepcopy(records)
    loss, counts = V.regression_loss(model, records, [], protocol()['fit'], np.random.default_rng(1))
    repeated = records[:1] * 20 + records[1:]
    other, _ = V.regression_loss(model, repeated, [], protocol()['fit'], np.random.default_rng(1))
    assert loss.item() == pytest.approx(other.item(), rel=1e-6)
    assert len(counts) == 6 and all(n > 0 for n in counts.values())
    for (_, t), (_, before) in zip(records, saved):
        np.testing.assert_array_equal(t.reach_target, before.reach_target)
        np.testing.assert_array_equal(t.stop_index, before.stop_index)


def test_exact_original_targets_minimize_stratified_regression():
    model = Toy()
    for role, value in (('critic_cost', .999), ('critic_reach', .001)):
        getattr(model, role).weight.data.zero_()
        getattr(model, role).bias.data.fill_(value)
    loss, _ = V.regression_loss(model, [(('map', 1.), truth())], [], protocol()['fit'], np.random.default_rng(1))
    assert loss.item() == pytest.approx(0, abs=1e-10)


def test_working_slack_allows_sign_progress_but_does_not_change_final_gate():
    before, trial = gate(), gate(mismatches=9, mae=.65)
    prediction = actual = np.array([True])
    allowed, _ = V.block_allowed(before, trial, before, prediction, actual, np.array([True]), REG, protocol()['fit'])
    assert allowed and not trial['pass']
    trial['additional_metrics']['vh']['relative_mae'] = .70
    assert not V.block_allowed(before, trial, before, prediction, actual, np.array([True]), REG, protocol()['fit'])[0]


@pytest.mark.parametrize('failure', ['old', 'locked', 'false', 'sign', 'correlation', 'undefined'])
def test_commit_protects_old_results_and_finite_source_anchored_limits(failure):
    before, trial = gate(), gate(mismatches=9)
    p = a = np.array([True])
    if failure == 'old': trial['original_grid']['pass'] = False
    if failure == 'locked': p = np.array([False])
    if failure == 'false': trial['additional_false_feasible'] = 1
    if failure == 'sign': trial['selected_sign_mismatches'] = 11
    if failure == 'correlation': trial['additional_metrics']['vh']['pearson'] = .3
    if failure == 'undefined': trial['additional_metrics']['vh']['relative_mae'] = None
    assert not V.block_allowed(before, trial, before, p, a, np.array([True]), REG, protocol()['fit'])[0]


def patch_fit(monkeypatch, reports):
    queue = iter(reports)
    monkeypatch.setattr(V.A, 'train_gate', lambda *a: copy.deepcopy(next(queue)))
    monkeypatch.setattr(V.A, 'classify', lambda *a: np.array([False, True]))


def test_first_full_pass_freezes_despite_temporary_mid_block_regression(tmp_path, monkeypatch):
    model = Toy()
    actor = V.R.digests(model)['policy']
    patch_fit(monkeypatch, [gate(), gate(mismatches=20, mae=.9), gate(True)])
    records = [(('map', 1.), truth(True)), (('map', 1.01), truth(False))]
    with (tmp_path / 'journal').open('x') as stream:
        trials, accepted, result = V.fit(model, records, [], protocol(), REG, 550., 0, stream)
    assert (trials, accepted) == (2, 2) and result['pass']
    assert V.R.digests(model)['policy'] == actor


def test_rejected_blocks_restore_model_optimizer_rng_and_count_actual_steps(tmp_path, monkeypatch):
    model, starts, choices, optimizers = Toy(), [], [], []
    before = V.R.digests(model)
    patch_fit(monkeypatch, [gate()] * 5)
    monkeypatch.setattr(V, 'block_allowed', lambda *a: (False, 'synthetic_reject'))
    original_loss = V.regression_loss
    def loss(net, old, new, settings, rng):
        choices.append(rng.integers(100000))
        starts.append(V.R.digests(net))
        return original_loss(net, old, new, settings, rng)
    monkeypatch.setattr(V, 'regression_loss', loss)
    adam = torch.optim.Adam
    def optimizer(*a, **kw):
        obj = adam(*a, **kw)
        optimizers.append(obj)
        return obj
    monkeypatch.setattr(torch.optim, 'Adam', optimizer)
    p = protocol()
    p['fit']['steps_per_block'] = p['fit']['block_backtracks_max'] = 2
    p['resource_limit']['optimizer_trials_per_candidate_max'] = 4
    records = [(('map', 1.), truth(True)), (('map', 1.01), truth(False))]
    with (tmp_path / 'journal').open('x') as stream:
        trials, accepted, result = V.fit(model, records, [], p, REG, 550., 0, stream)
    assert trials == 4 and accepted == 0 and not result['pass']
    assert V.R.digests(model) == before and starts[0] == starts[2]
    assert choices[:2] == choices[2:]
    assert not optimizers[0].state_dict()['state']


def test_already_passing_model_needs_no_optimizer(tmp_path, monkeypatch):
    patch_fit(monkeypatch, [gate(True)])
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **kw: pytest.fail('unexpected optimizer'))
    model = Toy()
    before = V.R.digests(model)
    with (tmp_path / 'journal').open('x') as stream:
        assert V.fit(model, [], [], protocol(), REG, 550., 0, stream)[:2] == (0, 0)
    assert V.R.digests(model) == before


def test_unapproved_entry_stops_before_preflight_or_output(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['refit', '--registration', str(V.PROPOSAL)])
    monkeypatch.setattr(V, 'sources', lambda: pytest.fail('unexpected preflight'))
    monkeypatch.setattr(V.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    with pytest.raises(ValueError, match='explicit approval'):
        V.main()


@pytest.mark.parametrize('passed', [False, True])
def test_production_run_saves_all_models_then_calls_original_g4_only_if_all_train_pass(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(V.R, 'WORK', tmp_path)
    prepared, models = [], {}
    for seed in range(6):
        model = Toy()
        c = {'run_id': str(seed), 'checkpoint': 'source', 'variant': 'full', 'seed': seed}
        prepared.append((c, model, [], []))
        models[str(seed)] = model
    monkeypatch.setattr(V, 'fit', lambda *a: (0, 0, gate(passed)))
    monkeypatch.setattr(V.R, 'load_network', lambda inputs, c, path: models[c['run_id']])
    called = []
    def evaluate(out, *args):
        assert len(list(out.glob('*.pt'))) == 6
        assert (out / 'fitted_candidates.json').exists()
        called.append(True)
    monkeypatch.setattr(V.R, 'evaluate', evaluate)
    V.run(protocol(), 'pin', {}, {}, NS(value={'quantities': {'Z_REF_J': {'value': 550.}}},
        train_pin='train', prereg={'implementation_versions': {}}), prepared)
    assert len(called) == int(passed)
    assert (tmp_path / protocol()['output_name'] / 'HELD_TRAIN_GATE.json').exists() != passed
