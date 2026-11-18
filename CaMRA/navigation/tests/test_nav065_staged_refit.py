"""Main-agent basic tests for v7's process freedom and unchanged final gate."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refit_nav065_staged as S
from test_nav065_block_refit import Toy, truth

REG = {'criteria': {'rel_mae_max': .35, 'pearson_min': .7}, 'primary_k': 1.15}


def gate(mae=.8, passed=False):
    metrics = {name: {'relative_mae': mae, 'pearson': .8, 'pearson_required': True}
               for name in ('vc_norm', 'vh')}
    return {'pass': passed, 'failures': [] if passed else ['synthetic original sign failure'],
        'original_grid': {'pass': passed, 'metrics': copy.deepcopy(metrics)},
        'additional_metrics': metrics}


def protocol():
    p = S.R.load(S.PROPOSAL)
    p['fit']['numeric_steps_max'] = p['fit']['boundary_steps_max'] = 2
    p['resource_limit']['optimizer_trials_per_candidate_max'] = 4
    return p


def records():
    return [(('map', 1.15), truth(False)), (('map', 1.0), truth(True))]


def patch_gates(monkeypatch, reports):
    sequence = iter(reports)
    monkeypatch.setattr(S.A, 'train_gate', lambda *a: copy.deepcopy(next(sequence)))


def test_numeric_gate_is_only_a_transition_not_scientific_pass():
    g = gate(.3)
    assert S.numeric_pass(g, REG) and not g['pass'] and not g['original_grid']['pass']
    g['original_grid']['metrics']['vh']['relative_mae'] = .36
    assert not S.numeric_pass(g, REG)


@pytest.mark.parametrize('field,value', [('relative_mae', None), ('relative_mae', float('nan')),
                                       ('relative_mae', float('inf')), ('pearson', None), ('pearson', float('nan'))])
def test_stage_transition_rejects_undefined_and_nonfinite_metrics(field, value):
    g = gate(.3)
    g['additional_metrics']['vh'][field] = value
    assert not S.numeric_pass(g, REG)


def test_two_stages_accept_temporary_regression_and_freeze_first_full_pass(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(), gate(.95), gate(.3), gate(.25, True)])
    model, snapshots, optimizers = Toy(), [], []
    actor = S.R.digests(model)['policy']
    adam = torch.optim.Adam
    def optimizer(*a, **kw):
        obj = adam(*a, **kw)
        optimizers.append(obj)
        return obj
    monkeypatch.setattr(torch.optim, 'Adam', optimizer)
    with (tmp_path / 'log').open('x') as stream:
        trials, result = S.fit(model, records(), [], protocol(), REG, 550., 0, stream,
            lambda phase, trial, net: snapshots.append((phase, trial)))
    rows = [json.loads(x) for x in (tmp_path / 'log').read_text().splitlines()]
    steps = [r for r in rows if r['kind'] == 'trial']
    assert trials == 3 and result['pass']
    assert [r['stage'] for r in steps] == ['numeric', 'numeric', 'boundary']
    assert steps[0]['report']['additional_metrics']['vh']['relative_mae'] == .95
    assert steps[0]['gradient_diagnostics'] is None and steps[2]['gradient_diagnostics'] is not None
    assert snapshots == [('numeric', 2)] and len(optimizers) == 2
    assert optimizers[0] is not optimizers[1]
    assert S.R.digests(model)['policy'] == actor
    assert all(len(r['starts']) == 2 for r in steps)


def test_early_numeric_transition_does_not_transfer_unused_steps(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(), gate(.3), gate(.6), gate(.7)])
    with (tmp_path / 'log').open('x') as stream:
        trials, result = S.fit(Toy(), records(), [], protocol(), REG, 550., 1, stream)
    assert trials == 3 and not result['pass']
    stages = [json.loads(x)['stage'] for x in (tmp_path / 'log').read_text().splitlines() if json.loads(x)['kind'] == 'trial']
    assert stages == ['numeric', 'boundary', 'boundary']


def test_already_numeric_pass_skips_first_stage_but_still_checks_full_gate(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.2), gate(.3), gate(.4)])
    with (tmp_path / 'log').open('x') as stream:
        trials, result = S.fit(Toy(), records(), [], protocol(), REG, 550., 0, stream)
    assert trials == 2 and not result['pass']
    assert 'stage_skipped' in (tmp_path / 'log').read_text()


def test_already_full_pass_has_no_optimizer(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate(.2, True)])
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **kw: pytest.fail('unexpected optimizer'))
    with (tmp_path / 'log').open('x') as stream:
        assert S.fit(Toy(), records(), [], protocol(), REG, 550., 0, stream)[0] == 0


def test_start_changes_distinguish_false_feasible_false_infeasible_and_primary():
    model = Toy()
    for role, value in (('critic_cost', .9), ('critic_reach', -.1)):
        getattr(model, role).weight.data.zero_()
        getattr(model, role).bias.data.fill_(value)
    before = S.start_snapshot(model, records()[:1], records()[1:], 550., 1.15)
    model.critic_reach.bias.data.fill_(.1)
    after = S.start_snapshot(model, records()[:1], records()[1:], 550., 1.15)
    delta = S.changes(before, after)
    assert [(r['previous_classification'], r['classification'], r['primary']) for r in delta] == [
        ('true_feasible', 'false_infeasible', True), ('false_feasible', 'true_infeasible', False)]
    assert [r['origin'] for r in delta] == ['old', 'new']
    assert all('previous_W' in r and 'true_W' in r for r in delta)
    after[0]['map_id'] = 'other'
    with pytest.raises(ValueError, match='identities'):
        S.changes(before, after)


def test_nonfinite_gradient_stops_before_any_optimizer_attempt(tmp_path, monkeypatch):
    patch_gates(monkeypatch, [gate()])
    model = Toy()
    model.critic_reach.bias.register_hook(lambda g: g * float('nan'))
    before = S.R.digests(model)
    with (tmp_path / 'log').open('x') as stream, pytest.raises(ValueError, match='gradient'):
        S.fit(model, records(), [], protocol(), REG, 550., 0, stream)
    assert S.R.digests(model) == before
    assert 'optimizer_attempt' not in (tmp_path / 'log').read_text()


def test_unapproved_entry_stops_before_sources_or_output(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['refit', '--registration', str(S.PROPOSAL)])
    monkeypatch.setattr(S, 'sources', lambda: pytest.fail('unexpected source access'))
    monkeypatch.setattr(S.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    with pytest.raises(ValueError, match='explicit approval'):
        S.main()


@pytest.mark.parametrize('passed', [False, True])
def test_all_six_models_are_saved_reloaded_before_one_original_evaluator_call(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(S.R, 'WORK', tmp_path)
    prepared, models = [], {}
    for seed in range(6):
        c = {'run_id': str(seed), 'checkpoint': 'source', 'variant': 'full', 'seed': seed}
        model = Toy()
        models[str(seed)] = model
        prepared.append((c, model, [], []))
    monkeypatch.setattr(S, 'fit', lambda *a: (0, gate(.2, passed)))
    reloaded, called = [], []
    def reload(inputs, c, path):
        assert path.is_file()
        reloaded.append(c['run_id'])
        return models[c['run_id']]
    def evaluate(out, *a):
        assert len(reloaded) == 6 and len(list(out.glob('*.pt'))) == 6
        assert (out / 'fitted_candidates.json').is_file()
        called.append(True)
    monkeypatch.setattr(S.R, 'load_network', reload)
    monkeypatch.setattr(S.R, 'evaluate', evaluate)
    inputs = NS(value={'quantities': {'Z_REF_J': {'value': 550.}}}, train_pin='pin', prereg={'implementation_versions': {}})
    S.run(protocol(), 'pin', {}, REG, inputs, prepared)
    assert len(called) == int(passed) and len(reloaded) == 6
    assert (tmp_path / protocol()['output_name'] / 'HELD_TRAIN_GATE.json').exists() != passed
    with pytest.raises(FileExistsError):
        S.run(protocol(), 'pin', {}, REG, inputs, prepared)


def test_run_stopped_never_reaches_development(tmp_path, monkeypatch):
    monkeypatch.setattr(S.R, 'WORK', tmp_path)
    def fail(*a):
        raise ValueError('nonfinite staged parameter')
    monkeypatch.setattr(S, 'fit', fail)
    monkeypatch.setattr(S.R, 'evaluate', lambda *a: pytest.fail('unexpected development scoring'))
    c = {'run_id': 'one', 'seed': 0}
    with pytest.raises(ValueError, match='nonfinite'):
        S.run(protocol(), 'pin', {}, REG, NS(value={'quantities': {'Z_REF_J': {'value': 550.}}}), [(c, Toy(), [], [])])
    assert json.loads((tmp_path / protocol()['output_name'] / 'STOPPED.json').read_text())['development_scoring_rounds'] == 0
