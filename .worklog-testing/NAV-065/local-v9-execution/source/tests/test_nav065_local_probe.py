"""Small main-agent checks for diagnosis selection and experimental isolation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_nav065_local as L
from test_nav065_block_refit import Toy, truth

REG = {'criteria': {'rel_mae_max': .35, 'pearson_min': .7, 'variance_ratio_min': .01}}


def start(k, actual, classification, map_id='map'):
    return {'map_id': map_id, 'k': k, 'budget_J': k * 550,
            'actual_feasible': actual, 'classification': classification}


def protocol():
    p = L.R.load(L.PROPOSAL)
    p['resource_limit']['optimizer_trials_total_max'] = 2
    return p


def batch():
    return (np.array([[0.], [1.]], np.float32), np.array([550., 550.]),
            np.array([.9, 1.001]), np.array([-3., .001]))


def test_selection_includes_mistakes_and_actual_nearest_feasible_without_monotonic_assumption():
    rows = [start(1., False, 'false_feasible'), start(.99, True, 'true_feasible'),
            start(1.2, True, 'true_feasible'), start(1.3, False, 'true_infeasible')]
    keys, pairs = L.select_local(rows)
    assert keys == [('map', .99), ('map', 1.)]
    assert pairs[0]['feasible'] == ['map', .99]
    assert L.select_local(list(reversed(rows))) == (keys, pairs)


def test_multiple_mistakes_share_one_counterpart_and_ties_use_lower_budget():
    rows = [start(1., False, 'false_feasible'), start(1.01, False, 'false_feasible'),
            start(.9, True, 'true_feasible'), start(1.1, True, 'true_feasible')]
    keys, pairs = L.select_local(rows)
    assert len(keys) == 4 and len(pairs) == 2
    assert pairs[0]['feasible'] == ['map', .9]


@pytest.mark.parametrize('rows', [[], [start(1., False, 'false_feasible')],
    [start(1., True, 'true_feasible')],
    [start(1., True, 'true_feasible'), start(1., True, 'true_feasible')]])
def test_empty_unsupported_or_duplicate_local_selection_is_rejected(rows):
    with pytest.raises(ValueError):
        L.select_local(rows)


def test_local_batch_uses_original_common_stop_labels_and_preserves_input():
    records = [(('map', 1.), truth(True)), (('map', 1.1), truth(False))]
    saved = copy.deepcopy(records)
    data = L.local_batch(records, [('map', 1.1), ('map', 1.)])
    np.testing.assert_array_equal(data[3], [-3., .001])
    for (_, old), (_, new) in zip(saved, records):
        np.testing.assert_array_equal(old.reach_target, new.reach_target)
        np.testing.assert_array_equal(old.stop_index, new.stop_index)
    with pytest.raises(ValueError):
        L.local_batch(records, [('unknown', 1.)])


def test_small_cost_error_still_reports_false_feasible_and_component_error():
    model = Toy()
    for role, value in (('critic_cost', .999), ('critic_reach', -3.)):
        getattr(model, role).weight.data.zero_()
        getattr(model, role).bias.data.fill_(value)
    data = (np.zeros((1, 1), np.float32), np.array([550.]), np.array([1.001]), np.array([-3.]))
    report = L.evaluate_local(model, data, REG, 550.)
    assert report['metrics']['vc_norm']['relative_mae'] < .01
    assert report['false_feasible'] == report['cost_sign_errors'] == 1
    assert report['reach_sign_errors'] == 0 and not report['local_pass']


def test_exact_targets_pass_local_checks_without_becoming_scientific_acceptance():
    model = Toy()
    for role, values in (('critic_cost', [.9, 1.001]), ('critic_reach', [-3., .001])):
        getattr(model, role).bias.data.fill_(values[0])
        getattr(model, role).weight.data.fill_(values[1] - values[0])
    report = L.evaluate_local(model, batch(), REG, 550.)
    assert report['local_pass'] and report['joint_sign_errors'] == 0
    assert 'scientific_nav065_pass' not in report


def test_tiny_actual_fit_descends_loss_and_keeps_actor_and_source_unchanged(tmp_path):
    torch.manual_seed(9)
    source = Toy()
    model = copy.deepcopy(source)
    before = L.R.digests(source)
    initial = float(L.local_loss(model, batch()).detach())
    with (tmp_path / 'journal').open('x') as f:
        steps, report = L.fit_local(model, batch(), protocol(), REG, 550., f)
    assert steps == 2 and float(L.local_loss(model, batch()).detach()) < initial
    assert L.R.digests(model)['policy'] == before['policy']
    assert L.R.digests(source) == before
    rows = [json.loads(x) for x in (tmp_path / 'journal').read_text().splitlines()]
    assert len([r for r in rows if r['kind'] == 'optimizer_attempt']) == 2


def test_first_local_pass_stops_and_preexisting_pass_has_no_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'evaluate_local', lambda *a: {'local_pass': True})
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **kw: pytest.fail('unexpected optimizer'))
    with (tmp_path / 'journal').open('x') as f:
        assert L.fit_local(Toy(), batch(), protocol(), REG, 550., f)[0] == 0


def test_nonfinite_gradient_stops_before_any_optimizer_attempt(tmp_path):
    model = Toy()
    model.critic_reach.bias.register_hook(lambda g: g * float('nan'))
    before = L.R.digests(model)
    with (tmp_path / 'journal').open('x') as f, pytest.raises(ValueError, match='gradient'):
        L.fit_local(model, batch(), protocol(), REG, 550., f)
    assert L.R.digests(model) == before
    assert 'optimizer_attempt' not in (tmp_path / 'journal').read_text()


def test_proposal_is_blocked_before_source_loading(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['probe', '--registration', str(L.PROPOSAL)])
    monkeypatch.setattr(L.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setattr(L, 'sources', lambda: pytest.fail('source loaded before approval'))
    with pytest.raises(ValueError, match='explicit approval'):
        L.main()


@pytest.mark.parametrize('local_pass', [False, True])
def test_probe_cannot_score_development_publish_or_change_originals(tmp_path, monkeypatch, local_pass):
    source = Toy()
    before = L.R.digests(source)
    candidate = {'run_id': L.P.TARGET}
    prepared = [(candidate, source, [(('map', 1.), truth())], [])]
    monkeypatch.setattr(L.R, 'WORK', tmp_path)
    monkeypatch.setattr(L.R, 'evaluate', lambda *a: pytest.fail('development not allowed'))
    monkeypatch.setattr(L, 'diagnose', lambda *a: ({'full_train_gate': {'pass': False}}, [('map', 1.)]))
    monkeypatch.setattr(L, 'fit_local', lambda *a: (0, {'local_pass': local_pass}))
    monkeypatch.setattr(L.P.A, 'train_gate', lambda *a: {'pass': True})
    monkeypatch.setattr(L.R, 'load_network', lambda *a: source)
    inputs = NS(value={'quantities': {'Z_REF_J': {'value': 550.}}})
    L.run(protocol(), 'pin', None, REG, inputs, prepared)
    out = tmp_path / protocol()['output_name']
    report = json.loads((out / 'PROBE_FINISHED.json').read_text())
    assert report['local']['local_pass'] == local_pass and report['full_train_after']['pass']
    assert not report['scientific_nav065_pass'] and not report['deployment_allowed']
    assert report['development_scoring_rounds'] == 0 and not report['automatic_expansion_allowed']
    assert L.R.digests(source) == before
    with pytest.raises(FileExistsError):
        L.run(protocol(), 'pin', None, REG, inputs, prepared)


def test_scope_expansion_rejected_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(L.R, 'WORK', tmp_path)
    settings = protocol()
    settings['resource_limit']['update_candidates'].append('other')
    with pytest.raises(ValueError, match='single-candidate'):
        L.run(settings, 'pin', None, REG, None, [])
    assert not list(tmp_path.iterdir())
