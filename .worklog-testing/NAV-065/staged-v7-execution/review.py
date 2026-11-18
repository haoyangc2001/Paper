"""Read-only main-agent v7 acceptance audit; no optimizer or rollout calls."""
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_v7')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_staged as S
A, R = S.A, S.R
read = lambda p: json.loads(p.read_bytes())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def finite(item):
    if isinstance(item, dict):
        return all(finite(v) for v in item.values())
    if isinstance(item, list):
        return all(finite(v) for v in item)
    return not isinstance(item, float) or math.isfinite(item)


def truth_check(ep, truth, spec, inputs):
    z = np.asarray([ep['z0_J'], *(s['z_J'] for s in ep['trace'])])
    d = np.asarray([np.linalg.norm(np.asarray(spec.start) - spec.goal),
                    *(s['goal_distance_m'] for s in ep['trace'])])
    y = np.asarray([0., *(s['safety_y'] for s in ep['trace'])])
    q, n = inputs.value['quantities'], len(ep['trace'])
    radius = inputs.prereg['environment']['r_goal_m']
    h = np.where(d <= radius, q['g_norm']['G_INSIDE'], (d - radius) / q['g_norm']['L_REF_m'])
    h[y > 0] = np.maximum(h[y > 0], y[y > 0] * q['C']['value'])
    hard = ep['label'] in ('collision', 'out_of_bounds')
    if hard:
        h[-1] = inputs.value['bootstrap']['FAIL_PENALTY']['value']
    score = np.maximum(h, -z / q['Z_REF_J']['value'])
    stop = np.full(n, n, dtype=int) if hard else np.asarray([i + np.argmin(score[i:]) for i in range(n)])
    cost = ((z[:-1] - z[-2]) / q['Z_REF_J']['value'] + inputs.value['bootstrap']['FAIL_PENALTY']['value']
            if hard else (z[:-1] - z[stop]) / q['Z_REF_J']['value'])
    assert np.array_equal(stop, truth.stop_index)
    for actual, expected in ((h[stop], truth.reach_target), (cost, truth.cost_target), (score[stop], truth.path_residual)):
        assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    feasible = ep['label'] == 'goal_reached' and y[-1] == 0 and z[-1] >= 0
    assert feasible == ep['feasible'] == truth.feasible
    assert np.isclose(z[0] - z[-1], ep['cost_model_J'], rtol=0, atol=1e-8)


def verify_starts(starts, records, gate, scale, primary_k):
    assert len(starts) == len(records) == 162
    counts = Counter()
    for index, (row, ((mid, k), truth)) in enumerate(zip(starts, records, strict=True)):
        assert (row['map_id'], row['k'], row['origin'], row['primary']) == (mid, k, 'old' if index < 90 else 'new', k == primary_k)
        assert row['actual_feasible'] == truth.feasible
        assert row['budget_J'] == truth.budget_J[0]
        assert row['target_cost_norm'] == truth.cost_target[0]
        assert row['target_reach'] == truth.reach_target[0] and row['true_W'] == truth.path_residual[0]
        w = float(max(np.float32(row['vh']), np.float32(row['vc_norm']) - np.float32(row['budget_J'] / scale)))
        assert w == row['W'] and (w <= 0) == row['predicted_feasible']
        label = ('true_feasible' if truth.feasible else 'false_feasible') if w <= 0 else ('false_infeasible' if truth.feasible else 'true_infeasible')
        assert label == row['classification']
        counts[row['origin'], label] += 1
    assert gate['additional_false_feasible'] == counts['new', 'false_feasible']
    for k, budget in gate['original_grid']['by_budget'].items():
        selected = [s for s in starts[:90] if s['k'] == float(k)]
        for name in ('true_feasible', 'false_feasible', 'false_infeasible', 'true_infeasible'):
            assert budget[name] == sum(s['classification'] == name for s in selected)
    return counts


def main():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    torch.set_num_threads(1)
    p, pin = S.approved_registration(NAV / 'config/nav065_staged_refit_prereg_v7.json')
    assert read(LAUNCH / 'EXIT.json')['returncode'] == 0
    assert not (LAUNCH / 'ERROR.json').exists() and not (OUT / 'STOPPED.json').exists()
    preparation = read(NAV / 'reports/nav065_staged_refit_v7_preparation_review.json')
    assert preparation['preparation_pass'] and preparation['basic_tests_passed'] == 79
    basic_log = ROOT / '.worklog-testing/NAV-065/staged-v7-preparation/basic-attempt1.log'
    R.verify_hash(basic_log, preparation['basic_log_sha256'])
    assert '79 passed' in basic_log.read_text()
    for name, digest in read(LAUNCH / 'source_before.json').items():
        R.verify_hash(NAV / name, digest)
        R.verify_hash(LAUNCH / 'source_snapshot' / name, digest)
    v2 = read(LAUNCH / 'resources.json')['v2_verified_files']
    assert len(v2) == 33
    for name, digest in v2.items():
        R.verify_hash(Path(name), digest)
    upstream, reg, inputs, prepared = S.sources()
    scale = inputs.value['quantities']['Z_REF_J']['value']
    manifest, gates = read(OUT / 'fitted_candidates.json'), read(OUT / 'train_gate.json')
    assert manifest['refit_pin'] == pin and (manifest['train_episodes'], manifest['train_steps']) == (972, 372471)
    sources = {c['run_id']: (c, model, old, new) for c, model, old, new in prepared}
    by_gate = {g['run_id']: g for g in gates['candidates']}
    assert len(sources) == len(by_gate) == len(manifest['candidates']) == 6
    maps, _ = R.train_grid(inputs)
    specs = {m['map_id']: R.E.M.build(m['level'], m['seed'], R.E.M.load_design(R.E.DESIGN)) for m in maps}
    old_manifest = read(R.WORK / 'nav065_refit_v1/fitted_candidates.json')
    raw_old = A.B.S.saved_train(inputs, maps, reg['grid_k'], R.WORK / 'nav065_refit_v1', old_manifest)
    raw_new = {rid: {} for rid in sources}
    for row in rows(R.WORK / 'nav065_adaptive_refit_v5/train_episodes.jsonl'):
        if row['kind'] == 'rollout_result':
            raw_new[row['run_id']][row['map_id'], row['k']] = row['episode']
    assert sum(len(v) for v in raw_new.values()) == 432
    audited, cases = [], []
    total, truth_steps = 0, 0
    for c in manifest['candidates']:
        rid = c['run_id']
        parent, source, old, new = sources[rid]
        records = old + new
        initial = A.train_gate(source, old, new, reg, p['fit'], scale)
        R.verify_hash(Path(c['checkpoint']), c['checkpoint_sha256'])
        model = R.load_network(inputs, c, c['checkpoint'])
        assert R.digests(model) == c['state_dict_sha256']
        assert R.digests(model)['policy'] == parent['state_dict_sha256']['policy']
        assert c['parent_staged_checkpoint'] == parent['checkpoint']
        journal = iter(rows(OUT / (rid + '.jsonl')))
        first = next(journal)
        assert first == {'kind': 'train_gate', 'trial': 0, 'report': initial,
                         'starts': S.start_snapshot(source, old, new, scale, reg['primary_k'])}
        verify_starts(first['starts'], records, initial, scale, reg['primary_k'])
        previous, current, trials, attempted = first['starts'], initial, 0, None
        stage, phase_counts, finished, checkpoint_rows = None, Counter(), False, []
        changes, first_pass, transitions = Counter(), None, []
        for row in journal:
            assert finite(row)
            kind = row['kind']
            if kind == 'stage_skipped':
                assert row['stage'] == 'numeric' and trials == 0 and S.numeric_pass(current, reg)
            elif kind == 'stage_started':
                stage = row['stage']
                assert stage in ('numeric', 'boundary') and row['trial'] == trials and row['fresh_adam']
                assert row['learning_rate'] == p['fit'][stage + '_learning_rate']
                assert not current['pass']
                if stage == 'numeric':
                    assert trials == 0 and not S.numeric_pass(current, reg)
                else:
                    assert S.numeric_pass(current, reg) or phase_counts['numeric'] == p['fit']['numeric_steps_max']
                transitions.append({'stage': stage, 'trial': trials, 'numeric_pass': S.numeric_pass(current, reg)})
            elif kind == 'optimizer_attempt':
                assert attempted is None and not current['pass'] and row['stage'] == stage
                assert row['trial'] == trials + 1 and row['stage_step'] == phase_counts[stage] + 1
                assert row['stage_step'] <= p['fit'][stage + '_steps_max']
                attempted = row
            elif kind == 'trial':
                assert attempted and all(row[k] == attempted[k] for k in ('trial', 'stage', 'stage_step'))
                assert not finished and first_pass is None
                trials += 1
                phase_counts[stage] += 1
                assert row['regression_loss_before'] >= 0
                assert (row['gradient_diagnostics'] is None) == (stage == 'numeric')
                assert row['start_changes'] == S.changes(previous, row['starts'])
                for change in row['start_changes']:
                    changes[change['previous_classification'] + '->' + change['classification']] += 1
                current, previous = row['report'], row['starts']
                verify_starts(previous, records, current, scale, reg['primary_k'])
                if current['pass']:
                    first_pass = trials
                attempted = None
            elif kind == 'stage_finished':
                assert row['stage'] == stage and row['trial'] == trials and not current['pass']
                assert row['numeric_pass'] == S.numeric_pass(current, reg) and row['full_pass'] is False
                if stage == 'numeric':
                    assert S.numeric_pass(current, reg) or phase_counts[stage] == p['fit'][stage + '_steps_max']
                else:
                    assert phase_counts[stage] == p['fit'][stage + '_steps_max']
                path = OUT / (rid + f'.{stage}-{trials}.pt')
                state = torch.load(path, map_location='cpu', weights_only=False)
                assert (state['staged_refit_pin'], state['trial'], state['stage']) == (pin, trials, stage)
                stage_model = R.load_network(inputs, c, path)
                assert R.digests(stage_model)['policy'] == parent['state_dict_sha256']['policy']
                assert A.train_gate(stage_model, old, new, reg, p['fit'], scale) == current
                assert S.start_snapshot(stage_model, old, new, scale, reg['primary_k']) == previous
                checkpoint_rows.append({'stage': stage, 'trial': trials, 'path': str(path),
                                        'sha256': R.file_sha(path), 'train_gate': current})
            elif kind == 'fit_finished':
                assert row == {'kind': kind, 'trials': trials, 'pass': current['pass']}
                assert not finished and attempted is None
                finished = True
                if current['pass']:
                    assert trials == first_pass
                else:
                    assert phase_counts['boundary'] == p['fit']['boundary_steps_max']
            else:
                raise AssertionError('unexpected journal row: ' + kind)
        assert finished and trials <= p['resource_limit']['optimizer_trials_per_candidate_max']
        declared = by_gate[rid]
        assert declared == {'run_id': rid, 'trials': trials, **current}
        assert A.train_gate(model, old, new, reg, p['fit'], scale) == current
        assert S.start_snapshot(model, old, new, scale, reg['primary_k']) == previous
        for origin, collection, raw in [('old', old, raw_old[rid]), ('new', new, raw_new[rid])]:
            for (mid, k), truth in collection:
                ep = raw[mid, k]
                truth_check(ep, truth, specs[mid], inputs)
                truth_steps += len(truth.observations)
                start = next(s for s in previous if s['origin'] == origin and s['map_id'] == mid and s['k'] == k)
                cases.append({'run_id': rid, **start, 'label': ep['label'], 'cost_model_J': ep['cost_model_J']})
        counts = Counter(s['classification'] for s in previous)
        original_false = sum(v['false_feasible'] for v in current['original_grid']['by_budget'].values())
        audited.append({**declared, 'initial_gate': initial, 'phase_counts': dict(phase_counts),
                        'stage_transitions': transitions, 'stage_checkpoints': checkpoint_rows,
                        'first_full_pass_trial': first_pass, 'classification_changes': dict(changes),
                        'all_start_confusion': dict(counts), 'original_false_feasible': original_false,
                        'initial_false_feasible': sum(s['classification'] == 'false_feasible' for s in first['starts'])})
        total += trials
        print(json.dumps({'reviewed': rid, 'trials': trials, 'pass': current['pass']}, ensure_ascii=False), flush=True)
    assert len(cases) == 972 and truth_steps == 372471
    assert total == manifest['optimizer_trials'] <= p['resource_limit']['optimizer_trials_total_max']
    assert gates['pass'] == all(g['pass'] for g in gates['candidates'])
    report = {'task': 'NAV-065', 'reviewer': 'main agent; create-plan-worklog-legacy; no test subagents',
              'independent': False, 'timestamp': datetime.now().astimezone().isoformat(),
              'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
              'execution_integrity_pass': True, 'scientific_pass': False, 'registration_pin': pin,
              'basic_tests_passed': 79, 'saved_train_episodes': 972, 'saved_train_steps': truth_steps,
              'new_train_episodes': 0, 'new_development_episodes': 0, 'optimizer_trials': total,
              'actor_unchanged_candidates': 6, 'v2_verified_files': len(v2),
              'train_pass_candidates': sum(g['pass'] for g in gates['candidates']),
              'initial_train_false_feasible': sum(c['initial_false_feasible'] for c in audited),
              'train_false_feasible': sum(c['classification'] == 'false_feasible' for c in cases),
              'train_false_infeasible': sum(c['classification'] == 'false_infeasible' for c in cases),
              'development_scoring_rounds': 0, 'original_gate_verdict': 'FAIL',
              'run_status': 'HELD_TRAIN_GATE', 'train_candidates': audited,
              'limitations': ['Per-step prediction arithmetic, labels, changes and schedule verified from journals; no optimizer replay. Intermediate model tensors exist only at saved stage boundaries.',
                             'Finite-network empirical gates provide no formal zero-error guarantee. Repeated development remains diagnostic rather than independent final evaluation.'],
              'evidence': {'launch': str(LAUNCH), 'output': str(OUT),
                           'output_hashes': {f.name: R.file_sha(f) for f in OUT.iterdir() if f.is_file()}}}
    if not gates['pass']:
        assert read(OUT / 'HELD_TRAIN_GATE.json') == {'development_scoring_rounds': 0, 'optimizer_trials': total}
        for name in ('development_scores.jsonl', 'gate_G4_alignment_v2.json', 'deployment_candidates_v2.json'):
            assert not (OUT / name).exists()
    else:
        dev = list(rows(OUT / 'development_scores.jsonl'))
        assert len(dev) == len({(r['run_id'], r['map_id'], r['k']) for r in dev}) == 180
        validation = read(OUT / 'deployment_critic_validation_v2.json')
        g4 = read(OUT / 'gate_G4_alignment_v2.json')
        for item in validation['candidates']:
            assert R.summarize([r for r in dev if r['run_id'] == item['run_id']], {m['map_id'] for m in inputs.maps}, reg) == {k: v for k, v in item.items() if k != 'run_id'}
        report.update(development_scoring_rounds=1, development_episodes=180,
                      development_pass_candidates=sum(c['pass'] for c in validation['candidates']),
                      development_false_feasible=sum(not r['actual_feasible'] and r['predicted_feasible'] for r in dev),
                      scientific_pass=g4['verdict'] == 'PASS', original_gate_verdict=g4['verdict'],
                      run_status='PASS' if g4['verdict'] == 'PASS' else 'FAIL_DEVELOPMENT')
        assert (OUT / 'deployment_candidates_v2.json').exists() == report['scientific_pass']
    for path, data in [(LAUNCH / 'review.json', report), (LAUNCH / 'train_cases.json', cases),
                       (NAV / 'reports/nav065_staged_refit_v7_review.json', report)]:
        write(path, data)
    print(json.dumps({k: v for k, v in report.items() if k not in ('train_candidates', 'evidence')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
