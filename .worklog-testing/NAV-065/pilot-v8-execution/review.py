"""Main-agent read-only acceptance of the approved v8 pilot; no training calls."""
from collections import Counter
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_v8')
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_pilot as P
A, R, S = P.A, P.R, P.S
spec = importlib.util.spec_from_file_location('v7_readonly_audit', ROOT / '.worklog-testing/NAV-065/staged-v7-execution/review.py')
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)
read, write = J.read, J.write

def main():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
    torch.set_num_threads(1)
    protocol, pin = P.approved_registration(NAV / 'config/nav065_pilot_refit_prereg_v8.json')
    assert read(LAUNCH / 'EXIT.json')['source_unchanged']
    assert not (LAUNCH / 'ERROR.json').exists() and not (OUT / 'STOPPED.json').exists()
    prep = read(NAV / 'reports/nav065_pilot_refit_v8_preparation_review.json')
    assert prep['preparation_pass'] and prep['basic_tests_passed'] == 57
    for name, digest in prep['evidence'].items():
        R.verify_hash(Path(name), digest)
    basic_log = ROOT / '.worklog-testing/NAV-065/pilot-v8-preparation/basic-attempt1.log'
    R.verify_hash(basic_log, prep['basic_log_sha256'])
    assert '57 passed' in basic_log.read_text()
    for name, digest in read(LAUNCH / 'source_before.json').items():
        R.verify_hash(NAV / name, digest)
        R.verify_hash(LAUNCH / 'source_snapshot' / name, digest)
    v2 = read(LAUNCH / 'resources.json')['v2_verified_files']
    assert len(v2) == 33
    for name, digest in v2.items():
        R.verify_hash(Path(name), digest)
    _, reg, inputs, prepared = P.sources()
    scale = inputs.value['quantities']['Z_REF_J']['value']
    parent, source, old, new = next(row for row in prepared if row[0]['run_id'] == P.TARGET)
    records = old + new
    assert len(old) == 90 and len(new) == 72
    result = read(OUT / 'PILOT_FINISHED.json')
    assert result['pilot_pin'] == pin and result['development_scoring_rounds'] == 0
    assert not result['scientific_nav065_pass'] and not result['automatic_expansion_allowed']
    candidate = result['candidate']
    assert candidate['run_id'] == P.TARGET and candidate['parent_pilot_checkpoint'] == parent['checkpoint']
    R.verify_hash(Path(candidate['checkpoint']), candidate['checkpoint_sha256'])
    model = R.load_network(inputs, candidate, candidate['checkpoint'])
    assert R.digests(model) == candidate['state_dict_sha256']
    assert R.digests(model)['policy'] == parent['state_dict_sha256']['policy']
    assert set(result['untouched_candidates']) == {c['run_id'] for c, _, _, _ in prepared if c['run_id'] != P.TARGET}
    for c, other, _, _ in prepared:
        assert R.digests(other) == c['state_dict_sha256']
    rows = list(J.rows(OUT / (P.TARGET + '.jsonl')))
    first = rows[0]
    initial = A.train_gate(source, old, new, reg, protocol['fit'], scale)
    assert first == {'kind': 'train_gate', 'trial': 0, 'report': initial,
                     'starts': S.start_snapshot(source, old, new, scale, reg['primary_k'])}
    J.verify_starts(first['starts'], records, initial, scale, reg['primary_k'])
    counts, transitions, reports = Counter(), [], {0: initial}
    previous, current, stage, trials, attempted = first['starts'], initial, None, 0, None
    changes, numeric_regression_stops = Counter(), []
    finished = False
    for row in rows[1:]:
        assert J.finite(row)
        kind = row['kind']
        if kind == 'stage_started':
            stage = row['stage']
            assert stage in ('numeric', 'boundary') and row['trial'] == trials and not current['pass']
            assert row['optimizer'] == ('Adam' if stage == 'numeric' else 'SGD_no_momentum')
            assert not transitions or (transitions[-1]['stage'], stage) == ('numeric', 'boundary')
            if stage == 'numeric':
                assert trials == 0 and not S.numeric_pass(current, reg)
            else:
                assert S.numeric_pass(current, reg), 'numeric cap never authorizes boundary on failure'
            transitions.append({'stage': stage, 'trial': trials, 'numeric_pass': S.numeric_pass(current, reg)})
        elif kind == 'optimizer_attempt':
            assert attempted is None and not finished and not current['pass']
            assert not numeric_regression_stops
            assert row['stage'] == stage and row['trial'] == trials + 1 and row['stage_step'] == counts[stage] + 1
            assert row['stage_step'] <= protocol['fit'][stage + '_steps_max']
            if stage == 'numeric':
                assert not S.numeric_pass(current, reg)
            attempted = row
        elif kind == 'trial':
            assert attempted and all(row[k] == attempted[k] for k in ('trial', 'stage', 'stage_step'))
            trials += 1
            counts[stage] += 1
            assert row['regression_loss_before'] >= 0
            assert (row['gradient_diagnostics'] is None) == (stage == 'numeric')
            for stats in (row['gradient_diagnostics'] or {}).values():
                rnorm = stats['regression_gradient_norm']
                assert 0 <= stats['sign_multiplier'] <= 1
                assert stats['correction_norm'] <= protocol['fit']['correction_fraction_max'] * rnorm * 1.0001 + 1e-7
                assert stats['regression_dot_direction'] >= rnorm ** 2 * .9999 - 1e-7
            assert row['start_changes'] == S.changes(previous, row['starts'])
            for change in row['start_changes']:
                changes[change['previous_classification'] + '->' + change['classification']] += 1
            current, previous = row['report'], row['starts']
            J.verify_starts(previous, records, current, scale, reg['primary_k'])
            reports[trials] = current
            if stage == 'boundary' and counts[stage] % protocol['fit']['numeric_review_interval'] == 0 and not S.numeric_pass(current, reg):
                numeric_regression_stops.append(trials)
            attempted = None
        elif kind == 'fit_finished':
            assert row == {'kind': kind, 'trials': trials, 'reason': 'numeric_gate_not_met', 'pass': False}
            assert not current['pass'] and not S.numeric_pass(current, reg)
            assert counts['numeric'] == protocol['fit']['numeric_steps_max'] and counts['boundary'] == 0
        elif kind == 'pilot_finished':
            assert not finished and attempted is None
            assert row['trials'] == trials and row['pass'] == current['pass'] and row['reason'] == result['stop_reason']
            finished = True
        else:
            raise AssertionError('unexpected journal row ' + kind)
    assert finished and trials == result['optimizer_trials'] <= 400
    assert current == result['train_gate'] == A.train_gate(model, old, new, reg, protocol['fit'], scale)
    assert previous == S.start_snapshot(model, old, new, scale, reg['primary_k'])
    if result['stop_reason'] == 'numeric_gate_not_met':
        assert counts['numeric'] == 300 and counts['boundary'] == 0 and not S.numeric_pass(current, reg)
    elif result['stop_reason'] == 'boundary_numeric_regression':
        assert numeric_regression_stops == [trials]
    elif result['stop_reason'] == 'train_pass':
        assert current['pass'] and not any(g['pass'] for i, g in reports.items() if i < trials)
    else:
        assert result['stop_reason'] == 'trial_cap_reached' and counts['boundary'] == 100 and not current['pass']
    checkpoints = []
    for path in sorted(OUT.glob(P.TARGET + '.*-*.pt')):
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['pilot_pin'] == pin and state['trial'] in reports
        stage_model = R.load_network(inputs, candidate, path)
        assert R.digests(stage_model)['policy'] == parent['state_dict_sha256']['policy']
        stage_gate = A.train_gate(stage_model, old, new, reg, protocol['fit'], scale)
        assert stage_gate == reports[state['trial']]
        checkpoints.append({'stage': state['stage'], 'trial': state['trial'], 'sha256': R.file_sha(path), 'train_gate': stage_gate})
    maps, _ = R.train_grid(inputs)
    specs = {m['map_id']: R.E.M.build(m['level'], m['seed'], R.E.M.load_design(R.E.DESIGN)) for m in maps}
    raw_old = A.B.S.saved_train(inputs, maps, reg['grid_k'], R.WORK / 'nav065_refit_v1', read(R.WORK / 'nav065_refit_v1/fitted_candidates.json'))[P.TARGET]
    raw_new = {(r['map_id'], r['k']): r['episode'] for r in J.rows(R.WORK / 'nav065_adaptive_refit_v5/train_episodes.jsonl')
               if r['kind'] == 'rollout_result' and r['run_id'] == P.TARGET}
    truth_steps, cases = 0, []
    for origin, collection, raw in [('old', old, raw_old), ('new', new, raw_new)]:
        for (mid, k), truth in collection:
            episode = raw[mid, k]
            J.truth_check(episode, truth, specs[mid], inputs)
            truth_steps += len(truth.observations)
            start = next(s for s in previous if s['origin'] == origin and s['map_id'] == mid and s['k'] == k)
            cases.append({**start, 'label': episode['label'], 'cost_model_J': episode['cost_model_J']})
    assert len(cases) == 162
    for name in ('development_scores.jsonl', 'gate_G4_alignment_v2.json', 'deployment_candidates_v2.json'):
        assert not (OUT / name).exists()
    report = {'task': 'NAV-065', 'timestamp': datetime.now().astimezone().isoformat(),
        'reviewer': 'main agent; create-plan-worklog-legacy; no test subagents', 'independent': False,
        'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
        'execution_integrity_pass': True, 'scientific_nav065_pass': False, 'registration_pin': pin,
        'run_status': 'PILOT_TRAIN_PASS' if current['pass'] else 'HELD_TRAIN_GATE', 'stop_reason': result['stop_reason'],
        'candidate': P.TARGET, 'optimizer_trials': trials, 'phase_counts': dict(counts),
        'basic_checks_passed': 57, 'saved_train_episodes': 162, 'saved_train_steps': truth_steps,
        'new_rollouts': 0, 'development_scoring_rounds': 0, 'automatic_expansion_allowed': False,
        'actor_unchanged_candidates': 6, 'untouched_other_candidates': 5, 'v2_verified_files': 33,
        'initial_gate': initial, 'final_gate': current,
        'initial_start_confusion': dict(Counter(s['classification'] for s in first['starts'])),
        'final_start_confusion': dict(Counter(s['classification'] for s in previous)),
        'stage_transitions': transitions, 'stage_checkpoints': checkpoints, 'classification_changes': dict(changes),
        'original_G4': 'FAIL', 'completed_tasks': 52, 'NAV-065': False, 'NAV-066_started': False,
        'limitations': ['Only one fixed train candidate; no new deployment/G4 or generalization result.',
            'Every logged start arithmetic/label/change and stage/cap verified; intermediate tensors exist only at saved stage boundaries. No optimizer replay.',
            'Prepared 57 checks have matching unchanged source; no repeated broad test suite. First-order gradient control does not guarantee finite-step metrics.'],
        'evidence': {'launch': str(LAUNCH), 'output': str(OUT), 'legacy_raw_truth_auditor_sha256': R.file_sha(Path(J.__file__)),
            'output_hashes': {p.name: R.file_sha(p) for p in OUT.iterdir() if p.is_file()}}}
    for path, payload in [(LAUNCH / 'review.json', report), (LAUNCH / 'train_cases.json', cases),
                          (NAV / 'reports/nav065_pilot_refit_v8_review.json', report)]:
        write(path, payload)
    print(json.dumps({k: v for k, v in report.items() if k not in ('initial_gate', 'final_gate', 'stage_checkpoints', 'evidence')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
