"""Explicit time-version compatibility and actual production consumers (CPU only)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

NAV = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'tests'), str(NAV / 'src'),
               str(NAV.parent / 'calibration/src')]
import train_formal as F
import eval_r14_repair as E
import run_feedback_round as R
import test_feedback_gate as B
from camra_navigation import observation as O
from camra_navigation.algorithm.validation import file_sha, verify_training_bindings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', help='Reuse a PASS component only if every bound source still matches')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    assert not out.exists()
    basepath = Path(args.base).resolve() if args.base else out.with_suffix('.feedback.json')
    if not args.base:
        subprocess.run([sys.executable, str(NAV / 'tests/test_feedback_gate.py'),
                        '--out', str(basepath)], check=True)
    base = json.loads(basepath.read_bytes())
    assert base['verdict'] == 'PASS' and all(c['ok'] for c in base['checks'])
    assert all(file_sha(NAV / rel) == digest for rel, digest in base['code_files'].items())
    assert file_sha(NAV / base['protocol']['path']) == base['protocol']['sha256']
    checks = list(base['checks'])

    def check(name, ok, detail=None):
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)

    ego = O.EgoState(0., 0., 0., 0., 0., 0.)
    extent = O.MapExtent(-15., 15., -15., 15.)
    ti = O.field_slices()['time_remaining_n'].start
    yi = O.field_slices()['safety_y'].start

    def encode(rem, version, y=.27):
        return np.asarray(O.encode_observation(ego, [], extent,
            O.TaskState(10., 0., y, 420., rem), version=version))

    times = np.arange(450, -1, -1, dtype=float) / 10.
    v6 = np.asarray([encode(t, 'obs-v6') for t in times])
    v5 = np.asarray([encode(t, 'obs-v5') for t in times])
    check('time_450_steps_strictly_decreasing', np.all(np.diff(v6[:, ti]) < 0)
          and np.allclose(v6[:, ti], times/45., atol=1e-12))
    check('time_distinguishes_45_40_30', len({encode(t, 'obs-v6')[ti]
          for t in (45., 40., 30.)}) == 3)
    check('legacy_first15_alias_reproduced', np.all(v5[:151, ti] == 1.))
    check('only_time_field_changes', np.array_equal(np.delete(v5, ti, axis=1),
                                                   np.delete(v6, ti, axis=1)))
    check('graded_history_preserved_v6', np.all(v6[:, yi] == .27))
    for version in ('obs-v2', 'obs-v3', 'obs-v4', 'obs-v5'):
        got = np.asarray([encode(t, version) for t in times])
        check('legacy_time_reproduced_' + version,
              np.array_equal(got[:, ti], np.clip(times/30., 0., 1.)))
    for rem in (46., float('inf'), float('nan')):
        try:
            encode(rem, 'obs-v6'); rejected = False
        except ValueError:
            rejected = True
        check('time_reject_invalid_' + str(rem), rejected)
    p = json.loads((NAV / 'config/training_loop_time_feedback_proposal_v1.json').read_bytes())
    nav = json.loads((NAV / p['binding']['navigation_contract']['path']).read_bytes())
    F.NAV_CONTRACT = str(NAV / p['binding']['navigation_contract']['path'])
    F.VALUE_CONTRACT = str(NAV / p['binding']['value_contract']['path'])
    envs, _, ids = F.build_envs(p, 'full')
    check('production_build18_train_time_v6', len(envs) == len(ids) == 18
          and all(e.observation_version == 'obs-v6' and e.t_max_s == 45.
                  and e.observation[0, ti] == 1. for e in envs))
    rec = envs[0].step(np.asarray([[.3, 0., 0.]]))
    check('production_step_time_decreases', abs(rec.obs[0, ti] - 44.9/45.) < 1e-12)
    # Exercise the actual evaluation constructor with zero rollout steps.
    # Its reset/observation is real; no checkpoint is evaluated or performance sampled.
    from unittest import mock
    captured = []
    real_env = E.E.VectorNominalNavEnv

    def constructor(*a, **kw):
        e = real_env(*a, **kw)
        captured.append((e.observation_version, e.observation[0, ti], e.t_max_s))
        e._done[0] = True
        return e

    ctx = E.E.build_rollout_ctx('full')
    with mock.patch.object(E.E, 'VectorNominalNavEnv', constructor):
        E.rollout(None, envs[0].spec, ctx, 400., p)
    check('production_development_constructor_time_v6', captured == [('obs-v6', 1., 45.)])
    paths = R.attempt_paths(p)
    check('production_coordinator_time_v6',
          paths['coordinator'].name == 'nav064_time_feedback_v1'
          and paths['development'].name == 'nav064_time_feedback_v1_development.json'
          and paths['closure'].name == 'nav064_time_feedback_v1_closure.json'
          and all(p['runtime_contract']['run_prefix'] in str(v) for v in paths.values())
          and all(r['run_id'] in R.training_command(r, 'time-proposal.json')
                  for r in p['run_registry']))
    wrong_attempt = copy.deepcopy(p)
    wrong_attempt['runtime_contract']['run_prefix'] = 'nav064_feedback_v1'
    try:
        R.attempt_paths(wrong_attempt); rejected = False
    except ValueError:
        rejected = True
    check('production_coordinator_rejects_old_prefix', rejected)
    files = set(base['code_files']) | {
        'tests/test_time_feedback_gate.py', 'config/navigation_contract_time_feedback_v1.json',
        'scripts/diagnose_time_feedback.py'}
    report = dict(base, schema='camra-time-feedback-gate-v1', checks=checks,
        implementation_versions=p['implementation_versions'],
        code_files={f: file_sha(NAV / f) for f in sorted(files)},
        protocol={'path': 'config/time_feedback_numerical_protocol_v1.json',
                  'sha256': file_sha(NAV / 'config/time_feedback_numerical_protocol_v1.json')},
        component_reports={str(basepath): file_sha(basepath)},
        verdict='PASS' if all(c['ok'] for c in checks) else 'FAIL')
    probe = out.with_suffix('.startup.json')
    with probe.open('x') as f:
        json.dump(report, f, indent=2)
    # Bind current files in memory only; this is not persisted training approval.
    fixture = copy.deepcopy(p)
    fixture['authorization']['training_approved'] = True
    for entry in fixture['binding'].values():
        if entry is fixture['binding']['numerical_gate']:
            entry['path'] = str(probe)
        entry['sha256'] = file_sha(NAV / entry['path'])
    verify_training_bindings(fixture, NAV)
    check('startup_v6_full_chain_accepted_in_memory', True)
    missing_gate = copy.deepcopy(report)
    missing_gate['checks'] = [c for c in checks if c['check'] != 'production_coordinator_time_v6']
    missing = out.with_suffix('.missing_time_check.json')
    with missing.open('x') as f: json.dump(missing_gate, f)
    bad = copy.deepcopy(fixture)
    bad['binding']['numerical_gate'] = {'path': str(missing), 'sha256': file_sha(missing)}
    try:
        verify_training_bindings(bad, NAV); rejected = False
    except ValueError:
        rejected = True
    check('startup_v6_reject_missing_time_evidence', rejected)
    for name in ('unapproved', 'wrong_scale', 'wrong_horizon', 'old_observation'):
        bad = copy.deepcopy(fixture)
        if name == 'unapproved': bad['authorization']['training_approved'] = False
        if name == 'wrong_horizon': bad['environment']['t_max_s'] = 30.
        if name == 'old_observation': bad['implementation_versions']['observation'] = 'obs-v5'
        if name == 'wrong_scale':
            bad_nav = copy.deepcopy(nav)
            bad_nav['observation_spec']['constants']['t_max_s'] = 30.
            wrong = out.with_suffix('.wrong_time_scale.json')
            with wrong.open('x') as f: json.dump(bad_nav, f)
            bad['binding']['navigation_contract'] = {'path': str(wrong), 'sha256': file_sha(wrong)}
        try:
            verify_training_bindings(bad, NAV); rejected = False
        except ValueError:
            rejected = True
        check('startup_v6_reject_' + name, rejected)
    report['checks'] = checks
    report['verdict'] = 'PASS' if all(c['ok'] for c in checks) else 'FAIL'
    report['limits'] = base['limits'] + [
        'Time-only version; safety features and lambda unchanged; no performance rollout or new training.',
        'Default remains obs-v5 for old callers; the candidate explicitly selects obs-v6.']
    with out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(report['verdict'], len(checks), 'checks')
    return int(report['verdict'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
