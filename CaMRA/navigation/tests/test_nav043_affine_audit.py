"""Synthetic output/tampering checks; no fit or real trajectory is loaded."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_nav043_affine_response as audit
import launch_nav043_affine_response_once as launcher


@pytest.fixture
def solution():
    # Orthogonal columns: the exact singular values are known without any SVD.
    x = np.array([[-1., 1., 1.], [1., -1., 1.], [1., 1., -1.], [-1., -1., -1.]])
    theta = np.array([.5, .3, -.02])
    y = x@theta+.1
    record = dict(a=.5, b=.3, d=-.02, rows=4, valid=True, rank=3,
                  singular_values=[2., 2., 2.], condition=1., residual_sum=[.04])
    return x, y, record


def test_saved_solution_checked_without_any_fit(solution, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('verification must not solve again')
    for name in ('lstsq', 'solve', 'inv', 'svd'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    check = audit.verify_solution(*solution)
    assert check['normal_residual_inf'] < 1.e-12
    assert check['residual_squared_sum'] == pytest.approx(.04)


@pytest.mark.parametrize('key,value', [('a', .6), ('d', .01), ('rows', 5),
    ('residual_sum', [.03]), ('singular_values', [3., 3., 3.]), ('rank', 2)])
def test_saved_parameter_or_solver_metadata_tampering_rejected(solution, key, value):
    x, y, record = solution
    record[key] = value
    with pytest.raises(ValueError):
        audit.verify_solution(x, y, record)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, .01])
def test_numeric_comparison_rejects_invalid_saved_values(bad):
    with pytest.raises(ValueError):
        audit.equivalent({'score': [0.]}, {'score': [bad]})


def test_replay_preserves_partial_tail_and_missing_prefix():
    base = audit.base
    parameters = np.array([[.5, .3, -.01]]*3)
    trace = SimpleNamespace(initial=np.zeros(3), ticks=21,
        interval_commands=np.array([[.4, 0., .2], [.1, 0., -.1]]),
        identity=dict(trial='synthetic', level=base.LEVELS[0], method='full', kind='baseline'))
    trace.actual, responses = audit.affine.free_replay(trace, parameters)
    trace.observed_twists = np.asarray([base.interval_twist(trace.initial, trace.actual[19])])
    row = audit.replay_row(trace, parameters)
    assert row['metrics']['ticks'] == 21 and row['metrics']['full_velocity_intervals'] == 1
    assert row['metrics']['position_RMSE_m'] == 0.
    assert all(value == {'complete': False} for value in row['prefixes'].values())
    bad = copy.deepcopy(row)
    bad['metrics']['ticks'] = 20
    with pytest.raises(ValueError):
        audit.equivalent(row, bad)


def test_expected_output_inventory_has_exactly_nine_solves():
    protocol = {'traces': [dict(level=level, method=method, kind=kind)
        for level in audit.base.LEVELS for method in ('full', 'uncalibrated')
        for kind in ('baseline', 'repeat', 'W_selected')]}
    files = audit.expected_files(protocol)
    assert len(files) == 41
    assert len([p for p in files if p.endswith('-solve-result.json')]) == 9


@pytest.fixture
def launch_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setenv('TMUX_PANE', '%synthetic')
    monkeypatch.setattr(launcher.sys, 'executable', launcher.base.PYTHON)
    proposal = {'output_root': str(tmp_path/'output'), 'limits': {'cpu_wall_s': 3600}}
    proposal_path, ready_path = tmp_path/'proposal.json', tmp_path/'ready.json'
    approval_path = tmp_path/'approval.json'
    proposal_path.write_text(json.dumps(proposal))
    approval_path.write_text(json.dumps({'status': 'SYNTHETIC'}))
    ready_path.write_text(json.dumps({'status': 'PREPARED_ONLY',
        'proposal_sha256': launcher.base.sha(proposal_path), 'source_hashes': {}}))
    Path(str(ready_path)+'.sha256').write_text(launcher.base.sha(ready_path)+'\n')
    monkeypatch.setattr(launcher.affine, 'PROPOSAL', proposal_path)
    monkeypatch.setattr(launcher.affine, 'APPROVAL', approval_path)
    monkeypatch.setattr(launcher, 'READINESS', ready_path)
    monkeypatch.setattr(launcher, 'HERE', tmp_path/'evidence')
    monkeypatch.setattr(launcher.affine, 'approval', lambda: (proposal, {'status': 'SYNTHETIC'}))
    return proposal


def test_missing_approval_cannot_create_marker_or_launch(launch_scope, monkeypatch):
    def denied():
        raise ValueError('missing approval')
    def forbidden(*args, **kwargs):
        pytest.fail('no subprocess before approval')
    monkeypatch.setattr(launcher.affine, 'approval', denied)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='missing approval'):
        launcher.main()
    assert not launcher.HERE.exists()


@pytest.mark.parametrize('existing', ['output', 'lock'])
def test_existing_output_or_prior_attempt_cannot_start(launch_scope, existing):
    if existing == 'output':
        Path(launch_scope['output_root']).mkdir()
    else:
        launcher.HERE.mkdir()
        (launcher.HERE/'launch.lock').touch()
    with pytest.raises(ValueError, match='already'):
        launcher.prepare()


def test_supporting_source_change_rejected(launch_scope):
    readiness = launcher.base.load(launcher.READINESS)
    readiness['source_hashes'] = {str(launcher.READINESS): 'not-the-source-hash'}
    launcher.READINESS.write_text(json.dumps(readiness))
    Path(str(launcher.READINESS)+'.sha256').write_text(launcher.base.sha(launcher.READINESS)+'\n')
    with pytest.raises(ValueError, match='changed supporting'):
        launcher.prepare()


def test_timeout_preserves_lock_and_never_retries(launch_scope, monkeypatch):
    calls = []
    class FakeProcess:
        pid = 1
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            if timeout is not None:
                assert timeout == 3600
                raise subprocess.TimeoutExpired('synthetic', timeout)
            return self.returncode
        def kill(self):
            self.returncode = -9
    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        return FakeProcess()
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    assert launcher.main() == -9
    assert len(calls) == 1 and (launcher.HERE/'launch.lock').is_file()
    record = launcher.base.load(launcher.HERE/'exit.json')
    assert not record['retry_performed'] and 'TimeoutExpired' in record['error']
    with pytest.raises(ValueError, match='already'):
        launcher.main()
    assert len(calls) == 1


def test_auditor_refuses_before_loading_trajectories(monkeypatch):
    monkeypatch.setattr(audit.socket, 'gethostname', lambda: 'tiger11')
    monkeypatch.setenv('TMUX', 'synthetic')
    monkeypatch.setattr(audit.sys, 'executable', audit.base.PYTHON)
    def denied():
        raise ValueError('unapproved')
    def forbidden(*args, **kwargs):
        pytest.fail('no data load before approval')
    monkeypatch.setattr(audit.affine, 'approval', denied)
    monkeypatch.setattr(audit.base, 'read_trace', forbidden)
    with pytest.raises(ValueError, match='unapproved'):
        audit.main()


def test_auditor_missing_output_cannot_read_real_data(launch_scope, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('no data access before a finished output exists')
    monkeypatch.setattr(audit.base, 'verify_protocol', forbidden)
    with pytest.raises(ValueError, match='no complete affine output'):
        audit.main()
