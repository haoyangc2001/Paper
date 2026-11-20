"""CPU-only approval/refusal, resource-budget and saved-capture checks."""
import io
import json
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import nav043_motion_probe_guard as G
import run_nav043_motion_probe as L
import audit_nav043_motion_probe as A
from test_motion_capture import FakeLow, recorder
from camra_navigation.envs.motion_capture_v1 import capture_case


def put(path, obj):
    path.write_text(json.dumps(obj)+'\n')
    Path(str(path)+'.sha256').write_text(G.sha(path)+'\n')


@pytest.fixture
def documents(tmp_path, monkeypatch):
    # Synthetic approval exists only in pytest's private temporary directory.
    for name, path in [('PROPOSAL',tmp_path/'proposal.json'),
                       ('APPROVAL',tmp_path/'approval.json'),
                       ('READINESS',tmp_path/'readiness.json'),
                       ('EVIDENCE',tmp_path/'evidence')]:
        monkeypatch.setattr(G, name, path)
    monkeypatch.setattr(G, 'ROOT', tmp_path)
    source = tmp_path/'source.py'
    source.write_text('# synthetic supporting identity\n')
    p = {'status':'PROPOSED', 'source_hashes':{'source.py':G.sha(source)},
         'output_root':str(tmp_path/'formal'), 'limits':{'gpu_wall_s':7200},
         'environment':{'physical_gpu_index':6, 'python':'/synthetic/python'},
         'schedule':{'cases':['synthetic']}}
    put(G.PROPOSAL, p)
    monkeypatch.setattr(G, 'PROPOSAL_PIN', G.sha(G.PROPOSAL))
    r = {'status':'PREPARED_ONLY', 'proposal_sha256':G.PROPOSAL_PIN,
         'source_hashes':{'source.py':G.sha(source)}}
    put(G.READINESS, r)
    a = {'schema':'nav043-motion-probe-approval-v1', 'status':'APPROVED',
         'proposal_sha256':G.PROPOSAL_PIN, 'readiness_sha256':G.sha(G.READINESS),
         **{key:p[key] for key in ('output_root','limits','environment','schedule')}}
    put(G.APPROVAL, a)
    return p, a


def test_exact_synthetic_approval_and_sources(documents):
    p, a, r = G.approved()
    assert a['status'] == 'APPROVED' and r['status'] == 'PREPARED_ONLY'
    assert not Path(p['output_root']).exists() and not G.EVIDENCE.exists()


@pytest.mark.parametrize('field', ['status','proposal_sha256','readiness_sha256',
                                 'output_root','limits','environment','schedule'])
def test_changed_scope_is_not_approved(documents, field):
    p, a = documents
    a[field] = 'changed'
    put(G.APPROVAL, a)
    with pytest.raises(ValueError, match='exact scope'):
        G.approved()
    assert not Path(p['output_root']).exists()


def test_prepared_readiness_is_not_approval(documents, monkeypatch):
    p, a = documents
    G.APPROVAL.unlink()
    monkeypatch.setattr(G, 'registered_context', lambda *_:pytest.fail('context reached without approval'))
    monkeypatch.setattr(G, 'gpu_ready', lambda *_:pytest.fail('GPU reached without approval'))
    with pytest.raises(ValueError, match='APPROVED.*missing'):
        L.coordinate()
    assert not G.EVIDENCE.exists() and not Path(p['output_root']).exists()


def test_changed_support_cannot_launch(documents):
    (G.ROOT/'source.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='supporting source'):
        G.approved()


@pytest.mark.parametrize('marker', ['formal','lock'])
def test_existing_attempt_refuses_without_overwrite(documents, marker):
    p, _ = documents
    if marker == 'formal':
        Path(p['output_root']).mkdir()
    else:
        G.EVIDENCE.mkdir()
        (G.EVIDENCE/'launch.lock').write_text('existing owner')
    with pytest.raises(ValueError, match='existing output/attempt'):
        G.refuse_conflicts(p, Path(L.__file__).resolve())
    if marker == 'lock':
        assert (G.EVIDENCE/'launch.lock').read_text() == 'existing owner'


def test_step_accounting_includes_setup_case_reset_and_cleanup():
    journal = io.StringIO()
    b = G.ExplicitStepBudget(journal, total_cap=8, reserve_cap=4,
                             case_ids=('M01','M02'), ticks_per_case=2)
    observed = []
    def step():
        observed.append(b.snapshot())
    b.call(step)
    for name in ('M01','M02'):
        b.begin_case(name)
        b.call(step)
        b.call(step)
        b.finish_case()
        b.call(step)
    assert b.calls == b.returned == 7 and b.reserve == 3 and b.finished_cases == 2
    assert observed[0]['in_flight'] and observed[0]['explicit_step_calls'] == 1
    assert observed[-1]['case_id'] is None


@pytest.mark.parametrize('cap', ['total','reserve','case'])
def test_cap_refusal_happens_before_original_step(cap):
    stream = io.StringIO()
    b = G.ExplicitStepBudget(stream, total_cap=1 if cap == 'total' else 9,
        reserve_cap=1 if cap == 'reserve' else 9, case_ids=('M01',), ticks_per_case=1)
    calls = []
    if cap == 'case':
        b.begin_case('M01')
    b.call(lambda:calls.append(1))
    with pytest.raises(ValueError, match='allocation exhausted'):
        b.call(lambda:calls.append(2))
    assert calls == [1] and b.calls == 1
    assert json.loads(stream.getvalue().splitlines()[-1])['kind'] == 'cap_refused'


def test_first_failed_physical_call_is_retained_and_blocks_more_steps():
    stream = io.StringIO()
    b = G.ExplicitStepBudget(stream)
    def fail():
        raise RuntimeError('synthetic physical interruption')
    with pytest.raises(RuntimeError):
        b.call(fail)
    assert b.calls == 1 and b.returned == 0 and b.in_flight
    with pytest.raises(ValueError, match='did not return'):
        b.call(lambda:pytest.fail('no next step after failure'))
    assert len(stream.getvalue().splitlines()) == 1


def test_wrong_case_order_and_retry_are_refused():
    b = G.ExplicitStepBudget(io.StringIO(), case_ids=('M01',), ticks_per_case=1)
    with pytest.raises(ValueError):
        b.begin_case('M02')
    b.begin_case('M01')
    with pytest.raises(ValueError):
        b.finish_case()
    b.call(lambda:None)
    b.finish_case()
    with pytest.raises(ValueError):
        b.begin_case('M01')


def test_cleanup_kills_descendants_even_after_leader_exit(monkeypatch):
    sent = []
    monkeypatch.setattr(L.os, 'killpg', lambda pid, sig:sent.append((pid,sig)))
    class Process:
        pid = 87654321
        def wait(self, timeout):
            return 0
    L.stop_group(Process())
    assert sent == [(87654321,signal.SIGTERM),(87654321,signal.SIGKILL)]


def fake_coordinate(monkeypatch, p):
    monkeypatch.setattr(L, 'prepare', lambda:(p, {'synthetic':True}, {'synthetic':True}))
    monkeypatch.setattr(G, 'gpu_ready', lambda *_:{'synthetic':True})
    original = L.subprocess.check_output
    monkeypatch.setattr(L.subprocess, 'check_output',
        lambda *a,**k:'synthetic-head\n' if a[0][:2] == ['git','rev-parse'] else original(*a,**k))


def test_output_appearing_after_preflight_is_never_modified(documents, monkeypatch):
    p, _ = documents
    fake_coordinate(monkeypatch, p)
    output = Path(p['output_root'])
    def race(*_):
        output.mkdir()
        (output/'precious').write_text('another allocation')
        return {'synthetic':True}
    monkeypatch.setattr(G, 'gpu_ready', race)
    monkeypatch.setattr(L.subprocess, 'Popen', lambda *a,**k:pytest.fail('application started after conflict'))
    with pytest.raises(FileExistsError):
        L.coordinate()
    assert [f.name for f in output.iterdir()] == ['precious']
    assert (G.EVIDENCE/'launch.lock').exists()


def test_wall_timeout_stops_one_group_and_preserves_held(documents, monkeypatch):
    p, _ = documents
    fake_coordinate(monkeypatch, p)
    times, started, stopped = iter([0.,0.]), [], []
    monkeypatch.setattr(L.time, 'monotonic', lambda:next(times,7201.))
    class Process:
        pid = 87654321
        returncode = None
        def poll(self):
            return self.returncode
    process = Process()
    def popen(*a, **k):
        started.append(k)
        return process
    def stop(proc):
        stopped.append(proc.pid)
        proc.returncode = -15
    monkeypatch.setattr(L.subprocess, 'Popen', popen)
    monkeypatch.setattr(L, 'stop_group', stop)
    with pytest.raises(ValueError, match='first application failure'):
        L.coordinate()
    output = Path(p['output_root'])
    held = G.load(output/'HELD.json')
    assert held['timeout'] and held['applications_started'] == 1
    assert len(started) == 1 and stopped and not (output/'FINISHED.json').exists()
    assert started[0]['start_new_session']
    assert 'calibration/src' in started[0]['env']['PYTHONPATH']


@pytest.mark.parametrize('completed_cases', [11,12])
def test_finished_ledger_is_checked_before_marking_finished(documents, monkeypatch, completed_cases):
    p, _ = documents
    fake_coordinate(monkeypatch, p)
    output = Path(p['output_root'])
    class Process:
        pid = 87654321
        returncode = 0
        def poll(self):
            return 0
    def popen(*a, **k):
        G.persist(output/'CAPTURE_FINISHED.json', {'synthetic':True})
        G.persist(output/'child_step_accounting.json', {
            'completed_cases':completed_cases, 'in_flight':False, 'case_id':None,
            'explicit_step_calls':26402, 'returned_calls':26402, 'reserve_calls':2})
        return Process()
    monkeypatch.setattr(L.subprocess, 'Popen', popen)
    monkeypatch.setattr(L, 'stop_group', lambda *_:None)
    if completed_cases == 11:
        with pytest.raises(ValueError, match='allocation/evidence mismatch'):
            L.coordinate()
        assert (output/'HELD.json').exists() and not (output/'FINISHED.json').exists()
    else:
        assert L.coordinate() == 0
        assert (output/'FINISHED.json').exists() and not (output/'HELD.json').exists()


@pytest.fixture
def saved_case(tmp_path):
    class StampedLow(FakeLow):
        def policy_step(self, *args, **kwargs):
            before = self.clock()
            super().policy_step(*args, **kwargs)
            self.evidence[-1].update(sim_time_before=before, sim_time_after=self.clock())
    low = StampedLow()
    case = {'id':'M01', 'command':[0.,0.,0.]}
    initial = vars(low.states()[0])
    G.persist(tmp_path/'reset.json', {'actual_clock_s':low.clock(),
        'actual_states':[initial], 'friction':{'ok':True}, 'mass':{'ok':True}})
    result = capture_case(low, case['command'], tmp_path, low.clock, recorder())
    result.update(case_id=case['id'], requested_command=case['command'])
    G.persist(tmp_path/'result.json', result)
    return tmp_path, case


def test_saved_actual_pose_q_work_and_commands_are_recomputed(saved_case):
    path, case = saved_case
    row = A.review_case(path, case)
    assert row['integrity_passed'] and row['total_J'] == pytest.approx(132.)
    assert row['measurement_mean_completed_q'][0] == pytest.approx(.2)
    assert row['measurement_mean_instantaneous_body_velocity'][0] == pytest.approx(99.)
    assert not row['steady_state_established']


@pytest.mark.parametrize('fault', ['pose','clock','command','q','work','missing_tick'])
def test_saved_capture_tampering_cannot_pass(saved_case, fault):
    path, case = saved_case
    if fault in ('pose','clock','missing_tick'):
        rows = [json.loads(line) for line in (path/'actual.jsonl').read_text().splitlines()]
        if fault == 'pose':
            rows[219]['state']['x'] += .1
        elif fault == 'clock':
            rows[19]['sim_time_s'] += .005
        else:
            rows.pop(17)
        (path/'actual.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    else:
        name = {'command':'commands.json','q':'completed_q.json','work':'result.json'}[fault]
        obj = G.load(path/name)
        if fault == 'command':
            obj[-1]['command_seen'] = [[.1,0.,0.]]
        elif fault == 'q':
            obj['completed_intervals'][0]['start_tick'] = 179
        else:
            obj['measurement_J'] += 1.
        (path/name).write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        A.review_case(path, case)


def test_whole_journal_and_missing_final_return(tmp_path):
    path = tmp_path/'steps.jsonl'
    with path.open('w') as stream:
        b = G.ExplicitStepBudget(stream)
        b.call(lambda:None)  # setup
        for i in range(1,13):
            b.begin_case(f'M{i:02}')
            for _ in range(2200):
                b.call(lambda:None)
            b.finish_case()
        b.call(lambda:None)  # cleanup
    assert A.review_steps(path) == {'explicit_step_calls':26402, 'returned_calls':26402,
                                    'reserve_calls':2, 'completed_cases':12}
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError, match='incomplete'):
        A.review_steps(path)
