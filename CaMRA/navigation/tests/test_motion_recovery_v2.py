"""Main-agent CPU regressions for the retained clock/cleanup counterexamples."""
import io
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from camra_navigation.envs.simulation_clock_v2 import SimulationClock
from camra_navigation.envs.motion_capture_v2 import capture_case
import nav043_process_cleanup_v2 as C
import run_nav043_motion_probe_recovery_v2 as L
import nav043_motion_probe_recovery_guard as G
import audit_nav043_motion_probe_recovery_v2 as A
from test_motion_capture import FakeLow, recorder, state


class ActualIndexedFake(FakeLow):
    def __init__(self, dt=.005, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt

    def readback(self):
        return .009999999776482582+self.tick*self.dt, 2+self.tick

    def policy_step(self, *args, **kwargs):
        before = self.readback()[0]
        super().policy_step(*args, **kwargs)
        self.evidence[-1].update(sim_time_before=before,
                                sim_time_after=self.readback()[0])


@pytest.mark.parametrize('dt',[.005,float(np.float32(.005))])
def test_full_capture_and_production_saved_review(tmp_path,dt):
    low=ActualIndexedFake(dt)
    result=capture_case(low,[.4,0,0],tmp_path,low.readback,recorder())
    rows=[json.loads(s) for s in (tmp_path/'actual.jsonl').read_text().splitlines()]
    assert len(rows)==2200 and rows[-1]['sim_step_index']==2202
    assert result['total_J']==pytest.approx(132.)
    assert result['completed_q_count']==101
    q=json.loads((tmp_path/'completed_q.json').read_text())
    assert q['clock_validation']['accepted_ticks']==2200
    reset={'actual_clock_s':.009999999776482582,'actual_step_index':2,
           'actual_states':[q['initial_actual_state']],
           'friction':{'ok':True},'mass':{'ok':True}}
    G.persist(tmp_path/'reset.json',reset)
    result.update(case_id='M01',requested_command=[.4,0,0])
    G.persist(tmp_path/'result.json',result)
    reviewed=A.review_case(tmp_path,{'id':'M01','command':[.4,0,0]})
    assert reviewed['integrity_passed'] and reviewed['actual_ticks_reviewed']==2200


def test_895_tick_counterexample_is_numerical_representation_only():
    origin=.009999999776482582
    dt=float(np.float32(.005))
    clock=SimulationClock((origin,2))
    for tick in range(1,2201):
        clock.advance(tick,(origin+tick*dt,2+tick))
    assert abs((origin+895*dt)-origin-895*.005)>1.e-7
    assert clock.tick==2200 and clock.candidates==(dt,)


@pytest.mark.parametrize('fault',['gap','duplicate','stale','fractional','bool','negative'])
def test_wrong_actual_step_index_rejected(fault):
    c=SimulationClock((0.,100))
    c.advance(1,(.005,101))
    index={'gap':103,'duplicate':101,'stale':100,'fractional':102.1,
           'bool':True,'negative':-1}[fault]
    with pytest.raises(ValueError,match='index'):
        c.advance(2,(.010,index))
    assert c.tick==1


@pytest.mark.parametrize('time',[0.,.01,.0049,float('nan'),float('inf')])
def test_wrong_or_stale_actual_time_rejected(time):
    c=SimulationClock((0.,100))
    with pytest.raises(ValueError,match='clock|time'):
        c.advance(1,(time,101))
    assert c.tick==0


def test_tick_sequence_still_rejects_gap():
    c=SimulationClock((0.,0))
    with pytest.raises(ValueError,match='index'):
        c.advance(2,(.010,2))


def test_interruption_saves_partial_without_retry(tmp_path):
    low=ActualIndexedFake(float(np.float32(.005)),fail_tick=238)
    with pytest.raises(RuntimeError,match='synthetic'):
        capture_case(low,[.1,0,0],tmp_path,low.readback,recorder())
    q=G.load(tmp_path/'completed_q.json')
    assert q['ticks_captured']==237 and q['partial_tail_ticks']==17
    assert q['clock_validation']['accepted_ticks']==237
    assert (tmp_path/'energy.npz').is_file() and (tmp_path/'commands.json').is_file()


class TimeoutProcess:
    pid=987654321
    returncode=None
    def wait(self,timeout):
        raise subprocess.TimeoutExpired(['synthetic-child'],timeout)
    def poll(self):
        return None


def test_repeated_wait_timeout_does_not_escape(monkeypatch):
    sent=[]
    monkeypatch.setattr(C.os,'killpg',lambda pid,sig:sent.append(sig))
    result=C.stop_group(TimeoutProcess(),wait_s=0.)
    assert sent==[signal.SIGTERM,signal.SIGKILL,0]
    assert len(result['errors'])==2 and not result['cleanup_confirmed']
    assert result['returncode'] is None and not result['group_absent']


def test_absent_process_group_is_confirmed(monkeypatch):
    def absent(*args):
        raise ProcessLookupError()
    class Done:
        pid=987654321
        def wait(self,timeout): return 1
        def poll(self): return 1
    monkeypatch.setattr(C.os,'killpg',absent)
    result=C.stop_group(Done())
    assert result['group_absent'] and result['cleanup_confirmed']
    assert result['returncode']==1


def test_parent_retains_primary_error_and_terminal_on_wait_timeouts(tmp_path,monkeypatch):
    out=tmp_path/'isolated'
    evidence=tmp_path/'evidence'
    evidence.mkdir()
    docs=[tmp_path/name for name in ('proposal','approval','readiness')]
    for doc in docs: doc.write_text('synthetic')
    for name,doc in zip(('PROPOSAL','APPROVAL','READINESS'),docs):
        monkeypatch.setattr(G,name,doc)
    monkeypatch.setattr(G,'EVIDENCE',evidence)
    p={'output_root':str(out),'environment':{'python':sys.executable},
       'limits':{'gpu_wall_s':60},
       'recovery':{'failed_slot_wall_charge_s':900}}
    monkeypatch.setattr(L,'prepare',lambda:(p,{},{}))
    monkeypatch.setattr(G,'gpu_ready',lambda *_:{})
    monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:'synthetic-head')
    def popen(*a,**k):
        G.persist(out/'ERROR.json',{'error':'original capture exception'})
        return TimeoutProcess()
    monkeypatch.setattr(L.subprocess,'Popen',popen)
    monkeypatch.setattr(C.os,'killpg',lambda *a:None)
    with pytest.raises(ValueError,match='first application'):
        L.coordinate()
    held=G.load(out/'HELD.json')
    assert held['primary_application_error']['error']=='original capture exception'
    assert held['cleanup_pending']
    cleanup=G.load(out/'process_cleanup.json')
    assert len(cleanup['errors'])==2 and not cleanup['cleanup_confirmed']
    exit_record=G.load(evidence/'exit.json')
    assert 'first application' in exit_record['error']
    assert not exit_record['retry_performed'] and not (out/'FINISHED.json').exists()


def test_real_missing_recovery_approval_blocks_before_resources(monkeypatch,tmp_path):
    monkeypatch.setattr(G,'APPROVAL',tmp_path/'missing-approval')
    monkeypatch.setattr(G,'gpu_ready',lambda *_:pytest.fail('GPU reached'))
    with pytest.raises(ValueError,match='recovery APPROVED.*missing'):
        L.coordinate()
