"""Synthetic adapter tests, with the actual frozen-format energy recorder."""
import io
import json
import math

import numpy as np
import pytest
import torch

from camra_navigation.envs.motion_capture_v1 import MotionCapture, capture_case
from camra_navigation.envs.physical_v2 import PhysicalState, require_command_evidence
from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute


def state(tick, *, height=.3, body_v=99.):
    # Actual planar speed .2m/s; deliberately disagreeing endpoint body velocity.
    return PhysicalState(tick*.005*.2, 0., 0., body_v, 0., 0., height, (1.,0.,0.,0.))


class FakeLow:
    def __init__(self, *, fail_tick=None, fall_tick=None):
        self.tick = self.policy_calls = 0
        self.evidence = []
        self.fail_tick, self.fall_tick = fail_tick, fall_tick

    def states(self):
        return [state(self.tick)]

    def clock(self):
        return 123. + self.tick*.005

    def policy_step(self, command, *, monitor, recorder):
        begin = self.tick
        for _ in range(4):
            if self.tick+1 == self.fail_tick:
                raise RuntimeError('synthetic interruption before step')
            self.tick += 1
            recorder.record(tau_applied_lab=torch.ones((1,12)),
                            tau_request_lab=torch.ones((1,12))*7,
                            qdot_pre_lab=torch.ones((1,12)),
                            qdot_post_lab=torch.ones((1,12)))
            fallen = self.tick == self.fall_tick
            events = monitor.advance([state(self.tick,height=.1 if fallen else .3)],
                                     physics_tick=self.tick)
            for i in events['ended']:
                recorder.mark_event([i], 'timeout' if monitor.labels[i] == 'timeout' else 'early_termination:fall')
            if not any(monitor.active):
                break
        self.policy_calls += 1
        self.evidence.append(require_command_evidence(command, command,
            tick_before=begin, tick_after=self.tick, substeps=self.tick-begin))


def recorder():
    return NavEnergyRecorder(2200, 1, device='cpu', expected_steps=2200)


def test_whole_case_captures_real_warmup_q_and_separate_work(tmp_path):
    low = FakeLow()
    rec = recorder()
    result = capture_case(low, [.4,0,0], tmp_path, low.clock, rec)
    rows = [json.loads(r) for r in (tmp_path/'actual.jsonl').read_text().splitlines()]
    q = json.loads((tmp_path/'completed_q.json').read_text())
    assert len(rows) == 2200 and rows[199]['phase'] == 'warmup' and rows[200]['phase'] == 'measurement'
    assert rows[199]['requested_command'] == [0.,0.,0.]
    assert rows[200]['requested_command'][0] == pytest.approx(.4)
    assert result['completed_q_count'] == 101
    assert q['completed_intervals'][0]['start_tick'] == 180
    assert q['completed_intervals'][0]['end_tick'] == 200
    assert q['completed_intervals'][0]['q_vx_vy_yaw_rate'][0] == pytest.approx(.2)
    assert rows[199]['state']['vx'] == 99.
    assert result['warmup_J'] == pytest.approx(12.)
    assert result['measurement_J'] == pytest.approx(120.)
    assert result['total_J'] == pytest.approx(132.)
    audit = offline_recompute(rec.finish(), 0)
    assert audit['steps_match'] and audit['rel_diff'] <= 1.e-10


@pytest.mark.parametrize('fall_tick', [17, 200, 237, 2200])
def test_fall_retains_tail_and_stops(tmp_path, fall_tick):
    low = FakeLow(fall_tick=fall_tick)
    with pytest.raises(ValueError, match='fall'):
        capture_case(low, [0.,0.,1.], tmp_path, low.clock, recorder())
    rows = [json.loads(r) for r in (tmp_path/'actual.jsonl').read_text().splitlines()]
    snap = json.loads((tmp_path/'completed_q.json').read_text())
    assert low.tick == len(rows) == snap['ticks_captured'] == fall_tick
    assert snap['label'] == 'fall'
    if fall_tick > 200:
        assert snap['partial_tail_ticks'] == (fall_tick-200) % 20
    assert (tmp_path/'energy.npz').is_file() and (tmp_path/'commands.json').is_file()


def test_interruption_preserves_partial_data_without_retry(tmp_path):
    low = FakeLow(fail_tick=238)
    with pytest.raises(RuntimeError, match='synthetic'):
        capture_case(low, [.1,0,0], tmp_path, low.clock, recorder())
    assert low.tick == 237
    saved = json.loads((tmp_path/'completed_q.json').read_text())
    assert saved['partial_tail_ticks'] == 17
    assert saved['last_completed_q']['end_tick'] == 220
    assert np.load(tmp_path/'energy.npz')['tau_applied'].shape == (237,1,12)
    with pytest.raises(ValueError):
        capture_case(low, [.1,0,0], tmp_path, low.clock, recorder())


@pytest.mark.parametrize('fault', ['stale', 'gap'])
def test_malformed_command_proof_cannot_pass(tmp_path, fault):
    class WrongProof(FakeLow):
        def policy_step(self, *args, **kwargs):
            super().policy_step(*args, **kwargs)
            if self.tick == 2200:
                if fault == 'stale':
                    self.evidence[-1]['command_seen'] = [[0.,0.,0.]]
                else:
                    self.evidence[-1]['tick_before'] -= 1
    low = WrongProof()
    with pytest.raises(ValueError, match='stale|gapped'):
        capture_case(low, [.4,0,0], tmp_path, low.clock, recorder())
    assert (tmp_path/'actual.jsonl').is_file() and (tmp_path/'commands.json').is_file()
    assert low.tick == 2200


@pytest.mark.parametrize('bad_tick', [0, 2, -1])
def test_missing_duplicate_samples_rejected(bad_tick):
    stream = io.StringIO()
    monitor = MotionCapture(state(0), lambda:123., stream)
    with pytest.raises(ValueError, match='tick'):
        monitor.advance([state(1)], physics_tick=bad_tick)


def test_actual_clock_mismatch_retains_readback():
    stream = io.StringIO()
    monitor = MotionCapture(state(0), lambda:123., stream)
    with pytest.raises(ValueError, match='clock'):
        monitor.advance([state(1)], physics_tick=1)
    assert json.loads(stream.getvalue())['physics_tick'] == 1


@pytest.mark.parametrize('command', [[1.,0,0], [-1.,0,0], [0,.4,0], [0,-.4,0], [0,0,1.], [0,0,-1.]])
def test_float32_frozen_endpoints_are_accepted(tmp_path, command):
    low = FakeLow(fall_tick=1)
    with pytest.raises(ValueError, match='actual fall'):
        capture_case(low, command, tmp_path, low.clock, recorder())
    assert low.tick == 1  # accepted input, then the deliberate terminal fixture


@pytest.mark.parametrize('command', [[1.1,0,0], [0,.41,0], [0,0,1.01], [math.nan,0,0], [0,0]])
def test_bad_command_refused_before_output(tmp_path, command):
    low = FakeLow()
    with pytest.raises(ValueError, match='command'):
        capture_case(low, command, tmp_path, low.clock, recorder())
    assert list(tmp_path.iterdir()) == [] and low.tick == 0
