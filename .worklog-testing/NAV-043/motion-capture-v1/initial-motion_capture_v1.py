"""Isolated fixed-command capture core; no simulator, model fit or launcher.

The caller supplies an already reset clean_v3-compatible low-level adapter,
an actual simulator clock and a fresh energy recorder. Tests can supply a
synthetic adapter without importing Isaac or authorizing physical execution.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from camra_navigation.motion_state import PoseSample, CompletedMotionTracker

WARMUP_TICKS = 200
TOTAL_TICKS = 2200
DT = .005


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class MotionCapture:
    """A clean_v3 monitor consuming consecutive actual 200Hz samples."""
    def __init__(self, initial_state, clock, stream):
        initial_state.validate()
        self.clock, self.stream = clock, stream
        self.clock_origin = float(clock())
        if not math.isfinite(self.clock_origin):
            raise ValueError('nonfinite actual initial clock')
        self.tick = 0
        self.active, self.labels = [True], ['active']
        self.tracker = None
        self.completed = []
        self.warmup_previous = None
        self.initial_state = asdict(initial_state)
        self.requested_command = [0., 0., 0.]

    def advance(self, states, *, physics_tick):
        if not self.active[0] or physics_tick != self.tick+1 or len(states) != 1:
            raise ValueError('duplicate/gapped tick, wrong environment or terminated capture')
        state, now = states[0], float(self.clock())
        # Preserve the physical readback before rejecting malformed input.
        try:
            state.validate()
        except ValueError:
            self.stream.write(json.dumps({'physics_tick':physics_tick,
                'invalid_actual_state':repr(asdict(state))})+'\n')
            self.stream.flush()
            raise
        row = {'physics_tick':physics_tick, 'sim_time_s':now,
               'phase':'warmup' if physics_tick <= WARMUP_TICKS else 'measurement',
               'state':asdict(state), 'requested_command':list(self.requested_command),
               'command_evidence':'join with saved 50Hz proof by tick_before/tick_after; missing proof is invalid'}
        if not math.isfinite(now):
            row['sim_time_s'] = repr(now)
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.stream.flush()
        if not math.isfinite(now) or abs(now-self.clock_origin-physics_tick*DT) > 1.e-7:
            raise ValueError('actual simulator clock differs from consecutive physics ticks')
        self.tick = physics_tick
        pose = PoseSample(self.tick, (state.x, state.y, state.yaw))
        fall = state.height < .15 or state.tilt_deg > 60.
        ended = fall or self.tick == TOTAL_TICKS
        if self.tick == WARMUP_TICKS-20:
            self.warmup_previous = pose
        if self.tick == WARMUP_TICKS:
            self.tracker = CompletedMotionTracker(self.warmup_previous, pose)
            self.completed.append(self.tracker.last_completed.as_dict())
        elif self.tick > WARMUP_TICKS:
            self.tracker.advance(pose, terminal=ended)
            if self.tracker.partial_ticks == 0:
                self.completed.append(self.tracker.last_completed.as_dict())
        if ended:
            self.active[0] = False
            self.labels[0] = 'fall' if fall else 'timeout'
        return {'ended':[0] if ended else []}

    def snapshot(self):
        tracker = self.tracker
        return {'initial_actual_state':self.initial_state, 'ticks_captured':self.tick,
                'warmup_ticks_captured':min(self.tick, WARMUP_TICKS),
                'measurement_ticks_captured':max(0, self.tick-WARMUP_TICKS),
                'label':self.labels[0], 'completed_intervals':self.completed,
                'partial_tail_ticks':tracker.partial_ticks if tracker else None,
                'partial_tail_endpoint':({'tick':tracker.latest.tick,
                    'pose':list(tracker.latest.pose)} if tracker else None),
                'last_completed_q':tracker.last_completed.as_dict() if tracker else None}


def capture_case(low, command, out, clock, recorder):
    """Capture one existing reset; never create an app, reset, retry or fit.

    Warmup work and measurement work share a saved raw buffer, with separate J
    subtotals. All partial records are saved in finally; a failure propagates.
    """
    command = np.asarray(command, dtype=np.float32)
    if (command.shape != (3,) or not np.isfinite(command).all()
            or np.any(np.abs(command) > [1., .4, 1.])):
        raise ValueError('command outside frozen domain')
    if low.tick != 0 or low.policy_calls != 0 or low.evidence:
        raise ValueError('fresh actual reset required before each case')
    out = Path(out)
    if any((out/name).exists() for name in ('actual.jsonl','energy.npz','completed_q.json','commands.json')):
        raise ValueError('capture output already exists; no overwrite or retry')
    initial = low.states()
    if len(initial) != 1:
        raise ValueError('exactly one physical environment required')
    energy = None
    with (out/'actual.jsonl').open('x') as stream:
        monitor = MotionCapture(initial[0], clock, stream)
        recorder.begin_window()
        try:
            while monitor.active[0]:
                applied = np.zeros(3, dtype=np.float32) if monitor.tick < WARMUP_TICKS else command
                monitor.requested_command = applied.tolist()
                low.policy_step([applied.tolist()], monitor=monitor, recorder=recorder)
                if low.tick != monitor.tick:
                    raise ValueError('low-level and capture ticks disagree')
        finally:
            energy = recorder.finish()
            with (out/'energy.npz').open('xb') as data:
                np.savez_compressed(data, **{k:v for k,v in energy.items() if isinstance(v,np.ndarray)})
            write_new(out/'energy_metadata.json',
                      {k:v for k,v in energy.items() if not isinstance(v,np.ndarray)})
            write_new(out/'completed_q.json', monitor.snapshot())
            write_new(out/'commands.json', low.evidence)
    # A fall is a retained terminal measurement, not a successful data package.
    if monitor.labels[0] == 'fall':
        raise ValueError('actual fall: retain first failure and stop allocation')
    if energy['problems'] or energy['steps'] != TOTAL_TICKS:
        raise ValueError('incomplete/faulted actual energy capture')
    if not low.evidence or low.evidence[-1]['tick_after'] != TOTAL_TICKS:
        raise ValueError('missing final actual command proof')
    return {'ticks':monitor.tick, 'warmup_J':float(energy['energy_step_j'][:WARMUP_TICKS,0].sum()),
            'measurement_J':float(energy['energy_step_j'][WARMUP_TICKS:,0].sum()),
            'total_J':float(energy['total_j'][0]), 'completed_q_count':len(monitor.completed),
            'diagnostic_only':True, 'physical_navigation_success':False}
