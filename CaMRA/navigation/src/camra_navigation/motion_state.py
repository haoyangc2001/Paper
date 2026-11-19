"""Candidate completed-interval state, isolated from all production launchers.

q is the SE(2) log of an already completed 0.1-second pose increment. It is
neither endpoint body velocity nor the command. Callers must save actual warmup
endpoints; this module cannot manufacture their provenance or missing history.
No response parameters, fitting, simulator, policy or calibration are supplied.
"""
from dataclasses import dataclass
import math
from numbers import Integral

from . import observation as O
from .horizon import PROFILE, encode_observation as encode_horizon90

STATE_VERSION = 'se2-completed-interval-v1'
OBSERVATION_VERSION = 'obs-v9-t90-q'
PHYSICS_DT_S = .005
INTERVAL_TICKS = 20
INTERVAL_DT_S = .1


def _tick(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('tick must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class PoseSample:
    tick: int
    pose: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, 'tick', _tick(self.tick))
        pose = tuple(float(x) for x in self.pose)
        if len(pose) != 3 or not all(math.isfinite(x) for x in pose):
            raise ValueError('finite planar x, y, yaw required')
        object.__setattr__(self, 'pose', pose)


@dataclass(frozen=True)
class CompletedInterval:
    start: PoseSample
    end: PoseSample

    def __post_init__(self):
        if not isinstance(self.start, PoseSample) or not isinstance(self.end, PoseSample):
            raise ValueError('two stamped planar poses required')
        if self.end.tick-self.start.tick != INTERVAL_TICKS:
            raise ValueError('only a completed twenty-tick interval defines q')
        if abs(self.yaw_increment) >= math.pi/2:
            raise ValueError('ambiguous large planar rotation in one interval')

    @property
    def yaw_increment(self):
        difference = self.end.pose[2]-self.start.pose[2]
        return math.atan2(math.sin(difference), math.cos(difference))

    @property
    def q(self):
        theta = self.yaw_increment
        if abs(theta) < 1.e-6:
            a = 1.-theta**2/6.+theta**4/120.
            b = theta/2.-theta**3/24.
        else:
            a, b = math.sin(theta)/theta, 2.*math.sin(theta/2.)**2/theta
        x0, y0, yaw0 = self.start.pose
        dx, dy = self.end.pose[0]-x0, self.end.pose[1]-y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        bx, by = c*dx+s*dy, -s*dx+c*dy
        denominator = INTERVAL_DT_S*(a*a+b*b)
        return ((a*bx+b*by)/denominator, (-b*bx+a*by)/denominator,
                theta/INTERVAL_DT_S)

    def as_dict(self):
        return {'state_version': STATE_VERSION, 'start_tick': self.start.tick,
                'end_tick': self.end.tick, 'start_pose': list(self.start.pose),
                'end_pose': list(self.end.pose), 'q_vx_vy_yaw_rate': list(self.q),
                'duration_s': INTERVAL_DT_S}


class CompletedMotionTracker:
    """Consume each future 200Hz sample; expose q only at completed decisions.

    Warmup must supply its final two endpoints twenty actual ticks apart.
    A terminal partial interval is retained as a tail, never padded into q.
    An episode gets a fresh tracker; a closed tracker cannot be advanced/reset.
    """
    def __init__(self, warmup_start, warmup_end):
        self._interval = CompletedInterval(warmup_start, warmup_end)
        self.window_start_tick = warmup_end.tick
        self._latest = warmup_end
        self._closed = False

    @property
    def partial_ticks(self):
        return self._latest.tick-self._interval.end.tick

    @property
    def latest(self):
        return self._latest

    @property
    def last_completed(self):
        return self._interval

    @property
    def closed(self):
        return self._closed

    def advance(self, sample, *, terminal=False):
        if self._closed:
            raise ValueError('terminated tracker cannot absorb another episode')
        if not isinstance(sample, PoseSample) or sample.tick != self._latest.tick+1:
            raise ValueError('missing, duplicate or out-of-order physics sample')
        elapsed = sample.tick-self.window_start_tick
        if elapsed > PROFILE.step_cap(PHYSICS_DT_S):
            raise ValueError('sample exceeds the candidate ninety-second window')
        interval = self._interval
        if sample.tick-interval.end.tick == INTERVAL_TICKS:
            interval = CompletedInterval(interval.end, sample)
        self._latest, self._interval = sample, interval
        self._closed = bool(terminal)

    def for_decision(self):
        if (self._closed or self.partial_ticks or self._latest.tick-self.window_start_tick
                >= PROFILE.step_cap(PHYSICS_DT_S)):
            raise ValueError('decision requires a live completed interval')
        return self._interval


def encode_completed_observation(interval, *, decision_tick, window_start_tick,
                                 obstacles, extent, task, inflation_radius_m):
    """Candidate v9: retain v7 geometry/budget, encode completed q and t/90."""
    decision_tick, window_start_tick = _tick(decision_tick), _tick(window_start_tick)
    if not isinstance(interval, CompletedInterval) or interval.end.tick != decision_tick:
        raise ValueError('q is stale or uses a future endpoint')
    if window_start_tick < INTERVAL_TICKS or decision_tick < window_start_tick:
        raise ValueError('warmup endpoints and measurement origin are required')
    elapsed = (decision_tick-window_start_tick)*PHYSICS_DT_S
    if ((decision_tick-window_start_tick) % INTERVAL_TICKS
            or elapsed >= PROFILE.t_max_s):
        raise ValueError('policy decisions require a live navigation boundary')
    remaining = PROFILE.remaining(elapsed)
    if not math.isfinite(task.time_remaining_s) or not math.isclose(
            task.time_remaining_s, remaining, rel_tol=0., abs_tol=1.e-8):
        raise ValueError('observation clock disagrees with actual decision tick')
    ego = O.EgoState(*interval.end.pose, *interval.q)
    return encode_horizon90(ego, obstacles, extent, task,
                            inflation_radius_m=inflation_radius_m)


def require_model_binding(binding, *, motion_model_sha256):
    """Identity compatibility only; this does not validate the named artifact."""
    if (not isinstance(motion_model_sha256, str) or len(motion_model_sha256) != 64
            or any(c not in '0123456789abcdef' for c in motion_model_sha256)):
        raise ValueError('an explicit frozen motion model identity is required')
    expected = {'time_profile': PROFILE.profile_id, 't_max_s': PROFILE.t_max_s,
                'observation_version': OBSERVATION_VERSION,
                'motion_state_version': STATE_VERSION,
                'motion_model_sha256': motion_model_sha256}
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError('candidate state/model/time binding mismatch')
