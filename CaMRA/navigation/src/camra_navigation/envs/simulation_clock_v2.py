"""Validate actual simulator time together with its actual integer step index.

Accept only the two preregistered representations of a 0.005 s callback.
The nominal integration interval for J and completed q remains 0.005 s.
"""
import math
from numbers import Integral

NOMINAL_DT = .005
CALLBACK_DTS = (NOMINAL_DT, 0.004999999888241291)


class SimulationClock:
    def __init__(self, readback):
        self.origin_s, self.origin_index = self.unpack(readback)
        self.previous_s = self.origin_s
        self.tick = 0
        self.candidates = CALLBACK_DTS

    @staticmethod
    def unpack(readback):
        now, index = readback
        if (not isinstance(index, Integral) or isinstance(index, bool)
                or index < 0 or not math.isfinite(float(now))):
            raise ValueError('invalid actual simulator clock/index readback')
        return float(now), int(index)

    def advance(self, tick, readback):
        now, index = self.unpack(readback)
        if (not isinstance(tick, Integral) or isinstance(tick, bool)
                or tick != self.tick+1 or index != self.origin_index+tick):
            raise ValueError('duplicate/gapped/stale actual simulator step index')
        # Bound double-precision arithmetic, not an empirical physics tolerance.
        tolerance = max(1.e-12, 32*math.ulp(max(abs(now), abs(self.origin_s), 1.)))
        candidates = tuple(dt for dt in self.candidates
            if abs(now-self.origin_s-tick*dt) <= tolerance
            and abs(now-self.previous_s-dt) <= tolerance)
        if not candidates:
            raise ValueError('actual simulator time is inconsistent with registered callback dt')
        self.candidates = candidates
        self.previous_s, self.tick = now, int(tick)
        return now, index

    def snapshot(self):
        return {'schema':'actual-simulator-clock-v2',
                'actual_origin_s':self.origin_s,
                'actual_origin_step_index':self.origin_index,
                'accepted_ticks':self.tick,
                'nominal_dt_s':NOMINAL_DT,
                'surviving_callback_dt_s':list(self.candidates),
                'actual_last_s':self.previous_s,
                'roundoff_rule':'max(1e-12, 32*ulp(max(abs(now),abs(origin),1)))'}
