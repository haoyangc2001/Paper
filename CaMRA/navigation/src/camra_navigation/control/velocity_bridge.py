"""10 / 50 / 200 Hz velocity execution bridge (NAV-025).

One navigation step is 0.1 s.  Inside it:

* the upper layer's requested command is clipped **once** and becomes the applied
  command for the whole interval;
* the applied command is written into the low-level command store, then the
  low-level observations are recomputed, then the frozen policy infers, then the
  joint targets are applied — and this sequence repeats for **5 policy steps** of
  0.02 s each;
* each policy step contains **4 physics substeps** of 0.005 s, so 20 substeps and
  exactly 0.1 s of simulated time pass per navigation step.

Two rules this module enforces structurally:

1. **the command is re-written before every observation recomputation.**  A low
   level that owns its command term (resampling, programs) would otherwise
   overwrite the upper layer's command between two policy steps — the HiRA
   failure recorded in NAV-005.  The bridge therefore writes at 50 Hz, not 10 Hz.
2. **the same applied command feeds everything.**  Motion, the frozen policy's
   observation, the cost model and the logs all read the same array; nothing is
   allowed to re-derive an applied command from the requested one.

The bridge never installs a ``TableProgram`` or any other self-resampling
command source: it owns the command store for the whole episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

DT_NAV_S = 0.1
STEP_DT_S = 0.02
SIM_DT_S = 0.005
DECIMATION = 4
POLICY_STEPS_PER_NAV = 5
COMMAND_MAX = (1.0, 0.4, 1.0)


class BridgeError(RuntimeError):
    """Raised when the bridge is used against an incompatible low level."""


@dataclass(frozen=True)
class CommandRecord:
    """One 10 Hz command application."""

    env_id: int
    nav_index: int
    t_s: float
    requested: tuple[float, float, float]
    applied: tuple[float, float, float]
    clipped: bool
    invalid: bool

    def as_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "nav_index": self.nav_index,
            "t_s": self.t_s,
            "requested": list(self.requested),
            "applied": list(self.applied),
            "clipped": self.clipped,
            "invalid": self.invalid,
        }


@dataclass
class PolicyStepTrace:
    """One 0.02 s policy step, with the evidence that the command was current."""

    nav_index: int
    policy_index: int
    t_start_s: float
    t_end_s: float
    command_written: list[tuple[float, float, float]]
    command_seen: list[tuple[float, float, float]]
    q_targets_head: list[float]
    substeps: int

    def as_dict(self) -> dict:
        return {
            "nav_index": self.nav_index,
            "policy_index": self.policy_index,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "command_written": [list(c) for c in self.command_written],
            "command_seen": [list(c) for c in self.command_seen],
            "q_targets_head": list(self.q_targets_head),
            "substeps": self.substeps,
        }


@dataclass
class NavStepTrace:
    """Everything one 0.1 s navigation step did."""

    nav_index: int
    t_start_s: float
    t_end_s: float
    requested: list[tuple[float, float, float]]
    applied: list[tuple[float, float, float]]
    clipped: list[bool]
    invalid: list[bool]
    policy_steps: list[PolicyStepTrace] = field(default_factory=list)

    @property
    def n_policy_steps(self) -> int:
        return len(self.policy_steps)

    @property
    def n_physics_substeps(self) -> int:
        return sum(p.substeps for p in self.policy_steps)

    @property
    def sim_time_advanced_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def command_current_everywhere(self) -> bool:
        """Every policy step saw exactly the applied command of this interval."""
        return all(
            p.command_seen == p.command_written for p in self.policy_steps
        )

    def as_dict(self) -> dict:
        return {
            "nav_index": self.nav_index,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "requested": [list(r) for r in self.requested],
            "applied": [list(a) for a in self.applied],
            "clipped": list(self.clipped),
            "invalid": list(self.invalid),
            "n_policy_steps": self.n_policy_steps,
            "n_physics_substeps": self.n_physics_substeps,
            "sim_time_advanced_s": self.sim_time_advanced_s,
            "command_current_everywhere": self.command_current_everywhere,
            "policy_steps": [p.as_dict() for p in self.policy_steps],
        }


class CommandBridge:
    """Per-environment applied-command store, written at 10 Hz, re-read at 50 Hz."""

    def __init__(
        self,
        num_envs: int,
        *,
        command_max: tuple[float, float, float] = COMMAND_MAX,
        dt_nav_s: float = DT_NAV_S,
    ) -> None:
        self.num_envs = int(num_envs)
        self.command_max = tuple(float(v) for v in command_max)
        self.dt_nav_s = float(dt_nav_s)
        self._current: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(self.num_envs)]
        self.records: list[CommandRecord] = []
        self.t_s = 0.0
        self.nav_index = 0

    def apply(self, env_id: int, requested: Sequence[float]) -> CommandRecord:
        """Clip once, store once; the stored value is the single applied command."""
        if not 0 <= env_id < self.num_envs:
            raise BridgeError(f"env_id {env_id} out of range for {self.num_envs} envs")
        req = [float(v) for v in requested]
        if len(req) != 3:
            raise BridgeError(f"command must have 3 components, got {req}")
        invalid = False
        clipped = False
        applied: list[float] = []
        for v, hi in zip(req, self.command_max):
            if not math.isfinite(v):
                v = 0.0
                invalid = True
            a = max(-hi, min(hi, v))
            if abs(a - v) > 1e-12:
                clipped = True
            applied.append(a)
        self._current[env_id] = applied
        rec = CommandRecord(
            env_id=env_id,
            nav_index=self.nav_index,
            t_s=self.t_s,
            requested=(req[0], req[1], req[2]),
            applied=(applied[0], applied[1], applied[2]),
            clipped=clipped,
            invalid=invalid,
        )
        self.records.append(rec)
        return rec

    def current(self, env_id: int) -> tuple[float, float, float]:
        if not 0 <= env_id < self.num_envs:
            raise BridgeError(f"env_id {env_id} out of range")
        c = self._current[env_id]
        return (c[0], c[1], c[2])

    def reset(self, env_id: int | None = None) -> None:
        ids = range(self.num_envs) if env_id is None else [env_id]
        for i in ids:
            if not 0 <= i < self.num_envs:
                raise BridgeError(f"env_id {i} out of range")
            self._current[i] = [0.0, 0.0, 0.0]
        if env_id is None:
            self.records = []
            self.t_s = 0.0
            self.nav_index = 0

    def advance_clock(self) -> None:
        self.t_s += self.dt_nav_s
        self.nav_index += 1


class VelocityBridge:
    """Drive a low-level environment from 10 Hz velocity commands."""

    def __init__(
        self,
        num_envs: int,
        controller: Callable[[Any], Any],
        *,
        command_max: tuple[float, float, float] = COMMAND_MAX,
        dt_nav_s: float = DT_NAV_S,
        step_dt_s: float = STEP_DT_S,
        sim_dt_s: float = SIM_DT_S,
        decimation: int = DECIMATION,
    ) -> None:
        self.num_envs = int(num_envs)
        self.controller = controller
        self.policy_steps_per_nav = int(round(dt_nav_s / step_dt_s))
        if self.policy_steps_per_nav * step_dt_s != dt_nav_s:
            raise BridgeError(
                f"nav dt {dt_nav_s} is not a multiple of the policy step {step_dt_s}"
            )
        if step_dt_s != decimation * sim_dt_s:
            raise BridgeError(
                f"policy step {step_dt_s} != decimation {decimation} x sim dt {sim_dt_s}"
            )
        self.decimation = int(decimation)
        self.sim_dt_s = float(sim_dt_s)
        self.bridge = CommandBridge(num_envs, command_max=command_max, dt_nav_s=dt_nav_s)
        self.t_s = 0.0
        self.traces: list[NavStepTrace] = []

    def reset(self) -> None:
        self.bridge.reset()
        self.t_s = 0.0
        self.traces = []

    def step_nav(self, low_env, requested_per_env: Sequence[Sequence[float]]) -> NavStepTrace:
        """One navigation step: apply at 10 Hz, infer at 50 Hz, step at 200 Hz."""
        if len(requested_per_env) != self.num_envs:
            raise BridgeError(
                f"expected {self.num_envs} commands, got {len(requested_per_env)}"
            )
        for attr in ("write_commands", "recompute_observations", "get_observations",
                     "apply_action", "physics_step", "commands_view"):
            if not hasattr(low_env, attr):
                raise BridgeError(
                    f"low-level env is missing {attr!r}; the bridge cannot verify "
                    "that a fresh command reaches the observations without it"
                )

        records = [
            self.bridge.apply(i, req) for i, req in enumerate(requested_per_env)
        ]
        applied = [list(r.applied) for r in records]
        trace = NavStepTrace(
            nav_index=self.bridge.nav_index,
            t_start_s=self.t_s,
            t_end_s=self.t_s,
            requested=[tuple(r.requested) for r in records],
            applied=[tuple(r.applied) for r in records],
            clipped=[r.clipped for r in records],
            invalid=[r.invalid for r in records],
        )

        for k in range(self.policy_steps_per_nav):
            t0 = self.t_s
            # Rule 1: the applied command is written immediately before the
            # observation recomputation, at policy rate, not once per nav step.
            low_env.write_commands(applied)
            low_env.recompute_observations()
            seen = low_env.commands_view()
            obs = low_env.get_observations()
            q = self.controller.infer(obs)
            low_env.apply_action(q)
            substeps = 0
            for _ in range(self.decimation):
                low_env.physics_step()
                self.t_s += self.sim_dt_s
                substeps += 1
            trace.policy_steps.append(
                PolicyStepTrace(
                    nav_index=trace.nav_index,
                    policy_index=k,
                    t_start_s=t0,
                    t_end_s=self.t_s,
                    command_written=[tuple(c) for c in applied],
                    command_seen=[tuple(map(float, c)) for c in seen],
                    q_targets_head=_head(q),
                    substeps=substeps,
                )
            )

        trace.t_end_s = self.t_s
        self.bridge.advance_clock()
        self.traces.append(trace)
        return trace


def _head(q: Any, n: int = 4) -> list[float]:
    """First few joint targets of env 0, for the trace."""
    try:
        import numpy as np

        arr = np.asarray(q)
        return [float(v) for v in arr.reshape(arr.shape[0], -1)[0][:n]]
    except Exception:
        try:
            return [float(v) for v in q[0][:n]]
        except Exception:
            return []


class V2VelocityController:
    """Adapt :class:`FrozenV2Policy` to the bridge's controller protocol.

    ``infer`` turns a 45-dim low-level observation batch into 12 joint targets
    via the frozen rule ``q_target = default + 0.25 * a``.  Nothing here trains
    or re-derives the policy; the adapter only marshals tensors.
    """

    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls = 0

    def infer(self, obs):
        targets, _report = self.policy.joint_targets(obs)
        self.calls += 1
        return targets.detach().cpu().numpy()
