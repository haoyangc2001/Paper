"""External velocity-command interface for the Go2 locomotion policy.

Contract source
---------------
``CaMRA/locomotion/docs/command_interface.json`` v1.0.0 (produced by TASK-016).

The interface lets an external caller (a navigation stack, an evaluation script,
a calibration sweep) drive the frozen or under-evaluation locomotion policy with
batched 3-D velocity commands, without touching the training code path.

Design decisions (each one is required by the TASK-016 contract)
---------------------------------------------------------------
1. **Lock the command buffer every policy step.**  ``CommandTerm.compute()`` is
   invoked once per policy step, *after* ``_reset_idx`` and *before*
   ``observation_manager.compute()``.  We wrap that call so the external value is
   written back right before the observation is assembled.  This is what makes
   "每次推理前重写命令缓冲" true in practice.
2. **Disable the internal generator instead of racing it.**  The training
   curriculum resamples every 10 s and zeroes the command for "standing" envs.
   Both would silently corrupt an external command, so the bridge pins
   ``resampling_time_range`` / ``heading_command`` / ``rel_standing_envs`` and
   clears ``is_standing_env`` for the lifetime of the bridge.
3. **Episode boundaries are reported, never hidden.**  IsaacLab auto-resets
   terminated envs inside ``step()``.  The record returned for that step carries
   ``terminated``/``truncated``/``new_episode`` so a trajectory consumer can drop
   the mixed sample instead of silently stitching two episodes together.

The training path is untouched: nothing here runs unless a bridge/wrapper is
explicitly constructed, so ``train.py`` keeps using the internal curriculum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import gymnasium as gym
import torch

__all__ = [
    "COMMAND_DIM",
    "DEFAULT_LIMITS",
    "clip_to_limits",
    "CommandProgram",
    "FixedProgram",
    "StepProgram",
    "RampProgram",
    "SineProgram",
    "ComboProgram",
    "PROGRAM_REGISTRY",
    "build_program",
    "CommandStepRecord",
    "ExternalCommandBridge",
    "ExternalCommandWrapper",
]

#: Number of command channels: [vx, vy, wz].
COMMAND_DIM = 3

#: Command limits from the contract (vx, vy, wz) -- the code values kept per the
#: user decision recorded on 2026-10-31 (TASK-016 work record).
DEFAULT_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),
    (-0.4, 0.4),
    (-1.0, 1.0),
)


# ---------------------------------------------------------------------------
# Command programs
# ---------------------------------------------------------------------------


class CommandProgram:
    """Base class for time-parameterised command programs.

    ``at(t)`` receives a 1-D tensor of per-environment times (seconds) and
    returns a ``(num_envs, 3)`` tensor of requested commands.
    """

    name: str = "base"

    def at(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    # convenience so a program is directly callable
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.at(t)


def _as_vel(value: Sequence[float] | torch.Tensor) -> torch.Tensor:
    vel = torch.as_tensor(value, dtype=torch.float32).flatten()
    if vel.numel() != COMMAND_DIM:
        raise ValueError(f"velocity must have {COMMAND_DIM} components, got {vel.numel()}")
    return vel


class FixedProgram(CommandProgram):
    """Constant command for the whole episode."""

    name = "fixed"

    def __init__(self, vel: Sequence[float]):
        self.vel = _as_vel(vel)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        return self.vel.to(t.device).unsqueeze(0).expand(t.shape[0], COMMAND_DIM).contiguous()

    def describe(self) -> str:
        v = ", ".join(f"{float(x):+.3f}" for x in self.vel)
        return f"fixed[{v}]"


class StepProgram(CommandProgram):
    """Piecewise-constant command.

    ``segments`` is a list of ``{"duration": seconds, "vel": [vx, vy, wz]}``.
    The final segment holds forever, so the program is always well defined.
    """

    name = "step"

    def __init__(self, segments: Iterable[dict[str, Any]]):
        segs = list(segments)
        if not segs:
            raise ValueError("StepProgram needs at least one segment")
        self.durations: list[float] = []
        self.vels: list[torch.Tensor] = []
        for seg in segs:
            self.durations.append(float(seg["duration"]))
            self.vels.append(_as_vel(seg["vel"]))
        # cumulative end time of every segment
        ends, acc = [], 0.0
        for d in self.durations:
            acc += d
            ends.append(acc)
        self._ends = ends
        self._ends_t = torch.tensor(ends, dtype=torch.float32)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        ends = self._ends_t.to(t.device)
        # right=True: a sample exactly on a segment boundary belongs to the NEXT
        # segment (segments are [start, end)); the default would delay every
        # switch by one policy step.
        idx = torch.bucketize(t.contiguous(), ends, right=True)
        idx = idx.clamp(max=len(self.vels) - 1)
        stacked = torch.stack([v.to(t.device) for v in self.vels], dim=0)  # (S, 3)
        return stacked[idx]

    def describe(self) -> str:
        parts = [
            f"{d:.2f}s->[{float(v[0]):+.2f},{float(v[1]):+.2f},{float(v[2]):+.2f}]"
            for d, v in zip(self.durations, self.vels)
        ]
        return "step[" + "; ".join(parts) + "]"


class RampProgram(CommandProgram):
    """Linear ramp between two commands, optionally returning to the start."""

    name = "ramp"

    def __init__(
        self,
        start: Sequence[float],
        end: Sequence[float],
        duration: float,
        return_to_start: bool = False,
        hold: float | None = None,
    ):
        self.start = _as_vel(start)
        self.end = _as_vel(end)
        self.duration = float(duration)
        if self.duration <= 0.0:
            raise ValueError("duration must be > 0")
        self.return_to_start = bool(return_to_start)
        self.hold = float(hold) if hold is not None else 0.0

    def at(self, t: torch.Tensor) -> torch.Tensor:
        start = self.start.to(t.device)
        end = self.end.to(t.device)
        d = self.duration
        if not self.return_to_start:
            alpha = (t / d).clamp(0.0, 1.0).unsqueeze(1)
            return start + (end - start) * alpha
        # up then (after hold) back down, total period 2*d + hold
        period = 2.0 * d + self.hold
        local = torch.remainder(t, period).unsqueeze(1)
        up = (local / d).clamp(0.0, 1.0)
        down_start = d + self.hold
        down = ((local - down_start) / d).clamp(0.0, 1.0)
        return torch.where(
            local <= d,
            start + (end - start) * up,
            torch.where(local <= down_start, end.expand_as(local), end + (start - end) * down),
        )

    def describe(self) -> str:
        s = ",".join(f"{float(x):+.2f}" for x in self.start)
        e = ",".join(f"{float(x):+.2f}" for x in self.end)
        return f"ramp[{s}->{e} over {self.duration:.2f}s, return={self.return_to_start}]"


class SineProgram(CommandProgram):
    """Continuous sinusoidal sweep on one channel (the others stay at ``base``)."""

    name = "sine"

    def __init__(
        self,
        axis: int,
        amplitude: float,
        frequency: float,
        base: Sequence[float] = (0.0, 0.0, 0.0),
        phase: float = 0.0,
    ):
        if not 0 <= axis < COMMAND_DIM:
            raise ValueError(f"axis must be in [0, {COMMAND_DIM})")
        self.axis = int(axis)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.base = _as_vel(base)
        self.phase = float(phase)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        base = self.base.to(t.device).unsqueeze(0).expand(t.shape[0], COMMAND_DIM).clone()
        wave = torch.sin(2.0 * math.pi * self.frequency * t + self.phase)
        base[:, self.axis] = base[:, self.axis] + self.amplitude * wave
        return base

    def describe(self) -> str:
        return (
            f"sine[axis={self.axis} amp={self.amplitude:+.2f} "
            f"f={self.frequency:.2f}Hz base={list(self.base.tolist())}]"
        )


class ComboProgram(CommandProgram):
    """Sequence of sub-programs, each active for its own duration."""

    name = "combo"

    def __init__(self, segments: Iterable[dict[str, Any]]):
        segs = list(segments)
        if not segs:
            raise ValueError("ComboProgram needs at least one segment")
        self.segments: list[tuple[float, CommandProgram]] = []
        for seg in segs:
            program = seg.get("program")
            if program is None:
                program = build_program(seg["type"], **seg.get("kwargs", {}))
            self.segments.append((float(seg["duration"]), program))
        ends, acc = [], 0.0
        for d, _ in self.segments:
            acc += d
            ends.append(acc)
        self._ends_t = torch.tensor(ends, dtype=torch.float32)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        ends = self._ends_t.to(t.device)
        idx = torch.bucketize(t.contiguous(), ends, right=True).clamp(max=len(self.segments) - 1)
        out = torch.zeros(t.shape[0], COMMAND_DIM, device=t.device)
        for i, (_, program) in enumerate(self.segments):
            mask = idx == i
            if mask.any():
                out[mask] = program.at(t[mask])
        return out

    def describe(self) -> str:
        parts = [f"{d:.2f}s:{p.describe()}" for d, p in self.segments]
        return "combo[" + " | ".join(parts) + "]"


PROGRAM_REGISTRY: dict[str, type[CommandProgram]] = {
    "fixed": FixedProgram,
    "step": StepProgram,
    "ramp": RampProgram,
    "sine": SineProgram,
    "combo": ComboProgram,
}


def build_program(kind: str, **kwargs: Any) -> CommandProgram:
    """Instantiate a program by name."""
    try:
        cls = PROGRAM_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"unknown program '{kind}', available: {sorted(PROGRAM_REGISTRY)}") from exc
    return cls(**kwargs)


def clip_to_limits(cmd: torch.Tensor, limits: Sequence[Sequence[float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip ``(N, 3)`` commands to the contract box.

    Returns ``(clipped, was_clipped, was_invalid)``. Non-finite rows are replaced
    by zeros and flagged; no other post-processing is applied here so the caller
    keeps full control of the command buffer.
    """
    cmd = torch.as_tensor(cmd, dtype=torch.float32)
    if cmd.dim() == 1:
        cmd = cmd.unsqueeze(0)
    if cmd.shape[-1] != COMMAND_DIM:
        raise ValueError(f"command must have {COMMAND_DIM} components, got {cmd.shape[-1]}")
    lim = torch.as_tensor(limits, dtype=torch.float32, device=cmd.device)
    if lim.shape != (COMMAND_DIM, 2):
        raise ValueError(f"limits must be shaped ({COMMAND_DIM}, 2), got {tuple(lim.shape)}")
    invalid = ~torch.isfinite(cmd).all(dim=1)
    safe = torch.where(invalid.unsqueeze(1), torch.zeros_like(cmd), cmd)
    clipped = torch.max(torch.min(safe, lim[:, 1].unsqueeze(0)), lim[:, 0].unsqueeze(0))
    changed = (clipped != safe).any(dim=1) & ~invalid
    return clipped, changed, invalid


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------


@dataclass
class CommandStepRecord:
    """What was actually applied during one ``step()`` call.

    All fields are per-environment tensors of length ``num_envs``.
    """

    sim_time: torch.Tensor  # (N,) seconds, global sim clock after the step
    command_time: torch.Tensor  # (N,) seconds, clock the program was evaluated at
    step_index: torch.Tensor  # (N,) steps inside the current episode
    episode_index: torch.Tensor  # (N,) monotonic per-env episode counter
    requested: torch.Tensor  # (N, 3) what the caller asked for
    applied: torch.Tensor  # (N, 3) what the command buffer held
    clipped: torch.Tensor  # (N,) bool: request exceeded the contract limits
    invalid: torch.Tensor  # (N,) bool: non-finite request, replaced by zeros
    terminated: torch.Tensor  # (N,) bool: failure termination
    truncated: torch.Tensor  # (N,) bool: time-out
    new_episode: torch.Tensor  # (N,) bool: env auto-reset inside this step

    def as_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time.detach().cpu().tolist(),
            "command_time": self.command_time.detach().cpu().tolist(),
            "step_index": self.step_index.detach().cpu().tolist(),
            "episode_index": self.episode_index.detach().cpu().tolist(),
            "requested": self.requested.detach().cpu().tolist(),
            "applied": self.applied.detach().cpu().tolist(),
            "clipped": self.clipped.detach().cpu().tolist(),
            "invalid": self.invalid.detach().cpu().tolist(),
            "terminated": self.terminated.detach().cpu().tolist(),
            "truncated": self.truncated.detach().cpu().tolist(),
            "new_episode": self.new_episode.detach().cpu().tolist(),
        }

    def trajectory_valid_mask(self) -> torch.Tensor:
        """Envs whose returned observation still belongs to the current episode.

        Envs that auto-reset inside the step are excluded: their observation is
        the first frame of a brand-new trajectory, so mixing it into the old
        one would silently corrupt any trajectory-level metric.
        """
        return ~self.new_episode


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class ExternalCommandBridge:
    """Locks a velocity-command term to externally supplied values."""

    def __init__(
        self,
        env: Any,
        term_name: str = "base_velocity",
        limits: Sequence[Sequence[float]] = DEFAULT_LIMITS,
        enforce_limits: bool = True,
        on_reset: str = "zero",
        restore_on_close: bool = True,
        time_source: str = "episode",
    ):
        if on_reset not in ("zero", "hold"):
            raise ValueError("on_reset must be 'zero' or 'hold'")
        if time_source not in ("episode", "global"):
            raise ValueError("time_source must be 'episode' or 'global'")

        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.term_name = term_name
        self.term = self._env.command_manager.get_term(term_name)
        self.num_envs = int(self.term.num_envs)
        self.device = self.term.device
        self.enforce_limits = bool(enforce_limits)
        self.on_reset = on_reset
        self.restore_on_close = bool(restore_on_close)
        self.time_source = time_source

        lim = torch.as_tensor(limits, dtype=torch.float32, device=self.device)
        if lim.shape != (COMMAND_DIM, 2):
            raise ValueError(f"limits must be shaped ({COMMAND_DIM}, 2), got {tuple(lim.shape)}")
        self.limits = lim

        self._requested = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._pending = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_episode_len: torch.Tensor | None = None

        self._saved_cfg: dict[str, Any] = {}
        self._orig_term_compute = None
        self._orig_resample = None
        self._installed = False

        # counters exposed for verification
        self.stats = {
            "install_count": 0,
            "apply_count": 0,
            "clipped_count": 0,
            "invalid_count": 0,
            "auto_reset_count": 0,
            "internal_resample_count": 0,
        }
        self._install()

    # -- installation -----------------------------------------------------

    def _install(self) -> None:
        if self._installed:
            return
        cfg = self.term.cfg
        self._saved_cfg = {
            "resampling_time_range": getattr(cfg, "resampling_time_range", None),
            "rel_standing_envs": getattr(cfg, "rel_standing_envs", None),
            "rel_heading_envs": getattr(cfg, "rel_heading_envs", None),
            "heading_command": getattr(cfg, "heading_command", None),
        }
        # 1. stop the curriculum from ever resampling
        cfg.resampling_time_range = (1.0e9, 1.0e9)
        # 2. stop standing/heading post-processing from rewriting the command
        if hasattr(cfg, "rel_standing_envs"):
            cfg.rel_standing_envs = 0.0
        if hasattr(cfg, "rel_heading_envs"):
            cfg.rel_heading_envs = 0.0
        if hasattr(cfg, "heading_command"):
            cfg.heading_command = False

        self.term.is_standing_env.zero_()
        self.term.is_heading_env.zero_()
        if hasattr(self.term, "time_left"):
            self.term.time_left.fill_(1.0e9)

        # 3. apply the external value right after each internal compute, i.e.
        #    before observation_manager.compute() assembles this step's obs.
        orig_compute = self.term.compute
        bridge = self

        def locked_compute(dt):
            orig_compute(dt)
            # An env that IsaacLab just auto-reset inside step() has its
            # episode_length_buf back at 0. Clear its command *here*, i.e.
            # before observation_manager.compute() assembles this step's obs,
            # so a fresh episode never starts on the previous command.
            bridge._handle_internal_reset()
            bridge._apply_locked()

        self._orig_term_compute = orig_compute
        self.term.compute = locked_compute

        # count any internal resample that slips through (should stay at zero
        # while the bridge owns the command buffer)
        orig_resample = getattr(self.term, "_resample", None)
        if orig_resample is not None:

            def counting_resample(env_ids):
                bridge.stats["internal_resample_count"] += int(len(env_ids))
                return orig_resample(env_ids)

            self._orig_resample = orig_resample
            self.term._resample = counting_resample

        self._installed = True
        self.stats["install_count"] += 1

    def close(self) -> None:
        """Restore the command term to its original (training) behaviour."""
        if not self._installed or not self.restore_on_close:
            return
        if self._orig_term_compute is not None:
            self.term.compute = self._orig_term_compute
        if self._orig_resample is not None:
            self.term._resample = self._orig_resample
        cfg = self.term.cfg
        for key, value in self._saved_cfg.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        # re-arm the resampling timer so the curriculum picks up where the
        # restored configuration expects it to
        if hasattr(self.term, "time_left"):
            lo, hi = cfg.resampling_time_range
            self.term.time_left.uniform_(float(lo), float(hi))
        self._installed = False

    # -- command setting --------------------------------------------------

    def set_command(self, vel: Sequence[float] | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set one command for all envs (or for ``env_ids`` only)."""
        v = torch.as_tensor(vel, dtype=torch.float32, device=self.device).flatten()
        if v.numel() != COMMAND_DIM:
            raise ValueError(f"velocity must have {COMMAND_DIM} components")
        if env_ids is None:
            self._requested[:] = v
            self._pending[:] = v
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self._requested[ids] = v
            self._pending[ids] = v
        self._apply_locked()

    def set_command_batch(self, vel: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set per-environment commands; ``vel`` is shaped ``(N, 3)`` or ``(N_env, 3)``."""
        v = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(self.num_envs, COMMAND_DIM)
        if v.shape[-1] != COMMAND_DIM:
            raise ValueError(f"velocity must have {COMMAND_DIM} components, got {v.shape}")
        if env_ids is None:
            if v.shape[0] != self.num_envs:
                raise ValueError(f"expected {self.num_envs} rows, got {v.shape[0]}")
            self._requested[:] = v
            self._pending[:] = v
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            sub = v if v.shape[0] == ids.numel() else v[ids]
            self._requested[ids] = sub
            self._pending[ids] = sub
        self._apply_locked()

    def zero_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._pending.zero_()
            self._requested.zero_()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self._pending[ids] = 0.0
            self._requested[ids] = 0.0
        self._apply_locked()

    # -- internal ---------------------------------------------------------

    def _handle_internal_reset(self) -> None:
        """Detect envs auto-reset by the env's own step and clear their command.

        ``episode_length_buf`` is incremented before the command term is
        computed and zeroed by ``_reset_idx`` for envs that terminated, so
        ``buf == 0 while previous buf != 0`` is exactly "this env started a new
        episode inside the current step".
        """
        buf = getattr(self._env, "episode_length_buf", None)
        if buf is None:
            return
        if self._prev_episode_len is None:
            self._prev_episode_len = buf.clone()
            return
        just_reset = (buf == 0) & (self._prev_episode_len != 0)
        self._prev_episode_len = buf.clone()
        if self.on_reset == "zero" and bool(just_reset.any()):
            self._pending[just_reset] = 0.0
            self._requested[just_reset] = 0.0

    def _apply_locked(self) -> None:
        """Write the pending command into the term's buffer (post-process safe)."""
        with torch.no_grad():
            limits = self.limits if self.enforce_limits else ((-1e9, 1e9),) * COMMAND_DIM
            clipped_cmd, clipped, invalid = clip_to_limits(self._pending, limits)
            if bool(invalid.any()):
                self.stats["invalid_count"] += int(invalid.sum().item())
            if bool(clipped.any()):
                self.stats["clipped_count"] += int(clipped.sum().item())
            self.term.vel_command_b.copy_(clipped_cmd.to(self.device))
            self.term.is_standing_env.zero_()
            self.stats["apply_count"] += 1

    # -- episode bookkeeping ---------------------------------------------

    def notify_reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Bookkeeping for a full ``env.reset()`` (all envs restart)."""
        # re-sync the reset detector: every buffer is 0 again, so the next
        # compute must not be mistaken for an auto-reset
        self._prev_episode_len = None
        if env_ids is None:
            self._episode_index += 1
            self._step_index.zero_()
            if self.on_reset == "zero":
                self.zero_command()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self._episode_index[ids] += 1
            self._step_index[ids] = 0
            if self.on_reset == "zero":
                self.zero_command(ids)

    def notify_step(self, terminated: torch.Tensor, truncated: torch.Tensor) -> CommandStepRecord:
        """Advance episode counters and build the per-step record."""
        # clone: the env reuses its termination buffers every step, so without a copy a
        # record handed to the caller would silently change on the next step()
        term = torch.as_tensor(terminated, dtype=torch.bool, device=self.device).flatten().clone()
        trunc = torch.as_tensor(truncated, dtype=torch.bool, device=self.device).flatten().clone()

        requested = self._requested.clone()
        applied = self.term.vel_command_b.clone()
        invalid = ~torch.isfinite(requested).all(dim=1)
        if self.enforce_limits:
            lo = self.limits[:, 0].unsqueeze(0)
            hi = self.limits[:, 1].unsqueeze(0)
            expect = torch.max(torch.min(torch.where(invalid.unsqueeze(1), torch.zeros_like(requested), requested), hi), lo)
        else:
            expect = torch.where(invalid.unsqueeze(1), torch.zeros_like(requested), requested)
        clipped = (expect != torch.where(invalid.unsqueeze(1), torch.zeros_like(requested), requested)).any(dim=1)

        sim_time = self._global_time()
        record = CommandStepRecord(
            sim_time=sim_time,
            command_time=self.command_time(),
            step_index=self._step_index.clone(),
            episode_index=self._episode_index.clone(),
            requested=requested,
            applied=applied,
            clipped=clipped,
            invalid=invalid,
            terminated=term,
            truncated=trunc,
            new_episode=term | trunc,
        )

        self._step_index += 1
        done = term | trunc
        n_done = int(done.sum().item())
        if n_done:
            self.stats["auto_reset_count"] += n_done
            self._episode_index[done] += 1
            self._step_index[done] = 0
            if self.on_reset == "zero":
                # IsaacLab already reset these envs inside step(); a fresh
                # episode must not inherit the previous command.
                self.zero_command(done)
        return record

    def _global_time(self) -> torch.Tensor:
        env = self._env
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        counter = int(getattr(env, "common_step_counter", 0))
        return torch.full((self.num_envs,), counter * step_dt, device=self.device)

    def _episode_time(self) -> torch.Tensor:
        """Per-env time inside the current episode (resets to 0 on reset)."""
        buf = getattr(self._env, "episode_length_buf", None)
        if buf is None:
            return self._global_time()
        step_dt = float(getattr(self._env, "step_dt", 0.0) or 0.0)
        return buf.to(self.device).float() * step_dt

    def command_time(self) -> torch.Tensor:
        """Clock handed to programs; see ``time_source``."""
        if self.time_source == "global":
            return self._global_time()
        return self._episode_time()

    # kept for callers written against the earlier name
    def _current_sim_time(self) -> torch.Tensor:
        return self._global_time()

    # -- introspection ----------------------------------------------------

    def describe(self) -> str:
        cfg = self.term.cfg
        return (
            f"term={self.term_name} num_envs={self.num_envs} "
            f"resampling_time_range={cfg.resampling_time_range} "
            f"rel_standing_envs={getattr(cfg, 'rel_standing_envs', None)} "
            f"heading_command={getattr(cfg, 'heading_command', None)} "
            f"limits={self.limits.tolist()} enforce_limits={self.enforce_limits} "
            f"on_reset={self.on_reset}"
        )


# ---------------------------------------------------------------------------
# Gym wrapper
# ---------------------------------------------------------------------------


class ExternalCommandWrapper(gym.Wrapper):
    """Drives the environment from an external command program.

    Usage::

        env = gym.make(task, cfg=env_cfg)
        env = ExternalCommandWrapper(env, on_reset="zero")
        env.set_program(FixedProgram([0.8, 0.0, 0.0]))
        obs, _ = env.reset()
        for _ in range(N):
            obs, rew, term, trunc, info = env.step(action)
            rec = info["external_command"]  # CommandStepRecord
    """

    def __init__(
        self,
        env: gym.Env,
        bridge: ExternalCommandBridge | None = None,
        term_name: str = "base_velocity",
        program: CommandProgram | None = None,
        **bridge_kwargs: Any,
    ):
        super().__init__(env)
        self.bridge = bridge if bridge is not None else ExternalCommandBridge(env, term_name=term_name, **bridge_kwargs)
        self.program: CommandProgram | None = None
        self._last_record: CommandStepRecord | None = None
        if program is not None:
            self.set_program(program)

    # -- program management ----------------------------------------------

    def set_program(self, program: CommandProgram) -> "ExternalCommandWrapper":
        self.program = program
        self.bridge.set_command_batch(program.at(self._time_tensor()))
        return self

    def set_command(self, vel: Sequence[float]) -> None:
        self.bridge.set_command(vel)

    def _time_tensor(self) -> torch.Tensor:
        return self.bridge.command_time()

    # -- gym API ---------------------------------------------------------

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.bridge.notify_reset()
        if self.program is not None:
            self.bridge.set_command_batch(self.program.at(self._time_tensor()))
        return out

    def step(self, action):
        # rewrite before running so the request is up to date even when no
        # program is installed (a caller may push commands manually)
        if self.program is not None:
            self.bridge.set_command_batch(self.program.at(self._time_tensor()))
        obs, reward, terminated, truncated, info = self.env.step(action)
        record = self.bridge.notify_step(terminated, truncated)
        info = dict(info) if isinstance(info, dict) else {}
        info["external_command"] = record
        self._last_record = record
        return obs, reward, terminated, truncated, info

    def close(self):
        self.bridge.close()
        return self.env.close()

    # -- helpers ---------------------------------------------------------

    @property
    def last_record(self) -> CommandStepRecord | None:
        return self._last_record
