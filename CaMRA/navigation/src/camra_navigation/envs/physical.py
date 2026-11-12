"""Online closed-loop navigation driven by the *actual* simulator state (NAV-042).

Structure of one navigation step (0.1 s):
``true pose`` -> ``obs-v3`` -> deterministic policy -> ``[vx, vy, wz]`` ->
clip once -> the applied command is written into the low-level command store ->
**5 policy steps of 0.02 s** (each: re-observe -> re-infer -> apply -> 4 physics
substeps of 0.005 s) -> ``true pose`` again.

Three things this module keeps apart on purpose:

1. **Internal budget vs measured energy.**  ``z`` is charged every nav step with
   ``(1+alpha) * c_nom`` from the frozen power model; the measured mechanical
   work arrives from the 200 Hz recorder and is stored beside it, never mixed.
   "over the internal budget" (``budget_z_J < 0``) and "measured work over z0"
   (BVR) are different facts and are recorded separately.  Neither ends the
   episode: since NAV-062 ``z <= 0`` is a reported fact, not a terminal.
2. **Actual state vs nominal reference.**  The reference is a twin integrated
   from the *initial* pose with the applied command sequence.  It is never
   re-anchored to the actual pose, so ``deviation_m`` measures real
   model-execution drift instead of being zero by construction.
3. **Nominal safety vs physical collision.**  ``danger`` is the frozen nominal
   predicate (footprint inflated by delta, history flag going to 1 and staying
   there); ``collision`` is the physical metric against the *original* geometry.
   Neither is derived from the other.

Everything environment-specific lives behind :class:`LowLevel`: the Isaac Lab
adapter in ``scripts/evaluate_navigation.py``, a stub in the unit test.  The
bookkeeping here is the part that must be identical in both.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .. import observation as obs
from ..costs import JCostModel, SignedBudget
from ..maps import geometry as G
from .nominal import (
    DT_NAV_S,
    LABELS,
    _require_frozen_integrator,
    integrate,
    wrap_angle,
)

POLICY_STEPS_PER_NAV = 5
SUBSTEPS_PER_POLICY = 4
PHYSICS_DT_S = 0.005

#: Terminal labels, frozen contract priority order (``event_table``).
#: ``fall`` and ``collision`` need real physics; the nominal model cannot
#: produce them, the physical loop can.
PHYSICAL_LABELS: tuple[str, ...] = LABELS
NO_FEASIBLE_BUDGET = "no_feasible_budget"


class PhysicalEnvError(RuntimeError):
    """Raised when the online loop is used inconsistently."""


class NoFeasibleBudget(RuntimeError):
    """The candidate has no selected budget on this map (NAV-041 record).

    Raised by :meth:`PhysicalNavEnv.step` so nothing fabricates a budget.  The
    caller records the episode as ``no_feasible_budget`` and keeps it in the
    denominator; it must not substitute a larger budget or another map.
    """


class EpisodeOverError(RuntimeError):
    """``step`` was called on a finished episode."""


class LowLevel(Protocol):
    """What the navigation layer needs from the locomotion side."""

    num_envs: int

    def reset(self) -> None: ...

    def write_commands(self, rows: Sequence[Sequence[float]]) -> None: ...

    def step_policy(self) -> dict[str, Any]:
        """One 0.02 s control step for every env.

        Returns ``{"obs_command": [[vx, vy, wz] per env], "substeps": int,
        "sim_time": float, "events": {env_id: str}, "obs_fingerprint": [...]}``.
        """

    def true_state(self, env_id: int) -> tuple[float, float, float, float, float, float]:
        """``(px, py, yaw, vx_body, vy_body, wz_body)`` from the simulator."""

    def measured_J(self) -> list[float]:
        """Cumulative measured mechanical work per env, joules (200 Hz)."""


# --------------------------------------------------------------------- config
@dataclass(frozen=True)
class PhysicalConfig:
    """Frozen thresholds of the task definition (all from the contracts)."""

    t_max_s: float = 30.0
    r_goal_m: float = 0.5           # navigation_contract_v1 event_table
    stagnation_window_s: float = 3.0
    stagnation_min_displacement_m: float = 0.05
    z_ref_J: float = 550.0          # observation.Z_REF_J (single owner)
    obstacle_capacity: int = obs.OBSTACLE_CAPACITY

    def validate(self) -> None:
        if abs(POLICY_STEPS_PER_NAV * SUBSTEPS_PER_POLICY * PHYSICS_DT_S - DT_NAV_S) > 1e-12:
            raise PhysicalEnvError(
                f"{POLICY_STEPS_PER_NAV} x {SUBSTEPS_PER_POLICY} x {PHYSICS_DT_S} "
                f"!= {DT_NAV_S}: the 10/50/200 Hz structure is not satisfied")
        if self.z_ref_J <= 0.0:
            raise PhysicalEnvError("z_ref_J must be positive")


# ---------------------------------------------------------------------- trace
@dataclass
class PolicyStepEvidence:
    """One 0.02 s policy step: proof that the command was current."""

    policy_index: int
    t_start_s: float
    t_end_s: float
    command_written: list[list[float]]
    command_seen: list[list[float]]
    obs_fingerprint: list[float]
    substeps: int

    #: The simulator's command channel is float32, so "the same command" has to
    #: be compared at float32 resolution (~1e-7 relative), not at 1e-9: a
    #: tighter bound reports representation noise as a staleness failure.
    COMMAND_MATCH_TOL = 1e-6

    @property
    def max_command_diff(self) -> float:
        diffs = [abs(a - b) for w, s in zip(self.command_written, self.command_seen)
                 for a, b in zip(w, s)]
        return max(diffs) if diffs else 0.0

    @property
    def command_current(self) -> bool:
        return self.max_command_diff <= self.COMMAND_MATCH_TOL

    def as_dict(self) -> dict:
        return {
            "policy_index": self.policy_index,
            "t_start_s": round(self.t_start_s, 6),
            "t_end_s": round(self.t_end_s, 6),
            "command_written": [[round(v, 6) for v in c] for c in self.command_written],
            "command_seen": [[round(v, 6) for v in c] for c in self.command_seen],
            "obs_fingerprint": [round(v, 6) for v in self.obs_fingerprint],
            "substeps": self.substeps,
            "command_current": self.command_current,
            "max_command_diff": self.max_command_diff,
        }


@dataclass
class NavStepRecord:
    """One 0.1 s navigation step, chained state -> budget -> action -> physics."""

    episode_id: int
    env_id: int
    candidate_id: str
    map_id: str
    nav_index: int
    t_start_s: float
    t_end_s: float
    pose_before: tuple[float, float, float]
    pose_after: tuple[float, float, float]
    state_after: tuple[float, float, float]        # (vx_body, vy_body, wz_body)
    obs_z_ratio: float
    obs_safety_y: float
    obs_time_remaining: float
    action: tuple[float, float, float]
    requested: tuple[float, float, float]
    applied: tuple[float, float, float]
    clipped: bool
    invalid: bool
    cost_model_J: float
    cost_nom_J: float
    budget_z_J: float
    budget_z_ratio: float
    measured_cum_J: float
    ref_pose: tuple[float, float, float]
    ref_cost_nom_J: float
    deviation_m: float
    danger: bool
    internal: bool
    collision: bool
    out_of_bounds: bool
    goal_distance_m: float
    safety_y: float
    z_J_before: float = 0.0
    z_ratio_before: float = 0.0
    safety_y_before: float = 0.0
    goal_distance_before: float = 0.0
    within_budget: bool = True
    policy_steps: list[PolicyStepEvidence] = field(default_factory=list)
    label: str | None = None
    done: bool = False

    @property
    def failure_mask(self) -> dict[str, bool]:
        return {
            "danger": self.danger,
            "internal_violation": self.internal,
            "collision": self.collision,
            "out_of_bounds": self.out_of_bounds,
            # kept for trace compatibility; a label is never budget_exhausted
            # any more (NAV-062), the budget fact lives in `within_budget`
            "budget_exhausted": self.budget_z_J < 0.0,
            "fall": self.label == "fall",
        }

    def as_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "candidate_id": self.candidate_id,
            "map_id": self.map_id,
            "nav_index": self.nav_index,
            "t_start_s": round(self.t_start_s, 6),
            "t_end_s": round(self.t_end_s, 6),
            "pose_before": [round(v, 6) for v in self.pose_before],
            "pose_after": [round(v, 6) for v in self.pose_after],
            "state_after": [round(v, 6) for v in self.state_after],
            "obs": {"z_ratio": round(self.obs_z_ratio, 6),
                    "safety_y": round(self.obs_safety_y, 6),
                    "time_remaining": round(self.obs_time_remaining, 6)},
            "action": [round(v, 6) for v in self.action],
            "requested": [round(v, 6) for v in self.requested],
            "applied": [round(v, 6) for v in self.applied],
            "clipped": self.clipped,
            "invalid": self.invalid,
            "cost_model_J": round(self.cost_model_J, 6),
            "cost_nom_J": round(self.cost_nom_J, 6),
            "budget": {"z_J": round(self.budget_z_J, 6),
                       "z_ratio": round(self.budget_z_ratio, 6)},
            "measured_cum_J": round(self.measured_cum_J, 6),
            "reference": {"pose": [round(v, 6) for v in self.ref_pose],
                          "cost_nom_J": round(self.ref_cost_nom_J, 6),
                          "deviation_m": round(self.deviation_m, 6)},
            "failure_mask": self.failure_mask,
            "goal_distance_m": round(self.goal_distance_m, 6),
            "safety_y": round(self.safety_y, 6),
            "t_side": {"z_J": round(self.z_J_before, 6),
                       "z_ratio": round(self.z_ratio_before, 6),
                       "safety_y": round(self.safety_y_before, 6),
                       "goal_distance_m": round(self.goal_distance_before, 6),
                       "within_budget": self.within_budget},
            "policy_steps": [p.as_dict() for p in self.policy_steps],
            "label": self.label,
            "done": self.done,
        }


@dataclass
class EpisodeTrace:
    episode_id: int
    env_id: int
    candidate_id: str
    map_id: str
    z0_J: float | None
    status: str = "running"
    label: str | None = None
    records: list[NavStepRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self, *, include_steps: bool = True) -> dict:
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "candidate_id": self.candidate_id,
            "map_id": self.map_id,
            "z0_J": None if self.z0_J is None else round(self.z0_J, 6),
            "status": self.status,
            "label": self.label,
            "obs_version": obs.OBS_VERSION,
            "z_ratio_clip": obs.z_ratio_clip(),
            "n_steps": len(self.records),
            "notes": list(self.notes),
            "steps": [r.as_dict() for r in self.records] if include_steps else [],
        }

    def summary(self) -> dict:
        last = self.records[-1] if self.records else None
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "candidate_id": self.candidate_id,
            "map_id": self.map_id,
            "z0_J": None if self.z0_J is None else round(self.z0_J, 3),
            "status": self.status,
            "label": self.label,
            "n_steps": len(self.records),
            "t_s": round(last.t_end_s, 3) if last else 0.0,
            "min_goal_distance_m": round(min((r.goal_distance_m for r in self.records),
                                             default=float("nan")), 3),
            "measured_J": round(last.measured_cum_J, 3) if last else 0.0,
            "total_cost_nom_J": round(sum(r.cost_nom_J for r in self.records), 3),
            "total_cost_model_J": round(sum(r.cost_model_J for r in self.records), 3),
            "clipped_steps": sum(1 for r in self.records if r.clipped),
            "invalid_steps": sum(1 for r in self.records if r.invalid),
            "policy_steps": sum(len(r.policy_steps) for r in self.records),
            "measured_over_nominal": (
                round(last.measured_cum_J / max(1e-9, sum(r.cost_nom_J for r in self.records)), 4)
                if last and self.records else None),
            "final_deviation_m": round(last.deviation_m, 3) if last else None,
            "danger_steps": sum(1 for r in self.records if r.danger),
            "collision_steps": sum(1 for r in self.records if r.collision),
            "max_budget_overspend_J": round(
                max((max(0.0, -r.budget_z_J) for r in self.records), default=0.0), 3),
            # NAV-062: arrival and budget are separate facts; a late arrival is
            # never a feasible arrival just because the label says goal_reached
            "arrived_within_budget": bool(
                self.label == "goal_reached" and last is not None
                and last.budget_z_J >= 0.0),
        }


# ------------------------------------------------------------------- the env
class PhysicalNavEnv:
    """One episode over ``num_envs`` lockstep environments.

    The navigation layer never touches Isaac: it reads ``true_state`` and writes
    commands.  That is what makes the loop identical for the stub and the
    simulator, and what lets the wiring be tested without a GPU.
    """

    def __init__(
        self,
        spec,
        geometry: G.SafetyGeometry,
        cost_model: JCostModel,
        low: LowLevel,
        policy,
        *,
        candidate_id: str,
        cfg: PhysicalConfig | None = None,
    ) -> None:
        self.spec = spec
        self.geometry = geometry
        self.cost_model = cost_model
        self.low = low
        self.policy = policy          # callable(list[float]) -> sequence[3]
        self.candidate_id = str(candidate_id)
        self.cfg = cfg or PhysicalConfig()
        self.cfg.validate()
        if int(low.num_envs) < 1:
            raise PhysicalEnvError("low-level adapter reports num_envs < 1")
        self.num_envs = int(low.num_envs)

        self._obstacles: tuple[Any, ...] = tuple(spec.obstacles())
        self._obstacle_obs = tuple(
            obs.obstacle_from_polygon(ob.polygon().vertices, label=f"island-{i}")
            for i, ob in enumerate(self._obstacles)
        )
        self._polys = tuple(ob.polygon() for ob in self._obstacles)
        from .nominal import wall_polygons

        self._walls = wall_polygons(spec.extent)
        self._extent = obs.MapExtent(*spec.extent)
        self._dropped = max(0, len(self._obstacle_obs) - self.cfg.obstacle_capacity)

        self._episode_counter = -1
        self.traces: list[EpisodeTrace] = []
        self._reset_state()

    # ---- lifecycle ----
    def _reset_state(self) -> None:
        n = self.num_envs
        self.done = [False] * n
        self.label: list[str | None] = [None] * n
        self.t_s = [0.0] * n
        self.index = [0] * n
        self.safety_y = [0.0] * n
        self.danger_steps = [0] * n
        self.collision_steps = [0] * n
        self._pose: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n
        self._history: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
        self._ref_pose: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n
        self._ref_cost_nom: list[float] = [0.0] * n
        self._budgets: list[SignedBudget] = []
        self._cost_models: list[JCostModel] = []
        self._trace: EpisodeTrace | None = None

    def reset(self, *, z0_J: float | None,
              seeds: Sequence[int] | None = None,
              reset_low: bool = True) -> EpisodeTrace:
        """Start a fresh episode; ``z0_J=None`` means "no feasible budget".

        ``reset_low=False`` lets the caller settle the low level first (gait
        warm-up) and then start the episode clock from the settled pose, instead
        of opening the trace on a pose the robot has already left.
        """
        self.cost_model.reset()
        if reset_low:
            self.low.reset()
        self._reset_state()
        self._episode_counter += 1
        self._trace = EpisodeTrace(
            episode_id=self._episode_counter, env_id=0,
            candidate_id=self.candidate_id, map_id=self.spec.map_id, z0_J=z0_J)

        if z0_J is None:
            # Fail loudly and traceably: a missing budget is a result, not a
            # reason to invent one (budget_comparison: no_solution keeps the
            # map in the denominator).
            self.done = [True] * self.num_envs
            self.label = [NO_FEASIBLE_BUDGET] * self.num_envs
            self._trace.status = "no_feasible_budget"
            self._trace.label = NO_FEASIBLE_BUDGET
            self._trace.notes.append(
                "z0_J is None: NAV-041 recorded no_feasible_budget for this "
                "(method, map); the episode is not simulated and no budget is "
                "substituted")
            self.traces.append(self._trace)
            return self._trace

        if not math.isfinite(z0_J) or z0_J <= 0.0:
            raise PhysicalEnvError(f"z0_J must be finite and positive, got {z0_J}")

        for i in range(self.num_envs):
            self._budgets.append(SignedBudget(z0_J=float(z0_J), z_ref_J=self.cfg.z_ref_J))
            self._cost_models.append(JCostModel(
                self.cost_model.model, self.cost_model.alpha,
                variant=self.cost_model.variant,
                alpha_source=self.cost_model.alpha_source))

        poses = []
        for i in range(self.num_envs):
            s = self.low.true_state(i)
            pose = (float(s[0]), float(s[1]), float(s[2]))
            poses.append(pose)
            self._pose[i] = pose
            self._ref_pose[i] = pose            # twin starts here, never re-anchored
            self._history[i] = [(0.0, pose[0], pose[1])]
        self._trace.notes.append(f"initial_poses={[[round(v, 6) for v in p] for p in poses]}")
        self.traces.append(self._trace)
        return self._trace

    # ---- one navigation step ----
    def step(self) -> list[NavStepRecord]:
        if self._trace is None:
            raise PhysicalEnvError("call reset() before step()")
        if any(self.done):
            raise EpisodeOverError(
                f"episode already ended with labels={self.label}; call reset()")
        if any(self.label):
            raise PhysicalEnvError("label set but done flag not raised: inconsistent state")

        n = self.num_envs
        if len(set(round(t, 9) for t in self.t_s)) != 1:
            raise PhysicalEnvError(
                f"navigation clocks diverged across envs: {self.t_s}; the "
                "lockstep structure is broken")
        out: list[NavStepRecord] = []
        actions, applied_rows = [], []
        for i in range(n):
            s = self.low.true_state(i)
            ego = obs.EgoState(px=float(s[0]), py=float(s[1]), yaw=float(s[2]),
                               vx_body=float(s[3]), vy_body=float(s[4]),
                               wz_body=float(s[5]))
            task = obs.TaskState(
                goal_x=float(self.spec.goal[0]), goal_y=float(self.spec.goal[1]),
                safety_y=self.safety_y[i], z_J=self._budgets[i].z_J,
                time_remaining_s=max(0.0, self.cfg.t_max_s - self.t_s[i]))
            vec = obs.encode_observation(ego, self._obstacle_obs, self._extent, task,
                                        z_ref_J=self.cfg.z_ref_J)
            action = tuple(float(v) for v in self.policy(vec))
            cmd = obs.command_from_action(action)
            actions.append(action)
            applied_rows.append(cmd.applied)

        self.low.write_commands(applied_rows)

        evidence: list[PolicyStepEvidence] = []
        t_cursor = self.t_s[0]
        for k in range(POLICY_STEPS_PER_NAV):
            info = self.low.step_policy()
            substeps = int(info.get("substeps", 0))
            if substeps != SUBSTEPS_PER_POLICY:
                raise PhysicalEnvError(
                    f"policy step {k} advanced {substeps} physics substeps, "
                    f"expected {SUBSTEPS_PER_POLICY}")
            seen = info.get("obs_command") or applied_rows
            evidence.append(PolicyStepEvidence(
                policy_index=k,
                t_start_s=t_cursor + k * (DT_NAV_S / POLICY_STEPS_PER_NAV),
                t_end_s=t_cursor + (k + 1) * (DT_NAV_S / POLICY_STEPS_PER_NAV),
                command_written=[list(map(float, r)) for r in applied_rows],
                command_seen=[list(map(float, r)) for r in seen],
                obs_fingerprint=[float(v) for v in (info.get("obs_fingerprint") or [])],
                substeps=substeps))
            for eid, ev in (info.get("events") or {}).items():
                if ev and not self.label[int(eid)]:
                    self.label[int(eid)] = str(ev)
        sim_t = self.low.step_time() if hasattr(self.low, "step_time") else None
        if sim_t is not None:
            expected = self.t_s[0] + DT_NAV_S
            if abs(float(sim_t) - expected) > 1e-6:
                raise PhysicalEnvError(
                    f"simulator clock {sim_t} != expected {expected}: a hidden "
                    "reset or a different substep count would break the trace")

        measured = self.low.measured_J()
        _require_frozen_integrator()

        for i in range(n):
            p0, y0, yaw0 = self._pose[i]
            s = self.low.true_state(i)
            p1, y1, yaw1 = float(s[0]), float(s[1]), float(s[2])
            # the t side, before anything is charged or updated (NAV-062)
            z_before = float(self._budgets[i].z_J)
            z_ratio_before = float(self._budgets[i].z_ratio)
            safety_before = float(self.safety_y[i])
            goal_d_before = math.hypot(float(self.spec.goal[0]) - p0,
                                       float(self.spec.goal[1]) - y0)
            cost = self._cost_models[i].step(applied_rows[i], budget=self._budgets[i])

            danger = any(
                G.nominal_violation_during_step((p0, y0), (p1, y1), poly, self.geometry)
                for poly in self._polys
            ) or G.boundary_violation((p1, y1), self.spec.extent, self.geometry)
            self.safety_y[i] = G.update_safety_history(self.safety_y[i], danger)
            if danger:
                self.danger_steps[i] += 1

            hit = any(
                G.physical_collision_during_step((p0, y0), yaw0, (p1, y1), yaw1, poly,
                                                self.geometry)
                for poly in self._polys + self._walls
            )
            if hit:
                self.collision_steps[i] += 1
            outside = not (self.spec.extent[0] <= p1 <= self.spec.extent[1]
                           and self.spec.extent[2] <= y1 <= self.spec.extent[3])

            # nominal reference: integrate the applied command from ITS OWN last
            # pose.  It never reads the actual pose -- re-anchoring would make
            # deviation_m structurally zero and hide execution error.
            ref = integrate(self._ref_pose[i][0], self._ref_pose[i][1],
                            self._ref_pose[i][2], applied_rows[i], DT_NAV_S)
            ref_yaw = float(ref.yaw)
            if wrap_angle is not None and not (-math.pi < ref_yaw <= math.pi):
                ref_yaw = float(wrap_angle(ref_yaw))
            self._ref_pose[i] = (float(ref.px), float(ref.py), ref_yaw)
            self._ref_cost_nom[i] += float(cost.cost_nom_J)
            deviation = math.hypot(p1 - self._ref_pose[i][0], y1 - self._ref_pose[i][1])

            goal_d = math.hypot(float(self.spec.goal[0]) - p1,
                                float(self.spec.goal[1]) - y1)
            self.t_s[i] += DT_NAV_S
            self._pose[i] = (p1, y1, yaw1)
            self._history[i].append((self.t_s[i], p1, y1))

            label = self.label[i]
            if label is None:
                if hit:
                    label = "collision"
                elif outside:
                    label = "out_of_bounds"
                elif goal_d <= self.cfg.r_goal_m:
                    label = "goal_reached"
                # no budget branch: z <= 0 never ends the episode (NAV-062)
                elif self._stagnated(i):
                    label = "stagnation"
                elif self.t_s[i] >= self.cfg.t_max_s - 1e-12:
                    label = "timeout"
            if label is not None:
                self.done[i] = True
                self.label[i] = label

            rec = NavStepRecord(
                episode_id=self._episode_counter, env_id=i,
                candidate_id=self.candidate_id, map_id=self.spec.map_id,
                nav_index=self.index[i],
                t_start_s=self.t_s[i] - DT_NAV_S, t_end_s=self.t_s[i],
                pose_before=(p0, y0, yaw0), pose_after=(p1, y1, yaw1),
                state_after=(float(s[3]), float(s[4]), float(s[5])),
                # tail order is [safety_y, z_ratio, time_remaining_n] in every
                # observation version so far; the version is recorded in the
                # trace header so a future reorder cannot go unnoticed
                obs_z_ratio=float(vec[-2]), obs_safety_y=float(vec[-3]),
                obs_time_remaining=float(vec[-1]),
                action=actions[i], requested=cmd.requested, applied=cmd.applied,
                clipped=cmd.clipped, invalid=cmd.invalid,
                cost_model_J=float(cost.cost_model_J),
                cost_nom_J=float(cost.cost_nom_J),
                budget_z_J=float(self._budgets[i].z_J),
                budget_z_ratio=float(self._budgets[i].z_ratio),
                measured_cum_J=float(measured[i]),
                ref_pose=self._ref_pose[i], ref_cost_nom_J=float(self._ref_cost_nom[i]),
                deviation_m=deviation,
                danger=bool(danger), internal=bool(danger),
                collision=bool(hit), out_of_bounds=bool(outside),
                goal_distance_m=goal_d, safety_y=float(self.safety_y[i]),
                z_J_before=z_before, z_ratio_before=z_ratio_before,
                safety_y_before=safety_before,
                goal_distance_before=goal_d_before,
                within_budget=bool(z_before >= 0.0),
                policy_steps=evidence, label=label, done=self.done[i])
            out.append(rec)
            self.index[i] += 1
            assert rec.episode_id == self._trace.episode_id

        assert self._trace is not None
        self._trace.records.extend(out)
        finished = [i for i in range(n) if self.done[i]]
        if finished:
            self._trace.status = "finished"
            self._trace.label = self.label[finished[0]]
        return out

    def _stagnated(self, i: int) -> bool:
        window = self.cfg.stagnation_window_s
        if self.t_s[i] < window - 1e-12:
            return False
        anchor = None
        for t, x, y in self._history[i]:
            if self.t_s[i] - t <= window + 1e-9:
                anchor = (x, y)
                break
        if anchor is None:
            return False
        return math.hypot(self._pose[i][0] - anchor[0],
                          self._pose[i][1] - anchor[1]) < self.cfg.stagnation_min_displacement_m

    # ---- evidence helpers ----
    def trace_digest(self) -> str:
        """Stable digest of the recorded trace (structure, not physics)."""
        if self._trace is None:
            return ""
        h = hashlib.sha256()
        for r in self._trace.records:
            h.update(f"{r.episode_id}|{r.nav_index}|{r.t_start_s:.6f}|"
                     f"{r.applied[0]:.9f}|{r.applied[2]:.9f}|"
                     f"{r.budget_z_J:.9f}|{r.measured_cum_J:.9f}".encode("ascii"))
        return h.hexdigest()
