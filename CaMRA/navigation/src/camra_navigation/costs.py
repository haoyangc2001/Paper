"""J cost and signed-budget adapter (NAV-014).

Wraps the frozen ``PowerModel`` / ``StepCost`` instead of re-deriving them, and
adds the two things the calibration module deliberately does not provide:

1. an **explicit** ``alpha`` (the frozen ``StepCost`` defaults to 0.0, which is
   the uncalibrated baseline, not the full method);
2. a **signed** budget with a scene-fixed normalisation.  ``BudgetState.z_ratio``
   divides by this episode's ``initial_J`` and clamps at zero, which destroys the
   over-budget information the navigation task needs.  That class is not used
   here; :class:`SignedBudget` keeps ``z_J`` signed and normalises by ``Z_REF_J``.

Three ledgers are kept strictly separate, because confusing them is the easiest
way to produce a plausible but wrong result:

===========================  ==============================  =======================
ledger                        what it is                      who may read it
===========================  ==============================  =======================
``cost_nom_J``                P_nom(u_applied) * 0.1 s        budget, critic target
``cost_model_J``              (1 + alpha) * cost_nom_J        budget, critic target
``execution_J``               sum_j max(tau_j * qdot_j, 0)     reporting only
===========================  ==============================  =======================

``execution_J`` never feeds back into the internal budget.  It is measured at
200 Hz from the physics and exists to audit how far the frozen model is from the
machine; using it to reset the budget would make the calibrated ``alpha``
meaningless.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .observation import Z_REF_J

__all__ = [
    "COST_UNIT",
    "VARIANTS",
    "CostAdapterError",
    "MarginSet",
    "load_margins",
    "SignedBudget",
    "StepCostRecord",
    "JCostModel",
    "ExecutionEnergy",
    "byte_sha256",
    "normalized_cost",
]

COST_UNIT = "J"
DT_NAV_S = 0.1
DT_PHYSICS_S = 0.005
VARIANTS = ("full", "uncalibrated")


class CostAdapterError(RuntimeError):
    """Misuse of the cost/budget interface (units, variant, sign, reset)."""


def byte_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- margins
@dataclass(frozen=True)
class MarginSet:
    """The frozen execution-error margins, loaded not typed."""

    alpha: float
    delta_m: float
    n: int
    source_path: str
    source_sha256: str
    declared_content_sha256: str


def load_margins(path: str) -> MarginSet:
    """Load ``margins_v1.json`` and return the exact values it holds.

    Values are never rounded or re-typed by hand: ``alpha`` enters the cost as a
    multiplier, so a 1e-4 transcription error is a 1e-4 relative cost error that
    no downstream check would catch.

    Two hashes are recorded on purpose.  ``source_sha256`` is the byte hash of the
    file (this is what ``data_lock_v1.margins.sha256`` binds);
    ``declared_content_sha256`` is the file's own ``sha256`` field, which
    ``data_lock_v1.margins.content_sha256`` binds.  They are different numbers and
    must not be compared against each other.
    """
    payload = json.load(open(path, encoding="utf-8"))
    try:
        alpha = float(payload["alpha"])
        delta = float(payload["delta"])
    except KeyError as exc:  # pragma: no cover - schema guard
        raise CostAdapterError(f"margins file missing key {exc}") from exc
    if not math.isfinite(alpha) or alpha < 0.0:
        raise CostAdapterError(f"alpha must be finite and >= 0, got {alpha}")
    if not math.isfinite(delta) or delta < 0.0:
        raise CostAdapterError(f"delta must be finite and >= 0, got {delta}")
    return MarginSet(
        alpha=alpha,
        delta_m=delta,
        n=int(payload.get("n_eps_c", payload.get("n_trials", 0)) or 0),
        source_path=path,
        source_sha256=byte_sha256(path),
        declared_content_sha256=str(payload.get("sha256", "")),
    )


# ---------------------------------------------------------------- budget
@dataclass
class SignedBudget:
    """Signed remaining budget in joules with a scene-fixed normalisation.

    ``z_J = z0_J - spent_J`` may go negative; that is the point.  ``z_ratio``
    divides by ``Z_REF_J`` (a scene constant), never by ``z0_J``, so two
    episodes with different budgets remain comparable and over-spend stays
    visible after clipping.
    """

    z0_J: float
    z_ref_J: float = Z_REF_J
    spent_J: float = 0.0
    unit: str = COST_UNIT
    charges: int = 0

    def __post_init__(self) -> None:
        if self.unit != COST_UNIT:
            raise CostAdapterError(f"only unit={COST_UNIT!r} is accepted, got {self.unit!r}")
        if not math.isfinite(self.z0_J) or self.z0_J <= 0.0:
            raise CostAdapterError(f"z0_J must be positive and finite, got {self.z0_J}")
        if not math.isfinite(self.z_ref_J) or self.z_ref_J <= 0.0:
            raise CostAdapterError(f"z_ref_J must be positive and finite, got {self.z_ref_J}")
        if not math.isfinite(self.spent_J):
            raise CostAdapterError(f"spent_J must be finite, got {self.spent_J}")

    # ---- state ----
    @property
    def z_J(self) -> float:
        return self.z0_J - self.spent_J

    @property
    def over_budget_J(self) -> float:
        return max(0.0, -self.z_J)

    @property
    def exhausted(self) -> bool:
        return self.z_J <= 0.0

    @property
    def affordable(self) -> bool:
        return self.z_J >= 0.0

    @property
    def z_ratio(self) -> float:
        """Clipped signed ratio, identical to the ``obs-v1`` observation field."""
        return max(-1.0, min(1.0, self.z_J / self.z_ref_J))

    # ---- mutation ----
    def charge(self, cost_J: float) -> float:
        """Spend ``cost_J`` joules and return the new signed remaining budget."""
        cost = float(cost_J)
        if not math.isfinite(cost):
            raise CostAdapterError(f"step cost must be finite, got {cost}")
        if cost < 0.0:
            raise CostAdapterError(f"step cost must be non-negative, got {cost}")
        self.spent_J += cost
        self.charges += 1
        return self.z_J

    def reset(self, z0_J: float | None = None) -> None:
        """Start a new episode.  Spending never carries across episodes."""
        if z0_J is not None:
            if not math.isfinite(z0_J) or z0_J <= 0.0:
                raise CostAdapterError(f"z0_J must be positive and finite, got {z0_J}")
            self.z0_J = float(z0_J)
        self.spent_J = 0.0
        self.charges = 0

    # ---- reversible normalisation ----
    def to_ratio(self, z_J: float) -> float:
        return max(-1.0, min(1.0, float(z_J) / self.z_ref_J))

    def from_ratio(self, ratio: float) -> float:
        """Inverse of :meth:`to_ratio`, exact only inside ``[-1, 1]``."""
        r = float(ratio)
        if not -1.0 <= r <= 1.0:
            raise CostAdapterError(
                f"ratio must lie in [-1, 1] to be invertible, got {r} "
                "(outside it the value was clipped and the inverse is not unique)"
            )
        return r * self.z_ref_J

    def to_dict(self) -> dict:
        return {
            "unit": self.unit,
            "z0_J": self.z0_J,
            "spent_J": self.spent_J,
            "z_J": self.z_J,
            "over_budget_J": self.over_budget_J,
            "exhausted": self.exhausted,
            "z_ref_J": self.z_ref_J,
            "z_ratio": self.z_ratio,
            "charges": self.charges,
        }


def normalized_cost(cost_J: float) -> float:
    """Convert a joule cost to the dimensionless scale used by ``V_c``."""
    return float(cost_J) / Z_REF_J


# ---------------------------------------------------------------- step cost
@dataclass(frozen=True)
class StepCostRecord:
    """One navigation step (0.1 s) of the model ledger."""

    t_s: float
    command: tuple[float, float, float]
    power_W: float
    cost_nom_J: float
    cost_model_J: float
    alpha: float
    unit: str = COST_UNIT

    def as_dict(self) -> dict:
        return {
            "t_s": self.t_s,
            "command": list(self.command),
            "power_W": self.power_W,
            "cost_nom_J": self.cost_nom_J,
            "cost_model_J": self.cost_model_J,
            "alpha": self.alpha,
            "unit": self.unit,
        }


class JCostModel:
    """Frozen power model + an explicitly chosen ``alpha``.

    ``alpha`` has **no default**.  The frozen ``StepCost`` defaults to 0.0, which
    silently turns the calibrated method into the uncalibrated baseline; that is
    exactly the failure this class refuses to reproduce.
    """

    def __init__(self, model, alpha: float, *, variant: str, alpha_source: str) -> None:
        if variant not in VARIANTS:
            raise CostAdapterError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if alpha is None:
            raise CostAdapterError("alpha must be given explicitly (no default)")
        a = float(alpha)
        if not math.isfinite(a) or a < 0.0:
            raise CostAdapterError(f"alpha must be finite and >= 0, got {a}")
        if variant == "full" and a <= 0.0:
            raise CostAdapterError(
                "variant='full' requires alpha > 0; alpha=0 is the uncalibrated baseline, "
                "use variant='uncalibrated' to say so explicitly"
            )
        if variant == "uncalibrated" and a != 0.0:
            raise CostAdapterError(
                f"variant='uncalibrated' requires alpha == 0, got {a}"
            )
        self.model = model
        self.alpha = a
        self.variant = variant
        self.alpha_source = alpha_source
        self.dt_s = float(model.step_cost_dt)
        if abs(self.dt_s - DT_NAV_S) > 1e-12:
            raise CostAdapterError(
                f"model step_cost_dt={self.dt_s} differs from the 0.1 s navigation clock"
            )
        self._t_s = 0.0
        self._steps = 0

    # ---- construction ----
    @classmethod
    def from_frozen(
        cls, power_model_path: str, margins_path: str, *, variant: str = "full"
    ) -> "JCostModel":
        """Build from the frozen files; ``variant`` decides which alpha is legal."""
        from camra_calibration.power import PowerModel  # frozen module, read-only use

        payload = json.load(open(power_model_path, encoding="utf-8"))
        model = PowerModel.from_dict(payload)
        if variant == "full":
            margins = load_margins(margins_path)
            return cls(
                model,
                margins.alpha,
                variant="full",
                alpha_source=f"{margins_path} (alpha={margins.alpha!r}, sha256={margins.source_sha256})",
            )
        return cls(model, 0.0, variant="uncalibrated", alpha_source="explicit baseline: alpha=0, delta=0")

    # ---- lifecycle ----
    def reset(self) -> None:
        self._t_s = 0.0
        self._steps = 0

    # ---- single step ----
    def step(self, applied_command: Sequence[float], *, budget: SignedBudget | None = None) -> StepCostRecord:
        """Cost of one 0.1 s interval driven by the *applied* command.

        Only ``record.applied`` (see ``observation.CommandRecord``) may be passed
        here.  Using the requested command would charge for motion the machine
        was never asked to perform.
        """
        cmd = tuple(float(v) for v in applied_command)
        if len(cmd) != 3:
            raise CostAdapterError(f"command must be [vx, vy, wz], got {cmd}")
        if not all(math.isfinite(v) for v in cmd):
            raise CostAdapterError(f"command contains NaN/inf: {cmd}")
        ok, bad = self.model.in_domain([cmd, cmd])
        if not ok:
            raise CostAdapterError(
                f"command outside the power-model domain {bad}; "
                "the model must not be extrapolated silently"
            )
        power = float(self.model.step_power([cmd, cmd])[0])
        cost_nom = power * self.dt_s
        cost_model = (1.0 + self.alpha) * cost_nom
        rec = StepCostRecord(
            t_s=self._t_s,
            command=cmd,
            power_W=power,
            cost_nom_J=cost_nom,
            cost_model_J=cost_model,
            alpha=self.alpha,
        )
        if budget is not None:
            budget.charge(cost_model)
        self._t_s += self.dt_s
        self._steps += 1
        return rec

    # ---- whole segment ----
    def segment_cost(
        self, applied_commands: Iterable[Sequence[float]], *, budget: SignedBudget | None = None
    ) -> dict:
        """Cost a whole segment, one 0.1 s interval per command point after the first.

        Returns both the step-wise sum and the closed-form ``(1+alpha) * J_nom`` so
        the caller can assert they agree; a mismatch means the adapter is charging
        per step in a way that does not compose.
        """
        cmds = [tuple(float(v) for v in c) for c in applied_commands]
        if len(cmds) < 2:
            raise CostAdapterError(f"need at least 2 command points, got {len(cmds)}")
        self.reset()
        records = [self.step(c, budget=budget) for c in cmds[:-1]]
        step_sum = sum(r.cost_model_J for r in records)
        whole = (1.0 + self.alpha) * float(self.model.j_nom(cmds))
        return {
            "n_intervals": len(records),
            "duration_s": len(records) * self.dt_s,
            "cost_nom_J": sum(r.cost_nom_J for r in records),
            "cost_model_J_stepwise": step_sum,
            "cost_model_J_whole": whole,
            "abs_diff_J": abs(step_sum - whole),
            "alpha": self.alpha,
            "variant": self.variant,
        }

    def schema(self) -> dict:
        return {
            "schema": "camra-navigation-j-cost-v1",
            "unit": COST_UNIT,
            "dt_s": self.dt_s,
            "alpha": self.alpha,
            "variant": self.variant,
            "alpha_source": self.alpha_source,
            "history_k": int(self.model.history_k),
            "model_sha256": self.model.sha256(),
            "z_ref_J": Z_REF_J,
            "rule": "c_nom = P_nom(u_applied)*0.1 s; c_model = (1+alpha)*c_nom; z -= c_model",
        }


# ---------------------------------------------------------------- execution ledger
@dataclass
class ExecutionEnergy:
    """Mechanical work measured at the physics rate, kept off the model budget.

    ``sum_j max(tau_j * qdot_j, 0) * dt`` over the executed trajectory.  Positive
    work only: negative work (the actuator absorbing energy) is not recoverable by
    the battery and must not be credited back.
    """

    total_J: float = 0.0
    substeps: int = 0
    dt_s: float = DT_PHYSICS_S

    def add_substep(self, tau: Sequence[float], qdot: Sequence[float]) -> float:
        if len(tau) != len(qdot):
            raise CostAdapterError(f"tau/qdot length mismatch: {len(tau)} vs {len(qdot)}")
        work = 0.0
        for t, q in zip(tau, qdot):
            tf, qf = float(t), float(q)
            if not (math.isfinite(tf) and math.isfinite(qf)):
                raise CostAdapterError("tau/qdot must be finite")
            p = tf * qf
            if p > 0.0:
                work += p
        self.total_J += work * self.dt_s
        self.substeps += 1
        return self.total_J

    def reset(self) -> None:
        self.total_J = 0.0
        self.substeps = 0

    def to_dict(self) -> dict:
        return {"total_J": self.total_J, "substeps": self.substeps, "dt_s": self.dt_s}
