"""Robust J-unit budget search (NAV-040).

The budget protocol (``budget_comparison_contract_v1`` 1.1.0) anchors budgets
at ``z0 = k * E_ref(map)`` and requires, per method, the **smallest** feasible
budget on a frozen grid.  This module provides the search that answers "what
is the smallest physical budget in joules at which the feasibility criterion
still holds", without ever confusing the physical budget ``z_J`` with the
normalised observation input ``z_ratio`` (``clip(z_J / Z_REF_J, -1, 1)``).

Contract-relevant properties, each enforced and tested:

* **Physical units only.**  The search interval is a pair of joule values
  chosen by the caller from train/development evidence; ``[-1, 1]`` is never
  used as a budget.  ``to_ratio``/``from_ratio`` are explicit and exact inside
  ``[-1, 1]`` (outside it the value was clipped, so the inverse is refused).
* **No fabricated answers.**  Four terminal states besides ``found``:
  ``all_feasible`` (the low endpoint already satisfies the criterion - the
  result is the endpoint, *not* a smaller invented budget), ``all_infeasible``
  (no root inside the interval), ``non_monotonic`` (a larger budget was
  observed infeasible than a smaller feasible one, so "the smallest feasible
  budget" is ill-defined) and ``invalid`` (NaN/inf score, bad arguments).
  The interval is **never** expanded automatically.
* **Traceable.**  Every iteration is recorded (budget, ratio, score, verdict)
  together with the final bracket, so the result is reproducible from the log.
* **Criterion-agnostic.**  ``score(z_J) -> float`` is supplied by the caller
  (in this project: the critic-based stop-rule residual, feasible when
  ``score >= 0``, i.e. ``W <= -beta``).  This module never assumes the score
  is a physical safety guarantee: an approximate critic says "feasible under
  the critic", which is not a safety certificate (contract limitation,
  repeated here on purpose).

``budget_J``/``k_of_budget`` convert between the contract's grid anchor and
the physical budget so that selection results stay separate from the common
reference budget: a searched budget is always reported as a J value with the
``k`` that produced it, never silently overwritten into the shared anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

#: Score threshold: ``score >= FEASIBLE_AT`` means feasible.  The project-wide
#: criterion is ``W <= -beta``; callers pass ``score = -(W + beta)`` so the
#: boundary sits at zero.
FEASIBLE_AT = 0.0

#: Statuses, in the order the module can return them.
STATUSES = ("found", "all_feasible", "all_infeasible", "non_monotonic", "invalid")


class BudgetSearchError(ValueError):
    """Raised for argument/contract violations that must not be guessed around."""


@dataclass(frozen=True)
class TraceEntry:
    iteration: int
    z_J: float
    z_ratio: float
    score: float
    feasible: bool


@dataclass(frozen=True)
class BudgetSearchResult:
    """Outcome of a budget search; ``status`` is one of :data:`STATUSES`."""

    status: str
    z_J: float | None
    z_ratio: float | None
    score: float | None
    #: final bracket ``(lo, hi)`` in joules (``hi`` is the smallest known
    #: feasible budget when ``status == "found"``)
    interval_J: tuple[float, float]
    trace: tuple[TraceEntry, ...]
    z_ref_J: float
    tol_J: float
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "z_J": self.z_J,
            "z_ratio": self.z_ratio,
            "score": self.score,
            "interval_J": list(self.interval_J),
            "z_ref_J": self.z_ref_J,
            "tol_J": self.tol_J,
            "iterations": len(self.trace),
            "trace": [t.__dict__ for t in self.trace],
            "notes": list(self.notes),
        }


# ------------------------------------------------------------ unit handling
def to_ratio(z_J: float, z_ref_J: float) -> float:
    """``clip(z_J / Z_REF_J, -1, 1)`` - the observation input, not a budget."""
    if not math.isfinite(z_J) or not math.isfinite(z_ref_J) or z_ref_J <= 0.0:
        raise BudgetSearchError(f"bad arguments to_ratio({z_J}, {z_ref_J})")
    return max(-1.0, min(1.0, float(z_J) / float(z_ref_J)))


def from_ratio(ratio: float, z_ref_J: float) -> float:
    """Inverse of :func:`to_ratio`; exact only inside ``[-1, 1]``."""
    r = float(ratio)
    if not -1.0 <= r <= 1.0:
        raise BudgetSearchError(
            f"ratio {r} lies outside [-1, 1]; the value was clipped and the "
            "inverse is not unique - refusing to invent a budget")
    if not math.isfinite(z_ref_J) or z_ref_J <= 0.0:
        raise BudgetSearchError(f"bad z_ref_J {z_ref_J}")
    return r * float(z_ref_J)


def budget_J(k: float, e_ref_J: float) -> float:
    """Contract anchor: ``z0 = k * E_ref(map)`` in joules."""
    kk, ee = float(k), float(e_ref_J)
    if not (math.isfinite(kk) and math.isfinite(ee)) or kk <= 0.0 or ee <= 0.0:
        raise BudgetSearchError(f"bad anchor arguments budget_J({k}, {e_ref_J})")
    return kk * ee


def k_of_budget(z_J: float, e_ref_J: float) -> float:
    """Inverse of :func:`budget_J` - reported alongside, never merged into it."""
    z, ee = float(z_J), float(e_ref_J)
    if not (math.isfinite(z) and math.isfinite(ee)) or ee <= 0.0:
        raise BudgetSearchError(f"bad arguments k_of_budget({z_J}, {e_ref_J})")
    return z / ee


# ------------------------------------------------------------------ search
def search_min_budget(
    score: Callable[[float], float],
    *,
    lo_J: float,
    hi_J: float,
    z_ref_J: float,
    tol_J: float = 0.5,
    max_iter: int = 60,
    probes: int = 8,
) -> BudgetSearchResult:
    """Smallest feasible budget in ``[lo_J, hi_J]`` (joules).

    ``score(z_J)`` returns the feasibility criterion; ``>= 0`` means feasible.
    The interval is supplied by the caller from train/development evidence and
    is never expanded.  Before narrowing, ``probes`` evenly spaced interior
    points are evaluated to check approximate monotonicity: feasibility has to
    look like "infeasible ... infeasible, feasible ... feasible" along
    increasing budget.  A single feasible-then-infeasible pair (an island)
    makes "the smallest feasible budget" ill-defined, and the search reports
    that instead of silently returning one of the island's edges.  Bisection
    then keeps the invariant "``a`` infeasible, ``b`` feasible" and stops when
    ``b - a <= tol_J``; the answer is ``b`` (a budget that was *measured*
    feasible, never an interpolation).
    """
    lo, hi = float(lo_J), float(hi_J)
    tol = float(tol_J)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise BudgetSearchError(f"interval must be finite, got [{lo}, {hi}]")
    if not lo < hi:
        raise BudgetSearchError(f"interval must satisfy lo < hi, got [{lo}, {hi}]")
    if not math.isfinite(tol) or tol <= 0.0:
        raise BudgetSearchError(f"tol_J must be positive and finite, got {tol}")
    if not math.isfinite(z_ref_J) or z_ref_J <= 0.0:
        raise BudgetSearchError(f"z_ref_J must be positive and finite, got {z_ref_J}")
    if int(max_iter) < 1:
        raise BudgetSearchError(f"max_iter must be >= 1, got {max_iter}")
    if lo <= 0.0:
        raise BudgetSearchError(
            f"budgets must be positive joules (the contract's z_J may go "
            f"negative at run time, but a search interval is a budget), got lo={lo}")
    probes = int(probes)
    if probes < 2:
        raise BudgetSearchError(
            f"probes must be >= 2 so monotonicity is actually inspected, got {probes}")

    trace: list[TraceEntry] = []
    notes: list[str] = []

    def evaluate(z: float, iteration: int) -> tuple[float, bool] | None:
        try:
            s = float(score(float(z)))
        except Exception as exc:  # a raising criterion is data, not a bug to hide
            notes.append(f"score raised at z_J={z}: {exc!r}")
            return None
        if not math.isfinite(s):
            trace.append(TraceEntry(iteration, float(z), to_ratio(z, z_ref_J), s, False))
            notes.append(f"non-finite score {s!r} at z_J={z}")
            return None
        feasible = s >= FEASIBLE_AT
        trace.append(TraceEntry(iteration, float(z), to_ratio(z, z_ref_J), s, feasible))
        return s, feasible

    s_lo = evaluate(lo, 0)
    s_hi = evaluate(hi, 0)
    if s_lo is None or s_hi is None:
        return BudgetSearchResult(
            "invalid", None, None, None, (lo, hi), tuple(trace), z_ref_J, tol,
            tuple(notes))
    if s_lo[1]:
        # the low endpoint already satisfies the criterion; the true minimum may
        # be smaller, but the interval is frozen - report the endpoint honestly
        notes.append("low endpoint already feasible; interval not expanded")
        return BudgetSearchResult(
            "all_feasible", lo, to_ratio(lo, z_ref_J), s_lo[0], (lo, hi),
            tuple(trace), z_ref_J, tol, tuple(notes))
    if not s_hi[1]:
        notes.append("high endpoint still infeasible; no root inside the interval")
        return BudgetSearchResult(
            "all_infeasible", None, None, s_hi[0], (lo, hi), tuple(trace),
            z_ref_J, tol, tuple(notes))

    # ---- approximate monotonicity probe ---------------------------------
    # Interior samples, increasing in z.  Feasibility must be non-decreasing.
    first_feasible_probe: float | None = None
    last_infeasible_probe: float | None = lo
    seen_feasible = False
    for i in range(1, probes + 1):
        z = lo + (hi - lo) * i / (probes + 1)
        got = evaluate(z, 0)
        if got is None:
            return BudgetSearchResult(
                "invalid", None, None, None, (lo, hi), tuple(trace), z_ref_J, tol,
                tuple(notes + ["probe evaluation failed; refusing to continue"]))
        s_p, feasible_p = got
        if feasible_p:
            if seen_feasible is False:
                first_feasible_probe = z
            seen_feasible = True
        else:
            if seen_feasible:
                notes.append(
                    f"non-monotonic: z_J={z} infeasible after a smaller budget "
                    f"({first_feasible_probe}) was feasible; the minimal feasible "
                    "budget is ill-defined")
                return BudgetSearchResult(
                    "non_monotonic", None, None, s_p, (lo, hi), tuple(trace),
                    z_ref_J, tol, tuple(notes))
            last_infeasible_probe = z
    if first_feasible_probe is None:
        first_feasible_probe = hi   # endpoints said hi is feasible
    a = float(last_infeasible_probe if last_infeasible_probe is not None else lo)
    b = float(first_feasible_probe)
    smallest_feasible = b
    smallest_feasible_score = None
    for t in reversed(trace):
        if t.feasible and t.z_J == b:
            smallest_feasible_score = t.score
            break
    k = 1
    while (b - a) > tol and k <= int(max_iter):
        mid = 0.5 * (a + b)
        got = evaluate(mid, k)
        if got is None:
            return BudgetSearchResult(
                "invalid", None, None, None, (a, b), tuple(trace), z_ref_J, tol,
                tuple(notes + ["midpoint evaluation failed; refusing to continue"]))
        s_mid, feasible = got
        if feasible:
            # monotonicity guard: any *larger* budget already measured
            # infeasible than this one means "smallest feasible" is ill-defined
            if mid < hi and _infeasible_above(trace, mid):
                notes.append(
                    f"non-monotonic: z_J={mid} feasible while a larger budget was "
                    "infeasible; the minimal feasible budget is ill-defined")
                return BudgetSearchResult(
                    "non_monotonic", None, None, s_mid, (a, b), tuple(trace),
                    z_ref_J, tol, tuple(notes))
            b = mid
            smallest_feasible = mid
            smallest_feasible_score = s_mid
        else:
            a = mid
        k += 1
    if (b - a) > tol:
        notes.append(f"iteration cap {max_iter} hit before tol; bracket {b - a:.4g} J")

    return BudgetSearchResult(
        "found", smallest_feasible, to_ratio(smallest_feasible, z_ref_J),
        smallest_feasible_score, (a, b), tuple(trace), z_ref_J, tol, tuple(notes))


def _infeasible_above(trace: Sequence[TraceEntry], z: float) -> bool:
    """Was any recorded budget strictly larger than ``z`` infeasible?"""
    return any((not t.feasible) and t.z_J > z for t in trace)


# ------------------------------------------------------- grid-based selection
def select_smallest_feasible_on_grid(
    score: Callable[[float], float],
    *,
    k_values: Sequence[float],
    e_ref_J: float,
    z_ref_J: float,
) -> dict:
    """Selection rule of the budget contract: walk the frozen k grid upward.

    The grid is consumed as-is (no densification, no shifting); the first ``k``
    whose measured criterion is feasible wins.  Returns the physical budget in
    joules **separately** from the ``k`` and from ``E_ref``, so downstream
    reports never conflate a selected budget with the shared reference budget.
    """
    ks = [float(k) for k in k_values]
    if not ks or any(not math.isfinite(k) or k <= 0.0 for k in ks):
        raise BudgetSearchError(f"bad k grid {k_values}")
    if any(ks[i] >= ks[i + 1] for i in range(len(ks) - 1)):
        raise BudgetSearchError(f"k grid must be strictly increasing, got {ks}")
    measured = []
    for k in ks:
        z = budget_J(k, e_ref_J)
        s = float(score(z))
        ok = math.isfinite(s) and s >= FEASIBLE_AT
        measured.append({"k": k, "z_J": z, "z_ratio": to_ratio(z, z_ref_J),
                         "score": s, "feasible": bool(ok)})
        if ok:
            return {
                "status": "found",
                "k": k,
                "z_J": z,
                "z_ratio": to_ratio(z, z_ref_J),
                "score": s,
                "e_ref_J": float(e_ref_J),
                "grid": measured,
                "note": "budget reported in J and k separately from E_ref; "
                        "critic feasibility is not a physical safety guarantee",
            }
    return {
        "status": "no_feasible_budget",
        "k": None, "z_J": None, "z_ratio": None, "score": None,
        "e_ref_J": float(e_ref_J), "grid": measured,
        "note": "no grid point feasible; the contract forbids extending the grid",
    }
