"""Budget search checks for NAV-040.

The search must return a **physical joule budget** together with the
criterion, the bracket and the iteration trace, must distinguish all six
outcomes the plan names (feasible / no root / all feasible / all infeasible /
non-monotonic / NaN), must never expand the interval or fabricate a midpoint,
and must invert exactly back to the same normalised input.

Two problem families are used:

* a **restricted small problem** whose minimal budget is known in closed form
  (a two-path chain: taking the cheaper path costs exactly 45.0 J), so the
  search result can be compared against an independent exact number;
* **hand-made value curves** for the degenerate cases, where the "critic" is a
  literal Python function and the expected status is unambiguous.

Run:

    <frozen python> CaMRA/navigation/tests/test_budget_search.py \
        --json CaMRA/navigation/reports/budget_search_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO, "CaMRA", "calibration", "src"))

from camra_navigation.budget import search as BS  # noqa: E402

RESULTS: list[dict] = []
Z_REF = 550.0


def check(name: str, ok: bool, detail) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


# ------------------------------------------------- restricted small problem
def toy_chain_min_budget() -> float:
    """Two paths from start to goal; the cheaper one costs 45.0 J.

    Path A: 18 + 27 = 45 J (two hops).  Path B: 30 + 25 = 55 J.
    A budget is feasible iff it covers the cheaper path, so the exact minimal
    feasible budget is 45.0 J - computed here by hand, independently of the
    search.
    """
    path_a = (18.0, 27.0)
    path_b = (30.0, 25.0)
    return min(sum(path_a), sum(path_b))


def toy_score(z_J: float) -> float:
    """``score >= 0`` iff the budget covers the cheaper path.

    The +0.0 boundary is exact: at z = 45 the score is 0, which counts as
    feasible (contract: ``W <= -beta`` is inclusive).
    """
    return z_J - toy_chain_min_budget()


# ------------------------------------------------------------ the checks
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    exact = toy_chain_min_budget()

    # --- 1. restricted small problem, root inside the bracket --------------
    res = BS.search_min_budget(toy_score, lo_J=10.0, hi_J=100.0, z_ref_J=Z_REF,
                               tol_J=0.25)
    # bisection returns a *measured* feasible budget, so the answer sits in
    # [exact, exact + tol] by construction - not exactly on the root
    ok = (res.status == "found"
          and res.z_J is not None and exact <= res.z_J <= exact + 0.25 + 1e-9
          and res.interval_J[0] <= exact <= res.interval_J[1]
          and len(res.trace) >= 3)
    check("toy_chain_finds_exact_min_budget", ok,
          {"status": res.status, "z_J": res.z_J, "exact": exact,
           "interval_J": list(res.interval_J), "iterations": len(res.trace)})

    # --- 2. trace and bracket are coherent --------------------------------
    measured = [t for t in res.trace if t.z_J == res.z_J and t.feasible]
    ok = (res.trace[0].z_J == 10.0
          and all((res.trace[i].z_J != res.trace[i + 1].z_J)
                  for i in range(len(res.trace) - 1))
          and res.interval_J[1] - res.interval_J[0] <= 0.25 + 1e-12
          # the returned budget must have been *measured* feasible, never
          # interpolated from two neighbours
          and len(measured) == 1)
    check("toy_chain_trace_and_bracket", ok,
          {"n_trace": len(res.trace), "bracket_width": res.interval_J[1] - res.interval_J[0],
           "answer_measured_feasible": bool(measured),
           "last_entry": res.trace[-1].__dict__})

    # --- 3. invertibility: same z from the ratio ---------------------------
    back = BS.from_ratio(res.z_ratio, Z_REF)
    ok = abs(back - res.z_J) <= 1e-9 and BS.to_ratio(res.z_J, Z_REF) == res.z_ratio
    check("inverse_returns_same_z_input", ok,
          {"z_J": res.z_J, "z_ratio": res.z_ratio, "from_ratio": back})
    # outside [-1, 1] the inverse must refuse rather than invent a budget
    refused = False
    try:
        BS.from_ratio(1.2, Z_REF)
    except BS.BudgetSearchError:
        refused = True
    check("inverse_refuses_clipped_ratio", refused, {"ratio": 1.2, "refused": refused})

    # --- 4. all feasible (low endpoint already ok) ------------------------
    r_all = BS.search_min_budget(toy_score, lo_J=60.0, hi_J=100.0, z_ref_J=Z_REF)
    ok = (r_all.status == "all_feasible" and r_all.z_J == 60.0
          and "expanded" in " ".join(r_all.notes))
    check("all_feasible_returns_endpoint_not_smaller_budget", ok,
          {"status": r_all.status, "z_J": r_all.z_J, "notes": list(r_all.notes)})

    # --- 5. all infeasible (no root inside) ------------------------------
    r_none = BS.search_min_budget(toy_score, lo_J=5.0, hi_J=40.0, z_ref_J=Z_REF)
    ok = (r_none.status == "all_infeasible" and r_none.z_J is None
          and r_none.interval_J == (5.0, 40.0))
    check("all_infeasible_reports_no_root", ok,
          {"status": r_none.status, "interval_J": list(r_none.interval_J)})

    # --- 6. non-monotonic hand curve -------------------------------------
    # feasible at 40, infeasible at 50 (an "island"), feasible again at 90:
    # "smallest feasible budget" is ill-defined and must be reported as such.
    def island(z: float) -> float:
        if 45.0 <= z <= 60.0:
            return -1.0        # budget is "not feasible" in this band
        return z - 40.0        # feasible otherwise above 40 (island below 45)

    r_non = BS.search_min_budget(island, lo_J=10.0, hi_J=100.0, z_ref_J=Z_REF,
                                 tol_J=0.5)
    ok = (r_non.status == "non_monotonic"
          and any("non-monotonic" in n for n in r_non.notes))
    check("non_monotonic_curve_detected", ok,
          {"status": r_non.status, "notes": list(r_non.notes)[:2],
           "n_trace": len(r_non.trace)})

    # --- 7. NaN / raising criterion --------------------------------------
    r_nan = BS.search_min_budget(lambda z: float("nan"), lo_J=10.0, hi_J=100.0,
                                 z_ref_J=Z_REF)
    ok = r_nan.status == "invalid" and r_nan.z_J is None
    check("nan_score_is_invalid_not_feasible", ok,
          {"status": r_nan.status, "notes": list(r_nan.notes)[:2]})

    def raising(z: float) -> float:
        if z > 55.0:
            raise RuntimeError("critic exploded")
        return z - 45.0

    r_raise = BS.search_min_budget(raising, lo_J=10.0, hi_J=100.0, z_ref_J=Z_REF)
    ok = r_raise.status == "invalid" and r_raise.z_J is None
    check("raising_criterion_is_invalid", ok,
          {"status": r_raise.status, "notes": list(r_raise.notes)[:2]})

    # --- 8. argument guards (no silent guessing) -------------------------
    guards = []
    for kw in ({"lo_J": 100.0, "hi_J": 10.0},
               {"lo_J": -5.0, "hi_J": 10.0},
               {"lo_J": 10.0, "hi_J": 100.0, "tol_J": 0.0},
               {"lo_J": 10.0, "hi_J": 100.0, "z_ref_J": 0.0}):
        try:
            BS.search_min_budget(toy_score, z_ref_J=kw.pop("z_ref_J", Z_REF), **kw)
            guards.append(False)
        except BS.BudgetSearchError:
            guards.append(True)
    check("argument_guards_raise", all(guards), {"results": guards})

    # --- 9. endpoint exactly at the root --------------------------------
    r_edge = BS.search_min_budget(toy_score, lo_J=45.0, hi_J=100.0, z_ref_J=Z_REF)
    ok = r_edge.status == "all_feasible" and r_edge.z_J == 45.0
    check("root_on_low_endpoint_counts_feasible", ok,
          {"status": r_edge.status, "z_J": r_edge.z_J})

    # --- 10. grid selection rule (contract) -----------------------------
    grid = (0.70, 0.85, 1.00, 1.15, 1.30)
    e_ref = 60.0     # budgets: 42, 51, 60, 69, 78 J; feasible from 51 J (k=0.85)
    g = BS.select_smallest_feasible_on_grid(
        lambda z: z - 50.0, k_values=grid, e_ref_J=e_ref, z_ref_J=Z_REF)
    ok = (g["status"] == "found" and abs(g["k"] - 0.85) < 1e-12
          and abs(g["z_J"] - 51.0) < 1e-9 and g["e_ref_J"] == 60.0
          and g["z_J"] == BS.budget_J(g["k"], g["e_ref_J"]))
    check("grid_selects_smallest_feasible_k_and_reports_j_separately", ok,
          {"status": g["status"], "k": g["k"], "z_J": g["z_J"], "e_ref_J": g["e_ref_J"]})

    g_none = BS.select_smallest_feasible_on_grid(
        lambda z: z - 1000.0, k_values=grid, e_ref_J=e_ref, z_ref_J=Z_REF)
    ok = g_none["status"] == "no_feasible_budget" and g_none["k"] is None
    check("grid_no_feasible_point_no_extension", ok,
          {"status": g_none["status"], "n_grid": len(g_none["grid"])})

    bad_grid = []
    for ks in ((1.0, 0.85), (0.0, 1.0), ()):
        try:
            BS.select_smallest_feasible_on_grid(toy_score, k_values=ks,
                                                e_ref_J=e_ref, z_ref_J=Z_REF)
            bad_grid.append(False)
        except BS.BudgetSearchError:
            bad_grid.append(True)
    check("grid_guards_raise", all(bad_grid), {"results": bad_grid})

    # --- 11. precision / iteration-cap behaviour -----------------------
    r_tight = BS.search_min_budget(toy_score, lo_J=10.0, hi_J=100.0, z_ref_J=Z_REF,
                                   tol_J=1e-6, max_iter=80)
    ok = (r_tight.status == "found" and r_tight.z_J is not None
          and exact <= r_tight.z_J <= exact + 1e-6 + 1e-12)
    check("tight_tolerance_reaches_root", ok,
          {"z_J": r_tight.z_J, "exact": exact, "iterations": len(r_tight.trace)})

    r_cap = BS.search_min_budget(toy_score, lo_J=10.0, hi_J=100.0, z_ref_J=Z_REF,
                                 tol_J=1e-12, max_iter=3)
    ok = (r_cap.status == "found" and any("cap" in n for n in r_cap.notes))
    check("iteration_cap_disclosed_not_hidden", ok,
          {"status": r_cap.status, "iterations": len(r_cap.trace),
           "notes": [n for n in r_cap.notes if "cap" in n]})

    # --- 12. result is JSON-ready and carries the criterion -------------
    payload = json.loads(json.dumps(res.as_dict(), ensure_ascii=False))
    ok = (payload["status"] == "found" and payload["z_J"] is not None
          and payload["score"] is not None and payload["iterations"] >= 3
          and "trace" in payload and "interval_J" in payload)
    check("result_serialises_with_criterion_and_trace", ok,
          {"keys": sorted(payload.keys())})

    verdict = "PASS" if all(r["ok"] for r in RESULTS) else "FAIL"
    report = {
        "task": "NAV-040",
        "check": "budget_search",
        "verdict": verdict,
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "checks": RESULTS,
    }
    print(json.dumps({"verdict": verdict, "n_checks": len(RESULTS),
                      "n_failed": report["n_failed"]}))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
