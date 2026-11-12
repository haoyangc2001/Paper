"""NAV-041 checks: the frozen budget-selection rule and its recorded output.

Three layers, all offline (no environment is stepped here):

1. **Rule layer** -- ``select_on_grid`` / ``monotonic_violations`` /
   ``feasible_from_label`` on synthetic rows: first-measured-feasible wins, no
   interpolation, no jumping to a larger k when non-monotonic, missing grid
   point is refused, all-infeasible yields ``no_feasible_budget``.
2. **Frozen-config layer** -- the config on disk matches its ``.sha256``, its
   grid equals the contract's five k values, its quota equals
   maps x methods x grid, and its actor binding is the lowest-seed candidate.
3. **Report layer** (only if the report exists) -- every ``(method, map, k)``
   appears exactly once, feasibility recomputes from the label, ``z0``
   recomputes as ``k * E_ref``, and the summary counts match the rows.

    <frozen python> CaMRA/navigation/tests/test_budget_selection.py \
        --json CaMRA/navigation/reports/budget_selection_rule_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))
sys.path.insert(0, NAV_ROOT)

from camra_navigation import observation as OBS  # noqa: E402
from scripts.run_budget_selection import (  # noqa: E402
    feasible_from_label,
    monotonic_violations,
    select_on_grid,
)

CONFIG = os.path.join(NAV_ROOT, "config", "budget_selection_v1.json")
CONFIG_SHA = CONFIG + ".sha256"
REPORT = os.path.join(NAV_ROOT, "reports", "budget_selection.json")
BUDGET_CONTRACT = os.path.join(NAV_ROOT, "config", "budget_comparison_contract_v1.json")
BUNDLE = os.path.join(NAV_ROOT, "delivery", "deployment_candidates_v1.json")

GRID = [0.7, 0.85, 1.0, 1.15, 1.3]
E_REF = 400.0
Z_REF = 550.0
CHECKS: list[dict] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: dict) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILED.append(name)


def rows_with(flags: dict[float, bool], e_ref: float = E_REF) -> list[dict]:
    # 不可行行用 timeout（到达之外的真实终止标签）；budget_exhausted 在
    # NAV-062 之后不再是终止标签，只能作诊断
    return [{"k": k, "z0_J": k * e_ref,
             "label": "goal_reached" if flags[k] else "timeout",
             "within_budget": True,
             "feasible": bool(flags[k])} for k in GRID]


def main() -> int:
    ap = argparse.ArgumentParser(description="NAV-041 rule checks")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # ---------------- 1. rule layer ----------------
    # NAV-062 起判据是合取：到达 **且** 预算非负。budget_exhausted 已不是终止标签，
    # 但旧报告里可能仍有它，规则层必须给出 False 而不是抛错。
    cases = {("goal_reached", True): True, ("goal_reached", False): False,
             ("collision", True): False, ("out_of_bounds", True): False,
             ("stagnation", True): False, ("timeout", True): False,
             ("budget_exhausted", True): False, (None, True): False}
    bad = {str(k): feasible_from_label(*k) for k in cases
           if feasible_from_label(*k) != cases[k]}
    check("feasible_requires_arrival_and_budget", not bad,
          {"mapping": {f"{k[0]}/{k[1]}": feasible_from_label(*k) for k in cases},
           "unexpected": list(bad)})
    raised = False
    try:
        feasible_from_label("goal_reached")          # 缺 within_budget 必须报错
    except ValueError:
        raised = True
    check("missing_budget_is_an_error_not_a_default", raised,
          {"note": "少了预算字段就抛错，避免悄悄退回只看标签的旧判据"})

    # first measured feasible wins on a monotone grid
    sel = select_on_grid(rows_with({0.7: False, 0.85: False, 1.0: True, 1.15: True, 1.3: True}),
                         grid=GRID, e_ref_J=E_REF, z_ref_J=Z_REF)
    check("picks_first_feasible_k", sel["status"] == "found" and sel["k"] == 1.0
          and abs(sel["z_J"] - 400.0) < 1e-12 and sel["monotonic"],
          {"selection": {k: v for k, v in sel.items() if k != "monotonicity_violations"}})

    # lowest point feasible -> k = 0.7, no earlier candidate exists
    sel_low = select_on_grid(rows_with({0.7: True, 0.85: True, 1.0: True, 1.15: True, 1.3: True}),
                             grid=GRID, e_ref_J=E_REF, z_ref_J=Z_REF)
    check("picks_lowest_grid_point", sel_low["status"] == "found" and sel_low["k"] == 0.7,
          {"k": sel_low["k"], "z_J": sel_low["z_J"]})

    # nothing feasible -> explicit no-solution record, never a fallback value
    none_sel = select_on_grid(rows_with({k: False for k in GRID}),
                              grid=GRID, e_ref_J=E_REF, z_ref_J=Z_REF)
    check("all_infeasible_is_no_feasible_budget",
          none_sel["status"] == "no_feasible_budget" and none_sel["k"] is None
          and none_sel["z_J"] is None and none_sel["monotonic"],
          {"selection": {k: v for k, v in none_sel.items()
                         if k != "monotonicity_violations"}})

    # non-monotonic: must keep the first feasible k and *record* the violation
    nm = select_on_grid(rows_with({0.7: False, 0.85: True, 1.0: False, 1.15: True, 1.3: True}),
                        grid=GRID, e_ref_J=E_REF, z_ref_J=Z_REF)
    check("non_monotonic_keeps_first_feasible_and_records",
          nm["status"] == "found" and nm["k"] == 0.85 and nm["monotonic"] is False
          and [v["k"] for v in nm["monotonicity_violations"]] == [1.0],
          {"selection": {k: v for k, v in nm.items() if k != "monotonicity_violations"},
           "violations": nm["monotonicity_violations"],
           "note": "契约：非单调必须记录并说明，不得悄悄改选更大的 k"})

    # clipping of the normalised value above Z_REF, but z_J itself is never clipped
    hi = select_on_grid(rows_with({0.7: False, 0.85: False, 1.0: False, 1.15: False, 1.3: True}),
                        grid=GRID, e_ref_J=500.0, z_ref_J=Z_REF)
    # obs-v3 的裁剪上限是 ±1.5（NAV-062）；650 J 的比值 1.1818 已不再饱和，
    # 所以这里同时验"未裁剪值保留"和"超过 1.5 才饱和"
    clip = OBS.z_ratio_clip()
    saturated = select_on_grid(
        rows_with({0.7: False, 0.85: False, 1.0: False, 1.15: False, 1.3: True}),
        grid=GRID, e_ref_J=1000.0, z_ref_J=Z_REF)
    check("z_ratio_clipped_but_budget_is_not",
          hi["status"] == "found" and abs(hi["z_J"] - 650.0) < 1e-9
          and abs(hi["z_ratio"] - 650.0 / 550.0) < 1e-12
          and abs(hi["z_ratio_clipped"] - min(650.0 / 550.0, clip)) < 1e-12
          and abs(saturated["z_ratio_clipped"] - clip) < 1e-12
          and abs(saturated["z_ratio"] - 1300.0 / 550.0) < 1e-12,
          {"clip": clip, "z_J": hi["z_J"], "z_ratio": hi["z_ratio"],
           "z_ratio_clipped": hi["z_ratio_clipped"],
           "saturated_ratio": saturated["z_ratio"],
           "saturated_clipped": saturated["z_ratio_clipped"]})

    # missing grid point must be refused rather than silently skipped
    partial = [r for r in rows_with({k: True for k in GRID}) if r["k"] != 1.15]
    try:
        select_on_grid(partial, grid=GRID, e_ref_J=E_REF, z_ref_J=Z_REF)
        refused = False
        msg = ""
    except ValueError as exc:
        refused = True
        msg = str(exc)
    check("missing_grid_point_refused", refused, {"error": msg})

    # monotonicity helper on its own
    check("monotonic_helper_matches_rule",
          monotonic_violations(rows_with({0.7: False, 0.85: False, 1.0: True,
                                          1.15: True, 1.3: True})) == []
          and [v["k"] for v in monotonic_violations(
              rows_with({0.7: False, 0.85: False, 1.0: True, 1.15: False, 1.3: True}))] == [1.15],
          {"note": "可行点之后不得再出现不可行点"})

    # ---------------- 2. frozen config ----------------
    if not os.path.exists(CONFIG):
        check("frozen_config_present", False, {"path": CONFIG, "note": "run --freeze first"})
        cfg = None
    else:
        raw = open(CONFIG, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        expected = open(CONFIG_SHA, encoding="utf-8").read().strip()
        cfg = json.loads(raw.decode("utf-8"))
        check("frozen_config_hash_matches_sidecar", sha == expected,
              {"sha256": sha, "sidecar": expected})
        contract = json.load(open(BUDGET_CONTRACT, encoding="utf-8"))
        check("grid_equals_contract_grid",
              [float(k) for k in cfg["grid"]["k_values"]]
              == [float(k) for k in contract["budget_grid"]["k_values"]],
              {"config": cfg["grid"]["k_values"], "contract": contract["budget_grid"]["k_values"]})
        n_maps = cfg["inputs"]["split_manifest"]["n_maps"]
        check("quota_is_maps_times_methods_times_grid",
              cfg["quota"]["business_episodes"] == n_maps * 2 * len(cfg["grid"]["k_values"])
              == cfg["quota"]["authorized_total"],
              {"quota": cfg["quota"], "n_maps": n_maps})
        bundle = json.load(open(BUNDLE, encoding="utf-8"))
        lowest = {m: min((c for c in bundle["candidates"] if c["method"] == m),
                         key=lambda c: c["seed"])
                  for m in ("uncalibrated", "full")}
        ok_actors = all(
            cfg["actor_binding"]["candidates"][m]["candidate_id"] == lowest[m]["candidate_id"]
            and cfg["actor_binding"]["candidates"][m]["actor_sha256"]
            == lowest[m]["weights"]["actor_sha256"]
            for m in lowest)
        check("actor_binding_is_lowest_seed_candidate", ok_actors,
              {m: cfg["actor_binding"]["candidates"][m]["candidate_id"] for m in lowest})
        check("mode_label_is_method_selected_budget",
              cfg["mode"]["label"] == "method_selected_budget"
              and "common_budget" in cfg["mode"]["not_common_budget"],
              {"label": cfg["mode"]["label"]})
        check("forbidden_rules_registered",
              len(cfg["forbidden"]) >= 5
              and any("重跑" in f for f in cfg["forbidden"])
              and any("final" in f for f in cfg["forbidden"]),
              {"n_forbidden": len(cfg["forbidden"]), "forbidden": cfg["forbidden"]})
        check("beta_and_units_declared",
              cfg["thresholds"]["beta"]["value"] == 0.0
              and "W <= -β" in cfg["thresholds"]["beta"]["meaning"]
              and cfg["inputs"]["z_ref"]["value"] > 0.0
              and cfg["units"]["not_a_unit"],
              {"beta": cfg["thresholds"]["beta"], "z_ref": cfg["inputs"]["z_ref"]["value"]})

    # ---------------- 3. report ----------------
    if not os.path.exists(REPORT) or cfg is None:
        check("report_present", False,
              {"path": REPORT, "note": "run the search after freezing"})
    else:
        rep = json.load(open(REPORT, encoding="utf-8"))
        rows = rep["rows"]
        keys = [(r["method"], r["map_id"], r["k"]) for r in rows]
        expected_rows = (len(cfg["inputs"]["split_manifest"]["maps"])
                         * len(rep["summary"]["per_method"]) * len(rep["grid"]))
        check("report_rows_unique_and_complete",
              len(keys) == len(set(keys)) == rep["n_episodes"] == expected_rows
              == cfg["quota"]["business_episodes"],
              {"n_rows": len(rows), "n_unique": len(set(keys)),
               "n_episodes": rep["n_episodes"], "expected": expected_rows})
        # 旧 v1 报告没有 within_budget 字段（那时预算耗尽还是终止标签），
        # 只能对带该字段的行复算；旧报告本身按旧语义保留，不改写
        scoped = [r for r in rows if "within_budget" in r]
        bad_feas = [f"{r['method']}/{r['map_id']}/k={r['k']}"
                    for r in scoped
                    if r["feasible"] != feasible_from_label(r["label"], r["within_budget"])]
        check("feasibility_recomputes_from_label_and_budget", not bad_feas,
              {"n_rows": len(rows), "n_scoped": len(scoped), "mismatches": bad_feas[:5],
               "note": "v1 报告无 within_budget，按旧语义只作历史证据"})
        bad_z = [f"{r['method']}/{r['map_id']}/k={r['k']}"
                 for r in rows if abs(r["z0_J"] - r["k"] * r["e_ref_J"]) > 1e-6]
        check("budget_recomputes_from_k_times_e_ref", not bad_z,
              {"n_rows": len(rows), "mismatches": bad_z[:5]})
        check("report_recompute_block_clean", rep["recompute"]["clean"],
              {"mismatches": rep["recompute"]["mismatches"][:5]})
        sel_bad = []
        for p in rep["per_method_map"]:
            sub = [r for r in rows if r["method"] == p["method"] and r["map_id"] == p["map_id"]]
            again = select_on_grid(sub, grid=rep["grid"], e_ref_J=p["e_ref_J"],
                                   z_ref_J=rep["z_ref_J"])
            if again["status"] != p["selection"]["status"] or again["k"] != p["selection"]["k"]:
                sel_bad.append(f"{p['method']}/{p['map_id']}")
        check("selection_recomputes_from_rows", not sel_bad,
              {"n_pairs": len(rep["per_method_map"]), "mismatches": sel_bad})
        counts_ok = True
        for method, s in rep["summary"]["per_method"].items():
            recs = [p for p in rep["per_method_map"] if p["method"] == method]
            found = [p for p in recs if p["selection"]["status"] == "found"]
            counts_ok &= (s["n_maps"] == len(recs)
                          and s["n_maps_with_feasible_budget"] == len(found)
                          and s["candidate_rows_measured"]
                          == sum(p["candidates_measured"] for p in recs))
        check("summary_counts_match_rows", bool(counts_ok),
              {m: rep["summary"]["per_method"][m]["candidate_rows_measured"]
               for m in rep["summary"]["per_method"]})
        check("report_keeps_mode_label_separate",
              rep["mode_label"] == "method_selected_budget"
              and "common_budget" in rep["not_common_budget"]
              and rep["limitations"],
              {"mode_label": rep["mode_label"], "n_limitations": len(rep["limitations"])})

    payload = {
        "task": "NAV-041", "schema": "camra-budget-selection-check-v1",
        "verdict": "PASS" if not FAILED else "FAIL",
        "n_checks": len(CHECKS), "failed": FAILED, "checks": CHECKS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    print(json.dumps({"verdict": payload["verdict"], "n_checks": len(CHECKS),
                      "failed": FAILED}, ensure_ascii=False))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
