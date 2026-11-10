"""G1 gate audit for the nominal environment stage (NAV-023).

G1 aggregates the evidence produced by NAV-012-022 and refuses to pass while any
scoring-relevant semantic is unresolved.  Every verdict is re-read from the
report files, every hash is recomputed, and the three semantic topics the plan
names - cost discounting, attitude handling, budget semantics - are checked
against their owning contracts rather than trusted from prose.

Run:

    <frozen python> CaMRA/navigation/scripts/check_gate_g1.py \
        --json CaMRA/navigation/reports/gate_G1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(NAV_ROOT, "scripts"))
sys.path.insert(0, os.path.join(CALIB, "src"))

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def byte_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfg = lambda name: os.path.join(NAV_ROOT, "config", name)  # noqa: E731
    rep = lambda name: os.path.join(NAV_ROOT, "reports", name)  # noqa: E731

    # --- 1. every stage report still passes ------------------------------
    stage_reports = {
        "NAV-013 observation": "observation_contract_check.json",
        "NAV-056 value objective": "value_objective_alignment.json",
        "NAV-014 cost/budget": "cost_budget_check.json",
        "NAV-057 pose footprint": "pose_footprint_alignment.json",
        "NAV-015 safety geometry": "safety_geometry_check.json",
        "NAV-016 map generator": "map_catalogue.json",
        "NAV-017 feasibility": "map_feasibility.json",
        "NAV-018 nominal env": "nominal_env_check.json",
        "NAV-019 math validation": "nominal_math_validation.json",
        "NAV-020 split manifest": "split_manifest_check.json",
        "NAV-022 scene mapping": "scene_mapping_check.json",
    }
    summary = []
    failed = []
    total = 0
    for stage, fname in stage_reports.items():
        path = rep(fname)
        if not os.path.isfile(path):
            failed.append({"stage": stage, "error": "report missing"})
            continue
        data = json.load(open(path, encoding="utf-8"))
        n = data.get("n_checks", 0)
        f = data.get("n_failed", 0)
        total += n
        summary.append({"stage": stage, "report": fname, "checks": n, "failed": f,
                        "verdict": data.get("verdict")})
        if data.get("verdict") != "PASS" or f != 0:
            failed.append({"stage": stage, "verdict": data.get("verdict"), "failed": f})
    check(
        "all_stage_reports_pass",
        not failed and total >= 160,
        {"total_checks": total, "reports": summary, "failed": failed},
    )

    # --- 2. G0 is still the recorded entry gate ---------------------------
    g0 = json.load(open(rep("gate_G0.json"), encoding="utf-8"))
    check(
        "gate_G0_recorded_pass",
        g0.get("verdict") == "PASS",
        {"verdict": g0.get("verdict"), "tasks_checked": len(g0.get("task_checklist", []))},
    )

    # --- 3. contract identities are pinned and current --------------------
    nav_contract = json.load(open(cfg("navigation_contract_v1.json"), encoding="utf-8"))
    geom_contract = json.load(open(cfg("geometry_contract_v1.json"), encoding="utf-8"))
    value_contract = json.load(open(cfg("value_objective_contract_v1.json"), encoding="utf-8"))
    budget_contract = json.load(open(cfg("budget_comparison_contract_v1.json"), encoding="utf-8"))
    manifest = json.load(open(cfg("split_manifest_v1.json"), encoding="utf-8"))
    prereg = json.load(open(cfg("experiment_prereg_v1.json"), encoding="utf-8"))
    design = json.load(open(cfg("map_design_v1.json"), encoding="utf-8"))
    identities = {
        "navigation_contract": nav_contract["version"],
        "geometry_contract": geom_contract["version"],
        "value_objective": value_contract["version"],
        "budget_comparison": budget_contract["version"],
        "split_manifest_hash": manifest["integrity"]["manifest_hash"],
        "map_design": design["version"],
        "experiment_prereg": prereg["version"],
        "resource_budget": json.load(open(cfg("resource_budget_v1.json"), encoding="utf-8"))["version"],
    }
    check(
        "contract_identities_pinned",
        nav_contract["observation_version"] == "obs-v2"
        and nav_contract["action_version"] == "cmd-v1"
        and all(v for v in identities.values()),
        identities,
    )

    # --- 4. the split manifest hash recomputes ---------------------------
    from build_split_manifest import build_manifest, canonical  # noqa: E402

    sys_path_note = "build_split_manifest imported from scripts/"
    rebuilt = build_manifest()
    check(
        "split_manifest_recomputes",
        canonical(rebuilt) == canonical(manifest)
        and rebuilt["integrity"]["manifest_hash"] == manifest["integrity"]["manifest_hash"],
        {
            "declared": manifest["integrity"]["manifest_hash"],
            "rebuilt": rebuilt["integrity"]["manifest_hash"],
            "n_maps": manifest["exclusivity"]["n_maps_declared"],
            "note": sys_path_note,
        },
    )

    # --- 5. the prereg binds the current manifest and design -------------
    check(
        "prereg_binds_current_manifest",
        manifest["integrity"]["manifest_hash"]
        in prereg["binding_contracts"]["splits"]
        and prereg["version"] == "1.0.0",
        {
            "declared": prereg["binding_contracts"]["splits"],
            "current": manifest["integrity"]["manifest_hash"],
        },
    )

    # --- 6. the three semantic topics are resolved ------------------------
    open_items = nav_contract.get("open_items", [])
    unresolved = [
        it for it in open_items
        if isinstance(it, dict) and it.get("status") != "resolved"
    ]
    topics = {
        "cost_discount": value_contract["estimator"]["gamma_cost"] == 1.0,
        "attitude": geom_contract["attitude_treatment"]["chosen"] == "rotation_invariant_circumscribed_disc",
        "budget_semantics": "Z_REF_J" in value_contract["quantities"]["z_ratio"]["definition"]
        and "z_0" not in value_contract["quantities"]["z_ratio"]["definition"],
    }
    check(
        "semantic_conflicts_resolved",
        not unresolved and all(topics.values()),
        {
            "unresolved_open_items": unresolved,
            "topics": topics,
            "note": "折扣、姿态、预算口径三项各自由其契约给出确定值",
        },
    )

    # --- 7. provisional decisions all name a review owner -----------------
    provisional = nav_contract.get("provisional_decisions", [])
    missing_owner = [
        p.get("item") for p in provisional
        if not (p.get("review") or p.get("confirm_by") or p.get("review_by"))
    ]
    check(
        "provisional_decisions_have_review_owner",
        provisional and not missing_owner,
        {
            "n_provisional": len(provisional),
            "items": [p.get("item") for p in provisional],
            "missing_owner": missing_owner,
        },
    )

    # --- 8. frozen calibration files are still byte-identical -------------
    lock = json.load(open(os.path.join(CALIB, "config", "data_lock_v1.json"), encoding="utf-8"))
    targets = {
        "protocol": lock["protocol"],
        "power_model": lock["power_model"],
        "footprint": lock["footprint"],
        "margins": lock["margins"],
        "step_cost_schema": lock["step_cost_schema"],
    }
    drift = []
    for name, entry in targets.items():
        path = os.path.join(REPO, entry["path"])
        if byte_sha256(path) != entry.get("sha256"):
            drift.append(name)
    check(
        "frozen_calibration_files_unchanged",
        not drift,
        {"checked": list(targets), "drifted": drift, "basis": "data_lock entry.sha256"},
    )

    # --- 9. one command domain, everywhere -------------------------------
    domain = nav_contract["validated_command_domain"]
    check(
        "command_domain_single_source",
        domain["vx_mps"] == [-1.0, 1.0]
        and domain["vy_mps"] == [-0.4, 0.4]
        and domain["wz_radps"] == [-1.0, 1.0]
        and domain["resolution"]["rejected_value"] == {"vx": [-2.0, 2.0], "vy": [-1.0, 1.0], "wz": [-2.0, 2.0]},
        {
            "validated": [domain["vx_mps"], domain["vy_mps"], domain["wz_radps"]],
            "rejected": domain["resolution"]["rejected_value"],
            "sources_agreeing": domain["resolution"]["sources_agreeing"],
        },
    )

    # --- 10. downstream fields all name a source -------------------------
    margins = json.load(open(os.path.join(CALIB, "config", "margins_v1.json"), encoding="utf-8"))
    fields = {
        "alpha_full": (margins["alpha"], "margins_v1 exact file value via data_lock"),
        "delta_m": (margins["delta"], "margins_v1 q90_eps_d via data_lock"),
        "Z_REF_J": (value_contract["quantities"]["Z_REF_J"]["value"], "power-v1 idle power x 30 s"),
        "r_goal_m": (nav_contract["task_definition"]["goal"]["r_goal_m"], "footprint circumradius"),
        "t_max_s": (nav_contract["task_definition"]["horizon_s"], "protocol window"),
        "budget_grid_k": (budget_contract["budget_grid"]["k_values"], "frozen before results"),
        "event_ids": (sorted(e["id"] for e in nav_contract["event_table"]["events"]), "NAV-012"),
        "obs_dim": (nav_contract["observation_spec"]["total_dim"], "obs-v2"),
        "inflation_radius_m": (geom_contract["constraint_construction"]["numeric_r_plus_delta_m"], "r_fp + delta"),
    }
    expected_events = {"fall", "collision", "out_of_bounds", "internal_violation",
                       "goal_reached", "budget_exhausted", "stagnation", "timeout"}
    missing = [k for k, (v, src) in fields.items() if v is None or not src]
    if not expected_events <= set(fields["event_ids"][0]):
        missing.append("event_ids")
    check(
        "downstream_fields_have_sources",
        not missing,
        {
            "fields": {k: {"value": v, "source": s} for k, (v, s) in fields.items()},
            "missing": missing,
        },
    )

    # --- 11. business failures and implementation errors are separable ----
    fh = manifest["failure_handling"]
    feas = json.load(open(rep("map_feasibility.json"), encoding="utf-8"))
    rejected = [c for c in feas.get("checks", []) if c["check"].startswith("rejects_")]
    reject_names = {c["check"] for c in rejected}
    check(
        "business_vs_implementation_separable",
        bool(fh["counted_as_failure"]) and bool(fh["infrastructure_errors"])
        and len(rejected) >= 3
        and {"rejects_timeout", "rejects_turn_limited_route"} <= reject_names,
        {
            "failure_labels": fh["counted_as_failure"],
            "infrastructure_rule": fh["infrastructure_errors"][:40],
            "feasibility_reject_checks": sorted(reject_names),
            "note": "超时再分 timeout 与 turn_limited，两类失败与基础设施错误分别记账",
        },
    )

    # --- 12. training has not started ------------------------------------
    repo_reports = sorted(
        os.path.basename(p) for p in glob_reports(rep(""))
        if os.path.basename(p).startswith(("train_", "checkpoint", "run_"))
    )
    check(
        "formal_training_not_started",
        not repo_reports and prereg["engineering_dependency"]["not_yet_implemented"] is True,
        {
            "training_artifacts": repo_reports,
            "vectorised_env_status": prereg["engineering_dependency"]["status"],
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    gate = {
        "task": "NAV-023",
        "gate": "G1",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "total_stage_checks": total,
        "identities": identities,
        "unlocks": (
            "G1 通过后允许进入阶段 2（接通 v2、时序/复位/计量验证）与名义训练实现；"
            "正式训练仍受 prereg 的资源与 checkpoint 规则约束"
        ),
        "checks": RESULTS,
    }
    print(json.dumps({k: gate[k] for k in ("task", "gate", "verdict", "n_checks", "total_stage_checks")},
                     ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:300])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh2:
            json.dump(gate, fh2, ensure_ascii=False, indent=2)
            fh2.write("\n")
    return 0 if verdict else 1


def glob_reports(directory: str):
    import glob as _g
    return _g.glob(os.path.join(directory, "*"))


if __name__ == "__main__":
    raise SystemExit(main())
