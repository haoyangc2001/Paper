"""E_ref for the 9 budget_selection maps (NAV-041 anchor).

The budget-comparison contract anchors every budget at ``z0 = k * E_ref(map)``,
with ``E_ref`` the NAV-017 admission route's deadline-speed nominal cost.  That
quantity only depends on geometry and the frozen power model, never on a policy
or a margin, so recomputing it for a new split introduces no result dependence.

``reports/train_map_reference_energy.json`` covers train (18) + development (6)
only; the budget-selection split (seeds 3001-3003, 3 levels) has no recorded
admission energy.  This script recomputes it through the *same frozen code path*
(``maps.feasibility.check_map``, resolution 0.15) and writes a separate evidence
file, so the existing report -- which other tasks cite by hash -- is untouched.

Run:

    <frozen python> CaMRA/navigation/scripts/compute_budget_selection_refs.py \
        --json CaMRA/navigation/reports/budget_selection_reference_energy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_calibration.power import PowerModel  # noqa: E402

from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
SPLIT = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")

RESOLUTION = 0.15  # same as tests/test_map_feasibility.py and the train/dev pass
SPLIT_NAME = "budget_selection"


def main() -> int:
    ap = argparse.ArgumentParser(description="E_ref for the budget_selection maps")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    split = json.load(open(SPLIT, encoding="utf-8"))
    maps = split["splits"][SPLIT_NAME]["maps"]

    records: list[dict[str, Any]] = []
    for m in maps:
        spec = M.build(m["level"], m["seed"], design)
        rec = F.check_map(spec, geometry, model, resolution=RESOLUTION)
        route = rec.get("route") or {}
        records.append({
            "map_id": m["map_id"],
            "split": SPLIT_NAME,
            "level": m["level"],
            "seed": m["seed"],
            "admitted": bool(rec.get("admitted")),
            "reasons": rec.get("reasons", []),
            "route_length_m": route.get("length_m"),
            "v_deadline_mps": route.get("v_deadline_mps"),
            "t_min_s": route.get("t_min_s"),
            "e_ref_J": route.get("predicted_energy_deadline_J"),
            "e_fast_J": route.get("predicted_energy_fast_J"),
            "geometry_hash_matches_split":
                rec.get("geometry_hash") == m.get("geometry_hash")
                or spec.geometry_hash() == m.get("geometry_hash"),
        })

    missing = [r["map_id"] for r in records if not r["admitted"] or r["e_ref_J"] is None]
    bad_hash = [r["map_id"] for r in records if not r["geometry_hash_matches_split"]]
    e_refs = [r["e_ref_J"] for r in records if r["e_ref_J"] is not None]
    z_ref = 550.0
    saturated = [r["map_id"] for r in records
                 if r["e_ref_J"] is not None and 1.30 * r["e_ref_J"] > z_ref]
    report = {
        "task": "NAV-041",
        "schema": "camra-budget-selection-reference-energy-v1",
        "verdict": "PASS" if (not missing and not bad_hash) else "FAIL",
        "split": SPLIT_NAME,
        "n_maps": len(records),
        "all_admitted": not missing,
        "missing_or_unadmitted": missing,
        "geometry_hash_mismatches": bad_hash,
        "e_ref_range_J": [min(e_refs), max(e_refs)] if e_refs else None,
        "z_ref_J": z_ref,
        "maps_saturating_at_k_1.30": saturated,
        "anchor_rule": "z0 = k * E_ref(map); E_ref is policy-independent",
        "maps": records,
        "note": "Recomputed with the frozen maps.feasibility.check_map code path "
                "(resolution 0.15, the same call the train/development pass uses). "
                "Separate file so reports/train_map_reference_energy.json, which "
                "other tasks cite by hash, stays byte-identical.",
    }
    out = args.json or os.path.join(NAV_ROOT, "reports",
                                    "budget_selection_reference_energy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"verdict": report["verdict"], "n_maps": report["n_maps"],
                      "e_ref_range_J": report["e_ref_range_J"],
                      "saturating_k130": saturated}, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
