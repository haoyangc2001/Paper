"""E_ref for the 18 training maps (NAV-036 prereg anchor).

The budget-comparison contract anchors every budget at ``z0 = k * E_ref(map)``
with ``E_ref(map)`` the NAV-017 admission route's deadline-speed nominal cost,
recomputed from geometry + the frozen power model and independent of any
policy.  ``reports/map_feasibility.json`` only covers the 9 catalogue maps
(seeds 11/...); the 18 train-split maps (seeds 1001-1006 x 3 levels) have no
recorded admission energy, so this script recomputes them with the same frozen
code path (``maps.feasibility.check_map``) and writes the evidence file the
formal-training preregistration cites.

Run:

    <frozen python> CaMRA/navigation/scripts/compute_train_map_refs.py \
        --json CaMRA/navigation/reports/train_map_reference_energy.json
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

RESOLUTION = 0.15  # same as tests/test_map_feasibility.py


def main() -> int:
    ap = argparse.ArgumentParser(description="E_ref for the 18 train maps")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    split = json.load(open(SPLIT, encoding="utf-8"))
    train_maps = split["splits"]["train"]["maps"]
    assert len(train_maps) == 18, len(train_maps)

    records: list[dict[str, Any]] = []
    for m in train_maps:
        spec = M.build(m["level"], m["seed"], design)
        rec = F.check_map(spec, geometry, model, resolution=RESOLUTION)
        route = rec.get("route") or {}
        records.append({
            "map_id": m["map_id"],
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

    missing = [r["map_id"] for r in records
               if not r["admitted"] or r["e_ref_J"] is None]
    e_refs = [r["e_ref_J"] for r in records if r["e_ref_J"] is not None]
    max_ref = max(e_refs) if e_refs else None
    # value_objective_contract 1.2.0: z_ratio saturation guard requires
    # z0 = k*E_ref <= Z_REF_J for the training anchor k=1.0.
    z_ref = json.load(open(
        os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json"),
        encoding="utf-8"))["quantities"]["Z_REF_J"]["value"]
    saturated = [r["map_id"] for r in records
                 if r["e_ref_J"] is not None and r["e_ref_J"] > float(z_ref)]

    verdict = "PASS" if not missing and not saturated else "FAIL"
    report = {
        "task": "NAV-036",
        "schema": "camra-train-map-reference-energy-v1",
        "verdict": verdict,
        "n_maps": len(records),
        "all_admitted": not missing,
        "e_ref_range_J": [min(e_refs), max(e_refs)] if e_refs else None,
        "z_ref_J": z_ref,
        "saturated_maps_at_k_1.0": saturated,
        "budget_rule": "z0 = k_train * E_ref(map); prereg pins k_train = 1.0",
        "maps": records,
        "note": "Recomputed with the frozen maps.feasibility.check_map code path "
                "(same as NAV-017 admission, resolution=0.15); policy-independent.",
    }
    print(json.dumps({"verdict": verdict, "n_maps": len(records),
                      "missing": missing, "saturated": saturated,
                      "e_ref_range_J": report["e_ref_range_J"]}))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
