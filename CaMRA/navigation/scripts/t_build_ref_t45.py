#!/usr/bin/env python3
"""Build the T=45 s reference-energy report for the 18 train maps (r13 contract).

Same schema as reports/train_map_reference_energy.json so train_formal.py can read
it through ``reference_energy.report``. ``e_ref_J`` is again
``predicted_energy_deadline_J`` — the energy of the reference route driven at
``v_deadline = route_length / 45`` — because the budget must stay in the same unit
as the spend ledger.

The 30 s report is left untouched: it is the frozen artefact of r1-r12.
"""
import json
import os
import sys

_NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
_CALIB = "/home/caohy/repositories/Paper/CaMRA/calibration"
sys.path.insert(0, os.path.join(_CALIB, "src"))
sys.path.insert(0, os.path.join(_NAV, "src"))

from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402
from camra_calibration.power import PowerModel  # noqa: E402

T = 45.0
DESIGN = os.path.join(_NAV, "config", "map_design_v1.json")
SPLIT = os.path.join(_NAV, "config", "split_manifest_v1.json")
FOOTPRINT = os.path.join(_CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(_CALIB, "config", "margins_v1.json")
GEOM = os.path.join(_NAV, "config", "geometry_contract_v1.json")
POWER = os.path.join(_CALIB, "config", "power_model_v1.json")
OUT = os.path.join(_NAV, "reports", "train_map_reference_energy_t45.json")


def main():
    design = json.load(open(DESIGN, encoding="utf-8"))
    split = json.load(open(SPLIT, encoding="utf-8"))
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM)
    model = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    maps = []
    for m in split["splits"]["train"]["maps"]:
        spec = M.build(m["level"], m["seed"], design)
        rec = F.check_map(spec, geom, model, resolution=0.1, t_deadline_s=T)
        route = rec.get("route") or {}
        maps.append({
            "map_id": m["map_id"], "split": "train", "level": m["level"],
            "seed": m["seed"], "admitted": bool(rec.get("admitted")),
            "reasons": rec.get("reasons", []),
            "route_length_m": route.get("length_m"),
            "v_deadline_mps": route.get("v_deadline_mps"),
            "e_ref_J": route.get("predicted_energy_deadline_J"),
            "e_fast_J": route.get("predicted_energy_fast_J"),
        })
        print("%-26s L=%6.2f v_d=%.3f e_ref=%7.1f admitted=%s"
              % (m["map_id"], route.get("length_m") or -1, route.get("v_deadline_mps") or -1,
                 route.get("predicted_energy_deadline_J") or -1, rec.get("admitted")))
    ok = all(r["admitted"] and r["e_ref_J"] for r in maps)
    e = [r["e_ref_J"] for r in maps if r["e_ref_J"]]
    payload = {
        "task": "NAV-064", "schema": "camra-train-map-reference-energy-v1",
        "verdict": "PASS" if ok else "FAIL",
        "t_deadline_s": T,
        "n_maps_train": len(maps),
        "all_admitted": bool(ok),
        "e_ref_range_J": [min(e), max(e)] if e else None,
        "z_ref_J": 550.0,
        "budget_rule": "z0 = k * E_ref(T=45)；E_ref = 参考路线按 v=L/45 跑完的能耗",
        "note": "r13 契约余量版专用；30 s 的冻结报告未改动",
        "maps": maps,
    }
    json.dump(payload, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nverdict=%s e_ref range %s J\nwrote %s" % (payload["verdict"],
          payload["e_ref_range_J"], OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
