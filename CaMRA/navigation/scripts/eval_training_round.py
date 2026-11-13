"""Evaluate one training round on the development split (NAV-064).

Only ``development`` maps are rolled out here; the closing tier is computed from
these numbers and nothing else.  ``budget_selection`` and ``final_*`` are never
loaded, and the report records the actual paths used so the isolation claim can
be audited instead of asserted.

    <py> scripts/eval_training_round.py --round 1 \
        --prereg training_loop_prereg_r1.json \
        --run-ids nav064_r1_uncalibrated_s0,... \
        --out reports/training_round_r1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.vector import VectorNominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

NAV_CONTRACT = os.path.join(NAV, "config", "navigation_contract_v1.json")
SPLIT = os.path.join(NAV, "config", "split_manifest_v1.json")
DESIGN = os.path.join(NAV, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM = os.path.join(NAV, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
E_REF = os.path.join(NAV, "reports", "train_map_reference_energy.json")
WORK_ROOT = "/pub/data/caohy/work/camra_navigation"


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_rollout_ctx(variant: str) -> dict[str, Any]:
    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM)
    if variant == "uncalibrated":
        import dataclasses
        geom = dataclasses.replace(geom, delta_m=0.0)
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    else:
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    return {"design": design, "geom": geom, "cm": cm}


def rollout(net, spec, ctx, z0: float, *, cap: int = 400) -> dict[str, Any]:
    env = VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
    env.reset()
    steps = 0
    min_d = math.inf
    dangers = 0
    while not env._done[0] and steps < cap:
        o = torch.as_tensor(env.observation, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(o, deterministic=True)
        rec = env.step(a.numpy())
        steps += 1
        min_d = min(min_d, float(rec.goal_distance_m[0]))
        dangers += int(bool(rec.danger_violation[0]))
    res = env.result(0)
    label = env._label[0] or "cap_step"
    return {
        "label": label,
        "n_steps": steps,
        "t_s": round(float(res.t_s), 3),
        "min_goal_distance_m": round(float(min_d), 3),
        "cost_model_J": round(float(res.cost_model_J), 2),
        "z0_J": round(z0, 2),
        "z_J_end": round(float(res.z_J), 2),
        "within_budget": bool(res.within_budget),
        "arrived_within_budget": bool(res.arrived_within_budget),
        "danger_steps": dangers,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--prereg", default="training_loop_prereg_r1.json")
    ap.add_argument("--run-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-root", default=WORK_ROOT)
    ap.add_argument("--ckpt-name", default="state.pt",
                    help="checkpoint file inside each run dir; a round whose runs "
                         "died before the final update is evaluated at its last "
                         "saved checkpoint and the report must say so")
    args = ap.parse_args()

    prereg_path = os.path.join(NAV, "config", args.prereg)
    pinned = open(prereg_path + ".sha256", encoding="utf-8").read().strip()
    if sha(prereg_path) != pinned:
        raise SystemExit("prereg hash mismatch; refusing to evaluate")
    prereg = json.load(open(prereg_path, encoding="utf-8"))

    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    split = json.load(open(SPLIT, encoding="utf-8"))
    refs = json.load(open(E_REF, encoding="utf-8"))
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
    dev_maps = split["splits"]["development"]["maps"]
    grid = prereg["round_acceptance"].get("budget_response_on_dev", {}).get(
        "grid_k", [0.7, 0.85, 1.0, 1.15, 1.3])

    runs = []
    for rid in args.run_ids.split(","):
        variant = "uncalibrated" if "uncalibrated" in rid else "full"
        run_dir = os.path.join(args.work_root, rid)
        state_path = os.path.join(run_dir, args.ckpt_name)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        net = NW.build(nav)
        net.load_state_dict(state["model"])
        net.eval()
        ctx = build_rollout_ctx(variant)

        # --- primary: k = 1.0 on the 6 development maps -------------------
        rows = []
        for m in dev_maps:
            spec = M.build(m["level"], m["seed"], ctx["design"])
            r = rollout(net, spec, ctx, e_ref[m["map_id"]])
            rows.append({"map_id": m["map_id"], **r})
        sr = sum(1 for r in rows if r["label"] == "goal_reached") / len(rows)
        sr_wb = sum(1 for r in rows if r["arrived_within_budget"]) / len(rows)

        # --- diagnostic: budget response on development ------------------
        budget_rows = []
        for k in grid:
            cells = []
            for m in dev_maps:
                spec = M.build(m["level"], m["seed"], ctx["design"])
                r = rollout(net, spec, ctx, k * e_ref[m["map_id"]])
                cells.append({"map_id": m["map_id"], "k": k, **r})
            budget_rows.append({
                "k": k,
                "sr": sum(1 for c in cells if c["label"] == "goal_reached") / len(cells),
                "sr_within_budget": sum(1 for c in cells if c["arrived_within_budget"]) / len(cells),
                "cells": cells,
            })

        log_path = os.path.join(run_dir, "train_log.jsonl")
        rows_log = [json.loads(x) for x in open(log_path, encoding="utf-8") if x.strip()]
        start = next((x for x in rows_log if x.get("kind") == "run_start"), {})
        end = next((x for x in rows_log if x.get("kind") == "run_end"), {})
        updates = [x for x in rows_log if x.get("kind") == "update"]
        clip = [u.get("clipped_ratio_fraction", 0.0) for u in updates if "clipped_ratio_fraction" in u]
        runs.append({
            "run_id": rid,
            "variant": variant,
            "seed": state.get("seed"),
            "state_sha256": sha(state_path),
            "checkpoint": args.ckpt_name,
            "updates": len(updates),
            "wall_hours": end.get("wall_hours"),
            "budget_rule": start.get("budget_rule"),
            "z0_J_initial": start.get("z0_J_initial"),
            "obs_version": start.get("obs_version"),
            "schema": start.get("schema"),
            "label_counts": end.get("episode_label_counts"),
            "clip_fraction_mean": float(np.mean(clip)) if clip else None,
            "dev": {"k": 1.0, "sr": sr, "sr_within_budget": sr_wb, "episodes": rows},
            "budget_response": budget_rows,
        })

    summary = {}
    for variant in ("uncalibrated", "full"):
        group = [r for r in runs if r["variant"] == variant]
        srs = sorted(r["dev"]["sr"] for r in group)
        summary[variant] = {
            "seeds": [r["seed"] for r in group],
            "sr_by_seed": {str(r["seed"]): r["dev"]["sr"] for r in group},
            "sr_median": statistics.median(srs) if srs else None,
            "sr_min": min(srs) if srs else None,
            "sr_max": max(srs) if srs else None,
            "n_runs": len(group),
        }
    threshold = float(prereg["round_acceptance"]["dev_success_rate_median"])
    met = {v: (summary[v]["sr_median"] is not None and summary[v]["sr_median"] >= threshold)
           for v in summary}

    report = {
        "schema": "camra-training-round-report-v1",
        "task": "NAV-064",
        "round": args.round,
        "prereg_version": prereg.get("version"),
        "prereg": {"path": os.path.relpath(prereg_path, NAV), "sha256": pinned},
        "gate_g3_alignment_v2": (prereg["binding"].get("gate_g3_alignment_v2")
                                 or prereg["binding"].get("gate_g3")),
        "warm_start": prereg["algorithm"].get("warm_start"),
        "objective_signal_fix": prereg.get("objective_signal_fix"),
        "closing_tier": {
            "criterion": f"development SR median >= {threshold} for both methods",
            "threshold": threshold,
            "by_method": summary,
            "met": met,
            "closing_tier_met": all(met.values()),
        },
        "runs": runs,
        "data_access": {
            "training_maps": sorted({m["map_id"] for m in dev_maps}) and "train split only (see run_start.map_ids)",
            "evaluated_maps": [m["map_id"] for m in dev_maps],
            "budget_selection_loaded": False,
            "final_loaded": False,
            "paths_loaded": [os.path.join(args.work_root, r) for r in args.run_ids.split(",")]
                            + [E_REF, SPLIT],
        },
    }
    out = os.path.join(NAV, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"round": args.round, "closing_tier_met": report["closing_tier"]["closing_tier_met"],
                      "by_method": {k: {"median": v["sr_median"], "by_seed": v["sr_by_seed"]}
                                    for k, v in summary.items()},
                      "out": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
