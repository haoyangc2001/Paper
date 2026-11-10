"""Run the nominal environment and dump its ledger (NAV-018).

This is the runnable companion to ``tests/test_nominal_env.py``: the test proves
the properties, this script produces an inspectable episode.

    <frozen python> CaMRA/navigation/scripts/check_nominal_env.py \
        --level L2_single_detour --preset forward --steps 200 \
        --z0 5000 --json /pub/data/caohy/logs/camra_navigation/nominal_demo.json

Nothing here is a controller.  The presets are fixed command sequences used to
exercise the environment end to end; they say nothing about navigation quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv, preset_actions  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L2_single_detour",
                    choices=["L1_empty", "L2_single_detour", "L3_multi_path"])
    ap.add_argument("--preset", default="forward",
                    choices=["forward", "creep", "turn_drift", "mixed"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--z0", type=float, default=5000.0)
    ap.add_argument("--variant", default="full", choices=["full", "uncalibrated"])
    ap.add_argument("--json", default="")
    ap.add_argument("--trace", default="", help="optional full per-step trace")
    args = ap.parse_args()

    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    model = JCostModel.from_frozen(POWER, MARGINS, variant=args.variant)
    spec = M.build(args.level, args.seed, M.load_design(DESIGN))
    env = NominalNavEnv(spec, geom, model, z0_J=args.z0)
    env.reset(seed=args.seed)

    print(f"map    : {spec.map_id}  seed={spec.seed}  extent={spec.extent}")
    print(f"start  : ({spec.start[0]:+.2f}, {spec.start[1]:+.2f}) yaw={spec.start_yaw:+.3f}")
    print(f"goal   : ({spec.goal[0]:+.2f}, {spec.goal[1]:+.2f})")
    print(f"model  : variant={model.variant} alpha={model.alpha!r} z0={args.z0:.1f} J")
    print(f"preset : {args.preset} x {args.steps} steps (not a controller)")
    print()
    print(f"{'step':>5} {'t_s':>6} {'x':>8} {'y':>8} {'yaw':>7} "
          f"{'vx':>6} {'wz':>6} {'J_step':>8} {'z_J':>9} {'d_goal':>8} {'y':>3} {'label':<15}")

    for _ in range(args.steps):
        if env.done:
            break
        acts = preset_actions(args.preset, 1)
        rec = env.step(acts[0])
        if rec.index % 20 == 0 or rec.done:
            print(f"{rec.index:>5} {rec.t_s:>6.2f} {rec.pose_after[0]:>8.3f} "
                  f"{rec.pose_after[1]:>8.3f} {rec.pose_after[2]:>7.3f} "
                  f"{rec.command.applied[0]:>6.2f} {rec.command.applied[2]:>6.2f} "
                  f"{rec.cost.cost_model_J:>8.3f} {rec.z_J:>9.3f} "
                  f"{rec.goal_distance_m:>8.3f} {int(rec.safety_y):>3} "
                  f"{(rec.label or ''):<15}")

    res = env.result()
    print()
    print("result :")
    for key, value in res.as_dict().items():
        print(f"  {key:<24} {value}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(env.trace(), fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        print(f"\nledger written to {args.json}")
    if args.trace:
        with open(args.trace, "w", encoding="utf-8") as fh:
            for rec in env.steps:
                fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
        print(f"per-step trace written to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
