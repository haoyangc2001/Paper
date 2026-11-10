"""Budget fairness check for NAV-058.

Two things are produced:

1. the per-map reference energy ``E_ref`` range that anchors the frozen budget
   grid, recomputed over every map in the split manifest;
2. a hand example that isolates the accounting effect from the behavioural one.

The example runs one fixed open-loop command sequence three times: the
uncalibrated variant at ``z0``, the full variant at the same ``z0``, and the full
variant at ``(1+alpha)*z0``.  Because alpha never enters the dynamics, the poses
must agree bit-for-bit over the common prefix; only the step at which the internal
budget is exhausted can move.  That is what makes "BVR improved" an empty claim
when nothing but ``z0`` changed.

Run:

    <frozen python> CaMRA/navigation/scripts/check_budget_fairness.py \
        --json CaMRA/navigation/reports/budget_fairness_check.json
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

from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.maps import feasibility as F  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
MANIFEST = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")

GRID_K = (0.7, 0.85, 1.0, 1.15, 1.3)
CMD = (0.6, 0.0, 0.0)
N_STEPS = 100
MECHANICS_MAP = ("L1_empty", 101)  # integration_pilot: wiring only, not a result


def e_ref_for(spec, geom, power) -> float:
    """Alpha-free reference energy of the admitted route at the deadline speed."""
    rec = F.check_map(spec, geom, power, resolution=0.15)
    if not rec["admitted"]:
        raise RuntimeError(f"{spec.map_id} is not admissible: {rec['reasons']}")
    return float(rec["route"]["predicted_energy_deadline_J"])


def run_sequence(level, seed, design, geom, variant, z0, n_steps):
    model = JCostModel.from_frozen(POWER, MARGINS, variant=variant)
    spec = M.build(level, seed, design)
    env = NominalNavEnv(spec, geom, model, z0_J=z0)
    poses, costs, zs = [], [], []
    for _ in range(n_steps):
        if env.done:
            break
        rec = env.step(CMD)
        poses.append(rec.pose_after)
        costs.append((rec.cost.cost_nom_J, rec.cost.cost_model_J))
        zs.append(rec.z_J)
    return {
        "variant": variant,
        "z0_J": z0,
        "alpha": model.alpha,
        "steps_completed": env.index,
        "label": env.label,
        "final_z_J": env.budget.z_J,
        "poses": poses,
        "costs": costs,
        "z_series": zs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    power = PowerModel.from_dict(json.load(open(POWER, encoding="utf-8")))
    manifest = json.load(open(MANIFEST, encoding="utf-8"))

    # --- 1. E_ref over every manifest map ---------------------------------
    per_level: dict[str, list[float]] = {}
    for body in manifest["splits"].values():
        for m in body["maps"]:
            spec = M.build(m["level"], int(m["seed"]), design)
            e = e_ref_for(spec, geom, power)
            per_level.setdefault(m["level"], []).append(e)
    all_e = [v for vals in per_level.values() for v in vals]
    e_ref_block = {
        "n_maps": len(all_e),
        "range_J": [min(all_e), max(all_e)],
        "per_level_range_J": {
            lv: [min(v), max(v)] for lv, v in per_level.items()
        },
        "grid_J_example_level_L2": [k * per_level["L2_single_detour"][0] for k in GRID_K],
        "note": "由脚本按 NAV-017 的准入路线重算，与任何策略无关",
    }

    # --- 2. hand example --------------------------------------------------
    full_alpha = JCostModel.from_frozen(POWER, MARGINS, variant="full").alpha
    one_plus = 1.0 + full_alpha
    c_nom_step = float(power.step_power([list(CMD), list(CMD)])[0]) * 0.1
    z0_base = 105.0 * c_nom_step          # 5 nominal steps of slack

    base = run_sequence(*MECHANICS_MAP, design, geom, "uncalibrated", z0_base, N_STEPS)
    same_budget = run_sequence(*MECHANICS_MAP, design, geom, "full", z0_base, N_STEPS)
    raised = run_sequence(*MECHANICS_MAP, design, geom, "full", one_plus * z0_base, N_STEPS)

    common = min(base["steps_completed"], same_budget["steps_completed"], raised["steps_completed"])
    prefixes_equal = (
        base["poses"][:common] == same_budget["poses"][:common] == raised["poses"][:common]
    )
    ratios = []
    for nom, mod in same_budget["costs"][:common]:
        if nom > 0.0:
            ratios.append(mod / nom)
    ratio_spread = (max(ratios) - min(ratios)) if ratios else None
    ratio_value = ratios[0] if ratios else None

    example = {
        "setup": {
            "level": MECHANICS_MAP[0],
            "seed": MECHANICS_MAP[1],
            "command": list(CMD),
            "n_steps_requested": N_STEPS,
            "c_nom_per_step_J": c_nom_step,
            "z0_J": z0_base,
            "one_plus_alpha": one_plus,
            "z0_raised_J": one_plus * z0_base,
        },
        "runs": [
            {"variant": r["variant"], "z0_J": r["z0_J"], "alpha": r["alpha"],
             "steps_completed": r["steps_completed"], "label": r["label"],
             "final_z_J": r["final_z_J"]}
            for r in (base, same_budget, raised)
        ],
        "prefix_steps_compared": common,
        "prefix_poses_bitwise_equal": prefixes_equal,
        "per_step_model_over_nominal_ratio": {
            "value": ratio_value,
            "spread_over_prefix": ratio_spread,
            "expected": one_plus,
        },
        "interpretation": (
            "三段运行的前 %d 步位姿逐位相同、逐步名义成本相同：α 不改变动力学。"
            "唯一变化的是内部预算耗尽发生的步数。因此「同 z0 下完整方法更早停止」"
            "是记账效应；把 z0 提高到 (1+α) 倍后停止点回到原处，而轨迹与能量都没有变。"
            % common
        ),
        "conclusion": (
            "BVR（或任何以 z0 为参照的指标）在只改 z0 时必然改变，"
            "所以它不能单独作为能耗改善或方法优劣的证据。"
        ),
    }

    # --- 3. monotonicity is measured, not assumed -------------------------
    grid_probe = []
    for k in GRID_K:
        r = run_sequence(*MECHANICS_MAP, design, geom, "full", k * 105.0 * c_nom_step, N_STEPS)
        grid_probe.append({"k_rel_to_base": k, "z0_J": k * 105.0 * c_nom_step,
                           "steps_completed": r["steps_completed"], "label": r["label"]})
    steps = [g["steps_completed"] for g in grid_probe]
    monotone = all(b >= a for a, b in zip(steps, steps[1:]))

    # --- 4. the trap the contract forbids ---------------------------------
    # Reporting model cost across methods with different z0 would look like an
    # energy difference even though the trajectory prefix is identical.
    trap = {
        "model_cost_at_same_z0_J": sum(c[1] for c in same_budget["costs"]),
        "model_cost_at_raised_z0_J": sum(c[1] for c in raised["costs"]),
        "nominal_cost_at_same_z0_J": sum(c[0] for c in same_budget["costs"]),
        "nominal_cost_at_raised_z0_J": sum(c[0] for c in raised["costs"]),
        "comparison_bases": {
            "steps_at_same_z0": same_budget["steps_completed"],
            "steps_at_raised_z0": raised["steps_completed"],
        },
        "why_forbidden": (
            "两列的比较基数不同（%d 步对 %d 步）：把总和相减得到的差值只是"
            "「多跑了 %d 步」，不是效率差异"
            % (
                same_budget["steps_completed"],
                raised["steps_completed"],
                raised["steps_completed"] - same_budget["steps_completed"],
            )
        ),
    }

    report: dict[str, Any] = {
        "task": "NAV-058",
        "schema": "camra-navigation-budget-fairness-check-v1",
        "contract": "config/budget_comparison_contract_v1.json",
        "contract_version": "1.0.0",
        "verdict": "PASS" if prefixes_equal and ratio_spread is not None
        and ratio_spread < 1e-12 and monotone else "FAIL",
        "reference_energy": e_ref_block,
        "hand_example": example,
        "monotonicity_probe": {
            "grid": grid_probe,
            "non_decreasing": monotone,
            "note": "本探针只证明该固定序列下单调；契约不假定它普遍成立，正式搜索须逐图记录",
        },
        "forbidden_comparison_demo": trap,
        "caveats": [
            "名义环境 physical_status=not_measured，因此示例里没有实测 J；BVR 只适用于物理评估。",
            "示例为开环固定命令序列；闭环策略下 z_ratio 进入观测，α 会改变动作，届时记账与行为无法分开。",
            "E_ref 是路线名义成本，不是可达性证明，也不随方法变化。",
        ],
    }

    print(json.dumps({k: report[k] for k in ("task", "verdict")}, ensure_ascii=False))
    print(f"E_ref over {e_ref_block['n_maps']} maps: "
          f"{e_ref_block['range_J'][0]:.2f} .. {e_ref_block['range_J'][1]:.2f} J")
    print(f"steps completed: uncal@z0={base['steps_completed']}, "
          f"full@z0={same_budget['steps_completed']}, full@(1+a)z0={raised['steps_completed']}")
    print(f"prefix equal: {prefixes_equal} over {common} steps | "
          f"model/nominal ratio {ratio_value!r} spread {ratio_spread}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        print(f"report written to {args.json}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
