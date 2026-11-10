"""Build the train / development / evaluation split manifest (NAV-020).

The manifest is a **function of this file and the frozen map design**, not a
hand-written table: every count, geometry hash, layout hash and the manifest hash
are derived.  ``tests/test_splits.py`` regenerates it and requires the committed
file to match, so a hand edit to the JSON is detected immediately.

Why the split axis is what it is
--------------------------------
``maps.generator.build(level, seed, design)`` jitters only the start, the goal
and the initial heading; the islands belong to the level.  So:

* members of different levels are different fields;
* members of the same level at different seeds share the island layout and differ
  in the task instance.

That is a real near-duplicate and it is reported, not hidden: a number measured on
``final_*`` says how the method generalises over starts, goals and budgets on
these three fields, and says nothing about unseen layouts.

Usage:

    <frozen python> CaMRA/navigation/scripts/build_split_manifest.py \
        --out CaMRA/navigation/config/split_manifest_v1.json [--check]
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

from camra_navigation.maps import generator as M  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")

SCHEMA = "camra-navigation-split-manifest-v1"
VERSION = "1.0.0"

LEVELS = ("L1_empty", "L2_single_detour", "L3_multi_path")

#: Seed blocks are globally disjoint and far apart, so a typo that puts two
#: splits in the same range is obvious rather than plausible.
SPLIT_SPEC: dict[str, dict] = {
    "integration_pilot": {
        "purpose": "阶段 2 的接线自检：v2 加载、10/50/200 Hz 时序、复位隔离、计量对账。",
        "seeds": [101],
        "phase": "阶段 2（NAV-024–028）",
        "reporting": "只作接线证据；不得作为任何性能数字，也不进入成功率分母。",
    },
    "train": {
        "purpose": "两组方法的训练集（无校准与完整方法）。",
        "seeds": [1001, 1002, 1003, 1004, 1005, 1006],
        "phase": "阶段 4（NAV-036–039）",
        "reporting": "训练曲线可报告，但不作为最终性能。",
    },
    "development": {
        "purpose": "开发期诊断：短训练策略复查、超参检查、失败模式归因。",
        "seeds": [2001, 2002],
        "phase": "阶段 4–5（NAV-034/035/042）",
        "reporting": "只用于诊断；不得写成最终数字，也不得据此挑选最终地图。",
    },
    "budget_selection": {
        "purpose": "预算搜索：为给定场景参考预算选出可行做法并检验无解比例。",
        "seeds": [3001, 3002, 3003],
        "phase": "阶段 5（NAV-040–045）",
        "reporting": "只报告可行/无解与搜索诊断；不报告导航成功率，也不参与最终对比。",
    },
    "final_nominal": {
        "purpose": "确定性名义环境上的最终评估。",
        "seeds": [4001, 4002],
        "phase": "阶段 5–6（NAV-043/044）",
        "reporting": "最终数字之一；训练冻结后才可运行。",
    },
    "final_clean": {
        "purpose": "Isaac Lab clean 物理（关闭扰动）上的最终评估。",
        "seeds": [5001, 5002],
        "phase": "阶段 5（NAV-045）",
        "reporting": "最终数字之一；训练冻结后才可运行。",
    },
    "final_perturbed": {
        "purpose": "固定策略在预登记扰动下的最终评估。",
        "seeds": [6001, 6002],
        "phase": "阶段 6（NAV-046–050）",
        "reporting": "最终数字之一；扰动 profile 必须在看结果前冻结。",
    },
}

FINAL_SPLITS = ("final_nominal", "final_clean", "final_perturbed")

METHOD_PAIRING = {
    "variants": ["uncalibrated", "full"],
    "rule": (
        "两组必须落在同一 split、同一 seed、同一初态、同一预算规则上；"
        "唯一允许的差别是 alpha（0 与 margins_v1 的精确值）以及由此得到的成本。"
    ),
    "why": "α=0 组是对照，不是「少调一点」；任何其它差异都会让对比失去意义。",
    "z0_owner": "本清单不固定预算数值：训练预算由 NAV-021 登记，搜索预算由 NAV-040–045 决定。",
}

FAILURE_HANDLING = {
    "counted_as_success": ["goal_reached"],
    "counted_as_failure": ["collision", "out_of_bounds", "budget_exhausted", "stagnation"],
    "counted_separately": {
        "timeout": "记为 truncated，单独报数；既不算成功也不算失败，也不从分母里删。",
    },
    "not_a_label": {
        "internal_violation": "与标签正交，只作诊断计数；不得据此改标签或剔除回合。",
        "fall": "名义模型无物理接触，不会发生；物理侧出现即记 fall 并计入失败。",
    },
    "infrastructure_errors": (
        "进程崩溃、NaN 进状态、读不到资源等属基础设施错误：不计入业务分母，"
        "但必须逐条记录并报出条数；不得悄悄丢弃。"
    ),
    "no_resampling": "禁止为凑成功率重跑同一 (split, seed)；重跑必须作为新的 run id 留档，两者都报。",
    "denominator": "分母由本清单的 (split, level, seed) 数量固定，不随结果变化。",
}

LEAKAGE_CONTROLS = [
    "(level, seed) 在全清单内唯一：同一任务实例不得出现在两个 split。",
    "final_* 的地图在训练冻结前不得构建或步进；reports/ 下不得出现 final_* 结果文件。",
    "budget_selection 只允许决定预算；其结果不得用于挑最终地图、改 α/δ 或改方法。",
    "split 归属只由几何与预登记种子决定；不得因策略表现调整归属。",
    "development 的数字不得写成最终数字；这一条是流程约束，代码只能检查文件命名。",
]


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def byte_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layout_hash(spec) -> str:
    """Islands plus extent only: what two seeds of the same level share."""
    payload = {
        "extent": [round(v, 9) for v in spec.extent],
        "islands": sorted([[round(v, 9) for v in isl] for isl in spec.islands]),
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def manifest_payload(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k != "integrity"}


def build_manifest(design_path: str = DESIGN) -> dict:
    design = M.load_design(design_path)
    splits: dict[str, dict] = {}
    seen: dict[tuple[str, int], str] = {}
    duplicates = []
    layouts: dict[str, set] = {level: set() for level in LEVELS}

    for split, spec in SPLIT_SPEC.items():
        maps = []
        for level in LEVELS:
            if level not in [e["level"] for e in design["levels"]]:
                raise KeyError(f"level {level} missing from the design file")
            for seed in spec["seeds"]:
                m = M.build(level, seed, design)
                key = (level, int(seed))
                if key in seen:
                    duplicates.append({"pair": list(key), "splits": [seen[key], split]})
                seen[key] = split
                lh = layout_hash(m)
                layouts[level].add(lh)
                maps.append(
                    {
                        "level": level,
                        "seed": int(seed),
                        "map_id": m.map_id,
                        "geometry_hash": m.geometry_hash(),
                        "layout_hash": lh,
                    }
                )
        splits[split] = {
            "purpose": spec["purpose"],
            "phase": spec["phase"],
            "reporting": spec["reporting"],
            "levels": list(LEVELS),
            "seeds": list(spec["seeds"]),
            "n_maps": len(maps),
            "maps": maps,
        }

    near_dup = {
        "finding": (
            "同一 level 的所有 seed 共享同一岛几何（seed 只扰动起终点与初始航向），"
            "因此同一 level 内任意两张地图是「近重复」而非独立布局。"
        ),
        "distinct_layouts_per_level": {lv: len(h) for lv, h in layouts.items()},
        "consequence": (
            "final_* 上的数字度量的是起终点与预算上的泛化，不是布局泛化；"
            "任何「未见地图」的表述必须限定为「未见任务实例」。"
        ),
        "not_a_leak": (
            "这是设计使然（移动岛可能让场地不可通行），不是数据泄漏；"
            "泄漏控制只要求 (level, seed) 不跨 split 复用。"
        ),
    }

    levels_of = {lv: [m for s in splits.values() for m in s["maps"] if m["level"] == lv] for lv in LEVELS}

    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "owner": "NAV-020",
        "created_at": "2026-11-10T13:40:00+08:00",
        "status": "membership_locked_evaluation_not_started",
        "generator": {
            "factory": "camra_navigation.maps.generator.build(level, seed, design)",
            "design_path": "CaMRA/navigation/config/map_design_v1.json",
            "design_version": str(design["version"]),
            "design_sha256": byte_sha256(design_path),
            "seed_effect": "只扰动 start/goal/start_yaw；岛几何由 level 决定",
            "map_id_rule": "<level.lower()>-s<seed>",
            "geometry_hash_rule": "sha256(canonical{extent,start,start_yaw,goal,islands})",
            "layout_hash_rule": "sha256(canonical{extent,islands})，用于识别近重复",
        },
        "levels": list(LEVELS),
        "splits": splits,
        "exclusivity": {
            "rule": "(level, seed) 在全清单内唯一",
            "n_pairs": len(seen),
            "n_maps_declared": sum(s["n_maps"] for s in splits.values()),
            "duplicates_found": duplicates,
            "holds": not duplicates,
        },
        "near_duplicates": {
            **near_dup,
            "maps_per_level": {lv: len(v) for lv, v in levels_of.items()},
        },
        "method_pairing": METHOD_PAIRING,
        "failure_handling": FAILURE_HANDLING,
        "leakage_controls": LEAKAGE_CONTROLS,
        "locks": {
            "membership_frozen": True,
            "final_evaluated": False,
            "unlock_condition": (
                "final_* 只允许在 NAV-037/039 的训练与 checkpoint 冻结之后运行；"
                "扰动 profile 必须在看结果前冻结。"
            ),
            "forbidden_before_unlock": [
                "构建或步进 final_* 地图",
                "在 reports/ 下写 final_* 结果文件",
                "用 development/budget_selection 的结果替代最终数字",
            ],
        },
    }
    manifest["integrity"] = {
        "manifest_hash": hashlib.sha256(
            canonical(manifest_payload(manifest)).encode("utf-8")
        ).hexdigest(),
        "hash_rule": (
            "sha256 over canonical JSON (sort_keys, compact separators, UTF-8) of the "
            "manifest with the integrity block removed"
        ),
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(NAV_ROOT, "config", "split_manifest_v1.json"))
    ap.add_argument("--check", action="store_true",
                    help="compare against the file instead of writing it")
    args = ap.parse_args()

    manifest = build_manifest()
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not os.path.isfile(args.out):
            print(f"MISSING: {args.out}")
            return 1
        current = open(args.out, encoding="utf-8").read()
        same = current == text
        print(f"{'MATCH' if same else 'DIFFERS'}: {args.out}")
        if not same:
            a = json.loads(current)
            b = manifest
            print("  committed hash:", a.get("integrity", {}).get("manifest_hash"))
            print("  rebuilt   hash:", b["integrity"]["manifest_hash"])
        return 0 if same else 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"manifest written: {args.out}")
    print(f"  splits      : {len(manifest['splits'])}")
    print(f"  maps        : {manifest['exclusivity']['n_maps_declared']}")
    print(f"  duplicates  : {len(manifest['exclusivity']['duplicates_found'])}")
    print(f"  manifest hash: {manifest['integrity']['manifest_hash']}")
    for name, split in manifest["splits"].items():
        print(f"  {name:<20} {split['n_maps']:>3} maps  seeds={split['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
