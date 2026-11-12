"""NAV-041: freeze and execute the budget-selection protocol.

The budget comparison contract has two modes.  This task owns exactly one of
them -- ``method_selected_budget`` -- on the *independent* ``budget_selection``
split (9 maps, seeds 3001-3003).  The selection set never takes part in actor
training or in the formal performance estimate; the numbers printed here are
feasibility/search diagnostics, **not** navigation success rates.

Two modes of this script:

``--freeze``
    Write ``config/budget_selection_v1.json`` (+ ``.sha256``) with the exact
    hashes of every input, the grid, the feasibility criterion, the search
    rule, the no-solution rule, the quota and the β/z_ref/unit semantics.
    Refuses to overwrite: a frozen protocol is immutable; a change needs a new
    file (and, if it changes the standard, the user's decision).

``(default)``
    Verify the frozen config hash, run one deterministic rollout per
    ``(method, map, k)`` -- 9 x 2 x 5 = 90 business episodes, the authorised
    quota -- and write ``reports/budget_selection.json`` with every row, the
    per-(method, map) selection, the no-solution ratios, the monotonicity
    findings and a recompute block.

Why measured labels and not the critic
--------------------------------------
NAV-039 settled the downstream route (O-3): the value-based search is not used
for budget selection; the criterion is the measured terminal label
(``goal_reached``) required by the contract's ``feasibility_requirement``.
The critic stays frozen and unused here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation import observation as OBS  # noqa: E402
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.vector import VectorNominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

CONFIG_PATH = os.path.join(NAV_ROOT, "config", "budget_selection_v1.json")
CONFIG_SHA = CONFIG_PATH + ".sha256"
SPLIT = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
E_REF_REPORT = os.path.join(NAV_ROOT, "reports", "budget_selection_reference_energy.json")
BUNDLE = os.path.join(NAV_ROOT, "delivery", "deployment_candidates_v1.json")
NAV_CONTRACT = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
BUDGET_CONTRACT = os.path.join(NAV_ROOT, "config", "budget_comparison_contract_v1.json")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")
EXPERIMENT_PREREG = os.path.join(NAV_ROOT, "config", "experiment_prereg_v1.json")
DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
OBS_MODULE = os.path.join(NAV_ROOT, "src", "camra_navigation", "observation.py")

WORK_ROOT = "/pub/data/caohy/work/camra_navigation"
STEP_CAP = 400          # safety stop only; the env's own timeout is 30 s = 300 steps
FEASIBLE_LABEL = "goal_reached"
METHODS = ("uncalibrated", "full")
VARIANT_OF = {"uncalibrated": "uncalibrated", "full": "full"}


# --------------------------------------------------------------------- utils
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha(sd: dict[str, torch.Tensor]) -> str:
    """Byte-identical to ``scripts/freeze_deployment.py`` (checked below)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_LAYOUT: NW.ObservationLayout | None = None


def z_ratio_slice() -> slice:
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = NW.ObservationLayout.from_contract(load_json(NAV_CONTRACT))
    return _LAYOUT.slice_of("z_ratio")


# ----------------------------------------------------------- pure rule layer
def feasible_from_label(label: str | None, within_budget: bool | None = None) -> bool:
    """Contract criterion: the episode reached the goal **inside the budget**.

    Before NAV-062 ``z <= 0`` was itself a terminal, so ``label == goal_reached``
    already implied "arrived first".  Budget exhaustion no longer ends an
    episode (the reference implementation's ``is_terminal`` only looks at the
    horizon), so the arrival label alone can no longer carry that meaning: an
    episode may arrive *after* overspending.  The conjunction is therefore
    explicit, and a call that omits ``within_budget`` is a programming error
    rather than a silently relaxed criterion.

    Danger exposure stays recorded but is not a nominal disqualifier: the
    forbidden band is δ's job, not the feasibility label's.
    """
    if label != FEASIBLE_LABEL:
        return False
    if within_budget is None:
        raise ValueError(
            "within_budget is required: arrival alone is not feasibility since "
            "budget exhaustion stopped being terminal (NAV-062)")
    return bool(within_budget)


def monotonic_violations(rows: Sequence[dict]) -> list[dict]:
    """Feasibility must be non-decreasing along increasing budget."""
    first = next((i for i, r in enumerate(rows) if r["feasible"]), None)
    if first is None:
        return []
    return [
        {"k": rows[i]["k"], "z0_J": rows[i]["z0_J"],
         "after_feasible_k": rows[first]["k"]}
        for i in range(first + 1, len(rows))
        if not rows[i]["feasible"]
    ]


def select_on_grid(
    rows: Sequence[dict], *, grid: Sequence[float], e_ref_J: float, z_ref_J: float
) -> dict:
    """Frozen rule: walk the grid upward, take the first *measured* feasible k.

    Never interpolates, never falls back to a larger k when non-monotonic, and
    never expands the grid.  Every grid point must be present: the quota funds
    the full grid precisely so monotonicity can be inspected.
    """
    by_k = {round(float(r["k"]), 10): r for r in rows}
    missing = [k for k in grid if round(float(k), 10) not in by_k]
    if missing:
        raise ValueError(f"grid points not measured: {missing}")
    ordered = [by_k[round(float(k), 10)] for k in grid]
    violations = monotonic_violations(ordered)
    for k, row in zip(grid, ordered):
        if row["feasible"]:
            z = float(k) * float(e_ref_J)
            return {
                "status": "found",
                "k": float(k),
                "z_J": z,
                "z_ratio": z / float(z_ref_J),
                "z_ratio_clipped": float(np.clip(
                    z / float(z_ref_J), -OBS.z_ratio_clip(), OBS.z_ratio_clip())),
                "evidence_label": row["label"],
                "monotonic": not violations,
                "monotonicity_violations": violations,
                "note": "selected k is a measured point; budget reported in J "
                        "separately from E_ref and from z_ratio",
            }
    return {
        "status": "no_feasible_budget",
        "k": None, "z_J": None, "z_ratio": None, "z_ratio_clipped": None,
        "evidence_label": None,
        "monotonic": not violations,
        "monotonicity_violations": violations,
        "note": "all grid points infeasible; kept in the map denominator, no "
                "larger k / other map / other seed substituted",
    }


# ------------------------------------------------------------------ freezing
def build_config() -> dict:
    split = load_json(SPLIT)
    bs = split["splits"]["budget_selection"]
    refs = load_json(E_REF_REPORT)
    if refs.get("verdict") != "PASS" or refs.get("n_maps") != 9:
        raise SystemExit(f"bad E_ref evidence: verdict={refs.get('verdict')!r} "
                         f"n_maps={refs.get('n_maps')}")
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
    bundle = load_json(BUNDLE)
    nav = load_json(NAV_CONTRACT)
    bc = load_json(BUDGET_CONTRACT)
    vc = load_json(VALUE_CONTRACT)
    ep = load_json(EXPERIMENT_PREREG)

    z_ref = float(OBS.Z_REF_J)                      # observation.py owns the constant
    contract_z_ref = float(vc["quantities"]["Z_REF_J"]["value"])
    if abs(z_ref - contract_z_ref) > 1e-9:
        raise SystemExit(
            f"Z_REF_J drift: observation.py={z_ref} vs value contract={contract_z_ref}")

    grid = [float(k) for k in bc["budget_grid"]["k_values"]]
    chosen = {}
    for method in METHODS:
        cands = sorted((c for c in bundle["candidates"] if c["method"] == method),
                       key=lambda c: c["seed"])
        if not cands:
            raise SystemExit(f"no deployment candidate for method {method!r}")
        c = cands[0]
        chosen[method] = {
            "method": method,
            "candidate_id": c["candidate_id"],
            "seed": int(c["seed"]),
            "source_run_id": c["source_run_id"],
            "actor_sha256": c["weights"]["actor_sha256"],
            "source_checkpoint_sha256": c["source_checkpoint_sha256"],
        }

    maps = [{"map_id": m["map_id"], "level": m["level"], "seed": int(m["seed"]),
             "e_ref_J": e_ref[m["map_id"]], "geometry_hash": m["geometry_hash"]}
            for m in bs["maps"]]

    return {
        "schema": "camra-budget-selection-v1",
        "version": "1.0.0",
        "owner": "NAV-041",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "status": "frozen_before_search",
        "task": "NAV-041 在独立预算选择集冻结选择规则",
        "mode": {
            "label": "method_selected_budget",
            "contract": "budget_comparison_contract_v1.modes.method_selected_budget",
            "not_common_budget": "本文件不覆盖 common_budget 模式：该模式仍需同一契约下"
                                 "独立执行、独立分母；两模式的标签、分母与样本不得合并，"
                                 "也不得互相作为证据",
            "split_reporting_rule": "budget_selection 只报告可行/无解与搜索诊断；"
                                    "本任务的任何计数都不是导航成功率，也不参与最终对比",
        },
        "units": {
            "budget": "物理焦耳 J：z0 = k · E_ref(map)；E_ref 是 NAV-017 准入路线的名义"
                      "成本，只由几何与冻结功率模型决定，与策略/裕度/结果无关",
            "selected_budget": "所选 z*_m(map)，单位 J（正数）",
            "normalised": "z_ratio = clip(z / Z_REF_J, -1, 1)，Z_REF_J 见 inputs；"
                          "它只是归一化尺度，不是可达预算，不得当作预算报告",
            "cost_model_J": "回合内部记账累计 Σ(1+α)·c_nom·0.1 s（uncalibrated α=0；"
                            "full α=0.19048279965649906）",
            "not_a_unit": [
                "z_ratio 不是预算",
                "c_model 不是实测能量（名义环境无 200 Hz 机械功）",
                "budget_exhausted 是业务失败标签，与实测超预算不是同一件事",
            ],
        },
        "thresholds": {
            "beta": {
                "value": 0.0,
                "meaning": "value_objective 的可行性阈值：残差判定可行当且仅当 W <= -β；"
                           "β>0 表示要求留出裕度；它是数值可行性容差，不是安全裕度",
                "role_here": "只登记含义。本任务用契约的 nominal_env_criterion（终止标签）"
                             "而不是 β 判据，β 不参与选择，也未改动",
            },
            "alpha": {"value": 0.19048279965649906,
                      "role": "full 的成本记账倍数 1+α，只影响内部预算消耗速度"},
            "delta_m": {"role": "full 的禁入带宽（几何），只影响可行空间"},
            "feasibility_criterion": "label == goal_reached AND within_budget；单一终止标签已排除 "
                                     "collision / out_of_bounds / stagnation / timeout",
            "danger_steps_not_disqualifying": "危险步数照实记录，但不构成名义不可行——"
                                              "禁入带由 δ 进入几何，与可行性标签分开",
        },
        "inputs": {
            "split_manifest": {"path": "config/split_manifest_v1.json",
                               "sha256": file_sha256(SPLIT),
                               "split": "budget_selection", "n_maps": len(maps),
                               "maps": maps},
            "e_ref_report": {
                "path": "reports/budget_selection_reference_energy.json",
                "sha256": file_sha256(E_REF_REPORT),
                "note": "同一冻结代码路径（maps.feasibility.check_map，resolution 0.15）"
                        "为预算选择集重算；与 reports/train_map_reference_energy.json "
                        "同源不同文件，后者保持逐字节不变以免其他任务的散列漂移",
            },
            "deployment_bundle": {"path": "delivery/deployment_candidates_v1.json",
                                  "sha256": file_sha256(BUNDLE),
                                  "n_candidates": len(bundle["candidates"])},
            "contracts": {
                "navigation": {"version": nav["version"],
                               "sha256": file_sha256(NAV_CONTRACT)},
                "budget_comparison": {"version": bc["version"],
                                      "sha256": file_sha256(BUDGET_CONTRACT)},
                "value_objective": {"version": vc["version"],
                                    "sha256": file_sha256(VALUE_CONTRACT)},
            },
            "experiment_prereg": {"path": "config/experiment_prereg_v1.json",
                                  "sha256": file_sha256(EXPERIMENT_PREREG),
                                  "selection_block": ep["selection"]},
            "frozen_calibration": {
                "footprint": file_sha256(FOOTPRINT),
                "margins": file_sha256(MARGINS),
                "power_model": file_sha256(POWER),
                "geometry_contract": file_sha256(GEOM_CONTRACT),
                "map_design": file_sha256(DESIGN),
            },
            "z_ref": {"value": z_ref, "owner": "src/camra_navigation/observation.py",
                      "sha256": file_sha256(OBS_MODULE)},
        },
        "actor_binding": {
            "rule": "每种方法取部署清单中 seed 最小（=0）的候选；该选择只由冻结索引规则"
                    "决定，与任何表现无关",
            "why_one_seed": "授权配额 90 回合 = 9 图 × 2 方法 × 5 网格点，"
                            "每 (method, map, k) 恰好一次；三种子需要 270 回合，超出配额",
            "consequence_limitation": "选择按单一种子报告，跨种子离散度在本配额下不可测；"
                                      "结论只能说「该候选在预算选择集上无可行预算」，"
                                      "不得外推为「该方法在其所有种子上都无解」",
            "candidates": chosen,
            "deterministic_transform": "policy mean（无采样、无噪声）→ action_to_requested "
                                       "→ clip_applied",
            "normalisation": "identity（obs-v2 已归一化）",
        },
        "grid": {
            "k_values": grid,
            "rule": "z0 = k · E_ref(map)",
            "frozen": True,
            "may_not": ["增删或加密网格点", "为某一方法平移网格", "看最终集结果后改网格"],
            "n_points": len(grid),
        },
        "feasibility": {
            "criterion": "label == goal_reached AND z_J >= 0 at the end",
            "identical_for_both_methods": True,
            "measured_not_critic": "用实测终止标签，不用 critic 的 value 搜索"
                                   "（NAV-039 的 O-3 决议）",
            "no_retry": "业务失败不重试；同一 (method, map, k) 不因结果重跑",
        },
        "search_rule": {
            "order": "沿 k_values 从 0.70 递增，取第一个满足 feasibility 的 k",
            "answer_must_be_measured": "所选 k 必须是实际跑过且判为可行的点，禁止插值",
            "grid_points_all_measured": "五个点全部实测，否则无法检查单调性",
            "no_solution": "五点全不可行 -> status=no_feasible_budget；保留在地图分母里，"
                           "禁止换更大 k / 换图 / 换种子 / 看 final 结果手填",
            "monotonicity": "逐 (method, map) 检查可行集合是否单调；非单调照实记录，"
                            "不得据此改选更大的 k",
        },
        "quota": {
            "business_episodes": len(maps) * len(METHODS) * len(grid),
            "derivation": f"{len(maps)} 图 × {len(METHODS)} 方法 × {len(grid)} 网格点",
            "authorized_total": int(ep["selection"]["total_episodes"]),
            "wall_clock_cap_hours": float(ep["selection"]["wall_clock_cap_hours"]),
            "retries": 0,
        },
        "saturation_disclosure": {
            "rule": "z0 > Z_REF_J 时观测里的 z_ratio 被裁到 1.0，策略前半程看不到富余；"
                    "逐行标记 z0_above_z_ref，照实披露，不得据此修改网格或 Z_REF",
            "affected_k": [k for k in grid if k * max(m["e_ref_J"] for m in maps) > z_ref],
        },
        "recompute": {
            "how": "报告逐行给出 (method, map, k, z0_J, label)；feasible 由 label 复算，"
                   "所选 k 由「从低到高的第一个 feasible」复算，"
                   "z_J = k·E_ref 与 z_ratio = z/Z_REF 由输入散列锁定后复算",
            "locked_by": "本文件 .sha256 + inputs 各散列",
        },
        "forbidden": [
            "用 budget_selection 的数字充当导航成功率或最终对比证据",
            "在任何 (method, map, k) 上因结果重跑或补跑",
            "看 final_* 结果后回填或改选预算",
            "把 z_ratio 当预算、把 c_model 当实测能量、把 budget_exhausted 当实测超预算",
            "非单调时悄悄改选更大的 k",
        ],
    }


def cmd_freeze() -> int:
    if os.path.exists(CONFIG_PATH):
        raise SystemExit(
            f"refusing to overwrite frozen {CONFIG_PATH}; a protocol change needs a new "
            "file (and the user's decision if the standard changes)")
    cfg = build_config()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    sha = file_sha256(CONFIG_PATH)
    with open(CONFIG_SHA, "w", encoding="utf-8") as f:
        f.write(sha + "\n")
    print(json.dumps({"frozen": os.path.basename(CONFIG_PATH), "sha256": sha,
                      "n_maps": cfg["inputs"]["split_manifest"]["n_maps"],
                      "grid": cfg["grid"]["k_values"],
                      "quota": cfg["quota"]["business_episodes"]}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ running
def verify_frozen() -> tuple[dict, str]:
    if not os.path.exists(CONFIG_SHA):
        raise SystemExit("missing .sha256; run --freeze first")
    expected = open(CONFIG_SHA, encoding="utf-8").read().strip()
    actual = file_sha256(CONFIG_PATH)
    if actual != expected:
        raise SystemExit(f"frozen config hash mismatch: {actual} != {expected}")
    cfg = load_json(CONFIG_PATH)
    if cfg["status"] != "frozen_before_search":
        raise SystemExit(f"unexpected config status {cfg['status']!r}")
    return cfg, actual


def build_context(variant: str):
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    if variant == "uncalibrated":
        geometry = dataclasses.replace(geometry, delta_m=0.0)
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    else:
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    return geometry, cm


def load_actor(candidate: dict, nav_contract: dict):
    run_id = candidate["source_run_id"]
    refit = os.path.join(WORK_ROOT, run_id, "deploy_refit", "state.pt")
    src = os.path.join(WORK_ROOT, run_id, "state.pt")
    path = refit if os.path.exists(refit) else src
    if not os.path.exists(path):
        raise SystemExit(f"missing checkpoint for {candidate['candidate_id']}: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["model"]
    actor_sd = {k: v for k, v in sd.items() if k.startswith("policy.")}
    got = state_dict_sha(actor_sd)
    if got != candidate["actor_sha256"]:
        raise SystemExit(
            f"actor hash mismatch for {candidate['candidate_id']}: {got} != "
            f"{candidate['actor_sha256']} (file {path})")
    net = NW.build(nav_contract)
    net.load_state_dict(sd)
    net.eval()
    return net, {"state_path": path, "actor_sha256": got}


def rollout(net, spec, geometry, cm, z0_J: float) -> dict:
    env = VectorNominalNavEnv(spec, geometry, cm, n_envs=1, z0_J=z0_J)
    env.reset()
    z_ratio_start = float(env.observation[0][z_ratio_slice()][0])
    steps = 0
    min_d = math.inf
    danger = 0
    internal = 0
    while not env._done[0] and steps < STEP_CAP:
        o = torch.as_tensor(env.observation, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(o, deterministic=True)
        rec = env.step(a.numpy())
        steps += 1
        min_d = min(min_d, float(rec.goal_distance_m[0]))
        danger += int(bool(rec.danger_violation[0]))
        internal += int(bool(rec.internal_violation[0]))
    res = env.result(0)
    label = env._label[0] or "cap_step"
    return {
        "label": label,
        "within_budget": bool(res.within_budget),
        "over_budget_J": round(float(res.over_budget_J), 3),
        "n_steps": steps,
        "t_s": round(float(res.t_s), 3),
        "cost_model_J": round(float(res.cost_model_J), 3),
        "z_J_end": round(float(res.z_J), 3),
        "travelled_m": round(float(res.travelled_m), 3),
        "goal_distance_m": round(float(res.goal_distance_m), 3),
        "min_goal_distance_m": round(float(min_d), 3),
        "z_ratio_start": round(z_ratio_start, 6),
        "danger_steps": danger,
        "internal_violation_steps": internal,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NAV-041 budget selection")
    ap.add_argument("--freeze", action="store_true", help="write the frozen config")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.freeze:
        return cmd_freeze()

    cfg, cfg_sha = verify_frozen()
    torch.set_num_threads(1)
    nav_contract = load_json(NAV_CONTRACT)
    design = M.load_design(DESIGN)
    z_ref = float(cfg["inputs"]["z_ref"]["value"])
    grid = [float(k) for k in cfg["grid"]["k_values"]]
    maps = cfg["inputs"]["split_manifest"]["maps"]

    actors = {}
    for method in METHODS:
        cand = cfg["actor_binding"]["candidates"][method]
        net, info = load_actor(cand, nav_contract)
        actors[method] = (net, info)

    t0 = time.time()
    rows: list[dict] = []
    for m in maps:
        spec = M.build(m["level"], m["seed"], design)
        if spec.map_id != m["map_id"]:
            raise SystemExit(f"map id mismatch {spec.map_id} != {m['map_id']}")
        got_hash = spec.geometry_hash()
        if got_hash != m["geometry_hash"]:
            raise SystemExit(
                f"geometry hash mismatch for {m['map_id']}: {got_hash} != "
                f"{m['geometry_hash']} (the frozen split names a different map)")
        for method in METHODS:
            net, _ = actors[method]
            geometry, cm = build_context(VARIANT_OF[method])
            for k in grid:
                z0 = k * float(m["e_ref_J"])
                out = rollout(net, spec, geometry, cm, z0)
                rows.append({
                    "method": method,
                    "map_id": m["map_id"],
                    "level": m["level"],
                    "map_seed": m["seed"],
                    "k": k,
                    # full precision on purpose: the recompute check multiplies
                    # k by e_ref and must agree exactly, rounding both sides
                    # independently would fake a mismatch of ~1e-3
                    "z0_J": z0,
                    "z_ratio": z0 / z_ref,
                    # saturation guard: the encoded z_ratio is clipped at
                    # clip*Z_REF (obs-v3: 1.5), not at Z_REF
                    "z0_above_z_ref": bool(z0 > z_ref),
                    "z0_above_clip": bool(z0 > z_ref * OBS.z_ratio_clip()),
                    "e_ref_J": float(m["e_ref_J"]),
                    "feasible": feasible_from_label(out["label"], out["within_budget"]),
                    **out,
                })
        print(f"  {m['map_id']} done", flush=True)
    wall = time.time() - t0

    per_map: list[dict] = []
    for m in maps:
        for method in METHODS:
            sub = [r for r in rows if r["method"] == method and r["map_id"] == m["map_id"]]
            sel = select_on_grid(sub, grid=grid, e_ref_J=m["e_ref_J"], z_ref_J=z_ref)
            per_map.append({
                "method": method,
                "map_id": m["map_id"],
                "level": m["level"],
                "e_ref_J": round(float(m["e_ref_J"]), 3),
                "candidates_measured": len(sub),
                "feasible_candidates": [r["k"] for r in sub if r["feasible"]],
                "selection": sel,
            })

    summary: dict[str, Any] = {"mode_label": cfg["mode"]["label"], "per_method": {}}
    for method in METHODS:
        recs = [p for p in per_map if p["method"] == method]
        found = [p for p in recs if p["selection"]["status"] == "found"]
        zs = [p["selection"]["z_J"] for p in found]
        ks = [p["selection"]["k"] for p in found]
        viol = [{"map_id": p["map_id"], **v}
                for p in recs for v in p["selection"]["monotonicity_violations"]]
        summary["per_method"][method] = {
            "n_maps": len(recs),
            "n_maps_with_feasible_budget": len(found),
            "n_maps_no_feasible_budget": len(recs) - len(found),
            "no_solution_share": round((len(recs) - len(found)) / len(recs), 4),
            "candidate_rows_measured": sum(p["candidates_measured"] for p in recs),
            "candidate_rows_feasible": sum(len(p["feasible_candidates"]) for p in recs),
            "selected_k": {"min": min(ks) if ks else None,
                           "median": float(np.median(ks)) if ks else None,
                           "max": max(ks) if ks else None},
            "selected_J": {"min": round(min(zs), 2) if zs else None,
                           "median": round(float(np.median(zs)), 2) if zs else None,
                           "max": round(max(zs), 2) if zs else None},
            "monotonicity_violations": viol,
            "saturated_rows": sum(1 for r in rows
                                  if r["method"] == method and r["z0_above_z_ref"]),
        }
    summary["all_no_solution"] = all(
        summary["per_method"][m]["n_maps_with_feasible_budget"] == 0 for m in METHODS)

    # ---- recompute block: selection must follow from the rows alone ----
    mismatches: list[str] = []
    for p in per_map:
        sub = [r for r in rows if r["method"] == p["method"] and r["map_id"] == p["map_id"]]
        again = select_on_grid(sub, grid=grid,
                               e_ref_J=[m for m in maps
                                        if m["map_id"] == p["map_id"]][0]["e_ref_J"],
                               z_ref_J=z_ref)
        if again["status"] != p["selection"]["status"] or again["k"] != p["selection"]["k"]:
            mismatches.append(f"{p['method']}/{p['map_id']}")
        for r in sub:
            if r["feasible"] != feasible_from_label(r["label"], r["within_budget"]):
                mismatches.append(f"{p['method']}/{p['map_id']}/k={r['k']}: label")
            if abs(r["z0_J"] - r["k"] * r["e_ref_J"]) > 1e-6:
                mismatches.append(f"{p['method']}/{p['map_id']}/k={r['k']}: z0")

    report = {
        "task": "NAV-041",
        "schema": "camra-budget-selection-report-v1",
        "verdict": "FROZEN_RULE_EXECUTED",
        "mode_label": cfg["mode"]["label"],
        "not_common_budget": cfg["mode"]["not_common_budget"],
        "split_reporting_rule": cfg["mode"]["split_reporting_rule"],
        "config": {"path": "config/budget_selection_v1.json", "sha256": cfg_sha},
        "inputs_sha256": {
            **{k: v["sha256"] for k, v in cfg["inputs"].items()
               if isinstance(v, dict) and "sha256" in v},
            **{f"contract.{k}": v["sha256"]
               for k, v in cfg["inputs"]["contracts"].items()},
            **{f"frozen.{k}": v
               for k, v in cfg["inputs"]["frozen_calibration"].items()},
        },
        "actors": {m: actors[m][1] for m in METHODS},
        "grid": grid,
        "z_ref_J": z_ref,
        "n_episodes": len(rows),
        "wall_seconds": round(wall, 2),
        "rows": rows,
        "per_method_map": per_map,
        "summary": summary,
        "recompute": {
            "selection_recomputed_from_rows": True,
            "feasibility_recomputed_from_label": True,
            "z0_recomputed_from_k_times_e_ref": True,
            "mismatches": mismatches,
            "clean": not mismatches,
        },
        "units_note": cfg["units"],
        "limitations": [
            "单一种子（部署清单中 seed 最小者）：跨种子离散度在 90 回合配额下不可测；"
            "因此结论限于「该候选在该地图上无可行预算」，不得外推为方法的普遍结论",
            "名义环境：无 200 Hz 机械功，实测能量与 BVR 无法在此评估，"
            "physical_status 恒为 not_measured",
            "z0 > Z_REF_J 的行在观测里 z_ratio 被裁到 1.0，策略前半程看不到富余；"
            "已逐行标记，不改网格也不改 Z_REF",
            "本表的可行/无解计数不是导航成功率，不得进入最终对比的分母",
            "危险步数照实记录，但不构成名义不可行（δ 进入几何，与标签判据分开）",
        ],
    }
    out_path = args.json or os.path.join(NAV_ROOT, "reports", "budget_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({
        "verdict": report["verdict"], "episodes": report["n_episodes"],
        "wall_s": report["wall_seconds"], "clean": report["recompute"]["clean"],
        "no_solution_share": {m: summary["per_method"][m]["no_solution_share"]
                              for m in METHODS},
        "selected": {m: [p["selection"]["k"] for p in per_map if p["method"] == m]
                     for m in METHODS},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
