"""Formal pre-registered training on the vectorised nominal env (NAV-036).

Implements the frozen preregistration ``config/formal_training_prereg_v1.json``
(sha256 pinned in ``config/formal_training_prereg_v1.json.sha256``; the runner
verifies the hash before doing anything else).  Semantics, in the order the
prereg pins them:

* 18 lockstep episodes, one per train-split map, each on its own
  ``VectorNominalNavEnv`` (bitwise-equivalent to the scalar env);
* an env whose episode finishes mid-window is closed immediately (terminal
  kind from its label) and reset on the same map, so every env contributes a
  step to every lockstep call;
* per update: 227 lockstep calls = 4086 env-steps; at the window boundary a
  still-running episode's training segment is closed as ``truncation`` with
  the critic bootstrap captured at the window-end observation (the
  *environment* episode continues into the next window; only the training
  segment ends);
* per segment: ``storage.compute_objectives`` under the contract bootstrap
  table; all segments of the window are concatenated into ONE batch and ONE
  gradient step (advantage normalised over the whole window);
* one audit store per env (NAV-032 RolloutStore: chunk-index / global-step /
  sample-count triple reconciliation; ``global_step`` is the per-env step
  counter, so continuity holds per store);
* checkpoints every 25 updates, final checkpoint = last update, no early
  stop, no hyperparameter changes after the first update;
* budget ``z0 = 1.0 * E_ref(map)`` from the frozen recomputation report.

Run:

    <frozen python> CaMRA/navigation/scripts/train_formal.py \
        --run-id formal_uncal_s0 --variant uncalibrated --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camra_navigation import observation as obs  # noqa: E402
from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402
from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm.validation import verify_training_bindings  # noqa: E402
from camra_navigation.budget import accounting as ACC  # noqa: E402
from camra_navigation.costs import JCostModel  # noqa: E402
from camra_navigation.envs.vector import VectorNominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402

NAV_CONTRACT = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")
PREREG_DIR = os.path.join(NAV_ROOT, "config")
DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
SPLIT = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
E_REF_REPORT = os.path.join(NAV_ROOT, "reports", "train_map_reference_energy.json")

FAILURE_LABELS = ("collision", "out_of_bounds")
SUCCESS_LABELS = ("goal_reached",)
TRUNCATION_LABELS = ("stagnation", "timeout")
ALL_LABELS = FAILURE_LABELS + SUCCESS_LABELS + TRUNCATION_LABELS
#: A v1 run could still produce this label; it is never terminal any more
#: (NAV-062), so it only appears in old logs and is counted separately.
LEGACY_LABELS = ("budget_exhausted",)


class BudgetSampler:
    """Per-episode initial budget, reproducible and resumable (NAV-062).

    The reference implementation draws ``uniform(min_energy, max_energy)`` per
    episode; a single ``k_train`` taught the policy one budget only, while every
    evaluation spans ``k`` 0.7-1.3.  Here each episode gets

        ``z0_J = k * E_ref(map)``,  ``k ~ Uniform(low, high)``

    from a per-env ``numpy`` generator, so (a) the sequence depends only on the
    run seed and env index, (b) an interrupted run resumes the *same* sequence
    (the generator state is checkpointed), and (c) the drawn value is written
    into the segment meta and the run log, never inferred afterwards.

    ``z0`` is capped at ``z_ratio_clip * Z_REF_J`` because beyond that the
    encoded ``z_ratio`` saturates and ``W`` stops being additive; the cap is a
    guard, not a redesign of the requested range.
    """

    def __init__(self, *, seed: int, n_envs: int, low: float, high: float,
                 e_ref: dict[str, float], z_ref_J: float, clip: float) -> None:
        if not (0.0 < low <= high):
            raise SystemExit(f"budget sampling range must satisfy 0 < low <= high, got {(low, high)}")
        self.low, self.high = float(low), float(high)
        self.e_ref = dict(e_ref)
        self.z_max = float(clip) * float(z_ref_J)
        self.gens = [np.random.default_rng([int(seed), int(i)]) for i in range(n_envs)]

    def draw(self, env_index: int, map_id: str) -> dict[str, float]:
        if map_id not in self.e_ref:
            raise SystemExit(f"no E_ref for map {map_id!r}")
        k = float(self.gens[env_index].uniform(self.low, self.high))
        z0 = k * float(self.e_ref[map_id])
        if z0 > self.z_max:
            raise SystemExit(
                f"sampled z0={z0:.2f} J > clip*Z_REF={self.z_max:.2f} J for {map_id}; "
                "the requested budget range does not fit the observation encoding")
        return {"k": k, "z0_J": z0, "map_id": map_id}

    def state(self) -> list[dict]:
        return [dict(g.bit_generator.state) for g in self.gens]

    def load(self, state: list[dict]) -> None:
        if len(state) != len(self.gens):
            raise SystemExit(f"budget RNG state has {len(state)} entries, expected {len(self.gens)}")
        for g, s in zip(self.gens, state):
            g.bit_generator.state = dict(s)


def verify_prereg(name: str) -> tuple[dict, str]:
    sha = open(os.path.join(PREREG_DIR, name + ".sha256"), encoding="utf-8").read().strip()
    payload = open(os.path.join(PREREG_DIR, name), "rb").read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha:
        raise SystemExit(
            f"prereg hash mismatch: {name} is {actual[:16]}... but the frozen "
            f"pin says {sha[:16]}...; a frozen prereg must never be edited in place"
        )
    prereg = json.loads(payload.decode("utf-8"))
    try:
        verify_training_bindings(prereg, NAV_ROOT)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"training release refused: {exc}") from exc
    return prereg, actual


def registered_run_id(prereg: dict, variant: str, seed: int) -> str:
    """Resolve a candidate from the frozen run registry, never a previous attempt."""
    if "run_registry" not in prereg:
        return f"nav064_r14_{variant}_s{seed}"
    matches = [r["run_id"] for r in prereg["run_registry"]
               if r["variant"] == variant and r["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("run registry requires exactly one candidate per method/seed")
    return matches[0]


def budget_rules(prereg: dict) -> dict:
    """Budget sampling rule of this preregistration.

    Two accepted forms:

    * structured (NAV-062 and later):
      ``budget.sampling = {"kind": "uniform", "k_low": 0.6, "k_high": 1.3}``
    * legacy: ``budget.rule`` containing ``k_train = <float>`` -- a single k for
      every episode, kept so v1/v2/v3 preregs stay runnable and their old runs
      stay reproducible.  A legacy prereg is *not* evidence for the new loop.
    """
    b = prereg.get("budget", {})
    s = b.get("sampling")
    if isinstance(s, dict):
        if str(s.get("kind")) != "uniform":
            raise SystemExit(f"unsupported budget sampling kind {s.get('kind')!r}")
        return {"kind": "uniform", "k_low": float(s["k_low"]), "k_high": float(s["k_high"])}
    rule = str(b.get("rule", ""))
    if "k_train = " not in rule:
        raise SystemExit("prereg does not state a budget rule (sampling or k_train)")
    k = float(rule.split("k_train = ")[1].split("（")[0].split(")")[0])
    return {"kind": "fixed", "k": k}


def build_envs(prereg: dict, variant: str):
    """18 vector envs, one per train map; z0 is sampled per episode later."""
    design = M.load_design(DESIGN)
    geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
    if variant == "uncalibrated":
        import dataclasses
        geometry = dataclasses.replace(geometry, delta_m=0.0)
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
    elif variant == "full":
        cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    # r13：参考能耗报告与预算系数可由预登记指定（默认沿用冻结件与 k=1.0，
    # 旧预登记因此逐位不变）。契约余量版必须同时改时限与 E_ref，否则训练与
    # 评估的口径不一致——r12 的评估修复就是踩过这个坑。
    ref_cfg = prereg.get("reference_energy", {}) or {}
    ref_name = str(ref_cfg.get("report", "reports/train_map_reference_energy.json"))
    ref_path = ref_name if os.path.isabs(ref_name) else os.path.join(NAV_ROOT, ref_name)
    k_budget = float(ref_cfg.get("budget_k", 1.0))
    if not (0.9 <= k_budget <= 1.3):
        raise SystemExit(f"reference_energy.budget_k={k_budget} 超出允许范围 [0.9,1.3]")
    refs = json.load(open(ref_path, encoding="utf-8"))
    if refs["verdict"] != "PASS":
        raise SystemExit("train-map reference-energy report is not PASS")
    e_ref_nominal = {r["map_id"]: k_budget * float(r["e_ref_J"]) for r in refs["maps"]}
    print(f"[REF-ENERGY] report={os.path.basename(ref_path)} "
          f"t_deadline={refs.get('t_deadline_s', 30.0)} budget_k={k_budget}", flush=True)
    # NAV-064 修正 A：预算基准与扣费同口径。默认 nominal（旧语义逐位回归），
    # 只有预登记显式声明 consistent 才启用；alpha 属于 full 变体。
    acct = str(prereg.get("budget_accounting", "nominal"))
    if acct == "consistent":
        e_ref = {m: ACC.consistent_e_ref(e, variant) for m, e in e_ref_nominal.items()}
    elif acct == "nominal":
        e_ref = e_ref_nominal
    else:
        raise SystemExit(f"unknown budget_accounting={acct!r}")
    print(f"[BUDGET-ACCT] {acct} | alpha={ACC.alpha_for_variant(variant):.6f} "
          f"| E_ref[0] {list(e_ref_nominal.values())[0]:.1f} -> "
          f"{list(e_ref.values())[0]:.1f} J", flush=True)

    split = json.load(open(SPLIT, encoding="utf-8"))
    train_maps = split["splits"]["train"]["maps"]
    z_max = obs.z_ratio_clip() * obs.Z_REF_J
    # r14（用户 2026-11-15T17:32 授权）：停滞阈值与安全严重度分级一律从预登记
    # 读取，训练与评估共用同一份口径（r12 的教训：评估悄悄用了另一套预算）。
    stag = prereg.get("stagnation", {}) or {}
    stag_window = float(stag.get("window_s", 3.0))
    stag_min_disp = float(stag.get("min_displacement_m", 0.05))
    if not (0.5 <= stag_window <= 10.0):
        raise SystemExit(f"stagnation.window_s={stag_window} 超出 [0.5, 10]")
    if not (0.005 <= stag_min_disp <= 0.2):
        raise SystemExit(f"stagnation.min_displacement_m={stag_min_disp} 超出 [0.005, 0.2]")
    sev = prereg.get("safety_continuity", {}) or {}
    sev_on = bool(sev.get("enabled", False))
    sev_floor = float(sev.get("severity_floor", 0.2))
    sev_ref = float(sev.get("severity_ref_depth_m", 0.25))
    if sev_on and not (0.0 < sev_floor < 1.0):
        raise SystemExit(f"safety_continuity.severity_floor={sev_floor} 超出 (0, 1)")
    if sev_on and sev_ref <= 0.0:
        raise SystemExit(f"safety_continuity.severity_ref_depth_m={sev_ref} 必须 > 0")
    print(f"[STAGNATION] window_s={stag_window} min_displacement_m={stag_min_disp}",
          flush=True)
    print(f"[SAFETY-CONT] enabled={sev_on} floor={sev_floor} "
          f"ref_depth_m={sev_ref}", flush=True)
    envs, map_ids = [], []
    for idx, m in enumerate(train_maps):
        spec = M.build(m["level"], m["seed"], design)
        z0 = float(e_ref[m["map_id"]])          # placeholder; reset() sets the real one
        if not (0.0 < z0 <= z_max):
            raise SystemExit(
                f"E_ref={z0:.2f} for {m['map_id']} outside (0, clip*Z_REF={z_max:.1f}); "
                "z_ratio would saturate"
            )
        envs.append(VectorNominalNavEnv(
            spec, geometry, cm, n_envs=1, z0_J=z0,
            t_max_s=float(prereg["environment"]["t_max_s"]),
            r_goal_m=float(prereg["environment"]["r_goal_m"]),
            stagnation_window_s=stag_window,
            stagnation_min_displacement_m=stag_min_disp,
            safety_severity_enabled=sev_on,
            safety_severity_floor=sev_floor,
            safety_severity_ref_depth_m=sev_ref,
            observation_version=prereg.get("implementation_versions", {}).get(
                "observation", "obs-v4" if int(prereg.get("round", 0)) >= 14 else "obs-v3"),
        ))
        map_ids.append(m["map_id"])
    return envs, e_ref, map_ids


class EpisodeState:
    """Per-env: current training segment + step counters."""

    def __init__(self, env_index: int, map_id: str, map_seed: int, obs_dim: int,
                 episode_z0: dict | None = None):
        self.env_index = env_index
        self.map_id = map_id
        self.map_seed = int(map_seed)
        self.obs_dim = obs_dim
        self.episode_counter = 0
        self.global_step = 0
        #: ``{"k", "z0_J", "map_id"}`` of the episode being collected.  Written
        #: into the chunk meta so a chunk always states its own budget.
        self.episode_z0 = dict(episode_z0 or {})
        self.new_segment()

    def new_segment(self) -> None:
        self.seg = S.Segment(
            episode_id=self.episode_counter, map_id=self.map_id,
            map_index=self.env_index, seed=self.map_seed, obs_dim=self.obs_dim,
        )

    def reset_episode(self) -> None:
        self.episode_counter += 1
        self.new_segment()


def reach_gamma_at_update(update_idx: int, total_updates: int,
                          gamma_init: float, gamma_final: float) -> float:
    """Reference gamma_2 schedule: reach final gamma halfway through training."""
    if total_updates <= 0 or not 0 <= update_idx < total_updates:
        raise ValueError("invalid update or training horizon")
    if not 0 < gamma_init <= gamma_final < 1:
        raise ValueError("reach discounts must be increasing and strictly below one")
    return min(gamma_final, gamma_init + (gamma_final - gamma_init)
               * 2.0 * update_idx / total_updates)


def segment_identity(state: EpisodeState) -> dict:
    """Capture identifiers and budget before reset mutates EpisodeState."""
    return {"env_index": int(state.env_index), "episode_id": int(state.episode_counter),
            "map_id": state.map_id, "map_seed": int(state.map_seed),
            "episode_z0_J": state.episode_z0.get("z0_J"),
            "episode_k": state.episode_z0.get("k")}


def close_segment(
    seg: S.Segment, kind: str, bvc: float, bvh: float,
    *, env_done: bool, label: str | None,
) -> tuple[dict[str, np.ndarray], str]:
    """Finalise a training segment under the contract bootstrap table."""
    arrays = seg.arrays()
    arrays["bootstrap_value_cost"][-1] = float(bvc)
    arrays["bootstrap_value_reach"][-1] = float(bvh)
    arrays["terminated"][-1] = bool(env_done and label in (FAILURE_LABELS + SUCCESS_LABELS))
    arrays["truncated"][-1] = bool(kind == "truncation")
    objs = S.compute_objectives(arrays, obj_cfg, terminal_kind=kind)
    return {**arrays, **objs}, kind


def main() -> int:
    global obj_cfg, NAV_CONTRACT, VALUE_CONTRACT
    ap = argparse.ArgumentParser(description="formal preregistered training")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", default="uncalibrated",
                    choices=("uncalibrated", "full"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work-root", default="/pub/data/caohy/work/camra_navigation")
    ap.add_argument("--prereg-name", default="formal_training_prereg_v1.json")
    ap.add_argument("--smoke-updates", type=int, default=0,
                    help="mechanical test only: N updates with 2 calls per window; "
                         "never for a formal run")
    ap.add_argument("--init-from", default=None,
                    help="warm start: load model/optimizers/torch RNG from a previous "
                         "round's state.pt (loop protocol allows continuing training; "
                         "must be declared in the round prereg)")
    args = ap.parse_args()

    prereg, prereg_sha = verify_prereg(args.prereg_name)
    if "run_registry" in prereg and args.run_id != registered_run_id(prereg, args.variant, args.seed):
        raise SystemExit("run ID differs from frozen attempt registry")
    for key, global_name in (("navigation_contract", "NAV_CONTRACT"),
                              ("value_contract", "VALUE_CONTRACT")):
        if key in prereg.get("binding", {}):
            globals()[global_name] = os.path.join(NAV_ROOT, prereg["binding"][key]["path"])
    smoke = bool(args.smoke_updates)
    if smoke:
        print(f"[SMOKE] mechanical test: {args.smoke_updates} updates, "
              "2 calls/window -- NOT a formal run", flush=True)

    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))

    run_dir = os.path.join(os.path.abspath(args.work_root), args.run_id)
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise SystemExit(f"{run_dir} already exists and is non-empty; retry_rule "
                         "requires a new run id instead of reusing an old run")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    nav_contract = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    obs_dim = int(nav_contract["observation_spec"]["total_dim"])
    envs, e_ref, map_ids = build_envs(prereg, args.variant)
    n_envs = len(envs)
    rules = budget_rules(prereg)
    if rules["kind"] == "uniform":
        sampler = BudgetSampler(
            seed=int(args.seed), n_envs=n_envs,
            low=rules["k_low"], high=rules["k_high"], e_ref=e_ref,
            z_ref_J=obs.Z_REF_J, clip=obs.z_ratio_clip(envs[0].observation_version))
        budget_draws = [sampler.draw(i, map_ids[i]) for i in range(n_envs)]
    else:
        sampler = None
        budget_draws = [{"k": rules["k"], "z0_J": rules["k"] * e_ref[m], "map_id": m}
                        for m in map_ids]
    envs = [envs[i] for i in range(n_envs)]
    for i in range(n_envs):
        envs[i].reset(z0_J=float(budget_draws[i]["z0_J"]))
    z0s = [float(b["z0_J"]) for b in budget_draws]
    # NAV-037 traceability: the calibration values and frozen-file hashes the
    # cost/geometry actually used, logged before the first update
    provenance = {
        "alpha": envs[0]._cost_models[0].alpha,
        "alpha_source": envs[0]._cost_models[0].alpha_source,
        "variant": envs[0]._cost_models[0].variant,
        "delta_m": envs[0].geometry.delta_m,
        "inflation_radius_m": envs[0].geometry.inflation_radius_m,
        "geometry_contract_version": envs[0].geometry.contract_version,
        "footprint_sha256": envs[0].geometry.footprint_sha256,
        "margins_sha256": envs[0].geometry.margins_sha256,
        "power_model_sha256": S.file_sha256(POWER),
        "geometry_contract_sha256": S.file_sha256(GEOM_CONTRACT),
    }

    # NAV-064 修正 B：从先冻结的预登记读取，默认 False（逐位回归）
    bounded = bool(prereg.get("action_bounded", False))
    # NAV-064 初始化修正：一律从先冻结的预登记读取，默认关闭（逐位回归）
    init_fix = prereg.get("action_init_fix", {})
    init_on = bool(init_fix.get("enabled", False))
    init_speed = float(init_fix.get("initial_forward_speed", 0.0)) if init_on else 0.0
    init_std_min = init_fix.get("log_std_min") if init_on else None
    if init_std_min is not None:
        init_std_min = float(init_std_min)
    init_std_init = init_fix.get("log_std_init") if init_on else None
    if init_std_init is not None:
        init_std_init = float(init_std_init)
    if not (0.0 <= init_speed < 1.0):
        raise SystemExit(f"initial_forward_speed={init_speed} 超出 [0,1)")
    # NAV-064 转向先验：一律从先冻结的预登记读取，默认关闭（逐位回归）
    steer = prereg.get("steering_prior", {})
    steer_on = bool(steer.get("enabled", False))
    steer_gain = float(steer.get("gain", 0.0)) if steer_on else 0.0
    if not (-5.0 <= steer_gain <= 5.0):
        raise SystemExit(f"steering gain={steer_gain} 超出 [-5,5]")
    print(f"[STEER-PRIOR] enabled={steer_on} gain={steer_gain} "
          f"field=goal_dir_y_body target=wz", flush=True)
    net = NW.build(nav_contract, action_bounded=bounded,
                   initial_forward_speed=init_speed, log_std_min=init_std_min,
                   log_std_init=init_std_init, steering_gain=steer_gain)
    print(f"[ACTION-BOUNDED] {bounded}", flush=True)
    print(f"[ACTION-INIT] enabled={init_on} v0={init_speed} "
          f"log_std_min={init_std_min} log_std_init={init_std_init}", flush=True)
    cfg = P.PPOConfig.from_contract(VALUE_CONTRACT)
    assert abs(cfg.max_grad_norm - 2.5) < 1e-12, (
        f"contract max_grad_norm={cfg.max_grad_norm}; prereg pins 2.5")
    # 熵系数是价值契约明文的"优化超参"（可由训练任务调整），且训练循环协议
    # 把每轮可调范围钉在 0-0.02；from_contract 的守卫针对的是语义量，这里按
    # 预登记显式放行并做带内校验，越界直接退出。
    import dataclasses as _dc
    ent = prereg["algorithm"].get("ent_coef")
    if ent is not None:
        ent = float(ent)
        if not (0.0 <= ent <= 0.02):
            raise SystemExit(f"ent_coef {ent} outside the loop protocol band [0, 0.02]")
        cfg = _dc.replace(cfg, ent_coef=ent)
    # NAV-064 策略层仪器化（policy_grad_decomp，默认关闭）：只额外计算「优势项」与
    # 「熵项」各自的梯度范数并记进日志，**不参与更新、不改任何损失**。用途是定位
    # 「优势项塌缩、熵占主导」这类只在行为上可见的失效（第 11 轮 X 光实测熵项占比
    # 约 1%，因此默认关闭；打开后每次 update 多两次只读反传）。
    decomp_on = bool(prereg.get("policy_grad_decomp", {}).get("enabled", False))
    updater = P.MinimizationUpdater(
        net, cfg,
        lr_policy=float(prereg["algorithm"]["lr_policy"]),
        lr_critic=float(prereg["algorithm"]["lr_critic"]),
        grad_decomp=decomp_on,
    )
    obj_cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    reach_estimator = json.load(open(VALUE_CONTRACT, encoding="utf-8"))["estimator"]
    # NAV-064 修正 ①/②：权重与开关一律从**先冻结的**预登记读取，不硬编码。
    # 默认 0/False 保证旧预登记（r1-r5）的语义逐位不变；新预登记必须显式声明。
    import dataclasses as _dc
    fix = prereg.get("objective_signal_fix", {})
    w = float(fix.get("joint_cost_weight", 0.0))
    if not (0.0 <= w <= 100.0):
        raise SystemExit(f"joint_cost_weight={w} 超出预登记允许范围 [0,100]")
    # r14（用户 2026-11-15T17:32 授权）：h 语义版本。v2_deep_well = 安全态
    # h=g_norm（深井 G_INSIDE=-3）+ 违规态 max(g,C·y) 不变；必须与值契约
    # v1.3.0（G_INSIDE 已改 -3）同现，否则深井被 max(·,0) 抹平，白改。
    h_sem = str(fix.get("h_semantics", "v1"))
    if h_sem not in ("v1", "v2_deep_well"):
        raise SystemExit(f"objective_signal_fix.h_semantics={h_sem!r} 不受支持"
                         "（允许 v1 / v2_deep_well）")
    if h_sem == "v2_deep_well" and obj_cfg.G_INSIDE > -0.5:
        raise SystemExit(
            f"h_semantics=v2_deep_well 但值契约 G_INSIDE={obj_cfg.G_INSIDE} 不是深井；"
            "深井语义要求值契约先升版到 G_INSIDE=-3.0（v1.3.0）")
    use_min = bool(fix.get("joint_min_recursion", False))
    if use_min and h_sem != "v2_deep_well":
        # amend3 的三条数值反例仍然成立：v1 语义（h 被 max(·,0) 抹平）下开
        # min 递推会翻转到达梯度。只有深井语义 + 门禁（corr(h,adv)>0）过了
        # 才允许开启。
        raise SystemExit("joint_min_recursion=True 需要 h_semantics=v2_deep_well；"
                         "v1 下会翻转到达梯度（amend3 反例），训练器拒绝启用")
    cost_mask = bool(fix.get("cost_on_overspend_only", False))
    # r13 候选：预算内逐步成本的基础权重。默认 0.0 = 与 r12 逐位等价。
    # 依据 reports/contract_margin_eval.json：预算抬高 15% 后策略把额度花掉
    # （能耗/预算 92%→114-123%），未超支步被整体遮蔽导致「省着用」没有梯度。
    wb = float(fix.get("joint_cost_base_weight", 0.0))
    if not (0.0 <= wb <= 10.0):
        raise SystemExit(f"joint_cost_base_weight={wb} 超出预登记允许范围 [0,10]")
    obj_cfg = _dc.replace(obj_cfg, joint_cost_weight=w, joint_min_recursion=use_min,
                          joint_cost_on_overspend_only=cost_mask,
                          joint_cost_base_weight=wb, h_semantics=h_sem)
    print(f"[OBJ-FIX] joint_cost_weight={w} joint_min_recursion={use_min} "
          f"cost_on_overspend_only={cost_mask} cost_base_weight={wb} "
          f"h_semantics={h_sem}", flush=True)

    # NAV-064 成本项课程（cost_curriculum，默认关闭）：前 warmup_updates 次更新
    # 把 joint_cost_weight 置 0（只练到达），之后恢复预登记的 w。只改训练的
    # 启用时序；不改 W、τ*、可行性判据、也不改固定 w 的语义。
    curr = prereg.get("cost_curriculum", {})
    curr_on = bool(curr.get("enabled", False))
    curr_warmup = int(curr.get("warmup_updates", 0)) if curr_on else 0
    if curr_on and not (0 <= curr_warmup <= 600):
        raise SystemExit(f"warmup_updates={curr_warmup} 超出 [0,600]")
    obj_cfg_base = obj_cfg
    if curr_on:
        obj_cfg = _dc.replace(obj_cfg_base, joint_cost_weight=0.0)
    print(f"[CURRICULUM] enabled={curr_on} warmup_updates={curr_warmup} "
          f"w_after_warmup={w}", flush=True)

    # NAV-064 盖章段隔离（dead_step_mask，默认关闭）：不可逆安全历史 y=1 之后，
    # h=max(g_norm,C·y) 恒为 1，目标与基线同时被钉死；这些步既没有「离终点多近」
    # 的信息，又会通过优势标准化把整体统计量往常数上拽。默认 False 保证旧预登记
    # 逐位不变；只改**哪些样本参与损失**，不改 W / τ* / 可行性判据 / 目标定义。
    dsm = prereg.get("dead_step_mask", {})
    dsm_on = bool(dsm.get("enabled", False))
    dsm_policy = bool(dsm.get("mask_policy_surrogate", True)) if dsm_on else False
    dsm_reach = bool(dsm.get("mask_reach_regression", True)) if dsm_on else False
    dsm_field = str(dsm.get("field", "safety_y"))
    if dsm_on and dsm_field != "safety_y":
        raise SystemExit(f"dead_step_mask.field={dsm_field!r} 不受支持（只允许 safety_y）")
    print(f"[DEAD-STEP-MASK] enabled={dsm_on} field={dsm_field} "
          f"policy={dsm_policy} reach={dsm_reach}", flush=True)
    print(f"[POLICY-DECOMP] enabled={decomp_on}", flush=True)

    if args.init_from:
        prev = S.atomic_load(args.init_from)
        net.load_state_dict(prev["model"])
        # amend2 实现点：r1-r4 链上 log_std 已发散（4.5-7.7），钳位上界 1.0 对
        # 它们是截断而非恢复；重置回契约初值让本轮从可学习范围起点重新估计
        # 探索幅度。均值头与 critic 保留已学行为。
        with torch.no_grad():
            net.policy.log_std.fill_(float(net.policy.cfg.log_std_init))
        for k, v in updater.optimizers.items():
            v.load_state_dict(prev["optimizers"][k])
        torch.set_rng_state(prev["torch_rng"])
        print(f"[WARMSTART] model/optimizers/rng loaded from {args.init_from}; "
              "log_std reset to contract init (amend2)", flush=True)

    calls_per_window = 2 if smoke else int(prereg["window_semantics"]["calls_per_update"])
    max_updates = args.smoke_updates if smoke else int(prereg["window_semantics"]["max_updates"])
    ckpt_every = int(prereg["checkpoints"]["save_every_updates"])

    states = [EpisodeState(i, map_ids[i], envs[i].spec.seed, obs_dim,
                          episode_z0=budget_draws[i])
              for i in range(n_envs)]
    stores = [S.RolloutStore(run_dir, f"rollout_env{i}", obs_dim=obs_dim)
              for i in range(n_envs)]
    # RolloutStore(root, run_id) joins root/run_id/rollout -> per-env dirs under
    # <run_dir>/rollout_env{i}/rollout; pass run_dir as root for tidiness.
    obs_np = np.stack([envs[i].observation[0] for i in range(n_envs)]).astype(np.float32)

    # r14 诊断日志的口径字段：与 build_envs 同一份预登记读取（同一表达式）
    stag_cfg = prereg.get("stagnation", {}) or {}
    stag_window = float(stag_cfg.get("window_s", 3.0))
    stag_min_disp = float(stag_cfg.get("min_displacement_m", 0.05))
    sev_cfg = prereg.get("safety_continuity", {}) or {}
    sev_on = bool(sev_cfg.get("enabled", False))
    sev_floor = float(sev_cfg.get("severity_floor", 0.2))
    sev_ref = float(sev_cfg.get("severity_ref_depth_m", 0.25))

    log_f = open(log_path, "a", encoding="utf-8")
    episode_labels: list[str | None] = []
    t_start = time.time()

    def log(row: dict) -> None:
        log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_f.flush()

    log({"kind": "run_start", "run_id": args.run_id, "variant": args.variant,
         "seed": args.seed, "prereg_sha256": prereg_sha,
         "n_envs": n_envs, "z0_J_range": [round(min(z0s), 2), round(max(z0s), 2)],
         "budget_rule": rules, "z0_J_initial": [round(z, 3) for z in z0s],
         "z_ratio_clip": obs.z_ratio_clip(envs[0].observation_version),
         "obs_version": envs[0].observation_version,
         "schema": S.SCHEMA_BUFFER,
         "runtime": {"host": socket.getfqdn(), "python": sys.executable,
                     "tmux": os.environ.get("TMUX"), "cwd": os.getcwd()},
         "data_access": {"split_used": "train", "map_ids": map_ids,
                         "inputs": {p: S.file_sha256(p) for p in
                                    (SPLIT, DESIGN, POWER, MARGINS, FOOTPRINT, GEOM_CONTRACT)}},
         "implementation_versions": prereg.get("implementation_versions", {}),
         "runtime_contract": prereg.get("runtime_contract", {}),
         "calls_per_window": calls_per_window, "max_updates": max_updates,
         "smoke": smoke, "provenance": provenance,
         "objective_signal_fix": {"joint_cost_weight": w,
                                  "joint_cost_base_weight": wb,
                                  "joint_min_recursion": use_min,
                                  "cost_on_overspend_only": cost_mask,
                                  "h_semantics": h_sem},
         "stagnation": {"window_s": stag_window,
                        "min_displacement_m": stag_min_disp},
         "safety_continuity": {"enabled": sev_on,
                               "severity_floor": sev_floor,
                               "severity_ref_depth_m": sev_ref},
         "budget_accounting": str(prereg.get("budget_accounting", "nominal")),
         "action_bounded": bool(prereg.get("action_bounded", False)),
         "action_init_fix": {"enabled": init_on,
                             "initial_forward_speed": init_speed,
                             "log_std_min": init_std_min,
                             "log_std_init": init_std_init},
         "steering_prior": {"enabled": steer_on, "gain": steer_gain,
                            "field": "goal_dir_y_body", "target": "wz"},
         "cost_curriculum": {"enabled": curr_on,
                             "warmup_updates": curr_warmup,
                             "w_after_warmup": w}})

    for update_idx in range(max_updates):
        # 课程切换点：只在这一处重绑 obj_cfg，close_segment 读到的就是当次权重
        if curr_on:
            obj_cfg = _dc.replace(
                obj_cfg_base,
                joint_cost_weight=0.0 if update_idx < curr_warmup else w)
            if update_idx in (0, curr_warmup - 1, curr_warmup) or update_idx == max_updates - 1:
                print(f"[CURRICULUM] update={update_idx} "
                      f"joint_cost_weight={float(obj_cfg.joint_cost_weight):.1f}",
                      flush=True)
        # Follow RC-PPO's actual gamma_2 consumer, independently of gamma_joint.
        obj_cfg = _dc.replace(obj_cfg, gamma_h=reach_gamma_at_update(
            update_idx, max_updates, float(reach_estimator["gamma_h_init"]),
            float(reach_estimator["gamma_h_final"])))
        t_win = time.time()
        window_segments: list[tuple[dict, str, dict[str, np.ndarray], bool]] = []

        for _call in range(calls_per_window):
            obs_in = obs_np
            obs_t = torch.as_tensor(obs_in, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, _ = net.act(obs_t)
                vc, vh = net.values(obs_t)
            recs = [envs[i].step(action[i].numpy()[None, :]) for i in range(n_envs)]

            terminal_obs_idx: list[int] = []
            next_obs = np.stack([r.obs[0] for r in recs]).astype(np.float32)
            for i, r in enumerate(recs):
                st = states[i]
                seg = st.seg
                done_i = bool(r.done[0])
                label_i = r.label[0]
                seg.append(
                    global_step=st.global_step,
                    episode_id=st.episode_counter,
                    t=len(seg),
                    obs=obs_in[i],
                    next_obs=next_obs[i],
                    action_raw=action[i].numpy(),
                    action_applied=r.applied[0],
                    log_prob=float(log_prob[i]),
                    value_cost=float(vc[i]),
                    value_reach=float(vh[i]),
                    cost_j=float(r.cost_model_J[0]),
                    cost_norm=float(r.cost_model_J[0]) / obs.Z_REF_J,
                    # t side: the state the critics and `obs` were evaluated at
                    z_j=float(r.z_J_before[0]),
                    z_norm=float(r.z_J_before[0]) / obs.Z_REF_J,
                    z_ratio=float(r.z_ratio_before[0]),
                    safety_y=float(r.safety_y_before[0]),
                    goal_distance_m=float(r.goal_distance_before[0]),
                    # t+1 side: what this transition actually arrived at
                    z_norm_next=float(r.z_J[0]) / obs.Z_REF_J,
                    z_ratio_next=float(r.z_ratio[0]),
                    safety_y_next=float(r.safety_y[0]),
                    goal_distance_m_next=float(r.goal_distance_m[0]),
                    terminated=bool(done_i and label_i in (FAILURE_LABELS + SUCCESS_LABELS)),
                    truncated=bool(done_i and label_i in TRUNCATION_LABELS),
                    event_code=S.EVENT_CODES[label_i or "none"],
                    map_index=st.env_index,
                    bootstrap_value_cost=0.0,
                    bootstrap_value_reach=0.0,
                    action_clipped=bool(r.clipped[0]),
                    action_invalid=bool(r.invalid[0]),
                )
                st.global_step += 1
                if done_i:
                    terminal_obs_idx.append(i)

            if terminal_obs_idx:
                # critic bootstrap at each terminal state x_T (batched)
                xt = torch.as_tensor(next_obs[terminal_obs_idx], dtype=torch.float32)
                with torch.no_grad():
                    bvc_t, bvh_t = net.values(xt)
                for k, i in enumerate(terminal_obs_idx):
                    st = states[i]
                    label_i = recs[i].label[0]
                    if label_i in SUCCESS_LABELS:
                        kind = "success"
                    elif label_i in TRUNCATION_LABELS:
                        kind = "truncation"
                    else:
                        kind = "failure"
                    # r14 诊断：在 reset 冲掉现场之前把整局证据抓下来
                    # （停滞触发现场、最高安全严重度、终点距离与余电）。
                    res_i = envs[i].result(0)
                    arrays, kind_used = close_segment(
                        st.seg, kind, float(bvc_t[k]), float(bvh_t[k]),
                        env_done=True, label=label_i)
                    window_segments.append((segment_identity(st), kind_used, arrays, True))
                    episode_labels.append(label_i)
                    log({"kind": "episode_diag", "env": st.env_index,
                         "map_id": st.map_id, "episode_id": st.episode_counter,
                         "label": label_i, "terminal_kind": kind_used,
                         "n_steps": int(res_i.n_steps), "t_s": round(float(res_i.t_s), 2),
                         "goal_distance_m": round(float(res_i.goal_distance_m), 4),
                         "z_J_end": round(float(res_i.z_J), 2),
                         "z0_J": round(float(res_i.z0_J), 2),
                         "max_safety_severity": round(float(res_i.max_safety_severity), 4),
                         "danger_steps": int(res_i.danger_steps),
                         "within_budget": bool(res_i.within_budget),
                         "arrived_within_budget": bool(res_i.arrived_within_budget),
                         "stagnation_context": res_i.stagnation_context})
                    # a new episode draws its own budget (NAV-062); the draw is
                    # logged so the episode's z0 never has to be inferred
                    if sampler is not None:
                        draw = sampler.draw(st.env_index, st.map_id)
                        envs[i].reset(z0_J=float(draw["z0_J"]))
                        st.episode_z0 = draw
                    else:
                        envs[i].reset()
                    st.reset_episode()
                    next_obs[i] = envs[i].observation[0]
            obs_np = next_obs

        # ---- window boundary: truncate still-running episodes -------------
        xt = torch.as_tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            bvc, bvh = net.values(xt)
        for i in range(n_envs):
            if not envs[i]._done[0]:
                st = states[i]
                if len(st.seg) == 0:
                    # episode finished on the very last call and was reset;
                    # nothing to close, the fresh segment carries over
                    continue
                arrays, kind_used = close_segment(
                    st.seg, "truncation", float(bvc[i]), float(bvh[i]),
                    env_done=False, label=None)
                window_segments.append((segment_identity(st), kind_used, arrays, False))
                # the environment episode continues; only the training segment
                # ended -- open a fresh segment for the same episode
                st.new_segment()
            # done envs were closed and reset mid-window; nothing to do here

        # ---- one batch, one gradient step ---------------------------------
        merged: dict[str, list[np.ndarray]] = {}
        for _, _, arrays, _ in window_segments:
            for k, v in arrays.items():
                merged.setdefault(k, []).append(v)
        batch_arrays = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
        batch = {
            "obs": torch.as_tensor(batch_arrays["obs"], dtype=torch.float32),
            "action": torch.as_tensor(batch_arrays["action_raw"], dtype=torch.float32),
            "log_prob_old": torch.as_tensor(batch_arrays["log_prob"], dtype=torch.float32),
            "adv_total": torch.as_tensor(batch_arrays["obj_adv_total"], dtype=torch.float32),
            "value_cost_old": torch.as_tensor(batch_arrays["value_cost"], dtype=torch.float32),
            "target_cost": torch.as_tensor(batch_arrays["obj_target_cost"], dtype=torch.float32),
            "value_reach_old": torch.as_tensor(batch_arrays["value_reach"], dtype=torch.float32),
            "target_reach": torch.as_tensor(batch_arrays["obj_target_reach"], dtype=torch.float32),
        }
        if dsm_on:
            # 有效样本 = 本步**尚未**触发不可逆安全历史（safety_y=0）。
            live = torch.as_tensor((np.asarray(batch_arrays["safety_y"]) <= 0.5),
                                   dtype=torch.float32)
            if dsm_policy:
                batch["live_policy"] = live
            if dsm_reach:
                batch["live_reach"] = live
        epochs = int(prereg["algorithm"].get("epochs_per_window", 1))
        mb = int(prereg["algorithm"].get("minibatch_size", 0)) or int(batch["obs"].shape[0])
        n_samples = int(batch["obs"].shape[0])
        stats_acc: dict[str, list[float]] = {}
        grad_steps = 0
        # 盖章段隔离打开时，一个 minibatch 可能整批落在 y=1 段（完整变体里成片出现）。
        # 这种批没有有效样本，跳过并计数；不抬高阈值、不放宽掩码。
        mask_keys = [k for k in ("live_policy", "live_reach") if k in batch]
        skipped_mb = 0
        for _epoch in range(epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, mb):
                sel = perm[start:start + mb]
                if int(sel.numel()) < 2:
                    continue
                sub = {k: v[sel] for k, v in batch.items()}
                if mask_keys:
                    kept = min(float((sub[k] > 0.5).sum()) for k in mask_keys)
                    if kept < 2:
                        skipped_mb += 1
                        continue
                st_sub = updater.update(sub)
                net.policy.clamp_log_std()   # amend2: log_std ∈ [-5, 1]
                grad_steps += 1
                for k, v in st_sub.items():
                    if isinstance(v, float):
                        stats_acc.setdefault(k, []).append(v)
        stats = {k: float(np.mean(v)) for k, v in stats_acc.items()}
        stats["grad_steps"] = float(grad_steps)
        if mask_keys:
            stats["mask_skipped_minibatches"] = float(skipped_mb)

        # ---- audit chunks (per-env stores keep global_step continuity) ----
        for identity, kind, arrays, env_done in window_segments:
            env_index = identity["env_index"]
            meta = {
                "event": "episode",
                "chunk_index": stores[env_index].expected_chunk_index(),
                "terminal_kind": kind,
                "env_done": env_done,
                "episode_id": identity["episode_id"],
                "map_id": identity["map_id"],
                "map_seed": identity["map_seed"],
                "n_samples": int(len(arrays["t"])),
                "episode_z0_J": identity["episode_z0_J"],
                "episode_k": identity["episode_k"],
                "gamma_h": obj_cfg.gamma_h,
                "gamma_joint": obj_cfg.gamma_joint,
                "update_idx": update_idx,
                "schema": S.SCHEMA_BUFFER,
            }
            assert np.all(arrays["episode_id"] == meta["episode_id"])
            stores[env_index].write_chunk(arrays, meta)

        tail = episode_labels[-len(window_segments):]
        # r14 诊断：本窗口的目标信号健康度。corr_g_adv 应 > 0（越靠近目标
        # 优势越大；深井语义下的核心验收线），corr_c_adv 在超支口径下应 < 0，
        # 全程撤罚（w=0）时应 ≈ 0（成本只经 −z 进 max，不直接进优势）。
        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
            sa, sb = a.std(), b.std()
            if sa < 1e-12 or sb < 1e-12:
                return float("nan")
            return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        _g = batch_arrays["obj_g_norm"]; _adv = batch_arrays["obj_adv_total"]
        _h = batch_arrays["obj_h"]; _c = batch_arrays["cost_norm"]
        _z = batch_arrays["z_norm"]
        win_diag = {
            "corr_g_adv": round(_corr(_g, _adv), 4),
            "corr_c_adv": round(_corr(_c, _adv), 4),
            "h_min": round(float(np.min(_h)), 4),
            "h_mean": round(float(np.mean(_h)), 4),
            "frac_h_below0": round(float(np.mean(_h < 0.0)), 4),
            "frac_overspend": round(float(np.mean(_z < 0.0)), 4),
            "adv_total_min": round(float(np.min(_adv)), 4),
            "adv_total_max": round(float(np.max(_adv)), 4),
        }
        row = {
            "kind": "update", "update": update_idx,
            "wall_seconds": round(time.time() - t_win, 3),
            "gamma_h": obj_cfg.gamma_h, "gamma_joint": obj_cfg.gamma_joint,
            "n_segments": len(window_segments),
            "n_samples": int(batch["obs"].shape[0]),
            "epochs": epochs, "minibatch_size": mb, "grad_steps": grad_steps,
            "win_diag": win_diag,
            "labels_last_window": {
                name: sum(1 for x in tail if x == name) for name in ALL_LABELS
            },
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in stats.items()},
        }
        log(row)
        if (update_idx + 1) % ckpt_every == 0 or update_idx == max_updates - 1:
            S.atomic_save({
                "model": net.state_dict(),
                "optimizers": {k: v.state_dict()
                               for k, v in updater.optimizers.items()},
                "torch_rng": torch.get_rng_state(),
                "update_idx": update_idx,
                "seed": args.seed,
                "variant": args.variant,
                "prereg_sha256": prereg_sha,
                "implementation_versions": prereg.get("implementation_versions", {}),
                "calls_per_window": calls_per_window,
                "z0_J": [round(z, 3) for z in z0s],
                "map_ids": map_ids,
                # NAV-062: the per-episode budget stream must be resumable, so
                # its generator states travel with the weights
                "budget_rule": rules,
                "budget_rng": None if sampler is None else sampler.state(),
                "episode_z0": [st.episode_z0 for st in states],
            }, os.path.join(run_dir, f"state_u{update_idx + 1}.pt"))

    wall = time.time() - t_start
    S.atomic_save({
        "model": net.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in updater.optimizers.items()},
        "torch_rng": torch.get_rng_state(),
        "update_idx": max_updates - 1,
        "seed": args.seed,
        "variant": args.variant,
        "prereg_sha256": prereg_sha,
        "implementation_versions": prereg.get("implementation_versions", {}),
    }, os.path.join(run_dir, "state.pt"))
    log({"kind": "run_end", "wall_hours": round(wall / 3600.0, 3),
         "updates": max_updates,
         "budget_rule": rules,
         "episode_label_counts": {name: sum(1 for x in episode_labels if x == name)
                                  for name in ALL_LABELS}})
    log_f.close()

    problems: list[str] = []
    for i, store in enumerate(stores):
        audit = S.audit_store(store)
        problems.extend(f"env{i}: {p}" for p in audit["problems"])

    summary = {
        "task": "NAV-036",
        "run_id": args.run_id,
        "variant": args.variant,
        "seed": args.seed,
        "smoke": smoke,
        "prereg_sha256": prereg_sha,
        "updates": max_updates,
        "env_steps": max_updates * calls_per_window * n_envs,
        "wall_hours": round(wall / 3600.0, 3),
        "episodes_completed": len(episode_labels),
        "episode_label_counts": {
            name: sum(1 for x in episode_labels if x == name) for name in ALL_LABELS
        },
        "z0_J_range": [round(min(z0s), 2), round(max(z0s), 2)],
        "store_problems": problems,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
