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

FAILURE_LABELS = ("collision", "out_of_bounds", "budget_exhausted")
SUCCESS_LABELS = ("goal_reached",)
TRUNCATION_LABELS = ("stagnation", "timeout")
ALL_LABELS = FAILURE_LABELS + SUCCESS_LABELS + TRUNCATION_LABELS


def verify_prereg(name: str) -> tuple[dict, str]:
    sha = open(os.path.join(PREREG_DIR, name + ".sha256"), encoding="utf-8").read().strip()
    payload = open(os.path.join(PREREG_DIR, name), "rb").read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha:
        raise SystemExit(
            f"prereg hash mismatch: {PREREG} is {actual[:16]}... but the frozen "
            f"pin says {sha[:16]}...; a frozen prereg must never be edited in place"
        )
    return json.loads(payload.decode("utf-8")), actual


def build_envs(prereg: dict, variant: str):
    """18 vector envs, one per train map, z0 = k_train * E_ref(map)."""
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

    refs = json.load(open(E_REF_REPORT, encoding="utf-8"))
    if refs["verdict"] != "PASS":
        raise SystemExit("train-map reference-energy report is not PASS")
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}

    split = json.load(open(SPLIT, encoding="utf-8"))
    train_maps = split["splits"]["train"]["maps"]
    # prereg pins k_train = 1.0 inside budget.rule ("k_train = 1.0")
    rule = prereg["budget"]["rule"]
    k_train = float(rule.split("k_train = ")[1].split("（")[0].split(")")[0])
    envs, z0s = [], []
    for idx, m in enumerate(train_maps):
        spec = M.build(m["level"], m["seed"], design)
        z0 = k_train * e_ref[m["map_id"]]
        if not (0.0 < z0 <= obs.Z_REF_J):
            raise SystemExit(
                f"z0={z0:.2f} for {m['map_id']} outside (0, Z_REF_J={obs.Z_REF_J}); "
                "z_ratio would saturate"
            )
        envs.append(VectorNominalNavEnv(
            spec, geometry, cm, n_envs=1, z0_J=z0,
            t_max_s=float(prereg["environment"]["t_max_s"]),
            r_goal_m=float(prereg["environment"]["r_goal_m"]),
        ))
        z0s.append(z0)
    return envs, z0s, [m["map_id"] for m in train_maps]


class EpisodeState:
    """Per-env: current training segment + step counters."""

    def __init__(self, env_index: int, map_id: str, map_seed: int, obs_dim: int):
        self.env_index = env_index
        self.map_id = map_id
        self.map_seed = int(map_seed)
        self.obs_dim = obs_dim
        self.episode_counter = 0
        self.global_step = 0
        self.new_segment()

    def new_segment(self) -> None:
        self.seg = S.Segment(
            episode_id=self.episode_counter, map_id=self.map_id,
            map_index=self.env_index, seed=self.map_seed, obs_dim=self.obs_dim,
        )

    def reset_episode(self) -> None:
        self.episode_counter += 1
        self.new_segment()


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
    global obj_cfg
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
    args = ap.parse_args()

    prereg, prereg_sha = verify_prereg(args.prereg_name)
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
    envs, z0s, map_ids = build_envs(prereg, args.variant)
    n_envs = len(envs)
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

    net = NW.build(nav_contract)
    cfg = P.PPOConfig.from_contract(VALUE_CONTRACT)
    assert abs(cfg.max_grad_norm - 2.5) < 1e-12, (
        f"contract max_grad_norm={cfg.max_grad_norm}; prereg pins 2.5")
    updater = P.MinimizationUpdater(
        net, cfg,
        lr_policy=float(prereg["algorithm"]["lr_policy"]),
        lr_critic=float(prereg["algorithm"]["lr_critic"]),
    )
    obj_cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)

    calls_per_window = 2 if smoke else int(prereg["window_semantics"]["calls_per_update"])
    max_updates = args.smoke_updates if smoke else int(prereg["window_semantics"]["max_updates"])
    ckpt_every = int(prereg["checkpoints"]["save_every_updates"])

    states = [EpisodeState(i, map_ids[i], envs[i].spec.seed, obs_dim)
              for i in range(n_envs)]
    stores = [S.RolloutStore(run_dir, f"rollout_env{i}", obs_dim=obs_dim)
              for i in range(n_envs)]
    # RolloutStore(root, run_id) joins root/run_id/rollout -> per-env dirs under
    # <run_dir>/rollout_env{i}/rollout; pass run_dir as root for tidiness.
    obs_np = np.stack([envs[i].observation[0] for i in range(n_envs)]).astype(np.float32)

    log_f = open(log_path, "a", encoding="utf-8")
    episode_labels: list[str | None] = []
    t_start = time.time()

    def log(row: dict) -> None:
        log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_f.flush()

    log({"kind": "run_start", "run_id": args.run_id, "variant": args.variant,
         "seed": args.seed, "prereg_sha256": prereg_sha,
         "n_envs": n_envs, "z0_J_range": [round(min(z0s), 2), round(max(z0s), 2)],
         "calls_per_window": calls_per_window, "max_updates": max_updates,
         "smoke": smoke, "provenance": provenance})

    for update_idx in range(max_updates):
        t_win = time.time()
        window_segments: list[tuple[EpisodeState, str, dict[str, np.ndarray], bool]] = []

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
                    z_j=float(r.z_J[0]),
                    z_ratio=float(r.z_ratio[0]),
                    safety_y=float(r.safety_y[0]),
                    goal_distance_m=float(r.goal_distance_m[0]),
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
                    arrays, kind_used = close_segment(
                        st.seg, kind, float(bvc_t[k]), float(bvh_t[k]),
                        env_done=True, label=label_i)
                    window_segments.append((st, kind_used, arrays, True))
                    episode_labels.append(label_i)
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
                window_segments.append((st, kind_used, arrays, False))
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
        stats = updater.update(batch)

        # ---- audit chunks (per-env stores keep global_step continuity) ----
        for st, kind, arrays, env_done in window_segments:
            meta = {
                "event": "episode",
                "chunk_index": stores[st.env_index].expected_chunk_index(),
                "terminal_kind": kind,
                "env_done": env_done,
                "episode_id": st.episode_counter,
                "map_id": st.map_id,
                "map_seed": st.map_seed,
                "n_samples": int(len(arrays["t"])),
            }
            stores[st.env_index].write_chunk(arrays, meta)

        tail = episode_labels[-len(window_segments):]
        row = {
            "kind": "update", "update": update_idx,
            "wall_seconds": round(time.time() - t_win, 3),
            "n_segments": len(window_segments),
            "n_samples": int(batch["obs"].shape[0]),
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
                "calls_per_window": calls_per_window,
                "z0_J": [round(z, 3) for z in z0s],
                "map_ids": map_ids,
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
    }, os.path.join(run_dir, "state.pt"))
    log({"kind": "run_end", "wall_hours": round(wall / 3600.0, 3),
         "updates": max_updates,
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
