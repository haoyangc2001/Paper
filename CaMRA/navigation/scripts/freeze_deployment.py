"""Deployment candidate freeze + critic validation (NAV-038).

Implements the frozen preregistration ``config/deployment_freeze_prereg_v1.json``
(sha256 pinned; verified before anything else).  Per method x seed candidate:

1. load the last-update training checkpoint (actor + both critics);
2. validate both critics against exact contract targets on complete
   deterministic rollouts over the 24 train+development maps;
3. if a critic misses the preregistered thresholds (rel MAE <= 0.35, Pearson
   >= 0.70), run the preregistered refit branch: actor frozen, critics only,
   3 rounds x (collect one 227-call window on the same 24 maps with the
   current critics for bootstrap, compute contract targets, 30 gradient
   steps of the contract value loss);
4. re-validate after every refit round; the last round's verdict stands.

Outputs:
* ``delivery/deployment_candidates_v1.json`` - the version bundle: per
  candidate the source checkpoint hash plus content hashes of the actor and
  both critic state_dicts, the deterministic transform, the normalization
  declaration, and the validation verdict (all items form one version);
* ``reports/deployment_critic_validation.json`` - full validation detail.

Budget-selection and final_* splits are never touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
PREREG = os.path.join(NAV_ROOT, "config", "deployment_freeze_prereg_v1.json")
PREREG_SHA = os.path.join(NAV_ROOT, "config", "deployment_freeze_prereg_v1.json.sha256")
DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
GEOM_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")
SPLIT = os.path.join(NAV_ROOT, "config", "split_manifest_v1.json")
E_REF_REPORT = os.path.join(NAV_ROOT, "reports", "train_map_reference_energy.json")
WORK_ROOT = "/pub/data/caohy/work/camra_navigation"

FAILURE_LABELS = ("collision", "out_of_bounds")
#: only reachable in v1 traces; budget exhaustion is not terminal since NAV-062
LEGACY_LABELS = ("budget_exhausted",)
SUCCESS_LABELS = ("goal_reached",)
TRUNCATION_LABELS = ("stagnation", "timeout")


def verify_prereg() -> tuple[dict, str]:
    sha = open(PREREG_SHA, encoding="utf-8").read().strip()
    payload = open(PREREG, "rb").read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha:
        raise SystemExit(f"prereg hash mismatch: frozen {sha[:16]} vs actual {actual[:16]}")
    return json.loads(payload.decode("utf-8")), actual


def state_dict_sha(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class Ctx:
    """Shared frozen context for one method variant."""

    def __init__(self, variant: str):
        self.variant = variant
        self.design = M.load_design(DESIGN)
        geometry = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOM_CONTRACT)
        if variant == "uncalibrated":
            geometry = dataclasses.replace(geometry, delta_m=0.0)
            self.cm = JCostModel.from_frozen(POWER, MARGINS, variant="uncalibrated")
        else:
            self.cm = JCostModel.from_frozen(POWER, MARGINS, variant="full")
        self.geometry = geometry
        refs = json.load(open(E_REF_REPORT, encoding="utf-8"))
        assert refs["verdict"] == "PASS"
        self.e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
        split = json.load(open(SPLIT, encoding="utf-8"))
        self.maps = [(m, "train") for m in split["splits"]["train"]["maps"]] + \
                    [(m, "development") for m in split["splits"]["development"]["maps"]]
        self.nav_contract = json.load(open(NAV_CONTRACT, encoding="utf-8"))
        self.obs_dim = int(self.nav_contract["observation_spec"]["total_dim"])
        self.obj_cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)


def load_net(ctx: Ctx, run_id: str) -> tuple[Any, dict]:
    path = os.path.join(WORK_ROOT, run_id, "state.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    net = NW.build(ctx.nav_contract)
    net.load_state_dict(state["model"])
    net.eval()
    return net, state


def step_row(obs_i: np.ndarray, vc: float, vh: float, rec_cost_J: float,
             rec) -> dict:
    """One per-step record for the objective machinery.

    Both ends are recorded (NAV-062): ``value_*`` and ``obs`` belong to ``x_t``,
    so they pair with the ``*_before`` quantities; the step's ``*_next`` values
    are the state the transition arrived at.
    """
    return {
        "obs": obs_i.copy(),
        "value_cost": float(vc),
        "value_reach": float(vh),
        "cost_norm": float(rec_cost_J) / obs.Z_REF_J,
        "z_norm": float(rec.z_J_before[0]) / obs.Z_REF_J,
        "z_ratio": float(rec.z_ratio_before[0]),
        "safety_y": float(rec.safety_y_before[0]),
        "goal_distance_m": float(rec.goal_distance_before[0]),
        "z_norm_next": float(rec.z_J[0]) / obs.Z_REF_J,
        "z_ratio_next": float(rec.z_ratio[0]),
        "safety_y_next": float(rec.safety_y[0]),
        "goal_distance_m_next": float(rec.goal_distance_m[0]),
    }


def arrays_from_steps(steps: list[dict], kind: str, bvc: float, bvh: float,
                      env_done: bool, label: str | None) -> dict[str, np.ndarray]:
    T = len(steps)
    arrays = {
        "obs": np.stack([s["obs"] for s in steps]).astype(np.float32),
        "value_cost": np.asarray([s["value_cost"] for s in steps], dtype=np.float64),
        "value_reach": np.asarray([s["value_reach"] for s in steps], dtype=np.float64),
        "cost_norm": np.asarray([s["cost_norm"] for s in steps], dtype=np.float64),
        "z_norm": np.asarray([s["z_norm"] for s in steps], dtype=np.float64),
        "z_ratio": np.asarray([s["z_ratio"] for s in steps], dtype=np.float64),
        "safety_y": np.asarray([s["safety_y"] for s in steps], dtype=np.float64),
        "goal_distance_m": np.asarray([s["goal_distance_m"] for s in steps], dtype=np.float64),
        "z_norm_next": np.asarray([s["z_norm_next"] for s in steps], dtype=np.float64),
        "z_ratio_next": np.asarray([s["z_ratio_next"] for s in steps], dtype=np.float64),
        "safety_y_next": np.asarray([s["safety_y_next"] for s in steps], dtype=np.float64),
        "goal_distance_m_next": np.asarray(
            [s["goal_distance_m_next"] for s in steps], dtype=np.float64),
        "terminated": np.zeros(T, dtype=bool),
        "truncated": np.zeros(T, dtype=bool),
        "bootstrap_value_cost": np.zeros(T, dtype=np.float64),
        "bootstrap_value_reach": np.zeros(T, dtype=np.float64),
    }
    arrays["bootstrap_value_cost"][-1] = bvc
    arrays["bootstrap_value_reach"][-1] = bvh
    arrays["terminated"][-1] = bool(env_done and label in (FAILURE_LABELS + SUCCESS_LABELS))
    arrays["truncated"][-1] = bool(kind == "truncation")
    objs = S.compute_objectives(arrays, ctx_global.obj_cfg, terminal_kind=kind)
    return {**arrays, **objs}


ctx_global: Ctx | None = None


def collect_complete_episodes(ctx: Ctx, net: Any) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Deterministic complete-episode rollouts; bootstrap-free exact targets."""
    global ctx_global
    ctx_global = ctx
    arrays: dict[str, list[np.ndarray]] = {}
    episode_rows = []
    for m, split_name in ctx.maps:
        spec = M.build(m["level"], m["seed"], ctx.design)
        env = VectorNominalNavEnv(spec, ctx.geometry, ctx.cm, n_envs=1,
                                  z0_J=ctx.e_ref[m["map_id"]])
        env.reset()
        steps: list[dict] = []
        n = 0
        while not env._done[0] and n < 300:
            o = torch.as_tensor(env.observation, dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = net.act(o, deterministic=True)
                vc, vh = net.values(o)
            obs_pre = env.observation[0]        # x_t, the state vc/vh belong to
            rec = env.step(a.numpy())
            steps.append(step_row(obs_pre, float(vc[0]), float(vh[0]),
                                  float(rec.cost_model_J[0]), rec))
            n += 1
        label = env._label[0] or "cap_300"
        if label == "cap_300" or label in TRUNCATION_LABELS:
            continue  # prereg: bootstrap-free episodes only
        kind = "success" if label in SUCCESS_LABELS else "failure"
        seg = arrays_from_steps(steps, kind, 0.0, 0.0, True, label)
        for k, v in seg.items():
            arrays.setdefault(k, []).append(v)
        episode_rows.append({"map_id": m["map_id"], "split": split_name,
                             "label": label, "n_steps": len(steps)})
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    return out, episode_rows


def critic_metrics(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    vc = arrays["value_cost"].astype(np.float64)
    vh = arrays["value_reach"].astype(np.float64)
    tc = arrays["obj_target_cost"].astype(np.float64)
    th = arrays["obj_target_reach"].astype(np.float64)

    def one(v: np.ndarray, t: np.ndarray) -> dict[str, float]:
        mae = float(np.mean(np.abs(v - t)))
        denom = float(np.mean(np.abs(t)))
        rel = mae / denom if denom > 1e-12 else float("inf")
        if v.std() > 1e-12 and t.std() > 1e-12:
            corr = float(np.corrcoef(v, t)[0, 1])
        else:
            corr = float("nan")
        return {"mae": round(mae, 5), "mean_abs_target": round(denom, 5),
                "rel_mae": round(rel, 5), "pearson": round(corr, 5)}

    return {"critic_cost": one(vc, tc), "critic_reach": one(vh, th)}


def verdict_of(metrics: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    problems = []
    for name in ("critic_cost", "critic_reach"):
        m = metrics[name]
        if not (m["rel_mae"] <= 0.35):
            problems.append(f"{name} rel_mae {m['rel_mae']} > 0.35")
        if not (m["pearson"] >= 0.70):
            problems.append(f"{name} pearson {m['pearson']} < 0.70")
    return (not problems), problems


def collect_refit_window(ctx: Ctx, net: Any, calls: int = 227) -> dict[str, np.ndarray]:
    """One 227-call window on the 24 maps; segments closed with bootstrap."""
    global ctx_global
    ctx_global = ctx
    envs = [VectorNominalNavEnv(M.build(m["level"], m["seed"], ctx.design),
                                ctx.geometry, ctx.cm, n_envs=1,
                                z0_J=ctx.e_ref[m["map_id"]]) for m, _ in ctx.maps]
    obs_np = np.stack([e.observation[0] for e in envs]).astype(np.float32)
    segments: list[dict[str, np.ndarray]] = []
    per_env: list[list[dict]] = [[] for _ in envs]

    def close(buf: list[dict], kind: str, bvc: float, bvh: float,
              env_done: bool, label: str | None) -> None:
        seg = arrays_from_steps(buf, kind, bvc, bvh, env_done, label)
        segments.append(seg)

    for _ in range(calls):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            a, _, _ = net.act(obs_t, deterministic=True)
            vc, vh = net.values(obs_t)
        recs = [e.step(a[i].numpy()[None, :]) for i, e in enumerate(envs)]
        next_obs = np.stack([r.obs[0] for r in recs]).astype(np.float32)
        done_idx = []
        for i, r in enumerate(recs):
            per_env[i].append(step_row(obs_np[i], float(vc[i]), float(vh[i]),
                                       float(r.cost_model_J[0]), r))
            if r.done[0]:
                done_idx.append(i)
        if done_idx:
            xt = torch.as_tensor(next_obs[done_idx], dtype=torch.float32)
            with torch.no_grad():
                bvc, bvh = net.values(xt)
            for k, i in enumerate(done_idx):
                label = recs[i].label[0]
                kind = ("success" if label in SUCCESS_LABELS
                        else "truncation" if label in TRUNCATION_LABELS else "failure")
                close(per_env[i], kind, float(bvc[k]), float(bvh[k]), True, label)
                per_env[i] = []
                envs[i].reset()
                next_obs[i] = envs[i].observation[0]
        obs_np = next_obs
    xt = torch.as_tensor(obs_np, dtype=torch.float32)
    with torch.no_grad():
        bvc, bvh = net.values(xt)
    for i, e in enumerate(envs):
        if len(per_env[i]):
            close(per_env[i], "truncation", float(bvc[i]), float(bvh[i]), False, None)
            per_env[i] = []
    merged: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        for k, v in seg.items():
            merged.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in merged.items()}


def refit_critics(ctx: Ctx, net: Any, rounds: int = 3, grad_steps: int = 30,
                  lr: float = 1e-3) -> list[dict]:
    """Preregistered refit branch: actor frozen, critics only, contract loss."""
    for p in net.policy.parameters():
        p.requires_grad_(False)
    params = list(net.critic_cost.parameters()) + list(net.critic_reach.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    cfg = P.PPOConfig.from_contract(VALUE_CONTRACT)
    hist = []
    for rnd in range(rounds):
        arrays = collect_refit_window(ctx, net)
        obs = torch.as_tensor(arrays["obs"], dtype=torch.float32)
        tc = torch.as_tensor(arrays["obj_target_cost"], dtype=torch.float32)
        th = torch.as_tensor(arrays["obj_target_reach"], dtype=torch.float32)
        vc_old = torch.as_tensor(arrays["value_cost"], dtype=torch.float32)
        vh_old = torch.as_tensor(arrays["value_reach"], dtype=torch.float32)
        losses: list[float] = []
        for _ in range(grad_steps):
            opt.zero_grad()
            loss_c, _ = P.value_loss(net.critic_cost(obs), vc_old, tc, cfg)
            loss_h, _ = P.value_loss(net.critic_reach(obs), vh_old, th, cfg)
            loss = cfg.vf_coef * (loss_c + loss_h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            losses.append(float(loss.detach()))
        hist.append({"round": rnd + 1, "n_samples": int(obs.shape[0]),
                     "grad_steps": grad_steps,
                     "mean_last5_loss": round(float(np.mean(losses[-5:])), 6)})
    for p in net.policy.parameters():
        p.requires_grad_(True)
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description="deployment freeze + critic validation")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prereg, prereg_sha = verify_prereg()
    torch.set_num_threads(1)
    assert "0.35" in json.dumps(prereg) and "0.70" in json.dumps(prereg)

    candidates = []
    validation_detail: list[dict] = []
    run_prefix = {"uncalibrated": "formal_uncal", "full": "formal_full"}
    for variant in ("uncalibrated", "full"):
        ctx = Ctx(variant)
        for seed in (0, 1, 2):
            run_id = f"{run_prefix[variant]}_s{seed}"
            net, state = load_net(ctx, run_id)
            rounds_hist: list[Any] = []
            arrays, ep_rows = collect_complete_episodes(ctx, net)
            metrics = critic_metrics(arrays)
            ok, problems = verdict_of(metrics)
            refit_rounds = 0
            validation_detail.append({
                "run_id": run_id, "round": 0, "metrics": metrics,
                "episodes_used": len(ep_rows), "n_samples": int(arrays["obs"].shape[0]),
                "verdict": "PASS" if ok else "FAIL", "problems": problems})
            if not ok:
                rounds_hist = refit_critics(ctx, net)
                refit_rounds = len(rounds_hist)
                for rnd in range(1, refit_rounds + 1):
                    arrays, ep_rows = collect_complete_episodes(ctx, net)
                    metrics = critic_metrics(arrays)
                    ok, problems = verdict_of(metrics)
                    validation_detail.append({
                        "run_id": run_id, "round": rnd, "metrics": metrics,
                        "episodes_used": len(ep_rows),
                        "n_samples": int(arrays["obs"].shape[0]),
                        "verdict": "PASS" if ok else "FAIL", "problems": problems})
                    if ok:
                        break

            if refit_rounds:
                out_dir = os.path.join(WORK_ROOT, run_id, "deploy_refit")
                os.makedirs(out_dir, exist_ok=True)
                torch.save({"model": net.state_dict(),
                            "refit_rounds": refit_rounds,
                            "refit_history": rounds_hist,
                            "source_prereg": prereg_sha},
                           os.path.join(out_dir, "state.pt"))
                sd = net.state_dict()
                src_sha = S.file_sha256(os.path.join(out_dir, "state.pt"))
                weights_path = f"{WORK_ROOT}/{run_id}/deploy_refit/state.pt"
            else:
                sd = state["model"]
                src_sha = S.file_sha256(os.path.join(WORK_ROOT, run_id, "state.pt"))
                weights_path = None

            actor_sd = {k: v for k, v in sd.items() if k.startswith("policy.")}
            cc_sd = {k: v for k, v in sd.items() if k.startswith("critic_cost.")}
            cr_sd = {k: v for k, v in sd.items() if k.startswith("critic_reach.")}
            final_detail = [d for d in validation_detail if d["run_id"] == run_id][-1]
            candidates.append({
                "candidate_id": f"{variant}_s{seed}",
                "method": variant,
                "seed": seed,
                "source_run_id": run_id,
                "source_checkpoint_sha256": src_sha,
                "refit_rounds": refit_rounds,
                "refit_history": rounds_hist,
                "weights": {
                    "actor_sha256": state_dict_sha(actor_sd),
                    "critic_cost_sha256": state_dict_sha(cc_sd),
                    "critic_reach_sha256": state_dict_sha(cr_sd),
                    "refit_state_path": weights_path,
                },
                "deterministic_transform": {
                    "action": "policy mean (no sampling, no noise)",
                    "pipeline": "mean -> action_to_requested -> clip_applied",
                    "navigation_contract_version": ctx.nav_contract["version"],
                },
                "normalization": {"obs": "identity (obs-v2 already normalised)"},
                "validation": {
                    "data": "24 maps (train 18 + development 6), deterministic "
                            "complete episodes only (bootstrap-free targets)",
                    "final": final_detail,
                },
            })

    os.makedirs(os.path.join(NAV_ROOT, "delivery"), exist_ok=True)
    bundle = {
        "schema": "camra-deployment-candidates-v1",
        "version": 1,
        "task": "NAV-038",
        "prereg_sha256": prereg_sha,
        "binding_contracts": {"value_objective": "1.2.0", "navigation": "1.4.0"},
        "n_candidates": len(candidates),
        "candidates": candidates,
        "forbidden": "budget_selection / final_* splits untouched; value-search "
                     "budgets from earlier tasks are void for refitted candidates",
    }
    with open(os.path.join(NAV_ROOT, "delivery", "deployment_candidates_v1.json"),
              "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=1)

    n_pass = sum(1 for c in candidates if c["validation"]["final"]["verdict"] == "PASS")
    report = {
        "task": "NAV-038",
        "prereg_sha256": prereg_sha,
        "thresholds": {"rel_mae_cut": 0.35, "pearson_cut": 0.70},
        "n_candidates": len(candidates),
        "n_pass_final": n_pass,
        "verdict": "PASS" if n_pass == len(candidates)
                   else ("PARTIAL" if n_pass else "FAIL"),
        "detail": validation_detail,
        "candidates_summary": [
            {"candidate_id": c["candidate_id"], "refit_rounds": c["refit_rounds"],
             "final_verdict": c["validation"]["final"]["verdict"],
             "rel_mae_c": c["validation"]["final"]["metrics"]["critic_cost"]["rel_mae"],
             "rel_mae_h": c["validation"]["final"]["metrics"]["critic_reach"]["rel_mae"]}
            for c in candidates
        ],
    }
    print(json.dumps(report["candidates_summary"], ensure_ascii=False, indent=1))
    print("verdict:", report["verdict"])
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
