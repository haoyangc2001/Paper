"""Refit the fixed NAV-065 critics on train and replay the frozen development grid."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import socket

import numpy as np
import torch

from evaluate_nav065_deployment import (
    E, NAV, REG_PATH, WORK, build_registered_network, check_inputs, consistent_e_ref,
    file_sha, load, read_registration, reconstruct_truth, review_documents, rollout,
    summarize, tensor_digest, value_comparison, verify_hash, write_record,
)
from camra_navigation.costs import DT_NAV_S

REGISTRATION = NAV / "config/nav065_refit_prereg_v1.json"
SCORE_ROLES = ("critic_cost", "critic_reach")


def load_network(inputs, candidate, checkpoint):
    model = build_registered_network(inputs.nav, inputs.prereg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def digests(model):
    return {role: tensor_digest(getattr(model, role)) for role in model.ROLES}


def reconstruct(episode, spec, inputs):
    return reconstruct_truth(
        episode, initial_goal_distance_m=math.dist(spec.start, spec.goal),
        value_contract=inputs.value, horizon_s=inputs.prereg["environment"]["t_max_s"],
        goal_radius_m=inputs.prereg["environment"]["r_goal_m"], step_s=DT_NAV_S,
    )


def refit(model, trajectories, protocol, seed):
    """Equal weight per episode, half at the initial state, half along the path."""
    model.policy.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for role in SCORE_ROLES for p in model.parameters_of(role)],
        lr=protocol["learning_rate"],
    )
    rng = np.random.default_rng(protocol["sampling_seed"] + seed)
    width = 1 + protocol["interior_samples_per_episode"]
    losses = []
    for _ in range(protocol["updates"]):
        obs, cost, reach = [], [], []
        for truth in trajectories:
            n = len(truth.observations)
            indices = np.empty(width, dtype=np.int64)
            indices[0] = 0
            indices[1:] = rng.choice(
                np.arange(1, n) if n > 1 else [0], size=width - 1, replace=n < width,
            )
            obs.append(truth.observations[indices])
            cost.append(truth.cost_target[indices])
            reach.append(truth.reach_target[indices])
        obs = torch.as_tensor(np.concatenate(obs), dtype=torch.float32)
        vc, vh = model.values(obs)
        errors = torch.stack(((vc - torch.as_tensor(np.concatenate(cost), dtype=vc.dtype)) ** 2,
                              (vh - torch.as_tensor(np.concatenate(reach), dtype=vh.dtype)) ** 2))
        errors = errors.reshape(2, len(trajectories), width)
        start_weight = protocol["start_weight"]
        loss = start_weight * errors[:, :, 0].mean() + (1 - start_weight) * errors[:, :, 1:].mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite refit loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train_grid(inputs):
    maps = load(Path(E.SPLIT))["splits"]["train"]["maps"]
    train_ids = {m["map_id"] for m in maps}
    refs = load(NAV / inputs.prereg["reference_energy"]["report"])
    records = refs["maps"]
    if (len(maps) != len(train_ids) or len(train_ids) != 18
            or train_ids.intersection(m["map_id"] for m in inputs.maps)
            or refs["verdict"] != "PASS" or refs["t_deadline_s"] != inputs.prereg["environment"]["t_max_s"]
            or len(records) != 18 or {r["map_id"] for r in records} != train_ids
            or not all(r["split"] == "train" and r["admitted"] and
                       math.isfinite(r["e_ref_J"]) and r["e_ref_J"] > 0 for r in records)):
        raise ValueError("incorrect train map or reference coverage")
    return maps, {r["map_id"]: r["e_ref_J"] for r in records}


def fixed_dev(inputs, grid, previous):
    """Read only the 108 source and 72 supplemental trajectories."""
    raw = {c["run_id"]: {(ep["map_id"], ep["k"]): ep for ep in inputs.runs[c["run_id"]]["episodes"]}
           for c in inputs.candidates}
    original_ids = {m["map_id"] for m in inputs.maps}
    counts = Counter()
    previous_scores = {}
    with (previous / "episodes.jsonl").open(encoding="utf-8") as f:
        from json import loads
        for line in f:
            row = loads(line)
            counts[row["kind"]] += 1
            if row["kind"] == "rollout_result":
                key = row["map_id"], row["k"]
                episodes = raw[row["run_id"]]
                if key in episodes or key[0] not in original_ids or key[1] not in grid:
                    raise ValueError("duplicate or unexpected saved development trajectory")
                episodes[key] = row["episode"]
            elif row["kind"] == "scored":
                key = row["run_id"], row["map_id"], row["k"]
                if key in previous_scores:
                    raise ValueError("duplicate saved score")
                previous_scores[key] = row["actual_feasible"]
    keys = {(m, k) for m in original_ids for k in grid}
    if (counts != {"start": 1, "rollout_start": 72, "rollout_result": 72, "scored": 180}
            or any(set(episodes) != keys for episodes in raw.values())
            or set(previous_scores) != {(rid, m, k) for rid in raw for m, k in keys}):
        raise ValueError("incomplete saved development grid")
    return raw, previous_scores


def preflight():
    protocol = load(REGISTRATION)
    pin = REGISTRATION.with_suffix(".json.sha256").read_text().strip()
    verify_hash(REGISTRATION, pin)
    if (protocol["schema"] != "camra-nav065-refit-v1" or protocol["approved"] is not True
            or protocol["fit_split"] != "train" or protocol["evaluation_split"] != "development"
            or protocol["max_train_episodes"] != 540 or protocol["new_dev_episodes"] != 0
            or protocol["fit"] != {"updates": 90, "learning_rate": 0.001, "interior_samples_per_episode": 8,
                                   "sampling_seed": 6501, "start_weight": 0.5}):
        raise ValueError("unapproved refit settings")
    for name, digest in protocol["inputs"].items():
        verify_hash(NAV / name, digest)
    previous = WORK / "nav065_deployment_v1"
    for name, digest in protocol["previous_run"].items():
        verify_hash(previous / name, digest)
    reg, eval_pin = read_registration(REG_PATH)
    if eval_pin != protocol["base_evaluation_pin"]:
        raise ValueError("evaluation gate registration changed")
    inputs = check_inputs()
    maps, refs = train_grid(inputs)
    quantities = inputs.value["quantities"]
    lo, hi = quantities["z_ratio"]["range"]
    scale = quantities["Z_REF_J"]["value"]
    for variant in {c["variant"] for c in inputs.candidates}:
        for value in refs.values():
            budgets = [consistent_e_ref(value, variant) * k / scale for k in reg["grid_k"]]
            if not all(lo < z < hi for z in budgets) or not all(a < b for a, b in zip(budgets, budgets[1:])):
                raise ValueError("train budget features lose resolution")
    for candidate in inputs.candidates:
        model = load_network(inputs, candidate, candidate["checkpoint"])
        if digests(model) != candidate["state_dict_sha256"]:
            raise ValueError("source tensor identity mismatch")
    return protocol, pin, reg, inputs, maps, refs, previous


def run(out, protocol, pin, reg, inputs, maps, refs, previous):
    out.mkdir(exist_ok=False)
    torch.manual_seed(6501)
    torch.use_deterministic_algorithms(True)
    fitted, total_episodes, total_steps = [], 0, 0
    with (out / "train_episodes.jsonl").open("x", encoding="utf-8") as journal:
        write_record(journal, {"kind": "start", "refit_pin": pin, "fit_split": "train",
                               "host": socket.getfqdn(), "git": protocol["base_git"]})
        for candidate in inputs.candidates:
            rid, method = candidate["run_id"], candidate["variant"]
            model = load_network(inputs, candidate, candidate["checkpoint"])
            before = digests(model)
            ctx = E.build_rollout_ctx(method)
            trajectories = []
            for m in maps:
                spec = E.M.build(m["level"], m["seed"], ctx["design"])
                for k in reg["grid_k"]:
                    if total_episodes >= protocol["max_train_episodes"]:
                        raise ValueError("train quota exhausted")
                    budget = k * consistent_e_ref(refs[m["map_id"]], method)
                    key = {"run_id": rid, "map_id": m["map_id"], "k": k}
                    write_record(journal, {"kind": "rollout_start", **key})
                    episode = rollout(model, spec, ctx, budget, inputs.prereg)
                    total_episodes += 1
                    total_steps += episode["n_steps"]
                    write_record(journal, {"kind": "rollout_result", **key, "episode": episode})
                    truth = reconstruct(episode, spec, inputs)
                    if truth.budget_feature_clipped.any():
                        raise ValueError("clipped training budget observation")
                    trajectories.append(truth)
            losses = refit(model, trajectories, protocol["fit"], candidate["seed"])
            after = digests(model)
            if (before != candidate["state_dict_sha256"] or before["policy"] != after["policy"]
                    or any(before[role] == after[role] for role in SCORE_ROLES)):
                raise ValueError("actor changed or a critic did not update")
            checkpoint = out / (rid + ".pt")
            with checkpoint.open("xb") as f:
                torch.save({"model": model.state_dict(), "refit_pin": pin,
                            "source_checkpoint_sha256": candidate["checkpoint_sha256"],
                            "implementation_versions": inputs.prereg["implementation_versions"],
                            "prereg_sha256": inputs.train_pin,
                            "variant": method, "seed": candidate["seed"]}, f)
            if digests(load_network(inputs, candidate, checkpoint)) != after:
                raise ValueError("CPU reload differs from fitted model")
            fitted.append({**candidate, "source_checkpoint": candidate["checkpoint"],
                           "source_checkpoint_sha256": candidate["checkpoint_sha256"],
                           "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint),
                           "state_dict_sha256": after, "refit_updates": len(losses),
                           "refit_pin": pin})
            write_record(journal, {"kind": "fit_result", "run_id": rid, "losses": losses,
                                   "before": before, "after": after,
                                   "checkpoint_sha256": fitted[-1]["checkpoint_sha256"]})
            print(f"{rid}: {len(trajectories)} complete train trajectories, {len(losses)} updates", flush=True)
    if total_episodes != protocol["max_train_episodes"] or total_steps > 243000:
        raise ValueError("incomplete or excessive refit coverage")
    with (out / "fitted_candidates.json").open("x", encoding="utf-8") as stream:
        write_record(stream, {"refit_pin": pin, "train_episodes": total_episodes, "train_steps": total_steps,
                              "train_journal_sha256": file_sha(out / "train_episodes.jsonl"),
                              "candidates": fitted})
    evaluate(out, protocol, pin, reg, inputs, fitted, previous, total_episodes, total_steps)


def evaluate(out, protocol, pin, reg, inputs, fitted, previous, total_episodes, total_steps):
    # Every model is frozen and saved before development is scored.
    raw, old_scores = fixed_dev(inputs, reg["grid_k"], previous)
    summaries = []
    z_ref = inputs.value["quantities"]["Z_REF_J"]["value"]
    with (out / "development_scores.jsonl").open("x", encoding="utf-8") as journal:
        for candidate in fitted:
            rid = candidate["run_id"]
            verify_hash(Path(candidate["checkpoint"]), candidate["checkpoint_sha256"])
            model = load_network(inputs, candidate, candidate["checkpoint"])
            ctx = E.build_rollout_ctx(candidate["variant"])
            scored = []
            with torch.inference_mode():
                for m in inputs.maps:
                    map_id = m["map_id"]
                    spec = E.M.build(m["level"], m["seed"], ctx["design"])
                    for k in reg["grid_k"]:
                        episode = raw[rid][map_id, k]
                        if not math.isclose(episode["z0_J"],
                                inputs.budget_J[candidate["variant"], map_id, k], rel_tol=1e-12):
                            raise ValueError("saved development budget differs")
                        truth = reconstruct(episode, spec, inputs)
                        obs = torch.as_tensor(truth.observations, dtype=torch.float32)
                        actions = model.act(obs, deterministic=True)[0].numpy()
                        recorded_actions = [row["action"] for row in episode["trace"]]
                        if (not np.allclose(actions, recorded_actions, atol=1e-6, rtol=0)
                                or truth.feasible != old_scores[rid, map_id, k]):
                            raise ValueError("deterministic behavior or episode outcome differs")
                        vc, vh = model.values(obs)
                        row = {"run_id": rid, "map_id": map_id, "k": k, "label": episode["label"],
                               **value_comparison(truth, vc.numpy(), vh.numpy(), z_ref)}
                        write_record(journal, row)
                        scored.append(row)
            summaries.append({"run_id": rid, **summarize(scored, {m["map_id"] for m in inputs.maps}, reg)})
    score = {"task": "NAV-065", "refit_pin": pin, "eval_pin": protocol["base_evaluation_pin"],
             "train_pin": inputs.train_pin, "train_episodes": total_episodes, "train_steps": total_steps,
             "development_episodes": 180, "reused_development_episodes": 180, "new_episodes": 0,
             "development_scores_sha256": file_sha(out / "development_scores.jsonl"),
             "candidates": summaries, "gate_eligible": all(s["pass"] for s in summaries)}
    path = out / "deployment_critic_validation_v2.json"
    with path.open("x", encoding="utf-8") as stream:
        write_record(stream, score)
    # The original evaluator expects 72 supplemental episodes. They are reused
    # here and verified against their frozen journal, with zero new dev runs.
    evidence = {**inputs.validation_evidence, "refit_pin": pin,
                "fitted_candidates_sha256": file_sha(out / "fitted_candidates.json"),
                "original_development_journal_sha256": protocol["previous_run"]["episodes.jsonl"]}
    gate, bundle = review_documents({**score, "new_episodes": 72}, fitted, evidence,
                                    file_sha(path), inputs.prereg["implementation_versions"])
    with (out / "gate_G4_alignment_v2.json").open("x", encoding="utf-8") as stream:
        write_record(stream, gate)
    if bundle is not None:
        with (out / "deployment_candidates_v2.json").open("x", encoding="utf-8") as stream:
            write_record(stream, {**bundle, "refit_pin": pin})
    print(f"G4 {gate['verdict']} ({sum(s['pass'] for s in summaries)}/6 candidates)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    torch.set_num_threads(1)
    protocol, *inputs = preflight()
    out = WORK / protocol["output_name"]
    if out.parent != WORK or out.name != "nav065_refit_v1" or out.exists():
        parser.error("expected an unused registered output directory")
    print("Preflight PASS: six candidates, 18 train maps, 33 frozen v2 files verified", flush=True)
    if not args.preflight:
        run(out, protocol, *inputs)


if __name__ == "__main__":
    main()
