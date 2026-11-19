"""Train-only boundary coverage and bounded fixed-actor critic refit (v5).

Diagnosis never samples or optimizes. A reviewed proposal is not permission to
run: formal collection and fitting require a separately approved, pinned file.
All old target reconstruction and the original development/G4 evaluator remain
unchanged. No assumption of monotone fixed-policy feasibility is made.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_guarded as G

B, R = G.B, G.R
PROPOSAL = R.NAV / "config/nav065_adaptive_refit_proposal_v5.json"
BOUND_INPUTS = (*G.BOUND_INPUTS, "scripts/refit_nav065_adaptive.py",
                "tests/test_nav065_adaptive_refit.py", "config/nav065_adaptive_refit_proposal_v5.json",
                "config/nav065_guarded_refit_prereg_v4.json",
                "reports/nav065_guarded_refit_v4_review.json",
                "scripts/refit_nav065_deployment.py", "scripts/evaluate_nav065_deployment.py",
                "src/camra_navigation/algorithm/deployment_oracle.py")


def approved_registration(path):
    p = R.load(path)
    if (p.get("approved") is not True or p.get("status") != "APPROVED"
            or not p.get("authorization_reference")):
        raise ValueError("new sampling and optimizer quotas require explicit approval")
    pin = path.with_suffix(".json.sha256").read_text().strip()
    R.verify_hash(path, pin)
    omit = {"approved", "status", "authorization_reference", "implementation_hashes"}
    if {k: v for k, v in p.items() if k not in omit} != {
            k: v for k, v in R.load(PROPOSAL).items() if k not in omit}:
        raise ValueError("registration differs from reviewed v5 proposal")
    if set(p.get("implementation_hashes", {})) != set(BOUND_INPUTS):
        raise ValueError("incomplete implementation binding")
    for name, digest in p["implementation_hashes"].items():
        R.verify_hash(R.NAV / name, digest)
    return p, pin


def adjacent_intervals(points):
    """Only observed labels, never critic predictions or a monotonicity claim."""
    items = sorted(points.items())
    if (len(items) < 2 or any(not np.isfinite(k) or type(f) is not bool for k, f in items)
            or items[0][0] < .7 or items[-1][0] > 1.3):
        raise ValueError("invalid train budget observations")
    return [{"lo": a, "hi": b, "width": b - a, "crossing": fa != fb,
             "reversal": fa and not fb} for (a, fa), (b, fb) in zip(items, items[1:])]


def next_budget(points):
    intervals = adjacent_intervals(points)
    crossing = [p for p in intervals if p["crossing"]]
    # Round only the tie-break width to remove binary float tie accidents.
    p = min(crossing or intervals, key=lambda x: (-round(x["width"], 12), x["lo"]))
    k = (p["lo"] + p["hi"]) / 2
    if k in points or not .7 < k < 1.3:
        raise ValueError("exhausted distinct in-range midpoints")
    return k, p


def selected_rows(records, settings):
    """Every start plus target-nearest interior boundary rows; no dev feedback."""
    obs, budgets, costs, reach = [], [], [], []
    for _, t in records:
        idx = np.flatnonzero(np.abs(t.path_residual[1:]) <= settings["boundary_absolute_W_max"]) + 1
        idx = idx[np.argsort(np.abs(t.path_residual[idx]), kind="stable")]
        idx = np.r_[0, idx[:settings["interior_samples_per_episode"]]]
        obs.append(t.observations[idx])
        budgets.append(t.budget_J[idx])
        costs.append(t.cost_target[idx])
        reach.append(t.reach_target[idx])
    return tuple(np.concatenate(x) for x in (obs, budgets, costs, reach))


def sign_losses(model, batch, settings, scale):
    obs, budgets, tc, th = (torch.as_tensor(x, dtype=torch.float32) for x in batch)
    vc, vh = model.values(obs)
    budgets = budgets / scale
    joint = B.sign_consistency(torch.maximum(vh, vc - budgets), torch.maximum(th, tc - budgets), settings)
    return (B.sign_consistency(vc - budgets, tc - budgets, settings) + joint,
            B.sign_consistency(vh, th, settings) + joint)


def classify(model, records, scale):
    return G.start_classification(model, records, scale)


def numeric_failures(metrics, reg):
    failures = []
    limits = reg["criteria"]
    for name, m in metrics.items():
        if m["relative_mae"] is None or m["relative_mae"] > limits["rel_mae_max"]:
            failures.append(name + " additional relative MAE")
        if m["pearson_required"] and (m["pearson"] is None or m["pearson"] < limits["pearson_min"]):
            failures.append(name + " additional Pearson")
    return failures


def train_gate(model, old, new, reg, settings, scale):
    # Never dilute old numeric errors with easier newly sampled trajectories.
    base = B.train_summary(model, old, reg, scale)
    metrics = {}
    failures = list(base["failures"])
    rows = []
    for (m, k), truth in new:
        vc, vh = B.S.predict(model, truth.observations)
        rows.append({"map_id": m, "k": k, **R.value_comparison(truth, vc, vh, scale)})
    if rows:
        from evaluate_nav065_deployment import error_metrics
        for pred, target in (("vc_norm", "vc_target_norm"), ("vh", "vh_target")):
            metrics[pred] = error_metrics(np.concatenate([r["rows"][pred] for r in rows]),
                np.concatenate([r["rows"][target] for r in rows]), reg["criteria"]["variance_ratio_min"])
        failures.extend(numeric_failures(metrics, reg))
    false = sum(not r["actual_feasible"] and r["predicted_feasible"] for r in rows)
    if false:
        failures.append("additional false feasible starts")
    if any(r["clipped_budget_rows"] for r in rows):
        failures.append("additional clipped path budgets")
    batch = selected_rows(old + new, settings)
    vc, vh = B.S.predict(model, batch[0])
    actual = np.maximum(batch[3], batch[2] - batch[1] / scale) <= 0
    predicted = np.maximum(vh, vc - batch[1] / scale) <= 0
    mismatches = int(np.count_nonzero(predicted != actual))
    if mismatches:
        failures.append("selected boundary row signs")
    return {"pass": not failures, "failures": failures, "original_grid": base,
            "additional_metrics": metrics, "additional_episodes": len(new),
            "additional_actual_feasible": sum(r["actual_feasible"] for r in rows),
            "additional_false_feasible": false, "selected_rows": len(actual),
            "selected_sign_mismatches": mismatches}


def allowed_trial(previous, trial, prediction, actual, locked, reg):
    if any(f != "false feasible budget at episode start" for f in trial["original_grid"]["failures"]):
        return False
    if np.any(prediction[locked] != actual[locked]):
        return False
    if trial["selected_sign_mismatches"] > previous["selected_sign_mismatches"]:
        return False
    if trial["additional_false_feasible"] > previous["additional_false_feasible"]:
        return False
    limits = reg["criteria"]
    for name, m in trial["additional_metrics"].items():
        prior = previous["additional_metrics"][name]
        if m["relative_mae"] is None or prior["relative_mae"] is None:
            return False
        if m["relative_mae"] > max(limits["rel_mae_max"], prior["relative_mae"]):
            return False
        if m["pearson_required"] and (m["pearson"] is None or
                m["pearson"] < min(limits["pearson_min"], prior["pearson"] if prior["pearson"] is not None else -1)):
            return False
    return True


def fit(model, old, new, protocol, reg, scale, seed, journal):
    records, settings = old + new, protocol["fit"]
    model.policy.requires_grad_(False)
    before = R.digests(model)["policy"]
    gate = train_gate(model, old, new, reg, settings, scale)
    R.write_record(journal, {"kind": "train_gate", "trials": 0, **gate})
    if gate["pass"]:
        return 0, 0, gate
    groups = {role: model.parameters_of(role) for role in R.SCORE_ROLES}
    optimizer = torch.optim.Adam([p for ps in groups.values() for p in ps], lr=settings["learning_rate"])
    rng = np.random.default_rng(settings["sampling_seed"] + seed)
    actual = np.array([t.feasible for _, t in records])
    locked = classify(model, records, scale) == actual
    batch = selected_rows(records, settings)
    accepted, trials = 0, 0
    cap = protocol["resource_limit"]["optimizer_trials_per_candidate_max"]
    while trials < cap and not gate["pass"]:
        regression, _ = B.S.regression_loss(model, records, settings, rng)
        signs = sign_losses(model, batch, settings, scale)
        optimizer.zero_grad()
        G.balanced_gradients(regression, signs, groups)
        if any(not torch.isfinite(p.grad).all() for ps in groups.values() for p in ps):
            raise ValueError("nonfinite projected gradient")
        state, optim = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        took_step = False
        for backtrack in range(settings["backtracks_max"]):
            if trials >= cap:
                break
            model.load_state_dict(state)
            optimizer.load_state_dict(optim)
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rate"] * settings["backtrack_factor"] ** backtrack
            optimizer.step()
            trials += 1
            trial = train_gate(model, old, new, reg, settings, scale)
            prediction = classify(model, records, scale)
            allowed = allowed_trial(gate, trial, prediction, actual, locked, reg)
            R.write_record(journal, {"kind": "trial", "trial": trials, "accepted": allowed,
                                    "backtrack": backtrack, "report": trial})
            if allowed:
                gate, took_step = trial, True
                accepted += 1
                locked |= prediction == actual
                break
        if not took_step:
            model.load_state_dict(state)
            optimizer.load_state_dict(optim)
            break
    if R.digests(model)["policy"] != before:
        raise ValueError("actor changed")
    R.write_record(journal, {"kind": "fit_finished", "trials": trials,
                            "accepted_updates": accepted, **gate})
    return trials, accepted, gate


def sources():
    upstream, reg, inputs, _, train, prepared = G.sources()
    p = R.load(PROPOSAL)
    root = R.WORK / p["source_run"]
    R.verify_hash(root / "fitted_candidates.json", p["source_manifest_sha256"])
    manifest = R.load(root / "fitted_candidates.json")
    v4, pin4 = G.approved_registration(G.REGISTRATION)
    review = R.load(R.NAV / "reports/nav065_guarded_refit_v4_review.json")
    for name, digest in review["evidence"]["output_hashes"].items():
        R.verify_hash(root / name, digest)
    if manifest["refit_pin"] != pin4 or upstream["base_evaluation_pin"] != p["base_evaluation_pin"]:
        raise ValueError("v4/evaluation pins differ")
    by_id = {c["run_id"]: c for c in manifest["candidates"]}
    if len(manifest["candidates"]) != 6 or set(by_id) != {c["run_id"] for c, *_ in prepared}:
        raise ValueError("v4 candidate coverage differs")
    latest = []
    for c, _, records, _ in prepared:
        source = by_id[c["run_id"]]
        R.verify_hash(Path(source["checkpoint"]), source["checkpoint_sha256"])
        model = R.load_network(inputs, source, source["checkpoint"])
        if (R.digests(model) != source["state_dict_sha256"] or
                R.digests(model)["policy"] != c["state_dict_sha256"]["policy"]):
            raise ValueError("source tensors/actor differ")
        baseline = B.train_summary(model, records, reg, inputs.value["quantities"]["Z_REF_J"]["value"])
        if not baseline["pass"]:
            raise ValueError("v4 original train gate no longer passes")
        latest.append((source, model, records, baseline))
    maps, refs = R.train_grid(inputs)
    return upstream, reg, inputs, train, maps, refs, latest


def coverage(prepared, scale):
    result = []
    for c, _, records, baseline in prepared:
        per_map = []
        for m in sorted({m for (m, _), _ in records}):
            subset = [(k, t) for (mid, k), t in records if mid == m]
            points = {k: t.feasible for k, t in subset}
            intervals = adjacent_intervals(points)
            k, bracket = next_budget(points)
            per_map.append({"map_id": m, "next_k": k, "selected_interval": bracket,
                "observed_crossings": sum(p["crossing"] for p in intervals),
                "observed_reversals": sum(p["reversal"] for p in intervals),
                "starts_abs_W_le_0_01": int(sum(abs(t.path_residual[0]) <= .01 for _, t in subset)),
                "starts": [{"k": k, "feasible": t.feasible, "W": float(t.path_residual[0]),
                    "cost_margin_J": float(t.cost_target[0] * scale - t.budget_J[0]),
                    "reach_target": float(t.reach_target[0]), "stop": int(t.stop_index[0]),
                    "steps": len(t.observations)} for k, t in subset]})
        result.append({"run_id": c["run_id"], "original_train_pass": baseline["pass"], "maps": per_map})
    return result


def collect(model, candidate, old, maps, refs, inputs, protocol, journal, totals):
    limits = protocol["resource_limit"]
    if {m["map_id"] for m in maps} != {m for (m, _), _ in old}:
        raise ValueError("sampler may only visit original train maps")
    before = R.digests(model)
    ctx, records = R.E.build_rollout_ctx(candidate["variant"]), []
    for m in maps:
        mid = m["map_id"]
        spec = R.E.M.build(m["level"], m["seed"], ctx["design"])
        points = {k: t.feasible for (name, k), t in old if name == mid}
        if set(points) != set(protocol["sampling"]["initial_grid"]):
            raise ValueError("initial train grid differs")
        for _ in range(limits["new_episodes_per_candidate_map"]):
            if totals["episodes"] >= limits["new_train_episodes_max"]:
                raise ValueError("new train episode quota exhausted")
            max_steps = int(round(inputs.prereg["environment"]["t_max_s"] / .1))
            if totals["steps"] + max_steps > limits["new_train_steps_max"]:
                raise ValueError("new train step quota exhausted")
            k, bracket = next_budget(points)
            key = {"run_id": candidate["run_id"], "map_id": mid, "k": k, "split": "train"}
            totals["episodes"] += 1  # Attempts count before the call, including infrastructure failure.
            R.write_record(journal, {"kind": "rollout_start", **key, "interval": bracket})
            episode = R.rollout(model, spec, ctx, k * R.consistent_e_ref(refs[mid], candidate["variant"]), inputs.prereg)
            totals["steps"] += episode["n_steps"]
            R.write_record(journal, {"kind": "rollout_result", **key, "episode": episode})
            truth = R.reconstruct(episode, spec, inputs)
            if episode["n_steps"] > max_steps or truth.budget_feature_clipped.any():
                raise ValueError("unexpected horizon or clipped new train observations")
            points[k] = truth.feasible
            records.append(((mid, k), truth))
        R.write_record(journal, {"kind": "map_sampling_finished", "run_id": candidate["run_id"],
                                "map_id": mid, "intervals": adjacent_intervals(points)})
    if R.digests(model) != before:
        raise ValueError("sampling changed network tensors")
    return records


def run(protocol, pin, upstream, reg, inputs, train, maps, refs, prepared):
    out = R.WORK / protocol["output_name"]
    out.mkdir(exist_ok=False)
    torch.use_deterministic_algorithms(True)
    totals = {"episodes": 0, "steps": 0, "trials": 0, "updates": 0}
    fitted, gates = [], []
    with (out / "train_episodes.jsonl").open("x") as journal:
        R.write_record(journal, {"kind": "start", "registration_pin": pin,
                                "host": socket.getfqdn(), "tmux": os.environ["TMUX"]})
        # Collect every registered actor/map before fitting; failures are never skipped.
        expanded = [(c, model, old, collect(model, c, old, maps, refs, inputs, protocol, journal, totals))
                    for c, model, old, _ in prepared]
    if totals["episodes"] != protocol["resource_limit"]["new_train_episodes_max"]:
        raise ValueError("incomplete registered sampling coverage")
    for c, model, old, new in expanded:
        with (out / (c["run_id"] + ".jsonl")).open("x") as journal:
            trials, updates, gate = fit(model, old, new, protocol, reg,
                inputs.value["quantities"]["Z_REF_J"]["value"], c["seed"], journal)
        totals["trials"] += trials
        totals["updates"] += updates
        if totals["trials"] > protocol["resource_limit"]["optimizer_trials_total_max"]:
            raise ValueError("optimizer trial quota exceeded")
        path = out / (c["run_id"] + ".pt")
        with path.open("xb") as stream:
            torch.save({"model": model.state_dict(), "adaptive_refit_pin": pin,
                        "prereg_sha256": inputs.train_pin,
                        "implementation_versions": inputs.prereg["implementation_versions"],
                        "variant": c["variant"], "seed": c["seed"]}, stream)
        if R.digests(R.load_network(inputs, c, path)) != R.digests(model):
            raise ValueError("saved CPU reload differs")
        fitted.append({**c, "parent_adaptive_checkpoint": c["checkpoint"],
            "parent_adaptive_checkpoint_sha256": c["checkpoint_sha256"], "checkpoint": str(path),
            "checkpoint_sha256": R.file_sha(path), "state_dict_sha256": R.digests(model)})
        gates.append({"run_id": c["run_id"], "trials": trials, "updates": updates, **gate})
        print(c["run_id"], "train", gate["pass"], "trials", trials, flush=True)
    for name, payload in (("fitted_candidates.json", {"refit_pin": pin, "candidates": fitted,
            "train_episodes": train["train_episodes"] + totals["episodes"],
            "train_steps": train["train_steps"] + totals["steps"], "resource_usage": totals,
            "train_journal_sha256": R.file_sha(out / "train_episodes.jsonl")}),
            ("train_gate.json", {"candidates": gates, "pass": all(g["pass"] for g in gates)})):
        with (out / name).open("x") as stream:
            R.write_record(stream, payload)
    if not all(g["pass"] for g in gates):
        with (out / "HELD_TRAIN_GATE.json").open("x") as stream:
            R.write_record(stream, {"development_scoring_rounds": 0, "resource_usage": totals})
        return
    R.evaluate(out, upstream, pin, reg, inputs, fitted, R.WORK / "nav065_deployment_v1",
               train["train_episodes"] + totals["episodes"], train["train_steps"] + totals["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--diagnose", type=Path)
    choice.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    torch.set_num_threads(1)
    # Approval is checked before expensive preflight and before any output directory.
    if args.registration:
        protocol, pin = approved_registration(args.registration)
        run(protocol, pin, *sources())
        return
    if args.diagnose.exists() or args.diagnose.resolve().parent != (R.NAV / "reports").resolve():
        parser.error("diagnosis requires a new report in navigation/reports")
    upstream, reg, inputs, train, maps, refs, prepared = sources()
    payload = {"schema": "camra-nav065-train-coverage-v5", "host": socket.getfqdn(),
               "tmux": os.environ["TMUX"], "optimizer_steps": 0, "new_rollouts": 0,
               "development_scoring_rounds": 0, "train_episodes": train["train_episodes"],
               "train_steps": train["train_steps"], "source_manifest_sha256": R.load(PROPOSAL)["source_manifest_sha256"],
               "candidates": coverage(prepared, inputs.value["quantities"]["Z_REF_J"]["value"])}
    with args.diagnose.open("x") as stream:
        R.write_record(stream, payload)
    print("train coverage verified:", train["train_episodes"], "episodes; no rollouts/updates", flush=True)


if __name__ == "__main__":
    main()
