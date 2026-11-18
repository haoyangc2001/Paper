"""Train-only, versioned NAV-065 refit with guarded finite update steps.

The original trajectory targets, deployment networks and evaluator are reused.
Diagnosis makes no optimizer steps and never scores development trajectories.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_boundary as B

R = B.R
TARGET = "nav064_terminal_tail_v1_full_s1"
REGISTRATION = R.NAV / "config/nav065_guarded_refit_prereg_v4.json"
BOUND_INPUTS = (*B.BOUND_INPUTS, "scripts/refit_nav065_guarded.py",
                "tests/test_nav065_guarded_refit.py")


def sources():
    """Verify the previous attempt, then prepare all 540 saved train episodes."""
    p3, pin3 = B.approved_registration(R.NAV / "config/nav065_boundary_refit_prereg_v3.json")
    upstream, reg, inputs, maps, _, train, raw = B.load_sources(p3)
    previous = R.WORK / "nav065_boundary_refit_v3"
    review = R.load(R.NAV / "reports/nav065_boundary_refit_v3_review.json")
    R.verify_hash(previous / "fitted_candidates.json", review["evidence"]["saved_tensor_manifest_sha256"])
    R.verify_hash(previous / "train_gate.json", review["evidence"]["train_gate_sha256"])
    source = R.load(previous / "fitted_candidates.json")
    if source["refit_pin"] != pin3 or review["registration_pin"] != pin3:
        raise ValueError("v3 source registration differs")
    preparation = copy.deepcopy(p3)
    preparation["resource_limit"]["update_candidates"] = [TARGET]
    prepared = B.prepare(preparation, reg, inputs, maps, source, train, raw)
    return upstream, reg, inputs, source, train, prepared


def start_tensors(records, scale):
    obs = torch.as_tensor(np.stack([t.observations[0] for _, t in records]), dtype=torch.float32)
    budgets = torch.as_tensor([t.budget_J[0] / scale for _, t in records], dtype=torch.float32)
    cost = torch.as_tensor([t.cost_target[0] for _, t in records], dtype=torch.float32)
    reach = torch.as_tensor([t.reach_target[0] for _, t in records], dtype=torch.float32)
    return obs, budgets, cost, reach


def diagnosis(model, records, scale):
    settings = R.load(B.PROPOSAL)["fit"]
    regression, _ = B.S.regression_loss(model, records, settings, np.random.default_rng(6505))
    obs, budgets, tc, th = start_tensors(records, scale)
    vc, vh = model.values(obs)
    penalties = (B.sign_consistency(vc - budgets, tc - budgets, settings),
                 B.sign_consistency(vh, th, settings))
    result = {}
    for role, penalty in zip(R.SCORE_ROLES, penalties):
        params = model.parameters_of(role)
        grads = [torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            term, params, retain_graph=True)]) for term in (regression, penalty)]
        norms = [g.norm().item() for g in grads]
        cosine = float(torch.dot(*grads).item() / (norms[0] * norms[1])) if min(norms) else None
        result[role] = {"regression_gradient_norm": norms[0], "sign_gradient_norm": norms[1],
                        "gradient_cosine": cosine, "sign_loss": penalty.item()}
    result["optimizer_steps"] = 0
    result["development_scoring_rounds"] = 0
    return result


def balanced_gradients(regression, signs, groups):
    """Remove the regression component opposing sign descent, per critic.

    Equal gradient norms avoid a change in value units changing the balance.
    Projection changes only the optimizer direction, never any target or W.
    """
    stats = {}
    for role, sign in zip(R.SCORE_ROLES, signs):
        params = groups[role]
        rg, sg = [torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            term, params, retain_graph=True)]) for term in (regression, sign)]
        rn, sn = rg.norm(), sg.norm()
        dot = torch.dot(rg, sg)
        cosine = (dot / (rn * sn)).item() if rn.item() and sn.item() else None
        if sn.item() > 0:
            if dot.item() < 0:
                rg = rg - dot / sn.square() * sg
            rg = rg + rn.clamp_min(1e-8) / sn * sg
        offset = 0
        for param in params:
            param.grad = rg[offset:offset + param.numel()].reshape_as(param).clone()
            offset += param.numel()
        stats[role] = {"regression_gradient_norm": rn.item(), "sign_gradient_norm": sn.item(),
                       "gradient_cosine": cosine, "direction_norm": rg.norm().item()}
    return stats


def start_classification(model, records, scale):
    obs, budgets, _, _ = start_tensors(records, scale)
    with torch.inference_mode():
        vc, vh = model.values(obs)
        w = torch.maximum(vh, vc - budgets)
    return w.numpy() <= 0


def trial_allowed(report, predicted, actual, locked):
    numerical = [f for f in report["failures"] if f != "false feasible budget at episode start"]
    if numerical:
        return False, "numerical_or_primary_gate"
    if np.any(predicted[locked] != actual[locked]):
        return False, "previously_correct_start_changed"
    return True, "accepted"


def fit(model, records, protocol, reg, scale, candidate, journal):
    model.policy.requires_grad_(False)
    before = R.digests(model)
    report = B.train_summary(model, records, reg, scale)
    R.write_record(journal, {"kind": "train_gate", "trial": 0, "updates": 0, **report})
    if report["pass"]:
        return 0, report
    if candidate["run_id"] != TARGET:
        raise ValueError("only the remaining failed candidate may update")
    if any(f != "false feasible budget at episode start" for f in report["failures"]):
        raise ValueError("source must already meet numerical and primary criteria")
    settings = protocol["fit"]
    groups = {role: model.parameters_of(role) for role in R.SCORE_ROLES}
    optimizer = torch.optim.Adam([p for ps in groups.values() for p in ps], lr=settings["learning_rate"])
    rng = np.random.default_rng(settings["sampling_seed"] + candidate["seed"])
    actual = np.array([t.feasible for _, t in records])
    locked = start_classification(model, records, scale) == actual
    accepted, trials, reason = 0, 0, None
    cap = protocol["resource_limit"]["optimizer_step_trials_max"]
    while trials < cap and not report["pass"]:
        regression, _ = B.S.regression_loss(model, records, settings, rng)
        obs, budgets, tc, th = start_tensors(records, scale)
        vc, vh = model.values(obs)
        signs = (B.sign_consistency(vc - budgets, tc - budgets, settings),
                 B.sign_consistency(vh, th, settings))
        optimizer.zero_grad()
        stats = balanced_gradients(regression, signs, groups)
        if any(not torch.isfinite(p.grad).all() for ps in groups.values() for p in ps):
            raise ValueError("nonfinite projected gradient")
        saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        took_step = False
        for backtrack in range(settings["backtracks_max"]):
            if trials >= cap:
                break
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            lr = settings["learning_rate"] * settings["backtrack_factor"] ** backtrack
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            trials += 1
            trial_report = B.train_summary(model, records, reg, scale)
            prediction = start_classification(model, records, scale)
            allowed, reason = trial_allowed(trial_report, prediction, actual, locked)
            R.write_record(journal, {"kind": "trial", "trial": trials, "accepted": allowed,
                "reason": reason, "learning_rate": lr, "backtrack": backtrack,
                "regression_loss_before": regression.item(),
                "sign_losses_before": [s.item() for s in signs], "gradients": stats,
                "previously_correct_starts": int(locked.sum()), "report": trial_report})
            if allowed:
                accepted += 1
                report = trial_report
                locked |= prediction == actual
                took_step = True
                R.write_record(journal, {"kind": "train_gate", "trial": trials,
                                        "updates": accepted, **report})
                break
        if not took_step:
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            reason = "bounded_line_search_exhausted"
            break
    after = R.digests(model)
    if before["policy"] != after["policy"]:
        raise ValueError("actor changed")
    R.write_record(journal, {"kind": "fit_finished", "optimizer_step_trials": trials,
                            "accepted_updates": accepted, "pass": report["pass"], "stop_reason": reason})
    return accepted, report


def approved_registration(path):
    p = R.load(path)
    R.verify_hash(path, path.with_suffix(".json.sha256").read_text().strip())
    if (p.get("approved") is not True or p.get("schema") != "camra-nav065-guarded-refit-v4"
            or not p.get("authorization_reference")):
        raise ValueError("v4 requires approved new trial quota")
    expected = {"new_rollouts": 0, "existing_train_episodes": 540,
                "existing_development_episodes": 180, "optimizer_step_trials_max": 300,
                "additional_updates_total_max": 300, "development_scoring_rounds_max": 1,
                "update_candidates": [TARGET]}
    if p["resource_limit"] != expected or p["output_name"] != "nav065_guarded_refit_v4":
        raise ValueError("v4 resource scope differs")
    settings = {**R.load(B.PROPOSAL)["fit"], "learning_rate": 1e-5, "sampling_seed": 6505,
                "backtracks_max": 8, "backtrack_factor": 0.5}
    if p["fit"] != settings:
        raise ValueError("v4 fitted optimization settings differ")
    if set(p["implementation_hashes"]) != set(BOUND_INPUTS):
        raise ValueError("incomplete implementation binding")
    for name, digest in p["implementation_hashes"].items():
        R.verify_hash(R.NAV / name, digest)
    return p, R.file_sha(path)


def run(protocol, pin, upstream, reg, inputs, source, train, prepared):
    out = R.WORK / protocol["output_name"]
    if out.exists():
        raise ValueError("existing v4 output; inspect instead of restarting")
    out.mkdir()
    torch.use_deterministic_algorithms(True)
    fitted, gates, total = [], [], 0
    for candidate, model, records, baseline in prepared:
        rid = candidate["run_id"]
        before = R.digests(model)
        with (out / (rid + ".jsonl")).open("x") as journal:
            R.write_record(journal, {"kind": "start", "registration_pin": pin,
                "source_checkpoint": candidate["checkpoint"], "source_digests": before,
                "host": socket.getfqdn(), "tmux": os.environ["TMUX"]})
            updates, gate = fit(model, records, protocol, reg,
                               inputs.value["quantities"]["Z_REF_J"]["value"], candidate, journal)
        total += updates
        after = R.digests(model)
        if after["policy"] != before["policy"] or (rid != TARGET and after != before):
            raise ValueError("protected tensors changed")
        checkpoint = out / (rid + ".pt")
        with checkpoint.open("xb") as stream:
            torch.save({"model": model.state_dict(), "guarded_refit_pin": pin,
                        "prereg_sha256": inputs.train_pin,
                        "implementation_versions": inputs.prereg["implementation_versions"],
                        "variant": candidate["variant"], "seed": candidate["seed"]}, stream)
        if R.digests(R.load_network(inputs, candidate, checkpoint)) != after:
            raise ValueError("CPU reload differs")
        fitted.append({**candidate, "parent_guarded_checkpoint": candidate["checkpoint"],
            "parent_guarded_checkpoint_sha256": candidate["checkpoint_sha256"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": R.file_sha(checkpoint),
            "state_dict_sha256": after, "guarded_refit_updates": updates, "guarded_refit_pin": pin})
        gates.append({"run_id": rid, "updates": updates, **gate})
        print(rid, "updates", updates, "train_pass", gate["pass"], flush=True)
    for name, payload in [("fitted_candidates.json", {"refit_pin": pin, "candidates": fitted,
            "train_episodes": train["train_episodes"], "train_steps": train["train_steps"],
            "new_rollouts": 0, "additional_updates": total}),
            ("train_gate.json", {"candidates": gates, "pass": all(g["pass"] for g in gates)})]:
        with (out / name).open("x") as stream:
            R.write_record(stream, payload)
    if not all(g["pass"] for g in gates):
        with (out / "HELD_TRAIN_GATE.json").open("x") as stream:
            R.write_record(stream, {"new_rollouts": 0, "development_scoring_rounds": 0})
        return
    R.evaluate(out, upstream, pin, reg, inputs, fitted, R.WORK / "nav065_deployment_v1",
               train["train_episodes"], train["train_steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    torch.set_num_threads(1)
    if args.diagnose == bool(args.registration):
        parser.error("select diagnosis or approved formal execution")
    if args.registration:
        protocol, pin = approved_registration(args.registration)
        upstream, reg, inputs, source, train, prepared = sources()
        if R.file_sha(R.WORK / "nav065_boundary_refit_v3/fitted_candidates.json") != protocol["source_manifest_sha256"]:
            raise ValueError("source manifest differs")
        run(protocol, pin, upstream, reg, inputs, source, train, prepared)
        return
    upstream, reg, inputs, source, train, prepared = sources()
    candidate, model, records, baseline = next(p for p in prepared if p[0]["run_id"] == TARGET)
    print(json.dumps({"task": "NAV-065", "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
        "train_episodes": train["train_episodes"], "train_steps": train["train_steps"],
        "baseline": baseline, "diagnosis": diagnosis(model, records,
            inputs.value["quantities"]["Z_REF_J"]["value"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
