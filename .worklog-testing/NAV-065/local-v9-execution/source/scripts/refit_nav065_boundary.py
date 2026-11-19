"""Versioned, train-only sign correction for fixed NAV-065 critics.

Preflight reads saved train data and performs zero optimizer steps. Formal
updates require a separately approved registration, matching code hashes and
new per-candidate quotas. All six train gates precede development scoring.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_saved as S

R = S.R
PROPOSAL = R.NAV / "config/nav065_boundary_refit_proposal_v3.json"
BOUND_INPUTS = (*S.BOUND_INPUTS, "scripts/refit_nav065_boundary.py",
                "tests/test_nav065_boundary_refit.py",
                "config/nav065_boundary_refit_proposal_v3.json")


def approved_registration(path):
    proposal, protocol = R.load(PROPOSAL), R.load(path)
    pin_path = path.with_suffix(".json.sha256")
    if not pin_path.is_file():
        raise ValueError("requires separately approved registration and SHA256 pin")
    pin = pin_path.read_text().strip()
    R.verify_hash(path, pin)
    if (protocol.get("approved") is not True or protocol.get("status") != "APPROVED"
            or not protocol.get("authorization_reference")):
        raise ValueError("new update quotas require explicit approval")
    expected = {k: v for k, v in proposal.items() if k not in ("approved", "status")}
    actual = {k: v for k, v in protocol.items() if k not in (
        "approved", "status", "authorization_reference", "implementation_hashes")}
    if actual != expected:
        raise ValueError("approved settings differ from reviewed v3 proposal")
    hashes = protocol.get("implementation_hashes", {})
    if set(hashes) != set(BOUND_INPUTS):
        raise ValueError("missing exact implementation binding")
    for name, digest in hashes.items():
        R.verify_hash(R.NAV / name, digest)
    return protocol, pin


def sign_consistency(prediction, target, settings):
    """Component sign objective; the exact unchanged target has zero penalty."""
    selected = target.abs() <= settings["sign_boundary_absolute_max"]
    if not selected.any():
        return prediction.sum() * 0
    p, t = prediction[selected], target[selected]
    margin = t.abs().clamp(max=settings["sign_margin_cap"])
    violation = torch.where(t == 0, p, margin - t.sign() * p)
    return (violation.relu() / settings["sign_normalization"]).square().mean()


def boundary_loss(model, records, settings, scale, rng):
    regression, boundary_count = S.regression_loss(model, records, settings, rng)
    obs = torch.as_tensor(np.stack([t.observations[0] for _, t in records]), dtype=torch.float32)
    cost, reach = model.values(obs)
    budgets = torch.as_tensor([t.budget_J[0] / scale for _, t in records], dtype=cost.dtype)
    target_c = torch.as_tensor([t.cost_target[0] for _, t in records], dtype=cost.dtype) - budgets
    target_h = torch.as_tensor([t.reach_target[0] for _, t in records], dtype=reach.dtype)
    signs = (sign_consistency(cost - budgets, target_c, settings)
             + sign_consistency(reach, target_h, settings)) / 2
    loss = regression + settings["sign_weight"] * signs
    return loss, {"regression_loss": regression.item(), "sign_loss": signs.item(),
                  "boundary_starts": boundary_count}


def recognition_summary(rows, map_ids, reg):
    """Keep actual policy outcomes separate from critic true-positive counts."""
    actual_primary = sum(row["actual_feasible"] for row in rows if row["k"] == reg["primary_k"])
    train_reg = copy.deepcopy(reg)
    train_reg["criteria"]["min_primary_true_feasible"] = actual_primary
    report = R.summarize(rows, map_ids, train_reg)
    # The old scorer checks start clipping; require the entire saved path here.
    if report["clipped_budget_rows"]:
        report["failures"].append("train path budget input clipped")
        report["pass"] = False
    report["policy_actual_feasible_by_budget"] = {
        str(k): {"feasible": sum(r["actual_feasible"] for r in rows if r["k"] == k),
                 "total": sum(r["k"] == k for r in rows)} for k in reg["grid_k"]}
    report["primary_actual_feasible"] = actual_primary
    report["primary_total"] = len(map_ids)
    report["primary_recognition_required"] = actual_primary
    return report


def train_summary(model, records, reg, scale):
    rows = []
    for (map_id, k), truth in records:
        vc, vh = S.predict(model, truth.observations)
        rows.append({"map_id": map_id, "k": k, **R.value_comparison(truth, vc, vh, scale)})
    return recognition_summary(rows, {m for (m, _), _ in records}, reg)


def fit(model, records, protocol, reg, scale, candidate, journal):
    model.policy.requires_grad_(False)
    before = R.digests(model)
    report = train_summary(model, records, reg, scale)
    R.write_record(journal, {"kind": "train_gate", "update": 0, **report})
    if report["pass"]:
        return 0, report
    if candidate["run_id"] not in protocol["resource_limit"]["update_candidates"]:
        raise ValueError("unexpected source failure outside approved update candidates")
    settings = protocol["fit"]
    optimizer = torch.optim.Adam([p for role in R.SCORE_ROLES for p in model.parameters_of(role)],
                                 lr=settings["learning_rate"])
    rng = np.random.default_rng(settings["sampling_seed"] + candidate["seed"])
    cap = protocol["resource_limit"]["additional_updates_per_candidate_max"]
    for update in range(1, cap + 1):
        loss, components = boundary_loss(model, records, settings, scale, rng)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite boundary loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        R.write_record(journal, {"kind": "update", "update": update,
                                 "loss": loss.item(), **components})
        if update % settings["gate_interval"] == 0 or update == cap:
            report = train_summary(model, records, reg, scale)
            R.write_record(journal, {"kind": "train_gate", "update": update, **report})
            if report["pass"]:
                break
    after = R.digests(model)
    if before["policy"] != after["policy"] or any(before[r] == after[r] for r in R.SCORE_ROLES):
        raise ValueError("actor changed or critic did not update")
    return update, report


def load_sources(protocol):
    upstream, upstream_pin, reg, inputs, maps, _, _ = R.preflight()
    _, saved_pin = S.approved_registration(R.NAV / "config/nav065_saved_refit_prereg_v2.json")
    if (upstream_pin != protocol["source_refit_pin"] or saved_pin != protocol["source_saved_refit_pin"]
            or upstream["base_evaluation_pin"] != protocol["evaluation_pin"]):
        raise ValueError("upstream registration pins differ")
    source_root = R.WORK / protocol["source_run"]
    train_root = R.WORK / protocol["train_source_run"]
    R.verify_hash(source_root / "fitted_candidates.json", protocol["source_manifest_sha256"])
    R.verify_hash(train_root / "fitted_candidates.json", protocol["train_manifest_sha256"])
    source, train = (R.load(p / "fitted_candidates.json") for p in (source_root, train_root))
    if (source["refit_pin"] != saved_pin or train["refit_pin"] != upstream_pin
            or train["train_journal_sha256"] != protocol["train_journal_sha256"]
            or train["train_episodes"] != protocol["resource_limit"]["existing_train_episodes"]):
        raise ValueError("saved source identity differs")
    expected = {c["run_id"] for c in inputs.candidates}
    if len(source["candidates"]) != 6 or {c["run_id"] for c in source["candidates"]} != expected:
        raise ValueError("candidate coverage differs")
    for c in source["candidates"]:
        R.verify_hash(Path(c["checkpoint"]), c["checkpoint_sha256"])
    raw = S.saved_train(inputs, maps, reg["grid_k"], train_root, train)
    return upstream, reg, inputs, maps, source, train, raw


def prepare(protocol, reg, inputs, maps, source, train, raw):
    """All saved actor actions and true primary counts are checked without fitting."""
    by_id = {c["run_id"]: c for c in source["candidates"]}
    design = R.E.M.load_design(R.E.DESIGN)
    specs = {m["map_id"]: R.E.M.build(m["level"], m["seed"], design) for m in maps}
    scale = inputs.value["quantities"]["Z_REF_J"]["value"]
    prepared, steps = [], 0
    for candidate in inputs.candidates:
        rid = candidate["run_id"]
        c = by_id[rid]
        model = R.load_network(inputs, c, c["checkpoint"])
        digests = R.digests(model)
        if digests != c["state_dict_sha256"] or digests["policy"] != candidate["state_dict_sha256"]["policy"]:
            raise ValueError("source tensors or original actor differ")
        records = []
        for m in maps:
            for k in reg["grid_k"]:
                ep = raw[rid][m["map_id"], k]
                truth = R.reconstruct(ep, specs[m["map_id"]], inputs)
                if truth.budget_feature_clipped.any():
                    raise ValueError("clipped saved train observations")
                with torch.inference_mode():
                    actions = model.act(torch.as_tensor(truth.observations, dtype=torch.float32),
                                        deterministic=True)[0].numpy()
                if not np.allclose(actions, [row["action"] for row in ep["trace"]], atol=1e-6, rtol=0):
                    raise ValueError("saved train actor actions differ")
                records.append(((m["map_id"], k), truth))
                steps += len(truth.observations)
        gate = train_summary(model, records, reg, scale)
        if gate["primary_actual_feasible"] != protocol["train_gate"]["primary_policy_feasible_by_candidate"][rid]:
            raise ValueError("fixed-policy primary outcomes differ from protocol")
        if not gate["pass"] and rid not in protocol["resource_limit"]["update_candidates"]:
            raise ValueError("unexpected baseline train failure; no optimization authorized")
        prepared.append((c, model, records, gate))
    if steps != train["train_steps"]:
        raise ValueError("saved train step coverage differs")
    return prepared


def run(protocol, pin, upstream, out, reg, inputs, train, prepared):
    out.mkdir(exist_ok=False)
    scale = inputs.value["quantities"]["Z_REF_J"]["value"]
    fitted, gates, total = [], [], 0
    torch.use_deterministic_algorithms(True)
    for c, model, records, baseline in prepared:
        rid = c["run_id"]
        with (out / (rid + ".jsonl")).open("x") as journal:
            R.write_record(journal, {"kind": "start", "registration_pin": pin,
                                     "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
                                     "source_checkpoint_sha256": c["checkpoint_sha256"]})
            updates, gate = fit(model, records, protocol, reg, scale, c, journal)
        total += updates
        if total > protocol["resource_limit"]["additional_updates_total_max"]:
            raise ValueError("total update cap exceeded")
        checkpoint = out / (rid + ".pt")
        with checkpoint.open("xb") as stream:
            torch.save({"model": model.state_dict(), "boundary_refit_pin": pin,
                        "source_checkpoint_sha256": c["checkpoint_sha256"],
                        "prereg_sha256": inputs.train_pin,
                        "implementation_versions": inputs.prereg["implementation_versions"],
                        "variant": c["variant"], "seed": c["seed"]}, stream)
        if R.digests(R.load_network(inputs, c, checkpoint)) != R.digests(model):
            raise ValueError("saved CPU reload differs")
        fitted.append({**c, "parent_boundary_checkpoint": c["checkpoint"],
                       "parent_boundary_checkpoint_sha256": c["checkpoint_sha256"],
                       "checkpoint": str(checkpoint), "checkpoint_sha256": R.file_sha(checkpoint),
                       "state_dict_sha256": R.digests(model), "boundary_refit_pin": pin,
                       "boundary_refit_updates": updates})
        gates.append({"run_id": rid, "updates": updates, **gate})
        print(rid, "train", gate["pass"], "updates", updates, flush=True)
    for name, record in (
        ("fitted_candidates.json", {"refit_pin": pin, "candidates": fitted,
          "train_episodes": train["train_episodes"], "train_steps": train["train_steps"],
          "new_rollouts": 0, "additional_updates": total}),
        ("train_gate.json", {"candidates": gates, "pass": all(g["pass"] for g in gates)}),
    ):
        with (out / name).open("x") as stream:
            R.write_record(stream, record)
    if not all(g["pass"] for g in gates):
        with (out / "HELD_TRAIN_GATE.json").open("x") as stream:
            R.write_record(stream, {"reason": "train gate failed; development not accessed",
                                    "new_rollouts": 0, "development_scoring_rounds": 0})
        return
    R.evaluate(out, upstream, pin, reg, inputs, fitted, R.WORK / "nav065_deployment_v1",
               train["train_episodes"], train["train_steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report", type=Path, help="new preflight report, outside run directory")
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    if args.preflight and args.registration or args.report and not args.preflight:
        parser.error("preflight and formal fitting are separate modes")
    torch.set_num_threads(1)
    try:
        if args.preflight:
            protocol, pin = R.load(PROPOSAL), None
        elif args.registration:
            protocol, pin = approved_registration(args.registration)
        else:
            raise ValueError("requires --preflight or separately approved --registration")
        out = R.WORK / protocol["output_name"]
        if out.exists():
            raise ValueError("output already exists; inspect old attempt before any action")
        if args.report and (args.report.exists() or args.report.resolve().is_relative_to(out.resolve())):
            raise ValueError("preflight report must be a new file outside the formal run")
        upstream, reg, inputs, maps, source, train, raw = load_sources(protocol)
        prepared = prepare(protocol, reg, inputs, maps, source, train, raw)
        if args.preflight:
            report = {"task": "NAV-065", "status": "PREFLIGHT_PASS_NOT_TRAINED",
                      "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
                      "proposal_sha256": R.file_sha(PROPOSAL), "new_rollouts": 0,
                      "optimizer_updates": 0, "development_scoring_rounds": 0,
                      "train_episodes": train["train_episodes"], "train_steps": train["train_steps"],
                      "candidates": [{"run_id": c["run_id"], **gate} for c, _, _, gate in prepared]}
            if args.report:
                with args.report.open("x") as stream:
                    R.write_record(stream, report)
            print(json.dumps({k: v for k, v in report.items() if k != "candidates"}), flush=True)
            for c, _, _, gate in prepared:
                print(c["run_id"], "baseline_pass", gate["pass"], "policy_primary",
                      gate["primary_actual_feasible"], "/", gate["primary_total"], flush=True)
            return
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    run(protocol, pin, upstream, out, reg, inputs, train, prepared)


if __name__ == "__main__":
    main()
