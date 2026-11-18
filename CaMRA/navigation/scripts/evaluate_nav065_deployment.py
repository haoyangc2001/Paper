"""Evaluate the six fixed NAV-064 networks after NAV-065 preregistration."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import torch

from audit_nav065_saved_trajectories import NAV, E, load, verify_hash
from eval_r14_repair import build_registered_network, rollout
from prepare_nav065_sources import REGISTRATION, tensor_digest
from train_formal import verify_prereg

from camra_navigation.algorithm.deployment_oracle import joint_residual, reconstruct_truth
from camra_navigation.algorithm.validation import file_sha
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.costs import DT_NAV_S

REG_PATH = NAV / "config/nav065_deployment_prereg_v1.json"
WORK = Path(E.WORK_ROOT)
EPISODE_GRID = (0.70, 0.85, 1.0, 1.15, 1.3)
SAVED_GRID = (1.0, 1.15, 1.3)
SCORE_KEYS = {"max_false_feasible", "min_primary_true_feasible",
              "rel_mae_max", "pearson_min", "variance_ratio_min"}


def read_registration(path):
    if not path.is_file() or not path.with_suffix(".json.sha256").is_file():
        raise ValueError("NAV-065 requires a separately approved registration and SHA256 pin")
    pin = path.with_suffix(".json.sha256").read_text().strip()
    verify_hash(path, pin)
    reg = load(path)
    authorization = reg["authorization"]
    if (reg["schema"] != "camra-nav065-deployment-v1" or authorization["evaluation_approved"] is not True
            or not authorization["decision_reference"] or reg["split"] != "development"):
        raise ValueError("invalid NAV-065 development approval")
    if (tuple(reg["grid_k"]) != EPISODE_GRID or reg["max_new_episodes"] != 72
            or reg["primary_k"] != 1.15):
        raise ValueError("registration does not match this evaluator's coverage")
    limits = reg["criteria"]
    if set(limits) != SCORE_KEYS or any(not math.isfinite(v) for v in limits.values()):
        raise ValueError("missing or invalid registered criteria")
    if not (limits["max_false_feasible"] >= 0 and 0 < limits["min_primary_true_feasible"] <= 6
            and limits["rel_mae_max"] >= 0 and -1 <= limits["pearson_min"] <= 1
            and limits["variance_ratio_min"] > 0):
        raise ValueError("invalid registered limits")
    return reg, pin


def value_comparison(truth, vc, vh, z_ref):
    """Feasibility counts use episode starts; error uses aligned per-step rows."""
    w = joint_residual(vc, vh, truth.budget_J, z_ref)
    predicted = bool(w[0] <= 0.0)
    path_feasible = bool(truth.path_residual[0] <= 0.0)
    if path_feasible != truth.feasible:
        raise ValueError("independent residual sign and trajectory feasibility disagree")
    return {"actual_feasible": truth.feasible, "path_feasible": path_feasible,
            "predicted_feasible": predicted,
            "start_budget_J": float(truth.budget_J[0]), "start_vc_norm": float(vc[0]),
            "start_vh": float(vh[0]), "start_W": float(w[0]),
            "start_budget_clipped": bool(truth.budget_feature_clipped[0]),
            "clipped_budget_rows": int(truth.budget_feature_clipped.sum()),
            "rows": {"vc_norm": vc.tolist(), "vc_target_norm": truth.cost_target.tolist(),
                     "vh": vh.tolist(), "vh_target": truth.reach_target.tolist(),
                     "W": w.tolist(), "path_W": truth.path_residual.tolist(),
                     "shared_stop": truth.stop_index.tolist()}}


def error_metrics(prediction, target, variance_ratio_min):
    if (prediction.shape != target.shape or prediction.size == 0
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("critic predictions and independent targets must be finite and aligned")
    error = np.abs(prediction - target)
    mae = float(error.mean())
    scale = float(np.abs(target).mean())
    target_std = float(target.std())
    pearson_required = target_std > 1e-12 and target_std >= variance_ratio_min * scale
    pearson = (float(np.corrcoef(prediction, target)[0, 1])
               if target_std > 1e-12 and prediction.std() > 1e-12 else None)
    return {"n": len(target), "mae": mae, "mean_abs_target": scale,
            "relative_mae": mae / scale if scale > 1e-12 else None,
            "target_std": target_std, "pearson": pearson, "pearson_required": pearson_required}


def summarize(episodes, map_ids, reg):
    expected = {(k, m) for k in reg["grid_k"] for m in map_ids}
    actual = [(ep["k"], ep["map_id"]) for ep in episodes]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicated episode coverage")
    counts = {}
    for k in reg["grid_k"]:
        group = [ep for ep in episodes if ep["k"] == k]
        sign_mismatches = 0
        for ep in group:
            predicted_path = np.asarray(ep["rows"]["W"]) <= 0.0
            actual_path = np.asarray(ep["rows"]["path_W"]) <= 0.0
            if predicted_path.shape != actual_path.shape:
                raise ValueError("critic and oracle residual horizons differ")
            sign_mismatches += int(np.count_nonzero(predicted_path != actual_path))
        counts[str(k)] = {
            "total": len(group),
            "true_feasible": sum(ep["actual_feasible"] and ep["predicted_feasible"] for ep in group),
            "false_feasible": sum(not ep["actual_feasible"] and ep["predicted_feasible"] for ep in group),
            "false_infeasible": sum(ep["actual_feasible"] and not ep["predicted_feasible"] for ep in group),
            "true_infeasible": sum(not ep["actual_feasible"] and not ep["predicted_feasible"] for ep in group),
            "path_sign_mismatches": sign_mismatches,
        }
    limits = reg["criteria"]
    metrics, failures = {}, []
    diagnostics = {}
    for name in ("vc_norm", "vh"):
        target_name = {"vc_norm": "vc_target_norm", "vh": "vh_target"}[name]
        pred, target = (np.concatenate([ep["rows"][key] for ep in episodes])
                        for key in (name, target_name))
        m = metrics[name] = error_metrics(pred, target, limits["variance_ratio_min"])
        if m["relative_mae"] is None or m["relative_mae"] > limits["rel_mae_max"]:
            failures.append(f"{name} relative MAE failed or undefined")
        if m["pearson_required"] and (m["pearson"] is None or m["pearson"] < limits["pearson_min"]):
            failures.append(f"{name} Pearson failed or undefined")
    pred_w = np.concatenate([ep["rows"]["W"] for ep in episodes])
    target_w = np.concatenate([ep["rows"]["path_W"] for ep in episodes])
    diagnostics["W"] = error_metrics(pred_w, target_w, limits["variance_ratio_min"])
    clipped = sum(ep["clipped_budget_rows"] for ep in episodes)
    start_clipped = sum(ep["start_budget_clipped"] for ep in episodes)
    if start_clipped:
        failures.append("initial budget input clipped")
    if sum(c["false_feasible"] for c in counts.values()) > limits["max_false_feasible"]:
        failures.append("false feasible budget at episode start")
    if counts[str(reg["primary_k"])]["true_feasible"] < limits["min_primary_true_feasible"]:
        failures.append("known feasible primary starts were not accepted")
    return {"pass": not failures, "failures": failures, "metrics": metrics,
            "diagnostic_metrics": diagnostics,
            "by_budget": counts, "clipped_budget_rows": clipped,
            "clipped_initial_budget_inputs": start_clipped}


def write_record(stream, record):
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


@dataclass(frozen=True)
class EvaluationInputs:
    prereg: dict
    train_pin: str
    nav: dict
    value: dict
    maps: list
    candidates: list
    runs: dict
    budget_J: dict
    validation_evidence: dict


def check_inputs():
    """Check read-only upstream evidence without loading networks or running episodes."""
    prereg, train_pin = verify_prereg(REGISTRATION)
    source = load(NAV / "reports/nav065_deployment_source_binding_v1.json")
    ready = load(NAV / "reports/nav065_saved_trajectory_compatibility_v1.json")
    if (source["prereg_sha256"] != train_pin or ready["prereg_sha256"] != train_pin
            or ready["status"] != "TRACE_COMPATIBILITY_ONLY"):
        raise ValueError("wrong upstream source")
    plan = load(NAV.parents[1] / "doc/plan/20261109_camra_navigation_implementation.json")
    status = {t["id"]: t["status"] for t in plan["tasks"]}
    if not (status["NAV-063"] and status["NAV-064"]) or status["NAV-065"]:
        raise ValueError("NAV-065 dependencies or execution state differ")
    upstream_gate = NAV / "reports/gate_feedback_v1_attempt4.json"
    if load(upstream_gate)["verdict"] != "PASS":
        raise ValueError("NAV-063 gate is not PASS")
    accepted_path = NAV / "reports/nav064_terminal_tail_acceptance_v1.json"
    verify_hash(accepted_path, source["source_evidence"]["sha256"])
    accepted = load(accepted_path)
    if accepted["verdict"] != "PASS" or accepted["prereg_sha256"] != train_pin:
        raise ValueError("NAV-064 acceptance mismatch")
    for path, digest in source["protected_previous_results"].items():
        verify_hash(Path(path), digest)
    for path, digest in source["contracts"].items():
        verify_hash(Path(path), digest)
    release_path = NAV / "reports/nav064_terminal_tail_training_release_v1.json"
    verify_hash(release_path, accepted["evidence"][str(release_path)])
    for path, digest in load(release_path)["v2_verified_files"].items():
        verify_hash(Path(path), digest)
    repair_path = NAV / "reports/nav064_terminal_tail_repair_v1.json"
    verify_hash(repair_path, accepted["evidence"][str(repair_path)])
    for path, item in load(repair_path)["source_after"].items():
        verify_hash(NAV / path, item["sha256"])

    accepted_dev = NAV / "reports/nav064_terminal_tail_v1_development.json"
    refs_path = accepted_dev.with_suffix(".references.json")
    if ready["development_sha256"] != accepted["evidence"][str(accepted_dev)]:
        raise ValueError("trajectory audit used a different development file")
    verify_hash(accepted_dev, accepted["evidence"][str(accepted_dev)])
    verify_hash(refs_path, accepted["evidence"][str(refs_path)])
    dev, refs = load(accepted_dev), load(refs_path)
    if (dev["data_access"]["rollout_split"] != "development" or dev["prereg_sha256"] != train_pin
            or refs["t_deadline_s"] != prereg["environment"]["t_max_s"]
            or list(SAVED_GRID) != prereg["round_acceptance"]["budget_response_on_dev"]["grid_k"]):
        raise ValueError("unexpected NAV-064 data protocol")
    for path in (E.DESIGN, E.SPLIT, E.POWER, E.MARGINS, E.FOOTPRINT, E.GEOM):
        verify_hash(Path(path), dev["data_access"]["paths"][path])
    maps = load(Path(E.SPLIT))["splits"]["development"]["maps"]
    map_ids = {m["map_id"] for m in maps}
    if (len(maps) != 6 or map_ids != set(refs["maps"]) or map_ids != set(dev["data_access"]["maps"])
            or map_ids & set(dev["data_access"]["training_map_ids"])
            or not all(r["admitted"] and math.isfinite(r["e_ref_J"]) and r["e_ref_J"] > 0
                       for r in refs["maps"].values())):
        raise ValueError("unexpected map or budget reference grid")
    nav, value = (load(NAV / prereg["binding"][kind]["path"])
                  for kind in ("navigation_contract", "value_contract"))
    q = value["quantities"]
    z_ref = q["Z_REF_J"]["value"]
    clip_min, clip_max = q["z_ratio"]["range"]
    runs = {r["run_id"]: r for r in dev["runs"]}
    candidates = source["candidates"]
    expected_saved = {(k, m) for k in SAVED_GRID for m in map_ids}
    accepted_runs = {r["run_id"]: r for r in accepted["runs"]}
    if not (len(dev["runs"]) == len(runs) == len(candidates) == len(accepted_runs) == 6
            and set(runs) == {r["run_id"] for r in candidates} == set(accepted_runs)):
        raise ValueError("saved development candidate coverage is inconsistent")
    budgets = {}
    for candidate, identity in zip(candidates, prereg["run_registry"], strict=True):
        if any(candidate[k] != v for k, v in identity.items()):
            raise ValueError("candidate identity mismatch")
        saved_run = runs[candidate["run_id"]]
        keys = [(e["k"], e["map_id"]) for e in saved_run["episodes"]]
        accepted_run = accepted_runs[candidate["run_id"]]
        if (len(keys) != len(expected_saved) or set(keys) != expected_saved
                or not saved_run["state_sha256"] == candidate["checkpoint_sha256"] == accepted_run["state_sha256"]
                or candidate["checkpoint"] != accepted_run["source_checkpoint"]):
            raise ValueError("incomplete saved episode grid or mismatched checkpoint")
        variant = candidate["variant"]
        for map_id in map_ids:
            e_ref = consistent_e_ref(refs["maps"][map_id]["e_ref_J"], variant)
            encoded = [k * e_ref / z_ref for k in EPISODE_GRID]
            if (not all(clip_min < z < clip_max for z in encoded)
                    or any(a >= b for a, b in zip(encoded, encoded[1:]))):
                raise ValueError(f"search budget inputs lose resolution: {variant} {map_id}")
            budgets.update({(variant, map_id, k): k * e_ref for k in EPISODE_GRID})
        for episode in saved_run["episodes"]:
            expected = budgets[variant, episode["map_id"], episode["k"]]
            if not math.isclose(episode["z0_J"], expected, rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"saved trajectory budget differs: {candidate['run_id']}")
        verify_hash(Path(candidate["checkpoint"]), candidate["checkpoint_sha256"])
    evidence = {name: {"path": str(path), "sha256": file_sha(path)} for name, path in (
        ("NAV-063", upstream_gate), ("NAV-064", accepted_path),
        ("source_binding", NAV / "reports/nav065_deployment_source_binding_v1.json"),
        ("original_development", accepted_dev), ("budget_references", refs_path),
    )}
    evidence["contracts"] = source["contracts"]
    evidence["protected_previous_results"] = source["protected_previous_results"]
    return EvaluationInputs(prereg, train_pin, nav, value, maps, candidates, runs, budgets, evidence)


def review_documents(score, candidates, evidence, validation_sha256, implementation_versions):
    """Prepare review documents in the attempt directory; never change plan state."""
    results = score["candidates"]
    expected = [c["run_id"] for c in candidates]
    if (len(set(expected)) != len(expected) or len(expected) != 6
            or [r["run_id"] for r in results] != expected):
        raise ValueError("candidate validation is not complete")
    gates = {"numerical_alignment": "NAV-063" in evidence,
             "learning_available": "NAV-064" in evidence,
             "deployment_values": bool(all(r["pass"] for r in results)),
             "evaluation_coverage": score["new_episodes"] == 72 and all(
                 len(r["by_budget"]) == len(EPISODE_GRID)
                 and all(b["total"] == 6 for b in r["by_budget"].values())
                 for r in results)}
    passed = all(gates.values())
    gate = {"task": "NAV-065", "schema": "camra-gate-G4-alignment-v2",
            "verdict": "PASS" if passed else "FAIL", "checks": gates,
            "validation_sha256": validation_sha256, "eval_pin": score["eval_pin"],
            "evidence": evidence, "release_state": "attempt_review"}
    bundle = None
    if passed:
        bundle = {
            "task": "NAV-065", "version": 2, "schema": "camra-deployment-candidates-v2",
            "release_state": "attempt_review", "n_candidates": len(candidates),
            "prereg_sha256": score["train_pin"], "eval_pin": score["eval_pin"],
            "validation_sha256": validation_sha256,
            "implementation_versions": implementation_versions,
            "binding_contracts": evidence["contracts"],
            "candidates": candidates,
        }
    return gate, bundle


def run(reg, pin, out):
    for name in (
        "scripts/evaluate_nav065_deployment.py",
        "src/camra_navigation/algorithm/deployment_oracle.py",
        "reports/nav065_deployment_source_binding_v1.json",
        "reports/nav065_saved_trajectory_compatibility_v1.json",
        "reports/gate_feedback_v1_attempt4.json",
    ):
        verify_hash(NAV / name, reg["inputs"][name])
    inputs = check_inputs()
    prereg, train_pin = inputs.prereg, inputs.train_pin
    z_ref = inputs.value["quantities"]["Z_REF_J"]["value"]
    map_ids = {m["map_id"] for m in inputs.maps}

    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    summaries = []
    used = 0
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as journal:
        write_record(journal, {"kind": "start", "eval_pin": pin, "train_pin": train_pin,
                              "host": socket.getfqdn(), "tmux": os.environ["TMUX"]})
        for candidate in inputs.candidates:
            rid = candidate["run_id"]
            variant = candidate["variant"]
            ctx = E.build_rollout_ctx(variant)
            net = build_registered_network(inputs.nav, prereg)
            state = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
            net.load_state_dict(state["model"], strict=True)
            net.eval()
            if any(tensor_digest(getattr(net, role)) != candidate["state_dict_sha256"][role]
                   for role in net.ROLES):
                raise ValueError("reloaded model differs from source binding")
            saved = {(ep["k"], ep["map_id"]): ep for ep in inputs.runs[rid]["episodes"]}
            rows = []
            for m in inputs.maps:
                map_id = m["map_id"]
                spec = E.M.build(m["level"], m["seed"], ctx["design"])
                for k in EPISODE_GRID:
                    key = {"run_id": rid, "k": k, "map_id": map_id}
                    budget_J = inputs.budget_J[variant, map_id, k]
                    entry = saved.get((k, map_id))
                    if entry is None:
                        if used >= reg["max_new_episodes"]:
                            raise ValueError("new-episode quota exhausted")
                        used += 1
                        write_record(journal, {"kind": "rollout_start", **key})
                        entry = rollout(net, spec, ctx, budget_J, prereg)
                        write_record(journal, {"kind": "rollout_result", **key, "episode": entry})
                    if not math.isclose(entry["z0_J"], budget_J, rel_tol=1e-12, abs_tol=0):
                        raise ValueError("episode budget differs from reference budget")
                    truth = reconstruct_truth(entry, initial_goal_distance_m=math.dist(spec.start, spec.goal),
                        value_contract=inputs.value, horizon_s=prereg["environment"]["t_max_s"],
                        goal_radius_m=prereg["environment"]["r_goal_m"], step_s=DT_NAV_S)
                    with torch.inference_mode():
                        vc, vh = net.values(torch.as_tensor(truth.observations, dtype=torch.float32))
                    row = {**key, "label": entry["label"],
                           "source": "saved" if (k, map_id) in saved else "new",
                           **value_comparison(truth, vc.numpy(), vh.numpy(), z_ref)}
                    write_record(journal, {"kind": "scored", **row})
                    rows.append(row)
            summaries.append({"run_id": rid, **summarize(rows, map_ids, reg)})
    score = {"task": "NAV-065", "gate_eligible": all(r["pass"] for r in summaries),
             "eval_pin": pin, "train_pin": train_pin, "new_episodes": used,
             "episodes_sha256": file_sha(out / "episodes.jsonl"), "candidates": summaries}
    validation_path = out / "deployment_critic_validation_v2.json"
    with validation_path.open("x", encoding="utf-8") as stream:
        write_record(stream, score)
    gate, bundle = review_documents(score, inputs.candidates, inputs.validation_evidence,
                                    file_sha(validation_path), prereg["implementation_versions"])
    with (out / "gate_G4_alignment_v2.json").open("x", encoding="utf-8") as stream:
        write_record(stream, gate)
    if bundle is not None:
        with (out / "deployment_candidates_v2.json").open("x", encoding="utf-8") as stream:
            write_record(stream, bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    if args.out.parent.resolve() != WORK.resolve() or not args.out.name.startswith("nav065_deployment_"):
        parser.error("expected a new nav065_deployment_* directory under the navigation work root")
    try:
        reg, pin = read_registration(REG_PATH)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.out.name != reg["output_name"]:
        parser.error("output directory differs from the approved registration")
    run(reg, pin, args.out)


if __name__ == "__main__":
    main()
