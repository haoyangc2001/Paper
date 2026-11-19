"""Read-only NAV-065 diagnosis using frozen train/development trajectories.

No rollout, optimization, checkpoint selection, or threshold change is performed.
Outputs describe the observed samples, not a safety certificate.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_deployment as R


def predict(model, observations):
    with torch.inference_mode():
        parts = [model.values(torch.as_tensor(part, dtype=torch.float32))
                 for part in np.array_split(observations, max(1, math.ceil(len(observations) / 2048)))]
    return tuple(np.concatenate([part[i].numpy() for part in parts]) for i in (0, 1))


def case(ep, truth, vc, vh, scale):
    residual = np.maximum(vh, vc - truth.budget_J / scale)
    stop = int(truth.stop_index[0])
    result = {
        "actual_feasible": truth.feasible,
        "predicted_feasible": bool(residual[0] <= 0),
        "false_feasible": bool(not truth.feasible and residual[0] <= 0),
        "false_infeasible": bool(truth.feasible and residual[0] > 0),
        "label": ep["label"], "steps": ep["n_steps"],
        "safe": ep["trace"][-1]["safety_y"] == 0,
        "start_budget_J": float(truth.budget_J[0]),
        "overspend_J": max(0.0, -ep["z_J_end"]),
        "stop_index": stop, "stop_before_terminal": stop < ep["n_steps"],
        "true_cost_target_J": float(truth.cost_target[0] * scale),
        "true_cost_margin_J": float(truth.cost_target[0] * scale - truth.budget_J[0]),
        "true_reach": float(truth.reach_target[0]),
        "true_W": float(truth.path_residual[0]),
        "predicted_cost_J": float(vc[0] * scale),
        "predicted_cost_margin_J": float(vc[0] * scale - truth.budget_J[0]),
        "predicted_reach": float(vh[0]), "predicted_W": float(residual[0]),
        "cost_error_J": float((vc[0] - truth.cost_target[0]) * scale),
        "reach_error": float(vh[0] - truth.reach_target[0]),
        "path_cost_absolute_error_sum_J": float(np.abs(vc - truth.cost_target).sum() * scale),
        "path_reach_absolute_error_sum": float(np.abs(vh - truth.reach_target).sum()),
        "path_sign_mismatches": int(np.count_nonzero((residual <= 0) != (truth.path_residual <= 0))),
        "target_reach_negative_rows": int(np.count_nonzero(truth.reach_target < 0)),
        "target_reach_nonnegative_rows": int(np.count_nonzero(truth.reach_target >= 0)),
    }
    if stop < ep["n_steps"]:
        result["stop_goal_distance_m"] = (ep["trace"][stop - 1]["goal_distance_m"] if stop else None)
        result["cost_after_stop_J"] = float(ep["cost_model_J"] - truth.cost_target[0] * scale)
    return result


def summarize(rows):
    n = len(rows)
    steps = sum(r["steps"] for r in rows)
    errors = np.asarray([r["cost_error_J"] for r in rows])
    return {
        "episodes": n, "steps": steps,
        "actual_feasible": sum(r["actual_feasible"] for r in rows),
        "false_feasible": sum(r["false_feasible"] for r in rows),
        "false_infeasible": sum(r["false_infeasible"] for r in rows),
        "start_cost_mae_J": float(np.abs(errors).mean()),
        "start_cost_signed_bias_J": float(errors.mean()),
        "start_reach_mae": sum(abs(r["reach_error"]) for r in rows) / n,
        "path_cost_mae_J": sum(r["path_cost_absolute_error_sum_J"] for r in rows) / steps,
        "path_reach_mae": sum(r["path_reach_absolute_error_sum"] for r in rows) / steps,
        "path_sign_mismatches": sum(r["path_sign_mismatches"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    if args.out.parent.resolve() != (R.NAV / "reports").resolve() or args.out.exists():
        parser.error("requires a new report in navigation/reports")
    torch.set_num_threads(1)
    protocol, pin, reg, inputs, train_maps, _, previous = R.preflight()
    fitted_root = R.WORK / protocol["output_name"]
    review_path = R.NAV / "reports/nav065_refit_v1_review.json"
    review = R.load(review_path)
    # Protect all original formal outputs and checkpoints, before and after scoring.
    guarded = {fitted_root / name: digest for name, digest in review["output_hashes"].items()}
    # The generated candidate manifest is the producer's journal binding. Check
    # its frozen digest before using it to diagnose any review transcription error.
    R.verify_hash(fitted_root / "fitted_candidates.json", review["output_hashes"]["fitted_candidates.json"])
    manifest = R.load(fitted_root / "fitted_candidates.json")
    metadata_discrepancies = []
    if guarded[fitted_root / "train_episodes.jsonl"] != manifest["train_journal_sha256"]:
        metadata_discrepancies.append({
            "source": str(review_path), "field": "output_hashes.train_episodes.jsonl",
            "recorded": guarded[fitted_root / "train_episodes.jsonl"],
            "producer_manifest_sha256": manifest["train_journal_sha256"],
            "action": "Preserve original review; verify journal against the hash-bound producer manifest.",
        })
        guarded[fitted_root / "train_episodes.jsonl"] = manifest["train_journal_sha256"]
    fitted = {c["run_id"]: c for c in manifest["candidates"]}
    guarded.update({Path(c["checkpoint"]): c["checkpoint_sha256"] for c in fitted.values()})
    guarded.update({Path(c["checkpoint"]): c["checkpoint_sha256"] for c in inputs.candidates})
    for path, digest in guarded.items():
        R.verify_hash(path, digest)
    train = defaultdict(dict)
    fit_records = {}
    with (fitted_root / "train_episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "rollout_result":
                key = row["map_id"], row["k"]
                if key in train[row["run_id"]]:
                    raise ValueError("duplicate training trajectory")
                train[row["run_id"]][key] = row["episode"]
            elif row["kind"] == "fit_result":
                if row["run_id"] in fit_records:
                    raise ValueError("duplicate fit record")
                fit_records[row["run_id"]] = row
    dev, _ = R.fixed_dev(inputs, reg["grid_k"], previous)
    saved_dev = {}
    with (fitted_root / "development_scores.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["run_id"], row["map_id"], row["k"]
            if key in saved_dev:
                raise ValueError("duplicate development score")
            saved_dev[key] = row
    scale = inputs.value["quantities"]["Z_REF_J"]["value"]
    rows, losses = [], {}
    saved_error = 0.0
    train_keys = {(m["map_id"], k) for m in train_maps for k in reg["grid_k"]}
    for candidate in inputs.candidates:
        rid = candidate["run_id"]
        if set(train[rid]) != train_keys:
            raise ValueError("training grid differs")
        models = {"before": R.load_network(inputs, candidate, candidate["checkpoint"]),
                  "after": R.load_network(inputs, candidate, fitted[rid]["checkpoint"])}
        if R.digests(models["before"])["policy"] != R.digests(models["after"])["policy"]:
            raise ValueError("actor identity differs")
        for model in models.values():
            model.requires_grad_(False)
        curve = fit_records[rid]["losses"]
        if len(curve) != protocol["fit"]["updates"]:
            raise ValueError("wrong update count")
        losses[rid] = {"initial": curve[0], "final": curve[-1], "updates": len(curve)}
        for split, maps, episodes in (("train", train_maps, train[rid]),
                                      ("development", inputs.maps, dev[rid])):
            specs = {m["map_id"]: R.E.M.build(m["level"], m["seed"], R.E.M.load_design(R.E.DESIGN))
                     for m in maps}
            for (map_id, k), ep in episodes.items():
                truth = R.reconstruct(ep, specs[map_id], inputs)
                if truth.budget_feature_clipped.any():
                    raise ValueError("budget clipping")
                for version, model in models.items():
                    vc, vh = predict(model, truth.observations)
                    result = case(ep, truth, vc, vh, scale)
                    rows.append({"run_id": rid, "variant": candidate["variant"], "seed": candidate["seed"],
                                 "split": split, "version": version, "map_id": map_id, "k": k, **result})
                    if split == "development" and version == "after":
                        saved = saved_dev[rid, map_id, k]
                        saved_error = max(saved_error, float(np.max(np.abs(vc - saved["rows"]["vc_norm"]))),
                                          float(np.max(np.abs(vh - saved["rows"]["vh"])) ))
                        if saved["predicted_feasible"] != result["predicted_feasible"]:
                            raise ValueError("saved decision differs")
        print(rid + " diagnosed", flush=True)
    coverage = Counter((r["split"], r["version"]) for r in rows)
    if coverage != {("train", "before"): 540, ("train", "after"): 540,
                    ("development", "before"): 180, ("development", "after"): 180}:
        raise ValueError("diagnostic coverage differs")
    if saved_error > 1e-6:
        raise ValueError("saved scores differ beyond floating-point tolerance")
    for path, digest in guarded.items():
        R.verify_hash(path, digest)
    summary = {split: {version: summarize([r for r in rows if r["split"] == split and r["version"] == version])
                       for version in ("before", "after")} for split in ("train", "development")}
    grouped = []
    for candidate in inputs.candidates:
        for split in ("train", "development"):
            for version in ("before", "after"):
                group = [r for r in rows if r["run_id"] == candidate["run_id"]
                         and r["split"] == split and r["version"] == version]
                grouped.append({"run_id": candidate["run_id"], "split": split, "version": version,
                                **summarize(group)})
    report = {"schema": "camra-nav065-boundary-diagnosis-v1", "task": "NAV-065",
              "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
              "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
              "refit_pin": pin, "new_rollouts": 0, "optimizer_steps": 0,
              "threshold_changes": 0, "saved_prediction_max_abs_difference": saved_error,
              "source_metadata_discrepancies": metadata_discrepancies,
              "source_hashes": {str(p): h for p, h in guarded.items()},
              "script_sha256": R.file_sha(Path(__file__)), "summary": summary,
              "per_candidate": grouped, "training_loss": losses,
              "false_feasible_cases": [r for r in rows if r["false_feasible"]],
              "episode_rows": rows}
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
