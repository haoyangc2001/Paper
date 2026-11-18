"""Bounded, approval-gated critic refit using only saved NAV-065 train traces.

--preflight is read-only. Actual optimization requires a separate approved,
hash-pinned registration; the proposal supplied with this script cannot run it.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

import refit_nav065_deployment as R
from analyze_nav065_boundary import predict

PROPOSAL = R.NAV / "config/nav065_saved_refit_proposal_v2.json"
OUTPUT_NAME = "nav065_saved_refit_v2"
BOUND_INPUTS = (
    "scripts/refit_nav065_saved.py", "scripts/analyze_nav065_boundary.py",
    "tests/test_nav065_saved_refit.py", "config/nav065_saved_refit_proposal_v2.json",
)


def approved_registration(path):
    """Require exact proposed settings, explicit approval and matching source pins."""
    proposal = R.load(PROPOSAL)
    protocol = R.load(path)
    pin_file = path.with_suffix(".json.sha256")
    if not pin_file.is_file():
        raise ValueError("requires approved registration and SHA256 pin")
    pin = pin_file.read_text().strip()
    R.verify_hash(path, pin)
    if (protocol.get("approved") is not True or protocol.get("status") != "APPROVED"
            or not protocol.get("authorization_reference")):
        raise ValueError("additional critic updates require explicit approval")
    expected = {k: v for k, v in proposal.items() if k not in ("approved", "status")}
    actual = {k: v for k, v in protocol.items()
              if k not in ("approved", "status", "authorization_reference", "implementation_hashes")}
    if actual != expected:
        raise ValueError("approved settings differ from the reviewed proposal")
    hashes = protocol.get("implementation_hashes", {})
    if set(hashes) != set(BOUND_INPUTS):
        raise ValueError("missing exact implementation binding")
    for name, digest in hashes.items():
        R.verify_hash(R.NAV / name, digest)
    return protocol, pin


def saved_train(inputs, maps, grid, root, manifest):
    """Check the full, unique train grid before any new optimization."""
    R.verify_hash(root / "train_episodes.jsonl", manifest["train_journal_sha256"])
    raw = {c["run_id"]: {} for c in inputs.candidates}
    expected = {(m["map_id"], k) for m in maps for k in grid}
    with (root / "train_episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] != "rollout_result":
                continue
            rid, key = row["run_id"], (row["map_id"], row["k"])
            if rid not in raw or key not in expected or key in raw[rid]:
                raise ValueError("duplicate or unknown saved train key")
            raw[rid][key] = row["episode"]
    if any(set(episodes) != expected for episodes in raw.values()):
        raise ValueError("incomplete saved train grid")
    return raw


def train_summary(model, records, reg, scale):
    rows = []
    map_ids = {m for (m, _), _ in records}
    for (map_id, k), truth in records:
        vc, vh = predict(model, truth.observations)
        rows.append({"map_id": map_id, "k": k,
                     **R.value_comparison(truth, vc, vh, scale)})
    train_reg = copy.deepcopy(reg)
    train_reg["criteria"]["min_primary_true_feasible"] = len(map_ids)
    return R.summarize(rows, map_ids, train_reg)


def regression_loss(model, records, settings, rng):
    """Three equally weighted regression terms with unchanged value targets."""
    starts, interior = [], []
    for _, truth in records:
        n = len(truth.observations)
        idx = rng.choice(np.arange(1, n) if n > 1 else [0],
                         size=settings["interior_samples_per_episode"],
                         replace=n - 1 < settings["interior_samples_per_episode"])
        starts.append((truth, np.asarray([0])))
        interior.append((truth, idx))
    boundary = [(t, idx) for t, idx in starts
                if abs(t.path_residual[0]) <= settings["boundary_absolute_W_max"]]

    def mse(samples):
        obs = torch.as_tensor(np.concatenate([t.observations[i] for t, i in samples]), dtype=torch.float32)
        targets = [torch.as_tensor(np.concatenate([getattr(t, name)[i] for t, i in samples]),
                                   dtype=torch.float32) for name in ("cost_target", "reach_target")]
        vc, vh = model.values(obs)
        return ((vc - targets[0]).square().mean() + (vh - targets[1]).square().mean()) / 2

    terms = [mse(starts), mse(interior)]
    if boundary:
        terms.append(mse(boundary))
    return torch.stack(terms).mean(), len(boundary)


def fit(model, records, protocol, reg, scale, seed, journal):
    model.policy.requires_grad_(False)
    before = R.digests(model)
    optimizer = torch.optim.Adam([p for role in R.SCORE_ROLES for p in model.parameters_of(role)],
                                 lr=protocol["fit"]["learning_rate"])
    rng = np.random.default_rng(protocol["fit"]["sampling_seed"] + seed)
    for update in range(1, protocol["resource_limit"]["additional_updates_per_candidate_max"] + 1):
        loss, boundary_count = regression_loss(model, records, protocol["fit"], rng)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite saved-trajectory refit loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        R.write_record(journal, {"kind": "update", "update": update,
                                 "loss": loss.item(), "boundary_starts": boundary_count})
        if update % 10 == 0:
            report = train_summary(model, records, reg, scale)
            R.write_record(journal, {"kind": "train_gate", "update": update, **report})
            if report["pass"]:
                break
    after = R.digests(model)
    if before["policy"] != after["policy"] or any(before[r] == after[r] for r in R.SCORE_ROLES):
        raise ValueError("actor changed or a critic did not update")
    return update, report


def run(protocol, pin, upstream, out, inputs, maps, reg, root, manifest):
    raw = saved_train(inputs, maps, reg["grid_k"], root, manifest)
    out.mkdir(exist_ok=False)
    torch.use_deterministic_algorithms(True)
    scale = inputs.value["quantities"]["Z_REF_J"]["value"]
    fits, gates = [], []
    by_id = {c["run_id"]: c for c in manifest["candidates"]}
    design = R.E.M.load_design(R.E.DESIGN)
    specs = {m["map_id"]: R.E.M.build(m["level"], m["seed"], design) for m in maps}
    for candidate in inputs.candidates:
        rid = candidate["run_id"]
        source = by_id[rid]
        model = R.load_network(inputs, source, source["checkpoint"])
        records = [((m["map_id"], k), R.reconstruct(raw[rid][m["map_id"], k], specs[m["map_id"]], inputs))
                   for m in maps for k in reg["grid_k"]]
        if any(t.budget_feature_clipped.any() for _, t in records):
            raise ValueError("clipped saved train observation")
        with (out / (rid + ".jsonl")).open("x") as journal:
            R.write_record(journal, {"kind": "start", "registration_pin": pin,
                                     "host": socket.getfqdn(), "tmux": os.environ["TMUX"]})
            updates, gate = fit(model, records, protocol, reg, scale, candidate["seed"], journal)
        checkpoint = out / (rid + ".pt")
        with checkpoint.open("xb") as stream:
            torch.save({"model": model.state_dict(), "saved_refit_pin": pin,
                        "source_checkpoint_sha256": source["checkpoint_sha256"],
                        "prereg_sha256": inputs.train_pin,
                        "implementation_versions": inputs.prereg["implementation_versions"],
                        "variant": candidate["variant"], "seed": candidate["seed"]}, stream)
        if R.digests(R.load_network(inputs, source, checkpoint)) != R.digests(model):
            raise ValueError("saved CPU reload differs")
        fits.append({**source, "parent_refit_checkpoint": source["checkpoint"],
                     "parent_refit_checkpoint_sha256": source["checkpoint_sha256"],
                     "checkpoint": str(checkpoint), "checkpoint_sha256": R.file_sha(checkpoint),
                     "state_dict_sha256": R.digests(model), "additional_refit_updates": updates,
                     "saved_refit_pin": pin})
        gates.append({"run_id": rid, "updates": updates, **gate})
        print(rid, "train gate", gate["pass"], "updates", updates, flush=True)
    with (out / "fitted_candidates.json").open("x") as stream:
        R.write_record(stream, {"refit_pin": pin, "candidates": fits,
                                "train_episodes": manifest["train_episodes"], "new_rollouts": 0})
    with (out / "train_gate.json").open("x") as stream:
        R.write_record(stream, {"candidates": gates, "pass": all(g["pass"] for g in gates)})
    if not all(g["pass"] for g in gates):
        with (out / "HELD_TRAIN_GATE.json").open("x") as stream:
            R.write_record(stream, {"reason": "train gate failed; no development scoring",
                                    "new_rollouts": 0, "development_scoring_rounds": 0})
        return
    # The existing scorer verifies the frozen actors against saved actions and
    # uses the unchanged development criteria. All six train gates precede it.
    R.evaluate(out, upstream, pin, reg, inputs, fits, R.WORK / "nav065_deployment_v1",
               manifest["train_episodes"], manifest["train_steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    torch.set_num_threads(1)
    try:
        if args.preflight:
            protocol, pin = R.load(PROPOSAL), None
        elif args.registration:
            protocol, pin = approved_registration(args.registration)
        else:
            raise ValueError("requires --preflight or a separately approved --registration")
        upstream, upstream_pin, reg, inputs, maps, _, _ = R.preflight()
        if upstream_pin != protocol["source_refit_pin"]:
            raise ValueError("upstream refit pin differs")
        root = R.WORK / protocol["source_run"]
        R.verify_hash(root / "fitted_candidates.json", protocol["source_manifest_sha256"])
        manifest = R.load(root / "fitted_candidates.json")
        if manifest["train_journal_sha256"] != protocol["source_train_journal_sha256"]:
            raise ValueError("train journal source differs")
        for c in manifest["candidates"]:
            R.verify_hash(Path(c["checkpoint"]), c["checkpoint_sha256"])
        if (len(manifest["candidates"]) != 6 or
                {c["run_id"] for c in manifest["candidates"]} != {c["run_id"] for c in inputs.candidates}):
            raise ValueError("candidate source coverage differs")
        out = R.WORK / OUTPUT_NAME
        if out.exists():
            raise ValueError("output already exists; inspect attempt before resuming")
        if args.preflight:
            saved_train(inputs, maps, reg["grid_k"], root, manifest)
            print("PREFLIGHT PASS: 540 saved train episodes, six checkpoints, 33 v2 files verified; "
                  "zero new rollouts/updates; proposal remains unapproved", flush=True)
            return
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    run(protocol, pin, upstream, out, inputs, maps, reg, root, manifest)


if __name__ == "__main__":
    main()
