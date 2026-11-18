"""Close NAV-064 from saved evidence, without generating any rollouts."""
import argparse
from collections import Counter
import datetime as dt
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

import torch

import train_formal as train
from camra_navigation.algorithm.validation import file_sha

NAV = Path(__file__).resolve().parents[1]
WORK = Path("/pub/data/caohy/work/camra_navigation")
PREFIX = "nav064_terminal_tail_v1"
REGISTRATION = "training_loop_prereg_terminal_tail_v1.json"
LAUNCH = WORK / "nav064_terminal_tail_launch_20261117_v1"


def load(path):
    return json.loads(path.read_bytes())


def assert_hashes(paths):
    for path, digest in paths.items():
        assert file_sha(Path(path)) == digest, path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert socket.gethostname().split(".")[0] == "tiger11" and os.environ.get("TMUX")
    assert sys.executable == "/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python"
    plan = load(NAV.parents[1] / "doc/plan/20261109_camra_navigation_implementation.json")
    states = {t["id"]: t["status"] for t in plan["tasks"]}
    task = next(t for t in plan["tasks"] if t["id"] == "NAV-064")
    assert all(states[i] for i in task["depends_on"])
    if not task["status"]:
        assert next(t["id"] for t in plan["tasks"] if not t["status"]
                    and all(states[i] for i in t["depends_on"])) == task["id"]

    prereg, pin = train.verify_prereg(REGISTRATION)
    assert prereg["authorization"]["training_approved"] and prereg["round"] <= prereg["resource"]["max_rounds"]
    release_path = NAV / "reports/nav064_terminal_tail_training_release_v1.json"
    repair_path = NAV / "reports/nav064_terminal_tail_repair_v1.json"
    launch_path = LAUNCH / "launch.json"
    release, repair, launch = map(load, (release_path, repair_path, launch_path))
    assert launch["training_release"]["sha256"] == file_sha(release_path)
    assert pin == launch["prereg_sha256"] == release["prereg"]["sha256"]
    assert_hashes(release["v2_verified_files"])
    assert_hashes(repair["preserved_results"])
    for rel, item in repair["source_after"].items():
        assert file_sha(NAV / rel) == file_sha(Path(item["snapshot"])) == item["sha256"], rel
    gate_path = NAV / prereg["binding"]["numerical_gate"]["path"]
    gate = load(gate_path)
    assert gate["verdict"] == "PASS" and gate["implementation_versions"] == prereg["implementation_versions"]
    assert any(c["check"] == "episode_old_log_prob_fixed" and c["ok"] for c in gate["checks"])

    run_root = WORK / PREFIX
    finish_path = run_root / "FINISHED.json"
    assert not (run_root / "HELD.json").exists()
    start, finish = load(run_root / "start.json"), load(finish_path)
    assert start["git_head"] == launch["commit"]
    assert start["prereg_sha256"] == finish["prereg_sha256"] == pin and finish["verdict"] == "PASS"
    assert start["runs"] == prereg["run_registry"] and not finish["automatic_downstream_execution"]
    exploration_path = run_root / "exploration_u50.json"
    stability_path = run_root / "learning_stability_u150.json"
    exploration, stability = load(exploration_path), load(stability_path)
    assert exploration["verdict"] == stability["verdict"] == "PASS"
    assert exploration["updates"] == 50 and stability["updates"] == 150
    assert exploration["split"] == stability["split"] == "train"
    for check in (exploration, stability):
        assert sorted(r["seed"] for r in check["seeds"]) == [0, 1, 2]
        assert all(r["passed"] for r in check["seeds"])

    audit_paths = [WORK / f"{PREFIX}_audit_u{u}/report.json" for u in (50, 150, 300)]
    audits = [load(p) for p in audit_paths]
    registry = prereg["run_registry"]
    for n, audit in zip((50, 150, 300), audits):
        assert audit["verdict"] == "PASS" and audit["prereg_sha256"] == pin and audit["updates_audited"] == n
        assert audit["script_sha256"] == prereg["binding"]["store_auditor"]["sha256"]
        assert {r["run_id"] for r in audit["runs"]} == {r["run_id"] for r in registry}
        assert all(r["checks_pass"] and r["counts"]["completed_episodes"] == n * 18 for r in audit["runs"])
        assert_hashes({r["canonical_index"]: r["canonical_index_sha256"] for r in audit["runs"]})

    split_path = NAV / prereg["binding"]["split"]["path"]
    split = load(split_path)["splits"]
    train_maps = {m["map_id"] for m in split["train"]["maps"]}
    dev_maps = {m["map_id"] for m in split["development"]["maps"]}
    assert len(train_maps) == 18 and len(dev_maps) == 6 and not (train_maps & dev_maps)
    dev_path = NAV / f"reports/{PREFIX}_development.json"
    dev = load(dev_path)
    assert dev["prereg_sha256"] == pin and dev["performance_pass"]
    assert dev["data_access"]["rollout_split"] == "development"
    assert set(dev["data_access"]["maps"]) == dev_maps and set(dev["data_access"]["training_map_ids"]) == train_maps
    assert_hashes(dev["data_access"]["paths"])
    assert dev["data_access"]["paths"][str(split_path)] == file_sha(split_path)
    dev_rows = {r["run_id"]: r for r in dev["runs"]}
    audited_runs = {r["run_id"]: r for r in audits[-1]["runs"]}
    assert len(dev["runs"]) == len(dev_rows) == len(registry) == 6
    assert set(dev_rows) == set(audited_runs)

    rows = []
    for candidate in registry:
        rid = candidate["run_id"]
        root = WORK / rid
        log_path, state_path = root / "train_log.jsonl", root / "state.pt"
        records = [json.loads(line) for line in log_path.read_bytes().splitlines()]
        begins, updates, episodes, ends = (
            [r for r in records if r["kind"] == kind]
            for kind in ("run_start", "update", "episode_diag", "run_end")
        )
        assert len(begins) == len(ends) == 1 and ends[0]["updates"] == len(updates) == 300
        assert [r["update"] for r in updates] == list(range(300))
        begin = begins[0]
        assert all(begin[k] == v for k, v in candidate.items())
        assert begin["prereg_sha256"] == pin and begin["implementation_versions"] == prereg["implementation_versions"]
        assert begin["budget_rule"] == prereg["budget"]["sampling"]
        assert begin["collection"] == "complete_episodes_v1" and not begin["smoke"]
        assert begin["runtime"]["host"].split(".")[0] == "tiger11" and begin["runtime"]["tmux"]
        assert begin["runtime"]["python"] == sys.executable
        access = begin["data_access"]
        assert access["split_used"] == "train" and set(access["map_ids"]) == train_maps
        assert access["inputs"] == dev["data_access"]["paths"]
        counts = audited_runs[rid]["counts"]
        assert counts["completed_episodes"] == counts["chunks"] == len(episodes) == 5400
        assert len({(r["map_id"], r["episode_id"]) for r in episodes}) == 5400
        assert set(r["map_id"] for r in episodes) == train_maps
        samples = sum(r["n_samples"] for r in updates)
        assert samples == sum(r["n_steps"] for r in episodes) == counts["samples"]
        assert counts["negative_budget_samples"] > 0 and counts["saturated_budget_samples"] == 0
        assert counts["clipped_actions"] == counts["manifest_metadata_mismatches"] == 0
        numeric = ("loss_policy", "loss_reach", "loss_cost", "grad_norm_policy",
                   "grad_norm_reach", "grad_norm_cost", "entropy", "clipped_ratio_fraction")
        for row in updates:
            assert row["epochs"] == 10 and row["minibatch_size"] == 512 and row["n_segments"] == 18
            assert 0 < row["n_samples"] <= 8100 and row["grad_steps"] > 10
            assert row["grad_norm_policy"] > 0 and all(math.isfinite(row[k]) for k in numeric)
            assert row["gamma_joint"] == 1.0
        assert updates[0]["gamma_h"] == 0.999 and updates[-1]["gamma_h"] == 0.99999

        evaluated = dev_rows[rid]
        state_sha = file_sha(state_path)
        assert state_sha == Path(str(state_path) + ".sha256").read_text().strip() == evaluated["state_sha256"]
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        periodic_path = root / "state_u300.pt"
        periodic_sha = file_sha(periodic_path)
        assert periodic_sha == Path(str(periodic_path) + ".sha256").read_text().strip()
        periodic = torch.load(periodic_path, map_location="cpu", weights_only=True)
        for key in ("prereg_sha256", "update_idx", "seed", "variant", "implementation_versions"):
            assert periodic[key] == state[key]
        assert periodic["model"].keys() == state["model"].keys()
        assert all(torch.equal(t, periodic["model"][key]) for key, t in state["model"].items())
        assert state["prereg_sha256"] == pin and state["update_idx"] == 299
        assert state["seed"] == candidate["seed"] and state["variant"] == candidate["variant"]
        assert state["implementation_versions"] == prereg["implementation_versions"]
        assert all(torch.isfinite(t).all() for t in state["model"].values())
        grid = prereg["round_acceptance"]["budget_response_on_dev"]["grid_k"]
        assert Counter((r["k"], r["map_id"]) for r in evaluated["episodes"]) == {
            (k, mid): 1 for k in grid for mid in dev_maps
        }
        for ep in evaluated["episodes"]:
            trace = ep["trace"]
            clean = all(t["safety_y"] == 0 for t in trace)
            assert len(trace) == ep["n_steps"] and ep["danger_steps"] == sum(t["danger_violation"] for t in trace)
            assert math.isclose(trace[-1]["z_J"], ep["z_J_end"], abs_tol=1e-8)
            assert math.isclose(ep["cost_model_J"], ep["z0_J"] - ep["z_J_end"], abs_tol=1e-7)
            assert ep["feasible"] == (ep["label"] == "goal_reached" and clean and ep["z_J_end"] >= 0)
            assert math.isclose(ep["ratio_cost_to_budget"], ep["cost_model_J"] / ep["z0_J"])
        primary = [ep for ep in evaluated["episodes"] if ep["k"] == 1.15]
        assert evaluated["sr_feasible"] == sum(e["feasible"] for e in primary) / 6
        assert evaluated["energy_ratio"] == statistics.median(e["ratio_cost_to_budget"] for e in primary)
        assert evaluated["danger_steps"] == sum(e["danger_steps"] for e in primary)
        budget_response = {
            str(k): {"episodes": 6, "feasible": sum(e["feasible"] for e in evaluated["episodes"] if e["k"] == k)}
            for k in grid
        }
        rows.append(dict(candidate, source_checkpoint=str(state_path), state_sha256=state_sha,
                         periodic_checkpoint_sha256=periodic_sha,
                         train_log=str(log_path), train_log_sha256=file_sha(log_path),
                         updates=len(updates), samples=samples, train_episodes=len(episodes),
                         evaluation={k: evaluated[k] for k in ("sr_feasible", "energy_ratio", "danger_steps")},
                         budget_response=budget_response))

    threshold = prereg["round_acceptance"]
    summary = {}
    for variant in prereg["method"]["variants"]:
        group = [r for r in rows if r["variant"] == variant]
        metrics = {k: statistics.median(r["evaluation"][k] for r in group)
                   for k in ("sr_feasible", "energy_ratio", "danger_steps")}
        metrics["performance_pass"] = (
            metrics["sr_feasible"] >= threshold["sr_feasible_median"]
            and metrics["energy_ratio"] <= threshold["energy_ratio_median"]
            and (variant != "full" or metrics["danger_steps"] <= threshold["danger_steps_median_full"])
        )
        summary[variant] = metrics
    assert summary == dev["summary"] == finish["performance"]
    assert all(r["performance_pass"] for r in summary.values())
    samples = sum(r["samples"] for r in rows)
    assert samples <= prereg["resource"]["requested_samples"]
    assert sum(r["updates"] for r in rows) == finish["updates"] == prereg["resource"]["requested_updates"]
    assert sum(len(r["episodes"]) for r in dev["runs"]) == prereg["resource"]["development_episodes_requested"]
    evidence = [release_path, repair_path, launch_path, finish_path, exploration_path, stability_path,
                gate_path, dev_path, NAV / f"reports/{PREFIX}_development.references.json",
                NAV / f"reports/{PREFIX}_closure.json", *audit_paths]
    report = dict(task="NAV-064", verdict="PASS", timestamp=dt.datetime.now().astimezone().isoformat(),
                  host=socket.getfqdn(), python=sys.executable, tmux=os.environ["TMUX"], cwd=str(Path.cwd()),
                  preregistration=str(NAV / "config" / REGISTRATION), prereg_sha256=pin,
                  source_commit=start["git_head"], implementation_versions=prereg["implementation_versions"],
                  fixed_resources=len(release["v2_verified_files"]), source_files=len(repair["source_after"]),
                  summary=summary, runs=rows, train_samples=samples, development_episodes=108,
                  evidence={str(p): file_sha(p) for p in evidence},
                  data_access={"train": sorted(train_maps), "development": sorted(dev_maps),
                               "actual_input_hashes": dev["data_access"]["paths"]},
                  limits=["Nominal evaluation only. The tight budget is diagnostic.",
                          "No NAV-065 deploy critic acceptance, budget search or physical release."])
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"PASS: {len(rows)} candidates, {samples} train samples, 108 development episodes")


if __name__ == "__main__":
    main()
