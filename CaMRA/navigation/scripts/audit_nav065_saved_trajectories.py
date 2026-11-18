"""Check the NAV-065 trajectory reader on the already accepted NAV-064 traces."""

import argparse
import json
import math
import os
from pathlib import Path
import socket

import eval_training_round as E
from train_formal import verify_prereg

from camra_navigation.algorithm.deployment_oracle import reconstruct_truth
from camra_navigation.algorithm.validation import file_sha
from camra_navigation.costs import DT_NAV_S

NAV = Path(E.NAV)


def load(path):
    return json.loads(path.read_bytes())


def verify_hash(path, expected):
    if file_sha(path) != expected:
        raise ValueError(f"input hash mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "tiger11" or not os.environ.get("TMUX"):
        parser.error("requires tiger11 and tmux")
    if args.out.exists():
        parser.error("output already exists")
    prereg, pin = verify_prereg("training_loop_prereg_terminal_tail_v1.json")
    accepted_path = NAV / "reports/nav064_terminal_tail_acceptance_v1.json"
    source = load(NAV / "reports/nav065_deployment_source_binding_v1.json")
    verify_hash(accepted_path, source["source_evidence"]["sha256"])
    accepted = load(accepted_path)
    if accepted["verdict"] != "PASS" or accepted["prereg_sha256"] != pin:
        raise ValueError("source was not accepted under the bound protocol")
    release_path = NAV / "reports/nav064_terminal_tail_training_release_v1.json"
    verify_hash(release_path, accepted["evidence"][str(release_path)])
    resources = load(release_path)["v2_verified_files"]
    for path, digest in resources.items():
        verify_hash(Path(path), digest)
    dev_path = NAV / "reports/nav064_terminal_tail_v1_development.json"
    verify_hash(dev_path, accepted["evidence"][str(dev_path)])
    dev = load(dev_path)
    if dev["data_access"]["rollout_split"] != "development" or dev["prereg_sha256"] != pin:
        raise ValueError("unexpected development source")
    for path in (E.DESIGN, E.SPLIT):
        verify_hash(Path(path), dev["data_access"]["paths"][path])

    value = load(NAV / prereg["binding"]["value_contract"]["path"])
    maps = load(Path(E.SPLIT))["splits"]["development"]["maps"]
    design = E.M.load_design(E.DESIGN)
    initial_distance = {}
    for m in maps:
        spec = E.M.build(m["level"], m["seed"], design)
        initial_distance[m["map_id"]] = math.dist(spec.start, spec.goal)
    grid = prereg["round_acceptance"]["budget_response_on_dev"]["grid_k"]
    expected = {(k, m) for k in grid for m in initial_distance}
    accepted_runs = {r["run_id"]: r["state_sha256"] for r in accepted["runs"]}
    seen = set()
    episodes = steps = saturated = 0
    for run in dev["runs"]:
        rid = run["run_id"]
        actual = [(ep["k"], ep["map_id"]) for ep in run["episodes"]]
        if (rid in seen or run["state_sha256"] != accepted_runs.get(rid)
                or len(actual) != len(expected) or set(actual) != expected):
            raise ValueError(f"inconsistent candidate identity or episode grid: {rid}")
        seen.add(rid)
        for ep in run["episodes"]:
            try:
                truth = reconstruct_truth(ep,
                    initial_goal_distance_m=initial_distance[ep["map_id"]],
                    value_contract=value, horizon_s=prereg["environment"]["t_max_s"],
                    goal_radius_m=prereg["environment"]["r_goal_m"], step_s=DT_NAV_S)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{rid} {ep['k']} {ep['map_id']}: {exc}") from exc
            episodes += 1
            steps += len(truth.budget_J)
            saturated += int(truth.budget_feature_clipped.sum())
    if seen != set(accepted_runs):
        raise ValueError("missing accepted run")

    report = {
        "task": "NAV-065", "status": "TRACE_COMPATIBILITY_ONLY",
        "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
        "prereg_sha256": pin, "development_sha256": file_sha(dev_path),
        "frozen_v2_files_verified": len(resources),
        "runs": len(seen), "episodes": episodes, "pre_action_rows": steps,
        "observed_budget_features_clipped": saturated,
        "checkpoints_loaded": 0, "critic_predictions": 0, "new_rollouts": 0,
    }
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
