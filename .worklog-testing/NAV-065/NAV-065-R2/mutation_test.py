import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


GRID = (0.70, 0.85, 1.0, 1.15, 1.30)
RUN_IDS = tuple(f"candidate-{i}" for i in range(6))
MAPS = tuple({"map_id": f"map-{i}"} for i in range(6))
EXPECTED_GUARD = "duplicate or unexpected saved development trajectory"
REPORT_DIR = Path(__file__).resolve().parent


def load_module(path):
    spec = importlib.util.spec_from_file_location("tested_refit_copy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(full_grid):
    source_grid = GRID if full_grid else GRID[2:]
    runs = {
        run_id: {
            "episodes": [
                {"map_id": map_row["map_id"], "k": k}
                for map_row in MAPS
                for k in source_grid
            ]
        }
        for run_id in RUN_IDS
    }
    inputs = SimpleNamespace(
        candidates=[{"run_id": run_id} for run_id in RUN_IDS],
        maps=list(MAPS),
        runs=runs,
    )
    records = [{"kind": "start"}]
    for run_id in RUN_IDS:
        for map_row in MAPS:
            for k in GRID[:2]:
                row = {"run_id": run_id, "map_id": map_row["map_id"], "k": k}
                records.append({"kind": "rollout_start", **row})
                records.append({
                    "kind": "rollout_result",
                    **row,
                    "episode": {"map_id": row["map_id"], "k": k},
                })
    for run_id in RUN_IDS:
        for map_row in MAPS:
            for k in GRID:
                records.append({
                    "kind": "scored",
                    "run_id": run_id,
                    "map_id": map_row["map_id"],
                    "k": k,
                    "actual_feasible": False,
                })
    return inputs, records


def run_fixed_dev(module, inputs, records):
    with tempfile.TemporaryDirectory(dir=REPORT_DIR, prefix="fixture-") as temp:
        previous = Path(temp)
        (previous / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )
        return module.fixed_dev(inputs, GRID, previous)


def assert_valid_baseline(module):
    inputs, records = fixture(full_grid=False)
    raw, scores = run_fixed_dev(module, inputs, records)
    assert len(scores) == 180
    assert all(len(episodes) == 30 for episodes in raw.values())


def malformed(module, kind):
    inputs, records = fixture(full_grid=True)
    changed = False
    for row in records:
        if row["kind"] != "rollout_result":
            continue
        if kind == "duplicate":
            row["map_id"], row["k"] = "map-0", 1.0
        else:
            row["map_id"] = "unknown-map"
        changed = True
        break
    assert changed
    try:
        run_fixed_dev(module, inputs, records)
    except ValueError as exc:
        assert str(exc) == EXPECTED_GUARD, (
            f"{kind} guard bypassed; later error was {str(exc)!r}"
        )
    else:
        raise AssertionError(f"{kind} key passed the duplicate/unknown guard")


def run_green(module, label):
    assert_valid_baseline(module)
    malformed(module, "duplicate")
    malformed(module, "unknown")
    print(f"{label} PASS: valid fixture accepted; duplicate and unknown keys rejected by targeted guard")


def run_mutant(module):
    assert_valid_baseline(module)
    killed = []
    for kind in ("duplicate", "unknown"):
        try:
            malformed(module, kind)
        except AssertionError as exc:
            killed.append(kind)
            print(f"MUTANT KILLED ({kind}): targeted assertion failed: {exc}")
        else:
            raise AssertionError(f"mutant survived {kind} guard assertion")
    assert killed == ["duplicate", "unknown"]
    print("MUTANT FAILED TARGET ASSERTIONS: weakened guard no longer raises the expected validation error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("green", "mutant"))
    parser.add_argument("module_path", type=Path)
    parser.add_argument("--label", default="ORIGINAL")
    args = parser.parse_args()
    module = load_module(args.module_path)
    if args.mode == "green":
        run_green(module, args.label)
    else:
        run_mutant(module)


if __name__ == "__main__":
    main()
