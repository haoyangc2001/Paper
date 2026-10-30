"""Assign predeclared IDs/conditions only: this script creates no error values."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = list(csv.DictReader((ROOT / "data/condition_catalog.csv").open(encoding="utf-8")))
    destination = ROOT / "data/split_manifest.csv"
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    for split, prefix, count in [("calibration", "CAL", 800), ("held_out", "TEST", 400)]:
        for index in range(count):
            condition = catalog[index % len(catalog)]
            pair_id = f"{prefix}-{index+1:04d}"
            rows.append({"pair_id": pair_id, "split": split, "condition_id": condition["condition_id"], "scenario_id": condition["scene_id"], "perturbation_profile_id": condition["perturbation_profile_id"], "duration_s": condition["duration_s"], "command_id": "assumed_cmd_" + pair_id})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len({r["pair_id"] for r in rows}) == 1200
    for split, expected in [("calibration", 40), ("held_out", 20)]:
        for condition in catalog:
            assert sum(r["split"] == split and r["condition_id"] == condition["condition_id"] for r in rows) == expected
    print("1200 unique IDs, correct 800/400 split and 40/20 per condition; no outcomes created.")


if __name__ == "__main__":
    main()
