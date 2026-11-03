"""Pooled TASK-024-style final-test summary from per-seed CSVs.

In-repo replacement of the temporary ``/pub/data/caohy/tools/t024_agg.py``
(TASK-034). ``freeze_controller_bundle.py`` calls it to copy a re-derived
``results/task024_aggregate.txt`` into the frozen bundle, so its arithmetic must
stay identical to the text already frozen in
``camra_go2_velocity_it2999_v2/results/task024_aggregate.txt``.

Scope, stated explicitly to avoid two judges drifting apart:

* this tool reproduces the **TASK-024** judgement: pool every seed of a group,
  take the mean over finite episode values, compare with the scenario threshold
  (rmse relaxed by 20 % for ``perturbed``), plus the group failure-rate limit;
* it is *not* the protocol v1.0.0 confirmatory judge. That one needs per-variant
  quotas, per-metric validity gating and an ``insufficient_evidence`` state, and
  lives in ``scripts/aggregate_eval_runs.py`` / ``eval_metrics.aggregate``.

Usage::

    python scripts/aggregate_final_test.py --root <TASK-024 root> [--groups both] \
        [--protocol docs/eval_protocol.json] [--json-out file]

Exit codes: 0 when every requested group is judged, 2 when the root or the
protocol is missing, 3 when a scenario has no rows at all.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camra_config import cfg  # noqa: E402


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_flag(value) -> bool:
    """Read a boolean CSV cell without the two classic mistakes.

    ``float("True")`` raises, and ``bool("0")`` is True — the second one turned
    every clean run into a 100 % failure rate once (TASK-032). Text forms are
    spelled out here, everything else falls back to a numeric comparison, and an
    unparsable cell counts as *not* failed rather than as a failure.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "failed"}:
            return True
        if text in {"0", "false", "no", "", "nan", "none"}:
            return False
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return False


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def load_rows(root: Path, group: str) -> tuple[dict[str, list[dict]], int]:
    rows_by_scenario: dict[str, list[dict]] = {}
    seed_dirs = sorted(glob.glob(str(root / group / "s*")))
    for seed_dir in seed_dirs:
        path = Path(seed_dir) / "per_episode.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows_by_scenario.setdefault(row["scenario"], []).append(row)
    return rows_by_scenario, len(seed_dirs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="final-test output root (default: $CAMRA_LOG_ROOT/TASK-024)")
    ap.add_argument("--protocol", default=None, help="default: docs/eval_protocol.json")
    ap.add_argument("--groups", default="both", choices=["nominal", "perturbed", "both"])
    ap.add_argument("--json-out", default=None, help="optional machine-readable copy")
    args = ap.parse_args()

    root = Path(args.root) if args.root else cfg.log_root("TASK-024")
    protocol_path = Path(args.protocol) if args.protocol else cfg.protocol()
    if not root.is_dir():
        print(f"[agg] FATAL: final-test root not found: {root}", file=sys.stderr)
        return 2
    if not protocol_path.is_file():
        print(f"[agg] FATAL: protocol not found: {protocol_path}", file=sys.stderr)
        return 2

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    specs = {s["id"]: s for s in protocol["scenarios"]}
    order = [s["id"] for s in protocol["scenarios"]]
    groups = ["nominal", "perturbed"] if args.groups == "both" else [args.groups]

    grand: dict[str, dict[str, bool]] = {}
    payload: dict = {"root": str(root), "protocol": str(protocol_path), "groups": {}}
    missing_scenarios = 0

    for group in groups:
        rows_by_scenario, seed_dirs = load_rows(root, group)
        relax = 1.2 if group == "perturbed" else 1.0
        fr_limit = 0.10 if group == "perturbed" else 0.05
        print("=" * 100)
        print(f"GROUP {group}  ({seed_dirs} seed dirs, 300 episodes/scenario expected)")
        print(f"{'scenario':<13} {'n':>4} {'fail':>6}  checks (value <= limit)")

        scenario_pass: dict[str, bool] = {}
        scenes: dict[str, dict] = {}
        for sid in order:
            rows = rows_by_scenario.get(sid, [])
            if not rows:
                print(f"{sid:<13}  missing")
                scenario_pass[sid] = False
                missing_scenarios += 1
                scenes[sid] = {"episodes": 0, "pass": False, "missing": True}
                continue
            spec = specs[sid]
            n = len(rows)
            failure_rate = sum(1 for r in rows if as_flag(r.get("failed"))) / n
            ok = failure_rate <= fr_limit
            checks = {"failure_rate": {"value": failure_rate, "limit": fr_limit, "pass": ok}}
            parts = [f"fail={failure_rate:.3f}<={fr_limit:.2f} {'ok' if ok else 'FAIL'}"]
            for metric in spec["metrics"]:
                if metric == "failure_rate":
                    continue
                limit = spec["thresholds"].get(metric)
                if limit is None:
                    continue
                limit = float(limit) * (relax if metric.startswith("rmse") else 1.0)
                value = mean_finite([num(r.get(metric)) for r in rows])
                good = (value == value) and value <= limit
                ok = ok and good
                checks[metric] = {"value": value, "limit": limit, "pass": good}
                parts.append(f"{metric}={value:.4f}<={limit:.2f} {'ok' if good else 'FAIL'}")
            scenario_pass[sid] = ok
            scenes[sid] = {"episodes": n, "pass": ok, "checks": checks}
            print(f"{sid:<13} {n:>4} {failure_rate:>6.3f}  {' | '.join(parts)}")

        passed = sum(1 for v in scenario_pass.values() if v)
        print(f"group pass: {passed}/{len(order)} scenarios")
        grand[group] = scenario_pass
        payload["groups"][group] = {"seed_dirs": seed_dirs, "scenarios": scenes,
                                    "passed": passed, "total": len(order)}

    verdict = None
    if len(groups) == 2:
        nominal_ok = all(grand["nominal"].values())
        perturbed_fraction = sum(1 for v in grand["perturbed"].values() if v) / len(order)
        print("=" * 100)
        print("acceptance.overall_pass = every scenario passes in nominal AND perturbed pass rate >= 0.80")
        print("nominal all pass:", nominal_ok)
        print(f"perturbed pass fraction: {perturbed_fraction:.2f} (need >= 0.80)")
        verdict = "PASS" if (nominal_ok and perturbed_fraction >= 0.80) else "FAIL"
        print("VERDICT:", verdict)
        payload["verdict"] = {
            "nominal_all_pass": nominal_ok,
            "perturbed_pass_fraction": perturbed_fraction,
            "pass": verdict == "PASS",
        }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[agg] -> {out}")

    if missing_scenarios:
        print(f"[agg] INCOMPLETE: {missing_scenarios} scenario(s) had no rows", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
