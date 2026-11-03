"""Pick the checkpoint of one training run with the protocol's dev rule.

In-repo replacement of the temporary ``/pub/data/caohy/tools/t023_select_r4.py``
(TASK-034). The rule is the one fixed in ``docs/eval_protocol.json`` under
``checkpoint_selection``:

    score = 0.4*norm_rmse_vx + 0.2*norm_rmse_vy + 0.2*norm_rmse_wz + 0.2*norm_failure
    each term divided by its hard threshold
    constraint: dev failure_rate <= 0.10; among those take the minimum score
    fallback: if none passes the constraint, take the minimum failure rate and
              record the selection as a failure of the acceptance criteria

Two normalisation modes exist because the protocol text says "its hard
threshold" while the thresholds are per scenario (``fwd_high`` and ``ramp`` use
0.18 m/s instead of 0.15):

``base`` (default)
    one threshold per metric — 0.15 / 0.10 / 0.15 / 0.05. This is what the
    delivered selection used, so it reproduces ``selection_r4.json`` bit for bit.
``per-scenario``
    divide each scenario's error by that scenario's own threshold, then average.
    Stricter for ``fwd_high`` and ``ramp``; reported next to ``base`` so the
    difference is visible instead of hidden in a default.

Usage::

    python scripts/repro/select_checkpoint.py --dev-root <root> [--suffix _r4]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camra_config import cfg  # noqa: E402

BASE_THRESHOLDS = {"rmse_vx": 0.15, "rmse_vy": 0.10, "rmse_wz": 0.15, "failure": 0.05}
WEIGHTS = {"rmse_vx": 0.4, "rmse_vy": 0.2, "rmse_wz": 0.2, "failure": 0.2}
FAILURE_CONSTRAINT = 0.10
METRICS = ("rmse_vx", "rmse_vy", "rmse_wz")


def _nan_mean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def load_candidate(path: Path, iteration: int, protocol: dict) -> dict | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    scenarios = summary.get("scenarios", [])
    if not scenarios:
        return None

    failures = [float(s["failure_rate"]) for s in scenarios]
    record = {
        "iteration": iteration,
        "scenarios": len(scenarios),
        "pass_count": sum(1 for s in scenarios if s.get("pass")),
        "failure_rate": _nan_mean(failures),
        "worst_failure": max(failures),
    }
    for metric in METRICS:
        record[metric] = _nan_mean([float(s[metric]) for s in scenarios if metric in s])

    # per-scenario normalisation: divide by that scenario's own threshold first
    terms = {metric: [] for metric in METRICS}
    for spec in protocol["scenarios"]:
        matched = next((s for s in scenarios if s["scenario"] == spec["id"]), None)
        if matched is None:
            continue
        for metric in METRICS:
            limit = spec["thresholds"].get(metric)
            if limit is None or metric not in matched:
                continue
            value = float(matched[metric])
            if value == value:
                terms[metric].append(value / float(limit))
    record["per_scenario_norm"] = {m: _nan_mean(v) for m, v in terms.items()}
    return record


def score_base(record: dict) -> float:
    total = 0.0
    for metric in METRICS:
        total += WEIGHTS[metric] * record[metric] / BASE_THRESHOLDS[metric]
    total += WEIGHTS["failure"] * record["failure_rate"] / BASE_THRESHOLDS["failure"]
    return total


def score_per_scenario(record: dict) -> float:
    total = 0.0
    for metric in METRICS:
        total += WEIGHTS[metric] * record["per_scenario_norm"][metric]
    total += WEIGHTS["failure"] * record["failure_rate"] / BASE_THRESHOLDS["failure"]
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev-root", default=None,
                    help="directory holding dev_it<iter><suffix>/summary.json "
                         "(default: $CAMRA_LOG_ROOT/repro/devsweep)")
    ap.add_argument("--suffix", default="",
                    help="suffix the sweep appended to its directories (delivered run: _r4)")
    ap.add_argument("--protocol", default=None, help="default: docs/eval_protocol.json")
    ap.add_argument("--out", default=None,
                    help="selection JSON path (default: <dev-root>/selection<suffix>.json)")
    args = ap.parse_args()

    protocol_path = Path(args.protocol) if args.protocol else cfg.protocol()
    if not protocol_path.is_file():
        print(f"[select] FATAL: protocol not found: {protocol_path}", file=sys.stderr)
        return 2
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    dev_root = Path(args.dev_root) if args.dev_root else (cfg.log_root() / "repro" / "devsweep")
    if not dev_root.is_dir():
        print(f"[select] FATAL: dev sweep root not found: {dev_root}", file=sys.stderr)
        print("[select] run scripts/repro/finalize.sh <RUN_DIR> [suffix] first", file=sys.stderr)
        return 2

    candidates: list[dict] = []
    skipped: list[str] = []
    # dev_it<iteration><suffix> and nothing else. The TASK-023 log root also holds the
    # 10-seed dev sweep as dev_it2999_s10x_r4; a plain glob accepts those too and the
    # iteration parse then dies on "2999_s100" (found while replaying the selection).
    name_re = re.compile(r"^dev_it(\d+)" + re.escape(args.suffix) + r"$")
    for path in sorted(glob.glob(os.path.join(str(dev_root), "dev_it*"))):
        if not os.path.isdir(path):
            continue
        base = os.path.basename(path)
        match = name_re.match(base)
        if not match:
            skipped.append(base)
            continue
        iters = int(match.group(1))
        record = load_candidate(Path(path) / "summary.json", iters, protocol)
        if record:
            record["dir"] = path
            candidates.append(record)
        else:
            skipped.append(base + " (no summary.json)")

    if skipped:
        print(f"[select] not considered ({len(skipped)}): {', '.join(skipped[:6])}"
              + (" ..." if len(skipped) > 6 else ""))

    if not candidates:
        print(f"[select] FATAL: no dev evaluations under {dev_root} matching {pattern}", file=sys.stderr)
        return 2

    for record in candidates:
        record["score"] = score_base(record)
        record["score_per_scenario"] = score_per_scenario(record)

    candidates.sort(key=lambda r: r["iteration"])
    print(f"{'iter':>6s} {'scen':>5s} {'pass':>5s} {'rmse_vx':>8s} {'rmse_vy':>8s} "
          f"{'rmse_wz':>8s} {'fail':>7s} {'worst':>7s} {'score':>7s} {'score_ps':>8s}")
    for r in candidates:
        print(f"{r['iteration']:>6d} {r['scenarios']:>5d} {r['pass_count']:>5d} "
              f"{r['rmse_vx']:>8.3f} {r['rmse_vy']:>8.3f} {r['rmse_wz']:>8.3f} "
              f"{r['failure_rate']:>7.3f} {r['worst_failure']:>7.3f} "
              f"{r['score']:>7.3f} {r['score_per_scenario']:>8.3f}")

    feasible = [r for r in candidates if r["failure_rate"] <= FAILURE_CONSTRAINT]
    if feasible:
        best = min(feasible, key=lambda r: r["score"])
        verdict = "selected"
    else:
        best = min(candidates, key=lambda r: (r["failure_rate"], r["score"]))
        verdict = "no_candidate_meets_failure_constraint"

    out_path = Path(args.out) if args.out else dev_root / f"selection{args.suffix}.json"
    payload = {
        "rule": protocol.get("checkpoint_selection", {}).get(
            "score", "0.4*rmse_vx/th + 0.2*rmse_vy/th + 0.2*rmse_wz/th + 0.2*failure/th"),
        "constraint": f"dev failure_rate <= {FAILURE_CONSTRAINT}",
        "normalization": "base (one threshold per metric: 0.15/0.10/0.15/0.05)",
        "base_thresholds": BASE_THRESHOLDS,
        "weights": WEIGHTS,
        "dev_root": str(dev_root),
        "suffix": args.suffix,
        "protocol": str(protocol_path),
        "candidates": candidates,
        "selected_iteration": best["iteration"],
        "selected_score": best["score"],
        "selected_score_per_scenario": best["score_per_scenario"],
        "selected_failure_rate": best["failure_rate"],
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSELECTED iteration={best['iteration']} score={best['score']:.3f} "
          f"failure_rate={best['failure_rate']:.3f} verdict={verdict}")
    print(f"[select] -> {out_path}")
    return 0 if verdict == "selected" else 4


if __name__ == "__main__":
    sys.exit(main())
