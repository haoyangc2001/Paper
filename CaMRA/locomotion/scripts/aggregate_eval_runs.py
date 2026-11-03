"""Aggregate confirmatory evaluation runs across seeds/groups (TASK-032).

The per-run ``summary.json`` answers "did this seed pass". The protocol verdict
needs one more step: combine every seed, group, scenario and variant, and refuse
to answer PASS when something is missing. This tool does exactly that and nothing
else - it never re-simulates and never relaxes a threshold.

Directory layout it expects (the one TASK-024/036 use)::

    <runs_root>/<group>/s<seed>/per_episode.csv
    <runs_root>/<group>/s<seed>/summary.json      (optional, recorded only)

Usage::

    python scripts/aggregate_eval_runs.py --runs_root /pub/data/caohy/logs/camra/TASK-036 \
        --seeds 300-309 --groups nominal,perturbed --out /tmp/aggregate.json

Exit codes: 0 = complete and passing, 1 = complete but failing,
2 = incomplete (missing seeds, groups, scenarios, variants or too little evidence).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "unitree_rl_lab"))
from unitree_rl_lab.utils.eval_metrics import MIN_VALID_RATIO, aggregate, program_variants_count  # noqa: E402

REPO = "/home/caohy/repositories/Paper"
DEFAULT_PROTOCOL = os.path.join(REPO, "CaMRA/locomotion/docs/eval_protocol.json")


def parse_seeds(text: str) -> list[int]:
    """``"300-309"`` / ``"300,302,305"`` / a mix of both."""
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    if not out:
        raise SystemExit("no seeds given")
    return out


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 300-309")
    ap.add_argument("--groups", default="nominal,perturbed")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_valid_ratio", type=float, default=MIN_VALID_RATIO)
    ap.add_argument("--expected_per_variant", type=int, default=None,
                    help="if given, every (group, scenario, variant) must reach this episode count")
    args = ap.parse_args()

    with open(args.protocol, encoding="utf-8") as fh:
        protocol = json.load(fh)
    specs = {s["id"]: s for s in protocol["scenarios"]}
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = parse_seeds(args.seeds)

    report: dict = {
        "runs_root": os.path.abspath(args.runs_root),
        "seeds": seeds,
        "groups": groups,
        "min_valid_ratio": args.min_valid_ratio,
        "expected_per_variant": args.expected_per_variant,
        "missing_runs": [],
        "scenarios": [],
        "warnings": [],
    }

    rows_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    present = 0
    for group in groups:
        for seed in seeds:
            run_dir = os.path.join(args.runs_root, group, f"s{seed}")
            csv_path = os.path.join(run_dir, "per_episode.csv")
            if not os.path.isfile(csv_path):
                report["missing_runs"].append({"group": group, "seed": seed, "path": run_dir})
                continue
            present += 1
            rows = load_rows(csv_path)
            info_path = os.path.join(run_dir, "run_info.json")
            if os.path.isfile(info_path):
                with open(info_path, encoding="utf-8") as fh:
                    info = json.load(fh)
                if info.get("env_seed") not in (None, seed):
                    report["warnings"].append(
                        f"{run_dir}: run_info.env_seed={info.get('env_seed')} but directory says s{seed}"
                    )
                if info.get("group") not in (None, group):
                    report["warnings"].append(
                        f"{run_dir}: run_info.group={info.get('group')} but path says {group}"
                    )
            for r in rows:
                r.setdefault("group", group)
                rows_by_key[(group, r["scenario"])].append(r)
    report["runs_present"] = present
    report["runs_expected"] = len(groups) * len(seeds)

    for group in groups:
        for sc_id, spec in specs.items():
            rows = rows_by_key.get((group, sc_id), [])
            n_var = program_variants_count(spec)
            variants = []
            for vi in range(n_var):
                vrows = [r for r in rows if int(r.get("variant_id", 0)) == vi]
                if not vrows:
                    variants.append({"scenario": f"{sc_id}#v{vi}", "variant_id": vi, "episodes": 0,
                                     "pass": False, "status": "missing", "uncovered": True})
                    continue
                agg = aggregate(vrows, {**spec, "id": f"{sc_id}#v{vi}"}, group,
                                min_valid_ratio=args.min_valid_ratio)
                agg["variant_id"] = vi
                agg["seeds_present"] = sorted({int(r.get("env_seed", -1)) for r in vrows})
                if args.expected_per_variant is not None and agg["episodes"] < args.expected_per_variant:
                    agg["pass"] = False
                    agg["status"] = "short_of_quota"
                variants.append(agg)
            complete = all(v.get("pass") is not None and v.get("status") not in ("missing", "short_of_quota")
                           for v in variants) and bool(variants)
            report["scenarios"].append({
                "group": group,
                "scenario": sc_id,
                "episodes": len(rows),
                "variants": variants,
                "complete": complete,
                "pass": bool(complete and all(v.get("pass") for v in variants)),
            })

    report["complete"] = bool(not report["missing_runs"] and all(s["complete"] for s in report["scenarios"]))
    report["overall_pass"] = bool(report["complete"] and all(s["pass"] for s in report["scenarios"]))
    if report["missing_runs"]:
        report["warnings"].append(
            f"{len(report['missing_runs'])} of {report['runs_expected']} runs missing"
        )

    print(f"[AGG] runs {present}/{report['runs_expected']} complete={report['complete']} "
          f"overall_pass={report['overall_pass']}")
    for s in report["scenarios"]:
        bad = [v["variant_id"] for v in s["variants"] if not v.get("pass")]
        print(f"[AGG] {s['group']:9s} {s['scenario']:12s} episodes={s['episodes']:4d} "
              f"variants={len(s['variants'])} pass={s['pass']} bad={bad}")
    for w in report["warnings"]:
        print(f"[AGG][WARN] {w}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"[AGG] report -> {args.out}")

    if not report["complete"]:
        return 2
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
