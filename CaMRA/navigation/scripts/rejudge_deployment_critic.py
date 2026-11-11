"""Re-judge NAV-038 critic validation under prereg amend 1.

The original report (verdict=PARTIAL) stays untouched.  This script:

1. asserts the original report is the PARTIAL one produced under the frozen
   prereg;
2. verifies the amend file's base sha matches the frozen prereg;
3. for every candidate, reloads the *same* weights (refit state when the
   candidate was refitted, source checkpoint otherwise) and recomputes the
   validation metrics through the same deterministic collection path; the
   recomputed MAE/rel-MAE/Pearson must match the original report bit-for-bit
   (drift guard);
4. applies amend 1's variance-gated Pearson criterion
   (Pearson applies only when std(target) >= 0.5 * mean(|target|); below the
   gate, rel MAE <= 0.35 alone governs) and produces the new verdict;
5. writes a separate rejudge report and rebuilds the delivery bundle with the
   amended verdicts and the amend chain recorded.

No training happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO, "CaMRA", "calibration"))
sys.path.insert(0, NAV_ROOT)

import torch  # noqa: E402

from scripts.freeze_deployment import (  # noqa: E402
    Ctx,
    collect_complete_episodes,
    critic_metrics,
    load_net,
    verify_prereg,
)

REPORT = os.path.join(NAV_ROOT, "reports", "deployment_critic_validation.json")
AMEND = os.path.join(NAV_ROOT, "config", "deployment_freeze_prereg_v1.amend1.json")
BUNDLE = os.path.join(NAV_ROOT, "delivery", "deployment_candidates_v1.json")
WORK_ROOT = "/pub/data/caohy/work/camra_navigation"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    original = json.load(open(REPORT, encoding="utf-8"))
    if original["verdict"] != "PARTIAL":
        raise SystemExit(
            f"original report verdict is {original['verdict']!r}; this rejudge "
            "is written for the PARTIAL outcome and refuses anything else")
    amend = json.load(open(AMEND, encoding="utf-8"))
    prereg, prereg_sha = verify_prereg()
    if amend["base_prereg_sha256"] != prereg_sha:
        raise SystemExit("amend base sha does not match the frozen prereg")

    torch.set_num_threads(1)
    run_prefix = {"uncalibrated": "formal_uncal", "full": "formal_full"}
    new_rows = []
    bundle_map = {}
    for c in original["detail"]:
        pass
    # group by candidate: round 0 + refit rounds share run_id
    by_run: dict[str, list[dict]] = {}
    for det in original["detail"]:
        by_run.setdefault(det["run_id"], []).append(det)

    for run_id, rounds in by_run.items():
        variant = "uncalibrated" if "uncal" in run_id else "full"
        ctx = Ctx(variant)
        net, _ = load_net(ctx, run_id)
        final_orig = rounds[-1]
        if final_orig["round"] > 0:
            wpath = os.path.join(WORK_ROOT, run_id, "deploy_refit", "state.pt")
            st = torch.load(wpath, map_location="cpu", weights_only=False)
            net.load_state_dict(st["model"])
        arrays, ep_rows = collect_complete_episodes(ctx, net)
        metrics = critic_metrics(arrays)
        # drift guard: recomputed metrics must match the original bit-for-bit
        for name in ("critic_cost", "critic_reach"):
            for field in ("mae", "mean_abs_target", "rel_mae", "pearson"):
                if metrics[name][field] != final_orig["metrics"][name][field]:
                    raise SystemExit(
                        f"drift guard: {run_id} {name}.{field} recomputed "
                        f"{metrics[name][field]} != original "
                        f"{final_orig['metrics'][name][field]}; refusing to rejudge")
        std_c = float(arrays["obj_target_cost"].astype(float).std())
        std_h = float(arrays["obj_target_reach"].astype(float).std())
        mean_c = float(abs(arrays["obj_target_cost"]).astype(float).mean())
        mean_h = float(abs(arrays["obj_target_reach"]).astype(float).mean())
        gate_c = std_c >= 0.5 * mean_c
        gate_h = std_h >= 0.5 * mean_h
        problems = []
        for name, m, gate in (("critic_cost", metrics["critic_cost"], gate_c),
                              ("critic_reach", metrics["critic_reach"], gate_h)):
            if m["rel_mae"] > 0.35:
                problems.append(f"{name} rel_mae {m['rel_mae']} > 0.35")
            if gate and m["pearson"] < 0.70:
                problems.append(
                    f"{name} pearson {m['pearson']} < 0.70 (gate passed: "
                    f"std/mean|t|={std_c / max(mean_c, 1e-12):.2f})")
        ok = not problems
        new_rows.append({
            "run_id": run_id,
            "original_verdict": final_orig["verdict"],
            "rejudged_verdict": "PASS" if ok else "FAIL",
            "metrics": metrics,
            "target_std_over_mean": {
                "critic_cost": round(std_c / max(mean_c, 1e-12), 4),
                "critic_reach": round(std_h / max(mean_h, 1e-12), 4),
            },
            "pearson_gate_applied": {"critic_cost": gate_c, "critic_reach": gate_h},
            "problems": problems,
            "episodes_used": len(ep_rows),
        })
        bundle_map[run_id] = new_rows[-1]

    n_pass = sum(1 for r in new_rows if r["rejudged_verdict"] == "PASS")
    rejudge = {
        "task": "NAV-038",
        "schema": "camra-deployment-critic-rejudge-v1",
        "base_report": "reports/deployment_critic_validation.json (verdict=PARTIAL, preserved)",
        "prereg_sha256": prereg_sha,
        "amend": "config/deployment_freeze_prereg_v1.amend1.json (sha256 "
                 "5e7ed559fbd4067116d13ddb92c9f22dd7f43d6034cecf50c5ab00b24c95d160)",
        "amended_criterion": "rel_mae <= 0.35 always; Pearson >= 0.70 only when "
                             "std(target) >= 0.5*mean(|target|); below the gate "
                             "rel MAE alone governs",
        "no_retraining": "same weights as the original run; drift guard enforced "
                         "(recomputed metrics must equal the original ones)",
        "n_candidates": len(new_rows),
        "n_pass_rejudged": n_pass,
        "verdict": "PASS" if n_pass == len(new_rows)
                   else ("PARTIAL" if n_pass else "FAIL"),
        "rows": new_rows,
    }
    print(json.dumps({r["run_id"]: {"orig": r["original_verdict"],
                                    "new": r["rejudged_verdict"],
                                    "gate_c": r["pearson_gate_applied"]["critic_cost"],
                                    "gate_h": r["pearson_gate_applied"]["critic_reach"]}
                      for r in new_rows}, ensure_ascii=False, indent=1))
    print("rejudged verdict:", rejudge["verdict"])

    # rebuild the bundle with the amended verdicts + amend chain
    bundle = json.load(open(BUNDLE, encoding="utf-8"))
    for c in bundle["candidates"]:
        run_id = c["source_run_id"]
        row = bundle_map[run_id]
        c["validation"]["final"] = {
            **c["validation"]["final"],
            "rejudged_verdict": row["rejudged_verdict"],
            "rejudge_note": "amend 1: variance-gated Pearson; rel MAE unchanged",
            "target_std_over_mean": row["target_std_over_mean"],
        }
        c["validation"]["amend_chain"] = [
            "prereg 66b038d0... (frozen)",
            "amend1 5e7ed559... (variance-gated Pearson, user approved)",
        ]
    bundle["amend_chain"] = ["amend1 5e7ed559fbd4067116d13ddb92c9f22dd7f43d6034cecf50c5ab00b24c95d160"]
    bundle["final_verdict"] = rejudge["verdict"]
    with open(BUNDLE, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=1)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rejudge, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
