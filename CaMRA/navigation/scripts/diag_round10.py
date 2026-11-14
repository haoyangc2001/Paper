"""Read-only: did round 8 break the no-arrival self-lock? Compare r7 vs r8."""
import json
import os
import statistics

ROOT = "/pub/data/caohy/work/camra_navigation"


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("kind") == "update":
                    yield d


for rd in ("r9", "r10"):
    print("=" * 12, rd, "=" * 12)
    tot_goal = tot_stag = tot_to = 0
    reach_g, cost_g = [], []
    for variant in ("uncalibrated", "full"):
        for seed in (0, 1, 2):
            p = os.path.join(ROOT, f"nav064_{rd}_{variant}_s{seed}", "train_log.jsonl")
            if not os.path.exists(p):
                print(f"nav064_{rd}_{variant}_s{seed}: MISSING")
                continue
            rs = list(rows(p))
            goals = sum(r.get("labels_last_window", {}).get("goal_reached", 0) for r in rs)
            stag = sum(r.get("labels_last_window", {}).get("stagnation", 0) for r in rs)
            to = sum(r.get("labels_last_window", {}).get("timeout", 0) for r in rs)
            tot_goal += goals
            tot_stag += stag
            tot_to += to
            rg = [r.get("grad_norm_reach", 0.0) for r in rs]
            cg = [r.get("grad_norm_cost", 0.0) for r in rs]
            reach_g.extend(rg)
            cost_g.extend(cg)
            ent = [r.get("entropy") for r in rs if r.get("entropy") is not None]
            print(f"nav064_{rd}_{variant}_s{seed}: goal={goals} stag={stag} timeout={to} "
                  f"reach_grad first={rg[0]:.4f} last50={statistics.mean(rg[-50:]):.4f} "
                  f"cost_grad last50={statistics.mean(cg[-50:]):.4f} "
                  f"entropy last50={statistics.mean(ent[-50:]):.3f}")
    print(f"--- {rd} totals: goal={tot_goal} stagnation={tot_stag} timeout={tot_to} ---")
    if reach_g and cost_g:
        print(f"--- {rd} pooled: reach_grad mean={statistics.mean(reach_g):.4f} "
              f"cost_grad mean={statistics.mean(cost_g):.4f} "
              f"ratio={statistics.mean(cost_g)/max(statistics.mean(reach_g),1e-9):.3f} ---")
