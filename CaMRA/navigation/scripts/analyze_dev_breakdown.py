#!/usr/bin/env python3
"""S0: break the dev-set result down so a single binary success rate stops hiding
three different failures.

For every run in the given round reports it prints, per level (L1/L2/L3):

  arrivals      label == goal_reached  (the historic SR definition, unchanged)
  feasible      arrived, no danger step, and within energy budget
  dist med/min  min_goal_distance_m median / minimum  (how close it actually got)
  timeout       share of episodes that used the full 300 steps
  t used        median seconds used

Read-only: consumes the report JSONs that eval_training_round.py already wrote.
"""
import argparse
import json
import statistics as st
import sys

LEVELS = ("l1", "l2", "l3")


def level_of(map_id):
    return str(map_id).split("_")[0].split("-")[0].lower()


def summarise(episodes):
    n = len(episodes)
    if n == 0:
        return None
    arr = sum(1 for e in episodes if e["label"] == "goal_reached")
    feas = sum(1 for e in episodes if e["label"] == "goal_reached"
               and int(e.get("danger_steps", 0)) == 0
               and bool(e.get("within_budget", False)))
    dists = [float(e["min_goal_distance_m"]) for e in episodes]
    to = sum(1 for e in episodes if e["label"] == "timeout")
    ts = [float(e.get("t_s", 0.0)) for e in episodes]
    return {
        "n": n, "arr": arr, "feas": feas,
        "dist_med": st.median(dists), "dist_min": min(dists),
        "timeout": to, "t_med": st.median(ts),
    }


def fmt(s):
    return ("n=%-2d 到达%d 可行%d 距离中位%5.2fm 最小%5.2fm 超时%d(%3.0f%%) 用时中位%4.1fs"
            % (s["n"], s["arr"], s["feas"], s["dist_med"], s["dist_min"],
               s["timeout"], 100.0 * s["timeout"] / s["n"], s["t_med"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="评估报告 JSON 路径")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    payload = []
    for path in args.reports:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        print("=" * 100)
        print("%s  round=%s  prereg=%s" % (path, d.get("round"), (d.get("prereg") or {}).get("sha256", "")[:12]))
        for r in d.get("runs", []):
            eps = (r.get("dev") or {}).get("episodes") or []
            if not eps:
                continue
            all_s = summarise(eps)
            print("  %-34s 全体  %s" % (r["run_id"], fmt(all_s)))
            per_level = {}
            for lv in LEVELS:
                sub = [e for e in eps if level_of(e["map_id"]) == lv]
                if not sub:
                    continue
                s = summarise(sub)
                per_level[lv] = s
                print("  %-34s %-3s   %s" % ("", lv.upper(), fmt(s)))
            payload.append({"report": path, "round": d.get("round"), "run_id": r["run_id"],
                            "variant": r["variant"], "seed": r["seed"],
                            "all": all_s, "per_level": per_level})
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
