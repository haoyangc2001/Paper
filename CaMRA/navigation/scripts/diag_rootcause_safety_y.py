"""NAV-064: is the full variant's reach critic really inverted, or is the
irreversible safety history pinning its target to a constant?

Part A  frozen r11 rollout chunks, stratified by safety_y.
        Decisive test: among y=0 steps, does V_h still rise with goal distance?
        If yes, the critic is fine and the earlier "inverted" reading was a
        composition artefact (y=1 steps concentrate near the goal).
Part B  geometry control: same policy, same maps, full vs uncalibrated geometry,
        count danger-triggered steps.

Read-only.  Writes reports/training_round_r11_rootcause.json (new file).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import diag_xray_signal as X  # noqa: E402
except ImportError:  # pragma: no cover
    sys.path.insert(0, "/tmp")
    import nav064_xray as X  # noqa: E402

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
WORK = X.WORK
RUNS = X.RUNS
FIELDS = ["safety_y", "safety_y_next", "obj_h", "obj_g_norm", "goal_distance_m",
          "value_reach", "obj_target_reach", "obj_stop_now", "obj_stop_residual",
          "obj_feasible", "terminated", "goal_distance_m_next", "t",
          "obj_adv_total", "action_raw"]
EDGES = [0, 3, 6, 9, 12, 100]


def corr(a, b):
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def vh_by_distance(dist, vh):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (dist >= lo) & (dist < hi)
        out.append({"dist_m": f"{lo}-{hi}",
                    "n": int(m.sum()),
                    "vh_mean": round(float(vh[m].mean()), 4) if m.sum() > 20 else None})
    return out


def part_a():
    out = {}
    for tag, rid in RUNS:
        acc = {f: [] for f in FIELDS}
        for env_i, idx, last, p in X.chunk_paths(rid):
            with np.load(p) as z:
                for f in FIELDS:
                    acc[f].append(np.asarray(z[f]))
        d = {f: np.concatenate(v) for f, v in acc.items()}
        y = d["safety_y"]
        y_next = d["safety_y_next"]
        h = d["obj_h"]
        g = d["obj_g_norm"]
        dist = d["goal_distance_m"]
        vh = d["value_reach"]
        stop = d["obj_stop_now"].astype(bool)
        feas = d["obj_feasible"].astype(bool)
        arr = d["terminated"].astype(bool) & (d["goal_distance_m_next"] <= 0.5)

        # C1: which y convention reconstructs h exactly?
        pred_after = np.maximum(g, y_next)
        pred_before = np.maximum(g, y)
        err_after = float(np.abs(h - pred_after).max())
        err_before = float(np.abs(h - pred_before).max())

        y1 = y_next >= 0.5
        y0 = ~y1
        # y just flipped on this step (first violation in the window)
        flip = y1 & (y < 0.5)

        out[tag] = {
            "n_steps": int(len(y)),
            "C1_h_equals_max_g_Cy": {
                "max_abs_err_with_y_after": round(err_after, 8),
                "max_abs_err_with_y_before": round(err_before, 8),
                "convention": "y_after" if err_after < err_before else "y_before",
            },
            "C2_safety_saturation": {
                "frac_y1": round(float(y1.mean()), 4),
                "frac_h_ge_0.99": round(float(np.mean(h >= 0.99)), 4),
                "frac_y1_and_h_ge_0.99": round(float(np.mean(y1 & (h >= 0.99))), 4),
                "n_first_violations_in_window": int(flip.sum()),
                "mean_t_at_first_violation": (round(float(d["t"][flip].mean()), 1)
                                              if flip.any() else None),
            },
            "C3_decisive_stratified": {
                "y0": {"n": int(y0.sum()),
                       "corr_vh_dist": round(corr(dist[y0], vh[y0]), 4),
                       "vh_mean": round(float(vh[y0].mean()), 4),
                       "h_mean": round(float(h[y0].mean()), 4),
                       "vh_by_distance": vh_by_distance(dist[y0], vh[y0])},
                "y1": {"n": int(y1.sum()),
                       "corr_vh_dist": round(corr(dist[y1], vh[y1]), 4),
                       "vh_mean": round(float(vh[y1].mean()), 4),
                       "h_mean": round(float(h[y1].mean()), 4),
                       "vh_by_distance": vh_by_distance(dist[y1], vh[y1])},
            },
            "C4_target_pinned": {
                "y1_target_reach_mean": (round(float(d["obj_target_reach"][y1].mean()), 4)
                                         if y1.any() else None),
                "y0_target_reach_mean": (round(float(d["obj_target_reach"][y0].mean()), 4)
                                         if y0.any() else None),
                "y1_stop_residual_mean": (round(float(d["obj_stop_residual"][y1].mean()), 4)
                                          if y1.any() else None),
                "y0_stop_residual_mean": (round(float(d["obj_stop_residual"][y0].mean()), 4)
                                          if y0.any() else None),
            },
            "C5_stop_and_feasible": {
                "stop_now_frac_y0": round(float(stop[y0].mean()), 4) if y0.any() else None,
                "stop_now_frac_y1": round(float(stop[y1].mean()), 4) if y1.any() else None,
                "feasible_frac_y0": round(float(feas[y0].mean()), 4) if y0.any() else None,
                "feasible_frac_y1": round(float(feas[y1].mean()), 4) if y1.any() else None,
                "adv_mean_y0": round(float(d["obj_adv_total"][y0].mean()), 5) if y0.any() else None,
                "adv_mean_y1": round(float(d["obj_adv_total"][y1].mean()), 5) if y1.any() else None,
                "vx_mean_y0": (round(float(d["action_raw"][:, 0][y0].mean()), 4)
                               if y0.any() else None),
                "vx_mean_y1": (round(float(d["action_raw"][:, 0][y1].mean()), 4)
                               if y1.any() else None),
            },
            "C6_arrival_vs_feasibility": {
                "n_arrivals": int(arr.sum()),
                "arrivals_with_y1": int((arr & y1).sum()),
                "arrivals_feasible": int((arr & feas).sum()),
                "arrivals_y1_but_env_goal_reached": int((arr & y1 & ~feas).sum()),
            },
        }
    return out


def part_b():
    """Same policy, same maps, only geometry differs (full has inflated radius)."""
    sys.path.insert(0, os.path.join(NAV, "scripts"))
    sys.path.insert(0, os.path.join(NAV, "src"))
    import eval_training_round as E  # noqa: E402
    from camra_navigation.algorithm import networks as NW  # noqa: E402
    from camra_navigation.budget import accounting as ACC  # noqa: E402

    nav = json.load(open(E.NAV_CONTRACT, encoding="utf-8"))
    split = json.load(open(E.SPLIT, encoding="utf-8"))
    refs = json.load(open(E.E_REF, encoding="utf-8"))
    e_ref = {r["map_id"]: float(r["e_ref_J"]) for r in refs["maps"]}
    dev = split["splits"]["development"]["maps"]
    ck = os.path.join(WORK, "nav064_r11_full_s0", "state.pt")
    state = torch.load(ck, map_location="cpu", weights_only=False)
    net = NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
                   log_std_min=-5.0, log_std_init=-3.0, steering_gain=0.2)
    net.load_state_dict(state["model"])
    net.eval()

    out = {}
    for variant in ("uncalibrated", "full"):
        ctx = E.build_rollout_ctx(variant)
        n_danger, n_steps, n_y1, episodes = 0, 0, 0, []
        for m in dev:
            spec = E.M.build(m["level"], m["seed"], ctx["design"])
            z0 = e_ref[m["map_id"]]
            if variant == "full":
                z0 = ACC.consistent_e_ref(z0, "full")
            env = E.VectorNominalNavEnv(spec, ctx["geom"], ctx["cm"], n_envs=1, z0_J=z0)
            env.reset()
            ep_danger, ep_steps, first_y = 0, 0, None
            torch.manual_seed(1234)
            with torch.no_grad():
                for step in range(300):
                    if env._done[0]:
                        break
                    ob = torch.as_tensor(env.observation, dtype=torch.float32)
                    a, _, _ = net.act(ob, deterministic=True)
                    rec = env.step(a.numpy())
                    ep_danger += int(bool(rec.danger_violation[0]))
                    ep_steps += 1
                    if first_y is None and float(env._safety_y[0]) >= 0.5:
                        first_y = step
            post_y = (ep_steps - first_y) if first_y is not None else 0
            episodes.append({"map": m["map_id"], "danger_steps": ep_danger,
                             "steps": ep_steps, "first_y_step": first_y,
                             "post_y_steps": post_y,
                             "post_y_frac": round(post_y / max(1, ep_steps), 3),
                             "label": env._label[0] or "cap"})
            n_danger += ep_danger
            n_steps += ep_steps
            n_y1 += ep_steps if first_y is None else (ep_steps - (first_y or 0))
        out[variant] = {
            "geom_delta_m": float(ctx["geom"].delta_m),
            "n_steps": n_steps, "danger_steps": n_danger,
            "danger_per_100_steps": round(100.0 * n_danger / max(1, n_steps), 2),
            "episodes_with_y_triggered": sum(1 for e in episodes if e["first_y_step"] is not None),
            "post_y_steps_total": sum(e["post_y_steps"] for e in episodes),
            "post_y_frac_of_all_steps": round(sum(e["post_y_steps"] for e in episodes)
                                              / max(1, n_steps), 3),
            "episodes": episodes,
        }
    return out


def main():
    a = part_a()
    b = part_b()
    rep = {
        "schema": "nav064-r11-rootcause-v1",
        "note": "Read-only. Part A stratifies frozen rollout chunks by the "
                "irreversible safety history y. Part B holds policy and maps "
                "fixed and changes only geometry.",
        "part_a": a, "part_b": b,
    }
    out = os.path.join(NAV, "reports", "training_round_r11_rootcause.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)

    for tag, r in a.items():
        print(f"\n=== {tag} n={r['n_steps']} ===")
        print(" C1", r["C1_h_equals_max_g_Cy"])
        print(" C2", r["C2_safety_saturation"])
        c3 = r["C3_decisive_stratified"]
        for k in ("y0", "y1"):
            print(f" C3 {k}: n={c3[k]['n']} corr(Vh,dist)={c3[k]['corr_vh_dist']} "
                  f"Vh={c3[k]['vh_mean']} h={c3[k]['h_mean']}")
            print("     ", " ".join(f"{x['dist_m']}m:{x['vh_mean']}" for x in c3[k]["vh_by_distance"]))
        print(" C4", r["C4_target_pinned"])
        print(" C5", r["C5_stop_and_feasible"])
        print(" C6", r["C6_arrival_vs_feasibility"])
    print("\n=== Part B geometry control ===")
    for v, r in b.items():
        print(f" {v}: delta_m={r['geom_delta_m']:.4f} steps={r['n_steps']} "
              f"danger={r['danger_steps']} per100={r['danger_per_100_steps']} "
              f"episodes_with_y={r['episodes_with_y_triggered']}/6")
        for e in r["episodes"]:
            print(f"    {e['map']:28s} danger={e['danger_steps']:3d} "
                  f"first_y={str(e['first_y_step']):>4s} label={e['label']}")
    print("\nOUT " + out)


if __name__ == "__main__":
    main()
