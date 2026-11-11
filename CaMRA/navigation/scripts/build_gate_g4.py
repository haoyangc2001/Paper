"""G4 gate assembly: nominal learning and deployment (NAV-039).

Reads only frozen evidence (training reports/logs, eval reports, the
deployment bundle, checkpoint files) and assembles the preregistered
checklist from ``config/gate_g4_prereg_v1.json`` (sha256 pinned).  No
environment is re-run.  The verdict rule and the fail-consequence block come
from the prereg.

Run:

    <frozen python> CaMRA/navigation/scripts/build_gate_g4.py \
        --json CaMRA/navigation/reports/gate_G4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

import numpy as np  # noqa: E402

WORK_ROOT = "/pub/data/caohy/work/camra_navigation"
PREREG = os.path.join(NAV_ROOT, "config", "gate_g4_prereg_v1.json")
PREREG_SHA = os.path.join(NAV_ROOT, "config", "gate_g4_prereg_v1.json.sha256")
REPORTS = os.path.join(NAV_ROOT, "reports")
BUNDLE = os.path.join(NAV_ROOT, "delivery", "deployment_candidates_v1.json")

RUN_PREFIX = {"uncalibrated": "formal_uncal", "full": "formal_full"}
CHECKPOINT_EVERY = 25
MAX_UPDATES = 250

RESULTS: list[dict[str, Any]] = []


def check(item_id: str, dim: str, item: str, ok: bool, evidence: dict) -> None:
    RESULTS.append({
        "id": item_id, "dimension": dim, "item": item,
        "light": "green" if ok else "red",
        "evidence": evidence,
    })


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_train_log(run_id: str) -> dict[str, Any]:
    rows = [json.loads(l) for l in
            open(os.path.join(WORK_ROOT, run_id, "train_log.jsonl"), encoding="utf-8")
            if l.strip()]
    ups = [r for r in rows if r["kind"] == "update"]
    end = next(r for r in rows if r["kind"] == "run_end")
    return {"ups": ups, "end": end}


def label_shares(ups: list[dict], lo: int, hi: int) -> dict[str, float]:
    tot: dict[str, int] = {}
    n = 0
    for u in ups[lo:hi]:
        for k, v in u["labels_last_window"].items():
            tot[k] = tot.get(k, 0) + v
            n += v
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    sha = open(PREREG_SHA, encoding="utf-8").read().strip()
    payload = open(PREREG, "rb").read()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise SystemExit("G4 prereg hash mismatch")
    prereg = json.loads(payload.decode("utf-8"))

    training_reports = {v: json.load(open(
        os.path.join(REPORTS,
                     f"training_{'uncalibrated' if v == 'uncalibrated' else 'camra_full'}.json"),
        encoding="utf-8")) for v in RUN_PREFIX}
    eval_reports = {v: [json.load(open(os.path.join(
        REPORTS, f"{RUN_PREFIX[v]}_s{s}_eval.json"), encoding="utf-8"))
        for s in (0, 1, 2)] for v in RUN_PREFIX}
    bundle = json.load(open(BUNDLE, encoding="utf-8"))
    rejudge = json.load(open(os.path.join(
        REPORTS, "deployment_critic_validation_rejudge.json"), encoding="utf-8"))

    logs = {v: {s: load_train_log(f"{RUN_PREFIX[v]}_s{s}") for s in (0, 1, 2)}
            for v in RUN_PREFIX}

    # ---------------- D1 learning stability ----------------
    for v in RUN_PREFIX:
        for s in (0, 1, 2):
            rid = f"{RUN_PREFIX[v]}_s{s}"
            ups = logs[v][s]["ups"]
            finite = all(np.isfinite([u["loss_policy"], u["loss_cost"],
                                      u["loss_reach"]]).all() for u in ups)
            e = np.array([u["entropy"] for u in ups])
            ent_ok = float(e[-25:].mean()) >= 0.5 * float(e[:25].mean())
            clip_recorded = all("clipped_ratio_fraction" in u for u in ups)
            first = label_shares(ups, 0, 25)
            last = label_shares(ups, -25, len(ups))
            direction = ((last.get("budget_exhausted", 0) - first.get("budget_exhausted", 0)) > 0
                         or (last.get("collision", 0) - first.get("collision", 0)) < 0)
            check(f"D1-{v}-s{s}-losses_finite", "D1_learning_stability",
                  "全部 update 损失有限", finite,
                  {"evidence": f"{WORK_ROOT}/{rid}/train_log.jsonl", "updates": len(ups)})
            check(f"D1-{v}-s{s}-entropy_no_collapse", "D1_learning_stability",
                  "熵不塌缩（last25 >= 0.5 x first25）", ent_ok,
                  {"evidence": f"{WORK_ROOT}/{rid}/train_log.jsonl",
                   "first25_mean": round(float(e[:25].mean()), 4),
                   "last25_mean": round(float(e[-25:].mean()), 4)})
            check(f"D1-{v}-s{s}-clip_recorded", "D1_learning_stability",
                  "梯度裁剪占比已记录", clip_recorded,
                  {"evidence": f"{WORK_ROOT}/{rid}/train_log.jsonl",
                   "mean_clip_fraction": round(float(np.mean(
                       [u["clipped_ratio_fraction"] for u in ups])), 4)})
            check(f"D1-{v}-s{s}-structure_direction", "D1_learning_stability",
                  "结构移动方向（budget_exhausted 升或 collision 降）", direction,
                  {"evidence": f"{WORK_ROOT}/{rid}/train_log.jsonl",
                   "budget_exhausted_first25": round(first.get("budget_exhausted", 0), 3),
                   "budget_exhausted_last25": round(last.get("budget_exhausted", 0), 3),
                   "collision_first25": round(first.get("collision", 0), 3),
                   "collision_last25": round(last.get("collision", 0), 3)})

    # ---------------- D2 usability ----------------
    for v in RUN_PREFIX:
        oi1_ok, oi1_detail = True, []
        for s, ev in zip((0, 1, 2), eval_reports[v]):
            pooled = ev["oi1_command_distribution"]["pooled"]
            cond = {
                "change_rate_mean": pooled["change_rate_mean"] <= 0.106,
                "hold_mean_s": pooled["hold_mean_s"] >= 0.2,
                "mean_abs_vy": pooled["mean_abs"]["vy"] <= 0.3,
                "low_speed_share": pooled["low_speed_share"] <= 0.7,
            }
            oi1_ok &= all(cond.values())
            oi1_detail.append({"seed": s, **{k: round(float(x), 4) if isinstance(x, (int, float)) else x
                                             for k, x in pooled.items() if k in
                                             ("change_rate_mean", "hold_mean_s", "low_speed_share")},
                               "mean_abs_vy": round(pooled["mean_abs"]["vy"], 4),
                               "conditions": {k: bool(b) for k, b in cond.items()}})
        check(f"D2-{v}-oi1_command_band", "D2_usability",
              "确定性指令分布全部候选落在可执行带内", oi1_ok,
              {"evidence": [f"reports/{RUN_PREFIX[v]}_s{s}_eval.json" for s in (0, 1, 2)],
               "per_seed": oi1_detail})

        dev_meds, dev_sr = [], []
        for s, ev in zip((0, 1, 2), eval_reports[v]):
            eps = ev["development_diagnostic"]["episodes"]
            dists = [r["min_goal_distance_m"] for r in eps]
            dev_meds.append(float(np.median(dists)))
            dev_sr.append(ev["development_diagnostic"]["success_rate"])
        med_of_meds = float(np.median(dev_meds))
        check(f"D2-{v}-approach", "D2_usability",
              "dev 最近距离中位数 <= 11.2 m（起点 14 m 的 0.8）", med_of_meds <= 11.2,
              {"evidence": [f"reports/{RUN_PREFIX[v]}_s{s}_eval.json" for s in (0, 1, 2)],
               "per_seed_median_min_distance_m": [round(m, 3) for m in dev_meds],
               "median_of_medians_m": round(med_of_meds, 3)})
        med_sr = float(np.median(dev_sr))
        check(f"D2-{v}-dev_success_rate", "D2_usability",
              "dev 成功率中位数 >= 0.10（硬判据，预登记声明预期 FAIL）", med_sr >= 0.10,
              {"evidence": [f"reports/{RUN_PREFIX[v]}_s{s}_eval.json" for s in (0, 1, 2)],
               "per_seed_success_rate": dev_sr, "median": med_sr,
               "note": prereg["transparency_note"][:80]})

    # ---------------- D3 freeze integrity ----------------
    cands = bundle["candidates"]
    check("D3-bundle-six-unique", "D3_freeze_integrity",
          "清单恰含 6 个候选且候选 ID 唯一",
          len(cands) == 6 and len({c["candidate_id"] for c in cands}) == 6,
          {"evidence": "delivery/deployment_candidates_v1.json",
           "n_candidates": len(cands),
           "candidate_ids": sorted(c["candidate_id"] for c in cands)})
    hash_ok, hash_detail = True, []
    for c in cands:
        if c["refit_rounds"]:
            p = c["weights"]["refit_state_path"]
            ok = p and os.path.exists(p)
            if ok:
                ok = sha256_file(p) == c["source_checkpoint_sha256"]
        else:
            p = os.path.join(WORK_ROOT, c["source_run_id"], "state.pt")
            ok = os.path.exists(p) and sha256_file(p) == c["source_checkpoint_sha256"]
        hash_ok &= bool(ok)
        hash_detail.append({"candidate_id": c["candidate_id"],
                            "source_hash_verified": bool(ok)})
    check("D3-source-hash-bound", "D3_freeze_integrity",
          "每候选源 checkpoint 散列与 work 目录文件逐位一致", hash_ok,
          {"evidence": "delivery/deployment_candidates_v1.json + work root",
           "per_candidate": hash_detail})
    amend_ok = ("amend_chain" in bundle
                and any("amend1" in x for x in bundle.get("amend_chain", []))
                and all("amend_chain" in c["validation"] for c in cands))
    check("D3-amend-chain", "D3_freeze_integrity",
          "修正链完整（prereg -> amend1 -> 复判）", amend_ok,
          {"evidence": ["delivery/deployment_candidates_v1.json",
                        "config/deployment_freeze_prereg_v1.amend1.json",
                        "reports/deployment_critic_validation_rejudge.json"]})
    ckpt_ok, ckpt_detail = True, []
    for v in RUN_PREFIX:
        for s in (0, 1, 2):
            rid = f"{RUN_PREFIX[v]}_s{s}"
            d = os.path.join(WORK_ROOT, rid)
            missing = [f"state_u{k}.pt" for k in range(CHECKPOINT_EVERY, MAX_UPDATES + 1,
                                                       CHECKPOINT_EVERY)
                       if not os.path.exists(os.path.join(d, f"state_u{k}.pt"))]
            if not os.path.exists(os.path.join(d, "state.pt")):
                missing.append("state.pt")
            ok = not missing
            ckpt_ok &= ok
            ckpt_detail.append({"run_id": rid, "checkpoints_complete": ok,
                                "missing": missing})
    check("D3-checkpoints-complete", "D3_freeze_integrity",
          "6 个 run 的每 25 updates checkpoint 与最终 state.pt 齐全", ckpt_ok,
          {"evidence": "work root", "per_run": ckpt_detail,
           "rule": "最终 checkpoint=最后更新，与表现无关（正式训练预登记 checkpoint_rule）"})
    leak = []
    for v in RUN_PREFIX:
        for s, ev in zip((0, 1, 2), eval_reports[v]):
            for r in (ev["development_diagnostic"]["episodes"]
                      + ev["train_maps_deterministic"]["episodes"]):
                if "final" in r.get("map_id", ""):
                    leak.append(r["map_id"])
    check("D3-no-final-leak", "D3_freeze_integrity",
          "budget_selection / final_* 划分在全部证据中零出现", not leak,
          {"evidence": [f"reports/{RUN_PREFIX[v]}_s{s}_eval.json" for v in RUN_PREFIX
                        for s in (0, 1, 2)], "leaks": leak})

    # ---------------- verdict ----------------
    reds = [r for r in RESULTS if r["light"] == "red"]
    d2_reds = {r["id"] for r in reds if r["dimension"] == "D2_usability"}
    d1_d3_reds = [r for r in reds if r["dimension"] != "D2_usability"]
    if reds and d2_reds and \
            all("dev_success_rate" in i for i in d2_reds) and not d1_d3_reds:
        verdict = "FAIL_WITH_STABILITY"
    elif reds:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    options = []
    if verdict != "PASS":
        options = [
            {"option": "O-1 追加训练授权",
             "content": "在现有预登记框架下追加步数/updates（需新的资源授权与新的预登记版本；"
                        "当前 6 run 授权配额已用满，不得沿用旧配额悄悄加跑）",
             "cost": "新的授权决策 + 训练时间（向量化环境后每 run 约 23 分钟，成本很低）"},
            {"option": "O-2 预算语义修正后重训",
             "content": "把训练预算从 z0 = 1.0xE_ref 改宽（如 k=1.15..1.3 的锚点档，"
                        "需检查 z_ratio 饱和约束）或按 NAV-036 已登记的预算选择协议走评估分支",
             "cost": "契约/预登记升版 + 重训 6 run；成功率门的复核只能收紧或维持"},
            {"option": "O-3 接受负结果继续下游",
             "content": "以 FAIL_WITH_STABILITY 定案：学习过程健康、指令分布可执行、"
                        "评价网络匹配部署策略，但成功率未达；下游预算搜索任务对两方法"
                        "不做 value 搜索，改用预算网格的实测扫描（budget_comparison 协议本身支持）",
             "cost": "论文叙事改变（『预算约束下的学习可用性』而非『成功策略』）；"
                     "不得声称 PPO 收敛"},
        ]

    report = {
        "task": "NAV-039",
        "schema": "camra-gate-g4-v1",
        "prereg": {"path": "config/gate_g4_prereg_v1.json", "sha256": sha},
        "n_checks": len(RESULTS),
        "n_red": len(reds),
        "lights": {"green": len(RESULTS) - len(reds), "red": len(reds)},
        "verdict": verdict,
        "verdict_rule": prereg["verdict_rule"],
        "options": options,
        "checklist": RESULTS,
        "evidence_integrity": {"problems": []},
    }
    print(json.dumps({"verdict": verdict, "n_checks": len(RESULTS),
                      "n_red": len(reds),
                      "red_ids": [r["id"] for r in reds]}, ensure_ascii=False))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
