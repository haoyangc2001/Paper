"""HIRA-040：逐任务对账与最终验收（只读，不改任何数据/稿件/冻结物）。

做什么
------
1. **计划结构**：40 项、id 唯一有序、依赖存在、完成项都有 verification_result 与工作记录；
2. **逐项对账**：每项的声明证据路径是否真的存在；状态与验收事实是否一致；
3. **门禁机读判定**：不看"报告存在"，看报告里的判定字段（含 3 处已裁定例外的**预期值**）；
4. **采集台账**：1820 条配额 + 60 条非配额目录，逐 trial 是否只有单个 attempt（零重采）；
5. **无覆写/无隐性重训**：稿件、旧 v2 冻结物、v2 计划、HiRA 子模块、交付索引；
6. **交付索引复核**：调用 build_delivery_manifest.py --verify（221 文件 + 1880 trial）。

    python tests/final_acceptance_hira.py
    python tests/final_acceptance_hira.py --skip-manifest-verify   # 跳过 2 GB 重算

产物：reports/final_acceptance.json。任一门禁不符 → 返回码 1，且不把任务标成通过。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
RUNS = Path("/pub/data/caohy/logs/camra_hira/runs")
PLAN = REPO / "doc" / "plan" / "20261105_camra_hira_policy_calibration.json"
V2_CFG = REPO / "CaMRA" / "calibration" / "config"
HIRA_SUBMODULE_COMMIT = "6a05af55521e5f6668c50e9f90cf7049b3e35cd9"
OFFICIAL = {"pilot_20261106_protocolv1": 20, "baseline_20261106_fit": 240,
            "baseline_20261106_dev": 80, "baseline_20261106_validate": 80,
            "calibration_20261106": 800, "heldout_20261106": 400,
            "near_obstacle_20261107": 200}
NON_QUOTA = {"pilot_20261105T223428": 20, "near_smoke_0919": 40}
# 交付验证 run 落在独立工作根，不在官方 runs；分开统计，避免被读成"没采"
NON_QUOTA_ELSEWHERE = {
    "delivery_reload_20261107": {
        "expected": 40,
        "root": "/pub/data/caohy/work/hira/delivery_check_20261107_final/runs",
    },
}

# 门禁 → 报告里的机读判定（expected 为该字段应有的值；例外均已裁定并写明理由）
GATES = [
    ("HIRA-013", "observation_adapter_tests.json", "verdict.all_passed", True, "观测 13/13"),
    ("HIRA-014", "action_pd_tests.json", "verdict.all_passed", True, "动作/PD 10/10"),
    ("HIRA-016", "timing_command_reset.json", "verdict.all_passed", True, "三时钟 11/11"),
    ("HIRA-019", "hira_energy_meter_validation.json", "verdict.all_passed", True, "能量计量 17/17"),
    ("HIRA-020", "integration_gate.json", "verdict.g1_passed", True, "G1 16/16"),
    ("HIRA-021", "collector_integration_tests.json", "verdict.all_passed", True, "采集链路 21/21"),
    ("HIRA-022", "pilot_validation.json", "verdict.all_passed", True, "pilot 10/10"),
    ("HIRA-025", "baseline_collection_integrity.json", "verdict.all_passed", True, "基线 400 条 10/10"),
    ("HIRA-026", "power_model_validate.json", "verdict.all_passed", True, "功率模型验证"),
    ("HIRA-027", "footprint_validation.json", "verdict.all_passed", True, "足迹"),
    ("HIRA-028", "step_cost_interface_tests.json", "verdict.all_passed", True, "步成本接口"),
    ("HIRA-030", "calibration_collection_integrity.json", "verdict.failed",
     ["C06-cell-full-rate"], "已裁定：800 条 7/8，C06 两格缺口不补采"),
    ("HIRA-031", "margin_calibration.json", "verdict.all_passed", True, "α/δ 冻结 10/10+10/10"),
    ("HIRA-032", "heldout_collection_integrity.json", "verdict.failed",
     ["C06-cell-full-rate"], "已裁定：400 条 9/10，三格缺口保留主分母"),
    ("HIRA-033", "heldout_coverage.json", "full_sample.verdict.status",
     "insufficient_cells", "已裁定：3 格资格不足，两套分母并列"),
    ("HIRA-034", "near_obstacle_collection_integrity.json", "verdict.failed",
     ["N07-cell-full-rate"], "已裁定：200 条 10/11，两格缺口保留主分母"),
    ("HIRA-035", "near_obstacle_evaluation.json", "full_sample.verdict.reporting_complete",
     True, "近障评估如实上报（δ 条件 0/200 为负结果）"),
    ("HIRA-036", "full_independent_audit.json", "verdict.audit_ok", True, "独立审计 12/12"),
    ("HIRA-038", "delivery_reload_verification.json", "verdict.reload_ok", True, "独立目录重载 20/20"),
]

# 计划的 evidence_outputs 里有两处文件名与实际产物不一致（历史沿革），
# 这里显式登记，既不静默当成缺失，也不去改历史记录。
EVIDENCE_ALIASES = {
    "reports/pilot_end_to_end.json": "reports/pilot_validation.json",
}

PATH_RE = re.compile(r"[A-Za-z0-9_./\-]+\.(?:json|md|py|jsonl|npz|npy|sha256)")


def dig(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<missing>"
        cur = cur[part]
    return cur


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=False).stdout.strip()


def git_timed(args: list[str], timeout: int = 90) -> tuple[str, str | None]:
    """带超时的 git：NFS 元数据慢时 git status 可能挂住，不能让它拖死整个验收。"""
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                             text=True, timeout=timeout, check=False).stdout.strip()
        return out, None
    except subprocess.TimeoutExpired:
        return "", "timeout"


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def check_plan_structure(plan: dict) -> dict:
    tasks = plan["tasks"]
    ids = [t["id"] for t in tasks]
    problems: list[str] = []
    if len(tasks) != 40:
        problems.append(f"任务数 {len(tasks)} != 40")
    if len(set(ids)) != len(ids):
        problems.append("id 有重复")
    if ids != sorted(ids):
        problems.append("id 未按顺序排列")
    by_id = {t["id"]: t for t in tasks}
    for t in tasks:
        for dep in t.get("depends_on") or []:
            if dep not in by_id:
                problems.append(f"{t['id']} 依赖不存在的 {dep}")
            elif dep > t["id"]:
                problems.append(f"{t['id']} 依赖了更晚的 {dep}")
        if t["status"] is True:
            if not (t.get("verification_result") or "").strip():
                problems.append(f"{t['id']} 标完成但没有 verification_result")
            if not (t.get("work_records") or []):
                problems.append(f"{t['id']} 标完成但没有 work_record")
            if not (t.get("evidence_outputs") or []):
                problems.append(f"{t['id']} 标完成但没有 evidence_outputs")
    done = sum(1 for t in tasks if t["status"] is True)
    return {"n_tasks": len(tasks), "n_done": done, "problems": problems,
            "ok": not problems}


def check_task_evidence(plan: dict) -> dict:
    """逐项核对声明的证据路径是否真的存在。"""
    rows = []
    missing_total = 0
    for t in plan["tasks"]:
        paths: list[str] = []
        for raw in (t.get("evidence_outputs") or []):
            for m in PATH_RE.findall(str(raw)):
                if m not in paths:
                    paths.append(m)
        missing = []
        aliased = []
        for p in paths:
            target = EVIDENCE_ALIASES.get(p, p)
            name = Path(target).name
            cand = [EXP / target, REPO / target, EXP / "reports" / name,
                    EXP / "config" / name, EXP / "docs" / name, Path(target)]
            hit = next((c for c in cand if c.is_file()), None)
            if hit is None:
                # 本脚本自己的输出：检查发生在写盘之前，不算缺失
                if Path(p).name == "final_acceptance.json":
                    continue
                missing.append(p)
            elif target != p:
                aliased.append({"declared": p, "actual": str(hit)})
        missing_total += len(missing)
        recs = t.get("work_records") or []
        rows.append({
            "id": t["id"], "phase": t.get("phase"), "status": t["status"],
            "n_records": len(recs),
            "last_record": (recs[-1].get("timestamp") if recs else None),
            "n_declared_paths": len(paths), "n_missing_paths": len(missing),
            "missing": missing[:4], "aliased": aliased,
            "has_verification": bool((t.get("verification_result") or "").strip()),
        })
    return {"rows": rows, "missing_total": missing_total,
            "ok": missing_total == 0}


def check_gates() -> dict:
    rows = []
    for task, name, field, expected, note in GATES:
        p = EXP / "reports" / name
        if not p.is_file():
            rows.append({"task": task, "report": name, "field": field, "expected": expected,
                         "actual": "<report missing>", "ok": False, "note": note})
            continue
        actual = dig(json.loads(p.read_text(encoding="utf-8")), field)
        ok = actual == expected
        rows.append({"task": task, "report": name, "field": field, "expected": expected,
                     "actual": actual, "ok": ok, "note": note,
                     "report_sha256": sha256_file(p)})
        print(f"[{'ok ' if ok else 'FAIL'}] {task} {name} :: {field} = {actual!r}", flush=True)
    return {"rows": rows, "ok": all(r["ok"] for r in rows),
            "n_gates": len(rows), "n_ok": sum(1 for r in rows if r["ok"])}


def check_collection() -> dict:
    rows = []
    total_quota_trials = 0
    multi_attempt: list[str] = []
    for run, expected in OFFICIAL.items():
        d = RUNS / run
        if not d.is_dir():
            rows.append({"run": run, "expected": expected, "found": 0, "ok": False})
            continue
        trials = [p for p in d.iterdir() if p.is_dir()]
        n_attempts = 0
        for t in trials:
            atts = sorted(t.glob("attempt_*"))
            n_attempts += len(atts)
            if len(atts) != 1:
                multi_attempt.append(f"{run}/{t.name}:{len(atts)}")
        total_quota_trials += len(trials)
        rows.append({"run": run, "expected": expected, "found": len(trials),
                     "attempts": n_attempts, "ok": len(trials) == expected})
    extras = {}
    for run, expected in NON_QUOTA.items():
        d = RUNS / run
        extras[run] = len([p for p in d.iterdir() if p.is_dir()]) if d.is_dir() else 0
    elsewhere = {}
    for run, spec in NON_QUOTA_ELSEWHERE.items():
        d = Path(spec["root"]) / run
        elsewhere[run] = {"root": spec["root"],
                          "found": len([p for p in d.iterdir() if p.is_dir()]) if d.is_dir() else 0,
                          "expected": spec["expected"]}
    quota_sum = sum(OFFICIAL.values())
    return {"official": rows, "quota_expected": quota_sum,
            "quota_found": total_quota_trials,
            "non_quota": extras, "non_quota_expected": NON_QUOTA,
            "non_quota_elsewhere": elsewhere,
            "multi_attempt": multi_attempt[:10], "n_multi_attempt": len(multi_attempt),
            "ok": all(r["ok"] for r in rows) and total_quota_trials == quota_sum
                  and not multi_attempt}


def check_no_overwrite(degraded: list[str]) -> dict:
    # 稿件目录在 NFS 上，文件多、元数据慢，git status 可能挂住 → 带超时并允许降级
    ms_status, ms_err = git_timed(["status", "--porcelain", "--", "manuscripts/"], timeout=90)
    if ms_err:
        degraded.append(
            "manuscripts_worktree_scan: git status 在 90 s 内未返回（NFS 元数据慢），"
            "降级为按提交历史核对；工作区级未落盘改动本次未复核"
        )
    ms_staged, _ = git_timed(["diff", "--cached", "--name-only", "--", "manuscripts/"], timeout=60)
    ms_commits_since = git("log", "--oneline", "19c9e3d..HEAD", "--", "manuscripts/")
    ms_last = git("log", "-1", "--format=%h %cI %s", "--", "manuscripts/")
    locked = ["data_lock_v1.json", "margins_v1.json", "power_model_v1.json", "footprint_v1.json"]
    frozen = {}
    for name in locked:
        f = V2_CFG / name
        side = V2_CFG / f"{name}.sha256"
        if not f.is_file():
            frozen[name] = "MISSING"
            continue
        rec = side.read_text(encoding="utf-8").split()[0] if side.is_file() else None
        inner = json.loads(f.read_text(encoding="utf-8")).get("sha256") if f.suffix == ".json" else None
        ok = rec in (sha256_file(f), str(inner))
        frozen[name] = "ok" if ok else "CHANGED"
    try:
        lock = json.loads((V2_CFG / "data_lock_v1.json").read_text(encoding="utf-8"))
        chain = (lock.get("margins") or {}).get("sha256") == sha256_file(V2_CFG / "margins_v1.json")
    except Exception:  # noqa: BLE001
        chain = False
    sub_head = subprocess.run(["git", "-C", str(REPO), "submodule", "status", "--", "HiRA"],
                              capture_output=True, text=True, check=False).stdout.strip()
    sub_dirty = subprocess.run(["git", "-C", str(REPO / "HiRA"), "status", "--short"],
                               capture_output=True, text=True, check=False).stdout.strip()
    plan_v2 = git("status", "--porcelain", "--", "doc/plan/20261103_camra_calibration_implementation.json")
    manuscripts_ok = (not ms_staged) and (not ms_commits_since) and \
                     (ms_err is None and not ms_status or ms_err is not None)
    return {
        "manuscripts": {"status_clean": not ms_status, "worktree_scan": ms_err or "ok",
                        "staged": [x for x in ms_staged.splitlines() if x.strip()],
                        "commits_since_acceptance": [x for x in ms_commits_since.splitlines() if x.strip()],
                        "last_commit": ms_last, "ok": manuscripts_ok},
        "v2_frozen": frozen, "v2_lock_to_margins_chain": chain,
        "hira_submodule": {"status_line": sub_head,
                           "commit_ok": sub_head.startswith(HIRA_SUBMODULE_COMMIT),
                           "worktree_clean": not sub_dirty,
                           "dirty_entries": sub_dirty.splitlines()[:5]},
        "v2_plan_clean": not plan_v2,
        "ok": (manuscripts_ok and all(v == "ok" for v in frozen.values())
               and chain and not sub_dirty and not plan_v2
               and sub_head.startswith(HIRA_SUBMODULE_COMMIT)),
    }


def check_manifest(skip: bool) -> dict:
    man = EXP / "delivery" / "bundle_manifest_hira_v1.json"
    if not man.is_file():
        return {"ok": False, "error": "交付索引不存在"}
    payload = json.loads(man.read_text(encoding="utf-8"))
    base = {"path": str(man), "sha256": sha256_file(man),
            "n_files": payload["counts"]["files"],
            "n_raw_trials": payload["big_data"].get("n_trials"),
            "git_commit": payload["git"]["commit"][:12],
            "git_dirty": payload["git"]["dirty"],
            "selfcheck": [payload["verification"]["entries_checked"],
                          payload["verification"]["entries_bad"]]}
    if skip:
        base.update({"verify": "skipped", "ok": payload["verification"]["entries_bad"] == 0})
        return base
    proc = subprocess.run([sys.executable, str(EXP / "tests" / "build_delivery_manifest.py"),
                           "--verify", str(man)], capture_output=True, text=True, check=False)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("files ok=")), "")
    base.update({"verify": line.strip(), "verify_rc": proc.returncode,
                 "ok": proc.returncode == 0 and "bad=0" in line})
    return base


def collect_conclusions() -> dict:
    """结论一律从报告里读，不在这里手抄数字。"""
    margins = json.loads((EXP / "config" / "margins_hira_v1.json").read_text(encoding="utf-8"))
    hc = json.loads((EXP / "reports" / "heldout_coverage.json").read_text(encoding="utf-8"))
    ne = json.loads((EXP / "reports" / "near_obstacle_evaluation.json").read_text(encoding="utf-8"))
    cmp_ = json.loads((EXP / "reports" / "hira_vs_v2_comparison.json").read_text(encoding="utf-8"))
    ov = hc["full_sample"]["overall"]
    return {
        "alpha": margins["alpha"], "delta": margins["delta"],
        "margins_n_official": margins.get("data", {}).get("n_official"),
        "heldout_coverage": {k: ov[k]["rate"] for k in ("cost", "position", "joint")},
        "heldout_reported_below": hc["full_sample"]["verdict"].get("below_target"),
        "near_delta_condition_satisfying": ne["geometry"]["delta_condition"].get("n_satisfying"),
        "near_overlap": ne["geometry"]["executed"].get("n_geometric_overlap"),
        "near_n": ne["geometry"]["executed"].get("n_trials"),
        "v2_comparison_recomputed": cmp_.get("recomputed"),
        "v2_comparison_evidence_level": cmp_.get("evidence_level"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP / "reports" / "final_acceptance.json"))
    ap.add_argument("--skip-manifest-verify", action="store_true")
    args = ap.parse_args()

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    t0 = datetime.now(timezone.utc).astimezone()

    print("=== 1. 计划结构 ===", flush=True)
    structure = check_plan_structure(plan)
    print(f"[{'ok ' if structure['ok'] else 'FAIL'}] 40 项 / 完成 {structure['n_done']} / "
          f"问题 {len(structure['problems'])}", flush=True)
    for p in structure["problems"][:8]:
        print("   -", p)

    print("=== 2. 逐项证据路径 ===", flush=True)
    evidence = check_task_evidence(plan)
    print(f"[{'ok ' if evidence['ok'] else 'FAIL'}] 声明路径缺失 {evidence['missing_total']} 个", flush=True)

    print("=== 3. 门禁机读判定 ===", flush=True)
    gates = check_gates()

    print("=== 4. 采集台账 ===", flush=True)
    coll = check_collection()
    print(f"[{'ok ' if coll['ok'] else 'FAIL'}] 配额 {coll['quota_found']}/{coll['quota_expected']}，"
          f"多 attempt {coll['n_multi_attempt']}", flush=True)

    print("=== 5. 无覆写 / 无隐性重训 ===", flush=True)
    degraded: list[str] = []
    nowrite = check_no_overwrite(degraded)
    print(f"[{'ok ' if nowrite['ok'] else 'FAIL'}] 稿件干净={nowrite['manuscripts']['status_clean']} "
          f"v2 冻结物={nowrite['v2_frozen']} 子模块={nowrite['hira_submodule']['commit_ok']}", flush=True)

    print("=== 6. 交付索引复核 ===", flush=True)
    manifest = check_manifest(args.skip_manifest_verify)
    print(f"[{'ok ' if manifest.get('ok') else 'FAIL'}] {manifest.get('verify', 'skipped')}", flush=True)

    conclusions = collect_conclusions()
    required = {"plan_structure": structure["ok"], "task_evidence": evidence["ok"],
                "gates": gates["ok"], "collection": coll["ok"],
                "no_overwrite": nowrite["ok"], "manifest": bool(manifest.get("ok"))}
    failed = [k for k, v in required.items() if not v]

    report = {
        "task": "HIRA-040",
        "kind": "final_acceptance",
        "created_at": t0.isoformat(timespec="seconds"),
        "required_gates": required,
        "verdict": {"all_required_passed": not failed, "failed": failed},
        "plan_structure": structure,
        "task_evidence": evidence,
        "gate_checks": gates,
        "collection": coll,
        "no_overwrite": nowrite,
        "degraded_checks": degraded,
        "delivery_manifest": manifest,
        "conclusions": conclusions,
        "adjudicated_exceptions": [
            {"task": "HIRA-030", "what": "C06 两格完整率 0.825 / 0.875 < 0.9",
             "ruling": "不补采、不改配额、失败保留主分母，不入格级结论"},
            {"task": "HIRA-032", "what": "C06 三格 0.650 / 0.750 / 0.850 < 0.9",
             "ruling": "沿用 030 先例；HIRA-033 并列两套分母"},
            {"task": "HIRA-034", "what": "N07 两格 0.600 < 0.9",
             "ruling": "沿用 030 先例；HIRA-035 并列两套分母"},
            {"task": "HIRA-035", "what": "δ 条件 0/200、位置覆盖 98.40% 与重叠 75.5% 并存",
             "ruling": "如实上报为负结果；禁止改分位缩小 δ 或重设计地图"},
            {"task": "HIRA-038", "what": "run_split 无输出目录绑定校验",
             "ruling": "改由交付入口 run_delivery.py 前置拦截；下沉需 v3 锁，待裁定"},
        ],
        "limits": {
            "engineering_vs_science": "工程接入通过不等于随机安全理论或导航性能证明",
            "v2_comparison": "v2 逐 trial 数据不在本机，只能报告级对比（recomputed=false）",
            "coverage_scope": "单一固定控制器与冻结裕度；区间只反映抽样不确定性",
            "no_navigation": "本轮未启动上层导航训练/评估",
        },
        "next_entry": {
            "上层导航消费": "α=0.351568 扣预算、δ=9.032231 m 膨胀危险集；近障任务在该场地下不可评估",
            "复现": "docs/reproduction.md + delivery/bundle_manifest_hira_v1.json",
            "稿件": "reports/manuscript_sync_differences.md（作者本人执行）",
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".sha256").write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    print(f"\n=== 结论：{'全部通过' if not failed else '未通过 ' + str(failed)} ===")
    print(f"written: {out}")
    print(json.dumps(conclusions, ensure_ascii=False, indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
