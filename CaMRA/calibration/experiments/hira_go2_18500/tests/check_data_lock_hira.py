"""HIRA-029 第二步：G2 门禁的负测试与资源复核（放行 800 条校准前必须全过）。

正品锁必须放行 `calibration`，而下列每一种"看起来能跑"的情形都必须被拒：
锁被改、模型/轮廓/清单/代码散列不符、缺 margin 时采 heldout/near、split 拼错、
正式 split 带 `allow_overwrite`。另外复核 1820 配额剩余、磁盘余量与并发。

    python tests/check_data_lock_hira.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from camra_calibration.lock import DataLock, LockError, enforce_collection  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_calibration_v1.json"))
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
parser.add_argument("--tmp", default="/pub/data/caohy/tmp/lock_neg")
parser.add_argument("--out", default=str(EXP / "reports/formal_gate.json"))
args_cli = parser.parse_args()

CHECKS: list[dict] = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> None:
    CHECKS.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


def expect_reject(label: str, payload_patch, *, split: str = "calibration",
                  with_sidecar: bool = True, allow_overwrite: bool = False,
                  margins: str | None = None, maps: str | None = None,
                  skip_lock_write: bool = False) -> tuple[bool, str]:
    """把锁改一处后落盘（可同步重算侧车），返回 (是否被拒, 错误摘要)。"""
    tmp = Path(args_cli.tmp) / label
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    path = tmp / "lock.json"
    payload = json.loads(Path(args_cli.lock).read_text(encoding="utf-8"))
    payload_patch(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if with_sidecar:
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    try:
        lock = DataLock.load(path, verify_hash=True)
        enforce_collection(lock, split, REPO, manifest_dir=Path(args_cli.manifests),
                           maps_path=maps, margins_path=margins, allow_overwrite=allow_overwrite)
    except LockError as exc:
        # 保留完整错误（多行）：负测试要断言"报的是哪一类"而不是笼统的"被拒"
        return True, str(exc).replace("\n", " ").strip()[:400]
    return False, "未被拒绝（不符合预期）"


def main() -> int:
    lock_path = Path(args_cli.lock)
    lock = DataLock.load(lock_path, verify_hash=True)
    print(f"[g2] lock {lock_path.name} sha={lock.sha256[:16]} stage={lock.payload.get('stage')}", flush=True)

    # ---- L01 正品锁放行 calibration ----
    ok = True
    try:
        enforce_collection(lock, "calibration", REPO, manifest_dir=Path(args_cli.manifests),
                           maps_path=None, margins_path=args_cli.margins)
    except LockError as exc:
        ok = False
        detail = str(exc)
    add("L01-allow-calibration", "正品锁放行 calibration（仿真启动前的唯一入口）", ok,
        "放行" if ok else f"被拒：{detail}")

    # ---- L02 锁内容被改（侧车不同步）----
    tmp = Path(args_cli.tmp) / "tamper_no_sidecar"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["purpose"] = raw.get("purpose", "") + "（被改）"
    path = tmp / "lock.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(lock_path.with_suffix(lock_path.suffix + ".sha256"),
                    path.with_suffix(path.suffix + ".sha256"))
    rejected = False
    try:
        DataLock.load(path, verify_hash=True)
    except LockError:
        rejected = True
    add("L02-lock-tamper", "锁内容被改而侧车未更新 → 拒绝加载", rejected,
        f"被拒={rejected}（侧车散列不符）")

    # ---- L03..L06 下游散列不符（侧车同步重算，专测 verify_artifacts）----
    cases = [
        ("L03-model-hash", "功率模型散列改错", lambda p: p["power_model"].__setitem__("sha256", "0" * 64), "power_model"),
        ("L04-footprint-hash", "足迹轮廓散列改错", lambda p: p["footprint"].__setitem__("sha256", "1" * 64), "footprint"),
        ("L05-manifest-hash", "清单散列改错",
         lambda p: p["trial_lists"]["files"]["calibration"].__setitem__("sha256", "2" * 64), "trial_lists"),
        ("L06-code-hash", "代码散列改错",
         lambda p: p["code"]["files"].__setitem__(next(iter(p["code"]["files"])), "3" * 64), "code"),
    ]
    for cid, name, patch, key in cases:
        rejected, detail = expect_reject(cid.split("-")[0].lower(), patch)
        add(cid, f"{name}（侧车已同步重算）→ verify_artifacts 报错", rejected and key in detail, detail)

    # ---- L07/L08 margin 未产出时检验与近障被拒 ----
    missing_margins = str(Path(args_cli.tmp) / "no_such_margins.json")
    for cid, split in (("L07-heldout-margin", "heldout"), ("L08-near-margin", "near_obstacle")):
        rejected, detail = expect_reject(cid.split("-")[0].lower(), lambda p: None,
                                         split=split, margins=missing_margins, maps=args_cli.maps)
        add(cid, f"{split} 在 margin 未产出时必须被拒", rejected and "margin" in detail.lower(), detail)

    # ---- L09 split 拼错 ----
    rejected, detail = expect_reject("l09", lambda p: None, split="calibraton")
    add("L09-unknown-split", "拼错的 split 被拒", rejected and "不在数据锁" in detail, detail)

    # ---- L10 正式 split 带 allow_overwrite ----
    rejected, detail = expect_reject("l10", lambda p: None, allow_overwrite=True)
    add("L10-no-overwrite", "正式 split 不允许 allow_overwrite", rejected and "overwrite" in detail.lower(), detail)

    # ---- L11 锁文件缺失 ----
    missing_rejected = False
    try:
        DataLock.load(Path(args_cli.tmp) / "not_there.json", verify_hash=True)
    except LockError:
        missing_rejected = True
    add("L11-missing-lock", "锁文件缺失即拒绝", missing_rejected, f"被拒={missing_rejected}")

    # ---- L12 地图副本与 v2 原始散列一致 ----
    maps_path = Path(args_cli.maps)
    v2_ref = "78c0145f0ecddc71edaf2f64de7be05af3bd4d690e110b54ebaedfa6d9952ec3"
    maps_ok = maps_path.is_file() and sha256_file(maps_path) == v2_ref
    add("L12-maps-unchanged", "近障地图是 v2 原始物理地图的只读副本（未改动）", maps_ok,
        f"{maps_path.name} sha={sha256_file(maps_path)[:16] if maps_path.is_file() else 'missing'} "
        f"（v2 原始 {v2_ref[:16]}）")

    # ---- L13 配额与资源复核 ----
    files = lock.payload["trial_lists"]["files"]
    quota_total = sum(item["count"] for item in files.values())
    collected = 0
    runs_root = Path("/pub/data/caohy/logs/camra_hira/runs")
    for run_dir in runs_root.glob("baseline_20261106_*"):
        collected += sum(1 for p in run_dir.iterdir() if p.is_dir())
    usage = shutil.disk_usage("/pub/data/caohy")
    free_gb = usage.free / 1e9
    running = subprocess.run(["pgrep", "-f", "run_[s]plit"], capture_output=True, text=True).stdout.split()
    resource = {
        "quota_total": quota_total,
        "baseline_collected": collected,
        "calibration_quota": files["calibration"]["count"],
        "disk_free_gb": round(free_gb, 1),
        "disk_required_gb_for_800": round(800 * 1.135 / 1000, 2),
        "concurrent_collectors": len(running),
    }
    add("L13-resources", "配额/磁盘/并发复核：可放行 800 条校准",
        quota_total == 1820 and free_gb >= 5.0 and len(running) == 0,
        f"总配额 {quota_total}（已采 baseline {collected}、calibration 待采 {resource['calibration_quota']}）；"
        f"磁盘余量 {free_gb:.1f} GB（需 {resource['disk_required_gb_for_800']} GB）；"
        f"并发采集进程 {len(running)}", resource=resource)

    failed = [c["id"] for c in CHECKS if not c["ok"]]
    gates = lock.payload.get("gates", {})
    gates_ok = all(v.get("passed") for v in gates.values())
    report = {
        "report": "formal_gate",
        "task": "HIRA-029",
        "lock": {"path": str(lock_path), "sha256": lock.sha256,
                 "stage": lock.payload.get("stage"),
                 "margins_state": lock.payload.get("margins", {}).get("state")},
        "doors_checked": {
            "controller_bundle": lock.payload["controller"]["bundle_id"],
            "protocol": lock.payload["protocol"]["sha256"][:16],
            "power_model": lock.payload["power_model"]["sha256"][:16],
            "footprint": lock.payload["footprint"]["sha256"][:16],
            "step_cost_schema": lock.payload["step_cost_schema"]["sha256"][:16],
            "trial_lists_total": lock.payload["trial_lists"]["total_trials"],
            "code_files": lock.payload["code"]["n_files"],
            "near_obstacle_maps": lock.payload["near_obstacle_maps"]["sha256"][:16],
        },
        "prerequisite_gates": gates,
        "prerequisite_gates_all_passed": gates_ok,
        "negative_tests": CHECKS,
        "verdict": {
            "g2_passed": not failed and gates_ok,
            "failed": failed,
            "note": "校准采集放行条件：6 项前置门禁全过 + 13 项锁门禁测试全过 + margin 未产出（检验/近障仍封锁）",
        },
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"G2 {'PASS' if report['verdict']['g2_passed'] else 'FAIL'} "
          f"checks={sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} failed={failed}", flush=True)
    return 0 if report["verdict"]["g2_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
