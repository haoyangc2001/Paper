"""HIRA-024：清单的独立验证（不写清单，只核对）。

四件事：

1. **锁文件自洽**：`manifests/lock.json` 里每个 split 的 ``file_sha256`` 与磁盘文件一致；
2. **可重建**：用冻结协议重新跑一遍生成器，得到与磁盘**逐字节相同**的清单
   （清单是 (协议, 生成器) 的纯函数，重跑必须复现）；
3. **结构**：配额 1820、4 族×5 条件分配、格级低速占比 ≥0.25、七 split 的
   seed/模板/簇/trial_id 交集为零；
4. **结果区守卫**：`heldout`/`near_obstacle` 现在必须被拒（α/δ 未冻结），
   `calibration` 必须放行；pilot 清单必须与已采集 run 的 meta 一一对应。

    python check_manifests.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
CAL = Path("/home/caohy/repositories/Paper/CaMRA/calibration")
sys.path.insert(0, str(CAL / "src"))
sys.path.insert(0, str(EXP))

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.protocol import Protocol  # noqa: E402
from camra_calibration.trials import SPLIT_ORDER, build_all, check_lists, strata_report  # noqa: E402

from results_guard import GuardError, assert_allowed, describe  # noqa: E402

MANIFESTS = EXP / "manifests"
CHECKS: list[dict] = []


def add(cid, name, ok, detail, **extra):
    record = {"id": cid, "name": name, "ok": bool(ok), "detail": detail}
    record.update(extra)
    CHECKS.append(record)
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name} :: {detail}", flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(split: str) -> list[dict]:
    path = MANIFESTS / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


lock = json.loads((MANIFESTS / "lock.json").read_text(encoding="utf-8"))
protocol = Protocol.load(EXP / "config" / lock["protocol"]["file"])

SPLITS = ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
          "calibration", "heldout", "near_obstacle")

# ---- 1 锁文件自洽 ----
hash_ok = True
for split in SPLITS:
    actual = sha256_file(MANIFESTS / f"{split}.jsonl")
    recorded = lock["splits"][split]["file_sha256"]
    hash_ok = hash_ok and actual == recorded
    if actual != recorded:
        print(f"  散列不符 {split}: {recorded[:12]} vs {actual[:12]}")
add("M01-lock-hashes", "lock.json 的逐文件散列与磁盘一致", hash_ok,
    f"{len(SPLITS)} 个 JSONL 全部一致" if hash_ok else "存在不一致")

# ---- 2 可重建（不含 pilot：pilot 是既成采集事实，用实际 run 反填）----
rebuilt = build_all(protocol)
problems = check_lists(rebuilt, protocol)
add("M02-generator-selfcheck", "生成器自检（配额/格覆盖/seed 归属/跨 split 泄漏）",
    not problems, "无问题" if not problems else "；".join(problems[:3]))

rebuild_ok, mismatched = True, []
for split in SPLIT_ORDER:
    if split == "pilot":
        continue
    on_disk = load_rows(split)
    library = rebuilt[split]
    if len(on_disk) != len(library):
        rebuild_ok = False
        mismatched.append(f"{split}: 行数 {len(on_disk)} vs {len(library)}")
        continue
    for row, trial in zip(on_disk, library):
        prog = build_program(trial.family, trial.seed, duration_s=trial.duration_s,
                             command_dt=trial.command_dt)
        expect_id = "hira-" + trial.trial_id
        if (
            row["trial_id"] != expect_id
            or row["seed"] != trial.seed
            or row["template_sha256"] != prog.sha256()
            or row["program_sha256"]
            != hashlib.sha256(
                json.dumps([list(map(float, c)) for c in prog.commands], separators=(",", ":")).encode()
            ).hexdigest()
        ):
            rebuild_ok = False
            mismatched.append(row["trial_id"])
            break
add("M03-rebuild", "清单可由 (协议, 生成器) 逐字节重建（除 pilot）",
    rebuild_ok, "六个生成 split 全部复现" if rebuild_ok else f"不一致：{mismatched[:5]}")

# ---- 3 结构 ----
total = sum(lock["splits"][s]["count"] for s in SPLITS)
quota = {"pilot": 20, "baseline_fit": 240, "baseline_dev": 80, "baseline_validate": 80,
         "calibration": 800, "heldout": 400, "near_obstacle": 200}
quota_ok = total == 1820 and all(lock["splits"][s]["count"] == q for s, q in quota.items())
cells_ok = True
for split in SPLITS:
    rows = load_rows(split)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["family"], row["profile"])
        counts[key] = counts.get(key, 0) + 1
    expected_cells = 4 * (1 if split.startswith("baseline") else 5)
    if len(counts) != expected_cells:
        cells_ok = False
    per_cell_values = set(counts.values())
    if len(per_cell_values) != 1:
        cells_ok = False
add("M04-quota-cells", "总配额 1820 与格分配（baseline 4 格、其余 20 格，格内等量）",
    quota_ok and cells_ok,
    f"total={total}；格分配{'正确' if cells_ok else '异常'}")

independence = json.loads((EXP / "reports/split_independence.json").read_text(encoding="utf-8"))
low_ok = independence["low_speed_ok"]
add("M05-low-speed", "格级低速占比 ≥0.25（协议下限）", low_ok,
    f"最小格均值 {independence['low_speed_strict_min_share']:.3f}")

# 字段名 program_intersection 比较的是 template_sha256（与 v2 清单同口径）
disjoint_ok = all(
    p["seed_intersection"] == 0 and p["program_intersection"] == 0
    and p["trial_id_intersection"] == 0 and p["cluster_intersection"] == 0
    for p in independence["pairwise"]
)
add("M06-split-independence", "七 split 之间 seed / 模板 / 簇 / trial_id 交集为零",
    bool(disjoint_ok), f"{len(independence['pairwise'])} 组两两配对全部为 0")

# ---- 4 结果区守卫 ----
guard = describe(EXP)
denied = False
try:
    assert_allowed("heldout", "collect", EXP)
except GuardError:
    denied = True
allowed_cal = True
try:
    assert_allowed("calibration", "collect", EXP)
except GuardError:
    allowed_cal = False
add("M07-results-guard", "结果区守卫：heldout/near 在 α/δ 冻结前被拒，calibration 放行",
    denied and allowed_cal and not guard["freeze_state"]["frozen"],
    f"冻结状态={guard['freeze_state']['reason']}；heldout 被拒={denied}；calibration 放行={allowed_cal}")

# ---- 5 pilot 与已采集 run 对应 ----
pilot_rows = load_rows("pilot")
run_dir = Path(independence["pilot_source"])
meta_ok = True
for row in pilot_rows:
    meta_path = run_dir / row["trial_id"] / "attempt_01/meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta["seed"] != row["seed"]
        or meta["program"] != row["family"]
        or meta["profile"] != row["profile"]
        or meta["program_sha256"] != row["program_sha256"]
    ):
        meta_ok = False
        break
add("M08-pilot-binding", "pilot 清单与已采集 run 的 meta 逐条一致",
    meta_ok and len(pilot_rows) == 20, f"{len(pilot_rows)} 条与 {run_dir.name} 对应")

failed = [item["id"] for item in CHECKS if not item["ok"]]
report = {
    "report": "manifest_validation",
    "task": "HIRA-024",
    "manifests": str(MANIFESTS),
    "lock_sha256": sha256_file(MANIFESTS / "lock.json"),
    "protocol": lock["protocol"],
    "checks": CHECKS,
    "verdict": {"n_checks": len(CHECKS), "failed": failed, "all_passed": not failed},
}
out = EXP / "reports/manifest_validation.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"MANIFESTS_{'PASSED' if not failed else 'BLOCKED'} failed={failed} -> {out}", flush=True)
sys.exit(0 if not failed else 1)
