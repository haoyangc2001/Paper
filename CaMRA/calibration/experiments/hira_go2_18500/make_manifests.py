"""HIRA-024：生成并锁定七分片清单与跨控制器匹配表。

清单是 **(协议, 生成器)** 的纯函数：`camra_calibration.trials.build_all(hi ra_protocol)`
直接复用 v2 那套已被 CPU 测试覆盖的生成器，只在写出时给 trial_id 加 `hira-` 前缀
（采集器白名单要求）。

七分片：pilot20 / baseline_fit240 / baseline_dev80 / baseline_validate80 /
calibration800 / heldout400 / near_obstacle200，合计 1820。

跨控制器匹配：按 **program_sha256 相等**（同 seed + 同生成器 = 同命令程序）在 v2 的
v1.0.2 清单里找对应 trial，写入 source_trial_id；找不到就记 null + 原因，不做近似匹配。

结果区隔离：heldout / near_obstacle 在 α/δ 冻结（HIRA-031）前不采、不分析——
写进 lock 的 consumption_policy，并由 `results_guard.py` 强制。

    python make_manifests.py --v2-lists v2_lists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent   # 本脚本位于实验目录根部
CAL = Path("/home/caohy/repositories/Paper/CaMRA/calibration")
sys.path.insert(0, str(CAL / "src"))

from camra_calibration.commands import build_program  # noqa: E402
from camra_calibration.protocol import Protocol  # noqa: E402
from camra_calibration.trials import (  # noqa: E402
    FAMILIES,
    SPLIT_ORDER,
    build_all,
    check_lists,
    hash_list,
    profiles_for,
    strata_report,
)

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", default=str(EXP / "config/calibration_protocol_hira_v1.json"))
parser.add_argument("--v2-lists", default=str(EXP / "v2_lists"))
parser.add_argument("--out", default=str(EXP / "manifests"))
parser.add_argument("--pilot-run", default="/pub/data/caohy/logs/camra_hira/runs/pilot_20261106_protocolv1")
args_cli = parser.parse_args()

NOW = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands_sha256(commands) -> str:
    """与采集器 TrialSpec.program_sha256 **完全同口径**（只哈希命令表）。

    采集器 meta 里记的就是这一个，所以清单必须登记同口径的字段，
    否则"清单与 meta 的 program_sha256 相等"根本对不上。
    """
    return hashlib.sha256(
        json.dumps([list(map(float, c)) for c in commands], separators=(",", ":")).encode()
    ).hexdigest()


def with_prefix(trial, prefix: str = "hira-"):
    """库里的 trial_id 是 '<split>-<family>-<profile>-<i>'；写出时加前缀。

    只改 id，不改 seed/配额/程序——生成逻辑仍是同一个函数。

    两个散列各司其职：
    * ``program_sha256``  = 命令表散列（= 采集器 meta 口径，供绑定校验）
    * ``template_sha256`` = ``Program.sha256()``（含 family/seed/参数；v2 清单的
      ``program_sha256`` 也是这个口径，跨控制器匹配用它）
    """
    data = trial.as_dict()
    program = build_program(
        data["family"], data["seed"], duration_s=data["duration_s"], command_dt=data["command_dt"]
    )
    data["trial_id"] = prefix + data["trial_id"]
    data["program_sha256"] = commands_sha256(program.commands)
    data["template_sha256"] = program.sha256()
    data["cluster_id"] = f"{data['family']}:{data['template_sha256'][:16]}"
    return data


# --------------------------------------------------------------------------- #
protocol = Protocol.load(Path(args_cli.protocol))
print(f"protocol {protocol.version} sha={protocol.sha256[:16]}")

lists = build_all(protocol)
problems = check_lists(lists, protocol)
if problems:
    print("SELF-CHECK FAILED")
    for item in problems:
        print("  -", item)
    raise SystemExit(2)
print("check_lists: OK（配额/格覆盖/seed 归属/跨 split 泄漏）")

# pilot 用实际采集过的那 20 条（seed 6110..6129，与 build_all 的 pilot 块不同）——
# 已采集的数据必须能被清单解释，不能拿"重新生成的 pilot"替换既成事实
pilot_run = Path(args_cli.pilot_run)
if not pilot_run.is_dir():
    raise SystemExit(f"pilot run 目录不存在：{pilot_run}")
pilot_rows = []
for meta_path in sorted(pilot_run.glob("*/attempt_01/meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pilot_rows.append(
        {
            "trial_id": meta["trial_id"],
            "split": "pilot",
            "family": meta["program"],
            "profile": meta["profile"],
            "seed": meta["seed"],
            "attempt": meta["attempt"],
            "command_dt": meta["timing"]["command_dt"],
            "duration_s": meta["timing"]["window_s"],
            "program_sha256": meta["program_sha256"],   # 采集器口径（命令表）
            "template_sha256": None,                    # 下面用 build_program 重建补齐
            "cluster_id": None,
            "source_run": str(pilot_run),
        }
    )
pilot_rows.sort(key=lambda row: (FAMILIES.index(row["family"]), row["profile"]))
# 一致性：pilot 的 (family, seed) 必须能用同一个生成器重建出同样的命令表散列
pilot_bad = []
for row in pilot_rows:
    prog = build_program(row["family"], row["seed"], duration_s=row["duration_s"])
    row["template_sha256"] = prog.sha256()
    row["cluster_id"] = f"{row['family']}:{row['template_sha256'][:16]}"
    if commands_sha256(prog.commands) != row["program_sha256"]:
        pilot_bad.append(row["trial_id"])
if pilot_bad:
    raise SystemExit(f"pilot 的 program_sha256 无法重建：{pilot_bad}")
print(f"pilot: {len(pilot_rows)} 条来自 {pilot_run.name}（program_sha256 可重建）")

# --------------------------------------------------------------------------- #
# 跨控制器匹配：v2 清单里按 program_sha256 查
v2_root = Path(args_cli.v2_lists)
v2_index: dict[str, dict] = {}
v2_seed_index: dict[int, dict] = {}
v2_counts: dict[str, int] = {}
for path in sorted(v2_root.glob("*.jsonl")):
    split = path.stem
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        v2_index.setdefault(row["program_sha256"], row)
        v2_seed_index.setdefault(row["seed"], row)
        n += 1
    v2_counts[split] = n
print(f"v2 清单索引：{len(v2_index)} 个 program，分片 {v2_counts}")

manifests: dict[str, list[dict]] = {"pilot": pilot_rows}
for split in SPLIT_ORDER:
    if split == "pilot":
        continue
    manifests[split] = [with_prefix(trial) for trial in lists[split]]

for split, rows in manifests.items():
    for row in rows:
        match = v2_index.get(row["template_sha256"])
        row["source_trial_id"] = match["trial_id"] if match else None
        row["source_match"] = "template_sha256_equal" if match else "none"
        if match:
            row["source_split"] = match["split"]

# --------------------------------------------------------------------------- #
# 自检：配额、格覆盖、低速、split 间不相交
lock = {
    "schema": "hira-trial-manifests-v1",
    "generated_at": NOW,
    "protocol": {"file": Path(args_cli.protocol).name, "sha256": protocol.sha256,
                 "version": protocol.version},
    "generator": "camra_calibration.trials.build_all（与 v2 同一实现；写出时仅加 hira- 前缀）",
    "splits": {},
}
independence = {"report": "split_independence", "task": "HIRA-024", "generated_at": NOW, "splits": {}}
seed_sets, program_sets, id_sets, cluster_sets = {}, {}, {}, {}

for split in ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
              "calibration", "heldout", "near_obstacle"):
    rows = manifests[split]
    path = OUT / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digests = [row["template_sha256"] for row in rows]
    lock["splits"][split] = {
        "count": len(rows),
        "sha256": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "file_sha256": sha256_file(path),
        "n_unique_programs": len(set(digests)),
        "n_unique_clusters": len({row["cluster_id"] for row in rows}),
        "n_matched_v2": sum(1 for row in rows if row["source_trial_id"]),
        "source_match_rate": round(sum(1 for row in rows if row["source_trial_id"]) / len(rows), 4),
    }
    seed_sets[split] = {row["seed"] for row in rows}
    program_sets[split] = set(digests)
    id_sets[split] = {row["trial_id"] for row in rows}
    cluster_sets[split] = {row["cluster_id"] for row in rows}

# 配额与低速
strata = strata_report(lists, protocol)
quota_ok = all(
    lock["splits"][split]["count"] ==
    {"pilot": 20, "baseline_fit": 240, "baseline_dev": 80, "baseline_validate": 80,
     "calibration": 800, "heldout": 400, "near_obstacle": 200}[split]
    for split in lock["splits"]
)
total = sum(lock["splits"][s]["count"] for s in lock["splits"])
low_ok = True
low_min = 1.0
for split in SPLIT_ORDER:
    if split == "pilot":
        continue
    info_split = strata["splits"][split]
    if not info_split["all_cells_low_speed_ok"]:
        low_ok = False
    for cell, info in info_split["detail"].items():
        # 协议判据是**格级平均占比** ≥0.25；逐条最小值只作观测
        low_min = min(low_min, info["low_speed_share_strict"])

# 跨 split 交集（seed / program / trial_id / cluster）
pairs = []
split_names = list(lock["splits"])
for i, a in enumerate(split_names):
    for b in split_names[i + 1:]:
        pairs.append(
            {
                "a": a, "b": b,
                "seed_intersection": len(seed_sets[a] & seed_sets[b]),
                "program_intersection": len(program_sets[a] & program_sets[b]),
                "trial_id_intersection": len(id_sets[a] & id_sets[b]),
                "cluster_intersection": len(cluster_sets[a] & cluster_sets[b]),
            }
        )
disjoint_ok = all(
    item["seed_intersection"] == 0 and item["program_intersection"] == 0
    and item["trial_id_intersection"] == 0 and item["cluster_intersection"] == 0
    for item in pairs
)

independence.update(
    {
        "total_trials": total,
        "quota_ok": quota_ok,
        "low_speed_ok": low_ok,
        "low_speed_strict_min_share": round(low_min, 4),
        "pairwise": pairs,
        "disjoint_ok": disjoint_ok,
        "pilot_source": str(pilot_run),
        "pilot_program_rebuildable": True,
        "note": "cluster_id = family:program_sha256[:16]；生成器为每个 (family, seed) 采样独立目标骨架，"
        "因此一条 trial 自成一簇（与协议『bootstrap 以整条轨迹为单位』一致），"
        "簇数 = trial 数，报告里不再另设簇内样本",
    }
)

# 结果区隔离
lock["consumption_policy"] = {
    "results_root": "/pub/data/caohy/logs/camra_hira/results",
    "protected_splits": ["heldout", "near_obstacle"],
    "not_before": "HIRA-031（α/δ 冻结）",
    "rule": "heldout / near_obstacle 在 α/δ 冻结前不得采集、不得分析；"
    "采集与读取入口由 results_guard.py 强制（缺 margins_hira_v1.json 即拒绝）",
    "unprotected_splits": ["pilot", "baseline_fit", "baseline_dev", "baseline_validate", "calibration"],
}
lock["total_trials"] = total
lock["quota_ok"] = quota_ok
lock["low_speed_ok"] = low_ok
lock["cross_split_disjoint"] = disjoint_ok
(OUT / "lock.json").write_text(
    json.dumps(lock, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
)
(EXP / "reports/split_independence.json").write_text(
    json.dumps(independence, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)

# 跨控制器匹配表
match = {
    "report": "cross_controller_match",
    "task": "HIRA-024",
    "generated_at": NOW,
    "v2_source": {
        "root": str(v2_root),
        "protocol_version": "1.0.2",
        "splits": v2_counts,
        "n_unique_programs": len(v2_index),
    },
    "hira_source": {"protocol_sha256": protocol.sha256, "bundle_controller": "hira_go2_18500"},
    "criterion": "template_sha256 逐字相等（= camra_calibration.commands.Program.sha256()，含 family/seed/参数/命令表；"
    "v2 清单里的 program_sha256 是同一口径）；不做近似匹配",
    "per_split": {
        split: {
            "n": lock["splits"][split]["count"],
            "matched": lock["splits"][split]["n_matched_v2"],
            "unmatched": lock["splits"][split]["count"] - lock["splits"][split]["n_matched_v2"],
            "rate": lock["splits"][split]["source_match_rate"],
        }
        for split in lock["splits"]
    },
    "unmatched_breakdown": None,   # 下面逐条计算后填入
    "path_separation": "匹配只共享**命令程序**；执行数据按 controller 分离"
    "（HiRA 在 /pub/data/caohy/logs/camra_hira/runs，v2 在 /pub/data/caohy/logs/camra/trials）",
    "hash_semantics": {
        "hira_manifest.program_sha256": "命令表散列（= 采集器 meta 的 program_sha256，供绑定校验）",
        "hira_manifest.template_sha256": "Program.sha256()（跨控制器匹配键）",
        "v2_manifest.program_sha256": "Program.sha256()（与 HiRA 的 template_sha256 同口径）",
    },
    "warning": "重用命令 ≠ 重用数据；也不保证同 seed 在两种控制器上实现了相同扰动"
    "（profile 采样按 seed 与 profile 决定，公式同源但执行侧物理不同）",
}
# 未匹配原因分解：seed 不存在 vs seed 存在但落在别的族（格边界错位）
breakdown = {}
for split, rows in manifests.items():
    seed_absent = 0
    other_family = 0
    for row in rows:
        if row["source_trial_id"]:
            continue
        other = v2_seed_index.get(row["seed"])
        if other is None:
            seed_absent += 1
        else:
            other_family += 1
    breakdown[split] = {"seed_absent_in_v2": seed_absent, "seed_exists_other_family": other_family}
match["unmatched_breakdown"] = breakdown
match["unmatched_explanation"] = (
    "seed_absent_in_v2：该 seed 不在 v2 的对应块内（例如 HiRA calibration 6130–6929 与 "
    "v2 6000–6799 的差集 6800–6929）。seed_exists_other_family：seed 相同但 v2 把它分配给了"
    "另一个命令族——因为 HiRA 的块起点与 v2 相差 130，而每格 40 个 seed、按 4 族×5 条件展开，"
    "130 mod 40 = 10 使格边界错位。baseline/heldout/near 的块起点与 v2 相同，故 100% 配对。"
)
(EXP / "reports/cross_controller_match.json").write_text(
    json.dumps(match, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)

print(f"total {total} trials; quota_ok={quota_ok} low_speed_ok={low_ok} disjoint_ok={disjoint_ok}")
for split, info in lock["splits"].items():
    print(f"  {split:<18} {info['count']:>4}  matched_v2={info['n_matched_v2']:>4}  "
          f"file={info['file_sha256'][:12]}")
print("MANIFESTS_OK", flush=True)
