"""HIRA-031：检验 stage lock 与 margin 的绑定核验（放行 heldout / near_obstacle 之前的最后一道）。

**为什么单独一个脚本而不是改 `camra_calibration/lock.py`**：
`enforce_collection` 对 MARGIN_GATED_SPLITS 只检查"margins_path 这个文件存在"，**没有**校验
锁里 `margins.state` 与 `margins.sha256`。想补这个洞就得改 `lock.py`，而 `lock.py` 在采集锁 v2 与
检验锁 v1 的 `code` 散列里 —— 改它会让已采 800 条数据的归属锁验签失败，代价大于收益。
所以：本脚本把绑定关系做成**可执行、可复现的断言**，heldout / near_obstacle 开跑前必须先跑它；
把"改 lock.py"记进下一版本协议的待办。

    python tests/check_margin_lock_binding.py
    python tests/check_margin_lock_binding.py --lock config/data_lock_hira_evaluation_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from camra_calibration.lock import DataLock, LockError, enforce_collection  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--lock", default=str(EXP / "config/data_lock_hira_evaluation_v1.json"))
parser.add_argument("--collect-lock", default=str(EXP / "config/data_lock_hira_calibration_v2.json"))
parser.add_argument("--margins", default=str(EXP / "config/margins_hira_v1.json"))
parser.add_argument("--maps", default=str(EXP / "config/near_obstacle_maps_v1.json"))
parser.add_argument("--manifests", default=str(EXP / "manifests"))
parser.add_argument("--out", default=str(EXP / "reports/margin_lock_binding.json"))
args_cli = parser.parse_args()

CHECKS: list[dict] = []


def add(cid: str, name: str, ok: bool, detail: str) -> None:
    CHECKS.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


CODE_GLOBS = (
    (REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2", "*.py"),
    (REPO / "CaMRA" / "calibration" / "src" / "camra_calibration", "*.py"),
    (EXP / "tests", "run_split.py"),
)


def code_tree() -> dict:
    """当前代码树的散列（与 make_data_lock_hira.py 的 CODE_GLOBS 同口径）。"""
    files = {}
    for base, pattern in CODE_GLOBS:
        for path in sorted(base.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            files[str(path)] = sha256_file(path)
    return {"files": files, "n_files": len(files)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    eval_lock = DataLock.load(Path(args_cli.lock), verify_hash=True)
    margins_path = Path(args_cli.margins)
    margins = json.loads(margins_path.read_text(encoding="utf-8"))
    recorded = eval_lock.payload.get("margins") or {}

    add("B01-stage", "检验锁的 stage 必须是 evaluation", eval_lock.payload.get("stage") == "evaluation",
        f"stage={eval_lock.payload.get('stage')!r}")
    add("B02-margins-frozen", "锁内 margins.state == frozen", recorded.get("state") == "frozen",
        f"state={recorded.get('state')!r}")
    add("B03-margins-file-hash", "margin 文件散列 = 锁记录",
        recorded.get("sha256") == sha256_file(margins_path),
        f"锁 {str(recorded.get('sha256'))[:16]} / 现场 {sha256_file(margins_path)[:16]}")
    add("B04-margins-content", "margin 内容散列 = 锁记录（防文件被替换但 sha 字段手改）",
        recorded.get("content_sha256") == margins.get("sha256"),
        f"锁 {str(recorded.get('content_sha256'))[:16]} / 内容 {str(margins.get('sha256'))[:16]}")
    add("B05-alpha-delta", "锁内 α/δ 与 margin 文件逐位一致",
        recorded.get("alpha") == margins.get("alpha") and recorded.get("delta") == margins.get("delta"),
        f"α {recorded.get('alpha')} vs {margins.get('alpha')}；δ {recorded.get('delta')} vs {margins.get('delta')}")
    add("B06-parent", "检验锁的 parent 是采集锁 v2（数据归属链完整）",
        (eval_lock.payload.get("parent_lock") or {}).get("sha256")
        == sha256_file(Path(args_cli.collect_lock)),
        f"parent={(eval_lock.payload.get('parent_lock') or {}).get('sha256', '')[:16]}")

    # 正测：检验锁下 heldout / near_obstacle 必须放行
    for split in ("heldout", "near_obstacle"):
        try:
            enforce_collection(eval_lock, split, REPO, manifest_dir=Path(args_cli.manifests),
                               maps_path=args_cli.maps if split == "near_obstacle" else None,
                               margins_path=margins_path)
            ok, detail = True, "放行"
        except LockError as exc:
            ok, detail = False, str(exc)
        add(f"B07-{split}", f"检验锁放行 {split}", ok, detail)

    # 负测：采集锁（margins 未产出）不得作为检验放行依据
    collect_lock = DataLock.load(Path(args_cli.collect_lock), verify_hash=True)
    recorded_c = collect_lock.payload.get("margins") or {}
    add("B08-collect-lock-state", "采集锁的 margins 必须是 not_yet_produced",
        recorded_c.get("state") == "not_yet_produced", f"state={recorded_c.get('state')!r}")

    # 已知缺口：enforce_collection 目前只看 margins 文件存在，不校验锁内绑定。
    #
    # 控制变量（HIRA-034 起必须这么做）：本轮给近障补了障碍注入，改了 adapter 代码，
    # 采集锁 v2 的 code 散列已与当前代码树不符。若直接拿它测，"采集锁 + margin 文件"
    # 会**先在 code 层被拒**——那不是缺口消失，而是被更强的原因掩盖，这个诊断项也就
    # 测不到它本来要测的东西（margins 绑定）。所以这里只隔离 code 一个变量：
    # 复制采集锁，把 code 段换成当前代码树散列，其余（margins=not_yet_produced 等）逐字不动。
    import copy
    import tempfile

    direct_rejected = None
    try:
        enforce_collection(collect_lock, "heldout", REPO, manifest_dir=Path(args_cli.manifests),
                           margins_path=margins_path)
        direct_rejected = False
    except LockError as exc:
        direct_rejected = str(exc)

    probe_payload = copy.deepcopy(collect_lock.payload)
    probe_payload["code"] = code_tree()
    probe_payload["probe"] = {
        "purpose": "B09 控制变量：仅把 code 段对齐当前代码树，用于测 margins 绑定缺口",
        "original_code_sha_mismatch": direct_rejected,
    }
    probe_dir = Path(tempfile.mkdtemp(prefix="lockprobe_"))
    probe_path = probe_dir / "collect_probe.json"
    probe_path.write_text(json.dumps(probe_payload, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    probe_path.with_suffix(probe_path.suffix + ".sha256").write_text(
        f"{sha256_file(probe_path)}  {probe_path.name}\n", encoding="utf-8")
    try:
        probe_lock = DataLock.load(probe_path, verify_hash=True)
        enforce_collection(probe_lock, "heldout", REPO, manifest_dir=Path(args_cli.manifests),
                           margins_path=margins_path)
        gap_reproduced, why = True, "code 对齐后仍放行 → lock.py 尚未校验锁内 margins 绑定"
    except LockError as exc:
        gap_reproduced, why = False, f"code 对齐后仍被拒（{exc}）→ 缺口可能已被修复"
    except Exception as exc:  # noqa: BLE001
        gap_reproduced, why = False, f"探针锁不可用：{exc}"
    add("B09-known-gap",
        "复现已知缺口：code 对齐后，采集锁 + 存在的 margin 文件仍会被放行（R-07，未修复）",
        gap_reproduced,
        f"{why}；直测原始采集锁的结果={direct_rejected!r}（code 散列已因 HIRA-034 变更失效，"
        "故必须用 code 对齐的探针锁才能测到 margins 层）")

    report = {
        "report": "margin_lock_binding",
        "task": "HIRA-031",
        "lock": {"path": str(args_cli.lock), "sha256": sha256_file(Path(args_cli.lock))},
        "margins": {"path": str(args_cli.margins), "sha256": sha256_file(margins_path)},
        "known_gap": {
            "id": "R-07",
            "what": "enforce_collection 对 margin-gated split 只校验 margins 文件存在，不校验锁内 margins.state/sha256",
            "why_not_fixed_now": "lock.py 在 v2 采集锁与 eval v1 检验锁的 code 散列内，改它会让已采 800 条的归属锁验签失败",
            "compensation": "本脚本作为 heldout/near_obstacle 开跑前的强制检查；下一版协议改 lock.py 后由 B09 转为负测",
        },
        "checks": CHECKS,
        "verdict": {"all_passed": all(c["ok"] for c in CHECKS),
                    "failed": [c["id"] for c in CHECKS if not c["ok"]]},
    }
    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"BINDING {'OK' if report['verdict']['all_passed'] else 'FAIL'} "
          f"checks={sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} failed={report['verdict']['failed']}",
          flush=True)
    return 0 if report["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
