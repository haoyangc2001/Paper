"""HIRA-034：派生检验锁 v2（只更新代码散列，α/δ 与父锁一律不动）。

为什么需要新锁：HIRA-034 给近障补了障碍注入，改了三个**在检验锁 code 散列里**的文件
（`env_runner.py`、`collector.py`、`run_split.py`）并新增 `obstacles.py`。按红线
"改了被锁绑定的代码就必须出新版本锁；锁只能新增，不能编辑"，旧锁当场失效。

这个脚本**不重算** α/δ、不重写 margins、不重写采集锁——它只做三件事：

1. 重新散列代码树（自动纳入新增的 obstacles.py）；
2. 其余各段（controller / protocol / power_model / footprint / step_cost / asset /
   trial_lists / near_obstacle_maps / margins / gates）**逐字沿用** v1；
3. stage 仍写 ``evaluation``（多个检查脚本硬校验该字符串），版本靠文件名与 amendment 区分。

heldout 400 条的数据归属仍留在 v1 锁（`data_lock_hira_evaluation_v1.json`），
本脚本不改它、不覆盖它。

    python tests/make_eval_lock_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent

CODE_GLOBS = (
    (REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2", "*.py"),
    (REPO / "CaMRA" / "calibration" / "src" / "camra_calibration", "*.py"),
    (EXP / "tests", "run_split.py"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--from", dest="src", default=str(EXP / "config/data_lock_hira_evaluation_v1.json"),
                    help="被沿用的检验锁（v1）")
parser.add_argument("--out", default=str(EXP / "config/data_lock_hira_evaluation_v2.json"))
parser.add_argument("--note", default=None)
args_cli = parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree() -> dict:
    files = {}
    for base, pattern in CODE_GLOBS:
        for path in sorted(base.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            files[str(path)] = sha256_file(path)
    return {"files": files, "n_files": len(files)}


def main() -> int:
    src_path = Path(args_cli.src)
    base = json.loads(src_path.read_text(encoding="utf-8"))
    if base.get("stage") != "evaluation":
        raise SystemExit(f"REFUSED：源锁 stage={base.get('stage')!r}，应为 evaluation")

    old_code = base["code"]
    new_code = code_tree()
    changed = sorted(f for f in new_code["files"]
                     if old_code["files"].get(f) != new_code["files"][f])
    removed = sorted(f for f in old_code["files"] if f not in new_code["files"])

    payload = dict(base)          # 逐字沿用所有段
    payload["code"] = new_code
    payload["amendment"] = {
        "reason": "HIRA-034 补近障障碍注入：新增 obstacles.py，改 env_runner.py / collector.py / run_split.py",
        "supersedes": {"path": str(src_path), "sha256": sha256_file(src_path)},
        "changed_code_files": [Path(f).name for f in changed],
        "removed_code_files": [Path(f).name for f in removed],
        "n_code_files_old": int(old_code.get("n_files", len(old_code["files"]))),
        "n_code_files_new": int(new_code["n_files"]),
        "margins_unchanged": True,
        "parent_unchanged": True,
        "heldout_attribution": "400 条 heldout 仍归属 v1 锁；本锁只用于 near_obstacle",
        "note": args_cli.note,
    }
    if payload.get("cal_task"):
        payload["cal_task"] = "HIRA-034"

    out = Path(args_cli.out)
    if out.exists():
        raise SystemExit(f"REFUSED：{out.name} 已存在，锁只能新增不能编辑")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")

    print(f"[lock] wrote {out.name} sha={sha256_file(out)[:16]}")
    print(f"[lock] code files {old_code.get('n_files', len(old_code['files']))} -> {new_code['n_files']}")
    print(f"[lock] changed: {[Path(f).name for f in changed]}")
    if removed:
        print(f"[lock] WARNING removed: {[Path(f).name for f in removed]}")
    print(f"[lock] margins untouched: {payload['margins']['sha256'][:16]} state={payload['margins']['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
