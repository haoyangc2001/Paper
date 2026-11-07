"""HIRA-038：把 stage lock 里的绝对路径改写成"相对新根"，生成派生锁（不改原锁）。

为什么需要
----------
三把锁（采集 v1/v2、检验 v1/v2）里 `code.files` 的键、`controller.*`、`protocol`、
`margins`、`trial_lists`、`gates`、`near_obstacle_maps` 全是
`/pub/data/caohy/work/hira/CaMRA/...` 的**绝对路径**。`lock._resolve()` 对绝对路径
原样返回，所以换个目录跑，`verify_artifacts()` 查的还是原位置——独立重载就成了假重载。

本脚本只做一件事：把路径前缀换成相对路径（`CaMRA/...`），让
`_resolve(<新根>, rel)` 落到新根的副本上。**所有 sha256 一律不动**，语义完全等价；
原锁一字不改（红线：锁只增不改）。

用法
----
    python tests/delivery_relock.py \
        --lock config/data_lock_hira_evaluation_v2.json \
        --out  <new_root>/CaMRA/calibration/experiments/hira_go2_18500/config/data_lock_hira_evaluation_delivery.json \
        --new-root <new_root> \
        --old-prefix /pub/data/caohy/work/hira/

成功后立刻用 `DataLock.load(..., verify_hash=True)` + `verify_artifacts(new_root)` 自证：
派生锁在新根下"散列全对"。返回码 0 = 通过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
sys.path.insert(0, str(REPO / "CaMRA" / "calibration" / "src"))

from camra_calibration.lock import DataLock, LockError, sha256_file  # noqa: E402

PATH_KEYS = ("path",)


def rebase(obj, old_prefix: str, new_prefix: str, hits: list[str], depth: int = 0):
    """递归改写所有 path 值与 code.files 的键。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in PATH_KEYS and isinstance(v, str) and v.startswith(old_prefix):
                new = new_prefix + v[len(old_prefix):]
                hits.append(f"{v} -> {new}")
                out[k] = new
                continue
            if k == "files" and isinstance(v, dict):
                files = {}
                for fk, fv in v.items():
                    if isinstance(fk, str) and fk.startswith(old_prefix):
                        nk = new_prefix + fk[len(old_prefix):]
                        hits.append(f"{fk} -> {nk}")
                        files[nk] = fv
                    else:
                        files[fk] = fv
                out[k] = files
                continue
            out[k] = rebase(v, old_prefix, new_prefix, hits, depth + 1)
        return out
    if isinstance(obj, list):
        return [rebase(v, old_prefix, new_prefix, hits, depth + 1) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-root", required=True, help="新根目录：相对路径将以此为基准解析")
    ap.add_argument("--old-prefix", default="/pub/data/caohy/work/hira/",
                    help="锁内绝对路径的公共前缀；该前缀之后的部分保留为相对路径")
    ap.add_argument("--new-prefix", default="", help="替换成的前缀（默认空 = 纯相对路径）")
    args = ap.parse_args()

    src = Path(args.lock)
    out = Path(args.out)
    new_root = Path(args.new_root).resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    original_sha = sha256_file(src)

    hits: list[str] = []
    rebased = rebase(payload, args.old_prefix, args.new_prefix, hits)
    rebased["delivery_rebase"] = {
        "kind": "path_rebase_for_independent_reload",
        "source_lock": str(src),
        "source_lock_sha256": original_sha,
        "old_prefix": args.old_prefix,
        "new_prefix": args.new_prefix,
        "new_root": str(new_root),
        "n_paths_rewritten": len(hits),
        "hash_unchanged": True,
        "note": ("只改路径前缀，所有 sha256 与判据逐字保留；原锁未修改。"
                 "派生锁必须在 new_root 布局下使用，否则 verify_artifacts 会报缺失。"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rebased, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")

    # 自证：新根下散列全对
    result = {
        "source_lock": str(src),
        "source_lock_sha256": original_sha,
        "out_lock": str(out),
        "out_lock_sha256": sha256_file(out),
        "new_root": str(new_root),
        "n_paths_rewritten": len(hits),
        "rewrite_sample": hits[:3],
        "verify_ok": False,
        "verify_bad": [],
    }
    try:
        lock = DataLock.load(out, verify_hash=True)
        bad = lock.verify_artifacts(new_root)
        result["verify_bad"] = bad
        result["verify_ok"] = not bad
        result["stage"] = lock.payload.get("stage")
        result["margins_state"] = (lock.payload.get("margins") or {}).get("state")
        result["n_code_files"] = len((lock.payload.get("code") or {}).get("files") or {})
    except LockError as exc:
        result["verify_bad"] = [f"加载失败：{exc}"]
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["verify_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
