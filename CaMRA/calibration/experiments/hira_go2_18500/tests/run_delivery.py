"""HIRA-038：交付包的唯一正式采集入口（带输出目录绑定守卫）。

背景
----
`tests/run_split.py` 只在 **首次** 写 `collect_config.json`，之后启动时**不校验**既有
run 目录绑的是哪个 split / 哪把锁 / 哪个 bundle。也就是说把 `--runs-root` 或 `--run-id`
指错了，它会顺顺当当把新数据写进一个别的 run 目录里，不报错（HIRA-038 的 D12 探针已实测）。

`run_split.py` 与 `collector.py` 都在 stage lock 的 `code.files` 内，按红线不能随手改
（要改就得出新版锁）。所以在**交付层**加一道守卫：本脚本先做绑定核验，再原样 exec
`run_split.py`。核验不过就直接拒绝启动，一个字节都不写。

用法（与 run_split 同名参数，其余参数原样透传）
------------------------------------------------
    python tests/run_delivery.py --split near_obstacle --run-id near_obstacle_20261107 \\
        --lock config/data_lock_hira_evaluation_v2.json \\
        --maps config/near_obstacle_maps_v1.json --num-envs 8

    python tests/run_delivery.py ... --preflight-only   # 只核验不启动（用于验证与演练）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
sys.path.insert(0, str(REPO / "CaMRA" / "calibration" / "src"))

from camra_calibration.lock import DataLock  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_binding(run_dir: Path, *, split: str, lock_sha: str, bundle_id: str,
                  manifest_sha: str) -> tuple[bool, list[str], str]:
    """返回 (是否允许, 不符项, 说明)。"""
    cfg_path = run_dir / "collect_config.json"
    if not run_dir.is_dir():
        return True, [], "run 目录不存在：首次采集"
    if not cfg_path.is_file():
        return True, [], "run 目录存在但无 collect_config：视为中断残留，允许继续（不算不符）"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"collect_config 无法解析：{exc}"], "配置损坏"
    bad: list[str] = []
    if cfg.get("split") != split:
        bad.append(f"split: 目录绑定 {cfg.get('split')!r} ≠ 本次 {split!r}")
    got_lock = (cfg.get("data_lock") or {}).get("sha256")
    if got_lock != lock_sha:
        bad.append(f"data_lock.sha256: 目录 {str(got_lock)[:16]} ≠ 本次 {lock_sha[:16]}")
    if cfg.get("bundle_id") and bundle_id and cfg["bundle_id"] != bundle_id:
        bad.append(f"bundle_id: 目录 {cfg['bundle_id'][:16]} ≠ 本次 {bundle_id[:16]}")
    if cfg.get("manifest_sha256") and manifest_sha and cfg["manifest_sha256"] != manifest_sha:
        bad.append("manifest_sha256: 目录记录的清单与本次不符")
    return (not bad), bad, ("绑定一致" if not bad else "绑定不符")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
    ap.add_argument("--lock", default=str(EXP / "config/data_lock_hira_calibration_v1.json"))
    ap.add_argument("--manifests", default=str(EXP / "manifests"))
    ap.add_argument("--bundle", default=str(EXP / "frozen/hira_go2_18500"))
    ap.add_argument("--preflight-only", action="store_true")
    known, passthrough = ap.parse_known_args(raw)

    run_dir = Path(known.runs_root) / known.run_id
    lock = DataLock.load(Path(known.lock), verify_hash=True)
    manifest_sha = sha256_file(Path(known.manifests) / f"{known.split}.jsonl")
    bundle_id = ""
    man_p = Path(known.bundle) / "manifest.json"
    if man_p.is_file():
        bundle_id = json.loads(man_p.read_text(encoding="utf-8")).get("bundle_id", "")

    ok, bad, why = check_binding(run_dir, split=known.split, lock_sha=lock.sha256,
                                 bundle_id=bundle_id, manifest_sha=manifest_sha)
    print(f"[delivery] run_dir={run_dir} lock={Path(known.lock).name} sha={lock.sha256[:16]} "
          f"-> {why}", flush=True)
    if not ok:
        print("[delivery] 拒绝启动：输出目录与本次调用不符（防止把数据写进别的 run）", flush=True)
        for b in bad:
            print("  -", b, flush=True)
        print("[delivery] 请换一个 --run-id / --runs-root，或先人工确认并移走该目录", flush=True)
        return 2
    if known.preflight_only:
        print("[delivery] preflight OK（未启动采集）", flush=True)
        return 0

    passthrough = [a for a in raw if a != "--preflight-only"]
    cmd = [sys.executable, str(EXP / "tests" / "run_split.py"), *passthrough]
    print(f"[delivery] exec: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(EXP), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
