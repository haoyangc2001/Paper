"""HIRA-038：构建可复现交付包索引（delivery/bundle_manifest_hira_v1.json）。

设计原则
--------
1. **只索引、不复制大数据**：1830 个 trial（约 2.1 GB）留在原处，索引里给"逻辑相对路径 +
   SHA256 + 取用说明"，随 Git 走的是代码、配置、冻结物、清单、报告。
2. **角色分区**：按 代码 / 契约 / 协议 / 预登记 / 功率 / 轮廓 / 裕度 / 锁 / 清单 / 报告
   分块，每块可单独核对，不堆成一个扁平列表。
3. **树散列口径固定**：目录级摘要用 `sha256_tree`（对 `相对路径\\0文件散列\\n` 排序后整体散列），
   任何人可用 `--verify` 重算复现。
4. **不打包凭证/机器私密配置**：只登记 `hira.env`（本机样例），不登记 SSH 密钥、令牌、
   `~/.ssh`、`fakehome` 内容。

用法
----
    python tests/build_delivery_manifest.py                # 生成
    python tests/build_delivery_manifest.py --skip-raw     # 跳过大数据散列（快速重建）
    python tests/build_delivery_manifest.py --verify PATH  # 按已有索引逐项复核
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2"
SRC = REPO / "CaMRA" / "calibration" / "src" / "camra_calibration"
SCHEMA = "camra-hira-delivery-manifest-v1"

# 验收报告会记录本索引的散列；若再把它收进索引，就成了散列自指（改谁都要改另一个）。
# 因此明确排除，并在 manifest 里登记原因。
INDEX_EXCLUDED = {
    "reports/final_acceptance.json":
        "该报告记录本索引的 sha256；收录会形成散列自指，故排除（文件本身仍在仓库内）",
}


# --------------------------------------------------------------------------- #
# 散列
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path, *, skip_names=("__pycache__",)) -> tuple[str, int, int]:
    """目录摘要：对 (相对路径, 文件散列) 排序后整体散列。返回 (摘要, 文件数, 字节数)。"""
    root = Path(root)
    entries: list[tuple[str, str]] = []
    n_files = 0
    n_bytes = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in rel.split("/") for part in skip_names):
            continue
        if p.suffix == ".pyc":
            continue
        entries.append((rel, sha256_file(p)))
        n_files += 1
        n_bytes += p.stat().st_size
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return h.hexdigest(), n_files, n_bytes


def rel_to_repo(path: Path) -> str:
    """仓库内相对路径（换机时按同样相对布局复原）。"""
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


# --------------------------------------------------------------------------- #
# 采集条目
# --------------------------------------------------------------------------- #
def entry(path: Path, *, role: str, note: str = "", **extra) -> dict:
    path = Path(path)
    ok = path.is_file()
    item = {
        "logical_path": rel_to_repo(path),
        "role": role,
        "bundled": True,
        "exists": ok,
        "sha256": sha256_file(path) if ok else None,
        "bytes": path.stat().st_size if ok else None,
    }
    if note:
        item["note"] = note
    item.update(extra)
    return item


def dir_items(base: Path, pattern: str, *, role: str, note: str = "") -> list[dict]:
    out = []
    for p in sorted(Path(base).glob(pattern)):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out.append(entry(p, role=role, note=note))
    return out


# --------------------------------------------------------------------------- #
# Git / 环境
# --------------------------------------------------------------------------- #
def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_identity() -> dict:
    status = git("status", "--porcelain")
    dirty = [ln[3:] for ln in status.splitlines() if ln.strip()]
    patch = git("diff", "HEAD")
    return {
        "commit": git("rev-parse", "HEAD"),
        "commit_date": git("log", "-1", "--format=%cI"),
        "commit_subject": git("log", "-1", "--format=%s"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": dirty,
        "dirty": bool(dirty),
        "diff_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else None,
        "diff_bytes": len(patch.encode("utf-8")),
        "note": "随包走的是 HEAD 提交；若 dirty=true，diff_sha256 覆盖工作区未提交改动",
    }


def env_identity() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        info["numpy"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        info["torch"] = None
    return info


# --------------------------------------------------------------------------- #
# 大数据索引
# --------------------------------------------------------------------------- #
def run_index(run_dir: Path, logical_root: str) -> dict:
    """一个 run 目录的索引：逐 trial 树散列 + 汇总。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {"logical_path": f"{logical_root}/{run_dir.name}", "exists": False}
    trials: dict[str, dict] = {}
    n_files = 0
    n_bytes = 0
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        digest, nf, nb = sha256_tree(d)
        trials[d.name] = {"sha256_tree": digest, "n_files": nf, "bytes": nb}
        n_files += nf
        n_bytes += nb
    top = {}
    for p in sorted(run_dir.glob("*")):
        if p.is_file():
            top[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {
        "logical_path": f"{logical_root}/{run_dir.name}",
        "exists": True,
        "n_trials": len(trials),
        "n_files": n_files,
        "bytes": n_bytes,
        "top_level_files": top,
        "trials": trials,
    }


def build_big_data(raw_root: Path, usd_cache: Path, skip_raw: bool) -> dict:
    """大数据：不打包，只索引；给取用说明与可用性状态。"""
    root = Path(raw_root)
    runs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []
    logical_root = str(root)
    out: dict = {
        "bundled": False,
        "reason": "体积过大（约 2.1 GB / 1830 个 trial）；按逻辑路径 + 散列索引，取用时按同样相对布局挂载",
        "logical_root": logical_root,
        "hash_rule": ("trial 级：sha256 over 排序后的 '相对路径\\0文件散列\\n'；"
                      "顶层文件：直接对文件字节散列"),
        "access": [
            f"挂载/复制到 {logical_root}（脚本默认 --runs-root 即此路径）",
            "或改写 config/hira.env 的 CAMRA_HIRA_DATA_ROOT 后在命令里显式给 --runs-root",
            "复核：python tests/build_delivery_manifest.py --verify delivery/bundle_manifest_hira_v1.json",
        ],
        "runs": {},
    }
    if skip_raw:
        out["runs_skipped"] = True
        out["runs_note"] = "本次 --skip-raw，未重算散列（沿用索引里的既有值）"
        return out
    t0 = time.time()
    for d in runs:
        print(f"[raw] hashing {d.name} ...", flush=True)
        out["runs"][d.name] = run_index(d, logical_root)
    out["n_runs"] = len(out["runs"])
    out["n_trials"] = sum(r.get("n_trials", 0) for r in out["runs"].values())
    out["bytes"] = sum(r.get("bytes", 0) for r in out["runs"].values())
    out["hash_seconds"] = round(time.time() - t0, 1)
    # USD 缓存：换机可重建，故只记状态与摘要
    usd = Path(usd_cache)
    if usd.is_dir():
        digest, nf, nb = sha256_tree(usd)
        out["usd_cache"] = {"logical_path": str(usd), "exists": True, "sha256_tree": digest,
                            "n_files": nf, "bytes": nb,
                            "note": "从 URDF 转换得到；换机可由 config/asset 流程重建，非唯一证据"}
    else:
        out["usd_cache"] = {"logical_path": str(usd), "exists": False}
    return out


# --------------------------------------------------------------------------- #
# 主构建
# --------------------------------------------------------------------------- #
def build(args) -> dict:
    cfg = EXP / "config"
    rep = EXP / "reports"
    man = EXP / "manifests"
    fz = EXP / "frozen" / "hira_go2_18500"

    artifacts: dict[str, list[dict]] = {}

    # 1. 代码（决定"加载的是哪个系统"）
    artifacts["code_adapter"] = dir_items(ADAPTER, "*.py", role="code_adapter",
                                          note="HiRA→IsaacLab 适配层；均在检验锁 code.files 内")
    artifacts["code_src"] = dir_items(SRC, "*.py", role="code_src",
                                      note="校准库（锁/清单/能量/裕度/覆盖/几何）")
    artifacts["code_tests"] = dir_items(EXP / "tests", "*.py", role="code_tests",
                                        note="只 run_split.py 在锁内；其余为检查/评估脚本")
    artifacts["reference_fixtures"] = dir_items(EXP / "tests" / "reference_fixtures", "*",
                                                role="reference_fixtures",
                                                note="9 个数值参照 fixture，用于重载 parity")

    # 2. 契约与协议（按角色分，不混成"config 一堆"）
    contract = ["hira_source_contract_v1.json", "go2_asset_geometry_hira_v1.json"]
    protocol = ["calibration_protocol_hira_v1.json"]
    prereg = ["integration_acceptance_prereg_v1.json", "integration_acceptance_amendment_t08_v1.json"]
    power = ["power_model_prereg_hira_v1.json", "power_model_hira_v1.json"]
    shape = ["footprint_hira_v1.json", "step_cost_schema_hira_v1.json"]
    margins = ["margins_hira_v1.json"]
    maps = ["near_obstacle_maps_v1.json"]

    def pick(names: list[str], role: str, note: str) -> list[dict]:
        items = []
        for name in names:
            for p in (cfg / name, cfg / f"{name}.sha256"):
                if p.is_file():
                    items.append(entry(p, role=role, note=note))
        return items

    artifacts["contract"] = pick(contract, "contract", "来源身份与资产几何（HIRA-007/012）")
    artifacts["protocol"] = pick(protocol, "protocol", "校准协议冻结版（HIRA-023）")
    artifacts["preregistration"] = pick(prereg, "preregistration", "验收预登记（结果前冻结）")
    artifacts["power_model"] = pick(power, "power_model", "功率模型预登记 + 十参数冻结模型")
    artifacts["geometry_schema"] = pick(shape, "geometry_schema", "足迹与步成本接口")
    artifacts["margins"] = pick(margins, "margins", "α/δ 冻结值（唯一权威副本）")
    artifacts["near_obstacle_maps"] = pick(maps, "near_obstacle_maps", "200 例冻结近障地图")
    # 锁按通配收录，新增版本不会漏（v1/v2/v3 各自对应不同的数据归属）
    artifacts["stage_locks"] = dir_items(cfg, "data_lock_hira_*.json*", role="stage_locks",
                                         note="全部采集锁与检验锁：锁只增不改；heldout 归 v1、near 归 v2、当前为 v3")
    artifacts["env_sample"] = [entry(cfg / "hira.env", role="env_sample",
                                     note="本机路径样例；换机只改此文件里的根路径")]

    # 3. 冻结控制器包
    artifacts["bundle"] = dir_items(fz, "*", role="bundle",
                                    note="bundle_id 3c32fe31…；policy.pt 身份用 actor tensor digest")

    # 4. 七清单
    artifacts["manifests"] = dir_items(man, "*", role="manifests",
                                       note="七个 split 的预登记清单 + 清单锁")

    # 5. 全部报告（科学结论的唯一出处）
    artifacts["reports"] = [
        it for it in dir_items(rep, "*", role="reports",
                               note="本任务全部判决性产物；均可在报告内定位判据与分母")
        if f"reports/{Path(it['logical_path']).name}" not in INDEX_EXCLUDED
    ]

    # 6. 文档
    docs = []
    for name in ("implementation_handoff.md", "reproduction.md"):
        p = EXP / "docs" / name
        docs.append(entry(p, role="docs", note="交接与复现说明"))
    artifacts["docs"] = docs

    # 汇总
    flat = [it for group in artifacts.values() for it in group]
    missing = [it["logical_path"] for it in flat if not it["exists"]]
    # 自校验：构建时立刻回读重算一遍（不能只信刚才写进索引的值）
    n_checked = 0
    n_bad = 0
    for it in flat:
        if not it["exists"]:
            continue
        if sha256_file(REPO / it["logical_path"]) == it["sha256"]:
            n_checked += 1
        else:
            n_bad += 1

    manifest = {
        "schema": SCHEMA,
        "kind": "hira_policy_calibration_delivery_bundle",
        "stage": "delivery",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git": git_identity(),
        "environment": env_identity(),
        "layout": {
            "repo_relative": True,
            "trees": [
                {"role": "code_adapter", "path": rel_to_repo(ADAPTER)},
                {"role": "code_src", "path": rel_to_repo(SRC)},
                {"role": "experiment", "path": rel_to_repo(EXP)},
            ],
            "path_rule": (
                "所有脚本用 __file__ 自定位（EXP = tests/../ ，REPO = EXP 上四级），"
                "因此把这三棵树按同样相对布局复制到新根目录后，源码零改动即可运行；"
                "唯一需要重定位的是 stage lock 内的绝对路径——用 tests/delivery_relock.py 生成派生锁。"
            ),
        },
        "roles": {
            "code_adapter": "HiRA→IsaacLab 适配层（含 obstacles.py）",
            "code_src": "校准库",
            "code_tests": "检查/评估/审计脚本",
            "reference_fixtures": "数值参照（9 fixture）",
            "contract": "来源身份 + 资产几何",
            "protocol": "校准协议（冻结）",
            "preregistration": "验收预登记（结果前冻结）",
            "power_model": "功率模型预登记与冻结参数",
            "geometry_schema": "足迹与步成本接口",
            "margins": "α/δ 冻结值",
            "near_obstacle_maps": "200 例冻结近障地图",
            "stage_locks": "采集锁与检验锁（只增不改）",
            "env_sample": "本机路径样例",
            "bundle": "冻结控制器包",
            "manifests": "七个 split 清单 + 清单锁",
            "reports": "全部判决性报告",
            "docs": "交接笔记与复现说明",
        },
        "artifacts": artifacts,
        "counts": {
            "groups": len(artifacts),
            "files": len(flat),
            "bytes": sum(it["bytes"] or 0 for it in flat),
            "missing": len(missing),
        },
        "big_data": (json.loads(Path(args.reuse_raw).read_text(encoding="utf-8"))["big_data"]
                     if args.reuse_raw else
                     build_big_data(Path(args.raw_runs_root), Path(args.usd_cache), args.skip_raw)),
        "external_dependencies": [
            {"name": "HiRA 训练仓库", "path": "/home/caohy/repositories/Paper/HiRA",
             "role": "source", "bundled": False,
             "note": "只用于来源语义核对；训练已完成，复现校准不需要它",
             "availability": "本机存在" if Path("/home/caohy/repositories/Paper/HiRA").is_dir() else "缺失"},
            {"name": "原始 checkpoint model_18500.pt",
             "path": "/home/caohy/repositories/Paper/HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt",
             "role": "source", "bundled": False,
             "note": "policy.pt 由它导出；复现校准不需要重导出",
             "availability": "本机存在" if Path(
                 "/home/caohy/repositories/Paper/HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt"
             ).is_file() else "缺失"},
            {"name": "Isaac Lab 环境", "path": "/pub/data/caohy/miniconda/envs/env_isaaclab",
             "role": "runtime", "bundled": False, "note": "采集需要；离线复算只需要 numpy"},
        ],
        "not_bundled": [
            {"what": "稿件 manuscripts/**、*.tex、*.bib、投稿 PDF",
             "why": "红线：本任务不得改动或打包稿件；差异以清单形式交由本人执行"},
            {"what": "旧 v2 的逐 trial 原始数据（68 GB）",
             "why": "已不在本机（四处核实），因此与 v2 只能作报告级对比，见 reports/hira_vs_v2_comparison.md"},
            {"what": "SSH 密钥、令牌、fakehome、机器私密配置",
             "why": "凭证与私密配置不入包；hira.env 只保留路径样例"},
        ],
        "verification": {
            "hash_rule": "文件：sha256(bytes)；目录：sha256（排序后的 '相对路径\\0文件散列\\n'）",
            "entries_checked": n_checked,
            "entries_bad": n_bad,
            "raw_hashed": not args.skip_raw,
            "reload_report": "reports/delivery_reload_verification.json",
            "reproduce_doc": "docs/reproduction.md",
            "note": ("entries_checked 为构建时逐项回读重算的结果（不是复用刚写入的值）；"
                     "reload 报告由验证流程在独立目录生成，其散列已收录在本索引内"),
        },
        "excluded_from_index": INDEX_EXCLUDED,
        "missing_artifacts": missing,
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP / "delivery" / "bundle_manifest_hira_v1.json"))
    ap.add_argument("--raw-runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
    ap.add_argument("--usd-cache", default="/pub/data/caohy/logs/camra_hira/assets/usd_cache")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--reuse-raw", default=None,
                    help="沿用既有索引里的 big_data（大数据只散列一次；用于提交后刷新 git 身份）")
    ap.add_argument("--verify", default=None, help="按已有索引逐项复核")
    args = ap.parse_args()

    if args.verify:
        return verify(Path(args.verify))

    manifest = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    side = out.with_suffix(out.suffix + ".sha256")
    side.write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    print(f"written: {out} ({out.stat().st_size} bytes)")
    print(f"groups={manifest['counts']['groups']} files={manifest['counts']['files']} "
          f"bytes={manifest['counts']['bytes']} missing={manifest['counts']['missing']}")
    if manifest["missing_artifacts"]:
        print("MISSING:", manifest["missing_artifacts"])
        return 1
    print("MANIFEST OK")
    return 0


def verify(manifest_path: Path) -> int:
    """按索引逐项复核：文件散列 + 大数据 trial 树散列。"""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    bad: list[str] = []
    n_ok = 0
    for group, items in payload["artifacts"].items():
        for it in items:
            p = REPO / it["logical_path"]
            if not p.is_file():
                bad.append(f"{group}: 缺失 {it['logical_path']}")
                continue
            if sha256_file(p) != it["sha256"]:
                bad.append(f"{group}: 散列不符 {it['logical_path']}")
                continue
            n_ok += 1
    raw = payload.get("big_data", {})
    n_raw = 0
    for name, run in (raw.get("runs") or {}).items():
        if not run.get("exists"):
            bad.append(f"raw: run 缺失 {name}")
            continue
        root = Path(raw["logical_root"]) / name
        for trial, info in run.get("trials", {}).items():
            digest, nf, nb = sha256_tree(root / trial)
            if digest != info["sha256_tree"]:
                bad.append(f"raw: trial 散列不符 {name}/{trial}")
            else:
                n_raw += 1
    print(f"files ok={n_ok} raw_trials ok={n_raw} bad={len(bad)}")
    for b in bad[:20]:
        print("  -", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
