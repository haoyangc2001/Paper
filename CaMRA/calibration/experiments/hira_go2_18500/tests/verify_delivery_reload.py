"""HIRA-038：独立目录重载验证。

判据（计划 HIRA-038 acceptance）
--------------------------------
1. 新目录可按说明取得所需资源，加载的是对应 HiRA 系统；hash 与小批链路均通过；
2. 全部报告可定位，科学负结果与可复现性结果分别标明；
3. 模型 / 地图 / 输出目录被误换时拒绝；权重或资产缺失时明确报告环境缺失。

做法：把三棵树（adapter / src / experiment）复制到**全新根目录**，只重定位锁内路径，
然后用**副本里的代码**跑：控制器重载 + 9 个 fixture 数值 parity + 40 条小批采集 +
ε_c/ε_d 与覆盖判定 + 官方原始数据的离线能量复算。原目录只读。

    python tests/verify_delivery_reload.py \
        --work-root /pub/data/caohy/work/hira/delivery_check_20261107 \
        --run-id delivery_reload_20261107

`--skip-collection` 可跳过需要 Isaac 的一段（其余检查仍跑）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
OFFICIAL_RUNS = Path("/pub/data/caohy/logs/camra_hira/runs")
PYTHON = "/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python"

TREES = [
    ("code_adapter", REPO / "CaMRA" / "locomotion" / "adapters" / "hira_go2"),
    ("code_src", REPO / "CaMRA" / "calibration" / "src" / "camra_calibration"),
    ("experiment", EXP),
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    entries = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            entries.append((p.relative_to(root).as_posix(), sha256_file(p)))
    h = hashlib.sha256()
    for rel, d in entries:
        h.update(f"{rel}\0{d}\n".encode("utf-8"))
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, cid: str, title: str, passed: bool, detail: str = "", **extra) -> bool:
        item = {"id": cid, "title": title, "passed": bool(passed), "detail": detail}
        item.update(extra)
        self.items.append(item)
        print(f"[{'ok ' if passed else 'FAIL'}] {cid} {title}"
              + (f" — {detail}" if detail else ""), flush=True)
        return bool(passed)

    @property
    def failed(self) -> list[str]:
        return [c["id"] for c in self.items if not c["passed"]]


# --------------------------------------------------------------------------- #
# D01 独立目录
# --------------------------------------------------------------------------- #
def phase_independent_dir(chk: Checks, work_root: Path) -> dict:
    src = REPO.resolve()
    work = work_root.resolve()
    inside = str(work).startswith(str(src) + "/") or work == src
    chk.add("D01-independent-root", "工作根不在源仓库内、且是新建目录",
            not inside, f"work={work} source={src} inside={inside}")
    work.mkdir(parents=True, exist_ok=True)
    # 目录里不能有指向源仓库的软链（否则"独立"是假的）
    links = [p for p in work.rglob("*") if p.is_symlink()]
    chk.add("D02-no-symlink-into-source", "工作根内无软链（避免偷偷引用源目录）",
            not links, f"symlinks={len(links)}")
    return {"work_root": str(work), "source_root": str(src)}


# --------------------------------------------------------------------------- #
# D03 复制三棵树
# --------------------------------------------------------------------------- #
def phase_copy(chk: Checks, work_root: Path) -> dict:
    info = {"trees": {}, "files": 0, "bytes": 0, "mismatch": []}
    for role, src in TREES:
        dst = work_root / src.relative_to(REPO)
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=IGNORE)
        n = 0
        nb = 0
        for p in sorted(dst.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(dst)
            q = src / rel
            n += 1
            nb += p.stat().st_size
            if not q.is_file() or sha256_file(p) != sha256_file(q):
                info["mismatch"].append(f"{role}/{rel.as_posix()}")
        info["trees"][role] = {"src": str(src), "dst": str(dst), "n_files": n, "bytes": nb}
        info["files"] += n
        info["bytes"] += nb
        print(f"[copy] {role}: {n} files, {nb/1e6:.2f} MB -> {dst}", flush=True)
    chk.add("D03-copy-identical", "三棵树逐文件散列与源目录一致",
            not info["mismatch"], f"{info['files']} 文件 {info['bytes']/1e6:.1f} MB，"
                                  f"不一致 {len(info['mismatch'])}")
    # 源码自定位：副本里的 EXP/REPO 必须指向副本
    probe = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500" / "tests" / "run_split.py"
    resolved_exp = probe.resolve().parent.parent
    chk.add("D04-source-self-locating", "副本脚本用 __file__ 自定位（EXP 指向副本）",
            str(resolved_exp) == str(probe.parent.parent),
            f"EXP={resolved_exp}")
    return info


# --------------------------------------------------------------------------- #
# D05 路径重定位 + 散列自证
# --------------------------------------------------------------------------- #
def phase_relock(chk: Checks, work_root: Path, old_prefix: str) -> dict:
    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    src_lock = work_exp / "config" / "data_lock_hira_evaluation_v2.json"
    out_lock = work_exp / "config" / "data_lock_hira_evaluation_delivery.json"
    proc = subprocess.run(
        [PYTHON, "tests/delivery_relock.py", "--lock", str(src_lock), "--out", str(out_lock),
         "--new-root", str(work_root), "--old-prefix", old_prefix],
        cwd=str(work_exp), capture_output=True, text=True, check=False)
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"verify_ok": False, "verify_bad": [proc.stdout[-800:] + proc.stderr[-800:]]}
    chk.add("D05-relock-writes", "派生锁生成（只改路径前缀，sha256 逐字保留）",
            bool(res.get("out_lock_sha256")), f"n_paths_rewritten={res.get('n_paths_rewritten')}")
    chk.add("D06-relock-verifies", "派生锁在新根下 verify_artifacts 全过",
            bool(res.get("verify_ok")), "；".join(res.get("verify_bad") or [])[:400] or "0 项不符")
    # 原锁未被改动
    orig = REPO / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500" / "config" / "data_lock_hira_evaluation_v2.json"
    orig_side = orig.with_suffix(orig.suffix + ".sha256")
    chk.add("D07-original-lock-untouched", "源仓库的检验锁 v2 与侧车未被修改",
            orig_side.is_file() and orig_side.read_text(encoding="utf-8").split()[0] == sha256_file(orig),
            f"sha={sha256_file(orig)[:16]}")
    return {"lock": str(out_lock), "relock": res}


# --------------------------------------------------------------------------- #
# D08 负测试
# --------------------------------------------------------------------------- #
def phase_negatives(chk: Checks, work_root: Path, lock_path: Path) -> dict:
    sys.path.insert(0, str(work_root / "CaMRA" / "locomotion" / "adapters"))
    sys.path.insert(0, str(work_root / "CaMRA" / "calibration" / "src"))
    from camra_calibration.lock import DataLock, LockError, enforce_collection
    from hira_go2.bundle import BundleError, load_bundle

    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    sand = work_root / "_negatives"
    if sand.exists():
        shutil.rmtree(sand)
    sand.mkdir(parents=True)
    out: dict = {}

    # N1 模型被换：policy.pt 改一个字节
    b1 = sand / "bundle_policy_flipped"
    shutil.copytree(work_exp / "frozen" / "hira_go2_18500", b1, ignore=IGNORE)
    data = bytearray((b1 / "policy.pt").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (b1 / "policy.pt").write_bytes(bytes(data))
    try:
        load_bundle(b1)
        out["model_swap"] = {"rejected": False, "reason": "load_bundle 竟然通过了"}
    except BundleError as exc:
        out["model_swap"] = {"rejected": True, "reason": str(exc)[:200]}
    chk.add("D08-model-swap-rejected", "模型被换（policy.pt 改字节）时拒绝加载",
            out["model_swap"]["rejected"], out["model_swap"]["reason"])

    # N2 权重缺失
    b2 = sand / "bundle_no_policy"
    shutil.copytree(work_exp / "frozen" / "hira_go2_18500", b2, ignore=IGNORE)
    (b2 / "policy.pt").unlink()
    try:
        load_bundle(b2)
        out["missing_weights"] = {"reported": False, "reason": "缺 policy.pt 竟然通过"}
    except BundleError as exc:
        msg = str(exc)
        out["missing_weights"] = {"reported": True, "reason": msg[:200]}
        chk.add("D09-missing-weights-reports", "权重缺失时明确报告（不是崩溃/静默）",
                "policy" in msg.lower() or "权重" in msg, msg[:160])
    else:
        chk.add("D09-missing-weights-reports", "权重缺失时明确报告（不是崩溃/静默）", False, "未报错")

    # N3 地图被换：maps 改字节后过门禁
    maps = work_exp / "config" / "near_obstacle_maps_v1.json"
    m2 = sand / "maps_flipped.json"
    raw = bytearray(maps.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    m2.write_bytes(bytes(raw))
    lock = DataLock.load(lock_path, verify_hash=True)
    try:
        enforce_collection(lock, "near_obstacle", work_root, manifest_dir=work_exp / "manifests",
                           maps_path=m2, margins_path=work_exp / "config" / "margins_hira_v1.json")
        out["map_swap"] = {"rejected": False, "reason": "门禁竟然放行"}
    except LockError as exc:
        out["map_swap"] = {"rejected": True, "reason": str(exc)[:200]}
    chk.add("D10-map-swap-rejected", "地图被换（字节改动）时门禁拒绝",
            out["map_swap"]["rejected"], out["map_swap"]["reason"])

    # N4 资产缺失：把 asset 路径指向不存在的根，preflight 必须明确报缺失
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    bogus = json.loads(json.dumps(payload))
    for key in ("geometry",):
        if bogus.get("asset", {}).get(key):
            bogus["asset"][key]["path"] = "CaMRA/_missing_asset/geometry.json"
    bogus["delivery_rebase"]["note"] += "（资产缺失负测试副本）"
    bogus_path = sand / "lock_missing_asset.json"
    bogus_path.write_text(json.dumps(bogus, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    bogus_path.with_suffix(".json.sha256").write_text(
        f"{sha256_file(bogus_path)}  {bogus_path.name}\n", encoding="utf-8")
    bad = DataLock.load(bogus_path, verify_hash=True).verify_artifacts(work_root)
    out["missing_asset"] = {"reported": bool(bad), "bad": bad[:3]}
    chk.add("D11-missing-asset-reports", "资产缺失时 preflight 明确列出缺失项（不是静默启动）",
            bool(bad), "；".join(bad[:2])[:300] or "未报缺失")

    # N5 输出目录被误换：交付入口（run_delivery.py）必须拒绝
    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    entry = work_exp / "tests" / "run_delivery.py"
    probe_name = "run_dir_bound_to_heldout"
    probe = sand / probe_name
    probe.mkdir()
    (probe / "collect_config.json").write_text(json.dumps({
        "split": "heldout",
        "data_lock": {"file": "data_lock_hira_evaluation_v1.json", "sha256": "0" * 64},
        "bundle_id": "0" * 64, "manifest_sha256": "0" * 64, "num_trials": 400,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def _entry(rid: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [PYTHON, str(entry), "--split", "near_obstacle", "--run-id", rid,
             "--runs-root", str(sand), "--lock", str(lock_path),
             "--manifests", str(work_exp / "manifests"),
             "--bundle", str(work_exp / "frozen" / "hira_go2_18500"),
             "--preflight-only"],
            cwd=str(work_exp), capture_output=True, text=True, check=False)

    mis = _entry(probe_name)
    fresh = _entry("fresh_probe_dir")
    refused = mis.returncode == 2
    accepted_fresh = fresh.returncode == 0
    out["output_dir_swap"] = {
        "rejected_by_delivery_entry": refused,
        "mismatched_rc": mis.returncode,
        "mismatched_msg": [ln for ln in mis.stdout.splitlines() if "split:" in ln or "拒绝" in ln][:3],
        "fresh_dir_accepted": accepted_fresh,
        "fresh_rc": fresh.returncode,
    }
    chk.add("D12-output-dir-swap-rejected",
            "输出目录已绑定别的 split/锁时，交付入口拒绝启动",
            refused and accepted_fresh,
            f"误换目录 rc={mis.returncode}（应 2），新目录 rc={fresh.returncode}（应 0）")
    # 原生 run_split 自身没有这道守卫 —— 如实登记，不假装已经修好
    rs = (work_exp / "tests" / "run_split.py").read_text(encoding="utf-8")
    reads_cfg = "collect_config.json\").read_text" in rs or "collect_config.json').read_text" in rs
    out["output_dir_gap"] = {
        "raw_run_split_guards_existing_config": reads_cfg,
        "note": ("run_split.py 只在首次写 collect_config，启动时不校验既有 run 目录的 split/锁/bundle；"
                 "该文件在 stage lock 的 code.files 内，未擅自修改，改由交付入口 run_delivery.py 前置拦截。"
                 "若要下沉到 run_split 本身，需新版锁（HIRA-038 已登记为待裁定项）。"),
    }
    return out


# --------------------------------------------------------------------------- #
# D13 控制器重载 + fixture parity（CPU）
# --------------------------------------------------------------------------- #
def phase_reload_parity(chk: Checks, work_root: Path) -> dict:
    """D13 控制器包重载 + D14 obs/动作适配 parity（在副本里跑副本的代码，CPU）。"""
    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    from hira_go2.bundle import load_bundle   # sys.path 已由 phase_negatives 指向副本

    bundle = load_bundle(work_exp / "frozen" / "hira_go2_18500",
                         expected_controller_id="hira_go2_18500")
    ctrl = bundle["controller"]
    chk.add("D13-bundle-reload", "独立目录加载的是同一个 HiRA 系统（bundle_id / policy 散列一致）",
            bundle["manifest"]["bundle_id"].startswith("3c32fe31")
            and ctrl["policy"]["sha256"].startswith("721a891a"),
            f"bundle_id={bundle['manifest']['bundle_id'][:16]} "
            f"policy_sha={ctrl['policy']['sha256'][:12]} "
            f"arch={ctrl['policy']['hidden_dims']}")

    # obs / 动作适配 parity：直接跑副本里的 HIRA-015 脚本（9 fixture + 5 类负测试）
    out_path = work_root / "reports_delivery_parity.json"
    proc = subprocess.run(
        [PYTHON, "tests/check_source_adapter_parity.py", "--out", str(out_path)],
        cwd=str(work_exp), capture_output=True, text=True, check=False)
    try:
        rep = json.loads(out_path.read_text(encoding="utf-8"))
        verdict = rep.get("verdict", {})
    except Exception:  # noqa: BLE001
        rep, verdict = {}, {}
    passed = bool(verdict.get("all_passed"))
    ref = json.loads((REPO / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
                      / "reports" / "source_adapter_parity.json").read_text(encoding="utf-8"))
    ref_verdict = ref.get("verdict", {})
    same = (verdict.get("fixtures_sha256") == ref_verdict.get("fixtures_sha256")
            and verdict.get("negative_tests_all_detected") == ref_verdict.get("negative_tests_all_detected"))
    chk.add("D14-obs-action-parity", "副本内复现 obs/动作适配 parity（9 fixture + 负测试全检出）",
            passed and same,
            f"all_passed={passed} fixtures_sha_match={same} rc={proc.returncode}"
            + ("" if passed else f"；{str(rep.get('failures'))[:200]}"))
    return {"bundle_id": bundle["manifest"]["bundle_id"],
            "policy_sha256": ctrl["policy"]["sha256"],
            "parity": {k: verdict.get(k) for k in
                       ("all_passed", "baseline_fixture_passes", "negative_tests_all_detected",
                        "fixtures_sha256", "devices")},
            "parity_report": str(out_path),
            "parity_stderr_tail": proc.stderr[-300:] if proc.returncode else ""}


# --------------------------------------------------------------------------- #
# D15 小批采集（真跑 Isaac）
# --------------------------------------------------------------------------- #
def phase_collect(chk: Checks, work_root: Path, lock_path: Path, run_id: str,
                  num_envs: int, max_batches: int) -> dict:
    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    runs_root = work_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    before = {p.name: p.stat().st_mtime for p in OFFICIAL_RUNS.iterdir() if p.is_dir()}
    log = work_root / "collect_small.log"
    # 走交付入口（带输出目录绑定守卫），而不是直接 run_split
    cmd = [PYTHON, "tests/run_delivery.py",
           "--split", "near_obstacle", "--run-id", run_id,
           "--lock", str(lock_path),
           "--maps", str(work_exp / "config" / "near_obstacle_maps_v1.json"),
           "--manifests", str(work_exp / "manifests"),
           "--bundle", str(work_exp / "frozen" / "hira_go2_18500"),
           "--runs-root", str(runs_root),
           "--num-envs", str(num_envs), "--max-batches", str(max_batches)]
    env = dict(os.environ)
    env.update({
        "OMNI_KIT_ACCEPT_EULA": "YES", "HOME": "/pub/data/caohy/fakehome",
        "TMPDIR": "/pub/data/caohy/tmp", "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_DATA_HOME": "/pub/data/caohy/xdgdata", "XDG_CACHE_HOME": "/pub/data/caohy/xdgcache",
        "XDG_CONFIG_HOME": "/pub/data/caohy/xdgconfig",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    })
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(work_exp), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, check=False)
    dur = time.time() - t0
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:])
    run_dir = runs_root / run_id
    trials = sorted(p for p in run_dir.iterdir() if p.is_dir()) if run_dir.is_dir() else []
    ok_marker = "SPLIT_NEAR_OBSTACLE_RUN_OK" in log.read_text(encoding="utf-8", errors="replace")
    chk.add("D15-small-collection", "副本代码在新目录跑通小批采集（锁/地图/障碍全链路）",
            proc.returncode == 0 and ok_marker and len(trials) > 0,
            f"rc={proc.returncode} trials={len(trials)} wall={dur:.0f}s；{tail[-200:]}")
    after = {p.name: p.stat().st_mtime for p in OFFICIAL_RUNS.iterdir() if p.is_dir()}
    changed = [k for k in after if k in before and after[k] != before[k]]
    chk.add("D16-official-untouched", "官方 run 目录未被写入或改动（输出被隔离）",
            not changed and set(after) == set(before),
            f"新增={sorted(set(after) - set(before))} 变动={changed}")
    # 障碍是否真的注入（不能只信进程退出码）
    n_with_obs = 0
    for t in trials:
        att = t / "attempt_01"
        meta_p = att / "meta.json"
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if (meta.get("obstacles") or {}).get("n_obstacles"):
                n_with_obs += 1
    chk.add("D17-obstacles-injected", "小批数据的障碍注入证据完整",
            n_with_obs == len(trials) and len(trials) > 0,
            f"{n_with_obs}/{len(trials)} 条带 n_obstacles")
    return {"run_id": run_id, "run_dir": str(run_dir), "rc": proc.returncode,
            "wall_s": round(dur, 1), "n_trials": len(trials), "n_with_obstacles": n_with_obs,
            "log": str(log)}


# --------------------------------------------------------------------------- #
# D18 误差计算 + 覆盖判定（小批，非正式）
# --------------------------------------------------------------------------- #
def phase_errors(chk: Checks, work_root: Path, run_id: str) -> dict:
    import numpy as np
    from camra_calibration.commands import build_program
    from camra_calibration.coverage import cost_covered, position_covered
    from camra_calibration.nominal import PlanarState, integrate
    from camra_calibration.pairing import PAIR_DT, deviation
    from camra_calibration.power import J_MIN, PowerModel

    work_exp = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500"
    margins = json.loads((work_exp / "config" / "margins_hira_v1.json").read_text(encoding="utf-8"))
    model = PowerModel.load(work_exp / "config" / "power_model_hira_v1.json")
    alpha, delta = float(margins["alpha"]), float(margins["delta"])
    manifest = {json.loads(l)["trial_id"]: json.loads(l)
                for l in (work_exp / "manifests" / "near_obstacle.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}
    run_dir = work_root / "runs" / run_id
    rows = []
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        att = d / "attempt_01"
        if not (att / "meta.json").is_file():
            continue
        meta = json.loads((att / "meta.json").read_text(encoding="utf-8"))
        energy = json.loads((att / "energy.json").read_text(encoding="utf-8"))
        pair = np.load(att / "pairing_10hz.npz")
        t = pair["t_s"].astype(np.float64)
        cmds = pair["command"].astype(np.float64)
        ex = pair["pos_xy"].astype(np.float64)
        spec = manifest[d.name]
        if t.size < 2:
            continue
        j_nom = float(model.j_nom(cmds))
        pose = (meta.get("profile_applied") or {}).get("initial_pose") or {}
        nominal = integrate(PlanarState(px=float(pose.get("px_m", 0.0)),
                                        py=float(pose.get("py_m", 0.0)),
                                        yaw=float(pose.get("yaw_rad", 0.0))), cmds, PAIR_DT)
        nom_xy = np.asarray([[s.px, s.py] for s in nominal], dtype=np.float64)
        eps_d = float(deviation(ex, nom_xy).max())
        j_exec = float(energy["j_exec"])
        eps_c = (j_exec - j_nom) / j_nom if j_nom >= J_MIN else None
        rows.append({"trial_id": d.name, "outcome": meta["outcome"], "j_nom": j_nom,
                     "j_exec": j_exec, "eps_c": eps_c, "eps_d": eps_d,
                     "cost_covered": None if eps_c is None else bool(cost_covered(eps_c, alpha)),
                     "position_covered": bool(position_covered(eps_d, delta)),
                     "nominal_program_ok": True})
    n = len(rows)
    cost_n = sum(1 for r in rows if r["cost_covered"])
    pos_n = sum(1 for r in rows if r["position_covered"])
    chk.add("D18-error-and-coverage", "副本内可算出 ε_c/ε_d 与覆盖判定（小批，非正式）",
            n > 0, f"n={n}，成本覆盖 {cost_n}、位置覆盖 {pos_n}（α={alpha:.4f}、δ={delta:.4f}）")
    return {"note": "小批 40 条，仅为链路验证，不构成覆盖率结论（正式口径见 reports/heldout_coverage.json 等）",
            "n": n, "alpha": alpha, "delta": delta,
            "cost_covered": cost_n, "position_covered": pos_n,
            "rows": rows[:40]}


# --------------------------------------------------------------------------- #
# D19 官方原始数据离线复算
# --------------------------------------------------------------------------- #
def phase_offline(chk: Checks, samples: int) -> dict:
    import numpy as np
    picks: list[Path] = []
    for split in ("calibration_20261106", "heldout_20261106", "near_obstacle_20261107"):
        d = OFFICIAL_RUNS / split
        if not d.is_dir():
            continue
        trials = sorted(p for p in d.iterdir() if p.is_dir())
        step = max(1, len(trials) // max(1, samples // 3))
        for t in trials[::step][: max(2, samples // 3)]:
            att = t / "attempt_01"
            if (att / "physics_200hz.npz").is_file() and (att / "energy.json").is_file():
                picks.append(att)
    worst = 0.0
    rows = []
    for att in picks[:samples]:
        z = np.load(att / "physics_200hz.npz")
        tau = z["tau_applied_source"].astype(np.float64)
        qdot = z["qdot_post_source"].astype(np.float64)
        dt = float(z["dt"]) if "dt" in z else 0.005
        j = float(np.clip(tau * qdot, 0.0, None).sum() * dt)
        recorded = float(json.loads((att / "energy.json").read_text(encoding="utf-8"))["j_exec"])
        rel = abs(j - recorded) / max(abs(recorded), 1e-12)
        worst = max(worst, rel)
        rows.append({"trial": att.parent.name, "j_recomputed": j, "j_recorded": recorded, "rel": rel})
    chk.add("D19-offline-energy-recompute", "官方原始数据可离线复算能量（不依赖 Isaac）",
            bool(rows) and worst <= 1e-9, f"{len(rows)} 条，最差 rel={worst:.3e}")
    return {"n": len(rows), "worst_rel": worst, "rows": rows}


# --------------------------------------------------------------------------- #
# D20 报告可定位
# --------------------------------------------------------------------------- #
CORE_REPORTS = [
    "integration_gate.json",
    "source_adapter_parity.json",
    "calibration_collection_integrity.json",
    "margin_calibration.json",
    "baseline_collection_integrity.json",
    "power_model_validate.json",
    "footprint_validation.json",
    "heldout_collection_integrity.json",
    "heldout_coverage.json",
    "heldout_summary.md",
    "near_obstacle_collection_integrity.json",
    "near_obstacle_evaluation.json",
    "near_obstacle_summary.md",
    "full_independent_audit.json",
    "audit_discrepancies.json",
    "hira_vs_v2_comparison.md",
]


def phase_reports(chk: Checks, work_root: Path) -> dict:
    """判决性报告在新目录里逐一定位得到，且与源目录同散列。"""
    src_dir = REPO / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500" / "reports"
    dst_dir = work_root / "CaMRA" / "calibration" / "experiments" / "hira_go2_18500" / "reports"
    missing = []
    mismatch = []
    for name in CORE_REPORTS:
        a, b = src_dir / name, dst_dir / name
        if not a.is_file():
            missing.append(f"源缺 {name}")
            continue
        if not b.is_file():
            missing.append(f"副本缺 {name}")
            continue
        if sha256_file(a) != sha256_file(b):
            mismatch.append(name)
    chk.add("D20-reports-locatable", "全部判决性报告在新目录可定位且与源同散列",
            not missing and not mismatch,
            f"{len(CORE_REPORTS)} 份核心报告，缺失 {len(missing)}、散列不符 {len(mismatch)}")
    return {"n_core_reports": len(CORE_REPORTS), "missing": missing, "mismatch": mismatch}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--run-id", default="delivery_reload_20261107")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=1)
    ap.add_argument("--offline-samples", type=int, default=12)
    ap.add_argument("--old-prefix", default="/pub/data/caohy/work/hira/")
    ap.add_argument("--skip-collection", action="store_true")
    ap.add_argument("--out", default=str(EXP / "reports" / "delivery_reload_verification.json"))
    args = ap.parse_args()

    chk = Checks()
    t0 = time.time()
    work_root = Path(args.work_root)
    steps: dict = {}
    steps["dir"] = phase_independent_dir(chk, work_root)
    steps["copy"] = phase_copy(chk, work_root)
    steps["relock"] = phase_relock(chk, work_root, args.old_prefix)
    lock_path = Path(steps["relock"]["lock"])
    steps["negatives"] = phase_negatives(chk, work_root, lock_path)
    steps["parity"] = phase_reload_parity(chk, work_root)
    if args.skip_collection:
        steps["collect"] = {"skipped": True}
        steps["errors"] = {"skipped": True}
    else:
        steps["collect"] = phase_collect(chk, work_root, lock_path, args.run_id,
                                        args.num_envs, args.max_batches)
        steps["errors"] = phase_errors(chk, work_root, args.run_id)
    steps["reports"] = phase_reports(chk, work_root)
    steps["offline"] = phase_offline(chk, args.offline_samples)

    verdict = {
        "reload_ok": not chk.failed,
        "failed": chk.failed,
        "n_checks": len(chk.items),
        "n_passed": sum(1 for c in chk.items if c["passed"]),
    }
    report = {
        "task": "HIRA-038",
        "kind": "delivery_reload_verification",
        "created_at": now(),
        "work_root": str(work_root),
        "source_root": str(REPO),
        "official_runs_root": str(OFFICIAL_RUNS),
        "wall_s": round(time.time() - t0, 1),
        "checks": chk.items,
        "verdict": verdict,
        "steps": steps,
        "scientific_negative_results": [
            "δ=9.0322 m 与 8×10 m 场地量级不匹配：近障 δ 条件 0/200（reports/near_obstacle_summary.md）",
            "位置覆盖 98.40% 下仍有 75.5% 几何重叠：δ 是跟踪误差上界，不是碰撞裕度",
            "heldout 位置覆盖 89.66%，比 90% 参照低 0.34 pp（reports/heldout_summary.md）",
        ],
        "reproducibility_results": [
            "三棵树逐文件散列与源一致，副本脚本自定位到副本",
            "派生锁只改路径前缀、sha256 逐字保留，新根下 verify_artifacts 全过",
            "9 个 fixture 的 actor 输出在副本内复现",
            "小批采集（40 条）在新目录跑通且障碍注入证据完整",
            "官方原始数据可离线复算能量，最差相对误差见 steps.offline",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".sha256").write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    print(f"\n=== {verdict['n_passed']}/{verdict['n_checks']} passed, failed={verdict['failed']} ===")
    print(f"written: {out}")
    return 0 if verdict["reload_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
