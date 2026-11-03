"""Preflight: can this machine reproduce the CaMRA locomotion evidence?

Written for the failure mode that actually happened during the project: a run
that dies because the asset root, the checkpoint or a package is missing looks
exactly like a controller failure in the logs, and an Isaac entry point can even
exit 0 without producing artefacts. This script decides *before* a run whether
the prerequisites are there, and it separates three outcomes:

``ready``
    everything the requested level needs is present and resolvable.
``infra_blocked`` (exit 2)
    resources are missing: checkpoint, asset tree, protocol, logs. The controller
    was never exercised, so nothing can be concluded about it.
``env_incompatible`` (exit 3)
    the interpreter or a package is wrong/missing. Also not a controller verdict.

Package checks use importlib.metadata only: importing torch or Isaac Sim here
would take minutes and can fail on this stack for reasons unrelated to the run.

Levels
    ``eval-now``   checkpoint + assets + protocol + entry points (enough to evaluate)
    ``full-repro`` additionally the TASK-024 / TASK-026 evidence and candidate bundle
                   needed to re-derive the frozen bundle

Usage::

    python scripts/check_repro_readiness.py --level eval-now
    python scripts/check_repro_readiness.py --level full-repro --json /tmp/ready.json
    python scripts/check_repro_readiness.py --level eval-now --out-dir /new/output/dir
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camra_config import ConfigError, cfg  # noqa: E402

REQUIRED_PACKAGES = ["torch", "numpy", "isaaclab", "isaacsim", "rsl-rl-lib", "onnxruntime"]
ENTRY_POINTS = [
    "scripts/eval_locomotion.py",
    "scripts/aggregate_eval_runs.py",
    "scripts/aggregate_final_test.py",
    "scripts/recompute_metrics.py",
    "scripts/export_policy_candidate.py",
    "scripts/verify_export_consistency.py",
    "scripts/rollout_exported_policy.py",
    "scripts/aggregate_rollout_arms.py",
    "scripts/freeze_controller_bundle.py",
    "scripts/verify_frozen_bundle.py",
    "scripts/verify_runtime_contracts.py",
    "scripts/repro/train.sh",
    "scripts/repro/devsweep.sh",
    "scripts/repro/finalize.sh",
    "scripts/repro/finaltest.sh",
    "scripts/repro/export.sh",
    "scripts/repro/rollout.sh",
    "scripts/repro/select_checkpoint.py",
]
PY_ENTRY_POINTS = [p for p in ENTRY_POINTS if p.endswith(".py")]

CHECKPOINT_REL = (
    "rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt"
)
TASK024_SEEDS = 20  # 10 seeds x 2 groups


def result(level: str, name: str, ok: bool, detail: str, blocking: bool = True) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail,
            "level": level, "blocking": blocking}


def check_interpreter(level: str, findings: list) -> None:
    exe = cfg.python_exe()
    if not exe.exists():
        findings.append(result(level, "interpreter",
                               False, f"CAMRA_PYTHON does not exist: {exe}"))
        return
    out = subprocess.run([str(exe), "-V"], capture_output=True, text=True)
    version = (out.stdout or out.stderr).strip()
    match = re.search(r"3\.(\d+)", version)
    ok = out.returncode == 0 and match is not None and int(match.group(1)) >= 10
    findings.append(result(level, "interpreter", ok,
                           f"{exe} -> {version} (need >= 3.10)"))
    # the script itself may run under another interpreter; say so instead of hiding it
    if Path(sys.executable).resolve() != exe.resolve():
        findings.append(result(level, "interpreter_mismatch", True,
                               f"this preflight runs under {sys.executable}, "
                               f"CAMRA_PYTHON is {exe}", blocking=False))


def check_packages(level: str, findings: list) -> None:
    missing = []
    versions = {}
    for name in REQUIRED_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            missing.append(name)
    detail = " ".join(f"{k}={v}" for k, v in versions.items()) or "none found"
    findings.append(result(level, "packages", not missing,
                           detail + (f" | MISSING: {missing}" if missing else "")))
    # torch must be importable, not just installed: the DLL failure seen on this
    # stack is exactly a metadata-present / import-fails case
    try:
        out = subprocess.run([str(cfg.python_exe()), "-c",
                              "import torch; print(torch.__version__, torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=300)
        ok = out.returncode == 0
        findings.append(result(level, "torch_import", ok,
                               (out.stdout or out.stderr).strip()[:200]))
    except subprocess.TimeoutExpired:
        findings.append(result(level, "torch_import", False, "import torch timed out (>300 s)"))


def check_assets(level: str, findings: list) -> None:
    try:
        go2 = cfg.asset_go2()
        usd = cfg.go2_usd()
    except ConfigError as exc:
        findings.append(result(level, "asset_root", False, str(exc)))
        return
    findings.append(result(level, "asset_root", go2.is_dir(), str(go2)))
    findings.append(result(level, "go2_usd", usd.is_file(), str(usd)))
    # other robots must not be required for this task; report but never block
    others = [p.name for p in go2.parent.iterdir() if p.is_dir()] if go2.parent.is_dir() else []
    findings.append(result(level, "asset_siblings", True,
                           f"present under {go2.parent}: {others}", blocking=False))


def check_checkpoint(level: str, findings: list) -> None:
    try:
        ckpt = cfg.path("CAMRA_RSL_LOG_ROOT") / CHECKPOINT_REL
    except ConfigError as exc:
        findings.append(result(level, "checkpoint", False, str(exc)))
        return
    findings.append(result(level, "checkpoint", ckpt.is_file(),
                           f"{ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)" if ckpt.is_file() else str(ckpt)))
    params = ckpt.parent / "params" / "env.yaml"
    findings.append(result(level, "checkpoint_params", params.is_file(), str(params)))


def check_protocol(level: str, findings: list) -> None:
    proto = cfg.protocol()
    findings.append(result(level, "protocol", proto.is_file(), str(proto)))
    supp = cfg.protocol_supplement()
    findings.append(result(level, "protocol_supplement", supp.is_file(), str(supp)))


def check_entry_points(level: str, findings: list) -> None:
    root = cfg.locomotion_root()
    missing = [p for p in ENTRY_POINTS if not (root / p).is_file()]
    findings.append(result(level, "entry_points", not missing,
                           f"{len(ENTRY_POINTS)} declared, missing: {missing}" if missing
                           else f"all {len(ENTRY_POINTS)} present under {root}"))
    # a syntax error is a code bug, not an infrastructure problem: surface it here
    broken = []
    for rel in PY_ENTRY_POINTS:
        path = root / rel
        if not path.is_file():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}:{exc.lineno}: {exc.msg}")
    findings.append(result(level, "entry_points_compile", not broken, "; ".join(broken) or "ok"))


def check_historical_evidence(level: str, findings: list, required: bool) -> None:
    block = "full-repro" if required else level
    task024 = cfg.log_root("TASK-024")
    seeds = sorted(task024.glob("*/s*")) if task024.is_dir() else []
    with_summary = [d for d in seeds if (d / "summary.json").is_file()]
    findings.append(result(block, "task024_raw_logs",
                           (len(with_summary) >= TASK024_SEEDS) if required else True,
                           f"{len(with_summary)}/{TASK024_SEEDS} seed dirs with summary.json "
                           f"under {task024}",
                           blocking=required))
    rollout = cfg.log_root("TASK-026") / "rollout"
    have = [rollout / f"rollout_{b}.json" for b in ("jit", "onnx", "trainer")]
    findings.append(result(block, "task026_rollout",
                           all(p.is_file() for p in have) if required else True,
                           f"{sum(p.is_file() for p in have)}/3 arms under {rollout}",
                           blocking=required))
    candidate = cfg.path("CAMRA_BUNDLE_ROOT") / "candidates" / "r4_it2999_v1"
    findings.append(result(block, "candidate_bundle",
                           candidate.is_dir() if required else True,
                           str(candidate), blocking=required))


def check_device(level: str, findings: list) -> None:
    if not shutil.which("nvidia-smi"):
        findings.append(result(level, "gpu", False, "nvidia-smi not on PATH"))
        return
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True)
    lines = [l for l in out.stdout.strip().splitlines() if l]
    findings.append(result(level, "gpu", bool(lines), " | ".join(lines) or out.stderr.strip()[:200]))

    device = cfg.device()
    index = 0
    match = re.search(r"(\d+)$", device)
    if match:
        index = int(match.group(1))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    findings.append(result(level, "device_setting", True,
                           f"CAMRA_DEVICE={device}, CUDA_VISIBLE_DEVICES={visible or 'unset'}",
                           blocking=False))
    if index >= max(1, len(lines)):
        findings.append(result(level, "device_index", False,
                               f"CAMRA_DEVICE={device} but only {len(lines)} GPU(s) visible"))


def check_tmux(level: str, findings: list) -> None:
    """Long runs must survive a dropped ssh session; report, never block."""
    inside = bool(os.environ.get("TMUX"))
    sessions = ""
    if shutil.which("tmux"):
        out = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        sessions = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""
    findings.append(result(level, "tmux", inside,
                           f"inside tmux={inside}; first session: {sessions or 'none'} "
                           f"(expected session: {cfg.raw('CAMRA_TMUX_SESSION')})",
                           blocking=False))


def check_output_policy(level: str, findings: list, out_dir: str | None) -> None:
    if not out_dir:
        findings.append(result(level, "output_dir", True,
                               "no --out-dir given; scripts default to $CAMRA_LOG_ROOT/repro/<task>",
                               blocking=False))
        return
    path = Path(out_dir)
    exists_non_empty = path.exists() and any(path.iterdir())
    findings.append(result(level, "output_dir", not exists_non_empty,
                           f"{path} {'already has results (use a new dir or --allow-existing)' if exists_non_empty else 'available'}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=["eval-now", "full-repro"], default="eval-now")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--out-dir", default=None, help="planned output directory to check")
    ap.add_argument("--require-evidence", action="store_true",
                    help="also fail when the historical TASK-024/TASK-026 logs are absent")
    args = ap.parse_args()

    findings: list[dict] = []
    check_interpreter(args.level, findings)
    check_packages(args.level, findings)
    check_assets(args.level, findings)
    check_checkpoint(args.level, findings)
    check_protocol(args.level, findings)
    check_entry_points(args.level, findings)
    check_device(args.level, findings)
    check_tmux(args.level, findings)
    check_output_policy(args.level, findings, args.out_dir)
    check_historical_evidence(args.level, findings,
                              required=(args.level == "full-repro" or args.require_evidence))

    print("resolved configuration (env > config/repro.env > built-in default)")
    print(f"{'key':<24} {'source':<8} value")
    for key, info in cfg.describe().items():
        print(f"{key:<24} {info['source']:<8} {info['value']}")
    print()

    print(f"{'check':<26} {'result':<8} detail")
    for item in findings:
        state = "ok" if item["ok"] else ("FAIL" if item["blocking"] else "warn")
        print(f"{item['check']:<26} {state:<8} {item['detail']}")

    blocking = [f["check"] for f in findings if not f["ok"] and f["blocking"]]
    env_names = {"interpreter", "packages", "torch_import", "device_index"}
    if not blocking:
        verdict = "ready"
        code = 0
    elif any(name in env_names for name in blocking):
        verdict = "env_incompatible"
        code = 3
    else:
        verdict = "infra_blocked"
        code = 2

    print()
    print(f"VERDICT: {verdict}")
    if blocking:
        print(f"blocking checks: {blocking}")
        print("This is not a controller result: the run cannot be judged until the "
              "resource above exists (TASK-034 acceptance).")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"level": args.level, "verdict": verdict,
                                   "config": cfg.describe(), "findings": findings},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[readiness] -> {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
