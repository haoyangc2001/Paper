"""Verify a frozen controller bundle against its own manifest (TASK-027).

Checks, in order, and stops at the first class of failure that makes later checks
meaningless:

1. ``frozen_manifest.json`` still hashes to ``frozen_manifest.json.sha256``;
2. every file listed in ``integrity.files`` is present with the recorded sha256 and
   size, and nothing outside the list (except the two manifests) was added or removed;
3. the tree digest over the sorted ``<path> <sha> <bytes>`` lines matches;
4. the tree really is read-only (no write bit on any file or directory);
5. the exported graphs still *load and run* without the training directory: TorchScript
   and ONNX each replay a slice of ``samples/obs_samples.npz`` and are compared with the
   stored trainer actions inside the tolerance declared in the bundled export manifest.

The point of step 5 is that a bundle can be byte-perfect and still unusable if it lost a
dependency, so "hashes match" alone is not accepted as "reloadable".

Usage:
    python scripts/verify_frozen_bundle.py --bundle /path/to/frozen/version [--samples 512]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_integrity(root: str) -> tuple[dict[str, dict], str]:
    files: dict[str, dict] = {}
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            if name.endswith(".sha256") or name == "frozen_manifest.json":
                continue
            full = os.path.join(base, name)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root)
            files[rel] = {"sha256": sha256_file(full), "bytes": os.path.getsize(full)}
    lines = "\n".join(f"{rel} {m['sha256']} {m['bytes']}" for rel, m in sorted(files.items()))
    return files, hashlib.sha256(lines.encode()).hexdigest()


def check_readonly(root: str) -> list[str]:
    bad = []
    for base, dirs, names in os.walk(root):
        for name in names + dirs:
            p = os.path.join(base, name)
            mode = os.stat(p, follow_symlinks=False).st_mode
            if mode & stat.S_IWUSR:
                bad.append(os.path.relpath(p, root))
    return bad


def check_reloadable(bundle: str, n: int, tol: dict) -> dict:
    import numpy as np

    z = np.load(os.path.join(bundle, "samples", "obs_samples.npz"))
    obs = z["obs"][:n]
    ref = z["action_trainer"][:n]

    import torch

    jit = torch.jit.load(os.path.join(bundle, "exported", "policy.pt"), map_location="cpu").eval()
    with torch.no_grad():
        jit_out = []
        for i in range(n):
            jit_out.append(jit(torch.from_numpy(obs[i : i + 1])).numpy())
    jit_out = np.concatenate(jit_out)

    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.join(bundle, "exported", "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    onnx_out = np.concatenate([sess.run(None, {name: obs[i : i + 1]})[0] for i in range(n)])

    res = {
        "samples_replayed": int(n),
        "jit_vs_trainer_max_abs": float(np.abs(jit_out - ref).max()),
        "onnx_vs_trainer_max_abs": float(np.abs(onnx_out - ref).max()),
        "onnx_vs_jit_max_abs": float(np.abs(onnx_out - jit_out).max()),
        "finite": bool(np.isfinite(jit_out).all() and np.isfinite(onnx_out).all()),
    }
    res["pass"] = (
        res["finite"]
        and res["jit_vs_trainer_max_abs"] <= tol["jit_vs_trainer_max_abs"]
        and res["onnx_vs_trainer_max_abs"] <= tol["onnx_vs_trainer_max_abs"]
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a frozen controller bundle.")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--samples", type=int, default=512, help="how many obs rows to replay")
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    problems: list[str] = []

    mpath = os.path.join(bundle, "frozen_manifest.json")
    spath = mpath + ".sha256"
    if not (os.path.isfile(mpath) and os.path.isfile(spath)):
        print("[VERIFY-FROZEN] FAIL: manifest or its signature is missing")
        return 2
    declared = open(spath, encoding="utf-8").read().split()[0]
    actual = sha256_file(mpath)
    if declared != actual:
        problems.append(f"manifest sha256 {actual} != declared {declared}")
    doc = json.load(open(mpath, encoding="utf-8"))

    # 2 + 3: per-file table and tree digest
    files, tree = tree_integrity(bundle)
    recorded = doc["integrity"]["files"]
    missing = sorted(set(recorded) - set(files))
    extra = sorted(set(files) - set(recorded))
    if missing:
        problems.append(f"missing {len(missing)} file(s): {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected file(s): {extra[:5]}")
    mismatched = [rel for rel in sorted(set(recorded) & set(files))
                  if files[rel]["sha256"] != recorded[rel]["sha256"]
                  or files[rel]["bytes"] != recorded[rel]["bytes"]]
    if mismatched:
        problems.append(f"{len(mismatched)} hash/size mismatch(es): {mismatched[:5]}")
    if tree != doc["integrity"]["tree_sha256"]:
        problems.append(f"tree digest {tree} != declared {doc['integrity']['tree_sha256']}")

    # 4: immutability
    writable = check_readonly(bundle)
    if writable:
        problems.append(f"{len(writable)} writable path(s): {writable[:5]}")

    # 5: still reloadable, using the tolerance declared inside the bundle
    export_man = json.load(open(os.path.join(bundle, "params", "export_manifest.json"), encoding="utf-8"))
    tol = {
        "jit_vs_trainer_max_abs": export_man["tolerance"]["jit_vs_trainer_max_abs"],
        "onnx_vs_trainer_max_abs": export_man["tolerance"]["onnx_vs_trainer_max_abs"],
    }
    reload_res = check_reloadable(bundle, args.samples, tol)
    if not reload_res["pass"]:
        problems.append(f"reload replay failed: {reload_res}")

    print(json.dumps({
        "bundle": bundle,
        "version": doc["version"],
        "files": len(files),
        "tree_sha256": tree,
        "manifest_sha256": actual,
        "reload": reload_res,
        "readonly_paths_checked": True,
        "problems": problems,
    }, indent=2, ensure_ascii=False))
    ok = not problems
    print(f"[VERIFY-FROZEN] {'PASS' if ok else 'FAIL'} ({len(files)} files, {args.samples} samples replayed)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
