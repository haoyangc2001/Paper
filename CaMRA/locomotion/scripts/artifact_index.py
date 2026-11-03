"""Index and verify the artefacts that live outside the git repository.

The repository is code only: models, raw evaluation logs and the frozen bundle
stay on the delivery machine (18 MB of robot assets, 28 MB of bundle, thousands of
raw log files). Without an index, "the bundle is at some path under /pub/data"
is the only record, which is exactly what a reader on another machine cannot
use. This tool writes and checks that index.

``build``  walks the declared entries, hashes them and writes the JSON index
``verify`` re-hashes and reports every difference, with a non-zero exit code

Hashes are sha256. Directories are hashed as a *tree*: sha256 over the sorted
list of ``<relative path>:<file sha256>`` lines, so both content and layout are
covered. No file content is copied into the index and nothing leaves the host.

Usage::

    python scripts/artifact_index.py build --out docs/artifacts_index.json
    python scripts/artifact_index.py verify --index docs/artifacts_index.json
    python scripts/artifact_index.py show --index docs/artifacts_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camra_config import cfg  # noqa: E402

#: Every entry names the configuration key its root depends on, so that changing
#: CAMRA_* moves the whole index instead of leaving stale absolute paths behind.
ENTRIES: list[dict] = [
    {
        "id": "frozen_bundle_v2",
        "kind": "dir",
        "root": "CAMRA_BUNDLE_ROOT",
        "rel": "frozen/camra_go2_velocity_it2999_v2",
        "required": True,
        "note": "delivered controller, 29 files, tree 0444/0555; reload entry point",
    },
    {
        "id": "frozen_bundle_v1_superseded",
        "kind": "dir",
        "root": "CAMRA_BUNDLE_ROOT",
        "rel": "frozen/camra_go2_velocity_it2999_v1",
        "required": False,
        "note": "kept for the record only: generated from a dirty tree, superseded by v2",
    },
    {
        "id": "candidate_bundle",
        "kind": "dir",
        "root": "CAMRA_BUNDLE_ROOT",
        "rel": "candidates/r4_it2999_v1",
        "required": False,
        "note": "exported JIT/ONNX + verification, input of the freeze step",
    },
    {
        "id": "checkpoint_selected",
        "kind": "file",
        "root": "CAMRA_RSL_LOG_ROOT",
        "rel": "rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt",
        "required": True,
        "note": "iteration 2999 of the round-4 run, single training seed (supplement D-1)",
    },
    {
        "id": "training_run_params",
        "kind": "file",
        "root": "CAMRA_RSL_LOG_ROOT",
        "rel": "rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/params/env.yaml",
        "required": True,
        "note": "the environment config actually trained; the bundle ships a copy",
    },
    {
        "id": "assets_go2",
        "kind": "dir",
        "root": "CAMRA_ASSET_ROOT",
        "rel": "Go2",
        "required": True,
        "note": "robot USD + meshes; copied into the bundle as assets/Go2",
    },
    {
        "id": "raw_logs_task024",
        "kind": "dir",
        "root": "CAMRA_LOG_ROOT",
        "rel": "TASK-024",
        "required": False,
        "note": "final-test raw episodes (10 seeds x 2 groups); not in git, thousands of files",
    },
    {
        "id": "raw_logs_task026",
        "kind": "dir",
        "root": "CAMRA_LOG_ROOT",
        "rel": "TASK-026/rollout",
        "required": False,
        "note": "three-arm closed-loop rollout evidence",
    },
    {
        "id": "eval_logs_task036",
        "kind": "dir",
        "root": "CAMRA_LOG_ROOT",
        "rel": "TASK-036",
        "required": False,
        "note": "confirmatory rerun (9000 episodes) plus the seven low-speed diagnostic experiments; "
                "chapters 39-47 of the handoff notes reference these files",
    },
]

#: Files inside the repository that the index pins by hash (they decide the verdict).
IN_REPO = [
    {"id": "protocol", "rel": "docs/eval_protocol.json", "required": True,
     "note": "evaluation protocol fixed before training"},
    {"id": "protocol_supplement", "rel": "docs/eval_protocol_supplement_v1.0.0.md", "required": True,
     "note": "supplement with the four decisions of TASK-030"},
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: Path) -> dict:
    """sha256 over sorted ``rel:sha256`` lines, plus file count and total bytes."""
    lines: list[str] = []
    total = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rel = item.relative_to(path).as_posix()
            digest = sha256_file(item)
            size = item.stat().st_size
            total += size
            lines.append(f"{rel}:{digest}")
    tree = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {"tree_sha256": tree, "files": len(lines), "bytes": total}


def describe(entry: dict) -> dict:
    """Resolve one entry: CAMRA_* key for out-of-repo artefacts, the module for in-repo ones."""
    root = cfg.path(entry["root"]) if entry.get("root") else cfg.locomotion_root()
    path = root / entry["rel"]
    record = {
        "id": entry["id"],
        "kind": entry["kind"],
        "required": entry["required"],
        "note": entry["note"],
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    if entry["kind"] == "dir":
        record.update(hash_tree(path))
    else:
        record["sha256"] = sha256_file(path)
        record["bytes"] = path.stat().st_size
    return record


def build(index_path: Path) -> int:
    payload = {
        "generated_by": "scripts/artifact_index.py",
        "purpose": "path + hash index for artefacts kept outside git; no content, no credentials",
        "config": {k: v for k, v in cfg.describe().items()
                   if k in {
                       "CAMRA_LOG_ROOT", "CAMRA_ASSET_ROOT", "CAMRA_BUNDLE_ROOT",
                       "CAMRA_RSL_LOG_ROOT",
                   }},
        "entries": [describe(e) for e in ENTRIES]
                   + [describe({**e, "kind": "file"}) for e in IN_REPO],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing_required = [e["id"] for e in payload["entries"] if e["required"] and not e["exists"]]
    print(f"[index] wrote {index_path} ({len(payload['entries'])} entries)")
    for entry in payload["entries"]:
        state = "ok" if entry["exists"] else ("MISSING(required)" if entry["required"] else "absent(optional)")
        detail = entry.get("tree_sha256", entry.get("sha256", ""))[:16]
        count = entry.get("files", "")
        print(f"[index] {entry['id']:<28} {state:<18} {detail} {count}")
    if missing_required:
        print(f"[index] WARNING: required entries missing: {missing_required}", file=sys.stderr)
        return 2
    return 0


def verify(index_path: Path) -> int:
    if not index_path.is_file():
        print(f"[index] FATAL: index not found: {index_path}", file=sys.stderr)
        print("[index] create it with: python scripts/artifact_index.py build", file=sys.stderr)
        return 2
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    drift: list[str] = []
    for stored in payload["entries"]:
        spec = next(
            (e for e in ENTRIES if e["id"] == stored["id"]),
            next((e for e in IN_REPO if e["id"] == stored["id"]), None),
        )
        if spec is None:
            drift.append(f"{stored['id']}: entry disappeared from the tool")
            continue
        current = describe({**spec, "kind": "file"} if spec.get("root") is None else spec)
        if current["exists"] != stored["exists"]:
            drift.append(f"{stored['id']}: existence changed ({stored['exists']} -> {current['exists']})")
            continue
        if not current["exists"]:
            if stored["required"]:
                drift.append(f"{stored['id']}: required artefact missing")
            continue
        key = "tree_sha256" if stored.get("tree_sha256") else "sha256"
        if current.get(key) != stored.get(key):
            drift.append(f"{stored['id']}: {key} differs (indexed {str(stored.get(key))[:12]}, "
                         f"now {str(current.get(key))[:12]})")

    if drift:
        print(f"[index] FAIL: {len(drift)} difference(s) against {index_path}", file=sys.stderr)
        for line in drift:
            print("  " + line, file=sys.stderr)
        return 4
    print(f"[index] PASS: all {len(payload['entries'])} entries match {index_path}")
    return 0


def show(index_path: Path) -> int:
    if not index_path.is_file():
        print(f"[index] FATAL: index not found: {index_path}", file=sys.stderr)
        return 2
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"{'id':<28} {'exists':<6} {'files':>6} {'MiB':>8}  path")
    for entry in payload["entries"]:
        mib = entry.get("bytes", 0) / (1 << 20)
        print(f"{entry['id']:<28} {str(entry['exists']):<6} {str(entry.get('files', '-')):>6} "
              f"{mib:>8.1f}  {entry['path']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "verify", "show"])
    ap.add_argument("--index", default=None,
                    help="default: <locomotion>/docs/artifacts_index.json")
    args = ap.parse_args()
    index_path = Path(args.index) if args.index else cfg.locomotion_root() / "docs" / "artifacts_index.json"
    if args.command == "build":
        return build(index_path)
    if args.command == "verify":
        return verify(index_path)
    return show(index_path)


if __name__ == "__main__":
    sys.exit(main())
