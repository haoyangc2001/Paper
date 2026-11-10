"""Run directory identity and collision refusal (NAV-010).

Registered roots live outside the repository so that outputs never mix with the
frozen v2 calibration data or the HiRA experiment directories.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")

DEFAULT_WORK_ROOT = "/pub/data/caohy/work/camra_navigation"
DEFAULT_OUTPUT_ROOT = "/pub/data/caohy/logs/camra_navigation"


class RunDirError(RuntimeError):
    """Raised when a run directory would be reused or a run id is invalid."""


def default_work_root() -> Path:
    return Path(os.environ.get("CAMRA_NAV_WORK_ROOT", DEFAULT_WORK_ROOT))


def default_output_root() -> Path:
    return Path(os.environ.get("CAMRA_NAV_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))


def resolve_run_dir(root: Path | str, run_id: str, *, allow_existing: bool = False) -> Path:
    """Return ``root/run_id``, refusing to reuse an existing non-empty directory."""
    if not RUN_ID_RE.match(run_id):
        raise RunDirError(f"invalid run id: {run_id!r} (must match {RUN_ID_RE.pattern})")
    root = Path(root)
    target = root / run_id
    if target.exists():
        if not allow_existing:
            raise RunDirError(f"run directory already exists: {target}")
        if target.is_dir() and any(target.iterdir()):
            raise RunDirError(f"run directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=False) if not target.exists() else None
    return target


def write_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    task: str,
    controller_id: str,
    extra: dict | None = None,
) -> Path:
    """Write ``run_manifest.json`` capturing host identity and start time."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunDirError(f"run directory does not exist: {run_dir}")
    manifest = {
        "schema": "camra-navigation-run-manifest-v1",
        "run_id": run_id,
        "task": task,
        "controller_id": controller_id,
        "hostname": os.uname().nodename,
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "cwd": os.getcwd(),
        "python": os.environ.get("CAMRA_NAV_PYTHON", ""),
        "extra": extra or {},
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        raise RunDirError(f"run manifest already exists: {path}")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
