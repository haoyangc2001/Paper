"""Single place where the CaMRA locomotion entry points resolve machine paths.

Before TASK-034 every script hard-coded ``/pub/data/caohy/...``, so a clone on
another machine could not even start: the asset root, the checkpoint, the log
root and the interpreter were all baked in. This module and ``config/repro.env``
replace those literals.

Resolution order for every key (first match wins):

1. the exported environment variable (``export CAMRA_ASSET_ROOT=...``),
2. the value in ``config/repro.env`` next to this repository,
3. the built-in default in ``DEFAULTS`` (the delivery machine, so that a run
   without any configuration reproduces the delivered numbers).

Nothing here imports Isaac Sim or torch: it must stay usable for preflight
checks and for tools that only read logs.

Usage::

    from camra_config import cfg
    ckpt = cfg.path("CAMRA_LOG_ROOT") / "rsl_rl/..."   # or cfg.asset_go2()
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_REL = Path("config") / "repro.env"

#: Delivery-machine values. Kept identical to config/repro.env so that a missing
#: or unreadable env file still resolves to the paths the frozen bundle was
#: produced with instead of failing silently.
DEFAULTS = {
    "CAMRA_PYTHON": "/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python",
    "CAMRA_LOG_ROOT": "/pub/data/caohy/logs/camra",
    "CAMRA_RSL_LOG_ROOT": "/pub/data/caohy/logs/rsl_rl",
    "CAMRA_RUN_ROOT": "/pub/data/caohy/run/camra_go2",
    "CAMRA_ASSET_ROOT": "/pub/data/caohy/unitree_model",
    "CAMRA_BUNDLE_ROOT": "/pub/data/caohy/camra_bundles",
    "CAMRA_XDG_DATA": "/pub/data/caohy/xdgdata",
    "CAMRA_XDG_CACHE": "/pub/data/caohy/xdgcache",
    "CAMRA_TMPDIR": "/pub/data/caohy/tmp",
    "CAMRA_DEVICE": "cuda:0",
    "CAMRA_TMUX_SESSION": "camra-locomotion",
}

#: Keys that are paths (they get expanded and turned into Path objects).
PATH_KEYS = {
    "CAMRA_PYTHON",
    "CAMRA_LOG_ROOT",
    "CAMRA_RSL_LOG_ROOT",
    "CAMRA_RUN_ROOT",
    "CAMRA_ASSET_ROOT",
    "CAMRA_BUNDLE_ROOT",
    "CAMRA_XDG_DATA",
    "CAMRA_XDG_CACHE",
    "CAMRA_TMPDIR",
}


class ConfigError(RuntimeError):
    """Raised when a required path or setting cannot be resolved."""


def locomotion_root() -> Path:
    """``CaMRA/locomotion`` — derived from this file, so it works in any cwd."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Repository root (two levels above ``CaMRA/locomotion``)."""
    return locomotion_root().parents[1]


def env_file() -> Path:
    return locomotion_root() / ENV_FILE_REL


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` starts a comment; blank lines ignored."""
    path = env_file() if path is None else path
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


class Config:
    """Resolved configuration; ``source()`` reports env / file / default per key."""

    def __init__(self) -> None:
        self._file = read_env_file()

    def source(self, key: str) -> str:
        if os.environ.get(key):
            return "env"
        if self._file.get(key):
            return "file"
        if key in DEFAULTS:
            return "default"
        return "unset"

    def raw(self, key: str) -> str:
        if os.environ.get(key):
            return os.environ[key]
        if self._file.get(key):
            return self._file[key]
        if key in DEFAULTS:
            return DEFAULTS[key]
        raise ConfigError(
            f"{key} is not set. Export it, or add a line '{key}=<value>' to {env_file()}."
        )

    def path(self, key: str) -> Path:
        if key not in PATH_KEYS:
            raise ConfigError(f"{key} is not a path key; use raw() for it")
        return Path(os.path.expanduser(os.path.expandvars(self.raw(key)))).resolve()

    # ---- convenience accessors -------------------------------------------------
    def locomotion_root(self) -> Path:
        return locomotion_root()

    def repo_root(self) -> Path:
        return repo_root()

    def python_exe(self) -> Path:
        return self.path("CAMRA_PYTHON")

    def log_root(self, task: str | None = None) -> Path:
        root = self.path("CAMRA_LOG_ROOT")
        return root / task if task else root

    def out_root(self, task: str) -> Path:
        """Default output root for a reproduction run of ``task`` (never in-repo)."""
        return self.log_root() / "repro" / task

    def asset_root(self) -> Path:
        return self.path("CAMRA_ASSET_ROOT")

    def asset_go2(self) -> Path:
        """The Go2 subtree that the frozen bundle copies as ``assets/Go2``."""
        return self.asset_root() / "Go2"

    def go2_usd(self) -> Path:
        return self.asset_go2() / "usd" / "go2.usd"

    def protocol(self) -> Path:
        return locomotion_root() / "docs" / "eval_protocol.json"

    def protocol_supplement(self) -> Path:
        return locomotion_root() / "docs" / "eval_protocol_supplement_v1.0.0.md"

    def device(self) -> str:
        return self.raw("CAMRA_DEVICE")

    def describe(self) -> dict[str, dict[str, str]]:
        """Every known key with its resolved value and where it came from."""
        out: dict[str, dict[str, str]] = {}
        for key in DEFAULTS:
            out[key] = {"value": self.raw(key), "source": self.source(key)}
        return out

    # ---- assertions ------------------------------------------------------------
    def require(self, path: Path, what: str) -> Path:
        if not path.exists():
            raise ConfigError(
                f"{what} not found at {path}. "
                f"Set the corresponding CAMRA_* variable or run "
                f"scripts/check_repro_readiness.py to see which resource is missing."
            )
        return path


cfg = Config()
