"""CaMRA upper-layer navigation (NAV-010 skeleton).

Conventions enforced here, not by documentation alone:

* The frozen low-level controller is ``camra_go2_velocity_it2999_v2``. This package
  never trains, replaces or re-derives it.
* Costs are positive joint mechanical work in joules. Nothing in this package may
  introduce an ``a.u.`` budget.
* Every run writes into its own directory under the registered output root; an
  existing non-empty run directory is refused instead of reused.
"""

from __future__ import annotations

__version__ = "0.1.0"

CONTROLLER_ID = "camra_go2_velocity_it2999_v2"
COST_UNIT = "J"

from .runs import (  # noqa: E402,F401
    RunDirError,
    default_output_root,
    default_work_root,
    resolve_run_dir,
    write_run_manifest,
)

__all__ = [
    "__version__",
    "CONTROLLER_ID",
    "COST_UNIT",
    "RunDirError",
    "default_output_root",
    "default_work_root",
    "resolve_run_dir",
    "write_run_manifest",
]
