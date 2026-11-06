"""诊断：AppLauncher 之后每一步的耗时（定位 run_split 卡在哪）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EXP = Path("/home/caohy/repositories/Paper/CaMRA/calibration/experiments/hira_go2_18500")
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

T = time.time()


def m(text: str) -> None:
    print(f"[diag] {time.time() - T:7.1f}s {text}", flush=True)


m("before AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
m("app ready")

import torch  # noqa: E402

m(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
m(f"cuda count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")

from hira_go2.bundle import load_bundle  # noqa: E402

m("imported bundle")
bundle = load_bundle(EXP / "frozen/hira_go2_18500", expected_controller_id="hira_go2_18500")
m(f"bundle loaded {bundle['manifest']['bundle_id'][:16]}")

from camra_calibration.commands import build_program  # noqa: E402

prog = build_program("straight_shift", 4000, duration_s=30.0)
m(f"program rebuilt {len(prog.commands)} cmds")

from hira_go2.collector import HiraCollector, TrialSpec, sample_spec  # noqa: E402

m("imported collector")
x = torch.zeros(64, device="cuda:0")
m(f"cuda alloc ok {float(x.sum())}")

print("DIAG_OK", flush=True)
import os

os._exit(0)
