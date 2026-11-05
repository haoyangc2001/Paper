"""分阶段定位资产导入卡在哪一步（HIRA-012 诊断用，不是验收脚本）。

    python tests/diag_asset_stage.py --headless --stage 4

每完成一阶段打印 ``STAGE<n>_OK`` 并 flush，用 ``python -u`` 跑即可看清卡点。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
sys.path.insert(0, str(ADAPTER))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--usd-dir", required=True)
parser.add_argument("--stage", type=int, default=5)
parser.add_argument("--merge-fixed", default="true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


T0 = time.time()


def mark(text: str) -> None:
    print(f"[diag] {time.time() - T0:7.1f}s {text}", flush=True)


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from hira_go2 import asset_cfg  # noqa: E402

mark("imports done")

sim = SimulationContext(SimulationCfg(dt=asset_cfg.HIRA_SIM_DT, device=args_cli.device, gravity=(0.0, 0.0, -9.81)))
mark("STAGE1_OK simulation context")

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
light_cfg = sim_utils.DomeLightCfg(intensity=300.0)
light_cfg.func("/World/Light", light_cfg)
sim.reset()
mark("STAGE2_OK ground+light+reset")

if args_cli.stage >= 3:
    cfg = asset_cfg.build_hira_go2_cfg(args_cli.urdf, usd_dir=args_cli.usd_dir)
    cfg.spawn.merge_fixed_joints = args_cli.merge_fixed.lower() == "true"
    cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, 0.42))
    mark("STAGE3_OK urdf spawned as usd")

if args_cli.stage >= 4:
    from isaaclab.assets import Articulation  # noqa: E402

    cfg = asset_cfg.build_hira_go2_cfg(args_cli.urdf, usd_dir=args_cli.usd_dir).replace(prim_path="/World/Robot")
    cfg.spawn.merge_fixed_joints = args_cli.merge_fixed.lower() == "true"
    robot = Articulation(cfg)
    mark("STAGE4_OK articulation object created")

if args_cli.stage >= 5:
    sim.reset()
    mark(f"STAGE5_OK reset; dof={robot.num_joints} names={robot.joint_names}")

if args_cli.stage >= 6:
    sim.step(render=False)
    mark("STAGE6_OK one physics step")

mark("CLOSING")
# Isaac Sim 5.1 在 headless + 多用户占卡的机器上 close() 会长时间不返回（本机实测 >5 min）。
# 诊断脚本只关心中间步骤，故直接强制退出；验收脚本同样处理，并在报告里登记这条差异。
os._exit(0)
