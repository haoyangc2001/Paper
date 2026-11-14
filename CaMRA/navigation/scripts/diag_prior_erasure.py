"""NAV-064 主臂「起点 4/6 → 第 50 轮 2~3/6」的机制测量：先验是被稀释还是被改写？

只读 checkpoint，不训练、不改权重。逐检查点读出策略头三个动作分量的**输出偏置**
（vx / vy / wz 在 tanh 之前的均值基准）、输出层权重行范数、探索噪声 log_std，
并与「未训练的初始网络（带转向先验）」逐项对照。

为什么看偏置：转向先验是加在 loc 上的加法项 gain·goal_dir_y_body，它不改权重、
不改 log_prob。若 50 次更新后偏置被改写到反号或归零，说明梯度把先验吃掉了；
若偏置没动多少而行为退化，说明问题不在先验。

只读：不写 checkpoint、不训练。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
from camra_navigation.algorithm import networks as NW  # noqa: E402

WORK = "/pub/data/caohy/work/camra_navigation"
OUT = os.path.join(NAV, "reports", "training_round_r11_prior_erasure.json")
RUNS = [("uncalibrated", "nav064_r11_uncalibrated_s0"),
        ("full", "nav064_r11_full_s0")]
NAMES = ("vx", "vy", "wz")


def init_net(nav: dict) -> NW:
    torch.manual_seed(0)
    return NW.build(nav, action_bounded=True, initial_forward_speed=0.465,
                    log_std_min=-5.0, log_std_init=-3.0, steering_gain=0.2)


def read(net_state: dict) -> dict:
    b = net_state["policy.head.bias"].detach().numpy()
    w = net_state["policy.head.weight"].detach().numpy()
    ls = net_state["policy.log_std"].detach().numpy()
    return {"head_bias": [round(float(x), 4) for x in b],
            "head_row_norm": [round(float(np.linalg.norm(w[i])), 4) for i in range(3)],
            "log_std": [round(float(x), 4) for x in ls],
            "log_std_mean": round(float(ls.mean()), 4)}


def main() -> int:
    nav = json.load(open(os.path.join(NAV, "config", "navigation_contract_v1.json"),
                        encoding="utf-8"))
    base = read(init_net(nav).state_dict())
    out = {"task": "NAV-064", "kind": "prior_erasure_timeline", "actions": NAMES,
           "init_reference": base, "runs": []}
    print("init(带先验):", base["head_bias"], "log_std", base["log_std_mean"])
    for variant, run_id in RUNS:
        cks = sorted(glob.glob(os.path.join(WORK, run_id, "state_u*.pt")),
                     key=lambda p: int(os.path.basename(p).split("_u")[1].split(".")[0]))
        rows = []
        for p in cks:
            st = torch.load(p, map_location="cpu", weights_only=False)
            d = read(st["model"])
            d["update"] = int(os.path.basename(p).split("_u")[1].split(".")[0])
            d["delta_vs_init"] = [round(d["head_bias"][i] - base["head_bias"][i], 4)
                                  for i in range(3)]
            rows.append(d)
        out["runs"].append({"variant": variant, "run": run_id, "checkpoints": rows})
        print(f"\n{variant}:")
        print(f"  {'u':>4} {'vx':>8} {'vy':>8} {'wz':>8} {'Δvx':>8} {'Δwz':>8} "
              f"{'|row_vx|':>9} {'log_std':>8}")
        for r in rows:
            print(f"  {r['update']:>4} {r['head_bias'][0]:>8.4f} {r['head_bias'][1]:>8.4f} "
                  f"{r['head_bias'][2]:>8.4f} {r['delta_vs_init'][0]:>8.4f} "
                  f"{r['delta_vs_init'][2]:>8.4f} {r['head_row_norm'][0]:>9.4f} "
                  f"{r['log_std_mean']:>8.4f}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nSAVED", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
