#!/usr/bin/env python3
"""诊断：深井为何不进策略优势？并验证 'h 右移一格' 修复是否成立。

背景：v2 门禁实测 h_T 从 0 变 -3，但 adv_total 逐位相同 → 联合 min 递推在
停止点取的是步前 h_t（t 侧距离），到达状态只存在于附加 h_T，被 done 截断。

本探针在同一批到达回合上对比三种口径的 adv_total：
  A. 现行 v2（h_t 为步前）          —— 预期与 v1 相同
  B. v1（对照）
  C. 修复候选：联合序列用步后 h（h_ext[1:]）—— 预期到达步拿到 -min(3, z_T) 量级的负值
另打印 V_h 通道（target_reach）的末步值，确认深井在 critic 通道是生效的。
"""
import dataclasses
import os
import sys

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

import numpy as np
from camra_navigation.algorithm import storage as S
from camra_navigation.algorithm import targets as T

WORK = "/pub/data/caohy/work/camra_navigation"
RUN = "nav064_r12_uncalibrated_s0"
VALUE_CONTRACT = os.path.join(NAV, "config", "value_objective_contract_v1.json")


def cfg_for(h_semantics: str) -> S.ObjectiveConfig:
    cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
    return dataclasses.replace(cfg, joint_cost_weight=0.0,
                               joint_cost_base_weight=0.0,
                               joint_cost_on_overspend_only=False,
                               joint_min_recursion=True, h_semantics=h_semantics)


def shifted_v2(arrays, cfg):
    """修复候选：联合序列的 h 用**步后状态**（h_{t+1}），终端补 h_T；其余不变。"""
    out = S.compute_objectives(arrays, cfg, terminal_kind="success")
    h_t = np.asarray(out["obj_h"], dtype=np.float64)
    h_T = float(out["obj_h_terminal"][0])
    z_next = np.asarray(arrays["z_norm_next"], dtype=np.float64)
    # 每步的"停止价值"取步后状态：t=0..T-1 → h_{t+1}，z_{t+1}
    g_arr = np.maximum(h_t[1:].tolist() + [h_T], -z_next)          # 长度 T
    g_ext = np.concatenate([g_arr, [max(h_T, -float(z_next[-1]))]])  # 长度 T+1
    done_ext = np.concatenate([np.asarray(out["obj_stop_now"], dtype=np.int64), [1]])
    total_value = np.maximum(np.asarray(arrays["value_reach"], dtype=np.float64),
                             np.asarray(arrays["value_cost"], dtype=np.float64)
                             - np.asarray(arrays["z_norm"], dtype=np.float64))
    target = T.joint_min_return(g_ext, total_value, done_ext,
                                cfg.gamma_h, cfg.gae_lambda)
    return target - total_value


def main() -> None:
    cfg_v1, cfg_v2 = cfg_for("v1"), cfg_for("v2_deep_well")
    run_dir = os.path.join(WORK, RUN)
    seen = 0
    for env_name in sorted(e for e in os.listdir(run_dir) if e.startswith("rollout_env")):
        store = S.RolloutStore(run_dir, env_name, obs_dim=41)
        for idx in range(0, store.n_chunks, 7):
            meta = store.entries[idx]
            if str(meta.get("terminal_kind")) != "success":
                continue
            arrays = store.load_chunk(idx)
            o1 = S.compute_objectives(arrays, cfg_v1, terminal_kind="success")
            o2 = S.compute_objectives(arrays, cfg_v2, terminal_kind="success")
            adv_shift = shifted_v2(arrays, cfg_v2)
            print(f"--- {env_name} chunk {idx} | steps={len(o2['obj_h'])} ---")
            print("  h_t tail      :", np.round(o2["obj_h"][-3:], 3).tolist(),
                  "| h_T:", round(float(o2["obj_h_terminal"][0]), 3))
            print("  adv v1 tail   :", np.round(o1["obj_adv_total"][-3:], 4).tolist())
            print("  adv v2 tail   :", np.round(o2["obj_adv_total"][-3:], 4).tolist())
            print("  adv shift tail:", np.round(np.asarray(adv_shift)[-3:], 4).tolist())
            print("  target_reach v1/v2 tail:",
                  np.round(o1["obj_target_reach"][-2:], 3).tolist(),
                  np.round(o2["obj_target_reach"][-2:], 3).tolist())
            print("  v1==v2 adv:", np.allclose(o1["obj_adv_total"], o2["obj_adv_total"]))
            seen += 1
            if seen >= 3:
                return
    print("no success chunk found")


if __name__ == "__main__":
    main()
