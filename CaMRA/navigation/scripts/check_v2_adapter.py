"""Parity check for the frozen v2 loader adapter (NAV-024).

The adapter must reproduce the old frozen run method exactly.  Three arms run
over the 19200 observation samples that shipped with the bundle:

* ``adapter``    - camra_navigation.control.v2_policy.FrozenV2Policy (jit)
* ``jit_reference`` - a fresh torch.jit.load, i.e. the old frozen run method
* ``onnx``       - onnxruntime, the independent export format

plus the joint-target rule ``q_target = default + 0.25 * a`` recomputed by hand.

Run:

    <frozen python> CaMRA/navigation/scripts/check_v2_adapter.py \
        --json CaMRA/navigation/reports/v2_adapter_parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.control.v2_policy import (  # noqa: E402
    ACTION_DIM,
    OBS_DIM,
    FrozenV2Policy,
    V2PolicyError,
)

BUNDLE = "/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2"

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def load_samples() -> np.ndarray:
    path = os.path.join(BUNDLE, "samples", "obs_samples.npz")
    data = np.load(path)
    keys = list(data.keys())
    for k in keys:
        arr = data[k]
        if arr.ndim == 2 and arr.shape[1] == OBS_DIM:
            return np.asarray(arr, dtype=np.float32), keys
    raise RuntimeError(f"no (N, {OBS_DIM}) array in {path}; keys={keys}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all 19200")
    args = ap.parse_args()

    # --- 1. load with verification ----------------------------------------
    adapter = FrozenV2Policy.load(BUNDLE, device="cpu", backend="jit")
    prov = adapter.provenance()
    check(
        "artifacts_verified_before_load",
        all(v["matches_export_manifest"] for v in prov["verified_artifacts"].values()),
        prov["verified_artifacts"],
    )
    check(
        "eval_mode_and_grads_off",
        adapter.policy.training is False
        and all(p.requires_grad is False for p in adapter.policy.parameters()),
        {
            "training": adapter.policy.training,
            "n_params_frozen": sum(
                1 for p in adapter.policy.parameters() if p.requires_grad is False
            ),
            "parameter_count": adapter.parameter_count,
            "parameter_count_matches": adapter.parameter_count_matches,
        },
    )

    # --- 2. joint names match the frozen contract -------------------------
    frozen = adapter.frozen_manifest
    names_export = list(adapter.action_contract["joint_names"])
    names_frozen = list(frozen["contract"]["action"]["joint_names"])
    check(
        "joint_order_matches_frozen_contract",
        names_export == names_frozen,
        {"export_manifest": names_export[:4], "frozen_manifest": names_frozen[:4],
         "equal": names_export == names_frozen, "n": len(names_export)},
    )

    # --- 3. samples --------------------------------------------------------
    obs, keys = load_samples()
    if args.max_samples:
        obs = obs[: args.max_samples]
    check(
        "sample_set_matches_manifest",
        obs.shape[1] == OBS_DIM
        and obs.shape[0] == adapter.export_manifest["sample_set"]["num_samples"],
        {
            "shape": list(obs.shape),
            "npz_keys": keys,
            "declared": adapter.export_manifest["sample_set"]["num_samples"],
            "seed": adapter.export_manifest["sample_set"]["seed"],
        },
    )

    # --- 4. reference arms -------------------------------------------------
    jit_ref = torch.jit.load(os.path.join(BUNDLE, "exported", "policy.pt"), map_location="cpu").eval()
    with torch.no_grad():
        out_jit_ref = jit_ref(torch.from_numpy(obs))
    adapter_out, anomaly = adapter.raw_action(obs)

    import onnxruntime as ort

    sess = ort.InferenceSession(
        os.path.join(BUNDLE, "exported", "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    batch_dim = sess.get_inputs()[0].shape[0]
    in_name = sess.get_inputs()[0].name
    if isinstance(batch_dim, int) and batch_dim == 1:
        # The export declares a fixed batch of 1; the reference rollout script runs
        # the same looped-batch1 mode, so this is the old way, not a workaround.
        rows = [sess.run(None, {in_name: obs[i : i + 1]})[0] for i in range(obs.shape[0])]
        out_onnx = torch.from_numpy(np.concatenate(rows, axis=0))
    else:
        out_onnx = torch.from_numpy(sess.run(None, {in_name: obs})[0])

    # --- 5. parity ---------------------------------------------------------
    diff_jit = float((adapter_out - out_jit_ref).abs().max())
    diff_onnx = float((adapter_out - out_onnx).abs().max())
    tol = adapter.export_manifest["tolerance"]
    check(
        "adapter_matches_old_frozen_method_bitwise",
        diff_jit == 0.0,
        {
            "max_abs_diff_adapter_vs_jit": diff_jit,
            "note": "同一图、同一权重、同一 dtype：必须逐位相同",
        },
    )
    check(
        "adapter_within_export_tolerance_vs_onnx",
        diff_onnx <= tol["onnx_vs_jit_max_abs"],
        {
            "max_abs_diff_adapter_vs_onnx": diff_onnx,
            "declared_tolerance": tol["onnx_vs_jit_max_abs"],
            "prior_verification": adapter.export_manifest["verification"]["max_abs_diff_onnx_vs_trainer"],
        },
    )

    # --- 6. joint-target rule ----------------------------------------------
    targets, _ = adapter.joint_targets(obs)
    offset = torch.tensor(adapter.offset)
    scale = torch.tensor(adapter.scale)
    hand = offset + scale * out_jit_ref
    diff_target = float((targets - hand).abs().max())
    check(
        "joint_targets_follow_frozen_rule",
        diff_target == 0.0 and targets.shape == (obs.shape[0], ACTION_DIM),
        {
            "max_abs_diff_vs_hand": diff_target,
            "rule": "q_target = default_joint_pos + 0.25 * a",
            "offset_head": list(adapter.offset[:4]),
            "scale": list(adapter.scale[:4]),
        },
    )

    # --- 7. clip does not bind ---------------------------------------------
    raw_max = float(adapter_out.abs().max())
    check(
        "raw_actions_far_from_clip",
        raw_max < 100.0,
        {
            "max_abs_raw_action": raw_max,
            "clip": [-100.0, 100.0],
            "note": "清单注明 clip 对 O(1) 输出不生效；这里用数据证实",
        },
    )

    # --- 8. anomalous inputs are traceable ---------------------------------
    bad = obs[:4].copy()
    bad[1] = np.nan
    bad[2] = np.inf
    try:
        _, report = adapter.raw_action(bad)
        flagged = sorted(report.non_finite_rows) == [1, 2]
        zero_ok = len(report.action_of_zero_obs) == ACTION_DIM and all(
            math_isfinite(v) for v in report.action_of_zero_obs
        )
    except V2PolicyError:
        flagged, zero_ok = False, False
    check(
        "anomalous_inputs_traceable",
        flagged and zero_ok,
        {
            "non_finite_rows": report.non_finite_rows,
            "replaced_with_zero_row": report.replaced_with_zero_row,
            "action_of_zero_obs_head": report.action_of_zero_obs[:4],
            "note": "NaN/Inf 行被置零并报告；置零观测的动作本身有限，可作为可追溯回退值",
        },
    )

    # --- 9. no gradient escapes --------------------------------------------
    obs_t = torch.from_numpy(obs[:8]).requires_grad_(True)
    out, _ = adapter.raw_action(obs_t.detach().numpy())
    check(
        "no_grad_on_forward",
        out.requires_grad is False,
        {
            "output_requires_grad": out.requires_grad,
            "note": "适配器在 no_grad 内前向；策略参数 requires_grad 全为 False",
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report_doc = {
        "task": "NAV-024",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "samples": int(obs.shape[0]),
        "provenance": prov,
        "parity": {
            "adapter_vs_jit_max_abs": diff_jit,
            "adapter_vs_onnx_max_abs": diff_onnx,
            "joint_target_max_abs": diff_target,
            "max_abs_raw_action": raw_max,
        },
        "checks": RESULTS,
    }
    print(json.dumps({k: report_doc[k] for k in ("task", "verdict", "n_checks", "samples", "parity")},
                     ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:300])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report_doc, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0 if verdict else 1


def math_isfinite(v: float) -> bool:
    from math import isfinite

    return isfinite(v)


if __name__ == "__main__":
    raise SystemExit(main())
