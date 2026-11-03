"""Check that the exported actors reproduce the trainer's inference (TASK-025).

Runs without Isaac Sim: it only needs torch (for the JIT module), onnxruntime
(for the ONNX module) and the observation samples written by
``export_policy_candidate.py``. That makes the check re-runnable at any time,
including from the frozen bundle in TASK-027.

The pass/fail thresholds are **read from the manifest**, not hard-coded here, so
the criterion has to be fixed before the comparison is made.

Checks
  1. sample set width equals the contract observation dim
  2. both formats produce the contract action dim
  3. JIT vs trainer   max|diff| <= tolerance.jit_vs_trainer_max_abs
  4. ONNX vs trainer  max|diff| <= tolerance.onnx_vs_trainer_max_abs
  5. ONNX vs JIT      max|diff| <= tolerance.onnx_vs_jit_max_abs
  6. repeated calls are identical (JIT and ONNX)
  7. no NaN/Inf in any of the three
  8. ONNX graph takes exactly one input of width == obs dim and returns one output
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch


def _stats(diff: np.ndarray) -> dict:
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "p999_abs": float(np.quantile(np.abs(diff), 0.999)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify JIT/ONNX actors against the trainer's inference.")
    ap.add_argument("--bundle", required=True, help="Candidate bundle directory produced by the exporter.")
    ap.add_argument("--report", default=None, help="Report path (default: <bundle>/params/export_verification.json).")
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    man_path = os.path.join(bundle, "params", "export_manifest.json")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)

    tol = man.get("tolerance")
    if not tol:
        raise SystemExit("manifest has no declared tolerance; refusing to invent one after the fact")

    samples = np.load(os.path.join(bundle, man["sample_set"]["path"]))
    obs = torch.from_numpy(samples["obs"]).float()
    ref = samples["action_trainer"].astype(np.float64)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: dict) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[CHECK] {'PASS' if ok else 'FAIL'}  {name}  {json.dumps(detail, ensure_ascii=False)}", flush=True)

    # 1 / 2 - dimensional contract
    check(
        "sample_width_matches_obs_contract",
        int(obs.shape[1]) == int(man["observation_contract"]["total_dim"]),
        {"samples": int(obs.shape[1]), "contract": int(man["observation_contract"]["total_dim"])},
    )

    # 3 - JIT
    jit = torch.jit.load(os.path.join(bundle, "exported", "policy.pt"), map_location="cpu").eval()
    with torch.inference_mode():
        jit_out = jit(obs).double().numpy()
        jit_out2 = jit(obs).double().numpy()
    check(
        "jit_shapes",
        jit_out.shape == ref.shape,
        {"jit": list(jit_out.shape), "trainer": list(ref.shape)},
    )
    check("jit_finite", bool(np.isfinite(jit_out).all()), {"any_non_finite": bool(not np.isfinite(jit_out).all())})

    # 4 - ONNX
    onnx_path = os.path.join(bundle, "exported", "policy.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()
    out_meta = sess.get_outputs()
    in_name = in_meta[0].name
    batch_dim = in_meta[0].shape[0]
    obs_np32 = obs.numpy()

    def run_onnx(x: np.ndarray) -> np.ndarray:
        """The exporter traces with batch 1, so a fixed batch dim has to be fed one by one.

        That is not a limitation for deployment (the C++ runtime infers one frame at a
        time), but it is recorded in the report so nobody assumes batched inference.
        """
        if isinstance(batch_dim, int) and batch_dim > 0 and batch_dim != x.shape[0]:
            outs = [sess.run(None, {in_name: x[i:i + 1]})[0] for i in range(x.shape[0])]
            return np.concatenate(outs, axis=0).astype(np.float64)
        return sess.run(None, {in_name: x})[0].astype(np.float64)

    onnx_out = run_onnx(obs_np32)
    onnx_out2 = run_onnx(obs_np32[:256])
    check(
        "onnx_graph_contract",
        len(in_meta) == 1 and len(out_meta) == 1 and int(in_meta[0].shape[-1]) == int(obs.shape[1]),
        {
            "inputs": [[m.name, list(m.shape)] for m in in_meta],
            "outputs": [[m.name, list(m.shape)] for m in out_meta],
            "providers": sess.get_providers(),
            "batch_mode": "single-call" if batch_dim in (-1, None) or batch_dim == int(obs.shape[0]) else "looped-batch1",
        },
    )
    check(
        "onnx_shapes",
        onnx_out.shape == ref.shape,
        {"onnx": list(onnx_out.shape), "trainer": list(ref.shape)},
    )
    check("onnx_finite", bool(np.isfinite(onnx_out).all()), {"any_non_finite": bool(not np.isfinite(onnx_out).all())})
    check("trainer_finite", bool(np.isfinite(ref).all()), {"any_non_finite": bool(not np.isfinite(ref).all())})

    # 5..7 - numerical agreement
    d_jit = jit_out - ref
    d_onnx = onnx_out - ref
    d_both = onnx_out - jit_out
    s_jit, s_onnx, s_both = _stats(d_jit), _stats(d_onnx), _stats(d_both)
    check("jit_vs_trainer", s_jit["max_abs"] <= float(tol["jit_vs_trainer_max_abs"]),
          {**s_jit, "limit": float(tol["jit_vs_trainer_max_abs"])})
    check("onnx_vs_trainer", s_onnx["max_abs"] <= float(tol["onnx_vs_trainer_max_abs"]),
          {**s_onnx, "limit": float(tol["onnx_vs_trainer_max_abs"])})
    check("onnx_vs_jit", s_both["max_abs"] <= float(tol["onnx_vs_jit_max_abs"]),
          {**s_both, "limit": float(tol["onnx_vs_jit_max_abs"])})

    # 8 - determinism
    det_jit = float(np.max(np.abs(jit_out - jit_out2)))
    det_onnx = float(np.max(np.abs(onnx_out[:256] - onnx_out2)))
    lim_det = float(tol.get("determinism_max_abs", 0.0))
    check("determinism", det_jit <= lim_det and det_onnx <= lim_det,
          {"jit": det_jit, "onnx": det_onnx, "limit": lim_det})

    # action scale context so the numbers above are interpretable
    scale = man["action_contract"]["scale"]
    joint_err_rad = float(np.max(np.abs(d_onnx)) * max(scale)) if scale else None

    passed = all(c["pass"] for c in checks)
    report = {
        "bundle": bundle,
        "checkpoint_sha256": man["checkpoint"]["sha256"],
        "samples": int(obs.shape[0]),
        "tolerance": tol,
        "max_abs_diff_in_joint_rad": joint_err_rad,
        "checks": checks,
        "result": "PASS" if passed else "FAIL",
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # A verification tool must not need write access to the thing it verifies. On a frozen
    # bundle (0444/0555) both the report and the manifest are read-only, so fall back to
    # writing the report next to the caller and leave the bundle untouched.
    man_path_writable = os.access(man_path, os.W_OK)
    report_path = args.report or os.path.join(bundle, "params", "export_verification.json")
    can_update_bundle = man_path_writable and os.access(os.path.dirname(report_path), os.W_OK)
    if args.report is None and not can_update_bundle:
        report_path = os.path.abspath("export_verification.json")
        print(f"[VERIFY] bundle is read-only; writing the report to {report_path} and leaving "
              f"the manifest untouched (a frozen bundle must not change)", flush=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if not can_update_bundle:
        print(f"[VERIFY] manifest not updated: {man_path} is read-only", flush=True)
        print(f"[VERIFY] samples={obs.shape[0]} jit_max={s_jit['max_abs']:.3e} "
              f"onnx_max={s_onnx['max_abs']:.3e} joint_rad={joint_err_rad:.3e}", flush=True)
        print(f"[VERIFY] {report['result']} -> {report_path}", flush=True)
        return 0 if passed else 1

    # record the outcome in the manifest; the bundle only becomes "verified candidate"
    man["verification"] = {
        "status": "pass" if passed else "fail",
        "script": "scripts/verify_export_consistency.py",
        "report": os.path.relpath(report_path, bundle),
        "samples": int(obs.shape[0]),
        "max_abs_diff_onnx_vs_trainer": s_onnx["max_abs"],
        "max_abs_diff_jit_vs_trainer": s_jit["max_abs"],
        "max_abs_diff_in_joint_rad": joint_err_rad,
        "verified_at": report["timestamp"],
    }
    man["release_state"] = "candidate_verified" if passed else "candidate_unverified"
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)

    print(f"[VERIFY] samples={obs.shape[0]} jit_max={s_jit['max_abs']:.3e} "
          f"onnx_max={s_onnx['max_abs']:.3e} joint_rad={joint_err_rad:.3e}", flush=True)
    print(f"[VERIFY] {report['result']} -> {report_path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
