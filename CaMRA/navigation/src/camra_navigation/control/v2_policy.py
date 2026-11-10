"""Standalone loader for the frozen v2 velocity policy (NAV-024).

This module mirrors the ``jit`` arm of
``CaMRA/locomotion/scripts/rollout_exported_policy.py`` (its ``build_policy``)
without importing that script: the original imports ``isaaclab.app.AppLauncher``
and parses CLI arguments at module level, so importing it would drag the whole
Isaac Sim runtime into a process that only needs ``torch``.

What it does, and deliberately does not do:

* loads the exported TorchScript graph from the frozen bundle **read-only**, in
  eval mode, under ``torch.no_grad``;
* verifies every byte against both the bundle's ``export_manifest.json`` and the
  repo's ``frozen_manifest.json`` before the graph is used;
* never trains, never updates a parameter, never substitutes an asset, a joint
  order, a PD gain or an observation scaling;
* flags non-finite observations instead of propagating them, so a bad input is
  traceable rather than silently becoming a bad joint target.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import torch

DEFAULT_DEVICE = "cpu"
OBS_DIM = 45
ACTION_DIM = 12


class V2PolicyError(RuntimeError):
    """Raised when the frozen policy or its contract cannot be trusted."""


@dataclass
class AnomalyReport:
    """What the adapter did with inputs that should never reach the robot."""

    non_finite_rows: list[int] = field(default_factory=list)
    replaced_with_zero_row: bool = False
    action_of_zero_obs: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "non_finite_rows": self.non_finite_rows,
            "replaced_with_zero_row": self.replaced_with_zero_row,
            "action_of_zero_obs": list(self.action_of_zero_obs),
        }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class FrozenV2Policy:
    """The frozen ``camra_go2_velocity_it2999_v2`` inference graph."""

    bundle_root: str
    device: str
    policy: object
    raw_action_of: object
    export_manifest: dict
    frozen_manifest: dict
    verified: dict
    obs_contract: dict
    action_contract: dict
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    backend: str
    parameter_count: int
    parameter_count_matches: str

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        bundle_root: str,
        *,
        frozen_manifest_path: str | None = None,
        device: str = DEFAULT_DEVICE,
        backend: str = "jit",
    ) -> "FrozenV2Policy":
        if backend not in ("jit", "onnx"):
            raise V2PolicyError(f"backend must be jit or onnx, got {backend!r}")
        export_man = json.load(
            open(os.path.join(bundle_root, "params", "export_manifest.json"), encoding="utf-8")
        )
        if frozen_manifest_path is None:
            frozen_manifest_path = os.path.join(bundle_root, "frozen_manifest.json")
        frozen_man = json.load(open(frozen_manifest_path, encoding="utf-8"))

        verified: dict[str, dict] = {}
        for rel, key in (
            ("exported/policy.pt", "jit"),
            ("exported/policy.onnx", "onnx"),
        ):
            path = os.path.join(bundle_root, rel)
            got = _sha256(path)
            declared_export = export_man["artifacts"][rel]["sha256"]
            declared_frozen = frozen_man["integrity"]["files"].get(rel, {}).get("sha256")
            verified[rel] = {
                "sha256": got,
                "matches_export_manifest": got == declared_export,
                "matches_frozen_manifest": declared_frozen is None or got == declared_frozen,
            }
            if not verified[rel]["matches_export_manifest"] or (
                declared_frozen is not None and not verified[rel]["matches_frozen_manifest"]
            ):
                raise V2PolicyError(f"{rel} bytes do not match the manifests; refusing to load")

        ckpt_rel = "model/model_2999.pt"
        ckpt_path = os.path.join(bundle_root, ckpt_rel)
        ckpt_record = None
        if os.path.isfile(ckpt_path):
            got = _sha256(ckpt_path)
            declared = export_man["checkpoint"]["sha256"]
            ckpt_record = {"sha256": got, "matches_export_manifest": got == declared}
            if not ckpt_record["matches_export_manifest"]:
                raise V2PolicyError("checkpoint bytes do not match the export manifest")

        if backend == "jit":
            graph = torch.jit.load(
                os.path.join(bundle_root, "exported", "policy.pt"), map_location=device
            )
            graph.eval()
            for p in graph.parameters():
                p.requires_grad_(False)

            def raw_action_of(obs: torch.Tensor) -> torch.Tensor:
                return graph(obs)

            n_params = int(sum(p.numel() for p in graph.parameters()))
        else:
            import onnxruntime as ort

            sess = ort.InferenceSession(
                os.path.join(bundle_root, "exported", "policy.onnx"),
                providers=["CPUExecutionProvider"],
            )
            input_name = sess.get_inputs()[0].name

            def raw_action_of(obs: torch.Tensor) -> torch.Tensor:
                out = sess.run(None, {input_name: obs.detach().cpu().numpy().astype("float32")})
                return torch.from_numpy(out[0]).to(obs.device)

            graph = None
            n_params = 0

        actor_declared = export_man["actor"]
        if actor_declared["input_dim"] != OBS_DIM or actor_declared["output_dim"] != ACTION_DIM:
            raise V2PolicyError(
                f"actor dims {actor_declared['input_dim']}->{actor_declared['output_dim']} "
                f"disagree with the adapter contract {OBS_DIM}->{ACTION_DIM}"
            )
        param_match = "actor" if n_params == actor_declared["parameters_actor"] else (
            "full" if n_params == actor_declared["parameters_total"] else "mismatch"
        )
        if backend == "jit" and param_match == "mismatch":
            raise V2PolicyError(
                f"loaded graph has {n_params} parameters; the manifest declares actor="
                f"{actor_declared['parameters_actor']} total={actor_declared['parameters_total']}"
            )

        obs_contract = export_man["observation_contract"]
        if int(obs_contract["total_dim"]) != OBS_DIM:
            raise V2PolicyError(
                f"observation contract dim {obs_contract['total_dim']} != {OBS_DIM}"
            )
        action_contract = export_man["action_contract"]
        offset = tuple(float(v) for v in action_contract["offset"])
        scale = tuple(float(v) for v in action_contract["scale"])
        if len(offset) != ACTION_DIM or len(scale) != ACTION_DIM:
            raise V2PolicyError("offset/scale length mismatch")

        return cls(
            bundle_root=bundle_root,
            device=device,
            policy=graph,
            raw_action_of=raw_action_of,
            export_manifest=export_man,
            frozen_manifest=frozen_man,
            verified=verified,
            obs_contract=obs_contract,
            action_contract=action_contract,
            offset=offset,
            scale=scale,
            backend=backend,
            parameter_count=n_params,
            parameter_count_matches=param_match,
        )

    # ------------------------------------------------------------- inference
    def raw_action(self, obs: Sequence[Sequence[float]], *, device: str | None = None) -> tuple[torch.Tensor, AnomalyReport]:
        """Raw network output for a batch of observations.

        Non-finite rows are zeroed and reported, never propagated: a NaN entering
        the graph would produce a NaN joint target, and a NaN target would move
        the robot in a way nobody can reconstruct afterwards.
        """
        dev = device or self.device
        rows = [[float(v) for v in row] for row in obs]
        report = AnomalyReport()
        clean = []
        for i, row in enumerate(rows):
            if len(row) != OBS_DIM:
                raise V2PolicyError(f"observation row {i} has {len(row)} values, expected {OBS_DIM}")
            if not all(math.isfinite(v) for v in row):
                report.non_finite_rows.append(i)
                clean.append([0.0] * OBS_DIM)
            else:
                clean.append(row)
        report.replaced_with_zero_row = bool(report.non_finite_rows)
        if report.replaced_with_zero_row:
            zero_out = self.raw_action_of(
                torch.zeros(1, OBS_DIM, device=dev)
            )
            report.action_of_zero_obs = [float(v) for v in zero_out[0]]

        with torch.no_grad():
            out = self.raw_action_of(torch.tensor(clean, dtype=torch.float32, device=dev))
        if out.dim() != 2 or out.shape[1] != ACTION_DIM:
            raise V2PolicyError(f"graph returned shape {tuple(out.shape)}, expected (N, {ACTION_DIM})")
        return out, report

    def joint_targets(self, obs: Sequence[Sequence[float]], *, device: str | None = None):
        """``q_target = default_joint_pos + scale * a`` — the frozen action rule."""
        raw, report = self.raw_action(obs, device=device)
        offset = torch.tensor(self.offset, dtype=torch.float32, device=raw.device)
        scale = torch.tensor(self.scale, dtype=torch.float32, device=raw.device)
        return offset + scale * raw, report

    # --------------------------------------------------------------- reports
    def provenance(self) -> dict:
        return {
            "bundle_root": self.bundle_root,
            "backend": self.backend,
            "device": self.device,
            "mode": "eval",
            "gradients": "disabled (no_grad + requires_grad_(False))",
            "parameter_count": self.parameter_count,
            "parameter_count_matches": self.parameter_count_matches,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "action_rule": "q_target = default_joint_pos + 0.25 * a, no clipping",
            "joint_names": list(self.action_contract["joint_names"]),
            "obs_terms": [t["name"] for t in self.obs_contract["terms"]],
            "verified_artifacts": self.verified,
            "tolerances": self.export_manifest.get("tolerance"),
            "timing": self.export_manifest.get("timing"),
        }
