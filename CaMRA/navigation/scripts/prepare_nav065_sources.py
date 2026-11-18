"""Bind the six final training checkpoints before deployment value validation."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
from zipfile import Path as ZipPath

import numpy as np
import torch

from camra_navigation.algorithm.validation import file_sha
from eval_r14_repair import build_registered_network
from train_formal import verify_prereg

NAV = Path(__file__).resolve().parents[1]
REGISTRATION = "training_loop_prereg_terminal_tail_v1.json"


def read(path):
    return json.loads(path.read_bytes())


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        data = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(data.dtype), list(data.shape)]).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname().split(".")[0] == "tiger11" and os.environ.get("TMUX")
    torch.set_num_threads(1)
    evidence = NAV / "reports/nav064_terminal_tail_acceptance_v1.json"
    accepted = read(evidence)
    prereg, pin = verify_prereg(REGISTRATION)
    assert accepted["verdict"] == "PASS" and accepted["prereg_sha256"] == pin
    dev_path = NAV / "reports/nav064_terminal_tail_v1_development.json"
    assert file_sha(dev_path) == accepted["evidence"][str(dev_path)]
    dev = {row["run_id"]: row for row in read(dev_path)["runs"]}
    nav_path = NAV / prereg["binding"]["navigation_contract"]["path"]
    nav = read(nav_path)
    value_path = NAV / prereg["binding"]["value_contract"]["path"]
    value = read(value_path)
    old_paths = [
        NAV / "delivery/deployment_candidates_v1.json",
        NAV / "reports/deployment_critic_validation.json",
        NAV / "reports/deployment_critic_validation_rejudge.json",
        NAV / "reports/gate_G4.json",
    ]
    rows = []
    for identity, source in zip(prereg["run_registry"], accepted["runs"], strict=True):
        assert all(source[k] == v for k, v in identity.items())
        path = Path(source["source_checkpoint"])
        assert file_sha(path) == source["state_sha256"] == dev[identity["run_id"]]["state_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["update_idx"] + 1 == prereg["window_semantics"]["max_updates"]
        assert state["variant"] == identity["variant"] and state["seed"] == identity["seed"]
        assert state["prereg_sha256"] == pin
        assert state["implementation_versions"] == prereg["implementation_versions"]
        model = build_registered_network(nav, prereg)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        obs_rows = [sample for ep in dev[identity["run_id"]]["episodes"] for sample in ep["trace"]]
        obs = torch.from_numpy(np.asarray([r["observation"] for r in obs_rows], dtype=np.float32))
        expected = torch.from_numpy(np.asarray([r["action"] for r in obs_rows], dtype=np.float32))
        with torch.inference_mode():
            action, _, _ = model.act(obs, deterministic=True)
        error = float((action - expected).abs().max())
        assert error <= 1e-6, (identity["run_id"], error)
        rows.append(dict(identity, checkpoint=str(path), checkpoint_sha256=source["state_sha256"],
                         update_idx=state["update_idx"],
                         state_dict_sha256={role: tensor_digest(getattr(model, role)) for role in model.ROLES},
                         action_replay={"count": len(obs_rows), "max_abs_error": error}))

    assert len(rows) == 6 and len({r["run_id"] for r in rows}) == 6
    report = {
        "task": "NAV-065", "status": "SOURCE_BOUND_PENDING_VALUE_VALIDATION",
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "host": socket.getfqdn(), "tmux": os.environ["TMUX"],
        "source_evidence": {"path": str(evidence), "sha256": file_sha(evidence)},
        "prereg_sha256": pin, "implementation_versions": prereg["implementation_versions"],
        "contracts": {str(p): file_sha(p) for p in (nav_path, value_path)},
        "value_contract_version": value["version"],
        "action": "eval_r14_repair.build_registered_network; act(deterministic=True), bounded mean",
        "avoidance_prior": prereg["avoidance_prior"],
        "protected_previous_results": {str(p): file_sha(p) for p in old_paths},
        "candidates": rows,
        "limits": ["No critic predictions or new rollouts; this is not G4 or a production bundle."],
    }
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Bound six candidate sources; replayed {sum(r['action_replay']['count'] for r in rows)} actions")


if __name__ == "__main__":
    main()
