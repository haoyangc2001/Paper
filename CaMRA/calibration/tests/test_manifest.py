import copy

import pytest

from camra_calibration.manifest import ManifestError, load_manifest, validate_manifest
from camra_calibration.protocol import Protocol

PROTO = Protocol.load()


def make_trial(tid="pilot-straight_shift-clean-0000", **over):
    rec = {
        "trial_id": tid,
        "split": "pilot",
        "family": "straight_shift",
        "profile": "clean",
        "seed": 5000,
        "attempt": 1,
        "controller": "camra_go2_velocity_it2999_v2",
        "protocol_version": PROTO.version,
        "protocol_sha256": PROTO.sha256,
        "bundle_tree_sha256": "d0f0ec3cf0dc56bf7943ff0f7853483123eb82497476b9a5786d5f17a1954d1f",
        "outcome": "completed",
        "steps_executed": 6000,
        "profile_applied": {"friction": 1.0, "added_mass_kg": 0.0, "push_events": [], "obs_noise": "disabled"},
        "initial_state": {"px": 0.1, "py": -0.2, "yaw": 0.3},
        "paths": {
            "physics_200hz": "pilot/t0000/physics_200hz.npz",
            "state_50hz": "pilot/t0000/state_50hz.npz",
            "pairing_10hz": "pilot/t0000/pairing_10hz.npz",
            "energy_summary": "pilot/t0000/energy.json",
        },
        "energy": {"unit": "J", "j_exec": 12.5, "j_nom": 11.9, "j_min": 1.0, "non_computable": False},
    }
    rec.update(over)
    return rec


def test_valid_trial_passes():
    r = validate_manifest([make_trial()], protocol=PROTO)
    assert r.ok, r.errors
    assert r.total == 1 and r.per_split["pilot"] == 1


def test_missing_field_rejected():
    rec = make_trial()
    del rec["energy"]
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and "缺少必备字段" in r.errors[0]


def test_duplicate_trial_id_rejected():
    a = make_trial()
    b = make_trial()
    b["paths"] = {k: v.replace("t0000", "t0001") for k, v in a["paths"].items()}
    r = validate_manifest([a, b], protocol=PROTO)
    assert not r.ok and any("重复" in e for e in r.errors)


def test_bad_split_rejected():
    r = validate_manifest([make_trial(split="training")], protocol=PROTO)
    assert not r.ok and any("split 非法" in e for e in r.errors)


def test_seed_outside_block_rejected():
    r = validate_manifest([make_trial(seed=6000)], protocol=PROTO)  # 6000 属 calibration
    assert not r.ok and any("种子块" in e for e in r.errors)


def test_path_escape_rejected():
    rec = make_trial()
    rec["paths"]["physics_200hz"] = "../escape/physics_200hz.npz"
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("逃逸" in e for e in r.errors)


def test_absolute_path_rejected():
    rec = make_trial()
    rec["paths"]["state_50hz"] = "/abs/state_50hz.npz"
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("绝对路径" in e for e in r.errors)


def test_wrong_suffix_rejected():
    rec = make_trial()
    rec["paths"]["energy_summary"] = "pilot/t0000/energy.npz"
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("后缀" in e for e in r.errors)


def test_reused_path_rejected():
    a = make_trial()
    b = make_trial(tid="pilot-straight_shift-clean-0001", seed=5001)  # 路径与 a 相同
    r = validate_manifest([a, b], protocol=PROTO)
    assert not r.ok and any("复用" in e for e in r.errors)


def test_energy_unit_rejected():
    rec = make_trial()
    rec["energy"]["unit"] = "Wh"
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("unit" in e for e in r.errors)


def test_non_computable_flag_consistency():
    rec = make_trial()
    rec["energy"].update({"j_nom": 0.4, "non_computable": False})
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("non_computable" in e for e in r.errors)
    rec2 = make_trial()
    rec2["energy"].update({"j_nom": 0.4, "non_computable": True})
    assert validate_manifest([rec2], protocol=PROTO).ok


def test_protocol_hash_mismatch_rejected():
    rec = make_trial(protocol_sha256="0" * 64)
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("protocol_sha256" in e for e in r.errors)


def test_completed_must_use_full_window():
    r = validate_manifest([make_trial(steps_executed=1200)], protocol=PROTO)
    assert not r.ok and any("6000" in e for e in r.errors)
    r2 = validate_manifest([make_trial(outcome="terminated", steps_executed=1200)], protocol=PROTO)
    assert r2.ok, r2.errors


def test_trial_id_segments_must_match_fields():
    rec = make_trial()
    rec["family"] = "arc_combined"
    r = validate_manifest([rec], protocol=PROTO)
    assert not r.ok and any("family" in e for e in r.errors)


def test_load_manifest_rejects_bad_json(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"trial_id": "x"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_cli_validate_manifest_exit_codes(tmp_path):
    from camra_calibration.cli import main
    good = tmp_path / "good.jsonl"
    import json
    good.write_text(json.dumps(make_trial(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(["validate-manifest", str(good)]) == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(make_trial(split="nope"), ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(["validate-manifest", str(bad)]) == 2
