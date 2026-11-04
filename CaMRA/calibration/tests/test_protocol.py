from pathlib import Path

import pytest

from camra_calibration.protocol import Protocol, ProtocolError, default_path


def test_default_protocol_loads_and_is_frozen():
    p = Protocol.load()
    assert p.version == "1.0.2"
    assert p.frozen
    assert p.unit == "J"
    assert p.j_min == 1.0
    assert p.measured_steps == 6000
    assert p.cells == 20
    assert len(p.family_ids()) == 4 and len(p.profile_ids()) == 5


def test_seed_blocks_match_quotas():
    p = Protocol.load()
    for split in ("pilot", "baseline_fit", "baseline_dev", "baseline_validate",
                  "calibration", "heldout", "near_obstacle"):
        start, count = p.seed_block(split)
        assert count == p.quota(split), split


def test_seed_blocks_disjoint_and_avoid_reserved():
    p = Protocol.load()
    blocks = p.payload["seeds"]["blocks"]
    seen = set()
    for spec in blocks.values():
        rng = set(range(spec["start"], spec["start"] + spec["count"]))
        assert not (seen & rng)
        seen |= rng
    assert not (seen & set(p.payload["seeds"]["reserved_excluded"]))
    assert len(seen) == 1820


def test_tampered_hash_rejected(tmp_path: Path):
    src = default_path()
    dst = tmp_path / "calibration_protocol_v1.0.2.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "calibration_protocol_v1.0.2.json.sha256").write_text("0" * 64 + "  x\n", encoding="utf-8")
    with pytest.raises(ProtocolError):
        Protocol.load(dst)


def test_non_frozen_rejected(tmp_path: Path):
    import hashlib
    import json
    pay = json.loads(default_path().read_text(encoding="utf-8"))
    pay["state"] = "draft"
    dst = tmp_path / "calibration_protocol_v1.0.2.json"
    dst.write_text(json.dumps(pay, ensure_ascii=False), encoding="utf-8")
    d = hashlib.sha256(dst.read_bytes()).hexdigest()
    (tmp_path / "calibration_protocol_v1.0.2.json.sha256").write_text(f"{d}  x\n", encoding="utf-8")
    with pytest.raises(ProtocolError):
        Protocol.load(dst)
