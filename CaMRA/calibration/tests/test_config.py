from pathlib import Path

import pytest

from camra_calibration.config import Config, ConfigError, DEFAULTS, parse_env_file


def test_env_overrides_file_and_default(tmp_path: Path):
    re = tmp_path / "repro.env"
    re.write_text("CAMRA_LOG_ROOT=/from/file\nCAMRA_ASSET_ROOT=/from/file/assets\nCAMRA_DEVICE=cuda:1\n",
                  encoding="utf-8")
    cfg = Config.load(environ={"CAMRA_LOG_ROOT": "/from/env"}, repro_env=re)
    assert str(cfg.log_root) == "/from/env"
    assert cfg.sources["CAMRA_LOG_ROOT"] == "env"
    assert str(cfg.asset_root) == "/from/file/assets"
    assert cfg.sources["CAMRA_ASSET_ROOT"] == "repro.env"
    assert cfg.device == "cuda:1"


def test_defaults_when_nothing_given(tmp_path: Path):
    re = tmp_path / "repro.env"
    re.write_text("# empty\n", encoding="utf-8")
    cfg = Config.load(environ={}, repro_env=re)
    assert str(cfg.log_root) == DEFAULTS["CAMRA_LOG_ROOT"]
    assert cfg.sources["CAMRA_LOG_ROOT"] == "default"


def test_bad_device_rejected(tmp_path: Path):
    re = tmp_path / "repro.env"
    re.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config.load(environ={"CAMRA_DEVICE": "gpu0"}, repro_env=re)


def test_missing_repro_env_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        Config.load(environ={}, repro_env=tmp_path / "nope.env")


def test_parse_env_file_ignores_comments_and_quotes(tmp_path: Path):
    p = tmp_path / "repro.env"
    p.write_text('# c\nA=1\n\nB="/x y"\nC=\'z\'\n', encoding="utf-8")
    vals = parse_env_file(p)
    assert vals == {"A": "1", "B": "/x y", "C": "z"}


def test_frozen_bundle_derived(tmp_path: Path):
    re = tmp_path / "repro.env"
    re.write_text("", encoding="utf-8")
    cfg = Config.load(environ={"CAMRA_BUNDLE_ROOT": "/b"}, repro_env=re)
    assert str(cfg.frozen_bundle) == "/b/frozen/camra_go2_velocity_it2999_v2"
