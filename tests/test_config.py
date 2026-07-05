"""Tests for config serialization + the reproducibility guard."""

import warnings

import pytest
import yaml

from fx_rl.config import Config


def test_roundtrip_no_warning(tmp_path):
    """A freshly saved config reloads cleanly (fingerprint matches, no warning)."""
    p = tmp_path / "cfg.yaml"
    Config().to_yaml(str(p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        cfg = Config.from_yaml(str(p))
    assert cfg.data_fingerprint() == Config().data_fingerprint()


def test_fingerprint_tracks_data_changes():
    a = Config()
    b = Config()
    assert a.data_fingerprint() == b.data_fingerprint()
    b.data.arb_magnitude = 0.05
    assert a.data_fingerprint() != b.data_fingerprint()


def test_missing_field_warns(tmp_path):
    """An old saved config missing a data-gen field warns about irreproducibility."""
    payload = Config().to_dict()
    del payload["data"]["arb_decay"]  # simulate a config saved before this field existed
    p = tmp_path / "old.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)
    with pytest.warns(UserWarning, match="may NOT reproduce"):
        Config.from_yaml(str(p))


def test_fingerprint_mismatch_warns(tmp_path):
    """A stored fingerprint that doesn't match the rebuilt config warns."""
    payload = Config().to_dict()
    payload["_data_fingerprint"] = "deadbeefcafe"  # wrong on purpose
    p = tmp_path / "drift.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        Config.from_yaml(str(p))


def test_partial_from_dict_is_silent():
    """Programmatic partial overrides must NOT warn (legitimate use)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = Config.from_dict({"train": {"total_timesteps": 5000}})
    assert cfg.train.total_timesteps == 5000
