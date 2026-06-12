"""Tests for the versioned model registry (A-4).

Filesystem-backed under a tmp root; one real fitted engine is reused across
versions (fitting is slow, content is irrelevant to registry mechanics).
"""

from __future__ import annotations

import logging

import pytest

from conftest import make_synthetic_ohlcv
from core.hmm_engine import HMMConfig, HMMEngine
from core.model_registry import ModelRegistry
from data.feature_engineering import FeatureEngineer

for _n in ("core.hmm_engine",):
    logging.getLogger(_n).setLevel(logging.CRITICAL)


@pytest.fixture(scope="module")
def fitted() -> HMMEngine:
    hmm = HMMEngine(HMMConfig(n_candidates=[3], n_init=1))
    hmm.fit(FeatureEngineer().build_features(make_synthetic_ohlcv()))
    return hmm


def test_save_version_then_load_roundtrips(tmp_path, fitted) -> None:
    """A saved version is listed and loads back as a fitted engine."""
    reg = ModelRegistry(tmp_path)
    v = reg.save_version(fitted, "SPY")
    assert v in reg.versions("SPY")
    reg.promote("SPY", v)
    loaded = reg.load_champion("SPY")
    assert loaded is not None and loaded.n_regimes == fitted.n_regimes


def test_champion_none_when_empty(tmp_path) -> None:
    """No champion before anything is promoted."""
    reg = ModelRegistry(tmp_path)
    assert reg.champion_version("UNKNOWN") is None
    assert reg.load_champion("UNKNOWN") is None


def test_rollback_restores_previous_champion(tmp_path, fitted) -> None:
    """Rollback re-promotes the version before the current champion."""
    reg = ModelRegistry(tmp_path)
    v1 = reg.save_version(fitted, "SPY", version="v1")
    v2 = reg.save_version(fitted, "SPY", version="v2")
    reg.promote("SPY", v2)
    assert reg.champion_version("SPY") == "v2"

    rolled = reg.rollback("SPY")
    assert rolled == "v1"
    assert reg.champion_version("SPY") == "v1"


def test_rollback_noop_with_single_version(tmp_path, fitted) -> None:
    """Nothing to roll back to when only one version exists."""
    reg = ModelRegistry(tmp_path)
    v1 = reg.save_version(fitted, "SPY", version="v1")
    reg.promote("SPY", v1)
    assert reg.rollback("SPY") is None
    assert reg.champion_version("SPY") == "v1"


# --- T0.4 pin-champion: promotion records the champion's transition hash ----------


def test_promote_records_champion_hash(tmp_path, fitted) -> None:
    """Promoting a version persists its transition hash for the daily drift assert."""
    reg = ModelRegistry(tmp_path)
    v = reg.save_version(fitted, "SPY", version="v1")
    reg.promote("SPY", v)
    assert reg.champion_hash("SPY") == fitted.transition_hash()


def test_champion_hash_none_when_no_champion(tmp_path) -> None:
    assert ModelRegistry(tmp_path).champion_hash("SPY") is None


def test_rollback_refreshes_champion_hash(tmp_path, fitted) -> None:
    """After rollback the recorded hash must match the re-promoted version."""
    reg = ModelRegistry(tmp_path)
    reg.save_version(fitted, "SPY", version="v1")
    reg.save_version(fitted, "SPY", version="v2")
    reg.promote("SPY", "v2")
    reg.rollback("SPY")
    assert reg.champion_hash("SPY") == fitted.transition_hash()


def test_load_pinned_champion_bootstraps_once_then_pins(tmp_path, fitted, monkeypatch) -> None:
    """First call adopts/trains the legacy pickle and promotes it; afterwards the
    champion is pinned — even a stale-by-age model must NOT trigger a retrain."""
    import main as m

    reg = ModelRegistry(tmp_path)
    legacy = tmp_path / "hmm_SPY.pkl"
    calls: list[int] = []

    def train() -> None:
        calls.append(1)
        fitted.save(legacy)

    hmm1, sha1 = m.load_pinned_champion(reg, "SPY", legacy, train)
    assert calls == [1]                          # legacy absent -> trained + promoted
    assert sha1 == fitted.transition_hash()
    assert reg.champion_version("SPY") is not None

    # force the old age rule to scream "stale": the pin must ignore it
    monkeypatch.setattr(m, "_needs_retrain", lambda *a, **k: True)
    hmm2, sha2 = m.load_pinned_champion(reg, "SPY", legacy, train)
    assert calls == [1]                          # no retrain: champion is pinned
    assert sha2 == sha1 == hmm2.transition_hash()
