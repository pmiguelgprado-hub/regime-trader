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
