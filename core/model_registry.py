"""Versioned HMM model registry with champion pointer + rollback (A-4).

Layout under ``<root>/<symbol>/``:

* ``hmm_<version>.pkl`` — one pickled :class:`~core.hmm_engine.HMMEngine` per
  trained version (versions sort chronologically by name).
* ``champion.txt`` — the version currently promoted to live use.

Enables champion-challenger promotion (keep the old model until a new one proves
itself) and a one-step rollback to the previous champion.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.hmm_engine import HMMEngine

_CHAMPION = "champion.txt"


class ModelRegistry:
    """Filesystem registry of versioned HMM models per symbol."""

    def __init__(self, root: str | Path = "models") -> None:
        """Args: root: directory under which ``<symbol>/`` folders live."""
        self.root = Path(root)

    def _dir(self, symbol: str) -> Path:
        return self.root / symbol

    @staticmethod
    def _new_version() -> str:
        """UTC timestamp + short uuid (sorts chronologically, never collides)."""
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:6]

    def save_version(self, hmm: HMMEngine, symbol: str, version: Optional[str] = None) -> str:
        """Persist a fitted engine as a new version (does NOT promote it).

        Args:
            hmm: Fitted engine.
            symbol: Ticker namespace.
            version: Explicit version id (defaults to a fresh timestamp id).

        Returns:
            The version id written.
        """
        version = version or self._new_version()
        d = self._dir(symbol)
        d.mkdir(parents=True, exist_ok=True)
        hmm.save(d / f"hmm_{version}.pkl")
        return version

    def versions(self, symbol: str) -> list[str]:
        """All version ids for a symbol, sorted chronologically."""
        d = self._dir(symbol)
        if not d.exists():
            return []
        return sorted(p.name[len("hmm_"):-len(".pkl")] for p in d.glob("hmm_*.pkl"))

    def champion_version(self, symbol: str) -> Optional[str]:
        """The currently promoted version, or None."""
        p = self._dir(symbol) / _CHAMPION
        return p.read_text().strip() if p.exists() else None

    def champion_path(self, symbol: str) -> Optional[Path]:
        """Path to the champion pickle, or None."""
        v = self.champion_version(symbol)
        if not v:
            return None
        return self._dir(symbol) / f"hmm_{v}.pkl"

    def promote(self, symbol: str, version: str) -> None:
        """Mark ``version`` as the live champion."""
        d = self._dir(symbol)
        d.mkdir(parents=True, exist_ok=True)
        (d / _CHAMPION).write_text(version)

    def load_champion(self, symbol: str) -> Optional[HMMEngine]:
        """Load the champion engine, or None if absent."""
        p = self.champion_path(symbol)
        if p is None or not p.exists():
            return None
        return HMMEngine.load(p)

    def rollback(self, symbol: str) -> Optional[str]:
        """Promote the version immediately before the current champion.

        Returns:
            The version rolled back to, or None if there is no earlier version.
        """
        vs = self.versions(symbol)
        cur = self.champion_version(symbol)
        if cur not in vs:
            return None
        i = vs.index(cur)
        if i == 0:
            return None
        prev = vs[i - 1]
        self.promote(symbol, prev)
        return prev
