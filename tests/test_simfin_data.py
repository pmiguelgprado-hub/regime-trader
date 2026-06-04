"""Tests for the SimFin fundamentals loader (no network — fetch is injected)."""

from __future__ import annotations

import json

import pytest

from data import simfin_data


def _fake_company(url, key):
    assert "ticker=AAPL" in url and key == "test-key"
    return 200, json.dumps([{"ticker": "AAPL", "name": "APPLE INC",
                             "sectorName": "Computer Hardware"}])


def test_company_info_parses_first_record() -> None:
    info = simfin_data.company_info("AAPL", key="test-key", fetch=_fake_company)
    assert info["ticker"] == "AAPL" and info["name"] == "APPLE INC"


def test_statements_builds_params_and_returns_list() -> None:
    seen = {}

    def fake(url, key):
        seen["url"] = url
        return 200, json.dumps([{"ticker": "AAPL", "statements": [
            {"statement": "PL", "data": [{"Fiscal Year": 2023, "Revenue": 383285000000,
                                          "Publish Date": "2023-11-03"}]}]}])

    out = simfin_data.statements("AAPL", statements="PL", period="fy", fyear=2023,
                                 key="test-key", fetch=fake)
    assert "statements=PL" in seen["url"] and "period=fy" in seen["url"]
    assert "fyear=2023" in seen["url"]
    assert out[0]["statements"][0]["data"][0]["Revenue"] == 383285000000


def test_non_200_raises() -> None:
    def fail(url, key):
        return 401, '{"error":"auth"}'

    with pytest.raises(RuntimeError, match="HTTP 401"):
        simfin_data.company_info("AAPL", key="k", fetch=fail)


def test_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("SIMFIN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SIMFIN_API_KEY"):
        simfin_data._api_key()
