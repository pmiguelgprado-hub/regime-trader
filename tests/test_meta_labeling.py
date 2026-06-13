"""Tests for the triple-barrier meta-labeling pipeline (T3.1, Lopez de Prado).

Build the LABELING now so labels accumulate during the gates; the secondary
model (sizing-only, on a new prereg) trains later at >=200 round-trips. The
labeler tags each entry by which barrier it hits first: profit-take (+1),
stop-loss (-1), or the vertical time barrier (0). Pure + unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import meta_labeling as ml


def _prices(vals):
    return pd.Series(vals, index=pd.bdate_range("2026-01-01", periods=len(vals)))


def test_profit_take_hit_first_labels_plus_one():
    px = _prices([100, 101, 103, 106])             # +6% within horizon
    lab = ml.triple_barrier_label(px, entry_i=0, pt=0.05, sl=0.05, max_hold=5)
    assert lab["label"] == 1 and lab["barrier"] == "pt"


def test_stop_loss_hit_first_labels_minus_one():
    px = _prices([100, 99, 96, 94])                # -6% within horizon
    lab = ml.triple_barrier_label(px, entry_i=0, pt=0.05, sl=0.05, max_hold=5)
    assert lab["label"] == -1 and lab["barrier"] == "sl"


def test_time_barrier_labels_zero():
    px = _prices([100, 100.5, 100.2, 100.8, 100.1])  # neither barrier in 3 bars
    lab = ml.triple_barrier_label(px, entry_i=0, pt=0.05, sl=0.05, max_hold=3)
    assert lab["label"] == 0 and lab["barrier"] == "time"


def test_pt_before_sl_when_both_would_hit():
    px = _prices([100, 106, 90])                   # +6% on bar 1 (pt) before -10%
    lab = ml.triple_barrier_label(px, entry_i=0, pt=0.05, sl=0.05, max_hold=5)
    assert lab["barrier"] == "pt"


def test_label_events_returns_one_row_per_event():
    px = _prices([100, 101, 106, 95, 96, 102, 108])
    events = [0, 3]
    df = ml.label_events(px, events, pt=0.05, sl=0.05, max_hold=3)
    assert len(df) == 2
    assert set(df.columns) >= {"entry_date", "label", "barrier", "ret", "holding_days"}


def test_append_labels_accumulates(tmp_path):
    p = str(tmp_path / "labels.csv")
    ml.append_labels(p, [{"entry_date": "2026-06-01", "symbol": "AAPL", "label": 1,
                          "barrier": "pt", "ret": 0.05, "holding_days": 2}])
    ml.append_labels(p, [{"entry_date": "2026-06-02", "symbol": "MSFT", "label": -1,
                          "barrier": "sl", "ret": -0.05, "holding_days": 3}])
    df = pd.read_csv(p)
    assert len(df) == 2 and set(df["symbol"]) == {"AAPL", "MSFT"}


def test_append_labels_dedups_same_entry(tmp_path):
    p = str(tmp_path / "labels.csv")
    row = {"entry_date": "2026-06-01", "symbol": "AAPL", "label": 1,
           "barrier": "pt", "ret": 0.05, "holding_days": 2}
    assert ml.append_labels(p, [row]) == 1
    assert ml.append_labels(p, [row]) == 0          # same (date, symbol) -> skipped
    assert ml.label_count(p) == 1


def test_label_count_and_readiness(tmp_path):
    p = str(tmp_path / "labels.csv")
    rows = [{"entry_date": f"2026-06-{i:02d}", "symbol": "X", "label": 1,
             "barrier": "pt", "ret": 0.05, "holding_days": 1} for i in range(1, 11)]
    ml.append_labels(p, rows)
    assert ml.label_count(p) == 10
    assert ml.ready_to_train(p, min_labels=200) is False
    assert ml.ready_to_train(p, min_labels=5) is True
