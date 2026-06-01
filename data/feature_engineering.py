"""Feature engineering: technical indicators and HMM feature computation.

Computes returns, volatility, volume, trend, mean-reversion, momentum, and
range features from OHLCV, then standardizes them with **trailing** rolling
z-scores. Every transform is strictly causal: the value at bar ``t`` uses only
data up to and including ``t``. This is a hard requirement — any leakage here
silently corrupts the HMM's filtered inference and every backtest downstream.

All feature computations are exposed as pure functions; ``FeatureEngineer``
is a thin stateful wrapper used by the live loop and backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# ---------------------------------------------------------------------------
# Canonical feature column order. The HMM consumes these columns in this order;
# index positions of ``ret_1`` and ``rvol_20`` are used for regime labelling.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "ret_1",        # log return, 1 period
    "ret_5",        # log return, 5 periods
    "ret_20",       # log return, 20 periods
    "rvol_20",      # realized volatility, 20-period rolling std of ret_1
    "vol_ratio",    # 5-period std / 20-period std (vol-of-vol regime)
    "vol_z",        # volume z-score vs 50-period mean
    "vol_trend",    # slope of 10-period volume SMA
    "adx_14",       # trend strength (ADX, 14)
    "sma50_slope",  # slope of 50-period close SMA
    "rsi_14",       # RSI(14) (standardized downstream)
    "dist_sma200",  # (close - SMA200) / close
    "roc_10",       # rate of change, 10 periods
    "roc_20",       # rate of change, 20 periods
    "natr_14",      # normalized ATR (ATR14 / close)
]

RET_1_IDX: int = FEATURE_COLUMNS.index("ret_1")
RVOL_20_IDX: int = FEATURE_COLUMNS.index("rvol_20")


# ===========================================================================
# Pure feature functions (causal)
# ===========================================================================
def log_return(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return over ``period`` bars: ``ln(close_t / close_{t-period})``.

    Args:
        close: Close-price series.
        period: Lookback in bars.

    Returns:
        Log-return series (first ``period`` values NaN).
    """
    return np.log(close / close.shift(period))


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Trailing rolling standard deviation of returns.

    Args:
        returns: Return series (typically 1-period log returns).
        window: Rolling window length.

    Returns:
        Rolling realized-volatility series.
    """
    return returns.rolling(window).std()


def vol_ratio(returns: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Ratio of short- to long-window realized volatility.

    Values > 1 indicate volatility expansion; < 1 indicate contraction.

    Args:
        returns: Return series.
        fast: Short window.
        slow: Long window.

    Returns:
        Volatility-ratio series.
    """
    return returns.rolling(fast).std() / returns.rolling(slow).std()


def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """Z-score of volume vs its trailing rolling mean/std.

    Args:
        volume: Volume series.
        window: Rolling window length.

    Returns:
        Volume z-score series.
    """
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Trailing rolling OLS slope of ``series`` against a time index.

    Slope of the best-fit line over the last ``window`` points, computed in
    closed form: ``cov(t, y) / var(t)``. Causal — uses only the trailing
    window ending at each bar.

    Args:
        series: Input series.
        window: Rolling window length.

    Returns:
        Rolling-slope series.
    """
    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    t_dev = t - t_mean
    denom = float((t_dev**2).sum())

    def _slope(y: np.ndarray) -> float:
        return float((t_dev * (y - y.mean())).sum() / denom)

    return series.rolling(window).apply(_slope, raw=True)


def volume_trend(volume: pd.Series, sma_window: int = 10) -> pd.Series:
    """Slope of the ``sma_window``-period volume SMA.

    Args:
        volume: Volume series.
        sma_window: SMA window whose slope is measured.

    Returns:
        Volume-trend (SMA slope) series.
    """
    sma = volume.rolling(sma_window).mean()
    return rolling_slope(sma, sma_window)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index (trend strength), via ``ta``.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: ADX window.

    Returns:
        ADX series.
    """
    return ADXIndicator(high=high, low=low, close=close, window=window).adx()


def sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
    """Slope of the ``window``-period close SMA.

    Args:
        close: Close-price series.
        window: SMA window.

    Returns:
        SMA-slope series.
    """
    sma = close.rolling(window).mean()
    return rolling_slope(sma, window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, via ``ta``.

    Args:
        close: Close-price series.
        window: RSI window.

    Returns:
        RSI series (0..100).
    """
    return RSIIndicator(close=close, window=window).rsi()


def distance_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """Distance of close from its SMA, as a fraction of price.

    Args:
        close: Close-price series.
        window: SMA window.

    Returns:
        ``(close - SMA) / close`` series.
    """
    sma = close.rolling(window).mean()
    return (close - sma) / close


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change over ``period`` bars, via ``ta``.

    Args:
        close: Close-price series.
        period: Lookback in bars.

    Returns:
        ROC series.
    """
    return ROCIndicator(close=close, window=period).roc()


def normalized_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """ATR normalized by price: ``ATR(window) / close``.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: ATR window.

    Returns:
        Normalized-ATR series.
    """
    atr = AverageTrueRange(
        high=high, low=low, close=close, window=window
    ).average_true_range()
    return atr / close


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Trailing rolling z-score: ``(x - mean) / std`` over ``window``.

    Args:
        series: Input series.
        window: Rolling window length.

    Returns:
        Rolling z-score series.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_raw_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all raw (un-standardized) features from OHLCV.

    Args:
        ohlcv: DataFrame with columns ``open, high, low, close, volume``,
            indexed by timestamp.

    Returns:
        DataFrame with columns == :data:`FEATURE_COLUMNS`, same index.
    """
    close, high, low, volume = (
        ohlcv["close"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["volume"],
    )
    r1 = log_return(close, 1)
    feats = {
        "ret_1": r1,
        "ret_5": log_return(close, 5),
        "ret_20": log_return(close, 20),
        "rvol_20": realized_volatility(r1, 20),
        "vol_ratio": vol_ratio(r1, 5, 20),
        "vol_z": volume_zscore(volume, 50),
        "vol_trend": volume_trend(volume, 10),
        "adx_14": adx(high, low, close, 14),
        "sma50_slope": sma_slope(close, 50),
        "rsi_14": rsi(close, 14),
        "dist_sma200": distance_from_sma(close, 200),
        "roc_10": roc(close, 10),
        "roc_20": roc(close, 20),
        "natr_14": normalized_atr(high, low, close, 14),
    }
    return pd.DataFrame(feats, index=ohlcv.index)[FEATURE_COLUMNS]


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""

    zscore_window: int = 252
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


class FeatureEngineer:
    """Builds the causal, standardized feature matrix for regime detection."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the feature engineer.

        Args:
            config: Feature windows and column selection (defaults applied).
        """
        self.config = config or FeatureConfig()

    def build_features(self, ohlcv: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Compute and standardize the full feature matrix.

        Pipeline: raw features -> trailing rolling z-score (``zscore_window``)
        -> drop warmup rows. Both stages are strictly causal.

        Args:
            ohlcv: OHLCV DataFrame indexed by timestamp.
            dropna: If True, drop warmup rows containing any NaN.

        Returns:
            Standardized feature matrix indexed by timestamp.
        """
        raw = compute_raw_features(ohlcv)
        std = raw.apply(lambda col: rolling_zscore(col, self.config.zscore_window))
        std = std[self.config.feature_columns]
        return std.dropna() if dropna else std

    def is_trending(self, ohlcv: pd.DataFrame, adx_threshold: float = 25.0) -> bool:
        """Whether a tradable trend is present on the latest bar.

        Uses raw ADX(14): values above ``adx_threshold`` indicate a trend.

        Args:
            ohlcv: OHLCV DataFrame.
            adx_threshold: ADX level above which a trend is considered present.

        Returns:
            True if the latest ADX exceeds the threshold.
        """
        adx_series = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna()
        if adx_series.empty:
            return False
        return bool(adx_series.iloc[-1] >= adx_threshold)

    def assert_no_lookahead(
        self, ohlcv: pd.DataFrame, probe_index: int | None = None, atol: float = 1e-9
    ) -> None:
        """Assert feature values do not depend on future data.

        Recomputes features on a prefix ending at ``probe_index`` and on the
        full series, then checks the feature row at ``probe_index`` is
        identical (aligned by index label, not position).

        Args:
            ohlcv: OHLCV DataFrame.
            probe_index: Positional bar to probe (defaults to mid-series).
            atol: Absolute tolerance for the comparison.

        Raises:
            AssertionError: If any feature at the probe bar changes when
                future bars are appended (i.e. look-ahead leakage).
        """
        n = len(ohlcv)
        if probe_index is None:
            probe_index = n // 2
        ts = ohlcv.index[probe_index]

        full = self.build_features(ohlcv, dropna=False)
        prefix = self.build_features(ohlcv.iloc[: probe_index + 1], dropna=False)

        if ts not in prefix.index or ts not in full.index:
            raise AssertionError("probe bar missing from feature frame")
        a = full.loc[ts].to_numpy(dtype=float)
        b = prefix.loc[ts].to_numpy(dtype=float)
        # NaN positions must match; finite values must be ~equal.
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError("LOOK-AHEAD: NaN pattern differs at probe bar")
        mask = ~np.isnan(a)
        if not np.allclose(a[mask], b[mask], atol=atol):
            raise AssertionError("LOOK-AHEAD BIAS DETECTED in features")
