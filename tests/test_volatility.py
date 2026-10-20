import numpy as np
import pandas as pd
import pytest

from analytics.volatility import term_structure, yang_zhang

from conftest import make_history


def _original_yang_zhang(price_data, window=30, trading_periods=252):
    """Verbatim from the original calculator (capitalised yfinance columns)."""
    log_ho = (price_data["High"] / price_data["Open"]).apply(np.log)
    log_lo = (price_data["Low"] / price_data["Open"]).apply(np.log)
    log_co = (price_data["Close"] / price_data["Open"]).apply(np.log)
    log_oc = (price_data["Open"] / price_data["Close"].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    log_cc = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.iloc[-1]


def test_yang_zhang_matches_original():
    history = make_history()
    expected = _original_yang_zhang(history.rename(columns=str.capitalize))
    assert yang_zhang(history).iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_yang_zhang_drops_warmup_rows():
    history = make_history(days=40)
    # First log return needs a previous close, so 30-bar windows start at bar 31.
    assert len(yang_zhang(history, window=30)) == 40 - 30


def test_yang_zhang_empty_when_history_too_short():
    assert yang_zhang(make_history(days=20)).empty


def test_term_structure_interpolates_and_clamps():
    curve = term_structure([25, 4, 46], [0.45, 0.80, 0.35])  # unsorted input
    assert curve(4) == pytest.approx(0.80)
    assert curve(30) == pytest.approx(0.45 - 0.10 * 5 / 21)
    assert curve(1) == pytest.approx(0.80)  # before first point: flat
    assert curve(90) == pytest.approx(0.35)  # after last point: flat


def test_term_structure_accepts_pandas():
    curve = term_structure(pd.Series([10, 20]), pd.Series([0.5, 0.3]))
    assert curve(15) == pytest.approx(0.4)
