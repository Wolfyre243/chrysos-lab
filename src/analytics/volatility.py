"""Pure volatility helpers shared by calculators. No data fetching here."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def yang_zhang(prices: pd.DataFrame, window: int = 30, trading_periods: int = 252) -> pd.Series:
    """Rolling annualised Yang-Zhang realised volatility from daily OHLC bars.

    Ported as-is from the original calculator, including two departures from the
    textbook estimator:
    - the ``k``-weighted term uses close-to-close returns, where Yang-Zhang uses
      open-to-close returns;
    - variances are sums of squares over ``window - 1`` with no mean subtraction.

    Returns the series with leading NaNs (insufficient history) dropped.
    """
    log_ho = np.log(prices["high"] / prices["open"])
    log_lo = np.log(prices["low"] / prices["open"])
    log_co = np.log(prices["close"] / prices["open"])

    log_oc = np.log(prices["open"] / prices["close"].shift(1))  # overnight gap
    log_cc = np.log(prices["close"] / prices["close"].shift(1))  # close to close

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)  # Rogers-Satchell

    scale = 1.0 / (window - 1.0)
    open_vol = (log_oc**2).rolling(window).sum() * scale
    close_vol = (log_cc**2).rolling(window).sum() * scale
    window_rs = rs.rolling(window).sum() * scale

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs) * np.sqrt(trading_periods)
    return result.dropna()


def total_variance(iv: float, t: float) -> float:
    """Variance accumulated to time ``t`` (years) at annualised vol ``iv``."""
    return iv**2 * t


def forward_variance(iv1: float, t1: float, iv2: float, t2: float) -> float:
    """Annualised variance implied between ``t1`` and ``t2`` (years, ``t2 > t1``).

    Negative when the two vols are inconsistent with any real forward vol.
    """
    return (total_variance(iv2, t2) - total_variance(iv1, t1)) / (t2 - t1)


def term_structure(dtes: Sequence[float], ivs: Sequence[float]) -> Callable[[float], float]:
    """Linear IV-vs-DTE curve, held flat beyond the first and last points."""
    dtes_arr = np.asarray(dtes, dtype=float)
    ivs_arr = np.asarray(ivs, dtype=float)
    order = dtes_arr.argsort()
    dtes_arr, ivs_arr = dtes_arr[order], ivs_arr[order]

    def curve(dte: float) -> float:
        # np.interp clamps to the end values outside the range, matching the original.
        return float(np.interp(dte, dtes_arr, ivs_arr))

    return curve
