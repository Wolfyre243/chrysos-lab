"""Black-Scholes greeks (no dividends). Pure math, no data fetching."""

import math

from data.models import OptionSide


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delta(side: OptionSide, spot: float, strike: float, iv: float, t: float, rate: float = 0.0) -> float:
    """Option delta; ``iv`` is an annualised decimal and ``t`` is in years.

    With no time or vol left, delta is the intrinsic step (1 or 0 for a call).
    """
    if iv <= 0 or t <= 0:
        call = 1.0 if spot > strike else 0.0
    else:
        d1 = (math.log(spot / strike) + (rate + iv**2 / 2.0) * t) / (iv * math.sqrt(t))
        call = norm_cdf(d1)
    return call if side == "call" else call - 1.0
