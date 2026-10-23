"""Pure helpers for picking quotes out of option chains. No data fetching here."""

import math
from collections.abc import Collection, Iterable

import pandas as pd

from analytics.greeks import delta
from data.models import SIDES, OptionChain, OptionSide
from errors import CalculatorError

# Yahoo reports IV as 0.00001 when it has no live quotes; no real ATM option trades this low.
STALE_IV_FLOOR = 1e-4


def nearest_quote(side: pd.DataFrame, target: float) -> pd.Series:
    """The row of a calls or puts frame whose strike is closest to ``target``."""
    return side.loc[(side["strike"] - target).abs().idxmin()]


def is_stale(quote: pd.Series) -> bool:
    """True when a quote has been cleared: both bid and ask zero, or IV at the floor."""
    return (quote["bid"] == 0 and quote["ask"] == 0) or float(quote["iv"]) <= STALE_IV_FLOOR


def stale_error(where: str) -> CalculatorError:
    """Error for quotes that failed is_stale; ``where`` names them, e.g. an expiry date."""
    return CalculatorError(
        f"Stale option quotes for {where}: bid/ask are zero or IV is at the provider's floor. "
        "Yahoo clears option quotes overnight; try again after the market opens."
    )


def mean_iv(*quotes: pd.Series) -> float:
    return sum(float(q["iv"]) for q in quotes) / len(quotes)


def mid(quote: pd.Series) -> float:
    return (float(quote["bid"]) + float(quote["ask"])) / 2.0


def spread_pct(quote: pd.Series) -> float:
    """Bid-ask spread as a fraction of mid; inf when there is no positive mid."""
    m = mid(quote)
    return (float(quote["ask"]) - float(quote["bid"])) / m if m > 0 else math.inf


def quote_at_delta(
    side_frame: pd.DataFrame,
    side: OptionSide,
    spot: float,
    t: float,
    target: float,
    strikes: Collection[float],
    rate: float = 0.0,
) -> pd.Series | None:
    """The row, among ``strikes``, whose delta is closest to ``target`` (e.g. 0.35 or -0.35)."""
    rows = side_frame[side_frame["strike"].isin(strikes)]
    deltas = pd.Series(
        [delta(side, spot, float(k), float(iv), t, rate) for k, iv in zip(rows["strike"], rows["iv"])],
        index=rows.index,
    ).dropna()
    if deltas.empty:
        return None
    return rows.loc[(deltas - target).abs().idxmin()]


def common_strikes(
    chains: Iterable[OptionChain], spot: float, n: int, sides: Iterable[OptionSide] = SIDES
) -> list[float]:
    """The ``n`` strikes nearest ``spot`` listed on every given side of every chain, ascending."""
    sides = tuple(sides)
    strikes: set[float] | None = None
    for chain in chains:
        listed = set.intersection(*(set(chain.side(side)["strike"]) for side in sides))
        strikes = listed if strikes is None else strikes & listed
    nearest = sorted(strikes or (), key=lambda k: (abs(k - spot), k))[:n]
    return sorted(nearest)
