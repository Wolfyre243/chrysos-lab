"""Calendar-spread IV crush screen.

Judges whether a stock suits a pre-earnings calendar spread that profits from the
post-earnings drop in front-month implied volatility. Three checks:

- avg_volume:    30-day average share volume is high enough to trade.
- iv30_rv30:     30-day implied vol is rich versus 30-day realised (Yang-Zhang) vol.
- ts_slope_0_45: the IV term structure slopes down from the front expiry to 45 DTE
                 (front month inflated relative to later months).

Verdict: all three pass -> Recommended; the slope passes and exactly one of the
other two passes -> Consider; otherwise Avoid.

Ported from a third-party GUI calculator. ``evaluate`` is pure (no network) and is
the function to call from other code; ``run`` fetches data through a
MarketDataProvider and then calls ``evaluate``.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

from analytics.volatility import term_structure, yang_zhang
from calculators.spec import CalculatorSpec
from data.models import OptionChain
from data.provider import MarketDataProvider
from errors import CalculatorError

SPEC = CalculatorSpec(
    id="calendar_iv_crush",
    name="Calendar IV Crush",
    version=1,
    scope="stock",
    requires=("chain", "price_history"),
)

Recommendation = Literal["Recommended", "Consider", "Avoid"]

# Yahoo reports IV as 0.00001 when it has no live quotes; no real ATM option trades this low.
STALE_IV_FLOOR = 1e-4


@dataclass(frozen=True)
class Params:
    """Thresholds and lookbacks. Defaults are the original calculator's values."""

    min_avg_volume: float = 1_500_000
    min_iv30_rv30: float = 1.25
    max_ts_slope_0_45: float = -0.00406
    max_dte: int = 45  # far end of the slope and of the expiries considered
    window: int = 30  # days for IV30, RV30 and average volume
    history_period: str = "3mo"  # price history fetched by run()


@dataclass(frozen=True)
class Inputs:
    """Everything evaluate() needs. Build it from any data source."""

    symbol: str
    as_of: date
    spot: float
    chains: list[OptionChain]
    price_history: pd.DataFrame  # daily bars with open, high, low, close, volume


@dataclass(frozen=True)
class Check:
    value: float
    threshold: float
    op: Literal[">=", "<="]
    passed: bool


@dataclass(frozen=True)
class TermPoint:
    expiry: date
    dte: int
    atm_iv: float


@dataclass(frozen=True)
class Result:
    symbol: str
    as_of: date
    spot: float
    recommendation: Recommendation
    avg_volume: Check
    iv30_rv30: Check
    ts_slope_0_45: Check
    iv30: float
    rv30: float
    front_expiry: date
    straddle: float | None  # front-expiry ATM call mid + put mid
    expected_move: float | None  # straddle / spot, as a fraction
    term_structure: list[TermPoint] = field(default_factory=list)
    calculator_version: int = SPEC.version

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (dates as ISO strings)."""
        return _jsonable(self)


def select_expiries(expiries: Iterable[date], as_of: date, max_dte: int = 45) -> list[date]:
    """Sorted expiries up to and including the first one at least ``max_dte`` days out.

    An expiry falling on ``as_of`` itself is dropped.
    """
    cutoff = as_of + timedelta(days=max_dte)
    ordered = sorted(expiries)
    for i, expiry in enumerate(ordered):
        if expiry >= cutoff:
            selected = ordered[: i + 1]
            return selected[1:] if selected[0] == as_of else selected
    raise CalculatorError("Not enough option data.")


def evaluate(inputs: Inputs, params: Params = Params()) -> Result:
    """Run the screen on already-fetched data. Pure: no network, no clock."""
    wanted = set(select_expiries((c.expiry for c in inputs.chains), inputs.as_of, params.max_dte))
    chains = sorted((c for c in inputs.chains if c.expiry in wanted), key=lambda c: c.expiry)

    points: list[TermPoint] = []
    straddle: float | None = None
    for chain in chains:
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            continue

        call = calls.loc[(calls["strike"] - inputs.spot).abs().idxmin()]
        put = puts.loc[(puts["strike"] - inputs.spot).abs().idxmin()]
        if _is_stale(call) or _is_stale(put):
            raise CalculatorError(
                f"Stale option quotes for {chain.expiry.isoformat()}: ATM bid/ask are zero or IV is at "
                "the provider's floor. Yahoo clears option quotes overnight; try again after the market opens."
            )
        atm_iv = (float(call["iv"]) + float(put["iv"])) / 2.0
        points.append(TermPoint(chain.expiry, (chain.expiry - inputs.as_of).days, atm_iv))

        if len(points) == 1:
            straddle = _straddle(call, put)

    if not points:
        raise CalculatorError("Could not determine ATM IV for any expiration dates.")
    if len(points) < 2:
        raise CalculatorError("Not enough option data to build a term structure.")

    front = points[0]
    if front.dte == params.max_dte:
        raise CalculatorError(f"Front expiry is exactly {params.max_dte} DTE; slope is undefined.")

    curve = term_structure([p.dte for p in points], [p.atm_iv for p in points])
    ts_slope = (curve(params.max_dte) - curve(front.dte)) / (params.max_dte - front.dte)

    history = inputs.price_history
    rv_series = yang_zhang(history, window=params.window)
    volume_series = history["volume"].rolling(params.window).mean().dropna()
    if rv_series.empty or volume_series.empty:
        raise CalculatorError("Not enough price history.")

    iv30 = curve(params.window)
    rv30 = float(rv_series.iloc[-1])
    avg_volume = float(volume_series.iloc[-1])

    avg_volume_check = _at_least(avg_volume, params.min_avg_volume)
    iv30_rv30_check = _at_least(iv30 / rv30, params.min_iv30_rv30)
    ts_slope_check = _at_most(ts_slope, params.max_ts_slope_0_45)

    return Result(
        symbol=inputs.symbol,
        as_of=inputs.as_of,
        spot=inputs.spot,
        recommendation=_recommend(avg_volume_check.passed, iv30_rv30_check.passed, ts_slope_check.passed),
        avg_volume=avg_volume_check,
        iv30_rv30=iv30_rv30_check,
        ts_slope_0_45=ts_slope_check,
        iv30=iv30,
        rv30=rv30,
        front_expiry=front.expiry,
        straddle=straddle,
        expected_move=straddle / inputs.spot if straddle else None,
        term_structure=points,
    )


def run(
    symbol: str,
    provider: MarketDataProvider | None = None,
    params: Params = Params(),
    as_of: date | None = None,
) -> Result:
    """Fetch data for ``symbol`` through ``provider`` (default: yfinance) and evaluate it."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise CalculatorError("No stock symbol provided.")
    if provider is None:
        from data.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider()
    as_of = as_of or date.today()

    expiries = provider.expirations(symbol)
    if not expiries:
        raise CalculatorError(f"No options found for stock symbol '{symbol}'.")
    selected = select_expiries(expiries, as_of, params.max_dte)
    chains = [provider.option_chain(symbol, expiry) for expiry in selected]

    spot = provider.spot(symbol)
    if spot is None:
        raise CalculatorError("Unable to retrieve underlying stock price.")

    history = provider.price_history(symbol, params.history_period)
    return evaluate(Inputs(symbol, as_of, spot, chains, history), params)


def _is_stale(quote: pd.Series) -> bool:
    """True when a quote has been cleared: both bid and ask zero, or IV at the floor."""
    return (quote["bid"] == 0 and quote["ask"] == 0) or float(quote["iv"]) <= STALE_IV_FLOOR


def _straddle(call: pd.Series, put: pd.Series) -> float | None:
    """ATM call mid + put mid; None if any quote is missing or the total is zero."""
    quotes = [call["bid"], call["ask"], put["bid"], put["ask"]]
    if any(q is None or pd.isna(q) for q in quotes):
        return None
    call_mid = (float(call["bid"]) + float(call["ask"])) / 2.0
    put_mid = (float(put["bid"]) + float(put["ask"])) / 2.0
    return (call_mid + put_mid) or None


def _at_least(value: float, threshold: float) -> Check:
    return Check(value, threshold, ">=", value >= threshold)


def _at_most(value: float, threshold: float) -> Check:
    return Check(value, threshold, "<=", value <= threshold)


def _recommend(avg_volume: bool, iv30_rv30: bool, ts_slope_0_45: bool) -> Recommendation:
    if avg_volume and iv30_rv30 and ts_slope_0_45:
        return "Recommended"
    if ts_slope_0_45 and avg_volume != iv30_rv30:
        return "Consider"
    return "Avoid"


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {name: _jsonable(getattr(obj, name)) for name in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
