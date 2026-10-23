"""Forward factor scan: find the best calendar trades for a ticker.

Picks front expiries near 30 DTE and back expiries near 60 DTE and, for every
front/back pair, scores three calendar structures (same strikes on both legs):

- call: call calendar at each of the strikes nearest spot; the one nearest spot is
  marked ATM, the default structure.
- put:  put calendar at the same kind of strikes.
- double_35d: call calendar at the +35 delta strike plus put calendar at the -35
  delta strike (deltas from the front expiry); scored on the averaged wing IVs.

A trade passes when FF >= min_ff, no earnings fall between today and its back
expiry, and every leg's bid-ask spread is within max_spread of mid. Trades blocked
by earnings or spread rank below the rest; each group is sorted by FF, highest
first. Each trade also carries a close-by date: the weekday before the front expiry.

A DTE window is widened only when no listed expiry falls inside it (e.g. names with
monthly expiries only), never because no trade passes.

The FF formula lives in ``calculators.forward_factor``; this module only chooses
what to feed it. ``scan`` is pure; ``run`` fetches through a MarketDataProvider.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from analytics.chain import (
    common_strikes,
    is_stale,
    mean_iv,
    nearest_quote,
    quote_at_delta,
    spread_pct,
    stale_error,
)
from calculators import forward_factor
from calculators.results import Check, Serializable, at_most
from data.fetch import require_expirations, require_spot, resolve
from data.market_calendar import market_date, previous_trading_day
from data.models import SIDES, OptionChain, OptionSide
from data.provider import MarketDataProvider
from errors import CalculatorError

Window = tuple[int, int]  # inclusive DTE range
Structure = Literal["call", "put", "double_35d"]
STRUCTURES: tuple[Structure, ...] = ("call", "put", "double_35d")


@dataclass(frozen=True)
class Params:
    front_window: Window = (21, 45)  # around 30 DTE
    back_window: Window = (50, 90)  # around 60 DTE
    widen_step: int = 7  # days added to each end of an empty window
    max_widen_steps: int = 6
    strikes: int = 5  # strikes nearest spot checked per pair, per side
    max_spread: float = 0.10  # widest leg bid-ask allowed, as a fraction of mid
    double_delta: float = 0.35  # wing delta of the double calendar
    rate: float = 0.0  # risk-free rate used for deltas
    calculator: forward_factor.Params = forward_factor.Params()


@dataclass(frozen=True)
class Inputs:
    symbol: str
    as_of: date
    spot: float
    chains: list[OptionChain]
    earnings: date | None = None  # next earnings date on or after as_of


@dataclass(frozen=True)
class Legs:
    front: list[date]
    back: list[date]
    front_window: Window
    back_window: Window
    widened: bool


@dataclass(frozen=True)
class Candidate:
    structure: Structure
    front_expiry: date
    front_dte: int
    back_expiry: date
    back_dte: int
    strikes: tuple[float, ...]  # one strike; (put, call) for double_35d
    atm: bool  # the ATM call calendar of its pair
    close_by: date  # weekday before the front expiry
    spread: Check  # widest leg bid-ask, as a fraction of mid
    crosses_earnings: bool  # earnings between as_of and the back expiry
    result: forward_factor.Result
    flags: tuple[str, ...]  # failed rules: "ff", "earnings", "spread"
    passed: bool


@dataclass(frozen=True)
class ScanResult(Serializable):
    symbol: str
    as_of: date
    spot: float
    front_window: Window
    back_window: Window
    widened: bool
    min_ff: float
    max_spread: float
    earnings: date | None
    candidates: list[Candidate] = field(default_factory=list)  # best first
    skipped_stale: int = 0  # trades dropped because a quote was stale


def pick_legs(expiries: Iterable[date], as_of: date, params: Params = Params()) -> Legs:
    """Expiries inside the front and back DTE windows, widening a window while it is empty.

    Back expiries must fall after the earliest front expiry; expiries on ``as_of`` are dropped.
    """
    dtes = {expiry: (expiry - as_of).days for expiry in sorted(set(expiries)) if expiry > as_of}
    front_window, back_window = params.front_window, params.back_window

    for step in range(params.max_widen_steps + 1):
        front = [e for e, d in dtes.items() if front_window[0] <= d <= front_window[1]]
        back = [
            e for e, d in dtes.items() if back_window[0] <= d <= back_window[1] and (not front or e > front[0])
        ]
        if front and back:
            return Legs(front, back, front_window, back_window, widened=step > 0)
        if step == params.max_widen_steps:
            break
        if not front:
            front_window = _widen(front_window, params.widen_step)
        if not back:
            back_window = _widen(back_window, params.widen_step)

    raise CalculatorError(
        f"No expiry pair fits front {window_label(front_window)} / back {window_label(back_window)} DTE, "
        "even after widening."
    )


def scan(inputs: Inputs, params: Params = Params()) -> ScanResult:
    """Rank every trade on already-fetched chains. Pure: no network, no clock."""
    legs = pick_legs((c.expiry for c in inputs.chains), inputs.as_of, params)
    chains = {c.expiry: c for c in inputs.chains}

    candidates: list[Candidate] = []
    skipped = 0
    for front_expiry in legs.front:
        for back_expiry in legs.back:
            if back_expiry <= front_expiry:
                continue
            front, back = chains[front_expiry], chains[back_expiry]

            for side in SIDES:
                strikes = common_strikes((front, back), inputs.spot, params.strikes, sides=(side,))
                atm_strike = min(strikes, key=lambda k: (abs(k - inputs.spot), k), default=None)
                for strike in strikes:
                    near = nearest_quote(front.side(side), strike)
                    far = nearest_quote(back.side(side), strike)
                    if is_stale(near) or is_stale(far):
                        skipped += 1
                        continue
                    atm = side == "call" and strike == atm_strike
                    candidates.append(_candidate(side, (strike,), atm, (near,), (far,), front, back, inputs, params))

            double = _double_legs(front, back, inputs, params)
            if double is None:
                continue
            nears, fars = double
            if any(is_stale(q) for q in (*nears, *fars)):
                skipped += 1
                continue
            wing_strikes = tuple(float(q["strike"]) for q in nears)
            candidates.append(_candidate("double_35d", wing_strikes, False, nears, fars, front, back, inputs, params))

    if not candidates:
        if skipped:
            raise stale_error(f"all {skipped} candidate trades")
        raise CalculatorError("No strike is listed on both legs of any expiry pair.")

    return ScanResult(
        symbol=inputs.symbol,
        as_of=inputs.as_of,
        spot=inputs.spot,
        front_window=legs.front_window,
        back_window=legs.back_window,
        widened=legs.widened,
        min_ff=params.calculator.min_ff,
        max_spread=params.max_spread,
        earnings=inputs.earnings,
        candidates=sorted(candidates, key=_rank),
        skipped_stale=skipped,
    )


def run(
    symbol: str,
    provider: MarketDataProvider | None = None,
    params: Params = Params(),
    as_of: date | None = None,
) -> ScanResult:
    """Fetch the chains ``symbol`` needs through ``provider`` (default: yfinance) and scan them.

    ``as_of`` defaults to today's US market date.
    """
    symbol, provider = resolve(symbol, provider)
    as_of = as_of or market_date()

    legs = pick_legs(require_expirations(provider, symbol), as_of, params)
    chains = [provider.option_chain(symbol, expiry) for expiry in sorted({*legs.front, *legs.back})]
    spot = require_spot(provider, symbol)
    earnings = provider.next_earnings(symbol, as_of)

    return scan(Inputs(symbol, as_of, spot, chains, earnings), params)


def window_label(window: Window) -> str:
    return f"{window[0]}-{window[1]}"


def _double_legs(
    front: OptionChain, back: OptionChain, inputs: Inputs, params: Params
) -> tuple[tuple[pd.Series, pd.Series], tuple[pd.Series, pd.Series]] | None:
    """(front put, front call), (back put, back call) at the front expiry's -/+ delta strikes."""
    t1 = (front.expiry - inputs.as_of).days / params.calculator.days_per_year
    nears: list[pd.Series] = []
    fars: list[pd.Series] = []
    wings: tuple[tuple[OptionSide, float], ...] = (("put", -params.double_delta), ("call", params.double_delta))
    for side, target in wings:
        listed = set(front.side(side)["strike"]) & set(back.side(side)["strike"])
        near = quote_at_delta(front.side(side), side, inputs.spot, t1, target, listed, params.rate)
        if near is None:
            return None
        nears.append(near)
        fars.append(nearest_quote(back.side(side), float(near["strike"])))
    return (nears[0], nears[1]), (fars[0], fars[1])


def _candidate(
    structure: Structure,
    strikes: tuple[float, ...],
    atm: bool,
    nears: Sequence[pd.Series],
    fars: Sequence[pd.Series],
    front: OptionChain,
    back: OptionChain,
    inputs: Inputs,
    params: Params,
) -> Candidate:
    """Score one trade from its front (near) and back (far) leg quotes."""
    front_dte = (front.expiry - inputs.as_of).days
    back_dte = (back.expiry - inputs.as_of).days
    pair = forward_factor.Inputs(dte1=front_dte, iv1=mean_iv(*nears), dte2=back_dte, iv2=mean_iv(*fars))
    result = forward_factor.evaluate(pair, params.calculator)

    spread = at_most(max(spread_pct(q) for q in (*nears, *fars)), params.max_spread)
    crosses_earnings = inputs.earnings is not None and inputs.as_of <= inputs.earnings <= back.expiry
    failed = {
        "ff": result.ff is None or not result.ff.passed,
        "earnings": crosses_earnings,
        "spread": not spread.passed,
    }
    flags = tuple(rule for rule, fail in failed.items() if fail)

    return Candidate(
        structure=structure,
        front_expiry=front.expiry,
        front_dte=front_dte,
        back_expiry=back.expiry,
        back_dte=back_dte,
        strikes=strikes,
        atm=atm,
        close_by=previous_trading_day(front.expiry),
        spread=spread,
        crosses_earnings=crosses_earnings,
        result=result,
        flags=flags,
        passed=not flags,
    )


def _widen(window: Window, step: int) -> Window:
    return (max(1, window[0] - step), window[1] + step)


def _rank(candidate: Candidate) -> tuple[bool, bool, float]:
    """Trades not blocked by earnings or spread first; within each group highest FF first,
    with undefined FF (negative forward variance) last."""
    blocked = "earnings" in candidate.flags or "spread" in candidate.flags
    ff = candidate.result.ff
    return (blocked, ff is None, 0.0 if ff is None else -ff.value)
