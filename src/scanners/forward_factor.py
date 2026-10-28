"""Forward factor scan: find the best calendar trades for a ticker.

Pairs every listed front expiry inside the ``front_dte`` range (default 7-60 DTE)
with every back expiry ``gap`` days after it (default 14-120) and, for every
front/back pair, scores three calendar structures (same strikes on both legs):

- call: call calendar at each strike in the window: the strike nearest spot plus
  ``strike_range`` strike steps above and below it, whatever the chain's increment
  (steps are counted on the strikes listed on both legs). The call strike nearest
  spot is marked ATM, the default structure.
- put:  put calendar at the same window of put strikes.
- double_35d: call calendar at the +35 delta strike plus put calendar at the -35
  delta strike (deltas from the front expiry); scored on the averaged wing IVs.

Given a ``strike``, only that exact strike is scanned (no window, no double
calendar), so the result ranks the expiry pairs that list it on both legs.

A trade passes when FF >= min_ff, no earnings fall between today and its back
expiry, and every leg has traded at least min_volume contracts today with at least
min_open_interest open. Trades blocked by earnings or liquidity rank below the rest;
each group is sorted by FF, highest first. Each trade also carries a close-by date:
the weekday before the front expiry.

The FF formula lives in ``calculators.forward_factor``; this module only chooses
what to feed it. ``scan`` is pure; ``run`` fetches through a MarketDataProvider.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from analytics.chain import (
    contracts,
    is_stale,
    listed_strikes,
    mean_iv,
    nearest_quote,
    quote_at_delta,
    stale_error,
    strikes_around,
)
from calculators import forward_factor
from calculators.results import Check, Serializable, at_least
from data.fetch import require_expirations, require_spot, resolve
from data.market_calendar import market_date, previous_trading_day
from data.models import SIDES, OptionChain, OptionSide
from data.provider import MarketDataProvider
from errors import CalculatorError

Window = tuple[int, int]  # inclusive range of days
Pair = tuple[date, date]  # (front, back) expiries
Structure = Literal["call", "put", "double_35d"]
STRUCTURES: tuple[Structure, ...] = ("call", "put", "double_35d")


@dataclass(frozen=True)
class Params:
    front_dte: Window = (7, 60)  # DTE of the front leg
    gap: Window = (14, 120)  # days from the front expiry to the back expiry
    strike: float | None = None  # scan only this strike; None means a window around spot
    strike_range: int = 1  # strike steps above and below the strike nearest spot
    min_volume: float = 100  # contracts each leg must have traded today
    min_open_interest: float = 1_000  # open contracts each leg must have
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
class Candidate:
    structure: Structure
    front_expiry: date
    front_dte: int
    back_expiry: date
    back_dte: int
    strikes: tuple[float, ...]  # one strike; (put, call) for double_35d
    atm: bool  # the ATM call calendar of its pair
    close_by: date  # weekday before the front expiry
    volume: Check  # thinnest leg's contracts traded today
    open_interest: Check  # thinnest leg's open interest
    crosses_earnings: bool  # earnings between as_of and the back expiry
    result: forward_factor.Result
    flags: tuple[str, ...]  # failed rules: "ff", "earnings", "volume", "oi"
    passed: bool


@dataclass(frozen=True)
class ScanResult(Serializable):
    symbol: str
    as_of: date
    spot: float
    front_dte: Window
    gap: Window
    min_ff: float
    min_volume: float
    min_open_interest: float
    earnings: date | None
    strike: float | None  # the only strike scanned; None means a window around spot
    strike_range: int  # 0 when a strike was given
    candidates: list[Candidate] = field(default_factory=list)  # best first
    skipped_stale: int = 0  # trades dropped because a quote was stale


def pick_pairs(expiries: Iterable[date], as_of: date, params: Params = Params()) -> list[Pair]:
    """Every (front, back) pair with the front inside ``front_dte`` and the back ``gap`` days after it.

    Pairs are ordered by front, then back; expiries on or before ``as_of`` are dropped.
    """
    listed = sorted({expiry for expiry in expiries if expiry > as_of})
    pairs = [
        (front, back)
        for front in listed
        if _within((front - as_of).days, params.front_dte)
        for back in listed
        if _within((back - front).days, params.gap)
    ]
    if not pairs:
        raise CalculatorError(
            f"No expiry pair has a front leg at {window_label(params.front_dte)} DTE and a back leg "
            f"{window_label(params.gap)} days after it."
        )
    return pairs


def scan(inputs: Inputs, params: Params = Params()) -> ScanResult:
    """Rank every trade on already-fetched chains. Pure: no network, no clock."""
    pairs = pick_pairs((c.expiry for c in inputs.chains), inputs.as_of, params)
    chains = {c.expiry: c for c in inputs.chains}

    candidates: list[Candidate] = []
    skipped = 0
    shared: set[float] = set()  # strikes listed on both legs of some pair
    for front_expiry, back_expiry in pairs:
        front, back = chains[front_expiry], chains[back_expiry]
        atm_strikes = strikes_around((front, back), inputs.spot, 0, sides=("call",))

        for side in SIDES:
            listed = listed_strikes((front, back), (side,))
            shared |= listed
            if params.strike is None:
                window = strikes_around((front, back), inputs.spot, params.strike_range, sides=(side,))
            else:
                window = [k for k in listed if math.isclose(k, params.strike)]
            for strike in window:
                near = nearest_quote(front.side(side), strike)
                far = nearest_quote(back.side(side), strike)
                if is_stale(near) or is_stale(far):
                    skipped += 1
                    continue
                atm = side == "call" and strike in atm_strikes
                candidates.append(_candidate(side, (strike,), atm, (near,), (far,), front, back, inputs, params))

        if params.strike is not None:
            continue
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
        if params.strike is not None and shared:
            raise CalculatorError(_unlisted_strike(params.strike, shared))
        raise CalculatorError("No strike is listed on both legs of any expiry pair.")

    return ScanResult(
        symbol=inputs.symbol,
        as_of=inputs.as_of,
        spot=inputs.spot,
        front_dte=params.front_dte,
        gap=params.gap,
        min_ff=params.calculator.min_ff,
        min_volume=params.min_volume,
        min_open_interest=params.min_open_interest,
        earnings=inputs.earnings,
        strike=params.strike,
        strike_range=params.strike_range if params.strike is None else 0,
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

    pairs = pick_pairs(require_expirations(provider, symbol), as_of, params)
    chains = [provider.option_chain(symbol, expiry) for expiry in sorted({e for pair in pairs for e in pair})]
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
        listed = listed_strikes((front, back), (side,))
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

    legs = (*nears, *fars)
    volume = at_least(min(contracts(q, "volume") for q in legs), params.min_volume)
    open_interest = at_least(min(contracts(q, "open_interest") for q in legs), params.min_open_interest)
    crosses_earnings = inputs.earnings is not None and inputs.as_of <= inputs.earnings <= back.expiry
    failed = {
        "ff": result.ff is None or not result.ff.passed,
        "earnings": crosses_earnings,
        "volume": not volume.passed,
        "oi": not open_interest.passed,
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
        volume=volume,
        open_interest=open_interest,
        crosses_earnings=crosses_earnings,
        result=result,
        flags=flags,
        passed=not flags,
    )


def _unlisted_strike(strike: float, shared: set[float]) -> str:
    below = max((k for k in shared if k < strike), default=None)
    above = min((k for k in shared if k > strike), default=None)
    nearest = ", ".join(f"{k:,.2f}" for k in (below, above) if k is not None)
    return f"No expiry pair lists the {strike:,.2f} strike on both legs. Nearest strikes that do: {nearest}."


def _within(days: int, window: Window) -> bool:
    return window[0] <= days <= window[1]


def _rank(candidate: Candidate) -> tuple[bool, bool, float]:
    """Trades not blocked by earnings or liquidity first; within each group highest FF first,
    with undefined FF (negative forward variance) last."""
    blocked = bool(set(candidate.flags) - {"ff"})
    ff = candidate.result.ff
    return (blocked, ff is None, 0.0 if ff is None else -ff.value)
