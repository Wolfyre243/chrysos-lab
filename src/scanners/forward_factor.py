"""Forward factor scan: find the best calendar pairs for a ticker.

Picks front expiries near 30 DTE and back expiries near 60 DTE, checks every
front/back pair at the strikes nearest spot (same strike on both legs), and ranks
the trades by forward factor, highest first. Each pair and strike is scanned twice,
as a call calendar (call IV on both legs) and as a put calendar (put IV on both
legs), so the two sides can be compared.

A DTE window is widened only when no listed expiry falls inside it (e.g. names with
monthly expiries only), never because no trade passes the FF threshold.

The FF formula lives in ``calculators.forward_factor``; this module only chooses
what to feed it. ``scan`` is pure; ``run`` fetches through a MarketDataProvider.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from analytics.chain import common_strikes, is_stale, nearest_quote, stale_error
from calculators import forward_factor
from calculators.results import Serializable
from data.fetch import market_date, require_expirations, require_spot, resolve
from data.models import SIDES, OptionChain, OptionSide
from data.provider import MarketDataProvider
from errors import CalculatorError

Window = tuple[int, int]  # inclusive DTE range


@dataclass(frozen=True)
class Params:
    front_window: Window = (21, 45)  # around 30 DTE
    back_window: Window = (50, 90)  # around 60 DTE
    widen_step: int = 7  # days added to each end of an empty window
    max_widen_steps: int = 6
    strikes: int = 5  # strikes nearest spot checked per pair
    calculator: forward_factor.Params = forward_factor.Params()


@dataclass(frozen=True)
class Inputs:
    symbol: str
    as_of: date
    spot: float
    chains: list[OptionChain]


@dataclass(frozen=True)
class Legs:
    front: list[date]
    back: list[date]
    front_window: Window
    back_window: Window
    widened: bool


@dataclass(frozen=True)
class Candidate:
    side: OptionSide
    front_expiry: date
    front_dte: int
    back_expiry: date
    back_dte: int
    strike: float
    result: forward_factor.Result


@dataclass(frozen=True)
class ScanResult(Serializable):
    symbol: str
    as_of: date
    spot: float
    front_window: Window
    back_window: Window
    widened: bool
    min_ff: float
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
    """Rank every side/front/back/strike trade on already-fetched chains. Pure: no network, no clock."""
    legs = pick_legs((c.expiry for c in inputs.chains), inputs.as_of, params)
    chains = {c.expiry: c for c in inputs.chains}

    candidates: list[Candidate] = []
    skipped = 0
    for front_expiry in legs.front:
        for back_expiry in legs.back:
            if back_expiry <= front_expiry:
                continue
            front, back = chains[front_expiry], chains[back_expiry]
            front_dte = (front_expiry - inputs.as_of).days
            back_dte = (back_expiry - inputs.as_of).days
            for side in SIDES:
                for strike in common_strikes((front, back), inputs.spot, params.strikes, sides=(side,)):
                    near = nearest_quote(front.side(side), strike)
                    far = nearest_quote(back.side(side), strike)
                    if is_stale(near) or is_stale(far):
                        skipped += 1
                        continue
                    pair = forward_factor.Inputs(
                        dte1=front_dte, iv1=float(near["iv"]), dte2=back_dte, iv2=float(far["iv"])
                    )
                    candidates.append(
                        Candidate(
                            side=side,
                            front_expiry=front_expiry,
                            front_dte=front_dte,
                            back_expiry=back_expiry,
                            back_dte=back_dte,
                            strike=strike,
                            result=forward_factor.evaluate(pair, params.calculator),
                        )
                    )

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

    return scan(Inputs(symbol, as_of, spot, chains), params)


def window_label(window: Window) -> str:
    return f"{window[0]}-{window[1]}"


def _widen(window: Window, step: int) -> Window:
    return (max(1, window[0] - step), window[1] + step)


def _rank(candidate: Candidate) -> tuple[bool, float]:
    """Highest FF first; trades with no FF (negative forward variance) last."""
    ff = candidate.result.ff
    return (ff is None, 0.0 if ff is None else -ff.value)
