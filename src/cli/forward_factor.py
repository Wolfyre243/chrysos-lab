"""Terminal shell for the forward factor scanner.

    uv run forward-factor AAPL MSFT                 # scan each symbol
    uv run forward-factor                           # prompt for symbols until blank / exit
    uv run forward-factor AAPL --json               # structured output
    uv run forward-factor AAPL --strike 350         # best expiry pairs for the 350 strike only
    uv run forward-factor AAPL --range 2 --top 15   # 2 strike steps above and below spot
    uv run forward-factor AAPL --min-oi 5000        # stricter open interest per leg
    uv run forward-factor AAPL --front-dte 21 45 --gap 28 60  # only ~30/60 style pairs
    uv run forward-factor --bulk ff.txt             # rank every ticker in ff.txt by passing trades
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from calculators.forward_factor import Result
from calculators.results import Serializable
from cli.shell import (
    CHECK_HEADERS,
    base_parser,
    check_row,
    failure_lines,
    format_table,
    read_symbols,
    report,
    run_symbols,
)
from data.market_calendar import market_date
from data.provider import MarketDataProvider
from data.yfinance_provider import YFinanceProvider
from scanners.forward_factor import (
    STRUCTURES,
    Candidate,
    Params,
    ScanResult,
    Structure,
    Window,
    run,
    window_label,
)

DETAIL = 3  # candidates given a full breakdown

_LABELS: dict[Structure, str] = {"call": "call", "put": "put", "double_35d": "dbl 35d"}
_NAMES: dict[Structure, str] = {
    "call": "call calendar",
    "put": "put calendar",
    "double_35d": "35-delta double calendar",
}

_TYPE_HEADERS = ("Type", "Trades", "Pass", "Best FF", "Best trade")
_TRADE_HEADERS = ("Type", "Front", "DTE", "Back", "DTE", "Strike", "IV1", "IV2", "FF", "Vol", "OI", "Close by")
_RANK_HEADERS = ("Rank", *_TRADE_HEADERS, "Result")
_BULK_HEADERS = ("Rank", "Symbol", "Spot", "Pass", "Trades", *_TRADE_HEADERS)


@dataclass(frozen=True)
class BulkTicker:
    symbol: str
    spot: float
    earnings: date | None
    trades: int
    passed: int
    best: Candidate  # highest-FF passing trade


@dataclass(frozen=True)
class BulkFailure:
    symbol: str
    error: str  # the lines a single-symbol run would print, newline-joined


@dataclass(frozen=True)
class BulkResult(Serializable):
    source: str  # the ticker file
    as_of: date
    total: int  # tickers in the file
    front_dte: Window
    gap: Window
    min_ff: float
    min_volume: float
    min_open_interest: float
    strike: float | None
    strike_range: int
    ranked: list[BulkTicker] = field(default_factory=list)  # most passing trades first
    empty: list[str] = field(default_factory=list)  # scanned, no passing trade
    failed: list[BulkFailure] = field(default_factory=list)


def scan_bulk(
    symbols: Sequence[str],
    compute: Callable[[str], ScanResult],
    params: Params,
    as_of: date,
    source: str,
) -> BulkResult:
    """Scan each symbol in turn, printing progress to stderr, and rank the ones with passing
    trades: most passes first, ties broken by the best trade's FF. A failing symbol is recorded
    and the run goes on."""
    ranked: list[BulkTicker] = []
    empty: list[str] = []
    failed: list[BulkFailure] = []
    for i, symbol in enumerate(symbols, 1):
        print(f"Scanning {symbol} ({i}/{len(symbols)})...", file=sys.stderr, flush=True)
        try:
            scan = compute(symbol)
        except Exception as e:  # same scope as report(): one bad ticker must not stop the run
            failed.append(BulkFailure(symbol, "\n".join(failure_lines(symbol, e))))
            continue
        passing = [c for c in scan.candidates if c.passed]  # candidates are already ranked
        if passing:
            ranked.append(
                BulkTicker(scan.symbol, scan.spot, scan.earnings, len(scan.candidates), len(passing), passing[0])
            )
        else:
            empty.append(scan.symbol)

    return BulkResult(
        source=source,
        as_of=as_of,
        total=len(symbols),
        front_dte=params.front_dte,
        gap=params.gap,
        min_ff=params.calculator.min_ff,
        min_volume=params.min_volume,
        min_open_interest=params.min_open_interest,
        strike=params.strike,
        strike_range=params.strike_range if params.strike is None else 0,
        ranked=sorted(ranked, key=_bulk_rank),
        empty=empty,
        failed=failed,
    )


def format_bulk(result: BulkResult) -> str:
    lines = [
        f"Bulk scan: {result.total} tickers from {result.source}  |  As of {result.as_of.isoformat()}",
        f"{len(result.ranked)} with passing trades, {len(result.empty)} with none, {len(result.failed)} failed. "
        f"Pass: {_rules(result.min_ff, result.min_volume, result.min_open_interest)}",
        _strike_line(result.strike, result.strike_range),
        _dte_line(result.front_dte, result.gap),
        "",
    ]
    if result.ranked:
        lines.extend(format_table(_BULK_HEADERS, [_bulk_row(i, t) for i, t in enumerate(result.ranked, 1)]))
        for i, t in enumerate(result.ranked, 1):
            lines += ["", f"#{i}  {t.symbol}  {_describe(t.best)}", format_candidate(t.best, t.earnings)]
    else:
        lines.append("No ticker has a passing trade.")

    if result.failed:
        lines += ["", f"Failed ({len(result.failed)}):"]
        width = max(len(f.symbol) for f in result.failed)
        for failure in result.failed:
            first, *rest = failure.error.splitlines()
            lines.append(f"{failure.symbol.ljust(width)}  {first}")
            lines += [f"{' ' * width}  {line}" for line in rest]
    return "\n".join(lines)


def format_scan(scan: ScanResult, top: int = 10) -> str:
    passing = sum(1 for c in scan.candidates if c.passed)
    earnings = (
        f"Next earnings {scan.earnings.isoformat()}: trades whose back expiry is on/after it fail."
        if scan.earnings
        else "Next earnings: none found."
    )
    lines = [
        scan.symbol,
        f"{passing} of {len(scan.candidates)} trades pass: "
        f"{_rules(scan.min_ff, scan.min_volume, scan.min_open_interest)}",
        "",
        f"Spot {scan.spot:,.2f}  |  {_dte_line(scan.front_dte, scan.gap)}  |  As of {scan.as_of.isoformat()}",
        _strike_line(scan.strike, scan.strike_range),
        earnings,
    ]
    if scan.skipped_stale:
        lines.append(f"{scan.skipped_stale} trades skipped for stale quotes.")

    lines.append("")
    lines.extend(format_table(_TYPE_HEADERS, [_type_row(scan, s) for s in STRUCTURES]))

    ranked = scan.candidates[:top]
    lines.append("")
    lines.extend(format_table(_RANK_HEADERS, [_rank_row(i, c) for i, c in enumerate(ranked, 1)]))

    for i, candidate in enumerate(ranked[:DETAIL], 1):
        lines += ["", f"#{i}  {_describe(candidate)}", format_candidate(candidate, scan.earnings)]
    return "\n".join(lines)


def format_candidate(c: Candidate, earnings: date | None) -> str:
    """Forward vol working plus the liquidity and earnings checks for one trade."""
    earnings_cells = (
        "FAIL" if c.crosses_earnings else "PASS",
        "none found" if earnings is None else earnings.isoformat(),
        f"> {c.back_expiry.isoformat()}",
    )
    rows = [
        *_breakdown_rows(c.result),
        check_row("min leg volume", c.volume, "{:,.0f}"),
        check_row("min leg open int", c.open_interest, "{:,.0f}"),
        ("earnings", *earnings_cells),
    ]
    return "\n".join(format_table(CHECK_HEADERS, rows))


def format_breakdown(result: Result) -> str:
    """The forward vol working for one pair, as in the original calculator."""
    return "\n".join(format_table(CHECK_HEADERS, _breakdown_rows(result)))


def _breakdown_rows(result: Result) -> list[tuple[str, str, str, str]]:
    fwd_var = f"{result.fwd_var:.8f}"
    if result.fwd_vol is None:
        fwd_var += " (negative)"
        fwd_vol, ff_note = "n/a (check inputs)", "negative fwd variance"
    else:
        fwd_vol, ff_note = f"{result.fwd_vol:.4%}", "fwd vol is 0"

    rows: list[tuple[str, str, str, str]] = [
        ("T1 = DTE1 / 365", "", f"{result.t1:.6f} yr", ""),
        ("sigma1", "", f"{result.iv1:.6f}", ""),
        ("total variance1", "", f"{result.tv1:.8f}", ""),
        ("T2 = DTE2 / 365", "", f"{result.t2:.6f} yr", ""),
        ("sigma2", "", f"{result.iv2:.6f}", ""),
        ("total variance2", "", f"{result.tv2:.8f}", ""),
        ("forward variance", "", fwd_var, ""),
        ("forward vol", "", fwd_vol, ""),
    ]
    if result.ff is None:
        rows.append(("ff", "FAIL", f"undefined ({ff_note})", ""))
    else:
        rows += [check_row("ff", result.ff, "{:.4f}"), ("ff (%)", "", f"{result.ff.value:.2%}", "")]
    return rows


def _rules(min_ff: float, min_volume: float, min_open_interest: float) -> str:
    return (
        f"FF >= {min_ff:g}, no earnings before the back expiry, every leg volume >= {min_volume:,.0f} and "
        f"open interest >= {min_open_interest:,.0f}"
    )


def _strike_line(strike: float | None, strike_range: int) -> str:
    if strike is not None:
        return f"Strike: {strike:,.2f} only (pairs listing it on both legs)  |  double calendar skipped"
    plural = "" if strike_range == 1 else "s"
    return f"Strikes: nearest spot +/- {strike_range} step{plural}"


def _dte_line(front_dte: Window, gap: Window) -> str:
    return f"Front leg {window_label(front_dte)} DTE, back leg {window_label(gap)} days after it"


def _type_row(scan: ScanResult, structure: Structure) -> tuple[str, ...]:
    """Trade count, passes and best trade for one structure (candidates are already ranked)."""
    trades = [c for c in scan.candidates if c.structure == structure]
    best = trades[0] if trades else None
    return (
        _LABELS[structure],
        str(len(trades)),
        str(sum(1 for c in trades if c.passed)),
        "n/a" if best is None or best.result.ff is None else f"{best.result.ff.value:.3f}",
        "" if best is None else _trade(best),
    )


def _rank_row(rank: int, c: Candidate) -> tuple[str, ...]:
    return (str(rank), *_trade_cells(c), "PASS" if c.passed else f"FAIL {','.join(c.flags)}")


def _bulk_row(rank: int, t: BulkTicker) -> tuple[str, ...]:
    return (str(rank), t.symbol, f"{t.spot:,.2f}", str(t.passed), str(t.trades), *_trade_cells(t.best))


def _trade_cells(c: Candidate) -> tuple[str, ...]:
    """One trade's cells under _TRADE_HEADERS."""
    r = c.result
    return (
        "call ATM" if c.atm else _LABELS[c.structure],
        c.front_expiry.isoformat(),
        str(c.front_dte),
        c.back_expiry.isoformat(),
        str(c.back_dte),
        _strikes(c),
        f"{r.iv1:.2%}",
        f"{r.iv2:.2%}",
        "n/a" if r.ff is None else f"{r.ff.value:.3f}",
        f"{c.volume.value:,.0f}",
        f"{c.open_interest.value:,.0f}",
        c.close_by.isoformat(),
    )


def _bulk_rank(ticker: BulkTicker) -> tuple[int, float]:
    ff = ticker.best.result.ff  # always set: a passing trade has a defined FF
    return (-ticker.passed, 0.0 if ff is None else -ff.value)


def _describe(c: Candidate) -> str:
    name = _NAMES[c.structure] + (" (ATM)" if c.atm else "")
    return f"{name}  {_trade(c)}  |  close by {c.close_by.isoformat()}"


def _trade(c: Candidate) -> str:
    return (
        f"{c.front_expiry.isoformat()} ({c.front_dte}d) / {c.back_expiry.isoformat()} ({c.back_dte}d) "
        f"@ {_strikes(c)}"
    )


def _strikes(c: Candidate) -> str:
    return "/".join(f"{k:,.2f}" for k in c.strikes)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def _run_bulk(
    parser: argparse.ArgumentParser, args: argparse.Namespace, provider: MarketDataProvider, params: Params
) -> int:
    """Scan every ticker in the --bulk file and print the ranking. Returns the exit code."""
    if args.symbols:
        parser.error("give symbols or --bulk, not both")
    try:
        symbols = read_symbols(args.bulk)
    except OSError as e:
        parser.error(f"cannot read {args.bulk}: {e.strerror or e}")
    if not symbols:
        parser.error(f"no tickers in {args.bulk}")

    as_of = market_date()  # one date for the whole run, even if it crosses midnight
    result = scan_bulk(
        symbols, lambda s: run(s, provider=provider, params=params, as_of=as_of), params, as_of, args.bulk
    )
    print(json.dumps(result.to_dict(), indent=2) if args.json else format_bulk(result))
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Forward factor scanner: ranks calendar trades across front/back expiry pairs.")
    strikes = parser.add_mutually_exclusive_group()
    strikes.add_argument(
        "--strike",
        type=_positive_float,
        help="Scan only this exact strike and rank the expiry pairs that list it on both legs.",
    )
    strikes.add_argument(
        "--range",
        type=_non_negative_int,
        default=Params.strike_range,
        dest="strike_range",
        help="Strike steps above and below the strike nearest spot, whatever the chain's increment "
        "(default %(default)s).",
    )
    parser.add_argument(
        "--top", type=_positive_int, default=10, help="Ranked trades to list (default %(default)s)."
    )
    parser.add_argument(
        "--min-volume",
        type=_non_negative_int,
        default=int(Params.min_volume),
        help="Contracts every leg must have traded today (default %(default)s).",
    )
    parser.add_argument(
        "--min-oi",
        type=_non_negative_int,
        default=int(Params.min_open_interest),
        help="Open interest every leg must have (default %(default)s).",
    )
    parser.add_argument(
        "--front-dte",
        type=_positive_int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=Params.front_dte,
        help="DTE range of the front leg (default %d %d)." % Params.front_dte,
    )
    parser.add_argument(
        "--gap",
        type=_positive_int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=Params.gap,
        help="Days from the front expiry to the back expiry (default %d %d)." % Params.gap,
    )
    parser.add_argument(
        "--bulk",
        metavar="FILE",
        help="Scan every ticker in FILE (one per line; blank lines and # comments skipped) and rank "
        "tickers by passing trades, each with its best trade. --top does not apply.",
    )
    args = parser.parse_args(argv)
    for flag, (low, high) in (("--front-dte", args.front_dte), ("--gap", args.gap)):
        if low > high:
            parser.error(f"{flag}: MIN {low} is above MAX {high}")

    provider = YFinanceProvider()
    params = Params(
        front_dte=tuple(args.front_dte),
        gap=tuple(args.gap),
        strike=args.strike,
        strike_range=args.strike_range,
        min_volume=args.min_volume,
        min_open_interest=args.min_oi,
    )
    if args.bulk is not None:
        return _run_bulk(parser, args, provider, params)

    def handle(symbol: str) -> bool:
        return report(
            symbol,
            lambda s: run(s, provider=provider, params=params),
            lambda scan: format_scan(scan, args.top),
            args.json,
        )

    return run_symbols(args.symbols, handle)


if __name__ == "__main__":
    sys.exit(main())
