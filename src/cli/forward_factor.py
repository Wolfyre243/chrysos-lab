"""Terminal shell for the forward factor scanner.

    uv run forward-factor AAPL MSFT                 # scan each symbol
    uv run forward-factor                           # prompt for symbols until blank / exit
    uv run forward-factor AAPL --json               # structured output
    uv run forward-factor AAPL --strike 350         # best expiry pairs for the 350 strike only
    uv run forward-factor AAPL --range 2 --top 15   # 2 strike steps above and below spot
    uv run forward-factor AAPL --min-oi 5000        # stricter open interest per leg
"""

import argparse
import sys

from calculators.forward_factor import Result
from cli.shell import CHECK_HEADERS, base_parser, check_row, format_table, report, run_symbols
from data.yfinance_provider import YFinanceProvider
from scanners.forward_factor import STRUCTURES, Candidate, Params, ScanResult, Structure, run, window_label

DETAIL = 3  # candidates given a full breakdown

_LABELS: dict[Structure, str] = {"call": "call", "put": "put", "double_35d": "dbl 35d"}
_NAMES: dict[Structure, str] = {
    "call": "call calendar",
    "put": "put calendar",
    "double_35d": "35-delta double calendar",
}

_TYPE_HEADERS = ("Type", "Trades", "Pass", "Best FF", "Best trade")
_RANK_HEADERS = ("Rank", "Type", "Front", "DTE", "Back", "DTE", "Strike", "IV1", "IV2", "FF", "Vol", "OI", "Close by", "Result")


def format_scan(scan: ScanResult, top: int = 10) -> str:
    passing = sum(1 for c in scan.candidates if c.passed)
    earnings = (
        f"Next earnings {scan.earnings.isoformat()}: trades whose back expiry is on/after it fail."
        if scan.earnings
        else "Next earnings: none found."
    )
    lines = [
        scan.symbol,
        f"{passing} of {len(scan.candidates)} trades pass: FF >= {scan.min_ff:g}, no earnings "
        f"before the back expiry, every leg volume >= {scan.min_volume:,.0f} and "
        f"open interest >= {scan.min_open_interest:,.0f}",
        "",
        f"Spot {scan.spot:,.2f}  |  DTE windows: front {window_label(scan.front_window)}, "
        f"back {window_label(scan.back_window)}  |  As of {scan.as_of.isoformat()}",
        _strike_line(scan),
        earnings,
    ]
    if scan.widened:
        lines.append("Windows widened: no listed expiries fit the default ones.")
    if scan.skipped_stale:
        lines.append(f"{scan.skipped_stale} trades skipped for stale quotes.")

    lines.append("")
    lines.extend(format_table(_TYPE_HEADERS, [_type_row(scan, s) for s in STRUCTURES]))

    ranked = scan.candidates[:top]
    lines.append("")
    lines.extend(format_table(_RANK_HEADERS, [_rank_row(i, c) for i, c in enumerate(ranked, 1)]))

    for i, candidate in enumerate(ranked[:DETAIL], 1):
        lines += ["", f"#{i}  {_describe(candidate)}", format_candidate(candidate, scan)]
    return "\n".join(lines)


def format_candidate(c: Candidate, scan: ScanResult) -> str:
    """Forward vol working plus the liquidity and earnings checks for one trade."""
    earnings = (
        "FAIL" if c.crosses_earnings else "PASS",
        "none found" if scan.earnings is None else scan.earnings.isoformat(),
        f"> {c.back_expiry.isoformat()}",
    )
    rows = [
        *_breakdown_rows(c.result),
        check_row("min leg volume", c.volume, "{:,.0f}"),
        check_row("min leg open int", c.open_interest, "{:,.0f}"),
        ("earnings", *earnings),
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


def _strike_line(scan: ScanResult) -> str:
    if scan.strike is not None:
        return f"Strike: {scan.strike:,.2f} only (pairs listing it on both legs)  |  double calendar skipped"
    plural = "" if scan.strike_range == 1 else "s"
    return f"Strikes: nearest spot +/- {scan.strike_range} step{plural}"


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
    r = c.result
    return (
        str(rank),
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
        "PASS" if c.passed else f"FAIL {','.join(c.flags)}",
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Forward factor scanner: ranks calendar trades near 30/60 DTE.")
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
    args = parser.parse_args(argv)

    provider = YFinanceProvider()
    params = Params(
        strike=args.strike,
        strike_range=args.strike_range,
        min_volume=args.min_volume,
        min_open_interest=args.min_oi,
    )

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
