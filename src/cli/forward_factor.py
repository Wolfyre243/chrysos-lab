"""Terminal shell for the forward factor scanner.

    uv run forward-factor AAPL MSFT                 # scan each symbol
    uv run forward-factor                           # prompt for symbols until blank / exit
    uv run forward-factor AAPL --json               # structured output
    uv run forward-factor AAPL --strikes 7 --top 15
"""

import argparse
import sys

from calculators.forward_factor import Result
from cli.shell import CHECK_HEADERS, base_parser, check_row, format_table, report, run_symbols
from data.yfinance_provider import YFinanceProvider
from data.models import SIDES, OptionSide
from scanners.forward_factor import Candidate, Params, ScanResult, run, window_label

DETAIL = 3  # candidates given a full breakdown

_SIDE_HEADERS = ("Side", "Trades", "Pass", "Best FF", "Best trade")
_RANK_HEADERS = ("Rank", "Side", "Front", "DTE", "Back", "DTE", "Strike", "IV1", "IV2", "Fwd vol", "FF", "Result")


def format_scan(scan: ScanResult, top: int = 10) -> str:
    passing = sum(1 for c in scan.candidates if _passed(c))
    lines = [
        scan.symbol,
        f"{passing} of {len(scan.candidates)} trades pass FF >= {scan.min_ff:g}",
        "",
        f"Spot {scan.spot:,.2f}  |  DTE windows: front {window_label(scan.front_window)}, "
        f"back {window_label(scan.back_window)}  |  As of {scan.as_of.isoformat()}",
    ]
    if scan.widened:
        lines.append("Windows widened: no listed expiries fit the default ones.")
    if scan.skipped_stale:
        lines.append(f"{scan.skipped_stale} trades skipped for stale quotes.")

    lines.append("")
    lines.extend(format_table(_SIDE_HEADERS, [_side_row(scan, side) for side in SIDES]))

    ranked = scan.candidates[:top]
    lines.append("")
    lines.extend(format_table(_RANK_HEADERS, [_rank_row(i, c) for i, c in enumerate(ranked, 1)]))

    for i, candidate in enumerate(ranked[:DETAIL], 1):
        lines += ["", f"#{i}  {candidate.side} calendar  {_trade(candidate)}", format_breakdown(candidate.result)]
    return "\n".join(lines)


def format_breakdown(result: Result) -> str:
    """The forward vol working for one pair, as in the original calculator."""
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
    return "\n".join(format_table(CHECK_HEADERS, rows))


def _side_row(scan: ScanResult, side: OptionSide) -> tuple[str, ...]:
    """Trade count, passes and best trade for one side (candidates are already ranked)."""
    trades = [c for c in scan.candidates if c.side == side]
    best = trades[0] if trades else None
    return (
        side,
        str(len(trades)),
        str(sum(1 for c in trades if _passed(c))),
        "n/a" if best is None or best.result.ff is None else f"{best.result.ff.value:.3f}",
        "" if best is None else _trade(best),
    )


def _rank_row(rank: int, c: Candidate) -> tuple[str, ...]:
    r = c.result
    return (
        str(rank),
        c.side,
        c.front_expiry.isoformat(),
        str(c.front_dte),
        c.back_expiry.isoformat(),
        str(c.back_dte),
        f"{c.strike:,.2f}",
        f"{r.iv1:.2%}",
        f"{r.iv2:.2%}",
        "n/a" if r.fwd_vol is None else f"{r.fwd_vol:.2%}",
        "n/a" if r.ff is None else f"{r.ff.value:.3f}",
        "PASS" if _passed(c) else "FAIL",
    )


def _passed(c: Candidate) -> bool:
    return c.result.ff is not None and c.result.ff.passed


def _trade(c: Candidate) -> str:
    return (
        f"{c.front_expiry.isoformat()} ({c.front_dte}d) / {c.back_expiry.isoformat()} ({c.back_dte}d) "
        f"@ {c.strike:,.2f}"
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Forward factor scanner: ranks calendar pairs near 30/60 DTE.")
    parser.add_argument(
        "--strikes",
        type=_positive_int,
        default=Params.strikes,
        help="Strikes nearest spot checked per expiry pair (default %(default)s).",
    )
    parser.add_argument(
        "--top", type=_positive_int, default=10, help="Ranked trades to list (default %(default)s)."
    )
    args = parser.parse_args(argv)

    provider = YFinanceProvider()
    params = Params(strikes=args.strikes)

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
