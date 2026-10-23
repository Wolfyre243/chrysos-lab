"""Terminal shell for the calendar IV crush calculator.

    uv run calendar-iv-crush AAPL MSFT     # evaluate each symbol
    uv run calendar-iv-crush               # prompt for symbols until blank / exit
    uv run calendar-iv-crush AAPL --json   # structured output
"""

import sys

from calculators.calendar_iv_crush import Result, run
from cli.shell import CHECK_HEADERS, base_parser, check_row, format_table, report, run_symbols
from data.yfinance_provider import YFinanceProvider


def format_result(result: Result) -> str:
    rows: list[tuple[str, str, str, str]] = [
        check_row("avg_volume", result.avg_volume, "{:,.0f}"),
        check_row("iv30_rv30", result.iv30_rv30, "{:.3f}"),
        check_row("ts_slope_0_45", result.ts_slope_0_45, "{:.5f}"),
        ("iv30", "", f"{result.iv30:.4f}", ""),
        ("rv30", "", f"{result.rv30:.4f}", ""),
        ("spot", "", f"{result.spot:,.2f}", ""),
        (
            "straddle",
            "",
            "n/a" if result.straddle is None else f"{result.straddle:,.2f}",
            f"(front expiry {result.front_expiry.isoformat()})",
        ),
        (
            "expected_move",
            "",
            "n/a" if result.expected_move is None else f"{result.expected_move:.2%}",
            "",
        ),
    ]

    lines = [result.symbol, result.recommendation, ""]
    lines.extend(format_table(CHECK_HEADERS, rows))

    curve = " | ".join(f"{p.dte}d {p.atm_iv:.3f}" for p in result.term_structure)
    lines += ["", f"Term structure (DTE / ATM IV): {curve}", f"As of {result.as_of.isoformat()}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = base_parser("Calendar spread IV crush checker.").parse_args(argv)
    provider = YFinanceProvider()

    def handle(symbol: str) -> bool:
        return report(symbol, lambda s: run(s, provider=provider), format_result, args.json)

    return run_symbols(args.symbols, handle)


if __name__ == "__main__":
    sys.exit(main())
