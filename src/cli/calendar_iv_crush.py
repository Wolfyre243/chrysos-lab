"""Terminal shell for the calendar IV crush calculator.

    uv run calendar-iv-crush AAPL MSFT     # evaluate each symbol
    uv run calendar-iv-crush               # prompt for symbols until blank / exit
    uv run calendar-iv-crush AAPL --json   # structured output
"""

import argparse
import json
import sys

from calculators.calendar_iv_crush import Check, Result, run
from data.yfinance_provider import YFinanceProvider
from errors import CalculatorError

_HEADERS = ("Item", "Result", "Value", "Threshold")


def format_result(result: Result) -> str:
    rows: list[tuple[str, str, str, str]] = [
        _check_row("avg_volume", result.avg_volume, "{:,.0f}"),
        _check_row("iv30_rv30", result.iv30_rv30, "{:.3f}"),
        _check_row("ts_slope_0_45", result.ts_slope_0_45, "{:.5f}"),
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

    widths = [max(len(row[i]) for row in (_HEADERS, *rows)) for i in range(len(_HEADERS))]
    lines = [result.symbol, result.recommendation, ""]
    lines.append(_join(_HEADERS, widths))
    lines.append(_join(tuple("-" * w for w in widths), widths))
    lines.extend(_join(row, widths) for row in rows)

    curve = " | ".join(f"{p.dte}d {p.atm_iv:.3f}" for p in result.term_structure)
    lines += ["", f"Term structure (DTE / ATM IV): {curve}", f"As of {result.as_of.isoformat()}"]
    return "\n".join(lines)


def _check_row(name: str, check: Check, fmt: str) -> tuple[str, str, str, str]:
    return (
        name,
        "PASS" if check.passed else "FAIL",
        fmt.format(check.value),
        f"{check.op} {fmt.format(check.threshold)}",
    )


def _join(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()


def _evaluate(symbol: str, provider: YFinanceProvider, as_json: bool) -> bool:
    try:
        result = run(symbol, provider=provider)
    except CalculatorError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:  # data provider failures, unexpected data shapes
        print(f"Error occurred processing {symbol.strip().upper()}: {e}")
        return False
    print(json.dumps(result.to_dict(), indent=2) if as_json else format_result(result))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calendar spread IV crush checker.")
    parser.add_argument("symbols", nargs="*", help="Stock symbols; omit to be prompted.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    args = parser.parse_args(argv)

    provider = YFinanceProvider()

    if args.symbols:
        ok = True
        for i, symbol in enumerate(args.symbols):
            if i:
                print()
            ok = _evaluate(symbol, provider, args.json) and ok
        return 0 if ok else 1

    while True:
        try:
            symbol = input("Enter stock symbol (blank to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not symbol or symbol.lower() in ("exit", "quit"):
            return 0
        print("Loading...")
        _evaluate(symbol, provider, args.json)
        print()


if __name__ == "__main__":
    sys.exit(main())
