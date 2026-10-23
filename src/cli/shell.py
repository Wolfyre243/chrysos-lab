"""Pieces shared by every calculator CLI: argument parsing, the prompt loop, error
reporting and plain-text tables."""

import argparse
import json
from collections.abc import Callable, Sequence
from typing import TypeVar

from calculators.results import Check, Serializable
from errors import CalculatorError

R = TypeVar("R", bound=Serializable)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("symbols", nargs="*", help="Stock symbols; omit to be prompted.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    return parser


def report(symbol: str, compute: Callable[[str], R], render: Callable[[R], str], as_json: bool) -> bool:
    """Compute and print one symbol's result; print the error instead if it fails."""
    try:
        result = compute(symbol)
    except CalculatorError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:  # data provider failures, unexpected data shapes
        print(f"Error occurred processing {symbol.strip().upper()}: {e}")
        return False
    print(json.dumps(result.to_dict(), indent=2) if as_json else render(result))
    return True


def run_symbols(symbols: Sequence[str], handle: Callable[[str], bool]) -> int:
    """Handle each given symbol, or prompt for symbols until blank / exit. Returns the exit code."""
    if symbols:
        ok = True
        for i, symbol in enumerate(symbols):
            if i:
                print()
            ok = handle(symbol) and ok
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
        handle(symbol)
        print()


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-aligned columns: header line, dashed rule, then one line per row."""
    widths = [max(len(row[i]) for row in (headers, *rows)) for i in range(len(headers))]
    lines = [_join(headers, widths), _join(["-" * w for w in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    return lines


CHECK_HEADERS = ("Item", "Result", "Value", "Threshold")


def check_row(name: str, check: Check, fmt: str) -> tuple[str, str, str, str]:
    """(name, PASS/FAIL, value, threshold) cells for a Check."""
    return (
        name,
        "PASS" if check.passed else "FAIL",
        fmt.format(check.value),
        f"{check.op} {fmt.format(check.threshold)}",
    )


def _join(cells: Sequence[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
