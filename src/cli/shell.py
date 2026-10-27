"""Pieces shared by every calculator CLI: argument parsing, the prompt loop, error
reporting and plain-text tables."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
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
    except Exception as e:  # calculator errors, data provider failures, unexpected data shapes
        print("\n".join(failure_lines(symbol, e)))
        return False
    print(json.dumps(result.to_dict(), indent=2) if as_json else render(result))
    return True


def failure_lines(symbol: str, error: Exception) -> list[str]:
    """The lines reporting why ``symbol`` could not be computed."""
    if isinstance(error, CalculatorError):
        lines = [f"Error: {error}"]
        if error.__cause__ is not None:  # the data provider failure behind it
            lines.append(f"  Caused by {_describe(error.__cause__)}")
        return lines
    return [f"Error occurred processing {symbol.strip().upper()}: {_describe(error)}"]


def read_symbols(path: str | Path) -> list[str]:
    """Symbols from a text file, one per line; blank lines and ``#`` comments are skipped and
    duplicates dropped. Raises OSError when the file cannot be read."""
    symbols: dict[str, None] = {}  # insertion-ordered set
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        symbol = line.split("#", 1)[0].strip().upper()
        if symbol:
            symbols[symbol] = None
    return list(symbols)


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


def _describe(error: BaseException) -> str:
    """Exception type and message; the type alone tells e.g. a rate limit from a KeyError."""
    return f"{type(error).__name__}: {error}"


def _join(cells: Sequence[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
