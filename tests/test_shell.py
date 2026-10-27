import pytest

from cli.shell import failure_lines, read_symbols, report
from errors import CalculatorError


def _fail(error: Exception):
    def compute(symbol):
        raise error

    return compute


def test_report_prints_calculator_error(capsys):
    assert not report("x", _fail(CalculatorError("Not enough option data.")), str, as_json=False)
    assert capsys.readouterr().out == "Error: Not enough option data.\n"


def test_report_prints_original_error_behind_calculator_error(capsys):
    error = CalculatorError("Unable to retrieve underlying stock price.")
    error.__cause__ = ConnectionError("Too Many Requests")
    assert not report("x", _fail(error), str, as_json=False)
    assert capsys.readouterr().out.splitlines() == [
        "Error: Unable to retrieve underlying stock price.",
        "  Caused by ConnectionError: Too Many Requests",
    ]


def test_report_prints_unexpected_error_type(capsys):
    assert not report("x", _fail(KeyError("Close")), str, as_json=False)
    assert capsys.readouterr().out == "Error occurred processing X: KeyError: 'Close'\n"


def test_failure_lines():
    assert failure_lines(" xyz ", KeyError("Close")) == ["Error occurred processing XYZ: KeyError: 'Close'"]
    assert failure_lines("x", CalculatorError("Not enough option data.")) == ["Error: Not enough option data."]


def test_read_symbols(tmp_path):
    path = tmp_path / "ff.txt"
    path.write_text("aapl\n  msft  \n\n# watchlist\nSPY  # index\nAAPL\r\n", encoding="utf-8")
    assert read_symbols(path) == ["AAPL", "MSFT", "SPY"]


def test_read_symbols_missing_file(tmp_path):
    with pytest.raises(OSError):
        read_symbols(tmp_path / "missing.txt")
