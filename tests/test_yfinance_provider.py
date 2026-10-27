import logging

import pandas as pd
import pytest

from data.yfinance_provider import YFinanceError, YFinanceProvider


class EmptyTicker:
    """Stands in for yf.Ticker: history() comes back empty, optionally logging why."""

    def __init__(self, error: str | None = None):
        self._error = error

    def history(self, period):
        if self._error:
            logging.getLogger("yfinance").error(self._error)
        return pd.DataFrame()


def _provider(ticker: EmptyTicker) -> YFinanceProvider:
    provider = YFinanceProvider()
    provider._tickers["X"] = ticker
    return provider


def test_failed_fetch_raises_logged_error():
    provider = _provider(EmptyTicker("$X: No data found, symbol may be delisted"))
    with pytest.raises(YFinanceError, match="symbol may be delisted"):
        provider.spot("X")
    with pytest.raises(YFinanceError, match="symbol may be delisted"):
        provider.price_history("X")


def test_empty_fetch_without_error_is_no_spot():
    assert _provider(EmptyTicker()).spot("X") is None
