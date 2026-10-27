import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from data.models import CHAIN_COLUMNS, PRICE_COLUMNS, OptionChain

_CHAIN_RENAME = {"impliedVolatility": "iv", "openInterest": "open_interest"}


class YFinanceError(RuntimeError):
    """yfinance returned no data; the message is the error it logged about why."""


class YFinanceProvider:
    """MarketDataProvider backed by Yahoo Finance (free, delayed, unofficial)."""

    def __init__(self) -> None:
        self._tickers: dict[str, yf.Ticker] = {}

    def _ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]

    def _history(self, symbol: str, period: str) -> pd.DataFrame:
        # yfinance reports a failed fetch (delisted symbol, network error) by logging it and
        # returning an empty frame; raise what it logged so callers can show the cause.
        with _logged_errors() as errors:
            history = self._ticker(symbol).history(period=period)
        if history.empty and errors:
            raise YFinanceError("; ".join(errors))
        return history

    def expirations(self, symbol: str) -> list[date]:
        raw = self._ticker(symbol).options
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in raw]

    def option_chain(self, symbol: str, expiry: date) -> OptionChain:
        chain = self._ticker(symbol).option_chain(expiry.strftime("%Y-%m-%d"))
        return OptionChain(
            expiry=expiry,
            calls=_normalise_chain(chain.calls),
            puts=_normalise_chain(chain.puts),
        )

    def spot(self, symbol: str) -> float | None:
        today = self._history(symbol, "1d")
        if today.empty:
            return None
        return float(today["Close"].iloc[-1])

    def price_history(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        history = self._history(symbol, period).rename(columns=str.lower)
        return history[list(PRICE_COLUMNS)]

    def next_earnings(self, symbol: str, after: date) -> date | None:
        # Yahoo has no calendar for ETFs and indices; yfinance logs that 404 as an error.
        logger = logging.getLogger("yfinance")
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            dates = self._ticker(symbol).calendar.get("Earnings Date") or []
        except Exception:
            return None
        finally:
            logger.setLevel(level)
        days = [d.date() if isinstance(d, datetime) else d for d in dates if isinstance(d, date)]
        return min((d for d in days if d >= after), default=None)


class _ErrorLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _logged_errors() -> Iterator[list[str]]:
    """Messages yfinance logs at ERROR or above inside the block, collected instead of printed."""
    handler = _ErrorLog()
    logger = logging.getLogger("yfinance")
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


def _normalise_chain(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=_CHAIN_RENAME)[list(CHAIN_COLUMNS)]
