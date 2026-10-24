import logging
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from data.models import CHAIN_COLUMNS, PRICE_COLUMNS, OptionChain

_CHAIN_RENAME = {"impliedVolatility": "iv", "openInterest": "open_interest"}


class YFinanceProvider:
    """MarketDataProvider backed by Yahoo Finance (free, delayed, unofficial)."""

    def __init__(self) -> None:
        self._tickers: dict[str, yf.Ticker] = {}

    def _ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]

    def expirations(self, symbol: str) -> list[date]:
        try:
            raw = self._ticker(symbol).options
        except Exception:
            return []
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in raw]

    def option_chain(self, symbol: str, expiry: date) -> OptionChain:
        chain = self._ticker(symbol).option_chain(expiry.strftime("%Y-%m-%d"))
        return OptionChain(
            expiry=expiry,
            calls=_normalise_chain(chain.calls),
            puts=_normalise_chain(chain.puts),
        )

    def spot(self, symbol: str) -> float | None:
        today = self._ticker(symbol).history(period="1d")
        if today.empty:
            return None
        return float(today["Close"].iloc[-1])

    def price_history(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        history = self._ticker(symbol).history(period=period)
        history = history.rename(columns=str.lower)
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


def _normalise_chain(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=_CHAIN_RENAME)[list(CHAIN_COLUMNS)]
