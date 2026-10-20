from datetime import date, datetime

import pandas as pd
import yfinance as yf

from data.models import CHAIN_COLUMNS, PRICE_COLUMNS, OptionChain

_CHAIN_RENAME = {"impliedVolatility": "iv"}


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


def _normalise_chain(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=_CHAIN_RENAME)[list(CHAIN_COLUMNS)]
