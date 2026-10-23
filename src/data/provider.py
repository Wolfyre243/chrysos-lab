from datetime import date
from typing import Protocol

import pandas as pd

from data.models import OptionChain


class MarketDataProvider(Protocol):
    """Adapter interface in front of any market data source."""

    def expirations(self, symbol: str) -> list[date]:
        """All listed option expiries for the symbol (empty if none)."""
        ...

    def option_chain(self, symbol: str, expiry: date) -> OptionChain: ...

    def spot(self, symbol: str) -> float | None:
        """Latest underlying price, or None if unavailable."""
        ...

    def price_history(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Daily bars indexed by date, with PRICE_COLUMNS."""
        ...

    def next_earnings(self, symbol: str, after: date) -> date | None:
        """First scheduled earnings date on or after ``after``, or None if unknown or none (ETFs)."""
        ...
