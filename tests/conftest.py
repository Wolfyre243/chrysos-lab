from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data.models import OptionChain

AS_OF = date(2026, 10, 20)
SPOT = 100.0


def make_chain(expiry: date, atm_iv: float, bid: float = 2.0, ask: float = 2.2) -> OptionChain:
    """Three strikes around SPOT; the ATM strike (100) carries ``atm_iv`` and the given quotes."""
    side = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "iv": [atm_iv + 0.1, atm_iv, atm_iv + 0.1],
            "bid": [10.0, bid, 0.5],
            "ask": [10.5, ask, 0.6],
        }
    )
    return OptionChain(expiry=expiry, calls=side.copy(), puts=side.copy())


def make_history(days: int = 63, volume: float = 2_000_000, seed: int = 7) -> pd.DataFrame:
    """Seeded random-walk daily OHLCV bars."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
    open_ = close * np.exp(rng.normal(0, 0.005, days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, days)))
    index = pd.bdate_range(end=AS_OF, periods=days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


class FakeProvider:
    """MarketDataProvider over in-memory chains; records which expiries were fetched."""

    def __init__(self, chains, history=None, spot=SPOT):
        self._chains = {c.expiry: c for c in chains}
        self._history = history
        self._spot = spot
        self.fetched: list = []

    def expirations(self, symbol):
        return list(self._chains)

    def option_chain(self, symbol, expiry):
        self.fetched.append(expiry)
        return self._chains[expiry]

    def spot(self, symbol):
        return self._spot

    def price_history(self, symbol, period="3mo"):
        return self._history


@pytest.fixture
def chains() -> list[OptionChain]:
    """Expiries at 0 (dropped), 4, 11, 25, 46 and 60 (beyond the cut) DTE; downward-sloping IV."""
    ivs = {0: 0.9, 4: 0.80, 11: 0.60, 25: 0.45, 46: 0.35, 60: 0.30}
    return [make_chain(AS_OF + timedelta(days=d), iv) for d, iv in ivs.items()]


@pytest.fixture
def history() -> pd.DataFrame:
    return make_history()
