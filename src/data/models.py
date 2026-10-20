"""Provider-neutral market data shapes.

Calculators only ever see these, so swapping yfinance for moomoo (or a cached
Chrysos ChainSnapshot) needs no calculator changes.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

# Columns every provider must supply for each side of an option chain.
CHAIN_COLUMNS = ("strike", "iv", "bid", "ask")

# Columns every provider must supply for daily price history (matches the PriceBar entity).
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OptionChain:
    """Calls and puts for a single expiry. Each frame has at least CHAIN_COLUMNS."""

    expiry: date
    calls: pd.DataFrame
    puts: pd.DataFrame
