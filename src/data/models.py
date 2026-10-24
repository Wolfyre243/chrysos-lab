"""Provider-neutral market data shapes.

Calculators only ever see these, so swapping yfinance for moomoo (or a cached
Chrysos ChainSnapshot) needs no calculator changes.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

# Columns every provider must supply for each side of an option chain. volume is
# contracts traded today (NaN when none); open_interest is as of the last close.
CHAIN_COLUMNS = ("strike", "iv", "bid", "ask", "volume", "open_interest")

OptionSide = Literal["call", "put"]
SIDES: tuple[OptionSide, ...] = ("call", "put")

# Columns every provider must supply for daily price history (matches the PriceBar entity).
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OptionChain:
    """Calls and puts for a single expiry. Each frame has at least CHAIN_COLUMNS."""

    expiry: date
    calls: pd.DataFrame
    puts: pd.DataFrame

    def side(self, side: OptionSide) -> pd.DataFrame:
        return self.calls if side == "call" else self.puts
