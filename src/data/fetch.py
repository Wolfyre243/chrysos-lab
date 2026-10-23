"""Validated fetch steps shared by calculators and scanners that pull their own data."""

from datetime import date

from data.provider import MarketDataProvider
from errors import CalculatorError


def resolve(symbol: str, provider: MarketDataProvider | None) -> tuple[str, MarketDataProvider]:
    """Normalised symbol, plus ``provider`` or the default (yfinance) when None."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise CalculatorError("No stock symbol provided.")
    if provider is None:
        from data.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider()
    return symbol, provider


def require_expirations(provider: MarketDataProvider, symbol: str) -> list[date]:
    expiries = provider.expirations(symbol)
    if not expiries:
        raise CalculatorError(f"No options found for stock symbol '{symbol}'.")
    return expiries


def require_spot(provider: MarketDataProvider, symbol: str) -> float:
    spot = provider.spot(symbol)
    if spot is None:
        raise CalculatorError("Unable to retrieve underlying stock price.")
    return spot
