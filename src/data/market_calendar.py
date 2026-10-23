"""US exchange dates. DTEs and exit dates are counted on this calendar."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


def market_date(now: datetime | None = None) -> date:
    """Today's date on US exchanges, whatever the local time zone.

    ``now`` must be timezone-aware; it defaults to the current time.
    """
    return (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ).date()


def previous_trading_day(day: date) -> date:
    """The weekday before ``day``. Exchange holidays are not accounted for."""
    day -= timedelta(days=1)
    while day.weekday() >= 5:  # Saturday, Sunday
        day -= timedelta(days=1)
    return day
