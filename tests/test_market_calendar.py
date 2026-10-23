from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from data.market_calendar import market_date, previous_trading_day


@pytest.mark.parametrize(
    "now, expected",
    [
        # 00:29 in Kuala Lumpur on the 24th is 12:29 on the 23rd in New York.
        (datetime(2026, 10, 24, 0, 29, tzinfo=ZoneInfo("Asia/Kuala_Lumpur")), date(2026, 10, 23)),
        (datetime(2026, 10, 24, 3, 59, tzinfo=timezone.utc), date(2026, 10, 23)),  # 23:59 EDT
        (datetime(2026, 10, 24, 4, 0, tzinfo=timezone.utc), date(2026, 10, 24)),  # midnight EDT
        (datetime(2026, 12, 1, 4, 59, tzinfo=timezone.utc), date(2026, 11, 30)),  # 23:59 EST
    ],
)
def test_market_date_uses_new_york_calendar(now, expected):
    assert market_date(now) == expected


def test_market_date_defaults_to_now():
    assert market_date() == datetime.now(ZoneInfo("America/New_York")).date()


@pytest.mark.parametrize(
    "day, expected",
    [
        (date(2026, 11, 20), date(2026, 11, 19)),  # Friday expiry -> Thursday
        (date(2026, 11, 16), date(2026, 11, 13)),  # Monday -> previous Friday
        (date(2026, 11, 15), date(2026, 11, 13)),  # Sunday -> Friday
    ],
)
def test_previous_trading_day_skips_weekends(day, expected):
    assert previous_trading_day(day) == expected
