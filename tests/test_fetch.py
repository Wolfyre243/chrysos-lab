from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from data.fetch import market_date


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
