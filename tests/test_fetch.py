import pytest

from data.fetch import require_expirations, require_spot
from errors import CalculatorError

from conftest import FakeProvider


class BrokenProvider(FakeProvider):
    """Every fetch fails the way a rate-limited or offline data source would."""

    def expirations(self, symbol):
        raise ConnectionError("Too Many Requests")

    def spot(self, symbol):
        raise ConnectionError("Too Many Requests")


def test_require_expirations_keeps_provider_error():
    with pytest.raises(CalculatorError, match="Unable to retrieve options for stock symbol 'X'") as info:
        require_expirations(BrokenProvider([]), "X")
    assert isinstance(info.value.__cause__, ConnectionError)


def test_require_spot_keeps_provider_error():
    with pytest.raises(CalculatorError, match="underlying stock price") as info:
        require_spot(BrokenProvider([]), "X")
    assert isinstance(info.value.__cause__, ConnectionError)


def test_missing_data_has_no_cause():
    with pytest.raises(CalculatorError, match="No options found") as info:
        require_expirations(FakeProvider([]), "X")
    assert info.value.__cause__ is None
