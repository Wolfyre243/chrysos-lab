import math

import pandas as pd
import pytest

from analytics.chain import contracts, mid, quote_at_delta
from analytics.greeks import delta, norm_cdf

from conftest import AS_OF, make_chain


def test_norm_cdf():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.0) == pytest.approx(0.841344746)


def test_atm_call_delta_is_just_above_half():
    assert delta("call", 100.0, 100.0, 0.30, 30 / 365) == pytest.approx(norm_cdf(0.5 * 0.30 * math.sqrt(30 / 365)))
    assert 0.5 < delta("call", 100.0, 100.0, 0.30, 30 / 365) < 0.53


def test_put_delta_is_call_minus_one():
    call = delta("call", 100.0, 110.0, 0.40, 0.25)
    assert delta("put", 100.0, 110.0, 0.40, 0.25) == pytest.approx(call - 1.0)


def test_delta_known_value():
    # d1 = (ln(100/110) + (0.05 + 0.08) * 0.5) / (0.4 * sqrt(0.5)) = -0.107163, N(d1) = 0.457330
    assert delta("call", 100.0, 110.0, 0.40, 0.5, rate=0.05) == pytest.approx(0.457330, abs=1e-5)


def test_delta_without_time_or_vol_is_intrinsic():
    assert delta("call", 105.0, 100.0, 0.30, 0.0) == 1.0
    assert delta("call", 95.0, 100.0, 0.0, 0.1) == 0.0
    assert delta("put", 95.0, 100.0, 0.0, 0.1) == -1.0


def test_mid_and_contracts():
    quote = pd.Series({"bid": 2.0, "ask": 2.2, "volume": float("nan"), "open_interest": 1_200.0})
    assert mid(quote) == pytest.approx(2.1)
    assert contracts(quote, "volume") == 0.0
    assert contracts(quote, "open_interest") == 1_200.0


def test_quote_at_delta_picks_closest_listed_strike():
    calls = make_chain(AS_OF, 0.40).calls  # strikes 90 / 100 / 110
    t = 28 / 365
    assert quote_at_delta(calls, "call", 100.0, t, 0.35, {90.0, 100.0, 110.0})["strike"] == 110.0
    assert quote_at_delta(calls, "call", 100.0, t, 0.35, {90.0, 100.0})["strike"] == 100.0
    assert quote_at_delta(calls, "put", 100.0, t, -0.35, {90.0, 100.0, 110.0})["strike"] == 100.0
    assert quote_at_delta(calls, "call", 100.0, t, 0.35, set()) is None
