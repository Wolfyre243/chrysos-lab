import json
from datetime import timedelta

import pandas as pd
import pytest

from analytics.volatility import yang_zhang
from calculators import CALCULATORS
from calculators.calendar_iv_crush import (
    SPEC,
    Inputs,
    Params,
    _recommend,
    evaluate,
    run,
    select_expiries,
)
from cli.calendar_iv_crush import format_result
from errors import CalculatorError

from conftest import AS_OF, SPOT, make_chain, make_history


def _inputs(chains, history, spot=SPOT):
    return Inputs(symbol="TEST", as_of=AS_OF, spot=spot, chains=chains, price_history=history)


# --- expiry selection ------------------------------------------------------


def test_select_expiries_keeps_up_to_first_beyond_cutoff_and_drops_today():
    expiries = [AS_OF + timedelta(days=d) for d in (60, 0, 11, 46, 4, 25)]
    selected = select_expiries(expiries, AS_OF)
    assert [(e - AS_OF).days for e in selected] == [4, 11, 25, 46]


def test_select_expiries_includes_exact_cutoff():
    expiries = [AS_OF + timedelta(days=d) for d in (7, 45, 52)]
    assert [(e - AS_OF).days for e in select_expiries(expiries, AS_OF)] == [7, 45]


def test_select_expiries_raises_without_far_expiry():
    with pytest.raises(CalculatorError, match="Not enough option data"):
        select_expiries([AS_OF + timedelta(days=d) for d in (4, 11, 30)], AS_OF)


# --- core values -----------------------------------------------------------


def test_evaluate_values(chains, history):
    result = evaluate(_inputs(chains, history))

    assert [p.dte for p in result.term_structure] == [4, 11, 25, 46]
    assert result.front_expiry == AS_OF + timedelta(days=4)

    iv45 = 0.45 - 0.10 * 20 / 21
    assert result.ts_slope_0_45.value == pytest.approx((iv45 - 0.80) / (45 - 4))
    assert result.ts_slope_0_45.passed

    iv30 = 0.45 - 0.10 * 5 / 21
    rv30 = yang_zhang(history).iloc[-1]
    assert result.iv30 == pytest.approx(iv30)
    assert result.rv30 == pytest.approx(rv30)
    assert result.iv30_rv30.value == pytest.approx(iv30 / rv30)

    assert result.avg_volume.value == pytest.approx(2_000_000)
    assert result.avg_volume.passed

    # ATM call mid 2.1 + put mid 2.1
    assert result.straddle == pytest.approx(4.2)
    assert result.expected_move == pytest.approx(0.042)


def test_atm_uses_strike_closest_to_spot(chains, history):
    result = evaluate(_inputs(chains, history, spot=103.0))
    assert result.term_structure[0].atm_iv == pytest.approx(0.80)
    result = evaluate(_inputs(chains, history, spot=107.0))
    assert result.term_structure[0].atm_iv == pytest.approx(0.90)


def test_empty_front_chain_is_skipped(history):
    empty = pd.DataFrame(columns=["strike", "iv", "bid", "ask"])
    front = make_chain(AS_OF + timedelta(days=4), 0.8)
    front = type(front)(front.expiry, empty, empty)
    chains = [
        front,
        make_chain(AS_OF + timedelta(days=11), 0.6, bid=3.0, ask=3.2),
        make_chain(AS_OF + timedelta(days=46), 0.35),
    ]
    result = evaluate(_inputs(chains, history))
    assert result.front_expiry == AS_OF + timedelta(days=11)
    assert result.straddle == pytest.approx(6.2)


def test_missing_quote_gives_no_expected_move(history):
    chains = [
        make_chain(AS_OF + timedelta(days=4), 0.8, bid=float("nan"), ask=2.2),
        make_chain(AS_OF + timedelta(days=46), 0.35),
    ]
    result = evaluate(_inputs(chains, history))
    assert result.straddle is None
    assert result.expected_move is None


@pytest.mark.parametrize(
    "front, back",
    [
        ({"atm_iv": 0.8, "bid": 0.0, "ask": 0.0}, {"atm_iv": 0.35}),  # cleared quotes, front
        ({"atm_iv": 0.8}, {"atm_iv": 0.35, "bid": 0.0, "ask": 0.0}),  # cleared quotes, back
        ({"atm_iv": 1e-5}, {"atm_iv": 0.35}),  # Yahoo's IV floor
    ],
)
def test_stale_quotes_raise(history, front, back):
    chains = [
        make_chain(AS_OF + timedelta(days=4), **front),
        make_chain(AS_OF + timedelta(days=46), **back),
    ]
    with pytest.raises(CalculatorError, match="Stale option quotes"):
        evaluate(_inputs(chains, history))


def test_zero_bid_alone_is_not_stale(history):
    chains = [
        make_chain(AS_OF + timedelta(days=4), 0.8, bid=0.0, ask=0.1),
        make_chain(AS_OF + timedelta(days=46), 0.35),
    ]
    assert evaluate(_inputs(chains, history)).straddle == pytest.approx(0.1)


# --- verdict ---------------------------------------------------------------


@pytest.mark.parametrize(
    "volume, iv_rv, slope, expected",
    [
        (True, True, True, "Recommended"),
        (True, False, True, "Consider"),
        (False, True, True, "Consider"),
        (False, False, True, "Avoid"),
        (True, True, False, "Avoid"),
        (True, False, False, "Avoid"),
        (False, False, False, "Avoid"),
    ],
)
def test_recommend_truth_table(volume, iv_rv, slope, expected):
    assert _recommend(volume, iv_rv, slope) == expected


def test_params_thresholds_drive_checks(chains, history):
    strict = Params(min_avg_volume=5_000_000, min_iv30_rv30=0.0)
    result = evaluate(_inputs(chains, history), strict)
    assert not result.avg_volume.passed
    assert result.iv30_rv30.passed
    assert result.recommendation == "Consider"

    result = evaluate(_inputs(chains, history), Params(max_ts_slope_0_45=-1.0))
    assert result.recommendation == "Avoid"


# --- errors ----------------------------------------------------------------


def test_single_expiry_raises(history):
    chains = [make_chain(AS_OF + timedelta(days=50), 0.4)]
    with pytest.raises(CalculatorError, match="term structure"):
        evaluate(_inputs(chains, history))


def test_short_history_raises(chains):
    with pytest.raises(CalculatorError, match="price history"):
        evaluate(_inputs(chains, make_history(days=20)))


# --- run() with a fake provider ---------------------------------------------


class FakeProvider:
    def __init__(self, chains, history, spot=SPOT):
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


def test_run_fetches_only_selected_expiries(chains, history):
    provider = FakeProvider(chains, history)
    result = run(" test ", provider=provider, as_of=AS_OF)
    assert result.symbol == "TEST"
    assert [(e - AS_OF).days for e in provider.fetched] == [4, 11, 25, 46]


def test_run_errors(chains, history):
    with pytest.raises(CalculatorError, match="No stock symbol"):
        run("  ", provider=FakeProvider(chains, history), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="No options found"):
        run("X", provider=FakeProvider([], history), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="underlying stock price"):
        run("X", provider=FakeProvider(chains, history, spot=None), as_of=AS_OF)


# --- output ----------------------------------------------------------------


def test_to_dict_is_json_serialisable(chains, history):
    data = evaluate(_inputs(chains, history)).to_dict()
    json.dumps(data)
    assert data["front_expiry"] == (AS_OF + timedelta(days=4)).isoformat()
    assert data["iv30_rv30"]["op"] == ">="
    assert data["calculator_version"] == SPEC.version


def test_format_result_layout(chains, history):
    lines = format_result(evaluate(_inputs(chains, history))).splitlines()
    assert lines[0] == "TEST"
    assert lines[1] in ("Recommended", "Consider", "Avoid")
    assert lines[3].split() == ["Item", "Result", "Value", "Threshold"]
    assert any(line.startswith("expected_move") and "4.20%" in line for line in lines)


def test_registry_lists_calculator():
    assert CALCULATORS["calendar_iv_crush"].SPEC is SPEC
