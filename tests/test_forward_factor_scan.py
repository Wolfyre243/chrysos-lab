import json
from datetime import timedelta

import pandas as pd
import pytest

from analytics.chain import common_strikes
from calculators import forward_factor
from cli.forward_factor import format_scan
from data.models import OptionChain
from errors import CalculatorError
from scanners.forward_factor import Inputs, Params, pick_legs, run, scan

from conftest import AS_OF, SPOT, FakeProvider, make_chain


def _days(expiries):
    return [(e - AS_OF).days for e in expiries]


def _expiries(*dtes):
    return [AS_OF + timedelta(days=d) for d in dtes]


@pytest.fixture
def ff_chains() -> list[OptionChain]:
    """Fronts at 28 and 35 DTE, back at 63; 7 and 120 DTE sit outside the default windows."""
    ivs = {7: 0.70, 28: 0.40, 35: 0.35, 63: 0.30, 120: 0.25}
    return [make_chain(AS_OF + timedelta(days=d), iv) for d, iv in ivs.items()]


def _scan(chains, params=Params()):
    return scan(Inputs(symbol="TEST", as_of=AS_OF, spot=SPOT, chains=chains), params)


# --- leg selection ---------------------------------------------------------


def test_pick_legs_uses_default_windows():
    legs = pick_legs(_expiries(0, 7, 14, 28, 35, 49, 63, 91, 120), AS_OF)
    assert _days(legs.front) == [28, 35]
    assert _days(legs.back) == [63]
    assert (legs.front_window, legs.back_window) == ((21, 45), (50, 90))
    assert not legs.widened


def test_pick_legs_widens_only_empty_windows():
    legs = pick_legs(_expiries(30, 120), AS_OF)
    assert _days(legs.front) == [30]
    assert _days(legs.back) == [120]
    assert legs.front_window == (21, 45)  # front had an expiry, so it never moved
    assert legs.back_window == (15, 125)
    assert legs.widened


def test_pick_legs_widens_both_when_both_empty():
    legs = pick_legs(_expiries(10, 100), AS_OF)
    assert (_days(legs.front), _days(legs.back)) == ([10], [100])
    assert (legs.front_window, legs.back_window) == ((7, 59), (36, 104))


def test_pick_legs_raises_when_nothing_fits():
    with pytest.raises(CalculatorError, match="No expiry pair fits"):
        pick_legs(_expiries(0, 5), AS_OF)


# --- strikes ---------------------------------------------------------------


def _chain(dte, call_strikes, put_strikes, iv=0.3):
    def side(strikes):
        n = len(strikes)
        return pd.DataFrame({"strike": strikes, "iv": [iv] * n, "bid": [1.0] * n, "ask": [1.1] * n})

    return OptionChain(AS_OF + timedelta(days=dte), side(call_strikes), side(put_strikes))


def test_common_strikes_needs_every_given_side_of_every_chain():
    front = _chain(30, [90.0, 100.0, 110.0], [90.0, 100.0])
    back = _chain(60, [100.0, 110.0, 120.0], [90.0, 100.0, 110.0, 120.0])
    assert common_strikes([front, back], spot=104.0, n=5) == [100.0]
    assert common_strikes([front, back], spot=104.0, n=5, sides=("call",)) == [100.0, 110.0]
    assert common_strikes([front, back], spot=104.0, n=5, sides=("put",)) == [90.0, 100.0]
    assert common_strikes([front, back], spot=104.0, n=1, sides=("call",)) == [100.0]


# --- scan ------------------------------------------------------------------


def _trades(result):
    return {(c.side, c.front_dte, c.back_dte, c.strike) for c in result.candidates}


def _put_iv_shift(chain, shift):
    """Same chain with every put IV moved by ``shift``."""
    return OptionChain(chain.expiry, chain.calls, chain.puts.assign(iv=chain.puts["iv"] + shift))


def test_scan_ranks_every_side_pair_and_strike(ff_chains):
    result = _scan(ff_chains)

    assert _trades(result) == {
        (side, f, 63, k) for side in ("call", "put") for f in (28, 35) for k in (90.0, 100.0, 110.0)
    }
    ffs = [c.result.ff.value for c in result.candidates]
    assert ffs == sorted(ffs, reverse=True)
    assert result.min_ff == 0.2
    assert result.skipped_stale == 0


def test_scan_feeds_each_sides_iv_to_calculator(ff_chains):
    ff_chains[1] = _put_iv_shift(ff_chains[1], 0.02)  # 28 DTE puts at 42%, calls at 40%
    result = _scan(ff_chains, Params(strikes=1))
    by_side = {c.side: c for c in result.candidates if c.front_dte == 28}

    assert by_side["call"].strike == by_side["put"].strike == 100.0
    assert by_side["call"].result == forward_factor.evaluate(forward_factor.Inputs(28, 0.40, 63, 0.30))
    assert by_side["put"].result == forward_factor.evaluate(forward_factor.Inputs(28, 0.40 + 0.02, 63, 0.30))
    assert result.candidates[0].side == "put"  # richer front put IV ranks first


def test_scan_puts_undefined_ff_last():
    chains = [
        make_chain(AS_OF + timedelta(days=28), 0.90),  # forward variance negative vs the back
        make_chain(AS_OF + timedelta(days=35), 0.35),
        make_chain(AS_OF + timedelta(days=63), 0.30),
    ]
    result = _scan(chains, Params(strikes=1))
    assert [c.front_dte for c in result.candidates] == [35, 35, 28, 28]
    assert all(c.result.ff is None for c in result.candidates[-2:])


def test_scan_skips_stale_quotes_per_side(ff_chains):
    stale = make_chain(AS_OF + timedelta(days=28), 0.40, bid=0.0, ask=0.0)  # ATM cleared
    ff_chains[1] = OptionChain(stale.expiry, stale.calls, ff_chains[1].puts)  # calls only
    result = _scan(ff_chains)
    assert result.skipped_stale == 1
    assert ("call", 28, 63, 100.0) not in _trades(result)
    assert ("put", 28, 63, 100.0) in _trades(result)


def test_scan_raises_when_every_trade_is_stale():
    chains = [
        make_chain(AS_OF + timedelta(days=28), 0.40, bid=0.0, ask=0.0),
        make_chain(AS_OF + timedelta(days=63), 0.30),
    ]
    with pytest.raises(CalculatorError, match="Stale option quotes"):
        _scan(chains, Params(strikes=1))


def test_scan_raises_without_shared_strikes():
    chains = [_chain(28, [90.0, 100.0], [90.0, 100.0]), _chain(63, [105.0, 115.0], [105.0, 115.0])]
    with pytest.raises(CalculatorError, match="No strike is listed"):
        _scan(chains)


# --- run() with a fake provider ---------------------------------------------


def test_run_fetches_only_window_expiries(ff_chains):
    provider = FakeProvider(ff_chains)
    result = run(" test ", provider=provider, as_of=AS_OF)
    assert result.symbol == "TEST"
    assert sorted(_days(provider.fetched)) == [28, 35, 63]


def test_run_errors(ff_chains):
    with pytest.raises(CalculatorError, match="No stock symbol"):
        run("  ", provider=FakeProvider(ff_chains), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="No options found"):
        run("X", provider=FakeProvider([]), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="underlying stock price"):
        run("X", provider=FakeProvider(ff_chains, spot=None), as_of=AS_OF)


# --- output ----------------------------------------------------------------


def test_to_dict_is_json_serialisable(ff_chains):
    data = _scan(ff_chains).to_dict()
    json.dumps(data)
    assert data["front_window"] == [21, 45]
    best = data["candidates"][0]
    assert best["front_expiry"] == (AS_OF + timedelta(days=best["front_dte"])).isoformat()
    assert best["result"]["ff"]["op"] == ">="
    assert best["side"] in ("call", "put")


def test_format_scan_compares_sides(ff_chains):
    ff_chains[1] = _put_iv_shift(ff_chains[1], 0.02)
    scan_result = _scan(ff_chains, Params(strikes=1))
    lines = format_scan(scan_result).splitlines()

    header = next(i for i, line in enumerate(lines) if line.startswith("Side"))
    call_row, put_row = lines[header + 2].split(), lines[header + 3].split()
    best = {c.side: c for c in reversed(scan_result.candidates)}  # first-ranked of each side wins
    assert call_row[:4] == ["call", "2", "2", f"{best['call'].result.ff.value:.3f}"]
    assert put_row[:4] == ["put", "2", "2", f"{best['put'].result.ff.value:.3f}"]
    assert any(line.startswith("#1  put calendar") for line in lines)


def test_format_scan_lists_ranked_trades_and_top_three_breakdowns(ff_chains):
    text = format_scan(_scan(ff_chains))
    lines = text.splitlines()
    assert lines[0] == "TEST"
    assert "trades pass FF >= 0.2" in lines[1]
    assert "Rank" in text and "PASS" in text
    assert all(f"#{i} " in text for i in (1, 2, 3))
    assert "#4 " not in text
    assert text.count("forward variance") == 3


def test_format_scan_respects_top(ff_chains):
    text = format_scan(_scan(ff_chains), top=2)
    assert "#2 " in text and "#3 " not in text


def test_format_scan_shows_undefined_ff():
    chains = [make_chain(AS_OF + timedelta(days=28), 0.90), make_chain(AS_OF + timedelta(days=63), 0.30)]
    text = format_scan(_scan(chains, Params(strikes=1)))
    assert "FAIL" in text
    assert "undefined (negative fwd variance)" in text
