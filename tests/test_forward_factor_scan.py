import json
from datetime import date, timedelta

import pandas as pd
import pytest

from analytics.chain import common_strikes
from calculators import forward_factor
from cli.forward_factor import format_scan
from data.market_calendar import previous_trading_day
from data.models import OptionChain
from errors import CalculatorError
from scanners.forward_factor import Inputs, Params, pick_legs, run, scan

from conftest import AS_OF, SPOT, FakeProvider, make_chain

# make_chain quotes: strike 90 spread 4.9% of mid, 100 (ATM) 9.5%, 110 18.2%. With the
# default 10% cap, trades using the 110 strike fail on spread. The 35-delta double
# calendar lands on the 100 put and the 110 call.
DOUBLE_STRIKES = (100.0, 110.0)


def _days(expiries):
    return [(e - AS_OF).days for e in expiries]


def _expiries(*dtes):
    return [AS_OF + timedelta(days=d) for d in dtes]


@pytest.fixture
def ff_chains() -> list[OptionChain]:
    """Fronts at 28 and 35 DTE, back at 63; 7 and 120 DTE sit outside the default windows."""
    ivs = {7: 0.70, 28: 0.40, 35: 0.35, 63: 0.30, 120: 0.25}
    return [make_chain(AS_OF + timedelta(days=d), iv) for d, iv in ivs.items()]


def _scan(chains, params=Params(), earnings=None):
    return scan(Inputs(symbol="TEST", as_of=AS_OF, spot=SPOT, chains=chains, earnings=earnings), params)


def _blocked(c):
    return "earnings" in c.flags or "spread" in c.flags


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
    return {(c.structure, c.front_dte, c.back_dte, c.strikes) for c in result.candidates}


def _put_iv_shift(chain, shift):
    """Same chain with every put IV moved by ``shift``."""
    return OptionChain(chain.expiry, chain.calls, chain.puts.assign(iv=chain.puts["iv"] + shift))


def test_scan_scores_every_structure_pair_and_strike(ff_chains):
    result = _scan(ff_chains)

    singles = {(s, f, 63, (k,)) for s in ("call", "put") for f in (28, 35) for k in (90.0, 100.0, 110.0)}
    doubles = {("double_35d", f, 63, DOUBLE_STRIKES) for f in (28, 35)}
    assert _trades(result) == singles | doubles
    assert result.min_ff == 0.2
    assert result.max_spread == 0.10
    assert result.skipped_stale == 0


def test_scan_ranks_blocked_trades_last_then_by_ff(ff_chains):
    result = _scan(ff_chains)
    blocked = [_blocked(c) for c in result.candidates]
    assert blocked == sorted(blocked)  # every unblocked trade before every blocked one
    assert any(blocked) and not all(blocked)
    for group in (False, True):
        ffs = [c.result.ff.value for c in result.candidates if _blocked(c) == group]
        assert ffs == sorted(ffs, reverse=True)


def test_scan_feeds_each_sides_iv_to_calculator(ff_chains):
    ff_chains[1] = _put_iv_shift(ff_chains[1], 0.02)  # 28 DTE puts at 42%, calls at 40%
    result = _scan(ff_chains, Params(strikes=1))
    by_side = {c.structure: c for c in result.candidates if c.front_dte == 28 and c.structure != "double_35d"}

    assert by_side["call"].strikes == by_side["put"].strikes == (100.0,)
    assert by_side["call"].result == forward_factor.evaluate(forward_factor.Inputs(28, 0.40, 63, 0.30))
    assert by_side["put"].result == forward_factor.evaluate(forward_factor.Inputs(28, 0.40 + 0.02, 63, 0.30))
    assert result.candidates[0].structure == "put"  # richer front put IV ranks first


def test_scan_double_calendar_averages_wing_ivs(ff_chains):
    result = _scan(ff_chains)
    double = next(c for c in result.candidates if c.structure == "double_35d" and c.front_dte == 28)

    assert double.strikes == DOUBLE_STRIKES  # (put, call)
    # Front: put 100 at 40%, call 110 at 50%. Back: put 100 at 30%, call 110 at 40%.
    expected = forward_factor.evaluate(forward_factor.Inputs(28, (0.40 + 0.50) / 2, 63, (0.30 + 0.40) / 2))
    assert double.result.fwd_var == pytest.approx(expected.fwd_var)
    assert double.result.ff.value == pytest.approx(expected.ff.value)
    assert not double.atm
    assert "spread" in double.flags  # the 110 call legs are wide


def test_scan_marks_one_atm_call_per_pair(ff_chains):
    atm = [c for c in _scan(ff_chains).candidates if c.atm]
    assert sorted((c.front_dte, c.structure, c.strikes) for c in atm) == [
        (28, "call", (100.0,)),
        (35, "call", (100.0,)),
    ]


def test_scan_close_by_is_weekday_before_front_expiry(ff_chains):
    result = _scan(ff_chains)
    assert all(c.close_by == previous_trading_day(c.front_expiry) for c in result.candidates)
    front_28 = next(c for c in result.candidates if c.front_dte == 28)
    assert front_28.close_by == date(2026, 11, 16)  # front expiry Tuesday 17 Nov


@pytest.mark.parametrize(
    "earnings_dte, crosses",
    [
        (40, True),  # between the fronts and the back
        (63, True),  # on the back expiry
        (0, True),  # today
        (70, False),  # after the back expiry
        (None, False),  # no earnings known
    ],
)
def test_scan_earnings_guard(ff_chains, earnings_dte, crosses):
    earnings = None if earnings_dte is None else AS_OF + timedelta(days=earnings_dte)
    result = _scan(ff_chains, earnings=earnings)
    assert result.earnings == earnings
    assert all(c.crosses_earnings is crosses for c in result.candidates)
    assert all(("earnings" in c.flags) is crosses for c in result.candidates)
    assert not any(c.passed for c in result.candidates) if crosses else any(c.passed for c in result.candidates)


def test_scan_spread_check(ff_chains):
    wide = [c for c in _scan(ff_chains).candidates if "spread" in c.flags]
    assert {c.strikes for c in wide} == {(110.0,), DOUBLE_STRIKES}
    assert all(c.spread.value == pytest.approx(0.1 / 0.55) for c in wide)

    loose = _scan(ff_chains, Params(max_spread=0.2))
    assert not any("spread" in c.flags for c in loose.candidates)


def test_scan_puts_undefined_ff_last():
    chains = [
        make_chain(AS_OF + timedelta(days=28), 0.90),  # forward variance negative vs the back
        make_chain(AS_OF + timedelta(days=35), 0.35),
        make_chain(AS_OF + timedelta(days=63), 0.30),
    ]
    result = _scan(chains, Params(strikes=1, max_spread=1.0))
    assert [c.front_dte for c in result.candidates] == [35, 35, 35, 28, 28, 28]
    assert all(c.result.ff is None for c in result.candidates[3:])


def test_scan_skips_stale_quotes_per_side(ff_chains):
    stale = make_chain(AS_OF + timedelta(days=28), 0.40, bid=0.0, ask=0.0)  # ATM cleared
    ff_chains[1] = OptionChain(stale.expiry, stale.calls, ff_chains[1].puts)  # calls only
    result = _scan(ff_chains)
    assert result.skipped_stale == 1
    assert ("call", 28, 63, (100.0,)) not in _trades(result)
    assert ("put", 28, 63, (100.0,)) in _trades(result)


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
    assert result.earnings is None


def test_run_passes_next_earnings(ff_chains):
    earnings = AS_OF + timedelta(days=40)
    result = run("TEST", provider=FakeProvider(ff_chains, earnings=earnings), as_of=AS_OF)
    assert result.earnings == earnings
    assert all(c.crosses_earnings for c in result.candidates)


def test_run_errors(ff_chains):
    with pytest.raises(CalculatorError, match="No stock symbol"):
        run("  ", provider=FakeProvider(ff_chains), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="No options found"):
        run("X", provider=FakeProvider([]), as_of=AS_OF)
    with pytest.raises(CalculatorError, match="underlying stock price"):
        run("X", provider=FakeProvider(ff_chains, spot=None), as_of=AS_OF)


# --- output ----------------------------------------------------------------


def test_to_dict_is_strict_json(ff_chains):
    stale_bid = make_chain(AS_OF + timedelta(days=35), 0.35, bid=0.0, ask=0.1)  # mid > 0
    ff_chains[2] = stale_bid
    data = _scan(ff_chains, earnings=AS_OF + timedelta(days=40)).to_dict()
    json.dumps(data, allow_nan=False)
    assert data["front_window"] == [21, 45]
    assert data["earnings"] == (AS_OF + timedelta(days=40)).isoformat()
    best = data["candidates"][0]
    assert best["front_expiry"] == (AS_OF + timedelta(days=best["front_dte"])).isoformat()
    assert best["result"]["ff"]["op"] == ">="
    assert best["structure"] in ("call", "put", "double_35d")
    assert isinstance(best["strikes"], list)


def _section(lines, header):
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
    return lines[start + 2 : end]  # skip header and dashed rule


def test_format_scan_compares_structures(ff_chains):
    scan_result = _scan(ff_chains)
    rows = _section(format_scan(scan_result).splitlines(), "Type ")

    assert [row.split()[0] for row in rows] == ["call", "put", "dbl"]
    for row, structure in zip(rows, ("call", "put", "double_35d")):
        trades = [c for c in scan_result.candidates if c.structure == structure]
        cells = row.replace("dbl 35d", "dbl35d").split()
        assert cells[1:4] == [
            str(len(trades)),
            str(sum(c.passed for c in trades)),
            f"{trades[0].result.ff.value:.3f}",
        ]


def test_format_scan_ranked_table(ff_chains):
    lines = format_scan(_scan(ff_chains)).splitlines()
    rows = _section(lines, "Rank ")
    header = next(line for line in lines if line.startswith("Rank "))

    assert "Spread" in header and "Close by" in header
    assert any("call ATM" in row for row in rows)
    assert any("dbl 35d" in row and "100.00/110.00" in row for row in rows)
    assert any(row.rstrip().endswith("FAIL spread") for row in rows)
    assert any(row.rstrip().endswith("PASS") for row in rows)
    assert any("2026-11-16" in row for row in rows)  # close by for the 28 DTE front


def test_format_scan_header_and_breakdowns(ff_chains):
    text = format_scan(_scan(ff_chains, earnings=AS_OF + timedelta(days=40)))
    lines = text.splitlines()
    assert lines[0] == "TEST"
    assert "trades pass: FF >= 0.2" in lines[1] and "10% of mid" in lines[1]
    assert "Next earnings 2026-11-29: trades whose back expiry is on/after it fail." in lines
    assert all(f"#{i} " in text for i in (1, 2, 3))
    assert "#4 " not in text
    assert "close by 2026-11-" in text
    assert text.count("forward variance") == 3
    assert text.count("max leg spread") == 3
    assert "earnings          FAIL" in text


def test_format_scan_without_earnings(ff_chains):
    text = format_scan(_scan(ff_chains))
    assert "Next earnings: none found." in text
    assert "none found" in text.split("#1 ")[1]


def test_format_scan_respects_top(ff_chains):
    text = format_scan(_scan(ff_chains), top=2)
    assert "#2 " in text and "#3 " not in text


def test_format_scan_shows_undefined_ff():
    chains = [make_chain(AS_OF + timedelta(days=28), 0.90), make_chain(AS_OF + timedelta(days=63), 0.30)]
    text = format_scan(_scan(chains, Params(strikes=1)))
    assert "FAIL ff" in text
    assert "undefined (negative fwd variance)" in text
