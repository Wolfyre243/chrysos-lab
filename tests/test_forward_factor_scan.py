import json
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from analytics.chain import strikes_around
from calculators import forward_factor
from cli import forward_factor as cli
from cli.forward_factor import format_bulk, format_scan, main, scan_bulk
from data.market_calendar import previous_trading_day
from data.models import OptionChain
from errors import CalculatorError
from scanners.forward_factor import Inputs, Params, pick_legs, run, scan

from conftest import AS_OF, SPOT, FakeProvider, make_chain

# make_chain liquidity: strike 110 trades only 50 contracts, so with the default 100-contract
# minimum every trade using it fails on volume. The 35-delta double
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
    return bool(set(c.flags) - {"ff"})


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
        return pd.DataFrame(
            {
                "strike": strikes,
                "iv": [iv] * n,
                "bid": [1.0] * n,
                "ask": [1.1] * n,
                "volume": [500.0] * n,
                "open_interest": [5_000.0] * n,
            }
        )

    return OptionChain(AS_OF + timedelta(days=dte), side(call_strikes), side(put_strikes))


LADDER = [80.0, 90.0, 100.0, 110.0, 120.0]


@pytest.mark.parametrize(
    "target, width, expected",
    [
        (100.0, 1, [90.0, 100.0, 110.0]),  # center +/- 1
        (100.0, 2, LADDER),
        (100.0, 0, [100.0]),
        (82.0, 1, [80.0, 90.0]),  # clamped at the bottom
        (500.0, 1, [110.0, 120.0]),  # clamped at the top; far target snaps to the last strike
        (108.0, 0, [110.0]),  # unlisted target snaps to the nearest
        (105.0, 0, [100.0]),  # tie goes to the lower strike
    ],
)
def test_strikes_around(target, width, expected):
    chains = [_chain(30, LADDER, LADDER), _chain(60, LADDER, LADDER)]
    assert strikes_around(chains, target, width) == expected


def test_strikes_around_needs_every_given_side_of_every_chain():
    front = _chain(30, [90.0, 100.0, 110.0], [90.0, 100.0])
    back = _chain(60, [100.0, 110.0, 120.0], [90.0, 100.0, 110.0, 120.0])
    assert strikes_around([front, back], 104.0, 1) == [100.0]
    assert strikes_around([front, back], 104.0, 1, sides=("call",)) == [100.0, 110.0]
    assert strikes_around([front, back], 104.0, 1, sides=("put",)) == [90.0, 100.0]
    assert strikes_around([_chain(30, [], [])], 104.0, 1) == []


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
    assert (result.min_volume, result.min_open_interest) == (100, 1_000)
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
    result = _scan(ff_chains, Params(strike_range=0))
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
    assert "volume" in double.flags  # the 110 call legs are thin


def test_scan_marks_one_atm_call_per_pair(ff_chains):
    atm = [c for c in _scan(ff_chains).candidates if c.atm]
    assert sorted((c.front_dte, c.structure, c.strikes) for c in atm) == [
        (28, "call", (100.0,)),
        (35, "call", (100.0,)),
    ]


def test_scan_given_strike_scans_only_that_strike(ff_chains):
    result = _scan(ff_chains, Params(strike=110.0, strike_range=3))  # range is ignored
    assert _trades(result) == {(s, f, 63, (110.0,)) for s in ("call", "put") for f in (28, 35)}
    assert result.strike == 110.0
    assert result.strike_range == 0


def test_scan_given_strike_keeps_only_pairs_that_list_it():
    ladder = [90.0, 100.0, 110.0]
    chains = [
        _chain(28, ladder + [105.0], ladder + [105.0]),
        _chain(35, ladder, ladder),  # no 105
        _chain(63, ladder + [105.0], ladder + [105.0]),
    ]
    result = _scan(chains, Params(strike=105.0))
    assert _trades(result) == {("call", 28, 63, (105.0,)), ("put", 28, 63, (105.0,))}


@pytest.mark.parametrize(
    "strike, nearest",
    [
        (108.0, "100.00, 110.00"),
        (200.0, "110.00"),
        (50.0, "90.00"),
    ],
)
def test_scan_given_unlisted_strike_raises_with_nearest(ff_chains, strike, nearest):
    with pytest.raises(CalculatorError, match=f"lists the {strike:,.2f} strike.*Nearest strikes that do: {nearest}\\."):
        _scan(ff_chains, Params(strike=strike))


def test_scan_marks_atm_only_when_it_is_scanned(ff_chains):
    assert not any(c.atm for c in _scan(ff_chains, Params(strike=90.0)).candidates)
    atm = [c for c in _scan(ff_chains, Params(strike=100.0)).candidates if c.atm]
    assert {(c.structure, c.strikes) for c in atm} == {("call", (100.0,))}


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


def test_scan_volume_check_uses_thinnest_leg(ff_chains):
    thin = [c for c in _scan(ff_chains).candidates if "volume" in c.flags]
    assert {c.strikes for c in thin} == {(110.0,), DOUBLE_STRIKES}
    assert all(c.volume.value == 50 and c.volume.threshold == 100 for c in thin)

    assert not any("volume" in c.flags for c in _scan(ff_chains, Params(min_volume=50)).candidates)


def test_scan_missing_volume_counts_as_zero(ff_chains):
    ff_chains[1] = make_chain(AS_OF + timedelta(days=28), 0.40, volume=float("nan"))
    atm = [c for c in _scan(ff_chains).candidates if c.front_dte == 28 and c.strikes == (100.0,)]
    assert atm and all(c.volume.value == 0 and "volume" in c.flags for c in atm)


def test_scan_open_interest_check_uses_thinnest_leg(ff_chains):
    assert not any("oi" in c.flags for c in _scan(ff_chains).candidates)  # every leg has >= 3,000

    strict = _scan(ff_chains, Params(min_open_interest=5_000)).candidates
    assert {c.strikes for c in strict if "oi" in c.flags} == {(90.0,), (110.0,), DOUBLE_STRIKES}
    assert {c.open_interest.value for c in strict if c.strikes == (90.0,)} == {4_000}


def test_scan_puts_undefined_ff_last():
    chains = [
        make_chain(AS_OF + timedelta(days=28), 0.90),  # forward variance negative vs the back
        make_chain(AS_OF + timedelta(days=35), 0.35),
        make_chain(AS_OF + timedelta(days=63), 0.30),
    ]
    result = _scan(chains, Params(strike_range=0, min_volume=0))
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
        _scan(chains, Params(strike_range=0))


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
    ff_chains[2] = make_chain(AS_OF + timedelta(days=35), 0.35, volume=float("nan"))
    data = _scan(ff_chains, earnings=AS_OF + timedelta(days=40)).to_dict()
    json.dumps(data, allow_nan=False)
    assert data["front_window"] == [21, 45]
    assert data["earnings"] == (AS_OF + timedelta(days=40)).isoformat()
    best = data["candidates"][0]
    assert best["front_expiry"] == (AS_OF + timedelta(days=best["front_dte"])).isoformat()
    assert best["result"]["ff"]["op"] == ">="
    assert best["structure"] in ("call", "put", "double_35d")
    assert isinstance(best["strikes"], list)
    assert data["strike"] is None and data["strike_range"] == 1


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

    assert header.split()[10:12] == ["Vol", "OI"] and "Close by" in header
    assert any("call ATM" in row for row in rows)
    assert any("dbl 35d" in row and "100.00/110.00" in row for row in rows)
    assert any(row.rstrip().endswith("FAIL volume") for row in rows)
    assert any(row.rstrip().endswith("PASS") for row in rows)
    assert any("2026-11-16" in row for row in rows)  # close by for the 28 DTE front


def test_format_scan_header_and_breakdowns(ff_chains):
    text = format_scan(_scan(ff_chains, earnings=AS_OF + timedelta(days=40)))
    lines = text.splitlines()
    assert lines[0] == "TEST"
    assert "trades pass: FF >= 0.2" in lines[1]
    assert "every leg volume >= 100 and open interest >= 1,000" in lines[1]
    assert "Next earnings 2026-11-29: trades whose back expiry is on/after it fail." in lines
    assert all(f"#{i} " in text for i in (1, 2, 3))
    assert "#4 " not in text
    assert "close by 2026-11-" in text
    assert text.count("forward variance") == 3
    assert text.count("min leg volume") == 3
    assert text.count("min leg open int") == 3
    assert "earnings          FAIL" in text


def test_format_scan_without_earnings(ff_chains):
    text = format_scan(_scan(ff_chains))
    assert "Next earnings: none found." in text
    assert "none found" in text.split("#1 ")[1]


def test_format_scan_strike_line(ff_chains):
    assert "Strikes: nearest spot +/- 1 step" in format_scan(_scan(ff_chains)).splitlines()
    assert "Strikes: nearest spot +/- 2 steps" in format_scan(_scan(ff_chains, Params(strike_range=2))).splitlines()

    lines = format_scan(_scan(ff_chains, Params(strike=110.0))).splitlines()
    assert "Strike: 110.00 only (pairs listing it on both legs)  |  double calendar skipped" in lines
    assert not any("dbl 35d  " in line and "@" in line for line in lines)


def test_cli_rejects_strike_with_range():
    with pytest.raises(SystemExit):
        main(["AAPL", "--strike", "66", "--range", "2"])


def test_format_scan_respects_top(ff_chains):
    text = format_scan(_scan(ff_chains), top=2)
    assert "#2 " in text and "#3 " not in text


def test_format_scan_shows_undefined_ff():
    chains = [make_chain(AS_OF + timedelta(days=28), 0.90), make_chain(AS_OF + timedelta(days=63), 0.30)]
    text = format_scan(_scan(chains, Params(strike_range=0)))
    assert "FAIL ff" in text
    assert "undefined (negative fwd variance)" in text


# --- bulk ------------------------------------------------------------------


def _symbol_scan(symbol, chains, params=Params(), earnings=None):
    return scan(Inputs(symbol=symbol, as_of=AS_OF, spot=SPOT, chains=chains, earnings=earnings), params)


def _bulk(outcomes, params=Params()):
    """scan_bulk over symbol -> ScanResult or exception to raise."""

    def compute(symbol):
        outcome = outcomes[symbol]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return scan_bulk(list(outcomes), compute, params, AS_OF, "ff.txt")


@pytest.fixture
def bulk_outcomes(ff_chains):
    """FEW passes some trades, MANY more (the thin 110 strike clears a 0 volume floor), NONE is
    blocked by earnings everywhere and BAD fails to fetch."""
    error = CalculatorError("No options found for BAD.")
    error.__cause__ = ConnectionError("Too Many Requests")
    return {
        "FEW": _symbol_scan("FEW", ff_chains),
        "NONE": _symbol_scan("NONE", ff_chains, earnings=AS_OF + timedelta(days=40)),
        "BAD": error,
        "MANY": _symbol_scan("MANY", ff_chains, Params(min_volume=0)),
    }


def test_scan_bulk_ranks_by_passing_trades(bulk_outcomes):
    result = _bulk(bulk_outcomes)

    assert [t.symbol for t in result.ranked] == ["MANY", "FEW"]
    assert result.ranked[0].passed > result.ranked[1].passed > 0
    assert result.empty == ["NONE"]
    assert [f.symbol for f in result.failed] == ["BAD"]
    assert result.failed[0].error.splitlines() == [
        "Error: No options found for BAD.",
        "  Caused by ConnectionError: Too Many Requests",
    ]
    assert result.total == 4


def test_scan_bulk_best_is_highest_ff_passing_trade(bulk_outcomes):
    ticker = _bulk(bulk_outcomes).ranked[1]
    scan_result = bulk_outcomes["FEW"]
    passing = [c for c in scan_result.candidates if c.passed]

    assert ticker.best.passed
    assert ticker.best.result.ff.value == max(c.result.ff.value for c in passing)
    assert (ticker.trades, ticker.passed) == (len(scan_result.candidates), len(passing))


def test_scan_bulk_breaks_ties_by_best_ff(ff_chains):
    lo = _symbol_scan("LO", ff_chains)
    best = lo.candidates[0]
    assert best.passed
    boosted = replace(best, result=replace(best.result, ff=replace(best.result.ff, value=best.result.ff.value + 1)))
    hi = replace(lo, symbol="HI", candidates=[boosted, *lo.candidates[1:]])

    result = _bulk({"LO": lo, "HI": hi})
    assert result.ranked[0].passed == result.ranked[1].passed
    assert [t.symbol for t in result.ranked] == ["HI", "LO"]


def test_scan_bulk_prints_progress_to_stderr(bulk_outcomes, capsys):
    _bulk(bulk_outcomes)
    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines()[0] == "Scanning FEW (1/4)..."
    assert err.splitlines()[-1] == "Scanning MANY (4/4)..."


def test_format_bulk(bulk_outcomes):
    text = format_bulk(_bulk(bulk_outcomes))
    lines = text.splitlines()
    rows = _section(lines, "Rank ")
    header = next(line for line in lines if line.startswith("Rank "))

    assert lines[0] == "Bulk scan: 4 tickers from ff.txt  |  As of 2026-10-20"
    assert lines[1].startswith("2 with passing trades, 1 with none, 1 failed. Pass: FF >= 0.2")
    assert header.split()[:6] == ["Rank", "Symbol", "Spot", "Pass", "Trades", "Type"]
    assert "Close by" in header and "Result" not in header
    assert [row.split()[1] for row in rows] == ["MANY", "FEW"]
    assert not any("NONE" in line for line in lines)
    assert any(line.startswith("#1  MANY  ") for line in lines)
    assert any(line.startswith("#2  FEW  ") for line in lines)
    assert text.count("forward variance") == 2
    assert lines[-3:] == [
        "Failed (1):",
        "BAD  Error: No options found for BAD.",
        "       Caused by ConnectionError: Too Many Requests",
    ]


def test_format_bulk_without_passing_tickers(bulk_outcomes):
    text = format_bulk(_bulk({"NONE": bulk_outcomes["NONE"]}))
    assert "No ticker has a passing trade." in text.splitlines()
    assert "Rank " not in text and "Failed" not in text


def test_bulk_to_dict_is_strict_json(bulk_outcomes):
    data = _bulk(bulk_outcomes).to_dict()
    json.dumps(data, allow_nan=False)
    assert data["as_of"] == AS_OF.isoformat()
    assert [t["symbol"] for t in data["ranked"]] == ["MANY", "FEW"]
    assert data["ranked"][0]["best"]["passed"] is True
    assert data["empty"] == ["NONE"]
    assert data["failed"][0]["symbol"] == "BAD"


def test_cli_bulk_scans_file(ff_chains, tmp_path, monkeypatch, capsys):
    path = tmp_path / "ff.txt"
    path.write_text("aapl\n\n# watchlist\nmsft\n", encoding="utf-8")
    monkeypatch.setattr(cli, "YFinanceProvider", lambda: FakeProvider(ff_chains))
    monkeypatch.setattr(cli, "market_date", lambda: AS_OF)

    assert main(["--bulk", str(path)]) == 0
    out, err = capsys.readouterr()
    assert "Scanning MSFT (2/2)..." in err
    assert [row.split()[1] for row in _section(out.splitlines(), "Rank ")] == ["AAPL", "MSFT"]

    assert main(["--bulk", str(path), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["ranked"]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--bulk", "missing.txt"],
        ["AAPL", "--bulk", "{empty}"],
        ["--bulk", "{empty}"],
    ],
)
def test_cli_bulk_rejects_bad_input(argv, tmp_path, monkeypatch):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n# nothing here\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main([arg.format(empty=empty) for arg in argv])
