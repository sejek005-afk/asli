import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asli import backtest as bt  # noqa: E402
from asli import fusion  # noqa: E402
from asli.broker import PaperBroker, lot_size  # noqa: E402
from asli.config import load_config  # noqa: E402
from asli.data import SymbolMeta, synthetic  # noqa: E402
from asli.devil import DevilsAdvocate  # noqa: E402
from asli.fundamental import Fundamentals  # noqa: E402
from asli.learner import Learner  # noqa: E402
from asli.strategies import Context, Signal, build  # noqa: E402
from asli.timeutil import server_naive_to_utc, trading_day  # noqa: E402

CFG = load_config(Path(__file__).parent / "none.yaml")


def ctx(days=120, seed=3):
    return Context(synthetic(days=days, seed=seed), SymbolMeta.guess("EURUSD"), CFG)


def test_server_time_ny_plus_7():
    # 2026-01-15 is US winter: NY = UTC-5, server = UTC+2
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-15 10:00")])
    assert server_naive_to_utc(idx, "ny_plus_7")[0] == pd.Timestamp("2026-01-15 08:00", tz="UTC")
    # 2026-07-15 is US summer: NY = UTC-4, server = UTC+3
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-15 10:00")])
    assert server_naive_to_utc(idx, "ny_plus_7")[0] == pd.Timestamp("2026-07-15 07:00", tz="UTC")


def test_strategies_produce_valid_signals():
    c = ctx()
    for s in build(CFG):
        for sig in s.generate(c):
            assert sig.direction in (1, -1)
            assert (sig.entry - sig.sl) * sig.direction > 0, (s.name, sig)
            assert (sig.tp - sig.entry) * sig.direction > 0, (s.name, sig)
            assert 0 <= sig.bar < len(c.df)


def test_no_lookahead_signals_stable_when_future_removed():
    """A signal decided at bar i must not change when bars after i are removed."""
    full = synthetic(days=90, seed=5)
    cut = len(full) - 1500
    meta = SymbolMeta.guess("EURUSD")
    a = Context(full, meta, CFG)
    b = Context(full.iloc[:cut], meta, CFG)
    for strat in build(CFG):
        sa = {(x.bar, x.direction, round(x.entry, 6)) for x in strat.generate(a) if x.bar < cut - 300}
        sb = {(x.bar, x.direction, round(x.entry, 6)) for x in strat.generate(b) if x.bar < cut - 300}
        assert sa == sb, strat.name


def test_simulator_stop_first_and_costs():
    c = ctx(30)
    i = 500
    px = c.df["close"].values[i]
    s = Signal(i, c.df.index[i], "EURUSD", "test", 1, "market", px, px - 0.0005, px + 0.0010)
    t = bt.simulate(s, c)
    assert t is not None
    assert t.r <= 2.0 + 1e-9
    assert t.r >= -1.5


def test_learner_learns_simple_rule():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(600):
        ba = int(rng.choice([-1, 0, 1]))
        win = rng.random() < (0.65 if ba == 1 else 0.25)
        rows.append({"strategy": "ict", "session": "london", "ny_hour": 3, "bias_align": ba, "trend_align": 0,
                     "confluence": 0, "opposition": 0, "rr": 2, "sl_atr": 1, "key_level": True, "killzone": True,
                     "pd_align": 0, "r": 1.5 if win else -1.0})
    df = pd.DataFrame(rows)
    lr = Learner()
    lr.fit(df)
    good = lr.prob({**rows[0], "bias_align": 1})
    bad = lr.prob({**rows[0], "bias_align": -1})
    assert good > bad + 0.2
    lr.update({**rows[0], "bias_align": 1, "r": -1}, won=False)
    assert lr.n_live_updates == 1


def test_devil_vetoes_news_and_small_stop():
    d = DevilsAdvocate(CFG)
    base = {"direction": 1, "rr": 2.5, "sl_pips": 8, "spread_pips": 0.5, "prob": 0.6, "ev": 0.4, "bias_align": 1,
            "trend_align": 1, "confluence": 2, "opposition": 0, "killzone": True, "key_level": True, "pd_align": 0.5,
            "hour_edge": 0.1}
    assert d.review(base).approved
    v = d.review({**base, "news_blackout": {"title": "NFP", "cur": "USD", "impact": "High"}})
    assert not v.approved and any("NFP" in x for x in v.vetoes)
    assert not d.review({**base, "sl_pips": 2}).approved
    assert not d.review({**base, "bias_align": -1, "trend_align": -1, "opposition": 2, "killzone": False,
                         "hour_edge": -0.1, "confluence": 0, "pd_align": -0.5}).approved


def test_fundamental_calendar_blackout_and_headlines(tmp_path):
    f = Fundamentals(CFG, str(tmp_path))
    now = datetime(2026, 11, 6, 13, 0, tzinfo=timezone.utc)
    f.events = [{"t": now + timedelta(minutes=10), "cur": "USD", "impact": "High", "title": "Non-Farm Payrolls",
                 "forecast": None, "previous": None}]
    assert f.blackout("EURUSD", now) is not None
    assert f.blackout("EURJPY", now) is None
    f.headlines = [f.score_headline("ECB signals more rate hikes as euro inflation stays hot", now),
                   f.score_headline("Fed's Powell hints at rate cut as US jobs data weak", now)]
    assert f.bias("EURUSD", now) > 0


def test_lot_size():
    v = lot_size(10000, 1.0, 0.0010, 0.00001, 1.0, 0.01, 0.01, 100)
    assert abs(v - 1.0) < 1e-9


def test_paper_broker_roundtrip():
    pb = PaperBroker(10000)
    now = datetime.now(timezone.utc)
    pb.set_quote("EURUSD", 1.1000, 1.1001, now)
    t = pb.market("EURUSD", 1, 1.0, 1.0990, 1.1020, "x")
    pb.set_quote("EURUSD", 1.1021, 1.1022, now)
    assert not pb.positions() and pb.balance > 10000 and t == 1


def test_research_pipeline_runs():
    frames = {"EURUSD": synthetic(days=150, seed=1)}
    res = fusion.research(frames, {"EURUSD": SymbolMeta.guess("EURUSD")}, CFG)
    assert "per_strategy" in res and res["learner"].n_trained > 0


def test_utc_to_server_inverse():
    from asli.live import utc_to_server
    t = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
    s = utc_to_server(t, "ny_plus_7")
    assert s.replace(tzinfo=None) == datetime(2026, 1, 15, 10, 0)
    back = server_naive_to_utc(pd.DatetimeIndex([pd.Timestamp(s.replace(tzinfo=None))]), "ny_plus_7")[0]
    assert back == pd.Timestamp(t)


def test_limit_that_hits_entry_and_stop_in_one_bar_is_a_loss():
    c = ctx(30)
    i = 600
    df = c.df
    lo, hi = df["low"].values[i + 1], df["high"].values[i + 1]
    entry = hi - 0.1 * (hi - lo)  # long limit inside next bar's range
    sl = lo + 0.05 * (hi - lo)  # stop also inside it
    if sl >= entry - 1e-6:
        return
    s = Signal(i, df.index[i], "EURUSD", "t", 1, "limit", entry, sl, entry + 3 * (entry - sl), valid_bars=1)
    t = bt.simulate(s, c)
    assert t is not None and t.r < -0.9


def test_gold_gets_its_own_stop_range_and_costs():
    from asli.config import for_symbol
    g = for_symbol(CFG, "XAUUSD_o")
    assert g["risk"]["max_sl_pips"] == 100.0 and g["costs"]["commission_pips"] == 2.0
    assert for_symbol(CFG, "EURUSD") is CFG  # FX untouched
    cfg = {**CFG, "symbol_overrides": {"XAUUSD_o": {"risk": {"max_sl_pips": 80}}}}
    assert for_symbol(cfg, "XAUUSD_o")["risk"]["max_sl_pips"] == 80
    assert Context(synthetic(days=30, seed=5), SymbolMeta.guess("XAUUSD_o"), CFG).cfg["risk"]["max_sl_pips"] == 100.0


def test_devil_uses_symbol_stop_limits():
    d = DevilsAdvocate(CFG)
    base = {"direction": 1, "rr": 2.5, "sl_pips": 60, "spread_pips": 3, "prob": 0.6, "ev": 0.4, "bias_align": 1,
            "trend_align": 1, "confluence": 2, "opposition": 0, "killzone": True, "key_level": True, "pd_align": 0.5,
            "hour_edge": 0.1}
    assert not d.review({**base, "symbol": "EURUSD"}).approved  # 60 FX pips > 25
    assert d.review({**base, "symbol": "XAUUSD_o"}).approved  # $6 gold stop is normal
    assert not d.review({**base, "symbol": "XAUUSD_o", "sl_pips": 5}).approved  # $0.5 is too tight


def test_server_clock_not_guessed_on_weekend():
    from asli.data import market_closed
    assert market_closed(datetime(2026, 11, 7, 12, 0))  # Saturday
    assert market_closed(datetime(2026, 11, 6, 21, 0))  # Friday after close
    assert not market_closed(datetime(2026, 11, 9, 10, 0))  # Monday


def test_gold_scalp_only_on_gold_and_signals_are_valid():
    from asli.strategies.scalp import GoldScalp
    s = GoldScalp(CFG["strategies"]["scalp"])
    df = synthetic(days=60, seed=7)
    assert s.generate(Context(df, SymbolMeta.guess("EURUSD"), CFG)) == []
    gold = df.copy()
    gold[["open", "high", "low", "close", "spread"]] *= 3500.0
    sig = s.generate(Context(gold, SymbolMeta.guess("XAUUSD_o"), CFG))
    assert sig
    for x in sig:
        assert x.strategy == "scalp" and x.order == "market" and x.max_hold_bars == 48
        assert (x.tp - x.entry) * x.direction > 0 and (x.entry - x.sl) * x.direction > 0
        assert abs(x.rr - 0.75 / 4.0) < 1e-6


def test_devil_lets_scalp_use_its_own_limits():
    d = DevilsAdvocate(CFG)
    row = {"symbol": "XAUUSD_o", "strategy": "scalp", "direction": 1, "rr": 0.19, "sl_pips": 177, "spread_pips": 2.2,
           "prob": 0.88, "ev": 0.03, "bias_align": 1, "trend_align": 1, "confluence": 0, "opposition": 0,
           "killzone": True, "key_level": False, "pd_align": 0.0, "hour_edge": 0.05}
    assert d.review(row).approved
    v = d.review({**row, "strategy": "lit"})  # the same numbers fail the normal limits
    assert not v.approved and len(v.vetoes) >= 3


def test_learner_knows_the_scalp_style():
    from asli.learner import feature_names
    assert "s_scalp" in feature_names()


def test_style_can_set_its_own_debate_margin_and_skip_learner_vetoes():
    d = DevilsAdvocate(CFG)
    row = {"symbol": "XAUUSD_o", "strategy": "scalp", "direction": -1, "rr": 0.19, "sl_pips": 177,
           "spread_pips": 2.2, "prob": 0.30, "ev": -0.4, "bias_align": 1, "trend_align": 1, "confluence": 0,
           "opposition": 0, "killzone": True, "key_level": False, "pd_align": 0.0, "hour_edge": 0.0}
    assert d.review(row).approved  # scalp: learner prob/EV are advice only
    assert not d.review({**row, "news_blackout": {"title": "CPI", "cur": "USD", "impact": "High"}}).approved
    cfg = {**CFG, "strategies": {**CFG["strategies"], "scalp": {**CFG["strategies"]["scalp"], "approve_margin": 9}}}
    assert not DevilsAdvocate(cfg).review(row).approved


def test_server_time_eet_follows_eu_dst_dates():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-15 10:00"), pd.Timestamp("2026-07-15 10:00"),
                            pd.Timestamp("2026-03-16 10:00")])  # 16 Mar: US already on summer time, EU not yet
    u = server_naive_to_utc(idx, "eet")
    assert list(u) == [pd.Timestamp("2026-01-15 08:00", tz="UTC"), pd.Timestamp("2026-07-15 07:00", tz="UTC"),
                       pd.Timestamp("2026-03-16 08:00", tz="UTC")]
    assert server_naive_to_utc(idx[2:], "ny_plus_7")[0] == pd.Timestamp("2026-03-16 07:00", tz="UTC")  # 1 h off
    from asli.live import utc_to_server
    for t in u:
        s = utc_to_server(t.to_pydatetime(), "eet")
        assert server_naive_to_utc(pd.DatetimeIndex([pd.Timestamp(s.replace(tzinfo=None))]), "eet")[0] == t


def test_two_calendars_merge_and_go_stale(tmp_path):
    import time as _time
    f = Fundamentals(CFG, str(tmp_path))
    f._ff = Fundamentals._parse_calendar([{"date": "2026-11-13T08:30:00-05:00", "country": "USD", "impact": "High",
                                           "title": "CPI m/m"}])
    f._tv = Fundamentals._parse_tradingview({"status": "ok", "result": [
        {"date": "2026-11-12T19:00:00.000Z", "currency": "USD", "importance": 1, "title": "Fed Chair Speech"},
        {"date": "2026-11-13T13:30:00.000Z", "currency": "USD", "importance": 1, "title": "CPI m/m"},
        {"date": "2026-11-13T10:00:00.000Z", "currency": "EUR", "importance": -1, "title": "ZEW"}]})
    f._merge()
    assert len(f.events) == 3  # the CPI in both feeds is kept once
    t = datetime(2026, 11, 12, 19, 20, tzinfo=timezone.utc)
    assert f.blackout("XAUUSD_o", t)["title"] == "Fed Chair Speech"  # only TradingView had it
    assert f.blackout("XAUUSD_o", datetime(2026, 11, 13, 13, 45, tzinfo=timezone.utc)) is not None
    f.ok_at = {"tradingview": _time.time()}
    assert f.calendar_ok
    f.ok_at = {"forexfactory": _time.time() - 5 * 3600}
    assert not f.calendar_ok  # nothing answered for 5 h: news risk unknown, not "all clear"


def test_rss_dates_from_investing_feeds():
    from asli.fundamental import _parse_pubdate
    assert _parse_pubdate("2026-11-06 22:48:20") == datetime(2026, 11, 6, 22, 48, 20, tzinfo=timezone.utc)
    assert _parse_pubdate("Nov 06, 2026 20:56 GMT") == datetime(2026, 11, 6, 20, 56, tzinfo=timezone.utc)
    assert _parse_pubdate("Fri, 06 Nov 2026 21:52:39 GMT") == datetime(2026, 11, 6, 21, 52, 39, tzinfo=timezone.utc)


def test_engine_reconnects_after_terminal_restart(monkeypatch):
    import asli.live as live
    calls = []

    class FakeMT5:
        def terminal_info(self):
            return None

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(live, "mt5_connect", lambda cfg: calls.append("connect"))
    eng = live.Engine.__new__(live.Engine)
    eng.mt5, eng.cfg = FakeMT5(), CFG
    eng._ensure_mt5()
    assert calls == ["shutdown", "connect"]
    eng._ensure_mt5()  # throttled: no second attempt within 30 s
    assert calls == ["shutdown", "connect"]


def _orb_day_frame(break_up=True, days=60, jump_after=9.0):
    """Synthetic gold M5 bars: a quiet range 08:30-09:00 NY, then a clean break (prices rise over the days
    so the 50-day trend is up)."""
    idx = pd.date_range("2026-01-05 00:00", periods=days * 288, freq="5min", tz="UTC")
    ny = idx.tz_convert("America/New_York")
    hc = ny.hour + (ny.minute + 5) / 60.0
    base = 2000 + np.arange(len(idx)) * 0.002
    px = base.copy()
    after = (hc > jump_after) & (hc <= 11.0)
    px = np.where(after, px + (6.0 if break_up else -6.0), px)
    df = pd.DataFrame({"open": px, "high": px + 1.0, "low": px - 1.0, "close": px, "spread": 0.2}, index=idx)
    return df


def test_ny_opening_range_style():
    from asli.strategies.nyorb import NYOpeningRange
    cfg = {**CFG, "strategies": {**CFG["strategies"], "nyorb": {**CFG["strategies"]["nyorb"], "enabled": True}}}
    s = NYOpeningRange(cfg["strategies"]["nyorb"])
    df = _orb_day_frame(break_up=True)
    assert s.generate(Context(df, SymbolMeta.guess("EURUSD"), cfg)) == []  # gold only
    sig = s.generate(Context(df, SymbolMeta.guess("XAUUSD_o"), cfg))
    assert sig and all(x.direction == 1 for x in sig)  # up-breaks in an uptrend
    days = [str(trading_day(pd.DatetimeIndex([x.time]))[0]) for x in sig]
    assert len(days) == len(set(days))  # one trade per day at most
    for x in sig:
        ny_h = x.time.tz_convert("America/New_York")
        assert 9 <= ny_h.hour + ny_h.minute / 60 + 5 / 60 <= 11.5 + 1e-9
        assert x.sl < x.entry < x.tp and x.max_hold_bars >= 1
    down = s.generate(Context(_orb_day_frame(break_up=False), SymbolMeta.guess("XAUUSD_o"), cfg))
    assert down == []  # down-breaks against the 50-day uptrend are skipped


def test_nyorb_agent_is_advisory_but_safety_vetoes_hold():
    d = DevilsAdvocate(CFG)
    row = {"symbol": "XAUUSD_o", "strategy": "nyorb", "direction": 1, "rr": 1.6, "sl_pips": 150, "spread_pips": 2.2,
           "prob": 0.2, "ev": -0.3, "bias_align": -1, "trend_align": -1, "confluence": 0, "opposition": 1,
           "killzone": False, "key_level": False, "pd_align": -0.5, "hour_edge": -0.1}
    assert d.review(row).approved
    assert not d.review({**row, "news_blackout": {"title": "CPI", "cur": "USD", "impact": "High"}}).approved
    assert not d.review({**row, "sl_pips": 600}).approved


def test_nyorb_reads_per_symbol_settings():
    from asli.strategies.nyorb import NYOpeningRange
    base = {**CFG["strategies"]["nyorb"], "enabled": True, "symbols": ["XAU", "SPX"]}
    cfg = {**CFG, "strategies": {**CFG["strategies"], "nyorb": base},
           "symbol_overrides": {"SPX": {"strategies": {"nyorb": {"range_start_ny": 9.5}}}}}
    s = NYOpeningRange(base)
    df = _orb_day_frame(break_up=True)
    gold = s.generate(Context(df, SymbolMeta.guess("XAUUSD_o"), cfg))
    spx = s.generate(Context(_orb_day_frame(break_up=True, jump_after=10.0), SymbolMeta("SPX", 1, 0.1, 0.1), cfg))
    assert gold and spx
    ny = lambda x: x.time.tz_convert("America/New_York")  # noqa: E731
    assert all(ny(x).hour + ny(x).minute / 60 + 5 / 60 > 9.0 for x in gold)
    assert all(ny(x).hour + ny(x).minute / 60 + 5 / 60 > 10.0 for x in spx)  # its range is 09:30-10:00
