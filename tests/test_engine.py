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
from asli.timeutil import server_naive_to_utc  # noqa: E402

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
