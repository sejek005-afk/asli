"""Bar-by-bar trade simulator and statistics.

Conventions (conservative):
- Bars are BID. Longs enter at ASK (bid + spread) and exit at BID; shorts enter at BID and exit at ASK.
- If stop and target are both inside one bar, the STOP is assumed to hit first.
- Commission and stop slippage are charged in price units from config `costs`.
- Results are in R (multiples of initial risk), so they don't depend on account size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies.base import Context, Signal
from .timeutil import NY, session_of_hour


@dataclass
class Trade:
    signal: Signal
    fill_bar: int
    exit_bar: int
    fill_price: float
    exit_price: float
    r: float
    outcome: str  # tp | sl | be | time | tp1+tp | tp1+be

    def row(self, index) -> dict:
        s = self.signal
        ny = index[s.bar].tz_convert(NY)
        hour = ny.hour + ny.minute / 60.0
        d = {
            "time": index[s.bar], "fill_time": index[self.fill_bar], "exit_time": index[self.exit_bar],
            "symbol": s.symbol, "strategy": s.strategy, "direction": s.direction, "order": s.order,
            "entry": s.entry, "sl": s.sl, "tp": s.tp, "tp1": s.tp1, "rr": s.rr, "r": self.r,
            "outcome": self.outcome, "ny_hour": int(ny.hour), "session": session_of_hour(hour),
            "weekday": int(ny.dayofweek), "bars_held": self.exit_bar - self.fill_bar,
        }
        for k, v in s.features.items():
            d[f"f_{k}"] = v
        return d


def simulate(sig: Signal, ctx: Context) -> Trade | None:
    df = ctx.df
    h, l, o = df["high"].values, df["low"].values, df["open"].values
    sp = ctx.spread
    n = len(df)
    pip = ctx.meta.pip
    comm = ctx.cfg["costs"]["commission_pips"] * pip
    slip = ctx.cfg["costs"]["slippage_pips"] * pip
    d = sig.direction
    start = sig.bar + 1
    if start >= n:
        return None
    # ---------------------------------------------------------------- fill
    if sig.order == "market":
        fill_bar = start
        fill = o[start] + (sp[start] if d == 1 else 0.0) + slip * 1.0
        fill = fill if d == 1 else o[start] - slip
    else:
        fill_bar = -1
        for b in range(start, min(start + sig.valid_bars, n)):
            # fill first: a bar that trades through entry AND stop fills, then stops out (-1R) below
            if d == 1 and l[b] + sp[b] <= sig.entry:
                fill_bar, fill = b, min(sig.entry, o[b] + sp[b])
                break
            if d == -1 and h[b] >= sig.entry:
                fill_bar, fill = b, max(sig.entry, o[b])
                break
            # stop traded without ever reaching the entry (only possible with gaps): order is cancelled
            if d == 1 and l[b] <= sig.sl:
                return None
            if d == -1 and h[b] + sp[b] >= sig.sl:
                return None
        if fill_bar < 0:
            return None
    risk = abs(fill - sig.sl)
    if risk <= 0 or (sig.sl - fill) * d >= 0:
        return None  # gapped through the stop before entry
    # ---------------------------------------------------------------- manage
    stop = sig.sl
    tp1_done = False
    realized = 0.0  # in price, weighted by remaining size
    size = 1.0
    end = min(fill_bar + sig.max_hold_bars, n - 1)
    outcome = "time"
    exit_price = None
    exit_bar = end
    for b in range(fill_bar, end + 1):
        bid_h, bid_l = h[b], l[b]
        ask_h, ask_l = h[b] + sp[b], l[b] + sp[b]
        if d == 1:
            hit_sl = bid_l <= stop
            hit_tp1 = sig.tp1 is not None and not tp1_done and bid_h >= sig.tp1
            hit_tp = bid_h >= sig.tp
        else:
            hit_sl = ask_h >= stop
            hit_tp1 = sig.tp1 is not None and not tp1_done and ask_l <= sig.tp1
            hit_tp = ask_l <= sig.tp
        if b == fill_bar and sig.order == "limit":
            # on the fill bar only count the stop (we can't know intrabar order)
            hit_tp = hit_tp1 = False
        if hit_sl:
            px = stop - d * slip
            exit_price, exit_bar = px, b
            outcome = ("tp1+be" if tp1_done else ("be" if stop == fill else "sl"))
            break
        if hit_tp1:
            realized += 0.5 * (sig.tp1 - fill) * d
            size = 0.5
            tp1_done = True
            stop = fill  # break-even on the rest
        if hit_tp:
            exit_price, exit_bar = sig.tp, b
            outcome = "tp1+tp" if tp1_done else "tp"
            break
    if exit_price is None:
        exit_price = df["close"].values[end] + (sp[end] if d == -1 else 0.0)
        exit_bar = end
    pnl = realized + size * (exit_price - fill) * d - comm
    return Trade(sig, fill_bar, exit_bar, float(fill), float(exit_price), float(pnl / risk), outcome)


def run(signals: list[Signal], ctx: Context, one_at_a_time: bool = True) -> pd.DataFrame:
    """Simulate a list of signals for one symbol. With one_at_a_time, a strategy can't stack trades."""
    rows = []
    busy_until: dict[str, int] = {}
    for s in sorted(signals, key=lambda x: x.bar):
        key = s.strategy if one_at_a_time else f"{s.strategy}:{s.bar}"
        if busy_until.get(key, -1) >= s.bar:
            continue
        t = simulate(s, ctx)
        if t is None:
            continue
        busy_until[key] = t.exit_bar
        rows.append(t.row(ctx.df.index))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics

def stats(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "profit_factor": 0.0,
                "max_dd_r": 0.0, "max_losing_streak": 0, "sharpe_like": 0.0}
    r = trades["r"].values
    wins = r[r > 0]
    losses = r[r <= 0]
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "total_r": float(r.sum()),
        "profit_factor": float(pf),
        "max_dd_r": float(dd.min()) if len(dd) else 0.0,
        "max_losing_streak": int(best),
        "sharpe_like": float(r.mean() / (r.std() + 1e-9) * np.sqrt(min(len(r), 250))),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
    }


def shrunk_edge(trades: pd.DataFrame, by: list[str], prior_n: float = 20.0) -> pd.DataFrame:
    """Average R per group shrunk toward 0 (Bayesian-style), so 5 lucky trades don't look like an edge."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=by + ["n", "win_rate", "avg_r", "edge"])
    g = trades.groupby(by)["r"]
    out = pd.DataFrame({"n": g.size(), "win_rate": g.apply(lambda x: (x > 0).mean()), "avg_r": g.mean(),
                        "sum_r": g.sum()})
    out["edge"] = out["sum_r"] / (out["n"] + prior_n)
    return out.reset_index().sort_values("edge", ascending=False)
