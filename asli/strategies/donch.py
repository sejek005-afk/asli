"""Gold hourly channel breakout with the daily trend (trend following on H1).

    trigger an hourly candle closes above the highest high of the previous 20 hourly candles (long) or
            below the lowest low (short)
    trend   only with the daily trend: previous day's close above its 50-day average for longs, below
            for shorts
    exits   stop 1.5 x ATR14(H1), target 3 x ATR14(H1), flat after 48 hours
    skip    no new trades on the candles that close 16:00-18:00 New York (rollover, wide spreads)

Why it is here — round 7b (Nov 2026), the user's "the market changed, judge on the last 2 years":
    by year (6-year M5 data, spread max(recorded, 0.55 bp), $0.05/oz commission, 1 pip slippage):
        2021 -34R, 2022 -10R, 2023 -22R, 2024 -2R, 2025 +68R, 2026 (to Nov) +24R
    chosen on Nov 2024-Nov 2025 out of 370 styles, then judged once on the unseen Nov 2025-Nov 2026:
        163 trades, win 42%, +0.19R per trade, both half-years positive; t = 1.7 (not conclusive)
    trading costs fell from ~17-21% of a 5-minute gold move (2021-23) to 7.5% (2026) and gold has trended
    since 2025; in 2021-24 the same rules lost every year. So it runs with `auto_switch` (see live.py):
    it only trades while its own last 12 months are profitable — that switch would have kept it off in
    2021-23 and on in 2025-26.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import atr
from ..timeutil import ny_hour_float
from .base import Context, Signal, Strategy


class HourlyChannelBreakout(Strategy):
    name = "donch"

    def applies_to(self, symbol: str) -> bool:
        prefixes = self.p.get("symbols", ["XAU"])
        return any(str(symbol).upper().startswith(str(p).upper()) for p in prefixes)

    def generate(self, ctx: Context) -> list[Signal]:
        if not self.applies_to(ctx.symbol):
            return []
        p = {**self.p, **(((ctx.cfg.get("strategies") or {}).get(self.name)) or {})}
        n_ch = int(p.get("channel_bars", 20))
        sl_k, tp_k = float(p.get("sl_atr", 1.5)), float(p.get("tp_atr", 3.0))
        hold = int(round(float(p.get("max_hold_hours", 48)) * 12))
        use_trend = bool(p.get("with_daily_trend", True))
        skip_a, skip_b = p.get("skip_ny_hours", [16, 18])
        df = ctx.df
        n = len(df)
        if n < 12 * (n_ch + 15):
            return []
        h1 = df[["open", "high", "low", "close"]].resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        h1_close_t = h1.index + pd.Timedelta(hours=1)
        hi = h1["high"].rolling(n_ch, min_periods=n_ch).max().shift(1).values  # previous n_ch hours only
        lo = h1["low"].rolling(n_ch, min_periods=n_ch).min().shift(1).values
        a1 = atr(h1, 14)
        c1 = h1["close"].values
        # the M5 bar whose close completes an hourly candle
        k = h1_close_t.get_indexer(df.index + pd.Timedelta(minutes=5))
        is_close = k >= 0
        hc = ny_hour_float(df.index) + 5.0 / 60.0  # New York clock at each M5 bar's close
        skip = (hc >= skip_a) & (hc < skip_b)
        trend = ctx.daily.trend50
        out: list[Signal] = []
        for i in np.flatnonzero(is_close & ~skip):
            j = k[i]
            if not (np.isfinite(hi[j]) and np.isfinite(lo[j]) and np.isfinite(a1[j]) and a1[j] > 0):
                continue
            d = 1 if c1[j] > hi[j] else (-1 if c1[j] < lo[j] else 0)
            if d == 0 or (use_trend and trend[i] != d):
                continue
            e = float(df["close"].values[i])
            a = float(a1[j])
            out.append(Signal(bar=int(i), time=df.index[i], symbol=ctx.symbol, strategy=self.name, direction=d,
                              order="market", entry=e, sl=e - d * sl_k * a, tp=e + d * tp_k * a, valid_bars=1,
                              max_hold_bars=hold, features={"h1_atr": round(a, 3), "channel": n_ch}))
        return out
