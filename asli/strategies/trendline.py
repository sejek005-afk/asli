"""Trendline liquidity on objective DeMark TD lines (M15).

A TD demand line joins the two most recent TD lows (level 1) when the newer low is higher. Instead of
buying the touch or selling the break, we trade the ICT view: price sweeps the stops under a line that
has been touched at least `min_touches` times, then closes back above it -> long (mirror for supply lines).
Only with the trend (EMA50 vs EMA200 on M15, and the daily SMA200 trend not against)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import resample
from ..features import atr, confirmed_levels, ema
from .base import Context, Signal, Strategy, nearest_target

_M15 = pd.Timedelta(minutes=15)


class TrendlineLiquidity(Strategy):
    name = "trendline"

    def generate(self, ctx: Context) -> list[Signal]:
        m = resample(ctx.df, "15min")
        if len(m) < 250:
            return []
        h, l, c = m["high"].values, m["low"].values, m["close"].values
        a = atr(m, 14)
        e50, e200 = ema(c, 50), ema(c, 200)
        sh, _, slv, _ = confirmed_levels(h, l, 3)
        min_touch = int(self.p.get("min_touches", 3))
        min_rr = float(ctx.cfg["risk"]["min_rr"])
        n = len(m)
        lows, highs = [], []  # confirmed TD points (index)
        used = set()
        out: list[Signal] = []
        for i in range(2, n):
            j = i - 1  # TD point at j confirmed when bar i closes
            if l[j] < l[j - 1] and l[j] < l[i]:
                lows.append(j)
            if h[j] > h[j - 1] and h[j] > h[i]:
                highs.append(j)
            if not np.isfinite(a[i]) or i < 200:
                continue
            for dirn, pts, arr in ((+1, lows, l), (-1, highs, h)):
                if len(pts) < 2:
                    continue
                p1, p2 = pts[-2], pts[-1]
                if p2 >= i or (dirn, p1, p2) in used:
                    continue
                if dirn == +1 and not arr[p2] > arr[p1]:
                    continue
                if dirn == -1 and not arr[p2] < arr[p1]:
                    continue
                slope = (arr[p2] - arr[p1]) / (p2 - p1)
                line = arr[p1] + slope * (i - p1)
                # touches: bars between p1 and i-1 within 0.15 ATR of the line (p1 and p2 count)
                xs = np.arange(p1, i)
                lv = arr[p1] + slope * (xs - p1)
                touches = int(np.sum(np.abs(arr[xs] - lv) <= 0.15 * a[i]))
                if touches < min_touch:
                    continue
                trend_ok = (e50[i] > e200[i]) if dirn == +1 else (e50[i] < e200[i])
                if not trend_ok:
                    continue
                if dirn == +1:
                    swept = l[i] < line - 0.05 * a[i] and c[i] > line
                else:
                    swept = h[i] > line + 0.05 * a[i] and c[i] < line
                if not swept:
                    continue
                bar = ctx.m5_bar_at_close(m.index[i] + _M15)
                if bar < 0 or bar >= len(ctx.df) - 1:
                    continue
                if ctx.daily.trend[bar] == -dirn:
                    continue
                used.add((dirn, p1, p2))
                sp = ctx.spread[bar]
                entry = ctx.df["close"].values[bar] + (sp if dirn == +1 else 0.0)
                sl = (l[i] - 0.15 * a[i] - sp) if dirn == +1 else (h[i] + 0.15 * a[i] + sp)
                risk = abs(entry - sl)
                if risk <= 0:
                    continue
                target_levels = [sh[i]] if dirn == +1 else [slv[i]]
                tp = nearest_target(target_levels, entry, risk, dirn, min_rr) or entry + dirn * 2.0 * risk
                out.append(Signal(bar=bar, time=ctx.df.index[bar], symbol=ctx.symbol, strategy=self.name,
                                  direction=dirn, order="market", entry=entry, sl=sl, tp=tp, max_hold_bars=288,
                                  features={"touches": touches, "sweep_depth_atr": float(abs(
                                      (l[i] if dirn == +1 else h[i]) - line) / a[i]), "key_level": True}))
        return out
