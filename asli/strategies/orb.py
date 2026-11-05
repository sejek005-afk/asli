"""ORB (Opening Range Breakout) at the London open, with the filters the evidence points to:
trade only with the daily SMA200 trend, only when the opening range is narrower than its 20-day median,
one trade per day, stop at the other side of the range (capped), 2R target, flat by 16:00 London."""
from __future__ import annotations

import numpy as np

from ..timeutil import LONDON
from .base import Context, Signal, Strategy


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def generate(self, ctx: Context) -> list[Signal]:
        df = ctx.df
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        ldn = df.index.tz_convert(LONDON)
        mins = ldn.hour.values * 60 + ldn.minute.values
        date = ldn.date
        rmin = int(self.p.get("range_minutes", 30))
        until = int(self.p.get("trade_until_minutes", 150))
        max_pct = float(self.p.get("max_range_pct_of_median", 1.0))
        max_sl = ctx.cfg["risk"]["max_sl_pips"] * ctx.meta.pip
        open_min = 8 * 60
        out: list[Signal] = []
        widths: list[float] = []
        n = len(df)
        i = 0
        while i < n:
            day = date[i]
            j = i
            while j < n and date[j] == day:
                j += 1
            in_rng = (mins[i:j] >= open_min) & (mins[i:j] < open_min + rmin)
            if in_rng.sum() >= max(rmin // 5 - 1, 1):
                rh = float(np.max(h[i:j][in_rng]))
                rl = float(np.min(l[i:j][in_rng]))
                width = rh - rl
                med = float(np.median(widths[-20:])) if len(widths) >= 5 else np.inf
                widths.append(width)
                if width <= max_pct * med:
                    for b in range(i, j - 1):
                        t = mins[b]
                        if t < open_min + rmin or t >= open_min + until:
                            continue
                        dirn = +1 if c[b] > rh else (-1 if c[b] < rl else 0)
                        if dirn == 0 or ctx.daily.trend[b] != dirn:
                            continue
                        sp = ctx.spread[b]
                        entry = c[b] + (sp if dirn == +1 else 0.0)
                        sl = rl - sp if dirn == +1 else rh + sp
                        if abs(entry - sl) > max_sl:
                            sl = entry - dirn * max_sl
                        risk = abs(entry - sl)
                        if risk <= 0:
                            break
                        hold = max(1, ((16 * 60) - t) // 5)
                        out.append(Signal(bar=b, time=df.index[b], symbol=ctx.symbol, strategy=self.name,
                                          direction=dirn, order="market", entry=entry, sl=sl,
                                          tp=entry + dirn * 2.0 * risk, max_hold_bars=int(hold),
                                          features={"range_pips": width / ctx.meta.pip,
                                                    "range_vs_median": width / med if np.isfinite(med) else 1.0,
                                                    "key_level": False}))
                        break
            i = j
        return out
