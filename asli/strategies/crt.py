"""CRT (Candle Range Theory) on the New York 4H grid (1/5/9/13/17/21 NY).

Candle 2 must sweep exactly one side of candle 1 and close back inside it. Entry at the open after
candle 2 closes (Model #1 on the higher timeframe), stop beyond the sweep, 50% partial at candle 1's
midpoint, final target at the opposite extreme. Optional key-level filter: the sweep must also take
the previous day's high/low or the Asian range."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import resample
from ..features import atr
from ..timeutil import NY
from .base import Context, Signal, Strategy

_H4 = pd.Timedelta(hours=4)


class CandleRangeTheory(Strategy):
    name = "crt"

    def generate(self, ctx: Context) -> list[Signal]:
        h4 = resample(ctx.df, "4h", ny_anchor_hour=1)
        if len(h4) < 30:
            return []
        o, h, l, c = (h4[k].values for k in ("open", "high", "low", "close"))
        a = atr(h4, 14)
        hours = set(int(x) for x in self.p.get("candle2_ny_hours", [5, 9]))
        need_key = bool(self.p.get("require_key_level", True))
        min_rr = float(ctx.cfg["risk"]["min_rr"])
        ny_start = h4.index.tz_convert(NY).hour
        d = ctx.daily
        out: list[Signal] = []
        for k in range(15, len(h4)):
            if ny_start[k] not in hours or not np.isfinite(a[k]):
                continue
            if (h4.index[k] - h4.index[k - 1]) != _H4:
                continue  # weekend / data gap — candles are not consecutive
            r1 = h[k - 1] - l[k - 1]
            if r1 < 0.5 * a[k]:
                continue
            bear = h[k] > h[k - 1] and l[k] > l[k - 1] and c[k] < h[k - 1]
            bull = l[k] < l[k - 1] and h[k] < h[k - 1] and c[k] > l[k - 1]
            if not (bear or bull):
                continue
            dirn = -1 if bear else +1
            bar = ctx.m5_bar_at_close(h4.index[k] + _H4)
            if bar < 1 or bar >= len(ctx.df):
                continue
            key_hit = False
            # key level: the sweep candle took a daily/Asian level and closed back on the right side of it
            if dirn == -1:
                for lv in (d.pdh[bar], d.asia_high[bar]):
                    if np.isfinite(lv) and h[k] > lv > c[k]:
                        key_hit = True
            else:
                for lv in (d.pdl[bar], d.asia_low[bar]):
                    if np.isfinite(lv) and l[k] < lv < c[k]:
                        key_hit = True
            if need_key and not key_hit:
                continue
            sp = ctx.spread[bar]
            entry = ctx.df["close"].values[bar] + (sp if dirn == +1 else 0.0)
            sl = h[k] + 0.1 * a[k] + sp if dirn == -1 else l[k] - 0.1 * a[k] - sp
            mid = (h[k - 1] + l[k - 1]) / 2.0
            tp = l[k - 1] if dirn == -1 else h[k - 1]
            risk = abs(entry - sl)
            if risk <= 0 or (mid - entry) * dirn <= 0:
                continue
            if (tp - entry) * dirn / risk < min_rr * 0.75:  # partial at mid lowers the bar slightly
                continue
            out.append(Signal(bar=bar, time=ctx.df.index[bar], symbol=ctx.symbol, strategy=self.name,
                              direction=dirn, order="market", entry=entry, sl=sl, tp=tp, tp1=mid,
                              max_hold_bars=144,
                              features={"candle2_ny": int(ny_start[k]), "key_level": key_hit,
                                        "range_atr": float(r1 / a[k]),
                                        "sweep_depth_atr": float(((h[k] - h[k - 1]) if dirn == -1 else
                                                                  (l[k - 1] - l[k])) / a[k])}))
        return out
