"""Gold New-York opening range: the 08:30-09:00 NY range (US data hour), traded only with the daily trend.

    range   high/low of the M5 bars from 08:30 to 09:00 New York (17:00/18:00 Tehran)
    trigger the first M5 close outside the range before 11:30 NY; long above / short below
    trend   only if the previous day's close is on the same side of the 50-day average
            (a breakout against the trend skips the day — the first breakout decides)
    exits   stop at the other side of the range, target 2x the range, flat at 15:00 NY

Chosen in a 6-year study on this broker's XAUUSD M5 data (Nov 2020 - Nov 2026; spread = max(recorded,
0.55 bp of price), commission $0.05/oz, 1 pip slippage). Five new families and all six existing ASLI
styles were tested; settings were picked on Nov 2020 - Dec 2023 only with a rule fixed in advance, then
judged once on 2024 - Nov 2026. It was the only style that passed:

    in-sample 2021-23:      344 trades, win 50.6%, +0.151R, PF 1.37, positive every year
    unseen 2024-Nov 2026:   387 trades, win 45.5%, +0.049R, PF 1.10, positive every year
    1-minute re-simulation of Nov 2023-Nov 2026 agrees (+0.063R vs +0.057R on M5)

The edge is modest and shrank in 2025-26; doubled costs leave +0.026R unseen. Win rate is ~46-50%:
winners are ~2x the losers.
"""
from __future__ import annotations

import numpy as np

from ..timeutil import ny_hour_float
from .base import Context, Signal, Strategy


class NYOpeningRange(Strategy):
    name = "nyorb"

    def applies_to(self, symbol: str) -> bool:
        prefixes = self.p.get("symbols", ["XAU"])
        return any(str(symbol).upper().startswith(str(p).upper()) for p in prefixes)

    def generate(self, ctx: Context) -> list[Signal]:
        if not self.applies_to(ctx.symbol):
            return []
        p = self.p
        start = float(p.get("range_start_ny", 8.5))
        end = start + float(p.get("range_minutes", 30)) / 60.0
        until = float(p.get("entry_until_ny", 11.5))
        flat = float(p.get("flat_at_ny", 15.0))
        tp_k = float(p.get("tp_range", 2.0))
        use_trend = bool(p.get("with_daily_trend", True))
        df = ctx.df
        n = len(df)
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        hc = ny_hour_float(df.index) + 5.0 / 60.0  # New York clock at each bar's close
        day = ctx.daily.day
        trend = ctx.daily.trend50
        in_rng = (hc > start + 1e-9) & (hc <= end + 1e-9)
        out: list[Signal] = []
        cur_day, hi, lo, done = None, -np.inf, np.inf, False
        for i in range(n):
            if day[i] != cur_day:
                cur_day, hi, lo, done = day[i], -np.inf, np.inf, False
            if in_rng[i]:
                hi, lo = max(hi, h[i]), min(lo, l[i])
                continue
            if done or not (end + 1e-9 < hc[i] <= until + 1e-9) or not np.isfinite(hi) or hi <= lo:
                continue
            d = 1 if c[i] > hi else (-1 if c[i] < lo else 0)
            if d == 0:
                continue
            done = True  # the first breakout of the day decides, even when the trend filter rejects it
            if use_trend and trend[i] != d:
                continue
            if i >= n:  # pragma: no cover
                continue
            r = hi - lo
            e = float(c[i])
            stop = lo if d == 1 else hi
            hold = max(1, int(round((flat - hc[i]) * 12)))
            out.append(Signal(bar=i, time=df.index[i], symbol=ctx.symbol, strategy=self.name, direction=d,
                              order="market", entry=e, sl=float(stop), tp=e + d * tp_k * r, valid_bars=1,
                              max_hold_bars=hold, features={"range": round(float(r), 2)}))
        return out
