"""Gold scalp: buy short, sharp dips inside an hourly uptrend (sell spikes inside a downtrend).

Chosen from a walk-forward study on XAUUSD M5 (Jun 2025 - Nov 2026, broker spread per bar,
commission $0.05/oz, 1 pip slippage; first 60% to choose, last 40% unseen), with the highest win rate
among the variants that stayed profitable on the unseen part and under doubled costs:

    trend   price above the hourly 50 & 200 averages (SMA 600 / 2400 on M5) and, on M5,
            EMA50 above EMA200 with EMA50 rising over the last hour (mirror for shorts)
    trigger RSI(2) <= 5 on the bar that just closed (>= 95 for shorts) — a 2-3 bar flush
    when    02:00-12:00 New York (London + New York mornings)
    exits   target 0.75 ATR, stop 4 ATR, flat after 48 bars (4 h); most trades last 10-30 min

Result in that study: 346 trades, win rate 87.9% (unseen part 90.1%), +0.05R per trade, PF 1.48,
worst losing streak 3. The win rate comes from a target ~5x closer than the stop, so one loss costs
about five wins: the edge is real but thin, and it needs the opposing agent's news blackout.
Simple averages (not EMAs) for the hourly trend so the 3000-bar live window computes exactly what the
backtest computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..brain import tehran_hm
from ..features import ema, sma
from ..timeutil import ny_hour_float
from .base import Context, Signal, Strategy


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    """Wilder RSI."""
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


class GoldScalp(Strategy):
    name = "scalp"

    def applies_to(self, symbol: str) -> bool:
        prefixes = self.p.get("symbols", ["XAU"])
        return any(str(symbol).upper().startswith(str(p).upper()) for p in prefixes)

    def generate(self, ctx: Context) -> list[Signal]:
        if not self.applies_to(ctx.symbol):
            return []
        p = self.p
        rsi_n, rsi_lo = int(p.get("rsi_period", 2)), float(p.get("rsi_level", 5))
        sl_k, tp_k = float(p.get("sl_atr", 4.0)), float(p.get("tp_atr", 0.75))
        hold = int(p.get("max_hold_bars", 48))
        h0, h1 = p.get("ny_hours", [2, 12])
        fast, slow = int(p.get("h1_fast_bars", 600)), int(p.get("h1_slow_bars", 2400))
        df = ctx.df
        n = len(df)
        if n < slow + 1:
            return []
        c = df["close"].values
        a = ctx.atr
        e50, e200 = ema(c, 50), ema(c, 200)
        slope = e50 - np.r_[np.full(12, np.nan), e50[:-12]]
        sf, ss = sma(c, fast), sma(c, slow)
        r = rsi(c, rsi_n)
        hr = ny_hour_float(df.index)
        ok = (hr >= h0) & (hr < h1) & np.isfinite(ss) & np.isfinite(slope) & (a > 0)
        with np.errstate(invalid="ignore"):
            up = ok & (c > e200) & (e50 > e200) & (slope > 0) & (c > sf) & (sf > ss) & (r <= rsi_lo)
            dn = ok & (c < e200) & (e50 < e200) & (slope < 0) & (c < sf) & (sf < ss) & (r >= 100 - rsi_lo)
        out: list[Signal] = []
        for i in np.flatnonzero(up | dn):
            if i >= n:  # pragma: no cover
                continue
            d = 1 if up[i] else -1
            e = float(c[i])
            out.append(Signal(bar=int(i), time=df.index[i], symbol=ctx.symbol, strategy=self.name, direction=d,
                              order="market", entry=e, sl=e - d * sl_k * a[i], tp=e + d * tp_k * a[i],
                              valid_bars=1, max_hold_bars=hold,
                              features={"rsi": round(float(r[i]), 1), "atr": round(float(a[i]), 3)}))
        return out

    def explain(self, ctx: Context) -> dict | None:
        if not self.applies_to(ctx.symbol):
            return None
        p = self.p
        rsi_n, rsi_lo = int(p.get("rsi_period", 2)), float(p.get("rsi_level", 5))
        h0, h1 = p.get("ny_hours", [2, 12])
        fast, slow = int(p.get("h1_fast_bars", 600)), int(p.get("h1_slow_bars", 2400))
        df = ctx.df
        n = len(df)
        if n < slow + 1:
            return {"state": "waiting", "levels": {}, "text": "هنوز داده‌ی کافی برای روند ساعتی ندارد"}
        c = df["close"].values
        i = n - 1
        e50, e200 = ema(c, 50), ema(c, 200)
        slope = e50[i] - e50[i - 12]
        sf, ss = sma(c, fast)[i], sma(c, slow)[i]
        r = float(rsi(c, rsi_n)[i])
        up = c[i] > e200[i] and e50[i] > e200[i] and slope > 0 and c[i] > sf and sf > ss
        dn = c[i] < e200[i] and e50[i] < e200[i] and slope < 0 and c[i] < sf and sf < ss
        now = df.index[i]
        win = f"{tehran_hm(float(h0), now)} تا {tehran_hm(float(h1), now)} تهران"
        hr = float(ny_hour_float(df.index[-1:])[0])
        when = "الان در بازه‌ی کاری است" if h0 <= hr < h1 else f"بازه‌ی کاری: {win}"
        if up:
            text = f"روند ساعتی و ۵ دقیقه‌ای صعودی است؛ منتظر یک افت تند (RSI(2) زیر {rsi_lo:g}، الان {r:.0f}) برای خرید؛ {when}"
            state = "armed"
        elif dn:
            text = f"روند ساعتی و ۵ دقیقه‌ای نزولی است؛ منتظر یک جهش تند (RSI(2) بالای {100 - rsi_lo:g}، الان {r:.0f}) برای فروش؛ {when}"
            state = "armed"
        else:
            text = "روندهای ساعتی و ۵ دقیقه‌ای هم‌جهت نیستند؛ فعلاً اسکلپ نمی‌کند"
            state = "idle"
        return {"state": state, "levels": {"rsi2": round(r, 1)}, "text": text}
