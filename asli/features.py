"""Price-action primitives shared by all strategies. Everything here is causal:
a value at bar i only uses bars <= i (swings are reported at their confirmation bar)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import daily_ny
from .timeutil import NY, trading_day


def ema(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).ewm(span=n, adjust=False).mean().values


def sma(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).rolling(n, min_periods=n).mean().values


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


def pivots(high: np.ndarray, low: np.ndarray, k: int):
    """Swing highs/lows with k bars on each side. Returns boolean arrays marking the PIVOT bar.
    A pivot at bar j is only known at bar j+k (use confirmed_levels for causal access)."""
    w = 2 * k + 1
    hmax = pd.Series(high).rolling(w, center=True, min_periods=w).max().values
    lmin = pd.Series(low).rolling(w, center=True, min_periods=w).min().values
    sh = (high == hmax)
    sl = (low == lmin)
    return sh, sl


def confirmed_levels(high, low, k: int):
    """For every bar i: price and index of the most recent swing high/low CONFIRMED by bar i."""
    n = len(high)
    sh, sl = pivots(high, low, k)
    sh_price = np.full(n, np.nan)
    sh_idx = np.full(n, -1)
    sl_price = np.full(n, np.nan)
    sl_idx = np.full(n, -1)
    last_h, last_hi, last_l, last_li = np.nan, -1, np.nan, -1
    for i in range(n):
        j = i - k
        if j >= 0:
            if sh[j]:
                last_h, last_hi = high[j], j
            if sl[j]:
                last_l, last_li = low[j], j
        sh_price[i], sh_idx[i], sl_price[i], sl_idx[i] = last_h, last_hi, last_l, last_li
    return sh_price, sh_idx, sl_price, sl_idx


def fvgs(high, low):
    """Fair value gaps detected at the 3rd candle i. Bullish: low[i] > high[i-2] -> zone [high[i-2], low[i]].
    Returns (bull_lo, bull_hi, bear_lo, bear_hi) arrays, NaN where none."""
    n = len(high)
    bl = np.full(n, np.nan); bh = np.full(n, np.nan)
    sl_ = np.full(n, np.nan); sh_ = np.full(n, np.nan)
    h2 = np.r_[np.nan, np.nan, high[:-2]]
    l2 = np.r_[np.nan, np.nan, low[:-2]]
    bull = low > h2
    bear = high < l2
    bl[bull], bh[bull] = h2[bull], low[bull]
    sl_[bear], sh_[bear] = high[bear], l2[bear]
    return bl, bh, sl_, sh_


def body_strength(df: pd.DataFrame, atr_arr: np.ndarray):
    o, c, h, l = df["open"].values, df["close"].values, df["high"].values, df["low"].values
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    return body / np.maximum(atr_arr, 1e-12), body / rng


class DailyContext:
    """Per-bar daily context mapped onto an intraday frame (trading day rolls at 17:00 NY)."""

    def __init__(self, df: pd.DataFrame, daily: pd.DataFrame | None = None):
        """`daily` (optional): broker D1 candles indexed by trading-day date, used live where the intraday
        window is too short for a 200-day SMA. Otherwise daily candles are built from `df`."""
        self.df = df
        days = trading_day(df.index)
        if daily is not None and len(daily):
            d = daily[["open", "high", "low", "close"]].copy()
            d.index = pd.Index([x.date() if hasattr(x, "date") else x for x in d.index])
        else:
            d = daily_ny(df)
            d.index = pd.Index(d.index.date)  # python dates, same type as `days`
        prev = d.shift(1)
        prev2 = d.shift(2)
        # ICT daily bias for day D, decided from the candle of D-1 against D-2
        bull = (prev["close"] > prev2["high"]) | ((prev["low"] < prev2["low"]) & (prev["close"] > prev2["low"]))
        bear = (prev["close"] < prev2["low"]) | ((prev["high"] > prev2["high"]) & (prev["close"] < prev2["high"]))
        bias = np.where(bull & ~bear, 1, np.where(bear & ~bull, -1, 0))
        sma200 = d["close"].rolling(200, min_periods=50).mean().shift(1)
        trend = np.where(prev["close"] > sma200, 1, np.where(prev["close"] < sma200, -1, 0))
        med_range = (d["high"] - d["low"]).rolling(20, min_periods=5).median().shift(1)
        table = pd.DataFrame({
            "pdh": prev["high"], "pdl": prev["low"], "bias": bias, "trend": trend,
            "adr": med_range,
        }, index=d.index)
        mapped = table.reindex(pd.Index(days))
        self.pdh = mapped["pdh"].values.astype(float)
        self.pdl = mapped["pdl"].values.astype(float)
        self.bias = np.nan_to_num(mapped["bias"].values.astype(float)).astype(int)
        self.trend = np.nan_to_num(mapped["trend"].values.astype(float)).astype(int)
        self.adr = mapped["adr"].values.astype(float)
        self.day = np.asarray(days)
        # Asian range: NY 20:00-24:00 of the trading day, valid after midnight NY
        ny = df.index.tz_convert(NY)
        hr = ny.hour.values
        asia = (hr >= 20)
        hi = pd.Series(np.where(asia, df["high"].values, np.nan)).groupby(self.day).transform("max").values
        lo = pd.Series(np.where(asia, df["low"].values, np.nan)).groupby(self.day).transform("min").values
        valid = ~asia & (hr < 17)
        self.asia_high = np.where(valid, hi, np.nan)
        self.asia_low = np.where(valid, lo, np.nan)
        # running intraday high/low of the trading day (causal)
        g = pd.DataFrame({"h": df["high"].values, "l": df["low"].values, "d": self.day})
        self.day_high = g.groupby("d")["h"].cummax().values
        self.day_low = g.groupby("d")["l"].cummin().values
        self.ny_hour = ny.hour.values + ny.minute.values / 60.0

    def key_levels_above(self, i: int):
        return [x for x in (self.pdh[i], self.asia_high[i]) if np.isfinite(x)]

    def key_levels_below(self, i: int):
        return [x for x in (self.pdl[i], self.asia_low[i]) if np.isfinite(x)]

    def premium_discount(self, i: int, price: float) -> float:
        """Position of price inside the previous day's range: 0 = PDL, 1 = PDH."""
        a, b = self.pdl[i], self.pdh[i]
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            return 0.5
        return float((price - a) / (b - a))
