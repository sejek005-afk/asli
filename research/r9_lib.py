"""Round 9 research library: data loading, candle features and triple-barrier net-R labels.

Offline research only (no MetaTrader needed). See r9_plan.md for the pre-registered design.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EET = ZoneInfo("Europe/Athens")
NY = ZoneInfo("America/New_York")

# price-unit costs per symbol: commission is round trip, slippage is per market/stop fill
SPECS = {
    "XAUUSD_o": {"point": 0.01, "comm": 0.20, "slip": 0.10, "spread_floor": 0.0},
    "EURUSD_o": {"point": 1e-5, "comm": 0.00008, "slip": 0.00002, "spread_floor": 0.00002},
}
HORIZONS = (12, 48, 288)
SL_K = 0.6
TP_MULT = 2.0
ATR_N = 288


def load_raw(path: str | Path, symbol: str) -> pd.DataFrame:
    """Raw MT5 export (server-naive epoch seconds) -> UTC-indexed OHLC with spread in price units."""
    spec = SPECS[symbol]
    d = pd.read_csv(path)
    naive = pd.to_datetime(d["time"], unit="s")
    utc = pd.DatetimeIndex(naive).tz_localize(EET, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    out = pd.DataFrame({
        "open": d["open"].to_numpy(float), "high": d["high"].to_numpy(float), "low": d["low"].to_numpy(float),
        "close": d["close"].to_numpy(float), "vol": d["tick_volume"].to_numpy(float),
        "spread": np.maximum(d["spread"].to_numpy(float) * spec["point"], spec["spread_floor"]),
    }, index=utc)
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated()].sort_index()
    return out


def atr_ema(df: pd.DataFrame, n: int = ATR_N) -> np.ndarray:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Value k bars ago (NaN for the first k)."""
    out = np.full(len(a), np.nan)
    if k == 0:
        return a.astype(float).copy()
    out[k:] = a[:-k]
    return out


def _roll_max(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).max().to_numpy()


def _roll_min(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).min().to_numpy()


def build_features(df: pd.DataFrame, other: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scale-free features known at each bar's close. Row t uses bars <= t only."""
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "vol"))
    A = atr_ema(df)
    f: dict[str, np.ndarray] = {}
    # A. last 12 candles: body and wicks
    body = (c - o) / A
    up = (h - np.maximum(o, c)) / A
    dn = (np.minimum(o, c) - l) / A
    for k in range(12):
        f[f"b{k}"] = _shift(body, k)
        f[f"u{k}"] = _shift(up, k)
        f[f"d{k}"] = _shift(dn, k)
    # B. block candles (hourly, 4-hourly, daily) ending at t
    for name, n, cnt in (("h", 12, 8), ("q", 48, 6), ("D", 288, 5)):
        hi, lo = _roll_max(h, n), _roll_min(l, n)
        op = _shift(o, n - 1)
        sc = A * np.sqrt(n)
        bb = (c - op) / sc
        rr = (hi - lo) / sc
        pos = (c - lo) / np.where(hi > lo, hi - lo, np.nan)
        for k in range(cnt):
            f[f"{name}b{k}"] = _shift(bb, k * n)
            f[f"{name}r{k}"] = _shift(rr, k * n)
            if name == "h":
                f[f"{name}p{k}"] = _shift(pos, k * n)
    # C. returns
    for n in (3, 6, 12, 24, 48, 96, 288, 864, 2880):
        f[f"ret{n}"] = (c - _shift(c, n)) / (A * np.sqrt(n))
    # D. position in range
    for n in (12, 48, 288, 1440):
        hi, lo = _roll_max(h, n), _roll_min(l, n)
        f[f"pos{n}"] = (c - lo) / np.where(hi > lo, hi - lo, np.nan)
    # E. volatility / activity
    tr = np.maximum(h - l, 1e-12)
    a12 = pd.Series(h - l).rolling(12, min_periods=12).mean().to_numpy()
    a2880 = pd.Series(h - l).rolling(2880, min_periods=2880).mean().to_numpy()
    a288 = pd.Series(h - l).rolling(288, min_periods=288).mean().to_numpy()
    f["vol_s"] = a12 / A
    f["vol_l"] = a288 / a2880
    f["rng0"] = tr / A
    vm = pd.Series(v).rolling(288, min_periods=288).mean().to_numpy()
    f["tv0"] = v / vm
    f["tv12"] = pd.Series(v).rolling(12, min_periods=12).mean().to_numpy() / vm
    # F. time
    ny = df.index.tz_convert(NY)
    f["ny_hour"] = ny.hour.to_numpy() + ny.minute.to_numpy() / 60.0
    f["dow"] = ny.dayofweek.to_numpy().astype(float)
    # G. spread
    f["spr"] = df["spread"].to_numpy() / A
    # H. other market (last bar at or before t)
    if other is not None:
        oc = other["close"].to_numpy()
        oA = atr_ema(other)
        g: dict[str, np.ndarray] = {}
        for n in (12, 48, 288, 864):
            g[f"x_ret{n}"] = (oc - _shift(oc, n)) / (oA * np.sqrt(n))
        hi, lo = _roll_max(other["high"].to_numpy(), 288), _roll_min(other["low"].to_numpy(), 288)
        g["x_pos288"] = (oc - lo) / np.where(hi > lo, hi - lo, np.nan)
        gdf = pd.DataFrame(g, index=other.index).reindex(df.index, method="ffill", tolerance=pd.Timedelta("30min"))
        for k in gdf.columns:
            f[k] = gdf[k].to_numpy()
    X = pd.DataFrame(f, index=df.index).astype(np.float32)
    X["atr"] = A  # not a feature; used for stops
    return X


PATTERN_COLS_PREFIX = ("b", "u", "d", "hb", "hr", "hp", "qb", "qr", "Db", "Dr", "ret", "pos")


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [k for k in X.columns if k != "atr"]


def pattern_cols(X: pd.DataFrame) -> list[str]:
    out = []
    for k in feature_cols(X):
        if k.startswith("x_") or k in ("vol_s", "vol_l", "rng0", "tv0", "tv12", "ny_hour", "dow", "spr"):
            continue
        out.append(k)
    return out


def barrier_labels(df: pd.DataFrame, A: np.ndarray, h: int, symbol: str) -> dict[str, np.ndarray]:
    """Net R of the standard long and short trade signalled at each bar t (entry at open of t+1).

    Returns arrays aligned with df rows: r_long, r_short, cost_r (one extra cost charge, for the
    doubled-cost check), exit_off (bars from signal bar to exit bar, per side), mkt (long move over the
    time window in units of S, per side's holding period).
    """
    spec = SPECS[symbol]
    comm, slip = spec["comm"], spec["slip"]
    o, hi, lo, c, sp = (df[k].to_numpy() for k in ("open", "high", "low", "close", "spread"))
    n = len(df)
    S = SL_K * A * np.sqrt(h)
    T = TP_MULT * S
    e = np.arange(n) + 1  # entry bar
    valid = (e + h - 1 < n) & np.isfinite(S) & (S > 0)
    ei = np.where(valid, e, 0)
    eo = o[ei]
    esp = sp[ei]
    res = {}
    for side in (1, -1):
        if side == 1:
            entry = eo + esp + slip
            stop, tp = eo - S, eo + T
        else:
            entry = eo - slip
            stop, tp = eo + S, eo - T
        exit_px = np.full(n, np.nan)
        exit_off = np.full(n, -1, dtype=np.int32)
        open_ = valid.copy()
        for k in range(h):
            b = ei + k
            b = np.where(open_, b, 0)
            if side == 1:
                hit_sl = lo[b] <= stop
                hit_tp = hi[b] >= tp
                px_sl = np.minimum(stop, o[b]) - slip
                px_tp = tp
            else:
                hit_sl = hi[b] + sp[b] >= stop
                hit_tp = lo[b] + sp[b] <= tp
                px_sl = np.maximum(stop, o[b] + sp[b]) + slip
                px_tp = tp
            m_sl = open_ & hit_sl
            exit_px[m_sl] = px_sl[m_sl]
            exit_off[m_sl] = k + 1
            open_ &= ~m_sl
            m_tp = open_ & hit_tp
            exit_px[m_tp] = px_tp[m_tp]
            exit_off[m_tp] = k + 1
            open_ &= ~m_tp
        last = np.where(open_, ei + h - 1, 0)
        if side == 1:
            px_t = c[last] - slip
        else:
            px_t = c[last] + sp[last] + slip
        exit_px[open_] = px_t[open_]
        exit_off[open_] = h
        pnl = (exit_px - entry) * side - comm
        r = pnl / S
        r[~valid] = np.nan
        # market move over the holding window (bid open of entry -> bid close of exit bar), in S units
        xb = np.where(valid, np.arange(n) + np.maximum(exit_off, 1), 0)
        mkt = (c[xb] - eo) / S
        mkt[~valid] = np.nan
        key = "long" if side == 1 else "short"
        res[f"r_{key}"] = r
        res[f"off_{key}"] = exit_off
        res[f"mkt_{key}"] = mkt
    cost = (esp + comm + 2 * slip) / S
    cost[~valid] = np.nan
    res["cost_r"] = cost
    res["S"] = S
    return res
