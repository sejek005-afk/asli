"""Market data: MetaTrader 5 history, CSV import, local cache, and a synthetic generator for tests.

Every frame used by the engine has a tz-aware UTC DatetimeIndex and columns
open, high, low, close (BID prices) and spread (in PRICE units, not points).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .timeutil import NY, UTC, server_naive_to_utc

log = logging.getLogger("asli.data")


@dataclass
class SymbolMeta:
    name: str
    digits: int = 5
    point: float = 0.00001
    pip: float = 0.0001
    tick_value: float = 1.0  # account currency per tick per 1 lot
    tick_size: float = 0.00001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    currency_base: str = ""
    currency_profit: str = ""

    @staticmethod
    def guess(name: str) -> "SymbolMeta":
        n = name.upper()
        if "JPY" in n:
            return SymbolMeta(n, 3, 0.001, 0.01, 1.0, 0.001, currency_base=n[:3], currency_profit=n[3:6])
        if n.startswith("XAU"):
            return SymbolMeta(n, 2, 0.01, 0.1, 1.0, 0.01, currency_base="XAU", currency_profit="USD")
        return SymbolMeta(n, 5, 0.00001, 0.0001, 1.0, 0.00001, currency_base=n[:3], currency_profit=n[3:6])


def pip_from_digits(digits: int, point: float) -> float:
    return point * 10 if digits in (2, 3, 5) else point


# ----------------------------------------------------------------------------- MT5

def mt5_module():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:  # pragma: no cover - only on Windows servers
        raise RuntimeError("MetaTrader5 package not installed (Windows only): pip install MetaTrader5") from exc
    return mt5


def mt5_connect(cfg: dict):
    mt5 = mt5_module()
    m = cfg["mt5"]
    kwargs = {}
    if m.get("login"):
        kwargs.update(login=int(m["login"]), password=str(m.get("password", "")), server=str(m.get("server", "")))
    ok = mt5.initialize(m["path"], **kwargs) if m.get("path") else mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5


def mt5_symbol_meta(mt5, symbol: str) -> SymbolMeta:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol {symbol} not available in MT5")
    si = mt5.symbol_info(symbol)
    return SymbolMeta(
        name=symbol, digits=si.digits, point=si.point, pip=pip_from_digits(si.digits, si.point),
        tick_value=si.trade_tick_value, tick_size=si.trade_tick_size, volume_min=si.volume_min,
        volume_max=si.volume_max, volume_step=si.volume_step,
        currency_base=si.currency_base, currency_profit=si.currency_profit,
    )


def market_closed(utc_now: datetime) -> bool:
    """FX weekend in UTC, padded by 2h either side: Friday 20:00 -> Sunday 23:59."""
    wd, h = utc_now.weekday(), utc_now.hour
    return wd == 5 or (wd == 4 and h >= 20) or wd == 6


def detect_server_time_mode(mt5, symbol: str) -> str:
    """Compare the last tick's server timestamp with real UTC to find the broker clock."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not tick.time:
        return "ny_plus_7"
    server_now = datetime.utcfromtimestamp(tick.time)  # naive server clock
    real_utc = datetime.utcnow()
    if market_closed(real_utc) or abs((real_utc - server_now).total_seconds()) > 14 * 3600:
        # weekend: the last tick is Friday's close, so the gap says nothing about the broker clock
        return "ny_plus_7"
    offset_h = round((server_now - real_utc).total_seconds() / 1800) / 2  # nearest half hour
    ny_off = datetime.now(NY).utcoffset().total_seconds() / 3600
    if abs(offset_h - (ny_off + 7)) < 0.01:
        return "ny_plus_7"
    return f"fixed:{offset_h}"


def _rates_to_frame(rates, meta: SymbolMeta, time_mode: str) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    idx = pd.to_datetime(df["time"], unit="s")
    df.index = server_naive_to_utc(pd.DatetimeIndex(idx), time_mode)
    df = df[~df.index.isna()]
    out = df[["open", "high", "low", "close"]].astype(float).copy()
    out["spread"] = df["spread"].astype(float) * meta.point if "spread" in df else 0.0
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "time"
    return out


def mt5_history(mt5, symbol: str, years: float, time_mode: str, timeframe: str = "M5") -> pd.DataFrame:
    tf = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
          "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}[timeframe]
    meta = mt5_symbol_meta(mt5, symbol)
    end = datetime.now() + timedelta(days=2)
    start = end - timedelta(days=int(365 * years) + 2)
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=120), end)
        rates = mt5.copy_rates_range(symbol, tf, cur, nxt)
        if rates is not None and len(rates):
            frames.append(_rates_to_frame(rates, meta, time_mode))
        cur = nxt
    if not frames:
        raise RuntimeError(f"no history for {symbol} {timeframe}: {mt5.last_error()} — "
                           "in MT5 set Tools > Options > Charts > Max bars = Unlimited and scroll the chart back")
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def mt5_recent(mt5, symbol: str, bars: int, time_mode: str, timeframe: str = "M5") -> pd.DataFrame:
    tf = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}[timeframe]
    meta = mt5_symbol_meta(mt5, symbol)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    return _rates_to_frame(rates, meta, time_mode)


# ----------------------------------------------------------------------------- cache / CSV

def cache_path(data_dir: str, symbol: str, timeframe: str = "M5") -> Path:
    return Path(data_dir) / f"{symbol.upper()}_{timeframe}.pkl"


def save_cache(df: pd.DataFrame, data_dir: str, symbol: str, timeframe: str = "M5") -> Path:
    p = cache_path(data_dir, symbol, timeframe)
    df.to_pickle(p)
    return p


def load_cache(data_dir: str, symbol: str, timeframe: str = "M5") -> pd.DataFrame | None:
    p = cache_path(data_dir, symbol, timeframe)
    return pd.read_pickle(p) if p.exists() else None


def load_csv(path: str, time_mode: str = "utc", spread_pips: float | None = None, pip: float = 0.0001) -> pd.DataFrame:
    """CSV with columns time,open,high,low,close[,spread]. `spread` must be in points of 1/10 pip
    (MT5 export convention) unless `spread_pips` is given to set a constant spread."""
    df = pd.read_csv(path)
    cols = {c.lower().strip("<>"): c for c in df.columns}
    tcol = cols.get("time") or cols.get("datetime") or cols.get("date")
    idx = pd.DatetimeIndex(pd.to_datetime(df[tcol]))
    out = pd.DataFrame({k: df[cols[k]].astype(float).values for k in ("open", "high", "low", "close")}, index=idx)
    if spread_pips is not None:
        out["spread"] = spread_pips * pip
    elif "spread" in cols:
        out["spread"] = df[cols["spread"]].astype(float).values * pip / 10
    else:
        out["spread"] = 0.0
    out.index = server_naive_to_utc(out.index, time_mode)
    out = out[~out.index.isna()]
    out.index.name = "time"
    return out.sort_index()


# ----------------------------------------------------------------------------- resampling

def resample(df: pd.DataFrame, rule: str, ny_anchor_hour: int | None = None) -> pd.DataFrame:
    """Resample OHLC. With ny_anchor_hour, bins are aligned in New York local time
    (e.g. rule='4h', anchor=1 gives the 1/5/9/13/17/21 NY grid used by CRT)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}
    if ny_anchor_hour is None:
        out = df.resample(rule, label="left", closed="left").agg(agg)
    else:
        ny = df.copy()
        ny.index = ny.index.tz_convert(NY).tz_localize(None)
        out = ny.resample(rule, label="left", closed="left", offset=f"{ny_anchor_hour}h").agg(agg)
        out.index = out.index.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert(UTC)
        out = out[~out.index.isna()]
    return out.dropna(subset=["open"])


def daily_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Forex daily candles that roll at 17:00 New York."""
    ny = df.copy()
    ny.index = ny.index.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)
    d = ny.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d.index = pd.DatetimeIndex(d.index.date)
    return d


# ----------------------------------------------------------------------------- synthetic (tests only)

def synthetic(days: int = 120, seed: int = 7, start: str = "2025-01-06", spread_pips: float = 0.8) -> pd.DataFrame:
    """Random-walk M5 EURUSD-like data with session-dependent volatility, liquidity sweeps and
    trending regimes. ONLY for testing the pipeline — results on it mean nothing about real markets."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz=UTC)
    ny = idx.tz_convert(NY)
    keep = ~(((ny.dayofweek == 4) & (ny.hour >= 17)) | (ny.dayofweek == 5) | ((ny.dayofweek == 6) & (ny.hour < 17)))
    idx = idx[keep]
    ny = idx.tz_convert(NY)
    h = ny.hour.values
    vol = np.where((h >= 2) & (h < 5), 2.2, np.where((h >= 7) & (h < 11), 2.6, np.where((h >= 20) | (h < 2), 0.7, 1.1)))
    vol = vol * 0.00008
    n = len(idx)
    drift = np.repeat(rng.normal(0, 0.000004, n // 576 + 1), 576)[:n]
    ret = rng.standard_t(4, n) * vol * 0.6 + drift
    close = 1.08 + np.cumsum(ret)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 1, (2, n))) * vol * 0.8
    spike = rng.random(n) < 0.01
    wick[0][spike] *= 4
    wick[1][rng.random(n) < 0.01] *= 4
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "spread": spread_pips * 0.0001}, index=idx)
    df.index.name = "time"
    return df
