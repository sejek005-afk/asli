"""Time handling: broker server clock -> UTC, New York sessions and killzones.

All ICT/CRT/ORB times are defined in New York local time. Tehran has no DST (UTC+03:30),
so Tehran = NY + 8:30 in US winter and NY + 7:30 in US summer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
EET = ZoneInfo("Europe/Athens")  # UTC+2 winter / UTC+3 summer with EU daylight-saving dates
TEHRAN = ZoneInfo("Asia/Tehran")
UTC = timezone.utc

# NY-local killzones (start_hour, end_hour) — end exclusive
KILLZONES = {
    "london": (2, 5),
    "newyork": (7, 10),
}
SESSIONS = [
    # name, NY start hour, NY end hour (exclusive); asia wraps midnight
    ("asia", 20, 24),
    ("asia", 0, 2),
    ("london", 2, 7),
    ("newyork", 7, 12),
    ("late", 12, 20),
]


def server_naive_to_utc(index: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    """Convert a naive broker-server-time index to tz-aware UTC.

    mode: "ny_plus_7" — server clock = New York local + 7h (EET with US DST, used by most ECN brokers)
          "eet"       — server clock = Eastern European Time with EU DST dates (e.g. LiteFinance). Differs
                        from ny_plus_7 by 1 h for ~5 weeks a year, when US and EU clocks change on different days
          "fixed:<h>" — server clock = UTC + h hours (h may be negative / fractional)
          "utc"       — already UTC
    """
    if index.tz is not None:
        return index.tz_convert(UTC)
    if mode == "utc":
        return index.tz_localize(UTC)
    if mode.startswith("fixed:"):
        h = float(mode.split(":", 1)[1])
        return (index - pd.Timedelta(hours=h)).tz_localize(UTC)
    if mode == "eet":
        return index.tz_localize(EET, ambiguous="NaT", nonexistent="shift_forward").tz_convert(UTC)
    if mode == "ny_plus_7":
        ny_naive = index - pd.Timedelta(hours=7)
        loc = ny_naive.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward")
        return loc.tz_convert(UTC)
    raise ValueError(f"unknown server time mode: {mode}")


def ny_index(index_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index_utc.tz_convert(NY)


def ny_hour_float(index_utc: pd.DatetimeIndex) -> np.ndarray:
    ny = ny_index(index_utc)
    return ny.hour.values + ny.minute.values / 60.0


def session_of_hour(h: float) -> str:
    for name, a, b in SESSIONS:
        if a <= h < b:
            return name
    return "late"


def session_labels(index_utc: pd.DatetimeIndex) -> np.ndarray:
    hrs = ny_hour_float(index_utc)
    out = np.empty(len(hrs), dtype=object)
    for i, h in enumerate(hrs):
        out[i] = session_of_hour(h)
    return out


def in_killzone(index_utc: pd.DatetimeIndex, zones=("london", "newyork")) -> np.ndarray:
    hrs = ny_hour_float(index_utc)
    mask = np.zeros(len(hrs), dtype=bool)
    for z in zones:
        a, b = KILLZONES[z]
        mask |= (hrs >= a) & (hrs < b)
    return mask


def trading_day(index_utc: pd.DatetimeIndex) -> pd.Index:
    """Forex trading day rolls at 17:00 New York. Returns the date label of each bar's trading day."""
    ny = ny_index(index_utc)
    shifted = ny + pd.Timedelta(hours=7)  # 17:00 NY becomes 00:00 of next day
    return pd.Index(shifted.date)


def tehran_label(ny_hour: int, when: datetime | None = None) -> str:
    """Human label 'HH:MM' in Tehran for a NY hour, using the DST state of `when` (default now)."""
    when = when or datetime.now(UTC)
    ny_dt = when.astimezone(NY).replace(hour=int(ny_hour) % 24, minute=0, second=0, microsecond=0)
    t = ny_dt.astimezone(TEHRAN)
    return t.strftime("%H:%M")


def now_utc() -> datetime:
    return datetime.now(UTC)


def minutes_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0


__all__ = [
    "NY", "LONDON", "TEHRAN", "UTC", "KILLZONES", "server_naive_to_utc", "ny_index", "ny_hour_float",
    "session_of_hour", "session_labels", "in_killzone", "trading_day", "tehran_label", "now_utc",
    "minutes_between", "timedelta",
]
