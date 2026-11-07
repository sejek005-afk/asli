"""The bot's "brain": a plain-language picture of what it sees and why it acts.

- market view: price, daily trend, how volatile today is, session, the day's range
- per style: what it is waiting for right now (each style's `explain`), its automatic switch, and its
  live results next to what the backtest expected
- learning: live results vs the backtest distribution of the last 2 years. A style whose live average
  falls clearly below its expectation (`drift_z` standard errors, after `drift_min_trades` trades, and below 0)
  is paused automatically — the "it noticed the market changed" rule
- risk preview: what each risk-per-trade level did in the backtest (yearly return, worst drop, typical leverage)
Everything here is read-only except `Expectations`, which the engine reloads after each backtest.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .timeutil import NY, TEHRAN, ny_hour_float, session_of_hour

UTC = timezone.utc

TITLES = {"nyorb": "رنج نیویورک", "donch": "شکست کانال ساعتی", "scalp": "اسکلپ طلا", "ict": "ICT", "lit": "LIT",
          "trendline": "خط روند", "crt": "CRT", "orb": "ORB لندن"}
VERDICTS = {
    "nyorb": ("تأییدشده", "روی ۶ سال طلا تست شد و قبول شد"),
    "donch": ("آزمایشی", "برنده‌ی ۲ سال اخیر؛ با کلید خودکار ۱۲ ماهه"),
    "scalp": ("آزمایشی", "روی ۲۰۲۵ تا ۲۰۲۶ ساخته شد؛ با کلید خودکار ۱۲ ماهه"),
}
REJECTED = ("رد شده", "روی ۶ سال طلا ضرر داد")
SESSIONS_FA = {"asia": "آسیا", "london": "لندن", "newyork": "نیویورک", "late": "بعد از ظهر نیویورک"}


def title(name: str) -> str:
    return TITLES.get(name, name)


def verdict(name: str) -> tuple[str, str]:
    return VERDICTS.get(name, REJECTED)


def tehran_hm(ny_hour: float, ref_utc: pd.Timestamp | datetime | None = None) -> str:
    """'HH:MM' in Tehran for a New-York clock time on the New-York date of `ref_utc` (DST-correct)."""
    ref = pd.Timestamp(ref_utc or datetime.now(UTC))
    ref = ref.tz_localize(UTC) if ref.tzinfo is None else ref
    d = ref.tz_convert(NY)
    h = int(math.floor(ny_hour)) % 24
    m = int(round((ny_hour - math.floor(ny_hour)) * 60))
    t = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=m, tz=NY)
    return t.tz_convert(TEHRAN).strftime("%H:%M")


def fmt_price(x: float) -> str:
    return f"{x:,.2f}"


# ------------------------------------------------------------------------------------------------ market view

def market_view(ctx, now: datetime | None = None) -> dict:
    """What the bot sees on one symbol right now (from the last closed M5 bar)."""
    df = ctx.df
    i = len(df) - 1
    c = float(df["close"].values[i])
    t50 = int(ctx.daily.trend50[i])
    t200 = int(ctx.daily.trend[i])
    trend_txt = {1: "صعودی", -1: "نزولی", 0: "نامشخص"}
    a = np.asarray(ctx.atr, dtype=float)
    hc = ny_hour_float(df.index)
    same_hour = np.flatnonzero(np.floor(hc[-min(len(hc), 288 * 20):]) == np.floor(hc[i]))
    ref = a[-min(len(a), 288 * 20):][same_hour] if len(same_hour) else a[-2000:]
    med = float(np.nanmedian(ref)) if len(ref) else float("nan")
    ratio = float(a[i] / med) if med and np.isfinite(med) and med > 0 else 1.0
    vol = "آرام" if ratio < 0.75 else ("پرنوسان" if ratio > 1.4 else "عادی")
    hi, lo = float(ctx.daily.day_high[i]), float(ctx.daily.day_low[i])
    pdh, pdl = float(ctx.daily.pdh[i]), float(ctx.daily.pdl[i])
    session = session_of_hour(float(hc[i]))
    return {
        "price": round(c, 2), "time": df.index[i].isoformat(),
        "trend": {"d50": t50, "d200": t200, "text": f"روند روزانه {trend_txt[t50]} (نسبت به میانگین ۵۰ روزه)"},
        "volatility": {"ratio": round(ratio, 2), "label": vol,
                       "text": f"نوسان الان {vol} است ({ratio:.1f} برابر معمولِ همین ساعت در ۲۰ روز اخیر)"},
        "session": {"name": session, "label": SESSIONS_FA.get(session, session)},
        "day": {"high": round(hi, 2), "low": round(lo, 2), "prev_high": round(pdh, 2) if np.isfinite(pdh) else None,
                "prev_low": round(pdl, 2) if np.isfinite(pdl) else None},
    }


# ------------------------------------------------------------------------------------------------ expectations

class Expectations:
    """Per-style backtest distribution of R (last `years` of the latest backtest's candidate trades)."""

    def __init__(self, trades: pd.DataFrame | None = None, years: float = 2.0):
        self.trades = trades if trades is not None else pd.DataFrame(columns=["strategy", "time", "r"])
        self.years = years
        self.by_style: dict[str, dict] = {}
        if len(self.trades):
            t = pd.to_datetime(self.trades["time"], utc=True)
            self.trades = self.trades.assign(time=t, exit_time=pd.to_datetime(self.trades.get("exit_time", t), utc=True))
            end = t.max()
            recent = self.trades[t >= end - pd.Timedelta(days=365.25 * years)]
            for name, g in recent.groupby("strategy"):
                r = g["r"].astype(float).values
                stop = (np.abs(g["entry"].astype(float) - g["sl"].astype(float)) / g["entry"].astype(float)).values \
                    if {"entry", "sl"} <= set(g.columns) else np.array([])
                self.by_style[str(name)] = {
                    "n": int(len(r)), "mean": float(r.mean()), "sd": float(r.std(ddof=1)) if len(r) > 1 else 1.0,
                    "win": float((r > 0).mean()), "per_year": float(len(r) / years),
                    "stop_frac": float(np.median(stop)) if len(stop) else None,
                }

    @classmethod
    def load(cls, reports_dir: str | Path, years: float = 2.0) -> "Expectations":
        p = Path(reports_dir) / "all_candidate_trades.csv"
        if not p.exists():
            return cls(None, years)
        try:
            cols = ["strategy", "time", "exit_time", "r", "entry", "sl"]
            df = pd.read_csv(p, usecols=lambda c: c in cols)
            return cls(df, years)
        except Exception:  # noqa: BLE001
            return cls(None, years)

    def switched(self, style: str, lookback_days: float = 365, min_trades: int = 20) -> pd.DataFrame:
        """Trades of `style` that its automatic switch would have allowed: the style's finished trades of the
        previous `lookback_days` made money (>= `min_trades` of them) when the trade was taken."""
        g = self.trades[self.trades["strategy"] == style].sort_values("time")
        if not len(g):
            return g
        ex = g.sort_values("exit_time")
        ext = ex["exit_time"].values
        csum = np.r_[0.0, np.cumsum(ex["r"].astype(float).values)]
        t = g["time"].values
        hi = np.searchsorted(ext, t, side="left")  # finished before the trade
        lo = np.searchsorted(ext, t - np.timedelta64(int(lookback_days * 86400), "s"), side="left")
        n = hi - lo
        s = csum[hi] - csum[lo]
        return g[(n >= min_trades) & (s > 0)]

    def preview(self, risk_pct: float, styles: list[str], switched: tuple | list | set = ()) -> dict:
        """Compounded backtest at `risk_pct` per trade with these styles (styles in `switched` only where their
        automatic switch was on): yearly return and worst drop, over the last `years` and the whole history."""
        out = {"risk": risk_pct}
        tr = self.trades[self.trades["strategy"].isin(styles)] if len(self.trades) else self.trades
        if len(tr) and switched:
            parts = [tr[~tr["strategy"].isin(switched)]] + [self.switched(s) for s in styles if s in switched]
            tr = pd.concat(parts)
        if not len(tr):
            return {**out, "cagr_recent": None, "dd_recent": None, "cagr_all": None, "dd_all": None, "trades_year": 0}
        tr = tr.sort_values("exit_time").reset_index(drop=True)
        end = tr["time"].max()
        recent = tr["time"] >= end - pd.Timedelta(days=365.25 * self.years)
        # "realistic": every style keeps its spread of outcomes but earns only half its backtest edge
        r_all = tr["r"].astype(float)
        m = tr["strategy"].map(r_all[recent].groupby(tr["strategy"][recent]).mean())
        m = m.fillna(tr["strategy"].map(r_all.groupby(tr["strategy"]).mean()))
        half = r_all - 0.5 * m
        for tag, sub, rr in (("recent", tr[recent], tr["r"][recent]), ("all", tr, tr["r"]),
                             ("recent_half", tr[recent], half[recent])):
            if len(sub) < 5:
                out[f"cagr_{tag}"], out[f"dd_{tag}"] = None, None
                continue
            eq = np.cumprod(1 + risk_pct / 100.0 * rr.astype(float).values)
            eq = np.maximum(eq, 1e-9)
            peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
            yrs = max((sub["exit_time"].max() - sub["time"].min()).days / 365.25, 0.25)
            out[f"cagr_{tag}"] = round(float(eq[-1] ** (1 / yrs) - 1) * 100, 1)
            out[f"dd_{tag}"] = round(float((eq / peak - 1).min()) * 100, 1)
        out["trades_year"] = int(round(sum(self.by_style.get(s, {}).get("per_year", 0) for s in styles)))
        out["leverage"] = {s: round(risk_pct / 100.0 / self.by_style[s]["stop_frac"], 2)
                           for s in styles if self.by_style.get(s, {}).get("stop_frac")}
        return out


def drift(exp: dict | None, live_r: list[float], min_trades: int = 20, z: float = 2.0) -> dict:
    """Live results of one style against its backtest expectation."""
    n = len(live_r)
    if not exp or not exp.get("n"):
        return {"status": "unknown", "text": "انتظار بک‌تست برای این سبک موجود نیست", "n": n}
    if n < min_trades:
        return {"status": "collecting", "n": n,
                "text": f"{n} از {min_trades} معامله‌ی واقعی برای مقایسه با بک‌تست جمع شده"}
    m = float(np.mean(live_r))
    se = exp["sd"] / math.sqrt(n)
    zscore = (m - exp["mean"]) / se if se > 0 else 0.0
    if m < exp["mean"] - z * se and m < 0:
        return {"status": "drift", "n": n, "z": round(zscore, 2),
                "text": (f"نتیجه‌ی واقعی ({m:+.2f}R در {n} معامله) خیلی بدتر از انتظار بک‌تست ({exp['mean']:+.2f}R) است؛ "
                         f"احتمال اینکه فقط بدشانسی باشد کمتر از ۳٪ است")}
    if zscore < -1:
        return {"status": "watch", "n": n, "z": round(zscore, 2),
                "text": f"کمی ضعیف‌تر از انتظار ({m:+.2f}R در برابر {exp['mean']:+.2f}R)؛ زیر نظر است"}
    return {"status": "ok", "n": n, "z": round(zscore, 2),
            "text": f"مطابق انتظار ({m:+.2f}R در {n} معامله؛ بک‌تست {exp['mean']:+.2f}R)"}


# ------------------------------------------------------------------------------------------------ summary

def summary_lines(brain: dict) -> list[str]:
    """A few sentences in Persian: what the bot thinks right now."""
    out = []
    if not brain.get("market_open", True):
        out.append("بازار طلا الان بسته است؛ ربات منتظر باز شدن بازار است.")
    for s in brain.get("symbols", []):
        mv = s.get("market")
        if mv:
            out.append(f"{s['symbol']}: قیمت {fmt_price(mv['price'])}، {mv['trend']['text']}؛ {mv['volatility']['text']}.")
        for st in s.get("styles", []):
            if st.get("enabled"):
                gate = st.get("gate") or {}
                extra = f" ({gate.get('why')})" if gate and not gate.get("ok", True) else ""
                out.append(f"{st['title']}: {st.get('text', '')}{extra}")
    news = brain.get("news") or {}
    up = [e for e in news.get("upcoming", []) if e.get("impact") == "High"]
    if news.get("blackout"):
        b = news["blackout"]
        out.append(f"الان به خاطر خبر «{b.get('title')}» معامله‌ی تازه باز نمی‌شود.")
    elif up:
        e = up[0]
        out.append(f"خبر مهم بعدی: {e['cur']} «{e['title']}» ساعت {e.get('tehran', '')} به وقت تهران.")
    else:
        out.append("تا ۲۴ ساعت آینده خبر مهمی در تقویم نیست.")
    for ls in (brain.get("learning") or {}).get("styles", []):
        if ls.get("enabled") and ls.get("drift", {}).get("status") in ("drift", "watch"):
            out.append(f"یادگیری — {ls['title']}: {ls['drift']['text']}")
    return out


def to_tehran(ts) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize(UTC) if t.tzinfo is None else t
    return t.tz_convert(TEHRAN).strftime("%a %H:%M").replace("Mon", "دوشنبه").replace("Tue", "سه‌شنبه") \
        .replace("Wed", "چهارشنبه").replace("Thu", "پنجشنبه").replace("Fri", "جمعه").replace("Sat", "شنبه") \
        .replace("Sun", "یکشنبه")


def recent_cutoff(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)
