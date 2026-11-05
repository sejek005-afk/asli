"""Fundamentals: economic calendar (blackouts + event risk) and live news headline sentiment.

- Calendar: ForexFactory weekly JSON feed. High-impact events block new trades on the affected
  currencies for +-30 min (configurable), medium for +-10 min; upcoming high impact raises `news_risk`.
- News: RSS headlines scored with a small hawkish/dovish + risk lexicon per currency, decaying with a
  half-life. `bias(symbol)` = score(base) - score(quote) in [-1, 1]. A crude but transparent signal:
  the opposing agent uses it as an argument, never as a trade trigger.
- Every network failure is fail-safe: the bot reports "news unknown" and the opposing agent treats
  that as a reason for caution.
"""
from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("asli.fundamental")
UTC = timezone.utc
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ASLI-bot/1.0"}

CURRENCY_WORDS = {
    "USD": ["dollar", "usd", "fed", "fomc", "powell", "treasury", "u.s.", "us ", "nfp", "payrolls", "american"],
    "EUR": ["euro", "eur", "ecb", "lagarde", "eurozone", "euro zone", "german", "germany"],
    "GBP": ["pound", "sterling", "gbp", "boe", "bank of england", "bailey", "uk ", "britain", "british"],
    "JPY": ["yen", "jpy", "boj", "bank of japan", "ueda", "japan"],
    "AUD": ["aussie", "aud", "rba", "australia"],
    "CAD": ["loonie", "cad", "boc", "bank of canada", "canada"],
    "CHF": ["franc", "chf", "snb", "swiss"],
    "NZD": ["kiwi", "nzd", "rbnz", "new zealand"],
    "XAU": ["gold", "xau", "bullion"],
}
POS = ["hawkish", "hike", "hikes", "raise rates", "strong", "stronger", "beats", "beat", "surge", "surges",
       "jumps", "rally", "rallies", "gains", "rises", "hot inflation", "robust", "upbeat", "accelerat"]
NEG = ["dovish", "cut", "cuts", "rate cut", "weak", "weaker", "misses", "miss", "slump", "slumps", "falls",
       "drops", "plunge", "tumbles", "slides", "recession", "contraction", "cools", "slowdown", "dives"]
RISK_OFF = ["war", "attack", "missile", "invasion", "crisis", "default", "panic", "crash", "sanction", "escalat"]
SAFE_HAVENS = {"JPY": 1.0, "CHF": 1.0, "XAU": 1.0, "USD": 0.5}


def _get(url: str, timeout: float = 15.0) -> bytes | None:
    try:
        import requests
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code == 200:
            return r.content
        log.warning("GET %s -> %s", url, r.status_code)
    except Exception as exc:  # noqa: BLE001
        log.warning("GET %s failed: %s", url, exc)
    return None


def symbol_currencies(symbol: str) -> tuple[str, str]:
    s = symbol.upper()
    return s[:3], s[3:6]


class Fundamentals:
    def __init__(self, cfg: dict, state_dir: str):
        self.cfg = cfg["fundamental"]
        self.enabled = bool(self.cfg.get("enabled", True))
        self.cache = Path(state_dir) / "calendar.json"
        self.events: list[dict] = []
        self.headlines: list[dict] = []  # {t, title, scores{cur: s}, risk_off}
        self._cal_at = 0.0
        self._news_at = 0.0
        self.calendar_ok = False
        self.news_ok = False
        if self.cache.exists():
            try:
                self.events = self._parse_calendar(json.loads(self.cache.read_text(encoding="utf-8")))
                self.calendar_ok = bool(self.events)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ calendar
    @staticmethod
    def _parse_calendar(raw) -> list[dict]:
        out = []
        for e in raw or []:
            try:
                t = datetime.fromisoformat(str(e["date"])).astimezone(UTC)
            except Exception:  # noqa: BLE001
                continue
            out.append({"t": t, "cur": str(e.get("country", "")).upper(), "impact": str(e.get("impact", "")),
                        "title": str(e.get("title", "")), "forecast": e.get("forecast"), "previous": e.get("previous")})
        return sorted(out, key=lambda x: x["t"])

    def refresh(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if force or now - self._cal_at > 60 * float(self.cfg.get("calendar_refresh_min", 60)):
            self._cal_at = now
            raw = _get(self.cfg["calendar_url"])
            if raw:
                try:
                    data = json.loads(raw)
                    self.events = self._parse_calendar(data)
                    self.cache.write_text(json.dumps(data), encoding="utf-8")
                    self.calendar_ok = True
                except Exception as exc:  # noqa: BLE001
                    log.warning("calendar parse failed: %s", exc)
        if force or now - self._news_at > 60 * float(self.cfg.get("news_refresh_min", 15)):
            self._news_at = now
            got = False
            for url in self.cfg.get("news_feeds", []):
                raw = _get(url)
                if raw:
                    got = True
                    self._ingest_rss(raw)
            self.news_ok = got
            cutoff = datetime.now(UTC) - timedelta(hours=48)
            self.headlines = [h for h in self.headlines if h["t"] > cutoff][-500:]

    def blackout(self, symbol: str, t: datetime) -> dict | None:
        """The calendar event that currently blocks trading this symbol, if any."""
        curs = set(symbol_currencies(symbol))
        hi = timedelta(minutes=float(self.cfg.get("blackout_high_min", 30)))
        md = timedelta(minutes=float(self.cfg.get("blackout_medium_min", 10)))
        for e in self.events:
            if e["cur"] not in curs:
                continue
            win = hi if e["impact"] == "High" else (md if e["impact"] == "Medium" else None)
            if win is not None and e["t"] - win <= t <= e["t"] + win:
                return e
        return None

    def event_risk(self, symbol: str, t: datetime, hours: float = 4.0) -> tuple[float, list[dict]]:
        """0..1 score for high/medium impact events on this symbol's currencies in the next `hours`."""
        curs = set(symbol_currencies(symbol))
        soon = [e for e in self.events if e["cur"] in curs and t <= e["t"] <= t + timedelta(hours=hours)
                and e["impact"] in ("High", "Medium")]
        score = 0.0
        for e in soon:
            mins = (e["t"] - t).total_seconds() / 60
            w = 1.0 if e["impact"] == "High" else 0.4
            score += w * max(0.2, 1 - mins / (hours * 60))
        return min(score, 1.0), soon

    # ------------------------------------------------------------------ news
    def _ingest_rss(self, raw: bytes) -> None:
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return
        seen = {h["title"] for h in self.headlines}
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title or title in seen:
                continue
            pub = item.findtext("pubDate") or ""
            try:
                from email.utils import parsedate_to_datetime
                t = parsedate_to_datetime(pub).astimezone(UTC)
            except Exception:  # noqa: BLE001
                t = datetime.now(UTC)
            self.headlines.append(self.score_headline(title, t))

    @staticmethod
    def score_headline(title: str, t: datetime) -> dict:
        low = " " + title.lower() + " "
        tone = sum(1 for w in POS if w in low) - sum(1 for w in NEG if w in low)
        risk_off = any(w in low for w in RISK_OFF)
        scores = {}
        for cur, words in CURRENCY_WORDS.items():
            if any(re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low) for w in words):
                scores[cur] = float(np.sign(tone)) if tone else 0.0
        return {"t": t, "title": title, "scores": scores, "risk_off": risk_off}

    def currency_score(self, cur: str, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        hl = float(self.cfg.get("news_halflife_hours", 6))
        num = den = 0.0
        for h in self.headlines:
            age = (now - h["t"]).total_seconds() / 3600
            if age < 0 or age > 48:
                continue
            w = 0.5 ** (age / hl)
            s = h["scores"].get(cur)
            if s is not None:
                num += w * s
                den += w
            if h["risk_off"] and cur in SAFE_HAVENS:
                num += w * 0.5 * SAFE_HAVENS[cur]
                den += w * 0.5
        return float(np.tanh(num / den * 1.5)) if den > 0 else 0.0

    def bias(self, symbol: str, now: datetime | None = None) -> float:
        base, quote = symbol_currencies(symbol)
        return float(np.clip(self.currency_score(base, now) - self.currency_score(quote, now), -1, 1))

    def risk_off_level(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        recent = [h for h in self.headlines if (now - h["t"]) < timedelta(hours=6)]
        if not recent:
            return 0.0
        return sum(h["risk_off"] for h in recent) / len(recent)

    def snapshot(self, symbols: list[str]) -> dict:
        now = datetime.now(UTC)
        up = [e for e in self.events if now - timedelta(hours=1) <= e["t"] <= now + timedelta(hours=24)
              and e["impact"] in ("High", "Medium")]
        return {
            "calendar_ok": self.calendar_ok, "news_ok": self.news_ok,
            "upcoming": [{"time": e["t"].isoformat(), "cur": e["cur"], "impact": e["impact"], "title": e["title"]}
                         for e in up[:20]],
            "bias": {s: round(self.bias(s, now), 2) for s in symbols},
            "risk_off": round(self.risk_off_level(now), 2),
            "headlines": [{"time": h["t"].isoformat(), "title": h["title"]} for h in self.headlines[-15:]],
        }

