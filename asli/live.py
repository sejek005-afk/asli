"""Live engine: every closed M5 bar, run all five styles on recent data, score new candidates with the
learner, attach fundamentals, let the opposing agent decide, size the position, send the order, manage
open positions (50% partial + break-even), and learn from every closed trade.

State (positions meta, decisions, journal) is kept in state/*.json so a restart loses nothing.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import notify
from . import brain as brainlib
from .broker import lot_size, round_volume
from .config import ROOT, for_symbol
from .data import market_closed, mt5_connect, mt5_history, mt5_recent, mt5_symbol_meta
from .devil import DevilsAdvocate
from .fundamental import Fundamentals
from .fusion import annotate
from .learner import Learner
from .strategies import ALL as ALL_STYLES
from .strategies import Context, build
from .timeutil import EET, NY, trading_day

log = logging.getLogger("asli.live")
UTC = timezone.utc
BARS = 3000  # ~10 trading days of M5
# what the app may change at runtime, with hard bounds (state/settings.json, applied over config.yaml)
SETTING_BOUNDS = {"risk_per_trade_pct": (0.05, 2.0), "max_daily_loss_pct": (0.5, 6.0), "max_open_positions": (1, 6),
                  "max_leverage": (1.0, 30.0), "capital_usd": (10.0, 10_000_000.0), "trade_risk_usd": (0.5, 1_000_000.0)}
MODE_KEYS = {"capital_mode": ("auto", "manual"), "trade_risk_mode": ("auto", "manual")}
MAX_TRADE_RISK_PCT = 2.0  # a hand-typed amount per trade is capped at this share of the capital


def utc_to_server(t: datetime, mode: str) -> datetime:
    """Inverse of timeutil.server_naive_to_utc: express a UTC instant on the broker's server clock.
    Returned as a tz-aware 'UTC' datetime whose wall clock equals the server clock, because MT5's Python API
    interprets epoch seconds as server time."""
    t = t.astimezone(UTC)
    if mode == "utc":
        return t
    if mode.startswith("fixed:"):
        return t + timedelta(hours=float(mode.split(":", 1)[1]))
    if mode == "eet":
        return t.astimezone(EET).replace(tzinfo=None).replace(tzinfo=UTC)
    ny_wall = t.astimezone(NY).replace(tzinfo=None)  # ny_plus_7: server = New York wall clock + 7h
    return (ny_wall + timedelta(hours=7)).replace(tzinfo=UTC)


class Engine:
    def __init__(self, cfg: dict, broker, mt5=None, time_mode: str = "ny_plus_7"):
        self.cfg = cfg
        self.broker = broker
        self.mt5 = mt5
        self.time_mode = time_mode
        self.state_dir = Path(cfg["paths"]["state"])
        self.strategies = build(cfg)
        self.learner = Learner.load(self.state_dir / "learner.json")
        self.devil = DevilsAdvocate(cfg)
        self.fund = Fundamentals(cfg, str(self.state_dir))
        self.metas = {}
        self.last_bar: dict[str, pd.Timestamp] = {}
        self._d1_warned: set[str] = set()
        self.seen: set[str] = set()
        self.decisions: deque = deque(maxlen=200)
        self.trade_meta: dict[str, dict] = self._load("trade_meta.json", {})
        self.journal: list[dict] = self._load("journal.json", [])
        self.paused = False
        self.status = "starting"
        self.lock = threading.RLock()
        self.loss_streak = 0
        self.day_start_balance = None
        self.day_key = None
        self.switch: dict = self._load("switch.json", {})
        self.last_ctx: dict[str, Context] = {}
        self.expect = brainlib.Expectations.load(cfg["paths"]["reports"])
        self.update_state: dict = {"state": "idle", "log": []}
        self.settings: dict = {}
        saved = self._load("settings.json", {})
        self.telegram = notify.Telegram(cfg, self.state_dir, saved)
        self.apply_settings(saved, save=False)
        # look back to the oldest trade we still track (covers restarts after long downtime)
        opened = [datetime.fromisoformat(m["opened"]) for m in self.trade_meta.values() if m.get("opened")]
        self._deals_since = min(opened + [datetime.now(UTC)]) - timedelta(days=1)

    # ------------------------------------------------------------------ persistence
    def _load(self, name, default):
        p = self.state_dir / name
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
        except Exception:  # noqa: BLE001
            return default

    def _save(self, name, obj):
        (self.state_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")

    # ------------------------------------------------------------------ data
    def frame(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        df = mt5_recent(self.mt5, symbol, BARS, self.time_mode, "M5")
        daily = None
        try:
            d1 = mt5_recent(self.mt5, symbol, 260, self.time_mode, "D1")
            # broker D1 candles open at 17:00 NY (server midnight) for ny_plus_7 brokers -> label by trading day
            d1.index = pd.Index(trading_day(d1.index + pd.Timedelta(hours=1)))
            daily = d1
        except Exception as exc:  # noqa: BLE001
            if symbol not in self._d1_warned:
                self._d1_warned.add(symbol)
                log.warning("D1 history unavailable for %s (daily context built from M5): %s", symbol, exc)
        return df, daily

    # ------------------------------------------------------------------ account guards
    def _roll_day(self, acc: dict) -> None:
        """Daily-loss baseline: balance at the forex day roll (17:00 New York), persisted across restarts."""
        today = str(trading_day(pd.DatetimeIndex([pd.Timestamp.now(tz=UTC)]))[0])
        if self.day_key != today:
            saved = self._load("day.json", {})
            if saved.get("day") == today:
                self.day_key, self.day_start_balance = today, float(saved["balance"])
            else:
                self.day_key, self.day_start_balance = today, acc["balance"]
                self._save("day.json", {"day": today, "balance": acc["balance"]})

    def _owner(self, ticket, comment: str) -> str | None:
        """Which style opened a position/order: our trade journal first, else the 'asli <style>' comment."""
        m = self.trade_meta.get(str(ticket))
        if m and m.get("strategy"):
            return str(m["strategy"])
        parts = str(comment or "").split()
        return parts[1] if len(parts) >= 2 and parts[0] == "asli" else None

    def one_position_per(self) -> str:
        mode = str(self.cfg["risk"].get("one_position_per", "symbol"))
        if mode == "strategy" and not getattr(self.broker, "hedging", lambda: False)():
            return "symbol"  # a netting account merges positions of one symbol: styles can't share it
        return mode

    def account_state(self, symbol: str, strategy: str | None = None) -> dict:
        acc = self.broker.account()
        self._roll_day(acc)
        start = self.capital(self.day_start_balance or acc["balance"])
        loss_pct = max(0.0, -self.own_day_pl()) / start * 100 if start else 0.0  # this bot's own trades only
        pos = self.broker.positions()
        orders = self.broker.orders()
        mine = [x for x in list(pos) + list(orders) if x.symbol == symbol]
        if self.one_position_per() == "strategy" and strategy:
            same_sym = False
            same_strat = any(self._owner(x.ticket, x.comment) in (strategy, None) for x in mine)
        else:
            same_sym, same_strat = bool(mine), False
        return {"daily_loss_pct": loss_pct, "open_positions": len(pos) + len(orders),
                "same_symbol_open": same_sym, "same_strategy_open": same_strat,
                "loss_streak": self.loss_streak, "balance": acc["balance"], "equity": acc["equity"]}

    # ------------------------------------------------------------------ automatic on/off switch
    def _refresh_switches(self, force: bool = False) -> None:
        """Once per trading day, replay every `auto_switch` style over its last `switch_lookback_days` of this
        broker's history (same simulator as the backtest). It may trade only while that replay made money
        with at least `switch_min_trades` finished trades."""
        strats = [s for s in self.strategies if s.p.get("auto_switch")]
        if not strats or self.mt5 is None:
            return
        now = pd.Timestamp.now(tz=UTC)
        today = str(trading_day(pd.DatetimeIndex([now]))[0])
        if not force and self.switch.get("_day") == today:
            retry = self.switch.get("_retry_at")
            if not retry or now < pd.Timestamp(retry):
                return
        out: dict = {"_day": today, "_at": now.isoformat()}
        look_max = max(float(s.p.get("switch_lookback_days", 365)) for s in strats)
        for symbol in self.cfg["symbols"]:
            todo = [s for s in strats if not hasattr(s, "applies_to") or s.applies_to(symbol)]
            if not todo:
                continue
            try:
                meta = self.metas.get(symbol) or mt5_symbol_meta(self.mt5, symbol)
                df = mt5_history(self.mt5, symbol, (look_max + 60) / 365.25, self.time_mode, "M5")
                df = df[df.index < now - pd.Timedelta(minutes=5)]  # closed bars only
                try:
                    d1 = mt5_recent(self.mt5, symbol, int(look_max) + 400, self.time_mode, "D1")
                    d1.index = pd.Index(trading_day(d1.index + pd.Timedelta(hours=1)))
                except Exception:  # noqa: BLE001 - the daily context is then built from the M5 history
                    d1 = None
                ctx = Context(df, meta, self.cfg, d1)
            except Exception as exc:  # noqa: BLE001
                log.warning("auto switch: history for %s unavailable: %s", symbol, exc)
                for s in todo:
                    out[f"{symbol}|{s.name}"] = {"on": False, "why": f"history unavailable: {exc}"}
                continue
            for s in todo:
                look = float(s.p.get("switch_lookback_days", 365))
                min_n = int(s.p.get("switch_min_trades", 20))
                start = now - pd.Timedelta(days=look)
                info = {"on": False, "since": start.isoformat()}
                try:
                    covered = (df.index[-1] - df.index[0]).days
                    tr = bt.run(s.generate(ctx), ctx)
                    if len(tr):
                        t0 = pd.to_datetime(tr["time"], utc=True)
                        t1 = pd.to_datetime(tr["exit_time"], utc=True)
                        tr = tr[(t0 >= start) & (t1 < df.index[-1])]  # finished trades inside the window
                    n = len(tr)
                    tot = float(tr["r"].sum()) if n else 0.0
                    info.update({"trades": n, "sum_r": round(tot, 2), "avg_r": round(tot / n, 3) if n else 0.0,
                                 "win_rate": round(float((tr["r"] > 0).mean()), 3) if n else 0.0,
                                 "covered_days": int(covered)})
                    if covered < 0.9 * look:
                        info["why"] = f"only {covered} days of history"
                    elif n < min_n:
                        info["why"] = f"only {n} trades in the window (need {min_n})"
                    elif tot <= 0:
                        info["why"] = f"last {int(look)} days lost {tot:+.1f}R"
                    else:
                        info["on"] = True
                except Exception as exc:  # noqa: BLE001
                    info["why"] = f"replay failed: {exc}"
                    log.exception("auto switch replay %s %s failed", symbol, s.name)
                out[f"{symbol}|{s.name}"] = info
                log.info("auto switch %s %s: %s", symbol, s.name, info)
        if any(isinstance(v, dict) and str(v.get("why", "")).startswith(("history unavailable", "replay failed"))
               for v in out.values()):
            out["_retry_at"] = (now + pd.Timedelta(minutes=30)).isoformat()  # try again soon, not tomorrow
        self.switch = out
        self._save("switch.json", out)

    def switch_on(self, symbol: str, strategy: str) -> tuple[bool, str]:
        s = next((x for x in self.strategies if x.name == strategy), None)
        if s is None or not s.p.get("auto_switch"):
            return True, ""
        info = self.switch.get(f"{symbol}|{strategy}")
        if not info:
            return False, "کلید خودکار هنوز محاسبه نشده"
        if info.get("on"):
            return True, ""
        return False, f"کلید خودکار خاموش است ({info.get('why', '')})"

    # ------------------------------------------------------------------ runtime settings (from the app)
    def apply_settings(self, s: dict, save: bool = True) -> dict:
        """Validate and apply settings sent by the app; unknown keys are ignored, numbers are clamped."""
        s = dict(s or {})
        cur = dict(self.settings)
        for k, (lo, hi) in SETTING_BOUNDS.items():
            if k in s and s[k] is not None:
                v = float(s[k])
                v = int(round(v)) if k == "max_open_positions" else round(v, 2)
                cur[k] = min(max(v, lo), hi)
        for k, allowed in MODE_KEYS.items():
            if s.get(k) in allowed:
                cur[k] = s[k]
        if isinstance(s.get("telegram_token"), str):
            if self.telegram.set_token(s["telegram_token"]):
                cur["telegram_token"] = self.telegram.token
                self._tg_note = ""
            else:
                self._tg_note = "توکن تلگرام شکل درستی ندارد و ذخیره نشد (باید چیزی مثل 123456789:ABC... باشد)"
        styles = dict(cur.get("styles", {}))
        for name, on in (s.get("styles") or {}).items():
            if name in ALL_STYLES:
                styles[name] = bool(on)
        cur["styles"] = styles
        resets = dict(cur.get("learning_reset", {}))
        now = datetime.now(UTC).isoformat()
        for name in s.get("reset_learning") or []:
            if name in ALL_STYLES:
                resets[name] = now
        for name, on in (s.get("styles") or {}).items():  # switching a style on again starts a fresh comparison
            if on and name in ALL_STYLES and self._drift_of(name).get("status") == "drift":
                resets[name] = now
        cur["learning_reset"] = resets
        # apply to the running config
        r = self.cfg["risk"]
        for k in MODE_KEYS:
            if k in cur:
                r[k] = cur[k]
        for k in SETTING_BOUNDS:
            if k in cur:
                r[k] = cur[k]
                for ov in (self.cfg.get("symbol_overrides") or {}).values():
                    if isinstance(ov, dict) and k in (ov.get("risk") or {}):
                        ov["risk"][k] = cur[k]
        for name, on in styles.items():
            self.cfg["strategies"].setdefault(name, {})["enabled"] = on
        self.strategies = build(self.cfg)
        self.devil.max_daily = float(r["max_daily_loss_pct"])
        self.devil.max_open = int(r["max_open_positions"])
        cur["updated_at"] = now if save else cur.get("updated_at")
        self.settings = cur
        if save:
            self._save("settings.json", cur)
            log.info("settings changed from the app: %s",
                     {k: v for k, v in cur.items() if k not in ("learning_reset", "telegram_token")})
        return self.settings_view()

    def settings_view(self) -> dict:
        r = self.cfg["risk"]
        enabled = [s.name for s in self.strategies]
        styles = []
        for name in ALL_STYLES:
            badge, note = brainlib.verdict(name)
            styles.append({"name": name, "title": brainlib.title(name), "enabled": name in enabled, "badge": badge,
                           "note": note, "auto_switch": bool(self.cfg["strategies"].get(name, {}).get("auto_switch"))})
        sw = {s.name for s in self.strategies if s.p.get("auto_switch")}
        preview = [self.expect.preview(x, enabled, sw) for x in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
        order = {"تأییدشده": 0, "آزمایشی": 1}
        styles.sort(key=lambda x: (not x["enabled"], order.get(x["badge"], 2)))
        return {"values": {**{k: r[k] for k in SETTING_BOUNDS}, **{k: r.get(k, "auto") for k in MODE_KEYS}},
                "bounds": SETTING_BOUNDS, "styles": styles, "sizing": self.sizing_view(),
                "preview": preview, "current_preview": self.expect.preview(float(r["risk_per_trade_pct"]), enabled, sw),
                "leverage": self.leverage_view(), "updated_at": self.settings.get("updated_at"),
                "telegram": {**self.telegram.view(), "note": getattr(self, "_tg_note", "")}}

    # ------------------------------------------------------------------ money: capital and amount per trade
    def capital(self, balance: float) -> float:
        """Money the bot sizes from: the whole balance (auto) or the amount typed in the app, at most the balance."""
        r = self.cfg["risk"]
        if r.get("capital_mode") == "manual" and float(r.get("capital_usd") or 0) > 0:
            return min(float(r["capital_usd"]), balance)
        return balance

    def trade_risk(self, balance: float, risk_pct: float | None = None) -> tuple[float, list[str]]:
        """Money at risk on the next trade (lost if its stop is hit) and why."""
        r = self.cfg["risk"]
        cap = self.capital(balance)
        notes = []
        if r.get("trade_risk_mode") == "manual" and float(r.get("trade_risk_usd") or 0) > 0:
            money = float(r["trade_risk_usd"])
            top = cap * MAX_TRADE_RISK_PCT / 100.0
            if money > top:
                notes.append(f"مبلغ دستی {money:,.0f} دلار به سقف {MAX_TRADE_RISK_PCT:g}٪ سرمایه ({top:,.0f} دلار) کم شد")
                money = top
            return money, notes
        pct = float(r["risk_per_trade_pct"]) if risk_pct is None else risk_pct
        if self.loss_streak >= int(r.get("halve_after_losses", 3)):
            pct /= 2
            notes.append(f"بعد از {self.loss_streak} ضرر پشت سر هم، مبلغ هر معامله نصف شد")
        return cap * pct / 100.0, notes

    def sizing_view(self) -> dict:
        try:
            bal = float(self.broker.account()["balance"])
        except Exception:  # noqa: BLE001
            return {}
        r = self.cfg["risk"]
        cap = self.capital(bal)
        money, notes = self.trade_risk(bal)
        return {"balance": round(bal, 2), "capital": round(cap, 2), "capital_mode": r.get("capital_mode", "auto"),
                "trade_risk_mode": r.get("trade_risk_mode", "auto"), "risk_money": round(money, 2),
                "risk_pct_of_capital": round(money / cap * 100, 3) if cap > 0 else None, "notes": notes,
                "max_trade_risk_pct": MAX_TRADE_RISK_PCT}

    def own_day_pl(self) -> float:
        """This bot's own profit since the 17:00 New-York day roll: closed trades + open positions at current
        prices. Other bots on the same account (e.g. NeuroTrader) do not count."""
        today = self.day_key or str(trading_day(pd.DatetimeIndex([pd.Timestamp.now(tz=UTC)]))[0])
        closed = [j for j in getattr(self, "journal", []) if j.get("closed")]
        pl = 0.0
        if closed:
            days = trading_day(pd.DatetimeIndex(pd.to_datetime([j["closed"] for j in closed], utc=True)))
            pl += sum(float(j.get("profit") or 0) for j, d in zip(closed, days) if str(d) == today)
        for p in self.broker.positions():
            try:
                tk = self.broker.tick(p.symbol)
                px = tk.bid if p.direction == 1 else tk.ask
                pl += p.direction * (px - p.entry) * p.volume * self._value_per_lot(p.symbol, 1.0)
            except Exception:  # noqa: BLE001
                continue
        return pl

    def _value_per_lot(self, symbol: str, price: float) -> float:
        """Account-currency value of one lot at `price` (gold: 100 oz x price)."""
        meta = getattr(self, "metas", {}).get(symbol)
        return price * (meta.tick_value / meta.tick_size) if meta and meta.tick_size else price

    def exposure(self) -> float:
        """Account-currency value of every open ASLI position and pending order."""
        tot = 0.0
        for p in self.broker.positions():
            tot += abs(p.volume) * self._value_per_lot(p.symbol, p.entry)
        for o in self.broker.orders():
            tot += abs(o.volume) * self._value_per_lot(o.symbol, o.price)
        return tot

    def leverage_view(self) -> dict:
        """Account leverage (set by the broker), the leverage the open positions really use, and our cap."""
        cap = float(self.cfg["risk"].get("max_leverage", 0) or 0)
        try:
            acc = self.broker.account()
            exposure = self.exposure()
        except Exception:  # noqa: BLE001
            return {"account": None, "effective": None, "exposure": None, "cap": cap}
        eq = float(acc.get("equity") or 0.0)
        return {"account": acc.get("leverage"), "exposure": round(exposure, 2),
                "effective": round(exposure / eq, 2) if eq > 0 else None, "cap": cap}

    def cap_leverage(self, symbol: str, vol: float, price: float, equity: float) -> tuple[float, str]:
        """Shrink an order so open positions + orders + this one stay within risk.max_leverage x the bot's
        capital (the equity, or the hand-typed capital if smaller)."""
        cap = float(self.cfg["risk"].get("max_leverage", 0) or 0)
        meta = self.metas.get(symbol)
        equity = self.capital(equity)
        if cap <= 0 or equity <= 0 or vol <= 0 or meta is None:
            return vol, ""
        per_lot = self._value_per_lot(symbol, price)
        room = cap * equity - self.exposure()
        if per_lot <= 0 or vol * per_lot <= room:
            return vol, ""
        new = round_volume(max(room, 0.0) / per_lot, meta.volume_step, meta.volume_min, meta.volume_max)
        if new <= 0:
            return 0.0, f"وتو: سقف اهرم واقعی {cap:g}× پر است (حجم باز ربات {self.exposure() / equity:.1f}× اکوییتی)"
        return new, f"حجم به خاطر سقف اهرم واقعی {cap:g}× از {vol:g} به {new:g} لات کم شد"

    # ------------------------------------------------------------------ learning: live vs backtest
    def _live_r(self, strategy: str) -> list[float]:
        since = (self.settings.get("learning_reset") or {}).get(strategy)
        return [float(j["r"]) for j in self.journal
                if j.get("strategy") == strategy and (not since or str(j.get("closed", "")) >= since)]

    def _drift_of(self, strategy: str) -> dict:
        lc = self.cfg.get("learning") or {}
        return brainlib.drift(self.expect.by_style.get(strategy), self._live_r(strategy),
                              int(lc.get("drift_min_trades", 20)), float(lc.get("drift_z", 2.0)))

    def learning_gate(self, strategy: str) -> tuple[bool, str]:
        if not (self.cfg.get("learning") or {}).get("auto_pause_on_drift", True):
            return True, ""
        d = self._drift_of(strategy)
        if d.get("status") == "drift":
            return False, "یادگیری: " + d["text"] + " — از اپ دوباره روشنش کنید تا از نو مقایسه شود"
        return True, ""

    def styles_view(self) -> list[dict]:
        out = []
        for s in self.strategies:
            live = self._live_r(s.name)
            exp = self.expect.by_style.get(s.name) or {}
            sw = {k.split("|", 1)[1]: v for k, v in self.switch.items() if "|" in k and k.split("|", 1)[1] == s.name}
            out.append({"name": s.name, "title": brainlib.title(s.name), "badge": brainlib.verdict(s.name)[0],
                        "switch": sw.get(s.name), "auto_switch": bool(s.p.get("auto_switch")),
                        "live": {"n": len(live), "win": round(float(np.mean([x > 0 for x in live])), 3) if live else None,
                                 "avg_r": round(float(np.mean(live)), 3) if live else None,
                                 "total_r": round(float(np.sum(live)), 2) if live else 0.0},
                        "expected": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in exp.items()},
                        "drift": self._drift_of(s.name)})
        return out

    # ------------------------------------------------------------------ the brain page
    def brain(self) -> dict:
        now = datetime.now(UTC)
        with self.lock:
            out = {"time": now.isoformat(), "market_open": not market_closed(now), "symbols": []}
            for symbol in self.cfg["symbols"]:
                ctx = self.last_ctx.get(symbol)
                entry = {"symbol": symbol, "market": None, "styles": []}
                if ctx is not None:
                    try:
                        entry["market"] = brainlib.market_view(ctx, now)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("brain market view failed: %s", exc)
                for s in self.strategies:
                    info = None
                    if ctx is not None:
                        try:
                            info = s.explain(ctx)
                        except Exception as exc:  # noqa: BLE001
                            info = {"state": "error", "text": f"خطا در توضیح: {exc}"}
                    if info is None and ctx is not None:
                        continue
                    ok1, why1 = self.switch_on(symbol, s.name)
                    ok2, why2 = self.learning_gate(s.name)
                    entry["styles"].append({"name": s.name, "title": brainlib.title(s.name), "enabled": True,
                                            **(info or {"state": "waiting", "text": "منتظر اولین داده از متاتریدر"}),
                                            "gate": {"ok": ok1 and ok2, "why": why1 or why2}})
                out["symbols"].append(entry)
            fs = self.fund.snapshot(self.cfg["symbols"])
            for e in fs.get("upcoming", []):
                e["tehran"] = brainlib.to_tehran(e["time"])
            bo = self.fund.blackout(self.cfg["symbols"][0], now) if self.cfg["symbols"] else None
            out["news"] = {"upcoming": fs.get("upcoming", []), "calendar_ok": fs.get("calendar_ok"),
                           "risk_off": fs.get("risk_off"), "headlines": fs.get("headlines", [])[-5:],
                           "blackout": bo and {"title": bo.get("title"), "cur": bo.get("cur"), "impact": bo.get("impact")}}
            styles = self.styles_view()
            best = self.learner.best_times(min_edge=0.03)
            for st in styles:
                st["enabled"] = True
                st["best_hours"] = [f"{brainlib.tehran_hm(float(b['ny_hour']))} ({b['edge_r']:+.2f}R)"
                                    for b in best if b["strategy"] == st["name"]][:4]
            out["learning"] = {"styles": styles, "learner": {"n_trained": self.learner.n_trained,
                                                             "live_updates": self.learner.n_live_updates}}
            out["summary"] = brainlib.summary_lines(out)
            return out

    # ------------------------------------------------------------------ update from the app
    def start_update(self) -> dict:
        """git pull + requirements + backtest, then exit so start_live.cmd's loop restarts the new code."""
        if self.update_state.get("state") == "running":
            return self.update_state
        self.update_state = {"state": "running", "log": [], "started": datetime.now(UTC).isoformat()}

        def run(cmd: list[str], timeout: int) -> tuple[int, str]:
            try:
                p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
                out = (p.stdout + p.stderr).strip()
                return p.returncode, out[-3000:]
            except Exception as exc:  # noqa: BLE001
                return 1, str(exc)

        def job():
            st = self.update_state
            code, out = run(["git", "pull", "--ff-only"], 180)
            st["log"].append(f"$ git pull\n{out}")
            if code != 0:
                st["state"] = "failed"
                return
            if "Already up to date" in out or "Already up-to-date" in out:
                st["state"] = "up_to_date"
                return
            code, out = run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], 600)
            st["log"].append(f"$ pip install\n{out or 'ok'}")
            code, out = run([sys.executable, "run_backtest.py", "--years", "6"], 1200)
            st["log"].append(f"$ backtest\n{out[-1500:]}")
            st["state"] = "restarting"
            log.warning("update installed; exiting so the start_live loop restarts the new version")
            time.sleep(3)
            os._exit(0)

        threading.Thread(target=job, daemon=True).start()
        return self.update_state

    # ------------------------------------------------------------------ main step
    def _ensure_mt5(self) -> None:
        """Reconnect after the MetaTrader terminal restarts (its old IPC link is dead until initialize())."""
        if self.mt5 is None:
            return
        try:
            alive = self.mt5.terminal_info() is not None
        except Exception:  # noqa: BLE001
            alive = False
        if alive:
            return
        now = time.time()
        if now - getattr(self, "_reconnect_at", 0.0) < 30:
            return
        self._reconnect_at = now
        try:
            self.mt5.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            mt5_connect(self.cfg)
            log.warning("MetaTrader connection was lost; reconnected")
        except Exception as exc:  # noqa: BLE001
            log.error("MetaTrader reconnect failed: %s", exc)

    def step(self) -> None:
        with self.lock:
            self._ensure_mt5()
            self.fund.refresh()
            try:
                self._refresh_switches()
            except Exception as exc:  # noqa: BLE001
                log.exception("auto switch refresh failed: %s", exc)
            self._learn_from_closed()
            for symbol in self.cfg["symbols"]:
                try:
                    self._symbol_step(symbol)
                except Exception as exc:  # noqa: BLE001
                    log.exception("step %s failed: %s", symbol, exc)
            self._manage_positions()
            self._nightly_report()
            self.status = "paused" if self.paused else "running"

    def _nightly_report(self) -> None:
        now = datetime.now(UTC)
        if not self.telegram.report_due(now):
            return
        try:
            self.telegram.send(notify.daily_text(self.journal, self.broker.positions(), self.switch,
                                                 self.broker.account(), now))
        except Exception as exc:  # noqa: BLE001
            log.warning("nightly report failed: %s", exc)
        self.telegram.mark_report(now)

    def _log_decision(self, decision: dict) -> None:
        """Append-only record of every decision (approved or not) with its reasons: state/decisions_log.jsonl"""
        try:
            with open(self.state_dir / "decisions_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(decision, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _symbol_step(self, symbol: str) -> None:
        if symbol not in self.metas:
            self.metas[symbol] = mt5_symbol_meta(self.mt5, symbol)
        meta = self.metas[symbol]
        df, daily = self.frame(symbol)
        if len(df) < 500:
            return
        df = df.iloc[:-1]  # drop the forming bar: decide only on closed bars
        last = df.index[-1]
        if self.last_bar.get(symbol) == last:
            return
        self.last_bar[symbol] = last
        ctx = Context(df, meta, self.cfg, daily)
        self.last_ctx[symbol] = ctx
        sigs = []
        for s in self.strategies:
            try:
                sigs.extend(s.generate(ctx))
            except Exception as exc:  # noqa: BLE001
                log.exception("strategy %s failed: %s", s.name, exc)
        rows = annotate(sigs, ctx, int(self.cfg["fusion"]["confluence_window_min"]))
        fresh = [r for r in rows if r["bar"] == len(df) - 1]  # signals decided on the bar that just closed
        sig_by_key = {(s.strategy, s.bar): s for s in sigs}
        for row in fresh:
            key = f"{symbol}|{row['strategy']}|{row['time']}"
            if key in self.seen:
                continue
            self.seen.add(key)
            self._decide(row, sig_by_key[(row["strategy"], row["bar"])], meta)

    def _decide(self, row: dict, sig, meta) -> None:
        now = datetime.now(UTC)
        sym = row["symbol"]
        bo = self.fund.blackout(sym, now)
        nr, soon = self.fund.event_risk(sym, now)
        fb = self.fund.bias(sym, now)
        row.update({
            "news_blackout": bo and {"title": bo["title"], "cur": bo["cur"], "impact": bo["impact"]},
            "news_risk": nr, "fund_align": fb * row["direction"],
            "news_unknown": self.fund.enabled and not self.fund.calendar_ok,
            "risk_off": self.fund.risk_off_level(now),
        })
        row["prob"] = self.learner.prob(row)
        row["ev"] = self.learner.expected_r(row)
        row["hour_edge"] = self.learner.edges["hour"].get(f"{row['strategy']}|{row['ny_hour']}", 0.0)
        acc = self.account_state(sym, row["strategy"])
        verdict = self.devil.review(row, acc)
        if self.paused:
            verdict.approved = False
            verdict.vetoes.append("ربات از اپ متوقف شده")
        ok, why = self.switch_on(sym, row["strategy"])
        if not ok:
            verdict.approved = False
            verdict.vetoes.append(why)
        ok, why = self.learning_gate(row["strategy"])
        if not ok:
            verdict.approved = False
            verdict.vetoes.append(why)
        decision = {"time": now.isoformat(), "symbol": sym, "strategy": row["strategy"],
                    "direction": row["direction"], "entry": row["entry"], "sl": row["sl"], "tp": row["tp"],
                    "prob": round(row["prob"], 3), "ev": round(row["ev"], 3), "approved": verdict.approved,
                    "debate": verdict.summary(), "ticket": None}
        if verdict.approved:
            sym_risk = for_symbol(self.cfg, sym)["risk"]
            risk_money, size_notes = self.trade_risk(float(acc["balance"]), float(sym_risk["risk_per_trade_pct"]))
            if self.mt5 is not None:
                try:  # fresh tick value (it moves for pairs whose quote currency isn't the account currency)
                    meta = mt5_symbol_meta(self.mt5, sym)
                    self.metas[sym] = meta
                except Exception as exc:  # noqa: BLE001
                    log.warning("symbol_info refresh failed: %s", exc)
            sl_dist = abs(row["entry"] - row["sl"])
            if sig.order == "market":
                try:
                    tk = self.broker.tick(sym)
                    sl_dist = abs((tk.ask if sig.direction == 1 else tk.bid) - sig.sl)
                except Exception:  # noqa: BLE001
                    pass
            vol = lot_size(risk_money, 100.0, sl_dist, meta.tick_size,  # lots that lose risk_money at the stop
                           meta.tick_value, meta.volume_step, meta.volume_min, meta.volume_max)
            note = "\n".join(size_notes)
            if vol > 0:
                vol, lev_note = self.cap_leverage(sym, vol, float(row["entry"]), float(acc.get("equity") or acc["balance"]))
                note = "\n".join(x for x in (note, lev_note) if x)
            if vol <= 0:
                decision["approved"] = False
                decision["debate"] += "\n" + (note if "وتو" in note else
                                              f"وتو: با {risk_money:,.2f} دلار ریسک، حجم کمتر از حداقل لات بروکر می‌شود")
            else:
                comment = f"asli {row['strategy']}"
                try:
                    if sig.order == "limit":
                        # MT5 reads the expiration in broker SERVER time, not UTC
                        exp = utc_to_server(now + timedelta(minutes=5 * sig.valid_bars), self.time_mode)
                        t = self.broker.limit(sym, sig.direction, vol, round(sig.entry, meta.digits),
                                              round(sig.sl, meta.digits), round(sig.tp, meta.digits), exp, comment)
                    else:
                        t = self.broker.market(sym, sig.direction, vol, round(sig.sl, meta.digits),
                                               round(sig.tp, meta.digits), comment)
                    decision["ticket"] = t
                    if note:
                        decision["debate"] += "\n" + note
                    self.trade_meta[str(t)] = {**{k: v for k, v in row.items() if k not in ("time",)},
                                               "opened": now.isoformat(), "volume": vol,
                                               "risk_money": risk_money,
                                               "tp1": sig.tp1, "tp1_done": False, "order": sig.order,
                                               "max_hold_bars": sig.max_hold_bars,
                                               "valid_until": (now + timedelta(minutes=5 * sig.valid_bars)).isoformat()}
                    self._save("trade_meta.json", self.trade_meta)
                    log.info("OPEN %s %s %s vol=%s ticket=%s", sym, row["strategy"], row["direction"], vol, t)
                    self.telegram.send(notify.open_text(sym, row, vol, risk_money))
                except Exception as exc:  # noqa: BLE001
                    decision["approved"] = False
                    decision["debate"] += f"\nخطای ارسال سفارش: {exc}"
                    log.error("order failed: %s", exc)
        self.decisions.appendleft(decision)
        self._save("decisions.json", list(self.decisions))
        self._log_decision(decision)

    # ------------------------------------------------------------------ management
    def _manage_positions(self) -> None:
        for p in self.broker.positions():
            m = self.trade_meta.get(str(p.ticket))
            if not m:
                continue
            tp1 = m.get("tp1")
            if tp1 and not m.get("tp1_done"):
                t = self.broker.tick(p.symbol)
                px = t.bid if p.direction == 1 else t.ask
                if (px - tp1) * p.direction >= 0:
                    meta = self.metas.get(p.symbol)
                    half = (round_volume(p.volume / 2, meta.volume_step, meta.volume_min, meta.volume_max)
                            if meta else round(p.volume / 2, 2))
                    try:
                        if 0 < half < p.volume:
                            self.broker.close(p, half)
                        # mark done right after the close so a failing SL change can never close half again
                        m["tp1_done"] = True
                        m["be_pending"] = True
                        self._save("trade_meta.json", self.trade_meta)
                    except Exception as exc:  # noqa: BLE001
                        log.error("partial close failed: %s", exc)
                    continue  # positions changed; re-read them next loop
            if m.get("be_pending"):
                try:
                    if (p.sl - p.entry) * p.direction < 0:
                        self.broker.modify_sl(p, p.entry)
                    m["be_pending"] = False
                    self._save("trade_meta.json", self.trade_meta)
                except Exception as exc:  # noqa: BLE001
                    log.warning("break-even move failed, will retry: %s", exc)
            # time stop — same holding limit as in the backtest
            opened = datetime.fromisoformat(m["opened"])
            hold = int(m.get("max_hold_bars", 288))
            if datetime.now(UTC) - opened > timedelta(minutes=5 * hold):
                try:
                    self.broker.close(p)
                except Exception as exc:  # noqa: BLE001
                    log.error("time-stop close failed: %s", exc)

    def _learn_from_closed(self) -> None:
        open_tickets = {str(p.ticket) for p in self.broker.positions()} | {str(o.ticket) for o in self.broker.orders()}
        try:
            deals = self.broker.closed_deals(self._deals_since)
        except Exception as exc:  # noqa: BLE001
            log.warning("history unavailable: %s", exc)
            return
        by_pos: dict[str, float] = {}
        for d in deals:
            by_pos[str(d["position"])] = by_pos.get(str(d["position"]), 0.0) + d["profit"]
        changed = False
        now = datetime.now(UTC)
        for tk, m in list(self.trade_meta.items()):
            if tk in open_tickets:
                continue
            if tk not in by_pos:
                # a limit order that expired / was cancelled without filling: forget it
                vu = m.get("valid_until")
                if m.get("order") == "limit" and vu and now > datetime.fromisoformat(vu) + timedelta(hours=1):
                    del self.trade_meta[tk]
                    changed = True
                continue
            profit = by_pos[tk]
            r = profit / m["risk_money"] if m.get("risk_money") else (1.0 if profit > 0 else -1.0)
            row = {**m, "r": r}
            self.learner.update(row, won=profit > 0)
            self.loss_streak = 0 if profit > 0 else self.loss_streak + 1
            self.journal.append({"ticket": tk, "symbol": m["symbol"], "strategy": m["strategy"],
                                 "direction": m["direction"], "opened": m["opened"],
                                 "closed": datetime.now(UTC).isoformat(), "profit": round(profit, 2),
                                 "r": round(r, 2), "prob": m.get("prob")})
            del self.trade_meta[tk]
            changed = True
            log.info("CLOSED %s %s profit=%.2f r=%.2f — learner updated", m["symbol"], m["strategy"], profit, r)
            self.telegram.send(notify.close_text(self.journal[-1], sum(j["profit"] for j in self.journal),
                                                 len(self.journal)))
        if changed:
            self.learner.save(self.state_dir / "learner.json")
            self._save("trade_meta.json", self.trade_meta)
            self._save("journal.json", self.journal[-2000:])

    # ------------------------------------------------------------------ dashboard data
    def snapshot(self) -> dict:
        with self.lock:
            acc = self.broker.account()
            pos = [{"ticket": p.ticket, "symbol": p.symbol, "direction": p.direction, "volume": p.volume,
                    "entry": p.entry, "sl": p.sl, "tp": p.tp, "strategy": self.trade_meta.get(str(p.ticket), {}).get("strategy")}
                   for p in self.broker.positions()]
            j = self.journal[-500:]
            wins = sum(1 for x in j if x["profit"] > 0)
            return {
                "status": self.status, "mode": self.cfg["mode"], "paused": self.paused, "account": acc,
                "positions": pos, "decisions": list(self.decisions)[:50], "journal": j[-50:][::-1],
                "live_stats": {"trades": len(j), "win_rate": wins / len(j) if j else 0.0,
                               "total_r": round(sum(x["r"] for x in j), 2), "profit": round(sum(x["profit"] for x in j), 2)},
                "learner": {"n_trained": self.learner.n_trained, "live_updates": self.learner.n_live_updates,
                            "top": self.learner.top_weights(6), "best_times": self.learner.best_times()[:10]},
                "fundamental": self.fund.snapshot(self.cfg["symbols"]),
                "auto_switch": {k: v for k, v in self.switch.items() if not k.startswith("_")},
                "styles": self.styles_view(), "leverage": self.leverage_view(),
                "risk": {**{k: self.cfg["risk"][k] for k in SETTING_BOUNDS},
                         **{k: self.cfg["risk"].get(k, "auto") for k in MODE_KEYS}},
                "sizing": self.sizing_view(),
                "today": {"start_balance": self.day_start_balance, "pl": round(self.own_day_pl(), 2)},
                "market_open": not market_closed(datetime.now(UTC)),
                "time": datetime.now(UTC).isoformat(),
            }

    def run_forever(self, poll_sec: float = 10.0) -> None:
        log.info("engine started: symbols=%s mode=%s", self.cfg["symbols"], self.cfg["mode"])
        while True:
            t0 = time.time()
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                self.status = f"error: {exc}"
                log.exception("step failed")
            time.sleep(max(1.0, poll_sec - (time.time() - t0)))
