"""Live engine: every closed M5 bar, run all five styles on recent data, score new candidates with the
learner, attach fundamentals, let the opposing agent decide, size the position, send the order, manage
open positions (50% partial + break-even), and learn from every closed trade.

State (positions meta, decisions, journal) is kept in state/*.json so a restart loses nothing.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .broker import lot_size, round_volume
from .data import mt5_recent, mt5_symbol_meta
from .devil import DevilsAdvocate
from .fundamental import Fundamentals
from .fusion import annotate
from .learner import Learner
from .strategies import Context, build
from .timeutil import NY, trading_day

log = logging.getLogger("asli.live")
UTC = timezone.utc
BARS = 3000  # ~10 trading days of M5


def utc_to_server(t: datetime, mode: str) -> datetime:
    """Inverse of timeutil.server_naive_to_utc: express a UTC instant on the broker's server clock.
    Returned as a tz-aware 'UTC' datetime whose wall clock equals the server clock, because MT5's Python API
    interprets epoch seconds as server time."""
    t = t.astimezone(UTC)
    if mode == "utc":
        return t
    if mode.startswith("fixed:"):
        return t + timedelta(hours=float(mode.split(":", 1)[1]))
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

    def account_state(self, symbol: str) -> dict:
        acc = self.broker.account()
        self._roll_day(acc)
        start = self.day_start_balance or acc["balance"]
        loss_pct = max(0.0, (start - acc["equity"]) / start * 100) if start else 0.0
        pos = self.broker.positions()
        return {"daily_loss_pct": loss_pct, "open_positions": len(pos) + len(self.broker.orders()),
                "same_symbol_open": any(p.symbol == symbol for p in pos) or
                any(o.symbol == symbol for o in self.broker.orders()),
                "loss_streak": self.loss_streak, "balance": acc["balance"], "equity": acc["equity"]}

    # ------------------------------------------------------------------ main step
    def step(self) -> None:
        with self.lock:
            self.fund.refresh()
            self._learn_from_closed()
            for symbol in self.cfg["symbols"]:
                try:
                    self._symbol_step(symbol)
                except Exception as exc:  # noqa: BLE001
                    log.exception("step %s failed: %s", symbol, exc)
            self._manage_positions()
            self.status = "paused" if self.paused else "running"

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
        acc = self.account_state(sym)
        verdict = self.devil.review(row, acc)
        if self.paused:
            verdict.approved = False
            verdict.vetoes.append("ربات از اپ متوقف شده")
        decision = {"time": now.isoformat(), "symbol": sym, "strategy": row["strategy"],
                    "direction": row["direction"], "entry": row["entry"], "sl": row["sl"], "tp": row["tp"],
                    "prob": round(row["prob"], 3), "ev": round(row["ev"], 3), "approved": verdict.approved,
                    "debate": verdict.summary(), "ticket": None}
        if verdict.approved:
            risk_pct = float(self.cfg["risk"]["risk_per_trade_pct"])
            if self.loss_streak >= int(self.cfg["risk"]["halve_after_losses"]):
                risk_pct /= 2
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
            vol = lot_size(acc["balance"], risk_pct, sl_dist, meta.tick_size,
                           meta.tick_value, meta.volume_step, meta.volume_min, meta.volume_max)
            if vol <= 0:
                decision["approved"] = False
                decision["debate"] += "\nوتو: حجم محاسبه‌شده کمتر از حداقل لات بروکر است"
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
                    self.trade_meta[str(t)] = {**{k: v for k, v in row.items() if k not in ("time",)},
                                               "opened": now.isoformat(), "volume": vol,
                                               "risk_money": acc["balance"] * risk_pct / 100,
                                               "tp1": sig.tp1, "tp1_done": False, "order": sig.order,
                                               "max_hold_bars": sig.max_hold_bars,
                                               "valid_until": (now + timedelta(minutes=5 * sig.valid_bars)).isoformat()}
                    self._save("trade_meta.json", self.trade_meta)
                    log.info("OPEN %s %s %s vol=%s ticket=%s", sym, row["strategy"], row["direction"], vol, t)
                except Exception as exc:  # noqa: BLE001
                    decision["approved"] = False
                    decision["debate"] += f"\nخطای ارسال سفارش: {exc}"
                    log.error("order failed: %s", exc)
        self.decisions.appendleft(decision)
        self._save("decisions.json", list(self.decisions))

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
