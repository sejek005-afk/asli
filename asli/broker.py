"""Order execution: MetaTrader 5 broker and an in-memory paper broker with the same interface.

Safety: MT5Broker refuses to trade a REAL account unless cfg mode == "real" AND allow_real == true.
With mode "demo" it refuses to trade anything that is not a demo account.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger("asli.broker")
UTC = timezone.utc


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: int
    volume: float
    entry: float
    sl: float
    tp: float
    opened: datetime
    comment: str = ""
    tp1: float | None = None
    tp1_done: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class PendingOrder:
    ticket: int
    symbol: str
    direction: int
    volume: float
    price: float
    sl: float
    tp: float
    expires: datetime
    comment: str = ""
    meta: dict = field(default_factory=dict)


def round_volume(v: float, step: float, vmin: float, vmax: float) -> float:
    v = math.floor(v / step) * step
    return round(min(max(v, 0.0), vmax), 8) if v >= vmin else 0.0


def lot_size(balance: float, risk_pct: float, sl_dist: float, tick_size: float, tick_value: float,
             step: float, vmin: float, vmax: float) -> float:
    """Lots so that hitting the stop loses risk_pct of balance."""
    if sl_dist <= 0 or tick_size <= 0 or tick_value <= 0:
        return 0.0
    loss_per_lot = sl_dist / tick_size * tick_value
    return round_volume(balance * risk_pct / 100.0 / loss_per_lot, step, vmin, vmax)


# ============================================================================ MT5

class MT5Broker:
    def __init__(self, mt5, cfg: dict):
        self.mt5 = mt5
        self.cfg = cfg
        self.magic = int(cfg["mt5"]["magic"])
        self.dev = int(cfg["mt5"]["deviation_points"])
        self.check_account_mode()

    def check_account_mode(self) -> None:
        info = self.mt5.account_info()
        if info is None:
            raise RuntimeError(f"MT5 account_info failed: {self.mt5.last_error()}")
        demo = info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO
        mode = self.cfg["mode"]
        if not demo and not (mode == "real" and self.cfg.get("allow_real") is True):
            raise RuntimeError("حساب واقعی است ولی mode=real و allow_real=true تنظیم نشده؛ ربات معامله نمی‌کند")
        if demo and mode == "real":
            log.warning("mode=real but the connected account is DEMO")
        self.is_demo = demo
        log.info("MT5 account %s (%s) balance=%.2f %s", info.login, "DEMO" if demo else "REAL", info.balance,
                 info.currency)

    # ------------------------------------------------------------------ account
    def account(self) -> dict:
        a = self.mt5.account_info()
        return {"login": a.login, "balance": a.balance, "equity": a.equity, "currency": a.currency,
                "demo": a.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO, "leverage": a.leverage,
                "margin_free": a.margin_free, "server": a.server}

    def positions(self) -> list[Position]:
        out = []
        for p in self.mt5.positions_get() or []:
            if p.magic != self.magic:
                continue
            out.append(Position(p.ticket, p.symbol, 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1, p.volume,
                                p.price_open, p.sl, p.tp, datetime.fromtimestamp(p.time, UTC), p.comment))
        return out

    def orders(self) -> list[PendingOrder]:
        out = []
        for o in self.mt5.orders_get() or []:
            if o.magic != self.magic:
                continue
            d = 1 if o.type in (self.mt5.ORDER_TYPE_BUY_LIMIT, self.mt5.ORDER_TYPE_BUY_STOP) else -1
            out.append(PendingOrder(o.ticket, o.symbol, d, o.volume_current, o.price_open, o.sl, o.tp,
                                    datetime.fromtimestamp(o.time_expiration or 0, UTC), o.comment))
        return out

    def tick(self, symbol: str):
        return self.mt5.symbol_info_tick(symbol)

    # ------------------------------------------------------------------ orders
    def _filling(self, symbol: str):
        si = self.mt5.symbol_info(symbol)
        fm = getattr(si, "filling_mode", 0)
        if fm & 1:
            return self.mt5.ORDER_FILLING_FOK
        if fm & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, req: dict):
        # re-check the account on every order: someone may have switched the terminal to another account
        info = self.mt5.account_info()
        demo = info is not None and info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO
        if info is None or (not demo and not (self.cfg["mode"] == "real" and self.cfg.get("allow_real") is True)):
            raise RuntimeError("حساب متصل دمو نیست و معامله‌ی واقعی مجاز نشده؛ سفارش ارسال نشد")
        res = self.mt5.order_send(req)
        if res is None or res.retcode not in (self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED,
                                               self.mt5.TRADE_RETCODE_DONE_PARTIAL):
            raise RuntimeError(f"order_send failed: {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')} "
                               f"{self.mt5.last_error()}")
        return res

    def market(self, symbol: str, direction: int, volume: float, sl: float, tp: float, comment: str) -> int:
        t = self.tick(symbol)
        price = t.ask if direction == 1 else t.bid
        req = {"action": self.mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
               "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
               "price": price, "sl": sl, "tp": tp, "deviation": self.dev, "magic": self.magic,
               "comment": comment[:31], "type_time": self.mt5.ORDER_TIME_GTC,
               "type_filling": self._filling(symbol)}
        return int(self._send(req).order)

    def limit(self, symbol: str, direction: int, volume: float, price: float, sl: float, tp: float,
              expires: datetime, comment: str) -> int:
        req = {"action": self.mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": volume,
               "type": self.mt5.ORDER_TYPE_BUY_LIMIT if direction == 1 else self.mt5.ORDER_TYPE_SELL_LIMIT,
               "price": price, "sl": sl, "tp": tp, "deviation": self.dev, "magic": self.magic,
               "comment": comment[:31], "type_time": self.mt5.ORDER_TIME_SPECIFIED,
               "expiration": int(expires.timestamp()), "type_filling": self.mt5.ORDER_FILLING_RETURN}
        return int(self._send(req).order)

    def modify_sl(self, pos: Position, sl: float) -> None:
        self._send({"action": self.mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": pos.ticket,
                    "sl": sl, "tp": pos.tp, "magic": self.magic})

    def close(self, pos: Position, volume: float | None = None) -> None:
        t = self.tick(pos.symbol)
        vol = volume or pos.volume
        self._send({"action": self.mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": vol,
                    "type": self.mt5.ORDER_TYPE_SELL if pos.direction == 1 else self.mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket, "price": t.bid if pos.direction == 1 else t.ask,
                    "deviation": self.dev, "magic": self.magic, "comment": "asli close",
                    "type_filling": self._filling(pos.symbol)})

    def cancel(self, order: PendingOrder) -> None:
        self._send({"action": self.mt5.TRADE_ACTION_REMOVE, "order": order.ticket})

    def closed_deals(self, since: datetime) -> list[dict]:
        # date_to +1 day: deal times are on the server clock, which runs ahead of UTC
        deals = self.mt5.history_deals_get(since, datetime.now(UTC) + timedelta(days=1)) or []
        out = []
        for d in deals:
            if d.magic != self.magic or d.entry != self.mt5.DEAL_ENTRY_OUT:
                continue
            out.append({"ticket": d.ticket, "position": d.position_id, "symbol": d.symbol,
                        "profit": d.profit + d.commission + d.swap, "price": d.price, "volume": d.volume,
                        "time": datetime.fromtimestamp(d.time, UTC).isoformat()})
        return out


# ============================================================================ paper

class PaperBroker:
    """Simulated broker used by `--paper` mode and by tests. Fills on given bid/ask quotes."""

    def __init__(self, balance: float = 10000.0):
        self.balance = balance
        self.equity = balance
        self._pos: list[Position] = []
        self._ord: list[PendingOrder] = []
        self._next = 1
        self.quotes: dict[str, tuple[float, float]] = {}
        self.history: list[dict] = []
        self.tick_value = 1.0
        self.tick_size = 0.00001
        self.is_demo = True

    def account(self) -> dict:
        return {"login": 0, "balance": self.balance, "equity": self.equity, "currency": "USD", "demo": True,
                "leverage": 100, "margin_free": self.equity, "server": "paper"}

    def positions(self):
        return list(self._pos)

    def orders(self):
        return list(self._ord)

    def set_quote(self, symbol: str, bid: float, ask: float, now: datetime) -> None:
        self.quotes[symbol] = (bid, ask)
        for o in list(self._ord):
            if o.symbol != symbol:
                continue
            if now > o.expires:
                self._ord.remove(o)
                continue
            if (o.direction == 1 and ask <= o.price) or (o.direction == -1 and bid >= o.price):
                self._ord.remove(o)
                self._pos.append(Position(o.ticket, symbol, o.direction, o.volume, o.price, o.sl, o.tp, now,
                                          o.comment, meta=o.meta))
        for p in list(self._pos):
            if p.symbol != symbol:
                continue
            px = bid if p.direction == 1 else ask
            if (p.direction == 1 and (bid <= p.sl or bid >= p.tp)) or (p.direction == -1 and (ask >= p.sl or ask <= p.tp)):
                self._realize(p, p.sl if (p.direction == 1 and bid <= p.sl) or (p.direction == -1 and ask >= p.sl)
                              else p.tp, p.volume, now)
        self.equity = self.balance + sum(((self.quotes[p.symbol][0] if p.direction == 1 else self.quotes[p.symbol][1])
                                          - p.entry) * p.direction * p.volume / self.tick_size * self.tick_value
                                         for p in self._pos if p.symbol in self.quotes)

    def _realize(self, p: Position, price: float, vol: float, now: datetime) -> None:
        pnl = (price - p.entry) * p.direction * vol / self.tick_size * self.tick_value
        self.balance += pnl
        self.history.append({"ticket": self._next, "position": p.ticket, "symbol": p.symbol, "profit": pnl,
                             "price": price, "volume": vol, "time": now.isoformat()})
        self._next += 1
        if vol >= p.volume - 1e-9:
            self._pos.remove(p)
        else:
            p.volume = round(p.volume - vol, 8)

    def tick(self, symbol):
        b, a = self.quotes[symbol]

        class T:
            bid, ask = b, a
        return T

    def market(self, symbol, direction, volume, sl, tp, comment) -> int:
        b, a = self.quotes[symbol]
        t = self._next; self._next += 1
        self._pos.append(Position(t, symbol, direction, volume, a if direction == 1 else b, sl, tp,
                                  datetime.now(UTC), comment))
        return t

    def limit(self, symbol, direction, volume, price, sl, tp, expires, comment) -> int:
        t = self._next; self._next += 1
        self._ord.append(PendingOrder(t, symbol, direction, volume, price, sl, tp, expires, comment))
        return t

    def modify_sl(self, pos, sl):
        pos.sl = sl

    def close(self, pos, volume=None):
        b, a = self.quotes[pos.symbol]
        self._realize(pos, b if pos.direction == 1 else a, volume or pos.volume, datetime.now(UTC))

    def cancel(self, order):
        self._ord.remove(order)

    def closed_deals(self, since):
        return [h for h in self.history if h["time"] >= since.isoformat()]
