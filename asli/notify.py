"""Telegram reports: a message when ASLI opens or closes a trade and one nightly summary.

Set-up (once): create a bot with @BotFather, paste its token in the app (Settings -> Telegram), send any
message to the bot from your own Telegram, then press "test" in the app — the bot remembers that chat.
The token lives in state/settings.json on the server; it is never shown again by the API.

Sending never blocks trading: messages go through a small background queue with short timeouts.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("asli.notify")
UTC = timezone.utc
TEHRAN = ZoneInfo("Asia/Tehran")
TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
STYLE_FA = {"nyorb": "رنج نیویورک", "donch": "شکست کانال ساعتی", "scalp": "اسکلپ"}


def style_fa(name: str) -> str:
    return STYLE_FA.get(name, name)


def _teh(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(TEHRAN).strftime("%H:%M")


def open_text(symbol: str, row: dict, vol: float, risk_money: float) -> str:
    side = "خرید" if row["direction"] == 1 else "فروش"
    return (f"🔵 معامله‌ی تازه — {style_fa(row['strategy'])}\n{side} {symbol}، حجم {vol:g} لات\n"
            f"ورود {row['entry']:,.2f} | حد ضرر {row['sl']:,.2f} | حد سود {row['tp']:,.2f}\n"
            f"ریسک این معامله: {risk_money:,.2f} دلار")


def close_text(j: dict, total: float, n: int) -> str:
    icon = "🟢" if j["profit"] > 0 else "🔴"
    side = "خرید" if j["direction"] == 1 else "فروش"
    return (f"{icon} معامله بسته شد — {style_fa(j['strategy'])}\n{side} {j['symbol']}: {j['profit']:+,.2f} دلار "
            f"({j['r']:+.2f}R)\nجمع ربات تا الان: {total:+,.2f} دلار از {n} معامله")


def daily_text(journal: list[dict], positions: list, switch: dict, account: dict, now: datetime) -> str:
    since = now - timedelta(hours=24)
    day = [j for j in journal if j.get("closed") and datetime.fromisoformat(j["closed"]) >= since]
    lines = [f"📊 گزارش شبانه‌ی ASLI — {now.astimezone(TEHRAN).strftime('%Y/%m/%d %H:%M')} تهران", ""]
    if day:
        lines.append(f"۲۴ ساعت اخیر: {len(day)} معامله، {sum(j['profit'] for j in day):+,.2f} دلار")
        for j in day:
            side = "خرید" if j["direction"] == 1 else "فروش"
            lines.append(f"  {'🟢' if j['profit'] > 0 else '🔴'} {style_fa(j['strategy'])} · {side} · بسته {_teh(j['closed'])} · "
                         f"{j['profit']:+,.2f} دلار ({j['r']:+.2f}R)")
    else:
        lines.append("۲۴ ساعت اخیر: معامله‌ای بسته نشد")
    lines.append("")
    if journal:
        wins = sum(1 for j in journal if j["profit"] > 0)
        lines.append(f"از شروع: {len(journal)} معامله، {wins} سود و {len(journal) - wins} ضرر، "
                     f"{sum(j['profit'] for j in journal):+,.2f} دلار ({sum(j['r'] for j in journal):+.2f}R)")
        by: dict[str, list] = {}
        for j in journal:
            by.setdefault(j["strategy"], []).append(j)
        for name, js in by.items():
            lines.append(f"  {style_fa(name)}: {len(js)} معامله، {sum(j['profit'] for j in js):+,.2f} دلار")
    else:
        lines.append("از شروع: هنوز معامله‌ای بسته نشده")
    lines.append("")
    if positions:
        lines.append(f"پوزیشن‌های باز: {len(positions)}")
        for p in positions:
            side = "خرید" if p.direction == 1 else "فروش"
            lines.append(f"  {side} {p.symbol} {p.volume:g} لات از {p.entry:,.2f}")
    else:
        lines.append("پوزیشن باز: ندارد")
    off = [k.split("|")[1] for k, v in switch.items() if not k.startswith("_") and isinstance(v, dict) and not v.get("on")]
    if off:
        lines.append("کلید خودکار خاموش: " + "، ".join(style_fa(x) for x in off))
    if account:
        lines.append(f"موجودی حساب: {account.get('balance', 0):,.2f} دلار")
    return "\n".join(lines)


class Telegram:
    def __init__(self, cfg: dict, state_dir, settings: dict | None = None):
        self.cfg = (cfg.get("telegram") or {})
        self.path = Path(state_dir) / "telegram.json"
        try:
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.state = {}
        self.token = str((settings or {}).get("telegram_token") or self.cfg.get("bot_token") or "")
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self._thread = None

    # ------------------------------------------------------------------ state
    @property
    def chat_id(self):
        return self.cfg.get("chat_id") or self.state.get("chat_id")

    @property
    def ready(self) -> bool:
        return bool(self.token and self.chat_id)

    def view(self) -> dict:
        return {"configured": bool(self.token), "chat": bool(self.chat_id), "chat_name": self.state.get("chat_name"),
                "token_hint": ("…" + self.token[-4:]) if self.token else "", "last_report": self.state.get("last_report"),
                "report_time": self.report_time, "last_error": self.state.get("last_error")}

    @property
    def report_time(self) -> str:
        return str(self.cfg.get("report_time", "23:30"))

    def set_token(self, token: str) -> bool:
        token = (token or "").strip()
        if token and not TOKEN_RE.match(token):
            return False
        if token != self.token:
            self.token = token
            self.state.pop("chat_id", None)
            self.state.pop("chat_name", None)
            self._save()
        return True

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ transport
    def _call(self, method: str, payload: dict, timeout: float = 10.0) -> dict:
        import requests
        proxy = self.cfg.get("proxy")
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=payload, timeout=timeout,
                          proxies={"https": proxy, "http": proxy} if proxy else None)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(str(data.get("description") or r.status_code))
        return data

    def learn_chat(self) -> bool:
        """The chat of the newest message anyone sent to the bot becomes the report chat."""
        ups = self._call("getUpdates", {"timeout": 0, "allowed_updates": ["message"]}).get("result", [])
        msgs = [u["message"] for u in ups if u.get("message", {}).get("chat", {}).get("id")]
        if not msgs:
            return False
        chat = msgs[-1]["chat"]
        self.state["chat_id"] = chat["id"]
        self.state["chat_name"] = chat.get("first_name") or chat.get("title") or chat.get("username")
        self._save()
        return True

    def test(self) -> dict:
        """Used by the app's button: find the chat if needed and send a test message, synchronously."""
        if not self.token:
            return {"ok": False, "message": "اول توکن ربات تلگرام را وارد و ذخیره کنید"}
        try:
            if not self.chat_id and not self.learn_chat():
                return {"ok": False, "message": "توکن درست است، ولی هنوز پیامی به ربات نرسیده. در تلگرام به ربات خودتان "
                                                "یک پیام (مثلاً «سلام») بفرستید و دوباره همین دکمه را بزنید"}
            self._call("sendMessage", {"chat_id": self.chat_id, "text": "✅ ASLI به تلگرام وصل شد. از این به بعد باز و "
                                       f"بسته شدن هر معامله و گزارش شبانه (ساعت {self.report_time} تهران) همین‌جا می‌آید."})
            self.state.pop("last_error", None)
            self._save()
            return {"ok": True, "message": "پیام آزمایشی فرستاده شد ✔"}
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if self.token in msg:
                msg = msg.replace(self.token, "<token>")
            return {"ok": False, "message": f"تلگرام جواب نداد: {msg[:200]}"}

    def send(self, text: str) -> None:
        """Queue a message; never raises, never blocks."""
        if not self.ready:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                self._call("sendMessage", {"chat_id": self.chat_id, "text": text[:4000]})
                if self.state.pop("last_error", None):
                    self._save()
            except Exception as exc:  # noqa: BLE001
                err = str(exc).replace(self.token, "<token>")[:200]
                if self.state.get("last_error") != err:
                    self.state["last_error"] = err
                    self._save()
                    log.warning("telegram send failed: %s", err)

    # ------------------------------------------------------------------ nightly report
    def report_due(self, now: datetime) -> bool:
        if not self.ready:
            return False
        local = now.astimezone(TEHRAN)
        hh, mm = (int(x) for x in self.report_time.split(":"))
        return (local.hour, local.minute) >= (hh, mm) and self.state.get("last_report") != str(local.date())

    def mark_report(self, now: datetime) -> None:
        self.state["last_report"] = str(now.astimezone(TEHRAN).date())
        self._save()
