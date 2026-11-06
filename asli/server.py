"""Small HTTP API + web dashboard (standard library only). The Android app talks to this.

All endpoints except "/" need the token from config server.token, sent as header
"Authorization: Bearer <token>" or query "?token=<token>".
GET  /api/status        everything the app shows
GET  /api/report        latest research summary (json)
GET  /report            latest research report (html)
POST /api/pause         stop opening new trades
POST /api/resume        allow new trades again
POST /api/close_all     close every ASLI position and cancel orders
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("asli.server")

DASH = """<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ASLI</title>
<style>body{font-family:Tahoma,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:16px}
input,button{font:inherit;padding:8px;border-radius:8px;border:1px solid #333;background:#1c2230;color:#eee}
.card{background:#161a22;border-radius:12px;padding:12px;margin:10px 0}pre{white-space:pre-wrap}
table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #2a2f3a;padding:4px;text-align:center}
.ok{color:#5fd47a}.bad{color:#ff6b6b}</style></head><body>
<h2>ASLI</h2><div><input id="t" placeholder="توکن" size="24"><button onclick="save()">ورود</button>
<button onclick="post('pause')">توقف</button><button onclick="post('resume')">ادامه</button>
<button onclick="if(confirm('همه بسته شود؟'))post('close_all')">بستن همه</button>
<a href="#" onclick="location='/report?token='+tok();return false" style="color:#8ab4ff">گزارش بک‌تست</a></div>
<div id="out" class="card">...</div>
<script>
function tok(){try{return localStorage.getItem('t')||''}catch(e){return ''}}
function save(){try{localStorage.setItem('t',document.getElementById('t').value)}catch(e){};load()}
async function post(a){await fetch('/api/'+a+'?token='+tok(),{method:'POST'});load()}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){const r=await fetch('/api/status?token='+tok());if(!r.ok){out.textContent='توکن اشتباه';return}
const s=await r.json();const a=s.account;let h=`<b>وضعیت:</b> ${s.status} — ${s.mode}<br><b>بالانس:</b> ${a.balance} ${a.currency} | <b>اکوییتی:</b> ${a.equity}
<br><b>آمار زنده:</b> ${s.live_stats.trades} معامله، وین‌ریت ${(s.live_stats.win_rate*100).toFixed(0)}٪، ${s.live_stats.total_r}R`;
h+='<h3>پوزیشن‌ها</h3><table><tr><th>نماد</th><th>جهت</th><th>سبک</th><th>حجم</th><th>ورود</th><th>SL</th><th>TP</th></tr>'+s.positions.map(p=>`<tr><td>${p.symbol}</td><td>${p.direction>0?'خرید':'فروش'}</td><td>${p.strategy||''}</td><td>${p.volume}</td><td>${p.entry}</td><td>${p.sl}</td><td>${p.tp}</td></tr>`).join('')+'</table>';
h+='<h3>تصمیم‌های ایجنت مخالف</h3>'+s.decisions.slice(0,15).map(d=>`<div class="card"><b class="${d.approved?'ok':'bad'}">${d.symbol} ${d.strategy} ${d.direction>0?'خرید':'فروش'}</b> ${d.time.slice(0,16)}<pre>${esc(d.debate)}</pre></div>`).join('');
h+='<h3>تقویم اقتصادی (۲۴ ساعت)</h3>'+s.fundamental.upcoming.map(e=>`${e.time.slice(5,16)} ${e.cur} ${e.impact} — ${esc(e.title)}`).join('<br>');
out.innerHTML=h}
document.getElementById('t').value=tok();load();setInterval(load,15000)</script></body></html>"""


def make_handler(engine, cfg):
    token = str(cfg["server"]["token"])
    reports = Path(cfg["paths"]["reports"])

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _auth(self) -> bool:
            q = parse_qs(urlparse(self.path).query)
            given = (self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                     or (q.get("token") or [""])[0])
            return bool(token) and token != "change-me" and hmac.compare_digest(given, token)

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str))

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, DASH, "text/html; charset=utf-8")
            if not self._auth():
                return self._json({"error": "unauthorized"}, 401)
            if path == "/api/status":
                return self._json(engine.snapshot())
            if path == "/api/report":
                p = reports / "summary.json"
                return self._send(200, p.read_bytes() if p.exists() else b"{}")
            if path == "/report":
                p = reports / "report.html"
                return self._send(200, p.read_bytes() if p.exists() else "<p>هنوز بک‌تست اجرا نشده</p>",
                                  "text/html; charset=utf-8")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if not self._auth():
                return self._json({"error": "unauthorized"}, 401)
            path = urlparse(self.path).path
            if path == "/api/pause":
                engine.paused = True
            elif path == "/api/resume":
                engine.paused = False
            elif path == "/api/close_all":
                with engine.lock:
                    for o in engine.broker.orders():
                        try:
                            engine.broker.cancel(o)
                        except Exception as exc:  # noqa: BLE001
                            log.error("cancel failed: %s", exc)
                    for p in engine.broker.positions():
                        try:
                            engine.broker.close(p)
                        except Exception as exc:  # noqa: BLE001
                            log.error("close failed: %s", exc)
            else:
                return self._json({"error": "not found"}, 404)
            return self._json({"ok": True, "paused": engine.paused})

    return H


def serve(engine, cfg) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((cfg["server"]["host"], int(cfg["server"]["port"])), make_handler(engine, cfg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("dashboard on http://%s:%s", cfg["server"]["host"], cfg["server"]["port"])
    return srv
