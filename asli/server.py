"""Small HTTP API + web dashboard (standard library only). The Android app shows this page.

All endpoints except "/" need the token from config server.token, sent as header
"Authorization: Bearer <token>" or query "?token=<token>".
GET  /api/status        account, positions, decisions, styles, leverage, calendar
GET  /api/brain         what the bot sees and waits for, news, learning (live vs backtest)
GET  /api/settings      risk / daily loss / max positions / styles + backtest preview per risk level
POST /api/settings      change them (JSON body; values are clamped to safe bounds)
GET  /api/update        state of the last in-app update
POST /api/update        git pull + backtest, then restart (start_live.cmd's loop restarts the bot)
GET  /api/report        latest research summary (json)
GET  /report            latest research report (html)
POST /api/telegram/test find the user's chat (after they messaged the bot) and send a test message
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

DASH = r"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>ASLI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0d14;--card:#121826;--card2:#171f31;--line:#232c40;--text:#e9edf5;--muted:#8d97ab;
--green:#34d399;--red:#f87171;--amber:#fbbf24;--blue:#60a5fa;--violet:#a78bfa;--gold:#f5c451}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text)}
body{font-family:Vazirmatn,Tahoma,sans-serif;font-size:15px;line-height:1.75;padding-bottom:86px}
header{position:sticky;top:0;z-index:5;background:rgba(10,13,20,.93);backdrop-filter:blur(8px);
padding:12px 16px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--line)}
.logo{font-weight:800;font-size:20px;letter-spacing:.5px;margin-left:auto;display:flex;align-items:center;gap:8px}
.logo i{width:12px;height:12px;border-radius:50%;background:var(--gold);box-shadow:0 0 12px var(--gold)}
.pill{font-size:12px;padding:3px 10px;border-radius:99px;background:var(--card2);border:1px solid var(--line);white-space:nowrap}
.pill.g{color:var(--green);border-color:#1f5f4a}.pill.r{color:var(--red);border-color:#6b2b2b}.pill.a{color:var(--amber);border-color:#6b5420}
.pill.b{color:var(--blue);border-color:#284a7a}.pill.v{color:var(--violet);border-color:#4a3c7a}
main{padding:6px 14px;max-width:780px;margin:auto}
h2{font-size:15px;color:var(--muted);font-weight:600;margin:20px 4px 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;margin:10px 0}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:10px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px}
.kpi .l{color:var(--muted);font-size:12px}.kpi .v{font-size:21px;font-weight:800;direction:ltr;text-align:right}.kpi .s{font-size:12px;color:var(--muted)}
.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.sp{flex:1}
.muted{color:var(--muted)}.g{color:var(--green)}.r{color:var(--red)}.a{color:var(--amber)}
.style{border-right:4px solid var(--line)}.style.on{border-right-color:var(--green)}.style.off{border-right-color:#555}
.style .t{font-weight:800;font-size:16px}.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.mini div{background:var(--card2);border-radius:10px;padding:6px 8px;font-size:12px}.mini b{display:block;font-size:15px;direction:ltr;text-align:right}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:6px 4px;text-align:center}
th{color:var(--muted);font-weight:600}
details.dec{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:8px 0;padding:10px 12px}
details.dec summary{list-style:none;cursor:pointer;display:flex;gap:8px;align-items:center}
details.dec summary::-webkit-details-marker{display:none}
details.dec pre{white-space:pre-wrap;font-family:inherit;font-size:13px;color:#c9d1e0;background:var(--card2);border-radius:10px;padding:10px;margin:10px 0 0}
.think li{margin:6px 0}.think{padding-right:18px;margin:6px 0}
.state{font-size:12px;padding:2px 9px;border-radius:99px;background:var(--card2)}
.st-armed{color:var(--gold)}.st-waiting,.st-idle{color:var(--muted)}.st-building{color:var(--blue)}.st-done{color:var(--violet)}
.lvl{direction:ltr;display:inline-block}
input[type=range]{width:100%;accent-color:var(--gold)}
.big{font-size:28px;font-weight:800;direction:ltr;text-align:right}
.btn{font:inherit;border:0;border-radius:12px;padding:10px 14px;background:#24304a;color:var(--text);cursor:pointer}
.btn.p{background:var(--gold);color:#1b1503;font-weight:800}.btn.d{background:#3b1d22;color:#ffb4b4}.btn.w{width:100%}
.sw{position:relative;width:46px;height:26px;flex:none}.sw input{display:none}
.sw span{position:absolute;inset:0;background:#2a3348;border-radius:99px;transition:.2s}
.sw span:before{content:"";position:absolute;width:20px;height:20px;top:3px;right:3px;background:#cfd6e6;border-radius:50%;transition:.2s}
.sw input:checked+span{background:#1f7a57}.sw input:checked+span:before{right:23px;background:#fff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.box{background:var(--card2);border-radius:12px;padding:8px 10px}
.box .l{font-size:12px;color:var(--muted)}.box b{font-size:17px;direction:ltr;display:block;text-align:right}
nav{position:fixed;bottom:0;left:0;right:0;display:flex;background:rgba(14,18,29,.97);border-top:1px solid var(--line);
padding:6px 6px calc(6px + env(safe-area-inset-bottom));z-index:6}
nav button{flex:1;font:inherit;background:none;border:0;color:var(--muted);padding:6px 0;border-radius:12px;font-size:13px}
nav button b{display:block;font-size:19px;line-height:1.2}nav button.on{color:var(--gold);background:#1a2133}
.note{font-size:12px;color:var(--muted)}.log{white-space:pre-wrap;font-size:11px;direction:ltr;text-align:left;background:#0d1220;border-radius:10px;padding:8px;max-height:220px;overflow:auto}
input.tok{font:inherit;width:100%;padding:10px;border-radius:12px;border:1px solid var(--line);background:var(--card2);color:var(--text);direction:ltr}
.empty{color:var(--muted);text-align:center;padding:10px}
.seg{display:flex;background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:3px;margin:10px 0}
.seg button{flex:1;font:inherit;border:0;background:none;color:var(--muted);padding:8px;border-radius:9px}
.seg button.on{background:#2a3550;color:var(--text);font-weight:800}
input.num{font:inherit;font-size:22px;font-weight:800;width:150px;padding:8px 12px;border-radius:12px;border:1px solid var(--line);
background:var(--card2);color:var(--text);direction:ltr;text-align:center}
</style></head><body>
<header><div class="logo"><i></i>ASLI</div><span id="hs" class="pill">…</span><span id="hm" class="pill">…</span><span id="hd" class="pill b">DEMO</span></header>
<main>
<section id="tab-dash"></section>
<section id="tab-brain" hidden></section>
<section id="tab-set" hidden></section>
</main>
<nav><button data-t="dash" class="on"><b>◧</b>داشبورد</button><button data-t="brain"><b>✦</b>مغز ربات</button><button data-t="set"><b>⚙</b>تنظیمات</button></nav>
<script>
(function(){const q=new URLSearchParams(location.search).get('token');if(q){try{localStorage.setItem('t',q)}catch(e){};window._t=q}})();
function tok(){if(window._t)return window._t;try{return localStorage.getItem('t')||''}catch(e){return ''}}
const $=id=>document.getElementById(id);
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function api(p,opt){opt=opt||{};opt.headers=Object.assign({'Authorization':'Bearer '+tok()},opt.headers||{});
 const r=await fetch(p,opt);if(r.status==401)throw new Error('auth');return r.json()}
const num=(x,d=2)=>x==null||isNaN(x)?'—':Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sgn=(x,d=2)=>x==null?'—':(x>0?'+':'')+Number(x).toFixed(d);
const cls=x=>x==null?'':(x>0?'g':(x<0?'r':''));
const thr=iso=>{try{return new Date(iso).toLocaleString('fa-IR',{timeZone:'Asia/Tehran',weekday:'short',hour:'2-digit',minute:'2-digit'})}catch(e){return iso}};
const side=d=>d>0?'<span class="g">خرید</span>':'<span class="r">فروش</span>';
let TAB='dash',S=null,B=null,SET=null;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.t));
function show(t){TAB=t;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t==t));
 ['dash','brain','set'].forEach(x=>$('tab-'+x).hidden=x!=t);refresh()}
function login(msg){$('tab-'+TAB).innerHTML=`<div class="card"><b>ورود</b><p class="note">${esc(msg||'توکن ربات را وارد کنید (server.token در config.yaml).')}</p>
<input class="tok" id="tk" placeholder="token"><p><button class="btn p w" onclick="window._t=$('tk').value;try{localStorage.setItem('t',window._t)}catch(e){};refresh()">ورود</button></p></div>`}
function header(s){const st=s.paused?['متوقف','r']:(String(s.status).startsWith('error')?['خطا','r']:['روشن','g']);
 $('hs').textContent='ربات '+st[0];$('hs').className='pill '+st[1];
 $('hm').textContent=s.market_open?'بازار باز':'بازار بسته';$('hm').className='pill '+(s.market_open?'g':'a');
 $('hd').textContent=(s.mode||'').toUpperCase()}
function styleCard(x){const sw=x.switch;let swt='';if(x.auto_switch){swt=sw?(sw.on?`<span class="pill g">کلید خودکار: روشن</span>`:`<span class="pill r">کلید خودکار: خاموش</span>`):`<span class="pill a">کلید: در حال محاسبه</span>`}
 const d=x.drift||{};const dc={ok:'g',watch:'a',drift:'r'}[d.status]||'';
 return `<div class="card style on"><div class="row"><span class="t">${esc(x.title)}</span><span class="pill v">${esc(x.badge)}</span><span class="sp"></span>${swt}</div>
 <div class="mini"><div>معامله‌ی واقعی<b>${x.live.n}</b></div><div>میانگین واقعی<b class="${cls(x.live.avg_r)}">${x.live.avg_r==null?'—':sgn(x.live.avg_r)+'R'}</b></div>
 <div>انتظار بک‌تست<b>${x.expected&&x.expected.mean!=null?sgn(x.expected.mean)+'R':'—'}</b></div></div>
 ${sw&&sw.trades!=null?`<div class="note">۱۲ ماه اخیر روی داده‌ی بروکر: ${sw.trades} معامله، وین‌ریت ${Math.round(sw.win_rate*100)}٪، ${sgn(sw.sum_r,1)}R</div>`:''}
 <div class="note ${dc}">${esc(d.text||'')}</div></div>`}
function decCard(d){return `<details class="dec"><summary><span class="${d.approved?'g':'r'}">${d.approved?'✔':'✖'}</span>
 <b>${esc(d.strategy)}</b> ${side(d.direction)}<span class="sp"></span><span class="muted" style="font-size:12px">${thr(d.time)}</span></summary>
 <div class="note lvl">entry ${num(d.entry)} · SL ${num(d.sl)} · TP ${num(d.tp)}</div><pre>${esc(d.debate)}</pre></details>`}
function renderDash(s){const a=s.account,l=s.leverage||{},t=s.today||{};
 let h=`<div class="kpis" style="margin-top:12px"><div class="kpi"><div class="l">موجودی</div><div class="v">${num(a.balance)}</div><div class="s">${esc(a.currency)}</div></div>
 <div class="kpi"><div class="l">اکوییتی</div><div class="v">${num(a.equity)}</div><div class="s">سود/زیان امروزِ ربات: <span class="${cls(t.pl)}">${t.pl==null?'—':sgn(t.pl)}</span></div></div>
 <div class="kpi"><div class="l">اهرم واقعیِ پوزیشن‌ها</div><div class="v">${l.effective==null?'—':num(l.effective)+'×'}</div><div class="s">سقف ${l.cap?num(l.cap,0)+'×':'—'} · اهرم حساب ۱:${esc(l.account||'—')}</div></div>
 <div class="kpi"><div class="l">مبلغ هر معامله ${s.risk.trade_risk_mode=='manual'?'(دستی)':'(خودکار)'}</div><div class="v">${s.sizing&&s.sizing.risk_money!=null?num(s.sizing.risk_money)+' $':num(s.risk.risk_per_trade_pct)+'٪'}</div><div class="s">${s.sizing&&s.sizing.risk_pct_of_capital!=null?num(s.sizing.risk_pct_of_capital)+'٪ از سرمایه‌ی '+num(s.sizing.capital,0)+' $'+(s.risk.capital_mode=='manual'?' (دستی)':''):''} · سقف ضرر روزانه ${num(s.risk.max_daily_loss_pct,1)}٪ · حداکثر ${s.risk.max_open_positions} پوزیشن</div></div></div>`;
 h+=`<div class="card"><div class="row"><b>نتیجه‌ی زنده</b><span class="sp"></span><span class="muted">${s.live_stats.trades} معامله · وین‌ریت ${Math.round(s.live_stats.win_rate*100)}٪ · <span class="${cls(s.live_stats.total_r)}">${sgn(s.live_stats.total_r)}R</span></span></div></div>`;
 h+='<h2>سبک‌های فعال</h2>'+(s.styles||[]).map(styleCard).join('');
 h+='<h2>پوزیشن‌های باز</h2>'+(s.positions.length?`<div class="card"><table><tr><th>سبک</th><th>جهت</th><th>حجم</th><th>ورود</th><th>SL</th><th>TP</th></tr>`+
 s.positions.map(p=>`<tr><td>${esc(p.strategy||'')}</td><td>${side(p.direction)}</td><td>${p.volume}</td><td>${num(p.entry)}</td><td>${num(p.sl)}</td><td>${num(p.tp)}</td></tr>`).join('')+'</table></div>':'<div class="card empty">پوزیشن بازی نیست</div>');
 h+='<h2>تصمیم‌های ربات</h2>'+(s.decisions.length?s.decisions.slice(0,15).map(decCard).join(''):'<div class="card empty">هنوز تصمیمی گرفته نشده</div>');
 const up=(s.fundamental&&s.fundamental.upcoming)||[];
 h+='<h2>تقویم اقتصادی ۲۴ ساعت آینده</h2><div class="card">'+(up.length?up.map(e=>`<div class="row"><span class="pill ${e.impact=='High'?'r':'a'}">${esc(e.cur)}</span><span>${esc(e.title)}</span><span class="sp"></span><span class="muted" style="font-size:12px">${thr(e.time)}</span></div>`).join(''):'<div class="empty">خبر مهمی نیست</div>')+'</div>';
 $('tab-dash').innerHTML=h}
function renderBrain(b){let h=`<div class="card" style="margin-top:12px"><b>ربات الان چه فکر می‌کند</b><ul class="think">${(b.summary||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>`;
 for(const s of b.symbols||[]){const m=s.market;if(m){h+=`<h2>${esc(s.symbol)} از نگاه ربات</h2><div class="card"><div class="grid2">
 <div class="box"><div class="l">قیمت</div><b>${num(m.price)}</b></div><div class="box"><div class="l">روند روزانه</div><b style="direction:rtl">${m.trend.d50>0?'<span class="g">صعودی ▲</span>':(m.trend.d50<0?'<span class="r">نزولی ▼</span>':'نامشخص')}</b></div>
 <div class="box"><div class="l">نوسان</div><b style="direction:rtl">${esc(m.volatility.label)} <span class="muted" style="font-size:12px">(${m.volatility.ratio}×)</span></b></div><div class="box"><div class="l">سشن</div><b style="direction:rtl">${esc(m.session.label)}</b></div>
 <div class="box"><div class="l">سقف / کف امروز</div><b>${num(m.day.high)} / ${num(m.day.low)}</b></div><div class="box"><div class="l">سقف / کف دیروز</div><b>${num(m.day.prev_high)} / ${num(m.day.prev_low)}</b></div></div></div>`}
 h+='<h2>هر سبک منتظر چیست</h2>'+(s.styles||[]).map(x=>`<div class="card"><div class="row"><b>${esc(x.title)}</b><span class="sp"></span><span class="state st-${esc(x.state)}">${({armed:'آماده',waiting:'منتظر',building:'در حال ساخت',done:'تمام برای امروز',idle:'بیکار',on:'فعال',error:'خطا'})[x.state]||esc(x.state)}</span></div>
 <div>${esc(x.text)}</div>${x.gate&&!x.gate.ok?`<div class="note r">${esc(x.gate.why)}</div>`:''}</div>`).join('')}
 const n=b.news||{};h+=`<h2>خبرها</h2><div class="card">${n.blackout?`<div class="r">الان خبر «${esc(n.blackout.title)}»: معامله‌ی تازه ممنوع</div>`:''}
 ${(n.upcoming||[]).length?(n.upcoming||[]).map(e=>`<div class="row"><span class="pill ${e.impact=='High'?'r':'a'}">${esc(e.cur)}</span>${esc(e.title)}<span class="sp"></span><span class="muted" style="font-size:12px">${esc(e.tehran)}</span></div>`).join(''):'<div class="empty">تا ۲۴ ساعت آینده خبر مهمی نیست</div>'}
 <div class="note">تقویم: ${n.calendar_ok?'به‌روز':'قدیمی'} · ریسک‌گریزی بازار از تیترها: ${n.risk_off==null?'—':Math.round(n.risk_off*100)+'٪'}</div></div>`;
 const L=b.learning||{};h+='<h2>یادگیری: واقعیت در برابر انتظار</h2>'+(L.styles||[]).map(x=>`<div class="card"><div class="row"><b>${esc(x.title)}</b><span class="sp"></span>
 <span class="pill ${({ok:'g',watch:'a',drift:'r'})[x.drift.status]||'b'}">${({ok:'مطابق انتظار',watch:'زیر نظر',drift:'متوقف شد',collecting:'در حال جمع کردن نمونه',unknown:'بدون انتظار'})[x.drift.status]||''}</span></div>
 <div class="mini"><div>بک‌تست ۲ سال<b>${x.expected.mean!=null?sgn(x.expected.mean)+'R':'—'}</b></div><div>وین‌ریت بک‌تست<b>${x.expected.win!=null?Math.round(x.expected.win*100)+'٪':'—'}</b></div><div>معامله در سال<b>${x.expected.per_year!=null?Math.round(x.expected.per_year):'—'}</b></div></div>
 <div class="note">${esc(x.drift.text)}</div>${x.best_hours&&x.best_hours.length?`<div class="note">بهترین ساعت‌ها (تهران): ${x.best_hours.map(esc).join('، ')}</div>`:''}</div>`).join('');
 h+=`<div class="note" style="margin:8px 4px">مدل یادگیرنده: آموزش روی ${L.learner?L.learner.n_trained:0} معامله‌ی بک‌تست، ${L.learner?L.learner.live_updates:0} به‌روزرسانی از معامله‌های واقعی.</div>`;
 $('tab-brain').innerHTML=h}
function prevFor(r){const P=(SET&&SET.preview)||[];let best=null;for(const p of P){if(!best||Math.abs(p.risk-r)<Math.abs(best.risk-r))best=p}return best}
function seg(id,val,opts){return `<div class="seg" id="${id}" data-v="${val}">`+opts.map(o=>`<button data-o="${o[0]}" class="${val==o[0]?'on':''}" onclick="segSet('${id}','${o[0]}')">${o[1]}</button>`).join('')+'</div>'}
function segSet(id,v){$(id).dataset.v=v;$(id).querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.o==v));moneyShow();riskInfo()}
function moneyShow(){const cm=$('cm').dataset.v,tm=$('tm').dataset.v;$('cm-auto').hidden=cm!='auto';$('cm-man').hidden=cm!='manual';
 $('tm-auto').hidden=tm!='auto';$('tm-man').hidden=tm!='manual'}
function capNow(){const z=(SET&&SET.sizing)||{};return $('cm').dataset.v=='manual'?Math.min(Number($('cap').value)||0,z.balance||0):(z.balance||0)}
function pctNow(){if($('tm').dataset.v=='manual'){const c=capNow();return c>0?Math.min(100*(Number($('amt').value)||0)/c,2):0}return Number($('risk').value)}
function renderSet(x){const v=x.values,z=x.sizing||{};let h=`<div class="card" style="margin-top:12px"><b>پولی که ربات با آن معامله می‌کند</b>
 ${seg('cm',v.capital_mode||'auto',[['auto','خودکار: کل موجودی'],['manual','دستی: مبلغ دلخواه']])}
 <div id="cm-auto" class="note">ربات با کل موجودی حساب کار می‌کند: ${num(z.balance)} دلار. هر وقت موجودی بالا یا پایین برود، خودش اندازه‌ی معامله‌ها را با آن تنظیم می‌کند.</div>
 <div id="cm-man"><div class="row"><input class="num" id="cap" type="number" min="10" step="10" value="${v.capital_usd>0?v.capital_usd:Math.round(z.balance||1000)}" oninput="riskInfo()"><span>دلار</span></div>
 <div class="note">ربات فقط با همین مبلغ حساب می‌کند (حداکثر به اندازه‌ی موجودی حساب: ${num(z.balance)} دلار). سقف ضرر روزانه و سقف اهرم هم از همین مبلغ حساب می‌شوند.</div></div></div>
 <div class="card"><b>مبلغ هر معامله</b><div class="note">یعنی اگر استاپ بخورد، ربات این‌قدر ضرر می‌کند. اندازه‌ی معامله (اهرم واقعی) از همین درمی‌آید؛ اهرم خود حساب (۱:${esc(x.leverage.account||'—')}) را فقط بروکر عوض می‌کند.</div>
 ${seg('tm',v.trade_risk_mode||'auto',[['auto','خودکار'],['manual','دستی']])}
 <div id="tm-auto"><div class="row"><span>درصد سرمایه در هر معامله</span><span class="sp"></span><span class="big" id="rv">${num(v.risk_per_trade_pct)}٪</span></div>
 <input type="range" id="risk" min="0.05" max="2" step="0.05" value="${v.risk_per_trade_pct}" oninput="riskInfo()">
 <div class="note">ربات خودش هر بار مبلغ را از روی سرمایه حساب می‌کند و بعد از ۳ ضرر پشت سر هم نصفش می‌کند تا دوباره ببرد.</div></div>
 <div id="tm-man"><div class="row"><input class="num" id="amt" type="number" min="0.5" step="0.5" value="${v.trade_risk_usd>0?v.trade_risk_usd:Math.max(1,Math.round(z.risk_money||10))}" oninput="riskInfo()"><span>دلار در هر معامله</span></div>
 <div class="note">همیشه همین مبلغ؛ حداکثر ${num(z.max_trade_risk_pct||2,0)}٪ سرمایه برای حفاظت.</div></div>
 <div class="box" style="margin-top:10px"><div class="l">هر معامله با این تنظیم</div><b id="mv">—</b></div>
 <div id="ri"></div></div>
 <div class="card"><div class="row"><span>سقف ضرر روزانه</span><span class="sp"></span><b id="dv">${num(v.max_daily_loss_pct,1)}٪</b></div>
 <input type="range" id="daily" min="0.5" max="6" step="0.5" value="${v.max_daily_loss_pct}" oninput="$('dv').textContent=Number(this.value).toFixed(1)+'٪'">
 <div class="row" style="margin-top:8px"><span>سقف اهرم واقعی</span><span class="sp"></span><b id="lv">${num(v.max_leverage,0)}×</b></div>
 <input type="range" id="lev" min="1" max="30" step="1" value="${v.max_leverage}" oninput="$('lv').textContent=this.value+'×'">
 <div class="note">کل حجم پوزیشن‌های باز ربات حداکثر چند برابر اکوییتی باشد. اگر معامله‌ای از این سقف بیشتر شود، حجمش کم می‌شود. الان: ${x.leverage.effective==null?'—':num(x.leverage.effective)+'×'}</div>
 <div class="row" style="margin-top:8px"><span>حداکثر پوزیشن هم‌زمان</span><span class="sp"></span><button class="btn" onclick="stepPos(-1)">−</button><b id="pv" style="min-width:28px;text-align:center">${v.max_open_positions}</b><button class="btn" onclick="stepPos(1)">+</button></div></div>
 <h2>سبک‌ها</h2>`+x.styles.map(s=>`<div class="card"><div class="row"><label class="sw"><input type="checkbox" data-s="${esc(s.name)}" ${s.enabled?'checked':''} onchange="riskInfo()"><span></span></label>
 <b>${esc(s.title)}</b><span class="pill ${s.badge=='تأییدشده'?'g':(s.badge=='آزمایشی'?'a':'r')}">${esc(s.badge)}</span></div><div class="note">${esc(s.note)}${s.auto_switch?' · کلید خودکار ۱۲ ماهه روشن':''}</div></div>`).join('')+
 `<p><button class="btn p w" onclick="saveSet()">ذخیره‌ی تنظیمات</button></p><div id="sm" class="note"></div>
 <h2>کنترل ربات</h2><div class="card"><div class="row"><button class="btn" onclick="post('pause')">توقف معامله‌ی تازه</button><button class="btn" onclick="post('resume')">ادامه</button><span class="sp"></span>
 <button class="btn d" onclick="if(confirm('همه‌ی پوزیشن‌های ربات بسته شود؟'))post('close_all')">بستن همه</button></div></div>
 <h2>گزارش</h2><div class="card"><button class="btn w" onclick="location='/report?token='+encodeURIComponent(tok())">گزارش کامل بک‌تست</button></div>
 <h2>گزارش در تلگرام</h2><div class="card">${tgCard(x.telegram||{})}</div>
 <h2>به‌روزرسانی</h2><div class="card"><p class="note">آخرین نسخه را از GitHub می‌گیرد، بک‌تست را دوباره اجرا می‌کند و ربات را دوباره راه می‌اندازد (حدود ۱ دقیقه).</p>
 <button class="btn w" onclick="doUpdate()">به‌روزرسانی ربات</button><div id="um" class="note"></div></div>
 <div class="note" style="margin:10px 4px">آخرین تغییر تنظیمات: ${x.updated_at?thr(x.updated_at):'—'}</div>`;
 $('tab-set').innerHTML=h;moneyShow();riskInfo()}
function tgCard(t){const st=t.configured?(t.chat?`<span class="g">وصل است${t.chat_name?' به '+esc(t.chat_name):''}</span> · توکن ${esc(t.token_hint)}`:'توکن ذخیره شده؛ هنوز چت پیدا نشده'):'وصل نیست';
 return `<div class="note">ربات باز و بسته شدن هر معامله و هر شب ساعت ${esc(t.report_time||'23:30')} تهران یک گزارش کامل می‌فرستد. وضعیت: ${st}${t.last_error?'<br><span class="r">آخرین خطا: '+esc(t.last_error)+'</span>':''}${t.note?'<br><span class="r">'+esc(t.note)+'</span>':''}</div>
 <div class="note">۱) در تلگرام به <b>@BotFather</b> پیام بدهید، <b>/newbot</b> را بزنید و توکنی را که می‌دهد اینجا بگذارید و «ذخیره‌ی تنظیمات» را بزنید. ۲) به ربات تازه‌ی خودتان یک پیام بفرستید. ۳) «آزمایش» را بزنید.</div>
 <input id="tgt" type="password" autocomplete="off" placeholder="${t.configured?'توکن تازه (خالی = بدون تغییر)':'توکن ربات تلگرام'}" style="font:inherit;width:100%;padding:10px 12px;margin-top:8px;border-radius:12px;border:1px solid var(--line);background:var(--card2);color:var(--text);direction:ltr">
 <div class="row" style="margin-top:8px"><button class="btn w" onclick="tgTest()">آزمایش</button>${t.configured?'<button class="btn" onclick="tgOff()">قطع تلگرام</button>':''}</div><div id="tgm" class="note"></div>`}
async function tgTest(){$('tgm').textContent='در حال فرستادن…';try{const r=await api('/api/telegram/test',{method:'POST'});$('tgm').textContent=r.message;if(r.ok){SET=await api('/api/settings');renderSet(SET);$('tgm').textContent=r.message}}catch(e){$('tgm').textContent='خطا: '+e.message}}
async function tgOff(){if(!confirm('گزارش تلگرام قطع شود؟'))return;SET=await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({telegram_token:''})});renderSet(SET)}
function chosen(){return [...document.querySelectorAll('input[data-s]')].filter(i=>i.checked).map(i=>i.dataset.s)}
function riskInfo(){$('rv').textContent=Number($('risk').value).toFixed(2)+'٪';const r=pctNow(),c=capNow();
 $('mv').innerHTML=`${num(c*r/100)} دلار <span class="muted" style="font-size:13px">(${num(r)}٪ از ${num(c,0)} دلار)</span>`;
 const p=prevFor(r);if(!p||r<=0){$('ri').innerHTML='';return}
 const T={};((SET&&SET.styles)||[]).forEach(s=>T[s.name]=s.title);
 const lev=p.leverage||{};const levTxt=Object.keys(lev).map(k=>`${esc(T[k]||k)} <span class="lvl">${num(lev[k]*r/p.risk)}×</span>`).join('، ');
 const pc=x=>x==null?'—':sgn(x,1)+'٪',dd=x=>x==null?'—':num(x,1)+'٪';
 $('ri').innerHTML=`<div class="grid2" style="margin-top:8px"><div class="box"><div class="l">سود سالانه · واقع‌بینانه</div><b class="${cls(p.cagr_recent_half)}">${pc(p.cagr_recent_half)}</b></div>
 <div class="box"><div class="l">بدترین افت · واقع‌بینانه</div><b class="r">${dd(p.dd_recent_half)}</b></div>
 <div class="box"><div class="l">سود سالانه · بک‌تست ۲ سال</div><b class="${cls(p.cagr_recent)}">${pc(p.cagr_recent)}</b></div>
 <div class="box"><div class="l">بدترین افت · بک‌تست ۲ سال</div><b class="r">${dd(p.dd_recent)}</b></div></div>
 <div class="note">«واقع‌بینانه» یعنی اگر هر سبک در عمل فقط نصف سودِ هر معامله‌ی بک‌تست را بیاورد (رایج است، چون این سبک‌ها روی همین داده انتخاب شده‌اند). ۶ سال کامل: سود سالانه ${pc(p.cagr_all)}، بدترین افت ${dd(p.dd_all)}. همه با کلید خودکار، همان‌طور که ربات معامله می‌کند. اهرم یک معامله‌ی معمولی: ${levTxt||'—'}. نتیجه‌ی گذشته تضمین آینده نیست.</div>`}
function stepPos(d){const e=$('pv');e.textContent=Math.min(6,Math.max(1,Number(e.textContent)+d))}
async function saveSet(){const st={};document.querySelectorAll('input[data-s]').forEach(i=>st[i.dataset.s]=i.checked);
 const body={risk_per_trade_pct:Number($('risk').value),max_daily_loss_pct:Number($('daily').value),max_open_positions:Number($('pv').textContent),max_leverage:Number($('lev').value),styles:st,
 capital_mode:$('cm').dataset.v,trade_risk_mode:$('tm').dataset.v};
 if($('tgt')&&$('tgt').value.trim())body.telegram_token=$('tgt').value.trim();
 if(body.capital_mode=='manual')body.capital_usd=Number($('cap').value);if(body.trade_risk_mode=='manual')body.trade_risk_usd=Number($('amt').value);
 try{SET=await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});renderSet(SET);$('sm').textContent='ذخیره شد ✔'}catch(e){$('sm').textContent='ذخیره نشد: '+e.message}}
async function post(a){try{await api('/api/'+a,{method:'POST'});refresh()}catch(e){}}
async function doUpdate(){if(!confirm('ربات به‌روزرسانی و دوباره راه‌اندازی شود؟'))return;try{const u=await api('/api/update',{method:'POST'});$('um').textContent='شروع شد…';pollUpdate()}catch(e){$('um').textContent='خطا'}}
async function pollUpdate(){try{const u=await api('/api/update');const m={running:'در حال اجرا…',up_to_date:'ربات همین الان آخرین نسخه است',failed:'ناموفق',restarting:'نصب شد؛ ربات دوباره راه می‌افتد (حدود ۱ دقیقه)',idle:''};
 $('um').innerHTML=esc(m[u.state]||u.state)+(u.log&&u.log.length?`<div class="log">${esc(u.log.join('\n\n'))}</div>`:'');if(u.state=='running')setTimeout(pollUpdate,3000)}catch(e){$('um').textContent='ربات در حال راه‌اندازی دوباره است…';setTimeout(pollUpdate,8000)}}
async function refresh(){if(!tok()){login();return}
 try{if(TAB=='dash'){S=await api('/api/status');header(S);renderDash(S)}
 else if(TAB=='brain'){B=await api('/api/brain');renderBrain(B);if(!S){S=await api('/api/status');header(S)}}
 else{SET=await api('/api/settings');renderSet(SET);if(!S){S=await api('/api/status');header(S)}}}
 catch(e){if(e.message=='auth')login('توکن اشتباه است.');else $('tab-'+TAB).innerHTML='<div class="card r">اتصال به ربات برقرار نشد</div>'}}
refresh();setInterval(()=>{if(TAB!='set')refresh()},15000);
</script></body></html>"""


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

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 100_000:
                return {}
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
                return obj if isinstance(obj, dict) else {}
            except Exception:  # noqa: BLE001
                return {}

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, DASH, "text/html; charset=utf-8")
            if not self._auth():
                return self._json({"error": "unauthorized"}, 401)
            if path == "/api/status":
                return self._json(engine.snapshot())
            if path == "/api/brain":
                return self._json(engine.brain())
            if path == "/api/settings":
                with engine.lock:
                    return self._json(engine.settings_view())
            if path == "/api/update":
                return self._json(engine.update_state)
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
            if path == "/api/settings":
                body = self._body()
                with engine.lock:
                    return self._json(engine.apply_settings(body))
            if path == "/api/update":
                return self._json(engine.start_update())
            if path == "/api/telegram/test":
                return self._json(engine.telegram.test())  # network call: outside the engine lock
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
