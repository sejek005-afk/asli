"""Research report (Persian, RTL HTML + Markdown) written to reports/."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .timeutil import tehran_label

NAMES = {"ict": "ICT (Model 2022)", "lit": "LIT (Inducement)", "trendline": "خط روند (TD + نقدینگی)",
         "crt": "CRT (4H)", "orb": "ORB (لندن)", "fusion": "ASLI ترکیبی"}
COLS = [("trades", "معاملات"), ("win_rate", "وین‌ریت"), ("avg_r", "میانگین R"), ("total_r", "جمع R"),
        ("profit_factor", "Profit Factor"), ("max_dd_r", "حداکثر افت (R)"), ("max_losing_streak", "ضرر پیاپی")]


def _fmt(k, v):
    if k == "win_rate":
        return f"{v:.0%}"
    if isinstance(v, float):
        return "∞" if v == float("inf") else f"{v:.2f}"
    return str(v)


def _table(rows: list[list[str]], head: list[str]) -> tuple[str, str]:
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    md += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in head) + "</tr></thead><tbody>"
    h += "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    h += "</tbody></table>"
    return md, h


def build(res: dict, out_dir: str, meta: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    hb: list[str] = []

    def section(title, text_md="", text_html=None):
        md.append(f"\n## {title}\n\n{text_md}\n")
        hb.append(f"<h2>{html.escape(title)}</h2>{text_html if text_html is not None else '<p>' + html.escape(text_md) + '</p>'}")

    now = datetime.now(timezone.utc)
    cut = res.get("cut")
    intro = (f"داده: {meta.get('symbols')} از {meta.get('start')} تا {meta.get('end')} "
             f"({meta.get('bars')} کندل M5). آموزش تا {cut}، تست روی داده‌ی بعد از آن (دیده‌نشده). "
             "هزینه‌ها: اسپرد واقعی کندل‌ها + کمیسیون + اسلیپیج. نتایج بر حسب R (مضرب ریسک هر معامله).")
    section("خلاصه", intro)

    # per strategy, out-of-sample
    rows = []
    for name, st in res["per_strategy"].items():
        t = st["test"]
        rows.append([NAMES.get(name, name)] + [_fmt(k, t.get(k, 0)) for k, _ in COLS])
    ft = res["fusion_test"]
    rows.append([NAMES["fusion"]] + [_fmt(k, ft.get(k, 0)) for k, _ in COLS])
    m, h = _table(rows, ["سبک"] + [c for _, c in COLS])
    section("نتیجه‌ی هر سبک روی داده‌ی تست (خارج از نمونه)", m, h)

    rows = []
    for name, st in res["per_strategy"].items():
        t = st["all"]
        rows.append([NAMES.get(name, name)] + [_fmt(k, t.get(k, 0)) for k, _ in COLS])
    m, h = _table(rows, ["سبک"] + [c for _, c in COLS])
    section("نتیجه‌ی هر سبک روی کل داده", m, h)

    # strengths
    def top(df: pd.DataFrame, col: str, label_fn, k=12, min_n=15):
        if df is None or len(df) == 0:
            return [], ""
        d = df[df["n"] >= min_n].head(k)
        rows = [[NAMES.get(r["strategy"], r["strategy"]) if "strategy" in r else "-", label_fn(r), str(int(r["n"])),
                 f"{r['win_rate']:.0%}", f"{r['avg_r']:+.2f}", f"{r['edge']:+.3f}"] for r in d.to_dict("records")]
        return rows

    rows = top(res["edges_session"], "session", lambda r: r["session"])
    m, h = _table(rows, ["سبک", "سشن", "تعداد", "وین‌ریت", "میانگین R", "لبه (تعدیل‌شده)"])
    section("نقطه‌ی قوت هر سبک: کدام سشن", m, h)

    rows = top(res["edges_hour"], "ny_hour",
               lambda r: f"{int(r['ny_hour']):02d}:00 نیویورک (≈{tehran_label(int(r['ny_hour']))} تهران)", k=15)
    m, h = _table(rows, ["سبک", "ساعت", "تعداد", "وین‌ریت", "میانگین R", "لبه (تعدیل‌شده)"])
    section("بهترین ساعت‌ها (ساعت تهران با ساعت فعلی آمریکا محاسبه شده)", m, h)

    rows = top(res["edges_bias"], "bias_align",
               lambda r: {1: "هم‌جهت Daily Bias", 0: "بدون جهت", -1: "خلاف Daily Bias"}.get(int(r["bias_align"]), "?"))
    m, h = _table(rows, ["سبک", "نسبت به جهت روز", "تعداد", "وین‌ریت", "میانگین R", "لبه"])
    section("اثر جهت روز روی هر سبک", m, h)

    ec = res["edges_conf"]
    rows = [[str(int(r["confluence"])), str(int(r["n"])), f"{r['win_rate']:.0%}", f"{r['avg_r']:+.2f}"]
            for r in ec.sort_values("confluence").to_dict("records")] if len(ec) else []
    m, h = _table(rows, ["تعداد سبک‌های هم‌جهت دیگر", "تعداد", "وین‌ریت", "میانگین R"])
    section("اثر هم‌جهتی سبک‌ها (Confluence)", m, h)

    lw = res["learner"].top_weights(10)
    rows = [[n, f"{w:+.2f}"] for n, w in lw]
    m, h = _table(rows, ["ویژگی", "وزن (مثبت = احتمال برد بیشتر)"])
    section("آنچه مدل یادگیرنده یاد گرفت", m, h)

    vc = res.get("veto_counts", {})
    rows = [[k, str(v)] for k, v in sorted(vc.items(), key=lambda kv: -kv[1])]
    m, h = _table(rows, ["دلیل وتو (کلمه‌ی اول)", "تعداد"])
    section("ایجنت مخالف چه چیزهایی را رد کرد (داده‌ی تست)", m, h)

    deb = res.get("debates", [])[:12]
    md.append("\n## نمونه‌ی مناظره‌های ایجنت مخالف\n")
    hb.append("<h2>نمونه‌ی مناظره‌های ایجنت مخالف</h2>")
    for d in deb:
        md.append(f"\n**{d['time']} {d['symbol']} {d['strategy']} {'خرید' if d['direction'] == 1 else 'فروش'}** "
                  f"(نتیجه‌ی واقعی: {d['r']:+.2f}R)\n\n```\n{d['debate']}\n```\n")
        hb.append(f"<h3>{html.escape(d['time'])} {d['symbol']} {d['strategy']} "
                  f"{'خرید' if d['direction'] == 1 else 'فروش'} — نتیجه {d['r']:+.2f}R</h3>"
                  f"<pre>{html.escape(d['debate'])}</pre>")

    warn = ("بک‌تست تضمین آینده نیست. اگر ستون ASLI ترکیبی روی داده‌ی تست منفی است، ربات را روی حساب واقعی "
            "روشن نکنید؛ روی دمو بماند تا با داده‌ی بیشتر یاد بگیرد.")
    section("هشدار", warn)

    css = ("body{font-family:Tahoma,Vazirmatn,sans-serif;direction:rtl;max-width:980px;margin:auto;padding:16px;"
           "background:#0f1115;color:#e6e6e6}table{border-collapse:collapse;width:100%;margin:8px 0 20px}"
           "td,th{border:1px solid #333;padding:6px;text-align:center}th{background:#1c2230}"
           "pre{white-space:pre-wrap;background:#161a22;padding:10px;border-radius:8px;direction:rtl}"
           "h2{border-bottom:1px solid #333;padding-bottom:4px}")
    page = (f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>گزارش ASLI</title>"
            f"<style>{css}</style></head><body><h1>گزارش بک‌تست ASLI</h1><p>{now:%Y-%m-%d %H:%M} UTC</p>"
            + "".join(hb) + "</body></html>")
    (out / "report.html").write_text(page, encoding="utf-8")
    (out / "report.md").write_text("# گزارش بک‌تست ASLI\n" + "".join(md), encoding="utf-8")
    summary = {
        "generated": now.isoformat(), "meta": meta,
        "per_strategy": {k: v["test"] for k, v in res["per_strategy"].items()},
        "fusion_test": res["fusion_test"], "best_times": res["learner"].best_times(),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if "candidates" in res and len(res["candidates"]):
        res["candidates"].to_csv(out / "all_candidate_trades.csv", index=False)
    return summary
