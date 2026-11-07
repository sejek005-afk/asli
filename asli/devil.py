"""The opposing agent (devil's advocate).

Every candidate position is guilty until proven innocent. The agent:
1. checks hard vetoes — any one of them kills the trade, no matter how good the rest looks;
2. lists objections (con) and the evidence for the trade (pro), each with a weight;
3. approves only if pro - con >= approve_margin, i.e. the trade has to *convince* it.

The full debate is returned (in Persian, for the dashboard/app) and logged for every candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import for_symbol


@dataclass
class Verdict:
    approved: bool
    score_pro: float
    score_con: float
    vetoes: list[str] = field(default_factory=list)
    pro: list[str] = field(default_factory=list)
    con: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        head = "✅ قانع شد" if self.approved else "⛔ رد شد"
        lines = [f"{head} — امتیاز موافق {self.score_pro:g} / مخالف {self.score_con:g}"]
        lines += [f"وتو: {v}" for v in self.vetoes]
        lines += [f"+ {p}" for p in self.pro]
        lines += [f"- {c}" for c in self.con]
        return "\n".join(lines)


class DevilsAdvocate:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.margin = float(cfg["devil"].get("approve_margin", 2))
        self.min_prob = float(cfg["fusion"]["min_prob"])
        self.min_ev = float(cfg["fusion"]["min_ev_r"])
        r = cfg["risk"]
        self.min_rr, self.min_sl, self.max_sl = float(r["min_rr"]), float(r["min_sl_pips"]), float(r["max_sl_pips"])
        self.max_open = int(r["max_open_positions"])
        self.max_daily = float(r["max_daily_loss_pct"])
        self._limits: dict[tuple[str, str], dict] = {}

    def limits(self, symbol: str | None, strategy: str | None = None) -> dict:
        """Hard-veto thresholds for this symbol and style: gold has its own stop range, and a style may set
        its own min_rr / min_sl_pips / max_sl_pips / min_ev_r / min_prob / approve_margin in config `strategies`."""
        key = (symbol or "", strategy or "")
        if key not in self._limits:
            cfg = for_symbol(self.cfg, symbol) if symbol else self.cfg
            r, f = cfg["risk"], cfg["fusion"]
            lim = {"min_rr": float(r["min_rr"]), "min_sl": float(r["min_sl_pips"]), "max_sl": float(r["max_sl_pips"]),
                   "min_ev": float(f["min_ev_r"]), "min_prob": float(f["min_prob"]), "margin": self.margin}
            sp = (cfg.get("strategies") or {}).get(strategy or "", {}) or {}
            for src, dst in (("min_rr", "min_rr"), ("min_sl_pips", "min_sl"), ("max_sl_pips", "max_sl"),
                             ("min_ev_r", "min_ev"), ("min_prob", "min_prob"), ("approve_margin", "margin")):
                if src in sp:
                    lim[dst] = float(sp[src])
            self._limits[key] = lim
        return self._limits[key]

    def review(self, c: dict, account: dict | None = None) -> Verdict:
        account = account or {}
        v = Verdict(False, 0.0, 0.0)
        d = c.get("direction", 0)
        side = "خرید" if d == 1 else "فروش"
        lim = self.limits(c.get("symbol"), c.get("strategy"))
        min_rr, min_sl, max_sl, min_prob, min_ev = lim["min_rr"], lim["min_sl"], lim["max_sl"], lim["min_prob"], lim["min_ev"]

        # ------------------------------------------------------------ hard vetoes
        if c.get("rr", 0) < min_rr * (0.75 if c.get("tp1") else 1.0):
            v.vetoes.append(f"ریسک به ریوارد {c.get('rr', 0):.2f} کمتر از حداقل {min_rr} است")
        sl_p = c.get("sl_pips", 0)
        if sl_p < min_sl:
            v.vetoes.append(f"استاپ {sl_p:.1f} پیپ است؛ با اسپرد و کمیسیون، این معامله از اول بازنده است")
        if sl_p > max_sl:
            v.vetoes.append(f"استاپ {sl_p:.1f} پیپ بیشتر از سقف {max_sl} پیپ است")
        ev = c.get("news_blackout")
        if ev:
            v.vetoes.append(f"خبر {ev.get('impact')} «{ev.get('title')}» برای {ev.get('cur')} در بازه‌ی ممنوعه است")
        if account.get("daily_loss_pct", 0) >= self.max_daily:
            v.vetoes.append(f"ضرر امروز به سقف {self.max_daily}% رسیده؛ امروز تمام است")
        if account.get("open_positions", 0) >= self.max_open:
            v.vetoes.append("تعداد پوزیشن‌های باز به حداکثر رسیده")
        if account.get("same_symbol_open", False):
            v.vetoes.append("روی همین نماد پوزیشن باز داریم")
        p = c.get("prob")
        if p is not None and p < min_prob:
            v.vetoes.append(f"مدل یادگیرنده احتمال برد را {p:.0%} می‌داند (حداقل {min_prob:.0%})")
        e = c.get("ev")
        if e is not None and e < min_ev:
            v.vetoes.append(f"امید ریاضی {e:+.2f}R است؛ در بلندمدت سودی ندارد")
        spread_share = c.get("spread_pips", 0) / max(sl_p, 1e-9)
        if spread_share > 0.25:
            v.vetoes.append(f"اسپرد {spread_share:.0%} از ریسک را می‌خورد")

        # ------------------------------------------------------------ arguments
        def pro(w, text):
            v.pro.append(f"{text} (+{w:g})"); v.score_pro += w

        def con(w, text):
            v.con.append(f"{text} (-{w:g})"); v.score_con += w

        ba = c.get("bias_align", 0)
        if ba > 0:
            pro(2, f"جهت روز (Daily Bias) با {side} یکی است")
        elif ba < 0:
            con(2, f"{side} خلاف جهت روز است")
        else:
            con(0.5, "جهت روز مشخص نیست (Inside Day یا روز مبهم)")
        ta = c.get("trend_align", 0)
        if ta > 0:
            pro(1, "هم‌جهت با روند بلندمدت (SMA200 روزانه)")
        elif ta < 0:
            con(1, "خلاف روند بلندمدت (SMA200 روزانه)")
        conf = int(c.get("confluence", 0))
        if conf:
            pro(min(conf, 3), f"{conf} سبک دیگر هم‌زمان همین جهت را تأیید می‌کنند")
        opp = int(c.get("opposition", 0))
        if opp:
            con(min(opp, 3) * 1.5, f"{opp} سبک دیگر هم‌زمان جهت مخالف می‌دهند")
        if c.get("killzone"):
            pro(1, "داخل کیل‌زون (لندن یا نیویورک)")
        else:
            con(1, "بیرون کیل‌زون؛ حجم و نوسان کمتر")
        if c.get("key_level"):
            pro(1, "روی سطح کلیدی نقدینگی (PDH/PDL/آسیا/خط روند)")
        pda = c.get("pd_align", 0.0)
        if pda > 0.1:
            pro(1, "قیمت برای این جهت در ناحیه‌ی مناسب است (خرید در دیسکانت / فروش در پریمیوم)")
        elif pda < -0.1:
            con(1, "قیمت برای این جهت در ناحیه‌ی نامناسب است (خرید در پریمیوم / فروش در دیسکانت)")
        he = c.get("hour_edge", 0.0)
        if he > 0.03:
            pro(1, f"این ساعت در بک‌تست برای این سبک مثبت بوده ({he:+.2f}R)")
        elif he < -0.03:
            con(2, f"این ساعت در بک‌تست برای این سبک منفی بوده ({he:+.2f}R)")
        if p is not None:
            if p >= 0.5:
                pro(2, f"مدل یادگیرنده احتمال برد را {p:.0%} می‌داند")
            elif p >= min_prob:
                pro(1, f"احتمال برد مدل {p:.0%}")
        if e is not None and e >= 0.2:
            pro(1, f"امید ریاضی مثبت {e:+.2f}R")
        fa = c.get("fund_align", 0.0)
        if fa > 0.3:
            pro(1, "اخبار و احساسات بازار هم‌جهت است")
        elif fa < -0.3:
            con(2, "اخبار روز خلاف این جهت است")
        nr = c.get("news_risk", 0.0)
        if nr > 0.5:
            con(1.5, "خبر مهم در ۴ ساعت آینده؛ ممکن است ستاپ را به هم بزند")
        if c.get("news_unknown"):
            con(1, "تقویم اقتصادی در دسترس نیست؛ ریسک خبر نامعلوم است")
        if c.get("risk_off", 0) > 0.3:
            con(1, "فضای ریسک‌گریزی در اخبار (جنگ/بحران)")
        if account.get("loss_streak", 0) >= 3:
            con(1, f"{account['loss_streak']} ضرر پشت‌سرهم؛ بازار با سبک‌ها هماهنگ نیست")

        v.approved = not v.vetoes and (v.score_pro - v.score_con) >= lim["margin"]
        return v
