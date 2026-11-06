"""ASLI Fusion — the combined style.

How the five styles are merged (each contributes what it is best at):
- ICT  -> direction of the day (Daily Bias), timing (killzones), premium/discount, raid->MSS->FVG entries
- LIT  -> the inducement rule: enter only after the first pullback's liquidity is taken
- Trendline -> where retail stops sit (trendline liquidity) as a key level
- CRT  -> higher-timeframe candle ranges: sweep of one side, target the other
- ORB  -> session-open momentum on trend days

Every style produces candidates. Each candidate is scored with context features (bias/trend alignment,
how many other styles agree or disagree within the last hour, killzone, key level, premium/discount,
learned hour/session edge, news), the learner turns that into P(win) and expected R, and the opposing
agent must be convinced before anything is traded. Only one fusion position per symbol at a time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import backtest as bt
from .devil import DevilsAdvocate
from .learner import Learner
from .strategies import Context, Signal, build
from .timeutil import KILLZONES, NY, session_of_hour

log = logging.getLogger("asli.fusion")


def annotate(signals: list[Signal], ctx: Context, window_min: int = 60) -> list[dict]:
    """Context features for every candidate. Causal: confluence only counts signals at or before it."""
    win = max(1, window_min // 5)
    by_bar = sorted(signals, key=lambda s: s.bar)
    bars = np.array([s.bar for s in by_bar])
    rows = []
    d = ctx.daily
    for s in by_bar:
        lo = np.searchsorted(bars, s.bar - win, side="left")
        hi = np.searchsorted(bars, s.bar, side="right")
        others = [o for o in by_bar[lo:hi] if o.strategy != s.strategy]
        conf = len({o.strategy for o in others if o.direction == s.direction})
        opp = len({o.strategy for o in others if o.direction != s.direction})
        ts = ctx.index[s.bar].tz_convert(NY)
        hour = ts.hour + ts.minute / 60.0
        kz = any(a <= hour < b for a, b in KILLZONES.values())
        pdv = d.premium_discount(s.bar, s.entry)
        pd_align = float(np.clip((0.5 - pdv) * 2 * s.direction, -1, 1))
        a = ctx.atr[s.bar]
        rows.append({
            "bar": s.bar, "time": ctx.index[s.bar], "symbol": s.symbol, "strategy": s.strategy,
            "direction": s.direction, "order": s.order, "entry": s.entry, "sl": s.sl, "tp": s.tp, "tp1": s.tp1,
            "rr": s.rr, "ny_hour": int(ts.hour), "session": session_of_hour(hour), "killzone": bool(kz),
            "key_level": bool(s.features.get("key_level", False)),
            "bias_align": int(d.bias[s.bar] * s.direction), "trend_align": int(d.trend[s.bar] * s.direction),
            "confluence": conf, "opposition": opp, "pd_align": pd_align,
            "sl_atr": float(s.risk / a) if a > 0 else 1.0,
            "sl_pips": float(s.risk / ctx.meta.pip), "spread_pips": float(ctx.spread[s.bar] / ctx.meta.pip),
            "news_risk": 0.0, "fund_align": 0.0,
        })
    return rows


def candidate_trades(ctx: Context, cfg: dict) -> pd.DataFrame:
    """All candidates from all styles with their context AND their simulated outcome."""
    sigs: list[Signal] = []
    per = {}
    for strat in build(cfg):
        s = strat.generate(ctx)
        per[strat.name] = s
        sigs.extend(s)
    rows = annotate(sigs, ctx, int(cfg["fusion"]["confluence_window_min"]))
    key = {(r["strategy"], r["bar"]): r for r in rows}
    out = []
    for name, slist in per.items():
        busy = -1
        for s in sorted(slist, key=lambda x: x.bar):
            if s.bar <= busy:
                continue
            t = bt.simulate(s, ctx)
            if t is None:
                continue
            busy = t.exit_bar
            r = dict(key[(name, s.bar)])
            r.update({"r": t.r, "outcome": t.outcome, "fill_time": ctx.index[t.fill_bar],
                      "exit_time": ctx.index[t.exit_bar], "exit_bar": t.exit_bar})
            out.append(r)
    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values("time").reset_index(drop=True)
    return df


def select(cands: pd.DataFrame, learner: Learner, devil: DevilsAdvocate, keep_debates: int = 0):
    """Walk the candidates in time order and keep the ones the learner + opposing agent accept.
    One fusion position per symbol at a time."""
    chosen = []
    debates = []
    busy: dict[str, pd.Timestamp] = {}
    veto_counts: dict[str, int] = {}
    for row in cands.to_dict("records"):
        sym = row["symbol"]
        if sym in busy and row["time"] <= busy[sym]:
            continue
        row["prob"] = learner.prob(row)
        row["ev"] = learner.expected_r(row)
        row["hour_edge"] = learner.edges["hour"].get(f"{row['strategy']}|{row['ny_hour']}", 0.0)
        verdict = devil.review(row)
        for vt in verdict.vetoes:
            k = vt.split(" ")[0]
            veto_counts[k] = veto_counts.get(k, 0) + 1
        if keep_debates and len(debates) < keep_debates:
            debates.append({"time": str(row["time"]), "symbol": sym, "strategy": row["strategy"],
                            "direction": row["direction"], "approved": verdict.approved,
                            "debate": verdict.summary(), "r": row["r"]})
        if verdict.approved:
            chosen.append(row)
            busy[sym] = row["exit_time"]
    return pd.DataFrame(chosen), debates, veto_counts


def research(frames: dict[str, pd.DataFrame], metas: dict, cfg: dict) -> dict:
    """Full study: every style on every symbol, walk-forward fusion, learned session/hour edges."""
    allc = []
    for sym, df in frames.items():
        ctx = Context(df, metas[sym], cfg)
        c = candidate_trades(ctx, cfg)
        log.info("%s: %d candidate trades", sym, len(c))
        allc.append(c)
    cands = pd.concat([c for c in allc if len(c)], ignore_index=True) if any(len(c) for c in allc) else pd.DataFrame()
    if cands.empty:
        return {"candidates": cands, "error": "no trades"}
    cands = cands.sort_values("time").reset_index(drop=True)
    cut = cands["time"].quantile(float(cfg["fusion"]["train_fraction"]))
    train = cands[cands["time"] <= cut]
    test = cands[cands["time"] > cut]
    devil = DevilsAdvocate(cfg)

    # walk-forward: learn only from the past, judge on the unseen future
    wf = Learner()
    fit_info = wf.fit(train)
    fused_test, debates, vetoes = select(test, wf, devil, keep_debates=40)
    fused_train, _, _ = select(train, wf, devil)

    # final model for live trading uses everything
    final = Learner()
    final.fit(cands)

    per_strategy = {}
    for s, g in cands.groupby("strategy"):
        per_strategy[s] = {"all": bt.stats(g), "train": bt.stats(g[g["time"] <= cut]),
                           "test": bt.stats(g[g["time"] > cut])}
    return {
        "candidates": cands, "cut": cut, "train_fit": fit_info,
        "per_strategy": per_strategy,
        "fusion_test": bt.stats(fused_test), "fusion_train": bt.stats(fused_train),
        "fusion_trades": fused_test, "debates": debates, "veto_counts": vetoes,
        "edges_session": bt.shrunk_edge(cands, ["strategy", "session"]),
        "edges_hour": bt.shrunk_edge(cands, ["strategy", "ny_hour"]),
        "edges_bias": bt.shrunk_edge(cands, ["strategy", "bias_align"]),
        "edges_conf": bt.shrunk_edge(cands, ["confluence"]),
        "learner": final, "learner_wf": wf,
    }
