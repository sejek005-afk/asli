"""Round 9 research: turn walk-forward predictions into trades and apply the pre-registered rules.

    python r9_eval.py <work_dir> [out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SYMS = ("XAUUSD_o", "EURUSD_o")
HORIZONS = (12, 48, 288)
LEARNERS = ("gbtall", "gbt24", "gbtpool", "knn")
THETAS = (0.02, 0.05, 0.10, 0.15, 0.20)
TRADE_START = pd.Timestamp("2023-01-01", tz="UTC")
JUDGE_A = pd.Timestamp("2024-11-07", tz="UTC")
JUDGE_MID = pd.Timestamp("2025-11-07", tz="UTC")
JUDGE_B = pd.Timestamp("2026-11-07", tz="UTC")
T_MIN, MEAN_MIN, N_MIN, ALPHA_T_MIN = 2.9, 0.04, 150, 2.0


def sequence(pl: np.ndarray, ps: np.ndarray, lab: dict, h: int, theta: float) -> pd.DataFrame:
    """Non-overlapping trades: enter when flat and the larger prediction >= theta."""
    best = np.fmax(pl, ps)
    ok = np.isfinite(pl) & np.isfinite(ps) & np.isfinite(lab[f"r_long_{h}"]) & (best >= theta)
    cand = np.flatnonzero(ok)
    rows, free, i = [], 0, 0
    while i < len(cand):
        p = cand[i]
        side = 1 if pl[p] >= ps[p] else -1
        key = "long" if side == 1 else "short"
        off = int(lab[f"off_{key}_{h}"][p])
        rows.append((p, side, lab[f"r_{key}_{h}"][p], lab[f"cost_r_{h}"][p], lab[f"mkt_{key}_{h}"][p], p + off))
        free = p + off
        i = np.searchsorted(cand, free, side="left")
    df = pd.DataFrame(rows, columns=["pos", "side", "r", "cost", "mkt", "exit_pos"])
    t = pd.DatetimeIndex(lab["t"], tz="UTC")
    df["time"] = t[df["pos"].to_numpy()] if len(df) else pd.DatetimeIndex([], tz="UTC")
    df["exit_time"] = t[np.minimum(df["exit_pos"].to_numpy(), len(t) - 1)] if len(df) else pd.DatetimeIndex([], tz="UTC")
    return df


def adaptive(seqs: dict[float, pd.DataFrame]) -> tuple[pd.DataFrame, list]:
    """Each quarter trade the theta that made the most R over the previous 4 quarters (or stay out)."""
    out, picks, last_exit = [], [], -1
    a = TRADE_START
    while a < JUDGE_B:
        b = a + pd.DateOffset(months=3)
        best, best_sum = None, 0.0
        for th, s in seqs.items():
            w = s[(s["time"] >= a - pd.DateOffset(months=12)) & (s["exit_time"] < a)]
            if len(w) >= 30 and w["r"].sum() > best_sum:
                best, best_sum = th, w["r"].sum()
        picks.append((str(a.date()), best))
        if best is not None:
            s = seqs[best]
            q = s[(s["time"] >= a) & (s["time"] < b)]
            for row in q.itertuples(index=False):
                if row.pos >= last_exit:
                    out.append(row)
                    last_exit = row.exit_pos
        a = b
    cols = list(next(iter(seqs.values())).columns)
    return pd.DataFrame(out, columns=cols), picks


def stats(tr: pd.DataFrame) -> dict:
    n = len(tr)
    if n == 0:
        return {"n": 0}
    r = tr["r"].to_numpy()
    d = {"n": n, "mean": float(r.mean()), "sum": float(r.sum()), "win": float((r > 0).mean())}
    d["t"] = float(r.mean() / (r.std(ddof=1) / np.sqrt(n))) if n > 2 and r.std(ddof=1) > 0 else 0.0
    d["mean_2x_cost"] = float((r - tr["cost"].to_numpy()).mean())
    if n > 10:
        x = tr["mkt"].to_numpy()
        Xm = np.c_[np.ones(n), x]
        beta, *_ = np.linalg.lstsq(Xm, r, rcond=None)
        res = r - Xm @ beta
        cov = np.linalg.inv(Xm.T @ Xm) * (res @ res) / (n - 2)
        d["alpha"], d["alpha_t"], d["beta"] = float(beta[0]), float(beta[0] / np.sqrt(cov[0, 0])), float(beta[1])
    for name, m in (("long", tr["side"] == 1), ("short", tr["side"] == -1)):
        d[f"n_{name}"] = int(m.sum())
        d[f"mean_{name}"] = float(tr.loc[m, "r"].mean()) if m.any() else None
    return d


def judge(tr: pd.DataFrame) -> dict:
    j = tr[(tr["time"] >= JUDGE_A) & (tr["time"] < JUDGE_B)]
    h1, h2 = j[j["time"] < JUDGE_MID], j[j["time"] >= JUDGE_MID]
    s = stats(j)
    full = stats(tr)
    checks = {
        "trades>=150": s["n"] >= N_MIN,
        "mean>=0.04R": s.get("mean", -1) >= MEAN_MIN,
        "t>=2.9": s.get("t", 0) >= T_MIN,
        "both_halves_positive": len(h1) > 0 and len(h2) > 0 and h1["r"].mean() > 0 and h2["r"].mean() > 0,
        "positive_at_2x_cost": s.get("mean_2x_cost", -1) > 0,
        "alpha_t>=2": s.get("alpha_t", 0) >= ALPHA_T_MIN,
        "full_period_positive": full.get("mean", -1) > 0,
    }
    by_year = {int(y): {"n": int(len(g)), "sum": float(g["r"].sum()), "mean": float(g["r"].mean())}
               for y, g in tr.groupby(tr["time"].dt.year)} if len(tr) else {}
    return {"judge": s, "half1": stats(h1), "half2": stats(h2), "full_2023_2026": full, "by_year": by_year,
            "checks": checks, "passed": bool(all(checks.values()))}


def main() -> int:
    work = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else work / "r9_results.json"
    labs = {s: dict(np.load(work / f"labels_{s}.npz")) for s in SYMS}
    results = {}
    for ln in LEARNERS:
        for s in SYMS:
            for h in HORIZONS:
                f = work / f"pred_{ln}_{s}_{h}.npz"
                if not f.exists():
                    continue
                p = np.load(f)
                pl, ps = p["long"].astype(float), p["short"].astype(float)
                lab = labs[s]
                seqs = {th: sequence(pl, ps, lab, h, th) for th in THETAS}
                tr, picks = adaptive(seqs)
                res = judge(tr)
                res["theta_picks"] = picks
                # side reports
                t = pd.DatetimeIndex(lab["t"], tz="UTC")
                skill = {}
                for y in range(2022, 2027):
                    m = (t.year == y) & np.isfinite(pl) & np.isfinite(lab[f"r_long_{h}"])
                    if m.sum() > 1000:
                        skill[y] = {"long": float(spearmanr(pl[m], lab[f"r_long_{h}"][m]).statistic),
                                    "short": float(spearmanr(ps[m], lab[f"r_short_{h}"][m]).statistic)}
                res["oos_rank_corr"] = skill
                fx = seqs[0.05]
                res["fixed_theta_0.05_by_year"] = {
                    int(y): {"n": int(len(g)), "sum": float(g["r"].sum()), "mean": float(g["r"].mean())}
                    for y, g in fx.groupby(fx["time"].dt.year)} if len(fx) else {}
                res["fixed_theta_0.05_all"] = stats(fx)
                results[f"{ln}|{s}|{h}"] = res
                j = res["judge"]
                print(f"{ln:8s} {s:9s} h={h:3d}  n={j['n']:4d} mean={j.get('mean', float('nan')):+.3f} "
                      f"t={j.get('t', float('nan')):+.2f} 2x={j.get('mean_2x_cost', float('nan')):+.3f} "
                      f"alpha_t={j.get('alpha_t', float('nan')):+.2f} full={res['full_2023_2026'].get('mean', float('nan')):+.3f} "
                      f"{'PASS' if res['passed'] else 'fail'}", flush=True)
    out_path.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("tests:", len(results), "passed:", sum(r["passed"] for r in results.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
