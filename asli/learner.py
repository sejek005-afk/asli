"""The learning part: an L2 logistic regression (pure numpy) that predicts P(win) for a candidate trade
from its context, plus per-(strategy, hour) and per-(strategy, session) edge tables.

- `fit` trains on backtest trades (and later on live trades, weighted higher).
- `update` takes one SGD step after every closed live trade, so the model keeps adapting.
- Everything is saved as JSON in state/learner.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STRATS = ["ict", "lit", "trendline", "crt", "orb", "scalp", "nyorb"]
SESSIONS = ["asia", "london", "newyork", "late"]


def feature_names() -> list[str]:
    return (["bias"] + [f"s_{s}" for s in STRATS] + [f"sess_{s}" for s in SESSIONS] +
            ["hour_sin", "hour_cos", "bias_align", "trend_align", "confluence", "opposition", "rr", "sl_atr",
             "key_level", "killzone", "pd_align", "hour_edge", "session_edge", "news_risk", "fund_align"])


def featurize(row: dict, edges: dict | None = None) -> np.ndarray:
    edges = edges or {}
    h = float(row.get("ny_hour", 0))
    strat = row.get("strategy", "")
    sess = row.get("session", "")
    he = edges.get("hour", {}).get(f"{strat}|{int(h)}", 0.0)
    se = edges.get("session", {}).get(f"{strat}|{sess}", 0.0)
    v = [1.0]
    v += [1.0 if strat == s else 0.0 for s in STRATS]
    v += [1.0 if sess == s else 0.0 for s in SESSIONS]
    v += [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]
    v += [
        float(row.get("bias_align", 0)),
        float(row.get("trend_align", 0)),
        min(float(row.get("confluence", 0)), 3.0) / 3.0,
        min(float(row.get("opposition", 0)), 3.0) / 3.0,
        min(float(row.get("rr", 2.0)), 5.0) / 5.0,
        min(float(row.get("sl_atr", 1.0)), 6.0) / 6.0,
        1.0 if row.get("key_level") in (True, 1, "True", 1.0) else 0.0,
        1.0 if row.get("killzone") in (True, 1, "True", 1.0) else 0.0,
        float(np.clip(row.get("pd_align", 0.0), -1, 1)),
        float(np.clip(he * 4, -1, 1)),
        float(np.clip(se * 4, -1, 1)),
        float(np.clip(row.get("news_risk", 0.0), 0, 1)),
        float(np.clip(row.get("fund_align", 0.0), -1, 1)),
    ]
    return np.asarray(v, float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Learner:
    def __init__(self, l2: float = 1.0, lr: float = 0.05):
        self.w = np.zeros(len(feature_names()))
        self.l2 = l2
        self.lr = lr
        self.edges: dict = {"hour": {}, "session": {}}
        self.payoff: dict = {}  # strategy -> {"win": avg win R, "loss": avg loss R}
        self.n_trained = 0
        self.n_live_updates = 0

    # ------------------------------------------------------------------ edge tables
    def build_edges(self, trades: pd.DataFrame, prior_n: float = 20.0) -> None:
        self.edges = {"hour": {}, "session": {}}
        if trades is None or len(trades) == 0:
            return
        for (s, h), g in trades.groupby(["strategy", "ny_hour"]):
            self.edges["hour"][f"{s}|{int(h)}"] = float(g["r"].sum() / (len(g) + prior_n))
        for (s, ss), g in trades.groupby(["strategy", "session"]):
            self.edges["session"][f"{s}|{ss}"] = float(g["r"].sum() / (len(g) + prior_n))
        self.payoff = {}
        for s, g in trades.groupby("strategy"):
            w = g.loc[g["r"] > 0, "r"]
            lo = g.loc[g["r"] <= 0, "r"]
            self.payoff[s] = {"win": float(w.mean()) if len(w) else 1.5,
                              "loss": float(-lo.mean()) if len(lo) else 1.0}

    # ------------------------------------------------------------------ training
    def fit(self, trades: pd.DataFrame, epochs: int = 300, weights: np.ndarray | None = None) -> dict:
        self.build_edges(trades)
        X = np.vstack([featurize(r, self.edges) for r in trades.to_dict("records")])
        y = (trades["r"].values > 0).astype(float)
        sw = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        w = np.zeros(X.shape[1])
        n = len(y)
        for _ in range(epochs):  # full-batch gradient descent, stable and deterministic
            p = _sigmoid(X @ w)
            grad = X.T @ ((p - y) * sw) / sw.sum() + self.l2 * np.r_[0.0, w[1:]] / n
            w -= 0.5 * grad
        self.w = w
        self.n_trained = n
        p = _sigmoid(X @ w)
        ll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        return {"n": n, "logloss": float(ll), "base_rate": float(y.mean())}

    def update(self, row: dict, won: bool, weight: float = 2.0) -> None:
        """One online SGD step from a closed live trade (live results count double)."""
        x = featurize(row, self.edges)
        p = _sigmoid(x @ self.w)
        self.w -= self.lr * weight * ((p - float(won)) * x + self.l2 * np.r_[0.0, self.w[1:]] / max(self.n_trained, 100))
        # also nudge the hour/session edge tables
        key_h = f"{row.get('strategy')}|{int(row.get('ny_hour', 0))}"
        key_s = f"{row.get('strategy')}|{row.get('session')}"
        r = float(row.get("r", 1.0 if won else -1.0))
        for tab, key in (("hour", key_h), ("session", key_s)):
            old = self.edges[tab].get(key, 0.0)
            self.edges[tab][key] = 0.95 * old + 0.05 * r
        self.n_live_updates += 1

    # ------------------------------------------------------------------ inference
    def prob(self, row: dict) -> float:
        return float(_sigmoid(featurize(row, self.edges) @ self.w))

    def expected_r(self, row: dict) -> float:
        p = self.prob(row)
        po = self.payoff.get(row.get("strategy"), {"win": float(row.get("rr", 2.0)), "loss": 1.0})
        return p * po["win"] - (1 - p) * po["loss"]

    def top_weights(self, k: int = 8) -> list[tuple[str, float]]:
        names = feature_names()
        idx = np.argsort(-np.abs(self.w[1:]))[:k] + 1
        return [(names[i], float(self.w[i])) for i in idx]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "w": self.w.tolist(), "names": feature_names(), "edges": self.edges, "payoff": self.payoff,
            "n_trained": self.n_trained, "n_live_updates": self.n_live_updates,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Learner":
        obj = cls()
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("names") == feature_names():
                obj.w = np.asarray(d["w"], float)
            obj.edges = d.get("edges", obj.edges)
            obj.payoff = d.get("payoff", {})
            obj.n_trained = d.get("n_trained", 0)
            obj.n_live_updates = d.get("n_live_updates", 0)
        return obj

    def best_times(self, min_edge: float = 0.03) -> list[dict]:
        out = []
        for key, e in sorted(self.edges.get("hour", {}).items(), key=lambda kv: -kv[1]):
            if e < min_edge:
                break
            s, h = key.split("|")
            out.append({"strategy": s, "ny_hour": int(h), "edge_r": round(e, 3)})
        return out
