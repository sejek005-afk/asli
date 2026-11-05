"""Shared types for strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import SymbolMeta
from ..features import DailyContext, atr


@dataclass
class Signal:
    bar: int  # index in the base M5 frame; the decision is made when this bar CLOSES
    time: pd.Timestamp  # open time (UTC) of that bar
    symbol: str
    strategy: str
    direction: int  # +1 long, -1 short
    order: str  # "limit" | "market"
    entry: float
    sl: float
    tp: float
    tp1: float | None = None  # optional partial (50%), stop then moves to break-even
    valid_bars: int = 12  # limit orders expire after this many M5 bars
    max_hold_bars: int = 288
    features: dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def rr(self) -> float:
        return abs(self.tp - self.entry) / max(self.risk, 1e-12)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("bar", "symbol", "strategy", "direction", "order", "entry", "sl",
                                            "tp", "tp1", "valid_bars", "max_hold_bars")}
        d["time"] = self.time.isoformat()
        d["rr"] = round(self.rr, 2)
        d["features"] = self.features
        return d


class Context:
    """Everything a strategy may read. Built once per symbol per run."""

    def __init__(self, df: pd.DataFrame, meta: SymbolMeta, cfg: dict):
        self.df = df
        self.meta = meta
        self.cfg = cfg
        self.symbol = meta.name
        self.atr = atr(df, 14)
        self.daily = DailyContext(df)
        self.spread = df["spread"].values if "spread" in df else np.zeros(len(df))
        default_spread = cfg["costs"]["default_spread_pips"] * meta.pip
        self.spread = np.where(self.spread > 0, self.spread, default_spread)
        self.index = df.index

    def m5_bar_at_close(self, close_time: pd.Timestamp) -> int:
        """Index of the last M5 bar that opens before `close_time` (i.e. the bar whose close completes a
        higher-timeframe candle)."""
        return int(self.index.searchsorted(close_time, side="left")) - 1


class Strategy:
    name = "base"

    def __init__(self, params: dict):
        self.p = params

    def generate(self, ctx: Context) -> list[Signal]:  # pragma: no cover - interface
        raise NotImplementedError


def nearest_target(levels, entry: float, risk: float, direction: int, min_rr: float) -> float | None:
    """Closest liquidity level beyond entry that still pays at least min_rr."""
    cands = []
    for lv in levels:
        if lv is None or not np.isfinite(lv):
            continue
        dist = (lv - entry) * direction
        if dist > 0 and dist / max(risk, 1e-12) >= min_rr:
            cands.append((dist, lv))
    if not cands:
        return None
    return min(cands)[1]
