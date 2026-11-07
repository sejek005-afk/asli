"""Configuration loading with safe defaults.

The real config lives in config.yaml (never committed). config.example.yaml documents every key.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "mode": "paper",  # paper | demo | real
    "allow_real": False,  # must be true AND mode=real before the bot touches a real-money account
    "symbols": ["EURUSD", "GBPUSD"],
    "mt5": {
        "path": "",  # terminal64.exe of a dedicated MT5 install; empty = default terminal
        "login": 0,
        "password": "",  # or set env ASLI_MT5_PASSWORD
        "server": "",
        # broker server clock: "auto" | "ny_plus_7" (EET with NY-close DST, most ECN brokers) | "fixed:<hours>"
        "server_time": "auto",
        "magic": 26110601,
        "deviation_points": 20,
    },
    "costs": {
        "default_spread_pips": 0.8,  # used when bars carry no spread
        "commission_pips": 0.7,  # round trip, in pips ($7/lot ECN on EURUSD ~ 0.7 pip)
        "slippage_pips": 0.2,  # on stop / market fills
    },
    "risk": {
        "risk_per_trade_pct": 0.5,
        "max_daily_loss_pct": 2.0,
        "max_open_positions": 2,
        "halve_after_losses": 3,
        "min_sl_pips": 3.0,
        "max_sl_pips": 25.0,
        "min_rr": 1.8,
    },
    "strategies": {
        "ict": {"enabled": True, "arm_bars": 24, "order_valid_bars": 12, "disp_atr": 1.0},
        "lit": {"enabled": True, "swing_k": 3, "order_valid_bars": 48},
        "trendline": {"enabled": True, "min_touches": 3},
        "crt": {"enabled": True, "candle2_ny_hours": [5, 9], "require_key_level": True},
        "orb": {"enabled": True, "range_minutes": 30, "trade_until_minutes": 150, "max_range_pct_of_median": 1.0},
        # gold scalp (see asli/strategies/scalp.py). Its own limits for the opposing agent: the target is
        # ~0.19R by design (win rate ~88%), and the 4-ATR stop is often wider than the gold cap of 100 pips.
        "scalp": {"enabled": True, "symbols": ["XAU"], "rsi_period": 2, "rsi_level": 5, "sl_atr": 4.0,
                  "tp_atr": 0.75, "max_hold_bars": 48, "ny_hours": [2, 12], "h1_fast_bars": 600,
                  "h1_slow_bars": 2400, "min_rr": 0.15, "max_sl_pips": 500.0, "min_ev_r": 0.0},
    },
    "fusion": {
        "min_prob": 0.40,
        "min_ev_r": 0.05,
        "confluence_window_min": 60,
        "train_fraction": 0.6,
    },
    "devil": {"approve_margin": 2, "strict": True},
    "fundamental": {
        "enabled": True,
        "calendar_url": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "calendar_refresh_min": 60,
        "blackout_high_min": 30,
        "blackout_medium_min": 10,
        "news_feeds": [
            "https://www.fxstreet.com/rss/news",
            "https://www.investing.com/rss/news_1.rss",
        ],
        "news_refresh_min": 15,
        "news_halflife_hours": 6,
    },
    "server": {"host": "0.0.0.0", "port": 8787, "token": "change-me"},
    "paths": {"data": "data", "state": "state", "reports": "reports"},
    # per-symbol changes to costs / risk / strategies, keyed by the exact MT5 name, e.g.
    # symbol_overrides: {XAUUSD_o: {risk: {max_sl_pips: 120}}}
    "symbol_overrides": {},
}

# Defaults by asset class, applied before symbol_overrides. Pips are meta.pip (gold: 0.1 = 10 cents),
# so the FX numbers above would cap a gold stop at $2.5 and veto nearly every gold setup.
ASSET_CLASS_DEFAULTS: dict = {
    "XAU": {
        "costs": {"default_spread_pips": 3.0, "commission_pips": 2.0, "slippage_pips": 1.0},
        "risk": {"min_sl_pips": 10.0, "max_sl_pips": 100.0},
    },
}


def asset_class(symbol: str) -> str | None:
    n = str(symbol).upper()
    for prefix in ASSET_CLASS_DEFAULTS:
        if n.startswith(prefix):
            return prefix
    return None


def for_symbol(cfg: dict, symbol: str) -> dict:
    """The config as seen by one symbol: asset-class defaults, then config symbol_overrides[symbol]."""
    cls = asset_class(symbol)
    ov = (cfg.get("symbol_overrides") or {}).get(symbol)
    if not cls and not ov:
        return cfg
    out = cfg
    if cls:
        out = _merge(out, ASSET_CLASS_DEFAULTS[cls])
    if ov:
        out = _merge(out, ov)
    return out


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    user: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    env_pw = os.environ.get("ASLI_MT5_PASSWORD")
    if env_pw:
        cfg["mt5"]["password"] = env_pw
    for key in ("data", "state", "reports"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    return cfg
