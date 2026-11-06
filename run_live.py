"""Start ASLI live: connects to MetaTrader 5, trades the fusion style, serves the dashboard/API.

    python run_live.py            # uses config.yaml (mode demo by default)

Run run_backtest.py first so the learner starts from what the backtest learned.
"""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from asli.broker import MT5Broker
from asli.config import ROOT, load_config
from asli.data import detect_server_time_mode, mt5_connect
from asli.live import Engine
from asli.server import serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--poll", type=float, default=10.0)
    args = ap.parse_args(argv)
    (ROOT / "logs").mkdir(exist_ok=True)
    handlers = [logging.StreamHandler(),
                RotatingFileHandler(ROOT / "logs" / "asli.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    cfg = load_config(args.config)
    if cfg["mode"] not in ("demo", "real"):
        logging.error("config mode must be demo or real for live trading (now: %s)", cfg["mode"])
        return 2
    if str(cfg["server"]["token"]) in ("", "change-me"):
        logging.error("set server.token in config.yaml (a long random string) before starting")
        return 2
    if not (Path(cfg["paths"]["state"]) / "learner.json").exists():
        logging.warning("no trained learner found — run run_backtest.py first; starting untrained")
    import os
    (Path(cfg["paths"]["state"]) / "live.pid").write_text(str(os.getpid()))
    mt5 = mt5_connect(cfg)
    mode = cfg["mt5"]["server_time"]
    if mode == "auto":
        mode = detect_server_time_mode(mt5, cfg["symbols"][0])
    logging.info("broker server time mode: %s", mode)
    broker = MT5Broker(mt5, cfg)
    engine = Engine(cfg, broker, mt5, mode)
    serve(engine, cfg)
    engine.run_forever(args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
