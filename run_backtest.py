"""Backtest all 5 styles + the ASLI fusion, learn session/hour edges, train the learner, write the report.

Usage (on the Windows server, MT5 open and logged in):
    python run_backtest.py                 # history from MetaTrader 5 (default 5 years)
    python run_backtest.py --years 3
    python run_backtest.py --csv data/csv  # CSV files named EURUSD.csv ... (time,open,high,low,close[,spread])
    python run_backtest.py --synthetic     # pipeline self-test on random data (results are meaningless)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from asli import fusion, report
from asli.config import load_config
from asli.data import (SymbolMeta, detect_server_time_mode, load_cache, load_csv, mt5_connect, mt5_history,
                       mt5_symbol_meta, save_cache, synthetic)

log = logging.getLogger("asli")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv-time", default="utc", help="utc | ny_plus_7 | fixed:<h> (clock of the CSV times)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore cached history and download again")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    symbols = args.symbols or cfg["symbols"]
    frames, metas = {}, {}
    if args.synthetic:
        for i, s in enumerate(symbols):
            frames[s] = synthetic(days=400, seed=11 + i)
            metas[s] = SymbolMeta.guess(s)
        log.warning("SYNTHETIC DATA — results only prove the pipeline runs, not that anything works.")
    elif args.csv:
        for s in symbols:
            meta = SymbolMeta.guess(s)
            frames[s] = load_csv(str(Path(args.csv) / f"{s}.csv"), args.csv_time, pip=meta.pip)
            metas[s] = meta
    else:
        mt5 = mt5_connect(cfg)
        mode = cfg["mt5"]["server_time"]
        if mode == "auto":
            mode = detect_server_time_mode(mt5, symbols[0])
        log.info("broker server time mode: %s", mode)
        for s in symbols:
            metas[s] = mt5_symbol_meta(mt5, s)
            df = None if args.refresh else load_cache(cfg["paths"]["data"], s)
            if df is None:
                log.info("downloading %s M5 history (%s years)...", s, args.years)
                df = mt5_history(mt5, s, args.years, mode)
                save_cache(df, cfg["paths"]["data"], s)
            frames[s] = df
            log.info("%s: %d bars %s -> %s", s, len(df), df.index[0], df.index[-1])
    res = fusion.research(frames, metas, cfg)
    if "error" in res:
        log.error("research failed: %s", res["error"])
        return 1
    state = Path(cfg["paths"]["state"])
    res["learner"].save(state / "learner.json")
    meta = {"symbols": list(frames), "start": str(min(f.index[0] for f in frames.values())),
            "end": str(max(f.index[-1] for f in frames.values())),
            "bars": int(sum(len(f) for f in frames.values())), "synthetic": bool(args.synthetic)}
    summary = report.build(res, cfg["paths"]["reports"], meta)
    (state / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                 encoding="utf-8")
    print("\n=== out-of-sample results (R) ===")
    for k, v in summary["per_strategy"].items():
        print(f"{k:10s} trades={v['trades']:4d} win={v['win_rate']:.0%} avgR={v['avg_r']:+.3f} PF={v['profit_factor']:.2f}")
    f = summary["fusion_test"]
    print(f"{'FUSION':10s} trades={f['trades']:4d} win={f['win_rate']:.0%} avgR={f['avg_r']:+.3f} PF={f['profit_factor']:.2f}")
    print(f"report: {Path(cfg['paths']['reports']) / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
