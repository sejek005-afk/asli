"""Round 9 research: export broker candles (M1 + M5, gold and euro) to compact .npz files.

Read-only: it only downloads price history from the MetaTrader 5 terminal the bot already uses.
It never opens, changes or closes a trade.

Run on the trading server:
    C:\asli\.venv\Scripts\python.exe r9_export.py [out_dir] [copy_dir]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ASLI = Path(r"C:\asli")
sys.path.insert(0, str(ASLI))

from asli.config import load_config  # noqa: E402
from asli.data import detect_server_time_mode, mt5_connect, mt5_history, mt5_symbol_meta  # noqa: E402

SYMBOLS = ["XAUUSD_o", "EURUSD_o"]
FRAMES = [("M1", 6.2), ("M5", 6.2)]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\asli_r9\out")
    copy_to = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ASLI / "config.yaml"))
    mt5 = mt5_connect(cfg)
    mode = cfg["mt5"]["server_time"]
    if mode == "auto":
        mode = detect_server_time_mode(mt5, SYMBOLS[0])
    info = {"time_mode": mode, "exported_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "symbols": {}}
    for sym in SYMBOLS:
        meta = mt5_symbol_meta(mt5, sym)
        si = mt5.symbol_info(sym)
        info["symbols"][sym] = {
            "digits": meta.digits, "point": meta.point, "pip": meta.pip, "tick_value": meta.tick_value,
            "tick_size": meta.tick_size, "volume_min": meta.volume_min, "volume_step": meta.volume_step,
            "contract_size": getattr(si, "trade_contract_size", None), "spread_now_points": getattr(si, "spread", None),
            "frames": {},
        }
        for tf, years in FRAMES:
            t0 = time.time()
            df = mt5_history(mt5, sym, years, mode, tf)
            pt = meta.point
            t = ((df.index.tz_convert("UTC").tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)
            arr = {k: np.rint(df[k].to_numpy() / pt).astype(np.int32) for k in ("open", "high", "low", "close")}
            sp = np.clip(np.rint(df["spread"].to_numpy() / pt), 0, 65535).astype(np.uint16)
            f = out / f"{sym}_{tf}.npz"
            np.savez_compressed(f, t=t, o=arr["open"], h=arr["high"] - arr["open"], l=arr["open"] - arr["low"],
                                c=arr["close"] - arr["open"], s=sp)
            info["symbols"][sym]["frames"][tf] = {
                "bars": int(len(df)), "first": str(df.index[0]), "last": str(df.index[-1]),
                "bytes": f.stat().st_size, "seconds": round(time.time() - t0, 1),
            }
            print(sym, tf, len(df), df.index[0], "->", df.index[-1], f.stat().st_size, "bytes", flush=True)
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    try:
        mt5.shutdown()
    except Exception:
        pass
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        for f in sorted(out.iterdir()):
            shutil.copy2(f, copy_to / f.name)
            print("copied", f.name, flush=True)
    print("R9 EXPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
