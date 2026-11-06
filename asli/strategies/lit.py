"""LIT (Liquidity Inducement Theorem): after a break of structure, wait for the first valid pullback
(inside bars ignored), let price take that pullback's extreme (the inducement, IDM), then buy/sell
at the extreme order block of the leg — which must have an FVG after it. Runs on M15."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import resample
from ..features import atr, confirmed_levels, fvgs
from .base import Context, Signal, Strategy

M15 = 3  # M5 bars per M15 bar
_M15 = pd.Timedelta(minutes=15)


class LITInducement(Strategy):
    name = "lit"

    def generate(self, ctx: Context) -> list[Signal]:
        m = resample(ctx.df, "15min")
        if len(m) < 100:
            return []
        o, h, l, c = (m[k].values for k in ("open", "high", "low", "close"))
        a = atr(m, 14)
        k = int(self.p.get("swing_k", 3))
        valid = int(self.p.get("order_valid_bars", 48)) * M15
        min_rr = float(ctx.cfg["risk"]["min_rr"])
        sh, sh_i, sl, sl_i = confirmed_levels(h, l, k)
        bl, _, sl_lo, _ = fvgs(h, l)
        n = len(m)
        out: list[Signal] = []
        setup = None  # dict with phase info
        last_bos_level = {+1: np.nan, -1: np.nan}
        for i in range(10, n):
            if not np.isfinite(a[i]):
                continue
            # ---------------------------------------------------------------- new BOS resets the setup
            for dirn in (+1, -1):
                lvl = sh[i - 1] if dirn == +1 else sl[i - 1]
                if not np.isfinite(lvl) or lvl == last_bos_level[dirn]:
                    continue
                broke = c[i] > lvl if dirn == +1 else c[i] < lvl
                if not broke:
                    continue
                last_bos_level[dirn] = lvl
                start = int(sl_i[i - 1] if dirn == +1 else sh_i[i - 1])
                if start < 0:
                    continue
                # leg extreme and its extreme order block (last opposite-colour candle at the leg origin)
                seg = slice(max(start - 3, 0), i + 1)
                ext_idx = int(np.argmin(l[seg]) if dirn == +1 else np.argmax(h[seg])) + seg.start
                ob = None
                for j in range(min(ext_idx + 1, i), max(ext_idx - 6, 0), -1):
                    if (dirn == +1 and c[j] < o[j]) or (dirn == -1 and c[j] > o[j]):
                        ob = j
                        break
                if ob is None:
                    continue
                has_fvg = any(np.isfinite(bl[x] if dirn == +1 else sl_lo[x]) for x in range(ob + 1, min(ob + 5, i + 1)))
                if not has_fvg:
                    continue
                setup = {"dir": dirn, "bos": i, "ob_lo": l[ob], "ob_hi": h[ob], "phase": "wait_pb",
                         "pb": np.nan, "leg_ext": h[i] if dirn == +1 else l[i], "idm": np.nan}
            if setup is None or i <= setup["bos"]:
                continue
            s = setup
            dirn = s["dir"]
            prev_h, prev_l = h[i - 1], l[i - 1]
            inside = h[i] <= prev_h and l[i] >= prev_l
            # invalidation: close beyond the order block
            if (dirn == +1 and c[i] < s["ob_lo"]) or (dirn == -1 and c[i] > s["ob_hi"]):
                setup = None
                continue
            if dirn == +1:
                if s["phase"] != "wait_idm":
                    s["leg_ext"] = max(s["leg_ext"], h[i])
                if s["phase"] == "wait_pb":
                    if not inside and l[i] < prev_l:
                        s["phase"], s["pb"] = "in_pb", l[i]
                elif s["phase"] == "in_pb":
                    s["pb"] = min(s["pb"], l[i])
                    if not inside and h[i] > prev_h:
                        s["phase"], s["idm"] = "wait_idm", s["pb"]
                elif s["phase"] == "wait_idm":
                    s["leg_ext"] = max(s["leg_ext"], h[i])
                    if l[i] < s["idm"]:
                        out.extend(self._emit(ctx, m, i, s, a[i], valid, min_rr))
                        setup = None
            else:
                if s["phase"] != "wait_idm":
                    s["leg_ext"] = min(s["leg_ext"], l[i])
                if s["phase"] == "wait_pb":
                    if not inside and h[i] > prev_h:
                        s["phase"], s["pb"] = "in_pb", h[i]
                elif s["phase"] == "in_pb":
                    s["pb"] = max(s["pb"], h[i])
                    if not inside and l[i] < prev_l:
                        s["phase"], s["idm"] = "wait_idm", s["pb"]
                elif s["phase"] == "wait_idm":
                    s["leg_ext"] = min(s["leg_ext"], l[i])
                    if h[i] > s["idm"]:
                        out.extend(self._emit(ctx, m, i, s, a[i], valid, min_rr))
                        setup = None
        return out

    def _emit(self, ctx, m, i, s, a_i, valid, min_rr):
        dirn = s["dir"]
        bar = ctx.m5_bar_at_close(m.index[i] + _M15)
        if bar < 0 or bar >= len(ctx.df):
            return []
        sp = ctx.spread[bar]
        if dirn == +1:
            entry, sl, tp = s["ob_hi"], s["ob_lo"] - 0.1 * a_i - sp, s["leg_ext"]
        else:
            entry, sl, tp = s["ob_lo"], s["ob_hi"] + 0.1 * a_i + sp, s["leg_ext"]
        risk = abs(entry - sl)
        if risk <= 0 or (tp - entry) * dirn / risk < min_rr:
            return []
        return [Signal(bar=bar, time=ctx.df.index[bar], symbol=ctx.symbol, strategy=self.name, direction=dirn,
                       order="limit", entry=entry, sl=sl, tp=tp, valid_bars=valid, max_hold_bars=288,
                       features={"idm_taken": True, "ob_size_atr": float((s["ob_hi"] - s["ob_lo"]) / a_i),
                                 "key_level": False})]
