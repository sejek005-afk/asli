"""ICT Model 2022 / Silver-Bullet style: liquidity raid -> displacement MSS -> limit entry in the FVG,
inside the London and New York killzones. Runs on M5."""
from __future__ import annotations

import numpy as np

from ..features import body_strength, confirmed_levels, fvgs
from ..timeutil import KILLZONES
from .base import Context, Signal, Strategy, nearest_target


class ICTModel2022(Strategy):
    name = "ict"

    def generate(self, ctx: Context) -> list[Signal]:
        df, d = ctx.df, ctx.daily
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        a = ctx.atr
        n = len(df)
        arm_bars = int(self.p.get("arm_bars", 24))
        valid = int(self.p.get("order_valid_bars", 12))
        disp_atr = float(self.p.get("disp_atr", 1.0))
        min_rr = float(ctx.cfg["risk"]["min_rr"])
        big_sh, _, big_sl, _ = confirmed_levels(h, l, 6)  # liquidity pools
        ms_sh, _, ms_sl, _ = confirmed_levels(h, l, 2)  # structure for MSS
        bl, bh, sl_lo, sl_hi = fvgs(h, l)
        bstr, _ = body_strength(df, a)
        hr = d.ny_hour

        def kz_name(x):
            for z, (s, e) in KILLZONES.items():
                if s <= x < e:
                    return z
            return None

        out: list[Signal] = []
        used = set()
        # state per direction: [armed_until, raid_idx, raid_ext, raid_level, level_name, mss_ref]
        st = {+1: None, -1: None}
        for i in range(30, n - 1):
            if not np.isfinite(a[i]) or a[i] <= 0:
                continue
            # ---- detect raids (sell-side raid arms a LONG, buy-side raid arms a SHORT)
            above = [("pdh", d.pdh[i]), ("asia_high", d.asia_high[i]), ("swing_high", big_sh[i - 1])]
            below = [("pdl", d.pdl[i]), ("asia_low", d.asia_low[i]), ("swing_low", big_sl[i - 1])]
            for name, lv in above:
                if np.isfinite(lv) and h[i] > lv and c[i] < lv:
                    cur = st[-1]
                    if cur is None or i > cur[0]:
                        st[-1] = [i + arm_bars, i, h[i], lv, name, ms_sl[i]]
                    else:
                        cur[2] = max(cur[2], h[i])
                    break
            for name, lv in below:
                if np.isfinite(lv) and l[i] < lv and c[i] > lv:
                    cur = st[+1]
                    if cur is None or i > cur[0]:
                        st[+1] = [i + arm_bars, i, l[i], lv, name, ms_sh[i]]
                    else:
                        cur[2] = min(cur[2], l[i])
                    break
            kz = kz_name(hr[i])
            for dirn in (+1, -1):
                s = st[dirn]
                if s is None or i > s[0]:
                    st[dirn] = None
                    continue
                until, ridx, rext, rlevel, lname, ref = s
                # raid extends / fails
                if dirn == -1:
                    if h[i] > rext:
                        s[2] = rext = h[i]
                    if c[i] > rlevel and i > ridx:
                        st[dirn] = None
                        continue
                else:
                    if l[i] < rext:
                        s[2] = rext = l[i]
                    if c[i] < rlevel and i > ridx:
                        st[dirn] = None
                        continue
                if i <= ridx or kz is None or not np.isfinite(ref):
                    continue
                shifted = (c[i] < ref) if dirn == -1 else (c[i] > ref)
                if not shifted:
                    continue
                disp = float(np.max(bstr[ridx + 1:i + 1])) if i > ridx else 0.0
                if disp < disp_atr:
                    continue
                # most recent FVG in the shift leg, in trade direction
                zone = None
                for j in range(i, ridx, -1):
                    if dirn == -1 and np.isfinite(sl_lo[j]):
                        zone = (sl_lo[j], sl_hi[j]); break
                    if dirn == +1 and np.isfinite(bl[j]):
                        zone = (bl[j], bh[j]); break
                st[dirn] = None
                if zone is None:
                    continue
                key = (d.day[i], kz, dirn)
                if key in used:
                    continue
                entry = (zone[0] + zone[1]) / 2.0
                buf = 0.1 * a[i] + ctx.spread[i]
                sl = rext + buf if dirn == -1 else rext - buf
                risk = abs(entry - sl)
                if risk <= 0:
                    continue
                if dirn == -1:
                    levels = [d.pdl[i], d.asia_low[i], big_sl[i]]
                else:
                    levels = [d.pdh[i], d.asia_high[i], big_sh[i]]
                tp = nearest_target(levels, entry, risk, dirn, min_rr)
                if tp is None:
                    tp = entry + dirn * min_rr * risk
                used.add(key)
                out.append(Signal(
                    bar=i, time=df.index[i], symbol=ctx.symbol, strategy=self.name, direction=dirn,
                    order="limit", entry=entry, sl=sl, tp=tp, valid_bars=valid, max_hold_bars=144,
                    features={"killzone": kz, "raid_level": lname,
                              "sweep_depth_atr": float(abs(rext - rlevel) / a[i]), "disp_atr": disp,
                              "key_level": lname in ("pdh", "pdl", "asia_high", "asia_low")},
                ))
        return out
