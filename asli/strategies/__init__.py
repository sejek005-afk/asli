from .base import Context, Signal, Strategy
from .crt import CandleRangeTheory
from .ict import ICTModel2022
from .lit import LITInducement
from .orb import OpeningRangeBreakout
from .trendline import TrendlineLiquidity

ALL = {
    "ict": ICTModel2022,
    "lit": LITInducement,
    "trendline": TrendlineLiquidity,
    "crt": CandleRangeTheory,
    "orb": OpeningRangeBreakout,
}


def build(cfg: dict) -> list[Strategy]:
    out = []
    for name, cls in ALL.items():
        p = cfg["strategies"].get(name, {})
        if p.get("enabled", True):
            out.append(cls(p))
    return out


__all__ = ["Context", "Signal", "Strategy", "ALL", "build"]
