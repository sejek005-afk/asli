"""Round 9 research: quarterly walk-forward predictions (see r9_plan.md).

    python r9_run.py <gold_m5.csv.gz> <euro_m5.csv.gz> <work_dir> [learner ...]

Writes <work_dir>/data_<symbol>.npz (features + labels) and <work_dir>/pred_<learner>_<symbol>_<h>.npz.
Finished files are skipped, so the run can be resumed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r9_lib import HORIZONS, barrier_labels, build_features, feature_cols, load_raw, pattern_cols  # noqa: E402

SYMS = ("XAUUSD_o", "EURUSD_o")
FIRST_TEST = pd.Timestamp("2022-01-01", tz="UTC")
LAST = pd.Timestamp("2026-11-07", tz="UTC")
SUB = 3
KNN_K = 300
PCA_N = 12
MAXH = max(HORIZONS)


def gbt() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                         l2_regularization=1.0, max_bins=63, early_stopping=False, random_state=9)


def quarters() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, a = [], FIRST_TEST
    while a < LAST:
        b = a + pd.DateOffset(months=3)
        out.append((a, min(b, LAST)))
        a = b
    return out


def prepare(paths: dict[str, str], work: Path) -> dict[str, dict]:
    raw = {s: load_raw(paths[s], s) for s in SYMS}
    data = {}
    for s in SYMS:
        other = raw[SYMS[1 - SYMS.index(s)]]
        X = build_features(raw[s], other)
        cols = feature_cols(X)
        d = {"index": raw[s].index, "X": X[cols].to_numpy(np.float32), "cols": cols,
             "pat_idx": np.array([cols.index(k) for k in pattern_cols(X)]), "atr": X["atr"].to_numpy()}
        for h in HORIZONS:
            lab = barrier_labels(raw[s], d["atr"], h, s)
            for k, v in lab.items():
                d[f"{k}_{h}"] = v
        data[s] = d
        np.savez_compressed(work / f"labels_{s}.npz", t=raw[s].index.asi8,
                            **{k: v for k, v in d.items() if k.split("_")[0] in ("r", "off", "mkt", "cost", "S")})
        print(s, "rows", len(X), "features", len(cols), "pattern features", len(d["pat_idx"]), flush=True)
    return data


def train_rows(d: dict, start_pos: int, h: int, since: pd.Timestamp | None) -> np.ndarray:
    """Subsampled training positions whose label window ends before the test block."""
    last = start_pos - h - 1
    pos = np.arange(0, max(last, 0), SUB)
    ok = np.isfinite(d["atr"][pos]) & np.isfinite(d[f"r_long_{h}"][pos]) & np.isfinite(d[f"r_short_{h}"][pos])
    if since is not None:
        ok &= d["index"][pos] >= since
    return pos[ok]


def run_gbt(data: dict, work: Path, learner: str) -> None:
    for h in HORIZONS:
        outs = {s: work / f"pred_{learner}_{s}_{h}.npz" for s in SYMS}
        if all(p.exists() for p in outs.values()):
            continue
        pred = {s: {"long": np.full(len(data[s]["index"]), np.nan, np.float32),
                    "short": np.full(len(data[s]["index"]), np.nan, np.float32)} for s in SYMS}
        t0 = time.time()
        for a, b in quarters():
            since = a - pd.DateOffset(months=24) if learner == "gbt24" else None
            tr, te = {}, {}
            for s in SYMS:
                d = data[s]
                p0, p1 = d["index"].searchsorted(a), d["index"].searchsorted(b)
                tr[s] = train_rows(d, p0, h, since)
                tp = np.arange(p0, p1)
                te[s] = tp[np.isfinite(d["atr"][tp])]
            groups = [SYMS] if learner == "gbtpool" else [(s,) for s in SYMS]
            for grp in groups:
                def mat(rows: dict) -> np.ndarray:
                    parts = []
                    for s in grp:
                        x = data[s]["X"][rows[s]]
                        if learner == "gbtpool":
                            x = np.c_[x, np.full(len(x), float(SYMS.index(s)), np.float32)]
                        parts.append(x)
                    return np.vstack(parts)
                Xtr, Xte = mat(tr), mat(te)
                for side in ("long", "short"):
                    y = np.concatenate([data[s][f"r_{side}_{h}"][tr[s]] for s in grp])
                    m = gbt().fit(Xtr, y)
                    p = m.predict(Xte).astype(np.float32)
                    k = 0
                    for s in grp:
                        pred[s][side][te[s]] = p[k:k + len(te[s])]
                        k += len(te[s])
            print(learner, h, str(a.date()), "train", {s: len(tr[s]) for s in SYMS}, f"{time.time() - t0:.0f}s", flush=True)
        for s in SYMS:
            np.savez_compressed(outs[s], long=pred[s]["long"], short=pred[s]["short"])


def run_knn(data: dict, work: Path) -> None:
    for s in SYMS:
        outs = {h: work / f"pred_knn_{s}_{h}.npz" for h in HORIZONS}
        if all(p.exists() for p in outs.values()):
            continue
        d = data[s]
        P = d["X"][:, d["pat_idx"]]
        n = len(d["index"])
        pred = {h: {"long": np.full(n, np.nan, np.float32), "short": np.full(n, np.nan, np.float32)} for h in HORIZONS}
        t0 = time.time()
        for a, b in quarters():
            p0, p1 = d["index"].searchsorted(a), d["index"].searchsorted(b)
            tr = train_rows(d, p0, MAXH, None)  # one neighbour set for all horizons: purge by the longest
            tp = np.arange(p0, p1)
            te = tp[np.isfinite(d["atr"][tp])]
            mu = np.nanmean(P[tr], axis=0)
            sd = np.nanstd(P[tr], axis=0) + 1e-9
            z = lambda rows: np.clip(np.nan_to_num((P[rows] - mu) / sd, nan=0.0), -5, 5)  # noqa: E731
            pca = PCA(n_components=PCA_N, random_state=9).fit(z(tr))
            Ztr, Zte = pca.transform(z(tr)), pca.transform(z(te))
            nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="kd_tree", leaf_size=40, n_jobs=2).fit(Ztr)
            for c0 in range(0, len(te), 20000):
                idx = nn.kneighbors(Zte[c0:c0 + 20000], return_distance=False)
                rows = te[c0:c0 + 20000]
                for h in HORIZONS:
                    for side in ("long", "short"):
                        y = d[f"r_{side}_{h}"][tr]
                        pred[h][side][rows] = y[idx].mean(axis=1)
            print("knn", s, str(a.date()), "train", len(tr), f"{time.time() - t0:.0f}s", flush=True)
        for h in HORIZONS:
            np.savez_compressed(outs[h], long=pred[h]["long"], short=pred[h]["short"])


def main() -> int:
    work = Path(sys.argv[3])
    work.mkdir(parents=True, exist_ok=True)
    learners = sys.argv[4:] or ["gbtall", "gbt24", "gbtpool", "knn"]
    data = prepare({"XAUUSD_o": sys.argv[1], "EURUSD_o": sys.argv[2]}, work)
    for ln in learners:
        if ln == "knn":
            run_knn(data, work)
        else:
            run_gbt(data, work, ln)
    print("R9 RUN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
