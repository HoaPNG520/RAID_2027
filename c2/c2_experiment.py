#!/usr/bin/env python3
"""
C2: quantitative comparison of missing-data strategies for motif discovery (Appendix H4).

Methods
  exclusion     : SCAMP/STUMPY semantics -- only pairs of complete windows are scored
  linear        : linear interpolation, then the exact matrix profile
  spline        : cubic spline through the observed points, then the exact profile
  locf          : last observation carried forward (first values backfilled)
  mdms_window   : MDMS lower bound, per-window observed limits (MDMS paper practice)
  mdms_declared : MDMS lower bound, declared limits = true range of the complete series
  mdms_ours     : proposed fix -- declared range intersected with a declared slew-rate limit,
                  with the optimal-centre variance bound (both declared from the complete series)

Metrics (all relative to the oracle = exact profile of the complete series)
  top1_hit       : the method's top-1 motif pair matches the oracle top-1 pair (tolerance m/4)
  topk_recall    : share of the oracle top-k motif pairs found among the method's top-k (k=3)
  planted_hit    : (planted series) top-1 pair = two planted occurrences
  coverage       : share of windows with a finite nearest-neighbour value
  mdms_* only    : guarantee_holds -- the oracle top-1 pair passes the admissible screen
                   (LB <= true top-1 distance); candidates -- pairs that pass that screen
                   (the verification workload the guarantee implies)

Usage: python3 -I c2/c2_experiment.py --data-dir DATA --out results/c2.csv [--n 2000 --m 100 --seeds 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mdms_ref as M  # noqa: E402

K_TOP = 3


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
def bump(m, rng):
    t = np.linspace(-1, 1, m)
    return np.exp(-1.5 * t ** 2) * np.sin(2 * np.pi * 2.5 * t) * (1 + 0.05 * rng.standard_normal())


def planted(background, m, rng, n_occ=3, amp=6.0):
    """Plant n_occ noisy copies of one shape; return (series, start positions)."""
    T = background.copy()
    n = len(T)
    shape = bump(m, rng)
    starts = []
    while len(starts) < n_occ:
        s = int(rng.integers(0, n - m))
        if all(abs(s - q) > 2 * m for q in starts):
            starts.append(s)
    scale = amp * np.std(np.diff(background)) * np.sqrt(m)
    for s in sorted(starts):
        T[s:s + m] += scale * (shape + 0.02 * rng.standard_normal(m))
    return T, sorted(starts)


def load_series(data_dir, n, seed=0):
    out = {}
    names = [("cwru_IR007", "1797_IR_7_DE12.npz"), ("cwru_OR007", "1797_OR@6_7_DE12.npz"),
             ("cwru_B007", "1797_B_7_DE12.npz"), ("cwru_normal", "1797_Normal.npz")]
    for name, fn in names:
        p = os.path.join(data_dir, fn)
        if os.path.exists(p):
            de = np.load(p, allow_pickle=False)["DE"].ravel().astype(np.float64)
            out[name] = (de[20_000:20_000 + n].copy(), None)   # different segment from C3
    rng = np.random.default_rng(seed)
    m = 100
    rw = np.cumsum(rng.standard_normal(n))
    out["planted_random_walk"] = planted(rw, m, rng)
    sn = np.sin(2 * np.pi * np.arange(n) / 37) + 0.5 * rng.standard_normal(n)
    out["planted_sine_noise"] = planted(sn, m, rng)
    if "cwru_normal" in out:
        p = os.path.join(data_dir, "1797_Normal.npz")
        bg = np.load(p, allow_pickle=False)["DE"].ravel().astype(np.float64)[60_000:60_000 + n]
        out["planted_cwru_normal"] = planted(bg, m, rng)
    return out


def masks_for(Tc, m, seeds):
    n = len(Tc)
    res = []
    for s in range(seeds):
        for rate in (0.05, 0.10, 0.20, 0.40):
            res.append((f"random_{int(rate*100)}", s, M.generate_missing_mask(n, m, rate, "random", seed=s)[0]))
        for rate in (0.10, 0.20, 0.40):
            for bf in (0.10, 0.30, 0.50):
                res.append((f"block_{int(rate*100)}_b{int(bf*100)}", s,
                            M.generate_missing_mask(n, m, rate, "block", bf, seed=s)[0]))
    for q in (0.99, 0.98, 0.95):
        res.append((f"clip_q{int(q*100)}", 0, M.clipping_mask(Tc, q)[0]))
    return res


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def _upd(P2, I, rows, cols, d2):
    better = (d2 < P2[rows]) | ((d2 == P2[rows]) & np.isfinite(d2) & (cols < I[rows]))
    P2[rows[better]] = d2[better]
    I[rows[better]] = cols[better]


def exact_profile(Tc, m, excl=0.25, pair_filter=None):
    os_ = M.oracle_stats(Tc, m)
    L = os_["L"]
    P2 = np.full(L, np.inf)
    I = np.full(L, -1, dtype=np.int64)
    for k in range(M.excl_zone(m, excl) + 1, L):
        d2 = M.oracle_diagonal_d2(os_, k)
        ii = np.arange(L - k)
        jj = ii + k
        if pair_filter is not None:
            d2 = np.where(pair_filter[ii] & pair_filter[jj], d2, np.inf)
        _upd(P2, I, ii, jj, d2)
        _upd(P2, I, jj, ii, d2)
    return P2, I


def impute(Tc, mask, kind):
    x = np.arange(len(Tc))
    xo, yo = x[mask], Tc[mask]
    if kind == "linear":
        return np.interp(x, xo, yo)
    if kind == "spline":
        y = CubicSpline(xo, yo)(x)
        y[:xo[0]] = yo[0]
        y[xo[-1] + 1:] = yo[-1]
        return y
    if kind == "locf":
        idx = np.where(mask, x, 0)
        np.maximum.accumulate(idx, out=idx)
        y = Tc[idx].copy()
        y[:xo[0]] = yo[0]
        return y
    raise ValueError(kind)


def top_motifs(P2, I, m, k=K_TOP):
    """Greedy non-overlapping motif pairs from a profile (both members at least m/2 apart
    from every member already chosen)."""
    order = np.argsort(P2, kind="stable")
    chosen = []
    sep = m // 2
    for i in order:
        if not np.isfinite(P2[i]) or I[i] < 0:
            break
        a, b = int(i), int(I[i])
        if any(min(abs(a - c), abs(b - c)) < sep for pair in chosen for c in pair):
            continue
        chosen.append((min(a, b), max(a, b)))
        if len(chosen) == k:
            break
    return chosen


def match(p, q, tol):
    return (abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol) or \
           (abs(p[0] - q[1]) <= tol and abs(p[1] - q[0]) <= tol)


def mdms_pass(Tc, mask, m, vlimits, limits, d_thr2, pair, excl=0.25, opt_center=False):
    """LB profile, the number of pairs passing the admissible screen LB^2 <= d_thr2, and the
    rank of `pair` (the oracle top-1) when all pairs are ordered by their lower bound."""
    ws = M.window_stats(np.where(mask, Tc, 0.0), mask, m, vlimits, limits, opt_center)
    lb_pair = M.pair_lb_d2(ws, *pair)
    below = 0
    L = ws["L"]
    P2 = np.full(L, np.inf)
    I = np.full(L, -1, dtype=np.int64)
    cand = 0
    for k in range(M.excl_zone(m, excl) + 1, L):
        d2 = M.diagonal_lb_d2(ws, k)
        ii = np.arange(L - k)
        jj = ii + k
        cand += int((d2 <= d_thr2).sum())
        below += int((d2 < lb_pair).sum())
        _upd(P2, I, ii, jj, d2)
        _upd(P2, I, jj, ii, d2)
    return P2, I, ws, cand, below + 1, lb_pair


# --------------------------------------------------------------------------- #
def evaluate(name, Tc, starts, mname, seed, mask, m, oracle, tol):
    P2o, Io, top_o = oracle
    a, b = top_o[0]
    d_top1 = float(P2o[a])                                 # true top-1 squared distance
    lo, hi = float(Tc.min()), float(Tc.max())
    complete = np.lib.stride_tricks.sliding_window_view(mask, m).all(axis=1)
    rows = []

    def record(method, P2, I, extra=None):
        mot = top_motifs(P2, I, m)
        r = dict(series=name, mask=mname, seed=seed, method=method,
                 missing_rate=float((~mask).mean()), frac_windows_incomplete=float((~complete).mean()),
                 coverage=float(np.isfinite(P2).mean()),
                 top1_hit=bool(mot and match(mot[0], top_o[0], tol)),
                 topk_recall=float(np.mean([any(match(q, p, tol) for p in mot) for q in top_o])) if top_o else np.nan,
                 n_motifs_found=len(mot))
        if starts is not None:
            r["planted_hit"] = bool(mot and any(abs(mot[0][0] - s) <= tol for s in starts)
                                    and any(abs(mot[0][1] - s) <= tol for s in starts))
        if extra:
            r.update(extra)
        rows.append(r)

    record("exclusion", *exact_profile(Tc, m, pair_filter=complete))
    for kind in ("linear", "spline", "locf"):
        record(kind, *exact_profile(impute(Tc, mask, kind), m))
    slew = float(np.max(np.abs(np.diff(Tc))))
    for method, vl, lim in (("mdms_window", "window", None), ("mdms_declared", "declared", (lo, hi)),
                            ("mdms_ours", "slew", (slew, lo, hi))):
        P2, I, ws, cand, rank, lb_top1 = mdms_pass(Tc, mask, m, vl, lim,
                                                    d_top1 * (1 + 1e-9) + 1e-9, (a, b),
                                                    opt_center=method == "mdms_ours")
        record(method, P2, I, dict(guarantee_holds=bool(lb_top1 <= d_top1 * (1 + 1e-9) + 1e-9),
                                   candidates=cand, oracle_top1_lb_rank=rank,
                                   total_pairs=int((ws["L"] - M.excl_zone(m) - 1) * (ws["L"] - M.excl_zone(m)) / 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    series = load_series(a.data_dir, a.n)
    if a.only:
        series = {k: v for k, v in series.items() if k in a.only.split(",")}
    tol = a.m // 4
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    first = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as fh:
        wr = None
        for name, (Tc, starts) in series.items():
            P2o, Io = exact_profile(Tc, a.m)
            top_o = top_motifs(P2o, Io, a.m)
            meta = dict(series=name, oracle_top=top_o, planted_starts=starts)
            print(json.dumps(meta), flush=True)
            for mname, seed, mask in masks_for(Tc, a.m, a.seeds):
                rows = evaluate(name, Tc, starts, mname, seed, mask, a.m, (P2o, Io, top_o), tol)
                for r in rows:
                    if wr is None:
                        keys = ["series", "mask", "seed", "method", "missing_rate",
                                "frac_windows_incomplete", "coverage", "top1_hit", "topk_recall",
                                "n_motifs_found", "planted_hit", "guarantee_holds", "candidates",
                                "oracle_top1_lb_rank", "total_pairs"]
                        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="raise")
                        if first:
                            wr.writeheader()
                    wr.writerow({k: r.get(k, "") for k in wr.fieldnames})
                fh.flush()
                print(f"[{time.time()-t0:7.1f}s] {name} {mname} s{seed}: " +
                      " ".join(f"{r['method']}={int(r['top1_hit'])}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
