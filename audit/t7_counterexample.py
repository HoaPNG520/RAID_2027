#!/usr/bin/env python3
"""
T7: adversarial search for window-mode admissibility violations (Appendix H2/H3).

Part A: a hand-built family (a spike hidden at the same position in both windows) and
        how the violation grows with the spike height.
Part B: random search over small windows for the smallest m with a violation.

The lower bound here is an INDEPENDENT implementation written from the derivation
(midrange form of the variance bound), not from the six-sum code in mdms_ref, so
agreement between the two is itself a check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mdms_ref as M  # noqa: E402


def lb_direct(x, bx, y, by, lim_x=None, lim_y=None):
    """Squared MDMS lower bound for one pair, written directly from the derivation.
    lim_* = (lo, hi) bounds for the missing values; None -> window's observed range."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx, by = np.asarray(bx, bool), np.asarray(by, bool)
    if not bx.any() or not by.any():
        return np.inf
    if (bx.all() and x.var() <= 1e-12) or (by.all() and y.var() <= 1e-12):
        return np.inf
    R = bx & by
    r = int(R.sum())
    if r == 0:
        return 0.0
    xr, yr = x[R], y[R]
    vx, vy = xr.var(), yr.var()
    q = 0.0 if vx * vy <= 1e-24 else float(np.clip(np.mean((xr - xr.mean()) * (yr - yr.mean()))
                                                   / np.sqrt(vx * vy), -1, 1))
    if r == 2 and vx * vy > 1e-24:
        q = float(np.sign(np.mean((xr - xr.mean()) * (yr - yr.mean()))))
    corr = 1 - q * q if q > 0 else 1.0
    if bx.all() and by.all():
        zx = (x - x.mean()) / x.std()
        zy = (y - y.mean()) / y.std()
        return float(np.sum((zx - zy) ** 2))
    if by.all():
        return r * vy / y.var() * corr
    if bx.all():
        return r * vx / x.var() * corr

    def smax2(v, bv, lim):
        obs = v[bv]
        lo, hi = (obs.min(), obs.max()) if lim is None else lim
        c = 0.5 * (lo + hi)
        return (np.sum((obs - c) ** 2) + (len(v) - len(obs)) * (0.5 * (hi - lo)) ** 2) / len(v)

    # spec rule: a zero variance bound contributes 0 (Fden <= eps -> inv_den = 0)
    sx, sy = smax2(x, bx, lim_x), smax2(y, by, lim_y)
    fx = vx / sx if sx > 1e-12 else 0.0
    fy = vy / sy if sy > 1e-12 else 0.0
    return r * corr * max(fx, fy)


def true_d2(x, y):
    return M.two_window_true_d2(x, y)


def part_a():
    base = np.array([0.0, 1.0, 0.0, 6.0, 0.0, -1.0, 0.0, 0.5])
    noise = np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.1, -0.1, 0.0])
    bx = np.ones(8, bool)
    bx[3] = False
    rows = []
    for h in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]:
        x = base.copy()
        x[3] = h
        y = x + noise
        tr = true_d2(x, y)
        lw = lb_direct(x, bx, y, bx)
        ld = lb_direct(x, bx, y, bx, (-12, 12), (-12, 12))
        lref = M.two_window_lb_d2(x, bx, y, bx, "window")
        rows.append(dict(spike=h, true_d=float(np.sqrt(tr)), lb_window=float(np.sqrt(lw)),
                         lb_window_ref=float(np.sqrt(lref)), lb_declared=float(np.sqrt(ld)),
                         ratio_window=float(np.sqrt(lw / tr)), violation=bool(lw > tr * (1 + 1e-9))))
    return dict(x_base=base.tolist(), noise=noise.tolist(), missing_position=3, rows=rows)


def part_b(seed=0, trials=20000, ms=(3, 4, 5, 6, 8)):
    rng = np.random.default_rng(seed)
    out = {}
    for m in ms:
        best = None
        nviol = 0
        for _ in range(trials):
            x = rng.standard_normal(m) * rng.uniform(0.3, 3)
            y = x + rng.standard_normal(m) * rng.uniform(0.01, 1.0)   # near-motif pairs
            bx = rng.random(m) > rng.uniform(0.1, 0.5)
            by = rng.random(m) > rng.uniform(0.1, 0.5)
            if bx.all() or by.all() or bx.sum() < 2 or by.sum() < 2:
                continue
            # hide an extreme: put the missing positions far outside the observed range
            amp = rng.uniform(1, 8)
            x[~bx] = x[bx].max() + amp * np.abs(rng.standard_normal((~bx).sum()))
            y[~by] = y[by].max() + amp * np.abs(rng.standard_normal((~by).sum()))
            tr = true_d2(x, y)
            if not np.isfinite(tr) or tr <= 1e-12:
                continue
            lw = lb_direct(x, bx, y, by)
            if lw > tr * (1 + 1e-9):
                nviol += 1
                ratio = float(np.sqrt(lw / tr))
                if best is None or ratio > best["ratio"]:
                    best = dict(ratio=ratio, x=x.round(4).tolist(), y=y.round(4).tolist(),
                                bx=bx.tolist(), by=by.tolist())
        out[m] = dict(trials=trials, violations=nviol, best=best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=20000)
    args = ap.parse_args()
    res = dict(part_a=part_a(), part_b=part_b(trials=args.trials))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    for r in res["part_a"]["rows"]:
        print("A", r)
    for m, r in res["part_b"].items():
        print("B m=%s violations=%d/%d best_ratio=%s" % (m, r["violations"], r["trials"],
              None if r["best"] is None else round(r["best"]["ratio"], 3)))


if __name__ == "__main__":
    main()
