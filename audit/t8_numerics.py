#!/usr/bin/env python3
"""
T8 (CPU emulation): numerical admissibility of a float32 diagonal recurrence.

Emulates what a GPU f32 kernel does on each diagonal: seed the six sums for the first
pair, then update them with float32 additions along the diagonal (sequential order),
and evaluate the bound in float32. Compares against the oracle distance using the
*declared* (valid) limits, where the exact bound has zero violations.

Variants:
  raw      : no global standardization (the series keeps its DC offset)
  centered : globally standardized over observed values (spec Appendix H2)
  reseed   : centered + accumulators re-seeded from exact sums every K rows
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mdms_ref as M  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c3_audit import load_datasets, make_masks  # noqa: E402

F = np.float32


def f32_diag_sums(z, b, x, m, k, reseed=None):
    """Six sums along diagonal k in float32: exact seeds every `reseed` rows, float32 updates."""
    n = len(z)
    L = n - m + 1
    cnt = L - k
    zi, zj = z[:n - k].astype(F), z[k:].astype(F)
    bi, bj = b[:n - k].astype(F), b[k:].astype(F)
    xi, xj = (zi * zi), (zj * zj)       # x recomputed as z*z in float32, as on the GPU
    prods = [bi * bj, zi * bj, bi * zj, xi * bj, bi * xj, zi * zj]
    out = []
    step = reseed or cnt
    for p in prods:
        res = np.empty(cnt, dtype=F)
        for s0 in range(0, cnt, step):
            s1 = min(cnt, s0 + step)
            seed = F(np.sum(p[s0:s0 + m].astype(np.float64)))     # exact seed
            delta = (p[s0 + m:s1 - 1 + m] - p[s0:s1 - 1]).astype(F)
            acc = np.empty(s1 - s0, dtype=F)
            acc[0] = seed
            if s1 - s0 > 1:
                acc[1:] = seed + np.cumsum(delta, dtype=F)          # sequential float32
            res[s0:s1] = acc
        out.append(res)
    out[0] = np.rint(out[0]).astype(np.int64)  # R is an integer accumulator on the GPU
    return out


def f32_bound(R, Si, Sj, Qi, Qj, Pij, nobs_i, nobs_j, inv_i, inv_j, valid_i, valid_j, m):
    """Section 2 bound evaluated in float32 arithmetic."""
    with np.errstate(all="ignore"):
        Rf = R.astype(F)
        Rs = np.where(R > 0, Rf, F(1))
        mi, mj = Si / Rs, Sj / Rs
        vi = np.maximum(Qi / Rs - mi * mi, F(0))
        vj = np.maximum(Qj / Rs - mj * mj, F(0))
        cov = Pij / Rs - mi * mj
        vv = vi * vj
        q = np.where(vv > F(1e-24), cov / np.sqrt(np.where(vv > 0, vv, F(1))), F(0))
        q = np.clip(q, F(-1), F(1)).astype(F)
        corr = np.where(q > 0, F(1) - q * q, F(1)).astype(F)
        ii, jj = inv_i.astype(F), inv_j.astype(F)
        ci, cj = nobs_i == m, nobs_j == m
        d1 = np.maximum(F(2 * m) * (F(1) - q), F(0))
        d2c = Rf * np.where(ci, vi, vj) * np.where(ci, ii, jj) * corr
        d3 = Rf * np.maximum(vi * ii, vj * jj) * corr
        d2 = np.where(ci & cj, d1, np.where(ci | cj, d2c, d3)).astype(np.float64)
    d2 = np.where(R == 0, 0.0, d2)
    d2 = np.where(~(valid_i & valid_j), np.inf, d2)
    d2 = np.where(np.isnan(d2), np.inf, d2)
    return np.maximum(d2, 0.0)


def run(Tc, mask, m, variant, offset, reseed_k, slacks):
    T = Tc + offset
    lo, hi = float(T.min()), float(T.max())
    ws = M.window_stats(np.where(mask, T, 0.0), mask, m, "declared", (lo, hi))
    if variant == "raw":
        # undo the global standardization for the accumulators (raw values),
        # keep the per-window statistics consistent with raw units
        z = np.where(mask, T, 0.0)
        scale = ws["sd"]
        inv_den = ws["inv_den"] / scale ** 2
    else:
        z = ws["z"]
        inv_den = ws["inv_den"]
    b = mask.astype(np.int64)
    x = z * z
    os_ = M.oracle_stats(Tc, m)
    L = ws["L"]
    kmin = M.excl_zone(m) + 1
    stats = {s: 0 for s in slacks}
    pairs = 0
    exact_viol = 0
    max_rel = 0.0
    for k in range(kmin, L):
        ii = np.arange(L - k)
        jj = ii + k
        R, Si, Sj, Qi, Qj, Pij = f32_diag_sums(z, b, x, m, k,
                                              reseed=reseed_k if variant == "reseed" else None)
        d2f = f32_bound(R, Si, Sj, Qi, Qj, Pij, ws["nobs"][ii], ws["nobs"][jj],
                        inv_den[ii], inv_den[jj], ws["valid"][ii], ws["valid"][jj], m)
        d2o = M.oracle_diagonal_d2(os_, k)
        d2e = M.diagonal_lb_d2(ws, k)
        ok = np.isfinite(d2f) & np.isfinite(d2o)
        pairs += int(ok.sum())
        exact_viol += int((ok & (d2e > d2o * (1 + 1e-12) + 1e-12)).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(ok & (d2o > 0), d2f / d2o - 1.0, 0.0)
        if rel.size:
            max_rel = max(max_rel, float(np.max(rel)))
        for s in slacks:
            d2s = np.maximum(0.0, d2f * (1 - s))
            stats[s] += int((ok & (d2s > d2o)).sum())
    return pairs, exact_viol, max_rel, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--reseed", type=int, default=256)
    args = ap.parse_args()
    data = load_datasets(args.data_dir, args.n)
    slacks = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    rows = []
    for dname in ["cwru_IR007", "cwru_normal", "syn_random_walk"]:
        Tc = data[dname]
        mask = make_masks(Tc, args.m, 0)["random_1pct"]
        for offset_sd in [0.0, 100.0]:
            offset = offset_sd * Tc.std()
            for variant in ["raw", "centered", "reseed"]:
                pairs, ev, mr, st = run(Tc, mask, args.m, variant, offset, args.reseed, slacks)
                row = dict(dataset=dname, offset_in_sd=offset_sd, variant=variant,
                           reseed_k=args.reseed if variant == "reseed" else "",
                           pairs=pairs, exact_f64_viol=ev, max_rel_excess=mr,
                           **{f"viol_slack_{s:g}": st[s] for s in slacks})
                rows.append(row)
                print(row, flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
