#!/usr/bin/env python3
"""Independent re-check of audit violations: direct sums for the bound, direct oracle distance,
and the mechanism (hidden values outside the window's observed range)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mdms_ref as M
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c3_audit import load_datasets, make_masks

def main(data_dir, dname, mname, n=3000, m=100, nshow=5):
    Tc = load_datasets(data_dir, n)[dname]
    mask = make_masks(Tc, m, 0)[mname]
    T_in = np.where(mask, Tc, 0.0)
    ws = M.window_stats(T_in, mask, m, "window")
    os_ = M.oracle_stats(Tc, m)
    L = ws["L"]; kmin = M.excl_zone(m) + 1
    found = []
    for k in range(kmin, L):
        d2 = M.diagonal_lb_d2(ws, k); d2o = M.oracle_diagonal_d2(os_, k)
        v = np.flatnonzero(np.isfinite(d2) & np.isfinite(d2o) & (d2 > d2o * (1 + 1e-9) + 1e-9))
        for i in v:
            found.append((float(np.sqrt(d2[i] / d2o[i])), int(i), int(i + k)))
    found.sort(reverse=True)
    print(f"{dname} {mname}: {len(found)} violating pairs")
    agree = 0
    for ratio, i, j in found:
        lb = M.pair_lb_d2(ws, i, j); tr = M.oracle_pair_d2(Tc, m, i, j)
        agree += lb > tr * (1 + 1e-9) + 1e-9
    print(f"  re-checked with direct sums + direct oracle: {agree}/{len(found)} confirmed")
    hid_out = 0
    for ratio, i, j in found:
        out = False
        for w in (i, j):
            miss = ~mask[w:w + m]
            if miss.any():
                obs = Tc[w:w + m][~miss]
                hv = Tc[w:w + m][miss]
                out |= bool((hv > obs.max()).any() or (hv < obs.min()).any())
        hid_out += out
    print(f"  pairs where a hidden value lies outside its window's observed range: {hid_out}/{len(found)}")
    for ratio, i, j in found[:nshow]:
        print(f"  pair ({i},{j}) LB/true = {ratio:.3f}  LB={np.sqrt(M.pair_lb_d2(ws,i,j)):.4f} "
              f"true={np.sqrt(M.oracle_pair_d2(Tc,m,i,j)):.4f} nobs=({ws['nobs'][i]},{ws['nobs'][j]})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
