#!/usr/bin/env python3
"""
C3 admissibility audit (Appendix H3).

For each (dataset, missingness pattern) the audit computes, over ALL valid pairs
outside the exclusion zone:
  * the MDMS lower bound under each vlimits mode, and
  * the exact z-normalized distance on the complete (oracle) series,
and reports where the lower bound exceeds the true distance (a false-negative risk).

Usage:
  python3 -I audit/c3_audit.py --data-dir DIR --out results/c3_audit.csv [--n 3000 --m 100]

The CWRU files are the .npz mirror of the CWRU Bearing Data Center (DE channel).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mdms_ref as M  # noqa: E402


# --------------------------------------------------------------------------- #
def load_datasets(data_dir, n, seed=0):
    """Return {name: complete series of length n}."""
    out = {}
    cwru = [("cwru_IR007", "1797_IR_7_DE12.npz"), ("cwru_OR007", "1797_OR@6_7_DE12.npz"),
            ("cwru_B007", "1797_B_7_DE12.npz"), ("cwru_normal", "1797_Normal.npz")]
    for name, fn in cwru:
        path = os.path.join(data_dir, fn)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            de = d["DE"].ravel().astype(np.float64)
            start = 10_000                       # skip the first samples
            out[name] = de[start:start + n].copy()
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    out["syn_sine_noise"] = np.sin(2 * np.pi * t / 50) + 0.2 * rng.standard_normal(n)
    out["syn_random_walk"] = np.cumsum(rng.standard_normal(n))
    # idealized fault signal: periodic impacts with decaying ringing + noise
    imp = np.zeros(n)
    period = 75
    for s0 in range(5, n, period):
        amp = 8.0 * (1 + 0.1 * rng.standard_normal())
        tt = np.arange(0, min(60, n - s0))
        imp[s0:s0 + len(tt)] += amp * np.exp(-tt / 8.0) * np.cos(2 * np.pi * tt / 6.0)
    out["syn_impulse_train"] = imp + 0.3 * rng.standard_normal(n)
    return out


def make_masks(Tc, m, seed):
    n = len(Tc)
    masks = {}
    masks["random_1pct"] = M.generate_missing_mask(n, m, 0.01, "random", seed=seed)[0]
    masks["random_5pct"] = M.generate_missing_mask(n, m, 0.05, "random", seed=seed)[0]
    masks["block_10pct_b10"] = M.generate_missing_mask(n, m, 0.10, "block", 0.10, seed=seed)[0]
    masks["block_10pct_b30"] = M.generate_missing_mask(n, m, 0.10, "block", 0.30, seed=seed)[0]
    masks["clip_q99.5"] = M.clipping_mask(Tc, 0.995)[0]
    masks["clip_q99"] = M.clipping_mask(Tc, 0.99)[0]
    masks["clip_q98"] = M.clipping_mask(Tc, 0.98)[0]
    masks["clip_q95"] = M.clipping_mask(Tc, 0.95)[0]
    return masks


def modes_for(Tc):
    lo, hi = float(Tc.min()), float(Tc.max())
    span = hi - lo
    return {
        "window": ("window", None),
        "global": ("global", None),
        "declared_true": ("declared", (lo, hi)),            # true range of the series
        "declared_wide": ("declared", (lo - 0.5 * span, hi + 0.5 * span)),
        "declared_slew": ("slew", float(np.max(np.abs(np.diff(Tc))))),   # true max slew
        "declared_both": ("slew", (float(np.max(np.abs(np.diff(Tc)))), lo, hi), True),  # ours
    }


# --------------------------------------------------------------------------- #
def audit_one(Tc, mask, m, modes, excl=0.25, topk=10, rtol=1e-9, atol=1e-9):
    """Audit one series + mask. Returns {mode: metrics dict}."""
    T_obs = np.where(mask, Tc, np.nan)
    T_in = np.where(mask, Tc, 0.0)
    os_ = M.oracle_stats(Tc, m)
    wss = {k: M.window_stats(T_in, mask, m, v[0], v[1], *(tuple(v[2:]) or (False,)))
           for k, v in modes.items()}
    ref = next(iter(wss.values()))
    L = ref["L"]
    kmin = M.excl_zone(m, excl) + 1

    acc = {k: dict(pairs=0, viol=0, viol_1pct=0, viol_10pct=0, max_ratio=0.0,
                   viol_by_case={1: 0, 2: 0, 3: 0, 4: 0}, ratios=[],
                   P2=np.full(L, np.inf), lb_all=[]) for k in modes}
    Po = np.full(L, np.inf)
    or_all = []
    idx_i, idx_j = [], []

    for k in range(kmin, L):
        sums = M.diagonal_sums(ref, k)
        ii = np.arange(L - k)
        jj = ii + k
        d2o = M.oracle_diagonal_d2(os_, k)
        np.minimum.at(Po, ii, d2o)
        np.minimum.at(Po, jj, d2o)
        or_all.append(d2o.astype(np.float64))
        idx_i.append(ii.astype(np.int32))
        idx_j.append(jj.astype(np.int32))
        for name, ws in wss.items():
            d2, case = M.lb_d2_from_sums(*sums, ws["nobs"][ii], ws["nobs"][jj],
                                         ws["inv_den"][ii], ws["inv_den"][jj],
                                         ws["valid"][ii], ws["valid"][jj], m,
                                         return_case=True)
            a = acc[name]
            np.minimum.at(a["P2"], ii, d2)
            np.minimum.at(a["P2"], jj, d2)
            a["lb_all"].append(d2.astype(np.float64))
            ok = np.isfinite(d2) & np.isfinite(d2o)
            a["pairs"] += int(ok.sum())
            v = ok & (d2 > d2o * (1 + rtol) + atol)
            a["viol"] += int(v.sum())
            for c in (1, 2, 3, 4):
                a["viol_by_case"][c] += int((v & (case == c)).sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.sqrt(d2[ok]) / np.sqrt(d2o[ok])
            pos = np.isfinite(r)
            r = r[pos]
            if r.size:
                a["viol_1pct"] += int((r > 1.01).sum())
                a["viol_10pct"] += int((r > 1.10).sum())
                a["max_ratio"] = max(a["max_ratio"], float(r.max()))
                a["ratios"].append(r.astype(np.float32))

    or_all = np.concatenate(or_all)
    I_all = np.concatenate(idx_i)
    J_all = np.concatenate(idx_j)
    # Oracle top-k motif pairs (non-trivial, distinct; pairs are already outside excl)
    order = np.argsort(or_all, kind="stable")
    top = order[:topk]

    res = {}
    complete = ref["complete"]
    for name, a in acc.items():
        lb = np.concatenate(a["lb_all"])
        ratios = np.concatenate(a["ratios"]) if a["ratios"] else np.array([np.nan])
        lost = [int(t) for t in top if lb[t] > or_all[t] * (1 + rtol) + atol]
        # rank of the oracle top-1 pair among all lower bounds
        t1 = top[0]
        rank1 = int((lb < lb[t1]).sum()) + 1 if np.isfinite(lb[t1]) else -1
        rowv = np.isfinite(a["P2"]) & np.isfinite(Po) & (a["P2"] > Po * (1 + rtol) + atol)
        res[name] = dict(
            pairs=a["pairs"], pair_viol=a["viol"],
            pair_viol_rate=a["viol"] / max(a["pairs"], 1),
            pair_viol_gt1pct=a["viol_1pct"], pair_viol_gt10pct=a["viol_10pct"],
            max_ratio=a["max_ratio"],
            viol_case1=a["viol_by_case"][1], viol_case2=a["viol_by_case"][2],
            viol_case3=a["viol_by_case"][3],
            rows=int((np.isfinite(a["P2"]) & np.isfinite(Po)).sum()),
            row_viol=int(rowv.sum()),
            row_viol_rate=float(rowv.sum()) / max(int(np.isfinite(Po).sum()), 1),
            topk_lost=len(lost), top1_lb_rank=rank1,
            top1_pair=f"{int(I_all[t1])}-{int(J_all[t1])}",
            tightness_median=float(np.nanmedian(ratios)),
            coverage_mdms=float(np.isfinite(a["P2"]).mean()),
            coverage_exclusion=float(complete.mean()),
            missing_rate=float((~mask).mean()),
            frac_windows_incomplete=float((~complete).mean()),
        )
    return res


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    args = ap.parse_args(argv)

    data = load_datasets(args.data_dir, args.n, args.seed)
    if args.only:
        keep = set(args.only.split(","))
        data = {k: v for k, v in data.items() if k in keep}
    if not data:
        raise SystemExit("no datasets loaded")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    t0 = time.time()
    for dname, Tc in data.items():
        modes = modes_for(Tc)
        for mname, mask in make_masks(Tc, args.m, args.seed).items():
            t = time.time()
            res = audit_one(Tc, mask, args.m, modes)
            for mode, r in res.items():
                row = dict(dataset=dname, mask=mname, mode=mode, n=args.n, m=args.m,
                           seed=args.seed, **r)
                rows.append(row)
            w = res["window"]
            print(f"[{time.time() - t0:7.1f}s] {dname:16s} {mname:16s} "
                  f"miss={w['missing_rate']:.3f} pairs={w['pairs']} "
                  f"viol(window)={w['pair_viol']} rowviol(window)={w['row_viol']} "
                  f"viol(global)={res['global']['pair_viol']} "
                  f"viol(decl)={res['declared_true']['pair_viol']} "
                  f"({time.time() - t:.1f}s)", flush=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(dict(argv=sys.argv, n=args.n, m=args.m, seed=args.seed,
                       datasets=list(data.keys()), seconds=time.time() - t0), f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
