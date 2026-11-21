#!/usr/bin/env python3
"""
Wider admissibility audit: 40 CWRU recordings (4 motor speeds x {inner race, outer race @6,
ball} x {7, 14, 21} mil faults + normal baseline), and limits that a practitioner could
actually declare.

For each recording the analysed segment is samples [10000, 10000+n); the "history" is the
10,000 samples before it (an earlier, complete record of the same machine). Limit settings:

  window        per-window observed range (MDMS practice)
  global        observed range of the segment
  oracle_range  true range of the complete segment (best case for a declared range)
  oracle_ours   true range + true max slew + optimal centre
  hist_range    range of the history
  hist_ours     range and max slew of the history + optimal centre
  hist_ours_m10 history range widened by 10% of its span on each side, slew x1.10
  hist_ours_m25 history range widened by 25% of its span on each side, slew x1.25

A declared limit that the observed segment values contradict is rejected (reported as
rejected=1, no metrics): an analyst would see the contradiction. Hidden values that exceed the
limit cannot be seen, and that is what the audit measures.

Usage:
  python3 -I audit/wide_audit.py --data-dir DIR --out results/wide/wide_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import mdms_ref as M  # noqa: E402
from c3_audit import audit_one  # noqa: E402

SEG0, HIST = 10_000, 10_000


def recordings(data_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        if not (name.endswith("_DE12") or name.endswith("_Normal")):
            continue
        d = np.load(path, allow_pickle=False)
        out[name] = d["DE"].ravel().astype(np.float64)
    return out


def masks_for(Tc, m, seed):
    n = len(Tc)
    return {
        "random_5pct": M.generate_missing_mask(n, m, 0.05, "random", seed=seed)[0],
        "block_10pct_b30": M.generate_missing_mask(n, m, 0.10, "block", 0.30, seed=seed)[0],
        "clip_q99": M.clipping_mask(Tc, 0.99)[0],
        "clip_q95": M.clipping_mask(Tc, 0.95)[0],
    }


def widen(lo, hi, slew, f):
    span = hi - lo
    return (slew * (1 + f), lo - f * span, hi + f * span)


def modes_for(Tc, H):
    lo, hi = float(Tc.min()), float(Tc.max())
    sl = float(np.max(np.abs(np.diff(Tc))))
    hlo, hhi = float(H.min()), float(H.max())
    hsl = float(np.max(np.abs(np.diff(H))))
    return {
        "window": ("window", None),
        "global": ("global", None),
        "oracle_range": ("declared", (lo, hi)),
        "oracle_ours": ("slew", (sl, lo, hi), True),
        "hist_range": ("declared", (hlo, hhi)),
        "hist_ours": ("slew", (hsl, hlo, hhi), True),
        "hist_ours_m10": ("slew", widen(hlo, hhi, hsl, 0.10), True),
        "hist_ours_m25": ("slew", widen(hlo, hhi, hsl, 0.25), True),
    }


def accepted(Tc, mask, m, spec):
    """True if the observed values are consistent with the declared limits."""
    T_in = np.where(mask, Tc, 0.0)
    try:
        M.window_stats(T_in, mask, m, spec[0], spec[1], *(tuple(spec[2:]) or (False,)))
        return True
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated recording names")
    a = ap.parse_args(argv)
    recs = recordings(a.data_dir)
    if a.only:
        keep = set(a.only.split(","))
        recs = {k: v for k, v in recs.items() if k in keep}
    if not recs:
        raise SystemExit("no recordings found")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows, t0 = [], time.time()
    for name, de in recs.items():
        Tc = de[SEG0:SEG0 + a.n].copy()
        H = de[SEG0 - HIST:SEG0].copy()
        modes = modes_for(Tc, H)
        for mname, mask in masks_for(Tc, a.m, a.seed).items():
            ok = {k: accepted(Tc, mask, a.m, v) for k, v in modes.items()}
            live = {k: v for k, v in modes.items() if ok[k]}
            res = audit_one(Tc, mask, a.m, live)
            for mode in modes:
                base = dict(recording=name, mask=mname, mode=mode, n=a.n, m=a.m, seed=a.seed,
                            rejected=int(not ok[mode]),
                            hidden_outside=int(_hidden_outside(Tc, mask, modes[mode])))
                rows.append({**base, **res.get(mode, {})})
            w = res["window"]
            print(f"[{time.time() - t0:7.1f}s] {name:22s} {mname:16s} "
                  f"rowviol window={w['row_viol']} hist_ours="
                  f"{res['hist_ours']['row_viol'] if 'hist_ours' in res else 'rej'} "
                  f"m10={res['hist_ours_m10']['row_viol'] if 'hist_ours_m10' in res else 'rej'}",
                  flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as f:
        json.dump(dict(argv=sys.argv, n=a.n, m=a.m, seed=a.seed, recordings=list(recs),
                       seconds=time.time() - t0), f, indent=2)
    print("wrote", a.out)


def _hidden_outside(Tc, mask, spec):
    """1 if some missing value lies outside a declared range (assumption (A) fails)."""
    kind, lim = spec[0], spec[1]
    if kind == "declared":
        lo, hi = lim
    elif kind == "slew" and isinstance(lim, tuple):
        lo, hi = lim[1], lim[2]
    else:
        return 0
    hid = Tc[~mask]
    return bool(hid.size and (hid.min() < lo or hid.max() > hi))


if __name__ == "__main__":
    main()
