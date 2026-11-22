#!/usr/bin/env python3
"""
Real-saturation case study on PAMAP2 (Reiss and Stricker, ISWC 2012; CC BY 4.0).

Each PAMAP2 IMU carries a +-6 g and a +-16 g accelerometer side by side. During running and
rope jumping the +-6 g sensor saturates. We treat its saturated samples as missing (as MDMS
recommends for clipped plateaus) and use the co-located +-16 g sensor, linearly calibrated to
the +-6 g one on unsaturated samples, as the ground truth for the lost values.

Per segment (n samples of one axis of one IMU, inside running or rope jumping, where the +-6 g
sensor saturates and the +-16 g one does not):
  T_true = x6 where unsaturated, a * x16 + b where saturated
  mask   = not saturated
and the audit compares the MDMS bound with the exact distances of T_true under:
  window       per-window observed limits (MDMS practice)
  global       observed range of the segment
  sensor_range the +-6 g sensor's own range: what a datasheet of the saturating sensor gives
  physical     +-16 g, the range of the co-located reference sensor, declared as a bound on
               the physical acceleration (valid when the +-16 g sensor itself does not saturate)
  physical_ours  +-16 g plus a slew-rate limit taken from the preceding HIST samples of the
               reference sensor, widened by 25%, plus the optimal centre

Usage: python3 -I audit/pamap2_case.py --data-dir pamap2_out --out results/pamap2/pamap2_case.csv
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
from c3_audit import audit_one  # noqa: E402

G = 9.80665
ACTIVITIES = {5: "running", 24: "rope_jumping"}
HIST = 10_000


def saturation_mask(x6, level=6 * G, rel=0.98):
    """Samples of the +-6 g channel at its rails. The rail sits near +-6 g but is not exactly
    calibrated, so it is taken as the extreme observed values when they form a plateau."""
    hi, lo = np.nanmax(x6), np.nanmin(x6)
    sat = np.zeros(len(x6), bool)
    if hi > rel * level:
        sat |= x6 >= hi - 1e-3 * level
    if lo < -rel * level:
        sat |= x6 <= lo + 1e-3 * level
    return sat


def calibrate(x6, x16, sat, level=6 * G):
    ok = (~sat) & np.isfinite(x6) & np.isfinite(x16) & (np.abs(x16) < 0.9 * level)
    A = np.c_[x16[ok], np.ones(ok.sum())]
    (a, b), *_ = np.linalg.lstsq(A, x6[ok], rcond=None)
    resid = x6[ok] - (a * x16[ok] + b)
    return float(a), float(b), float(np.sqrt(np.mean(resid ** 2)))


def segments(rec, imu, axis, n, max_per_series):
    """Contiguous runs of a high-impact activity; non-overlapping n-sample segments inside them
    where the +-6 g channel saturates and the +-16 g channel stays below its own rail."""
    act = rec["act"]
    x6 = rec[imu + "6"][:, axis].astype(np.float64)
    x16 = rec[imu + "16"][:, axis].astype(np.float64)
    out = []
    for aid in ACTIVITIES:
        idx = np.flatnonzero(act == aid)
        if idx.size < n + HIST:
            continue
        # split into contiguous runs of the activity
        cuts = np.flatnonzero(np.diff(idx) != 1)
        runs = np.split(idx, cuts + 1)
        for run in runs:
            for s in range(run[0] + HIST, run[-1] - n + 2, n):
                seg6, seg16 = x6[s:s + n], x16[s:s + n]
                if not (np.isfinite(seg6).all() and np.isfinite(seg16).all()):
                    continue
                if np.abs(seg16).max() >= 15.5 * G:
                    continue
                sat = saturation_mask(seg6)
                if sat.mean() < 0.001:
                    continue
                out.append((aid, s, seg6, seg16, x16[s - HIST:s], sat))
                if len(out) >= max_per_series:
                    return out
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--max-per-series", type=int, default=2)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.data_dir, "*.npz")))
    if not files:
        raise SystemExit("no .npz files")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows, t0 = [], time.time()
    for f in files:
        rec = dict(np.load(f, allow_pickle=False))
        name = os.path.basename(f)[:-4]
        for imu in ("hand", "chest", "ankle"):
            for axis in range(3):
                for aid, s, x6, x16, h16, sat in segments(rec, imu, axis, a.n, a.max_per_series):
                    ca, cb, crms = calibrate(x6, x16, sat)
                    T = np.where(sat, ca * x16 + cb, x6)
                    mask = ~sat
                    hs = ca * h16 + cb
                    slew = float(np.max(np.abs(np.diff(hs)))) * 1.25
                    lo16, hi16 = ca * (-16 * G) + cb, ca * (16 * G) + cb
                    modes = {
                        "window": ("window", None),
                        "global": ("global", None),
                        "sensor_range": ("declared", (-6 * G, 6 * G)),
                        "physical": ("declared", (min(lo16, hi16), max(lo16, hi16))),
                        "physical_ours": ("slew", (slew, min(lo16, hi16), max(lo16, hi16)), True),
                    }
                    live = {}
                    for k, v in modes.items():
                        obs = T[mask]
                        rng = v[1] if v[0] == "declared" else (v[1][1:] if v[0] == "slew" else None)
                        if rng is not None and (obs.min() < rng[0] or obs.max() > rng[1]):
                            continue
                        live[k] = v
                    res = audit_one(T, mask, a.m, live)
                    hidden = T[~mask]
                    for k in modes:
                        base = dict(file=name, imu=imu, axis=axis, activity=ACTIVITIES[aid], start=s, n=a.n, m=a.m,
                                    sat_frac=float(sat.mean()), calib_a=ca, calib_b=cb, calib_rms=crms,
                                    hidden_max_g=float(np.abs(hidden).max() / G), mode=k,
                                    rejected=int(k not in live))
                        rows.append({**base, **res.get(k, {})})
                    w = res["window"]
                    print(f"[{time.time() - t0:6.1f}s] {name} {imu}{'xyz'[axis]} {ACTIVITIES[aid]} sat={sat.mean():.3%} "
                          f"window row_viol={w['row_viol']}/{w['rows']} "
                          f"sensor={res.get('sensor_range', {}).get('row_viol', 'rej')} "
                          f"physical={res.get('physical', {}).get('row_viol', 'rej')}", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as fh:
        json.dump(dict(argv=sys.argv, files=[os.path.basename(f) for f in files], seconds=time.time() - t0), fh, indent=2)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
