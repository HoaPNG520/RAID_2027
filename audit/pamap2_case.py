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
  physical_ours  +-16 g plus a slew-rate limit taken from the preceding 5 s (HIST samples) of the
               reference sensor, widened by 25%, plus the optimal centre

Defaults: segments of n = 1000 samples (10 s at 100 Hz), windows of m = 50 (0.5 s, about one
footstep).

Usage: python3 -I audit/pamap2_case.py --data-dir data_pamap2 --out results/pamap2/pamap2_case.csv
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
import mdms_ref as M  # noqa: E402

G = 9.80665
ACTIVITIES = {5: "running", 24: "rope_jumping"}
HIST = 500            # 5 s of the reference sensor before the segment, for the slew-rate limit


RAIL6 = 6.29 * G      # the +-6 g channel's rails sit at about +-6.31 to 6.33 g in every recording
RAIL16 = 15.85 * G    # the +-16 g channel's rails; segments touching them have no reliable truth


def saturation_mask(x6):
    """Samples of the +-6 g channel at either rail."""
    return (x6 >= RAIL6) | (x6 <= -RAIL6)


def best_lag(x6, x16, max_lag=3):
    """The two accelerometers of an IMU are offset by a sample in the PAMAP2 files. Return the
    shift L maximising corr(x6[t], x16[t - L]) over unsaturated finite samples."""
    best, arg = -2.0, 0
    for L in range(-max_lag, max_lag + 1):
        a = x6[max(0, L):len(x6) + min(0, L)]
        b = x16[max(0, -L):len(x16) - max(0, L)]
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 0.9 * RAIL6)
        if ok.sum() < 1000:
            continue
        c = np.corrcoef(a[ok], b[ok])[0, 1]
        if c > best:
            best, arg = c, L
    return arg, best


def shift(x, L):
    """y[t] = x[t - L] (NaN where undefined)."""
    y = np.full_like(x, np.nan)
    if L > 0:
        y[L:] = x[:-L]
    elif L < 0:
        y[:L] = x[-L:]
    else:
        y[:] = x
    return y


def calibrate(x6, x16, sat):
    ok = (~sat) & np.isfinite(x6) & np.isfinite(x16) & (np.abs(x16) < 0.9 * RAIL6)
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
                if np.abs(seg16).max() >= RAIL16:
                    continue
                sat = saturation_mask(seg6)
                if sat.sum() < 2:
                    continue
                out.append((aid, s, seg6, seg16, x16[s - HIST:s], sat))
                if max_per_series and len(out) >= max_per_series:
                    return out
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--max-per-series", type=int, default=0,
                    help="cap on segments per sensor axis (0 = no cap: every qualifying segment)")
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
                full6 = rec[imu + "6"][:, axis].astype(np.float64)
                lag, lag_corr = best_lag(full6, rec[imu + "16"][:, axis].astype(np.float64))
                rec[imu + "16"][:, axis] = shift(rec[imu + "16"][:, axis].astype(np.float64), lag)
                rails = (float(np.nanmin(full6)), float(np.nanmax(full6)))   # what the sensor reports at its limits
                for aid, s, x6, x16, h16, sat in segments(rec, imu, axis, a.n, a.max_per_series):
                    ca, cb, crms = calibrate(x6, x16, sat)
                    T = np.where(sat, ca * x16 + cb, x6)
                    mask = ~sat
                    hs = ca * h16 + cb
                    dh = np.abs(np.diff(hs))                 # PAMAP2 has dropped packets (NaN)
                    slew = float(np.nanmax(dh)) * 1.25 if np.isfinite(dh).any() else float("nan")
                    if not slew > 0:
                        continue
                    lo16, hi16 = ca * (-16 * G) + cb, ca * (16 * G) + cb
                    modes = {
                        "window": ("window", None),
                        "global": ("global", None),
                        "sensor_range": ("declared", rails),
                        "physical": ("declared", (min(lo16, hi16), max(lo16, hi16))),
                        "physical_ours": ("slew", (slew, min(lo16, hi16), max(lo16, hi16)), True),
                    }
                    live = {}
                    for k, v in modes.items():
                        # a declared limit that the observed values contradict is rejected
                        try:
                            M.window_stats(np.where(mask, T, 0.0), mask, a.m, v[0], v[1],
                                           *(tuple(v[2:]) or (False,)))
                        except ValueError:
                            continue
                        live[k] = v
                    res = audit_one(T, mask, a.m, live)
                    hidden = T[~mask]
                    for k in modes:
                        base = dict(file=name, imu=imu, axis=axis, activity=ACTIVITIES[aid], start=s, n=a.n, m=a.m,
                                    lag=lag, lag_corr=lag_corr,
                                    hidden_below_rail=float(np.mean(np.abs(T[~mask]) < RAIL6)),
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
