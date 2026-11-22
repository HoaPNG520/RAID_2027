#!/usr/bin/env python3
"""
Extract the accelerometer channels of PAMAP2 (Reiss and Stricker, ISWC 2012; UCI ML Repository,
CC BY 4.0) for the real-saturation case study.

Every PAMAP2 IMU carries two co-located 3-axis accelerometers: one with a +-16 g range and one
with a +-6 g range. In the data the +-6 g sensor sits at its rails (about +-6.3 g) during
high-impact movements such as running, while the +-16 g one does not: real saturation with a
co-located reference for the lost values.

Input : the UCI zip (outer zip containing PAMAP2_Dataset.zip, or the inner zip itself).
Output: one compressed .npz per subject and session with
          act      int8   activity id per sample (100 Hz)
          <imu>16  float32 (N, 3)  +-16 g acceleration (m/s^2), imu in hand, chest, ankle
          <imu>6   float32 (N, 3)  +-6 g acceleration (m/s^2)
        only rows whose activity is in KEEP are stored (the high-impact activities), so the
        files stay small; plus MANIFEST.json with sha256 of every output file.

Usage: python3 -I scripts/extract_pamap2.py --zip PAMAP2.zip --out data_pamap2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile

import numpy as np

# 1-based column layout from the PAMAP2 readme: 1 timestamp, 2 activityID, 3 heart rate,
# 4-20 IMU hand, 21-37 IMU chest, 38-54 IMU ankle; inside an IMU: 1 temperature,
# 2-4 acc +-16 g, 5-7 acc +-6 g, ...
IMU_START = {"hand": 4, "chest": 21, "ankle": 38}
KEEP = (4, 5, 6, 12, 13, 24)   # walking, running, cycling, ascending/descending stairs, rope jumping


def columns():
    cols = {"act": 1}
    for imu, s in IMU_START.items():
        base = s - 1                                  # 0-based index of the temperature column
        cols[imu + "16"] = [base + 1, base + 2, base + 3]
        cols[imu + "6"] = [base + 4, base + 5, base + 6]
    return cols


def parse_dat(raw: bytes):
    arr = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 54:
        raise ValueError(f"expected 54 columns, got {arr.shape}")
    act = arr[:, 1].astype(np.int64)
    keep = np.isin(act, KEEP)
    out = {"act": act[keep].astype(np.int8)}
    for name, idx in columns().items():
        if name == "act":
            continue
        out[name] = arr[keep][:, idx].astype(np.float32)
    return out


def inner_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if any(n.endswith(".dat") for n in names):
        return z
    inner = [n for n in names if n.lower().endswith(".zip")]
    if not inner:
        raise SystemExit("no .dat files and no inner zip found")
    return zipfile.ZipFile(io.BytesIO(z.read(inner[0])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    z = inner_zip(a.zip)
    manifest = {}
    for name in sorted(n for n in z.namelist() if n.endswith(".dat")):
        session = "protocol" if "/Protocol/" in name else "optional" if "/Optional/" in name else "other"
        subj = os.path.basename(name)[:-4]
        d = parse_dat(z.read(name))
        if d["act"].size == 0:
            continue
        fn = f"{session}_{subj}.npz"
        dst = os.path.join(a.out, fn)
        np.savez_compressed(dst, **d)
        manifest[fn] = dict(rows=int(d["act"].size), sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest())
        print(fn, manifest[fn]["rows"])
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(dict(source="UCI ML Repository, PAMAP2 Physical Activity Monitoring (id 231), CC BY 4.0",
                       citation="A. Reiss and D. Stricker, Introducing a New Benchmarked Dataset for "
                                "Activity Monitoring, ISWC 2012",
                       activities_kept=KEEP, files=manifest), f, indent=2)


if __name__ == "__main__":
    main()
