#!/usr/bin/env python3
"""
Speed benchmark (experiment E3) and GPU-vs-twin agreement check.

  python3 -I gpu/bench.py --suite agree --gpu bin/mdms_gpu --twin bin/mdms_twin --out out/agree.jsonl
  python3 -I gpu/bench.py --suite speed --gpu bin/mdms_gpu --twin bin/mdms_twin --out out/speed_d0.jsonl \
          --device 0 --shard 0/2

The speed suite times the GPU kernel (cudaEvent) for MDMS (f64, f32c, f32) and for the
plain STOMP baseline built from the same code (same banding, one accumulator), and the
single-threaded CPU twin for small n. Optionally STUMPY's gpu_stump as an external baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from compare import run_bin  # noqa: E402
import mdms_ref as M  # noqa: E402


def series(n, seed=0, missing=0.01, m=100):
    rng = np.random.default_rng(seed)
    T = np.cumsum(rng.standard_normal(n))
    mask, _ = M.generate_missing_mask(n, m, missing, "random", seed=seed)
    return T, mask


def suite_agree(gpu, twin, out, device):
    rows, ok = [], True
    for n, m in [(3000, 100), (20000, 64)]:
        T, mask = series(n, 1, 0.05, m)
        Tin = np.where(mask, T, np.nan)
        for prec in ("f64", "f32c", "f32"):
            for vl in ("window", "declared"):
                lim = (T.min(), T.max()) if vl == "declared" else None
                Pg, Ig, sg = run_bin(gpu, Tin, m, prec, vl, lim, device=device)
                Pt, It, st = run_bin(twin, Tin, m, prec, vl, lim)
                same = np.array_equal(np.isfinite(Pg), np.isfinite(Pt))
                f = np.isfinite(Pt)
                diff = float(np.max(np.abs(Pg[f] - Pt[f]))) if f.any() else 0.0
                r = dict(n=n, m=m, precision=prec, vlimits=vl, finite_equal=bool(same),
                         bitwise_equal_frac=float(np.mean(Pg[f] == Pt[f])) if f.any() else 1.0,
                         max_abs_diff=diff, index_equal_frac=float(np.mean(Ig == It)),
                         device=sg["device_name"])
                tol = 1e-9 if prec == "f64" else 1e-4
                r["pass"] = bool(same and diff <= tol * max(1.0, float(np.max(Pt[f])) if f.any() else 1.0))
                ok &= r["pass"]
                rows.append(r)
                print(json.dumps(r), flush=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return ok


def speed_cases():
    cases = []
    for e in (14, 15, 16, 17, 18, 19, 20):
        n = 2 ** e
        cases.append(("mdms", "f32c", n))
        cases.append(("stomp", "f32c", n))
        if e <= 18:
            cases.append(("mdms", "f32", n))
        if e <= 17:
            cases.append(("mdms", "f64", n))
            cases.append(("stomp", "f64", n))
        if e <= 15:
            cases.append(("twin_mdms", "f32c", n))
            cases.append(("twin_mdms", "f64", n))
    return cases


def suite_speed(gpu, twin, out, device, shard, m=100, stumpy=False):
    k, K = (int(x) for x in shard.split("/"))
    cases = [c for i, c in enumerate(speed_cases()) if i % K == k]
    rows = []
    for mode, prec, n in cases:
        T, mask = series(n, 2, 0.01, m)
        t = time.time()
        if mode == "stomp":
            P, I, st = run_bin(gpu, T, m, prec, mode="stomp", device=device)
        elif mode == "mdms":
            P, I, st = run_bin(gpu, np.where(mask, T, np.nan), m, prec, device=device)
        else:
            P, I, st = run_bin(twin, np.where(mask, T, np.nan), m, prec)
        r = dict(mode=mode, precision=prec, n=n, m=m, kernel_ms=st["kernel_ms"],
                 total_ms=st["total_ms"], gpairs_per_s=st["gpairs_per_s"], pairs=st["pairs"],
                 device=st["device_name"], backend=st["backend"], wall_s=time.time() - t)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if stumpy and k == 0:
        try:
            import stumpy
            from numba import cuda
            for e in (14, 16, 18):
                n = 2 ** e
                T, _ = series(n, 2, 0.0, m)
                stumpy.gpu_stump(T[:2048], m, device_id=device)      # JIT warm-up
                cuda.synchronize()
                t = time.time()
                stumpy.gpu_stump(T, m, device_id=device)
                cuda.synchronize()
                r = dict(mode="stumpy_gpu_stump", precision="f64", n=n, m=m,
                         kernel_ms=(time.time() - t) * 1000, device=f"cuda:{device}",
                         backend="stumpy " + stumpy.__version__)
                rows.append(r)
                print(json.dumps(r), flush=True)
        except Exception as exc:  # recorded, not hidden
            r = dict(mode="stumpy_gpu_stump", error=repr(exc))
            rows.append(r)
            print(json.dumps(r), flush=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["agree", "speed"], required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stumpy", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.suite == "agree":
        ok = suite_agree(a.gpu, a.twin, a.out, a.device)
        print("AGREE:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    suite_speed(a.gpu, a.twin, a.out, a.device, a.shard, stumpy=a.stumpy)


if __name__ == "__main__":
    main()
