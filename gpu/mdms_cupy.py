#!/usr/bin/env python3
"""
Python launcher for the MDMS CUDA kernels (CuPy RawModule, nvcc backend).

The per-diagonal arithmetic is the same header as the C++ program (src/mdms_core.cuh, via
src/mdms_kernels.cuh); the per-window statistics come from the Python reference
(mdms_ref.window_stats). Use it from Python without files or subprocesses:

    from gpu.mdms_cupy import lb_profile
    P, I = lb_profile(T, m, precision="f32c", vlimits="declared", limits=(lo, hi))

T is a float array with NaN for missing samples. Returns lower-bound distances (not squared)
and neighbour indices (-1 where no neighbour), like bin/mdms_gpu.

Self-test against the C++ binary (needs a GPU and CuPy):
    python3 -I gpu/mdms_cupy.py --check bin/mdms_gpu --out out/cupy_check.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import shutil

import numpy as np

# CuPy's nvcc backend looks for NVCC / CUDA_PATH or nvcc on PATH; Kaggle's image has nvcc only
# under /usr/local/cuda/bin. Set them before cupy is imported.
_nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
os.environ.setdefault("NVCC", _nvcc)
os.environ.setdefault("CUDA_PATH", os.path.dirname(os.path.dirname(_nvcc)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import mdms_ref as M  # noqa: E402

WIN_DTYPE = np.dtype([("nobs", "<i4"), ("valid", "<i4"), ("inv_den", "<f8")], align=True)
TPB = 128

WRAPPERS = r"""
#include "mdms_kernels.cuh"
#define MDMS_WRAP(NAME, A, KAHAN, EMIT_INIT)                                                   \
extern "C" __global__ void NAME(const A* z, const uint8_t* b, const Win* W, int n, int m,     \
                                int L, int kmin, double slack_rel, double slack_abs, int i0, \
                                int s1, ull* P, int* I) {                                    \
    Params p{n, m, L, kmin, slack_rel, slack_abs};                                           \
    const int k = kmin + blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (k >= L - i0) return;                                                                 \
    auto em = EMIT_INIT;                                                                     \
    mdms_segment<A, KAHAN>(z, b, W, p, k, i0, s1, em);                                       \
}
MDMS_WRAP(mdms_f64_pass1, double, false, (EmitF64Pass1{P}))
MDMS_WRAP(mdms_f64_pass2, double, false, (EmitF64Pass2{P, I}))
MDMS_WRAP(mdms_f32c, float, true, (EmitF32{P}))
MDMS_WRAP(mdms_f32, float, false, (EmitF32{P}))
"""

_module = None


def _mod():
    global _module
    if _module is None:
        import cupy as cp
        _module = cp.RawModule(code=WRAPPERS, backend="nvcc",
                               options=("-std=c++17", "-fmad=false", "-I" + os.path.join(ROOT, "src")))
    return _module


def lb_profile(T, m, precision="f32c", vlimits="window", limits=None, opt_center=False,
               excl=0.25, slack_rel=None, slack_abs=0.0, band_rows=None):
    import cupy as cp
    T = np.asarray(T, dtype=np.float64)
    mask = np.isfinite(T)
    ws = M.window_stats(np.where(mask, T, 0.0), mask, m, vlimits, limits, opt_center)
    n, L = len(T), ws["L"]
    kmin = M.excl_zone(m, excl) + 1
    if slack_rel is None:
        slack_rel = 1e-3 if precision != "f64" else 0.0
    W = np.zeros(L, dtype=WIN_DTYPE)
    W["nobs"] = ws["nobs"]
    W["valid"] = ws["valid"].astype(np.int32)
    W["inv_den"] = np.where(ws["valid"], ws["inv_den"], 0.0)
    A = np.float64 if precision == "f64" else np.float32
    dz = cp.asarray(np.asarray(ws["z"], dtype=A))
    db = cp.asarray(mask.astype(np.uint8))
    dW = cp.asarray(W.view(np.uint8))
    band = band_rows or int(min(16384, max(2048, 4 * m)))
    mod = _mod()
    if precision == "f64":
        dP = cp.full(L, np.uint64(0x7FF0000000000000), dtype=cp.uint64)
        dI = cp.full(L, np.iinfo(np.int32).max, dtype=cp.int32)
        passes = ("mdms_f64_pass1", "mdms_f64_pass2")
    else:
        dP = cp.full(L, np.uint64((0x7F800000 << 32) | 0xFFFFFFFF), dtype=cp.uint64)
        dI = cp.zeros(1, dtype=cp.int32)
        passes = ("mdms_f32c" if precision == "f32c" else "mdms_f32",)
    for name in passes:
        k = mod.get_function(name)
        for i0 in range(0, L, band):
            s1 = min(L, i0 + band)
            nd = L - i0 - kmin
            if nd <= 0:
                break
            grid = (nd + TPB - 1) // TPB
            k((grid,), (TPB,), (dz, db, dW, np.int32(n), np.int32(m), np.int32(L), np.int32(kmin),
                                np.float64(slack_rel), np.float64(slack_abs), np.int32(i0),
                                np.int32(s1), dP, dI))
        cp.cuda.runtime.deviceSynchronize()
    hP = cp.asnumpy(dP)
    if precision == "f64":
        d2 = hP.view(np.float64)
        hI = cp.asnumpy(dI).astype(np.int64)
        I = np.where(np.isfinite(d2) & (hI != np.iinfo(np.int32).max), hI, -1)
    else:
        d2 = (hP >> np.uint64(32)).astype(np.uint32).view(np.float32).astype(np.float64)
        idx = (hP & np.uint64(0xFFFFFFFF)).astype(np.int64)
        I = np.where(np.isfinite(d2) & (idx != 0xFFFFFFFF), idx, -1)
    return np.sqrt(d2), I


def self_check(binary, out):
    sys.path.insert(0, HERE)
    from compare import run_bin
    rng = np.random.default_rng(11)
    rows, ok = [], True
    for n, m in ((3000, 100), (20000, 64)):
        T = np.cumsum(rng.standard_normal(n))
        T[rng.random(n) < 0.05] = np.nan
        lo, hi = float(np.nanmin(T)) - 1, float(np.nanmax(T)) + 1
        for prec in ("f64", "f32c"):
            for vl, lim in (("window", None), ("declared", (lo, hi))):
                slack = 0.0 if prec == "f64" else 1e-3
                Pc, Ic = lb_profile(T, m, prec, vl, lim, slack_rel=slack)
                Pb, Ib, _ = run_bin(binary, T, m, prec, vl, lim, slack_rel=slack)
                fin = np.isfinite(Pb)
                same_fin = bool((np.isfinite(Pc) == fin).all())
                rel = float(np.max(np.abs(Pc[fin] - Pb[fin]) / np.maximum(1.0, Pb[fin]))) if fin.any() else 0.0
                tol = 1e-9 if prec == "f64" else 1e-5
                r = dict(n=n, m=m, precision=prec, vlimits=vl, finite_equal=same_fin, max_rel_diff=rel,
                         index_equal_frac=float((Ic == Ib).mean()), tol=tol, ok=same_fin and rel <= tol)
                ok &= r["ok"]
                rows.append(r)
                print(json.dumps(r), flush=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("CUPY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="path to bin/mdms_gpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sys.exit(0 if self_check(a.check, a.out) else 1)
