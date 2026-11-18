#!/usr/bin/env python3
"""
Run the CUDA program (or its CPU twin) and compare with the Python reference.

  python3 -I gpu/compare.py --bin bin/mdms_twin --suite t5     # T5/T6 correctness suite
  python3 -I gpu/compare.py --bin bin/mdms_gpu  --suite t5
  python3 -I gpu/compare.py --bin bin/mdms_gpu  --suite t8     # float32 numerics (T8)

Writes one JSON line per case to --out and exits nonzero if any T5/T6 case fails.

T5/T6 pass rule (Appendix I2), for f64:
  same finite pattern; |P_a^2 - P_b^2| <= 1e-9 * max(1, P_b^2);
  index identical, or the reference bound of the returned index equals the minimum (tie).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mdms_ref as M  # noqa: E402

TOL = 1e-9


def run_bin(binary, T, m, precision="f64", vlimits="window", limits=None, mode="mdms",
            band_rows=None, slack_rel=0.0, slack_abs=0.0, center=True, device=0, workdir=None,
            opt_center=False):
    workdir = workdir or tempfile.mkdtemp(prefix="mdms_")
    inp = os.path.join(workdir, "in.f64")
    out = os.path.join(workdir, "out")
    np.asarray(T, dtype=np.float64).tofile(inp)
    cmd = [binary, "--input", inp, "--out", out, "--n", str(len(T)), "--m", str(m),
           "--precision", precision, "--vlimits", vlimits, "--mode", mode,
           "--slack-rel", repr(slack_rel), "--slack-abs", repr(slack_abs)]
    if limits is not None and vlimits == "slew":
        lim = np.atleast_1d(np.asarray(limits, float))
        cmd += ["--slew", repr(float(lim[0]))]
        if lim.size == 3:
            cmd += ["--lo", repr(float(lim[1])), "--hi", repr(float(lim[2]))]
    elif limits is not None:
        cmd += ["--lo", repr(float(limits[0])), "--hi", repr(float(limits[1]))]
    if band_rows:
        cmd += ["--band-rows", str(band_rows)]
    if not center:
        cmd += ["--no-center"]
    if opt_center:
        cmd += ["--opt-center"]
    if "mdms_gpu" in os.path.basename(binary):
        cmd += ["--device", str(device)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    P = np.fromfile(out + ".P.f64", dtype=np.float64)
    I = np.fromfile(out + ".I.i32", dtype=np.int32).astype(np.int64)
    with open(out + ".stats.json") as f:
        stats = json.load(f)
    return P, I, stats


def compare_f64(P, I, T, mask, m, vlimits, limits, opt_center=False):
    ws_ref = M.window_stats(np.where(mask, T, 0.0), mask, m, vlimits, limits, opt_center)
    Pr, Ir = profile_from_ws(ws_ref)
    fin, finr = np.isfinite(P), np.isfinite(Pr)
    res = dict(finite_pattern_equal=bool(np.array_equal(fin, finr)))
    a, b = P[finr] ** 2, Pr[finr] ** 2
    err = np.abs(a - b) / np.maximum(1.0, b) if a.size else np.array([0.0])
    res["max_scaled_err"] = float(np.nanmax(err)) if err.size else 0.0
    ws = ws_ref
    bad_idx = 0
    for i in np.flatnonzero(I != Ir):
        if not finr[i]:
            if I[i] != -1:
                bad_idx += 1
            continue
        if I[i] < 0 or abs(int(I[i]) - i) <= M.excl_zone(m):
            bad_idx += 1
            continue
        d = M.pair_lb_d2(ws, i, int(I[i]))
        if abs(d - Pr[i] ** 2) > TOL * max(1.0, Pr[i] ** 2):
            bad_idx += 1
    res["index_mismatch_not_tie"] = int(bad_idx)
    res["index_differs"] = int((I != Ir).sum())
    res["pass"] = bool(res["finite_pattern_equal"] and res["max_scaled_err"] <= TOL and bad_idx == 0)
    return res


def profile_from_ws(ws, excl=0.25):
    L, m = ws["L"], ws["m"]
    P2 = np.full(L, np.inf)
    I = np.full(L, -1, dtype=np.int64)
    for k in range(M.excl_zone(m, excl) + 1, L):
        d2 = M.diagonal_lb_d2(ws, k)
        ii = np.arange(L - k)
        M._update_profile(P2, I, ii, ii + k, d2)
        M._update_profile(P2, I, ii + k, ii, d2)
    return np.sqrt(P2), I


def t5_cases():
    """(name, T, mask, m, vlimits, limits, band_rows) -- T5 and T6 cases."""
    cases = []
    rng = np.random.default_rng(0)
    for n, m in [(500, 8), (4096, 64), (20000, 300)]:
        T = np.cumsum(rng.standard_normal(n))
        for pat in ["random", "block", "mixed"]:
            mask, _ = M.generate_missing_mask(n, m, 0.1, pat, 0.2, seed=n + m)
            for vl in ["window", "global", "declared", "slew"]:
                lim = ((T.min() - 1, T.max() + 1) if vl == "declared" else
                       float(np.max(np.abs(np.diff(T)))) if vl == "slew" else None)
                if n == 20000 and vl != "window":
                    continue                      # keep the CPU reference affordable
                cases.append((f"t5_n{n}_m{m}_{pat}_{vl}", T, mask, m, vl, lim, None))
            if n != 20000:
                ours = (float(np.max(np.abs(np.diff(T)))), T.min(), T.max())
                cases.append((f"t5_n{n}_m{m}_{pat}_ours", T, mask, m, "slew", ours, "opt"))
    # T6 edge cases
    n, m = 600, 16
    T = np.cumsum(rng.standard_normal(n))
    mask = np.ones(n, bool)
    mask[100:140] = False                      # windows with no observed value
    mask[300:304] = False
    T2 = T.copy()
    T2[200:240] = 3.0                          # flat complete windows
    cases.append(("t6_all_missing_and_flat", T2, mask, m, "window", None, None))
    mask2, _ = M.generate_missing_mask(n, m, 0.3, "block", 1.0, seed=5)
    cases.append(("t6_band_boundaries_small_band", T, mask2, m, "window", None, 37))
    cases.append(("t6_band_rows_eq_1", T[:200], np.ones(200, bool), 8, "window", None, 1))
    cases.append(("t6_n_close_to_m", T[:20], np.ones(20, bool), 8, "window", None, None))
    cases.append(("t6_no_missing", T, np.ones(n, bool), m, "window", None, None))
    return cases


def suite_t5(binary, out, device):
    ok = True
    rows = []
    for name, T, mask, m, vl, lim, band in t5_cases():
        Tin = np.where(mask, T, np.nan)
        t = time.time()
        opt = band == "opt"
        band = None if opt else band
        P, I, st = run_bin(binary, Tin, m, "f64", vl, lim, band_rows=band, device=device,
                           opt_center=opt)
        r = compare_f64(P, I, T, mask, m, vl, lim, opt_center=opt)
        r.update(case=name, n=len(T), m=m, vlimits=vl, backend=st["backend"],
                 device=st["device_name"], kernel_ms=st["kernel_ms"], seconds=time.time() - t)
        rows.append(r)
        ok &= r["pass"]
        print(json.dumps(r), flush=True)
    # stomp mode on complete data must equal the oracle profile
    rng = np.random.default_rng(1)
    for n, m in [(3000, 100), (500, 8)]:
        T = np.cumsum(rng.standard_normal(n))
        P, I, st = run_bin(binary, T, m, "f64", mode="stomp", device=device)
        Pr, Ir = M.mdms_profile(T, np.ones(n, bool), m)
        err = float(np.max(np.abs(P ** 2 - Pr ** 2) / np.maximum(1.0, Pr ** 2)))
        r = dict(case=f"stomp_vs_ref_n{n}_m{m}", max_scaled_err=err, pass_=err <= TOL,
                 index_differs=int((I != Ir).sum()), backend=st["backend"])
        r["pass"] = bool(r.pop("pass_"))
        ok &= r["pass"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return ok


def suite_t8(binary, out, device, data_dir=None):
    """float32 numerics: max relative excess over the f64 bound and over the oracle."""
    sys.path.insert(0, os.path.join(HERE, "..", "audit"))
    from c3_audit import load_datasets, make_masks
    rows = []
    n, m = 4000, 100
    data = load_datasets(data_dir, n) if data_dir else {}
    rng = np.random.default_rng(2)
    data.setdefault("syn_random_walk", np.cumsum(rng.standard_normal(n)))
    for dname, Tc in data.items():
        if dname not in ("cwru_IR007", "cwru_normal", "syn_random_walk"):
            continue
        mask = make_masks(Tc, m, 0)["random_1pct"]
        for offset_sd in (0.0, 100.0):
            T = Tc + offset_sd * Tc.std()
            lim = (float(T.min()), float(T.max()))
            Tin = np.where(mask, T, np.nan)
            P64, _, _ = run_bin(binary, Tin, m, "f64", "declared", lim, device=device)
            for prec, center in (("f32", False), ("f32", True), ("f32c", True)):
                for slack in (0.0, 1e-4, 1e-3):
                    P, _, st = run_bin(binary, Tin, m, prec, "declared", lim, slack_rel=slack,
                                       center=center, device=device)
                    fin = np.isfinite(P) & np.isfinite(P64) & (P64 > 0)
                    rel = (P[fin] ** 2 / P64[fin] ** 2) - 1.0
                    r = dict(dataset=dname, offset_sd=offset_sd, precision=prec, centered=center,
                             slack_rel=slack, rows=int(fin.sum()),
                             rows_above_f64=int((rel > 0).sum()),
                             max_rel_excess=float(rel.max()) if rel.size else 0.0,
                             median_abs_rel=float(np.median(np.abs(rel))) if rel.size else 0.0,
                             kernel_ms=st["kernel_ms"], backend=st["backend"])
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--suite", choices=["t5", "t8"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.suite == "t5":
        ok = suite_t5(a.bin, a.out, a.device)
        print("T5/T6:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    suite_t8(a.bin, a.out, a.device, a.data_dir)


if __name__ == "__main__":
    main()
