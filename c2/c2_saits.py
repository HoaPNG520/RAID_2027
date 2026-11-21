#!/usr/bin/env python3
"""
Learned-imputer baseline for the comparison (Sect. 8): SAITS (Du et al., Expert Systems with
Applications 2023) from PyPOTS, trained self-supervised on the observed values of each gapped
series, then the exact matrix profile of the imputed series.

Same series, masks, seeds, oracle and metrics as c2_experiment.py; writes rows with
method="saits" to a separate CSV that paper/make_numbers.py merges.

Per (series, mask): the observed values are standardized; overlapping windows of length W
(stride S) form the training samples; SAITS is trained on them (its masked-imputation and
reconstruction losses use only observed values); each missing sample's imputation is the mean
over the windows that contain it; observed samples are kept as they are.

Usage:
  python3 -I c2/c2_saits.py --data-dir DATA --out results/c2/c2_saits_cwru.csv --only cwru_IR007,...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from c2_experiment import load_series, masks_for, exact_profile, top_motifs, match  # noqa: E402

W, STRIDE = 100, 10


def impute_saits(Tc, mask, seed, epochs, threads=1):
    import torch
    from pypots.imputation import SAITS
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    mu, sd = Tc[mask].mean(), Tc[mask].std() or 1.0
    x = np.where(mask, (Tc - mu) / sd, np.nan)
    n = len(x)
    starts = list(range(0, n - W + 1, STRIDE))
    if starts[-1] != n - W:
        starts.append(n - W)
    X = np.stack([x[s:s + W] for s in starts])[:, :, None].astype(np.float32)
    model = SAITS(n_steps=W, n_features=1, n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16,
                  d_ffn=128, dropout=0.1, epochs=epochs, batch_size=32, device="cpu", verbose=False)
    model.fit({"X": X})
    imp = model.predict({"X": X})["imputation"][:, :, 0]
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for row, s in zip(imp, starts):
        acc[s:s + W] += row
        cnt[s:s + W] += 1
    y = acc / np.maximum(cnt, 1)
    return np.where(mask, Tc, y * sd + mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", default=None)
    ap.add_argument("--masks", default=None, help="comma-separated mask names to run (default all)")
    a = ap.parse_args()
    series = load_series(a.data_dir, a.n)
    if a.only:
        series = {k: v for k, v in series.items() if k in a.only.split(",")}
    keep = set(a.masks.split(",")) if a.masks else None
    tol = a.m // 4
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["series"], r["mask"], r["seed"]))
    keys = ["series", "mask", "seed", "method", "missing_rate", "frac_windows_incomplete", "coverage",
            "top1_hit", "topk_recall", "n_motifs_found", "planted_hit", "impute_rmse", "seconds"]
    first = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        if first:
            wr.writeheader()
        for name, (Tc, starts) in series.items():
            P2o, Io = exact_profile(Tc, a.m)
            top_o = top_motifs(P2o, Io, a.m)
            for mname, seed, mask in masks_for(Tc, a.m, a.seeds):
                if keep and mname not in keep:
                    continue
                if (name, mname, str(seed)) in done:
                    continue
                t = time.time()
                Ti = impute_saits(Tc, mask, seed, a.epochs)
                P2, I = exact_profile(Ti, a.m)
                mot = top_motifs(P2, I, a.m)
                complete = np.lib.stride_tricks.sliding_window_view(mask, a.m).all(axis=1)
                r = dict(series=name, mask=mname, seed=seed, method="saits",
                         missing_rate=float((~mask).mean()), frac_windows_incomplete=float((~complete).mean()),
                         coverage=float(np.isfinite(P2).mean()),
                         top1_hit=bool(mot and match(mot[0], top_o[0], tol)),
                         topk_recall=float(np.mean([any(match(q, p, tol) for p in mot) for q in top_o])),
                         n_motifs_found=len(mot),
                         planted_hit=(bool(mot and any(abs(mot[0][0] - s) <= tol for s in starts)
                                           and any(abs(mot[0][1] - s) <= tol for s in starts))
                                      if starts is not None else ""),
                         impute_rmse=float(np.sqrt(np.mean((Ti[~mask] - Tc[~mask]) ** 2)) / Tc.std()),
                         seconds=round(time.time() - t, 1))
                wr.writerow(r)
                fh.flush()
                print(f"[{time.time() - t0:7.1f}s] {name} {mname} s{seed}: top1={int(r['top1_hit'])} "
                      f"planted={r['planted_hit']} nrmse={r['impute_rmse']:.3f} ({r['seconds']}s)", flush=True)
    print(json.dumps(dict(done=True, seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
