#!/usr/bin/env python3
"""Figures for the paper (PDF, print-safe). Every figure is drawn from result files only.
Colours: validated categorical slots 1-3 (blue, orange, aqua); every series also has its
own marker and line style, so identity never depends on colour alone."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})
ALT = {}


def save(fig, name, alt):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    ALT[name] = alt


def fig_violations(c3csv):
    rows = list(csv.DictReader(open(c3csv)))
    clips = [("clip_q99.5", 0.5), ("clip_q99", 1.0), ("clip_q98", 2.0), ("clip_q95", 5.0)]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
    alt_parts = []
    for ax, (ds, title) in zip(axes, [("cwru_IR007", "CWRU inner-race fault"),
                                      ("cwru_OR007", "CWRU outer-race fault")]):
        for mode, lab, col, mk, ls in [("window", "per-window observed limits (MDMS practice)", C2, "o", "-"),
                                       ("global", "global observed limits", C1, "s", "--"),
                                       ("declared_both", "declared limits (ours)", C3, "^", ":")]:
            ys = []
            for mname, x in clips:
                r = next(r for r in rows if r["dataset"] == ds and r["mask"] == mname and r["mode"] == mode)
                ys.append(100 * float(r["row_viol"]) / float(r["rows"]))
            ax.plot([c[1] for c in clips], ys, color=col, marker=mk, ls=ls, ms=4, label=lab)
            alt_parts.append(f"{title}, {lab}: " + ", ".join(f"{y:.0f}%" for y in ys))
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("samples lost to saturation (%)")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.minorticks_off()
    axes[0].set_ylabel("windows whose nearest-neighbour\nbound exceeds the true distance (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_violations",
         "Two line charts, one per CWRU fault recording. The x-axis is the share of samples lost "
         "to simulated saturation (0.5, 1, 2 and 5 percent); the y-axis is the share of windows "
         "whose lower bound on the nearest-neighbour distance exceeds the true distance. "
         + "; ".join(alt_parts) + ".")


def fig_counterexample(t7json):
    d = json.load(open(t7json))["part_a"]["rows"]
    h = [r["spike"] for r in d]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(h, [r["true_d"] for r in d], color=C1, marker="o", ms=4, label="true distance")
    ax.plot(h, [r["lb_window"] for r in d], color=C2, marker="s", ms=4, ls="--",
            label="bound, per-window limits")
    ax.plot(h, [r["lb_declared"] for r in d], color=C3, marker="^", ms=4, ls=":",
            label="bound, declared limits")
    ax.set_xlabel("height of the hidden spike")
    ax.set_ylabel("z-normalized distance")
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.64))
    save(fig, "fig_counterexample",
         "Line chart of the two-window counterexample. As the hidden spike grows from 1 to 10, the "
         "true distance falls from {:.2f} to {:.2f}; the bound with per-window limits stays at {:.2f} "
         "and exceeds the true distance from spike height 1.5 on; the bound with declared limits stays "
         "at {:.2f}, below the true distance.".format(d[0]["true_d"], d[-1]["true_d"],
                                                       d[0]["lb_window"], d[0]["lb_declared"]))


def fig_coverage(m=100, n=20000, seeds=5):
    sys.path.insert(0, ROOT)
    import mdms_ref as M
    from numpy.lib.stride_tricks import sliding_window_view
    rates = [0.1, 0.2, 0.5, 1, 2, 5]
    meas = []
    for p in rates:
        v = [sliding_window_view(M.generate_missing_mask(n, m, p / 100, "random", seed=s)[0], m)
             .all(axis=1).mean() for s in range(seeds)]
        meas.append(100 * float(np.mean(v)))
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(rates, meas, color=C2, marker="o", ms=4, label="exclusion (measured)")
    ax.plot(rates, [100 * (1 - p / 100) ** m for p in rates], color=C1, ls="--", label=r"exclusion, $(1-p)^m$")
    ax.plot(rates, [100.0] * len(rates), color=C3, ls=":", marker="^", ms=4, label="MDMS / imputation")
    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel("random missing rate p (%)")
    ax.set_ylabel("windows with a score (%)")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "fig_coverage",
         "Line chart of coverage against the random missing rate (0.1 to 5 percent, log scale) for "
         "window length 100. The share of windows exclusion can score is "
         + ", ".join(f"{v:.1f}" for v in meas)
         + " percent, matching (1-p)^m; MDMS and imputation score every window.")
    return meas


def fig_speed(jsonls):
    rows = [json.loads(l) for f in jsonls if os.path.exists(f) for l in open(f) if l.strip()]
    rows = [r for r in rows if "kernel_ms" in r and r.get("mode") in ("mdms", "stomp", "twin_mdms")]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    series = [("mdms", "f32c", "MDMS GPU, f32c", C1, "o", "-"),
              ("mdms", "f64", "MDMS GPU, f64", C2, "s", "--"),
              ("stomp", "f32c", "STOMP GPU, f32c (same code)", C3, "^", ":")]
    alt = []
    for mode, prec, lab, col, mk, ls in series:
        pts = sorted((r["n"], r["kernel_ms"] / 1000) for r in rows if r["mode"] == mode and r["precision"] == prec)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ls=ls, ms=4, label=lab)
            alt.append(lab + ": " + ", ".join(f"n={p[0]} {p[1]:.2g}s" for p in pts))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("series length n")
    ax.set_ylabel("kernel time (s)")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    save(fig, "fig_speed", "Log-log line chart of GPU kernel time against series length on a Tesla T4. "
         + "; ".join(alt) + ".")
    return True


def main():
    fig_violations(os.path.join(ROOT, "results", "v2", "c3_audit_n3000_m100.csv"))
    fig_counterexample(os.path.join(ROOT, "results", "t7_counterexample.json"))
    fig_coverage()
    fig_speed([os.path.join(ROOT, "results", "gpu", f) for f in ("speed_d0.jsonl", "speed_d1.jsonl")])
    with open(os.path.join(FIG, "alt_text.json"), "w") as f:
        json.dump(ALT, f, indent=2)
    print("\n".join(f"{k}: {v}" for k, v in ALT.items()))


if __name__ == "__main__":
    main()
