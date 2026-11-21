#!/usr/bin/env python3
"""Write paper/numbers.tex: every number quoted in the paper, computed from result files.
Re-run after any experiment; the paper only uses these macros (no hand-typed results)."""
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
out = {}


def put(name, value):
    out[name] = value


def sci(x):
    if not np.isfinite(x):
        return "n/a"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_int(x):
    return f"{int(x):,}".replace(",", "{,}")


# ---------------- C3 audit (main run, m = 100) ----------------
c3 = list(csv.DictReader(open(os.path.join(RES, "v2", "c3_audit_n3000_m100.csv"))))
modes = sorted(set(r["mode"] for r in c3))
runs = len(set((r["dataset"], r["mask"]) for r in c3))
pairs_run = int(c3[0]["pairs"])
put("AuditRuns", runs)
put("AuditPairsPerRun", fmt_int(pairs_run))
put("AuditPairsPerMode", fmt_int(sum(int(r["pairs"]) for r in c3 if r["mode"] == "window")))
for mode, key in [("window", "Window"), ("global", "Global"), ("declared_true", "DeclTrue"),
                  ("declared_wide", "DeclWide"), ("declared_slew", "Slew"), ("declared_both", "Ours")]:
    put(f"Viol{key}Total", fmt_int(sum(int(r["pair_viol"]) for r in c3 if r["mode"] == mode)))
put("ViolWindowCaseThree", fmt_int(sum(int(r["viol_case3"]) for r in c3 if r["mode"] == "window")))
put("ViolWindowCaseOneTwo", fmt_int(sum(int(r["viol_case1"]) + int(r["viol_case2"]) for r in c3 if r["mode"] == "window")))
put("ViolWindowRandom", fmt_int(sum(int(r["pair_viol"]) for r in c3 if r["mode"] == "window" and r["mask"].startswith("random"))))


def row(ds, mask, mode):
    return next(r for r in c3 if r["dataset"] == ds and r["mask"] == mask and r["mode"] == mode)


for ds, key in [("cwru_IR007", "IR"), ("cwru_OR007", "OR")]:
    for mk, mkey in [("clip_q95", "Five"), ("clip_q99.5", "Half"), ("clip_q98", "Two")]:
        w = row(ds, mk, "window")
        g = row(ds, mk, "global")
        put(f"{key}{mkey}PairViol", fmt_int(w["pair_viol"]))
        put(f"{key}{mkey}RowViol", fmt_int(w["row_viol"]))
        put(f"{key}{mkey}Rows", fmt_int(w["rows"]))
        put(f"{key}{mkey}RowViolPct", f"{100 * int(w['row_viol']) / int(w['rows']):.0f}")
        put(f"{key}{mkey}GlobalRowViolPct", f"{100 * int(g['row_viol']) / int(g['rows']):.0f}")
        put(f"{key}{mkey}GlobalPairViol", fmt_int(g["pair_viol"]))
        put(f"{key}{mkey}TopLost", w["topk_lost"])
        put(f"{key}{mkey}MaxRatio", f"{float(w['max_ratio']):.2f}")
        put(f"{key}{mkey}TightWindow", f"{float(w['tightness_median']):.2f}")
        put(f"{key}{mkey}TightDecl", f"{float(row(ds, mk, 'declared_true')['tightness_median']):.2f}")
        put(f"{key}{mkey}TightOurs", f"{float(row(ds, mk, 'declared_both')['tightness_median']):.2f}")
        put(f"{key}{mkey}TightWide", f"{float(row(ds, mk, 'declared_wide')['tightness_median']):.2f}")
for mk, mkey in [("random_1pct", "RandOne"), ("random_5pct", "RandFive")]:
    for ds, key in [("syn_random_walk", "RW"), ("cwru_IR007", "IR")]:
        put(f"{key}{mkey}TightWindow", f"{float(row(ds, mk, 'window')['tightness_median']):.2f}")
        put(f"{key}{mkey}TightDecl", f"{float(row(ds, mk, 'declared_true')['tightness_median']):.2f}")
        put(f"{key}{mkey}TightOurs", f"{float(row(ds, mk, 'declared_both')['tightness_median']):.2f}")
r = row("syn_impulse_train", "block_10pct_b30", "window")
put("ImpBlockPairViol", fmt_int(r["pair_viol"]))
put("ImpBlockMaxRatio", f"{float(r['max_ratio']):.2f}")
r = row("syn_random_walk", "block_10pct_b10", "window")
put("RWBlockPairViol", fmt_int(r["pair_viol"]))
put("BlockRowViolTotal", fmt_int(sum(int(r["row_viol"]) for r in c3 if r["mode"] == "window" and r["mask"].startswith("block"))))
nb = [r for r in c3 if r["mode"] == "window" and r["dataset"] in ("cwru_B007", "cwru_normal") and r["mask"].startswith("clip")]
put("BallNormalClipRowViolMax", fmt_int(max(int(r["row_viol"]) for r in nb)))

# robustness runs (window vs declared only)
for fn, key in [("c3_audit_n3000_m200.csv", "MTwo"), ("c3_audit_n3000_m100_seed1.csv", "SeedOne")]:
    p = os.path.join(RES, fn)
    rr = list(csv.DictReader(open(p)))
    put(f"Rob{key}DeclViol", fmt_int(sum(int(x["pair_viol"]) for x in rr if x["mode"].startswith("declared"))))
    w = next(x for x in rr if x["dataset"] == "cwru_IR007" and x["mask"] == "clip_q95" and x["mode"] == "window")
    put(f"Rob{key}IRFiveRowViol", fmt_int(w["row_viol"]))
    put(f"Rob{key}IRFiveRows", fmt_int(w["rows"]))
    w = next(x for x in rr if x["dataset"] == "cwru_OR007" and x["mask"] == "clip_q95" and x["mode"] == "window")
    put(f"Rob{key}ORFiveRowViol", fmt_int(w["row_viol"]))
    put(f"Rob{key}ORFiveRows", fmt_int(w["rows"]))

# ---------------- T7 ----------------
t7 = json.load(open(os.path.join(RES, "t7_counterexample.json")))
a = t7["part_a"]["rows"]
put("CexLBWindow", f"{a[0]['lb_window']:.3f}")
put("CexLBDecl", f"{a[0]['lb_declared']:.3f}")
put("CexTrueFirst", f"{a[0]['true_d']:.3f}")
put("CexTrueLast", f"{a[-1]['true_d']:.3f}")
put("CexRatioLast", f"{a[-1]['ratio_window']:.1f}")
b = t7["part_b"]
ms = sorted(int(k) for k, v in b.items() if v["violations"] > 0)
put("CexMinM", ms[0])
put("CexTrials", fmt_int(b[str(ms[0])]["trials"]))
put("CexMaxRatio", f"{max(v['best']['ratio'] for v in b.values() if v['best']):.0f}")

# ---------------- T8 (CPU twin of the GPU code; GPU run overrides if present) ----------------
for fn, key in [("t8_twin.jsonl", "Twin"), (os.path.join("gpu", "t8_gpu.jsonl"), "Gpu")]:
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        continue
    t8 = [json.loads(l) for l in open(p) if l.strip()]

    def mx(cond):
        v = [x["max_rel_excess"] for x in t8 if cond(x)]
        return max(v) if v else float("nan")
    put(f"TE{key}FcCwru", sci(mx(lambda x: x['precision']=='f32c' and x['slack_rel']==0 and x['dataset'].startswith('cwru'))))
    put(f"TE{key}FcRw", sci(mx(lambda x: x['precision']=='f32c' and x['slack_rel']==0 and x['dataset']=='syn_random_walk')))
    put(f"TE{key}PlainRw", sci(mx(lambda x: x['precision']=='f32' and x['centered'] and x['slack_rel']==0 and x['dataset']=='syn_random_walk')))
    put(f"TE{key}RawOffset", f"{100*mx(lambda x: x['precision']=='f32' and not x['centered'] and x['offset_sd']==100.0 and x['slack_rel']==0):.0f}")
    put(f"TE{key}FcSlackAbove", fmt_int(sum(x["rows_above_f64"] for x in t8 if x["precision"] == "f32c" and x["slack_rel"] == 1e-3)))
    put(f"TE{key}FcSlackSmallAbove", fmt_int(sum(x["rows_above_f64"] for x in t8 if x["precision"] == "f32c" and x["slack_rel"] == 1e-4)))

# ---------------- coverage ----------------
alt = json.load(open(os.path.join(ROOT, "paper", "figures", "alt_text.json")))
put("CovOneTenth", "90")    # overwritten below from the figure data when available
cov_txt = alt["fig_coverage"]
nums = [float(x) for x in cov_txt.split("is ")[1].split(" percent")[0].split(", ")]
put("CovOneTenth", f"{nums[0]:.0f}")
put("CovOne", f"{nums[3]:.0f}")
put("CovFive", f"{nums[5]:.1f}")

# ---------------- C2 ----------------
c2 = []
for fn in ("c2_cwru.csv", "c2_planted.csv"):
    p = os.path.join(RES, "c2", fn)
    if os.path.exists(p):
        c2 += list(csv.DictReader(open(p)))


def fam(mk):
    return "random" if mk.startswith("random") else ("block" if mk.startswith("block") else "clip")


agg = defaultdict(list)
for r in c2:
    grp = "cwru" if r["series"].startswith("cwru") else "planted"
    agg[(grp, fam(r["mask"]), r["method"])].append(r)
table = {}
for (grp, f, meth), rs in agg.items():
    d = dict(n=len(rs),
             top1=float(np.mean([x["top1_hit"] == "True" for x in rs])),
             topk=float(np.mean([float(x["topk_recall"]) for x in rs])),
             cov=float(np.mean([float(x["coverage"]) for x in rs])))
    if grp == "planted":
        d["planted"] = float(np.mean([x["planted_hit"] == "True" for x in rs]))
    if meth.startswith("mdms"):
        d["guar"] = float(np.mean([x["guarantee_holds"] == "True" for x in rs]))
        d["cand"] = float(np.median([int(x["candidates"]) / int(x["total_pairs"]) for x in rs]))
    table[(grp, f, meth)] = d
with open(os.path.join(RES, "c2", "c2_summary.json"), "w") as fh:
    json.dump({"|".join(k): v for k, v in table.items()}, fh, indent=2)
put("CTwoRows", fmt_int(len(c2)))
put("CTwoMasks", fmt_int(len(set((r["series"], r["mask"], r["seed"]) for r in c2))))
for grp in ("cwru", "planted"):
    g = "Cw" if grp == "cwru" else "Pl"
    for f in ("random", "block", "clip"):
        F = f.capitalize()
        for meth, mk in [("exclusion", "Ex"), ("linear", "Lin"), ("spline", "Spl"), ("locf", "Locf"),
                         ("mdms_window", "Win"), ("mdms_declared", "Decl"), ("mdms_ours", "Ours")]:
            d = table.get((grp, f, meth))
            if not d:
                continue
            put(f"C{g}{F}{mk}TopOne", f"{100*d['top1']:.0f}")
            put(f"C{g}{F}{mk}TopK", f"{100*d['topk']:.0f}")
            if "planted" in d:
                put(f"C{g}{F}{mk}Planted", f"{100*d['planted']:.0f}")
            if "guar" in d:
                put(f"C{g}{F}{mk}Guar", f"{100*d['guar']:.0f}")
                put(f"C{g}{F}{mk}Cand", f"{100*d['cand']:.2g}")

# per-pattern numbers quoted in the C2 text
per = defaultdict(list)
for r in c2:
    grp = "cwru" if r["series"].startswith("cwru") else "planted"
    per[(grp, r["mask"], r["method"])].append(r)


def pm(grp, mask, meth, what):
    rs = per.get((grp, mask, meth), [])
    if not rs:
        return "--"
    if what == "cand":
        return f"{100*np.median([int(x['candidates']) / int(x['total_pairs']) for x in rs]):.2g}"
    key = {"top1": "top1_hit", "planted": "planted_hit", "guar": "guarantee_holds"}[what]
    return f"{100*np.mean([x[key] == 'True' for x in rs]):.0f}"


for name, args in {
    "CwRandFiveCandDecl": ("cwru", "random_5", "mdms_declared", "cand"),
    "CwRandFiveCandOurs": ("cwru", "random_5", "mdms_ours", "cand"),
    "CwRandFortyCandDecl": ("cwru", "random_40", "mdms_declared", "cand"),
    "CwBlockFortyCandDecl": ("cwru", "block_40_b30", "mdms_declared", "cand"),
    "CwRandFiveTopSpl": ("cwru", "random_5", "spline", "top1"),
    "CwRandFiveTopLin": ("cwru", "random_5", "linear", "top1"),
    "CwRandFiveTopOurs": ("cwru", "random_5", "mdms_ours", "top1"),
    "CwRandTwentyTopLin": ("cwru", "random_20", "linear", "top1"),
    "PlRandFiveCandDecl": ("planted", "random_5", "mdms_declared", "cand"),
    "PlRandFiveCandOurs": ("planted", "random_5", "mdms_ours", "cand"),
    "PlRandFivePlDecl": ("planted", "random_5", "mdms_declared", "planted"),
    "PlRandFivePlOurs": ("planted", "random_5", "mdms_ours", "planted"),
    "PlRandFivePlSpl": ("planted", "random_5", "spline", "planted"),
    "PlRandFivePlLin": ("planted", "random_5", "linear", "planted"),
    "PlRandFortyPlSpl": ("planted", "random_40", "spline", "planted"),
    "PlRandFortyPlLin": ("planted", "random_40", "linear", "planted"),
    "PlBlockTwentyThirtyPlSpl": ("planted", "block_20_b30", "spline", "planted"),
    "PlClipFivePlLin": ("planted", "clip_q95", "linear", "planted"),
    "PlClipFivePlOurs": ("planted", "clip_q95", "mdms_ours", "planted"),
    "PlClipFiveGuarWin": ("planted", "clip_q95", "mdms_window", "guar"),
    "CwClipFiveGuarWin": ("cwru", "clip_q95", "mdms_window", "guar"),
}.items():
    put(name, pm(*args))

# ---------------- wider audit: 40 recordings, history-based limits ----------------
wp = os.path.join(RES, "wide", "wide_audit_n3000_m100.csv")
if os.path.exists(wp):
    W = list(csv.DictReader(open(wp)))
    recs = sorted(set(r["recording"] for r in W))
    put("WideRecs", str(len(recs)))
    put("WidePairsPerRun", fmt_int(int(next(r for r in W if r["mode"] == "window")["pairs"])))

    def sel(mask=None, mode=None, accepted=True):
        return [r for r in W if (mask is None or r["mask"] == mask)
                and (mode is None or r["mode"] == mode)
                and (not accepted or r["rejected"] == "0")]

    def kind(name):
        return "IR" if "_IR_" in name else "OR" if "_OR" in name else "B" if "_B_" in name else "N"

    for mk, tag in (("clip_q95", "QFive"), ("clip_q99", "QOne")):
        for mode, mt in (("window", "Win"), ("global", "Glob")):
            rs = sel(mk, mode)
            put(f"Wide{mt}Runs{tag}", str(sum(int(r["row_viol"]) > 0 for r in rs)))
            put(f"Wide{mt}MaxPct{tag}",
                f"{max(100 * int(r['row_viol']) / int(r['rows']) for r in rs):.0f}")
        rs = sel(mk, "window")
        for k in ("IR", "OR", "B", "N"):
            put(f"Wide{k}Runs{tag}", str(sum(int(r["row_viol"]) > 0 for r in rs if kind(r["recording"]) == k)))
            put(f"Wide{k}Total{tag}", str(sum(1 for r in rs if kind(r["recording"]) == k)))
        put(f"WideTopLostAny{tag}", str(sum(int(r["topk_lost"]) > 0 for r in rs)))
        put(f"WideTopLostAll{tag}", str(sum(int(r["topk_lost"]) == 10 for r in rs)))
    put("WideRandBlockRowViolWin", fmt_int(sum(int(r["row_viol"]) for r in W if r["mode"] == "window"
                                                 and r["mask"] in ("random_5pct", "block_10pct_b30"))))
    put("WideBlockPairViolWin", fmt_int(sum(int(r["pair_viol"]) for r in sel("block_10pct_b30", "window"))))
    valid_modes = ("oracle_range", "oracle_ours", "hist_range", "hist_ours", "hist_ours_m10", "hist_ours_m25")
    put("WideValidViol", fmt_int(sum(int(r["pair_viol"]) + int(r["row_viol"])
                                     for r in W if r["mode"] in valid_modes and r["rejected"] == "0")))
    for mode, mt in (("hist_range", "HistRange"), ("hist_ours", "HistOurs"),
                     ("hist_ours_m10", "MTen"), ("hist_ours_m25", "MTwentyFive")):
        for mk, tag in (("random_5pct", "Rand"), ("block_10pct_b30", "Block"),
                        ("clip_q95", "QFive"), ("clip_q99", "QOne")):
            allr = sel(mk, mode, accepted=False)
            put(f"Wide{mt}Rej{tag}", str(sum(r["rejected"] == "1" for r in allr)))
            put(f"Wide{mt}HidOut{tag}", str(sum(int(r["hidden_outside"]) for r in allr if r["rejected"] == "0")))
    for mode, mt in (("window", "Win"), ("oracle_ours", "OracleOurs"), ("hist_ours_m10", "MTen"),
                     ("hist_ours_m25", "MTwentyFive")):
        for mk, tag in (("random_5pct", "Rand"), ("clip_q95", "QFive")):
            t = [float(r["tightness_median"]) for r in sel(mk, mode)]
            put(f"WideTight{mt}{tag}", f"{np.median(t):.2f}")

# ---------------- GPU speed (if available) ----------------
sp = []
for fn in ("speed_d0.jsonl", "speed_d1.jsonl"):
    p = os.path.join(RES, "gpu", fn)
    if os.path.exists(p):
        sp += [json.loads(l) for l in open(p) if l.strip()]
put("HaveGpu", "1" if sp else "0")

with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as fh:
    fh.write("% generated by paper/make_numbers.py -- do not edit\n")
    for k in sorted(out):
        fh.write(f"\\newcommand{{\\{k}}}{{{out[k]}}}\n")
print(f"{len(out)} macros")
