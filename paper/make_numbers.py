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


def tex_escape(t):
    for a, b in (("\\", " "), ("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#"),
                 ("^", ""), ("$", ""), ("{", ""), ("}", ""), ("~", " ")):
        t = t.replace(a, b)
    return t


# alt text for every figure, as macros for \includegraphics[alt=...] and as a plain file
for key, txt in alt.items():
    name = "Alt" + "".join(w.capitalize() for w in key.split("_"))
    put(name, tex_escape(txt))
with open(os.path.join(ROOT, "paper", "figures", "alt_text.txt"), "w") as fh:
    for key, txt in alt.items():
        fh.write(f"{key}:\n{txt}\n\n")

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
if sp:
    def pick(mode, prec, n=None):
        rs = [r for r in sp if r["mode"] == mode and r["precision"] == prec and (n is None or r["n"] == n)]
        return rs
    big = max(r["n"] for r in pick("mdms", "f32c"))
    put("SpMaxN", fmt_int(big))
    put("SpMaxNSec", f"{pick('mdms', 'f32c', big)[0]['kernel_ms'] / 1000:.0f}")
    put("SpMaxNPairs", f"{pick('mdms', 'f32c', big)[0]['pairs'] / 1e9:.0f}")
    put("SpMdmsPeak", f"{max(r['gpairs_per_s'] for r in pick('mdms', 'f32c')):.1f}")
    put("SpStompPeak", f"{max(r['gpairs_per_s'] for r in pick('stomp', 'f32c')):.1f}")
    common = sorted(set(r["n"] for r in pick("mdms", "f32c")) & set(r["n"] for r in pick("stomp", "f32c")))
    nb = max(common)
    put("SpRatioN", fmt_int(nb))
    put("SpMdmsOverStomp", f"{pick('mdms', 'f32c', nb)[0]['kernel_ms'] / pick('stomp', 'f32c', nb)[0]['kernel_ms']:.2f}")
    put("SpFSixtyFourRate", f"{max(r['gpairs_per_s'] for r in pick('mdms', 'f64')):.2f}")
    n64 = max(r["n"] for r in pick("mdms", "f64"))
    put("SpFSixtyFourN", fmt_int(n64))
    put("SpFcOverFSixtyFour", f"{pick('mdms', 'f64', n64)[0]['kernel_ms'] / pick('mdms', 'f32c', n64)[0]['kernel_ms']:.0f}")
    tw = pick("twin_mdms", "f32c")
    nt = max(r["n"] for r in tw)
    put("SpTwinRate", f"{max(r['gpairs_per_s'] for r in sp if r['mode'] == 'twin_mdms'):.3f}")
    put("SpTwinN", fmt_int(nt))
    put("SpGpuOverTwin", f"{pick('twin_mdms', 'f32c', nt)[0]['kernel_ms'] / pick('mdms', 'f32c', nt)[0]['kernel_ms']:.0f}")
    st = [r for r in sp if r["mode"] == "stumpy_gpu_stump"]
    if st:
        ns = 65536
        s_ = [r for r in st if r["n"] == ns][0]
        put("SpStumpyN", fmt_int(ns))
        put("SpStumpySec", f"{s_['kernel_ms'] / 1000:.1f}")
        put("SpStumpyVersion", s_["backend"].split()[-1])
        put("SpOursStompSixtyFourWall", f"{pick('stomp', 'f64', ns)[0]['wall_s']:.1f}")
        put("SpOursMdmsFcWall", f"{pick('mdms', 'f32c', ns)[0]['wall_s']:.1f}")

# plain (uncompensated) centered float32 with the recommended margin, on the GPU
t8g = os.path.join(RES, "gpu", "t8_gpu.jsonl")
if os.path.exists(t8g):
    T8 = [json.loads(l) for l in open(t8g) if l.strip()]
    put("TEGpuPlainSlackAbove", str(sum(r["rows_above_f64"] for r in T8 if r["precision"] == "f32"
                                       and r["centered"] and r["slack_rel"] == 0.001)))

# ---------------- bench2: window lengths, f64 at 2^18, fair STUMPY comparison ----------------
s2 = []
for fn in ("speed2_d0.jsonl", "speed2_d1.jsonl"):
    p_ = os.path.join(RES, "gpu", fn)
    if os.path.exists(p_):
        s2 += [json.loads(l) for l in open(p_) if l.strip()]
if s2:
    big = [r for r in s2 if r["mode"] == "mdms" and r["n"] == 2 ** 20]
    put("SpTwoMinGps", f"{min(r['gpairs_per_s'] for r in big):.1f}")
    put("SpTwoMaxGps", f"{max(r['gpairs_per_s'] for r in big):.1f}")
    put("SpTwoMs", ", ".join(str(m) for m in sorted(set(r["m"] for r in big))))
    f64 = [r for r in s2 if r["mode"] == "mdms" and r["precision"] == "f64"][0]
    put("SpFSixtyFourBigN", fmt_int(f64["n"]))
    put("SpFSixtyFourBigSec", f"{f64['kernel_ms'] / 1000:.0f}")
sf = os.path.join(RES, "gpu", "stumpy_fair.jsonl")
if os.path.exists(sf):
    F = [json.loads(l) for l in open(sf) if l.strip()]
    for r, tag in zip(sorted(F, key=lambda r: r["n"]), ("Small", "Mid", "Big")):
        put(f"SfN{tag}", fmt_int(r["n"]))
        put(f"SfStumpy{tag}", f"{r['stumpy_wall_s']:.1f}")
        put(f"SfOurs{tag}", f"{r['ours_stomp_f64_wall_s']:.2f}" if r["ours_stomp_f64_wall_s"] < 1 else f"{r['ours_stomp_f64_wall_s']:.1f}")
        put(f"SfRatio{tag}", f"{r['stumpy_wall_s'] / r['ours_stomp_f64_wall_s']:.1f}")
    put("SfMaxDiff", sci(max(r["profile_max_abs_diff"] for r in F)))
    put("SfIdxEq", "all" if all(r["index_equal_frac"] == 1.0 for r in F) else "not all")
    put("SfVersion", F[0]["stumpy_version"])

# ---------------- GPU validation (smoke + validate jobs) ----------------
ag = os.path.join(RES, "gpu", "agree_smoke.jsonl")
if os.path.exists(ag):
    A = [json.loads(l) for l in open(ag) if l.strip()]
    put("GpuAgreeCases", str(len(A)))
    put("GpuAgreeBitwise", str(sum(r["bitwise_equal_frac"] == 1.0 and r["index_equal_frac"] == 1.0 for r in A)))
t5g = os.path.join(RES, "gpu", "t5_gpu.jsonl")
if os.path.exists(t5g):
    G5 = [json.loads(l) for l in open(t5g) if l.strip()]
    put("GpuTFiveCases", str(len(G5)))
    put("GpuTFivePass", str(sum(bool(r.get("pass")) for r in G5)))
    put("GpuTFiveMaxErr", sci(max(r["max_scaled_err"] for r in G5)))
sz = os.path.join(RES, "gpu", "sanitizer.txt")
if os.path.exists(sz):
    txt = open(sz).read()
    put("GpuSanRuns", str(txt.count("\n== ") + txt.startswith("== ")))
    put("GpuSanErrors", "0" if ("sanitizer: all clean" in txt and "FAILED" not in txt) else "see log")

with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as fh:
    fh.write("% generated by paper/make_numbers.py -- do not edit\n")
    for k in sorted(out):
        fh.write(f"\\newcommand{{\\{k}}}{{{out[k]}}}\n")
print(f"{len(out)} macros")
