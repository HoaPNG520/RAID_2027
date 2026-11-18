#!/usr/bin/env python3
"""Turn the audit CSVs into the markdown tables used in RESULTS_C3.md."""
import csv, collections, sys
a = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(dict)
for r in a:
    by[(r["dataset"], r["mask"])][r["mode"]] = r
print("| dataset | missingness | missing % | pair violations (window / global / declared-true / declared-wide / slew+range) | row violations (window) | oracle top-10 pairs lost (window) | max LB/true (window) | median tightness (window / declared-true / slew+range) | exclusion coverage |")
print("|---|---|---|---|---|---|---|---|---|")
for (d, mk), R in by.items():
    w, g, dt, dw = R["window"], R["global"], R["declared_true"], R["declared_wide"]
    sl = R.get("declared_both", R.get("declared_slew", {"pair_viol": "-", "tightness_median": "nan"}))
    print(f"| {d} | {mk} | {100*float(w['missing_rate']):.1f} | {w['pair_viol']} / {g['pair_viol']} / {dt['pair_viol']} / {dw['pair_viol']} / {sl['pair_viol']} "
          f"| {w['row_viol']} of {w['rows']} | {w['topk_lost']} | {float(w['max_ratio']):.3f} "
          f"| {float(w['tightness_median']):.2f} / {float(dt['tightness_median']):.2f} / {float(sl['tightness_median']):.2f} | {100*float(w['coverage_exclusion']):.1f}% |")
tot = lambda mode, key: sum(int(r[key]) for r in a if r["mode"] == mode)
print()
print(f"Pairs per run: {a[0]['pairs']}; runs: {len(by)}; total pair checks per mode: {sum(int(r['pairs']) for r in a if r['mode']=='window')}")
print(f"Window-mode violations by case: case 1 = {tot('window','viol_case1')}, case 2 = {tot('window','viol_case2')}, case 3 = {tot('window','viol_case3')}")
print(f"Declared-true violations: {tot('declared_true','pair_viol')}; declared-wide violations: {tot('declared_wide','pair_viol')}; slew violations: {tot('declared_slew','pair_viol') if any(r['mode']=='declared_slew' for r in a) else 'n/a'}; slew+range violations: {tot('declared_both','pair_viol') if any(r['mode']=='declared_both' for r in a) else 'n/a'}")
