# Result files and where the paper uses them

All numbers in the paper are generated from these files by `paper/make_numbers.py`.

| File | Paper |
|---|---|
| `v2/c3_audit_n3000_m100.csv` | Sect. 6, admissibility audit (main run, Fig. 2) |
| `c3_audit_n3000_m200.csv`, `c3_audit_n3000_m100_seed1.csv` | Sect. 6, robustness (m = 200, second seed) |
| `t7_counterexample.json` | Sect. 3, counterexample (Fig. 1) and random search |
| `t8_twin.jsonl` | Sect. 5, float32 error of the GPU code (CPU twin) |
| `t5_twin.jsonl` | Sect. 5, twin vs Python reference validation |
| `wide/wide_audit_n3000_m100.csv` | Sect. 7, wider audit and history-based limits (Table 1) |
| `c2/c2_cwru.csv`, `c2/c2_planted.csv`, `c2/c2_summary.json` | Sect. 8, comparison with exclusion and imputation |
| `gpu/` (after the Kaggle runs) | Sect. 5, GPU agreement, sanitizer and speed |

`t8_numerics*` is an earlier NumPy float32 emulation, kept for reference; the paper uses the twin.
`*_tables*.md` are human-readable summaries of the matching CSVs.
