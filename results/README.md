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
| `wide/wide_audit_n3000_m50.csv`, `_m200.csv`, `_m100_seed1.csv`, `_m100_seg40000.csv` | Sect. 7, robustness of the wider audit (window length, seed, segment) |
| `pamap2/pamap2_case.csv` | Sect. 7, real-saturation case study (PAMAP2, co-located +-16 g sensor as truth) |
| `c2/c2_cwru.csv`, `c2/c2_planted.csv`, `c2/c2_summary.json` | Sect. 8, comparison with exclusion and imputation |
| `c2/c2_saits_cwru.csv`, `c2/c2_saits_planted.csv` | Sect. 8, SAITS learned-imputer baseline (merged into the same tables) |
| `gpu/` (after the Kaggle runs) | Sect. 5, GPU agreement, sanitizer and speed |

`t8_numerics*` is an earlier NumPy float32 emulation, kept for reference; the paper uses the twin.
`*_tables*.md` are human-readable summaries of the matching CSVs.
`t5_twin.jsonl` is the twin-vs-reference pass/fail record (40 cases, all pass); the paper's T5
macros come from `gpu/t5_gpu.jsonl`, the same suite on the GPU.
Coverage (Sect. 4) is the one figure recomputed on the fly by `paper/make_figures.py` rather than
read from a file; it depends only on the mask generator and its seeds.
Each `*.meta.json` records the exact argv, n, m, seed and inputs of the run that wrote the CSV
beside it; each `gpu/job_summary_*.json` records the commit, GPU, per-step exit codes and timing
of the Kaggle run that produced the files in `gpu/`.
