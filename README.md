# Admissible motif discovery under missing data: audit, fix and GPU implementation

This repository accompanies a paper under review. It contains:

- **`mdms_ref.py`**: CPU reference for the MDMS lower-bound matrix profile (Zhu, Mueen and Keogh, IEEE TKDE 2021) with five ways to set the limits on missing values:
  - `window`: per-window observed range, the MDMS paper's practice
  - `global`: the series' observed range
  - `declared`: a declared value range
  - `slew`: a declared slew-rate limit, optionally intersected with a declared range
  - `opt_center=True`: the optimal-centre variance bound
- **`src/`**: the CUDA implementation (`mdms_gpu.cu`) and a CPU twin (`mdms_twin.cpp`) compiled from the same per-diagonal code (`mdms_core.cuh`). Precision modes are `f64` (reference) and `f32c` (Kahan-compensated float32 with exact re-seeding, the production mode), with `f32` as an ablation.
- **`audit/`**: the admissibility audit:
  - `c3_audit.py`: every pair of windows against the exact distance of the complete series
  - `t7_counterexample.py`: counterexamples
  - `t8_numerics.py`: float32 error
  - `verify_violations.py`, `verify_wide.py`: independent re-checks of reported violations
  - `wide_audit.py`: 40 CWRU recordings (4 speeds, 3 fault types x 3 sizes, normal) with limits
    taken from an earlier part of each recording, with and without a safety margin
- **`c2/`**: comparison with exclusion and imputation (linear, spline, LOCF).
- **`gpu/`**: GPU-vs-reference checks (`compare.py`) and the speed benchmark (`bench.py`).
- **`kaggle/`**: runs GPU jobs on Kaggle (T4 x2) from a local machine and fetches the results.
- **`results/`**: every result file used in the paper.

## Quick start (CPU)

```
pip install numpy scipy pytest
python3 scripts/fetch_cwru.py --out data          # 4 CWRU recordings, checksummed
python3 scripts/fetch_cwru.py --out data_wide --set wide   # 40 recordings for the wider audit
python3 -m pytest -q -m fast tests/               # about 20 s
make bin/mdms_twin                                # CPU twin of the CUDA code
python3 -I gpu/compare.py --bin bin/mdms_twin --suite t5 --out results/t5_twin.jsonl
python3 -I audit/c3_audit.py --data-dir data --out results/v2/c3_audit_n3000_m100.csv
python3 -I audit/wide_audit.py --data-dir data_wide --out results/wide/wide_audit_n3000_m100.csv
python3 -I c2/c2_experiment.py --data-dir data --out results/c2/c2_cwru.csv \
        --only cwru_IR007,cwru_OR007,cwru_B007,cwru_normal
```

## GPU (Kaggle T4)

```
make all                                          # needs nvcc; builds bin/mdms_gpu (sm_75)
export KAGGLE_USERNAME=<your kaggle username>
python3 kaggle/kaggle_job.py list
python3 kaggle/kaggle_job.py run smoke --approved     # build, unit tests, compute-sanitizer, GPU vs CPU twin
python3 kaggle/kaggle_job.py run validate --approved  # T5/T6 and float32 numerics on the GPU
python3 kaggle/kaggle_job.py run bench --approved     # speed, both T4s in parallel
```

`kaggle_job.py` does the following:
- refuses to run on a dirty working tree, or on a commit that is not pushed;
- uploads one generated script, which clones this repository at the exact commit;
- validates the returned `job_summary.json` (job id, commit, GPU, exit codes);
- keeps a weekly GPU-hour ledger in `bench/quota.json`.

## Command line

```
bin/mdms_gpu --input series.f64 --n N --m M [--precision f64|f32c|f32]
             [--vlimits window|global|declared|slew] [--lo LO --hi HI] [--slew DELTA]
             [--opt-center] [--slack-rel 1e-3] [--mode mdms|stomp] [--band-rows K]
             --out PREFIX
```

**Input:** raw little-endian float64, with NaN marking missing samples.

**Outputs:**
- `PREFIX.P.f64`: lower-bound distances
- `PREFIX.I.i32`: neighbour indices
- `PREFIX.stats.json`: run statistics and timing

## Data

The CWRU Bearing Data Center recordings are downloaded from a public `.npz` mirror by `scripts/fetch_cwru.py`, with checksums verified. They are not redistributed here.
