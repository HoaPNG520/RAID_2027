#!/bin/sh
# Run the CUDA binary under compute-sanitizer (memcheck, racecheck, initcheck) on small inputs
# with gaps, in every precision and mode. Exits non-zero on any reported error.
# Usage: sh scripts/sanitize.sh bin/mdms_gpu out/sanitizer.txt
set -u
BIN=${1:-bin/mdms_gpu}
LOG=${2:-sanitizer.txt}
SAN=$(command -v compute-sanitizer 2>/dev/null || echo /usr/local/cuda/bin/compute-sanitizer)
TMP=$(mktemp -d)
: > "$LOG"
if [ ! -x "$SAN" ]; then
    echo "compute-sanitizer not found" | tee -a "$LOG"
    exit 2
fi
python3 -I -c "
import numpy as np, sys
rng = np.random.default_rng(7)
T = np.cumsum(rng.standard_normal(700))
T[rng.random(700) < 0.05] = np.nan
T[300:340] = np.nan
T.astype('<f8').tofile(sys.argv[1])
np.cumsum(rng.standard_normal(700)).astype('<f8').tofile(sys.argv[2])
" "$TMP/x.f64" "$TMP/full.f64"
fail=0
for tool in memcheck racecheck initcheck; do
  for prec in f64 f32c f32; do
    for args in "--vlimits window" "--vlimits slew --slew 4 --lo -100 --hi 10 --opt-center" "--mode stomp"; do
      # band boundaries inside the series: --band-rows 64
      inp=$TMP/x.f64
      case "$args" in *stomp*) inp=$TMP/full.f64 ;; esac
      cmd="$SAN --tool $tool --error-exitcode 9 $BIN --input $inp --n 700 --m 32 --precision $prec $args --band-rows 64 --out $TMP/o"
      echo "== $cmd" >> "$LOG"
      if ! $cmd >> "$LOG" 2>&1; then
        echo "FAILED: $tool $prec $args" | tee -a "$LOG"
        fail=1
      fi
    done
  done
done
rm -rf "$TMP"
[ $fail -eq 0 ] && echo "sanitizer: all clean" | tee -a "$LOG"
exit $fail
