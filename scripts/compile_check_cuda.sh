#!/bin/sh
# Compile the CUDA code for sm_75 without a GPU: the host program (bin/mdms_gpu) and, separately,
# the kernel source as gpu/mdms_cupy.py hands it to CuPy's nvcc backend, checking that the four
# entry points the launcher looks up are in the cubin. Catches CUDA compile breakage locally and
# in CI, so it does not have to be found by spending GPU quota on Kaggle.
# Usage: sh scripts/compile_check_cuda.sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NVCC=${NVCC:-$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)}
[ -x "$NVCC" ] || { echo "nvcc not found (set NVCC)"; exit 2; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== host program (Makefile, sm_75)"
make -C "$ROOT" bin/mdms_gpu

echo "== kernels as CuPy compiles them"
python3 -I -c "
import re, sys
src = open(sys.argv[1] + '/gpu/mdms_cupy.py').read()
open(sys.argv[2], 'w').write(re.search(r'WRAPPERS = r\"\"\"(.*?)\"\"\"', src, re.S).group(1))
" "$ROOT" "$TMP/wrappers.cu"
# the same options as the RawModule call in gpu/mdms_cupy.py
"$NVCC" -cubin -arch=sm_75 -std=c++17 -fmad=false -I "$ROOT/src" -o "$TMP/wrappers.cubin" "$TMP/wrappers.cu"

for k in mdms_f64_pass1 mdms_f64_pass2 mdms_f32c mdms_f32; do
    if cuobjdump -symbols "$TMP/wrappers.cubin" | grep -qE "STO_ENTRY[[:space:]]+$k\$"; then
        echo "   $k ok"
    else
        echo "   $k MISSING from the cubin"; exit 1
    fi
done
echo "cuda compile check: all clean"
