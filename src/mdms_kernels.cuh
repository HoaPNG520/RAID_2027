// mdms_kernels.cuh -- device-only part of the CUDA implementation: result emitters and the
// band kernels. Included by mdms_gpu.cu (the C++ host program) and compiled on its own by the
// Python launcher gpu/mdms_cupy.py (CuPy RawModule, nvcc backend).
#pragma once
#include <climits>
#include "mdms_core.cuh"

typedef unsigned long long ull;

__device__ __forceinline__ ull dbits(double d) { return (ull)__double_as_longlong(d); }
__device__ __forceinline__ ull fpack(float d2, int idx) {
    return ((ull)__float_as_uint(d2) << 32) | (ull)(unsigned)idx;
}

struct EmitF64Pass1 {
    ull* P;
    __device__ void operator()(int i, int j, double d2) {
        const ull v = dbits(d2);
        atomicMin(&P[i], v);
        atomicMin(&P[j], v);
    }
};
struct EmitF64Pass2 {
    const ull* P;
    int* I;
    __device__ void operator()(int i, int j, double d2) {
        const ull v = dbits(d2);
        if (v == P[i] && d2 < INFINITY) atomicMin(&I[i], j);
        if (v == P[j] && d2 < INFINITY) atomicMin(&I[j], i);
    }
};
struct EmitF32 {
    ull* P;
    __device__ void operator()(int i, int j, float d2) {
        atomicMin(&P[i], fpack(d2, j));
        atomicMin(&P[j], fpack(d2, i));
    }
};

template <class A, bool KAHAN, class Emit>
__global__ void mdms_band(const A* z, const uint8_t* b, const Win* W, Params p, int i0, int s1, Emit emit) {
    const int k = p.kmin + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= p.L - i0) return;
    mdms_segment<A, KAHAN>(z, b, W, p, k, i0, s1, emit);
}

template <class A, class Emit>
__global__ void stomp_band(const A* z, const StompWin* W, Params p, int i0, int s1, Emit emit) {
    const int k = p.kmin + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= p.L - i0) return;
    stomp_segment<A>(z, W, p, k, i0, s1, emit);
}

__global__ void fill_ull(ull* a, int n, ull v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}
__global__ void fill_int(int* a, int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}
