// mdms_gpu.cu -- CUDA implementation of the MDMS lower-bound matrix profile (self-join).
//
// One thread per diagonal per band of rows; the per-diagonal arithmetic is mdms_core.cuh,
// shared with the CPU twin (mdms_twin.cpp). Build: see Makefile (T4: sm_75, -fmad=false).
//
// Precision modes (Appendix H2/I3):
//   f64  : double everywhere; two passes (pass 1: atomicMin on the bits of the double d2,
//          pass 2: recompute and keep the smallest index whose d2 equals the minimum).
//   f32c : float sums with Kahan-compensated updates, exact double re-seed every band,
//          global centering; result packed as (float bits << 32 | index).
//   f32  : plain float updates (ablation); --no-center allowed.
#include <cuda_runtime.h>
#include <climits>

#include "host_common.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

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

static const int TPB = 128;

template <class A, bool KAHAN>
static double run_gpu(const Options& o, const Prepared& P, std::vector<double>& P2, std::vector<int32_t>& I) {
    const int n = P.n, L = P.L;
    Params p{P.n, P.m, P.L, P.kmin, o.slack_rel, o.slack_abs};
    std::vector<A> zh(n);
    for (int t = 0; t < n; ++t) zh[t] = A(P.z[t]);
    A* dz; uint8_t* db = nullptr; Win* dW = nullptr; StompWin* dSW = nullptr;
    CK(cudaMalloc(&dz, n * sizeof(A)));
    CK(cudaMemcpy(dz, zh.data(), n * sizeof(A), cudaMemcpyHostToDevice));
    const bool stomp = o.mode == "stomp";
    if (stomp) {
        CK(cudaMalloc(&dSW, L * sizeof(StompWin)));
        CK(cudaMemcpy(dSW, P.SW.data(), L * sizeof(StompWin), cudaMemcpyHostToDevice));
    } else {
        CK(cudaMalloc(&db, n));
        CK(cudaMemcpy(db, P.b.data(), n, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dW, L * sizeof(Win)));
        CK(cudaMemcpy(dW, P.W.data(), L * sizeof(Win), cudaMemcpyHostToDevice));
    }
    ull* dP; int* dI = nullptr;
    CK(cudaMalloc(&dP, L * sizeof(ull)));
    const bool f64 = sizeof(A) == 8;
    const int fb = (L + 255) / 256;
    if (f64) {
        CK(cudaMalloc(&dI, L * sizeof(int)));
        fill_ull<<<fb, 256>>>(dP, L, 0x7FF0000000000000ull);
        fill_int<<<fb, 256>>>(dI, L, INT_MAX);
    } else {
        fill_ull<<<fb, 256>>>(dP, L, (0x7F800000ull << 32) | 0xFFFFFFFFull);
    }
    CK(cudaGetLastError());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    const int passes = f64 ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        for (int i0 = 0; i0 < L; i0 += o.band_rows) {
            const int s1 = std::min(L, i0 + o.band_rows);
            const int ndiag = L - i0 - P.kmin;
            if (ndiag <= 0) break;
            const int grid = (ndiag + TPB - 1) / TPB;
            if constexpr (sizeof(A) == 8) {
                if (pass == 0) {
                    EmitF64Pass1 em{dP};
                    if (stomp) stomp_band<A><<<grid, TPB>>>(dz, dSW, p, i0, s1, em);
                    else mdms_band<A, KAHAN><<<grid, TPB>>>(dz, db, dW, p, i0, s1, em);
                } else {
                    EmitF64Pass2 em{dP, dI};
                    if (stomp) stomp_band<A><<<grid, TPB>>>(dz, dSW, p, i0, s1, em);
                    else mdms_band<A, KAHAN><<<grid, TPB>>>(dz, db, dW, p, i0, s1, em);
                }
            } else {
                EmitF32 em{dP};
                if (stomp) stomp_band<A><<<grid, TPB>>>(dz, dSW, p, i0, s1, em);
                else mdms_band<A, KAHAN><<<grid, TPB>>>(dz, db, dW, p, i0, s1, em);
            }
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    std::vector<ull> hP(L);
    CK(cudaMemcpy(hP.data(), dP, L * sizeof(ull), cudaMemcpyDeviceToHost));
    P2.resize(L); I.resize(L);
    if (f64) {
        std::vector<int> hI(L);
        CK(cudaMemcpy(hI.data(), dI, L * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < L; ++i) {
            double d; memcpy(&d, &hP[i], 8);
            P2[i] = d;
            I[i] = (std::isfinite(d) && hI[i] != INT_MAX) ? hI[i] : -1;
        }
    } else {
        for (int i = 0; i < L; ++i) {
            uint32_t bits = (uint32_t)(hP[i] >> 32);
            float d; memcpy(&d, &bits, 4);
            P2[i] = double(d);
            uint32_t idx = (uint32_t)(hP[i] & 0xFFFFFFFFull);
            I[i] = (std::isfinite(d) && idx != 0xFFFFFFFFu) ? (int32_t)idx : -1;
        }
    }
    cudaFree(dz); cudaFree(db); cudaFree(dW); cudaFree(dSW); cudaFree(dP); cudaFree(dI);
    return ms;
}

int main(int argc, char** argv) {
    const double t0 = now_ms();
    Options o = parse_args(argc, argv);
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) die("no CUDA device");
    if (o.device < 0 || o.device >= ndev) die("bad --device");
    CK(cudaSetDevice(o.device));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, o.device));
    std::vector<double> T = read_input(o);
    Prepared P = prepare(o, T);
    std::vector<double> P2;
    std::vector<int32_t> I;
    double ms;
    if (o.precision == "f64") ms = run_gpu<double, false>(o, P, P2, I);
    else if (o.precision == "f32c") ms = run_gpu<float, true>(o, P, P2, I);
    else ms = run_gpu<float, false>(o, P, P2, I);
    write_outputs(o, P, P2, I, ms, now_ms() - t0, "cuda", prop.name, prop.major, prop.minor);
    return 0;
}
