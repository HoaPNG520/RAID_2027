// mdms_core.cuh -- per-diagonal MDMS lower-bound computation shared by the CUDA kernel
// and the CPU twin (compiled as plain C++). Spec section 2 + Appendices G, H, I.
//
// The same code runs on the GPU (one thread per diagonal per band) and on the CPU
// (serial loop), so the CPU twin validates the arithmetic against mdms_ref.py and the
// GPU run only has to validate the launch, the atomics and the memory layout.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

struct Win {
    int32_t nobs;     // observed values in the window
    int32_t valid;    // 0 = invalid window (no observed value, or complete and flat)
    double inv_den;   // 1/var_full (complete) or 1/Fden (incomplete), 0 if Fden <= eps
};

struct StompWin {     // per-window stats for the plain STOMP baseline (complete data)
    double mu, sd;
    int32_t valid;
};

struct Params {
    int n, m, L, kmin;
    double slack_rel, slack_abs;
};

template <class A> HD A a_inf();
template <> HD double a_inf<double>() { return INFINITY; }
template <> HD float a_inf<float>() { return INFINITY; }

// Additions that the compiler may not contract or reorder (needed for Kahan on the GPU;
// harmless elsewhere). The host build is compiled without -ffast-math.
template <class A> HD A add_rn(A a, A b);
template <class A> HD A sub_rn(A a, A b);
template <class A> HD A mul_rn(A a, A b);
#ifdef __CUDA_ARCH__
template <> HD double add_rn<double>(double a, double b) { return __dadd_rn(a, b); }
template <> HD float add_rn<float>(float a, float b) { return __fadd_rn(a, b); }
template <> HD double sub_rn<double>(double a, double b) { return __dsub_rn(a, b); }
template <> HD float sub_rn<float>(float a, float b) { return __fsub_rn(a, b); }
template <> HD double mul_rn<double>(double a, double b) { return __dmul_rn(a, b); }
template <> HD float mul_rn<float>(float a, float b) { return __fmul_rn(a, b); }
#else
template <> HD double add_rn<double>(double a, double b) { volatile double r = a + b; return r; }
template <> HD float add_rn<float>(float a, float b) { volatile float r = a + b; return r; }
template <> HD double sub_rn<double>(double a, double b) { volatile double r = a - b; return r; }
template <> HD float sub_rn<float>(float a, float b) { volatile float r = a - b; return r; }
template <> HD double mul_rn<double>(double a, double b) { volatile double r = a * b; return r; }
template <> HD float mul_rn<float>(float a, float b) { volatile float r = a * b; return r; }
#endif

// Squared lower bound from the six sums (section 2; G1 precedence; G6 bad values).
template <class A>
HD A eval_d2(int R, A Si, A Sj, A Qi, A Qj, A Pij, const Win& wi, const Win& wj, int m) {
    if (!wi.valid || !wj.valid) return a_inf<A>();
    if (R == 0) return A(0);
    const A Rf = A(R);
    const A mi = Si / Rf, mj = Sj / Rf;
    A vi = Qi / Rf - mi * mi;
    A vj = Qj / Rf - mj * mj;
    if (vi < A(0)) vi = A(0);
    if (vj < A(0)) vj = A(0);
    const A cov = Pij / Rf - mi * mj;
    const A vv = vi * vj;
    A q = (vv > A(1e-24)) ? cov / std::sqrt(vv) : A(0);
    if (q > A(1)) q = A(1);
    if (q < A(-1)) q = A(-1);
    // exactly two co-observed points: the correlation is exactly +-1 (see mdms_ref)
    if (R == 2 && vv > A(1e-24)) q = (cov > A(0)) ? A(1) : ((cov < A(0)) ? A(-1) : A(0));
    const A corr = (q > A(0)) ? A(1) - q * q : A(1);
    const bool ci = wi.nobs == m, cj = wj.nobs == m;
    A d2;
    if (ci && cj) {
        d2 = A(2) * A(m) * (A(1) - q);
    } else if (ci || cj) {
        const A v_c = ci ? vi : vj;
        const A inv_c = A(ci ? wi.inv_den : wj.inv_den);
        d2 = Rf * v_c * inv_c * corr;
    } else {
        const A a = vi * A(wi.inv_den), c = vj * A(wj.inv_den);
        d2 = Rf * (a > c ? a : c) * corr;
    }
    if (d2 != d2) return a_inf<A>();       // NaN -> INF
    if (d2 < A(0)) d2 = A(0);
    return d2;
}

template <class A>
HD A apply_slack(A d2, const Params& p) {
    if (!(d2 < a_inf<A>())) return d2;
    A s = d2 * A(1.0 - p.slack_rel) - A(p.slack_abs);
    return s > A(0) ? s : A(0);
}

// Compensated (Kahan) accumulator.
template <class A, bool KAHAN>
struct Acc {
    A s, c;
    HD void init(A v) { s = v; c = A(0); }
    HD void add(A v) {
        if (KAHAN) {
            const A y = sub_rn(v, c);
            const A t = add_rn(s, y);
            c = sub_rn(sub_rn(t, s), y);
            s = t;
        } else {
            s = add_rn(s, v);
        }
    }
};

// Process pairs (s, s+k) for s in [s0, s1) on diagonal k. The six sums are seeded exactly
// (double) at s0 and then updated by the diagonal recurrence in precision A.
// emit(i, j, d2) receives every evaluated pair.
template <class A, bool KAHAN, class Emit>
HD void mdms_segment(const A* z, const uint8_t* b, const Win* W, const Params& p,
                     int k, int s0, int s1, Emit& emit) {
    const int m = p.m;
    if (s1 > p.L - k) s1 = p.L - k;
    if (s0 >= s1) return;
    int R = 0;
    double dSi = 0, dSj = 0, dQi = 0, dQj = 0, dPij = 0;
    for (int t = 0; t < m; ++t) {
        const double zi = double(z[s0 + t]), zj = double(z[s0 + k + t]);
        const int bi = b[s0 + t], bj = b[s0 + k + t];
        R += bi & bj;
        dSi += zi * bj;
        dSj += bi * zj;
        dQi += zi * zi * bj;
        dQj += bi * zj * zj;
        dPij += zi * zj;
    }
    Acc<A, KAHAN> Si, Sj, Qi, Qj, Pij;
    Si.init(A(dSi)); Sj.init(A(dSj)); Qi.init(A(dQi)); Qj.init(A(dQj)); Pij.init(A(dPij));
    for (int s = s0; s < s1; ++s) {
        const int i = s, j = s + k;
        const A d2 = eval_d2<A>(R, Si.s, Sj.s, Qi.s, Qj.s, Pij.s, W[i], W[j], m);
        emit(i, j, apply_slack<A>(d2, p));
        if (s + 1 < s1) {
            const int i1 = i + m, j1 = j + m;
            const int b0i = b[i], b0j = b[j], b1i = b[i1], b1j = b[j1];
            R += (b1i & b1j) - (b0i & b0j);
            const A z0i = z[i], z0j = z[j], z1i = z[i1], z1j = z[j1];
            const A B0i = A(b0i), B0j = A(b0j), B1i = A(b1i), B1j = A(b1j);
            Si.add(sub_rn(mul_rn(z1i, B1j), mul_rn(z0i, B0j)));
            Sj.add(sub_rn(mul_rn(B1i, z1j), mul_rn(B0i, z0j)));
            Qi.add(sub_rn(mul_rn(mul_rn(z1i, z1i), B1j), mul_rn(mul_rn(z0i, z0i), B0j)));
            Qj.add(sub_rn(mul_rn(B1i, mul_rn(z1j, z1j)), mul_rn(B0i, mul_rn(z0j, z0j))));
            Pij.add(sub_rn(mul_rn(z1i, z1j), mul_rn(z0i, z0j)));
        }
    }
}

// Plain STOMP baseline on complete data (one accumulator), same banding.
template <class A, class Emit>
HD void stomp_segment(const A* z, const StompWin* W, const Params& p, int k, int s0, int s1,
                      Emit& emit) {
    const int m = p.m;
    if (s1 > p.L - k) s1 = p.L - k;
    if (s0 >= s1) return;
    double dP = 0;
    for (int t = 0; t < m; ++t) dP += double(z[s0 + t]) * double(z[s0 + k + t]);
    A P = A(dP);
    for (int s = s0; s < s1; ++s) {
        const int i = s, j = s + k;
        A d2;
        if (!W[i].valid || !W[j].valid) {
            d2 = a_inf<A>();
        } else {
            A rho = (P / A(m) - A(W[i].mu) * A(W[j].mu)) / (A(W[i].sd) * A(W[j].sd));
            if (rho > A(1)) rho = A(1);
            if (rho < A(-1)) rho = A(-1);
            d2 = A(2) * A(m) * (A(1) - rho);
            if (d2 < A(0)) d2 = A(0);
        }
        emit(i, j, d2);
        if (s + 1 < s1) P = add_rn(P, sub_rn(mul_rn(A(z[i + m]), A(z[j + m])), mul_rn(A(z[i]), A(z[j]))));
    }
}
