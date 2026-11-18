// mdms_twin.cpp -- CPU twin of the CUDA program: identical per-diagonal code (mdms_core.cuh),
// identical banding and tie rules, run serially. Used to validate the arithmetic against
// mdms_ref.py without a GPU, and as the reference the GPU output must match.
#include "host_common.hpp"

template <class A>
struct HostEmit {
    std::vector<A>* P2;
    std::vector<int32_t>* I;
    void operator()(int i, int j, A d2) { upd(i, j, d2); upd(j, i, d2); }
    void upd(int r, int c, A d2) {
        A& cur = (*P2)[r];
        int32_t& ci = (*I)[r];
        if (d2 < cur || (d2 == cur && d2 < a_inf<A>() && (ci < 0 || c < ci))) { cur = d2; ci = c; }
    }
};

template <class A, bool KAHAN>
static void run(const Options& o, const Prepared& P, std::vector<double>& P2out, std::vector<int32_t>& I) {
    std::vector<A> z(P.n);
    for (int t = 0; t < P.n; ++t) z[t] = A(P.z[t]);
    std::vector<A> P2(P.L, a_inf<A>());
    I.assign(P.L, -1);
    Params p{P.n, P.m, P.L, P.kmin, o.slack_rel, o.slack_abs};
    HostEmit<A> emit{&P2, &I};
    for (int i0 = 0; i0 < P.L; i0 += o.band_rows) {
        const int s1 = std::min(P.L, i0 + o.band_rows);
        for (int k = P.kmin; k < P.L - i0; ++k) {
            if (o.mode == "stomp") stomp_segment<A>(z.data(), P.SW.data(), p, k, i0, s1, emit);
            else mdms_segment<A, KAHAN>(z.data(), P.b.data(), P.W.data(), p, k, i0, s1, emit);
        }
    }
    P2out.resize(P.L);
    for (int i = 0; i < P.L; ++i) P2out[i] = double(P2[i]);
}

int main(int argc, char** argv) {
    const double t0 = now_ms();
    Options o = parse_args(argc, argv);
    std::vector<double> T = read_input(o);
    Prepared P = prepare(o, T);
    std::vector<double> P2;
    std::vector<int32_t> I;
    const double tk = now_ms();
    if (o.precision == "f64") run<double, false>(o, P, P2, I);
    else if (o.precision == "f32c") run<float, true>(o, P, P2, I);
    else run<float, false>(o, P, P2, I);
    const double kernel_ms = now_ms() - tk;
    write_outputs(o, P, P2, I, kernel_ms, now_ms() - t0, "cpu_twin", "cpu", 0, 0);
    return 0;
}
