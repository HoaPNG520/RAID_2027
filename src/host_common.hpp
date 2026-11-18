// host_common.hpp -- CLI, input/output and per-window preprocessing shared by the CUDA
// program (mdms_gpu.cu) and the CPU twin (mdms_twin.cpp). Mirrors mdms_ref.window_stats.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "mdms_core.cuh"

struct Options {
    std::string input, out, precision = "f64", vlimits = "window", mode = "mdms";
    long n = -1;
    int m = -1;
    double excl = 0.25, lo = NAN, hi = NAN, slew = NAN, slack_rel = 0.0, slack_abs = 0.0;
    int band_rows = 0;   // 0 -> default clamp(4m, 2048, 16384)
    int device = 0;
    bool center = true;  // global standardization (f32 ablation: --no-center)
    bool opt_center = false;  // optimal-centre variance bound (mdms_ref opt_center=True)
};

[[noreturn]] inline void die(const std::string& msg) {
    fprintf(stderr, "ERROR: %s\n", msg.c_str());
    exit(2);
}

inline Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) die(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input") o.input = need(i);
        else if (a == "--out") o.out = need(i);
        else if (a == "--n") o.n = atol(need(i));
        else if (a == "--m") o.m = atoi(need(i));
        else if (a == "--precision") o.precision = need(i);
        else if (a == "--vlimits") o.vlimits = need(i);
        else if (a == "--lo") o.lo = atof(need(i));
        else if (a == "--hi") o.hi = atof(need(i));
        else if (a == "--slew") o.slew = atof(need(i));
        else if (a == "--excl") o.excl = atof(need(i));
        else if (a == "--band-rows") o.band_rows = atoi(need(i));
        else if (a == "--slack-rel") o.slack_rel = atof(need(i));
        else if (a == "--slack-abs") o.slack_abs = atof(need(i));
        else if (a == "--mode") o.mode = need(i);
        else if (a == "--device") o.device = atoi(need(i));
        else if (a == "--no-center") o.center = false;
        else if (a == "--opt-center") o.opt_center = true;
        else die("unknown argument " + a);
    }
    if (o.input.empty() || o.out.empty() || o.n < 0 || o.m < 2) die("need --input --out --n --m (m >= 2)");
    if (o.precision != "f64" && o.precision != "f32c" && o.precision != "f32") die("--precision must be f64, f32c or f32");
    if (o.vlimits != "window" && o.vlimits != "global" && o.vlimits != "declared" && o.vlimits != "slew") die("--vlimits must be window, global, declared or slew");
    if (o.vlimits == "slew" && !(std::isfinite(o.slew) && o.slew > 0)) die("--vlimits slew needs --slew DELTA > 0");
    if (o.vlimits == "declared" && !(std::isfinite(o.lo) && std::isfinite(o.hi) && o.lo <= o.hi)) die("--vlimits declared needs --lo LO --hi HI with LO <= HI");
    if (o.vlimits == "slew" && (std::isfinite(o.lo) != std::isfinite(o.hi) || (std::isfinite(o.lo) && o.lo > o.hi))) die("--vlimits slew: give both --lo and --hi or neither");
    if (o.mode != "mdms" && o.mode != "stomp") die("--mode must be mdms or stomp");
    if (!o.center && o.precision != "f32") die("--no-center is only allowed with --precision f32 (ablation)");
    if (o.slack_rel < 0 || o.slack_rel >= 1 || o.slack_abs < 0) die("invalid slack");
    if (o.n < o.m + 1) die("n must be > m");
    if (o.band_rows <= 0) o.band_rows = std::min(16384, std::max(2048, 4 * o.m));
    return o;
}

inline std::vector<double> read_input(const Options& o) {
    FILE* f = fopen(o.input.c_str(), "rb");
    if (!f) die("cannot open " + o.input);
    std::vector<double> T(o.n);
    size_t got = fread(T.data(), sizeof(double), o.n, f);
    char extra;
    bool more = fread(&extra, 1, 1, f) == 1;
    fclose(f);
    if ((long)got != o.n || more) die("input size does not match --n (raw float64 expected)");
    return T;
}

// Preprocessed series and windows (standardized as in mdms_ref).
struct Prepared {
    int n = 0, m = 0, L = 0, kmin = 0;
    double mu = 0, sd = 1, scale = 1;   // scale = sd if centered else 1
    std::vector<double> z;              // standardized (or raw) values, 0 where missing
    std::vector<uint8_t> b;             // 1 = observed
    std::vector<Win> W;
    std::vector<StompWin> SW;
    long n_missing = 0, n_valid_windows = 0, n_complete_windows = 0;
};

inline Prepared prepare(const Options& o, const std::vector<double>& T) {
    Prepared P;
    P.n = (int)o.n; P.m = o.m; P.L = P.n - P.m + 1;
    P.kmin = (int)std::ceil(o.m * o.excl) + 1;
    const int n = P.n, m = P.m, L = P.L;
    P.b.resize(n); P.z.assign(n, 0.0);
    long double s = 0, s2 = 0; long cnt = 0;
    for (int t = 0; t < n; ++t) {
        P.b[t] = std::isfinite(T[t]) ? 1 : 0;
        if (P.b[t]) { s += T[t]; ++cnt; }
    }
    P.n_missing = n - cnt;
    if (cnt == 0) die("series has no observed values");
    const double mu = (double)(s / cnt);
    for (int t = 0; t < n; ++t) if (P.b[t]) { long double d = T[t] - mu; s2 += d * d; }
    const double sd = std::sqrt((double)(s2 / cnt));
    if (!(sd > 0)) die("observed values are constant");
    P.mu = mu; P.sd = sd;
    const bool center = o.center;
    P.scale = center ? sd : 1.0;
    for (int t = 0; t < n; ++t) if (P.b[t]) P.z[t] = center ? (T[t] - mu) / sd : T[t];

    if (o.mode == "stomp") {
        if (P.n_missing) die("--mode stomp needs a complete series (no NaN)");
        P.SW.resize(L);
        for (int i = 0; i < L; ++i) {
            long double a = 0, a2 = 0;
            for (int t = 0; t < m; ++t) a += P.z[i + t];
            const long double mm = a / m;
            for (int t = 0; t < m; ++t) { long double d = P.z[i + t] - mm; a2 += d * d; }
            P.SW[i].mu = (double)mm;
            P.SW[i].sd = std::sqrt((double)(a2 / m));
            P.SW[i].valid = P.SW[i].sd * P.SW[i].sd > 1e-10 * (center ? 1.0 : sd * sd);
        }
        return P;
    }

    // Statistics in standardized units (exactly as mdms_ref), then rescaled for raw mode.
    std::vector<double> zs(n, 0.0);
    for (int t = 0; t < n; ++t) if (P.b[t]) zs[t] = (T[t] - mu) / sd;
    double gmin = INFINITY, gmax = -INFINITY;
    for (int t = 0; t < n; ++t) if (P.b[t]) { gmin = std::min(gmin, zs[t]); gmax = std::max(gmax, zs[t]); }
    double dlo = 0, dhi = 0;
    if (o.vlimits == "declared") {
        double omin = INFINITY, omax = -INFINITY;
        for (int t = 0; t < n; ++t) if (P.b[t]) { omin = std::min(omin, T[t]); omax = std::max(omax, T[t]); }
        if (omin < o.lo || omax > o.hi) die("declared limits do not contain the observed range");
        dlo = (o.lo - mu) / sd; dhi = (o.hi - mu) / sd;
    }
    // sliding min/max over observed values (monotone deques)
    std::vector<double> wmin(L), wmax(L);
    {
        std::deque<int> qmin, qmax;
        for (int t = 0; t < n; ++t) {
            if (P.b[t]) {
                while (!qmin.empty() && zs[qmin.back()] >= zs[t]) qmin.pop_back();
                qmin.push_back(t);
                while (!qmax.empty() && zs[qmax.back()] <= zs[t]) qmax.pop_back();
                qmax.push_back(t);
            }
            const int i = t - m + 1;
            if (i >= 0) {
                while (!qmin.empty() && qmin.front() < i) qmin.pop_front();
                while (!qmax.empty() && qmax.front() < i) qmax.pop_front();
                wmin[i] = qmin.empty() ? INFINITY : zs[qmin.front()];
                wmax[i] = qmax.empty() ? -INFINITY : zs[qmax.front()];
            }
        }
    }
    // slew-rate limits: per-position bounds from the nearest observed neighbours (mdms_ref.slew_intervals)
    std::vector<double> slo, shi;
    if (o.vlimits == "slew") {
        slo.assign(n, -INFINITY); shi.assign(n, INFINITY);
        const double d = o.slew;
        int prev = -1;
        for (int t = 0; t < n; ++t) {
            if (P.b[t]) {
                if (prev >= 0 && std::fabs(T[t] - T[prev]) > d * (t - prev) * (1 + 1e-12) + 1e-12)
                    die("observed data violate the declared slew-rate limit");
                prev = t; slo[t] = shi[t] = T[t];
            } else if (prev >= 0) {
                slo[t] = T[prev] - d * (t - prev); shi[t] = T[prev] + d * (t - prev);
            }
        }
        int next = -1;
        for (int t = n - 1; t >= 0; --t) {
            if (P.b[t]) next = t;
            else if (next >= 0) {
                slo[t] = std::max(slo[t], T[next] - d * (next - t));
                shi[t] = std::min(shi[t], T[next] + d * (next - t));
            }
        }
        if (std::isfinite(o.lo)) {
            double omin = INFINITY, omax = -INFINITY;
            for (int t = 0; t < n; ++t) if (P.b[t]) { omin = std::min(omin, T[t]); omax = std::max(omax, T[t]); }
            if (omin < o.lo || omax > o.hi) die("declared range does not contain the observed range");
            for (int t = 0; t < n; ++t) { slo[t] = std::max(slo[t], o.lo); shi[t] = std::min(shi[t], o.hi); }
        }
        for (int t = 0; t < n; ++t) { slo[t] = (slo[t] - mu) / sd; shi[t] = (shi[t] - mu) / sd; }
        // sliding min/max over MISSING positions only
        std::deque<int> qmin, qmax;
        for (int t = 0; t < n; ++t) {
            if (!P.b[t]) {
                while (!qmin.empty() && slo[qmin.back()] >= slo[t]) qmin.pop_back();
                qmin.push_back(t);
                while (!qmax.empty() && shi[qmax.back()] <= shi[t]) qmax.pop_back();
                qmax.push_back(t);
            }
            const int i = t - m + 1;
            if (i >= 0) {
                while (!qmin.empty() && qmin.front() < i) qmin.pop_front();
                while (!qmax.empty() && qmax.front() < i) qmax.pop_front();
                wmin[i] = qmin.empty() ? 0.0 : slo[qmin.front()];
                wmax[i] = qmax.empty() ? 0.0 : shi[qmax.front()];
            }
        }
    }
    P.W.resize(L);
    for (int i = 0; i < L; ++i) {
        int nobs = 0; long double sZ = 0, sX = 0;
        for (int t = 0; t < m; ++t) if (P.b[i + t]) { ++nobs; sZ += zs[i + t]; sX += (long double)zs[i + t] * zs[i + t]; }
        const bool complete = nobs == m;
        const double var_full = (double)(sX / m - (sZ / m) * (sZ / m));
        double vmin, vmax;
        if (o.vlimits == "window" || o.vlimits == "slew") { vmin = wmin[i]; vmax = wmax[i]; }
        else if (o.vlimits == "global") { vmin = gmin; vmax = gmax; }
        else { vmin = dlo; vmax = dhi; }
        double Fden = 0;
        if (nobs > 0) {
            const double mub = double(nobs) / m;
            const double ur = (double)(sZ / nobs);
            double vr = (double)(sX / nobs) - ur * ur;
            if (vr < 0) vr = 0;
            const double A = vmax + vmin, Vp = vmax * vmin, C = vmax - vmin;
            Fden = mub * (vr + Vp + ur * (ur - A)) + C * C / 4.0;
            if (o.opt_center) {
                const double nm = m - nobs, mid = 0.5 * (vmax + vmin), S = (double)sZ, Q = (double)sX;
                const double c1 = (S + nm * vmin) / m, c2 = (S + nm * vmax) / m;
                const double c = c1 >= mid ? c1 : (c2 <= mid ? c2 : mid);
                const double e = 0.5 * C + std::fabs(c - mid);
                const double Fopt = (Q - 2 * c * S + nobs * c * c + nm * e * e) / m;
                if (std::isfinite(Fopt) && Fopt < Fden) Fden = Fopt;
            }
        }
        Win w;
        w.nobs = nobs;
        w.valid = (nobs > 0) && !(complete && var_full <= 1e-10);
        w.inv_den = 0.0;
        if (w.valid && complete) w.inv_den = 1.0 / var_full;
        else if (w.valid && Fden > 1e-10) w.inv_den = 1.0 / Fden;
        if (!center) w.inv_den /= sd * sd;   // raw units for the f32 ablation
        P.W[i] = w;
        P.n_valid_windows += w.valid;
        P.n_complete_windows += complete;
    }
    return P;
}

inline void write_outputs(const Options& o, const Prepared& P, const std::vector<double>& P2,
                          const std::vector<int32_t>& I, double kernel_ms, double total_ms,
                          const std::string& backend, const std::string& device_name,
                          int cc_major, int cc_minor) {
    std::vector<double> Pd(P.L);
    for (int i = 0; i < P.L; ++i) Pd[i] = std::isfinite(P2[i]) ? std::sqrt(P2[i]) : INFINITY;
    FILE* f = fopen((o.out + ".P.f64").c_str(), "wb");
    if (!f) die("cannot write output");
    fwrite(Pd.data(), sizeof(double), Pd.size(), f); fclose(f);
    f = fopen((o.out + ".I.i32").c_str(), "wb");
    fwrite(I.data(), sizeof(int32_t), I.size(), f); fclose(f);
    const double nd = std::max(0, P.L - P.kmin);
    const double pairs = nd * (nd + 1) / 2.0;
    f = fopen((o.out + ".stats.json").c_str(), "w");
    fprintf(f, "{\n  \"backend\": \"%s\",\n  \"device_name\": \"%s\",\n  \"compute_capability\": \"%d.%d\",\n",
            backend.c_str(), device_name.c_str(), cc_major, cc_minor);
    fprintf(f, "  \"n\": %d,\n  \"m\": %d,\n  \"L\": %d,\n  \"kmin\": %d,\n  \"mode\": \"%s\",\n",
            P.n, P.m, P.L, P.kmin, o.mode.c_str());
    fprintf(f, "  \"precision\": \"%s\",\n  \"centered\": %s,\n  \"opt_center\": %s,\n  \"vlimits\": \"%s\",\n",
            o.precision.c_str(), o.center ? "true" : "false", o.opt_center ? "true" : "false", o.vlimits.c_str());
    fprintf(f, "  \"band_rows\": %d,\n  \"slack_rel\": %.17g,\n  \"slack_abs\": %.17g,\n",
            o.band_rows, o.slack_rel, o.slack_abs);
    fprintf(f, "  \"n_missing\": %ld,\n  \"n_valid_windows\": %ld,\n  \"n_complete_windows\": %ld,\n",
            P.n_missing, P.n_valid_windows, P.n_complete_windows);
    fprintf(f, "  \"pairs\": %.0f,\n  \"kernel_ms\": %.6f,\n  \"total_ms\": %.6f,\n  \"gpairs_per_s\": %.6f,\n",
            pairs, kernel_ms, total_ms, kernel_ms > 0 ? pairs / (kernel_ms * 1e6) : 0.0);
    fprintf(f, "  \"p_file\": \"%s.P.f64\",\n  \"i_file\": \"%s.I.i32\"\n}\n", o.out.c_str(), o.out.c_str());
    fclose(f);
}

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
