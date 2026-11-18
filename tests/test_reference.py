"""
Tests for the MDMS CPU reference (mdms_ref.py) and the audit helpers.

Tolerance policy (fixed; do not change without approval):
  squared distances are compared with |a - b| <= 1e-9 * max(1, |b|).
  The absolute floor exists because d2 ranges over [0, 4m]; values that are 0 in
  exact arithmetic come out as ~1e-15 after floating-point cancellation.
"""
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "audit"))

import mdms_ref as M  # noqa: E402
from t7_counterexample import lb_direct  # noqa: E402

TOL = 1e-9


def close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.all(np.abs(a[fa] - b[fa]) <= TOL * np.maximum(1.0, np.abs(b[fa]))))


def rw(n, seed):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


# --------------------------------------------------------------------------- #
@pytest.mark.fast
def test_no_missing_equals_oracle_profile():
    n, m = 240, 12
    T = rw(n, 0)
    P, I = M.mdms_profile(T, np.ones(n, bool), m)
    os_ = M.oracle_stats(T, m)
    L = n - m + 1
    Po = np.full(L, np.inf)
    for k in range(M.excl_zone(m) + 1, L):
        d2 = M.oracle_diagonal_d2(os_, k)
        ii = np.arange(L - k)
        np.minimum.at(Po, ii, d2)
        np.minimum.at(Po, ii + k, d2)
    assert close(P ** 2, Po)


@pytest.mark.fast
@pytest.mark.parametrize("pattern", ["random", "block", "mixed"])
@pytest.mark.parametrize("vlimits", ["window", "global", "declared"])
def test_profile_matches_bruteforce(pattern, vlimits):
    n, m = 180, 12
    T = rw(n, 1)
    mask, _ = M.generate_missing_mask(n, m, 0.25, pattern, 0.4, seed=2)
    lim = (T.min() - 1.0, T.max() + 1.0) if vlimits == "declared" else None
    P, I = M.mdms_profile(T, mask, m, vlimits=vlimits, limits=lim)
    Pb, Ib = M.brute_force_profile(T, mask, m, vlimits=vlimits, limits=lim)
    assert close(P ** 2, Pb ** 2)
    # indices: either identical, or a tie (the brute-force distance of the chosen index
    # equals the minimum)
    ws = M.window_stats(T, mask, m, vlimits, lim)
    for i in np.flatnonzero(I != Ib):
        assert I[i] >= 0
        assert close([M.pair_lb_d2(ws, i, int(I[i]))], [Pb[i] ** 2]), f"row {i} is not a tie"


@pytest.mark.fast
def test_two_window_reference_matches_direct_derivation():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        m = int(rng.integers(4, 12))
        x = rng.standard_normal(m) * rng.uniform(0.5, 3)
        y = rng.standard_normal(m)
        bx = rng.random(m) > 0.3
        by = rng.random(m) > 0.3
        if not bx.any() or not by.any():
            continue
        for mode in ("window", "declared"):
            lim = (min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1)
            a = M.two_window_lb_d2(x, bx, y, by, mode, lim if mode == "declared" else None)
            b = lb_direct(x, bx, y, by, *(2 * [lim] if mode == "declared" else [None, None]))
            assert close([a], [b]), (mode, x, bx, y, by, a, b)
            checked += 1
    assert checked > 500


@pytest.mark.fast
def test_case2_closed_form_equals_numerical_infimum():
    """Case 2 (j complete): closed form == inf over (mu, sigma>0) of the co-observed cost.
    For q > 0 the infimum is attained; for q <= 0 it is approached as sigma -> inf, so the
    numerical value must stay >= the closed form and approach it as the bound grows."""
    rng = np.random.default_rng(4)
    seen_pos = seen_neg = 0
    for trial in range(60):
        m = 10
        y = rng.standard_normal(m)
        x = (0.8 * y if trial % 2 == 0 else -0.8 * y) + 0.5 * rng.standard_normal(m)
        bx = np.ones(m, bool)
        bx[rng.choice(m, 3, replace=False)] = False
        yh = (y - y.mean()) / y.std()
        xr, yr = x[bx], yh[bx]
        closed = lb_direct(x, bx, y, np.ones(m, bool))

        def cost(a):  # optimal offset for slope a = 1/sigma is closed form
            res = yr - a * xr
            return float(np.sum((res - res.mean()) ** 2))

        xc, yc = xr - xr.mean(), yr - yr.mean()
        q = float(np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
        if q > 0:
            seen_pos += 1
            r = minimize_scalar(cost, bounds=(1e-12, 1e3), method="bounded",
                                options=dict(xatol=1e-12))
            assert abs(r.fun - closed) <= 1e-6 * max(1, closed)
        else:
            seen_neg += 1
            # infimum is the limit a -> 0+ (sigma -> inf); cost is continuous in a
            assert abs(cost(0.0) - closed) <= 1e-9 * max(1, closed)
            for a in np.geomspace(1e-9, 1e3, 200):
                assert cost(a) >= closed - 1e-9 * max(1, closed)
    assert seen_pos > 5 and seen_neg > 5


@pytest.mark.fast
def test_case3_declared_bound_below_every_completion():
    """Admissibility of the Case 3 bound itself: with valid declared limits, the bound is
    <= the distance of every completion inside the limits (checked by optimizing the
    completion adversarially)."""
    rng = np.random.default_rng(5)
    lo, hi = -3.0, 3.0
    worst = 0.0
    for _ in range(25):
        m = 7
        x = np.clip(rng.standard_normal(m), lo, hi)
        y = np.clip(x + 0.3 * rng.standard_normal(m), lo, hi)
        bx = np.ones(m, bool)
        by = np.ones(m, bool)
        bx[rng.choice(m, 2, replace=False)] = False
        by[rng.choice(m, 2, replace=False)] = False
        lb = lb_direct(x, bx, y, by, (lo, hi), (lo, hi))
        mx, my = np.flatnonzero(~bx), np.flatnonzero(~by)

        def f(v):
            xx, yy = x.copy(), y.copy()
            xx[mx] = v[:len(mx)]
            yy[my] = v[len(mx):]
            d = M.two_window_true_d2(xx, yy)
            return d if np.isfinite(d) else 1e6

        best = np.inf
        for _r in range(12):
            v0 = rng.uniform(lo, hi, len(mx) + len(my))
            r = minimize(f, v0, method="L-BFGS-B", bounds=[(lo, hi)] * len(v0))
            best = min(best, r.fun)
        assert lb <= best * (1 + 1e-9) + 1e-9, (lb, best)
        worst = max(worst, lb / best if best > 0 else 0)
    assert worst <= 1.0 + 1e-9


@pytest.mark.fast
def test_window_mode_counterexample():
    """Hidden spike at the same position in both windows (Appendix H2, T7)."""
    x = np.array([0.0, 1.0, 0.0, 6.0, 0.0, -1.0, 0.0, 0.5])
    y = x + np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.1, -0.1, 0.0])
    b = np.ones(8, bool)
    b[3] = False
    true = M.two_window_true_d2(x, y)
    lw = M.two_window_lb_d2(x, b, y, b, "window")
    ld = M.two_window_lb_d2(x, b, y, b, "declared", (-10.0, 10.0))
    assert lw > 2.5 ** 2 * true          # the bound exceeds the true distance by > 2.5x
    assert ld <= true                    # valid limits restore admissibility


@pytest.mark.fast
def test_declared_limits_zero_violations_on_series():
    """End-to-end: the whole audit pipeline finds no violation when limits are valid."""
    from c3_audit import audit_one
    n, m = 400, 20
    rng = np.random.default_rng(6)
    t = np.arange(n)
    Tc = np.sin(2 * np.pi * t / 25) + 0.2 * rng.standard_normal(n)
    Tc[::50] += 4.0                                  # spikes
    mask, _ = M.clipping_mask(Tc, 0.97)
    modes = {"window": ("window", None),
             "declared": ("declared", (Tc.min(), Tc.max()))}
    res = audit_one(Tc, mask, m, modes)
    assert res["declared"]["pair_viol"] == 0
    assert res["declared"]["pairs"] > 10000


# --------------------------------------------------------------------------- #
# Edge cases and input validation (Appendix G)
# --------------------------------------------------------------------------- #
@pytest.mark.fast
def test_G1_invalid_window_beats_R0():
    n, m = 60, 6
    T = rw(n, 7)
    mask = np.ones(n, bool)
    mask[10:16] = False                         # window 10 has no observed value
    ws = M.window_stats(T, mask, m)
    assert not ws["valid"][10]
    assert M.pair_lb_d2(ws, 10, 40) == np.inf   # R = 0 here, but invalid wins


@pytest.mark.fast
def test_G1_R0_between_valid_windows_is_zero():
    n, m = 60, 6
    T = rw(n, 8)
    mask = np.ones(n, bool)
    mask[10:13] = False
    mask[43:46] = False                         # windows 10 and 40 never co-observed
    ws = M.window_stats(T, mask, m)
    assert ws["valid"][10] and ws["valid"][40]
    R = M._sums_brute(ws, 10, 40)[0]
    assert R == 0
    assert M.pair_lb_d2(ws, 10, 40) == 0.0


@pytest.mark.fast
def test_G2_flat_check_only_for_complete_windows():
    n, m = 60, 6
    T = rw(n, 9)
    T[20:26] = 5.0                              # complete flat window 20
    T[40:46] = 5.0
    mask = np.ones(n, bool)
    mask[43] = False                            # window 40: flat observed values, incomplete
    ws = M.window_stats(T, mask, m)
    assert not ws["valid"][20]
    assert ws["valid"][40]


@pytest.mark.fast
def test_G4_declared_requires_valid_limits():
    T = rw(50, 10)
    mask = np.ones(50, bool)
    with pytest.raises(ValueError):
        M.window_stats(T, mask, 6, "declared", None)
    with pytest.raises(ValueError):
        M.window_stats(T, mask, 6, "declared", (T.min() + 0.1, T.max()))   # too narrow
    with pytest.raises(ValueError):
        M.window_stats(T, mask, 6, "nonsense")


@pytest.mark.fast
def test_G6_inputs_and_bad_values():
    T = rw(50, 11)
    mask = np.ones(50, bool)
    T2 = T.copy()
    T2[5] = np.nan
    with pytest.raises(ValueError):
        M.window_stats(T2, mask, 6)               # NaN must be marked as missing
    mask[5] = False
    ws = M.window_stats(T2, mask, 6)              # now fine
    d2 = M.diagonal_lb_d2(ws, 7)
    assert not np.isnan(d2).any() and (d2 >= 0).all()


@pytest.mark.fast
@pytest.mark.parametrize("pattern", ["random", "block", "mixed"])
def test_G7_generator_exact_rate(pattern):
    n, m = 2000, 100
    for rate in (0.05, 0.2, 0.4):
        for bf in (0.05, 0.5):
            mask, achieved = M.generate_missing_mask(n, m, rate, pattern, bf, seed=1)
            assert int((~mask).sum()) == round(rate * n)
            assert achieved == pytest.approx(round(rate * n) / n)


@pytest.mark.fast
def test_G7_generator_raises_when_unreachable():
    with pytest.raises(ValueError):
        M.generate_missing_mask(100, 50, 0.95, "block", 1.0, seed=0, max_attempts=2000)
    with pytest.raises(ValueError):
        M.generate_missing_mask(100, 10, 1.5, "random")


# --------------------------------------------------------------------------- #
# Slew-rate limits (declared maximum change per sample)
# --------------------------------------------------------------------------- #
@pytest.mark.fast
def test_slew_intervals_contain_truth_and_validate():
    rng = np.random.default_rng(12)
    Tc = np.cumsum(rng.standard_normal(500))
    delta = float(np.max(np.abs(np.diff(Tc))))
    mask, _ = M.generate_missing_mask(500, 20, 0.3, "mixed", 0.5, seed=3)
    lo, hi = M.slew_intervals(Tc, mask, delta)
    assert np.all(lo <= Tc + 1e-12) and np.all(Tc <= hi + 1e-12)
    with pytest.raises(ValueError):
        M.slew_intervals(Tc, mask, delta / 10)          # contradicted by the observed data
    with pytest.raises(ValueError):
        M.window_stats(Tc, mask, 20, "slew", None)


@pytest.mark.fast
def test_slew_limits_are_admissible_on_series():
    """With a correct slew limit the bound never exceeds the true distance (all pairs),
    including saturation dropout, where per-window limits fail."""
    from c3_audit import audit_one
    n, m = 500, 20
    rng = np.random.default_rng(13)
    t = np.arange(n)
    Tc = np.sin(2 * np.pi * t / 25) + 0.2 * rng.standard_normal(n)
    Tc[::50] += 4.0
    delta = float(np.max(np.abs(np.diff(Tc))))
    for mask in (M.clipping_mask(Tc, 0.97)[0],
                 M.generate_missing_mask(n, m, 0.2, "block", 0.5, seed=1)[0]):
        res = audit_one(Tc, mask, m, {"slew": ("slew", delta)})
        assert res["slew"]["pair_viol"] == 0 and res["slew"]["pairs"] > 10000


@pytest.mark.fast
def test_slew_tighter_than_global_range_on_random_walk():
    rng = np.random.default_rng(14)
    Tc = np.cumsum(rng.standard_normal(800))
    mask, _ = M.generate_missing_mask(800, 40, 0.1, "random", seed=2)
    ws_s = M.window_stats(Tc, mask, 40, "slew", float(np.max(np.abs(np.diff(Tc)))))
    ws_d = M.window_stats(Tc, mask, 40, "declared", (Tc.min(), Tc.max()))
    inc = ~ws_s["complete"] & ws_s["valid"]
    # a tighter variance bound means a larger (tighter) inv_den
    assert np.median(ws_s["inv_den"][inc] / ws_d["inv_den"][inc]) > 1.5


@pytest.mark.fast
def test_slew_plus_range_is_at_least_as_tight_and_admissible():
    from c3_audit import audit_one
    n, m = 500, 20
    rng = np.random.default_rng(15)
    Tc = np.sin(2 * np.pi * np.arange(n) / 25) + 0.2 * rng.standard_normal(n)
    Tc[::50] += 4.0
    d = float(np.max(np.abs(np.diff(Tc))))
    mask = M.clipping_mask(Tc, 0.97)[0]
    lo, hi = float(Tc.min()), float(Tc.max())
    ws_b = M.window_stats(Tc, mask, m, "slew", (d, lo, hi), opt_center=True)
    for other in (M.window_stats(Tc, mask, m, "slew", d), M.window_stats(Tc, mask, m, "declared", (lo, hi))):
        inc = ws_b["valid"] & ~ws_b["complete"]
        assert np.all(ws_b["Fden"][inc] <= other["Fden"][inc] + 1e-12)
    res = audit_one(Tc, mask, m, {"both": ("slew", (d, lo, hi), True)})
    assert res["both"]["pair_viol"] == 0


@pytest.mark.fast
def test_opt_center_bound_is_valid_and_tighter():
    """Optimal-centre variance bound: never above Eq. 7, and still >= the true variance of
    every completion inside the limits (random and extreme completions)."""
    rng = np.random.default_rng(16)
    lo, hi = -3.0, 3.0
    checked = 0
    for _ in range(300):
        m = int(rng.integers(4, 12))
        x = np.clip(rng.standard_normal(m) + rng.uniform(-2, 2), lo, hi)
        b = rng.random(m) > 0.4
        if b.all() or not b.any():
            continue
        T = np.concatenate([x, x[::-1]])
        mask = np.concatenate([b, np.ones(m, bool)])
        w0 = M.window_stats(T, mask, m, "declared", (lo, hi))
        w1 = M.window_stats(T, mask, m, "declared", (lo, hi), opt_center=True)
        sd = w0["sd"]
        F0, F1 = w0["Fden"][0] * sd ** 2, w1["Fden"][0] * sd ** 2   # original units
        assert F1 <= F0 + 1e-12
        miss = np.flatnonzero(~b)
        for trial in range(30):
            y = x.copy()
            y[miss] = rng.choice([lo, hi], len(miss)) if trial % 2 else rng.uniform(lo, hi, len(miss))
            assert y.var() <= F1 * (1 + 1e-9) + 1e-12
            checked += 1
    assert checked > 3000
