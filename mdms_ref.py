"""
CPU reference for the MDMS lower-bound matrix profile (spec section 2, Appendices E, G, H).

Ground truth for the GPU kernel and for the C3 admissibility audit.

Conventions
-----------
* ``mask``: boolean array, True = observed, False = missing.
* All distances are z-normalized Euclidean distances. Functions that return ``d2``
  return the *squared* distance; profiles return the distance (sqrt of d2).
* The series is standardized once over its observed values before any statistics
  are computed (z-normalized distances are invariant to this; it removes the
  catastrophic cancellation of E[x^2] - E[x]^2 on raw data with a DC offset).
  Declared value limits are transformed with the same affine map.
* vlimits modes (bounds assumed for the missing values in Case 3):
    - ``"window"``   : min/max of the observed values of each window (MDMS paper practice)
    - ``"global"``   : min/max of all observed values of the series
    - ``"declared"`` : user-supplied (vmin, vmax), e.g. sensor datasheet limits.
                       Called ``file`` in earlier spec sections.
    - ``"slew"``     : user-supplied maximum change per sample, delta >= |x[t] - x[t-1]|
                       (a declared slew-rate limit). Each missing value is bounded by its
                       nearest observed neighbours: x_l - delta*(t-l) <= x_t <= x_l + delta*(t-l)
                       and likewise from the right; a window's limits cover its missing values.
                       limits=(delta, lo, hi) also intersects with a declared value range.
  Invalid input raises ``ValueError``. There are no silent fallbacks.
* Window validity (Appendix G1/G2): a window is invalid if it has no observed value,
  or if it is complete and flat (var_full <= EPS_VAR). Invalid windows give d2 = INF
  for every pair, and this takes precedence over the R = 0 rule (R = 0 -> d2 = 0).
* Bad values (G6): NaN -> INF, negative -> 0, everywhere.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

INF = np.inf
EPS_VAR = 1e-10      # flat-window threshold on standardized data
EPS_Q = 1e-24        # vi*vj below this -> q = 0
VLIMIT_MODES = ("window", "global", "declared", "slew")

# Case ids used by the per-pair case arrays
CASE_INVALID, CASE_R0, CASE1, CASE2, CASE3 = 0, 4, 1, 2, 3


# --------------------------------------------------------------------------- #
# Input checks and preprocessing
# --------------------------------------------------------------------------- #
def _check_inputs(T, mask, m):
    T = np.asarray(T, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if T.ndim != 1 or mask.shape != T.shape:
        raise ValueError("T and mask must be 1-D arrays of the same length")
    if not isinstance(m, (int, np.integer)) or m < 2:
        raise ValueError("m must be an integer >= 2")
    if len(T) < m + 1:
        raise ValueError("series must be longer than m")
    if not np.all(np.isfinite(T[mask])):
        raise ValueError("observed values must be finite; mark missing values in mask")
    if not mask.any():
        raise ValueError("series has no observed values")
    return T, mask


def standardize(T, mask):
    """Return (Ts, mu, sd): T standardized over observed values; missing set to 0."""
    obs = T[mask]
    mu = float(obs.mean())
    sd = float(obs.std())
    if not sd > 0:
        raise ValueError("observed values are constant; cannot standardize")
    Ts = np.where(mask, (T - mu) / sd, 0.0)
    return Ts, mu, sd


def slew_intervals(T, mask, delta):
    """Per-position bounds [lo, hi] implied by a declared slew-rate limit `delta`
    (original units per sample). Observed positions get their own value. Raises ValueError
    if the observed data violate the declared limit."""
    T = np.asarray(T, float)
    n = len(T)
    idx = np.arange(n)
    obs = np.flatnonzero(mask)
    gaps = np.diff(obs)
    if np.any(np.abs(np.diff(T[obs])) > delta * gaps * (1 + 1e-12) + 1e-12):
        raise ValueError("observed data violate the declared slew-rate limit")
    # nearest observed index to the left / right of every position
    left = np.where(mask, idx, -1)
    np.maximum.accumulate(left, out=left)
    right = np.where(mask, idx, n)
    right = np.minimum.accumulate(right[::-1])[::-1]
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    hl = left >= 0
    lo[hl] = T[left[hl]] - delta * (idx[hl] - left[hl])
    hi[hl] = T[left[hl]] + delta * (idx[hl] - left[hl])
    hr = right < n
    lo[hr] = np.maximum(lo[hr], T[right[hr]] - delta * (right[hr] - idx[hr]))
    hi[hr] = np.minimum(hi[hr], T[right[hr]] + delta * (right[hr] - idx[hr]))
    lo[mask] = T[mask]
    hi[mask] = T[mask]
    return lo, hi


def window_stats(T, mask, m, vlimits="window", limits=None, opt_center=False):
    """
    Per-window statistics on the standardized series.

    opt_center=False uses the MDMS paper's variance bound (Eq. 7: deviations from the
    midrange c of [vmin, vmax]). opt_center=True uses the tightest bound of the same
    family: Var <= min_c [sum_obs (t-c)^2 + n_miss * max((vmin-c)^2, (vmax-c)^2)] / m,
    which holds for every c and is minimized in closed form (convex in c). It is never
    larger than Eq. 7 and needs the same assumption (missing values in [vmin, vmax]).

    Returns a dict with arrays of length len = n - m + 1:
      nobs (int64), valid (bool), complete (bool), var_full, Fden, inv_den, vmin, vmax
    plus z, b, x (position arrays) and the standardization (mu, sd).
    """
    T, mask = _check_inputs(T, mask, m)
    if vlimits not in VLIMIT_MODES:
        raise ValueError(f"vlimits must be one of {VLIMIT_MODES}, got {vlimits!r}")
    Ts, mu, sd = standardize(T, mask)
    n = len(T)
    L = n - m + 1

    z = Ts
    b = mask.astype(np.int64)
    x = z * z

    zw = sliding_window_view(z, m)
    bw = sliding_window_view(b, m)
    nobs = bw.sum(axis=1)                      # exact (integers)
    sumZ = zw.sum(axis=1)                      # pairwise summation
    sumX = sliding_window_view(x, m).sum(axis=1)
    complete = nobs == m
    var_full = sumX / m - (sumZ / m) ** 2

    if vlimits == "window":
        obs_w = sliding_window_view(mask, m)
        with np.errstate(invalid="ignore"):
            vmin = np.where(obs_w, zw, np.inf).min(axis=1)
            vmax = np.where(obs_w, zw, -np.inf).max(axis=1)
    elif vlimits == "global":
        vmin = np.full(L, z[mask].min())
        vmax = np.full(L, z[mask].max())
    elif vlimits == "slew":
        # limits = delta, or (delta, lo, hi) to intersect with a declared value range
        lim = np.atleast_1d(np.asarray(limits if limits is not None else [np.nan], float))
        if lim.size not in (1, 3) or not np.isfinite(lim[0]) or lim[0] <= 0:
            raise ValueError("vlimits='slew' requires limits=delta > 0 or (delta, lo, hi)")
        lo_t, hi_t = slew_intervals(T, mask, float(lim[0]))
        if lim.size == 3:
            lo_d, hi_d = float(lim[1]), float(lim[2])
            obs = T[mask]
            if not (lo_d <= hi_d) or obs.min() < lo_d or obs.max() > hi_d:
                raise ValueError("declared range does not contain the observed range")
            lo_t = np.maximum(lo_t, lo_d)
            hi_t = np.minimum(hi_t, hi_d)
        lo_s, hi_s = (lo_t - mu) / sd, (hi_t - mu) / sd
        miss_w = sliding_window_view(~mask, m)
        vmin = np.where(miss_w, sliding_window_view(lo_s, m), np.inf).min(axis=1)
        vmax = np.where(miss_w, sliding_window_view(hi_s, m), -np.inf).max(axis=1)
        vmin = np.where(np.isfinite(vmin), vmin, 0.0)        # complete windows: unused
        vmax = np.where(np.isfinite(vmax), vmax, 0.0)
    else:  # declared
        if limits is None:
            raise ValueError("vlimits='declared' requires limits=(vmin, vmax)")
        lo, hi = (float(limits[0]), float(limits[1]))
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError(f"invalid declared limits {limits!r}")
        obs = T[mask]
        if obs.min() < lo or obs.max() > hi:
            raise ValueError(
                f"declared limits {limits!r} do not contain the observed range "
                f"[{obs.min()}, {obs.max()}]; the limits are wrong")
        vmin = np.full(L, (lo - mu) / sd)
        vmax = np.full(L, (hi - mu) / sd)

    with np.errstate(divide="ignore", invalid="ignore"):
        mub = nobs / m
        ur = np.where(nobs > 0, sumZ / np.maximum(nobs, 1), 0.0)
        vr = np.maximum(np.where(nobs > 0, sumX / np.maximum(nobs, 1) - ur ** 2, 0.0), 0.0)
        A = vmax + vmin
        Vp = vmax * vmin
        C = vmax - vmin
        Fden = mub * (vr + Vp + ur * (ur - A)) + C ** 2 / 4.0
        if opt_center:
            nm = m - nobs
            mid = 0.5 * (vmax + vmin)
            c1 = (sumZ + nm * vmin) / m
            c2 = (sumZ + nm * vmax) / m
            c = np.where(c1 >= mid, c1, np.where(c2 <= mid, c2, mid))
            Fopt = (sumX - 2 * c * sumZ + nobs * c * c + nm * (0.5 * C + np.abs(c - mid)) ** 2) / m
            Fden = np.where(np.isfinite(Fopt), np.minimum(Fden, Fopt), Fden)

    valid = (nobs > 0) & ~(complete & (var_full <= EPS_VAR))     # G1, G2
    inv_den = np.zeros(L)
    cmask = valid & complete
    inv_den[cmask] = 1.0 / var_full[cmask]
    imask = valid & ~complete & (Fden > EPS_VAR)
    inv_den[imask] = 1.0 / Fden[imask]

    return dict(n=n, m=m, L=L, z=z, b=b, x=x, mu=mu, sd=sd, nobs=nobs,
                valid=valid, complete=complete, var_full=var_full, Fden=Fden,
                inv_den=inv_den, vmin=vmin, vmax=vmax, vlimits=vlimits, opt_center=opt_center)


# --------------------------------------------------------------------------- #
# Section 2: lower bound from the six sums (vectorized over pairs)
# --------------------------------------------------------------------------- #
def lb_d2_from_sums(R, Si, Sj, Qi, Qj, Pij, nobs_i, nobs_j, inv_i, inv_j,
                    valid_i, valid_j, m, return_case=False):
    """Squared lower-bound distance for arrays of pairs (spec section 2 + G1, G6)."""
    R = np.asarray(R)
    Rf = R.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rs = np.where(R > 0, Rf, 1.0)
        mi = Si / Rs
        mj = Sj / Rs
        vi = np.maximum(Qi / Rs - mi * mi, 0.0)
        vj = np.maximum(Qj / Rs - mj * mj, 0.0)
        cov = Pij / Rs - mi * mj
        vv = vi * vj
        q = np.where(vv > EPS_Q, cov / np.sqrt(np.where(vv > EPS_Q, vv, 1.0)), 0.0)
        q = np.clip(q, -1.0, 1.0)
        # Exactly two co-observed points: the correlation is exactly +-1. Computing it
        # from the sums leaves ~1e-9 of rounding in 1 - q^2, so set it exactly.
        q = np.where((R == 2) & (vv > EPS_Q), np.sign(cov), q)
        corr = np.where(q > 0, 1.0 - q * q, 1.0)

        ci = nobs_i == m
        cj = nobs_j == m
        case = np.where(ci & cj, CASE1, np.where(ci | cj, CASE2, CASE3))
        d2_c1 = np.maximum(2.0 * m * (1.0 - q), 0.0)
        v_c = np.where(ci, vi, vj)
        inv_c = np.where(ci, inv_i, inv_j)
        d2_c2 = Rf * v_c * inv_c * corr
        d2_c3 = Rf * np.maximum(vi * inv_i, vj * inv_j) * corr
        d2 = np.where(case == CASE1, d2_c1, np.where(case == CASE2, d2_c2, d2_c3))

    d2 = np.where(R == 0, 0.0, d2)
    case = np.where(R == 0, CASE_R0, case)
    invalid = ~(valid_i & valid_j)
    d2 = np.where(invalid, INF, d2)                      # G1: precedence over R = 0
    case = np.where(invalid, CASE_INVALID, case)
    d2 = np.where(np.isnan(d2), INF, d2)                  # G6
    d2 = np.where(d2 < 0, 0.0, d2)
    return (d2, case) if return_case else d2


def _sums_brute(ws, i, j):
    """Six sums for one pair by direct summation over the window (for tests)."""
    m = ws["m"]
    z, b, x = ws["z"], ws["b"], ws["x"]
    zi, zj = z[i:i + m], z[j:j + m]
    bi, bj = b[i:i + m], b[j:j + m]
    xi, xj = x[i:i + m], x[j:j + m]
    return (int(np.sum(bi * bj)), float(np.sum(zi * bj)), float(np.sum(bi * zj)),
            float(np.sum(xi * bj)), float(np.sum(bi * xj)), float(np.sum(zi * zj)))


def pair_lb_d2(ws, i, j):
    """Squared lower bound for one pair (i, j) by direct summation."""
    R, Si, Sj, Qi, Qj, Pij = _sums_brute(ws, i, j)
    d2 = lb_d2_from_sums(np.array([R]), np.array([Si]), np.array([Sj]), np.array([Qi]),
                         np.array([Qj]), np.array([Pij]),
                         ws["nobs"][i:i + 1], ws["nobs"][j:j + 1],
                         ws["inv_den"][i:i + 1], ws["inv_den"][j:j + 1],
                         ws["valid"][i:i + 1], ws["valid"][j:j + 1], ws["m"])
    return float(d2[0])


def excl_zone(m, excl=0.25):
    return int(np.ceil(m * excl))


# --------------------------------------------------------------------------- #
# Diagonal-wise exact sums (extended-precision prefix sums)
# --------------------------------------------------------------------------- #
def _win_sums(prod, m):
    """Sliding window sums of length m of a 1-D product array (long double prefix sums)."""
    c = np.empty(len(prod) + 1, dtype=np.longdouble)
    c[0] = 0
    np.cumsum(prod.astype(np.longdouble), out=c[1:])
    return (c[m:] - c[:-m]).astype(np.float64)


def diagonal_sums(ws, k):
    """Six sums for all pairs (i, i+k), i = 0 .. L-k-1."""
    m = ws["m"]
    z, b, x = ws["z"], ws["b"], ws["x"]
    n = ws["n"]
    zi, zj = z[:n - k], z[k:]
    bi, bj = b[:n - k], b[k:]
    xi, xj = x[:n - k], x[k:]
    bb = bi * bj
    cb = np.concatenate(([0], np.cumsum(bb)))
    R = cb[m:] - cb[:-m]                                  # exact integers
    return (R, _win_sums(zi * bj, m), _win_sums(bi * zj, m), _win_sums(xi * bj, m),
            _win_sums(bi * xj, m), _win_sums(zi * zj, m))


def diagonal_lb_d2(ws, k, return_case=False):
    """Squared lower bounds for all pairs on diagonal k (i, i+k)."""
    L = ws["L"]
    R, Si, Sj, Qi, Qj, Pij = diagonal_sums(ws, k)
    ii = np.arange(L - k)
    jj = ii + k
    return lb_d2_from_sums(R, Si, Sj, Qi, Qj, Pij,
                           ws["nobs"][ii], ws["nobs"][jj],
                           ws["inv_den"][ii], ws["inv_den"][jj],
                           ws["valid"][ii], ws["valid"][jj], ws["m"],
                           return_case=return_case)


def mdms_profile(T, mask, m, excl=0.25, vlimits="window", limits=None):
    """
    O(n^2) lower-bound matrix profile (symmetric self-join).
    Returns (P, I) with P the lower-bound distance (INF where no valid neighbour).
    Ties go to the smaller index.
    """
    ws = window_stats(T, mask, m, vlimits, limits)
    L = ws["L"]
    kmin = excl_zone(m, excl) + 1
    P2 = np.full(L, INF)
    I = np.full(L, -1, dtype=np.int64)
    for k in range(kmin, L):
        d2 = diagonal_lb_d2(ws, k)
        ii = np.arange(L - k)
        jj = ii + k
        _update_profile(P2, I, ii, jj, d2)
        _update_profile(P2, I, jj, ii, d2)
    return np.sqrt(P2), I


def _update_profile(P2, I, rows, cols, d2):
    """P2[rows] = min(P2[rows], d2) keeping the smallest column index on ties."""
    better = (d2 < P2[rows]) | ((d2 == P2[rows]) & (cols < I[rows]) & np.isfinite(d2))
    P2[rows[better]] = d2[better]
    I[rows[better]] = cols[better]


def brute_force_profile(T, mask, m, excl=0.25, vlimits="window", limits=None):
    """O(n^2 m) reference: every pair by direct summation. Use for small n only."""
    ws = window_stats(T, mask, m, vlimits, limits)
    L = ws["L"]
    ez = excl_zone(m, excl)
    P2 = np.full(L, INF)
    I = np.full(L, -1, dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= ez:
                continue
            d2 = pair_lb_d2(ws, i, j)
            if d2 < P2[i] or (d2 == P2[i] and np.isfinite(d2) and j < I[i]):
                P2[i] = d2
                I[i] = j
    return np.sqrt(P2), I


# --------------------------------------------------------------------------- #
# Oracle: exact z-normalized distances on the complete series
# --------------------------------------------------------------------------- #
def oracle_stats(Tc, m):
    """Per-window mean/std of the complete series (standardized globally first)."""
    Tc = np.asarray(Tc, dtype=np.float64)
    if not np.all(np.isfinite(Tc)):
        raise ValueError("oracle series must be complete")
    s = (Tc - Tc.mean()) / Tc.std()
    w = sliding_window_view(s, m)
    mu = w.mean(axis=1)
    sd = w.std(axis=1)
    return dict(s=s, mu=mu, sd=sd, m=m, n=len(Tc), L=len(Tc) - m + 1,
                flat=sd ** 2 <= EPS_VAR)


def oracle_diagonal_d2(os_, k):
    """Exact squared z-normalized distances for pairs (i, i+k) of the complete series."""
    s, m, n, L = os_["s"], os_["m"], os_["n"], os_["L"]
    dot = _win_sums(s[:n - k] * s[k:], m)
    ii = np.arange(L - k)
    jj = ii + k
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (dot / m - os_["mu"][ii] * os_["mu"][jj]) / (os_["sd"][ii] * os_["sd"][jj])
    rho = np.clip(rho, -1.0, 1.0)
    d2 = np.maximum(2.0 * m * (1.0 - rho), 0.0)
    bad = os_["flat"][ii] | os_["flat"][jj]
    return np.where(bad, INF, d2)


def oracle_pair_d2(Tc, m, i, j):
    """Direct oracle squared distance for one pair (independent check)."""
    a = np.asarray(Tc[i:i + m], dtype=np.float64)
    c = np.asarray(Tc[j:j + m], dtype=np.float64)
    sa, sc = a.std(), c.std()
    if sa ** 2 <= EPS_VAR * np.var(Tc) or sc ** 2 <= EPS_VAR * np.var(Tc):
        return INF
    return float(np.sum(((a - a.mean()) / sa - (c - c.mean()) / sc) ** 2))


# --------------------------------------------------------------------------- #
# Two-window version (used by the adversarial search, T7)
# --------------------------------------------------------------------------- #
def two_window_lb_d2(x, bx, y, by, vlimits="window", limits=None):
    """
    Lower bound for a single pair of windows, given complete values x, y and
    observation masks bx, by (missing values are hidden from the bound).
    'window' uses each window's observed min/max; 'declared' uses limits.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    bx = np.asarray(bx, bool)
    by = np.asarray(by, bool)
    m = len(x)
    T = np.concatenate([x, y])
    mask = np.concatenate([bx, by])
    # Build a series where windows 0 and m are exactly x and y.
    # (Windows in between are irrelevant; standardization is distance-invariant
    # except for the limits, which are transformed consistently.)
    ws = window_stats(T, mask, m, vlimits, limits)
    return pair_lb_d2(ws, 0, m)


def two_window_true_d2(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return INF
    return float(np.sum(((x - x.mean()) / x.std() - (y - y.mean()) / y.std()) ** 2))


# --------------------------------------------------------------------------- #
# Missing-data generation (G7)
# --------------------------------------------------------------------------- #
def generate_missing_mask(n, m, rate, pattern="random", block_frac=0.2, seed=0,
                          max_attempts=None):
    """
    Return (mask, achieved_rate). True = observed.

    pattern:
      'random' : MCAR, exactly round(rate * n) missing positions
      'block'  : non-overlapping blocks of length max(1, round(block_frac*m)),
                 placed until the target count is reached (last block truncated)
      'mixed'  : half the budget as blocks (placed first), half random
    Raises ValueError when the target cannot be reached.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    if pattern not in ("random", "block", "mixed"):
        raise ValueError(f"unknown pattern {pattern!r}")
    rng = np.random.default_rng(seed)
    target = int(round(rate * n))
    mask = np.ones(n, dtype=bool)
    if target == 0:
        return mask, 0.0
    blen = max(1, int(round(block_frac * m)))
    max_attempts = max_attempts or 100 * n

    def place_blocks(budget):
        placed, attempts = 0, 0
        while placed < budget:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"could not place {budget} missing points in blocks of {blen} "
                    f"(placed {placed}); lower the rate or the block length")
            length = min(blen, budget - placed)
            start = int(rng.integers(0, n - length + 1))
            # require a one-sample gap so blocks stay distinct
            lo, hi = max(0, start - 1), min(n, start + length + 1)
            if not mask[lo:hi].all():
                continue
            mask[start:start + length] = False
            placed += length

    if pattern == "block":
        place_blocks(target)
    elif pattern == "mixed":
        place_blocks(target // 2)
        remaining = target - int((~mask).sum())
        free = np.flatnonzero(mask)
        if remaining > len(free):
            raise ValueError("not enough observed points left for the random part")
        mask[rng.choice(free, size=remaining, replace=False)] = False
    else:
        mask[rng.choice(n, size=target, replace=False)] = False

    achieved = float((~mask).sum()) / n
    if int((~mask).sum()) != target:
        raise ValueError(f"achieved {int((~mask).sum())} missing points, target {target}")
    return mask, achieved


def clipping_mask(T, quantile=0.99):
    """Saturation dropout: samples with |T - median| above the quantile are missing."""
    dev = np.abs(np.asarray(T, float) - np.median(T))
    thr = np.quantile(dev, quantile)
    mask = dev <= thr
    return mask, float((~mask).mean())
