"""The Fit quality tab's numbers (fit_quality): the S rows with their F,
MISFIT, and the ideals and 1σ of χ² and D.

Pure functions on made-up spectra -- Poisson counts drawn around a known model
-- each against an independent reference: a hand computation, the scale
invariance the measures are built on, the smooths written out as dense
matrices, the Poisson probabilities summed with math.lgamma.
docs/residual_quality.md derives them.
"""
import math

import numpy as np

from syncmoss import fit_quality as fq

N0 = 1e5
S_ROWS = (fq.STRUCTURE, fq.STRUCTURE_FREE, fq.STRUCTURE_SG)
NO_MODEL = (fq.STRUCTURE_FREE, fq.STRUCTURE_SG)


def _sextet(n=512, v_max=10.0, counts=N0):
    """Velocity, model counts and baseline of a thin alpha-Fe-like sextet."""
    v = -v_max + 2 * v_max * (np.arange(n) + 0.5) / n
    absorption = sum(depth * 0.125 ** 2 / ((v - c) ** 2 + 0.125 ** 2)
                     for c, depth in zip((-5.31, -3.08, -0.84, 0.84, 3.08, 5.31),
                                         (0.12, 0.08, 0.04, 0.04, 0.08, 0.12)))
    baseline = np.full(n, counts)
    return v, baseline * (1 - absorption), baseline


def _rows(quality):
    """{name: (ideal, value, 1σ, F, F's 1σ)}"""
    return {row[0]: row[1:] for row in quality['rows']}


def _quality(v, y, m, baseline, fitted=None, n_free=0, chi2=None, deviance=1.0):
    fitted = np.ones(len(v), bool) if fitted is None else fitted
    if chi2 is None:
        chi2 = np.sum((y[fitted] - m[fitted]) ** 2 / m[fitted]) / (fitted.sum() - n_free)
    return fq.fit_quality([(v, y, m, baseline, fitted)], chi2, deviance, n_free)[0]


def _f_reads_one(rows, name, spreads=3):
    F, sigma = rows[name][3], rows[name][4]
    return abs(F - 1) < spreads * sigma


def test_noise_only_reads_ideal_and_a_misfit_does_not():
    """Poisson counts around the model: every row within 3σ of its ideal and
    every F within 3σ of 1. The same counts over a smooth misfit the model
    lacks (a curved baseline): the S rows and MISFIT far beyond theirs, the F
    of the counts alone still at 1."""
    v, m, baseline = _sextet()
    y = np.random.default_rng(7).poisson(m).astype(float)
    rows = _rows(_quality(v, y, m, baseline))
    assert set(S_ROWS) <= set(rows)
    for name, (ideal, value, sigma, F, F_sigma) in rows.items():
        assert abs(value - ideal) < 3 * sigma, name
        assert F is None or abs(F - 1) < 3 * F_sigma, name

    curved = y * (1 + 0.01 * ((v / 10) ** 2 - 1 / 3))
    rows = _rows(_quality(v, curved, m, baseline))
    for name in S_ROWS + (fq.MISFIT,):
        ideal, value, sigma = rows[name][:3]
        assert value - ideal > 5 * sigma, name
    for name in NO_MODEL:
        assert _f_reads_one(rows, name), name           # the counting is still Poisson


def test_a_spectrum_multiplied_by_ten_changes_only_f():
    """Counts multiplied by 10 (not Poisson any more): χ² and every F exactly
    10 times larger, the S rows and MISFIT exactly the same -- the mask, the
    smooths and their weights scale with the counts."""
    v, m, baseline = _sextet()
    y = np.random.default_rng(3).poisson(m).astype(float)
    one, ten = _rows(_quality(v, y, m, baseline)), _rows(_quality(v, 10 * y, 10 * m, 10 * baseline))
    assert np.isclose(ten[fq.CHI2][1], 10 * one[fq.CHI2][1], rtol=1e-9)
    for name in S_ROWS:
        assert np.isclose(ten[name][1], one[name][1], rtol=1e-9), name
        assert np.isclose(ten[name][3], 10 * one[name][3], rtol=1e-9), name
    assert np.isclose(ten[fq.MISFIT][1], one[fq.MISFIT][1], rtol=1e-9)


def test_s_fit_by_hand_with_an_exclusion_region_and_any_point_order():
    """S (fit) = (sum z^2 / nu) / (sum (z_k+1 - z_k)^2 / (2 pairs)), the pairs
    only between neighbours that are both fitted -- never across an excluded
    run -- whatever order the points come in; its F is that denominator."""
    v, m, baseline = _sextet(n=64)
    y = np.random.default_rng(11).poisson(m).astype(float)
    y[20:30] += 50000                     # excluded: would dominate the differences
    fitted = np.ones(64, bool)
    fitted[20:30] = False
    z = (y - m) / np.sqrt(m)
    kept = [z[:20], z[30:]]
    noise = sum(np.sum(np.diff(part) ** 2) for part in kept) / (2 * sum(len(part) - 1 for part in kept))
    nu = 54 - 3
    expected = (np.sum(z[fitted] ** 2) / nu) / noise

    rows = _rows(_quality(v, y, m, baseline, fitted, n_free=3))
    assert np.isclose(rows[fq.STRUCTURE][1], expected, rtol=1e-12)
    assert np.isclose(rows[fq.STRUCTURE][3], noise, rtol=1e-12)
    shuffled = np.random.default_rng(1).permutation(64)
    again = _rows(_quality(v[shuffled], y[shuffled], m[shuffled], baseline[shuffled], fitted[shuffled], n_free=3))
    assert np.isclose(again[fq.STRUCTURE][1], expected, rtol=1e-12)


def _gap_case(n=96, seed=21):
    """Counts of a sextet on n points with an excluded run, their local mean
    and the mask."""
    v, m, baseline = _sextet(n=n)
    y = np.random.default_rng(seed).poisson(m).astype(float)
    fitted = np.ones(n, bool)
    fitted[40:47] = False
    counts, mean = np.where(fitted, y, 0.0), fq.counts_mean(y, fitted)
    channels = fq.keep_away_from_lines(counts, mean, fitted)
    assert 4 < channels.sum() < fitted.sum()      # the mask left lines out, not everything
    return counts, mean, channels


def _dense_pass(y, keep, weights):
    """One pass of the masked smoother written out: (W + alpha D'D)^-1 W
    inverted at every alpha of the search, W = the weights at mean 1 on the
    kept points and 0 elsewhere. Returns M = I - hat and the weights."""
    n = len(y)
    P = fq.penalty(n)
    w = np.zeros(n)
    w[keep] = weights / np.mean(weights)

    def hat(alpha):
        return np.linalg.inv(np.diag(w) + alpha * P) * w[None, :]

    def gcv(alpha):
        H = hat(alpha)
        r = y - H @ y
        return keep.sum() * np.sum(w[keep] * r[keep] ** 2) / (keep.sum() - np.trace(H[np.ix_(keep, keep)])) ** 2

    best = int(np.argmin([gcv(a) for a in fq.ALPHAS]))
    fine = np.geomspace(fq.ALPHAS[max(best - 1, 0)], fq.ALPHAS[min(best + 1, len(fq.ALPHAS) - 1)], fq.REFINE)
    return np.eye(n) - hat(fine[int(np.argmin([gcv(a) for a in fine]))] / fq.UNDERSMOOTH), w


def test_the_smoother_is_its_dense_form():
    """The masked smoother through the eigendecomposition against its dense
    form, both passes: equal weights give the counts' own smooth, then the
    weights 1/smooth give F = sum_U w r^2 / sum_U w_i sum_j M_ij^2 y_j and the
    spread sqrt(2 sum_jk C_jk^2 mu_j mu_k) / sum_k C_kk mu_k of the residual's
    form C = M' W M, mu the smooth."""
    y, mean, U = _gap_case()
    M, _ = _dense_pass(y, U, np.ones(U.sum()))
    smooth = np.where(U, y - M @ y, 0.0)
    keep = U & (smooth > 0)
    M, w = _dense_pass(y, keep, 1.0 / smooth[keep])
    C = (M[keep] * w[keep][:, None]).T @ M[keep]
    F = np.sum(w[keep] * (M @ y)[keep] ** 2) / (np.diag(C) @ y)
    sigma = np.sqrt(2 * np.sum((C * smooth[None, :] * smooth[:, None]) * C)) / (np.diag(C) @ smooth)

    got = fq.smoother_noise(y, U)
    assert np.allclose(got[:2], (F, sigma), rtol=1e-8)
    assert np.allclose(got[2], smooth, rtol=1e-8)


def test_savgol_noise_is_its_dense_form():
    """The Savitzky-Golay residual's sums over a band against the residual
    matrix written out: rows at the points whose whole window is kept, each
    weighted by 1/smooth, F = sum w r^2 / sum_k (B'WB)_kk y_k, spread
    sqrt(2 sum_ij w_i w_j (B diag(mu) B')_ij^2) / sum_k (B'WB)_kk mu_k."""
    y, mean, U = _gap_case(seed=23)
    smooth = fq.smoother_noise(y, U)[2]
    n = len(y)
    half = fq.SG_POINTS // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    V = offsets[:, None] ** np.arange(fq.SG_DEGREE + 1)[None, :]
    fit = (V @ np.linalg.inv(V.T @ V) @ V.T)[half]
    rows = [i for i in range(half, n - half) if U[i - half:i + half + 1].all() and smooth[i] > 0]
    B = np.zeros((len(rows), n))
    for r, i in enumerate(rows):
        B[r, i - half:i + half + 1] = -fit
        B[r, i] += 1.0
    w = 1.0 / smooth[rows]
    share = np.sum(B ** 2 * w[:, None], axis=0)
    F = np.sum(w * (B @ y) ** 2) / (share @ y)
    BVB = (B * smooth[None, :]) @ B.T
    sigma = np.sqrt(2 * np.sum(w[:, None] * w[None, :] * BVB ** 2)) / (share @ smooth)

    assert np.allclose(fq.savgol_noise(y, U, smooth), (F, sigma), rtol=1e-8)


def test_a_long_spectrum_is_smoothed_in_blocks_whose_sums_add():
    """1024 points: two blocks of 512, each smoothed on its own. F of the
    whole is the blocks' F weighted by their normalisations -- between the two
    -- and its 1σ smaller than either's."""
    v, m, baseline = _sextet(n=1024)
    y = np.random.default_rng(31).poisson(m).astype(float)
    fitted = np.ones(1024, bool)
    U = fq.keep_away_from_lines(y, fq.counts_mean(y, fitted), fitted)
    whole = fq.smoother_noise(y, U)
    halves = [fq.smoother_noise(y[part], U[part]) for part in (slice(0, 512), slice(512, 1024))]
    low, high = sorted(half[0] for half in halves)
    assert low <= whole[0] <= high and low < high
    assert whole[1] < min(half[1] for half in halves)
    assert np.array_equal(whole[2], np.concatenate([half[2] for half in halves]))


def test_a_line_saturated_to_no_counts_at_its_bottom():
    """A singlet exp(-20 L): about 1e-8 counts at its bottom, 2 counts per
    point at the baseline. The smooth is 0 or below there -- weight 0, no noise
    to measure -- and every F stays at 1 within its 1σ."""
    v = -4 + 8 * (np.arange(512) + 0.5) / 512
    baseline = np.full(512, 2.0)
    m = baseline * np.exp(-20 * 0.1 ** 2 / (v ** 2 + 0.1 ** 2))
    y = np.random.default_rng(37).poisson(m).astype(float)
    rows = _rows(_quality(v, y, m, baseline))
    assert np.sum(y[np.abs(v) < 0.3] == 0) > 30               # the bottom is empty
    for name in S_ROWS:
        assert np.all(np.isfinite(rows[name])) and _f_reads_one(rows, name), name


def test_f_without_a_model_does_not_see_the_model_but_f_of_the_fit_does():
    """A model without any line: the F of the counts alone is exactly the
    right model's -- no model enters it -- while the neighbour differences of
    the residual measure the lines the model lacks."""
    v, m, baseline = _sextet()
    y = np.random.default_rng(13).poisson(m).astype(float)
    right, no_line = _rows(_quality(v, y, m, baseline)), _rows(_quality(v, y, baseline, baseline))
    for name in NO_MODEL:
        assert no_line[name][3:] == right[name][3:], name
    F, sigma = no_line[fq.STRUCTURE][3:]
    assert F - 1 > 10 * sigma


def test_f_without_a_model_survives_an_excluded_range_through_a_line():
    """An excluded range through the deepest line, its counts spoiled: the
    smooths never reach them, so the F of the counts alone still reads the
    noise."""
    v, m, baseline = _sextet()
    y = np.random.default_rng(17).poisson(m).astype(float)
    fitted = ~((v > -5.31) & (v < -4.8))           # half of line 1 out
    y[~fitted] += 50000.0
    rows = _rows(_quality(v, y, m, baseline, fitted))
    for name in NO_MODEL:
        assert _f_reads_one(rows, name), name


def test_two_counts_per_point_zeros_included():
    """Two counts per point, many of them zero: a division by each point's own
    count would break (by zero) or bias (by 1/counts); with the counts only in
    the sums every F stays at 1 within its 1σ."""
    v, m, baseline = _sextet(counts=2.0)
    y = np.random.default_rng(29).poisson(m).astype(float)
    quality = _quality(v, y, m, baseline)
    rows = _rows(quality)
    assert np.sum(y == 0) > 50 and quality['kept'] > 0.5
    for name in S_ROWS:
        assert np.all(np.isfinite(rows[name])), name
        assert _f_reads_one(rows, name), name


def test_the_smooths_need_the_natural_line_on_two_channels():
    """256 points over +-10 mm/s: the natural line (0.194 mm/s) spans 2.5
    channels; the mask leaves the lines out and every S row reads its ideal.
    64 points: 0.6 channel, every line a spike -- no S (no model) or S (SG),
    whose F would read about 4, while S (fit) stays. The step that counts is
    where the spectrum absorbs."""
    v, m, baseline = _sextet(n=256)
    y = np.random.default_rng(5).poisson(m).astype(float)
    quality = _quality(v, y, m, baseline)
    rows = _rows(quality)
    assert np.isclose(quality['line'], fq.NARROWEST_LINE / (v[1] - v[0])) and 0.3 < quality['kept'] < 1
    for name in S_ROWS:
        ideal, value, sigma, F, F_sigma = rows[name]
        assert abs(value - ideal) < 3 * sigma and abs(F - 1) < 3 * F_sigma, name
    # a last step three times the others (a fold's turning point), where
    # nothing absorbs, does not count
    v_odd = v.copy()
    v_odd[-1] += 2 * (v[1] - v[0])
    assert np.isclose(_quality(v_odd, y, m, baseline)['line'], quality['line'])

    v, m, baseline = _sextet(n=64)
    y = np.random.default_rng(5).poisson(m).astype(float)
    quality = _quality(v, y, m, baseline)
    rows = _rows(quality)
    assert quality['line'] < fq.SAMPLES_PER_LINE and quality['kept'] == 0
    assert fq.STRUCTURE in rows and not set(NO_MODEL) & set(rows)
    assert 'no S (no model)' in fq.quality_line(quality)


def test_misfit_of_a_noise_free_misfit_is_its_power_over_the_absorption():
    """No noise: MISFIT is the misfit's power over the absorption's, the
    'measured noise' it subtracts being the misfit's own tiny leak."""
    v, m, baseline = _sextet()
    misfit = 0.004 * N0 * ((v / 10) ** 2 - 1 / 3)
    rows = _rows(_quality(v, m + misfit, m, baseline))
    expected = 100 * np.sum(misfit ** 2 / m) / np.sum((baseline - m) ** 2 / m)
    assert np.isclose(rows[fq.MISFIT][1], expected, rtol=1e-3)


def test_deviance_moments_are_the_poisson_sums():
    """The mean and the variance of a point's deviance: the sums over the
    Poisson probabilities below SERIES_FROM_COUNTS, the series above -- both
    the sums here (minimization.md §7.2: 1.147 at 1 count, 1.019 at 10)."""
    def reference(mu):
        top = int(mu + 40 * math.sqrt(mu) + 40)
        p = np.array([math.exp(y * math.log(mu) - mu - math.lgamma(y + 1)) for y in range(top + 1)])
        y = np.arange(top + 1, dtype=float)
        d = 2 * (mu - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0))
        mean = np.sum(p * d)
        return mean, np.sum(p * d * d) - mean ** 2

    mus = np.array([0.5, 1.0, 10.0, 99.0, 100.0, 1000.0, 1e5])
    mean, variance = fq.deviance_moments(mus)
    for mu, got_mean, got_variance in zip(mus, mean, variance):
        ref_mean, ref_variance = reference(mu)
        assert abs(got_mean - ref_mean) < 1e-6, mu
        assert abs(got_variance - ref_variance) < 1e-4 * ref_variance, mu
    assert round(mean[1], 3) == 1.147 and round(mean[2], 3) == 1.019


def test_chi2_and_d_rows_are_the_fit_s_with_their_poisson_spreads():
    """χ² and D carry the fit's values; χ²'s 1σ is sqrt(sum(2 + 1/m))/nu, D's
    ideal 1 + sum(E[d] - 1)/nu -- above 1 for a spectrum of few counts. They
    have no F."""
    v, m, baseline = _sextet()
    y = np.random.default_rng(9).poisson(m).astype(float)
    rows = _rows(_quality(v, y, m, baseline, n_free=8, chi2=1.234, deviance=1.111))
    assert rows[fq.CHI2][1] == 1.234 and rows[fq.DEVIANCE][1] == 1.111
    assert np.isclose(rows[fq.CHI2][2], np.sqrt(np.sum(2 + 1 / m)) / (512 - 8))
    assert rows[fq.CHI2][3] is None and rows[fq.DEVIANCE][3] is None and rows[fq.MISFIT][3] is None

    few = m / N0 * 3                      # about 3 counts per point
    rows = _rows(_quality(v, np.random.default_rng(9).poisson(few).astype(float), few, baseline / N0 * 3))
    assert rows[fq.DEVIANCE][0] > 1.05
