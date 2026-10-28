"""Fit quality: numbers about a fit's residual that do not move the fit.

Computed once, when a fit of counts has converged (fitting_io.fit_model), and
shown in the "Fit quality" tab of the results table. docs/residual_quality.md
derives every one of them and tests them on simulated and measured spectra;
its §12 is the specification of this module.

Per spectrum, in velocity order, at the fitted points:

  χ²            Pearson's chi-square per degree of freedom -- the fit's
  D             the Poisson deviance per degree of freedom -- what the fit minimised
  S (fit)       χ² over the F of the fit's residual: the noise that the
                differences of neighbouring residuals measure. 1 when the
                residual is noise only, whatever the counting statistics
  S (no model)  χ² over the F of the counts alone: their scatter about their
                own smooth, the masked smoother. 1 when counting noise explains
                χ²; its F is the counting statistics
  S (SG)        the same with a Savitzky-Golay smooth: holds at very high
                counts, less precise
  MISFIT, %     the misfit's power over the absorption's (Ruby 1973), with the
                F of S (no model) in place of Poisson's

The S rows carry the F they divide χ² by. The smooths use only the points away
from the lines (keep_away_from_lines, from the counts): a line's peak is what a
smooth cannot follow. The F of S (no model) and of S (SG) come from the counts
alone -- the model enters nowhere: each point weighs the inverse of its Poisson
variance as the counts' own smooth gives it, and the counts enter the
normalisation only as a sum, whose mean is the Poisson variance exactly, so
that zero counts or a few counts per point bias nothing. They need the lines
sampled: none where the narrowest line spans fewer than SAMPLES_PER_LINE
channels.

Every row has its ideal value and its 1σ: the spread of the value over
repeated measurements when the model is right and the counting Poisson, as the
ideals assume; the same for each F.
"""
import numpy as np

from syncmoss.constants import NAT_WIDTH

# The names of the rows, as the tab shows them
CHI2 = 'χ²'
DEVIANCE = 'D'
STRUCTURE = 'S (fit)'
STRUCTURE_FREE = 'S (no model)'
STRUCTURE_SG = 'S (SG)'
MISFIT = 'MISFIT, %'

# From this many counts per point up, the mean and the variance of a point's
# Poisson deviance are the series 1 + 1/(6m) + 1/(6m^2) and 2 + 2/(3m): exact
# to 1e-6 and 1e-4 there (checked against the sums over the Poisson
# probabilities, docs/residual_quality.md §12); below, they are those sums.
SERIES_FROM_COUNTS = 100.0

# The narrowest line a spectrum can have, mm/s: the natural source line on the
# natural absorber line -- two Lorentzians of NAT_WIDTH (constants.py), whose
# FWHMs add when they are convolved. A broader source (the SMS instrumental
# function) or a thicker absorber only widens every line.
NARROWEST_LINE = 2 * NAT_WIDTH
# The smooths need it sampled, two points across its width: narrower, a line is
# a spike of a point or two that no smooth tells from the noise -- the F of the
# counts alone then reads 4 at 0.6 channel per line, and carries twice its 1σ
# at 1.2 (residual_quality.md §12)
SAMPLES_PER_LINE = 2

# The points the smooths use (keep_away_from_lines; the masked high-pass of
# residual_quality.md §6.6, with the values tested there): normalised third
# differences of the counts at the spacings 1..4 channels. Where one,
# divided by the cube of its spacing, exceeds both sqrt(LEAK) of a point's
# noise and SIGNIFICANCE spreads of its own, the third differences within
# REACH channels are left out, and the points they span.
DIFFERENCE = np.array([-1.0, 3.0, -3.0, 1.0]) / np.sqrt(20.0)
SPACINGS = (1, 2, 3, 4)
LEAK = 0.02              # the share of a point's noise that a leak may add
SIGNIFICANCE = 5.0       # in spreads of the leak's estimate
REACH = 1.5              # channels
ENDS = 2                 # third differences left out at each end of the spectrum
MEDIAN_CHI2_1 = 0.45494  # the median of chi-square of one degree of freedom
ITERATIONS = 5

# The masked smoother (residual_quality.md §6.6): the smooth of weights 1/variance on
# the kept points and a penalty on its third differences, whose weight is
# chosen by generalised cross-validation over ALPHAS, refined over REFINE
# values around the best and divided by UNDERSMOOTH -- undersmoothed, so that
# a line's flank stays in the smooth, not in the residual.
ALPHAS = 10.0 ** np.arange(-3, 9)
REFINE = 7
UNDERSMOOTH = 100.0
# The smoother reaches a few channels; longer spectra are smoothed in blocks of
# at most BLOCK points, whose sums add up (smoother_noise). Its cost grows with
# the cube of a block's length: about 50 ms for a block of 512 points.
BLOCK = 512

# The Savitzky-Golay smooth: the polynomial of degree SG_DEGREE fitted to the
# SG_POINTS points around each
SG_POINTS = 7
SG_DEGREE = 4


def deviance_moments(m):
    """Mean and variance of one point's Poisson deviance 2[m - y + y ln(y/m)],
    y Poisson with the mean *m* (an array)."""
    m = np.asarray(m, dtype=float)
    mean = 1 + 1 / (6 * m) + 1 / (6 * m ** 2)
    variance = 2 + 2 / (3 * m)
    low = m < SERIES_FROM_COUNTS
    if np.any(low):
        mu = m[low][:, None]
        # every y with a probability above double precision: mean + 12 sigma
        top = int(np.ceil(np.max(mu) + 12 * np.sqrt(np.max(mu)) + 12))
        y = np.arange(top + 1, dtype=float)[None, :]
        log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, top + 1)))))[None, :]
        p = np.exp(y * np.log(mu) - mu - log_factorial)
        y_ln = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        d = 2 * (mu - y + y_ln)
        mean[low] = np.sum(p * d, axis=1)
        variance[low] = np.sum(p * d * d, axis=1) - mean[low] ** 2
    return mean, variance


def line_in_channels(v, y, fitted):
    """The narrowest line in channels where the spectrum absorbs: NARROWEST_LINE
    over the largest velocity step there -- where the counts *y* dip below
    their mean by more than one point's noise, not at an edge of the axis (a
    fold's turning point) where nothing absorbs. None when they never dip."""
    counts = y[fitted]
    mean = np.mean(counts)
    dips = np.zeros(len(v), bool)
    dips[fitted] = mean - counts > np.sqrt(mean)
    steps = np.diff(v)[dips[:-1] | dips[1:]]
    return NARROWEST_LINE / np.max(steps) if len(steps) else None


def _spaced(x, h, coefficients):
    """sum_k coefficients[k] x[i + k h], for every i it reaches."""
    n = len(x) - (len(coefficients) - 1) * h
    return sum(c * x[k * h:k * h + n] for k, c in enumerate(coefficients))


def counts_mean(y, fitted):
    """Each point's mean count over the SG_POINTS points around it, of the
    fitted points only: the counts' own estimate of each point's Poisson
    variance, for the mask's thresholds (0 where no fitted point is near)."""
    window = np.ones(SG_POINTS)
    total = np.convolve(np.where(fitted, y, 0.0), window, 'same')
    number = np.convolve(fitted.astype(float), window, 'same')
    return np.where(number > 0, total / np.maximum(number, 1.0), 0.0)


def _leaks(y, mean, fitted, F):
    """The fine third differences kept: four fitted points, none within REACH
    channels of a centre where some spacing sees the spectrum leak through,
    none at the ends. *mean* is counts_mean."""
    flagged = np.zeros(2 * len(y) + 1, bool)          # the centres, in half channels
    for h in SPACINGS:
        if len(y) <= 3 * h:
            break
        d = _spaced(y, h, DIFFERENCE)
        sigma = np.sqrt(F * _spaced(mean, h, DIFFERENCE ** 2))
        whole = _spaced(fitted.astype(float), h, np.ones(4)) == 4
        bad = whole & (np.abs(d) / h ** 3 > np.maximum(np.sqrt(LEAK) * sigma, SIGNIFICANCE * sigma / h ** 3))
        flagged[2 * np.flatnonzero(bad) + 3 * h] = True
    reach = int(round(2 * REACH))
    near = np.convolve(flagged.astype(float), np.ones(2 * reach + 1), 'same') > 0
    keep = (_spaced(fitted.astype(float), 1, np.ones(4)) == 4) & ~near[2 * np.arange(len(y) - 3) + 3]
    keep[:ENDS] = keep[len(keep) - ENDS:] = False
    return keep


def keep_away_from_lines(y, mean, fitted):
    """The points the smooths use: fitted, and away from every sharp feature of
    the counts *y* -- outside every third difference the mask leaves out but
    those at the ends (_leaks; the mask's noise level is iterated with it).
    *mean* is counts_mean: the counts alone, no model."""
    n = len(y)
    channels = np.zeros(n, bool)
    if n < 4 + 2 * ENDS:
        return channels
    d = _spaced(y, 1, DIFFERENCE)
    var = _spaced(mean, 1, DIFFERENCE ** 2)
    whole = (_spaced(fitted.astype(float), 1, np.ones(4)) == 4) & (var > 0)
    if not whole.any():
        return channels
    F = np.median(d[whole] ** 2 / var[whole]) / MEDIAN_CHI2_1
    keep = None
    for _ in range(ITERATIONS):
        new = _leaks(y, mean, fitted, F)
        if not new.any() or (keep is not None and np.array_equal(new, keep)):
            keep = new
            break
        keep = new
        F = np.sum(d[keep] ** 2) / np.sum(var[keep])
    left_out = ~keep
    left_out[:ENDS] = left_out[len(left_out) - ENDS:] = False
    channels = fitted.copy()
    for k in range(4):
        channels[k:k + len(keep)] &= ~left_out
    return channels


def penalty(n):
    """D'D for the third differences D of n points ((n-3) x n)."""
    P = np.zeros((n, n))
    block = np.outer([-1.0, 3.0, -3.0, 1.0], [-1.0, 3.0, -3.0, 1.0])
    for i in range(n - 3):
        P[i:i + 4, i:i + 4] += block
    return P


def _eliminated_penalty(n, channels):
    """The penalty D'D on the kept points *channels*, the others eliminated:
    with weight 0 the penalty alone sets the smooth there, which leaves its
    Schur complement on the kept points."""
    P = penalty(n)
    others = ~channels
    Pkk = P[np.ix_(channels, channels)]
    if np.any(others):
        Pko = P[np.ix_(channels, others)]
        Pkk = Pkk - Pko @ np.linalg.solve(P[np.ix_(others, others)], Pko.T)
    return Pkk


def _smoother_pass(y, channels, weights):
    """One pass of the masked smoother with *weights* on the kept points (any
    positive scale): (g, Q, c, w, residual). w is the weights at mean 1,
    Q diag(theta) Q' the eliminated penalty in units of the weights,
    g = alpha theta/(1 + alpha theta) at the alpha cross-validation chooses
    (trace of the hat matrix sum 1/(1 + alpha theta), weighted residual sum
    sum (g c)^2), c = Q' W^(1/2) y, and the residual y - smooth on the kept
    points."""
    w = weights / np.mean(weights)
    scale = 1.0 / np.sqrt(w)
    theta, Q = np.linalg.eigh(_eliminated_penalty(len(y), channels) * scale[:, None] * scale[None, :])
    theta = np.maximum(theta, 0.0)                # the quadratics, zero to rounding
    c = Q.T @ (y[channels] / scale)
    kept = len(c)

    def gcv(alpha):
        return kept * np.sum((alpha * theta / (1 + alpha * theta) * c) ** 2) / (
            kept - np.sum(1 / (1 + alpha * theta))) ** 2

    best = int(np.argmin([gcv(alpha) for alpha in ALPHAS]))
    fine = np.geomspace(ALPHAS[max(best - 1, 0)], ALPHAS[min(best + 1, len(ALPHAS) - 1)], REFINE)
    alpha = fine[int(np.argmin([gcv(a) for a in fine]))] / UNDERSMOOTH
    g = alpha * theta / (1 + alpha * theta)
    return g, Q, c, w, scale * (Q @ (g * c))


def _smoother_block(y, channels):
    """The masked smoother's sums on one block (smoother_noise): (weighted
    residual sum, its normalisation, sum g^2, sum g^4, the counts' own smooth)
    with the weights 1/smooth themselves, or None with too few kept points."""
    if int(np.sum(channels)) <= 3:                # the penalty leaves quadratics free
        return None
    g, Q, c, w, residual = _smoother_pass(y, channels, np.ones(int(np.sum(channels))))
    smooth = np.zeros(len(y))
    smooth[channels] = y[channels] - residual
    weighted = channels & (smooth > 0)
    if int(np.sum(weighted)) <= 3:
        return None
    weights = 1.0 / smooth[weighted]
    g, Q, c, w, residual = _smoother_pass(y, weighted, weights)
    g2 = g ** 2
    scale = np.mean(weights)                      # back from the weights at mean 1
    return (scale * np.sum(g2 * c ** 2), scale * (((Q ** 2) @ g2) @ (w * y[weighted])), np.sum(g2),
            np.sum(g2 ** 2), smooth)


def smoother_noise(y, channels):
    """F of the counts' residual from the masked smoother on the kept points
    *channels*: (F, its 1σ for Poisson counts, the counts' own smooth over all
    points -- 0 off the kept ones), or None with too few kept points. The
    counts alone.

    Two passes. The first weighs the kept points equally and gives the counts'
    own smooth; the second weighs each kept point by 1/smooth, the inverse of
    its Poisson variance as the counts themselves estimate it, and by 0 where
    the smooth is not positive (no counts expected there, no noise to
    measure). Each pass in its eigendecomposition form (_smoother_pass). With
    M the residual's matrix and W the weights,

        F = sum_i w_i r_i^2 / sum_i w_i sum_j M_ij^2 y_j = sum g^2 c^2 / sum_j s_j w_j y_j ,

    s_j = (Q^2 g^2)_j: the counts enter the normalisation only as a sum, whose
    mean is the Poisson variance exactly -- a zero is just a count -- and the
    weights, whatever they are, set only its spread, not its mean. For Poisson
    counts with the smooth as their mean that spread is sqrt(2 sum g^4)/sum g^2
    (the cumulants of numerator and normalisation cancel).

    A spectrum of more than BLOCK points is smoothed in that many nearly equal
    blocks, each with its own two passes. Their noises are independent, so the
    sums add: F = sum of the residual sums / sum of the normalisations, and the
    1σ from the sums of g^2 and g^4 of all blocks. docs/residual_quality.md §12
    checks it against the dense form."""
    n = len(y)
    parts = []
    for block in np.array_split(np.arange(n), int(np.ceil(n / BLOCK))):
        part = _smoother_block(y[block], channels[block])
        if part is not None:
            parts.append((block, part))
    if not parts:
        return None
    residual_sum = sum(part[0] for _, part in parts)
    total = sum(part[1] for _, part in parts)
    if not total > 0:
        return None
    smooth = np.zeros(n)
    for block, part in parts:
        smooth[block] = part[4]
    g2 = sum(part[2] for _, part in parts)
    g4 = sum(part[3] for _, part in parts)
    return residual_sum / total, np.sqrt(2 * g4) / g2, smooth


def _savgol_filter():
    """The residual filter of the Savitzky-Golay smooth: 1 minus the centre
    row of the fit of SG_DEGREE to SG_POINTS points."""
    half = SG_POINTS // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    V = offsets[:, None] ** np.arange(SG_DEGREE + 1)[None, :]
    f = -(V @ np.linalg.pinv(V))[half]
    f[half] += 1.0
    return f


def savgol_noise(y, channels, smooth):
    """F of the counts' residual from their Savitzky-Golay smooth at the points
    whose whole window lies in *channels*: (F, its 1σ for Poisson counts), or
    None without such a point. Each such point weighs 1/smooth, *smooth* the
    counts' own (smoother_noise), 0 where it is not positive; as there,
    F = sum w r^2 over the sum of each count times its weighted share of the
    residual's noise. The residual is local, so every sum runs over a band of
    lags."""
    f = _savgol_filter()
    width = len(f)
    half = width // 2
    n = len(y)
    if n < width:
        return None
    inside = np.convolve(channels.astype(float), np.ones(width), 'valid') == width
    U = np.zeros(n, bool)
    U[half:n - half] = inside
    U &= smooth > 0
    if not U.any():
        return None
    w = np.where(U, 1.0 / np.where(U, smooth, 1.0), 0.0)
    mu = np.maximum(smooth, 0.0)
    r = np.zeros(n)
    r[half:n - half] = np.correlate(y, f, 'valid')
    # each count's share of the residual's noise: sum_i w_i f^2 over the windows on it
    share = np.convolve(w, f ** 2)[half:half + n]
    total = share @ y
    expected = share @ mu
    if not (total > 0 and expected > 0):
        return None
    F = np.sum(w * r ** 2) / total
    # the covariance of the weighted residuals at the lags d < width
    gauss = 0.0
    for d in range(width):
        lagged = np.correlate(mu, f[d:] * f[:width - d], 'valid')
        i = np.flatnonzero(U[:n - d] & U[d:])
        gauss += (1 if d == 0 else 2) * np.sum(w[i] * w[i + d] * lagged[i - half + d] ** 2)
    return F, np.sqrt(2 * gauss) / expected


def _spectrum_rows(v, y, m, baseline, fitted, dof):
    """The rows of one spectrum (the S rows with their F, MISFIT) and what they
    rest on."""
    order = np.argsort(v, kind='stable')
    v, y, m, baseline, fitted = v[order], y[order], m[order], baseline[order], fitted[order]
    z = np.full(len(v), np.nan)
    z[fitted] = (y[fitted] - m[fitted]) / np.sqrt(m[fitted])
    zk = z[fitted]
    n = len(zk)
    # neighbour differences only between points that are neighbours on the
    # axis and both fitted: none across an exclusion region
    pairs = fitted[:-1] & fitted[1:]
    differences = np.diff(z)[pairs]
    if len(differences) == 0:
        return None
    neighbours = np.sum(differences ** 2) / (2 * len(differences))
    chi2 = np.sum(zk ** 2) / dof
    chi2_variance = np.sum(2 + 1 / m[fitted]) / dof ** 2
    rows = [(STRUCTURE, 1.0, chi2 / neighbours, 1 / np.sqrt(n), neighbours, np.sqrt(3 / n))]

    # the rows without a model: the counts alone, the model nowhere
    counts = np.where(fitted, y, 0.0)
    mean = counts_mean(y, fitted)
    line = line_in_channels(v, y, fitted)
    if line is None or line >= SAMPLES_PER_LINE:
        channels = keep_away_from_lines(counts, mean, fitted)
    else:
        channels = np.zeros(len(v), bool)        # the lines are spikes: no smooth
    free = smoother_noise(counts, channels)
    savgol = None if free is None else savgol_noise(counts, channels, free[2])
    sigma_free = None
    for name, noise in ((STRUCTURE_FREE, free), (STRUCTURE_SG, savgol)):
        if noise is None:
            continue
        F, sigma = noise[:2]
        # χ² and F share the noise (Cov = 2/nu for Poisson counts): the spread of
        # their ratio is the smaller
        sigma_s = np.sqrt(max(sigma ** 2 + chi2_variance - 4 / dof, 0.0))
        if name == STRUCTURE_FREE:
            sigma_free = sigma_s
        rows.append((name, 1.0, chi2 / F, sigma_s, F, sigma))

    absorption_power = float(np.sum((baseline[fitted] - m[fitted]) ** 2 / m[fitted]))
    if absorption_power > 0:
        if free is not None:
            level, misfit_sd = free[0], free[0] * dof * sigma_free
        else:
            level, misfit_sd = neighbours, neighbours * np.sqrt(max(dof * (3 * dof / n - 2), 0))
        misfit = (np.sum(zk ** 2) - dof * level) / absorption_power
        rows.append((MISFIT, 0.0, 100 * misfit, 100 * misfit_sd / absorption_power, None, None))
    return {'points': n, 'dof': dof, 'line': line, 'kept': float(np.mean(channels[fitted])), 'rows': rows}


def quality_line(quality):
    """What one spectrum's Fit quality rests on, for the terminal (ASCII):
    '512 points, 504 degrees of freedom; S (no model) and S (SG) use 71 % of
    them, away from the lines'."""
    start = f"{quality['points']} points, {quality['dof']:.0f} degrees of freedom; "
    line = quality['line']
    if line is not None and line < SAMPLES_PER_LINE:
        return (start + f"no {STRUCTURE_FREE} or {STRUCTURE_SG}: the natural line spans {line:.1f} channels, "
                        f"fewer than {SAMPLES_PER_LINE}")
    return start + f"{STRUCTURE_FREE} and {STRUCTURE_SG} use {100 * quality['kept']:.0f} % of them, away from the lines"


def fit_quality(spectra, chi2, deviance, n_free):
    """The Fit quality of a fit: one dict per spectrum.

    *spectra* holds, for every spectrum of the fit, (velocity, counts, model,
    baseline, fitted): arrays over all its points, *fitted* True where the fit
    used the point. *chi2* and *deviance* are the fit's Pearson chi-square and
    deviance per degree of freedom, *n_free* its free values. χ² and D are the
    fit's, the same for every spectrum of a simultaneous fit; the S rows and
    MISFIT are each spectrum's own, from its own residual and counts, with the
    degrees of freedom shared out by the points.

    Each dict: 'points', 'dof', 'line' (line_in_channels), 'kept' (the share
    of its fitted points the smooths use) and 'rows' -- (name, ideal, value,
    1σ, F, F's 1σ) in the tab's order, F and its 1σ None but in the S rows.
    None for every spectrum
    when the model is not positive at a fitted point (not counts), and for a
    spectrum with too few fitted neighbours to measure its noise.
    """
    arrays = []
    for v, y, m, baseline, fitted in spectra:
        fitted = np.asarray(fitted, dtype=bool)
        arrays.append((np.asarray(v, dtype=float), np.asarray(y, dtype=float), np.asarray(m, dtype=float),
                       np.asarray(baseline, dtype=float), fitted))
    counts = [int(a[4].sum()) for a in arrays]
    n_total = sum(counts)
    dof_total = n_total - n_free
    models = np.concatenate([m[fitted] for _, _, m, _, fitted in arrays])
    if dof_total <= 0 or not np.all(np.isfinite(models) & (models > 0)):
        return [None] * len(arrays)
    mean, variance = deviance_moments(models)
    fit_rows = [(CHI2, 1.0, chi2, np.sqrt(np.sum(2 + 1 / models)) / dof_total, None, None),
                (DEVIANCE, 1 + np.sum(mean - 1) / dof_total, deviance, np.sqrt(np.sum(variance)) / dof_total,
                 None, None)]
    qualities = []
    for (v, y, m, baseline, fitted), n in zip(arrays, counts):
        if n < 3:
            qualities.append(None)
            continue
        quality = _spectrum_rows(v, y, m, baseline, fitted, n - n_free * n / n_total)
        if quality is not None:
            quality['rows'] = fit_rows + quality['rows']
        qualities.append(quality)
    return qualities
