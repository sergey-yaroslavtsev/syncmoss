"""The line set of a 'Spectrum' row: a measured reference fitted with lines.

The reference (.dat) is fitted through its own transmission integral -- its
instrumental function, as any spectrum of it -- with the baseline and a few
Singlets of the natural L, each with its own T_q, position δ_q (free within the
span) and Gaussian G_q in [0, G_MAX]; minimi_hi with Poisson weights, as every
count fit. The baseline is the method's: a CMS reference has non-resonant counts,
Nnr a fitted fraction of Ns within NNR_FRACTION (from CMS_NNR_RATIO), an SMS one
none (Onr, c²nr, linnr follow Os, c²s, lins), so T_q is an absorber's whichever
method the row is used in. How the lines are found (docs/spectrum_as_model.md
§33):

1. the baseline from the edges (Ns, lins; c²s = 0 and fitted, Os the middle of
   the range, fixed);
2. the span: where the absorption, averaged over two natural widths, exceeds
   SPAN_SIGMAS of its noise, widened by two natural widths;
3. one line per maximum of that averaged absorption;
4. after every fit: a line whose T_q is below its 1σ, or whose position sits at
   the span's end, has vanished and is removed; a G_q at 0 or G_MAX is moved
   G_STEP_OFF off it, once per line and limit, and the lines refitted;
5. growth, while the Fit quality says the lines do not describe the reference
   (quality_marks: S (no model) or MISFIT more than QUALITY_SIGMAS from its
   ideal): the score test's line is added and the lines refitted; it is kept
   when the deviance drops by GAIN·F (F the counts' own noise factor, the Fit
   quality's 'S (no model)'), and the first that does not stops the growth.

With a number of lines set by the user (with_number), lines are added by the
score test whatever their gain, or the least significant removed, until the
number is reached; a vanished line is placed anew instead of removed.

Everything is computed through LinesModel: TI's node sum for the reference's
velocities, prepared once, in which a line moved is the only one computed again.
"""
import numpy as np

import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.constants import CMS_NNR_RATIO, NAT_WIDTH, number_of_baseline_parameters as NB
from syncmoss.fit_quality import MISFIT, STRUCTURE_FREE, counts_noise, fit_quality
from syncmoss.spectrum_io import estimate_edge_background, ns_for_background

# A line set describes the reference when its S (no model) and MISFIT are
# within this many of their 1σ of the ideal (quality_marks): the growth stops
# there, and the window marks the rows beyond it
QUALITY_SIGMAS = 2.0
_JUDGED = (STRUCTURE_FREE, MISFIT)
G_MAX = 3.0            # mm/s, the widest Gaussian of a line
G_LIMIT = 1e-3         # mm/s: a G this close to 0 or G_MAX sits at the limit ...
G_STEP_OFF = 0.1       # ... and is moved this far off it, once per line and limit
# The deviance a new line (T, δ, G) must gain: the 99 % point of χ² with 3
# degrees of freedom (a likelihood-ratio test)
GAIN = 11.34
SPAN_SIGMAS = 5.0
# The widths the score test tries for a new line, mm/s
CANDIDATE_G = (0.0, 0.15, 0.4, 1.0, 2.0, 3.0)
# A candidate's column is the change of the counts by a line this thin (its
# optical depth at most PROBE_T): the derivative to that relative accuracy
PROBE_T = 1e-3
# minimi_hi's schedule, the fit's (fitting_io)
MI, MI2, TAU0, EPS = 20, 10, 1e-3, 1e-6
# The non-resonant counts of a CMS reference, Nnr/Ns, are fitted within these
# bounds (the user's, against fits that run away: a thin reference cannot tell
# them from T, a thick one can -- MossLab 036 has 0.41)
NNR_FRACTION = (0.1, 1.2)

_PER_LINE = 4          # a Singlet's slots: T, δ, L, G
_NNR = 4               # Nnr's slot


def quality_marks(quality):
    """The rows of a Fit quality (fit_quality's dict) that say the lines do not
    describe the reference: S (no model) or MISFIT further than QUALITY_SIGMAS
    of its 1σ from its ideal. None when neither could be measured."""
    rows = [] if quality is None else [row for row in quality['rows'] if row[0] in _JUDGED]
    if not rows:
        return None
    return [name for name, ideal, value, sigma, *_ in rows if abs(value - ideal) > QUALITY_SIGMAS * sigma]


def nnr_start(method):
    """The Nnr/Ns a reference measured with *method* ('CMS' or 'SMS') starts
    with: CMS the program's (and fitted within NNR_FRACTION), SMS 0."""
    return CMS_NNR_RATIO if method == 'CMS' else 0.0


def flat_parameters(base, lines):
    """p of the baseline *base* and the (n, 3) *lines* (T_q, δ_q, G_q)."""
    p = list(base)
    for T, d, G in np.reshape(lines, (-1, 3)):
        p += [T, d, NAT_WIDTH, G]
    return np.array(p, dtype=float)


def lines_of(p):
    """The (n, 3) lines T_q, δ_q, G_q of the flat parameters p."""
    q = np.reshape(np.asarray(p[NB:], dtype=float), (-1, _PER_LINE))
    return q[:, [0, 1, 3]].copy()


def with_links(q):
    """The flat parameters of the fitter's own vector *q* -- the flat ones and
    Nnr/Ns after them (fraction_vector): Nnr = that fraction times Ns, and the
    non-resonant part's shape following the resonant one (Onr, c²nr, linnr =
    Os, c²s, lins)."""
    p = np.array(q[:-1], dtype=float)
    p[_NNR] = q[-1] * p[0]
    p[5:NB] = p[1:4]
    return p


def fraction_vector(p):
    """The fitter's own vector of the flat parameters *p*: Nnr/Ns appended, the
    LAST value, so that minimi_hi reports it undetermined (and not a line's T)
    when a thin reference cannot tell it from the lines' T."""
    p = np.asarray(p, dtype=float)
    return np.append(p, p[_NNR] / p[0] if p[0] else 0.0)


def edge_baseline(x, y, fraction):
    """The baseline from the edges, Nnr = *fraction*·Ns: Ns + Nnr the
    off-resonance counts as every spectrum starts it (estimate_edge_background,
    ns_for_background), lins from the difference of the two ends, c²s = 0, Os
    the middle of the range (x ascending)."""
    background = estimate_edge_background(y)
    yl, yr = float(np.mean(y[:5])), float(np.mean(y[-5:]))
    xl, xr = float(np.mean(x[:5])), float(np.mean(x[-5:]))
    base = np.zeros(NB)
    base[0] = ns_for_background(background, base, [_NNR], [0], [fraction])
    base[1] = 0.5 * (x[0] + x[-1])
    base[3] = 1e2 * (yr - yl) / (background * (xr - xl))
    return with_links(np.append(base, fraction))


def averaged_absorption(x, y, baseline):
    """The absorption 1 - y/baseline averaged over two natural widths, and the
    threshold SPAN_SIGMAS of its noise."""
    k = max(3, int(round(2 * NAT_WIDTH / np.median(np.abs(np.diff(x))))))
    a = np.convolve(1 - y / baseline, np.ones(k) / k, mode='same')
    return a, SPAN_SIGMAS / np.sqrt(np.mean(baseline) * k)


def span_of(x, a, threshold):
    """(low, high, reaches_an_end): from the first to the last significant
    point, widened by two natural widths; reaches_an_end when that touches an
    end of the spectrum (its edges are not baseline)."""
    significant = np.flatnonzero(a > threshold)
    if len(significant) == 0:
        return x[0], x[-1], False
    low, high = x[significant[0]] - 2 * NAT_WIDTH, x[significant[-1]] + 2 * NAT_WIDTH
    return max(low, x[0]), min(high, x[-1]), bool(low <= x[0] or high >= x[-1])


def prominent_maxima(a, threshold):
    """The maxima of *a* that stand out by *threshold*: above it, and above the
    higher of the valleys that part them from a higher point on each side (or
    from the end) by as much."""
    n = len(a)
    peaks = []
    for j in range(1, n - 1):
        if not (a[j] > threshold and a[j] >= a[j - 1] and a[j] > a[j + 1]):
            continue
        valleys = []
        for step in (-1, 1):
            i, low = j, a[j]
            while 0 <= i + step < n and a[i + step] <= a[j]:
                i += step
                low = min(low, a[i])
            valleys.append(low)
        if a[j] - max(valleys) > threshold:
            peaks.append(j)
    return peaks


def start_lines(x, a, threshold, fraction):
    """One line per prominent maximum of the averaged absorption *a*: T_q from
    the area of the resonant optical depth over its basin (valley to valley; the
    resonant counts are 1/(1 + fraction) of the off-resonance ones, *fraction*
    Nnr/Ns), G_q from its width beyond twice the natural one."""
    tau = -np.log(np.clip(1 - a * (1 + fraction), 1e-6, None))
    n = len(x)
    peaks = prominent_maxima(a, threshold)
    lines = []
    for k, j in enumerate(peaks):
        half = tau[j] / 2
        left, right = j, j
        while left > 0 and tau[left] > half:
            left -= 1
        while right < n - 1 and tau[right] > half:
            right += 1
        lo = (int(np.argmin(tau[peaks[k - 1]:j + 1])) + peaks[k - 1] if k > 0
              else max(j - 3 * (j - left), 0))
        hi = (int(np.argmin(tau[j:peaks[k + 1] + 1])) + j if k + 1 < len(peaks)
              else min(j + 3 * (right - j), n - 1))
        area = float(np.trapezoid(np.clip(tau[lo:hi + 1], 0, None), x[lo:hi + 1]))
        G = min(max(x[right] - x[left] - 2 * NAT_WIDTH, G_STEP_OFF), G_MAX - G_STEP_OFF)
        lines.append([area / (np.pi * NAT_WIDTH / 2), x[j], G])
    return np.array(lines, dtype=float).reshape(-1, 3)


def fitted_points(x, y):
    """The points of a reference that are fitted, velocities ascending: those
    with counts (a spectrum saved without excluded points has nan). Its source
    is prepared for exactly these velocities."""
    order = np.argsort(x, kind='stable')
    x, y = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    counted = np.isfinite(y)
    return x[counted], y[counted]


class LinesModel:
    """The reference's counts for a baseline and lines (flat p), through TI's
    node sum of its velocities x prepared once (models.node_parts, with the
    source *norm* and *smearing* source_for prepared): a line's transmission at
    the nodes is TImod's, so a line moved is the only one computed again.
    Interrupt (models.FIT_CANCEL) stops every evaluation."""

    def __init__(self, x, JN, method, norm, smearing, pol):
        self.x, self.JN, self.method, self.norm, self.pol = np.asarray(x, dtype=float), JN, method, norm, pol
        self.E, self.W = m5.node_parts(self.x, JN, method['x0'], method['MulCo'], method['INS'], method['Met'],
                                       pol, smearing)

    def transmission(self, line):
        """A line's (T, δ, G) transmission at every node and velocity."""
        if m5.FIT_CANCEL.is_set():
            raise m5.FitInterrupted()
        mp = self.method
        return np.asarray(m5.TImod(self.x, np.array([line[0], line[1], NAT_WIDTH, line[2]]), ['Singlet'],
                                   self.E.ravel(), mp['x0'], mp['MulCo'], mp['INS'], [0], [0], -1),
                          dtype=float).reshape(self.W.shape)

    def spectrum(self, p, transmission):
        """The counts for the baseline p[:NB] and the lines' joint transmission."""
        mp = self.method
        H = np.sum(self.W * transmission, axis=0)
        return np.asarray(m5.spectrum_of_node_sum(self.x, np.asarray(p[:NB], dtype=float), H, self.JN, mp['x0'],
                                                  mp['MulCo'], mp['INS'], mp['Met'], self.norm, self.pol),
                          dtype=float)

    def transmissions(self, p):
        return [self.transmission(line) for line in lines_of(p)]

    def counts(self, p):
        return self.spectrum(p, _product(self.transmissions(p), self.W.shape))

    def counts_of_many(self, ps):
        """The counts of each of *ps*: minimi_hi's Jacobian vectors (batch_eval),
        each the fit's point with one parameter moved (and what is linked to
        it). The point they share is where, at every slot, two of the first
        three agree (a slot is moved in one vector at most); its lines are
        computed once, a moved line once per vector."""
        ps = [np.asarray(p, dtype=float) for p in ps]
        if len(ps) < 3:
            return [self.counts(p) for p in ps]
        shared = np.where(ps[0] == ps[1], ps[0], ps[2])
        lines = lines_of(shared)
        base = self.transmissions(shared)
        out = []
        for p in ps:
            moved = np.flatnonzero(np.any(lines_of(p) != lines, axis=1))
            parts = list(base)
            for q in moved:
                parts[q] = self.transmission(lines_of(p)[q])
            out.append(self.spectrum(p, _product(parts, self.W.shape)))
        return out


def _product(parts, shape):
    joint = np.ones(shape)
    for part in parts:
        joint = joint * part
    return joint


class LineFit:
    """One fit of the reference: its flat parameters, the counts, the deviance
    (a sum) and Pearson's reduced χ²; *errors* and *covariance* are minimi_hi's,
    of the fitter's own vector (fraction_vector)."""

    def __init__(self, p, errors, covariance, model, deviance, chi2):
        self.p = np.asarray(p, dtype=float)
        self.errors = np.asarray(errors, dtype=float)
        self.covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        self.model = np.asarray(model, dtype=float)
        self.deviance = float(deviance)
        self.chi2 = float(chi2)

    @property
    def fraction(self):
        """Nnr/Ns."""
        return fraction_vector(self.p)[-1]

    @property
    def lines(self):
        return lines_of(self.p)

    def significance(self):
        """T_q over its 1σ, line by line, with Nnr/Ns held where it is: it
        scales every line alike -- a thin reference cannot tell it from their
        T -- but cannot stand in for one line. 0 where the 1σ is not known (a
        T_q at its bound has no error)."""
        kept = np.flatnonzero(np.isfinite(self.errors))     # the covariance's rows, in order
        cov = self.covariance
        last = len(self.errors) - 1
        if last in kept:
            j = int(np.searchsorted(kept, last))
            if cov[j, j] > 0:
                cov = cov - np.outer(cov[:, j], cov[j, :]) / cov[j, j]
        T = self.lines[:, 0]
        sd = np.full(len(T), np.nan)
        for q in range(len(T)):
            k = NB + _PER_LINE * q
            if k in kept:
                r = int(np.searchsorted(kept, k))
                sd[q] = np.sqrt(max(cov[r, r], 0.0))
        return np.where(np.isfinite(sd) & (sd > 0), T / np.where(sd > 0, sd, 1.0), 0.0)


class ReferenceFitter:
    """Fits the reference (x, y) -- fitted_points' -- measured with *method*
    ('CMS' or 'SMS') with lines through *model* (a LinesModel of these x): a
    CMS reference's Nnr/Ns fitted within NNR_FRACTION, an SMS one's 0.
    *report(text)* hears what it does (the window's status).

    minimi_hi fits the fitter's own vector, the flat parameters with Nnr/Ns
    after them (fraction_vector; with_links makes the flat ones of it), so the
    fraction has plain bounds."""

    def __init__(self, x, y, model, method, report=None):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.model = model
        self.cms = method == 'CMS'
        self.report = report if report is not None else (lambda text: None)
        self.base0 = edge_baseline(self.x, self.y, nnr_start(method))
        self.baseline = self.model.counts(self.base0)            # the start's, for the span
        self.a, self.threshold = averaged_absorption(self.x, self.y, self.baseline)
        low, high, self.edges_absorb = span_of(self.x, self.a, self.threshold)
        self.span = (float(low), float(high))
        # what the Fit quality takes from the counts alone, the same for every fit
        self.noise = counts_noise(self.x, self.y, np.ones(len(self.x), bool))
        self.F = 1.0                                             # set by the first fit
        self.fits = 0
        self.removed = []            # where step 4 removed lines since the last growth step

    # --- one fit ----------------------------------------------------------------

    def free_indices(self, n):
        """In the fitter's own vector of *n* lines: Ns, c²s, lins, every line's
        T, δ, G and a CMS reference's Nnr/Ns (the last)."""
        return ([0, 2, 3] + [NB + _PER_LINE * q + k for q in range(n) for k in (0, 1, 3)]
                + ([NB + _PER_LINE * n] if self.cms else []))

    def bounds(self, n):
        """Of the fitter's own vector of *n* lines."""
        low = np.full(NB + _PER_LINE * n + 1, -np.inf)
        high = np.full(NB + _PER_LINE * n + 1, np.inf)
        low[0] = 0.0
        for q in range(n):
            i = NB + _PER_LINE * q
            low[i] = 0.0
            low[i + 1], high[i + 1] = self.span
            low[i + 3], high[i + 3] = 0.0, G_MAX
        if self.cms:
            low[-1], high[-1] = NNR_FRACTION
        return np.array([low, high])

    def fit(self, base, lines):
        """The lines refitted from *lines* with the baseline from *base* (flat)."""
        lines = np.reshape(lines, (-1, 3)).copy()
        lines[:, 1] = np.clip(lines[:, 1], *self.span)
        lines[:, 2] = np.clip(lines[:, 2], 0.0, G_MAX)
        n = len(lines)
        q0 = fraction_vector(flat_parameters(base, lines))
        q0[-1] = np.clip(q0[-1], *NNR_FRACTION) if self.cms else 0.0
        fix = np.array(sorted(set(range(len(q0))) - set(self.free_indices(n))), dtype=int)
        q, errors, _, covariance, chi2 = mi.minimi_hi(
            lambda _x, v: self.model.counts(with_links(v)), self.x, self.y, q0, fix=fix,
            bounds=self.bounds(n), MI=MI, MI2=MI2, tau0=TAU0, eps=EPS,
            batch_eval=lambda vs: self.model.counts_of_many([with_links(v) for v in vs]), weights='P')
        self.fits += 1
        p = with_links(q)
        model = self.model.counts(p)
        return LineFit(p, errors, covariance, model, mi.poisson_deviance(self.y, model), chi2)

    def quality(self, fit):
        """The Fit quality of *fit* (fit_quality's dict for one spectrum)."""
        n_free = len(self.free_indices(len(fit.lines)))
        dof = len(self.x) - n_free
        baseline = self.model.counts(fit.p[:NB])
        return fit_quality([(self.x, self.y, fit.model, baseline, np.ones(len(self.x), bool))],
                           fit.chi2, fit.deviance / dof, n_free, noises=[self.noise])[0]

    def noise_factor(self, fit):
        """F of the 'S (no model)' row: the counts' noise against Poisson's (1
        when it cannot be measured)."""
        quality = self.quality(fit)
        rows = {} if quality is None else {row[0]: row for row in quality['rows']}
        F = rows.get(STRUCTURE_FREE, (None,) * 5)[4]
        return 1.0 if F is None or not np.isfinite(F) else float(F)

    # --- the score test -------------------------------------------------------

    def _weighted_jacobian(self, fit, w):
        """The weighted Jacobian of the fit's free values, in its own vector."""
        q0 = fraction_vector(fit.p)
        bumped, steps = [], []
        for i in self.free_indices(len(fit.lines)):
            q = q0.copy()
            steps.append(1e-6 * max(abs(q[i]), 1.0))
            q[i] += steps[-1]
            bumped.append(with_links(q))
        return np.array([(m - fit.model) / h * w
                         for m, h in zip(self.model.counts_of_many(bumped), steps)]).T

    def best_candidate(self, fit, exclude=()):
        """The line the score test finds most useful: for every width of
        CANDIDATE_G and position where the absorption is significant (the span
        without its widening: a line offered at its ends is pushed onto them
        and vanishes), the deviance its best positive T would gain with the
        fit's other parameters re-adjusting linearly (its column projected out
        of the fit's Jacobian); a coarse grid, then a quarter of a natural width
        about the best. (gain, [T, δ, G]) or (0, None)."""
        inner = (self.span[0] + 2 * NAT_WIDTH, self.span[1] - 2 * NAT_WIDTH)
        if inner[0] >= inner[1]:
            inner = self.span
        w = 1 / np.sqrt(fit.model)
        r = (self.y - fit.model) * w
        Q, _ = np.linalg.qr(self._weighted_jacobian(fit, w))
        r = r - Q @ (Q.T @ r)
        joint = _product(self.model.transmissions(fit.p), self.model.W.shape)

        def scores(positions, G):
            positions = np.array([xi for xi in positions
                                  if all(abs(xi - gone) > NAT_WIDTH for gone in exclude)])
            if len(positions) == 0:
                return None
            cols = np.array([(self.model.spectrum(fit.p, joint * self.model.transmission([PROBE_T, xi, G]))
                              - fit.model) / PROBE_T * w for xi in positions]).T
            cols = cols - Q @ (Q.T @ cols)
            num, den = cols.T @ r, np.sum(cols ** 2, axis=0)
            amp = num / np.where(den > 0, den, np.inf)
            score = np.where(amp > 0, num * amp, 0.0)
            k = int(np.argmax(score))
            return score[k], positions[k], amp[k]

        best = (0.0, None)
        coarse_step = {}
        for G in CANDIDATE_G:
            coarse_step[G] = max(NAT_WIDTH, G / 4)
            found = scores(np.arange(inner[0], inner[1] + 1e-9, coarse_step[G]), G)
            if found is not None and found[0] > best[0]:
                best = (found[0], [found[2], found[1], G])
        if best[1] is None:
            return best
        _, (T, xi, G) = best
        step = coarse_step[G]
        fine = np.clip(xi + np.arange(-step, step + 1e-9, NAT_WIDTH / 4), *inner)
        found = scores(fine, G)
        if found is not None and found[0] > best[0]:
            best = (found[0], [found[2], found[1], G])
        return best

    # --- the rules after every fit (4) ------------------------------------------

    def vanished(self, fit):
        """The lines that vanished: T_q below its 1σ, or at the span's end."""
        lines = fit.lines
        at_end = abs(np.median(np.diff(self.x))) / 10
        return ((fit.significance() < 1.0) | (lines[:, 1] - self.span[0] < at_end)
                | (self.span[1] - lines[:, 1] < at_end))

    def clean(self, fit, nudged, keep_number=False):
        """Step 4, repeated until nothing changes. *nudged* holds (place, limit)
        of every G already moved. With *keep_number* the lines that vanish
        first are placed anew where the score test says instead of being
        removed -- not where a line has vanished; lines placed anew that vanish
        too are removed (the data do not take that many)."""
        gone_here = []
        for _ in range(4 * max(len(fit.lines), 1) + 4):
            lines = fit.lines
            if len(lines) == 0:
                return fit
            gone = self.vanished(fit)
            if gone.any():
                self.removed += list(lines[gone, 1])
                gone_here += list(lines[gone, 1])
                kept = lines[~gone]
                self.report(f"{int(gone.sum())} line(s) vanished at "
                            f"{', '.join(f'{d:.2f}' for d in lines[gone, 1])} mm/s")
                fit = self.fit(fit.p[:NB], kept)
                if keep_number:
                    for _ in range(int(gone.sum())):
                        fit = self.add_line(fit, exclude=gone_here)
                    keep_number = False
                continue
            moved = lines.copy()
            for q, (_, d, G) in enumerate(lines):
                for limit, off in ((0.0, G_STEP_OFF), (G_MAX, G_MAX - G_STEP_OFF)):
                    key = (round(d / (NAT_WIDTH / 2)), limit)
                    if abs(G - limit) < G_LIMIT and key not in nudged:
                        nudged.add(key)
                        moved[q, 2] = off
            if np.array_equal(moved, lines):
                return fit
            self.report(f"G moved off its limit for {int(np.sum(moved[:, 2] != lines[:, 2]))} line(s)")
            fit = self.fit(fit.p[:NB], moved)
        return fit

    # --- the line set ---------------------------------------------------------------

    def add_line(self, fit, exclude=()):
        """*fit* refitted with the score test's line added (itself when there is none)."""
        _, line = self.best_candidate(fit, exclude)
        if line is None:
            return fit
        line[2] = min(max(line[2], G_LIMIT), G_MAX - G_LIMIT)
        return self.fit(fit.p[:NB], np.vstack([fit.lines, line]))

    def automatic(self, max_lines=60):
        """Steps 3-5: the line set the data ask for."""
        lines = start_lines(self.x, self.a, self.threshold, fraction_vector(self.base0)[-1])
        lines = lines[(lines[:, 1] >= self.span[0]) & (lines[:, 1] <= self.span[1])]
        self.report(f"{len(lines)} line(s) to start, span {self.span[0]:.2f} .. {self.span[1]:.2f} mm/s")
        nudged = set()
        self.removed = []
        fit = self.clean(self.fit(self.base0, lines), nudged)
        self.F = self.noise_factor(fit)
        while len(fit.lines) < max_lines:
            if quality_marks(self.quality(fit)) == []:
                break                                        # the lines describe the reference
            # a line step 4 has just removed is not offered again at its place
            gain, line = self.best_candidate(fit, exclude=self.removed)
            if line is None or gain < GAIN * self.F / 2:     # the score understates the gain, not by half
                break
            line[2] = min(max(line[2], G_LIMIT), G_MAX - G_LIMIT)
            self.removed = []
            new = self.clean(self.fit(fit.p[:NB], np.vstack([fit.lines, line])), nudged)
            self.report(f"{len(new.lines)} lines: deviance {fit.deviance:.1f} -> {new.deviance:.1f}")
            if fit.deviance - new.deviance < GAIN * self.F:
                break
            fit = new
        return fit

    def with_number(self, fit, n):
        """*fit* brought to *n* lines: the score test's lines added whatever
        their gain, or the least significant removed, one at a time; vanished
        lines are placed anew (clean). Fewer when the data do not take *n*: a
        line added that vanishes again ends it."""
        n = max(1, int(n))
        nudged = set()
        fit = self.clean(fit, nudged, keep_number=True)
        while len(fit.lines) != n:
            before = len(fit.lines)
            if before < n:
                fit = self.add_line(fit)
            else:
                weakest = int(np.argmin(fit.significance()))
                fit = self.fit(fit.p[:NB], np.delete(fit.lines, weakest, axis=0))
            fit = self.clean(fit, nudged, keep_number=True)
            self.report(f"{len(fit.lines)} lines: deviance {fit.deviance:.1f}")
            if len(fit.lines) == before:
                break                                  # the data take no other number
        return fit
