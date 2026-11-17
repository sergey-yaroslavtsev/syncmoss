"""Components of many spectra, lines first (docs/instead_of_pca.md).

Many spectra of one sample on one velocity axis -- an operando series, a map
-- described by K components. A component is a set of lines (Singlets of the
natural width, each with its own place δ_q and Gaussian G_q) with fixed
relative strengths β_kq, and spectrum s holds component k with the weight
w_sk:

    τ_s(E) = Σ_k w_sk Σ_q β_kq ℓ_q(E),    w ≥ 0, β ≥ 0, Σ_q β_kq = 1,

ℓ_q the optical depth of line q per unit T at TI's nodes, so that w_sk is the
T of component k in spectrum s. No physical model is assumed: recognising a
sextet, a doublet or a relaxation spectrum in a component is the user's step.

What run does (§6 of the note):
1. the summed spectrum's lines: the 'Spectrum' row's own fit
   (spectrum_reference.ReferenceFitter), close lines merged (merge_close);
   its baseline's curvature where every spectrum's own starts
   (Axis.curvature);
2. missing lines by misfit (§2.10, search_lines), round after round: every
   spectrum fitted with every line's own T, the groups of equal composition,
   in every group the sum of its misfitting spectra and the sum of the others,
   both judged by the 'Spectrum' row's own fit;
3. many spectra: every count to one of two halves, the groups formed on the
   first, their sums of the second fitted (§2.9, fitted_rows); else the spectra;
4. every fitted row with every line's own T, the components from all the lines
   at once (§2.12): the anchors, the NMF of the lines' T; a line with
   significant coefficients in several components copied into each;
5. the loop: the components fitted (line places free), merged (proportional
   weights, one pair at a time), a component that is a mixture of others
   dissolved into them, components recombined so that they hold fewer lines
   with the same counts (recombine), the extra copies of a line pruned (only
   splittings: never a line, never a component), the lines at an end of the
   range removed, then the missing components searched for on every spectrum
   (§2.11, afterwards); anything added, again from the fit. The tests of the
   fit's own choices are scaled by how much more than Poisson the counts
   scatter about it (misfit_factor);
6. the joint fit; every spectrum's weights and baseline on all its counts.

Every fit of counts is minimi_hi's with Poisson weights and exact derivatives
(a line's depth with its derivatives in its place and width from dual_kernels'
twin of the Voigt, everything else in closed form); every small linear fit
minimi_hi's too; the fits of single spectra run on the program's pool, a
group of spectra per process. Every test is at the program's 99 % level
(GAIN). Interrupt (models.FIT_CANCEL) stops it between two steps. It starts
with a warm-up (warm_up) where one is needed -- the first search of a
session: the compiled kernels loaded in every process, so that no step's
time holds their loading (or their compilation, the first time ever).
"""
import time
from statistics import NormalDist

import numpy as np

import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.constants import NAT_WIDTH, number_of_baseline_parameters as NB
import syncmoss.faddeeva as faddeeva
from syncmoss.dual_kernels import _faddeeva_parts, _voigt_parts, voigt_value
from syncmoss.fit_quality import deviance_moments
from syncmoss.instrumental_io import source_for
from syncmoss.spectrum_io import estimate_edge_background, ns_for_background
from syncmoss.spectrum_reference import (EPS, G_MAX, GAIN, MI, MI2, TAU0, LinesModel, ReferenceFitter, at_an_end,
                                         singlet_depth, singlet_parts)

TOL = 0.5                 # an alternation (and the NMF) stops when its deviance (χ²) gains less
MAX_ROUNDS = 60           # ... or after this many rounds
PROJECTED_MI2 = 1         # projected_fit: steps per Jacobian (its Jacobian costs less than a model evaluation)
MI_LINES = 2              # minimi_hi's outer iterations for the places and widths in one alternation round
                          # (a round needs a step, not the optimum: the alternation goes on, the joint fit ends it)
SEED = 1e-6              # a strength or weight at 0 starts a fit of squares at this fraction of the largest
START_WEIGHT = 2.0        # every component's T in every spectrum where a fit starts from nothing
SEARCH_ROUNDS = 8         # rounds of the search by misfit, at most
AFTER_ROUNDS = 8          # rounds of the search for missing components, at most
MAX_LOOPS = 6             # rounds of the loop (step 5), at most
PLACES = 10               # places a round of the search for missing components tries
CANDIDATE_G = (0.0, 0.15, 0.4)    # mm/s, the widths the matched filter tries
RESPONSE_FLOOR = 1e-4     # a line's response is kept down to this fraction of its largest value
HALVES_SEED = 23          # the split of the counts into halves for the rows (fitted_rows) ...
AFTER_SEED = 18           # ... and for the search for missing components
# A joint fit whose Jacobian (minimi_hi holds it twice) would take more memory
# than this, in bytes, is made by the alternation instead (the note's 4 GB free)
JOINT_FIT_MEMORY = 2e9


# --- levels ---------------------------------------------------------------------

def normal_point():
    """The one-sided normal point of the program's 99 % level: from GAIN (the
    99 % point of χ² with 3 degrees of freedom) through Wilson-Hilferty, so
    that every test here is at that level (2.32)."""
    return ((GAIN / 3) ** (1 / 3) - 1 + 2 / 27) / np.sqrt(2 / 27)


def chi2_point(dof, z):
    """The point of χ² with *dof* degrees of freedom that the normal point z
    marks (Wilson-Hilferty)."""
    dof = max(dof, 1)
    return dof * (1 - 2 / (9 * dof) + z * np.sqrt(2 / (9 * dof))) ** 3


def chi2_level(dof):
    """The 99 % point of χ² with *dof* degrees of freedom."""
    return chi2_point(dof, normal_point())


def shared_point(m):
    """The normal point of the 99 % level shared among m tests (Bonferroni)."""
    nd = NormalDist()
    return nd.inv_cdf(1 - (1 - nd.cdf(normal_point())) / max(m, 1))


def _check_interrupt():
    if m5.FIT_CANCEL.is_set():
        raise m5.FitInterrupted()


# --- the warm-up -------------------------------------------------------------------

def _ready():
    """Whether numba has the search's compiled kernels in this process
    already: a line's depth and its depth with the derivatives (dual_kernels'
    twins of the program's line shape, models.Voigt_r -- the exact Voigt's or
    the pseudo-Voigt's, models.exact_voigt), the line shape itself
    (line_fwhm), minimi_hi's linear solve."""
    if m5.exact_voigt():
        kernels = (faddeeva.voigt_line, _faddeeva_parts)
    else:
        kernels = (voigt_value, _voigt_parts, m5.Voigt)
    return all(kernel.signatures for kernel in kernels + (mi.solution,))


def _warm_up():
    """Those kernels run once in this process, unless it has them (_ready):
    what a process pays for them the first time -- numba compiling them, or
    loading them from its cache on disk -- is paid here. Whether it had to."""
    if _ready():
        return False
    E = np.linspace(-1.0, 1.0, 8)
    singlet_depth(E.reshape(2, 4), 1.0, 1.0, 0.0, 0.1)
    singlet_parts(E.reshape(2, 4), 1.0, 0.0, 0.1)
    m5.Voigt_r(NAT_WIDTH, 0.1, E)
    mi.warmup()
    return True


def warm_up(pool, report=None):
    """_warm_up in this process first -- numba compiles once and writes its
    cache -- then in the processes of the pool, a task each, which load it.
    Where it is done already (an earlier search in this session) nothing is
    run: the pool is only asked (a few ms). Whether anything had to be done."""
    if not _ready():
        _say(report, "Warm-up...")
    here = _warm_up()
    return any(m5._pool_starmap(pool, _warm_up, [()] * (getattr(pool, '_processes', None) or 1))) or here


def _say(report, text):
    if report is not None:
        report(text)


# --- the spectra's velocities and source ---------------------------------------

def nonresonant(method, k=None, f_lm=None, c_high=0.0, c_static=0.0):
    """(r, C_high, C_static) of a spectrum's non-resonant counts, Nnr = r Ns +
    C_high with the baseline's shape and C_static added flat. SMS: none. CMS,
    as the user says: a coefficient, Nnr = k Ns; or the source's form of the
    Δθ dialogs (instrumental_io.dtheta_alpha_fe_reference), Nnr = Ns (1 −
    f_LM)/f_LM + C_high, C_static the detector's static noise."""
    if method == 'SMS':
        return 0.0, 0.0, 0.0
    if f_lm is None:
        if not k >= 0:
            raise ValueError("Nnr/Ns must not be negative")
        return float(k), 0.0, 0.0
    if not 0 < f_lm < 1:
        raise ValueError("f_LM must be between 0 and 1")
    if c_high < 0 or c_static < 0:
        raise ValueError("C_high and the static noise must not be negative")
    return (1 - f_lm) / f_lm, float(c_high), float(c_static)


class Axis:
    """The spectra's velocities x and their source, prepared once: TI's node
    sum for x (spectrum_reference.LinesModel, its source from source_for), the
    known part -- rows every spectrum holds besides the sample, Be or KB_nano
    -- folded into the node weights W (every line is scalar, so it multiplies
    their joint transmission), and the counts of a node sum H with a
    spectrum's baseline (Ns, lins, c²s; Os the middle of the range):

        μ = g (Ns (R H + F + r) + C_high) + C_static,
        g = 1 + lins (v − Os)/10² + c²s (v − Os)²/10⁴,

    R and F the source's normalisation (CMS: D and 1 − Norm; SMS: D/Norm and 0),
    taken from models.spectrum_of_node_sum as are the terms of g, and
    (r, C_high, C_static) from nonresonant: Nnr = r Ns + C_high follows the
    baseline's shape (Onr, c²nr, linnr linked to Os, c²s, lins). Every
    spectrum has its own Ns, lins and c²s (a pixel's parabola can be more
    pronounced than another's); *curvature* = (lins, c²s) is where they start
    (run takes it from the summed spectrum's fit)."""

    def __init__(self, x, JN, method, norm, smearing, pol, known=(), nonres=(0.0, 0.0, 0.0)):
        self.model = LinesModel(x, JN, method, norm, smearing, pol)
        self.x, self.method, self.JN = self.model.x, method, JN
        self.curvature = (0.0, 0.0)
        if known:
            names = [name for name, _ in known]
            p = np.concatenate([np.asarray(q, dtype=float) for _, q in known])
            self.model.W = self.model.W * self.transmission(names, p)
        self.W = self.model.W
        self.ratio, self.c_high, self.c_static = (float(v) for v in nonres)
        nv = len(self.x)
        middle = 0.5 * (self.x[0] + self.x[-1])

        def level(ns, c2s, lins, nnr, H):
            p = np.array([ns, middle, c2s, lins, nnr, middle, c2s, lins], dtype=float)
            return np.asarray(m5.spectrum_of_node_sum(self.x, p, np.full(nv, float(H)), JN, method['x0'],
                                                      method['MulCo'], method['INS'], method['Met'], norm, pol),
                              dtype=float)
        self.flat = level(1, 0, 0, 0, 0)
        self.res = level(1, 0, 0, 0, 1) - self.flat
        self.lin = level(0, 0, 1, 1, 0) - 1
        self.quad = level(0, 1, 0, 1, 0) - 1

    def transmission(self, rows, p):
        """The transmission of *rows* (flat p, no baseline) at every node and
        velocity, as the source reads it out (CMS unpolarized, SMS polarized)."""
        mp = self.method
        return np.asarray(m5.TImod(self.x, np.asarray(p, dtype=float), list(rows), self.model.E.ravel(), mp['x0'],
                                   mp['MulCo'], mp['INS'], [0], [0], Met=-1, sms_pol=self.model.pol, Mett=mp['Met']),
                          dtype=float).reshape(self.model.W.shape)

    def line_depth(self, delta, G):
        """The optical depth of a line of T = 1 at delta with the Gaussian G at
        every node and velocity (spectrum_reference.singlet_depth)."""
        _check_interrupt()
        return singlet_depth(self.model.E, self.method['MulCo'], 1.0, delta, G)

    def line_parts(self, delta, G):
        """That line's depth ℓ with dℓ/dδ and dℓ/dG, exact
        (spectrum_reference.singlet_parts): (3, JN, NV)."""
        _check_interrupt()
        return singlet_parts(self.model.E, self.method['MulCo'], delta, G)

    def node_sum(self, tau):
        """H of the optical depths tau (..., JN, NV)."""
        return np.sum(self.W * np.exp(-tau), axis=-2)

    def shape(self, base):
        """g of the baselines base (..., 3)."""
        base = np.asarray(base, dtype=float)
        return 1 + base[..., 1:2] * self.lin + base[..., 2:3] * self.quad

    def counts(self, H, base):
        """μ of the node sums H (..., NV) with the baselines base (..., 3)."""
        base = np.asarray(base, dtype=float)
        return self.shape(base) * (base[..., 0:1] * (self.res * H + self.flat + self.ratio) + self.c_high) \
            + self.c_static

    def derivatives(self, shapes, T, base):
        """One spectrum with the optical depths T · shapes and the baseline
        base: its counts μ and their derivatives, one row per T_k, then Ns,
        lins and c²s."""
        wtr = self.W * np.exp(-np.tensordot(T, shapes, axes=1))
        inner = self.res * wtr.sum(axis=0) + self.flat + self.ratio
        g = self.shape(base)
        level = base[0] * inner + self.c_high
        rows = np.vstack([-base[0] * g * self.res * np.einsum('mj,kmj->kj', wtr, shapes),
                          g * inner, self.lin * level, self.quad * level])
        return g * level + self.c_static, rows

    def start_baselines(self, Y):
        """Baselines of the curvature whose off-resonance counts are each
        spectrum's edges' (estimate_edge_background, ns_for_background)."""
        out = np.zeros((len(Y), 3))
        out[:, 1:] = self.curvature
        for s, y in enumerate(Y):
            level = estimate_edge_background(y) - self.c_high - self.c_static
            out[s, 0] = ns_for_background(level, np.zeros(NB), [4], [0], [self.ratio])
        return out


def node_products(wtr, arrays):
    """Σ_m wtr[s, m, j] arrays[q, m, j] for every row s, array q and velocity
    j: (q, s, j), as matrix products over the nodes, one per velocity (BLAS)."""
    return np.matmul(wtr.transpose(2, 0, 1), np.asarray(arrays).transpose(2, 1, 0)).transpose(2, 1, 0)


def axis_for(pool, x, JN, method, pol, known=(), nonres=(0.0, 0.0, 0.0)):
    """The Axis of the velocities x, its source prepared for them (source_for)."""
    norm, smearing = source_for(pool, x, JN, method)
    return Axis(x, JN, method, norm, smearing, pol, known, nonres)


class Problem:
    """Rows of counts Y on one Axis and a line list (T_q, δ_q, G_q): every
    line's optical depth per unit T at the nodes, L (P, JN, NV)."""

    def __init__(self, axis, Y, lines):
        self.axis, self.Y = axis, np.asarray(Y, dtype=float)
        self.lines = np.asarray(lines, dtype=float).reshape(-1, 3).copy()
        self.L = np.array([axis.line_depth(d, G) for _, d, G in self.lines]).reshape(-1, *axis.W.shape)
        self.n, self.P = len(self.Y), len(self.lines)

    def counts(self, w, beta, base):
        _check_interrupt()
        return self.axis.counts(self.axis.node_sum(np.tensordot(w @ beta, self.L, axes=1)), base)

    def deviance(self, w, beta, base):
        return float(mi.poisson_deviance(self.Y.ravel(), self.counts(w, beta, base).ravel()))

    def start_baselines(self):
        return self.axis.start_baselines(self.Y)


# --- one spectrum's fits, over the pool ---------------------------------------------

def fit_one(axis, y, shapes, T0, base0, signed=False, T_free=True):
    """T_k of the optical depths *shapes* (per unit T) and the baseline (Ns,
    lins, c²s) fitted to the counts y of one spectrum; *signed*: the T_k free
    in sign (a line a spectrum lacks then scatters about 0), *T_free* False:
    held. Returns (T, baseline, minimi_hi's tuple)."""
    shapes = np.asarray(shapes, dtype=float)
    K = len(shapes)

    def model(q):
        return axis.counts(axis.node_sum(np.tensordot(q[:K], shapes, axes=1)), q[K:])

    def jacobian(q):
        return axis.derivatives(shapes, q[:K], q[K:])[1]
    q0 = np.concatenate([np.maximum(T0, 0.0) if T_free else T0, base0])
    lower = np.concatenate([np.full(K, -np.inf if signed else 0.0), [0.0, -np.inf, -np.inf]])
    fix = [] if T_free else list(range(K))
    out = mi.minimi_hi(lambda _x, v: model(v), np.arange(len(y)), y, q0, fix=np.array(fix, dtype=int),
                       bounds=np.array([lower, np.full(K + 3, np.inf)]), MI=MI, MI2=MI2, tau0=TAU0, eps=EPS,
                       weights='P', jacobian=jacobian)
    return out[0][:K], out[0][K:], out


def _fit_rows(axis, Y, shapes, T0, base0, signed, T_free):
    """One pool task of fit_rows: (T, baselines, errors of T (nan where
    minimi_hi gave none), deviances)."""
    K = len(shapes)
    T, base = np.zeros((len(Y), K)), np.zeros((len(Y), 3))
    S, dev = np.full((len(Y), K), np.nan), np.zeros(len(Y))
    for i, y in enumerate(Y):
        T[i], base[i], out = fit_one(axis, y, shapes, T0[i], base0[i], signed, T_free)
        if len(out[1]) == len(out[0]):
            S[i] = out[1][:K]
        dev[i] = float(mi.poisson_deviance(y, axis.counts(axis.node_sum(np.tensordot(T[i], shapes, axes=1)),
                                                            base[i])))
    return T, base, S, dev


def _chunks(pool, n):
    """The rows of n spectra in groups, one per process of the pool (a task
    carries the optical depths: sending it costs more than a row's fit)."""
    workers = getattr(pool, '_processes', None) or 1
    return [idx for idx in np.array_split(np.arange(n), min(n, workers)) if len(idx)]


def fit_rows(pool, axis, Y, shapes, T0, base0, signed=False, T_free=True):
    """fit_one for every row of Y with the same optical depths *shapes*, over
    the pool: (T, baselines, errors of T, deviances)."""
    Y, T0, base0 = np.asarray(Y, dtype=float), np.asarray(T0, dtype=float), np.asarray(base0, dtype=float)
    shapes = np.asarray(shapes, dtype=float)
    K = len(shapes)
    T, base = np.zeros((len(Y), K)), np.zeros((len(Y), 3))
    S, dev = np.full((len(Y), K), np.nan), np.zeros(len(Y))
    chunks = _chunks(pool, len(Y))
    results = m5._pool_starmap(pool, _fit_rows, [(axis, Y[idx], shapes, T0[idx], base0[idx], signed, T_free)
                                                 for idx in chunks])
    for idx, (t, b, s, d) in zip(chunks, results):
        T[idx], base[idx], S[idx], dev[idx] = t, b, s, d
    return T, base, S, dev


def line_fits(pool, pb, A0, base0):
    """Every row fitted with every line's own T, free in sign (places and
    widths held): (A (n, P), its errors, baselines, deviances)."""
    A, base, S, dev = fit_rows(pool, pb.axis, pb.Y, pb.L, A0, base0, signed=True)
    return A, S, base, dev


def weight_step(pool, pb, w, beta, base):
    """Every row's K weights and baseline, the line sets beta held."""
    w, base, _, _ = fit_rows(pool, pb.axis, pb.Y, np.tensordot(beta, pb.L, axes=1), w, base)
    return w, base


# --- the component fits ------------------------------------------------------------

def normalise(w, beta):
    """Σ_q β_kq = 1, w rescaled: the counts do not change."""
    t = beta.sum(axis=1)
    t = np.where(t > 0, t, 1.0)
    return w * t[None, :], beta / t[:, None]


def _fit_counts(model, y, q0, lower, upper, jacobian, MI_=MI):
    return mi.minimi_hi(lambda _x, v: model(v), np.arange(len(y)), y, np.asarray(q0, dtype=float),
                        bounds=np.array([lower, upper], dtype=float), MI=MI_, MI2=MI2,
                        tau0=TAU0, eps=EPS, weights='P', jacobian=jacobian)


def beta_step(pb, w, beta, base, mask):
    """The strengths the mask allows, all at once, the weights and baselines
    held: minimi_hi on every count, the strengths fitted as squares (β = θ²,
    no bound to stop at), a strength at 0 starting at SEED times the largest."""
    K, P = beta.shape
    ax = pb.axis
    free = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    b = beta.ravel()[free]
    theta0 = np.sqrt(np.maximum(b, SEED * b.max()))
    slope = -base[:, 0:1] * ax.shape(base) * ax.res               # dμ/dH times dH/dτ's sign

    def full(th):
        out = np.zeros(K * P)
        out[free] = th * th
        return out.reshape(K, P)

    def model(th):
        return pb.counts(w, full(th), base).ravel()

    def jacobian(th):
        wtr = ax.W * np.exp(-np.tensordot(w @ full(th), pb.L, axes=1))         # (n, JN, NV)
        X = node_products(wtr, pb.L)                                            # (P, n, NV)
        return np.array([(2 * t * w[:, i // P][:, None] * slope * X[i % P]).ravel() for i, t in zip(free, th)])
    th = _fit_counts(model, pb.Y.ravel(), theta0, np.full(len(free), -np.inf), np.full(len(free), np.inf),
                     jacobian=jacobian)[0]
    return full(th)


def place_width_derivatives(wtr, slope, T, parts):
    """The counts' derivatives with respect to the places, then the widths, of
    lines whose T in every row is T (n, lines) and whose *parts* are
    Axis.line_parts' (one per line): dμ_s/dδ_q = slope_s T_sq Σ_m wtr[s, m]
    dℓ_q/dδ, slope = −Ns g R the rows' dμ/dH with dH/dτ's sign, wtr = W e^{−τ}
    where they stand. (2 lines, n, NV)."""
    D = node_products(wtr, [part[1] for part in parts] + [part[2] for part in parts])
    return slope[None] * np.concatenate([T, T], axis=1).T[:, :, None] * D


def line_step(pb, w, beta, base):
    """The places and widths of the lines in use, shared by all rows, the
    weights, strengths and baselines held: minimi_hi on every count, its
    derivatives exact (Axis.line_parts); pb's lines and depths are updated."""
    used = np.flatnonzero(beta.sum(axis=0) > 0)
    nu = len(used)
    A = w @ beta
    ax = pb.axis
    slope = -base[:, 0:1] * ax.shape(base) * ax.res               # dμ/dH times dH/dτ's sign

    def depths(q):
        L = pb.L.copy()
        for i, line in enumerate(used):
            L[line] = ax.line_depth(q[i], q[nu + i])
        return L

    def model(q):
        _check_interrupt()
        return ax.counts(ax.node_sum(np.tensordot(A, depths(q), axes=1)), base).ravel()

    def jacobian(q):
        parts = [ax.line_parts(q[i], q[nu + i]) for i in range(nu)]
        L = pb.L.copy()
        for line, part in zip(used, parts):
            L[line] = part[0]
        wtr = ax.W * np.exp(-np.tensordot(A, L, axes=1))                  # (n, JN, NV)
        return place_width_derivatives(wtr, slope, A[:, used], parts).reshape(2 * nu, -1)
    q0 = np.concatenate([pb.lines[used, 1], pb.lines[used, 2]])
    lower = np.concatenate([np.full(nu, ax.x.min()), np.zeros(nu)])
    upper = np.concatenate([np.full(nu, ax.x.max()), np.full(nu, G_MAX)])
    q = _fit_counts(model, pb.Y.ravel(), q0, lower, upper, jacobian=jacobian, MI_=MI_LINES)[0]
    pb.L = depths(q)
    pb.lines[used, 1], pb.lines[used, 2] = q[:nu], q[nu:]


def alternate(pool, pb, beta, mask=None, refine=False, warm=None, max_rounds=MAX_ROUNDS, report=None):
    """From the line sets beta: the weights and baselines (from *warm* = (w,
    baselines) when given), then in turn the strengths the mask allows
    (beta_step), with *refine* the places and widths (line_step), and the
    weights and baselines, until a round gains less than TOL. Returns (w,
    beta, baselines, history of the deviance)."""
    K = len(beta)
    if mask is None:
        mask = np.ones(beta.shape, dtype=bool)
    if warm is None:
        _, beta = normalise(np.ones((pb.n, K)), beta)
        w, base = weight_step(pool, pb, np.full((pb.n, K), START_WEIGHT), beta, pb.start_baselines())
    else:
        w, beta = normalise(warm[0], beta)
        w, base = weight_step(pool, pb, w, beta, warm[1])
    history = [pb.deviance(w, beta, base)]
    for r in range(max_rounds):
        beta = beta_step(pb, w, beta, base, mask)
        w, beta = normalise(w, beta)
        if refine:
            line_step(pb, w, beta, base)
        w, base = weight_step(pool, pb, w, beta, base)
        history.append(pb.deviance(w, beta, base))
        _say(report, f"round {r + 1}: deviance {history[-1]:.1f} ({history[-2] - history[-1]:+.1f})")
        if history[-2] - history[-1] < TOL:
            break
    return w, beta, base, history


def joint_fit_bytes(n, nv, K, strengths, places):
    """The memory of joint_fit's Jacobian, held twice: n rows of nv counts;
    K weights and Ns per row, the strengths, and with places refined two
    values per line."""
    return 16.0 * n * nv * (n * (K + 1) + strengths + 2 * places)


def joint_fit_fits(pb, K, mask, refine):
    """Whether joint_fit of pb's rows with the strengths of *mask* fits JOINT_FIT_MEMORY."""
    mask = np.asarray(mask, dtype=bool)
    return joint_fit_bytes(pb.n, len(pb.axis.x), K, int(mask.sum()),
                           int(mask.any(axis=0).sum()) if refine else 0) <= JOINT_FIT_MEMORY


def joint_fit(pb, w, beta, base, mask, refine=True, MI_=MI):
    """Every value of the components fitted at once, one minimi_hi call: every
    row's weights (as squares, w = v²) and Ns (its lins and c²s held), every
    strength the mask allows (as squares, β = u²), and with *refine* the
    places and widths of the lines in use. Its derivatives are exact
    (jacobian=): a row's weights and Ns touch only its own counts, a line's
    place and width through Axis.line_parts. A weight or strength at 0 starts
    at SEED times the largest. pb's lines are updated. Returns (w, beta,
    baselines, deviance)."""
    n, K = w.shape
    P, nv, ax = pb.P, len(pb.axis.x), pb.axis
    mk = np.asarray(mask, dtype=bool)
    free_u = np.flatnonzero(mk.ravel())
    used = np.flatnonzero(mk.any(axis=0)) if refine else np.zeros(0, dtype=int)
    nu = len(used)
    o_b, o_u = n * K, n * K + n
    o_l = o_u + len(free_u)
    g = ax.shape(base)

    def unpack(q):
        u = np.zeros(K * P)
        u[free_u] = q[o_u:o_l]
        return q[:o_b].reshape(n, K), q[o_b:o_u], u.reshape(K, P), q[o_l:o_l + nu], q[o_l + nu:o_l + 2 * nu]

    def base_of(ns):
        return np.column_stack([ns, base[:, 1], base[:, 2]])

    def depths(pos, G):
        L = pb.L.copy()
        for t, line in enumerate(used):
            L[line] = ax.line_depth(pos[t], G[t])
        return L

    def model(q):
        _check_interrupt()
        v, ns, u, pos, G = unpack(q)
        return ax.counts(ax.node_sum(np.tensordot((v * v) @ (u * u), depths(pos, G), axes=1)), base_of(ns)).ravel()

    def jacobian(q):
        _check_interrupt()
        v, ns, u, pos, G = unpack(q)
        parts = [ax.line_parts(pos[t], G[t]) for t in range(nu)]
        L = pb.L.copy()
        for line, part in zip(used, parts):
            L[line] = part[0]
        wv, bu = v * v, u * u
        wtr = ax.W * np.exp(-np.tensordot(wv @ bu, L, axes=1))           # (n, JN, NV)
        inner = ax.res * wtr.sum(axis=1) + ax.flat + ax.ratio
        slope = -ns[:, None] * g * ax.res
        Qphi = node_products(wtr, np.tensordot(bu, L, axes=1))           # (K, n, NV)
        X = node_products(wtr, L)                                         # (P, n, NV)
        rows = np.zeros((len(q), n, nv))
        for s in range(n):                                                # a row's weights: its own counts
            rows[s * K:(s + 1) * K, s] = 2 * v[s][:, None] * slope[s] * Qphi[:, s]
        rows[o_b + np.arange(n), np.arange(n)] = g * inner                # a row's Ns
        for t, i in enumerate(free_u):                                    # the strengths
            k, line = divmod(int(i), P)
            rows[o_u + t] = 2 * q[o_u + t] * slope * wv[:, k][:, None] * X[line]
        if nu:                                                            # the lines' places and widths
            rows[o_l:] = place_width_derivatives(wtr, slope, wv @ bu[:, used], parts)
        return rows.reshape(len(q), n * nv)
    v0 = np.sqrt(np.maximum(w, SEED * w.max()))
    b0 = beta.ravel()[free_u]
    u0 = np.sqrt(np.maximum(b0, SEED * b0.max()))
    q0 = np.concatenate([v0.ravel(), base[:, 0], u0, pb.lines[used, 1], pb.lines[used, 2]])
    low = np.concatenate([np.full(n * K, -np.inf), np.zeros(n), np.full(len(free_u), -np.inf),
                          np.full(nu, ax.x.min()), np.zeros(nu)])
    high = np.concatenate([np.full(o_l, np.inf), np.full(nu, ax.x.max()), np.full(nu, G_MAX)])
    q = _fit_counts(model, pb.Y.ravel(), q0, low, high, jacobian=jacobian, MI_=MI_)[0]
    v, ns, u, pos, G = unpack(q)
    pb.L = depths(pos, G)
    pb.lines[used, 1], pb.lines[used, 2] = pos, G
    w_, beta_ = normalise(v * v, u * u)
    base_ = base_of(ns)
    return w_, beta_, base_, pb.deviance(w_, beta_, base_)


def fit_jointly(pool, pb, w, beta, base, mask, refine):
    """joint_fit, or the alternation from (w, base) when its Jacobian would
    take more than JOINT_FIT_MEMORY. Returns (w, beta, baselines, deviance)."""
    if joint_fit_fits(pb, len(beta), mask, refine):
        return joint_fit(pb, w, beta, base, mask, refine)
    w, beta, base, hist = alternate(pool, pb, beta, mask=mask, refine=refine, warm=(w, base))
    return w, beta, base, hist[-1]


def strength_derivatives(pb, w, beta, base, free):
    """The counts' derivatives in the strengths *free* (indices into beta's
    K·P), every row's own values profiled out: of row s's derivatives J_s,

        J_s − A_s (A_sᵀ W_s A_s)⁻¹ A_sᵀ W_s J_s,    W_s = 1/μ_s,

    A_s the derivatives of its counts in its weights above 0, its Ns, lins
    and c²s -- what a strength changes when the row's own values follow it to
    their best. Returns ((strengths, n·NV), the counts μ (n·NV,))."""
    ax = pb.axis
    K, P = beta.shape
    wtr = ax.W * np.exp(-np.tensordot(w @ beta, pb.L, axes=1))             # (n, JN, NV)
    inner = ax.res * wtr.sum(axis=1) + ax.flat + ax.ratio
    g = ax.shape(base)
    level = base[:, 0:1] * inner + ax.c_high
    mu = g * level + ax.c_static
    slope = -base[:, 0:1] * g * ax.res                            # dμ/dH times dH/dτ's sign
    Qphi = node_products(wtr, np.tensordot(beta, pb.L, axes=1))             # (K, n, NV)
    X = node_products(wtr, pb.L)                                             # (P, n, NV)
    J = np.array([w[:, i // P][:, None] * slope * X[i % P] for i in free])
    for s in range(pb.n):
        own = np.array([slope[s] * Qphi[k, s] for k in range(K) if w[s, k] > 0]
                       + [g[s] * inner[s], ax.lin * level[s], ax.quad * level[s]])
        weighted = own / mu[s]
        J[:, s, :] -= (np.linalg.pinv(weighted @ own.T, hermitian=True) @ (weighted @ J[:, s, :].T)).T @ own
    return J.reshape(len(free), -1), mu.ravel()


class NoStepLeft(Exception):
    """projected_fit: minimi_hi has refused every step until its damping overflowed."""


def projected_fit(pool, pb, w, beta, base, mask):
    """The strengths the mask allows (as squares, β = θ²) fitted with every
    row's own weights and baseline at their best for them, the places and
    widths held: minimi_hi on the strengths alone, each model evaluation a
    weight_step from the rows' last values, its Jacobian strength_derivatives
    (every row's own derivatives projected out). It is the joint fit's step
    at the cost of the strengths alone (a row's own values touch only its own
    counts), with no Jacobian over every row's values to hold, and it ends
    where the alternation would after all its rounds. One step per Jacobian
    (PROJECTED_MI2): the Jacobian costs less than a model evaluation. A step
    whose rows cannot be fitted is refused (minimi_hi refuses a model that is
    not finite); when it refuses every step its damping overflows and the
    strengths it hands over are no longer finite -- NoStepLeft ends the fit
    there (minimi_hi's own loop would not end). The best strengths fitted are
    the result. Returns (w, beta, baselines, deviance)."""
    K, P = beta.shape
    w, beta = normalise(w, beta)
    free = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    y = pb.Y.ravel()
    solved = [dict(theta=None, w=w, base=base, beta=beta, D=np.inf)]      # the rows fitted so far: see solve

    def full(th):
        out = np.zeros(K * P)
        out[free] = th * th
        return out.reshape(K, P)

    def solve(th):
        """The rows fitted at the strengths th, or None when they cannot be.
        Two are kept, the newest and the best before it -- minimi_hi comes
        back to the point it stands on after a step it refuses -- and a fit
        starts from the rows of the nearer."""
        for rows in solved:
            if rows['theta'] is not None and np.array_equal(th, rows['theta']):
                return rows
        b = full(th)
        if not np.all(np.isfinite(b)):
            return None
        near = min(solved, key=lambda r: 0.0 if r['theta'] is None else float(np.sum((r['theta'] - th) ** 2)))
        try:
            w_, base_ = weight_step(pool, pb, near['w'], b, near['base'])
        except ValueError:                                   # counts not positive in a row: a wild step
            return None
        solved[:] = [dict(theta=np.array(th, dtype=float), w=w_, base=base_, beta=b, D=pb.deviance(w_, b, base_)),
                     min(solved, key=lambda r: r['D'])]
        return solved[0]

    def model(th):
        _check_interrupt()
        if not np.all(np.isfinite(th)):
            raise NoStepLeft()
        rows = solve(th)
        if rows is None:
            return np.full(len(y), np.nan)
        return pb.counts(rows['w'], rows['beta'], rows['base']).ravel()

    def jacobian(th):
        rows = solve(th)
        return 2 * th[:, None] * strength_derivatives(pb, rows['w'], rows['beta'], rows['base'], free)[0]
    b0 = beta.ravel()[free]
    theta0 = np.sqrt(np.maximum(b0, SEED * b0.max()))
    try:
        mi.minimi_hi(lambda _x, v: model(v), np.arange(len(y)), y, theta0, fix=np.array([], dtype=int),
                     bounds=np.array([np.full(len(free), -np.inf), np.full(len(free), np.inf)]), MI=MI,
                     MI2=PROJECTED_MI2, tau0=TAU0, eps=EPS, weights='P', jacobian=jacobian)
    except NoStepLeft:
        pass
    rows = min(solved, key=lambda r: r['D'])                 # minimi_hi takes a step only when the deviance falls
    w, beta = normalise(rows['w'], rows['beta'])
    return w, beta, rows['base'], pb.deviance(w, beta, rows['base'])


def weight_covariances(pb, w, beta, base):
    """Every row's weights' covariance, the inverse of its own Fisher
    information (the K x K block of its weights): (n, K, K), inf where it
    cannot be inverted."""
    ax = pb.axis
    wtr = ax.W * np.exp(-np.tensordot(w @ beta, pb.L, axes=1))
    m = pb.counts(w, beta, base)
    C = (-base[:, 0:1] * ax.shape(base) * ax.res)[:, None, :] \
        * node_products(wtr, np.tensordot(beta, pb.L, axes=1)).transpose(1, 0, 2)     # dμ_s/dw_sk
    out = np.full((len(w), w.shape[1], w.shape[1]), np.inf)
    for s in range(len(w)):
        try:
            out[s] = np.linalg.inv((C[s] / m[s]) @ C[s].T)
        except np.linalg.LinAlgError:
            pass
    return out


def weight_errors(pb, w, beta, base):
    """Every row's weights' errors (weight_covariances): (n, K)."""
    return np.sqrt(np.maximum(np.diagonal(weight_covariances(pb, w, beta, base), axis1=1, axis2=2), 0.0))


# --- small linear fits on the lines' T --------------------------------------------

def line_errors(S):
    """Errors with the missing ones (nan, 0) replaced by that column's usual
    error (its median), or the median of all."""
    valid = np.isfinite(S) & (S > 0)
    typical = np.nanmedian(np.where(valid, S, np.nan), axis=0)
    typical = np.where(np.isfinite(typical), typical, np.nanmedian(np.where(valid, S, np.nan)))
    return np.where(valid, S, typical[None, :])


def nonneg_fit(a, s, profiles):
    """a (errors s) as a sum of the profiles (K, n) with coefficients >= 0,
    weights 1/s²: minimi_hi with the coefficients bounded at 0. When no
    profile goes with a (Σ a M_k/s² <= 0 for every k) every step from 0
    raises χ²: 0 is the answer, and is not handed to minimi_hi. Returns
    (coefficients, their errors (inf where 0), χ²)."""
    M = np.atleast_2d(np.asarray(profiles, dtype=float))
    K = len(M)
    wt = 1 / s ** 2
    norm = (M * M) @ wt
    along = (M * a) @ wt
    if np.all(along <= 0):
        return np.zeros(K), np.full(K, np.inf), float(np.sum(wt * a * a))
    c0 = np.where(norm > 0, np.maximum(along, 0.0) / np.where(norm > 0, norm, 1.0), 0.0)
    c = mi.minimi_hi(lambda _x, v: v @ M, np.arange(len(a)), a, c0, bounds=np.array([np.zeros(K), np.full(K, np.inf)]),
                     MI=MI, MI2=MI2, tau0=TAU0, eps=EPS, weights=wt, chi2_correction=False,
                     jacobian=lambda v: M)[0]
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    err = np.full(K, np.inf)
    active = np.flatnonzero(c > 0)
    if len(active):
        Ma = M[active]
        try:
            err[active] = np.sqrt(np.maximum(np.diag(np.linalg.inv((Ma * wt) @ Ma.T)), 0.0))
        except np.linalg.LinAlgError:
            pass
    return c, err, float(np.sum(wt * (a - c @ M) ** 2))


def rank_one(A, S):
    """Two profiles A (n, 2), errors S, as one: A[s, c] ≈ w_s b_c, weights
    1/S². The ratio b of the weaker to the stronger is fitted (minimi_hi; the
    w_s for a ratio are each spectrum's own weighted mean). Returns (w, b of
    largest |b| 1 and positive sum, χ²)."""
    if not np.any(A):
        return np.zeros(len(A)), np.ones(2), np.inf
    swap = abs(A[:, 1].sum()) > abs(A[:, 0].sum())
    a1, a2 = (A[:, 1], A[:, 0]) if swap else (A[:, 0], A[:, 1])
    W1, W2 = (1 / S[:, 1] ** 2, 1 / S[:, 0] ** 2) if swap else (1 / S[:, 0] ** 2, 1 / S[:, 1] ** 2)

    def residuals(r):
        return (a2 - r[0] * a1) * np.sqrt(W1 * W2 / (W1 * r[0] ** 2 + W2))
    r0 = a2.sum() / a1.sum() if a1.sum() else 0.0
    r = float(mi.minimi_hi(lambda _x, v: residuals(v), np.arange(len(a1)), np.zeros(len(a1)), np.array([r0]),
                           MI=MI, MI2=MI2, tau0=TAU0, eps=EPS, chi2_correction=False)[0][0])
    w = (W1 * a1 + W2 * r * a2) / (W1 + W2 * r * r)
    b = np.array([r, 1.0]) if swap else np.array([1.0, r])
    top = np.max(np.abs(b))
    w, b = w * top, b / top
    if b.sum() < 0:
        w, b = -w, -b
    return w, b, float(np.sum((A - np.outer(w, b)) ** 2 / S ** 2))


# --- the components from all the lines at once (§2.12) ------------------------------

def anchors(A, S):
    """The components from the correlations of all the lines at once. Every
    line's profile -- its T over the rows -- is a non-negative combination of
    the components' weights: the lines of one component alone lie on the edges
    of the cone the profiles span (the anchors), a line of two components
    between two edges. The anchors one at a time (successive projections, on
    the profiles scaled to unit sum): of the lines the anchors so far do not
    explain, the one farthest from their span. A line is explained when its
    profile is a non-negative combination of the anchors' within its errors
    (nonneg_fit, the anchors' own errors added): its weighted χ² below the 99 %
    point of n − K, shared among the lines. Returns (anchors, coefficients
    (P, K), their errors, χ² per line)."""
    n, P = A.shape
    S = line_errors(S)
    total = A.sum(axis=0)
    X = A / np.where(total > 0, total, np.inf)
    z = shared_point(P)
    chosen, coef, err, chi2 = [], np.zeros((P, 0)), np.zeros((P, 0)), np.full(P, np.inf)
    while True:
        _check_interrupt()
        if chosen:
            fits = []
            for q in range(P):
                c0 = nonneg_fit(A[:, q], S[:, q], A[:, chosen].T)[0]
                fits.append(nonneg_fit(A[:, q], np.sqrt(S[:, q] ** 2 + S[:, chosen] ** 2 @ c0 ** 2), A[:, chosen].T))
            coef, err = np.array([f[0] for f in fits]), np.array([f[1] for f in fits])
            chi2 = np.array([f[2] for f in fits])
            unexplained = [q for q in range(P) if q not in chosen and chi2[q] > chi2_point(n - len(chosen), z)]
            if not unexplained or len(chosen) == P:
                return chosen, coef, err, chi2
            Q, _ = np.linalg.qr(X[:, chosen])
            R = X - Q @ (Q.T @ X)
        else:
            unexplained, R = list(range(P)), X
        chosen.append(max(unexplained, key=lambda q: float(np.sum(R[:, q] ** 2))))


def nmf(A, S, W, B, max_rounds=MAX_ROUNDS):
    """Weighted non-negative matrix factorisation A ≈ W B -- W (n, K) and B
    (K, P) >= 0, weights 1/S² -- by alternating non-negative fits from (W, B):
    every row's weights, then every line's coefficients (nonneg_fit), until the
    weighted χ² gains less than TOL. Returns (W, B, B's errors, χ² per line)."""
    S = line_errors(S)
    n, P = A.shape
    history = [float(np.sum((A - W @ B) ** 2 / S ** 2))]
    Berr, chi2 = np.full(B.shape, np.inf), np.zeros(P)
    for _ in range(max_rounds):
        _check_interrupt()
        W = np.array([nonneg_fit(A[s], S[s], B)[0] for s in range(n)])
        fits = [nonneg_fit(A[:, q], S[:, q], W.T) for q in range(P)]
        B = np.array([f[0] for f in fits]).T
        Berr = np.array([f[1] for f in fits]).T
        chi2 = np.array([f[2] for f in fits])
        history.append(float(chi2.sum()))
        if history[-2] - history[-1] < TOL:
            break
    return W, B, Berr, chi2


def columns_of(B, members):
    """The lines with copies: one column per (line, component) membership.
    Returns (the line of every column, mask (K, columns), strengths to start)."""
    cols, owners = [], []
    for q, m in enumerate(members):
        for k in m:
            cols.append(q)
            owners.append(k)
    mask = np.zeros((len(B), len(cols)), dtype=bool)
    beta = np.zeros((len(B), len(cols)))
    for j, (q, k) in enumerate(zip(cols, owners)):
        mask[k, j], beta[k, j] = True, max(float(B[k, q]), SEED)
    return cols, mask, beta


# --- merging and dissolving components, pruning copies ---------------------------------

def merged_components(w, beta, mask, groups, sw):
    """The components of *groups* (lists of component indices) made one each:
    lines and strengths together, a component's strengths scaled by its share
    of the group's profile (rank_one of their weights, errors sw)."""
    b_n = np.zeros((len(groups), beta.shape[1]))
    m_n = np.zeros((len(groups), beta.shape[1]), dtype=bool)
    w_n = np.zeros((len(w), len(groups)))
    for t, g in enumerate(groups):
        if len(g) == 1:
            b_n[t], m_n[t], w_n[:, t] = beta[g[0]], mask[g[0]], w[:, g[0]]
            continue
        prof, b, _ = rank_one(w[:, g], sw[:, g])
        b_n[t], m_n[t], w_n[:, t] = np.maximum(b, 0.0) @ beta[g], mask[g].any(axis=0), np.maximum(prof, 0.0)
    return b_n, m_n, w_n


def misfit_factor(pb, w, beta, base, mask):
    """How much more the rows' counts scatter about the component fit than
    Poisson counts would: its deviance over its degrees of freedom, at least
    1. Measured spectra are never described perfectly (the instrumental
    function, a line shape), and a likelihood-ratio test of the fit's own
    choices then scales its level by it (as the 'Spectrum' row's GAIN by F)."""
    mask = np.asarray(mask, dtype=bool)
    dof = pb.Y.size - pb.n * (len(beta) + 3) - int(mask.sum()) - 2 * int(mask.any(axis=0).sum())
    return max(1.0, pb.deviance(w, beta, base) / max(dof, 1))


def merge_pair(pool, pb, w, beta, base, mask, report=None):
    """Two components whose weights are proportional over the rows (the
    weighted rank-1 χ² of the pair, errors weight_errors, below
    chi2_level(n − 1)) merged, the nearest pair first, and kept when the fit's
    deviance rises by less than chi2_level(n − 1), the values one merge
    removes -- both scaled by the misfit factor. One merge at a time: a weak
    component is not merged away together with others. Returns (w, beta,
    base, mask, deviance, pair), or None (pb as it was)."""
    n, K = w.shape
    if K < 2:
        return None
    F = misfit_factor(pb, w, beta, base, mask)
    sw = line_errors(weight_errors(pb, w, beta, base)) * np.sqrt(F)
    level = chi2_level(n - 1)
    D_old = pb.deviance(w, beta, base)
    pairs = sorted((rank_one(w[:, [a, b]], sw[:, [a, b]])[2], a, b) for a in range(K) for b in range(a + 1, K))
    for chi, a, b in pairs:
        _check_interrupt()
        if chi > level:
            return None
        groups = [[a, b]] + [[k] for k in range(K) if k not in (a, b)]
        b_n, m_n, w_n = merged_components(w, beta, mask, groups, sw)
        w2, b2, base2, D2 = projected_fit(pool, pb, w_n, b_n, base, m_n)
        kept = D2 - D_old < level * F
        _say(report, f"merge components {a} and {b}: deviance {D2 - D_old:+.1f} (level {level * F:.1f}): "
                     f"{'kept' if kept else 'not kept'}")
        if kept:
            return w2, b2, base2, m_n, D2, [int(a), int(b)]
    return None


def dissolve_components(pool, pb, w, beta, base, mask, report=None):
    """A component whose weights are a sum of other components' (coefficients
    >= 0, two or more of them significant: nonneg_fit) is lines those
    components share: its lines are copied into each of them, with its
    strengths times the coefficients, and the component removed; kept when
    the fit's deviance (places held) rises by less than chi2_level of the
    values removed less those added -- the errors and the level scaled by the
    misfit factor. Tried from the fewest lines up; the first kept is returned
    as (Problem, w, beta, base, mask, deviance, record), or None."""
    n, K = w.shape
    if K < 3:
        return None
    F = misfit_factor(pb, w, beta, base, mask)
    sw = line_errors(weight_errors(pb, w, beta, base)) * np.sqrt(F)
    z = normal_point()
    D_old = pb.deviance(w, beta, base)
    for k in sorted(range(K), key=lambda k: int(mask[k].sum())):
        _check_interrupt()
        others = [j for j in range(K) if j != k]
        coef, err, _ = nonneg_fit(w[:, k], sw[:, k], w[:, others].T)
        members = [others[i] for i in range(len(others)) if coef[i] > z * err[i]]
        if len(members) < 2:
            continue
        lines_k = np.flatnonzero(mask[k])
        cols = list(range(pb.P))
        mk = np.zeros((K - 1, pb.P + len(lines_k) * (len(members) - 1)), dtype=bool)
        b0 = np.zeros(mk.shape)
        mk[:, :pb.P], b0[:, :pb.P] = mask[others], beta[others]
        for q in lines_k:
            for t, mbr in enumerate(members):
                if t == 0:
                    j = q
                else:
                    j = len(cols)
                    cols.append(q)
                r = others.index(mbr)
                mk[r, j] = True
                b0[r, j] += coef[r] * beta[k, q]
        new = Problem(pb.axis, pb.Y, pb.lines[cols])
        w2, b2, base2, D2 = projected_fit(pool, new, w[:, others], b0, base, mk)
        fewer = n - 3 * len(lines_k) * (len(members) - 1) - len(lines_k)
        level = chi2_level(fewer) * F
        kept = D2 - D_old < level
        record = dict(component=int(k), lines=[float(pb.lines[q, 1]) for q in lines_k], into=members,
                      coefficients=[float(coef[others.index(m)]) for m in members],
                      deviance_rise=float(D2 - D_old), level=float(level), columns=[int(j) for j in cols])
        _say(report, f"dissolve component {k} into {members}: deviance {D2 - D_old:+.1f} "
                     f"(level {level:.1f}): {'kept' if kept else 'not kept'}")
        if kept:
            return new, w2, b2, base2, mk, D2, record
    return None


def strength_evidence(pb, w, beta, base, mask):
    """Every column's strength over its error, squared -- the deviance its
    removal adds after a refit, to second order (Wald) -- from the Fisher
    information of all the strengths the mask allows, every row's own weights
    and baseline profiled out (strength_derivatives): (columns,), inf where it
    cannot be told."""
    P = beta.shape[1]
    free = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    J, mu = strength_derivatives(pb, w, beta, base, free)
    variance = np.diag(np.linalg.pinv((J / mu) @ J.T, hermitian=True))
    out = np.full(P, np.inf)
    for t, i in enumerate(free):
        if variance[t] > 0:
            out[i % P] = beta.ravel()[i] ** 2 / variance[t]
    return out


def prune(pool, pb, w, beta, base, mask, cols, report=None):
    """The copies of a line that is in several components tried in the order
    of their evidence (strength_evidence, recomputed after a removal), the
    least first: removed when the refit without it (projected_fit) raises
    the deviance by no more than the 99 % point of χ²(1), shared among the
    copies and scaled by the misfit factor. It ends at the first copy the
    counts need whose evidence was above the level too: the refit and its
    prediction agree, and every copy left has more evidence. A line's last
    copy stays. Returns (w, beta, base, mask, deviance, the copies removed)."""
    D = pb.deviance(w, beta, base)
    shared = [j for j in range(len(cols)) if cols.count(cols[j]) > 1]
    level = chi2_point(1, shared_point(len(shared))) * misfit_factor(pb, w, beta, base, mask)
    removed, tried = [], set()
    evidence = strength_evidence(pb, w, beta, base, mask) if shared else None
    while True:
        _check_interrupt()
        alive = [j for j in shared if j not in tried and mask[:, j].any()
                 and sum(1 for i in range(len(cols)) if cols[i] == cols[j] and mask[:, i].any()) > 1]
        if not alive:
            break
        j = min(alive, key=lambda c: evidence[c])
        tried.add(j)
        less = mask.copy()
        less[:, j] = False
        w2, beta2, b2, D2 = projected_fit(pool, pb, w, np.where(less, beta, 0.0), base, less)
        gone = D2 - D <= level
        _say(report, f"copy of {pb.lines[j, 1]:+.3f} mm/s: deviance {D2 - D:+.1f} (evidence "
                     f"{evidence[j]:.1f}, level {level:.1f}): {'removed' if gone else 'kept'}")
        if gone:
            mask, w, beta, base, D = less, w2, beta2, b2, D2
            removed.append(j)
            evidence = strength_evidence(pb, w, beta, base, mask)
        elif evidence[j] > level:
            break
    return w, beta, base, mask, D, removed


def recombine(pool, pb, w, beta, base, mask, cols, report=None):
    """Recombinations that take lines out of a component and leave the counts
    as they are. When every line of component j is in component k too, k less
    α times j and j's weights plus α times k's give every row the same optical
    depth (up to the places and widths of the two copies of each line); α the
    largest that keeps k's strengths >= 0 takes at least one line out of k.
    Each kept when the refit (places held) raises the deviance by no more than
    the 99 % point of χ² of the lines taken out, scaled by the misfit factor;
    repeated while one is kept. Returns (w, beta, base, mask, recombinations)."""
    F = misfit_factor(pb, w, beta, base, mask)
    D = pb.deviance(w, beta, base)
    done = []
    while True:
        _check_interrupt()
        K = len(beta)
        owned = [{cols[c]: c for c in np.flatnonzero(mask[k])} for k in range(K)]
        tries = []
        for k in range(K):
            for j in range(K):
                if j == k or not owned[j] or not set(owned[j]) <= set(owned[k]):
                    continue
                ratio = {q: beta[k, owned[k][q]] / beta[j, owned[j][q]] for q in owned[j]}
                alpha = min(ratio.values())
                out = [q for q, r in ratio.items() if r <= alpha]
                tries.append((-len(out), k, j, alpha, out))
        kept = False
        for _, k, j, alpha, out in sorted(tries, key=lambda t: t[:3]):
            b2, m2, w2 = beta.copy(), mask.copy(), w.copy()
            for q, c in owned[j].items():
                b2[k, owned[k][q]] = max(beta[k, owned[k][q]] - alpha * beta[j, c], 0.0)
            for q in out:
                b2[k, owned[k][q]], m2[k, owned[k][q]] = 0.0, False
            w2[:, j] = w[:, j] + alpha * w[:, k]
            w3, beta3, base3, D3 = projected_fit(pool, pb, w2, b2, base, m2)
            level = chi2_level(len(out)) * F
            _say(report, f"component {k + 1} less {alpha:.3f} of component {j + 1}: {len(out)} line(s) out, "
                         f"deviance {D3 - D:+.1f} (level {level:.1f})")
            if D3 - D <= level:
                w, beta, base, mask, D = w3, beta3, base3, m2, D3
                done.append(dict(component=int(k), less=int(j), alpha=float(alpha),
                                 lines_out=[float(pb.lines[owned[k][q], 1]) for q in out]))
                kept = True
                break
        if not kept:
            return w, beta, base, mask, done


def compact(lines, beta, mask, cols):
    """The columns no component holds any more removed."""
    used = mask.any(axis=0)
    return lines[used], beta[:, used], mask[:, used], [q for q, u in zip(cols, used) if u]


def drop_vanished(axis, lines, beta, mask, cols, w):
    """A line whose place has run to an end of the range has vanished (the
    'Spectrum' row's rule, spectrum_reference.at_an_end): its columns removed,
    then any component left without a line, every component's strengths
    summing to 1 again. Returns (lines, beta, mask, cols, w)."""
    inside = ~at_an_end(lines[:, 1], (axis.x.min(), axis.x.max()), axis.x)
    mask = mask & inside[None, :]
    lines, beta, mask, cols = compact(lines, np.where(mask, beta, 0.0), mask, cols)
    keep = mask.any(axis=1)
    w, beta = normalise(w[:, keep], beta[keep])
    return lines, beta, mask[keep], cols, w


def merge_duplicates(lines, beta, mask, cols):
    """Two lines of one component closer than a natural width are one line of
    it (copies of one line that ended in the same component, or two lines that
    came together): their strengths summed, their strength-weighted place and
    width, the closest pair first. Returns (lines, beta, mask, cols)."""
    lines, beta, mask = lines.copy(), beta.copy(), mask.copy()
    while True:
        pairs = [(abs(lines[i, 1] - lines[j, 1]), k, i, j) for k in range(len(beta))
                 for i in np.flatnonzero(mask[k]) for j in np.flatnonzero(mask[k]) if i < j]
        close = [p for p in pairs if p[0] < NAT_WIDTH]
        if not close:
            return compact(lines, beta, mask, cols)
        _, k, i, j = min(close)
        share = np.array([beta[k, i], beta[k, j]])
        share = share / share.sum() if share.sum() > 0 else np.full(2, 0.5)
        lines[i, 1:] = share @ lines[[i, j], 1:]
        beta[k, i] += beta[k, j]
        beta[k, j], mask[k, j] = 0.0, False


# --- the lines of a summed spectrum: the 'Spectrum' row's own fit ------------------------

def line_fwhm(G):
    """The full width at half maximum (mm/s) of a line of the natural width
    and the Gaussian G, measured on the program's line shape (models.Voigt_r)."""
    half = 4 * (NAT_WIDTH + G) + 0.5
    E = np.linspace(-half, half, 40001)
    v = m5.Voigt_r(NAT_WIDTH, float(G), E)
    above = E[v >= v.max() / 2]
    return float(above[-1] - above[0])


def merge_close(fitter, fit):
    """Two lines of *fit* closer than a natural width are one line; two that
    overlap (closer than the wider one's FWHM) are one unless two lower the
    deviance by the 'Spectrum' row's own rule for a line, GAIN F, against one
    with a free G. The closest pair first; the one line has their T summed,
    their T-weighted place, the wider G plus half their distance, and the
    lines are refitted. Returns the fit kept."""
    tested = set()
    while True:
        lines = fit.lines
        pairs = sorted((abs(lines[i, 1] - lines[j, 1]), i, j) for i in range(len(lines))
                       for j in range(i + 1, len(lines)))
        pair = next(((d, i, j) for d, i, j in pairs
                     if d < max(line_fwhm(lines[i, 2]), line_fwhm(lines[j, 2]))
                     and (d < NAT_WIDTH or (lines[i, 1], lines[j, 1]) not in tested)), None)
        if pair is None:
            return fit
        d, i, j = pair
        Ti, Tj = max(lines[i, 0], 0.0), max(lines[j, 0], 0.0)
        place = (Ti * lines[i, 1] + Tj * lines[j, 1]) / max(Ti + Tj, 1e-300)
        one = np.vstack([np.delete(lines, [i, j], axis=0), [[Ti + Tj, place, max(lines[i, 2], lines[j, 2]) + d / 2]]])
        merged = fitter.fit(fit.p[:NB], one)
        if d < NAT_WIDTH or merged.deviance - fit.deviance < GAIN * fitter.F:
            fit = merged
        else:
            tested.add((lines[i, 1], lines[j, 1]))


def fit_of_sum(axis, y, report=None):
    """The 'Spectrum' row's own fit of the counts y (ReferenceFitter.automatic,
    its baseline's curvature free), close lines merged: its LineFit."""
    fitter = ReferenceFitter(axis.x, y, axis.model, axis.method['method'], report=report)
    return merge_close(fitter, fitter.automatic())


def lines_of_sum(axis, y):
    """The lines (T, δ, G) of the counts y (fit_of_sum)."""
    return fit_of_sum(axis, y).lines


def _lines_of_sum_task(axis, y):
    return lines_of_sum(axis, y)


def new_lines(found, lines):
    """The lines of *found* with no line of *lines* (nor an earlier one of
    *found*) within half the wider width (FWHM), nor within a natural width;
    T 1."""
    kept = []
    for line in found:
        w = line_fwhm(line[2])
        if all(abs(line[1] - d) >= max(w / 2, line_fwhm(G) / 2, NAT_WIDTH) for _, d, G in list(lines) + kept):
            kept.append(np.array([1.0, line[1], line[2]]))
    return kept


# --- spectra of equal composition (§2.9) and the search by misfit (§2.10) ------------------

def deviance_norm(y, mu, n_free):
    """D (norm) of the counts y against the model mu with n_free fitted values,
    and its 1σ, as the Fit quality tab forms them (fit_quality)."""
    dof = len(y) - n_free
    mean, variance = deviance_moments(mu)
    ideal = 1 + np.sum(mean - 1) / dof
    return float(mi.poisson_deviance(y, mu)) / dof / ideal, float(np.sqrt(np.sum(variance)) / dof / ideal)


def covariance_T(axis, L, T, base):
    """The T block of the inverse Poisson Fisher matrix of one spectrum's fit
    with every line's own T: T's covariance, the baseline profiled out."""
    mu, rows = axis.derivatives(L, T, base)
    return np.linalg.pinv((rows / mu) @ rows.T, hermitian=True)[:len(L), :len(L)]


def _line_task(axis, L, Y, T0, base0):
    """Spectra of one task, every line its own T: (T, baselines, T's
    covariances, deviances, D (norm), its 1σ)."""
    n, P = len(Y), len(L)
    T, base, C = np.zeros((n, P)), np.zeros((n, 3)), np.zeros((n, P, P))
    dev, dn, sn = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, y in enumerate(Y):
        T[i], base[i], _ = fit_one(axis, y, L, T0[i], base0[i], signed=True)
        mu = axis.counts(axis.node_sum(np.tensordot(T[i], L, axes=1)), base[i])
        C[i] = covariance_T(axis, L, T[i], base[i])
        dev[i] = float(mi.poisson_deviance(y, mu))
        dn[i], sn[i] = deviance_norm(y, mu, P + 3)
    return T, base, C, dev, dn, sn


def line_fits_for_tests(pool, pb, T0):
    """Every row with every line its own T, over the pool: (T, baselines,
    covariances of T, deviances, D (norm), its 1σ)."""
    chunks = _chunks(pool, pb.n)
    base0 = pb.start_baselines()
    results = m5._pool_starmap(pool, _line_task, [(pb.axis, pb.L, pb.Y[idx], T0[idx], base0[idx])
                                                  for idx in chunks])
    out = [np.zeros((pb.n, pb.P)), np.zeros((pb.n, 3)), np.zeros((pb.n, pb.P, pb.P)), np.zeros(pb.n),
           np.zeros(pb.n), np.zeros(pb.n)]
    for idx, res in zip(chunks, results):
        for a, r in zip(out, res):
            a[idx] = r
    return out


def _sum_task(axis, L, Ys, T0, members):
    """Spectra Ys summed, every line its own T: (T, baseline, D (norm), its
    1σ, and with *members* the deviance of the spectra themselves with that
    one composition, each its own baseline)."""
    y = Ys.sum(axis=0)
    T, base, _ = fit_one(axis, y, L, T0, axis.start_baselines(y[None])[0], signed=True)
    mu = axis.counts(axis.node_sum(np.tensordot(T, L, axes=1)), base)
    dn, sn = deviance_norm(y, mu, len(L) + 3)
    joint = None
    if members:
        tau = np.tensordot(T, L, axes=1)
        _, bases, _, dev = _fit_rows(axis, Ys, tau[None], np.ones((len(Ys), 1)), axis.start_baselines(Ys),
                                     False, False)
        joint = float(dev.sum())
    return T, base, dn, sn, joint


def wald_distances(T, C):
    """(n, n): every pair's (T_a − T_b)ᵀ (C_a + C_b)⁻¹ (T_a − T_b), a χ² of P
    degrees of freedom when the two have one composition."""
    n = len(T)
    D = np.zeros((n, n))
    for a in range(n - 1):
        d = T[a] - T[a + 1:]
        M = C[a][None] + C[a + 1:]
        try:
            x = np.linalg.solve(M, d[..., None])[..., 0]
        except np.linalg.LinAlgError:
            x = np.einsum('ipq,iq->ip', np.linalg.pinv(M, hermitian=True), d)
        D[a, a + 1:] = D[a + 1:, a] = np.einsum('ij,ij->i', d, x)
    return np.where(np.isfinite(D), D, np.inf)


def complete_linkage(D):
    """The merge tree of complete linkage on the distances D (n, n): two
    groups are as far apart as their farthest pair, and the two nearest are
    merged, again and again. Returns [(left, right, distance)], the node
    n + t made by merge t, the leaves 0 .. n − 1."""
    n = len(D)
    D = np.array(D, dtype=float)
    np.fill_diagonal(D, np.inf)
    node = np.arange(n)
    alive = np.ones(n, dtype=bool)
    best = D.argmin(axis=1)
    merges = []
    for t in range(n - 1):
        nearest = np.where(alive, D[np.arange(n), best], np.inf)
        i = int(np.argmin(nearest))
        if np.isfinite(nearest[i]):
            j = int(best[i])
        else:                                       # only infinite distances are left
            i, j = (int(k) for k in np.flatnonzero(alive)[:2])
        merges.append((int(node[i]), int(node[j]), float(D[i, j])))
        D[i] = np.maximum(D[i], D[j])
        D[:, i] = D[i]
        D[i, i] = np.inf
        D[j], D[:, j] = np.inf, np.inf
        alive[j] = False
        node[i] = n + t
        redo = np.flatnonzero(alive & ((best == i) | (best == j) | (np.arange(n) == i)))
        best[redo] = D[redo].argmin(axis=1)
    return merges


def groups_of(pool, pb, Y, T, C, dev):
    """The groups of equal composition (§2.9): complete linkage on the pair
    test (wald_distances), a node no farther than chi2_level(P) confirmed by
    its likelihood ratio -- the deviance of its spectra with the one
    composition of their sum against their own -- below chi2_level(P (m − 1));
    a node that fails split into its two branches. Returns (members of every
    group, the fit of every confirmed group's sum by its first member: (T,
    baseline, D (norm), 1σ))."""
    n = len(Y)
    level = chi2_level(pb.P)
    merges = complete_linkage(wald_distances(T, C))
    children = {n + t: m for t, m in enumerate(merges)}

    def members(node):
        out, stack = [], [node]
        while stack:
            k = stack.pop()
            if k < n:
                out.append(k)
            else:
                stack += [children[k][1], children[k][0]]
        return out
    frontier, groups, sum_fits = [2 * n - 2 if n > 1 else 0], [], {}
    while frontier:
        _check_interrupt()
        test, nxt = [], []
        for node in frontier:
            if node < n:
                groups.append([node])
            elif children[node][2] > level:
                nxt += [children[node][0], children[node][1]]
            else:
                test.append(node)
        idx_of = [members(node) for node in test]
        results = m5._pool_starmap(pool, _sum_task, [(pb.axis, pb.L, Y[idx], T[idx].mean(axis=0), True)
                                                     for idx in idx_of])
        for node, idx, (Tc, bc, dn, sn, joint) in zip(test, idx_of, results):
            if joint - float(dev[idx].sum()) <= chi2_level(pb.P * (len(idx) - 1)):
                groups.append(idx)
                sum_fits[idx[0]] = (Tc, bc, dn, sn)
            else:
                nxt += [children[node][0], children[node][1]]
        frontier = nxt
    return groups, sum_fits


def search_lines(pool, axis, Y, lines, report=None, rounds=SEARCH_ROUNDS):
    """Missing lines by misfit (§2.10), from the line list *lines*. Every
    round: every spectrum fitted with every line's own T and its D (norm); the
    groups of equal composition; in every group two sums at most, no spectrum
    in both -- its spectra whose D (norm) is above its 99 % point, and the
    others; the first goes to the 'Spectrum' row's own fit, the second too
    when its own D (norm) with one composition is above its 99 % point; a line
    found there with no line of the list nearby is new (new_lines); the sums
    with nothing missing refine the places and widths of the lines. Until no
    sum gives a new line. Returns (lines, log)."""
    z = normal_point()
    lines = np.asarray(lines, dtype=float).reshape(-1, 3).copy()
    T0 = np.zeros((len(Y), len(lines)))
    log = []
    for r in range(rounds):
        t0 = time.time()
        pb = Problem(axis, Y, lines)
        T, base, C, dev, dn, sn = line_fits_for_tests(pool, pb, T0)
        groups, _ = groups_of(pool, pb, Y, T, C, dev)
        bad = dn > 1 + z * sn
        jobs, good = [], []
        for idx in groups:
            idx = np.asarray(idx)
            misfit, rest = idx[bad[idx]], idx[~bad[idx]]
            if len(misfit):
                jobs.append(('misfit', Y[misfit].sum(axis=0)))
            if len(rest):
                good.append(rest)
        sums = m5._pool_starmap(pool, _sum_task, [(axis, pb.L, Y[rest], T[rest].mean(axis=0), False)
                                                  for rest in good])
        flagged = [k for k, (_, _, dnk, snk, _) in enumerate(sums) if dnk > 1 + z * snk]
        refine = [k for k in range(len(good)) if k not in flagged]
        jobs += [('others', Y[good[k]].sum(axis=0)) for k in flagged]
        found = m5._pool_starmap(pool, _lines_of_sum_task, [(axis, y) for _, y in jobs])
        added = []
        for f in found:
            added += new_lines(f, list(lines) + added)
        if refine:
            sub = Problem(axis, np.array([Y[good[k]].sum(axis=0) for k in refine]), lines)
            A, _, bs, _ = line_fits(pool, sub, np.array([T[good[k]].mean(axis=0) for k in refine]),
                                    sub.start_baselines())
            line_step(sub, A, np.eye(sub.P), bs)
            lines = sub.lines.copy()
        log.append(dict(round=r + 1, lines=int(len(lines)), groups=len(groups), misfitting_spectra=int(bad.sum()),
                        misfit_sums=sum(1 for kind, _ in jobs if kind == 'misfit'), others_flagged=len(flagged),
                        added=[[float(line[1]), float(line[2])] for line in added], seconds=time.time() - t0))
        _say(report, f"Search by misfit, round {r + 1}: {len(lines)} lines, {len(groups)} groups, "
                     f"{int(bad.sum())} misfitting spectra; {len(added)} line(s) added")
        if not added:
            break
        lines = np.vstack([lines] + [np.reshape(line, (1, 3)) for line in added])
        T0 = np.hstack([T, np.zeros((len(Y), len(added)))])
    return lines, log


def fitted_rows(pool, axis, Y, lines):
    """The rows the components are fitted on. The spectra themselves when
    every step can take them -- the joint fit on them fits JOINT_FIT_MEMORY,
    with a component per line (the most there can be): they keep a phase
    confined to a few spectra in those spectra, where sums of spectra that
    agree only within their noise dilute it. More spectra: every count to one
    of two halves (binomial thinning: two independent Poisson halves), the
    groups of equal composition formed on the first, and the sums of the
    second (§2.9). Returns (the rows, which row every spectrum is in, sums
    used)."""
    n, P = len(Y), len(lines)
    if joint_fit_bytes(n, Y.shape[1], P, P, P) <= JOINT_FIT_MEMORY:
        return Y, np.arange(n), False
    first = np.random.default_rng(HALVES_SEED).binomial(np.rint(Y).astype(np.int64), 0.5).astype(float)
    pb = Problem(axis, first, lines)
    T, _, C, dev, _, _ = line_fits_for_tests(pool, pb, np.zeros((n, pb.P)))
    groups, _ = groups_of(pool, pb, first, T, C, dev)
    which = np.zeros(n, dtype=int)
    for k, idx in enumerate(groups):
        which[idx] = k
    second = Y - first
    return np.array([second[idx].sum(axis=0) for idx in groups]), which, True


# --- missing components after the components (§2.11) ----------------------------------

def responses(axis):
    """Per width of CANDIDATE_G, every place's response: h_j(v) = Σ_m W ℓ_j,
    the node sum of a line of unit T at the place x_j to first order, on
    every velocity; scaled to its largest value 1, kept down to RESPONSE_FLOOR
    of it. (widths, places, velocities)."""
    out = np.zeros((len(CANDIDATE_G), len(axis.x), len(axis.x)))
    for wi, G in enumerate(CANDIDATE_G):
        for j, place in enumerate(axis.x):
            _check_interrupt()
            h = np.sum(axis.W * axis.line_depth(float(place), G), axis=0)
            h = h / max(h.max(), 1e-300)
            out[wi, j] = np.where(h > RESPONSE_FLOOR, h, 0.0)
    return out


def filtered(Y, M, H):
    """Per width and place, every spectrum's U = Σ (m − y) h and I = Σ m h²:
    (widths, n, places) each. A missing line makes U > 0; U²/I is the
    deviance a line of the width at the place would gain, to first order (the
    matched filter)."""
    U = np.tensordot(M - Y, H, axes=([1], [2])).transpose(1, 0, 2)
    I = np.tensordot(M, H * H, axes=([1], [2])).transpose(1, 0, 2)
    return U, np.maximum(I, 1e-300)


def places_of(U, I, level, x):
    """Per width and place, the spectra whose own gain there passes *level*,
    and their pooled gain (Σ U)²/Σ I; the PLACES best places at least a
    natural width apart: [(width, place, members)]."""
    e = np.where(U > 0, U * U, 0.0) / I
    over = e > level
    S = np.where(over, U, 0.0).sum(axis=1) ** 2 / np.maximum(np.where(over, I, 0.0).sum(axis=1), 1e-300)
    S = np.where(over.any(axis=1), S, 0.0)
    taken, out = np.zeros_like(S, dtype=bool), []
    for _ in range(PLACES):
        w, j = np.unravel_index(int(np.argmax(np.where(taken, -np.inf, S))), S.shape)
        if S[w, j] <= 0:
            break
        out.append((int(w), int(j), np.flatnonzero(over[w, :, j])))
        taken[:, np.abs(x - x[j]) <= NAT_WIDTH] = True
    return out


def afterwards(pool, axis, Y, lines, beta, H, report=None):
    """Missing components after the components, several of them. Every count
    to one of two halves; round after round: every spectrum's weights on each
    half; the matched filter of the first half's residuals (filtered), the
    places where spectra pass their 99 % point, pooled (places_of); at the
    best place, the same spectra, place and width on the OTHER half: a line
    is there when their pooled score passes the 99 % point shared among the
    PLACES tried (counts that took no part in the choice); its spectra's other
    half summed and fitted by the 'Spectrum' row's own fit, its new lines
    (none new: the line at that place and width) joining a component added
    before whose spectra are most of these, else making a new one; every
    strength refitted on all counts. Returns (lines, beta, the new
    components' rows, log)."""
    z = normal_point()
    z_judge = shared_point(PLACES)
    choose = np.random.default_rng(AFTER_SEED).binomial(np.rint(Y).astype(np.int64), 0.5).astype(float)
    judge = Y - choose
    w_c = w_j = np.full((len(Y), len(beta)), START_WEIGHT)
    base_c, base_j = axis.start_baselines(choose), axis.start_baselines(judge)
    log, new_rows, holders = [], [], {}
    for r in range(AFTER_ROUNDS):
        _check_interrupt()
        entry = dict(round=r + 1)
        pb_c, pb_j = Problem(axis, choose, lines), Problem(axis, judge, lines)
        w_c, base_c = weight_step(pool, pb_c, w_c, beta, base_c)
        w_j, base_j = weight_step(pool, pb_j, w_j, beta, base_j)
        U, I = filtered(choose, pb_c.counts(w_c, beta, base_c), H)
        Uj, Ij = filtered(judge, pb_j.counts(w_j, beta, base_j), H)
        tried, added, target, members = [], [], None, None
        for wi, j, members in places_of(U, I, z * z, axis.x):
            zj = float(Uj[wi, members, j].sum() / np.sqrt(Ij[wi, members, j].sum()))
            tried.append(dict(place=float(axis.x[j]), G=CANDIDATE_G[wi], spectra=len(members), z_other_half=zj))
            if zj <= z_judge:
                continue                                     # no line there on the other half
            found = lines_of_sum(axis, judge[members].sum(axis=0))
            kept = new_lines(found, list(lines))
            added = ([f for f in found if any(abs(f[1] - k[1]) < 1e-12 and abs(f[2] - k[2]) < 1e-12 for k in kept)]
                     or [np.array([1.0, axis.x[j], CANDIDATE_G[wi]])])
            tried[-1]['new'] = [[float(line[1]), float(line[2])] for line in added]
            target = next((k for k, held in holders.items() if np.isin(members, held).mean() > 0.5), None)
            break
        entry.update(tried=tried, level=float(z_judge))
        log.append(entry)
        _say(report, f"Missing components, round {r + 1}: {len(tried)} place(s) tried, "
                     f"{len(added)} line(s) found")
        if not added:
            break
        T_new = np.array([max(float(line[0]), SEED) for line in added])
        lines = np.vstack([lines] + [np.array([[1.0, line[1], line[2]]]) for line in added])
        beta = np.hstack([beta, np.zeros((len(beta), len(added)))])
        if target is None:                                   # a new component, its strengths as fitted
            beta = np.vstack([beta, np.concatenate([np.zeros(beta.shape[1] - len(added)), T_new / T_new.sum()])])
            new_rows.append(len(beta) - 1)
            holders[r + 1] = members
            w_c = np.hstack([w_c, np.ones((len(Y), 1))])
            w_j = np.hstack([w_j, np.ones((len(Y), 1))])
        else:                                                # the lines join it, at its lines' mean strength
            k = new_rows[list(holders).index(target)]
            beta[k, -len(added):] = beta[k][beta[k] > 0].mean()
        pb = Problem(axis, Y, lines)
        w_all, base_all = weight_step(pool, pb, np.ones((len(Y), len(beta))), beta, pb.start_baselines())
        beta = normalise(w_all, beta_step(pb, w_all, beta, base_all, beta > 0))[1]
    return lines, beta, new_rows, log


# --- the whole search ------------------------------------------------------------------

class Components:
    """What run finds: the lines (T, δ, G; T the mean over the spectra of the
    components' line T), every component's strengths beta (K, P) and mask,
    which line every column is a copy of (cols), every spectrum's weights w
    (n, K) with their covariance (n, K, K) and errors, baseline (Ns, lins,
    c²s), deviance and D (norm) with its 1σ, and the record of the search."""

    def __init__(self, lines, beta, mask, cols, w, w_cov, base, deviance, dnorm, dnorm_sigma, record):
        self.lines, self.beta, self.mask, self.cols = lines, beta, mask, cols
        self.w, self.w_cov, self.base = w, w_cov, base
        self.w_err = np.sqrt(np.maximum(np.diagonal(w_cov, axis1=1, axis2=2), 0.0))
        self.deviance, self.dnorm, self.dnorm_sigma = deviance, dnorm, dnorm_sigma
        self.record = record


def run(pool, axis, Y, report=None):
    """The whole search (the module docstring) on the spectra Y (n, NV) of
    *axis*, after the warm-up where one is needed (its seconds recorded
    apart, then: every step's are after it). *report(text)* hears what it
    does. Returns Components."""
    Y = np.asarray(Y, dtype=float)
    t0 = time.time()
    rec = dict(seconds_warm_up=time.time() - t0) if warm_up(pool, report) else {}
    t0 = time.time()
    _say(report, "Lines of the summed spectrum...")
    summed = fit_of_sum(axis, Y.sum(axis=0))
    lines = summed.lines
    axis.curvature = (float(summed.p[3]), float(summed.p[2]))        # its lins and c²s: the start
    rec.update(summed_lines=[[float(v) for v in line] for line in lines], curvature=list(axis.curvature))
    lines, rec['search'] = search_lines(pool, axis, Y, lines, report)
    rec['seconds_search'] = time.time() - t0
    t = time.time()
    _say(report, f"{len(lines)} lines; the rows to fit...")
    Yfit, which, rec['sums'] = fitted_rows(pool, axis, Y, lines) if len(Y) > 1 \
        else (Y, np.zeros(1, dtype=int), False)
    pb = Problem(axis, Yfit, lines)
    A, S, base, _ = line_fits(pool, pb, np.zeros((pb.n, pb.P)), pb.start_baselines())
    _say(report, f"{len(Yfit)} rows; the components from the lines...")
    anchor, coef, _, _ = anchors(A, S)
    W, B, Berr, _ = nmf(A, S, A[:, anchor].copy(), coef.T.copy())
    z = normal_point()
    members = [[k for k in range(len(B)) if B[k, q] > z * Berr[k, q]] or [int(np.argmax(B[:, q]))]
               for q in range(len(lines))]
    cols, mask, beta = columns_of(B, members)
    lines = lines[cols]
    rec.update(rows=int(len(Yfit)), anchors=[float(pb.lines[q, 1]) for q in anchor], K0=int(len(B)),
               seconds_components=time.time() - t)
    H = responses(axis)
    warm = (W, base)
    rec['loops'] = []
    for loop in range(MAX_LOOPS):
        entry, t = dict(loop=loop + 1), time.time()
        _say(report, f"Loop {loop + 1}: {len(beta)} components, {len(lines)} lines; fitting...")
        pbf = Problem(axis, Yfit, lines)
        w, beta, b, _ = alternate(pool, pbf, beta, mask=mask, refine=True, warm=warm)
        w, beta, b, _ = projected_fit(pool, pbf, w, beta, b, mask)   # where every test's refit ends too
        merges = dissolved = 0
        while True:
            res = merge_pair(pool, pbf, w, beta, b, mask, report)
            if res is not None:
                w, beta, b, mask, _, _ = res
                merges += 1
                continue
            res = dissolve_components(pool, pbf, w, beta, b, mask, report)
            if res is None:
                break
            pbf, w, beta, b, mask, _, record = res
            cols = [cols[j] for j in record['columns']]
            dissolved += 1
        w, beta, b, mask, recombined = recombine(pool, pbf, w, beta, b, mask, cols, report)
        w, beta, b, mask, _, removed = prune(pool, pbf, w, beta, b, mask, cols, report)
        lines, beta, mask, cols, w = drop_vanished(axis, pbf.lines.copy(), beta, mask, cols, w)
        entry.update(merges=merges, dissolved=dissolved, recombined=recombined, pruned=len(removed),
                     K=int(len(beta)))
        new_lines_, new_beta, new_rows, log = afterwards(pool, axis, Y, lines, beta, H, report)
        added = len(new_lines_) - len(lines)
        entry.update(added_lines=added, added_components=len(new_rows), afterwards=log, seconds=time.time() - t)
        rec['loops'].append(entry)
        if added == 0:
            warm = (w, b)
            break
        cols = cols + list(range(max(cols) + 1, max(cols) + 1 + added))
        lines, beta, mask = new_lines_, new_beta, new_beta > 0
        warm = None
    t = time.time()
    _say(report, f"The joint fit of {len(beta)} components...")
    merged = merge_duplicates(lines, beta, mask, cols)
    if len(merged[0]) < len(lines):
        rec['duplicates_merged'] = int(len(lines) - len(merged[0]))
        lines, beta, mask, cols = merged
    pbf = Problem(axis, Yfit, lines)
    w, beta, b, _ = alternate(pool, pbf, beta, mask=mask, warm=warm, max_rounds=2)
    rec['joint_fit'] = joint_fit_fits(pbf, len(beta), mask, True)
    w, beta, b, _ = fit_jointly(pool, pbf, w, beta, b, mask, True)
    lines, beta, mask, cols, w = drop_vanished(axis, pbf.lines.copy(), beta, mask, cols, w)
    rec['seconds_joint'] = time.time() - t
    t = time.time()
    _say(report, "Every spectrum's weights on all its counts...")
    pb_all = Problem(axis, Y, lines)
    w_all, base_all = weight_step(pool, pb_all, w[which], beta, pb_all.start_baselines())
    w_cov = weight_covariances(pb_all, w_all, beta, base_all)
    mu = pb_all.counts(w_all, beta, base_all)
    dev = np.array([float(mi.poisson_deviance(y, m)) for y, m in zip(Y, mu)])
    n_free = len(beta) + 3
    dn = np.array([deviance_norm(y, m, n_free) for y, m in zip(Y, mu)])
    lines[:, 0] = (w_all @ beta).mean(axis=0)
    rec.update(seconds_final=time.time() - t, seconds=time.time() - t0)
    _say(report, f"{len(beta)} components found ({time.time() - t0:.0f} s)")
    return Components(lines, beta, mask, cols, w_all, w_cov, base_all, dev, dn[:, 0], dn[:, 1], rec)
