"""Components of many spectra (components_search).

Pinned:
* the counts of the line model on one axis are TI's -- SMS and CMS, the CMS
  non-resonant counts as a coefficient and in the source's form of the Δθ
  dialogs (CMS_source_geometry_calibration._baseline), a polarized known part
  (a Be-like Doublet) folded into the node weights, a smeared source -- and its
  derivatives are the counts' own (the joint fit's, a line's place and width
  among them, exact); Interrupt stops it;
* complete linkage against a brute-force one, the non-negative fit against
  every subset solved exactly, the rank-1 fit against the SVD;
* the whole search on a small series -- a sextet turning into a doublet --
  finds the two with their lines and weights;
* the results folder: every file, the tables, the component's spectrum, and
  the .mdl files load into the table as the rows they stand for; their rows
  start as the table starts them.
"""
import itertools
import json
import os

import numpy as np
import pytest

import syncmoss.components_results as cr
import syncmoss.components_search as cs
import syncmoss.models as m5
from syncmoss import instrumental_io as iio
from syncmoss.CMS_source_geometry_calibration import _baseline
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import (LINE_RATIO_25, LINE_RATIO_34, NAT_WIDTH, SMS_POL_DEFAULT, TESLA_PER_MMS,
                                number_of_baseline_parameters as NB)
from syncmoss.model_io import load_model_from_path, model_file_rows, read_model
from syncmoss.spectrum_model import parse_spectrum_lines

from conftest import FROZEN_PARAMETERS, SerialPool


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_MULCO_SMS, _X0_SMS = _frozen("INSint.txt")
METHODS = {
    "CMS": dict(method='CMS', x0=0.0, MulCo=MulCoCMS, INS=float(_frozen("GCMS.txt", delimiter='\t')), Met=1),
    "SMS": dict(method='SMS', x0=float(_X0_SMS), MulCo=float(_MULCO_SMS), INS=np.atleast_1d(_frozen("INSexp.txt")),
                Met=0),
}
JN = 16
X = np.linspace(-4.0, 4.0, 96)
# A polarized Doublet every spectrum holds (T, δ, ε, L, G, θk, φh, A, G2/G1): the known part
KNOWN = [0.3, 0.25, 0.4, NAT_WIDTH, 0.1, 90.0, 0.0, -0.2, 1.0]
LINES = np.array([[0.8, -1.2, 0.0], [1.4, 0.3, 0.25], [0.5, 1.6, 0.6]])


def _smearing(method, x):
    mp = METHODS[method]
    return m5.source_smearing(x, JN, mp['MulCo'], mp['INS'], mp['Met'], 0.2 if method == 'CMS' else 0.0, 0.0,
                              0.25, channel_widths=np.full(len(x), x[1] - x[0]), x0=mp['x0'])


def _axis(method, nonres=(0.0, 0.0, 0.0), smeared=False, known=True, x=X):
    smearing = _smearing(method, x) if smeared else None
    mp = METHODS[method]
    norm = iio.compute_norm(SerialPool(), JN, mp, x if smeared else None, smearing)
    return cs.Axis(x, JN, mp, norm, smearing, SMS_POL_DEFAULT, [('Doublet', KNOWN)] if known else (), nonres), \
        norm, smearing


@pytest.mark.parametrize("method, source, smeared", [("SMS", None, False), ("SMS", None, True),
                                                     ("CMS", (0.67,), False), ("CMS", (0.7, 30.0, 12.0), True)])
def test_the_line_model_is_ti(method, source, smeared):
    """μ = g (Ns (R H + F + r) + C_high) + C_static against TI of the known
    Doublet and three Singlets with the baseline of the user's choice."""
    if source is None:
        nonres = cs.nonresonant(method)
    elif len(source) == 1:
        nonres = cs.nonresonant(method, k=source[0])
    else:
        nonres = cs.nonresonant(method, f_lm=source[0], c_high=source[1], c_static=source[2])
    axis, norm, smearing = _axis(method, nonres, smeared)
    ns, lins, c2s = 5000.0, 0.3, -2.0
    middle = 0.5 * (X[0] + X[-1])
    if source is not None and len(source) == 3:
        p8 = np.array(_baseline(ns, source[0], source[1]), dtype=float)      # the Δθ dialogs' Nnr
        offset = source[2]
    else:
        p8 = np.array([ns, 0, 0, 0, nonres[0] * ns, 0, 0, 0], dtype=float)
        offset = 0.0
    p8[[1, 2, 3]] = p8[[5, 6, 7]] = [middle, c2s, lins]
    mp = METHODS[method]
    p = np.concatenate([p8, KNOWN] + [[T, d, NAT_WIDTH, G] for T, d, G in LINES])
    want = np.asarray(m5.TI(X, p, ['Doublet'] + ['Singlet'] * 3, JN, SerialPool(), mp['x0'], mp['MulCo'],
                            mp['INS'], [0], [0], Met=mp['Met'], Norm=norm, smearing=smearing), float) + offset
    L = np.array([axis.line_depth(d, G) for _, d, G in LINES])
    got = axis.counts(axis.node_sum(np.tensordot(LINES[:, 0], L, axes=1)), [ns, lins, c2s])
    assert np.allclose(got, want, rtol=1e-10, atol=0)
    # not vacuous: the known part absorbs where no line is
    plain = _axis(method, nonres, smeared, known=False)[0]
    bare = plain.counts(plain.node_sum(np.tensordot(LINES[:, 0], L, axes=1)), [ns, lins, c2s])
    assert np.max(np.abs(bare - got)) > 0.01 * ns
    # the derivatives are the counts' own (central differences)
    mu, rows = axis.derivatives(L, LINES[:, 0], np.array([ns, lins, c2s]))
    assert np.allclose(mu, got, rtol=1e-12, atol=0)
    q0 = np.concatenate([LINES[:, 0], [ns, lins, c2s]])
    for i in range(len(q0)):
        h = 1e-6 * max(abs(q0[i]), 1.0)
        up, down = q0.copy(), q0.copy()
        up[i] += h
        down[i] -= h
        diff = (axis.counts(axis.node_sum(np.tensordot(up[:3], L, axes=1)), up[3:])
                - axis.counts(axis.node_sum(np.tensordot(down[:3], L, axes=1)), down[3:])) / (2 * h)
        assert np.allclose(rows[i], diff, rtol=1e-5, atol=1e-6 * np.max(np.abs(diff))), i
    m5.FIT_CANCEL.set()
    try:
        with pytest.raises(m5.FitInterrupted):
            axis.line_depth(0.0, 0.1)
    finally:
        m5.FIT_CANCEL.clear()


def test_the_place_and_width_derivatives_are_the_counts_own():
    """place_width_derivatives (the line step's and the joint fit's Jacobian
    for the shared places and widths) against central differences of the
    counts of two spectra: exact, from the depth's twin with its derivatives
    (Axis.line_parts), whose depth is the program's Voigt's. Every G above 0:
    at 0 the program's pseudo-Voigt has a slope next to nothing in G (a true
    Voigt's is 0), so a difference there tells nothing."""
    axis = _axis("SMS")[0]
    T = np.array([[0.9, 1.6, 0.4], [0.2, 0.8, 1.1]])
    base = np.array([[4000.0, 0.2, -1.0], [2500.0, -0.1, 0.5]])
    places, widths = LINES[:, 1], np.array([0.05, 0.25, 0.6])

    def counts(pl, wd):
        L = np.array([axis.line_depth(d, G) for d, G in zip(pl, wd)])
        return axis.counts(axis.node_sum(np.tensordot(T, L, axes=1)), base)
    L = np.array([axis.line_depth(d, G) for d, G in zip(places, widths)])
    parts = [axis.line_parts(d, G) for d, G in zip(places, widths)]
    assert np.allclose([part[0] for part in parts], L, rtol=1e-12, atol=0)
    wtr = axis.W * np.exp(-np.tensordot(T, L, axes=1))
    slope = -base[:, 0:1] * axis.shape(base) * axis.res
    got = cs.place_width_derivatives(wtr, slope, T, parts)
    for q in range(len(LINES)):
        for which, values in ((0, places), (1, widths)):
            h = 1e-5
            up, down = values.copy(), values.copy()
            up[q] += h
            down[q] -= h
            pair = (up, widths) if which == 0 else (places, up)
            low = (down, widths) if which == 0 else (places, down)
            want = (counts(*pair) - counts(*low)) / (2 * h)
            assert np.allclose(got[which * len(LINES) + q], want, rtol=1e-6, atol=1e-7 * np.max(np.abs(want))), \
                (q, which)
    # the line's depth is TImod's Singlet (whose exp, then log, rounds a small depth at 1e-16)
    mp = METHODS["SMS"]
    for d, G in ((-1.2, 0.0), (0.3, 0.25), (1.6, 0.6)):
        timod = np.asarray(m5.TImod(axis.x, np.array([1.0, d, NAT_WIDTH, G]), ['Singlet'], axis.model.E.ravel(),
                                    mp['x0'], mp['MulCo'], mp['INS'], [0], [0], -1), dtype=float)
        assert np.allclose(axis.line_depth(d, G), -np.log(timod).reshape(axis.W.shape), rtol=1e-12, atol=1e-15)


def test_the_joint_fit_derivatives_are_the_counts_own(monkeypatch):
    """joint_fit's jacobian= -- every row's weights (as squares) and Ns, the
    strengths the mask allows (as squares), the places and widths of the
    lines in use -- against central differences of its own model, at its
    start: minimi_hi is replaced by a stand-in that keeps both."""
    axis = _axis("SMS")[0]
    lines = np.column_stack([np.ones(3), LINES[:, 1], [0.05, 0.25, 0.6]])
    pb = cs.Problem(axis, np.ones((3, len(X))), lines)
    w = np.array([[1.2, 0.3], [0.5, 0.9], [0.1, 1.4]])
    beta = np.array([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])
    base = np.array([[4000.0, 0.2, -1.0], [2500.0, -0.1, 0.5], [3000.0, 0.0, 0.3]])
    kept = {}

    def stand_in(model, y, q0, lower, upper, jacobian, MI_=cs.MI):
        kept.update(model=model, jacobian=jacobian, q0=np.array(q0, dtype=float))
        return (np.array(q0, dtype=float),)
    monkeypatch.setattr(cs, '_fit_counts', stand_in)
    cs.joint_fit(pb, w, beta, base, beta > 0, refine=True)
    q0, model = kept['q0'], kept['model']
    rows = kept['jacobian'](q0)
    assert rows.shape == (w.size + len(w) + 4 + 2 * 3, len(w) * len(X))
    for i in range(len(q0)):
        h = 1e-6 * max(abs(q0[i]), 1.0)
        up, down = q0.copy(), q0.copy()
        up[i] += h
        down[i] -= h
        want = (model(up) - model(down)) / (2 * h)
        assert np.allclose(rows[i], want, rtol=1e-5, atol=1e-6 * np.max(np.abs(want))), i


def test_the_warm_up_is_done_once():
    """The search's warm-up loads its compiled kernels in this process and in
    the pool's; once numba has them it runs nothing and says nothing."""
    said = []
    cs.warm_up(SerialPool(), said.append)
    assert cs._ready()
    said.clear()
    assert cs.warm_up(SerialPool(), said.append) is False and said == []


def test_nonresonant_refuses_what_the_dtheta_dialogs_refuse():
    assert cs.nonresonant('SMS', k=0.67) == (0.0, 0.0, 0.0)
    for bad in (dict(f_lm=0.0), dict(f_lm=1.0), dict(f_lm=0.7, c_high=-1.0), dict(f_lm=0.7, c_static=-1.0),
                dict(k=-0.1)):
        with pytest.raises(ValueError):
            cs.nonresonant('CMS', **bad)


def _brute_complete_linkage(D):
    """Complete linkage the plain way: every step, every pair of clusters at
    the distance of its farthest members, the nearest merged."""
    clusters = [frozenset([i]) for i in range(len(D))]
    out = []
    while len(clusters) > 1:
        d, a, b = min((max(D[i, j] for i in clusters[a] for j in clusters[b]), a, b)
                      for a in range(len(clusters)) for b in range(a + 1, len(clusters)))
        out.append((clusters[a] | clusters[b], d))
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [clusters[a] | clusters[b]]
    return out


def test_complete_linkage_is_the_plain_one():
    rng = np.random.default_rng(3)
    for n in (2, 5, 13):
        D = rng.uniform(0, 10, (n, n))
        D = D + D.T
        np.fill_diagonal(D, 0.0)
        merges = cs.complete_linkage(D)
        members = {i: frozenset([i]) for i in range(n)}
        got = []
        for t, (a, b, d) in enumerate(merges):
            members[n + t] = members[a] | members[b]
            got.append((members[n + t], d))
        want = _brute_complete_linkage(D)
        assert [m for m, _ in got] == [m for m, _ in want]
        assert np.allclose([d for _, d in got], [d for _, d in want], rtol=1e-12, atol=0)
    # only infinite distances left: everything still ends in one tree
    D = np.full((4, 4), np.inf)
    D[0, 1] = D[1, 0] = 1.0
    merges = cs.complete_linkage(D)
    assert len(merges) == 3 and merges[0][2] == 1.0 and np.isinf(merges[-1][2])


def _exact_nonneg(a, s, M):
    """Every subset of the profiles solved exactly: the best with all its
    coefficients positive."""
    wt = 1 / s ** 2
    best = (np.zeros(len(M)), float(np.sum(wt * a * a)))
    for r in range(1, len(M) + 1):
        for idx in itertools.combinations(range(len(M)), r):
            Ms = M[list(idx)]
            c = np.linalg.solve((Ms * wt) @ Ms.T, (Ms * wt) @ a)
            chi = float(np.sum(wt * (a - c @ Ms) ** 2))
            if np.all(c > 0) and chi < best[1]:
                full = np.zeros(len(M))
                full[list(idx)] = c
                best = (full, chi)
    return best


def test_nonneg_fit_and_rank_one_against_exact_solutions():
    rng = np.random.default_rng(8)
    M = rng.uniform(0, 1, (3, 40))
    s = rng.uniform(0.5, 1.5, 40)
    for c_true in ([1.0, 0.5, 0.0], [2.0, -0.7, 0.3]):              # the second forces a coefficient to 0
        a = np.asarray(c_true) @ M + 0.05 * rng.standard_normal(40)
        c, err, chi = cs.nonneg_fit(a, s, M)
        want, want_chi = _exact_nonneg(a, s, M)
        assert np.allclose(c, want, rtol=1e-5, atol=1e-7) and abs(chi - want_chi) < 1e-6 * max(want_chi, 1)
        assert np.all(np.isinf(err[c == 0])) and np.all(np.isfinite(err[c > 0]))
    assert np.any(cs.nonneg_fit(np.array([2.0, -0.7, 0.3]) @ M, s, M)[0] == 0)
    # equal errors: the rank-1 fit of two profiles is the leading singular pair
    A = np.outer(rng.uniform(1, 3, 30), [1.0, 0.6]) + 0.05 * rng.standard_normal((30, 2))
    w, b, chi = cs.rank_one(A, np.ones_like(A))
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    assert np.allclose(b / b[np.argmax(np.abs(b))], Vt[0] / Vt[0][np.argmax(np.abs(Vt[0]))], rtol=1e-6)
    assert abs(chi - sv[1] ** 2) < 1e-6 * sv[1] ** 2 and np.max(np.abs(b)) == 1.0
    assert np.allclose(np.outer(w, b), sv[0] * np.outer(U[:, 0], Vt[0]), rtol=1e-6, atol=1e-9)


# --- the whole search and its results folder -------------------------------------------

SERIES_X = np.linspace(-6.0, 6.0, 160)


def _sextet(delta, H, G):
    """A powder sextet's lines (T per unit T of the phase, place, G)."""
    h = H / TESLA_PER_MMS / 2
    places = [delta - h, delta - LINE_RATIO_25 * h, delta - LINE_RATIO_34 * h, delta + LINE_RATIO_34 * h,
              delta + LINE_RATIO_25 * h, delta + h]
    return np.array([[t, d, G] for t, d in zip([1 / 4, 1 / 6, 1 / 12, 1 / 12, 1 / 6, 1 / 4], places)])


PHASES = (_sextet(0.0, 33.0, 0.1), np.array([[0.5, -0.05, 0.15], [0.5, 0.75, 0.15]]))


@pytest.fixture(scope="module")
def found():
    """Twelve SMS spectra, a sextet turning into a doublet, 300 counts per
    point off resonance, searched once."""
    pool = SerialPool()
    axis = cs.axis_for(pool, SERIES_X, JN, METHODS['SMS'], SMS_POL_DEFAULT)
    depth = np.array([np.tensordot(ph[:, 0], [axis.line_depth(d, G) for _, d, G in ph], axes=1) for ph in PHASES])
    s = np.linspace(0, 1, 12)
    T = np.column_stack([10 * (1 - s) + 0.5, 6 * s + 0.3])
    base = np.column_stack([np.full(12, 300.0), np.zeros(12), np.zeros(12)])
    mu = axis.counts(axis.node_sum(np.einsum('sk,kmj->smj', T, depth)), base)
    Y = np.random.default_rng(2).poisson(mu).astype(float)
    return dict(axis=axis, Y=Y, T=T, mu=mu, res=cs.run(pool, axis, Y))


def _owner(res, phase):
    """The component holding every line of *phase* (within 0.05 mm/s), or None."""
    for k in range(len(res.beta)):
        own = res.lines[res.mask[k], 1]
        if all(np.min(np.abs(own - d)) < 0.05 for _, d, _ in phase):
            return k
    return None


@pytest.mark.xdist_group("components_search_run")
def test_the_search_finds_the_two_phases(found):
    res, T = found['res'], found['T']
    assert len(res.beta) == 2
    owners = [_owner(res, ph) for ph in PHASES]
    assert None not in owners and owners[0] != owners[1]
    for k, Tk in zip(owners, T.T):
        assert np.corrcoef(res.w[:, k], Tk)[0, 1] > 0.99
        assert np.all(np.isfinite(res.w_err[:, k])) and np.all(res.w_err[:, k] > 0)
    # it describes the counts as well as the truth does (it has more values)
    truth = sum(float(cs.mi.poisson_deviance(y, m)) for y, m in zip(found['Y'], found['mu']))
    assert res.deviance.sum() < truth
    assert abs(np.mean(res.dnorm) - 1) < 3 * np.mean(res.dnorm_sigma) / np.sqrt(len(res.dnorm))


@pytest.mark.xdist_group("components_search_run")
def test_the_projected_fit_ends_where_the_alternation_cannot_gain(found):
    """projected_fit (the strengths, every row's weights and baseline at their
    best for them): from strengths off by up to a third it ends no higher than
    the alternation with the places held, and a round of the alternation from
    there gains less than its own tolerance."""
    pool, res = SerialPool(), found['res']
    pb = cs.Problem(found['axis'], found['Y'], res.lines)
    off = np.where(res.mask, res.beta * np.random.default_rng(5).uniform(0.7, 1.3, res.beta.shape), 0.0)
    history = cs.alternate(pool, pb, off, mask=res.mask, warm=(res.w, res.base))[3]
    w, beta, base, D = cs.projected_fit(pool, pb, res.w, off, res.base, res.mask)
    assert D <= history[-1] + cs.TOL and D < history[0]
    assert np.allclose(beta.sum(axis=1), 1.0) and np.all(beta[~res.mask] == 0)
    again = cs.alternate(pool, pb, beta, mask=res.mask, warm=(w, base), max_rounds=1)[3]
    assert again[0] - again[-1] < cs.TOL


@pytest.mark.xdist_group("components_search_run")
def test_the_projected_fit_ends_when_every_step_is_refused(found, monkeypatch):
    """Rows that can be fitted at the start's strengths only: minimi_hi
    refuses every step, its damping overflows and its own loop would not end
    -- projected_fit ends there (NoStepLeft) with the start's strengths."""
    pool, res = SerialPool(), found['res']
    pb = cs.Problem(found['axis'], found['Y'], res.lines)
    off = np.where(res.mask, res.beta * np.random.default_rng(5).uniform(0.7, 1.3, res.beta.shape), 0.0)
    step, calls = cs.weight_step, []

    def only_once(*args):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("weights='P' needs counts")
        return step(*args)
    monkeypatch.setattr(cs, 'weight_step', only_once)
    w, beta, base, D = cs.projected_fit(pool, pb, res.w, off, res.base, res.mask)
    assert len(calls) > 2 and np.isfinite(D)
    assert np.allclose(beta, off / off.sum(axis=1, keepdims=True), rtol=0, atol=1e-5)


def _job(found, files=None, coordinates=None):
    x = found['axis'].x
    header = ['# the tests\' instrumental function']
    settings = dict(method_params=METHODS['SMS'], JN=JN, pol=SMS_POL_DEFAULT, known_rows=[], nonres=[0.0, 0.0, 0.0],
                    header=header)
    return dict(x=x, Y=found['Y'], A=x, files=files or [f'spectrum_{s}.dat' for s in range(len(found['Y']))],
                coordinates=coordinates, method_params=METHODS['SMS'], header=header,
                JN=JN, pol=SMS_POL_DEFAULT, known=(), nonres=(0.0, 0.0, 0.0), settings=settings)


def _spectra_files(found, folder, header=()):
    """The series as .dat files named X_Y: four columns, three rows."""
    from syncmoss.spectrum_io import save_spectrum_with_metadata
    os.makedirs(folder, exist_ok=True)
    files = []
    for s, y in enumerate(found['Y']):
        files.append(os.path.join(folder, f'map_{s % 4}_{s // 4}.dat'))
        save_spectrum_with_metadata(files[-1], found['axis'].x, y, list(header))
    return files


@pytest.mark.gui
@pytest.mark.xdist_group("components_search_run")
def test_the_results_folder(found, physics_app, tmp_path):
    res = found['res']
    folder = str(tmp_path / 'components')
    written = cr.write_results(folder, res, _job(found), SerialPool())
    K = len(res.beta)
    names = {os.path.basename(p) for p in written}
    assert names == {cr.RUN_FILE, cr.ARRAYS_FILE, cr.LINES_MODEL, cr.COMPONENTS_TABLE, cr.LINES_TABLE} | \
        {cr.component_file(k, e) for k in range(K) for e in ('mdl', 'dat')}
    run, again, arrays = cr.read_results(folder)
    assert np.array_equal(arrays['w'], res.w) and len(run['components']) == K
    assert np.array_equal(again.w_err, res.w_err) and again.cols == list(res.cols)
    table = np.genfromtxt(os.path.join(folder, cr.COMPONENTS_TABLE), skip_header=1, usecols=range(2, 2 + 2 * K))
    assert np.array_equal(table[:, 0::2], res.w) and np.array_equal(table[:, 1::2], res.w_err)
    # the component's spectrum: the model's counts of its 'Spectrum' row, with the header
    k = _owner(res, PHASES[1])
    path = os.path.join(folder, cr.component_file(k, 'dat'))
    with open(path, encoding='utf-8') as f:
        assert f.readline().startswith('#') and "the tests' instrumental function" in f.read()
    counts = np.loadtxt(path, comments='#')
    assert counts.shape == (len(SERIES_X), 2) and np.min(counts[:, 1]) < 0.8 * np.max(counts[:, 1])
    # the .mdl files load as the rows they stand for
    load_model_from_path(physics_app, os.path.join(folder, cr.component_file(k, 'mdl')))
    model, p, *_, Distri, _Cor, _Expr, _NExpr, _DN, _Recon, _RN = read_model(physics_app)
    own = np.flatnonzero(res.mask[k])
    assert model == ['Spectrum'] and abs(p[NB] - cr.component_T(res.w, res.w_err)[k]) < 1e-9
    assert np.allclose(parse_spectrum_lines(Distri[0]), np.column_stack([res.beta[k, own], res.lines[own, 1:]]),
                       rtol=1e-12, atol=0)
    load_model_from_path(physics_app, os.path.join(folder, cr.LINES_MODEL))
    model, p, *_ = read_model(physics_app)
    lines, _ = cr.unique_lines(res)
    assert model == ['Singlet'] * len(lines)
    assert np.allclose(np.reshape(p[NB:], (-1, 4))[:, [0, 1, 3]], lines, rtol=1e-12, atol=0)
    with open(os.path.join(folder, cr.RUN_FILE), encoding='utf-8') as f:
        assert json.load(f)['record']['rows'] == len(found['Y'])


@pytest.mark.gui
def test_the_model_rows_start_as_the_table_starts_them(physics_app):
    """The results' .mdl rows carry the bounds and fixes a table row of the
    model starts with: the baseline of a new table, a Singlet, a 'Spectrum'."""
    table = physics_app.params_table
    for row, model in ((1, 'Singlet'), (2, 'Spectrum')):
        table.select_model(row, model)
    _names, _colors, rows = model_file_rows(physics_app)

    def bounds_and_fix(fields):
        return [(f[1], f[2], f[4]) for f in fields]
    assert bounds_and_fix(rows[0][:NB]) == bounds_and_fix(cr.baseline_fields(1.0, 0.0, 0))
    assert [f[0] for f in rows[0][4:NB]] == [f[0] for f in cr.baseline_fields(1.0, 0.0, 0)[4:]]
    assert bounds_and_fix(rows[1][:4]) == bounds_and_fix(cr.singlet_fields(1.0, 0.0, 0.1))
    assert bounds_and_fix(rows[2][:4]) == bounds_and_fix(cr.spectrum_fields(1.0, np.zeros((1, 3))))


class _Event:
    """What the window's mouse handlers read of a matplotlib event."""

    def __init__(self, x, y, button=1):
        self.xdata, self.ydata, self.button = x, y, button


def _plotted_data(window, canvas):
    """The counts plotted as data on a canvas of the window (plot_fitting_result's data line)."""
    from syncmoss.spectrum_plotter import DATA_GID
    ax = window.canvases[canvas].figure.axes[0]
    return next(line.get_ydata() for line in ax.lines if line.get_gid() == DATA_GID)


@pytest.mark.gui
@pytest.mark.xdist_group("components_search_run")
def test_the_window_selects_and_sums(found, physics_app, tmp_path, monkeypatch):
    """The window on a map: the threshold on the doublet's T selects the
    spectra above it, the brush adds and erases pixels, the Summed tab plots
    their sum with its fit, "Save summed spectrum" is "Sum all spectra"'s
    function on them, a click on the map shows the spectrum with its fit, the
    table of an image is saved."""
    import shiboken6
    import syncmoss.components_window as cw
    import syncmoss.spectrum_io as sio
    app, res = physics_app, found['res']
    app.pool = SerialPool()
    files = _spectra_files(found, str(tmp_path / 'spectra'))
    folder = str(tmp_path / 'components')
    cr.write_results(folder, res, _job(found, files, cw.coordinates_from_names(files)), SerialPool())
    window = cw.open_components_window(app, folder)
    try:
        assert window.is_map and window.grid.shape == (3, 4)
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == \
            ["Components", "Lines", "Map", "Spectrum", "Summed spectrum", "Scatter"]

        def red():
            return np.asarray(window.canvases["Map"].figure.axes[0].images[0].get_array())[..., 0]
        assert np.isclose(red().max(), 1.0) and np.sum(red() == 1.0) == 1      # R: T1, scaled to its largest
        window.high[0].setValue(50)
        assert np.sum(red() == 1.0) > 1                                        # half of it is full red now
        window.full_range()
        assert np.sum(red() == 1.0) == 1
        k = _owner(res, PHASES[1])
        window.threshold_image.setCurrentIndex(k)
        window.threshold.setValue(500)
        window.select_by_threshold()
        w = res.w[:, k]
        above = w > w.min() + 0.5 * (w.max() - w.min())
        assert np.array_equal(window.selected, above) and above.any() and not above.all()
        # the brush: erase a selected pixel, add one that is not
        on, off = int(np.flatnonzero(above)[0]), int(np.flatnonzero(~above)[0])
        window.brush.setChecked(True)
        window.erase.setChecked(True)
        window._pressed(_Event(window.rowcol[on, 1], window.rowcol[on, 0]))
        window.erase.setChecked(False)
        window._pressed(_Event(window.rowcol[off, 1], window.rowcol[off, 0]))
        want = above.copy()
        want[on], want[off] = False, True
        assert np.array_equal(window.selected, want)
        # the Summed tab: their sum with its fit (the main window's fit display)
        assert np.array_equal(_plotted_data(window, "Summed"), found['Y'][want].sum(axis=0))
        # the sum saved: "Sum all spectra"'s function on the selected files
        out = str(tmp_path / 'sum.dat')
        monkeypatch.setattr(sio.QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **kw: (out, '')))
        monkeypatch.setattr(app, 'show_pressed', lambda: None)
        window.sum_selected()
        assert np.array_equal(np.loadtxt(out)[:, 1], found['Y'][want].sum(axis=0))
        # a click on the map: the spectrum with its fit
        s = 5
        window._clicked(_Event(window.rowcol[s, 1], window.rowcol[s, 0]))
        assert window.tabs.currentIndex() == 3 and np.array_equal(_plotted_data(window, "Spectrum"), found['Y'][s])
        table = str(tmp_path / 'images.txt')
        monkeypatch.setattr(cw.QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **kw: (table, '')))
        window.save_table()
        saved = np.genfromtxt(table, skip_header=1)
        assert saved.shape[0] == len(files) and np.array_equal(saved[:, 2:4], cw.coordinates_from_names(files))
    finally:
        shiboken6.delete(window)


@pytest.mark.gui
@pytest.mark.xdist_group("components_search_run")
def test_the_window_on_a_series(found, physics_app, tmp_path, monkeypatch):
    """The window on a series: the Evolution tab plots every T over every
    spectrum's D (norm); the brush dragged along the series adds (erases) the
    spectra between press and release; Combine makes T1+T2 an image; the
    components are shown through the instrumental function first; the
    spectra are saved by their number in the series, with the header."""
    import shiboken6
    from PySide6.QtCore import QItemSelectionModel
    import syncmoss.components_window as cw
    app, res = physics_app, found['res']
    app.pool = SerialPool()
    files = _spectra_files(found, str(tmp_path / 'spectra'))
    folder = str(tmp_path / 'components')
    job = _job(found, files)
    cr.write_results(folder, res, job, SerialPool())
    window = cw.open_components_window(app, folder)
    try:
        assert not window.is_map and window.tabs.tabText(2) == "Evolution"
        top, bottom = window.canvases["Evolution"].figure.axes
        assert bottom.get_ylabel() == "D (norm)"
        # the brush: dragged over spectra 2 to 4 (numbered from 1), then 3 erased
        window.brush.setChecked(True)
        window._pressed(_Event(2.2, 0.0))
        window._released(_Event(3.8, 0.0))
        want = np.zeros(window.n, dtype=bool)
        want[1:4] = True
        assert np.array_equal(window.selected, want)
        window.erase.setChecked(True)
        window._pressed(_Event(3.0, 0.0))
        window._released(_Event(3.0, 0.0))
        want[2] = False
        assert np.array_equal(window.selected, want)
        assert np.array_equal(_plotted_data(window, "Summed"), found['Y'][want].sum(axis=0))
        # Combine: T1+T2 with the error of the sum
        rows = window.component_table.selectionModel()
        for k in (0, 1):
            rows.select(window.component_table.model().index(k, 0),
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window.combine_selected()
        name, values, errors = window.images[-1]
        assert name == "T1+T2" and np.array_equal(values, res.w[:, 0] + res.w[:, 1])
        assert np.allclose(errors ** 2, res.w_cov[:, 0, 0] + res.w_cov[:, 1, 1] + 2 * res.w_cov[:, 0, 1])
        assert window.threshold_image.itemText(window.threshold_image.count() - 1) == name
        # the components through the instrumental function first: transmissions; then the lines alone
        assert window.which.currentIndex() == 0
        shown = window.canvases["Components"].figure.axes[0].lines
        for k in range(window.K):
            _T, transmission = window.component_spectrum(k)
            assert np.array_equal(shown[k].get_ydata(), transmission) and transmission.max() <= 1.0 + 1e-12
        window.which.setCurrentIndex(window.K + 1)
        shown = window.canvases["Components"].figure.axes[0].lines
        assert len(shown) == window.K and all(line.get_ydata().max() <= 0.0 for line in shown)
        # the spectra by the layout: their number in the series, the header kept
        out = str(tmp_path / 'by_layout')
        os.makedirs(out)
        monkeypatch.setattr(cw.QFileDialog, 'getExistingDirectory', staticmethod(lambda *a, **kw: out))
        window.save_spectra_by_layout()
        assert sorted(os.listdir(out)) == [f"{s + 1:02d}.dat" for s in range(window.n)]
        s = 4
        path = os.path.join(out, f"{s + 1:02d}.dat")
        assert np.array_equal(np.loadtxt(path, comments='#')[:, 1], found['Y'][s])
        with open(path, encoding='utf-8') as f:
            assert job['header'][0] in f.read()
    finally:
        shiboken6.delete(window)


@pytest.mark.gui
def test_show_found_components_warns_first(physics_app, monkeypatch):
    """A new feature: its window's entry warns before anything else, as the
    search's does (test_find_components_runs_from_the_path_box)."""
    import syncmoss.components_window as cw
    import syncmoss.experimental_features as ef
    shown = []
    monkeypatch.setattr(ef.QMessageBox, 'warning', lambda parent, title, text: shown.append(title))
    monkeypatch.setattr(cw.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **kw: shown.append('folder') or ''))
    cw.show_components(physics_app)
    assert shown == ["Show found components: new feature", 'folder']


@pytest.mark.gui
def test_the_dialog_reads_its_choices(physics_app):
    """CMS: the source's non-resonant counts; the map's order; the known part
    read with the program's own reader of the preset."""
    import shiboken6
    import syncmoss.components_window as cw
    from syncmoss.Calibration import _read_optics_preset
    files = [f'spectrum_{s}.dat' for s in range(7)]
    job = dict(files=files, x=X, method_params=dict(METHODS['CMS'], source='internal'))
    dialog = cw.SearchDialog(physics_app, job)
    try:
        assert not dialog.nonres_box.isHidden() and dialog.known.currentText() == 'nothing'
        assert dialog.series.isChecked() and not dialog.by_names.isEnabled()      # no X_Y in these names
        dialog.by_source.setChecked(True)
        dialog.f_lm.setValue(0.8)
        dialog.c_high.setValue(5.0)
        dialog.c_static.setValue(2.0)
        dialog.by_grid.setChecked(True)
        dialog.columns.setValue(3)
        dialog.rows.setValue(3)
        dialog.order.setCurrentIndex(1)
        dialog.known.setCurrentText('KB_nano')
        choices = dialog._read()
        assert choices['nonres'] == pytest.approx(((1 - 0.8) / 0.8, 5.0, 2.0))
        assert np.array_equal(choices['coordinates'], cw.grid_coordinates(7, 3, True))
        assert choices['coordinates'][3].tolist() == [2.0, 1.0]              # the second row runs back
        assert np.array_equal(choices['known'][0][1], _read_optics_preset(physics_app.params_dir, 'KB'))
        dialog.rows.setValue(2)
        with pytest.raises(ValueError, match="hold fewer"):
            dialog._read()
    finally:
        shiboken6.delete(dialog)


@pytest.mark.gui
@pytest.mark.xdist_group("components_search_run")
def test_find_components_runs_from_the_path_box(found, physics_app, tmp_path, monkeypatch):
    """Supp -> Find principal components...: the path box's .dat files (their
    header's instrumental function), the dialog's answers, the search in its
    thread, the results written and the window opened; the app is free."""
    import shiboken6
    from PySide6.QtWidgets import QApplication, QDialog
    import syncmoss.components_window as cw
    import syncmoss.experimental_features as ef
    app = physics_app
    app.pool = SerialPool()
    app.jn0_input.setText(str(JN))
    header, method = iio.build_dat_metadata_lines(app)
    assert method == 'SMS'
    files = _spectra_files(found, str(tmp_path / 'spectra'), header)
    app.process_path.setPlainText(repr(files))
    folder = str(tmp_path / 'components')
    warned = []
    monkeypatch.setattr(ef.QMessageBox, 'warning', lambda parent, title, text: warned.append(text))

    def answer(dialog):
        dialog.folder.setText(folder)
        dialog.known.setCurrentText('nothing')
        assert dialog.by_names.isChecked()                  # the files are named X_Y
        dialog.choices = dialog._read()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(cw.SearchDialog, 'exec', answer)
    cw.find_components(app)
    app.components_thread.wait()
    QApplication.processEvents()
    assert len(warned) == 1 and "new feature" in warned[0]
    assert not app.inprogress
    window = app.components_window
    try:
        assert window.is_map and len(window.res.beta) == 2
        assert {_owner(window.res, ph) for ph in PHASES} == {0, 1}
        assert window.run['settings']['method_params']['source'] == 'file'
    finally:
        shiboken6.delete(window)
