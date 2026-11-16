"""models_dual.TI: the spectrum of models.TI, and its exact derivatives.

Every case is computed twice: by the untouched models.TI -- the value, and
central differences in every fitted parameter -- and by models_dual.TI on the
same parameters, plain (the value) and seeded as Duals (the Jacobian). The
sources are the tests' frozen SMS and CMS instrumental functions.

Pure NumPy/Numba path: the serial pool, a few nodes and points.
"""
import os

import numpy as np
import pytest

import syncmoss.dual_numbers as dn
import syncmoss.minimi_lib as mi
import syncmoss.models as m5
import syncmoss.models_dual as md
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import number_of_baseline_parameters as NB, numco

from conftest import FROZEN_PARAMETERS, SerialPool

JN = 16
X = np.linspace(-10.0, 10.0, 96)
POL = 0.9
EPS = np.finfo(float).eps
STEP = EPS ** (1 / 3)                 # of a central difference: good to eps^(2/3) of the function's scale
ACCURACY = 1e3 * EPS ** (2 / 3)


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_MULCO, _X0 = _frozen("INSint.txt")
SOURCES = {
    # Norm: any number serves (the same goes to both); the CMS one is not 1, so that its term is there
    'SMS': dict(Met=0, INS=np.atleast_1d(_frozen("INSexp.txt")), MulCo=float(_MULCO), x0=float(_X0), Norm=1.0),
    'CMS': dict(Met=1, INS=float(_frozen("GCMS.txt", delimiter='\t')), MulCo=MulCoCMS, x0=0.0, Norm=0.995),
}

#           Ns     Os   c2    lin    Nnr   Os'  c2'   lin'
BASELINE = [1.0e4, 0.3, 0.02, -0.15, 3.0e3, -0.2, 0.01, 0.1]
SINGLET = [3.0, 0.4, 0.098, 0.15]
DOUBLET = [6.0, 0.2, 0.8, 0.098, 0.2, 35.0, 50.0, 0.6, 1.1]
#          T    delta  eps    H     WL     WG    theta phi   A    Am   a16   a25    GH   I13
SEXTET = [8.0, 0.1, -0.05, 33.0, 0.098, 0.12, 40.0, 60.0, 0.6, 0.5, 0.02, -0.01, 0.3, 3.0]
POWDER = [8.0, 0.1, -0.05, 33.0, 0.098, 0.12, 54.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
MDGD = [8.0, 0.1, -0.05, 33.0, 0.098, 0.12, 0.3, 0.1, 0.1, 0.1, 40.0, 60.0, 0.6, 0.5, 0.02, -0.01, 3.0]
SPECTRUM = [5.0, 0.2, 0.1, 0.0]
REFERENCE_LINES = "1.5, -1.0, 0.0, 0.5, 0.3, 0.1, 2.0, 2.5, 0.05"
#           T    d1    Q1    H1    d2    Q2    H2     WL   theta phi   A    Am   rate  R
RELAX_2S = [5.0, 0.1, -0.05, 33.0, 0.05, 0.05, -30.0, 0.12, 40.0, 60.0, 0.6, 0.5, 0.3, 1.2]
#           T    delta eps    H     W     theta phi   A    R    alfa S
RELAX_MS = [5.0, 0.1, -0.05, 33.0, 0.098, 40.0, 60.0, 0.6, 0.5, 1.0, 5.0]
#      T    delta eps_m eps_l His  Han   WL     WG    P(m) theta phi   A    Num   I13  omega
ASM = [5.0, 0.1, 0.02, 0.2, 30.0, 5.0, 0.098, 0.12, 0.3, 35.0, 50.0, 0.6, 25.0, 3.0, 20.0]
#       T    delta eps   H0    WL   WG   theta phi   A    Am   KdH    KeH    Phi   h1   h3   h5 ... h15 ...   d2 ... d8   N/G I13
SCDW = [5.0, 0.1, 0.05, 30.0, 0.3, 0.2, 35.0, 50.0, 0.7, 0.6, 0.003, 0.005, 25.0, 5.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.03, 0.0, 0.0, 0.0, 12, 3.0]
#              T    delta Q    H     WL     WG    eta  tet   phi   tetr  phir  alfak A     Am   Ah
HAMILTONIAN = [5.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55]
VARIABLES = [33.0, 1.5, 0.01] + [0.0] * (numco - 3)        # a PDF's centre and sigma, a Corr's slope
V0 = NB                                                    # where a model's first row starts in p


def _case(model, rows, fixed=(), Distri=(0,), Cor=(0,), Recon=(0,)):
    """A model, its parameters after the baseline, and the slots (counted from
    the first row) that are not fitted: indices, counts, a text's placeholder,
    and a component's parameters that its distribution replaces."""
    return dict(model=list(model), p=np.array(BASELINE + [v for row in rows for v in row], dtype=float),
                fixed=[V0 + s for s in fixed], Distri=list(Distri), Cor=list(Cor), Recon=list(Recon))


CASES = {
    'Singlet': _case(['Singlet'], [SINGLET]),
    'Doublet': _case(['Doublet'], [DOUBLET]),
    'Sextet': _case(['Sextet'], [SEXTET]),
    'Sextet powder': _case(['Sextet'], [POWDER]),
    'MDGD': _case(['MDGD'], [MDGD]),
    'Spectrum': _case(['Spectrum'], [SPECTRUM], fixed=[3], Distri=[REFERENCE_LINES]),
    'layers': _case(['Doublet', 'Layer', 'Sextet', 'Layer', 'Singlet', 'Doublet'], [DOUBLET, SEXTET, SINGLET, DOUBLET]),
    'rows of slots': _case(['Variables', 'Singlet', 'Expression', 'Sextet'], [VARIABLES, SINGLET, [0.7], POWDER],
                           fixed=range(3, numco)),
    # H of a Sextet over 7 values, the PDF and the Corr of delta reading the Variables row
    'Distr and Corr': _case(['Variables', 'Sextet', 'Distr', 'Corr'], [VARIABLES, SEXTET, [3, 29.0, 37.0, 7, 0], [1, 0]],
                            fixed=list(range(3, numco)) + [numco + s for s in (1, 3, 14, 17, 18, 19, 20)],
                            Distri=[f"exp(-(X-p[{V0}])**2/2/p[{V0 + 1}]**2)"], Cor=[f"0.1+p[{V0 + 2}]*(X-33)"]),
    # ... and its width over 3 more: a product of the two axes
    'two distributions': _case(['Variables', 'Sextet', 'Distr', 'Corr', 'Distr'],
                               [VARIABLES, SEXTET, [3, 29.0, 37.0, 5, 0], [1, 0], [5, 0.05, 0.25, 3, 0]],
                               fixed=list(range(3, numco)) + [numco + s for s in (1, 3, 5, 14, 17, 18, 19, 20, 21, 24, 25)],
                               Distri=[f"exp(-(X-p[{V0}])**2/2/p[{V0 + 1}]**2)", "1+X"], Cor=[f"0.1+p[{V0 + 2}]*(X-33)"]),
    'Distr of a Spectrum': _case(['Spectrum', 'Distr'], [SPECTRUM, [1, -0.3, 0.6, 5, 0]], fixed=[1, 3, 4, 7, 8],
                                 Distri=[REFERENCE_LINES, "exp(-X**2)"]),
    'Recon': _case(['Sextet', 'Recon', 'Corr'], [POWDER, [3, 29.0, 37.0, 6, 0, 0, 0], [1, 0]],
                   fixed=[1, 3, 14, 17, 18, 19, 20, 21, 22], Cor=["0.1+0.02*(X-33)"],
                   Recon=[np.array([0.5, 1.0, 2.0, 3.0, 1.5, 0.4])]),
    'Relax_2S': _case(['Relax_2S'], [RELAX_2S]),
    'Relax_MS': _case(['Relax_MS'], [RELAX_MS], fixed=[10]),                       # the spin S: a count
    'ASM': _case(['ASM'], [ASM], fixed=[12]),                                      # its number of points
    # where K turns from the AGM to its limit ln 4/(1 - |P|) (k'^2 = 2**-52): the differences straddle it
    'ASM at the join of K': _case(['ASM'], [ASM[:8] + [13 / 14] + ASM[9:]], fixed=[12]),
    'ASM across the easy axis': _case(['ASM'], [ASM[:8] + [-0.6] + ASM[9:]], fixed=[12]),
    'SCDW': _case(['SCDW'], [SCDW], fixed=[25]),                                   # its grid's steps per width
    'Hamiltonian': _case(['Hamiltonian'], [HAMILTONIAN]),
    'Hamiltonian powder': _case(['Hamiltonian'], [HAMILTONIAN[:12] + [0.0, 0.0, 0.0]]),
    'Hamilton_mc': _case(['Hamilton_mc'], [HAMILTONIAN[:12]]),
    'Hamilton_pc': _case(['Hamilton_pc'], [HAMILTONIAN[:9]]),
    # a field over 4 values: the Hamiltonian's lines copy by copy
    'Distr of a Hamiltonian': _case(['Hamiltonian', 'Distr'], [HAMILTONIAN, [3, 30.0, 36.0, 4, 0]],
                                    fixed=[3, 15, 18, 19], Distri=["exp(-(X-33)**2/8)"]),
    # no field: the levels coincide in pairs, each pair's lines come as their sum and first moment
    'Hamiltonian, no field': _case(['Hamiltonian'], [HAMILTONIAN[:3] + [0.0] + HAMILTONIAN[4:]]),
    'Hamilton_pc, no field': _case(['Hamilton_pc'], [HAMILTONIAN[:3] + [0.0] + HAMILTONIAN[4:9]]),
    # ... and as the first of 4 values of a field, whose two ends are fitted
    'Distr of a Hamiltonian from no field': _case(['Hamiltonian', 'Distr'], [HAMILTONIAN, [3, 0.0, 6.0, 4, 0]],
                                                  fixed=[3, 15, 18, 19], Distri=["1+X"]),
    'Distr of a Relax_2S': _case(['Relax_2S', 'Distr', 'Corr'], [RELAX_2S, [12, 0.1, 0.6, 4, 0], [1, 0]],
                                 fixed=[1, 12, 14, 17, 18, 19, 20], Distri=["1+X"], Cor=["0.1+0.2*X"]),
    # a Doublet above the row, not scaled; a Sextet and a Singlet below it, over 4 thicknesses
    'Distr_T': _case(['Doublet', 'Distr_T', 'Sextet', 'Singlet'], [DOUBLET, [3.0, 0.2, 4.0, 4, 0], SEXTET, SINGLET],
                     fixed=[9 + 3, 9 + 4], Distri=["X*exp(-X)"]),
}
# The step of the central differences of models.TI and what they are then good to, where it is not (STEP,
# ACCURACY). SCDW's grid gives the model a kink wherever one of the wave's 2000 positions crosses a node, and
# the Lorentzian width, which sets the grid's step, moves all of them: with STEP its difference is off by 1e-3
# of the derivative. A step with no crossing inside it is limited by rounding instead.
REFERENCE = {'SCDW': (1e-7, 1e-5)}


# Under a polarised beam every fitted parameter of these moves the spectrum: the axis, the texture, the polar order
TEXTURED = ('Doublet', 'Sextet', 'MDGD', 'layers', 'Distr and Corr', 'two distributions')


def _ti(case, source, p=None):
    s = SOURCES[source]
    return np.asarray(m5.TI(X, case['p'] if p is None else p, case['model'], JN, SerialPool(), s['x0'], s['MulCo'],
                            s['INS'], case['Distri'], case['Cor'], Met=s['Met'], Norm=s['Norm'], pol=POL,
                            Recon=case['Recon']), dtype=float)


def _prepared(case, source):
    s = SOURCES[source]
    return md.prepare(X, case['model'], JN, s['x0'], s['MulCo'], s['INS'], s['Met'], s['Norm'], POL)


def _central(f, p, k, step=STEP):
    h = step * max(abs(p[k]), 1.0)
    up, down = np.array(p, dtype=float), np.array(p, dtype=float)
    up[k] += h
    down[k] -= h
    return (f(up) - f(down)) / (2 * h)


@pytest.mark.parametrize('source', sorted(SOURCES))
@pytest.mark.parametrize('name', sorted(CASES))
def test_the_spectrum_and_its_derivatives_are_those_of_models_ti(name, source):
    case = CASES[name]
    reference = _ti(case, source)
    assert np.ptp(reference) > 0.05 * BASELINE[0]                           # lines, not a flat baseline
    prepared = _prepared(case, source)
    plain = md.TI(list(case['p']), prepared, SerialPool(), case['Distri'], case['Cor'], case['Recon'])
    assert isinstance(plain, np.ndarray)
    assert np.max(np.abs(plain - reference)) < 1e4 * EPS * np.max(reference)

    free = [k for k in range(len(case['p'])) if k not in case['fixed']]
    y = md.TI(dn.seed(case['p'], free), prepared, SerialPool(), case['Distri'], case['Cor'], case['Recon'])
    assert np.max(np.abs(y.v - reference)) < 1e4 * EPS * np.max(reference)
    J = dn.jacobian(y, free)
    step, accuracy = REFERENCE.get(name, (STEP, ACCURACY))
    differences = np.array([_central(lambda q: _ti(case, source, q), case['p'], k, step) for k in free])
    scale = np.max(np.abs(differences), axis=1)
    worst = np.max(np.abs(J - differences), axis=1)
    # every parameter's own row, to the accuracy of the differences on the spectrum's scale
    assert np.all(worst < accuracy * np.maximum(scale, np.max(reference))), \
        [(k, w, s) for k, w, s in zip(free, worst, scale) if not w < accuracy * max(s, np.max(reference))]
    if source == 'SMS' and name in TEXTURED:
        assert np.all(scale > 0), [k for k, s in zip(free, scale) if s == 0]       # every fitted parameter shows


def test_the_weights_of_a_recon_are_differentiated():
    case = CASES['Recon']
    prepared = _prepared(case, 'SMS')
    weights = case['Recon'][0]
    n = len(case['p'])
    # the weights after p, as in a fit, and the model's own parameters with them: inside TI the
    # normalised weights are parameters of their own, resolved into these at the end
    free = [k for k in range(n) if k not in case['fixed']] + list(range(n, n + len(weights)))
    seeds = dn.seed(list(case['p']) + list(weights), free)
    y = md.TI(seeds[:n], prepared, SerialPool(), case['Distri'], case['Cor'], [dn.array(seeds[n:])])
    assert set(y.idx) == set(free)                                 # no stand-in is left
    J = dn.jacobian(y, free)

    def spectrum(pw):
        s = SOURCES['SMS']
        return np.asarray(m5.TI(X, pw[:n], case['model'], JN, SerialPool(), s['x0'], s['MulCo'], s['INS'],
                                case['Distri'], case['Cor'], Met=s['Met'], Norm=s['Norm'], pol=POL, Recon=[pw[n:]]), float)
    start = np.concatenate([case['p'], weights])
    differences = np.array([_central(spectrum, start, k) for k in free])
    assert np.all(np.max(np.abs(differences[-len(weights):]), axis=1) > 0)      # every weight shows
    assert np.all(np.max(np.abs(J - differences), axis=1)
                  < ACCURACY * np.maximum(np.max(np.abs(differences), axis=1), np.max(spectrum(start))))


def test_two_spectra_each_with_its_source():
    # An Nbaseline model: a CMS spectrum and an SMS one of different lengths
    xs = [X[:60], X[10:]]
    model = ['Sextet', 'Nbaseline', 'Doublet', 'Singlet']
    p = np.array(BASELINE + SEXTET + [2.0e4, -0.1, 0.0, 0.2, 1.0e3, 0.0, 0.0, 0.0] + DOUBLET + SINGLET)
    order = ('CMS', 'SMS')
    args = dict(x0=[SOURCES[s]['x0'] for s in order], MulCo=[SOURCES[s]['MulCo'] for s in order],
                INS=[SOURCES[s]['INS'] for s in order], Met=[SOURCES[s]['Met'] for s in order],
                Norm=[SOURCES[s]['Norm'] for s in order])
    lengths = [len(x) for x in xs]

    def spectrum(q):
        return np.asarray(m5.TI(np.concatenate(xs), q, model, JN, SerialPool(), args['x0'], args['MulCo'], args['INS'],
                                [0], [0], Met=args['Met'], Norm=args['Norm'], pol=POL, lengths=lengths), float)
    prepared = md.prepare(np.concatenate(xs), model, JN, args['x0'], args['MulCo'], args['INS'], args['Met'],
                          args['Norm'], POL, lengths=lengths)
    free = list(range(len(p)))
    y = md.TI(dn.seed(p, free), prepared, SerialPool())
    reference = spectrum(p)
    assert y.v.shape == reference.shape == (sum(lengths),)
    assert np.max(np.abs(y.v - reference)) < 1e4 * EPS * np.max(reference)
    J = dn.jacobian(y, free)
    differences = np.array([_central(spectrum, p, k) for k in free])
    assert np.max(np.abs(J - differences)) < ACCURACY * np.max(reference)
    first = slice(0, lengths[0])
    assert not np.any(J[NB + len(SEXTET):, first]) and np.any(J[:NB + len(SEXTET), first])     # each its own parameters


def test_widths_that_start_at_zero_have_their_column():
    # |p| and the quadrature of the two widths at 0: the one-sided derivative,
    # what a forward difference gives -- not a zero column, which the fit
    # would take for a parameter that does nothing
    p = np.array(BASELINE + POWDER)
    p[V0 + 5] = p[V0 + 12] = 0.0
    case = dict(CASES['Sextet powder'], p=p)
    y = md.TI(dn.seed(p, [V0 + 5, V0 + 12]), _prepared(case, 'CMS'), SerialPool())
    J = dn.jacobian(y, [V0 + 5, V0 + 12])
    h = STEP * m5.NAT_WIDTH
    for row, k in zip(J, (V0 + 5, V0 + 12)):
        def at(width):
            q = p.copy()
            q[k] = width
            return _ti(case, 'CMS', q)
        one_sided = (-3 * at(0.0) + 4 * at(h) - at(2 * h)) / (2 * h)          # of second order, from 0 upwards
        assert np.max(np.abs(one_sided)) > 0
        assert np.max(np.abs(row - one_sided)) < ACCURACY * np.max(at(0.0)) / m5.NAT_WIDTH


@pytest.mark.parametrize('P', [1.0, -1.0])
def test_the_anharmonicity_of_an_asm_at_a_collinear_end_has_its_column(P):
    # P(m) = +-1 (the bounds): the share of the period off the collinear line goes
    # linearly to 0, so the slope from inside is finite -- a fit that lands there can leave
    p = np.array(BASELINE + ASM[:8] + [P] + ASM[9:])
    case = dict(CASES['ASM'], p=p)
    k = V0 + 8
    row = dn.jacobian(md.TI(dn.seed(p, [k]), _prepared(case, 'SMS'), SerialPool()), [k])[0]

    def at(x):
        q = p.copy()
        q[k] = x
        return _ti(case, 'SMS', q)
    s = np.sign(P)
    inside = s * (3 * at(P) - 4 * at(P - s * STEP) + at(P - 2 * s * STEP)) / (2 * STEP)     # of second order
    assert np.max(np.abs(inside)) > 0.1 * np.ptp(at(P))
    assert np.max(np.abs(row - inside)) < ACCURACY * np.max(at(P))


def test_links_and_expressions_are_in_the_derivatives():
    # the Singlet's T linked to the Sextet's (x 0.5), the Expression slot read by nothing: a fit's vector
    case = CASES['rows of slots']
    p = case['p'].copy()
    singlet, expression, sextet = V0 + numco, V0 + numco + 4, V0 + numco + 5
    confu = np.array([[singlet], [sextet], [0.5]])
    Expr, NExpr = [f"p[{sextet + 3}]/10"], np.array([expression])
    free = [sextet, sextet + 1, sextet + 3]

    def spectrum(q):
        q = np.array(q, dtype=float)
        q[singlet] = 0.5 * q[sextet]
        q[expression] = q[sextet + 3] / 10
        return _ti(case, 'SMS', q)
    q = md.fit_parameters(spectrum_start := p, free, confu, Expr, NExpr)
    assert dn.value(q[singlet]) == 0.5 * p[sextet] and dn.value(q[expression]) == p[sextet + 3] / 10
    y = md.TI(q, _prepared(case, 'SMS'), SerialPool())
    J = dn.jacobian(y, free)
    differences = np.array([_central(spectrum, spectrum_start, k) for k in free])
    assert np.max(np.abs(J - differences)) < ACCURACY * np.max(y.v)
    unlinked = dn.jacobian(md.TI(dn.seed(p, free), _prepared(case, 'SMS'), SerialPool()), free)
    assert np.max(np.abs(unlinked[0] - J[0])) > 1e-3 * np.max(np.abs(J[0]))             # the link is in the row


def test_what_is_not_covered_is_named():
    assert md.uncovered(list(m5._COMPONENT_SLOTS) + ['Distr', 'Corr', 'Recon', 'Distr_T', 'Nbaseline', 'Variables',
                                                     'Expression', 'Layer']) == []            # every row TImod has
    assert md.uncovered(['Sextet', 'A new row']) == ['A new row']
    s = SOURCES['SMS']
    with pytest.raises(md.NotCovered):
        md.prepare(X, ['A new row'], JN, s['x0'], s['MulCo'], s['INS'], s['Met'], 1.0, POL)
    layered = _case(['Sextet', 'Layer', 'Distr_T', 'Doublet'], [SEXTET, [3.0, 0.2, 4.0, 3, 0], DOUBLET], Distri=["X"])
    with pytest.raises(md.NotCovered):
        md.TI(list(layered['p']), _prepared(layered, 'SMS'), SerialPool(), layered['Distri'])
    # a distribution of a slot that is not its component's, and a row with nothing to distribute
    for model, rows in ((['Singlet', 'Distr'], [SINGLET, [5, 0.0, 1.0, 3, 0]]),
                        (['Variables', 'Distr'], [VARIABLES, [0, 0.0, 1.0, 3, 0]])):
        case = _case(model, rows, Distri=["1+0*X"])
        with pytest.raises(md.NotCovered):
            md.TI(list(case['p']), _prepared(case, 'SMS'), SerialPool(), case['Distri'])


def test_a_fit_with_the_exact_jacobian_ends_where_the_differences_end():
    case = CASES['Doublet']
    truth = case['p'].copy()
    counts = np.random.default_rng(7).poisson(_ti(case, 'SMS', truth)).astype(float)
    start = truth.copy()
    start[V0:V0 + 3] *= [0.8, 1.3, 0.9]                                    # T, delta, eps
    start[0] *= 1.02
    # as a fit of the application has it: Nnr linked to Ns, so fixed for the minimiser
    confu = np.array([[4], [0], [truth[4] / truth[0]]])
    start[4] = confu[2][0] * start[0]
    fix = np.array([k for k in range(len(truth)) if k not in (0, V0, V0 + 1, V0 + 2, V0 + 4)])
    free = [k for k in range(len(truth)) if k not in fix]
    prepared = _prepared(case, 'SMS')
    exact_rows = md.fit_jacobian(prepared, SerialPool(), [0], [0], [0], fix, confu, [], np.array([], dtype=int))
    rows = exact_rows(start)
    assert rows.shape == (len(truth), len(X)) and not np.any(rows[fix]) and np.all(np.any(rows[free] != 0, axis=1))
    kw = dict(fix=fix, confu=confu, MI=20, MI2=10, tau0=1e-3, eps=1e-8, weights='P')
    by_differences = mi.minimi_hi(lambda _x, q: _ti(case, 'SMS', q), X, counts, start.copy(), **kw)
    exact = mi.minimi_hi(lambda _x, q: md.TI(list(q), prepared, SerialPool()), X, counts, start.copy(),
                         jacobian=exact_rows, **kw)
    assert not np.allclose(by_differences[0], start)
    assert exact[0][4] == pytest.approx(confu[2][0] * exact[0][0])
    assert np.allclose(exact[0], by_differences[0], rtol=1e-5, atol=1e-7)
    assert np.allclose(exact[1][free], by_differences[1][free], rtol=1e-3)                 # the errors
    assert exact[2] == pytest.approx(by_differences[2], rel=1e-6)


@pytest.mark.parametrize("lam, text", [(1000.0, "0*X"), (0.0, "X")])
def test_a_thickness_distribution_that_is_refused_gives_nan_as_models_ti(lam, text):
    case = _case(['Distr_T', 'Sextet'], [[lam, 0.0, 1.0, 3, 0.0], SEXTET], Distri=[text])
    assert np.all(np.isnan(_ti(case, 'SMS')))
    assert np.all(np.isnan(md.TI(list(case['p']), _prepared(case, 'SMS'), SerialPool(), case['Distri'])))


def test_the_widths_of_an_mdgd_that_start_at_zero_have_their_column():
    # Its six widths are the root of a quadratic form of the two widths
    # Guni and Gh: at 0 the one-sided derivative, as a forward difference
    p = np.array(BASELINE + MDGD)
    p[V0 + 5] = p[V0 + 6] = 0.0
    case = dict(CASES['MDGD'], p=p)
    J = dn.jacobian(md.TI(dn.seed(p, [V0 + 5, V0 + 6]), _prepared(case, 'CMS'), SerialPool()), [V0 + 5, V0 + 6])
    h = STEP * m5.NAT_WIDTH
    for row, k in zip(J, (V0 + 5, V0 + 6)):
        def at(width):
            q = p.copy()
            q[k] = width
            return _ti(case, 'CMS', q)
        one_sided = (-3 * at(0.0) + 4 * at(h) - at(2 * h)) / (2 * h)
        assert np.max(np.abs(one_sided)) > 0
        assert np.max(np.abs(row - one_sided)) < ACCURACY * np.max(at(0.0)) / m5.NAT_WIDTH


def test_the_field_of_a_hamiltonian_that_starts_at_zero_has_its_column():
    # With no field the levels coincide in pairs; the two lines of a pair have no slope in the
    # field, their sum has (the circular part, which a polarised beam through a thick absorber
    # shows). It is carried by the pair's first moment -- without it the column is not there.
    case = CASES['Hamiltonian, no field']
    H = V0 + 3
    row = dn.jacobian(md.TI(dn.seed(case['p'], [H]), _prepared(case, 'SMS'), SerialPool()), [H])[0]
    reference = _central(lambda q: _ti(case, 'SMS', q), case['p'], H)
    assert np.max(np.abs(reference)) > 1e-4 * BASELINE[0]                  # per tesla
    assert np.max(np.abs(row - reference)) < ACCURACY * np.max(_ti(case, 'SMS'))
    lines = md._hamiltonian(dn.seed(case['p'], [H])[V0:], SOURCES['SMS']['MulCo'], None, False)
    shape, _, _, P = lines[0]
    assert list(shape) == [md.VOIGT, md.SLOPE] * 4 and not np.any(dn.value(P)[1::2]) and np.any(dn.jacobian(P, [H])[0][1::2])


def test_a_parameter_at_a_symmetric_point_keeps_the_column_the_fit_gives_it_now():
    # The Sextet's axis across the beam (theta = 90, no polar order): the
    # spectrum is even in theta about it, its exact slope there is 0. A zero
    # column would tell the fit that theta does nothing, for good. The
    # forward difference it uses now -- the curvature times the step, small
    # but there -- is taken instead: it is what lets the fit leave such a
    # point today. A parameter the model does not read (a free slot of the
    # Variables row) stays a zero row.
    case = CASES['rows of slots']
    p = case['p'].copy()
    sextet = V0 + numco + 5
    p[sextet + 6], p[sextet + 8], p[sextet + 9] = 90.0, 0.6, 0.0           # theta; a texture, no polar order
    theta, unread = sextet + 6, V0 + 1
    free = [sextet, theta, unread]
    prepared = _prepared(case, 'SMS')
    exact = dn.jacobian(md.TI(dn.seed(p, free), prepared, SerialPool()), free)
    assert np.max(np.abs(exact[1])) < 1e3 * EPS * np.max(np.abs(exact[0]))             # no slope: cos(90 deg) = 6e-17
    fix = np.array([k for k in range(len(p)) if k not in free])
    rows = md.fit_jacobian(prepared, SerialPool(), [0], [0], [0], fix, np.array([[-1], [-1], [0]]), [],
                           np.array([], dtype=int))(p)

    def model(_x, q):
        return _ti(dict(case, p=np.asarray(q)), 'SMS')
    now, inactive = mi.compute_jacobian(model, p, X, fix, np.array([[-1], [-1], [0]]), model(X, p), [],
                                        np.array([], dtype=int))
    assert list(inactive) == [unread] and not np.any(rows[unread])
    # theta's column as the fit has it now -- the curvature times the step, well above the model's last digits
    # over the step -- and as it gets it
    last_digits = 8 * EPS * np.max(model(X, p)) / (p[theta] / mi.JACOBIAN_STEP_DIVISOR)
    assert np.max(np.abs(now[1])) > 100 * last_digits
    assert np.allclose(rows[theta], now[1], rtol=1e-3, atol=last_digits)
    assert np.max(np.abs(exact[1])) < 1e-6 * np.max(np.abs(rows[theta]))
    assert np.allclose(rows[sextet], now[0], rtol=1e-4)
