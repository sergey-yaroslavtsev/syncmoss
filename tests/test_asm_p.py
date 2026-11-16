"""ASM's anharmonicity P(m) = sign(m) L/(L + ln 16), L = -ln(1 - |m|)
(models.ASM_modulus, models.ASM_thick_terms, legacy.asm_p_of_m).

In m the spectrum follows 1/K(m) ~ 1/ln(1 - |m|): every m the table shows as
1.000 gave another spectrum, and a fit could neither reach nor leave |m| = 1.
In P(m) it changes evenly up to the collinear ends +-1. Checked against closed
forms, a Sextet (the collinear end), finer sampling, the table's shown step,
the shipped expressions m <-> P(m), a fit, and the model files.

Pure NumPy/Numba path with the serial pool, except the last test (a window).
"""
import math
import os

import numpy as np
import pytest

import syncmoss.legacy as legacy
import syncmoss.minimi_lib as mi
import syncmoss.model_io as model_io
import syncmoss.models as m5
import syncmoss.models_dual as md
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.expression_eval import eval_expr
from syncmoss.expressions_io import list_expressions, substitute_expression
from syncmoss.parameters_table import SHOWN_DECIMALS
from syncmoss.support_math import calculate_expression_error

from conftest import FROZEN_PARAMETERS, SerialPool

JN = 16
X = np.linspace(-10.0, 10.0, 96)
POL = 0.9
EPS = np.finfo(float).eps
K = NB + legacy.ASM_P_SLOT                         # P(m) in p


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_MULCO, _X0 = _frozen("INSint.txt")
SOURCES = {
    'SMS': dict(Met=0, INS=np.atleast_1d(_frozen("INSexp.txt")), MulCo=float(_MULCO), x0=float(_X0), Norm=1.0),
    'CMS': dict(Met=1, INS=float(_frozen("GCMS.txt", delimiter='\t')), MulCo=MulCoCMS, x0=0.0, Norm=0.995),
}
BASELINE = [1.0e4, 0.3, 0.02, -0.15, 3.0e3, -0.2, 0.01, 0.1]
#      T    delta eps_m eps_l His  Han  WL     WG    P(m) theta phi   A    Num   I13  omega
ASM = [5.0, 0.1, 0.02, 0.2, 30.0, 5.0, 0.098, 0.12, 0.3, 35.0, 50.0, 0.6, 25.0, 3.0, 20.0]

_SHIPPED = {e['name']: e['expression']
            for e in list_expressions(os.path.join(os.path.dirname(m5.__file__), 'Expressions'))[0]}
M_TEXT = _SHIPPED['ASM m from P(m)']                  # {P}: P(m)
P_TEXT = _SHIPPED['ASM P(m) from m']                  # {m}: m


def _p(P, num=None, row=ASM):
    p = np.array(BASELINE + row, dtype=float)
    p[K] = P
    if num is not None:
        p[NB + 12] = num
    return p


def _ti(p, source='SMS', model=('ASM',)):
    s = SOURCES[source]
    return np.asarray(m5.TI(X, p, list(model), JN, SerialPool(), s['x0'], s['MulCo'], s['INS'], [0], [0],
                            Met=s['Met'], Norm=s['Norm'], pol=POL), dtype=float)


def test_p_of_m_gives_k_prime_exactly_and_h_is_pi_over_2k():
    for m in (0.0, 0.3, 0.9, 0.999999, 1 - 1e-12):
        kc2, h = m5.ASM_modulus(legacy.asm_p_of_m(m))
        assert kc2 == pytest.approx(1 - m, rel=1e-12)
        assert h == pytest.approx(np.pi / 2 / m5.K_cei(m), rel=1e-12)
        assert m5.ASM_modulus(-legacy.asm_p_of_m(m)) == (kc2, h)
    # past k'^2 = 2**-52 the quarter period is its limit ln(4/k') = ln 4/(1 - |P|) (A&S 17.3.26)
    for P in (0.95, 0.999):
        assert m5.ASM_modulus(P) == (pytest.approx(16.0 ** (-P / (1 - P)), rel=1e-12),
                                     pytest.approx(np.pi * (1 - P) / 2 / math.log(4.0), rel=1e-15))
    assert m5.ASM_modulus(1.0) == (0.0, 0.0) and m5.ASM_modulus(-1.0) == (0.0, 0.0)
    # the join of the two (k'^2 = 2**-52 at |P| = 13/14): no step
    step = 1e-9
    below, above = m5.ASM_modulus(13 / 14 - step)[1], m5.ASM_modulus(13 / 14 + step)[1]
    slope = np.pi / 2 / math.log(4.0)                                  # -dh/dP there
    assert below - above == pytest.approx(2 * step * slope, rel=1e-5)


@pytest.mark.parametrize('source', sorted(SOURCES))
def test_the_collinear_end_is_the_sextet_of_the_easy_axis(source):
    # P(m) = 1: every moment along u, a Sextet of H = His + Han and eps = eps_m + eps_l there
    T, delta, eps_m, eps_l, His, Han, WL, WG = ASM[:8]
    theta, phi, A, I13 = ASM[9], ASM[10], ASM[11], ASM[13]
    #           T  delta  eps            H          WL  WG  theta  phi  A  Am   a16  a25  GH   I13
    sextet = [T, delta, eps_m + eps_l, His + Han, WL, WG, theta, phi, A, 0.0, 0.0, 0.0, 0.0, I13]
    reference = _ti(np.array(BASELINE + sextet, dtype=float), source, model=('Sextet',))
    assert np.max(np.abs(_ti(_p(1.0), source) - reference)) < 1e3 * EPS * np.max(reference)
    assert np.max(np.abs(_ti(_p(0.9), source) - reference)) > 1e-3 * np.ptp(reference)      # not a Sextet before


@pytest.mark.parametrize('P', [0.2, 0.62, 0.9, 0.99, -0.6])
def test_the_spectrum_has_converged_at_the_table_s_number_of_points(P):
    # The points are the moment's angle at equal steps, weighted by the time it spends there:
    # at Num 25 (lines swept by Han = 5 T) 2-3e-4 of the depth from Num 385, and 1e-7 at Num 49.
    # (Before, equal weights at equal steps along the period: 2-3e-2, halving with each doubling.)
    fine = _ti(_p(P, num=385))
    assert np.max(np.abs(_ti(_p(P, num=25)) - fine)) < 1e-3 * np.ptp(fine)


@pytest.mark.parametrize('P', [0.5, 0.9, 0.99, 0.999, 1.0, -0.999, -1.0])
def test_a_shown_step_of_p_of_m_moves_the_spectrum_less_than_one_of_delta(P):
    # The table shows SHOWN_DECIMALS decimals: a value typed as shown must give the spectrum shown
    step = 10.0 ** -SHOWN_DECIMALS
    p = _p(P)
    delta = p.copy()
    delta[NB + 1] += step
    inside = p.copy()
    inside[K] -= np.sign(P) * step
    assert np.max(np.abs(_ti(inside) - _ti(p))) < np.max(np.abs(_ti(delta) - _ti(p)))


def test_the_shipped_expressions_turn_m_and_p_of_m_into_each_other():
    # "ASM m from P(m)" with {P} = p[K], as loaded under an ASM row; "ASM P(m) from m" with a number
    m_of_p = substitute_expression(M_TEXT, {'{P}': f'p[{K}]'})
    for m in (-1.0, -0.999999, -0.3, 0.0, 0.3, 0.9, 0.97, 1 - 1e-12, 1.0):
        p = np.zeros(K + 1)
        p[K] = legacy.asm_p_of_m(m)
        with np.errstate(divide='ignore'):                             # 1/(1 - |P|) at the ends
            assert float(eval_expr(m_of_p, p)) == pytest.approx(m, abs=1e-15)
        assert m5.ASM_modulus(p[K])[0] == pytest.approx(1 - abs(m), rel=1e-12, abs=0)   # what the model takes
        if abs(m) < 1:                                                 # a published m is below 1
            P = float(eval_expr(substitute_expression(P_TEXT, {'{m}': repr(m)}), np.zeros(1)))
            assert P == pytest.approx(legacy.asm_p_of_m(m), rel=1e-14, abs=1e-300)


def test_a_fit_reaches_p_of_m_and_its_m_near_the_collinear_end():
    # In m a fit towards 0.97 could step onto the bound 1, where the column of m was 0, and
    # stay there. In P(m) it reaches the truth, and m back from P(m) -- the shipped
    # "ASM m from P(m)", its error by the results table's rule -- is the true m within its error.
    m_true = 0.97
    truth = _p(legacy.asm_p_of_m(m_true))
    counts = np.random.default_rng(5).poisson(_ti(truth)).astype(float)
    start = truth.copy()
    start[K] = legacy.asm_p_of_m(0.8)
    start[[NB, NB + 4, NB + 5]] *= [0.9, 1.01, 0.8]                  # T, His, Han
    free = [0, NB, NB + 1, NB + 4, NB + 5, K]
    fix = np.array([i for i in range(len(truth)) if i not in free])
    confu = np.array([[-1], [-1], [0]], dtype=float)
    lower, upper = np.full(len(truth), -np.inf), np.full(len(truth), np.inf)
    lower[K], upper[K] = -1.0, 1.0
    s = SOURCES['SMS']
    prepared = md.prepare(X, ['ASM'], JN, s['x0'], s['MulCo'], s['INS'], s['Met'], s['Norm'], POL)
    rows = md.fit_jacobian(prepared, SerialPool(), [0], [0], [0], fix, confu, [], np.array([], dtype=int))
    fitted, errors, hi2, covariance, _chi2 = mi.minimi_hi(
        lambda _x, q: _ti(np.asarray(q, dtype=float)), X, counts, start.copy(), fix=fix, confu=confu,
        bounds=np.array([lower, upper]), MI=20, MI2=10, nu0=2.618, tau0=1e-3, eps=1e-6, weights='P', jacobian=rows)
    assert abs(start[K] - truth[K]) > 3 * errors[K]                   # it had to move
    assert abs(fitted[K] - truth[K]) < 3 * errors[K] and abs(fitted[K]) < 1
    text = substitute_expression(M_TEXT, {'{P}': f'p[{K}]'})
    m_fit = float(eval_expr(text, fitted))
    m_error = calculate_expression_error(text, fitted, errors, covariance, fix)
    assert 0 < m_error < 1 - m_true and abs(m_fit - m_true) < 3 * m_error
    assert hi2 < 1.2


@pytest.mark.gui
def test_a_model_file_keeps_p_of_m_and_an_older_one_has_its_m_converted(physics_app, tmp_path):
    table = physics_app.params_table
    table.select_model(1, 'ASM')

    def field():
        return table.row_widgets[1].layout().itemAt(legacy.ASM_P_SLOT + 1).widget().layout().itemAt(1).widget()
    field().setText('0.73')
    _names, _colors, rows = model_io.model_file_rows(physics_app)
    assert [g[3] for g in rows[1]][:15] == [''] * legacy.ASM_P_SLOT + [legacy.ASM_P_NAME] + [''] * 6
    path = tmp_path / 'asm.mdl'
    model_io._save_model_to_file(physics_app, str(path))
    text = path.read_text(encoding='utf-8')
    assert text.count(f'\t{legacy.ASM_P_NAME}\t') == 1
    older = tmp_path / 'asm_before_p.mdl'                             # the same file before P(m): m = 0.73
    older.write_text(text.replace(f'\t{legacy.ASM_P_NAME}\t', '\t\t'), encoding='utf-8')
    model_io.load_model_from_path(physics_app, str(older))
    assert float(field().text()) == legacy.asm_p_of_m(0.73)
    model_io.load_model_from_path(physics_app, str(path))
    assert field().text() == '0.73'
