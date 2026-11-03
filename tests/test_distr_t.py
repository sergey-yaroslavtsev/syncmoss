"""Distr_T, the sample's thickness distribution (syncmoss.thickness_distribution).

The model against its definition, written out here by hand on tiny grids (the
mathematics, not the physics): every patch a spectrum of its own -- one TI, the
components below the row with T X/<X>, the row above it as it is, its own
Ns_j = Ns w_j with w_j ~ P(X_j) e^(-X_j/lambda) -- and Nnr split over the
patches in two ways (static 0 and 1), which must not matter. Eight larger cases
are docs/thickness_distribution/part_p_distr_t_vs_brute.py. Then the axis
either way, NaN for a density without a mean, the shipped densities, the
components' curves, and the table: the row, its placement, read_model, the
checks before a fit, a fit. The kernel tests are NumPy/Numba with the serial
pool; the table tests use physics_app.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.fitting_io import create_subspectra
from syncmoss.instrumental_io import compute_norm
from syncmoss.model_io import read_model
from syncmoss.models_positions import mod_pos

from conftest import FROZEN_PARAMETERS, SerialPool

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _GOLDEN = json.load(_f)["cases"]
_SINGLET = [3.0] + list(_GOLDEN["Singlet"]["params"])[1:]
_SEXTET = [6.0] + list(_GOLDEN["Sextet"]["params"])[1:]       # 2x2 matrices, textured, Faraday
_DOUBLET = [2.0] + list(_GOLDEN["Doublet"]["params"])[1:]
_ABOVE = [0.4] + list(_GOLDEN["Doublet"]["params"])[1:]       # a Doublet above the row, where Be/KB go
JN = 8
_X = np.linspace(-6.4, 6.4, 17)
_BASE = [1.0e4, 0.3, 0.04, 0.2, 300.0, 0.1, 0.08, -0.15]       # Ns Os c2s lins Nnr Onr c2nr linnr
_GAMMA = "X**5.25*exp(-6.25*X/2)"            # a Gamma of CV 0.4 and mean 2 (in X)


def _gamma(X):
    return X ** 5.25 * np.exp(-6.25 * X / 2)


def _sms():
    ins = np.atleast_1d(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "INSexp.txt"), delimiter=' '))
    mulco, x0 = np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "INSint.txt"), delimiter=' ')
    return dict(x0=float(x0), MulCo=float(mulco), INS=ins, Met=0)


def _ti(p, model, Distri=(0,)):
    method = _sms()
    m5._NODE_SUMS.clear()
    norm = compute_norm(SerialPool(), JN, method)
    return np.asarray(m5.TI(_X, np.asarray(p, dtype=float), list(model), JN, SerialPool(), method['x0'],
                            method['MulCo'], method['INS'], list(Distri), [0], Met=method['Met'],
                            Norm=norm), dtype=float)


def _scaled(component, s):
    return [component[0] * s] + list(component[1:])


@pytest.mark.parametrize("static", [0, 1])
def test_the_model_is_the_sum_of_its_patches(static):
    """A Doublet above the row (not scaled), a Sextet and a Doublet below it,
    absorption, a baseline with slopes and curvatures. Each patch its own TI
    with its Ns_j and Nnr_j: Nnr in proportion to the shares (static 0) or
    equally (static 1)."""
    lam, L, R, num = 3.0, 0.2, 6.0, 3
    spread = _ti(_BASE + _ABOVE + [lam, L, R, num, 0.0] + _SEXTET + _DOUBLET,
                 ['Doublet', 'Distr_T', 'Sextet', 'Doublet'], [_GAMMA])

    X = np.linspace(L, R, num)                       # the patches, by hand
    P = _gamma(X)
    s = X / (np.sum(P * X) / np.sum(P))
    w = P * np.exp(-X / lam) / np.sum(P * np.exp(-X / lam))
    nnr = w if static == 0 else np.full(num, 1 / num)
    reference = 0
    for sj, wj, nj in zip(s, w, nnr):
        base = [_BASE[0] * wj] + _BASE[1:4] + [_BASE[4] * nj] + _BASE[5:]
        reference = reference + _ti(base + _ABOVE + _scaled(_SEXTET, sj) + _scaled(_DOUBLET, sj),
                                    ['Doublet', 'Sextet', 'Doublet'])
    assert np.max(np.abs(spread - reference)) < 1e-12 * _BASE[0]
    homogeneous = _ti(_BASE + _ABOVE + _SEXTET + _DOUBLET, ['Doublet', 'Sextet', 'Doublet'])
    assert np.max(np.abs(spread - homogeneous)) > 1e-3 * _BASE[0]       # the spread does change it


def test_the_axis_may_run_either_way():
    model = ['Distr_T', 'Sextet']
    forward = _ti(_BASE + [3.0, 0.5, 5.0, 4, 0.0] + _SEXTET, model, [_GAMMA])
    backward = _ti(_BASE + [3.0, 5.0, 0.5, 4, 0.0] + _SEXTET, model, [_GAMMA])
    assert np.max(np.abs(forward - backward)) < 1e-12 * _BASE[0]


@pytest.mark.parametrize("lam, text", [(1000.0, "0*X"), (0.0, "X")])
def test_no_mean_or_a_zero_attenuation_length_gives_nan(lam, text):
    """A fit refuses such a trial step (minimi_hi's non-finite guard)."""
    spread = _ti(_BASE + [lam, 0.0, 1.0, 3, 0.0] + _SEXTET, ['Distr_T', 'Sextet'], [text])
    assert np.all(np.isnan(spread))


_SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "syncmoss", "Expressions")
_NUM = 200


@pytest.mark.parametrize("name, values, L, R, cv", [
    # a centre 3 times the rim (docs/thickness_distribution.md, section 3.9)
    ("Thickness PDF, pressed window (membrane)", {}, 1.0, 3.0, 2 / (np.sqrt(3) * 4)),
    ("Thickness PDF, pressed window (clamped plate)", {'{rim}': '1'}, 1.0 + 2.0 / (2 * _NUM), 3.0,
     2 * np.sqrt(4 / 45) / (1 + 2 / 3)),
    ("Thickness PDF, Gamma", {'{CV}': '0.4', '{mean}': '2'}, 0.0, 2.0 * (1 + 5 * 0.4), 0.4),
])
def test_the_shipped_thickness_densities_give_their_width(name, values, L, R, cv):
    """Each saved Distr_T density, set up as its comment says, has the width
    (CV of the relative thickness over the area) it stands for."""
    from syncmoss.expressions_io import list_expressions, substitute_expression
    from syncmoss.thickness_distribution import thickness_points
    text = next(e for e in list_expressions(_SHIPPED)[0] if e['name'] == name)['expression']
    s, w = thickness_points(substitute_expression(text, values), np.zeros(1), np.inf, L, R, _NUM)
    assert np.isclose(np.sum(w * s), 1.0)                      # T is the mean
    assert np.sqrt(np.sum(w * (s - 1) ** 2)) == pytest.approx(cv, rel=0.03)


def test_the_components_curves_carry_the_distr_t():
    """create_subspectra puts the Distr_T in front of every component below it;
    the line markers still come from the component."""
    p = np.array(_BASE + _ABOVE + [3.0, 0.2, 6.0, 15, 0.0] + _SINGLET, dtype=float)
    Ps, Psm, Distri_t, _, _, _ = create_subspectra(['Doublet', 'Distr_T', 'Singlet'], [_GAMMA], [], p)
    assert Psm == [['Doublet'], ['Distr_T', 'Singlet']]
    assert np.array_equal(Ps[1], np.array(_BASE + [3.0, 0.2, 6.0, 15, 0.0] + _SINGLET))
    assert [text.strip() for text in Distri_t] == [_GAMMA]      # substituted, as a Distr's
    positions = mod_pos(Ps[1], Psm[1], _sms()['INS'])
    assert len(positions[0]) > 0


# --- the table ------------------------------------------------------------------

def _model_btn(pt, row):
    return pt.row_widgets[row].layout().itemAt(0).widget().layout().itemAt(1).widget()


def _cell(pt, row, col):
    param_widget = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    name = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
    fix = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
    value = param_widget.layout().itemAt(1).widget()
    return name, fix, value


@pytest.mark.gui
def test_the_row_and_what_read_model_makes_of_it(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Distr_T')
    pt.select_model(2, 'Singlet')
    names = [(_cell(pt, 1, c)[0].original_text or _cell(pt, 1, c)[0].text()) for c in range(5)]
    assert names == ['λ, X', 'L', 'R', 'Num', 'Probability density function']
    assert [_cell(pt, 1, c)[2].text() for c in range(5)] == ['1000', '0', '1', '20', 'X']
    assert all(_cell(pt, 1, c)[1].isChecked() for c in range(5))      # fixed by default
    assert not _cell(pt, 1, 3)[1].isEnabled()                          # Num cannot be freed
    assert _cell(pt, 1, 0)[1].isEnabled()                              # lambda can
    model, p, *_, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(physics_app)
    assert model == ['Distr_T', 'Singlet']
    assert list(p[NB:NB + 5]) == [1000.0, 0.0, 1.0, 20.0, 0.0]
    assert Distri == ['X'] and list(DistriN) == [NB + 4]


@pytest.mark.gui
def test_placement(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Distr_T')
    pt.select_model_by_button('Distr', _model_btn(pt, 2))
    assert _model_btn(pt, 2).text() == 'None'            # a Distr has no component there
    pt.select_model_by_button('Distr_T', _model_btn(pt, 2))
    assert _model_btn(pt, 2).text() == 'None'            # one per spectrum
    pt.select_model_by_button('Layer', _model_btn(pt, 2))
    assert _model_btn(pt, 2).text() == 'None'            # no Layer with a Distr_T
    pt.select_model(2, 'Sextet')
    pt.select_model(3, 'Distr')
    pt.select_model(3, 'Insert')                          # an empty row between the Sextet and its Distr
    pt.select_model_by_button('Distr_T', _model_btn(pt, 3))
    assert _model_btn(pt, 3).text() != 'Distr_T'         # it would cut the Distr off its Sextet
    pt.select_model(5, 'Nbaseline')
    pt.select_model_by_button('Distr_T', _model_btn(pt, 6))
    assert _model_btn(pt, 6).text() == 'Distr_T'         # the second spectrum takes its own


@pytest.mark.gui
def test_no_distr_t_with_a_layer(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Layer')
    pt.select_model_by_button('Distr_T', _model_btn(pt, 2))
    assert _model_btn(pt, 2).text() == 'None'


@pytest.mark.gui
def test_a_fit_fixes_the_text_slot_and_draws_the_component(physics_app, tmp_path, monkeypatch):
    """The whole fit path (the minimiser returns its start; the curves after it
    are computed for real): the PDF's placeholder is fixed, the Sextet's T is
    free, and the components' curves hold the Sextet alone."""
    from syncmoss import fitting_io
    from conftest import redirect_params_dir_to_tmp
    from test_fit_core import _capture_minimiser
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    pt = app.params_table
    pt.select_model(1, 'Distr_T')
    pt.select_model(2, 'Sextet')
    _cell(pt, 1, 1)[2].setText('0.2')
    _cell(pt, 1, 2)[2].setText('6')
    _cell(pt, 1, 4)[2].setText(_GAMMA)
    app.initialize_parameters()
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, app.calibration_path, SerialPool())

    assert result['success'], result.get('message')
    fix = {int(i) for i in np.ravel(calls[0]['fix'])}
    assert NB + 4 in fix and NB + 5 not in fix           # the PDF's slot fixed, the Sextet's T free
    assert len(result['FS']) == 1 and np.all(np.isfinite(np.asarray(result['FS'][0], dtype=float)))


@pytest.mark.gui
@pytest.mark.parametrize("case", ['valid', 'Layer', 'two', 'no mean', 'zero lambda'])
def test_the_checks_before_a_fit(physics_app, case):
    """What the menu refuses can still be built (a row added, then an Nbaseline
    deleted); Show model and Fit refuse it."""
    pt = physics_app.params_table
    pt.select_model(1, 'Distr_T')
    pt.select_model(2, 'Singlet')
    if case == 'Layer':
        pt.select_model(3, 'Layer')
    elif case == 'two':
        pt.select_model(3, 'Distr_T')
    elif case == 'no mean':
        _cell(pt, 1, 2)[2].setText('0')                   # L = R = 0: every patch has no thickness
    elif case == 'zero lambda':
        _cell(pt, 1, 0)[2].setText('0')
    assert physics_app.check_user_expressions("Fit") == (case == 'valid')
