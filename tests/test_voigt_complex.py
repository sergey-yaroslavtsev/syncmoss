"""The complex line shape of the thick components -- models.Voigt_complex and
Voigt_complex_easy -- where the Gaussian width is exactly 0.

The dispersive part of each of its shapes f is 2/gL (S f + sigma^2 df/dS),
sigma the Gaussian one; with sigma = 0 that is the dispersion of the
Lorentzian. Until 2026-11-12 the kernels divided by sigma and took 0.01 for a
zero one, which stayed in the result: a model jumped between the width 0 and
any width above it, and a fit whose Gaussian width started at 0 got a
derivative many orders too large. These tests hold the width 0 to what the
widths above it lead to.

Pure NumPy/Numba path -- no Qt, no pool.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.minimi_lib as mi
import syncmoss.models as m5

S = np.linspace(-3.0, 3.0, 601)
EPS = np.finfo(float).eps
STEP = EPS ** (1 / 3)                 # of a difference of second order: good to eps^(2/3) of the function's scale
ACCURACY = 1e3 * EPS ** (2 / 3)


@pytest.mark.parametrize('gL', [0.097, 0.3])
def test_no_gaussian_width_is_the_lorentzian_and_its_dispersion(gL):
    line = m5.Voigt_complex(gL, 0.0, S, 1.0, 1j)
    q = S ** 2 + (gL / 2) ** 2
    absorptive, dispersive = gL / 2 / np.pi / q, S / np.pi / q
    assert np.max(np.abs(line.real - absorptive)) < 100 * EPS * np.max(absorptive)
    assert np.max(np.abs(line.imag - dispersive)) < 100 * EPS * np.max(dispersive)


@pytest.mark.parametrize('kernel', [m5.Voigt_complex, m5.Voigt_complex_easy], ids=['pseudo', 'pseudo_easy'])
@pytest.mark.parametrize('gL', [0.097, 0.3])
def test_the_width_zero_is_where_the_widths_above_it_lead(kernel, gL):
    # 2 f(h) - f(2h): the line through two widths just above 0, taken to 0
    h = STEP * gL
    led_to = 2 * kernel(gL, h, S, 1.0, 1j) - kernel(gL, 2 * h, S, 1.0, 1j)
    at_zero = kernel(gL, 0.0, S, 1.0, 1j)
    assert np.max(np.abs(at_zero.imag)) > 0
    assert np.max(np.abs(at_zero - led_to)) < ACCURACY * np.max(np.abs(at_zero))


# A Doublet and a Sextet in one layer under a polarised beam: the golden case
# in which the dispersion shows most. Their Gaussian widths are the slots
# DOUBLET_WIDTH and SEXTET_WIDTH; the Sextet's field broadening, which adds to
# its width in quadrature, is taken out.
with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _GOLDEN = json.load(_f)
_E = np.linspace(_GOLDEN["E_lo"], _GOLDEN["E_hi"], _GOLDEN["E_n"])
_MIX = _GOLDEN["cases"]["mix_doublet_sextet"]
DOUBLET_WIDTH, SEXTET_WIDTH, SEXTET_FIELD_BROADENING = 4, 9 + 5, 9 + 12


def _spectrum(p, method='pseudo'):
    shipped = m5.COMPLEX_VOIGT_METHOD
    m5.COMPLEX_VOIGT_METHOD = method
    m5.DISPERSION_SIGN = +1.0
    try:
        return np.asarray(m5.TImod(_E, np.array(p, float), np.array(_MIX["model"]), _E, 0.0, 1.0,
                                   np.array([]), [], [], Met=-1, Mett=0), dtype=float)
    finally:
        m5.COMPLEX_VOIGT_METHOD = shipped


def _widths(doublet, sextet):
    p = np.array(_MIX["params"], float)
    p[DOUBLET_WIDTH], p[SEXTET_WIDTH], p[SEXTET_FIELD_BROADENING] = doublet, sextet, 0.0
    return p


def test_a_model_does_not_jump_at_the_gaussian_width_zero():
    assert _MIX["model"] == ["Doublet", "Sextet"]
    at_zero = _spectrum(_widths(0.0, 0.0))
    assert np.max(np.abs(at_zero - _spectrum(_widths(0.0, 0.0), method='off'))) > 1e-3     # the dispersion shows
    h = STEP * m5.NAT_WIDTH
    assert np.max(np.abs(at_zero - (2 * _spectrum(_widths(h, h)) - _spectrum(_widths(2 * h, 2 * h))))) < ACCURACY


def test_a_fit_starting_at_the_gaussian_width_zero_gets_its_derivative():
    # minimi_lib's own Jacobian column of the Sextet's width, from 0, against
    # the derivative just above 0 (one-sided, of second order). The forward
    # difference's step for a parameter at 0 is tiny, so the column carries
    # rounding that is a fraction of this derivative: what is held is that the
    # two agree to within the derivative's size -- it was many orders off.
    p = _widths(0.0, 0.0)
    fixed = np.array([k for k in range(len(p)) if k != SEXTET_WIDTH])
    column, inactive = mi.compute_jacobian(lambda _x, q: _spectrum(q), p, _E, fixed,
                                           np.array([[-1], [-1], [0]], dtype=float), _spectrum(p),
                                           [], np.array([], dtype=int))
    assert len(inactive) == 0 and column.shape == (1, len(_E))
    h = STEP * m5.NAT_WIDTH
    above = (-3 * _spectrum(_widths(0.0, h)) + 4 * _spectrum(_widths(0.0, 2 * h))
             - _spectrum(_widths(0.0, 3 * h))) / (2 * h)
    assert np.max(np.abs(above)) > 0
    assert np.max(np.abs(column[0] - above)) < np.max(np.abs(above))
