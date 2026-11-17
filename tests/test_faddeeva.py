"""faddeeva.py and COMPLEX_VOIGT_METHOD = 'faddeeva', the shipped line shape:
the Faddeeva function against scipy's and against w taken in high-precision
decimal arithmetic, its compiled Voigt line, and the models with it against
the same models with 'wofz' (scipy) -- the line shape, every golden model, a
thick single crystal whose forbidden lines only the dispersion brings out --
and the scalar path of a component at its matrix path.

numba kernels, no window, no pool.
"""
import json
import os
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.special import wofz

import syncmoss.faddeeva as fd
import syncmoss.models as m5

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _GOLDEN = json.load(_f)
_E = np.linspace(_GOLDEN["E_lo"], _GOLDEN["E_hi"], _GOLDEN["E_n"])


def test_the_line_shape_shipped_is_the_faddeeva_one():
    assert m5.COMPLEX_VOIGT_METHOD == 'faddeeva' and m5.exact_voigt()


# ---------------------------------------------------------------------------
# w(z)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('imaginary', [1e-10, 1e-3, 0.04, 0.3, 1.0, 30.0, 300.0])
def test_the_faddeeva_function_is_scipys(imaginary):
    """Along a line of the upper half-plane, the centre and the far wings of a
    line (a Voigt line is w((S + i gL/2)/(sigma sqrt 2))); scipy is right to
    about 1e-14 itself."""
    real = np.concatenate((np.linspace(-30, 30, 2001), np.logspace(1.5, 5, 200), -np.logspace(1.5, 5, 200)))
    z = real + 1j * imaginary
    reference = wofz(z)
    assert np.max(np.abs(fd.faddeeva(z) - reference) / np.abs(reference)) < 1e-13


def _w_decimal(x, y):
    """w(x + iy) from its Taylor series sum (iz)^n / Gamma(n/2 + 1) in decimal
    arithmetic, with the digits its cancellation takes (~0.434 |z|^2) and 40
    more: right to far below the double's rounding, for |z| up to about 7."""
    r2 = x * x + y * y
    with localcontext() as ctx:
        ctx.prec = int(0.4343 * r2) + 40
        sqrt_pi = _decimal_pi().sqrt()
        u = (Decimal(-y), Decimal(x))                       # i z
        u2 = (u[0] * u[0] - u[1] * u[1], 2 * u[0] * u[1])
        tiny = Decimal(10) ** (-(ctx.prec - 5))
        even = total_even = (Decimal(1), Decimal(0))
        odd = total_odd = (2 * u[0] / sqrt_pi, 2 * u[1] / sqrt_pi)
        m = 1
        while m <= r2 + 10 or abs(even[0]) + abs(even[1]) + abs(odd[0]) + abs(odd[1]) >= tiny:
            even = ((even[0] * u2[0] - even[1] * u2[1]) / m, (even[0] * u2[1] + even[1] * u2[0]) / m)
            h = Decimal(m) + Decimal("0.5")
            odd = ((odd[0] * u2[0] - odd[1] * u2[1]) / h, (odd[0] * u2[1] + odd[1] * u2[0]) / h)
            total_even = (total_even[0] + even[0], total_even[1] + even[1])
            total_odd = (total_odd[0] + odd[0], total_odd[1] + odd[1])
            m += 1
        return complex(float(total_even[0] + total_odd[0]), float(total_even[1] + total_odd[1]))


def _decimal_pi():
    """pi to the context's precision (the decimal module's recipe)."""
    with localcontext() as ctx:
        ctx.prec += 5
        lasts, t, s, n, na, d, da = 0, Decimal(3), 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    return +s


def _expansion(z, terms):
    """w(z) by the expansion with *terms* terms (faddeeva.faddeeva's steps)."""
    A, L = fd.coefficients(terms)
    lz = L - 1j * z
    Z = (L + 1j * z) / lz
    p = np.full(Z.shape, A[0], dtype=complex)
    for coefficient in A[1:]:
        p = p * Z + coefficient
    return 2.0 * p / lz ** 2 + (1.0 / np.sqrt(np.pi)) / lz


# Near the real axis and 2.5 < Re z < 7 is where the expansion is worst
_POINTS = [(x, y) for y in (1e-10, 1e-3, 0.3, 2.0) for x in (0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5)]


def test_the_expansion_is_right_to_the_rounding():
    """Against w in high precision: FADDEEVA_TERMS is where the error has
    fallen to the rounding of the double arithmetic (8.6e-16 at most, measured
    on 297 points to |z| = 1e5); fewer terms are measurably worse."""
    z = np.array([complex(x, y) for x, y in _POINTS])
    reference = np.array([_w_decimal(x, y) for x, y in _POINTS])
    error = np.abs(fd.faddeeva(z) - reference) / np.abs(reference)
    assert np.max(error) < 2e-15
    assert np.max(np.abs(_expansion(z, fd.FADDEEVA_TERMS - 8) - reference) / np.abs(reference)) > 1e-14
    assert np.max(np.abs(_expansion(z, fd.FADDEEVA_TERMS) - fd.faddeeva(z)) / np.abs(reference)) < 1e-15


# ---------------------------------------------------------------------------
# the compiled line
# ---------------------------------------------------------------------------
def _wofz_line(gL, gG, S):
    if gG <= 0:
        return (1 / np.pi) / (gL / 2 - 1j * S)
    sigma = gG / np.sqrt(8 * np.log(2))
    return wofz((S + 0.5j * gL) / (sigma * np.sqrt(2))) / (sigma * np.sqrt(2 * np.pi))


@pytest.mark.parametrize('n', [1, 63, 64, 65, 300])
@pytest.mark.parametrize('gL, gG', [(0.097, 0.15), (0.097, 0.0), (0.0, 0.3), (0.3, 1e-9), (0.02, 3.0)])
def test_the_line_over_chunks_of_any_length(gL, gG, n):
    """The Voigt line with its dispersion, in chunks of 64 points: a Lorentzian
    without a Gaussian, a Gaussian (and its Dawson-function dispersion) without
    a Lorentzian, a Gaussian of 1e-9."""
    S = np.linspace(-9.0, 9.0, n) if n > 1 else np.array([0.37])
    reference = _wofz_line(gL, gG, S)
    got = fd.voigt_complex(gL, gG, S)
    assert got.shape == S.shape and got.dtype == complex
    assert np.max(np.abs(got - reference)) < 1e-13 * np.max(np.abs(reference))
    assert np.array_equal(fd.voigt(gL, gG, S), got.real)


def test_the_line_takes_points_of_any_shape():
    S = np.linspace(-5.0, 5.0, 300).reshape(20, 15)
    assert np.array_equal(fd.voigt_complex(0.1, 0.2, S), fd.voigt_complex(0.1, 0.2, S.ravel()).reshape(20, 15))
    assert fd.voigt(0.1, 0.2, 0.5).shape == ()


# ---------------------------------------------------------------------------
# the models with it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('gL, gG', [(0.097, 0.0), (0.097, 0.15), (0.3, 0.02), (0.02, 0.6), (0.097, 2.0)])
def test_the_models_line_shape_is_scipys(gL, gG, monkeypatch):
    S = np.linspace(-20.0, 20.0, 2001)
    shipped = m5.Voigt_c(gL, gG, S)
    monkeypatch.setattr(m5, 'COMPLEX_VOIGT_METHOD', 'wofz')
    reference = m5.Voigt_c(gL, gG, S)
    assert np.max(np.abs(shipped - reference)) < 1e-13 * np.max(np.abs(reference))
    assert np.min(reference.imag[S > 0]) > 0                   # the dispersion's sign: D > 0 above the line


@pytest.mark.parametrize('method', ['faddeeva', 'wofz', 'pseudo', 'off'])
def test_a_thin_line_is_the_real_part_of_a_thick_one(method, monkeypatch):
    """Voigt_r (Singlet, Spectrum, the Hamiltonian's scalar path) is Voigt_c's
    real part, bit for bit: a component's scalar path is its matrix path."""
    monkeypatch.setattr(m5, 'COMPLEX_VOIGT_METHOD', method)
    S = np.linspace(-6.0, 6.0, 301)
    for gL, gG in ((0.097, 0.15), (0.097, 0.0)):
        assert np.array_equal(m5.Voigt_r(gL, gG, S), m5.Voigt_c(gL, gG, S).real)


def _timod(model, params, pol=0.98, E=_E):
    return np.asarray(m5.TImod(E, np.array(params, float), np.array(model), E, 0.0, 1.0, np.array([]), [], [],
                               Met=-1, Mett=0, sms_pol=pol), dtype=float)


@pytest.mark.parametrize('name', sorted(_GOLDEN["cases"]))
def test_every_golden_model_is_its_scipy_spectrum(name, monkeypatch):
    """Every model of tests/golden_models.json with the shipped line shape and
    with scipy's (4e-16 at most, measured)."""
    case = _GOLDEN["cases"][name]
    shipped = _timod(case["model"], case["params"])
    monkeypatch.setattr(m5, 'COMPLEX_VOIGT_METHOD', 'wofz')
    assert np.max(np.abs(shipped - _timod(case["model"], case["params"]))) < 1e-12


#          T    delta Q    H     WL     WG    eta  thetaH phiH theta phi alpha_k
_HAM = [8.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0, 70.0]


def test_the_hamiltonians_scalar_path_meets_its_matrix_path():
    """A random powder (A = Ah = 0) takes the scalar path, with the thin line
    shape; an order of 1e-9 already the matrix path. The two differ by the
    order alone, not by a line shape (with the exact Voigt in the matrix path
    and the pseudo-Voigt in the scalar one they would by ~1e-4)."""
    powder = _timod(['Hamiltonian'], _HAM + [0.0, 0.0, 0.0])
    ordered = _timod(['Hamiltonian'], _HAM + [1e-9, 0.0, 1e-9])
    assert 0 < np.max(np.abs(ordered - powder)) < 1e-8


# A magnetised single crystal (A = Am = 1) with its field 20 deg off the beam
# and across the polarisation (phi_h = 90): the Delta m = 0 lines 2 and 5 are
# forbidden in the thin limit. At a thickness of hundreds the Faraday rotation
# -- the dispersive wings of the Delta m = +-1 lines -- has turned the
# polarisation and they are there, deep; without the dispersion they are not.
#              T    delta eps   H    WL     WG   theta  phi   A    Am   a+   a-   GH   I13
_SINGLE_CRYSTAL = [1.0, 0.0, 0.0, 33.0, 0.098, 0.1, 20.0, 90.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]
_LINE_2, _BETWEEN_1_2 = -3.08, -4.2                     # mm/s, alpha-Fe at 33 T
_E_FINE = np.linspace(-7.0, 7.0, 1401)


def _line_2(T):
    """Line 2's dip: the transmission between lines 1 and 2 minus the least
    within 0.25 mm/s of line 2."""
    p = list(_SINGLE_CRYSTAL)
    p[0] = T
    y = _timod(['Sextet'], p, pol=1.0, E=_E_FINE)
    return float(np.interp(_BETWEEN_1_2, _E_FINE, y)) - float(y[np.abs(_E_FINE - _LINE_2) < 0.25].min()), y


def test_the_dispersion_brings_out_the_forbidden_lines(monkeypatch):
    """Measured: the dip is below 0 (no line) up to T ~ 50, 0.47 at T = 1000;
    without the dispersion ('off') it stays below 0 (-0.18 at T = 1000). The
    same spectrum with scipy's line shape to rounding (2.5e-14 at T = 1000)."""
    thin, _ = _line_2(1e-3)
    assert thin < 0
    thick, shipped = _line_2(1000.0)
    assert thick > 0.3
    monkeypatch.setattr(m5, 'COMPLEX_VOIGT_METHOD', 'wofz')
    _, reference = _line_2(1000.0)
    assert np.max(np.abs(shipped - reference)) < 1e-11
    monkeypatch.setattr(m5, 'COMPLEX_VOIGT_METHOD', 'off')
    assert _line_2(1000.0)[0] < 0
