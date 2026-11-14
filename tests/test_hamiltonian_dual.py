"""hamiltonian_dual.Ham_mosaic: models.Ham_mosaic's eight lines for arguments
that may be Duals. The values against the kernel; the derivatives against
central differences of the kernel -- line by line where the levels are
apart, and of the lines summed as a spectrum where they come close, since a
close pair's two lines have large derivatives of opposite sign that only
their sum resolves. A pair that coincides comes as its sum and its first
moment, which only the summed spectrum can be compared with.

Pure NumPy/Numba path.
"""
import numpy as np
import pytest

import syncmoss.dual_numbers as dn
import syncmoss.hamiltonian_dual as hd
import syncmoss.models as m5

EPS = np.finfo(float).eps
STEP = EPS ** (1 / 3)                 # of a central difference: good to eps^(2/3) of the function's scale
ACCURACY = 1e3 * EPS ** (2 / 3)
E = np.linspace(-8.0, 8.0, 401)
FREE = range(11)

#  Q    H     eta  phi   tet   tetr  phir  alfak A     Am   Ah
APART = {
    'mosaic': (0.4, 33.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
    'single crystal': (0.4, 33.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 1.0, 1.0, 1.0),
    'powder': (0.4, 33.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.0, 0.0, 0.0),
    'strong quadrupole': (2.0, 5.0, 0.6, 10.0, 70.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
    'no quadrupole': (0.0, 33.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
}
CLOSE = {
    'weak field 0.5 T': (1.0, 0.5, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
    'weak field 1e-5 T': (1.0, 1e-5, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
    'small quadrupole': (1e-4, 33.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
}
# Pairs whose distance is at the rounding of the levels (or 0): how a pair's intensity is
# shared between its two lines is then rounding too, and only the summed lines are compared
SUMMED_ONLY = {
    'weak field 1e-9 T': (1.0, 1e-9, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
    'no field': (1.0, 0.0, 0.3, 30.0, 20.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55),
}
NO_FIELD = SUMMED_ONLY['no field']


def _lorentz(S):
    """A Lorentzian of the width 0.2 mm/s at every position S, and its derivative in the position."""
    L = 1 / ((E[None, :] - S[:, None]) ** 2 + 0.01)
    return L, 2 * (E[None, :] - S[:, None]) * L ** 2


def _summed(P, S, moment=np.zeros(8, dtype=bool)):
    """The lines as a spectrum of 2x2 matrices; a first moment multiplies the shape's slope."""
    L, L_S = _lorentz(S)
    return np.sum(P[:, None, :, :] * np.where(moment[:, None], L_S, L)[:, :, None, None], axis=0)


def _summed_slope(P, S, dP, dS, moment=np.zeros(8, dtype=bool)):
    """The derivative of _summed, from those of the lines; a first moment is 0 and has only dP."""
    L, L_S = _lorentz(S)
    assert not np.any(P[moment])
    return np.sum(dP[:, None, :, :] * np.where(moment[:, None], L_S, L)[:, :, None, None]
                  + P[:, None, :, :] * (L_S * dS[:, None])[:, :, None, None], axis=0)


def _central(args, i, cms):
    """Central differences of the kernel in its argument i: of P, of S, and of the lines summed."""
    h = STEP * max(abs(args[i]), 1.0)
    up, down = list(args), list(args)
    up[i] += h
    down[i] -= h
    (Pu, Su), (Pd, Sd) = m5.Ham_mosaic(*up, cms), m5.Ham_mosaic(*down, cms)
    return (Pu - Pd) / (2 * h), (Su - Sd) / (2 * h), (_summed(Pu, Su) - _summed(Pd, Sd)) / (2 * h)


@pytest.mark.parametrize('cms', [False, True], ids=['SMS', 'CMS'])
@pytest.mark.parametrize('name', sorted(APART) + sorted(CLOSE))
def test_the_lines_are_the_kernels(name, cms):
    args = {**APART, **CLOSE}[name]
    P0, S0 = m5.Ham_mosaic(*args, cms)
    for arguments in (args, dn.seed(args, FREE)):               # on numbers, and on Duals
        P, S, moment = hd.Ham_mosaic(*arguments, cms)
        assert not np.any(moment)
        assert np.max(np.abs(dn.value(P) - P0)) < 100 * EPS * np.max(np.abs(P0))
        assert np.max(np.abs(dn.value(S) - S0)) < 100 * EPS * max(np.max(np.abs(S0)), 1.0)
    assert isinstance(hd.Ham_mosaic(*args, cms)[0], np.ndarray)


@pytest.mark.parametrize('cms', [False, True], ids=['SMS', 'CMS'])
def test_levels_that_coincide_are_their_sum_and_a_first_moment(cms):
    P0, S0 = m5.Ham_mosaic(*NO_FIELD, cms)
    assert np.max(np.abs(S0[1] - S0[0])) == 0                    # the levels do coincide
    for arguments in (NO_FIELD, dn.seed(NO_FIELD, FREE)):
        P, S, moment = hd.Ham_mosaic(*arguments, cms)
        assert list(moment) == [False, True, False, True] * 2    # two pairs, for either ground level
        assert np.max(np.abs(dn.value(S) - S0)) < 100 * EPS * max(np.max(np.abs(S0)), 1.0)
        assert not np.any(dn.value(P)[moment])                   # a moment is 0: the spectrum is the kernel's
        assert np.max(np.abs(dn.value(P)[~moment] - (P0[0::2] + P0[1::2]))) < 100 * EPS * np.max(np.abs(P0))


def test_no_quadrupole_and_no_field_is_the_kernels_one_line():
    args = (0.0, 0.0) + NO_FIELD[2:]
    P0, S0 = m5.Ham_mosaic(*args, False)
    P, S, moment = hd.Ham_mosaic(*dn.seed(args, FREE), False)
    assert not np.any(moment) and not np.any(dn.value(S)) and not np.any(S0)
    assert np.max(np.abs(dn.value(P) - P0)) < 100 * EPS * np.max(np.abs(P0))


@pytest.mark.parametrize('cms', [False, True], ids=['SMS', 'CMS'])
@pytest.mark.parametrize('name', sorted(APART))
def test_levels_apart_every_line_has_its_derivatives(name, cms):
    args = APART[name]
    P0, S0 = m5.Ham_mosaic(*args, cms)
    P, S, _ = hd.Ham_mosaic(*dn.seed(args, FREE), cms)
    dP, dS = dn.jacobian(P, FREE), dn.jacobian(S, FREE)
    moved = 0
    for i in FREE:
        P_i, S_i, _ = _central(args, i, cms)
        assert np.max(np.abs(dP[i] - P_i)) < ACCURACY * np.max(np.abs(P0))
        assert np.max(np.abs(dS[i] - S_i)) < ACCURACY * max(np.max(np.abs(S0)), 1.0)
        moved += bool(np.any(P_i != 0) or np.any(S_i != 0))
    assert moved >= (5 if name == 'powder' or cms else 9)        # not a check of zeros against zeros


@pytest.mark.parametrize('cms', [False, True], ids=['SMS', 'CMS'])
@pytest.mark.parametrize('name', sorted(CLOSE) + sorted(SUMMED_ONLY))
def test_close_levels_the_summed_lines_have_their_derivatives(name, cms):
    # 'no field': the pairs coincide, and the field that splits them is among the arguments
    args = {**CLOSE, **SUMMED_ONLY}[name]
    P0, S0 = m5.Ham_mosaic(*args, cms)
    P, S, moment = hd.Ham_mosaic(*dn.seed(args, FREE), cms)
    dP, dS = dn.jacobian(P, FREE), dn.jacobian(S, FREE)
    scale = np.max(np.abs(_summed(P0, S0)))
    assert np.max(np.abs(_summed(dn.value(P), dn.value(S), moment) - _summed(P0, S0))) < 100 * EPS * scale
    for i in FREE:
        reference = _central(args, i, cms)[2]
        # the reference's own limit: the Lorentzians' third derivative over the step
        slope = _summed_slope(dn.value(P), dn.value(S), dP[i], dS[i], moment)
        assert np.max(np.abs(slope - reference)) < 1e-6 * max(np.max(np.abs(reference)), scale)


def test_no_field_the_slope_in_the_field_is_the_first_moments():
    # No field: the pairs' half-distance is |H| times a number, with no slope at 0; the two lines
    # of a pair have no derivative in H, their sum has -- all of it in the first moment, which the
    # mosaic's SMS geometry shows (the circular part of the lines' matrices)
    P, S, moment = hd.Ham_mosaic(*dn.seed(NO_FIELD, FREE), False)
    dP, dS = dn.jacobian(P, FREE), dn.jacobian(S, FREE)
    reference = _central(NO_FIELD, 1, False)[2]
    scale = np.max(np.abs(_summed(*m5.Ham_mosaic(*NO_FIELD, False))))
    assert np.max(np.abs(reference)) > 1e-3 * scale
    with_moments = _summed_slope(dn.value(P), dn.value(S), dP[1], dS[1], moment)
    assert np.max(np.abs(with_moments - reference)) < 1e-6 * np.max(np.abs(reference))
    dP[1][moment] = 0
    without = _summed_slope(dn.value(P), dn.value(S), dP[1], dS[1], moment)
    assert np.max(np.abs(without - reference)) > 0.5 * np.max(np.abs(reference))
    # the other arguments do not split the pairs (Kramers): their moments have no slope
    for i in (0, 2, 3, 4):
        assert np.max(np.abs(dP[i][moment])) < 100 * EPS * np.max(np.abs(dn.value(P)))
