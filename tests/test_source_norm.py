"""How models.TI uses the source normalisation Norm ("SOURCE NORMALISATION" there).

CMS: the source's Lorentzian wings reach past the integration grid and the
photons out there are transmitted, so they are added as unabsorbed,
N0*(H*D + 1 - Norm) + N1. SMS: the grid holds the whole source, Norm - 1 is the
sum's own error, and it is divided out, H*N0/Norm*D + N1.

Pinned three ways: a CMS line against the exact transmission integral (what the
CMS form is for), both forms against TI with Norm = 1 (single spectrum, Nbaseline
sections, and a CMS + SMS mix), and the CMS instrumental-function search
following the G it fits. Pure NumPy/Numba, serial pool.
"""
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss import instrumental_io as iio
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import NAT_WIDTH, number_of_baseline_parameters as NB

from conftest import FROZEN_PARAMETERS


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_G = float(_frozen("GCMS.txt", delimiter='\t'))
_MULCO_SMS, _X0_SMS = _frozen("INSint.txt")
METHODS = {
    "CMS": dict(x0=0.0, MulCo=MulCoCMS, INS=_G, Met=1),
    "SMS": dict(x0=float(_X0_SMS), MulCo=float(_MULCO_SMS),
                INS=np.atleast_1d(_frozen("INSexp.txt")), Met=0),
}


def _singlet(thickness, N0=1.0, N1=0.0):
    """Baseline (N0, N1, nothing else) + a natural-width singlet at 0 mm/s."""
    p = np.zeros(NB)
    p[0], p[4] = N0, N1
    return np.concatenate([p, [thickness, 0.0, NAT_WIDTH, 0.0]])


def _ti(x, p, model, JN, method, Norm, lengths=None):
    mp = METHODS[method]
    return np.asarray(m5.TI(np.asarray(x, dtype=float), p, list(model), JN, _SerialPool(),
                            mp['x0'], mp['MulCo'], mp['INS'], [0], [0], Met=mp['Met'],
                            Norm=Norm, lengths=lengths), dtype=float)


# --- CMS: the line depth of the exact transmission integral ---------------------

def _exact_cms(v, thickness):
    """N0 = 1 count rate of the CMS source over the singlet, integrated over the
    ABSORBER energy: the source enters at every offset, none of its wings is cut.
    Step: the trapezoid error is ~exp(-2 pi HWHM / h) for a line of half width
    HWHM, so h = NAT_WIDTH / 20 puts it at exp(-20 pi), far below anything tested.
    The range: past +-40 mm/s the singlet absorbs nothing the test can see."""
    h = NAT_WIDTH / 20
    E = np.arange(-40.0, 40.0 + h / 2, h)
    absorbed = 1 - np.exp(-np.pi * NAT_WIDTH / 2 * thickness * m5.Voight(NAT_WIDTH, 0.0, E))
    source = np.array([m5.Voight(NAT_WIDTH, _G, E - vi) for vi in v])
    return 1 - source @ absorbed * h


def test_cms_line_depth_follows_the_exact_integral():
    """The sampled source is 0.6 % short of the whole line (its Lorentzian wings
    beyond +-4.7 mm/s). Dividing by Norm, as before, made the line 1/Norm too
    deep; with the wing photons counted as transmitted the depth is right up to
    the integration error, which must be far below that former bias."""
    JN = 64                                              # the CMS default
    v = np.linspace(-3.0, 3.0, 61)
    thickness = 8.0                                      # deep: depth ~0.77
    Norm = iio.compute_norm(_SerialPool(), JN, METHODS["CMS"])
    model = _ti(v, _singlet(thickness), ['Singlet'], JN, "CMS", Norm)
    exact = _exact_cms(v, thickness)

    former_bias = (1 / Norm - 1) * (1 - exact.min())
    assert np.abs(model - exact).max() < former_bias / 4
    former = (model - (1 - Norm)) / Norm                 # the division, from the same sum
    assert np.abs(former - exact).max() > former_bias / 2  # ... which this test rejects


# --- both forms, everywhere TI applies them --------------------------------------

_NORM = 0.9          # any Norm that is not 1 separates the two forms


def _expected(at_norm_1, N0, N1, met):
    """TI at Norm = _NORM, from TI at Norm = 1 (where the two forms agree)."""
    if met == 1:
        return at_norm_1 + N0 * (1 - _NORM)             # the wing photons, unabsorbed
    return (at_norm_1 - N1) / _NORM + N1                 # divided out


@pytest.mark.parametrize("method", sorted(METHODS))
def test_single_spectrum_cms_adds_and_sms_divides(method):
    x = np.linspace(-6.0, 6.0, 25)
    N0, N1 = 2.0, 0.5
    p = _singlet(3.0, N0, N1)
    at_norm_1 = _ti(x, p, ['Singlet'], 16, method, 1)
    got = _ti(x, p, ['Singlet'], 16, method, _NORM)
    np.testing.assert_allclose(got, _expected(at_norm_1, N0, N1, METHODS[method]['Met']),
                               rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_nbaseline_sections_cms_adds_and_sms_divides(method):
    x = np.linspace(-6.0, 6.0, 25)
    n = len(x)
    p = np.concatenate([_singlet(3.0, 2.0, 0.5), _singlet(1.0, 3.0, 0.25)])
    model = ['Singlet', 'Nbaseline', 'Singlet']
    xx = np.concatenate([x, x])
    at_norm_1 = _ti(xx, p, model, 16, method, 1, lengths=[n, n])
    got = _ti(xx, p, model, 16, method, _NORM, lengths=[n, n])
    met = METHODS[method]['Met']
    expected = np.concatenate([_expected(at_norm_1[:n], 2.0, 0.5, met),
                               _expected(at_norm_1[n:], 3.0, 0.25, met)])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_a_cms_and_an_sms_section_each_keep_their_own_form():
    x = np.linspace(-6.0, 6.0, 25)
    n = len(x)
    p = np.concatenate([_singlet(3.0, 2.0, 0.5), _singlet(1.0, 3.0, 0.25)])
    model = ['Singlet', 'Nbaseline', 'Singlet']
    cms, sms = METHODS["CMS"], METHODS["SMS"]

    def mixed(norms):
        return np.asarray(m5.TI(np.concatenate([x, x]), p, model, 16, _SerialPool(),
                                [cms['x0'], sms['x0']], [cms['MulCo'], sms['MulCo']],
                                [cms['INS'], sms['INS']], [0], [0], Met=[1, 0],
                                Norm=norms, lengths=[n, n]), dtype=float)

    at_norm_1 = mixed([1, 1])
    got = mixed([_NORM, _NORM])
    expected = np.concatenate([_expected(at_norm_1[:n], 2.0, 0.5, 1),
                               _expected(at_norm_1[n:], 3.0, 0.25, 0)])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


# --- the CMS instrumental-function search fits G: its Norm must follow -----------

def test_cms_search_norm_follows_g_and_jn(monkeypatch):
    calls = []
    real = iio.compute_norm

    def counted(pool, JN, method_params):
        calls.append((JN, method_params['INS']))
        return real(pool, JN, method_params)

    monkeypatch.setattr(iio, 'compute_norm', counted)
    norm = iio.cms_norm_following_g(_SerialPool(), MulCoCMS)

    first = norm(64, _G)
    assert first == real(_SerialPool(), 64, METHODS["CMS"])    # compute_norm's own value
    assert norm(64, _G) == first and len(calls) == 1           # same G and JN: kept
    assert norm(64, 2 * _G) != first and len(calls) == 2       # G moved: recomputed
    assert norm(128, 2 * _G) != norm(64, 2 * _G)               # JN too
