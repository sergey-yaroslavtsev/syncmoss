"""models.TI_recon_variants: the spectra of several Recon weight lists that share
every other input, computed together (a fit's Recon weight columns).

At every node each copy of the distributed component is computed once at unit
weight and recombined for every list (thin part CH_c**g_c, thick part
g_c*Smat_c), so the result must equal TI for each list up to rounding: one
spectrum and Nbaseline models, thin and thick components, the varying Recon in a
later spectrum or in two spectra at once, a multidimensional Distr + Recon, and
a Recon over the amplitude itself (the weights then act elsewhere, and the
plain path is used). Pure NumPy/Numba, serial pool.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import NAT_WIDTH, number_of_baseline_parameters as NB

from conftest import FROZEN_PARAMETERS

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _SEXTET = list(json.load(_f)["cases"]["Sextet"]["params"])
_SEXTET[0] = 10.0                                       # thick
_H = _SEXTET[3]
JN = 16
_X = np.linspace(-10.0, 10.0, 41)
_G = float(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "GCMS.txt"), delimiter='\t'))
_BASE = [1.0e4] + [0.0] * (NB - 1)
_RECON_H = [3.0, _H - 3.0, _H + 3.0, 5.0, 0.0, 0.0, 0.0]    # over H, 5 weights
_RECON_D = [1.0, -0.5, 0.5, 5.0, 0.0, 0.0, 0.0]             # over delta
_RECON_I = [0.0, 1.0, 9.0, 5.0, 0.0, 0.0, 0.0]              # over the amplitude itself
_SINGLET = [3.0, 0.1, NAT_WIDTH, 0.15]
_W = [np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
      np.array([0.1, 0.3, 0.4 * (1 + 1e-6), 0.15, 0.05]),
      np.array([0.2, 0.2, 0.2, 0.2, 0.2])]
_W2 = [np.array([0.3, 0.3, 0.2, 0.1, 0.1]), np.array([0.05, 0.1, 0.2, 0.3, 0.35]),
       np.array([0.3, 0.3, 0.2, 0.1, 0.1 * (1 + 1e-6)])]


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*a) for a in iterable]


def _case(name):
    """(model, p, Distri, the Recon lists of the weight variants, lengths)"""
    if name == "one thick spectrum":
        return ['Sextet', 'Recon'], _BASE + _SEXTET + _RECON_H, [0], [[w] for w in _W], None
    if name == "one thin spectrum":
        return ['Singlet', 'Recon'], _BASE + _SINGLET + _RECON_D, [0], [[w] for w in _W], None
    if name == "second spectrum varies":
        return (['Sextet', 'Nbaseline', 'Sextet', 'Recon'], _BASE + _SEXTET + _BASE + _SEXTET + _RECON_H,
                [0], [[w] for w in _W], [len(_X)] * 2)
    if name == "both spectra vary":
        return (['Sextet', 'Recon', 'Nbaseline', 'Sextet', 'Recon'],
                _BASE + _SEXTET + _RECON_H + _BASE + _SEXTET + _RECON_D, [0],
                [[w, w2] for w, w2 in zip(_W, _W2)], [len(_X)] * 2)
    if name == "Distr x Recon":
        return (['Sextet', 'Distr', 'Recon'], _BASE + _SEXTET + [1.0, -0.5, 0.5, 5.0, 0.0] + _RECON_H,
                ["np.exp(-X**2 / 0.08)"], [[w] for w in _W], None)
    if name == "Recon over the amplitude":
        return ['Sextet', 'Recon'], _BASE + _SEXTET + _RECON_I, [0], [[w] for w in _W], None
    raise KeyError(name)


CASES = ["one thick spectrum", "one thin spectrum", "second spectrum varies", "both spectra vary",
         "Distr x Recon", "Recon over the amplitude"]


@pytest.fixture(autouse=True)
def _empty_store():
    m5._NODE_SUMS.clear()
    yield
    m5._NODE_SUMS.clear()


@pytest.mark.parametrize("name", CASES)
def test_every_weight_list_gives_the_spectrum_ti_gives(name):
    model, p, Distri, recons, lengths = _case(name)
    x = np.concatenate([_X] * (len(lengths) if lengths else 1))
    p = np.array(p, dtype=float)
    got = m5.TI_recon_variants(x, p, model, JN, _SerialPool(), 0.0, MulCoCMS, _G, list(Distri), [0], 1, 1,
                               m5.SMS_POL_DEFAULT, recons, lengths)
    assert len(got) == len(recons)
    for R, spectrum in zip(recons, got):
        m5._NODE_SUMS.clear()
        expected = np.asarray(m5.TI(x, p, model, JN, _SerialPool(), 0.0, MulCoCMS, _G, list(Distri), [0],
                                    Met=1, Norm=1, Recon=R, lengths=lengths), dtype=float)
        np.testing.assert_allclose(np.asarray(spectrum, dtype=float), expected, rtol=1e-12, atol=0)
