"""A multidimensional distribution in a LATER spectrum of an Nbaseline model reads
its own Distri / Cor / Recon entries.

TImod's nested call of a multidimensional distribution counts the entries from
the start of the model it is given; for a later Nbaseline section that is the
section's own model, while the lists are global, so the section's offsets
(Di0/Co0/Re0) must be added. Without them the second spectrum took the FIRST
spectrum's entries. Each case: the second spectrum, inside the Nbaseline model
next to a first spectrum with DIFFERENT entries, equals the same spectrum
computed alone, bit for bit. Pure NumPy/Numba, serial pool.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import number_of_baseline_parameters as NB

from conftest import FROZEN_PARAMETERS

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _SEXTET = list(json.load(_f)["cases"]["Sextet"]["params"])

JN = 16
_X = np.linspace(-10.0, 10.0, 61)
_G = float(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "GCMS.txt"), delimiter='\t'))
_H = _SEXTET[3]
_BASE = [1.0e4] + [0.0] * (NB - 1)
_DISTR_H = [3.0, _H - 4.0, _H + 4.0, 7.0, 0.0]            # over H (slot 3)
_DISTR_D = [1.0, -0.5, 0.5, 5.0, 0.0]                      # over delta (slot 1)
_RECON_H = [3.0, _H - 4.0, _H + 4.0, 5.0, 0.0, 0.0, 0.0]   # Recon over H, 5 weights
_RECON_D = [1.0, -0.5, 0.5, 5.0, 0.0, 0.0, 0.0]            # Recon over delta
_W_FIRST = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
_W_SECOND = np.array([0.05, 0.15, 0.3, 0.3, 0.2])


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*a) for a in iterable]


def _ti(x, p, model, Distri, Recon, lengths=None):
    return np.asarray(m5.TI(x, np.array(p, dtype=float), model, JN, _SerialPool(), 0.0, MulCoCMS, _G,
                            list(Distri), [0], Met=1, Norm=1, Recon=list(Recon), lengths=lengths), dtype=float)


CASES = {
    # (first spectrum: model rows, params, Distri, Recon) | (second spectrum: the same)
    "Distr x Distr": (
        (['Distr'], _DISTR_H, [f"np.exp(-(X - {_H})**2 / 2)"], []),
        (['Distr', 'Distr'], _DISTR_H + _DISTR_D, [f"np.exp(-(X - {_H})**2 / 18)", "np.exp(-X**2 / 0.08)"], [])),
    "Recon x Distr": (
        (['Recon'], _RECON_H, [], [_W_FIRST]),
        (['Recon', 'Distr'], _RECON_H + _DISTR_D, ["np.exp(-X**2 / 0.08)"], [_W_SECOND])),
    "Distr x Recon": (
        (['Distr'], _DISTR_H, [f"np.exp(-(X - {_H})**2 / 2)"], []),
        (['Distr', 'Recon'], _DISTR_H + _RECON_D, [f"np.exp(-(X - {_H})**2 / 18)"], [_W_SECOND])),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_second_spectrum_reads_its_own_entries(name):
    (rows1, par1, d1, r1), (rows2, par2, d2, r2) = CASES[name]
    alone = _ti(_X, _BASE + _SEXTET + par2, ['Sextet'] + rows2, d2 or [0], r2 or [0])
    model = ['Sextet'] + rows1 + ['Nbaseline', 'Sextet'] + rows2
    together = _ti(np.concatenate([_X, _X]), _BASE + _SEXTET + par1 + _BASE + _SEXTET + par2, model,
                   (d1 + d2) or [0], (r1 + r2) or [0], lengths=[len(_X), len(_X)])
    assert np.array_equal(together[len(_X):], alone)
