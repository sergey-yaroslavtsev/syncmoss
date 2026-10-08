"""TImod skips a component that a Distr/Recon row right after it replaces by its
distributed copies (models.SKIP_REPLACED_COMPONENT): that row rewinds to before
the component, so its own contribution was computed only to be thrown away.
Skipping it must change nothing: every model below is bit-identical with the
skip on and off. Pure NumPy/Numba, serial pool.
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
    _GOLDEN = json.load(_f)["cases"]
_SEXTET = list(_GOLDEN["Sextet"]["params"])
_DOUBLET = list(_GOLDEN["Doublet"]["params"])
_RELAX = list(_GOLDEN["Relax_MS"]["params"])
_H = _SEXTET[3]
JN = 16
_X = np.linspace(-10.0, 10.0, 41)
_G = float(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "GCMS.txt"), delimiter='\t'))
_BASE = [1.0e4] + [0.0] * (NB - 1)
_DISTR_H = [3.0, _H - 3.0, _H + 3.0, 5.0, 0.0]
_DISTR_D = [1.0, -0.4, 0.4, 4.0, 0.0]
_RECON_H = [3.0, _H - 3.0, _H + 3.0, 5.0, 0.0, 0.0, 0.0]
_W = np.array([0.1, 0.3, 0.4, 0.15, 0.05])

CASES = {
    "Sextet + Distr + Corr": (['Sextet', 'Distr', 'Corr'], _SEXTET + _DISTR_H + [1.0, 0.0],
                              [f"np.exp(-(X - {_H})**2 / 2)"], [f"0.01 * (X - {_H})"], [0], None),
    "Sextet + Recon": (['Sextet', 'Recon'], _SEXTET + _RECON_H, [0], [0], [_W], None),
    "Sextet + Distr x Distr": (['Sextet', 'Distr', 'Distr'], _SEXTET + _DISTR_H + _DISTR_D,
                               [f"np.exp(-(X - {_H})**2 / 2)", "np.exp(-X**2 / 0.08)"], [0], [0], None),
    "Relax_MS + Recon": (['Relax_MS', 'Recon'], _RELAX + _RECON_H, [0], [0], [_W], None),
    "Doublet + Distr, then a Sextet": (['Doublet', 'Distr', 'Sextet'], _DOUBLET + [2.0, 0.3, 1.3, 5.0, 0.0] + _SEXTET,
                                       ["np.exp(-(X - 0.8)**2 / 0.1)"], [0], [0], None),
    "Nbaseline: Sextet + Recon | Sextet + Distr": (
        ['Sextet', 'Recon', 'Nbaseline', 'Sextet', 'Distr'], _SEXTET + _RECON_H + _BASE + _SEXTET + _DISTR_H,
        [f"np.exp(-(X - {_H})**2 / 2)"], [0], [_W], [len(_X), len(_X)]),
}


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*a) for a in iterable]


@pytest.mark.parametrize("name", sorted(CASES))
def test_skipping_the_replaced_component_changes_nothing(name, monkeypatch):
    model, comp, Distri, Cor, Recon, lengths = CASES[name]
    x = np.concatenate([_X] * (len(lengths) if lengths else 1))
    p = np.array(_BASE + comp, dtype=float)

    def ti():
        m5._NODE_SUMS.clear()
        return np.asarray(m5.TI(x, p, model, JN, _SerialPool(), 0.0, MulCoCMS, _G, list(Distri), list(Cor),
                                Met=1, Norm=1, Recon=list(Recon), lengths=lengths), dtype=float)

    monkeypatch.setattr(m5, 'SKIP_REPLACED_COMPONENT', False)
    computed = ti()
    monkeypatch.setattr(m5, 'SKIP_REPLACED_COMPONENT', True)
    skipped = ti()
    assert np.array_equal(skipped, computed)
