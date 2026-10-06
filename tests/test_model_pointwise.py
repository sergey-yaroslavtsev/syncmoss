"""The model at a velocity does not depend on which other velocities are computed.

Exclusion regions rest on this: a fit with exclusion regions fits only the
points that are left (fitting_io.fit_model hands the model A[keep]), which is
right only because models.TI computes every velocity on its own -- the
instrumental function is convolved as an integral over the source energy for
each velocity, never across neighbouring data points. If this fails for a
model, excluding points changes that model at the points that are kept.

Every golden model, with the tests' frozen SMS (Gaussian sum and theoretical
shape) and CMS instrumental functions, and an Nbaseline pair. Pure NumPy/Numba.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5
import syncmoss.sms_theory as smst
from syncmoss import exclusion_regions as er
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import number_of_baseline_parameters as NB

from conftest import FROZEN_PARAMETERS

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _GOLDEN = json.load(_f)

JN = 16
_A = np.linspace(-10.0, 10.0, 256)
_KEEP = er.kept(_A, er.parse("-12:-8; -3:-2; 1:2"))
_BASELINE = np.zeros(NB)
_BASELINE[0] = 1.0e4


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


def _methods():
    """The instrumental functions the model is computed with, as TI takes them."""
    MulCo, x0 = _frozen("INSint.txt")
    theory = np.atleast_1d(np.array(_frozen("INSth.txt"), dtype=float))
    assert smst.ins_kind(theory) != smst.KIND_GAUSS, "INSth.txt must hold the theoretical shape"
    return {
        "SMS Gaussians": dict(Met=0, INS=np.atleast_1d(_frozen("INSexp.txt")),
                              MulCo=float(MulCo), x0=float(x0)),
        "SMS theory": dict(Met=0, INS=theory, MulCo=float(MulCo), x0=float(x0)),
        "CMS": dict(Met=1, INS=float(_frozen("GCMS.txt", delimiter='\t')),
                    MulCo=MulCoCMS, x0=0.0),
    }


_METHODS = _methods()


def _ti(x, p, model, method, Distri=(0,), Cor=(0,), **kwargs):
    mp = _METHODS[method]
    return np.asarray(m5.TI(np.asarray(x, dtype=float), p, list(model), JN, _SerialPool(),
                            mp['x0'], mp['MulCo'], mp['INS'], list(Distri), list(Cor),
                            Met=mp['Met'], Norm=1, **kwargs), dtype=float)


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("name", sorted(_GOLDEN["cases"]))
def test_the_model_at_the_kept_points_does_not_change(name, method):
    case = _GOLDEN["cases"][name]
    p = np.concatenate([_BASELINE, case["params"]])
    full = _ti(_A, p, case["model"], method)
    kept = _ti(_A[_KEEP], p, case["model"], method)
    assert np.array_equal(full[_KEEP], kept)       # bit for bit


def _distributed_sextets():
    """The golden Sextet with its H (index 3) spread: by a Distr with a Corr
    moving delta (index 1) along, and by a Recon with free weights."""
    sextet = list(_GOLDEN["cases"]["Sextet"]["params"])
    H = sextet[3]
    distr = [3.0, H - 3.0, H + 3.0, 7.0, 0.0]
    corr = [1.0, 0.0]
    recon = [3.0, H - 3.0, H + 3.0, 5.0, 0.0, 0.0, 0.0]
    return {
        "Distr + Corr": dict(model=["Sextet", "Distr", "Corr"], params=sextet + distr + corr,
                             Distri=[f"np.exp(-(X - {H})**2 / 2)"], Cor=[f"0.01 * (X - {H})"]),
        "Recon": dict(model=["Sextet", "Recon"], params=sextet + recon,
                      Recon=[np.array([0.1, 0.3, 0.4, 0.15, 0.05])]),
    }


_DISTRIBUTED = _distributed_sextets()


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("name", sorted(_DISTRIBUTED))
def test_distributions_are_computed_point_by_point_too(name, method):
    case = dict(_DISTRIBUTED[name])
    model, params = case.pop("model"), case.pop("params")
    p = np.concatenate([_BASELINE, params])
    full = _ti(_A, p, model, method, **case)
    kept = _ti(_A[_KEEP], p, model, method, **case)
    assert np.array_equal(full[_KEEP], kept)


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_two_spectra_of_an_nbaseline_model_too(method):
    first, second = _GOLDEN["cases"]["Sextet"], _GOLDEN["cases"]["Doublet"]
    model = list(first["model"]) + ['Nbaseline'] + list(second["model"])
    p = np.concatenate([_BASELINE, first["params"], _BASELINE, second["params"]])
    n, k = len(_A), int(_KEEP.sum())
    full = _ti(np.concatenate([_A, _A]), p, model, method, lengths=[n, n])
    kept = _ti(np.concatenate([_A[_KEEP], _A[_KEEP]]), p, model, method, lengths=[k, k])
    assert np.array_equal(np.concatenate([full[:n][_KEEP], full[n:][_KEEP]]), kept)
