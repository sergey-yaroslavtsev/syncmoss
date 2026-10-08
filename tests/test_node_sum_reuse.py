"""models.TI reuses a spectrum's node sum in an Nbaseline model ("Reuse of a
spectrum's node sum" there) when nothing it depends on has changed.

Every reused result must be bit-identical to a fresh computation on an empty
store, and a pool that counts its tasks shows what was recomputed: a moved
parameter of one spectrum recomputes that spectrum only, a baseline move
recomputes nothing, a parameter named in another spectrum's distribution
expression recomputes that one too, and changed Recon weights or a changed
instrumental function are never reused. Pure NumPy/Numba, serial pool.
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

JN = 16
_N = 40                                                   # points per spectrum
_X = np.linspace(-10.0, 10.0, _N)
_G = float(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "GCMS.txt"), delimiter='\t'))
_SEXTET = list(_GOLDEN["Sextet"]["params"])
_DOUBLET = list(_GOLDEN["Doublet"]["params"])


class _CountingPool:
    """Serial stand-in for the pool that counts the node tasks it is given."""

    def __init__(self):
        self.tasks = 0

    def starmap(self, func, iterable):
        args = list(iterable)
        self.tasks += len(args)
        return [func(*a) for a in args]


@pytest.fixture(autouse=True)
def _empty_store():
    m5._NODE_SUMS.clear()
    yield
    m5._NODE_SUMS.clear()


def _baseline():
    b = np.zeros(NB)
    b[0] = 1.0e4
    return list(b)


def _three_spectra():
    """Sextet | Doublet | Sextet+Distr+Corr, the Distr's expression naming the
    FIRST spectrum's field (an absolute p index)."""
    h_index = NB + 3                                      # spectrum 1's H slot
    p = (_baseline() + _SEXTET + _baseline() + _DOUBLET + _baseline() + _SEXTET
         + [3.0, _SEXTET[3] - 3.0, _SEXTET[3] + 3.0, 5.0, 0.0] + [1.0, 0.0])
    model = ['Sextet', 'Nbaseline', 'Doublet', 'Nbaseline', 'Sextet', 'Distr', 'Corr']
    starts = [0, NB + 14, 2 * NB + 23]                    # each spectrum's baseline
    return dict(p=np.array(p, dtype=float), model=model, starts=starts,
                Distri=[f"np.exp(-(X - p[{h_index}])**2 / 2)"], Cor=[f"0.01 * (X - {_SEXTET[3]})"],
                Recon=[0], h_index=h_index)


def _ti(case, p, pool, G=_G, Recon=None):
    return np.asarray(m5.TI(np.concatenate([_X] * len(case['starts'])), p, case['model'], JN, pool, 0.0,
                            MulCoCMS, G, list(case['Distri']), list(case['Cor']), Met=1, Norm=1,
                            Recon=case['Recon'] if Recon is None else Recon,
                            lengths=[_N] * len(case['starts'])), dtype=float)


def _fresh(case, p, **kw):
    m5._NODE_SUMS.clear()
    return _ti(case, p, _CountingPool(), **kw)


def test_a_repeated_call_reuses_every_spectrum_bit_for_bit():
    case = _three_spectra()
    pool = _CountingPool()
    first = _ti(case, case['p'], pool)
    assert pool.tasks == 3 * JN
    again = _ti(case, case['p'], pool)
    assert pool.tasks == 3 * JN                           # nothing recomputed
    assert np.array_equal(first, again)


def test_a_parameter_of_one_spectrum_recomputes_that_spectrum_only():
    case = _three_spectra()
    pool = _CountingPool()
    _ti(case, case['p'], pool)
    p = case['p'].copy()
    p[case['starts'][1] + NB + 1] += 0.01                 # the Doublet's centre shift
    before = pool.tasks
    got = _ti(case, p, pool)
    assert pool.tasks - before == JN
    assert np.array_equal(got, _fresh(case, p))


def test_a_baseline_move_needs_no_integration():
    case = _three_spectra()
    pool = _CountingPool()
    _ti(case, case['p'], pool)
    p = case['p'].copy()
    p[case['starts'][2]] *= 1.001                         # the third spectrum's Ns
    before = pool.tasks
    got = _ti(case, p, pool)
    assert pool.tasks == before
    assert np.array_equal(got, _fresh(case, p))


def test_a_parameter_named_in_another_spectrums_expression_recomputes_it_too():
    case = _three_spectra()
    pool = _CountingPool()
    _ti(case, case['p'], pool)
    p = case['p'].copy()
    p[case['h_index']] += 0.1                             # spectrum 1's H, named by spectrum 3's Distr
    before = pool.tasks
    got = _ti(case, p, pool)
    assert pool.tasks - before == 2 * JN                  # spectra 1 and 3
    assert np.array_equal(got, _fresh(case, p))


def test_another_instrumental_function_is_never_reused():
    case = _three_spectra()
    pool = _CountingPool()
    _ti(case, case['p'], pool)
    before = pool.tasks
    got = _ti(case, case['p'], pool, G=2 * _G)
    assert pool.tasks - before == 3 * JN
    assert np.array_equal(got, _fresh(case, case['p'], G=2 * _G))


def test_changed_recon_weights_are_never_reused():
    H = _SEXTET[3]
    case = dict(p=np.array(_baseline() + _SEXTET + _baseline() + _SEXTET
                           + [3.0, H - 3.0, H + 3.0, 5.0, 0.0, 0.0, 0.0], dtype=float),
                model=['Sextet', 'Nbaseline', 'Sextet', 'Recon'], starts=[0, NB + 14],
                Distri=[0], Cor=[0], Recon=[np.array([0.1, 0.3, 0.4, 0.15, 0.05])])
    pool = _CountingPool()
    _ti(case, case['p'], pool)
    other = [np.array([0.2, 0.2, 0.2, 0.2, 0.2])]
    before = pool.tasks
    got = _ti(case, case['p'], pool, Recon=other)
    assert pool.tasks - before == JN                      # the Recon spectrum only
    assert np.array_equal(got, _fresh(case, case['p'], Recon=other))


def test_an_index_that_cannot_be_read_off_is_never_reused():
    case = _three_spectra()
    case['Distri'] = [f"np.exp(-(X - p[{NB} + 3])**2 / 2)"]   # the same field, written as a sum
    pool = _CountingPool()
    first = _ti(case, case['p'], pool)
    before = pool.tasks
    again = _ti(case, case['p'], pool)
    assert pool.tasks - before == JN                      # the third spectrum, every time
    assert np.array_equal(first, again)
