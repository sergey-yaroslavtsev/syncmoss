"""The baseline polynomial and its default links.

    N0 = Ns  * (1 + lins/10^2  (v - Os)  + c²s/10^4  (v - Os)^2)
    N1 = Nnr * (1 + linnr/10^2 (v - Onr) + c²nr/10^4 (v - Onr)^2)

Both polynomials are centred, the linear term too: it used to be lins * v, so
moving Os moved the parabola and not the slope. Pinned against TI with an empty
model -- transmission 1, so TI is exactly N0 + N1 -- for one spectrum (CMS and
SMS) and for Nbaseline sections, and against the plotted baseline.

The default links: Onr, c²nr, linnr start linked to Os, c²s, lins of their own
row, in the baseline row of a new table and in every Nbaseline row picked, and
stay on their own row when rows above it are changed, inserted or deleted.
"""
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss import instrumental_io as iio
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.spectrum_plotter import calculate_baseline

from conftest import FROZEN_PARAMETERS


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_MULCO_SMS, _X0_SMS = _frozen("INSint.txt")
METHODS = {
    "CMS": dict(x0=0.0, MulCo=MulCoCMS, INS=float(_frozen("GCMS.txt", delimiter='\t')), Met=1),
    "SMS": dict(x0=float(_X0_SMS), MulCo=float(_MULCO_SMS),
                INS=np.atleast_1d(_frozen("INSexp.txt")), Met=0),
}
JN = 16

# Ns, Os, c²s, lins, Nnr, Onr, c²nr, linnr -- every term non-zero, the two
# centres different
P = np.array([1000.0, 0.7, 3.0, -2.0, 400.0, -0.4, 5.0, 1.5])
P2 = np.array([2500.0, -1.1, -4.0, 3.5, 900.0, 0.9, 2.0, -0.8])
V = np.linspace(-6.0, 6.0, 41)


def _expected(p, v):
    def part(N, centre, c2, lin):
        return N * (1 + lin / 1e2 * (v - centre) + c2 / 1e4 * (v - centre) ** 2)
    return part(p[0], p[1], p[2], p[3]) + part(p[4], p[5], p[6], p[7])


def _ti(x, p, model, method, lengths=None):
    mp = METHODS[method]
    norm = iio.compute_norm(_SerialPool(), JN, mp)
    return np.asarray(m5.TI(np.asarray(x, dtype=float), np.asarray(p, dtype=float), list(model),
                            JN, _SerialPool(), mp['x0'], mp['MulCo'], mp['INS'], [0], [0],
                            Met=mp['Met'], Norm=norm, lengths=lengths), dtype=float)


@pytest.mark.parametrize("method", ["CMS", "SMS"])
def test_ti_baseline_is_the_centred_polynomial(method):
    assert len(P) == NB
    assert np.allclose(_ti(V, P, [], method), _expected(P, V), rtol=1e-10, atol=0)


@pytest.mark.parametrize("method", ["CMS", "SMS"])
def test_moving_the_centres_moves_the_whole_baseline(method):
    """Os and Onr shifted by d, the velocities too: the same counts. With lins * v
    the linear term did not move, and this failed for any lins != 0."""
    d = 1.3
    moved = P.copy()
    moved[1] += d
    moved[5] += d
    assert np.allclose(_ti(V + d, moved, [], method), _ti(V, P, [], method), rtol=1e-10, atol=0)


def test_nbaseline_sections_each_have_their_own_centred_polynomial():
    v2 = np.linspace(-3.0, 9.0, 23)
    out = _ti(np.concatenate([V, v2]), np.concatenate([P, P2]), ['Nbaseline'], "SMS",
              lengths=[len(V), len(v2)])
    assert np.allclose(out, np.concatenate([_expected(P, V), _expected(P2, v2)]), rtol=1e-10, atol=0)


def test_the_plotted_baseline_is_the_same_polynomial():
    assert np.allclose(calculate_baseline(P, V), _expected(P, V), rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# The default links
# ---------------------------------------------------------------------------

SHAPE_COLS = (5, 6, 7)     # Onr, c²nr, linnr -> Os, c²s, lins (1, 2, 3)


def _shape_texts(pt, row):
    return [pt._value_input(row, col).text() for col in SHAPE_COLS]


def _own_links(pt, row):
    start = pt._flat_starts()[row]
    return [f'=[{start + 1},1]', f'=[{start + 2},1]', f'=[{start + 3},1]']


@pytest.mark.gui
def test_a_new_table_links_the_nonresonant_shape_to_the_resonant_one(physics_app):
    from syncmoss.model_io import read_model
    pt = physics_app.params_table
    assert _shape_texts(pt, 0) == ['=[1,1]', '=[2,1]', '=[3,1]']
    _model, _p, con1, con2, con3, *_ = read_model(physics_app, substitute_names=False)
    assert {int(t): (int(s), f) for t, s, f in zip(con1, con2, con3)} == \
        {5: (1, 1.0), 6: (2, 1.0), 7: (3, 1.0)}


@pytest.mark.gui
def test_cms_and_sms_keep_the_shape_links(physics_app):
    """The CMS switch gives Nnr its =[0,0.67] and leaves the shape links alone,
    both ways."""
    pt = physics_app.params_table
    physics_app.MS_fit.setChecked(True)
    assert pt._value_input(0, 4).text() == '=[0,0.67]'
    assert _shape_texts(pt, 0) == ['=[1,1]', '=[2,1]', '=[3,1]']
    physics_app.SMS_fit.setChecked(True)
    assert pt._value_input(0, 4).text() == '0'
    assert _shape_texts(pt, 0) == ['=[1,1]', '=[2,1]', '=[3,1]']


@pytest.mark.gui
def test_an_nbaseline_links_its_own_shape_through_every_edit(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')               # p[8:22]
    pt.select_model(2, 'Nbaseline')            # p[22:30]
    assert _shape_texts(pt, 2) == ['=[23,1]', '=[24,1]', '=[25,1]']

    pt.select_model(2, 'Nbaseline')            # picked again: same row, same links
    assert _shape_texts(pt, 2) == ['=[23,1]', '=[24,1]', '=[25,1]']

    pt.select_model(1, 'Doublet')              # 5 slots fewer above it
    assert _shape_texts(pt, 2) == _own_links(pt, 2) == ['=[18,1]', '=[19,1]', '=[20,1]']

    pt.select_model(1, 'Insert')
    pt.select_model(1, 'Singlet')              # 4 slots more above it, now row 3
    assert _shape_texts(pt, 3) == _own_links(pt, 3) == ['=[22,1]', '=[23,1]', '=[24,1]']

    pt.select_model(1, 'Delete')
    assert _shape_texts(pt, 2) == _own_links(pt, 2) == ['=[18,1]', '=[19,1]', '=[20,1]']
    # the baseline row's own links never moved
    assert _shape_texts(pt, 0) == ['=[1,1]', '=[2,1]', '=[3,1]']
