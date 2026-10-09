"""The instrumental-function search with the model: links, several spectra,
exclusion regions, and the results table afterwards.

* A link ``=[X,Y]`` and an Expression follow their source during the search.
  They used to be applied once, at the start, and then held: a linked
  parameter whose source was free stayed at its start value while the source
  moved.
* With several spectra in the path box the model is fitted to all of them with
  ONE instrumental function: a model with Nbaseline rows as built, any other
  expanded as the simultaneous one-model fit expands it.
* Every search leaves out the exclusion regions.
* Afterwards the results table shows the model as after a fit: Take result
  takes it, Save result refuses it.

SPEED. What is checked is the plumbing, so the minimiser is cut to one
iteration (``short_minimizer``) -- the real code path still runs every step of
both searches, with the real minimiser.
"""
import os
import shutil
import types

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss import instrumental_io as iio, model_io
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.spectrum_plotter import EXCLUSION_GID, spectrum_axes_of

pytestmark = [pytest.mark.gui]

SEXTET = 14
T = NB              # the Sextet's thickness (row 1, column 0)
DELTA = NB + 1      # its delta (row 1, column 1)
L = NB + SEXTET     # one spectrum's section: baseline + Sextet


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _searches_minimize_with(monkeypatch, minimi_hi):
    """The searches call *minimi_hi*; everything else (models.limits, ...) keeps
    the real one."""
    monkeypatch.setattr(iio, 'mi', types.SimpleNamespace(
        minimi_hi=minimi_hi, _eval_expr=iio.mi._eval_expr))


@pytest.fixture
def short_minimizer(monkeypatch):
    """Every minimisation of the searches cut to one iteration; the keyword
    arguments of each call are recorded."""
    real = iio.mi.minimi_hi
    calls = []

    def short(model_func, x, y, p0, **kwargs):
        calls.append(kwargs)
        return real(model_func, x, y, p0, **dict(kwargs, MI=1, MI2=1))

    _searches_minimize_with(monkeypatch, short)
    return calls


@pytest.fixture
def quick_theory(monkeypatch):
    """The theoretical search's schedule cut to one short pass (as in
    test_theory_search.py)."""
    monkeypatch.setattr(iio, "THEORY_PASSES", (('shift',),))
    monkeypatch.setattr(iio, "THEORY_PASS_MI", 2)
    monkeypatch.setattr(iio, "THEORY_ESCALATION_CHI2", 1e9)


@pytest.fixture
def app(physics_app):
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)
    physics_app.jn0_input.setText("16")
    cal = physics_app.calibration_path
    physics_app.process_path.setPlainText(repr([cal]))
    physics_app.path_list = [cal]
    assert physics_app.initialize_parameters()
    return physics_app


def _fix_box(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(0).layout().itemAt(1).widget()


def _sextet(app, row=1, free=(0,)):
    """A Sextet in *row* with only the columns *free* free."""
    pt = app.params_table
    pt.select_model(row, 'Sextet')
    for col in range(SEXTET):
        _fix_box(pt, row, col).setChecked(col not in free)


def _spectra(app, tmp_path, n):
    """*n* copies of the calibration spectrum in the path box."""
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        shutil.copy2(app.calibration_path, path)
        paths.append(path)
    app.process_path.setPlainText(repr(paths))
    return paths


def _search(app, theory=False, mode=1, refine=True):
    request = app._instrumental_search_request(mode)
    assert request is not None, app.log.toPlainText()
    if theory:
        return iio.instrumental_theory(app, ref=1 if refine else 0, mode=mode,
                                       pool=_SerialPool(), request=request)
    return iio.instrumental(app, 1 if refine else 0, mode, pool=_SerialPool(), request=request)


# --- links follow their source ---------------------------------------------------

@pytest.mark.parametrize("theory", [False, True], ids=["gaussians", "theory"])
def test_a_link_follows_its_free_source(app, short_minimizer, quick_theory, theory):
    """delta = 0.001 * T, T free: at the end delta is still 0.001 * T."""
    _sextet(app)
    pt = app.params_table
    model_io._set_row_param(pt.row_widgets[1], 0, '2', '0.001', '', 'False')
    model_io._set_row_param(pt.row_widgets[1], 1, f'=[{T},0.001]', '', '', 'False')

    result = _search(app, theory=theory)

    p = result['p']
    assert p[T] != pytest.approx(2.0), "T did not move: the test shows nothing"
    assert p[DELTA] == pytest.approx(0.001 * p[T], rel=1e-12)
    assert short_minimizer
    # the baseline's default Onr/c²nr/linnr -> Os/c²s/lins links, then this one
    links = [[5, 1, 1], [6, 2, 1], [7, 3, 1], [DELTA, T, 0.001]]
    assert all(np.array_equal(call['confu'].T, links) for call in short_minimizer)


def test_the_theoretical_search_returns_its_spectra_after_an_escalation(app, short_minimizer,
                                                                        monkeypatch):
    """The escalation loops over a variable called 'extra'; what the search
    returns about its spectra must survive it."""
    monkeypatch.setattr(iio, "THEORY_PASSES", (('shift',),))
    monkeypatch.setattr(iio, "THEORY_PASS_MI", 2)
    monkeypatch.setattr(iio, "THEORY_ESCALATION_CHI2", 0.0)     # always escalate
    _sextet(app)
    result = _search(app, theory=True)
    assert result['starts'] == [0] and result['model'] == ['Sextet']


def test_an_expression_follows_its_free_source(app, short_minimizer):
    _sextet(app)
    pt = app.params_table
    model_io._set_row_param(pt.row_widgets[1], 0, '2', '0.001', '', 'False')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, f'p[{T}]/1000', '', '', 'False')
    slot = int(model_io.read_model(app)[8][0])
    model_io._set_row_param(pt.row_widgets[1], 1, f'=[{slot},1]', '', '', 'False')

    p = _search(app)['p']

    assert p[T] != pytest.approx(2.0)
    assert p[slot] == pytest.approx(p[T] / 1000, rel=1e-12)
    assert p[DELTA] == pytest.approx(p[T] / 1000, rel=1e-12)


# --- which spectra --------------------------------------------------------------

def test_one_spectrum_takes_an_independent_value_as_its_value(app):
    _sextet(app)
    model_io._set_row_param(app.params_table.row_widgets[1], 1, '=(0.05)', '-1', '1', 'False')
    request = app._instrumental_search_request(1)
    assert request['kind'] == 'single' and len(request['files']) == 1


def test_several_spectra_without_nbaseline_are_one_model(app, tmp_path):
    paths = _spectra(app, tmp_path, 3)
    _sextet(app)
    request = app._instrumental_search_request(1)
    assert request['kind'] == 'one_model' and request['files'] == paths
    # the standards are single-spectrum procedures
    assert app._instrumental_search_request(2)['kind'] == 'single'
    assert app._instrumental_search_request(0)['files'] == [os.path.abspath(app.path_list[0])]


@pytest.mark.parametrize("case, expected", [
    ('recon', "Recon"),
    ('nbaseline_and_independent', "independent =(X)"),
    ('nbaseline_count', "fitted to 2 spectra, but the path box has 3"),
])
def test_what_the_search_with_the_model_refuses(app, tmp_path, case, expected):
    _spectra(app, tmp_path, 3)
    pt = app.params_table
    _sextet(app)
    if case == 'recon':
        pt.select_model(2, 'Recon')
    else:
        pt.select_model(2, 'Nbaseline')
        pt.select_model(3, 'Sextet')
    if case == 'nbaseline_and_independent':
        model_io._set_row_param(pt.row_widgets[1], 1, '=(0.05)', '-1', '1', 'False')
    app.instrumental_pressed(1, 1)
    assert expected in app.log.toPlainText()
    assert "was not started" in app.log.toPlainText()
    assert app.inprogress is False


# --- several spectra, one instrumental function ----------------------------------

def test_two_spectra_one_model_share_the_free_parameters(app, tmp_path, short_minimizer):
    _spectra(app, tmp_path, 2)
    _sextet(app, free=(0, 1))
    model_io._set_row_param(app.params_table.row_widgets[1], 1, '=(0.05)', '-1', '1', 'False')

    result = _search(app)

    p, n = result['p'], len(result['A']) // 2
    assert result['lengths'] == [n, n] and result['starts'] == [0, L]
    assert result['model'] == ['Sextet', 'Nbaseline', 'Sextet']
    assert p[L + T] == p[T]                           # shared: a link to spectrum 1
    assert p[DELTA] != 0.05 and p[L + DELTA] != 0.05  # independent: each its own
    assert len(result['F']) == 2 * n


def test_rescaling_the_gaussians_keeps_every_spectrum_s_curve(app, tmp_path, monkeypatch):
    """INS_norm brings the Gaussian amplitudes back to sum a^2 = 1 and scales
    the baseline to keep the curve: of EVERY spectrum, not only the first."""
    _spectra(app, tmp_path, 2)
    _sextet(app)
    app.params_table.select_model(2, 'Nbaseline')
    _sextet(app, row=3)
    returned = []

    def doubles_the_amplitudes(model_func, x, y, p0, **kwargs):
        p = np.array(p0, dtype=float)
        p[mod_p_len + 2::3] *= 2
        returned.append(p.copy())
        return p, np.zeros(len(p)), 1.0, np.eye(1)

    mod_p_len = len(model_io.read_model(app)[1])
    _searches_minimize_with(monkeypatch, doubles_the_amplitudes)

    result = _search(app)

    raw = returned[-1]
    expected = m5.TI(result['A'], raw[:mod_p_len], result['model'], app.JN0, _SerialPool(),
                     result['x0'], result['MulCo'], raw[mod_p_len:], [], [], Recon=[],
                     lengths=result['lengths'])
    assert np.allclose(np.asarray(result['F'], dtype=float), np.asarray(expected, dtype=float),
                       rtol=1e-10)


# --- exclusion regions ------------------------------------------------------------

def test_the_search_fits_only_the_points_outside_the_regions(app, tmp_path):
    _spectra(app, tmp_path, 2)
    request = {'kind': 'one_model', 'files': app.parse_process_path(),
               'exclusion_regions': ((-1.0, 1.0),)}
    data = iio.search_data(app, request)
    n = len(data['A_list'][0])
    assert len(data['A']) == 2 * n and data['lengths'] == [n, n]
    assert not np.any(np.abs(data['A_fit']) <= 1.0)
    assert sum(data['lengths_fit']) == len(data['A_fit']) < 2 * n

    request['exclusion_regions'] = ((-100.0, 100.0),)
    with pytest.raises(iio.SearchRefused, match="every point"):
        iio.search_data(app, request)


# --- the window afterwards -------------------------------------------------------

def test_after_a_search_with_the_model(app, tmp_path, short_minimizer):
    """Every spectrum in its own panel with the regions, the model in the
    results table as after a fit: Take result takes it, Save result refuses."""
    _spectra(app, tmp_path, 2)
    _sextet(app, free=(0, 1))
    model_io._set_row_param(app.params_table.row_widgets[1], 1, '=(0.05)', '-1', '1', 'False')
    app.toolbar.exclusion_action.trigger()
    app.exclusion_bar.text.setText("-0.5:0.5")
    assert app.exclusion_bar.commit()
    result = _search(app)
    assert result['exclusion_regions'] == ((-0.5, 0.5),)

    app.on_instrumental_finished(result)

    axes = spectrum_axes_of(app.figure)
    assert len(axes) == 2
    assert all(any(patch.get_gid() == EXCLUSION_GID for patch in ax.patches) for ax in axes)
    rt = app.results_table
    assert rt.from_search and rt.one_model is not None
    assert rt.current_model_list == ['baseline', 'Sextet', 'Nbaseline', 'Sextet']
    assert np.array_equal(rt.current_parameters, result['p'][:result['mod_p_len']])
    assert app.last_plot_data is None and app.inprogress is False

    app.save_path.setText(str(tmp_path / "out"))
    app.save_result_pressed()
    assert "not saved as a fit result" in app.log.toPlainText()
    assert not os.path.exists(str(tmp_path / "out_param.txt"))

    app.take_result()
    pt = app.params_table
    assert pt.get_model_list() == ['baseline', 'Sextet']
    value = pt.row_widgets[1].layout().itemAt(1).widget().layout().itemAt(1).widget().text()
    assert float(value) == pytest.approx(result['p'][T], abs=1e-4)
