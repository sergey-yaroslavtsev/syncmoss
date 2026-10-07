"""The simultaneous one-model fit in the window.

One model, all the spectra of the path box at once: every free parameter shared,
an independent ``=(X)`` one with a value per spectrum, every spectrum its own
baseline. Checked here: the =(X) field, when Fit / Show model work this way (and
when they refuse), how a result is shown (results table, main plot, result
window), what "Take result" and "Save result" do with it, and a real fit.

Most tests use a made-up result -- what is tested is the window, not the
physics; the real fit is marked slow.
"""
import os
import re
import shutil

import numpy as np
import pytest
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QMessageBox

import syncmoss.syncmoss_main as sm
from syncmoss import fitting_io, model_io, one_model
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.models import FitInterrupted

from conftest import redirect_params_dir_to_tmp

pytestmark = [pytest.mark.gui]

SEXTET = 14
DELTA = NB + 1           # the Sextet's delta in row 1


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _value_input(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(1).widget()


def _fix_box(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(0).layout().itemAt(1).widget()


def _spectra(app, tmp_path, n):
    """*n* copies of the bundled spectrum in the path box, with N1 = 10, 20, ..."""
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        shutil.copy2(app.calibration_path, path)
        paths.append(path)
    app.process_path.setPlainText(repr([(path, 10.0 * (k + 1)) for k, path in enumerate(paths)]))
    return paths


def _sextet_with_independent_delta(app):
    """Row 1: a Sextet whose delta is independent, =(0.05)."""
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    model_io._set_row_param(pt.row_widgets[1], 1, '=(0.05)', '-1', '1', 'False')


def _fake_fit(app):
    """A finished one-model fit of the table over the path box, made up: every
    free parameter moved a little, the curves are flat lines. Returns the
    result dict as the fitting thread hands it over."""
    template = one_model.read_template(app)
    spectra = one_model.spectra_of(app.parse_process_entries())
    inputs = one_model.expand(template, spectra, [1000.0 + 100 * k for k in range(len(spectra))])
    p = inputs['p'].copy()
    fixed = (set(inputs['fix'].tolist()) | set(inputs['con1'].astype(int).tolist())
             | set(inputs['NExpr'].tolist()))
    free = [i for i in range(len(p)) if i not in fixed]
    p[free] += 0.001 * (np.array(free) + 1)
    for target, source, factor in zip(inputs['con1'], inputs['con2'], inputs['con3']):
        p[int(target)] = p[int(source)] * factor
    errors = np.full(len(p), np.nan)
    errors[free] = 0.01
    drawn = sum(1 for m in template['model'] if m not in ('Expression', 'Distr', 'Corr', 'Recon'))
    A = np.linspace(-10.0, 10.0, 64)
    curves = [1000.0 + 100 * k - 5 * np.exp(-A ** 2) for k in range(len(spectra))]
    result = {
        'success': True, 'parameters': p, 'errors': errors, 'chi2': 1.05, 'chi2_spread': 0.05,
        'covariance_matrix': np.eye(len(free)) * 1e-4, 'fix': np.array(sorted(fixed)),
        'model': inputs['model'], 'is_simultaneous': True,
        'A_list': [A] * len(spectra), 'B_list': curves, 'SPC_f_list': [c * 0.999 for c in curves],
        'FS_list': [[c * 0.998] * drawn for c in curves],
        'FS_pos_list': [[[[]]] * drawn for _ in curves],
        'hires_diff_list': [None] * len(spectra),
        'Distri_substituted': list(inputs['Distri']), 'Cor_substituted': list(inputs['Cor']),
        'Recon': list(inputs['Recon']), 'instrumental_note': '',
    }
    result['one_model'] = one_model.OneModelResult(template, spectra, inputs, result)
    return result


def _finish_fake_fit(app):
    """Snapshot the table as a fit start does, then hand over a made-up result."""
    app._fit_links_snapshot = app.params_table.get_link_snapshot()
    app._fit_model_snapshot = model_io.model_file_rows(app)
    result = _fake_fit(app)
    app.on_one_model_fit_done(result)
    return result['one_model']


# --- the independent value field --------------------------------------------------

def test_the_field_takes_an_independent_value(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    field = _value_input(pt, 1, 1)
    validator = field.validator()
    for text in ('=()', '=(0.5)', '=(-3)', '=(', '=(1.'):
        assert validator.validate(text, len(text))[0] != QValidator.State.Invalid, text
    for text in ('=(a)', '=(1)2', '=((1))'):
        assert validator.validate(text, len(text))[0] == QValidator.State.Invalid, text


def test_a_finished_independent_value_is_light_green(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    field = _value_input(pt, 1, 1)
    field.setText('=(0.5)')
    assert 'lightgreen' in field.styleSheet() and 'color: black' in field.styleSheet()
    field.setText('=()')
    assert field.styleSheet() == ''
    field.setText('=[9,1]')
    assert 'darkorange' in field.styleSheet()


def _menu_texts(pt, field):
    menu = pt.value_context_menu(field)
    try:
        return [(a.text(), a.isEnabled()) for a in menu.actions()]
    finally:
        menu.deleteLater()


def test_make_it_independent_is_offered_with_several_spectra_and_no_nbaseline(physics_app, tmp_path):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    field = _value_input(pt, 1, 1)
    independent = "make it independent (own value in every spectrum)"

    app.process_path.setPlainText(repr([app.calibration_path]))
    assert [t for t, _ in _menu_texts(pt, field)] == ["link to another parameter"]

    _spectra(app, tmp_path, 2)
    assert (independent, True) in _menu_texts(pt, field)
    assert independent not in [t for t, _ in _menu_texts(pt, _value_input(pt, 0, 0))]  # baseline

    pt.select_model(2, 'Distr')                      # its 'par' is structural (locked)
    assert (independent, False) in _menu_texts(pt, _value_input(pt, 2, 0))

    pt.select_model(3, 'Nbaseline')
    assert independent not in [t for t, _ in _menu_texts(pt, field)]


def test_make_it_independent_puts_empty_brackets_with_the_cursor_inside(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    field = _value_input(pt, 1, 2)
    before = field.text()
    pt.start_independent_value(field)
    assert field.text() == '=()' and field.cursorPosition() == 2   # nothing pre-filled
    field.insert('0.7')
    assert field.text() == '=(0.7)'
    field.undo()
    field.undo()
    assert field.text() == before


def test_read_model_reads_x_and_the_table_knows_the_independent_slots(physics_app):
    app = physics_app
    _sextet_with_independent_delta(app)
    p = model_io.read_model(app)[1]
    assert p[DELTA] == 0.05
    assert app.params_table.get_independent_slots() == [DELTA]
    assert app.params_table.get_link_snapshot()[DELTA] == '=(0.05)'


def test_an_unfinished_independent_value_blocks_the_start(physics_app):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Doublet')
    _value_input(pt, 1, 1).setText('=()')
    assert [s['reason'] for s in pt.get_empty_parameter_slots()] == ['unfinished independent']
    assert app.check_user_expressions("Fit") is False
    assert "without its start value" in app.log.toPlainText()


def test_the_start_value_must_be_inside_the_bounds(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    model_io._set_row_param(pt.row_widgets[1], 1, '=(5)', '-1', '1', 'False')
    outside = pt.get_out_of_bounds_parameters()
    assert [(s['row'], s['col'], s['upper']) for s in outside] == [(1, 1, '1')]


# --- when Fit / Show model work this way -------------------------------------------

@pytest.mark.parametrize("independent, mode, nbaseline, spectra, expected", [
    (True, 0, False, 2, (True, None)),
    (False, sm.ONE_MODEL, False, 2, (True, None)),
    (False, 0, False, 2, (False, None)),                 # a sequence
    (True, 0, False, 1, (False, None)),                  # one spectrum: X is a value
    (True, 0, True, 2, (False, 'refused')),
    (False, sm.ONE_MODEL, True, 2, (False, 'refused')),
    (False, 1, True, 2, (False, None)),                  # an Nbaseline fit
])
def test_one_model_request(physics_app, independent, mode, nbaseline, spectra, expected):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    if independent:
        _value_input(pt, 1, 1).setText('=(0.05)')
    if nbaseline:
        pt.select_model(2, 'Nbaseline')
        pt.select_model(3, 'Sextet')
    app.set_sequence_fitting_type(mode)
    wanted, reason = app.one_model_request(['a.dat', 'b.dat'][:spectra])
    assert wanted is expected[0]
    assert (reason is not None) == (expected[1] == 'refused')


def test_the_menu_shows_the_mode_on_the_button(physics_app):
    app = physics_app
    assert app.seq_fit_btn.text() == "Multispectra settings\n(sequence - initial)"
    app.set_sequence_fitting_type(1)
    assert app.seq_fit_btn.text() == "Multispectra settings\n(sequence - result)"
    app.set_sequence_fitting_type(sm.ONE_MODEL)
    assert app.seq_fit_btn.text() == "Multispectra settings\n(simultaneous - one model)"


def test_the_instrumental_search_with_independent_values_is_a_one_model_search(physics_app,
                                                                               tmp_path):
    """Several spectra: the model is expanded as for the one-model fit (the
    search itself: tests/test_instrumental_joint.py)."""
    app = physics_app
    _spectra(app, tmp_path, 2)
    _sextet_with_independent_delta(app)
    app.path_list = [app.calibration_path]
    request = app._instrumental_search_request(1)
    assert request['kind'] == 'one_model' and len(request['spectra']) == 2


# --- a result ----------------------------------------------------------------------

def test_a_result_fills_the_whole_expanded_model_and_shows_two_spectra(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    fit = _finish_fake_fit(app)

    rt = app.results_table
    assert rt.one_model is fit
    assert rt.current_model_list == ['baseline', 'Sextet', 'Nbaseline', 'Sextet', 'Nbaseline', 'Sextet']
    assert rt.interactive_table.rowCount() == 3 * 6
    data = app.last_plot_data
    assert len(data['A_list']) == 2
    assert data['labels'] == [fit.label(0), fit.label(2)] and data['labels'][1].startswith('3 of 3')
    assert app.inprogress is False
    window = app.result_window
    assert window is not None and window.slider.maximum() == 3


def test_take_result_writes_the_model_the_fit_started_from(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    fit = _finish_fake_fit(app)
    pt = app.params_table
    _value_input(pt, 1, 1).setText('=(0.2)')            # edited after the fit
    _value_input(pt, 1, 0).setText('7')

    app.take_result()
    assert pt.get_model_list() == ['baseline', 'Sextet']  # one spectrum's model
    assert _value_input(pt, 1, 1).text() == '=(0.05)'     # the X the fit started from
    assert float(_value_input(pt, 1, 0).text()) == pytest.approx(fit.values(0)[NB], abs=1e-4)
    assert float(_value_input(pt, 0, 0).text()) == pytest.approx(fit.values(0)[0], abs=1e-4)


def test_take_result_of_a_single_fit_keeps_the_brackets_around_the_fitted_value(physics_app):
    app = physics_app
    _sextet_with_independent_delta(app)
    pt = app.params_table
    p = model_io.read_model(app)[1]
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    p[DELTA] = 0.123456
    errors = np.full(len(p), 0.01)
    errors[fix] = np.nan
    rt = app.results_table
    rt.fill_table(p, pt.get_model_list(), pt.get_current_colors(), pt.get_parameter_names(),
                  np.eye(int(np.sum(~np.isnan(errors)))) * 1e-4, errors, fix, pt.get_expression_texts())
    rt.current_links = pt.get_link_snapshot()
    app.take_result()
    assert _value_input(pt, 1, 1).text() == '=(0.123456)'           # every digit ...
    assert _value_input(pt, 1, 1).displayText() == '=(0.123)'       # ... shown rounded


def test_a_click_on_a_spectrum_not_shown_offers_the_result_window(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    fit = _finish_fake_fit(app)
    window = app.result_window
    window.show_result(fit, 0)
    asked = []

    def answer(*args, **kwargs):
        asked.append(args[2])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    app.replot_result(3 * 3)                             # spectrum 2's Sextet (row 9)
    assert len(asked) == 1 and "2 of 3" in asked[0] and "not shown" in asked[0]
    assert window.spectrum() == 1


def test_a_click_on_a_shown_spectrum_brings_its_component_up(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    _finish_fake_fit(app)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("no question"))
    app.replot_result(3 * 5)                             # spectrum 3's Sextet: shown
    assert app.last_plot_data['z_order'] is not None
    assert "Brought 'Sextet' to top" in app.log.toPlainText()


def test_the_result_window_shows_every_spectrum_with_its_own_part(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    fit = _finish_fake_fit(app)
    window = app.result_window
    for k in range(3):
        window.slider.setValue(k + 1)
        assert window.spectrum_label.text() == fit.label(k)
        assert window.parameters_line.text() == f"N = {k + 1}   N1 = {10 * (k + 1):g}"
        assert window.parameters_line.isReadOnly()
        table = window.table
        assert table.current_model_list == ['baseline', 'Sextet']
        assert table.interactive_table.rowCount() == 6
        assert np.array_equal(table.current_parameters, fit.values(k))
    assert window.figure.axes and window.figure.axes[0].texts
    # one spectrum's block of the joint covariance is no correlation matrix
    assert not window.table.tabs.isTabVisible(1)


def test_the_result_window_shows_a_spectrum_s_formula_as_the_whole_model_has_it(physics_app,
                                                                                tmp_path):
    """Spectrum 2's Expression with the slot of its own (independent) delta and
    its N1 in place -- the formula of the fitted model -- evaluated with the
    whole fit."""
    app = physics_app
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    pt = app.params_table
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, f'p[{DELTA}]*N1', '', '', 'False')
    fit = _finish_fake_fit(app)
    window = app.result_window
    window.slider.setValue(2)
    table = window.table
    delta_2 = (NB + SEXTET + 1) + DELTA                   # spectrum 2's copy of delta
    rows = [3 * 2 + row for row in range(3)]              # the Expression's three rows
    cells = [table._cell_text(table.interactive_table, row, 1) for row in rows]
    assert cells == [f'p[{delta_2}]*(20.0)', f"{fit.p[delta_2] * 20.0:.3f}", "±0.200"]
    assert fit.template_view()['texts'][2] == f'p[{DELTA}]*N1'    # Take result: as typed


def test_save_writes_a_line_per_spectrum_and_the_model_the_fit_started_from(physics_app, tmp_path):
    app = physics_app
    paths = _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    _finish_fake_fit(app)
    out = tmp_path / "out"
    out.mkdir()
    app.save_path.setText(str(out / "series"))

    app._save_result_files('new')

    with open(out / "series_param.txt", encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert len(lines) == 1 + 3
    assert lines[0].startswith('#File\tN\tN1')
    assert [line.split('\t')[0] for line in lines[1:]] == [os.path.basename(p) for p in paths]
    for k, path in enumerate(paths):
        base = out / os.path.splitext(os.path.basename(path))[0]
        for suffix in ('_graf.txt', '_combo.png', '.svg'):
            assert os.path.exists(str(base) + suffix), suffix
    with open(out / "series_result_table_PNG.html", encoding='utf-8') as f:
        html = f.read()
    assert len(re.findall(r'<h3>', html)) == 3 and 'N1 = 30' in html
    assert os.path.exists(out / "series_inputs.txt")
    with open(out / "series_result_model.mdl", encoding='utf-8') as f:
        assert '=(0.05)' in f.read()


def test_the_result_crosses_from_the_fitting_thread_intact(physics_app, tmp_path, monkeypatch):
    """A real thread: the result -- arrays and the OneModelResult -- reaches the
    window through the queued 'done' signal as it was made."""
    from PySide6.QtCore import QCoreApplication
    app = physics_app
    _spectra(app, tmp_path, 2)
    _sextet_with_independent_delta(app)
    app._fit_links_snapshot = app.params_table.get_link_snapshot()
    app._fit_model_snapshot = model_io.model_file_rows(app)
    prepared = _fake_fit(app)
    monkeypatch.setattr(one_model, 'fit', lambda *a, **k: prepared)
    thread = sm.OneModelFitThread(app, None, None, None)
    thread.done.connect(app.on_one_model_fit_done)
    app.inprogress = True
    thread.start()
    assert thread.wait(10000)
    for _ in range(5):
        QCoreApplication.processEvents()
    assert app.results_table.one_model is prepared['one_model']
    assert app.inprogress is False


def test_an_interrupted_one_model_fit_releases_the_window(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 2)
    _sextet_with_independent_delta(app)
    app.initialize_parameters()

    def interrupted(*args, **kwargs):
        raise FitInterrupted("Interrupted")

    monkeypatch.setattr(one_model, 'fit', interrupted)
    monkeypatch.setattr(sm.OneModelFitThread, 'start', lambda thread: thread.run())
    monkeypatch.setattr(app, 'confirm_instrumental_methods', lambda *a: True)
    app.inprogress = True
    app.start_one_model_fit()
    assert app.inprogress is False
    assert "Interrupted" in app.log.toPlainText()


# --- with real curves -----------------------------------------------------------------

def test_show_model_draws_the_first_and_the_last_spectrum(physics_app, tmp_path, monkeypatch):
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    _spectra(app, tmp_path, 3)
    _sextet_with_independent_delta(app)
    app.jn0_input.setText("16")
    app.pool = _SerialPool()
    monkeypatch.setattr(sm.ShowModelThread, 'start', lambda thread: thread.run())

    app.showM_pressed()
    data = app.last_plot_data
    assert data['has_nbaseline'] and data['model'] == ['Sextet', 'Nbaseline', 'Sextet']
    assert data['labels'][0].startswith('1 of 3') and data['labels'][1].startswith('3 of 3')
    assert app.results_table.interactive_table.rowCount() == 0
    assert "red" not in app.log.styleSheet().lower(), app.log.toPlainText()


def test_show_model_of_a_sequence_draws_the_first_and_the_last_spectrum(physics_app, tmp_path,
                                                                       monkeypatch):
    """No =(X), a sequence mode: the table's model on the first and the last
    spectrum, each with its own baseline and N values -- where the sequence
    starts every spectrum from."""
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    _spectra(app, tmp_path, 3)
    app.params_table.select_model(1, 'Sextet')
    app.set_sequence_fitting_type(0)
    app.jn0_input.setText("16")
    app.pool = _SerialPool()
    monkeypatch.setattr(sm.ShowModelThread, 'start', lambda thread: thread.run())

    app.showM_pressed()
    data = app.last_plot_data
    assert data['has_nbaseline'] and data['model'] == ['Sextet', 'Nbaseline', 'Sextet']
    assert data['labels'][0].startswith('1 of 3') and data['labels'][1].startswith('3 of 3')
    assert len(app.figure.axes) == 2
    assert "red" not in app.log.styleSheet().lower(), app.log.toPlainText()


@pytest.mark.slow
def test_two_identical_spectra_fit_like_one_of_them(physics_app, tmp_path):
    """The same spectrum twice: the one-model fit gives each copy the delta a
    fit of the spectrum alone gives, and the shared T is that fit's T."""
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    paths = _spectra(app, tmp_path, 2)
    _sextet_with_independent_delta(app)
    pt = app.params_table
    for col in range(2, SEXTET):                         # T and delta (and Ns) free only
        _fix_box(pt, 1, col).setChecked(True)
    app.jn0_input.setText("16")
    app.initialize_parameters()
    pool = _SerialPool()

    template = one_model.read_template(app)
    spectra = one_model.spectra_of(app.parse_process_entries())
    joint = one_model.fit(app, template, spectra, pool)
    assert joint['success'], joint.get('message')
    fit = joint['one_model']

    single = fitting_io.fit_single_spectrum(app, paths[0], pool)
    assert single['success'], single.get('message')
    alone = single['parameters']

    assert abs(alone[DELTA] - 0.05) > 0.01 and alone[NB] > 2       # the fit did move
    for k in (0, 1):
        assert fit.values(k)[DELTA] == pytest.approx(alone[DELTA], abs=1e-4)  # error ~2e-3
        assert fit.values(k)[NB] == pytest.approx(alone[NB], rel=1e-3)     # shared T
        assert fit.values(k)[0] == pytest.approx(alone[0], rel=1e-4)       # own Ns
    assert fit.chi2 == pytest.approx(single['chi2'], rel=0.01)
