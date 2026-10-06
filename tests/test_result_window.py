"""The result window after a sequence and after an Nbaseline fit, and the grey
Nbaseline row.

The window (result_window.ResultWindow) shows the spectra of a fit of several
spectra one at a time. After a sequence every spectrum was fitted on its own, so
it shows each one's own chi2 and correlation matrix; after a simultaneous fit
one spectrum's block of the joint covariance is no correlation matrix, so that
tab is hidden. The one-model fit's window is in test_one_model_gui.

Made-up results throughout: what is tested is the window, not the physics.
"""
import shutil

import numpy as np
import pytest

import syncmoss.syncmoss_main as sm
from syncmoss import model_io
from syncmoss.constants import model_colors, number_of_baseline_parameters as NB, NBASELINE_COLOR

from conftest import redirect_params_dir_to_tmp

pytestmark = [pytest.mark.gui]

SEXTET = 14


def _spectra(app, tmp_path, n):
    """*n* copies of the bundled spectrum in the path box, with N1 = 10, 20, ..."""
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        shutil.copy2(app.calibration_path, path)
        paths.append(path)
    app.process_path.setPlainText(repr([(path, 10.0 * (k + 1)) for k, path in enumerate(paths)]))
    return paths


def _fit_of_the_table(app, shift):
    """Values, errors and covariance a fit of the current table could give:
    every free parameter moved by *shift*."""
    model, p, con1, _, _, _, _, _, NExpr, DistriN, _, ReconN = model_io.read_model(app)
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    fixed = set(np.concatenate([fix, con1, DistriN, NExpr, ReconN]).astype(int).tolist())
    free = [i for i in range(len(p)) if i not in fixed]
    p = np.array(p, dtype=float)
    p[free] += shift
    errors = np.full(len(p), np.nan)
    errors[free] = 0.01
    return model, p, errors, np.eye(len(free)) * 1e-4, np.array(sorted(fixed))


def _curve(k, drawn):
    A = np.linspace(-10.0, 10.0, 64)
    B = 1000.0 + 100 * k - 5 * np.exp(-A ** 2)
    return A, B, B * 0.999, [B * 0.998] * drawn, [[[]]] * drawn


# --- the grey Nbaseline row ---------------------------------------------------

def _color_button(pt, row):
    return pt.row_widgets[row].layout().itemAt(0).widget().layout().itemAt(0).widget()


def test_selecting_nbaseline_greys_the_row(physics_app):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    assert app.model_colors[2] == NBASELINE_COLOR
    assert NBASELINE_COLOR in _color_button(pt, 2).styleSheet()
    assert app.model_colors[1] == model_colors[0]       # other rows are left alone


# --- after an Nbaseline fit -----------------------------------------------------

def _nbaseline_result(app, paths):
    """A made-up simultaneous fit of the table: Sextet | Nbaseline + Doublet."""
    model, p, errors, covariance, fix = _fit_of_the_table(app, 0.01)
    curves = [_curve(k, 1) for k in range(2)]
    return {
        'success': True, 'parameters': p, 'errors': errors, 'chi2': 1.07, 'chi2_spread': 0.04,
        'covariance_matrix': covariance, 'fix': fix, 'model': model, 'is_simultaneous': True,
        'A_list': [c[0] for c in curves], 'B_list': [c[1] for c in curves],
        'SPC_f_list': [c[2] for c in curves], 'FS_list': [c[3] for c in curves],
        'FS_pos_list': [c[4] for c in curves], 'hires_diff_list': [None, None],
        'model_separate': [['Sextet'], ['Doublet']], 'begining_spc': [0, NB + SEXTET],
        'spectrum_files': paths, 'Distri': [], 'Cor': [], 'Distri_substituted': [],
        'Cor_substituted': [], 'Recon': [], 'instrumental_note': '',
    }


def test_an_nbaseline_fit_shows_every_spectrum_without_a_correlation_matrix(physics_app, tmp_path):
    app = physics_app
    paths = _spectra(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    result = _nbaseline_result(app, paths)

    app.on_fitting_finished(result)
    window = app.result_window
    assert window is not None and window.slider.maximum() == 2
    assert not window.table.tabs.isTabVisible(1)
    window.slider.setValue(2)
    assert window.spectrum_label.text().startswith('2 of 2')
    assert window.parameters_line.text() == 'N = 2   N1 = 20'
    table = window.table
    assert table.current_model_list == ['baseline', 'Doublet']   # its own components
    assert np.array_equal(table.current_parameters, result['parameters'][NB + SEXTET:])
    assert '1.070' in window.figure.axes[0].get_title()           # the fit's one chi2


def test_a_spectrum_s_formula_is_shown_as_typed_and_evaluated_with_the_whole_fit(physics_app,
                                                                                tmp_path):
    """Spectrum 2's Expression uses its own delta and spectrum 1's: its window
    shows it as typed, with the value and the error of the whole fit."""
    app = physics_app
    paths = _spectra(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    pt.select_model(4, 'Expression')
    delta_1, delta_2 = NB + 1, NB + SEXTET + NB + 1          # p[9] and p[31]
    model_io._set_row_param(pt.row_widgets[1], 1, '0.1', '-1', '1', 'False')
    model_io._set_row_param(pt.row_widgets[3], 1, '0.35', '-1', '1', 'False')
    typed = f'p[{delta_2}]-p[{delta_1}]'
    model_io._set_row_param(pt.row_widgets[4], 0, typed, '', '', 'False')
    result = _nbaseline_result(app, paths)
    result['model_separate'] = [['Sextet'], ['Doublet', 'Expression']]

    app.on_fitting_finished(result)
    window = app.result_window
    window.slider.setValue(2)
    table = window.table
    assert table.current_model_list == ['baseline', 'Doublet', 'Expression']
    rows = [3 * 2 + row for row in range(3)]                  # the Expression's three rows
    cells = [table._cell_text(table.interactive_table, row, 1) for row in rows]
    p = result['parameters']
    assert cells == [typed, f"{p[delta_2] - p[delta_1]:.3f}", "±0.014"]   # error: both deltas'
    assert cells[1] == "0.250"


# --- after a sequence -----------------------------------------------------------------

def test_a_sequence_shows_every_spectrum_with_its_own_correlation_matrix(physics_app, tmp_path, monkeypatch):
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    paths = _spectra(app, tmp_path, 3)
    app.params_table.select_model(1, 'Sextet')
    app.initialize_parameters()
    out = tmp_path / "out"
    out.mkdir()
    app.save_path.setText(str(out / "run"))

    def fake_fit(window, spectrum_file, pool, background=None, sequence_params=None,
                 spectrum_parameters=None):
        k = paths.index(spectrum_file)
        model, p, errors, covariance, fix = _fit_of_the_table(app, 0.01 * (k + 1))
        A, B, fit, components, positions = _curve(k, 1)
        return {'success': True, 'parameters': p, 'errors': errors, 'chi2': 1.0 + k,
                'chi2_spread': 0.05, 'covariance_matrix': covariance, 'fix': fix,
                'model': model, 'is_simultaneous': False, 'A': A, 'B': B, 'SPC_f': fit,
                'FS': components, 'FS_pos': positions, 'hires_diff': None,
                'spectrum_file': spectrum_file, 'Distri': [], 'Cor': [],
                'Distri_substituted': [], 'Cor_substituted': [], 'Recon': [],
                'instrumental_note': '', 'spectrum_parameters': spectrum_parameters}

    monkeypatch.setattr(sm.fitting_io, 'fit_single_spectrum', fake_fit)
    monkeypatch.setattr(sm.SequentialFittingThread, 'start', lambda thread: thread.run())
    app.inprogress = True
    app.start_sequential_fitting(paths, app.parse_process_entries())

    window = app.result_window
    assert window is not None and window.slider.maximum() == 3
    assert window.table.tabs.isTabVisible(1)
    window.slider.setValue(2)
    assert window.spectrum_label.text().startswith('2 of 3')
    assert window.parameters_line.text() == 'N = 2   N1 = 20'
    assert '2.000' in window.figure.axes[0].get_title()           # spectrum 2's own chi2
    free = int(np.sum(~np.isnan(window.table.current_errors)))
    assert window.table.correlation_table.rowCount() == free + 2
    assert app.inprogress is False
