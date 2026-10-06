"""Multispectra settings -> Fit one spectrum.

One spectrum of the path box -- chosen by its number or its name, each field of
the dialog following the other -- or a file next to them, fitted on its own with
the table's model and that spectrum's own N, N1, ...; nothing is saved. Refused
for a model with Nbaseline rows, nothing at all with fewer than two spectra.

The fit itself is made up (what is tested is which spectrum is fitted and how);
the spectra are copies of the tests' frozen calibration spectrum.
"""
import os
import shutil

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

import syncmoss.syncmoss_main as sm
from syncmoss import model_io
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.one_spectrum import OneSpectrumDialog, find_spectrum

pytestmark = [pytest.mark.gui]

DELTA = NB + 1           # the Sextet's delta in row 1


def _spectra(app, tmp_path, n):
    """*n* copies of the spectrum in the path box, with N1 = 10, 20, ..."""
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        shutil.copy2(app.calibration_path, path)
        paths.append(path)
    app.process_path.setPlainText(repr([(path, 10.0 * (k + 1)) for k, path in enumerate(paths)]))
    return paths


# --- which spectrum a name stands for ----------------------------------------------

def test_a_name_finds_a_spectrum_of_the_path_box_or_a_file_next_to_them(tmp_path):
    entries = [(str(tmp_path / 'Fe_4K.dat'), (4.2,)), (str(tmp_path / 'Fe_77K.dat'), (77.0,))]
    (tmp_path / 'Fe_300K.dat').write_text('1 2\n')           # not in the path box
    assert find_spectrum(entries, 'Fe_77K') == (2, (77.0,), entries[1][0])
    assert find_spectrum(entries, ' Fe_77K.dat ') == (2, (77.0,), entries[1][0])
    assert find_spectrum(entries, 'Fe_300K') == (None, (), str(tmp_path / 'Fe_300K.dat'))
    assert find_spectrum(entries, 'Fe_300K.dat') == (None, (), str(tmp_path / 'Fe_300K.dat'))
    assert find_spectrum(entries, 'Fe_500K') is None
    assert find_spectrum(entries, '  ') is None


# --- the dialog -----------------------------------------------------------------------

def test_the_dialog_starts_at_the_first_spectrum_and_each_field_follows_the_other(physics_app,
                                                                                 tmp_path):
    entries = [(str(tmp_path / f'{name}.dat'), ()) for name in ('Fe_4K', 'Fe_77K', 'Fe_300K')]
    dialog = OneSpectrumDialog(physics_app, entries)
    assert dialog.number.value() == 1 and dialog.name.text() == 'Fe_4K'

    dialog.number.setValue(3)                                # a number puts in its name
    assert dialog.name.text() == 'Fe_300K'

    dialog.name.clear()
    QTest.keyClicks(dialog.name, 'Fe_77K')                   # a name puts in its number
    assert dialog.number.value() == 2

    QTest.keyClicks(dialog.name, '_other')                   # a name not in the path box
    assert dialog.number.value() == 0 and dialog.number.text() == '—'
    dialog.deleteLater()


def test_ok_does_not_close_the_dialog_on_a_spectrum_that_is_nowhere(physics_app, tmp_path):
    entries = [(str(tmp_path / 'a.dat'), ()), (str(tmp_path / 'b.dat'), ())]
    dialog = OneSpectrumDialog(physics_app, entries)
    dialog.name.setText('nowhere')
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted and dialog.chosen is None
    assert 'nowhere' in dialog.message.text()

    dialog.name.setText('b')
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.chosen == (2, (), entries[1][0])
    dialog.deleteLater()


# --- the action -------------------------------------------------------------------------

def _choose(monkeypatch, chosen, opened=None):
    """The dialog answering OK with *chosen*; *opened* records that it was shown."""

    class Chosen:
        def __init__(self, parent, entries):
            if opened is not None:
                opened.append(list(entries))
            self.chosen = chosen

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(sm, 'OneSpectrumDialog', Chosen)


def _fake_fit(monkeypatch, app):
    """fit_single_spectrum made up, run where it is called; returns its calls."""
    calls = []

    def fake(window, spectrum_file, pool, background=None, sequence_params=None,
             spectrum_parameters=None):
        calls.append((spectrum_file, spectrum_parameters))
        model, p = model_io.read_model(app)[:2]
        _, fix = model_io.read_bounds_and_fix(app, len(p))
        free = [i for i in range(len(p)) if i not in set(np.ravel(fix).astype(int))]
        errors = np.full(len(p), np.nan)
        errors[free] = 0.01
        A = np.linspace(-10.0, 10.0, 64)
        B = 1000.0 - 5 * np.exp(-A ** 2)
        drawn = sum(1 for m in model if m not in ('Expression', 'Distr', 'Corr', 'Recon'))
        return {'success': True, 'parameters': np.array(p, dtype=float), 'errors': errors,
                'chi2': 1.0, 'chi2_spread': 0.05, 'covariance_matrix': np.eye(len(free)) * 1e-4,
                'fix': np.array(fix), 'model': model, 'is_simultaneous': False, 'A': A, 'B': B,
                'SPC_f': B * 0.999, 'FS': [B * 0.998] * drawn, 'FS_pos': [[[]]] * drawn,
                'hires_diff': None, 'spectrum_file': spectrum_file, 'Distri_substituted': [],
                'Cor_substituted': [], 'Recon': [], 'instrumental_note': '',
                'spectrum_parameters': spectrum_parameters}

    monkeypatch.setattr(sm.fitting_io, 'fit_single_spectrum', fake)
    monkeypatch.setattr(sm.FittingThread, 'start', lambda thread: thread.run())
    return calls


def test_a_model_with_nbaseline_is_refused(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 3)
    app.params_table.select_model(1, 'Sextet')
    app.params_table.select_model(2, 'Nbaseline')
    opened = []
    _choose(monkeypatch, None, opened)
    app.fit_one_spectrum()
    assert opened == [] and 'red' in app.log.styleSheet().lower()
    assert 'Nbaseline' in app.log.toPlainText()


def test_one_spectrum_only_gets_a_message(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 1)
    app.params_table.select_model(1, 'Sextet')
    opened = []
    _choose(monkeypatch, None, opened)
    calls = _fake_fit(monkeypatch, app)
    app.fit_one_spectrum()
    assert opened == [] and calls == []
    assert 'only one spectrum' in app.log.toPlainText()


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def test_a_spectrum_of_the_path_box_brings_its_n_and_n1_into_the_formulas(physics_app, tmp_path,
                                                                         monkeypatch):
    """Spectrum 2 of 3 (N1 = 20): the minimiser gets its formula with N = 2 and
    N1 = 20 -- the real fit up to the minimiser, which only records."""
    app = physics_app
    paths = _spectra(app, tmp_path, 3)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, 'N*1000+N1', '', '', 'False')
    app.jn0_input.setText("16")
    app.pool = _SerialPool()
    _choose(monkeypatch, (2, (20.0,), paths[1]))
    seen = []

    def recording_minimiser(func, x, y, p0, **kwargs):
        seen.append(list(kwargs['Expr']))
        p0 = np.array(p0, dtype=float)
        fixed = {int(i) for i in np.ravel(kwargs['fix'])}
        free = [i for i in range(len(p0)) if i not in fixed]
        errors = np.full(len(p0), np.nan)
        errors[free] = 0.01
        return p0, errors, 1.0, np.eye(len(free)) * 1e-4

    monkeypatch.setattr(sm.fitting_io.mi, 'minimi_hi', recording_minimiser)
    monkeypatch.setattr(sm.FittingThread, 'start', lambda thread: thread.run())

    app.fit_one_spectrum()

    assert seen == [['(2)*1000+(20.0)']]
    assert app.results_table.current_spectrum_parameters.number == 2


def test_the_chosen_spectrum_is_fitted_with_its_own_values_and_nothing_is_saved(
        physics_app, tmp_path, monkeypatch):
    app = physics_app
    paths = _spectra(app, tmp_path, 3)
    app.params_table.select_model(1, 'Sextet')
    out = tmp_path / 'out'
    out.mkdir()
    app.save_path.setText(str(out / 'result'))
    opened = []
    _choose(monkeypatch, (2, (20.0,), paths[1]), opened)
    calls = _fake_fit(monkeypatch, app)

    app.fit_one_spectrum()

    assert len(opened) == 1 and [p for p, _ in opened[0]] == paths
    [(fitted, spectrum)] = calls
    assert fitted == paths[1] and spectrum.number == 2 and spectrum.values == (20.0,)
    assert app.last_plot_data['filepath'] == paths[1]
    assert os.listdir(out) == []                             # nothing saved
    assert app.inprogress is False

    # Save result afterwards names the row after the spectrum that was fitted
    app._save_result_files('new')
    with open(out / 'result_param.txt', encoding='utf-8') as f:
        assert f.read().splitlines()[1].startswith('spectrum_2.dat')


def test_a_file_next_to_the_spectra_is_fitted_unless_the_formulas_need_its_values(
        physics_app, tmp_path, monkeypatch):
    app = physics_app
    _spectra(app, tmp_path, 2)
    outside = str(tmp_path / 'outside.dat')
    shutil.copy2(app.calibration_path, outside)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    _choose(monkeypatch, (None, (), outside))
    calls = _fake_fit(monkeypatch, app)

    app.fit_one_spectrum()
    [(fitted, spectrum)] = calls
    assert fitted == outside and spectrum.values == ()

    pt.select_model(2, 'Expression')                         # now the formulas use N1
    model_io._set_row_param(pt.row_widgets[2], 0, f'p[{DELTA}]*N1', '', '', 'False')
    app.fit_one_spectrum()
    assert len(calls) == 1 and 'red' in app.log.styleSheet().lower()
    assert 'N1' in app.log.toPlainText() and 'outside.dat' in app.log.toPlainText()
