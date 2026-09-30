"""Saving a result also saves the fitted model.

A ``_param.txt`` on its own is not reproducible — the parameters it lists mean
nothing without the model rows they belong to — so "Save result" / "Save result
as" write ``<base>_result_model.mdl`` next to the other artifacts: the model the
fit started from (taken at fit start) with every free parameter at its fitted
value, and the links, fixed values, bounds and expressions exactly as they were
fitted. The name is its own so a result save can never overwrite the
``<base>.mdl`` that "Save model" writes, and so is its overwrite question:
declining it must not touch what was already written.

``<base>`` is the save path without the spectrum's extension (``Fe_4.2K.dat``
-> ``Fe_4.2K``), and never cut at the dot of a sample name (``Fe_4.2K`` stays).
"""
import copy
import os

import numpy as np
import pytest

from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss import model_io
import syncmoss.syncmoss_main as syncmoss_main


# --- naming -------------------------------------------------------------------

@pytest.mark.parametrize("path, expected", [
    ('Fe_4.2K.dat', 'Fe_4.2K'),
    ('Fe_4.2K', 'Fe_4.2K'),                  # the sample name's own dot stays
    ('run1.TXT', 'run1'),
    ('sample.001', 'sample.001'),            # not a file type SYNCmoss knows
    (os.path.join('my.data', 'run1'), os.path.join('my.data', 'run1')),
])
def test_only_a_known_extension_is_stripped(path, expected):
    assert model_io.strip_known_extension(path) == expected


def test_model_path_replaces_the_extension_of_the_save_path(tmp_path):
    work = str(tmp_path)
    base = os.path.join(work, 'data', 'run1')
    assert model_io.model_path_for_save_path(base, work) == base + '.mdl'
    assert model_io.model_path_for_save_path(base + '.txt', work) == base + '.mdl'


def test_model_path_keeps_a_dot_in_a_directory_name(tmp_path):
    """splitext, not rsplit('.') — a dotted folder is not an extension."""
    work = str(tmp_path)
    base = os.path.join(work, 'my.data', 'run1')
    assert model_io.model_path_for_save_path(base, work) == base + '.mdl'


def test_model_path_keeps_the_dot_of_a_sample_name(tmp_path):
    """"Save model" used to write Fe_4.mdl for a save path Fe_4.2K."""
    work = str(tmp_path)
    base = os.path.join(work, 'Fe_4.2K')
    assert model_io.model_path_for_save_path(base, work) == base + '.mdl'
    assert model_io.model_path_for_save_path(base + '.dat', work) == base + '.mdl'


def test_model_path_falls_back_to_the_workfolder(tmp_path):
    work = str(tmp_path)
    assert model_io.model_path_for_save_path('', work) == os.path.join(work, 'model.mdl')
    assert model_io.model_path_for_save_path(work, work) == os.path.join(work, 'model.mdl')
    assert model_io.model_path_for_save_path('run1', work) == os.path.join(work, 'run1.mdl')


def test_the_result_model_shares_the_base_of_the_result_files(tmp_path):
    """<base>_result_model.mdl next to <base>_param.txt — the same <base>."""
    work = str(tmp_path)
    for base in (os.path.join(work, 'run1'), os.path.join(work, 'Fe_4.2K'), 'run1'):
        assert model_io.result_model_path(base) == base + '_result_model.mdl'


# --- the fitted model, as data ------------------------------------------------

def _field(value, fix=False, lower='', upper=''):
    return [value, lower, upper, '', 'True' if fix else 'False']


def _snapshot():
    """A model_file_rows() result: baseline, Sextet, Distr, Recon, Expression.

    Flat slots 0-7 baseline, 8-21 Sextet, 22-26 Distr, 27-33 Recon, 34 Expression.
    """
    baseline = [_field('1000')] + [_field('0', fix=True)] * (NB - 1)
    sextet = [_field('1')] * 14 + [_field('')] * 3      # + unused columns of the row
    sextet[1] = _field('=[8,1]')                        # a link
    sextet[2] = _field('0.5', fix=True)                 # a fixed value
    sextet[3] = _field('1', lower='0', upper='2')       # bounds
    # the Distr's PDF text is '1': a number, but a text slot all the same
    distr = [_field('3'), _field('0'), _field('10'), _field('20', fix=True), _field('1')]
    recon = [_field('3'), _field('0'), _field('10'), _field('4', fix=True),
             _field('0'), _field('0'), _field('0.25,0.25,0.25,0.25')]
    expression = [_field('p[8]*2')]
    names = ['baseline', 'Sextet', 'Distr', 'Recon', 'Expression']
    return names, ['red'] * len(names), [baseline, sextet, distr, recon, expression]


def test_free_numbers_take_the_fitted_values_everything_else_stays():
    snapshot = _snapshot()
    before = copy.deepcopy(snapshot)
    p = np.arange(35, dtype=float) + 0.5

    names, colors, rows = model_io.fitted_model_rows(
        snapshot, p, [np.array([0.1, 0.2, 0.3, 0.4])])

    values = [[field[0] for field in row] for row in rows]
    assert values[0] == ['0.5'] + ['0'] * (NB - 1)
    assert values[1][:5] == ['8.5', '=[8,1]', '0.5', '11.5', '12.5']
    assert values[1][13:] == ['21.5', '', '', '']
    assert values[2] == ['22.5', '23.5', '24.5', '20', '1']
    assert values[3] == ['27.5', '28.5', '29.5', '4', '31.5', '32.5', '0.1,0.2,0.3,0.4']
    assert values[4] == ['p[8]*2']
    assert rows[1][3][1:3] == ['0', '2']              # bounds kept
    assert rows[1][2][4] == 'True'                    # fix kept
    assert (names, colors) == (before[0], before[1])
    assert snapshot == before                         # the snapshot itself is untouched


def test_fitted_values_are_written_to_full_precision():
    """The same digits as _param.txt, so the file reproduces the fit exactly."""
    p = np.arange(35, dtype=float) + 0.5
    p[8] = 1.0 / 3.0
    _, _, rows = model_io.fitted_model_rows(_snapshot(), p)
    assert float(rows[1][0][0]) == p[8]
    assert rows[1][0][0] == str(p[8])


def test_no_snapshot_means_no_fitted_model():
    assert model_io.fitted_model_rows(None, np.zeros(3)) is None


# --- the fitted model, through the GUI -----------------------------------------

def _cell(pt, row, col):
    """(value, lower, upper, fixed) of one table cell (model_io's layout contract)."""
    param_widget = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    bounds = param_widget.layout().itemAt(2).layout()
    fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
    return (param_widget.layout().itemAt(1).widget().text(),
            bounds.itemAt(0).widget().text(), bounds.itemAt(1).widget().text(),
            fix_cb.isChecked())


def _fit(app):
    """A baseline + Sextet fit the way the GUI runs one: the model snapshot that
    fit_pressed takes, then the real on_fitting_finished with the result.

    The Sextet carries a link, a fixed value and bounds, and the fitted vector
    differs from every start value, so each rule shows in the saved model.
    """
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    row = pt.row_widgets[1]
    model_io._set_row_param(row, 0, '1', '', '', 'False')         # free
    model_io._set_row_param(row, 1, '=[8,1]', '', '', 'False')    # a link
    model_io._set_row_param(row, 2, '0.5', '', '', 'True')        # a fixed value
    model_io._set_row_param(row, 3, '1', '0', '2', 'False')       # free, with bounds
    app._fit_model_snapshot = model_io.model_file_rows(app)       # what fit_pressed does

    n = len(model_io.read_model(app)[1])
    p = np.arange(n, dtype=float) + 0.5
    app.on_fitting_finished({
        'success': True, 'parameters': p, 'errors': np.full(n, 0.01),
        'chi2': 1.0, 'covariance_matrix': np.eye(n), 'fix': np.array([], dtype=int),
        'Recon': [],
    })
    return p


@pytest.mark.gui
def test_the_saved_model_is_the_fitted_one(physics_app, tmp_path):
    app = physics_app
    p = _fit(app)
    base = os.path.join(str(tmp_path), 'run1')
    app.save_path.setText(base)

    app._save_result_files('new')

    assert os.path.exists(base + '_param.txt')
    assert not os.path.exists(base + '.mdl')    # the user's own Save model file
    assert 'run1_result_model.mdl' in app.log.toPlainText()

    # Loading it puts the RESULT into the table, links and all
    model_io.load_model_from_path(app, base + '_result_model.mdl')
    pt = app.params_table
    assert pt.get_model_list()[:2] == ['baseline', 'Sextet']
    assert _cell(pt, 1, 0) == (str(p[NB]), '', '', False)
    assert _cell(pt, 1, 1)[0] == '=[8,1]'
    assert _cell(pt, 1, 2) == ('0.5', '', '', True)
    assert _cell(pt, 1, 3) == (str(p[NB + 3]), '0', '2', False)


@pytest.mark.gui
def test_editing_the_table_after_the_fit_does_not_reach_the_saved_model(physics_app, tmp_path):
    app = physics_app
    _fit(app)
    pt = app.params_table
    pt.select_model(2, 'Doublet')                                   # the next model...
    model_io._set_row_param(pt.row_widgets[1], 2, '9', '', '', 'False')   # ...and an edit
    base = os.path.join(str(tmp_path), 'run1')
    app.save_path.setText(base)

    app._save_result_files('new')

    model_io.load_model_from_path(app, base + '_result_model.mdl')
    assert 'Doublet' not in pt.get_model_list()
    assert _cell(pt, 1, 2) == ('0.5', '', '', True)


@pytest.mark.gui
def test_the_result_model_never_overwrites_a_hand_saved_model(physics_app, tmp_path):
    """"Save model" writes <base>.mdl; saving a result must not touch it."""
    app = physics_app
    _fit(app)
    base = os.path.join(str(tmp_path), 'run1')
    app.save_path.setText(base)
    with open(base + '.mdl', 'w', encoding='utf-8') as f:
        f.write('my own model\n')

    app._save_result_files('new')

    with open(base + '.mdl', encoding='utf-8') as f:
        assert f.read() == 'my own model\n'
    assert os.path.exists(base + '_result_model.mdl')


@pytest.mark.gui
def test_declining_the_model_overwrite_keeps_the_result_files(physics_app, tmp_path, monkeypatch):
    app = physics_app
    _fit(app)
    base = os.path.join(str(tmp_path), 'run1')
    app.save_path.setText(base)

    with open(base + '_result_model.mdl', 'w', encoding='utf-8') as f:
        f.write('older model\n')

    # The model asks its own overwrite question — answer "No".
    asked = []
    monkeypatch.setattr(model_io.QMessageBox, 'question',
                        lambda *a, **k: asked.append(a) or model_io.QMessageBox.StandardButton.No)
    app._save_result_files('new')

    assert len(asked) == 1
    assert os.path.exists(base + '_param.txt')          # result files still written
    with open(base + '_result_model.mdl', encoding='utf-8') as f:
        assert f.read() == 'older model\n'              # model untouched
    status = app.log.toPlainText()
    assert 'Model NOT saved' in status
    assert 'canceled' in status                         # the model's own reason kept


@pytest.mark.gui
def test_the_spectrum_extension_is_not_part_of_the_result_names(physics_app, tmp_path):
    """The save path is usually the spectrum itself: Fe_4.2K.dat -> Fe_4.2K_*."""
    app = physics_app
    _fit(app)
    base = os.path.join(str(tmp_path), 'Fe_4.2K')
    app.save_path.setText(base + '.dat')

    app._save_result_files('new')

    assert os.path.exists(base + '_param.txt')
    assert os.path.exists(base + '_result_model.mdl')
    assert not os.path.exists(base + '.dat_param.txt')


@pytest.mark.gui
def test_a_dotted_save_path_keeps_its_full_name(physics_app, tmp_path):
    """Fe_4.2K_param.txt goes with Fe_4.2K_result_model.mdl, never Fe_4_*."""
    app = physics_app
    _fit(app)
    base = os.path.join(str(tmp_path), 'Fe_4.2K')
    app.save_path.setText(base)

    app._save_result_files('new')

    assert os.path.exists(base + '_param.txt')
    assert os.path.exists(base + '_result_model.mdl')
    assert not os.path.exists(os.path.join(str(tmp_path), 'Fe_4_param.txt'))


@pytest.mark.gui
@pytest.mark.parametrize("typed", ['Fe_4.2K', 'Fe_4.2K.txt'])
def test_save_result_as_keeps_a_dotted_name(physics_app, tmp_path, monkeypatch, typed):
    """It used to strip the ".2K" as an extension: Fe_4_param.txt."""
    app = physics_app
    _fit(app)
    base = os.path.join(str(tmp_path), 'Fe_4.2K')
    monkeypatch.setattr(syncmoss_main.QFileDialog, 'getSaveFileName',
                        lambda *a, **k: (os.path.join(str(tmp_path), typed), ''))

    app.save_result_as_pressed()

    assert app.save_path.text() == base
    assert os.path.exists(base + '_param.txt')
    assert os.path.exists(base + '_result_model.mdl')


@pytest.mark.gui
def test_sequential_results_are_named_after_the_spectrum(physics_app, tmp_path):
    app = physics_app
    _fit(app)
    app.save_path.setText(os.path.join(str(tmp_path), 'first.dat'))

    app._save_sequential_result_files(os.path.join('elsewhere', 'Fe_4.2K.dat'))

    assert os.path.exists(os.path.join(str(tmp_path), 'Fe_4.2K_param.txt'))


@pytest.mark.gui
def test_a_result_without_a_fitted_model_saves_the_results_only(physics_app, tmp_path):
    """A results table filled without a fit has nothing to write as its model."""
    app = physics_app
    p = np.arange(NB, dtype=float) + 1.0
    app.results_table.current_parameters = p
    app.results_table.current_errors = np.zeros_like(p)
    app.results_table.current_model_list = ['baseline']
    app.results_table.current_parameter_names = [['N', 'O', 'c2', 'lin', 'a', 'b', 'c', 'd']]
    base = os.path.join(str(tmp_path), 'run1')
    app.save_path.setText(base)

    app._save_result_files('new')

    assert os.path.exists(base + '_param.txt')
    assert not os.path.exists(base + '_result_model.mdl')
    assert 'Model NOT saved' in app.log.toPlainText()
