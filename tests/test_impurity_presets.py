"""The Be / KB impurity presets: Doublets kept in parameters/Be.txt and KB.txt.

The files describe the CURRENT state of the beamline (Be: the optics, KB: the
Nanoscope) and are edited when it changes -- in Supp -> "Set parameters of ...".
So every expected number below is read from the files, never written into the
test: they are meant to change.

The results table reports a Doublet equal to a preset as the 'Impurity' and
leaves it out of the intensity percentages of the sample's components. It
matches to 1e-4, so a model saved before the presets were rounded to four
decimals (the same values with longer digits) is still recognised.
"""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QMessageBox

from syncmoss import model_io, supp_menu
from syncmoss.parameters_table import DOUBLET_NAMES, format_parameter_value
from syncmoss.results_table import _PRESET_ATOL

from conftest import redirect_params_dir_to_tmp

_PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "syncmoss", "parameters")

# (model-menu entry, (file, Supp label)) for each preset
PRESETS = list(zip(("Be", "KB_nano"), supp_menu.IMPURITY_PRESETS))
IDS = [entry for entry, _ in PRESETS]


def _file_fields(app, file_name):
    with open(os.path.join(app.params_dir, file_name), encoding="utf-8") as f:
        return f.read().split()


def _value_input(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(1).widget()


def _percent_button(app):
    """Fill the results table the way a fit of the current table does and
    return the button that shows component 2's intensity (or 'Impurity')."""
    pt = app.params_table
    p = model_io.read_model(app)[1]
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    errors = np.full(len(p), 0.01)
    errors[fix] = np.nan
    n_free = int(np.sum(~np.isnan(errors)))
    rt = app.results_table
    rt.fill_table(p, pt.get_model_list(), pt.get_current_colors(), pt.get_parameter_names(),
                  np.eye(n_free) * 1e-4, errors, fix, pt.get_expression_texts())
    return rt.buttons[2 * 3 + 1]


def _drive_dialog(monkeypatch, edit=None, accept=True):
    """Stand in for the modal exec: record what the dialog shows, type *edit*
    ({field index: text}) and press OK or Cancel. Returns (shown, warnings)."""
    shown = {}

    def fake_exec(self):
        editors = self.findChildren(QLineEdit)
        shown['texts'] = [editor.text() for editor in editors]
        shown['labels'] = [label.text() for label in self.findChildren(QLabel)
                           if label.text() in DOUBLET_NAMES]
        for index, text in (edit or {}).items():
            editors[index].setText(text)
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warned.append(args[-1]))
    return shown, warned


# --- the files ---------------------------------------------------------------

@pytest.mark.parametrize("file_name", [f for f, _ in supp_menu.IMPURITY_PRESETS])
def test_each_preset_file_is_nine_numbers(file_name):
    """What every reader of the files expects (np.genfromtxt, tab-separated)."""
    values = np.genfromtxt(os.path.join(_PARAMS, file_name), delimiter='\t')
    assert values.shape == (len(DOUBLET_NAMES),)
    assert np.all(np.isfinite(values))


# --- recognised as the impurity ----------------------------------------------

@pytest.mark.gui
@pytest.mark.parametrize("entry, preset", PRESETS, ids=IDS)
def test_preset_row_is_the_impurity(physics_app, entry, preset):
    file_name, _label = preset
    a_file = np.genfromtxt(os.path.join(physics_app.params_dir, file_name), delimiter='\t')[7]
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, entry)
    assert _percent_button(physics_app).text() == 'Impurity'

    a_field = _value_input(pt, 2, 7)
    # the same value with longer digits, as a model saved before the rounding has it
    a_field.setText(format_parameter_value(a_file + 0.4 * _PRESET_ATOL, decimals=None))
    assert _percent_button(physics_app).text() == 'Impurity'

    # a Doublet that really differs is a component of the sample
    a_field.setText(format_parameter_value(a_file + 10 * _PRESET_ATOL, decimals=None))
    assert _percent_button(physics_app).text() != 'Impurity'


# --- the Supp editors ----------------------------------------------------------

@pytest.mark.gui
def test_supp_menu_opens_each_preset(physics_app, monkeypatch):
    actions = {action.text(): action for action in physics_app.supp_menu.actions()}
    for file_name, label in supp_menu.IMPURITY_PRESETS:
        shown, _ = _drive_dialog(monkeypatch, accept=False)
        actions[f"Set parameters of {label}"].trigger()
        assert shown['labels'] == list(DOUBLET_NAMES)
        assert shown['texts'] == _file_fields(physics_app, file_name)


@pytest.mark.gui
@pytest.mark.parametrize("entry, preset", PRESETS, ids=IDS)
def test_cancel_leaves_the_file_alone(physics_app, monkeypatch, tmp_path, entry, preset):
    file_name, label = preset
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    before = _file_fields(physics_app, file_name)
    _drive_dialog(monkeypatch, edit={0: '0.5'}, accept=False)
    assert supp_menu.open_impurity_preset_dialog(physics_app, file_name, label) is False
    assert _file_fields(physics_app, file_name) == before


@pytest.mark.gui
@pytest.mark.parametrize("entry, preset", PRESETS, ids=IDS)
def test_ok_writes_the_file_and_the_preset_follows(physics_app, monkeypatch, tmp_path,
                                                   entry, preset):
    file_name, label = preset
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    fields = _file_fields(physics_app, file_name)
    new_t = format_parameter_value(float(fields[0]) + 0.001, decimals=None)
    _drive_dialog(monkeypatch, edit={0: new_t})

    assert supp_menu.open_impurity_preset_dialog(physics_app, file_name, label) is True
    with open(os.path.join(physics_app.params_dir, file_name), encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines == ['\t'.join([new_t] + fields[1:])]   # one line, the others unchanged

    # the model menu now fills in the new value, and the results table knows it
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, entry)
    assert float(_value_input(pt, 2, 0).text()) == float(new_t)
    assert _percent_button(physics_app).text() == 'Impurity'


@pytest.mark.gui
def test_ok_without_a_change_writes_the_same_numbers(physics_app, monkeypatch, tmp_path):
    file_name, label = supp_menu.IMPURITY_PRESETS[0]
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    before = _file_fields(physics_app, file_name)
    _drive_dialog(monkeypatch)
    assert supp_menu.open_impurity_preset_dialog(physics_app, file_name, label) is True
    assert _file_fields(physics_app, file_name) == before


@pytest.mark.gui
def test_a_field_that_is_not_a_number_is_refused(physics_app, monkeypatch, tmp_path):
    file_name, label = supp_menu.IMPURITY_PRESETS[1]
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    before = _file_fields(physics_app, file_name)
    _, warned = _drive_dialog(monkeypatch, edit={3: ''})
    assert supp_menu.open_impurity_preset_dialog(physics_app, file_name, label) is False
    assert warned and DOUBLET_NAMES[3] in warned[0]
    assert _file_fields(physics_app, file_name) == before
