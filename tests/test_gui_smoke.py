"""Head-less smoke tests for the SYNCmoss GUI.

These build the real :class:`PhysicsApp` main window (offscreen) and check what a
new window holds, the parameters table -> ``read_model`` bridge, saving and
reloading a model, the theme switch and synchronous spectrum display. The
asynchronous fit is covered by ``syncmoss_test.py`` (the ``--test`` flow).
"""
import pytest

from syncmoss.constants import number_of_baseline_parameters, numro
from syncmoss.model_io import read_model, _save_model_to_file, load_model_from_path
from syncmoss.supp_menu import theme_action_text

pytestmark = pytest.mark.gui


def test_a_new_window_starts_empty(physics_app):
    """Every row of the parameters table, nothing plotted or fitted yet, an
    empty results table, a model of the baseline alone -- and it resizes
    (resizeEvent reads the splitter). One window for all: each costs ~2 s."""
    w = physics_app
    assert len(w.params_table.row_widgets) == numro           # incl. the baseline row
    assert w.last_plot_data is None and w.last_fitting_data is None
    assert w._main_splitter is not None and w._left_panel_ideal_width is not None
    assert w.toggle_dat_ins_action is not None
    assert isinstance(w.gridcolor, str) and w.gridcolor

    rt = w.results_table
    assert rt.parameter_names == [] and rt.covariance_matrix is None
    assert rt.errors is None and rt.fit_parameters is None
    intensities, errors = rt._calculate_intensities()        # nothing to compute yet
    assert intensities.size == 0 and errors.size == 0

    model, p, *_ = read_model(w)
    assert model == []
    assert len(p) == number_of_baseline_parameters

    w.resize(1700, 950)
    w.resize(1300, 720)


def test_select_model_then_read_model(physics_app):
    w = physics_app
    w.params_table.select_model(1, "Sextet")

    # The model button text must reflect the selection.
    row_widget = w.params_table.row_widgets[1]
    model_btn = row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
    assert model_btn.text() == "Sextet"

    model, p, *_ = read_model(w)
    assert model == ["Sextet"]
    # baseline (8) + polarized Sextet (14) parameters.
    assert len(p) == number_of_baseline_parameters + 14


def test_fields_share_the_tables_two_validators(physics_app):
    """Every value and bound field uses one of two validators owned by the
    table. One validator per field was never freed: ~4000 per window, more at
    every model change -- the memory a parallel test run ran out of."""
    from PySide6.QtGui import QValidator
    from PySide6.QtWidgets import QLineEdit
    pt = physics_app.params_table
    for model in ("Sextet", "Distr", "Doublet"):
        pt.select_model(1, model)
    assert len(pt.findChildren(QValidator)) == 2
    used = {f.validator() for f in pt.findChildren(QLineEdit) if f.validator() is not None}
    assert used == {pt._value_validator, pt._bound_validator}


def test_save_model_drops_empty_rows_and_reloads(physics_app, tmp_path):
    """Saving must omit empty (None) rows; loading restores the same models."""
    w = physics_app
    # Two models placed in non-contiguous rows, leaving empty rows around/between.
    w.params_table.select_model(2, "Sextet")
    w.params_table.select_model(5, "Doublet")

    file_path = str(tmp_path / "model.mdl")
    _save_model_to_file(w, file_path)

    with open(file_path, encoding="utf-8") as fh:
        data_lines = [ln.rstrip("\n") for ln in fh
                      if ln.strip() and not ln.lstrip().startswith("#")]

    # Header (model names) holds only baseline + the two real models, no 'None'.
    model_names = data_lines[0].split("\t")
    assert model_names == ["baseline", "Sextet", "Doublet"]
    assert "None" not in model_names
    # One param-data line per kept row: header + colors + 3 param rows.
    assert len(data_lines) == 2 + 3

    # Reloading into a fresh table reproduces the same model sequence.
    load_model_from_path(w, file_path)
    assert "red" not in w.log.styleSheet().lower(), w.log.toPlainText()
    model, p, *_ = read_model(w)
    assert model == ["Sextet", "Doublet"]
    assert len(p) == number_of_baseline_parameters + 14 + 9


def test_theme_toggle_lives_in_the_supp_menu(physics_app):
    """The light/dark switch is a Supp-menu entry, not a top-row button."""
    w = physics_app
    assert not hasattr(w, 'theme_btn')
    assert w.theme_action in w.supp_menu.actions()

    before_dark = w._is_dark_mode
    before_text = w.theme_action.text()
    assert before_text == theme_action_text(before_dark)

    w.theme_action.trigger()
    assert w._is_dark_mode is (not before_dark)
    # The entry always names the mode the NEXT click switches to.
    assert w.theme_action.text() == theme_action_text(w._is_dark_mode)
    assert w.theme_action.text() != before_text

    w.theme_action.trigger()  # restore
    assert w._is_dark_mode is before_dark
    assert w.theme_action.text() == before_text


def test_show_spectrum_calibration_dat(physics_app):
    """Showing the tests' Calibration.dat must not raise or log an error."""
    w = physics_app
    w.process_path.setPlainText(repr([w.calibration_path]))
    w.show_pressed()  # synchronous

    # The log turns red on error; success leaves it non-red.
    assert "red" not in w.log.styleSheet().lower(), w.log.toPlainText()
    assert w.inprogress is False
