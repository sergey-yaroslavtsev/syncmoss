"""Head-less smoke tests for the SYNCmoss GUI.

These build the real :class:`PhysicsApp` main window (offscreen) and exercise the
GUI-facing paths that the refactor touched: lazy-attribute initialisation, the
parameters table -> ``read_model`` bridge, the ``resizeEvent`` guard, the results
table guards, and synchronous spectrum display. They intentionally avoid the
asynchronous fit, which needs a multiprocessing pool and is covered by the manual
smoke test (``python -m syncmoss.main --test``).
"""
import os

import numpy as np
import pytest

from syncmoss.constants import number_of_baseline_parameters
from syncmoss.model_io import read_model, _save_model_to_file, load_model_from_path

pytestmark = pytest.mark.gui


def test_window_builds_with_expected_widgets(physics_app):
    w = physics_app
    # Parameters table has numro (50) rows incl. the baseline row.
    assert len(w.params_table.row_widgets) == 50
    assert w.results_table is not None


def test_lazy_attributes_initialised(physics_app):
    """The getattr/hasattr -> attribute refactor must initialise these up-front."""
    w = physics_app
    assert w.last_plot_data is None
    assert w.last_fitting_data is None
    assert w._main_splitter is not None
    assert w._left_panel_ideal_width is not None
    assert w.toggle_dat_ins_action is not None
    assert isinstance(w.gridcolor, str) and w.gridcolor


def test_results_table_attributes_initialised(physics_app):
    rt = physics_app.results_table
    assert rt.parameter_names == []
    assert rt.covariance_matrix is None
    assert rt.errors is None
    assert rt.fit_parameters is None
    # Guard converted from hasattr -> attribute: must early-return cleanly.
    intensities, errors = rt._calculate_intensities()
    assert intensities.size == 0
    assert errors.size == 0


def test_read_model_empty_state_returns_only_baseline(physics_app):
    model, p, *_ = read_model(physics_app)
    assert model == []
    assert len(p) == number_of_baseline_parameters  # 8


def test_select_model_then_read_model(physics_app):
    w = physics_app
    w.params_table.select_model(1, "Sextet")

    # The model button text must reflect the selection.
    row_widget = w.params_table.row_widgets[1]
    model_btn = row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
    assert model_btn.text() == "Sextet"

    model, p, *_ = read_model(w)
    assert model == ["Sextet"]
    # baseline (8) + Sextet (11) parameters.
    assert len(p) == number_of_baseline_parameters + 11


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
    assert len(p) == number_of_baseline_parameters + 11 + 7


def test_resize_event_does_not_crash(physics_app):
    # resizeEvent reads _main_splitter / _left_panel_ideal_width (now plain attrs).
    physics_app.resize(1700, 950)
    physics_app.resize(1300, 720)


def test_toggle_use_dat_instrumental_action_text(physics_app):
    w = physics_app
    # Exercises the converted `if self.toggle_dat_ins_action is None` guard.
    before = w.use_dat_instrumental_metadata
    w.toggle_use_dat_instrumental()
    assert w.use_dat_instrumental_metadata is (not before)
    assert w.toggle_dat_ins_action.text()  # text was updated, not crashed


def test_show_spectrum_calibration_dat(physics_app):
    """Showing the bundled Calibration.dat must not raise or log an error."""
    w = physics_app
    cal_path = w.calibration_path
    if not os.path.exists(cal_path):
        pytest.skip(f"Calibration.dat not found at {cal_path}")

    w.process_path.setPlainText(repr([cal_path]))
    w.show_pressed()  # synchronous

    # The log turns red on error; success leaves it non-red.
    assert "red" not in w.log.styleSheet().lower(), w.log.toPlainText()
    assert w.inprogress is False
