"""Exclusion regions in the main window: the bar, the bands on the plot on
screen, Pick on plot, what a change does to a fit result, and what Save result,
Save/Load and "Save spectrum without excluded points" write.

A fit here is the real fit with the minimiser replaced by one that returns the
start vector (no time spent fitting).
"""
import os
import shutil

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from syncmoss import exclusion_regions as er
from syncmoss import fitting_io, spectrum_io
from syncmoss.instrumental_io import build_dat_metadata_lines, resolve_instrumental_for_file
from syncmoss.model_io import model_file_rows
from syncmoss.spectrum_io import load_spectrum, save_spectrum_with_metadata
from syncmoss.spectrum_plotter import (
    EXCLUSION_GID, CHI2_GID, SPECTRUM_AXES_GID, plot_model, spectrum_axes_of,
)

from conftest import FROZEN_PARAMETERS

pytestmark = [pytest.mark.gui]

REGIONS = "-3:-2; 1:2"


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _fake_minimiser(monkeypatch):
    def fake(func, x, y, p0, **kwargs):
        p0 = np.array(p0, dtype=float)
        fixed = {int(i) for i in np.ravel(kwargs['fix'])}
        free = [i for i in range(len(p0)) if i not in fixed]
        errors = np.full(len(p0), np.nan)
        errors[free] = 0.01
        return p0, errors, 1.0, np.eye(len(free)) * 1e-4
    monkeypatch.setattr(fitting_io.mi, 'minimi_hi', fake)


def _spectra(app, tmp_path, n=1, metadata=()):
    """*n* copies of the frozen calibration spectrum as .dat files, in the path box."""
    A, B = np.loadtxt(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), comments='#', unpack=True)
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        save_spectrum_with_metadata(path, A, B, list(metadata))
        paths.append(path)
    app.process_path.setPlainText(repr(paths))
    app.dir_path = str(tmp_path)            # result.png/.svg go here, not into the package
    app.params_table.select_model(1, 'Sextet')
    app.initialize_parameters()
    return paths


def _apply(app, text):
    if not app.toolbar.exclusion_action.isChecked():
        app.toolbar.exclusion_action.trigger()
    app.exclusion_bar.text.setText(text)
    assert app.exclusion_bar.commit()
    return app.exclusion_regions_in_use()


def _fit(app, path, monkeypatch):
    """A fit shown as the window shows one."""
    _fake_minimiser(monkeypatch)
    app._fit_links_snapshot = app.params_table.get_link_snapshot()
    app._fit_model_snapshot = model_file_rows(app)
    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())
    assert result['success'], result.get('message')
    app.on_fitting_finished(result)
    return result


def _bands(app):
    return [[(p.get_x(), p.get_x() + p.get_width()) for p in ax.patches if p.get_gid() == EXCLUSION_GID]
            for ax in spectrum_axes_of(app.figure)]


def _chi2_title(app):
    return [ax.title.get_text() for ax in app.figure.axes if ax.title.get_gid() == CHI2_GID]


def _click(app, x, ax=None):
    ax = ax if ax is not None else spectrum_axes_of(app.figure)[0]
    app.canvas.draw()
    px, py = ax.transData.transform((x, sum(ax.get_ylim()) / 2))
    app.canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', app.canvas, px, py, button=1))


# --- the bar ---------------------------------------------------------------------

def test_apply_exclusion_shows_the_bar_and_puts_the_regions_in_use(physics_app):
    app = physics_app
    app.exclusion_bar.text.setText(REGIONS)
    assert app.exclusion_bar.commit()
    assert app.exclusion_bar.isHidden() and app.exclusion_regions_in_use() == ()

    app.toolbar.exclusion_action.trigger()
    assert not app.exclusion_bar.isHidden()
    assert app.exclusion_regions_in_use() == ((-3.0, -2.0), (1.0, 2.0))

    app.toolbar.exclusion_action.trigger()
    assert app.exclusion_bar.isHidden() and app.exclusion_regions_in_use() == ()
    assert app.exclusion_bar.text.text() == "-3:-2; 1:2"        # kept for the next time


def test_the_bar_takes_only_its_own_two_lines(physics_app):
    """The label and the buttons on one line, the regions box under them at full
    width -- and no more height than that: the plot takes the rest."""
    app = physics_app
    app.resize(1600, 1000)
    app.show()
    app.toolbar.exclusion_action.trigger()
    QApplication.processEvents()
    bar = app.exclusion_bar
    assert bar.height() == bar.sizeHint().height()
    assert bar.text.y() >= bar.pick_btn.y() + bar.pick_btn.height()
    assert bar.text.width() == bar.width()


def test_the_text_is_cleaned_and_bad_text_is_refused(physics_app):
    app = physics_app
    bar = app.exclusion_bar
    _apply(app, "2:1; -2:-3; 1.5:3")
    assert bar.text.text() == "-3:-2; 1:3"
    bar.text.setText("1,5:2")
    assert not bar.commit()
    assert bar.regions == ((-3.0, -2.0), (1.0, 3.0))
    assert "red" in bar.text.styleSheet() and "';'" in app.log.toPlainText()


def test_enter_in_the_box_applies_the_regions_and_not_show_model(physics_app):
    app = physics_app
    _apply(app, "")
    override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(app.exclusion_bar.text, override)
    assert override.isAccepted()                # the window's Return shortcut does not fire
    app.exclusion_bar.text.setText(REGIONS)
    app.exclusion_bar.text.editingFinished.emit()
    assert app.exclusion_regions_in_use() == ((-3.0, -2.0), (1.0, 2.0))


# --- the plot on screen ----------------------------------------------------------

def test_bands_are_drawn_only_while_applied_and_never_widen_the_axis(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path)
    app.show_pressed()
    xlim = spectrum_axes_of(app.figure)[0].get_xlim()
    assert _bands(app) == [[]]

    _apply(app, "-100:-8; 1:2")
    assert _bands(app) == [[(-100.0, -8.0), (1.0, 2.0)]]
    assert spectrum_axes_of(app.figure)[0].get_xlim() == xlim

    app.toolbar.exclusion_action.trigger()
    assert _bands(app) == [[]]


def test_changing_the_regions_drops_a_fit_result_but_keeps_its_curves(physics_app, tmp_path, monkeypatch):
    app = physics_app
    path, = _spectra(app, tmp_path)
    _apply(app, REGIONS)
    _fit(app, path, monkeypatch)
    ax = spectrum_axes_of(app.figure)[0]
    assert _chi2_title(app)[0].startswith('χ²') and _bands(app) == [[(-3.0, -2.0), (1.0, 2.0)]]

    _apply(app, "4:5")

    assert spectrum_axes_of(app.figure)[0] is ax                 # the same image, not redrawn
    assert _chi2_title(app) == ['']
    assert any(line.get_label() == 'Fit' for line in ax.lines)   # its curves stay
    assert _bands(app) == [[(4.0, 5.0)]]
    assert app.results_table.current_parameters is None
    app.take_result()
    assert app.log.toPlainText() == "No fitting results available"
    app.save_result_pressed()
    assert app.log.toPlainText().startswith("No results to save")


def test_clean_removes_every_region_and_drops_the_fit_result(physics_app, tmp_path, monkeypatch):
    app = physics_app
    path, = _spectra(app, tmp_path)
    _apply(app, REGIONS)
    _fit(app, path, monkeypatch)
    app.exclusion_bar.pick_btn.setChecked(True)

    app.exclusion_bar.clean_btn.click()

    assert app.exclusion_bar.text.text() == ""
    assert app.exclusion_regions_in_use() == ()
    assert app.toolbar.exclusion_action.isChecked()          # Apply exclusion stays on
    assert not app.exclusion_bar.pick_btn.isChecked()
    assert _bands(app) == [[]] and _chi2_title(app) == ['']
    assert app.results_table.current_parameters is None


def test_clean_is_refused_while_a_calculation_runs(physics_app):
    app = physics_app
    _apply(app, REGIONS)
    app.inprogress, app.busy_with = True, 'Fitting'
    try:
        app.exclusion_bar.clean_btn.click()
        assert app.exclusion_bar.text.text() == "-3:-2; 1:2"
        assert app.exclusion_regions_in_use() == ((-3.0, -2.0), (1.0, 2.0))
    finally:
        app.inprogress = False


def test_changing_the_regions_on_a_show_model_image_keeps_the_model(physics_app, tmp_path, monkeypatch):
    app = physics_app
    path, = _spectra(app, tmp_path)
    result = _fit(app, path, monkeypatch)
    plot_model(app.figure, result['A'], result['B'], result['SPC_f'], result['FS'], result['FS_pos'],
               result['parameters'], app.params_table.get_current_colors())
    app.results_table.current_parameters = None               # as after Show model
    app.last_plot_data = {'is_show_model': True}
    lines = len(spectrum_axes_of(app.figure)[0].lines)

    _apply(app, REGIONS)

    assert len(spectrum_axes_of(app.figure)[0].lines) == lines
    assert _bands(app) == [[(-3.0, -2.0), (1.0, 2.0)]]
    assert app.last_plot_data['exclusion_regions'] == ((-3.0, -2.0), (1.0, 2.0))


# --- Pick on plot ----------------------------------------------------------------

def _data_points(app):
    ax = spectrum_axes_of(app.figure)[0]
    return np.sort(np.concatenate([line.get_xdata() for line in ax.lines if line.get_gid() == 'spectrum data']))


@pytest.mark.parametrize("image", ["spectrum", "fit"])
def test_two_clicks_in_either_order_add_a_region_between_points(physics_app, tmp_path, monkeypatch, image):
    app = physics_app
    path, = _spectra(app, tmp_path)
    _apply(app, "-3:-2")
    if image == "fit":
        _fit(app, path, monkeypatch)
    else:
        app.show_pressed()
    v = _data_points(app)

    app.exclusion_bar.pick_btn.setChecked(True)
    _click(app, 2.013)
    _click(app, 0.987)

    assert not app.exclusion_bar.pick_btn.isChecked()
    (lo0, hi0), (lo, hi) = app.exclusion_regions_in_use()
    assert (lo0, hi0) == (-3.0, -2.0)
    for end, click in ((lo, 0.987), (hi, 2.013)):
        i = np.searchsorted(v, click)
        assert v[i - 1] < end < v[i]                    # between the points around the click
    assert app.exclusion_bar.text.text() == er.format_regions(app.exclusion_regions_in_use())


def test_clicks_in_pan_mode_or_off_a_spectrum_are_ignored_and_esc_cancels(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path)
    _apply(app, "")
    app.show_pressed()
    bar = app.exclusion_bar
    bar.pick_btn.setChecked(True)

    app.toolbar.pan()
    _click(app, 1.0)
    app.toolbar.pan()
    assert bar._first is None

    other = app.figure.add_axes([0.0, 0.0, 0.1, 0.1])         # not a spectrum plot
    _click(app, 0.5, ax=other)
    assert bar._first is None
    other.remove()

    _click(app, 1.0)
    assert bar._first is not None and bar._marks
    bar.escape.activated.emit()
    assert not bar.pick_btn.isChecked() and bar._first is None and not bar._marks
    assert app.exclusion_regions_in_use() == ()


# --- saving and loading ----------------------------------------------------------

def test_save_result_writes_the_regions_only_when_applied(physics_app, tmp_path, monkeypatch):
    app = physics_app
    path, = _spectra(app, tmp_path)
    regions = _apply(app, REGIONS)
    result = _fit(app, path, monkeypatch)
    app.save_path.setText(str(tmp_path / "out" / "with"))
    app._save_result_files('new')

    names, row = [line.rstrip('\n').split('\t')
                  for line in open(tmp_path / "out" / "with_param.txt", encoding='utf-8')]
    assert names[-2:] == ['χ²', 'Exclusion regions'] and row[-1] == "-3:-2; 1:2"
    graf = np.genfromtxt(tmp_path / "out" / "with_graf.txt", names=True, deletechars='')
    assert graf.dtype.names[-1] == 'Fitted'
    assert np.array_equal(graf['Fitted'] == 1, er.kept(result['A'], regions))
    assert er.read_file(str(tmp_path / "out" / "with_exclusion.txt")) == regions

    app.toolbar.exclusion_action.trigger()                  # off
    _fit(app, path, monkeypatch)
    app.save_path.setText(str(tmp_path / "out" / "without"))
    app._save_result_files('new')
    names = open(tmp_path / "out" / "without_param.txt", encoding='utf-8').readline().rstrip('\n').split('\t')
    assert names[-1] == 'χ²'
    assert 'Fitted' not in open(tmp_path / "out" / "without_graf.txt", encoding='utf-8').readline()
    assert not os.path.exists(tmp_path / "out" / "without_exclusion.txt")


def test_save_and_load_and_load_replaces(physics_app, tmp_path, monkeypatch):
    app = physics_app
    bar = app.exclusion_bar
    _apply(app, REGIONS)
    target = str(tmp_path / "mine_exclusion.txt")
    monkeypatch.setattr("syncmoss.exclusion_bar.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (target, ''))
    bar.save()
    assert er.read_file(target) == ((-3.0, -2.0), (1.0, 2.0))

    other = tmp_path / "other_exclusion.txt"
    other.write_text("# hand made\n5:6\n", encoding='utf-8')
    assert bar.load_file(str(other))
    assert app.exclusion_regions_in_use() == ((5.0, 6.0),)
    assert bar.text.text() == "5:6"

    bad = tmp_path / "bad.txt"
    bad.write_text("5,5:6\n", encoding='utf-8')
    assert not bar.load_file(str(bad))
    assert app.exclusion_regions_in_use() == ((5.0, 6.0),)


def test_the_regions_do_not_change_while_a_calculation_runs(physics_app):
    app = physics_app
    _apply(app, REGIONS)
    app.inprogress, app.busy_with = True, 'Fitting'
    try:
        app.exclusion_bar.text.setText("4:5")
        assert not app.exclusion_bar.commit()
        assert app.exclusion_bar.text.text() == "-3:-2; 1:2"
        app.toolbar.exclusion_action.trigger()
        assert app.toolbar.exclusion_action.isChecked()
        app.exclusion_bar.pick_btn.setChecked(True)
        assert not app.exclusion_bar.pick_btn.isChecked()
        assert app.exclusion_regions_in_use() == ((-3.0, -2.0), (1.0, 2.0))
    finally:
        app.inprogress = False


def test_every_selected_spectrum_is_saved_without_its_excluded_points(physics_app, tmp_path, monkeypatch):
    app = physics_app
    paths = _spectra(app, tmp_path, 2, metadata=["#@GCMS 0.1"])
    regions = _apply(app, REGIONS)
    app.save_path.setText(str(tmp_path / "cut"))
    monkeypatch.setattr(spectrum_io.QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    spectrum_io.save_without_excluded_points(app)

    for path in paths:
        A, B = load_spectrum(app, [path], calibration_path=app.calibration_path)
        keep = er.kept(A[0], regions)
        saved = tmp_path / "cut" / f"excl_{os.path.basename(path)}"
        lines = saved.read_text().splitlines()
        assert "#@GCMS 0.1" in lines
        assert "# Excluded regions, mm/s: -3:-2; 1:2" in lines
        A_saved, B_saved = load_spectrum(app, [str(saved)], calibration_path=app.calibration_path)
        assert np.array_equal(A_saved[0], A[0][keep]) and np.array_equal(B_saved[0], B[0][keep])


@pytest.mark.parametrize("method", ["SMS", "CMS"])
def test_raw_counts_are_saved_with_the_instrumental_function_in_use(physics_app, tmp_path, method):
    """An .mca is calibrated when it is read: the .dat written from it gets the
    instrumental function in use, as the RAW -> .dat conversion writes it."""
    app = physics_app
    (app.MS_fit if method == "CMS" else app.SMS_fit).setChecked(True)
    mca = str(tmp_path / "Fe.mca")
    shutil.copy2(os.path.join(os.path.dirname(__file__), "data", "alpha_fe_sms_000.mca"), mca)
    regions = er.parse(REGIONS)
    saved = str(tmp_path / "excl_Fe.dat")

    written = spectrum_io.process_without_excluded(app, mca, saved, regions)

    lines = open(saved, encoding='utf-8').read().splitlines()
    expected, in_use = build_dat_metadata_lines(app)
    assert in_use == method and all(line in lines for line in expected)
    used = resolve_instrumental_for_file(app, saved)
    assert used['source'] == 'file' and used['method'] == method
    A, B = load_spectrum(app, [mca], calibration_path=app.calibration_path)
    keep = er.kept(A[0], regions)
    A_saved, B_saved = load_spectrum(app, [saved], calibration_path=app.calibration_path)
    assert written == keep.sum()
    assert np.array_equal(B_saved[0], B[0][keep])
    assert np.allclose(A_saved[0], A[0][keep], rtol=0, atol=5e-7)     # written with 6 decimals


def test_saving_without_excluded_points_needs_applied_regions(physics_app, tmp_path):
    app = physics_app
    _spectra(app, tmp_path)
    spectrum_io.save_without_excluded_points(app)
    assert "apply exclusion regions first" in app.log.toPlainText()
    assert not os.path.exists(tmp_path / "excl_spectrum_1.dat")
