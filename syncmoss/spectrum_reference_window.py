"""The window of a 'Spectrum' row: the reference (.dat) fitted with lines.

Choosing "Spectrum" in a row's model menu asks for the reference (.dat only)
and fits it (spectrum_reference) in a calculation thread, with the .dat's own
instrumental function -- for a .dat without one the user says whether it is SMS
or CMS, and the program's is used. The window says first what the tool is for
(DISCLAIMER), then shows how the reference will be imported:
the reference with the fit, its baseline and every line, the residuals, the
fitted Nnr/Ns of a CMS reference, and the Fit quality -- χ², S (no model) and
MISFIT, marked as spectrum_reference.quality_marks says. "Number of lines"
starts at the number found; Refit fits with the number set; OK writes the row
(T = Σ T_q, δ = G = 0, the line set); Cancel leaves the table as it was.
"""
import os
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSpinBox, QVBoxLayout)
from matplotlib.figure import Figure

import syncmoss.models as m5
from syncmoss.constants import NAT_WIDTH, SMS_POL_DEFAULT, model_colors, number_of_baseline_parameters as NB
from syncmoss.fit_quality import CHI2, MISFIT, STRUCTURE_FREE
from syncmoss.instrumental_io import drive_values, resolve_instrumental_for_file, source_for
from syncmoss.models_positions import mod_pos
from syncmoss.result_window import _Canvas
from syncmoss.spectrum_io import load_spectrum
from syncmoss.spectrum_model import format_spectrum_lines
from syncmoss.spectrum_plotter import plot_fitting_result
from syncmoss.spectrum_reference import LinesModel, ReferenceFitter, fitted_points, quality_marks

# The Fit quality rows the window shows
_SHOWN = (CHI2, STRUCTURE_FREE, MISFIT)

# The tool's colour: its entry in the model menu and its window's disclaimer
SPECTRUM_TOOL_COLOR = '#b5651d'

DISCLAIMER = ("It is a good tool to compare spectra or to do a preliminary fit. It is not recommended for a "
              "final result: this method has no information about polarization, it estimates the thickness "
              "inaccurately, and if the content (amount) is significantly different it can give a wrong "
              "result. Please use it knowingly and at your own discretion.")


def choose_reference(main_window, row):
    """The model menu's "Spectrum" on table *row*: the .dat, then the window."""
    path, _ = QFileDialog.getOpenFileName(main_window, "Reference spectrum", main_window.workfolder,
                                          "Spectra (*.dat)")
    if not path:
        return
    if not path.lower().endswith('.dat'):
        main_window.set_status("A reference must be a .dat file.", "orange")
        return
    if main_window._reject_if_busy("The reference fit"):
        return
    # JN, G, the SMS polarization and the global instrumental function, as
    # Fit and Show model read them
    if not main_window.initialize_parameters():
        return
    method = reference_method(main_window, path)
    if method is None:
        return
    try:
        job = prepare_reference(main_window, path, method)
    except Exception as e:
        main_window.set_status(f"The reference could not be read: {e}", "red")
        return
    window = SpectrumReferenceWindow(main_window, row, job)
    main_window.spectrum_reference_window = window
    window.show()
    window.start(None)


def reference_method(main_window, path):
    """The reference's method and instrumental function: its .dat header's,
    whatever the setting; for a .dat without one, the user says whether it is
    SMS or CMS and the program's own is used. None: cancelled."""
    method = resolve_instrumental_for_file(main_window, path, use_dat_metadata=True)
    if method['source'] == 'file':
        return method
    box = QMessageBox(main_window)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Reference spectrum")
    box.setText(f"{os.path.basename(path)} has no instrumental function. Is it SMS or CMS?\n"
                f"The program's instrumental function of that method is used.")
    sms = box.addButton("SMS", QMessageBox.ButtonRole.AcceptRole)
    cms = box.addButton("CMS", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    answer = {sms: 'SMS', cms: 'CMS'}.get(box.clickedButton())
    if answer is None:
        return None
    return resolve_instrumental_for_file(main_window, path, use_dat_metadata=True, force_method=answer)


def prepare_reference(main_window, path, method):
    """What the fit of the reference *path* needs, read on the main thread:
    its fitted points, and its *method* (reference_method) with the drive, as
    a fit of that spectrum has them (fitting_io)."""
    A, B = load_spectrum(main_window, path, calibration_path=main_window.calibration_path)
    A, B = np.asarray(A[0], dtype=float), np.asarray(B[0], dtype=float)
    mp = dict(method)
    mp.update(drive_values(main_window, A, os.path.basename(path)))
    x, y = fitted_points(A, B)
    return {'path': path, 'x': x, 'y': y, 'mp': mp, 'JN': int(main_window.JN0),
            'pol': float(getattr(main_window, 'SMS_pol', SMS_POL_DEFAULT))}


def lines_model(main_window, job):
    """The LinesModel of *job*: its source prepared for the fitted points (on
    the window's pool, the one Interrupt leaves), its node sum."""
    mp, JN, x = job['mp'], job['JN'], job['x']
    norm, smearing = source_for(main_window.pool, x, JN, mp)
    return LinesModel(x, JN, mp, norm, smearing, job['pol'])


class ReferenceFitThread(QThread):
    """The fit of the reference: automatic (*number* None) or with *number*
    lines from *start* (a LineFit). Its signal is 'done', not 'finished' (that
    would shadow QThread.finished)."""
    done = Signal(object)       # the result dict (see _result)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, main_window, job, fitter, number=None, start=None):
        super().__init__()
        self.main_window = main_window
        self.job = job
        self.fitter = fitter
        self.number = number
        self.start_fit = start

    def run(self):
        try:
            if self.fitter is None:
                self.fitter = ReferenceFitter(self.job['x'], self.job['y'], lines_model(self.main_window, self.job),
                                              self.job['mp']['method'])
            self.fitter.report = self.progress.emit         # this thread's, a refit's own
            fit = (self.fitter.automatic() if self.number is None
                   else self.fitter.with_number(self.start_fit, self.number))
            self.done.emit(_result(self.fitter, self.job, fit))
        except m5.FitInterrupted as e:
            self.error.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{e}")


def _result(fitter, job, fit):
    """A fit with what the window draws: every line on the baseline alone, its
    marker, and the Fit quality."""
    components, positions = [], []
    for line in fit.lines:
        components.append(fitter.model.spectrum(fit.p, fitter.model.transmission(line)))
        p = np.concatenate([fit.p[:NB], [line[0], line[1], NAT_WIDTH, line[2]]])
        positions.append(mod_pos(p, ['Singlet'], job['mp']['INS'], Met=job['mp']['Met']))
    return {'fit': fit, 'components': components, 'positions': positions, 'quality': fitter.quality(fit)}


class SpectrumReferenceWindow(QDialog):
    """How the reference will be imported (the module docstring)."""

    def __init__(self, main_window, row, job):
        super().__init__(main_window)
        self.main_window = main_window
        self.row = row
        self.job = job
        self.fitter = None
        self.result = None
        self.thread = None
        self.setWindowTitle(f"Spectrum: {os.path.basename(job['path'])} ({job['mp']['method']})")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1000, 750)

        layout = QVBoxLayout(self)
        self.disclaimer = QLabel(DISCLAIMER, self)
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setStyleSheet(f"color: white; background-color: {SPECTRUM_TOOL_COLOR}; padding: 6px;")
        layout.addWidget(self.disclaimer)
        self.figure = Figure()
        self.canvas = _Canvas(self.figure)
        layout.addWidget(self.canvas, 1)
        self.warning = QLabel("", self)
        self.warning.setStyleSheet("color: orange;")
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.quality = QLabel("", self)
        self.quality.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.quality)
        self.status = QLabel("", self)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Number of lines", self))
        self.number = QSpinBox(self)
        self.number.setRange(1, 200)
        buttons.addWidget(self.number)
        self.refit_button = QPushButton("Refit", self)
        self.refit_button.setToolTip("Fit with the number of lines set")
        self.refit_button.clicked.connect(self.refit)
        buttons.addWidget(self.refit_button)
        buttons.addStretch(1)
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept_reference)
        buttons.addWidget(self.ok_button)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self._set_running(False)

    # --- the fit ------------------------------------------------------------------

    def _set_running(self, running):
        ready = not running and self.result is not None
        for widget in (self.refit_button, self.ok_button, self.number):
            widget.setEnabled(ready)
        self.cancel_button.setText("Interrupt" if running else "Cancel")

    def start(self, number):
        """Fit the reference: automatically (*number* None) or with *number* lines."""
        main = self.main_window
        main.inprogress = True
        main.busy_with = 'The reference fit'
        self.status.setText("Fitting the reference..." if number is None else f"Refitting with {number} lines...")
        self.thread = ReferenceFitThread(main, self.job, self.fitter, number,
                                         None if self.result is None else self.result['fit'])
        self.thread.progress.connect(self.status.setText)
        self.thread.done.connect(self._done)
        self.thread.error.connect(self._failed)
        main.reference_fit_thread = self.thread
        self._set_running(True)
        main._start_calculation(self.thread)

    def refit(self):
        if self.main_window._reject_if_busy("Refit"):
            return
        self.start(self.number.value())

    def _done(self, result):
        self.fitter = self.thread.fitter
        self.result = result
        self.number.setValue(len(result['fit'].lines))
        self.status.setText(f"{len(result['fit'].lines)} lines, {self.fitter.fits} fits")
        self.main_window.set_status(f"The reference is fitted with {len(result['fit'].lines)} lines.", "green")
        self.main_window._calculation_ended()
        self._set_running(False)
        self.draw()

    def _failed(self, message):
        self.fitter = self.thread.fitter
        self.main_window.set_status(f"The reference fit stopped: {message}", "orange")
        self.main_window._calculation_ended()
        self._set_running(False)
        if self.result is None:
            self.close()
            return
        self.status.setText(f"Stopped: {message}")

    # --- what it shows ---------------------------------------------------------------

    def draw(self):
        fit = self.result['fit']
        n = len(fit.lines)
        main = self.main_window
        colors = ['silver'] + [model_colors[q % len(model_colors)] for q in range(n)]
        plot_fitting_result(self.figure, self.fitter.x, self.fitter.y, fit.model, self.result['components'],
                            self.result['positions'], fit.p, colors, fit.chi2, self.job['path'],
                            main.dir_path, gridcolor=main.gridcolor, theme=getattr(main, '_theme', None),
                            model=['Singlet'] * n, save=False)
        self.canvas.draw()
        self.warning.setText("The absorption reaches an end of the spectrum: its edges are not baseline."
                             if self.fitter.edges_absorb else "")
        self.quality.setText(self._quality_text(fit))

    def _quality_text(self, fit):
        parts = [f"{len(fit.lines)} lines, T<sub>ref</sub> = {fit.lines[:, 0].sum():.3f}"]
        if self.fitter.cms:
            parts.append(f"Nnr/Ns = {fit.fraction:.3f}")
        quality = self.result['quality']
        marked = quality_marks(quality) or []
        for name, ideal, value, sigma, _F, _sF in ([] if quality is None else quality['rows']):
            if name not in _SHOWN:
                continue
            text = f"{name} = {value:.3f} ± {sigma:.3f}"
            if name in marked:
                text = f"<span style='color: red;'>{text}</span>"
            parts.append(text)
        return ", ".join(parts)

    # --- the answer --------------------------------------------------------------------

    def accept_reference(self):
        """Write the row: T the reference's whole thickness, δ and G 0, the line set."""
        lines = self.result['fit'].lines
        table = self.main_window.params_table
        table.select_model(self.row, 'Spectrum')
        table._value_input(self.row, 0).setText(repr(float(lines[:, 0].sum())))
        table.expression_value_input(self.row).setText(format_spectrum_lines(lines))
        self.main_window.set_status(f"Spectrum row {self.row}: {os.path.basename(self.job['path'])}, "
                                    f"{len(lines)} lines.", "green")
        self.accept()

    def reject(self):
        """Cancel: while fitting, Interrupt (the window stays until the fit has
        stopped); else close, the table untouched."""
        if self.thread is not None and self.thread.isRunning():
            self.main_window.interrupt()
            return
        super().reject()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.main_window.interrupt()
            event.ignore()
            return
        super().closeEvent(event)
