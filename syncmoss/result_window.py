"""The result of a fit of several spectra, one spectrum at a time.

Opened at the end of a simultaneous one-model fit, of a sequence fit and of a
simultaneous fit of a model built with Nbaseline rows. The slider at the bottom
picks the spectrum: it is drawn the way a single fit is drawn (data, fit,
components, residual, x4 integration check), and below it the spectrum's part of
the result -- its own N, N1, N2, ... in a line of their own and its parameters
in a results table.

What is shown comes from a "series": one_model.OneModelResult, SequenceSeries or
NbaselineSeries. Every one of them answers the same questions per spectrum k
(curves, values, errors, model, ...) and says whether a correlation matrix of
one spectrum means anything: it does for a sequence, where every spectrum has
been fitted on its own; it does not for a simultaneous fit -- one spectrum's
block of the joint covariance is no correlation matrix -- so that tab is hidden.

Nothing is saved or pre-drawn: every spectrum is drawn from the result when the
slider reaches it. One window is reused for every later fit, and it keeps the
slider where it was. It carries its own figure, canvas and toolbar, so zooming
here never touches the main plot.
"""
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from syncmoss.one_model import parameters_line
from syncmoss.results_table import ResultsTable
from syncmoss.spectrum_parameters import SpectrumParameters
from syncmoss.spectrum_plotter import calculate_z_order, plot_fitting_result

# Components drawing no curve of their own (as in the main window's replot)
_NOT_DRAWN = {'Nbaseline', 'Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables'}


def _texts_with_weights(texts, model_list, weights):
    """*texts* with every Recon cell of *model_list* holding its fitted weights."""
    texts = dict(texts)
    weights = iter(weights)
    for component, name in enumerate(model_list):
        if name == 'Recon':
            w = next(weights, None)
            if w is not None:
                texts[component] = ','.join(f'{v:.6g}' for v in np.ravel(w))
    return texts


def _errors(result, n):
    """A fit's errors, or NaN for all when the minimiser could not move (it
    gives a single 0 then)."""
    errors = np.asarray(result['errors'], dtype=float).ravel()
    return errors if len(errors) == n else np.full(n, np.nan)


class SequenceSeries:
    """The spectra of a sequence fit: every one fitted on its own, with its own
    chi2 and its own -- meaningful -- correlation matrix.

    Made when the sequence starts, from the parameters table (the model every
    spectrum is fitted with); add() takes each spectrum's result as it comes.
    A spectrum whose fit failed is not in it.
    """

    shows_correlations = True

    def __init__(self, model_list, colors, names, texts, spectra):
        self.model_list = list(model_list)
        self.colors = list(colors)
        self.names = [list(n) for n in names]
        self.texts = dict(texts)
        self.spectra = list(spectra)         # SpectrumParameters of the whole run
        self.items = []                      # (SpectrumParameters, result) of the fitted

    def add(self, spectrum_file, result):
        taken = {id(spectrum) for spectrum, _ in self.items}
        spectrum = next((s for s in self.spectra
                         if s.path == spectrum_file and id(s) not in taken),
                        SpectrumParameters(len(self.items) + 1, (), spectrum_file))
        self.items.append((spectrum, result))

    def count(self):
        return len(self.items)

    def title(self):
        return f"sequence fit: {self.count()} of {len(self.spectra)} spectra fitted"

    def label(self, k):
        spectrum = self.items[k][0]
        return f"{spectrum.number} of {len(self.spectra)} · {os.path.basename(spectrum.path)}"

    def parameters_line(self, k):
        return parameters_line(self.items[k][0])

    def path_of(self, k):
        return self.items[k][1].get('spectrum_file') or self.items[k][0].path

    def spectrum_parameters_of(self, k):
        return self.items[k][0]

    def curves(self, k):
        r = self.items[k][1]
        return r['A'], r['B'], r['SPC_f'], r['FS'], r['FS_pos'], r.get('hires_diff')

    def exclusion_regions_of(self, k):
        """The exclusion regions spectrum k's fit left out."""
        return tuple(self.items[k][1].get('exclusion_regions') or ())

    def values(self, k):
        return np.asarray(self.items[k][1]['parameters'], dtype=float)

    def errors_of(self, k):
        return _errors(self.items[k][1], len(self.values(k)))

    def covariance_of(self, k):
        return np.atleast_2d(np.asarray(self.items[k][1]['covariance_matrix'], dtype=float))

    def fix_of(self, k):
        return np.flatnonzero(np.isnan(self.errors_of(k)))

    def model_of(self, k):
        return list(self.items[k][1].get('model', self.model_list[1:]))

    def model_list_of(self, k):
        return list(self.model_list)

    def colors_of(self, k):
        return list(self.colors)

    def names_of(self, k):
        return [list(n) for n in self.names]

    def texts_of(self, k):
        return _texts_with_weights(self.texts, self.model_list, self.items[k][1].get('Recon', []))

    def whole_fit(self):
        """Every spectrum's fit is a whole fit of its own."""
        return None

    def chi2_of(self, k):
        return self.items[k][1]['chi2']

    def chi2_spread_of(self, k):
        return self.items[k][1].get('chi2_spread')


class NbaselineSeries:
    """The spectra of a simultaneous fit of a model built with Nbaseline rows:
    one section of the model per spectrum, each with its own components.

    *model_list*, *colors*, *names* and *texts* are the parameters table's (the
    results table's lists: baseline first, every Nbaseline row opening the next
    spectrum's section); *entries* the path box's (path, values).
    """

    shows_correlations = False

    def __init__(self, result, model_list, colors, names, texts, entries=()):
        self.result = result
        self.p = np.asarray(result['parameters'], dtype=float)
        self.errors = _errors(result, len(self.p))
        self.covariance = np.atleast_2d(np.asarray(result['covariance_matrix'], dtype=float))
        free = np.flatnonzero(~np.isnan(self.errors))
        self._covariance_row = {int(index): row for row, index in enumerate(free)}
        starts = list(result['begining_spc']) + [len(self.p)]
        self._slots = [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
        cuts = [i for i, name in enumerate(model_list) if name == 'Nbaseline'] + [len(model_list)]
        self._components = [(0 if k == 0 else cuts[k - 1], cuts[k]) for k in range(len(cuts))]
        self.model_list = list(model_list)
        self.colors = list(colors)
        self.names = [list(n) for n in names]
        self.texts = dict(texts)
        self.entries = list(entries)

    def count(self):
        return len(self._slots)

    def title(self):
        return f"simultaneous fit of {self.count()} spectra (Nbaseline)"

    def path_of(self, k):
        return self.result['spectrum_files'][k]

    def label(self, k):
        return f"{k + 1} of {self.count()} · {os.path.basename(self.path_of(k))}"

    def spectrum_parameters_of(self, k):
        values = self.entries[k][1] if k < len(self.entries) else ()
        return SpectrumParameters(k + 1, values, self.path_of(k))

    def parameters_line(self, k):
        return parameters_line(self.spectrum_parameters_of(k))

    def curves(self, k):
        r = self.result
        hires = r.get('hires_diff_list') or [None] * self.count()
        return (r['A_list'][k], r['B_list'][k], r['SPC_f_list'][k], r['FS_list'][k],
                r['FS_pos_list'][k], hires[k])

    def exclusion_regions_of(self, k):
        """The exclusion regions the fit left out (the same for every spectrum)."""
        return tuple(self.result.get('exclusion_regions') or ())

    def values(self, k):
        start, end = self._slots[k]
        return self.p[start:end].copy()

    def errors_of(self, k):
        start, end = self._slots[k]
        return self.errors[start:end].copy()

    def covariance_of(self, k):
        start, end = self._slots[k]
        rows = [self._covariance_row[i] for i in range(start, end) if i in self._covariance_row]
        if not rows or max(rows) >= self.covariance.shape[0]:
            return np.zeros((len(rows), len(rows)))
        return self.covariance[np.ix_(rows, rows)]

    def fix_of(self, k):
        return np.flatnonzero(np.isnan(self.errors_of(k)))

    def model_of(self, k):
        return list(self.result['model_separate'][k])

    def model_list_of(self, k):
        """The spectrum's components, its own baseline (the Nbaseline row) first."""
        return ['baseline'] + self.model_of(k)

    def colors_of(self, k):
        first, end = self._components[k]
        return self.colors[first:end]

    def names_of(self, k):
        first, end = self._components[k]
        return self.names[first:end]

    def texts_of(self, k):
        """The section's Expression/Distr/Corr texts, as typed, and Recon weights."""
        first, end = self._components[k]
        texts = {component - first: text for component, text in self.texts.items()
                 if first <= component < end
                 and self.model_list[component] in ('Expression', 'Distr', 'Corr')}
        recons_before = self.model_list[:first].count('Recon')
        weights = list(self.result.get('Recon', []))[recons_before:]
        return _texts_with_weights(texts, self.model_list_of(k), weights)

    def whole_fit(self):
        """The Expressions are typed with the whole model's p[i] and may use
        another spectrum's parameters: they are evaluated with all of them."""
        return self.p, self.errors, self.covariance

    def chi2_of(self, k):
        """The fit's one chi2: the spectra were fitted together."""
        return self.result['chi2']

    def chi2_spread_of(self, k):
        return self.result.get('chi2_spread')


class _Canvas(FigureCanvas):
    """matplotlib's canvas, its delayed redraw bound to the canvas itself.

    FigureCanvasQT.draw_idle posts the redraw with a bare QTimer.singleShot
    (resizing the window does it), so the redraw still fires after the window is
    gone and fails on the deleted widget. With the canvas as the timer's
    context, Qt drops it together with the canvas.
    """

    def draw_idle(self):
        if not (getattr(self, '_draw_pending', False) or getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            QTimer.singleShot(0, self, self._draw_idle)


class ResultWindow(QMainWindow):
    """The spectra of a fit of several spectra, one at a time."""

    def __init__(self, main_window):
        super().__init__(main_window)
        # Standalone top-level window (its own taskbar entry), not a child panel.
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.main_window = main_window
        self.result = None
        self.z_order = None
        self.resize(1200, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        plot_area = QWidget(self)
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure()
        self.canvas = _Canvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)

        table_area = QWidget(self)
        table_layout = QVBoxLayout(table_area)
        table_layout.setContentsMargins(0, 0, 0, 0)
        # The spectrum's own N, N1, N2, ... -- a line of its own, not editable
        self.parameters_line = QLineEdit(self)
        self.parameters_line.setReadOnly(True)
        table_layout.addWidget(self.parameters_line)
        self.table = ResultsTable(main_window)
        table_layout.addWidget(self.table, 1)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(plot_area)
        splitter.addWidget(table_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        slider_row = QHBoxLayout()
        self.spectrum_label = QLabel("", self)
        self.spectrum_label.setMinimumWidth(320)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setPageStep(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self._slider_moved)
        slider_row.addWidget(self.spectrum_label)
        slider_row.addWidget(self.slider, 1)
        root.addLayout(slider_row)

    def show_result(self, result, spectrum=None):
        """Show *result* (a series, see the module docstring) at *spectrum*
        (0-based; by default where the slider was, kept inside the new result),
        and raise the window."""
        self.result = result
        count = result.count()
        self.setWindowTitle(f"SYNCmoss - {result.title()}")
        # A correlation matrix of one spectrum: only where it means something
        self.table.tabs.setTabVisible(1, result.shows_correlations)
        if not result.shows_correlations:
            self.table.tabs.setCurrentIndex(0)
        if spectrum is None:
            spectrum = min(self.slider.value(), count) - 1
        self.slider.blockSignals(True)
        self.slider.setMaximum(count)
        self.slider.setValue(spectrum + 1)
        self.slider.blockSignals(False)
        self._slider_moved(spectrum + 1)
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def spectrum(self):
        """The spectrum shown (0-based)."""
        return self.slider.value() - 1

    def _slider_moved(self, value):
        if self.result is None:
            return
        self.z_order = None
        self.draw(value - 1)

    def draw(self, k):
        """Draw spectrum *k*: the plot, its N line and its results."""
        result = self.result
        self.spectrum_label.setText(result.label(k))
        self.parameters_line.setText(result.parameters_line(k))
        self._draw_plot(k)
        self.table.fill_table(
            result.values(k), result.model_list_of(k), result.colors_of(k),
            result.names_of(k), result.covariance_of(k), result.errors_of(k),
            result.fix_of(k), result.texts_of(k),
            spectrum_parameters=result.spectrum_parameters_of(k), whole_fit=result.whole_fit())

    def _draw_plot(self, k):
        result = self.result
        A, B, fit, components, positions, hires = result.curves(k)
        main = self.main_window
        if self.z_order is None:
            self.z_order = calculate_z_order(components)
        plot_fitting_result(
            self.figure, A, B, fit, components, positions, result.values(k),
            result.colors_of(k), result.chi2_of(k), result.path_of(k),
            main.dir_path, z_order=self.z_order, gridcolor=main.gridcolor,
            theme=main._theme, model=result.model_of(k), hires_diff=hires,
            chi2_spread=result.chi2_spread_of(k), save=False,
            exclusion_regions=result.exclusion_regions_of(k))
        self.canvas.draw()

    def replot_result(self, row_index):
        """A click on a component of the table below brings its curve to the top
        (as in the main window)."""
        if self.result is None or self.z_order is None:
            return
        model = self.result.model_list_of(self.spectrum())
        component = row_index // 3
        if component == 0 or component >= len(model) or model[component] in _NOT_DRAWN:
            return
        drawn = sum(1 for name in model[1:component] if name not in _NOT_DRAWN)
        if drawn >= len(self.z_order):
            return
        self.z_order[drawn] = max(self.z_order) + 1
        self._draw_plot(self.spectrum())
