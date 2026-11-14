"""Supp -> Find principal components... and Show found components...

Find: the spectra of the path box -- one velocity axis, one instrumental
function, as any fit reads them -- searched for their components
(components_search) in a calculation thread; the dialog asks for the results
folder, what every spectrum holds besides the sample (nothing, Be or KB_nano:
the current presets), a CMS spectrum's non-resonant counts (a coefficient, or
the source's form of the Δθ dialogs) and, for a map, its layout. The results
go to the folder (components_results) and the window opens on it.

Show: the window on a results folder, a tab for each view with its own
controls -- the components (through the instrumental function, or their
lines alone; combined into one image); the lines and which components hold
each; every component's T along a series with every spectrum's D (norm), or
the map of up to three images in red, green and blue (component T's, line
T's, D (norm), combinations), each channel the sum of the images ticked for
it, darkened where it is not significant; a clicked spectrum with its fit in
the main window's style; the summed spectrum of a selection -- by threshold on
an image, or painted on the map or dragged along the series with the brush --
with its fit, saved and loaded to fit by "Sum all spectra"'s own function; two
images against each other. The names say "principal components", the
familiar words; they are not those of a PCA.
"""
import os
import re
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QRadioButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import syncmoss.components_results as cr
import syncmoss.components_search as cs
import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.Calibration import _read_optics_preset, default_impurities
from syncmoss.constants import CMS_NNR_RATIO, NAT_WIDTH, SMS_POL_DEFAULT, model_colors
from syncmoss.exclusion_regions import fitted
from syncmoss.experimental_features import warn_if_experimental
from syncmoss.instrumental_dialogs import DTHETA_SOURCE_F_LM
from syncmoss.instrumental_io import (analyze_instrumental_methods, build_dat_metadata_lines, drive_values,
                                      read_dat_metadata_lines)
from syncmoss.model_io import strip_known_extension
from syncmoss.result_window import _Canvas
from syncmoss.spectrum_io import load_spectrum, save_spectrum_with_metadata, sum_spectra
from syncmoss.spectrum_plotter import _style_axis, _tc, plot_fitting_result

ACTION = "Find principal components"
# The known part's rows, by the name the model menu gives it, and the preset each reads
KNOWN_PARTS = {'nothing': None, 'Be': 'Be', 'KB_nano': 'KB'}
# A map's pixel without a spectrum (an incomplete scan)
MISSING_COLOR = (0.5, 0.5, 0.5)
# X_Y at the end of a file name: the map coordinates of its spectrum
_COORDINATES = re.compile(r'(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$')


# --- the layout of a map -------------------------------------------------------------

def coordinates_from_names(files):
    """(n, 2) X and Y of every file whose name (its extension off) ends in
    X_Y; None when one does not."""
    out = []
    for path in files:
        match = _COORDINATES.search(os.path.basename(strip_known_extension(path)))
        if match is None:
            return None
        out.append([float(match.group(1)), float(match.group(2))])
    return np.array(out)


def grid_coordinates(n, columns, snake):
    """X and Y of n spectra filling rows of *columns* one after the other:
    every row from the left (zig-zag), or every second one from the right
    (snake); an incomplete last row stays short."""
    rows, cols = np.arange(n) // columns, np.arange(n) % columns
    if snake:
        cols = np.where(rows % 2 == 1, columns - 1 - cols, cols)
    return np.column_stack([cols, rows]).astype(float)


def pixel_grid(coordinates):
    """The map of spectra at *coordinates*: (row and column of every
    spectrum, the grid of spectrum indices with −1 where none is)."""
    xs, ys = np.unique(coordinates[:, 0]), np.unique(coordinates[:, 1])
    cols, rows = np.searchsorted(xs, coordinates[:, 0]), np.searchsorted(ys, coordinates[:, 1])
    grid = np.full((len(ys), len(xs)), -1)
    grid[rows, cols] = np.arange(len(coordinates))
    return np.column_stack([rows, cols]), grid


# --- Find principal components... --------------------------------------------------------

def prepare_search(main_window, paths):
    """What the search of *paths* needs, read on the main thread: the
    spectra on their one velocity axis (ascending), the points every fit
    takes (exclusion regions and points without counts left out), the one
    instrumental function with its drive, the .dat header it is written with.
    ValueError says why the spectra cannot be searched."""
    A_list, B_list = load_spectrum(main_window, paths, calibration_path=main_window.calibration_path)
    if not A_list or any(len(a) != len(A_list[0]) for a in A_list):
        raise ValueError("the spectra must have the same number of points")
    order = np.argsort(np.asarray(A_list[0], dtype=float), kind='stable')
    A = np.asarray(A_list[0], dtype=float)[order]
    step = np.min(np.diff(A)) if len(A) > 1 else 0.0
    for path, a in zip(paths, A_list):
        if np.max(np.abs(np.sort(np.asarray(a, dtype=float)) - A)) > step / 2:
            raise ValueError(f"{os.path.basename(path)} has other velocities: the components need one axis")
    Y = np.array([np.asarray(b, dtype=float)[order] for b in B_list])
    use_dat = bool(getattr(main_window, 'use_dat_instrumental_metadata', True))
    _overridden, nonuniform, resolved = analyze_instrumental_methods(main_window, paths, use_dat)
    if nonuniform:
        raise ValueError("the spectra have different instrumental functions: the components need one")
    method_params = dict(resolved[0])
    method_params.update(drive_values(main_window, A, os.path.basename(paths[0])))
    regions = tuple(main_window.exclusion_regions_in_use())
    keep = np.all([fitted(A, y, regions) for y in Y], axis=0)
    if keep.sum() < 2 * len(A) // 3:
        raise ValueError("fewer than two thirds of the points are counted in every spectrum")
    header = (read_dat_metadata_lines(paths[0]) if method_params['source'] == 'file'
              else build_dat_metadata_lines(main_window)[0])
    return dict(files=list(paths), A=A, x=A[keep], Y=Y[:, keep], method_params=method_params, header=header,
                JN=int(main_window.JN0), pol=float(getattr(main_window, 'SMS_pol', SMS_POL_DEFAULT)),
                regions=regions)


class SearchDialog(QDialog):
    """What the search asks: the results folder, the known part, a CMS
    spectrum's non-resonant counts, the layout (module docstring)."""

    def __init__(self, main_window, job):
        super().__init__(main_window)
        self.main_window, self.job = main_window, job
        self.cms = job['method_params']['method'] == 'CMS'
        self.setWindowTitle(ACTION)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(job['files'])} spectra, {len(job['x'])} points each, "
                                f"{job['method_params']['method']}.", self))

        row = QHBoxLayout()
        row.addWidget(QLabel("Results folder", self))
        # a new folder in the spectra's own folder
        self.folder = QLineEdit(os.path.join(os.path.dirname(os.path.abspath(job['files'][0])), 'components'), self)
        row.addWidget(self.folder, 1)
        browse = QPushButton("Browse...", self)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Every spectrum also holds", self))
        self.known = QComboBox(self)
        self.known.addItems(list(KNOWN_PARTS))
        default = default_impurities(1 if self.cms else 3)[1]
        self.known.setCurrentText('Be' if default == 'Be' else 'nothing')
        row.addWidget(self.known)
        row.addStretch(1)
        layout.addLayout(row)

        self.nonres_box = QGroupBox("Non-resonant counts (CMS)", self)
        grid = QGridLayout(self.nonres_box)
        self.by_ratio = QRadioButton("Nnr = k Ns,  k", self.nonres_box)
        self.by_source = QRadioButton("The source's: Nnr = Ns (1 − f_LM)/f_LM + C_high; static noise added flat",
                                      self.nonres_box)
        self.by_ratio.setChecked(True)
        self.k = self._number(CMS_NNR_RATIO)
        self.f_lm, self.c_high, self.c_static = (self._number(v) for v in (DTHETA_SOURCE_F_LM, 0.0, 0.0))
        grid.addWidget(self.by_ratio, 0, 0)
        grid.addWidget(self.k, 0, 1)
        grid.addWidget(self.by_source, 1, 0, 1, 2)
        for r, (label, editor) in enumerate((("f_LM of the source", self.f_lm), ("C_high [counts/point]", self.c_high),
                                             ("Static noise [counts/point]", self.c_static))):
            grid.addWidget(QLabel(label, self.nonres_box), 2 + r, 0)
            grid.addWidget(editor, 2 + r, 1)
        self.nonres_box.setVisible(self.cms)
        layout.addWidget(self.nonres_box)

        box = QGroupBox("Layout", self)
        grid = QGridLayout(box)
        self.series = QRadioButton("A series, in the order of the path box", box)
        self.by_names = QRadioButton("A map, X_Y from the file names", box)
        self.by_grid = QRadioButton("A map of", box)
        self.columns = QSpinBox(box)
        self.columns.setRange(1, len(job['files']))
        side = int(np.ceil(np.sqrt(len(job['files']))))
        self.columns.setValue(side)
        self.rows = QSpinBox(box)
        self.rows.setRange(1, len(job['files']))
        self.rows.setValue(int(np.ceil(len(job['files']) / side)))
        self.order = QComboBox(box)
        self.order.addItems(["row by row (zig-zag)", "alternating (snake)"])
        group = QButtonGroup(self)
        for button in (self.series, self.by_names, self.by_grid):
            group.addButton(button)
        named = coordinates_from_names(job['files']) is not None
        (self.by_names if named else self.series).setChecked(True)
        self.by_names.setEnabled(named)
        grid.addWidget(self.series, 0, 0, 1, 4)
        grid.addWidget(self.by_names, 1, 0, 1, 4)
        grid.addWidget(self.by_grid, 2, 0)
        grid.addWidget(self.columns, 2, 1)
        grid.addWidget(QLabel("columns ×", box), 2, 2)
        grid.addWidget(self.rows, 2, 3)
        grid.addWidget(QLabel("rows,", box), 2, 4)
        grid.addWidget(self.order, 2, 5)
        layout.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.choices = None

    def _number(self, value):
        editor = QDoubleSpinBox(self)
        editor.setDecimals(4)
        editor.setRange(0.0, 1e9)
        editor.setValue(value)
        return editor

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Results folder", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _accept(self):
        """Read and check the answers into self.choices; a refused one says why."""
        try:
            self.choices = self._read()
        except ValueError as e:
            QMessageBox.warning(self, ACTION, str(e))
            return
        self.accept()

    def _read(self):
        n = len(self.job['files'])
        method = self.job['method_params']['method']
        if not self.cms:
            nonres, how = cs.nonresonant(method), dict(mode='none')
        elif self.by_ratio.isChecked():
            nonres, how = cs.nonresonant(method, k=self.k.value()), dict(mode='coefficient', k=self.k.value())
        else:
            nonres = cs.nonresonant(method, f_lm=self.f_lm.value(), c_high=self.c_high.value(),
                                    c_static=self.c_static.value())
            how = dict(mode='source', f_LM=self.f_lm.value(), C_high=self.c_high.value(),
                       C_static=self.c_static.value())
        if self.series.isChecked():
            coordinates, layout = None, dict(kind='series')
        elif self.by_names.isChecked():
            coordinates, layout = coordinates_from_names(self.job['files']), dict(kind='names')
            if len(np.unique(coordinates, axis=0)) < n:
                raise ValueError("two files have the same X_Y")
        else:
            if self.columns.value() * self.rows.value() < n:
                raise ValueError(f"{self.columns.value()} × {self.rows.value()} pixels hold fewer than {n} spectra")
            snake = self.order.currentIndex() == 1
            coordinates = grid_coordinates(n, self.columns.value(), snake)
            layout = dict(kind='grid', columns=self.columns.value(), rows=self.rows.value(),
                          order='snake' if snake else 'zig-zag')
        name = self.known.currentText()
        known = () if KNOWN_PARTS[name] is None else \
            [('Doublet', np.asarray(_read_optics_preset(self.main_window.params_dir, KNOWN_PARTS[name]), dtype=float))]
        folder = self.folder.text().strip()
        if not folder:
            raise ValueError("no results folder")
        return dict(folder=folder, known=known, nonres=nonres, coordinates=coordinates,
                    settings=dict(known=name, known_rows=[[r, p] for r, p in known], nonresonant=how,
                                  layout=layout))


class SearchThread(QThread):
    """The search and the writing of its results. Its signal is 'done', not
    'finished' (that would shadow QThread.finished)."""
    done = Signal(str)          # the results folder
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, main_window, job):
        super().__init__()
        self.main_window, self.job = main_window, job

    def run(self):
        try:
            job, pool = self.job, self.main_window.pool
            axis = cs.axis_for(pool, job['x'], job['JN'], job['method_params'], job['pol'], job['known'],
                               job['nonres'])
            res = cs.run(pool, axis, job['Y'], report=self.progress.emit)
            self.progress.emit("Writing the results...")
            cr.write_results(job['folder'], res, job, pool)
            self.done.emit(job['folder'])
        except m5.FitInterrupted as e:
            self.error.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{e}")


def find_components(main_window):
    """Supp -> Find principal components...: checks, the dialog, the thread."""
    if main_window._reject_if_busy(ACTION):
        return
    warn_if_experimental(main_window, ACTION)
    paths = main_window.parse_process_path()
    if len(paths) < 2:
        main_window.set_status(f"{ACTION} needs two spectra or more in the path box.", "orange")
        return
    problem = main_window.check_spectrum_paths_exist(paths)
    if problem:
        main_window.set_status(problem, "red")
        return
    if main_window.channel_average_refused(paths, ACTION) or not main_window.initialize_parameters():
        return
    try:
        job = prepare_search(main_window, paths)
    except ValueError as e:
        main_window.set_status(f"{ACTION} was not started — {e}", "red")
        return
    dialog = SearchDialog(main_window, job)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        main_window.set_status(f"{ACTION}: canceled", "orange")
        return
    job.update(dialog.choices)
    job['settings'].update(method_params=job['method_params'], JN=job['JN'], pol=job['pol'],
                           nonres=list(job['nonres']), exclusion_regions=[list(r) for r in job['regions']],
                           header=list(job['header']))
    folder = job['folder']
    if os.path.isdir(folder) and os.listdir(folder):
        if QMessageBox.question(main_window, ACTION, f"{folder} is not empty: write the results into it?") \
                != QMessageBox.StandardButton.Yes:
            return
    main_window.inprogress = True
    main_window.busy_with = ACTION
    thread = SearchThread(main_window, job)
    thread.progress.connect(lambda text: main_window.set_status(text, "blue"))
    thread.done.connect(lambda path: _search_done(main_window, path))
    thread.error.connect(lambda message: _search_failed(main_window, message))
    main_window.components_thread = thread
    main_window.set_status(f"{ACTION}: {len(job['files'])} spectra...", "blue")
    main_window._start_calculation(thread)


def _search_done(main_window, folder):
    main_window.set_status(f"The components are in {folder}", "green")
    main_window._calculation_ended()
    open_components_window(main_window, folder)


def _search_failed(main_window, message):
    main_window.set_status(f"{ACTION} stopped: {message}", "orange")
    main_window._calculation_ended()


# --- Show found components... ------------------------------------------------------------

def show_components(main_window):
    """Supp -> Show found components...: a results folder, then its window."""
    warn_if_experimental(main_window, "Show found components")
    folder = QFileDialog.getExistingDirectory(main_window, "Results folder of the components",
                                              main_window.workfolder or os.getcwd())
    if folder:
        open_components_window(main_window, folder)


def open_components_window(main_window, folder):
    try:
        run, res, arrays = cr.read_results(folder)
    except (OSError, ValueError, KeyError) as e:
        main_window.set_status(f"{folder} holds no components found ({e})", "red")
        return None
    window = ComponentsWindow(main_window, folder, run, res, arrays)
    main_window.components_window = window
    window.show()
    return window


class ComponentsWindow(QDialog):
    """The found components (module docstring). Every tab has its own
    controls; the plots follow the main window's theme, the spectra its fit
    display (spectrum_plotter.plot_fitting_result)."""

    def __init__(self, main_window, folder, run, res, arrays):
        super().__init__(main_window)
        self.main_window, self.folder, self.run, self.res, self.arrays = main_window, folder, run, res, arrays
        self.theme = _tc(getattr(main_window, '_theme', None))
        self.files = run['files']
        self.n, self.K = res.w.shape
        self.coordinates = arrays['coordinates']
        self.is_map = len(self.coordinates) == self.n
        self.selected = np.zeros(self.n, dtype=bool)
        self.shown = 0                                     # the spectrum the Spectrum tab shows
        self.images = self._images()
        self._prepare_models()
        self.setWindowTitle(f"Found components: {os.path.basename(folder)} ({self.K} components, {self.n} spectra)")
        self.resize(1300, 860)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)
        self.canvases = {}
        self._components_tab()
        self._lines_tab()
        self._map_tab() if self.is_map else self._evolution_tab()
        self._spectrum_tab()
        self._summed_tab()
        self._scatter_tab()
        row = QHBoxLayout()
        for label, slot in (("Save picture...", self.save_picture), ("Save table...", self.save_table),
                            ("Save spectra by layout...", self.save_spectra_by_layout)):
            button = QPushButton(label, self)
            button.clicked.connect(slot)
            row.addWidget(button)
        self.info = QLabel("", self)
        self.info.setWordWrap(True)
        row.addWidget(self.info, 1)
        layout.addLayout(row)
        self.draw_main()
        self.set_selection(self.selected)
        self.draw_spectrum(0)

    # --- the models of every spectrum, made once --------------------------------------

    def _prepare_models(self):
        """The line model of the search (run.json's settings) on the velocities
        fitted, with the known part and without it, and every spectrum's fit,
        each component on its baseline and the known part alone."""
        settings, res = self.run['settings'], self.res
        method = dict(settings['method_params'])
        method['INS'] = (np.asarray(method['INS'], dtype=float) if isinstance(method['INS'], list)
                         else float(method['INS']))
        if method.get('drive') is not None:
            method['drive'] = tuple(method['drive'])
        known = [(row, np.asarray(p, dtype=float)) for row, p in settings['known_rows']]
        pool, x = self.main_window.pool, self.arrays['x']
        JN, pol, nonres = int(settings['JN']), float(settings['pol']), tuple(settings['nonres'])
        self.axis = cs.axis_for(pool, x, JN, method, pol, known, nonres)
        self.bare = cs.axis_for(pool, x, JN, method, pol, (), nonres) if known else self.axis
        L = np.array([self.axis.line_depth(d, G) for _, d, G in res.lines])
        self.phi = np.tensordot(res.beta, L, axes=1)                       # (K, JN, NV)
        self.fits = self.axis.counts(self.axis.node_sum(np.tensordot(res.w, self.phi, axes=1)), res.base)
        self.parts = np.array([self.bare.counts(self.bare.node_sum(res.w[:, k, None, None] * self.phi[k]), res.base)
                               for k in range(self.K)])                     # (K, n, NV)
        self.known_part = self.axis.counts(self.axis.node_sum(np.zeros_like(L[0])), res.base) if known else None
        self.known_name = settings.get('known', '')

    def baseline_row(self, s):
        """The 8 baseline values of spectrum s (or of the sum of the spectra s)
        as the table writes them: Nnr = r Ns + C_high, its shape linked."""
        s = np.atleast_1d(s)
        base = self.res.base[s]
        ns = base[:, 0].sum()
        lins, c2s = (base[:, 0] @ base[:, 1]) / ns, (base[:, 0] @ base[:, 2]) / ns
        middle = 0.5 * (self.arrays['x'][0] + self.arrays['x'][-1])
        nnr = self.axis.ratio * ns + len(s) * self.axis.c_high
        return np.array([ns, middle, c2s, lins, nnr, middle, c2s, lins])

    # --- styling ----------------------------------------------------------------------

    def _canvas(self, name, parent):
        figure = Figure()
        canvas = _Canvas(figure)
        self.canvases[name] = canvas
        box = QVBoxLayout()
        box.addWidget(NavigationToolbar(canvas, parent))
        box.addWidget(canvas, 1)
        return canvas, box

    def _style(self, figure, axes):
        tc = self.theme
        figure.patch.set_facecolor(tc['figure_facecolor'])
        for ax in axes:
            _style_axis(ax, tc)
            ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
            for text in (ax.xaxis.label, ax.yaxis.label, ax.title):
                text.set_color(tc['axes_text_color'])

    def _legend(self, ax, **kw):
        tc = self.theme
        return ax.legend(fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'],
                         labelcolor=tc['legend_textcolor'], **kw)

    # --- the images (component T's, line T's, D (norm), combinations) -------------------

    def _images(self):
        """[(name, values over the spectra, errors)]."""
        res = self.res
        out = [(f"T{k + 1}", res.w[:, k], res.w_err[:, k]) for k in range(self.K)]
        lines, _ = cr.unique_lines(res)
        T, err = cr.line_T(res, res.w_cov)
        out += [(f"line {lines[q, 1]:+.3f}", T[:, q], err[:, q]) for q in range(len(lines))]
        out.append(("D (norm)", res.dnorm, res.dnorm_sigma))
        return out

    def add_image(self, name, values, errors):
        """A new image (a combination) everywhere images are offered."""
        self.images.append((name, values, errors))
        for combo in (self.threshold_image, self.scatter_x, self.scatter_y):
            combo.addItem(name)
        if self.is_map:
            self._fill_image_table()
        self.draw_main()

    # --- Components ---------------------------------------------------------------------

    def _components_tab(self):
        page = QWidget(self.tabs)
        canvas, box = self._canvas("Components", page)
        page.setLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show", page))
        self.which = QComboBox(page)
        self.which.addItems(["all (IF)"] + [f"component {k + 1} (IF)" for k in range(self.K)]
                            + ["all (lines only)"] + [f"component {k + 1} (lines only)" for k in range(self.K)])
        self.which.setToolTip("IF: the spectrum the component makes at its mean T, through the instrumental function")
        self.which.currentIndexChanged.connect(self.draw_components)
        row.addWidget(self.which)
        row.addStretch(1)
        combine = QPushButton("Combine selected", page)
        combine.setToolTip("Components of one phase as one image: their T summed")
        combine.clicked.connect(self.combine_selected)
        row.addWidget(combine)
        box.addLayout(row)
        T = cr.component_T(self.res.w, self.res.w_err)
        self.component_table = QTableWidget(self.K, 4, page)
        self.component_table.setHorizontalHeaderLabels(["Component", "T (mean where significant)", "Lines",
                                                        "Places, mm/s (G)"])
        self.component_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for k in range(self.K):
            own = np.flatnonzero(self.res.mask[k])
            order = own[np.argsort(self.res.lines[own, 1])]
            cells = [f"{k + 1}", f"{T[k]:.3f}", f"{len(own)}",
                     ", ".join(f"{self.res.lines[j, 1]:+.3f} ({self.res.lines[j, 2]:.2f})" for j in order)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(model_colors[k % len(model_colors)]) if c == 0 else item.foreground())
                self.component_table.setItem(k, c, item)
        self.component_table.resizeColumnsToContents()
        self.component_table.setMaximumHeight(30 + 26 * min(self.K, 6))
        box.addWidget(self.component_table)
        self.tabs.addTab(page, "Components")
        self.draw_components()

    def component_spectrum(self, k):
        """Component k alone at its mean T (where significant) through the
        instrumental function, the known part left out: its transmission."""
        T = cr.component_T(self.res.w, self.res.w_err)[k]
        T = float(T) if np.isfinite(T) else float(self.res.w[:, k].mean())
        flat = np.array([1.0, 0.0, 0.0])
        empty = self.bare.counts(self.bare.node_sum(np.zeros_like(self.phi[k])), flat)
        return T, self.bare.counts(self.bare.node_sum(T * self.phi[k]), flat) / empty

    def _profile(self, lines):
        """An absorption profile of unit area of the lines (strength, δ, G) on the velocities."""
        E = np.linspace(self.arrays['A'].min(), self.arrays['A'].max(), 2000)
        out = np.zeros_like(E)
        for t, d, G in lines:
            out += t * m5.Voigt(NAT_WIDTH, float(G), E - d)
        return E, out

    def draw_components(self):
        figure = self.canvases["Components"].figure
        figure.clear()
        ax = figure.add_subplot(111)
        index = self.which.currentIndex()
        through_if = index <= self.K
        one = index % (self.K + 1) - 1
        for k in (range(self.K) if one < 0 else [one]):
            color = model_colors[k % len(model_colors)]
            if through_if:
                T, tr = self.component_spectrum(k)
                ax.plot(self.arrays['x'], tr, color=color, label=f"component {k + 1}, T {T:.2f}")
            else:
                own = np.flatnonzero(self.res.mask[k])
                E, v = self._profile(np.column_stack([self.res.beta[k, own], self.res.lines[own, 1:]]))
                ax.plot(E, -v, color=color, label=f"component {k + 1}")
        ax.set_xlabel("Velocity, mm/s")
        ax.set_ylabel("Transmission (through the instrumental function)" if through_if
                      else "Absorption of the lines alone (area 1)")
        self._style(figure, [ax])
        self._legend(ax, loc='lower right')
        self.canvases["Components"].draw_idle()

    def combine_selected(self):
        rows = sorted({index.row() for index in self.component_table.selectedIndexes()})
        if len(rows) < 2:
            self.info.setText("Select two components or more in the table to combine")
            return
        cov = self.res.w_cov[:, rows][:, :, rows]
        values = self.res.w[:, rows].sum(axis=1)
        errors = np.sqrt(np.maximum(cov.sum(axis=(1, 2)), 0.0))
        self.add_image("+".join(f"T{k + 1}" for k in rows), values, errors)
        self.info.setText(f"Combined: {'+'.join(f'T{k + 1}' for k in rows)} is now an image")

    # --- Lines --------------------------------------------------------------------------

    def _lines_tab(self):
        page = QWidget(self.tabs)
        canvas, box = self._canvas("Lines", page)
        page.setLayout(box)
        note = QLabel("A line held by several components is one line copied into each: "
                      "the table gives its strength in every component that holds it.", page)
        note.setWordWrap(True)
        box.addWidget(note)
        lines, which = cr.unique_lines(self.res)
        order = np.argsort(lines[:, 1])
        table = QTableWidget(len(lines), 2 + self.K, page)
        table.setHorizontalHeaderLabels(["Place, mm/s", "G, mm/s"] + [f"in {k + 1}" for k in range(self.K)])
        for r, q in enumerate(order):
            table.setItem(r, 0, QTableWidgetItem(f"{lines[q, 1]:+.3f}"))
            table.setItem(r, 1, QTableWidgetItem(f"{lines[q, 2]:.3f}"))
            for k in range(self.K):
                cols = np.flatnonzero((which == q) & self.res.mask[k])
                if len(cols):
                    item = QTableWidgetItem(f"{100 * self.res.beta[k, cols].sum():.1f} %")
                    item.setForeground(QColor(model_colors[k % len(model_colors)]))
                    table.setItem(r, 2 + k, item)
        table.resizeColumnsToContents()
        table.setMaximumHeight(240)
        box.addWidget(table)
        self.tabs.addTab(page, "Lines")
        figure = canvas.figure
        ax = figure.add_subplot(111)
        for q in order:
            E, v = self._profile([[1.0, lines[q, 1], lines[q, 2]]])
            ax.plot(E, -v / v.max(), color=self.theme['axes_text_color'], lw=0.8)
            holders = [k for k in range(self.K) if np.any((which == q) & self.res.mask[k])]
            for t, k in enumerate(holders):                        # under the line: who holds it
                ax.plot([lines[q, 1]], [-1.08 - 0.07 * t], 's', ms=5, color=model_colors[k % len(model_colors)])
        for k in range(self.K):
            ax.plot([], [], 's', color=model_colors[k % len(model_colors)], label=f"held by component {k + 1}")
        ax.set_xlabel("Velocity, mm/s")
        ax.set_ylabel("Absorption of each line (scaled to 1)")
        ax.set_title(f"{len(lines)} lines, {int(self.res.mask.sum())} with their copies")
        self._style(figure, [ax])
        self._legend(ax, loc='lower right')

    # --- Evolution (a series) and Map ----------------------------------------------------

    def _evolution_tab(self):
        page = QWidget(self.tabs)
        canvas, box = self._canvas("Evolution", page)
        page.setLayout(box)
        canvas.mpl_connect('button_press_event', self._clicked)
        canvas.mpl_connect('motion_notify_event', self._hovered)
        self.tabs.addTab(page, "Evolution")

    def draw_evolution(self, figure, selection=False):
        """Every component's T along the series with its error; under it every
        spectrum's D (norm) with its 1σ (how well the components fit it).
        *selection*: the selected spectra marked."""
        figure.clear()
        top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
        number = np.arange(1, self.n + 1)
        for k in range(self.K):
            top.errorbar(number, self.res.w[:, k], yerr=self.res.w_err[:, k], fmt='o-', ms=3,
                         color=model_colors[k % len(model_colors)], label=f"T{k + 1}")
        for name, values, errors in self.images[len(self.images) - len(self._combined()):]:
            top.errorbar(number, values, yerr=errors, fmt='s--', ms=3, color=self.theme['axes_text_color'],
                         label=name)
        bottom.errorbar(number, self.res.dnorm, yerr=self.res.dnorm_sigma, fmt='o', ms=3,
                        color=self.theme['axes_text_color'])
        bottom.axhline(1.0, color='lime', lw=0.8, ls='--')
        if selection and self.selected.any():
            for ax in (top, bottom):
                for s in np.flatnonzero(self.selected):
                    ax.axvspan(s + 0.5, s + 1.5, color='gold', alpha=0.25, lw=0)
        top.set_ylabel("T")
        bottom.set_ylabel("D (norm)")
        bottom.set_xlabel("Spectrum")
        self._style(figure, [top, bottom])
        self._legend(top, loc='best')
        return top

    def _combined(self):
        """The images made by Combine."""
        return [im for im in self.images if '+' in im[0]]

    def _map_tab(self):
        page = QWidget(self.tabs)
        outer = QHBoxLayout(page)
        canvas, box = self._canvas("Map", page)
        outer.addLayout(box, 3)
        canvas.mpl_connect('button_press_event', self._clicked)
        canvas.mpl_connect('motion_notify_event', self._hovered)
        self.rowcol, self.grid = pixel_grid(self.coordinates)
        panel = QVBoxLayout()
        outer.addLayout(panel, 1)
        panel.addWidget(QLabel("Images: R, G, B (a channel shows the sum of its images)", page))
        self.table = QTableWidget(0, 6, page)
        self.table.setHorizontalHeaderLabels(["Image", "min", "max", "R", "G", "B"])
        self._fill_image_table()
        self.table.itemChanged.connect(lambda _item: self.draw_main())
        panel.addWidget(self.table, 1)
        self.common = QCheckBox("One scale for R, G, B", page)
        self.common.setToolTip("Equal brightness is equal T in every colour")
        self.dim = QCheckBox("Dark where not significant", page)
        self.dim.setChecked(True)
        for box_ in (self.common, self.dim):
            box_.toggled.connect(lambda _on: self.draw_main())
            panel.addWidget(box_)
        scale = QGridLayout()
        self.low, self.high = [], []
        for c, colour in enumerate("RGB"):
            scale.addWidget(QLabel(f"{colour} from", page), c, 0)
            for col, (spins, value) in enumerate(((self.low, 0), (self.high, 100))):
                spin = QSpinBox(page)
                spin.setRange(0, 100)
                spin.setSuffix(" %")
                spin.setValue(value)
                spin.valueChanged.connect(lambda _v: self.draw_main())
                spins.append(spin)
                scale.addWidget(spin, c, 1 + col)
        scale.addWidget(QLabel("of the largest", page), 0, 3)
        full = QPushButton("Full range", page)
        full.clicked.connect(self.full_range)
        scale.addWidget(full, 1, 3)
        panel.addLayout(scale)
        self.tabs.addTab(page, "Map")

    def _fill_image_table(self):
        self.table.blockSignals(True)
        ticked = {(r, c) for r in range(self.table.rowCount()) for c in range(3)
                  if self.table.item(r, 3 + c) is not None
                  and self.table.item(r, 3 + c).checkState() == Qt.CheckState.Checked}
        first = self.table.rowCount() == 0
        self.table.setRowCount(len(self.images))
        for i, (name, values, _err) in enumerate(self.images):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(f"{np.nanmin(values):.3g}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{np.nanmax(values):.3g}"))
            for c in range(3):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                on = (i == c and i < self.K) if first else (i, c) in ticked
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                self.table.setItem(i, 3 + c, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def full_range(self):
        """Every channel from 0 to 100 % of its largest value."""
        for spins, value in ((self.low, 0), (self.high, 100)):
            for spin in spins:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self.draw_main()

    def channels(self):
        """Per channel R, G, B: (its sum image over the spectra, its error), or None when no image is ticked."""
        out = []
        for c in range(3):
            ticked = [i for i in range(len(self.images))
                      if self.table.item(i, 3 + c).checkState() == Qt.CheckState.Checked]
            if not ticked:
                out.append(None)
                continue
            out.append((np.sum([self.images[i][1] for i in ticked], axis=0),
                        np.sqrt(np.sum([self.images[i][2] ** 2 for i in ticked], axis=0))))
        return out

    def composite(self):
        """The RGB picture of the map (rows, columns, 3) and the channels' names."""
        shown = []
        for ch in self.channels():
            if ch is None:
                shown.append(None)
                continue
            values, errors = ch
            shown.append(np.where(values > cs.normal_point() * errors, values, 0.0) if self.dim.isChecked()
                         else values)
        tops = [np.nanmax(v) if v is not None and np.any(np.isfinite(v)) else 0.0 for v in shown]
        rgb = np.zeros(self.grid.shape + (3,))
        for c, values in enumerate(shown):
            if values is None:
                continue
            top = max(tops) if self.common.isChecked() else tops[c]
            lo, hi = self.low[c].value() / 100 * top, self.high[c].value() / 100 * top
            rgb[self.rowcol[:, 0], self.rowcol[:, 1], c] = (np.clip((values - lo) / (hi - lo), 0.0, 1.0)
                                                            if hi > lo else 0.0)
        rgb[self.grid < 0] = MISSING_COLOR
        names = ["+".join(self.images[i][0] for i in range(len(self.images))
                          if self.table.item(i, 3 + c).checkState() == Qt.CheckState.Checked) or "—"
                 for c in range(3)]
        return rgb, names

    def draw_map_on(self, figure, selection=False):
        """The RGB map on *figure* (with the selection overlaid when asked)."""
        figure.clear()
        ax = figure.add_subplot(111)
        rgb, names = self.composite()
        ax.imshow(rgb, origin='lower', interpolation='nearest', aspect='equal')
        if selection:
            overlay = np.zeros(self.grid.shape + (4,))
            overlay[self.rowcol[self.selected, 0], self.rowcol[self.selected, 1]] = (1.0, 1.0, 1.0, 0.45)
            ax.imshow(overlay, origin='lower', interpolation='nearest', aspect='equal')
        xs, ys = np.unique(self.coordinates[:, 0]), np.unique(self.coordinates[:, 1])
        step_x, step_y = (int(np.ceil(len(v) / 10)) for v in (xs, ys))       # ten labels at most
        ax.set_xticks(range(0, len(xs), step_x), [f"{v:g}" for v in xs[::step_x]])
        ax.set_yticks(range(0, len(ys), step_y), [f"{v:g}" for v in ys[::step_y]])
        ax.set_title(f"R: {names[0]}   G: {names[1]}   B: {names[2]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self._style(figure, [ax])
        ax.grid(False)
        return ax

    def draw_main(self):
        """The Map or the Evolution tab, and the selection's plot."""
        if self.is_map:
            self.draw_map_on(self.canvases["Map"].figure)
        else:
            self.draw_evolution(self.canvases["Evolution"].figure)
        self.canvases["Map" if self.is_map else "Evolution"].draw_idle()
        if "Selection" in self.canvases:
            self.draw_selection()

    # --- Spectrum ---------------------------------------------------------------------

    def _spectrum_tab(self):
        page = QWidget(self.tabs)
        canvas, box = self._canvas("Spectrum", page)
        page.setLayout(box)
        self.spectrum_table = QTableWidget(1, self.K + 1, page)
        self.spectrum_table.setHorizontalHeaderLabels([f"T{k + 1}" for k in range(self.K)] + ["D (norm)"])
        self.spectrum_table.setMaximumHeight(64)
        box.addWidget(self.spectrum_table)
        self.tabs.addTab(page, "Spectrum")

    def plot_fit(self, figure, s, title):
        """Spectra s (one, or several summed) with the components' fit in the
        main window's style: every component on the baseline, the known part,
        the residuals, the χ² of the counts."""
        s = np.atleast_1d(s)
        x = self.arrays['x']
        y = self.arrays['Y'][s].sum(axis=0)
        fit = self.fits[s].sum(axis=0)
        parts = [self.parts[k, s].sum(axis=0) for k in range(self.K)]
        names = [f"Component {k + 1}" for k in range(self.K)]
        colors = ['silver'] + [model_colors[k % len(model_colors)] for k in range(self.K)]
        positions = []
        for k in range(self.K):
            own = np.flatnonzero(self.res.mask[k])
            positions.append((self.res.lines[own, 1], self.res.beta[k, own]))
        if self.known_part is not None:
            parts.append(self.known_part[s].sum(axis=0))
            names.append(self.known_name)
            colors.append('grey')
            positions.append(([], []))
        n_free = len(s) * (self.K + 3)
        chi2 = float(np.sum((y - fit) ** 2 / fit) / max(len(x) - n_free, 1))
        plot_fitting_result(figure, x, y, fit, parts, positions, self.baseline_row(s), colors, chi2, title,
                            self.main_window.dir_path, gridcolor=self.theme['gridcolor'], theme=self.theme,
                            model=names, chi2_spread=mi.chi2_spread(len(x), n_free), save=False)
        legend = figure.axes[0].get_legend()
        if legend is not None:
            legend.set_visible(True)

    def draw_spectrum(self, s):
        """Spectrum s in the Spectrum tab, and the table of its components' T."""
        self.shown = s
        self.plot_fit(self.canvases["Spectrum"].figure, s, self.files[s])
        self.canvases["Spectrum"].draw_idle()
        for k in range(self.K):
            self.spectrum_table.setItem(0, k, QTableWidgetItem(f"{self.res.w[s, k]:.3f} ± {self.res.w_err[s, k]:.3f}"))
        self.spectrum_table.setItem(0, self.K, QTableWidgetItem(
            f"{self.res.dnorm[s]:.3f} ± {self.res.dnorm_sigma[s]:.3f}"))

    def _spectrum_at(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        if not self.is_map:
            s = int(round(event.xdata)) - 1
            return s if 0 <= s < self.n else None
        col, row = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1] and self.grid[row, col] >= 0:
            return int(self.grid[row, col])
        return None

    def _clicked(self, event):
        s = self._spectrum_at(event)
        if s is not None:
            self.draw_spectrum(s)
            self.tabs.setCurrentIndex(3)

    def _hovered(self, event):
        s = self._spectrum_at(event)
        if s is None:
            return
        where = f"X {self.coordinates[s, 0]:g}, Y {self.coordinates[s, 1]:g}, " if self.is_map else ""
        self.info.setText(f"{where}{os.path.basename(self.files[s])}: " +
                          ", ".join(f"T{k + 1} {self.res.w[s, k]:.3f} ± {self.res.w_err[s, k]:.3f}"
                                    for k in range(self.K)) +
                          f"; D (norm) {self.res.dnorm[s]:.3f} ± {self.res.dnorm_sigma[s]:.3f}")

    # --- Summed spectrum: the selection and its sum -------------------------------------

    def _summed_tab(self):
        page = QWidget(self.tabs)
        outer = QVBoxLayout(page)
        selection_canvas, selection_box = self._canvas("Selection", page)
        summed_canvas, summed_box = self._canvas("Summed", page)
        plots = QHBoxLayout()
        plots.addLayout(selection_box, 1)
        plots.addLayout(summed_box, 1)
        outer.addLayout(plots, 1)
        selection_canvas.mpl_connect('button_press_event', self._pressed)
        selection_canvas.mpl_connect('motion_notify_event', self._dragged)
        selection_canvas.mpl_connect('button_release_event', self._released)
        controls = QGridLayout()
        self.threshold_image = QComboBox(page)
        self.threshold_image.addItems([name for name, _, _ in self.images])
        self.threshold_side = QComboBox(page)
        self.threshold_side.addItems(["above", "below"])
        self.threshold = QSlider(Qt.Orientation.Horizontal, page)
        self.threshold.setRange(0, 1000)
        self.threshold.setValue(500)
        self.threshold_value = QLabel("", page)
        self.threshold.valueChanged.connect(self._threshold_moved)
        self.threshold_image.currentIndexChanged.connect(self._threshold_moved)
        select = QPushButton("Select", page)
        select.setToolTip("The spectra where the image is above (below) the level")
        select.clicked.connect(self.select_by_threshold)
        self.brush = QCheckBox("Brush", page)
        self.brush.setToolTip("Paint on the map, or drag along the series, to add spectra")
        self.erase = QCheckBox("erase", page)
        self.brush_size = QSpinBox(page)
        self.brush_size.setRange(0, 20)
        self.brush_size.setPrefix("radius ")
        self.brush_size.setEnabled(self.is_map)
        clear = QPushButton("Clear", page)
        clear.clicked.connect(lambda: self.set_selection(np.zeros(self.n, dtype=bool)))
        self.save_sum = QPushButton("Save summed spectrum...", page)
        self.save_sum.setToolTip("The selected files summed, saved and loaded to fit (as Sum all spectra)")
        self.save_sum.clicked.connect(self.sum_selected)
        for col, widget in enumerate((self.threshold_image, self.threshold_side, self.threshold,
                                      self.threshold_value, select)):
            controls.addWidget(widget, 0, col)
        for col, widget in enumerate((self.brush, self.erase, self.brush_size, clear)):
            controls.addWidget(widget, 1, col)
        controls.addWidget(self.save_sum, 1, 4)
        self.selection_text = QLabel("", page)
        self.selection_text.setWordWrap(True)
        controls.addWidget(self.selection_text, 2, 0, 1, 5)
        outer.addLayout(controls)
        self.tabs.addTab(page, "Summed spectrum")
        self._threshold_moved()
        self._drag_start = None

    def draw_selection(self):
        figure = self.canvases["Selection"].figure
        if self.is_map:
            self.draw_map_on(figure, selection=True)
        else:
            self.draw_evolution(figure, selection=True)
        self.canvases["Selection"].draw_idle()
        figure = self.canvases["Summed"].figure
        figure.clear()
        chosen = np.flatnonzero(self.selected)
        if len(chosen):
            self.plot_fit(figure, chosen, f"{len(chosen)} spectra summed")
        else:
            ax = figure.add_subplot(111)
            ax.set_title("Select spectra: by threshold, or with the brush")
            self._style(figure, [ax])
        self.canvases["Summed"].draw_idle()

    def _threshold_level(self):
        _name, values, _err = self.images[self.threshold_image.currentIndex()]
        lo, hi = np.nanmin(values), np.nanmax(values)
        return lo + (hi - lo) * self.threshold.value() / 1000

    def _threshold_moved(self):
        self.threshold_value.setText(f"{self._threshold_level():.4g}")

    def select_by_threshold(self):
        """The spectra where the image is above (below) the level: the selection."""
        values = self.images[self.threshold_image.currentIndex()][1]
        level = self._threshold_level()
        self.set_selection(values > level if self.threshold_side.currentIndex() == 0 else values < level)

    def set_selection(self, selected):
        self.selected = np.asarray(selected, dtype=bool)
        count = int(self.selected.sum())
        if count:
            means = self.res.w[self.selected].mean(axis=0)
            self.selection_text.setText(f"{count} spectra; mean " +
                                        ", ".join(f"T{k + 1} {means[k]:.3f}" for k in range(self.K)))
        else:
            self.selection_text.setText("No spectrum selected")
        self.save_sum.setEnabled(count >= 2)
        self.draw_selection()
        self.draw_scatter()

    def paint(self, event):
        """The brush on the map: the pixels within its radius added (erased)."""
        if event.xdata is None or event.ydata is None:
            return
        near = np.hypot(self.rowcol[:, 1] - event.xdata, self.rowcol[:, 0] - event.ydata) \
            <= self.brush_size.value() + 0.5
        selected = self.selected.copy()
        selected[near] = not self.erase.isChecked()
        self.set_selection(selected)

    def _pressed(self, event):
        if not self.brush.isChecked() or event.xdata is None:
            return
        if self.is_map:
            self.paint(event)
        else:
            self._drag_start = event.xdata

    def _dragged(self, event):
        if self.brush.isChecked() and self.is_map and event.button is not None:
            self.paint(event)

    def _released(self, event):
        """The brush along a series: the spectra between press and release added (erased)."""
        if self.is_map or self._drag_start is None or event.xdata is None:
            self._drag_start = None
            return
        lo, hi = sorted((self._drag_start, event.xdata))
        self._drag_start = None
        number = np.arange(1, self.n + 1)
        span = (number >= lo - 0.5) & (number <= hi + 0.5)
        selected = self.selected.copy()
        selected[span] = not self.erase.isChecked()
        self.set_selection(selected)

    def sum_selected(self):
        """The selected spectra's files summed by "Sum all spectra"'s function, saved and loaded to fit."""
        sum_spectra(self.main_window, [self.files[s] for s in np.flatnonzero(self.selected)])

    # --- Scatter ------------------------------------------------------------------------

    def _scatter_tab(self):
        page = QWidget(self.tabs)
        canvas, box = self._canvas("Scatter", page)
        page.setLayout(box)
        row = QHBoxLayout()
        self.scatter_x, self.scatter_y = QComboBox(page), QComboBox(page)
        for combo, label, start in ((self.scatter_x, "x", 0), (self.scatter_y, "y", min(1, len(self.images) - 1))):
            combo.addItems([name for name, _, _ in self.images])
            combo.setCurrentIndex(start)
            combo.currentIndexChanged.connect(self.draw_scatter)
            row.addWidget(QLabel(label, page))
            row.addWidget(combo)
        row.addStretch(1)
        box.addLayout(row)
        self.tabs.addTab(page, "Scatter")

    def draw_scatter(self):
        if "Scatter" not in self.canvases:
            return
        figure = self.canvases["Scatter"].figure
        figure.clear()
        ax = figure.add_subplot(111)
        (nx, vx, ex), (ny, vy, ey) = self.images[self.scatter_x.currentIndex()], \
            self.images[self.scatter_y.currentIndex()]
        ax.errorbar(vx, vy, xerr=ex, yerr=ey, fmt='o', ms=3, color='gray', alpha=0.6)
        if self.selected.any():
            ax.plot(vx[self.selected], vy[self.selected], 'o', ms=5, color='gold')
        ax.set_xlabel(nx)
        ax.set_ylabel(ny)
        self._style(figure, [ax])
        self.canvases["Scatter"].draw_idle()

    # --- export --------------------------------------------------------------------------

    def save_picture(self):
        name = self.tabs.tabText(self.tabs.currentIndex())
        canvas = self.canvases.get(name) or self.canvases.get("Summed")
        path, _ = QFileDialog.getSaveFileName(self, "Save picture", os.path.join(self.folder, f"{name}.png"),
                                              "PNG (*.png)")
        if path:
            canvas.figure.savefig(path, dpi=200, facecolor=self.theme['figure_facecolor'])
            self.info.setText(f"Saved {os.path.basename(path)}")

    def save_table(self):
        """The images shown (on the map: those ticked; else all) as a table: the spectrum, X, Y, value and error of each."""
        if self.is_map:
            shown = [i for i in range(len(self.images))
                     if any(self.table.item(i, 3 + c).checkState() == Qt.CheckState.Checked for c in range(3))]
        else:
            shown = list(range(len(self.images)))
        path, _ = QFileDialog.getSaveFileName(self, "Save table", os.path.join(self.folder, "images.txt"),
                                              "Text (*.txt)")
        if not path:
            return
        head = ['spectrum', 'file'] + (['X', 'Y'] if self.is_map else []) + \
            sum(([self.images[i][0], f"d {self.images[i][0]}"] for i in shown), [])
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\t'.join(head) + '\n')
            for s in range(self.n):
                cells = [str(s + 1), os.path.basename(self.files[s])]
                if self.is_map:
                    cells += [repr(float(v)) for v in self.coordinates[s]]
                cells += [repr(float(v)) for i in shown for v in (self.images[i][1][s], self.images[i][2][s])]
                f.write('\t'.join(cells) + '\n')
        self.info.setText(f"Saved {os.path.basename(path)}")

    def save_spectra_by_layout(self):
        """The input spectra saved into a folder, named by the layout -- X_Y.dat
        on a map, the number in the series -- each with its instrumental
        header: a sequence for the user's own fits."""
        folder = QFileDialog.getExistingDirectory(self, "Folder for the spectra",
                                                  os.path.join(self.folder, 'spectra_by_layout'))
        if not folder:
            return
        header = list(self.run['settings'].get('header', []))
        A_list, B_list = load_spectrum(self.main_window, list(self.files),
                                       calibration_path=self.main_window.calibration_path)
        width = len(str(self.n))
        for s, (a, b) in enumerate(zip(A_list, B_list)):
            name = (f"{self.coordinates[s, 0]:g}_{self.coordinates[s, 1]:g}.dat" if self.is_map
                    else f"{s + 1:0{width}d}.dat")
            save_spectrum_with_metadata(os.path.join(folder, name), a, b, header)
        self.info.setText(f"{len(A_list)} spectra saved in {folder}")
