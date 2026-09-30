# -*- coding: utf-8 -*-
"""
@author: YAROSLAVTSEV S

The MIT license follows:

Copyright (c) European Synchrotron Radiation Facility (ESRF)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import re
import sys
import json
import shutil
import warnings
import traceback
import ast
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
# NOTE: user-typed Expression/Distr/Corr strings are evaluated through
# minimi_lib._eval_expr, whose module namespace provides the bare numpy names
# (sin, sqrt, ...). Do not re-introduce a ``from numpy import ...`` block here
# for that purpose.
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # select the Qt backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QCheckBox, QScrollArea, QGridLayout,
    QSplitter, QFrame, QFileDialog, QMessageBox, QMenu, QSizePolicy,
    QAbstractScrollArea
)
from PySide6.QtGui import (
    QImage, QFont, QIcon, QAction, QDoubleValidator, QIntValidator,
    QPalette, QShortcut, QKeySequence
)
from PySide6.QtCore import Qt, Signal, QThread, QSize, QLocale, QStandardPaths, QCoreApplication, QTimer

# ---------------------------------------------------------------------------
# Local package
# ---------------------------------------------------------------------------
import syncmoss.minimi_lib as mi
import syncmoss.fitting_io as fitting_io
from syncmoss.models import TI, FIT_CANCEL, FitInterrupted
from syncmoss.Calibration import Calibration
from syncmoss.constants import (model_colors, number_of_baseline_parameters,
                               SMS_POL_DEFAULT)
from syncmoss.parameters_table import ParametersTable
from syncmoss.results_table import ResultsTable
from syncmoss.model_io import (
    load_model, read_model, save_model, save_model_as, mod_len_def,
    model_file_rows, fitted_model_rows, save_result_model, result_model_path,
    strip_known_extension, validate_user_expressions,
)
from syncmoss.spectrum_io import (
    load_spectrum, sum_all_spectra, subtract_model_from_spectrum,
    half_points, calculate_backgrounds,
)
from syncmoss import bliss_channel
from syncmoss.spectrum_plotter import (
    plot_fitting_result, plot_simultaneous_fitting_result, plot_instrumental_result,
    plot_distribution, plot_calibration, plot_model, plot_model_with_nbaseline,
    plot_spectrum, plot_model_without_spectrum, calculate_z_order, distribution_curves,
    calculate_baseline,
)
from syncmoss.instrumental_io import (
    DEFAULT_INSTRUMENTAL_METHOD,
    instrumental,
    instrumental_theory,
    recentre_instrumental_after_calibration,
    reset_instrumental_defaults,
    resolve_instrumental_for_file,
    analyze_instrumental_methods,
    build_dat_metadata_lines,
    compute_norm,
    hires_model_diff,
    get_sms_instrumental_from_global_files,
)
from syncmoss.Library_window import save_to_library_via_dialog
from syncmoss.supp_menu import (
    build_supp_menu, contact_icon, open_theory_find_dialog, theme_action_text,
)


class CustomNavigationToolbar(NavigationToolbar):
    """Custom matplotlib toolbar with toggle button for model line positions and legend."""
    
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.parent_window = parent
        
        # Add separator
        self.addSeparator()
        
        # Add toggle positions button
        self.toggle_positions_action = self.addAction('Toggle\nPositions', self.toggle_positions)
        self.toggle_positions_action.setCheckable(True)
        self.toggle_positions_action.setChecked(True)  # Start as visible (positions default ON)
        self.toggle_positions_action.setToolTip('Show/Hide model line positions')
        self.toggle_positions_action.setEnabled(False)  # Disabled by default

        # Add toggle legend button
        self.toggle_legend_action = self.addAction('Toggle\nLegend', self.toggle_legend)
        self.toggle_legend_action.setCheckable(True)
        self.toggle_legend_action.setChecked(False)  # Start as hidden (legend default OFF)
        self.toggle_legend_action.setToolTip('Show/Hide legend')
        self.toggle_legend_action.setEnabled(False)  # Disabled by default
    
    def home(self, *args):
        """Reset view and reapply tight layout."""
        super().home(*args)
        self.canvas.draw_idle()

    def toggle_positions(self):
        """Toggle visibility of model line positions."""
        if hasattr(self.parent_window, 'toggle_position_markers'):
            show = self.toggle_positions_action.isChecked()
            self.parent_window.toggle_position_markers(show)
    
    def toggle_legend(self):
        """Toggle visibility of legend."""
        if hasattr(self.parent_window, 'toggle_legend_visibility'):
            show = self.toggle_legend_action.isChecked()
            self.parent_window.toggle_legend_visibility(show)


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore', '.*object is not callable.*', )

# Dark plot styling shared by every matplotlib figure created by the app.
plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
plt.rcParams['axes.labelcolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'w'
plt.rcParams['xtick.color'] = 'w'
plt.rcParams['ytick.color'] = 'w'

# Optional Tango (beamline control) integration is disabled by default.
# To enable, install PyTango and provide get_data()/tango_uri below.
check_tango = False
# from PyTango import DeviceProxy
# def get_data(tango_uri):
#     proxy = DeviceProxy(tango_uri)
#     return proxy.data
# tango_uri = 'moesa:20000/id14/Can556/6a2'  # could be different
# check_tango = True

def _is_frozen_macos():
    """True inside the frozen macOS ``.app``: its bundled folders are not
    reliably writable (code-signed, or launched from a read-only mount)."""
    return getattr(sys, 'frozen', False) and sys.platform == 'darwin'


def _macos_app_data_dir():
    """The per-user writable location of the frozen macOS app
    (``QStandardPaths.AppDataLocation`` -> ``~/Library/Application Support/SYNCmoss``)."""
    QCoreApplication.setApplicationName('SYNCmoss')  # deterministic AppData path
    target = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not target:
        target = os.path.expanduser('~/Library/Application Support/SYNCmoss')
    return target


def _seed_from_bundle(bundled, target):
    """Make ``target`` the writable copy of the bundled folder ``bundled``.

    If any bundled entry is missing in ``target``, ALL bundled entries are
    (re-)copied, overwriting existing ones: a missing file means the software
    logic changed, so to be safe everything is redone. While everything exists
    the user's changes are preserved. Entries only ``target`` has (the user's
    own files) are never touched.
    """
    os.makedirs(target, exist_ok=True)

    required = os.listdir(bundled) if os.path.isdir(bundled) else []
    if not all(os.path.exists(os.path.join(target, name)) for name in required):
        for name in required:
            src = os.path.join(bundled, name)
            dst = os.path.join(target, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    return target


def _resolve_params_dir(base_dir):
    """Return the directory that holds the parameter files, Calibration.dat and
    the generated calibr.png.

    Normally this is the bundled ``parameters/`` folder shipped next to the app.
    A *frozen macOS* ``.app`` is code-signed / launched from a read-only mount, so
    that folder is not reliably writable; redirect to a per-user writable location
    (``QStandardPaths.AppDataLocation`` -> ``~/Library/Application Support/SYNCmoss``)
    and seed it from the bundled originals (``_seed_from_bundle``: if any required
    file is missing there, ALL bundled files are (re-)copied, overwriting existing
    ones).

    Windows and source checkouts keep using the in-place ``parameters/`` folder.
    """
    bundled = os.path.join(base_dir, 'parameters')
    if not _is_frozen_macos():
        return bundled
    return _seed_from_bundle(bundled, _macos_app_data_dir())


def _resolve_library_dir(base_dir):
    """Return the model Library folder ("Save to library" and "Import Library"
    write into it).

    The same redirect as ``_resolve_params_dir``, one level down: on a frozen
    macOS ``.app`` the bundled ``Library/`` becomes
    ``~/Library/Application Support/SYNCmoss/Library``, seeded from the bundled
    models by the same rule. Windows and source checkouts keep using the
    in-place ``Library/`` folder.
    """
    bundled = os.path.join(base_dir, 'Library')
    if not _is_frozen_macos():
        return bundled
    return _seed_from_bundle(bundled, os.path.join(_macos_app_data_dir(), 'Library'))


class CalibrationThread(QThread):
    """Thread for running calibration without blocking the UI"""
    finished = Signal(object, object, object)  # A, B, C
    error = Signal(str)
    
    def __init__(self, dir_path, file, experimental_method, INS, JN, x0, MulCo, vel_start, pool, GCMS=0.1):
        super().__init__()
        self.dir_path = dir_path
        self.file = file
        self.experimental_method = experimental_method #previously called "VVV"
        self.INS = INS
        self.JN = JN
        self.x0 = x0
        self.MulCo = MulCo
        self.vel_start = vel_start
        self.pool = pool  # Store reference to global pool
        self.GCMS = GCMS  # single Gaussian width used as the CMS instrumental function

    def run(self):
        try:
            # Use the global pool (passed to constructor)
            pool = self.pool

            # Run calibration
            A, B, C = Calibration(
                self.dir_path, self.file, pool,
                self.experimental_method, self.INS, self.JN,
                self.x0, self.MulCo, self.vel_start, GCMS=self.GCMS
            )
            
            # Don't close the global pool - it's reused throughout the app
            
            # Emit success signal
            self.finished.emit(A, B, C)
        except Exception as e:
            # Emit error signal
            self.error.emit(str(e))

class InstrumentalThread(QThread):
    """Thread for running instrumental function calculation without blocking the UI"""
    finished = Signal(dict)  # result dictionary
    error = Signal(str)
    
    def __init__(self, main_window, ref, mode, pool):
        super().__init__()
        self.main_window = main_window
        self.ref = ref
        self.mode = mode
        self.pool = pool
    
    def run(self):
        try:
            print(f"[DEBUG] InstrumentalThread started: ref={self.ref}, mode={self.mode}")
            if self.ref in (2, 3):
                # ref 2/3 = Find/Refine the THEORETICAL (simulated 57FeBO3) shape.
                result = instrumental_theory(self.main_window, ref=self.ref - 2,
                                             mode=self.mode, pool=self.pool)
            else:
                result = instrumental(self.main_window, self.ref, self.mode, pool=self.pool)
            print(f"[DEBUG] InstrumentalThread completed successfully")
            self.finished.emit(result)
        except FitInterrupted as e:
            self.error.emit(str(e))
        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] InstrumentalThread failed: {error_msg}")
            self.error.emit(error_msg)

class FittingThread(QThread):
    """Thread for running spectrum fitting without blocking the UI"""
    finished = Signal(dict)  # result dictionary
    error = Signal(str)
    
    def __init__(self, main_window, spectrum_file, pool):
        super().__init__()
        self.main_window = main_window
        self.spectrum_file = spectrum_file
        self.pool = pool
    
    def run(self):
        try:
            print(f"[DEBUG] FittingThread started for: {self.spectrum_file}")
            result = fitting_io.fit_single_spectrum(self.main_window, self.spectrum_file, self.pool)
            print(f"[DEBUG] FittingThread completed successfully")
            self.finished.emit(result)
        except FitInterrupted as e:
            self.error.emit(str(e))
        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] FittingThread failed: {error_msg}")
            self.error.emit(error_msg)


class SequentialFittingThread(QThread):
    """Thread for running sequential fitting of multiple spectra"""
    progress = Signal(int, int, str, str)  # index, total, spectrum_file, status
    spectrum_fitted = Signal(str, dict)  # spectrum_file, result (for GUI updates in main thread)
    finished = Signal(dict)  # summary dictionary
    
    def __init__(self, main_window, spectrum_files, pool, sequence_fitting_type, backgrounds):
        super().__init__()
        self.main_window = main_window
        self.spectrum_files = spectrum_files
        self.pool = pool
        self.sequence_fitting_type = sequence_fitting_type  # 0=initial, 1=result
        self.backgrounds = backgrounds  # Pre-calculated backgrounds for all spectra
    
    def run(self):
        """Simple loop through spectra, fitting each one"""
        total = len(self.spectrum_files)
        errors = []
        succeeded = 0
        
        for index, spectrum_file in enumerate(self.spectrum_files):
            try:
                # Update progress
                self.progress.emit(index, total, spectrum_file, 'fitting')
                
                # Set background for this spectrum
                background = self.backgrounds[index]
                
                # Get sequence_params from main_window (None for initial mode, parameters for result mode)
                sequence_params = getattr(self.main_window, 'sequence_params', None)
                
                # Fit this spectrum
                result = fitting_io.fit_single_spectrum(
                    self.main_window, spectrum_file, self.pool, 
                    background=background, sequence_params=sequence_params
                )
                
                if result['success']:
                    succeeded += 1
                    # Emit result for main thread to handle GUI updates and saving
                    self.spectrum_fitted.emit(spectrum_file, result)
                    self.progress.emit(index, total, spectrum_file, 'saved')
                    
                    # For 'result' mode: update sequence_params for next fit
                    if self.sequence_fitting_type == 1:
                        # Store fitted parameters for next iteration
                        self.main_window.sequence_params = result['parameters']
                else:
                    errors.append((spectrum_file, result['message']))
                    self.progress.emit(index, total, spectrum_file, 'failed')

            except FitInterrupted:
                break   # the rest of the batch is not attempted
            except Exception as e:
                error_msg = f"{e}\n{traceback.format_exc()}"
                errors.append((spectrum_file, error_msg))
                self.progress.emit(index, total, spectrum_file, 'error')
        
        # Clear temporary background storage
        self.main_window.current_spectrum_background = None  
        # Emit final summary
        failed = total - succeeded
        self.finished.emit({
            'success': failed == 0,
            'total': total,
            'succeeded': succeeded,
            'failed': failed,
            'errors': errors
        })


def _recon_weight_texts(model_list, recon_weights):
    """Serialize fitted 'Recon' weight vectors into ``{component_index: text}``,
    keyed by position in ``model_list`` (0 = baseline). Merged into the results
    table's ``expression_texts`` so the reconstructed distribution is shown in the
    weight column and "Take result as model" round-trips it back to the table.
    Recon models are matched to ``recon_weights`` in table order.
    """
    texts = {}
    if not recon_weights:
        return texts
    re = 0
    for idx, name in enumerate(model_list):
        if name == 'Recon':
            if re < len(recon_weights):
                w = np.asarray(recon_weights[re], dtype=float).ravel()
                texts[idx] = ','.join(f'{v:.6g}' for v in w)
            re += 1
    return texts


class ShowModelThread(QThread):
    """Thread for running show model calculation without blocking the UI"""
    finished = Signal(object, object, object, object, object, object, object, object, object, str, object, object, object, object)  # A, B, SPC_f, FS, FS_pos, p, model, has_nbaseline, backgrounds, instrumental_note, hires_diff, Distri_sub, Cor_sub, Recon
    error = Signal(str)
    
    def __init__(self, main_window, path_list, pool, velocity_range=15.0):
        super().__init__()
        self.main_window = main_window
        self.path_list = path_list
        self.pool = pool
        # ± x-axis range (mm/s) for the synthetic grid used in model-only mode.
        self.velocity_range = velocity_range

    def run(self):
        try:
            # Read model from parameter table
            model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(self.main_window)
            no_spectrum_mode = len(self.path_list) == 0
            
            # Validate Nbaseline count matches number of spectra
            num_nbaseline = model.count('Nbaseline')
            num_spectra = len(self.path_list)
            if not no_spectrum_mode and num_nbaseline > 0:
                if num_nbaseline != num_spectra - 1:
                    self.error.emit(f"Error: Found {num_nbaseline} Nbaseline model(s) but have {num_spectra} spectrum/spectra. "
                                  f"Need exactly {num_spectra - 1} Nbaseline(s) for {num_spectra} spectra (or 0 for single spectrum).")
                    return
            
            # Allow showing just baseline (an empty model list is OK)

            # Apply linked expressions and =[..] constraints exactly like the
            # fit does (same eval namespace: see minimi_lib._eval_expr)
            for i in range(len(NExpr)):
                try:
                    p[NExpr[i]] = mi._eval_expr(Expr[i], p)
                except Exception as e:
                    print(f"Error evaluating expression {Expr[i]}: {e}")

            # Apply constraints
            for i in range(len(con1)):
                p[int(con1[i])] = p[int(con2[i])] * con3[i]

            # Calculate backgrounds for dual y-axis support
            backgrounds = calculate_backgrounds(self.path_list, self.main_window.calibration_path) if not no_spectrum_mode else []

            # Per-spectrum sections of the model (a model without Nbaseline is
            # exactly one section)
            model_sections = fitting_io.split_model_sections(model)

            if no_spectrum_mode:
                if num_nbaseline > 0:
                    # Synthetic mode with Nbaseline sections
                    synthetic_grid = np.linspace(-self.velocity_range, self.velocity_range, 4096)
                    A_list = [synthetic_grid.copy() for _ in range(len(model_sections))]
                    A = A_list
                    B = None
                else:
                    # Single-spectrum synthetic mode
                    A = np.linspace(-self.velocity_range, self.velocity_range, 4096)
                    B = None
            elif num_nbaseline > 0:
                # Multiple spectra case - handle each spectrum section separately
                # Load all spectra separately (don't concatenate yet)
                A_list = []
                B_list = []
                for i in range(len(self.path_list)):
                    file = bliss_channel.abspath(self.path_list[i])
                    A_temp, B_temp = load_spectrum(self.main_window, [file], calibration_path=self.main_window.calibration_path)
                    if not A_temp or not B_temp:
                        self.error.emit(f"Could not load spectrum {i+1}: {file}")
                        return
                    A_list.append(A_temp[0])
                    B_list.append(B_temp[0])

                # Concatenate for the emitted full-range arrays
                A = np.concatenate(A_list)
                B = np.concatenate(B_list)
            else:
                # Single spectrum case
                file = bliss_channel.abspath(self.path_list[0])
                A_list, B_list = load_spectrum(self.main_window, [file], calibration_path=self.main_window.calibration_path)
                if not A_list or not B_list:
                    self.error.emit("Could not load spectrum")
                    return

                A = A_list[0]
                B = B_list[0]
            
            # Get experimental method parameters
            JN = int(self.main_window.jn0_input.text())
            pol = float(getattr(self.main_window, 'SMS_pol', SMS_POL_DEFAULT))  # SMS beam polarization degree

            pool = self.pool

            # Resolve instrumental parameters per spectrum section. Each section
            # may be CMS or SMS, depending on its own .dat metadata (when the
            # "use instrumental function from .dat file" option is enabled) or on
            # the UI-selected method with the internal values otherwise.
            num_sections = num_nbaseline + 1
            if no_spectrum_mode:
                section_files = [None] * num_sections
            else:
                section_files = []
                for i in range(num_sections):
                    try:
                        section_files.append(os.path.abspath(self.path_list[i]))
                    except Exception:
                        section_files.append(None)

            use_dat_metadata = bool(getattr(self.main_window, 'use_dat_instrumental_metadata', True))
            method_params_list = []
            for section_file in section_files:
                mp_section = resolve_instrumental_for_file(self.main_window, section_file, use_dat_metadata=use_dat_metadata)
                mp_section['Norm'] = compute_norm(pool, JN, mp_section)
                method_params_list.append(mp_section)

            note_lines = [mp_section['note'] for mp_section in method_params_list]
            if len({mp_section['method'] for mp_section in method_params_list}) > 1:
                note_lines.insert(0, "Mixed-method model: CMS and SMS spectra are calculated together.")
            instrumental_note = '\n'.join(note_lines)
            print(f"[Show model] {instrumental_note}")
            method_params = method_params_list[0]

            # Substitute p[i] references in Distri/Cor expressions ONCE against
            # the FULL parameter array — the same way the fit does — so the
            # per-section slices below are already numeric.
            Distri_sub, Cor_sub = list(Distri), list(Cor)
            if len(Distri) > 0 or len(Cor) > 0:
                Distri_sub, Cor_sub = fitting_io.create_subspectra(model, Distri, Cor, p)[2:4]
            distr_bounds = np.cumsum([0] + [ms.count('Distr') for ms in model_sections])
            corr_bounds = np.cumsum([0] + [ms.count('Corr') for ms in model_sections])
            recon_bounds = np.cumsum([0] + [ms.count('Recon') for ms in model_sections])

            if num_nbaseline > 0:
                # Find parameter boundaries for each section: the first section
                # starts at 0 (it includes the main baseline); each Nbaseline
                # marks the start of a new section with its own baseline
                begining_spc = [0]
                param_idx = number_of_baseline_parameters  # Start after main baseline
                for name in model:
                    if name == 'Nbaseline':
                        begining_spc.append(param_idx)
                        param_idx += number_of_baseline_parameters
                    else:
                        param_idx += mod_len_def(name, include_special=True)

                # Calculate spectrum, convergence check and subspectra for each
                # section separately (concatenated for the emit below)
                SPC_f_sections = []
                hires_diff_sections = []
                FS_all = []
                FS_pos_all = []
                p_all = []

                for spc_idx in range(len(model_sections)):
                    # Instrumental parameters resolved for this section's spectrum
                    mp_section = method_params_list[spc_idx] if spc_idx < len(method_params_list) else method_params_list[0]

                    # Get parameters for this section
                    if spc_idx < len(begining_spc) - 1:
                        p_section = p[begining_spc[spc_idx]:begining_spc[spc_idx + 1]]
                    else:
                        p_section = p[begining_spc[spc_idx]:]
                    p_all.append(p_section)

                    A_section = A_list[spc_idx]
                    d_slice = Distri_sub[distr_bounds[spc_idx]:distr_bounds[spc_idx + 1]]
                    c_slice = Cor_sub[corr_bounds[spc_idx]:corr_bounds[spc_idx + 1]]
                    r_slice = list(Recon[recon_bounds[spc_idx]:recon_bounds[spc_idx + 1]])
                    d_arg = d_slice if len(d_slice) > 0 else [0]
                    c_arg = c_slice if len(c_slice) > 0 else [0]
                    r_arg = r_slice if len(r_slice) > 0 else [0]

                    # Calculate fitted spectrum for this section
                    SPC_f_section = TI(A_section, p_section, model_sections[spc_idx], JN, pool,
                                       mp_section['x0'], mp_section['MulCo'],
                                       mp_section['INS'], d_arg, c_arg,
                                       Met=mp_section['Met'], Norm=mp_section['Norm'], pol=pol, Recon=r_arg)
                    SPC_f_sections.append(SPC_f_section)

                    # High-resolution convergence check for this section (cyan line)
                    hires_diff_sections.append(hires_model_diff(
                        pool, JN, A_section, p_section, model_sections[spc_idx],
                        mp_section, SPC_f_section, d_arg, c_arg, pol=pol, Recon=r_arg))

                    # Subspectra of this section (shared decomposition with the fit)
                    Ps, Psm, Distri_t, Cor_t, _, _ = fitting_io.create_subspectra(
                        model_sections[spc_idx], d_slice, c_slice, p_section)
                    FS, FS_pos = fitting_io.compute_component_curves(
                        A_section, Ps, Psm, Distri_t, Cor_t, JN, pool, mp_section, pol, Recon=r_slice)
                    FS_all.append(FS)
                    FS_pos_all.append(FS_pos)

                if no_spectrum_mode:
                    # Keep sectioned arrays for dedicated model-only plotting;
                    # the hires diff stays a per-section list to match.
                    self.finished.emit(A, B, SPC_f_sections, FS_all, FS_pos_all, p_all, model, True, backgrounds, instrumental_note, hires_diff_sections, Distri_sub, Cor_sub, Recon)
                else:
                    # Concatenate fitted spectrum and hires diff (plot splits them back)
                    SPC_f = np.concatenate(SPC_f_sections)
                    hires_diff = np.concatenate(hires_diff_sections) if hires_diff_sections else None
                    self.finished.emit(A, B, SPC_f, FS_all, FS_pos_all, p_all, model, True, backgrounds, instrumental_note, hires_diff, Distri_sub, Cor_sub, Recon)
            else:
                # Single spectrum case: full spectrum + convergence check
                SPC_f = TI(A, p, model, JN, pool, method_params['x0'], method_params['MulCo'],
                           method_params['INS'], Distri, Cor, Met=method_params['Met'], Norm=method_params['Norm'], pol=pol, Recon=Recon)

                hires_diff = hires_model_diff(pool, JN, A, p, model, method_params,
                                              SPC_f, Distri, Cor, pol=pol, Recon=Recon)

                # Subspectra (shared decomposition with the fit)
                Ps, Psm, Distri_t, Cor_t, _, _ = fitting_io.create_subspectra(model, Distri, Cor, p)
                FS, FS_pos = fitting_io.compute_component_curves(
                    A, Ps, Psm, Distri_t, Cor_t, JN, pool, method_params, pol, Recon=Recon)

                # Emit with single list of subspectra
                self.finished.emit(A, B, SPC_f, FS, FS_pos, p, model, False, backgrounds, instrumental_note, hires_diff, Distri_sub, Cor_sub, Recon)

        except FitInterrupted as e:
            self.error.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class RawToDatThread(QThread):
    """Thread for converting RAW spectra to .dat format"""
    finished = Signal(str)  # Success message
    error = Signal(str)     # Error message
    
    def __init__(self, main_window, file_paths, calibration_path, save_path):
        super().__init__()
        self.main_window = main_window
        self.file_paths = file_paths
        self.calibration_path = calibration_path
        self.save_path = save_path
    
    def run(self):
        try:
            metadata_lines = []
            metadata_method = None
            metadata_warning = None

            try:
                metadata_lines, metadata_method = build_dat_metadata_lines(self.main_window)
            except Exception as e:
                metadata_warning = f"Could not embed instrumental metadata: {e}. Conversion continued without metadata."

            # Determine output paths based on single/multiple files and save_path
            is_single_file = len(self.file_paths) == 1
            raw_files = [fp for fp in self.file_paths if os.path.splitext(fp)[1].lower() in ['.mca', '.cmca', '.ws5', '.w98', '.moe', '.m1', '.mcs']]
            
            if not raw_files:
                self.error.emit("No RAW files found in selection")
                return
            
            # Determine output directory and base names
            if is_single_file:
                # Single file case
                if self.save_path.endswith('/') or self.save_path.endswith('\\'):
                    # save_path is a directory
                    output_dir = self.save_path
                    base_name = os.path.splitext(os.path.basename(raw_files[0]))[0]
                else:
                    # save_path contains a filename
                    output_dir = os.path.dirname(self.save_path)
                    base_name = os.path.splitext(os.path.basename(self.save_path))[0]
                
                output_paths = [os.path.join(output_dir, f"{base_name}.dat")]
            else:
                # Multiple files case
                if self.save_path.endswith('/') or self.save_path.endswith('\\'):
                    output_dir = self.save_path
                else:
                    output_dir = os.path.dirname(self.save_path)
                
                output_paths = []
                for raw_file in raw_files:
                    base_name = os.path.splitext(os.path.basename(raw_file))[0]
                    output_paths.append(os.path.join(output_dir, f"{base_name}.dat"))
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            converted_count = 0
            error_count = 0
            
            for i, raw_file in enumerate(raw_files):
                try:
                    # Load and calibrate the spectrum
                    A_list, B_list = load_spectrum(self.main_window, [raw_file], 
                                                 calibration_path=self.calibration_path)
                    
                    if A_list and B_list and len(A_list) > 0 and len(B_list) > 0:
                        A = A_list[0]
                        B = B_list[0]
                        
                        # Save as .dat file
                        output_path = output_paths[i] if i < len(output_paths) else os.path.join(output_dir, f"{os.path.splitext(os.path.basename(raw_file))[0]}.dat")
                        
                        with open(output_path, 'w') as f:
                            if metadata_lines:
                                f.write('# Converted by SYNCMoss\n')
                                for metadata_line in metadata_lines:
                                    f.write(metadata_line + '\n')
                            for j in range(len(A)):
                                f.write(f"{A[j]}\t{B[j]}\n")
                            f.write('\n')  # Empty line at end as in original
                        
                        converted_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    print(f"Error converting {raw_file}: {e}")
                    error_count += 1
            
            # Report results
            if converted_count > 0:
                if metadata_method == 'CMS':
                    metadata_suffix = "\nEmbedded #@GCMS metadata (CMS mode)."
                elif metadata_method == 'SMS':
                    metadata_suffix = "\nEmbedded #@INSexp/#@INSint metadata (SMS mode)."
                else:
                    metadata_suffix = ''
                warning_suffix = f"\n{metadata_warning}" if metadata_warning else ''
                if error_count == 0:
                    self.finished.emit(f"Successfully converted {converted_count} RAW file(s) to .dat format{metadata_suffix}{warning_suffix}")
                else:
                    self.finished.emit(f"Converted {converted_count} file(s), {error_count} failed{metadata_suffix}{warning_suffix}")
            else:
                self.error.emit("No RAW files were successfully converted")
                
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Conversion failed: {str(e)}")


class PhysicsApp(QMainWindow):
    def __init__(self, pool=None):
        super().__init__()
        self.pool = pool  # Store reference to global pool

        # Attributes that are assigned lazily (only after the first full layout,
        # plot or fit). They are initialised here so that Qt event handlers which
        # can fire during construction (e.g. resizeEvent, triggered by
        # setGeometry below) and methods that may run before the first user
        # action can use plain attribute access instead of hasattr() guards.
        self._main_splitter = None
        self._left_panel_ideal_width = None
        self.toggle_dat_ins_action = None
        self.last_plot_data = None
        self.last_fitting_data = None
        self.models_description_window = None
        self.help_window = None
        self._fit_links_snapshot = {}
        self._fit_model_snapshot = None

        self.setWindowTitle('SYNCMoss ESRF ID14')
        self.setGeometry(50, 50, 1600, 900)
        self.setMinimumSize(1270, 710)
        self.inprogress = False  # Flag to indicate if a process is running
        # Cooperative cancellation for a running search: the model function
        # polls it (see interrupt()). An Event rather than a bool so a worker
        # THREAD reads it without a lock. Same mechanism and same name as the
        # SYNCtime branch, so the two stay portable.
        # It is models.FIT_CANCEL, so the pool calls in TI see it too.
        self.fit_cancel = FIT_CANCEL
        self._interrupting = False  # from an Interrupt click until the work stopped

        # Icon
        if getattr(sys, 'frozen', False):
            icon_dir = os.path.dirname(sys.executable)
        else:
            icon_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(icon_dir, "icons", "icon_r.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else: 
            print("Icon file not found:", icon_path)

        # Initialize variables
        self.MulCoCMS = 0.28
        self.current_spectrum_background = None  
        # When frozen by PyInstaller, __file__ is inside _internal/ but our
        # resource files (icons/, parameters/, themes) are next to the .exe.
        if getattr(sys, 'frozen', False):
            self.dir_path = os.path.dirname(sys.executable)
        else:
            self.dir_path = os.path.dirname(os.path.abspath(__file__))
        # Writable directory for parameter files, Calibration.dat and calibr.png.
        # On a frozen macOS app this redirects to ~/Library/Application Support;
        # everywhere else it is the bundled parameters/ folder next to the app.
        self.params_dir = _resolve_params_dir(self.dir_path)
        # The model Library is written to as well ("Save to library", "Import
        # Library"), so it gets the same redirect (<AppData>/Library there).
        self.library_dir = _resolve_library_dir(self.dir_path)
        self.workfolder = None  # Start with no workfolder selected
        self.workfolder_check = 1
        self.check_points_match = False
        self.newfilename = str('')
        self.newfilename2 = str('')
        self.calibration_path = os.path.join(self.params_dir, "Calibration.dat")
        self.points_match = True
        self.path_list = []
        self.backgrounds = []  # List to store calculated backgrounds
        self.sequence_fitting_type = 0  # 0 = initial, 1 = result
        self.use_dat_instrumental_metadata = True
        # How "Find/Refine Instr. func." approximates the source, and which
        # stored shape every fit reads: 'gauss' the empirical sum of Gaussians,
        # 'theory' the simulated 57FeBO3 source. Set from Supp -> "Choose how to
        # approximate instrumental function". It does NOT override a spectrum
        # that carries its own instrumental function in the .dat file -- that is
        # the separate "use instrumental function from .dat file" toggle; when a
        # file carries BOTH descriptions this picks between them.
        self.instrumental_method = DEFAULT_INSTRUMENTAL_METHOD
        # Set once the user ticks "do not ask again" on the mixed/different
        # instrumental-function warning (per-session, resets on restart).
        self.suppress_mixed_metadata_warning = False
        self.x0 = 0.0
        self.MulCo = 0.0

        # Detect OS dark/light mode and load matching theme
        self._is_dark_mode = self._detect_os_dark_mode()
        self._load_theme()  # Sets self.gridcolor, self.BGcolor, and self._theme

        # Model colors from constants
        # Need to extend the list to fill more lines in the table
        extended_model_colors = model_colors.copy()
        for i in range(0, 10):
            extended_model_colors.extend(model_colors)
        self.model_colors = extended_model_colors

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter for left and right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel (now first)
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)

        # Top controls
        top_controls = QHBoxLayout()

        self.loadmod_btn = QPushButton("Load\nmodel")
        self.loadmod_btn.setFont(QFont('Arial', 16))
        self.loadmod_btn.clicked.connect(self.loadmod_pressed)

        self.btncleanmodel = QPushButton("Clean\nmodel\n(2 clk)")
        self.btncleanmodel.setFont(QFont('Arial', 16))
        self.btncleanmodel.setStyleSheet("background-color: rgb(255, 191, 0); color: black;")
        # Override mouseDoubleClickEvent for double-click detection
        self.btncleanmodel.mouseDoubleClickEvent = lambda event: self.clean_model()

        # self.switch = QCheckBox()  # NFS switch removed - only calibration supported
        # self.switch.setChecked(True)

        self.cal_cho_btn = QPushButton("Choose\ncalibration\nfile")
        self.cal_cho_btn.setFont(QFont('Arial', 16))
        self.cal_cho_btn.clicked.connect(self.choose_calibration_file)

        # Calibration group
        cal_frame = QFrame()
        cal_layout = QHBoxLayout(cal_frame)
        cal_layout.setContentsMargins(5, 5, 5, 5)

        self.cal_cho_title = QLabel("Velocity\ndown-up:")
        self.cal_cho_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cal_cho_title.setFont(QFont('Arial', 16))

        self.velocity_btn = QPushButton()
        self.velocity_btn.setFlat(True)
        self.velocity_btn.setIconSize(QSize(32, 32))
        self.velocity_btn.clicked.connect(self.toggle_velocity_direction)
        self.velocity_direction = False  # False for down-up, True for up-down
        self.update_velocity_label()  # Set initial state

        cal_layout.addWidget(self.cal_cho_title)
        cal_layout.addWidget(self.velocity_btn)

        self.cal_btn = QPushButton("Calibrate")
        self.cal_btn.setFont(QFont('Arial', 16))
        self.cal_btn.clicked.connect(self.calibration)

        self.vel_btn = QPushButton("RAW\nto\ndat")
        self.vel_btn.setFont(QFont('Arial', 16))
        self.vel_btn.clicked.connect(self.raw_to_dat)

        self.interrupt_btn = QPushButton('! INTERRUPT !')
        self.interrupt_btn.setFont(QFont('Arial', 16))
        self.interrupt_btn.setStyleSheet("background-color: red; color: white;")
        self.interrupt_btn.clicked.connect(self.interrupt)

        # The Supp menu (support tools & settings dialogs) lives in supp_menu.py.
        # It also owns the light/dark toggle (self.theme_action), which is why it
        # must be built after _load_theme() set self._is_dark_mode.
        self.supp_btn = QPushButton('Supp')
        self.supp_btn.setFont(QFont('Arial', 16))
        self.supp_menu = build_supp_menu(self)
        self.supp_btn.setMenu(self.supp_menu)

        # Add all buttons to top controls
        top_buttons = [self.loadmod_btn, self.btncleanmodel, self.cal_cho_btn,
                   self.cal_btn, self.vel_btn, self.interrupt_btn, self.supp_btn]
        for btn in top_buttons:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            top_controls.addWidget(btn)
        # cal_frame also needs expanding policy
        cal_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_controls.insertWidget(3, cal_frame)  # Insert after cal_cho_btn

        left_layout.addLayout(top_controls)

        # Parameters table
        self.params_table = ParametersTable(self)
        scroll_params = QScrollArea()
        scroll_params.setWidget(self.params_table)
        scroll_params.setWidgetResizable(False)
        scroll_params.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        left_layout.addWidget(scroll_params)

        # Bottom controls
        bottom_layout = QHBoxLayout()

        self.play_btn = QPushButton("Fit\n(F5)")
        self.play_btn.setFont(QFont('Arial', 21))
        self.play_btn.setStyleSheet("background-color: rgb(0, 200, 0); color: black;")
        self.play_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.play_btn.clicked.connect(self.fit_pressed)

        # Fit options
        fit_options_widget = QWidget()
        fit_options_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        fit_group = QGridLayout(fit_options_widget)
        self.MS_fit = QCheckBox("CMS")
        self.SMS_fit = QCheckBox("SMS")
        self.SMS_fit.setChecked(True)
        self.APS_fit = QCheckBox("APS")
        self.APS_fit.setEnabled(False)
        self.APS_fit.setToolTip("It will become available soon once APS stabilizes this technique and makes it available to users.")
        
        # Connect MS and SMS to be mutually exclusive
        self.MS_fit.stateChanged.connect(lambda: self.on_ms_sms_changed(self.MS_fit))
        self.SMS_fit.stateChanged.connect(lambda: self.on_ms_sms_changed(self.SMS_fit))

        fit_par_layout = QHBoxLayout()
        g_label = QLabel("G")
        self.GCMS_input = QLineEdit(str(np.genfromtxt(os.path.join(self.params_dir, 'GCMS.txt'), delimiter='\t')))
        fit_par_layout.addWidget(g_label)
        fit_par_layout.addWidget(self.GCMS_input)

        # Two-row layout with aligned right column:
        # row 1 -> CMS | SMS
        # row 2 -> G   | APS
        fit_group.addWidget(self.MS_fit, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        fit_group.addWidget(self.SMS_fit, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        fit_group.addLayout(fit_par_layout, 1, 0)
        fit_group.addWidget(self.APS_fit, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        fit_group.setColumnStretch(0, 1)
        fit_group.setColumnStretch(1, 0)

        # These settings are still used by the existing fitting logic; they are
        # edited via small dialogs opened from the Supp menu.
        # Points for the full transmission integral. SMS and CMS need different
        # numbers, so each has its own store; ``jn0_input`` holds whichever the
        # current mode selects and is what the rest of the code reads.
        # ``refresh_jn0_for_mode`` keeps it in step.
        self.jn0_sms_input = QLineEdit("32", self)
        self.jn0_sms_input.setValidator(QIntValidator(1, 1000000, self))
        self.jn0_sms_input.hide()
        self.jn0_cms_input = QLineEdit("64", self)
        self.jn0_cms_input.setValidator(QIntValidator(1, 1000000, self))
        self.jn0_cms_input.hide()
        self.jn0_input = QLineEdit("32", self)
        self.jn0_input.setValidator(QIntValidator(1, 1000000, self))
        self.jn0_input.hide()
        self.refresh_jn0_for_mode()

        self.instrumental_number = QLineEdit("3", self)
        self.instrumental_number.setValidator(QIntValidator(1, 1000, self))
        self.instrumental_number.hide()

        # SMS beam linear polarization degree (0..1). Mirrors jn0_input above:
        # a hidden store edited via the "Set polarization" dialog in the Supp
        # menu. Its value is mirrored into self.SMS_pol and passed to TI as the
        # ``pol`` argument, which forwards it into the transmission-integral
        # workers. Default from constants.SMS_POL_DEFAULT.
        self.polarization_input = QLineEdit(str(SMS_POL_DEFAULT), self)
        pol_validator = QDoubleValidator(0.0, 1.0, 6, self)
        pol_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        pol_validator.setLocale(QLocale(QLocale.Language.C))
        self.polarization_input.setValidator(pol_validator)
        self.polarization_input.hide()
        self.SMS_pol = SMS_POL_DEFAULT

        # File chooser
        file_layout = QVBoxLayout()
        file_choose_layout = QHBoxLayout()
        self.btnchoose = QPushButton("Choose\nspectrum")
        self.btnchoose.setFont(QFont('Arial', 18))
        self.btnchoose.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btnchoose.clicked.connect(self.choose_file)
        self.process_path = QTextEdit(f"['{self.calibration_path}']")
        self.process_path.setFont(QFont('Arial', 14))
        self.process_path.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.process_path.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.process_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Allow editing of file paths
        file_choose_layout.addWidget(self.process_path)
        file_layout.addLayout(file_choose_layout)

        # Second row: choose-spectrum button on the left, compact fit options on the right.
        file_controls_layout = QHBoxLayout()
        file_controls_layout.addWidget(self.btnchoose, 1)
        file_controls_layout.addWidget(fit_options_widget, 0)
        file_layout.addLayout(file_controls_layout)

        bottom_layout.addWidget(self.play_btn)
        bottom_layout.addLayout(file_layout)

        # Show buttons and INS
        show_layout = QVBoxLayout()
        show_buttons = QHBoxLayout()
        self.show_btn = QPushButton("Show spectrum")
        self.show_btn.setFont(QFont('Arial', 18))
        self.show_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.show_btn.clicked.connect(self.show_pressed)

        self.showM_btn = QPushButton("Show model")
        self.showM_btn.setFont(QFont('Arial', 18))
        self.showM_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.showM_btn.clicked.connect(self.showM_pressed)

        show_buttons.addWidget(self.show_btn)
        show_buttons.addWidget(self.showM_btn)

        instrumental_layout = QHBoxLayout()
        self.instrumental_btn = QPushButton("Instrumental\nfunction")
        self.instrumental_btn.setFont(QFont('Arial', 16))
        self.instrumental_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Create dropdown menu for INS button
        self.instrumental_menu = QMenu()
        # WHICH search these run -- the theoretical simulated 57FeBO3 shape or
        # the empirical sum of Gaussians -- is a setting, not a separate set of
        # menu entries: Supp -> "Choose how to approximate instrumental
        # function". self.instrumental_ref() turns it into the ref code.
        find_single = QAction("Find\nInstr. func.\n single line", self)
        find_single.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(False), 0))
        find_aFe = QAction("Find\nInstr. func.\n pure a-Fe", self)
        find_aFe.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(False), 2))
        find_model = QAction("Find\nInstr. func.\nmodel", self)
        find_model.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(False), 1))
        self.toggle_dat_ins_action = QAction(self)
        self._update_use_dat_instrumental_action_text()
        self.toggle_dat_ins_action.triggered.connect(self.toggle_use_dat_instrumental)
        reset_defaults = QAction("Reset to\ndefault values", self)
        reset_defaults.triggered.connect(lambda: reset_instrumental_defaults(self))
        self.instrumental_menu.addAction(find_single)
        self.instrumental_menu.addAction(find_aFe)
        self.instrumental_menu.addAction(find_model)
        self.instrumental_menu.addSeparator()
        self.instrumental_menu.addAction(self.toggle_dat_ins_action)
        self.instrumental_menu.addSeparator()
        self.instrumental_menu.addAction(reset_defaults)
        self.instrumental_btn.setMenu(self.instrumental_menu)

        self.instrumental_btn2 = QPushButton("Refine\nInstr. func.")
        self.instrumental_btn2.setFont(QFont('Arial', 15))
        self.instrumental_btn2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Create dropdown menu for INS refine button
        self.instrumental_menu2 = QMenu()
        refine_single = QAction("Refine\nInstr. func.\n single line", self)
        refine_single.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(True), 0))
        refine_aFe = QAction("Refine\nInstr. func.\n pure a-Fe", self)
        refine_aFe.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(True), 2))
        refine_model = QAction("Refine\nInstr. func.\nmodel", self)
        refine_model.triggered.connect(
            lambda: self.instrumental_pressed(self.instrumental_ref(True), 1))
        self.instrumental_menu2.addAction(refine_single)
        self.instrumental_menu2.addAction(refine_aFe)
        self.instrumental_menu2.addAction(refine_model)
        self.instrumental_btn2.setMenu(self.instrumental_menu2)

        instrumental_layout.addWidget(self.instrumental_btn)
        instrumental_layout.addWidget(self.instrumental_btn2)

        show_layout.addLayout(show_buttons)
        show_layout.addLayout(instrumental_layout)

        bottom_layout.addLayout(show_layout)

        left_layout.addLayout(bottom_layout)

        # Second row: spectrum options and sequence fitting
        seq_fit_layout = QHBoxLayout()
        
        # Change spectrum dropdown button
        self.change_spectrum_btn = QPushButton("Change\nspectrum(a)")
        self.change_spectrum_btn.setFont(QFont('Arial', 16))
        self.change_spectrum_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.change_spectrum_btn.clicked.connect(self.show_spectrum_options)
        
        # Choose workfolder button
        self.choose_workfolder_btn = QPushButton("Choose\nworkfolder")
        self.choose_workfolder_btn.setFont(QFont('Arial', 16))
        self.choose_workfolder_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.choose_workfolder_btn.clicked.connect(self.choose_workfolder)
        
        # Sequence fitting button
        self.sequence_fitting_type = 0  # 0 = initial, 1 = result
        self.seq_fit_btn = QPushButton("Sequence Fitting\n(initial)")
        self.seq_fit_btn.setFont(QFont('Arial', 16))
        self.seq_fit_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.seq_fit_btn.clicked.connect(self.show_sequence_fitting_options)
        
        seq_fit_layout.addWidget(self.change_spectrum_btn)
        seq_fit_layout.addWidget(self.choose_workfolder_btn)
        seq_fit_layout.addWidget(self.seq_fit_btn)
        
        left_layout.addLayout(seq_fit_layout)

        # Save buttons with save path between
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save\nresult")
        self.save_btn.setFont(QFont('Arial', 18))
        self.save_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # self.save_btn.clicked.connect(self.save_pressed)

        self.saveas_btn = QPushButton("Save\nresult as")
        self.saveas_btn.setFont(QFont('Arial', 18))
        self.saveas_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # self.saveas_btn.clicked.connect(self.save_as_pressed)

        # Editable save path field
        self.save_path = QLineEdit("")
        self.save_path.setFont(QFont('Arial', 14))
        self.save_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.save_path.setPlaceholderText("Save path")

        self.save_model_btn = QPushButton("Save\nmodel")
        self.save_model_btn.setFont(QFont('Arial', 18))
        self.save_model_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.save_model_btn.clicked.connect(self.save_model_pressed)

        self.save_model_as_btn = QPushButton("Save\nmodel as")
        self.save_model_as_btn.setFont(QFont('Arial', 18))
        self.save_model_as_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.save_model_as_btn.clicked.connect(self.save_model_as_pressed)

        self.save_to_library_btn = QPushButton("Save to\nlibrary")
        self.save_to_library_btn.setFont(QFont('Arial', 18))
        self.save_to_library_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.save_to_library_btn.clicked.connect(self.save_to_library_pressed)

        self.save_btn.clicked.connect(self.save_result_pressed)
        self.saveas_btn.clicked.connect(self.save_result_as_pressed)
        
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.saveas_btn)
        save_layout.addWidget(self.save_path, 1)  # Stretch to fill space
        save_layout.addWidget(self.save_model_btn)
        save_layout.addWidget(self.save_model_as_btn)
        save_layout.addWidget(self.save_to_library_btn)

        left_layout.addLayout(save_layout)

        splitter.addWidget(self.left_panel)

        # Right panel (now second)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)

        # Top part: spectrum plot and controls
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Spectrum plot
        self.figure = plt.Figure(figsize=(6, 4), dpi=100, tight_layout=True)
        self.figure.patch.set_facecolor('black')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        
        # Store position markers data and artists
        self.current_FS_pos = None
        self.position_artists = []  # Store line artists for toggling
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        top_layout.addLayout(plot_layout)

        # Connect to scroll event for zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)

        # Top controls for image
        image_controls = QHBoxLayout()

        title = QLabel("Result")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont('Arial', 24))
        title.setStyleSheet("color: red;")

        self.SP_DI = QPushButton('Distribution')
        self.SP_DI.setFont(QFont('Arial', 18))
        self.SP_DI.clicked.connect(self.toggle_distribution)
        self._showing_distribution = False  # Track current view state

        image_controls.addWidget(title, 1)
        image_controls.addWidget(self.SP_DI)
        top_layout.addLayout(image_controls)

        # Bottom part: results table and log
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        # Results table with tabs
        self.results_table = ResultsTable(self)
        bottom_layout.addWidget(self.results_table)

        # Log and take result
        log_layout = QHBoxLayout()
        self.log = QTextEdit()
        self.log.setFont(QFont('Arial', 14))
        self.log.setMaximumHeight(100)
        self.log.setReadOnly(True)  # Status field, read-only
        self.set_status("Ready")  # Initial status

        self.take_result_btn = QPushButton('Take result as model (F8)')
        self.take_result_btn.setFont(QFont('Arial', 18))
        self.take_result_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.take_result_btn.clicked.connect(self.take_result)

        log_layout.addWidget(self.log)
        log_layout.addWidget(self.take_result_btn)
        bottom_layout.addLayout(log_layout)

        # Vertical splitter for right panel
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(top_widget)
        right_splitter.addWidget(bottom_widget)
        right_splitter.setSizes([400, 300])  # Initial sizes

        right_layout.addWidget(right_splitter)

        splitter.addWidget(self.right_panel)
        self._main_splitter = splitter  # Store reference for resizeEvent
        # Calculate ideal left panel width from actual params table size + scroll area overhead + layout margins
        margins = left_layout.contentsMargins()
        self._left_panel_ideal_width = self.params_table.sizeHint().width() + 25 + margins.left() + margins.right()
        splitter.setSizes([self._left_panel_ideal_width, 500])

        # Load and plot default spectrum
        self.plot_default_spectrum()

        # Apply initial theme (matplotlib + Qt color scheme)
        self._force_color_scheme(self._is_dark_mode)
        self._apply_theme()

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_F5), self).activated.connect(self.fit_pressed)
        QShortcut(QKeySequence(Qt.Key.Key_F8), self).activated.connect(self.take_result)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self).activated.connect(self.showM_pressed)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self).activated.connect(self.showM_pressed)

    def set_status(self, message, color=None):
        """Show *message* in the status/log box, optionally re-coloring it.

        Single replacement for the setPlainText/setStyleSheet pair repeated
        throughout the app. Color convention: green = success, red = error,
        orange = warning/canceled, blue = setting changed / info,
        cyan = long-running action started. ``color=None`` keeps the current
        color (matches the historical bare setPlainText call sites).
        """
        self.log.setPlainText(message)
        if color:
            self.log.setStyleSheet(f"color: {color};")

    def plot_default_spectrum(self):
        """Load and plot the default spectrum from Calibration.dat"""
        try:
            A_list, B_list = load_spectrum(self, [self.calibration_path], calibration_path=self.calibration_path)
            if A_list and B_list:
                # Calculate background for calibration
                backgrounds = calculate_backgrounds([self.calibration_path], self.calibration_path)
                plot_spectrum(self.figure, A_list, B_list, ["Calibration.dat"], backgrounds=backgrounds, theme=self._theme)
                self._update_legend_toggle()
                self.toolbar.push_current()  # Set current view as home
        except Exception as e:
            self.set_status(f"Could not load default spectrum: {e}", "red")

    def toggle_position_markers(self, show):
        """Toggle visibility of position marker artists."""
        for artist in self.position_artists:
            artist.set_visible(show)
        self.canvas.draw()

    def toggle_legend_visibility(self, show):
        """Toggle visibility of legend on all axes."""
        for ax in self.figure.get_axes():
            legend = ax.get_legend()
            if legend:
                legend.set_visible(show)
        self.canvas.draw()

    def _update_legend_toggle(self):
        """Enable/disable the legend toggle button based on whether legends exist.
        Resets to unchecked (hidden) for every new plot."""
        has_legend = any(
            ax.get_legend() is not None for ax in self.figure.get_axes()
        )
        self.toolbar.toggle_legend_action.setEnabled(has_legend)
        self.toolbar.toggle_legend_action.setChecked(False)
    
    def resizeEvent(self, event):
        """Keep left panel width matched to params table on resize/maximize"""
        super().resizeEvent(event)
        if self._main_splitter is not None and self._left_panel_ideal_width is not None:
            total = self._main_splitter.width()
            left_w = min(self._left_panel_ideal_width, int(total * 0.75))
            right_w = total - left_w
            self._main_splitter.setSizes([left_w, right_w])

    def on_scroll_zoom(self, event):
        """Zoom in/out on scroll, centered on mouse position"""
        if event.inaxes is None:
            return
        ax = event.inaxes
        # Zoom factor (reversed: scroll up zooms out, scroll down zooms in)
        zoom_factor = 1/1.1 if event.button == 'up' else 1.1

        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Mouse position in data coordinates
        x_mouse = event.xdata
        y_mouse = event.ydata

        # New ranges
        x_range = (x_max - x_min) * zoom_factor
        y_range = (y_max - y_min) * zoom_factor

        # New limits centered on mouse
        x_new_min = x_mouse - (x_mouse - x_min) * zoom_factor
        x_new_max = x_mouse + (x_max - x_mouse) * zoom_factor
        y_new_min = y_mouse - (y_mouse - y_min) * zoom_factor
        y_new_max = y_mouse + (y_max - y_mouse) * zoom_factor

        ax.set_xlim(x_new_min, x_new_max)
        ax.set_ylim(y_new_min, y_new_max)
        self.canvas.draw()

    def on_ms_sms_changed(self, changed_checkbox):
        """Handle MS/SMS checkbox mutual exclusivity"""
        if changed_checkbox == self.MS_fit and self.MS_fit.isChecked():  # MS checked
            self.SMS_fit.setChecked(False)  # Uncheck SMS
        elif changed_checkbox == self.MS_fit:  # MS checked
            self.SMS_fit.setChecked(True)  # Check SMS
        elif changed_checkbox == self.SMS_fit and self.SMS_fit.isChecked():  # SMS checked
            self.MS_fit.setChecked(False)  # Uncheck MS
        elif changed_checkbox == self.SMS_fit:  # SMS unchecked
            self.MS_fit.setChecked(True)  # Check MS
        self.refresh_jn0_for_mode()

    def refresh_jn0_for_mode(self):
        """Point ``jn0_input`` at the SMS or the CMS setting, whichever applies.

        SMS and CMS need different numbers of integration points and now each
        has its own stored value (Supp -> "Set number of points for full
        transmission integral"). This used to be a switch that doubled 32 to 64
        on the way into CMS and halved it back -- and only when the value was
        still exactly one of those two, so any edited value stopped tracking the
        mode at all. The two values are simply settable now.

        Everything downstream keeps reading ``jn0_input``, which holds the
        ACTIVE one.
        """
        source = (self.jn0_cms_input if self.MS_fit.isChecked()
                  else self.jn0_sms_input)
        self.jn0_input.setText(source.text().strip())

    def _update_use_dat_instrumental_action_text(self):
        if self.toggle_dat_ins_action is None:
            return
        if self.use_dat_instrumental_metadata:
            self.toggle_dat_ins_action.setText("do not use instrumental function from .dat file (now it is used)")
        else:
            self.toggle_dat_ins_action.setText("use instrumental function from .dat file (now it is not used)")

    def toggle_use_dat_instrumental(self):
        self.use_dat_instrumental_metadata = not self.use_dat_instrumental_metadata
        self._update_use_dat_instrumental_action_text()
        if self.use_dat_instrumental_metadata:
            self.set_status("Using instrumental function from .dat file is now enabled (SMS).", "blue")
        else:
            self.set_status("Using instrumental function from .dat file is now disabled (SMS).", "blue")

    def loadmod_pressed(self):
        load_model(self)

    def save_model_pressed(self):
        save_model(self)

    def save_model_as_pressed(self):
        save_model_as(self)

    def save_to_library_pressed(self):
        save_to_library_via_dialog(self)

    def update_velocity_label(self):
        """Update the velocity label and button icon based on velocity direction"""
        if self.velocity_direction:
            self.cal_cho_title.setText("Velocity\nup-down:")
            icon_path = os.path.join(self.dir_path, "icons", "UD.png")
        else:
            self.cal_cho_title.setText("Velocity\ndown-up:")
            icon_path = os.path.join(self.dir_path, "icons", "DU.png")
        
        if os.path.exists(icon_path):
            self.velocity_btn.setIcon(QIcon(icon_path))
        else:
            print(f"Icon file not found: {icon_path}")

    def toggle_velocity_direction(self):
        """Toggle the velocity direction"""
        self.velocity_direction = not self.velocity_direction
        self.update_velocity_label()

    def clean_model(self):
        """Clean all model parameters except baseline"""
        try:
            # Clear all rows except row 0 (baseline)
            # clear_row_params now handles resetting model button to "None"
            for row in range(1, len(self.params_table.row_widgets)):
                self.params_table.clear_row_params(row)
            self.set_status("Model cleaned (baseline preserved)", "green")
        except Exception as e:
            self.set_status(f"Error cleaning model: {e}", "red")

    def take_result(self):
        """Copy fitting results to parameter table as new model"""
        try:
            # Check if results are available
            if not hasattr(self.results_table, 'current_model_list') or not self.results_table.current_model_list:
                self.set_status("No fitting results available", "orange")
                return
            
            # Get data from results table
            model_list = self.results_table.current_model_list
            model_colors = self.results_table.current_model_colors
            parameter_names = self.results_table.current_parameter_names
            parameters = self.results_table.current_parameters
            errors = self.results_table.current_errors
            expression_texts = getattr(self.results_table, 'expression_texts', {})
            result_links = getattr(self.results_table, 'current_links', {})

            if self.params_table.get_model_list() != model_list:
                msg = QMessageBox(self)
                msg.setWindowTitle("Take Result As Model")
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setText(
                    "Result model is different from current one. Current model will "
                    "be overwritten. Continue?"
                )
                continue_btn = msg.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                if msg.clickedButton() is not continue_btn:
                    self.set_status("Take result cancelled", "orange")
                    return

            # Rebuild table structure from result model.
            for row in range(1, len(self.params_table.row_widgets)):
                self.params_table.clear_row_params(row)

            for model_idx, model_name in enumerate(model_list):
                if model_name != 'baseline':
                    self.params_table.select_model(model_idx, model_name)
                    color = model_colors[model_idx] if model_idx < len(model_colors) else 'blue'
                    self.params_table.select_color(model_idx, color)

            # Copy parameter values and restore links captured at fit time when
            # available.
            # Expression texts are written in a second pass after all structure
            # edits to prevent any internal index-shifting from touching them.
            param_index = 0
            pending_expression_updates = []  # (model_idx, col, text)
            for model_idx, model_name in enumerate(model_list):
                
                # Get number of parameters for this model
                num_params = len(parameter_names[model_idx]) if model_idx < len(parameter_names) else 0
                
                # Copy parameter values and fix checkboxes
                for col in range(num_params):
                    if param_index >= len(parameters):
                        break
                    
                    # Get widgets (positional layout contract documented in
                    # model_io's module docstring)
                    row_widget = self.params_table.row_widgets[model_idx]
                    param_widget = row_widget.layout().itemAt(col + 1).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    top_layout = param_widget.layout().itemAt(0).layout()
                    fix_cb = top_layout.itemAt(1).widget()
                    
                    # Expression / weight-vector placeholders carry no numeric value.
                    if model_name in ['Distr', 'Corr', 'Recon', 'Expression'] and col == num_params - 1:
                        expr_text = expression_texts.get(model_idx)
                        if expr_text is not None:
                            pending_expression_updates.append((model_idx, col, expr_text))
                        param_index += 1
                        continue

                    # Restore links captured at fit time, if available and model
                    # layout index is present in the saved snapshot.
                    if param_index in result_links:
                        value_input.setText(result_links[param_index])
                        param_index += 1
                        continue
                    
                    # Set parameter value
                    value = parameters[param_index]
                    value_input.setText(f"{value:.6g}")
                    
                    # Check fix checkbox if error is nan
                    if errors is not None and param_index < len(errors):
                        error = errors[param_index]
                        if np.isnan(error):
                            fix_cb.setChecked(True)
                        else:
                            fix_cb.setChecked(False)
                    
                    param_index += 1

            # Write equations last, after any model insert/delete operations,
            # so p[...] indices stay exactly as they are in the result table.
            for model_idx, col, expr_text in pending_expression_updates:
                row_widget = self.params_table.row_widgets[model_idx]
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                value_input.setText(expr_text)

            # A 'par' restored to the value auto_fill_params already put there
            # emits no textChanged, so re-draw the grey frames once.
            self.params_table.update_distr_corr_highlights()
            self.set_status("Result copied to model", "green")
            
        except Exception as e:
            self.set_status(f"Error copying result: {e}", "red")
            traceback.print_exc()

    def choose_calibration_file(self):
        """Choose a calibration file (.dat/.txt/.exp)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose calibration file",
            self.workfolder,
            "Calibration files (*.dat *.txt *.exp);;All files (*.*)"
        )
        if file_path:
            self.calibration_path = file_path
            self.set_status(f"Calibration file set: {os.path.basename(file_path)}", "green")
            # Optionally replot the default spectrum with new calibration
            self.plot_default_spectrum()
        else:
            self.set_status("Calibration selection canceled", "orange")

    def show_spectrum_options(self):
        """Show dropdown menu for spectrum change options"""
        menu = QMenu(self)
        
        # Create actions for each option
        options = [
            ("Sum all\nspectra", "sum_all"),
            ("Subtract\nmodel from\nspectrum", "subtract_model"), 
            ("Half points", "half_points")
        ]
        
        for text, action_name in options:
            action = QAction(text.replace('\n', ' '), self)
            action.triggered.connect(lambda checked, name=action_name: self.on_spectrum_option_selected(name))
            menu.addAction(action)
        
        # Show menu below the button
        menu.exec(self.change_spectrum_btn.mapToGlobal(self.change_spectrum_btn.rect().bottomLeft()))

    def on_spectrum_option_selected(self, option):
        """Handle spectrum option selection"""
        if option == "sum_all":
            sum_all_spectra(self)
        elif option == "subtract_model":
            subtract_model_from_spectrum(self)
        elif option == "half_points":
            half_points(self)
        else:
            self.set_status(f"Unknown spectrum option: {option}", "red")

    def check_user_expressions(self, action_label):
        """Validate the model before starting a fit or a show-model run.

        Three checks run up front:

        * no active numeric parameter slot may be empty or hold a half-written
          link (a =[X,y] reference to a deleted parameter leaves its field empty,
          and '=[,1]' is a link the user has not finished typing; read_model
          would silently read either as 0.0),
        * no two Distr/Corr/Recon rows of one chain may share a 'par' (they would
          overwrite each other in the same parameter slot), and
        * every user-typed Expression/Distr/Corr text must evaluate.

        On failure the offending table fields turn red (they recover as soon as
        the user clicks into them), the log box explains each problem, and False
        is returned so the caller can abort before any thread starts.
        """
        empty_slots = self.params_table.get_empty_parameter_slots()
        if empty_slots:
            lines = []
            for slot in empty_slots:
                self.params_table.mark_parameter_error(slot['row'], slot['col'])
                param = slot['param'] or f"column {slot['col']}"
                if slot['reason'] == 'empty':
                    what = "is empty"
                else:
                    what = (f"holds an unfinished link '{slot['text']}' — a link "
                            f"needs both numbers, =[parameter,factor]")
                lines.append(f"{slot['model']} (table row {slot['row']}): "
                             f"parameter '{param}' {what}")
            self.set_status(
                f"{action_label} was not started — empty or unfinished "
                f"parameter(s) (fill them in; a value referenced by =[...] may "
                f"have been deleted):\n" + "\n".join(lines),
                "red",
            )
            return False

        conflicts = self.params_table.get_conflicting_distr_targets()
        if conflicts:
            lines = []
            for clash in conflicts:
                # The first claimant may well be the one the user meant, so only
                # the later duplicates are marked: fixing them clears the whole
                # error, with no leftover red field to click just to clean up.
                for clash_row in clash['rows'][1:]:
                    self.params_table.mark_parameter_error(clash_row, 0)
                named = [f"{name} (table row {clash_row})" for name, clash_row
                         in zip(clash['models'], clash['rows'])]
                who = " and ".join([", ".join(named[:-1]), named[-1]])
                target = f"'{clash['param']}'" if clash['param'] else f"number {clash['par']}"
                base = (f"{clash['base_model']} (table row {clash['base_row']})"
                        if clash['base_row'] is not None else "the component above")
                lines.append(f"{who} all use par = {clash['par']} "
                             f"→ parameter {target} of {base}")
            self.set_status(
                f"{action_label} was not started — Distr/Corr/Recon rows applied to "
                f"the same component must each take a DIFFERENT 'par' (they write "
                f"into the same parameter slot, so only the last one would "
                f"survive):\n" + "\n".join(lines),
                "red",
            )
            return False

        try:
            problems = validate_user_expressions(self)
        except Exception as e:
            self.set_status(f"{action_label} was not started — could not read the model: {e}", "red")
            return False

        if not problems:
            return True

        lines = []
        for prob in problems:
            if prob.get('row') is not None:
                self.params_table.mark_expression_error(prob['row'])
                label = f"{prob['kind']} (table row {prob['row']})"
            else:
                label = prob['kind']
            lines.append(f"{label}: '{prob['text']}' could not be evaluated: {prob['error']}")
        self.set_status(
            f"{action_label} was not started — invalid expression(s):\n" + "\n".join(lines),
            "red",
        )
        return False

    def confirm_instrumental_methods(self, spectrum_files, mode):
        """Warn the user, before a fit starts, about two instrumental-method
        situations. Returns True to proceed, False to abort.

        * Method override (any mode): when "use instrumental function from .dat
          file" is on and a spectrum's metadata selects a different method than
          the CMS/SMS checkbox, that file's method wins — tell the user how to
          change it. Always shown (independent of the warning below).
        * Different instrumental functions (multi-spectrum): when the spectra do
          not all share one instrumental function (a CMS+SMS mix, or the same
          method with different values), confirm the user means it. Carries a
          "do not ask again" tick that suppresses it for the rest of the session.

        When .dat metadata is disabled every spectrum resolves to the same
        internal method, so neither warning fires and a heterogeneous selection
        is fitted uniformly.
        """
        use_dat = bool(getattr(self, 'use_dat_instrumental_metadata', True))
        ui_method = 'CMS' if self.MS_fit.isChecked() else 'SMS'
        overridden, nonuniform, resolved = analyze_instrumental_methods(self, spectrum_files, use_dat)

        if overridden and not self._warn_method_override(overridden, ui_method):
            return False

        if (nonuniform and mode != 'single'
                and not getattr(self, 'suppress_mixed_metadata_warning', False)):
            proceed, dont_ask = self._warn_mixed_metadata(resolved, mode)
            if dont_ask:
                self.suppress_mixed_metadata_warning = True
            if not proceed:
                return False

        return True

    def _warn_method_override(self, overridden, ui_method):
        """Popup: the .dat metadata overrides the selected CMS/SMS method.
        Returns True to proceed, False to abort."""
        lines = []
        for spectrum_file, resolved in overridden:
            tag = '#@GCMS' if resolved['method'] == 'CMS' else '#@INSexp/#@INSint'
            lines.append(f"• {os.path.basename(spectrum_file)} → {resolved['method']} mode (file contains {tag})")
        message = (
            f"{ui_method} mode is selected, but the .dat metadata of the following "
            f"spectrum(a) selects a different method, which will be used instead:\n\n"
            + "\n".join(lines)
            + f"\n\nTo fit in {ui_method} mode, either disable \"use instrumental "
              f"function from .dat file\" in the Instrumental function menu, or remove "
              f"the metadata from the file(s).\n\nProceed anyway?"
        )
        reply = QMessageBox.warning(
            self, "Instrumental method from .dat file", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _warn_mixed_metadata(self, resolved, mode):
        """Popup with a "do not ask again" tick: the spectra do not share one
        instrumental function. Returns (proceed: bool, dont_ask: bool)."""
        methods = {r['method'] for r in resolved}
        if len(methods) > 1:
            what = "a mix of CMS and SMS spectra"
        else:
            what = f"several {next(iter(methods))} spectra with different instrumental functions"
        if mode == 'simultaneous':
            how = "They will be fitted together in one simultaneous fit, each section with its own instrumental function."
        else:
            how = "Each spectrum will be fitted separately with its own instrumental function."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Different instrumental functions")
        box.setText(f"You are about to fit {what}.\n\n{how}\n\nContinue?")
        dont_ask_cb = QCheckBox("Do not ask again")
        box.setCheckBox(dont_ask_cb)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        reply = box.exec()
        return (reply == QMessageBox.StandardButton.Yes, dont_ask_cb.isChecked())

    def initialize_parameters(self):
        """Initialize parameters from INSint.txt and INSexp.txt"""
        try:
            try:
                JN0 = int(self.jn0_input.text())
            except:
                JN0 = 32
            self.JN0 = JN0

            try:
                GCMS = float(self.GCMS_input.text())
            except:
                GCMS = 0.1
            self.GCMS = GCMS

            # Resolve the SMS polarization degree from its store; passed to TI as
            # the ``pol`` argument, which forwards it to the transmission workers.
            try:
                self.SMS_pol = float(self.polarization_input.text())
            except (ValueError, AttributeError):
                self.SMS_pol = SMS_POL_DEFAULT

            # The instrumental function currently in use: the theoretical one from
            # INSacc.txt when a NEW search has stored it, else INSexp.txt.
            self.INS, self.MulCo, self.x0 = get_sms_instrumental_from_global_files(self)

            print(f'Initialized: MulCo={self.MulCo}, x0={self.x0}')

        except Exception as e:
            self.set_status(f"Error loading INS files: {e}", "red")
            return False
        return True

    def calibration(self):
        """Perform calibration on RAW spectrum files"""
        if self._reject_if_busy('Calibration'):
            return
        self.inprogress = True
        self.busy_with = 'Calibration'
        if not self.path_list:
            self.set_status("No spectrum selected for calibration", "orange")
            return
        
        file = os.path.abspath(self.path_list[0])
        
        # Check if file is RAW format (not calibrated)
        if not (file.endswith('.mca') or file.endswith('.cmca') or file.endswith('.ws5') or 
                file.endswith('.w98') or file.endswith('.moe') or file.endswith('.m1') or 
                file.lower().endswith('.mcs')):
            self.set_status("Calibration only works with RAW files (.mca, .cmca, .ws5, .w98, .moe, .m1, .mcs)", "orange")
            return
        
        # Initialize
        if not self.initialize_parameters():
                return
        
        # Get parameters
        JN = int(self.jn0_input.text())
        # Determine experimental_method based on selected fit method (MS=1, SMS=3)
        # Note: Future functions will depend on experimental_method value for different fitting approaches
        experimental_method = 1 if self.MS_fit.isChecked() else 3
        
        # Get velocity direction
        vel_start = int(self.velocity_direction)
        
        # Disable calibration button during processing
        self.cal_btn.setEnabled(False)
        self.set_status("Calibration in progress...", "blue")
        
        # Start calibration in a separate thread
        self.calibration_thread = CalibrationThread(
            self.params_dir, file, experimental_method, self.INS, JN, self.x0, self.MulCo, vel_start, self.pool,
            GCMS=self.GCMS
        )
        self.calibration_thread.finished.connect(self.on_calibration_finished)
        self.calibration_thread.error.connect(self.on_calibration_error)
        self.calibration_thread.start()
    
    def on_calibration_finished(self, A, B, C):
        """Handle calibration completion"""
        try:
            # Plot calibration results
            plot_calibration(self.figure, A, B, C, gridcolor=self.gridcolor, theme=self._theme)
            self._update_legend_toggle()
            self.canvas.draw()
            self.toolbar.push_current()  # Set current view as home
            
            # Update calibration path (Calibration.dat was already saved by Calibration function)
            self.calibration_path = os.path.join(self.params_dir, 'Calibration.dat')

            # Calibration moved the velocity axis by the instrumental function's
            # gravity centre; the source has to move with it or every later fit
            # is off by that amount. See recentre_instrumental_after_calibration.
            moved = recentre_instrumental_after_calibration(self)
            note = (f" (instrumental function re-centred by {moved:+.4f} mm/s)"
                    if moved else "")
            # the in-memory copy must follow the file, or this session keeps
            # fitting with the old one
            self.INS, self.MulCo, self.x0 = \
                get_sms_instrumental_from_global_files(self)

            self.set_status(f"Calibration completed successfully{note}", "green")
        except Exception as e:
            self.set_status(f"Error processing calibration results: {e}", "red")
        finally:
            self.inprogress = False
            self.cal_btn.setEnabled(True)
            
    
    def on_calibration_error(self, error_msg):
        """Handle calibration error"""
        self.set_status(f"Calibration error: {error_msg}", "red")
        self.inprogress = False
        self.cal_btn.setEnabled(True)
    
    def showM_pressed(self):
        """Show model with spectrum and subspectra (non-blocking)"""
        # Parse the current content of process_path
        if self._reject_if_busy('Show model'):
            return
        self.inprogress = True
        self.busy_with = 'Show model'

        # Model-only mode: "Model_<N>" in the path box plots just the model on a
        # synthetic ±N mm/s grid, with no experimental spectrum.
        is_model_only, velocity_range, error_message = self.parse_model_only_request()
        if is_model_only:
            if error_message:
                self.set_status(error_message, "red")
                self.inprogress = False
                return
            self.path_list = []
        else:
            self.path_list = self.parse_process_path()
            velocity_range = 15.0
            path_error = self.check_spectrum_paths_exist(self.path_list)
            if path_error:
                self.set_status(path_error, "red")
                self.inprogress = False
                return

        # Initialize
        if not self.initialize_parameters():
            self.inprogress = False
            return

        # Refuse to start when an Expression/Distr/Corr text does not evaluate
        if not self.check_user_expressions("Show model"):
            self.inprogress = False
            return

        # Start model calculation in a separate thread
        if not self.path_list:
            self.set_status(
                f"Calculating model on synthetic grid (-{velocity_range:g}..{velocity_range:g} mm/s, 4096 points)...",
                "blue",
            )
        else:
            self.set_status("Calculating model...", "blue")

        self.show_model_thread = ShowModelThread(self, self.path_list, self.pool, velocity_range=velocity_range)
        self.show_model_thread.finished.connect(self.on_show_model_finished)
        self.show_model_thread.error.connect(self.on_show_model_error)
        self.show_model_thread.start()
    
    def on_show_model_finished(self, A, B, SPC_f, FS, FS_pos, p, model, has_nbaseline, backgrounds, instrumental_note='', hires_diff=None, Distri=None, Cor=None, Recon=None):
        """Handle show model completion"""
        try:
            # Show-model is not a fit result: wipe stale fit table content so
            # result-row clicks cannot affect the current model plot.
            self.results_table.clear_table()
            self.results_table.current_parameters = None

            # If we were showing a distribution, switch back to spectrum view
            if self._showing_distribution:
                self._showing_distribution = False
                self.SP_DI.setText('Distribution')

            # Get current colors from table (fresh read to handle delete/insert)
            current_colors = self.params_table.get_current_colors()

            flat_p = p if not has_nbaseline else np.concatenate(p)
            
            # Store FS_pos for toggle functionality
            self.current_FS_pos = FS_pos

            # Store show-model data so the distribution view can be opened here too
            self.last_plot_data = {
                'A': A,
                'B': B,
                'SPC_f': SPC_f,
                'hires_diff': hires_diff,
                'FS': FS,
                'FS_pos': FS_pos,
                'p': p,
                'p_flat': flat_p,
                'model_colors': current_colors,
                'chi2': None,
                'filepath': self.path_list[0] if self.path_list else None,
                'is_simultaneous': has_nbaseline,
                'is_show_model': True,
                'has_nbaseline': has_nbaseline,
                'backgrounds': backgrounds,
                'z_order': None,
                'model': model,
                'Distri': Distri or [],
                'Cor': Cor or [],
                'Recon': Recon or [],
            }

            self._plot_show_model_data(self.last_plot_data)
            
            note_suffix = f"\n{instrumental_note}" if instrumental_note else ""
            self.set_status("Model displayed successfully" + note_suffix, "green")
        except Exception as e:
            traceback.print_exc()
            self.set_status(f"Error plotting model: {e}", "red")
        finally:
            self.inprogress = False

    def _plot_show_model_data(self, data):
        """Render a show-model dataset and refresh canvas/toolbar state."""
        current_colors = data.get('model_colors', self.params_table.get_current_colors())
        if data['B'] is None:
            position_artists = plot_model_without_spectrum(
                self.figure, data['A'], data['SPC_f'], data['FS'], data['FS_pos'],
                data['p'], current_colors, gridcolor=self.gridcolor,
                theme=self._theme, model=data.get('model'),
                has_nbaseline=data.get('has_nbaseline', False),
                hires_diff=data.get('hires_diff')
            )
        elif data.get('has_nbaseline', False):
            position_artists = plot_model_with_nbaseline(
                self.figure, data['A'], data['B'], data['SPC_f'], data['FS'],
                data['FS_pos'], data['p'], data.get('model', []), current_colors,
                data.get('backgrounds'), gridcolor=self.gridcolor,
                theme=self._theme, hires_diff=data.get('hires_diff')
            )
        else:
            position_artists = plot_model(
                self.figure, data['A'], data['B'], data['SPC_f'], data['FS'],
                data['FS_pos'], data['p'], current_colors, data.get('backgrounds'),
                gridcolor=self.gridcolor, theme=self._theme,
                model=data.get('model'), hires_diff=data.get('hires_diff')
            )

        self.position_artists = position_artists if position_artists else []
        if self.position_artists:
            self.toolbar.toggle_positions_action.setEnabled(True)
            self.toolbar.toggle_positions_action.setChecked(True)
        else:
            self.toolbar.toggle_positions_action.setEnabled(False)

        self._update_legend_toggle()
        self.canvas.draw()
        self.toolbar.push_current()
    
    def on_show_model_error(self, error_msg):
        """Handle show model error"""
        self.set_status(f"Error showing model: {error_msg}", "red")
        self.inprogress = False
    
    def on_raw_to_dat_finished(self, success_msg):
        """Handle raw to dat conversion completion"""
        self.set_status(success_msg, "green")
    
    def on_raw_to_dat_error(self, error_msg):
        """Handle raw to dat conversion error"""
        self.set_status(f"RAW to DAT conversion error: {error_msg}", "red")
    
    def raw_to_dat(self):
        """Convert RAW spectra to .dat format using current calibration"""
        # Parse the current content of process_path
        self.path_list = self.parse_process_path()
        
        if not self.path_list:
            self.set_status("No spectrum selected", "orange")
            return
        
        # Check if calibration exists
        if not os.path.exists(self.calibration_path):
            self.set_status("Calibration file not found. Please calibrate first.", "red")
            return
        
        # Get save path from save_path field
        save_path = self.save_path.text().strip()
        if not save_path:
            self.set_status("Please specify save path", "orange")
            return
        
        # Start conversion in a separate thread
        self.set_status("Converting RAW to DAT...", "blue")
        
        self.raw_to_dat_thread = RawToDatThread(self, self.path_list, self.calibration_path, save_path)
        self.raw_to_dat_thread.finished.connect(self.on_raw_to_dat_finished)
        self.raw_to_dat_thread.error.connect(self.on_raw_to_dat_error)
        self.raw_to_dat_thread.start()

    def parse_model_only_request(self):
        """Inspect the path box for the model-only ``Model_<N>`` syntax.

        Typing e.g. ``Model_6`` (or ``Model_6.5``) into the spectrum path box
        requests model-only mode: just the model is plotted on a synthetic
        ±N mm/s grid, with no experimental spectrum loaded or fitted.

        Returns a ``(is_model_only, velocity_range, error_message)`` tuple:
          - ``is_model_only``: True when the box starts with ``Model_``
            (case-insensitive), regardless of whether the range is valid.
          - ``velocity_range``: the ± x-axis range in mm/s (float) when valid,
            else None.
          - ``error_message``: a human-readable reason when the syntax is
            malformed, else None.

        Anything that is NOT this keyword is a spectrum path, however it is
        spelled; a path that turns out not to exist is reported by
        :meth:`check_spectrum_paths_exist`, which names both possibilities.
        """
        text = self.process_path.toPlainText().strip().strip("[]'\" ")
        match = re.match(r'^model_(.*)$', text, re.IGNORECASE)
        if not match:
            return False, None, None
        raw = match.group(1).strip()
        try:
            value = float(raw)
        except ValueError:
            return True, None, (f"Invalid model-only range '{raw}'. "
                                "Use e.g. 'Model_6' or 'Model_6.5'.")
        if value <= 0:
            return True, None, f"Model-only range must be positive (got {value:g})."
        return True, value, None

    @staticmethod
    def check_spectrum_paths_exist(paths):
        """Return an error message if any of *paths* is not on disk, else None.

        The path box holds either spectrum files or the model-only ``Model_<N>``
        keyword, so anything that is neither has exactly one diagnosis and the
        message names both ways out. Callers must run this BEFORE starting a
        worker thread: the spectrum loader reports its own failures through the
        main window's status widget, which is not safe to touch from a thread —
        a bad path getting that far used to take the application down.

        A Bliss channel name is READ here instead of looked up on disk, for a
        similar reason: Bliss may be used from the main thread only, so the
        worker threads get the spectrum read now (see bliss_channel).
        """
        missing = []
        for path in paths:
            if bliss_channel.is_channel(path):
                try:
                    bliss_channel.refresh(path)
                except Exception as e:
                    return f"Could not read Bliss channel {path}: {e}"
            elif not os.path.exists(path):
                missing.append(path)
        if not missing:
            return None
        return ("Not a spectrum file: " + ", ".join(missing)
                + ". Enter an existing path, or 'Model_<range>' (e.g. 'Model_6') "
                  "to calculate the model without a spectrum.")

    def parse_process_path(self):
        """Parse the process_path text field to extract file paths"""
        try:
            text = self.process_path.toPlainText().strip()
            # Try to evaluate as Python list
            paths = ast.literal_eval(text)
            if isinstance(paths, list):
                return paths
            elif isinstance(paths, str):
                return [paths]
            else:
                return []
        except:
            # If parsing fails, try simple comma-separated values
            text = self.process_path.toPlainText().strip()
            if not text:
                return []
            # Remove brackets and quotes, split by comma
            text = text.strip("[]'\"")
            paths = [p.strip().strip("'\" ") for p in text.split(',') if p.strip()]
            return paths

    def _reject_if_busy(self, what):
        """True -- and SAYS SO -- when another calculation is already running.

        Only one calculation may run at a time: they share the process pool,
        the integration grid and the instrumental function, so a fit started
        while an instrumental-function search is running would read parameters
        the search is in the middle of changing.

        The guard itself is not new; the message is. Every entry point used to
        ``return`` silently, so clicking Fit during a search did nothing at all
        and looked like a broken button rather than a refusal.
        """
        if self._interrupting:
            self.set_status(f"{what} is not available right now: the "
                            f"interrupted calculation is still stopping.", "orange")
            return True
        if not self.inprogress:
            return False
        running = getattr(self, 'busy_with', 'another calculation')
        self.set_status(f"{what} is not available right now: {running} is "
                        f"still running. Press ! INTERRUPT ! to stop it.",
                        "orange")
        return True

    def interrupt(self):
        """Emergency abort: kill the current pool and create a new one.

        TWO mechanisms, because neither one alone stops everything:

        * the flag. A thread cannot be killed, so the model function is asked to
          give up instead: every evaluation checks ``fit_cancel`` and raises
          ``FitInterrupted``, which unwinds out of the minimiser. This is what
          stops the THEORETICAL instrumental-function search, whose cost is
          rebuilding the dynamical source shape in its own thread -- terminating
          the pool did nothing to that, so Interrupt did nothing.
        * the pool. An ordinary fit farms every evaluation out to the process
          pool, so terminating it aborts the fit from underneath. Kept, because
          it is also the only way to stop work already handed to a worker.

        The flag is set FIRST: a pool terminated while the search is between
        evaluations would otherwise let it run on to the next one.

        Every pool call in TI checks the flag too (models._pool_starmap), so
        the fit, sequential fit, Show model, calibration and the Gaussian search
        stop within ~50 ms as well, without anything being killed.

        Gentle first: the pool is killed only if its workers are STILL busy a
        second after the click (a long evaluation already handed out). The app
        is released only once the calculation thread has ended, so a new
        calculation never starts next to a dying one. Clicks while waiting, or
        with nothing running, touch nothing.
        """
        if self._interrupting:
            return
        if not self.inprogress:
            self.set_status("Nothing to interrupt", "orange")
            return
        self._interrupting = True
        self._pool_killed = False
        self.fit_cancel.set()
        self.set_status("Interrupting...", "orange")
        QTimer.singleShot(1000, self._finish_interrupt)

    def _finish_interrupt(self, first=True):
        """Second half of interrupt(): kill the pool if still needed, then wait
        for the calculation thread to end."""
        # _cache holds the pool's unfinished jobs (multiprocessing internals)
        if first and self.pool is not None and getattr(self.pool, '_cache', None):
            try:
                # Terminate the old pool
                self.pool.terminate()
                self.pool.join()
                # Create a new pool
                num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
                self.pool = mp.Pool(processes=num_processes)
                self._pool_killed = True
            except Exception as e:
                self.set_status(f"Error during interrupt: {str(e)}", "red")
        if any(isinstance(t, QThread) and t.isRunning() for t in vars(self).values()):
            QTimer.singleShot(100, lambda: self._finish_interrupt(first=False))
            return
        self.fit_cancel.clear()
        self.inprogress = False
        self._interrupting = False
        self.set_status("Interrupted (pool terminated and recreated)"
                        if self._pool_killed else "Interrupted", "orange")

    @staticmethod
    def _detect_os_dark_mode():
        """Detect whether the OS is currently in dark mode."""
        app = QApplication.instance()
        if app is None:
            return True
        try:
            return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        except AttributeError:
            # Fallback for older Qt: check window background luminance
            bg = app.palette().color(QPalette.ColorRole.Window)
            return bg.lightness() < 128

    def _force_color_scheme(self, dark):
        """Force the Qt application to dark or light color scheme.

        Uses QStyleHints.setColorScheme (Qt 6.8+) which makes every
        QPalette-aware widget follow the requested scheme automatically,
        without any setStyleSheet calls.
        """
        app = QApplication.instance()
        if app is None:
            return
        scheme = Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
        try:
            app.styleHints().setColorScheme(scheme)
        except AttributeError:
            pass  # Qt < 6.8 — native palette stays as-is

    def _load_theme(self):
        """Load matplotlib theme from JSON file based on current mode."""
        theme_file = 'theme_dark.json' if self._is_dark_mode else 'theme_light.json'
        theme_path = os.path.join(self.dir_path, theme_file)
        try:
            with open(theme_path, 'r') as f:
                self._theme = json.load(f)
        except Exception:
            # Fallback defaults (original dark mode)
            self._theme = {
                'name': 'Dark mode',
                'figure_facecolor': 'black',
                'axes_facecolor': 'black',
                'axes_text_color': 'white',
                'gridcolor': 'white',
                'legend_facecolor': 'black',
                'legend_edgecolor': 'white',
                'legend_textcolor': 'white',
            }
        self.gridcolor = self._theme.get('gridcolor', 'white')
        self.BGcolor = self._theme.get('figure_facecolor', 'black')

    def _apply_theme(self):
        """Apply the loaded theme to matplotlib figure.

        Qt widgets follow the QPalette set by _force_color_scheme.
        Only the matplotlib figure and the Supp-menu theme entry text
        are touched here.
        """
        t = self._theme

        # Matplotlib figure background
        self.figure.patch.set_facecolor(t['figure_facecolor'])

        # Relabel the Supp-menu theme entry (no setStyleSheet — palette handles colors)
        self.theme_action.setText(theme_action_text(self._is_dark_mode))
        # ... and repaint the "Contact the author" accent for the new background.
        self.contact_action.setIcon(contact_icon(self._is_dark_mode))

        # Style instrumental buttons with a color distinct from background
        if self._is_dark_mode:
            instrumental_style = "background-color: rgb(70, 70, 70); color: white;"
        else:
            instrumental_style = "background-color: rgb(200, 200, 200); color: black;"
        self.instrumental_btn.setStyleSheet(instrumental_style)
        self.instrumental_btn2.setStyleSheet(instrumental_style)

        # Style the matplotlib navigation toolbar to match the mode
        if self._is_dark_mode:
            self.toolbar.setStyleSheet(
                "QToolBar { background-color: #2d2d2d; border: none; }"
                "QToolButton { background-color: #2d2d2d; color: white; border: none; padding: 2px; }"
                "QToolButton:hover { background-color: #505050; }"
                "QToolButton:checked { background-color: #606060; }"
            )
        else:
            self.toolbar.setStyleSheet(
                "QToolBar { background-color: #e8e8e8; border: none; }"
                "QToolButton { background-color: #e8e8e8; color: black; border: none; padding: 2px; }"
                "QToolButton:hover { background-color: #c0c0c0; }"
                "QToolButton:checked { background-color: #b0b0b0; }"
            )

        # Force toolbar icons to refresh for the new palette
        for text, tooltip_text, image_file, callback in self.toolbar.toolitems:
            if text is not None and callback in self.toolbar._actions:
                self.toolbar._actions[callback].setIcon(
                    self.toolbar._icon(image_file + '.png'))

        # Refresh existing matplotlib axes with theme colors
        if self.figure.get_axes():
            for ax in self.figure.get_axes():
                ax.set_facecolor(t['axes_facecolor'])
                ax.tick_params(colors=t['axes_text_color'])
                ax.xaxis.label.set_color(t['axes_text_color'])
                ax.yaxis.label.set_color(t['axes_text_color'])
                if ax.get_title():
                    ax.title.set_color('r')  # Keep chi2 title red
                for spine in ax.spines.values():
                    spine.set_edgecolor(t['axes_text_color'])
                for line in ax.xaxis.get_gridlines():
                    line.set_color(t['gridcolor'])
                for line in ax.yaxis.get_gridlines():
                    line.set_color(t['gridcolor'])
                for line in ax.get_lines():
                    if line.get_linestyle() == '--':
                        line.set_color(t['gridcolor'])
                for child_ax in ax.child_axes:
                    child_ax.tick_params(colors=t['axes_text_color'])
                    child_ax.yaxis.label.set_color(t['axes_text_color'])
                    for spine in child_ax.spines.values():
                        spine.set_edgecolor(t['axes_text_color'])
                legend = ax.get_legend()
                if legend:
                    legend.get_frame().set_facecolor(t.get('legend_facecolor', t['axes_facecolor']))
                    legend.get_frame().set_edgecolor(t.get('legend_edgecolor', t['axes_text_color']))
                    for text in legend.get_texts():
                        text.set_color(t.get('legend_textcolor', t['axes_text_color']))
            self.canvas.draw()

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self._is_dark_mode = not self._is_dark_mode
        self._force_color_scheme(self._is_dark_mode)
        self._load_theme()
        self._apply_theme()
        mode_name = self._theme.get('name', 'Dark mode' if self._is_dark_mode else 'Light mode')
        self.set_status(f"Switched to {mode_name}")

    def choose_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Pick a spectrum...",
            self.workfolder,
            "DAT (*.dat *.spc *.exp);;RAW (*.mca *.cmca *.ws5 *.moe *.w98 *.m1 *.mcs);;DAT/RAW (*.dat *.mca *.cmca *.ws5 *.moe *.w98 *.m1 *.mcs);;TXT (*.txt);;All files (*.*)"
        )

        if file_paths:
            self.path_list = file_paths
            # Format for display: add newlines after each comma
            display_text = str(file_paths).replace("', '", "',\n'")
            self.process_path.setPlainText(display_text)
            # Set save path: if multiple files, use first_folder/result/result (no extension)
            if len(file_paths) > 1:
                first_folder = os.path.dirname(file_paths[0])
                result_folder = os.path.join(first_folder, 'result')
                result_path = os.path.join(result_folder, 'result')
                # Normalize path to use consistent separators
                self.save_path.setText(os.path.normpath(result_path))
            else:
                self.save_path.setText(os.path.normpath(file_paths[0]))
            # Automatically show the selected spectrum(s)
            self.show_pressed()
        else:
            self.set_status("Selection canceled", "orange")

    def instrumental_ref(self, refine):
        """The ``ref`` code for the current approximation method.

        0 / 1 are Find / Refine with the empirical sum of Gaussians, 2 / 3 the
        same two with the theoretical simulated shape. The menu entries no
        longer choose between them -- ``self.instrumental_method`` does.
        """
        return (1 if refine else 0) + (
            2 if getattr(self, 'instrumental_method',
                         DEFAULT_INSTRUMENTAL_METHOD) == 'theory' else 0)

    def instrumental_pressed(self, ref, mode):
        """
        Handle instrumental function calculation/refinement button press.
        
        Args:
            ref: 0 for Find, 1 for Refine (empirical sum of Gaussians);
                 2 for Find, 3 for Refine the THEORETICAL simulated 57FeBO3 shape
            mode: 0 for single line, 1 for model, 2 for pure a-Fe
        """
        if self._reject_if_busy('The instrumental-function search'):
            return
        # A THEORY "Find" (ref 2) discards the stored shape and restarts from
        # the built-in values, so it asks first -- both to say that Refine is
        # usually the better button and to let the user correct the starting
        # numbers for a source run far from the usual settings. Confirmed
        # BEFORE inprogress is claimed, so cancelling leaves nothing latched.
        self.theory_find_overrides = None
        if ref == 2:
            overrides = open_theory_find_dialog(self)
            if overrides is None:
                self.set_status("Instrumental-function search cancelled",
                                "orange")
                return
            self.theory_find_overrides = overrides
        self.inprogress = True
        self.busy_with = 'The instrumental-function search'
        # Ensure parameters are initialized
        if not self.initialize_parameters():
            self.inprogress = False
            return
        
        # Validation - check if model is defined for mode == 1
        if mode == 1:
            # Check if there's at least one model component (row 1 or above)
            has_model = False
            if len(self.params_table.row_widgets) > 1:
                # Check if any row beyond baseline (row 0) has a model defined
                for row in range(1, len(self.params_table.row_widgets)):
                    row_widget = self.params_table.row_widgets[row]
                    # Navigate to model button: row_widget -> layout -> first item (start_widget) -> layout -> second item (model_btn)
                    start_widget = row_widget.layout().itemAt(0).widget()
                    model_btn = start_widget.layout().itemAt(1).widget()
                    if model_btn.text() != 'None':
                        has_model = True
                        break
            
            if not has_model:
                self.set_status("Specify model to restore instrumental function", "red")
                self.inprogress = False
                return
        
        if (self.MS_fit.isChecked() and mode == 0):
            self.set_status("This will not work...", "red")
            self.inprogress = False
            return

        if ref in (2, 3) and self.MS_fit.isChecked():
            # The theoretical shape is the pure-nuclear reflection of an iron
            # borate crystal: there is no such thing for a radioactive source.
            self.set_status(
                "The theoretical instrumental function describes a synchrotron "
                "Mössbauer source (⁵⁷FeBO₃). Switch to SMS, or use the standard "
                "search for CMS.", "red")
            self.inprogress = False
            return
        
        if not self.path_list or len(self.path_list) == 0:
            self.set_status("No spectrum loaded", "red")
            self.inprogress = False
            return
        
        file = os.path.abspath(self.path_list[0])
        if not os.path.exists(file):
            self.set_status("Spectrum file does not exist", "red")
            self.inprogress = False
            return
        

        
        # Start instrumental calculation in a thread
        self.set_status(f"Running instrumental function (ref={ref}, mode={mode})...", "cyan")
        
        self.instrumental_thread = InstrumentalThread(self, ref, mode, self.pool)
        self.instrumental_thread.finished.connect(self.on_instrumental_finished)
        self.instrumental_thread.error.connect(self.on_instrumental_error)
        self.instrumental_thread.start()

    def on_instrumental_finished(self, result):
        """Handle instrumental function completion"""
        try:
            # Plot results on the figure
            gridcolor = self.gridcolor
            result_svg, result_png = plot_instrumental_result(
                self.figure, result['A'], result['B'], result['F'], result['F2'],
                result['p'], result['hi2'], result['file'], self.dir_path, gridcolor,
                theme=self._theme
            )
            
            # Update canvas the same way as showM_pressed does
            self.canvas.draw()
            self.toolbar.push_current()
            
            if result.get('theory') is not None:
                th = result['theory']
                er = result.get('theory_err', {})
                detail = ", ".join(
                    f"{k.replace('_urad', '')} = {th[k]:+.4g}"
                    + (f" ± {er[k]:.2g}" if np.isfinite(er.get(k, np.nan)) else "")
                    for k in th)
                self.set_status(
                    f"Theoretical instrumental function found. χ² = {result['hi2']:.3f}\n"
                    f"{detail}\nFWHM = {result['fwhm']:.4f} mm/s, "
                    f"centre = {result['centre']:+.4f} mm/s\n"
                    f"Results saved to {self.dir_path}", "green")
            else:
                self.set_status(f"Instrumental function completed. χ² = {result['hi2']:.3f}\nResults saved to {self.dir_path}", "green")
            
            # Update parameters if mode == 1
            if result['mode'] == 1 and result['mod_p_len']:
                self.p = result['p'][:result['mod_p_len']]
            
            # Update x0 and MulCo if available
            if self.SMS_fit.isChecked():
                if result['x0'] is not None:
                    self.x0 = result['x0']
                if result['MulCo'] is not None:
                    self.MulCo = result['MulCo']
            elif self.MS_fit.isChecked():
                self.GCMS = result['G']
                self.GCMS_input.setText(str(self.GCMS))
        except Exception as e:
            self.set_status(f"Error plotting instrumental results: {e}\n{traceback.format_exc()}", "red")
        finally:
            self.inprogress = False
    
    def on_instrumental_error(self, error_msg):
        """Handle instrumental function error"""
        self.set_status(f"Instrumental function error: {error_msg}", "red")
        self.inprogress = False

    def show_pressed(self):
        """Load and display the selected spectrum(s)"""
        if self._reject_if_busy('Show spectrum'):
            return
        self.inprogress = True
        self.busy_with = 'Show spectrum'

        # Model-only mode (Model_<N> in the path box) has no spectrum to show.
        if self.parse_model_only_request()[0]:
            self.set_status("Model-only mode (Model_N): no spectrum to show. "
                                  "Use 'Show model', or enter a spectrum path.", "orange")
            self.inprogress = False
            return

        # Parse the current content of process_path
        self.path_list = self.parse_process_path()

        if not self.path_list:
            self.set_status("No spectrum selected", "orange")
            self.inprogress = False
            return

        path_error = self.check_spectrum_paths_exist(self.path_list)
        if path_error:
            self.set_status(path_error, "red")
            self.inprogress = False
            return
        try:
            A_list, B_list = load_spectrum(self, self.path_list, calibration_path=self.calibration_path)
            if A_list and B_list:              
                self.backgrounds = calculate_backgrounds(self.path_list, self.calibration_path)
                plot_spectrum(self.figure, A_list, B_list, self.path_list, self.backgrounds, theme=self._theme)
                self._update_legend_toggle()
                self.toolbar.push_current()  # Set current view as home
                self.set_status(f"Spectra displayed ({len(A_list)})", "green")
                # Update baseline Ns based on new spectrum
                self.params_table.update_baseline_from_bg()
            else:
                self.set_status("Could not load spectrum", "red")
        except Exception as e:
            self.set_status(f"Error displaying spectrum: {e}", "red")
        finally:
            self.inprogress = False

    def choose_workfolder(self):
        """Open folder selection dialog to choose workfolder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Choose Workfolder", self.workfolder)
        if folder_path:
            self.workfolder = folder_path
            self.set_status(f"Workfolder changed to: {folder_path}", "green")
        else:
            self.set_status("Workfolder selection canceled", "orange")

    def show_sequence_fitting_options(self):
        """Show dropdown menu for sequence fitting options"""
        menu = QMenu(self)
        
        # Add options
        initial_action = QAction("Take always initial guess for the sequence of spectra", self)
        initial_action.triggered.connect(lambda: self.set_sequence_fitting_type(0))
        menu.addAction(initial_action)
        
        result_action = QAction("Take result as initial guess for the sequence of spectra", self)
        result_action.triggered.connect(lambda: self.set_sequence_fitting_type(1))
        menu.addAction(result_action)
        
        # Show menu at button position
        menu.exec(self.seq_fit_btn.mapToGlobal(self.seq_fit_btn.rect().bottomLeft()))

    def set_sequence_fitting_type(self, fitting_type):
        """Set sequence fitting type and update button text"""
        self.sequence_fitting_type = fitting_type
        if fitting_type == 0:
            self.seq_fit_btn.setText("Sequence Fitting\n(initial)")
        else:
            self.seq_fit_btn.setText("Sequence Fitting\n(result)")
        self.set_status(f"Sequence fitting type set to: {'initial' if fitting_type == 0 else 'result'}", "blue")

    def replot_result(self, row_index):
        """
        Replot results with subspectra reordered to bring clicked component to top.

        Args:
            row_index: Index of the clicked results-table row (each component
                occupies 3 rows, so component = row_index // 3)

        The function moves the selected subspectrum to the highest z-order (on top),
        allowing users to click buttons in any order to customize the display.
        Works for both single and simultaneous fitting.
        """
        try:
            if self.results_table.current_parameters is None:
                self.set_status("No fitting results to reorder", "orange")
                return

            # Check if we have plot data
            if self.last_plot_data is None:
                self.set_status("No plot data available. Please fit spectrum first.", "orange")
                return
            
            component_index = row_index // 3  # Which component (0, 1, 2, ...)
            
            # Skip baseline (component 0) - it's not a subspectrum
            if component_index == 0:
                self.set_status("Cannot reorder baseline", "orange")
                return
            
            # Get model list to check for special models
            model_list = self.params_table.get_model_list()
            
            # Special models that don't produce individual subspectra
            non_subspectrum_models = {'Nbaseline', 'Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables'}
            
            # Skip special models when counting subspectra
            # Count how many actual subspectra appear before this component
            subspectrum_index = 0
            for i in range(1, component_index):  # Start from 1 to skip baseline
                if i < len(model_list) and model_list[i] not in non_subspectrum_models:
                    subspectrum_index += 1
            
            # Check if clicked component is a special model (no reordering)
            if component_index < len(model_list) and model_list[component_index] in non_subspectrum_models:
                self.set_status(f"Cannot reorder {model_list[component_index]}", "orange")
                return
            
            # Handle simultaneous vs single fitting
            if self.last_plot_data['is_simultaneous']:
                # For simultaneous fitting, flatten all FS lists to get total subspectra count
                FS_list = self.last_plot_data['FS_list']
                total_subspectra = sum(len(FS) for FS in FS_list)
                
                if subspectrum_index >= total_subspectra:
                    self.set_status(f"Invalid component index: {component_index}", "red")
                    return
                
                # Initialize z_order if not set (flattened across all spectra)
                if self.last_plot_data['z_order'] is None:
                    z_order = []
                    for FS in FS_list:
                        if len(FS) > 0:
                            z_order.extend(calculate_z_order(FS).tolist())
                    self.last_plot_data['z_order'] = np.array(z_order)
            else:
                # Single spectrum fitting
                FS = self.last_plot_data['FS']
                if subspectrum_index >= len(FS):
                    self.set_status(f"Invalid component index: {component_index}", "red")
                    return
                
                # Initialize z_order if not set
                if self.last_plot_data['z_order'] is None:
                    self.last_plot_data['z_order'] = calculate_z_order(FS)
            
            # Bring selected subspectrum to top by setting highest z-order
            max_z = max(self.last_plot_data['z_order'])
            self.last_plot_data['z_order'][subspectrum_index] = max_z + 1
            
            # Replot with custom z-order
            self._replot_with_custom_order()
            
            model_name = self.params_table.get_model_list()[component_index] if component_index < len(self.params_table.get_model_list()) else f"Component {component_index}"
            self.set_status(f"Brought '{model_name}' to top", "green")
            
        except Exception as e:
            traceback.print_exc()
            self.set_status(f"Replot error: {e}", "red")

    def _replot_with_custom_order(self):
        """
        Replot the spectrum using stored data with custom z-order.
        """
        # If we were showing a distribution, switch back to spectrum view
        if self._showing_distribution:
            self._showing_distribution = False
            self.SP_DI.setText('Distribution')

        data = self.last_plot_data

        if data.get('is_show_model'):
            self._plot_show_model_data(data)
            return
        
        if data['is_simultaneous']:
            # Simultaneous fitting - multiple spectra
            _, _, position_artists_list = plot_simultaneous_fitting_result(
                self.figure, data['A_list'], data['B_list'], data['SPC_f_list'],
                data['FS_list'], data['FS_pos_list'], data['p'], data['begining_spc'],
                data['model_colors'], data['chi2'], data['spectrum_files'],
                self.dir_path, z_order=data['z_order'], gridcolor=self.gridcolor,
                theme=self._theme, model=data.get('model'),
                hires_diff_list=data.get('hires_diff_list'),
                chi2_spread=data.get('chi2_spread')
            )
            self.position_artists = [artist for sublist in position_artists_list for artist in sublist]
        else:
            # Single spectrum fitting - use plot_fitting_result with z_order parameter
            _, _, position_artists = plot_fitting_result(
                self.figure, data['A'], data['B'], data['SPC_f'], data['FS'],
                data['FS_pos'], data['p'], data['model_colors'], data['chi2'],
                data['filepath'], self.dir_path, z_order=data['z_order'], gridcolor=self.gridcolor,
                theme=self._theme, model=data.get('model'),
                hires_diff=data.get('hires_diff'),
                chi2_spread=data.get('chi2_spread')
            )
            self.position_artists = position_artists if position_artists else []
        
        # Redraw canvas
        self._update_legend_toggle()
        self.canvas.draw()
        self.toolbar.push_current()
    
    def toggle_distribution(self):
        """Toggle between spectrum and distribution view."""
        try:
            if self.last_plot_data is None:
                self.set_status("No plot data available. Please fit spectrum first.", "orange")
                return
            
            data = self.last_plot_data
            model = data.get('model', [])
            Distri = data.get('Distri', [])
            Cor = data.get('Cor', [])
            Recon = data.get('Recon', [])

            if self._showing_distribution:
                # Switch back to spectrum view
                self._showing_distribution = False
                self.SP_DI.setText('Distribution')
                self._replot_with_custom_order()
                self.set_status("Showing spectrum", "green")
            else:
                # Switch to distribution view. A parametric Distr contributes a PDF
                # string; a Recon contributes a free-weight vector — either one is a
                # distribution to show.
                if not Distri and 'Recon' not in (model or []):
                    self.set_status("No distribution in the model", "orange")
                    return

                parameter_names = self.params_table.get_parameter_names()
                gridcolor = self.gridcolor
                p_for_distribution = data.get('p_flat', data['p'])

                success = plot_distribution(
                    self.figure, model, p_for_distribution, Distri, Cor,
                    parameter_names, gridcolor=gridcolor,
                    model_colors=data.get('model_colors'), theme=self._theme, Recon=Recon
                )
                
                if success:
                    self._showing_distribution = True
                    self.SP_DI.setText('Spectrum')
                    self.canvas.draw()
                    self.toolbar.push_current()
                    self.set_status("Distributions and correlations", "green")
                else:
                    self.set_status("No distribution in the model", "orange")
                    
        except Exception as e:
            traceback.print_exc()
            self.set_status(f"Distribution error: {e}", "red")


    def validate_spectrum_files(self, spectrum_files):
        """
        Validate that all spectrum files exist and can be loaded.
        
        Args:
            spectrum_files: List of spectrum file paths to validate
            
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        # Check file existence first
        path_error = self.check_spectrum_paths_exist(spectrum_files)
        if path_error:
            return False, path_error
        
        # Try to load all files (without plotting)
        A_list, B_list = load_spectrum(self, spectrum_files, calibration_path=self.calibration_path)
        
        if not A_list or not B_list:
            return False, "Could not load spectrum files. Please check file format."
        
        # Check each file was loaded successfully
        if len(A_list) != len(spectrum_files) or len(B_list) != len(spectrum_files):
            # Try to identify which file caused the problem
            for i, file_path in enumerate(spectrum_files):
                try:
                    A_list, B_list = load_spectrum(self, [file_path], calibration_path=self.calibration_path)
                    if not A_list or not B_list:
                        return False, f"Invalid spectrum file: {file_path}"
                except Exception as file_error:
                    return False, f"Error loading {file_path}: {str(file_error)}"
            
            # If we get here, the error is not file-specific
            return False, "Error loading spectra. Could not identify problematic file."
        
        # Check for empty data
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            if len(A) == 0 or len(B) == 0:
                return False, f"File contains no data: {spectrum_files[i]}"
        
        return True, None
            
    
    def fit_pressed(self):
        """
        Handle fit button press to perform spectrum fitting.
        
        Workflow:
        1. Validate parameters are initialized
        2. Check for sequential fitting conditions
        3. Get spectrum file(s) to fit
        4. Start fitting in background thread
        5. Update results table and plot when finished
        """
        if self._reject_if_busy('Fitting'):
            return
        self.inprogress = True
        self.busy_with = 'Fitting'

        # Model-only mode (Model_<N> in the path box) has no spectrum to fit.
        if self.parse_model_only_request()[0]:
            self.set_status("Model-only mode (Model_N): nothing to fit. "
                                  "Use 'Show model', or enter a spectrum path to fit.", "orange")
            self.inprogress = False
            return

        self.set_status("Starting fit...", "cyan")

        if not self.initialize_parameters():
            self.inprogress = False
            return

        # Refuse to start when an Expression/Distr/Corr text does not evaluate
        if not self.check_user_expressions("Fit"):
            self.inprogress = False
            return

        try:
            # Get spectrum files
            spectrum_files = self.parse_process_path()
            
            if not spectrum_files:
                self.set_status("No spectrum file loaded. Please load a file first.", "orange")
                self.inprogress = False
                return
            
            # Validate all spectrum files can be loaded
            valid, error_msg = self.validate_spectrum_files(spectrum_files)
            if not valid:
                self.set_status(f"Spectrum validation failed: {error_msg}", "red")
                self.inprogress = False
                return
            
            # Check for sequential fitting: no Nbaseline AND multiple spectra
            fitting_mode = fitting_io.determine_fitting_mode(self, spectrum_files)
            
            if fitting_mode == 'sequential':
                # Sequential fitting - ask user
                reply = QMessageBox.question(
                    self, 'Sequential Fitting',
                    f"Do you want to start sequential fitting of {len(spectrum_files)} spectra?\n\n"
                    f"Please check the save path:\n{self.save_path.text() or 'NOT SET'}\n\n"
                    f"Results will be saved with spectrum basenames.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Batch = each spectrum fitted on its own (its own metadata).
                    # Confirm any method override / different instrumental
                    # functions across the batch before starting.
                    if not self.confirm_instrumental_methods(spectrum_files, 'sequential'):
                        self.inprogress = False
                        return
                    self.start_sequential_fitting(spectrum_files)
                    return
                else:
                    # User declined - fit only first spectrum
                    self.set_status("Sequential fitting canceled. Fitting first spectrum only.", "orange")
                    spectrum_files = [spectrum_files[0]]
                    fitting_mode = 'single'

            # Single spectrum or simultaneous fitting (with Nbaseline): confirm
            # the instrumental method(s) before spawning the fitting thread.
            files_for_confirm = spectrum_files if fitting_mode == 'simultaneous' else [spectrum_files[0]]
            if not self.confirm_instrumental_methods(files_for_confirm, fitting_mode):
                self.inprogress = False
                return

            # Snapshot current parameter links (=[X,Y]) and the whole model at
            # fit start; the latter becomes the model "Save result" writes.
            self._fit_links_snapshot = self.params_table.get_link_snapshot()
            self._fit_model_snapshot = model_file_rows(self)

            spectrum_file = spectrum_files[0]
            self.set_status(f"Fitting spectrum: {os.path.basename(spectrum_file)}", "cyan")

            # Start fitting in background thread
            self.fitting_thread = FittingThread(self, spectrum_file, self.pool)
            self.fitting_thread.finished.connect(self.on_fitting_finished)
            self.fitting_thread.error.connect(self.on_fitting_error)
            self.fitting_thread.start()
            
        except Exception as e:
            error_msg = f"Fit error: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.set_status(f"Fit error: {e}", "red")
            self.inprogress = False
    
    def start_sequential_fitting(self, spectrum_files):
        """
        Start sequential fitting of multiple spectra using fitting_io logic.
        
        Args:
            spectrum_files: List of spectrum file paths to fit sequentially
        """
        # Check if save path is set
        if not self.save_path.text().strip():
            self.set_status("Sequential fitting requires save path to be set", "red")
            return
        
        # Initialize sequence_params for result mode (None for initial mode)
        self.sequence_params = None

        # Snapshot links (and the model) once at batch start so UI edits during
        # fitting do not leak into the stored result-table metadata.
        self._fit_links_snapshot = self.params_table.get_link_snapshot()
        self._fit_model_snapshot = model_file_rows(self)

        # Calculate backgrounds for all spectra upfront
        self.set_status(f"Calculating backgrounds for {len(spectrum_files)} spectra...", "cyan")
        
        backgrounds = calculate_backgrounds(spectrum_files, self.calibration_path)
        
        print("calculated backgrounds:", backgrounds)

        # Determine mode name for logging
        mode_name = 'initial guess' if self.sequence_fitting_type == 0 else 'result as model'
        self.set_status(f"Starting sequential fitting of {len(spectrum_files)} spectra (mode: {mode_name})...", "cyan")
        
        # Start sequential fitting thread
        self.sequential_fitting_thread = SequentialFittingThread(
            self, spectrum_files, self.pool, self.sequence_fitting_type, backgrounds
        )
        self.sequential_fitting_thread.progress.connect(self._on_sequential_progress)
        self.sequential_fitting_thread.spectrum_fitted.connect(self._on_spectrum_fitted)
        self.sequential_fitting_thread.finished.connect(self._on_sequential_finished)
        self.sequential_fitting_thread.start()
    
    def _on_spectrum_fitted(self, spectrum_file, result):
        """Handle GUI updates and file saving for one fitted spectrum (runs in main thread)"""
        try:
            # Update results table
            model_list = self.params_table.get_model_list()
            model_colors = self.params_table.get_current_colors()
            parameter_names = self.params_table.get_parameter_names()
            expression_texts = self.params_table.get_expression_texts()
            expression_texts = {**expression_texts, **_recon_weight_texts(model_list, result.get('Recon', []))}

            fitted_parameters = result['parameters']
            errors = result['errors']
            chi2 = result['chi2']
            covariance_matrix = result['covariance_matrix']
            fix = result.get('fix', np.array([], dtype=int))

            self.results_table.fill_table(
                fitted_parameters,
                model_list,
                model_colors,
                parameter_names,
                covariance_matrix,
                errors,
                fix,
                expression_texts
            )
            self.results_table.current_links = dict(self._fit_links_snapshot)
            self.results_table.current_model_rows = fitted_model_rows(
                self._fit_model_snapshot, fitted_parameters, result.get('Recon', []))
            self.results_table.current_chi2 = chi2

            # Plot the result
            self.plot_fitting_result(result)
            
            # Save files
            self._save_sequential_result_files(spectrum_file)
            
        except Exception as e:
            print(f"Error handling fitted spectrum: {e}\n{traceback.format_exc()}")
    
    def _on_sequential_progress(self, index, total, spectrum_file, status):
        """Handle progress updates from sequential fitting"""
        if status == 'fitting':
            self.set_status(
                f"Sequential fitting [{index + 1}/{total}]: {os.path.basename(spectrum_file)}",
                "cyan",
            )
        elif status == 'saved':
            self.set_status(
                f"Saved [{index + 1}/{total}]: {os.path.basename(spectrum_file)}",
                "blue",
            )
        elif status == 'failed':
            self.set_status(
                f"Failed [{index + 1}/{total}]: {os.path.basename(spectrum_file)}",
                "orange",
            )
        elif status == 'error':
            self.set_status(
                f"Error [{index + 1}/{total}]: {os.path.basename(spectrum_file)}",
                "red",
            )
    
    def _on_sequential_finished(self, summary):
        """Handle completion of sequential fitting"""
        total = summary['total']
        succeeded = summary['succeeded']
        failed = summary['failed']
        errors = summary['errors']
        
        # Clean up sequence_params
        self.sequence_params = None
        
        if failed == 0:
            self.set_status(f"Sequential fitting complete! All {total} spectra fitted and saved.", "green")
        else:
            error_summary = "\n".join([f"  - {os.path.basename(f)}: {e}" for f, e in errors[:5]])  # Show first 5 errors
            if len(errors) > 5:
                error_summary += f"\n  ... and {len(errors) - 5} more errors"
            
            self.set_status(
                f"Sequential fitting complete: {succeeded}/{total} succeeded, {failed} failed.\n"
                f"Errors:\n{error_summary}",
                "orange",
            )
        self.inprogress = False
    
    def _save_sequential_result_files(self, spectrum_file):
        """Save result files for one spectrum (file I/O only, called from main thread)"""
        try:
            # Get base path from save_path and spectrum filename
            save_dir = os.path.dirname(self.save_path.text())
            spectrum_basename = strip_known_extension(os.path.basename(spectrum_file))
            base_path = os.path.join(save_dir, spectrum_basename)

            # Create directory if needed
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Append to an existing parameter file, otherwise start a new one
            mode = 'append' if os.path.exists(base_path + '_param.txt') else 'new'
            self._write_result_files(base_path, os.path.basename(spectrum_file), mode)

            print(f"[Sequential] Saved results for {spectrum_basename}")

        except Exception as e:
            print(f"Error saving sequential result: {e}\n{traceback.format_exc()}")
    
    def plot_fitting_result(self, result):
        """
        Plot fitting result (single or simultaneous).
        
        Args:
            result: Dictionary with fitting results
        """
        try:
            # Get model info for plotting
            model_colors = self.params_table.get_current_colors()
            fitted_parameters = result['parameters']
            chi2 = result['chi2']
            chi2_spread = result.get('chi2_spread')
            is_simultaneous = result.get('is_simultaneous', False)
            instrumental_note = str(result.get('instrumental_note', '') or '').strip()
            note_suffix = f"\n{instrumental_note}" if instrumental_note else ""
            
            if is_simultaneous:
                # Simultaneous fitting - multiple spectra
                gridcolor = self.gridcolor
                
                # Store FS_pos for toggle functionality (from first spectrum)
                self.current_FS_pos = result.get('FS_pos_list', [[]])[0] if result.get('FS_pos_list') else []
                
                # Store data for replotting
                self.last_plot_data = {
                    'A_list': result['A_list'],
                    'B_list': result['B_list'],
                    'SPC_f_list': result['SPC_f_list'],
                    'hires_diff_list': result.get('hires_diff_list'),
                    'FS_list': result['FS_list'],
                    'FS_pos_list': result['FS_pos_list'],
                    'p': fitted_parameters,
                    'begining_spc': result['begining_spc'],
                    'model_colors': model_colors,
                    'chi2': chi2,
                    'chi2_spread': chi2_spread,
                    'spectrum_files': result['spectrum_files'],
                    'is_simultaneous': True,
                    'z_order': None,
                    'model': result.get('model', []),
                    'Distri': result.get('Distri_substituted', []),
                    'Cor': result.get('Cor_substituted', []),
                    'Recon': result.get('Recon', []),
                }
                self._showing_distribution = False
                self.SP_DI.setText('Distribution')

                # Store fitting data for graf.txt saving. Every entry is a LIST
                # with one item per spectrum; 'begining_spc' lets the save code
                # slice each section's own baseline parameters out of the flat p.
                self.last_fitting_data = {
                    'A': result['A_list'],  # List of arrays
                    'B': result['B_list'],
                    'SPC_f': result['SPC_f_list'],
                    'FS': result['FS_list'],
                    'is_simultaneous': True,
                    'begining_spc': result['begining_spc'],
                    'spectrum_files': result['spectrum_files'],
                }
                
                result_svg, result_png, position_artists_list = plot_simultaneous_fitting_result(
                    self.figure, result['A_list'], result['B_list'], result['SPC_f_list'],
                    result['FS_list'], result['FS_pos_list'], fitted_parameters,
                    result['begining_spc'], model_colors, chi2, result['spectrum_files'],
                    self.dir_path, z_order=None, gridcolor=gridcolor,
                    theme=self._theme, model=result.get('model'),
                    hires_diff_list=result.get('hires_diff_list'),
                    chi2_spread=chi2_spread
                )
                
                # Store position artists (from all subplots)
                self.position_artists = [artist for sublist in position_artists_list for artist in sublist]
                if self.position_artists:
                    self.toolbar.toggle_positions_action.setEnabled(True)
                    self.toolbar.toggle_positions_action.setChecked(True)
                else:
                    self.toolbar.toggle_positions_action.setEnabled(False)
                
                self._update_legend_toggle()
                # Update canvas
                self.canvas.draw()
                self.toolbar.push_current()
                
                self.set_status(f"Simultaneous fit completed! χ² = {chi2:.3f}{note_suffix}", "green")
                
            elif 'A' in result and 'B' in result and 'SPC_f' in result and 'FS' in result:
                # Single spectrum fitting
                gridcolor = self.gridcolor
                FS_pos = result.get('FS_pos', [])
                
                # Store FS_pos for toggle functionality
                self.current_FS_pos = FS_pos
                
                # Store fitting data for graf.txt saving
                self.last_fitting_data = {
                    'A': result['A'],
                    'B': result['B'],
                    'SPC_f': result['SPC_f'],
                    'FS': result['FS'],
                    'is_simultaneous': False,
                }
                
                # Store data for replotting with z-order changes
                self.last_plot_data = {
                    'A': result['A'],
                    'B': result['B'],
                    'SPC_f': result['SPC_f'],
                    'hires_diff': result.get('hires_diff'),
                    'FS': result['FS'],
                    'FS_pos': FS_pos,
                    'p': fitted_parameters,
                    'model_colors': model_colors,
                    'chi2': chi2,
                    'chi2_spread': chi2_spread,
                    'filepath': result['spectrum_file'],
                    'is_simultaneous': False,
                    'z_order': None,
                    'model': result.get('model', []),
                    'Distri': result.get('Distri_substituted', []),
                    'Cor': result.get('Cor_substituted', []),
                    'Recon': result.get('Recon', []),
                }
                self._showing_distribution = False
                self.SP_DI.setText('Distribution')

                result_svg, result_png, position_artists = plot_fitting_result(
                    self.figure, result['A'], result['B'], result['SPC_f'], result['FS'],
                    FS_pos, fitted_parameters, model_colors, chi2, result['spectrum_file'],
                    self.dir_path, z_order=None, gridcolor=gridcolor,
                    theme=self._theme, model=result.get('model'),
                    hires_diff=result.get('hires_diff'),
                    chi2_spread=chi2_spread
                )
                
                # Store position artists and enable toggle button if positions exist
                self.position_artists = position_artists
                if position_artists:
                    self.toolbar.toggle_positions_action.setEnabled(True)
                    self.toolbar.toggle_positions_action.setChecked(True)
                else:
                    self.toolbar.toggle_positions_action.setEnabled(False)
                
                self._update_legend_toggle()
                # Update canvas
                self.canvas.draw()
                self.toolbar.push_current()
                
                self.set_status(f"Fit completed! χ² = {chi2:.3f}{note_suffix}", "green")
            else:
                self.set_status(f"Fit completed! χ² = {chi2:.3f} (no plot data){note_suffix}", "green")
                
        except Exception as e:
            print(f"Error plotting result: {e}\n{traceback.format_exc()}")
            self.set_status(f"Plot error: {e}", "orange")
    
    def on_fitting_finished(self, result):
        """Handle fitting completion"""
        try:
            if not result['success']:
                # fit_single_spectrum puts the whole traceback in 'message'. Print
                # it like every other error handler here: the status box can only
                # show the first line, and the terminal is what the bug reporter
                # watches.
                print(result['message'])
                self.set_status(f"Fitting failed: {result['message']}", "red")
                return
            
            # Read model configuration for results table
            model_list = self.params_table.get_model_list()
            model_colors = self.params_table.get_current_colors()
            parameter_names = self.params_table.get_parameter_names()
            expression_texts = self.params_table.get_expression_texts()
            expression_texts = {**expression_texts, **_recon_weight_texts(model_list, result.get('Recon', []))}

            # Extract results
            fitted_parameters = result['parameters']
            errors = result['errors']
            chi2 = result['chi2']
            covariance_matrix = result['covariance_matrix']
            fix = result.get('fix', np.array([], dtype=int))
            is_simultaneous = result.get('is_simultaneous', False)
            
            chi2_spread = result.get('chi2_spread')
            print(f"[Fit] Fitting successful!")
            if chi2_spread is not None:
                print(f"[Fit] chi^2 = {chi2:.3f} ± {chi2_spread:.3f}")
            else:
                print(f"[Fit] chi^2 = {chi2:.3f}")
            print(f"[Fit] Simultaneous: {is_simultaneous}")
            
            # Update results table
            self.results_table.fill_table(
                fitted_parameters,
                model_list,
                model_colors,
                parameter_names,
                covariance_matrix,
                errors,
                fix,
                expression_texts
            )
            self.results_table.current_links = dict(self._fit_links_snapshot)
            # ... and the model it was fitted with, fitted values written in
            self.results_table.current_model_rows = fitted_model_rows(
                self._fit_model_snapshot, fitted_parameters, result.get('Recon', []))

            # Store chi2 for saving
            self.results_table.current_chi2 = chi2
            
            # Plot results
            self.plot_fitting_result(result)
            
        except Exception as e:
            error_msg = f"Error processing fit results: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.set_status(f"Error processing fit results: {e}", "red")
        finally:
            self.inprogress = False
    
    def on_fitting_error(self, error_msg):
        """Handle fitting error"""
        self.set_status(f"Fitting error: {error_msg}", "red")
        self.inprogress = False

    def _result_base_path(self):
        """The base every result file is named from: the save path without the
        spectrum's extension (``Fe_4.2K.dat`` -> ``Fe_4.2K_param.txt``, ...)."""
        return strip_known_extension(self.save_path.text().strip())

    def save_result_pressed(self):
        """Save fitting results to file"""
        # Check if we have results to save
        if not hasattr(self.results_table, 'current_parameters') or self.results_table.current_parameters is None:
            self.set_status("No results to save. Please run fitting first.", "red")
            return
        
        # Check if save path is set
        if not self.save_path.text().strip():
            self.set_status("Please specify save path", "orange")
            return
        
        # Check if file exists and ask user
        param_file = self._result_base_path() + '_param.txt'
        if os.path.exists(param_file):
            reply = QMessageBox.question(
                self, 'File exists',
                f"File {os.path.basename(param_file)} already exists.\n\n"
                "Save: _params.txt will be appended, others overwritten\n"
                "Discard: Overwrite all files\n"
                "Cancel: Do nothing",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                self.set_status("Saving canceled", "orange")
                return
            elif reply == QMessageBox.StandardButton.Save:
                mode = 'append'  # Append to parameter file
            else:
                mode = 'overwrite'  # Overwrite all
        else:
            mode = 'new'
        
        self._save_result_files(mode)
    
    def save_result_as_pressed(self):
        """Save fitting results with file chooser"""
        # Check if we have results to save
        if not hasattr(self.results_table, 'current_parameters') or self.results_table.current_parameters is None:
            self.set_status("No results to save. Please run fitting first.", "red")
            return
        
        # Open file dialog
        save_dir = self.workfolder if self.workfolder else os.path.dirname(__file__)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save results as. Careful: it overwrites existing files without confirmation", save_dir, "All files (*)", options=QFileDialog.Option.DontConfirmOverwrite
        )
        
        if file_path:
            # Remove extension if user added one — only a real one, so a typed
            # name like Fe_4.2K is kept whole
            base_path = strip_known_extension(file_path)
            self.save_path.setText(base_path)
            self._save_result_files('new')
        else:
            self.set_status("Saving canceled", "orange")
    
    def _save_result_files(self, mode):
        """
        Save all result files, then the fitted model.

        mode: 'new', 'append', or 'overwrite' — applies to the result files only.
        The model goes to ``<base>_result_model.mdl`` (its own name, so it can
        never overwrite the model file the user is working on): the model the
        fit started from with the fitted values written in, links, fixes, bounds
        and expressions as they were fitted. It asks its own overwrite question.
        Sequential fitting bypasses this and calls ``_write_result_files``
        directly, so it never pops a dialog per spectrum.
        """
        try:
            base_path = self._result_base_path()

            # Create directory if needed
            save_dir = os.path.dirname(base_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # A simultaneous fit produced ONE parameter row for several spectra —
            # name them all, otherwise the saved row looks like a single-file fit.
            fit_data = self.last_fitting_data or {}
            sim_files = fit_data.get('spectrum_files') if fit_data.get('is_simultaneous') else None
            if sim_files:
                spectrum_file = '; '.join(os.path.basename(f) for f in sim_files)
            else:
                spectrum_file = os.path.basename(self.path_list[0]) if self.path_list else "unknown"
            self._write_result_files(base_path, spectrum_file, mode)

            if mode == 'append':
                message = "Results appended to parameter file, others overwritten"
            else:
                message = "Results saved successfully"

            # A result is only reproducible together with the model that produced
            # it, so the fitted model (taken at fit start, whatever the table
            # holds by now) is written next to the other artifacts. It runs its
            # own, independent overwrite question: answering "No" there must
            # leave the already-written result files alone.
            model_rows = self.results_table.current_model_rows
            model_path = result_model_path(base_path)
            if model_rows is None:
                self.set_status(f"{message}. Model NOT saved: no fitted model with this result", "orange")
            elif save_result_model(self, model_path, model_rows):
                self.set_status(f"{message}; model saved as {os.path.basename(model_path)}", "green")
            else:
                # Keep save_result_model's own reason (canceled / could not
                # write) — it is the only place it was reported.
                reason = self.log.toPlainText().strip() or "canceled"
                self.set_status(f"{message}. Model NOT saved: {reason}", "orange")

        except Exception as e:
            print(f"Error saving results: {e}\n{traceback.format_exc()}")
            self.set_status(f"Error saving results: {e}", "red")

    def _write_result_files(self, base_path, spectrum_file, mode):
        """Write the four result artifacts of the current fit next to *base_path*.

        Shared by "Save result"/"Save result as" and by sequential fitting:
        ``<base>_param.txt`` (parameters + errors; appended in 'append' mode),
        ``<base>_graf.txt`` (the plotted curves), ``<base>_combo.png`` (figure +
        rendered results table) and ``<base>.svg`` (copy of the last figure the
        plot functions saved into the app directory).
        """
        # 1. Parameters + errors table
        self._save_parameters_file(base_path + '_param.txt',
                                   self.results_table.current_parameters,
                                   self.results_table.current_errors,
                                   self.results_table.current_model_list,
                                   self.results_table.current_parameter_names,
                                   spectrum_file,
                                   self.results_table.current_chi2,
                                   mode)

        # 2. Graph data from the fitting arrays
        if self.last_fitting_data:
            self._save_graf_file(base_path + '_graf.txt', self.last_fitting_data)

        # 3. Combo image (figure png rendered by the plot functions + table)
        result_png_src = os.path.join(self.dir_path, 'result.png')
        table_image = self.results_table.render_table_to_image()
        if os.path.exists(result_png_src):
            self._save_combo_image_from_qimage(result_png_src, table_image,
                                               base_path + '_combo.png')

        # 4. Copy SVG
        result_svg_src = os.path.join(self.dir_path, 'result.svg')
        if os.path.exists(result_svg_src):
            shutil.copyfile(result_svg_src, base_path + '.svg')

        # 5. Distributions image (all Distr/Corr/Recon curves in one PNG), only
        #    when the model has a distribution. No separate SVG.
        self._save_distributions_png(base_path + '_distributions.png')

    def _save_distributions_png(self, filepath):
        """Render every distribution/correlation (Distr, Corr, Recon) into one PNG.

        Reuses :func:`plot_distribution` (the same view as the Distribution toggle)
        on a stand-alone Agg figure. Writes nothing when the model has no
        distribution; failures are logged but never block the other result files.
        """
        data = getattr(self, 'last_plot_data', None)
        if not data:
            return
        model = data.get('model', []) or []
        Distri = data.get('Distri', []) or []
        if not Distri and 'Recon' not in model:
            return  # no distribution to draw
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            fig = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig)
            ok = plot_distribution(
                fig, model, data.get('p_flat', data.get('p')),
                Distri, data.get('Cor', []),
                self.params_table.get_parameter_names(),
                model_colors=data.get('model_colors'),
                gridcolor=self.gridcolor, theme=self._theme,
                Recon=data.get('Recon', []),
            )
            if ok:
                fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
        except Exception as e:
            print(f"Error saving distributions png: {e}\n{traceback.format_exc()}")

    def _graf_section_columns(self, A, B, SPC_f, FS, p_section, section_model, prefix=''):
        """Build the (names, columns) of one spectrum's block of the graf file.

        The block is ``Velocity Data Baseline Fit <subspectrum> ...``, every name
        carrying *prefix* (empty for a single spectrum, ``S<n>_`` per section of a
        simultaneous fit). ``p_section`` is the parameter array whose FIRST eight
        slots are this spectrum's baseline; ``section_model`` the model names of
        this section only.
        """
        A = np.asarray(A, dtype=float)
        baseline = (calculate_baseline(p_section, A) if p_section is not None and len(p_section) >= 8
                    else np.zeros_like(A))

        # Names of the actually plotted subspectra only. Keep this aligned with the
        # plotting filters: 'Layer' is a boundary marker that draws no curve of its
        # own (FS has no entry for it), so it is excluded here just like
        # Nbaseline/Distr/Corr/Recon/Expression/Variables.
        excluded = {'baseline', 'Nbaseline', 'Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables'}
        model_names = [m for m in section_model if m not in excluded]
        while len(model_names) < len(FS):
            model_names.append(f'Submodel{len(model_names) + 1}')

        names = ['Velocity', 'Data', 'Baseline', 'Fit'] + model_names[:len(FS)]
        columns = [A, B, baseline, SPC_f] + list(FS)
        return [prefix + n for n in names], columns

    def _save_graf_file(self, filepath, fitting_data):
        """
        Save graph data to text file with columns: A (velocity), B (data), Baseline, SPC_f (fit), model1, model2, ...

        For a simultaneous (Nbaseline) fit the same block is written for EVERY
        spectrum, each column prefixed ``S<n>_`` (``S1_Velocity``, ``S1_Baseline``,
        ``S1_Fit``, ..., ``S2_Velocity``, ...), so all baselines, all fits and all
        subspectra land in the one file the user plots later. The spectra usually
        differ in length, so every column is NaN-padded to the longest one.

        Args:
            filepath: Path to save the file
            fitting_data: Dictionary with 'A', 'B', 'SPC_f', 'FS' arrays. For a
                simultaneous fit these are LISTS with one entry per spectrum and
                'is_simultaneous'/'begining_spc' are set as well.
        """
        p_all = getattr(self.results_table, 'current_parameters', None)
        model_list = list(getattr(self.results_table, 'current_model_list', None) or [])

        if fitting_data.get('is_simultaneous'):
            # Per-spectrum blocks: each section has its OWN baseline parameters,
            # which start at begining_spc[i] in the flat fitted array.
            sections = fitting_io.split_model_sections(model_list)
            begining_spc = list(fitting_data.get('begining_spc') or [])
            column_names, data_columns = [], []
            for i in range(len(fitting_data['A'])):
                if p_all is not None and i < len(begining_spc):
                    p_section = np.asarray(p_all, dtype=float)[begining_spc[i]:]
                else:
                    p_section = None
                names, columns = self._graf_section_columns(
                    fitting_data['A'][i], fitting_data['B'][i],
                    fitting_data['SPC_f'][i], fitting_data['FS'][i],
                    p_section, sections[i] if i < len(sections) else [],
                    prefix=f'S{i + 1}_')
                column_names.extend(names)
                data_columns.extend(columns)
        else:
            column_names, data_columns = self._graf_section_columns(
                fitting_data['A'], fitting_data['B'], fitting_data['SPC_f'],
                fitting_data['FS'], p_all, model_list)

        # Append distribution / correlation curves as extra x/y column pairs
        # (Distr_x_N/Distr_y_N, Corr_x_N/Corr_y_N, Recon_x_N/Recon_y_N in model
        # order). Emitted once for the whole model — distribution_curves walks the
        # flat parameter array with mod_len_def, so it crosses Nbaseline sections
        # correctly. Their length is the grid size Num, generally != the spectrum
        # length, so every column is padded with NaN to a common row count.
        # Guarded so a failure here never blocks the rest of the graf file.
        try:
            data = getattr(self, 'last_plot_data', None)
            if data is not None:
                curves = distribution_curves(
                    data.get('model', []), data.get('p_flat', data.get('p')),
                    data.get('Distri', []), data.get('Cor', []), data.get('Recon', []))
                for name, values in curves:
                    column_names.append(name)
                    data_columns.append(np.asarray(values, dtype=float))
        except Exception as e:
            print(f"[graf] could not add distribution columns: {e}")

        # Pad every column to the longest length with NaN, then stack.
        n_rows = max(len(np.atleast_1d(c)) for c in data_columns)
        padded = []
        for c in data_columns:
            c = np.asarray(c, dtype=float).ravel()
            if len(c) < n_rows:
                c = np.concatenate([c, np.full(n_rows - len(c), np.nan)])
            padded.append(c)
        data_array = np.column_stack(padded)

        # Build header
        header = '\t'.join(column_names)

        # Save with tab separation
        np.savetxt(filepath, data_array, delimiter='\t', fmt='%.6e',
                   header=header, comments='')
    
    def _save_parameters_file(self, filepath, parameters, errors, model_list, 
                             parameter_names, spectrum_file, chi2, mode):
        """Save parameters and errors to text file with proper names and model info"""
        # Build header row with model names and parameter/error pairs
        names = ['#File']
        values = []
        
        # Process each model component
        param_idx = 0
        for comp_idx, (model_name, param_names) in enumerate(zip(model_list, parameter_names)):
            # Add model column
            names.append('model')
            values.append(model_name)
            
            # Add each parameter and its error
            for param_name in param_names:
                # Add parameter name
                names.append(param_name)
                if model_name == 'Recon' and param_name == 'weights':
                    # The reconstruction weight vector is not a scalar parameter
                    # (it is stored with the model / shown in the Distribution plot);
                    # write a placeholder 1 so the column stays aligned.
                    values.append(1)
                elif param_idx < len(parameters):
                    values.append(parameters[param_idx])
                else:
                    values.append('')

                # Add error name and value
                names.append(f'd_{param_name}')
                if model_name == 'Recon' and param_name == 'weights':
                    values.append('nan')
                elif errors is not None and param_idx < len(errors):
                    values.append(errors[param_idx])
                else:
                    values.append('nan')

                param_idx += 1
        
        # Add chi2
        names.append('χ²')
        values.append(chi2)
        
        # Write to file
        if mode == 'append':
            # Read existing header
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_line = f.readline().rstrip()
                existing_names = re.split(r'\t+', first_line)
            except:
                existing_names = []
            
            # Append mode
            with open(filepath, 'a', encoding='utf-8') as f:
                if existing_names != names:
                    # Write header if different
                    f.write('\t'.join(map(str, names)) + '\n')
                # Write data
                f.write(spectrum_file + '\t' + '\t'.join(map(str, values)) + '\n')
        else:
            # New or overwrite mode
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header
                f.write('\t'.join(map(str, names)) + '\n')
                # Write data
                f.write(spectrum_file + '\t' + '\t'.join(map(str, values)) + '\n')
    
    def _save_combo_image_from_qimage(self, plot_path, table_qimage, output_path):
        """Combine plot image file and table QImage and save"""
        # Load plot image
        im1 = matplotlib.image.imread(plot_path)
        
        # Convert QImage to numpy array
        width = table_qimage.width()
        height = table_qimage.height()
        
        # Convert to format compatible with numpy
        table_qimage = table_qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        
        # Get bytes and convert to numpy array
        ptr = table_qimage.constBits()
        nbytes = table_qimage.sizeInBytes()
        # PySide6 (current/newer) returns a buffer/memoryview here.
        im2 = np.frombuffer(ptr, dtype=np.uint8, count=nbytes).reshape((height, width, 4)).copy()
        
        # Convert im1 to uint8 if it's float
        if im1.dtype == np.float32 or im1.dtype == np.float64:
            im1 = (im1 * 255).astype(np.uint8)
        
        # Ensure both images have 4 channels (RGBA)
        if im1.shape[2] == 3:
            # Add alpha channel
            alpha = np.ones((im1.shape[0], im1.shape[1], 1), dtype=np.uint8) * 255
            im1 = np.concatenate([im1, alpha], axis=2)
        
        # Resize images to have the same width (maintain aspect ratio)
        target_width = max(im1.shape[1], im2.shape[1])
        
        # Helper: resize a numpy RGBA image using QImage smooth scaling
        def _qimage_resize(arr, new_w, new_h):
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, w * ch, QImage.Format.Format_RGBA8888)
            qimg = qimg.scaled(new_w, new_h, Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
            ptr = qimg.constBits()
            nbytes = qimg.sizeInBytes()
            return np.frombuffer(ptr, dtype=np.uint8, count=nbytes).reshape((new_h, new_w, 4)).copy()
        
        # Resize im1 if needed
        if im1.shape[1] != target_width:
            scale_factor = target_width / im1.shape[1]
            new_height = int(im1.shape[0] * scale_factor)
            im1 = _qimage_resize(im1, target_width, new_height)
        
        # Resize im2 if needed
        if im2.shape[1] != target_width:
            scale_factor = target_width / im2.shape[1]
            new_height = int(im2.shape[0] * scale_factor)
            im2 = _qimage_resize(im2, target_width, new_height)
        
        # Stack vertically
        combo_image = np.concatenate((im1, im2), axis=0)
        matplotlib.image.imsave(output_path, combo_image)
