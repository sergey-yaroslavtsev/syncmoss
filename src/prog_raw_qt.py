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
import signal
import os
import re
import os.path
import numpy as np
from numpy import *
import builtins as bu
def max(*args):
    return bu.max(*args)
def min(*args):
    return bu.min(*args)
import minimi_lib as mi
from models import TI
from models_positions import mod_pos
from Calibration import Calibration
import multiprocessing as mp
# import dual_v3 as dn
import Instrumental as ins
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.image
from matplotlib import colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class CustomNavigationToolbar(NavigationToolbar):
    """Custom matplotlib toolbar with toggle button for model line positions."""
    
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.parent_window = parent
        
        # Add separator
        self.addSeparator()
        
        # Add toggle positions button
        self.toggle_positions_action = self.addAction('Toggle\nPositions', self.toggle_positions)
        self.toggle_positions_action.setCheckable(True)
        self.toggle_positions_action.setChecked(True)  # Start as visible
        self.toggle_positions_action.setToolTip('Show/Hide model line positions')
        self.toggle_positions_action.setEnabled(False)  # Disabled by default
    
    def toggle_positions(self):
        """Toggle visibility of model line positions."""
        if hasattr(self.parent_window, 'toggle_position_markers'):
            show = self.toggle_positions_action.isChecked()
            self.parent_window.toggle_position_markers(show)
import threading
import queue
import time
from functools import partial
import shutil
import platform
# import asyncio
# import psutil
import base64
import warnings
import copy
from multiprocessing.pool import ThreadPool
import gc

warnings.filterwarnings('ignore', '.*object is not callable.*', )

plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
plt.rcParams['axes.labelcolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'w'
plt.rcParams['xtick.color'] = 'w'
plt.rcParams['ytick.color'] = 'w'

path = None
params = None
image = None
initial = 0

# Constants
from constants import numro, numco, model_colors, number_of_baseline_parameters

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QScrollArea, QGridLayout, QSplitter, QFrame, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QSpinBox, QDoubleSpinBox,
    QMenu, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon, QAction, QDoubleValidator, QRegularExpressionValidator
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize, QLocale, QRegularExpression
import sys
import os

from parameters_table import ParametersTable
from results_table import ResultsTable
from model_io import load_model, read_model, save_model, save_model_as, mod_len_def
from spectrum_io import load_spectrum, sum_all_spectra, subtract_model_from_spectrum, half_points, calculate_backgrounds
from spectrum_plotter import plot_calibration, plot_model, plot_model_with_nbaseline, plot_spectrum
from instrumental_io import instrumental
import traceback
import ast

MulCoCMS = 0.28

check_tango = False
# from Pytango import DeviceProxy
# def get_data(tango_uri):
#     proxy = DeviceProxy(tango_uri)
#     return proxy.data
# tango_uri = 'moesa:20000/id14/Can556/6a2' # could be different
# check_tango = True

warnings.filterwarnings('ignore', '.*object is not callable.*', )

# plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
# plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
# plt.rcParams['axes.labelcolor'] = 'w'
# plt.rcParams['axes.edgecolor'] = 'w'
# plt.rcParams['xtick.color'] = 'w'
# plt.rcParams['ytick.color'] = 'w'

class CalibrationThread(QThread):
    """Thread for running calibration without blocking the UI"""
    finished = Signal(object, object, object)  # A, B, C
    error = Signal(str)
    
    def __init__(self, dir_path, file, experimental_method, INS, JN, x0, MulCo, vel_start, pool):
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
    
    def run(self):
        try:
            # Use the global pool (passed to constructor)
            pool = self.pool
            
            # Run calibration
            A, B, C = Calibration(
                self.dir_path, self.file, pool, 
                self.experimental_method, self.INS, self.JN, 
                self.x0, self.MulCo, self.vel_start
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
            result = instrumental(self.main_window, self.ref, self.mode, pool=self.pool)
            print(f"[DEBUG] InstrumentalThread completed successfully")
            self.finished.emit(result)
        except Exception as e:
            import traceback
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
            import fitting_io
            print(f"[DEBUG] FittingThread started for: {self.spectrum_file}")
            result = fitting_io.fit_single_spectrum(self.main_window, self.spectrum_file, self.pool)
            print(f"[DEBUG] FittingThread completed successfully")
            self.finished.emit(result)
        except Exception as e:
            import traceback
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
        import fitting_io
        
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
                    
            except Exception as e:
                import traceback
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


class ShowModelThread(QThread):
    """Thread for running show model calculation without blocking the UI"""
    finished = Signal(object, object, object, object, object, object, object, object, object)  # A, B, SPC_f, FS, FS_pos, p, model, has_nbaseline, backgrounds
    error = Signal(str)
    
    def __init__(self, main_window, path_list, pool):
        super().__init__()
        self.main_window = main_window
        self.path_list = path_list
        self.pool = pool
    
    def run(self):
        try:
            # Read model from parameter table
            model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN = read_model(self.main_window)
            
            # Validate Nbaseline count matches number of spectra
            num_nbaseline = model.count('Nbaseline')
            num_spectra = len(self.path_list)
            if num_nbaseline > 0:
                if num_nbaseline != num_spectra - 1:
                    self.error.emit(f"Error: Found {num_nbaseline} Nbaseline model(s) but have {num_spectra} spectrum/spectra. "
                                  f"Need exactly {num_spectra - 1} Nbaseline(s) for {num_spectra} spectra (or 0 for single spectrum).")
                    return
            
            # Allow showing just baseline (empty model list is OK)
            # if len(model) == 0:
            #     self.error.emit("No model defined. Please add model components.")
            #     return
            
            # Apply expressions
            for i in range(len(NExpr)):
                try:
                    p[NExpr[i]] = eval(Expr[i])
                except Exception as e:
                    print(f"Error evaluating expression {Expr[i]}: {e}")
            
            # Apply constraints
            for i in range(len(con1)):
                p[int(con1[i])] = p[int(con2[i])] * con3[i]
            
            # Check if we have Nbaseline models (multiple spectra case)
            num_nbaseline = model.count('Nbaseline')
            
            # Calculate backgrounds for dual y-axis support
            backgrounds = calculate_backgrounds(self.path_list, self.main_window.calibration_path)
            
            if num_nbaseline > 0:
                # Multiple spectra case - handle each spectrum section separately
                # Load all spectra separately (don't concatenate yet)
                A_list = []
                B_list = []
                for i in range(len(self.path_list)):
                    file = os.path.abspath(self.path_list[i])
                    A_temp, B_temp = load_spectrum(self.main_window, [file], calibration_path=self.main_window.calibration_path)
                    if not A_temp or not B_temp:
                        self.error.emit(f"Could not load spectrum {i+1}: {file}")
                        return
                    A_list.append(A_temp[0])
                    B_list.append(B_temp[0])
                
                # Concatenate for main model calculation
                A_combined = A_list[0]
                B_combined = B_list[0]
                for i in range(1, len(A_list)):
                    A_combined = np.concatenate((A_combined, A_list[i]))
                    B_combined = np.concatenate((B_combined, B_list[i]))
                
                A = A_combined
                B = B_combined
            else:
                # Single spectrum case
                file = os.path.abspath(self.path_list[0])
                A_list, B_list = load_spectrum(self.main_window, [file], calibration_path=self.main_window.calibration_path)
                if not A_list or not B_list:
                    self.error.emit("Could not load spectrum")
                    return
                
                A = A_list[0]
                B = B_list[0]
            
            # Get experimental method parameters
            JN = int(self.main_window.jn0_input.text())
            experimental_method = 1 if self.main_window.MS_fit.isChecked() else 3
            
            pool = self.pool
            
            # Setup parameters based on experimental method
            if experimental_method == 1:  # MS
                INS = float(self.main_window.GCMS_input.text())
                pNorm = np.array([float(0)] * number_of_baseline_parameters)
                pNorm[0] = 1
                Norm = TI(np.array([float(1000)]), pNorm, [], JN, pool, 0, self.MulCoCMS, INS, [0], [0], Met=1)[0]
                method_params = {
                    'x0': 0.0,
                    'MulCo': self.MulCoCMS,
                    'INS': INS,
                    'Met': 1,
                    'Norm': Norm
                }
            else:  # SMS (experimental_method == 3)
                INS = self.main_window.INS
                pNorm = np.array([float(0)] * number_of_baseline_parameters)
                pNorm[0] = 1
                Norm = TI(np.array([float(1000)]), pNorm, [], JN, pool, self.main_window.x0, self.main_window.MulCo, INS, [0], [0])[0]
                method_params = {
                    'x0': self.main_window.x0,
                    'MulCo': self.main_window.MulCo,
                    'INS': INS,
                    'Met': 0,
                    'Norm': Norm
                }
            
            if num_nbaseline > 0:
                # Multiple spectra case - calculate subspectra for each section separately
                # Split model at Nbaseline boundaries
                model_sections = []
                start_idx = 0
                for i, m in enumerate(model):
                    if m == 'Nbaseline':
                        model_sections.append(model[start_idx:i])
                        start_idx = i + 1
                model_sections.append(model[start_idx:])
                
                # Find parameter boundaries for each section
                # First section starts at 0 (includes main baseline)
                # Each Nbaseline marks the start of a new section with its own baseline
                begining_spc = [0]
                param_idx = number_of_baseline_parameters  # Start after main baseline
                
                for i in range(len(model)):
                    if model[i] == 'Nbaseline':
                        begining_spc.append(param_idx)
                        param_idx += number_of_baseline_parameters  # Nbaseline has number_of_baseline_parameters parameters
                    else:
                        # Count parameters for this model
                        param_idx += mod_len_def(model[i], include_special=True)
                
                # Calculate spectrum for each section separately, then concatenate
                SPC_f_sections = []
                FS_all = []
                FS_pos_all = []
                p_all = []
                
                for spc_idx in range(len(model_sections)):
                    # Get parameters for this section
                    if spc_idx < len(begining_spc) - 1:
                        p_section = p[begining_spc[spc_idx]:begining_spc[spc_idx + 1]]
                    else:
                        p_section = p[begining_spc[spc_idx]:]
                    p_all.append(p_section)
                    
                    # Get spectrum section
                    A_section = A_list[spc_idx]
                    
                    # Calculate fitted spectrum for this section
                    SPC_f_section = TI(A_section, p_section, model_sections[spc_idx], JN, pool, 
                                         method_params['x0'], method_params['MulCo'], 
                                         method_params['INS'], Distri, Cor, 
                                         Met=method_params['Met'], Norm=method_params['Norm'])
                    SPC_f_sections.append(SPC_f_section)
                    
                    # Create subspectra for this section
                    Ps, Psm, Distri_t, Cor_t = self.main_window.create_subspectra(model_sections[spc_idx], Distri, Cor, p_section, number_of_baseline_parameters)[:4]
                    
                    # Calculate each subspectrum for this section
                    FS = []
                    FS_pos = []
                    CoEn = 0
                    DiEn = 0
                    
                    for i in range(len(Ps)):
                        CoSt = CoEn
                        DiSt = DiEn
                        CoEn += Psm[i].count('Corr')
                        DiEn += Psm[i].count('Distr')
                        
                        distri_slice = Distri_t[DiSt:DiEn] if DiEn > DiSt else [0]
                        cor_slice = Cor_t[CoSt:CoEn] if CoEn > CoSt else [0]
                        
                        FS_i = TI(A_section, Ps[i], Psm[i], JN, pool, method_params['x0'], method_params['MulCo'], 
                                   method_params['INS'], distri_slice, cor_slice, 
                                   Met=method_params['Met'], Norm=method_params['Norm'])
                        FS.append(FS_i)
                        
                        if experimental_method == 1:
                            FS_pos.append(mod_pos(Ps[i], Psm[i], method_params['INS'], Met=1))
                        else:
                            FS_pos.append(mod_pos(Ps[i], Psm[i], method_params['INS']))
                    
                    FS_all.append(FS)
                    FS_pos_all.append(FS_pos)
                
                # Concatenate fitted spectrum sections
                SPC_f = SPC_f_sections[0]
                for i in range(1, len(SPC_f_sections)):
                    SPC_f = np.concatenate((SPC_f, SPC_f_sections[i]))
                
                # Emit with lists of subspectra for each section
                self.finished.emit(A, B, SPC_f, FS_all, FS_pos_all, p_all, model, True, backgrounds)
            else:
                # Single spectrum case
                # Calculate full spectrum
                SPC_f = TI(A, p, model, JN, pool, method_params['x0'], method_params['MulCo'], 
                             method_params['INS'], Distri, Cor, Met=method_params['Met'], Norm=method_params['Norm'])
                
                # Create subspectra as before
                Ps, Psm, Distri_t, Cor_t = self.main_window.create_subspectra(model, Distri, Cor, p, number_of_baseline_parameters)[:4]
                
                # Filter out Nbaseline models from subspectra
                Ps_filtered = []
                Psm_filtered = []
                for i in range(len(Psm)):
                    if 'Nbaseline' not in Psm[i]:
                        Ps_filtered.append(Ps[i])
                        Psm_filtered.append(Psm[i])
                
                # Calculate each subspectrum
                FS = []
                FS_pos = []
                CoEn = 0
                DiEn = 0
                
                for i in range(len(Ps_filtered)):
                    CoSt = CoEn
                    DiSt = DiEn
                    CoEn += Psm_filtered[i].count('Corr')
                    DiEn += Psm_filtered[i].count('Distr')
                    
                    distri_slice = Distri_t[DiSt:DiEn] if DiEn > DiSt else [0]
                    cor_slice = Cor_t[CoSt:CoEn] if CoEn > CoSt else [0]
                    
                    FS_i = TI(A, Ps_filtered[i], Psm_filtered[i], JN, pool, method_params['x0'], method_params['MulCo'], 
                               method_params['INS'], distri_slice, cor_slice, 
                               Met=method_params['Met'], Norm=method_params['Norm'])
                    FS.append(FS_i)
                    
                    if experimental_method == 1:
                        FS_pos.append(mod_pos(Ps_filtered[i], Psm_filtered[i], method_params['INS'], Met=1))
                    else:
                        FS_pos.append(mod_pos(Ps_filtered[i], Psm_filtered[i], method_params['INS']))
                
                # Emit with single list of subspectra
                self.finished.emit(A, B, SPC_f, FS, FS_pos, p, model, False, backgrounds)
            
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
                if error_count == 0:
                    self.finished.emit(f"Successfully converted {converted_count} RAW file(s) to .dat format")
                else:
                    self.finished.emit(f"Converted {converted_count} file(s), {error_count} failed")
            else:
                self.error.emit("No RAW files were successfully converted")
                
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Conversion failed: {str(e)}")


class PhysicsApp(QMainWindow):
    def __init__(self, pool=None):
        super().__init__()
        self.pool = pool  # Store reference to global pool
        self.setWindowTitle('SYNCMoss ESRF ID14')
        self.setGeometry(50, 50, 1600, 900)
        self.setMinimumSize(1270, 710)

        # Icon
        icon_path = os.path.join(os.getcwd(), "icon_r.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else: 
            print("Icon file not found:", icon_path)

        # Initialize variables
        self.MulCoCMS = 0.28
        self.current_spectrum_background = None  
        self.dir_path = os.path.dirname(__file__)
        self.workfolder = None  # Start with no workfolder selected
        self.workfolder_check = 1
        self.gridcolor = 'w'
        self.check_points_match = False
        self.newfilename = str('')
        self.newfilename2 = str('')
        self.BGcolor = 'k'
        self.calibration_path = os.path.join(self.dir_path, "Calibration.dat")
        self.points_match = True
        self.path_list = []
        self.backgrounds = []  # List to store calculated backgrounds
        self.sequence_fitting_type = 0  # 0 = initial, 1 = result
        self.x0 = 0.0
        self.MulCo = 0.0

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
        self.loadmod_btn.setStyleSheet("background-color: rgb(127, 255, 255); color: black;")
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
        self.cal_cho_btn.setStyleSheet("background-color: rgb(63, 127, 127); color: white;")
        self.cal_cho_btn.clicked.connect(self.choose_calibration_file)

        # Calibration group
        cal_frame = QFrame()
        cal_frame.setStyleSheet("QFrame { border: 2px solid gray; border-radius: 5px; }")
        cal_layout = QHBoxLayout(cal_frame)
        cal_layout.setContentsMargins(5, 5, 5, 5)

        self.cal_cho_title = QLabel("Velocity\ndown-up:")
        self.cal_cho_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cal_cho_title.setFont(QFont('Arial', 16))
        self.cal_cho_title.setStyleSheet("color: red; background-color: transparent; border: none;")

        self.velocity_btn = QPushButton()
        self.velocity_btn.setFlat(True)
        self.velocity_btn.setIconSize(QSize(32, 32))
        self.velocity_btn.setStyleSheet("QPushButton { border: none; }")
        self.velocity_btn.clicked.connect(self.toggle_velocity_direction)
        self.velocity_direction = False  # False for down-up, True for up-down
        self.update_velocity_label()  # Set initial state

        cal_layout.addWidget(self.cal_cho_title)
        cal_layout.addWidget(self.velocity_btn)

        self.cal_btn = QPushButton("Calibrate")
        self.cal_btn.setFont(QFont('Arial', 16))
        self.cal_btn.setStyleSheet("background-color: rgb(127, 255, 255); color: black;")
        self.cal_btn.clicked.connect(self.calibration)

        self.vel_btn = QPushButton("RAW\nto\ndat")
        self.vel_btn.setFont(QFont('Arial', 16))
        self.vel_btn.setStyleSheet("background-color: rgb(127, 255, 255); color: black;")
        self.vel_btn.clicked.connect(self.raw_to_dat)

        self.interrupt_btn = QPushButton('! INTERRUPT !')
        self.interrupt_btn.setFont(QFont('Arial', 16))
        self.interrupt_btn.setStyleSheet("background-color: red; color: white;")
        self.interrupt_btn.clicked.connect(self.interrupt)

        top_controls.addWidget(self.loadmod_btn)
        top_controls.addWidget(self.btncleanmodel)
        # top_controls.addWidget(self.switch)  # NFS switch removed
        top_controls.addWidget(self.cal_cho_btn)
        top_controls.addWidget(cal_frame)
        top_controls.addWidget(self.cal_btn)
        top_controls.addWidget(self.vel_btn)
        top_controls.addWidget(self.interrupt_btn)

        left_layout.addLayout(top_controls)

        # Parameters table
        self.params_table = ParametersTable(self)
        scroll_params = QScrollArea()
        scroll_params.setWidget(self.params_table)
        scroll_params.setWidgetResizable(True)
        left_layout.addWidget(scroll_params)

        # Bottom controls
        bottom_layout = QHBoxLayout()

        self.play_btn = QPushButton("Fit\n(F5)")
        self.play_btn.setFont(QFont('Arial', 21))
        self.play_btn.setStyleSheet("background-color: rgb(0, 191, 0); color: black;")
        self.play_btn.clicked.connect(self.fit_pressed)

        # Fit options
        fit_group = QVBoxLayout()
        fit_way_layout = QHBoxLayout()
        self.MS_fit = QCheckBox("MS")
        self.SMS_fit = QCheckBox("SMS")
        self.SMS_fit.setChecked(True)
        self.APS_fit = QCheckBox("APS")
        self.APS_fit.setEnabled(False)
        self.APS_fit.setToolTip("Coming soon")
        
        # Connect MS and SMS to be mutually exclusive
        self.MS_fit.stateChanged.connect(lambda: self.on_ms_sms_changed(self.MS_fit))
        self.SMS_fit.stateChanged.connect(lambda: self.on_ms_sms_changed(self.SMS_fit))
        
        fit_way_layout.addWidget(self.MS_fit)
        fit_way_layout.addWidget(self.SMS_fit)
        fit_way_layout.addWidget(self.APS_fit)

        fit_par_layout = QHBoxLayout()
        g_label = QLabel("G")
        self.GCMS_input = QLineEdit(str(np.genfromtxt(os.path.join(self.dir_path, 'GCMS.txt'), delimiter='\t')))
        integral_label = QLabel("Integral")
        self.jn0_input = QLineEdit("32")
        fit_par_layout.addWidget(g_label)
        fit_par_layout.addWidget(self.GCMS_input)
        fit_par_layout.addWidget(integral_label)
        fit_par_layout.addWidget(self.jn0_input)

        fit_group.addLayout(fit_way_layout)
        fit_group.addLayout(fit_par_layout)

        # File chooser
        file_layout = QVBoxLayout()
        file_choose_layout = QHBoxLayout()
        self.btnchoose = QPushButton("Choose\nspectrum")
        self.btnchoose.setFont(QFont('Arial', 18))
        self.btnchoose.clicked.connect(self.choose_file)
        self.process_path = QTextEdit("['Calibration.dat']")
        self.process_path.setMaximumHeight(100)  # Make it taller
        # Allow editing of file paths
        file_choose_layout.addWidget(self.btnchoose)
        file_choose_layout.addWidget(self.process_path)
        file_layout.addLayout(file_choose_layout)
        file_layout.addLayout(fit_group)

        bottom_layout.addWidget(self.play_btn)
        bottom_layout.addLayout(file_layout)

        # Show buttons and INS
        show_layout = QVBoxLayout()
        show_buttons = QHBoxLayout()
        self.show_btn = QPushButton("Show spectrum")
        self.show_btn.setFont(QFont('Arial', 18))
        self.show_btn.clicked.connect(self.show_pressed)

        self.showM_btn = QPushButton("Show model")
        self.showM_btn.setFont(QFont('Arial', 18))
        self.showM_btn.clicked.connect(self.showM_pressed)

        show_buttons.addWidget(self.show_btn)
        show_buttons.addWidget(self.showM_btn)

        ins_layout = QHBoxLayout()
        self.ins_btn = QPushButton("Instrumental\nfunction")
        self.ins_btn.setFont(QFont('Arial', 16))
        self.ins_btn.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        
        # Create dropdown menu for INS button
        self.ins_menu = QMenu()
        find_single = QAction("Find\nInstr. func.\n single line", self)
        find_single.triggered.connect(lambda: self.instrumental_pressed(0, 0))
        find_aFe = QAction("Find\nInstr. func.\n pure a-Fe", self)
        find_aFe.triggered.connect(lambda: self.instrumental_pressed(0, 2))
        find_model = QAction("Find\nInstr. func.\nmodel", self)
        find_model.triggered.connect(lambda: self.instrumental_pressed(0, 1))
        self.ins_menu.addAction(find_single)
        self.ins_menu.addAction(find_aFe)
        self.ins_menu.addAction(find_model)
        self.ins_btn.setMenu(self.ins_menu)

        ins_num_layout = QVBoxLayout()
        ins_num_label = QLabel("№ of lines")
        ins_num_label.setFont(QFont('Arial', 18))
        self.ins_number = QLineEdit("3")
        ins_num_layout.addWidget(ins_num_label)
        ins_num_layout.addWidget(self.ins_number)

        self.ins_btn2 = QPushButton("Refine\nInstr. func.\nESRF")
        self.ins_btn2.setFont(QFont('Arial', 15))
        self.ins_btn2.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        
        # Create dropdown menu for INS refine button
        self.ins_menu2 = QMenu()
        refine_single = QAction("Refine\nInstr. func.\n single line", self)
        refine_single.triggered.connect(lambda: self.instrumental_pressed(1, 0))
        refine_aFe = QAction("Refine\nInstr. func.\n pure a-Fe", self)
        refine_aFe.triggered.connect(lambda: self.instrumental_pressed(1, 2))
        refine_model = QAction("Refine\nInstr. func.\nmodel", self)
        refine_model.triggered.connect(lambda: self.instrumental_pressed(1, 1))
        self.ins_menu2.addAction(refine_single)
        self.ins_menu2.addAction(refine_aFe)
        self.ins_menu2.addAction(refine_model)
        self.ins_btn2.setMenu(self.ins_menu2)

        ins_layout.addWidget(self.ins_btn)
        ins_layout.addLayout(ins_num_layout)
        ins_layout.addWidget(self.ins_btn2)

        show_layout.addLayout(show_buttons)
        show_layout.addLayout(ins_layout)

        bottom_layout.addLayout(show_layout)

        left_layout.addLayout(bottom_layout)

        # Second row: spectrum options and sequence fitting
        seq_fit_layout = QHBoxLayout()
        
        # Change spectrum dropdown button
        self.change_spectrum_btn = QPushButton("Change\nspectrum(a)")
        self.change_spectrum_btn.setFont(QFont('Arial', 16))
        self.change_spectrum_btn.clicked.connect(self.show_spectrum_options)
        
        # Choose workfolder button
        self.choose_workfolder_btn = QPushButton("Choose\nworkfolder")
        self.choose_workfolder_btn.setFont(QFont('Arial', 16))
        self.choose_workfolder_btn.clicked.connect(self.choose_workfolder)
        
        # Sequence fitting button
        self.sequence_fitting_type = 0  # 0 = initial, 1 = result
        self.seq_fit_btn = QPushButton("Sequence Fitting\n(initial)")
        self.seq_fit_btn.setFont(QFont('Arial', 16))
        self.seq_fit_btn.clicked.connect(self.show_sequence_fitting_options)
        
        seq_fit_layout.addWidget(self.change_spectrum_btn)
        seq_fit_layout.addWidget(self.choose_workfolder_btn)
        seq_fit_layout.addWidget(self.seq_fit_btn)
        
        left_layout.addLayout(seq_fit_layout)

        # Save buttons with save path between
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save\nresult")
        self.save_btn.setFont(QFont('Arial', 18))
        # self.save_btn.clicked.connect(self.save_pressed)

        self.saveas_btn = QPushButton("Save\nresult as")
        self.saveas_btn.setFont(QFont('Arial', 18))
        # self.saveas_btn.clicked.connect(self.save_as_pressed)

        # Editable save path field
        self.save_path = QLineEdit("")
        self.save_path.setFont(QFont('Arial', 18))
        self.save_path.setPlaceholderText("Save path")

        self.save_model_btn = QPushButton("Save\nmodel")
        self.save_model_btn.setFont(QFont('Arial', 18))
        self.save_model_btn.clicked.connect(self.save_model_pressed)

        self.save_model_as_btn = QPushButton("Save\nmodel as")
        self.save_model_as_btn.setFont(QFont('Arial', 18))
        self.save_model_as_btn.clicked.connect(self.save_model_as_pressed)

        self.save_btn.clicked.connect(self.save_result_pressed)
        self.saveas_btn.clicked.connect(self.save_result_as_pressed)
        
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.saveas_btn)
        save_layout.addWidget(self.save_path, 1)  # Stretch to fill space
        save_layout.addWidget(self.save_model_btn)
        save_layout.addWidget(self.save_model_as_btn)

        left_layout.addLayout(save_layout)

        splitter.addWidget(self.left_panel)

        # Right panel (now second)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)

        # Top part: spectrum plot and controls
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Spectrum plot
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
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

        # Connect to resize event to maintain tight layout
        self.canvas.mpl_connect('resize_event', self.on_figure_resize)
        # Connect to scroll event for zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)

        # Top controls for image
        image_controls = QHBoxLayout()
        self.dark_light_mode = QPushButton('Light spc')
        self.dark_light_mode.setFont(QFont('Arial', 18))
        self.dark_light_mode.setStyleSheet("background-color: rgb(191, 191, 191);")
        # self.dark_light_mode.clicked.connect(self.change_visual)

        title = QLabel("Result")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont('Arial', 24))
        title.setStyleSheet("color: red;")

        self.SP_DI = QPushButton('Distribution')
        self.SP_DI.setFont(QFont('Arial', 18))
        # self.SP_DI.clicked.connect(self.change_image)

        image_controls.addWidget(self.dark_light_mode)
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
        self.log.setFont(QFont('Arial', 21))
        self.log.setMaximumHeight(100)
        self.log.setReadOnly(True)  # Status field, read-only
        self.log.setPlainText("Ready")  # Initial status

        self.take_result_btn = QPushButton('Take result as model (F8)')
        self.take_result_btn.setFont(QFont('Arial', 18))
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
        splitter.setSizes([1000, 600])  # Left larger

        # Load and plot default spectrum
        self.plot_default_spectrum()

        # Keyboard shortcuts
        # self.play_btn.setShortcut("F5")
        # etc.

    # Placeholder methods - to be implemented
    def plot_default_spectrum(self):
        """Load and plot the default spectrum from Calibration.dat"""
        try:
            A_list, B_list = load_spectrum(self, ["Calibration.dat"], calibration_path=self.calibration_path)
            if A_list and B_list:
                # Calculate background for calibration
                backgrounds = calculate_backgrounds(["Calibration.dat"], self.calibration_path)
                plot_spectrum(self.figure, A_list, B_list, ["Calibration.dat"], backgrounds=backgrounds)
                self.toolbar.push_current()  # Set current view as home
        except Exception as e:
            self.log.setPlainText(f"Could not load default spectrum: {e}")
            self.log.setStyleSheet("color: red;")

    def toggle_position_markers(self, show):
        """Toggle visibility of position marker artists."""
        for artist in self.position_artists:
            artist.set_visible(show)
        self.canvas.draw()
    
    def on_figure_resize(self, event):
        """Reapply tight layout when the figure is resized"""
        self.figure.tight_layout()
        self.canvas.draw()

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

    def insert_row(self, row):
        # Simplified: just clear the current row and shift or something
        # For now, just clear
        self.params_table.clear_row_params(row)
        # TODO: Actually insert a new row in the grid

    def loadmod_pressed(self):
        load_model(self)

    def save_model_pressed(self):
        save_model(self)

    def save_model_as_pressed(self):
        save_model_as(self)
        pass

    def update_velocity_label(self):
        """Update the velocity label and button icon based on velocity direction"""
        if self.velocity_direction:
            self.cal_cho_title.setText("Velocity\nup-down:")
            icon_path = os.path.join(self.dir_path, "UD.png")
        else:
            self.cal_cho_title.setText("Velocity\ndown-up:")
            icon_path = os.path.join(self.dir_path, "DU.png")
        
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
            self.log.setPlainText("Model cleaned (baseline preserved)")
            self.log.setStyleSheet("color: green;")
        except Exception as e:
            self.log.setPlainText(f"Error cleaning model: {e}")
            self.log.setStyleSheet("color: red;")

    def take_result(self):
        """Copy fitting results to parameter table as new model"""
        try:
            # Check if results are available
            if not hasattr(self.results_table, 'current_model_list') or not self.results_table.current_model_list:
                self.log.setPlainText("No fitting results available")
                self.log.setStyleSheet("color: orange;")
                return
            
            # Get data from results table
            model_list = self.results_table.current_model_list
            model_colors = self.results_table.current_model_colors
            parameter_names = self.results_table.current_parameter_names
            parameters = self.results_table.current_parameters
            errors = self.results_table.current_errors
            
            # Clear all model rows except baseline
            for row in range(1, len(self.params_table.row_widgets)):
                self.params_table.clear_row_params(row)
            
            # Copy each model from results to parameters
            param_index = 0
            for model_idx, model_name in enumerate(model_list):
                row_idx = model_idx
                
                # Get model color
                color = model_colors[model_idx] if model_idx < len(model_colors) else 'blue'
                
                # Select the model (skip for baseline - it's already there)
                if model_name != 'baseline':
                    self.params_table.select_model(row_idx, model_name)
                    # Set the color (only for non-baseline rows)
                    self.params_table.select_color(row_idx, color)
                
                # Get number of parameters for this model
                num_params = len(parameter_names[model_idx]) if model_idx < len(parameter_names) else 0
                
                # Copy parameter values and fix checkboxes
                for col in range(num_params):
                    if param_index >= len(parameters):
                        break
                    
                    # Get widgets
                    row_widget = self.params_table.row_widgets[row_idx]
                    param_widget = row_widget.layout().itemAt(col + 1).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    top_layout = param_widget.layout().itemAt(0).layout()
                    fix_cb = top_layout.itemAt(1).widget()
                    
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
            
            self.log.setPlainText("Result copied to model")
            self.log.setStyleSheet("color: green;")
            
        except Exception as e:
            self.log.setPlainText(f"Error copying result: {e}")
            self.log.setStyleSheet("color: red;")
            import traceback
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
            self.log.setPlainText(f"Calibration file set: {os.path.basename(file_path)}")
            self.log.setStyleSheet("color: green;")
            # Optionally replot the default spectrum with new calibration
            self.plot_default_spectrum()
        else:
            self.log.setPlainText("Calibration selection canceled")
            self.log.setStyleSheet("color: orange;")

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
            self.log.setPlainText(f"Unknown spectrum option: {option}")
            self.log.setStyleSheet("color: red;")

    def choose_workfolder(self):
        """Choose work folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Choose work folder",
            self.workfolder
        )
        if folder_path:
            self.workfolder = folder_path
            self.log.setPlainText(f"Work folder set: {folder_path}")
            self.log.setStyleSheet("color: green;")
            # TODO: Implement any folder change logic
        else:
            self.log.setPlainText("Work folder selection canceled")
            self.log.setStyleSheet("color: orange;")

    # def show_sequence_fitting_options(self):
    #     """Show dropdown menu for sequence fitting options"""
    #     menu = QMenu(self)
        
    #     # Create actions for each option
    #     options = [
    #         ("take always initial guess\nfor the sequence of spectra", 0, "initial"),
    #         ("take result as initial guess\nfor the sequence of spectra", 1, "result")
    #     ]
        
    #     for text, value, display_name in options:
    #         action = QAction(text.replace('\n', ' '), self)
    #         action.triggered.connect(lambda checked, val=value, name=display_name: self.on_sequence_fitting_selected(val, name))
    #         menu.addAction(action)
        
    #     # Show menu below the button
    #     menu.exec(self.seq_fit_btn.mapToGlobal(self.seq_fit_btn.rect().bottomLeft()))

    # def on_sequence_fitting_selected(self, value, display_name):
    #     """Handle sequence fitting option selection"""
    #     self.sequence_fitting_type = value
    #     self.seq_fit_btn.setText(f"Sequence Fitting\n({display_name})")
    #     self.log.setPlainText(f"Sequence fitting set to: {display_name}")
    #     self.log.setStyleSheet("color: blue;")
    #     # TODO: Implement actual functionality

    def initialize_parameters(self):
        """Initialize parameters from INSint.txt and INSexp.txt"""
        try:
            try:
                JN0 = int(self.jn0_input.text())
            except:
                JN0 = 32
            self.JN0 = JN0

            try:
                GCMS = int(self.GCMS_input.text())
            except:
                GCMS = 0.1
            self.GCMS = GCMS

            insint_path = os.path.join(self.dir_path, 'INSint.txt')
            self.MulCo, self.x0 = np.genfromtxt(insint_path, delimiter=' ', skip_footer=0)
            
            ins_path = os.path.join(self.dir_path, 'INSexp.txt')
            self.INS = np.genfromtxt(ins_path, delimiter=' ', skip_footer=0)
            
            print(f'Initialized: MulCo={self.MulCo}, x0={self.x0}')

        except Exception as e:
            self.log.setPlainText(f"Error loading INS files: {e}")
            self.log.setStyleSheet("color: red;")
            return False
        return True

    def calibration(self):
        """Perform calibration on RAW spectrum files"""
        if not self.path_list:
            self.log.setPlainText("No spectrum selected for calibration")
            self.log.setStyleSheet("color: orange;")
            return
        
        file = os.path.abspath(self.path_list[0])
        
        # Check if file is RAW format (not calibrated)
        if not (file.endswith('.mca') or file.endswith('.cmca') or file.endswith('.ws5') or 
                file.endswith('.w98') or file.endswith('.moe') or file.endswith('.m1') or 
                file.lower().endswith('.mcs')):
            self.log.setPlainText("Calibration only works with RAW files (.mca, .cmca, .ws5, .w98, .moe, .m1, .mcs)")
            self.log.setStyleSheet("color: orange;")
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
        self.log.setPlainText("Calibration in progress...")
        self.log.setStyleSheet("color: blue;")
        
        # Start calibration in a separate thread
        self.calibration_thread = CalibrationThread(
            self.dir_path, file, experimental_method, self.INS, JN, self.x0, self.MulCo, vel_start, self.pool
        )
        self.calibration_thread.finished.connect(self.on_calibration_finished)
        self.calibration_thread.error.connect(self.on_calibration_error)
        self.calibration_thread.start()
    
    def on_calibration_finished(self, A, B, C):
        """Handle calibration completion"""
        try:
            # Plot calibration results
            plot_calibration(self.figure, A, B, C, gridcolor=self.gridcolor)
            self.canvas.draw()
            self.toolbar.push_current()  # Set current view as home
            
            # Update calibration path (Calibration.dat was already saved by Calibration function)
            self.calibration_path = os.path.join(self.dir_path, 'Calibration.dat')
            
            self.log.setPlainText("Calibration completed successfully")
            self.log.setStyleSheet("color: green;")
        except Exception as e:
            self.log.setPlainText(f"Error processing calibration results: {e}")
            self.log.setStyleSheet("color: red;")
        finally:
            self.cal_btn.setEnabled(True)
    
    def on_calibration_error(self, error_msg):
        """Handle calibration error"""
        self.log.setPlainText(f"Calibration error: {error_msg}")
        self.log.setStyleSheet("color: red;")
        self.cal_btn.setEnabled(True)
    
    def showM_pressed(self):
        """Show model with spectrum and subspectra (non-blocking)"""
        # Parse the current content of process_path
        self.path_list = self.parse_process_path()
        
        # Check if spectrum is loaded
        if not self.path_list:
            self.log.setPlainText("No spectrum selected. Please choose a spectrum first.")
            self.log.setStyleSheet("color: orange;")
            return
        
        # Initialize
        if not self.initialize_parameters():
            return
        
        # Start model calculation in a separate thread
        self.log.setPlainText("Calculating model...")
        self.log.setStyleSheet("color: blue;")
        
        self.show_model_thread = ShowModelThread(self, self.path_list, self.pool)
        self.show_model_thread.finished.connect(self.on_show_model_finished)
        self.show_model_thread.error.connect(self.on_show_model_error)
        self.show_model_thread.start()
    
    def on_show_model_finished(self, A, B, SPC_f, FS, FS_pos, p, model, has_nbaseline, backgrounds):
        """Handle show model completion"""
        try:
            # Get current colors from table (fresh read to handle delete/insert)
            current_colors = self.params_table.get_current_colors()
            
            # Store FS_pos for toggle functionality
            self.current_FS_pos = FS_pos
            
            # Plot model - use different function for Nbaseline case
            if has_nbaseline:
                # FS and FS_pos are already lists of lists (one list per spectrum section)
                # p is already a list of parameter arrays (one per spectrum section)
                position_artists = plot_model_with_nbaseline(self.figure, A, B, SPC_f, FS, FS_pos, p, model, current_colors, backgrounds, gridcolor=self.gridcolor)
            else:
                # FS and FS_pos are simple lists, p is a single array
                position_artists = plot_model(self.figure, A, B, SPC_f, FS, FS_pos, p, current_colors, backgrounds, gridcolor=self.gridcolor)
            
            # Store position artists and enable toggle button
            self.position_artists = position_artists if position_artists else []
            if self.position_artists:
                self.toolbar.toggle_positions_action.setEnabled(True)
                self.toolbar.toggle_positions_action.setChecked(True)
            else:
                self.toolbar.toggle_positions_action.setEnabled(False)
            
            self.canvas.draw()
            self.toolbar.push_current()
            
            self.log.setPlainText("Model displayed successfully")
            self.log.setStyleSheet("color: green;")
        except Exception as e:
            traceback.print_exc()
            self.log.setPlainText(f"Error plotting model: {e}")
            self.log.setStyleSheet("color: red;")
    
    def on_show_model_error(self, error_msg):
        """Handle show model error"""
        self.log.setPlainText(f"Error showing model: {error_msg}")
        self.log.setStyleSheet("color: red;")
    
    def on_raw_to_dat_finished(self, success_msg):
        """Handle raw to dat conversion completion"""
        self.log.setPlainText(success_msg)
        self.log.setStyleSheet("color: green;")
    
    def on_raw_to_dat_error(self, error_msg):
        """Handle raw to dat conversion error"""
        self.log.setPlainText(f"RAW to DAT conversion error: {error_msg}")
        self.log.setStyleSheet("color: red;")
    
    def create_subspectra(self, model, Distri, Cor, p, number_of_baseline_parameters):
        """
        Create subspectra from model components.
        
        Args:
            model: list of model names
            Distri: distribution expressions
            Cor: correlation expressions
            p: parameter array
            number_of_baseline_parameters: number of baseline parameters
            
        Returns:
            tuple: (Ps, Psm, Distri_t, Cor_t, Di, Co)
        """
        Ps = []
        Psm = []
        Distri_t = []
        Cor_t = []
        Di = 0
        Co = 0
        V = number_of_baseline_parameters
        
        for i in range(len(model)):
            ps = np.array(p[0:number_of_baseline_parameters], dtype=float)
            Psm.append([model[i]])
            LenM = mod_len_def(model[i], include_special=False) + 1
            
            for j in range(1, LenM):
                ps = np.append(ps, p[V])
                V += 1
            
            Ps.append(ps)
            
            if model[i] == 'Expression':
                del Ps[-1]
                ps = np.append(ps, 1.0)
                Ps.append(ps)
                V += 1
            
            if model[i] == 'Distr':
                del Ps[-1]
                for j in range(1, 6):
                    Ps[-1] = np.append(Ps[-1], p[V])
                    V += 1
                del Psm[-1]
                Psm[-1].append(model[i])
                
                STR = Distri[Di] + str(' ')
                # Evaluate p[] references in distribution expression
                st_ = []
                en_ = []
                for k in range(len(STR) - 2):
                    if STR[k] == 'p' and STR[k + 1] == '[':
                        st_.append(k)
                        for kk in range(k, len(STR)):
                            if STR[kk] == ']':
                                en_.append(kk)
                                break
                st_ = st_[::-1]
                en_ = en_[::-1]
                for k in range(len(st_)):
                    try:
                        STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
                    except:
                        pass
                Distri_t.append(STR)
                Di += 1
            
            if model[i] == 'Corr':
                del Ps[-1]
                for j in range(1, 3):
                    Ps[-1] = np.append(Ps[-1], p[V])
                    V += 1
                del Psm[-1]
                Psm[-1].append(model[i])
                
                STR = Cor[Co] + str(' ')
                # Evaluate p[] references in correlation expression
                st_ = []
                en_ = []
                for k in range(len(STR) - 2):
                    if STR[k] == 'p' and STR[k + 1] == '[':
                        st_.append(k)
                        for kk in range(k, len(STR)):
                            if STR[kk] == ']':
                                en_.append(kk)
                                break
                st_ = st_[::-1]
                en_ = en_[::-1]
                for k in range(len(st_)):
                    try:
                        STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
                    except:
                        pass
                Cor_t.append(STR)
                Co += 1
        
        return (Ps, Psm, Distri_t, Cor_t, Di, Co)

    def raw_to_dat(self):
        """Convert RAW spectra to .dat format using current calibration"""
        # Parse the current content of process_path
        self.path_list = self.parse_process_path()
        
        if not self.path_list:
            self.log.setPlainText("No spectrum selected")
            self.log.setStyleSheet("color: orange;")
            return
        
        # Check if calibration exists
        if not os.path.exists(self.calibration_path):
            self.log.setPlainText("Calibration file not found. Please calibrate first.")
            self.log.setStyleSheet("color: red;")
            return
        
        # Get save path from save_path field
        save_path = self.save_path.text().strip()
        if not save_path:
            self.log.setPlainText("Please specify save path")
            self.log.setStyleSheet("color: orange;")
            return
        
        # Start conversion in a separate thread
        self.log.setPlainText("Converting RAW to DAT...")
        self.log.setStyleSheet("color: blue;")
        
        self.raw_to_dat_thread = RawToDatThread(self, self.path_list, self.calibration_path, save_path)
        self.raw_to_dat_thread.finished.connect(self.on_raw_to_dat_finished)
        self.raw_to_dat_thread.error.connect(self.on_raw_to_dat_error)
        self.raw_to_dat_thread.start()

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

    def interrupt(self):
        """Emergency abort: kill the current pool and create a new one"""
        try:
            # Terminate the old pool
            if self.pool:
                self.pool.terminate()
                self.pool.join()
            # Create a new pool
            num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
            self.pool = mp.Pool(processes=num_processes)
            self.log.setPlainText("Pool terminated and recreated")
            self.log.setStyleSheet("color: orange;")
        except Exception as e:
            self.log.setPlainText(f"Error during interrupt: {str(e)}")
            self.log.setStyleSheet("color: red;")

    def play_pressed(self):
        pass

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
            # Update baseline Ns based on new spectrum
            self.params_table.update_baseline_from_bg()
            # Log will be updated by update_baseline_from_bg
            # Optionally call show_pressed here, but user didn't specify
        else:
            self.log.setPlainText("Selection canceled")
            self.log.setStyleSheet("color: orange;")

    def instrumental_pressed(self, ref, mode):
        """
        Handle instrumental function calculation/refinement button press.
        
        Args:
            ref: 0 for Find, 1 for Refine
            mode: 0 for single line, 1 for model, 2 for pure a-Fe
        """
        # Ensure parameters are initialized
        if not self.initialize_parameters():
                return
        
        # Validation - check if model is defined for mode == 1
        if mode == 1:
            # Check if there's at least one model component (row 1 or above)
            has_model = False
            if hasattr(self, 'params_table') and len(self.params_table.row_widgets) > 1:
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
                self.log.setPlainText("Specify model to restore instrumental function")
                self.log.setStyleSheet("color: red;")
                return
        
        if (self.MS_fit.isChecked() and mode == 0):
            self.log.setPlainText("This will not work...")
            self.log.setStyleSheet("color: red;")
            return
        
        if not self.path_list or len(self.path_list) == 0:
            self.log.setPlainText("No spectrum loaded")
            self.log.setStyleSheet("color: red;")
            return
        
        file = os.path.abspath(self.path_list[0])
        if not os.path.exists(file):
            self.log.setPlainText("Spectrum file does not exist")
            self.log.setStyleSheet("color: red;")
            return
        

        
        # Start instrumental calculation in a thread
        self.log.setPlainText(f"Running instrumental function (ref={ref}, mode={mode})...")
        self.log.setStyleSheet("color: cyan;")
        
        self.instrumental_thread = InstrumentalThread(self, ref, mode, self.pool)
        self.instrumental_thread.finished.connect(self.on_instrumental_finished)
        self.instrumental_thread.error.connect(self.on_instrumental_error)
        self.instrumental_thread.start()
    
    def on_instrumental_finished(self, result):
        """Handle instrumental function completion"""
        try:
            from spectrum_plotter import plot_instrumental_result
            
            # Plot results on the figure
            gridcolor = getattr(self, 'gridcolor', 'gray')
            result_svg, result_png = plot_instrumental_result(
                self.figure, result['A'], result['B'], result['F'], result['F2'],
                result['p'], result['hi2'], result['file'], self.dir_path, gridcolor
            )
            
            # Update canvas the same way as showM_pressed does
            self.canvas.draw()
            self.toolbar.push_current()
            
            self.log.setPlainText(f"Instrumental function completed. χ² = {result['hi2']:.3f}\nResults saved to {self.dir_path}")
            self.log.setStyleSheet("color: green;")
            
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
            import traceback
            self.log.setPlainText(f"Error plotting instrumental results: {e}\n{traceback.format_exc()}")
            self.log.setStyleSheet("color: red;")
    
    def on_instrumental_error(self, error_msg):
        """Handle instrumental function error"""
        self.log.setPlainText(f"Instrumental function error: {error_msg}")
        self.log.setStyleSheet("color: red;")

    def show_pressed(self):
        """Load and display the selected spectrum(s)"""
        # Parse the current content of process_path
        self.path_list = self.parse_process_path()
        
        if not self.path_list:
            self.log.setPlainText("No spectrum selected")
            self.log.setStyleSheet("color: orange;")
            return
        try:
            A_list, B_list = load_spectrum(self, self.path_list, calibration_path=self.calibration_path)
            if A_list and B_list:              
                self.backgrounds = calculate_backgrounds(self.path_list, self.calibration_path)
                plot_spectrum(self.figure, A_list, B_list, self.path_list, self.backgrounds)
                self.toolbar.push_current()  # Set current view as home
                self.log.setPlainText(f"Spectra displayed ({len(A_list)})")
                self.log.setStyleSheet("color: green;")
            else:
                self.log.setPlainText("Could not load spectrum")
                self.log.setStyleSheet("color: red;")
        except Exception as e:
            self.log.setPlainText(f"Error displaying spectrum: {e}")
            self.log.setStyleSheet("color: red;")

    # def showM_pressed(self):
    #     pass

    def choose_workfolder(self):
        """Open folder selection dialog to choose workfolder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Choose Workfolder", self.workfolder)
        if folder_path:
            self.workfolder = folder_path
            self.log.setPlainText(f"Workfolder changed to: {folder_path}")
            self.log.setStyleSheet("color: green;")
        else:
            self.log.setPlainText("Workfolder selection canceled")
            self.log.setStyleSheet("color: orange;")

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
        self.log.setPlainText(f"Sequence fitting type set to: {'initial' if fitting_type == 0 else 'result'}")
        self.log.setStyleSheet("color: blue;")

    def replot_result(self, row_index):
        """
        Replot results for a specific row when button is clicked.
        
        Args:
            row_index: Index of the row (0 to numro*3-1)
        
        TODO: Implement replotting logic based on original prog_raw.py replot_result method.
        This should:
        - Determine which component based on row (row_index // 3)
        - Parse the row to determine which model component to highlight
        - Adjust Z_order, Color_order arrays
        - Replot the spectrum with highlighted component
        - Update the figure canvas
        """
        # TODO: Implement
        component_index = row_index // 3  # Which component (0, 1, 2, ...)
        row_type = row_index % 3  # 0=name, 1=value, 2=error
        
        print(f"Replot requested for row {row_index}: component {component_index}, type {row_type}")
        self.log.setPlainText(f"Replot component {component_index} - TODO: implement")
        self.log.setStyleSheet("color: orange;")

    def save_pressed(self):
        pass

    def save_as_pressed(self):
        pass
    
    def validate_spectrum_files(self, spectrum_files):
        """
        Validate that all spectrum files exist and can be loaded.
        
        Args:
            spectrum_files: List of spectrum file paths to validate
            
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        from spectrum_io import load_spectrum
        
        # Check file existence first
        for i, file_path in enumerate(spectrum_files):
            if not os.path.exists(file_path):
                return False, f"File does not exist: {file_path}"
        
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
        self.log.setPlainText("Starting fit...")
        self.log.setStyleSheet("color: cyan;")
        
        if not self.initialize_parameters():
            return

        try:
            # Get spectrum files
            spectrum_files = self.parse_process_path()
            
            if not spectrum_files:
                self.log.setPlainText("No spectrum file loaded. Please load a file first.")
                self.log.setStyleSheet("color: orange;")
                return
            
            # Validate all spectrum files can be loaded
            valid, error_msg = self.validate_spectrum_files(spectrum_files)
            if not valid:
                self.log.setPlainText(f"Spectrum validation failed: {error_msg}")
                self.log.setStyleSheet("color: red;")
                return
            
            # Check for sequential fitting: no Nbaseline AND multiple spectra
            import fitting_io
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
                    self.start_sequential_fitting(spectrum_files)
                    return
                else:
                    # User declined - fit only first spectrum
                    self.log.setPlainText("Sequential fitting canceled. Fitting first spectrum only.")
                    self.log.setStyleSheet("color: orange;")
                    spectrum_files = [spectrum_files[0]]
            
            # Single spectrum or simultaneous fitting (with Nbaseline)
            spectrum_file = spectrum_files[0]
            self.log.setPlainText(f"Fitting spectrum: {os.path.basename(spectrum_file)}")
            self.log.setStyleSheet("color: cyan;")
            
            # Start fitting in background thread
            self.fitting_thread = FittingThread(self, spectrum_file, self.pool)
            self.fitting_thread.finished.connect(self.on_fitting_finished)
            self.fitting_thread.error.connect(self.on_fitting_error)
            self.fitting_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"Fit error: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.log.setPlainText(f"Fit error: {e}")
            self.log.setStyleSheet("color: red;")
    
    def start_sequential_fitting(self, spectrum_files):
        """
        Start sequential fitting of multiple spectra using fitting_io logic.
        
        Args:
            spectrum_files: List of spectrum file paths to fit sequentially
        """
        # Check if save path is set
        if not self.save_path.text().strip():
            self.log.setPlainText("Sequential fitting requires save path to be set")
            self.log.setStyleSheet("color: red;")
            return
        
        # Initialize sequence_params for result mode (None for initial mode)
        self.sequence_params = None
        
        # Calculate backgrounds for all spectra upfront
        self.log.setPlainText(f"Calculating backgrounds for {len(spectrum_files)} spectra...")
        self.log.setStyleSheet("color: cyan;")
        
        from spectrum_io import calculate_backgrounds
        backgrounds = calculate_backgrounds(spectrum_files, self.calibration_path)
        
        print("calculated backgrounds:", backgrounds)

        # Determine mode name for logging
        mode_name = 'initial guess' if self.sequence_fitting_type == 0 else 'result as model'
        self.log.setPlainText(f"Starting sequential fitting of {len(spectrum_files)} spectra (mode: {mode_name})...")
        self.log.setStyleSheet("color: cyan;")
        
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
                fix
            )
            self.results_table.current_chi2 = chi2
            
            # Plot the result
            self.plot_fitting_result(result)
            
            # Save files
            self._save_sequential_result_files(spectrum_file)
            
        except Exception as e:
            import traceback
            print(f"Error handling fitted spectrum: {e}\n{traceback.format_exc()}")
    
    def _on_sequential_progress(self, index, total, spectrum_file, status):
        """Handle progress updates from sequential fitting"""
        if status == 'fitting':
            self.log.setPlainText(
                f"Sequential fitting [{index + 1}/{total}]: {os.path.basename(spectrum_file)}"
            )
            self.log.setStyleSheet("color: cyan;")
        elif status == 'saved':
            self.log.setPlainText(
                f"Saved [{index + 1}/{total}]: {os.path.basename(spectrum_file)}"
            )
            self.log.setStyleSheet("color: blue;")
        elif status == 'failed':
            self.log.setPlainText(
                f"Failed [{index + 1}/{total}]: {os.path.basename(spectrum_file)}"
            )
            self.log.setStyleSheet("color: orange;")
        elif status == 'error':
            self.log.setPlainText(
                f"Error [{index + 1}/{total}]: {os.path.basename(spectrum_file)}"
            )
            self.log.setStyleSheet("color: red;")
    
    def _on_sequential_finished(self, summary):
        """Handle completion of sequential fitting"""
        total = summary['total']
        succeeded = summary['succeeded']
        failed = summary['failed']
        errors = summary['errors']
        
        # Clean up sequence_params
        self.sequence_params = None
        
        if failed == 0:
            self.log.setPlainText(f"Sequential fitting complete! All {total} spectra fitted and saved.")
            self.log.setStyleSheet("color: green;")
        else:
            error_summary = "\n".join([f"  - {os.path.basename(f)}: {e}" for f, e in errors[:5]])  # Show first 5 errors
            if len(errors) > 5:
                error_summary += f"\n  ... and {len(errors) - 5} more errors"
            
            self.log.setPlainText(
                f"Sequential fitting complete: {succeeded}/{total} succeeded, {failed} failed.\n"
                f"Errors:\n{error_summary}"
            )
            self.log.setStyleSheet("color: orange;")
    
    def _save_sequential_result_files(self, spectrum_file):
        """Save result files for one spectrum (file I/O only, called from main thread)"""
        try:
            # Get base path from save_path and spectrum filename
            save_dir = os.path.dirname(self.save_path.text())
            spectrum_basename = os.path.splitext(os.path.basename(spectrum_file))[0]
            base_path = os.path.join(save_dir, spectrum_basename)
            
            # Create directory if needed
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Get current results
            parameters = self.results_table.current_parameters
            errors = self.results_table.current_errors
            model_list = self.results_table.current_model_list
            model_colors = self.results_table.current_model_colors
            parameter_names = self.results_table.current_parameter_names
            chi2 = self.results_table.current_chi2
            
            # Determine save mode
            param_file = base_path + '_param.txt'
            mode = 'append' if os.path.exists(param_file) else 'new'
            
            # 1. Save parameters
            self._save_parameters_file(param_file, parameters, errors, model_list, 
                                      parameter_names, os.path.basename(spectrum_file), chi2, mode)
            
            # 2. Save graph data
            result_txt_dst = base_path + '_graf.txt'
            if hasattr(self, 'last_fitting_data') and self.last_fitting_data:
                self._save_graf_file(result_txt_dst, self.last_fitting_data)
            
            # 3. Save combo image
            result_png_src = os.path.join(self.dir_path, 'result.png')
            table_image = self.results_table.render_table_to_image()
            if os.path.exists(result_png_src):
                self._save_combo_image_from_qimage(result_png_src, table_image, 
                                                    base_path + '_combo.png')
            
            # 4. Copy SVG
            result_svg_src = os.path.join(self.dir_path, 'result.svg')
            result_svg_dst = base_path + '.svg'
            if os.path.exists(result_svg_src):
                shutil.copyfile(result_svg_src, result_svg_dst)
            
            print(f"[Sequential] Saved results for {spectrum_basename}")
            
        except Exception as e:
            import traceback
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
            is_simultaneous = result.get('is_simultaneous', False)
            
            if is_simultaneous:
                # Simultaneous fitting - multiple spectra
                from spectrum_plotter import plot_simultaneous_fitting_result
                
                gridcolor = getattr(self, 'gridcolor', 'gray')
                
                # Store FS_pos for toggle functionality (from first spectrum)
                self.current_FS_pos = result.get('FS_pos_list', [[]])[0] if result.get('FS_pos_list') else []
                
                # Store fitting data for graf.txt saving
                self.last_fitting_data = {
                    'A': result['A_list'],  # List of arrays
                    'B': result['B_list'],
                    'SPC_f': result['SPC_f_list'],
                    'FS': result['FS_list']
                }
                
                result_svg, result_png, position_artists_list = plot_simultaneous_fitting_result(
                    self.figure, result['A_list'], result['B_list'], result['SPC_f_list'], 
                    result['FS_list'], result['FS_pos_list'], fitted_parameters, 
                    result['begining_spc'], model_colors, chi2, result['spectrum_files'],
                    self.dir_path, gridcolor
                )
                
                # Store position artists (from all subplots)
                self.position_artists = [artist for sublist in position_artists_list for artist in sublist]
                if self.position_artists:
                    self.toolbar.toggle_positions_action.setEnabled(True)
                    self.toolbar.toggle_positions_action.setChecked(True)
                else:
                    self.toolbar.toggle_positions_action.setEnabled(False)
                
                # Update canvas
                self.canvas.draw()
                self.toolbar.push_current()
                
                self.log.setPlainText(f"Simultaneous fit completed! χ² = {chi2:.3f}")
                self.log.setStyleSheet("color: green;")
                
            elif 'A' in result and 'B' in result and 'SPC_f' in result and 'FS' in result:
                # Single spectrum fitting
                from spectrum_plotter import plot_fitting_result
                
                gridcolor = getattr(self, 'gridcolor', 'gray')
                FS_pos = result.get('FS_pos', [])
                
                # Store FS_pos for toggle functionality
                self.current_FS_pos = FS_pos
                
                # Store fitting data for graf.txt saving
                self.last_fitting_data = {
                    'A': result['A'],
                    'B': result['B'],
                    'SPC_f': result['SPC_f'],
                    'FS': result['FS']
                }
                
                result_svg, result_png, position_artists = plot_fitting_result(
                    self.figure, result['A'], result['B'], result['SPC_f'], result['FS'],
                    FS_pos, fitted_parameters, model_colors, chi2, result['spectrum_file'], 
                    self.dir_path, gridcolor
                )
                
                # Store position artists and enable toggle button if positions exist
                self.position_artists = position_artists
                if position_artists:
                    self.toolbar.toggle_positions_action.setEnabled(True)
                    self.toolbar.toggle_positions_action.setChecked(True)
                else:
                    self.toolbar.toggle_positions_action.setEnabled(False)
                
                # Update canvas
                self.canvas.draw()
                self.toolbar.push_current()
                
                self.log.setPlainText(f"Fit completed! χ² = {chi2:.3f}")
                self.log.setStyleSheet("color: green;")
            else:
                self.log.setPlainText(f"Fit completed! χ² = {chi2:.3f} (no plot data)")
                self.log.setStyleSheet("color: green;")
                
        except Exception as e:
            import traceback
            print(f"Error plotting result: {e}\n{traceback.format_exc()}")
            self.log.setPlainText(f"Plot error: {e}")
            self.log.setStyleSheet("color: orange;")
    
    def on_fitting_finished(self, result):
        """Handle fitting completion"""
        try:
            if not result['success']:
                self.log.setPlainText(f"Fitting failed: {result['message']}")
                self.log.setStyleSheet("color: red;")
                return
            
            # Read model configuration for results table
            model_list = self.params_table.get_model_list()
            model_colors = self.params_table.get_current_colors()
            parameter_names = self.params_table.get_parameter_names()
            
            # Extract results
            fitted_parameters = result['parameters']
            errors = result['errors']
            chi2 = result['chi2']
            covariance_matrix = result['covariance_matrix']
            fix = result.get('fix', np.array([], dtype=int))
            is_simultaneous = result.get('is_simultaneous', False)
            
            print(f"[Fit] Fitting successful!")
            print(f"[Fit] χ² = {chi2:.3f}")
            print(f"[Fit] Simultaneous: {is_simultaneous}")
            
            # Update results table
            self.results_table.fill_table(
                fitted_parameters,
                model_list,
                model_colors,
                parameter_names,
                covariance_matrix,
                errors,
                fix
            )
            
            # Store chi2 for saving
            self.results_table.current_chi2 = chi2
            
            # Plot results
            self.plot_fitting_result(result)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing fit results: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.log.setPlainText(f"Error processing fit results: {e}")
            self.log.setStyleSheet("color: red;")
    
    def on_fitting_error(self, error_msg):
        """Handle fitting error"""
        self.log.setPlainText(f"Fitting error: {error_msg}")
        self.log.setStyleSheet("color: red;")
    def save_result_pressed(self):
        """Save fitting results to file"""
        # Check if we have results to save
        if not hasattr(self.results_table, 'current_parameters') or self.results_table.current_parameters is None:
            self.log.setPlainText("No results to save. Please run fitting first.")
            self.log.setStyleSheet("color: red;")
            return
        
        # Check if save path is set
        if not self.save_path.text().strip():
            self.log.setPlainText("Please specify save path")
            self.log.setStyleSheet("color: orange;")
            return
        
        # Check if file exists and ask user
        param_file = self.save_path.text() + '_param.txt'
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
                self.log.setPlainText("Saving canceled")
                self.log.setStyleSheet("color: orange;")
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
            self.log.setPlainText("No results to save. Please run fitting first.")
            self.log.setStyleSheet("color: red;")
            return
        
        # Open file dialog
        save_dir = self.workfolder if self.workfolder else os.path.dirname(__file__)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save results as. Careful: it overwrites existing files without confirmation", save_dir, "All files (*)", options=QFileDialog.Option.DontConfirmOverwrite
        )
        
        if file_path:
            # Remove extension if user added one
            base_path = os.path.splitext(file_path)[0]
            self.save_path.setText(base_path)
            self._save_result_files('new')
        else:
            self.log.setPlainText("Saving canceled")
            self.log.setStyleSheet("color: orange;")
    
    def _save_result_files(self, mode):
        """
        Save all result files
        mode: 'new', 'append', or 'overwrite'
        """
        try:
            base_path = self.save_path.text()
            
            # Create directory if needed
            save_dir = os.path.dirname(base_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Get current results
            parameters = self.results_table.current_parameters
            errors = self.results_table.current_errors
            model_list = self.results_table.current_model_list
            model_colors = self.results_table.current_model_colors
            parameter_names = self.results_table.current_parameter_names
            chi2 = self.results_table.current_chi2
            
            # Get spectrum file name
            if self.path_list:
                spectrum_file = os.path.basename(self.path_list[0])
            else:
                spectrum_file = "unknown"
            
            # 1. Save parameters to CSV-like text file
            param_file = base_path + '_param.txt'
            self._save_parameters_file(param_file, parameters, errors, model_list, 
                                      parameter_names, spectrum_file, chi2, mode)
            
            # 2. Save result graph data from fitting arrays
            result_txt_dst = base_path + '_graf.txt'
            if hasattr(self, 'last_fitting_data') and self.last_fitting_data:
                self._save_graf_file(result_txt_dst, self.last_fitting_data)
            
            # 3. Save combo image (spectrum + table)
            result_png_src = os.path.join(self.dir_path, 'result.png')
            
            # Render results table to image instead of loading from file
            table_image = self.results_table.render_table_to_image()
            
            if os.path.exists(result_png_src):
                self._save_combo_image_from_qimage(result_png_src, table_image, 
                                                    base_path + '_combo.png')
            
            # 4. Copy SVG
            result_svg_src = os.path.join(self.dir_path, 'result.svg')
            result_svg_dst = base_path + '.svg'
            if os.path.exists(result_svg_src):
                shutil.copyfile(result_svg_src, result_svg_dst)
            
            # Success message
            if mode == 'append':
                self.log.setPlainText("Results appended to parameter file, others overwritten")
            else:
                self.log.setPlainText("Results saved successfully")
            self.log.setStyleSheet("color: green;")
            
        except Exception as e:
            import traceback
            print(f"Error saving results: {e}\n{traceback.format_exc()}")
            self.log.setPlainText(f"Error saving results: {e}")
            self.log.setStyleSheet("color: red;")
    
    def _save_graf_file(self, filepath, fitting_data):
        """
        Save graph data to text file with columns: A (velocity), B (data), Baseline, SPC_f (fit), model1, model2, ...
        
        Args:
            filepath: Path to save the file
            fitting_data: Dictionary with 'A', 'B', 'SPC_f', 'FS' arrays
        """
        A = fitting_data['A']
        B = fitting_data['B']
        SPC_f = fitting_data['SPC_f']
        FS = fitting_data['FS']
        
        # Calculate baseline from current parameters
        if hasattr(self.results_table, 'current_parameters') and self.results_table.current_parameters is not None:
            p = self.results_table.current_parameters
            # Baseline formula: p[0] + p[3]*p[0]/100*A + p[2]*p[0]/10000*(A-p[1])^2 + p[6]*p[4]/10000*(A-p[5])^2 + p[4] + p[7]*p[4]/100*A
            baseline = (p[0] + p[3] * p[0]/100 * A + p[2] * p[0] / 10000 * (A - p[1])**2 + 
                       p[6] * p[4] / 10000 * (A - p[5])**2 + p[4] + p[7] * p[4]/100 * A)
        else:
            # Fallback: use zeros
            baseline = np.zeros_like(A)
        
        # Get model names (skip baseline which is first)
        if hasattr(self.results_table, 'current_model_list') and len(self.results_table.current_model_list) > 1:
            model_names = self.results_table.current_model_list[1:]  # Skip 'baseline'
        else:
            model_names = [f'Submodel{i+1}' for i in range(len(FS))]
        
        # Pad model_names if needed
        while len(model_names) < len(FS):
            model_names.append(f'Submodel{len(model_names)+1}')
        
        # Create data columns: A, B, Baseline, SPC_f, then each subspectrum
        data_columns = [A, B, baseline, SPC_f]
        data_columns.extend(FS)
        
        # Transpose to row format
        data_array = np.column_stack(data_columns)
        
        # Build header
        header = 'Velocity\tData\tBaseline\tFit\t' + '\t'.join(model_names[:len(FS)])
        
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
                if param_idx < len(parameters):
                    values.append(parameters[param_idx])
                else:
                    values.append('')
                
                # Add error name and value
                names.append(f'd_{param_name}')
                if errors is not None and param_idx < len(errors):
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
    
    def _save_combo_image(self, plot_path, table_path, output_path):
        """Combine plot and table images and save"""
        im1 = matplotlib.image.imread(plot_path)
        im2 = matplotlib.image.imread(table_path)
        
        # Determine background color
        bg_color = [0, 0, 0, 1] if self.BGcolor == 'k' else [1, 1, 1, 1]
        
        # Make widths match
        if im2.shape[1] > im1.shape[1]:
            # Pad im1
            padding = np.array([[bg_color] * (im2.shape[1] - im1.shape[1])] * im1.shape[0], np.uint8)
            im1 = np.concatenate((im1, padding), axis=1)
        elif im2.shape[1] < im1.shape[1]:
            # Pad im2
            padding = np.array([[bg_color] * (im1.shape[1] - im2.shape[1])] * im2.shape[0], np.uint8)
            im2 = np.concatenate((im2, padding), axis=1)
        
        # Stack vertically
        combo_image = np.concatenate((im1, im2), axis=0)
        matplotlib.image.imsave(output_path, combo_image)
    
    def _save_combo_image_from_qimage(self, plot_path, table_qimage, output_path):
        """Combine plot image file and table QImage and save"""
        from PySide6.QtGui import QImage
        from PIL import Image
        import io
        
        # Load plot image
        im1 = matplotlib.image.imread(plot_path)
        
        # Convert QImage to numpy array
        width = table_qimage.width()
        height = table_qimage.height()
        
        # Convert to format compatible with numpy
        table_qimage = table_qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        
        # Get bytes and convert to numpy array
        ptr = table_qimage.constBits()
        ptr.setsize(height * width * 4)
        im2 = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        
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
        
        # Resize im1 if needed
        if im1.shape[1] != target_width:
            scale_factor = target_width / im1.shape[1]
            new_height = int(im1.shape[0] * scale_factor)
            im1_pil = Image.fromarray(im1)
            im1_pil = im1_pil.resize((target_width, new_height), Image.Resampling.LANCZOS)
            im1 = np.array(im1_pil)
        
        # Resize im2 if needed
        if im2.shape[1] != target_width:
            scale_factor = target_width / im2.shape[1]
            new_height = int(im2.shape[0] * scale_factor)
            im2_pil = Image.fromarray(im2)
            im2_pil = im2_pil.resize((target_width, new_height), Image.Resampling.LANCZOS)
            im2 = np.array(im2_pil)
        
        # Stack vertically
        combo_image = np.concatenate((im1, im2), axis=0)
        matplotlib.image.imsave(output_path, combo_image)
