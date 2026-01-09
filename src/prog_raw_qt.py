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
import models as m5
import models_positions as modpos
import models_NFS as mN
import Calibration as cal
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
from constants import numro, numco

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QScrollArea, QGridLayout, QSplitter, QFrame, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QSpinBox, QDoubleSpinBox,
    QMenu
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon, QAction, QDoubleValidator, QRegularExpressionValidator
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QLocale, QRegularExpression
import sys

from parameters_table import ParametersTable
from model_io import load_model
from spectrum_plotter import plot_spectrum

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

class PhysicsApp(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.dir_path = os.path.dirname(__file__)
        self.workfolder = os.getcwd()
        self.workfolder_check = 1
        self.gridcolor = 'w'
        self.check_points_match = False
        self.newfilename = str('')
        self.newfilename2 = str('')
        self.BGcolor = 'k'
        self.cal_path = os.path.join(self.dir_path, "Calibration.dat")
        self.points_match = True
        self.path_list = []
        self.backgrounds = []  # List to store calculated backgrounds

        # Model colors
        ct = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato']
        ct.extend(ct)
        ct.extend(ct)
        ct.extend(ct)  # now it is 80
        self.model_colors = ct[:numro]  # One color per row

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
        # self.btncleanmodel.clicked.connect(self.clean_model)

        self.switch = QCheckBox()
        self.switch.setChecked(True)

        self.cal_cho_btn = QPushButton("Choose\ncalibration\nfile")
        self.cal_cho_btn.setFont(QFont('Arial', 16))
        self.cal_cho_btn.setStyleSheet("background-color: rgb(63, 127, 127); color: white;")
        # self.cal_cho_btn.clicked.connect(self.calibration_path)

        # Calibration group
        cal_group = QVBoxLayout()
        self.cal_cho_title = QLabel("Velocity\ndown-up:")
        self.cal_cho_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cal_cho_title.setFont(QFont('Arial', 16))
        self.cal_cho_title.setStyleSheet("color: red;")

        self.vel_start = QCheckBox()
        # self.vel_start.stateChanged.connect(...)

        cal_group.addWidget(self.cal_cho_title)
        cal_group.addWidget(self.vel_start)

        self.cal_btn = QPushButton("Calibrate")
        self.cal_btn.setFont(QFont('Arial', 16))
        self.cal_btn.setStyleSheet("background-color: rgb(127, 255, 255); color: black;")
        # self.cal_btn.clicked.connect(self.calibration)

        self.vel_btn = QPushButton("RAW\nto\ndat")
        self.vel_btn.setFont(QFont('Arial', 16))
        self.vel_btn.setStyleSheet("background-color: rgb(127, 255, 255); color: black;")
        # self.vel_btn.clicked.connect(self.velocityscale)

        interrupt_btn = QPushButton('! INTERRUPT !')
        interrupt_btn.setFont(QFont('Arial', 16))
        interrupt_btn.setStyleSheet("background-color: red; color: white;")
        # interrupt_btn.clicked.connect(self.interrupt)

        top_controls.addWidget(self.loadmod_btn)
        top_controls.addWidget(self.btncleanmodel)
        top_controls.addWidget(self.switch)
        top_controls.addWidget(self.cal_cho_btn)
        top_controls.addLayout(cal_group)
        top_controls.addWidget(self.cal_btn)
        top_controls.addWidget(self.vel_btn)
        top_controls.addWidget(interrupt_btn)

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
        # self.play_btn.clicked.connect(self.play_pressed)

        # Fit options
        fit_group = QVBoxLayout()
        fit_way_layout = QHBoxLayout()
        self.chb1 = QCheckBox("MS")
        self.chb3 = QCheckBox("SMS")
        self.chb3.setChecked(True)
        self.chb4 = QCheckBox("APS")
        fit_way_layout.addWidget(self.chb1)
        fit_way_layout.addWidget(self.chb3)
        fit_way_layout.addWidget(self.chb4)

        fit_par_layout = QHBoxLayout()
        g_label = QLabel("G")
        self.l0_input = QLineEdit(str(np.genfromtxt(os.path.join(self.dir_path, 'GCMS.txt'), delimiter='\t')))
        integral_label = QLabel("Integral")
        self.jn0_input = QLineEdit("32")
        fit_par_layout.addWidget(g_label)
        fit_par_layout.addWidget(self.l0_input)
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
        self.process_path.setReadOnly(True)  # Prevent editing
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
        self.showM_btn.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        # self.showM_btn.clicked.connect(self.showM_pressed)

        show_buttons.addWidget(self.show_btn)
        show_buttons.addWidget(self.showM_btn)

        ins_layout = QHBoxLayout()
        self.ins_btn = QPushButton("Instrumental\nfunction")
        self.ins_btn.setFont(QFont('Arial', 16))
        self.ins_btn.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        # Dropdown for INS - simplified as combo for now
        self.ins_combo = QComboBox()
        self.ins_combo.addItems(['Find Instr. func. single line', 'Find Instr. func. pure a-Fe', 'Find Instr. func. model'])

        ins_num_layout = QVBoxLayout()
        ins_num_label = QLabel("№ of lines")
        ins_num_label.setFont(QFont('Arial', 18))
        self.ins_number = QLineEdit("3")
        ins_num_layout.addWidget(ins_num_label)
        ins_num_layout.addWidget(self.ins_number)

        self.ins_btn2 = QPushButton("Refine\nInstr. func.\nESRF")
        self.ins_btn2.setFont(QFont('Arial', 15))
        self.ins_btn2.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        # Similar combo

        ins_layout.addWidget(self.ins_btn)
        ins_layout.addLayout(ins_num_layout)
        ins_layout.addWidget(self.ins_btn2)

        show_layout.addLayout(show_buttons)
        show_layout.addLayout(ins_layout)

        bottom_layout.addLayout(show_layout)

        left_layout.addLayout(bottom_layout)

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
        self.toolbar = NavigationToolbar(self.canvas, self)
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

        # Results table
        self.results_table = QTableWidget(numro * 2, numco + 1)
        self.results_table.setHorizontalHeaderLabels([''] + [f'Col {i}' for i in range(numco)])
        self.results_table.setVerticalHeaderLabels([f'Row {i}' for i in range(numro * 2)])
        self.results_table.setFont(QFont('Arial', 18))
        self.results_table.setStyleSheet("color: white; background-color: black;")

        scroll_results = QScrollArea()
        scroll_results.setWidget(self.results_table)
        scroll_results.setWidgetResizable(True)
        bottom_layout.addWidget(scroll_results)

        # Log and take result
        log_layout = QHBoxLayout()
        self.log = QTextEdit()
        self.log.setFont(QFont('Arial', 21))
        self.log.setMaximumHeight(100)
        self.log.setReadOnly(True)  # Status field, read-only
        self.log.setPlainText("Ready")  # Initial status

        self.take_result_btn = QPushButton('Take result as model (F8)')
        self.take_result_btn.setFont(QFont('Arial', 18))
        # self.take_result_btn.clicked.connect(self.take_result)

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
            from model_io import load_spectrum, calculate_backgrounds
            A_list, B_list = load_spectrum(self, "Calibration.dat")
            if A_list and B_list:
                # Calculate background for calibration
                backgrounds = calculate_backgrounds(["Calibration.dat"])
                plot_spectrum(self.figure, A_list, B_list, ["Calibration.dat"], backgrounds=backgrounds)
                self.toolbar.push_current()  # Set current view as home
        except Exception as e:
            self.log.setPlainText(f"Could not load default spectrum: {e}")
            self.log.setStyleSheet("color: red;")

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



    def insert_row(self, row):
        # Simplified: just clear the current row and shift or something
        # For now, just clear
        self.params_table.clear_row_params(row)
        # TODO: Actually insert a new row in the grid

    def loadmod_pressed(self):
        load_model(self)

    def save_model_pressed(self):
        from model_io import save_model
        save_model(self)

    def save_model_as_pressed(self):
        from model_io import save_model_as
        save_model_as(self)
        pass

    def calibration_path(self):
        pass

    def calibration(self):
        pass

    def velocityscale(self):
        pass

    def interrupt(self):
        pass

    def play_pressed(self):
        pass

    def choose_file(self):
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
            # Set save path based on first file
            if file_paths:
                self.save_path.setText(file_paths[0])
            self.log.setPlainText("Spectrum(s) selected")
            self.log.setStyleSheet("color: green;")
            # Optionally call show_pressed here, but user didn't specify
        else:
            self.log.setPlainText("Selection canceled")
            self.log.setStyleSheet("color: orange;")

    def show_pressed(self):
        """Load and display the selected spectrum(s)"""
        if not self.path_list:
            self.log.setPlainText("No spectrum selected")
            self.log.setStyleSheet("color: orange;")
            return
        try:
            from model_io import load_spectrum, calculate_backgrounds
            A_list, B_list = load_spectrum(self, self.path_list)
            if A_list and B_list:
                # Calculate backgrounds if not already done or if list changed
                if not hasattr(self, 'backgrounds') or len(self.backgrounds) != len(self.path_list):
                    self.backgrounds = calculate_backgrounds(self.path_list)

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

    def showM_pressed(self):
        pass

    def save_pressed(self):
        pass

    def save_as_pressed(self):
        pass