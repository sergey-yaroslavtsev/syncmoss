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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.image
from matplotlib import colors
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
numro = 50
numco = 15

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

class ClickableLabel(QLabel):
    def __init__(self, text, row, col):
        super().__init__(text)
        self.row = row
        self.col = col
        self.original_text = text
        self.pressed = False
        self.showing_index = False
        self.setMouseTracking(True)

    def setText(self, text):
        if not self.showing_index:
            self.original_text = text
        super().setText(text)

    def mousePressEvent(self, event):
        if self.original_text == "":
            return
        self.pressed = True
        self.showing_index = True
        main_window = self.window()
        index = sum(main_window.row_params[:self.row]) + self.col
        super().setText(f"p[{int(index)}]")
        self.update()
        self.repaint()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.pressed:
            self.pressed = False
            self.showing_index = False
            self.setText(self.original_text)
            self.update()
            self.repaint()
        event.accept()

    def leaveEvent(self, event):
        if self.pressed:
            self.pressed = False
            self.showing_index = False
            self.setText(self.original_text)
            self.update()
            self.repaint()
        event.accept()

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
        self.workfolder = None
        self.workfolder_check = 0
        self.gridcolor = 'w'
        self.check_points_match = False
        self.newfilename = str('')
        self.newfilename2 = str('')
        self.BGcolor = 'k'
        self.row_params = [8] + [0] * (numro - 1)  # baseline has 8 params

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
        # self.loadmod_btn.clicked.connect(self.loadmod_pressed)

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

        # Parameters table - using QVBoxLayout for dynamic rows
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        self.params_layout.setSpacing(1)
        self.row_widgets = []

        # Create baseline row
        baseline_row = self.create_baseline_row()
        self.row_widgets.append(baseline_row)
        self.params_layout.addWidget(baseline_row)

        # Create model rows
        for row in range(1, numro):
            row_widget = self.create_model_row(row)
            self.row_widgets.append(row_widget)
            self.params_layout.addWidget(row_widget)

        scroll_params = QScrollArea()
        scroll_params.setWidget(self.params_widget)
        scroll_params.setWidgetResizable(True)
        left_layout.addWidget(scroll_params)

        # Set initial colors cycling through the available colors
        color_cycle = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']
        for r in range(1, len(self.row_widgets)):
            color = color_cycle[(r - 1) % len(color_cycle)]
            self.select_color(r, color)

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
        # self.btnchoose.clicked.connect(self.choose_file)
        self.process_path = QLineEdit("['Calibration.dat']")
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
        self.show_btn.setStyleSheet("background-color: rgb(127, 127, 127); color: white;")
        # self.show_btn.clicked.connect(self.show_pressed)

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
        # self.save_model_btn.clicked.connect(self.save_model_pressed)

        self.save_model_as_btn = QPushButton("Save\nmodel as")
        self.save_model_as_btn.setFont(QFont('Arial', 18))
        # self.save_model_as_btn.clicked.connect(self.save_model_as_pressed)

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

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.image_label)

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
        right_layout.addLayout(image_controls)

        # Results table
        self.results_table = QTableWidget(numro * 2, numco + 1)
        self.results_table.setHorizontalHeaderLabels([''] + [f'Col {i}' for i in range(numco)])
        self.results_table.setVerticalHeaderLabels([f'Row {i}' for i in range(numro * 2)])
        self.results_table.setFont(QFont('Arial', 18))
        self.results_table.setStyleSheet("color: white; background-color: black;")

        scroll_results = QScrollArea()
        scroll_results.setWidget(self.results_table)
        scroll_results.setWidgetResizable(True)
        right_layout.addWidget(scroll_results)

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
        right_layout.addLayout(log_layout)

        splitter.addWidget(self.right_panel)
        splitter.setSizes([1000, 600])  # Left larger

        # Keyboard shortcuts
        # self.play_btn.setShortcut("F5")
        # etc.

    # Placeholder methods - to be implemented
    def change_visual(self):
        pass

    def change_image(self):
        pass

    def take_result(self):
        pass

    def create_baseline_row(self):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setSpacing(1)
        row_layout.setContentsMargins(1,1,1,1)

        # start_widget
        start_widget = QWidget()
        start_layout = QVBoxLayout(start_widget)
        start_layout.setSpacing(1)
        start_layout.setContentsMargins(1,1,1,1)
        name_fix_label = QLabel("Name | fix")
        name_fix_label.setFont(QFont('Arial', 12))
        name_fix_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        baseline_btn = QPushButton("baseline")
        baseline_btn.setFont(QFont('Arial', 12))
        boundaries_label = QLabel("boundaries")
        boundaries_label.setFont(QFont('Arial', 12))
        boundaries_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        start_layout.addWidget(name_fix_label)
        start_layout.addWidget(baseline_btn)
        start_layout.addWidget(boundaries_label)
        start_widget.setFixedWidth(110)
        row_layout.addWidget(start_widget)

        # param_widgets
        for col in range(numco):
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.setSpacing(1)
            param_layout.setContentsMargins(1,1,1,1)
            # Top row: name and fix
            top_layout = QHBoxLayout()
            top_layout.setSpacing(0)
            name_label = ClickableLabel("", 0, col)
            name_label.setFont(QFont('Arial', 8))
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setFixedWidth(50)
            fix_cb = QCheckBox()
            fix_cb.setFixedWidth(30)
            fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox.png); } QCheckBox::indicator:checked { image: url(CheckBox_L.png); }")
            top_layout.addWidget(name_label)
            top_layout.addWidget(fix_cb)
            # Value input
            value_input = QLineEdit("")
            value_input.setFont(QFont('Arial', 10))
            value_input.setFixedWidth(80)
            validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
            value_input.setValidator(validator_value)
            value_input.textChanged.connect(lambda text, inp=value_input: self.check_reference(inp))
            # Bounds layout
            bounds_layout = QHBoxLayout()
            bounds_layout.setSpacing(0)
            lower_input = QLineEdit("")
            lower_input.setFont(QFont('Arial', 10))
            lower_input.setFixedWidth(40)
            validator = QRegularExpressionValidator(QRegularExpression(r'^-?\d+(\.\d+)?$'))
            lower_input.setValidator(validator)
            upper_input = QLineEdit("")
            upper_input.setFont(QFont('Arial', 10))
            upper_input.setFixedWidth(40)
            validator2 = QRegularExpressionValidator(QRegularExpression(r'^-?\d+(\.\d+)?$'))
            upper_input.setValidator(validator2)
            bounds_layout.addWidget(lower_input)
            bounds_layout.addWidget(upper_input)
            param_layout.addLayout(top_layout)
            param_layout.addWidget(value_input)
            param_layout.addLayout(bounds_layout)
            row_layout.addWidget(param_widget)

        # Set initial values for baseline
        initial_values = [10000, 0, 0, 0, 0, 0, 0, 0]
        name_labels = ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr']
        for i in range(8):
            param_widget = row_layout.itemAt(i+1).widget()
            top_layout = param_widget.layout().itemAt(0).layout()
            name_label = top_layout.itemAt(0).widget()
            fix_cb = top_layout.itemAt(1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            bounds_layout = param_widget.layout().itemAt(2).layout()
            lower_input = bounds_layout.itemAt(0).widget()
            name_label.setText(name_labels[i])
            name_label.original_text = name_labels[i]
            value_input.setText(str(initial_values[i]))
            validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
            value_input.setValidator(validator_value)
            if i in [0,1,2,3,4,5,6,7]:
                fix_cb.setChecked(True)
            if i == 0:
                lower_input.setText("1")

        # Set read-only for parameters with empty names
        for col in range(numco):
            param_widget = row_layout.itemAt(col + 1).widget()
            name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            lower_input = param_widget.layout().itemAt(2).layout().itemAt(0).widget()
            upper_input = param_widget.layout().itemAt(2).layout().itemAt(1).widget()
            if name_label.text() == "":
                value_input.setReadOnly(True)
                lower_input.setReadOnly(True)
                upper_input.setReadOnly(True)

        return row_widget

    def create_model_row(self, row):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setSpacing(1)
        row_layout.setContentsMargins(1,1,1,1)

        # start_widget
        start_widget = QWidget()
        start_layout = QVBoxLayout(start_widget)
        start_layout.setSpacing(1)
        start_layout.setContentsMargins(1,1,1,1)
        color_btn = QPushButton("Color")
        color_btn.setFont(QFont('Arial', 12))
        color_btn.setStyleSheet("background-color: rgb(127, 127, 255);")
        # Add menu to color_btn
        color_menu = QMenu(self)
        color_codes = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']
        for code in color_codes:
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(self.get_color_from_code(code)))
            icon = QIcon(pixmap)
            action = QAction(code, self)
            action.setIcon(icon)
            action.triggered.connect(lambda checked, c=code, btn=color_btn: self.select_color_by_button(c, btn))
            color_menu.addAction(action)
        color_btn.setMenu(color_menu)
        model_btn = QPushButton("None")
        model_btn.setFont(QFont('Arial', 12))
        # Add menu to model_btn
        model_menu = QMenu(self)
        model_options = ['Singlet', 'Doublet', 'Sextet', 'MDGD', 'Relax_MS', 'Relax_2S', 'Hamilton_mc', 'Hamilton_pc', 'ASM', 'Be', 'KB_nano', 'Distr', 'Corr', 'Variables', 'Expression', 'Delete', 'Insert', 'Nbaseline']
        for option in model_options:
            action = QAction(option, self)
            action.triggered.connect(lambda checked, opt=option, btn=model_btn: self.select_model_by_button(opt, btn))
            model_menu.addAction(action)
        model_btn.setMenu(model_menu)
        fix_model_btn = QPushButton("fix model")
        fix_model_btn.setFont(QFont('Arial', 12))
        start_layout.addWidget(color_btn)
        start_layout.addWidget(model_btn)
        start_layout.addWidget(fix_model_btn)
        start_widget.setFixedWidth(110)
        row_layout.addWidget(start_widget)

        # param_widgets
        for col in range(numco):
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.setSpacing(1)
            param_layout.setContentsMargins(1,1,1,1)
            # Top row: name and fix
            top_layout = QHBoxLayout()
            top_layout.setSpacing(0)
            name_label = ClickableLabel("", row, col)
            name_label.setFont(QFont('Arial', 8))
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setFixedWidth(50)
            fix_cb = QCheckBox()
            fix_cb.setFixedWidth(30)
            fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox.png); } QCheckBox::indicator:checked { image: url(CheckBox_L.png); }")
            top_layout.addWidget(name_label)
            top_layout.addWidget(fix_cb)
            # Value input
            value_input = QLineEdit("")
            value_input.setFont(QFont('Arial', 10))
            value_input.setFixedWidth(80)
            validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
            value_input.setValidator(validator_value)
            value_input.textChanged.connect(lambda text, inp=value_input: self.check_reference(inp))
            # Bounds layout
            bounds_layout = QHBoxLayout()
            bounds_layout.setSpacing(0)
            lower_input = QLineEdit("")
            lower_input.setFont(QFont('Arial', 10))
            lower_input.setFixedWidth(40)
            validator = QRegularExpressionValidator(QRegularExpression(r'^-?\d+(\.\d+)?$'))
            lower_input.setValidator(validator)
            upper_input = QLineEdit("")
            upper_input.setFont(QFont('Arial', 10))
            upper_input.setFixedWidth(40)
            validator2 = QRegularExpressionValidator(QRegularExpression(r'^-?\d+(\.\d+)?$'))
            upper_input.setValidator(validator2)
            bounds_layout.addWidget(lower_input)
            bounds_layout.addWidget(upper_input)
            if name_label.text() == "":
                value_input.setReadOnly(True)
                lower_input.setReadOnly(True)
                upper_input.setReadOnly(True)
            param_layout.addLayout(top_layout)
            param_layout.addWidget(value_input)
            param_layout.addLayout(bounds_layout)
            row_layout.addWidget(param_widget)

        return row_widget

    def get_color_from_code(self, code):
        color_map = {'blue': 'blue', 'red': 'red', 'yellow': 'yellow', 'cyan': 'cyan', 'fuchsia': 'fuchsia', 'lime': 'lime', 'darkorange': 'darkorange', 'blueviolet': 'blueviolet', 'green': 'green', 'tomato': 'tomato', 'white': 'white', 'silver': 'silver', 'lightgreen': 'lightgreen', 'pink': 'pink'}
        return color_map.get(code, 'white')

    def select_color_by_button(self, c, btn):
        for r, rw in enumerate(self.row_widgets):
            start = rw.layout().itemAt(0).widget()
            if start.layout().itemAt(0).widget() == btn:
                self.select_color(r, c)
                return

    def select_model_by_button(self, opt, btn):
        for r, rw in enumerate(self.row_widgets):
            start = rw.layout().itemAt(0).widget()
            if start.layout().itemAt(1).widget() == btn:
                self.select_model(r, opt)
                return

    def select_color(self, row, color):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        color_btn = start_widget.layout().itemAt(0).widget()
        bg_color = self.get_color_from_code(color)
        text_color = 'black' if color in ['red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] else 'white'
        color_btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")

    def set_row_styles_red(self, row):
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()
        model_btn.setStyleSheet("background-color: red; color: white;")
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col+1).widget()
            param_widget.setStyleSheet("background-color: lightcoral;")

    def reset_row_styles(self, row):
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()
        model_btn.setStyleSheet("")
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col+1).widget()
            param_widget.setStyleSheet("")

    def select_model(self, row, model):
        if row >= len(self.row_widgets):
            return
        if model == 'Delete':
            if row == 0:
                return  # can't delete baseline
            deleted_start = sum(self.row_params[:row])
            deleted_count = self.row_params[row]
            # Clear the row
            self.clear_row_params(row)
            # Remove from layout and list
            row_widget = self.row_widgets.pop(row)
            self.params_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            # Remove from row_params
            self.row_params.pop(row)
            # Update references
            self.update_references(deleted_start, -deleted_count)
            # Update row numbers for subsequent rows
            for r in range(row, len(self.row_widgets)):
                row_widget = self.row_widgets[r]
                for col in range(numco):
                    param_widget = row_widget.layout().itemAt(col+1).widget()
                    top_layout = param_widget.layout().itemAt(0).layout()
                    name_label = top_layout.itemAt(0).widget()
                    name_label.row = r
        elif model == 'Insert':
            if row >= len(self.row_widgets):
                return
            inserted_start = sum(self.row_params[:row])
            # Create new empty row
            new_row_widget = self.create_model_row(row)
            # Insert into list and layout
            self.row_widgets.insert(row, new_row_widget)
            self.params_layout.insertWidget(row, new_row_widget)
            # Insert 0 in row_params
            self.row_params.insert(row, 0)
            # Update references (no change yet)
            self.update_references(inserted_start, 0)
            # Set the model button to "insert" with red background and param boxes to light red
            self.set_row_styles_red(row)
            # Set the model button text to "insert"
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_btn.setText('insert')
            # Preselect color to red
            self.select_color(row, 'red')
            # Update row numbers for subsequent rows
            for r in range(row+1, len(self.row_widgets)):
                row_widget = self.row_widgets[r]
                for col in range(numco):
                    param_widget = row_widget.layout().itemAt(col+1).widget()
                    top_layout = param_widget.layout().itemAt(0).layout()
                    name_label = top_layout.itemAt(0).widget()
                    name_label.row = r
        else:
            # Normal model selection
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            # Reset styles
            self.reset_row_styles(row)
            model_btn.setText(model)
            old_count = self.row_params[row]
            self.clear_row_params(row)
            self.auto_fill_params(row, model)
            # Update references if param count changed
            new_count = self.row_params[row]
            delta = new_count - old_count
            if delta != 0:
                start = sum(self.row_params[:row])
                self.update_references(start, delta)
            # Set validators for parameters
            for col in range(numco):
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                if col < self.row_params[row]:
                    if model in ['Distr', 'Corr', 'Expression'] and col == self.row_params[row] - 1:
                        value_input.setValidator(None)
                    else:
                        validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
                        value_input.setValidator(validator_value)
                    name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
                    lower_input = param_widget.layout().itemAt(2).layout().itemAt(0).widget()
                    upper_input = param_widget.layout().itemAt(2).layout().itemAt(1).widget()
                    if name_label.text() == "":
                        value_input.setReadOnly(True)
                        lower_input.setReadOnly(True)
                        upper_input.setReadOnly(True)
                    else:
                        value_input.setReadOnly(False)
                        lower_input.setReadOnly(False)
                        upper_input.setReadOnly(False)
                else:
                    value_input.setValidator(None)
                    lower_input = param_widget.layout().itemAt(2).layout().itemAt(0).widget()
                    upper_input = param_widget.layout().itemAt(2).layout().itemAt(1).widget()
                    value_input.setReadOnly(True)
                    lower_input.setReadOnly(True)
                    upper_input.setReadOnly(True)

    def update_references(self, start_index, delta):
        for r in range(len(self.row_widgets)):
            for col in range(numco):
                if col < self.row_params[r]:
                    param_widget = self.row_widgets[r].layout().itemAt(col + 1).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    text = value_input.text()
                    match = re.match(r'=\[(\d+),(.+)\]', text)
                    if match:
                        ref_index = int(match.group(1))
                        value_part = match.group(2)
                        if delta < 0:  # deleting
                            deleted_count = -delta
                            if ref_index >= start_index and ref_index < start_index + deleted_count:
                                # deleted, clear and red
                                value_input.setText("")
                                value_input.setStyleSheet("background-color: red;")
                            elif ref_index >= start_index + deleted_count:
                                # shift down
                                new_index = ref_index + delta
                                value_input.setText(f"=[{new_index},{value_part}]")
                                value_input.setStyleSheet("")
                        elif delta > 0:  # inserting
                            if ref_index >= start_index:
                                new_index = ref_index + delta
                                value_input.setText(f"=[{new_index},{value_part}]")
                                value_input.setStyleSheet("")

    def check_reference(self, input):
        text = input.text()
        if not re.match(r'=\[\d+,.+\]', text):
            input.setStyleSheet("")

    def auto_fill_params(self, row, model):
        # Auto-fill params based on model, mimicking original
        if model == 'Singlet':
            names = ['T', 'δ, mm/s', 'L, mm/s', 'G, mm/s']
            values = ['1.0', '0.0', '0.098', '0.1']
            lowers = ['0', '', '0.098', '0']
            uppers = ['', '', '', '']
            fixes = [False, False, True, False]
        elif model == 'Doublet':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1']
            values = ['1.0', '0.0', '1.0', '0.098', '0.1', '0.5', '1.0']
            lowers = ['0', '', '', '0.098', '0', '0', '0']
            uppers = ['', '', '', '', '', '1', '']
            fixes = [False, False, False, True, False, True, True]  # 3,5,6
        elif model == 'Sextet':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'A', 'a+', 'a-', 'GH, T', 'I1/I3']
            values = ['1.0', '0.0', '0.0', '33.0', '0.098', '0.1', '0.5', '0.0', '0.0', '0.0', '3.0']
            lowers = ['0', '', '', '', '0.098', '0', '0', '', '', '0', '0']
            uppers = ['', '', '', '', '', '', '1', '', '', '', '']
            fixes = [False, False, False, False, True, False, True, True, True, True, True]  # 4,6,7,8,9,10
        elif model == 'MDGD':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'GH, T', 'Dδε', 'DδH', 'DεH', 'A', 'a+', 'a-', 'I1/I3']
            values = ['1.0', '0.0', '0.0', '33.0', '0.098', '0.1', '0.0', '0.0', '0.0', '0.0', '0.5', '0.0', '0.0', '3.0']
            lowers = ['0', '', '', '', '0.098', '0', '0', '-1', '-1', '-1', '0', '', '', '0']
            uppers = ['', '', '', '', '', '', '', '1', '1', '1', '1', '', '', '']
            fixes = [False, False, False, False, True, False, True, True, True, True, True, True, True, True]  # 4 to 13
        elif model == 'Hamilton_mc':
            names = ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °', 'θ, °', 'φ, °']
            values = ['1.0', '0.0', '0.0', '33.0', '0.098', '0.1', '0.0', '0.0', '0.0', '0.0', '0.0']
            lowers = ['0', '', '', '', '0.098', '0', '-1', '-180', '-360', '-180', '-360']
            uppers = ['', '', '', '', '', '', '1', '180', '360', '180', '360']
            fixes = [False, False, False, False, True, False, False, False, False, False, False]  # 4
        elif model == 'Hamilton_pc':
            names = ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °']
            values = ['1.0', '0.0', '0.0', '33.0', '0.098', '0.1', '0.0', '0.0', '0.0']
            lowers = ['0', '', '', '', '0.098', '0', '-1', '-180', '-360']
            uppers = ['', '', '', '', '', '', '1', '180', '360']
            fixes = [False, False, False, False, True, False, False, False, False]  # 4
        elif model == 'Relax_MS':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'A', 'R', 'alfa', 'S']
            values = ['1.0', '0.0', '0.0', '33.0', '0.098', '0.1', '0.5', '1.0', '101']
            lowers = ['0', '', '', '', '0.098', '0', '0', '0', '0.5']
            uppers = ['', '', '', '', '', '1', '', '100', '']
            fixes = [False, False, False, False, False, True, False, False, True]
        elif model == 'Relax_2S':
            names = ['T', 'δ1, mm/s', 'ε1, mm/s', 'H1, T', 'δ2, mm/s', 'ε2, mm/s', 'H2, T', 'L, mm/s', 'A', 'Ω12', 'P1/P2']
            values = ['1.0', '0.0', '0.0', '33.0', '0.0', '0.0', '-33.0', '0.1', '0.5', '0.3', '1']
            lowers = ['', '', '', '', '', '', '', '0.098', '0', '0', '0']
            uppers = ['', '', '', '', '', '', '', '', '1', '', '']
            fixes = [False, False, False, False, False, False, False, False, True, False, True]
        elif model == 'ASM':
            names = ['T', 'δ, mm/s', 'εm, mm/s', 'εl, mm/s', 'His, T', 'Han, T', 'L, mm/s', 'G, mm/s', 'm', 'A', 'Num', 'I13']
            values = ['1.0', '0.0', '0.0', '0.0', '30.0', '5.0', '0.098', '0.1', '0.1', '0.5', '25', '3.0']
            lowers = ['0', '', '', '', '', '', '0.098', '0', '-1', '0', '7', '0']
            uppers = ['', '', '', '', '', '', '', '', '1', '1', '', '']
            fixes = [False, False, False, False, False, False, True, False, False, True, True, False]
        elif model == 'Be':
            # Based on Doublet, load from Be.txt or defaults
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1']
            try:
                be_param = np.genfromtxt(os.path.join(self.dir_path, 'Be.txt'), delimiter='\t')
                values = [str(be_param[i]) for i in range(7)]
                self.log.setPlainText("Be.txt loaded successfully.")
            except:
                values = ['0.048', '0.103', '-0.259', '0.098', '0.105', '0.265', '1.0']
                self.log.setPlainText("Default Be values used. Could not load Be.txt.")
            lowers = ['0', '', '', '0.098', '0', '0', '0']
            uppers = ['', '', '', '', '', '1', '']
            fixes = [True] * 7
        elif model == 'KB_nano':
            # Based on Doublet, load from KB.txt
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1']
            try:
                kb_param = np.genfromtxt(os.path.join(self.dir_path, 'KB.txt'), delimiter='\t')
                values = [str(kb_param[i]) for i in range(7)]
                self.log.setPlainText("KB.txt loaded successfully.")
            except:
                values = ['0.065', '0.234', '0.37', '0.098', '0.373', '0.5', '1.0']
                self.log.setPlainText("Default KB values used. Could not load KB.txt.")
            lowers = ['0', '', '', '0.098', '0', '0', '0']
            uppers = ['', '', '', '', '', '1', '']
            fixes = [True] * 7
        elif model == 'Variables':
            # Fill all columns with V1, V2, ...
            names = [f'V{i+1}' for i in range(numco)]
            values = ['0'] * numco
            lowers = [''] * numco
            uppers = [''] * numco
            fixes = [True] * numco
        elif model == 'Nbaseline':
            # Similar to baseline
            names = ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr']
            values = ['10000', '0', '0', '0', '0', '0', '0', '0']
            lowers = ['', '', '', '', '0', '', '', '']
            uppers = ['', '', '', '', '', '', '', '']
            fixes = [False, True, True, True, True, True, True, True]
        elif model == 'Expression':
            names = ['Expression']
            values = ['p[0]']
            lowers = ['']
            uppers = ['']
            fixes = [True]
        elif model == 'Average_H':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'Hin, T', 'L, mm/s', 'G, mm/s', 'Hex, T', 'K', 'J', 'θ, °', 'N']
            values = ['1.0', '0.0', '0.0', '15.0', '0.098', '0.1', '5', '1', '-1', '90', '100']
            lowers = ['0', '', '', '', '0.098', '0', '0', '0', '0', '0', '1']
            uppers = ['', '', '', '', '', '', '', '', '', '90', '']
            fixes = [False, False, False, False, True, False, False, False, False, False, True]
        elif model == 'Distr':
            names = ['par', 'L', 'R', 'Num', 'Probability density function']
            values = ['1', '0', '1', '20', 'X']  # par depends on previous
            lowers = ['1', '', '', '1', '']
            uppers = ['', '', '', '1000', '']
            fixes = [True, False, False, True, True]
        elif model == 'Corr':
            names = ['par', 'Dependency function']
            values = ['1', 'X']
            lowers = ['1', '']
            uppers = ['', '']
            fixes = [True, True]
        else:
            return  # No auto-fill for others

        # Set the params
        for i, (name, value, lower, upper, fix) in enumerate(zip(names, values, lowers, uppers, fixes)):
            if i < numco:
                row_widget = self.row_widgets[row]
                param_widget = row_widget.layout().itemAt(i+1).widget()
                top_layout = param_widget.layout().itemAt(0).layout()
                name_label = top_layout.itemAt(0).widget()
                fix_cb = top_layout.itemAt(1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                bounds_layout = param_widget.layout().itemAt(2).layout()
                lower_input = bounds_layout.itemAt(0).widget()
                upper_input = bounds_layout.itemAt(1).widget()
                name_label.setText(name)
                value_input.setText(value)
                lower_input.setText(lower)
                upper_input.setText(upper)
                fix_cb.setChecked(fix)

        # Special fixed checkboxes for certain models
        if model == 'Relax_MS':
            i = len(names) - 1  # last 'S'
            if i < numco:
                param_widget = self.row_widgets[row].layout().itemAt(i+1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox_L2.png); }")
        elif model == 'ASM':
            i = len(names) - 2  # one before last 'Num'
            if i < numco:
                param_widget = self.row_widgets[row].layout().itemAt(i+1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox_L2.png); }")
        elif model == 'Distr':
            for idx in [0, len(names)-2]:  # first 'par' and one before last 'Num'
                if idx < numco:
                    param_widget = self.row_widgets[row].layout().itemAt(idx+1).widget()
                    fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                    fix_cb.setChecked(True)
                    fix_cb.setEnabled(False)
                    fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox_L2.png); }")
        elif model == 'Corr':
            i = 0  # first 'par'
            if i < numco:
                param_widget = self.row_widgets[row].layout().itemAt(i+1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox_L2.png); }")

        self.row_params[row] = len(names)

    def clear_row_params(self, row):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col+1).widget()
            top_layout = param_widget.layout().itemAt(0).layout()
            name_label = top_layout.itemAt(0).widget()
            fix_cb = top_layout.itemAt(1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            bounds_layout = param_widget.layout().itemAt(2).layout()
            lower_input = bounds_layout.itemAt(0).widget()
            upper_input = bounds_layout.itemAt(1).widget()
            name_label.setText("")
            fix_cb.setChecked(False)
            fix_cb.setEnabled(True)
            fix_cb.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px; image: url(CheckBox.png); } QCheckBox::indicator:checked { image: url(CheckBox_L.png); }")
            value_input.setText("")
            lower_input.setText("")
            upper_input.setText("")

    def insert_row(self, row):
        # Simplified: just clear the current row and shift or something
        # For now, just clear
        self.clear_row_params(row)
        # TODO: Actually insert a new row in the grid

    def loadmod_pressed(self):
        pass

    def clean_model(self):
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

    def show_pressed(self):
        pass

    def showM_pressed(self):
        pass

    def save_pressed(self):
        pass

    def save_as_pressed(self):
        pass

if __name__ == '__main__':
    mp.freeze_support()
    app = QApplication(sys.argv)
    window = PhysicsApp()
    window.show()
    sys.exit(app.exec())