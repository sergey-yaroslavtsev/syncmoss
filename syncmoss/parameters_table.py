import os
import re
import sys
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QMenu, QWidgetAction,
)
from PySide6.QtCore import Qt, QRegularExpression, QEvent
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QRegularExpressionValidator, QAction
from syncmoss.constants import numro, numco, model_colors, number_of_baseline_parameters, contrast_text_color, NAT_WIDTH

# Natural line width as a table string: the default and the lower bound of every
# Lorentzian width L (a fitted line can never be narrower than the natural width).
_NAT = str(NAT_WIDTH)
from syncmoss.spectrum_io import calculate_backgrounds
from syncmoss.model_io import mod_len_def, append_model_via_dialog
from syncmoss.Library_window import open_library_model_dialog

# Absolute path to the icons directory.
# Used to build absolute url() paths in Qt stylesheets so they work both
# when running from source and when frozen by PyInstaller.
if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable).replace('\\', '/')
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
_ICONS_DIR = f"{_BASE_DIR}/icons"
_CB = f"{_ICONS_DIR}/CheckBox.png"
_CB_ = f"{_ICONS_DIR}/CheckBox_.png"
_CBL = f"{_ICONS_DIR}/CheckBox_L.png"
_CBL2 = f"{_ICONS_DIR}/CheckBox_L2.png"

MODEL_OPTIONS = [
    # polarized fittable models (each reduces to its former scalar form at
    # texture A = 0; SMS uses the full polarized readout, CMS the half-trace)
    'Singlet', 'Doublet', 'Sextet', 'MDGD', 'Relax_MS', 'Relax_2S',
    'Hamiltonian', 'ASM', 'SCDW',
    # NOT listed (deprecated, superseded by the textured 'Hamiltonian'):
    # 'Hamilton_mc' (== A, Am, Ah = 1, 1, 1) and 'Hamilton_pc' (== 0, 0, 0).
    # Their rows still build and evaluate if a hand-written file names them, and
    # syncmoss.legacy rewrites both to 'Hamiltonian' when a model file is opened.
    # presets / structural / utility
    'Be', 'KB_nano', 'Layer', 'Distr', 'Corr', 'Recon',
    # 'Library' picks a model out of the internal Library folder, 'Load model'
    # picks any .mdl through a file browser; both ADD the chosen model's
    # components to the current one (see model_io.load_model_from_path).
    'Variables', 'Expression', 'Library', 'Load model', 'Delete', 'Insert', 'Nbaseline',
    'Copy', 'Paste'
]

# Insert a separator line in the model dropdown BEFORE each of these entries,
# to visually split: fittable models | presets/utility.
_MENU_SEPARATOR_BEFORE = {'Be'}

# Rows that do not stand on their own: each one re-shoots the fittable component
# in front of it, replacing that component's parameter number 'par'. Consecutive
# ones form a single chain over one base component (see get_distribution_chains).
_DISTRIBUTION_MODELS = ('Distr', 'Corr', 'Recon')

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
        index = sum(main_window.params_table.row_params[:self.row]) + self.col
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

class ParametersTable(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.copied_model_name = str("None")
        self.copied_values = []
        self.copied_fixes = []
        self.table_of_fix = [[False] * numco for _ in range(numro)]
        self.row_fix_locked = [False] * numro
        self.row_params = [number_of_baseline_parameters] + [0] * (numro - 1)  # baseline has baseline parameters
        self.row_widgets = []
        self.params_layout = QVBoxLayout(self)
        self.params_layout.setSpacing(1)
        self.params_layout.setContentsMargins(1,1,1,1)

        # Create baseline row
        baseline_row = self.create_baseline_row()
        self.row_widgets.append(baseline_row)
        self.params_layout.addWidget(baseline_row)

        # Create model rows
        for row in range(1, numro):
            row_widget = self.create_model_row(row)
            self.row_widgets.append(row_widget)
            self.params_layout.addWidget(row_widget)

        # Set initial colors cycling through the available colors
        for r in range(1, len(self.row_widgets)):
            color = model_colors[(r - 1) % len(model_colors)]
            self.select_color(r, color)

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
        baseline_btn.clicked.connect(self.update_baseline_from_bg)
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
            fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CB}); }} QCheckBox::indicator:checked {{ image: url({_CBL}); }}")
            fix_cb.stateChanged.connect(self.on_fix_checkbox_changed)
            top_layout.addWidget(name_label)
            top_layout.addWidget(fix_cb)
            # Value input
            value_input = QLineEdit("")
            value_input.setFont(QFont('Arial', 10))
            value_input.setFixedWidth(80)
            validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
            value_input.setValidator(validator_value)
            value_input.textChanged.connect(lambda text, inp=value_input, r=0, c=col: self.on_value_changed(inp, r, c))
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
            if i in [1,2,3,4,5,6,7]:
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
        color = self.main_window.model_colors[row]
        bg_color = color if color.startswith('#') else self.get_color_from_code(color)
        color_btn.setStyleSheet(f"background-color: {bg_color}; color: {contrast_text_color(color)};")
        # Add menu to color_btn (menu order is deliberate; not the same order
        # as constants.model_colors, which is the auto-assignment sequence)
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
        model_options = MODEL_OPTIONS
        # Background colors for menu item groups
        _menu_colors = {
            'Insert': '#cc4444', 'Delete': '#cc4444',
            'Nbaseline': '#224477',
            'Library': "#319B00", 'Load model': "#1F6600",
            'Expression': '#5599cc', 'Variables': '#5599cc',
            'Distr': '#774488', 'Corr': '#774488', 'Recon': '#774488',
            'KB_nano': '#aaaaaa', 'Be': '#aaaaaa',
            'Layer': '#2a8a8a',
        }
        for option in model_options:
            if option in _MENU_SEPARATOR_BEFORE:
                model_menu.addSeparator()
            if option in _menu_colors:
                wa = QWidgetAction(self)
                lbl = QPushButton(option)
                lbl.setStyleSheet(f"QPushButton {{ background-color: {_menu_colors[option]}; color: white; border: none; padding: 4px 20px; text-align: left; font: 12px Arial; }} QPushButton:hover {{ background-color: {_menu_colors[option]}; color: yellow; }}")
                lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                lbl.clicked.connect(lambda checked=False, opt=option, btn=model_btn, m=model_menu: (self.select_model_by_button(opt, btn), m.close()))
                wa.setDefaultWidget(lbl)
                model_menu.addAction(wa)
            else:
                action = QAction(option, self)
                action.triggered.connect(lambda checked, opt=option, btn=model_btn: self.select_model_by_button(opt, btn))
                model_menu.addAction(action)
        model_btn.setMenu(model_menu)
        fix_model_btn = QPushButton("fix model")
        fix_model_btn.setFont(QFont('Arial', 12))
        fix_model_btn.clicked.connect(lambda checked=False, btn=fix_model_btn: self.toggle_fix_model_by_button(btn))
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
            fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CB}); }} QCheckBox::indicator:checked {{ image: url({_CBL}); }}")
            fix_cb.stateChanged.connect(self.on_fix_checkbox_changed)
            top_layout.addWidget(name_label)
            top_layout.addWidget(fix_cb)
            # Value input
            value_input = QLineEdit("")
            value_input.setFont(QFont('Arial', 10))
            value_input.setFixedWidth(80)
            validator_value = QRegularExpressionValidator(QRegularExpression(r'^(-?\d+(\.\d+)?|=\[\d+,-?\d+(\.\d+)?\])$'))
            value_input.setValidator(validator_value)
            value_input.textChanged.connect(lambda text, inp=value_input, r=row, c=col: self.on_value_changed(inp, r, c))
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
                if opt == 'Library':
                    self._open_library_model_dialog(r)
                    return
                if opt == 'Load model':
                    append_model_via_dialog(self.main_window, r)
                    return
                if opt == 'Copy':
                    self.copy_model_to_memory(r)
                    return
                if opt == 'Paste':
                    self.paste_model_from_memory(r)
                    return
                # Distr/Corr/Recon attach to the PRECEDING component, so they cannot
                # follow the baseline, a Layer marker, an Expression, or an empty
                # row. Corr is stricter: it may only follow a Distr/Corr/Recon
                # (the distribution it correlates onto). Recon places like Distr.
                # If the placement is invalid, do nothing (the row stays as it was).
                if opt in _DISTRIBUTION_MODELS:
                    prev_model = ''
                    if r > 0:
                        prev_start = self.row_widgets[r - 1].layout().itemAt(0).widget()
                        prev_model = prev_start.layout().itemAt(1).widget().text()
                    if opt == 'Corr':
                        allowed = prev_model in _DISTRIBUTION_MODELS
                    else:  # Distr / Recon
                        allowed = prev_model not in ('baseline', 'Layer', 'Expression', 'None', '')
                    if not allowed:
                        self.main_window.set_status(
                            f"'{opt}' cannot be placed after '{prev_model or 'nothing'}' "
                            f"(it must follow a "
                            + ("'Distr'/'Corr'/'Recon'" if opt == 'Corr' else "fittable component") + ").",
                            "orange",
                        )
                        return
                self.select_model(r, opt)
                return

    def _open_library_model_dialog(self, insert_row):
        open_library_model_dialog(self.main_window, self, insert_row, model_options=MODEL_OPTIONS)

    def _get_row_fix_states(self, row):
        states = []
        if row >= len(self.row_widgets):
            return states
        row_widget = self.row_widgets[row]
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col + 1).widget()
            fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
            states.append(fix_cb.isChecked())
        return states

    def _set_row_fix_states(self, row, states):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col + 1).widget()
            fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
            target = states[col] if col < len(states) else False
            fix_cb.setChecked(bool(target))

    def _set_fix_model_button_state(self, row, is_unfix):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        fix_model_btn = start_widget.layout().itemAt(2).widget()
        if is_unfix:
            fix_model_btn.setText('unfix model')
            fix_model_btn.setStyleSheet("background-color: #E6D86A; color: black;")
        else:
            fix_model_btn.setText('fix model')
            fix_model_btn.setStyleSheet("")

    def on_fix_checkbox_changed(self, state):
        _ = state
        sender_cb = self.sender()
        if sender_cb is None:
            return

        for row, row_widget in enumerate(self.row_widgets):
            if row >= len(self.row_fix_locked) or not self.row_fix_locked[row]:
                continue
            for col in range(numco):
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                if fix_cb is sender_cb:
                    if not fix_cb.isChecked():
                        self.row_fix_locked[row] = False
                        self._set_fix_model_button_state(row, False)
                    return

    def toggle_fix_model_by_button(self, btn):
        for row, rw in enumerate(self.row_widgets):
            start = rw.layout().itemAt(0).widget()
            if start.layout().itemAt(2).widget() == btn:
                if row >= len(self.row_fix_locked):
                    self.row_fix_locked.extend([False] * (row - len(self.row_fix_locked) + 1))
                if row >= len(self.table_of_fix):
                    self.table_of_fix.extend([[False] * numco for _ in range(row - len(self.table_of_fix) + 1)])

                if not self.row_fix_locked[row]:
                    self.table_of_fix[row] = self._get_row_fix_states(row)
                    self.row_fix_locked[row] = True
                    self._set_fix_model_button_state(row, True)
                    self._set_row_fix_states(row, [True] * numco)
                else:
                    self._set_row_fix_states(row, self.table_of_fix[row])
                    self.row_fix_locked[row] = False
                    self._set_fix_model_button_state(row, False)
                return

    def copy_model_to_memory(self, row):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()
        model_name = model_btn.text().strip() if model_btn.text() else str("None")

        values = []
        fixes = []
        n_params = self.row_params[row] if row < len(self.row_params) else 0
        for col in range(min(n_params, numco)):
            param_widget = row_widget.layout().itemAt(col + 1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
            values.append(value_input.text())
            fixes.append(fix_cb.isChecked())

        self.copied_model_name = model_name if model_name else str("None")
        self.copied_values = values
        self.copied_fixes = fixes
        self.main_window.set_status(f"Copied model from row {row}: {self.copied_model_name}", "green")

    def paste_model_from_memory(self, row):
        if self.copied_model_name == str("None"):
            self.main_window.set_status("Nothing is in the memory", "orange")
            return

        self.select_model(row, self.copied_model_name)
        if row >= len(self.row_widgets):
            return

        row_widget = self.row_widgets[row]
        n_params = self.row_params[row] if row < len(self.row_params) else 0
        limit = min(n_params, len(self.copied_values), numco)
        for col in range(limit):
            param_widget = row_widget.layout().itemAt(col + 1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
            value_input.setText(self.copied_values[col])
            if col < len(self.copied_fixes):
                fix_cb.setChecked(bool(self.copied_fixes[col]))

        if row < len(self.row_fix_locked) and self.row_fix_locked[row]:
            self._set_row_fix_states(row, [True] * numco)

        self.main_window.set_status(f"Pasted model to row {row}: {self.copied_model_name}", "green")

    def select_color(self, row, color):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        color_btn = start_widget.layout().itemAt(0).widget()
        bg_color = self.get_color_from_code(color)
        color_btn.setStyleSheet(f"background-color: {bg_color}; color: {contrast_text_color(color)};")
        # Update main_window.model_colors
        if hasattr(self.main_window, 'model_colors') and row < len(self.main_window.model_colors):
            self.main_window.model_colors[row] = color

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
        if row > 0 and row < len(self.row_fix_locked):
            self.row_fix_locked[row] = False
            self._set_fix_model_button_state(row, False)
        if model == 'Delete':
            if row == 0:
                return  # can't delete baseline
            deleted_start = sum(self.row_params[:row])
            deleted_count = self.row_params[row]
            # Keep color list aligned with table rows
            if hasattr(self.main_window, 'model_colors') and row < len(self.main_window.model_colors):
                self.main_window.model_colors.pop(row)
            # Clear the row
            self.clear_row_params(row)
            # Remove from layout and list
            row_widget = self.row_widgets.pop(row)
            self.params_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            # Remove from row_params
            self.row_params.pop(row)
            if row < len(self.table_of_fix):
                self.table_of_fix.pop(row)
            if row < len(self.row_fix_locked):
                self.row_fix_locked.pop(row)
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
            # Append a new empty row at the end to keep table size stable
            new_row_idx = len(self.row_widgets)
            new_row_widget = self.create_model_row(new_row_idx)
            self.row_widgets.append(new_row_widget)
            self.params_layout.addWidget(new_row_widget)
            self.row_params.append(0)
            self.table_of_fix.append([False] * numco)
            self.row_fix_locked.append(False)
            self._set_fix_model_button_state(new_row_idx, False)
            if hasattr(self.main_window, 'model_colors'):
                fallback = model_colors[(new_row_idx - 1) % len(model_colors)] if new_row_idx > 0 else model_colors[0]
                self.main_window.model_colors.append(fallback)
            # Refresh highlights after structural changes
            self.update_distr_corr_highlights()
        elif model == 'Insert':
            if row >= len(self.row_widgets):
                return
            inserted_start = sum(self.row_params[:row])
            # Insert color aligned with "next" row color (after insertion).
            if hasattr(self.main_window, 'model_colors'):
                copied_color = self.main_window.model_colors[row]
                self.main_window.model_colors.insert(row, copied_color)
            # Create new empty row
            new_row_widget = self.create_model_row(row)
            # Insert into list and layout
            self.row_widgets.insert(row, new_row_widget)
            self.params_layout.insertWidget(row, new_row_widget)
            # Insert 0 in row_params
            self.row_params.insert(row, 0)
            self.table_of_fix.insert(row, [False] * numco)
            self.row_fix_locked.insert(row, False)
            # Update references (no change yet)
            self.update_references(inserted_start, 0)
            # Set the model button to "insert" with red background and param boxes to light red
            self.set_row_styles_red(row)
            # Set the model button text to "insert"
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_btn.setText('insert')
            # Keep color equal to the next row's color
            if hasattr(self.main_window, 'model_colors') and row < len(self.main_window.model_colors):
                self.select_color(row, self.main_window.model_colors[row])
            # Update row numbers for subsequent rows
            for r in range(row+1, len(self.row_widgets)):
                row_widget = self.row_widgets[r]
                for col in range(numco):
                    param_widget = row_widget.layout().itemAt(col+1).widget()
                    top_layout = param_widget.layout().itemAt(0).layout()
                    name_label = top_layout.itemAt(0).widget()
                    name_label.row = r
            # Refresh highlights after structural changes
            self.update_distr_corr_highlights()
            # Grow the table widget so QScrollArea scrolls instead of squeezing rows.
            self.adjustSize()
            self.updateGeometry()
        else:
            # Normal model selection
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            # Reset styles
            self.reset_row_styles(row)
            old_count = self.row_params[row]
            # Clear the row (this will set button to "None")
            self.clear_row_params(row)
            # Now set the actual model name
            display_name = "Doublet" if model in ['Be', 'KB_nano'] else model
            model_btn.setText(display_name)
            # Auto-fill parameters
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
                    if model in ['Distr', 'Corr', 'Recon', 'Expression'] and col == self.row_params[row] - 1:
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
            
            # Always refresh highlights; deleting/changing any model can affect
            # Distr/Corr target parameter highlighting.
            self.update_distr_corr_highlights()

            # Apply expression expansion for Distr/Corr/Expression models
            # (their trailing column holds a free-text PDF/dependency/expression).
            if model in ['Distr', 'Corr', 'Expression']:
                last_col = self.row_params[row] - 1  # 0-based last meaningful column
                self._apply_expression_expansion(row, last_col)
            elif model == 'Recon':
                # The reconstruction weight vector is managed internally (the fit
                # determines it; it is viewed in the Distribution plot), so it is
                # NOT shown as an editable row field. Keep its fixed flat slot
                # (row_params is unchanged) but hide the trailing weights column so
                # the row shows only the 6 controls (par, L, R, Num, D_dif, D_dif2).
                self._hide_recon_weight_column(row)
            elif model == 'Layer':
                # Layer has no parameters; show it as one long, locked box
                # (red lock), purely cosmetic, like the expanded field of Distr.
                param_widget = row_widget.layout().itemAt(1).widget()
                name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
                name_label.setText('Layer')
                name_label.original_text = 'Layer'
                value_input = param_widget.layout().itemAt(1).widget()
                value_input.setText('')
                value_input.setValidator(None)
                value_input.setReadOnly(True)
                self._apply_expression_expansion(row, 0)

            if row < len(self.row_fix_locked) and self.row_fix_locked[row]:
                self._set_row_fix_states(row, [True] * numco)

    def model_name_at(self, row):
        """Model button text of a table row ('baseline' for row 0, 'None' for an
        empty row)."""
        if row < 0 or row >= len(self.row_widgets):
            return 'None'
        start_widget = self.row_widgets[row].layout().itemAt(0).widget()
        return start_widget.layout().itemAt(1).widget().text()

    def _value_input(self, row, col):
        """The value QLineEdit of one table cell."""
        param_widget = self.row_widgets[row].layout().itemAt(col + 1).widget()
        return param_widget.layout().itemAt(1).widget()

    def _name_label(self, row, col):
        """The parameter-name ClickableLabel of one table cell."""
        param_widget = self.row_widgets[row].layout().itemAt(col + 1).widget()
        return param_widget.layout().itemAt(0).layout().itemAt(0).widget()

    def _parameter_name(self, row, col):
        """Displayed name of one table cell ('' outside the table)."""
        if row is None or not (0 <= col < numco) or row >= len(self.row_widgets):
            return ''
        name_label = self._name_label(row, col)
        return name_label.original_text or name_label.text()

    def _par_value(self, row):
        """The 'par' of a Distr/Corr/Recon row as the fit sees it.

        read_model stores the field as a float and models.TImod indexes with
        ``int(p[V])``, so '2' and '2.0' are the same target. Returns None when the
        field is empty or holds a =[X,y] link instead of a plain number.
        """
        try:
            return int(float(self._value_input(row, 0).text().strip()))
        except ValueError:
            return None

    def get_distribution_chains(self):
        """Group the Distr/Corr/Recon rows by the component they attach to.

        A marker row re-shoots the fittable component in front of it, so a run of
        consecutive markers forms one chain over one base component — exactly the
        walk models.TImod does backwards from a 'Distr' to the first non-marker
        entry. Rows left at 'None' are invisible to the fit (read_model skips
        them), so they are skipped here too instead of cutting a chain in half.

        Returns:
            list of ``(base_row, [marker_row, ...])`` in table order. base_row is
            None for a chain with no component in front of it (only reachable
            from a hand-written file — the model menu refuses such a placement).
        """
        chains = []
        base_row = None
        markers = []
        for row in range(1, len(self.row_widgets)):
            model_name = self.model_name_at(row)
            if model_name == 'None':
                continue
            if model_name in _DISTRIBUTION_MODELS:
                markers.append(row)
                continue
            if markers:
                chains.append((base_row, markers))
                markers = []
            base_row = row
        if markers:
            chains.append((base_row, markers))
        return chains

    def get_conflicting_distr_targets(self):
        """Distr/Corr/Recon rows of one chain that claim the SAME 'par'.

        Every marker row of a chain writes into slot 'par' of the one parameter
        block of its base component (models.TImod: ``pN[int(p[V])] = X`` for the
        Distr/Recon axis, ``= f(X)`` for a Corr). Two rows with the same 'par'
        therefore overwrite each other and only the last one survives — silently,
        with no error from the fit. Show model / Fit refuse to start until the
        clash is resolved.

        Returns:
            list of dicts ``{'par', 'rows', 'models', 'base_row', 'base_model',
            'param'}`` — one entry per clashing 'par', in table order. 'rows'
            lists every claimant in table order; the caller flags rows[1:], since
            the first one may keep the target and only the later ones need moving.
        """
        conflicts = []
        for base_row, marker_rows in self.get_distribution_chains():
            rows_by_par = {}
            for row in marker_rows:
                par = self._par_value(row)
                if par is not None:
                    rows_by_par.setdefault(par, []).append(row)
            for par, rows in rows_by_par.items():
                if len(rows) > 1:
                    conflicts.append({
                        'par': par,
                        'rows': rows,
                        'models': [self.model_name_at(r) for r in rows],
                        'base_row': base_row,
                        'base_model': self.model_name_at(base_row) if base_row is not None else None,
                        'param': self._parameter_name(base_row, par),
                    })
        return conflicts

    def update_distr_corr_highlights(self):
        """Grey out every parameter driven by a Distr/Corr/Recon row.

        The 'par' of each marker row picks a parameter of its base component;
        that field stops being a free number (the distribution axis overwrites
        it), so it gets a grey frame and is made read-only.
        """
        # First, clear all grey frames
        self.clear_all_grey_frames()

        # Then, add grey frames for the target of every Distr/Corr/Recon row
        for base_row, marker_rows in self.get_distribution_chains():
            if base_row is None:
                continue
            for row in marker_rows:
                par = self._par_value(row)
                # par=1 means the second field of the base row (par=0 is its
                # amplitude, which a distribution shares rather than replaces).
                if par is None or not (1 <= par < numco):
                    continue
                self._name_label(base_row, par).setStyleSheet("border: 2px solid grey;")
                value_input = self._value_input(base_row, par)
                value_input.setReadOnly(True)
                value_input.setStyleSheet("background-color: lightgrey;")

    def clear_all_grey_frames(self):
        """Clear all grey frame highlights from parameter name labels and restore editability"""
        for row in range(len(self.row_widgets)):
            row_widget = self.row_widgets[row]
            for col in range(numco):
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                top_layout = param_widget.layout().itemAt(0).layout()
                name_label = top_layout.itemAt(0).widget()
                # Reset name label style (no border)
                name_label.setStyleSheet("")
                value_input = param_widget.layout().itemAt(1).widget()
                # Reset value input to editable if it has a name
                if name_label.text() != "":
                    value_input.setReadOnly(False)
                # A field marked red by mark_parameter_error/mark_expression_error
                # keeps its warning: only its own eventFilter (a click into the
                # field) clears that, so an unrelated refresh elsewhere in the
                # table cannot hide the reason a fit was refused.
                if not value_input.property('expression_error'):
                    value_input.setStyleSheet("")

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
                    elif 'p[' in text:
                        # Free-text expression field (Distr/Corr/Expression): shift
                        # every p[N] parameter reference the same way as =[X,y].
                        new_text, has_deleted_ref = self._shift_p_references(
                            text, start_index, delta)
                        if new_text != text:
                            value_input.setText(new_text)
                        if has_deleted_ref:
                            value_input.setStyleSheet("background-color: red;")
                        elif new_text != text:
                            value_input.setStyleSheet("")

    def _shift_p_references(self, text, start_index, delta):
        """Shift every p[N] parameter reference in an expression by the same
        rule update_references applies to =[X,y]: on insert (delta>0) bump refs
        at/after start_index; on delete (delta<0) drop refs inside the deleted
        window and bump refs above it. Returns (new_text, has_deleted_ref)."""
        has_deleted_ref = [False]

        def repl(match):
            ref_index = int(match.group(1))
            if delta < 0:  # deleting
                deleted_count = -delta
                if start_index <= ref_index < start_index + deleted_count:
                    # References a deleted parameter -> expression is now broken.
                    has_deleted_ref[0] = True
                    return match.group(0)
                if ref_index >= start_index + deleted_count:
                    return f"p[{ref_index + delta}]"
                return match.group(0)
            if delta > 0:  # inserting
                if ref_index >= start_index:
                    return f"p[{ref_index + delta}]"
            return match.group(0)

        new_text = re.sub(r'p\[(\d+)\]', repl, text)
        return new_text, has_deleted_ref[0]

    def _row_of_value_input(self, input_widget, fallback_row):
        """Row a value QLineEdit sits in RIGHT NOW.

        The textChanged lambdas capture the row index the widget was built with,
        but Insert/Delete (and a Library append, which inserts) move whole row
        widgets between positions and leave that index pointing at a neighbour.
        Locating the row widget itself keeps the live 'par' highlighting correct
        after such a move; the captured index is only a fallback for a widget no
        longer in the table (mid-deletion).
        """
        param_widget = input_widget.parentWidget()
        row_widget = param_widget.parentWidget() if param_widget is not None else None
        for row, candidate in enumerate(self.row_widgets):
            if candidate is row_widget:
                return row
        return fallback_row

    def on_value_changed(self, input_widget, row, col):
        """Handle value changes - check references and update grey frames for Distr/Corr"""
        # Check reference validity
        self.check_reference(input_widget)

        # Editing the 'par' of a Distr/Corr/Recon row (col 0) re-aims it at
        # another parameter of the component above: move the grey frame WHILE the
        # user types, not only when a model is picked from the menu.
        if col != 0:
            return
        row = self._row_of_value_input(input_widget, row)
        if row < len(self.row_widgets) and self.model_name_at(row) in _DISTRIBUTION_MODELS:
            self.update_distr_corr_highlights()

    def check_reference(self, input):
        text = input.text()
        if not re.match(r'=\[\d+,.+\]', text):
            input.setStyleSheet("")

    def auto_fill_params(self, row, model):
        # Auto-fill params based on model, mimicking original
        is_cms_mode = bool(getattr(self.main_window, 'MS_fit', None) is not None and self.main_window.MS_fit.isChecked())
        theta_default = '0' if is_cms_mode else '90'
        if model == 'Singlet':
            names = ['T', 'δ, mm/s', 'L, mm/s', 'G, mm/s']
            values = ['1.0', '0.0', _NAT, '0.1']
            lowers = ['0', '', _NAT, '0']
            uppers = ['', '', '', '']
            fixes = [False, False, True, False]
        # --- Polarized fittable models. Every anisotropic component carries the
        # orientation angles (theta_k, phi_h) of its axis in the lab frame
        # (theta_k from the beam k, phi_h from the polarization h) followed by the
        # uniaxial (fiber) texture parameter A in [-0.5, 1] (A=0 random powder ->
        # former scalar model, A=1 single crystal at the axis). 'Hamiltonian'
        # keeps its crystal angles plus the beam rotation alpha_k and needs THREE
        # order parameters (A, Am, Ah) for the mosaic of an anisotropic EFG.
        # Singlet is isotropic, so it has neither. The Faraday-active models
        # (Sextet, MDGD, Relax_2S) also carry a magnetic polar-order parameter Am
        # in [-1, 1] immediately after A (the net-magnetisation fraction
        # S1/sqrt((1+2A)/3) scaling the resolved sigma+- Faraday term; Am=0
        # unmagnetised, Am=1 fully magnetised at the axis). The angles (theta_k,
        # phi_h; alpha_k), A and Am are FIXED by default -- untick "fix" to refine. ---
        elif model == 'Doublet':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'θk, °', 'φh, °', 'A', 'G2/G1']
            values = ['1.0', '0.0', '1.0', _NAT, '0.1', theta_default, '0.0', '0', '1.0']
            lowers = ['0', '', '', _NAT, '0', '-180', '-360', '-0.5', '0']
            uppers = ['', '', '', '', '', '180', '360', '1', '']
            fixes = [False, False, False, True, False, True, True, True, True]  # theta_k, phi_h, A locked by default
        elif model == 'Sextet':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'θk, °', 'φh, °', 'A', 'Am', 'a+', 'a-', 'GH, T', 'I1/I3']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, '0.1', theta_default, '0.0', '0', '0', '0.0', '0.0', '0.0', '3.0']
            lowers = ['0', '', '', '', _NAT, '0', '-180', '-360', '-0.5', '-1', '', '', '0', '0']
            uppers = ['', '', '', '', '', '', '180', '360', '1', '1', '', '', '', '']
            fixes = [False, False, False, False, True, False, True, True, True, True, True, True, True, True]  # theta_k, phi_h, A, Am locked by default
        elif model == 'MDGD':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'GH, T', 'Dδε', 'DδH', 'DεH', 'θk, °', 'φh, °', 'A', 'Am', 'a+', 'a-', 'I1/I3']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, '0.1', '0.0', '0.0', '0.0', '0.0', theta_default, '0.0', '0', '0', '0.0', '0.0', '3.0']
            lowers = ['0', '', '', '', _NAT, '0', '0', '-1', '-1', '-1', '-180', '-360', '-0.5', '-1', '', '', '0']
            uppers = ['', '', '', '', '', '', '', '1', '1', '1', '180', '360', '1', '1', '', '', '']
            fixes = [False, False, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True]  # theta_k, phi_h, A, Am locked by default
        elif model == 'Hamiltonian':
            # Mosaic textured full Hamiltonian (supersedes Hamilton_mc/_pc).
            # (θH, φH) is B_hf in the EFG frame; (θ, φ) the lab reference axis in
            # the EFG frame -- the radiation field h for SMS, the beam k for CMS
            # -- and αk the beam rotation about h (SMS only). A/Am/Ah are the
            # mosaic order parameters: A = <P2(cos chi)> of the crystal wobble
            # about that reference axis, Am its polar order (Faraday), Ah the
            # order of the crystal azimuth about it. DEFAULT (0, 0, 0) = random
            # powder, i.e. exactly the former scalar 'Hamilton_pc' -- matching
            # every other model, whose A = 0 default is also its former scalar
            # form. (1, 1, 1) is the single crystal (former 'Hamilton_mc').
            # Because the reference-orientation angles (θ, φ, αk) do nothing in
            # the powder default, they are FIXED with A/Am/Ah; untick them
            # together with A (and Ah) when fitting an oriented sample.
            names = ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °', 'θ, °', 'φ, °', 'αk, °', 'A', 'Am', 'Ah']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, '0.1', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0', '0', '0']
            lowers = ['0', '', '', '', _NAT, '0', '-1', '-180', '-360', '-180', '-360', '-360', '-0.5', '-1', '0']
            uppers = ['', '', '', '', '', '', '1', '180', '360', '180', '360', '360', '1', '1', '1']
            fixes = [False, False, False, False, True, False, False, False, False, True, True, True, True, True, True]  # reference angles + A, Am, Ah locked by default
        elif model == 'Hamilton_mc':
            # DEPRECATED (superseded by 'Hamiltonian'); no longer in the dropdown,
            # kept so an old model file naming it still builds a valid row.
            names = ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °', 'θ, °', 'φ, °', 'αk, °']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, '0.1', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
            lowers = ['0', '', '', '', _NAT, '0', '-1', '-180', '-360', '-180', '-360', '-360']
            uppers = ['', '', '', '', '', '', '1', '180', '360', '180', '360', '360']
            fixes = [False, False, False, False, True, False, False, False, False, False, False, True]  # alpha_k locked by default
        elif model == 'Hamilton_pc':
            # DEPRECATED (superseded by 'Hamiltonian' at A = Am = Ah = 0).
            names = ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, '0.1', '0.0', '0.0', '0.0']
            lowers = ['0', '', '', '', _NAT, '0', '-1', '-180', '-360']
            uppers = ['', '', '', '', '', '', '1', '180', '360']
            fixes = [False, False, False, False, True, False, False, False, False]  # 4
        elif model == 'Relax_MS':
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'θk, °', 'φh, °', 'A', 'R', 'alfa', 'S']
            values = ['1.0', '0.0', '0.0', '33.0', _NAT, theta_default, '0.0', '0', '0.5', '1.0', '101']
            lowers = ['0', '', '', '', _NAT, '-180', '-360', '-0.5', '0', '0', '0.5']
            uppers = ['', '', '', '', '', '180', '360', '1', '', '100', '']
            fixes = [False, False, False, False, False, True, True, True, False, False, True]  # theta_k, phi_h, A locked by default
        elif model == 'Relax_2S':
            names = ['T', 'δ1, mm/s', 'ε1, mm/s', 'H1, T', 'δ2, mm/s', 'ε2, mm/s', 'H2, T', 'L, mm/s', 'θk, °', 'φh, °', 'A', 'Am', 'Ω12', 'P1/P2']
            values = ['1.0', '0.0', '0.0', '33.0', '0.0', '0.0', '-33.0', '0.1', theta_default, '0.0', '0', '0', '0.3', '1']
            lowers = ['', '', '', '', '', '', '', _NAT, '-180', '-360', '-0.5', '-1', '0', '0']
            uppers = ['', '', '', '', '', '', '', '', '180', '360', '1', '1', '', '']
            fixes = [False, False, False, False, False, False, False, False, True, True, True, True, False, True]  # theta_k, phi_h, A, Am locked by default
        elif model == 'ASM':
            names = ['T', 'δ, mm/s', 'εm, mm/s', 'εl, mm/s', 'His, T', 'Han, T', 'L, mm/s', 'G, mm/s', 'm', 'θk, °', 'φh, °', 'A', 'Num', 'I13', 'ω, °']
            values = ['1.0', '0.0', '0.0', '0.0', '30.0', '5.0', _NAT, '0.1', '0.1', theta_default, '0.0', '0', '25', '3.0', '90']
            lowers = ['0', '', '', '', '', '', _NAT, '0', '-1', '-180', '-360', '-0.5', '7', '0', '-360']
            uppers = ['', '', '', '', '', '', '', '', '1', '180', '360', '1', '', '', '360']
            fixes = [False, False, False, False, False, False, True, False, False, True, True, True, True, False, True]  # theta_k, phi_h, A, omega locked by default
        elif model == 'SCDW':
            # Spin/charge density wave (SpectrRelax SDW/CDW layout, 27 slots):
            # I, base delta/eps, base field H0, widths, spin-axis (theta_k,
            # phi_h), texture A, magnetic polar order Am, the field->shift
            # correlations KdH/KeH, CDW phase Phi, 8 odd field harmonics
            # (h1..h15), 4 even shift harmonics (d2..d8), grid resolution N/Gamma
            # (grid steps per line width -- the accuracy<->speed knob, replaces the
            # old 'Num' which is now fixed internally at models.SDW_NUM), ratio
            # I13. Unused harmonics stay fitted-fixed at 0. Am behaves as in the
            # Sextet model (Faraday polar order). See models.SDW_thick_terms.
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'H0, T', 'L, mm/s', 'G, mm/s', 'θk, °', 'φh, °', 'A', 'Am', 'KδH', 'KεH', 'Φ, °', 'h1, T', 'h3, T', 'h5, T', 'h7, T', 'h9, T', 'h11, T', 'h13, T', 'h15, T', 'd2, mm/s', 'd4, mm/s', 'd6, mm/s', 'd8, mm/s', 'N/Γ', 'I13']
            values = ['1.0', '0.0', '0.0', '30.0', _NAT, '0.1', theta_default, '0.0', '0', '0', '0.0', '0.0', '0.0', '5.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '4', '3.0']
            lowers = ['0', '', '', '', _NAT, '0', '-180', '-360', '-0.5', '-1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '1', '0']
            uppers = ['', '', '', '', '', '', '180', '360', '1', '1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
            fixes = [False, False, False, False, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False]  # theta_k, phi_h, A, Am, Num and all optional harmonics/correlations locked by default
        elif model == 'Be':
            # Impurity preset based on the polarized Doublet, loaded from Be.txt
            # (9-value polarized layout: T, d, e, L, G, theta_k, phi_h, A, G2/G1).
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'θk, °', 'φh, °', 'A', 'G2/G1']
            try:
                be_param = np.genfromtxt(os.path.join(self.main_window.params_dir, 'Be.txt'), delimiter='\t')
                values = [str(be_param[i]) for i in range(9)]
                self.main_window.set_status("Be.txt loaded successfully.")
            except:
                values = ['0.048', '0.103', '-0.259', _NAT, '0.105', '90', '0', '-0.1880264375', '1.0']
                self.main_window.set_status("Default Be values used. Could not load Be.txt.")
            lowers = ['0', '', '', _NAT, '0', '-180', '-360', '-0.5', '0']
            uppers = ['', '', '', '', '', '180', '360', '1', '']
            fixes = [True] * 9
        elif model == 'KB_nano':
            # Impurity preset based on the polarized Doublet, loaded from KB.txt
            # (9-value polarized layout, as for 'Be').
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'θk, °', 'φh, °', 'A', 'G2/G1']
            try:
                kb_param = np.genfromtxt(os.path.join(self.main_window.params_dir, 'KB.txt'), delimiter='\t')
                values = [str(kb_param[i]) for i in range(9)]
                self.main_window.set_status("KB.txt loaded successfully.")
            except:
                values = ['0.065', '0.234', '0.37', _NAT, '0.373', '90', '0', '0', '1.0']
                self.main_window.set_status("Default KB values used. Could not load KB.txt.")
            lowers = ['0', '', '', _NAT, '0', '-180', '-360', '-0.5', '0']
            uppers = ['', '', '', '', '', '180', '360', '1', '']
            fixes = [True] * 9
        elif model == 'Variables':
            # Fill all columns with V1, V2, ...
            names = [f'V{i+1}' for i in range(numco)]
            values = ['0'] * numco
            lowers = [''] * numco
            uppers = [''] * numco
            fixes = [True] * numco
        elif model == 'Nbaseline':
            # Similar to baseline but auto-fill Ns with BG from corresponding spectrum
            names = ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr']
            
            # Calculate which Nbaseline this is (1st, 2nd, etc.)
            nbaseline_count = 0
            for r in range(row):
                row_widget = self.row_widgets[r]
                start_widget = row_widget.layout().itemAt(0).widget()
                model_btn = start_widget.layout().itemAt(1).widget()
                if model_btn.text() == 'Nbaseline':
                    nbaseline_count += 1
            
            # Calculate BG for the corresponding spectrum (nbaseline_count is 0-based index)
            spectrum_index = nbaseline_count + 1  # First Nbaseline corresponds to 2nd spectrum
            
            if hasattr(self.main_window, 'path_list') and self.main_window.path_list and spectrum_index < len(self.main_window.path_list):
                try:
                    spectrum_path = self.main_window.path_list[spectrum_index]
                    backgrounds = calculate_backgrounds([spectrum_path], self.main_window.calibration_path)
                    if backgrounds and len(backgrounds) > 0:
                        BG = int(round(backgrounds[0]))
                        values = [str(BG), '0', '0', '0', '0', '0', '0', '0']
                    else:
                        values = ['10000', '0', '0', '0', '0', '0', '0', '0']
                except:
                    values = ['10000', '0', '0', '0', '0', '0', '0', '0']
            else:
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
            values = ['1.0', '0.0', '0.0', '15.0', _NAT, '0.1', '5', '1', '-1', '90', '100']
            lowers = ['0', '', '', '', _NAT, '0', '0', '0', '0', '0', '1']
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
        elif model == 'Recon':
            # Reconstructor of the distribution: par/L/R/Num as for Distr, then two
            # smoothness regularization knobs D_dif (1st derivative) and D_dif2 (2nd
            # derivative) in [0,1] (0 = free, 1 = maximally smooth), then the free
            # weight vector (left empty -> a uniform start is used; the fit fills it).
            names = ['par', 'L', 'R', 'Num', 'D_dif', 'D_dif2', 'weights']
            values = ['1', '0', '1', '20', '0', '0', '']  # par depends on previous
            lowers = ['1', '', '', '1', '0', '0', '']
            uppers = ['', '', '', '1000', '1', '1', '']
            fixes = [True, False, False, True, True, True, True]
        elif model == 'Layer':
            # Layer boundary marker: no parameters.
            self.row_params[row] = 0
            return
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

        # Structural parameters (e.g. a Distr's target index or point count) are
        # hard-locked: checked, disabled, with the "locked" indicator icon.
        def _hard_lock_fix_checkbox(col):
            if col < numco:
                param_widget = self.row_widgets[row].layout().itemAt(col + 1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")

        if model == 'Relax_MS':
            _hard_lock_fix_checkbox(len(names) - 1)      # last 'S'
        elif model == 'ASM':
            _hard_lock_fix_checkbox(len(names) - 3)      # 'Num' (now followed by I13, ω)
        elif model == 'SCDW':
            _hard_lock_fix_checkbox(len(names) - 2)      # 'N/Γ' grid resolution (followed by I13)
        elif model == 'Distr':
            for idx in [0, len(names) - 2]:              # first 'par' and 'Num'
                _hard_lock_fix_checkbox(idx)
        elif model == 'Corr':
            _hard_lock_fix_checkbox(0)                   # first 'par'
        elif model == 'Recon':
            for idx in [0, 3, 4, 5]:                     # 'par', 'Num', 'D_dif', 'D_dif2'
                _hard_lock_fix_checkbox(idx)

        self.row_params[row] = len(names)

    def _apply_expression_expansion(self, row, last_col):
        """
        For Distr/Corr/Expression models, expand the last meaningful parameter's
        value_input to visually occupy the space of all subsequent columns.
        Uses layout stretch factors instead of size policies to avoid breaking row height.
        
        Args:
            row: Row index
            last_col: 0-based index of the last meaningful parameter column
        """
        row_widget = self.row_widgets[row]
        row_layout = row_widget.layout()

        # Hide param_widgets after last_col
        for col in range(last_col + 1, numco):
            row_layout.itemAt(col + 1).widget().setVisible(False)

        # Set stretch: only the expression column gets stretch=1
        for col in range(numco):
            row_layout.setStretch(col + 1, 1 if col == last_col else 0)

        # Expand value_input in the expression column
        param_widget = row_layout.itemAt(last_col + 1).widget()
        value_input = param_widget.layout().itemAt(1).widget()
        value_input.setMinimumWidth(80)
        value_input.setMaximumWidth(16777215)

        # Expand name_label so long names are fully visible
        top_layout = param_widget.layout().itemAt(0).layout()
        name_label = top_layout.itemAt(0).widget()
        name_label.setMinimumWidth(50)
        name_label.setMaximumWidth(16777215)

        # Disable fix checkbox and use L2 icon (permanently fixed indicator)
        fix_cb = top_layout.itemAt(1).widget()
        fix_cb.setEnabled(False)
        fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")

        bounds_layout = param_widget.layout().itemAt(2).layout()
        bounds_layout.itemAt(0).widget().setReadOnly(True)
        bounds_layout.itemAt(1).widget().setReadOnly(True)

    def _hide_recon_weight_column(self, row):
        """Hide a Recon row's trailing weight-vector column.

        The column still exists (it holds the serialized weights so they round-trip
        through .mdl save/load and carry the fitted reconstruction), and its flat
        p-slot is still counted in ``row_params`` — it is only made invisible so the
        visible row shows just the six control parameters. The reconstruction itself
        is viewed in the Distribution plot.
        """
        weight_col = self.row_params[row] - 1  # last Recon slot = weights
        if 0 <= weight_col < numco:
            self.row_widgets[row].layout().itemAt(weight_col + 1).widget().setVisible(False)

    def _restore_normal_columns(self, row):
        """
        Restore normal column layout for a row (undo expression expansion).
        """
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        row_layout = row_widget.layout()
        for col in range(numco):
            param_widget = row_layout.itemAt(col + 1).widget()
            # Show all param_widgets
            param_widget.setVisible(True)
            # Reset stretch
            row_layout.setStretch(col + 1, 0)

            # Restore value_input fixed width
            value_input = param_widget.layout().itemAt(1).widget()
            value_input.setFixedWidth(80)

            # Restore name_label fixed width
            top_layout = param_widget.layout().itemAt(0).layout()
            name_label = top_layout.itemAt(0).widget()
            name_label.setFixedWidth(50)

            # Re-enable fix checkbox and restore normal style
            fix_cb = top_layout.itemAt(1).widget()
            fix_cb.setEnabled(True)
            fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CB}); }} QCheckBox::indicator:checked {{ image: url({_CBL}); }}")

            # Re-enable bounds
            bounds_layout = param_widget.layout().itemAt(2).layout()
            bounds_layout.itemAt(0).widget().setReadOnly(False)
            bounds_layout.itemAt(1).widget().setReadOnly(False)

    def clear_row_params(self, row):
        if row >= len(self.row_widgets):
            return
        # Restore normal column layout first (undo any expression expansion)
        self._restore_normal_columns(row)
        row_widget = self.row_widgets[row]
        # Reset model button to "None"
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()
        model_btn.setText("None")
        # Keep the active-parameter count in step with the "None" button, else a
        # later show/fit (get_empty_parameter_slots) still walks the now-empty
        # fields and flags them as missing. select_model captures old/new counts
        # around this call, so zeroing here is safe for that path too.
        if row < len(self.row_params):
            self.row_params[row] = 0
        # Clear all parameter values
        for col in range(numco):
            param_widget = row_widget.layout().itemAt(col+1).widget()
            top_layout = param_widget.layout().itemAt(0).layout()
            name_label = top_layout.itemAt(0).widget()
            fix_cb = top_layout.itemAt(1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            bounds_layout = param_widget.layout().itemAt(2).layout()
            lower_input = bounds_layout.itemAt(0).widget()
            upper_input = bounds_layout.itemAt(1).widget()
            name_label.setStyleSheet("")
            name_label.setText("")
            fix_cb.setChecked(False)
            fix_cb.setEnabled(True)
            fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CB}); }} QCheckBox::indicator:checked {{ image: url({_CBL}); }}")
            value_input.setStyleSheet("")
            value_input.setText("")
            lower_input.setText("")
            upper_input.setText("")

    def update_baseline_from_bg(self):
        """Update baseline Ns parameter based on current spectrum background"""
        # Get the first spectrum path
        if not hasattr(self.main_window, 'path_list') or not self.main_window.path_list:
            self.main_window.set_status("No spectrum loaded. Cannot update baseline.", "orange")
            return
        
        # Calculate BG directly for the first spectrum
        first_spectrum = self.main_window.path_list[0]
        backgrounds = calculate_backgrounds([first_spectrum], self.main_window.calibration_path)
        
        if not backgrounds or len(backgrounds) == 0:
            self.main_window.set_status("Could not calculate background.", "red")
            return
        
        BG = backgrounds[0]
        
        # Get baseline row (row 0)
        row_widget = self.row_widgets[0]
        
        # Get Ns (parameter 0) and Nnr (parameter 4)
        ns_widget = row_widget.layout().itemAt(1).widget()  # column 0 + 1
        ns_value_input = ns_widget.layout().itemAt(1).widget()
        
        nnr_widget = row_widget.layout().itemAt(5).widget()  # column 4 + 1
        nnr_value_input = nnr_widget.layout().itemAt(1).widget()
        nnr_text = nnr_value_input.text().strip()
        
        # Calculate new Ns based on Nnr
        if nnr_text.startswith('=[0,'):
            # Case: Nnr = =[0,X] -> Ns = BG / (1 + X)
            match = re.match(r'=\[0,(.+)\]', nnr_text)
            if match:
                try:
                    X = float(match.group(1))
                    new_ns = BG / (1 + X)
                except:
                    new_ns = BG
            else:
                new_ns = BG
        elif nnr_text and not nnr_text.startswith('='):
            # Case: Nnr is a constant -> Ns = BG - Nnr
            try:
                nnr_value = float(nnr_text)
                new_ns = BG - nnr_value
            except:
                new_ns = BG
        else:
            # Default case: Ns = BG
            new_ns = BG
        
        # Round to integer and ensure minimum of 1
        new_ns = max(1, int(round(new_ns)))
        
        # Update Ns value
        ns_value_input.setText(str(new_ns))
        self.main_window.set_status(f"Baseline Ns updated to {new_ns} (BG={int(round(BG))})", "green")
    
    def get_current_colors(self):
        """Get current colors from all table rows (fresh read after delete/insert)"""
        colors = []
        
        # First row (baseline) doesn't have a color button
        # Add a placeholder for experimental data color
        colors.append(self.main_window.model_colors[0] if hasattr(self.main_window, 'model_colors') else 'blue')
        
        # Read colors from model rows (rows 1+)
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            # The color button is in start_widget (first item in row layout)
            start_widget = row_widget.layout().itemAt(0).widget()
            # Color button is the first button in start_widget layout
            color_btn = start_widget.layout().itemAt(0).widget()
            
            # Extract color from stylesheet
            stylesheet = color_btn.styleSheet()
            if 'background-color:' in stylesheet:
                # Parse "background-color: <color>;" from stylesheet
                match = re.search(r'background-color:\s*([^;]+);', stylesheet)
                if match:
                    color = match.group(1).strip()
                    colors.append(color)
                else:
                    colors.append('white')  # fallback
            else:
                colors.append('white')  # fallback
        
        return colors
    
    def get_model_list(self):
        """
        Get list of model names from the parameters table.
        
        Returns:
            list: List of model names (excluding 'None')
        """
        models = ['baseline']  # First row is always baseline
        
        # Read model names from rows 1+
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            
            if model_name != 'None':
                models.append(model_name)
        
        return models
    
    def get_expression_texts(self):
        """
        Get expression texts for Distr/Corr/Expression models.
        
        Returns:
            dict: {component_index: expression_text} where component_index matches
                  the index in get_model_list() output.
        """
        texts = {}
        component_idx = 0  # 0 is baseline
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            if model_name != 'None':
                component_idx += 1
                if model_name == 'Expression':
                    param_widget = row_widget.layout().itemAt(1).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    texts[component_idx] = value_input.text()
                elif model_name == 'Distr':
                    param_widget = row_widget.layout().itemAt(5).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    texts[component_idx] = value_input.text()
                elif model_name == 'Corr':
                    param_widget = row_widget.layout().itemAt(2).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    texts[component_idx] = value_input.text()
                elif model_name == 'Recon':
                    param_widget = row_widget.layout().itemAt(7).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    texts[component_idx] = value_input.text()
        return texts

    # Column (layout index) of the free-text field per model type: an Expression /
    # PDF / dependency string, or (Recon) the reconstruction weight vector.
    _EXPRESSION_COLUMNS = {'Expression': 1, 'Distr': 5, 'Corr': 2, 'Recon': 7}

    def get_expression_rows(self):
        """
        Table rows holding free-text expressions, per model kind.

        Returns:
            dict: {'Expression': [row, ...], 'Distr': [...], 'Corr': [...]} in
                  table order — the same order read_model() collects the
                  Expr/Distri/Cor lists in.
        """
        rows = {'Expression': [], 'Distr': [], 'Corr': [], 'Recon': []}
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            if model_name in rows:
                rows[model_name].append(row_idx)
        return rows

    def get_empty_parameter_slots(self):
        """Active numeric parameter value fields left empty.

        This happens when a =[X,y] reference is deleted: update_references()
        clears the referring field. read_model() would silently read an empty
        field as 0.0, so show/fit must be blocked until it is filled. The
        free-text expression column of Distr/Corr/Expression rows is excluded —
        its emptiness is reported by validate_user_expressions instead.

        Returns:
            list of dicts ``{'row', 'col', 'param', 'model'}``.
        """
        empties = []
        for row in range(len(self.row_widgets)):
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            expr_layout_col = self._EXPRESSION_COLUMNS.get(model_name)
            for col in range(self.row_params[row]):
                if expr_layout_col is not None and col == expr_layout_col - 1:
                    continue  # free-text expression field, handled elsewhere
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                if not value_input.text().strip():
                    name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
                    empties.append({'row': row, 'col': col,
                                    'param': name_label.original_text or name_label.text(),
                                    'model': model_name})
        return empties

    def mark_parameter_error(self, row, col):
        """Turn an empty/invalid numeric parameter field red, cleared as soon as
        the user clicks into it (reuses the expression-error eventFilter)."""
        if row < 0 or row >= len(self.row_widgets):
            return
        param_widget = self.row_widgets[row].layout().itemAt(col + 1).widget()
        value_input = param_widget.layout().itemAt(1).widget()
        value_input.setStyleSheet("background-color: red; color: white;")
        value_input.setProperty('expression_error', True)
        value_input.installEventFilter(self)

    def expression_value_input(self, row):
        """The QLineEdit holding the free-text expression of an
        Expression/Distr/Corr row (None for other rows)."""
        if row < 0 or row >= len(self.row_widgets):
            return None
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()
        col = self._EXPRESSION_COLUMNS.get(model_btn.text())
        if col is None or col >= row_widget.layout().count():
            return None
        param_widget = row_widget.layout().itemAt(col).widget()
        return param_widget.layout().itemAt(1).widget()

    def mark_expression_error(self, row):
        """Turn an invalid expression field red. The highlight is removed the
        moment the user clicks into (or focuses) the field — see eventFilter."""
        value_input = self.expression_value_input(row)
        if value_input is None:
            return
        value_input.setStyleSheet("background-color: red; color: white;")
        value_input.setProperty('expression_error', True)
        value_input.installEventFilter(self)

    def eventFilter(self, obj, event):
        # Clear the red invalid-expression highlight as soon as the user clicks
        # into or focuses the offending field.
        if isinstance(obj, QLineEdit) and obj.property('expression_error'):
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
                obj.setStyleSheet("")
                obj.setProperty('expression_error', False)
                obj.removeEventFilter(self)
        return super().eventFilter(obj, event)

    def get_parameter_names(self):
        """
        Get list of parameter names for each component.
        
        Returns:
            list: List of lists, where each inner list contains parameter names for one component
        """
        param_names_list = []
        
        # Baseline parameters (row 0)
        baseline_names = []
        baseline_row = self.row_widgets[0]
        for j in range(1, number_of_baseline_parameters + 1):
            param_widget = baseline_row.layout().itemAt(j).widget()
            # Get name label (first widget in param layout)
            name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
            baseline_names.append(name_label.text())
        param_names_list.append(baseline_names)
        
        # Model parameters (rows 1+)
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            
            if model_name != 'None':
                param_names = []
                LenM = mod_len_def(model_name, include_special=True) + 1
                
                for j in range(1, min(LenM, numco + 1)):
                    if j < row_widget.layout().count():
                        param_widget = row_widget.layout().itemAt(j).widget()
                        # Get name label
                        name_label = param_widget.layout().itemAt(0).layout().itemAt(0).widget()
                        param_names.append(name_label.text())
                
                param_names_list.append(param_names)

        return param_names_list

    def get_link_snapshot(self):
        """Snapshot parameter-link fields (``=[X,Y]``) keyed by flat parameter index.

        The flat index matches the parameter traversal used by read_model() and
        the results table (baseline first, then each active model row in table
        order, including special-model placeholder slots).

        Returns:
            dict[int, str]: ``{flat_param_index: '=[X,Y]'}`` for link fields.
        """
        links = {}
        param_index = 0

        for row_idx, row_widget in enumerate(self.row_widgets):
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            if row_idx > 0 and model_name == 'None':
                continue

            if row_idx == 0:
                num_params = number_of_baseline_parameters
            else:
                num_params = mod_len_def(model_name, include_special=True)

            for col in range(num_params):
                param_widget = row_widget.layout().itemAt(col + 1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                text = value_input.text().strip()
                if text.startswith('=['):
                    links[param_index] = text
                param_index += 1

        return links