import os
import re
import sys
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QMenu, QWidgetAction,
    QDialog, QListWidget, QDialogButtonBox, QMessageBox
)
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QRegularExpressionValidator, QAction
from src.constants import numro, numco, model_colors, number_of_baseline_parameters
from src.spectrum_io import calculate_backgrounds
from src.model_io import mod_len_def, load_model_from_path

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
        text_color = 'black' if color in ['red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] or color.startswith('#') else 'white'
        color_btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")
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
        model_options = ['Singlet', 'Doublet', 'Sextet', 'MDGD', 'Relax_MS', 'Relax_2S', 'Hamilton_mc', 'Hamilton_pc', 'ASM', 'Be', 'KB_nano', 'Distr', 'Corr', 'Variables', 'Expression', 'Library', 'Delete', 'Insert', 'Nbaseline', 'Copy', 'Paste']
        # Background colors for menu item groups
        _menu_colors = {
            'Insert': '#cc4444', 'Delete': '#cc4444',
            'Nbaseline': '#224477',
            'Library': "#319B00",
            'Expression': '#5599cc', 'Variables': '#5599cc',
            'Distr': '#774488', 'Corr': '#774488',
            'KB_nano': '#aaaaaa', 'Be': '#aaaaaa',
        }
        for option in model_options:
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
                if opt == 'Copy':
                    self.copy_model_to_memory(r)
                    return
                if opt == 'Paste':
                    self.paste_model_from_memory(r)
                    return
                # Validate Corr can only be selected if previous model is Distr
                if opt == 'Corr' and r > 0:
                    prev_row_widget = self.row_widgets[r-1]
                    prev_start = prev_row_widget.layout().itemAt(0).widget()
                    prev_model_btn = prev_start.layout().itemAt(1).widget()
                    prev_model = prev_model_btn.text()
                    if prev_model != 'Distr' and prev_model != 'Corr':
                        self.main_window.log.setPlainText("Error: 'Corr' can only be selected if previous model is 'Distr' or 'Corr'")
                        self.main_window.log.setStyleSheet("color: red;")
                        return
                self.select_model(r, opt)
                return

    def _open_library_model_dialog(self, insert_row):
        """Open internal Library list and load selected model file."""
        library_dir = os.path.join(self.main_window.dir_path, 'Library')
        if not os.path.isdir(library_dir):
            self.main_window.log.setPlainText(f"Library folder not found: {library_dir}")
            self.main_window.log.setStyleSheet("color: red;")
            return

        model_files = sorted([
            f for f in os.listdir(library_dir)
            if os.path.isfile(os.path.join(library_dir, f))
        ])
        if not model_files:
            self.main_window.log.setPlainText("Library folder is empty")
            self.main_window.log.setStyleSheet("color: orange;")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Library")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget(dialog)
        list_widget.addItems(model_files)
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            current_item = list_widget.currentItem()
            if current_item is None:
                QMessageBox.warning(self, "Library", "Please select a model file.")
                return
            selected_file = os.path.join(library_dir, current_item.text())
            load_model_from_path(self.main_window, selected_file, insert_row=insert_row)

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
        self.main_window.log.setPlainText(f"Copied model from row {row}: {self.copied_model_name}")
        self.main_window.log.setStyleSheet("color: green;")

    def paste_model_from_memory(self, row):
        if self.copied_model_name == str("None"):
            self.main_window.log.setPlainText("Nothing is in the memory")
            self.main_window.log.setStyleSheet("color: orange;")
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

        self.main_window.log.setPlainText(f"Pasted model to row {row}: {self.copied_model_name}")
        self.main_window.log.setStyleSheet("color: green;")

    def select_color(self, row, color):
        if row >= len(self.row_widgets):
            return
        row_widget = self.row_widgets[row]
        start_widget = row_widget.layout().itemAt(0).widget()
        color_btn = start_widget.layout().itemAt(0).widget()
        bg_color = self.get_color_from_code(color)
        text_color = 'black' if color in ['red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] else 'white'
        color_btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")
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
            
            # Always refresh highlights; deleting/changing any model can affect
            # Distr/Corr target parameter highlighting.
            self.update_distr_corr_highlights()

            # Apply expression expansion for Distr/Corr/Expression models
            if model in ['Distr', 'Corr', 'Expression']:
                last_col = self.row_params[row] - 1  # 0-based last meaningful column
                self._apply_expression_expansion(row, last_col)

            if row < len(self.row_fix_locked) and self.row_fix_locked[row]:
                self._set_row_fix_states(row, [True] * numco)

    def update_distr_corr_highlights(self):
        """Update grey frame highlights for parameters referenced by Distr/Corr models"""
        # First, clear all grey frames
        self.clear_all_grey_frames()
        
        # Then, add grey frames for all Distr/Corr models
        for row in range(len(self.row_widgets)):
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            
            if model_name in ['Distr', 'Corr']:
                # Find the last non-Distr/non-Corr model before this row
                target_row = None
                for k in range(row - 1, -1, -1):
                    prev_row_widget = self.row_widgets[k]
                    prev_start = prev_row_widget.layout().itemAt(0).widget()
                    prev_model_btn = prev_start.layout().itemAt(1).widget()
                    prev_model = prev_model_btn.text()
                    if prev_model not in ['Distr', 'Corr']:
                        target_row = k
                        break
                
                if target_row is not None:
                    # Get the 'par' value (first parameter)
                    param_widget = row_widget.layout().itemAt(1).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    par_text = value_input.text().strip()
                    
                    try:
                        par_value = int(par_text)
                        if par_value >= 1:
                            # The parameter index in the target row is par_value (since it's 1-based and we add 1 to get actual column)
                            # Column index = par_value (0-based counting where par=1 means first parameter)
                            param_col = par_value  # par=1 means column 1 (index 1), which is the second widget
                            
                            # Set grey frame on the name label and make value uneditable
                            if param_col < numco:
                                target_param_widget = self.row_widgets[target_row].layout().itemAt(param_col + 1).widget()
                                top_layout = target_param_widget.layout().itemAt(0).layout()
                                name_label = top_layout.itemAt(0).widget()
                                name_label.setStyleSheet("border: 2px solid grey;")
                                # Make value input read-only and grey
                                value_input = target_param_widget.layout().itemAt(1).widget()
                                value_input.setReadOnly(True)
                                value_input.setStyleSheet("background-color: lightgrey;")
                    except ValueError:
                        pass  # Invalid par value, skip

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
                # Reset value input to editable if it has a name
                value_input = param_widget.layout().itemAt(1).widget()
                if name_label.text() != "":
                    value_input.setReadOnly(False)
                    value_input.setStyleSheet("")
                else:
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

    def on_value_changed(self, input_widget, row, col):
        """Handle value changes - check references and update grey frames for Distr/Corr"""
        # Check reference validity
        self.check_reference(input_widget)
        
        # If this is a Distr/Corr model and we're changing the 'par' parameter (col 0)
        # update the grey frame highlights
        if row < len(self.row_widgets) and col == 0:
            row_widget = self.row_widgets[row]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            if model_name in ['Distr', 'Corr']:
                self.update_distr_corr_highlights()

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
                be_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'parameters', 'Be.txt'), delimiter='\t')
                values = [str(be_param[i]) for i in range(7)]
                self.main_window.log.setPlainText("Be.txt loaded successfully.")
            except:
                values = ['0.048', '0.103', '-0.259', '0.098', '0.105', '0.265', '1.0']
                self.main_window.log.setPlainText("Default Be values used. Could not load Be.txt.")
            lowers = ['0', '', '', '0.098', '0', '0', '0']
            uppers = ['', '', '', '', '', '1', '']
            fixes = [True] * 7
        elif model == 'KB_nano':
            # Based on Doublet, load from KB.txt
            names = ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1']
            try:
                kb_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'parameters', 'KB.txt'), delimiter='\t')
                values = [str(kb_param[i]) for i in range(7)]
                self.main_window.log.setPlainText("KB.txt loaded successfully.")
            except:
                values = ['0.065', '0.234', '0.37', '0.098', '0.373', '0.5', '1.0']
                self.main_window.log.setPlainText("Default KB values used. Could not load KB.txt.")
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
                fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")
        elif model == 'ASM':
            i = len(names) - 2  # one before last 'Num'
            if i < numco:
                param_widget = self.row_widgets[row].layout().itemAt(i+1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")
        elif model == 'Distr':
            for idx in [0, len(names)-2]:  # first 'par' and one before last 'Num'
                if idx < numco:
                    param_widget = self.row_widgets[row].layout().itemAt(idx+1).widget()
                    fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                    fix_cb.setChecked(True)
                    fix_cb.setEnabled(False)
                    fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")
        elif model == 'Corr':
            i = 0  # first 'par'
            if i < numco:
                param_widget = self.row_widgets[row].layout().itemAt(i+1).widget()
                fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
                fix_cb.setChecked(True)
                fix_cb.setEnabled(False)
                fix_cb.setStyleSheet(f"QCheckBox::indicator {{ width: 30px; height: 30px; image: url({_CBL2}); }}")

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
            self.main_window.log.setPlainText("No spectrum loaded. Cannot update baseline.")
            self.main_window.log.setStyleSheet("color: orange;")
            return
        
        # Calculate BG directly for the first spectrum
        first_spectrum = self.main_window.path_list[0]
        backgrounds = calculate_backgrounds([first_spectrum], self.main_window.calibration_path)
        
        if not backgrounds or len(backgrounds) == 0:
            self.main_window.log.setPlainText("Could not calculate background.")
            self.main_window.log.setStyleSheet("color: red;")
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
        self.main_window.log.setPlainText(f"Baseline Ns updated to {new_ns} (BG={int(round(BG))})")
        self.main_window.log.setStyleSheet("color: green;")
    
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
        return texts

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
    
    def get_parameter_values(self):
        """
        Get all parameter values as a flat list.
        
        Returns:
            list: List of float values for all parameters (baseline + models)
        """
        values = []
        
        # Baseline parameters (row 0)
        baseline_row = self.row_widgets[0]
        for j in range(1, number_of_baseline_parameters + 1):
            param_widget = baseline_row.layout().itemAt(j).widget()
            # Get value input (second widget in param layout)
            value_input = param_widget.layout().itemAt(1).widget()
            try:
                values.append(float(value_input.text()))
            except (ValueError, AttributeError):
                values.append(0.0)
        
        # Model parameters (rows 1+)
        for row_idx in range(1, len(self.row_widgets)):
            row_widget = self.row_widgets[row_idx]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            
            if model_name != 'None':
                LenM = mod_len_def(model_name, include_special=False) + 1
                
                for j in range(1, min(LenM, numco + 1)):
                    if j < row_widget.layout().count():
                        param_widget = row_widget.layout().itemAt(j).widget()
                        # Get value input
                        value_input = param_widget.layout().itemAt(1).widget()
                        try:
                            values.append(float(value_input.text()))
                        except (ValueError, AttributeError):
                            values.append(0.0)
        
        return values