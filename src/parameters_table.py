import os
import re
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QMenu
)
from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QRegularExpressionValidator, QAction
from constants import numro, numco

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
        self.row_params = [8] + [0] * (numro - 1)  # baseline has 8 params
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
        color_cycle = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']
        for r in range(1, len(self.row_widgets)):
            color = color_cycle[(r - 1) % len(color_cycle)]
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
                be_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'Be.txt'), delimiter='\t')
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
                kb_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'KB.txt'), delimiter='\t')
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