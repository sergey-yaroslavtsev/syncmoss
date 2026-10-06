"""
Results table widget for SYNCMoss application.
Displays fitting results in a tabbed interface with interactive controls.

Structure:
-----------
The table holds the current result and nothing else: a group of 3 rows per
component,
    Row i*3 + 0: Parameter names
    Row i*3 + 1: Parameter values (from fitting)
    Row i*3 + 2: Parameter errors (calculated from correlation matrix)
created by fill_table for exactly as many components as the result has, and no
row at all before the first fit or after Show model (clear_table). The row limit
of the parameters table (constants.numro) does not apply here: a result may have
more components than the parameters table has rows.

Column 0: Model names with colors
Columns 1-numco: Parameter data

Usage Example:
--------------
# After fitting is complete:
parameters = np.array([...])  # Fitted parameter values
model_list = ['baseline', 'Doublet', 'Sextet']
model_colors = ['gray', 'blue', 'red']
parameter_names = [
    ['BG', 'A', 'B', 'C', ...],  # baseline parameter names
    ['Intens', 'CS', 'QS', ...],  # Doublet parameter names
    ['Intens', 'CS', 'Bhf', ...]  # Sextet parameter names
]
correlation_matrix = np.array([...])  # From fitting routine

# Fill the table
results_table.fill_table(parameters, model_list, model_colors, parameter_names, correlation_matrix)
"""
import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QAbstractItemView,
    QApplication
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont, QColor, QImage, QPainter, QKeySequence, QShortcut, QTextDocument
from syncmoss.constants import numco, contrast_text_color
from syncmoss.support_math import calculate_intensity_percentage_error
from syncmoss.spectrum_parameters import substitute
# NOTE: the eval() calls in this module run against an explicit math_namespace
# built from np.* — no bare ``from numpy import ...`` block is needed here.

# How close a Doublet must be to the Be/KB preset to count as that impurity
# (see _calculate_intensities). Rounding to four decimals moves a value by at
# most 5e-5, so this takes a preset rounded or not.
_PRESET_ATOL = 1e-4

class ClickableResultButton(QPushButton):
    """Clickable button for result table rows that triggers replotting."""
    
    def __init__(self, text, row_index, parent=None):
        super().__init__(text, parent)
        self.row_index = row_index
        self.setFont(QFont('Arial', 10))
        self.setMinimumHeight(20)
        self.setMaximumHeight(20)
        
        # Connect click to replot handler
        self.clicked.connect(self.on_button_clicked)
    
    def on_button_clicked(self):
        """Handle button click to replot results."""
        # Get reference to main window through parent hierarchy
        main_window = self.window()
        if hasattr(main_window, 'replot_result'):
            main_window.replot_result(self.row_index)


class ResultsTable(QWidget):
    """
    Results table widget with tabbed interface.
    
    Features:
    - Tab 1: Interactive results table (buttons + labels) like original
    - Tab 2: Correlation matrix display
    - 3 rows (parameter name, value, error) per component of the current
      result, none without a result
    - First column contains model names/colors
    - Rows organized as: name (1,4,7...), value (2,5,8...), error (3,6,9...)
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.num_cols = numco + 1  # +1 for model column
        
        # Storage for correlation matrix and fitting results.
        # These are (re)assigned in fill_table(); initialised here so methods
        # that may run before the first fit can rely on direct attribute access
        # instead of hasattr() guards.
        self.correlation_matrix = None
        self.fit_parameters = None
        self.parameter_names = []
        self.covariance_matrix = None
        self.errors = None
        self.whole_fit = None
        self.model_list = []
        self.model_colors = []

        # Storage for current results (for saving)
        self.current_parameters = None
        self.current_errors = None
        self.current_model_list = []
        self.current_model_colors = []
        self.current_parameter_names = []
        self.current_chi2 = 0.0
        self.current_spectrum_parameters = None   # N, N1, ... the fit used
        self.current_exclusion_regions = ()       # the regions the fit left out
        self.current_links = {}
        # The fitted model (model_io.fitted_model_rows) that "Save result"
        # writes as <base>_result_model.mdl; set by the fit handlers.
        self.current_model_rows = None
        # A simultaneous one-model fit's result (one_model.OneModelResult) when
        # the table shows one -- set after fill_table, dropped by clear_table.
        # "Take result" and "Save result" work on it instead of the table.
        self.one_model = None

        # Initialize layout
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabbed widget
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont('Arial', 14))
        
        # Tab 1: Interactive results (like original)
        self.interactive_tab = self.create_interactive_tab()
        self.tabs.addTab(self.interactive_tab, "Results")
        
        # Tab 2: Correlation matrix
        self.correlation_tab = self.create_correlation_tab()
        self.tabs.addTab(self.correlation_tab, "Correlation Matrix")
        
        layout.addWidget(self.tabs)

    def create_interactive_tab(self):
        """
        Create the interactive results tab (Tab 1).

        Structure:
        - First column: model names/colors (buttons or labels)
        - Remaining columns: parameter data
        - Row pattern (for each component):
          - Row i*3: parameter names
          - Row i*3+1: parameter values
          - Row i*3+2: parameter errors

        The table starts without rows; fill_table creates them (_set_row_count).
        """
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setSpacing(0)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget for interactive display
        self.interactive_table = QTableWidget(0, self.num_cols)
        self.interactive_table.setFont(QFont('Arial', 14))
        self.interactive_table.horizontalHeader().setVisible(False)
        self.interactive_table.verticalHeader().setVisible(False)
        self.interactive_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.interactive_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._enable_copy(self.interactive_table)

        # Set column widths
        for col in range(self.num_cols):
            self.interactive_table.setColumnWidth(col, 64)

        # The widgets of each row: a model button and one label per parameter
        # column (self.buttons[row], self.labels[row][col - 1])
        self.buttons = []
        self.labels = []

        tab_layout.addWidget(self.interactive_table)
        return tab_widget

    def _set_row_count(self, rows):
        """Give the interactive table exactly *rows* rows.

        Every new row gets its widgets: the clickable model button in column 0
        and a label in each parameter column. Rows that go take their widgets
        with them (the table hides and deletes them).
        """
        table = self.interactive_table
        table.setRowCount(rows)
        del self.buttons[rows:]
        del self.labels[rows:]
        for row in range(len(self.buttons), rows):
            table.setRowHeight(row, 22)

            # First column: model name/color indicator (clickable button)
            btn = ClickableResultButton('', row, self)
            self.buttons.append(btn)
            table.setCellWidget(row, 0, btn)

            # Remaining columns: labels for displaying results
            row_widgets = []
            for col in range(1, self.num_cols):
                label = QLabel('')
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setFont(QFont('Arial', 10))
                label.setTextFormat(Qt.TextFormat.RichText)  # Support markup
                row_widgets.append(label)
                table.setCellWidget(row, col, label)
            self.labels.append(row_widgets)
    
    def create_correlation_tab(self):
        """
        Create the correlation matrix tab (Tab 2).
        
        Displays the correlation matrix from fitting results.
        """
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setSpacing(0)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create table for correlation matrix
        self.correlation_table = QTableWidget()
        self.correlation_table.setFont(QFont('Arial', 10))
        self.correlation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.correlation_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._enable_copy(self.correlation_table)

        # Will be populated when correlation matrix is available
        tab_layout.addWidget(self.correlation_table)
        return tab_widget

    def _enable_copy(self, table):
        """Make Ctrl+C copy the selected cells of *table* as tab-separated text.

        The cells hold widgets (buttons / rich-text labels) instead of plain
        items, so Qt's own copy path finds nothing: the selection highlights but
        the clipboard stays empty. WidgetWithChildren context so the shortcut
        also fires while a cell widget has the focus.
        """
        shortcut = QShortcut(QKeySequence.StandardKey.Copy, table)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(lambda t=table: self.copy_selection(t))

    @staticmethod
    def _cell_text(table, row, col):
        """Plain text of a cell, whether it holds a widget, an item or nothing."""
        widget = table.cellWidget(row, col)
        if widget is not None:
            text = widget.text() if hasattr(widget, 'text') else ''
        else:
            item = table.item(row, col)
            text = item.text() if item is not None else ''
        if '<' in text:  # labels are RichText: strip markup for the clipboard
            doc = QTextDocument()
            doc.setHtml(text)
            text = doc.toPlainText()
        return text.strip()

    def copy_selection(self, table):
        """Copy *table*'s selected cells to the clipboard as tab-separated text."""
        cells = {(idx.row(), idx.column()) for idx in table.selectedIndexes()}
        cols = sorted({c for _, c in cells})
        rows = []
        for row in sorted({r for r, _ in cells}):
            texts = [self._cell_text(table, row, c) if (row, c) in cells else ''
                     for c in cols]
            if any(texts):  # skip empty rows (a baseline's % rows, ...)
                rows.append(texts)
        text = '\n'.join('\t'.join(r) for r in rows)
        QApplication.clipboard().setText(text)
        return text

    def fill_table(self, parameters, model_list, model_colors, parameter_names, covariance_matrix, errors=None, fix=None, expression_texts=None,
                   spectrum_parameters=None, whole_fit=None):
        """
        Main function to fill the results table with fitting results.
        
        Args:
            parameters: Array of fitted parameter values
            model_list: List of model names for each component
            model_colors: List of colors for each model
            parameter_names: List of parameter names for each component
            covariance_matrix: Covariance matrix from fitting (numpy array)
            errors: Array of parameter errors from fitting (optional)
            fix: Array of indices of fixed parameters (optional)
            expression_texts: Dict {component_index: expression_text} for Distr/Corr/Expression models (optional)
            spectrum_parameters: SpectrumParameters the fit used for N, N1, ... (optional).
                The texts are kept as typed ("Take result" puts them back into
                the table); the values are substituted only to evaluate them.
            whole_fit: (parameters, errors, covariance_matrix) of the whole fit when
                *parameters* are one spectrum's part of it (optional). The
                Expressions are typed with the whole model's p[i] and may use other
                spectra's parameters, so they are evaluated with it.

        Workflow:
            1. Clear existing table, then create 3 rows per component
            2. Fill parameter values (rows 1, 4, 7, ...)
            3. Fill model names/colors (column 0)
            4. Fill parameter names (rows 0, 3, 6, ...)
            5. Calculate and fill errors (rows 2, 5, 8, ...)
            6. Display correlation matrix in tab 2
        """
        # Store data for saving functionality
        self.current_parameters = parameters
        self.current_errors = errors
        self.current_model_list = model_list
        self.current_model_colors = model_colors
        self.current_parameter_names = parameter_names
        self.current_chi2 = 0.0  # Will be set separately by main window
        self.current_spectrum_parameters = spectrum_parameters

        # Store expression texts
        self.expression_texts = expression_texts if expression_texts is not None else {}
        
        # Store data for internal use
        self.fit_parameters = parameters
        self.model_list = model_list
        self.model_colors = model_colors
        self.parameter_names = parameter_names
        self.covariance_matrix = covariance_matrix
        self.errors = errors
        self.whole_fit = whole_fit
        self.fix = fix if fix is not None else np.array([], dtype=int)
        
        # Orchestrate filling, on fresh rows: exactly three per component
        self.clear_table()
        self._set_row_count(3 * max(len(model_list), len(parameter_names)))
        self.fill_values(parameters)
        self.fill_model_column(model_list, model_colors)
        self.fill_parameter_names(parameter_names)
        self.fill_errors(errors)
        self._apply_expression_spanning()
        self.display_correlation_matrix(covariance_matrix)

    def clear_table(self):
        """Remove every row of the results table, and the correlation matrix.

        Show model leaves the table like this -- a model is not a fit result --
        and fill_table starts from it, so nothing of an earlier result (a text,
        a colour, a spanned Expression cell) can survive into the next one.
        """
        self.interactive_table.clearSpans()
        self._set_row_count(0)
        self.one_model = None

        # Clear correlation matrix
        self.correlation_table.clear()
        self.correlation_table.setRowCount(0)
        self.correlation_table.setColumnCount(0)
    
    def fill_values(self, parameters):
        """
        Fill parameter values into value rows (1, 4, 7, 10, ...).
        For Distr/Corr/Expression models, the last parameter column shows the expression text.
        
        Args:
            parameters: Array of parameter values from fitting
        """
        if parameters is None:
            return
        
        param_index = 0
        
        # Process each component
        for component in range(len(self.model_list)):
            value_row = component * 3 + 1  # Rows 1, 4, 7, ...

            # Get number of parameters for this component from parameter_names
            if component < len(self.parameter_names):
                num_params = len(self.parameter_names[component])
            else:
                num_params = 0
            
            model_name = self.model_list[component] if component < len(self.model_list) else ''
            
            # Fill values for this component
            for col in range(min(num_params, numco)):
                if param_index < len(parameters):
                    # Check if this is an expression / weight-vector column
                    if model_name == 'Recon' and col == num_params - 1:
                        # The reconstruction weight vector is not printed here (it is
                        # viewed in the Distribution plot); the value still lives in
                        # self.expression_texts so "take result as model" round-trips it.
                        self.labels[value_row][col].setText('(see Distribution)')
                    elif model_name in ['Distr', 'Corr'] and col == num_params - 1:
                        # Show expression text for Distr/Corr
                        if component in self.expression_texts:
                            self.labels[value_row][col].setText(self.expression_texts[component])
                        else:
                            self.labels[value_row][col].setText('')
                    elif model_name == 'Expression' and col == num_params - 1:
                        # For Expression: value row shows the calculated result
                        calc_value, _ = self._evaluate_expression_with_error(component)
                        if calc_value is not None:
                            self.labels[value_row][col].setText(f"{calc_value:.3f}")
                        else:
                            self.labels[value_row][col].setText('eval error')
                    else:
                        value = parameters[param_index]
                        # Format the value appropriately (3 decimal places for display)
                        formatted_value = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                        self.labels[value_row][col].setText(formatted_value)
                    param_index += 1
    
    def fill_model_column(self, model_list, model_colors):
        """
        Fill model names and colors in the first column, and intensity percentages.
        
        For each model (except baseline and Nbaseline):
        - Row 0: model name
        - Row 1: intensity percentage "x%"
        - Row 2: intensity error "±y%"
        
        Args:
            model_list: List of model names
            model_colors: List of colors for each model
        """
        if not model_list:
            return
        
        # Calculate intensity percentages
        intensities, intensity_errors = self._calculate_intensities()
        
        for i, (model_name, color) in enumerate(zip(model_list, model_colors)):
            # Each model occupies 3 rows (name, value, error)
            base_row = i * 3

            # Baseline should always be light gray
            if model_name == 'baseline':
                color = 'lightgray'
            
            # Determine text color based on background
            text_color = contrast_text_color(color)
            
            # Row 0: Model name
            self.buttons[base_row].setText(model_name)
            self.buttons[base_row].setStyleSheet(f"background-color: {color}; color: {text_color};")
            
            # Row 1: Intensity percentage (or model name for baseline/Nbaseline)
            if model_name in ['baseline', 'Nbaseline', 'Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables']:
                self.buttons[base_row + 1].setText('')  # Empty for baseline/Nbaseline
            elif self.buttons[base_row + 1].text() == 'Impurity':
                pass
            else:
                if i < len(intensities):
                    self.buttons[base_row + 1].setText(f"{intensities[i]:.1f}%")
                else:
                    self.buttons[base_row + 1].setText('')
            self.buttons[base_row + 1].setStyleSheet(f"background-color: {color}; color: {text_color};")
            
            # Row 2: Intensity error (or model name for baseline/Nbaseline)
            if model_name in ['baseline', 'Nbaseline', 'Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables']:
                self.buttons[base_row + 2].setText('')  # Empty for baseline/Nbaseline
            elif self.buttons[base_row + 2].text() == 'no %':
                pass
            else:
                if i < len(intensity_errors):
                    self.buttons[base_row + 2].setText(f"±{intensity_errors[i]:.1f}%")
                else:
                    self.buttons[base_row + 2].setText('')
            self.buttons[base_row + 2].setStyleSheet(f"background-color: {color}; color: {text_color};")
    
    def _calculate_intensities(self):
        """
        Calculate intensity percentages and their errors using covariance matrix.
        Handles Nbaseline models by calculating intensities separately for each spectrum.
        
        Returns:
            tuple: (intensities, intensity_errors) - arrays of percentages, one per model
        """
        if self.fit_parameters is None or not self.parameter_names:
            return np.array([]), np.array([])
        
        # Group models by spectrum (separated by Nbaseline)
        spectrum_groups = []  # List of lists of (model_index, param_index)
        current_group = []
        param_index = 0
        
        for i, (model_name, param_names) in enumerate(zip(self.model_list, self.parameter_names)):
            # Skip baseline (it's not part of any spectrum group)
            if model_name == 'baseline':
                param_index += len(param_names)
                continue
            
            # Nbaseline marks the start of a new spectrum
            if model_name == 'Nbaseline':
                if current_group:  # Save previous group
                    spectrum_groups.append(current_group)
                current_group = []  # Start new group
                param_index += len(param_names)
                continue
            
            # Skip models without T parameter
            if model_name in ['Layer', 'Distr', 'Corr', 'Recon', 'Expression', 'Variables']:
                param_index += len(param_names)
                continue
            
            if model_name == 'Doublet':
                try:
                    # Be.txt / KB.txt hold the polarized Doublet preset (9 values,
                    # the CURRENT beamline state, edited in Supp); a Be/KB_nano row
                    # is fixed to those, so a match flags it. To 1e-4, not exactly:
                    # a model saved before the presets were rounded to four
                    # decimals still carries the longer digits.
                    be_param = np.genfromtxt(os.path.join(self.main_window.params_dir, 'Be.txt'), delimiter='\t')
                    kb_param = np.genfromtxt(os.path.join(self.main_window.params_dir, 'KB.txt'), delimiter='\t')
                    fitted = self.fit_parameters[param_index:param_index+len(param_names)]
                    if (len(fitted) == len(be_param) and np.allclose(fitted, be_param, rtol=0, atol=_PRESET_ATOL)) \
                        or (len(fitted) == len(kb_param) and np.allclose(fitted, kb_param, rtol=0, atol=_PRESET_ATOL)):
                            self.buttons[i*3 + 1].setText('Impurity')
                            self.buttons[i*3 + 2].setText('no %')
                            param_index += len(param_names)
                            continue
                except Exception:
                    pass

            # Check if first parameter is 'T' (intensity/transmission)
            if param_names and param_names[0] in ['T', 'Intens', 'Int']:
                current_group.append((i, param_index))
            
            param_index += len(param_names)
            
        # Don't forget the last group
        if current_group:
            spectrum_groups.append(current_group)
        
        # Calculate intensities for each group separately
        full_intensities = np.zeros(len(self.model_list))
        full_errors = np.zeros(len(self.model_list))
        
        for group in spectrum_groups:
            if not group:
                continue
            
            # Extract T indices for this group
            t_indices = [param_idx for _, param_idx in group]
            
            # Calculate intensities with proper error propagation
            if self.covariance_matrix is not None:
                # Identify fixed parameters (where er is nan)
                errors = self.errors if self.errors is not None else np.zeros_like(self.fit_parameters)
                fixed_params = [i for i in range(len(errors)) if np.isnan(errors[i])]
                
                intensities, intensity_errors = calculate_intensity_percentage_error(
                    self.fit_parameters,
                    errors,
                    self.covariance_matrix,
                    t_indices,
                    fixed_params=fixed_params
                )
                
                # Map back to model indices
                for idx, (model_idx, _) in enumerate(group):
                    if idx < len(intensities):
                        full_intensities[model_idx] = intensities[idx]
                        full_errors[model_idx] = intensity_errors[idx]
        
        return full_intensities, full_errors
    
    def fill_parameter_names(self, parameter_names):
        """
        Fill parameter names into name rows (0, 3, 6, 9, ...).
        
        Args:
            parameter_names: List of parameter name lists for each component
        """
        if not parameter_names:
            return
        
        for component, names in enumerate(parameter_names):
            name_row = component * 3  # Rows 0, 3, 6, ...

            model_name = self.model_list[component] if component < len(self.model_list) else ''
            
            # Fill names starting from column 1
            for col, name in enumerate(names[:numco]):
                # For Expression: name row shows the expression text
                if model_name == 'Expression' and col == len(names) - 1:
                    if component in self.expression_texts:
                        self.labels[name_row][col].setText(self.expression_texts[component])
                    else:
                        self.labels[name_row][col].setText(str(name))
                else:
                    self.labels[name_row][col].setText(str(name))
    
    def fill_errors(self, errors):
        """
        Fill parameter errors into error rows (2, 5, 8, 11, ...).
        For Expression model: shows 'calculated_value ± propagated_error'.
        For Distr/Corr expression column: shows empty.
        
        Args:
            errors: Array of parameter errors from fitting (returned by minimi_hi as 'er')
        """
        if errors is None:
            return
        
        error_index = 0
        
        # Process each component
        for component in range(len(self.model_list)):
            error_row = component * 3 + 2  # Rows 2, 5, 8, ...

            # Get number of parameters for this component
            if component < len(self.parameter_names):
                num_params = len(self.parameter_names[component])
            else:
                num_params = 0
            
            model_name = self.model_list[component] if component < len(self.model_list) else ''
            
            # Fill errors for this component
            for col in range(min(num_params, numco)):
                if error_index < len(errors):
                    if model_name in ['Distr', 'Corr', 'Recon', 'Expression'] and col == num_params - 1:
                        # Expression / weight-vector column
                        if model_name == 'Expression':
                            # Show propagated error for the expression
                            _, calc_error = self._evaluate_expression_with_error(component)
                            if calc_error is not None:
                                self.labels[error_row][col].setText(f"±{calc_error:.3f}")
                            else:
                                self.labels[error_row][col].setText('')
                        else:
                            # Distr/Corr expression column - no error
                            self.labels[error_row][col].setText('')
                    else:
                        error_value = errors[error_index]
                        # Format with ± prefix (3 decimal places for display)
                        formatted_error = f"±{error_value:.3f}" if isinstance(error_value, (int, float)) else str(error_value)
                        self.labels[error_row][col].setText(formatted_error)
                    error_index += 1
    
    def _evaluate_expression_with_error(self, component):
        """
        Evaluate an Expression model's expression using fitted parameters and compute
        propagated error using the covariance matrix.
        
        Args:
            component: Component index in model_list
            
        Returns:
            tuple: (calculated_value, propagated_error) or (None, None) on failure
        """
        if component not in self.expression_texts:
            return None, None
        
        expr_text = substitute(self.expression_texts[component], self.current_spectrum_parameters)
        params, errors, cov = self.whole_fit or (self.fit_parameters, self.errors,
                                                 self.covariance_matrix)

        if params is None:
            return None, None
        
        # Evaluate expression
        math_namespace = {
            'p': params, 'np': np,
            'exp': np.exp, 'log': np.log, 'log10': np.log10, 'sqrt': np.sqrt,
            'abs': np.abs, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'pi': np.pi, 'e': np.e, 'power': np.power,
            'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
            'sum': np.sum, 'mean': np.mean, 'std': np.std,
        }
        try:
            value = float(eval(expr_text, {"__builtins__": {}}, math_namespace))
        except Exception:
            return None, None
        
        if cov is None or errors is None:
            return value, 0.0
        
        # Build mapping from full parameter index to covariance matrix index
        # (only free parameters are in the covariance matrix)
        full_to_cov = {}
        cov_idx = 0
        for i in range(len(errors)):
            if not np.isnan(errors[i]):
                full_to_cov[i] = cov_idx
                cov_idx += 1
        
        # Compute partial derivatives via numerical differentiation
        n_free = cov.shape[0]
        partials = np.zeros(n_free)
        delta = 1e-6
        
        for full_idx, c_idx in full_to_cov.items():
            p_plus = params.copy()
            p_minus = params.copy()
            h = max(abs(params[full_idx]) * delta, delta)
            p_plus[full_idx] += h
            p_minus[full_idx] -= h
            try:
                namespace_plus = {**math_namespace, 'p': p_plus}
                namespace_minus = {**math_namespace, 'p': p_minus}
                f_plus = float(eval(expr_text, {"__builtins__": {}}, namespace_plus))
                f_minus = float(eval(expr_text, {"__builtins__": {}}, namespace_minus))
                partials[c_idx] = (f_plus - f_minus) / (2 * h)
            except Exception:
                partials[c_idx] = 0.0
        
        # Error propagation: var(f) = J^T * Cov * J
        variance = partials @ cov @ partials
        error = np.sqrt(max(variance, 0))
        
        return value, error

    def _apply_expression_spanning(self):
        """
        Apply column spanning for Distr/Corr/Expression models in the results table.
        The last meaningful parameter cell expands to span all remaining columns.
        Uses QTableWidgetItem text instead of QLabel so text fills the full spanned area.
        """
        if not self.model_list:
            return

        # Map model names to their number of parameters. 'Recon' is intentionally
        # absent: its trailing weight column is not spanned into a long line — the
        # reconstruction is shown in the Distribution plot, and its results cell is
        # just a short "(see Distribution)" marker.
        special_models = {
            'Expression': 1,
            'Corr': 2,
            'Distr': 5,
        }

        for i, model_name in enumerate(self.model_list):
            if model_name in special_models:
                num_params = special_models[model_name]
                # Table column of the last meaningful param (col 0 is buttons, params start at col 1)
                last_col = num_params  # e.g. Expression: col 1, Corr: col 2, Distr: col 5
                span_cols = self.num_cols - last_col  # how many columns to span

                if span_cols > 1:
                    name_row = i * 3
                    value_row = i * 3 + 1
                    error_row = i * 3 + 2

                    # Apply span for all 3 rows (name, value, error)
                    for table_row in [name_row, value_row, error_row]:
                        self.interactive_table.setSpan(table_row, last_col, 1, span_cols)

                    # Replace QLabel cell widgets with QTableWidgetItems for spanned cells
                    # so the text naturally fills the full spanned width
                    label_idx = num_params - 1  # index into self.labels[row]
                    for table_row in [name_row, value_row, error_row]:
                        if label_idx < len(self.labels[table_row]):
                            text = self.labels[table_row][label_idx].text()
                            # Remove the QLabel widget and use a QTableWidgetItem instead.
                            # removeCellWidget only unregisters the label: it stays a
                            # visible child of the table until Qt deletes it later, so
                            # the picture of the table taken right after a fit drew it
                            # under the item (the texts on top of each other). Hide it now.
                            old_label = self.interactive_table.cellWidget(table_row, last_col)
                            self.interactive_table.removeCellWidget(table_row, last_col)
                            if old_label is not None:
                                old_label.hide()
                                old_label.deleteLater()
                            item = QTableWidgetItem(text)
                            item.setFont(QFont('Arial', 10))
                            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                            self.interactive_table.setItem(table_row, last_col, item)

    def display_correlation_matrix(self, covariance_matrix):
        """
        Display correlation matrix in the correlation tab.
        
        Args:
            covariance_matrix: Numpy array containing covariance matrix from fitting
        
        The correlation matrix is calculated from the covariance matrix:
            correlation[i,j] = covariance[i,j] / sqrt(covariance[i,i] * covariance[j,j])
        
        Structure:
            - Row 0: Model names (shown at first parameter of each model)
            - Row 1: Parameter names
            - Col 0: Model names (shown at first parameter of each model)
            - Col 1: Parameter names
            - Data: (2+, 2+)
        
        Only includes parameters where er[i] is not NaN (i.e., fitted parameters, not fixed).
        """
        if covariance_matrix is None:
            return
        
        # Convert to numpy array if needed
        if not isinstance(covariance_matrix, np.ndarray):
            covariance_matrix = np.array(covariance_matrix)
        
        # Calculate correlation matrix from covariance matrix
        n_params = covariance_matrix.shape[0]
        correlation_matrix = np.zeros_like(covariance_matrix)
        
        for i in range(n_params):
            for j in range(n_params):
                denom = np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])
                if denom > 0:
                    correlation_matrix[i, j] = covariance_matrix[i, j] / denom
                else:
                    correlation_matrix[i, j] = 0.0
        
        # Store for later use
        self.correlation_matrix = correlation_matrix
        
        # Build parameter labels and model labels, filtering based on errors array
        # Only include parameters where er[i] is not NaN
        param_labels = []
        model_labels = []
        param_index = 0
        
        if len(self.parameter_names) > 0 and self.errors is not None:
            # Baseline is always the first component
            baseline_names = self.parameter_names[0]
            for i, param_name in enumerate(baseline_names):
                if param_index < len(self.errors) and not np.isnan(self.errors[param_index]):
                    param_labels.append(param_name)
                    model_labels.append('baseline')
                param_index += 1
            
            # Add model parameters
            for comp_idx in range(1, len(self.parameter_names)):
                model_name = self.model_list[comp_idx] if comp_idx < len(self.model_list) else ''
                param_names = self.parameter_names[comp_idx]
                for param_name in param_names:
                    if param_index < len(self.errors) and not np.isnan(self.errors[param_index]):
                        param_labels.append(param_name)
                        model_labels.append(model_name)
                    param_index += 1
        else:
            # Fallback to p0, p1, etc. if parameter_names or errors not available
            param_labels = [f'p{i}' for i in range(n_params)]
            model_labels = [''] * n_params
        
        # Verify we have the right number of labels
        if len(param_labels) != n_params:
            print(f"Warning: Expected {n_params} parameter labels but got {len(param_labels)}")
            print(f"  parameter_names available: {len(self.parameter_names) > 0}")
            print(f"  errors available: {self.errors is not None}")
            if self.errors is not None:
                print(f"  errors length: {len(self.errors)}, non-NaN count: {np.sum(~np.isnan(self.errors))}")
            # Fallback
            param_labels = [f'p{i}' for i in range(n_params)]
            model_labels = [''] * n_params
        
        # Set up table dimensions: +2 rows and +2 columns for model names and parameter names
        self.correlation_table.setRowCount(n_params + 2)
        self.correlation_table.setColumnCount(n_params + 2)
        
        # Set top-left corner cells (empty)
        for i in range(2):
            for j in range(2):
                corner_item = QTableWidgetItem('')
                corner_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                corner_item.setBackground(QColor(200, 200, 200))
                corner_item.setForeground(QColor(255, 255, 255))
                self.correlation_table.setItem(i, j, corner_item)
        
        # Set first row (model names as column headers)
        current_model = None
        for j in range(n_params):
            model_name = model_labels[j] if j < len(model_labels) else ''
            # Only show model name at the first parameter of each model
            if model_name != current_model:
                header_item = QTableWidgetItem(model_name)
                current_model = model_name
            else:
                header_item = QTableWidgetItem('')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
            header_item.setForeground(QColor(255, 255, 255))
            self.correlation_table.setItem(0, j + 2, header_item)
        
        # Set second row (parameter names as column headers)
        for j in range(n_params):
            header_item = QTableWidgetItem(param_labels[j] if j < len(param_labels) else f'p{j}')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
            header_item.setForeground(QColor(255, 255, 255))
            self.correlation_table.setItem(1, j + 2, header_item)
        
        # Set first column (model names as row headers)
        current_model = None
        for i in range(n_params):
            model_name = model_labels[i] if i < len(model_labels) else ''
            # Only show model name at the first parameter of each model
            if model_name != current_model:
                header_item = QTableWidgetItem(model_name)
                current_model = model_name
            else:
                header_item = QTableWidgetItem('')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
            header_item.setForeground(QColor(255, 255, 255))
            self.correlation_table.setItem(i + 2, 0, header_item)
        
        # Set second column (parameter names as row headers)
        for i in range(n_params):
            header_item = QTableWidgetItem(param_labels[i] if i < len(param_labels) else f'p{i}')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            header_item.setBackground(QColor(70, 70, 70))
            header_item.setForeground(QColor(255, 255, 255))
            self.correlation_table.setItem(i + 2, 1, header_item)
        
        # Fill matrix values (starting at row 2, col 2)
        for i in range(n_params):
            for j in range(n_params):
                value = correlation_matrix[i, j]
                item = QTableWidgetItem(f"{value:.4f}")
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Color code based on correlation strength
                if abs(value) > 0.9 and i != j:
                    item.setBackground(QColor(255, 100, 100))  # High correlation - red
                elif abs(value) > 0.7 and i != j:
                    item.setBackground(QColor(255, 200, 100))  # Medium correlation - orange
                elif i == j:
                    item.setBackground(QColor(100, 100, 255))  # Diagonal - blue
                
                self.correlation_table.setItem(i + 2, j + 2, item)
        
        # Adjust column widths
        self.correlation_table.resizeColumnsToContents()
        self.correlation_table.resizeRowsToContents()
    
    def _rows_with_content(self):
        """Table rows with something in them: a button, a label or a table item.

        The items matter: the spanned last column of Distr/Corr/Expression is a
        QTableWidgetItem, not a label (see _apply_expression_spanning), and an
        Expression's value and ± rows have nothing else in them.
        """
        rows = []
        table = self.interactive_table
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                widget = table.cellWidget(row, col)
                item = table.item(row, col)
                if (widget is not None and hasattr(widget, 'text') and widget.text().strip()) \
                        or (item is not None and item.text().strip()):
                    rows.append(row)
                    break
        return rows

    def render_table_to_image(self):
        """
        Render the interactive results table to a QImage (full content, no scrollbars, no empty rows).

        Returns:
            QImage: Rendered table image
        """
        # Find non-empty rows (rows where at least one cell has content)
        non_empty_rows = self._rows_with_content()

        if not non_empty_rows:
            # Return a minimal black image if no content
            image = QImage(100, 100, QImage.Format.Format_ARGB32)
            image.fill(QColor(0, 0, 0))
            return image
        
        # Calculate full table width
        total_width = 0
        for col in range(self.interactive_table.columnCount()):
            total_width += self.interactive_table.columnWidth(col)
        
        # Calculate height only for non-empty rows
        total_height = 0
        for row in non_empty_rows:
            total_height += self.interactive_table.rowHeight(row)
        
        # Add margins for borders
        total_width = int(1.03 * total_width)
        # total_width += 2
        total_height = int(1.03 * total_height)
        # total_height += 17
        
        # Temporarily hide empty rows
        hidden_rows = []
        for row in range(self.interactive_table.rowCount()):
            if row not in non_empty_rows:
                self.interactive_table.setRowHidden(row, True)
                hidden_rows.append(row)
        
        # Create image with calculated size
        image = QImage(total_width, total_height, QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0))  # Black background
        
        # Render table to image using QPainter
        painter = QPainter(image)

        # Temporarily adjust table size to show all content
        old_min_size = self.interactive_table.minimumSize()
        old_max_size = self.interactive_table.maximumSize()
        old_size = self.interactive_table.size()

        try:
            self.interactive_table.setFixedSize(total_width, total_height)

            # PySide6 expects targetOffset when using QPainter overload.
            self.interactive_table.render(painter, QPoint(0, 0))
        finally:
            # Always restore original widget state
            self.interactive_table.setMinimumSize(old_min_size)
            self.interactive_table.setMaximumSize(old_max_size)
            self.interactive_table.resize(old_size)

            # Restore hidden rows
            for row in hidden_rows:
                self.interactive_table.setRowHidden(row, False)

            # Ensure painting is finished even on exceptions
            if painter.isActive():
                painter.end()
        
        return image