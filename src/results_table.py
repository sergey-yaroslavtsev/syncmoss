"""
Results table widget for SYNCMoss application.
Displays fitting results in a tabbed interface with interactive controls.

Structure:
-----------
The table has numro*3 rows, organized in groups of 3 for each component:
    Row i*3 + 0: Parameter names
    Row i*3 + 1: Parameter values (from fitting)
    Row i*3 + 2: Parameter errors (calculated from correlation matrix)

Column 0: Model names with colors
Columns 1-15: Parameter data

Usage Example:
--------------
# After fitting is complete:
parameters = np.array([...])  # Fitted parameter values
model_list = ['baseline', 'Doublet', 'Sextet']
model_colors = ['gray', 'blue', 'red']
parameter_names = [
    ['BG', 'A', 'B', 'C', ...],  # baseline parameter names
    ['Intens', 'IS', 'QS', ...],  # Doublet parameter names
    ['Intens', 'IS', 'Bhf', ...]  # Sextet parameter names
]
correlation_matrix = np.array([...])  # From fitting routine

# Fill the table
results_table.fill_table(parameters, model_list, model_colors, parameter_names, correlation_matrix)
"""
import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from constants import numro, numco


class ClickableResultButton(QPushButton):
    """Clickable button for result table rows that triggers replotting."""
    
    def __init__(self, text, row_index, parent=None):
        super().__init__(text, parent)
        self.row_index = row_index
        self.setFont(QFont('Arial', 10))
        self.setStyleSheet("background-color: black; color: white;")
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
    - numro*3 rows (parameter name, value, error for each component)
    - First column contains model names/colors
    - Rows organized as: name (1,4,7...), value (2,5,8...), error (3,6,9...)
    """
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.num_rows = numro * 3
        self.num_cols = numco + 1  # +1 for model column
        
        # Storage for correlation matrix and fitting results
        self.correlation_matrix = None
        self.fit_parameters = None
        self.model_list = []
        self.model_colors = []
        
        # Storage for current results (for saving)
        self.current_parameters = None
        self.current_errors = None
        self.current_model_list = []
        self.current_model_colors = []
        self.current_parameter_names = []
        self.current_chi2 = 0.0
        
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
        
        # Storage for results data
        self.results_data = [['' for _ in range(numco)] for _ in range(self.num_rows)]
        self.row_labels = ['' for _ in range(self.num_rows)]
    
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
        """
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setSpacing(0)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create table widget for interactive display
        self.interactive_table = QTableWidget(self.num_rows, self.num_cols)
        self.interactive_table.setFont(QFont('Arial', 14))
        self.interactive_table.setStyleSheet("background-color: black; color: white;")
        self.interactive_table.horizontalHeader().setVisible(False)
        self.interactive_table.verticalHeader().setVisible(False)
        
        # Set column widths
        for col in range(self.num_cols):
            self.interactive_table.setColumnWidth(col, 64)
        
        # Set row heights
        for row in range(self.num_rows):
            self.interactive_table.setRowHeight(row, 22)
        
        # Populate table with buttons and labels
        self.buttons = []
        self.labels = []
        
        for row in range(self.num_rows):
            row_widgets = []
            
            # First column: model name/color indicator (clickable button)
            btn = ClickableResultButton('', row, self)
            self.buttons.append(btn)
            self.interactive_table.setCellWidget(row, 0, btn)
            
            # Remaining columns: labels for displaying results
            for col in range(1, self.num_cols):
                label = QLabel('')
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setFont(QFont('Arial', 10))
                label.setStyleSheet("color: white;")
                label.setTextFormat(Qt.TextFormat.RichText)  # Support markup
                row_widgets.append(label)
                self.interactive_table.setCellWidget(row, col, label)
            
            self.labels.append(row_widgets)
        
        tab_layout.addWidget(self.interactive_table)
        return tab_widget
    
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
        self.correlation_table.setStyleSheet("background-color: black; color: white;")
        
        # Will be populated when correlation matrix is available
        tab_layout.addWidget(self.correlation_table)
        return tab_widget
    
    def fill_table(self, parameters, model_list, model_colors, parameter_names, covariance_matrix, errors=None, fix=None):
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
        
        Workflow:
            1. Clear existing table
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
        
        # Store data for internal use
        self.fit_parameters = parameters
        self.model_list = model_list
        self.model_colors = model_colors
        self.parameter_names = parameter_names
        self.covariance_matrix = covariance_matrix
        self.errors = errors
        self.fix = fix if fix is not None else np.array([], dtype=int)
        
        # Orchestrate filling
        self.clear_table()
        self.fill_values(parameters)
        self.fill_model_column(model_list, model_colors)
        self.fill_parameter_names(parameter_names)
        self.fill_errors(errors)
        self.display_correlation_matrix(covariance_matrix)
    
    def clear_table(self):
        """Clear all data from the results table."""
        for row in range(self.num_rows):
            # Clear button text
            self.buttons[row].setText('')
            self.buttons[row].setStyleSheet("background-color: black; color: white;")
            
            # Clear labels
            for col in range(len(self.labels[row])):
                self.labels[row][col].setText('')
        
        # Clear correlation matrix
        self.correlation_table.clear()
        self.correlation_table.setRowCount(0)
        self.correlation_table.setColumnCount(0)
    
    def fill_values(self, parameters):
        """
        Fill parameter values into value rows (1, 4, 7, 10, ...).
        
        Args:
            parameters: Array of parameter values from fitting
        
        Note: This should follow the same structure as fill_parameter_names,
        respecting component boundaries and parameter counts.
        """
        if parameters is None:
            return
        
        param_index = 0
        
        # Process each component
        for component in range(len(self.model_list)):
            value_row = component * 3 + 1  # Rows 1, 4, 7, ...
            
            if value_row >= self.num_rows:
                break
            
            # Get number of parameters for this component from parameter_names
            if component < len(self.parameter_names):
                num_params = len(self.parameter_names[component])
            else:
                num_params = 0
            
            # Fill values for this component
            for col in range(min(num_params, numco)):
                if param_index < len(parameters):
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
            
            if base_row >= self.num_rows:
                break
            
            # Baseline should always be light gray
            if model_name == 'baseline':
                color = 'lightgray'
            
            # Determine text color based on background
            text_color = 'black' if color in ['lightgray', 'red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] or (isinstance(color, str) and color.startswith('#')) else 'white'
            
            # Row 0: Model name
            self.buttons[base_row].setText(model_name)
            self.buttons[base_row].setStyleSheet(f"background-color: {color}; color: {text_color};")
            
            # Row 1: Intensity percentage (or model name for baseline/Nbaseline)
            if model_name in ['baseline', 'Nbaseline', 'Distr', 'Corr', 'Expression', 'Variables']:
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
            if model_name in ['baseline', 'Nbaseline', 'Distr', 'Corr', 'Expression', 'Variables']:
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
        from support_math import calculate_intensity_percentage_error
        
        if not hasattr(self, 'fit_parameters') or not hasattr(self, 'parameter_names'):
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
                    print(f"The {i} group is {current_group}")
                current_group = []  # Start new group
                param_index += len(param_names)
                continue
            
            # Skip models without T parameter
            if model_name in ['Distr', 'Corr', 'Expression', 'Variables']:
                param_index += len(param_names)
                continue
            
            if model_name == 'Doublet':
                try:
                    be_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'Be.txt'), delimiter='\t')
                    kb_param = np.genfromtxt(os.path.join(self.main_window.dir_path, 'KB.txt'), delimiter='\t')
                    if self.fit_parameters[param_index:param_index+len(param_names)].tolist() == be_param.tolist() \
                        or self.fit_parameters[param_index:param_index+len(param_names)].tolist() == kb_param.tolist():
                            print("Impurity detected, skipping intensity calculation for Doublet impurity.")
                            self.buttons[i*3 + 1].setText('Impurity')
                            self.buttons[i*3 + 2].setText('no %')
                            param_index += len(param_names)
                            print(f"Skipped successfully.")
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
            if hasattr(self, 'covariance_matrix') and self.covariance_matrix is not None:
                # Identify fixed parameters (where er is nan)
                errors = self.errors if hasattr(self, 'errors') and self.errors is not None else np.zeros_like(self.fit_parameters)
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
            
            if name_row >= self.num_rows:
                break
            
            # Fill names starting from column 1
            for col, name in enumerate(names[:numco]):
                self.labels[name_row][col].setText(str(name))
    
    def fill_errors(self, errors):
        """
        Fill parameter errors into error rows (2, 5, 8, 11, ...).
        
        Args:
            errors: Array of parameter errors from fitting (returned by minimi_hi as 'er')
        """
        if errors is None:
            return
        
        error_index = 0
        
        # Process each component
        for component in range(len(self.model_list)):
            error_row = component * 3 + 2  # Rows 2, 5, 8, ...
            
            if error_row >= self.num_rows:
                break
            
            # Get number of parameters for this component
            if component < len(self.parameter_names):
                num_params = len(self.parameter_names[component])
            else:
                num_params = 0
            
            # Fill errors for this component
            for col in range(min(num_params, numco)):
                if error_index < len(errors):
                    error_value = errors[error_index]
                    # Format with ± prefix (3 decimal places for display)
                    formatted_error = f"±{error_value:.3f}" if isinstance(error_value, (int, float)) else str(error_value)
                    self.labels[error_row][col].setText(formatted_error)
                    error_index += 1
    
    def calculate_fraction_errors(self, correlation_matrix):
        """
        Calculate errors for the fraction of each submodel.
        
        Args:
            correlation_matrix: Correlation matrix from fitting
        
        Returns:
            dict: Dictionary with fraction errors for each component
        
        TODO: Implement fraction error calculation using correlation matrix.
        This requires:
        - Identifying which parameters contribute to each fraction
        - Propagating errors through the fraction calculation
        - Using correlation matrix for covariance terms
        """
        # TODO: Implement
        fraction_errors = {}
        return fraction_errors
    
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
        
        if hasattr(self, 'parameter_names') and len(self.parameter_names) > 0 and hasattr(self, 'errors') and self.errors is not None:
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
            print(f"  parameter_names available: {hasattr(self, 'parameter_names')}")
            print(f"  errors available: {hasattr(self, 'errors')}")
            if hasattr(self, 'errors') and self.errors is not None:
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
                corner_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                corner_item.setBackground(QColor(200, 200, 200))
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
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
            self.correlation_table.setItem(0, j + 2, header_item)
        
        # Set second row (parameter names as column headers)
        for j in range(n_params):
            header_item = QTableWidgetItem(param_labels[j] if j < len(param_labels) else f'p{j}')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
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
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            header_item.setBackground(QColor(70, 70, 70))
            self.correlation_table.setItem(i + 2, 0, header_item)
        
        # Set second column (parameter names as row headers)
        for i in range(n_params):
            header_item = QTableWidgetItem(param_labels[i] if i < len(param_labels) else f'p{i}')
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            header_item.setBackground(QColor(70, 70, 70))
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
    
    def set_button_color(self, row, color):
        """
        Set the background color of a button.
        
        Args:
            row: Row index
            color: Color name or QColor object
        """
        if row < 0 or row >= self.num_rows:
            return
        
        # Determine background color
        if isinstance(color, str):
            bg_color = color
        elif isinstance(color, QColor):
            bg_color = f"rgb({color.red()}, {color.green()}, {color.blue()})"
        else:
            bg_color = "white"
        
        # Determine text color using same logic as parameters_table.py line 205
        text_color = 'black' if color in ['red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] or (isinstance(color, str) and color.startswith('#')) else 'white'
        
        self.buttons[row].setStyleSheet(f"background-color: {bg_color}; color: {text_color};")    
    def render_table_to_image(self):
        """
        Render the interactive results table to a QImage (full content, no scrollbars, no empty rows).
        
        Returns:
            QImage: Rendered table image
        """
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtCore import QSize, QRect
        
        # Find non-empty rows (rows where at least one cell has content)
        non_empty_rows = []
        for row in range(self.interactive_table.rowCount()):
            has_content = False
            # Check button column (column 0)
            button = self.interactive_table.cellWidget(row, 0)
            if button and button.text().strip():
                has_content = True
            # Check label columns
            if not has_content:
                for col in range(1, self.interactive_table.columnCount()):
                    widget = self.interactive_table.cellWidget(row, col)
                    if widget and hasattr(widget, 'text') and widget.text().strip():
                        has_content = True
                        break
            if has_content:
                non_empty_rows.append(row)
        
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
        
        self.interactive_table.setFixedSize(total_width, total_height)
        
        # Render the table widget to the painter
        self.interactive_table.render(painter)
        
        # Restore original size constraints
        self.interactive_table.setMinimumSize(old_min_size)
        self.interactive_table.setMaximumSize(old_max_size)
        self.interactive_table.resize(old_size)
        
        # Restore hidden rows
        for row in hidden_rows:
            self.interactive_table.setRowHidden(row, False)
        
        painter.end()
        
        return image