"""
Module for handling spectra and model I/O operations.
Includes functions for loading, saving, and reading model files.
"""

import os
import numpy as np
import re
import platform
from PySide6.QtWidgets import QFileDialog, QMessageBox
from constants import numco, numro, number_of_baseline_parameters


def mod_len_def(mod, include_special=True):
    """
    Calculate the number of parameters for a given model type.
    
    Args:
        mod: Model name string
        include_special: If True, includes Distr/Corr/Expression parameters.
                        If False, excludes them (used for model parameter counting).
    
    Returns:
        int: Number of parameters
    """
    base_params = int(
        4 * (mod == 'Singlet') + 7 * (mod == 'Doublet') + 11 * (mod == 'Sextet') + 
        14 * (mod == 'Sextet(rough)') + 11 * (mod == 'Relax_2S') + 11 * (mod == 'Average_H') + 
        9 * (mod == 'Relax_MS') + 12 * (mod == 'ASM') + 11 * (mod == 'Hamilton_mc') + 
        9 * (mod == 'Hamilton_pc') + numco * (mod == 'Variables') + 14 * (mod == 'MDGD') +
        number_of_baseline_parameters * (mod == 'Nbaseline')  # Nbaseline has baseline parameters
    )
    
    if include_special:
        base_params += 5 * (mod == 'Distr') + 2 * (mod == 'Corr') + 1 * (mod == 'Expression')
    
    return base_params


def load_model(main_window):
    """
    Load a model from a .mdl or .txt file and apply it to the parameters table.

    Args:
        main_window: The main PhysicsApp window instance
    """
    # Open file dialog
    file_path, _ = QFileDialog.getOpenFileName(
        main_window,
        "Pick a model...",
        main_window.dir_path,
        "Model files (*.mdl);;All files (*.*)"
    )

    if not file_path:
        main_window.log.setPlainText("Selection was canceled")
        return

    try:
        # Read the model file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Parse the model data
        M_list = []
        for line in lines:
            M_list.append(line.strip().split('\t'))

        # Check if second line is colors (backward compatibility)
        loaded_colors = []
        has_colors = False
        if len(M_list) > 1:
            # Simple check: if all fields in second line are color names or empty
            color_names = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'pink', 'crimson', 'orange', 'purple', 'brown', 'gray', 'black', 'white', 'silver', 'lightgreen']
            second_line = M_list[1]
            if all((field.startswith('#') and len(field) == 7) or field in color_names or field == '' for field in second_line):
                has_colors = True
                loaded_colors = second_line
            else:
                # Old model without colors, keep current colors
                loaded_colors = main_window.model_colors[:len(M_list[0])]

        # Handle special case for baseline row
        counter = 0
        for line in M_list:
            if line[0] == 'False':
                # Insert empty fields for baseline
                line.insert(0, '')
                line.insert(0, '')
                line.insert(0, '')
                line.insert(0, '')
            counter += 1

        # Clear existing models by setting them to None (instead of deleting to preserve color order)
        for i in range(1, len(main_window.params_table.row_widgets)):  # Skip baseline
            main_window.params_table.select_model(i, 'None')

        # Load model names for each row (starting from row 1, row 0 is always baseline)
        num_rows_to_load = min(len(main_window.params_table.row_widgets) - 1, len(M_list[0]) - 1)
        for i in range(1, num_rows_to_load + 1):
            if i < len(M_list[0]):
                model_name = M_list[0][i]
                if model_name and model_name != 'None':
                    main_window.params_table.select_model(i, model_name)

        # Assign loaded colors to model_colors
        for i in range(min(len(loaded_colors), len(main_window.model_colors))):
            main_window.model_colors[i] = loaded_colors[i]

        # Update color button styles
        for r in range(1, len(main_window.params_table.row_widgets)):
            if r < len(main_window.model_colors):
                color = main_window.model_colors[r]
                row_widget = main_window.params_table.row_widgets[r]
                start_widget = row_widget.layout().itemAt(0).widget()
                color_btn = start_widget.layout().itemAt(0).widget()
                bg_color = color if color.startswith('#') else main_window.params_table.get_color_from_code(color)
                text_color = 'black' if color in ['red', 'yellow', 'cyan', 'lime', 'darkorange', 'white', 'silver', 'lightgreen', 'pink'] or color.startswith('#') else 'white'
                color_btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")

        # Load parameter data for each row (starting from row 0)
        param_start_idx = 2 if has_colors else 1
        for k in range(len(M_list) - param_start_idx):  # Skip header and colors if present
            if k >= len(main_window.params_table.row_widgets):
                break

            row_data = M_list[k + param_start_idx]
            num_params = len(row_data) // 5  # Each param has 5 fields: value, lower, upper, name?, fix

            for i in range(num_params):
                if i >= numco:
                    break

                base_idx = i * 5
                if base_idx + 4 >= len(row_data):
                    break

                # Extract parameter data
                value = row_data[base_idx]
                lower = row_data[base_idx + 1]
                upper = row_data[base_idx + 2]
                # name = row_data[base_idx + 3]  # Not used in current implementation
                fix_str = row_data[base_idx + 4]

                # Set the parameter values in the table
                row_widget = main_window.params_table.row_widgets[k]
                if i + 1 < row_widget.layout().count():
                    param_widget = row_widget.layout().itemAt(i + 1).widget()

                    # Value input
                    value_input = param_widget.layout().itemAt(1).widget()
                    value_input.setText(value)

                    # Bounds
                    bounds_layout = param_widget.layout().itemAt(2).layout()
                    lower_input = bounds_layout.itemAt(0).widget()
                    upper_input = bounds_layout.itemAt(1).widget()
                    lower_input.setText(lower)
                    upper_input.setText(upper)

                    # Fix checkbox
                    top_layout = param_widget.layout().itemAt(0).layout()
                    fix_cb = top_layout.itemAt(1).widget()
                    fix_cb.setChecked(fix_str.lower() == 'true')

        # Special handling for baseline shifting (similar to original code)
        if len(main_window.params_table.row_widgets) > 0:
            baseline_row = main_window.params_table.row_widgets[0]
            # Check if we need to shift baseline parameters (when position 7 value is empty)
            if baseline_row.layout().count() > 8:
                param_widget_7 = baseline_row.layout().itemAt(8).widget()
                value_input_7 = param_widget_7.layout().itemAt(1).widget()
                if value_input_7.text() == '':
                    # Shift values from positions 4,5,6,7,8 to positions 3,4,5,6,7
                    for shift in range(7, 3, -1):
                        if shift + 1 < baseline_row.layout().count():
                            src_widget = baseline_row.layout().itemAt(shift + 1).widget()
                            dst_widget = baseline_row.layout().itemAt(shift).widget()

                            # Copy value
                            src_value = src_widget.layout().itemAt(1).widget().text()
                            dst_widget.layout().itemAt(1).widget().setText(src_value)

                            # Copy bounds
                            src_bounds = src_widget.layout().itemAt(2).layout()
                            dst_bounds = dst_widget.layout().itemAt(2).layout()
                            dst_bounds.itemAt(0).widget().setText(src_bounds.itemAt(0).widget().text())
                            dst_bounds.itemAt(1).widget().setText(src_bounds.itemAt(1).widget().text())

                            # Copy fix
                            src_fix = src_widget.layout().itemAt(0).layout().itemAt(1).widget().isChecked()
                            dst_widget.layout().itemAt(0).layout().itemAt(1).widget().setChecked(src_fix)

                    # Set specific values to 0
                    param_widget_4 = baseline_row.layout().itemAt(4).widget()
                    param_widget_8 = baseline_row.layout().itemAt(8).widget()
                    param_widget_4.layout().itemAt(1).widget().setText('0')
                    param_widget_8.layout().itemAt(1).widget().setText('0')

                    # Clear bounds for positions 3 and 7
                    param_widget_3 = baseline_row.layout().itemAt(3).widget()
                    param_widget_7 = baseline_row.layout().itemAt(7).widget()

                    bounds_3 = param_widget_3.layout().itemAt(2).layout()
                    bounds_7 = param_widget_7.layout().itemAt(2).layout()

                    bounds_3.itemAt(0).widget().setText('')
                    bounds_3.itemAt(1).widget().setText('')
                    bounds_7.itemAt(0).widget().setText('')
                    bounds_7.itemAt(1).widget().setText('')

                    # Set fixes for positions 3 and 7
                    param_widget_3.layout().itemAt(0).layout().itemAt(1).widget().setChecked(True)
                    param_widget_7.layout().itemAt(0).layout().itemAt(1).widget().setChecked(True)

        main_window.log.setPlainText("Model loaded successfully")
        main_window.log.setStyleSheet("color: green;")

    except Exception as e:
        main_window.log.setPlainText(f"Could not load model: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def _save_model_to_file(main_window, file_path):
    """
    Internal function to save the model data to a file.

    Args:
        main_window: The main PhysicsApp window instance
        file_path: The path to save the file to
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write model names (first row)
            model_names = []
            for row_widget in main_window.params_table.row_widgets:
                model_btn = row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
                model_names.append(model_btn.text())
            f.write('\t'.join(model_names) + '\n')

            # Write colors (second row)
            colors = main_window.model_colors[:len(model_names)]
            f.write('\t'.join(colors) + '\n')

            # Write parameter data for each row
            for row_idx, row_widget in enumerate(main_window.params_table.row_widgets):
                row_data = []
                for param_idx in range(1, row_widget.layout().count()):
                    param_widget = row_widget.layout().itemAt(param_idx).widget()
                    if param_widget:
                        # Value
                        value_input = param_widget.layout().itemAt(1).widget()
                        value = value_input.text()

                        # Bounds
                        bounds_layout = param_widget.layout().itemAt(2).layout()
                        lower_input = bounds_layout.itemAt(0).widget()
                        upper_input = bounds_layout.itemAt(1).widget()
                        lower = lower_input.text()
                        upper = upper_input.text()

                        # Name (not used in current implementation, empty)
                        name = ''

                        # Fix
                        top_layout = param_widget.layout().itemAt(0).layout()
                        fix_cb = top_layout.itemAt(1).widget()
                        fix = 'True' if fix_cb.isChecked() else 'False'

                        row_data.extend([value, lower, upper, name, fix])
                f.write('\t'.join(row_data) + '\n')

        main_window.log.setPlainText("Model saved successfully")
        main_window.log.setStyleSheet("color: green;")

    except Exception as e:
        main_window.log.setPlainText(f"Could not save model: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def save_model(main_window):
    """
    Save the current model to a .mdl file based on the save path field.

    Args:
        main_window: The main PhysicsApp window instance
    """
    save_path_text = main_window.save_path.text().strip()
    workfolder = main_window.workfolder or os.getcwd()

    # Determine the file path
    if not save_path_text:
        file_path = os.path.join(workfolder, "model.mdl")
    elif os.path.isdir(save_path_text):
        file_path = os.path.join(save_path_text, "model.mdl")
    elif os.path.isfile(save_path_text) or '.' in save_path_text:
        if '.' in save_path_text:
            base = save_path_text.rsplit('.', 1)[0]
            file_path = base + '.mdl'
        else:
            file_path = save_path_text + '.mdl'
    else:
        # Single name
        file_path = os.path.join(workfolder, save_path_text + '.mdl')

    # Check if file exists and ask for overwrite
    if os.path.exists(file_path):
        reply = QMessageBox.question(
            main_window,
            'File exists',
            f'File {file_path} already exists. Overwrite?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            main_window.log.setPlainText("Saving canceled")
            main_window.log.setStyleSheet("color: orange;")
            return

    _save_model_to_file(main_window, file_path)


def save_model_as(main_window):
    """
    Save the current model to a .mdl file using file dialog.

    Args:
        main_window: The main PhysicsApp window instance
    """
    workfolder = main_window.workfolder or os.getcwd()
    file_path, _ = QFileDialog.getSaveFileName(
        main_window,
        "Save model as...",
        workfolder,
        "Model files (*.mdl);;All files (*.*)"
    )

    if not file_path:
        main_window.log.setPlainText("Saving canceled")
        main_window.log.setStyleSheet("color: orange;")
        return

    # Ensure .mdl extension
    if not file_path.lower().endswith('.mdl'):
        file_path += '.mdl'

    _save_model_to_file(main_window, file_path)


def read_model(main_window):
    """
    Read model parameters from the parameters table.
    
    Args:
        main_window: The main PhysicsApp window instance
        
    Returns:
        tuple: (model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN)
            - model: list of model names
            - p: numpy array of parameters
            - con1, con2, con3: constraint arrays
            - Distri: distribution expressions
            - Cor: correlation expressions
            - Expr: expressions
            - NExpr: expression indices
            - DistriN: distribution indices
    """
    
    model = []
    p = np.array([], dtype=float)
    con1 = np.array([], dtype=float)
    con2 = np.array([], dtype=float)
    con3 = np.array([], dtype=float)
    Distri = []
    Cor = []
    Expr = []
    NExpr = np.array([], dtype=int)
    DistriN = np.array([], dtype=float)
    
    # Read baseline parameters from first row (row 0)
    baseline_row = main_window.params_table.row_widgets[0]
    for i in range(1, number_of_baseline_parameters + 1):
        # Access parameter widget: layout().itemAt(i) gives the i-th param widget
        param_widget = baseline_row.layout().itemAt(i).widget()
        # Get the value input (second widget in param_layout)
        value_input = param_widget.layout().itemAt(1).widget()
        param_text = value_input.text()
        
        if param_text.startswith('=[') and param_text.endswith(']'):
            # Constraint format: =[index,multiplier]
            b = param_text[2:-1].split(',')
            con1 = np.append(con1, len(p))
            con2 = np.append(con2, float(b[0]))
            con3 = np.append(con3, float(b[1]))
            p = np.append(p, 1)
        else:
            p = np.append(p, float(param_text) if param_text else 0.0)
    
    # Read model parameters from remaining rows
    for i in range(1, len(main_window.params_table.row_widgets)):
        row_widget = main_window.params_table.row_widgets[i]
        
        # Get model name from the start_widget (first item in layout)
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()  # Second widget is model button
        model_name = model_btn.text()
        
        if model_name != 'None' and model_name != 'baseline':
            model.append(model_name)
            
        LenM = mod_len_def(model_name, include_special=False) + 1
        
        # Read parameters for this model
        for j in range(1, min(LenM, numco + 1)):
            # Access parameter widget: layout().itemAt(j) gives the j-th param widget (j=1 to numco)
            if j < row_widget.layout().count():
                param_widget = row_widget.layout().itemAt(j).widget()
                # Get the value input (second widget in param_layout)
                value_input = param_widget.layout().itemAt(1).widget()
                param_text = value_input.text()
                
                if param_text.startswith('=[') and param_text.endswith(']'):
                    # Constraint
                    b = param_text[2:-1].split(',')
                    con1 = np.append(con1, len(p))
                    con2 = np.append(con2, float(b[0]))
                    con3 = np.append(con3, float(b[1]))
                    p = np.append(p, 1)
                else:
                    p = np.append(p, float(param_text) if param_text else 0.0)
        
        # Handle special model types
        if model_name == 'Expression':
            # Expression should have text in first parameter field
            param_widget = row_widget.layout().itemAt(1).widget()
            value_input = param_widget.layout().itemAt(1).widget()
            expr_text = value_input.text()
            Expr.append(expr_text)
            NExpr = np.append(NExpr, len(p))
            p = np.append(p, 0)
            
        elif model_name == 'Distr':
            # Distribution has 5 parameters plus expression
            for j in range(1, 5):
                if j < row_widget.layout().count():
                    param_widget = row_widget.layout().itemAt(j).widget()
                    value_input = param_widget.layout().itemAt(1).widget()
                    param_text = value_input.text()
                    
                    if param_text.startswith('=[') and param_text.endswith(']'):
                        b = param_text[2:-1].split(',')
                        con1 = np.append(con1, len(p))
                        con2 = np.append(con2, float(b[0]))
                        con3 = np.append(con3, float(b[1]))
                        p = np.append(p, 1)
                    else:
                        p = np.append(p, float(param_text) if param_text else 0.0)
            p = np.append(p, 0)
            # Get distribution expression from 5th field
            if 5 < row_widget.layout().count():
                param_widget = row_widget.layout().itemAt(5).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                distri_text = value_input.text()
            else:
                distri_text = ''
            Distri.append(distri_text)
            DistriN = np.append(DistriN, len(p) - 1)
            
        elif model_name == 'Corr':
            # Correlation has 1 parameter (par) plus expression (Dependency function)
            # Only process first parameter as float
            if 1 < row_widget.layout().count():
                param_widget = row_widget.layout().itemAt(1).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                param_text = value_input.text()
                
                if param_text.startswith('=[') and param_text.endswith(']'):
                    b = param_text[2:-1].split(',')
                    con1 = np.append(con1, len(p))
                    con2 = np.append(con2, float(b[0]))
                    con3 = np.append(con3, float(b[1]))
                    p = np.append(p, 1)
                else:
                    p = np.append(p, float(param_text) if param_text else 0.0)
            # Add placeholder for expression parameter
            p = np.append(p, 0)
            # Get correlation expression from 2nd field (Dependency function)
            if 2 < row_widget.layout().count():
                param_widget = row_widget.layout().itemAt(2).widget()
                value_input = param_widget.layout().itemAt(1).widget()
                cor_text = value_input.text()
            else:
                cor_text = ''
            Cor.append(cor_text)
    
    return (model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN)