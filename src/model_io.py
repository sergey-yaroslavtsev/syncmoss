"""
Module for handling spectra and model I/O operations.
Includes functions for loading, saving, and reading model files.
"""

import os
import numpy as np
import re
import platform
from PyQt6.QtWidgets import QFileDialog
from constants import numco


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
        "Model files (*.mdl *.txt);;All files (*.*)"
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
        from PyQt6.QtWidgets import QMessageBox
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


def load_spectrum(main_window, file_paths, calibration_path="Calibration.dat", points_match=True):
    """
    Load spectrum file(s).

    Args:
        main_window: The main PhysicsApp window instance
        file_paths: String path or list of string paths to spectrum files
        calibration_path: Name of calibration file (default: "Calibration.dat")
        points_match: Whether points match (default: True)

    Returns:
        A_list, B_list: Lists of numpy arrays for energy/intensity data
    """
    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Set calibration path
    if main_window is not None:
        calibration_path = os.path.join(main_window.dir_path, calibration_path)
    else:
        # Use directory of first file
        first_file_dir = os.path.dirname(file_paths[0])
        calibration_path = os.path.join(first_file_dir, calibration_path)

    acceptable_formats = ['.dat', '.txt', '.exp', '.ws5', '.w98', '.moe', '.m1', '.mca', '.cmca', 'tango', '.mcs']

    A_list = []
    B_list = []

    for file in file_paths:
        A_list_single = []
        B_list_single = []
        A = np.array([float(0)])
        B = np.array([float(0)])

        # Handle directory: if ends with / or \, list .dat or .mca files
        if str(file).endswith('\\') or str(file).endswith('/'):
            try:
                path_tmp_local = []
                for files in os.listdir(str(file)):
                    if files.endswith(".dat") or files.endswith(".mca"):
                        path_tmp_local.append(os.path.join(str(file), files))
                if path_tmp_local:
                    file = path_tmp_local[0]  # Use first file
                else:
                    if main_window is not None:
                        main_window.log.setPlainText("No .dat or .mca files found in directory")
                        main_window.log.setStyleSheet("color: orange;")
                    continue
            except:
                if main_window is not None:
                    main_window.log.setPlainText("Directory does not exist")
                    main_window.log.setStyleSheet("color: yellow;")
                continue

        in_format_list = file.lower().endswith(tuple(acceptable_formats))

        # Text-based files
        if file.endswith('.dat') or file.endswith('.txt') or file.endswith('.exp') or not in_format_list:
            try:
                with open(file, 'r') as catalog:
                    lines = (line.rstrip() for line in catalog)
                    lines = (line for line in lines if line)  # skipping white lines
                    for line in lines:
                        column = line.split()
                        if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                            x = float(column[0])
                            y = float(column[1])
                            A_list_single.append(x)
                            B_list_single.append(y)
                B = np.array(B_list_single)
                # NFS filtering removed (switch hardcoded to SMS)
            except:
                if main_window is not None:
                    if not in_format_list:
                        main_window.log.setPlainText("File could not be opened as two-column. Please check the file.")
                    else:
                        main_window.log.setPlainText("Unexpected problem while opening file. Please check the file.")
                    main_window.log.setStyleSheet("color: red;")
                continue

        # MCA files
        elif file.endswith('.mca') or file.endswith('.cmca') or file == 'tango':
            # Calibration
            with open(calibration_path, 'r') as catalog:
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                for line in lines:
                    column = line.split()
                    if not line.startswith('#'):  # skipping column labels
                        x = float(column[0])
                        A_list_single.append(x)

            cal_type = open(cal_path, 'r')
            try:
                cal_info = (cal_type.readline()).split()[1:]
                cal_method = cal_info[0]
                n1 = int(cal_info[1])
                n2 = int(cal_info[2])
            except:
                cal_method = 'sin'
                n1 = 0
                n2 = int(len(A_list_single)) * 2 - 1
            cal_type.close()

            if file == 'tango':
                # Tango not implemented
                if main_window is not None:
                    main_window.log.setPlainText("Tango not supported")
                    main_window.log.setStyleSheet("color: orange;")
                continue
            else:
                LS = len(open(file, 'r').readlines())
                with open(file, 'r') as fi:
                    id_data = []
                    n = 0
                    k = 0
                    for i in range(0, LS):
                        for ln in fi:
                            if ln.startswith("@A"):
                                k += 1
                            if k > n and ln.startswith("@A"):
                                id_data.append(re.findall(r'[\d.]+', ln[2:]))
                            if k > n and ln.startswith("#"):
                                break
                            if k > n and not ln.startswith("@A"):
                                id_data[n].extend(re.findall(r'[\d.]+', ln[0:]))
                    n += 1
                id_data = np.array(id_data, dtype=float)

            # SMS mode (switch.active = True hardcoded)
            if file.endswith('.cmca'):
                if (id_data[-1][-1] == 0 and id_data[-1][0] != 0) or (id_data[-1][0] == 0 and id_data[-1][-1] != 0):
                    id_half = (id_data[-1][-1] + id_data[-1][0]) / 2
                    id_data[-1][-1] = id_half
                    id_data[-1][0] = id_half
            try:
                if cal_method == 'sin':
                    spc_1h = id_data[-1][:int(n1 / 2)] + id_data[-1][n1 - int(n1 / 2):n1][::-1]
                    spc_2h = id_data[-1][n2+1:n2 + int((len(id_data[-1]) - 1 - n2) / 2)+1][::-1] + id_data[-1][len(id_data[-1]) - int((len(id_data[-1]) - 1 - n2) / 2):len(id_data[-1])]
                    spc_3h = id_data[-1][n1:n1 + int((n2 - n1 + 1) / 2)] + id_data[-1][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
                    B = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
                elif cal_method == 'lin':
                    n_sh = 2 * (n1 + (int(len(id_data[-1])) - 1 - n2))
                    B = np.array([float(0)] * int((len(id_data[-1]) - n_sh) / 2))
                    for i in range(0, int((len(id_data[-1]) - n_sh) / 2)):
                        B[i] = id_data[-1][n1 + i] + id_data[-1][n2 - i]
            except:
                points_match = False
                B = np.array([float(1)] * (len(A_list_single) + 1))
                main_window.log.setPlainText("Something wrong with points in .mca or calibration.dat")
                main_window.log.setStyleSheet("color: red;")

        # Other formats (Wissel files)
        elif file.endswith('.ws5') or file.endswith('.w98') or file.endswith('.moe') or file.endswith('.m1') or file.lower().endswith('.mcs'):
            # Calibration
            with open(calibration_path, 'r') as catalog:
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                for line in lines:
                    column = line.split()
                    if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                        x = float(column[0])
                        A_list_single.append(x)

            cal_type = open(calibration_path, 'r')
            try:
                cal_info = (cal_type.readline()).split()[1:]
                cal_method = cal_info[0]
                n1 = int(cal_info[1])
                n2 = int(cal_info[2])
            except:
                cal_method = 'sin'
                n1 = 0
                n2 = int(len(A_list_single)) * 2 - 1
            cal_type.close()

            if file.lower().endswith('.mcs'):
                f = open(file, mode='rb')
                id_data = []
                entete1 = f.read(256)
                array = np.fromfile(f, dtype=np.uint32)
                id_data.append(array)
                f.close()
            else:
                with open(file, 'r') as catalog:
                    id_data = []
                    id_data.append([])
                    lines = (line.rstrip() for line in catalog)
                    lines = (line for line in lines if line)  # skipping white lines
                    k = 0
                    for line in lines:
                        if file.endswith('.m1'):
                            while '  ' in line:
                                line = line.replace('  ', ' ')
                            column = line.split(' ')
                            x = float(column[4])
                            if k > 0:
                                id_data[0].append(x)
                            k += 1
                        else:
                            column = line.split()
                            if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                                x = float(column[0])
                                if file.endswith('.moe') or not ('.' in str(column[0])):
                                    id_data[0].append(x)
                id_data = np.array(id_data, dtype=float)

            # SMS mode
            try:
                if cal_method == 'sin':
                    spc_1h = id_data[-1][:int(n1 / 2)] + id_data[-1][n1 - int(n1 / 2):n1][::-1]
                    spc_2h = id_data[-1][n2+1:n2 + int((len(id_data[-1]) - 1 - n2) / 2)+1][::-1] + id_data[-1][len(id_data[-1]) - int((len(id_data[-1]) - 1 - n2) / 2):len(id_data[-1])]
                    spc_3h = id_data[-1][n1:n1 + int((n2 - n1 + 1) / 2)] + id_data[-1][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
                    B = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
                elif cal_method == 'lin':
                    n_sh = 2 * (n1 + (int(len(id_data[-1])) - 1 - n2))
                    B = np.array([float(0)] * int((len(id_data[-1]) - n_sh) / 2))
                    for i in range(0, int((len(id_data[-1]) - n_sh) / 2)):
                        B[i] = id_data[-1][n1 + i] + id_data[-1][n2 - i]
            except:
                points_match = False
                B = np.array([float(1)] * (len(A_list_single) + 1))
                main_window.log.setPlainText("Something wrong with points in file or calibration.dat")
                main_window.log.setStyleSheet("color: red;")

        A = np.array(A_list_single)
        A_list.append(A)
        B_list.append(B)

    return A_list, B_list


def calculate_backgrounds(paths):
    """
    Calculate background levels for a list of spectrum files.

    Args:
        paths: List of file paths to spectrum files

    Returns:
        List of background values (N0) for each spectrum
    """
    backgrounds = []
    for path in paths:
        try:
            # Load the spectrum
            A_list, B_list = load_spectrum(None, [path])  # Pass None as main_window since we don't need it
            if A_list and B_list:
                A = A_list[0]
                B = B_list[0]

                # Calculate background as in original code (transmission mode)
                edge_avg = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10
                max_minus_sqrt = max(B) - 3 * np.sqrt(max(B))

                if edge_avg < max_minus_sqrt:
                    N0 = max_minus_sqrt
                else:
                    N0 = edge_avg

                # Ensure N0 is at least 1
                N0 = max(N0, 1)

                backgrounds.append(N0)
            else:
                backgrounds.append(0)  # Or some default
        except Exception as e:
            print(f"Error calculating background for {path}: {e}")
            backgrounds.append(0)

    return backgrounds