"""
Module for handling spectra and model I/O operations.
Includes functions for loading, saving, and reading model files.

NOTE on widget navigation: this module (like fitting_io and instrumental_io)
reaches into the ParametersTable rows positionally. The layout contract is:
``row.layout().itemAt(0)`` = start widget (color btn @0, model btn @1), and
``row.layout().itemAt(col+1)`` = parameter widget for column ``col``, whose own
layout holds the name/fix row @0 (name label @0, fix checkbox @1), the value
QLineEdit @1 and the lower/upper bounds layout @2. Any change to the row layout
in parameters_table.py must keep these indices stable (or update every user of
:func:`_set_row_param` / :func:`read_model` / :func:`read_bounds_and_fix`).
"""

import os
import numpy as np
import re
from PySide6.QtWidgets import QFileDialog, QMessageBox
from syncmoss.constants import numco, number_of_baseline_parameters, mdl_color_names, contrast_text_color
from syncmoss.Library_io import LIBRARY_METADATA_FIELDS, LIBRARY_METADATA_DEFAULTS, compute_versioned_title_if_needed
from syncmoss.legacy import upgrade_mdl_row, normalize_legacy_model_name


def mod_len_def(mod, include_special=True):
    """
    Calculate the number of parameters for a given model type.

    Heavily used across the package (fitting_io, instrumental_io, parameters_table,
    spectrum_plotter, syncmoss_main) to walk the flat parameter array model by model.

    Args:
        mod: Model name string
        include_special: If True, includes Distr/Corr/Expression parameters.
                        If False, excludes them (used for model parameter counting).
    
    Returns:
        int: Number of parameters
    """
    # Every component type is now the polarized model: the former scalar
    # asymmetry A was replaced by the orientation angles (theta_k, phi_h) plus
    # the uniaxial texture parameter A. 'Hamiltonian' keeps its crystal angles
    # (plus the beam rotation alpha_k) and carries THREE mosaic order parameters
    # (A, Am, Ah) instead; it supersedes the deprecated 'Hamilton_mc' (12) and
    # 'Hamilton_pc' (9), whose counts stay listed only so an old model file still
    # walks. See syncmoss.legacy for how pre-merge model files / presets (and the
    # two old Hamiltonian names) are upgraded to these counts.
    base_params = int(
        4 * (mod == 'Singlet') + 9 * (mod == 'Doublet') + 14 * (mod == 'Sextet') +
        14 * (mod == 'Sextet(rough)') + 14 * (mod == 'Relax_2S') + 11 * (mod == 'Average_H') +
        11 * (mod == 'Relax_MS') + 15 * (mod == 'ASM') + 27 * (mod == 'SCDW') +
        15 * (mod == 'Hamiltonian') + 12 * (mod == 'Hamilton_mc') +
        9 * (mod == 'Hamilton_pc') + numco * (mod == 'Variables') + 17 * (mod == 'MDGD') +
        number_of_baseline_parameters * (mod == 'Nbaseline')  # Nbaseline has baseline parameters
        # 'Layer' has 0 parameters (handled by the default for unknown names).
    )
    
    if include_special:
        # 'Recon' keeps a CONSTANT footprint of 7 flat slots regardless of Num:
        # par, L, R, Num, D_dif, D_dif2 and a single weight-vector placeholder.
        # The Num free reconstruction weights ride in a parallel list (like a
        # Distr's PDF string in Distri), so parameter counting / =[links] / p[i]
        # expressions never shift with Num. See read_model's 'Recon' branch.
        base_params += (5 * (mod == 'Distr') + 2 * (mod == 'Corr') + 1 * (mod == 'Expression')
                        + 7 * (mod == 'Recon'))

    return base_params


def _set_row_param(row_widget, col, value, lower, upper, fix_str):
    """Write one parameter (value, bounds, fix state) into a table-row column.

    ``col`` is the 0-based parameter column; the widget layout contract is
    described in the module docstring. Silently ignores columns beyond the
    row's widget count (rows are built with a fixed number of columns).
    """
    if col + 1 >= row_widget.layout().count():
        return
    param_widget = row_widget.layout().itemAt(col + 1).widget()
    param_widget.layout().itemAt(1).widget().setText(value)

    bounds_layout = param_widget.layout().itemAt(2).layout()
    bounds_layout.itemAt(0).widget().setText(lower)
    bounds_layout.itemAt(1).widget().setText(upper)

    fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
    fix_cb.setChecked(str(fix_str).lower() == 'true')


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
        main_window.workfolder,
        "Model files (*.mdl);;All files (*.*)"
    )

    if not file_path:
        main_window.set_status("Selection was canceled", "orange")
        return

    _load_model_from_path_impl(main_window, file_path)


def load_model_from_path(main_window, file_path, insert_row=None):
    """
    Load a model from a known file path and apply it to the parameters table.

    Args:
        main_window: The main PhysicsApp window instance
        file_path: Absolute path to a model file
    """
    if not file_path:
        main_window.set_status("Selection was canceled", "orange")
        return
    _load_model_from_path_impl(main_window, file_path, insert_row=insert_row)


def _last_parameter_index_before_row(params_table, row):
    """Return global parameter index of the last parameter before row."""
    row = max(0, min(row, len(params_table.row_params)))
    for rr in range(row - 1, -1, -1):
        if params_table.row_params[rr] > 0:
            return int(sum(params_table.row_params[:rr + 1]) - 1)
    return -1


def _remap_reference_text(text, z_value):
    """Remap references for appended library models.

    Rules:
    - '=[X,Y]' -> '=[Z+X-number_of_baseline_parameters+1,Y]'
    - 'p[X]'   -> 'p[Z+X-number_of_baseline_parameters+1]'
    """
    if not isinstance(text, str) or not text:
        return text

    def repl_ref(m):
        x = int(m.group(1))
        y = m.group(2)
        new_x = z_value + x - number_of_baseline_parameters + 1
        return f"=[{new_x},{y}]"

    def repl_p(m):
        x = int(m.group(1))
        new_x = z_value + x - number_of_baseline_parameters + 1
        return f"p[{new_x}]"

    out = re.sub(r"=\[(\d+)\s*,\s*([^\]]+)\]", repl_ref, text)
    out = re.sub(r"p\[(\d+)\]", repl_p, out)
    return out


def _load_model_from_path_impl(main_window, file_path, insert_row=None):
    try:
        # Read the model file, skipping comments and empty lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.rstrip('\n') for line in file if line.strip() and not line.lstrip().startswith('#')]

        # Parse the model data
        M_list = [line.split('\t') for line in lines]
        if not M_list:
            raise ValueError("Model file is empty")

        # Check if second line is colors (backward compatibility)
        loaded_colors = []
        has_colors = False
        if len(M_list) > 1:
            # Simple check: if all fields in second line are color names or empty
            # (mdl_color_names is shared with Library_io so both accept the same files)
            second_line = M_list[1]
            if all((field.startswith('#') and len(field) == 7) or field in mdl_color_names or field == '' for field in second_line):
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

        if insert_row is not None:
            insert_row = int(insert_row)
            insert_row = max(1, min(insert_row, len(main_window.params_table.row_widgets) - 1))

            # Build source model list without baseline (first entry)
            source_models = M_list[0][1:] if len(M_list[0]) > 1 else []
            param_start_idx = 2 if has_colors else 1

            # Reference remap anchor from existing destination table
            z_value = _last_parameter_index_before_row(main_window.params_table, insert_row)

            # Keep only rows that are actual models to be appended
            active_src_indices = [
                idx for idx, mod in enumerate(source_models)
                if mod.strip() and mod.strip() not in ['None', 'baseline']
            ]

            # Insert from end so repeated inserts at same row preserve order
            for src_idx in reversed(active_src_indices):
                model_name = source_models[src_idx].strip()
                # Insert blank row then select model
                main_window.params_table.select_model(insert_row, 'Insert')
                main_window.params_table.select_model(insert_row, model_name)

            # Fill inserted rows in final order using ONE fixed Z anchor.
            # This avoids iterative drift from repeated row insert/update cycles.
            for j, src_idx in enumerate(active_src_indices):
                dst_row = insert_row + j

                # Load row parameter data corresponding to this source model row
                row_data_idx = param_start_idx + src_idx + 1  # +1 skips baseline parameter row
                if row_data_idx >= len(M_list):
                    continue

                row_data = M_list[row_data_idx]
                # Upgrade a pre-merge (scalar) submodel row to the polarized layout.
                row_data = upgrade_mdl_row(normalize_legacy_model_name(source_models[src_idx]), row_data)
                num_params = len(row_data) // 5
                dst_row_widget = main_window.params_table.row_widgets[dst_row]

                for i in range(min(num_params, numco)):
                    base_idx = i * 5
                    if base_idx + 4 >= len(row_data):
                        break

                    _set_row_param(
                        dst_row_widget, i,
                        _remap_reference_text(row_data[base_idx], z_value),
                        _remap_reference_text(row_data[base_idx + 1], z_value),
                        _remap_reference_text(row_data[base_idx + 2], z_value),
                        row_data[base_idx + 4],
                    )

            main_window.params_table.update_distr_corr_highlights()
            main_window.set_status("Library submodel appended successfully", "green")
            return

        # Clear existing models by setting them to None (instead of deleting to preserve color order)
        for i in range(1, len(main_window.params_table.row_widgets)):  # Skip baseline
            main_window.params_table.select_model(i, 'None')

        # Load model names for each row (starting from row 1, row 0 is always baseline)
        num_rows_to_load = min(len(main_window.params_table.row_widgets) - 1, len(M_list[0]) - 1)
        for i in range(1, num_rows_to_load + 1):
            if i < len(M_list[0]):
                # Map any pre-merge name (e.g. 'Doublet_(thick)') to its current
                # name so the row can be built with the polarized model.
                model_name = normalize_legacy_model_name(M_list[0][i])
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
                color_btn.setStyleSheet(f"background-color: {bg_color}; color: {contrast_text_color(color)};")

        # Load parameter data for each row (starting from row 0)
        param_start_idx = 2 if has_colors else 1
        for k in range(len(M_list) - param_start_idx):  # Skip header and colors if present
            if k >= len(main_window.params_table.row_widgets):
                break

            row_data = M_list[k + param_start_idx]
            # Upgrade a pre-merge (scalar) component row to the current polarized
            # layout (insert theta_k/phi_h, remap the asymmetry). No-op for files
            # already in the new layout. Row k's model name is M_list[0][k].
            if k >= 1 and k < len(M_list[0]):
                row_data = upgrade_mdl_row(normalize_legacy_model_name(M_list[0][k]), row_data)
            num_params = len(row_data) // 5  # Each param has 5 fields: value, lower, upper, name?, fix

            for i in range(num_params):
                if i >= numco:
                    break

                base_idx = i * 5
                if base_idx + 4 >= len(row_data):
                    break

                # Each param has 5 fields: value, lower, upper, name (unused), fix
                _set_row_param(
                    main_window.params_table.row_widgets[k], i,
                    row_data[base_idx],
                    row_data[base_idx + 1],
                    row_data[base_idx + 2],
                    row_data[base_idx + 4],
                )

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

        # A loaded 'par' that happens to equal the auto-filled default emits no
        # textChanged, so re-draw the grey frames once the whole table is in.
        main_window.params_table.update_distr_corr_highlights()
        main_window.set_status("Model loaded successfully", "green")

    except Exception as e:
        main_window.set_status(f"Could not load model: {str(e)}", "red")


def _save_model_to_file(main_window, file_path, comment=None, metadata=None):
    """
    Internal function to save the model data to a file.

    Args:
        main_window: The main PhysicsApp window instance
        file_path: The path to save the file to
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(metadata, dict):
                metadata_to_write = dict(LIBRARY_METADATA_DEFAULTS)
                for k in LIBRARY_METADATA_FIELDS:
                    if k in metadata:
                        metadata_to_write[k] = str(metadata[k]).strip()
                for key in LIBRARY_METADATA_FIELDS:
                    f.write(f"#@{key} {metadata_to_write[key]}\n")

            if comment is not None and str(comment).strip() != '':
                for line in str(comment).splitlines():
                    f.write(f"#@Comment {line}\n")

            # Determine which rows to write: always keep baseline (row 0),
            # drop empty model rows (model name 'None') so the saved file stays clean.
            # Empty rows carry no parameters, so dropping them keeps the parameter
            # references (=[X,Y] / p[X]) intact for loading.
            kept_rows = []
            for row_idx, row_widget in enumerate(main_window.params_table.row_widgets):
                model_btn = row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
                if row_idx == 0 or model_btn.text() != 'None':
                    kept_rows.append((row_idx, row_widget, model_btn.text()))

            # Write model names (first row)
            model_names = [name for _, _, name in kept_rows]
            f.write('\t'.join(model_names) + '\n')

            # Write colors (second row)
            colors = [main_window.model_colors[row_idx]
                      if row_idx < len(main_window.model_colors) else ''
                      for row_idx, _, _ in kept_rows]
            f.write('\t'.join(colors) + '\n')

            # Write parameter data for each kept row
            for row_idx, row_widget, _ in kept_rows:
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

        main_window.set_status("Model saved successfully", "green")

    except Exception as e:
        main_window.set_status(f"Could not save model: {str(e)}", "red")


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
            main_window.set_status("Saving canceled", "orange")
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
        main_window.set_status("Saving canceled", "orange")
        return

    # Ensure .mdl extension
    if not file_path.lower().endswith('.mdl'):
        file_path += '.mdl'

    _save_model_to_file(main_window, file_path)


def save_model_to_library(main_window, title, comment=None, metadata=None, notify_rename=False):
    """Save current model to internal Library folder with optional comment."""
    model, *_ = read_model(main_window)
    if 'Nbaseline' in model:
        main_window.set_status("Model with 'Nbaseline' could not be saved to library", "orange")
        return False

    title = (title or '').strip()
    if not title:
        main_window.set_status("Library title is empty", "orange")
        return False

    library_dir = os.path.join(main_window.dir_path, 'Library')
    os.makedirs(library_dir, exist_ok=True)
    final_title, version_idx = compute_versioned_title_if_needed(library_dir, title)
    file_path = os.path.join(library_dir, f"{final_title}.mdl")

    if notify_rename and version_idx is not None:
        msg = QMessageBox(main_window)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle('Library title exists')
        msg.setText(f"Such title already exists. It will become ({version_idx}).")
        msg.setInformativeText("Are you sure you want to add the same sample to the library?")
        add_btn = msg.addButton('add', QMessageBox.ButtonRole.AcceptRole)
        skip_btn = msg.addButton('do not add', QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(skip_btn)
        msg.exec()
        if msg.clickedButton() is not add_btn:
            main_window.set_status("Saving to library canceled", "orange")
            return False

    _save_model_to_file(main_window, file_path, comment=comment, metadata=metadata)
    return True


def parse_recon_weights(text, num):
    """Parse a 'Recon' weight-vector text field into a length-``num`` float array.

    The reconstruction weights of a 'Recon' model are stored as a comma- (or
    whitespace-) separated list of ``Num`` non-negative numbers in the model
    row's trailing text column (mirroring how a Distr's PDF string is stored).
    An empty / unparsable / wrong-length field falls back to a uniform
    distribution of length ``num`` — the natural starting guess for a fit. The
    physics core normalises the vector, so the absolute scale here is irrelevant.
    """
    num = max(1, int(num))
    if text is not None and str(text).strip():
        try:
            parts = [float(v) for v in str(text).replace(';', ',').replace(' ', ',').split(',') if v.strip() != '']
            if len(parts) == num:
                arr = np.array(parts, dtype=float)
                if np.all(np.isfinite(arr)):
                    return arr
        except Exception:
            pass
    return np.full(num, 1.0 / num, dtype=float)


def read_model(main_window):
    """
    Read model parameters from the parameters table.

    Central GUI->fit bridge: used by fitting_io, instrumental_io, spectrum_io,
    Library_window and syncmoss_main to turn the on-screen table into the flat
    parameter array, constraints and expression lists the solver expects.

    Args:
        main_window: The main PhysicsApp window instance

    Returns:
        tuple: (model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN,
                Recon, ReconN)
            - model: list of model names
            - p: numpy array of parameters
            - con1, con2, con3: constraint arrays
            - Distri: distribution expressions
            - Cor: correlation expressions
            - Expr: expressions
            - NExpr: expression indices
            - DistriN: distribution indices
            - Recon: list of reconstruction weight arrays (one per 'Recon' model,
                     in table order) — the free fit values of the distribution
                     shape, held alongside p exactly like Distri holds PDF strings
            - ReconN: flat-array indices of the single weight-vector placeholder
                      slot of each 'Recon' (force-fixed by the fit, like DistriN)
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
    Recon = []
    ReconN = np.array([], dtype=float)

    def _field_text(row_widget, item_idx):
        """Text of the value QLineEdit in the row's item_idx-th param widget."""
        param_widget = row_widget.layout().itemAt(item_idx).widget()
        return param_widget.layout().itemAt(1).widget().text()

    def _append_param(param_text):
        """Append one table field to p.

        A ``=[source,factor]`` text registers a linear constraint (con1 gets
        this slot's index, con2/con3 the source index and factor; the slot
        itself receives placeholder 1). Anything else is parsed as a float
        (empty field -> 0.0).
        """
        nonlocal p, con1, con2, con3
        if param_text.startswith('=[') and param_text.endswith(']'):
            constraint_parts = param_text[2:-1].split(',')
            con1 = np.append(con1, len(p))
            con2 = np.append(con2, float(constraint_parts[0]))
            con3 = np.append(con3, float(constraint_parts[1]))
            p = np.append(p, 1)
        else:
            p = np.append(p, float(param_text) if param_text else 0.0)

    # Read baseline parameters from first row (row 0)
    baseline_row = main_window.params_table.row_widgets[0]
    for i in range(1, number_of_baseline_parameters + 1):
        _append_param(_field_text(baseline_row, i))

    # Read model parameters from remaining rows
    for i in range(1, len(main_window.params_table.row_widgets)):
        row_widget = main_window.params_table.row_widgets[i]

        # Get model name from the start_widget (first item in layout)
        start_widget = row_widget.layout().itemAt(0).widget()
        model_btn = start_widget.layout().itemAt(1).widget()  # Second widget is model button
        model_name = model_btn.text()

        if model_name != 'None' and model_name != 'baseline':
            model.append(model_name)

        model_param_count = mod_len_def(model_name, include_special=False) + 1

        # Read parameters for this model
        for j in range(1, min(model_param_count, numco + 1)):
            if j < row_widget.layout().count():
                _append_param(_field_text(row_widget, j))

        # Handle special model types (their expression texts live in the last
        # column; the expression itself gets a placeholder slot in p)
        if model_name == 'Expression':
            Expr.append(_field_text(row_widget, 1))
            NExpr = np.append(NExpr, len(p))
            p = np.append(p, 0)

        elif model_name == 'Distr':
            # Distribution has 4 numeric parameters plus the expression
            for j in range(1, 5):
                if j < row_widget.layout().count():
                    _append_param(_field_text(row_widget, j))
            p = np.append(p, 0)
            # Get distribution expression from 5th field
            distri_text = _field_text(row_widget, 5) if 5 < row_widget.layout().count() else ''
            Distri.append(distri_text)
            DistriN = np.append(DistriN, len(p) - 1)

        elif model_name == 'Corr':
            # Correlation has 1 numeric parameter (par) plus the expression
            # (Dependency function)
            if 1 < row_widget.layout().count():
                _append_param(_field_text(row_widget, 1))
            p = np.append(p, 0)
            cor_text = _field_text(row_widget, 2) if 2 < row_widget.layout().count() else ''
            Cor.append(cor_text)

        elif model_name == 'Recon':
            # Reconstruction: 6 numeric control params (par, L, R, Num, D_dif,
            # D_dif2) followed by a single weight-vector placeholder slot. The
            # Num free reconstruction weights are stored as a text vector in the
            # 7th column and travel in the parallel Recon list (like a Distr's PDF
            # string in Distri); the placeholder keeps the flat count fixed at 7.
            for j in range(1, 7):
                if j < row_widget.layout().count():
                    _append_param(_field_text(row_widget, j))
            num_text = _field_text(row_widget, 4) if 4 < row_widget.layout().count() else ''
            try:
                num = int(float(num_text)) if str(num_text).strip() else 1
            except ValueError:
                num = 1
            p = np.append(p, 0)                                # weight-vector placeholder slot
            weights_text = _field_text(row_widget, 7) if 7 < row_widget.layout().count() else ''
            Recon.append(parse_recon_weights(weights_text, num))
            ReconN = np.append(ReconN, len(p) - 1)

    return (model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN)


def read_bounds_and_fix(main_window, p_len):
    """Read box bounds and fixed-parameter indices from the parameters table.

    Companion to :func:`read_model` — the same table walk, but collecting the
    lower/upper bound fields and the "fix" checkboxes into the solver inputs.
    One shared implementation for the fit (fitting_io) and the instrumental
    refinement (instrumental_io).

    Args:
        main_window: The main PhysicsApp window instance
        p_len: Length of the flat parameter array read_model produced

    Returns:
        tuple: (bounds, fix) — bounds is a (2, p_len) float array with
        -inf/+inf where a field is empty; fix is a unique-free int array of
        parameter indices whose checkbox is ticked.
    """
    bounds = np.array([[-np.inf] * p_len, [np.inf] * p_len], dtype=float)
    fix = np.array([], dtype=int)

    def _read_column(row_widget, item_idx, param_idx):
        nonlocal fix
        param_widget = row_widget.layout().itemAt(item_idx).widget()
        bounds_layout = param_widget.layout().itemAt(2).layout()
        lower_text = bounds_layout.itemAt(0).widget().text()
        upper_text = bounds_layout.itemAt(1).widget().text()
        if lower_text:
            bounds[0][param_idx] = float(lower_text)
        if upper_text:
            bounds[1][param_idx] = float(upper_text)
        fix_cb = param_widget.layout().itemAt(0).layout().itemAt(1).widget()
        if fix_cb.isChecked():
            fix = np.append(fix, param_idx)

    # Baseline (row 0); itemAt(0) is the start widget, columns begin at 1
    baseline_row = main_window.params_table.row_widgets[0]
    for j in range(number_of_baseline_parameters):
        _read_column(baseline_row, j + 1, j)

    # Model rows
    V = number_of_baseline_parameters - 1
    for i in range(1, len(main_window.params_table.row_widgets)):
        row_widget = main_window.params_table.row_widgets[i]
        model_btn = row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
        model_name = model_btn.text()
        if model_name == 'None':
            continue
        for j in range(mod_len_def(model_name, include_special=True)):
            V += 1
            if j + 1 < row_widget.layout().count():
                _read_column(row_widget, j + 1, V)

    return bounds, fix


def validate_user_expressions(main_window):
    """Try to evaluate every user-typed Expression/Distr/Corr text.

    Called before a fit or show-model starts, so a meaningless expression is
    rejected up front instead of crashing mid-procedure. Each expression is
    evaluated in the same namespace the real computation uses: Expression via
    minimi_lib._eval_expr (bare numpy names + the parameter array ``p``),
    Distr/Corr in the models-module namespace with the distribution axis ``X``
    and ``p`` available (mimicking ``eval(...) + 0*X`` in models.TImod).

    Returns:
        list of problem dicts ``{'kind', 'occurrence', 'row', 'text', 'error'}``;
        empty when everything evaluates. ``row`` is the parameters-table row of
        the offending field (None if it could not be located).
    """
    import syncmoss.minimi_lib as mi
    import syncmoss.models as m5

    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(main_window)
    X = np.linspace(-1.0, 1.0, 8)
    rows = main_window.params_table.get_expression_rows()
    problems = []

    def _check(kind, occurrence, text, evaluator):
        try:
            if not str(text).strip():
                raise ValueError("expression is empty")
            value = evaluator(str(text))
            np.asarray(value, dtype=float)
        except Exception as e:
            kind_rows = rows.get(kind, [])
            row = kind_rows[occurrence] if occurrence < len(kind_rows) else None
            problems.append({'kind': kind, 'occurrence': occurrence, 'row': row,
                             'text': text, 'error': e})

    for k, text in enumerate(Expr):
        _check('Expression', k, text, lambda s: float(mi._eval_expr(s, p)))
    for k, text in enumerate(Distri):
        _check('Distr', k, text, lambda s: eval(s, vars(m5), {'X': X, 'p': p}) + 0 * X)
    for k, text in enumerate(Cor):
        _check('Corr', k, text, lambda s: eval(s, vars(m5), {'X': X, 'p': p}) + 0 * X)

    return problems