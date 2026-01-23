"""
Module for spectrum I/O operations and processing.
Includes functions for loading spectra, summing, subtracting models, and halving points.
"""

import os
import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox
import re
import platform
from models import TI
from models_positions import mod_pos
from constants import number_of_baseline_parameters
from model_io import read_model


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
        if not os.path.isabs(calibration_path):
            calibration_path = os.path.join(main_window.dir_path, calibration_path)
    else:
        # Use directory of first file
        if not os.path.isabs(calibration_path):
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


def sum_all_spectra(main_window):
    """Sum all selected spectra into one and save it"""
    try:
        # Parse the current content of process_path
        main_window.path_list = main_window.parse_process_path()

        if not main_window.path_list:
            main_window.log.setPlainText("No spectra selected")
            main_window.log.setStyleSheet("color: orange;")
            return

        if len(main_window.path_list) < 2:
            main_window.log.setPlainText("Need at least 2 spectra to sum")
            main_window.log.setStyleSheet("color: orange;")
            return

        main_window.log.setPlainText(f"Loading {len(main_window.path_list)} spectra for summing...")
        main_window.log.setStyleSheet("color: blue;")

        # Load all spectra
        A_list, B_list = load_spectrum(main_window, main_window.path_list, calibration_path=main_window.calibration_path)

        if not A_list or not B_list or len(A_list) == 0 or len(B_list) == 0:
            main_window.log.setPlainText("Failed to load spectra")
            main_window.log.setStyleSheet("color: red;")
            return

        # Check if all spectra have the same x-axis (length)
        if not all(len(A) == len(A_list[0]) for A in A_list):
            main_window.log.setPlainText("Spectra have different lengths - cannot sum")
            main_window.log.setStyleSheet("color: red;")
            return

        # Sum the y-values (intensities)
        summed_B = np.sum(B_list, axis=0)
        summed_A = A_list[0]  # Use x-axis from first spectrum

        # Ask user where to save
        default_name = f"sum.dat"
        save_path, _ = QFileDialog.getSaveFileName(
            main_window,
            "Save summed spectrum",
            os.path.join(main_window.workfolder or "", default_name),
            "DAT files (*.dat);;TXT files (*.txt);;All files (*.*)"
        )

        if not save_path:
            main_window.log.setPlainText("Save canceled")
            main_window.log.setStyleSheet("color: orange;")
            return

        # Save the summed spectrum
        np.savetxt(save_path, np.column_stack((summed_A, summed_B)), delimiter='\t', fmt='%.6f')

        # Auto-load the saved spectrum
        main_window.process_path.setPlainText(f"['{save_path}']")
        main_window.show_pressed()  # This will load and display the summed spectrum

        main_window.log.setPlainText(f"Summed {len(main_window.path_list)} spectra and saved to {os.path.basename(save_path)}")
        main_window.log.setStyleSheet("color: green;")

    except Exception as e:
        main_window.log.setPlainText(f"Error summing spectra: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def subtract_model_from_spectrum(main_window):
    """Subtract the current model from the selected spectrum"""
    try:
        # Parse the current content of process_path
        main_window.path_list = main_window.parse_process_path()

        if not main_window.path_list:
            main_window.log.setPlainText("No spectrum selected")
            main_window.log.setStyleSheet("color: orange;")
            return

        if len(main_window.path_list) != 1:
            main_window.log.setPlainText("Please select exactly ONE spectrum for this operation")
            main_window.log.setStyleSheet("color: orange;")
            return

        spectrum_path = main_window.path_list[0]
        if not os.path.exists(spectrum_path):
            main_window.log.setPlainText("Selected spectrum file does not exist")
            main_window.log.setStyleSheet("color: red;")
            return

        main_window.log.setPlainText("Loading spectrum and calculating model...")
        main_window.log.setStyleSheet("color: blue;")

        # Load the spectrum
        A_list, B_list = load_spectrum(main_window, [spectrum_path], calibration_path=main_window.calibration_path)

        if not A_list or not B_list or len(A_list) == 0 or len(B_list) == 0:
            main_window.log.setPlainText("Failed to load spectrum")
            main_window.log.setStyleSheet("color: red;")
            return

        A = A_list[0]
        B = B_list[0]

        # Read current model and parameters
        from model_io import read_model
        model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN = read_model(main_window)

        if len(model) == 0:
            main_window.log.setPlainText("No model defined - nothing to subtract")
            main_window.log.setStyleSheet("color: orange;")
            return

        # Apply expressions
        for i in range(len(NExpr)):
            try:
                p[NExpr[i]] = eval(Expr[i])
            except Exception as e:
                main_window.log.setPlainText(f"Error evaluating expression: {e}")
                main_window.log.setStyleSheet("color: red;")
                return

        # Apply constraints
        for i in range(len(con1)):
            p[int(con1[i])] = p[int(con2[i])] * con3[i]

        # Get experimental method parameters
        JN = int(main_window.jn0_input.text())
        experimental_method = 1 if main_window.chb1.isChecked() else 3  # 1=MS, 3=SMS

        # Setup parameters based on experimental method
        if experimental_method == 1:  # MS
            INS = float(main_window.l0_input.text())
            pNorm = np.array([float(0)] * number_of_baseline_parameters)
            pNorm[0] = 1
            Norm = 1.0
            method_params = {
                'x0': 0.0,
                'MulCo': 1.0,
                'INS': INS,
                'Met': 1,
                'Norm': Norm
            }
        else:  # SMS
            INS = main_window.INS
            pNorm = np.array([float(0)] * number_of_baseline_parameters)
            pNorm[0] = 1
            Norm = TI(np.array([float(1000)]), pNorm, [], JN, main_window.pool, main_window.x0, main_window.MulCo, INS, [0], [0])[0]
            method_params = {
                'x0': main_window.x0,
                'MulCo': main_window.MulCo,
                'INS': INS,
                'Met': 0,
                'Norm': Norm
            }

        # Calculate model spectrum
        try:
            SPC_f = TI(A, p, model, JN, main_window.pool,
                         method_params['x0'], method_params['MulCo'],
                         method_params['INS'], Distri, Cor,
                         Met=method_params['Met'], Norm=method_params['Norm'])
        except Exception as e:
            main_window.log.setPlainText(f"Error calculating model spectrum: {e}")
            main_window.log.setStyleSheet("color: red;")
            return

        # Calculate baseline properly as in the TI function
        # N0 = p[0] + p[3]*p[0]/100 * A + p[2]*p[0]/10000 * (A - p[1])**2
        # N1 = p[4] + p[7]*p[4]/100 * A + p[6]*p[4]/10000 * (A - p[5])**2
        # baseline = N0 + N1
        baseline = (p[0] + p[3] * p[0] / 100 * A + p[2] * p[0] / 10000 * (A - p[1])**2) + \
                  (p[4] + p[7] * p[4] / 100 * A + p[6] * p[4] / 10000 * (A - p[5])**2)

        # Subtract model from spectrum: B = B - SPC_f + baseline
        B_subtracted = B - SPC_f + baseline

        # Create default filename
        basename = os.path.splitext(os.path.basename(spectrum_path))[0]
        default_name = f"subtracted_{basename}.dat"

        # Ask user where to save
        save_path, _ = QFileDialog.getSaveFileName(
            main_window,
            "Save subtracted spectrum",
            os.path.join(main_window.workfolder or "", default_name),
            "DAT files (*.dat);;TXT files (*.txt);;All files (*.*)"
        )

        if not save_path:
            main_window.log.setPlainText("Save canceled")
            main_window.log.setStyleSheet("color: orange;")
            return

        # Save the subtracted spectrum
        np.savetxt(save_path, np.column_stack((A, B_subtracted)), delimiter='\t', fmt='%.6f')

        # Auto-load the saved spectrum
        main_window.process_path.setPlainText(f"['{save_path}']")
        main_window.show_pressed()  # This will load and display the subtracted spectrum

        main_window.log.setPlainText(f"Model subtracted and saved to {os.path.basename(save_path)}")
        main_window.log.setStyleSheet("color: green;")

    except Exception as e:
        main_window.log.setPlainText(f"Error subtracting model: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def half_points(main_window):
    """Reduce spectrum points by half by averaging consecutive pairs"""
    try:
        # Parse the current content of process_path
        main_window.path_list = main_window.parse_process_path()

        if not main_window.path_list:
            main_window.log.setPlainText("No spectrum selected")
            main_window.log.setStyleSheet("color: orange;")
            return

        num_spectra = len(main_window.path_list)

        if num_spectra == 1:
            # Single spectrum case
            half_single_spectrum(main_window)
        else:
            # Multiple spectra case - ask for confirmation
            reply = QMessageBox.question(
                main_window,
                "Process Multiple Spectra",
                f"Process {num_spectra} spectra? This will create half-point versions of all selected spectra.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                half_multiple_spectra(main_window)
            else:
                main_window.log.setPlainText("Half points operation canceled")
                main_window.log.setStyleSheet("color: orange;")

    except Exception as e:
        main_window.log.setPlainText(f"Error in half points: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def half_single_spectrum(main_window):
    """Process single spectrum - show save dialog"""
    try:
        spectrum_path = main_window.path_list[0]

        # Create default filename
        basename = os.path.splitext(os.path.basename(spectrum_path))[0]
        default_name = f"half_{basename}.dat"

        # Ask user where to save
        save_path, _ = QFileDialog.getSaveFileName(
            main_window,
            "Save half-points spectrum",
            os.path.join(main_window.workfolder or "", default_name),
            "DAT files (*.dat);;TXT files (*.txt);;All files (*.*)"
        )

        if not save_path:
            main_window.log.setPlainText("Save canceled")
            main_window.log.setStyleSheet("color: orange;")
            return

        # Process and save the spectrum
        success = process_half_spectrum(main_window, spectrum_path, save_path, auto_load=True)

        if success:
            main_window.log.setPlainText(f"Half-points spectrum saved to {os.path.basename(save_path)}")
            main_window.log.setStyleSheet("color: green;")

    except Exception as e:
        main_window.log.setPlainText(f"Error processing single spectrum: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def process_half_spectrum(main_window, spectrum_path, save_path, auto_load=False):
    """Core function to process a single spectrum by halving points"""
    try:
        # Load the spectrum
        A_list, B_list = load_spectrum(main_window, [spectrum_path], calibration_path=main_window.calibration_path)

        if not A_list or not B_list or len(A_list) == 0 or len(B_list) == 0:
            return False

        A = A_list[0]
        B = B_list[0]

        # Half the points
        if len(A) < 2:
            return False

        half_len = len(A) // 2
        A_half = np.zeros(half_len)
        B_half = np.zeros(half_len)

        for i in range(half_len):
            A_half[i] = (A[i*2] + A[i*2+1]) / 2  # Average velocity
            B_half[i] = B[i*2] + B[i*2+1]  # Sum intensity

        # Save the half-points spectrum
        np.savetxt(save_path, np.column_stack((A_half, B_half)), delimiter='\t', fmt='%.6f')

        # Auto-load if requested
        if auto_load:
            main_window.process_path.setPlainText(f"['{save_path}']")
            main_window.show_pressed()  # This will load and display the half-points spectrum

        return True

    except Exception as e:
        return False


def half_multiple_spectra(main_window):
    """Process multiple spectra - save to save_path directory"""
    try:
        save_path_text = main_window.save_path.text().strip()

        if not save_path_text:
            main_window.log.setPlainText("Please specify save path")
            main_window.log.setStyleSheet("color: orange;")
            return

        # Determine if save_path_text is a file path or directory path
        if '.' in os.path.basename(save_path_text):
            # Looks like a file path (has extension), use its directory
            save_dir = os.path.dirname(save_path_text)
        else:
            # Looks like a directory path
            save_dir = save_path_text

        # Remove trailing slashes for consistency
        save_dir = save_dir.rstrip('\\/')

        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                main_window.log.setPlainText(f"Cannot create directory {save_dir}: {e}")
                main_window.log.setStyleSheet("color: red;")
                return

        # Verify it's a directory
        if not os.path.isdir(save_dir):
            main_window.log.setPlainText("Save path must be a directory for multiple spectra")
            main_window.log.setStyleSheet("color: orange;")
            return

        processed_count = 0

        for spectrum_path in main_window.path_list:
            try:
                # Create filename
                basename = os.path.splitext(os.path.basename(spectrum_path))[0]
                save_filename = f"half_{basename}.dat"
                save_path = os.path.join(save_dir, save_filename)

                # Process and save the spectrum
                success = process_half_spectrum(main_window, spectrum_path, save_path, auto_load=False)

                if success:
                    processed_count += 1

            except Exception as e:
                # Continue with next spectrum if one fails
                continue

        if processed_count > 0:
            main_window.log.setPlainText(f"Processed {processed_count} spectra, saved to {save_dir}")
            main_window.log.setStyleSheet("color: green;")
        else:
            main_window.log.setPlainText("No spectra were successfully processed")
            main_window.log.setStyleSheet("color: orange;")

    except Exception as e:
        main_window.log.setPlainText(f"Error processing multiple spectra: {str(e)}")
        main_window.log.setStyleSheet("color: red;")


def calculate_backgrounds(paths, calibration_path="Calibration.dat"):
    """
    Calculate background levels for a list of spectrum files.

    Args:
        paths: List of file paths to spectrum files
        calibration_path: Path to calibration file (default: "Calibration.dat")

    Returns:
        List of background values (N0) for each spectrum
    """
    backgrounds = []
    for path in paths:
        try:
            # Load the spectrum
            A_list, B_list = load_spectrum(None, [path], calibration_path=calibration_path)  # Pass None as main_window since we don't need it
            if A_list and B_list and len(A_list) > 0 and len(B_list) > 0 and len(A_list[0]) > 0 and len(B_list[0]) > 0:
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
                backgrounds.append(1)
        except Exception as e:
            print(f"Error calculating background for {path}: {e}")
            backgrounds.append(1)

    return backgrounds