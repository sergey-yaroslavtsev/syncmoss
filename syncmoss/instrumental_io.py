"""
Instrumental function calculation and refinement module for SYNCMoss.
Handles the calculation, fitting, and refinement of instrumental functions.
"""
import os
import time
import numpy as np
from PySide6.QtWidgets import QMessageBox
from syncmoss.constants import number_of_baseline_parameters
import syncmoss.models as m5
import syncmoss.minimi_lib as mi
from syncmoss.spectrum_io import load_spectrum
from syncmoss.model_io import read_model, mod_len_def
from syncmoss.spectrum_plotter import plot_instrumental_result


# Built-in defaults used by "Reset to default values" in Instrumental function menu.
DEFAULT_INSEXP_TEXT = (
    "0.16472623639571624 0.11257879642037069 0.7019583961622452 "
    "0.08985576770449756 -0.05913725754557238 0.42733862867647215 "
    "-0.5403075049599122 0.012656364797966438 0.5697684674481738 "
)
DEFAULT_INSINT_TEXT = "2.448293819453453 0.03380920627191801 "

DAT_INS_EXP_PREFIX = '#@INSexp'
DAT_INS_INT_PREFIX = '#@INSint'


def _parse_float_sequence(text):
    """Parse whitespace/comma separated float values from string."""
    if text is None:
        return np.array([], dtype=float)
    cleaned = str(text).replace(',', ' ').strip()
    if not cleaned:
        return np.array([], dtype=float)
    values = []
    for token in cleaned.split():
        values.append(float(token))
    return np.array(values, dtype=float)


def parse_dat_instrumental_metadata(file_path):
    """Read #@INSexp / #@INSint metadata from a .dat file header."""
    result = {
        'INS': None,
        'MulCo': None,
        'x0': None,
        'has_insexp': False,
        'has_insint': False,
    }

    if not file_path or not os.path.exists(file_path):
        return result

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith(DAT_INS_EXP_PREFIX):
                    payload = stripped[len(DAT_INS_EXP_PREFIX):].strip()
                    parsed = _parse_float_sequence(payload)
                    if parsed.size > 0:
                        result['INS'] = parsed
                        result['has_insexp'] = True
                    continue

                if stripped.startswith(DAT_INS_INT_PREFIX):
                    payload = stripped[len(DAT_INS_INT_PREFIX):].strip()
                    parsed = _parse_float_sequence(payload)
                    if parsed.size >= 2:
                        result['MulCo'] = float(parsed[0])
                        result['x0'] = float(parsed[1])
                        result['has_insint'] = True
                    continue

                if not stripped.startswith('#') and not stripped.startswith('<'):
                    break
    except Exception as e:
        print(f"[Instrumental function] Could not read DAT metadata from {file_path}: {e}")

    return result


def get_sms_instrumental_from_global_files(app):
    """Read SMS instrumental function from global INSexp/INSint files."""
    insexp_path = os.path.join(app.dir_path, 'parameters', 'INSexp.txt')
    instrumental_int_path = os.path.join(app.dir_path, 'parameters', 'INSint.txt')

    INS = np.genfromtxt(insexp_path, delimiter=' ', skip_footer=0)
    MulCo, x0 = np.genfromtxt(instrumental_int_path, delimiter=' ', skip_footer=0)
    return np.array(INS, dtype=float), float(MulCo), float(x0)


def resolve_sms_instrumental_for_file(app, spectrum_file, use_dat_metadata=True):
    """
    Resolve SMS instrumental parameters for a spectrum.

    Returns:
        INS, MulCo, x0, note
    """
    INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(app)

    if not use_dat_metadata:
        return INS_global, MulCo_global, x0_global, "Using global INSexp/INSint files (.dat metadata use is disabled)."

    if not spectrum_file or not str(spectrum_file).lower().endswith('.dat'):
        return INS_global, MulCo_global, x0_global, "Using global INSexp/INSint files (input is not a .dat file)."

    meta = parse_dat_instrumental_metadata(spectrum_file)
    if meta['INS'] is not None and meta['MulCo'] is not None and meta['x0'] is not None:
        return meta['INS'], meta['MulCo'], meta['x0'], f"Using instrumental function from .dat file: {os.path.basename(spectrum_file)}"

    missing = []
    if meta['INS'] is None:
        missing.append('#@INSexp')
    if meta['MulCo'] is None or meta['x0'] is None:
        missing.append('#@INSint')
    missing_txt = ', '.join(missing) if missing else 'metadata'

    return (
        INS_global,
        MulCo_global,
        x0_global,
        f".dat instrumental metadata not found ({missing_txt}) in {os.path.basename(spectrum_file)}. Fallback to global INSexp/INSint files.",
    )


def reset_instrumental_defaults(app):
    """Restore default values for INSexp.txt and INSint.txt."""
    try:
        insexp_path = os.path.join(app.dir_path, 'parameters', 'INSexp.txt')
        instrumental_int_path = os.path.join(app.dir_path, 'parameters', 'INSint.txt')
        os.makedirs(os.path.dirname(insexp_path), exist_ok=True)

        with open(insexp_path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_INSEXP_TEXT)

        with open(instrumental_int_path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_INSINT_TEXT)

        app.log.setPlainText("Instrumental defaults restored (INSexp.txt, INSint.txt)")
        app.log.setStyleSheet("color: green;")

        QMessageBox.information(app, "Instrumental function", "Default instrumental values were restored.")
        return True
    except Exception as e:
        app.log.setPlainText(f"Failed to restore instrumental defaults: {e}")
        app.log.setStyleSheet("color: red;")
        return False




def instrumental(app, ref, mode=0, pool=None):
    """
    Calculate or refine instrumental function.
    
    Args:
        app: The main application instance
        ref: Reference mode (0=find, 1=refine)
        mode: Calculation mode (0=single line, 1=model, 2=pure a-Fe)
        pool: Multiprocessing pool for parallel computation
    
    Returns:
        dict: Results containing parameters, errors, chi-squared, and file paths
    """
    print(f"[Instrumental function] instrumental() called with ref={ref}, mode={mode}")
    
    # Get parameters from app
    JN0 = app.JN0
    x0 = app.x0
    MulCo = app.MulCo
    MulCoCMS = 0.28
    
    file = os.path.abspath(app.path_list[0])
    A_list, B_list = load_spectrum(app, [file], calibration_path=app.calibration_path)
    A, B = A_list[0], B_list[0]
    
    CMS_ch = 0
    
    # Initialize parameters based on mode
    if ref == 0:
        x0 = -0.01
        MulCo = 2.2
        n = int(app.instrumental_number.text())
        print('Instrumental procedure start with', ref, mode, n)
        p0 = np.array([])
        
        if mode == 0 or mode == 2:
            p0 = np.array([float(0.001)] * (3 * (n >= 3) + n * (n < 3)) * 3)
            try:
                p0[0] = 0.2
                p0[1] = 0.055
                p0[2] = 0.8
            except:
                pass
            try:
                p0[3] = 0.11
                p0[4] = -0.13
                p0[5] = 0.44
            except:
                pass
            try:
                p0[6] = 0.555
                p0[7] = -0.07
                p0[8] = 0.4
            except:
                pass
        
        if n > 3 or mode == 1:
            p00c = np.array([float(0.001)] * ((n-3)*(mode==0 + mode==2) + n*(mode==1)) * 3)
            n0 = int(len(p00c)/3)
            for i in range(0, int(n0 // 2)):
                p00c[i * 3] = 0.3
                p00c[i * 3 + 1] = -1.5 + 3 * i / max(int(n0 // 2 - 1), 1)
                p00c[i * 3 + 2] = 1 / n0
            for i in range(int(n0 // 2), n0):
                p00c[i * 3] = 0.15
                p00c[i * 3 + 1] = -0.4 + 0.8 * (i - int(n0 // 2)) / max(int(n0 // 2 + n0 % 2 - 1), 1)
                p00c[i * 3 + 2] = 1 / n0
            p0 = np.concatenate((p0, p00c))
            print(p00c)
        
        print('Here is initial guess for INS:')
        print(p0)
    
    if ref == 1:
        # Check for CMS mode
        if app.MS_fit.isChecked():
            INS = np.array([float(app.GCMS)])
            p0 = np.copy(INS)
            print('Instrumental procedure for CMS start with', ref, mode)
            print('Here is initial guess for INS: ', p0)
            CMS_ch = 1
        elif app.SMS_fit.isChecked():
            use_dat_metadata = bool(getattr(app, 'use_dat_instrumental_metadata', True))
            INS, MulCo, x0, note = resolve_sms_instrumental_for_file(app, file, use_dat_metadata=use_dat_metadata)
            print(f"[Instrumental function] {note}")
            p0 = np.copy(INS)
            n = int(len(INS)/3)
            print('Instrumental procedure start with', ref, mode, n)
            print('Here is initial guess for INS:')
            print(p0)
        else:
            print('No instrumental function refinement mode selected.')
            return

    # Normalize instrumental function parameters
    if CMS_ch == 0:
        SC = 0
        for i in range(0, int((len(p0)) / 3)):
            SC += p0[i * 3 + 2]**2
        for i in range(0, int((len(p0)) / 3)):
            p0[i * 3 + 2] = np.sqrt(p0[i * 3 + 2]**2 / SC)
        
        bounds0 = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])
        for i in range(0, int((len(p0)) / 3)):
            bounds0[0][i * 3] = -0.7
            bounds0[1][i * 3] = 0.7
            bounds0[0][i * 3 + 1] = -1.5
            bounds0[1][i * 3 + 1] = 1.5
    else:
        bounds0 = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])
        bounds0[0][0] = 0.001
        bounds0[1][0] = 1
    
    # Read model if mode == 1
    if mode == 1:
        model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN = read_model(app)
        
        # Apply expressions
        for i in range(0, len(NExpr)):
            p[NExpr[i]] = eval(Expr[i])
        for i in range(0, len(con1)):
            p[int(con1[i])] = p[int(con2[i])] * con3[i]
        
        confu = np.array([con1, con2, con3]) if len(con1) > 0 else np.array([[-1], [-1], [-1]])
        
        # Build bounds and fix arrays from params_table
        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        fix = np.array([], dtype=int)
        
        # Read bounds and fix from baseline (row 0)
        baseline_row = app.params_table.row_widgets[0]
        for j in range(number_of_baseline_parameters):
            param_widget = baseline_row.layout().itemAt(j + 1).widget()  # +1 because itemAt(0) is start_widget
            # Get bounds
            bounds_layout = param_widget.layout().itemAt(2).layout()  # Third item is bounds
            lower_input = bounds_layout.itemAt(0).widget()
            upper_input = bounds_layout.itemAt(1).widget()
            if lower_input.text():
                bounds[0][j] = float(lower_input.text())
            if upper_input.text():
                bounds[1][j] = float(upper_input.text())
            # Get fix checkbox
            top_layout = param_widget.layout().itemAt(0).layout()
            fix_cb = top_layout.itemAt(1).widget()
            if fix_cb.isChecked():
                fix = np.append(fix, j)
        
        # Read bounds and fix from model rows
        V = number_of_baseline_parameters - 1
        for i in range(1, len(app.params_table.row_widgets)):
            row_widget = app.params_table.row_widgets[i]
            start_widget = row_widget.layout().itemAt(0).widget()
            model_btn = start_widget.layout().itemAt(1).widget()
            model_name = model_btn.text()
            
            if model_name != 'None':
                LenM = mod_len_def(model_name)
                for j in range(LenM):
                    V += 1
                    if j + 1 < row_widget.layout().count():
                        param_widget = row_widget.layout().itemAt(j + 1).widget()
                        # Get bounds
                        bounds_layout = param_widget.layout().itemAt(2).layout()
                        lower_input = bounds_layout.itemAt(0).widget()
                        upper_input = bounds_layout.itemAt(1).widget()
                        if lower_input.text():
                            bounds[0][V] = float(lower_input.text())
                        if upper_input.text():
                            bounds[1][V] = float(upper_input.text())
                        # Get fix checkbox
                        top_layout = param_widget.layout().itemAt(0).layout()
                        fix_cb = top_layout.itemAt(1).widget()
                        if fix_cb.isChecked():
                            fix = np.append(fix, V)
        
        # Add constraint and distribution indices to fix
        fix = np.concatenate((fix, con1), axis=0)
        fix = np.concatenate((fix, DistriN), axis=0)
        fix = np.concatenate((fix, NExpr), axis=0)
        fix = np.unique(fix)
        
        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)
        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:], Distri, Cor)
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Distri, Cor, Met=1)
    
    # CMS_ch could not be equal to 1 here
    # Not meaningful to use ESRF standard single line absorber for CMS
    elif mode == 0: 
        model = ['Doublet', 'Singlet']
        if CMS_ch == 0:
            p = np.array([(B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10])
            p = np.concatenate((p, np.array([0, 0, 0, 0, 0, 0, 0])))
        
        try:
            be_path = os.path.join(app.dir_path, 'parameters', 'Be.txt')
            Be_param = np.genfromtxt(be_path, delimiter='\t', skip_footer=0)
            print('Be file was read')
        except:
            Be_param = np.array([0.057, 0.066, -0.261, 0.098, 0.375, 0.772, 1])
            print('COULD NOT READ Be.txt')

        p = np.concatenate((p, Be_param))
        p1 = np.array([4.6, -0.097, 0.098, 0.0])
        p = np.concatenate((p, p1))
        
        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][15] = 0.001
        if CMS_ch == 0:
            # Fix everything except Ns + T of singlet
            fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18], dtype=int)
        
        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)
        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        print(f"[Instrumental function] Mode 0: Creating INSSS function, CMS_ch={CMS_ch}")
        
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:])
            print(f"[Instrumental function] INSSS function created for CMS_ch=0")
    
    elif mode == 2:
        model = ['Doublet', 'Sextet']
        if CMS_ch == 0:
            p = np.array([(B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10])
            p = np.concatenate((p, np.array([0, 0, 0, 0, 0, 0, 0])))
        if CMS_ch == 1:
            bg_tmp = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10
            p = np.array([bg_tmp*0.6])
            p = np.concatenate((p, np.array([0, 0, 0, bg_tmp*0.4, 0, 0, 0])))
        
        try:
            be_path = os.path.join(app.dir_path, 'parameters', 'Be.txt')
            Be_param = np.genfromtxt(be_path, delimiter='\t', skip_footer=0)
            print('Be file was read')
        except:
            Be_param = np.array([0.057, 0.066, -0.261, 0.098, 0.375, 0.772, 1])
            print('COULD NOT READ Be.txt')
        
        if CMS_ch == 1:
            Be_param[0] = 0
        
        p = np.concatenate((p, Be_param))
        p1 = np.array([7.5, 0, 0, 33.04, 0.098, 0, 0.5, 0, 0, 0, 3])
        p = np.concatenate((p, p1))
        
        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][15] = 0.001
        if CMS_ch == 0:
            # Fix everything except Ns + T and A of sextet
            fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25], dtype=int)
        if CMS_ch == 1:
            # Fix everything except Ns, Nnr + T and A of sextet
            fix = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25], dtype=int)
            bounds[0][4] = 0
        
        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)
        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:])
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Met=1)
    
    # Normalization function
    # to insure sum of amplitudes squared = 1 in case of SMS
    def INS_norm(p):
        SC = 0
        for i in range(0, int((len(p[mod_p_len:])) / 3)):
            SC += p[mod_p_len + i * 3 + 2]**2
        for i in range(0, int((len(p[mod_p_len:])) / 3)):
            p[mod_p_len + i * 3 + 2] = np.sqrt(p[mod_p_len + i * 3 + 2]**2 / SC)
        # scale the baseline (Ns)
        p[0] = p[0] * SC
        return p
    
    JN = np.copy(JN0)
    JN = max(JN*2, 64)
    print(f"[Instrumental function] Starting minimization: JN={JN}, len(p)={len(p)}, len(fix)={len(fix)}")
    
    start_time = time.time()
    
    # Preliminary minimization
    print(f"[Instrumental function] Calling minimization procedure - preliminary step...")
    p, er, hi2, covariance_matrix = mi.minimi_hi(INSSS, A, B, p, fix=fix, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10**-6, fixCH=1)
    print(f"[Instrumental function] Preliminary minimization complete, hi2={hi2}")
    if CMS_ch == 0:
        p = INS_norm(p)        
        x0t, MulCot = np.copy(x0), np.copy(MulCo)
        hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])
        hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        if hi2 - hi2t > 0.01:
            x0, MulCo = x0t, MulCot
            hi2 = np.copy(hi2t)
    print(f"[Instrumental function] parameters after preliminary minimization:")
    print(p)
    
    # Try to reduce number of lines in INS 
    # (only if ref == 0 and CMS_ch == 0)
    for Nc in range(0, 2*(1-ref)*(CMS_ch==0)):
        n_s = int((len(p[mod_p_len:])) / 3)
        if n_s > 3:
            print('Trying to reduce number of lines in INS')
            J = 0
            for j in range(0, n_s):
                I_ch = mod_p_len + J * 3 + 2
                pt = np.copy(p)
                pt = np.delete(pt, I_ch)
                pt = np.delete(pt, I_ch-1)
                pt = np.delete(pt, I_ch-2)
                boundst = np.copy(bounds[0])
                boundst = np.delete(boundst, I_ch)
                boundst = np.delete(boundst, I_ch - 1)
                boundst = np.delete(boundst, I_ch - 2)
                boundstt = np.copy(bounds[1])
                boundstt = np.delete(boundstt, I_ch)
                boundstt = np.delete(boundstt, I_ch - 1)
                boundstt = np.delete(boundstt, I_ch - 2)
                boundsttt = np.array([boundst, boundstt])
                pt = INS_norm(pt)
                
                pt, ert, hi2t, covariance_matrix_t = mi.minimi_hi(INSSS, A, B, pt, fix=fix, bounds=boundsttt, tau0=0.0001, MI=20, MI2=10, eps=10**-6, fixCH=1)
                
                x0t, MulCot = np.copy(x0), np.copy(MulCo)
                hi2tt = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
                x0, MulCo = m5.limits(pool, int(JN), pt[mod_p_len:])
                hi2t = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
                if hi2t - hi2tt > 0.01:
                    x0, MulCo = x0t, MulCot
                    hi2t = hi2tt
                
                if hi2t - hi2 < 0.01:
                    p = pt
                    bounds = boundsttt
                    p = INS_norm(p)
                    print('Chi squared:', hi2, hi2t)
                    hi2 = np.copy(hi2t)
                else:
                    x0, MulCo = x0t, MulCot
                    J += 1
        
        p, er, hi2, covariance_matrix = mi.minimi_hi(INSSS, A, B, p, fix=fix, bounds=bounds, tau0=0.0001, MI=20, MI2=10, eps=10**-6, fixCH=1)
        print(f"[Instrumental Function] chi-squared after reducing number of lines: {hi2}")
        p = INS_norm(p)
        print(f"[Instrumental Function] parameters after reducing number of lines:")
        print(p)
        
        x0t, MulCot = np.copy(x0), np.copy(MulCo)
        hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])
        hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        if hi2 - hi2t > 0.01:
            x0, MulCo = x0t, MulCot
            hi2 = hi2t
        print(f"[Instrumental function] x0: {x0}, MulCo: {MulCo}")
    
    JN = np.copy(JN0)
    
    # Final refinement
    for Nc in range(0, 3):
        p, er, hi2, covariance_matrix = mi.minimi_hi(INSSS, A, B, p, fix=fix, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10**-6, fixCH=1)
        if CMS_ch == 0:
            p = INS_norm(p)
            x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])

    print(f"[Instrumental function] Final chi-squared: {hi2}")
    print(f"[Instrumental function] Final parameters after refinement:")
    print(p)
    print(f"[Instrumental function] x0: {x0}, MulCo: {MulCo}")
    
    print(f"[Instrumental function] took", time.time() - start_time, "seconds")
    
    p = np.array(p)
    print(p)
    if CMS_ch == 0:
        SC = 0
        for k in range(0, int((len(p[mod_p_len:])) / 3)):
            SC += p[mod_p_len + k * 3 + 2]**2
        print('Sum of INS:', SC)
        print('x0:', x0)
        print('MulCo:', MulCo)
    if CMS_ch == 1:
        print('G:', p[-1])
    
    # Calculate fitted spectra for plotting
    F = INSSS(A, p)
    JN_save = np.copy(JN)
    JN = JN * 4
    F2 = INSSS(A, p)
    JN = JN_save
    
    # Save instrumental function parameters
    INSp = p[mod_p_len:]
    
    if CMS_ch == 0:
        insexp_path = os.path.join(app.dir_path, 'parameters', 'INSexp.txt')
    else:
        insexp_path = os.path.join(app.dir_path, 'parameters', 'GCMS.txt')
    
    if CMS_ch == 0:
        with open(insexp_path, "w") as f:
            for i in range(0, len(INSp)):
                f.write(str(INSp[i]) + ' ')
        
        instrumental_int_path = os.path.join(app.dir_path, 'parameters', 'INSint.txt')
        with open(instrumental_int_path, "w") as f:
            f.write(str(MulCo) + ' ')
            f.write(str(x0) + ' ')
    else:
        with open(insexp_path, "w") as f:
            f.write(str('%.3f' % INSp[-1]))
    
    # Return results (plotting will be done in main thread)
    return {
        'p': p,
        'er': er,
        'hi2': hi2,
        'mod_p_len': mod_p_len,
        'x0': x0 if CMS_ch == 0 else None,
        'MulCo': MulCo if CMS_ch == 0 else None,
        'G': INSp[-1] if CMS_ch == 1 else None,
        'insexp_path': insexp_path,
        'mode': mode,
        'CMS_ch': CMS_ch,
        'A': A,
        'B': B,
        'F': F,
        'F2': F2,
        'file': os.path.basename(file)
    }
