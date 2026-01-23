"""
Fitting module for Mössbauer spectroscopy data analysis.
Handles spectrum fitting using minimization algorithms from minimi_lib.
"""

import numpy as np
import platform
import os
from functools import partial
import models as m5
import minimi_lib as mi
from constants import number_of_baseline_parameters, numco
from model_io import mod_len_def as mod_len_def_full
from models_positions import mod_pos
import builtins as bu



def mod_len_def(mod_name):
    """
    Returns the number of parameters for a given model type.
    
    Args:
        mod_name: String name of the model
        numco: Integer, used for 'Variables' model
        number_of_baseline_parameters: Integer, used for 'Nbaseline' model
    
    Returns:
        Integer number of parameters for that model
    """
    if mod_name == 'Singlet':
        return 4
    elif mod_name == 'Doublet':
        return 7
    elif mod_name == 'Sextet':
        return 11
    elif mod_name == 'Sextet(rough)':
        return 14
    elif mod_name == 'Relax_2S':
        return 11
    elif mod_name == 'Average_H':
        return 11
    elif mod_name == 'Relax_MS':
        return 9
    elif mod_name == 'ASM':
        return 12
    elif mod_name == 'Hamilton_mc':
        return 11
    elif mod_name == 'Hamilton_pc':
        return 9
    elif mod_name == 'Variables':
        return numco
    elif mod_name == 'MDGD':
        return 14
    elif mod_name == 'Nbaseline':
        return number_of_baseline_parameters
    else:
        return 0


def read_model(app):
    """
    Read model configuration from the parameters table.
    
    Args:
        app: Main application object with parameters_table reference
    
    Returns:
        tuple: (model, numro) where:
            - model: list of model names (e.g., ['baseline', 'Voigt', 'Gauss'])
            - numro: total number of rows in the model
    """
    model = []
    numro = 0
    
    # Get model list from parameters table
    model_list = app.params_table.get_model_list()
    
    # Build model array
    for model_name in model_list:
        if model_name != 'baseline':
            model.append(model_name)
    
    # Count total rows (baseline + models)
    numro = 1  # baseline row
    
    for model_name in model:
        numro += 1  # model row itself
        model_len = mod_len_def(model_name)
        numro += model_len  # parameter rows
    
    return model, numro


def create_subspectra(app, model, Distri, Cor, p):
    """
    Create subspectra from full model by splitting into individual components.
    
    Args:
        app: Main application object
        model: List of model names
        Distri: Distribution expressions
        Cor: Correlation expressions
        p: Parameter array
    
    Returns:
        tuple: (Ps, Psm, Distri_t, Cor_t, Di, Co) where:
            - Ps: List of parameter arrays for each subspectrum
            - Psm: List of model lists for each subspectrum
            - Distri_t: Distribution expressions with substituted values
            - Cor_t: Correlation expressions with substituted values
            - Di: Distribution counter
            - Co: Correlation counter
    """
    
    Ps = []
    Psm = []
    Distri_t = []
    Cor_t = []
    Di = 0
    Co = 0
    
    V = 0
    for i in range(0, len(model)):
        # Create parameter array for this model (starts with baseline)
        ps = np.array([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
        Psm.append([model[i]])
        
        # Add model-specific parameters
        if model[i] != 'Distr' and model[i] != 'Corr' and model[i] != 'Nbaseline' and model[i] != 'Expr':
            for j in range(0, mod_len_def(model[i])):
                ps = np.append(ps, p[number_of_baseline_parameters + V])
                V += 1
        
        # Append to list
        Ps.append(ps)
        
        # Handle special cases that modify previous subspectra
        if model[i] == 'Distr':
            # Delete the just-added entry and append params to previous one
            del Ps[-1]
            for j in range(0, 5):  # 5 parameters including expression placeholder
                Ps[-1] = np.append(Ps[-1], p[number_of_baseline_parameters + V])
                V += 1
            del Psm[-1]
            Psm[-1].append(model[i])
            
            # Substitute parameter values into distribution expression
            STR = Distri[Di] + str(' ')
            st_ = []
            en_ = []
            for k in range(0, len(STR) - 2):
                if STR[k] == 'p' and STR[k + 1] == '[':
                    st_.append(k)
                    for kk in range(k, len(STR)):
                        if STR[kk] == ']':
                            en_.append(kk)
                            break
            st_ = st_[::-1]
            en_ = en_[::-1]
            for k in range(0, len(st_)):
                STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
            Distri_t.append(STR)
            Di += 1
        
        if model[i] == 'Corr':
            # Delete the just-added entry and append params to previous one
            del Ps[-1]
            for j in range(1, 3):  # 2 parameters
                Ps[-1] = np.append(Ps[-1], p[number_of_baseline_parameters + V])
                V += 1
            del Psm[-1]
            Psm[-1].append(model[i])
            
            # Substitute parameter values into correlation expression
            STR = Cor[Co] + str(' ')
            st_ = []
            en_ = []
            for k in range(0, len(STR) - 2):
                if STR[k] == 'p' and STR[k + 1] == '[':
                    st_.append(k)
                    for kk in range(k, len(STR)):
                        if STR[kk] == ']':
                            en_.append(kk)
                            break
            st_ = st_[::-1]
            en_ = en_[::-1]
            for k in range(0, len(st_)):
                STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
            Cor_t.append(STR)
            Co += 1
    
    return (Ps, Psm, Distri_t, Cor_t, Di, Co)


def fit_single_spectrum(app, spectrum_file, pool, background=None, sequence_params=None):
    """
    Fit single or multiple Mössbauer spectra (simultaneous fitting with Nbaseline).
    
    Args:
        app: Main application object
        spectrum_file: Path to spectrum file (or list of files for simultaneous fitting)
        pool: Multiprocessing pool for parallel computation
        background: Optional background value (Ns) to override parameter table (for sequential fitting)
        sequence_params: Optional parameter array to use instead of reading from table (for sequential fitting)
    
    Returns:
        dict with keys:
            - 'success': bool
            - 'parameters': fitted parameter array
            - 'errors': parameter error array
            - 'chi2': chi-squared value
            - 'correlation_matrix': correlation matrix
            - 'model': model list
            - 'message': status message
            - 'is_simultaneous': bool (True if Nbaseline fitting)
    """
    try:
        # Import spectrum reading function
        from spectrum_io import load_spectrum
        from model_io import read_model as read_model_full
        
        # Read model configuration using the full read_model function
        model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN = read_model_full(app)

        p = np.array(p, dtype=float)
        
        # Override parameters if sequence_params provided (for sequential fitting)
        if sequence_params is not None:
            p = np.array(sequence_params, dtype=float)
            print(f"[Fitting] Using sequence parameters (result mode)")
        
        # Override background (Ns, p[1]) if provided (for sequential fitting)
        if background is not None:
            p[0] = background
            print(f"[Fitting] Using provided background: Ns = {background}")
        
        p0 = np.copy(p)
        
        print(f"[Fitting] Read {len(p)} parameters from table")
        print(f"[Fitting] Parameters: {p}")
        print(f"[Fitting] Model: {model}")
        
        # Check for Nbaseline (simultaneous fitting)
        num_nbaseline = model.count('Nbaseline')
        is_simultaneous = num_nbaseline > 0
        
        if is_simultaneous:
            # Simultaneous fitting mode
            number_of_spectra = num_nbaseline + 1
            print(f"[Fitting] Simultaneous fitting mode: {number_of_spectra} spectra expected")
            
            # Get list of spectrum files from path_list
            spectrum_files = app.parse_process_path()
            
            if len(spectrum_files) != number_of_spectra:
                return {
                    'success': False,
                    'message': f'Model has {num_nbaseline} Nbaseline(s), expecting {number_of_spectra} spectra, but {len(spectrum_files)} files selected.\nPlease select exactly {number_of_spectra} spectrum files.'
                }
            
            # Load and concatenate all spectra
            A_list, B_list = [], []
            for i, spec_file in enumerate(spectrum_files):
                A_temp, B_temp = load_spectrum(app, spec_file)
                A_temp, B_temp = A_temp[0], B_temp[0]
                A_list.append(A_temp)
                B_list.append(B_temp)
                print(f"[Fitting] Spectrum {i+1} loaded: {len(A_temp)} points, file: {os.path.basename(spec_file)}")
            
            # Concatenate spectra
            A = np.concatenate(A_list)
            B = np.concatenate(B_list)
            
            print(f"[Fitting] Total concatenated spectrum: {len(A)} points")
            
        else:
            # Single spectrum fitting mode
            A, B = load_spectrum(app, spectrum_file)
            A, B = A[0], B[0]  # Unpack from list
            A_list, B_list = [A], [B]
            
            print(f"[Fitting] Single spectrum loaded: {len(A)} points")
            print(f"[Fitting] X range: {A[0]:.2f} to {A[-1]:.2f}")
            print(f"[Fitting] Y range: {B.min():.2f} to {B.max():.2f}")
        
        # Determine fitting method
        if app.MS_fit.isChecked():
            VVV = 1  # MS method
            experimental_method = 1
        elif app.SMS_fit.isChecked():
            VVV = 3  # SMS method
            experimental_method = 3
        else:
            return {
                'success': False,
                'message': 'No fitting method selected (MS or SMS)'
            }
        
        # Get instrumental function parameters
        JN = int(app.JN0)
        
        if VVV == 1:  # MS method
            # CMS method with MulCoCMS
            INS = float(app.L0.text())
            x0_val = 0.0
            MulCo_val = app.MulCoCMS
            
            # Calculate normalization integral
            pNorm = np.array([float(0)] * number_of_baseline_parameters)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCo_val, INS, [0], [0], Met=1)[0]
            print('Normalization integral equal to', Norm)
            
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCo_val, INS, Distri, Cor, Met=1, Norm=Norm)
        
        elif VVV == 3:  # SMS method
            # Standard method with experimental INS
            if platform.system() == 'Windows':
                realpath = str(app.dir_path) + str('\\\\INSexp.txt')
            else:
                realpath = str(app.dir_path) + str('/INSexp.txt')
            
            if not os.path.exists(realpath):
                return {
                    'success': False,
                    'message': f'Instrumental function file not found: {realpath}'
                }
            
            INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
            x0_val = app.x0
            MulCo_val = app.MulCo
            
            # Calculate normalization integral
            pNorm = np.array([float(0)] * number_of_baseline_parameters)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0_val, MulCo_val, INS, [0], [0])[0]
            print('Normalization integral equal to', Norm)
            
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, x0_val, MulCo_val, INS, Distri, Cor, Norm=Norm)
        
        # Set up bounds and fixed parameters
        # For now, use unbounded optimization
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
                LenM = mod_len_def_full(model_name, include_special=True)
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
        
        
        # TODO: Read bounds and fix from parameters table
        # This would require additional UI elements
        
        # Add automatic fixes for constraints, distribution expressions, and expression models
        fix = np.concatenate((fix, con1.astype(int)), axis=0) if len(con1) > 0 else fix
        fix = np.concatenate((fix, DistriN.astype(int)), axis=0) if len(DistriN) > 0 else fix
        fix = np.concatenate((fix, NExpr.astype(int)), axis=0) if len(NExpr) > 0 else fix
        fix = np.unique(fix)

        # Set up constraints from con1, con2, con3
        if len(con1) > 0:
            confu = np.array([con1, con2, con3])
            # Apply constraints to initial parameters p0
            # Constrained parameters should start with the value of their source
            for i in range(len(con1)):
                constrained_idx = int(con1[i])
                source_idx = int(con2[i])
                multiplier = con3[i]
                p0[constrained_idx] = p0[source_idx] * multiplier
        else:
            confu = np.array([[-1], [-1], [-1]])
        
        # Perform minimization
        tau0 = 10 ** -3
        eps = 10 ** -6
        
        print('[Fitting] Starting minimization...')
        print(f'[Fitting] Initial parameters: {p}')
        print(f'[Fitting] Model: {model}')
        print(f'[Fitting] Fixed parameters (indices): {fix}')
        print(f'[Fitting] Constraints (confu): {confu}')
        
        p, er, hi2, covariance_matrix = mi.minimi_hi(
            func, A, B, p0,
            fix=fix,
            confu=confu,
            bounds=bounds,
            Expr=Expr,
            NExpr=NExpr,
            MI=20,
            MI2=10,
            nu0=2.618,
            tau0=tau0,
            eps=eps
        )
        if hi2 > 1.25:
            print('hi2 is too high let me try to continue')
            pO, erO, hi2O = p, er, hi2
            p, er, hi2, covariance_matrix = mi.minimi_hi(
                func,  A, B, p, 
                fix = fix,
                confu=confu,
                bounds = bounds,
                Expr = Expr,
                NExpr = NExpr,
                MI=20,
                MI2=10,
                nu0=2.618,
                tau0=tau0,
                eps=eps
            )
            if np.array_equal(p, pO) == True:
                if len(er) == 1:
                    p, er, hi2 = pO, erO, hi2O
                print('It was real end')


        print(f'[Fitting] Fitted parameters: {p}')
        print(f'[Fitting] Errors: {er}')
        print(f'[Fitting] Chi-squared: {hi2}')
        print(f'[Fitting] Covariance matrix shape: {covariance_matrix.shape}')
        
        # Calculate fitted spectrum for plotting
        SPC_f = func(A, p)
        
        # For simultaneous fitting, we need to separate results for each spectrum
        if is_simultaneous:
            # Separate model and parameters for each spectrum
            model_separate = []
            startM = 0
            for i in range(len(model)):
                if model[i] == 'Nbaseline':
                    model_separate.append(model[startM:i])
                    startM = i + 1
            model_separate.append(model[startM:])
            
            # Calculate parameter indices for each spectrum
            begining_spc = [0]
            start_cont_par = number_of_baseline_parameters
            param_names_full = app.params_table.get_parameter_names()
            for i in range(1, len(param_names_full)):
                param_names = param_names_full[i]
                if len(param_names) > 0 and param_names[0] == 'Ns':  # Start of new spectrum
                    begining_spc.append(start_cont_par)
                for j in range(len(param_names)):
                    if param_names[j] != '':
                        start_cont_par += 1
            
            print(f"[Fitting] Simultaneous - model_separate: {model_separate}")
            print(f"[Fitting] Simultaneous - begining_spc: {begining_spc}")
            
            # Substitute Distri and Cor parameter values ONCE using full model and full p
            # This ensures constrained parameters are correctly substituted
            Distri_substituted = np.copy(Distri)
            Cor_substituted = np.copy(Cor)
            if len(Distri) > 0 or len(Cor) > 0:
                _, _, Distri_substituted, Cor_substituted, _, _ = create_subspectra(app, model, Distri, Cor, p)
            
            # Calculate fitted spectrum for each section separately
            # Use model_separate (without Nbaseline) and parameter subset for each
            SPC_f_list = []
            for NumSpc in range(number_of_spectra):
                # Get parameters for this spectrum
                if NumSpc < len(begining_spc) - 1:
                    p_separate = p[begining_spc[NumSpc]:begining_spc[NumSpc + 1]]
                else:
                    p_separate = p[begining_spc[NumSpc]:]
                
                # Calculate spectrum using model_separate and p_separate
                # Use pre-substituted Distri and Cor
                model_for_spectrum = model_separate[NumSpc]
                if VVV == 1:  # MS method
                    SPC_f_separate = m5.TI(A_list[NumSpc], p_separate, model_for_spectrum, JN, pool, 0.0, MulCo_val, INS, Distri_substituted, Cor_substituted, Met=1, Norm=Norm)
                elif VVV == 3:  # SMS method
                    SPC_f_separate = m5.TI(A_list[NumSpc], p_separate, model_for_spectrum, JN, pool, x0_val, MulCo_val, INS, Distri_substituted, Cor_substituted, Norm=Norm)
                
                SPC_f_list.append(SPC_f_separate)
            
            # Now calculate subspectra for plotting
            # Substitute parameter values in Distri and Cor expressions ONCE with full model and full p
            Distri_save = np.copy(Distri)
            Cor_save = np.copy(Cor)
            if len(Distri) > 0 or len(Cor) > 0:
                _, _, Distri_substituted, Cor_substituted, _, _ = create_subspectra(app, model, Distri, Cor, p)
                Distri = Distri_substituted
                Cor = Cor_substituted
            
            FS_list = []
            FS_pos_list = []
            Distri_work = np.copy(Distri)
            Cor_work = np.copy(Cor)
            
            for NumSpc in range(number_of_spectra):
                # Get parameters for this spectrum
                if NumSpc < len(begining_spc) - 1:
                    p_separate = p[begining_spc[NumSpc]:begining_spc[NumSpc + 1]]
                else:
                    p_separate = p[begining_spc[NumSpc]:]
                
                # Calculate subspectra
                Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(app, model_separate[NumSpc], Distri_work, Cor_work, p_separate)
                
                FS = []
                FS_pos = []
                for i in range(len(Ps)):
                    CoSt = sum([Psm[j].count('Corr') for j in range(i)])
                    DiSt = sum([Psm[j].count('Distr') for j in range(i)])
                    CoEn = CoSt + Psm[i].count('Corr')
                    DiEn = DiSt + Psm[i].count('Distr')
                    
                    if VVV == 1:  # MS method
                        subspectrum = m5.TI(A_list[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCo_val, INS, 
                                           Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=1, Norm=Norm)
                        positions = mod_pos(Ps[i], Psm[i], INS, Met=1)
                    elif VVV == 3:  # SMS method
                        subspectrum = m5.TI(A_list[NumSpc], Ps[i], Psm[i], JN, pool, x0_val, MulCo_val, INS,
                                           Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Norm=Norm)
                        positions = mod_pos(Ps[i], Psm[i], INS)
                    
                    FS.append(subspectrum)
                    FS_pos.append(positions)
                
                FS_list.append(FS)
                FS_pos_list.append(FS_pos)
                
                # Update Distri and Cor for next spectrum
                Distri_work = Distri_work[Di:]
                Cor_work = Cor_work[Co:]
            
            return {
                'success': True,
                'parameters': p,
                'errors': er,
                'chi2': hi2,
                'covariance_matrix': covariance_matrix,
                'fix': fix,
                'model': model,
                'message': f'Simultaneous fit completed successfully. χ² = {hi2:.3f}',
                # Plotting data for simultaneous fitting
                'is_simultaneous': True,
                'A_list': A_list,  # List of A arrays
                'B_list': B_list,  # List of B arrays
                'SPC_f_list': SPC_f_list,  # List of fitted spectra
                'FS_list': FS_list,  # List of subspectra lists
                'FS_pos_list': FS_pos_list,  # List of position lists
                'model_separate': model_separate,  # Separated models
                'begining_spc': begining_spc,  # Parameter indices
                'spectrum_files': spectrum_files
            }
        
        else:
            # Single spectrum - original code
            # Calculate subspectra for plotting
            Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(app, model, Distri, Cor, p)
            
            # Calculate each subspectrum and its positions
            FS = []
            FS_pos = []
            for i in range(len(Ps)):
                CoSt = sum([Psm[j].count('Corr') for j in range(i)])
                DiSt = sum([Psm[j].count('Distr') for j in range(i)])
                CoEn = CoSt + Psm[i].count('Corr')
                DiEn = DiSt + Psm[i].count('Distr')
                print(Ps[i], Psm[i])
                if VVV == 1:  # MS method
                    subspectrum = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCo_val, INS, 
                                       Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=1, Norm=Norm)
                    # Calculate positions for this subspectrum
                    positions = mod_pos(Ps[i], Psm[i], INS, Met=1)
                elif VVV == 3:  # SMS method
                    subspectrum = m5.TI(A, Ps[i], Psm[i], JN, pool, x0_val, MulCo_val, INS,
                                       Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Norm=Norm)
                    # Calculate positions for this subspectrum
                    positions = mod_pos(Ps[i], Psm[i], INS)
                FS.append(subspectrum)
                FS_pos.append(positions)
            
            return {
                'success': True,
                'parameters': p,
                'errors': er,
                'chi2': hi2,
                'covariance_matrix': covariance_matrix,
                'fix': fix,
                'model': model,
                'message': f'Fit completed successfully. χ² = {hi2:.3f}',
                # Plotting data
                'is_simultaneous': False,
                'A': A,
                'B': B,
                'SPC_f': func(A, p),  # Calculate fitted spectrum
                'FS': FS,
                'FS_pos': FS_pos,
                'spectrum_file': spectrum_file
            }
    
    except Exception as e:
        import traceback
        return {
            'success': False,
            'message': f'Fitting failed: {str(e)}\n{traceback.format_exc()}'
        }


def determine_fitting_mode(app, spectrum_files):
    """
    Determine the fitting mode based on model and number of spectra.
    
    Args:
        app: Main application object
        spectrum_files: List of spectrum file paths
    
    Returns:
        str: 'single', 'simultaneous', or 'sequential'
    """
    if not spectrum_files:
        return 'single'
    
    # Get model list
    model_list = app.params_table.get_model_list()
    
    # Check for Nbaseline (simultaneous fitting)
    has_nbaseline = bu.any('Nbaseline' in model for model in model_list)
    num_spectra = len(spectrum_files)
    
    if has_nbaseline:
        return 'simultaneous'
    elif num_spectra > 1:
        return 'sequential'
    else:
        return 'single'
