"""
Fitting module for Mössbauer spectroscopy data analysis.
Handles spectrum fitting using minimization algorithms from minimi_lib.

This module is also the home of the model-decomposition helpers shared by the
fit and by "Show model" (``syncmoss_main.ShowModelThread``):
``split_model_sections``, ``create_subspectra`` and
``compute_component_curves``. Keeping one implementation guarantees that the
model preview and the fit result are computed identically.
"""

import os
import traceback
import numpy as np
import syncmoss.models as m5
import syncmoss.minimi_lib as mi
from syncmoss import exclusion_regions
from syncmoss.constants import number_of_baseline_parameters, numco
from syncmoss.model_io import mod_len_def, read_model as read_model_full, read_bounds_and_fix
from syncmoss.models_positions import mod_pos
from syncmoss.constants import SMS_POL_DEFAULT
from syncmoss.spectrum_io import load_spectrum
from syncmoss.spectrum_parameters import first_spectrum_parameters, table_uses_names
from syncmoss.instrumental_io import (
    resolve_instrumental_for_file,
    compute_norm,
    same_method_params,
    hires_model_diff,
)


def split_model_sections(model):
    """Split a flat model list into per-spectrum sections at 'Nbaseline' markers.

    A model for N spectra contains N-1 'Nbaseline' entries; the returned list
    always has ``model.count('Nbaseline') + 1`` sections (a model without
    Nbaseline yields ``[model]``). The markers themselves are not included.
    """
    sections = []
    start = 0
    for i, name in enumerate(model):
        if name == 'Nbaseline':
            sections.append(model[start:i])
            start = i + 1
    sections.append(model[start:])
    return sections


def _substitute_p_refs(expr_text, p):
    """Replace every literal ``p[<idx>]`` reference in a Distr/Corr expression
    with its current numeric value.

    Needed because the expression is later re-evaluated inside ``models.TImod``
    where ``p`` is a *component-local* parameter slice — the global indices the
    user typed would resolve against the wrong array there. Indices refer to
    the parameter array passed here (the full flat array in all callers).
    """
    STR = str(expr_text) + ' '
    starts = []
    ends = []
    for k in range(len(STR) - 2):
        if STR[k] == 'p' and STR[k + 1] == '[':
            starts.append(k)
            for kk in range(k, len(STR)):
                if STR[kk] == ']':
                    ends.append(kk)
                    break
    # Replace right-to-left so earlier offsets stay valid.
    for k in range(len(starts) - 1, -1, -1):
        STR = STR[:starts[k]] + str(eval(STR[starts[k]:ends[k] + 1])) + STR[ends[k] + 1:]
    return STR


# --- Recon (distribution reconstruction) fit support ------------------------
# The Num reconstruction weights of every 'Recon' model are the free fit values
# of the distribution shape. They occupy a single fixed placeholder slot in the
# canonical p (so counting/links/expressions never shift), and their values live
# in the parallel Recon list. For the fit they are EXPANDED onto the tail of a
# scalar working vector the minimiser varies, and their smoothness is enforced by
# Tikhonov "restrictions" appended as regularization pseudo-observation rows (see
# minimi_lib.minimi_hi's n_reg). The D_dif/D_dif2 knobs in [0,1] map to an
# internal penalty weight lambda = D/(1-D+eps): finite everywhere (D=1 reachable),
# 0 when D=0 (free distribution), ~1e6 when D=1 (effectively forces a flat /
# straight-line channel density). lambda is scaled by REG_SCALE below so the same
# D behaves consistently regardless of the spectrum's count level.
_RECON_EPS = 1e-6


def _recon_lambda(d):
    """Map a regularization knob D in [0,1] to a finite internal penalty weight."""
    d = min(max(float(d), 0.0), 1.0)
    return d / (1.0 - d + _RECON_EPS)


def recon_fit_layout(model, p, share=None):
    """Locate every 'Recon' block in the flat p.

    Returns ``(infos, n_weights)`` where each info is a dict with the block's
    ``off`` (flat index of its first slot), ``num`` (grid size), ``d1``/``d2``
    (the D_dif/D_dif2 knobs), ``re`` (its index into the parallel Recon list) and
    ``wstart`` (offset of its weights within the appended weight tail). Walks the
    model with mod_len_def, so it is correct across Nbaseline sections.

    *share* (optional) gives, for every Recon block in model order, the block
    whose weights it uses -- itself when it has its own. The simultaneous
    one-model fit makes every spectrum's copy of a Recon use the first
    spectrum's weights. Only the blocks with weights of their own get an info
    (so weights in the tail and smoothness rows); each lists in ``followers``
    the Recon-list indices of the blocks that use its weights.
    """
    infos = []
    owners = {}
    V = number_of_baseline_parameters
    re = 0
    wstart = 0
    for name in model:
        if name == 'Recon':
            owner = re if share is None else int(share[re])
            if owner == re:
                num = max(1, int(round(float(p[V + 3]))))
                owners[re] = {'off': int(V), 'num': num,
                              'd1': float(p[V + 4]), 'd2': float(p[V + 5]),
                              're': re, 'wstart': wstart, 'followers': []}
                infos.append(owners[re])
                wstart += num
            else:
                owners[owner]['followers'].append(re)
            re += 1
        V += mod_len_def(name, include_special=True)
    return infos, wstart


def _recon_penalty_length(infos):
    """Number of regularization pseudo-observation rows the infos produce."""
    n = 0
    for info in infos:
        num = info['num']
        n += max(0, num - 1)      # first-difference rows
        n += max(0, num - 2)      # second-difference rows
    return n


def _recon_penalty_rows(weight_tail, infos, reg_scale):
    """Build the stacked √λ·(finite difference of the normalized density) rows.

    Operates on the CHANNEL density g = w/Σw (scale-invariant) so D behaves the
    same regardless of the weight normalization; ``reg_scale`` makes the penalty
    commensurate with the data sum-of-squares. Zero λ (D=0) yields zero rows that
    leave the fit unconstrained.
    """
    rows = []
    for info in infos:
        num = info['num']
        w = np.asarray(weight_tail[info['wstart']:info['wstart'] + num], dtype=float)
        s = np.sum(w)
        g = w / s if s != 0 else np.full(num, 1.0 / num)
        if num >= 2:
            rows.append(np.sqrt(_recon_lambda(info['d1']) * reg_scale) * (g[1:] - g[:-1]))
        if num >= 3:
            rows.append(np.sqrt(_recon_lambda(info['d2']) * reg_scale) * (g[2:] - 2.0 * g[1:-1] + g[:-2]))
    return np.concatenate(rows) if rows else np.array([], dtype=float)


def create_subspectra(model, Distri, Cor, p):
    """
    Create subspectra from full model by splitting into individual components.

    Single source of truth for both the fit (this module) and "Show model"
    (``syncmoss_main.ShowModelThread``).

    Args:
        model: List of model names
        Distri: Distribution expressions
        Cor: Correlation expressions
        p: Flat parameter array (baseline first). ``p[i]`` references inside
           Distri/Cor texts are substituted against THIS array, so pass the
           full array (or an already-substituted Distri/Cor with a section
           slice of p — re-substitution is then a no-op).

    Returns:
        tuple: (Ps, Psm, Distri_t, Cor_t, Di, Co) where:
            - Ps: List of parameter arrays for each subspectrum
            - Psm: List of model lists for each subspectrum
            - Distri_t: Distribution expressions with substituted values
            - Cor_t: Correlation expressions with substituted values
            - Di: Number of Distr entries consumed
            - Co: Number of Corr entries consumed
    """
    Ps = []
    Psm = []
    Distri_t = []
    Cor_t = []
    Di = 0
    Co = 0
    # Models that consume parameter slots but never draw their own subspectrum.
    # 'Layer' is a boundary marker (0 parameters); 'Nbaseline' opens the next
    # spectrum section and carries that section's baseline parameters. Listing
    # them here only affects plotting decomposition, never save/load behavior.
    passthrough_non_spectral = {
        'Expression': 1,
        'Variables': numco,
        'Layer': 0,
        'Nbaseline': number_of_baseline_parameters,
    }

    V = number_of_baseline_parameters  # index of the next unread slot in p
    for model_name in model:
        if model_name in passthrough_non_spectral:
            V += passthrough_non_spectral[model_name]
            continue

        if model_name == 'Distr':
            # 5 slots (start, end, N points, target index, expression
            # placeholder) appended to the PREVIOUS component's parameter set.
            for _ in range(5):
                Ps[-1] = np.append(Ps[-1], p[V])
                V += 1
            Psm[-1].append(model_name)
            Distri_t.append(_substitute_p_refs(Distri[Di], p))
            Di += 1
            continue

        if model_name == 'Corr':
            # 2 slots (target index, expression placeholder), also appended to
            # the previous component (a Corr always follows a Distr).
            for _ in range(2):
                Ps[-1] = np.append(Ps[-1], p[V])
                V += 1
            Psm[-1].append(model_name)
            Cor_t.append(_substitute_p_refs(Cor[Co], p))
            Co += 1
            continue

        if model_name == 'Recon':
            # 7 slots (par, L, R, Num, D_dif, D_dif2, weight placeholder) appended
            # to the PREVIOUS component, exactly like Distr. No expression string —
            # the reconstruction weights travel in the parallel Recon list, sliced
            # per component in compute_component_curves via the 'Recon' marker count.
            for _ in range(7):
                Ps[-1] = np.append(Ps[-1], p[V])
                V += 1
            Psm[-1].append(model_name)
            continue

        # Ordinary spectral component: it is computed standalone on top of the
        # shared baseline parameters, followed by its own parameter slots.
        ps = np.array(p[:number_of_baseline_parameters], dtype=float)
        for _ in range(mod_len_def(model_name, include_special=False)):
            ps = np.append(ps, p[V])
            V += 1
        Ps.append(ps)
        Psm.append([model_name])

    return (Ps, Psm, Distri_t, Cor_t, Di, Co)


def compute_component_curves(A, Ps, Psm, Distri_t, Cor_t, JN, pool, method_params, pol, Recon=None):
    """Compute the plotted curve and line positions for each subspectrum.

    Takes the per-component parameter sets from :func:`create_subspectra` and
    runs the transmission integral / position calculation for each, slicing the
    (already substituted) Distri/Cor texts — and the ``Recon`` weight arrays —
    to the entries belonging to that component. Shared by the fit and by "Show
    model".

    Returns:
        tuple: (FS, FS_pos) — lists with one entry per component.
    """
    FS = []
    FS_pos = []
    DiEn = 0
    CoEn = 0
    ReEn = 0
    for i in range(len(Ps)):
        DiSt, CoSt, ReSt = DiEn, CoEn, ReEn
        DiEn += Psm[i].count('Distr')
        CoEn += Psm[i].count('Corr')
        ReEn += Psm[i].count('Recon')
        # The [0] placeholder is never read (the slice is empty exactly when
        # the component has no Distr/Corr/Recon); it just keeps TI's signature happy.
        distri_slice = Distri_t[DiSt:DiEn] if DiEn > DiSt else [0]
        cor_slice = Cor_t[CoSt:CoEn] if CoEn > CoSt else [0]
        recon_slice = list(Recon[ReSt:ReEn]) if (Recon is not None and ReEn > ReSt) else [0]

        FS.append(m5.TI(A, Ps[i], Psm[i], JN, pool,
                        method_params['x0'], method_params['MulCo'],
                        method_params['INS'], distri_slice, cor_slice,
                        Met=method_params['Met'], Norm=method_params['Norm'], pol=pol,
                        Recon=recon_slice))
        FS_pos.append(mod_pos(Ps[i], Psm[i], method_params['INS'], Met=method_params['Met']))
    return FS, FS_pos


def chi2_spread(n_points, n_free):
    """The 1-sigma spread a reduced chi-square has when the model is RIGHT.

    ``sqrt(2/dof)``. Without it a user cannot tell whether 1.3 is a bad fit or
    an ordinary fluctuation: on 40 degrees of freedom 1.3 is one sigma and
    means nothing, on 4000 it is nine sigma and means the model is wrong.
    """
    dof = max(int(n_points) - int(n_free), 1)
    return float(np.sqrt(2.0 / dof))


def section_starts(model):
    """Flat index at which each spectrum's parameters start in a model with
    Nbaseline sections: 0, then the slot of every Nbaseline (the next spectrum's
    baseline). Walks the model with mod_len_def, as TI does."""
    starts = [0]
    index = number_of_baseline_parameters
    for name in model:
        if name == 'Nbaseline':
            starts.append(index)
        index += mod_len_def(name, include_special=True)
    return starts


def read_fit_inputs(app, spectrum_parameters=None):
    """The model of the parameters table as fit_model takes it.

    The names N, N1, N2, ... are replaced by *spectrum_parameters*' values (see
    read_model). Keys: model, p, con1, con2, con3, Distri, Cor, Expr, NExpr,
    DistriN, Recon, ReconN (read_model's), bounds, fix (read_bounds_and_fix's)
    and exclusion_regions (the applied ones; () when none).
    """
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = \
        read_model_full(app, spectrum_parameters=spectrum_parameters)
    p = np.array(p, dtype=float)
    bounds, fix = read_bounds_and_fix(app, len(p))
    return {'model': model, 'p': p, 'con1': con1, 'con2': con2, 'con3': con3,
            'Distri': Distri, 'Cor': Cor, 'Expr': Expr, 'NExpr': NExpr,
            'DistriN': DistriN, 'Recon': Recon, 'ReconN': ReconN,
            'bounds': bounds, 'fix': fix,
            'exclusion_regions': app.exclusion_regions_in_use()}


def fixed_parameters(inputs):
    """The slots of *inputs* the fit does not vary: the user's fixes and every
    slot set by something else -- a link, a distribution expression, an
    Expression, a Recon weight-vector placeholder (the weights themselves are
    free and varied at the end of the working vector)."""
    fix = inputs['fix']
    for slots in (inputs['con1'], inputs['DistriN'], inputs['NExpr'], inputs['ReconN']):
        if len(slots) > 0:
            fix = np.concatenate((fix, np.asarray(slots).astype(int)), axis=0)
    return np.unique(fix)


def free_parameter_count(inputs):
    """How many values a fit of *inputs* varies: its free slots plus every free
    Recon weight."""
    _, n_weights = recon_fit_layout(inputs['model'], inputs['p'], share=inputs.get('recon_share'))
    return len(inputs['p']) + n_weights - len(fixed_parameters(inputs))


def too_few_points(n_points, n_free, regions=()):
    """Why a fit of *n_points* points with *n_free* free values cannot be
    started, or None when it can."""
    if n_points > n_free:
        return None
    left = "left after the exclusion regions" if regions else "in the spectrum"
    return f"{n_points} points are {left}, but {n_free} parameters are free"


def fit_single_spectrum(app, spectrum_file, pool, background=None, sequence_params=None,
                        spectrum_parameters=None):
    """
    Fit single or multiple Mössbauer spectra (simultaneous fitting with Nbaseline).

    Reads the model from the parameters table and fits it with fit_model.

    Args:
        app: Main application object
        spectrum_file: Path to spectrum file (or list of files for simultaneous fitting)
        pool: Multiprocessing pool for parallel computation
        background: Optional background value (Ns) to override parameter table (for sequential fitting)
        sequence_params: Optional parameter array to use instead of reading from table (for sequential fitting)
        spectrum_parameters: SpectrumParameters -- what N, N1, N2, ... stand for in
            this fit (for sequential fitting); by default the first spectrum
            of the path box

    Returns:
        dict with keys:
            - 'success': bool
            - 'parameters': fitted parameter array
            - 'errors': parameter error array
            - 'chi2': chi-squared value
            - 'chi2_spread': 1-sigma spread of chi2 for a correct model, sqrt(2/dof)
            - 'correlation_matrix': correlation matrix
            - 'model': model list
            - 'message': status message
            - 'is_simultaneous': bool (True if Nbaseline fitting)
    """
    try:
        # Read model configuration using the full read_model function, with the
        # names N, N1, N2, ... replaced by this spectrum's values
        if spectrum_parameters is None:
            spectrum_parameters = first_spectrum_parameters(app)
        inputs = read_fit_inputs(app, spectrum_parameters)
        model = inputs['model']
        # Recorded with the result (the _param.txt columns and the parameters
        # file) when the spectrum has values or the formulas use the names
        if 'Nbaseline' in model or not (spectrum_parameters.values or table_uses_names(app)):
            spectrum_parameters = None

        # Override parameters if sequence_params provided (for sequential fitting)
        if sequence_params is not None:
            inputs['p'] = np.array(sequence_params, dtype=float)
            print(f"[Fitting] Using sequence parameters (result mode)")
        
        # Override background (Ns, p[1]) if provided (for sequential fitting)
        if background is not None:
            inputs['p'][0] = background
            print(f"[Fitting] Using provided background: Ns = {background}")

        # An Nbaseline model is fitted to every spectrum of the path box
        spectrum_files = app.parse_process_path() if 'Nbaseline' in model else [spectrum_file]
    except m5.FitInterrupted:
        raise   # not a failure; the thread reports it (without a traceback)
    except Exception as e:
        return {
            'success': False,
            'message': f'Fitting failed: {str(e)}\n{traceback.format_exc()}'
        }

    result = fit_model(app, inputs, spectrum_files, pool)
    if result.get('success') and not result.get('is_simultaneous'):
        result['spectrum_parameters'] = spectrum_parameters   # N, N1, ... of this fit (or None)
    return result


def fit_model(app, inputs, spectrum_files, pool):
    """Fit *spectrum_files* with the model *inputs* -- the core of every fit.

    *inputs* is what read_fit_inputs makes of the parameters table, or a model
    built elsewhere: the simultaneous one-model fit expands the table's model
    over its spectra (one_model.expand) and adds 'recon_share' (see
    recon_fit_layout). A model with Nbaseline sections is fitted to the spectra
    together, one section per spectrum; otherwise *spectrum_files* holds the one
    spectrum. *inputs* is not changed, except that its Recon list receives the
    fitted weights.

    Returns the dict described in fit_single_spectrum; raises FitInterrupted
    when the fit is stopped.
    """
    try:
        instrumental_note = ''
        model = inputs['model']
        p = np.array(inputs['p'], dtype=float)
        con1, con2, con3 = inputs['con1'], inputs['con2'], inputs['con3']
        Distri, Cor, Expr, NExpr = inputs['Distri'], inputs['Cor'], inputs['Expr'], inputs['NExpr']
        DistriN, Recon, ReconN = inputs['DistriN'], inputs['Recon'], inputs['ReconN']
        
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
            
            if len(spectrum_files) != number_of_spectra:
                return {
                    'success': False,
                    'message': f'Model has {num_nbaseline} Nbaseline(s), expecting {number_of_spectra} spectra, but {len(spectrum_files)} files selected.\nPlease select exactly {number_of_spectra} spectrum files.'
                }
            
            # Load and concatenate all spectra
            A_list, B_list = [], []
            for i, spec_file in enumerate(spectrum_files):
                A_temp, B_temp = load_spectrum(
                    app, spec_file, calibration_path=app.calibration_path)
                A_temp, B_temp = A_temp[0], B_temp[0]
                A_list.append(A_temp)
                B_list.append(B_temp)
                print(f"[Fitting] Spectrum {i+1} loaded: {len(A_temp)} points, file: {os.path.basename(spec_file)}")
            
            # Concatenate spectra
            A = np.concatenate(A_list)
            B = np.concatenate(B_list)
            
            print(f"[Fitting] Total concatenated spectrum: {len(A)} points")
            
            # Where each spectrum ends in the joined data, for TI -- whichever
            # way its velocities run
            lengths = [len(a) for a in A_list]

        else:
            # Single spectrum fitting mode
            spectrum_file = spectrum_files[0]
            A, B = load_spectrum(app, spectrum_file,
                                 calibration_path=app.calibration_path)
            A, B = A[0], B[0]  # Unpack from list
            A_list, B_list = [A], [B]
            lengths = None

            print(f"[Fitting] Single spectrum loaded: {len(A)} points")
            print(f"[Fitting] X range: {A[0]:.2f} to {A[-1]:.2f}")
            print(f"[Fitting] Y range: {B.min():.2f} to {B.max():.2f}")

        # Exclusion regions: their points are left out of the fit. The model is
        # computed velocity by velocity (tests/test_model_pointwise.py), so the
        # fit takes just the points that are left (A_data, B_data); the curves
        # drawn after it are computed on every point.
        regions = tuple(inputs.get('exclusion_regions') or ())
        if regions:
            keeps = [exclusion_regions.kept(a, regions) for a in A_list]
            emptied = [os.path.basename(f) for f, k in zip(spectrum_files, keeps) if not k.any()]
            if emptied:
                return {
                    'success': False,
                    'message': f"every point of {', '.join(emptied)} is in an exclusion region"
                }
            A_data = np.concatenate([a[k] for a, k in zip(A_list, keeps)])
            B_data = np.concatenate([b[k] for b, k in zip(B_list, keeps)])
            lengths_data = [int(k.sum()) for k in keeps] if is_simultaneous else None
            print(f"[Fitting] Exclusion regions {exclusion_regions.format_regions(regions)}: "
                  f"{len(A_data)} of {len(A)} points are fitted")
        else:
            A_data, B_data, lengths_data = A, B, lengths

        # A method checkbox must be selected (it is the fallback when a spectrum
        # carries no .dat instrumental metadata)
        if not app.MS_fit.isChecked() and not app.SMS_fit.isChecked():
            return {
                'success': False,
                'message': 'No fitting method selected (MS or SMS)'
            }

        # Resolve instrumental parameters per spectrum. Each spectrum may be CMS
        # or SMS depending on its own .dat metadata (#@GCMS vs #@INSexp/#@INSint)
        # when the "use instrumental function from .dat file" option is enabled;
        # otherwise the UI-selected method with the internal values is used.
        JN = int(app.JN0)
        pol = float(getattr(app, 'SMS_pol', SMS_POL_DEFAULT))  # SMS beam polarization degree
        use_dat_metadata = bool(getattr(app, 'use_dat_instrumental_metadata', True))
        files_for_ins = list(spectrum_files) if is_simultaneous else [spectrum_file]

        method_params_list = []
        for ins_file in files_for_ins:
            mp_i = resolve_instrumental_for_file(app, ins_file, use_dat_metadata=use_dat_metadata)
            mp_i['Norm'] = compute_norm(pool, JN, mp_i)
            print('Normalization integral equal to', mp_i['Norm'])
            method_params_list.append(mp_i)
        mp0 = method_params_list[0]

        note_lines = [mp_i['note'] for mp_i in method_params_list]
        if len({mp_i['method'] for mp_i in method_params_list}) > 1:
            note_lines.insert(0, "Mixed-method simultaneous fit: CMS and SMS spectra are fitted together.")
        instrumental_note = '\n'.join(note_lines)
        print(f"[Fitting] {instrumental_note}")

        if is_simultaneous:
            # Split model at Nbaseline boundaries
            model_separate = split_model_sections(model)

            # Calculate parameter indices for each spectrum
            begining_spc = section_starts(model)

            print(f"[Fitting] Simultaneous - model_separate: {model_separate}")
            print(f"[Fitting] Simultaneous - begining_spc: {begining_spc}")

            # Per-section slices of the Distri/Cor/Recon lists (used after the
            # fit to rebuild each section's sub-spectra for plotting)
            distr_bounds = np.cumsum([0] + [ms.count('Distr') for ms in model_separate])
            corr_bounds = np.cumsum([0] + [ms.count('Corr') for ms in model_separate])
            recon_bounds = np.cumsum([0] + [ms.count('Recon') for ms in model_separate])

            def section_parameters(p_full, idx):
                if idx < len(begining_spc) - 1:
                    return p_full[begining_spc[idx]:begining_spc[idx + 1]]
                return p_full[begining_spc[idx]:]

        uniform_method = all(same_method_params(mp0, mp_i) for mp_i in method_params_list[1:])

        if not is_simultaneous or uniform_method:
            # Uniform instrumental settings: a single TI call over the whole model
            # (TI splits Nbaseline sections internally, at the spectra's lengths)
            # — the original code path.
            def func(x, p, recon=Recon, lengths=lengths):
                return m5.TI(x, p, model, JN, pool, mp0['x0'], mp0['MulCo'], mp0['INS'],
                             Distri, Cor, Met=mp0['Met'], Norm=mp0['Norm'], pol=pol, Recon=recon,
                             lengths=lengths)
        else:
            # Dedicated per-section instrumental parameters (e.g. mixing CMS and
            # SMS): TI receives one value per section as lists. The full model and
            # full p are still passed, so cross-spectrum links and Distr/Cor p[i]
            # references resolve exactly as in the uniform path — no per-section
            # bookkeeping leaks into this module.
            x0_list = [mp_i['x0'] for mp_i in method_params_list]
            mulco_list = [mp_i['MulCo'] for mp_i in method_params_list]
            ins_list = [mp_i['INS'] for mp_i in method_params_list]
            met_list = [mp_i['Met'] for mp_i in method_params_list]
            norm_list = [mp_i['Norm'] for mp_i in method_params_list]

            def func(x, p, recon=Recon, lengths=lengths):
                return m5.TI(x, p, model, JN, pool, x0_list, mulco_list, ins_list,
                             Distri, Cor, Met=met_list, Norm=norm_list, pol=pol, Recon=recon,
                             lengths=lengths)

        # The model at the fitted points (each spectrum keeps its own number of
        # them when exclusion regions are applied)
        if regions:
            def func_data(x, p, recon=Recon):
                return func(x, p, recon, lengths_data)
        else:
            func_data = func

        # Box bounds (from the table: read_fit_inputs), and the user's fixes
        # with the automatic ones: constraints, distribution expressions,
        # expression models and Recon weight-vector placeholder slots (the real
        # weights are free and appended to the working vector below).
        bounds = inputs['bounds']
        fix = fixed_parameters(inputs)

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

        # Start from the model Show model draws: evaluate every Expression on p0
        # and carry the parameters linked to it along, in the order minimi_hi
        # uses on every trial point. read_model leaves 0 in an Expression's slot
        # and minimi_hi takes p0 as given, so a parameter linked to an Expression
        # used to start at 0; when 0 fitted as well as the Expression's value the
        # fit never moved and returned the parameter as 0.
        for e_i in range(len(Expr)):
            p0[NExpr[e_i]] = mi._eval_expr(str(Expr[e_i]), p0)
            for c_i in np.where(confu[1] == NExpr[e_i])[0]:
                p0[int(confu[0][c_i])] = p0[int(confu[1][c_i])] * confu[2][c_i]

        # --- Recon (distribution reconstruction) fit expansion --------------
        # Every 'Recon' contributes Num FREE weights. They are appended to the TAIL
        # of the working vector the minimiser varies (the canonical head keeps its
        # fixed 7 Recon slots, so links / p[i] expressions / counters are untouched)
        # with a >=0 lower bound, and their smoothness "restrictions" enter as
        # Tikhonov pseudo-observation rows (n_reg) that participate in the LM step
        # but not in the reported chi-square / covariance. When there is no Recon,
        # func_fit/A_fit/B_fit/p0_fit collapse to the original inputs (n_reg == 0),
        # so this path is byte-identical to the previous behaviour.
        recon_infos, n_weights = recon_fit_layout(model, p, share=inputs.get('recon_share'))
        base_len = len(p0)
        if recon_infos:
            def _rebuild_recon(pw):
                rl = list(Recon)
                for info in recon_infos:
                    s = base_len + info['wstart']
                    weights = np.asarray(pw[s:s + info['num']], dtype=float)
                    for re in [info['re']] + info['followers']:
                        rl[re] = weights
                return rl

            init_w = np.concatenate([
                (np.asarray(Recon[info['re']], dtype=float).ravel()
                 if np.asarray(Recon[info['re']]).ravel().size == info['num']
                 else np.full(info['num'], 1.0 / info['num']))
                for info in recon_infos])
            p0_fit = np.concatenate([p0, init_w])
            bounds_fit = np.concatenate(
                (bounds, np.array([[0.0] * n_weights, [np.inf] * n_weights])), axis=1)
            n_reg = _recon_penalty_length(recon_infos)
            reg_scale = float(np.sum((np.asarray(B_data, dtype=float) - np.mean(B_data)) ** 2)) or 1.0
            A_fit = np.concatenate([np.asarray(A_data, dtype=float), np.zeros(n_reg)])
            B_fit = np.concatenate([np.asarray(B_data, dtype=float), np.zeros(n_reg)])

            def func_fit(_x, pw):
                spec = np.asarray(func_data(A_data, pw[:base_len], _rebuild_recon(pw)), dtype=float)
                pen = _recon_penalty_rows(pw[base_len:], recon_infos, reg_scale)
                return np.concatenate([spec, pen])
        else:
            func_fit, A_fit, B_fit, p0_fit, bounds_fit, n_reg = func_data, A_data, B_data, p0, bounds, 0

        # Not more free values than points: the fit is not started
        n_free = len(p0_fit) - len(fix)
        refusal = too_few_points(len(B_data), n_free, regions)
        if refusal:
            return {'success': False, 'message': f"{refusal}: the fit was not started"}

        # Perform minimization
        tau0 = 10 ** -3
        eps = 10 ** -6

        print('[Fitting] Starting minimization...')
        print(f'[Fitting] Initial parameters: {p0}')
        print(f'[Fitting] Model: {model}')
        print(f'[Fitting] Fixed parameters (indices): {fix}')
        print(f'[Fitting] Constraints (confu): {confu}')

        pfit, er, hi2, covariance_matrix = mi.minimi_hi(
            func_fit, A_fit, B_fit, p0_fit,
            fix=fix,
            confu=confu,
            bounds=bounds_fit,
            Expr=Expr,
            NExpr=NExpr,
            MI=20,
            MI2=10,
            nu0=2.618,
            tau0=tau0,
            eps=eps,
            n_reg=n_reg
        )
        if hi2 > 1.25:
            print('hi2 is too high let me try to continue')
            pO, erO, hi2O = pfit, er, hi2
            pfit, er, hi2, covariance_matrix = mi.minimi_hi(
                func_fit,  A_fit, B_fit, pfit,
                fix = fix,
                confu=confu,
                bounds = bounds_fit,
                Expr = Expr,
                NExpr = NExpr,
                MI=20,
                MI2=10,
                nu0=2.618,
                tau0=tau0,
                eps=eps,
                n_reg=n_reg
            )
            if np.array_equal(pfit, pO) == True:
                if len(er) == 1:
                    pfit, er, hi2 = pO, erO, hi2O
                print('It was real end')

        # Map the working vector back to the canonical p (head) and write the
        # fitted reconstruction weights into the parallel Recon list IN PLACE, so
        # func()'s default `recon=Recon` (and the plotting/result path below) use
        # the fitted distribution. When there is no Recon this is a plain identity.
        p = np.asarray(pfit[:base_len], dtype=float)
        er = np.asarray(er[:base_len], dtype=float) if np.ndim(er) and len(er) >= base_len else er
        for info in recon_infos:
            s = base_len + info['wstart']
            for re in [info['re']] + info['followers']:
                Recon[re] = np.asarray(pfit[s:s + info['num']], dtype=float)

        # Degrees of freedom as the minimiser counts them: the fitted data rows
        # (not the Recon penalty rows, not the excluded points) minus the free
        # entries of the working vector (the Recon weights are free). Parameters
        # the minimiser pins to a bound on its own are not subtracted -- a shift
        # far below the spread itself.
        spread = chi2_spread(len(B_data), n_free)

        print(f'[Fitting] Fitted parameters: {p}')
        print(f'[Fitting] Errors: {er}')
        print(f'[Fitting] Chi-squared: {hi2} ± {spread:.4f}')
        print(f'[Fitting] Covariance matrix shape: {covariance_matrix.shape}')

        # Calculate fitted spectrum for plotting
        SPC_f = func(A, p)
        
        # For simultaneous fitting, we need to separate results for each spectrum
        # (model_separate and begining_spc were computed before the minimization)
        if is_simultaneous:
            # Substitute p[i] references in Distri/Cor ONCE, using the full model
            # and the full fitted p, so constrained and cross-section references
            # resolve correctly; the per-section work below uses slices of the
            # substituted lists (re-substitution inside create_subspectra is then
            # a no-op, making the section-local p_separate safe).
            Distri_save = list(Distri)
            Cor_save = list(Cor)
            Distri_substituted = list(Distri)
            Cor_substituted = list(Cor)
            if len(Distri) > 0 or len(Cor) > 0:
                _, _, Distri_substituted, Cor_substituted, _, _ = create_subspectra(model, Distri, Cor, p)

            # Per section: fitted spectrum, convergence check and subspectra,
            # each with the instrumental parameters resolved for that section
            SPC_f_list = []
            hires_diff_list = []
            FS_list = []
            FS_pos_list = []
            for NumSpc in range(number_of_spectra):
                p_separate = section_parameters(p, NumSpc)
                mp_i = method_params_list[NumSpc]
                d_slice = list(Distri_substituted[distr_bounds[NumSpc]:distr_bounds[NumSpc + 1]])
                c_slice = list(Cor_substituted[corr_bounds[NumSpc]:corr_bounds[NumSpc + 1]])
                r_slice = list(Recon[recon_bounds[NumSpc]:recon_bounds[NumSpc + 1]])
                d_arg = d_slice if len(d_slice) > 0 else [0]
                c_arg = c_slice if len(c_slice) > 0 else [0]
                r_arg = r_slice if len(r_slice) > 0 else [0]
                SPC_f_separate = m5.TI(A_list[NumSpc], p_separate, model_separate[NumSpc], JN, pool,
                                       mp_i['x0'], mp_i['MulCo'], mp_i['INS'],
                                       d_arg, c_arg,
                                       Met=mp_i['Met'], Norm=mp_i['Norm'], pol=pol, Recon=r_arg)
                SPC_f_list.append(SPC_f_separate)

                # High-resolution convergence check for this section (cyan line)
                hires_diff_list.append(hires_model_diff(
                    pool, JN, A_list[NumSpc], p_separate, model_separate[NumSpc],
                    mp_i, SPC_f_separate, d_arg, c_arg, pol=pol, Recon=r_arg))

                Ps, Psm, Distri_t, Cor_t, _, _ = create_subspectra(
                    model_separate[NumSpc], d_slice, c_slice, p_separate)
                FS, FS_pos = compute_component_curves(
                    A_list[NumSpc], Ps, Psm, Distri_t, Cor_t, JN, pool, mp_i, pol, Recon=r_slice)
                FS_list.append(FS)
                FS_pos_list.append(FS_pos)

            return {
                'success': True,
                'parameters': p,
                'errors': er,
                'chi2': hi2,
                'chi2_spread': spread,
                'covariance_matrix': covariance_matrix,
                'fix': fix,
                'model': model,
                'message': f'Simultaneous fit completed successfully. χ² = {hi2:.3f}',
                # Plotting data for simultaneous fitting
                'is_simultaneous': True,
                'A_list': A_list,  # List of A arrays
                'B_list': B_list,  # List of B arrays
                'SPC_f_list': SPC_f_list,  # List of fitted spectra
                'hires_diff_list': hires_diff_list,  # Per-section 4x-integration convergence check
                'FS_list': FS_list,  # List of subspectra lists
                'FS_pos_list': FS_pos_list,  # List of position lists
                'model_separate': model_separate,  # Separated models
                'begining_spc': begining_spc,  # Parameter indices
                'spectrum_files': spectrum_files,
                'Distri': list(Distri_save),  # Distribution expressions (original)
                'Cor': list(Cor_save),  # Correlation expressions (original)
                'Distri_substituted': list(Distri_substituted),
                'Cor_substituted': list(Cor_substituted),
                'Recon': [np.asarray(w, dtype=float) for w in Recon],  # fitted reconstruction weights
                'instrumental_note': instrumental_note,
                'exclusion_regions': regions,  # their points were not fitted
            }

        else:
            # Single spectrum: decompose into subspectra and compute each curve
            Ps, Psm, Distri_t, Cor_t, _, _ = create_subspectra(model, Distri, Cor, p)
            FS, FS_pos = compute_component_curves(A, Ps, Psm, Distri_t, Cor_t, JN, pool, mp0, pol, Recon=Recon)

            # High-resolution convergence check (cyan line). Uses the same
            # instrumental settings and Distri/Cor as func(), so it matches SPC_f.
            hires_diff = hires_model_diff(pool, JN, A, p, model, mp0, SPC_f, Distri, Cor, pol=pol, Recon=Recon)

            return {
                'success': True,
                'parameters': p,
                'errors': er,
                'chi2': hi2,
                'chi2_spread': spread,
                'covariance_matrix': covariance_matrix,
                'fix': fix,
                'model': model,
                'message': f'Fit completed successfully. χ² = {hi2:.3f}',
                # Plotting data
                'is_simultaneous': False,
                'A': A,
                'B': B,
                'SPC_f': SPC_f,  # Fitted spectrum (already computed above)
                'hires_diff': hires_diff,  # 4x-integration convergence check
                'FS': FS,
                'FS_pos': FS_pos,
                'spectrum_file': spectrum_file,
                'Distri': list(Distri),  # Distribution expressions (original)
                'Cor': list(Cor),  # Correlation expressions (original)
                'Distri_substituted': list(Distri_t),  # Distribution expressions (substituted)
                'Cor_substituted': list(Cor_t),  # Correlation expressions (substituted)
                'Recon': [np.asarray(w, dtype=float) for w in Recon],  # fitted reconstruction weights
                'instrumental_note': instrumental_note,
                'exclusion_regions': regions,  # their points were not fitted
            }

    except m5.FitInterrupted:
        raise   # not a failure; the thread reports it (without a traceback)
    except Exception as e:
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
    has_nbaseline = any('Nbaseline' in model for model in model_list)
    num_spectra = len(spectrum_files)
    
    if has_nbaseline:
        return 'simultaneous'
    elif num_spectra > 1:
        return 'sequential'
    else:
        return 'single'
