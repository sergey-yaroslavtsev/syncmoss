"""
Instrumental function calculation and refinement module for SYNCMoss.
Handles the calculation, fitting, and refinement of instrumental functions.
"""
import os
import time
import numpy as np
from PySide6.QtWidgets import QMessageBox
from syncmoss.constants import (number_of_baseline_parameters, ALPHA_FE_FIELD,
                                NAT_WIDTH, SMS_POL_DEFAULT)
import syncmoss.models as m5
import syncmoss.minimi_lib as mi
import syncmoss.sms_theory as smst
from syncmoss.spectrum_io import load_spectrum, estimate_edge_background
from syncmoss.model_io import read_model, read_bounds_and_fix


# Built-in defaults used by "Reset to default values" in Instrumental function menu.
DEFAULT_INSEXP_TEXT = (
    "0.16472623639571624 0.11257879642037069 0.7019583961622452 "
    "0.08985576770449756 -0.05913725754557238 0.42733862867647215 "
    "-0.5403075049599122 0.012656364797966438 0.5697684674481738 "
)
DEFAULT_INSINT_TEXT = "2.448293819453453 0.03380920627191801 "

DAT_INS_EXP_PREFIX = '#@INSexp'
DAT_INS_INT_PREFIX = '#@INSint'
DAT_GCMS_PREFIX = '#@GCMS'
# The THEORETICAL (simulated 57FeBO3) SMS instrumental function. It is written
# IN ADDITION to #@INSexp/#@INSint, not instead of them: a converted .dat
# therefore still carries the conventional Gaussian-sum description (the best
# empirical stand-in for the same source, so any reader that does not know this
# marker degrades gracefully), and #@INSint still carries the integration
# constants.
DAT_INS_TH_PREFIX = '#@INSth'
# Read-only legacy spelling. Files written before the rename carry #@INSacc and
# must keep loading; nothing writes it any more.
DAT_INS_ACC_PREFIX_LEGACY = '#@INSacc'

# Global store of the theoretical instrumental function, next to INSexp/INSint.
# BOTH are kept: INSth.txt holds the simulated shape and INSexp.txt the Gaussian
# sum, so switching "Choose how to approximate instrumental function" switches
# the description without refitting, and switching back is free. Which one is
# USED is the setting, not which file exists -- see
# get_sms_instrumental_from_global_files.
INS_TH_FILE = 'INSth.txt'
INS_TH_FILE_LEGACY = 'INSacc.txt'
INS_EXP_FILE = 'INSexp.txt'

# Below this the instrumental function counts as already centred. Far under any
# velocity resolution (a natural linewidth is 0.098 mm/s); it exists only so a
# second calibration of an unchanged setup does not rewrite the file.
INS_RECENTRE_TOL = 1e-9

# Backwards-compatible aliases for callers outside this module.
DAT_INS_ACC_PREFIX = DAT_INS_TH_PREFIX
INS_ACC_FILE = INS_TH_FILE


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
    """Read #@INSexp / #@INSint / #@GCMS metadata from a .dat file header.

    #@INSexp + #@INSint mark a spectrum converted in SMS mode, #@GCMS marks a
    spectrum converted in CMS mode (single G value of the single-line absorber).
    """
    result = {
        'INS': None,
        'MulCo': None,
        'x0': None,
        'GCMS': None,
        'INSacc': None,
        'has_insexp': False,
        'has_insint': False,
        'has_gcms': False,
        'has_insacc': False,
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

                # #@INSth must be tested BEFORE #@INSexp: neither is a prefix of
                # the other, but keeping them adjacent makes that obvious.
                # #@INSacc is the pre-rename spelling and is still READ.
                th_prefix = next(
                    (p for p in (DAT_INS_TH_PREFIX, DAT_INS_ACC_PREFIX_LEGACY)
                     if stripped.startswith(p)), None)
                if th_prefix is not None:
                    parsed = _parse_float_sequence(
                        stripped[len(th_prefix):].strip())
                    if parsed.size > 0 and smst.ins_kind(parsed) != smst.KIND_GAUSS:
                        result['INSacc'] = parsed
                        result['has_insacc'] = True
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

                if stripped.startswith(DAT_GCMS_PREFIX):
                    payload = stripped[len(DAT_GCMS_PREFIX):].strip()
                    parsed = _parse_float_sequence(payload)
                    if parsed.size > 0:
                        result['GCMS'] = float(parsed[0])
                        result['has_gcms'] = True
                    continue

                if not stripped.startswith('#') and not stripped.startswith('<'):
                    break
    except Exception as e:
        print(f"[Instrumental function] Could not read DAT metadata from {file_path}: {e}")

    return result


def get_legacy_sms_instrumental(app):
    """The conventional (Gaussian-sum) instrumental function in INSexp.txt."""
    insexp_path = os.path.join(app.params_dir, 'INSexp.txt')
    return np.atleast_1d(np.array(
        np.genfromtxt(insexp_path, delimiter=' ', skip_footer=0), dtype=float))


def read_accurate_instrumental(app):
    """The theoretical instrumental function in INSth.txt, or None.

    None means "the conventional Gaussian-sum instrumental function is the
    current one" -- which is the shipping state, and the state the legacy search
    and "Reset to default values" restore.
    """
    path = os.path.join(app.params_dir, INS_TH_FILE)
    if not os.path.exists(path):
        # the pre-rename name, so an existing installation keeps working
        legacy = os.path.join(app.params_dir, INS_TH_FILE_LEGACY)
        if not os.path.exists(legacy):
            return None
        path = legacy
    try:
        INS = np.atleast_1d(np.array(
            np.genfromtxt(path, delimiter=' ', skip_footer=0), dtype=float))
    except Exception as e:
        print(f"[Instrumental function] Could not read {path}: {e}")
        return None
    if INS.size == 0 or smst.ins_kind(INS) == smst.KIND_GAUSS:
        print(f"[Instrumental function] {path} does not hold a theoretical "
              f"instrumental function; ignoring it")
        return None
    return INS


def write_accurate_instrumental(app, INS):
    """Store the theoretical instrumental function (or clear it with None)."""
    path = os.path.join(app.params_dir, INS_TH_FILE)
    if INS is None:
        # clear BOTH spellings, or a stale INSacc.txt would keep being read
        for p in (path, os.path.join(app.params_dir, INS_TH_FILE_LEGACY)):
            if os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"[Instrumental function] removed {p} "
                          f"(back to the conventional instrumental function)")
                except OSError as e:
                    print(f"[Instrumental function] could not remove {p}: {e}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in np.atleast_1d(INS):
            f.write(str(float(v)) + ' ')
    print(f"[Instrumental function] wrote {path}")


class FitInterrupted(Exception):
    """Raised inside the theory search when the user presses "! INTERRUPT !".

    Same name and role as the SYNCtime branch's, so the mechanism is portable.
    """


def search_cancelled(app):
    """True when the user has asked for the current search to stop.

    ``fit_cancel`` is a threading.Event on the main window; anything driving the
    search head-less (a test, a script) simply has none, and never cancels.
    """
    event = getattr(app, 'fit_cancel', None)
    return event is not None and event.is_set()


def recentre_instrumental_after_calibration(app):
    """Put the instrumental function's gravity centre at zero. Returns the
    shift removed, or 0.0 when there was nothing to do.

    THE OTHER HALF OF WHAT CALIBRATION DOES. Calibration relabels the velocity
    axis by ``INS_shift = ins_centroid(INS)`` (Calibration.py, step 11). That is
    a GAUGE transformation: for the transmission integral

        T(v) = int dE S(E) exp(-sigma(E + v))

    replacing S(E) by S(E - d) gives exactly T(v + d), so moving the source and
    moving the axis are the same operation -- verified numerically to 4e-6 of
    the dip depth. Which means the axis move is only correct if the SOURCE moves
    with it. Relabelling v' = v + c requires the model to become

        M(v') = TI(v' - c; INS) = TI(v'; INS shifted by -c)

    and shifting the stored INS by -c is precisely putting its gravity centre at
    zero, since centroid(INS) - c = 0.

    Doing only the axis half leaves every later fit off by c. That is small for
    the legacy Gaussian sum, whose first moment is +0.0008 mm/s by accident of
    how it was fitted, and NOT small for the simulated source, which is
    asymmetric by construction and sits at -0.0144 -- about 15 % of a natural
    linewidth, and exactly the shift a sextet needed opening to absorb.

    Doing both halves leaves the fit untouched, puts the alpha-Fe pattern's
    centre of gravity at zero in the picture, and makes the NEXT calibration
    compute INS_shift = 0, so the pair is a fixed point rather than something
    the user has to iterate.

    BOTH descriptions are re-centred, because the argument is about the gauge,
    not about which shape is used: the simulated one by its ``shift``, the
    Gaussian sum by moving every line position. The Gaussian sum's centroid is
    small by accident (+0.0008 mm/s in the shipped file, because it was fitted
    against an axis that had already absorbed it) -- that is luck, not a
    guarantee, and a freshly searched sum has no reason to be centred at all.

    CMS needs nothing: its instrumental function is a single Gaussian width, so
    its centroid is zero by construction.
    """
    def recentre(INS, write):
        centre = float(smst.ins_centroid(INS))
        # Already centred: leave the file alone rather than rewriting it for a
        # rounding residue. Re-calibrating an unchanged setup must be a no-op.
        if not np.isfinite(centre) or abs(centre) < INS_RECENTRE_TOL:
            return 0.0
        write(centre)
        print(f"[Instrumental function] re-centred after calibration: gravity "
              f"centre {centre:+.5f} -> 0 mm/s")
        return centre

    INS = read_accurate_instrumental(app)
    if INS is not None and smst.ins_kind(INS) == smst.KIND_PHYSICAL:
        def write_physical(centre):
            ph = smst.decode_physical(INS)
            ph['shift'] = float(ph['shift']) - centre
            write_accurate_instrumental(
                app, smst.encode_physical(**{k: ph[k] for k in smst.PHYS_FIELDS}))
        return recentre(INS, write_physical)

    gauss = get_legacy_sms_instrumental(app)
    if gauss is None or gauss.size < 3 or smst.ins_kind(gauss) != smst.KIND_GAUSS:
        return 0.0

    def write_gauss(centre):
        moved = np.array(gauss, dtype=float)
        # a sum of Gaussians is translated by moving every line position
        moved[1::3] -= centre
        path = os.path.join(app.params_dir, INS_EXP_FILE)
        with open(path, "w") as f:
            f.write(' '.join(str(float(v)) for v in moved) + ' ')
        print(f"[Instrumental function] wrote {path}")

    return recentre(gauss, write_gauss)


# Which description "Find / Refine Instr. func." uses, and which stored shape
# every fit reads, when the user has not chosen. 'gauss' is the empirical sum of
# Gaussians SYNCmoss has always shipped; 'theory' is the simulated 57FeBO3
# source. The Gaussian sum is the DEFAULT for now so that the simulated shape is
# something a user opts into while it settles -- switching costs nothing, since
# both descriptions are stored side by side (INSexp.txt and INSth.txt).
DEFAULT_INSTRUMENTAL_METHOD = 'gauss'


def instrumental_method(app):
    """'theory' or 'gauss' -- which description the user has selected.

    Set from Supp -> "Choose how to approximate instrumental function"; defaults
    to 'theory' for anything that does not carry the attribute (tests, scripts).
    """
    return getattr(app, 'instrumental_method', DEFAULT_INSTRUMENTAL_METHOD)


def get_sms_instrumental_from_global_files(app):
    """The SMS instrumental function currently in use, with its grid constants.

    WHICH of the two it is comes from the user's setting, not from which file
    happens to exist. ``INSth.txt`` holds the theoretical shape and
    ``INSexp.txt`` the conventional Gaussian sum; both are kept, so switching
    the setting switches the description with no refitting either way.

    This used to return the theoretical one whenever INSth.txt was present,
    which meant selecting "Set of Gaussians" changed only which SEARCH ran --
    the fit went on using the theoretical shape.

    Falls back to the other description, with a printed note, when the selected
    one has not been found yet. MulCo/x0 come from INSint.txt in both cases:
    they describe the transmission-integral grid, not the shape.
    """
    instrumental_int_path = os.path.join(app.params_dir, 'INSint.txt')
    MulCo, x0 = np.genfromtxt(instrumental_int_path, delimiter=' ', skip_footer=0)

    if instrumental_method(app) == 'gauss':
        INS = get_legacy_sms_instrumental(app)
    else:
        INS = read_accurate_instrumental(app)
        if INS is None:
            print(f"[Instrumental function] no theoretical shape stored in "
                  f"{INS_TH_FILE}; using the Gaussian sum from {INS_EXP_FILE}")
            INS = get_legacy_sms_instrumental(app)
    return np.array(INS, dtype=float), float(MulCo), float(x0)


def get_internal_gcms(app):
    """Current internal G value for CMS: the G input field, falling back to
    parameters/GCMS.txt, then to 0.1."""
    try:
        return float(app.GCMS_input.text())
    except Exception:
        pass
    try:
        gcms_path = os.path.join(app.params_dir, 'GCMS.txt')
        return float(np.genfromtxt(gcms_path, delimiter='\t'))
    except Exception:
        return 0.1


def resolve_instrumental_for_file(app, spectrum_file, use_dat_metadata=True, force_method=None):
    """
    Resolve the instrumental parameters (CMS or SMS) for one spectrum.

    The single source of truth for "what instrumental function does this spectrum
    use". When ``use_dat_metadata`` is enabled the spectrum's own .dat header
    decides the method: a #@GCMS line marks a CMS spectrum, #@INSexp + #@INSint
    mark an SMS spectrum (#@GCMS wins if a file carries both). Otherwise — or when
    the file has no usable metadata — the UI-selected method (CMS/SMS checkboxes)
    with the internal values is used: the G input field for CMS, the global
    INSexp/INSint files for SMS. This is what lets different spectra of one
    simultaneous fit use dedicated values, including mixing CMS and SMS spectra.

    ``force_method`` ('CMS' or 'SMS') locks the method instead of letting the file
    or the checkboxes choose it: only metadata of that method is honoured (the
    other method's metadata is ignored), and the internal fallback is that method.
    Used by the instrumental-function refinement, where the user explicitly
    refines one method and the reference file must not flip it.

    Returns:
        dict with keys:
            'method' : 'CMS' or 'SMS'
            'Met'    : models.TI Met argument (1 for CMS, 0 for SMS)
            'INS'    : float G (CMS) or instrumental-function array (SMS)
            'x0'     : float
            'MulCo'  : float
            'source' : 'file' or 'internal'
            'note'   : human-readable description for the log box
    """
    ui_method = force_method or (
        'CMS' if (getattr(app, 'MS_fit', None) is not None and app.MS_fit.isChecked()) else 'SMS'
    )
    name = os.path.basename(str(spectrum_file)) if spectrum_file else ''
    mulco_cms = float(getattr(app, 'MulCoCMS', 0.28))
    fallback_reason = None

    if use_dat_metadata and spectrum_file and str(spectrum_file).lower().endswith('.dat'):
        meta = parse_dat_instrumental_metadata(spectrum_file)
        has_sms = meta['INS'] is not None and meta['MulCo'] is not None and meta['x0'] is not None
        # File metadata may set the method, unless we are locked to the other one.
        if meta['has_gcms'] and force_method != 'SMS':
            g = float(meta['GCMS'])
            return {
                'method': 'CMS', 'Met': 1, 'INS': g, 'x0': 0.0,
                'MulCo': mulco_cms, 'source': 'file',
                'note': f"{name}: CMS — G = {g:g} from .dat metadata (#@GCMS)",
            }
        if has_sms and force_method != 'CMS':
            # A .dat written by this program carries BOTH descriptions of the
            # same source -- #@INSth (simulated) and #@INSexp (Gaussian sum) --
            # and #@INSint supplies the integration constants for either. WHICH
            # one is used is the user's setting, exactly as for the global
            # files; that is what makes switching free, and switching back free.
            #
            # #@INSth used to win unconditionally here, so selecting "Set of
            # Gaussians" changed nothing for any spectrum whose file carried a
            # theoretical shape -- the fit went on using it.
            #
            # The spectrum's OWN instrumental function is still what is used:
            # the setting chooses between the two the file carries, it never
            # replaces them with the global ones.
            if meta['has_insacc'] and instrumental_method(app) == 'theory':
                return {
                    'method': 'SMS', 'Met': 0, 'INS': meta['INSacc'],
                    'x0': float(meta['x0']), 'MulCo': float(meta['MulCo']),
                    'source': 'file',
                    'note': f"{name}: SMS — {smst.describe_ins(meta['INSacc'])} "
                            f"from .dat metadata (#@INSth)",
                }
            return {
                'method': 'SMS', 'Met': 0, 'INS': meta['INS'], 'x0': float(meta['x0']),
                'MulCo': float(meta['MulCo']), 'source': 'file',
                'note': f"{name}: SMS — instrumental function from .dat metadata (#@INSexp/#@INSint)",
            }
        # No usable metadata for the (possibly forced) method -> internal fallback.
        missing = []
        if force_method != 'CMS':
            if meta['INS'] is None:
                missing.append('#@INSexp')
            if meta['MulCo'] is None or meta['x0'] is None:
                missing.append('#@INSint')
        if force_method != 'SMS':
            missing.append('#@GCMS')
        fallback_reason = f"no usable .dat metadata ({', '.join(missing)})"
    elif not use_dat_metadata:
        fallback_reason = ".dat metadata use is disabled"
    elif spectrum_file:
        fallback_reason = "input is not a .dat file"

    suffix = f" ({fallback_reason})" if fallback_reason else ""
    if ui_method == 'CMS':
        g = get_internal_gcms(app)
        return {
            'method': 'CMS', 'Met': 1, 'INS': float(g), 'x0': 0.0,
            'MulCo': mulco_cms, 'source': 'internal',
            'note': f"{name or 'model'}: CMS — internal G = {g:g}{suffix}",
        }
    INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(app)
    which = (f"global {INS_TH_FILE} ({smst.describe_ins(INS_global)})"
             if smst.ins_kind(INS_global) != smst.KIND_GAUSS
             else "global INSexp/INSint files")
    return {
        'method': 'SMS', 'Met': 0, 'INS': INS_global, 'x0': float(x0_global),
        'MulCo': float(MulCo_global), 'source': 'internal',
        'note': f"{name or 'model'}: SMS — {which}{suffix}",
    }


def same_method_params(a, b):
    """True when two resolve_instrumental_for_file() results are numerically identical."""
    return (
        a['method'] == b['method']
        and a['Met'] == b['Met']
        and float(a['x0']) == float(b['x0'])
        and float(a['MulCo']) == float(b['MulCo'])
        and np.array_equal(np.atleast_1d(a['INS']), np.atleast_1d(b['INS']))
    )


def analyze_instrumental_methods(app, spectrum_files, use_dat_metadata):
    """Resolve the instrumental method of every spectrum and report the two
    situations the fit dialog warns about.

    Returns ``(overridden, nonuniform, resolved)``:
      * ``resolved``   : list of resolve_instrumental_for_file() results, in order.
      * ``overridden`` : list of ``(file, resolved)`` whose method comes from the
                         .dat metadata and differs from the UI-selected method
                         (i.e. the file overrides the CMS/SMS checkbox). Empty
                         when .dat metadata is disabled (everything is internal).
      * ``nonuniform`` : True when more than one spectrum is involved and they do
                         not all share identical instrumental parameters (a mix
                         of CMS and SMS, or the same method with different values).
    """
    ui_method = 'CMS' if (getattr(app, 'MS_fit', None) is not None and app.MS_fit.isChecked()) else 'SMS'
    resolved = [resolve_instrumental_for_file(app, f, use_dat_metadata=use_dat_metadata) for f in spectrum_files]
    overridden = [(f, r) for f, r in zip(spectrum_files, resolved)
                  if r['source'] == 'file' and r['method'] != ui_method]
    nonuniform = len(resolved) > 1 and not all(same_method_params(resolved[0], r) for r in resolved[1:])
    return overridden, nonuniform, resolved


def compute_norm(pool, JN, method_params):
    """Normalization integral for a resolved method (CMS or SMS)."""
    pNorm = np.array([float(0)] * number_of_baseline_parameters)
    pNorm[0] = 1
    return m5.TI(
        np.array([float(1000)]), pNorm, [], JN, pool,
        method_params['x0'], method_params['MulCo'], method_params['INS'],
        [0], [0], Met=method_params['Met'],
    )[0]


# How many times more integration points the high-resolution convergence check
# uses. Kept in sync with the "(re)find instrumental function" fit, which
# recomputes its result at JN*4 to draw the cyan F2-F difference line.
HIRES_INTEGRATION_FACTOR = 4


def hires_model_diff(pool, JN, A, p, model, method_params, SPC_f,
                     Distri=[0], Cor=[0], pol=SMS_POL_DEFAULT, Recon=[0]):
    """High-resolution convergence check: (model at JN*4) - (model at JN).

    ``SPC_f`` is the already-computed model at the displayed ``JN`` (so it is
    not recomputed here). The same model is recomputed with
    ``JN*HIRES_INTEGRATION_FACTOR`` integration points and its own
    normalization, and the difference is returned. Drawn as the cyan line on
    the spectrum plot: a visible wiggle means the numerical transmission
    integral is under-sampled at the current JN. Mirrors the F2-F line drawn
    by the instrumental-function fit.
    """
    JN_hi = int(JN) * HIRES_INTEGRATION_FACTOR
    norm_hi = compute_norm(pool, JN_hi, method_params)
    SPC_hi = m5.TI(
        A, p, model, JN_hi, pool,
        method_params['x0'], method_params['MulCo'], method_params['INS'],
        Distri, Cor, Met=method_params['Met'], Norm=norm_hi, pol=pol, Recon=Recon,
    )
    return SPC_hi - SPC_f


def build_dat_metadata_lines(app):
    """Instrumental header lines to embed into .dat files converted from RAW.

    CMS mode (the CMS checkbox is checked): a single #@GCMS line with the
    current G value. SMS mode: #@INSexp / #@INSint lines from the global
    parameter files, exactly as before -- plus an #@INSth line carrying the
    theoretical instrumental function whenever one is current. The conventional
    lines are never dropped, so a .dat converted with the accurate shape in use
    still describes that same source to any reader that only knows #@INSexp.

    Returns:
        (lines, method): list of header strings (without newlines) and the
        method name ('CMS' or 'SMS').
    """
    if getattr(app, 'MS_fit', None) is not None and app.MS_fit.isChecked():
        g = get_internal_gcms(app)
        return [f'{DAT_GCMS_PREFIX} {float(g)}'], 'CMS'

    instrumental_int_path = os.path.join(app.params_dir, 'INSint.txt')
    MulCo, x0 = np.genfromtxt(instrumental_int_path, delimiter=' ', skip_footer=0)
    INS = np.atleast_1d(get_legacy_sms_instrumental(app))
    lines = [
        DAT_INS_EXP_PREFIX + ' ' + ' '.join(str(float(v)) for v in INS),
        f'{DAT_INS_INT_PREFIX} {float(MulCo)} {float(x0)}',
    ]
    acc = read_accurate_instrumental(app)
    if acc is not None:
        lines.append(DAT_INS_ACC_PREFIX + ' '
                     + ' '.join(str(float(v)) for v in np.atleast_1d(acc)))
    return lines, 'SMS'


def read_dat_metadata_lines(file_path):
    """Verbatim instrumental header lines (#@INSexp/#@INSint/#@GCMS) of a .dat
    file, without trailing newlines. Empty list when the file has none.

    Used to carry instrumental metadata across spectrum-processing operations
    (subtract model, half points, sum) so a converted .dat does not silently
    lose its #@GCMS / #@INSexp / #@INSint header.
    """
    lines = []
    if not file_path or not os.path.exists(file_path):
        return lines
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                # the LEGACY spelling has to be here too: a .dat written before
                # the rename carries #@INSacc, and dropping it here would
                # silently lose that file's theoretical shape the first time any
                # operation (subtract model, half points, sum) copied its
                # metadata forward
                if stripped.startswith((DAT_INS_EXP_PREFIX, DAT_INS_INT_PREFIX,
                                        DAT_GCMS_PREFIX, DAT_INS_TH_PREFIX,
                                        DAT_INS_ACC_PREFIX_LEGACY)):
                    lines.append(stripped)
                elif not stripped.startswith('#') and not stripped.startswith('<'):
                    break
    except Exception as e:
        print(f"[Instrumental function] Could not read DAT metadata lines from {file_path}: {e}")
    return lines


def dat_metadata_key(file_path):
    """A hashable, comparable representation of a .dat file's instrumental
    metadata, or None when the file carries none. Two files share metadata iff
    their keys are equal."""
    meta = parse_dat_instrumental_metadata(file_path)
    if meta['has_gcms']:
        return ('CMS', round(float(meta['GCMS']), 9))
    if meta['has_insexp'] and meta['has_insint']:
        ins = tuple(round(float(v), 9) for v in np.atleast_1d(meta['INS']))
        acc = (tuple(round(float(v), 9) for v in np.atleast_1d(meta['INSacc']))
               if meta['has_insacc'] else None)
        return ('SMS', ins, round(float(meta['MulCo']), 9),
                round(float(meta['x0']), 9), acc)
    return None


def shared_dat_metadata_lines(file_paths):
    """Instrumental header lines to carry over when several .dat spectra are
    combined into one (the "sum all spectra" case).

    Files without metadata are ignored. If every file that *has* metadata agrees,
    those lines are returned; if they disagree, [] is returned so the combined
    file is written without (now-ambiguous) instrumental metadata.
    """
    keyed = [(dat_metadata_key(fp), fp) for fp in file_paths]
    keyed = [(k, fp) for k, fp in keyed if k is not None]
    if not keyed:
        return []
    first_key = keyed[0][0]
    if all(k == first_key for k, _ in keyed):
        return read_dat_metadata_lines(keyed[0][1])
    return []


def default_theory_instrumental():
    """The built-in theoretical instrumental function, as a KIND_PHYSICAL array.

    THEORY_START plus THEORY_FIXED: the operating point of spectrum 008 with the
    crystal constants from the 83-spectrum global fit.
    """
    kw = dict(THEORY_FIXED)
    kw.update({k: THEORY_START[k] for k in THEORY_START})
    return smst.encode_physical(**kw)


def reset_instrumental_defaults(app):
    """Restore the default instrumental function OF THE SELECTED DESCRIPTION.

    The two descriptions are independent stores and the user switches between
    them, so a reset resets the one that is selected and leaves the other alone
    -- resetting both would throw away a search the user is not even looking at.

        gauss   INSexp.txt (with INSint.txt, its integration grid)
        theory  INSth.txt, back to the built-in starting values

    It used to always rewrite INSexp/INSint and DELETE the theoretical shape,
    which from "theory" mode silently discarded the thing being reset.
    """
    try:
        method = instrumental_method(app)
        os.makedirs(app.params_dir, exist_ok=True)
        if method == 'theory':
            write_accurate_instrumental(app, default_theory_instrumental())
            what = f"{INS_TH_FILE} ({smst.describe_ins(default_theory_instrumental())})"
        else:
            insexp_path = os.path.join(app.params_dir, INS_EXP_FILE)
            instrumental_int_path = os.path.join(app.params_dir, 'INSint.txt')
            with open(insexp_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_INSEXP_TEXT)
            with open(instrumental_int_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_INSINT_TEXT)
            what = f"{INS_EXP_FILE}, INSint.txt"

        app.set_status(f"Instrumental defaults restored: {what}", "green")
        QMessageBox.information(
            app, "Instrumental function",
            f"Default values were restored for the selected description "
            f"({'Theory' if method == 'theory' else 'Set of Gaussians'}).\n\n"
            f"{what}\n\nThe other description was left untouched.")
        return True
    except Exception as e:
        app.set_status(f"Failed to restore instrumental defaults: {e}", "red")
        return False




def _load_be_param(params_dir):
    """Reference-absorber (Be window Doublet) parameters from Be.txt, with the
    built-in fallback values when the file is missing or unreadable."""
    try:
        be_path = os.path.join(params_dir, 'Be.txt')
        Be_param = np.genfromtxt(be_path, delimiter='\t', skip_footer=0)
        print('Be file was read')
    except Exception:
        Be_param = np.array([0.057, 0.066, -0.261, NAT_WIDTH, 0.375, 90, 0, 0.427037824, 1])
        print('COULD NOT READ Be.txt')
    return Be_param


# Lorentzian FWHM of the ESRF standard single-line absorber, measured with this
# model on the 83 raw spectra of the (T, theta) study: 0.128 mm/s, i.e. 1.30
# natural widths, with a line-centre optical depth of 4.18. It is NOT known to
# the +-0.002 the covariance reports: width and thickness are strongly anti-
# correlated (the fit trades t against g at nearly constant t*g^2), and between
# stages of the same global fit the pair moved over 0.128-0.146 mm/s and
# 3.2-4.2. Take +-0.01 mm/s, i.e. 1.3 +- 0.1 natural widths, as the honest
# uncertainty; what the data say firmly is that it is NOT the natural width.
#
# The number is only this stable once the fit is allowed what the experiment
# actually had: a polynomial baseline per spectrum and a velocity zero per
# measurement session. Without those it drifts higher still -- the absorber
# width absorbs the mismatch. Holding the reference absorber at exactly the
# natural width, as the legacy search does, forces the difference into the
# instrumental function instead, which the 4-parameter theoretical shape cannot
# absorb; so the theoretical search starts here and lets it vary.
ESRF_STANDARD_LINE_WIDTH = 0.1277       # mm/s


def build_reference_model(app, mode, B, CMS_ch=0, free_absorber_width=False):
    """Reference absorber used to see the instrumental function, for one mode.

    This is the part of the instrumental-function procedure that does NOT depend
    on the shape being fitted, so the legacy Gaussian-sum search and the NEW
    theoretical search share it verbatim.

    mode 0 : ESRF single-line standard  -- Be-window Doublet + Singlet;
             only the baseline Ns (0) and the Singlet effective thickness T (17)
             are free.
    mode 1 : the model currently in the parameters table (constraints,
             distributions, expressions and Recon weights all held fixed).
    mode 2 : pure alpha-Fe -- Be-window Doublet + polarized Sextet; Ns (0),
             T (17) and the texture A (25) are free, plus Nnr (4) in CMS.

    ``free_absorber_width`` (modes 0 and 2) additionally releases the reference
    absorber's own Lorentzian width and starts it at the measured
    ESRF_STANDARD_LINE_WIDTH instead of the natural one. Default False, so the
    legacy search is untouched.

    Returns ``(model, p, bounds, fix, extra)`` where ``extra`` carries the
    mode-1 distribution / correlation / reconstruction lists (empty otherwise).
    ``p`` holds ONLY the model parameters: the caller appends its own
    instrumental parameters and the matching bounds.
    """
    def release_width(p, bounds, fix, idx, start=None):
        """Free parameter ``idx`` (an absorber Lorentzian width) and bound it."""
        if start is not None:
            p[idx] = start
        bounds[0][idx] = 0.3 * NAT_WIDTH
        bounds[1][idx] = 6.0 * NAT_WIDTH
        return p, bounds, np.asarray(fix, dtype=int)[np.asarray(fix, dtype=int) != idx]

    extra = {'Distri': [0], 'Cor': [0], 'Recon': [0], 'confu': None}

    if mode == 1:
        model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(app)

        # Apply expressions in the same namespace the fit uses (bare numpy
        # names + p); a plain eval() here would miss those names.
        for i in range(0, len(NExpr)):
            p[NExpr[i]] = mi._eval_expr(Expr[i], p)
        for i in range(0, len(con1)):
            p[int(con1[i])] = p[int(con2[i])] * con3[i]

        extra['confu'] = np.array([con1, con2, con3]) if len(con1) > 0 else np.array([[-1], [-1], [-1]])

        # Box bounds and user-fixed parameters straight from the table
        bounds, fix = read_bounds_and_fix(app, len(p))

        # Add constraint, distribution and reconstruction placeholder indices to
        # fix (during an instrumental-function fit the model params, including the
        # Recon weights, are held fixed -- only the INS parameters vary).
        fix = np.concatenate((fix, con1), axis=0)
        fix = np.concatenate((fix, DistriN), axis=0)
        fix = np.concatenate((fix, NExpr), axis=0)
        fix = np.concatenate((fix, ReconN), axis=0)
        fix = np.unique(fix)
        extra['Distri'], extra['Cor'], extra['Recon'] = Distri, Cor, Recon
        return model, p, bounds, fix, extra

    if mode == 0:
        model = ['Doublet', 'Singlet']
        p = np.array([estimate_edge_background(B)])
        p = np.concatenate((p, np.array([0, 0, 0, 0, 0, 0, 0])))

        Be_param = _load_be_param(app.params_dir)
        p = np.concatenate((p, Be_param))
        p1 = np.array([4.6, -0.097, NAT_WIDTH, 0.0])
        p = np.concatenate((p, p1))
        # Layout: baseline(8) + polarized Doublet(9) + Singlet(4); Singlet T is 17.
        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][17] = 0.001
        # Fix everything except Ns + T of singlet (indices 0 and 17)
        fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20], dtype=int)
        if free_absorber_width:
            # 19 = the Singlet's Lorentzian width
            p, bounds, fix = release_width(p, bounds, fix, 19,
                                           ESRF_STANDARD_LINE_WIDTH)
        return model, p, bounds, fix, extra

    if mode == 2:
        model = ['Doublet', 'Sextet']
        if CMS_ch == 0:
            p = np.array([estimate_edge_background(B)])
            p = np.concatenate((p, np.array([0, 0, 0, 0, 0, 0, 0])))
        else:
            bg_tmp = estimate_edge_background(B)
            p = np.array([bg_tmp * 0.6])
            p = np.concatenate((p, np.array([0, 0, 0, bg_tmp * 0.4, 0, 0, 0])))

        Be_param = _load_be_param(app.params_dir)
        if CMS_ch == 1:
            Be_param[0] = 0

        p = np.concatenate((p, Be_param))
        # Polarized Sextet (14): I, d, e, H, L, G, theta_k=90, phi_h=0, A=0, Am=0, a+, a-, GH, I13.
        p1 = np.array([7.5, 0, 0, ALPHA_FE_FIELD, NAT_WIDTH, 0, 90, 0, 0, 0, 0, 0, 0, 3])
        p = np.concatenate((p, p1))
        # Layout: baseline(8) + polarized Doublet(9) + polarized Sextet(14);
        # Sextet T is index 17, its texture A is index 25, its Am is index 26.
        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][17] = 0.001
        if CMS_ch == 0:
            # Fix everything except Ns + T and A of sextet (indices 0, 17, 25)
            fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30], dtype=int)
        else:
            # Fix everything except Ns, Nnr + T and A of sextet (indices 0, 4, 17, 25)
            fix = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30], dtype=int)
            bounds[0][4] = 0
        if free_absorber_width:
            # 21 = the Sextet's Lorentzian width. An alpha-Fe foil's lines are
            # not at the natural width either, and the theoretical instrumental
            # function has no freedom to absorb the difference -- measured on
            # for_tests/000_sms.dat, releasing it takes chi2 from 2.17 to 1.95.
            p, bounds, fix = release_width(p, bounds, fix, 21)
        return model, p, bounds, fix, extra

    raise ValueError(f"build_reference_model: unknown mode {mode!r}")


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
            # Refinement is locked to SMS here, so the reference file's metadata
            # must not flip it to CMS (force_method='SMS').
            mp = resolve_instrumental_for_file(app, file, use_dat_metadata=use_dat_metadata, force_method='SMS')
            INS, MulCo, x0, note = mp['INS'], mp['MulCo'], mp['x0'], mp['note']
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
    
    model, p, bounds, fix, extra = build_reference_model(app, mode, B, CMS_ch)
    Distri, Cor, Recon = extra['Distri'], extra['Cor'], extra['Recon']

    bounds = np.concatenate((bounds, bounds0), axis=1)
    mod_p_len = len(p)
    p = np.concatenate((p, p0))
    print(p)
    print(fix)

    if mode == 1:
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:], Distri, Cor, Recon=Recon)
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Distri, Cor, Met=1, Recon=Recon)

    # CMS_ch could not be equal to 1 for mode 0
    # Not meaningful to use ESRF standard single line absorber for CMS
    elif mode == 0:
        print(f"[Instrumental function] Mode 0: Creating INSSS function, CMS_ch={CMS_ch}")
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:])
            print(f"[Instrumental function] INSSS function created for CMS_ch=0")

    elif mode == 2:
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
        insexp_path = os.path.join(app.params_dir, 'INSexp.txt')
    else:
        insexp_path = os.path.join(app.params_dir, 'GCMS.txt')
    
    if CMS_ch == 0:
        with open(insexp_path, "w") as f:
            for i in range(0, len(INSp)):
                f.write(str(INSp[i]) + ' ')

        instrumental_int_path = os.path.join(app.params_dir, 'INSint.txt')
        with open(instrumental_int_path, "w") as f:
            f.write(str(MulCo) + ' ')
            f.write(str(x0) + ' ')

        # THE THEORETICAL SHAPE IS *NOT* DELETED HERE any more.
        #
        # It used to be: presence of INSth.txt was what made the theoretical
        # shape "the current one", so the only way back to the Gaussians was to
        # remove it, and the conventional search did that on every run.
        #
        # Which description is used is now the user's SETTING (Supp -> "Choose
        # how to approximate instrumental function"), and both are kept side by
        # side precisely so switching costs nothing and switching back is free.
        # Deleting one of them here threw that away: with the Gaussian sum as
        # the default, one conventional search silently destroyed the
        # theoretical shape, after which selecting "Theory" fell back to the
        # Gaussians for ever and looked like the setting had stopped working.
        #
        # Use "Reset to default values" to restore either description; it acts
        # on the selected one and leaves the other alone.
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


# ======================================================================
#  "Find instrumental function NEW" -- the theoretical SMS shape
# ======================================================================
#
# The legacy search above fits an EMPIRICAL sum of Gaussians: a flexible basis
# with no physics in it, whose tails fall off as exp(-v^2) where the real source
# tails fall off as v^-4, and whose parameters mean nothing individually.
#
# This search fits the simulated instrumental function of a 57FeBO3 synchrotron
# Moessbauer source instead (syncmoss.sms_theory): the exact hyperfine
# Hamiltonian -- the same one the 'Hamiltonian' model uses, checked against it to
# 2e-15 mm/s -- feeding the M1 scattering amplitude, the pure-nuclear structure
# factor of the (NNN)_rh reflection and exact two-beam dynamical diffraction.
# Four physical numbers vary:
#
#   theta  the position on the rocking curve       [urad]
#   B_s    the staggered hyperfine field           [T]   (the temperature knob)
#   dEQ    the quadrupole splitting of FeBO3       [mm/s]
#   shift  the isomer + second-order-Doppler shift [mm/s]
#
# Everything else (crystal thickness, mosaicity, setting accuracy,
# Lamb-Moessbauer factor, reflection order) is held at its measured value; those
# are exposed as THEORY_FIXED so they can be changed in one place.

# Parameters the search CAN vary, in the order they are appended to ``p``.
# Which of them a given pass actually releases is THEORY_PASSES; a field that no
# pass releases simply stays at its starting value.
#
#   mosaic_urad   THE incoherent broadening: the incidence-angle spread, i.e.
#                 the crystal slope error quadrature-summed with the beam
#                 divergence and the angular setting accuracy. It does not move
#                 any resonance energy -- it averages the DIFFRACTION over the
#                 rocking curve, which re-weights the four poles. That is why it
#                 matters most at theta = 0, where the reflectivity sits in the
#                 local minimum between the two rocking peaks: there it changes
#                 the shape, not just its smoothness (chi2 1.286 -> 1.060 on the
#                 theta = 0 subset, and nothing at all at 17 or 44 urad).
#                 Global fit: 14.49 +- 0.11 urad, against the nominal 5 urad
#                 mosaic + 3 urad setting = 5.8 urad in quadrature. A fit left
#                 to run further wants 58 urad, which the 17 measured rocking
#                 curves exclude (that would predict 102-175 urad widths against
#                 70-133 measured), so the bound matters.
#   dBs_rel_fwhm  relative spread of B_s from the +-1..5 mK temperature stability
#                 and the +-4 Oe field inhomogeneity. MEASURED AND REJECTED:
#                 released from the converged optimum it refines to 0.389 -- a
#                 39 % spread of B_s, eight times what that stability allows --
#                 and buys +0.0053 in chi2 against sqrt(2/42579) = 0.0069 for one
#                 sigma, while making the intensity agreement worse. A sharply
#                 determined value that buys nothing is a parameter absorbing
#                 model error. Kept at zero; sms_study/test_field_spread.py has
#                 the A/B.
#   gauss_fwhm    Gaussian energy imperfection [mm/s]: drive jitter and
#                 nonlinearity, drift of the velocity zero over a long count,
#                 vibration. RELEASED -- see THEORY_PASSES. The ideal source is a
#                 SQUARED Lorentzian of 0.644 natural widths
#                 (sms_theory.squared_lorentzian_limit), narrower than anything
#                 measured, so something must broaden it.
#
#                 An earlier version pinned this at zero on the strength of a
#                 test that found it 59 % redundant with mosaic_urad. That test
#                 was run with a free ADDITIVE baseline and with B_s free per
#                 spectrum -- both of which absorb a uniform broadening, leaving
#                 the Gaussian nothing to do. Re-measured with the baseline
#                 constrained to a pure scale and one B_s per temperature, the
#                 two are 100 % ADDITIVE and the Gaussian is the STRONGER of the
#                 pair (see the note on THEORY_PASSES below). It is released.
THEORY_FREE_FIELDS = ('theta_urad', 'B_s', 'dEQ', 'shift',
                      'mosaic_urad', 'dBs_rel_fwhm', 'gauss_fwhm', 'f_LM')

# B_s = 0.50 T is the middle of the range the ESRF source is actually operated
# over (0.19-2.4 T across the published temperature series), i.e. T ~ 75.93 C by
# the calibrated law.
#
# dEQ = -0.4216 +- 0.0038 mm/s (2026-09-29 global fit of the 83-spectrum ESRF
# series: 123 parameters, shape chi2 1.317, and the independent intensity
# cross-check agreeing to 0.91-1.05 across all nine temperatures).
#
# QUOTE THE ERROR AS +-0.03, NOT +-0.004. dEQ tracks how much freedom the model
# is given rather than converging on a value: -0.3228, then -0.4056, then
# -0.3900, now -0.4216, each time the source model changed. The +-0.0038 is the
# covariance of one fit; the spread ACROSS defensible fits is ten times that. So
# the laboratory value of Lyubutin et al. (2022), 2q = -0.3815 mm/s in this
# parametrisation, is consistent with this one -- do not report a disagreement.
#
# f_LM is the opposite and IS well determined: 0.7718, 0.7661, 0.7714, 0.7711
# across those same four models. It is pinned by the width of the saturated core
# (see the dynamical-saturation argument in sms_theory), which the model gets
# right, while dEQ trades against whatever is absorbing line asymmetry.
#
# mosaic_urad starts at ZERO: the angle deviation is off by default (see
# THEORY_PASSES), and setting_urad = 3 urad is applied regardless, so the source
# is never completely unsmeared. Released only at the rocking minimum, where the
# 83-spectrum fit puts it at 20-28 urad (it is a property of one measurement's
# setting, not a constant -- so this is a start, not a value).
#
# theta and B_s are the OPERATING POINT: spectrum 008 of the ESRF series
# (T = 75.825 C, the angle the source is actually used at), refitted under these
# very conventions -- setting_urad = 3, n_angle = 11 -- at chi2 = 0.97. Starting
# a search at a guessed point instead is what the B_s scan exists to repair.
#
# dEQ, gauss_fwhm and f_LM come from the GLOBAL fit, not from 008: they are
# properties of the crystal and the instrument, and one spectrum determines dEQ
# only to about +-0.04 (008 alone prefers -0.24).
THEORY_START = {'theta_urad': 87.3, 'B_s': 0.7898, 'dEQ': -0.4216,
                'shift': -0.2882, 'mosaic_urad': 0.0, 'dBs_rel_fwhm': 0.0,
                'gauss_fwhm': 0.1303, 'f_LM': 0.7711}

# Levenberg-Marquardt budget for the SCHEDULE passes. The final polish keeps
# the full budget; the intermediate passes do not need it and each iteration is
# expensive here -- every step changes the physical parameters, so the whole
# dynamical-diffraction shape is rebuilt (~0.25 s) instead of being read from
# the cache, which is why an ordinary fit is fast and this search is not.
THEORY_PASS_MI = 10

# B_s is scanned over this grid once, at the moment the schedule first releases
# it. The lower bound is a GENUINE local minimum -- a source collapsed to a
# single narrow line fits a smooth spectrum tolerably, and Levenberg-Marquardt
# cannot climb out of it. On the two alpha-Fe test spectra, which are the same
# source measured twice, a fit started at 0.50 T found B_s = 0.88 T on one and
# stuck at the 0.02 T bound on the other, for a chi2 of 1.75 against 2.24. One
# scan of nine cheap model evaluations removes that coin flip.
THEORY_BS_SCAN = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.10, 1.60, 2.30)

# Shift offsets tried at each scanned field, relative to the starting value.
# The velocity zero is the number a user is least able to supply -- it belongs
# to the drive calibration of one measurement -- so Find maps it instead of
# asking.
#
# The span matters more than the spacing. The map only chooses a STARTING
# point, which the next pass then fits properly, so a coarse grid is fine -- but
# one that cannot reach the answer is useless. Measured: started 0.69 mm/s away
# with a +-0.45 grid, every point came back at chi2 ~202 because none of them
# was right, and the map picked noise. +-0.9 covers the whole of
# THEORY_BOUNDS['shift'] at 0.15, which is the width of the line itself.
THEORY_SHIFT_SCAN = tuple(round(0.15 * k, 4)
                          for k in range(-6, 7) if k != 0)

# Starting field spread used when dBs_rel_fwhm is actually released: at exactly
# zero it sits on its lower bound with a vanishing derivative (a zero-width
# average collapses to a single quadrature node), so Levenberg-Marquardt could
# never move it off. 2 % is the order the published +-1..5 mK stability and
# +-4 Oe inhomogeneity give a few tenths of a degree above T_N.
THEORY_DBS_START = 0.02

# Box bounds. theta spans both sides of the Bragg peak; B_s stops well above any
# near-T_N value and well below a spectrum that would show resolved lines; dEQ
# covers either sign (its sign sets the asymmetry of the source line); shift is
# the source position on the velocity scale.
THEORY_BOUNDS = {
    # theta is a ROCKING-CURVE POSITION, and the curve is 70-133 urad wide with
    # its humps within +-50; the ESRF series spans 0 to 209 urad. The old
    # (-300, 600) let the fit leave the rocking curve altogether, where the
    # reflectivity is tiny but its NORMALISED shape is broad and featureless and
    # can imitate almost any smeared line. That is not hypothetical: refining
    # 008 from a sensible start ran theta to exactly -300, the bound, and stored
    # it. Statistically it costs nothing to forbid -- the runaway beat the
    # physical point by 0.04 in chi2, against 0.063 for one sigma.
    'theta_urad': (-120.0, 260.0),
    # 0.02 T is a TRAP, not a bound: a source collapsed to a single narrow line
    # describes a smooth spectrum tolerably and Levenberg-Marquardt cannot climb
    # back out. The smallest field in the measured series is 0.19 T.
    'B_s': (0.10, 3.00),
    'dEQ': (-1.50, 1.50),
    'shift': (-1.00, 1.00),
    'mosaic_urad': (0.0, 60.0),
    'dBs_rel_fwhm': (0.0, 0.60),
    'gauss_fwhm': (0.0, 0.60),
    # the nuclear amplitude scale: not far from the crystal's own value, which
    # is what the bounds enforce. Below ~0.4 the rocking curve collapses and the
    # fit is in a different regime altogether.
    'f_LM': (0.55, 0.95),
}

# Held fixed. n_angle / n_field are the quadrature orders of the mosaic and field
# averages, not physics: n_field > 1 costs nothing while dBs_rel_fwhm is 0,
# because a zero-width Gaussian average collapses to a single node.
THEORY_FIXED = {
    'thickness_um': smst.DEFAULT_THICKNESS_UM,
    'setting_urad': smst.DEFAULT_SETTING_URAD,
    'phi_m_deg': 0.0,
    'n_angle': 11,
    'n_field': 5,
    'n_gauss': smst.DEFAULT_N_GAUSS,
    'N_order': 1,
}

# Which physical parameters each pass releases, on top of the reference model's
# own free parameters (baseline Ns, absorber thickness T, and the texture A in
# the alpha-Fe mode). Freeing everything at once from a cold start walks straight
# into the theta / B_s correlation and stalls -- both change the width -- so the
# position and the angle are given to the fit first.
#
# THE ESCALATION ORDER. Only the GAUSSIAN imperfection is released by default.
# The angle deviation and the nuclear amplitude scale are held, in that order of
# priority, and are released only if the residual demands it (see
# THEORY_ESCALATION below and the ``free_fields`` argument).
#
# mosaic_urad IS NOT A DEFAULT, and the reason is measured rather than assumed.
# The angular average matters only at theta = 0, where the crystal sits in the
# local minimum between the two rocking humps and averaging mixes them in,
# changing the SHAPE. On either flank the rocking curve is smooth, so the same
# average only smooths the line -- which is exactly what gauss_fwhm already
# does, and the two become degenerate. Measured on six flank spectra with the
# Gaussian free (sms_study/test_spread_from_zero.py), removing the angle
# deviation entirely costs +0.0003 in chi2, and three of the six refine it to
# zero on their own. In real use the source is not operated at theta = 0, so the
# default is off.
#
# It also cannot be transplanted. Measured AT theta = 0 with B_s held at the
# value the temperature fixes, the deviation is 27.5-50.8 urad across the nine
# temperatures with individual errors of +-1-2 urad -- a 20-sigma scatter, so it
# is not one number, and it is 5-9x the nominal 5 (+) 3 = 5.8 urad. Pinning the
# theta = 0 value onto the flank costs +1.23 in chi2. Whatever it is absorbing
# at theta = 0, it is not beam divergence.
# dEQ IS HELD. It is the quadrupole coupling of FeBO3 -- one number for the
# crystal, not something a single spectrum may choose. Measured, one spectrum
# determines it to about half a sigma: on 008, moving it from -0.24 to -0.39
# costs 0.032 in chi2 against 0.063 for one sigma. Releasing it therefore does
# not measure dEQ, it just lets dEQ absorb whatever else is wrong -- and it was
# doing exactly that, drifting to -0.21 on a spectrum already fitted at 0.93.
# The value comes from the 83-spectrum global fit; release it with
# ``free_fields`` if a genuinely different crystal is ever used.
#
# theta IS HELD TOO, for the same reason and with better evidence. It is the
# rocking-curve position the experimenter SET; it is not the fit's to discover.
# Released against a single-line spectrum it is not measured but invented: on
# 008, starting from the right answer, it runs to whatever the lower bound is --
# -300 with the old bounds, -120 with the tightened ones -- taking B_s down to
# its bound with it, because far off the rocking curve the normalised shape is
# broad and featureless and imitates any smeared line. Its error comes back nan
# and corr(theta, B_s) = -0.46. It costs nothing to hold: theta fixed at the set
# 87.3 urad gives chi2 0.972 against 0.952 for the runaway, well inside the
# 0.063 that is one sigma here.
#
# So the schedule fits what a spectrum can actually say: the velocity zero, the
# field, and the incoherent broadening. theta and dEQ come from the experiment
# and the crystal. Release either with ``free_fields``, or let the escalation do
# it when the residual justifies it.
THEORY_PASSES = (
    ('shift',),
    ('shift', 'B_s'),
    ('shift', 'B_s', 'gauss_fwhm'),
    ('shift', 'B_s', 'gauss_fwhm'),
)

# If the default schedule leaves the residual above this, the search escalates:
# it re-runs releasing the extra fields in order, keeping a step only when it
# lowers the reduced chi2 by more than THEORY_ESCALATION_GAIN. The order is the
# physical one -- the angle deviation is an instrument property of THIS
# measurement, the amplitude scale a property of the crystal, so the cheaper and
# more local explanation is tried first.
#
# ORDER: the rocking angle first, then the incidence-angle deviation, then the
# quadrupole coupling, and the nuclear amplitude scale last. That is increasing
# reach -- theta belongs to this measurement, the deviation to this instrument,
# dEQ and the amplitude scale to the crystal itself, and a crystal constant
# should be the last thing a single spectrum is allowed to move.
THEORY_ESCALATION = ('theta_urad', 'mosaic_urad', 'dEQ', 'f_LM')
# A released parameter that ends ON a bound has not been fitted, it has run
# away -- and it can still show a chi2 gain while doing it (theta escapes the
# rocking curve entirely, where the normalised shape is featureless and fits
# anything). Such a step is rejected however much it "buys".
THEORY_ESCALATION_EDGE = 1e-3

# AT THE ROCKING MINIMUM the angle deviation stops being optional. theta = 0 is
# the local minimum between the two rocking humps, so an angular average mixes
# the humps in and changes the SHAPE; on either flank the curve is smooth and
# the same average only smooths the line, which gauss_fwhm already does (dropping
# it there costs +0.0003 in chi2 over six flank spectra). So the deviation
# starts at zero and is held -- except at theta = 0, where it starts at a small
# fitted value and is released with the rest.
#
# 27 urad is the LOW end of what the nine theta = 0 spectra want when B_s is
# held at the value their temperature fixes (27.5 to 50.8 urad, coldest first).
# A start, not a claim: those nine do not agree within their own +-1-2 urad
# errors, so this is not an instrument constant and the fit must move it.
THEORY_MOSAIC_AT_ZERO = 27.0
THETA_ZERO_TOL_URAD = 1.0
THEORY_ESCALATION_CHI2 = 1.30
THEORY_ESCALATION_GAIN = 0.02


def theory_start_from_app(app, ref):
    """Starting physical parameters: the stored ones when refining, else defaults.

    REFINE (ref 1 or 3) continues from the theoretical instrumental function
    currently in INSth.txt; FIND (ref 0 or 2) always starts from the built-in
    values. When refining and there is nothing stored, the defaults are used and
    a note says so (a conventional Gaussian sum carries no physical parameters
    to continue from).

    ``ref`` was compared against 1 alone until the theoretical search stopped
    being a separate set of menu entries. Theory-Refine is ref = 3, so that test
    sent Refine back to the defaults every time instead of continuing.
    """
    start = dict(THEORY_START)
    fixed = dict(THEORY_FIXED)
    if ref not in (1, 3):
        return start, fixed, 'built-in starting values'
    INS = read_accurate_instrumental(app)
    if INS is None or smst.ins_kind(INS) != smst.KIND_PHYSICAL:
        return start, fixed, (f'built-in starting values (no theoretical '
                              f'instrumental function stored in {INS_ACC_FILE})')
    stored = smst.decode_physical(INS)
    for k in THEORY_FREE_FIELDS:
        start[k] = float(stored[k])
    for k in list(fixed):
        fixed[k] = stored[k]
    return start, fixed, f'the theoretical instrumental function in {INS_ACC_FILE}'


def instrumental_theory(app, ref=0, mode=0, pool=None, n_rational=2,
                        free_fields=None, temperature_C=None):
    """
    Find (ref=0) or refine (ref=1) the THEORETICAL SMS instrumental function.

    Same reference absorbers as the legacy search -- mode 0 the ESRF single-line
    standard, mode 1 the model in the parameters table, mode 2 pure alpha-Fe --
    but the fitted shape is the simulated one, with the four physical parameters
    above in place of 3*n free Gaussian numbers.

    On success INSth.txt holds the theoretical instrumental function, INSint.txt
    the matching integration constants, and INSexp.txt the best conventional
    Gaussian-sum stand-in for the same source -- so every later fit and every
    RAW->DAT conversion uses the theoretical shape, while a .dat still carries a
    usable #@INSexp for anything that does not know the #@INSth marker.

    ``temperature_C`` is the iron borate crystal temperature, when it was
    recorded. Temperature enters the physics only through B_s, so it is turned
    into a starting B_s by the calibrated law (sms_theory.Bs_from_temperature,
    good to ~10 %); B_s is then still fitted, since that law is a calibration of
    one crystal in one magnet. Pass it when you have it -- it is worth more than
    any of the default starting values.

    ``free_fields`` replaces the LAST pass's set of released physical parameters
    (names from THEORY_FREE_FIELDS), so the two incoherent broadenings can be
    released on data that warrants it without editing the module. The default
    schedule holds them at their measured values on purpose: on the alpha-Fe test
    spectrum a free mosaic runs straight to its upper bound (60 urad against the
    5 urad nominal) for a chi2 gain of 0.2, which is the fit using it as a
    generic smoothing knob -- most of that same gain comes instead from letting
    the alpha-Fe line width breathe, i.e. it was absorber-model error, not
    crystal physics. Releasing them turns the instrumental function back into the
    flexible empirical basis this procedure exists to replace.

    SMS only: a conventional radioactive source has a single-Gaussian CMS
    instrumental function and no iron borate crystal to compute.
    """
    if getattr(app, 'MS_fit', None) is not None and app.MS_fit.isChecked():
        raise ValueError(
            "The theoretical instrumental function describes a synchrotron "
            "Moessbauer source (57FeBO3 pure nuclear reflection). It does not "
            "apply to CMS; switch to SMS, or use the standard search.")

    print(f"[Instrumental function: theory] theoretical SMS shape, ref={ref}, mode={mode}")
    start_time = time.time()

    JN0 = app.JN0
    file = os.path.abspath(app.path_list[0])
    A_list, B_list = load_spectrum(app, [file], calibration_path=app.calibration_path)
    A, B = A_list[0], B_list[0]

    # The reference absorber's own line width is released here (and only here):
    # the theoretical instrumental function has no spare freedom to absorb a
    # wrong one, where the empirical Gaussian sum silently does.
    model, p_model, bounds_model, fix_base, extra = build_reference_model(
        app, mode, B, 0, free_absorber_width=(mode in (0, 2)))
    Distri, Cor, Recon = extra['Distri'], extra['Cor'], extra['Recon']
    mod_p_len = len(p_model)
    fix_base = np.asarray(fix_base, dtype=int)

    # ref 1 and 3 are Refine (continue from what is stored); 0 and 2 are Find
    # (start from the built-in values).
    refining = ref in (1, 3)
    # a stale flag would abort this run at once
    if getattr(app, 'fit_cancel', None) is not None:
        app.fit_cancel.clear()
    start, fixed, start_note = theory_start_from_app(app, ref)
    # What the Find dialog collected, if anything (see syncmoss_main).
    overrides = getattr(app, 'theory_find_overrides', None) or {}
    if overrides:
        start.update({k: float(v) for k, v in overrides.items()})
        start_note += (" with " + ", ".join(f"{k}={float(v):g}"
                                            for k, v in sorted(overrides.items())))
    if free_fields and 'dBs_rel_fwhm' in free_fields \
            and start['dBs_rel_fwhm'] <= 0:
        start['dBs_rel_fwhm'] = THEORY_DBS_START
    if temperature_C is not None:
        start['B_s'] = smst.Bs_from_temperature(float(temperature_C))
        start_note += (f", B_s from T = {float(temperature_C):.3f} C "
                       f"-> {start['B_s']:.4f} T")
    print(f"[Instrumental function: theory] starting from {start_note}")
    for k in THEORY_FREE_FIELDS:
        print(f"    {k:12s} = {start[k]:+.5f}")

    p0 = np.array([float(start[k]) for k in THEORY_FREE_FIELDS])
    bounds0 = np.array([[THEORY_BOUNDS[k][0] for k in THEORY_FREE_FIELDS],
                        [THEORY_BOUNDS[k][1] for k in THEORY_FREE_FIELDS]], dtype=float)

    bounds = np.concatenate((bounds_model, bounds0), axis=1)
    p = np.concatenate((p_model, p0))

    def make_ins(theory_p):
        kw = {k: float(theory_p[i]) for i, k in enumerate(THEORY_FREE_FIELDS)}
        kw.update(fixed)
        return smst.encode_physical(**kw)

    # x0, MulCo and JN are reassigned between passes; INSSS LATE-BINDS them so it
    # always integrates on the current grid (the pattern Calibration.py uses).
    JN = max(int(JN0) * 2, 64)
    x0, MulCo = m5.limits(pool, JN, make_ins(p0))
    print(f"[Instrumental function: theory] integration grid: x0={x0:.5f}, MulCo={MulCo:.5f}")

    def INSSS(x_exp, pp):
        # COOPERATIVE INTERRUPT. "! INTERRUPT !" terminates and recreates the
        # multiprocessing pool, which stops an ordinary fit because an ordinary
        # fit spends its time IN the pool. This search does not: the cost is
        # rebuilding the dynamical shape (~1100 times), and that runs in the
        # search thread, so the old abort left it grinding on until it happened
        # to touch the pool again. Every model evaluation checks the flag, so
        # the answer comes back within one evaluation of the click.
        if search_cancelled(app):
            raise FitInterrupted()
        return m5.TI(x_exp, pp[:mod_p_len], model, JN, pool, x0, MulCo,
                     make_ins(pp[mod_p_len:]), Distri, Cor, Recon=Recon)

    def chi2_of(pp):
        return float(np.sum((B - INSSS(A, pp)) ** 2 / (abs(B) + 1) / len(B)))

    def fix_for(released):
        """Absolute fix indices: the reference model's, plus every physical
        parameter this pass does not release."""
        held = [mod_p_len + i for i, k in enumerate(THEORY_FREE_FIELDS)
                if k not in released]
        if not held:
            return fix_base
        return np.unique(np.concatenate((fix_base, np.array(held, dtype=int))))

    def refresh_grid():
        """Re-centre the transmission-integral grid on the current shape, keeping
        the old one if that made the fit worse (as the legacy search does)."""
        nonlocal x0, MulCo
        hi_before = chi2_of(p)
        old = (x0, MulCo)
        try:
            x0, MulCo = m5.limits(pool, int(JN), make_ins(p[mod_p_len:]))
        except Exception as e:
            x0, MulCo = old
            print(f"    limits() failed ({e}); integration grid kept")
            return
        hi_after = chi2_of(p)
        if hi_after - hi_before > 0.01:
            x0, MulCo = old
            print(f"    grid change rejected (chi2 {hi_before:.4f} -> {hi_after:.4f})")
        else:
            print(f"    grid -> x0={x0:.5f}, MulCo={MulCo:.5f} "
                  f"(chi2 {hi_before:.4f} -> {hi_after:.4f})")

    er = np.array([np.nan] * len(p))
    covariance_matrix = None
    hi2 = chi2_of(p)
    print(f"[Instrumental function: theory] chi2 at the starting point: {hi2:.4f}")

    passes = tuple(THEORY_PASSES) if free_fields is None else (
        tuple(THEORY_PASSES[:-1]) + (tuple(free_fields),))

    # theta = 0 is the one place the angle deviation is an observable, so there
    # it joins the schedule instead of waiting for the escalation.
    if free_fields is None and abs(float(start['theta_urad'])) <= THETA_ZERO_TOL_URAD:
        if abs(float(start['mosaic_urad'])) <= 0.0:
            start['mosaic_urad'] = THEORY_MOSAIC_AT_ZERO
            # p is already assembled at this point, so set it there, not in p0
            p[mod_p_len + THEORY_FREE_FIELDS.index('mosaic_urad')] = \
                THEORY_MOSAIC_AT_ZERO
        passes = tuple(pf + ('mosaic_urad',) if 'gauss_fwhm' in pf else pf
                       for pf in passes)
        print(f"[Instrumental function: theory] theta = 0 (the rocking "
              f"minimum): releasing the angle deviation, starting it at "
              f"{start['mosaic_urad']:.1f} urad")
    final_free = passes[-1] if passes else THEORY_FREE_FIELDS

    def scan_Bs():
        """Map the starting point over B_s AND shift, once, before either is
        released.

        Two dimensions rather than one because the velocity zero is the
        parameter a user is least able to supply: B_s at least follows from the
        temperature, while shift depends on the drive calibration of that
        particular measurement. Mapping it removes the need to ask.

        It is affordable because shift only TRANSLATES the source, which
        sms_theory._physical_table now exploits: the dynamical shape is computed
        once per field and every shift reuses it (0.05 ms against 115 ms). So
        the grid costs one shape per B_s, not one per point -- the loops are
        ordered field-outer for exactly that reason -- and the whole map is
        about as expensive as the old 1-D scan.

        The best point becomes the starting value. Nothing is fixed; the
        following passes still fit both.
        """
        i_B = mod_p_len + THEORY_FREE_FIELDS.index('B_s')
        i_s = mod_p_len + THEORY_FREE_FIELDS.index('shift')
        lo_B, hi_B = THEORY_BOUNDS['B_s']
        lo_s, hi_s = THEORY_BOUNDS['shift']
        s0 = float(p[i_s])
        shifts = [s0] + [s0 + d for d in THEORY_SHIFT_SCAN
                         if lo_s <= s0 + d <= hi_s]

        trial = np.array(p, dtype=float)
        best = (float(p[i_B]), s0)
        best_chi = chi2_of(p)
        n = 1
        for B in ([float(p[i_B])]
                  + [b for b in THEORY_BS_SCAN if lo_B <= b <= hi_B]):
            trial[i_B] = B
            row = []
            for s in shifts:
                trial[i_s] = s
                c = chi2_of(trial)
                row.append(c)
                n += 1
                if c < best_chi:
                    best, best_chi = (B, s), c
            print(f"    B_s {B:5.2f}: best over shift {min(row):.4f} "
                  f"at {shifts[int(np.argmin(row))]:+.4f}")
        print(f"    start map: {n} points over {len(shifts)} shifts x "
              f"{len(set([float(p[i_B])] + list(THEORY_BS_SCAN)))} fields")
        if (best[0], best[1]) != (float(p[i_B]), s0):
            print(f"    start moved: B_s {p[i_B]:.4f} -> {best[0]:.4f} T, "
                  f"shift {s0:+.4f} -> {best[1]:+.4f} mm/s "
                  f"(chi2 -> {best_chi:.4f})")
            p[i_B], p[i_s] = best[0], best[1]

    def run_passes(p_start, quiet=False, scan=True):
        """The whole annealing schedule from one starting point."""
        nonlocal p, er, hi2, covariance_matrix
        p = np.array(p_start, dtype=float)
        # THE MAP COMES FIRST, before any pass. It used to run at the moment
        # B_s was released, i.e. after pass 1 had already fitted the shift --
        # which is useless when pass 1 is the thing that fails. Measured:
        # started 0.69 mm/s off in shift, pass 1 stalled at chi2 205 and every
        # point of the map then came back at ~202, so the map ranked noise. Run
        # first it sees the real surface and gives pass 1 somewhere sane to
        # begin.
        if scan:
            scan_Bs()
        for n_pass, pass_free in enumerate(passes, start=1):
            p, er, hi2, covariance_matrix = mi.minimi_hi(
                INSSS, A, B, p, fix=fix_for(pass_free), bounds=bounds,
                tau0=0.0001, MI=THEORY_PASS_MI, MI2=THEORY_PASS_MI,
                eps=10 ** -6, fixCH=1)
            if not quiet:
                print(f"[Instrumental function: theory] pass {n_pass} "
                      f"(free: {', '.join(pass_free)}): chi2 = {hi2:.4f}")
                for i, k in enumerate(THEORY_FREE_FIELDS):
                    held = '' if k in pass_free else '   (held)'
                    print(f"    {k:12s} = {p[mod_p_len + i]:+.5f}{held}")
            refresh_grid()
        return np.array(p), np.array(er), hi2, covariance_matrix

    # The B_s scan is a FIND tool. Find starts from built-in values that may be
    # a long way from this source, and one cheap 1-D map over the field is what
    # keeps it out of the 0.10 T collapse. REFINE already starts from a fitted
    # parameter set for this very spectrum, so a scan there is at best ten
    # wasted model evaluations and at worst harmful: it ranks candidates at the
    # START point rather than after fitting, so it can walk a converged answer
    # away from its own minimum. Refine goes straight to the schedule.
    # An interrupt keeps the BEST POINT REACHED SO FAR rather than throwing the
    # run away: the schedule improves monotonically, so whatever pass had
    # finished is a better instrumental function than the one it started from,
    # and the user can refine again from it.
    interrupted = False
    try:
        run_passes(p, scan=not refining)
    except FitInterrupted:
        interrupted = True
        print("[Instrumental function: theory] INTERRUPTED; keeping the best "
              "point reached so far")

    # B_s at its lower bound is not a fit, it is a trap: a source collapsed to a
    # single narrow line describes a smooth spectrum tolerably and LM cannot
    # climb back out, while the real minimum sits near 0.9 T with a chi2 half a
    # unit lower. The scan alone does not always avoid it, because it ranks
    # candidates at the START point rather than after fitting. So if the answer
    # came back at the bound, pay for one restart from a genuinely wide source
    # and keep whichever is better.
    i_B = mod_p_len + THEORY_FREE_FIELDS.index('B_s')
    lo_B = THEORY_BOUNDS['B_s'][0]
    if p[i_B] < lo_B * 1.5 or p[i_B] < 0.05:
        print(f"[Instrumental function: theory] B_s came back at its lower bound "
              f"({p[i_B]:.4f} T); restarting from a wide source")
        keep = (np.array(p), np.array(er), hi2, covariance_matrix)
        p2 = np.array(keep[0])
        p2[i_B] = 0.90
        # no scan on the restart either: p2 already carries the deliberate wide
        # starting field this recovery exists to try
        r = run_passes(p2, quiet=True, scan=False)
        if r[2] < keep[2]:
            print(f"    restart wins: chi2 {keep[2]:.4f} -> {r[2]:.4f}, "
                  f"B_s {keep[0][i_B]:.4f} -> {r[0][i_B]:.4f} T")
        else:
            p, er, hi2, covariance_matrix = keep
            print(f"    restart rejected (chi2 {r[2]:.4f} vs {keep[2]:.4f})")

    # ESCALATION. The default schedule holds the angle deviation and the nuclear
    # amplitude scale: away from theta = 0 the first is degenerate with
    # gauss_fwhm (measured: removing it costs +0.0003 in chi2), and the second is
    # a property of the crystal that the 83-spectrum study already measured. If
    # the residual says otherwise, release them one at a time -- the cheaper,
    # more local explanation first -- and keep a step only when it actually pays.
    # An explicit ``free_fields`` overrides the schedule, so escalation is skipped
    # there: the caller has already said what should be free.
    if free_fields is None and hi2 > THEORY_ESCALATION_CHI2 and not interrupted:
        for extra in THEORY_ESCALATION:
            if extra in final_free:
                continue
            trial_free = tuple(final_free) + (extra,)
            print(f"[Instrumental function: theory] chi2 = {hi2:.4f} exceeds "
                  f"{THEORY_ESCALATION_CHI2}; trying with {extra} released")
            r = mi.minimi_hi(INSSS, A, B, np.array(p),
                             fix=fix_for(trial_free), bounds=bounds,
                             tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
            gain = hi2 - r[2]
            val = r[0][mod_p_len + THEORY_FREE_FIELDS.index(extra)]
            lo_x, hi_x = THEORY_BOUNDS[extra]
            span = max(hi_x - lo_x, 1e-30)
            at_edge = (min(abs(val - lo_x), abs(val - hi_x)) / span
                       < THEORY_ESCALATION_EDGE)
            if at_edge:
                print(f"    rejected: {extra} ran to its bound "
                      f"({val:+.4f}); that is not a fit, whatever the chi2 "
                      f"({hi2:.4f} -> {r[2]:.4f})")
            elif gain > THEORY_ESCALATION_GAIN:
                print(f"    kept: chi2 {hi2:.4f} -> {r[2]:.4f}, "
                      f"{extra} = {val:+.4f}")
                p, er, hi2, covariance_matrix = r
                final_free = trial_free
            else:
                print(f"    rejected: chi2 {hi2:.4f} -> {r[2]:.4f}, gain "
                      f"{gain:+.4f} below {THEORY_ESCALATION_GAIN}")
            if hi2 <= THEORY_ESCALATION_CHI2:
                break
        refresh_grid()

    # Final polish on the displayed integration grid, with the SAME parameters
    # free as the last pass -- not all of them, or a field the schedule
    # deliberately held would be released here without ever being annealed.
    JN = int(JN0)
    if not interrupted:
        try:
            p, er, hi2, covariance_matrix = mi.minimi_hi(
                INSSS, A, B, p, fix=fix_for(final_free), bounds=bounds,
                tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
        except FitInterrupted:
            interrupted = True
            print("[Instrumental function: theory] INTERRUPTED during the "
                  "final polish; keeping the annealed point")
    if interrupted:
        if getattr(app, 'fit_cancel', None) is not None:
            app.fit_cancel.clear()
        app.set_status("Instrumental-function search interrupted; the best "
                       "point reached was kept", "orange")

    INS = make_ins(p[mod_p_len:])
    theory = {k: float(p[mod_p_len + i]) for i, k in enumerate(THEORY_FREE_FIELDS)}
    theory_err = {k: (float(er[mod_p_len + i]) if k in final_free else np.nan)
                  for i, k in enumerate(THEORY_FREE_FIELDS)}

    fwhm, centre, area = smst.ins_metrics(INS)
    print(f"[Instrumental function: theory] final chi2 = {hi2:.4f} "
          f"({time.time() - start_time:.1f} s)")
    print("[Instrumental function: theory] physical parameters:")
    for k in THEORY_FREE_FIELDS:
        if k in final_free:
            print(f"    {k:12s} = {theory[k]:+.5f} +- {theory_err[k]:.5f}")
        else:
            print(f"    {k:12s} = {theory[k]:+.5f}   (held fixed)")
    print("    held at their measured values: "
          + ", ".join(f"{k}={v}" for k, v in fixed.items()))
    print(f"    source FWHM   = {fwhm:.4f} mm/s = {fwhm / NAT_WIDTH:.2f} natural widths")
    print(f"    source centre = {centre:+.4f} mm/s   (first moment "
          f"{smst.ins_centroid(INS):+.4f} mm/s)")
    print(f"    x0 = {x0:.5f}, MulCo = {MulCo:.5f}")

    # theta and B_s both set the width, so report how correlated they came out
    corr = None
    if covariance_matrix is not None:
        try:
            held = set(np.asarray(fix_for(final_free), dtype=int).tolist())
            free_idx = [i for i in range(len(p)) if i not in held]
            it = free_idx.index(mod_p_len + THEORY_FREE_FIELDS.index('theta_urad'))
            ib = free_idx.index(mod_p_len + THEORY_FREE_FIELDS.index('B_s'))
            cv = np.asarray(covariance_matrix, dtype=float)
            corr = float(cv[it, ib] / np.sqrt(cv[it, it] * cv[ib, ib]))
            print(f"    correlation(theta, B_s) = {corr:+.3f}   (both set the "
                  f"width; |corr| near 1 means this spectrum does not separate them)")
        except Exception:
            corr = None

    # The fast closed-form equivalent, for reference: same E^-4 tails, no
    # dynamical-diffraction evaluation. Reported, not stored.
    rational = None
    try:
        rational, r_rms, r_max = smst.physical_to_rational(INS, n_terms=n_rational)
        print(f"[Instrumental function: theory] {n_rational}-term rational reduction: "
              f"rms {100 * r_rms:.2f} %, max {100 * r_max:.2f} % of the peak")
        print("    " + " ".join("%.6g" % v for v in smst.decode_rational(rational)))
    except Exception as e:
        print(f"[Instrumental function: theory] rational reduction failed: {e}")

    # The conventional stand-in that goes into INSexp.txt / #@INSexp, so the
    # Gaussian-sum description a reader without #@INSth sees still describes
    # THIS source. Its rms is the error such a reader inherits (and its tails are
    # Gaussian, where the real ones go as v^-4 -- that is the whole point).
    n_gauss = 3
    try:
        # "Set number of lines to reconstruct the instrumental function", clamped:
        # more than a handful of Gaussians on a shape this smooth is unstable and
        # buys nothing, and this is only a fallback representation.
        n_gauss = min(max(int(app.instrumental_number.text()), 1), 6)
    except Exception:
        pass
    gauss, g_rms, g_max = None, np.nan, np.nan
    try:
        gauss, g_rms, g_max = smst.physical_to_gaussians(INS, n_terms=n_gauss)
        print(f"[Instrumental function: theory] {n_gauss}-Gaussian conventional stand-in "
              f"for INSexp.txt: rms {100 * g_rms:.2f} %, max {100 * g_max:.2f} % "
              f"of the peak")
    except Exception as e:
        print(f"[Instrumental function: theory] could not build the Gaussian stand-in "
              f"({e}); INSexp.txt is left as it was")

    F = INSSS(A, p)
    JN_save = int(JN)
    JN = JN * HIRES_INTEGRATION_FACTOR
    F2 = INSSS(A, p)
    JN = JN_save

    # Convergence of the transmission integral. The integration grid is uniform
    # in EE with E = MulCo*v + x0*MulCo + log((1+EE)/(1-EE)), which was chosen
    # for a compact, smooth sum of Gaussians. A theoretical source that is wide
    # AND structured -- two resolved peaks with a deep valley, which is what low
    # temperature near the Bragg peak gives -- needs many more points: measured
    # on this model, JN = 32 reaches 0.1 % of the dip depth at the usual
    # operating points (the same as the Gaussian sum), but the B_s ~ 2 T,
    # theta < 20 urad corner still has ~1 % error at JN = 512. Silently returning
    # an under-integrated fit would look like a bad instrumental function, so
    # say it out loud.
    span = float(np.max(F) - np.min(F))
    integ_err = float(np.max(np.abs(F2 - F)) / span) if span > 0 else np.nan
    print(f"[Instrumental function: theory] integration check: the same model at "
          f"JN*{HIRES_INTEGRATION_FACTOR} differs by {100 * integ_err:.3f} % "
          f"of the dip depth (JN = {JN})")
    if np.isfinite(integ_err) and integ_err > 0.01:
        print(f"[Instrumental function: theory] WARNING: the transmission integral "
              f"is NOT converged at JN = {JN}. This source is wide and "
              f"structured; raise 'number of points for full transmission "
              f"integral' (Supp menu) until this falls below ~0.1 %, or the fit "
              f"is reporting quadrature error as instrumental-function error.")

    write_accurate_instrumental(app, INS)
    insexp_path = os.path.join(app.params_dir, 'INSexp.txt')
    # THE GAUSSIAN DESCRIPTION IS *NOT* OVERWRITTEN here any more.
    #
    # This used to write the best Gaussian-sum stand-in for the shape just
    # found, so that a reader which only knows #@INSexp still saw the same
    # source. That was reasonable when only one description was "current", and
    # it is destructive now: it replaced a Gaussian sum the user had fitted with
    # a derived stand-in, which is the mirror image of the conventional search
    # deleting INSth.txt. Both descriptions are kept independently so that
    # switching between them costs nothing and neither can be lost by running
    # the other search.
    #
    # The cost is that #@INSexp in a .dat converted afterwards may describe an
    # older state of the source than #@INSth does. That is the lesser evil: a
    # slightly stale second opinion, rather than silently destroying a fit.
    if gauss is not None:
        print(f"[Instrumental function: theory] {len(gauss) // 3}-Gaussian "
              f"stand-in computed but NOT written: {insexp_path} holds the "
              f"user's own Gaussian description and is left alone")
    instrumental_int_path = os.path.join(app.params_dir, 'INSint.txt')
    with open(instrumental_int_path, "w") as f:
        f.write(str(MulCo) + ' ')
        f.write(str(x0) + ' ')
    print(f"[Instrumental function: theory] {smst.describe_ins(INS)}")

    return {
        'p': p,
        'er': er,
        'hi2': hi2,
        'mod_p_len': mod_p_len,
        'x0': x0,
        'MulCo': MulCo,
        'G': None,
        'insexp_path': insexp_path,
        'mode': mode,
        'CMS_ch': 0,
        'A': A,
        'B': B,
        'F': F,
        'F2': F2,
        'file': os.path.basename(file),
        'theory': theory,
        'theory_err': theory_err,
        'theory_fixed': fixed,
        'INS': INS,
        'rational': rational,
        'gaussian': gauss,
        'gaussian_rms': g_rms,
        'fwhm': fwhm,
        'centre': centre,
        'corr_theta_Bs': corr,
        'integration_error': integ_err,
    }
