"""Calibrate the drive and find the CMS source's G, Δθ and f_LM in ONE fit, of
two raw spectra measured with one source, geometry and drive: pure alpha-Fe and
a single line (docs/cosine_smearing.md, §7: parts H3, H4 and J).

Δθ moves the alpha-Fe lines out, so an axis is the transducer's only when it is
calibrated with the right Δθ, and Δθ and G are found on a calibrated axis. Done
one after the other the two need rounds (part H3). Here they are fitted
together, on both raw spectra at once: the drive curve, its count distortion,
G, Δθ, f_LM and both absorbers. The drive's scale then takes the line positions
and Δθ the widths, as an axis scale does in part H4; the saturated single line
fixes f_LM, which takes Δθ off the alpha-Fe's saturation (part J).

1. Start: Calibration.Calibration on the alpha-Fe with the start G and Δθ
   (the dialog's; START_G, START_DTHETA when the fields hold none). It decides
   the drive's waveform, sine or triangle -- kept from here on -- and gives
   the start of everything else: the drive curve (with the phase of the
   recording), its count distortion, the alpha-Fe's T, texture and count rate.
2. The single line on its own, on that drive, with G, Δθ and f_LM held: its
   thickness from a coarse scan, then its T, shift and count rate fitted.
3. Everything together, all free at once: the drive curve (3), its count
   distortion (5, one for both spectra: one source, one geometry), G, Δθ,
   f_LM, and each spectrum's own T and count rate, the alpha-Fe's texture and
   the single line's shift.
4. Both spectra folded with the drive found (the fold window from it);
   Calibration.dat written from the alpha-Fe as Calibration writes it, and G,
   Δθ, d and D into GCMS.txt.

The references: alpha-Fe, a polarized Sextet at the alpha-Fe field with δ 0
(the axis' zero), the natural width and no Gaussian of its own; a single line,
a Singlet at the natural width with no Gaussian of its own. Each spectrum's
non-resonant counts are the source's (as dtheta_alpha_fe_reference has them):

    Nnr = Ns (1 - f_LM) / f_LM + C_high,

C_high the 122/136 keV quanta counted per channel; they follow the source's
distance, so the count distortion multiplies them too, while the detector's
static noise is added flat. C_high and the static noise are measured, one pair
per spectrum (the counting times differ); with both 0, f_LM would take them up.
The fit takes 1 - cos Δθ, not Δθ: every spectrum is flat in Δθ at 0 (the
smearing grows as Δθ^2), and a fit that reached it could not come back.
"""
import os

import numpy as np

import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.Calibration import (Calibration, MulCoCMS, _fold_sinusoidal, _fold_triangular, _load_raw_counts,
                                  _write_calibration_dat)
from syncmoss.constants import ALPHA_FE_FIELD, NAT_WIDTH, number_of_baseline_parameters as NB
from syncmoss.instrumental_io import compute_norm, drive_rest_velocity, r_ratio_of, write_gcms_file
from syncmoss.spectrum_io import estimate_edge_background, load_spectrum

# Where G and Δθ start when their fields hold no usable value (not a number, 0
# or less, Δθ from π/2): a usual source and geometry
START_G = 0.09          # mm/s
START_DTHETA = 0.1      # rad

# The raw formats Calibration reads (Calibration._load_raw_counts)
RAW_SUFFIXES = ('.mca', '.cmca', '.ws5', '.w98', '.moe', '.m1', '.mcs')

# The single line's thicknesses tried before its own fit: decades, not steps,
# since a dip saturates
THICKNESS_SCAN = (0.3, 1.0, 3.0, 10.0, 30.0)

# f_LM is kept off 0 and 1, where Nnr = Ns (1 - f_LM)/f_LM is infinite or 0
F_LM_MARGIN = 1e-3

# The fitted vector: the drive curve (Calibration's sin_cal / lin_cal
# parameters); the count distortion, the counts times
#     1 + a1 (ch - c1)^2 + o1  in the first half-sweep,  1 + a2 (ch - c2)^2  in the second
# (a1, c1, o1, a2, c2), the solid angle changing with the source's position;
# G, 1 - cos Δθ and f_LM; alpha-Fe's T, texture A and count rate Ns; the single
# line's T, shift and Ns
DRIVE = slice(0, 3)
DISTORTION = slice(3, 8)
G_I, U_I, F_I = 8, 9, 10
ALPHA_T, ALPHA_A, ALPHA_NS = 11, 12, 13
LINE_T, LINE_SHIFT, LINE_NS = 14, 15, 16
N_FIT = 17
LINE_ONLY = [LINE_T, LINE_SHIFT, LINE_NS]

# The joint model: alpha-Fe's baseline and Sextet, then the single line's
# baseline (Nbaseline) and Singlet
MODEL = ['Sextet', 'Nbaseline', 'Singlet']
LINE_START = NB + 14


def start_values(G, dtheta):
    """G and Δθ as the procedure starts from them: the fields' values, or
    START_G / START_DTHETA for one that is not usable."""
    try:
        G = float(G)
    except (TypeError, ValueError):
        G = START_G
    try:
        dtheta = float(dtheta)
    except (TypeError, ValueError):
        dtheta = START_DTHETA
    if not 0 < G < 1:
        G = START_G
    if not 0 < dtheta < np.pi / 2:
        dtheta = START_DTHETA
    return G, dtheta


def refusal(files):
    """Why the procedure cannot start on *files* (alpha-Fe first, then the
    single line), or None."""
    if len(files) < 2:
        return "needs two raw spectra in the path box: alpha-Fe first, then the single line"
    for f in files[:2]:
        if not os.path.exists(f):
            return f"{os.path.basename(f)} does not exist"
        if not f.lower().endswith(RAW_SUFFIXES):
            return f"{os.path.basename(f)} is not a raw spectrum ({', '.join(RAW_SUFFIXES)})"
    channels = [len(_load_raw_counts(f)) for f in files[:2]]
    if channels[0] != channels[1]:
        return (f"the two spectra have {channels[0]} and {channels[1]} channels: "
                "they must come from the same drive")
    return None


def source_refusal(source):
    """Why the source's numbers cannot be used, or None: f_LM in (0, 1), C_high
    and the static noise not negative, one of each per spectrum."""
    if not 0 < source['f_LM'] < 1:
        return "f_LM must be between 0 and 1"
    if len(source['C_high']) != 2 or len(source['C_static']) != 2:
        return "C_high and the static noise are needed for both spectra"
    if min(source['C_high']) < 0 or min(source['C_static']) < 0:
        return "C_high and the static noise must not be negative"
    return None


def counts(x, p, model, lengths, G, dtheta, r_ratio, JN, pool):
    """The counts of *model* (parameters p) at the velocities x, with the CMS
    source of Gaussian width G and smearing Δθ: the smearing and its Norm
    prepared for these velocities, each spectrum from where its drive is at
    rest, as every CMS search prepares them (instrumental_io.instrumental)."""
    x = np.asarray(x, dtype=float)
    pieces = [x] if lengths is None else np.split(x, np.cumsum(lengths)[:-1])
    v_center = [drive_rest_velocity(piece) for piece in pieces]
    smearing = m5.cms_smearing(x, JN, MulCoCMS, G, 1, dtheta, v_center if lengths else v_center[0], r_ratio,
                               lengths)
    source = {'x0': 0, 'MulCo': MulCoCMS, 'INS': G, 'Met': 1}
    if lengths is None:
        norm = compute_norm(pool, JN, source, x, smearing)
    else:
        norm = [compute_norm(pool, JN, source, piece, s)
                for piece, s in zip(pieces, smearing or [None] * len(pieces))]
    return np.asarray(m5.TI(x, np.asarray(p, dtype=float), model, JN, pool, 0, MulCoCMS, G, [0], [0], Met=1,
                            Norm=norm, lengths=lengths, smearing=smearing), dtype=float)


def distortion(n_ch, d):
    """The count distortion's factor at every channel (see DISTORTION)."""
    ch = np.arange(n_ch, dtype=float)
    first = ch < n_ch // 2
    return np.where(first, 1 + d[0] * (ch - d[1]) ** 2 + d[2], 1 + d[3] * (ch - d[4]) ** 2)


def _baseline(ns, f_lm, c_high):
    """A spectrum's baseline: Ns, and Nnr the source's (see the module docstring)."""
    return [ns, 0, 0, 0, ns * (1 - f_lm) / f_lm + c_high, 0, 0, 0]


def alpha_fe_parameters(q, c_high):
    """alpha-Fe's baseline and polarized Sextet (T, δ 0, ε 0, the alpha-Fe
    field, the natural width, no Gaussian, θk 90, φh 0, texture A, ..., I1/I3 3)."""
    return (_baseline(q[ALPHA_NS], q[F_I], c_high)
            + [q[ALPHA_T], 0, 0, ALPHA_FE_FIELD, NAT_WIDTH, 0, 90, 0, q[ALPHA_A], 0, 0, 0, 0, 3])


def single_line_parameters(q, c_high):
    """The single line's baseline and Singlet (T, shift, the natural width, no
    Gaussian)."""
    return _baseline(q[LINE_NS], q[F_I], c_high) + [q[LINE_T], q[LINE_SHIFT], NAT_WIDTH, 0]


def raw_counts(q, velocity, n_ch, source, r_ratio, JN, pool, line_only=False):
    """Both raw spectra (alpha-Fe's channels, then the single line's) for the
    fitted vector q, or the single line's alone: the absorbers on the drive's
    velocities, times the count distortion, plus each spectrum's static
    noise."""
    c_high, c_static = source['C_high'], source['C_static']
    v = velocity(np.arange(n_ch, dtype=float), q[DRIVE])
    w = distortion(n_ch, q[DISTORTION])
    G, dtheta = q[G_I], float(np.arccos(1 - q[U_I]))
    if line_only:
        return (counts(v, single_line_parameters(q, c_high[1]), ['Singlet'], None, G, dtheta, r_ratio, JN, pool) * w
                + c_static[1])
    joint = counts(np.concatenate((v, v)), alpha_fe_parameters(q, c_high[0]) + single_line_parameters(q, c_high[1]),
                   MODEL, [n_ch, n_ch], G, dtheta, r_ratio, JN, pool)
    return joint * np.concatenate((w, w)) + np.repeat(np.asarray(c_static, dtype=float), n_ch)


def _resonant_start(level, f_lm, c_high, c_static, name):
    """Ns to start from: the counts off resonance less C_high and the static
    noise, times f_LM (as dtheta_alpha_fe_reference starts)."""
    ns = (level - c_high - c_static) * f_lm
    if ns <= 0:
        raise ValueError(f"{name}: C_high + static noise is not below its counts off resonance ({level:g})")
    return ns


def start_vector(details, line_counts, line_folded, source, G, dtheta, r_ratio, JN, pool):
    """The fitted vector to start from: the drive, its distortion and alpha-Fe
    from the start calibration (*details*, Calibration's), the single line
    fitted on its own on that drive (G, Δθ and f_LM held)."""
    pS, n_ch, f_lm = details['pS'], len(line_counts), source['f_LM']
    (ca, cl), (sa, sl) = source['C_high'], source['C_static']
    level = pS[11] + pS[12]                      # alpha-Fe's counts off resonance
    q = np.zeros(N_FIT)
    q[DRIVE] = pS[0:3]
    # Calibration's distortion is added counts, p5 (c1 - ch)^2 + p6 and p8 (c2 - ch)^2
    q[DISTORTION] = [pS[5] / level, pS[4], pS[6] / level, pS[8] / level, pS[7]]
    q[G_I], q[U_I], q[F_I] = G, 1 - np.cos(dtheta), f_lm
    q[[ALPHA_T, ALPHA_A]] = [pS[9], pS[10]]
    q[ALPHA_NS] = _resonant_start(level, f_lm, ca, sa, "alpha-Fe")
    # the single line: the edges' counts, the shift where its folded dip is
    # deepest (a 5-point average), the best thickness of THICKNESS_SCAN
    q[LINE_NS] = _resonant_start(estimate_edge_background(line_counts), f_lm, cl, sl, "single line")
    x_f, B_f = line_folded
    smooth = np.convolve(B_f, np.ones(5) / 5, mode='same')
    q[LINE_SHIFT] = float(x_f[2:-2][np.argmin(smooth[2:-2])])

    def line_alone(trial):
        return raw_counts(trial, details['velocity'], n_ch, source, r_ratio, JN, pool, line_only=True)

    def chi2_at(t):
        trial = np.copy(q)
        trial[LINE_T] = t
        return float(np.sum((line_counts - line_alone(trial)) ** 2 / np.maximum(line_counts, 1)))
    q[LINE_T] = min(THICKNESS_SCAN, key=chi2_at)
    fix = np.setdiff1d(np.arange(N_FIT), LINE_ONLY)
    q = mi.minimi_hi(lambda _x, trial: line_alone(trial), np.arange(n_ch), line_counts, q, fix=fix,
                     bounds=_bounds(x_f), MI=20, MI2=10, tau0=1e-4, eps=1e-6)[0]
    return np.array(q, dtype=float)


def _bounds(line_axis):
    bounds = np.array([[-np.inf] * N_FIT, [np.inf] * N_FIT], dtype=float)
    bounds[0][G_I], bounds[1][G_I] = 0.001, 1.0
    bounds[0][U_I], bounds[1][U_I] = 0.0, 1.0
    bounds[0][F_I], bounds[1][F_I] = F_LM_MARGIN, 1 - F_LM_MARGIN
    for i in (ALPHA_T, LINE_T):
        bounds[0][i] = 0.001
    for i in (ALPHA_NS, LINE_NS):
        bounds[0][i] = 0.0
    bounds[0][LINE_SHIFT], bounds[1][LINE_SHIFT] = float(np.min(line_axis)), float(np.max(line_axis))
    return bounds


def fit_together(q, data, velocity, n_ch, line_axis, source, r_ratio, JN, pool):
    """Everything fitted together on both raw spectra (*data*: alpha-Fe's
    counts, then the single line's). Returns (q, errors, chi2)."""
    def f(_x, trial):
        return raw_counts(trial, velocity, n_ch, source, r_ratio, JN, pool)
    x, bounds = np.arange(2 * n_ch), _bounds(line_axis)
    q, errors, chi2 = np.array(q, dtype=float), None, np.inf
    # until a pass no longer lowers chi2, as the instrumental-function searches refine
    for _ in range(3):
        p, er, c2, _cov = mi.minimi_hi(f, x, data, q, bounds=bounds, MI=20, MI2=20, tau0=1e-4, eps=1e-6)
        if errors is not None and not c2 < chi2:
            break
        q, errors, chi2 = np.array(p, dtype=float), np.array(er, dtype=float), float(c2)
    return q, errors, chi2


def _fold(method, v_axis, raw, fit, drive):
    """A raw spectrum and its fit folded as Calibration folds them."""
    if method == 0:
        return _fold_sinusoidal(v_axis, raw, fit, drive[2])
    return _fold_triangular(v_axis, raw, fit, drive)


def calibrate_source(params_dir, files, pool, JN, vel_start, G, dtheta, diameters, source, report=None,
                     cancelled=None):
    """Calibrate the drive on the alpha-Fe and find G, Δθ and f_LM, in one fit
    with the single line (see the module docstring). *files*: the alpha-Fe and
    the single-line raw spectra; *diameters*: the source's and the aperture's
    (d, D), held; *source*: 'f_LM' to start from, and 'C_high' and 'C_static'
    per channel, one for each spectrum (alpha-Fe, single line). Writes
    Calibration.dat and GCMS.txt (G, Δθ, d, D) into *params_dir*.

    *report*, when given, takes a progress line; *cancelled* (a callable, the
    window's Interrupt) stops it between steps -- TI stops it inside them.

    Returns 'G', 'dtheta' and 'f_LM' with their 'errors', the 'start' G and Δθ,
    the 'drive' found, and the fit folded for the plot: 'A' (the folded axis,
    once per spectrum), 'B', 'F' (the model), 'F2' (the model at 4 JN, the cyan
    line), 'p' (each spectrum's model, its baseline as folded), 'hi2',
    'lengths', 'starts', 'files', 'calibration_path'.
    """
    def check():
        if cancelled is not None and cancelled():
            raise m5.FitInterrupted()

    if report is None:
        def report(_line):
            pass

    refused = refusal(files) or source_refusal(source)
    if refused:
        raise ValueError(refused)
    files = list(files[:2])
    r_ratio = r_ratio_of(*diameters)
    G, dtheta = start = start_values(G, dtheta)
    calibration_path = os.path.join(params_dir, 'Calibration.dat')

    report(f"Start: calibrating with G {G:.4f} mm/s, Δθ {dtheta:.4f} rad...")
    details = {}
    Calibration(params_dir, files[0], pool, 1, np.array([G]), JN, 0.0, MulCoCMS, Vel_start=vel_start, GCMS=G,
                dtheta=dtheta, r_ratio=r_ratio, details=details)
    check()
    report("Start: the single line on its own...")
    line_raw = np.asarray(_load_raw_counts(files[1]), dtype=float)
    alpha_raw = np.asarray(details['counts'], dtype=float)
    A_f, B_f = load_spectrum(None, [files[1]], calibration_path=calibration_path)
    q = start_vector(details, line_raw, (A_f[0], B_f[0]), source, G, dtheta, r_ratio, JN, pool)
    check()
    report("Calibration, G, Δθ and f_LM together...")
    n_ch, velocity = len(alpha_raw), details['velocity']
    q, errors, chi2 = fit_together(q, np.concatenate((alpha_raw, line_raw)), velocity, n_ch, A_f[0], source,
                                   r_ratio, JN, pool)
    check()

    G, dtheta, f_lm = float(q[G_I]), float(np.arccos(1 - q[U_I])), float(q[F_I])
    # σ(Δθ) from σ(1 - cos Δθ): d(1 - cos Δθ) = sin Δθ dΔθ
    dtheta_error = float(errors[U_I] / np.sin(dtheta)) if dtheta > 0 else float('nan')
    v_axis = velocity(np.arange(n_ch, dtype=float), q[DRIVE])

    def folded(jn):
        model = raw_counts(q, velocity, n_ch, source, r_ratio, jn, pool)
        return [_fold(details['method'], v_axis, raw, fit, q[DRIVE])
                for raw, fit in ((alpha_raw, model[:n_ch]), (line_raw, model[n_ch:]))]
    alpha, line = folded(JN)
    v_f, n1, n2 = alpha[0], alpha[4], alpha[5]
    _write_calibration_dat(params_dir, details['method'], n1, n2, v_f, alpha[1])
    write_gcms_file(params_dir, G, dtheta, *diameters)
    fine = folded(4 * JN)
    # each spectrum's model with its baseline as folded (two channels to a
    # point), the static noise in its Nnr: the plot fills down to the baseline
    shown = np.array(alpha_fe_parameters(q, source['C_high'][0]) + single_line_parameters(q, source['C_high'][1]),
                     dtype=float)
    for section, static in zip((0, LINE_START), source['C_static']):
        shown[section] *= 2
        shown[section + 4] = 2 * (shown[section + 4] + static)
    return {
        'G': G, 'dtheta': dtheta, 'f_LM': f_lm,
        'errors': (float(errors[G_I]), dtheta_error, float(errors[F_I])), 'start': start,
        'drive': np.array(q[DRIVE]), 'method': details['method'], 'A': np.concatenate((v_f, line[0])),
        'B': np.concatenate((alpha[1], line[1])), 'F': np.concatenate((alpha[2], line[2])),
        'F2': np.concatenate((fine[0][2], fine[1][2])), 'p': shown, 'hi2': chi2, 'lengths': [len(v_f), len(line[0])],
        'starts': [0, LINE_START], 'files': files, 'calibration_path': calibration_path,
    }
