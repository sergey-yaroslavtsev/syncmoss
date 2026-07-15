# -*- coding: utf-8 -*-
"""Automatic velocity calibration of a Moessbauer drive from an alpha-Fe spectrum.

@author: YAROSLAVTSEV S

The MIT license follows:

Copyright (c) European Synchrotron Radiation Facility (ESRF)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------

WHY THIS MODULE IS COMPLICATED
==============================
The calibration is fully automatic: the user only points at a raw file of a
standard alpha-Fe absorber. NOTHING is preselected -- not the file format, not
the drive waveform (sinusoidal or triangular), not the velocity amplitude,
direction or phase, not the impurity content of the reference foil. Every one
of those unknowns is detected from the data itself, which is why the pipeline
has many steps. The steps, in order (mirrored by the STEP banners in
:func:`Calibration`):

1.  **Fit function** -- build the transmission-integral model of the alpha-Fe
    spectrum for the measurement mode: SMS (synchrotron source, ``VVV == 3``,
    multi-line instrumental function ``INS``) or CMS (conventional source,
    ``VVV == 1``, single Gaussian width ``GCMS``).
2.  **Load** the raw counts; seven file formats are auto-detected by extension
    (.mca/.cmca/.ws5/.w98/.moe/.m1/.mcs).
3.  **Phase detection** -- the recording is not synchronised with the drive, so
    the channel at which the velocity sweep starts is unknown. It is found by
    sliding the two half-sweeps against each other until they mirror-match.
4.  **Reference model** -- alpha-Fe sextet with the six known line velocities;
    ``Vel_start`` orients the sweep (up-down vs down-up).
5.  **Coarse search** -- the velocity amplitude is unknown, so every pair of
    sextet lines is hypothetically assigned to the two deepest absorption
    minima; each hypothesis fixes the drive amplitude/offset, a quick fit
    scores it, and the best chi-square wins. Done with the sinusoidal drive
    model always, and additionally with the triangular (linear) model for CMS;
    the waveform is then chosen by chi-square (SMS drives are always
    sinusoidal).
6.  **Iterative refinement** (4 rounds) -- fit the spectrum on the current
    velocity axis, predict the channel of each of the 12 line occurrences
    (6 lines x 2 half-sweeps), refit the drive-curve parameters to those
    (channel, velocity) pairs, rebuild the axis.
7.  **Distortion fit** -- the source-absorber solid angle changes with the
    drive position, adding a parabolic count modulation that differs between
    the half-sweeps; it is fitted from their difference.
8.  **Final global fit** -- drive parameters, parabolic distortion, absorber
    intensities (and, for SMS, the impurity components: a second sextet and
    the Be-window doublet from ``Be.txt``) are all refined together.
9.  **Folding** -- each velocity is visited twice per period; the two visits
    are averaged onto a single velocity axis. For SMS the multi-line
    instrumental function also shifts the apparent line positions; the
    computed ``INS_shift`` corrects that.
10. **Output** -- diagnostic figure ``calibr.png`` (SMS), a summary printout,
    and ``Calibration.dat`` (velocity/counts table consumed by the rest of
    SYNCmoss); the folded arrays are returned to the GUI.

Parameter-vector layouts used throughout (``nbp`` = number_of_baseline_parameters = 8):

* Spectrum model ``p`` / ``p00`` / ``pCAL`` -- ``[0..nbp-1]`` baseline
  (``[0]`` count rate of half-sweep 1, ``[4]`` count rate of half-sweep 2 for
  CMS), then one polarized Sextet block of 14:
  ``T(+0) d(+1) e(+2) H(+3) L(+4) G(+5) theta_k(+6) phi_h(+7) A(+8) A_m(+9)
  a+(+10) a-(+11) GH(+12) I13(+13)``.
  ``theta_k=90, phi_h=0, A=0`` make the polarized Sextet identical to the old
  scalar one (isotropic powder); ``A_m=0`` is also a no-op since
  ``theta_k=90 -> n_z=0`` kills the Faraday term.
* Drive-curve parameters ``ps`` (3): sinusoidal ``[amplitude, phase(rad),
  source shift]``; triangular ``[v at ch 0, v at last ch, v step/channel]``.
* Final-fit vector ``ps1`` / ``pS``: ``[0..2]`` drive curve, ``[3]`` global
  intensity scale, ``[4..6]`` parabola of half-sweep 1 (centre, curvature,
  offset), ``[7..8]`` parabola of half-sweep 2 (centre, curvature),
  ``[9]`` main sextet intensity, ``[10]`` texture parameter A, then
  ``[11]`` impurity sextet intensity (SMS) or ``[11..12]`` the two baseline
  count rates (CMS).
"""

import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.constants import number_of_baseline_parameters

# Effective absorber thickness multiplier used for the CMS transmission integral
# (the SMS one is passed in as MulCo, fitted by the instrumental-function step).
MulCoCMS = 0.28

# alpha-Fe sextet line positions in mm/s -- the absolute velocity references
# that the whole calibration is anchored to.
ALPHA_FE_LINE_VELOCITIES = np.array([-5.3123, -3.0760, -0.8397, 0.8397, 3.0760, 5.3123])

# Hyperfine field of alpha-Fe (Tesla) and the T -> (outer line splitting, mm/s)
# conversion factor for 57Fe: 33.04 T / 3.101 = 10.655 mm/s between lines 1 and 6.
ALPHA_FE_FIELD = 33.04
TESLA_PER_MMS = 3.101


# ========================================================================= #
#  Raw-file loading                                                         #
# ========================================================================= #

def _load_raw_counts(path):
    """Read a raw calibration spectrum and return its counts as a 1-D float array.

    The format is auto-detected from the file extension, because the user never
    tells us which spectrometer produced the file:

    * ``.mca`` / ``.cmca`` -- SPEC-style text: counts follow an ``@A`` marker,
      possibly wrapped over several lines, terminated by a ``#`` line.
      ``.cmca`` is a circular buffer whose first/last channels may need
      averaging; an odd trailing channel is dropped so the spectrum can be
      split into two equal half-sweeps.
    * ``.ws5`` / ``.w98`` -- WissEl text, one count per line (``#``/``<``
      comment lines skipped).
    * ``.moe`` -- like WissEl but header lines containing a ``.`` are skipped
      (counts are integers).
    * ``.m1`` -- multi-column text; counts are in column 5, first data row is
      a header and skipped.
    * ``.mcs`` / ``.Mcs`` -- binary: 256-byte header then uint32 counts.

    Raises ValueError for an unrecognised extension.
    """
    path = str(path)
    is_mca = path[-4:] == '.mca' or path[-5:] == '.cmca'
    is_text_or_bin = (path[-4:] in ('.ws5', '.w98', '.moe', '.mcs', '.Mcs')
                      or path[-3:] == '.m1')

    if is_mca:
        n_lines = len(open(path, 'r').readlines())
        with open(path, 'r') as f:
            blocks = []
            n = 0          # index of the data block being collected
            k = 0          # number of '@A' markers seen so far
            for _ in range(0, n_lines):
                for line in f:
                    if line.startswith("@A"):
                        k += 1
                    if k > n and line.startswith("@A"):
                        blocks.append(re.findall(r'[\d.]+', line[2:]))
                    if k > n and line.startswith("#"):
                        break
                    if k > n and line.startswith("@A") == 0:
                        blocks[n].extend(re.findall(r'[\d.]+', line[0:]))
                n += 1
        if path[-5:] == '.cmca':
            # Circular buffer: if recording stopped mid-cycle one end channel is
            # empty; average the two ends. (NOTE: kept verbatim from the original
            # code -- the entries are still strings here, so the ``== 0`` tests
            # never fire; do not "fix" without validating against real .cmca data.)
            if ((blocks[0][-1] == 0 and blocks[0][0] != 0)
                    or (blocks[0][0] == 0 and blocks[0][-1] != 0)):
                end_avg = (blocks[0][-1] + blocks[0][0]) / 2
                blocks[0][-1] = end_avg
                blocks[0][0] = end_avg
        if len(blocks[0]) % 2 == 1:
            blocks[0] = blocks[0][:-1]
        return np.array(blocks, dtype=float)[0]

    if is_text_or_bin:
        if path[-4:] == '.mcs' or path[-4:] == '.Mcs':
            with open(path, mode='rb') as f:
                f.read(256)  # skip the fixed-size binary header
                counts = np.fromfile(f, dtype=np.uint32)
            return np.array([counts], dtype=float)[0]

        with open(path, 'r') as f:
            values = []
            k = 0
            lines = (line.rstrip() for line in f)
            lines = (line for line in lines if line)          # skip blank lines
            for line in lines:
                if line.startswith('#') or line.startswith('<'):
                    continue                                   # column labels / headers
                if path[-3:] == '.m1':
                    while '  ' in line:
                        line = line.replace('  ', ' ')
                    column = line.split(' ')
                    if k > 0:                                  # first data row is a header
                        values.append(float(column[4]))
                    k += 1
                else:
                    column = line.split()
                    # .moe: header lines contain floats (with a '.'), counts are ints
                    if path[-4:] != '.moe' or not ('.' in str(column[0])):
                        values.append(float(column[0]))
                k += 1
        return np.array([values], dtype=float)[0]

    raise ValueError(
        f"unsupported calibration file type: {path!r} "
        "(expected .mca, .cmca, .ws5, .w98, .moe, .m1 or .mcs)")


# ========================================================================= #
#  Geometry helpers (pure functions)                                        #
# ========================================================================= #

def _detect_phase_offset(half1, half2):
    """Find the phase offset (in channels of the full spectrum) between the
    velocity drive and the recording.

    The two half-sweeps traverse the same velocities in opposite directions,
    so with zero phase ``half1`` equals ``half2`` reversed. The recording,
    however, starts at an arbitrary point of the drive period; this slides the
    halves against each other (both directions) and takes the shift with the
    best mirror match (minimal mean absolute difference). The result seeds the
    drive-curve phase in the coarse search -- without it the line-position
    hypotheses of step 5 would systematically miss.
    """
    best = np.sum(np.abs(half1 - half2[::-1])) / len(half1)
    phase = 0
    for i in range(1, int(len(half1) / 2)):
        trial = np.sum(np.abs(half1[i:] - half2[i:][::-1])) / len(half1[i:])
        if trial <= best:
            best = trial
            phase = -i
    if phase == 0:
        for i in range(1, int(len(half1) / 2)):
            trial = np.sum(np.abs(half1[:-i] - half2[:-i][::-1])) / len(half1[:-i])
            if trial <= best:
                best = trial
                phase = i
    return -phase * 2


def _deepest_line_channels(counts):
    """Channels of the two deepest absorption minima in the first half-sweep.

    The coarse search (step 5) needs two anchor points to hypothesise a drive
    amplitude. The deepest minima of the first and second quarter of the
    spectrum are (for alpha-Fe) two of the six sextet lines -- which two is
    unknown, hence the pair loop in the caller. If the same minimal count
    occurs in several channels the first occurrence inside the quarter is used.
    """
    quarter = int(len(counts) / 4)
    ch1_candidates = np.where(counts == min(counts[:quarter]))[0]
    ch_min1 = int(ch1_candidates[-1])
    for c in ch1_candidates:
        if c < quarter:
            ch_min1 = int(c)
            break
    ch2_candidates = np.where(counts == min(counts[quarter:2 * quarter]))[0]
    ch_min2 = int(ch2_candidates[-1])
    for c in ch2_candidates:
        if quarter <= c < 2 * quarter:
            ch_min2 = int(c)
            break
    return ch_min1, ch_min2


def _parabola(x, p):
    """``p[1]*(p[0]-x)**2 + p[2]`` -- the solid-angle count distortion model."""
    return p[1] * (p[0] - x) ** 2 + p[2]


def _subtract_distortion(counts, channels, pS):
    """Remove the fitted parabolic distortion from the raw counts.

    Each half-sweep has its own parabola (``pS[4..6]`` and ``pS[7..8]``, the
    second one sharing no offset); used for the diagnostic plot only.
    """
    half = int(len(channels) / 2)
    flat = np.array([float(0)] * len(channels))
    flat[:half] = counts[:half] - pS[5] * (channels[:half] - pS[4]) ** 2 - pS[6]
    flat[half:] = counts[half:] - pS[8] * (channels[half:] - pS[7]) ** 2
    return flat


# ========================================================================= #
#  Folding the two half-sweeps onto one velocity axis                       #
# ========================================================================= #

def _fold_sinusoidal(v_axis, counts, fit_curve, source_shift):
    """Fold a sinusoidal-drive spectrum: average the two visits of each velocity.

    A sinusoidal sweep turns around inside the channel range, so the fold
    points (velocity extrema) generally do NOT sit exactly at the ends/middle
    of the spectrum. ``n1``/``n2`` locate the matching turnaround channels
    (via the channel closest to the source velocity ``source_shift`` and its
    mirror in the second half); the spectrum then splits into three segments
    (before n1 / after n2 / between), each folded onto itself.

    Returns ``(v_folded, data_folded, fit_folded, fold_residual, n1, n2)``
    where ``fold_residual`` (difference of the two visits) should be pure
    noise if the calibration is right -- it is plotted as a diagnostic.
    """
    n = len(v_axis)
    half = int(n / 2)
    min1 = np.abs(v_axis[:half] - source_shift).argmin()
    min2 = np.abs(v_axis[half:] - v_axis[min1]).argmin() + half

    if min1 == (n - 1 - min2):
        n1, n2 = 0, n - 1
    elif min1 > (n - 1 - min2):
        n1, n2 = min1 - (n - min2) + 1, n - 1
    else:
        n1, n2 = 0, min2 + min1

    # Segment before n1 folded onto itself
    x_1h = (v_axis[:int(n1 / 2)] + v_axis[n1 - int(n1 / 2):n1][::-1]) / 2
    spc_1h = counts[:int(n1 / 2)] + counts[n1 - int(n1 / 2):n1][::-1]
    fit_1h = fit_curve[:int(n1 / 2)] + fit_curve[n1 - int(n1 / 2):n1][::-1]
    delt_1h = counts[:int(n1 / 2)] - counts[n1 - int(n1 / 2):n1][::-1]

    # Segment after n2 folded onto itself
    x_2h = (v_axis[n2 + 1:n2 + int((n - 1 - n2) / 2) + 1][::-1]
            + v_axis[n - int((n - 1 - n2) / 2):n]) / 2
    spc_2h = (counts[n2 + 1:n2 + int((n - 1 - n2) / 2) + 1][::-1]
              + counts[n - int((n - 1 - n2) / 2):n])
    fit_2h = (fit_curve[n2 + 1:n2 + int((n - 1 - n2) / 2) + 1][::-1]
              + fit_curve[n - int((n - 1 - n2) / 2):n])
    delt_2h = (counts[n2 + 1:n2 + int((n - 1 - n2) / 2) + 1][::-1]
               - counts[n - int((n - 1 - n2) / 2):n])

    # Central segment n1..n2 folded onto itself
    x_3h = (v_axis[n1:n1 + int((n2 - n1 + 1) / 2)]
            + v_axis[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]) / 2
    spc_3h = (counts[n1:n1 + int((n2 - n1 + 1) / 2)]
              + counts[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1])
    fit_3h = (fit_curve[n1:n1 + int((n2 - n1 + 1) / 2)]
              + fit_curve[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1])
    delt_3h = (counts[n1:n1 + int((n2 - n1 + 1) / 2)]
               - counts[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1])

    v_folded = np.concatenate((np.concatenate((x_1h, x_2h)), x_3h))
    data_folded = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
    fit_folded = np.concatenate((np.concatenate((fit_1h, fit_2h)), fit_3h))
    fold_residual = np.concatenate((np.concatenate((delt_1h, delt_2h)), delt_3h))
    return v_folded, data_folded, fit_folded, fold_residual, n1, n2


def _fold_triangular(v_axis, counts, fit_curve, pS):
    """Fold a triangular-drive spectrum.

    The two linear ramps may not reach the same extreme velocity (``pS[0]`` vs
    ``pS[1]``); the channels outside the common velocity range are discarded
    (``n1``/``n2`` mark the usable window) and the remaining pairs are averaged
    symmetrically from the ends inward.

    Returns ``(v_folded, data_folded, fit_folded, fold_residual, n1, n2)``.
    """
    n = len(v_axis)
    half = int(n / 2)
    if pS[0] * pS[2] == pS[1] * pS[2]:
        min1, min2 = 0, n - 1
    elif pS[0] * pS[2] > pS[1] * pS[2]:
        min1 = np.abs(v_axis[:half] - pS[1]).argmin()
        min2 = n - 1
    else:
        min1 = 0
        min2 = np.abs(v_axis[half:] - pS[0]).argmin() + half
    n1, n2 = min1, min2
    n_cut = 2 * (n1 + (n - 1 - n2))          # channels dropped from both ends

    # int() also swallows a possibly odd number of remaining points.
    n_fold = int((n - n_cut) / 2)
    i = np.arange(n_fold)
    v_folded = (v_axis[n1 + i] + v_axis[n2 - i]) / 2
    data_folded = counts[n1 + i] + counts[n2 - i]
    fit_folded = fit_curve[n1 + i] + fit_curve[n2 - i]
    fold_residual = counts[n1 + i] - counts[n2 - i]
    return v_folded, data_folded, fit_folded, fold_residual, n1, n2


# ========================================================================= #
#  Output                                                                   #
# ========================================================================= #

def _write_calibration_dat(dir_path, method, n1, n2, v_folded, data_folded):
    """Write ``Calibration.dat``: the folded velocity/counts table.

    The header line records the drive waveform (``sin``/``lin``) and the fold
    window ``n1 n2`` so that measurement spectra recorded with the same drive
    can be folded identically (see spectrum_io.load_spectrum).
    """
    rpath = os.path.join(str(dir_path), 'Calibration.dat')
    with open(rpath, "w") as f:
        f.write(str('#') + '\t' + str('lin ') * (method == 1)
                + str('sin ') * (method == 0)
                + '\t' + str(n1) + '\t' + str(n2) + '\n')
        for i in range(0, int(len(v_folded))):
            f.write(str(v_folded[i]) + '\t' + str(data_folded[i]) + '\n')
        f.write('\n')


def _save_sms_diagnostic_plot(dir_path, channels, counts, final_curve, flat_counts,
                              v_axis, v_folded, data_folded, fold_residual,
                              pS, baseline, method, n1, n2):
    """Save ``calibr.png``: raw data + fit vs channel, folded halves vs velocity,
    the fold residual, and the key numbers (amplitude, distortion, impurity)."""
    fig, ax = plt.subplots(dpi=300)
    plt.plot(channels, counts, 'm')
    plt.plot(channels, final_curve, 'b')
    plt.plot(channels, counts - final_curve + 0.9 * min(counts), 'lime')
    offset = max(counts) - min(flat_counts) + 10 * np.sqrt(max(counts))
    half = int(len(channels) / 2)
    plt.plot(channels[:half] * 2, (counts + offset)[:half][::-1],
             'r', linestyle='None', marker='o', markersize=2)
    plt.plot(channels[:half] * 2, (counts + offset)[half:],
             'yellow', linestyle='None', marker='o', markersize=2)
    sax = ax.twiny()
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('channel', color='m')
    sax.set_xlabel('Velocity, mm/s', color='r')
    half_f = int(len(flat_counts) / 2)
    sax.plot(v_axis[:half_f], (flat_counts + 2 * offset)[:half_f],
             'r', linestyle='None', marker='o', markersize=2)
    sax.plot(v_axis[half_f:], (flat_counts + 2 * offset)[half_f:],
             'yellow', linestyle='None', marker='o', markersize=2)
    sax.plot(v_folded, fold_residual + 4 * offset, 'lime')
    sax.plot(v_folded, data_folded / 2 + 2 * offset, 'm', marker='o', markersize=1)

    ax.text(0, max(counts) + 4 * np.sqrt(max(counts)),
            'Va = %.3f mm/s' % pS[0], color='w', fontsize=8)
    ax.text(len(channels) / 2, max(counts) + 4 * np.sqrt(max(counts)),
            'ΔN0 = %.1f ' % (abs(pS[6]) / (baseline + pS[6] * (1 + np.sign(pS[6])) / 2) * 100) + '%',
            color='w', fontsize=8, horizontalalignment='center')
    ax.text(len(channels), max(counts) + 4 * np.sqrt(max(counts)),
            'impurity %.1f ' % (pS[11] / (pS[9] + pS[11]) * 100) + '%',
            color='w', fontsize=8, horizontalalignment='right')
    ax.text(len(channels) / 2, max(counts) + 10 * np.sqrt(max(counts)),
            str('lin ') * (method == 1) + str('sin ') * (method == 0)
            + str(n1) + str(' ') + str(n2),
            color='r', fontsize=8, horizontalalignment='center')

    fig.savefig(os.path.join(dir_path, 'calibr.png'), bbox_inches='tight')
    plt.close()


# ========================================================================= #
#  Main entry point                                                         #
# ========================================================================= #

def Calibration(dir_path, Cal_file, pool, VVV, INS, JN, x0, MulCo, Vel_start=1, GCMS=0.1):
    """Fit the velocity calibration of a standard alpha-Fe absorber spectrum.

    Loads the calibration spectrum ``Cal_file``, auto-detects everything about
    the measurement (see the module docstring for the step-by-step pipeline)
    and writes ``Calibration.dat`` + ``calibr.png`` into ``dir_path``.

    Used by syncmoss_main.py (CalibrationThread) to build the
    channel -> velocity scale.

    Args:
        dir_path: Writable directory holding ``Be.txt`` and receiving
            ``Calibration.dat`` / ``calibr.png``.
        Cal_file: Path of the raw calibration spectrum (format by extension).
        pool: Shared multiprocessing pool for the transmission integral.
        VVV: Experimental method -- ``1`` conventional source (CMS / MS mode),
            ``3`` synchrotron source (SMS).
        INS: SMS multi-line instrumental function (``#@INSexp``); ignored for
            CMS, which uses a single Gaussian of width ``GCMS`` instead.
        JN: Number of integration nodes for the transmission integral.
        x0: SMS instrumental-function shift (``#@INSint``); unused for CMS.
        MulCo: SMS effective-thickness multiplier (``#@INSint``); CMS uses the
            module constant ``MulCoCMS``.
        Vel_start: Sweep orientation: ``1`` velocity starts at its maximum
            (down-up), ``0`` at its minimum (up-down).
        GCMS: Gaussian instrumental linewidth for CMS (the GUI GCMS box).

    Returns:
        tuple ``(v_folded, data_folded, fit_folded)`` -- the folded velocity
        axis (mm/s), the folded experimental counts and the folded fit curve.
    """
    print('experimental method VVV =', VVV)
    nbp = number_of_baseline_parameters
    counts = _load_raw_counts(Cal_file)
    n_ch = len(counts)
    channels = np.linspace(0, n_ch - 1, n_ch)          # 0, 1, ..., n_ch-1 as floats
    half1 = counts[:int(n_ch / 2)]
    half2 = counts[int(n_ch / 2):]
    if len(half2) > len(half1):                        # odd channel count
        half2 = half2[1:]

    # --------------------------------------------------------------------- #
    # STEP 1: transmission-integral fit function for the measurement mode.  #
    # The closures below deliberately LATE-BIND ``model``, ``Norm`` and     #
    # ``JN``: those are reassigned further down (final model with impurity  #
    # components, finer integration grid) and ``fit_func`` must pick the    #
    # current values up at call time.                                       #
    # --------------------------------------------------------------------- #
    model = ['Sextet']
    pNorm = np.array([float(0)] * nbp)
    pNorm[0] = 1
    if VVV == 1:
        # CMS instrumental function is a single Gaussian width (GCMS), NOT the
        # multi-line SMS INS array; Met==1 expects exactly one width.
        INS = np.array([float(GCMS)])
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0,
                     MulCoCMS, INS, [0], [0], Met=1)[0]

        def fit_func(x, p):
            return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, [], [],
                         Met=1, Norm=Norm)

        INS_shift = 0
    elif VVV == 3:
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0,
                     MulCo, INS, [0], [0])[0]

        def fit_func(x, p):
            return m5.TI(x, p, model, JN, pool, x0, MulCo, INS, [], [], Norm=Norm)

        # The multi-line instrumental function displaces the apparent line
        # positions by the intensity-weighted sum of the squared line shifts;
        # the folded velocity axis is corrected by this at the very end.
        INS_shift = 0
        for i in range(0, int(len(INS) / 3)):
            INS_shift += INS[i * 3 + 1] * INS[i * 3 + 2] ** 2
    else:
        raise ValueError(f"unknown experimental method VVV={VVV!r} (expected 1 or 3)")

    # --------------------------------------------------------------------- #
    # Drive-curve models (channel -> velocity) and the final-fit model.     #
    #                                                                       #
    # sin_cal / lin_cal map channel numbers to velocities for a sinusoidal  #
    # resp. triangular drive. *_fin2 are the models of the FINAL global     #
    # fit: raw counts as a function of channel = transmission integral on   #
    # the drive curve + per-half-sweep parabolic distortion.                #
    #                                                                       #
    # WARNING (deliberate, load-bearing side effects):                      #
    #   * ``pCAL2 = pCAL`` is an ALIAS -- the *_fin2 models write the       #
    #     fitted intensities/texture into the shared ``pCAL`` array.        #
    #   * They also clamp entries of the optimiser's own vector ``p`` in    #
    #     place (texture reset for tiny velocity ranges, baseline-2 clamp), #
    #     which steers minimi_hi away from unphysical regions.              #
    # --------------------------------------------------------------------- #

    def sin_cal(x, p):
        """Velocity at channel x for a sinusoidal drive.

        p = [amplitude (mm/s), phase (rad), source shift (mm/s)]. The
        len/pi*sin(pi/len) factor averages the sine over one channel width.
        """
        return (p[0] / np.pi * n_ch * np.sin(np.pi / n_ch)
                * np.cos(p[1] + np.pi / n_ch * (2 * x + 1)) + p[2])

    def lin_cal(x, p):
        """Velocity at channel x for a triangular drive: two mirrored linear
        ramps. p = [v at channel 0, v at the last channel, v step/channel].
        """
        return np.where(x <= int(n_ch / 2),
                        p[0] - p[2] * x,
                        p[1] - p[2] * (n_ch - 1 - x))

    def sin_cal_fin2(x, p):
        """Final-fit model for the sinusoidal drive (see ps1 layout in the
        module docstring for the meaning of p)."""
        pCAL2 = pCAL                                   # ALIAS: writes into pCAL
        pCAL2[nbp] = p[9]                              # main sextet intensity
        if VVV == 3:
            # second Sextet (impurity) intensity: baseline(8) + first
            # Sextet(14) -> the second Sextet block starts at nbp+14.
            pCAL2[nbp + 14] = p[11]
        Hx = (p[0] / np.pi * n_ch * np.sin(np.pi / n_ch)
              * np.cos(p[1] + np.pi / n_ch * (2 * x + 1)) + p[2])
        if min(Hx) > -2.95 and max(Hx) < 2.95:
            p[10] = 0.0    # tiny velocity range: texture undefined -> isotropic
        pCAL2[nbp + 8] = p[10]                         # texture order parameter A
        if VVV == 1:
            pCAL2[0] = p[11]                           # baseline of half-sweep 1
            if p[12] > p[11] * 10:
                p[12] = p[11] * 10                     # keep baseline 2 sane
            pCAL2[4] = p[12]                           # baseline of half-sweep 2
        half = int(len(x) / 2)
        # Velocity of the same channel in the OTHER half-sweep: sign(|Hx-Hx1|)
        # masks the parabola of half 1 to half 2 and vice versa.
        Hx1 = np.concatenate((Hx[:half], Hx[:half]), axis=0)
        Hx2 = np.concatenate((Hx[half:], Hx[half:]), axis=0)
        return (p[3] * fit_func(Hx, pCAL2)
                + (p[5] * (p[4] - channels) ** 2 + p[6]) * np.sign(abs(Hx - Hx2))
                + (p[8] * (p[7] - channels) ** 2) * np.sign(abs(Hx - Hx1)))

    def lin_cal_fin2(x, p):
        """Final-fit model for the triangular drive (CMS only)."""
        pCAL2 = pCAL                                   # ALIAS: writes into pCAL
        pCAL2[nbp] = p[9]                              # main sextet intensity
        first = x <= int(n_ch / 2)
        Hx = np.where(first, p[0] - p[2] * x, p[1] - p[2] * (n_ch - 1 - x))
        # Indicators of the two half-sweeps (x is always the sorted channel
        # axis here, so elementwise selection == the original concatenation).
        Hx1 = np.where(first, 1.0, 0.0)
        Hx2 = 1.0 - Hx1
        if min(Hx) > -2.95 and max(Hx) < 2.95:
            p[10] = 0.0    # tiny velocity range: texture undefined -> isotropic
        pCAL2[nbp + 8] = p[10]                         # texture order parameter A
        pCAL2[0] = p[11]                               # baseline of half-sweep 1
        if p[12] > p[11] * 10:
            p[12] = p[11] * 10                         # keep baseline 2 sane
        pCAL2[4] = p[12]                               # baseline of half-sweep 2
        return (fit_func(Hx, pCAL2)
                + (p[5] * (p[4] - channels) ** 2 + p[6]) * Hx1
                + (p[8] * (p[7] - channels) ** 2) * Hx2)

    # --------------------------------------------------------------------- #
    # STEP 3: phase between the drive and the recording.                    #
    # --------------------------------------------------------------------- #
    phase0 = _detect_phase_offset(half1, half2)
    print('PHASE  ', phase0)

    # --------------------------------------------------------------------- #
    # STEP 4: alpha-Fe reference model, start values and bounds.            #
    # --------------------------------------------------------------------- #
    # Baseline guess: max counts minus 2 sigma; CMS splits it 60/40 between
    # the two half-sweep count rates (indices 0 and 4).
    baseline_guess = max(counts) - 2 * np.sqrt(max(counts))
    p00 = np.array([baseline_guess * (1 - 0.4 * (VVV == 1)), 0, 0, 0,
                    baseline_guess * (0.4 * (VVV == 1)), 0, 0, 0,
                    # Sextet: T   d  e  H(T)             L      G  th ph A ...
                    8, 0.0, 0, ALPHA_FE_FIELD, 0.098, 0.0, 90, 0, 0, 0, 0, 0, 0, 3])
    bounds = np.array([[-np.inf] * len(p00), [np.inf] * len(p00)], dtype=float)
    bounds[0][nbp + 1] = -0.05                         # isomer shift d
    bounds[1][nbp + 1] = 0.05
    bounds[0][nbp + 3] = 32.54                         # hyperfine field H (33.04 +/- 0.5 T)
    bounds[1][nbp + 3] = 33.54

    # The six reference line velocities, oriented by the sweep direction.
    # This FIRST orientation is the only place Vel_start acts: it propagates
    # into the coarse-search axis and from there into everything else.
    line_v = np.copy(ALPHA_FE_LINE_VELOCITIES)
    if Vel_start == 1:
        line_v = line_v[::-1]

    # Parameters held fixed during the coarse search / refinement fits
    # (minimi_hi convention). Coarse: H free within bounds, linewidth L fixed.
    # Refinement: H fixed at the reference, L free.
    fix_coarse = np.array([1, 2, 3, 1 + int(VVV), 5, 6, 7,
                           nbp + 2, nbp + 4, nbp + 5, nbp + 6, nbp + 7,
                           nbp + 8, nbp + 9, nbp + 10, nbp + 11, nbp + 12, nbp + 13])
    fix_refine = np.array([1, 2, 3, 1 + int(VVV), 5, 6, 7,
                           nbp + 2, nbp + 3, nbp + 4, nbp + 6, nbp + 7,
                           nbp + 8, nbp + 9, nbp + 10, nbp + 11, nbp + 12, nbp + 13])

    # --------------------------------------------------------------------- #
    # STEP 5a: coarse search, sinusoidal drive (both SMS and CMS).          #
    # Hypothesis: sextet lines (i, j) sit in the two deepest minima; that   #
    # fixes amplitude+shift analytically, a short fit scores the hypothesis.#
    # --------------------------------------------------------------------- #
    ch_min1, ch_min2 = _deepest_line_channels(counts)
    print('channels of minimum ', ch_min1, ch_min2)
    start_time = time.time()
    chi2_sin = 10000.0
    for i in range(0, 6):
        for j in range(i + 1, 6):
            Vel_max_m = ((line_v[i] - line_v[j]) * np.pi / n_ch / np.sin(np.pi / n_ch)
                         / (np.cos(np.pi / n_ch * (2 * ch_min1 + 1))
                            - np.cos(np.pi / n_ch * (2 * ch_min2 + 1))))
            shift_m = line_v[i] - (Vel_max_m / np.pi * n_ch * np.sin(np.pi / n_ch)
                                   * np.cos(np.pi / n_ch * (2 * ch_min1 + 1)
                                            + (np.pi / n_ch * phase0)))
            x_try = sin_cal(channels, [Vel_max_m, (np.pi / n_ch * phase0), shift_m])
            res = mi.minimi_hi(fit_func, x_try, counts, p00, fix=fix_coarse,
                               bounds=bounds, MI=3, MI2=5)
            if res[2] <= chi2_sin:
                chi2_sin = res[2]
                Vel_max_sin = Vel_max_m
                shift = shift_m
                p_sin = res[0]
    print('best sinusoidal hypothesis:', Vel_max_sin, shift, chi2_sin)
    x_sin = sin_cal(channels, [Vel_max_sin, (np.pi / n_ch * phase0), shift])

    method = 0                       # 0 = sinusoidal, 1 = triangular (linear)
    x = x_sin
    p0 = p_sin
    Vel_max = Vel_max_sin
    cal, cal_fin2 = sin_cal, sin_cal_fin2
    print('coarse sinusoidal search took', time.time() - start_time, 'seconds')

    # --------------------------------------------------------------------- #
    # STEP 5b (CMS only): coarse search with a triangular drive, then pick  #
    # the waveform by chi-square. Conventional drives can run either        #
    # waveform and the user does not tell us which. The phase acts per half #
    # sweep here, hence phase0/2.                                           #
    # --------------------------------------------------------------------- #
    if VVV == 1:
        lin_phase = phase0 / 2
        chi2_lin = 10000.0
        for i in range(0, 6):
            for j in range(i + 1, 6):
                velocity_step_m = -(line_v[j] - line_v[i]) / (ch_min2 - ch_min1)
                Vel_max_m = line_v[i] + velocity_step_m * ch_min1
                x_try = lin_cal(channels, [Vel_max_m,
                                           Vel_max_m - velocity_step_m * lin_phase,
                                           velocity_step_m])
                res = mi.minimi_hi(fit_func, x_try, counts, p00, fix=fix_coarse,
                                   bounds=bounds, MI=3, MI2=5)
                if res[2] <= chi2_lin:
                    chi2_lin = res[2]
                    Vel_max_lin = Vel_max_m
                    velocity_step = velocity_step_m
                    p_lin = res[0]
        print('best triangular hypothesis:', Vel_max_lin, velocity_step, chi2_lin)
        x_lin = lin_cal(channels, [Vel_max_lin,
                                   Vel_max_lin - velocity_step * lin_phase,
                                   velocity_step])

        if chi2_sin <= chi2_lin:
            print('sinus mode')
        else:
            method = 1
            x = x_lin
            p0 = p_lin
            Vel_max = Vel_max_lin
            cal, cal_fin2 = lin_cal, lin_cal_fin2
            print('triangular mode')

    # --------------------------------------------------------------------- #
    # STEP 6: iterative refinement (4 rounds). Fit the spectrum on the      #
    # current axis; from the fitted sextet predict WHERE (which channel)    #
    # each of the 12 line occurrences must sit; refit the drive-curve       #
    # parameters to those (channel, velocity) pairs; rebuild the axis.      #
    # This bootstraps the axis far more robustly than fitting everything    #
    # at once, because each round only trusts the line POSITIONS.           #
    # --------------------------------------------------------------------- #
    p0[nbp + 1] = 0                                    # reset isomer shift
    p0[nbp + 3] = ALPHA_FE_FIELD                       # reset hyperfine field
    print('method ', method)
    print(p0)
    start_time = time.time()
    for it in range(0, 4):
        p = mi.minimi_hi(fit_func, x, counts, p0, fix=fix_refine, MI=3, MI2=5)[0]
        # 12 reference velocities: the 6 lines, seen once per half-sweep.
        # No reversal is needed here even for Vel_start == 1: the velocity
        # direction is already encoded in the calibrated axis `x` (built from
        # the oriented `line_v`). Each reference velocity ref_v[k] is paired
        # with the channel predicted_ch[k] located by argmin on that oriented
        # axis, so the pairing is correct in either direction. (A vestigial
        # `if Vel_start == 1: sex0[::-1]` no-op lived here historically;
        # actually reversing would break the pairing.)
        ref_v = np.tile(ALPHA_FE_LINE_VELOCITIES, 2)
        predicted_ch = np.array([float(0)] * 12)
        V = nbp
        x1 = x[:int(len(x) / 2)]
        x2 = x[int(len(x) / 2):]
        # H in mm/s of outer-line half-splitting; a+ (V+10) and a- (V+11) are
        # the outer/inner line-shift corrections, alternating sign per line.
        p[V + 3] = p[V + 3] / TESLA_PER_MMS
        predicted_ch[0] = (np.abs(x1 - (p[V + 1] - p[V + 3] / 2 + p[V + 2]) - p[V + 10])).argmin()
        predicted_ch[1] = (np.abs(x1 - (p[V + 1] - 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 11])).argmin()
        predicted_ch[2] = (np.abs(x1 - (p[V + 1] - 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 11])).argmin()
        predicted_ch[3] = (np.abs(x1 - (p[V + 1] + 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 11])).argmin()
        predicted_ch[4] = (np.abs(x1 - (p[V + 1] + 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 11])).argmin()
        predicted_ch[5] = (np.abs(x1 - (p[V + 1] + p[V + 3] / 2 + p[V + 2]) + p[V + 10])).argmin()
        predicted_ch[6] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] - p[V + 3] / 2 + p[V + 2]) - p[V + 10])).argmin()
        predicted_ch[7] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] - 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 11])).argmin()
        predicted_ch[8] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] - 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 11])).argmin()
        predicted_ch[9] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] + 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 11])).argmin()
        predicted_ch[10] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] + 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 11])).argmin()
        predicted_ch[11] = int(len(x) / 2) + (np.abs(x2 - (p[V + 1] + p[V + 3] / 2 + p[V + 2]) + p[V + 10])).argmin()
        p[V + 3] = p[V + 3] * TESLA_PER_MMS

        # Discard lines that fell within 5 channels of the spectrum edges or
        # of the fold point -- their argmin is saturated, not a real position.
        n_half = len(x1)
        keep = ~((predicted_ch < 5)
                 | ((predicted_ch > n_half - 1 - 5) & (predicted_ch < n_half + 5))
                 | (predicted_ch > len(x) - 1 - 5))
        predicted_ch = predicted_ch[keep]
        ref_v = ref_v[keep]

        if it == 0:
            if method == 0:
                ps = mi.minimi_hi(sin_cal, predicted_ch, ref_v,
                                  p0=np.array([Vel_max, (np.pi / n_ch * phase0), shift]),
                                  tau0=1)[0]
            if method == 1:
                ps = mi.minimi_hi(lin_cal, predicted_ch, ref_v,
                                  p0=np.array([Vel_max, Vel_max - velocity_step * lin_phase,
                                               velocity_step]),
                                  tau0=1)[0]
        else:
            ps = mi.minimi_hi(cal, predicted_ch, ref_v, p0=ps, eps=10 ** -40)[0]
        x = cal(channels, ps)
        p0 = np.copy(p)
        p0[7] = 0                                      # reset last baseline parameter
    print('refinement loop took', time.time() - start_time, 'seconds')

    # --------------------------------------------------------------------- #
    # STEP 7 (SMS): redo the normalisation on a finer integration grid for  #
    # the final fit (fit_func late-binds JN and Norm).                      #
    # --------------------------------------------------------------------- #
    if VVV == 3:
        JN = 64
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0,
                     MulCo, INS, [0], [0])[0]

    ps1 = np.append(ps, 1)                             # ps1[3]: global intensity scale

    # --------------------------------------------------------------------- #
    # STEP 8: parabolic count distortion. The solid angle seen by the       #
    # detector varies with drive position, modulating the count rate as a   #
    # parabola with opposite sign in the two half-sweeps; fitting           #
    # half1 - reversed(half2) isolates exactly that asymmetry.              #
    # --------------------------------------------------------------------- #
    start_time = time.time()
    half2_rev = half2[::-1]
    parab = mi.minimi_hi(_parabola, channels[:int(n_ch / 2)],
                         half1 - half2_rev + half1[0],
                         p0=[int(n_ch / 4), 5,
                             half1[int(len(half1) / 2)] - half2_rev[int(len(half2_rev) / 2)] + half1[0]])[0]
    print('parabola fit took', time.time() - start_time, 'seconds')
    parab[2] = parab[2] - half1[0]

    # Split the fitted asymmetry into the two per-half parabolas of ps1.
    ps1 = np.append(ps1, parab[0])                     # ps1[4]: centre, half 1
    ps1 = np.append(ps1, parab[1] / 2)                 # ps1[5]: curvature, half 1
    ps1 = np.append(ps1, parab[2])                     # ps1[6]: offset, half 1
    ps1 = np.append(ps1, parab[0] + int(n_ch / 2))     # ps1[7]: centre, half 2
    ps1 = np.append(ps1, (-1) * parab[1] / 2)          # ps1[8]: curvature, half 2

    # --------------------------------------------------------------------- #
    # STEP 9: final model. SMS reference foils show a second (impurity)     #
    # sextet and the Be-window doublet (parameters from Be.txt); CMS keeps  #
    # the single sextet but frees both half-sweep count rates.              #
    # --------------------------------------------------------------------- #
    if method == 0:
        if VVV == 3:
            model = ['Sextet', 'Sextet', 'Doublet']
            try:
                Be_param = np.genfromtxt(os.path.join(dir_path, 'Be.txt'),
                                         delimiter='\t', skip_footer=0)
                print('Be.txt was read')
            except Exception:
                Be_param = np.array([0.057, 0.066, -0.261, 0.098, 0.375, 90, 0,
                                     0.427037824, 1])
                print('COULD NOT READ Be.txt')
            # baseline(8) + two polarized Sextets(14 each) + Be Doublet(9).
            pCAL = np.array([p[0], 0, 0, 0, 0, 0, 0, 0,
                             8.08, 0, 0, 33.04, 0.098, 0, 90, 0, 0, 0, 0, 0, 0, 3,
                             0.451, -0.041, 0.003, 30.88, 0.098, 0.1, 90, 0, 0, 0, 0, 0, 0, 3])
            pCAL = np.concatenate((pCAL, Be_param))
        if VVV == 1:
            model = ['Sextet']
            pCAL = np.array([p[0], 0, 0, 0, p[3], 0, 0, 0,
                             8.08, 0, 0, 33.04, 0.098, 0, 90, 0, 0, 0, 0, 0, 0, 3])
            print('background ', pCAL[0], pCAL[4], ps1[3])
    if method == 1:
        model = ['Sextet']
        pCAL = p0
        pCAL[nbp + 13] = 3                             # I1/I3 line-intensity ratio
        pCAL[nbp + 5] = 0                              # Gaussian broadening G
        print('background ', pCAL[0], pCAL[4], ps1[3])

    ps1 = np.append(ps1, 8)                            # ps1[9]:  main sextet intensity
    ps1 = np.append(ps1, 0)                            # ps1[10]: texture parameter A
    if VVV == 3:
        ps1 = np.append(ps1, 0.5)                      # ps1[11]: impurity intensity
    if VVV == 1:
        # Free both half-sweep count rates, starting from a 60/40 split of the
        # fitted total; rescale the intensity guess accordingly.
        tot = (pCAL[0] + pCAL[4])
        ps1[9] = pCAL[nbp] * pCAL[0] / tot / 3 * 5
        pCAL[0] = tot * 0.6
        pCAL[4] = tot * 0.4
        ps1 = np.append(ps1, pCAL[0])                  # ps1[11]: count rate, half 1
        ps1 = np.append(ps1, pCAL[4])                  # ps1[12]: count rate, half 2
        print('baseline ', ps1[11], ps1[12])

    print('parameter set after preliminary fit ', ps1)
    print('model parameters after preliminary fit ', pCAL)

    # --------------------------------------------------------------------- #
    # STEP 10: final global fit -- drive curve + distortion + intensities.  #
    # --------------------------------------------------------------------- #
    start_time = time.time()
    res = mi.minimi_hi(cal_fin2, channels, counts, p0=ps1, MI=20, MI2=20, eps=10 ** -6)
    if abs(res[0][0]) < 2.95:
        print('very small velocity range - texture could not be defined')
    print('model parameters ', pCAL)
    print('variable parameters ', res[0])
    print('hi2 ', res[2])
    print('main minimization took', time.time() - start_time, 'seconds')
    pS = res[0]
    v_axis = cal(channels, pS)                         # final channel -> velocity map

    # Fold the fitted global scale into the count rates so pS[3] == 1 from
    # here on (the folded outputs must be in true count units).
    pCAL[0] = pCAL[0] * pS[3]
    if VVV == 1:
        pCAL[4] = pCAL[4] * pS[3]
    pS[3] = 1

    flat_counts = _subtract_distortion(counts, channels, pS)
    final_curve = cal_fin2(channels, pS)

    # --------------------------------------------------------------------- #
    # STEP 11: fold the two half-sweeps onto one velocity axis and apply    #
    # the instrumental line-shift correction.                               #
    # --------------------------------------------------------------------- #
    if method == 0:
        v_folded, data_folded, fit_folded, fold_residual, n1, n2 = \
            _fold_sinusoidal(v_axis, counts, final_curve, pS[2])
        print(n1, n2, len(v_folded))
    if method == 1:
        v_folded, data_folded, fit_folded, fold_residual, n1, n2 = \
            _fold_triangular(v_axis, counts, final_curve, pS)
    v_folded = v_folded + INS_shift

    # --------------------------------------------------------------------- #
    # STEP 12: diagnostics and output files.                                #
    # --------------------------------------------------------------------- #
    if VVV == 3:
        _save_sms_diagnostic_plot(dir_path, channels, counts, final_curve,
                                  flat_counts, v_axis, v_folded, data_folded,
                                  fold_residual, pS, pCAL[0], method, n1, n2)
        print('Shift due to instrumental function ', INS_shift)
        print('Velocity range: ', min(v_folded), ' to ', max(v_folded),
              'mm/s, absolute shift ', (max(v_folded) + min(v_folded)) / 2)
        print('Velocity amplitude is', pS[0], 'mm/s')
        print('Source shift is', pS[2], 'mm/s')
        print('difference of N0:',
              abs(pS[6]) / (pCAL[0] + pS[6] * (1 + np.sign(pS[6])) / 2) * 100, '%')
        print('impurity is', pS[11] / (pS[9] + pS[11]) * 100, '%')
        print('maximum deviation is ', max(np.abs(data_folded - fit_folded)),
              'or ', max(np.abs(data_folded - fit_folded)) / pCAL[0], '%')
        print('Texture parameter is', pS[10], 'should be 0 for isotropic')

    _write_calibration_dat(dir_path, method, n1, n2, v_folded, data_folded)

    return (v_folded, data_folded, fit_folded)
