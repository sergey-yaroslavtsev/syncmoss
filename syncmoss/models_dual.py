"""models.TI for parameters that may be Duals (dual_numbers): the spectrum
together with its exact derivatives in the fit parameters.

    prepared = prepare(x, model, JN, x0, MulCo, INS, Met, Norm, pol, lengths, smearing)   # once per fit
    q = fit_parameters(p, free, confu, Expr, NExpr)       # p with its free parameters as Duals
    y = TI(q, prepared, pool, Distri, Cor, Recon)         # a Dual: the spectrum and dy/dp
    J = dual_numbers.jacobian(y, free)

How it differs from models.TImod, which it must equal to rounding:

  * a component is a LIST OF LINES -- each a shape (Voigt, Blume's two
    states, a group of the many-state relaxation) with its own parameters,
    an amplitude and, for a thick component, a 2x2 matrix -- made from the
    parameters ONCE per spectrum, not at every node. This is arithmetic on
    numbers and small arrays, where a Dual is cheap;
  * a distribution (Distr, Recon, their Corr rows) is MORE LINES: the copies'
    parameters are arrays over the distribution's axis, and the component's
    line function runs on them. The texts go through expression_eval, which
    takes Duals, so a PDF's or a Corr's own parameters are differentiated too;
  * all the work on the energy axis -- the shapes, the sum of the lines, the
    layers' matrix exponentials, the readout, the patches of a thickness
    distribution -- is ONE compiled kernel per node
    (dual_kernels._transmission), with the derivatives.

With plain parameters it returns the plain spectrum, by the same kernel.

The derivatives are exact (the Hamiltonian's levels that coincide included:
hamiltonian_dual).

Every row of models.TImod is covered; uncovered() names what is not (a new
row), and TI raises NotCovered for what only the values tell: a distribution
of a slot that is not its component's, a Layer with a Distr_T.


ADDING A MODEL
==============
A row added to models.TImod is NOT covered here until it has its line
function: uncovered() names it, prepare() raises NotCovered, and the fit
keeps models.TI with finite differences -- nothing breaks, it is only slower.
To cover it:

1. Write its line function, `_name(p, MulCo, text, cms)` (see _sextet, the
   plainest; _asm for one with a loop): p is the row's slots, numbers or
   Duals; it returns the row's lines, [(shape, parameters, amplitude, matrix)]:
     * shape -- VOIGT (pos, gL, gG), BLUME (alf, s, L, W, R) or RELAX + g
       (Sig, eps, Hv, W, R, alfa, S) of dual_kernels, the parameters in
       TImod's energy units (MulCo times mm/s);
     * amplitude -- real, TImod's factor of the line WITHOUT its matrix
       (_kpref(MulCo) times the thickness times the line's share);
     * matrix -- None for a row TImod adds to the thin exponent; for a row it
       adds to the cross-section matrix, the line's 2x2 matrix. Lines given
       the SAME matrix object are summed before it multiplies.
   Repeat TImod's formulas for the row in TImod's order, so that the value
   agrees to rounding. Write them with numpy functions (np.sin, np.sqrt,
   np.hypot, abs, np.where, np.maximum): they take Duals, math.* and float()
   do not. Counts (a number of points, a spin) are read with int(dn.value(x)).
2. Register it in COMPONENTS: name: (slots in p, the function, takes_arrays).
   takes_arrays = True if every formula broadcasts over arrays of parameters
   (a distribution of it is then one call); False if it has Python branches
   or loops on values -- it is then called once per copy of a distribution.
3. Only if no shape fits: a new shape is a compiled function of
   dual_kernels (F[0] the shape, F[1 + i] its derivative in parameter i,
   written by hand -- see _shape_blume), a code next to VOIGT, its count in
   _COUNT and its branch in _transmission. A helper kernel that numba
   compiles (as Ham_mosaic) gets a twin on Duals in a module of its own
   (hamiltonian_dual): the same formulas with numpy; what has no derivative
   as written (roots that meet) is differentiated implicitly there.
4. Where a formula has a kink or a 0/0 at a value a fit can START from (a
   width 0, a field 0), the Dual's convention decides the column -- |x| at 0:
   |dx|; sqrt at 0: 0 -- and it may not be the one-sided slope the fit needs:
   see _root_at_zero (MDGD) and hamiltonian_dual's first moment. A switch
   between two formulas at a threshold must join them, or the derivative is
   right on each side and the fit still sees a step (ASM's SN at m = 0.981).
5. Tests: add the row to CASES of tests/test_models_dual.py (and to it under
   a Distr if it can be distributed): the value against models.TI to
   rounding, every column against central differences of models.TI, both
   sources. Add the start-at-zero cases the model has.
6. Nothing to do in minimi_lib or fitting_io: the fit asks uncovered().
"""
import numpy as np

import syncmoss.dual_numbers as dn
import syncmoss.minimi_lib as mi
import syncmoss.models as m5
from syncmoss.constants import (NAT_WIDTH, TESLA_PER_MMS, MMS_PER_T_PER_G, LINE_RATIO_25, LINE_RATIO_34,
                                LINE_SHIFT_16, LINE_SHIFT_25, LINE_SHIFT_34, E0_J, c, ggr, gex, mun,
                                number_of_baseline_parameters, numco)
from syncmoss.dual_kernels import BLUME, PARAMETERS, RELAX, SLOPE, VOIGT, _transmission
from syncmoss.expression_eval import eval_expr
from syncmoss.hamiltonian_dual import Ham_mosaic
from syncmoss.spectrum_model import parse_spectrum_lines


class NotCovered(ValueError):
    """The model has something TI of this module does not compute: the caller
    keeps models.TI and its finite differences."""


# =============================================================================
# The components as lists of lines
# =============================================================================
# A line is (shape, parameters, amplitude, matrix): a shape of dual_kernels
# with its own parameters (in TImod's energy units, MulCo times mm/s), a real
# amplitude, and the 2x2 matrix of a thick component's line -- None for a thin
# one, exp(-amplitude shape). Each value may be a number, a Dual, or an array
# of them over the copies of a distribution (a matrix: (..., 2, 2); the shape:
# a code, or an array of codes). Lines that hold the SAME matrix object are
# summed before the matrix multiplies. How to add a model: the module's text.
_I2 = np.eye(2, dtype=complex)
_J2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
_E11 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
_E22 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
_E12_21 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _kpref(MulCo):
    return np.pi * (NAT_WIDTH / 2 * MulCo)


def _m(a):
    """A number, or an array over the copies, as a factor of 2x2 matrices."""
    return a[..., None, None] if getattr(a, 'ndim', 0) else a


# models' _mm_perp, _texture_blend and _texture_s1, for Duals and for arrays
# over the copies (models' own build a 2x2 array of numbers).
def _mm_perp(mx, my):
    return _m(mx * mx) * _E11 + _m(mx * my) * _E12_21 + _m(my * my) * _E22


def _blend(M, A):
    return _m(1.0 - A) * _I2 + _m(A) * M


def _s1(A, Am):
    return Am * np.sqrt(np.maximum(0.0, (1.0 + 2.0 * A) / 3.0))


def _sextet_matrices(theta, phi, A, Am):
    """sigma-, pi and sigma+ of the axis (theta, phi), the texture A and the
    polar order Am: TImod's Msm, Mpi, Msp."""
    mx, my, mz = m5._axis_xyz(theta, phi)
    mm = _mm_perp(mx, my)
    Msig = _blend(1.5 * (_I2 - mm), A)
    Mfar = _m(1.5 * _s1(A, Am) * mz) * (1j * _J2)
    return Msig - Mfar, _blend(3.0 * mm, A), Msig + Mfar


def _sextet_intensities(I, I13, MulCo):
    K = _kpref(MulCo)
    Aeff = 0.5
    return (K * I * (4 * I13 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff), K * I * 2 * Aeff / (8 - 4 * Aeff),
            K * I * (4 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff))


def _singlet(p, MulCo, text, cms):
    return [(VOIGT, (p[1] * MulCo, abs(p[2]) * MulCo, abs(p[3]) * MulCo), _kpref(MulCo) * abs(p[0]), None)]


def _spectrum(p, MulCo, text, cms):
    lines = parse_spectrum_lines(text)
    if not len(lines):
        return []
    Tq = np.abs(lines[:, 0])
    return [(VOIGT, ((float(dq) + p[1]) * MulCo, NAT_WIDTH * MulCo, np.hypot(float(Gq), p[2]) * MulCo),
             _kpref(MulCo) * abs(p[0]) * float(Iq), None)
            for Iq, dq, Gq in zip(Tq / np.sum(Tq), lines[:, 1], lines[:, 2])]


def _doublet(p, MulCo, text, cms):
    half = _kpref(MulCo) * 0.5 * abs(p[0])
    WL = abs(p[3]) * MulCo
    WG1 = abs(p[4]) * MulCo
    mx, my, mz = m5._axis_xyz(p[5], p[6])
    mm = _mm_perp(mx, my)
    return [(VOIGT, ((p[1] + p[2]) * MulCo, WL, WG1 * p[8]), half, _blend(1.5 * (_I2 - mm), p[7])),       # line A
            (VOIGT, ((p[1] - p[2]) * MulCo, WL, WG1), half, _blend(0.5 * _I2 + 1.5 * mm, p[7]))]         # line B


def _six_lines(I, I13, delta, eps, H, a16, a25, WL, G, theta, phi, A, Am, MulCo):
    """The six lines of a sextet with the Gaussian widths G (Sextet, MDGD)."""
    I1, I2, I3 = _sextet_intensities(I, I13, MulCo)
    HH = H / TESLA_PER_MMS
    pos = [(delta - HH / 2 + eps) * MulCo + a16 * MulCo,
           (delta - LINE_RATIO_25 * HH / 2 - eps) * MulCo - a25 * MulCo,
           (delta - LINE_RATIO_34 * HH / 2 - eps) * MulCo + a25 * MulCo,
           (delta + LINE_RATIO_34 * HH / 2 - eps) * MulCo - a25 * MulCo,
           (delta + LINE_RATIO_25 * HH / 2 - eps) * MulCo + a25 * MulCo,
           (delta + HH / 2 + eps) * MulCo - a16 * MulCo]
    Msm, Mpi, Msp = _sextet_matrices(theta, phi, A, Am)
    return [(VOIGT, (pos[q], WL, G[q]), amplitude, M)
            for q, (amplitude, M) in enumerate(((I1, Msm), (I2, Mpi), (I3, Msp), (I3, Msm), (I2, Mpi), (I1, Msp)))]


def _sextet(p, MulCo, text, cms):
    WG = abs(p[5]) * MulCo
    GaH = abs(p[12]) / 2 / TESLA_PER_MMS * MulCo
    # hypot: TImod's (WG**2 + GaH**2)**(1/2), with a derivative where both are 0
    Ga16, Ga25, Ga34 = np.hypot(WG, GaH), np.hypot(WG, LINE_RATIO_25 * GaH), np.hypot(WG, LINE_RATIO_34 * GaH)
    return _six_lines(abs(p[0]), p[13], p[1], p[2], p[3], p[10], p[11], abs(p[4]) * MulCo,
                      [Ga16, Ga25, Ga34, Ga34, Ga25, Ga16], p[6], p[7], p[8], p[9], MulCo)


def _root_at_zero(G, a, b, c2, u, v):
    """G = sqrt(|a u^2 + 2 b u v + c2 v^2|), a width made of the two widths u
    and v, where both are 0: there its derivative along a parameter is
    sqrt(|a du^2 + 2 b du dv + c2 dv^2|), one-sided, as abs and hypot have it
    at 0 -- so a width that starts at 0 has its column and can leave 0."""
    zero = (np.asarray(dn.value(u)) == 0) & (np.asarray(dn.value(v)) == 0)
    if not isinstance(G, dn.Dual) or not np.any(zero):
        return G
    du, dv = dn.columns(u, G.idx), dn.columns(v, G.idx)
    a, b, c2 = (np.asarray(dn.value(x))[..., None] for x in (a, b, c2))
    slope = np.sqrt(np.abs(a * du ** 2 + 2 * b * du * dv + c2 * dv ** 2))
    return dn.Dual(G.v, np.where(zero[..., None], slope, G.d), G.idx)


def _mdgd(p, MulCo, text, cms):
    Guni = abs(p[5]) * MulCo
    Gh = abs(p[6]) * MulCo
    Gde, Gdh, Geh = p[7], p[8], p[9]
    Ce = [1, -1, -1, -1, -1, 1]
    Ch = [-LINE_SHIFT_16, -LINE_SHIFT_25, -LINE_SHIFT_34, LINE_SHIFT_34, LINE_SHIFT_25, LINE_SHIFT_16]
    independent = np.maximum(0, (1 - (abs(Gdh) + abs(Geh)) ** 2))
    G = []
    for j in range(6):
        width = np.sqrt(abs(Guni ** 2 + Ch[j] ** 2 * Gh ** 2 + Ce[j] * Gde * Guni ** 2 * independent
                            + Ch[j] * 2 * Gdh * Guni * Gh + Ce[j] * Ch[j] * 2 * Geh * Guni * Gh))
        G.append(_root_at_zero(width, 1 + Ce[j] * Gde * independent, Ch[j] * (Gdh + Ce[j] * Geh), Ch[j] ** 2,
                               Guni, Gh))
    return _six_lines(abs(p[0]), p[16], p[1], p[2], p[3], p[14], p[15], abs(p[4]) * MulCo, G,
                      p[10], p[11], p[12], p[13], MulCo)


def _relax_ms(p, MulCo, text, cms):
    I = _kpref(MulCo) * abs(p[0] * 2) * 2          # TImod's I, and the 2 of relax_MS_thick
    Ah = 0.5
    parameters = (p[1] * MulCo, p[2] * MulCo, p[3] * MulCo / 2 / TESLA_PER_MMS, abs(p[4]) * MulCo / 2,
                  p[8], p[9], p[10])               # Sig, eps, Hv, W, R, alfa, S
    mx, my, mz = m5._axis_xyz(p[5], p[6])
    mm = _mm_perp(mx, my)
    Md1, Mpi = _blend(1.5 * (_I2 - mm), p[7]), _blend(3.0 * mm, p[7])
    return [(RELAX + 0, parameters, I * 3 * (1 - Ah) / (8 - 4 * Ah), Md1),
            (RELAX + 1, parameters, I * 2 * Ah / (8 - 4 * Ah), Mpi),
            (RELAX + 2, parameters, I * 1 * (1 - Ah) / (8 - 4 * Ah), Md1)]


def _relax_2s(p, MulCo, text, cms):
    K = _kpref(MulCo) * abs(p[0])
    Aeff = 0.5
    I1, I2, I3 = K * 3 * (1 - Aeff) / (8 - 4 * Aeff), K * 2 * Aeff / (8 - 4 * Aeff), K * 1 * (1 - Aeff) / (8 - 4 * Aeff)
    Sig1, Q1, H1 = p[1] * MulCo, p[2] / 3 * MulCo, p[3] * MMS_PER_T_PER_G * MulCo
    Sig2, Q2, H2 = p[4] * MulCo, p[5] / 3 * MulCo, p[6] * MMS_PER_T_PER_G * MulCo
    WL, We, R = abs(p[7]) * MulCo, p[12] * MulCo, p[13]
    Msm, Mpi, Msp = _sextet_matrices(p[8], p[9], p[10], p[11])

    def line(m0, m1, amplitude, M):
        # Blume's alf and what it subtracts from the energy, of the transition m0 -> m1
        field, quad = ggr * m0 - gex * m1, 3 * m1 ** 2 - 15 / 4
        alf = field * (H2 - H1) / 2 + (Q2 - Q1) / 2 * quad + (Sig2 - Sig1) / 2
        mean = (Sig1 + Sig2) / 2 + (Q2 + Q1) / 2 * quad + field * (H2 + H1) / 2
        return (BLUME, (alf, mean, WL, We, R), amplitude, M)
    return [line(-1 / 2, -3 / 2, I1, Msm), line(1 / 2, 3 / 2, I1, Msp), line(-1 / 2, -1 / 2, I2, Mpi),
            line(1 / 2, 1 / 2, I2, Mpi), line(-1 / 2, 1 / 2, I3, Msp), line(1 / 2, -1 / 2, I3, Msm)]


def _hamiltonian_lines(I, delt, WL, WG, arguments, MulCo, cms, powder=False):
    """The eight lines of Ham_mosaic(*arguments, cms) (hamiltonian_dual);
    *powder*: the scalar ones of the deprecated Hamilton_pc. Of a pair of
    levels that coincide the second "line" is the pair's first moment, in
    mm/s: its shape is the line's derivative in the position."""
    P, S, moment = Ham_mosaic(*arguments, cms)
    shape = np.where(moment, SLOPE, VOIGT)
    if np.any(moment):
        P = P * np.where(moment, MulCo, 1.0)[:, None, None]         # to the positions' units
    amplitude = _kpref(MulCo) * I
    if powder:
        return [(shape, (S * MulCo + delt, WL, WG), amplitude * P[:, 0, 0].real, None)]
    return [(shape, (S * MulCo + delt, WL, WG), amplitude, P)]


def _hamiltonian(p, MulCo, text, cms):
    return _hamiltonian_lines(abs(p[0]), p[1] * MulCo, abs(p[4]) * MulCo, abs(p[5]) * MulCo,
                              (p[2], p[3], p[6], p[8], p[7], p[9], p[10], p[11], p[12], p[13], p[14]), MulCo, cms)


def _hamilton_mc(p, MulCo, text, cms):
    return _hamiltonian_lines(abs(p[0]), p[1] * MulCo, abs(p[4]) * MulCo, abs(p[5]) * MulCo,
                              (p[2], p[3], p[6], p[8], p[7], p[9], p[10], p[11], 1.0, 1.0, 1.0), MulCo, cms)


def _hamilton_pc(p, MulCo, text, cms):
    return _hamiltonian_lines(abs(p[0]), p[1] * MulCo, abs(p[4]) * MulCo, abs(p[5]) * MulCo,
                              (p[2], p[3], p[6], p[8], p[7], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), MulCo, False, powder=True)


# models.K_cei and models.SN, written with numpy so that they take Duals
def _k_cei(M):
    M = abs(M)
    if M <= 0.9999978056985459:
        m = np.sqrt(1 - M)
        return np.pi * np.sqrt(2) / np.sqrt((1 + m) * np.sqrt(m)) * (
            1 - 2 ** (1 / 4) / 4 * (1 + np.sqrt(m)) / ((1 + m) * np.sqrt(m)) ** (1 / 4))
    if M < 1:
        return 1 / 2 * np.log((1 + M) / (1 - M))
    return 99999


def _sn(x, m):
    m = abs(m)
    if m <= 0.981:
        mu = np.pi / 4 / (_k_cei(m) + 10 ** -12 * (m == 0))
        t = 1 / mu * np.tan(mu * x)
        z = t ** 2
        a1 = -1 / 6 * (1 + m + 2 * mu ** 2)
        a2 = 1 / 120 * (1 + m) ** 2 + mu ** 2 / 6 * (1 + m) + m / 10 + mu ** 4 / 5
        p1 = (mu ** 4 + a1 * mu ** 2 + a1 ** 2 - a2) / (mu ** 2 + a1)
        p2 = mu ** 4
        q1 = (mu ** 4 - a2) / (mu ** 2 + a1)
        q2 = mu ** 2 * (mu ** 4 - a2) / (mu ** 2 + a1)
        q3 = mu ** 6
        return (1 + p1 * z + p2 * z ** 2) / (1 + q1 * z + q2 * z ** 2 + q3 * z ** 3) * t
    if m < 1:
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    return 1 + 0 * x


def _asm(p, MulCo, text, cms):
    """TImod's 'ASM' and models.ASM_thick_terms: at every point of the
    modulation six lines, with the matrices of the local moment."""
    sigm, eps_m, eps_lat, His, Han = p[1] * MulCo, p[2] * MulCo, p[3] * MulCo, p[4] * MulCo, p[5] * MulCo
    WL, WG = abs(p[6]) * MulCo, abs(p[7]) * MulCo
    m = p[8]
    Num = int(int(abs(p[12])) / 6) * 6 + 1
    X = np.linspace(0, _k_cei(m), Num)
    co = _sn(X, m) ** 2 if m >= 0 else 1 - _sn(X, m) ** 2
    eps = eps_m + eps_lat * (3 * co - 1) / 2
    H = His + Han * (3 * co - 1) / 2
    a1 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0)) * (co + 1 / 8 * (1 - co)) * (1 - co)
    a2 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0)) * (co - 1 / 8 * (1 - co)) * (1 - co)
    H = H / E0_J * c
    v1 = sigm + eps + a1 + mun * (3 * gex - ggr) / 2 * H
    v6 = sigm + eps - a1 - mun * (3 * gex - ggr) / 2 * H
    v2 = sigm - eps - a2 + mun * (gex - ggr) / 2 * H
    v5 = sigm - eps + a2 - mun * (gex - ggr) / 2 * H
    v3 = sigm - eps + a2 - mun * (gex + ggr) / 2 * H
    v4 = sigm - eps - a2 + mun * (gex + ggr) / 2 * H

    I1, I2, I3 = (I / Num for I in _sextet_intensities(abs(p[0]), p[13], MulCo))
    th, ph = p[9], p[10]
    ux, uy, uz = m5._axis_xyz(th, ph)                       # the easy axis u
    thr, phr, omr = th / 180.0 * np.pi, ph / 180.0 * np.pi, p[14] / 180.0 * np.pi
    etx, ety = np.cos(thr) * np.cos(phr), np.cos(thr) * np.sin(phr)
    epx, epy = -np.sin(phr), np.cos(phr)
    vx = np.cos(omr) * etx + np.sin(omr) * epx              # the second axis of the cycloid's plane
    vy = np.cos(omr) * ety + np.sin(omr) * epy
    cphi, sphi = np.sqrt(abs(co)), np.sqrt(np.maximum(0.0, 1.0 - co))
    mm = 0.5 * (_mm_perp(cphi * ux + sphi * vx, cphi * uy + sphi * vy)       # the mirror pair +/- v
                + _mm_perp(cphi * ux - sphi * vx, cphi * uy - sphi * vy))
    Md1, Mpi = _blend(1.5 * (_I2 - mm), p[11]), _blend(3.0 * mm, p[11])
    return [(VOIGT, (v, WL, WG), amplitude, M) for v, amplitude, M in (
        (v1, I1, Md1), (v6, I1, Md1), (v2, I2, Mpi), (v5, I2, Mpi), (v3, I3, Md1), (v4, I3, Md1))]


def _binned(v, dg):
    """models._bin_positions for positions that may be a Dual: each split
    between the two nodes around it and the next one on either side (the
    cubic convolution kernel's weights) of a grid of step dg from the lowest
    one. The nodes move with the lowest position and with dg, the weights with
    the positions: all in the derivatives."""
    values = np.asarray(dn.value(v))
    lowest = int(np.argmin(values))
    lo = v[lowest]
    span = np.max(values) - values[lowest]
    if span <= 0.0 or dn.value(dg) <= 0.0:
        return lo + np.zeros(1), np.ones(1)
    nbins = int(np.ceil(span / dn.value(dg))) + 3            # a node beyond each end
    x = (v - lo) * (1.0 / dg) + 1.0
    i0 = np.clip(np.floor(dn.value(x)).astype(int), 1, nbins - 3)
    frac = x - i0
    weights = np.zeros(nbins)
    slopes = np.zeros((nbins, len(frac.idx))) if isinstance(frac, dn.Dual) else None
    for node, w in ((i0 - 1, (-frac ** 3 + 2 * frac ** 2 - frac) / 2), (i0, (3 * frac ** 3 - 5 * frac ** 2 + 2) / 2),
                    (i0 + 1, (-3 * frac ** 3 + 4 * frac ** 2 + frac) / 2), (i0 + 2, (frac ** 3 - frac ** 2) / 2)):
        np.add.at(weights, node, dn.value(w))
        if slopes is not None:
            np.add.at(slopes, node, w.d)
    weights = weights / len(values)
    if slopes is not None:
        weights = dn.Dual(weights, slopes / len(values), frac.idx)
    return lo + dg * (np.arange(nbins) - 1), weights


def _scdw(p, MulCo, text, cms):
    """TImod's 'SCDW' with models.SDW_thick_terms(_direct): the six lines'
    positions along the wave, binned on a grid tied to the line width."""
    d0, eps0, H0 = p[1] * MulCo, p[2] * MulCo, p[3] * MulCo
    WL, WG = abs(p[4]) * MulCo, abs(p[5]) * MulCo
    KdH, KeH, phi = p[10], p[11], p[12]
    psi = np.arange(m5.SDW_NUM) * (2.0 * np.pi / m5.SDW_NUM)
    H = H0 + 0.0 * psi
    for i in range(8):                                  # the odd harmonics of the field; a zero one that is
        hk = p[13 + i] * MulCo                          # fitted is kept: its derivative is not zero
        if isinstance(hk, dn.Dual) or hk != 0.0:
            H = H + hk * np.sin((2 * i + 1) * psi)
    absH = abs(H)
    dl = d0 + KdH * absH
    for i in range(4):                                  # the even harmonics of the shift
        dk = p[21 + i] * MulCo
        if isinstance(dk, dn.Dual) or dk != 0.0:
            dl = dl + dk * np.sin((2 * (i + 1)) * psi + phi / 180.0 * np.pi)
    ep = eps0 + KeH * absH
    Hc = H / E0_J * c
    v = [dl + ep + mun * (3 * gex - ggr) / 2 * Hc, dl - ep + mun * (gex - ggr) / 2 * Hc,
         dl - ep - mun * (gex + ggr) / 2 * Hc, dl - ep + mun * (gex + ggr) / 2 * Hc,
         dl - ep - mun * (gex - ggr) / 2 * Hc, dl + ep - mun * (3 * gex - ggr) / 2 * Hc]
    steps = max(abs(float(dn.value(p[25]))), 1.0)
    dg = np.maximum(np.maximum(WL, WG), NAT_WIDTH * MulCo) / steps
    I1, I2, I3 = _sextet_intensities(abs(p[0]), p[26], MulCo)
    Msm, Mpi, Msp = _sextet_matrices(p[6], p[7], p[8], p[9])
    lines = []
    for positions, amplitude, M in zip(v, (I1, I2, I3, I3, I2, I1), (Msm, Mpi, Msp, Msm, Mpi, Msp)):
        nodes, weights = _binned(positions, dg)
        lines.append((VOIGT, (nodes, WL, WG), amplitude * weights, M))
    return lines


# name: (slots in p, the lines of its parameters, whether the line function
# takes arrays over the copies of a distribution -- else it is called per copy)
COMPONENTS = {'Singlet': (4, _singlet, True), 'Spectrum': (4, _spectrum, True), 'Doublet': (9, _doublet, True),
              'Sextet': (14, _sextet, True), 'MDGD': (17, _mdgd, True), 'Relax_MS': (11, _relax_ms, True),
              'Relax_2S': (14, _relax_2s, True), 'Hamiltonian': (15, _hamiltonian, False),
              'Hamilton_mc': (12, _hamilton_mc, False), 'Hamilton_pc': (9, _hamilton_pc, False),
              'ASM': (15, _asm, False), 'SCDW': (27, _scdw, False)}
# Rows that only hold slots
_SLOTS = {'Variables': numco, 'Expression': 1, 'Layer': 0}
_DISTRIBUTION_SLOTS = {'Distr': 5, 'Recon': 7}
_CORR_SLOTS = 2
_DISTR_T_SLOTS = 5


def uncovered(model):
    """The rows of *model* TI of this module does not compute (none: [])."""
    known = set(COMPONENTS) | set(_SLOTS) | set(_DISTRIBUTION_SLOTS) | {'Corr', 'Nbaseline', 'Distr_T'}
    return sorted({str(name) for name in model} - known)


# =============================================================================
# A spectrum's rows as lines
# =============================================================================
def _copies(block, chain):
    """A component's parameters *block* as arrays over the axes of the
    distributions after it. *chain*: per distribution row, in model order,
    (weights, slot, X, [(slot, values) of its Corr rows]). The last row is
    applied first and the first one last, as TImod's nested calls do: every
    row scales the amplitude by its weights, then puts its axis X and its
    Corr values in their slots."""
    block = list(block)
    for axis in reversed(range(len(chain))):
        ge, slot, X, corrs = chain[axis]
        along = (1,) * axis + (-1,) + (1,) * (len(chain) - axis - 1)
        block[0] = ge.reshape(along) * block[0]
        block[slot] = X.reshape(along)
        for target, values in corrs:
            block[target] = values.reshape(along)
    return block


def _one_copy(block, shape, index):
    """The parameters of the copy *index* of a block over the axes *shape*."""
    return [entry[tuple(i if size > 1 else 0 for i, size in zip(index, entry.shape))]
            if getattr(entry, 'ndim', 0) else entry for entry in block]


def _slot(value, slots, row):
    slot = int(value)
    if not 0 <= slot < slots:
        raise NotCovered(f"a {row} of the slot {slot}, which is not one of its component's {slots}")
    return slot


def _patches(text, p, lam, L, R, num):
    """thickness_distribution.thickness_points for parameters that may be
    Duals: the patches' relative thicknesses and their shares of the detected
    beam. None where it raises (TImod then gives NaN)."""
    X = np.linspace(L, R, int(num))
    with np.errstate(all='ignore'):
        P = eval_expr(str(text), p, X) + 0 * X
        area = np.sum(P)
        mean = np.sum(P * X) / area
        if dn.value(lam) == 0 or not (np.all(np.isfinite(dn.value(P))) and dn.value(area) > 0 and dn.value(mean) > 0):
            return None
        w = P * np.exp(-X / lam)
        return X / mean, w / np.sum(w)


# A Recon's weights that are fitted: every normalised weight w_i / sum(w)
# depends on every weight, so each line of each copy would be differentiated
# in all of them -- work growing as the square of their number. Instead the
# normalised weights go to the lines as parameters of their own (stand-ins,
# numbered from _STAND_IN: a line then depends on ONE of them), and TI turns
# the derivatives in the stand-ins into those in the weights afterwards, by
# the chain rule (_resolved).
_STAND_IN = 10 ** 9


def _stood_in(weights, deferred):
    """The Dual array *weights* with a stand-in parameter per element in the
    place of its own; (the stand-ins, the weights as they were) is added to
    *deferred*."""
    first = _STAND_IN + sum(len(idx) for idx, _ in deferred)
    idx = tuple(range(first, first + weights.v.size))
    deferred.append((idx, weights))
    return dn.Dual(weights.v, np.eye(weights.v.size).reshape(weights.v.shape + (weights.v.size,)), idx)


def _resolved(H, deferred):
    """H, whose derivatives are partly in the stand-ins of *deferred*, with
    those in the true parameters: dH/dp = sum_i dH/d(stand-in i) d(weight i)/dp."""
    if not deferred or not isinstance(H, dn.Dual):
        return H
    def columns_of(wanted):                     # H's derivatives in the parameters *wanted*, all its own
        return H.d[..., [H.idx.index(k) for k in wanted]]
    stand_ins = {k for idx, _ in deferred for k in idx}
    real = tuple(k for k in H.idx if k not in stand_ins)
    out = dn.Dual(H.v, columns_of(real), real) if real else H.v
    for idx, weights in deferred:
        out = out + dn.Dual(np.zeros_like(H.v), np.matmul(columns_of(idx), weights.d.reshape(len(idx), -1)),
                            weights.idx)
    return out


def _lines_of_rows(p, rows, V, Di, Co, Re, MulCo, Distri, Cor, Recon, cms, deferred):
    """One spectrum's model rows as TImod's loop takes them: its lines,
    [(shape, parameters, amplitude, matrix, layer, scaled)] -- *scaled* 1 for
    the rows below a Distr_T -- and the Distr_T's patches (None without one,
    False when its PDF is refused). *p* is the whole model's parameters (what
    the texts read as p[i]); V, Di, Co, Re are where this spectrum's entries
    start in p, Distri, Cor and Recon. *deferred*: a list, to which the
    fitted weights of the Recon rows are added (_stood_in)."""
    lines = []
    layer = scaled = 0
    patches = None
    i = 0
    while i < len(rows):
        name = rows[i]
        i += 1
        if name == 'Layer':
            layer += 1
        elif name in _SLOTS:
            V += _SLOTS[name]
        elif name == 'Distr_T':
            if scaled:
                raise ValueError("a spectrum can have one Distr_T only")
            patches = _patches(Distri[Di], p, p[V], p[V + 1], p[V + 2], p[V + 3]) or False
            scaled = 1
            Di += 1
            V += _DISTR_T_SLOTS
        elif name in COMPONENTS:
            slots, lines_of, takes_arrays = COMPONENTS[name]
            block = [p[V + s] for s in range(slots)]
            V += slots
            text = None
            if name == 'Spectrum':
                text = Distri[Di]
                Di += 1
            chain = []
            while i < len(rows) and rows[i] in _DISTRIBUTION_SLOTS:
                row = rows[i]
                i += 1
                X = np.linspace(p[V + 1], p[V + 2], int(p[V + 3]))
                if row == 'Distr':
                    ge = eval_expr(str(Distri[Di]), p, X) + 0 * X
                    Di += 1
                else:
                    ge = Recon[Re] if isinstance(Recon[Re], dn.Dual) else np.asarray(Recon[Re], dtype=float).ravel()
                    Re += 1
                slot = _slot(p[V], slots, row)
                V += _DISTRIBUTION_SLOTS[row]
                corrs = []
                while i < len(rows) and rows[i] == 'Corr':
                    i += 1
                    corrs.append((_slot(p[V], slots, 'Corr'), eval_expr(str(Cor[Co]), p, X) + 0 * X))
                    Co += 1
                    V += _CORR_SLOTS
                weights = ge / np.sum(ge, axis=0)
                if row == 'Recon' and isinstance(weights, dn.Dual):
                    weights = _stood_in(weights, deferred)
                chain.append((weights, slot, X, corrs))
            if not chain:
                blocks = [block]
            elif takes_arrays:
                blocks = [_copies(block, chain)]
            else:
                over = tuple(len(X) for _, _, X, _ in chain)
                block = _copies(block, chain)
                blocks = [_one_copy(block, over, index) for index in np.ndindex(over)]
            for block in blocks:
                lines += [line + (layer, scaled) for line in lines_of(block, MulCo, text, cms)]
        else:
            raise NotCovered(f"the row {name}" + (" without a component before it" if name in (
                'Distr', 'Recon', 'Corr') else ""))
    if patches is not None and layer:
        raise NotCovered("a Distr_T with Layer rows")
    return lines, patches


def _packed(lines, patches, idx):
    """The lines and patches as dual_kernels._transmission's arguments: a row
    per line (a line over the copies of a distribution: one per copy), the
    groups' matrices, and the derivatives in the parameters *idx*."""
    k = len(idx)
    shapes, par, dpar, amp, damp, group, scaled = [], [], [], [], [], [], []
    g_M, g_dM, g_layer, g_scaled = [], [], [], []
    first_group = {}                # id of a matrix -> its first group and its copies' shape
    n_groups = 0
    for shape, parameters, amplitude, M, layer, is_scaled in lines:
        own = [np.shape(dn.value(x)) for x in parameters] + [np.shape(dn.value(amplitude))]
        if M is not None:
            own.append(np.shape(dn.value(M))[:-2])
        over = np.broadcast_shapes(*own)
        copies = int(np.prod(over))
        values = np.zeros((copies, PARAMETERS))
        slopes = np.zeros((copies, PARAMETERS, k))
        for i, x in enumerate(parameters):
            values[:, i] = np.broadcast_to(dn.value(x), over).ravel()
            slopes[:, i, :] = np.broadcast_to(dn.columns(x, idx), over + (k,)).reshape(copies, k)
        shapes.append(np.broadcast_to(shape, over).ravel())
        par.append(values)
        dpar.append(slopes)
        amp.append(np.broadcast_to(dn.value(amplitude), over).ravel())
        damp.append(np.broadcast_to(dn.columns(amplitude, idx), over + (k,)).reshape(copies, k))
        scaled.append(np.full(copies, is_scaled, dtype=np.int64))
        if M is None:
            group.append(np.full(copies, -1, dtype=np.int64))
            continue
        if id(M) not in first_group:
            lead = np.shape(dn.value(M))[:-2]
            count = int(np.prod(lead))
            g_M.append(np.reshape(dn.value(M), (count, 2, 2)))
            g_dM.append(np.reshape(dn.columns(M, idx), (count, 2, 2, k)))
            g_layer.append(np.full(count, layer, dtype=np.int64))
            g_scaled.append(np.full(count, is_scaled, dtype=np.int64))
            first_group[id(M)] = (n_groups, lead)
            n_groups += count
        first, lead = first_group[id(M)]
        group.append(first + np.broadcast_to(np.arange(int(np.prod(lead))).reshape(lead), over).ravel())

    def joined(parts, empty, dtype):
        return np.ascontiguousarray(np.concatenate(parts) if parts else np.zeros(empty), dtype=dtype)
    if patches:
        t = [np.asarray(dn.value(x), dtype=float) for x in patches] + [dn.columns(x, idx) for x in patches]
    else:
        t = [np.zeros(0), np.zeros(0), np.zeros((0, k)), np.zeros((0, k))]
    return (joined(shapes, 0, np.int64), joined(par, (0, PARAMETERS), float), joined(amp, 0, float),
            joined(group, 0, np.int64), joined(scaled, 0, np.int64), joined(dpar, (0, PARAMETERS, k), float),
            joined(damp, (0, k), float), joined(g_M, (0, 2, 2), complex), joined(g_dM, (0, 2, 2, k), complex),
            joined(g_layer, 0, np.int64), joined(g_scaled, 0, np.int64),
            *[np.ascontiguousarray(x, dtype=float) for x in t])


def _node(E, W, absorber, sign, rho11, rho22):
    """A node of TI's grid: its weight W times the transmission of the
    absorber at its energies E, and the derivatives. What a pool worker runs."""
    r, dr = _transmission(E, *absorber, sign, rho11, rho22)
    return W * r, W[:, None] * dr


# =============================================================================
# The spectrum
# =============================================================================
def prepare(x_exp, model, JN, x0, MulCo, INS, Met=0, Norm=1, pol=m5.SMS_POL_DEFAULT, lengths=None, smearing=None):
    """What TI needs that does not depend on the parameters, once per fit (per
    call where the source itself is fitted): the arguments are models.TI's.
    Per spectrum (Nbaseline section) models' own bookkeeping, and at every
    node of its grid the energies the components see and the node's weight --
    the source line and the node's Jacobian (models.node_parts)."""
    rows = [str(name) for name in model]
    missing = uncovered(rows)
    if missing:
        raise NotCovered("the rows " + ", ".join(missing))
    slots = {name: COMPONENTS[name][0] for name in COMPONENTS}
    slots.update(_SLOTS, Corr=_CORR_SLOTS, Distr_T=_DISTR_T_SLOTS, Nbaseline=number_of_baseline_parameters,
                 **_DISTRIBUTION_SLOTS)
    sections = m5._spectrum_sections(x_exp, np.zeros(number_of_baseline_parameters + sum(slots[name] for name in rows)),
                                     rows, JN, x0, MulCo, INS, [0], [0], Met, Norm, pol, [0], lengths, smearing)
    if smearing is not None and not isinstance(smearing, list):
        smearing = [smearing]
    for i, s in enumerate(sections):
        s['nodes'], s['weights'] = m5.node_parts(s['x'], JN, s['x0'], s['MulCo'], s['INS'], s['Met'], pol,
                                                 None if smearing is None else smearing[i])
        s['rho'] = (0.5, 0.5) if s['Met'] == 1 else (0.5 * (1.0 + pol), 0.5 * (1.0 - pol))     # _polarised_readout's
    return sections


def fit_parameters(p, free, confu, Expr, NExpr):
    """The parameters *p* of a fit as TI takes them: the *free* ones Duals,
    the linked ones (confu) and the Expression slots (Expr into NExpr) made of
    them, in the order minimi_hi sets a trial point -- so the derivatives in a
    free parameter carry what is linked to it."""
    q = dn.seed(p, free)
    for i in range(len(q)):
        if np.any(confu[0] == i):
            link = int(np.where(confu[0] == i)[0][0])
            q[i] = q[int(confu[1][link])] * confu[2][link]
    for e in range(len(Expr)):
        q[int(NExpr[e])] = eval_expr(str(Expr[e]), q)
        for link in np.where(confu[1] == NExpr[e])[0]:
            q[int(confu[0][link])] = q[int(confu[1][link])] * confu[2][link]
    return q


def fit_jacobian(prepared, pool, Distri, Cor, Recon, fix, confu, Expr, NExpr, model=None):
    """minimi_hi's jacobian= for a fit of the model *prepared* was made for:
    jacobian(p) returns the derivatives of TI's spectrum, one row per
    parameter. The rows of the parameters that are not fixed (*fix*, the
    fit's own: linked ones and Expression slots are in it) are TOTAL
    derivatives -- what is linked to a parameter, or an Expression of it,
    moves with it; the other rows are zero, so the chain rule minimi_hi
    applies to the links adds nothing twice.

    *model*, when the fit's vector is not TI's parameters as they are (a fit
    with a Recon: its weights follow them, and its smoothness rows follow the
    spectrum): what the fit minimises as a function of that vector, a list of
    numbers and Duals -- a Dual, or the plain array for plain numbers.

    Where the exact slope of a free parameter is nothing -- it stands where
    the model is even in it (an axis across the beam), or the model does not
    read it -- the row is minimi_hi's own forward difference instead, one
    evaluation. "Nothing": over that difference's step the slope would move
    the model by less than the model's rounding. There the difference is the
    curvature times the step, small but not zero, and the fit goes on
    treating such a parameter as it does now, not as one that does nothing."""
    fixed = {int(k) for k in fix}
    eps = np.finfo(float).eps
    if model is None:
        def model(q):
            return TI(q, prepared, pool, Distri, Cor, Recon)

    def jacobian(p):
        free = [k for k in range(len(p)) if k not in fixed]
        y = model(fit_parameters(p, free, confu, Expr, NExpr))
        value = dn.value(y)
        rows = np.zeros((len(p),) + np.shape(value))
        rows[free] = dn.jacobian(y, free)
        for k in free:
            step = max(10 ** -12, abs(p[k]) / mi.JACOBIAN_STEP_DIVISOR)                     # compute_jacobian's
            if np.max(np.abs(rows[k])) * step <= eps * np.max(np.abs(value)):
                bumped = mi._bump_parameter(np.asarray(p, dtype=float), k, step, confu, Expr, NExpr)
                rows[k] = (model(list(bumped)) - value) / step
        return rows
    return jacobian


def TI(p, prepared, pool, Distri=[0], Cor=[0], Recon=[0]):
    """models.TI's spectrum for the parameters *p*, a list of numbers and
    Duals: a Dual (the spectrum and its derivatives) when one of them is, else
    the plain array. *prepared*: prepare()'s, made with TI's other arguments.
    A Recon's weights may be a Dual too."""
    if m5.COMPLEX_VOIGT_METHOD not in ('pseudo', 'off'):
        raise NotCovered(f"COMPLEX_VOIGT_METHOD = {m5.COMPLEX_VOIGT_METHOD!r}")
    sign = float(m5.DISPERSION_SIGN) if m5.COMPLEX_VOIGT_METHOD == 'pseudo' else 0.0
    tasks, spans = [], []
    for s in prepared:
        deferred = []
        lines, patches = _lines_of_rows(p, [str(name) for name in s['model']], s['V'], s['Di'], s['Co'], s['Re'],
                                        s['MulCo'], Distri, Cor, Recon, s['Met'] == 1, deferred)
        idx = dn.parameters([x for line in lines for x in line[1] + line[2:4]] + list(patches or ()))
        spans.append((len(tasks), idx, patches is False, deferred))
        if patches is not False:
            absorber = _packed(lines, patches, idx)
            tasks += [(E, W, absorber, sign, *s['rho']) for E, W in zip(s['nodes'], s['weights'])]
    results = m5._pool_starmap(pool, _node, tasks)

    spectra = []
    for s, (first, idx, refused, deferred) in zip(prepared, spans):
        if refused:                                 # a Distr_T whose PDF is refused: NaN, as TImod
            H = np.full(len(s['x']), np.nan)
        else:
            mine = results[first:first + len(s['nodes'])]
            H = np.sum([value for value, _ in mine], axis=0)
            if idx:
                H = _resolved(dn.Dual(H, np.sum([slopes for _, slopes in mine], axis=0), idx), deferred)
        spectra.append(section_counts(p, s['V'] - number_of_baseline_parameters, s, H))
    return spectra[0] if len(spectra) == 1 else np.concatenate(spectra)


def section_counts(p, V, s, H):
    """The counts of a spectrum (a section of models._spectrum_sections, *s*)
    from its node sum H with the baseline p[V:V + 8]: N0 and N1 as
    models._spectrum_sections has them, the source's normalisation of
    models._section_spectrum. p's values and H may be Duals."""
    x = s['x']
    N0 = p[V] + p[V + 3] * p[V] / 10 ** 2 * ((-1) * p[V + 1] + x) + p[V + 2] * p[V] / 10 ** 4 * ((-1) * p[V + 1] + x) ** 2
    N1 = (p[V + 4] + p[V + 7] * p[V + 4] / 10 ** 2 * ((-1) * p[V + 5] + x)
          + p[V + 6] * p[V + 4] / 10 ** 4 * ((-1) * p[V + 5] + x) ** 2)
    return m5._section_spectrum(dict(s, N0=N0, N1=N1), H)
