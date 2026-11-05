# -*- coding: utf-8 -*-
"""
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

"""

# What models.py no longer needs, kept only to be looked at and checked. The
# program never imports this module; the tests use the *_classic functions as
# the independent (eigensolver) reference for the closed form in models.py.
#   - the Hamiltonian family by diagonalisation (*_classic), superseded by
#     models._ham_tensor_core / models.Ham_mosaic;
#   - the 'Average_H' component (Angles_min, Average, meshgrid, meshgrid3);
#   - the 'Sextet(rough)' component.
# The two components are their former TImod branches, as functions of the
# transmission CH they multiplied. Here max is the builtin, not the module-level
# wrapper of models.py that numba cannot compile inside Angles_min.

import numpy as np
from numba import njit
from numpy.linalg import eigh
from numpy import abs
# 57Fe physics constants; the njit kernels below read them as globals.
from syncmoss.constants import (
    NAT_WIDTH, E0_J, c, ggr, gex, mun, TESLA_PER_MMS, LINE_RATIO_25, LINE_RATIO_34)
from syncmoss.models import Voigt, _R0_rows, _rank2_scale


# ---------------------------------------------------------------------------
# The Hamiltonian family by diagonalisation (*_classic). Not called any more:
# TImod uses the closed form (_ham_tensor_core, Ham_mosaic in models.py), which
# returns the same lines without an eigensolver. Kept as the reference and for
# reuse.
# ---------------------------------------------------------------------------

# Eigensolver version, not called; kept for reuse (closed form: _ham_tensor_core).
@njit(cache=True)
def _ham_mono_core_classic(Q, Hhf, etto, phi, tet):
    """Shared SMS-Hamiltonian core: everything EXCEPT the polarization projection.

    Builds and diagonalises the combined quadrupole + magnetic Hamiltonian and
    returns, per transition (8 of them), the polarization-INDEPENDENT spherical
    transition amplitudes (g0, g1, g2 for q = +1, 0, -1, each already carrying the
    1/4*sqrt(...) prefactor) and the line positions S (mm/s). ``Ham_mono`` (thin,
    one polarization) and ``Ham_mono_thick`` (2x2 matrix) both project these onto
    their polarization geometry; this avoids duplicating the diagonalisation.

    CONVENTION (fixed 2026-08-26). The Hamiltonians below are the textbook ones:
    H_Q = (eQV_zz/12)[3I_z^2 - I(I+1) + eta(I_x^2 - I_y^2)] and H_Z = -mu.B with B
    at (tet, phi), so the Zeeman off-diagonal <m+1|H|m> carries exp(-i*phi). The
    amplitudes g_q are the true transition matrix elements <e|T_q|g> =
    sum conj(Vex[m_e]) CG(m_g, q; m_e) Vgr[m_g] -- note conj() on the EXCITED
    (bra) side. They used to be built with the conjugation on the GROUND side,
    which returns conj(<e|T_q|g>); combined with the mirrored azimuth of the old
    polarization components (exp(+i*phi_r) on the Delta m = +1 channel) that left
    every scalar INTENSITY |A|^2 exactly right, but transposed every 2x2
    cross-section block P -> P^T, i.e. it handed the Delta m = +1 (sigma+) lines
    the sigma- Faraday (magneto-optical) sign of the Sextet family. Invisible in
    one homogeneous layer (expm(-Smat^T) = expm(-Smat)^T and the readout takes
    diagonals), it flipped the relative Faraday polarity against every other
    polarized component in mixtures, stacks and non-diagonal readouts. Both
    halves are now in the standard right-handed convention: for a pure Zeeman
    field the Delta m = +1 block equals 1.5*[(I2 - m_perp m_perp) + i m_z J]
    exactly as ``_mhat_dm1(..., +1)`` builds it for the Sextet.
    """
    Q = Q / c * E0_J
    phi = phi / 180 * np.pi
    tet = tet / 180 * np.pi
    alf = Hhf * gex * mun
    bet = Hhf * ggr * mun
    A = Q / 12
    Haex = np.array([[0.0 + 0j] * 4] * 4)
    Haex[0][0] = 3 * A - 3 / 2 * alf * np.cos(tet)
    Haex[0][1] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[0][2] = np.sqrt(3) * A * etto
    Haex[1][0] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(1j * phi)
    Haex[1][1] = -3 * A - alf / 2 * np.cos(tet)
    Haex[1][2] = -alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[1][3] = Haex[0][2]
    Haex[2][0] = Haex[0][2]
    Haex[2][1] = -alf * np.sin(tet) * np.exp(1j * phi)
    Haex[2][2] = -3 * A + alf / 2 * np.cos(tet)
    Haex[2][3] = Haex[0][1]
    Haex[3][1] = Haex[0][2]
    Haex[3][2] = Haex[1][0]
    Haex[3][3] = 3 * A + 3 / 2 * alf * np.cos(tet)

    Hagr = np.array([[0.0 + 0j] * 2] * 2)
    Hagr[0][0] = -bet / 2 * np.cos(tet)
    Hagr[0][1] = -bet / 2 * np.sin(tet) * np.exp(-1j * phi)
    Hagr[1][0] = -bet / 2 * np.sin(tet) * np.exp(1j * phi)
    Hagr[1][1] = bet / 2 * np.cos(tet)

    # Hermitian solver: orthonormal eigenvectors also inside a degenerate level.
    # The general eig returned NON-orthogonal ones for the Kramers doublets at
    # H = 0 for some eta (not all -- LAPACK rounding), which broke the sums over
    # the coincident lines: a spurious Faraday term of up to ~3% of the depth of
    # a thick single-crystal spectrum (fixed 2026-11-04).
    Hex, Vex = eigh(Haex)
    Hgr, Vgr = eigh(Hagr)
    Hex = np.real(Hex)
    Hgr = np.real(Hgr) 

    g0 = np.array([0.0 + 0j] * 8)
    g1 = np.array([0.0 + 0j] * 8)
    g2 = np.array([0.0 + 0j] * 8)
    # <e|T_q|g> = sum_{m_e, m_g} conj(Vex[m_e]) * CG(1/2 m_g; 1 q | 3/2 m_e) * Vgr[m_g].
    # Basis order: excited m_e = 3/2, 1/2, -1/2, -3/2; ground m_g = 1/2, -1/2.
    for i in range(0, 8):
        g0[i] = (np.sqrt(1 / 3) * np.conjugate(Vex[1][i - 4 * (i // 4)]) * Vgr[1][i // 4]
                 + np.conjugate(Vex[0][i - 4 * (i // 4)]) * Vgr[0][i // 4]) \
                    * (1 / 4) * np.sqrt(3 / np.pi)

        g1[i] = (np.sqrt(2 / 3) * np.conjugate(Vex[1][i - 4 * (i // 4)]) * Vgr[0][i // 4]
                 + np.sqrt(2 / 3) * np.conjugate(Vex[2][i - 4 * (i // 4)]) * Vgr[1][i // 4]) \
                    * (1/4) * np.sqrt(3*2/np.pi)

        g2[i] = (np.sqrt(1 / 3) * np.conjugate(Vex[2][i - 4 * (i // 4)]) * Vgr[0][i // 4]
                 + np.conjugate(Vex[3][i - 4 * (i // 4)]) * Vgr[1][i // 4]) \
                    * (1 / 4) * np.sqrt(3 / np.pi)

    S = np.array([float(0)] * 8)
    for i in range(0, 8):
        S[i] = Hex[i - 4 * (i // 4)] - Hgr[i // 4]
    S = S / E0_J * c

    return (g0, g1, g2, S)


# Eigensolver version, not called; kept for reuse (closed form: _ham_tensor_core).
@njit(cache=True)
def Ham_mono_classic(Q, Hhf, etto, phi, tet, phir, tetr):
    """SMS single-crystal line intensities/positions for one linear polarization.

    Thin-sample projection of the shared core onto the radiation magnetic field
    at (tetr, phir): the per-transition intensity I[k] = pi*|<e|I.h|g>|^2.

    F2/F4/F6 are the (-1)^q h_{-q} polarization factors of the q = +1, 0, -1
    channels for h at (tetr, phir): c_{+-1}/c_0 = -+(1/sqrt2) sin(tetr)
    exp(-+i*phir), the standard right-handed convention (see _ham_mono_core_classic).
    The returned INTENSITIES are unchanged by that convention fix -- the old
    (mirrored) F's compensated the old (conjugated) g's exactly.
    """
    g0, g1, g2, S = _ham_mono_core_classic(Q, Hhf, etto, phi, tet)

    phir = phir / 180 * np.pi
    tetr = tetr / 180 * np.pi
    F2 = np.sqrt(2) * np.sin(tetr) * (-1j) * np.exp(-1j * (phir))
    F4 = np.sqrt(2) * np.cos(tetr) * (1j)
    F6 = np.sqrt(2) * np.sin(tetr) * (1j) * np.exp(1j * (phir))

    E = g0 * F2 + g1 * F4 + g2 * F6
    I = np.real(E * np.conjugate(E)) * np.pi

    return (I, S)

# Eigensolver version, not called; kept for reuse (closed form: _ham_tensor_core).
@njit(cache=True)
def Ham_mono_CMS_classic(Q, Hhf, etto, phi, tet, phir, tetr):
    """UNUSED (no caller): thin CMS single-crystal intensities, kept for reference.

    Duplicates the Hamiltonian construction instead of sharing
    ``_ham_mono_core_classic``, and still carries the old (conjugated) amplitude
    convention -- harmless because it only ever returns |amplitude|^2 sums, which
    that convention leaves exact, but do NOT build a 2x2 matrix from it without
    switching it over first (see _ham_mono_core_classic). Delete with the deprecated
    'Hamilton_mc'/'Hamilton_pc' branches.
    """
    Q = Q / c * E0_J
    phi = phi / 180 * np.pi
    tet = tet / 180 * np.pi
    alf = Hhf * gex * mun
    bet = Hhf * ggr * mun
    A = Q / 12
    Haex = np.array([[0.0 + 0j] * 4] * 4)
    Haex[0][0] = 3 * A - 3 / 2 * alf * np.cos(tet)
    Haex[0][1] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[0][2] = np.sqrt(3) * A * etto
    Haex[1][0] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(1j * phi)
    Haex[1][1] = -3 * A - alf / 2 * np.cos(tet)
    Haex[1][2] = -alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[1][3] = Haex[0][2]
    Haex[2][0] = Haex[0][2]
    Haex[2][1] = -alf * np.sin(tet) * np.exp(1j * phi)
    Haex[2][2] = -3 * A + alf / 2 * np.cos(tet)
    Haex[2][3] = Haex[0][1]
    Haex[3][1] = Haex[0][2]
    Haex[3][2] = Haex[1][0]
    Haex[3][3] = 3 * A + 3 / 2 * alf * np.cos(tet)

    Hagr = np.array([[0.0 + 0j] * 2] * 2)
    Hagr[0][0] = -bet / 2 * np.cos(tet)
    Hagr[0][1] = -bet / 2 * np.sin(tet) * np.exp(-1j * phi)
    Hagr[1][0] = -bet / 2 * np.sin(tet) * np.exp(1j * phi)
    Hagr[1][1] = bet / 2 * np.cos(tet)

    Hex, Vex = eigh(Haex)
    Hgr, Vgr = eigh(Hagr)
    Hex = np.real(Hex)
    Hgr = np.real(Hgr)

    phir = phir / 180 * np.pi
    tetr = tetr / 180 * np.pi

    aM = np.array([[[0.0 + 0j] * 2] * 8] * 3)
    for i in range(0, 8):
        aM[0][i][0] = (np.sqrt(1 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4]) + Vex[0][
            i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4])) \
                      * (1 / 4) * np.sqrt(3 / np.pi) * np.exp(1j * phir)
        aM[0][i][1] = (np.sqrt(1 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4]) + Vex[0][
            i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4])) \
                      * (1 / 4) * np.sqrt(3 / np.pi) * np.exp(1j * phir) * 1j * np.cos(tetr)

        aM[1][i][1] = (np.sqrt(2 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + np.sqrt(2 / 3) *
                       Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                      * (1 / 4) * np.sqrt(3 * 2 / np.pi) * 1j * np.sin(tetr)

        aM[2][i][0] = (np.sqrt(1 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + Vex[3][
            i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                      * (1 / 4) * np.sqrt(3 / np.pi) * np.exp(-1j * phir)
        aM[2][i][1] = (np.sqrt(1 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + Vex[3][
            i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                      * (1 / 4) * np.sqrt(3 / np.pi) * np.exp(-1j * phir) * (-1j) * np.cos(tetr)

    S = np.array([float(0)] * 8)
    for i in range(0, 8):
        S[i] = Hex[i - 4 * (i // 4)] - Hgr[i // 4]
    S = S / E0_J * c

    E = aM[0] + aM[1] + aM[2]
    I = np.sum(np.real(E * np.conjugate(E)), axis=1)
    I = I * np.pi

    return (I, S)

# Eigensolver version, not called; kept for reuse (closed form: Ham_mosaic at 0, 0, 0).
@njit(cache=True)
def Ham_poly_classic(Q, Hhf, etto, phi, tet):
    """DEPRECATED: random-powder Hamiltonian line intensities/positions.

    Only the deprecated scalar 'Hamilton_pc' component (and its line-position
    twin in models_positions) still calls this. ``Ham_mosaic(..., 0, 0, 0, cms)``
    reproduces it exactly (I_poly[k] * I2 per transition, verified to 3e-17), so
    it survives only as the independent powder reference for the tests. Like
    ``Ham_mono_CMS`` it duplicates the Hamiltonian construction and keeps the old
    (conjugated) amplitude convention, which is exact for the |amplitude|^2 sums
    it returns but must not be used to build a 2x2 matrix.
    """
    Q = Q / c * E0_J
    phi = phi / 180 * np.pi
    tet = tet / 180 * np.pi
    alf = Hhf * gex * mun
    bet = Hhf * ggr * mun
    A = Q / 12
    Haex = np.array([[0.0 + 0j] * 4] * 4)
    Haex[0][0] = 3 * A - 3 / 2 * alf * np.cos(tet)
    Haex[0][1] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[0][2] = np.sqrt(3) * A * etto
    Haex[1][0] = -np.sqrt(3) / 2 * alf * np.sin(tet) * np.exp(1j * phi)
    Haex[1][1] = -3 * A - alf / 2 * np.cos(tet)
    Haex[1][2] = -alf * np.sin(tet) * np.exp(-1j * phi)
    Haex[1][3] = Haex[0][2]
    Haex[2][0] = Haex[0][2]
    Haex[2][1] = -alf * np.sin(tet) * np.exp(1j * phi)
    Haex[2][2] = -3 * A + alf / 2 * np.cos(tet)
    Haex[2][3] = Haex[0][1]
    Haex[3][1] = Haex[0][2]
    Haex[3][2] = Haex[1][0]
    Haex[3][3] = 3 * A + 3 / 2 * alf * np.cos(tet)

    Hagr = np.array([[0.0 + 0j] * 2] * 2)
    Hagr[0][0] = -bet / 2 * np.cos(tet)
    Hagr[0][1] = -bet / 2 * np.sin(tet) * np.exp(-1j * phi)
    Hagr[1][0] = -bet / 2 * np.sin(tet) * np.exp(1j * phi)
    Hagr[1][1] = bet / 2 * np.cos(tet)

    Hex, Vex = eigh(Haex)
    Hgr, Vgr = eigh(Hagr)
    Hex = np.real(Hex)
    Hgr = np.real(Hgr)

    aM = np.array([[0.0 + 0j] * 8] * 3)
    for i in range(0, 8):
        aM[0][i] = (np.sqrt(1 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4]) \
                    + Vex[0][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4])) / 2
        aM[1][i] = (np.sqrt(2 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) \
                    + np.sqrt(2 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) / 2
        aM[2][i] = (np.sqrt(1 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) \
                    + Vex[3][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) / 2

    S = np.array([float(0)] * 8)
    for i in range(0, 8):
        S[i] = Hex[i - 4 * (i // 4)] - Hgr[i // 4]
    S = S / E0_J * c

    I = np.real(aM[0] * np.conjugate(aM[0]) + aM[1] * np.conjugate(aM[1]) + aM[2] * np.conjugate(aM[2]))

    return (I, S)


# Eigensolver version, not called; kept for reuse (closed form: Ham_mosaic at 1, 1, 1).
@njit(cache=True)
def Ham_mono_thick_classic(Q, Hhf, etto, phi, tet, phir, tetr, alfak):
    """Per-transition 2x2 cross-section matrices for the SMS Hamiltonian.

    DEPRECATED (2026-08-26): only the deprecated 'Hamilton_mc' component uses
    this. ``Ham_mosaic(..., 1, 1, 1, False)`` reproduces it exactly (to
    round-off) and is what the current 'Hamiltonian' component calls. Kept as the
    independent single-crystal reference for the tests; remove together with the
    'Hamilton_mc' branch.

    Mirrors ``Ham_mono`` (single-crystal, SMS) but, instead of a scalar
    intensity per transition, returns the 8 polarization matrices P[k] in the
    (e1=h, e2) basis. ``(phir, tetr)`` give the radiation magnetic field h in
    the EFG (PAS) frame and ``alfak`` is the rotation of the beam k around h
    that fixes the second polarization e2 = k x h. P[k][0,0] equals exactly the
    scalar ``Ham_mono`` intensity I[k] for the same (phir, tetr), so the thin
    limit reduces to the scalar SMS model.
    """
    g0, g1, g2, S = _ham_mono_core_classic(Q, Hhf, etto, phi, tet)

    # e1 = h at (tetr, phir); e2 = k x h with k rotated by alfak around h.
    tetr1 = tetr / 180 * np.pi
    phir1 = phir / 180 * np.pi
    ak = alfak / 180 * np.pi
    that = np.array([np.cos(tetr1) * np.cos(phir1), np.cos(tetr1) * np.sin(phir1), -np.sin(tetr1)])
    phat = np.array([-np.sin(phir1), np.cos(phir1), 0.0])
    e2 = np.sin(ak) * that - np.cos(ak) * phat
    tet2 = np.arccos(e2[2])
    phi2 = np.arctan2(e2[1], e2[0])

    F2a = np.sqrt(2) * np.sin(tetr1) * (-1j) * np.exp(-1j * phir1)
    F4a = np.sqrt(2) * np.cos(tetr1) * (1j)
    F6a = np.sqrt(2) * np.sin(tetr1) * (1j) * np.exp(1j * phir1)
    F2b = np.sqrt(2) * np.sin(tet2) * (-1j) * np.exp(-1j * phi2)
    F4b = np.sqrt(2) * np.cos(tet2) * (1j)
    F6b = np.sqrt(2) * np.sin(tet2) * (1j) * np.exp(1j * phi2)

    A1 = np.array([0.0 + 0j] * 8)
    A2 = np.array([0.0 + 0j] * 8)
    for i in range(0, 8):
        A1[i] = g0[i] * F2a + g1[i] * F4a + g2[i] * F6a
        A2[i] = g0[i] * F2b + g1[i] * F4b + g2[i] * F6b

    P = np.zeros((8, 2, 2), dtype=np.complex128)
    for i in range(0, 8):
        P[i, 0, 0] = np.pi * (A1[i] * np.conjugate(A1[i]))
        P[i, 0, 1] = np.pi * (A1[i] * np.conjugate(A2[i]))
        P[i, 1, 0] = np.pi * (A2[i] * np.conjugate(A1[i]))
        P[i, 1, 1] = np.pi * (A2[i] * np.conjugate(A2[i]))
    return (P, S)


# Eigensolver version, not called; kept for reuse (closed form: Ham_mosaic at 1, 1, 1).
@njit(cache=True)
def Ham_mono_thick_CMS_classic(Q, Hhf, etto, phi, tet, phir, tetr):
    """Per-transition 2x2 cross-section matrices for the CMS (unpolarized) Hamiltonian.

    DEPRECATED (2026-08-26) exactly like ``Ham_mono_thick``: superseded by
    ``Ham_mosaic(..., 1, 1, 1, True)``, kept as its independent reference and for
    the deprecated 'Hamilton_mc' branch.

    Like ``Ham_mono_thick`` (same shared core, same 8 matrices P[k]), but for a
    conventional radioactive (CMS, ``Met == 1``) source, which is unpolarized.
    The meaning of ``(phir, tetr)`` therefore changes: here they give the BEAM
    direction k in the EFG (PAS) frame -- NOT the radiation magnetic field h as
    in the SMS ``Ham_mono_thick``. The two polarization basis vectors are the
    spherical-basis vectors perpendicular to k (e1 = theta_hat, e2 = phi_hat).

    There is no beam-rotation angle ``alfak``: an unpolarized source is read out
    as the half-trace (1/2)tr[expm(-Sigma)] (see the readout in ``TImod``), which
    is invariant under any rotation of (e1, e2) within the plane perpendicular to
    k. ``alfak`` would only spin that arbitrary basis, so it is redundant for a
    CMS spectrum (the user leaves it fixed). The thin limit (1/2)tr P[k] equals
    the scalar ``Ham_mono_CMS`` intensity I[k] for the same beam (phir, tetr).
    """
    g0, g1, g2, S = _ham_mono_core_classic(Q, Hhf, etto, phi, tet)

    # (phir, tetr) is the beam direction k; e1 = theta_hat, e2 = phi_hat are the
    # spherical-basis unit vectors perpendicular to k (any orthonormal pair works,
    # the half-trace readout is basis-invariant -> alfak is redundant here).
    tetr1 = tetr / 180 * np.pi
    phir1 = phir / 180 * np.pi
    e1 = np.array([np.cos(tetr1) * np.cos(phir1), np.cos(tetr1) * np.sin(phir1), -np.sin(tetr1)])
    e2 = np.array([-np.sin(phir1), np.cos(phir1), 0.0])
    tet1 = np.arccos(e1[2])
    phi1 = np.arctan2(e1[1], e1[0])
    tet2 = np.arccos(e2[2])
    phi2 = np.arctan2(e2[1], e2[0])

    F2a = np.sqrt(2) * np.sin(tet1) * (-1j) * np.exp(-1j * phi1)
    F4a = np.sqrt(2) * np.cos(tet1) * (1j)
    F6a = np.sqrt(2) * np.sin(tet1) * (1j) * np.exp(1j * phi1)
    F2b = np.sqrt(2) * np.sin(tet2) * (-1j) * np.exp(-1j * phi2)
    F4b = np.sqrt(2) * np.cos(tet2) * (1j)
    F6b = np.sqrt(2) * np.sin(tet2) * (1j) * np.exp(1j * phi2)

    A1 = np.array([0.0 + 0j] * 8)
    A2 = np.array([0.0 + 0j] * 8)
    for i in range(0, 8):
        A1[i] = g0[i] * F2a + g1[i] * F4a + g2[i] * F6a
        A2[i] = g0[i] * F2b + g1[i] * F4b + g2[i] * F6b

    P = np.zeros((8, 2, 2), dtype=np.complex128)
    for i in range(0, 8):
        P[i, 0, 0] = np.pi * (A1[i] * np.conjugate(A1[i]))
        P[i, 0, 1] = np.pi * (A1[i] * np.conjugate(A2[i]))
        P[i, 1, 0] = np.pi * (A2[i] * np.conjugate(A1[i]))
        P[i, 1, 1] = np.pi * (A2[i] * np.conjugate(A2[i]))
    return (P, S)


# Eigensolver version, not called; kept for reuse (closed form: Ham_mosaic).
@njit(cache=True)
def Ham_mosaic_classic(Q, Hhf, etto, phi, tet, tetr, phir, alfak, Atex, Am, Ah, cms):
    """Per-transition 2x2 cross-section matrices of the mosaic textured Hamiltonian.

    Returns ``(P, S)`` exactly like ``Ham_mono_thick``: ``P[k]`` the ODF-averaged
    pi*<A A^dagger> block of transition k in the (e1, e2) basis and ``S[k]`` the
    line positions (mm/s). ONE diagonalisation -- the orientation average is
    closed in the three order parameters (see the block comment above
    models._R0_rows).
    ``(Atex, Am, Ah) = (1, 1, 1)`` reproduces ``Ham_mono_thick`` (SMS) or
    ``Ham_mono_thick_CMS`` (CMS) to round-off; ``(0, 0, 0)`` the random powder
    ``Ham_poly``; ``Ah = 0`` a fiber texture about the reference axis.
    """
    g0, g1, g2, S = _ham_mono_core_classic(Q, Hhf, etto, phi, tet)
    R0 = _R0_rows(tetr, phir, alfak, cms)
    # Lab reference (fiber) axis f: h = e1 for SMS, the beam k for CMS.
    fl = np.zeros(3)
    if cms:
        fl[2] = 1.0
    else:
        fl[0] = 1.0
    n0 = np.ascontiguousarray(R0[2] if cms else R0[0])   # same axis in EFG coords

    # order parameters -> the four moments of the two ODF stages
    c2 = (1.0 + 2.0 * Atex) / 3.0                # <cos^2 chi>
    if c2 < 0.0:                                 # guard an out-of-range A
        c2 = 0.0
    c1 = Am * np.sqrt(c2)                        # <cos chi> = S1
    r1 = Ah                                      # <cos alpha>
    r2 = Ah * Ah                                 # <cos 2 alpha>
    lam0 = 0.5 * (3.0 * c2 - 1.0)                # <d2_00> = <P2(cos chi)> = A
    lam1 = 0.5 * (2.0 * c2 + c1 - 1.0)           # <d2_11>
    lam2 = 0.25 * (1.0 + 2.0 * c1 + c2)          # <d2_22>

    # l = 1 Cartesian moment matrix <R> = <W> R0 <Z> (for the axial/Faraday part)
    AW = np.empty((3, 3))
    AZ = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            AW[i, j] = c1 * fl[i] * fl[j] + 0.5 * (1.0 + c1) * (dij - fl[i] * fl[j])
            AZ[i, j] = n0[i] * n0[j] + r1 * (dij - n0[i] * n0[j])
    Rm = AW @ R0 @ AZ

    P = np.zeros((8, 2, 2), dtype=np.complex128)
    for n in range(8):
        # complex dipole vector d in EFG coords: A(e) = sqrt(2)*1j * (d . e), so
        # P = pi*A A^dagger = 2*pi * d d^dagger (see _ham_mono_core_classic for the F_q).
        dx = g2[n] - g0[n]
        dy = 1j * (g0[n] + g2[n])
        dz = g1[n] + 0j
        s = (dx * np.conjugate(dx) + dy * np.conjugate(dy) + dz * np.conjugate(dz)).real
        # axial (antisymmetric, Faraday) vector  w_c = Im(d_a conj(d_b)) eps_abc
        w = np.array([(dy * np.conjugate(dz)).imag,
                      (dz * np.conjugate(dx)).imag,
                      (dx * np.conjugate(dy)).imag])
        # symmetric traceless part
        dr = np.array([dx.real, dy.real, dz.real])
        di = np.array([dx.imag, dy.imag, dz.imag])
        T = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                T[i, j] = dr[i] * dr[j] + di[i] * di[j] - (s / 3.0 if i == j else 0.0)
        wl = Rm @ w                                       # <R> w, in lab coords
        T1 = _rank2_scale(T, n0, 1.0, r1, r2)             # azimuth stage (about n0, EFG)
        T2 = R0 @ T1 @ np.ascontiguousarray(R0.T)         # to the lab frame
        T3 = _rank2_scale(T2, fl, lam0, lam1, lam2)       # wobble stage (about f, lab)
        # transverse 2x2 block of  2*pi * [ s/3*I + T3 + 1j*(axial from wl) ]
        P[n, 0, 0] = 2.0 * np.pi * (s / 3.0 + T3[0, 0])
        P[n, 1, 1] = 2.0 * np.pi * (s / 3.0 + T3[1, 1])
        P[n, 0, 1] = 2.0 * np.pi * (T3[0, 1] + 1j * wl[2])
        P[n, 1, 0] = 2.0 * np.pi * (T3[0, 1] - 1j * wl[2])
    return (P, S)


# ---------------------------------------------------------------------------
# 'Average_H'
# ---------------------------------------------------------------------------

# def Average(X, MulCo, Sig, eps, H, Ep, L, G, alf):
#     Num = len(H)
#     X, H = np.meshgrid(X, H)
#     S1 = (-1) * (Sig - H / 2 + eps) * MulCo + X
#     S2 = (-1) * (Sig - LINE_RATIO_25 * H / 2 - eps) * MulCo + X
#     S3 = (-1) * (Sig - LINE_RATIO_34 * H / 2 - eps) * MulCo + X
#     S4 = (-1) * (Sig + LINE_RATIO_34 * H / 2 - eps) * MulCo + X
#     S5 = (-1) * (Sig + LINE_RATIO_25 * H / 2 - eps) * MulCo + X
#     S6 = (-1) * (Sig + H / 2 + eps) * MulCo + X
#
#     PDF = np.sin(alf)
#     I2 = 1/2 * PDF * Ep
#     I1 = 1/2 * PDF * (1-Ep) * 3/4
#     I3 = 1/2 * PDF * (1-Ep) * 1/4
#     I6 = I1
#     I5 = I2
#     I4 = I3
#
#     spc = (I1[:, None]*Voigt(L, G, S1) + I2[:, None]*Voigt(L, G, S2)\
#          + I3[:, None]*Voigt(L, G, S3) + I4[:, None]*Voigt(L, G, S4)\
#          + I5[:, None]*Voigt(L, G, S5) + I6[:, None]*Voigt(L, G, S6)).sum(axis=0)
#
#     return(spc/Num*np.pi/2)

@njit(cache=True)#(nopython=True)
def meshgrid(x, y):
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for i in range(x.size):
        for j in range(y.size):
                xx[j, i] = x[i]  # change to x[k] if indexing xy
                yy[j, i] = y[j]  # change to y[j] if indexing xy
    return xx, yy

@njit(cache=True)
def meshgrid3(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[k,j,i] = x[k]  # change to x[k] if indexing xy
                yy[k,j,i] = y[j]  # change to y[j] if indexing xy
                zz[k,j,i] = z[i]  # change to z[i] if indexing xy
    return zz, yy, xx

@njit(cache=True)
def Average(X, MulCo, Sig, eps, H, Ep, L, G, alf):
    N = len(H)
    X, H = meshgrid(X, H)
    S1 = (-1) * (Sig - H / 2 + eps) * MulCo + X
    S2 = (-1) * (Sig - LINE_RATIO_25 * H / 2 - eps) * MulCo + X
    S3 = (-1) * (Sig - LINE_RATIO_34 * H / 2 - eps) * MulCo + X
    S4 = (-1) * (Sig + LINE_RATIO_34 * H / 2 - eps) * MulCo + X
    S5 = (-1) * (Sig + LINE_RATIO_25 * H / 2 - eps) * MulCo + X
    S6 = (-1) * (Sig + H / 2 + eps) * MulCo + X

    PDF = np.sin(alf)
    I2 = 1/2 * PDF * Ep
    I1 = 1/2 * PDF * (1-Ep) * 3/4
    I3 = 1/2 * PDF * (1-Ep) * 1/4

    spc = (np.reshape(I1, (-1, 1))*Voigt(L, G, S1) + np.reshape(I2, (-1, 1))*Voigt(L, G, S2)\
         + np.reshape(I3, (-1, 1))*Voigt(L, G, S3) + np.reshape(I3, (-1, 1))*Voigt(L, G, S4)\
         + np.reshape(I2, (-1, 1))*Voigt(L, G, S5) + np.reshape(I1, (-1, 1))*Voigt(L, G, S6)).sum(axis=0)

    return(spc/N*np.pi/2)

@njit(cache=True)
def Angles_min(tet, N, K, J, Hin, Hex, X, L, G, eps, Sig):
    alf1 = np.array([0, np.pi / 2 / N, np.pi / 2 / N * (N - 1), np.pi / 2])
    bet1, gam1 = np.linspace(0, np.pi / 2, 100), np.linspace(-np.pi / 2, np.pi, 300)
    b1, a1, c1 = meshgrid3(bet1, alf1, gam1)

    F = -Hex * np.cos(b1) - Hex * np.cos(c1) + K * (np.sin(a1 - b1)) ** 2 + K * (np.sin(a1 + c1)) ** 2 - J * np.cos(
        b1 + c1)

    Bs = np.array([float(0)] * len(F))
    Cs = np.array([float(0)] * len(F))
    for i in range(0, len(F)):
        D = np.argmin(F[i])
        Bs[i] = bet1[int(D // len(F[i][0]))]
        Cs[i] = gam1[int(D % len(F[i][0]))]

    alf = 0
    b, c = meshgrid(np.linspace(Bs[0] - np.pi / 100, Bs[0] + np.pi / 100, 100),
                       np.linspace(Cs[0] - np.pi / 100, Cs[0] + np.pi / 100, 100))
    B = np.array([float(0)] * N)
    C = np.array([float(0)] * N)
    Mi = np.array([float(0)] * N)

    F = -Hex * np.cos(b) - Hex * np.cos(c) + K * (np.sin(alf - b)) ** 2 + K * (np.sin(alf + c)) ** 2 - J * np.cos(b + c)
    D = np.argmin(F)
    B[0] = b[0][int(D % len(F[0]))]
    C[0] = c[:, 0][int(D // len(F[0]))]
    Mi[0] = F.min()
    alf += np.pi / 2 / N

    b += Bs[1] - Bs[0]
    c += Cs[1] - Cs[0]


    for i in range(1, N):
        F = -Hex * np.cos(b) - Hex * np.cos(c) + K * (np.sin(alf - b)) ** 2 + K * (np.sin(alf + c)) ** 2 - J * np.cos(
            b + c)
        D = np.argmin(F)
        B[i] = b[0][int(D % len(F[0]))]
        C[i] = c[:, 0][int(D // len(F[0]))]
        Mi[i] = F.min()
        alf += np.pi / 2 / N
        b = (b - (b[0][0] + b[0][-1]) / 2) / abs(b[0][-1] - b[0][0]) * 2 \
            * 2 * max(abs(B[i] - B[i - 1]), np.pi / 100) \
            + (b[0][0] + b[0][-1]) / 2 \
            + B[i] - B[i - 1]
        c = (c - (c[:, 0][0] + c[:, 0][-1]) / 2) / abs(c[:, 0][-1] - c[:, 0][0]) * 2 \
            * 2 * max(abs(C[i] - C[i - 1]), np.pi / 100) \
            + (c[:, 0][0] + c[:, 0][-1]) / 2 \
            + C[i] - C[i - 1]

    alf = np.pi / 2
    b, c = meshgrid(np.linspace(Bs[3] - np.pi / 100, Bs[3] + np.pi / 100, 100),
                       np.linspace(Cs[3] - np.pi / 100, Cs[3] + np.pi / 100, 100))
    B2 = np.array([float(0)] * N)
    C2 = np.array([float(0)] * N)
    Mi2 = np.array([float(0)] * N)
    F = -Hex * np.cos(b) - Hex * np.cos(c) + K * (np.sin(alf - b)) ** 2 + K * (np.sin(alf + c)) ** 2 - J * np.cos(b + c)
    D = np.argmin(F)
    B2[-1] = b[0][int(D % len(F[0]))]
    C2[-1] = c[:, 0][int(D // len(F[0]))]
    Mi2[-1] = F.min()

    alf += np.pi / 2 / N

    b += Bs[2] - Bs[3]
    c += Cs[2] - Cs[3]

    for i in range(1, N):
        F = -Hex * np.cos(b) - Hex * np.cos(c) + K * (np.sin(alf - b)) ** 2 + K * (np.sin(alf + c)) ** 2 - J * np.cos(
            b + c)
        D = np.argmin(F)
        B2[-i - 1] = b[0][int(D % len(F[0]))]
        C2[-i - 1] = c[:, 0][int(D // len(F[0]))]
        Mi2[-i - 1] = F.min()
        alf -= np.pi / 2 / N
        b = (b - (b[0][0] + b[0][-1]) / 2) / abs(b[0][-1] - b[0][0]) * 2 \
            * 2 * max(abs(B2[-i - 1] - B2[-i]), np.pi / 100) \
            + (b[0][0] + b[0][-1]) / 2 \
            + B2[-i - 1] - B2[-i]
        c = (c - (c[:, 0][0] + c[:, 0][-1]) / 2) / abs(c[:, 0][-1] - c[:, 0][0]) * 2 \
            * 2 * max(abs(C2[-i - 1] - C2[-i]), np.pi / 100) \
            + (c[:, 0][0] + c[:, 0][-1]) / 2 \
            + C2[-i - 1] - C2[-i]

    alf = np.linspace(0, np.pi / 2, N)


    for i in range(0, N):
        B[i] = B[i] * (Mi[i] < Mi2[i]) + B2[i] * (Mi[i] >= Mi2[i])
        C[i] = C[i] * (Mi[i] < Mi2[i]) + C2[i] * (Mi[i] >= Mi2[i])


    H1 = np.sqrt(Hex ** 2 + Hin ** 2 + 2 * Hex * Hin * np.cos(B))
    H2 = np.sqrt(Hex ** 2 + Hin ** 2 + 2 * Hex * Hin * np.cos(C))


    BC = np.concatenate((B, C))
    H = np.concatenate((H1, H2))

    CosE = (H ** 2 + Hex ** 2 - Hin ** 2) / (2 * H * Hex + 1 * (Hex == 0) + 1 * (H == 0)) * (Hex != 0) + (
                Hex == 0) * np.cos(BC) * np.sign(Hin)

    Ep = 1 / 2 * np.sin(tet) ** 2 * (1 - CosE ** 2) + np.cos(tet) ** 2 * CosE ** 2
    alf = np.concatenate((alf, alf))

    X, H = meshgrid(X, H)
    S1 = (-1) * (Sig - H / 2 + eps) + X
    S2 = (-1) * (Sig - LINE_RATIO_25 * H / 2 - eps) + X
    S3 = (-1) * (Sig - LINE_RATIO_34 * H / 2 - eps) + X
    S4 = (-1) * (Sig + LINE_RATIO_34 * H / 2 - eps) + X
    S5 = (-1) * (Sig + LINE_RATIO_25 * H / 2 - eps) + X
    S6 = (-1) * (Sig + H / 2 + eps) + X

    PDF = np.sin(alf)
    I2 = 1 / 4 * PDF * Ep
    I1 = 1 / 4 * PDF * (1 - Ep) * 3 / 4
    I3 = 1 / 4 * PDF * (1 - Ep) * 1 / 4

    spc = (np.reshape(I1, (-1, 1)) * Voigt(L, G, S1) + np.reshape(I2, (-1, 1)) * Voigt(L, G, S2) \
           + np.reshape(I3, (-1, 1)) * Voigt(L, G, S3) + np.reshape(I3, (-1, 1)) * Voigt(L, G, S4) \
           + np.reshape(I2, (-1, 1)) * Voigt(L, G, S5) + np.reshape(I1, (-1, 1)) * Voigt(L, G, S6)).sum(axis=0)

    return (spc / N * np.pi / 2)


def Average_H(CH, p, V, E, MulCo):
    """The former 'Average_H' branch of TImod: CH times the component's
    transmission, its 11 parameters read from p[V]."""
    I = abs(p[V])
    Sig = p[V+1] * MulCo
    Q = p[V+2] * MulCo
    Hin = p[V+3] / TESLA_PER_MMS * (-1) * MulCo
    WL = abs(p[V+4]) * MulCo
    WG = abs(p[V+5]) * MulCo
    Hex = p[V+6] / TESLA_PER_MMS * MulCo
    K = p[V+7] * MulCo
    J = p[V+8] * MulCo
    tet = p[V+9] / 180 * np.pi
    Num = max(int(p[V+10]), 1)

    CHt = CH * np.exp((-1) * np.pi * (NAT_WIDTH / 2 * MulCo) * I * Angles_min(tet, Num, K, J, Hin, Hex, E, WL, WG, Q, Sig))
    return CHt


# ---------------------------------------------------------------------------
# 'Sextet(rough)'
# ---------------------------------------------------------------------------

def Sextet_rough(CH, p, V, E, MulCo):
    """The former 'Sextet(rough)' branch of TImod: CH times the component's
    transmission, its 14 parameters read from p[V]."""
    I = abs(p[V])
    I1 = I * p[V + 9]  / (1 + p[V + 9] + p[V + 10]) * p[V + 11] / (1 + p[V + 11])
    I2 = I * p[V + 10] / (1 + p[V + 9] + p[V + 10]) * p[V + 12] / (1 + p[V + 12])
    I3 = I * 1         / (1 + p[V + 9] + p[V + 10]) * p[V + 13] / (1 + p[V + 13])
    I4 = I * 1         / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 13])
    I5 = I * p[V + 10] / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 12])
    I6 = I * p[V + 9]  / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 11])
    HH = p[V + 3] / TESLA_PER_MMS
    S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 6] * MulCo + E
    S2 = (-1) * (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo + p[V + 7] * MulCo + E
    S3 = (-1) * (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo - p[V + 7] * MulCo + E
    S4 = (-1) * (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo + p[V + 7] * MulCo + E
    S5 = (-1) * (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo - p[V + 7] * MulCo + E
    S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 6] * MulCo + E
    WL = abs(p[V + 4]) * MulCo
    WG = abs(p[V + 5]) * MulCo
    GaH = abs(p[V + 8]) / 2 / TESLA_PER_MMS * MulCo
    Ga16 = (WG**2 + GaH**2)** (1/2)
    Ga25 = (WG**2 + (LINE_RATIO_25 * GaH)**2)** (1/2)
    Ga34 = (WG**2 + (LINE_RATIO_34 * GaH)**2)** (1/2)
    Voi1 = Voigt(WL, Ga16, S1)
    Voi2 = Voigt(WL, Ga25, S2)
    Voi3 = Voigt(WL, Ga34, S3)
    Voi4 = Voigt(WL, Ga34, S4)
    Voi5 = Voigt(WL, Ga25, S5)
    Voi6 = Voigt(WL, Ga16, S6)
    CHt = CH * np.exp((-1) * np.pi * (NAT_WIDTH / 2 * MulCo)\
                      * (I1*Voi1+I6*Voi6+I2*Voi2+I5*Voi5+I3*Voi3+I4*Voi4))
    return CHt
