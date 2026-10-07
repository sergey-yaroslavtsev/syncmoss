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

import numpy as np

from numpy import *
import builtins as bu
def max(*args):
    return bu.max(*args)
def min(*args):
    return bu.min(*args)
# import dual_v3 as dn
import syncmoss.minimi_lib as mi
import os
import platform
import time
import threading
from numba import njit, prange
import scipy
import scipy.linalg
dummy = scipy.linalg.eig(np.array([[1,0], [0,1]])) #required to build exe
from syncmoss.constants import number_of_baseline_parameters, numco
# 57Fe physics constants (see constants.py for values, sources and the note on
# how `mun` is derived from the alpha-Fe standard). Imported into this module's
# namespace because the njit kernels below read them as globals.
from syncmoss.constants import (
    NAT_WIDTH, E0_J, c, ggr, gex, mun, MMS_PER_T_PER_G, SMS_POL_DEFAULT,
    LINE_SHIFT_16, LINE_SHIFT_25, LINE_SHIFT_34, TESLA_PER_MMS,
    LINE_RATIO_25, LINE_RATIO_34)
from numpy.linalg import eig
from numpy import linalg as LA
# from numpy.linalg import inv
from numpy import abs
# import matplotlib.pyplot as plt
# Theoretical (simulated 57FeBO3) SMS source line shapes. The module holds no
# state and imports nothing from here, so the pool workers can re-import it
# freely; see sms_theory.ins_kind for how the INS array selects a shape.
import syncmoss.sms_theory as smst

T = 1


def erf(z):
    if isinstance(z, np.ndarray):
        x = (z[0])
    else:
        x = z
    sign = 1 if x.real >= 0 else -1
    x = abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp((-1) * x * x)
    return sign * y



def limits(pool, JN0, INS):


    def integral_INS_p(R):
        iINS = 0
        for i in range(0, int(len(INS) / 3)):
            # iINS += INS[i*3+2]**2*1/2*(INS[i*3]**2+NAT_WIDTH/2)*np.sqrt(np.pi)*(1-erf((R - INS[i*3+1])/(INS[i*3]**2+NAT_WIDTH/2)))
            iINS += INS[i * 3 + 2] ** 2 * (
                        1 - erf((R - INS[i * 3 + 1]) / np.sqrt(2) / (INS[i * 3] ** 2 + NAT_WIDTH / 2))) / 2
        return iINS

    def integral_INS_m(R):
        iINS = 0
        for i in range(0, int(len(INS) / 3)):
            # iINS += INS[i*3+2]**2*1/2*(INS[i*3]**2+NAT_WIDTH/2)*np.sqrt(np.pi)*(1+erf((R - INS[i*3+1])/(INS[i*3]**2+NAT_WIDTH/2)))
            iINS += INS[i * 3 + 2] ** 2 * (
                        1 + erf((R - INS[i * 3 + 1]) / np.sqrt(2) / (INS[i * 3] ** 2 + NAT_WIDTH / 2))) / 2
        return iINS

    sp_l = np.linspace(0, -5, 4096)
    sp_r = np.linspace(0, 5, 4096)
    sp_int_l = np.array([float(0)] * len(sp_l))
    sp_int_r = np.array([float(0)] * len(sp_r))
    if smst.ins_kind(INS) != smst.KIND_GAUSS:
        # Theoretical SMS source: the same two cumulative integrals (lower tail
        # below sp_l, upper tail above sp_r) of a unit-area S(v), vectorised.
        sp_int_l = 1.0 - smst.ins_tail_above(INS, sp_l)
        sp_int_r = smst.ins_tail_above(INS, sp_r)
    else:
        for i in range(0, len(sp_l)):
            sp_int_l[i] = integral_INS_m(sp_l[i])
            sp_int_r[i] = integral_INS_p(sp_r[i])
    # print(sp_int_l)
    # print(sp_int_r)
    Ll = -5
    Rl = 5
    x0l = 0
    x0r = 0
    for i in range(0, len(sp_l)):
        if sp_int_l[i] < 0.0005:
            Ll = sp_l[i]
            break
    for i in range(0, len(sp_r)):
        if sp_int_r[i] < 0.0005:
            Rl = sp_r[i]
            break
    for i in range(0, len(sp_l)):
        if sp_int_l[i] < 0.5:
            x0l = sp_l[i]
            break
    for i in range(0, len(sp_r)):
        if sp_int_r[i] < 0.5:
            x0r = sp_r[i]
            break
    # print(Rl)
    # print(Ll)
    x0 = (x0r + x0l) / 2
    x0 = max(-0.1, x0)
    x0 = min(0.1, x0)
    MulCo = min(2.4 / (Rl - x0), 2.4 / (x0 - Ll))
    MulCo = max(MulCo, 2)
    MulCo = min(MulCo, 4)


    x = np.linspace(-5, 5, 1024)
    model = ['Singlet', 'Singlet', 'Singlet']
    p = [1000000, 0, 0, 0, 0, 0, 0, 0, 15, -3, NAT_WIDTH, 0, 10, 0, NAT_WIDTH, 0, 20, 3, NAT_WIDTH, 0.6]
    JN = max(JN0*4, 1024)
    F = TI(x, p, model, JN, pool, x0, MulCo, INS)
    # plt.figure(dpi=300)
    # plt.plot(x, F)
    # plt.show()

    def par_integ(x, p0):
            return TI(x, p, model, JN0, pool, p0[0], p0[1], INS)
    x0, MulCo = mi.minimi_hi(par_integ, x, F, np.array([x0, MulCo], dtype=float), fix=np.array([1], dtype=int), tau0 = 0.0001, eps=10 ** -5)[0]
    x0, MulCo = mi.minimi_hi(par_integ, x, F, np.array([x0, MulCo], dtype=float), fix=np.array([0], dtype=int), tau0 = 0.0001, eps=10 ** -5)[0]
    x0, MulCo = mi.minimi_hi(par_integ, x, F, np.array([x0, MulCo], dtype=float), tau0 = 0.0001, eps=10 ** -9)[0]

    # print('x0 ', x0)
    # print('MulCo ', MulCo)
    return (x0, MulCo)

@njit(cache=True)
def sim_inv(A):
    A_rre = np.copy(A)
    piv = 0
    Er = np.identity(len(A))
    A_rre = np.concatenate((A_rre, Er), axis=-1)
    for j in range(0, len(A)):
        idxs = np.nonzero(A_rre[piv:, j])[0]
        if idxs.size == 0:
            continue
        i = piv + idxs[0]

        tmp = A_rre[piv, :]
        A_rre[piv, :] = A_rre[i, :]
        A_rre[i, :] = tmp

        A_rre[piv, :] = A_rre[piv, :] / A_rre[piv, j]

        idxs = np.nonzero(A_rre[:, j])[0].flatten()
        idxs = np.delete(idxs, piv)

        for kk in range(0, len(idxs)):
            A_rre[idxs[kk], :] -= A_rre[idxs[kk], j] * A_rre[piv, :]

        piv += 1

        if piv == A_rre.shape[0]:
            break
    return  A_rre[:, -len(A):]

@njit(parallel=True, cache=True)
def sim_inv_3d(Ab):
    M =  np.array([[[float(0)]*len(Ab[0][0])]*len(Ab[0])]*len(Ab))
    for k in prange(0, len(Ab)):
        A = Ab[k]
        A_rre = np.copy(A)
        piv = 0
        Er = np.identity(len(A))
        A_rre = np.concatenate((A_rre, Er), axis=-1)
        for j in range(0, len(A)):
            idxs = np.nonzero(A_rre[piv:, j])[0]
            if idxs.size == 0:
                continue
            i = piv + idxs[0]

            tmp = A_rre[piv, :]
            A_rre[piv, :] = A_rre[i, :]
            A_rre[i, :] = tmp

            A_rre[piv, :] = A_rre[piv, :] / A_rre[piv, j]

            idxs = np.nonzero(A_rre[:, j])[0].flatten()
            idxs = np.delete(idxs, piv)

            for kk in range(0, len(idxs)):
                A_rre[idxs[kk], :] -= A_rre[idxs[kk], j] * A_rre[piv, :]

            piv += 1

            if piv == A_rre.shape[0]:
                break
        M[k] = A_rre[:, -len(A):]
    return M

@njit(cache=True)
def inv(A, D):
    A_rre = np.copy(A)
    D_rre = np.copy(D)
    piv = 0
    Er = np.identity(len(A))
    Ed = np.zeros((len(A), len(A)))
    A_rre = np.concatenate((A_rre, Er), axis=-1)
    D_rre = np.concatenate((D_rre, Ed), axis=-1)
    for j in range(0, len(A)):
        idxs = np.nonzero(np.absolute(A_rre[piv:, j]) + np.absolute(D_rre[piv:, j]))[0]
        if idxs.size == 0:
            continue
        i = piv + idxs[0]

        tmp = A_rre[piv, :]
        A_rre[piv, :] = A_rre[i, :]
        A_rre[i, :] = tmp
        tmp = D_rre[piv, :]
        D_rre[piv, :] = D_rre[i, :]
        D_rre[i, :] = tmp

        D_rre[piv, :] = (D_rre[piv, :] * A_rre[piv, j] - A_rre[piv, :] * D_rre[piv, j]) / A_rre[piv, j] ** 2
        A_rre[piv, :] = A_rre[piv, :] / A_rre[piv, j]

        idxs = np.nonzero(np.absolute(A_rre[:, j]) + np.absolute(D_rre[:, j]))[0].flatten()
        idxs = np.delete(idxs, piv)

        for kk in range(0, len(idxs)):
            D_rre[idxs[kk], :] -= (A_rre[idxs[kk], j] * D_rre[piv, :] + D_rre[idxs[kk], j] * A_rre[piv, :])
            A_rre[idxs[kk], :] -= A_rre[idxs[kk], j] * A_rre[piv, :]

        piv += 1

        if piv == A_rre.shape[0] and piv == D_rre.shape[0]:
            break
    return  A_rre[:, -len(A):], D_rre[:, -len(A):]

@njit(cache=True)
def D2A (A, n):
    B = np.array([[float(0)]*len(A)]*int(n))
    for i in range(0, len(A)):
        for j in range(0, int(n)):
            B[j, i] = A[i]
    return(B)

@njit(cache=True)
def D3A (A, n):
    B = np.array([[[float(0)]*len(A)]*len(A[0])]*int(n))
    for i in range(0, len(A)):
        for j in range(0, len(A[0])):
            for k in range(0, int(n)):
                B[k, j, i] = A[i, j]
    return(B)

@njit(cache=True)
def insert1D (A, n, m):
    B = np.array([float(0)]*(len(A)+1))
    for i in range(0, n):
        B[i] = A[i]
    B[n] = m
    for i in range(n+1, len(B)):
        B[i] = A[i-1]
    return(B)

@njit(cache=True)
def MatMul (A, B):
    C = np.array([[float(0)]*len(B[0])]*len(A))
    for i in range(0, len(A)):
        for j in range(0, len(B[0])):
            for k in range (0, len(B)):
                C[i, j] += A[i,k]*B[j,k]
    return(C)

@njit(cache=True)
def Lin_Mat3D_Mul (A, B):
    C = np.array([[float(0)]*len(B)]*len(B[0]))
    for i in range (0, len(B)):
        for j in range(0, len(B[0])):
            for k in range (0, len(B[0][0])):
                C[j, i] += A[k]*B[i, j, k]
    return(C)

@njit(cache=True)
def sol_t(S):
    Ed = np.array([[float(1)]]*len(S))
    S = np.append(S, Ed, axis=1)

    R = np.array([float(0)]*len(S))
    for i in range(0, len(S)):
        for j in range(i, len(S)):
            if S[j][i] != 0:
                k = j
                break
        if i != k:
            tmp = np.array([float(0)] * (len(S) + 1))
            for m in range(0, len(S) + 1):
                tmp[m] = S[i][m]
                S[i][m] = S[k][m]
                S[k][m] = tmp[m]
        for m in range(i + 1, len(S)):
            t = S[m][i]
            for n in range(i, len(S) + 1):
                S[m][n] = S[m][n] - t / S[i][i] * S[i][n]
    for i in range(0, len(S)):
        R[len(S) - 1 - i] = S[len(S) - 1 - i][len(S)] / S[len(S) - 1 - i][len(S) - 1 - i]
        for j in range(0, i):
            R[len(S) - 1 - i] += -S[len(S) - 1 - i][len(S) - 1 - j] * R[len(S) - 1 - j] \
                                        / S[len(S) - 1 - i][len(S) - 1 - i]
    return(R)

# @njit#(parallel=True)
# def solution(S):  # solution of linear equations system
#     R = np.array([[float(0)] * len(S[0])] * len(S))
#     a = np.empty(len(S[0])); a.fill(1)
#     for i in range(0, len(S)):
#         R[i] = np.linalg.solve(S[i],a)
#
#     return (R)


# @njit
# def Voight(WL, WG, S):
#     W = np.power(WG**5 + 2.69269*WL*WG**4 + 2.42843*WL**2*WG**3 + 4.47163*WL**3*WG**2 + 0.07842*WL**4*WG + WL**5, 1/5)
#     et = 1.36603 * WL / W - 0.47719 * WL ** 2 / W ** 2 + 0.11116 * WL ** 3 / W ** 3
#     Voi = (et *  W / 2 / np.pi / (S ** 2 + (W / 2) ** 2) \
#         + (1 - et) *  (np.exp((-1) * (S ** 2 / (2 * (W / 2 / np.sqrt(2 * np.log(2))) ** 2))) / (W / 2 / np.sqrt(2 * np.log(2))) / np.sqrt(2 * np.pi)))
#     return(Voi)
#
# def Voight(WL, WG, S):
#     d = (WL-WG)/(WL+WG)
#     W = (WL+WG)*(1 - 0.18121*(1-d**2)-(0.023665*np.exp(0.6*d)+0.00418*np.exp(-1.9*d))*np.sin(np.pi*d))
#     cl = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
#     cg = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
#     Voi = (cl *  W / 2 / np.pi / (S ** 2 + (W / 2) ** 2) \
#         + cg *  (np.exp((-1) * (S ** 2 / (2 * (W / 2 / np.sqrt(2 * np.log(2))) ** 2))) / (W / 2 / np.sqrt(2 * np.log(2))) / np.sqrt(2 * np.pi)))
#     return(Voi)

@njit(cache=True)
def Voight(gL, gG, S): # doi.org/10.1107/S0021889800010219
    ro = gL/(gL+gG)
    WG = (1 - ro*(0.66000 + 0.15021*ro - 1.24984*ro**2 + 4.74052*ro**3 - 9.48291*ro**4 + 8.48252*ro**5 - 2.95553*ro**6)) * (gG + gL)
    WL = (1 - (1-ro)*(-0.42179 - 1.25693*ro + 10.30003*ro**2 - 23.45651*ro**3 + 29.14158*ro**4 - 16.50453*ro**5 + 3.19974*ro**6)) * (gG + gL)
    WI = (1.19913 + 1.43021*ro - 15.36331*ro**2 + 47.06071*ro**3 - 73.61822*ro**4 + 57.92559*ro**5 - 17.80614*ro**6) * (gG + gL)
    WP = (1.10186 - 0.47745*ro - 0.68688*ro**2 + 2.76622*ro**3 - 4.55466*ro**4 + 4.05475*ro**5 - 1.26571*ro**6) * (gG + gL)

    cl = ro*(1+(1-ro)*(-0.30165-1.38927*ro+9.31550*ro**2-24.10743*ro**3+34.96491*ro**4-21.18862*ro**5+3.70290*ro**6))
    ci = ro*(1-ro)*(0.25437-0.14107*ro+3.23653*ro**2-11.09215*ro**3+22.10544*ro**4-24.12407*ro**5+9.76947*ro**6)
    cp = ro*(1-ro)*(1.01579+1.50429*ro-9.21815*ro**2+23.59717*ro**3-39.71134*ro**4+32.83023*ro**5-10.02142*ro**6)
    cg = 1 - cl - ci - cp

    Voi = cl * WL / 2 / np.pi / (S ** 2 + (WL / 2) ** 2)\
        + cg * (np.exp((-1) * (S ** 2 / (2 * (WG / 2 / np.sqrt(2 * np.log(2))) ** 2))) / (WG / 2 / np.sqrt(2 * np.log(2))) / np.sqrt(2 * np.pi))\
        + ci * (1/2/(WI/2/(2**(2/3)-1)**(1/2)))*(1 + (S/(WI/2/(2**(2/3)-1)**(1/2)))**2)**(-3/2)\
        + cp * (1/2/(WP/2/np.log(2**(1/2)+1)))*(4/(np.exp(S/(WP/2/np.log(2**(1/2)+1)))+np.exp(-S/(WP/2/np.log(2**(1/2)+1))))**2)
    return(Voi)


# ===========================================================================
# Dispersive (Kramers--Kronig) completion of the beam propagation.
#
# Every THICK (2x2 matrix) component -- 'Doublet', 'Sextet', 'MDGD',
# 'Relax_MS', 'Relax_2S', 'Hamiltonian', 'ASM', 'SCDW' -- fills the cross-section
# matrix Smat with a line shape. Smat/2 is the exponent of the AMPLITUDE
# transmission operator expm(-Smat/2), and causality ties every absorptive
# profile V(E) to a dispersive partner D(E) (its Hilbert / KK transform).
# Dropping D is exact for the measured intensity ONLY when all the matrices
# commute (a single line, hyperfine axis along or transverse to the beam, a
# powder, or scalar/identity components -- whose dispersion is a global phase
# that cancels in |T|^2). For a resolved multiplet at oblique geometry, or for
# stacked layers of different azimuth, the dispersive wings (which decay as
# 1/x, not 1/x^2) accumulate a Faraday-rotation / birefringence phase between
# the resonances; at thick absorption this rotates the h-polarized beam into
# e2 in between the lines, and fitting with the truncated (absorption-only)
# model biases the geometry. The complex line shapes below restore D, so
# Smat becomes complex non-Hermitian; the amplitude operator expm(-Smat/2),
# the field-ordered layer product and the Gram readout tr[rho T^H T] are all
# already fully general (see _expm_neg / the readout in TImod) and need no
# change. Scalar (thin) components keep the real Voight().
#
# DISPERSION_SIGN fixes the sign of D relative to the +/- 1j*n_z*J Faraday
# assignment of the sigma+- matrices: every complex line shape here returns
# Lambda = V + 1j*DISPERSION_SIGN*D, verified against a numerical Hilbert
# transform to be the SAME convention in all three kernels (Voight_c in both its
# 'pseudo' and 'wofz' forms, Blume_c, _relax_MS_groups_c -- Im/H[Re] = +1.000 for
# each at +1.0). One global value is meaningful for the whole model only since
# the 2026-08-26 amplitude-convention fix: before it, the Hamiltonian family's
# blocks were transposed relative to the sextet family's, so the two families
# effectively ran with OPPOSITE pairings and no single value was right for both.
#
# In a total-intensity spectrum the sign is exactly degenerate with flipping the
# Faraday term of EVERY component (theta_h -> 180 - theta_h, equivalently
# Am -> -Am, for the axis-based models; the mirrored crystal for 'Hamiltonian'):
# a global complex conjugation of the exponent leaves every diagonal of T^H T
# unchanged. Measured: bit-identical (0.0) for a Sextet, a two-Sextet stack, a
# Relax_2S, and Relax_2S/Relax_MS + Sextet stacks. Flipping it ALONE does move a
# spectrum (a few % of the line depth: 1.8e-2 on a single magnetised sextet of
# depth 0.57, 5.0e-2 on a two-layer stack of depth 0.74), so it must be
# CALIBRATED once on a spectrum of known geometry, not fitted.
#
# TO CHANGE IT, EDIT THE LITERAL BELOW -- assigning it at runtime is NOT enough.
# `Blume_c` and `_relax_MS_groups_c` read it as a GLOBAL from inside njit code,
# so they use the value baked in when they were compiled; being njit(cache=True)
# they may even reload a stale on-disk cache, which defeats "set it before the
# first call" as well. Editing this line does work (it changes the source stamp,
# so numba recompiles): verified that all three kernels then report
# Im/H[Re] = -1.000 and that the cross-family Hamiltonian == Sextet limit still
# holds. Verified failure mode of the runtime route: with
# `models.DISPERSION_SIGN = -1.0` set immediately after import, the Voigt-shaped
# components switched but Relax_MS/Relax_2S silently kept +1 -- a model with
# MIXED dispersion signs across its components. (For the same reason the
# `m5.DISPERSION_SIGN = +1.0` lines in the tests are a no-op safeguard for those
# two components; they are only meaningful because this literal is +1.0.)
#
# COMPLEX_VOIGT_METHOD selects how Voight_c() builds the complex Voigt of the
# Voigt-shaped components. It is a DEVELOPER switch (edit here + restart to
# compare); it is deliberately NOT exposed in the GUI -- the dispersion is
# meant to be always on and invisible to the user. The relaxation components
# ('Relax_MS' via relax_MS_thick_c, 'Relax_2S' via Blume_c) get their
# dispersion straight from the resolvent, not from a Faddeeva evaluation, so
# they do not depend on this choice -- except that 'off' disables dispersion
# for them too (they fall back to the real relax_MS_thick / Blume).
#
#   'wofz'        exact complex Voigt from scipy.special.wofz (Faddeeva). Both
#                 the absorption AND the dispersion are exact; the absorption
#                 then differs from the pseudo-Voigt Voight() by the
#                 Ida-Ando-Toraya approximation error (~1e-3 of the peak).
#   'pseudo'      4-term pseudo-Voigt (Voight_complex). Its REAL part is
#                 bit-identical to Voight(), so switching dispersion on does
#                 NOT move any absorption-only quantity; only the KK dispersion
#                 is added (error vs exact D <~1% of the peak). numba-jitted.
#                 -- default: closest to "invisible to the user".
#   'pseudo_easy' 2-term pseudo-Voigt (Voight_complex_easy). Cheapest/simplest;
#                 absorption off by ~1%, dispersion off by up to ~7% in the
#                 Gaussian-dominated regime. numba-jitted.
#   'off'         real Voigt, no dispersion -- reproduces the former
#                 absorption-only thick model exactly (for A/B comparison).
# ===========================================================================
DISPERSION_SIGN = +1.0
COMPLEX_VOIGT_METHOD = 'pseudo'

from scipy.special import wofz as _wofz


def _voight_c_wofz(gL, gG, S):
    """Exact complex Voigt Lambda = V + 1j*DISPERSION_SIGN*D via the Faddeeva
    function w(z). Real part is the EXACT area-normalised Voigt that the
    pseudo-Voigt Voight() approximates; imaginary part is exactly its
    Kramers--Kronig partner. Not numba-jitted (scipy.wofz is a vectorised C
    call of negligible cost)."""
    S = np.asarray(S, dtype=np.float64)
    gamma = 0.5 * gL + 1e-9
    sigma = gG / 2.3548200450309493 + 1e-7          # FWHM -> standard deviation
    z = (S + 1j * gamma) / (sigma * np.sqrt(2.0))
    P = _wofz(z) / (sigma * np.sqrt(2.0 * np.pi))
    if DISPERSION_SIGN > 0:
        return P
    return np.conj(P)


@njit(cache=True)
def Voight_complex(gL, gG, S, C1, C2): # doi.org/10.1107/S0021889800010219
    """Complex-capable 4-term pseudo-Voigt. With (C1, C2) = (1, 0) this is the
    real Voight() bit-for-bit; with (C1, C2) = (1, 1j*DISPERSION_SIGN) it
    returns V + 1j*DISPERSION_SIGN*D, the absorptive profile plus its
    (approximate) Kramers--Kronig dispersive partner, in one call. The dispersive
    factor per component is C2/y * (x - <derivative-of-log-profile term>)."""
    delt = gG / 2 / np.sqrt(2 * np.log(2)) + 0.01 * (gG == 0)
    x = S / delt / np.sqrt(2)
    y = gL / 2 / delt / np.sqrt(2)

    ro = gL/(gL+gG)
    WG = (1 - ro*(0.66000 + 0.15021*ro - 1.24984*ro**2 + 4.74052*ro**3 - 9.48291*ro**4 + 8.48252*ro**5 - 2.95553*ro**6)) * (gG + gL)
    WL = (1 - (1-ro)*(-0.42179 - 1.25693*ro + 10.30003*ro**2 - 23.45651*ro**3 + 29.14158*ro**4 - 16.50453*ro**5 + 3.19974*ro**6)) * (gG + gL)
    WI = (1.19913 + 1.43021*ro - 15.36331*ro**2 + 47.06071*ro**3 - 73.61822*ro**4 + 57.92559*ro**5 - 17.80614*ro**6) * (gG + gL)
    WP = (1.10186 - 0.47745*ro - 0.68688*ro**2 + 2.76622*ro**3 - 4.55466*ro**4 + 4.05475*ro**5 - 1.26571*ro**6) * (gG + gL)

    cl = ro*(1+(1-ro)*(-0.30165-1.38927*ro+9.31550*ro**2-24.10743*ro**3+34.96491*ro**4-21.18862*ro**5+3.70290*ro**6))
    ci = ro*(1-ro)*(0.25437-0.14107*ro+3.23653*ro**2-11.09215*ro**3+22.10544*ro**4-24.12407*ro**5+9.76947*ro**6)
    cp = ro*(1-ro)*(1.01579+1.50429*ro-9.21815*ro**2+23.59717*ro**3-39.71134*ro**4+32.83023*ro**5-10.02142*ro**6)
    cg = 1 - cl - ci - cp

    NG = WG / 2 / np.sqrt(2 * np.log(2))
    NI = WI/2/(2**(2/3)-1)**(1/2)
    NP = WP/2/np.log(2**(1/2)+1)

    Voi = cl * WL / 2 / np.pi / (S ** 2 + (WL / 2) ** 2)\
                    *(C1 + C2 / y * (x - S/(S**2 + (WL / 2) ** 2)*delt*np.sqrt(2)))\
        + cg * (np.exp((-1) * (S ** 2 / (2 * NG ** 2))) / NG / np.sqrt(2 * np.pi)) \
                    *(C1 + C2 / y * (x - S * delt / NG ** 2 / np.sqrt(2)))\
        + ci * (1/2/NI)*(1 + (S/NI)**2)**(-3/2) \
                    *(C1 + C2 / y * (x - 3*S/NI**2/(1+(S/NI)**2)*delt/np.sqrt(2)))\
        + cp * (1/2/NP)*(4/(np.exp(S/NP)+np.exp(-S/NP))**2)\
                    *(C1 + C2 / y * (x - 2/NP*np.tanh(S/NP)*delt/np.sqrt(2)))
                    # ^ np.tanh(S/NP) is the EXACT identity for (e^{2 S/NP}-1)/(e^{2 S/NP}+1);
                    #   the raw-exponential form overflows to inf/inf = NaN in the far wings
                    #   of a narrow line, which the source integration reaches (poisoning Smat).
    return(Voi)


@njit(cache=True)
def Voight_complex_easy(WL, WG, S, C1, C2):
    """Complex-capable 2-term (Lorentzian + Gaussian) pseudo-Voigt. Same (C1,
    C2) convention as Voight_complex: (1, 0) is the older 2-term real Voigt,
    (1, 1j*DISPERSION_SIGN) is V + 1j*DISPERSION_SIGN*D. Cheapest of the three;
    least accurate in both parts."""
    delt = WG / 2 / np.sqrt(2 * np.log(2)) + 0.01 * (WG == 0)
    x = S / delt / np.sqrt(2)
    y = WL / 2 / delt / np.sqrt(2)

    d = (WL-WG)/(WL+WG)
    W = (WL+WG)*(1 - 0.18121*(1-d**2)-(0.023665*np.exp(0.6*d)+0.00418*np.exp(-1.9*d))*np.sin(np.pi*d))
    delt_n = W / 2 / np.sqrt(2 * np.log(2))
    cl = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    cg = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    Voi = cl * W / 2 / np.pi / (S ** 2 + (W / 2) ** 2) \
                    * (C1 + C2 / y * (x - S/(S**2 + (W / 2) ** 2)*delt*np.sqrt(2)))\
         + cg * (np.exp((-1) * (S ** 2 / (2 * delt_n ** 2))) / delt_n / np.sqrt(2 * np.pi))\
                    * (C1 + C2 / y * (x - S*delt/delt_n**2/np.sqrt(2)))
    return(Voi)


def Voight_c(gL, gG, S):
    """Complex line shape used by every THICK (matrix) component, dispatched on
    COMPLEX_VOIGT_METHOD. Returns V + 1j*DISPERSION_SIGN*D (or the real V for
    'off'). Drop-in complex counterpart of Voight(gL, gG, S)."""
    if COMPLEX_VOIGT_METHOD == 'wofz':
        return _voight_c_wofz(gL, gG, S)
    S = np.asarray(S, dtype=np.float64)
    if COMPLEX_VOIGT_METHOD == 'pseudo':
        return Voight_complex(gL, gG, S, 1.0, 1j * DISPERSION_SIGN)
    if COMPLEX_VOIGT_METHOD == 'pseudo_easy':
        return Voight_complex_easy(gL, gG, S, 1.0, 1j * DISPERSION_SIGN)
    return Voight(gL, gG, S) + 0j          # 'off': real Voigt, no dispersion


@njit(cache=True)
def _ham_mono_core(Q, Hhf, etto, phi, tet):
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

    Hex, Vex = eig(Haex)#LA.eig(Haex)
    Hgr, Vgr = eig(Hagr)#LA.eig(Hagr)
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


@njit(cache=True)
def Ham_mono(Q, Hhf, etto, phi, tet, phir, tetr):
    """SMS single-crystal line intensities/positions for one linear polarization.

    Thin-sample projection of the shared core onto the radiation magnetic field
    at (tetr, phir): the per-transition intensity I[k] = pi*|<e|I.h|g>|^2.

    F2/F4/F6 are the (-1)^q h_{-q} polarization factors of the q = +1, 0, -1
    channels for h at (tetr, phir): c_{+-1}/c_0 = -+(1/sqrt2) sin(tetr)
    exp(-+i*phir), the standard right-handed convention (see _ham_mono_core).
    The returned INTENSITIES are unchanged by that convention fix -- the old
    (mirrored) F's compensated the old (conjugated) g's exactly.
    """
    g0, g1, g2, S = _ham_mono_core(Q, Hhf, etto, phi, tet)

    phir = phir / 180 * np.pi
    tetr = tetr / 180 * np.pi
    F2 = np.sqrt(2) * np.sin(tetr) * (-1j) * np.exp(-1j * (phir))
    F4 = np.sqrt(2) * np.cos(tetr) * (1j)
    F6 = np.sqrt(2) * np.sin(tetr) * (1j) * np.exp(1j * (phir))

    E = g0 * F2 + g1 * F4 + g2 * F6
    I = np.real(E * np.conjugate(E)) * np.pi

    return (I, S)

@njit(cache=True)
def Ham_mono_CMS(Q, Hhf, etto, phi, tet, phir, tetr):
    """UNUSED (no caller): thin CMS single-crystal intensities, kept for reference.

    Duplicates the Hamiltonian construction instead of sharing
    ``_ham_mono_core``, and still carries the old (conjugated) amplitude
    convention -- harmless because it only ever returns |amplitude|^2 sums, which
    that convention leaves exact, but do NOT build a 2x2 matrix from it without
    switching it over first (see _ham_mono_core). Delete with the deprecated
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

    Hex, Vex = eig(Haex)
    Hgr, Vgr = eig(Hagr)
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

@njit(cache=True)
def Ham_poly(Q, Hhf, etto, phi, tet):
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

    Hex, Vex = eig(Haex)
    Hgr, Vgr = eig(Hagr)
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
#     spc = (I1[:, None]*Voight(L, G, S1) + I2[:, None]*Voight(L, G, S2)\
#          + I3[:, None]*Voight(L, G, S3) + I4[:, None]*Voight(L, G, S4)\
#          + I5[:, None]*Voight(L, G, S5) + I6[:, None]*Voight(L, G, S6)).sum(axis=0)
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

    spc = (np.reshape(I1, (-1, 1))*Voight(L, G, S1) + np.reshape(I2, (-1, 1))*Voight(L, G, S2)\
         + np.reshape(I3, (-1, 1))*Voight(L, G, S3) + np.reshape(I3, (-1, 1))*Voight(L, G, S4)\
         + np.reshape(I2, (-1, 1))*Voight(L, G, S5) + np.reshape(I1, (-1, 1))*Voight(L, G, S6)).sum(axis=0)

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

    spc = (np.reshape(I1, (-1, 1)) * Voight(L, G, S1) + np.reshape(I2, (-1, 1)) * Voight(L, G, S2) \
           + np.reshape(I3, (-1, 1)) * Voight(L, G, S3) + np.reshape(I3, (-1, 1)) * Voight(L, G, S4) \
           + np.reshape(I2, (-1, 1)) * Voight(L, G, S5) + np.reshape(I1, (-1, 1)) * Voight(L, G, S6)).sum(axis=0)

    return (spc / N * np.pi / 2)

@njit(cache=True)
def _relax_MS_groups(S, x, Sig, eps, Hv, W, R, alfa):
    """Shared many-state relaxation core: the three Delta m line groups.

    Returns (tmp1, tmp2, tmp3) — the relaxation-broadened spectra of the outer
    (Delta m = +/-1), middle (Delta m = 0) and inner (Delta m = +/-1) groups,
    BEFORE the intensity weighting. ``relax_MS`` (thin, scalar) weights them by
    the asymmetry Ah; ``relax_MS_thick`` weights them at the isotropic asymmetry
    and keeps them separate for the per-group 2x2 polarization matrices.
    """
    numb = np.linspace(0, 2 * S, int(2 * S + 1))#, dtype=int)
    numb2 = np.linspace(1, 2 * S + 1, int(2 * S + 1))#, dtype=int)


    # M11 = (-1)*R*(S*(S+1)-((numb[1:]+1)-S-1)*((numb[1:]+1)-S-2))*np.exp(alfa*(np.cos(np.pi/2*(((numb[1:]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[1:]+1)-1)-S-1)/S)**2)**2))
    # M22 = (-1)*R*(S*(S+1)-((numb[:-1]+1)-S-1)*((numb[:-1]+1)-S))*np.exp(alfa*(np.cos(np.pi/2*(((numb[:-1]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[:-1]+1)+1)-S-1)/S)**2)**2))
    ### was a mistake - exponential coefficient should be substituted by unit if "jump" to lower energy
    M111 = (-1)*R*(S*(S+1)-((numb[1:S+1]+1)-S-1)*((numb[1:S+1]+1)-S-2))*np.exp(alfa*(np.cos(np.pi/2*(((numb[1:S+1]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[1:S+1]+1)-1)-S-1)/S)**2)**2))
    M112 = (-1)*R*(S*(S+1)-((numb[S+1:]+1)-S-1)*((numb[S+1:]+1)-S-2))#*np.exp(alfa*(np.cos(np.pi/2*(((numb[S+1:]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[S+1:]+1)-1)-S-1)/S)**2)**2))
    M11 = np.concatenate((M111,M112))
    M22 = M11[::-1]

    M33 = W - M11[:-1] - M22[1:]
    M33 = insert1D(M33, 0, (W - M22[0]))
    M33 = insert1D(M33, len(numb) - 1, (W - M11[-1]))

    M441 = (D2A(x, len(numb)).transpose(1,0) - Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig - eps)
    M442 = (D2A(x, len(numb)).transpose(1,0) - LINE_RATIO_25 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)
    M443 = (D2A(x, len(numb)).transpose(1,0) - LINE_RATIO_34 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)

    V1 = np.array([[float(0)] * len(M33)] * len(x))
    V2 = np.array([[float(0)] * len(M33)] * len(x))
    V3 = np.array([[float(0)] * len(M33)] * len(x))

    for j in range(0, len(x)):
        M331 = np.copy(M33)
        M332 = np.copy(M33)
        M333 = np.copy(M33)
        r1 = np.empty(len(M331)); r1.fill(1)
        r2 = np.empty(len(M332)); r2.fill(1)
        r3 = np.empty(len(M333)); r3.fill(1)
        i1 = np.empty(len(M331)); i1.fill(0)
        i2 = np.empty(len(M332)); i2.fill(0)
        i3 = np.empty(len(M333)); i3.fill(0)
        for i in range(1, len(M22)):
            M331[i] -= M331[i - 1] * M22[i - 1] * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            M332[i] -= M332[i - 1] * M22[i - 1] * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            M333[i] -= M333[i - 1] * M22[i - 1] * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            M441[j][i] += M441[j][i - 1] * M22[i - 1] * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            M442[j][i] += M442[j][i - 1] * M22[i - 1] * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            M443[j][i] += M443[j][i - 1] * M22[i - 1] * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            r1[i] -= (M331[i - 1] * r1[i - 1] + M441[j][i - 1] * i1[i - 1]) * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            r2[i] -= (M332[i - 1] * r2[i - 1] + M442[j][i - 1] * i2[i - 1]) * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            r3[i] -= (M333[i - 1] * r3[i - 1] + M443[j][i - 1] * i3[i - 1]) * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            i1[i] -= (M331[i - 1] * i1[i - 1] - M441[j][i - 1] * r1[i - 1]) * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            i2[i] -= (M332[i - 1] * i2[i - 1] - M442[j][i - 1] * r2[i - 1]) * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            i3[i] -= (M333[i - 1] * i3[i - 1] - M443[j][i - 1] * r3[i - 1]) * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
        M331[-1] -= M331[-2] * M22[-1] * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        M332[-1] -= M332[-2] * M22[-1] * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        M333[-1] -= M333[-2] * M22[-1] * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        M441[j][-1] += M441[j][-2] * M22[-1] * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        M442[j][-1] += M442[j][-2] * M22[-1] * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        M443[j][-1] += M443[j][-2] * M22[-1] * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        r1[-1] -= (M331[-2] * r1[-2] + M441[j][-2] * i1[-2]) * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        r2[-1] -= (M332[-2] * r2[-2] + M442[j][-2] * i2[-2]) * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        r3[-1] -= (M333[-2] * r3[-2] + M443[j][-2] * i3[-2]) * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        i1[-1] -= (M331[-2] * i1[-2] - M441[j][-2] * r1[-2]) * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        i2[-1] -= (M332[-2] * i2[-2] - M442[j][-2] * r2[-2]) * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        i3[-1] -= (M333[-2] * i3[-2] - M443[j][-2] * r3[-2]) * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        for i in range(2, len(M33)+1):
            r1[-i] -= (M331[-i + 1] * r1[-i + 1] + M441[j][-i + 1] * i1[-i + 1]) * M22[-i + 1]  / (M331[-i + 1] ** 2 + M441[j][-i + 1] ** 2)
            r2[-i] -= (M332[-i + 1] * r2[-i + 1] + M442[j][-i + 1] * i2[-i + 1]) * M22[-i + 1]  / (M332[-i + 1] ** 2 + M442[j][-i + 1] ** 2)
            r3[-i] -= (M333[-i + 1] * r3[-i + 1] + M443[j][-i + 1] * i3[-i + 1]) * M22[-i + 1]  / (M333[-i + 1] ** 2 + M443[j][-i + 1] ** 2)
            i1[-i] -= (M331[-i + 1] * i1[-i + 1] - M441[j][-i + 1] * r1[-i + 1]) * M22[-i + 1]  / (M331[-i + 1] ** 2 + M441[j][-i + 1] ** 2)
            i2[-i] -= (M332[-i + 1] * i2[-i + 1] - M442[j][-i + 1] * r2[-i + 1]) * M22[-i + 1]  / (M332[-i + 1] ** 2 + M442[j][-i + 1] ** 2)
            i3[-i] -= (M333[-i + 1] * i3[-i + 1] - M443[j][-i + 1] * r3[-i + 1]) * M22[-i + 1]  / (M333[-i + 1] ** 2 + M443[j][-i + 1] ** 2)

        V1[j] = (r1 * M331 + i1 * M441[j]) / (M331 ** 2 + M441[j] ** 2)
        V2[j] = (r2 * M332 + i2 * M442[j]) / (M332 ** 2 + M442[j] ** 2)
        V3[j] = (r3 * M333 + i3 * M443[j]) / (M333 ** 2 + M443[j] ** 2)

    Weigth = np.exp(alfa*(-1) * np.cos(np.pi / 2 * ((numb2 - S - 1) / S) ** 2) ** 2)
    Weigth = Weigth / np.sum(Weigth) /2/np.pi

    tmp1 = (Weigth * V1).sum(axis=1)
    tmp2 = (Weigth * V2).sum(axis=1)
    tmp3 = (Weigth * V3).sum(axis=1)

    return (tmp1, tmp2, tmp3)


@njit(cache=True)
def _relax_MS_groups_c(S, x, Sig, eps, Hv, W, R, alfa):
    """Complex counterpart of ``_relax_MS_groups``: same elimination, but the
    solution of the (complex tridiagonal) stochastic system is returned in
    full, Lambda_g = V_g + 1j*DISPERSION_SIGN*D_g per Delta-m group, instead
    of its real (absorptive) part only. The (r, i)/(M33, M44) split arithmetic
    already carries the imaginary part; it is read out here at no extra cost
    as D = (r*M44 - i*M33)/(M33**2 + M44**2), the Kramers--Kronig partner in
    the ``Voight_c`` sign convention."""
    numb = np.linspace(0, 2 * S, int(2 * S + 1))
    numb2 = np.linspace(1, 2 * S + 1, int(2 * S + 1))

    M111 = (-1)*R*(S*(S+1)-((numb[1:S+1]+1)-S-1)*((numb[1:S+1]+1)-S-2))*np.exp(alfa*(np.cos(np.pi/2*(((numb[1:S+1]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[1:S+1]+1)-1)-S-1)/S)**2)**2))
    M112 = (-1)*R*(S*(S+1)-((numb[S+1:]+1)-S-1)*((numb[S+1:]+1)-S-2))
    M11 = np.concatenate((M111,M112))
    M22 = M11[::-1]

    M33 = W - M11[:-1] - M22[1:]
    M33 = insert1D(M33, 0, (W - M22[0]))
    M33 = insert1D(M33, len(numb) - 1, (W - M11[-1]))

    M441 = (D2A(x, len(numb)).transpose(1,0) - Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig - eps)
    M442 = (D2A(x, len(numb)).transpose(1,0) - LINE_RATIO_25 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)
    M443 = (D2A(x, len(numb)).transpose(1,0) - LINE_RATIO_34 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)

    V1 = np.array([[float(0)] * len(M33)] * len(x))
    V2 = np.array([[float(0)] * len(M33)] * len(x))
    V3 = np.array([[float(0)] * len(M33)] * len(x))
    V1d = np.array([[float(0)] * len(M33)] * len(x))
    V2d = np.array([[float(0)] * len(M33)] * len(x))
    V3d = np.array([[float(0)] * len(M33)] * len(x))

    for j in range(0, len(x)):
        M331 = np.copy(M33)
        M332 = np.copy(M33)
        M333 = np.copy(M33)
        r1 = np.empty(len(M331)); r1.fill(1)
        r2 = np.empty(len(M332)); r2.fill(1)
        r3 = np.empty(len(M333)); r3.fill(1)
        i1 = np.empty(len(M331)); i1.fill(0)
        i2 = np.empty(len(M332)); i2.fill(0)
        i3 = np.empty(len(M333)); i3.fill(0)
        for i in range(1, len(M22)):
            M331[i] -= M331[i - 1] * M22[i - 1] * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            M332[i] -= M332[i - 1] * M22[i - 1] * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            M333[i] -= M333[i - 1] * M22[i - 1] * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            M441[j][i] += M441[j][i - 1] * M22[i - 1] * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            M442[j][i] += M442[j][i - 1] * M22[i - 1] * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            M443[j][i] += M443[j][i - 1] * M22[i - 1] * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            r1[i] -= (M331[i - 1] * r1[i - 1] + M441[j][i - 1] * i1[i - 1]) * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            r2[i] -= (M332[i - 1] * r2[i - 1] + M442[j][i - 1] * i2[i - 1]) * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            r3[i] -= (M333[i - 1] * r3[i - 1] + M443[j][i - 1] * i3[i - 1]) * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
            i1[i] -= (M331[i - 1] * i1[i - 1] - M441[j][i - 1] * r1[i - 1]) * M11[i - 1] / (M331[i - 1] ** 2 + M441[j][i - 1] ** 2)
            i2[i] -= (M332[i - 1] * i2[i - 1] - M442[j][i - 1] * r2[i - 1]) * M11[i - 1] / (M332[i - 1] ** 2 + M442[j][i - 1] ** 2)
            i3[i] -= (M333[i - 1] * i3[i - 1] - M443[j][i - 1] * r3[i - 1]) * M11[i - 1] / (M333[i - 1] ** 2 + M443[j][i - 1] ** 2)
        M331[-1] -= M331[-2] * M22[-1] * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        M332[-1] -= M332[-2] * M22[-1] * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        M333[-1] -= M333[-2] * M22[-1] * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        M441[j][-1] += M441[j][-2] * M22[-1] * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        M442[j][-1] += M442[j][-2] * M22[-1] * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        M443[j][-1] += M443[j][-2] * M22[-1] * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        r1[-1] -= (M331[-2] * r1[-2] + M441[j][-2] * i1[-2]) * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        r2[-1] -= (M332[-2] * r2[-2] + M442[j][-2] * i2[-2]) * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        r3[-1] -= (M333[-2] * r3[-2] + M443[j][-2] * i3[-2]) * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        i1[-1] -= (M331[-2] * i1[-2] - M441[j][-2] * r1[-2]) * M11[-1] / (M331[-2] ** 2 + M441[j][-2] ** 2)
        i2[-1] -= (M332[-2] * i2[-2] - M442[j][-2] * r2[-2]) * M11[-1] / (M332[-2] ** 2 + M442[j][-2] ** 2)
        i3[-1] -= (M333[-2] * i3[-2] - M443[j][-2] * r3[-2]) * M11[-1] / (M333[-2] ** 2 + M443[j][-2] ** 2)
        for i in range(2, len(M33)+1):
            r1[-i] -= (M331[-i + 1] * r1[-i + 1] + M441[j][-i + 1] * i1[-i + 1]) * M22[-i + 1]  / (M331[-i + 1] ** 2 + M441[j][-i + 1] ** 2)
            r2[-i] -= (M332[-i + 1] * r2[-i + 1] + M442[j][-i + 1] * i2[-i + 1]) * M22[-i + 1]  / (M332[-i + 1] ** 2 + M442[j][-i + 1] ** 2)
            r3[-i] -= (M333[-i + 1] * r3[-i + 1] + M443[j][-i + 1] * i3[-i + 1]) * M22[-i + 1]  / (M333[-i + 1] ** 2 + M443[j][-i + 1] ** 2)
            i1[-i] -= (M331[-i + 1] * i1[-i + 1] - M441[j][-i + 1] * r1[-i + 1]) * M22[-i + 1]  / (M331[-i + 1] ** 2 + M441[j][-i + 1] ** 2)
            i2[-i] -= (M332[-i + 1] * i2[-i + 1] - M442[j][-i + 1] * r2[-i + 1]) * M22[-i + 1]  / (M332[-i + 1] ** 2 + M442[j][-i + 1] ** 2)
            i3[-i] -= (M333[-i + 1] * i3[-i + 1] - M443[j][-i + 1] * r3[-i + 1]) * M22[-i + 1]  / (M333[-i + 1] ** 2 + M443[j][-i + 1] ** 2)

        V1[j] = (r1 * M331 + i1 * M441[j]) / (M331 ** 2 + M441[j] ** 2)
        V2[j] = (r2 * M332 + i2 * M442[j]) / (M332 ** 2 + M442[j] ** 2)
        V3[j] = (r3 * M333 + i3 * M443[j]) / (M333 ** 2 + M443[j] ** 2)
        V1d[j] = (r1 * M441[j] - i1 * M331) / (M331 ** 2 + M441[j] ** 2)
        V2d[j] = (r2 * M442[j] - i2 * M332) / (M332 ** 2 + M442[j] ** 2)
        V3d[j] = (r3 * M443[j] - i3 * M333) / (M333 ** 2 + M443[j] ** 2)

    Weigth = np.exp(alfa*(-1) * np.cos(np.pi / 2 * ((numb2 - S - 1) / S) ** 2) ** 2)
    Weigth = Weigth / np.sum(Weigth) /2/np.pi

    tmp1 = (Weigth * V1).sum(axis=1) + 1j * DISPERSION_SIGN * (Weigth * V1d).sum(axis=1)
    tmp2 = (Weigth * V2).sum(axis=1) + 1j * DISPERSION_SIGN * (Weigth * V2d).sum(axis=1)
    tmp3 = (Weigth * V3).sum(axis=1) + 1j * DISPERSION_SIGN * (Weigth * V3d).sum(axis=1)

    return (tmp1, tmp2, tmp3)


@njit(cache=True)
def relax_MS(S, x, I, Sig, eps, Hv, W, Ah, R, alfa):
    """Many-state relaxation sextet (thin, scalar): weight the three Delta m
    groups by the 3:2:1 asymmetry Ah and sum into one absorber."""
    tmp1, tmp2, tmp3 = _relax_MS_groups(S, x, Sig, eps, Hv, W, R, alfa)
    I1 = I * 3 * (1 - Ah) / (8 - 4 * Ah)
    I2 = I * 2 * Ah / (8 - 4 * Ah)
    I3 = I * 1 * (1 - Ah) / (8 - 4 * Ah)
    return (2 * tmp1 * I1 + 2 * tmp2 * I2 + 2 * tmp3 * I3)


# @njit # (1) https://doi.org/10.1016/j.cam.2008.04.040
# def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R):
#     WG = 0.1
#     delt = WG/2/np.sqrt(2*np.log(2)) + 0.01*(WG==0)
#     gamm = L/2
#     alf = (ggr * m0 - gex * m1) * (H2 - H1) / 2 + (Q2 - Q1) / 2 * (3 * m1 ** 2 - 15 / 4) + (Sig2 - Sig1) / 2
#     V0 = (Sig1 + Sig2) / 2 + (Q2 + Q1) / 2 * (3 * m1 ** 2 - 15 / 4) + (ggr * m0 - gex * m1) * (H2 + H1) / 2
#     A0 = -1j * (-V0 + alf) - gamm - W
#     lam1 = (-1j * 2*(- V0) - 2*gamm - W*(R+1) + np.sqrt(W**2*(R+1)**2 - 4*alf**2 - 1j*4*alf*W*(R-1))) / 2
#     lam2 = (-1j * 2*(- V0) - 2*gamm - W*(R+1) - np.sqrt(W**2*(R+1)**2 - 4*alf**2 - 1j*4*alf*W*(R-1))) / 2
#     T = np.linspace(0, 20, 512)
#     n = (T[1]-T[0])
#     Line = np.array([float(0)] * len(V))
#     for i in range(0, len(V)):
#         L = (A0-lam1-R*W)*(A0-lam2+W)*np.exp(lam1*T) - (A0-lam2-R*W)*(A0-lam1+W)*np.exp(lam2*T)
#         Line[i] = (np.real(L / (lam2 - lam1) / W / (R + 1) * n * np.exp(-delt ** 2 * T ** 2 / 2 - 1j * V[i] * T))).sum()
#
#     return(Line/np.pi)


@njit(cache=True)
def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R): # https://doi.org/10.1103/PhysRev.165.446
    alf = (ggr * m0 - gex * m1) * (H2-H1)/2 + (Q2-Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (Sig2-Sig1)/2
    a = V - (Sig1+Sig2)/2 - (Q2+Q1)/2 * (3 * m1 ** 2 - 15 / 4) - (ggr * m0 - gex * m1) * (H2+H1)/2
    Line = np.array([float(0)]*len(V))
    for i in range(0, len(V)):
        a1 = 1j*(a[i]+alf) + L/2 + W
        a3 = -R*W
        a2 = -W
        a4 = 1j*(a[i]-alf) + L/2 + R*W
        delimeter = 1/(a1*a4-a2*a3)
        b1 = np.real(a4 *  delimeter)
        b2 = np.real(-a2 * delimeter)
        b3 = np.real(-a3 * delimeter)
        b4 = np.real(a1 *  delimeter)
        Line[i] = (b1+b2)*R/(R+1) + (b3+b4)/(R+1)
    return(Line/np.pi)


@njit(cache=True)
def Blume_c(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R):
    """Complex Blume two-state line shape: Re = ``Blume(...)`` (bit-identical
    arithmetic), Im = DISPERSION_SIGN * its Kramers--Kronig partner. Obtained
    by simply NOT taking np.real() of the resolvent elements; the natural
    pivot 1j*(a +/- alf) + L/2 + ... carries the opposite dispersion sign to
    ``Voight_c``, hence the conjugation for DISPERSION_SIGN = +1."""
    alf = (ggr * m0 - gex * m1) * (H2-H1)/2 + (Q2-Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (Sig2-Sig1)/2
    a = V - (Sig1+Sig2)/2 - (Q2+Q1)/2 * (3 * m1 ** 2 - 15 / 4) - (ggr * m0 - gex * m1) * (H2+H1)/2
    Line = np.zeros(len(V), dtype=np.complex128)
    for i in range(0, len(V)):
        a1 = 1j*(a[i]+alf) + L/2 + W
        a3 = -R*W
        a2 = -W
        a4 = 1j*(a[i]-alf) + L/2 + R*W
        delimeter = 1/(a1*a4-a2*a3)
        b1 = a4 *  delimeter
        b2 = -a2 * delimeter
        b3 = -a3 * delimeter
        b4 = a1 *  delimeter
        Line[i] = (b1+b2)*R/(R+1) + (b3+b4)/(R+1)
    if DISPERSION_SIGN > 0:
        return(np.conj(Line)/np.pi)
    return(Line/np.pi)


# @njit(parallel=True)
# def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, GG, W, m0, m1, V, R):
#     alf = (ggr * m0 - gex * m1) * (H2-H1)/2 + (Q2-Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (Sig2-Sig1)/2
#     a = V - np.array([(Sig1+Sig2)/2 + (Q2+Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (ggr * m0 - gex * m1) * (H2+H1)/2]*len(V))
#     W = np.array([[[W, -R*W], [-W, R*W]]] * len(V))
#     F = np.array([[[1, 0], [0, -1]]] * len(V))
#     p_tmp = 1j*a + L/2
#     p = np.array([[[complex(0)]*len(V),[complex(0)]*len(V)],[[complex(0)]*len(V),[complex(0)]*len(V)]])
#     for i in range(0, len(V)):
#         p[0][0][i] = p_tmp[i]
#         p[1][1][i] = p_tmp[i]
#     p = np.swapaxes(p, 0, 2)
#     M = p+W+F*1j*alf
#     for i in range(0, len(V)):
#         M[i][0][0], M[i][0][1], M[i][1][0], M[i][1][1] = np.linalg.inv(np.array([[M[i][0][0], M[i][0][1]], [M[i][1][0], M[i][1][1]]])).flatten()
#     Line = np.real((M.sum(axis=1) * np.array([R, 1])/(R+1)).sum(axis=1))/np.pi
#     Line = Line[:-1]*abs(V[1:]-V[:-1])
#     GS = (GG * abs(V[1:] - V[:-1])).sum(axis=0)
#     GG = GG * Line / GS
#     Res = GG.sum(axis=1)
#     return(Res)


@njit(cache=True)
def K_cei(M): # Selescu, R. (2021) for <0.98 and direct approximation above
    M = np.abs(M)
    if M <= 0.9999978056985459:
        m = np.sqrt(1-M)
        K = np.pi*np.sqrt(2) / np.sqrt((1+m)*np.sqrt(m)) * (1 - 2**(1/4)/4 * (1+np.sqrt(m))/((1+m)*np.sqrt(m))**(1/4))
    elif M < 1:
        K = 1 / 2 * np.log((1 + M) / (1 - M))
    else:
        K = 99999
    return K


@njit(cache=True)
def SN(x, m): #  doi.org/10.1063/1.527661
    m = np.abs(m)
    if m <= 0.981:
        mu = np.pi/4/(K_cei(m) + 10**-12 * (m==0))
        t = 1/mu * np.tan(mu*x)
        z = t**2

        a1 = -1/6 * (1 + m + 2*mu**2)
        a2 = 1/120 * (1+m)**2 + mu**2/6 * (1+m) + m/10 + mu**4/5

        ### n = 3
        p1 = (mu ** 4 + a1*mu**2 + a1**2 - a2)/(mu**2 + a1)
        p2 = mu**4
        q1 = (mu ** 4 - a2)/(mu**2+a1) # error in article
        q2 = mu ** 2 * (mu**4-a2)/(mu**2+a1)
        q3 = mu**6
        sn = (1 + p1 * z + p2 * z**2) \
             / (1 + q1 * z + q2 * z ** 2 + q3 * z ** 3) * t
    elif m < 1:
        sn = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)  # np.tanh(x)
    else:
        sn = np.array([float(1)] * len(x))
    return sn

@njit(cache=True)
def ASM(T, sigm, eps_m, eps_lat, His, Han, WL, WG, m, A, Num, I13, E):
    K = K_cei(m)
    Num = int(Num / 6) * 6 + 1
    X = np.linspace(0, K, Num)
    if m >= 0:
        co = SN(X, m) ** 2
        # co = 0.7*SN(X, m) ** 2 + 0.3*SN(X, m) ** 4
    else:
        co = 1 - SN(X, m) ** 2

    eps = eps_m + eps_lat * (3 * co - 1) / 2
    H = His + Han * (3 * co - 1) / 2
    a1 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0)) * (co + 1 / 8 * (1 - co)) * (1 - co)
    a2 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0)) * (co - 1 / 8 * (1 - co)) * (1 - co)

    H = H / E0_J * c

    v1 = sigm + eps + a1 + mun * (3 * gex - ggr) / 2 * H
    v6 = sigm + eps - a1 - mun * (3 * gex - ggr) / 2 * H
    v2 = sigm - eps[0::2] - a2[0::2] + mun * (gex - ggr) / 2 * H[0::2]
    v5 = sigm - eps[0::2] + a2[0::2] - mun * (gex - ggr) / 2 * H[0::2]
    v3 = sigm - eps[0::3] + a2[0::3] - mun * (gex + ggr) / 2 * H[0::3]
    v4 = sigm - eps[0::3] - a2[0::3] + mun * (gex + ggr) / 2 * H[0::3]

    # v = [v1, v2, v3, v4, v5, v6]

    I = T
    I1 = I * (4 * I13 / (I13 + 1)) * (1 - A) / (8 - 4 * A)
    I2 = I * 2 * A / (8 - 4 * A)
    I3 = I * (4 / (I13 + 1)) * (1 - A) / (8 - 4 * A)
    # I = [I1, I2, I3, I3, I2, I1]

    Line = np.array([float(0)] * len(E))
    # for i in range(0, 6):
    #     for j in range(0, len(X)):
    #         Line += Voight(WL, WG, E - v[i][j]) * I[i] / len(X)
    for j in range(0, Num):
        Line += Voight(WL, WG, E - v1[j]) * I1 / Num
        Line += Voight(WL, WG, E - v6[j]) * I1 / Num
    for j in range(0, int(Num/2)+1):
        Line += Voight(WL, WG, E - v2[j]) * I2 / int(Num/2+1)
        Line += Voight(WL, WG, E - v5[j]) * I2 / int(Num/2+1)
    for j in range(0, int(Num/3)+1):
        Line += Voight(WL, WG, E - v3[j]) * I3 / int(Num / 3 + 1)
        Line += Voight(WL, WG, E - v4[j]) * I3 / int(Num / 3 + 1)


    return Line


# ===========================================================================
# Polarized ("thick") transmission: 2x2 cross-section matrix machinery.
#
# For a polarized (synchrotron) source the scalar transmission integral breaks
# down once the absorber cannot attenuate every polarization equally: a resolved
# circular/linear line would absorb more than 50% of a linearly polarized beam,
# which is unphysical. The correct object is a 2x2 cross-section matrix
# sigma(E) acting in the polarization plane; the transmission for the incident
# polarization e1 = h (the radiation magnetic field) is [expm(-sigma(E))]_11.
#
# Each "(thick)" component adds its own 2x2 matrix to a shared accumulator and
# the matrices are summed BEFORE the single matrix exponential (homogeneous
# sample) - that is what lets the polarization state evolve coherently through
# the mixture. Scalar (non-thick) components are isotropic in the polarization
# plane (proportional to the identity) and keep multiplying the running scalar
# transmission exactly as before; the two factor cleanly because the identity
# commutes with everything.
#
# Normalisation: every matrix below satisfies the powder average <Mhat> = I, so
# a thick component reduces EXACTLY to its scalar counterpart when its
# orientation is averaged over the sphere, and to the correct polarized line
# intensities in the thin limit. Geometry (lab frame): beam k = z, radiation
# magnetic field h = e1 = x, second polarization e2 = y. (theta_h, phi_h) is the
# orientation of the component symmetry axis (V_zz for a doublet, B_hf for a
# sextet) in that frame - theta_h from the beam k, phi_h from h - so the angle
# beta between h and the axis obeys cos(beta) = sin(theta_h) cos(phi_h).
# ===========================================================================

# --- Incident-beam polarization (the SINGLE manual knob) -------------------
# The transmission is read out from the polarization density matrix
# rho = diag((1+p)/2, (1-p)/2) of the incident beam, p being the degree of
# LINEAR polarization along h:  C_a = tr[expm(-Sigma) rho]
#                                   = (1+p)/2 [expm(-Sigma)]_11 + (1-p)/2 [expm(-Sigma)]_22.
#
#   p = SMS_POL_DEFAULT -> a realistic synchrotron (SMS) beam (DEFAULT).
#   p = 1.0  -> fully polarized: reads the (1,1) element only, i.e. the original
#               SMS behaviour.
#   p = 0.0  -> unpolarized: half the trace, the conventional radioactive-source
#               (CMS) readout.
#
# The degree p is the ``pol`` argument of ``TI`` (default SMS_POL_DEFAULT), which forwards
# it to ``TImod`` as ``sms_pol``. It is editable at runtime from the GUI
# (Supp -> "Set polarization"). It applies to SMS spectra only; CMS spectra
# (Met == 1) are always read out unpolarized (rho = I/2, a fixed 1:1 mixture)
# regardless of this value, because an unpolarized source defines no direction
# in the polarization plane.

_I2 = np.eye(2, dtype=complex)
_J2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)  # 1j*_J2 is Hermitian


def _axis_xyz(theta_deg, phi_deg):
    """Unit vector of a symmetry axis at (theta, phi) [deg] in the lab frame."""
    th = theta_deg / 180.0 * np.pi
    ph = phi_deg / 180.0 * np.pi
    s = np.sin(th)
    return s * np.cos(ph), s * np.sin(ph), np.cos(th)


def _mm_perp(mx, my):
    """2x2 projector m_perp m_perp^T onto the polarization plane (e1=h, e2)."""
    return np.array([[mx * mx, mx * my], [mx * my, my * my]], dtype=complex)


def _mhat_dm0(mx, my):
    """Normalised matrix for a Delta m = 0 (pi) line; powder average = I."""
    return 3.0 * _mm_perp(mx, my)


def _mhat_dm1(mx, my, mz, sign):
    """Normalised matrix for a Delta m = +/-1 (sigma) line; powder average = I.

    ``sign`` = +1 for sigma+ (Delta m = +1), -1 for sigma- (Delta m = -1). The
    +/- i*mz term is the magneto-optical (Faraday) part that distinguishes the
    two circular eigenpolarizations; it vanishes for the axis in the
    polarization plane (mz = 0) and is what makes a single sigma line saturate
    at 50% transmission for thick absorbers.
    """
    return 1.5 * ((_I2 - _mm_perp(mx, my)) + sign * 1j * mz * _J2)


def _mhat_dm1_sym(mx, my):
    """Faraday-averaged Delta m = +/-1 matrix (sigma+ and sigma- blended).

    Used where a model lumps the +/-1 lines together (the relaxation groups):
    the +/- i*mz terms cancel, leaving the symmetric (linear-eigenpolarization)
    part. Powder average = I; still saturates at 50% for an axis lying in the
    polarization plane.
    """
    return 1.5 * (_I2 - _mm_perp(mx, my))


def _texture_blend(M, A):
    """Uniaxial (fiber) texture average of a thick building-block matrix.

    A textured powder whose symmetry axes follow an axially symmetric orientation
    distribution about the lab-frame axis (theta_h, phi_h) averages, *before* the
    matrix exponential, to

        <M> = (1 - A) * I2 + A * M(axis),

    where the order parameter ``A = <P2(cos psi)>`` (psi = angle to the texture
    axis) interpolates continuously between a random powder (A = 0 -> I2, i.e. the
    scalar model recovered exactly at any thickness) and a perfectly aligned
    single crystal (A = 1 -> M, the current single-orientation thick model). Every
    building block powder-averages to I2, so this is exact for the quadratic
    (symmetric) parts it is applied to (``_mhat_dm0``, ``_mhat_dm1_sym``, doublet
    Mh_A/Mh_B). The magneto-optical (Faraday) term of a resolved sigma+- line is
    linear (not quadratic) in the axis and carries the SEPARATE polar-order
    parameter S1 instead of A -- it is added on top of this symmetric blend via
    ``_texture_s1`` (see there), not folded into it.
    """
    return (1.0 - A) * _I2 + A * M


def _texture_s1(A, Am):
    """Polar-order parameter S1 that scales the Faraday (sigma+-) term of a texture.

    While A = <P2(cos chi)> measures the *alignment* of the axes about the texture
    axis (and scales the quadratic parts, via ``_texture_blend``), the Faraday term
    of a resolved sigma+- line is linear in the axis and averages instead to the
    first moment S1 = <cos chi>, the net *polar* (magnetic) order along the texture
    axis. The two are not independent: Cauchy-Schwarz bounds S1**2 <= (1 + 2A)/3.
    We therefore parametrise S1 by ``Am`` in [-1, 1], the fraction of that bound,

        S1 = Am * sqrt((1 + 2A) / 3),

    so the fit can never leave the physical region whatever A is. S1 is non-zero
    only for a magnetised texture (domains with +B_hf and -B_hf unequally
    populated), so ``Am`` defaults to 0 -- an unmagnetised (even if perfectly
    aligned) sample keeps the Faraday-averaged sigma matrix. ``Am = A = 1`` recovers
    the fully-magnetised single-crystal sigma+- lines exactly. In the
    absorption-only limit only |S1| was observable from one homogeneous layer
    (its sign merely conjugated the Hermitian cross-section, which the intensity
    readout cannot see). With the dispersive (Kramers--Kronig) completion the
    cross-section is complex non-Hermitian and S1 -> -S1 is no longer a pure
    conjugation, so the SIGN of ``Am`` becomes observable for a POLARIZED (SMS)
    beam -- while an unpolarized (CMS, half-trace) readout stays invariant and
    still cannot see it. The sign of ``Am`` also matters for the RELATIVE sign
    between mixed/stacked Faraday-active components.
    """
    return Am * np.sqrt(max(0.0, (1.0 + 2.0 * A) / 3.0))


def _expm_neg(Sig):
    """Matrix exponential expm(-Sig) for a stack of 2x2 matrices.

    ``Sig`` has shape (N, 2, 2), complex; Hermitian OR general non-Hermitian
    (the Cayley--Hamilton closed form holds for any 2x2 matrix, as needed for
    the dispersive completion). Uses the closed-form 2x2 matrix exponential
    vectorised over the energy axis N; returns shape (N, 2, 2). Equals the
    identity wherever Sig is zero.
    """
    M = -Sig
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    cc = M[:, 1, 0]
    dd = M[:, 1, 1]
    s = 0.5 * (a + dd)
    det = a * dd - b * cc
    disc = np.sqrt(s * s - det + 0j)
    tiny = np.abs(disc) < 1e-12
    disc_safe = np.where(tiny, 1.0 + 0j, disc)
    sinhc = np.where(tiny, 1.0 + 0j, np.sinh(disc_safe) / disc_safe)
    es = np.exp(s)
    ch = np.cosh(disc)
    out = np.empty_like(M)
    out[:, 0, 0] = es * (ch + sinhc * (a - s))
    out[:, 0, 1] = es * (sinhc * b)
    out[:, 1, 0] = es * (sinhc * cc)
    out[:, 1, 1] = es * (ch + sinhc * (dd - s))
    return out


def _expm_neg11(Sig):
    """Real part of [expm(-Sig)]_00 for a stack of 2x2 matrices (shape (N,)).

    Equals 1 wherever Sig is zero, so a spectrum with no thick component is
    multiplied by 1 (a no-op).
    """
    return np.real(_expm_neg(Sig)[:, 0, 0])


@njit(cache=True)
def Ham_mono_thick(Q, Hhf, etto, phi, tet, phir, tetr, alfak):
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
    g0, g1, g2, S = _ham_mono_core(Q, Hhf, etto, phi, tet)

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


@njit(cache=True)
def Ham_mono_thick_CMS(Q, Hhf, etto, phi, tet, phir, tetr):
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
    g0, g1, g2, S = _ham_mono_core(Q, Hhf, etto, phi, tet)

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


# ---------------------------------------------------------------------------
# 'Hamiltonian': the MOSAIC TEXTURED full-Hamiltonian component.
#
# It supersedes both former Hamiltonian models: it keeps all 12 parameters of
# the single-crystal 'Hamilton_mc' (including tetr, phir, alfak) and adds three
# order parameters (A, Am, Ah) describing an axially symmetric orientation
# distribution (ODF) of the crystal about the lab reference axis. Exact limits:
#
#   (A, Am, Ah) = (1, 1, 1) -> the single crystal ('Hamilton_mc'), exactly;
#   Ah = 0                   -> fiber texture about the reference axis;
#   (0, 0, 0)                 -> random powder ('Hamilton_pc'), exactly;
#   Q = 0 and Ah = 0         -> the textured Sextet with its axis along the
#                                reference axis and A_eff = A*P2(cos th_Bh).
#
# Order parameters (the sextet family's A / Am carry over unchanged in meaning):
#   A   in [-1/2, 1] : S  = <P2(cos chi)> of the twist-free wobble of the crystal
#                      about the reference orientation; chi = tilt of the crystal
#                      direction set by (tetr, phir) away from the lab reference
#                      axis. A = 1 no wobble, A = 0 isotropic tilt, A = -1/2 the
#                      tilt confined to 90 deg.
#   Am in [-1, 1]   : polar order of that same wobble, S1 = <cos chi> =
#                      Am*sqrt((1+2A)/3) (the Cauchy--Schwarz bound, as in
#                      _texture_s1) -- does the mosaic distinguish +axis from
#                      -axis. Only a magnetised/polar mosaic has Am != 0 and it
#                      acts only through the Faraday (magneto-optical) term.
#   Ah in [0, 1]    : order of the crystal AZIMUTH about the reference axis,
#                      <cos(m*alpha)> = Ah^|m| (exactly a wrapped-Cauchy
#                      azimuth). Ah = 1 alfak sharply defined (single crystal),
#                      Ah = 0 crystallites uniformly spun about the axis.
#
# THE ODF. R = W(beta, chi) * R0 * Z(alpha) with R0 the reference orientation,
# Z(alpha) a rotation of the crystal about the reference axis and
# W = R_f(beta) R_perp(chi) R_f(-beta) the twist-free wobble about the lab
# reference axis f (beta uniform). "Spread the axis but keep alfak" is ambiguous
# as such (transporting an azimuth along a tilt has holonomy); the twist-free
# wobble is the convention-free realisation. The ODF-averaged 2x2 block of each
# transition depends on the distribution ONLY through <cos chi>, <cos^2 chi>,
# <cos alpha> and <cos 2 alpha> -- exactly the three order parameters -- because
# the average of a rank-l tensor about an axially symmetric rotation
# distribution scales its m-components by <d^(l)_mm(chi)>, and
# d^(1)_00 = cos chi, d^(1)_11 = (1+cos chi)/2, d^(2)_00 = P2(cos chi),
# d^(2)_11 = (1+cos chi)(2cos chi - 1)/2, d^(2)_22 = ((1+cos chi)/2)^2.
# So ONE diagonalisation still suffices: the average is closed in the moments.
#
# The lab reference axis f is the axis that (tetr, phir) points at: the radiation
# field h = e1 for an SMS source, the BEAM k for a CMS one (which is exactly how
# (tetr, phir) itself is already read in each case, see Ham_mono_thick /
# Ham_mono_thick_CMS). A CMS mosaic is therefore a foil-normal (beam-axis)
# texture, which keeps the CMS spectrum independent of alfak -- for an
# unpolarized source no transverse direction is observable.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _R0_rows(tetr, phir, alfak, cms):
    """Reference lab axes (e1, e2, k) as ROWS, in EFG (PAS) coordinates.

    So ``R0 @ v`` gives the lab components of an EFG-frame vector v, and
    ``R0[j]`` is lab axis j expressed in EFG coordinates.

    ``cms = False`` (SMS): (tetr, phir) is the radiation magnetic field h = e1
    and alfak rotates the beam k about it, e2 = k x h -- the geometry of
    ``Ham_mono_thick``. ``cms = True``: (tetr, phir) is the BEAM k and
    (e1, e2) = (theta_hat, phi_hat) -- the geometry of ``Ham_mono_thick_CMS``
    (alfak unused there: the half-trace readout is invariant under a spin of
    that basis).
    """
    t = tetr / 180 * np.pi
    ph = phir / 180 * np.pi
    ak = alfak / 180 * np.pi
    nr = np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)])
    that = np.array([np.cos(t) * np.cos(ph), np.cos(t) * np.sin(ph), -np.sin(t)])
    phat = np.array([-np.sin(ph), np.cos(ph), 0.0])
    R0 = np.empty((3, 3))
    if cms:
        R0[0] = that                                   # e1 = theta_hat
        R0[1] = phat                                   # e2 = phi_hat
        R0[2] = nr                                     # k  = (tetr, phir)
    else:
        R0[0] = nr                                     # e1 = h = (tetr, phir)
        R0[1] = np.sin(ak) * that - np.cos(ak) * phat  # e2 = k x h
        R0[2] = np.cos(ak) * that + np.sin(ak) * phat  # k
    return R0


@njit(cache=True)
def _rank2_scale(X, axis, lam0, lam1, lam2):
    """Uniaxial rotation average of a symmetric traceless 3x3 tensor.

    Splits ``X`` into its m = 0, +-1, +-2 parts about ``axis`` and scales them by
    (lam0, lam1, lam2) -- which is what <Q X Q^T> does for any rotation
    distribution Q that is axially symmetric about that axis (lam_m =
    <d^(2)_mm>). The transverse frame used to do the split is arbitrary because
    the scaling is diagonal in |m|.
    """
    z = axis / np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if np.abs(z[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    x = np.array([t[1] * z[2] - t[2] * z[1], t[2] * z[0] - t[0] * z[2], t[0] * z[1] - t[1] * z[0]])
    x = x / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    y = np.array([z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]])
    L = np.empty((3, 3))
    L[:, 0] = x
    L[:, 1] = y
    L[:, 2] = z
    Lt = np.ascontiguousarray(L.T)
    Xl = Lt @ np.ascontiguousarray(X) @ L
    a = Xl[2, 2]                        # m = 0  amplitude
    b = 0.5 * (Xl[0, 0] - Xl[1, 1])     # m = +-2 (real part)
    Xp = np.zeros((3, 3))
    Xp[0, 0] = -0.5 * lam0 * a + lam2 * b
    Xp[1, 1] = -0.5 * lam0 * a - lam2 * b
    Xp[2, 2] = lam0 * a
    Xp[0, 1] = lam2 * Xl[0, 1]          # m = +-2 (imaginary part)
    Xp[1, 0] = Xp[0, 1]
    Xp[0, 2] = lam1 * Xl[0, 2]          # m = +-1
    Xp[2, 0] = Xp[0, 2]
    Xp[1, 2] = lam1 * Xl[1, 2]
    Xp[2, 1] = Xp[1, 2]
    return L @ Xp @ Lt


@njit(cache=True)
def Ham_mosaic(Q, Hhf, etto, phi, tet, tetr, phir, alfak, Atex, Am, Ah, cms):
    """Per-transition 2x2 cross-section matrices of the mosaic textured Hamiltonian.

    Returns ``(P, S)`` exactly like ``Ham_mono_thick``: ``P[k]`` the ODF-averaged
    pi*<A A^dagger> block of transition k in the (e1, e2) basis and ``S[k]`` the
    line positions (mm/s). ONE diagonalisation -- the orientation average is
    closed in the three order parameters (see the block comment above).
    ``(Atex, Am, Ah) = (1, 1, 1)`` reproduces ``Ham_mono_thick`` (SMS) or
    ``Ham_mono_thick_CMS`` (CMS) to round-off; ``(0, 0, 0)`` the random powder
    ``Ham_poly``; ``Ah = 0`` a fiber texture about the reference axis.
    """
    g0, g1, g2, S = _ham_mono_core(Q, Hhf, etto, phi, tet)
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
        # P = pi*A A^dagger = 2*pi * d d^dagger (see _ham_mono_core for the F_q).
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


@njit(cache=True)
def relax_MS_thick(S, x, I, Sig, eps, Hv, W, R, alfa):
    """Many-state relaxation sextet (polarized): the three Delta m groups,
    each scaled by the 3:2:1 isotropic (Ah = 1/2) weights and the thickness I and
    returned SEPARATELY so ``TImod`` can multiply each by its own 2x2 matrix.
    Summing the three returned arrays reproduces ``relax_MS`` at Ah = 1/2. The
    relaxation/diagonalisation lives in the shared core ``_relax_MS_groups``.
    """
    tmp1, tmp2, tmp3 = _relax_MS_groups(S, x, Sig, eps, Hv, W, R, alfa)
    Ah = 0.5
    I1 = I * 3 * (1 - Ah) / (8 - 4 * Ah)
    I2 = I * 2 * Ah / (8 - 4 * Ah)
    I3 = I * 1 * (1 - Ah) / (8 - 4 * Ah)
    return (2 * tmp1 * I1, 2 * tmp2 * I2, 2 * tmp3 * I3)

@njit(cache=True)
def relax_MS_thick_c(S, x, I, Sig, eps, Hv, W, R, alfa):
    """Complex counterpart of ``relax_MS_thick``: same three Delta-m group
    profiles at the isotropic weights, but each is the COMPLEX line shape
    Lambda_g = V_g + 1j*DISPERSION_SIGN*D_g from ``_relax_MS_groups_c``.
    Re(returned) is bit-identical to ``relax_MS_thick``."""
    tmp1, tmp2, tmp3 = _relax_MS_groups_c(S, x, Sig, eps, Hv, W, R, alfa)
    Ah = 0.5
    I1 = I * 3 * (1 - Ah) / (8 - 4 * Ah)
    I2 = I * 2 * Ah / (8 - 4 * Ah)
    I3 = I * 1 * (1 - Ah) / (8 - 4 * Ah)
    return (2 * tmp1 * I1, 2 * tmp2 * I2, 2 * tmp3 * I3)


@njit(cache=True)
def ASM_thick_terms(sigm, eps_m, eps_lat, His, Han, m, Num):
    """Per-modulation-point line positions for the anharmonic spin modulation.

    Same modulation physics as ``ASM`` but (a) returns ``co`` = cos^2(theta) of
    the local moment (used to reconstruct the local moment direction) and (b)
    samples the six line positions at the SAME Num points (no per-line
    subsampling), so ``TImod`` can build a 2x2 cross-section matrix per point
    from the local moment direction and average them over the modulation.
    """
    K = K_cei(m)
    Num = int(Num / 6) * 6 + 1
    X = np.linspace(0, K, Num)
    if m >= 0:
        co = SN(X, m) ** 2
    else:
        co = 1 - SN(X, m) ** 2

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

    return (co, v1, v2, v3, v4, v5, v6)

@njit(cache=True)
def SDW_thick_terms_direct(d0, eps0, KeH, H0, hodd, phi_deg, KdH, dev, Num):
    """Six SDW/CDW line positions at ``Num`` wave phases (direct/reference form).

    Fixed spin axis; only the scalar hyperfine parameters vary along the wave.
    Zeeman positions carry the SIGNED field; the KeH/KdH shift correlations use
    |H| (central/quadrupole shift tracks magnitude, not sign). Signs and the
    field->velocity conversion follow ASM_thick_terms / SpectrRelax (Matsnev &
    Rusakov 2014, Eqs. 1-3, 8-10, a+- = 0). ``hodd`` (8 odd field harmonics) and
    ``dev`` (4 even shift harmonics) are float64 arrays; ``psi`` is built with
    ``arange`` (== ``linspace(0, 2*pi, Num, endpoint=False)``, numba-friendly).
    """
    n = int(Num)
    psi = np.arange(n) * (2.0 * np.pi / n)
    H = H0 + 0.0 * psi
    for i in range(len(hodd)):                       # k = 1, 3, 5, ...
        hk = hodd[i]
        if hk != 0.0:
            H = H + hk * np.sin((2 * i + 1) * psi)
    ph_r = phi_deg / 180.0 * np.pi
    # the central shift is a scalar (s-electron density + second-order Doppler) and the quadrupole shift is a lattice/EFG property
    # neither knows the direction of the moment, only its magnitude.
    absH = np.abs(H)
    dl = d0 + KdH * absH
    for i in range(len(dev)):                         # k = 2, 4, 6, 8
        dk = dev[i]
        if dk != 0.0:
            dl = dl + dk * np.sin((2 * (i + 1)) * psi + ph_r)
    ep = eps0 + KeH * absH
    Hc = H / E0_J * c                                 # same conversion as ASM_thick_terms
    v1 = dl + ep + mun * (3 * gex - ggr) / 2 * Hc
    v6 = dl + ep - mun * (3 * gex - ggr) / 2 * Hc
    v2 = dl - ep + mun * (gex - ggr) / 2 * Hc
    v5 = dl - ep - mun * (gex - ggr) / 2 * Hc
    v3 = dl - ep - mun * (gex + ggr) / 2 * Hc
    v4 = dl - ep + mun * (gex + ggr) / 2 * Hc
    return (v1, v2, v3, v4, v5, v6)


# Fixed wave-phase sampling count (positions per period) for SCDW. Not a fit
# parameter: the grid binning keeps it off the Voigt count, so it is set high
# enough to be converged for any wave (~4e-5 even for a pathological all-harmonics
# wave). The accuracy<->speed knob is the per-component grid resolution 'N/Γ'
# (table slot), passed to SDW_thick_terms as ``steps``.
SDW_NUM = 2000


@njit(cache=True)
def _bin_positions(vk, dg, norm):
    """Linear-deposit positions ``vk`` onto a uniform grid of step ``dg``.

    Returns ``(centres, weights)``: each point is split between its two nearest
    grid nodes (piecewise-linear density estimate), weights summing to
    ``len(vk)/norm``. Degenerate input (all equal) -> a single node. Explicit
    loop (numba-friendly; ``np.bincount(minlength=)`` is unsupported under njit).
    """
    lo = np.min(vk)
    hi = np.max(vk)
    span = hi - lo
    if span <= 0.0 or dg <= 0.0:
        centres = np.empty(1, dtype=np.float64)
        centres[0] = lo
        weights = np.empty(1, dtype=np.float64)
        weights[0] = len(vk) / norm
        return centres, weights
    nbins = int(np.ceil(span / dg)) + 1
    centres = lo + dg * np.arange(nbins)
    weights = np.zeros(nbins)
    inv = 1.0 / dg
    for j in range(len(vk)):
        x = (vk[j] - lo) * inv
        i0 = int(np.floor(x))
        if i0 < 0:
            i0 = 0
        elif i0 > nbins - 2:
            i0 = nbins - 2
        frac = x - i0
        weights[i0] += 1.0 - frac
        weights[i0 + 1] += frac
    return centres, weights / norm


@njit(cache=True)
def SDW_thick_terms(d0, eps0, KeH, H0, hodd, phi_deg, KdH, dev, Num, WL, WG, MulCo, steps):
    """Grid-binned line positions + weights for the SCDW branch.

    Same positions as :func:`SDW_thick_terms_direct`, but each line's ``Num``
    positions are binned onto a per-line grid of step ``dg = width / steps`` so
    the caller evaluates the Voigt once per node -- the Voigt count follows
    span/width, NOT ``Num``. ``width`` = max(WL, WG) (WG is often 0, so min would
    give dg=0) floored at the natural line width (NAT_WIDTH), MulCo-scaled like the
    positions. ``steps`` is the per-component grid resolution. The full period is
    always binned (H0/KeH/KdH correlate the six lines, so it cannot be folded).
    Returns ``(grids, weights)`` -- two 6-tuples of float64 arrays (the six
    lines' grids have DIFFERENT lengths, so they cannot be one 2-D array); each
    weight vector sums to 1.
    """
    v = SDW_thick_terms_direct(d0, eps0, KeH, H0, hodd, phi_deg, KdH, dev, Num)
    width = abs(WL)
    if abs(WG) > width:
        width = abs(WG)
    natural = NAT_WIDTH * MulCo          # natural Lorentzian width (WL default), MulCo-scaled
    if width < natural:
        width = natural
    dg = width / steps
    norm = float(len(v[0]))
    g1, w1 = _bin_positions(v[0], dg, norm)
    g2, w2 = _bin_positions(v[1], dg, norm)
    g3, w3 = _bin_positions(v[2], dg, norm)
    g4, w4 = _bin_positions(v[3], dg, norm)
    g5, w5 = _bin_positions(v[4], dg, norm)
    g6, w6 = _bin_positions(v[5], dg, norm)
    return (g1, g2, g3, g4, g5, g6), (w1, w2, w3, w4, w5, w6)


def TImod (x_exp, p, model, EE, x0, MulCo, INS, Distri, Cor, Met = 0, sms_pol=SMS_POL_DEFAULT, Mett = -2, O=[], Di=0, Co=0, V=number_of_baseline_parameters, return_layer_matrix=False, Recon=[], Re=0):
        # SCR = np.array(x_exp)
        SCR = x_exp
        N = np.array([float(0)]*len(SCR))
        # Di = 0
        # Co = 0
        # V = number_of_baseline_parameters
        CH = 1
        CHold = CH


        if Met == -1:
            V = 0
            E = EE
            # print(Distri)
            for i in range(0, len(Distri)):
                Distri[i] = Distri[i].replace('p[', 'O[')
            for i in range(0, int((model=='Corr').sum())):
                Cor[i] = Cor[i].replace('p[', 'O[')
            # print(Distri)
        elif Met == 0:
            Mett = Met
            E = MulCo*SCR + x0*MulCo + np.log((1+EE)/(1-EE))
            if smst.ins_kind(INS) == smst.KIND_GAUSS:
                for i in range (0, int((len(INS))/3)):
                        N += 1*INS[i*3+2]**2*np.exp((-1)*((E-(INS[i*3+1]+SCR)*MulCo)**2/(2*((INS[i*3]**2+NAT_WIDTH/2)*MulCo)**2)))/((INS[i*3]**2+NAT_WIDTH/2)*MulCo)/np.sqrt(2*np.pi)
            else:
                # Theoretical SMS source: S(v) is a unit-area density in mm/s.
                # The Gaussian sum above is, for every term, a density in the same
                # variable -- its argument E - (pos + SCR)*MulCo equals
                # MulCo*(v - pos) with v = log((1+EE)/(1-EE))/MulCo + x0, i.e. it
                # does not depend on SCR at all, and the 1/(width*MulCo) prefactor
                # is the 1/MulCo Jacobian of that substitution. So the drop-in
                # replacement is S(v)/MulCo, a scalar per integration node that
                # broadcasts over the velocity axis exactly as the sum does.
                N = N + smst.ins_shape(
                    INS, np.log((1 + EE) / (1 - EE)) / MulCo + x0) / MulCo
        elif Met == 1:
            Mett = Met
            Wid = INS*MulCo

            # cof = np.array([0.00254718, -0.00993933, 0.64000022, 0.10557177, 0.20270726])
            # E = MulCo * SCR + cof[0] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 4) \
            #                 + cof[1] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 2) \
            #                 + cof[2] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 1 \
            #                 + cof[3] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 2 \
            #                 + cof[4] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 3

            cof = np.array([2.09026977e-02,  2.22979289e+01, -3.35214526e+01])
            E = MulCo * SCR + cof[0] * np.log((1 + EE) / (1 - EE)) \
                            + cof[1] * np.log((2 + EE) / (2 - EE)) \
                            + cof[2] * np.log((3 + EE) / (3 - EE))

            N += Voight(NAT_WIDTH*MulCo, Wid, E - SCR * MulCo)
            # N += Wid/2/np.pi/((E - SCR*MulCo)**2 + (Wid/2)**2)
            # N += (Wid / 2 / np.pi / ((E - SCR * MulCo) ** 2 + (Wid / 2) ** 2)) ** 2 * Wid * np.pi
        elif Met == 2:
            Mett = Met
            # cof = np.array([0.00254718, -0.00993933, 0.64000022, 0.10557177, 0.20270726])
            # E = MulCo * SCR + cof[0] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 4) \
            #                 + cof[1] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 2) \
            #                 + cof[2] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 1 \
            #                 + cof[3] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 2 \
            #                 + cof[4] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 3
            cof = np.array([2.09026977e-02, 2.22979289e+01, -3.35214526e+01])
            E = MulCo * SCR + cof[0] * np.log((1 + EE) / (1 - EE)) \
                            + cof[1] * np.log((2 + EE) / (2 - EE)) \
                            + cof[2] * np.log((3 + EE) / (3 - EE))

            for i in range(0, int(len(INS)/4)):
                S = E - (INS[i * 4 + 2] + SCR) * MulCo
                N += Voight((abs(INS[i * 4]) + 0.0001) * MulCo,
                            (abs(INS[i * 4 + 1]) + 0.0001) * MulCo,
                            S)\
                     * INS[i * 4 + 3]
        elif Met == 3:
            Mett = Met
            Wid = INS*MulCo

            # cof = np.array([0.00254718, -0.00993933, 0.64000022, 0.10557177, 0.20270726])
            # E = MulCo * SCR + cof[0] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 4) \
            #                 + cof[1] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** (1 / 2) \
            #                 + cof[2] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 1 \
            #                 + cof[3] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 2 \
            #                 + cof[4] * np.sign(EE) * (np.tan(np.abs(np.pi / 2 * EE))) ** 3

            cof = np.array([2.09026977e-02,  2.22979289e+01, -3.35214526e+01])
            E = MulCo * SCR + cof[0] * np.log((1 + EE) / (1 - EE)) \
                            + cof[1] * np.log((2 + EE) / (2 - EE)) \
                            + cof[2] * np.log((3 + EE) / (3 - EE))

            N += (Wid / 2 / np.pi / ((E - SCR * MulCo) ** 2 + (Wid / 2) ** 2)) ** 2 * Wid * np.pi

        Kpref = np.pi * (NAT_WIDTH / 2 * MulCo)
        Smat = None       # 2x2 cross-section accumulator for the CURRENT layer (thick components)
        Smat_old = None   # mirrors CHold for the matrix path (used by the Distr reset)
        Tprod = None      # running product of completed-layer AMPLITUDE matrices expm(-Smat/2) (None = identity)

        for i in range (0, len(model)):
            Smat_t = Smat
            if model[i] == 'Singlet':
                I = abs(p[V])
                S = (-1) * p[V + 1]*MulCo + E
                WL = abs(p[V + 2]) * MulCo
                WG = abs(p[V + 3]) * MulCo
                V += 4
                Voi = Voight(WL, WG, S)
                CHt = CH*np.exp((-1)*np.pi*(NAT_WIDTH/2*MulCo)*I*Voi)
            if model[i] == 'Sextet(rough)':
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
                Voi1 = Voight(WL, Ga16, S1)
                Voi2 = Voight(WL, Ga25, S2)
                Voi3 = Voight(WL, Ga34, S3)
                Voi4 = Voight(WL, Ga34, S4)
                Voi5 = Voight(WL, Ga25, S5)
                Voi6 = Voight(WL, Ga16, S6)
                CHt = CH * np.exp((-1) * np.pi * (NAT_WIDTH / 2 * MulCo)\
                                  * (I1*Voi1+I6*Voi6+I2*Voi2+I5*Voi5+I3*Voi3+I4*Voi4))
                V += 14
            if model[i] == 'Hamilton_pc':
                # DEPRECATED (2026-08-26), superseded by 'Hamiltonian' at
                # (A, Am, Ah) = (0, 0, 0), which reproduces this scalar powder
                # exactly (its cross-section matrix is then a multiple of the
                # identity, so the matrix path collapses onto this Beer--Lambert
                # one; verified to 3e-17 per transition). Kept only so a
                # hand-written model file with the old name still evaluates; it
                # is gone from the GUI dropdown and syncmoss.legacy rewrites it
                # to 'Hamiltonian' on load. Remove this branch and Ham_poly in a
                # later release.
                I = abs(p[V])
                delt = p[V+1] * MulCo
                Q = p[V+2]
                H = p[V+3]
                WL = p[V+4] * MulCo
                WG = p[V+5] * MulCo
                eto = p[V+6]
                tet = p[V+7]
                phi = p[V+8]
                Itmp, S = Ham_poly(Q, H, eto, phi, tet)
                I = Itmp * I
                S = S * MulCo
                S += delt
                CHt = CH * np.exp((-1) * np.pi * (NAT_WIDTH / 2 * MulCo)\
                                  * (I[0]*Voight(WL,WG,E-S[0])+I[1]*Voight(WL,WG,E-S[1])+I[2]*Voight(WL,WG,E-S[2])+I[3]*Voight(WL,WG,E-S[3])\
                                    +I[4]*Voight(WL,WG,E-S[4])+I[5]*Voight(WL,WG,E-S[5])+I[6]*Voight(WL,WG,E-S[6])+I[7]*Voight(WL,WG,E-S[7])))
                V += 9
            if model[i] == 'Average_H':
                I = abs(p[V])
                Sig = p[V+1] * MulCo
                Q = p[V+2] * MulCo
                Hin = p[V+3] / TESLA_PER_MMS * (-1) * MulCo
                WL = p[V+4] * MulCo
                WG = p[V+5] * MulCo
                Hex = p[V+6] / TESLA_PER_MMS * MulCo
                K = p[V+7] * MulCo
                J = p[V+8] * MulCo
                tet = p[V+9] / 180 * np.pi
                Num = max(int(p[V+10]), 1)

                CHt = CH * np.exp((-1) * np.pi * (NAT_WIDTH / 2 * MulCo) * I * Angles_min(tet, Num, K, J, Hin, Hex, E, WL, WG, Q, Sig))
                V += 11
            # --- Polarized model components ---------------------------------
            # Each adds a 2x2 cross-section matrix to Smat instead of a scalar
            # exponent; the single matrix exponential is taken after the loop.
            # The texture order parameter A blends between a random powder
            # (A = 0, every Mhat below averages to I and the component reduces
            # EXACTLY to its former scalar form) and a single crystal (A = 1).
            # The Faraday-active models (Sextet, MDGD, Relax_2S) additionally carry
            # the magnetic polar-order parameter Am in [-1, 1] (S1 = Am*sqrt((1+2A)
            # /3), see _texture_s1) that scales the resolved sigma+- Faraday term;
            # Am = 0 (the default) is an unmagnetised texture, Am = A = 1 the
            # fully-magnetised single crystal.
            # (Singlet is isotropic, so it has no polarized form and stays scalar.)
            # DISPERSION: the Voigt-shaped thick components below multiply their
            # matrices by the COMPLEX line shape Voight_c (= V + 1j*D), and the
            # relaxation ones by Blume_c / relax_MS_thick_c, so Smat carries the
            # Kramers--Kronig (dispersive) part and expm(-Smat/2) propagates the
            # nuclear Faraday rotation between non-commuting lines and layers.
            # COMPLEX_VOIGT_METHOD picks the Voigt implementation (and 'off'
            # disables dispersion everywhere for A/B tests); see DISPERSION_SIGN
            # near Voight_c for the one sign convention.
            if model[i] == 'Doublet':
                I = abs(p[V])
                WL = abs(p[V + 3]) * MulCo
                WG1 = abs(p[V + 4]) * MulCo
                WG2 = WG1 * p[V + 8]
                Atex = p[V + 7]                 # uniaxial (fiber) texture order parameter
                S1 = (-1) * (p[V + 1] - p[V + 2]) * MulCo + E   # delta - eps -> line B
                S2 = (-1) * (p[V + 1] + p[V + 2]) * MulCo + E   # delta + eps -> line A
                mx, my, mz = _axis_xyz(p[V + 5], p[V + 6])
                mm = _mm_perp(mx, my)
                Mh_A = _texture_blend(1.5 * (_I2 - mm), Atex)   # line A: Delta m = +/-1 (the 3/4 sin^2 line)
                Mh_B = _texture_blend(0.5 * _I2 + 1.5 * mm, Atex)  # line B: mixed Delta m = 0 and +/-1
                VoiB = Voight_c(WL, WG1, S1)
                VoiA = Voight_c(WL, WG2, S2)
                add = Kpref * (0.5 * I) * (VoiA[:, None, None] * Mh_A[None, :, :]
                                               + VoiB[:, None, None] * Mh_B[None, :, :])
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 9
            if model[i] == 'Sextet':
                I = abs(p[V])
                I13 = p[V + 13]
                Aeff = 0.5
                I1 = I * (4 * I13 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                I2 = I * 2 * Aeff / (8 - 4 * Aeff)
                I3 = I * (4 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                HH = p[V + 3] / TESLA_PER_MMS
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 10] * MulCo + E
                S2 = (-1) * (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo + p[V + 11] * MulCo + E
                S3 = (-1) * (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo - p[V + 11] * MulCo + E
                S4 = (-1) * (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo + p[V + 11] * MulCo + E
                S5 = (-1) * (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo - p[V + 11] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 10] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                WG = abs(p[V + 5]) * MulCo
                GaH = abs(p[V + 12]) / 2 / TESLA_PER_MMS * MulCo
                Ga16 = (WG ** 2 + GaH ** 2) ** (1 / 2)
                Ga25 = (WG ** 2 + (LINE_RATIO_25 * GaH) ** 2) ** (1 / 2)
                Ga34 = (WG ** 2 + (LINE_RATIO_34 * GaH) ** 2) ** (1 / 2)
                Voi1 = Voight_c(WL, Ga16, S1)
                Voi2 = Voight_c(WL, Ga25, S2)
                Voi3 = Voight_c(WL, Ga34, S3)
                Voi4 = Voight_c(WL, Ga34, S4)
                Voi5 = Voight_c(WL, Ga25, S5)
                Voi6 = Voight_c(WL, Ga16, S6)
                Atex = p[V + 8]                   # uniaxial (fiber) texture order parameter
                Am = p[V + 9]                     # magnetic polar order Am in [-1,1] (S1 fraction), right after A
                mx, my, mz = _axis_xyz(p[V + 6], p[V + 7])
                Msig = _texture_blend(_mhat_dm1_sym(mx, my), Atex)       # symmetric (quadratic) sigma part
                Mfar = 1.5 * _texture_s1(Atex, Am) * 1j * mz * _J2       # +/- Faraday (magneto-optical) term
                Msp = Msig + Mfar                                        # sigma+ (Delta m = +1): lines 3, 6
                Msm = Msig - Mfar                                        # sigma- (Delta m = -1): lines 1, 4
                Mpi = _texture_blend(_mhat_dm0(mx, my), Atex)            # pi (Delta m = 0): lines 2, 5
                add = Kpref * (
                    I1 * (Voi1[:, None, None] * Msm[None, :, :] + Voi6[:, None, None] * Msp[None, :, :])
                    + I2 * (Voi2[:, None, None] * Mpi[None, :, :] + Voi5[:, None, None] * Mpi[None, :, :])
                    + I3 * (Voi3[:, None, None] * Msp[None, :, :] + Voi4[:, None, None] * Msm[None, :, :]))
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 14
            if model[i] == 'MDGD':
                I = abs(p[V])
                I13 = p[V + 16]
                Aeff = 0.5
                I1 = I * (4 * I13 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                I2 = I * 2 * Aeff / (8 - 4 * Aeff)
                I3 = I * (4 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                HH = p[V + 3] / TESLA_PER_MMS
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 14] * MulCo + E
                S2 = (-1) * (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo + p[V + 15] * MulCo + E
                S3 = (-1) * (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo - p[V + 15] * MulCo + E
                S4 = (-1) * (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2]) * MulCo + p[V + 15] * MulCo + E
                S5 = (-1) * (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2]) * MulCo - p[V + 15] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 14] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                Guni = abs(p[V + 5]) * MulCo
                Gh = abs(p[V + 6]) * MulCo
                Gde = p[V + 7]
                Gdh = p[V + 8]
                Geh = p[V + 9]
                Cd = [1, 1, 1, 1, 1, 1]
                Ce = [1, -1, -1, -1, -1, 1]
                # dv_k/dH per line, mm/s per T; signed, for the Gdh/Geh cross terms
                Ch = [-LINE_SHIFT_16, -LINE_SHIFT_25, -LINE_SHIFT_34,
                      LINE_SHIFT_34, LINE_SHIFT_25, LINE_SHIFT_16]
                Gfinal = []
                for j in range(0, 6):
                    Gfinal.append(np.sqrt(abs(Guni ** 2 + Ch[j] ** 2 * Gh ** 2 + Cd[j] * Ce[j] * Gde * Guni ** 2 * max(0, (1 - (abs(Gdh) + abs(Geh)) ** 2)) + Cd[j] * Ch[j] * 2 * Gdh * Guni * Gh + Ce[j] * Ch[j] * 2 * Geh * Guni * Gh)))
                Voi1 = Voight_c(WL, Gfinal[0], S1)
                Voi2 = Voight_c(WL, Gfinal[1], S2)
                Voi3 = Voight_c(WL, Gfinal[2], S3)
                Voi4 = Voight_c(WL, Gfinal[3], S4)
                Voi5 = Voight_c(WL, Gfinal[4], S5)
                Voi6 = Voight_c(WL, Gfinal[5], S6)
                Atex = p[V + 12]                  # uniaxial (fiber) texture order parameter
                Am = p[V + 13]                    # magnetic polar order Am in [-1,1] (S1 fraction), right after A
                mx, my, mz = _axis_xyz(p[V + 10], p[V + 11])
                Msig = _texture_blend(_mhat_dm1_sym(mx, my), Atex)       # symmetric (quadratic) sigma part
                Mfar = 1.5 * _texture_s1(Atex, Am) * 1j * mz * _J2       # +/- Faraday (magneto-optical) term
                Msp = Msig + Mfar
                Msm = Msig - Mfar
                Mpi = _texture_blend(_mhat_dm0(mx, my), Atex)
                add = Kpref * (
                    I1 * (Voi1[:, None, None] * Msm[None, :, :] + Voi6[:, None, None] * Msp[None, :, :])
                    + I2 * (Voi2[:, None, None] * Mpi[None, :, :] + Voi5[:, None, None] * Mpi[None, :, :])
                    + I3 * (Voi3[:, None, None] * Msp[None, :, :] + Voi4[:, None, None] * Msm[None, :, :]))
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 17
            if model[i] == 'Relax_MS':
                I = abs(float(p[V]) * 2)
                sig0 = float(p[V + 1]) * MulCo
                eps = float(p[V + 2]) * MulCo
                Hv = float(p[V + 3]) * MulCo / 2 / TESLA_PER_MMS
                W = float(p[V + 4]) * MulCo / 2
                th = p[V + 5]
                ph = p[V + 6]
                Atex = p[V + 7]              # uniaxial (fiber) texture order parameter
                R = float(p[V + 8])
                alfa = float(p[V + 9])
                Sspin = float(p[V + 10])
                _relax_grp = relax_MS_thick if COMPLEX_VOIGT_METHOD == 'off' else relax_MS_thick_c
                g1, g2, g3 = _relax_grp(Sspin, E, I, sig0, eps, Hv, W, R, alfa)
                mx, my, mz = _axis_xyz(th, ph)
                Md1 = _texture_blend(_mhat_dm1_sym(mx, my), Atex)  # outer/inner Delta m = +/-1 (groups blend sigma+/-)
                Mpi = _texture_blend(_mhat_dm0(mx, my), Atex)      # middle Delta m = 0
                add = Kpref * (g1[:, None, None] * Md1[None, :, :]
                                   + g2[:, None, None] * Mpi[None, :, :]
                                   + g3[:, None, None] * Md1[None, :, :])
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 11
            if model[i] == 'Relax_2S':
                I = abs(p[V])
                Aeff = 0.5
                I1 = I * 3 * (1 - Aeff) / (8 - 4 * Aeff)
                I2 = I * 2 * Aeff / (8 - 4 * Aeff)
                I3 = I * 1 * (1 - Aeff) / (8 - 4 * Aeff)
                Sig1 = p[V + 1] * MulCo
                Q1 = p[V + 2] / 3 * MulCo
                # Blume wants the field in mm/s per unit g (it applies ggr/gex itself)
                H1 = p[V + 3] * MMS_PER_T_PER_G * MulCo
                Sig2 = p[V + 4] * MulCo
                Q2 = p[V + 5] / 3 * MulCo
                H2 = p[V + 6] * MMS_PER_T_PER_G * MulCo
                WL = p[V + 7] * MulCo
                Atex = p[V + 10]             # uniaxial (fiber) texture order parameter
                Am = p[V + 11]               # magnetic polar order Am in [-1,1] (S1 fraction), right after A
                We = p[V + 12] * MulCo
                R = p[V + 13]
                mx, my, mz = _axis_xyz(p[V + 8], p[V + 9])
                Msig = _texture_blend(_mhat_dm1_sym(mx, my), Atex)       # symmetric (quadratic) sigma part
                Mfar = 1.5 * _texture_s1(Atex, Am) * 1j * mz * _J2       # +/- Faraday (magneto-optical) term
                Msp = Msig + Mfar
                Msm = Msig - Mfar
                Mpi = _texture_blend(_mhat_dm0(mx, my), Atex)
                _Bl = Blume if COMPLEX_VOIGT_METHOD == 'off' else Blume_c
                B1 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1 / 2, -3 / 2, E, R)  # sigma-
                B2 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1 / 2, 3 / 2, E, R)    # sigma+
                B3 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1 / 2, -1 / 2, E, R)  # pi
                B4 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1 / 2, 1 / 2, E, R)    # pi
                B5 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1 / 2, 1 / 2, E, R)   # sigma+
                B6 = _Bl(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1 / 2, -1 / 2, E, R)   # sigma-
                add = Kpref * (
                    I1 * (B1[:, None, None] * Msm[None, :, :] + B2[:, None, None] * Msp[None, :, :])
                    + I2 * (B3[:, None, None] * Mpi[None, :, :] + B4[:, None, None] * Mpi[None, :, :])
                    + I3 * (B5[:, None, None] * Msp[None, :, :] + B6[:, None, None] * Msm[None, :, :]))
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 14
            if model[i] == 'Hamilton_mc':
                # DEPRECATED (2026-08-26), superseded by 'Hamiltonian' at
                # (A, Am, Ah) = (1, 1, 1). Kept only so a hand-written model
                # file with the old name still evaluates; it is gone from the GUI
                # dropdown and syncmoss.legacy rewrites it (plus its 11-parameter
                # pre-alfak form) to 'Hamiltonian' on load. Remove this branch,
                # Ham_mono_thick and Ham_mono_thick_CMS in a later release.
                I = abs(p[V])
                delt = p[V + 1] * MulCo
                Q = p[V + 2]
                H = p[V + 3]
                WL = p[V + 4] * MulCo
                WG = p[V + 5] * MulCo
                eto = p[V + 6]
                tet = p[V + 7]
                phi = p[V + 8]
                tetr = p[V + 9]
                phir = p[V + 10]
                alfak = p[V + 11]
                # SMS (Mett != 1): (tetr, phir) is the radiation field h and
                # alfak rotates the beam k about h. CMS (Mett == 1, unpolarized):
                # (tetr, phir) is the beam k itself and alfak is redundant -- the
                # half-trace readout (below) is basis-invariant. Same shared core.
                if Mett == 1:
                    Pmat, S = Ham_mono_thick_CMS(Q, H, eto, phi, tet, phir, tetr)
                else:
                    Pmat, S = Ham_mono_thick(Q, H, eto, phi, tet, phir, tetr, alfak)
                S = S * MulCo + delt
                add = np.zeros((len(E), 2, 2), dtype=complex)
                for k in range(0, 8):
                    add += Voight_c(WL, WG, E - S[k])[:, None, None] * Pmat[k][None, :, :]
                add = Kpref * I * add
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 12
            if model[i] == 'Hamiltonian':
                # Mosaic textured full Hamiltonian: the 12 single-crystal
                # parameters plus the three ODF order parameters A, Am, Ah.
                # It replaces BOTH former Hamiltonian models -- (1, 1, 1) is the
                # single crystal ('Hamilton_mc'), (0, 0, 0) the random powder
                # ('Hamilton_pc'), Ah = 0 a fiber texture -- see Ham_mosaic.
                # SMS (Mett != 1): (tetr, phir) is the radiation field h and
                # alfak rotates the beam k about it; the mosaic is textured about
                # h. CMS (Mett == 1, unpolarized): (tetr, phir) is the beam k
                # itself, the mosaic is textured about the beam and alfak stays
                # redundant (the half-trace readout below is basis-invariant).
                I = abs(p[V])
                delt = p[V + 1] * MulCo
                Q = p[V + 2]
                H = p[V + 3]
                WL = p[V + 4] * MulCo
                WG = p[V + 5] * MulCo
                eto = p[V + 6]
                tet = p[V + 7]
                phi = p[V + 8]
                tetr = p[V + 9]
                phir = p[V + 10]
                alfak = p[V + 11]
                Atex = p[V + 12]                 # A   : <P2(cos chi)> of the mosaic wobble
                Am = p[V + 13]                   # Am : polar order of that wobble (Faraday)
                Ah = p[V + 14]                   # Ah : azimuthal order about the reference axis
                Pmat, S = Ham_mosaic(Q, H, eto, phi, tet, tetr, phir, alfak,
                                     Atex, Am, Ah, Mett == 1)
                S = S * MulCo + delt
                Piso = 0.5 * (Pmat[:, 0, 0] + Pmat[:, 1, 1]).real   # scalar part per line
                if np.max(np.abs(Pmat - Piso[:, None, None] * _I2[None, :, :])) <= 1e-12:
                    # A fully disordered mosaic (the random-powder limit, and any
                    # A = Ah = 0) makes every P[k] a multiple of the identity, so
                    # the component is isotropic in the polarization plane and can
                    # take the SCALAR path -- exactly what the former 'Hamilton_pc'
                    # did, at ~2.4x less cost. NOT an approximation: a scalar
                    # contribution s(E)*1 to Smat factors out of the matrix
                    # exponential and out of every layer product as exp(-s/2), and
                    # |exp(-s/2)|^2 = exp(-Re s), i.e. its dispersive part is a
                    # global phase that the Gram readout cancels -- which is also
                    # why the real Voight() is the right line shape here.
                    Voi = np.zeros(len(E))
                    for k in range(0, 8):
                        Voi = Voi + Piso[k] * Voight(WL, WG, E - S[k])
                    CHt = CH * np.exp((-1) * Kpref * I * Voi)
                else:
                    add = np.zeros((len(E), 2, 2), dtype=complex)
                    for k in range(0, 8):
                        add += Voight_c(WL, WG, E - S[k])[:, None, None] * Pmat[k][None, :, :]
                    add = Kpref * I * add
                    Smat_t = add if Smat is None else Smat + add
                    CHt = CH
                V += 15
            if model[i] == 'ASM':
                # Anharmonic spin modulation, polarized. The moment direction
                # rotates along the cycloid (a distribution of H orientations
                # with the radiation h held fixed); each modulation point is a
                # sextet whose 2x2 matrix is built from the LOCAL moment
                # direction, and the matrices are averaged over the modulation.
                # Geometry: (theta_k, phi_h) is the easy (anharmonicity) axis u
                # in the lab frame (beam k = z, h = x); the NEW angle omega orients
                # the cycloid PLANE about u by choosing the second in-plane axis v.
                # The moment swings in the (u, v) plane with polar angle theta where
                # cos^2(theta) = co. Each sample point is averaged over the mirror
                # pair (+/- v): psi and -psi are equally populated in the ideal
                # cycloid and the symmetric matrices are even under m -> -m, so this
                # two-point average turns the quarter-period sum into an EXACT
                # full-period average (it also cancels the u-v cross terms that the
                # old single-direction sampling retained -- a correction, not a
                # regression). The cycloid samples both senses of the moment, so the
                # magneto-optical term averages out and the symmetric Delta m = +/-1
                # matrix is used. Old fits use omega = omega_0(theta_k, phi_h) =
                # degrees(arctan2(-sin(phi_h), cos(theta_k) * cos(phi_h))), the
                # transverse-to-u part of h (=90 in the degenerate u || h case).
                I = abs(p[V])
                sigm = p[V + 1] * MulCo
                eps_m = p[V + 2] * MulCo
                eps_lat = p[V + 3] * MulCo
                His = p[V + 4] * MulCo
                Han = p[V + 5] * MulCo
                WL = abs(p[V + 6]) * MulCo
                WG = abs(p[V + 7]) * MulCo
                m_asm = p[V + 8]
                th = p[V + 9]
                ph = p[V + 10]
                Atex = p[V + 11]                 # uniaxial (fiber) texture order parameter
                Num = int(abs(p[V + 12]))
                I13 = p[V + 13]
                om = p[V + 14]                   # cycloid-plane angle about the easy axis, deg
                co, v1, v2, v3, v4, v5, v6 = ASM_thick_terms(sigm, eps_m, eps_lat, His, Han, m_asm, Num)
                Nn = len(co)
                Aeff = 0.5
                I1 = I * (4 * I13 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                I2 = I * 2 * Aeff / (8 - 4 * Aeff)
                I3 = I * (4 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)
                # easy (anharmonicity) axis u at (theta_k, phi_h)
                ux, uy, uz = _axis_xyz(th, ph)
                # spherical tangent vectors at (th, ph): e_theta, e_phi (both
                # perpendicular to u). Only the transverse (x, y) parts are used,
                # as the merged matrices carry no Faraday term; e_theta, e_phi
                # exist at every (th, ph), so no degenerate fallback is needed.
                thr = th / 180.0 * np.pi
                phr = ph / 180.0 * np.pi
                etx, ety = np.cos(thr) * np.cos(phr), np.cos(thr) * np.sin(phr)   # e_theta
                epx, epy = -np.sin(phr), np.cos(phr)                             # e_phi
                # second in-plane axis v, one angle omega about u
                omr = om / 180.0 * np.pi
                vx = np.cos(omr) * etx + np.sin(omr) * epx
                vy = np.cos(omr) * ety + np.sin(omr) * epy
                add = np.zeros((len(E), 2, 2), dtype=complex)
                for j in range(0, Nn):
                    cphi = np.sqrt(abs(co[j]))
                    sphi = np.sqrt(max(0.0, 1.0 - co[j]))
                    # mirror pair (+/- v) about the easy axis; the symmetric
                    # matrices are even under m -> -m, so this two-point average
                    # IS the exact full-period average (see the branch comment).
                    mxa, mya = cphi * ux + sphi * vx, cphi * uy + sphi * vy
                    mxb, myb = cphi * ux - sphi * vx, cphi * uy - sphi * vy
                    Md1 = _texture_blend(0.5 * (_mhat_dm1_sym(mxa, mya)
                                                + _mhat_dm1_sym(mxb, myb)), Atex)
                    Mpi = _texture_blend(0.5 * (_mhat_dm0(mxa, mya)
                                                + _mhat_dm0(mxb, myb)), Atex)
                    add += (I1 * (Voight_c(WL, WG, E - v1[j]) + Voight_c(WL, WG, E - v6[j]))[:, None, None] * Md1[None, :, :]
                            + I2 * (Voight_c(WL, WG, E - v2[j]) + Voight_c(WL, WG, E - v5[j]))[:, None, None] * Mpi[None, :, :]
                            + I3 * (Voight_c(WL, WG, E - v3[j]) + Voight_c(WL, WG, E - v4[j]))[:, None, None] * Md1[None, :, :]) / Nn
                add = Kpref * add
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 15
            if model[i] == 'SCDW':
                # Spin/charge density wave, polarized (thick). The spin AXIS is
                # FIXED at (theta_k, phi_h); only the SCALAR hyperfine parameters
                # (signed field H, central shift, quadrupole shift) are modulated
                # along the wave. Unlike ASM the 2x2 matrices are therefore the
                # plain Sextet ones evaluated ONCE and hoisted out of the
                # modulation loop; only the six line positions vary per point.
                #
                # The field carries its SIGN (SDW_thick_terms does NOT take
                # abs(H)): a sign change of H swaps v1<->v6 and v3<->v4 while the
                # matrix assignment ({1,4}->sigma-, {3,6}->sigma+, {2,5}->pi)
                # stays fixed, reproducing the helicity swap of a reversed moment.
                #
                # Am is a fit parameter, EXACTLY the Sextet branch's magnetic
                # polar-order parameter: each site carries the resolved sigma+-
                # Faraday term Am*sqrt((1+2A)/3)*i*mz*J2 (do NOT use the merged
                # _mhat_dm1_sym). Whether that term survives the modulation is set
                # by the WAVE, independently of Am's value: for a BALANCED wave
                # (H0 = 0 and KeH = KdH = 0, pure odd harmonics) the positions pair
                # up as v1(psi+pi) = v6(psi) etc., so L1 == L6 and L3 == L4 and the
                # +/- Faraday cancels EXACTLY for any Am. With a non-zero base
                # field H0 (the usual case) or a field-shift correlation KeH/KdH
                # the wave is offset, L1 != L6, and a real (Am-scaled),
                # thickness-dependent Faraday signal remains. EDGE CASE: all
                # harmonics = KeH = KdH = 0 with H0 != 0 gives constant positions,
                # i.e. a single sextet reproducing the 'Sextet' branch at the same
                # (delta, eps, H0, widths, theta_k, phi_h, A, Am, I13).
                I    = abs(p[V]);        d0  = p[V + 1] * MulCo
                eps0 = p[V + 2] * MulCo; H0  = p[V + 3] * MulCo
                WL   = abs(p[V + 4]) * MulCo
                WG   = abs(p[V + 5]) * MulCo
                th   = p[V + 6];   ph = p[V + 7];   Atex = p[V + 8];  Am = p[V + 9]
                KdH  = p[V + 10];  KeH = p[V + 11]
                phi  = p[V + 12]                                    # CDW phase Phi [deg]
                hodd = np.array([p[V + 13 + k] * MulCo for k in range(8)])   # h1, h3, ..., h15
                dev  = np.array([p[V + 21 + k] * MulCo for k in range(4)])   # d2, d4, d6, d8
                steps = max(abs(p[V + 25]), 1.0);  I13 = p[V + 26]   # slot 25: grid steps per line width

                # The settable accuracy<->speed knob is 'steps' (grid resolution,
                # per component, from the table). Num (wave-phase sampling) is
                # fixed internally at SDW_NUM -- cheap and converged for any wave.
                grids, wts = SDW_thick_terms(d0, eps0, KeH, H0, hodd, phi, KdH, dev,
                                             SDW_NUM, WL, WG, MulCo, steps)
                Aeff = 0.5                       # isotropic weights; orientation lives
                I1 = I * (4 * I13 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)   # in the matrices
                I2 = I * 2 * Aeff / (8 - 4 * Aeff)
                I3 = I * (4 / (I13 + 1)) * (1 - Aeff) / (8 - 4 * Aeff)

                # Matrices EXACTLY as in the 'Sextet' branch at (th, ph), texture
                # Atex, magnetic polar order Am: resolved sigma-/sigma+ (Faraday
                # included) and pi -- NOT the merged _mhat_dm1_sym.
                mx, my, mz = _axis_xyz(th, ph)
                Msig = _texture_blend(_mhat_dm1_sym(mx, my), Atex)       # symmetric (quadratic) sigma part
                Mfar = 1.5 * _texture_s1(Atex, Am) * 1j * mz * _J2       # +/- Faraday (magneto-optical) term
                Msp = Msig + Mfar                                        # sigma+ (Delta m = +1): lines 3, 6
                Msm = Msig - Mfar                                        # sigma- (Delta m = -1): lines 1, 4
                Mpi = _texture_blend(_mhat_dm0(mx, my), Atex)            # pi (Delta m = 0): lines 2, 5

                # Positions vary, matrices don't: each line shape is the Voigt
                # summed over its GRID nodes weighted by the binned wave-position
                # density (weights already sum to 1 -> the wave average). The
                # Voigt count is the grid size, independent of Num.
                def _line_shape(g, w):
                    return (Voight_c(WL, WG, E[:, None] - g[None, :]) * w[None, :]).sum(axis=1)
                L1 = _line_shape(grids[0], wts[0]);  L2 = _line_shape(grids[1], wts[1])
                L3 = _line_shape(grids[2], wts[2]);  L4 = _line_shape(grids[3], wts[3])
                L5 = _line_shape(grids[4], wts[4]);  L6 = _line_shape(grids[5], wts[5])
                add = Kpref * (I1 * (L1[:, None, None] * Msm + L6[:, None, None] * Msp)
                             + I2 * (L2 + L5)[:, None, None] * Mpi
                             + I3 * (L3[:, None, None] * Msp + L4[:, None, None] * Msm))
                Smat_t = add if Smat is None else Smat + add
                CHt = CH
                V += 27
            if model[i] == 'Layer':
                # Physical layer boundary (no parameters). Within a layer the
                # cross-sections add (into Smat); between layers the beam
                # AMPLITUDE is propagated coherently, because the polarization
                # state leaving one layer enters the next. The per-layer
                # amplitude operator is expm(-Smat/2) (half the intensity
                # cross-section, as a matrix); these multiply in beam order and
                # do NOT commute, so the layer order is physical for a polarized
                # source. Flush the current layer's amplitude matrix into the
                # running product and start a fresh (empty) layer. (Multiplying
                # the full expm(-Smat) and reading its diagonal instead -- the
                # former behaviour -- gives an order-BLIND result even for an
                # SMS, because Smat is Hermitian; see the readout after the loop.)
                if Smat is not None:
                    layerT = _expm_neg(Smat / 2)
                    Tprod = layerT if Tprod is None else np.matmul(layerT, Tprod)
                CHt = CH
                Smat_t = None

            if model[i] == 'Variables':
                V += numco
                CHt = CH
            if model[i] == 'Expression':
                V += 1
                CHt = CH

            if model[i] == 'Distr':
                Num = int(p[V + 3])
                CH = CHold
                Smat = Smat_old      # mirror the CH reset for the matrix accumulator
                Smat_t = Smat
                X = np.linspace(np.array(p[V+1]), np.array(p[V+2]), Num)
                # print(p[V+1], p[V+2], eval(str(Distri[Di])))
                # print(Distri[Di])
                ge = eval(str(Distri[Di])) + 0*X
                ge = ge / np.sum(ge, axis=0) #*(len(ge)!=1) + Num*ge*(len(ge)==1))
                # print(Distri[Di])
                # print(ge)
                k = i-1
                # kk = []
                Dk = 0
                Ck = 0
                Rk = 0

                while model[k] == 'Distr' or model[k] == 'Corr' or model[k] == 'Recon':
                    if model[k] == 'Distr':
                        # kk.append(k)
                        Dk += 1
                    if model[k] == 'Corr':
                        Ck += 1
                    if model[k] == 'Recon':
                        Rk += 1
                    k -= 1

                Vnum = int(4*(model[k]=='Singlet') + 9*(model[k]=='Doublet') + 14*(model[k]=='Sextet') + 14*(model[k]=='Sextet(rough)') + 17 * (model[k] == 'MDGD')\
                           + 11*(model[k]=='Relax_MS') + numco*(model[k]=='Variables') + 11*(model[k]=='Average_H') + 15*(model[k]=='ASM') + 27*(model[k]=='SCDW')\
                           + 14*(model[k]=='Relax_2S')) + 15*(model[k]=='Hamiltonian') + 12*(model[k]=='Hamilton_mc') + 9*(model[k]=='Hamilton_pc') + 1*(model[k]=='Expression')

                # Total flat slots from the base component through every preceding
                # distribution marker (Corr=2, Distr=5, Recon=7 each): the base
                # amplitude sits at p[V-Prev] and pN is reshaped as (Prev, Num).
                Prev = Vnum + Ck*2 + Dk*5 + Rk*7

                model_d = np.array([model[k:i]] * Num).flatten()
                # print('distr model', model_d, str(Distri[Di]))
                # print(Vnum, k, i)

                pN = np.reshape(np.ravel(np.array([p[V-Prev:V]]*Num), order='F'), (Prev, Num))

                pN[0] = ge * p[V-Prev]

                pN[int(p[V])] = X
                V += 5

                for j in range(1, len(model)-i):
                    if model[i+j] == 'Corr':
                        # print(Co, 'Corr', str(Cor[Co]), X)
                        pN[int(p[V])] = eval(str(Cor[Co])) + 0*X
                        V += 2
                        Co += 1
                        # Ck += 1 # CHECK!!!!!!!
                    else:
                        break
                pN = np.reshape(np.ravel(pN, order='F'), (Num, Prev)).flatten()

                MultiDistr = 0
                for j in range(i+1, len(model)):
                    if model[j] != 'Corr':
                        # print(str('it is MULTIDIMENTIONAL!')*(model[j] == 'Distr') + str('it is single!')*(model[j] != 'Distr'))
                        MultiDistr = 1*(model[j] == 'Distr' or model[j] == 'Recon')
                        break


                # A distributed component joins the CURRENT layer. The recursive
                # call below evaluates the Num copies and -- because it is asked
                # with return_layer_matrix=True -- hands back (scalar_part,
                # Sigma_matrix) WITHOUT exponentiating or reading out. The copies'
                # matrices are already summed into Sigma inside the recursion (the
                # fine-grain mixture: matrices averaged before the exponential),
                # and here Sigma is ADDED to this layer's Smat while the scalar
                # part multiplies into CH. So a distributed component behaves
                # exactly like an undistributed one placed in the same layer:
                # everything between 'Layer' markers sums into one Smat, one
                # expm(-Smat/2), one rho-readout at the top level -- no double
                # exponential and no doubled polarization readout. Scalars
                # (Smat_in is None) simply multiply into CH, unchanged whether or
                # not they sit in a Layer. The parent's matrix accumulator was
                # rewound to Smat_old above, so the component's own undistributed
                # contribution from the previous loop iteration is removed first.
                if MultiDistr == 0:
                    # print('Dk =', Dk, ' Ck =', Ck)
                    if len(O)==0 and (Dk!=0 or Rk!=0):
                        O = p
                    # CHt = CH * (TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[Di-Dk:Di]]*Num).flatten(), np.array([Cor[Co-Ck:Co]]*Num).flatten(), Met = -1, Mett = Mett, O=O))
                    mDk = 0
                    mCk = 0
                    mRk = 0
                    for mk in range(0, k):
                        if model[mk] == 'Distr':
                            mDk += 1
                        if model[mk] == 'Corr':
                            mCk += 1
                        if model[mk] == 'Recon':
                            mRk += 1
                    CH_in, Smat_in = TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[mDk:mDk+Dk]]*Num).flatten(), np.array([Cor[mCk:mCk+Ck]]*Num).flatten(), Met = -1, Mett = Mett, O=O, return_layer_matrix=True, sms_pol=sms_pol, Recon=list(Recon[mRk:mRk+Rk])*Num)
                    CHt = CH * CH_in                                  # scalar (thin) part multiplies in, as before
                    if Smat_in is not None:                          # thick part joins the CURRENT layer's Smat
                        Smat_t = Smat_in if Smat is None else Smat + Smat_in
                else:
                    CHt = CH
                Di += 1
                # print('Distri proceed')

            if model[i] == 'Recon':
                # Model-independent reconstruction. Identical replication machinery
                # to 'Distr', except the per-channel density `ge` is the vector of
                # FREE fit weights held in the parallel Recon list (like a Distr's
                # PDF string in Distri) rather than an evaluated PDF expression.
                # Recon's 7 flat slots are par, L, R, Num, D_dif, D_dif2 and a single
                # weight-vector placeholder; only par/L/R/Num are read here --
                # D_dif/D_dif2 are fit-side smoothness regularisation and never enter
                # the forward spectrum, and the weights arrive numerically via Recon.
                Num = int(p[V + 3])
                CH = CHold
                Smat = Smat_old
                Smat_t = Smat
                X = np.linspace(np.array(p[V+1]), np.array(p[V+2]), Num)
                ge = np.asarray(Recon[Re], dtype=float).flatten()
                ge = ge / np.sum(ge, axis=0)
                k = i-1
                Dk = 0
                Ck = 0
                Rk = 0

                while model[k] == 'Distr' or model[k] == 'Corr' or model[k] == 'Recon':
                    if model[k] == 'Distr':
                        Dk += 1
                    if model[k] == 'Corr':
                        Ck += 1
                    if model[k] == 'Recon':
                        Rk += 1
                    k -= 1

                Vnum = int(4*(model[k]=='Singlet') + 9*(model[k]=='Doublet') + 14*(model[k]=='Sextet') + 14*(model[k]=='Sextet(rough)') + 17 * (model[k] == 'MDGD')\
                           + 11*(model[k]=='Relax_MS') + numco*(model[k]=='Variables') + 11*(model[k]=='Average_H') + 15*(model[k]=='ASM') + 27*(model[k]=='SCDW')\
                           + 14*(model[k]=='Relax_2S')) + 15*(model[k]=='Hamiltonian') + 12*(model[k]=='Hamilton_mc') + 9*(model[k]=='Hamilton_pc') + 1*(model[k]=='Expression')

                Prev = Vnum + Ck*2 + Dk*5 + Rk*7

                model_d = np.array([model[k:i]] * Num).flatten()

                pN = np.reshape(np.ravel(np.array([p[V-Prev:V]]*Num), order='F'), (Prev, Num))

                pN[0] = ge * p[V-Prev]

                pN[int(p[V])] = X
                V += 7

                for j in range(1, len(model)-i):
                    if model[i+j] == 'Corr':
                        pN[int(p[V])] = eval(str(Cor[Co])) + 0*X
                        V += 2
                        Co += 1
                    else:
                        break
                pN = np.reshape(np.ravel(pN, order='F'), (Num, Prev)).flatten()

                MultiDistr = 0
                for j in range(i+1, len(model)):
                    if model[j] != 'Corr':
                        MultiDistr = 1*(model[j] == 'Distr' or model[j] == 'Recon')
                        break

                if MultiDistr == 0:
                    if len(O)==0 and (Dk!=0 or Rk!=0):
                        O = p
                    mDk = 0
                    mCk = 0
                    mRk = 0
                    for mk in range(0, k):
                        if model[mk] == 'Distr':
                            mDk += 1
                        if model[mk] == 'Corr':
                            mCk += 1
                        if model[mk] == 'Recon':
                            mRk += 1
                    CH_in, Smat_in = TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[mDk:mDk+Dk]]*Num).flatten(), np.array([Cor[mCk:mCk+Ck]]*Num).flatten(), Met = -1, Mett = Mett, O=O, return_layer_matrix=True, sms_pol=sms_pol, Recon=list(Recon[mRk:mRk+Rk])*Num)
                    CHt = CH * CH_in
                    if Smat_in is not None:
                        Smat_t = Smat_in if Smat is None else Smat + Smat_in
                else:
                    CHt = CH
                Re += 1

            if model[i] == 'Nbaseline':
                break

            CHold = CH
            CH = CHt
            Smat_old = Smat
            Smat = Smat_t

        if return_layer_matrix:
            # Inner Distr recursion: hand back the accumulated layer cross-section
            # matrix WITHOUT exponentiating or reading it out, so the distribution
            # joins the caller's current layer (matrices sum before the single
            # exponential) and the rho-readout happens once at the top level. CH
            # carries any scalar (thin) part. model_d never contains a 'Layer', so
            # Tprod is None here and Smat holds the whole distributed cross-section.
            return (CH, Smat)
        if Smat is not None:
            layerT = _expm_neg(Smat / 2)
            Tprod = layerT if Tprod is None else np.matmul(layerT, Tprod)
        if Tprod is not None:
            # Amplitude readout. Tprod = expm(-Smat_n/2) ... expm(-Smat_1/2) is
            # the stack's 2x2 amplitude transmission operator (beam order, last
            # layer leftmost). The detected intensity for an incident beam with
            # polarization density matrix rho = diag(rho11, rho22) is
            #     C_a = tr[Tprod rho Tprod^H] = tr[rho . Tprod^H Tprod]
            #         = rho11 * Gram[0,0] + rho22 * Gram[1,1],  Gram = Tprod^H Tprod,
            # i.e. C_a = sum_i rho_ii ||Tprod e_i||^2. Propagating the amplitude
            # (half cross-section) and only then forming the intensity is what
            # makes the layer order visible to a polarized source. Limits:
            # whenever the absorptive and dispersive parts of Smat commute
            # (a single line, n_z = 0, theta_h = 0/180, a single doublet, a
            # powder, scalar components, or COMPLEX_VOIGT_METHOD == 'off') the
            # Gram reduces to expm(-Re Smat) and this readout coincides with the
            # dispersion-truncated (Hermitian) one; otherwise they differ -- the
            # nuclear Faraday rotation -- and the difference survives the CMS
            # half-trace as well.
            #
            # rho = diag(rho11, rho22). A CMS (radioactive source, Mett == 1) is
            # unpolarized -> fixed rho = I/2 (half the trace). An SMS is linearly
            # polarized to the degree ``sms_pol`` (p=1 -> pure (1,1)).
            #
            # The branch keys on ``Mett`` (the source CLASS), NOT ``Met`` (which
            # also encodes the source-line shape and the recursion sentinel).
            # ``Mett == 1`` is the sole unpolarized (CMS) source; every other
            # value is polarized SMS:
            #   * Mett == 0                -> SMS (the GUI's only SMS value);
            #   * Mett == 2, 3             -> currently unreachable source-line
            #                                 variants (no caller sets Met 2/3),
            #                                 SMS-type, so they read out polarized;
            #   * Met == -1 (per-energy recursion) does not touch Mett -- the
            #     parent forwards its own Mett here, so a CMS *or* SMS parent is
            #     classified correctly inside the recursion.
            if Mett == 1:
                rho11 = rho22 = 0.5
            else:
                # SMS beam linear polarization degree, passed in from TI (which
                # forwards the GUI value into the pool workers). Default SMS_POL_DEFAULT.
                pol = sms_pol
                rho11 = 0.5 * (1.0 + pol)
                rho22 = 0.5 * (1.0 - pol)
            Tdag = np.conjugate(np.transpose(Tprod, (0, 2, 1)))   # per-energy Hermitian conjugate
            Gram = np.matmul(Tdag, Tprod)                         # Gram = Tprod^H Tprod (Hermitian, PSD)
            CH = CH * (rho11 * np.real(Gram[:, 0, 0]) + rho22 * np.real(Gram[:, 1, 1]))
        if Met == 0:
            CH = CH * N * 2 / (1 - (EE) ** 2)
        elif Met == 1 or Met == 2 or Met == 3:
            CH = CH * N * (cof[0] * 2 / (1 - (EE) ** 2) \
                         + cof[1] * 4 / (4 - (EE) ** 2) \
                         + cof[2] * 6 / (9 - (EE) ** 2))
            # CH = CH * N * (cof[0] * (1/4) * (np.tan(np.abs(np.pi / 2 * EE))) ** ((1/4) - 1 + 1*(EE==0)) / (np.cos(np.pi / 2 * EE)) ** 2 * np.pi / 2\
            #              + cof[1] * (1/2) * (np.tan(np.abs(np.pi / 2 * EE))) ** ((1/2) - 1 + 1*(EE==0)) / (np.cos(np.pi / 2 * EE)) ** 2 * np.pi / 2\
            #              + cof[2] * 1     * (np.tan(np.abs(np.pi / 2 * EE))) ** (1     - 1            ) / (np.cos(np.pi / 2 * EE)) ** 2 * np.pi / 2 * (1 - (EE==0)*(1-2/np.pi/cof[2])) \
            #              + cof[3] * 2     * (np.tan(np.abs(np.pi / 2 * EE))) ** (2     - 1            ) / (np.cos(np.pi / 2 * EE)) ** 2 * np.pi / 2\
            #              + cof[4] * 3     * (np.tan(np.abs(np.pi / 2 * EE))) ** (3     - 1            ) / (np.cos(np.pi / 2 * EE)) ** 2 * np.pi / 2)
        return(CH)


# Set by "! INTERRUPT !" (PhysicsApp.interrupt, where it is ``fit_cancel``).
# Every calculation that uses the pool goes through TI, so checking it there
# stops the fit, the sequential fit, Show model, calibration and both
# instrumental-function searches.
FIT_CANCEL = threading.Event()


class FitInterrupted(Exception):
    """Raised in a calculation thread when the user presses "! INTERRUPT !"."""

    def __init__(self, message="Interrupted by the user"):
        super().__init__(message)


def _pool_starmap(pool, func, args):
    """``pool.starmap`` that gives up as soon as FIT_CANCEL is set.

    Same work and same result as ``pool.starmap`` (which is itself
    ``starmap_async(...).get()``), but waited for in short slices: a plain
    starmap waits for ever once Interrupt has terminated the pool, leaving the
    calculation thread stuck.
    """
    if FIT_CANCEL.is_set():
        raise FitInterrupted()
    if not hasattr(pool, 'starmap_async'):      # serial stand-ins in the tests
        return pool.starmap(func, args)
    result = pool.starmap_async(func, args)
    while not result.ready():
        # a terminated pool never completes the result, flag or not
        if FIT_CANCEL.is_set() or getattr(pool, '_state', None) == 'TERMINATE':
            raise FitInterrupted()
        result.wait(0.05)
    return result.get()


def TI(x_exp, p, model, JN, pool, x0, MulCo, INS, Distri=[0], Cor = [0], Met=0, Norm = 1, pol=SMS_POL_DEFAULT, Recon=[0], lengths=None):  # num - number of Gausians # PS - spc, p - InsFun
    """Compute the Mossbauer transmission spectrum (full transmission integral).

    Integrates the per-energy model ``TImod`` over the source line shape using
    ``pool`` (multiprocessing) and adds the polynomial baseline. ``model`` is the
    list of component names, ``p`` the flat parameter array, ``JN`` the number of
    integration samples and ``INS`` the instrumental-function parameters.

    ``Norm`` is the same integration sum for an empty model
    (instrumental_io.compute_norm, with the same JN, grid and source): the part
    of the unit-area source line that the grid samples. CMS and SMS use it
    differently -- see "SOURCE NORMALISATION" in the body.

    For a model with Nbaseline sections ``x_exp`` holds the velocities of every
    spectrum, one spectrum after the other. ``lengths`` gives the number of
    points of each of them. Without it the spectra are found from the velocity
    steps -- a new one starts wherever a step does not have the sign of the very
    first step -- which cuts wrongly spectra running in opposite directions or
    with a repeated velocity.

    Core physics entry point: used outside this module by Calibration.py,
    fitting_io.py and syncmoss_main.py to simulate and fit every spectrum.

    Returns:
        numpy.ndarray: model intensity sampled at the experimental points ``x_exp``.
    """
    # ``pol`` is the SMS beam linear polarization degree (0..1, default SMS_POL_DEFAULT). It
    # travels to every ``TImod`` worker as the positional ``sms_pol`` argument
    # (right after ``Met`` in the tuples below) so it is pickled through to the
    # spawned pool workers -- they re-import this module fresh and cannot see a
    # value that was only set in the main process, so it MUST travel as an argument.

    # INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
    # Per-section instrumental parameters: for an Nbaseline model, x0, MulCo, INS,
    # Met and Norm may each be a list/tuple/array with one entry per section, so a
    # CMS spectrum and an SMS spectrum can be computed together (each section then
    # gets its own source-line grid E, since E depends on Met). A list ``Met`` is
    # the unambiguous signal for this mode (Met is otherwise always a scalar int);
    # passing scalars reproduces the original single-method behaviour exactly.
    per_section = isinstance(Met, (list, tuple, np.ndarray))
    Met_repr = Met[0] if per_section else Met
    E = np.linspace(-1 + (10 ** -2)*(Met_repr == 1 or Met_repr ==2) + 10**-3, 1 - (10 ** -2)*(Met_repr == 1 or Met_repr ==2) - 10**-3, JN)

    D = (E[1] - E[0])
    if model.count('Nbaseline') == 0:
        # Recon travels after the default-valued middle args (Mett,O,Di,Co,V,
        # return_layer_matrix) so the parallel weight list reaches every TImod
        # worker positionally (starmap cannot pass keywords).
        H = _pool_starmap(pool, TImod, [(x_exp, p, model, Ex, x0, MulCo, INS, Distri, Cor, Met, pol, -2, [], 0, 0, number_of_baseline_parameters, False, Recon) for Ex in E])
        H = np.array(H, dtype=object).sum(axis=0)

        # Ht = np.array([[float(0)] * len(x_exp)] * JN)
        # for i in range(0, JN):
        #     # print('Number', i)
        #     Ht[i] = np.array((TImod(x_exp, p, model, E[i], x0, MulCo, INS, Distri, Cor, Met)))
        # H = Ht.sum(axis=0)

        N0 =                     (p[0] + p[3] * p[0]/10**2 * x_exp + p[2] * p[0] / 10 ** 4 * ((-1) * p[1] + x_exp) ** 2)
        N1 =                     (p[4] + p[7] * p[4]/10**2 * x_exp + p[6] * p[4] / 10 ** 4 * ((-1) * p[5] + x_exp) ** 2)

        # SOURCE NORMALISATION. This belongs to HOW the integral above is done:
        # if that changes, revisit it (see "WHEN THE CMS INTEGRATION CHANGES").
        #
        # H*D is the node sum  Q[S*T] = sum_k w_k T(v + u_k)  over the source
        # offsets u_k of the grid E (w_k = source density * Jacobian * D, formed
        # in TImod), and Norm is the same sum for an empty model (T = 1), i.e.
        # Q[S]: the part of the unit-area source line S that the grid samples.
        # The exact count rate is  N0 * int S(u) T(v + u) du + N1  over ALL u,
        # and what to do with 1 - Norm depends on WHY Norm is not 1.
        #
        # CMS (Met == 1). The source is a Voigt with the natural Lorentzian,
        # whose wings fall only as 1/u^2, and the Met == 1 grid (the three-log
        # map of TImod, e = +-0.989, MulCoCMS = 0.28) ends at +-4.716 mm/s. About
        # 0.6 % of the source lies beyond: 0.65 % analytically for G ~ 0.1 mm/s,
        # 1 - Norm = 0.60 % at JN = 64 because the end nodes take in part of it.
        # Those photons are real and, being far from almost every absorber line,
        # almost all TRANSMITTED, so they are added as unabsorbed:
        #     N0 * (Q[S*T] + (1 - Norm)) + N1.
        # Dividing by Norm (the former CMS form, still the SMS one) gave them to
        # the sampled core, where they were absorbed like core photons: every
        # CMS line came out 1/Norm ~ 0.6 % too deep. Still neglected: the far
        # wing photons that ARE absorbed, by a line more than 4.7 mm/s from the
        # source energy -- up to a few 1e-4 N0, more for thicker absorbers.
        # The new form is exactly  Norm * (former) + (1 - Norm):  with Nnr free a
        # fit gives the same chi2 and line parameters and only Ns and Nnr move
        # (Ns + Nnr stays, the resonant fraction Ns/(Ns + Nnr) grows by 1/Norm);
        # with the CMS default Nnr = =[0,0.67] the former results are the new
        # ones with Nnr = 0.66 Ns.
        #
        # SMS (Met == 0). The window (INSint.txt, set by `limits`) holds the
        # whole source: Gaussians have no wings, the theoretical shape falls as
        # E^-4 (~4e-5 of it outside). Norm - 1 is then the sum's OWN quadrature
        # error, of either sign (theoretical shape, JN = 32: Norm = 1.0008), and
        # dividing removes it. 1 - Norm would be a negative number of missing
        # photons: an absorber black across the window would give N1 - 8e-4 N0,
        # below the non-resonant level. Met 2/3 (not reachable from the GUI) keep
        # the division too: Met 2's amplitudes are not normalised, and dividing
        # is what normalises them.
        #
        # Both cases are  N0 * (W * Q[S*T]/Norm + (1 - W)) + N1,  W = the true
        # area of S inside the window: W = 1 for SMS, and for CMS W is taken as
        # Norm itself, the grid's own measure of what it samples.
        #
        # WHEN THE CMS INTEGRATION CHANGES:
        # * another map, edge or JN: nothing to do, PROVIDED Norm comes from
        #   compute_norm on the same grid, JN and G -- the term follows by itself;
        # * a grid reaching far into the wings: 1 - Norm -> 0, the forms meet;
        # * integrating over the ABSORBER energy instead (T once on one uniform
        #   grid, the source evaluated analytically at every offset): the source
        #   then has no window and there is no Norm at all,
        #     C = N0 * (1 - sum_m S(E_m - v) * h * (1 - T(E_m))) + N1;
        # * a CMS source that is not unit-area, or a window that holds all of
        #   it: back to the division (or the W form above).
        # Every caller passes the Norm of the G it integrates with. The CMS
        # instrumental-function search fits G, so it recomputes Norm whenever G
        # changes (instrumental_io.cms_norm_following_g).
        if Met_repr == 1:
            Hc = N0 * (H * D + (1 - Norm)) + N1
        else:
            Hc = H * N0 / Norm * D + N1
    else:
        Di, Co, V, MV = 0, 0, 0, 0
        Re = 0
        Hc = []
        if lengths is not None:
            # Every spectrum's own number of points: nothing to guess
            if len(lengths) != model.count('Nbaseline') + 1 or sum(lengths) != len(x_exp):
                raise ValueError(f"spectra of {list(lengths)} points do not match the "
                                 f"{model.count('Nbaseline') + 1} sections of the model and "
                                 f"the {len(x_exp)} velocities given")
            x_separate = []
            start = 0
            for length in lengths:
                x_separate.append(x_exp[start:start + length])
                start += length
        else:
            step_sign = np.sign(x_exp[1]-x_exp[0])
            x_separate = []
            start = 0
            Num_x = 0
            for i in range(1, len(x_exp)):
                if step_sign != np.sign(x_exp[i]-x_exp[i-1]):
                    x_separate.append(x_exp[start:i])
                    start = i
                    Num_x += 1
            x_separate.append(x_exp[start:])
        model_separate = []
        startM = 0
        Num_m = 0
        for i in range(0, len(model)):
            if model[i] == 'Nbaseline':
                model_separate.append(model[startM:i])
                startM = i+1
                Num_m += 1
        model_separate.append(model[startM:])

        for i in range(0, model.count('Nbaseline')+1):
            # Instrumental parameters (and the matching source-line grid) for this
            # section: per-section values when lists were passed, else the shared
            # scalars — in which case E/D keep the values computed above.
            if per_section:
                x0_i, MulCo_i, INS_i, Met_i, Norm_i = x0[i], MulCo[i], INS[i], Met[i], Norm[i]
                E = np.linspace(-1 + (10 ** -2)*(Met_i == 1 or Met_i ==2) + 10**-3, 1 - (10 ** -2)*(Met_i == 1 or Met_i ==2) - 10**-3, JN)
                D = (E[1] - E[0])
            else:
                x0_i, MulCo_i, INS_i, Met_i, Norm_i = x0, MulCo, INS, Met, Norm
            N0 = (p[V]   + p[V+3] * p[V]  /10**2 * x_separate[i] + p[V+2] * p[V]   / 10 ** 4 * ((-1) * p[V+1] + x_separate[i]) ** 2)
            N1 =  p[V+4] + p[V+7] * p[V+4]/10**2 * x_separate[i] + p[V+6] * p[V+4] / 10 ** 4 * ((-1) * p[V+5] + x_separate[i]) ** 2
            V = V + number_of_baseline_parameters
            H = _pool_starmap(pool, TImod, [(x_separate[i], p, model_separate[i], Ex, x0_i, MulCo_i, INS_i, Distri, Cor, Met_i, pol, -2, [], Di, Co, V, False, Recon, Re) for Ex in E])
            # Di = H[0][1]
            # Co = H[0][2]
            # V = H[0][3]
            H = np.array(H, dtype=object).sum(axis=0)
            # per section, the SOURCE NORMALISATION of the single-spectrum branch:
            # CMS adds the unsampled source photons as unabsorbed, SMS divides
            if Met_i == 1:
                H = N0 * (H * D + (1 - Norm_i)) + N1
            else:
                H = H * N0 / Norm_i * D + N1
            Hc = np.concatenate((Hc, H))

            for j in range(MV, len(model)):
                MV += 1
                V += int(4 * (model[j] == 'Singlet') + 9 * (model[j] == 'Doublet') + 14 * (model[j] == 'Sextet') + 14 * (model[j] == 'Sextet(rough)') + 17 * (model[j] == 'MDGD')\
                    + 14 * (model[j] == 'Relax_2S') + 11 * (model[j] == 'Average_H') + 11 * (model[j] == 'Relax_MS') + 15*(model[j]=='ASM') + 27*(model[j]=='SCDW')\
                    + 15 * (model[j] == 'Hamiltonian') + 12 * (model[j] == 'Hamilton_mc') + 9 * (model[j] == 'Hamilton_pc')\
                    + 5 * (model[j] == 'Distr') + 2 * (model[j] == 'Corr') + 7 * (model[j] == 'Recon') \
                    + numco * (model[j] == 'Variables') + 1*(model[j] =='Expression')) # + number_of_baseline_parameters * (model[j] == 'Nbaseline')
                # print('V is equal to ', V)
                if model[j] == 'Distr':
                    Di += 1
                if model[j] == 'Corr':
                    Co += 1
                if model[j] == 'Recon':
                    Re += 1
                if model[j] == 'Nbaseline':
                    break
            # print('finally V is equal to ', V)

    return Hc



