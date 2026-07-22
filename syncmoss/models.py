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
from numba import njit, prange
import scipy
import scipy.linalg
dummy = scipy.linalg.eig(np.array([[1,0], [0,1]])) #required to build exe
from syncmoss.constants import number_of_baseline_parameters, numco
from numpy.linalg import eig
from numpy import linalg as LA
# from numpy.linalg import inv
from numpy import abs
# import matplotlib.pyplot as plt

G = 4.7 * 10 ** -9  # natural width in eV*10**-9 # 4.7 value from Ralf Rohlsberger
# Flm = 0.4                  # Lamb Mossbauer factor for source
E0 = 14412  # energy of resonance
E0_J = E0 * 1.602176634 * 10**-19
c = 2.99792458 * 10 ** 11  # speed of light at mm/s
d = 0.005  # area density
# ro = 7880                # density
Fa = 0.54676  # fraction of resonance absorption
etto = 1  # percent of 57Fe
sigma = 2.464 * 10 ** -22  # max resonance cross section
mun = 5.050783699*10**-27
ggr = 0.18121
gex = -0.10353

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
            # iINS += INS[i*3+2]**2*1/2*(INS[i*3]**2+G/E0*c/2)*np.sqrt(np.pi)*(1-erf((R - INS[i*3+1])/(INS[i*3]**2+G/E0*c/2)))
            iINS += INS[i * 3 + 2] ** 2 * (
                        1 - erf((R - INS[i * 3 + 1]) / np.sqrt(2) / (INS[i * 3] ** 2 + G / E0 * c / 2))) / 2
        return iINS

    def integral_INS_m(R):
        iINS = 0
        for i in range(0, int(len(INS) / 3)):
            # iINS += INS[i*3+2]**2*1/2*(INS[i*3]**2+G/E0*c/2)*np.sqrt(np.pi)*(1+erf((R - INS[i*3+1])/(INS[i*3]**2+G/E0*c/2)))
            iINS += INS[i * 3 + 2] ** 2 * (
                        1 + erf((R - INS[i * 3 + 1]) / np.sqrt(2) / (INS[i * 3] ** 2 + G / E0 * c / 2))) / 2
        return iINS

    sp_l = np.linspace(0, -5, 4096)
    sp_r = np.linspace(0, 5, 4096)
    sp_int_l = np.array([float(0)] * len(sp_l))
    sp_int_r = np.array([float(0)] * len(sp_r))
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
    p = [1000000, 0, 0, 0, 0, 0, 0, 0, 15, -3, 0.098, 0, 10, 0, 0.098, 0, 20, 3, 0.098, 0.6]
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
# 'Relax_MS', 'Relax_2S', 'Hamilton_mc', 'ASM' -- fills the cross-section
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
# assignment of the sigma+- matrices. In a total-intensity spectrum this sign
# is exactly degenerate with theta_h -> 180 - theta_h (a global complex
# conjugation of the exponent leaves every diagonal of T^H T unchanged), so it
# must be CALIBRATED once on a spectrum of known geometry, not fitted. All
# complex line shapes honour this single switch; the numba-compiled ones
# (Blume_c, _relax_MS_groups_c) bake it in at their first compilation, so set
# it BEFORE the first call.
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
    for i in range(0, 8):
        g0[i] = (np.sqrt(1 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4]) + Vex[0][
            i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4])) \
                    * (1 / 4) * np.sqrt(3 / np.pi)

        g1[i] = (np.sqrt(2 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + np.sqrt(2 / 3) *
                    Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                    * (1/4) * np.sqrt(3*2/np.pi)

        g2[i] = (np.sqrt(1 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + Vex[3][
            i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
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
    """
    g0, g1, g2, S = _ham_mono_core(Q, Hhf, etto, phi, tet)

    phir = phir / 180 * np.pi
    tetr = tetr / 180 * np.pi
    F2 = np.sqrt(2) * np.sin(tetr) * (-1j) * np.exp(1j * (phir))
    F4 = np.sqrt(2) * np.cos(tetr) * (1j)
    F6 = np.sqrt(2) * np.sin(tetr) * (1j) * np.exp(-1j * (phir))

    E = g0 * F2 + g1 * F4 + g2 * F6
    I = np.real(E * np.conjugate(E)) * np.pi

    return (I, S)

@njit(cache=True)
def Ham_mono_CMS(Q, Hhf, etto, phi, tet, phir, tetr):
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
#     S2 = (-1) * (Sig - 3.0760 / 5.3123 * H / 2 - eps) * MulCo + X
#     S3 = (-1) * (Sig - 0.8397 / 5.3123 * H / 2 - eps) * MulCo + X
#     S4 = (-1) * (Sig + 0.8397 / 5.3123 * H / 2 - eps) * MulCo + X
#     S5 = (-1) * (Sig + 3.0760 / 5.3123 * H / 2 - eps) * MulCo + X
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
    S2 = (-1) * (Sig - 3.0760 / 5.3123 * H / 2 - eps) * MulCo + X
    S3 = (-1) * (Sig - 0.8397 / 5.3123 * H / 2 - eps) * MulCo + X
    S4 = (-1) * (Sig + 0.8397 / 5.3123 * H / 2 - eps) * MulCo + X
    S5 = (-1) * (Sig + 3.0760 / 5.3123 * H / 2 - eps) * MulCo + X
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
    S2 = (-1) * (Sig - 3.0760 / 5.3123 * H / 2 - eps) + X
    S3 = (-1) * (Sig - 0.8397 / 5.3123 * H / 2 - eps) + X
    S4 = (-1) * (Sig + 0.8397 / 5.3123 * H / 2 - eps) + X
    S5 = (-1) * (Sig + 3.0760 / 5.3123 * H / 2 - eps) + X
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
    M442 = (D2A(x, len(numb)).transpose(1,0) - 3.0760 / 5.3123 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)
    M443 = (D2A(x, len(numb)).transpose(1,0) - 0.8397 / 5.3123 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)

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
    M442 = (D2A(x, len(numb)).transpose(1,0) - 3.0760 / 5.3123 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)
    M443 = (D2A(x, len(numb)).transpose(1,0) - 0.8397 / 5.3123 * Hv * (S - (D2A(numb, len(x)) + 1) + 1) / S - Sig + eps)

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
#   p = 0.98 -> a realistic synchrotron (SMS) beam (DEFAULT).
#   p = 1.0  -> fully polarized: reads the (1,1) element only, i.e. the original
#               SMS behaviour.
#   p = 0.0  -> unpolarized: half the trace, the conventional radioactive-source
#               (CMS) readout.
#
# The degree p is the ``pol`` argument of ``TI`` (default 0.98), which forwards
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

    F2a = np.sqrt(2) * np.sin(tetr1) * (-1j) * np.exp(1j * phir1)
    F4a = np.sqrt(2) * np.cos(tetr1) * (1j)
    F6a = np.sqrt(2) * np.sin(tetr1) * (1j) * np.exp(-1j * phir1)
    F2b = np.sqrt(2) * np.sin(tet2) * (-1j) * np.exp(1j * phi2)
    F4b = np.sqrt(2) * np.cos(tet2) * (1j)
    F6b = np.sqrt(2) * np.sin(tet2) * (1j) * np.exp(-1j * phi2)

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

    F2a = np.sqrt(2) * np.sin(tet1) * (-1j) * np.exp(1j * phi1)
    F4a = np.sqrt(2) * np.cos(tet1) * (1j)
    F6a = np.sqrt(2) * np.sin(tet1) * (1j) * np.exp(-1j * phi1)
    F2b = np.sqrt(2) * np.sin(tet2) * (-1j) * np.exp(1j * phi2)
    F4b = np.sqrt(2) * np.cos(tet2) * (1j)
    F6b = np.sqrt(2) * np.sin(tet2) * (1j) * np.exp(-1j * phi2)

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
    |H| (isomer/quadrupole shift tracks magnitude, not sign). Signs and the
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
    # the isomer shift is a scalar (s-electron density) and the quadrupole shift is a lattice/EFG property
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


# Fixed wave-phase sampling count (positions per period) for S/C_DW. Not a fit
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
    """Grid-binned line positions + weights for the S/C_DW branch.

    Same positions as :func:`SDW_thick_terms_direct`, but each line's ``Num``
    positions are binned onto a per-line grid of step ``dg = width / steps`` so
    the caller evaluates the Voigt once per node -- the Voigt count follows
    span/width, NOT ``Num``. ``width`` = max(WL, WG) (WG is often 0, so min would
    give dg=0) floored at the natural line width (0.098), MulCo-scaled like the
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
    natural = 0.098 * MulCo          # natural Lorentzian width (WL default), MulCo-scaled
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


def TImod (x_exp, p, model, EE, x0, MulCo, INS, Distri, Cor, Met = 0, sms_pol=0.98, Mett = -2, O=[], Di=0, Co=0, V=number_of_baseline_parameters, return_layer_matrix=False):
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
            for i in range (0, int((len(INS))/3)):
                    N += 1*INS[i*3+2]**2*np.exp((-1)*((E-(INS[i*3+1]+SCR)*MulCo)**2/(2*((INS[i*3]**2+G/E0*c/2)*MulCo)**2)))/((INS[i*3]**2+G/E0*c/2)*MulCo)/np.sqrt(2*np.pi)
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

            N += Voight(0.098*MulCo, Wid, E - SCR * MulCo)
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

        Kpref = np.pi * (G / 2 / E0 * c * MulCo)
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
                CHt = CH*np.exp((-1)*np.pi*(G/2/E0*c*MulCo)*I*Voi)
            if model[i] == 'Sextet(rough)':
                I = abs(p[V])
                I1 = I * p[V + 9]  / (1 + p[V + 9] + p[V + 10]) * p[V + 11] / (1 + p[V + 11])
                I2 = I * p[V + 10] / (1 + p[V + 9] + p[V + 10]) * p[V + 12] / (1 + p[V + 12])
                I3 = I * 1         / (1 + p[V + 9] + p[V + 10]) * p[V + 13] / (1 + p[V + 13])
                I4 = I * 1         / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 13])
                I5 = I * p[V + 10] / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 12])
                I6 = I * p[V + 9]  / (1 + p[V + 9] + p[V + 10]) * 1         / (1 + p[V + 11])
                HH = p[V + 3] / 3.101
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 6] * MulCo + E
                S2 = (-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 7] * MulCo + E
                S3 = (-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 7] * MulCo + E
                S4 = (-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 7] * MulCo + E
                S5 = (-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 7] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 6] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                WG = abs(p[V + 5]) * MulCo
                GaH = abs(p[V + 8]) / 2 / 3.101 * MulCo
                Ga16 = (WG**2 + GaH**2)** (1/2)
                Ga25 = (WG**2 + (3.0760 / 5.3123 * GaH)**2)** (1/2)
                Ga34 = (WG**2 + (0.8397 / 5.3123 * GaH)**2)** (1/2)
                Voi1 = Voight(WL, Ga16, S1)
                Voi2 = Voight(WL, Ga25, S2)
                Voi3 = Voight(WL, Ga34, S3)
                Voi4 = Voight(WL, Ga34, S4)
                Voi5 = Voight(WL, Ga25, S5)
                Voi6 = Voight(WL, Ga16, S6)
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo)\
                                  * (I1*Voi1+I6*Voi6+I2*Voi2+I5*Voi5+I3*Voi3+I4*Voi4))
                V += 14
            if model[i] == 'Hamilton_pc':
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
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo)\
                                  * (I[0]*Voight(WL,WG,E-S[0])+I[1]*Voight(WL,WG,E-S[1])+I[2]*Voight(WL,WG,E-S[2])+I[3]*Voight(WL,WG,E-S[3])\
                                    +I[4]*Voight(WL,WG,E-S[4])+I[5]*Voight(WL,WG,E-S[5])+I[6]*Voight(WL,WG,E-S[6])+I[7]*Voight(WL,WG,E-S[7])))
                V += 9
            if model[i] == 'Average_H':
                I = abs(p[V])
                Sig = p[V+1] * MulCo
                Q = p[V+2] * MulCo
                Hin = p[V+3] / 3.101 * (-1) * MulCo
                WL = p[V+4] * MulCo
                WG = p[V+5] * MulCo
                Hex = p[V+6] / 3.101 * MulCo
                K = p[V+7] * MulCo
                J = p[V+8] * MulCo
                tet = p[V+9] / 180 * np.pi
                Num = max(int(p[V+10]), 1)

                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * I * Angles_min(tet, Num, K, J, Hin, Hex, E, WL, WG, Q, Sig))
                V += 11
            # --- Polarized model components ---------------------------------
            # Each adds a 2x2 cross-section matrix to Smat instead of a scalar
            # exponent; the single matrix exponential is taken after the loop.
            # The texture order parameter A blends between a random powder
            # (A = 0, every Mhat below averages to I and the component reduces
            # EXACTLY to its former scalar form) and a single crystal (A = 1).
            # The Faraday-active models (Sextet, MDGD, Relax_2S) additionally carry
            # the magnetic polar-order parameter A_m in [-1, 1] (S1 = A_m*sqrt((1+2A)
            # /3), see _texture_s1) that scales the resolved sigma+- Faraday term;
            # A_m = 0 (the default) is an unmagnetised texture, A_m = A = 1 the
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
                HH = p[V + 3] / 3.101
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 10] * MulCo + E
                S2 = (-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 11] * MulCo + E
                S3 = (-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 11] * MulCo + E
                S4 = (-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 11] * MulCo + E
                S5 = (-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 11] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 10] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                WG = abs(p[V + 5]) * MulCo
                GaH = abs(p[V + 12]) / 2 / 3.101 * MulCo
                Ga16 = (WG ** 2 + GaH ** 2) ** (1 / 2)
                Ga25 = (WG ** 2 + (3.0760 / 5.3123 * GaH) ** 2) ** (1 / 2)
                Ga34 = (WG ** 2 + (0.8397 / 5.3123 * GaH) ** 2) ** (1 / 2)
                Voi1 = Voight_c(WL, Ga16, S1)
                Voi2 = Voight_c(WL, Ga25, S2)
                Voi3 = Voight_c(WL, Ga34, S3)
                Voi4 = Voight_c(WL, Ga34, S4)
                Voi5 = Voight_c(WL, Ga25, S5)
                Voi6 = Voight_c(WL, Ga16, S6)
                Atex = p[V + 8]                   # uniaxial (fiber) texture order parameter
                Am = p[V + 9]                     # magnetic polar order A_m in [-1,1] (S1 fraction), right after A
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
                HH = p[V + 3] / 3.101
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 14] * MulCo + E
                S2 = (-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 15] * MulCo + E
                S3 = (-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 15] * MulCo + E
                S4 = (-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 15] * MulCo + E
                S5 = (-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 15] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 14] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                Guni = abs(p[V + 5]) * MulCo
                Gh = abs(p[V + 6]) * MulCo
                Gde = p[V + 7]
                Gdh = p[V + 8]
                Geh = p[V + 9]
                Cd = [1, 1, 1, 1, 1, 1]
                Ce = [1, -1, -1, -1, -1, 1]
                Ch = [-1 / 6.202, -1 / 10.71, -1 / 39.24, 1 / 39.24, 1 / 10.71, 1 / 6.202]
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
                Am = p[V + 13]                    # magnetic polar order A_m in [-1,1] (S1 fraction), right after A
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
                Hv = float(p[V + 3]) * MulCo / 2 / 3.1098
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
                H1 = p[V + 3] / (abs(ggr) + 3 * abs(gex)) * 2 * MulCo / 3.101 / 2
                Sig2 = p[V + 4] * MulCo
                Q2 = p[V + 5] / 3 * MulCo
                H2 = p[V + 6] / (abs(ggr) + 3 * abs(gex)) * 2 * MulCo / 3.101 / 2
                WL = p[V + 7] * MulCo
                Atex = p[V + 10]             # uniaxial (fiber) texture order parameter
                Am = p[V + 11]               # magnetic polar order A_m in [-1,1] (S1 fraction), right after A
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
                V += 13
            if model[i] == 'Hamilton_mc':
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
            if model[i] == 'S/C_DW':
                # Spin/charge density wave, polarized (thick). The spin AXIS is
                # FIXED at (theta_k, phi_h); only the SCALAR hyperfine parameters
                # (signed field H, isomer shift, quadrupole shift) are modulated
                # along the wave. Unlike ASM the 2x2 matrices are therefore the
                # plain Sextet ones evaluated ONCE and hoisted out of the
                # modulation loop; only the six line positions vary per point.
                #
                # The field carries its SIGN (SDW_thick_terms does NOT take
                # abs(H)): a sign change of H swaps v1<->v6 and v3<->v4 while the
                # matrix assignment ({1,4}->sigma-, {3,6}->sigma+, {2,5}->pi)
                # stays fixed, reproducing the helicity swap of a reversed moment.
                #
                # A_m is a fit parameter, EXACTLY the Sextet branch's magnetic
                # polar-order parameter: each site carries the resolved sigma+-
                # Faraday term A_m*sqrt((1+2A)/3)*i*mz*J2 (do NOT use the merged
                # _mhat_dm1_sym). Whether that term survives the modulation is set
                # by the WAVE, independently of A_m's value: for a BALANCED wave
                # (H0 = 0 and KeH = KdH = 0, pure odd harmonics) the positions pair
                # up as v1(psi+pi) = v6(psi) etc., so L1 == L6 and L3 == L4 and the
                # +/- Faraday cancels EXACTLY for any A_m. With a non-zero base
                # field H0 (the usual case) or a field-shift correlation KeH/KdH
                # the wave is offset, L1 != L6, and a real (A_m-scaled),
                # thickness-dependent Faraday signal remains. EDGE CASE: all
                # harmonics = KeH = KdH = 0 with H0 != 0 gives constant positions,
                # i.e. a single sextet reproducing the 'Sextet' branch at the same
                # (delta, eps, H0, widths, theta_k, phi_h, A, A_m, I13).
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
                # Atex, magnetic polar order A_m: resolved sigma-/sigma+ (Faraday
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

                while model[k] == 'Distr' or model[k] == 'Corr':
                    if model[k] == 'Distr':
                        # kk.append(k)
                        Dk += 1
                    if model[k] == 'Corr':
                        Ck += 1
                    k -= 1

                Vnum = int(4*(model[k]=='Singlet') + 9*(model[k]=='Doublet') + 14*(model[k]=='Sextet') + 14*(model[k]=='Sextet(rough)') + 17 * (model[k] == 'MDGD')\
                           + 11*(model[k]=='Relax_MS') + numco*(model[k]=='Variables') + 11*(model[k]=='Average_H') + 15*(model[k]=='ASM') + 27*(model[k]=='S/C_DW')\
                           + 14*(model[k]=='Relax_2S')) + 12*(model[k]=='Hamilton_mc') + 9*(model[k]=='Hamilton_pc') + 1*(model[k]=='Expression')

                model_d = np.array([model[k:i]] * Num).flatten()
                # print('distr model', model_d, str(Distri[Di]))
                # print(Vnum, k, i)

                pN = np.reshape(np.ravel(np.array([p[V-Vnum-Ck*2-Dk*5:V]]*Num), order='F'), (Vnum+Ck*2+Dk*5, Num))

                pN[0] = ge * p[V-Vnum-Ck*2-Dk*5]

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
                pN = np.reshape(np.ravel(pN, order='F'), (Num, Vnum+Ck*2+Dk*5)).flatten()

                MultiDistr = 0
                for j in range(i+1, len(model)):
                    if model[j] != 'Corr':
                        # print(str('it is MULTIDIMENTIONAL!')*(model[j] == 'Distr') + str('it is single!')*(model[j] != 'Distr'))
                        MultiDistr = 1*(model[j] == 'Distr')
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
                    if len(O)==0 and Dk!=0:
                        O = p
                    # CHt = CH * (TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[Di-Dk:Di]]*Num).flatten(), np.array([Cor[Co-Ck:Co]]*Num).flatten(), Met = -1, Mett = Mett, O=O))
                    mDk = 0
                    mCk = 0
                    for mk in range(0, k):
                        if model[mk] == 'Distr':
                            mDk += 1
                        if model[mk] == 'Corr':
                            mCk += 1
                    CH_in, Smat_in = TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[mDk:mDk+Dk]]*Num).flatten(), np.array([Cor[mCk:mCk+Ck]]*Num).flatten(), Met = -1, Mett = Mett, O=O, return_layer_matrix=True, sms_pol=sms_pol)
                    CHt = CH * CH_in                                  # scalar (thin) part multiplies in, as before
                    if Smat_in is not None:                          # thick part joins the CURRENT layer's Smat
                        Smat_t = Smat_in if Smat is None else Smat + Smat_in
                else:
                    CHt = CH
                Di += 1
                # print('Distri proceed')

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
                # forwards the GUI value into the pool workers). Default 0.98.
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


def TI(x_exp, p, model, JN, pool, x0, MulCo, INS, Distri=[0], Cor = [0], Met=0, Norm = 1, pol=0.98):  # num - number of Gausians # PS - spc, p - InsFun
    """Compute the Mossbauer transmission spectrum (full transmission integral).

    Integrates the per-energy model ``TImod`` over the source line shape using
    ``pool`` (multiprocessing) and adds the polynomial baseline. ``model`` is the
    list of component names, ``p`` the flat parameter array, ``JN`` the number of
    integration samples and ``INS`` the instrumental-function parameters.

    Core physics entry point: used outside this module by Calibration.py,
    fitting_io.py and syncmoss_main.py to simulate and fit every spectrum.

    Returns:
        numpy.ndarray: model intensity sampled at the experimental points ``x_exp``.
    """
    # ``pol`` is the SMS beam linear polarization degree (0..1, default 0.98). It
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
        H = pool.starmap(TImod, [(x_exp, p, model, Ex, x0, MulCo, INS, Distri, Cor, Met, pol) for Ex in E])
        H = np.array(H, dtype=object).sum(axis=0)

        # Ht = np.array([[float(0)] * len(x_exp)] * JN)
        # for i in range(0, JN):
        #     # print('Number', i)
        #     Ht[i] = np.array((TImod(x_exp, p, model, E[i], x0, MulCo, INS, Distri, Cor, Met)))
        # H = Ht.sum(axis=0)

        N0 =                     (p[0] + p[3] * p[0]/10**2 * x_exp + p[2] * p[0] / 10 ** 4 * ((-1) * p[1] + x_exp) ** 2)
        Hc = H * N0 / Norm * D + (p[4] + p[7] * p[4]/10**2 * x_exp + p[6] * p[4] / 10 ** 4 * ((-1) * p[5] + x_exp) ** 2)
    else:
        Di, Co, V, MV = 0, 0, 0, 0
        Hc = []
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
            H = pool.starmap(TImod, [(x_separate[i], p, model_separate[i], Ex, x0_i, MulCo_i, INS_i, Distri, Cor, Met_i, pol, -2, [], Di, Co, V) for Ex in E])
            # Di = H[0][1]
            # Co = H[0][2]
            # V = H[0][3]
            H = np.array(H, dtype=object).sum(axis=0)
            H = H * N0 / Norm_i * D + N1
            Hc = np.concatenate((Hc, H))

            for j in range(MV, len(model)):
                MV += 1
                V += int(4 * (model[j] == 'Singlet') + 9 * (model[j] == 'Doublet') + 14 * (model[j] == 'Sextet') + 14 * (model[j] == 'Sextet(rough)') + 17 * (model[j] == 'MDGD')\
                    + 14 * (model[j] == 'Relax_2S') + 11 * (model[j] == 'Average_H') + 11 * (model[j] == 'Relax_MS') + 15*(model[j]=='ASM') + 27*(model[j]=='S/C_DW')\
                    + 12 * (model[j] == 'Hamilton_mc') + 9 * (model[j] == 'Hamilton_pc')\
                    + 5 * (model[j] == 'Distr') + 2 * (model[j] == 'Corr') \
                    + numco * (model[j] == 'Variables') + 1*(model[j] =='Expression')) # + number_of_baseline_parameters * (model[j] == 'Nbaseline')
                # print('V is equal to ', V)
                if model[j] == 'Distr':
                    Di += 1
                if model[j] == 'Corr':
                    Co += 1
                if model[j] == 'Nbaseline':
                    break
            # print('finally V is equal to ', V)

    return Hc



