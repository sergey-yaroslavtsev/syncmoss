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
import scipy
import builtins as bu
def max(*args):
    return bu.max(*args)
# import dual_v3 as dn
import syncmoss.minimi_lib as mi
import syncmoss.sms_theory as smst
import os
import platform
import time
from numba import njit, prange

from numpy import linalg as LA
# from numpy.linalg import inv
from numpy import abs
from syncmoss.constants import number_of_baseline_parameters, numco
# 57Fe physics constants (see constants.py for values, sources and the note on
# how `mun` is derived from the alpha-Fe standard). Imported into this module's
# namespace because the njit kernels below read them as globals; the markers
# drawn from here therefore share one definition with the spectra in models.py.
from syncmoss.constants import (
    E0_J, c, ggr, gex, mun, TESLA_PER_MMS, LINE_RATIO_25, LINE_RATIO_34)
# import matplotlib.pyplot as plt

T = 1




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
# def Voigt(WL, WG, S):
#     W = np.power(WG**5 + 2.69269*WL*WG**4 + 2.42843*WL**2*WG**3 + 4.47163*WL**3*WG**2 + 0.07842*WL**4*WG + WL**5, 1/5)
#     et = 1.36603 * WL / W - 0.47719 * WL ** 2 / W ** 2 + 0.11116 * WL ** 3 / W ** 3
#     Voi = (et *  W / 2 / np.pi / (S ** 2 + (W / 2) ** 2) \
#         + (1 - et) *  (np.exp((-1) * (S ** 2 / (2 * (W / 2 / np.sqrt(2 * np.log(2))) ** 2))) / (W / 2 / np.sqrt(2 * np.log(2))) / np.sqrt(2 * np.pi)))
#     return(Voi)
#
# def Voigt(WL, WG, S):
#     d = (WL-WG)/(WL+WG)
#     W = (WL+WG)*(1 - 0.18121*(1-d**2)-(0.023665*np.exp(0.6*d)+0.00418*np.exp(-1.9*d))*np.sin(np.pi*d))
#     cl = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
#     cg = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
#     Voi = (cl *  W / 2 / np.pi / (S ** 2 + (W / 2) ** 2) \
#         + cg *  (np.exp((-1) * (S ** 2 / (2 * (W / 2 / np.sqrt(2 * np.log(2))) ** 2))) / (W / 2 / np.sqrt(2 * np.log(2))) / np.sqrt(2 * np.pi)))
#     return(Voi)

@njit(cache=True)
def Voigt(gL, gG, S): # doi.org/10.1107/S0021889800010219
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

@njit(cache=True)
def Ham_positions(Q, Hhf, etto, phi, tet):
    """Line positions S (mm/s) of the Hamiltonian components, for the markers.

    Local copy of models._ham_tensor_core up to where the positions are known:
    the four excited levels as the roots of the quartic of Szymanski (2026),
    eq. (10), solved as in its Appendix A, then eq. (28) -- no intensities, no
    eigensolver. The lines in the order of the model: ground level k // 4 (the
    lower first), excited level k % 4 ascending.
    """
    b = Hhf * gex * mun * c / E0_J              # excited and ground Zeeman, mm/s
    bg = Hhf * ggr * mun * c / E0_J
    # the levels in units of a power of two near max(|Q|, |b|): exact, and it
    # keeps p^3 and Z^3, sixth powers of that scale, inside the float range
    sc = np.abs(Q) if np.abs(Q) > np.abs(b) else np.abs(b)
    sc = 2.0 ** np.floor(np.log2(sc)) if sc > 0.0 else 1.0
    Q = Q / sc
    b = b / sc
    t = tet / 180 * np.pi
    f = phi / 180 * np.pi
    m = np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])
    Vd = np.array([-(1 - etto) / 2 * Q, -(1 + etto) / 2 * Q, Q])
    Vm = Vd * m
    V2m = Vd * Vm
    sm = np.sum(m * Vm)
    Sm = np.sum(m * V2m)
    D2 = Q * Q * (3 + etto * etto) / 12
    # eq. (10): the excited levels are the roots of lambda^4 + p lambda^2 + q lambda + r
    p = -(5 * b * b + D2) / 2
    q = -b * b * sm
    r = (9 * b ** 4 + D2 * D2 + b * b * (6 * D2 - 4 * Sm)) / 16
    S = np.zeros(8)
    if p == 0.0:                       # Q = H = 0: one line
        return S
    # Appendix A; the clips only absorb rounding at a double root
    Z = np.sqrt(p * p + 12 * r)
    Y = (2 * p ** 3 + 27 * q * q - 72 * p * r) / (2 * Z ** 3)
    if Y > 1.0:
        Y = 1.0
    if Y < -1.0:
        Y = -1.0
    X = -p / 6 + Z / 6 * np.cos(np.arccos(Y) / 3)
    xi1 = np.sqrt(X)
    xi2 = q / (4 * xi1)
    xi3 = -(X + p / 2)
    xi4 = np.sqrt(xi3 + xi2) if xi3 + xi2 > 0.0 else 0.0
    xi5 = np.sqrt(xi3 - xi2) if xi3 - xi2 > 0.0 else 0.0
    lam = np.sort(np.array([xi1 + xi5, xi1 - xi5, -xi1 + xi4, -xi1 - xi4]))
    for jb in range(2):
        beta = 1.0 - 2.0 * jb
        if bg > 0.0:                    # the lower ground level first
            beta = -beta
        for a in range(4):
            S[4 * jb + a] = sc * lam[a] - beta * bg / 2
    return S

@njit(cache=True)
def relax_MS(S, x, I, Sig, eps, Hv, W, Ah, R, alfa):

    numb = np.linspace(0, 2 * S, int(2 * S + 1))#, dtype=int)
    numb2 = np.linspace(1, 2 * S + 1, int(2 * S + 1))#, dtype=int)

    # M11 = (-1)*R*(S*(S+1)-((numb[1:]+1)-S-1)*((numb[1:]+1)-S-2))*np.exp(alfa*(np.cos(np.pi/2*(((numb[1:]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[1:]+1)-1)-S-1)/S)**2)**2))
    # M22 = (-1)*R*(S*(S+1)-((numb[:-1]+1)-S-1)*((numb[:-1]+1)-S))*np.exp(alfa*(np.cos(np.pi/2*(((numb[:-1]+1)-S-1)/S)**2)**2 - np.cos(np.pi/2*((((numb[:-1]+1)+1)-S-1)/S)**2)**2))

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

    I1 = I * 3 * (1 - Ah) / (8 - 4 * Ah)
    I2 = I * 2 * Ah / (8 - 4 * Ah)
    I3 = I * 1 * (1 - Ah) / (8 - 4 * Ah)

    tmp1 = (Weigth * V1).sum(axis=1)
    tmp2 = (Weigth * V2).sum(axis=1)
    tmp3 = (Weigth * V3).sum(axis=1)

    return(2 * tmp1 * I1 + 2 * tmp2 * I2 + 2 * tmp3 * I3)


@njit(cache=True)
def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R): # based on eq. 3.5 (p.450) from https://doi.org/10.1103/PhysRev.165.446
    alf = (ggr * m0 - gex * m1) * (H2-H1)/2 + (Q2-Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (Sig2-Sig1)/2
    a = V - (Sig1+Sig2)/2 - (Q2+Q1)/2 * (3 * m1 ** 2 - 15 / 4) - (ggr * m0 - gex * m1) * (H2+H1)/2
    Line = np.array([float(0)]*len(V))
    for i in range(0, len(V)):
        a1 = 1j*(a[i]+alf) + L/2 + W
        a2 = -R*W
        a3 = -W
        a4 = 1j*(a[i]-alf) + L/2 + R*W
        delimeter = 1/(a1*a4-a2*a3)
        b1 = np.real(a4 *  delimeter)
        b2 = np.real(-a2 * delimeter)
        b3 = np.real(-a3 * delimeter)
        b4 = np.real(a1 *  delimeter)
        Line[i] = (b1+b3)*R/(R+1) + (b2+b4)/(R+1)
    return(Line/np.pi)




def pos_ac (p, model, INS, Met = 0, V=number_of_baseline_parameters):

        ZERO = 0
        SET = []

        if Met == 0:
            # Apparent displacement of every absorber line by the multi-line
            # source: the first moment of the instrumental function. For the
            # legacy sum of unit-area Gaussians with sum amp^2 = 1 that is the
            # sum below; the theoretical shapes compute it in sms_theory.
            if smst.ins_kind(INS) == smst.KIND_GAUSS:
                for i in range (0, int((len(INS))/3)):
                    ZERO +=  INS[i*3+1]*INS[i*3+2]**2
            else:
                ZERO = smst.ins_centroid(INS)


        for i in range (0, len(model)):
            if model[i] == 'Singlet':
                S = p[V + 1]
                V += 4
                SET.append([S])
            if model[i] == 'Spectrum':
                # One marker at the reference's shift δ (its lines are not here)
                S = p[V + 1]
                V += 4
                SET.append([S])
            if model[i] == 'Hamilton_pc':
                # DEPRECATED, superseded by 'Hamiltonian' (see models.py).
                delt = p[V+1]
                Q = p[V+2]
                H = p[V+3]
                eto = p[V+6]
                tet = p[V+7]
                phi = p[V+8]
                S = Ham_positions(Q, H, eto, phi, tet)
                S += delt
                SET.append(S)
                V += 9
            # --- Polarized model components: same line POSITIONS as before the
            # thin/thick merge (polarization changes intensities, not positions);
            # only the parameter indices shifted because A -> (theta_k, phi_h). ---
            if model[i] == 'Doublet':
                S1 = (p[V + 1] - p[V + 2])
                S2 = (p[V + 1] + p[V + 2])
                SET.append([S1, S2])
                V += 9
            if model[i] == 'Sextet':
                HH = p[V + 3] / TESLA_PER_MMS
                S1 = (p[V + 1] - HH / 2 + p[V + 2]) + p[V + 10]
                S2 = (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2]) - p[V + 11]
                S3 = (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2]) + p[V + 11]
                S4 = (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2]) - p[V + 11]
                S5 = (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2]) + p[V + 11]
                S6 = (p[V + 1] + HH / 2 + p[V + 2]) - p[V + 10]
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 14
            if model[i] == 'MDGD':
                HH = p[V + 3] / TESLA_PER_MMS
                S1 = (p[V + 1] - HH / 2 + p[V + 2]) + p[V + 14]
                S2 = (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2]) - p[V + 15]
                S3 = (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2]) + p[V + 15]
                S4 = (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2]) - p[V + 15]
                S5 = (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2]) + p[V + 15]
                S6 = (p[V + 1] + HH / 2 + p[V + 2]) - p[V + 14]
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 17
            if model[i] == 'Relax_MS':
                HH = float(p[V + 3]) / TESLA_PER_MMS
                S1 = (p[V + 1] - HH / 2 + p[V + 2])
                S2 = (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2])
                S3 = (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2])
                S4 = (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2])
                S5 = (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2])
                S6 = (p[V + 1] + HH / 2 + p[V + 2])
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 11
            if model[i] == 'Relax_2S':
                HH = float(p[V + 3]) / TESLA_PER_MMS
                S1 = (p[V + 1] - HH / 2 + p[V + 2])
                S2 = (p[V + 1] - LINE_RATIO_25 * HH / 2 - p[V + 2])
                S3 = (p[V + 1] - LINE_RATIO_34 * HH / 2 - p[V + 2])
                S4 = (p[V + 1] + LINE_RATIO_34 * HH / 2 - p[V + 2])
                S5 = (p[V + 1] + LINE_RATIO_25 * HH / 2 - p[V + 2])
                S6 = (p[V + 1] + HH / 2 + p[V + 2])
                HH2 = float(p[V + 6]) / TESLA_PER_MMS
                S12 = (p[V + 4] - HH2 / 2 + p[V + 5])
                S22 = (p[V + 4] - LINE_RATIO_25 * HH2 / 2 - p[V + 5])
                S32 = (p[V + 4] - LINE_RATIO_34 * HH2 / 2 - p[V + 5])
                S42 = (p[V + 4] + LINE_RATIO_34 * HH2 / 2 - p[V + 5])
                S52 = (p[V + 4] + LINE_RATIO_25 * HH2 / 2 - p[V + 5])
                S62 = (p[V + 4] + HH2 / 2 + p[V + 5])
                SET.append([S1, S2, S3, S4, S5, S6, S12, S22, S32, S42, S52, S62])
                V += 14
            if model[i] == 'Hamilton_mc':
                # DEPRECATED, superseded by 'Hamiltonian' (see models.py).
                delt = p[V + 1]
                Q = p[V + 2]
                H = p[V + 3]
                eto = p[V + 6]
                tet = p[V + 7]
                phi = p[V + 8]
                S = Ham_positions(Q, H, eto, phi, tet)
                S += delt
                SET.append(S)
                V += 12
            if model[i] == 'Hamiltonian':
                # Mosaic textured Hamiltonian: the ODF order parameters A, Am,
                # Ah redistribute INTENSITY between the 8 transitions, never
                # their positions, so the markers are the single-crystal ones.
                delt = p[V + 1]
                Q = p[V + 2]
                H = p[V + 3]
                eto = p[V + 6]
                tet = p[V + 7]
                phi = p[V + 8]
                S = Ham_positions(Q, H, eto, phi, tet)
                S += delt
                SET.append(S)
                V += 15
            if model[i] == 'ASM':
                Sig = p[V + 1]
                eps_m = p[V + 2]
                eps_lat = p[V + 3]
                His = p[V + 4]
                Han = p[V + 5]
                eps = eps_m + eps_lat
                H = His + Han
                a1 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0))
                a2 = eps_lat ** 2 * 3 / c * E0_J / (gex * mun * H + (H == 0))
                H = H / E0_J * c
                S1 = Sig + eps + a1 + mun * (3 * gex - ggr) / 2 * H
                S6 = Sig + eps - a1 - mun * (3 * gex - ggr) / 2 * H
                S2 = Sig - eps - a2 + mun * (gex - ggr) / 2 * H
                S5 = Sig - eps + a2 - mun * (gex - ggr) / 2 * H
                S3 = Sig - eps + a2 - mun * (gex + ggr) / 2 * H
                S4 = Sig - eps - a2 + mun * (gex + ggr) / 2 * H
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 15   # ASM: +1 for the cycloid-plane angle omega (no effect on positions)
            if model[i] == 'SCDW':
                # Representative sextet at the BASE (mean) field H0: the density
                # wave sweeps positions around this, so the markers show the
                # unmodulated sextet. Same signed-field position kernel as
                # SDW_thick_terms (with all harmonics = 0), no MulCo here.
                d0 = p[V + 1]
                eps0 = p[V + 2]
                H0 = p[V + 3]
                KdH = p[V + 10]
                KeH = p[V + 11]
                dl = d0 + KdH * H0
                ep = eps0 + KeH * H0
                H = H0 / E0_J * c
                S1 = dl + ep + mun * (3 * gex - ggr) / 2 * H
                S6 = dl + ep - mun * (3 * gex - ggr) / 2 * H
                S2 = dl - ep + mun * (gex - ggr) / 2 * H
                S5 = dl - ep - mun * (gex - ggr) / 2 * H
                S3 = dl - ep - mun * (gex + ggr) / 2 * H
                S4 = dl - ep + mun * (gex + ggr) / 2 * H
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 27
            if model[i] == 'Layer':
                SET.append([])
                V += 0

            if model[i] == 'Variables':
                SET.append([])
                V += numco
            if model[i] == 'Expression':
                SET.append([])
                V += 1
            if model[i] == 'Distr':
                del SET[-1]
                SET.append([])
                SET.append([])
                V += 5
            if model[i] == 'Corr':
                SET.append([])
                V += 2
            if model[i] == 'Distr_T':
                # The sample's thickness distribution moves no line. It opens a
                # component's sub-model (fitting_io.create_subspectra), so it adds
                # no entry: the component's own set stays the first one, which the
                # plot draws.
                V += 5
            if model[i] == 'Recon':
                # Reconstructed distribution: like Distr, the base component has no
                # single line set (it is spread over the grid), so drop its markers
                # and keep SET index-aligned; skip the 7 fixed Recon slots.
                del SET[-1]
                SET.append([])
                SET.append([])
                V += 7
            if model[i] == 'Nbaseline':
                break

        SET = np.array(SET) + ZERO

        return(SET)


def mod_pos(p, model, INS, Met=0):
    """Compute sub-spectrum line positions/intensities for a fitted model.

    Given the flat parameter array ``p`` and the component list ``model`` (and the
    instrumental parameters ``INS``), returns the per-component line positions and
    intensities produced by ``pos_ac``. Handles multi-spectrum models split by
    ``Nbaseline``.

    Used outside this module by fitting_io.py, spectrum_io.py and syncmoss_main.py
    to draw the line-position markers and component decomposition on the plot.
    """
    if model.count('Nbaseline') == 0:
        positions = pos_ac(p, model, INS, Met)
    else:
        Di, Co, V, MV = 0, 0, 0, 0

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
            V = V + number_of_baseline_parameters
            positions = pos_ac(p, model_separate[i], INS, Met, V)

            for j in range(MV, len(model)):
                MV += 1
                V += int(4 * (model[j] == 'Singlet') + 9 * (model[j] == 'Doublet') + 14 * (model[j] == 'Sextet') + 17 * (model[j] == 'MDGD')\
                    + 14 * (model[j] == 'Relax_2S') + 11 * (model[j] == 'Relax_MS') + 15 * (model[j] == 'ASM') + 27 * (model[j] == 'SCDW')\
                    + 15 * (model[j] == 'Hamiltonian') + 12 * (model[j] == 'Hamilton_mc') + 9 * (model[j] == 'Hamilton_pc')\
                    + 5 * (model[j] == 'Distr') + 2 * (model[j] == 'Corr') + 7 * (model[j] == 'Recon') \
                    + numco * (model[j] == 'Variables') + 1*(model[j] =='Expression') + 4 * (model[j] == 'Spectrum')
                    + 5 * (model[j] == 'Distr_T')) # + number_of_baseline_parameters * (model[j] == 'Nbaseline')
                # print('V is equal to ', V)
                if model[j] == 'Distr':
                    Di += 1
                if model[j] == 'Corr':
                    Co += 1
                if model[j] == 'Nbaseline':
                    break
            # print('finally V is equal to ', V)

    return positions



