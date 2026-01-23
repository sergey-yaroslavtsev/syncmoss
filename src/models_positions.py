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
from numpy import *
import builtins as bu
def max(*args):
    return bu.max(*args)
# import dual_v3 as dn
import minimi_lib as mi
import os
import platform
import time
from numba import njit, prange

from numpy.linalg import eig
from numpy import linalg as LA
# from numpy.linalg import inv
from numpy import abs
from constants import number_of_baseline_parameters
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

@njit(cache=True)
def Ham_mono(Q, Hhf, etto, phi, tet, phir, tetr):
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

    phir = phir / 180 * np.pi
    tetr = tetr / 180 * np.pi
    aM = np.array([[0.0 + 0j] * 8] * 3)

    F2 = np.sqrt(2) * np.sin(tetr) * (-1j) * np.exp(1j * (phir))
    F4 = np.sqrt(2) * np.cos(tetr) * (1j)
    F6 = np.sqrt(2) * np.sin(tetr) * (1j) * np.exp(-1j * (phir))

    for i in range(0, 8):
        aM[0][i] = (np.sqrt(1 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4]) + Vex[0][
            i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4])) \
                    * (1 / 4) * np.sqrt(3 / np.pi) * F2

        aM[1][i] = (np.sqrt(2 / 3) * Vex[1][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + np.sqrt(2 / 3) *
                    Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                    * (1/4) * np.sqrt(3*2/np.pi) * F4

        aM[2][i] = (np.sqrt(1 / 3) * Vex[2][i - 4 * (i // 4)] * np.conjugate(Vgr[0][i // 4]) + Vex[3][
            i - 4 * (i // 4)] * np.conjugate(Vgr[1][i // 4])) \
                    * (1 / 4) * np.sqrt(3 / np.pi) * F6

    S = np.array([float(0)] * 8)
    for i in range(0, 8):
        S[i] = Hex[i - 4 * (i // 4)] - Hgr[i // 4]
    S = S / E0_J * c

    E = aM[0] + aM[1] + aM[2]
    I = np.real(E * np.conjugate(E))
    I = I * np.pi

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
            for i in range (0, int((len(INS))/3)):
                ZERO +=  INS[i*3+1]*INS[i*3+2]**2


        for i in range (0, len(model)):
            if model[i] == 'Singlet':
                S = p[V + 1]
                V += 4
                SET.append([S])
            if model[i] == 'Doublet':
                S1 = (p[V + 1] - p[V + 2])
                S2 = (p[V + 1] + p[V + 2])
                SET.append([S1, S2])
                V += 7
            if model[i] == 'Sextet':
                HH = p[V + 3] / 3.101
                S1 = (p[V + 1] - HH / 2 + p[V + 2]) + p[V + 7]
                S2 = (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 8]
                S3 = (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 8]
                S4 = (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 8]
                S5 = (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 8]
                S6 = (p[V + 1] + HH / 2 + p[V + 2]) - p[V + 7]
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 11
            if model[i] == 'Sextet(rough)':
                HH = p[V + 3] / 3.101
                S1 = (p[V + 1] - HH / 2 + p[V + 2]) + p[V + 6]
                S2 = (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 7]
                S3 = (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 7]
                S4 = (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 7]
                S5 = (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 7]
                S6 = (p[V + 1] + HH / 2 + p[V + 2]) - p[V + 6]
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 14
            if model[i] == 'MDGD':
                HH = p[V + 3] / 3.101
                S1 = (p[V + 1] - HH / 2 + p[V + 2]) + p[V + 11]
                S2 = (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 12]
                S3 = (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 12]
                S4 = (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 12]
                S5 = (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 12]
                S6 = (p[V + 1] + HH / 2 + p[V + 2]) - p[V + 11]
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 14
            if model[i] == 'Hamilton_mc':
                delt = p[V+1]
                Q = p[V+2]
                H = p[V+3]
                eto = p[V+6]
                tet = p[V+7]
                phi = p[V+8]
                tetr = p[V+9]
                phir = p[V+10]
                if Met == 0: # is it useless? polarization do not changes positions but intensities
                    S = Ham_mono(Q, H, eto, phi, tet, phir, tetr)[1]
                if Met == 1:
                    S = Ham_mono_CMS(Q, H, eto, phi, tet, phir, tetr)[1]
                S += delt
                SET.append(S)
                V += 11
            if model[i] == 'Hamilton_pc':
                delt = p[V+1]
                Q = p[V+2]
                H = p[V+3]
                eto = p[V+6]
                tet = p[V+7]
                phi = p[V+8]
                S = Ham_poly(Q, H, eto, phi, tet)[1]
                S += delt
                SET.append(S)
                V += 9
            if model[i] == 'Relax_MS':
                HH = float(p[V + 3]) / 3.1098
                S1 = (p[V + 1] - HH / 2 + p[V + 2])
                S2 = (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2])
                S3 = (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2])
                S4 = (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2])
                S5 = (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2])
                S6 = (p[V + 1] + HH / 2 + p[V + 2])
                SET.append([S1, S2, S3, S4, S5, S6])
                V += 9

            if model[i] == 'Relax_2S':
                HH = float(p[V + 3]) / 3.1098
                S1 = (p[V + 1] - HH / 2 + p[V + 2])
                S2 = (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2])
                S3 = (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2])
                S4 = (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2])
                S5 = (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2])
                S6 = (p[V + 1] + HH / 2 + p[V + 2])
                HH2 = float(p[V + 6]) / 3.1098
                S12 = (p[V + 4] - HH2 / 2 + p[V + 5])
                S22 = (p[V + 4] - 3.0760 / 5.3123 * HH2 / 2 - p[V + 5])
                S32 = (p[V + 4] - 0.8397 / 5.3123 * HH2 / 2 - p[V + 5])
                S42 = (p[V + 4] + 0.8397 / 5.3123 * HH2 / 2 - p[V + 5])
                S52 = (p[V + 4] + 3.0760 / 5.3123 * HH2 / 2 - p[V + 5])
                S62 = (p[V + 4] + HH2 / 2 + p[V + 5])
                SET.append([S1, S2, S3, S4, S5, S6, S12, S22, S32, S42, S52, S62])
                V += 11

            if model[i] == 'ASM':
                I = abs(p[V])
                Sig = p[V + 1]
                eps_m = p[V + 2]
                eps_lat = p[V + 3]
                His = p[V + 4] #/ 3.101
                Han = p[V + 5] #/ 3.101

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

                V += 12

            if model[i] == 'Variables':
                SET.append([])
                V += 15
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
            if model[i] == 'Nbaseline':
                break

        SET = np.array(SET) + ZERO

        return(SET)


def mod_pos(p, model, INS, Met=0):

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
                V += int(4 * (model[j] == 'Singlet') + 7 * (model[j] == 'Doublet') + 11 * (model[j] == 'Sextet') + 14 * (model[j] == 'Sextet(rough)') + 14 * (model[j] == 'MDGD')\
                    + 11 * (model[j] == 'Relax_2S') + 11 * (model[j] == 'Average_H') + 9 * (model[j] == 'Relax_MS') + 12 * (model[j] == 'ASM')\
                    + 11 * (model[j] == 'Hamilton_mc') + 9 * (model[j] == 'Hamilton_pc')\
                    + 5 * (model[j] == 'Distr') + 2 * (model[j] == 'Corr') \
                    + 15 * (model[j] == 'Variables') + 1*(model[j] =='Expression')) # + number_of_baseline_parameters * (model[j] == 'Nbaseline')
                # print('V is equal to ', V)
                if model[j] == 'Distr':
                    Di += 1
                if model[j] == 'Corr':
                    Co += 1
                if model[j] == 'Nbaseline':
                    break
            # print('finally V is equal to ', V)

    return positions



