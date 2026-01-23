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
import minimi_lib as mi
import os
import platform
import time
from numba import njit, prange
import scipy
import scipy.linalg
dummy = scipy.linalg.eig(np.array([[1,0], [0,1]])) #required to build exe
from constants import number_of_baseline_parameters
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

    I1 = I * 3 * (1 - Ah) / (8 - 4 * Ah)
    I2 = I * 2 * Ah / (8 - 4 * Ah)
    I3 = I * 1 * (1 - Ah) / (8 - 4 * Ah)

    tmp1 = (Weigth * V1).sum(axis=1)
    tmp2 = (Weigth * V2).sum(axis=1)
    tmp3 = (Weigth * V3).sum(axis=1)

    return(2 * tmp1 * I1 + 2 * tmp2 * I2 + 2 * tmp3 * I3)


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
def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R): # based on eq. 3.5 (p.450) from https://doi.org/10.1103/PhysRev.165.446
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


def TImod (x_exp, p, model, EE, x0, MulCo, INS, Distri, Cor, Met = 0, Mett = -2, O=[], Di=0, Co=0, V=number_of_baseline_parameters):
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

        for i in range (0, len(model)):
            if model[i] == 'Singlet':
                I = abs(p[V])
                S = (-1) * p[V + 1]*MulCo + E
                WL = abs(p[V + 2]) * MulCo
                WG = abs(p[V + 3]) * MulCo
                V += 4
                Voi = Voight(WL, WG, S)
                CHt = CH*np.exp((-1)*np.pi*(G/2/E0*c*MulCo)*T*I*Voi)
            if model[i] == 'Doublet':
                I = abs(p[V])
                I1 = I * (p[V+5]+1)/2/(2-p[V+5]) #(2-p[V+5]*2) / 2
                I2 = I * 3*(1-p[V+5])/2/(2-p[V+5]) #p[V+5]*2 / 2
                WL = abs(p[V + 3]) * MulCo
                WG1 = abs(p[V + 4]) * MulCo
                WG2 = WG1 * p[V + 6]
                S1 = (-1) * (p[V + 1] - p[V + 2])*MulCo + E
                S2 = (-1) * (p[V + 1] + p[V + 2])*MulCo + E
                Voi1 = Voight(WL, WG1, S1)
                Voi2 = Voight(WL, WG2, S2)
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * T * (I1 * Voi1 + I2 * Voi2))
                V += 7
            if model[i] == 'Sextet':
                I = abs(p[V])
                I1 = I * (4*p[V+10]/(p[V+10]+1)) * (1 - p[V + 6]) / (8 - 4 * p[V + 6])
                I2 = I * 2 * p[V + 6] / (8 - 4 * p[V + 6])
                I3 = I * (4/(p[V+10]+1)) * (1 - p[V + 6]) / (8 - 4 * p[V + 6])
                HH = p[V + 3] / 3.101
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 7] * MulCo + E
                S2 = (-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 8] * MulCo + E
                S3 = (-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 8] * MulCo + E
                S4 = (-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 8] * MulCo + E
                S5 = (-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 8] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 7] * MulCo + E
                WL = abs(p[V + 4]) * MulCo
                WG = abs(p[V + 5]) * MulCo
                GaH = abs(p[V + 9]) / 2 / 3.101 * MulCo
                Ga16 = (WG**2 + GaH**2)** (1/2)
                Ga25 = (WG**2 + (3.0760 / 5.3123 * GaH)**2)** (1/2)
                Ga34 = (WG**2 + (0.8397 / 5.3123 * GaH)**2)** (1/2)
                Voi1 = Voight(WL, Ga16, S1)
                Voi2 = Voight(WL, Ga25, S2)
                Voi3 = Voight(WL, Ga34, S3)
                Voi4 = Voight(WL, Ga34, S4)
                Voi5 = Voight(WL, Ga25, S5)
                Voi6 = Voight(WL, Ga16, S6)
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * (I1*(Voi1+Voi6)+I2*(Voi2+Voi5)+I3*(Voi3+Voi4)))
                V += 11
            if model[i] == 'MDGD': # not finished
                I = abs(p[V])
                I1 = I * (4*p[V+13]/(p[V+13]+1)) * (1 - p[V + 10]) / (8 - 4 * p[V + 10])
                I2 = I * 2 * p[V + 10] / (8 - 4 * p[V + 10])
                I3 = I * (4/(p[V+13]+1)) * (1 - p[V + 10]) / (8 - 4 * p[V + 10])
                HH = p[V + 3] / 3.101
                S1 = (-1) * (p[V + 1] - HH / 2 + p[V + 2]) * MulCo - p[V + 11] * MulCo + E
                S2 = (-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 12] * MulCo + E
                S3 = (-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 12] * MulCo + E
                S4 = (-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) * MulCo + p[V + 12] * MulCo + E
                S5 = (-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) * MulCo - p[V + 12] * MulCo + E
                S6 = (-1) * (p[V + 1] + HH / 2 + p[V + 2]) * MulCo + p[V + 11] * MulCo + E
                WL = abs(p[V + 4]) * MulCo

                Guni = abs(p[V + 5]) * MulCo
                Gh = abs(p[V + 6]) * MulCo #/ 2 / 3.101
                Gde = p[V + 7] #* MulCo
                Gdh = p[V + 8] #* MulCo
                Geh = p[V + 9] #* MulCo

                Cd = [1,1,1,1,1,1]
                Ce = [1,-1,-1,-1,-1,1]
                Ch = [-1/6.202,-1/10.71,-1/39.24,1/39.24,1/10.71,1/6.202]

                Gfinal = []
                for j in range(0, 6):
                    Gfinal.append(np.sqrt(abs(Guni**2 + Ch[j]**2*Gh**2 + Cd[j]*Ce[j]*Gde*Guni**2 * max( 0, (1 - (abs(Gdh)+abs(Geh))**2)) + Cd[j]*Ch[j]*2*Gdh*Guni*Gh + Ce[j]*Ch[j]*2*Geh*Guni*Gh )))

                Voi1 = Voight(WL, Gfinal[0], S1)
                Voi2 = Voight(WL, Gfinal[1], S2)
                Voi3 = Voight(WL, Gfinal[2], S3)
                Voi4 = Voight(WL, Gfinal[3], S4)
                Voi5 = Voight(WL, Gfinal[4], S5)
                Voi6 = Voight(WL, Gfinal[5], S6)
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * (I1*(Voi1+Voi6)+I2*(Voi2+Voi5)+I3*(Voi3+Voi4)))
                V += 14
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
            if model[i] == 'Hamilton_mc':
                I = abs(p[V])
                delt = p[V+1] * MulCo
                Q = p[V+2]
                H = p[V+3]
                WL = p[V+4] * MulCo
                WG = p[V+5] * MulCo
                eto = p[V+6]
                tet = p[V+7]
                phi = p[V+8]
                tetr = p[V+9]
                phir = p[V+10]
                if Mett == 0 or Met == 3:
                    Itmp, S = Ham_mono(Q, H, eto, phi, tet, phir, tetr)
                if Mett == 1:
                    Itmp, S = Ham_mono_CMS(Q, H, eto, phi, tet, phir, tetr)
                # if Mett != 1 and Mett != 0:

                I = Itmp * I
                S = S * MulCo
                S += delt
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo)\
                                  * (I[0]*Voight(WL,WG,E-S[0])+I[1]*Voight(WL,WG,E-S[1])+I[2]*Voight(WL,WG,E-S[2])+I[3]*Voight(WL,WG,E-S[3])\
                                    +I[4]*Voight(WL,WG,E-S[4])+I[5]*Voight(WL,WG,E-S[5])+I[6]*Voight(WL,WG,E-S[6])+I[7]*Voight(WL,WG,E-S[7])))
                V += 11
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
            if model[i] == 'Relax_MS':
                I = abs(float(p[V]) * 2)
                Sig = float(p[V + 1]) * MulCo
                eps = float(p[V + 2]) * MulCo
                Hv = float(p[V + 3]) * MulCo / 2 / 3.1098
                W = float(p[V + 4]) * MulCo / 2
                Ah = float(p[V + 5])
                R = float(p[V + 6])
                alfa = float(p[V + 7])
                S = float(p[V + 8])

                CHt = CH * (np.exp(relax_MS(S, E, I, Sig, eps, Hv, W, Ah, R, alfa) *(-1)*np.pi*(G/2/E0*c*MulCo)*T))

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

                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * T * I * Angles_min(tet, Num, K, J, Hin, Hex, E, WL, WG, Q, Sig))
                V += 11
            if model[i] == 'Relax_2S':
                I = abs(p[V])
                I1 = I * 3 * (1 - p[V + 8]) / (8 - 4 * p[V + 8])
                I2 = I * 2 * p[V + 8]       / (8 - 4 * p[V + 8])
                I3 = I * 1 * (1 - p[V + 8]) / (8 - 4 * p[V + 8])
                Sig1 = p[V + 1] * MulCo
                Q1 = p[V + 2] / 3 * MulCo
                H1 = p[V + 3] / (abs(ggr) + 3 * abs(gex)) * 2 * MulCo / 3.101 / 2
                Sig2 = p[V + 4] * MulCo
                Q2 = p[V + 5] / 3 * MulCo
                H2 = p[V + 6] / (abs(ggr) + 3 * abs(gex)) * 2 * MulCo / 3.101 / 2
                WL = p[V + 7] * MulCo
                # WG = p[V + 8] * MulCo
                # A = p[V + 9]
                We = p[V + 9] * MulCo
                R = p[V + 10]

                # Enew = E
                ## Enew = np.sort(np.concatenate((E, (E[:-1] + E[1:]) / 2)))
                # WGS = WG / 2 / np.sqrt(2 * np.log(2)) + 0.001*(WG==0)
                # Ga, GV = np.meshgrid(Enew[:-1], E)
                # GG = (1 / WGS / np.sqrt(2 * np.pi) * np.exp(-(Ga - GV) ** 2 / WGS ** 2 / 2))


                # Blu = Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, -1/2, -3/2, Enew, R) * I1 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, 1/2,  3/2, Enew, R) * I1\
                #     + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, -1/2, -1/2, Enew, R) * I2 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, 1/2,  1/2, Enew, R) * I2\
                #     + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, -1/2,  1/2, Enew, R) * I3 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, GG, We, 1/2, -1/2, Enew, R) * I3

                Blu = Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1/2, -3/2, E, R) * I1 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1/2,  3/2, E, R) * I1\
                    + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1/2, -1/2, E, R) * I2 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1/2,  1/2, E, R) * I2\
                    + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, -1/2,  1/2, E, R) * I3 + Blume(Sig1, Sig2, Q1, Q2, H1, H2, WL, We, 1/2, -1/2, E, R) * I3


                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo) * Blu)
                V += 11
            if model[i] == 'ASM':
                I = abs(p[V])
                Sig = p[V + 1] * MulCo
                eps_m = p[V + 2] * MulCo
                eps_lat = p[V + 3] * MulCo
                His = p[V + 4] * MulCo #/ 3.101
                Han = p[V + 5] * MulCo #/ 3.101
                WL = abs(p[V + 6]) * MulCo
                WG = abs(p[V + 7]) * MulCo
                m = p[V + 8] #np.sign(p[V + 8]) * np.abs((p[V + 8]))**(1/5)
                A = p[V + 9]
                Num = int(abs(p[V + 10]))
                I13 = p[V + 11]
                CHt = CH * np.exp((-1) * np.pi * (G / 2 / E0 * c * MulCo)\
                                  * (I * ASM(T, Sig, eps_m, eps_lat, His, Han, WL, WG, m, A, Num, I13, E)))
                V += 12

            if model[i] == 'Variables':
                V += 15
                CHt = CH
            if model[i] == 'Expression':
                V += 1
                CHt = CH

            if model[i] == 'Distr':
                Num = int(p[V + 3])
                CH = CHold
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

                Vnum = int(4*(model[k]=='Singlet') + 7*(model[k]=='Doublet') + 11*(model[k]=='Sextet') + 14*(model[k]=='Sextet(rough)') + 14 * (model[k] == 'MDGD')\
                           + 9*(model[k]=='Relax_MS') + 15*(model[k]=='Variables') + 11*(model[k]=='Average_H') + 12*(model[k]=='ASM')\
                           + 11*(model[k]=='Relax_2S')) + 11*(model[k]=='Hamilton_mc') + 9*(model[k]=='Hamilton_pc') + 1*(model[k]=='Expression')

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
                    CHt = CH * (TImod(x_exp, pN, model_d, E, x0, MulCo, INS, np.array([Distri[mDk:mDk+Dk]]*Num).flatten(), np.array([Cor[mCk:mCk+Ck]]*Num).flatten(), Met = -1, Mett = Mett, O=O))
                else:
                    CHt = CH
                Di += 1
                # print('Distri proceed')

            if model[i] == 'Nbaseline':
                break

            CHold = CH
            CH = CHt

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


def TI(x_exp, p, model, JN, pool, x0, MulCo, INS, Distri=[0], Cor = [0], Met=0, Norm = 1):  # num - number of Gausians # PS - spc, p - InsFun
    # INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
    E = np.linspace(-1 + (10 ** -2)*(Met == 1 or Met ==2) + 10**-3, 1 - (10 ** -2)*(Met == 1 or Met ==2) - 10**-3, JN)

    D = (E[1] - E[0])
    if model.count('Nbaseline') == 0:
        H = pool.starmap(TImod, [(x_exp, p, model, Ex, x0, MulCo, INS, Distri, Cor, Met) for Ex in E])
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
            N0 = (p[V]   + p[V+3] * p[V]  /10**2 * x_separate[i] + p[V+2] * p[V]   / 10 ** 4 * ((-1) * p[V+1] + x_separate[i]) ** 2)
            N1 =  p[V+4] + p[V+7] * p[V+4]/10**2 * x_separate[i] + p[V+6] * p[V+4] / 10 ** 4 * ((-1) * p[V+5] + x_separate[i]) ** 2
            V = V + number_of_baseline_parameters
            H = pool.starmap(TImod, [(x_separate[i], p, model_separate[i], Ex, x0, MulCo, INS, Distri, Cor, Met, -2, [], Di, Co, V) for Ex in E])
            # Di = H[0][1]
            # Co = H[0][2]
            # V = H[0][3]
            H = np.array(H, dtype=object).sum(axis=0)
            H = H * N0 / Norm * D + N1
            Hc = np.concatenate((Hc, H))

            for j in range(MV, len(model)):
                MV += 1
                V += int(4 * (model[j] == 'Singlet') + 7 * (model[j] == 'Doublet') + 11 * (model[j] == 'Sextet') + 14 * (model[j] == 'Sextet(rough)') + 14 * (model[j] == 'MDGD')\
                    + 11 * (model[j] == 'Relax_2S') + 11 * (model[j] == 'Average_H') + 9 * (model[j] == 'Relax_MS') + 12*(model[j]=='ASM')\
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

    return Hc



