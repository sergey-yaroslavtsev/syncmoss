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
import os
import platform
import time
from numba import njit, prange

NBA = 6

G = 4.7 * 10 ** -9  # natural width in eV*10**-9 # 4.7 value from Ralf Rohlsberger
E0 = 14412  # energy of resonance
c = 2.99792458 * 10 ** 11  # speed of light at mm/s
h = 6.5821 * 10**-16
t0 = 141 * 10**-9
#
# dir_path = os.getcwd()
# if platform.system() == 'Windows':
#     realpath = str(dir_path) + str('\\\\INSexp.txt')
# else:
#     realpath = str(dir_path) + str('/INSexp.txt')

@njit
def sUm(n, t, CHt, CHti):
    for k in range(1, n):
        for kk in range(1, len(t)):
            CHt[k][kk] = abs(t[1] - t[0]) * np.sum(CHt[k - 1][0:kk] * (CHt[0][0:kk][::-1]) - CHti[k - 1][0:kk] * (CHti[0][0:kk][::-1])) / t0
            CHti[k][kk] = abs(t[1] - t[0]) * np.sum(CHt[k - 1][0:kk] * (CHti[0][0:kk][::-1]) + CHti[k - 1][0:kk] * (CHt[0][0:kk][::-1])) / t0
    return CHt, CHti

@njit
def sUm2(n, t, CHtr, CHtir, CHtd, CHtid):
    for k in range(1, n):
        for kk in range(1, len(t)):
            CHtr[k][kk] = abs(t[kk] - t[kk-1]) * np.sum(CHtr[k - 1][0:kk] * (CHtr[0][0:kk][::-1]) - CHtir[k - 1][0:kk] * (CHtir[0][0:kk][::-1])) / t0
            CHtir[k][kk] = abs(t[kk] - t[kk-1]) * np.sum(CHtr[k - 1][0:kk] * (CHtir[0][0:kk][::-1]) + CHtir[k - 1][0:kk] * (CHtr[0][0:kk][::-1])) / t0

            CHtd[k][kk] = abs(t[kk] - t[kk-1]) * np.sum(CHtr[k - 1][0:kk] * (CHtd[0][0:kk][::-1]) - CHtir[k - 1][0:kk] * (CHtid[0][0:kk][::-1]) +\
                                                            CHtd[k - 1][0:kk] * (CHtr[0][0:kk][::-1]) - CHtid[k - 1][0:kk] * (CHtir[0][0:kk][::-1])) / t0
            CHtid[k][kk] = abs(t[kk] - t[kk-1]) * np.sum(CHtr[k - 1][0:kk] * (CHtid[0][0:kk][::-1]) + CHtir[k - 1][0:kk] * (CHtd[0][0:kk][::-1]) +\
                                                                 CHtd[k - 1][0:kk] * (CHtir[0][0:kk][::-1]) + CHtid[k - 1][0:kk] * (CHtr[0][0:kk][::-1])) / t0


    return CHtr, CHtir, CHtd, CHtid

def TImod(t, p, model, ttn, ttn2, L, T, orderT):


    CH = 0
    CHi = 0

    V = NBA
    mas = 0
    for i in range(0, len(model)):
        if model[i] == 'Singlet':
            I = abs(p[V]) /4
            mas += I
            V += 4
        if model[i] == 'Doublet':
            I = abs(p[V]) /4
            mas += I
            V += 7
        if model[i] == 'Sextet':
            I = abs(p[V]) /4
            mas += I
            V += 11

    V = NBA
    for i in range(0, len(model)):
        if model[i] == 'Singlet':
            I = abs(p[V]) /4
            WL = abs(p[V + 2]) * E0 / c
            WG = abs(p[V + 3]) * E0 / c
            W = np.power(WG**5 + 2.69269*WL*WG**4 + 2.42843*WL**2*WG**3 + 4.47163*WL**3*WG**2 + 0.07842*WL**4*WG + WL**5, 1/5)
            et = 1.36603 * WL / W - 0.47719 * WL ** 2 / W ** 2 + 0.11116 * WL ** 3 / W ** 3
            S = (-1) * p[V + 1] * E0 / c
            CH +=  np.cos(S * (-1) * t / h) * I * et * np.exp(W * (-1/2) * t / h) +\
                   np.cos(S * (-1) * t / h) * (1 - et) * I * np.exp((-1) * (W * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)
            CHi += np.sin(S * (-1) * t / h) * (-1) * I * et * np.exp(W * (-1/2) * t / h) -\
                   np.sin(S * (-1) * t / h) * (1 - et) * I * np.exp((-1) * (W * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)
            V += 4
        if model[i] == 'Doublet':
            I = abs(p[V]) /4
            I1 = I * (p[V + 5] + 1) / 2 / (2 - p[V + 5])
            I2 = I * 3 * (1 - p[V + 5]) / 2 / (2 - p[V + 5])
            # I1 = abs(p[V]) * (2-p[V+5]*2) /4
            # I2 = abs(p[V]) * p[V+5]*2 /4
            WL = abs(p[V + 3]) * E0 / c
            WG1 = abs(p[V + 4]) * E0 / c
            WG2 = WG1 * p[V + 6]
            # WG = abs(p[V + 4]) * E0 / c
            W1 = np.power(WG1**5 + 2.69269*WL*WG1**4 + 2.42843*WL**2*WG1**3 + 4.47163*WL**3*WG1**2 + 0.07842*WL**4*WG1 + WL**5, 1/5)
            W2 = np.power(WG2**5 + 2.69269*WL*WG2**4 + 2.42843*WL**2*WG2**3 + 4.47163*WL**3*WG2**2 + 0.07842*WL**4*WG2 + WL**5, 1/5)
            et1 = 1.36603 * WL / W1 - 0.47719 * WL ** 2 / W1 ** 2 + 0.11116 * WL ** 3 / W1 ** 3
            et2 = 1.36603 * WL / W2 - 0.47719 * WL ** 2 / W2 ** 2 + 0.11116 * WL ** 3 / W2 ** 3
            S1 = (-1) * (p[V + 1] - p[V + 2]) * E0 / c
            S2 = (-1) * (p[V + 1] + p[V + 2]) * E0 / c
            CH +=  np.cos(S1 * (-1) * t / h) * I1 * et1 * np.exp(W1 * (-1/2) * t / h) +\
                   np.cos(S1 * (-1) * t / h) * (1 - et1) * I1 * np.exp((-1) * (W1 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S2 * (-1) * t / h) * I2 * et2 * np.exp(W2 * (-1/2) * t / h) +\
                   np.cos(S2 * (-1) * t / h) * (1 - et2) * I2 * np.exp((-1) * (W2 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)
            CHi += np.sin(S1 * (-1) * t / h) * (-1) * I1 * et1 * np.exp(W1 * (-1/2) * t / h) -\
                   np.sin(S1 * (-1) * t / h) * (1 - et1) * I1 * np.exp((-1) * (W1 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S2 * (-1) * t / h) * (-1) * I2 * et2 * np.exp(W2 * (-1/2) * t / h) -\
                   np.sin(S2 * (-1) * t / h) * (1 - et2) * I2 * np.exp((-1) * (W2 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)
            V += 7
        if model[i] == 'Sextet':
            # I1 = abs(p[V]) * 3 * (1 - p[V + 5]) / (8 - 4 * p[V + 5]) /4
            # I2 = abs(p[V]) * 2 * p[V + 5] / (8 - 4 * p[V + 5]) /4
            # I3 = abs(p[V]) * 1 * (1 - p[V + 5]) / (8 - 4 * p[V + 5]) /4
            I1 = abs(p[V]) * (4*p[V+10]/(p[V+10]+1)) * (1 - p[V + 6]) / (8 - 4 * p[V + 6]) /4
            I2 = abs(p[V]) * 2 * p[V + 6] / (8 - 4 * p[V + 6]) /4
            I3 = abs(p[V]) * (4/(p[V+10]+1)) * (1 - p[V + 6]) / (8 - 4 * p[V + 6]) /4

            WL = abs(p[V + 4]) * E0 / c
            WG = abs(p[V + 5]) * E0 / c
            GaH = abs(p[V + 9]) / 2 / 3.101 * E0 / c
            Ga16 = (WG ** 2 + GaH ** 2) ** (1 / 2)
            Ga25 = (WG ** 2 + (3.0760 / 5.3123 * GaH) ** 2) ** (1 / 2)
            Ga34 = (WG ** 2 + (0.8397 / 5.3123 * GaH) ** 2) ** (1 / 2)
            W16 = np.power(Ga16**5 + 2.69269*WL*Ga16**4 + 2.42843*WL**2*Ga16**3 + 4.47163*WL**3*Ga16**2 + 0.07842*WL**4*Ga16 + WL**5, 1/5)
            W25 = np.power(Ga25**5 + 2.69269*WL*Ga25**4 + 2.42843*WL**2*Ga25**3 + 4.47163*WL**3*Ga25**2 + 0.07842*WL**4*Ga25 + WL**5, 1/5)
            W34 = np.power(Ga34**5 + 2.69269*WL*Ga34**4 + 2.42843*WL**2*Ga34**3 + 4.47163*WL**3*Ga34**2 + 0.07842*WL**4*Ga34 + WL**5, 1/5)
            et16 = 1.36603 * WL / W16 - 0.47719 * WL ** 2 / W16 ** 2 + 0.11116 * WL ** 3 / W16 ** 3
            et25 = 1.36603 * WL / W25 - 0.47719 * WL ** 2 / W25 ** 2 + 0.11116 * WL ** 3 / W25 ** 3
            et34 = 1.36603 * WL / W34 - 0.47719 * WL ** 2 / W34 ** 2 + 0.11116 * WL ** 3 / W34 ** 3
            HH = p[V + 3] / 3.101

            S1 = ((-1) * (p[V + 1] - HH / 2 + p[V + 2]) - p[V + 7]) * E0 / c
            S2 = ((-1) * (p[V + 1] - 3.0760 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 8]) * E0 / c
            S3 = ((-1) * (p[V + 1] - 0.8397 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 8]) * E0 / c
            S4 = ((-1) * (p[V + 1] + 0.8397 / 5.3123 * HH / 2 - p[V + 2]) + p[V + 8]) * E0 / c
            S5 = ((-1) * (p[V + 1] + 3.0760 / 5.3123 * HH / 2 - p[V + 2]) - p[V + 8]) * E0 / c
            S6 = ((-1) * (p[V + 1] + HH / 2 + p[V + 2]) + p[V + 7]) * E0 / c

            CH +=  np.cos(S1 * (-1) * t / h) * I1 * et16 * np.exp(W16 * (-1/2) * t / h) +\
                   np.cos(S1 * (-1) * t / h) * (1 - et16) * I1 * np.exp((-1) * (W16 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S2 * (-1) * t / h) * I2 * et25 * np.exp(W25 * (-1/2) * t / h) +\
                   np.cos(S2 * (-1) * t / h) * (1 - et25) * I2 * np.exp((-1) * (W25 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S3 * (-1) * t / h) * I3 * et34 * np.exp(W34 * (-1/2) * t / h) +\
                   np.cos(S3 * (-1) * t / h) * (1 - et34) * I3 * np.exp((-1) * (W34 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S4 * (-1) * t / h) * I3 * et34 * np.exp(W34 * (-1/2) * t / h) +\
                   np.cos(S4 * (-1) * t / h) * (1 - et34) * I3 * np.exp((-1) * (W34 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S5 * (-1) * t / h) * I2 * et25 * np.exp(W25 * (-1/2) * t / h) +\
                   np.cos(S5 * (-1) * t / h) * (1 - et25) * I2 * np.exp((-1) * (W25 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.cos(S6 * (-1) * t / h) * I1 * et16 * np.exp(W16 * (-1/2) * t / h) +\
                   np.cos(S6 * (-1) * t / h) * (1 - et16) * I1 * np.exp((-1) * (W16 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)
            CHi += np.sin(S1 * (-1) * t / h) * (-1) * I1 * et16 * np.exp(W16 * (-1/2) * t / h) +\
                   np.sin(S1 * (-1) * t / h) * (-1) * (1 - et16) * I1 * np.exp((-1) * (W16 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S2 * (-1) * t / h) * (-1) * I2 * et25 * np.exp(W25 * (-1/2) * t / h) +\
                   np.sin(S2 * (-1) * t / h) * (-1) * (1 - et25) * I2 * np.exp((-1) * (W25 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S3 * (-1) * t / h) * (-1) * I3 * et34 * np.exp(W34 * (-1/2) * t / h) +\
                   np.sin(S3 * (-1) * t / h) * (-1) * (1 - et34) * I3 * np.exp((-1) * (W34 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S4 * (-1) * t / h) * (-1) * I3 * et34 * np.exp(W34 * (-1/2) * t / h) +\
                   np.sin(S4 * (-1) * t / h) * (-1) * (1 - et34) * I3 * np.exp((-1) * (W34 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S5 * (-1) * t / h) * (-1) * I2 * et25 * np.exp(W25 * (-1/2) * t / h) +\
                   np.sin(S5 * (-1) * t / h) * (-1) * (1 - et25) * I2 * np.exp((-1) * (W25 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2) +\
                   np.sin(S6 * (-1) * t / h) * (-1) * I1 * et16 * np.exp(W16 * (-1/2) * t / h) +\
                   np.sin(S6 * (-1) * t / h) * (-1) * (1 - et16) * I1 * np.exp((-1) * (W16 * t / 2 / np.sqrt(2 * np.log(2)) / h)**2 / 2)

            V += 11



    CH = CH / mas
    CHi = CHi / mas
    # phase = (np.arctan(CHi / CH))
    n = 20


    CHt = np.array([[float(0)] * L] * n)
    CHti = np.array([[float(0)] * L] * n)

    CHt[0] = CH
    CHti[0] = CHi

    # start_time = time.time()
    #
    # B = np.array([[float(0)]*len(t)]*len(t))
    # Bi = np.array([[float(0)]*len(t)]*len(t))
    # A = np.tril(np.full(len(t),1), -1)
    # for k in range(0, len(t)-1):
    #     B[k][k+1:len(t)] = CH[0:len(t)-1-k]
    #     Bi[k][k+1:len(t)] = CHi[0:len(t)-1-k]
    # B = np.transpose(B)
    # Bi = np.transpose(Bi)

    # print('precalc for dynamic', time.time() - start_time, 'seconds')
    # start_time = time.time()

    CHt, CHti = sUm(n, T, CHt, CHti)

    # for k in range(1, n):
    #     for kk in range(1, len(t)):
    #         # CHt[k][kk] = abs(x_exp[1]-x_exp[0]) * np.sum(CHt[k-1][0:kk]*(CHt[0][0:kk][::-1])) /t0
    #
    #         CHt[k][kk] = abs(t[1] - t[0]) * np.sum(CHt[k - 1][0:kk] * (CHt[0][0:kk][::-1]) - CHti[k - 1][0:kk] * (CHti[0][0:kk][::-1])) / t0
    #         CHti[k][kk] = abs(t[1] - t[0]) * np.sum(CHt[k - 1][0:kk] * (CHti[0][0:kk][::-1]) + CHti[k - 1][0:kk] * (CHt[0][0:kk][::-1])) / t0
    #
    #         # CHt[k] = abs(t[k] - t[k-1]) * np.sum(A * CHt[k - 1] * B - A * CHti[k - 1] * Bi, axis=1)[::-1] / t0
    #         # CHti[k] = abs(t[k] - t[k-1]) * np.sum(A * CHt[k - 1] * Bi + A * CHti[k - 1] * B, axis=1)[::-1] / t0

    # print('dynamic sequence', time.time() - start_time, 'seconds')

    # print(CHt[1])
    # print(CHti[1])

    CH = np.array([float(0)] * L) [ttn:-ttn2] [::orderT]
    CHi = np.array([float(0)] * L) [ttn:-ttn2] [::orderT]

    for k in range(1, n+1):
        CH += (CHt[k-1]*float (mas**k)/float(np.math.factorial(k))*(-1)**(k+1)) [ttn:-ttn2] [::orderT]
        CHi += (CHti[k-1]*float (mas**k)/float(np.math.factorial(k))*(-1)**(k+1)) [ttn:-ttn2] [::orderT]

    if mas == 0:
        return (np.array([float(0)] * L) [ttn:-ttn2])
    else:
        return ((CH ** 2 + CHi ** 2) / mas ** 2)


def TI(x_exp, p, model, pool):  # num - number of Gausians # PS - spc, p - InsFun
    t = np.array(x_exp)
    orderT = 1
    if t[0] > t[-1]:
        t = t[::-1]
        orderT = -1
    try:
        shi = p.real[4]*10**-9
    except:
        shi = p[4]*10**-9
    if t[0] + shi != 0:
        ttn = max(int(t[0] / abs(x_exp[1] - x_exp[0])), 1)
        tt = np.linspace(0, 2 * t[0] - t[1], ttn)
        t = np.concatenate((tt, t), axis=0)
    else:
        ttn = 0
    ttn2 = 20 #int(21*10**-9 / abs(t[1] - t[0]))
    ttt = np.linspace(t[-1], t[-1] + abs(t[1] - t[0])*ttn2, ttn2)
    t = np.concatenate((t, ttt), axis=0)
    L = len(t)



    t = t + p[4]*10**-9
    T = t



    psh = p[3]
    pn = p
    pM = p[0]

    # N0 = (pn[0] + pn[2] * pn[0] * ((-1) * pn[1] + x_exp) ** 2)
    # H = TImod(t, pn, model, ttn, ttn2, L, T, orderT) * pM + psh
    H = pool.starmap(TImod, [(t, pn, model, ttn, ttn2, L, T, orderT)])[0] * pM + psh # pool to have opportunity for easy stop
    # H = H * N0 + pn[5] * ((-1) * pn[4] + x_exp) ** 2 + pn[3]

    return H