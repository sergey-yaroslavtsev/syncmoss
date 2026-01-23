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
from numba import njit, prange

G = 4.7*10**-9          # natural width in eV*10**-9 # 4.7 value from Ralf Rohlsberger
E0 = 14412          # energy of resonance
c = 2.99792458*10**11     # speed of light at mm/s
d = 0.005           # area density
Fa = 0.54676                 # fraction of resonance absorption
etto = 1           # percent of 57Fe
sigma = 2.464*10**-22     # max resonance cross section
If = 1

x0 = -0.097



def erf(z):
    if isinstance(z, np.ndarray):
        x = (z[0])
    else:
        x = z
    sign = 1 if x.real >= 0 else -1
    x = (x**2)**1/2
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp((-1)*x*x)
    return sign*y

def Gau(p, m, n, k):
    return k*np.exp(-(p-n)**2/2/(m)**2)/m/np.sqrt(2*np.pi)
def s_Gau(x, *m):
    f = 0
    for i in range (0, int(len(m)/3)):
        f = f + Gau(x, m[i*3], m[i*3+1], m[i*3+2])
    return(f)
def Lor(p, m, n, k):
    return k*m/2/np.pi/((p-n)**2+(m/2)**2)
def s_Lor(x, *m):
    f = 0
    for i in range (0, int(len(m)/3)):
        f = f + Lor(x, m[i*3], m[i*3+1], m[i*3+2])
    return(f)
def MeanV (a, b):
    return a + b*np.sqrt(2/np.pi)*np.exp(-1/2*a**2/b**2)/(1 - erf(-a/b/np.sqrt(2)))

@njit
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

def INSt (x_exp, p, EE, x00, MulCo, PPP, Be_p):

    # MulCo = 1
    SCR = x_exp
    Fg = ((PPP[1])**2)*MulCo *c/E0

    G = 0.098 * MulCo
    # G = 4.7 * 10 ** -9 * MulCo
    F = (Fg**5 + 2.69269*Fg**4*G+2.42843*Fg**3*G**2+4.47163*Fg**2*G**3+0.07842*Fg*G**4+G**5)**(1/5)
    et = 1.36603*G/F - 0.47719*G**2/F**2 +0.11116*G**3/F**3
    S = F/2/np.sqrt(2*np.log(2))
    SIG = ((PPP[2])**2)
    T = PPP[3]
    x0 = -0.097 * MulCo

    CH = 1

    N = 0

    E = np.log((1+EE)/(1-EE)) + SCR*MulCo + x00*MulCo
    G0 = 4.7 * 10 ** -9

    INSp = p[1:]
    for i in range (0, int((len(INSp))/3)):
                    N += 1*INSp[i*3+2]**2*np.exp((-1)*((E-(INSp[i*3+1]+SCR)*MulCo)**2/(2*((INSp[i*3]**2+G0/E0*c/2)*MulCo)**2)))/((INSp[i*3]**2+G0/E0*c/2)*MulCo)/np.sqrt(2*np.pi)

    # for i in range (0, int((len(p)-1)/3)):
    #         N = N + abs(p[1+i*3+2])*np.exp((-1)*(((-1)*(p[1+i*3+1]+SCR)*MulCo + E)**2/(2*((abs(p[1+i*3])+G0/E0*c/2)*MulCo)**2)))/((abs(p[1+i*3])+G0/E0*c/2)*MulCo)/np.sqrt(2*np.pi)

    # E = E*E0/c + E0
    # CH = CH*np.exp(((-1)*np.pi*(G/2)*T*(et*(F/2)/np.pi/((E-E0*(1+(x0)/c))**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-E0-E0*x0/c)**2/(2*S**2)))/S/np.sqrt(2*np.pi))) \
    #      + (np.pi*(G/2)*SIG)**2 / 2 * (et*(F/2)/np.pi/((E-E0*(1+(x0)/c))**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-E0-E0*x0/c)**2/(2*S**2)))/S/np.sqrt(2*np.pi))**2) \
    #     *(1 - erf( (np.pi*(G/2)*(et*(F/2)/np.pi/((E-E0*(1+(x0)/c))**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-E0-E0*x0/c)**2/(2*S**2)))/S/np.sqrt(2*np.pi))*SIG**2 - T)/np.sqrt(2)/SIG ) ) \
    #         / (1 - erf((-1)*T/SIG/np.sqrt(2)))

    CH = CH*np.exp(((-1)*np.pi*(G/2)*T*(et*(F/2)/np.pi/((E-x0)**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-x0)**2/(2*S**2)))/S/np.sqrt(2*np.pi))) \
         + (np.pi*(G/2)*SIG)**2 / 2 * (et*(F/2)/np.pi/((E-x0)**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-x0)**2/(2*S**2)))/S/np.sqrt(2*np.pi))**2) \
        *(1 - erf( (np.pi*(G/2)*(et*(F/2)/np.pi/((E-x0)**2+(F/2)**2) + (1-et)*np.exp((-1)*((E-x0)**2/(2*S**2)))/S/np.sqrt(2*np.pi))*SIG**2 - T)/np.sqrt(2)/SIG ) ) \
            / (1 - erf((-1)*T/SIG/np.sqrt(2)))

    # Be_p = np.array([0.046, 0.091, -0.249, 0.478, 0.415])
    # I1 = abs(Be_p[0]) * (2-Be_p[4])
    # I2 = abs(Be_p[0]) * Be_p[4]
    # W = abs(Be_p[3])*MulCo
    # S1 = (-1) * (Be_p[1] - Be_p[2])*MulCo + E
    # S2 = (-1) * (Be_p[1] + Be_p[2])*MulCo + E
    # FL = G0/E0*c*MulCo
    # et = 1.36603*FL/W - 0.47719*FL**2/W**2 +0.11116*FL**3/W**3
    # CH = CH*np.exp((-1)*np.pi*(G0/2/E0*c*MulCo)*(et*I1/2*W/2/np.pi/(S1**2 + (W/2)**2) \
    #                             + (1-et) * I1/2*(np.exp((-1)*(S1**2/(2*(W/2/np.sqrt(2*np.log(2)))**2)))/(W/2/np.sqrt(2*np.log(2)))/np.sqrt(2*np.pi)) \
    #                             + et*I2/2*W/2/np.pi/(S2**2 + (W/2)**2) \
    #                             + (1-et) * I2/2*(np.exp((-1)*(S2**2/(2*(W/2/np.sqrt(2*np.log(2)))**2)))/(W/2/np.sqrt(2*np.log(2)))/np.sqrt(2*np.pi)) \
    #                                   ))

    I = abs(Be_p[0])
    I1 = I * (Be_p[5] + 1) / 2 / (2 - Be_p[5])  # (2-Be_p[5]*2) / 2
    I2 = I * 3 * (1 - Be_p[5]) / 2 / (2 - Be_p[5])  # Be_p[5]*2 / 2
    WL = abs(Be_p[3]) * MulCo
    WG1 = abs(Be_p[4]) * MulCo
    WG2 = WG1 * Be_p[6]
    S1 = (-1) * (Be_p[1] - Be_p[2]) * MulCo + E
    S2 = (-1) * (Be_p[1] + Be_p[2]) * MulCo + E
    Voi1 = Voight(WL, WG1, S1)
    Voi2 = Voight(WL, WG2, S2)
    CH = CH * np.exp((-1) * np.pi * (G0 / 2 / E0 * c * MulCo) * 1 * (I1 * Voi1 + I2 * Voi2))

    CH = N * CH * 2 / (1 - (EE) ** 2)

    return(CH)

def INS(x_exp, p, JN, pool, PPP, x00 = -0.01, MulCo = 2.2, Be_p=np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])):

    E = np.linspace(-1 + 10 ** -3, 1 - 10 ** -3, JN)
    D = (E[1] - E[0])

    pn = p

    H = np.array(pool.starmap(INSt, [(x_exp, pn,  Ex, x00, MulCo, PPP, Be_p) for Ex in E])).sum(axis=0) * pn[0] * D
    return H

    # EE = np.linspace(-1 + 10 ** -4, 1 - 10 ** -4, JN)
    # N0 = p[0]
    # D = (EE[1] - EE[0])
    # H = np.array(pool.starmap(INSt, [(x_exp, p, EE, x00) for K in range(0, len(x_exp))]))*D*N0
    # return (H)




