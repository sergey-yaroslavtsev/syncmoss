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
from numpy import *
import builtins as bu
def max(*args):
    return bu.max(*args)
import numpy as np
from numba import njit


def Jac(Nn, po, x_exp, fix, confu, Nd, Expr, NExpr):
    J = np.array([[float(0)] * len(x_exp)])
    t = np.array([], dtype=int)
    # for kk in range(0, len(Expr)):  # initial expressions
    #     p[NExpr[kk]] = eval(str(Expr[kk]))

    for k in range(0, len(po)):
        if any(fix == k) == False:
            p = np.copy(po)
            step = max(10 ** -12, abs(p[k]) / 10 ** 6)
            p[k] = p[k] + step
            for kk in range(0, len(Expr)): #expression after step
                p[NExpr[kk]] = eval(str(Expr[kk]))
                Estep = p[NExpr[kk]] - po[NExpr[kk]]
                if any(confu[1] == NExpr[kk]) == True:
                    n = np.where(confu[1] == NExpr[kk])[0]
                    for kkk in range(0, len(n)):
                        p[int(confu[0][n[kkk]])] = p[int(confu[0][n[kkk]])] + Estep * confu[2][n[kkk]]
            if any(confu[1] == k) == True and any(NExpr == k) == False:
                n = np.where(confu[1] == k)[0]
                for kk in range(0, len(n)):
                    p[int(confu[0][n[kk]])] = p[int(confu[0][n[kk]])] + step * confu[2][n[kk]]
            Nup = Nn(x_exp, p)
            # Ndo = Nn(x_exp, pdo)
            Jtmp = (Nup-Nd)/step
            if all(Jtmp == 0):
                t = np.append(t, k)
                print('paramter № ' + str(k) + ' do not do anything')
            else:
                J = np.append(J, np.array([Jtmp]), axis=0)

            # p[k] = p[k] - step
            # if any(confu[1] == k) == True:
            #     n = np.where(confu[1] == k)[0]
            #     for kk in range(0, len(n)):
            #         p[int(confu[0][n[kk]])] = p[int(confu[0][n[kk]])] - step * confu[2][n[kk]]
            # p = np.copy(po)

    J = np.delete(J, 0, axis=0)
    # print(J)
    if len(J) == 0:
        print('what am I suppouse to do if everything is fixed?')
    return J, t

@njit
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

@njit
def solution(S):  # solution of linear equations system
    R = np.array([float(0)] * len(S))
    for i in range(0, len(S)):
        for j in range(i, len(S)):
            if abs(S[j][i]) > 0:
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
            if (abs(t) > 0):
                for n in range(i, len(S) + 1):
                    S[m][n] = S[m][n] - t / S[i][i] * S[i][n]
    for i in range(0, len(S)):
        R[len(S) - 1 - i] = S[len(S) - 1 - i][len(S)] / S[len(S) - 1 - i][len(S) - 1 - i]
        for j in range(0, i):
            R[len(S) - 1 - i] += -S[len(S) - 1 - i][len(S) - 1 - j] * R[len(S) - 1 - j] / S[len(S) - 1 - i][
                len(S) - 1 - i]
    # print(R)
    return (R)


def invertible_matrix(S):
    R = np.array([[float(0)] * len(S)] * len(S))
    for i in range(0, len(R)):
        R[i][i] = 1
    for i in range(0, len(S)):
        for j in range(i, len(S)):
            if abs(S[j][i]) > 0:
                k = j
                break
        if i != k:
            tmp = np.array([float(0)] * (len(S)))
            tmp2 = np.array([float(0)] * (len(S)))
            for m in range(0, len(S)):
                tmp[m] = S[i][m]
                tmp2[m] = R[i][m]
                S[i][m] = S[k][m]
                R[i][m] = R[k][m]
                S[k][m] = tmp[m]
                R[k][m] = tmp2[m]
        for m in range(i + 1, len(S)):
            t = S[m][i]
            # t2 = R[m][i]
            if (abs(t) > 0):
                for n in range(i, len(S)):
                    S[m][n] = S[m][n] - t / S[i][i] * S[i][n]
                    R[m][n] = R[m][n] - t / R[i][i] * R[i][n]

    for j in range(1, len(S)):
        for i in range(j, len(S)):
            m = S[len(S) - 1 - i][len(S) - j] / S[len(S) - j][len(S) - j]
            S[len(S) - 1 - i] = S[len(S) - 1 - i] - S[len(S) - j] * m
            R[len(R) - 1 - i] = R[len(R) - 1 - i] - R[len(R) - j] * m

    for i in range(0, len(S)):
        R[i] = R[i] / S[i][i]
    return (R)


# @njit(parallel=False)
def norm(X):
    # return np.sqrt(np.sum((np.array([X]) ** 2))) / len(X)
    return np.sqrt(np.sum(( np.array([X])/ len(X)) ** 2))

def norm2(X):
    a = float(0)
    for i in range(0, len(X)):
        a += abs(X[i])
    a = a / len(X)
    return (a)


def norm3(X):
    a = float(1)
    for i in range(0, len(X)):
        a = a * np.power(X[i], 1 / len(X))
    return (a)


def minimi_hi(Nn, x_exp, y_exp, p0, fix=np.array([], dtype=int), confu=np.array([[-1], [-1], [0]], dtype=float),
              bounds=np.array([[], []], dtype=float), Expr=[], NExpr=np.array([], dtype=int), MI=10, MI2=20, nu0=2.618, tau0=0.001, eps=10 ** -10,
              fixCH=0):  # MI - maximum iterations, NE - number of not Gaussian's parameters

    # print(PSS)
    # print(Expr)
    # print(NExpr)
    x_exp = np.array(x_exp, dtype=float)
    y_exp = np.array(y_exp, dtype=float)
    fix0 = np.copy(fix)
    ero = np.array([float(0)] * len(y_exp))
    for zc in range(0, len(y_exp)):
        ero[zc] = np.sqrt(abs(y_exp[zc]))
    nero = norm(ero)
    stop = 0
    tau = tau0
    ch1step = 0
    nu = nu0
    FixN = 0.5
    ro = float(0)
    iteration = 0
    DIF = 0
    DIF2 = 0
    SM = 0
    ch = 0
    inde = 0
    chb = np.array([int(0)] * len(p0))
    p = np.copy(p0)
    if len(bounds[0]) == 0:
        bounds = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)], dtype=float)
    if len(bounds[0]) != len(p0) and len(bounds[1]) != len(p0):
        print('error: number of bounds do not match number of parameters')
        # break
    # print(p)
    # print(bounds)
    for i in range(0, len(p)):
        if p[i] < bounds[0][i] or p[i] > bounds[1][i]:
            print('initial parameters are out of bounds (bounds are included)')
            # break
    lenX = len(x_exp)
    N_s = Nn(x_exp, p)
    # print('parameters for Jacobian ', p[0], N_s[0])
    p_tmp_J = p#np.copy(p)
    J, FT = Jac(Nn, p_tmp_J, x_exp, fix, confu, N_s, Expr, NExpr)
    Jt = J.T
    # Jt = np.array([[float(0)]*(len(J))]*(lenX))
    # for i in range(0, len(J)):
    #     for j in range(0, lenX):
    #         Jt[j][i] = J[i][j]
    A = np.matmul(J, Jt)
    # A = np.array([[float(0)]*(len(J))]*(len(J))) # лучше распаралелить, 34сек на умножение!
    # for i in range(0, (len(J))):
    #     for j in range(0, (len(J))):
    #         for k in range(0, lenX):
    #             A[i][j] += J[i][k]*Jt[k][j]
    #
    hi = y_exp - N_s
    hin = 0
    for i in range(0, len(y_exp)):
        hin += hi[i] ** 2 / (abs(y_exp[i]) + 1)
    hin = hin / (len(y_exp) - len(p))
    # print(hin)
    # print('start hi2 = ', np.sum(hi**2/y_exp)/(len(y_exp)-len(p)))
    g = np.array([float(0)] * (len(J)))
    for i in range(0, (len(J))):
        for j in range(0, lenX):
            g[i] += hi[j] * Jt[j][i]
    # print(g)
    if len(g) == 0:
        print('there is no parameters to minimize')
        stop = 1
    else:
        if max(abs(g)) < eps:
            stop = 1
    mu = np.array([float(0)] * (len(J)))
    for i in range(0, (len(J))):
        mu[i] = A[i][i] * tau
    #             mu[i] = np.amax(A)*tau
    Meth = 2
    for iteration in range(0, MI):
        # iteration += 1
        if (ch == 2 and Meth == 1) or (ch == 3 and Meth == 1):
            # print('no move is found')
            break
        if stop == 1:
            break
        ro = -1
        iteration2 = 0
        # SMo = SM
        # # if (abs(DIF2-DIF) < 0.01*DIF and abs(DIF3-DIF2) < 0.01*DIF) or (stop != 0) or (Meth == 2 and iteration!= 1):
        # if (abs(DIF2-DIF) < 0.01*DIF) or (stop != 0): # or (Meth == 2 and iteration!= 1):
        #     SM += 1
        # if SMo != SM:
        SM += 1
        if True:
            if SM % 2 == 0:  # changing mu back to Hesse diagonal
                # print('parameters for Jacobian ', p[0], N_s[0])
                p_tmp_J = p#np.copy(p)
                J, FT = Jac(Nn, p_tmp_J, x_exp, fix, confu, N_s, Expr, NExpr)
                if (len(J) == 0):
                    print('no move is found, J=0')
                    break
                mu = np.array([float(0)] * (len(J)))
                Jt = J.T
                A = np.matmul(J, Jt)
                # Jt = np.array([[float(0)] * (len(J))] * (lenX))
                # A = np.array([[float(0)] * (len(J))] * (len(J)))
                # for i in range(0, (len(J))):
                #     for j in range(0, lenX):
                #         Jt[j][i] = J[i][j]
                # for i in range(0, (len(J))):
                #     for j in range(0, (len(J))):
                #         for k in range(0, lenX):
                #             A[i][j] += J[i][k]*Jt[k][j]
                for i in range(0, (len(J))):
                    mu[i] = A[i][i] * tau
                g = np.array([float(0)] * (len(J)))
                for i in range(0, (len(J))):
                    for j in range(0, lenX):
                        g[i] += hi[j] * Jt[j][i]
                if max(abs(g)) < eps:
                    stop = 1
                # print(J)
                # print('OK')
                # print(f'{Fore.RED}{Style.BRIGHT}! changing Mu ! now it is diagonal of Hesse{Style.RESET_ALL}')
                DIF2 = 0
                ch += 1
                Meth = 2
                inde = 0

            if SM % 2 == 1:  # changing mu to identity matrix multiplied by MAX of Hesse diagonal
                for i in range(0, (len(J))):
                    mu[i] = np.amax(A) * tau
                # print(f'{Fore.RED}{Style.BRIGHT}! changing Mu ! now it is MAX from diagonal of Hesse{Style.RESET_ALL}')
                DIF2 = 0
                ch += 1
                Meth = 1
                inde = 0
                # print('OK2')
        nu = nu0
        stop = 0
        while (ro <= 0 and stop == 0 and iteration2 < MI2):
            # S = A
            # for i in range(0, len(J)):
            #     S[i][i] += mu[i]
            # S = np.insert(S, len(J), g, axis=1)
            S = np.array([[float(0)] * ((len(J)) + 1)] * (len(J)))
            for i in range(0, (len(J))):
                for j in range(0, (len(J))):
                    S[i][j] = A[i][j]
                S[i][(len(J))] = g[i]
                S[i][i] += mu[i]

            if len(S) != 0:
                R = solution(S)
            else:
                R = []
            ptmp = np.array([], dtype=float)  # questionable !
            for i in range(0, len(p)):
                if any(fix == i) == False:
                    ptmp = np.append(ptmp, p[i])
            if norm(R) < (eps ** 2) * norm(ptmp):  # maybe p instead of ptmp
                stop = 2
                # print(norm(R))
                # print(norm(ptmp))
            pn = np.copy(p)

            v = 0
            Rb = np.array([float(np.inf)] * len(p))
            for i in range(0, len(p)):
                if any(i == fix) == False and any(i == FT) == False:
                    pn[i] = p[i] + R[v]
                    v += 1
                    if pn[i] < bounds[0][i]:
                        chb[i] = -1
                    if pn[i] > bounds[1][i]:
                        chb[i] = 1
            for i in range(0, len(p)):
                if any(i == confu[0]) == True:
                    n = int(np.where(confu[0] == i)[0][0])
                    pn[i] = pn[int(confu[1][n])] * confu[2][n]
                    # if pn[i] < bounds[0][i]:
                    #     chb[i] = -1
                    # if pn[i] > bounds[1][i]:
                    #     chb[i] = 1
            if any(chb != 0):  # find coeficient to linear approximation for lower step
                for i in range(0, len(p)):
                    if bounds[0][i] != -np.inf and p[i] != pn[i] and chb[i] != 0:
                        Rb[i] = (p[i] - bounds[0][i]) / (p[i] - pn[i]) * (abs(chb[i]) - np.sign(chb[i])) / 2
                    if bounds[1][i] != np.inf and p[i] != pn[i] and chb[i] != 0:
                        Rb[i] = (p[i] - bounds[1][i]) / (p[i] - pn[i]) * (abs(chb[i]) + np.sign(chb[i])) / 2
                Rn = R * min(Rb)  # choose lowest one not to overreach through any boundary
                FixN = int(np.where(Rb == min(Rb))[0][0])
                v = 0
                for i in range(0, len(p)):
                    if any(i == fix) == False and any(i == FT) == False:
                        pn[i] = p[i] + Rn[v]  # making step lower to rich closest boundary
                        v += 1
                for i in range(0, len(p)):
                    if any(i == confu[0]) == True:
                        n = int(np.where(confu[0] == i)[0][0])
                        pn[i] = pn[int(confu[1][n])] * confu[2][n]
                pn[FixN] = bounds[int((abs(chb[i]) + np.sign(chb[i])) / 2)][FixN]

            pini = np.copy(p)
            p = pn
            for kk in range(0, len(Expr)):  # expressions
                p[NExpr[kk]] = eval(str(Expr[kk]))
                if any(confu[1] == NExpr[kk]) == True:
                    n = np.where(confu[1] == NExpr[kk])[0]
                    for kkk in range(0, len(n)):
                        p[int(confu[0][n[kkk]])] = p[int(confu[1][n[kkk]])] * confu[2][n[kkk]]
            pn = np.copy(p)
            p = pini


            ThV = Nn(x_exp, pn)
            D = y_exp - ThV

            a = float(0)
            for i in range(0, (len(J))):
                a += R[i] * (R[i] * mu[i] + g[i])
            dummy_check = norm(np.array([hi], dtype=np.float64))
            dummy_check = norm(D)
            ro = float((norm(hi) ** 2 - norm(D) ** 2) / abs(a))
            DIF = float((norm(hi)) ** 2)
            if ro <= 0:  # try to deacrease step if step is wrong
                if ((ch == 2 and Meth == 1) or (ch == 3 and Meth == 1)) and FixN != 0.5 and stop == 2:
                    fix = np.append(fix, FixN)
                    ch = 0
                    # print('ch to zero')

                mu = mu * nu
                nu = nu0 * nu
                chb = np.array([int(0)] * len(p0))  # if move do not match then forgot about boundaries overreaching
                FixN = 0.5
                inde = 1
            else:

                hi_old = hi
                hi = D
                g = np.array([float(0)] * (len(J)))
                for i in range(0, (len(J))):
                    for j in range(0, lenX):
                        g[i] += hi[j] * Jt[j][i]
                if len(g) != 0:
                    if max(abs(g)) < eps:
                        stop = 3
                    # print('estimation is small. hi2 = ', np.sum(hi ** 2 / y_exp) / (len(y_exp) - len(p)))
                    # break

                if ch1step != 0:
                    if np.sum(hi_old ** 2 / (abs(y_exp) + 1)) / (len(y_exp) - len(p) + len(fix) + len(FT)) - np.sum(
                            hi ** 2 / (abs(y_exp) + 1)) / (len(y_exp) - len(p) + len(fix) + len(FT)) < eps:
                        stop = 4
                        tau = tau0
                        nu = nu0
                        # print('change is small. hi2 = ', np.sum(hi ** 2 / y_exp) / (len(y_exp) - len(p)))
                        # break
                p = pn
                for kk in range(0, len(Expr)):  # expressions
                    p[NExpr[kk]] = eval(str(Expr[kk]))
                    if any(confu[1] == NExpr[kk]) == True:
                        n = np.where(confu[1] == NExpr[kk])[0]
                        for kkk in range(0, len(n)):
                            p[int(confu[0][n[kkk]])] = p[int(confu[1][n[kkk]])] * confu[2][n[kkk]]

                N_s = ThV

                if Meth == 2:
                    if inde == 0:
                        for i in range(0, (len(J))):
                            mu[i] = A[i][i] * tau
                        tau = tau / nu
                    if inde == 1:
                        mu = mu * nu0
                        tau = tau0
                if Meth == 1:
                    if inde == 0:
                        if abs((float(1 - (2 * ro - 1) ** 3))) < 1 / nu0:
                            mu = mu * abs(float(1 - (2 * ro - 1) ** 3))
                        else:
                            mu = mu / nu0
                    if inde == 1:
                        mu = mu * nu0
                    tau = tau0
                # DIF3 = DIF2
                DIF2 = DIF
                nu = nu0
                ro = -1
                ch1step = 1
                iteration2 += 1
                if stop != 4 and stop != 3:
                    ch = 0
                    # print('ch to zero step')

                if FixN != 0.5:  # if some element came to boundary then fix it
                    fix = np.append(fix, FixN)
                    fix = np.sort(fix)
                    # bounds[0][FixN] = -np.inf
                    # bounds[1][FixN] = np.inf
                    chb = np.array([int(0)] * len(p0))
                    FixN = 0.5
                    iteration2 = MI2
                    ch = 0
                    if SM % 2 == 0:
                        SM += 1
                    # print(fix)
                # print('got ', p)
                # print('hi2 = ', np.sum(hi**2/abs(y_exp))/(len(y_exp)-len(p)+len(fix)+len(FT)))  # norm(hi**2)/norm(y_exp))  #2*np.sum(ThV-y_exp+y_exp*np.log(y_exp/ThV)))          # np.sum((hi**2/ThV))/len(y_exp))
                # print(ch)
                # print(iteration)
                # print('hi2 = ', norm(hi)**2/norm(np.sqrt(y_exp))**2)
    # print('stop ', stop)
    # print('end hi2 = ', np.sum(hi**2/(abs(y_exp) + 1))/(len(y_exp)-len(p)+len(fix)+len(FT))) #norm(hi)**2/norm(np.sqrt(y_exp))**2)   # norm(hi**2)/norm(y_exp))        # np.sum((hi**2/ThV))/len(y_exp))
    hi2 = np.sum(hi ** 2 / (abs(y_exp) + 1)) / (len(y_exp) - len(p) + len(fix) + len(FT))
    # print(J)
    # print(Jt)
    if stop != 1:
        W = np.array([[float(0)] * (len(y_exp))] * (len(y_exp)))
        for i in range(0, len(y_exp)):
            W[i][i] = 1 / ThV[i]
        V = np.matmul(J, W)
        VV = np.matmul(V, Jt)
        try:
            # print('J')
            # print(J)
            # print('W')
            # print(W)
            # print('VV')
            # print(VV)
            VV = np.array(VV, dtype=np.float64)
            try:
                VVV = sim_inv(VV) # faster, does not work for two spectra fitting
            except:
                print('error in fast calculation of inverse matrix')
                VVV = np.linalg.inv(VV) # stable
            er = np.array([float(0)] * len(p))
            v = 0
            for i in range(0, len(p)):
                if any(fix == i) == False and any(FT == i) == False:
                    er[i] = np.sqrt(np.diag(VVV)[v])
                    v += 1
                else:
                    er[i] = None
        except np.linalg.LinAlgError: # or np.UFuncTypeError:
            print('numpy error! parameters errors could not be found')
            er = np.array([float(0)] * len(p))
        er = np.sqrt(np.sum(hi ** 2 / (abs(y_exp) + 1)) / (len(y_exp) - len(p) + len(fix) + len(FT))) * er
    else:
        er = np.array([0])
        print('there was no move')

        # er = np.sqrt(np.diag(VVV))
    # Cov = np.linalg.inv(np.matmul(J, Jt))
    # er = np.sqrt(np.diag(Cov))

    # for i in range(0, len(fix)):
    #     er = np.insert(er, int(fix[i]), None)

    # print('errors', er)
    # print('errors ', er)
    # print('or errors', np.sqrt(np.sum(hi**2/y_exp)/(len(y_exp)-len(p)) * er))

    if len(fix) != len(fix0) and fixCH == 0:
        print('one or more parameters fall to boundary, so I am going to recheck if it is true')
        print('fix is ', fix)
        er_b = np.copy(er)
        p_b = np.copy(p)

        check_idiot = 0
        fix_ch = np.setdiff1d(fix, fix0)
        for i in range(0,len(fix_ch)):
            N_f = int(fix_ch[i])
            if p_b[N_f] == bounds[0][N_f]:
                p[N_f] += max(10 ** -12, abs(p[N_f]) / 10 ** 6)
                if p[N_f] > bounds[1][N_f]:
                    check_idiot = 1
            if p_b[N_f] == bounds[1][N_f]:
                p[N_f] -= max(10 ** -12, abs(p[N_f]) / 10 ** 6)
                if p[N_f] < bounds[0][N_f]:
                    check_idiot = 1

        if check_idiot == 0:
            p, er, hi2, VVV = minimi_hi(Nn, x_exp, y_exp, p, fix0, confu, bounds, Expr, NExpr, max(int(MI / 2), 1), MI2, nu0, tau0, eps, 1)
            if all(p == p_b):
                er = er_b

    # print(J)
    # print(Jt)
    # print(Cov)
    return (p, er, hi2, VVV)





