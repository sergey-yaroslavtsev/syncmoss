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
import minimi_lib as mi
import models as m5
import multiprocessing as mp
import matplotlib.pyplot as plt
import platform
import re
import time
from constants import number_of_baseline_parameters
MulCoCMS = 0.28

def Calibration(dir_path, Cal_file, pool, VVV, INS, JN, x0, MulCo, Vel_start = 1):
    print('VVV = ', VVV)
    if VVV == 1:
        Li_Lo = 1
        pNorm = np.array([float(0)] * number_of_baseline_parameters)
        pNorm[0] = 1
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoCMS, INS, [0], [0], Met=1)[0]
        def func(x, p):
            #p = np.insert(p, 3, 0)
            #p = np.insert(p, 7, 0)
            return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, [], [], Met=1, Norm=Norm)
            # return m5.PV(x, p, model, pool)
        INS_shift = 0

    if VVV == 3:
        Li_Lo = 2
        pNorm = np.array([float(0)] * number_of_baseline_parameters)
        pNorm[0] = 1
        # JN = 32
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]
        def func(x, p):
            #p = np.insert(p, 3, 0)
            #p = np.insert(p, 7, 0)
            return m5.TI(x, p, model, JN, pool, x0, MulCo, INS, [], [], Norm=Norm)
        INS_shift = 0
        for i in range (0, int((len(INS))/3)):
            INS_shift += INS[i*3 + 1] * INS[i*3 + 2]**2


    def lin_cal(x, p):
        H1, H2 = [], []
        for i in range(0, len(x)):
            if x[i] <= int(len(xn2)/2):
                H1.append(p[0] - p[2]*x[i])
            else:
                H2.append(p[1] - p[2]*(len(xn2)-1-x[i]))
        H1 = np.array(H1)
        H2 = np.array(H2)
        # print(H1, H2)
        H = np.concatenate((H1, H2), axis=0)
        # H1 = p[0] - p[2]*x[:int(len(x)/2)]
        # H2 = p[1] - p[2]*x[:int(len(x)/2)]
        # H = np.concatenate((H2, H1[::-1]),axis=0)
        return H
    def lin_cal_fin2(x, p):
        # pCAL2 = pCAL
        # pCAL2[number_of_baseline_parameters] = p[9]
        # pCAL2[number_of_baseline_parameters + 11] = p[10]
        # H1 = p[0] - p[2] * x[:int(len(x) / 2)]
        # H2 = p[1] - p[2] * x[:int(len(x) / 2)]
        # Hx1 = np.concatenate((np.array([float(1)]*len(H1)),np.array([float(0)]*len(H2))), axis=0)
        # Hx2 = np.concatenate((np.array([float(0)]*len(H1)),np.array([float(1)]*len(H2))), axis=0)
        # Hx = np.concatenate((H2, H1[::-1]), axis=0)
        # H = p[3]*func(Hx, pCAL2) + (p[5]*(p[4]-xn2)**2 + p[6])*Hx2 + (p[8]*(p[7]-xn2)**2)*Hx1

        pCAL2 = pCAL
        pCAL2[number_of_baseline_parameters] = p[9] # intensity of main (only) component
        # pCAL2[12] = p[10] # asymmetry parameter
        H1, H2 = [], []
        for i in range(0, len(x)):
            if x[i] <= int(len(xn2) / 2):
                H1.append(p[0] - p[2] * x[i])
            else:
                H2.append(p[1] - p[2] * (len(xn2) - 1 - x[i]))
        H1 = np.array(H1)
        H2 = np.array(H2)
        Hx1 = np.concatenate((np.array([float(1)] * len(H1)), np.array([float(0)] * len(H2))), axis=0)
        Hx2 = np.concatenate((np.array([float(0)] * len(H1)), np.array([float(1)] * len(H2))), axis=0)
        Hx = np.concatenate((H1, H2), axis=0)
        if min(Hx) > -2.95 and max(Hx) < 2.95:
            # pCAL2[number_of_baseline_parameters + 6] = 0.5  # asymmetry parameter
            p[10] = 0.5 # in this case this parameter do not affect the model
        # else:
        #     pCAL2[number_of_baseline_parameters + 6] = p[10]  # asymmetry parameter
        pCAL2[number_of_baseline_parameters + 6] = p[10]  # asymmetry parameter
        pCAL2[0] = p[11]
        if p[12] > p[11]*10:
            p[12] = p[11]*10
        pCAL2[4] = p[12] # * pCAL2[0]
        H = func(Hx, pCAL2) + (p[5] * (p[4] - xn2) ** 2 + p[6]) * Hx1 + (p[8] * (p[7] - xn2) ** 2) * Hx2
        # H = func(Hx, pCAL2) + (p[5] * (p[4] - xn2) ** 2 + p[6]) * np.sign(abs(Hx-Hx2)) + (p[8] * (p[7] - xn2) ** 2) * np.sign(abs(Hx-Hx1))
        # H = p[3] * func(Hx, pCAL2) + (p[5] * (p[4] - xn2) ** 2 + p[6]) * Hx2 + (p[8] * (p[7] - xn2) ** 2) * Hx1

        return H

    def sin_cal(x, p):
        H = p[0] / np.pi * len(xn2) * np.sin(np.pi / len(xn2)) * np.cos(p[1] + np.pi / len(xn2) * (2 * x + 1)) + p[2]
        return H
    def sin_cal_fin2(x, p):
        pCAL2 = pCAL
        pCAL2[number_of_baseline_parameters] = p[9] # intensity of main component

        if VVV == 3:
            pCAL2[number_of_baseline_parameters + 11] = p[11] # intensity of impurity
        Hx = p[0] / np.pi * len(xn2) * np.sin(np.pi / len(xn2)) * np.cos(p[1] + np.pi / len(xn2) * (2 * x + 1)) + p[2]
        if min(Hx) > -2.95 and max(Hx) < 2.95:
            # pCAL2[number_of_baseline_parameters + 6] = 0.5  # asymmetry parameter
            p[10] = 0.5
            # print('very small velocity range - texture could not be defined')
        # else:
        #     pCAL2[number_of_baseline_parameters + 6] = p[10]  # asymmetry parameter
        pCAL2[number_of_baseline_parameters + 6] = p[10]  # asymmetry parameter
        if VVV == 1:
            pCAL2[0] = p[11]
            if p[12] > p[11] * 10:
                p[12] = p[11] * 10
            pCAL2[4] = p[12]  # * pCAL2[0]
        Hx1 = np.concatenate((Hx[:int(len(x)/2)],Hx[:int(len(x)/2)]), axis = 0)
        Hx2 = np.concatenate((Hx[int(len(x)/2):],Hx[int(len(x)/2):]), axis = 0)
        H = p[3]*func(Hx, pCAL2) + (p[5]*(p[4]-xn2)**2 + p[6])*np.sign(abs(Hx-Hx2)) + (p[8]*(p[7]-xn2)**2)*np.sign(abs(Hx-Hx1))
        # H = p[3]*TI(Hx, pCAL2, model, 64, -0.01, 3) + (p[5]*(p[4]-xn2)**2 + (p[8]*(p[7]+int(len(xn2)/2))**2 - p[5]*(p[4]+int(len(xn2)/2))**2))*np.sign(abs(Hx.real-Hx2)) + (p[8]*(p[7]-xn2)**2)*np.sign(abs(Hx.real-Hx1))
        return H
    def fix(y, x, p):
        Hx = np.array([float(0)]*len(x))
        for i in range(0, int(len(x) / 2)):
            Hx[i] = y[i] - p[5]*(x[i]-p[4])**2 - p[6] #- p[6]
            # Hx[i] = y[i] - p[5] * (x[i] - p[4]) ** 2 - (p[8]*(p[7]+p[0])**2 - p[5]*(p[4]+p[0])**2)
        for i in range(int(len(x) / 2), len(x)):
            Hx[i] = y[i] - p[8]*(x[i]-p[7])**2 #- p[9]
            # Hx[i] = y[i] - p[8]*(x[i]-p[7])**2 + (p[8] * (p[7] + p[0]) ** 2 - p[5] * (p[4] + p[0]) ** 2)
        return Hx
    a = np.array([0, 1.9 ,1.1, 3.4])
    value = 2.1
    idx = (np.abs(a - value)).argmin()

    PA = str(Cal_file)
    if PA[-4:] == '.mca' or PA[-5:] == '.cmca':
        LS = len(open(PA, 'r').readlines())
        with open(PA, 'r') as fi:
            id = []
            n = 0
            k = 0
            for i in range(0, LS):
                for ln in fi:
                    if ln.startswith("@A"):
                        k += 1
                    if k > n and ln.startswith("@A"):
                        id.append(re.findall(r'[\d.]+', ln[2:]))
                    if k > n and ln.startswith("#"):
                        break
                    if k > n and ln.startswith("@A") == 0:
                        id[n].extend(re.findall(r'[\d.]+', ln[0:]))
                n += 1
        if PA[-5:] == '.cmca':
            if (id[0][-1] == 0 and id[0][0] != 0) or (id[0][0] == 0 and id[0][-1] != 0):
                id_half = (id[0][-1] + id[0][0]) / 2
                id[0][-1] = id_half
                id[0][0] = id_half
        if len(id[0])%2 == 1:
            id[0] = id[0][:-1]
        id = np.array(id, dtype=float)

        xn = np.linspace(0, len(id[0]) / 2 - 1, int(len(id[0]) / 2))
        xn2 = np.linspace(0, len(id[0]) - 1, int(len(id[0])))
        y1 = id[0][:int(len(id[0]) / 2)]
        y2 = id[0][int(len(id[0]) / 2):]
        y_ch = y1 * 2

    if PA[-4:] == '.ws5' or PA[-4:] == '.w98' or PA[-4:] == '.moe' or PA[-3:] == '.m1' or PA[-4:] == '.mcs' or PA[-4:] == '.Mcs':
        if PA[-4:] == '.mcs' or PA[-4:] == '.Mcs':
            f = open(PA, mode='rb')
            id = []
            entete1 = f.read(256)
            array = np.fromfile(f, dtype=np.uint32)
            print(len(array))
            id.append(array)
            f.close()
        else:
            with open(PA, 'r') as catalog:
                id = []
                id.append([])
                # id[0].append(float(1))
                # id[0].append(float(2))
                # print(id)
                k = 0
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                for line in lines:
                    if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                        if PA[-3:] == '.m1':
                            while '  ' in line:
                                line = line.replace('  ', ' ')
                            column = line.split(' ')
                            x = float(column[4])
                            if k > 0:
                                id[0].append(x)
                            k += 1
                        else:
                            column = line.split()
                            x = float(column[0])
                            if PA[-4:] != '.moe' or not ('.' in str(column[0])):
                                id[0].append(x)
                        k += 1
            # if len(id[0]) % 2 == 1:
            #     id[0] = id[0][:-1]
        id = np.array(id, dtype=float)

        xn = np.linspace(0, len(id[0]) / 2 - 1, int(len(id[0]) / 2))
        xn2 = np.linspace(0, len(id[0]) - 1, int(len(id[0])))
        y1 = id[0][:int(len(id[0]) / 2)]
        y2 = id[0][int(len(id[0]) / 2):]
        if len(y2) > len(y1):
            y2 = y2[1:]
        y_ch = y1 * 2


    # detecting dephase
    chi2_tmp = np.sum(np.abs(y1-y2[::-1]))/len(y1)
    phase0 = 0
    for i in range(1, int(len(y1)/2)):
        chi2_tmp_n = np.sum(np.abs(y1[i:]-y2[i:][::-1]))/len(y1[i:])
        if chi2_tmp_n <= chi2_tmp:
            chi2_tmp = chi2_tmp_n
            phase0 = -i
        # else:
        #     break
    if phase0 == 0:
        for i in range(1, int(len(y1)/2)):
            chi2_tmp_n = np.sum(np.abs(y1[:-i] - y2[:-i][::-1])) / len(y1[:-i])
            if chi2_tmp_n <= chi2_tmp:
                chi2_tmp = chi2_tmp_n
                phase0 = i
            # else:
            #     break
    phase0 = - phase0 * 2
    print('PHASE  ', phase0)


    model = ['Sextet']
    p00 = np.array([(max(id[0]) - 2 * np.sqrt(max(id[0])))*(1-0.4*(VVV==1)), 0, 0, 0, (max(id[0]) - 2 * np.sqrt(max(id[0])))*(0.4*(VVV==1)), 0, 0, 0, 8, 0.0, 0, 33.04, 0.098, 0.0, 0.5, 0, 0, 0, 3])
    sex0 = np.array([-5.3123, -3.0760, -0.8397, 0.8397, 3.0760, 5.3123])
    bounds = np.array([[-np.inf] * len(p00), [np.inf] * len(p00)], dtype=float)
    bounds[0][number_of_baseline_parameters+1] = -0.05
    bounds[1][number_of_baseline_parameters+1] = 0.05
    bounds[0][number_of_baseline_parameters+3] = 32.54
    bounds[1][number_of_baseline_parameters+3] = 33.54
    if Vel_start == 1:
        sex0 = sex0[::-1]

    start_time = time.time()
    if VVV == 3 or VVV == 1:
        ch_m = int(np.where(id[0] == min(id[0][5:-5]))[0][0])
        # ch_m1 = int(np.where(id[0] == min(id[0][:int(len(id[0]) / 4)]))[0][0])
        # ch_m2 = int(np.where(id[0] == min(id[0][int(len(id[0]) / 4):int(len(id[0]) / 2)]))[0][0])
        ch_m1_arr = (np.where(id[0] == min(id[0][:int(len(id[0]) / 4)]))[0])
        ch_m1 = int(ch_m1_arr[-1])
        for i in range(0, len(ch_m1_arr)):
            if ch_m1_arr[i] < int(len(id[0]) / 4):
                ch_m1 = int(ch_m1_arr[i])
                # print(ch_m1)
                break
        ch_m2_arr = (np.where(id[0] == min(id[0][int(len(id[0]) / 4):int(len(id[0]) / 2)]))[0])
        ch_m2 = int(ch_m2_arr[-1])
        for i in range(0, len(ch_m2_arr)):
            if ch_m2_arr[i] >= int(len(id[0]) / 4) and ch_m2_arr[i] < int(len(id[0]) / 2):
                ch_m2 = int(ch_m2_arr[i])
                # print(ch_m2)
                break
        hi_m = np.array([float(10000)] * 30)
        hi_c = 10000
        hi_sin = hi_c
        k = 0
        print('channels of minimum ', ch_m1, ch_m2)
        for i in range(0, 6):
            for j in range(i + 1, 6):
                # sex0[i] = p[0] / np.pi * len(xn2) * np.sin(np.pi / len(xn2)) * np.cos(np.pi / len(xn2) * (2 * ch_m1 + 1)) + p[2]
                # sex0[j] = p[0] / np.pi * len(xn2) * np.sin(np.pi / len(xn2)) * np.cos(np.pi / len(xn2) * (2 * ch_m2 + 1)) + p[2]
                Vel_max_m = (sex0[i] - sex0[j]) * np.pi / len(xn2) / np.sin(np.pi / len(xn2)) / (np.cos(np.pi / len(xn2) * (2 * ch_m1 + 1)) - np.cos(np.pi / len(xn2) * (2 * ch_m2 + 1)))
                # print((sex0[i] - sex0[j]), len(xn2), np.cos(np.pi / len(xn2)))
                shift_m = sex0[i] - Vel_max_m / np.pi * len(xn2) * np.sin(np.pi / len(xn2)) * np.cos(np.pi / len(xn2) * (2 * ch_m1 + 1) + (np.pi / len(xn2) * phase0))
                x = sin_cal(xn2, [Vel_max_m, (np.pi / len(xn2) * phase0), shift_m])  # p[1] mean different on method
                # A and I1/I3 are 12th and 16th parameters.
                res = mi.minimi_hi(func, x, id[0], p00, fix=np.array([1, 2, 3, 1+int(VVV), 5, 6, 7, number_of_baseline_parameters+2, number_of_baseline_parameters+4, number_of_baseline_parameters+5, number_of_baseline_parameters+6, number_of_baseline_parameters+7, number_of_baseline_parameters+8, number_of_baseline_parameters+9, number_of_baseline_parameters+10]), bounds=bounds, MI=3, MI2=5) # int(11*(VVV==3)+1), int(15*(VVV==3)+1)
                # if res[2] < 10000:
                hi_m[k] = res[2]
                hi_c = hi_m[k]
                # print(Vel_max_m, shift_m, hi_c)
                if all(hi_m >= hi_c) == True:
                    Vel_max_sin = Vel_max_m
                    shift = shift_m
                    p_sin = res[0]
                    hi_sin = hi_c
                k += 1
        print(Vel_max_sin,shift, hi_sin)
        x_sin = sin_cal(xn2, [Vel_max_sin, (np.pi / len(xn2) * phase0), shift])

        # fig = plt.figure(dpi=300)
        # plt.plot(x_sin, id[0])
        # plt.plot(x_sin, func(x_sin, p_sin))
        # # plt.show()
        # fig.savefig(r"C:\Users\yaroslav\Downloads\test_sin.png", bbox_inches='tight')
        # plt.close()

        method = 0
        x = x_sin
        p0 = p_sin
        Vel_max = Vel_max_sin

        def cal(x, p):
            return sin_cal(x, p)

        def cal_fin2(x, p):
            return sin_cal_fin2(x, p)
    print('first sin takes', time.time() - start_time, 'seconds')

    if VVV == 1:
        phase0 = phase0/2
        # ch_m1 = int(np.where(id[0] == min(id[0][:int(len(id[0])/4)]))[0][0])
        # ch_m2 = int(np.where(id[0] == min(id[0][int(len(id[0])/4):int(len(id[0])/2)]))[0][0])
        ch_m1_arr = (np.where(id[0] == min(id[0][:int(len(id[0]) / 4)]))[0])
        ch_m1 = int(ch_m1_arr[-1])
        for i in range(0, len(ch_m1_arr)):
            if ch_m1_arr[i] < int(len(id[0]) / 4):
                ch_m1 = int(ch_m1_arr[i])
                print(ch_m1)
                break
        ch_m2_arr = (np.where(id[0] == min(id[0][int(len(id[0]) / 4):int(len(id[0]) / 2)]))[0])
        ch_m2 = int(ch_m2_arr[-1])
        for i in range(0, len(ch_m2_arr)):
            if ch_m2_arr[i] >= int(len(id[0]) / 4) and ch_m2_arr[i] < int(len(id[0]) / 2):
                ch_m2 = int(ch_m2_arr[i])
                print(ch_m2)
                break
        hi_m_lin = np.array([float(10000)] * 30)
        hi_c = 10000
        hi_lin = hi_c
        k = 0
        for i in range(0, 6):
            for j in range(i+1, 6):
                velocity_step_m = -(sex0[j]-sex0[i])/(ch_m2-ch_m1)
                Vel_max_m = sex0[i] + velocity_step_m*ch_m1
                x = lin_cal(xn2, [Vel_max_m, Vel_max_m-velocity_step_m*phase0, velocity_step_m])  # p[1] mean different on method
                res = mi.minimi_hi(func, x, id[0], p00, fix=np.array([1, 2, 3, 1+int(VVV), 5, 6, 7, number_of_baseline_parameters+2, number_of_baseline_parameters+4, number_of_baseline_parameters+5, number_of_baseline_parameters+6, number_of_baseline_parameters+7, number_of_baseline_parameters+8, number_of_baseline_parameters+9, number_of_baseline_parameters+10]), bounds=bounds, MI=3, MI2=5)
                # if res[2] < 10000:
                hi_m_lin[k] = res[2]
                hi_c = hi_m_lin[k]
                #print(Vel_max_m, velocity_step_m, hi_m_lin[k])
                if all(hi_m_lin >= hi_c) == True:
                    Vel_max_lin = Vel_max_m
                    velocity_step = velocity_step_m
                    p_lin = res[0]
                    hi_lin = hi_c
                k += 1
        print(Vel_max_lin,velocity_step, hi_lin)
        x_lin = lin_cal(xn2, [Vel_max_lin, Vel_max_lin-velocity_step*phase0, velocity_step])
        phase0 = phase0 * 2
        # fig = plt.figure(dpi=300)
        # plt.plot(x_lin, id[0])
        # plt.plot(x_lin, func(x_lin, p_lin))
        # # plt.show()
        # # "C:\Users\yaroslav\Downloads"
        # fig.savefig(r"C:\Users\yaroslav\Downloads\test_lin.png", bbox_inches='tight')
        # plt.close()

        method = 1
        x = x_lin
        p0 = p_lin
        Vel_max = Vel_max_lin

        def cal(x, p):
            return lin_cal(x, p)

        def cal_fin2(x, p):
            return lin_cal_fin2(x, p)

    if VVV == 1:
        if hi_sin <= hi_lin: #  and shift < 1.0
            method = 0
            x = x_sin
            p0 = p_sin
            Vel_max = Vel_max_sin
            def cal(x, p):
                return sin_cal(x, p)
            def cal_fin2(x, p):
                return sin_cal_fin2(x, p)
            print('sinus mode')
        else:
            method = 1
            x = x_lin
            p0 = p_lin
            Vel_max = Vel_max_lin
            phase0 = phase0 / 2
            def cal(x, p):
                return lin_cal(x, p)
            def cal_fin2(x, p):
                return lin_cal_fin2(x, p)
            print('triangular mode')

    p0[number_of_baseline_parameters+1] = 0
    p0[number_of_baseline_parameters+3] = 33.04
    print('method ', method)
    print(p0)
    # x = sin_cal(xn2, [Vel_max, 0 * np.pi / len(xn2), 0.68])
    # x = sin_cal(xn2, [Vel_max, 0 * np.pi / len(xn2), 0.68])  # p[1] mean different on method
    p00 = np.copy(p0)
    start_time = time.time()
    for i in range(0, 4):
        p = mi.minimi_hi(func, x, id[0], p0, fix=np.array([1, 2, 3, 1+int(VVV), 5, 6, 7, number_of_baseline_parameters+2, number_of_baseline_parameters+3, number_of_baseline_parameters+4, number_of_baseline_parameters+6, number_of_baseline_parameters+7, number_of_baseline_parameters+8, number_of_baseline_parameters+9, number_of_baseline_parameters+10]), MI=3, MI2=5)[0] # int(15*(VVV==3)+1)  int(11*(VVV==3)+1)
        # print(p)
        sex0 = np.array([-5.3123, -3.0760, -0.8397, 0.8397, 3.0760, 5.3123, -5.3123, -3.0760, -0.8397, 0.8397, 3.0760, 5.3123])
        if Vel_start == 1:
            sex0[::-1]
        ps0x = np.array([float(0)]*12)
        V = number_of_baseline_parameters
        x1 = x[:int(len(x)/2)]
        x2 = x[int(len(x)/2):]
        p[V+3] = p[V+3] / 3.101
        ps0x[0]  = (np.abs(x1 - (p[V + 1] - p[V + 3] / 2 + p[V + 2]) - p[V + 7])).argmin()
        ps0x[1]  = (np.abs(x1 - (p[V + 1] - 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 8])).argmin()
        ps0x[2]  = (np.abs(x1 - (p[V + 1] - 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 8])).argmin()
        ps0x[3]  = (np.abs(x1 - (p[V + 1] + 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 8])).argmin()
        ps0x[4]  = (np.abs(x1 - (p[V + 1] + 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 8])).argmin()
        ps0x[5]  = (np.abs(x1 - (p[V + 1] + p[V + 3] / 2 + p[V + 2]) + p[V + 7])).argmin()
        ps0x[6]  = int(len(x)/2) + (np.abs(x2 - (p[V + 1] - p[V + 3] / 2 + p[V + 2]) - p[V + 7])).argmin()
        ps0x[7]  = int(len(x)/2) + (np.abs(x2 - (p[V + 1] - 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 8])).argmin()
        ps0x[8]  = int(len(x)/2) + (np.abs(x2 - (p[V + 1] - 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 8])).argmin()
        ps0x[9]  = int(len(x)/2) + (np.abs(x2 - (p[V + 1] + 0.8397 / 5.3123 * p[V + 3] / 2 - p[V + 2]) + p[V + 8])).argmin()
        ps0x[10] = int(len(x)/2) + (np.abs(x2 - (p[V + 1] + 3.0760 / 5.3123 * p[V + 3] / 2 - p[V + 2]) - p[V + 8])).argmin()
        ps0x[11] = int(len(x)/2) + (np.abs(x2 - (p[V + 1] + p[V + 3] / 2 + p[V + 2]) + p[V + 7])).argmin()
        p[V + 3] = p[V + 3] * 3.101
        j = 0
        for k in range(0, 12):
            # if ps0x[j] == 0 or ps0x[j] == len(x)-1 or ps0x[j] == len(x1)-1 or ps0x[j] == len(x1) or ps0x[j] == len(x1)-2 or ps0x[j] == len(x) - 2:
            if ps0x[j] < 5 or (ps0x[j] > len(x1)-1-5 and ps0x[j] < len(x1)+5) or ps0x[j] > len(x) - 1 - 5:
                ps0x = np.delete(ps0x, j)
                sex0 = np.delete(sex0, j)
                j = j-1
            j += 1
        # print(ps0x)
        if i == 0:
            if method == 0:
                ps = mi.minimi_hi(sin_cal, ps0x, sex0, p0=np.array([Vel_max, (np.pi / len(xn2) * phase0), shift]), tau0=1)[0]
            if method == 1:
                ps = mi.minimi_hi(lin_cal, ps0x, sex0, p0=np.array([Vel_max, Vel_max-velocity_step*phase0, velocity_step]), tau0=1)[0]
            # print(ps)
        else:
            ps = mi.minimi_hi(cal, ps0x, sex0, p0=ps, eps = 10**-40)[0]
            # print(ps)
        x = cal(xn2, ps)
        p0 = np.copy(p)
        p0[7] = 0
    print('first ps takes', time.time() - start_time, 'seconds')
    # plt.figure(dpi=300)
    # plt.plot(x, id[0])
    # plt.plot(x, func(x, p0))
    # plt.show()
    # plt.close()

    if VVV == 3:
        JN = 64
        Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]

    pCAL = np.array([p[0], 0, 0, 0, p[3], 0, 0, 0, 8.2, 0, 0, 33.04, 0.098, 0.0, 0.5, 0, 0, 0, 3])
    ps1 = np.append(ps, 1)

    def parabolic(x, p):
        H = p[1]*(p[0]-x)**2+p[2]
        return H
    y2 = y2[::-1]
    start_time = time.time()
    parab = mi.minimi_hi(parabolic, xn2[:int(len(xn2)/2)], y1-y2+y1[0], p0=[int(len(xn2)/4), 5, y1[int(len(y1) / 2)]-y2[int(len(y2) / 2)]+y1[0]])[0]
    print('parabola takes', time.time() - start_time, 'seconds')
    parab[2] = parab[2] - y1[0]
    y2 = y2[::-1]

    ps1[-1] = 1
    ps1 = np.append(ps1, parab[0])
    ps1 = np.append(ps1, parab[1] / 2)
    ps1 = np.append(ps1, parab[2])
    ps1 = np.append(ps1, parab[0] + int(len(xn2)/2))
    ps1 = np.append(ps1, (-1)*parab[1] / 2)

    # test_par1 = np.array([parab[0]                 , parab[1] / 2, parab[2]])
    # test_par2 = np.array([parab[0]+ int(len(xn2)/2),-parab[1] / 2, 0])
    # fig = plt.figure(dpi=300)
    # plt.plot(xn2[:int(len(xn2)/2)], p0[0]+p0[3] + parabolic(xn2[:int(len(xn2)/2)], test_par1), 'r')
    # plt.plot(xn2[int(len(xn2)/2):], p0[0]+p0[3] + parabolic(xn2[int(len(xn2)/2):], test_par2), 'r')
    # plt.plot(xn2, id[0], 'm')
    # fig.savefig(r"C:\Users\yaroslav\Downloads\test_parab.png", bbox_inches='tight')
    # plt.close()

    if method == 0:
        if VVV == 3:
            model = ['Sextet', 'Sextet', 'Doublet']

            try:
                Be_param = np.genfromtxt(str(dir_path) + str('\\\\Be.txt')*(platform.system() == 'Windows') + str('/Be.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
                print('file was read')
            except:
                Be_param = np.array([0.057, 0.066, -0.261, 0.098, 0.375, 0.772, 1])
                print('COULD NOT READ Be.txt')
            pCAL = np.array([p[0], 0, 0, 0, 0, 0, 0, 0, 8.08, 0, 0, 33.04, 0.098, 0, 0.5, 0, 0, 0, 3, 0.451, -0.041, 0.003, 30.88, 0.098, 0.1, 0.5, 0, 0, 0, 3])#, 0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
            pCAL = np.concatenate((pCAL, Be_param))
        if VVV == 1:
            # model = ['Sextet', 'Sextet']
            # pCAL = np.array([p[0], 0, 0, p[3], 0, 0, 8.08, 0, 0, 33.04, 0.098, 0, 0.5, 0, 0, 0, 3, 0.0, -0.041, 0.003, 30.88, 0.098, 0.1, 0.5, 0, 0, 0, 3])#, 0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
            model = ['Sextet']
            pCAL = np.array([p[0], 0, 0, 0, p[3], 0, 0, 0, 8.08, 0, 0, 33.04, 0.098, 0, 0.5, 0, 0, 0, 3])
            print('background ', pCAL[0], pCAL[4], ps1[3])

    if method == 1:
        model = ['Sextet']
        pCAL = p0
        pCAL[number_of_baseline_parameters+10] = 3
        pCAL[number_of_baseline_parameters+5] = 0
        print('background ', pCAL[0], pCAL[4], ps1[3])


    xT = cal(xn2, ps1)

    ps1 = np.append(ps1, 8)
    ps1 = np.append(ps1, 0.5)
    if VVV == 3:
        ps1 = np.append(ps1, 0.5)
    if VVV == 1:

        tot = (pCAL[0] + pCAL[4])
        ps1[9] = pCAL[number_of_baseline_parameters] * pCAL[0] / tot / 3 * 5
        pCAL[0] = tot * 0.6
        pCAL[4] = tot * 0.4
        ps1 = np.append(ps1, pCAL[0])
        ps1 = np.append(ps1, pCAL[4])
        print('baseline ', ps1[11], ps1[12])

    print('parameter set after preliminary fit ', ps1)
    print('model parameteres after preliminary fit ', pCAL)
    time.sleep(1)

    # tmpy = cal_fin2(xn2, ps1)
    # fig = plt.figure(dpi=300)
    # plt.plot(xn2, tmpy, 'r')
    # plt.plot(xn2, id[0], 'm')
    # plt.plot(xn2, id[0] - tmpy + min(id[0]) - max(id[0] - tmpy), 'b')
    # fig.savefig(r"C:\Users\yaroslav\Downloads\test_calibr1.png", bbox_inches='tight')
    # plt.close()


    start_time = time.time()
    res = mi.minimi_hi(cal_fin2, xn2, (id[0]), p0=ps1, MI=20, MI2=20, eps=10**-6)
    if abs(res[0][0]) < 2.95:
        print('very small velocity range - texture could not be defined')
    print('model parameteres ', pCAL)
    print('variable parameters ', res[0])
    print('hi2 ', res[2])
    print('main minimization takes', time.time() - start_time, 'seconds')
    # print('hihi')
    time.sleep(2)
    pS = res[0]
    hi2 = res[2]
    xT = cal(xn2, pS)

    # tmpy = cal_fin2(xn2, res[0])
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax2 = ax.twinx()
    # ax.plot(xn2, tmpy, 'r')
    # ax.plot(xn2, id[0], 'm')
    # ax.plot(xn2, id[0]-tmpy + min(id[0])-max(id[0]-tmpy), 'b')
    # ax2.plot(xn2, xT, 'y')
    # ax2.plot([xn2[0], xn2[-1]], [pS[0], pS[0]], 'cyan')
    # ax2.plot([xn2[0], xn2[-1]], [pS[1], pS[1]], 'lime')
    # ax2.plot([xn2[0], xn2[-1]], [pS[0] - int(len(xn2) / 2) * pS[2], pS[0] - int(len(xn2) / 2) * pS[2]], 'cyan')
    # ax2.plot([xn2[0], xn2[-1]], [pS[1] - (int(len(xn2)) / 2 -1) * pS[2], pS[1] - (int(len(xn2) / 2)-1) * pS[2]], 'lime')
    # fig.savefig(r"C:\Users\yaroslav\Downloads\test_calibr2.png", bbox_inches='tight')
    # plt.close()
    # print(xT)


    pCAL[0] = pCAL[0] * pS[3]
    if VVV == 1:
        pCAL[4] = pCAL[4] * pS[3]
        # pS[12] = pS[12] * pS[3]
    pS[3] = 1

    spc = fix(id[0], xn2, pS)
    tmp = cal_fin2(xn2, pS)

    if method == 0:
        min1 = np.abs(xT[:int(len(xT) / 2)]-pS[2]).argmin()
        min2 = np.abs(xT[int(len(xT) / 2):] - xT[min1]).argmin() + int(len(xT)/2)

        if min1 == (len(xT)-1-min2):
            n1, n2 = 0, len(xT)-1
        elif min1 > (len(xT)-1-min2):
            n1, n2 = min1 - (len(xT)-min2) + 1, len(xT) - 1
        elif min1 < (len(xT)-1-min2):
            n1, n2 = 0, min2+min1

        x_1h = (xT[:int(n1 / 2)] + xT[n1 - int(n1 / 2):n1][::-1]) / 2
        spc_1h = id[0][:int(n1 / 2)] + id[0][n1 - int(n1 / 2):n1][::-1]
        fit_1h = tmp[:int(n1 / 2)] + tmp[n1 - int(n1 / 2):n1][::-1]
        delt_1h = id[0][:int(n1 / 2)] - id[0][n1 - int(n1 / 2):n1][::-1]

        x_2h = (xT[n2+1:n2 + int((len(xT) - 1 - n2) / 2)+1][::-1] + xT[len(xT) - int((len(xT) - 1 - n2) / 2):len(xT)]) / 2
        spc_2h = id[0][n2+1:n2 + int((len(xT) - 1 - n2) / 2)+1][::-1] + id[0][len(xT) - int((len(xT) - 1 - n2) / 2):len(xT)]
        fit_2h = tmp[n2+1:n2 + int((len(xT) - 1 - n2) / 2)+1][::-1] + tmp[len(xT) - int((len(xT) - 1 - n2) / 2):len(xT)]
        delt_2h = id[0][n2+1:n2 + int((len(xT) - 1 - n2) / 2)+1][::-1] - id[0][len(xT) - int((len(xT) - 1 - n2) / 2):len(xT)]

        x_3h = (xT[n1:n1 + int((n2 - n1 + 1) / 2)] + xT[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]) / 2
        spc_3h = id[0][n1:n1 + int((n2 - n1 + 1) / 2)] + id[0][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
        fit_3h = tmp[n1:n1 + int((n2 - n1 + 1) / 2)] + tmp[n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
        delt_3h = id[0][n1:n1 + int((n2 - n1 + 1) / 2)] - id[0][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]

        delt_4h = id[0][n1+1:n1 + int((n2 - n1 + 1) / 2)+1] - id[0][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]

        delt_5h = id[0][n1:n1 + int((n2 - n1 + 1) / 2)] - id[0][n2 - int((n2 - n1 + 1) / 2):n2][::-1]

        # plt.figure(dpi=300)
        # plt.plot(x_3h, delt_3h, color='r')
        # plt.plot(x_3h, delt_4h, color='b')
        # plt.plot(x_3h, delt_5h, color='m')
        # plt.show()

        xT2 = np.concatenate((np.concatenate((x_1h, x_2h)), x_3h))
        spc2 = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
        spc3 = np.concatenate((np.concatenate((fit_1h, fit_2h)), fit_3h))
        delt = np.concatenate((np.concatenate((delt_1h, delt_2h)), delt_3h))
        print(n1, n2, len(xT2))
        # plt.figure(dpi=300)
        # plt.plot(xT2, spc2)
        # plt.show()

    if method == 1:
        if pS[0] * pS[2] == pS[1] * pS[2]:
            min1 = 0
            min2 = int(len(xT))-1
        elif pS[0] * pS[2] > pS[1] * pS[2]:
            min1 = np.abs(xT[:int(len(xT) / 2)] - pS[1]).argmin()
            min2 = int(len(xT))-1
        elif pS[0] * pS[2] < pS[1] * pS[2]:
            min1 = 0
            min2 = np.abs(xT[int(len(xT) / 2):] - pS[0]).argmin() + int(len(xT)/2)
        n1, n2 = min1, min2
        n_sh = 2 * (n1 + (int(len(xT))-1-n2))

        # xT2 = np.array([float(0)] * int((n2-n1+1) / 2))
        # spc2 = np.array([float(0)] * int((n2-n1+1) / 2))
        # spc3 = np.array([float(0)] * int((n2-n1+1) / 2))
        # delt = np.array([float(0)] * int((n2-n1+1) / 2))
        # for i in range(0, int((n2-n1+1) / 2)): # for i in range(0, int(len(xT) / 2) - min1):
        #     xT2[i] = (xT[n1 + i] + xT[n2 - i]) / 2
        #     spc2[i] = id[0][n1 + i] + id[0][n2 - i]
        #     spc3[i] = tmp[n1 + i] + tmp[n2 - i]
        #     delt[i] = id[0][n1 + i] - id[0][n2 - i]

        xT2 = np.array([float(0)] * int((len(xT) - n_sh)/2)) # int((len(xT) - n_sh)/2) - to take into account odd number of points # (511 - 2)/2 = 254,5 -> int = 254
        spc2 = np.array([float(0)] * int((len(xT) - n_sh)/2))
        spc3 = np.array([float(0)] * int((len(xT) - n_sh)/2))
        delt = np.array([float(0)] * int((len(xT) - n_sh)/2))
        for i in range(0, int((len(xT) - n_sh)/2)):
            xT2[i] = (xT[n1 + i] + xT[n2 - i]) / 2
            spc2[i] = id[0][n1 + i] + id[0][n2 - i]
            spc3[i] = tmp[n1 + i] + tmp[n2 - i]
            delt[i] = id[0][n1 + i] - id[0][n2 - i]

    xT2 = xT2 + INS_shift

    if VVV == 3: #method == 0
        # plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
        # plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
        # plt.rcParams['axes.labelcolor'] = 'w'
        # plt.rcParams['axes.edgecolor'] = 'w'
        # plt.rcParams['xtick.color'] = 'w'
        # plt.rcParams['ytick.color'] = 'w'
        fig, ax = plt.subplots(dpi=300)
        plt.plot(xn2, (id[0]), 'm')
        plt.plot(xn2, cal_fin2(xn2, pS), 'b')
        plt.plot(xn2, id[0] - tmp + 0.9 * min(id[0]), 'lime')
        plt.plot(xn2[:int(len(xn2)/2)]*2, (id[0] + max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0])))[:int(len(id[0])/2)][::-1], 'r', linestyle = 'None', marker = 'o', markersize = 2)
        plt.plot(xn2[:int(len(xn2)/2)]*2, (id[0] + max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0])))[int(len(id[0])/2):], 'yellow', linestyle = 'None', marker = 'o', markersize = 2)
        sax = ax.twiny()
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel('channel', color='m')
        sax.set_xlabel('Velocity, mm/s', color='r')
        sax.plot(xT[:int(len(spc)/2)], (spc + 2*(max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0]))))[:int(len(spc)/2)], 'r', linestyle = 'None', marker = 'o', markersize = 2)
        sax.plot(xT[int(len(spc)/2):], (spc + 2*(max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0]))))[int(len(spc)/2):], 'yellow', linestyle = 'None', marker = 'o', markersize = 2)
        sax.plot(xT2, delt + 4*(max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0]))), 'lime')
        sax.plot(xT2, spc2/2 + 2*(max(id[0]) - min(spc) + 10 * np.sqrt(max(id[0]))), 'm', marker = 'o', markersize = 1)



        ax.text(0, max(id[0])+4*np.sqrt(max(id[0])), 'Va = %.3f mm/s' % pS[0], color='w', fontsize=8)
        ax.text(len(xn2)/2, max(id[0]) + 4 * np.sqrt(max(id[0])), 'ΔN0 = %.1f ' %(abs(pS[6]) /  (pCAL[0] + pS[6]*(1+np.sign(pS[6])) / 2) * 100) +'%', color='w', fontsize=8, horizontalalignment='center')
        ax.text(len(xn2), max(id[0]) + 4 * np.sqrt(max(id[0])), 'impurity %.1f ' %(pS[11]/(pS[9]+pS[11])*100) +'%', color='w', fontsize=8, horizontalalignment='right')

        ax.text(len(xn2)/2, max(id[0]) + 10 * np.sqrt(max(id[0])), str('lin ')*(method==1) + str('sin ')*(method==0) + str(n1) + str(' ') + str(n2), color='r', fontsize=8, horizontalalignment='center')

        fig.savefig('calibr.png', bbox_inches='tight')
        plt.close()

        print('Shift due to instrumental function ', INS_shift)
        print('Velocity range: ', min(xT2), ' to ', max(xT2), 'mm/s, absolute shift ', (max(xT2)+min(xT2))/2  )
        print('Velocity amplitude is', pS[0], 'mm/s')
        print('Source shift is', pS[2], 'mm/s')
        print('difference of N0:', abs(pS[6]) /  (pCAL[0] + pS[6]*(1+np.sign(pS[6])) / 2) * 100, '%')
        print('impurity is', pS[11]/(pS[9]+pS[11])*100, '%')
        print('maximum deviation is ', max(np.abs(spc2-spc3)), 'or ',  max(np.abs(spc2-spc3))/pCAL[0], '%')
        # if VVV == 3:
        print('Texture parameter is', pS[10], 'should be 0.5 for isotropic')


    if platform.system() == 'Windows':
        rpath = str(dir_path) + str('\\\\Calibration.dat')
    else:
        rpath = str(dir_path) + str('/Calibration.dat')

    f = open(rpath, "w")
    f.write(str('#') + '\t' + str('lin ')*(method==1) + str('sin ')*(method==0) + '\t' + str(n1) + '\t' + str(n2) + '\n')
    for i in range(0, int(len(xT2))):
        f.write(str(xT2[i]) + '\t' + str(spc2[i]) + '\n')
    f.write('\n')
    f.close()

    pCAL2 = pCAL
    pCAL2[number_of_baseline_parameters] = pS[9]
    pCAL2[number_of_baseline_parameters + 8] = pS[11]

    return(xT2, spc2, spc3)
