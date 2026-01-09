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
# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import signal
import os
import re
import os.path
import numpy as np
from numpy import *
import builtins as bu
def max(*args):
    return bu.max(*args)
def min(*args):
    return bu.min(*args)
import minimi_lib as mi
import models as m5
import models_positions as modpos 
import models_NFS as mN
import Calibration as cal
import multiprocessing as mp
# import dual_v3 as dn
import Instrumental as ins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.image
from matplotlib import colors
import threading
import queue
import time
from functools import partial
import shutil
import platform
# import asyncio
# import psutil
import base64
import warnings
import copy
from multiprocessing.pool import ThreadPool
import gc

#import logging
#logging.getLogger('matplotlib.font_manager').disabled = True
#logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import logging
logging.getLogger('matplotlib').propagate=False
logging.getLogger('numba').propagate=False
# logger = logging.getLogger(__name__)
# logger.setLevel('ERROR')
# logging.ERROR

check_tango = False
# from Pytango import DeviceProxy
# def get_data(tango_uri):
#     proxy = DeviceProxy(tango_uri)
#     return proxy.data
# tango_uri = 'moesa:20000/id14/Can556/6a2' # could be different
# check_tango = True


warnings.filterwarnings('ignore', '.*object is not callable.*', )

plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
plt.rcParams['axes.labelcolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'w'
plt.rcParams['xtick.color'] = 'w'
plt.rcParams['ytick.color'] = 'w'

path = None
params = None
image = None
initial = 0
# global RA
RA = False
RR = False
RL = 3
RA2 = threading.Event()
tic = 1
tac = 1
Rpa = 0
Rlog = ''
RlogCol = []
table_save = ''
L0_text = ''
MulCoCMS=0.28
MulCoL2 = 0.3

RM = 1
CurMethod = 'SMS'

numro = 50
numco = 15

acceptable_formats = ['.dat', '.txt', '.exp', '.ws5', '.w98', '.moe', '.m1', '.mca', '.cmca', 'tango', '.mcs']

global fp
fp = os.getcwd()

# number of parameters for baseline
NBA = 8

# q = queue.Queue()
# def IMG_r(q, ):
#     global image
#     RA2.wait()
#     q.put(image.source)
#     # image.reload()
#     RA2.clear()

OSslesh = str('\\')*(platform.system() == 'Windows') + str('/')*(platform.system() != 'Windows')

def Bes0 (AA): # doi :10.1088/1742-6596/1043/1/012003
        return (np.exp(AA)+np.exp((-1)*AA))/2/(pow(1+(AA**2)/4, 1/4))*(1+0.24273*AA**2)/(1+0.43023*AA**2)


def mod_len_def (mod):
    mod_len = int(4 * (mod == 'Singlet') + 7 * (mod == 'Doublet') + 11 * (mod == 'Sextet') + 14 * (mod == 'Sextet(rough)') \
                + 11 * (mod == 'Relax_2S') + 11 * (mod == 'Average_H') + 9 * (mod == 'Relax_MS') + 12 * (mod == 'ASM')\
                + 11 * (mod == 'Hamilton_mc') + 9 * (mod == 'Hamilton_pc') + NBA * (mod == 'Nbaseline')\
                + 5 * (mod == 'Distr') + 2 * (mod == 'Corr') + 14 * (mod == 'MDGD')\
                + numco * (mod == 'Variables') + 1*(mod =='Expression'))
    return(mod_len)

def mod_len_def_M (mod):
    mod_len = int(4 * (mod == 'Singlet') + 7 * (mod == 'Doublet') + 11 * (mod == 'Sextet') + 14 * (mod == 'Sextet(rough)') \
                + 11 * (mod == 'Relax_2S') + 11 * (mod == 'Average_H') + 9 * (mod == 'Relax_MS') + 12 * (mod == 'ASM')\
                + 11 * (mod == 'Hamilton_mc') + 9 * (mod == 'Hamilton_pc') + numco * (mod == 'Variables')\
                + NBA * (mod == 'Nbaseline') + 14 * (mod == 'MDGD') )
    return(mod_len)

def ImgUpdate (self, image, _aa):

    # start_time = time.time()

    model_real_length = 0
    for i in range(2, len(self.realtableP)):
        if self.realtableP[-i][1].text != '':
            model_real_length = len(self.realtableP)-i+1
            break


    # time busy method
    global RA, RL, RR, tic, model, p, er, name, Rpa, fp, Distri, Expr, RM, CurMethod, confu, Rlog, RlogCol




    if self.Vel_start.active == True:
        self.cal_cho_title.text = "Velocity\ndown-up:"
    else:
        self.cal_cho_title.text = "Velocity\nup-down:"

    if CurMethod == 'SMS':
        self.SMS4 = str(self.realtableP[0][5].text)#
        self.SMS4lb = str(self.lboundstable[0][4].text)#
        self.SMS4rb = str(self.rboundstable[0][4].text)#
        self.SMS5 = str(self.realtableP[0][6].text)#
        self.SMS5lb = str(self.lboundstable[0][5].text)#
        self.SMS5rb = str(self.rboundstable[0][5].text)#
    if CurMethod == 'NFS':
        self.NFS4 = str(self.realtableP[0][5].text)#
        self.NFS4lb = str(self.lboundstable[0][4].text)#
        self.NFS4rb = str(self.rboundstable[0][4].text)#
        self.NFS5 = str(self.realtableP[0][6].text)#
        self.NFS5lb = str(self.lboundstable[0][5].text)#
        self.NFS5rb = str(self.rboundstable[0][5].text)#

    if self.switch.active == False and RM != 2:
        CurMethod = 'NFS'
        RM = 2
        try:
            self.realtableP[0][5].text = self.NFS4#
            self.lboundstable[0][4].text = self.NFS4lb#
            self.rboundstable[0][4].text = self.NFS4rb#
            self.realtableP[0][6].text = self.NFS5#
            self.lboundstable[0][5].text = self.NFS5lb#
            self.rboundstable[0][5].text = self.NFS5rb#
            # print(self.NFS4)
        except:
            pass
    if self.switch.active == True and RM != 1:
        CurMethod = 'SMS'
        RM = 1
        try:
            self.realtableP[0][5].text = self.SMS4#
            self.lboundstable[0][4].text = self.SMS4lb#
            self.rboundstable[0][4].text = self.SMS4rb#
            self.realtableP[0][6].text = self.SMS5#
            self.lboundstable[0][5].text = self.SMS5lb#
            self.rboundstable[0][5].text = self.SMS5rb#
        except:
            pass




    if self.switch.active == True:
        self.realtableP[0][2].background_color = [1, 1, 1, 1]
        self.realtableP[0][3].background_color = [1, 1, 1, 1]
        self.realtableP[0][4].background_color = [1, 1, 1, 1]###
        self.realtableP[0][7].background_color = [1, 1, 1, 1]#
        self.realtableP[0][8].background_color = [1, 1, 1, 1]###

    if Window.size[0] < 1450 and self.table.row_default_height != '80dp':
        self.table.row_default_height = '80dp'
        self.table.col_default_width = '55dp'
        self.tableR.row_default_height = '27dp'
        self.tableR.col_default_width = '65dp'
        self.table.height = '%.i dp' % int(numro * 80)
        self.table.width = '%.i dp' % int((numco + 1) * 55)
        self.tableR.height = '%.i dp' % int(numro * 2 * 27)
        self.tableR.width = '%.i dp' % int((numco + 1) * 65)
        self.INS_number.font_size = '11sp'
        self.JN0.font_size = '11sp'
        self.L0.font_size = '11sp'
        self.INS_number.size_hint = (1, 1.5)
        self.JN0.size_hint = (1, 1.2)
        self.L0.size_hint = (1, 1.2)
        for i in range(0, len(self.realtable)-1):
            for j in range(0, len(self.realtable[0])):
                self.realtable[i][j].font_size = '13sp'
        for i in range(0, len(self.realtableP)-1):
            for j in range(0, len(self.realtableP[0])-1):
                self.realtableP[i][j].font_size = '11sp'
                self.lboundstable[i][j].font_size = '10sp'
                self.rboundstable[i][j].font_size = '10sp'
                self.nametable[i][j].font_size = '9sp'
                self.fixtable[i][j].size_hint = (0.5, 0.25)
            self.realtableP[i][-1].font_size = '11sp'
            self.realtableP[i][0].height = '%.i dp' % int(80/3)
            self.realtableP[i][0].width = '%.i dp' % int(55)
            self.startlabel[i].font_size = '9sp'
            self.startlabel2[i].font_size = '9sp'
    elif Window.size[0] > 1450 and self.table.row_default_height != '100dp':
        self.table.row_default_height = '100dp'
        self.table.col_default_width = '95dp'
        self.tableR.row_default_height = '30dp'
        self.tableR.col_default_width = '105dp'
        self.table.height = '%.i dp' % int(numro * 100)
        self.table.width = '%.i dp' % int((numco + 1) * 95)
        self.tableR.height = '%.i dp' % int(numro * 2 * 30)
        self.tableR.width = '%.i dp' % int((numco + 1) * 105)
        self.INS_number.font_size = '16sp'
        self.JN0.font_size = '16sp'
        self.L0.font_size = '16sp'
        self.INS_number.size_hint = (1, 1)
        self.JN0.size_hint = (1, 1)
        self.L0.size_hint = (1, 1)
        for i in range(0, len(self.realtable)-1):
            for j in range(0, len(self.realtable[1])):
                self.realtable[i][j].font_size = '18sp'
        for i in range(0, len(self.realtableP) - 1):
            for j in range(0, len(self.realtableP[1])-1):
                self.realtableP[i][j].font_size = '18sp'
                self.lboundstable[i][j].font_size = '16sp'
                self.rboundstable[i][j].font_size = '16sp'
                self.nametable[i][j].font_size = '16sp'
                self.fixtable[i][j].size_hint = (1, 0.5)
            self.realtableP[i][-1].font_size = '18sp'
            self.realtableP[i][0].height = '%.i dp' % int(100/3)
            self.realtableP[i][0].width = '%.i dp' % int(95)
            self.startlabel[i].font_size = '18sp'
            self.startlabel2[i].font_size = '18sp'

    for i in range (0, model_real_length): #len(self.realtableP)
        for j in range(1, len(self.realtableP[i])):
            if self.realtableP[i][j].text != '' and (str(self.realtableP[i][0].text)[:5] != 'Distr' or j != 5):
                try:
                    float(self.realtableP[i][j].text)
                    self.realtableP[i][j].background_color = [1, 1, 1, 1]
                except ValueError:
                    # self.realtableP[i][j].background_color = [255, 0, 0, 1]
                    if self.realtableP[i][j].focus == False:
                        self.realtableP[i][j].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][j].background_color = [1, 0.8, 0.8, 1]
                try:
                    if self.realtableP[i][j].text[0:2] == '=[' and self.realtableP[i][j].text[-1] == ']':
                        b = np.array((self.realtableP[i][j].text[2:-1]).split(','))
                        if len(b) != 2:
                            # self.realtableP[i][j].background_color = [255, 0, 0, 1]
                            if self.realtableP[i][j].focus == False:
                                self.realtableP[i][j].background_color = [255, 0, 0, 1]
                            else:
                                self.realtableP[i][j].background_color = [1, 0.8, 0.8, 1]
                        else:
                            try:
                                float(b[0])
                                float(b[1])
                                self.realtableP[i][j].background_color = [0.5, 0.5, 1, 1]
                            except:
                                # self.realtableP[i][j].background_color = [255, 0, 0, 1]
                                if self.realtableP[i][j].focus == False:
                                    self.realtableP[i][j].background_color = [255, 0, 0, 1]
                                else:
                                    self.realtableP[i][j].background_color = [1, 0.8, 0.8, 1]
                except:
                    pass


            try:
                if self.nametable[i][j-1].text != '' and self.realtableP[i][j].text == '' and (str(self.realtableP[i][0].text)[:5] != 'Distr' or j != 4):
                    # self.realtableP[i][j+1].background_color = [255, 0, 0, 1]
                    if self.realtableP[i][j].focus == False:
                        self.realtableP[i][j].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][j].background_color = [1, 0.8, 0.8, 1]
                if self.nametable[i][j-1].text == '':
                    self.realtableP[i][j].background_color = [1, 1, 1, 1]
            except:
                pass

            if (self.realtableP[i][0].text in ['Distr', 'Corr', 'baseline', 'Nbaseline']) == False:
                self.realtableP[i][0].background_color = [1, 1, 1, 1]

            if self.realtableP[i][0].text == 'Distr':
                self.startlabel[i].color = self.startlabel[i-1].color
                M = 1
                for k in range(1, i):
                    M = i - k
                    if self.realtableP[i - k][0].text != 'Distr' and self.realtableP[i - k][0].text != 'Corr':
                        M += 1
                        break
                k = 1
                for k in 1, 4:
                    try:
                        if str(self.realtableP[i][k].text) != str(int(self.realtableP[i][k].text)):
                            self.realtableP[i][k].background_color = [255, 0, 0, 1]
                        elif k==1:
                            # if self.realtableP[i-1][int(self.realtableP[i][k].text)+1].background_color == [1, 1, 1, 1]:
                            #     self.realtableP[i-1][int(self.realtableP[i][k].text)+1].background_color = [1, 1, 1, 0.5]
                            if self.realtableP[M-1][int(self.realtableP[i][k].text)+1].background_color == [1, 1, 1, 1]:
                                self.realtableP[M-1][int(self.realtableP[i][k].text)+1].background_color = [1, 1, 1, 0.5]
                    except:
                        # self.realtableP[i][k].background_color = [255, 0, 0, 1]
                        if self.realtableP[i][k].focus == False:
                            self.realtableP[i][k].background_color = [255, 0, 0, 1]
                        else:
                            self.realtableP[i][k].background_color = [1, 0.8, 0.8, 1]
                # C = str(self.realtableP[M-1][0].text)
                self.rboundstable[i][0].text = str(mod_len_def_M(str(self.realtableP[M-1][0].text)) - 1)

                STR = str(self.realtableP[i][5].text) + str(' ')
                st_ = []
                en_ = []
                for k in range(0, len(STR) - 2):
                    if STR[k] == 'p' and STR[k + 1] == '[':
                        st_.append(k)
                        en_.append(k)
                        for kk in range(k, len(STR)):
                            if STR[kk] == ']':
                                en_[-1] = kk
                                break
                st_ = st_[::-1]
                en_ = en_[::-1]
                for k in range(0, len(st_)):
                    STR = str(STR[:st_[k]]) + str('sqrt(') + str(STR[(st_[k]+2):en_[k]]) + str(')') + str(STR[(en_[k] + 1):])
                STR = STR.replace('X', 'sqrt(1)')

                try:
                    C = eval(STR)
                    self.realtableP[i][5].background_color = [1, 1, 1, 1]
                except:
                    self.realtableP[i][10].background_color = [255, 0, 0, 1]
                    if self.realtableP[i][5].focus == False:
                        self.realtableP[i][5].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][5].background_color = [1, 0.8, 0.8, 1]

                # if self.realtableP[i-1][0].text in ['baseline', 'Corr', 'Distr', 'None']:
                if self.realtableP[i - 1][0].text in ['baseline', 'None', 'Nbaseline']:
                    self.realtableP[i][0].background_color = [255, 0, 0, 1]
                else:
                    self.realtableP[i][0].background_color = [1, 1, 1, 1]

                for k in range(M, i):
                    if self.realtableP[i][1].text == self.realtableP[k][1].text:
                        self.realtableP[i][1].background_color = [255, 0, 0, 1]

            if self.realtableP[i][0].text == 'Corr':
                self.startlabel[i].color = self.startlabel[i-1].color
                M = 1
                for k in range(1, i):
                    M = i - k
                    if self.realtableP[i-k][0].text != 'Distr' and self.realtableP[i-k][0].text != 'Corr':
                        M += 1
                        break
                k = 1
                try:
                    if str(self.realtableP[i][k].text) != str(int(self.realtableP[i][k].text)):
                        self.realtableP[i][k].background_color = [255, 0, 0, 1]
                    if self.realtableP[M-1][int(self.realtableP[i][k].text)+1].background_color == [1, 1, 1, 1]:
                        self.realtableP[M-1][int(self.realtableP[i][k].text)+1].background_color = [1, 1, 1, 0.5]
                except:
                    if self.realtableP[i][k].focus == False:
                        self.realtableP[i][k].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][k].background_color = [1, 0.8, 0.8, 1]
                # C = str(self.realtableP[M-1][0].text)
                self.rboundstable[i][0].text = str(mod_len_def_M(str(self.realtableP[M-1][0].text)) - 1)

                STR = str(self.realtableP[i][2].text) + str(' ')
                st_ = []
                en_ = []
                for k in range(0, len(STR) - 2):
                    if STR[k] == 'p' and STR[k + 1] == '[':
                        st_.append(k)
                        en_.append(k)
                        for kk in range(k, len(STR)):
                            if STR[kk] == ']':
                                en_[-1] = kk
                                break
                st_ = st_[::-1]
                en_ = en_[::-1]
                for k in range(0, len(st_)):
                    STR = str(STR[:st_[k]]) + str('sqrt(') + str(STR[(st_[k]+2):en_[k]]) + str(')') + str(STR[(en_[k] + 1):])
                STR = STR.replace('X', 'sqrt(1)')

                try:
                    C = eval(STR)
                    self.realtableP[i][2].background_color = [1, 1, 1, 1]
                except:
                    self.realtableP[i][10].background_color = [255, 0, 0, 1]
                    if self.realtableP[i][2].focus == False:
                        self.realtableP[i][2].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][2].background_color = [1, 0.8, 0.8, 1]
                if (self.realtableP[i-1][0].text in ['Distr', 'Corr'])==False:
                    self.realtableP[i][0].background_color = [255, 0, 0, 1]
                else:
                    self.realtableP[i][0].background_color = [1, 1, 1, 1]

                for k in range(M, i):
                    if self.realtableP[i][1].text == self.realtableP[k][1].text:
                        self.realtableP[i][1].background_color = [255, 0, 0, 1]

            if self.realtableP[i][0].text == 'Expression':
                STR = str(self.realtableP[i][1].text) + str(' ')
                st_ = []
                en_ = []
                for k in range(0, len(STR) - 2):
                    if STR[k] == 'p' and STR[k + 1] == '[':
                        st_.append(k)
                        en_.append(k)
                        for kk in range(k, len(STR)):
                            if STR[kk] == ']':
                                en_[-1] = kk
                                break
                st_ = st_[::-1]
                en_ = en_[::-1]
                for k in range(0, len(st_)):
                    STR = str(STR[:st_[k]]) + str('sqrt(') + str(STR[(st_[k]+2):en_[k]]) + str(')') + str(STR[(en_[k] + 1):])
                try:
                    C = eval(STR)
                    self.realtableP[i][1].background_color = [1, 1, 1, 1]
                except:
                    # self.realtableP[i][1].background_color = [255, 0, 0, 1]
                    if self.realtableP[i][1].focus == False:
                        self.realtableP[i][1].background_color = [255, 0, 0, 1]
                    else:
                        self.realtableP[i][1].background_color = [1, 0.8, 0.8, 1]

            if self.realtableP[i][0].text == 'Insert':
                self.realtableP[i][0].background_color = [255, 0, 0, 1]


            if self.realtableP[i][0].text == 'Nbaseline':
                self.realtableP[i][0].background_color = [0, 2, 5, 1]
                # if self.realtableP[i+1][0].text == 'Nbaseline' or self.realtableP[i+1][0].text == 'None':
                #     self.realtableP[i][0].background_color = [255, 0, 0, 1]


    for i in range (0, model_real_length): #len(self.lboundstable)
        for j in range(0, len(self.lboundstable[i])):

            if self.fixtable[i][j].disabled == False and\
                    ((self.realtableP[i][j + 1].text[0:2] == '=[' and self.realtableP[i][j + 1].text[-1] == ']')\
                     or (self.realtableP[i][j + 1].background_color == [1, 1, 1, 0.5]) or (self.realtableP[i][0].text == 'Expression')):
                self.fixtable[i][j].active = True

            if self.realtableP[i][j + 1].text[0:2] == '=[' and self.realtableP[i][j + 1].text[-1] == ']':
                self.lboundstable[i][j].background_color = [1, 1, 1, 0.5]
                self.rboundstable[i][j].background_color = [1, 1, 1, 0.5]
            elif (self.realtableP[i][0].text != 'Distr' and self.realtableP[i][0].text != 'Corr' and self.realtableP[i][0].text != 'Expression')\
                  or (self.realtableP[i][0].text == 'Distr' and j < 4)\
                  or (self.realtableP[i][0].text == 'Corr' and j < 1):
                self.lboundstable[i][j].background_color = [1, 1, 1, 1]
                self.rboundstable[i][j].background_color = [1, 1, 1, 1]


            if self.lboundstable[i][j].text != '' and (i!=0 or j not in [NBA+1,NBA+2,NBA+3]):
                # self.lboundstable[i][j].background_color = [255, 1, 255, 0.9]

                try:
                    float(self.lboundstable[i][j].text)
                    if self.realtableP[i][j + 1].text[0:2] != '=[':
                        self.lboundstable[i][j].background_color = [0.8, 1.1, 3, 0.9]
                except ValueError:
                    # self.lboundstable[i][j].background_color = [255, 0, 0, 1]
                    if self.lboundstable[i][j].focus == False:
                        self.lboundstable[i][j].background_color = [255, 0, 0, 1]
                    else:
                        self.lboundstable[i][j].background_color = [1, 0.8, 0.8, 1]
                try:
                    if float(self.realtableP[i][j + 1].text) == float(self.lboundstable[i][j].text) and self.realtableP[i][j + 1].background_color == [1, 1, 1, 1]:
                        self.realtableP[i][j + 1].background_color = [0.9, 0.9, 0.1, 1]
                        self.lboundstable[i][j].background_color = [0.9, 0.9, 0.1, 1]
                    if float(self.realtableP[i][j+1].text) < float(self.lboundstable[i][j].text):
                        # self.realtableP[i][j+1].background_color = [255, 0, 0, 1]
                        if self.realtableP[i][j+1].focus == False:
                            self.realtableP[i][j+1].background_color = [255, 0, 0, 1]
                        else:
                            self.realtableP[i][j+1].background_color = [1, 0.8, 0.8, 1]
                except:
                    pass
            # else:
            #     self.lboundstable[i][j].background_color = [1, 1, 1, 1]

            if self.rboundstable[i][j].text != '' and (i!=0 or j not in [NBA+1,NBA+2,NBA+3]):
                # self.rboundstable[i][j].background_color = [255, 1, 255, 0.9]


                try:
                    float(self.rboundstable[i][j].text)
                    if self.realtableP[i][j + 1].text[0:2] != '=[':
                        self.rboundstable[i][j].background_color = [0.8, 1.1, 3, 0.9]
                except ValueError:
                    # self.rboundstable[i][j].background_color = [255, 0, 0, 1]
                    if self.rboundstable[i][j].focus == False:
                        self.rboundstable[i][j].background_color = [255, 0, 0, 1]
                    else:
                        self.rboundstable[i][j].background_color = [1, 0.8, 0.8, 1]
                try:
                    if float(self.realtableP[i][j + 1].text) == float(self.rboundstable[i][j].text) and self.realtableP[i][j + 1].background_color == [1, 1, 1, 1]:
                        self.realtableP[i][j + 1].background_color = [0.9, 0.9, 0.1, 1]
                        self.rboundstable[i][j].background_color = [0.9, 0.9, 0.1, 1]
                    if float(self.realtableP[i][j+1].text) > float(self.rboundstable[i][j].text):
                        # self.realtableP[i][j+1].background_color = [255, 0, 0, 1]
                        if self.realtableP[i][j+1].focus == False:
                            self.realtableP[i][j+1].background_color = [255, 0, 0, 1]
                        else:
                            self.realtableP[i][j+1].background_color = [1, 0.8, 0.8, 1]
                except:
                    pass

                try:
                    if float(self.rboundstable[i][j].text) < float(self.lboundstable[i][j].text):
                        # self.rboundstable[i][j].background_color = [255, 0, 0, 1]
                        if self.rboundstable[i][j].focus == False:
                            self.rboundstable[i][j].background_color = [255, 0, 0, 1]
                        else:
                            self.rboundstable[i][j].background_color = [1, 0.8, 0.8, 1]
                        # self.lboundstable[i][j].background_color = [255, 0, 0, 1]
                        if self.lboundstable[i][j].focus == False:
                            self.lboundstable[i][j].background_color = [255, 0, 0, 1]
                        else:
                            self.lboundstable[i][j].background_color = [1, 0.8, 0.8, 1]
                except ValueError:
                    pass

    for i in (self.L0, self.JN0):
        try:
            if (float(i.text) == abs(float(i.text)) or int(i.text) == abs(int(i.text))) and float(str(i.text)) != 0:
                i.background_color = [1, 1, 1, 1]
            else:
                if i.focus == False:
                    i.background_color = [255, 0, 0, 1]
                else:
                    i.background_color = [1, 0.8, 0.8, 1]
        except:
            if i.focus == False:
                i.background_color = [255, 0, 0, 1]
            else:
                i.background_color = [1, 0.8, 0.8, 1]
    if self.fitway[1].active == True or self.fitway[2].active == True:
        self.L0.background_color = [1, 1, 1, 0.5]


    for i in range (0, len(self.lboundstable)): #len(self.lboundstable)
        for j in range(0, len(self.lboundstable[i])):
            # if self.startlabel2[i].text == 'unfix model':
            #     self.fixtable[i][j].active = True
            if self.fixtable[i][j].active == False and self.startlabel2[i].text == 'unfix model':
                self.startlabel2[i].color = 'cyan'
                self.startlabel2[i].text = 'fix model'
                self.fix_table_ch[i] = 0
            if self.realtableP[i][0].text == 'None' or self.realtableP[i][0].text == 'Insert':
                self.fixtable[i][j].active = False
                self.fixtable[i][j].disabled = True
                self.lboundstable[i][j].background_color = [0, 0, 0, 0]
                self.rboundstable[i][j].background_color = [0, 0, 0, 0]
                self.realtableP[i][j+1].background_color = [0, 0, 0, 0]
                self.lboundstable[i][j].disabled = True
                self.rboundstable[i][j].disabled = True
                self.realtableP[i][j+1].disabled = True
                if self.startlabel2[i].text == 'unfix model':
                    self.fix_all(i)
            if (self.realtableP[i][0].text == 'Distr' and j > 3)\
                  or (self.realtableP[i][0].text == 'Corr' and j > 0)\
                    or self.realtableP[i][0].text == 'Expression':
                self.lboundstable[i][j].disabled = True
                self.rboundstable[i][j].disabled = True
                self.lboundstable[i][j].background_color = [0, 0, 0, 0]
                self.rboundstable[i][j].background_color = [0, 0, 0, 0]


    if self.switch.active == False:
        for i in range(1, len(self.lboundstable)-1):
            if self.realtableP[i][0].text not in ["Sextet", "Doublet", "Singlet", "None", "Expression"]:
                self.realtableP[i][0].background_color = [255, 0, 0, 1]

    V = 0
    for i in range (0, len(self.lboundstable)):
        for j in range(0, len(self.lboundstable[i])):
            if self.realtableP[i][j+1].background_color == [255, 0, 0, 1]\
                or self.realtableP[i][j].background_color == [255, 0, 0, 1]\
                or self.lboundstable[i][j].background_color == [255, 0, 0, 1]\
                or self.rboundstable[i][j].background_color == [255, 0, 0, 1]\
                or self.realtableP[i][j+1].background_color == [1, 0.8, 0.8, 1]:
                V = 4
    if self.L0.background_color == [255, 0, 0, 1] or self.JN0.background_color == [255, 0, 0, 1]\
            or self.L0.background_color == [1, 0.8, 0.8, 1] or self.JN0.background_color == [1, 0.8, 0.8, 1]:
        V = 4

    if self.switch.active == False:
        self.check_points_match = True
    #if switch from NFS to SMS need to check if spectrum is compatible with calibration
    elif self.check_points_match == True and self.switch.active == True and self.points_match == True:
        for i in range(0, len(self.path_list)):
            file = os.path.abspath(self.path_list[i])
            A, B = read_spectrum(self, file)
            if len(A) != len(B):
                self.points_match = False
        self.check_points_match = False

    # check matching number of points in spc and calibration
    if self.points_match == False:
        self.vel_btn.disabled = True * (self.switch.active == True)
        self.desc.disabled = True
        V += 1
    elif self.points_match == True:
        self.vel_btn.disabled = False
        self.desc.disabled = False

    Tsum = 0
    for i in range(1, len(self.lboundstable)):
        try:
            Tsum += float(self.realtableP[i][1].text)
        except:
            pass

    # forbidden to plot just baseline in NFS mode
    if self.switch.active == False and Tsum == 0:
        V += 2


    if V > 0 or RL==1:
    # if RL == 1:
        self.play_btn.background_color = [0, 0.5, 0, 1]
        self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
        self.show_btn.background_color = np.array([0.5, 0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5, 0.5]) * (self.switch.active == False)
        self.cal_btn.background_color = np.array([0.5, 0.5, 0.5, 0.5]) + np.array([-0.1, 254.5, 254.5, 254.5]) * (V==1) * (RL!=1)
        self.play_btn.disabled = True
        self.showM_btn.disabled = True
        self.show_btn.disabled = True * (self.switch.active == True) #* ((V%2) == 1) # *(V==1 or V==3 or V==5 or V==7)
        # self.savemod_btn.disabled = True
        # self.savemodas_btn.disabled = True
        # self.saveas_btn.disabled = True
        self.cal_btn.disabled = False
        if V != 1 or RL == 1:
            self.cal_btn.disabled = True
        if self.points_match == False or RL == 1:
            self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
            self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
            self.INS_btn.disabled = True
            self.INS_btn2.disabled = True


    else:
        self.play_btn.disabled = False
        self.showM_btn.disabled = False
        self.show_btn.disabled = False
        # self.savemod_btn.disabled = False
        # self.savemodas_btn.disabled = False
        # self.saveas_btn.disabled = False
        self.cal_btn.disabled = False
        self.INS_btn.disabled = False
        self.INS_btn2.disabled = False
        self.play_btn.background_color = [0, 3, 0, 1]
        self.cal_btn.background_color = [0.4, 255, 255, 1]
        self.INS_btn.background_color = [1, 1, 1, 1]
        self.INS_btn2.background_color = [1, 1, 1, 1]
        self.showM_btn.background_color = [1, 1, 1, 1]
        self.show_btn.background_color = [1, 1, 1, 1]



    if self.switch.active == False:
        self.INS_btn.disabled = True
        self.INS_btn2.disabled = True
        self.cal_btn.disabled = True
        self.fitmodel1.text = 'Linear'
        self.fitmodel3.text = 'Log'
        self.fitmodel4.text = ''
        if self.fitway[2].active == True:
            self.fitway[2].active = False
            self.fitway[0].active = True
        self.chB4.disabled = True
        self.nametable[0][4].text = u'BackGr'  # \u2092'
        self.nametable[0][5].text = u'T shift'  # \u2092'

        try:
            file = os.path.abspath(self.path_list[0])
            A = read_spectrum(self, file)[0]
            # A_list = []
            # with open(file, 'r') as catalog:
            #     lines = (line.rstrip() for line in catalog)
            #     lines = (line for line in lines if line)  # skipping white lines
            #     for line in lines:
            #         column = line.split()
            #         if not line.startswith('#'):  # skipping column labels
            #             x = float(column[0])
            #             A_list.append(x)
            # A = np.array(A_list)
            if self.lboundstable[0][5].text == '' or float(self.lboundstable[0][5].text) < float(str("%.1f" % -min(A))):
                self.lboundstable[0][5].text = str("%.1f" % -min(A))
            # if self.fixtable[0][5].active == False and self.lboundstable[0][5].text != '':
            # if float(self.lboundstable[0][5].text) < float(str("%.1f" % -min(A))):
            #     self.lboundstable[0][5].background_color = [255, 0, 0, 1]
            # if self.fixtable[0][5].active == True:
            # if float(self.realtableP[0][6].text) < float(str("%.1f" % -min(A))):#
            #     self.realtableP[0][6].background_color = [255, 0, 0, 1]#

        except:
            self.lboundstable[0][5].background_color = [255, 0, 0, 1]
            self.rboundstable[0][5].background_color = [255, 0, 0, 1]
        # self.lboundstable[0][5].readonly = True

        self.fixtable[0][1].active = True
        self.fixtable[0][2].active = True
        self.fixtable[0][6].active = True
        self.fixtable[0][1].disabled = True
        self.fixtable[0][2].disabled = True
        self.fixtable[0][6].disabled = True
        self.fixtable[0][7].disabled = True
        self.realtableP[0][2].background_color = [1, 1, 1, 0.5]
        self.realtableP[0][3].background_color = [1, 1, 1, 0.5]
        self.realtableP[0][4].background_color = [1, 1, 1, 0.5]###
        self.realtableP[0][7].background_color = [1, 1, 1, 0.5]#
        self.realtableP[0][8].background_color = [1, 1, 1, 0.5]###



    else:
        self.fitmodel1.text = 'MS'
        self.fitmodel3.text = 'SMS'
        self.fitmodel4.text = 'APS'
        self.chB4.disabled = False
        self.nametable[0][4].text = u'Nnr'  # \u2092'
        self.nametable[0][5].text = u'Onr'  # \u2092'
        self.lboundstable[0][5].readonly = False
        self.fixtable[0][1].disabled = False
        self.fixtable[0][2].disabled = False
        self.fixtable[0][6].disabled = False
        self.fixtable[0][7].disabled = False
        if self.fitway[0].active == True:
            self.INS_btn.disabled = True
            # self.INS_btn2.disabled = True
            # self.cal_btn.disabled = True
        # else:
        #     self.INS_btn.disabled = False
        #     self.INS_btn2.disabled = False
    T = self.log.text
    if self.fitway[0].active == False and self.fitway[1].active == False and self.fitway[2].active == False:
        self.fitway[0].active = True

    Nb_counter = 0
    Nb_num = []
    for i in range(0, len(self.realtableP)-1):
        if self.realtableP[i][0].text == 'Nbaseline':
            Nb_counter += 1
            Nb_num.append(i)


    if Nb_counter != 0 and len(self.path_list) != Nb_counter + 1:
        self.play_btn.disabled = True
        self.showM_btn.disabled = True
        self.log.text = 'Number of files should coincide with number of baselines'
        self.log.background_color = [255, 0, 0, 1]
        for i in range(0, len(Nb_num)):
            self.realtableP[Nb_num[i]][0].background_color = [255, 0, 0, 1]
    else:
        if self.log.text == 'Number of files should coincide with number of baselines':
            self.log.text = 'Now it is ready'
            self.log.background_color = [0, 255, 0, 1]

    if RA == True:
        if self.SP_DI.text != 'Distribution':
            if os.path.exists(str(self.dir_path) + str('\\\\resultD.png')) == True\
                    or os.path.exists(str(self.dir_path) + str('/resultD.png')) == True:
                realpath = str(self.dir_path) + str('\\\\resultD.png')*(platform.system() == 'Windows')\
                                              + str('/resultD.png')   *(platform.system() != 'Windows')
                image.source = realpath
                image.reload()
                self.SP_DI.text = 'Spectrum'
                self.log.text = 'Distributions and correlations'
                self.log.background_color = [0, 255, 0, 1]
            else:
                self.SP_DI.text = 'Distribution'
                self.log.text = 'There is no distribution'
                self.log.background_color = [255, 0, 0, 1]
        else:
            realpath = str(self.dir_path) + str('\\\\result.png') * (platform.system() == 'Windows') \
                                          + str('/result.png')    * (platform.system() != 'Windows')
            # self.log.text = 'Result'
            image.source = realpath
            image.reload()
        RA = False



    # if Rpa == 1:
    #     raw_path = fp
    #     self.process_path.text = str(raw_path)
    #     self.save_path.text = str(raw_path)
    #     fp = os.getcwd()
    #     Rpa = 0


    if RR == True:
        if len(er) > 1:
            for i in range (0,len(self.realtable)-1):
                for j in range(0, len(self.realtable[0])):
                    self.realtable[i][j].text = ''
            self.realtable[0][0].text = 'baseline'
            self.realtable[0][1].text = 'Ns = %.0f' % p[0]
            self.realtable[1][1].text = "± %.0f" % er[0]
            self.realtable[0][2].text = 'Os = %.3f' % p[1]
            self.realtable[1][2].text = "± %.3f" % er[1]
            self.realtable[0][3].text = "c²s = %.3f" % p[2]
            self.realtable[1][3].text = "± %.3f" % er[2]
            self.realtable[0][4].text = 'lins = %.3f' % p[3]
            self.realtable[1][4].text = "± %.3f" % er[3]
            self.realtable[0][5].text = 'Nnr = %.0f' % p[4]
            self.realtable[1][5].text = "± %.0f" % er[4]
            self.realtable[0][6].text = 'Onr = %.3f' % p[5]
            self.realtable[1][6].text = "± %.3f" % er[5]
            self.realtable[0][7].text = "c² = %.3f" % p[6]
            self.realtable[1][7].text = "± %.3f" % er[6]
            self.realtable[0][8].text = 'linnr = %.3f' % p[7]
            self.realtable[1][8].text = "± %.3f" % er[7]

            # print('zero error is', er[0])
            # for i in range (3, len(p)): # now it is below - need check
            #     if abs(p[i]) > 9999:
            #         p[i] = np.inf
            #     if abs(er[i]) > 9999:
            #         er[i] = np.inf
        else:
            self.log.text = "there was no move at all"
            self.log.background_color = [255, 0, 0, 0.9]
            RL = 3

        Expression_table = []
        erExpression_table = []
        dirExpression_table = []


        ExprNum = 0
        if len(model) != 0:
            V = NBA
            # EV = 0
            for i in range(0, len(model)):
                C = model[i]
                if model[i] == 'Expression':
                    erExpression = 0
                    Expr_tmp = Expr[ExprNum]#str(self.realtableP[i+1][1].text)
                    ExprNum += 1
                    Expression_table.append(eval(Expr_tmp))
                    st = [pos for pos, char in enumerate(Expr_tmp) if char == '[']
                    en = [pos for pos, char in enumerate(Expr_tmp) if char == ']']
                    for j in range(0, len(st)):
                        Expr_tmp_dir = eval(Expr_tmp[:(st[j]-1)] + str('(p[') + Expr_tmp[(st[j]+1):en[j]] + str(']+10**(-8))') + Expr_tmp[(en[j]+1):])
                        # print(Expr_tmp_dir)
                        Expr_tmp_dir = (Expr_tmp_dir - Expression_table[-1])*10**8
                        # print(Expr_tmp_dir)
                        if str(er[int(Expr_tmp[(st[j]+1):en[j]])]) != 'nan':
                            erExpression += (Expr_tmp_dir*er[int(Expr_tmp[(st[j]+1):en[j]])])**2
                        # print('error tmp ', erExpression)
                    erExpression = np.sqrt(erExpression)
                    # print('error ', erExpression)
                    erExpression_table.append(erExpression)

                    er[V] = float(erExpression_table[-1])
                    # EV += 1
                V += mod_len_def(C)

        for i in range(0, len(er)):
            if any(i == confu[0]):
                n = int(np.where(confu[0] == i)[0][0])
                er[i] = er[int(confu[1][n])] * confu[2][n]
        # for kk in range(0, len(Expr)):  # expressions
        #     p[NExpr[kk]] = eval(str(Expr[kk]))
        #     if any(confu[1] == NExpr[kk]) == True:
        #         n = np.where(confu[1] == NExpr[kk])[0]
        #         for kkk in range(0, len(n)):
        #             p[int(confu[0][n[kkk]])] = p[int(confu[1][n[kkk]])] * confu[2][n[kkk]]

        #printing result parameters
        T_full = [0]
        T_table = [[]]
        erT_table = [[]]
        spc_table = []
        V = NBA
        Vn = 0
        Di = 0
        Co = 0
        Exr = 0
        try:
            Be_param = np.genfromtxt(str(self.dir_path) + str('\\\\Be.txt')*(platform.system() == 'Windows') + str('/Be.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
        except:
            Be_param = np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1.0])
        try:
            KB_param = np.genfromtxt(str(self.dir_path) + str('\\\\KB.txt')*(platform.system() == 'Windows') + str('/KB.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
        except:
            KB_param = np.array([0.065, 0.234, 0.37, 0.098, 0.373, 0.5, 1.0])
        if len(model) != 0:
            for i in range(1, len(model)+1):
                self.realtable[i*2][0].text = model[i-1]
                LenM = mod_len_def_M(model[i-1]) + 1
                if model[i-1] == 'Nbaseline':
                    T_full.append(0)
                    T_table.append([])
                    erT_table.append([])
                    # spc_table.append([])
                Be_check = (model[i-1] != 'Distr') * (model[i-1] != 'Corr') * (model[i-1] != 'Expression')
                for j in range(0, LenM-1):
                    if p[V+j] != Be_param[j] or model[i-1] != 'Doublet':
                        Be_check = 0
                        break
                KB_check = (model[i - 1] != 'Distr') * (model[i - 1] != 'Corr') * (model[i - 1] != 'Expression')
                for j in range(0, LenM - 1):
                    if p[V + j] != KB_param[j] or model[i - 1] != 'Doublet':
                        KB_check = 0
                        break

                for j in range(1, LenM):
                    self.realtable[i * 2][j].color = [1, 1, 1, 1]
                    self.realtable[i * 2 + 1][j].color = [1, 1, 1, 1]
                    # for k in range(i+Vn,  99):
                    #     try:
                    #         l = str(name[k][j-1]).split(',')[0]
                    #         break
                    #     except IndexError:
                    #         Vn += 1
                    l = str(name[i][j - 1]).split(',')[0]
                    if l=='Ns' or l=='Nnr':
                        try:
                            self.realtable[i*2][j].text = l + ' = ' + str(int(p[V]))
                        except:
                            self.realtable[i * 2][j].text = l + ' = error'
                            self.realtable[i * 2][j].color = [255, 0, 0, 1]
                    else:
                        self.realtable[i*2][j].text = l + ' = ' + str("%.3f" % p[V])
                    # self.realtable[i*2][j].text = l + ' = ' + str("%.3f" % p[V])*(l!='Ns') + str(int(p[V]))*(l=='Ns')
                    if p[V] > 9999 and l != 'Ns' and l != 'Nnr':
                        p[V] = np.inf
                        self.realtable[i * 2][j].color = [1, 255, 255, 1]


                    # if str(er[V]) == 'nan' and self.realtableP[i][j].text[0] == '=':
                    #     p_link_number, link_multiplier = str(self.realtableP[i][j].text[2:-1]).split(',')
                    #     # print(p_link_number, link_multiplier)
                    #     try:
                    #         er[V] = float(er[int(p_link_number)]) * float(link_multiplier)
                    #         # print(er[V])
                    #     except:
                    #         pass

                    if er[V] > 9999:
                        er[V] = np.inf
                        self.realtable[i*2+1][j].color = [1, 255, 255, 1]
                    if er[V] != np.nan and er[V] != np.inf and l=='Ns':
                        self.realtable[i * 2 + 1][j].text = "± " + str(int(er[V]))
                    else:
                        self.realtable[i * 2 + 1][j].text = "± %.3f" % er[V]
                                                        #"± %.3f" % er[V]

                    if l == 'T' and Be_check == 0 and KB_check == 0:
                        T_full[-1] += p[V]
                        T_table[-1].append(p[V])
                        if str(er[V]) != 'nan':
                            erT_table[-1].append(er[V])
                        else:
                            erT_table[-1].append(int(0))
                        spc_table.append(i)

                    V+=1

                if Be_check == 1:
                    self.realtable[i * 2 + 1][0].color = self.realtable[i * 2][0].color
                    self.realtable[i * 2 + 1][0].text = 'Be lenses'
                if KB_check == 1:
                    self.realtable[i * 2 + 1][0].color = self.realtable[i * 2][0].color
                    self.realtable[i * 2 + 1][0].text = 'KB mirror'

                if model[i - 1] == "Nbaseline":
                    self.realtable[i * 2][0].color = [1,1,1,1]

                if model[i - 1] == 'Expression':
                    self.realtable[i * 2][1].text = Expr[Exr] + str(" = %.3f" % Expression_table[Exr])
                    self.realtable[i * 2][1].text_size[0] = self.realtable[i * 2][1].size[0] * (1 + max(len(Expr[Exr])-5 + len(str(" = %.3f" % Expression_table[Exr])) - 5, 0) / 5)
                    self.realtable[i * 2][1].halign = "right"
                    self.realtable[i * 2 + 1][1].text = "± %.3f" % er[V]
                    self.realtable[i * 2 + 1][1].text_size[0] = self.realtable[i * 2][1].size[0] * (1 + max(len(Expr[Exr])-5 + len(str(" = %.3f" % Expression_table[Exr])) - 5, 0) / 5)
                    self.realtable[i * 2 + 1][1].halign = "right"
                    self.realtable[i * 2 + 1][1].color = [1, 1, 1, 1]
                    if er[V] > 9999:
                        er[V] = np.inf
                        self.realtable[i * 2 + 1][1].color = [1, 255, 255, 1]
                    Exr += 1
                    V += 1
                else:
                    # self.realtable[i * 2][1].text_size[0] = self.realtable[i * 2][1].text_size[0]
                    self.realtable[i * 2][1].text_size = (None, None)
                    self.realtable[i * 2][1].halign = "center"
                    self.realtable[i * 2 + 1][1].text_size = (None, None)
                    self.realtable[i * 2 + 1][1].halign = "center"


                if model[i-1] == 'Distr':
                    for j in range(1, 5):
                        self.realtable[i * 2][j].color = [1, 1, 1, 1]
                        self.realtable[i * 2 + 1][j].color = [1, 1, 1, 1]
                        # for k in range(i+Vn,  99):
                        #     try:
                        #         l = str(name[k][j-1]).split(',')[0]
                        #         break
                        #     except IndexError:
                        #         Vn += 1
                        l = str(name[i][j - 1]).split(',')[0]
                        self.realtable[i*2][j].text = l + ' = ' +  "%.3f" % p[V]
                        if p[V] > 9999:
                            p[V] = np.inf
                            self.realtable[i * 2][j].color = [1, 255, 255, 1]

                        # if str(er[V]) == 'nan' and self.realtableP[i][j].text[0] == '=':
                        #     p_link_number, link_multiplier = str(self.realtableP[i][j].text[2:-1]).split(',')
                        #     # print(p_link_number, link_multiplier)
                        #     try:
                        #         er[V] = float(er[int(p_link_number)]) * float(link_multiplier)
                        #         # print(er[V])
                        #     except:
                        #         pass
                        self.realtable[i * 2 + 1][j].text = "± %.3f" % er[V]
                        if er[V] > 9999:
                            er[V] = np.inf
                            self.realtable[i*2+1][j].color = [1, 255, 255, 1]
                        V+=1
                    self.realtable[i * 2][5].text = Distri[Di]
                    self.realtable[i * 2][5].text_size[0] = self.realtable[i * 2][5].size[0] * (1 + max(len(Distri[Di])-5 , 0) / 5)
                    self.realtable[i * 2][5].halign = "right"
                    Di += 1
                    V+=1
                else:
                    # self.realtable[i * 2][5].text_size = (None, None)
                    self.realtable[i * 2][5].text_size[0] = self.realtable[i * 2][4].text_size[0]
                    self.realtable[i * 2][5].text_size = (None, None)
                    self.realtable[i * 2][5].halign = "center"

                if model[i-1] == 'Corr':
                    for j in range(1, 2):
                        self.realtable[i * 2][j].color = [1, 1, 1, 1]
                        self.realtable[i * 2 + 1][j].color = [1, 1, 1, 1]
                        # for k in range(i+Vn,  99):
                        #     try:
                        #         l = str(name[k][j-1]).split(',')[0]
                        #         break
                        #     except IndexError:
                        #         Vn += 1
                        l = str(name[i][j - 1]).split(',')[0]
                        self.realtable[i*2][j].text = l + ' = ' +  "%.3f" % p[V]
                        if p[V] > 9999:
                            p[V] = np.inf
                            self.realtable[i * 2][j].color = [1, 255, 255, 1]
                        self.realtable[i * 2 + 1][j].text = "± %.3f" % er[V]
                        if er[V] > 9999:
                            er[V] = np.inf
                            self.realtable[i*2+1][j].color = [1, 255, 255, 1]
                        V+=1
                    self.realtable[i * 2][2].text = Cor[Co]
                    self.realtable[i * 2][2].text_size[0] = self.realtable[i * 2][2].size[0] * (1 + max(len(Cor[Co])-5 , 0) / 5)
                    self.realtable[i * 2][2].halign = "right"
                    Co += 1
                    V+=1
                else:
                    # self.realtable[i * 2][5].text_size = (None, None)
                    self.realtable[i * 2][2].text_size[0] = self.realtable[i * 2][1].text_size[0]
                    self.realtable[i * 2][2].text_size = (None, None)
                    self.realtable[i * 2][2].halign = "center"

            print(T_table)
            print(erT_table)
            print(T_full)
            print(spc_table)
            spc_counter = 0
            for i in range(0, len(T_table)):
                for k in range(0, len(T_table[i])):
                    self.realtable[spc_table[spc_counter] * 2 + 1][0].color = self.realtable[spc_table[spc_counter] * 2][0].color
                    erT = 0
                    for j in range(0, len(erT_table[i])):
                        erT += (k != j)*(float(T_table[i][k])*float(erT_table[i][j]))**2 + (k == j)*((T_full[i] - float(T_table[i][k]))*float(erT_table[i][k]))**2
                    erT = np.sqrt(erT) / (T_full[i] ** 2) * 100
                    self.realtable[spc_table[spc_counter] * 2 + 1][0].text = "%.1f" % float(100*float(T_table[i][k])/T_full[i]) + " ± %.1f" % erT + ' %'
                    spc_counter += 1

        RR = False


    if self.realtableP[0][NBA+1].text == 'online':
        self.realtableP[0][NBA+1].background_color = [0, 255, 0, 1]
        self.realtableP[0][NBA+2].background_color = [0, 255, 0, 1]
        self.realtableP[0][NBA+3].background_color = [0, 255, 0, 1]
        self.realtableP[0][NBA+4].background_color = [0, 255, 0, 1]
        self.lboundstable[0][NBA+1].text = str('T')
        self.rboundstable[0][NBA+1].text = str('int')
        self.lboundstable[0][NBA+2].text = str('chi2')
        self.rboundstable[0][NBA+2].text = str('max')
        self.lboundstable[0][NBA+3].text = str('BG')
        self.rboundstable[0][NBA+3].text = str('min')
        try:
            check = float(self.realtableP[0][NBA+2].text)#
            if float(self.realtableP[0][NBA+2].text) < 60:#
                self.realtableP[0][NBA+2].background_color = [2, 0.5, 0, 1]#
                self.play_btn.disabled = True
        except:
            self.realtableP[0][NBA+2].background_color = [255, 0, 0, 1]#
            self.play_btn.disabled = True
    else:
        self.realtableP[0][NBA+1].background_color = [255, 255, 255, 1]#
        self.realtableP[0][NBA+2].background_color = [255, 255, 255, 1]#
        self.realtableP[0][NBA+3].background_color = [255, 255, 255, 1]#
        self.realtableP[0][NBA+4].background_color = [255, 255, 255, 1]#
        self.lboundstable[0][NBA+1].text = str('')#
        self.lboundstable[0][NBA+2].text = str('')#
        self.lboundstable[0][NBA+3].text = str('')#
        self.rboundstable[0][NBA+1].text = str('')#
        self.rboundstable[0][NBA+2].text = str('')#
        self.rboundstable[0][NBA+3].text = str('')#

    if RL == 1:
        global tic, tac
        # tic = int(tac)
        self.log.text = int((tic - 2)*(tic - 3)/2)   *"\\\t"\
                      + int((tic - 1)*(tic - 3)*(-1))*"/\t"\
                      + int((tic - 1)*(tic - 2)/2)   *"—\t" \
                      + (4 - tic - 1) * ' ' + tic * '.'+ 'Fitting in progress' + tic * '.' + (4-tic-1) * ' '\
                      + int((tic - 2)*(tic - 3)/2)   *"\t\\"\
                      + int((tic - 1)*(tic - 3)*(-1))*"\t/"\
                      + int((tic - 1)*(tic - 2)/2)   *"\t—"
        self.log.background_color=[1, 0, 0.3, 1]
        # tac += 0.025
        # if tac == 4:
        #     tac = 1
        tic += 1
        if tic == 4:
            tic = 1
    if RL == 2:
        self.log.text = 'Finished'
        self.log.background_color = [0, 255, 0, 1]
        self.left.canvas.ask_update()
        RL = 3
    if RL == 4:
        self.log.text = "Calibration.dat was rewritten and will be used for RAW"
        self.log.background_color = [0, 255, 0, 1]
        self.left.canvas.ask_update()
        RL = 3
    if RL == 0:
        self.log.text = 'INTERRUPTED'
        self.log.background_color = [255, 255, 0, 1]
        self.left.canvas.ask_update()
        RL = 3
    if RL == 5:
        self.log.background_color = [1, 0.5, 0, 1]
        self.log.text = int((tic - 2) * (tic - 3) / 2) * "\\\t" \
                        + int((tic - 1) * (tic - 3) * (-1)) * "/\t" \
                        + int((tic - 1) * (tic - 2) / 2) * "—\t" \
                        + (4 - tic - 1) * ' ' + tic * '.' + 'Waiting' + tic * '.' + (4 - tic - 1) * ' ' \
                        + int((tic - 2) * (tic - 3) / 2) * "\t\\" \
                        + int((tic - 1) * (tic - 3) * (-1)) * "\t/" \
                        + int((tic - 1) * (tic - 2) / 2) * "\t—"
        tic += 1
        if tic == 4:
            tic = 1

    if Rlog != '':
        self.log.text = Rlog
        Rlog = ''
    if RlogCol != []:
        self.log.background_color = RlogCol
        RlogCol = []
    global L0_text
    if L0_text != '':
        self.L0.text = L0_text
        L0_text = ''
    # global table_save
    # if table_save != '':
    #     # tableR scrollboxR right
    #     self.tableR.export_to_png(table_save)
    #     table_save = ''

    # print('one frame time:', time.time() - start_time, ', FPS: ', 1/(time.time() - start_time))

def initialization (self, *args):
    global initial, x0, MulCo, RL, PPP
    if platform.system() == 'Windows':
        realpath = str(self.dir_path) + str('\\\\INSint.txt')
    else:
        realpath = str(self.dir_path) + str('/INSint.txt')
    MulCo, x0 = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
    print(MulCo, x0)
    initial = 1
    return()


def read_model(self):
    global val
    global model
    global p
    model = []
    p = np.array([], dtype=float)
    val = self.params
    con1 = np.array([], dtype=float)
    con2 = np.array([], dtype=float)
    con3 = np.array([], dtype=float)
    Distri = []
    Cor = []
    Expr = []
    NExpr = np.array([], dtype=int)
    DistriN = np.array([], dtype=float)
    fix_tmp = np.array([], dtype=int)

    for i in range(1, NBA+1): # baseline reader
        if self.realtableP[0][i].text[0:2] == '=[' and self.realtableP[0][i].text[-1] == "]":
            b = np.array((self.realtableP[0][i].text[2:-1]).split(','))
            con1 = np.append(con1, len(p))
            con2 = np.append(con2, float(b[0]))
            con3 = np.append(con3, float(b[1]))
            p = np.append(p, 1)
        else:
            p = np.append(p, float(self.realtableP[0][i].text))


    for i in range(0, len(val)):
        if str(val[i][0].text) != 'None' and str(val[i][0].text) != 'baseline':
            model.append(str(val[i][0].text))
        LenM = mod_len_def_M(str(val[i][0].text)) + 1
        for j in range(1, LenM):
            if val[i][j].text[0:2] == '=[' and val[i][j].text[-1] == "]":
                b = np.array((val[i][j].text[2:-1]).split(','))
                con1 = np.append(con1, len(p))
                con2 = np.append(con2, float(b[0]))
                con3 = np.append(con3, float(b[1]))
                p = np.append(p, 1)
                print(b[0],b[1])
            else:
                p = np.append(p, float(val[i][j].text))

        if str(val[i][0].text) == 'Expression':
            Expr.append(str(val[i][1].text))
            NExpr = np.append(NExpr, len(p))
            p = np.append(p, 0)
        if str(val[i][0].text) == 'Distr':
            for j in range(1, 5):
                if val[i][j].text[0:2] == '=[' and val[i][j].text[-1] == "]":
                    b = np.array((val[i][j].text[2:-1]).split(','))
                    con1 = np.append(con1, len(p))
                    con2 = np.append(con2, float(b[0]))
                    con3 = np.append(con3, float(b[1]))
                    p = np.append(p, 1)
                else:
                    p = np.append(p, float(val[i][j].text))
            p = np.append(p, 0)
            Distri.append(str(val[i][5].text))

            DistriN = np.append(DistriN, len(p) - 1)
            N = 0
            M = 0
            for k in range(1, i):
                M = i - k
                if model[i-k] != 'Distr' and model[i-k] != 'Corr':
                    M += 1
                    break
            for k in range(0, M):
                # C = str(val[k][0].text)
                N += mod_len_def(str(val[k][0].text)) + int(NBA * (k == 0))
            N += int(str(val[i][1].text))
            # print(N)
            fix_tmp = np.append(fix_tmp, N)


        if str(val[i][0].text) == 'Corr':
            for j in range(1, 2):
                if val[i][j].text[0:2] == '=[' and val[i][j].text[-1] == "]":
                    b = np.array((val[i][j].text[2:-1]).split(','))
                    con1 = np.append(con1, len(p))
                    con2 = np.append(con2, float(b[0]))
                    con3 = np.append(con3, float(b[1]))
                    p = np.append(p, 1)
                else:
                    p = np.append(p, float(val[i][j].text))
            p = np.append(p, 0)
            Cor.append(str(val[i][2].text))

            DistriN = np.append(DistriN, len(p) - 1)
            N = 0
            M = 0
            for k in range(1, i):
                M = i - k
                if model[i - k] != 'Distr' and model[i - k] != 'Corr':
                    M += 1
                    break
            for k in range(0, M):
                # C = str(val[k][0].text)
                N += mod_len_def(str(val[k][0].text)) + int(NBA * (k == 0))
            N += int(str(val[i][1].text))
            # print(N)
            fix_tmp = np.append(fix_tmp, N)

    return(con1, con2, con3, Distri, Cor, Expr, NExpr, fix_tmp, DistriN)

def read_spectrum(self, file):
    A_list = []
    B_list = []
    A = np.array([float(0)])
    B = np.array([float(0)])

    if str(file)[-1] == '\\' or str(file)[-1] == '/' or str(file)[-1] == '.':
        try:
            path_tmp_local = []
            for files in os.listdir(str(file)):
                if files.endswith(".dat") or files.endswith(".mca"):
                    path_tmp_local.append(os.path.join(str(file), files))
            for i in range(0, len(path_tmp_local)):
                if os.path.exists(path_tmp_local[i]) == False:
                    self.log.background_color = [255, 255, 0, 1]
                    self.log.text = "At least one path do not exist"
                else:
                    file = path_tmp_local[0]
        except:
            check = False
            self.log.background_color = [255, 255, 0, 1]
            self.log.text = "Directory do not exist"

    in_format_list = True
    if file.casefold().endswith(tuple(acceptable_formats))==False:
        in_format_list = False


    if file[-4:] == '.dat' or file[-4:] == '.txt' or file[-4:] == '.exp' or in_format_list == False:
        try:
            with open(file, 'r') as catalog:
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                for line in lines:
                    column = line.split()
                    if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                        x = float(column[0])
                        y = float(column[1])
                        A_list.append(x)
                        B_list.append(y)
            B = np.array(B_list)
            if self.switch.active == False:  # NFS version
                NFS_cal = np.genfromtxt(str(self.dir_path) +
                                        str('\\\\NFS.txt') * (platform.system() == 'Windows') +
                                        str('/NFS.txt') * (platform.system() != 'Windows'),
                                        delimiter='\t', skip_footer=0)
                k = 0
                for i in range(0, len(A_list)):
                    if A_list[k] < NFS_cal[3] or A_list[k] > NFS_cal[4]:
                        A_list.pop(k)
                        B_list.pop(k)
                        k -= 1
                    k += 1
                    if k >= len(A_list):
                        break
                B = np.array(B_list)
        except:
            if in_format_list == False:
                self.log.text = "File could not be open as two column. Please check the file."
            else:
                self.log.text = "Unexpected problem problem while opening file. Please checl the file."
    if file[-4:] == '.mca' or file[-5:] == '.cmca' or self.process_path.text == 'tango':
        # if platform.system() == 'Windows':
        #     rpath = str(self.dir_path) + str('\\\\Calibration.dat')
        # else:
        #     rpath = str(self.dir_path) + str('/Calibration.dat')
        rpath = self.cal_path
        with open(rpath, 'r') as catalog:
            lines = (line.rstrip() for line in catalog)
            lines = (line for line in lines if line)  # skipping white lines
            for line in lines:
                column = line.split()
                if not line.startswith('#'):  # skipping column labels
                    x = float(column[0])
                    A_list.append(x)

        cal_type = open(rpath, 'r')
        try:
            cal_info = (cal_type.readline()).split()[1:]
            cal_method = cal_info[0]
            n1 = int(cal_info[1])
            n2 = int(cal_info[2])
        except:
            cal_method = str('sin')
            n1 = 0
            n2 = int(len(np.array(A_list)))*2 - 1
            print('No information in first line of calibration')
        cal_type.close()

        if self.process_path.text == 'tango':
            id = [[0],[0]]
            print('TANGO is a lie!')
            # print('TANGO is alive!')
            # id = [get_data(tango_uri)]
            # id = np.array(id, dtype=float)
            # print('TANGO was read!')

        else:
            LS = len(open(file, 'r').readlines())
            with open(file, 'r') as fi:
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
            id = np.array(id, dtype=float)


        # B = id[0][::-1]
        # id_tmp = np.array([float(0)] * int(len(B) / 2))
        # for i in range(0, len(id_tmp)):
        #     id_tmp[i] = B[i] + B[len(B) - 1 - i]
        # B = id_tmp
        if self.switch.active == True:
            if file[-5:] == '.cmca':
                if (id[-1][-1] == 0 and id[-1][0] != 0) or (id[-1][0] == 0 and id[-1][-1] != 0) :
                    id_half = (id[-1][-1]+id[-1][0])/2
                    id[-1][-1] = id_half
                    id[-1][0] = id_half
            try:
                if cal_method == str('sin'):
                    spc_1h = id[-1][:int(n1 / 2)] + id[-1][n1 - int(n1 / 2):n1][::-1]
                    spc_2h = id[-1][n2+1:n2 + int((len(id[-1]) - 1 - n2) / 2)+1][::-1] + id[-1][len(id[-1]) - int((len(id[-1]) - 1 - n2) / 2):len(id[-1])]
                    spc_3h = id[-1][n1:n1 + int((n2 - n1 + 1) / 2)] + id[-1][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
                    B = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
                if cal_method == str('lin'):
                    # B = np.array([float(0)] * int((n2 - n1 + 1) / 2))
                    # for i in range(0, int(len(id[-1]) / 2) - n1):
                    #     B[i] = id[-1][n1 + i] + id[-1][n2 - i]
                    ### new correction
                    n_sh = 2 * (n1 + (int(len(id[-1])) - 1 - n2))
                    B = np.array([float(0)] * int((len(id[-1]) - n_sh) / 2))
                    for i in range(0, int((len(id[-1]) - n_sh) / 2)):
                        B[i] = id[-1][n1 + i] + id[-1][n2 - i]
            except:
                self.points_match = False
                B = np.array([float(1)]*(len(A_list)+1))
                print('something is wrong with number of points in *.mca or calibration.dat')

        if self.switch.active == False:  # NFS version
            NFS_cal = np.genfromtxt(str(self.dir_path) +
                                    str('\\\\NFS.txt') * (platform.system() == 'Windows') +
                                    str('/NFS.txt') * (platform.system() != 'Windows'),
                                    delimiter='\t', skip_footer=0)
            A_list = []
            # for i in range(0, len(id)):
            #     A_n.append([])
            for j in range(0, len(id[-1])):
                A_list.append(NFS_cal[0] + NFS_cal[1] * j + NFS_cal[2] * j ** 2)
            # A = np.copy(A_n)
            B = np.copy(id[-1])
    if file[-4:] == '.ws5' or file[-4:] == '.w98' or file[-4:] == '.moe' or file[-3:] == '.m1' or file[-4:].casefold() == '.mcs' : # Wissel files

        rpath = self.cal_path
        with open(rpath, 'r') as catalog:
            lines = (line.rstrip() for line in catalog)
            lines = (line for line in lines if line)  # skipping white lines
            for line in lines:
                column = line.split()
                if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                    x = float(column[0])
                    A_list.append(x)


        cal_type = open(rpath, 'r')
        try:
            cal_info = (cal_type.readline()).split()[1:]
            cal_method = cal_info[0]
            n1 = int(cal_info[1])
            n2 = int(cal_info[2])
        except:
            cal_method = str('sin')
            n1 = 0
            n2 = int(len(np.array(A_list)))*2 - 1
            print('No information in first line of calibration')
        cal_type.close()

        if file[-4:].casefold() == '.mcs':
            f = open(file, mode='rb')
            id = []
            entete1 = f.read(256)
            array = np.fromfile(f, dtype=np.uint32)
            print(len(array))
            id.append(array)
            f.close()
        else:
            with open(file, 'r') as catalog:
                id = []
                id.append([])
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                k = 0
                for line in lines:
                    if file[-3:] == '.m1':
                        while '  ' in line:
                            line = line.replace('  ', ' ')
                        column = line.split(' ')
                        x = float(column[4])
                        if k > 0:
                            id[0].append(x)
                        k += 1
                    else:
                        column = line.split()
                        if not (line.startswith('#') or line.startswith('<')):  # skipping column labels
                            x = float(column[0])
                            if file[-4:] != '.moe' or not('.' in str(column[0])):
                                id[0].append(x)
        id = np.array(id, dtype=float)

        if self.switch.active == True:
            try:
                if cal_method == str('sin'):
                    spc_1h = id[-1][:int(n1 / 2)] + id[-1][n1 - int(n1 / 2):n1][::-1]
                    spc_2h = id[-1][n2+1:n2 + int((len(id[-1]) - 1 - n2) / 2)+1][::-1] + id[-1][len(id[-1]) - int((len(id[-1]) - 1 - n2) / 2):len(id[-1])]
                    spc_3h = id[-1][n1:n1 + int((n2 - n1 + 1) / 2)] + id[-1][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
                    B = np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h))
                if cal_method == str('lin'):
                    # B = np.array([float(0)] * int((n2 - n1 + 1) / 2))
                    # for i in range(0, int(len(id[-1]) / 2) - n1):
                    #     B[i] = id[-1][n1 + i] + id[-1][n2 - i]
                    ### new correction
                    n_sh = 2 * (n1 + (int(len(id[-1])) - 1 - n2))
                    B = np.array([float(0)] * int((len(id[-1]) - n_sh) / 2) )
                    for i in range(0, int((len(id[-1]) - n_sh) / 2)):
                        B[i] = id[-1][n1 + i] + id[-1][n2 - i]
            except:
                self.points_match = False
                B = np.array([float(1)]*(len(A_list)+1))
                print('something is wrong with number of points in *.mca or calibration.dat')
        if self.switch.active == False:
            print('this file suppose to have CMS spectrum not NFS')
            self.log.background_color = [255, 255, 0, 1]
            self.log.text = "Please switch to energy domain"


    A = np.array(A_list)
    return (A, B)

def create_subspectra(self, model, Distri, Cor, p):
    Ps = []
    Psm = []
    Distri_t = []
    Cor_t = []
    Di = 0
    Co = 0
    Ex = 0
    V = NBA
    # global p

    for i in range(0, len(model)):
        ps = np.array([p[0:NBA]], dtype=float)
        Psm.append([model[i]])
        LenM = mod_len_def_M(model[i]) + 1
        for j in range(1, LenM):
            ps = np.append(ps, p[V])
            V += 1

        Ps.append(ps)
        if model[i] == 'Expression':
            del Ps[-1]
            ps = np.append(ps, 1.0)
            Ps.append(ps)
            V += 1
        if model[i] == 'Distr':
            del Ps[-1]
            for j in range(1, 6):
                Ps[-1] = np.append(Ps[-1], p[V])
                V += 1
            del Psm[-1]
            Psm[-1].append(model[i])
            STR = Distri[Di] + str(' ')
            # print(STR)
            st_ = []
            en_ = []
            for k in range(0, len(STR) - 2):
                if STR[k] == 'p' and STR[k + 1] == '[':
                    st_.append(k)
                    for kk in range(k, len(STR)):
                        if STR[kk] == ']':
                            en_.append(kk)
                            break
            st_ = st_[::-1]
            en_ = en_[::-1]
            for k in range(0, len(st_)):
                STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
            Distri_t.append(STR)
            Di += 1
        # else:
        #     Distri_t.append([])

        if model[i] == 'Corr':
            del Ps[-1]
            for j in range(1, 3):
                Ps[-1] = np.append(Ps[-1], p[V])
                V += 1
            del Psm[-1]
            Psm[-1].append(model[i])
            STR = Cor[Co] + str(' ')
            # print(STR)
            st_ = []
            en_ = []
            for k in range(0, len(STR) - 2):
                if STR[k] == 'p' and STR[k + 1] == '[':
                    st_.append(k)
                    for kk in range(k, len(STR)):
                        if STR[kk] == ']':
                            en_.append(kk)
                            break
            st_ = st_[::-1]
            en_ = en_[::-1]
            for k in range(0, len(st_)):
                STR = str(STR[:st_[k]]) + str(eval(STR[st_[k]:en_[k] + 1])) + str(STR[(en_[k] + 1):])
            Cor_t.append(STR)
            Co += 1


        # if model[i] == 'Distr':
        #     del Distri_t[-2]
        #
        # if model[i] == 'Corr':
        #     del Distri_t[-1]
    return (Ps, Psm, Distri_t, Cor_t, Di, Co)


# def work(self, _aa):
#     global RA, RL, TR
#     # RL = 1
#     RPM = numro - 1
#     for i in range(1, len(self.realtableP)):
#         if self.realtableP[len(self.realtableP)-1-i][0].text != 'None':
#             RPM = len(self.realtableP)-1-i
#             break
#     for i in range(1, RPM):
#         if self.realtableP[RPM-i][0].text == 'None':
#             self.select(self.realtableP[RPM-i][0], 'Delete', RPM-i, 0)
#
#     TR = threading.Thread(target=partial(work2, self))
#     TR.name = 'TTT'
#     TR.daemon = True
#     TR.start()

def work2(self):

    global x0, MulCo, RA, RL, RR, Rlog, RlogCol
    RL = 1
    # self.play_btn.background_color = [0, 0.5, 0, 1]
    # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
    # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
    # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
    # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
    # self.play_btn.disabled = True
    # self.INS_btn.disabled = True
    # self.INS_btn2.disabled = True
    # self.show_btn.disabled = True
    # self.showM_btn.disabled = True
    # self.save_path.readonly = True
    # self.cal_btn.disabled = True

    # self.switch.disabled = True




    v = 0
    vv = 0
    raw_path = self.process_path.text[2:-2]
    # print(raw_path)
    # print(str(raw_path)[-1])
    self.path_list = []

    if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/':
        try:
            for file in os.listdir(str(raw_path)):
                # if file.endswith(".txt"):
                #     self.path_list.append(os.path.join(str(raw_path), file))
                if file.endswith(".dat"):
                    self.path_list.append(os.path.join(str(raw_path), file))
            check = True
            for i in range(0, len(self.path_list)):
                if os.path.exists(self.path_list[i]) == False:
                    check = False
                    RlogCol = [255, 255, 0, 1]
                    Rlog = "At least one path do not exist"
            if ("." in str(self.save_path.text)) == True:
                check = False
                RlogCol = [255, 255, 0, 1]
                Rlog = "Directory path could not contain dots"
            dir_ch = 0
            raw_path_ch = (raw_path.replace("\\\\", "\\"))[:-1]
            dir_path_ch = os.path.dirname(str(self.save_path.text))
            # print(raw_path_ch)
            # print(dir_path_ch)
            if raw_path_ch == dir_path_ch:
                dir_ch = 1

            if (str(self.save_path.text)[-1] != '\\' or str(self.save_path.text)[-1] != '/') and dir_ch != 0:
                    self.save_path.text = str(self.save_path.text) + str('\\')*(platform.system() == 'Windows') + str('/')*(platform.system() != 'Windows')

            if raw_path_ch == self.save_path.text:
                self.save_path.text = self.save_path.text + str('result')\
                                        + str('\\')*(platform.system() == 'Windows') + str('/')*(platform.system() != 'Windows')



        except:
            check = False
            RlogCol = [255, 255, 0, 1]
            Rlog = "Directory do not exist"
    else:
        for i in range(0, len(raw_path) - 2):
            if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                vv = i
                self.path_list.append(str(raw_path)[v:vv])
                v = i + 4
        self.path_list.append(str(raw_path)[v:])
        check = True
        for i in range(0, len(self.path_list)):
            if os.path.exists(self.path_list[i]) == False:
                check = False
                RlogCol = [255, 255, 0, 1]
                Rlog = "At least one path do not exist"
    # if len(self.path_list) > 1:
    #     if platform.system() == 'Windows':
    #         if self.save_path.text[-1] != '\\':
    #             check = False
    #             self.log.background_color = [255, 255, 0, 1]
    #             self.log.text = "Save pass should be folder while fitting sequence"
    #     else:
    #         if self.save_path.text[-1] != '/':
    #             check = False
    #             self.log.background_color = [255, 255, 0, 1]
    #             self.log.text = "Save pass should be folder while fitting sequence"

    if self.process_path.text == 'tango' and check_tango == True:
        check = True

    if check == True:
        RL = 1

        if self.seq_fit[0].active == True:
            SeF = 1
        if self.seq_fit[1].active == True:
            SeF = 2
        global val
        global model
        global p
        global Distri
        global Cor
        global Expr, NExpr

        con1, con2, con3, Distri, Cor, Expr, NExpr, fix_tmp, DistriN = read_model(self)

        for i in range(0, len(NExpr)):
            p[NExpr[i]] = eval(Expr[i])
        for i in range(0, len(con1)):
            p[int(con1[i])] = p[int(con2[i])] * con3[i]
        # for i in range(0, len(NExpr)): # repeat for cases of two or more multiplication?expression bonds
        #     p[NExpr[i]] = eval(Expr[i])
        # for i in range(0, len(con1)):
        #     p[int(con1[i])] = p[int(con2[i])] * con3[i]


        global confu

        if len(con1) > 0:
            confu = np.array([con1, con2, con3])
        else:
            confu = np.array([[-1], [-1], [-1]])

        global p0, er, name

        name = [[]]
        for i in range(0, len(self.nametable)):
            tmpname = []
            for j in range(0, len(self.nametable[i])):
                tmpname.append(self.nametable[i][j].text)
            name.append(tmpname)
        name.pop(0)





        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        fix = np.array([], dtype=int)

        for j in range(0, NBA):
            if self.rboundstable[0][j].text != 'None' and self.rboundstable[0][j].text != '':
                # print(j)
                bounds[1][j] = float(self.rboundstable[0][j].text)
            if self.lboundstable[0][j].text != 'None' and self.lboundstable[0][j].text != '':
                bounds[0][j] = float(self.lboundstable[0][j].text)
            if self.fixtable[0][j].active == True:
                fix = np.append(fix, j)

        V = NBA-1

        for i in range(0, len(val)):
            LenM = mod_len_def(str(val[i][0].text))
            for j in range(0, LenM):
                V += 1
                if self.rboundstable[i][j].text != 'None' and self.rboundstable[i][j].text != '':
                    bounds[1][V] = float(self.rboundstable[i][j].text)
                if self.lboundstable[i][j].text != 'None' and self.lboundstable[i][j].text != '':
                    bounds[0][V] = float(self.lboundstable[i][j].text)
                if self.fixtable[i][j].active == True:
                    fix = np.append(fix, V)

        # print(bounds)

        fix = np.concatenate((fix, fix_tmp), axis=0) # fix_tmp do not work correctly with multidimensional distri
        fix = np.concatenate((fix, con1), axis=0)
        fix = np.concatenate((fix, DistriN), axis=0)
        fix = np.concatenate((fix, NExpr), axis=0)

        # print('fix_tmp')
        # print(fix_tmp)
        # print('con1')
        # print(con1)
        # print('DistriN')
        # print(DistriN)
        # print('NExpr')
        # print(NExpr)

        fix = np.unique(fix)

        # print(fix)
        VVV = 0
        Li_Lo = 0
        tau0 = 10 ** -3
        eps = 10 ** -6
        Es = 1
        JN = int(self.JN0.text)

        if self.fitway[0].active == True:
            VVV = 1
            Li_Lo = 1
            # EEs = 0.999*2/(int(self.JN0.text)-1)
            # cof = np.array([0.00254718, -0.00993933, 0.64000022, 0.10557177, 0.20270726])
            # Es =  cof[0] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** (1 / 4) \
            #     + cof[1] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** (1 / 2) \
            #     + cof[2] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 1 \
            #     + cof[3] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 2 \
            #     + cof[4] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 3
            # Es = Es/EEs # correction for integral in zero point if odd points entered - pass instead x0 because x0 is not used in this method
            INS = float(self.L0.text)
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoCMS, INS, [0], [0], Met=1)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, Distri, Cor, Met=1, Norm=Norm)
                #return m5.PV(x, p, model, pool)

        if self.fitway[1].active == True:
            VVV = 3
            Li_Lo = 1 # was 2 for some reason??? still did not work as further there is check for lilo & VVV
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\INSexp.txt')
            else:
                realpath = str(self.dir_path) + str('/INSexp.txt')
            INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
            pNorm = np.array([float(0)]*NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, x0, MulCo, INS, Distri, Cor, Norm=Norm)

        if self.fitway[1].active == True and str(self.L0.text)[:2] == 'L=':  # lorenz squared for Ilya Sergeev
            VVV = 6
            Li_Lo = 1
            INS = float(str(self.L0.text)[2:])
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoL2, INS, [0], [0], Met=3)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoL2, INS, Distri, Cor, Met=3, Norm=Norm)


        if self.fitway[2].active == True:
            VVV = 5
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\INS_APS.txt')
            else:
                realpath = str(self.dir_path) + str('/INS_APS.txt')
            INS = np.genfromtxt(realpath, delimiter='\t', skip_footer=0)
            print(INS)
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCoCMS, INS, [0], [0], Met=2)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, Distri, Cor, Met=2, Norm=Norm)

        if self.switch.active == False:
            VVV = 4
            tau0 = 10 ** -4
            eps = 10 ** -10
            def func(x, p):
                return mN.TI(x, p, model, pool)
            def funcL(x, p):
                return np.log(mN.TI(x, p, model, pool) + 1)*10+10000

        chi2max = 100
        BGmin = 100
        try:
            if self.realtableP[0][NBA+3].text != '':
                chi2max = float(self.realtableP[0][NBA+3].text)
        except:
            pass
        try:
            if self.realtableP[0][NBA+4].text != '':
                BGmin = float(self.realtableP[0][NBA+4].text)
        except:
            pass

        online_number = 0
        if self.realtableP[0][NBA+1].text == 'online':
            self.online_fit = True
            if self.realtableP[0][NBA+2].text != '':
                # if float(self.realtableP[0][NBA+2].text) < 60:
                #     self.realtableP[0][NBA+2].text = '60'
                self.time_interval = int(self.realtableP[0][NBA+2].text)
        else:
            self.online_fit = False

        p0 = np.copy(p)
        for l_p in range (0, (len(self.path_list) + 10**5*self.online_fit)*(1-(model.count('Nbaseline')>0)) + (model.count('Nbaseline')>0)):
          if RL == 1:
            online_start_time = time.time()
            l_p = l_p*(1-self.online_fit)
            file = os.path.abspath(self.path_list[l_p])
            A, B = read_spectrum(self, file)

            for i in range(1, model.count('Nbaseline') + 1):
                try:
                    file = os.path.abspath(self.path_list[i])
                    A1, B1 = read_spectrum(self, file)
                    A = np.concatenate((A, A1))
                    B = np.concatenate((B, B1))
                except:
                    pass


            self.X_axis = A
            self.SPC_plot = B

            if VVV == 4:
                A = A * 10 ** -9
                if A[0] > A[-1]:
                    A = A[::-1]
                    B = B[::-1]

            if l_p + self.online_fit > 0:
                # if (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10 < (max(B) - np.sqrt(max(B))) * 0.98:
                #     p0[0] = max(B) - np.sqrt(max(B))
                # else:
                    if p0[0] <= 0:
                        p0[0] = 1
                    BG_old = p0[0]
                    p0[0] = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10 * (p0[0]/(p0[0]+p0[3])) + 1
                    p0[3] = p0[0]/BG_old * p0[3]
                    if VVV == 4:
                        p[0] = max(B)
            print(p)
            print(model)
            start_time = time.time()




            global hi2
            if VVV != 4 or Li_Lo == 1:
                p, er, hi2 = mi.minimi_hi(func,  A, B, p0, fix = fix, confu=confu,  bounds = bounds, Expr = Expr, NExpr = NExpr, MI=20, MI2=10, nu0=2.618, tau0=tau0, eps=eps)
            if VVV == 4 and Li_Lo == 2:
                p, er, hi2 = mi.minimi_hi(funcL,  A, np.log(B+1)*10+10000, p0, fix = fix, confu=confu,  bounds = bounds, Expr = Expr, NExpr = NExpr, MI=20, MI2=10, nu0=2.618, tau0=tau0, eps=eps)
                hi2 = np.sum((B-func(A, p))**2 / (abs(B) + 1) /(len(B)-V+len(fix)))
                # p, er, hi2 = mi.minimi_hi(func, A, B, p0, fix=fix, bounds=bounds, MI=20, MI2=10, nu0=2.618, tau0=tau0, eps=eps)
            if hi2 > 1.25 and VVV != 4:
                print('hi2 is too high let me try to continue')
                pO, erO, hi2O = p, er, hi2
                p, er, hi2 = mi.minimi_hi(func,  A, B, p, fix = fix, confu=confu,  bounds = bounds, Expr = Expr, NExpr = NExpr, MI=20, MI2=10, nu0=2.618, tau0=tau0, eps=eps)
                # print('baseline error ', er[0], ' or ', erO[0])
                if np.array_equal(p, pO) == True:
                    if len(er) == 1:
                        p, er, hi2 = pO, erO, hi2O
                    print('It was real end')




            SPC_f = func(A, p)
            Total_SPC = SPC_f
            print('minimization takes', time.time() - start_time, 'seconds')

            print(p)
            print(er)
            Cur_model_len = 0
            subspc = []

            if model.count('Nbaseline') == 0:
                number_of_spectra = 1
                self.Model_full_plot = SPC_f
                fig, ax1 = plt.subplots(figsize=(2942 / 300 * number_of_spectra, 4.5), dpi=300)

                # fig = plt.figure(figsize=(2942/300*number_of_spectra, 4.5), dpi=300)
                # ax = fig.add_subplot(111)

                Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(self, model, Distri, Cor, p)

                # Ps_t = np.copy(Ps)
                # subspc = []
                CoEn = 0
                DiEn = 0
                FS = np.array([[float(0)] * len(A)] * len(Ps))
                self.FS_pos = []
                for i in range(0, len(Ps)):
                    CoSt = CoEn
                    DiSt = DiEn
                    CoEn += Psm[i].count('Corr')
                    DiEn += Psm[i].count('Distr')
                    # print(Psm[i], Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn])
                    if VVV == 1:
                        FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=1, Norm=Norm)
                        self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                    if VVV == 3:
                        FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, x0, MulCo, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Norm=Norm)
                        self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS))
                    if VVV == 5:
                        FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=2, Norm=Norm)
                        self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                    if VVV == 6:
                        FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=3, Norm=Norm)
                        self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                subspc = FS
                self.Y_axis = FS
                distri_counter = 0
                if VVV != 4:
                    V = 0
                    v = np.array([len(Ps)] * len(Ps))
                    for i in range(0, len(Ps)):
                        for k in range(0, len(Ps)):
                            if min(FS[i]) < min(FS[k]):
                                v[i] -= 1
                    self.Z_order = v
                    self.Color_order = []
                    skip_step = 0
                    for i in range(0, len(Ps)):
                        plt.plot(A, FS[i], color = self.startlabel[i+1+distri_counter].color, zorder=v[i])
                        self.Color_order.append(self.startlabel[i+1+distri_counter].color)
                        # self.realtable[(i + 1 + V) * 2][0].color = col_tab[i]
                        # if len(Psm[i]) == 2:
                        #     self.realtable[(i + 1 + V + 1) * 2][0].color = col_tab[i]
                        #     V += 1
                        for k in range(0, len(Psm[i])):
                            self.realtable[(i + 1 + V + k) * 2][0].color = self.startlabel[i+1+distri_counter].color
                        V += len(Psm[i])-1

                        plt.fill_between(A, np.array(p[0] + p[3] * p[0]/10**2 * A + p[2] * p[0] / 10**4 * (A - p[1])**2 + p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float), FS[i].astype(float), facecolor=self.startlabel[i+1+distri_counter].color, zorder=v[i])
                        if len(self.FS_pos[i][0]) != 0:
                            minpos = self.FS_pos[i][0][0]
                            maxpos = self.FS_pos[i][0][0]
                            H_step = (max(B) - min(B)) * 0.04
                            for j in range(0, len(self.FS_pos[i][0])):
                                if self.FS_pos[i][0][j] < minpos:
                                    minpos = self.FS_pos[i][0][j]
                                if self.FS_pos[i][0][j] > maxpos:
                                    maxpos = self.FS_pos[i][0][j]
                            plt.plot([minpos, maxpos], [max(B) + H_step*(1+(i-skip_step)*2), max(B) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter].color, zorder=v[i])
                            for j in range(0, len(self.FS_pos[i][0])):
                                plt.plot([self.FS_pos[i][0][j], self.FS_pos[i][0][j]], [max(B) + H_step*((i-skip_step)*2), max(B) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter].color, zorder=v[i])
                        else:
                            skip_step += 1


                        distri_counter += Psm[i].count('Corr') + Psm[i].count('Distr')
                    self.Baseline_plot = np.array(p[0] + p[3] * p[0]/10**2 * A +  p[2] * p[0] / 10**4 * (A - p[1])**2 + p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float)
                    plt.plot(A, B - SPC_f + min(B) - max(B - SPC_f), color='lime')
                    plt.plot(A, B - B + min(B) - max(B - SPC_f), linestyle='--', color=self.gridcolor)
                else:
                    for i in range(0, len(Ps)):
                        self.realtable[(i + 1) * 2][0].color = 'pink'
                    A = A * 10 ** 9
                plt.xlim(min(A), max(A))
                plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
                plt.plot(A, SPC_f, color='r', zorder=len(Ps))
                plt.plot(A, B, linestyle = 'None', marker = 'x', color='m', zorder=len(Ps)+1)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.ylabel('Transmission, counts')
                plt.xlabel('Velocity, mm/s')

                plt.title('χ² = %.3f' % hi2, y=1, color='r')
                plt.text(0, -0.1, os.path.basename(self.path_list[l_p]), horizontalalignment='left', verticalalignment='center', color='m', transform = ax1.transAxes)
                if self.online_fit == True:
                    print('online_number picture', online_number)
                    plt.text(1, -0.1, str('№ ' + str(online_number)), horizontalalignment='right', verticalalignment='center', color='m', transform = ax1.transAxes)
                self.SPC_numb = os.path.basename(self.path_list[l_p])

                if VVV == 3 or VVV == 1 or VVV == 5 or VVV == 6:
                    JNold = np.copy(JN)
                    JN = max(JN*4, 32)
                    F2 = func(A, p)
                    plt.plot(A, F2 - SPC_f + min(B)-max(B - SPC_f)+min(B - SPC_f)-max(F2 - SPC_f), color='cyan')
                    self.Integral_line_plot = F2 - SPC_f + min(B)-max(B - SPC_f)+min(B - SPC_f)-max(F2 - SPC_f)
                    JN = JNold
                if VVV == 4:
                    plt.yscale("log")
                    plt.xlabel('Time, ns')
                    plt.ylim(max(0.99, min(B) * 0.9), max(B) * 1.1)

                if p[2] == 0 and p[6] == 0:
                    ymin, ymax = ax1.get_ylim()
                    print(ymin, ymax)
                    ax2 = ax1.twinx()
                    ax2.set_ylim((ymin / (p[0]+p[4]), ymax / (p[0]+p[4])))
            else:
                number_of_spectra = model.count('Nbaseline') + 1
                print('number_of_spectra ', number_of_spectra)

                step_sign = np.sign(A[1] - A[0])
                x_separate = []  # [[] * number_of_spectra]
                y_separate = []  # [[] * number_of_spectra]
                start = 0
                Num_x = 0
                for i in range(1, len(A)):
                    if step_sign != np.sign(A[i] - A[i - 1]):
                        # x_separate[Num_x] = A[start:i]
                        # y_separate[Num_x] = B[start:i]
                        x_separate.append(A[start:i])
                        y_separate.append(B[start:i])
                        start = i
                        Num_x += 1
                        # if Num_x == model.count('Nbaseline')+1:
                        #     break
                x_separate.append(A[start:])
                y_separate.append(B[start:])
                # print(len(x_separate))

                model_separate = []  # [[] * number_of_spectra]
                startM = 0
                Num_m = 0
                for i in range(0, len(model)):
                    if model[i] == 'Nbaseline':
                        # model_separate[Num_m] = model[startM:i]
                        model_separate.append(model[startM:i])
                        startM = i + 1
                        Num_m += 1
                        # if Num_m == model.count('Nbaseline')+1:
                        #     break
                model_separate.append(model[startM:])
                print(model_separate)

                begining_spc = [0]
                start_cont_par = NBA
                for i in range(1, len(self.realtableP) - 1):
                    if self.realtableP[i][0].text != 'None':
                        if self.nametable[i][0].text == 'Ns':
                            begining_spc.append(start_cont_par)
                        for j in range(0, len(self.nametable[i])):
                            if self.nametable[i][j].text != '':
                                start_cont_par += 1
                    else:
                        break
                # begining_spc.append(-1)
                # print(begining_spc)

                Distri_save = np.copy(Distri)
                Cor_save = np.copy(Cor)
                Distri, Cor = create_subspectra(self, model, Distri, Cor, p)[2:4]
                model_save = model.copy()
                V = 0
                self.Color_order = []
                v = np.array([float(0)])
                self.Z_order = []
                self.SPC_numb = []
                self.Y_axis = []
                for i in range(0, len(self.path_list)):
                    self.SPC_numb.append(os.path.basename(self.path_list[i]))
                self.Color_order = []
                self.Model_full_plot = []
                self.X_axis = x_separate
                self.SPC_plot = y_separate
                self.Baseline_plot = []
                self.Integral_line_plot = []

                fig, ax1 = plt.subplots(figsize=(2942 / 300 * number_of_spectra, 4.5), dpi=300)
                ax1.set_axis_off()
                # fig = plt.figure(figsize=(2942/300*number_of_spectra, 4.5), dpi=300)
                for NumSpc in range(0, number_of_spectra):
                    ax = plt.subplot(1, number_of_spectra, NumSpc + 1)

                    model = model_separate[NumSpc]
                    if NumSpc != number_of_spectra - 1:
                        SPC_f = func(x_separate[NumSpc], p[begining_spc[NumSpc]:begining_spc[NumSpc + 1]])
                        p_separate = p[begining_spc[NumSpc]:begining_spc[NumSpc + 1]]
                    else:
                        SPC_f = func(x_separate[NumSpc], p[begining_spc[NumSpc]:])
                        p_separate = p[begining_spc[NumSpc]:]
                    self.Model_full_plot.append(SPC_f)
                    Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(self, model_separate[NumSpc], Distri, Cor, p_separate)

                    # print(Distri)
                    # print(Distri_t)
                    # print(Cor_t)
                    CoEn = 0
                    DiEn = 0
                    V = 0
                    FS = np.array([[float(0)] * len(x_separate[NumSpc])] * (len(Ps)+(len(Ps)==0)))
                    self.FS_pos = []
                    # self.Y_axis.append([])
                    for i in range(0, len(Ps)):
                        CoSt = CoEn
                        DiSt = DiEn
                        CoEn += Psm[i].count('Corr')
                        DiEn += Psm[i].count('Distr')
                        if VVV == 1:
                            FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS,
                                          Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn],
                                          Met=1, Norm=Norm)
                            # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                            self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                        if VVV == 3:
                            FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, x0, MulCo, INS,
                                          Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn],
                                          Norm=Norm)
                            self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS))
                        if VVV == 5:
                            FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS,
                                          Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn],
                                          Met=2, Norm=Norm)
                            # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                            self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                        if VVV == 6:
                            FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS,
                                          Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn],
                                          Met=3, Norm=Norm)
                            # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                            self.FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                        # print('time', i, time.time() - start_time)
                    self.Y_axis.append(FS)
                    # for NumSpc2 in range(0, NumSpc):
                    #     subspc_tmp = np.array([[float(0)]*x_separate[NumSpc2]]*len(FS))

                    if NumSpc == 0:
                        subspc = FS
                        for NumSpc2 in range(1, number_of_spectra):
                            subspc = np.concatenate((subspc, np.array([[float(0)] * x_separate[NumSpc2]] * len(FS))), axis=1)
                    else:
                        subspc_tmp = np.array([[float(0)] * x_separate[0]] * len(FS))
                        for NumSpc2 in range(1, NumSpc):
                            subspc_tmp = np.concatenate((subspc_tmp, np.array([[float(0)] * x_separate[NumSpc2]] * len(FS))), axis=1)
                        subspc_tmp = np.concatenate((subspc_tmp, FS), axis=1)
                        for NumSpc2 in range(NumSpc + 1, number_of_spectra):
                            subspc_tmp = np.concatenate((subspc_tmp, np.array([[float(0)] * x_separate[NumSpc2]] * len(FS))), axis=1)
                        subspc = np.concatenate((subspc, subspc_tmp))

                    distri_counter = 0
                    if VVV != 4:
                        # v_add = max(v)
                        v = np.array([int(len(Ps))] * len(Ps))
                        for i in range(0, len(Ps)):
                            for k in range(0, len(Ps)):
                                if min(FS[i]) < min(FS[k]):
                                    v[i] -= 1
                        self.Z_order.append(v)
                        # print('Z_order', self.Z_order)
                        self.Color_order.append([])
                        skip_step = 0
                        for i in range(0, len(Ps)):
                            self.Color_order[-1].append(self.startlabel[i + 1 + distri_counter + Cur_model_len].color)
                            plt.plot(x_separate[NumSpc], FS[i], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                            plt.fill_between(x_separate[NumSpc], np.array(
                                p[begining_spc[NumSpc] + 0] \
                                + p[begining_spc[NumSpc] + 3] * p[begining_spc[NumSpc] + 0]/10**2 * x_separate[NumSpc]
                                + p[begining_spc[NumSpc] + 2] * p[begining_spc[NumSpc] + 0] / 10 ** 4 * (x_separate[NumSpc] - p[begining_spc[NumSpc] + 1]) ** 2\
                                + p[begining_spc[NumSpc] + 6] * p[begining_spc[NumSpc] + 4] / 10**4   * (x_separate[NumSpc] - p[begining_spc[NumSpc] + 5]) ** 2\
                                + p[begining_spc[NumSpc] + 4]\
                                + p[begining_spc[NumSpc] + 7] * p[begining_spc[NumSpc] + 4]/10**2 * x_separate[NumSpc], dtype=float),
                                             FS[i].astype(float), facecolor=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])

                            for k in range(0, len(Psm[i])):
                                self.realtable[(i + 1 + V + k + Cur_model_len) * 2][0].color = self.startlabel[i + 1 + distri_counter + Cur_model_len].color
                            V += len(Psm[i]) - 1

                            if len(self.FS_pos[i][0]) != 0:
                                minpos = self.FS_pos[i][0][0]
                                maxpos = self.FS_pos[i][0][0]
                                H_step = (max(y_separate[NumSpc]) - min(y_separate[NumSpc])) * 0.04
                                for j in range(0, len(self.FS_pos[i][0])):
                                    if self.FS_pos[i][0][j] < minpos:
                                        minpos = self.FS_pos[i][0][j]
                                    if self.FS_pos[i][0][j] > maxpos:
                                        maxpos = self.FS_pos[i][0][j]
                                plt.plot([minpos, maxpos], [max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2), max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                                for j in range(0, len(self.FS_pos[i][0])):
                                    plt.plot([self.FS_pos[i][0][j], self.FS_pos[i][0][j]], [max(y_separate[NumSpc]) + H_step*((i-skip_step)*2), max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                            else:
                                skip_step += 1

                            distri_counter += Psm[i].count('Corr') + Psm[i].count('Distr')
                        plt.plot(x_separate[NumSpc],
                                 y_separate[NumSpc] - SPC_f + min(y_separate[NumSpc]) - max(y_separate[NumSpc] - SPC_f),
                                 color='lime')
                        plt.plot(x_separate[NumSpc],
                                 y_separate[NumSpc] - y_separate[NumSpc] + min(y_separate[NumSpc]) - max(
                                     y_separate[NumSpc] - SPC_f), linestyle='--', color=self.gridcolor)
                    else:
                        x_separate[NumSpc] = x_separate[NumSpc] * 10 ** 9
                    plt.xlim(min(x_separate[NumSpc]), max(x_separate[NumSpc]))
                    plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
                    plt.plot(x_separate[NumSpc], SPC_f, color='r', zorder=len(Ps))
                    plt.plot(x_separate[NumSpc], y_separate[NumSpc], linestyle='None', marker='x', color='m', zorder=len(Ps) + 1)
                    plt.text(0, -0.1, os.path.basename(self.path_list[NumSpc]), horizontalalignment='left', verticalalignment='center', color='m', transform = ax.transAxes)

                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    plt.xlabel('Velocity, mm/s')
                    plt.ylabel(str('Transmission, counts')*(NumSpc==0))

                    if VVV == 3 or VVV == 1 or VVV == 5 or VVV == 6:
                        JNold = np.copy(JN)
                        JN = max(JN * 4, 32)
                        F2 = func(x_separate[NumSpc], p_separate)
                        plt.plot(x_separate[NumSpc],F2 - SPC_f + min(y_separate[NumSpc]) - max(y_separate[NumSpc] - SPC_f)\
                                 + min(y_separate[NumSpc] - SPC_f) - max(F2 - SPC_f), color='cyan')
                        self.Integral_line_plot.append(F2 - SPC_f + min(y_separate[NumSpc])-max(y_separate[NumSpc] - SPC_f)\
                                 +min(y_separate[NumSpc] - SPC_f)-max(F2 - SPC_f))
                        JN = JNold
                    if VVV == 4:
                        plt.yscale("log")
                        plt.xlabel('Time, ns')
                        plt.ylim(max(0.99, min(y_separate[NumSpc]) * 0.9), max(y_separate[NumSpc]) * 1.1)
                    Distri = Distri[DiEn:]
                    Cor = Cor[CoEn:]
                    Cur_model_len += len(model) + 1
                    if NumSpc == int(number_of_spectra/2):
                        plt.title('χ² = %.3f' % hi2, y=1, color='r')
                    if p[begining_spc[NumSpc]+2] == 0 and p[begining_spc[NumSpc]+5] == 0:
                        ymin, ymax = ax.get_ylim()
                        ax2 = ax.twinx()
                        ax2.set_ylim((ymin / (p[begining_spc[NumSpc]+0] + p[begining_spc[NumSpc]+3]), ymax / (p[begining_spc[NumSpc]+0] + p[begining_spc[NumSpc]+3])))

                model = model_save
                Distri = Distri_save
                Cor = Cor_save
                for i in range(0, len(subspc)):
                    if np.count_nonzero(subspc[len(subspc)-1-i]) == 0:
                        subspc = np.delete(subspc, len(subspc)-1-i, 0)
            # if platform.system() == 'Windows':
            #     realpath = str(self.dir_path) + str('\\\\result.jpg')
            # else:
            #     realpath = str(self.dir_path) + str('/result.jpg')
            # fig.savefig(realpath, bbox_inches='tight')
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\result.svg')
            else:
                realpath = str(self.dir_path) + str('/result.svg')
            fig.savefig(realpath, bbox_inches='tight')
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\result.png')
            else:
                realpath = str(self.dir_path) + str('/result.png')
            fig.savefig(realpath, bbox_inches='tight')
            imgpath2 = realpath
            # plt.close()
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(fig)
            gc.collect()

            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\result.txt')
            else:
                realpath = str(self.dir_path) + str('/result.txt')
            f = open(realpath, "w")
            f.write('#velocity\t' + 'counts\t' + 'full_model\t' + 'difference\t' + 'baseline\t')
            for i in range (0, len(subspc)):
                f.write('subspectrum%.i\t' % (i+1))
            f.write('\n')
            if model.count('Nbaseline') == 0:
                baseline = np.array(p[0] + p[3] * p[0]/10**2 * A + p[2] * p[0] / 10**4 * (A - p[1])**2 + p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float)
                #baseline = np.array(p[0] + p[2] * p[0] / 10**4 * (A - p[1])**2 + p[5] * (A - p[4]) ** 2 + p[3], dtype=float)
            else:
                baseline = []
                for NumSpc in range(0, number_of_spectra):
                    baseline_tmp = np.array(p[begining_spc[NumSpc] + 0] \
                                + p[begining_spc[NumSpc] + 3] * p[begining_spc[NumSpc] + 0]/10**2 * x_separate[NumSpc]
                                + p[begining_spc[NumSpc] + 2] * p[begining_spc[NumSpc] + 0] / 10 ** 4 * (x_separate[NumSpc] - p[begining_spc[NumSpc] + 1]) ** 2\
                                + p[begining_spc[NumSpc] + 6] * p[begining_spc[NumSpc] + 4] / 10**4 * (x_separate[NumSpc] - p[begining_spc[NumSpc] + 5]) ** 2\
                                + p[begining_spc[NumSpc] + 4]\
                                + p[begining_spc[NumSpc] + 7] * p[begining_spc[NumSpc] + 4]/10**2 * x_separate[NumSpc], dtype=float)
                    self.Baseline_plot.append(baseline_tmp)
                    baseline = np.concatenate((baseline, baseline_tmp))

            for i in range(0, len(A)):
                f.write(str('%.3f' %A[i]) + '\t' + str('%.3f' %B[i]) + '\t' + str('%.3f' %Total_SPC[i]) + '\t' + str('%.3f' %(B[i] - Total_SPC[i])) + '\t' + str('%.3f' %(baseline[i])) + '\t')
                for j in range(0, len(subspc)):
                    f.write(str('%.3f' %subspc[j][i]) + '\t')
                f.write('\n')
            f.close()

            if platform.system() == 'Windows':
                realpathD = str(self.dir_path) + str('\\\\resultD.png')
            else:
                realpathD = str(self.dir_path) + str('/resultD.png')
            try:
                os.remove(realpathD)
            except:
                pass

            model_np = np.array(model)
            # print('recheck model: ', model)
            # print(np.where(model_np == 'Distr')[0])
            if len(np.where(model_np == 'Distr')[0]) != 0:
                Distri_D = np.where(model_np == 'Distr')[0]
                Distri_D = np.append(Distri_D, len(model_np) * 2)
                Corr_D = np.where(model_np == 'Corr')[0]
                CL = 0
                for j in range(0, len(Distri_D) - 1):
                    CLtmp = 0
                    for k in range(0, len(Corr_D)):
                        if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j + 1]:
                            CLtmp += 1
                    if CLtmp > CL:
                        CL = CLtmp

                fig = plt.figure(figsize=(8 * max(CL, 1), 6 * len(Distri_D)), dpi=300)
                for j in range(0, len(Distri_D) - 1):
                    Vnum = NBA
                    for k in range(0, Distri_D[j]):
                        Vnum += mod_len_def(model[k])
                    sp = plt.subplot(len(Distri_D) * (CL + 1), (CL + 1), j * (CL + 1) + 1)
                    # sp.yaxis.set_visible(False)
                    # sp.yaxis.set_ticks([])
                    X = np.linspace(p[Vnum + 1], p[Vnum + 2], int(p[Vnum + 3]))
                    Y = eval(Distri[j]) + 0 * X
                    Xo = np.copy(X)
                    X = np.linspace(p[Vnum + 1], p[Vnum + 2], 1024)
                    Xoo = np.copy(X)
                    Y2 = eval(Distri[j]) + 0 * X
                    S = Y2.sum()
                    Y2 = Y2 / S
                    plt.plot(X, Y2, color='m')
                    Y = Y / S
                    plt.plot(Xo, Y, marker='h', linestyle='', color='r')
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    if j == int((len(Distri_D)-1) / 2):
                        plt.ylabel('Probability Density')

                    M = 1
                    MD = int(Distri_D[j])
                    for k in range(1, len(model)):
                        M = MD - k
                        if model[MD - k] != 'Distr' and model[MD - k] != 'Corr':
                            M += 1
                            break
                    plt.xlabel(self.nametable[M][int(float(self.realtableP[MD + 1][1].text))].text)

                    # if j == int(len(Distri_D) - 1):
                    #     plt.xlabel('Parameter')

                    Co_n = 0
                    for k in range(0, len(Corr_D)):
                        if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j + 1]:
                            sp = plt.subplot(len(Distri_D) * (CL + 1), (CL + 1), j * (CL + 1) + 2 + Co_n)
                            Co_n += 1
                            X = Xoo
                            YY2 = eval(Cor[k]) + 0 * X
                            plt.plot(YY2, Y2, color='orange')
                            X = Xo
                            YY = eval(Cor[k]) + 0 * X
                            plt.plot(YY, Y, marker='H', linestyle='', color='m')
                            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                            plt.xlabel(self.nametable[M][int(float(self.realtableP[MD + 1 + Co_n][1].text))].text)
                            # sp.yaxis.set_visible(False)
                            # sp.yaxis.set_ticks([])
                fig.savefig(realpathD, bbox_inches='tight')
                # plt.close()
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
                gc.collect()

            if (len(self.path_list) + self.online_fit*2)*(1-(model.count('Nbaseline')>0)) > 1:
                if os.path.exists(self.save_path.text) == False:
                    try:
                        f = open(self.save_path.text + '_param.txt', "a", encoding="utf-8")
                        f.close
                        print('file exist')
                    except FileNotFoundError:
                        v = 0
                        for i in range(0, len(str(self.save_path.text)) - 2):
                            if str(self.save_path.text)[-1-i] == '\\' or str(self.save_path.text)[-1-i] == '/':
                                save_dir_path = str(self.save_path.text)[:-1-i]
                                break
                        os.makedirs(save_dir_path)
                        print('file location created')
                if os.path.exists(self.save_path.text + '_SVG\\') == False and os.path.exists(self.save_path.text + '_SVG/') == False and self.online_fit == False:
                    if platform.system() == 'Windows':
                        os.mkdir(self.save_path.text + '_SVG\\')
                    else:
                        os.mkdir(self.save_path.text + '_SVG/')



                # data_uri = base64.b64encode(open(imgpath, 'rb').read()).decode('utf-8')
                # img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
                # # print(img_tag)
                #
                # htmlfile = open(self.save_path.text + '_result_PNG.html', "a")
                # # htmlfile.write("<html>\n")
                # htmlfile.write(img_tag)
                # htmlfile.write("</html>\n")
                # htmlfile.close()

                for k in range(0, len(str(self.path_list[l_p]))):
                    j = len(self.path_list[l_p]) - 1 - k
                    if str(self.path_list[l_p])[j] == '/' or str(self.path_list[l_p])[j] == '\\':
                        break
                    file_name = str(self.path_list[l_p])[j:]

                if self.online_fit == False:
                    if platform.system() == 'Windows':
                        imgpath = str(self.dir_path) + str('\\\\result.svg')
                        file = self.save_path.text + '_SVG\\' + file_name + '_SVG.svg'
                    else:
                        imgpath = str(self.dir_path) + str('/result.svg')
                        file = self.save_path.text + '_SVG/' + file_name + '_SVG.svg'
                    try:
                        shutil.copyfile(imgpath, file)
                    except shutil.SameFileError:
                        pass


                file = self.save_path.text + file_name + str(online_number)*self.online_fit + '_graf.txt'
                try:
                    shutil.copyfile(realpath, file)
                except shutil.SameFileError:
                    pass

                RR = True
                RA = True
                RA2.set()
                time.sleep(1)

                realpath_table = str(self.dir_path) + str('\\\\result_table.png') * (platform.system() == 'Windows') \
                                                    + str('/result_table.png') * (platform.system() != 'Windows')
                table_plot = []
                row_colors = []
                row_names = []
                for i in range(0, len(model) * 2 + 2):
                    table_plot.append([])
                    row_colors.append(self.realtable[i][0].color)
                    row_names.append(str('       '))
                    for j in range(0, numco + 1):
                        k = j
                        if (self.realtable[i][0].text == 'Expression' or self.realtable[(i-1)*(i!=0)][0].text == 'Expression') and j == numco:
                            k = 1
                            table_plot[i][1] = ''
                        if (self.realtable[i][0].text == 'Distr' or self.realtable[(i-1)*(i!=0)][0].text == 'Distr') and j == numco:
                            k = 5
                            table_plot[i][5] = ''
                        if (self.realtable[i][0].text == 'Corr' or self.realtable[(i-1)*(i!=0)][0].text == 'Corr') and j == numco:
                            k = 2
                            table_plot[i][2] = ''
                        if '[size=' in self.realtable[i][k].text:
                            new_text = str(self.realtable[i][k].text)
                            new_text = re.sub(r"\[.*?\]", "", new_text)
                            # print(new_text)
                            table_plot[i].append(new_text)
                        else:
                            table_plot[i].append(self.realtable[i][k].text)


                # fig = plt.figure(figsize=(3206/300*number_of_spectra, 2000*(len(model) * 2 + 2)/300), dpi=300)
                # table = plt.table(cellText=table_plot, rowLabels=row_names, rowColours=row_colors, cellLoc='right',
                #                   colLoc='center', rowLoc='center')
                # table.auto_set_font_size(False)
                # table.set_fontsize(4)
                #
                # plt.axis('off')
                # plt.grid('off')
                # plt.gcf().canvas.draw()
                # points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
                # points[0, :] -= 10
                # points[1, :] += 10
                # nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
                # fig.savefig(realpath_table, bbox_inches=nbbox)
                fig = plt.figure(figsize=(3206 / 300 * number_of_spectra, 7 * (len(model) * 2 + 2) / 300), dpi=300)
                table = plt.table(cellText=table_plot, rowLabels=row_names, rowColours=row_colors, cellLoc='right',
                                  colLoc='center', rowLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(4)
                plt.axis('off')
                plt.grid('off')
                fig.savefig(realpath_table, bbox_inches='tight')
                # plt.close()
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
                gc.collect()

                im1 = matplotlib.image.imread(os.path.abspath(imgpath2))
                im2 = matplotlib.image.imread(os.path.abspath(realpath_table))
                im1.astype(np.uint8)
                im2.astype(np.uint8)
                # print(len(im2), len(im2[0]))
                if len(im2[0]) > len(im1[0]): # np.zeros((len(im1), len(im2[0]) - len(im1[0]), 4), np.uint8)
                    im1 = np.concatenate((im1, np.array([[[0,0,0,1]]*(len(im2[0]) - len(im1[0]))]*len(im1), np.uint8)), axis=1)
                if len(im2[0]) < len(im1[0]): # np.zeros((len(im2), len(im1[0]) - len(im2[0]), 4), np.uint8)
                    im2 = np.concatenate((im2, np.array([[[0,0,0,1]]*(len(im1[0]) - len(im2[0]))]*len(im2), np.uint8)), axis=1)
                combo_image = np.concatenate((im1, im2), axis=0)
                realpath_combo = str(self.dir_path) + str('\\\\result_combo.png') * (platform.system() == 'Windows') \
                                                    + str('/result_combo.png') * (platform.system() != 'Windows')
                matplotlib.image.imsave(os.path.abspath(realpath_combo), combo_image)


                data_uri = base64.b64encode(open(realpath_combo, 'rb').read()).decode('utf-8')
                img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
                htmlfile = open(self.save_path.text + '_result_table_PNG.html', "a")
                htmlfile.write(img_tag)
                htmlfile.write("</html>\n")
                htmlfile.close()



                file = self.save_path.text + '_param.txt'

                try:
                    with open(file, encoding="utf-8") as f:
                        first_line = f.readline().rstrip()
                    first_line = re.split(r'\t+', first_line)
                    # print(first_line)
                except:
                    first_line = []

                f = open(file, "a", encoding="utf-8")  # write to file output values

                name2 = []
                model_values = []
                v = 0
                vv = 0
                for i in range(0, len(self.realtable)):
                    for j in range(0, len(self.realtable[i])):
                        if len(str(self.realtable[i][j].text)) != 0:
                            if str(self.realtable[i][j].text)[0] == '±':
                                model_values.append(str(self.realtable[i][j].text)[2:])
                                name2.append('Δ' + str(name2[v]))
                                v += 1
                                vv += 1
                            else:
                                tmp = str(self.realtable[i][j].text).split(' ')
                                if self.realtable[i][0].text == 'Distr' and j == 5:
                                    model_values.append(self.realtable[i][j].text)
                                    name2.append('PDF')
                                    v += 1
                                elif self.realtable[i][0].text == 'Corr' and j == 2:
                                    model_values.append(self.realtable[i][j].text)
                                    name2.append('CoF')
                                    v += 1
                                elif self.realtable[i][0].text == 'Expression' and j == 1:
                                    model_values.append(tmp[-1])
                                    name2.append(str(' '.join(tmp[0:-2])))
                                elif len(tmp) == 3:
                                    model_values.append(str(tmp[2]))
                                    name2.append(str(tmp[0]))
                                elif len(tmp) == 1:
                                    model_values.append(str(self.realtable[i][j].text))
                                    name2.append(str('subspc'))
                                    v += 1
                                    tmp_next = str(self.realtable[i + 1][j].text).split(' ')
                                    if len(tmp_next) == 4:
                                        model_values.append(str(tmp_next[0]))
                                        name2.append(str('I%'))
                                        model_values.append(str(tmp_next[2]))
                                        name2.append(str('ΔI%'))
                                        v += 2
                                    elif tmp_next[0] == 'Be' or tmp_next[0] == 'KB':
                                        model_values.append(str(tmp_next[0]))
                                        name2.append(str('I%'))
                                        v += 1
                                # else:
                                #     model_values.append(str(self.realtable[i][j].text))
                                #     name2.append(str('subspc'))
                                #     v += 1


                    v += vv
                    vv = 0

                name2[1] = u'N\u2092'
                name2[2] = u'X\u2092'
                name2[3] = 'coef²'
                name2[4] = u'Nnr'
                name2[5] = u'Xnr'
                name2[6] = 'coef²nr'
                name2[7] = u'ΔN\u2092'
                name2[8] = u'ΔX\u2092'
                name2[9] = 'Δcoef²'
                name2[7] = u'ΔNnr'
                name2[8] = u'ΔXnr'
                name2[9] = 'Δcoef²nr'
                name2.append(str('χ²'))
                name2.insert(0, str('#File'))
                if l_p == 0 and first_line != name2:
                    for i in range(0, len(name2)):
                        f.write(name2[i] + '\t')
                    f.write('\n')
                f.write(os.path.basename(self.path_list[l_p])+str('\t'))
                for i in range(0, len(name2) - 2):
                    f.write(model_values[i] + '\t')
                f.write(str(hi2))
                f.write('\n')
                f.close()



            if SeF == 1 or p[0]+p[4] < BGmin or hi2 > chi2max:
                if p[0]+p[4] < BGmin:
                    print ('bacground ', p[0]+p[4], ' is less then ', BGmin)
                if hi2 > chi2max:
                    print ('hi2 ', hi2, ' is bigger then ', chi2max)
                if SeF != 1:
                    print('parameters from previous step were !NOT! accepted')
            if SeF == 2 and p[0]+p[4] >= BGmin and hi2 <= chi2max:
                p0 = np.copy(p)
                print('parameters from previous step were accepted')

            online_end_time = time.time() - online_start_time
            online_number += 1
            # print('online_number adding', online_number)
            if RL==3:
                break
            if online_end_time < self.time_interval * self.online_fit and RL==1:
                RL = 5
                time.sleep(self.time_interval - online_end_time)
                if RL==5:
                    RL = 1

        if RL != 3:
            RR = True
            RA = True
            RA2.set()
            # RR = True
            # RA = True
            if RL == 1:
                RL = 2
            time.sleep(1)
            realpath_table = str(self.dir_path) + str('\\\\result_table.png') * (platform.system() == 'Windows') \
                             + str('/result_table.png') * (platform.system() != 'Windows')
            table_plot = []
            row_colors = []
            row_names = []
            # colWidths = []
            for i in range(0, len(model) * 2 + 2):
                table_plot.append([])
                row_colors.append(self.realtable[i][0].color)
                row_names.append(str('       '))
                for j in range(0, numco+1):
                    k = j
                    if (self.realtable[i][0].text == 'Expression' or self.realtable[(i-1)*(i!=0)][0].text == 'Expression') and j == numco:
                        k = 1
                        table_plot[i][1] = ''
                    if (self.realtable[i][0].text == 'Distr' or self.realtable[(i-1)*(i!=0)][0].text == 'Distr') and j == numco:
                            k = 5
                            table_plot[i][5] = ''
                    if (self.realtable[i][0].text == 'Corr' or self.realtable[(i-1)*(i!=0)][0].text == 'Corr') and j == numco:
                            k = 2
                            table_plot[i][2] = ''
                    if '[size=' in self.realtable[i][k].text:
                        new_text = str(self.realtable[i][k].text)
                        # print(new_text)
                        new_text = re.sub(r"\[.*?\]", "", new_text)
                        # print(new_text)
                        table_plot[i].append(new_text)
                    else:
                        table_plot[i].append(self.realtable[i][k].text)

                    # if i == 0:
                    #     colWidths.append(float(2560/(numco+1)/300)) 3206
            # fig = plt.figure(figsize=(3206/300*number_of_spectra, 2000*(len(model) * 2 + 2)/300), dpi=300) #to hvae 2560pixels X variable
            fig = plt.figure(figsize=(3206/300*number_of_spectra, 7 * (len(model) * 2 + 2) / 300), dpi=300)
            table = plt.table(cellText=table_plot, rowLabels=row_names, rowColours=row_colors, cellLoc='right',
                              colLoc='center', rowLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(4)
            plt.axis('off')
            plt.grid('off')
            # plt.gcf().canvas.draw()
            # points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
            # points[0, :] -= 10
            # points[1, :] += 10
            # nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
            #fig.savefig(realpath_table, bbox_inches=nbbox)
            fig.savefig(realpath_table, bbox_inches='tight')
            #plt.close()
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close(fig)
            gc.collect()

            # global table_save
            # table_save = realpath_table



    self.SP_DI.text = 'Distribution'

    self.save_path.readonly = False
    self.play_btn.disabled = False
    self.INS_btn.disabled = False
    self.INS_btn2.disabled = False
    self.show_btn.disabled = False
    self.showM_btn.disabled = False
    self.cal_btn.disabled = False
    # self.switch.disabled = False

# def work3(self, _aa):
#     global RA, RL, TR
#     RL = 1
#     TR = threading.Thread(target=partial(work4, self))
#     TR.name = 'TTT'
#     TR.daemon = True
#     TR.start()

def work4(self):

    global x0, MulCo, RA, RL, RR
    # self.play_btn.background_color = [0, 0.5, 0, 1]
    # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
    # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
    # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
    # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
    # self.play_btn.disabled = True
    # self.INS_btn.disabled = True
    # self.INS_btn2.disabled = True
    # self.show_btn.disabled = True
    # self.showM_btn.disabled = True
    # self.cal_btn.disabled = True
    # self.switch.disabled = True
    RL = 1

    v = 0
    vv = 0
    raw_path = self.process_path.text[2:-2]
    self.path_list = []
    for i in range(0, len(raw_path) - 2):
        if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
            vv = i
            self.path_list.append(str(raw_path)[v:vv])
            v = i + 4
    self.path_list.append(str(raw_path)[v:])



    global val
    global model
    global p

    con1, con2, con3, Distri, Cor, Expr, NExpr, fix_tmp, DistriN = read_model(self)



    for i in range(0, len(NExpr)):
        p[NExpr[i]] = eval(Expr[i])
        # print(p[NExpr[i]])
    for i in range(0, len(con1)):
        p[int(con1[i])] = p[int(con2[i])] * con3[i]
    # for i in range(0, len(NExpr)):
    #     p[NExpr[i]] = eval(Expr[i])
    # for i in range(0, len(con1)):
    #     p[int(con1[i])] = p[int(con2[i])] * con3[i]

    confu = np.array([con1, con2, con3])

    file = os.path.abspath(self.path_list[0])
    if (raw_path[-1] == '\\' or raw_path[-1] == '/') and file[-1] != '.' and file[-1] != '/' and file[-1] != '\\':
        file += str('\\') * (platform.system() == 'Windows') + str('/') * (platform.system() != 'Windows')
    A, B = read_spectrum(self, file)
    for i in range(1, model.count('Nbaseline')+1):
        try:
            file = os.path.abspath(self.path_list[i])
            A1, B1 = read_spectrum(self, file)
            A = np.concatenate((A, A1))
            B = np.concatenate((B, B1))
        except:
            pass

    global p0, er

    VVV = 0
    Es = 1
    JN = int(self.JN0.text)

    if self.switch.active == True:
        if self.fitway[0].active == True:
            VVV = 1
            # EEs = 0.999 * 2 / (int(self.JN0.text) - 1)
            # cof = np.array([0.00254718, -0.00993933, 0.64000022, 0.10557177, 0.20270726])
            # Es = cof[0] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** (1 / 4) \
            #      + cof[1] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** (1 / 2) \
            #      + cof[2] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 1 \
            #      + cof[3] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 2 \
            #      + cof[4] * np.sign(EEs) * (np.tan(np.abs(np.pi / 2 * EEs))) ** 3
            # Es = Es / EEs  # correction for integral in zero point if odd points entered - pass instead x0 because x0 is not used in this method
            INS = float(self.L0.text)
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoCMS, INS, [0], [0], Met=1)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, Distri, Cor, Met=1, Norm=Norm)
                # return m5.PV(x, p, model, pool)
        if self.fitway[1].active == True:
            VVV = 3
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\INSexp.txt')
            else:
                realpath = str(self.dir_path) + str('/INSexp.txt')
            INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, x0, MulCo, INS, Distri, Cor, Norm=Norm)

        if self.fitway[1].active == True and str(self.L0.text)[:2] == 'L=':  # lorenz squared for Ilya Sergeev
            VVV = 6
            Li_Lo = 1
            INS = float(str(self.L0.text)[2:])
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoL2, INS, [0], [0], Met=3)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoL2, INS, Distri, Cor, Met=3, Norm=Norm)

        if self.fitway[2].active == True:
            VVV = 5
            if platform.system() == 'Windows':
                realpath = str(self.dir_path) + str('\\\\INS_APS.txt')
            else:
                realpath = str(self.dir_path) + str('/INS_APS.txt')
            INS = np.genfromtxt(realpath, delimiter='\t', skip_footer=0)
            print(INS)
            pNorm = np.array([float(0)] * NBA)
            pNorm[0] = 1
            Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoCMS, INS, [0], [0], Met=2)[0]
            print('Normalization integral equal to', Norm)
            def func(x, p):
                return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, Distri, Cor, Met=2, Norm=Norm)

    if self.switch.active == False:
        VVV = 4
        # if A[0] > A[-1]:
        #     A = A[::-1]
        #     B = B[::-1]
        A = A*10**-9
        def func(x, p):
            return mN.TI(x, p, model, pool)


    # p0 = np.copy(p)


    start_time = time.time()
    print(model)
    print(p)
    SPC_f = func(A, p)
    # print(SPC_f)
    print('spectrum time', time.time() - start_time)

    # print('spc')
    # col_tab = ['blue', 'red', 'yellow', 'cyan', 'pink', 'lime', 'darkorange', 'green', 'crimson', 'blueviolet']
    # col_tab.extend(col_tab)
    # col_tab.extend(col_tab)
    # col_tab.extend(col_tab) #now it is 80

    Cur_model_len = 0

    if model.count('Nbaseline') == 0:
        number_of_spectra = 1
        fig, ax1 = plt.subplots(figsize=(2942/300*number_of_spectra, 4.5), dpi=300)
        Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(self, model, Distri, Cor, p)
        print(Distri)
        print(Distri_t)
        print(Cor_t)
        CoEn = 0
        DiEn = 0
        FS = np.array([[float(0)]*len(A)]*len(Ps))
        FS_pos = []
        for i in range(0, len(Ps)):
            # print('subspc', i)
            CoSt = CoEn
            DiSt = DiEn
            CoEn += Psm[i].count('Corr')
            DiEn += Psm[i].count('Distr')
            # print(CoSt, CoEn)
            # print(DiSt, DiEn)
            # print(Psm[i], Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn])
            start_time = time.time()
            # print(Ps[i])
            # print(Psm[i])
            if VVV == 1:
                FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=1, Norm=Norm)
                FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                # FS[i] = m5.PV(A, Ps[i], Psm[i], pool)
            if VVV == 3:
                FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, x0, MulCo, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Norm=Norm)
                FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS))
            if VVV == 5:
                FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=2, Norm=Norm)
                FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
            if VVV == 6:
                FS[i] = m5.TI(A, Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=3, Norm=Norm)
                FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
            # print('time', i, time.time() - start_time)

        distri_counter = 0
        if VVV != 4:
            v = np.array([len(Ps)]*len(Ps))
            for i in range (0, len(Ps)):
                for k in range(0, len(Ps)):
                    if min(FS[i]) < min(FS[k]):
                        v[i] -= 1
            skip_step = 0
            for i in range(0, len(Ps)):
                plt.plot(A, FS[i], color=self.startlabel[i+1+distri_counter].color, zorder=v[i])
                plt.fill_between(A, np.array(p[0] + p[3] * p[0]/10**2 * A + p[2] * p[0] / 10**4 * (A - p[1])**2 + p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float), FS[i].astype(float), facecolor=self.startlabel[i+1+distri_counter].color, zorder=v[i])

                if len(FS_pos[i][0]) != 0:
                    minpos = FS_pos[i][0][0]
                    maxpos = FS_pos[i][0][0]
                    H_step = (max(B) - min(B)) * 0.04
                    for j in range(0, len(FS_pos[i][0])):
                        if FS_pos[i][0][j] < minpos:
                            minpos = FS_pos[i][0][j]
                        if FS_pos[i][0][j] > maxpos:
                            maxpos = FS_pos[i][0][j]
                    plt.plot([minpos, maxpos], [max(B) + H_step*(1+(i-skip_step)*2), max(B) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter].color, zorder=v[i])
                    for j in range(0, len(FS_pos[i][0])):
                        plt.plot([FS_pos[i][0][j], FS_pos[i][0][j]], [max(B) + H_step*((i-skip_step)*2), max(B) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter].color, zorder=v[i])
                else:
                    skip_step += 1


                distri_counter += Psm[i].count('Corr') + Psm[i].count('Distr')
            plt.plot(A, B - SPC_f + min(B) - max(B - SPC_f), color='lime')
            plt.plot(A, B - B + min(B) - max(B - SPC_f), linestyle='--', color=self.gridcolor)
        else:
            A = A * 10 ** 9
        plt.xlim(min(A), max(A))
        plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
        plt.plot(A, SPC_f, color='r', zorder=len(Ps)+2)
        plt.plot(A, B, linestyle='None', marker='x', color='m', zorder=len(Ps)+1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('Velocity, mm/s')
        plt.ylabel('Transmission, counts')

        if VVV == 3 or VVV == 1 or VVV == 5 or VVV == 6:
            JNold = np.copy(JN)
            JN = max(JN * 4, 32)
            F2 = func(A, p)
            plt.plot(A, F2 - SPC_f + min(B)-max(B - SPC_f)+min(B - SPC_f)-max(F2 - SPC_f), color='cyan')
            JN = JNold
        if VVV == 4:
            plt.yscale("log")
            plt.xlabel('Time, ns')
            plt.ylim(max(0.99, min(B)*0.9), max(B)*1.1)
        if p[2] == 0 and p[6] == 0:
            ymin, ymax = ax1.get_ylim()
            ax2 = ax1.twinx()
            ax2.set_ylim((ymin/(p[0]+p[4]), ymax/(p[0]+p[4])))

    else:
        number_of_spectra = model.count('Nbaseline') + 1
        print('number_of_spectra ', number_of_spectra)

        step_sign = np.sign(A[1] - A[0])
        x_separate = [] #[[] * number_of_spectra]
        y_separate = [] #[[] * number_of_spectra]
        start = 0
        Num_x = 0
        for i in range(1, len(A)):
            if step_sign != np.sign(A[i] - A[i - 1]):
                # x_separate[Num_x] = A[start:i]
                # y_separate[Num_x] = B[start:i]
                x_separate.append(A[start:i])
                y_separate.append(B[start:i])
                start = i
                Num_x += 1
                # if Num_x == model.count('Nbaseline')+1:
                #     break
        x_separate.append(A[start:])
        y_separate.append(B[start:])
        print(len(x_separate))

        model_separate = [] #[[] * number_of_spectra]
        startM = 0
        Num_m = 0
        for i in range(0, len(model)):
            if model[i] == 'Nbaseline':
                # model_separate[Num_m] = model[startM:i]
                model_separate.append(model[startM:i])
                startM = i + 1
                Num_m += 1
                # if Num_m == model.count('Nbaseline')+1:
                #     break
        model_separate.append(model[startM:])
        print(model_separate)

        begining_spc = [0]
        start_cont_par = NBA
        for i in range(1, len(self.realtableP)-1):
            if self.realtableP[i][0].text != 'None':
                if self.nametable[i][0].text == 'Ns':
                    begining_spc.append(start_cont_par)
                for j in range(0, len(self.nametable[i])):
                    if self.nametable[i][j].text != '':
                        start_cont_par += 1
            else:
                break
        # begining_spc.append(-1)
        print(begining_spc)

        Distri_save = np.copy(Distri)
        Cor_save = np.copy(Cor)
        Distri, Cor = create_subspectra(self, model, Distri, Cor, p)[2:4]
        print(Distri)
        print(Cor)
        model_save = model.copy()

        # Distri = Distri_t
        # Cor = Cor_t

        fig, ax1 = plt.subplots(figsize=(2942 / 300 * number_of_spectra, 4.5), dpi=300)
        ax1.set_axis_off()
        # fig = plt.figure(figsize=(2942/300*number_of_spectra, 4.5), dpi=300)
        for NumSpc in range(0, number_of_spectra):
            ax = plt.subplot(1, number_of_spectra, NumSpc+1)

            model = model_separate[NumSpc]
            if NumSpc != number_of_spectra-1:
                SPC_f = func(x_separate[NumSpc], p[begining_spc[NumSpc]:begining_spc[NumSpc+1]])
                p_separate = p[begining_spc[NumSpc]:begining_spc[NumSpc+1]]
            else:
                SPC_f = func(x_separate[NumSpc], p[begining_spc[NumSpc]:])
                p_separate = p[begining_spc[NumSpc]:]

            Ps, Psm, Distri_t, Cor_t, Di, Co = create_subspectra(self, model_separate[NumSpc], Distri, Cor, p_separate)
            print(Distri)
            print(Distri_t)
            print(Cor_t)
            CoEn = 0
            DiEn = 0
            FS = np.array([[float(0)] * len(x_separate[NumSpc])] * len(Ps))
            FS_pos = []
            for i in range(0, len(Ps)):
                # print('subspc', i)
                CoSt = CoEn
                DiSt = DiEn
                CoEn += Psm[i].count('Corr')
                DiEn += Psm[i].count('Distr')
                # print(CoSt, CoEn)
                # print(DiSt, DiEn)
                # print(Psm[i], Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn])
                # print(Ps[i])
                # print(Psm[i])
                if VVV == 1:
                    FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=1, Norm=Norm)
                    # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                    FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                if VVV == 3:
                    FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, x0, MulCo, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Norm=Norm)
                    FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS))
                if VVV == 5:
                    FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=2, Norm=Norm)
                    # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                    FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                if VVV == 6:
                    FS[i] = m5.TI(x_separate[NumSpc], Ps[i], Psm[i], JN, pool, 0.0, MulCoCMS, INS, Distri_t[DiSt:DiEn], Cor_t[CoSt:CoEn], Met=3, Norm=Norm)
                    # FS[i] = m5.PV(x_separate[NumSpc], Ps[i], Psm[i], pool)
                    FS_pos.append(modpos.mod_pos(Ps[i], Psm[i], INS, Met=1))
                # print('time', i, time.time() - start_time)


            distri_counter = 0
            if VVV != 4:
                v = np.array([len(Ps)] * len(Ps))
                for i in range(0, len(Ps)):
                    for k in range(0, len(Ps)):
                        if min(FS[i]) < min(FS[k]):
                            v[i] -= 1
                skip_step = 0
                for i in range(0, len(Ps)):
                    plt.plot(x_separate[NumSpc], FS[i], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                    plt.fill_between(x_separate[NumSpc], np.array(
                        p[begining_spc[NumSpc]+0] \
                        + p[begining_spc[NumSpc]+3] * p[begining_spc[NumSpc]+0]/10**2 * x_separate[NumSpc] \
                        + p[begining_spc[NumSpc]+2] * p[begining_spc[NumSpc]+0] / 10 ** 4 * (x_separate[NumSpc] - p[begining_spc[NumSpc]+1]) ** 2\
                        + p[begining_spc[NumSpc]+6] * p[begining_spc[NumSpc]+4] / 10**4   * (x_separate[NumSpc] - p[begining_spc[NumSpc]+5]) ** 2\
                        + p[begining_spc[NumSpc]+4]\
                        + p[begining_spc[NumSpc]+7] * p[begining_spc[NumSpc]+4]/10**2 * x_separate[NumSpc], dtype=float),
                                     FS[i].astype(float), facecolor=self.startlabel[i + 1 + distri_counter + Cur_model_len].color,
                                     zorder=v[i])

                    if len(FS_pos[i][0]) != 0:
                        minpos = FS_pos[i][0][0]
                        maxpos = FS_pos[i][0][0]
                        H_step = (max(y_separate[NumSpc]) - min(y_separate[NumSpc])) * 0.04
                        for j in range(0, len(FS_pos[i][0])):
                            if FS_pos[i][0][j] < minpos:
                                minpos = FS_pos[i][0][j]
                            if FS_pos[i][0][j] > maxpos:
                                maxpos = FS_pos[i][0][j]
                        plt.plot([minpos, maxpos], [max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2), max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                        for j in range(0, len(FS_pos[i][0])):
                            plt.plot([FS_pos[i][0][j], FS_pos[i][0][j]], [max(y_separate[NumSpc]) + H_step*((i-skip_step)*2), max(y_separate[NumSpc]) + H_step*(1+(i-skip_step)*2)], color=self.startlabel[i + 1 + distri_counter + Cur_model_len].color, zorder=v[i])
                    else:
                        skip_step += 1

                    distri_counter += Psm[i].count('Corr') + Psm[i].count('Distr')
                plt.plot(x_separate[NumSpc], y_separate[NumSpc] - SPC_f + min(y_separate[NumSpc]) - max(y_separate[NumSpc] - SPC_f), color='lime')
                plt.plot(x_separate[NumSpc], y_separate[NumSpc] - y_separate[NumSpc] + min(y_separate[NumSpc]) - max(y_separate[NumSpc] - SPC_f), linestyle='--', color=self.gridcolor)

            else:
                x_separate[NumSpc] = x_separate[NumSpc] * 10 ** 9
            plt.xlim(min(x_separate[NumSpc]), max(x_separate[NumSpc]))
            plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
            plt.plot(x_separate[NumSpc], SPC_f, color='r', zorder=len(Ps) + 2)
            plt.plot(x_separate[NumSpc], y_separate[NumSpc], linestyle='None', marker='x', color='m', zorder=len(Ps) + 1)
            plt.text(0, -0.1, os.path.basename(self.path_list[NumSpc]), horizontalalignment='left', verticalalignment='center', color='m', transform = ax.transAxes)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xlabel('Velocity, mm/s')
            plt.ylabel(str('Transmission, counts')*(NumSpc==0))

            if VVV == 3 or VVV == 1 or VVV == 5 or VVV == 6:
                JNold = np.copy(JN)
                JN = max(JN * 4, 32)
                F2 = func(x_separate[NumSpc], p_separate)
                plt.plot(x_separate[NumSpc], F2 - SPC_f + min(y_separate[NumSpc]) - max(y_separate[NumSpc] - SPC_f) + min(y_separate[NumSpc] - SPC_f) - max(F2 - SPC_f), color='cyan')
                JN = JNold
            if VVV == 4:
                plt.yscale("log")
                plt.xlabel('Time, ns')
                plt.ylim(max(0.99, min(y_separate[NumSpc]) * 0.9), max(y_separate[NumSpc]) * 1.1)

            if p[begining_spc[NumSpc]+2] == 0 and p[begining_spc[NumSpc]+5] == 0:
                ymin, ymax = ax.get_ylim()
                ax2 = ax.twinx()
                ax2.set_ylim((ymin / (p[begining_spc[NumSpc]+0] + p[begining_spc[NumSpc]+3]), ymax / (p[begining_spc[NumSpc]+0] + p[begining_spc[NumSpc]+3])))

            Distri = Distri[DiEn:]
            Cor = Cor[CoEn:]
            Cur_model_len += len(model) + 1
        model = model_save
        Distri = Distri_save
        Cor = Cor_save

    for pos_n in range (0, len(FS_pos)):
        print('model number', pos_n+1, 'has lines positions:', FS_pos[pos_n])

    if platform.system() == 'Windows':
        realpath = str(self.dir_path) + str('\\\\result.svg')
    else:
        realpath = str(self.dir_path) + str('/result.svg')
    fig.savefig(realpath, bbox_inches='tight')
    if platform.system() == 'Windows':
        realpath = str(self.dir_path) + str('\\\\result.png')
    else:
        realpath = str(self.dir_path) + str('/result.png')
    fig.savefig(realpath, bbox_inches='tight')
    # plt.show()
    # plt.close()
    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close(fig)
    gc.collect()
    self.SP_DI.text = 'Distribution'

    if platform.system() == 'Windows':
        realpathD = str(self.dir_path) + str('\\\\resultD.png')
    else:
        realpathD = str(self.dir_path) + str('/resultD.png')
    try:
        os.remove(realpathD)
    except:
        pass

    model_np = np.array(model)
    # print('recheck model: ', model)
    # print(np.where(model_np == 'Distr')[0])

    # M = 1
    # for k in range(1, i):
    #     M = i - k
    #     if self.realtableP[i - k][0].text != 'Distr' and self.realtableP[i - k][0].text != 'Corr':
    #         M += 1
    #         break
    # for k in 1, 4:
    #     if k == 1:
    #             self.realtableP[M - 1][int(self.realtableP[i][k].text) + 1].background_color = [1, 1, 1, 0.5]

    if len(np.where(model_np == 'Distr')[0]) != 0:
        Distri_D = np.where(model_np == 'Distr')[0]
        Distri_D = np.append(Distri_D, len(model_np)*2)
        Corr_D = np.where(model_np == 'Corr')[0]
        CL = 0
        for j in range(0, len(Distri_D)-1):
            CLtmp = 0
            for k in range(0, len(Corr_D)):
                    if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j + 1]:
                        CLtmp += 1
            if CLtmp > CL:
                CL = CLtmp
        fig = plt.figure(figsize=(8 * max(CL, 1), 6.15 * len(Distri_D)), dpi=300)
        for j in range(0, len(Distri_D)-1):
            Vnum = NBA
            for k in range(0, Distri_D[j]):
                Vnum += mod_len_def(model[k])
            sp = plt.subplot(len(Distri_D) * (CL+1), (CL+1), j * (CL+1) + 1)
            # sp.yaxis.set_visible(False)
            # sp.yaxis.set_ticks([])
            X = np.linspace(p[Vnum + 1], p[Vnum + 2], int(p[Vnum + 3]))
            Y = eval(Distri[j]) + 0*X
            Xo = np.copy(X)
            X = np.linspace(p[Vnum + 1], p[Vnum + 2], 1024)
            Xoo = np.copy(X)
            Y2 = eval(Distri[j]) + 0*X
            S = Y.sum()
            Y2 = Y2 / S
            plt.plot(X, Y2, color='m')
            Y = Y / S
            plt.plot(Xo, Y, marker='h', linestyle='', color='r')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            if j == int((len(Distri_D)-1) / 2):
                plt.ylabel('Probability Density')

            M = 1
            MD = int(Distri_D[j])
            for k in range(1, len(model)):
                M = MD - k
                if model[MD - k] != 'Distr' and model[MD - k] != 'Corr':
                    M += 1
                    break
            # print('distri ', self.nametable[M + 1][int(float(self.realtableP[MD+1][1].text))].text)
            # if j == int(len(Distri_D) - 1):
            #     plt.xlabel('Parameter')
            plt.xlabel(self.nametable[M][int(float(self.realtableP[MD+1][1].text))].text)
            Co_n = 0
            for k in range(0, len(Corr_D)):
                if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j+1]:
                    sp = plt.subplot(len(Distri_D) * (CL+1), (CL+1), j * (CL+1) + 2 + Co_n)
                    Co_n += 1
                    X = Xoo
                    YY2 = eval(Cor[k]) + 0*X
                    plt.plot(YY2, Y2, color='orange')
                    X = Xo
                    YY = eval(Cor[k]) + 0*X
                    plt.plot(YY, Y, marker='H', linestyle='', color='m')
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    # print('corr ', self.nametable[M + 1][int(float(self.realtableP[MD + 1 + k + 1][1].text))].text)
                    plt.xlabel(self.nametable[M][int(float(self.realtableP[MD + 1 + Co_n][1].text))].text)
                    # sp.yaxis.set_visible(False)
                    # sp.yaxis.set_ticks([])
        fig.savefig(realpathD, bbox_inches='tight')
        # plt.close()
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        gc.collect()
        # print('DISTRI WAS SAVED!!!')

    RA = True
    RL = 2
    RA2.set()

    # self.play_btn.disabled = False
    # self.INS_btn.disabled = False
    # self.INS_btn2.disabled = False
    # self.show_btn.disabled = False
    # self.showM_btn.disabled = False
    # self.cal_btn.disabled = False
    # self.switch.disabled = False

# def dummy_ins(self, ref, _aa):
#     global RA, RL, TR
#     RL = 1
#     TR = threading.Thread(target=partial(instrumental, self, ref))
#     TR.name = 'TTT'
#     TR.daemon = True
#     TR.start()
#
# def instrumental_old(self, ref, mode=0):
#     global RA, RL, RR, x0, MulCo, PPP
#     RL = 1
#     # self.play_btn.background_color = [0, 0.5, 0, 1]
#     # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
#     # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
#     # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
#     # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
#     # self.play_btn.disabled = True
#     # self.INS_btn.disabled = True
#     # self.INS_btn2.disabled = True
#     # self.show_btn.disabled = True
#     # self.showM_btn.disabled = True
#     # self.cal_btn.disabled = True
#
#
#     # file = os.path.relpath(self.process_path.text[2:-2])
#     file = os.path.abspath(self.path_list[0])
#
#     A, B = read_spectrum(self, file)
#
#     if ref == 0:
#         x0 = -0.01
#         MulCo = 2.2
#
#         n = int(self.INS_number.text)
#         p0 = np.array([float(0.001)] * n * 3)
#         if n > 3:
#             for i in range(0, int(n // 2)):
#                 p0[i * 3] = 0.3
#                 p0[i * 3 + 1] = -1.5 + 3 * i / max(int(n // 2 - 1), 1)
#                 p0[i * 3 + 2] = 1 / n
#             for i in range(int(n // 2), n):
#                 p0[i * 3] = 0.15
#                 p0[i * 3 + 1] = -0.4 + 0.8 * (i - int(n // 2)) / max(int(n // 2 + n % 2 - 1), 1)
#                 p0[i * 3 + 2] = 1 / n
#         try:
#             p0[0] = 0.276
#             p0[1] = 0.117
#             p0[2] = 0.655
#         except:
#             pass
#         try:
#             p0[3] = 0.5
#             p0[4] = -0.01
#             p0[5] = 0.222
#         except:
#             pass
#         try:
#             p0[6] = 0.25
#             p0[7] = -0.098
#             p0[8] = 0.114
#         except:
#             pass
#             # for i in range(0, n):
#             #     p0[i * 3] = 0.1
#             #     p0[i * 3 + 1] = 0.1 - 0.01 * i
#             #     p0[i * 3 + 2] = 1 / n
#
#
#     if ref == 1:
#         if platform.system() == 'Windows':
#             realpath = str(self.dir_path) + str('\\\\INSexp.txt')
#         else:
#             realpath = str(self.dir_path) + str('/INSexp.txt')
#         INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
#         p0 = np.array([float(0)]*len(INS))
#         n = int(len(INS)/3)
#         for i in range(0, n):
#             p0[i * 3] = INS[i * 3] ** 2
#             p0[i * 3 + 1] = INS[i * 3 + 1]
#             p0[i * 3 + 2] = INS[i * 3 + 2] ** 2
#
#     SC = 0
#     for i in range(0, int((len(p0)) / 3)):
#         SC += p0[i * 3 + 2]
#     for i in range(0, int((len(p0)) / 3)):
#         p0[i * 3 + 2] = p0[i * 3 + 2] / SC
#
#     # if (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10 < (max(B) - np.sqrt(max(B))) * 0.98:
#     #     p0 = np.insert(p0, 0, max(B) - np.sqrt(max(B)))
#     # else:
#     p0 = np.insert(p0, 0, (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10)
#
#
#     bounds = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])
#     for i in range(0, int((len(p0) - 1) / 3)):
#         bounds[0][1 + i * 3] = -1.1
#         bounds[1][1 + i * 3] = 1.1
#         bounds[0][1 + i * 3 + 1] = -1
#         bounds[1][1 + i * 3 + 1] = 1
#
#     if platform.system() == 'Windows':
#         realpath = str(self.dir_path) + str('\\\\ABSorber3.txt')
#     else:
#         realpath = str(self.dir_path) + str('/ABSorber3.txt')
#     PPP = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
#
#     if mode == 0:
#         try:
#             Be_p = np.genfromtxt(
#                 str(self.dir_path) + str('\\\\Be.txt') * (platform.system() == 'Windows') + str('/Be.txt') * (platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
#             print('Be file was read')
#         except:
#             Be_p = np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
#             print('COULD NOT READ Be.txt')
#         def INSSS(x_exp, p):
#             return ins.INS(x_exp, p, JN, pool, PPP, x00=x0, MulCo=MulCo, Be_p=Be_p)
#
#     elif mode == 1:
#         global p
#         read_spectrum(self)
#         def INSSS(x_exp, p):
#             pNorm = np.array([float(0)] * NBA)
#             pNorm[0] = 1
#             Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]
#             print('Normalization integral equal to', Norm)
#             return m5.TI(x_exp, p[:-len(p0)], model, JN, pool, x0, MulCo, p[-len(p0):], Distri, Cor, Norm=Norm)
#
#
#
#
#
#     JN00 = int(self.JN0.text)
#     JN = max(JN00*2, 64)
#
#     start_time = time.time()
#
#     if mode ==0:
#         p = p0
#         p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
#     elif mode == 1:
#         p = np.concatenate((p, p0))
#         p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
#         p0 = p[-len(p0):]
#         p = p[:-len(p0)]
#
#     SC = 0
#     for i in range(0, int((len(p) - 1) / 3)):
#         SC += abs(p[1 + i * 3 + 2])
#     for i in range(0, int((len(p) - 1) / 3)):
#         p[1 + i * 3 + 2] = p[1 + i * 3 + 2] / SC
#     p[0] = p[0] * SC
#
#     INSt = np.delete(p, 0)
#     for i in range(0, int(len(INSt) / 3)):
#         INSt[i * 3] = np.sqrt(abs(INSt[i * 3]))
#         INSt[i * 3 + 2] = np.sqrt(abs(INSt[i * 3 + 2]))
#
#     x0t, MulCot = x0, MulCo
#     hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
#     x0, MulCo = m5.limits(pool, int(JN), INSt)
#     hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
#     if hi2t - hi2 > 0.01:
#         x0, MulCo = x0t, MulCot
#
#     for Nc in range (0, 3*(1-ref)): #int(self.INS_number.text)):
#
#         hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
#         # print(x0, MulCo, hi2)
#
#         if len(p) > 4:
#             n_s = int((len(p)-1) / 3)
#             # I_ch = int(1 + 2)
#             # for i in range(0, n_s):
#             #     if p[1 + i * 3 + 2] < p[I_ch]:
#             #         I_ch = int(1 + i * 3 + 2)
#             J = 0
#             for j in range(0, n_s):
#                 I_ch = 1 + J * 3 + 2
#                 pt = np.copy(p)
#                 pt = np.delete(pt, I_ch)
#                 pt = np.delete(pt, I_ch-1)
#                 pt = np.delete(pt, I_ch-2)
#                 boundst = np.copy(bounds[0])
#                 boundst = np.delete(boundst, I_ch)
#                 boundst = np.delete(boundst, I_ch - 1)
#                 boundst = np.delete(boundst, I_ch - 2)
#                 boundstt = np.copy(bounds[1])
#                 boundstt = np.delete(boundstt, I_ch)
#                 boundstt = np.delete(boundstt, I_ch - 1)
#                 boundstt = np.delete(boundstt, I_ch - 2)
#                 boundsttt = np.array([boundst, boundstt])
#                 SC = 0
#                 for k in range(0, int((len(pt) - 1) / 3)):
#                     SC += abs(pt[1 + k * 3 + 2])
#                 for k in range(0, int((len(pt) - 1) / 3)):
#                     pt[1 + k * 3 + 2] = pt[1 + k * 3 + 2] / SC
#                 # pt[0] = pt[0] * SC
#
#                 pt, ert, hi2t = mi.minimi_hi(INSSS, A, B, pt, bounds=boundsttt, tau0=0.0001, MI=20, MI2=10, eps=10 ** -6, fixCH = 1)
#
#                 INSt = np.delete(pt, 0)
#                 for i in range(0, int(len(INSt) / 3)):
#                     INSt[i * 3] = np.sqrt(abs(INSt[i * 3]))
#                     INSt[i * 3 + 2] = np.sqrt(abs(INSt[i * 3 + 2]))
#
#                 x0t, MulCot = x0, MulCo
#                 hi2tt = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
#                 x0, MulCo = m5.limits(pool, int(JN), INSt)
#                 hi2t = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
#                 if hi2t - hi2tt > 0.01:
#                     x0, MulCo = x0t, MulCot
#                     hi2t = hi2tt
#
#                 if hi2t - hi2 < 0.001:
#                     p = pt
#                     bounds = boundsttt
#                     SC = 0
#                     for i in range(0, int((len(p) - 1) / 3)):
#                         SC += abs(p[1 + i * 3 + 2])
#                     for i in range(0, int((len(p) - 1) / 3)):
#                         p[1 + i * 3 + 2] = p[1 + i * 3 + 2] / SC
#                     p[0] = p[0] * SC
#                 else:
#                     x0, MulCo = x0t, MulCot
#                     J += 1
#
#         print(x0, MulCo)
#         p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, bounds=bounds, tau0=0.0001, MI=20, MI2=10, eps=10 ** -6, fixCH=1)
#         print(p)
#         print(hi2)
#         SC = 0
#         for i in range(0, int((len(p) - 1) / 3)):
#             SC += abs(p[1 + i * 3 + 2])
#         for i in range(0, int((len(p) - 1) / 3)):
#             p[1 + i * 3 + 2] = p[1 + i * 3 + 2] / SC
#         p[0] = p[0] * SC
#
#         INSt = np.delete(p, 0)
#         for i in range(0, int(len(INSt) / 3)):
#             INSt[i * 3] = np.sqrt(abs(INSt[i * 3]))
#             INSt[i * 3 + 2] = np.sqrt(abs(INSt[i * 3 + 2]))
#
#         x0t, MulCot = x0, MulCo
#         hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
#         x0, MulCo = m5.limits(pool, int(JN), INSt)
#         hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
#         if hi2t - hi2 > 0.01:
#             x0, MulCo = x0t, MulCot
#
#     JN = JN00
#     for Nc in range(0, 3):
#         p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
#
#         SC = 0
#         for i in range(0, int((len(p) - 1) / 3)):
#             SC += abs(p[1 + i * 3 + 2])
#         for i in range(0, int((len(p) - 1) / 3)):
#             p[1 + i * 3 + 2] = p[1 + i * 3 + 2] / SC
#         p[0] = p[0] * SC
#
#         INSt = np.delete(p, 0)
#         for i in range(0, int(len(INSt) / 3)):
#             INSt[i * 3] = np.sqrt(abs(INSt[i * 3]))
#             INSt[i * 3 + 2] = np.sqrt(abs(INSt[i * 3 + 2]))
#
#         x0, MulCo = m5.limits(pool, int(JN), INSt)
#
#
#     print('Instrumental function take ', time.time() - start_time, 'seconds')
#
#
#
#     p = np.array(p)
#     SC = 0
#     for k in range(0, int((len(p) - 1) / 3)):
#         SC += abs(p[1 + k * 3 + 2])
#     print('sum of INS', SC)
#     print('x0 ', x0)
#     print('MulCo ', MulCo)
#     F = INSSS(A, p)
#     JN = JN*4
#     F2 = INSSS(A, p)
#
#     fig, ax1 = plt.subplots(figsize=(2942/300, 4.5), dpi=300)
#     # fig = plt.figure(figsize=(2942/300, 4.5), dpi=300)
#     ax = fig.add_subplot(111)
#     plt.xlim(min(A), max(A))
#     plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
#     plt.plot(A, F, color='r')
#     plt.fill_between(A, F.astype(float), np.array(p[0], dtype=float), color='r', alpha=1, zorder=2)
#     plt.plot(A, B, linestyle='None', marker='x', color='m')
#     plt.plot(A, B - F + min(B) - max(B - F), color='lime')
#     plt.plot(A, B - B + min(B) - max(B - F), linestyle='--', color=self.gridcolor)
#     # plt.plot(A, F2 - F + max(B)+4*np.sqrt(max(B)), color='cyan')
#     plt.plot(A, F2 - F + min(B) - max(B - F) + min(B - F) - max(F2 - F), color='cyan')
#     plt.text(0, -0.1, os.path.abspath(self.path_list[0]), horizontalalignment='left', verticalalignment='center', color='m', transform = ax.transAxes)
#     plt.title('χ² = %.3f' % hi2, y=1, color='r')
#
#     if platform.system() == 'Windows':
#         realpath = str(self.dir_path) + str('\\\\result.svg')
#     else:
#         realpath = str(self.dir_path) + str('/result.svg')
#     fig.savefig(realpath, bbox_inches='tight')
#     if platform.system() == 'Windows':
#         realpath = str(self.dir_path) + str('\\\\result.png')
#     else:
#         realpath = str(self.dir_path) + str('/result.png')
#     fig.savefig(realpath, bbox_inches='tight')
#     # fig.savefig('result.png', bbox_inches='tight')
#     # fig.savefig('result.pdf', bbox_inches='tight')
#     # fig.savefig('result.jpg', bbox_inches='tight')
#     # fig.savefig('result.svg', bbox_inches='tight')
#     # plt.show()
#     # plt.close()
#     plt.cla()
#     plt.clf()
#     plt.close('all')
#     plt.close(fig)
#     gc.collect()
#
#     p = np.delete(p, 0)
#     for i in range(0, int(len(p)/3)):
#         p[i*3] = np.sqrt(abs(p[i*3]))
#         p[i*3+2] = np.sqrt(abs(p[i*3+2]))
#     if platform.system() == 'Windows':
#         realpath = str(self.dir_path) + str('\\\\INSexp.txt')
#     else:
#         realpath = str(self.dir_path) + str('/INSexp.txt')
#     f = open(realpath, "w")
#     for i in range(0, len(p)):
#         f.write(str(p[i]) + ' ')
#     f.close()
#
#
#
#     if platform.system() == 'Windows':
#         realpath = str(self.dir_path) + str('\\\\INSint.txt')
#     else:
#         realpath = str(self.dir_path) + str('/INSint.txt')
#     f = open(realpath, "w")
#     f.write(str(MulCo) + ' ')
#     f.write(str(x0) + ' ')
#     f.close()
#
#     RA = True
#     self.SP_DI.text = 'Distribution'
#     RL = 2
#     RA2.set()
#     # self.play_btn.disabled = False
#     # self.INS_btn.disabled = False
#     # self.INS_btn2.disabled = False
#     # self.show_btn.disabled = False
#     # self.showM_btn.disabled = False
#     # self.cal_btn.disabled = False




def instrumental(self, ref, mode=0):
    global RA, RL, RR, x0, MulCo, PPP
    RL = 1

    file = os.path.abspath(self.path_list[0])

    A, B = read_spectrum(self, file)

    CMS_ch = 0

    if ref == 0:
        x0 = -0.01
        MulCo = 2.2

        # n = int(self.INS_number.text)
        # p0 = np.array([float(0.001)] * n * 3)
        # if n > 3:
        #     for i in range(0, int(n // 2)):
        #         p0[i * 3] = 0.3
        #         p0[i * 3 + 1] = -1.5 + 3 * i / max(int(n // 2 - 1), 1)
        #         p0[i * 3 + 2] = 1 / n
        #     for i in range(int(n // 2), n):
        #         p0[i * 3] = 0.15
        #         p0[i * 3 + 1] = -0.4 + 0.8 * (i - int(n // 2)) / max(int(n // 2 + n % 2 - 1), 1)
        #         p0[i * 3 + 2] = 1 / n
        # if mode == 0 or n <= 3:
        #     try:
        #         p0[0] = 0.276
        #         p0[1] = 0.117
        #         p0[2] = 0.655
        #     except:
        #         pass
        #     try:
        #         p0[3] = 0.5
        #         p0[4] = -0.01
        #         p0[5] = 0.222
        #     except:
        #         pass
        #     try:
        #         p0[6] = 0.25
        #         p0[7] = -0.098
        #         p0[8] = 0.114
        #     except:
        #         pass

        n = int(self.INS_number.text)
        print('instrumental procedure start with', ref, mode, n)
        p0 = np.array([])

        if mode == 0 or mode == 2:
            p0 = np.array([float(0.001)] * (3 * (n >= 3) + n * (n < 3)) * 3)
            try:
                # p0[0] = 0.276
                # p0[1] = 0.117
                # p0[2] = 0.655
                p0[0] = 0.2
                p0[1] = 0.055
                p0[2] = 0.8
            except:
                pass
            try:
                # p0[3] = 0.5
                # p0[4] = -0.01
                # p0[5] = 0.222
                p0[3] = 0.11
                p0[4] = -0.13
                p0[5] = 0.44
            except:
                pass
            try:
                # p0[6] = 0.25
                # p0[7] = -0.098
                # p0[8] = 0.114
                p0[6] = 0.555
                p0[7] = -0.07
                p0[8] = 0.4
            except:
                pass

        if n > 3 or mode == 1:
            p00c = np.array([float(0.001)] * ((n-3)*(mode==0 + mode==2) + n*(mode==1)) * 3)
            n0 = int(len(p00c)/3)
            for i in range(0, int(n0 // 2)):
                p00c[i * 3] = 0.3
                p00c[i * 3 + 1] = -1.5 + 3 * i / max(int(n0 // 2 - 1), 1)
                p00c[i * 3 + 2] = 1 / n0
            for i in range(int(n0 // 2), n0):
                p00c[i * 3] = 0.15
                p00c[i * 3 + 1] = -0.4 + 0.8 * (i - int(n0 // 2)) / max(int(n0 // 2 + n0 % 2 - 1), 1)
                p00c[i * 3 + 2] = 1 / n0
            p0 = np.concatenate((p0, p00c))
            print(p00c)

        print('here is initial guess for INS')
        print(p0)

    if ref == 1:
        if self.switch.active == True:
            if self.fitway[0].active == True:
                if float(self.L0.text) > 0:
                    INS = np.array([float(self.L0.text)])
                else:
                    INS = np.array([0.1])
                p0 = np.copy(INS)
                print('instrumental procedure for CMS start with', ref, mode)
                print('here is initial guess for INS: ', p0)
                CMS_ch = 1
            else:
                realpath = str(self.dir_path) + OSslesh + str('INSexp.txt')
                INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
                p0 = np.array([float(0)]*len(INS))
                n = int(len(INS)/3)
                # for i in range(0, n):
                #     p0[i * 3] = INS[i * 3] ** 2
                #     p0[i * 3 + 1] = INS[i * 3 + 1]
                #     p0[i * 3 + 2] = INS[i * 3 + 2] ** 2
                p0 = np.copy(INS)
                print('instrumental procedure start with', ref, mode, n)
                print('here is initial guess for INS')
                print(p0)
    if CMS_ch == 0:
        SC = 0
        for i in range(0, int((len(p0)) / 3)):
            SC += p0[i * 3 + 2]**2
        for i in range(0, int((len(p0)) / 3)):
            p0[i * 3 + 2] = np.sqrt(p0[i * 3 + 2]**2 / SC)

        bounds0 = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])
        for i in range(0, int((len(p0)) / 3)):
            # bounds0[0][i * 3] = -1.1
            # bounds0[1][i * 3] = 1.1
            # bounds0[0][i * 3 + 1] = -2
            # bounds0[1][i * 3 + 1] = 2

            bounds0[0][i * 3] = -0.7
            bounds0[1][i * 3] = 0.7
            bounds0[0][i * 3 + 1] = -1.5
            bounds0[1][i * 3 + 1] = 1.5
    else:
        bounds0 = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])
        bounds0[0][0] = 0.001
        bounds0[1][0] = 1



    global p


    if mode == 1:
        global val
        global model
        global Distri
        global Cor
        global Expr, NExpr

        # read_spectrum(self)

        con1, con2, con3, Distri, Cor, Expr, NExpr, fix_tmp, DistriN = read_model(self)

        for i in range(0, len(NExpr)):
            p[NExpr[i]] = eval(Expr[i])
        for i in range(0, len(con1)):
            p[int(con1[i])] = p[int(con2[i])] * con3[i]

        global confu

        if len(con1) > 0:
            confu = np.array([con1, con2, con3])
        else:
            confu = np.array([[-1], [-1], [-1]])

        global er, name

        name = [[]]
        for i in range(0, len(self.nametable)):
            tmpname = []
            for j in range(0, len(self.nametable[i])):
                tmpname.append(self.nametable[i][j].text)
            name.append(tmpname)
        name.pop(0)

        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        fix = np.array([], dtype=int)

        for j in range(0, NBA):
            if self.rboundstable[0][j].text != 'None' and self.rboundstable[0][j].text != '':
                bounds[1][j] = float(self.rboundstable[0][j].text)
            if self.lboundstable[0][j].text != 'None' and self.lboundstable[0][j].text != '':
                bounds[0][j] = float(self.lboundstable[0][j].text)
            if self.fixtable[0][j].active == True:
                fix = np.append(fix, j)

        V = NBA - 1

        for i in range(0, len(val)):
            LenM = mod_len_def(str(val[i][0].text))
            for j in range(0, LenM):
                V += 1
                if self.rboundstable[i][j].text != 'None' and self.rboundstable[i][j].text != '':
                    bounds[1][V] = float(self.rboundstable[i][j].text)
                if self.lboundstable[i][j].text != 'None' and self.lboundstable[i][j].text != '':
                    bounds[0][V] = float(self.lboundstable[i][j].text)
                if self.fixtable[i][j].active == True:
                    fix = np.append(fix, V)

        fix = np.concatenate((fix, fix_tmp), axis=0)  # fix_tmp do not work correctly with multidimensional distri
        fix = np.concatenate((fix, con1), axis=0)
        fix = np.concatenate((fix, DistriN), axis=0)
        fix = np.concatenate((fix, NExpr), axis=0)
        fix = np.unique(fix)

        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)
        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                # pNorm = np.array([float(0)] * NBA)
                # pNorm[0] = 1
                # Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, p[mod_p_len:], [0], [0])[0]
                # print('Normalization integral equal to', Norm)
                # return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:], Distri, Cor, Norm=Norm)
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:], Distri, Cor)
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Distri, Cor, Met=1)
    elif mode == 0:
        # mod_p_len = 1
        # bounds_p = np.array([[-np.inf], [np.inf]])
        # bounds = np.concatenate((bounds_p, bounds0), axis = 1)
        # fix = np.array([], dtype=int)
        # p = np.array([(B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10])
        # p = np.concatenate((p, p0))
        # print(p)
        #
        # if platform.system() == 'Windows':
        #     realpath = str(self.dir_path) + str('\\\\ABSorber3.txt')
        # else:
        #     realpath = str(self.dir_path) + str('/ABSorber3.txt')
        # PPP = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
        #
        # try:
        #     Be_p = np.genfromtxt(str(self.dir_path) + str('\\\\Be.txt')*(platform.system() == 'Windows') + str('/Be.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
        #     print('Be file was read')
        # except:
        #     Be_p = np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
        #     print('COULD NOT READ Be.txt')
        #
        # def INSSS(x_exp, pP):
        #     return ins.INS(x_exp, pP, JN, pool, PPP, x00=x0, MulCo=MulCo, Be_p=Be_p)
        model = ['Doublet', 'Singlet']
        if CMS_ch == 0:
            p = np.array([(B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10])
            p = np.concatenate((p, np.array([0, 0, 0, 0, 0, 0, 0])))

        try:
            Be_param = np.genfromtxt(
                str(self.dir_path) + str('\\\\Be.txt') * (platform.system() == 'Windows') + str('/Be.txt') * (
                            platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
            print('Be file was read')
        except:
            Be_param = np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
            print('COULD NOT READ Be.txt')

        p = np.concatenate((p, Be_param))
        p1 = np.array([4.6, -0.097, 0.098, 0.0])
        p = np.concatenate((p, p1))

        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][15] = 0.001
        if CMS_ch == 0:
            fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18],
                           dtype=int)

        V = NBA - 1

        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)

        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:])
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Met=1)

    elif mode == 2:
        model = [ 'Doublet', 'Sextet']
        if CMS_ch == 0:
            p = np.array([(B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10])
            p = np.concatenate((p, np.array([0,0,0,0,0,0,0])))
        if CMS_ch == 1:
            bg_tmp = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10
            p = np.array([bg_tmp*0.6])
            p = np.concatenate((p, np.array([0,0,0,bg_tmp*0.4,0,0,0])))

        try:
            Be_param = np.genfromtxt(str(self.dir_path) + str('\\\\Be.txt')*(platform.system() == 'Windows') + str('/Be.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
            print('Be file was read')
        except:
            Be_param = np.array([0.048, 0.103, -0.259, 0.098, 0.105, 0.265, 1])
            print('COULD NOT READ Be.txt')
        if CMS_ch == 1:
            Be_param[0] = 0

        p = np.concatenate((p, Be_param))
        p1 = np.array([7.5, 0, 0, 33.04, 0.098, 0, 0.5, 0, 0, 0, 3])
        p = np.concatenate((p, p1))

        bounds = np.array([[-np.inf] * len(p), [np.inf] * len(p)], dtype=float)
        bounds[0][0] = 0
        bounds[0][15] = 0.001
        if CMS_ch == 0:
            fix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25], dtype=int)
        if CMS_ch == 1:
            fix = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25], dtype=int)
            bounds[0][4] = 0

        V = NBA - 1

        bounds = np.concatenate((bounds, bounds0), axis=1)
        mod_p_len = len(p)

        p = np.concatenate((p, p0))
        print(p)
        print(fix)
        if CMS_ch == 0:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, x0, MulCo, p[mod_p_len:])
        if CMS_ch == 1:
            def INSSS(x_exp, p):
                return m5.TI(x_exp, p[:mod_p_len], model, JN, pool, 0, MulCoCMS, p[-1], Met=1)


    def INS_norm(p):
        SC = 0
        for i in range(0, int((len(p[mod_p_len:])) / 3)):
            SC += p[mod_p_len + i * 3 + 2]**2
        for i in range(0, int((len(p[mod_p_len:])) / 3)):
            p[mod_p_len + i * 3 + 2] = np.sqrt(p[mod_p_len + i * 3 + 2]**2 / SC)
        p[0] = p[0] * SC
        return p

    JN00 = int(self.JN0.text)
    JN = max(JN00*2, 64)

    start_time = time.time()

    p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, fix = fix, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
    if CMS_ch == 0:
        p = INS_norm(p)

        x0t, MulCot = x0, MulCo
        hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])
        hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        if hi2 - hi2t > 0.01:
            x0, MulCo = x0t, MulCot
            hi2 = np.copy(hi2t)
    print(p)


    for Nc in range (0, 2*(1-ref)): #int(self.INS_number.text)):

        n_s = int((len(p[mod_p_len:])) / 3)
        # if len(p) > 3:
        if n_s > 3:
            print('trying to reduce number of lines in INS')
            J = 0
            for j in range(0, n_s):
                I_ch = mod_p_len + J * 3 + 2
                pt = np.copy(p)
                pt = np.delete(pt, I_ch)
                pt = np.delete(pt, I_ch-1)
                pt = np.delete(pt, I_ch-2)
                boundst = np.copy(bounds[0])
                boundst = np.delete(boundst, I_ch)
                boundst = np.delete(boundst, I_ch - 1)
                boundst = np.delete(boundst, I_ch - 2)
                boundstt = np.copy(bounds[1])
                boundstt = np.delete(boundstt, I_ch)
                boundstt = np.delete(boundstt, I_ch - 1)
                boundstt = np.delete(boundstt, I_ch - 2)
                boundsttt = np.array([boundst, boundstt])
                pt = INS_norm(pt)

                pt, ert, hi2t = mi.minimi_hi(INSSS, A, B, pt, fix = fix, bounds=boundsttt, tau0=0.0001, MI=20, MI2=10, eps=10 ** -6, fixCH = 1)

                x0t, MulCot = x0, MulCo
                hi2tt = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
                x0, MulCo = m5.limits(pool, int(JN), pt[mod_p_len:])
                hi2t = np.sum((B - INSSS(A, pt)) ** 2 / (abs(B) + 1) / (len(B)))
                if hi2t - hi2tt > 0.01:
                    x0, MulCo = x0t, MulCot
                    hi2t = hi2tt

                if hi2t - hi2 < 0.01:
                    p = pt
                    bounds = boundsttt
                    p = INS_norm(p)
                    print('hi squared:', hi2, hi2t)
                    hi2 = np.copy(hi2t)

                else:
                    x0, MulCo = x0t, MulCot
                    J += 1

        print(x0, MulCo)
        p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, fix = fix, bounds=bounds, tau0=0.0001, MI=20, MI2=10, eps=10 ** -6, fixCH=1)
        print(p)
        print(hi2)
        p = INS_norm(p)

        x0t, MulCot = x0, MulCo
        hi2t = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])
        hi2 = np.sum((B - INSSS(A, p)) ** 2 / (abs(B) + 1) / (len(B)))
        if hi2 - hi2t > 0.01:
            x0, MulCo = x0t, MulCot
            hi2 = hi2t

    JN = JN00

    for Nc in range(0, 3):
        p, er, hi2 = mi.minimi_hi(INSSS, A, B, p, fix = fix, bounds=bounds, tau0=0.0001, MI=20, MI2=20, eps=10 ** -6, fixCH=1)
        if CMS_ch == 0:
            p = INS_norm(p)
            x0, MulCo = m5.limits(pool, int(JN), p[mod_p_len:])


    print('Instrumental function take ', time.time() - start_time, 'seconds')

    p = np.array(p)
    print(p)
    if CMS_ch == 0:
        SC = 0
        for k in range(0, int((len(p[mod_p_len:])) / 3)):
            SC += p[mod_p_len + k * 3 + 2]**2
        print('sum of INS', SC)
        print('x0 ', x0)
        print('MulCo ', MulCo)
    if CMS_ch == 1:
        print('G  ', p[-1])

    F = INSSS(A, p)
    JN = JN*4
    F2 = INSSS(A, p)

    fig, ax1 = plt.subplots(figsize=(2942/300, 4.5), dpi=300)
    # fig = plt.figure(figsize=(2942/300, 4.5), dpi=300)
    # ax = fig.add_subplot(111)
    plt.xlim(min(A), max(A))
    plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
    plt.plot(A, F, color='r')
    plt.fill_between(A, F.astype(float), np.array(p[0] + p[3] * p[0]/10**2 * A + p[2] * p[0] / 10**4 * (A - p[1])**2 + p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float), color='r', alpha=1, zorder=2)
    plt.plot(A, B, linestyle='None', marker='x', color='m')
    plt.plot(A, B - F + min(B) - max(B - F), color='lime')
    plt.plot(A, B - B + min(B) - max(B - F), linestyle='--', color=self.gridcolor)
    # plt.plot(A, F2 - F + max(B)+4*np.sqrt(max(B)), color='cyan')
    plt.plot(A, F2 - F + min(B) - max(B - F) + min(B - F) - max(F2 - F), color='cyan')
    plt.text(0, -0.1, os.path.abspath(self.path_list[0]), horizontalalignment='left', verticalalignment='center', color='m', transform = ax1.transAxes)
    plt.title('χ² = %.3f' % hi2, y=1, color='r')
    ymin, ymax = ax1.get_ylim()
    ax2 = ax1.twinx()
    ax2.set_ylim((ymin / (p[0]), ymax / (p[0])))


    realpath = str(self.dir_path) + OSslesh + str('result.svg')
    fig.savefig(realpath, bbox_inches='tight')

    realpath = str(self.dir_path) + OSslesh + str('result.png')
    fig.savefig(realpath, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close(fig)
    gc.collect()

    INSp = p[mod_p_len:]
    # for i in range(0, int(len(p)/3)):
    #     p[i*3] = np.sqrt(abs(p[i*3]))
    #     p[i*3+2] = np.sqrt(abs(p[i*3+2]))

    realpath = str(self.dir_path) + OSslesh + str('INSexp.txt')*(CMS_ch == 0) + str('GCMS.txt')*(CMS_ch == 1)


    if CMS_ch == 0:
        f = open(realpath, "w")
        for i in range(0, len(INSp)):
            f.write(str(INSp[i]) + ' ')
        f.close()
        realpath = str(self.dir_path) + OSslesh + str('INSint.txt')
        f = open(realpath, "w")
        f.write(str(MulCo) + ' ')
        f.write(str(x0) + ' ')
        f.close()
    else:
        f = open(realpath, "w")
        f.write(str('%.3f' % INSp[-1]))
        f.close()
        global L0_text
        L0_text = str('%.3f' % INSp[-1])

    if mode == 1:
        p = p[:mod_p_len]
        RR = True
        for i in range(0, len(self.realtable)-1):
            self.realtable[i][0].color = [1,1,1,1]

    RA = True
    self.SP_DI.text = 'Distribution'
    RL = 2
    RA2.set()

if __name__ == '__main__':
    mp.freeze_support()
    global pool
    pool = mp.Pool((mp.cpu_count())*(mp.cpu_count()<=4) + (mp.cpu_count()-1)*(mp.cpu_count()>4))
    # print('number of cores are ', mp.cpu_count())
    from kivy.app import App
    from kivy.graphics import Color, Rectangle
    from kivy.uix.widget import Widget
    from kivy.uix.button import Button
    from kivy.uix.switch import Switch
    from kivy.uix.label import Label
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.textinput import TextInput
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.filechooser import FileChooserIconView
    from kivy.uix.filechooser import FileChooserController
    from kivy.uix.image import Image
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.uix.dropdown import DropDown
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.popup import Popup
    from kivy.factory import Factory
    from plyer import filechooser
    from kivy.properties import ListProperty
    from kivy.uix.scrollview import ScrollView
    from itertools import chain
    from kivy.uix.popup import Popup
    from kivy.config import Config
    # from kivy.modules import keybinding

    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    # Config.maxfps = 30

    class PhysicsApp(App):

        # global image
        image = None
        log = None
        process_path = None
        save_path = None

        def __init__(self, **kwargs):
            super(PhysicsApp, self).__init__(**kwargs)
            # Window.bind(on_request_close=self.on_request_close)
            Window.bind(on_request_close=self.on_quit)
            Window.bind(on_key_down = self.functionalkeys)
            Window.size = (1600, 900)
            Window.minimum_width = 1270
            Window.minimum_height = 710
            Window.top = 50
            Window.left = 10
            # Window.maximize()
            self.title = 'SYNCMoss ESRF ID14'
            icon = str(os.getcwd()) + ("\\\\icon_r.ico")*(platform.system() == 'Windows') + ("/icon_r.ico")*(platform.system() != 'Windows')
            self.icon = icon

            # print(icon)
            # Config.set('kivy', 'window_icon', icon)
            # Config.write()

            # self.config_keyboard()

        def functionalkeys(self, window, keycode, *args, **kwargs):
            if keycode == 286 and self.play_btn.disabled == False:
                self.play_pressed(self)
            if keycode == 289:
                self.take_result(self)
            if keycode == 27:
                print('do not press escape!')
                return(True)
            if keycode == 13 and self.showM_btn.disabled == False:
                self.showM_pressed(self)



        # def config_keyboard(self):
        #     self._keyboard = Window.request_keyboard(self._keyboard_closed, self.root)
        #     self._keyboard.bind(on_key_down=self._on_keyboard_down)
        #
        # def on_touch_down(self, touch):
        #     self.isTextInput = False
        #     def filter(widget):
        #         for child in widget.children:
        #             filter(child)
        #         if isinstance(widget, TextInput) and widget.collide_point(*touch.pos):
        #             self.isTextInput = True
        #             widget.on_touch_down(touch)
        #     filter(self)
        #     if not self.isTextInput and self._keyboard is None:
        #         self.config_keyboard()
        #
        # def _keyboard_closed(self):
        #     self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        #     self._keyboard = None
        #
        # def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        #     print('The key', keycode, 'have been pressed')
        #     print(' - text is %r' % text)
        #     print(' - modifiers are %r' % modifiers)
        #
        #     # Keycode is composed of an integer + a string
        #     # If we hit escape, release the keyboard
        #
        #     return True


        # Function where we build app gui
        def build(self):

            self.dir_path = os.getcwd()
            self.workfolder = None
            self.workfolder_check = 0
            self.gridcolor = 'w'
            self.check_points_match = False
            self.newfilename = str('')
            self.newfilename2 = str('')
            self.BGcolor = 'k'

            # icon = str(self.dir_path) + ("\\icon_old.ico")*(platform.system() == 'Windows') + ("/icon_old.ico")*(platform.system() != 'Windows')
            # print(icon)
            # Config.set('PhysicsApp', 'window_icon', icon)

            main = BoxLayout(orientation="horizontal", spacing=2)
            # keybinding.start(Window, main)
            right = BoxLayout(orientation="vertical", spacing=2)
            # global image
            self.image = Image()
            right.add_widget(self.image)

            blt = BoxLayout(orientation="horizontal", spacing=3, size_hint=[1, 0.1])
            self.dark_light_mode = Button(text='Light spc', size_hint=[0.25, 1], font_size='18sp', background_color = [1.5, 1.5, 1.5, 1.5])
            self.dark_light_mode.bind(on_release=self.change_visiual)
            title = Label(text="Result", halign="center", valign='middle', size_hint=[1, 1], font_size='24sp', color=[255, 1, 1, 1])
            self.SP_DI = Button(text='Distribution', size_hint=[0.4, 1], font_size='18sp')
            self.SP_DI.bind(on_release=self.change_image)
            blt.add_widget(self.dark_light_mode)
            blt.add_widget(title)
            blt.add_widget(self.SP_DI)
            right.add_widget(blt)



            # tableR = GridLayout(cols=numco+1)
            self.realtable = [[]]
            self.tableR = GridLayout(cols=numco + 1, size_hint_y=None, size_hint_x=None,
                               row_default_height='30dp', col_default_width='105dp')
            self.tableR.height = '%.i dp' % int(numro * 2 * 30)
            self.tableR.width = '%.i dp' % int((numco + 1) * 105)


            for rows in range(0, numro*2):
                row = []
                self.realtable.append([])
                btn = Button(text='', halign="center",size_hint=(1, None), height=30, font_size='18sp',  background_color=[0, 0, 0, 1])
                btn.bind(on_release=partial(self.replot_result, rows))
                self.realtable[rows].append(btn)
                self.tableR.add_widget(self.realtable[rows][-1])
                for cols in range(numco):
                    cell = Label(text="", halign="center",size_hint=(1, None), height=30, font_size='18sp')
                    self.realtable[rows].append(cell)
                    self.realtable[rows][-1].markup = True
                    self.tableR.add_widget(self.realtable[rows][-1])



            self.scrollboxR = ScrollView(size_hint=(1, 1), viewport_size = (0.1, 0.1) , do_scroll_y = True, scroll_wheel_distance = 60,
                                    do_scroll_x = True, always_overscroll=False, bar_width = 15, bar_pos_y = 'right', scroll_type = ['bars'],
                                    bar_color = [0.5, 0, 1, 0.9], bar_inactive_color = [0.5, 0, 1, 0.3]) #, effect_cls = "ScrollEffect")
            self.scrollboxR.add_widget(self.tableR)
            right.add_widget(self.scrollboxR)
            # print(self.realtable)
            # print(len(self.realtable))
            # print(len(self.realtable[0]))

            blt = BoxLayout(orientation="horizontal", spacing=2, size_hint=[1, 0.1])
            self.log = TextInput(text="", halign="center", multiline=True, font_size='21sp')
            TkRes = Button(text='Take result as model (F8)', size_hint=[0.4, 1], font_size='18sp')
            TkRes.bind(on_release = self.take_result)
            blt.add_widget(self.log)
            blt.add_widget(TkRes)
            right.add_widget(blt)

            self.left = BoxLayout(orientation="vertical", spacing=2)

            up = BoxLayout(orientation="horizontal", spacing=2, size_hint=[1, 0.1])
            loadmod_btn = Button(text="Load\nmodel", size_hint=[0.35, 1], color=[255, 255, 255, 1], background_color=[0.4, 255, 255, 1], halign='center', font_size='16sp')
            loadmod_btn.bind(on_release=self.loadmod_pressed)
            self.switch = Switch(active=True, size_hint=[0.4, 1])
            self.switch.canvas.children[2].source = "Switch.png"
            INTERUPT = Button(text='! INTERRUPT !', size_hint=[0.4, 1], color=[255, 255, 255, 1], background_color=[255, 0, 0, 1], halign='center', font_size='16sp')
            INTERUPT.bind(on_release = self.INTERUPT)
            self.btncleanmodel = Button(text="Clean\nmodel\n(2 clk)", halign='center', size_hint=[0.4, 1], background_color=[2.5, 1.5, 0, 1], color=[255, 255, 255, 1])
            self.btncleanmodel.bind(on_release=self.clean_model)
            self.waitingdoubleclick = 0

            # title = Label(text="Model", halign="center", valign='middle', color=[255, 1, 1, 1], font_size='24sp')


            self.cal_path = str(self.dir_path) + str('\\\\Calibration.dat')*(platform.system() == 'Windows') + str('/Calibration.dat')*(platform.system() != 'Windows')


            cal_cho_blt = BoxLayout(orientation="vertical", size_hint=[0.45, 1], spacing=2)
            self.cal_cho_title = Label(text="Velocity\ndown-up:", halign="center", valign='middle', color=[255, 1, 1, 1], font_size='16sp')
            self.Vel_start = CheckBox(background_checkbox_normal='UD.png', background_checkbox_down='DU.png')
            # self.cal_cho_text = Label(text=os.path.basename(self.cal_path), halign="center", valign='middle', color=[255, 1, 1, 1],  font_size='16sp')
            # cal_cho_text = TextInput(text="Calibration.dat", halign="center", size_hint=[1, 1], multiline=False, font_size='12sp')
            self.Vel_start.active = False
            cal_cho_blt.add_widget(self.cal_cho_title)
            cal_cho_blt.add_widget(self.Vel_start)
            # cal_cho_blt.add_widget(self.cal_cho_text)
            cal_cho_btn = Button(text="Choose\ncalibration\nfile", size_hint=[0.38, 1], color=[255, 255, 255, 1], background_color=[1, 2, 3, 1], halign='center', font_size='16sp')
            cal_cho_btn.bind(on_release=self.Calibration_path)
            self.cal_btn = Button(text="Calibrate", size_hint=[0.3, 1], color=[255, 255, 255, 1], background_color=[0.4, 255, 255, 1], halign='center', font_size='16sp',
                                  background_disabled_down='', background_disabled_normal='',disabled_color=[1, 1, 1, 1])
            self.cal_btn.bind(on_release=self.Calibration)
            self.vel_btn = Button(text="RAW\nto\ndat", size_hint=[0.15, 1], color=[255, 255, 255, 1], background_color=[0.4, 255, 255, 1], halign='center', font_size='16sp')
            self.vel_btn.bind(on_release=self.velocityscale)
            up.add_widget(loadmod_btn)
            up.add_widget(self.btncleanmodel)
            up.add_widget(self.switch)
            # up.add_widget(title)
            up.add_widget(cal_cho_btn)
            up.add_widget(cal_cho_blt)
            up.add_widget(self.cal_btn)
            up.add_widget(self.vel_btn)
            up.add_widget(INTERUPT)
            self.left.add_widget(up)



            # ['lr-tb', 'tb-lr', 'rl-tb', 'tb-rl', 'lr-bt', 'bt-lr', 'rl-bt', 'bt-rl']
            self.table = GridLayout(cols=numco+1, size_hint_y = None, size_hint_x = None,
                               row_default_height = '100dp', col_default_width = '95dp')
            self.table.height = '%.i dp' % int(numro * 100)
            self.table.width = '%.i dp' % int((numco+1) * 95)

            self.realtableP = [[]]
            self.params = []
            self.lboundstable = [[]]
            self.rboundstable = [[]]
            self.nametable = [[]]
            self.fixtable = [[]]
            mod1 = [[]]
            mod2 = [[]]
            mod3 = [[]]
            self.start = []
            self.startlabel = []
            self.startlabel2 = []


            self.buttons = []
            self.dd = []
            for rows in range(0, numro):
                self.buttons.append(Button())
                self.buttons[rows] = Button(text='None', size_hint=(None, None), height=30, font_size='18sp',  background_color=[1, 1, 1, 1])
                self.dd.append(DropDown(auto_width=False, width =120))
                for index in ['Singlet', 'Doublet', 'Sextet', 'MDGD', 'Relax_MS', 'Relax_2S', 'Hamilton_mc', 'Hamilton_pc', 'ASM', 'Be', 'KB_nano', 'Distr', 'Corr', 'Variables', 'Expression', 'Delete', 'Insert', 'Nbaseline']: #, 'Average_H'
                    btn = Button(text='%s' % index, size_hint_y=None, height=30, font_size='18sp',  background_color=[0.5, 0.5, 2, 1])
                    if index == 'Delete' or index == 'Insert':
                        btn.background_color = [4, 1, 0, 2]
                    if index == 'Distr' or index == 'Corr':
                        btn.background_color = [0, 2, 2, 2]
                    if index == 'Variables' or index == 'Expression':
                        btn.background_color = [0, 1, 2, 2]
                    if index == 'Nbaseline':
                        btn.background_color = [1, 0, 2, 2]
                    # btn.bind(on_release=lambda btn: dd[rows].select(btn.text))
                    btn.bind(on_release=partial(self.select, self.buttons[rows], btn.text, rows))
                    btn.bind(on_release=self.dd[rows].dismiss)
                    self.dd[rows].add_widget(btn)
                # btn.bind(on_release=dd[rows].dismiss)

                # dd[rows].bind(on_select=lambda v, x: setattr(buttons[rows], 'text', x))
                self.buttons[rows].bind(on_release=self.dd[rows].open)

            row = []
            self.realtableP.append([])
            self.lboundstable.append([])
            self.rboundstable.append([])
            self.nametable.append([])
            self.fixtable.append([])
            mod1.append([])
            mod2.append([])
            mod3.append([])
            self.startlabel.append(Label(text="Name  |  fix", halign="center", valign='middle', font_size='18sp'))
            self.startlabel2.append(Label(text="boundaries", halign="center", valign='middle', font_size='18sp'))
            self.btn_background = Button(text='baseline', size_hint=(None, None), height=30, color=[255, 255, 255, 1], background_color=[0, 2, 5, 1], font_size='18sp')
            self.realtableP[0].append(self.btn_background)
            # self.realtableP[0][-1].on_release = self.background_calc(self)
            self.start.append(BoxLayout(orientation="vertical", spacing=3))
            self.start[0].add_widget(self.startlabel[-1])
            self.start[0].add_widget(self.realtableP[0][-1])
            self.start[0].add_widget(self.startlabel2[-1])
            row.append(self.buttons[0])

            self.table.add_widget(self.start[0])
            for cols in range(numco):
                cell = TextInput(text="", halign="center", height=30, multiline=False)
                cell2 = TextInput(text="", halign="center", height=30, multiline=False)
                cell3 = TextInput(text="", halign="center", height=30, multiline=False)
                chB = CheckBox(background_checkbox_normal='CheckBox.png', background_checkbox_down='CheckBox_L.png', background_checkbox_disabled_down='CheckBox_L2.png', size_hint=[1, 0.5])
                lab = Button(text="", halign="center", valign='middle', background_color=[0, 0, 0, 0], color=(1, 1, 1, 1), disabled_color=(1, 1, 1, 1))
                self.realtableP[0].append(cell)
                self.lboundstable[0].append(cell2)
                self.rboundstable[0].append(cell3)
                self.nametable[0].append(lab)
                self.nametable[0][-1].on_press = partial(self.show_par_number, 0, cols)
                self.nametable[0][-1].on_release = partial(self.hide_par_number, 0, cols)
                self.fixtable[0].append(chB)
                mod1[0].append(BoxLayout(orientation="vertical", spacing=3))
                mod2[0].append(BoxLayout(orientation="horizontal", spacing=2))
                mod3[0].append(BoxLayout(orientation="horizontal", spacing=2))
                mod2[0][-1].add_widget(self.nametable[0][-1])
                mod2[0][-1].add_widget(self.fixtable[0][-1])
                mod3[0][-1].add_widget(self.lboundstable[0][-1])
                mod3[0][-1].add_widget(self.rboundstable[0][-1])
                mod1[0][-1].add_widget(mod2[0][-1])
                mod1[0][-1].add_widget(self.realtableP[0][-1])
                mod1[0][-1].add_widget(mod3[0][-1])
                self.table.add_widget(mod1[0][-1])
                # tmp_box_box = FloatLayout()
                # # tmp_box_box.canvas.color = (2, 0, 0, 1)
                # mod1[0].append(tmp_box_box)
                # tmp_box = BoxLayout(orientation="vertical", spacing=3,  size_hint=(0.55, 0.55))
                # mod2[0].append(BoxLayout(orientation="horizontal", spacing=2))
                # mod3[0].append(BoxLayout(orientation="horizontal", spacing=2))
                # mod2[0][-1].add_widget(self.nametable[0][-1])
                # mod2[0][-1].add_widget(self.fixtable[0][-1])
                # mod3[0][-1].add_widget(self.lboundstable[0][-1])
                # mod3[0][-1].add_widget(self.rboundstable[0][-1])
                # tmp_box.add_widget(mod2[0][-1])
                # tmp_box.add_widget(self.realtableP[0][-1])
                # tmp_box.add_widget(mod3[0][-1])
                # mod1[0][-1].add_widget(tmp_box)
                # self.table.add_widget(mod1[0][-1])

            self.params.append(self.realtableP[0])
            self.realtableP[0][1].text = str(10000)
            self.realtableP[0][2].text = str(0)
            self.realtableP[0][3].text = str(0)
            self.realtableP[0][4].text = str(0)
            self.realtableP[0][5].text = str(0)
            self.realtableP[0][6].text = str(0)
            self.realtableP[0][7].text = str(0)
            self.realtableP[0][8].text = str(0)
            self.fixtable[0][1].active = True
            self.fixtable[0][2].active = True
            self.fixtable[0][3].active = True
            self.fixtable[0][4].active = True
            self.fixtable[0][5].active = True
            self.fixtable[0][6].active = True
            self.fixtable[0][7].active = True
            self.nametable[0][0].text = u'Ns' #\u2092'
            self.nametable[0][1].text = u'Os' #\u2092'
            self.nametable[0][2].text = 'c²s'
            self.nametable[0][3].text = 'lins'
            self.nametable[0][4].text = u'Nnr'  # \u2092'
            self.nametable[0][5].text = u'Onr'  # \u2092'
            self.nametable[0][6].text = 'c²nr'
            self.nametable[0][7].text = 'linnr'
            self.lboundstable[0][0].text = '1'
            self.lboundstable[0][4].text = '0'

            ct = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato']
            ct.extend(ct)
            ct.extend(ct)
            ct.extend(ct)  # now it is 80
            self.col_tab = ct

            self.buttons_color = []
            self.dd_color = []
            for rows in range(0, numro):
                self.buttons_color.append(Button())
                self.buttons_color[rows] = Button(text='Color', color=np.array(colors.to_rgba(self.col_tab[rows])), size_hint=(None, None), height=30, font_size='18sp',  background_color=[0, 0, 0, 1])
                self.dd_color.append(DropDown(auto_width=False, width =120))
                for index in ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']:
                    btn = Button(text='%s' % index, size_hint_y=None, height=30, font_size='18sp', color=[0, 0, 0, 1], background_color=(2.5*np.array(colors.to_rgba(index))))
                    btn.bind(on_release=partial(self.select_color, self.buttons_color[rows], np.array(colors.to_rgba(index))))
                    btn.bind(on_release=self.dd_color[rows].dismiss)
                    self.dd_color[rows].add_widget(btn)
                self.buttons_color[rows].bind(on_release=self.dd_color[rows].open)


            self.fix_table_ch = np.array([int(0)]*numro)
            self.fix_memory_table = np.array([[False]*numco]*numro)

            # self.startlabel2[row_number].text = 'fix model'

            for rows in range(1, numro):
                row = []
                self.realtableP.append([])
                self.lboundstable.append([])
                self.rboundstable.append([])
                self.nametable.append([])
                self.fixtable.append([])
                mod1.append([])
                mod2.append([])
                mod3.append([])

                # self.startlabel.append(Label(text="Name  |  fix", halign="center", valign='middle', font_size='18sp'))
                self.startlabel.append(self.buttons_color[rows])
                # self.startlabel2.append(Label(text="boundaries", halign="center", valign='middle', font_size='18sp'))
                self.startlabel2.append(Button(text="fix model", halign="center", valign='middle', font_size='18sp', background_color=[0, 0, 0, 1], color='cyan'))
                self.startlabel2[-1].on_press = partial(self.fix_all, rows)

                # tmp_tmp_box = BoxLayout(orientation="vertical", spacing=1)
                # with tmp_tmp_box.canvas.before:
                #     Color(0, 1, 1, 1)
                #     self.rect = Rectangle(size=tmp_tmp_box.size,
                #                           pos=tmp_tmp_box.pos)
                # tmp_box = BoxLayout(orientation="vertical", spacing=1, size_hint=[0.5, 0.5])
                # with tmp_box.canvas.before:
                #     Color(0, 0, 0, 0.5)
                #     self.rect2 = Rectangle(size=tmp_box.size,
                #                           pos=tmp_box.pos)
                # tmp2_box = BoxLayout(orientation="vertical", spacing=3, size_hint=[None, None])
                # tmp2_box.add_widget(self.startlabel[-1])
                # tmp2_box.add_widget(self.buttons[rows])
                # tmp2_box.add_widget(self.startlabel2[-1])
                # tmp_box.add_widget(tmp2_box)
                # tmp_tmp_box.add_widget(tmp_box)
                # self.start.append(tmp_tmp_box)
                # self.rect.pos = tmp_tmp_box.pos
                # self.rect2.pos = tmp_box.pos
                # self.rect.size = tmp_tmp_box.size
                # self.rect2.size = tmp_box.size


                self.start.append(BoxLayout(orientation="vertical", spacing=3))
                self.start[rows].add_widget(self.startlabel[-1])
                self.start[rows].add_widget(self.buttons[rows])
                self.start[rows].add_widget(self.startlabel2[-1])

                row.append(self.buttons[rows])
                self.realtableP[rows].append(self.buttons[rows])
                self.table.add_widget(self.start[rows])
                for cols in range(numco):
                    cell = TextInput(text="", halign="center", height=30, multiline=False, font_size='18sp')
                    cell2 = TextInput(text="", halign="center", height=30, multiline=False)
                    cell3 = TextInput(text="", halign="center", height=30, multiline=False)
                    chB = CheckBox(background_checkbox_normal='CheckBox.png', background_checkbox_down='CheckBox_L.png', background_checkbox_disabled_down='CheckBox_L2.png', size_hint=[1, 0.5])
                    lab = Button(text="", halign="center", valign='middle', background_color=[0, 0, 0, 0], color=(1, 1, 1, 1), disabled_color=(1, 1, 1, 1), font_size='16sp')
                    self.realtableP[rows].append(cell)
                    self.lboundstable[rows].append(cell2)
                    self.rboundstable[rows].append(cell3)
                    self.nametable[rows].append(lab)
                    # self.nametable[rows][-1].bind(on_press=self.show_par_number(rows, cols))
                    self.nametable[rows][-1].on_press = partial(self.show_par_number, rows, cols)
                    self.nametable[rows][-1].on_release = partial(self.hide_par_number, rows, cols)
                    self.fixtable[rows].append(chB)
                    mod1[rows].append(BoxLayout(orientation="vertical", spacing=3))
                    mod2[rows].append(BoxLayout(orientation="horizontal", spacing=2)) # spacing=2
                    mod3[rows].append(BoxLayout(orientation="horizontal", spacing=2))
                    self.nametable[rows][-1].markup = True
                    mod2[rows][-1].add_widget(self.nametable[rows][-1])
                    mod2[rows][-1].add_widget(self.fixtable[rows][-1])
                    mod3[rows][-1].add_widget(self.lboundstable[rows][-1])
                    mod3[rows][-1].add_widget(self.rboundstable[rows][-1])
                    mod1[rows][-1].add_widget(mod2[rows][-1])
                    mod1[rows][-1].add_widget(self.realtableP[rows][-1])
                    mod1[rows][-1].add_widget(mod3[rows][-1])
                    self.table.add_widget(mod1[rows][-1])
                self.params.append(self.realtableP[rows])

            # , smooth_scroll_end = 3.3
            self.scrollbox = ScrollView(size_hint=(1, 1), size=(800, 600),viewport_size = (0.1, 0.1),  scroll_wheel_distance = 100,
                                   do_scroll_y = True, do_scroll_x = True, always_overscroll=False,
                                   bar_width = 15, bar_pos_y = 'right', scroll_type = ['bars'],
                                   bar_color = [0.5, 0, 1, 0.9], bar_inactive_color = [0.5, 0, 1, 0.3]) #, effect_cls = "ScrollEffect")
            self.scrollbox.add_widget(self.table)
            self.left.add_widget(self.scrollbox)

            # Add empty widget as spacing
            # left.add_widget(Widget())

            btm = BoxLayout(orientation="horizontal", spacing=3, size_hint=[1, 0.2])

            self.play_btn = Button(text="Fit\n(F5)", halign="center", color = [0, 0, 0, 1], size_hint=[0.3, 1],
                                   background_color=[0, 3, 0, 1], font_size='21sp',
                                   background_disabled_down='', background_disabled_normal='',disabled_color=[1, 1, 1, 1])
            self.play_btn.bind(on_release = self.play_pressed) # on_press
            # self.play_btn.bind(on_touch_down = self.online_play)
            # self.mouse = 'left'
            # self.mouse.bind(on_touch_down = self.online_play)

            btm.add_widget(self.play_btn)

            blt2 = BoxLayout(orientation="vertical", spacing=2)
            blt3 = BoxLayout(orientation="horizontal", spacing=4)
            self.chB1 = CheckBox(color = [2, 1, 0, 2])
            # chB2 = CheckBox(background_checkbox_normal='CheckBox.png')
            self.chB3 = CheckBox(color = [2, 1, 0, 2])
            self.chB4 = CheckBox(color=[2, 1, 0, 2])

            self.fitmodel1 = Label(text="MS", halign="center", valign='middle', font_size='18sp')
            # fitmodel2 = Label(text="Rough", halign="center", valign='middle', font_size='18sp')
            self.fitmodel3 = Label(text="SMS", halign="center", valign='middle', font_size='18sp')
            self.fitmodel4 = Label(text="APS", halign="center", valign='middle', font_size='18sp')

            self.fitway = [self.chB1, self.chB3, self.chB4]
            self.fitway[0].on_press = partial(self.chB_a, 0)
            # self.fitway[1].on_press = partial(self.chB_a, 1)
            self.fitway[1].on_press = partial(self.chB_a, 1)
            self.fitway[2].on_press = partial(self.chB_a, 2)
            self.fitway[1].active = True

            self.online_fit = False
            self.time_interval = 15

            bltFIT = BoxLayout(orientation="vertical", spacing=2)
            bltFITway = BoxLayout(orientation="horizontal", spacing=3)
            bltFITpar = BoxLayout(orientation="horizontal", spacing=4)

            blt4 = BoxLayout(orientation="vertical", spacing=2)
            blt5 = BoxLayout(orientation="horizontal", spacing=2)
            bltAPS = BoxLayout(orientation="horizontal", spacing=2)
            blt42 = BoxLayout(orientation="vertical", spacing=2)
            blt52 = BoxLayout(orientation="horizontal", spacing=2)
            blt53 = BoxLayout(orientation="horizontal", spacing=2)
            blt54 = BoxLayout(orientation="horizontal", spacing=2)
            self.JN0 = TextInput(text="32", halign="center")
            self.memoryJN = [32, 0]
            GCMS = np.genfromtxt(str(self.dir_path) + str('\\\\GCMS.txt')*(platform.system() == 'Windows') + str('/GCMS.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
            self.L0 = TextInput(text=str(GCMS), halign="center", background_color=[1, 1, 1, 0.5])

            bltAPS.add_widget(self.fitmodel4)
            bltAPS.add_widget(self.chB4)
            blt52.add_widget(self.fitmodel1)
            blt52.add_widget(self.chB1)
            # blt42.add_widget(blt52)
            blt53.add_widget(Label(text="G", halign="center", valign='middle', font_size='18sp'))
            blt53.add_widget(self.L0)
            # blt42.add_widget(blt53)
            blt5.add_widget(self.fitmodel3)
            blt5.add_widget(self.chB3)
            # blt4.add_widget(blt5)
            blt54.add_widget(Label(text="Integral", halign="center", valign='middle', font_size='18sp'))
            blt54.add_widget(self.JN0)
            # blt4.add_widget(blt54)
            # blt3.add_widget(blt42)

            bltFITway.add_widget(blt52)
            bltFITway.add_widget(blt5)
            bltFITway.add_widget(bltAPS)
            bltFITpar.add_widget(blt53)
            bltFITpar.add_widget(blt54)

            bltFIT.add_widget(bltFITway)
            bltFIT.add_widget(bltFITpar)

            # blt3.add_widget(blt4)
            blt3.add_widget(bltFIT)

            blt6 = BoxLayout(orientation="horizontal", spacing=2)
            self.process_path = TextInput(text="['Calibration.dat']", halign="center") #FileChooserListView(size_hint_y=1)
            self.btnchoose = Button(text="Choose\nspectrum", halign='center', size_hint=[0.5, 1], font_size='18sp')
            self.btnchoose.bind(on_release=self.choose_file)

            self.btn_background.bind(on_release=self.background_calc)

            blt6.add_widget(self.btnchoose)
            blt6.add_widget(self.process_path)
            blt2.add_widget(blt6)
            blt2.add_widget(blt3)
            btm.add_widget(blt2)

            btm2 = BoxLayout(orientation="vertical", spacing=2)

            self.show_btn = Button(text="Show spectrum", font_size='18sp', background_color=[0.5, 0.5, 0.5, 1], color = [1, 1, 1, 1],
                                   background_disabled_down='', background_disabled_normal='', disabled_color=[1, 1, 1, 1])
            self.showM_btn = Button(text="Show model", font_size='18sp', background_color=[0.5, 0.5, 0.5, 1], color = [1, 1, 1, 1],
                                    background_disabled_down='', background_disabled_normal='',disabled_color=[1, 1, 1, 1])
            self.show_btn.bind(on_release=self.show_pressed)
            self.showM_btn.bind(on_release=self.showM_pressed)

            btm3 = BoxLayout(orientation="horizontal", spacing=3)

            self.INS_btn = Button(text="Instrumental\nfunction", halign='center', font_size='16sp', background_color=[0.5, 0.5, 0.5, 1], color = [1, 1, 1, 1],
                                  disabled_color=[0, 0, 0, 1]) # background_disabled_down='', background_disabled_normal='',
            # self.INS_btn.bind(on_release=self.instrumental_pressed)


            self.INS_btndd = DropDown(auto_width=False, width=120)
            # for index in ['Find\nINS func\nESRF', 'Find\nINS func\nmodel', 'Refine\nINS func\nmodel']:
            btn = Button(text='Find\nInstr. func.\n single line', size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 0, 0))
            btn.bind(on_release=self.INS_btndd.dismiss)
            self.INS_btndd.add_widget(btn)
            btn = Button(text='Find\nInstr. func.\n pure a-Fe', size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 0, 2))
            btn.bind(on_release=self.INS_btndd.dismiss)
            self.INS_btndd.add_widget(btn)
            btn = Button(text='Find\nInstr. func.\nmodel', size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 0, 1))
            btn.bind(on_release=self.INS_btndd.dismiss)
            self.INS_btndd.add_widget(btn)


            self.INS_btn.bind(on_release=self.INS_btndd.open)

            ###
            self.INS_btn2 = Button(text="Refine\nInstr. func.\nESRF", halign='center', font_size='15sp', background_color=[0.5, 0.5, 0.5, 1], color = [1, 1, 1, 1],
                                   disabled_color=[0, 0, 0, 1]) # background_disabled_down='', background_disabled_normal='',

            self.INS_btndd2 = DropDown(auto_width=False, width=120)
            # for index in ['Find\nINS func\nESRF', 'Find\nINS func\nmodel', 'Refine\nINS func\nmodel']:
            btn = Button(text="Refine\nInstr. func.\n single line", size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 1, 0))
            btn.bind(on_release=self.INS_btndd2.dismiss)
            self.INS_btndd2.add_widget(btn)
            btn = Button(text="Refine\nInstr. func.\n pure a-Fe", size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 1, 2))
            btn.bind(on_release=self.INS_btndd2.dismiss)
            self.INS_btndd2.add_widget(btn)
            btn = Button(text='Refine\nInstr. func.\nmodel', size_hint_y=None, height=70, font_size='16sp',
                         background_color=[0.5, 0.5, 2, 1], halign='center')
            btn.bind(on_release=partial(self.instrumental_pressed, 1, 1))
            btn.bind(on_release=self.INS_btndd2.dismiss)
            self.INS_btndd2.add_widget(btn)

            self.INS_btn2.bind(on_release=self.INS_btndd2.open)

            # self.INS_btn2 = Button(text="Refine\nInstr. func.\nESRF", halign='center', font_size='15sp', background_color=[0.5, 0.5, 0.5, 1], color = [1, 1, 1, 1],
            #                        disabled_color=[0, 0, 0, 1]) # background_disabled_down='', background_disabled_normal='',
            # self.INS_btn2.bind(on_release=partial(self.instrumental_pressed, 1, 0))

            btm4 = BoxLayout(orientation="vertical", spacing=2)
            self.INS_number = TextInput(text="3", halign="center")
            INS_label = Label(text="№ of lines", halign="center", valign='middle', font_size='18sp')
            btm4.add_widget(INS_label)
            btm4.add_widget(self.INS_number)

            btm3.add_widget(self.INS_btn)
            btm3.add_widget(btm4)
            btm3.add_widget(self.INS_btn2)

            btm5 = BoxLayout(orientation="horizontal", spacing=2)
            btm5.add_widget(self.show_btn)
            btm5.add_widget(self.showM_btn)

            btm2.add_widget(btm5)
            btm2.add_widget(btm3)

            btm.add_widget(btm2)

            btm2 = BoxLayout(orientation="horizontal", spacing=1, size_hint=[1, 0.1])
            self.save_btn = Button(text="Save\nresult", size_hint=[0.3, 1], font_size='18sp', halign='center')
            self.saveas_btn = Button(text="Save\nresult as", size_hint=[0.3, 1], font_size='18sp', halign='center')
            self.save_btn.bind(on_release=self.save_pressed)
            self.saveas_btn.bind(on_release=self.save_as_pressed)

            btm2.add_widget(self.save_btn)
            btm2.add_widget(self.saveas_btn)

            self.save_path = TextInput(text="result", halign="center")
            btm2.add_widget(self.save_path)

            self.savemod_btn = Button(text="Save\nmodel", size_hint=[0.3, 1], font_size='18sp', halign='center')
            self.savemodas_btn = Button(text="Save\nmodel as", size_hint=[0.3, 1], font_size='18sp', halign='center')
            self.savemod_btn.bind(on_release=self.savemod_pressed)
            self.savemodas_btn.bind(on_release=self.savemod_as_pressed)

            btm2.add_widget(self.savemod_btn)
            btm2.add_widget(self.savemodas_btn)

            seq_fit = BoxLayout(orientation="horizontal", spacing=6, size_hint=[0.95, 0.1])
            chB4 = CheckBox(size_hint=[0.3, 1], color = [2, 1, 0, 2])
            chB5 = CheckBox(size_hint=[0.3, 1], color = [2, 1, 0, 2])
            self.seq_fit = [chB4, chB5]
            self.seq_fit[0].on_press = partial(self.chB_b, 0)
            self.seq_fit[1].on_press = partial(self.chB_b, 1)
            self.seq_fit[0].active = True
            # desc = Label(text="if few spectra are selected\n how sequence should be fitted:", halign="center", valign='middle', font_size='14sp')

            self.descdd = []
            self.desc = Button(text='Change\nspectrum(a)', size_hint=(1, 1), font_size='18sp',  background_color=[1, 1, 1, 1], halign='center')
            self.descdd = DropDown(auto_width=False, width =120)
            for index in ['Sum all\nspectra', 'Substract\nmodel from\nspectrum', 'Half points']:
                btn = Button(text='%s' % index, size_hint_y=None, height=70, font_size='18sp',  background_color=[0.5, 0.5, 2, 1], halign='center')
                btn.bind(on_release=partial(self.spc_changes, btn.text))
                btn.bind(on_release=self.descdd.dismiss)
                self.descdd.add_widget(btn)
            self.desc.bind(on_release=self.descdd.open)

            self.btnchoosefolder = Button(text="Choose\nworkfolder", halign='center', size_hint=[0.5, 1], font_size='16sp')
            self.btnchoosefolder.bind(on_release=self.choose_workfolder)

            # self.desc = Button(text='Sum\nspectra', size_hint=[0.5, 1], font_size='18sp', halign='center')
            # self.desc.bind(on_press=self.sum_spectra)
            # self.desc = Button(text='Change\nspectrum(a)', size_hint=[0.5, 1], font_size='18sp', halign='center')
            # self.desc.bind(on_press=self.sum_spectra)


            SF1 = Label(text="take always initial guess\n for the sequence of spectra", halign="center", valign='middle', font_size='18sp')
            SF2 = Label(text="take result as initial guess\n for the sequence of spectra", halign="center", valign='middle', font_size='18sp')
            seq_fit.add_widget(self.desc)
            seq_fit.add_widget(self.btnchoosefolder)
            seq_fit.add_widget(chB4)
            seq_fit.add_widget(SF1)
            seq_fit.add_widget(chB5)
            seq_fit.add_widget(SF2)


            self.left.add_widget(btm)
            self.left.add_widget(seq_fit)
            self.left.add_widget(btm2)

            main.add_widget(self.left)
            main.add_widget(right)

            self.points_match = True
            self.show_pressed(self)

            # threading.Thread(target=IMG_r, args=(q,)).start()
            # try:
            #     image.source = q.get(timeout=1)
            #     print('got reqeust')
            # except queue.Empty:
            #     print("timed out waiting for a request")

            #for replot picture in result:
            self.X_axis = []
            self.Y_axis = []
            self.Z_order = []
            self.Color_order = []
            self.SPC_plot = []
            self.SPC_numb = []
            self.Model_full_plot = []
            self.Baseline_plot = []
            self.Integral_line_plot = []


            global fp, model
            fp = self.dir_path
            model = []

            if platform.system() == 'Windows':
                licensepath = str(self.dir_path) + str('\\\\COPYING.txt')
            else:
                licensepath = str(self.dir_path) + str('/COPYING.txt')

            file = os.path.abspath(licensepath)
            A_list = []
            with open(file, 'r') as catalog:
                lines = (line.rstrip() for line in catalog)
                lines = (line for line in lines if line)  # skipping white lines
                for line in lines:
                    column = line.split()
                    x = str(column[0:30])
                    A_list.append(x)
            A = np.array(A_list)

            #if str(A[495][9:19]) == 'Limitation' and A[495][29:38] == 'Liability':
            if str(A[4][78:88]) == 'limitation' and A[14][2:11] == 'LIABILITY':
                Clock.schedule_interval(partial(ImgUpdate, self, self.image), 0.25)
            else:
                self.on_quit()

            return main

        def select_color(self, drop_button, color, _instance):
            drop_button.color = color

        def select(self, drop_button, text, numb, _instance):
            if text != 'Delete' and text != 'Insert' and drop_button.text != 'None':
                self.select(drop_button, 'Delete', numb, _instance)
                self.select(drop_button, 'Insert', numb, _instance)


            if text != 'Insert' and text != 'None':
                for i in range(0, len(self.lboundstable[numb])):
                    self.lboundstable[numb][i].text = ''
                    self.rboundstable[numb][i].text = ''
                    self.nametable[numb][i].text = ''
                    self.fixtable[numb][i].active = False
                for i in range(1, len(self.lboundstable[numb])+1):
                    self.realtableP[numb][i].text = ''

            if self.realtableP[numb][0].text == 'Insert':
                if text != 'Insert':
                    Model_del_name = text

                    for j in range(1, len(self.realtableP[numb])):
                        self.lboundstable[numb][j - 1].background_color = (0, 0, 0, 0)
                        self.rboundstable[numb][j - 1].background_color = (0, 0, 0, 0)
                        self.realtableP[numb][j].background_color = (0, 0, 0, 0)

                    Mo = mod_len_def(Model_del_name)

                    Parameter_counter = NBA
                    for k in range(1, numb):
                        Parameter_counter += mod_len_def(self.realtableP[k][0].text)
                    for k in range(0, len(self.realtableP)-1):
                        for kk in range(1, len(self.realtableP[k])):
                            kkk = 0
                            for kkkt in range(2, (len(str(self.realtableP[k][kk].text))+1)):
                                if str(self.realtableP[k][kk].text)[-kkkt + kkk] == '=' and str(self.realtableP[k][kk].text)[-kkkt+1+kkk] == '[':
                                    for kkkk in range(-kkkt+2+kkk, 0):
                                        if str(self.realtableP[k][kk].text)[kkkk] == ',':
                                            try:
                                                old_number = int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk])
                                            except:
                                                old_number = -1
                                            if old_number >= (Parameter_counter):
                                                new_number = str(int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk]) + Mo)
                                                self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                                kkk += len(str(old_number)) - len(str(new_number))
                                            break
                                if str(self.realtableP[k][kk].text)[-kkkt + kkk] == 'p' and str(self.realtableP[k][kk].text)[-kkkt+1+kkk] == '[':
                                    for kkkk in range(-kkkt+2+kkk, 0):
                                        if str(self.realtableP[k][kk].text)[kkkk] == ']':
                                            try:
                                                old_number = int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk])
                                            except:
                                                old_number = -1
                                            if old_number >= (Parameter_counter):
                                                new_number = str(int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk]) + Mo)
                                                self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                                kkk += len(str(old_number)) - len(str(new_number))
                                            break

            if text == 'Nbaseline':
                self.realtableP[numb][0].text = 'Nbaseline'
                self.realtableP[numb][1].text = str(10000)
                self.realtableP[numb][2].text = str(0)
                self.realtableP[numb][3].text = str(0)
                self.realtableP[numb][4].text = str(0)
                self.realtableP[numb][5].text = str(0)
                self.realtableP[numb][6].text = str(0)
                self.realtableP[numb][7].text = str(0)
                self.realtableP[numb][8].text = str(0)

                if self.realtableP[0][5].text != '0' and self.realtableP[0][5].text != '0.0':#
                    try:
                        p_counter = NBA
                        for i in range(0, numb):
                            p_counter += mod_len_def(self.realtableP[i][0].text)
                        global p
                        read_model(self)
                        if p[0] != 0:
                            if self.realtableP[0][5].text[0] == '=':#
                                pNM = (self.realtableP[0][5].text[2:-1]).split(',')#
                                pN, pM = int(pNM[0]), float(pNM[1])
                                NonRes = p[pN] * pM
                                Multipl_nrs_ns = NonRes/p[0]
                            else:
                                Multipl_nrs_ns = float(self.realtableP[0][5].text) / p[0]#
                            self.realtableP[numb][5].text = str('=[') + str(int(p_counter)) + str(',') + str(int(Multipl_nrs_ns*100)/100) + str(']')
                            if self.realtableP[0][5].text[0] == '=' and self.realtableP[0][5].text[2] != '0':
                                self.realtableP[numb][5].text = self.realtableP[0][5].text
                    except:
                        print('there was a try to correct none-resonant part but it failed due to problems in model')
                Nb_counter = 1
                for i in range(0, numb):
                    if self.realtableP[i][0].text == 'Nbaseline':
                        Nb_counter += 1
                try:
                    self.background_calc(self, POS=Nb_counter, numb=numb)
                except:
                    self.realtableP[numb][1].text = str(10000)

                if self.realtableP[numb][5].text[0] == '=' and self.realtableP[0][5].text[0] != '=':
                    self.realtableP[numb][5].text = str(int(int(self.realtableP[numb][1].text)*Multipl_nrs_ns))

                self.fixtable[numb][1].active = True
                self.fixtable[numb][2].active = True
                self.fixtable[numb][4].active = self.fixtable[0][4].active
                self.fixtable[numb][4].active = True
                self.fixtable[numb][5].active = True
                self.fixtable[numb][6].active = True
                self.fixtable[numb][7].active = True
                self.nametable[numb][0].text = u'Ns'  # \u2092'
                self.nametable[numb][1].text = u'Os'  # \u2092'
                self.nametable[numb][2].text = 'c²s'
                self.nametable[numb][3].text = 'lins'
                self.nametable[numb][4].text = u'Nnr'  # \u2092'
                self.nametable[numb][5].text = u'Onr'  # \u2092'
                self.nametable[numb][6].text = 'c²nr'
                self.nametable[numb][7].text = 'linnr'
                self.lboundstable[numb][0].text = '1'
                self.lboundstable[numb][4].text = '0'
                # self.nametable[numb][0].text = u'Ns'  # \u2092'
                # self.nametable[numb][1].text = u'Os'  # \u2092'
                # self.nametable[numb][2].text = 'c²s'
                # self.nametable[numb][3].text = u'Nnr'  # \u2092'
                # self.nametable[numb][4].text = u'Onr'  # \u2092'
                # self.nametable[numb][5].text = 'c²'
                # self.lboundstable[numb][0].text = '1'
                # self.lboundstable[numb][4].text = '0'
            if text == 'Singlet':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'L, mm/s'
                self.nametable[numb][3].text = 'G, mm/s'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][2].text = '0.098'
                self.lboundstable[numb][3].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.098)
                self.realtableP[numb][4].text = str(0.1)
                self.fixtable[numb][2].active = True
            if text == 'Doublet':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'L, mm/s'
                self.nametable[numb][4].text = 'G, mm/s'
                self.nametable[numb][5].text = 'A'
                self.nametable[numb][6].text = 'G2/G1'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][3].text = '0.098'
                self.lboundstable[numb][4].text = '0'
                self.lboundstable[numb][5].text = '0'
                self.rboundstable[numb][5].text = '1'
                self.lboundstable[numb][6].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(1.0)
                self.realtableP[numb][4].text = str(0.098)
                self.realtableP[numb][5].text = str(0.1)
                self.realtableP[numb][6].text = str(0.5)
                self.realtableP[numb][7].text = str(1.0)
                self.fixtable[numb][3].active = True
                self.fixtable[numb][5].active = True
                self.fixtable[numb][6].active = True
            if text == 'Sextet':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'A'
                self.nametable[numb][7].text = 'a+'
                self.nametable[numb][8].text = 'a-'
                self.nametable[numb][9].text = 'GH, T'
                self.nametable[numb][10].text = 'I1/I3'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][6].text = '0'
                self.rboundstable[numb][6].text = '1'
                self.lboundstable[numb][9].text = '0'
                self.lboundstable[numb][10].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(0.5)
                self.realtableP[numb][8].text = str(0.0)
                self.realtableP[numb][9].text = str(0.0)
                self.realtableP[numb][10].text = str(0.0)
                self.realtableP[numb][11].text = str(3.0)
                self.fixtable[numb][4].active = True
                self.fixtable[numb][6].active = True
                self.fixtable[numb][7].active = True
                self.fixtable[numb][8].active = True
                self.fixtable[numb][9].active = True
                self.fixtable[numb][10].active = True
            if text == 'Sextet(rough)':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'a+'
                self.nametable[numb][7].text = 'a-'
                self.nametable[numb][8].text = 'GH, T'
                # SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                # str('I16/I34').translate(SUB) # ---- DOES NOT WORK ----
                # self.nametable[numb][9].text = str(r"$\frac{16}{34}$") # ---- DOES NOT WORK ----
                self.nametable[numb][9].text = 'I[size=12]16/34[/size]'
                self.nametable[numb][10].text = 'I[size=12]25/34[/size]'
                self.nametable[numb][11].text = 'I[size=12]1/6[/size]'
                self.nametable[numb][12].text = 'I[size=12]2/5[/size]'
                self.nametable[numb][13].text = 'I[size=12]3/4[/size]' #[sub]3[/sub]/I[sub]4[/sub] # --- too small ---
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][8].text = '0'
                self.lboundstable[numb][9].text = '0'
                self.lboundstable[numb][10].text = '0'
                self.lboundstable[numb][11].text = '0'
                self.lboundstable[numb][12].text = '0'
                self.lboundstable[numb][13].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(0.0)
                self.realtableP[numb][8].text = str(0.0)
                self.realtableP[numb][9].text = str(0.0)
                self.realtableP[numb][10].text = str(3.0)
                self.realtableP[numb][11].text = str(2.0)
                self.realtableP[numb][12].text = str(1.0)
                self.realtableP[numb][13].text = str(1.0)
                self.realtableP[numb][14].text = str(1.0)
                self.fixtable[numb][4].active = True
                self.fixtable[numb][6].active = True
                self.fixtable[numb][7].active = True
                self.fixtable[numb][8].active = True
                self.fixtable[numb][9].active = True
                self.fixtable[numb][10].active = True
                self.fixtable[numb][11].active = True
                self.fixtable[numb][12].active = True
                self.fixtable[numb][13].active = True
            if text == 'MDGD':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'GH, T'
                self.nametable[numb][7].text = 'Dδε'
                self.nametable[numb][8].text = 'DδH'
                self.nametable[numb][9].text = 'DεH'
                self.nametable[numb][10].text = 'A'
                self.nametable[numb][11].text = 'a+'
                self.nametable[numb][12].text = 'a-'
                self.nametable[numb][13].text = 'I1/I3'
                # SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                # str('I16/I34').translate(SUB) # ---- DOES NOT WORK ----
                # self.nametable[numb][9].text = str(r"$\frac{16}{34}$") # ---- DOES NOT WORK ----
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][6].text = '0'
                self.lboundstable[numb][7].text = '-1'
                self.lboundstable[numb][8].text = '-1'
                self.lboundstable[numb][9].text = '-1'
                self.rboundstable[numb][7].text = '1'
                self.rboundstable[numb][8].text = '1'
                self.rboundstable[numb][9].text = '1'
                self.lboundstable[numb][10].text = '0'
                self.rboundstable[numb][10].text = '1'
                self.lboundstable[numb][13].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(0.0)
                self.realtableP[numb][8].text = str(0.0)
                self.realtableP[numb][9].text = str(0.0)
                self.realtableP[numb][10].text = str(0.0)
                self.realtableP[numb][11].text = str(0.5)
                self.realtableP[numb][12].text = str(0.0)
                self.realtableP[numb][13].text = str(0.0)
                self.realtableP[numb][14].text = str(3.0)
                self.fixtable[numb][4].active = True
                self.fixtable[numb][6].active = True
                self.fixtable[numb][7].active = True
                self.fixtable[numb][8].active = True
                self.fixtable[numb][9].active = True
                self.fixtable[numb][10].active = True
                self.fixtable[numb][11].active = True
                self.fixtable[numb][12].active = True
                self.fixtable[numb][13].active = True
            if text == 'Hamilton_mc':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'Q, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'η' #'A'
                self.nametable[numb][7].text = 'θH, °' #'a+'
                self.nametable[numb][8].text = 'φH, °' #'a-'
                self.nametable[numb][9].text = 'θ, °' #'GH, T'
                self.nametable[numb][10].text = 'φ, °' #'I1/I3'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][6].text = '-1'
                self.rboundstable[numb][6].text = '1'
                self.lboundstable[numb][7].text = '-180'
                self.rboundstable[numb][7].text = '180'
                self.lboundstable[numb][8].text = '-360'
                self.rboundstable[numb][8].text = '360'
                self.lboundstable[numb][9].text = '-180'
                self.rboundstable[numb][9].text = '180'
                self.lboundstable[numb][10].text = '-360'
                self.rboundstable[numb][10].text = '360'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(0.0)
                self.realtableP[numb][8].text = str(0.0)
                self.realtableP[numb][9].text = str(0.0)
                self.realtableP[numb][10].text = str(0.0)
                self.realtableP[numb][11].text = str(0.0)
                self.fixtable[numb][4].active = True
            if text == 'Hamilton_pc':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'Q, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'η' #'A'
                self.nametable[numb][7].text = 'θH, °' #'a+'
                self.nametable[numb][8].text = 'φH, °' #'a-'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][6].text = '-1'
                self.rboundstable[numb][6].text = '1'
                self.lboundstable[numb][7].text = '-180'
                self.rboundstable[numb][7].text = '180'
                self.lboundstable[numb][8].text = '-360'
                self.rboundstable[numb][8].text = '360'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(0.0)
                self.realtableP[numb][8].text = str(0.0)
                self.realtableP[numb][9].text = str(0.0)
                self.fixtable[numb][4].active = True
            if text == 'Relax_MS':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'H, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'A'
                self.nametable[numb][6].text = 'R'
                self.nametable[numb][7].text = 'alfa'
                self.nametable[numb][8].text = 'S'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.rboundstable[numb][5].text = '1'
                self.lboundstable[numb][6].text = '0'
                self.lboundstable[numb][7].text = '0'
                self.rboundstable[numb][7].text = '100'
                self.lboundstable[numb][8].text = '0.5'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.1)
                self.realtableP[numb][6].text = str(0.5)
                self.realtableP[numb][7].text = str(0.5)
                self.realtableP[numb][8].text = str(1.0)
                self.realtableP[numb][9].text = str(101)
                self.fixtable[numb][5].active = True
                self.fixtable[numb][8].active = True
            if text == 'Relax_2S':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ1, mm/s'
                self.nametable[numb][2].text = 'ε1, mm/s'
                self.nametable[numb][3].text = 'H1, T'
                self.nametable[numb][4].text = 'δ2, mm/s'
                self.nametable[numb][5].text = 'ε2, mm/s'
                self.nametable[numb][6].text = 'H2, T'
                self.nametable[numb][7].text = 'L, mm/s'
                # self.nametable[numb][8].text = 'G, mm/s'
                self.nametable[numb][8].text = 'A'
                self.nametable[numb][9].text = 'Ω12'
                self.nametable[numb][10].text = 'P1/P2'
                self.lboundstable[numb][7].text = '0.098'
                # self.lboundstable[numb][8].text = '0'
                self.lboundstable[numb][8].text = '0'
                self.rboundstable[numb][8].text = '1'
                self.lboundstable[numb][9].text = '0'
                self.lboundstable[numb][10].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(33.0)
                self.realtableP[numb][5].text = str(0.0)
                self.realtableP[numb][6].text = str(0.0)
                self.realtableP[numb][7].text = str(-33.0)
                self.realtableP[numb][8].text = str(0.1)
                # self.realtableP[numb][9].text = str(0.1)
                self.realtableP[numb][9].text = str(0.5)
                self.realtableP[numb][10].text = str(0.3)
                self.realtableP[numb][11].text = str(1)
                # self.fixtable[numb][7].active = True
                self.fixtable[numb][8].active = True
                self.fixtable[numb][10].active = True
            if text == 'ASM':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'εm, mm/s'
                self.nametable[numb][3].text = 'εl, mm/s'
                self.nametable[numb][4].text = 'His, T'
                self.nametable[numb][5].text = 'Han, T'
                self.nametable[numb][6].text = 'L, mm/s'
                self.nametable[numb][7].text = 'G, mm/s'
                self.nametable[numb][8].text = 'm'
                self.nametable[numb][9].text = 'A'
                self.nametable[numb][10].text = 'Num'
                self.nametable[numb][11].text = 'I13'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][6].text = '0.098'
                self.lboundstable[numb][7].text = '0'
                self.lboundstable[numb][8].text = '-1'
                self.rboundstable[numb][8].text = '1'
                self.lboundstable[numb][9].text = '0'
                self.rboundstable[numb][9].text = '1'
                self.lboundstable[numb][10].text = '7'
                self.lboundstable[numb][11].text = '0'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(0.0)
                self.realtableP[numb][5].text = str(30.0)
                self.realtableP[numb][6].text = str(5.0)
                self.realtableP[numb][7].text = str(0.098)
                self.realtableP[numb][8].text = str(0.1)
                self.realtableP[numb][9].text = str(0.1)
                self.realtableP[numb][10].text = str(0.5)
                self.realtableP[numb][11].text = str(25)
                self.realtableP[numb][12].text = str(3.0)
                self.fixtable[numb][6].active = True
                self.fixtable[numb][9].active = True
                self.fixtable[numb][10].active = True
                self.fixtable[numb][11].active = True
                self.lboundstable[numb][10].disabled = True
                self.rboundstable[numb][10].disabled = True
            if text == 'Be':
                self.select(drop_button, 'Doublet', numb, _instance)
                try:
                    Be_param = np.genfromtxt(str(self.dir_path) + str('\\\\Be.txt')*(platform.system() == 'Windows') + str('/Be.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
                    self.realtableP[numb][1].text = str(Be_param[0])
                    self.realtableP[numb][2].text = str(Be_param[1])
                    self.realtableP[numb][3].text = str(Be_param[2])
                    self.realtableP[numb][4].text = str(Be_param[3])
                    self.realtableP[numb][5].text = str(Be_param[4])
                    self.realtableP[numb][6].text = str(Be_param[5])
                    self.realtableP[numb][7].text = str(Be_param[6])
                    if self.log.text == 'Default values used. Could not open/read file. Please use TAB delimiter.':
                        self.log.text = 'Now Be.txt is correct.'
                        self.log.background_color = [0, 2, 0, 2]
                except:
                    self.realtableP[numb][1].text = str(0.048)
                    self.realtableP[numb][2].text = str(0.103)
                    self.realtableP[numb][3].text = str(-0.259)
                    self.realtableP[numb][4].text = str(0.098)
                    self.realtableP[numb][5].text = str(0.105)
                    self.realtableP[numb][6].text = str(0.265)
                    self.realtableP[numb][7].text = str(1.0)
                    self.log.text = 'Default values used. Could not open/read file. Please use TAB delimiter.'
                    self.log.background_color = [2, 2, 0, 2]

                self.fixtable[numb][0].active = True
                self.fixtable[numb][1].active = True
                self.fixtable[numb][2].active = True
                self.fixtable[numb][3].active = True
                self.fixtable[numb][4].active = True
                self.fixtable[numb][5].active = True
                self.fixtable[numb][6].active = True
                text = 'Doublet'
            if text == 'KB_nano':
                self.select(drop_button, 'Doublet', numb, _instance)
                try:
                    KB_param = np.genfromtxt(str(self.dir_path) + str('\\\\KB.txt')*(platform.system() == 'Windows') + str('/KB.txt')*(platform.system() != 'Windows'), delimiter='\t', skip_footer=0)
                    self.realtableP[numb][1].text = str(KB_param[0])
                    self.realtableP[numb][2].text = str(KB_param[1])
                    self.realtableP[numb][3].text = str(KB_param[2])
                    self.realtableP[numb][4].text = str(KB_param[3])
                    self.realtableP[numb][5].text = str(KB_param[4])
                    self.realtableP[numb][6].text = str(KB_param[5])
                    self.realtableP[numb][7].text = str(KB_param[6])
                    if self.log.text == 'Default values used. Could not open/read file. Please use TAB delimiter.':
                        self.log.text = 'Now KB.txt is correct.'
                        self.log.background_color = [0, 2, 0, 2]
                except:
                    self.realtableP[numb][1].text = str(0.065)
                    self.realtableP[numb][2].text = str(0.234)
                    self.realtableP[numb][3].text = str(0.37)
                    self.realtableP[numb][4].text = str(0.098)
                    self.realtableP[numb][5].text = str(0.373)
                    self.realtableP[numb][6].text = str(0.5)
                    self.realtableP[numb][7].text = str(1.0)
                    self.log.text = 'Default values used. Could not open/read file. Please use TAB delimiter.'
                    self.log.background_color = [2, 2, 0, 2]

                self.fixtable[numb][0].active = True
                self.fixtable[numb][1].active = True
                self.fixtable[numb][2].active = True
                self.fixtable[numb][3].active = True
                self.fixtable[numb][4].active = True
                self.fixtable[numb][5].active = True
                self.fixtable[numb][6].active = True
                text = 'Doublet'
            if text == 'Variables':
                for k in range(1, len(self.realtableP[numb])):
                    self.nametable[numb][k-1].text = str('V') + str(k)
                    self.realtableP[numb][k].text = str(0)
                    self.fixtable[numb][k-1].active = True

            for j in range(1, len(self.realtableP[numb])):
                self.realtableP[numb][j].size_hint = (1, 1)
                self.realtableP[numb][j].halign = "center"
                self.realtableP[numb][j].pos_hint = {'left': 1}
                self.fixtable[numb][j - 1].color = (1, 1, 1, 1)
                if text != 'None':
                    self.fixtable[numb][j - 1].disabled = False
                self.nametable[numb][j - 1].text_size = (None, None)
                self.nametable[numb][j - 1].halign = "center"
                self.nametable[numb][j - 1].disabled = False
                self.lboundstable[numb][j - 1].disabled = False
                self.rboundstable[numb][j - 1].disabled = False
                self.realtableP[numb][j].disabled = False

            if text == 'Relax_MS':
                self.fixtable[numb][8].disabled = True
            if text == 'ASM':
                self.fixtable[numb][10].disabled = True
            if text == 'Expression':
                self.realtableP[numb][1].size_hint = (len(self.realtableP[numb]) - 1, 1)
                self.realtableP[numb][1].halign = "left"
                for j in range(2, len(self.realtableP[numb])):
                    self.realtableP[numb][j].pos_hint = {'right': len(self.realtableP[numb]) - j}
                    self.fixtable[numb][j - 1].color = (0, 0, 0, 0)
                    self.nametable[numb][j - 1].disabled = True
                self.nametable[numb][0].text = 'Expression'
                self.fixtable[numb][0].color = (0, 0, 0, 0)
                # self.nametable[numb][0].disabled = True
                self.nametable[numb][0].text_size[0] = self.nametable[numb][0].size[0] * 2.8
                self.realtableP[numb][1].text = str('p[0]')
                self.fixtable[numb][0].active = True
                self.fixtable[numb][0].disabled = True
                self.lboundstable[numb][0].disabled = True
                self.rboundstable[numb][0].disabled = True
                self.nametable[numb][0].halign = "right"
            if text == 'Average_H':
                self.nametable[numb][0].text = 'T'
                self.nametable[numb][1].text = 'δ, mm/s'
                self.nametable[numb][2].text = 'ε, mm/s'
                self.nametable[numb][3].text = 'Hin, T'
                self.nametable[numb][4].text = 'L, mm/s'
                self.nametable[numb][5].text = 'G, mm/s'
                self.nametable[numb][6].text = 'Hex, T'
                self.nametable[numb][7].text = 'K'
                self.nametable[numb][8].text = 'J'
                self.nametable[numb][9].text = 'θ, °'
                self.nametable[numb][10].text = 'N'
                self.lboundstable[numb][0].text = '0'
                self.lboundstable[numb][4].text = '0.098'
                self.lboundstable[numb][5].text = '0'
                self.lboundstable[numb][6].text = '0'
                self.lboundstable[numb][7].text = '0'
                self.lboundstable[numb][9].text = '0'
                self.rboundstable[numb][9].text = '90'
                self.lboundstable[numb][10].text = '1'
                self.realtableP[numb][1].text = str(1.0)
                self.realtableP[numb][2].text = str(0.0)
                self.realtableP[numb][3].text = str(0.0)
                self.realtableP[numb][4].text = str(15.0)
                self.realtableP[numb][5].text = str(0.098)
                self.realtableP[numb][6].text = str(0.1)
                self.realtableP[numb][7].text = str(5)
                self.realtableP[numb][8].text = str(1)
                self.realtableP[numb][9].text = str(-1)
                self.realtableP[numb][10].text = str(90)
                self.realtableP[numb][11].text = str(100)
                self.fixtable[numb][4].active = True
                self.fixtable[numb][10].active = True
                self.lboundstable[numb][10].disabled = True
                self.fixtable[numb][10].disabled = True
            if text == 'Distr':
                self.realtableP[numb][5].size_hint = (len(self.realtableP[numb])-5, 1)
                self.realtableP[numb][5].halign = "left"
                for j in range (6, len(self.realtableP[numb])):
                    self.realtableP[numb][j].pos_hint = {'right': len(self.realtableP[numb])-j}
                    self.fixtable[numb][j-1].color = (0,0,0,0)
                    self.lboundstable[numb][j - 1].color = (0, 0, 0, 0)
                    self.rboundstable[numb][j - 1].color = (0, 0, 0, 0)
                    self.nametable[numb][j-1].disabled = True
                self.fixtable[numb][4].color = (0, 0, 0, 0)
                self.nametable[numb][4].disabled = True

                self.nametable[numb][0].text = 'par'
                self.nametable[numb][1].text = 'L'
                self.nametable[numb][2].text = 'R'
                self.nametable[numb][3].text = 'Num'
                self.nametable[numb][4].text = 'Probability density function'

                self.nametable[numb][4].text_size[0] = self.nametable[numb][4].size[0] * 8
                self.nametable[numb][4].halign = "right"
                # self.nametable[numb][4].size = self.nametable[numb][4].texture_size
                self.lboundstable[numb][3].text = '1'
                self.rboundstable[numb][3].text = '1000'
                if self.realtableP[numb-1][0].text == 'Corr' or self.realtableP[numb-1][0].text == 'Distr':
                    self.realtableP[numb][1].text = str(int(self.realtableP[numb-1][1].text)+1)
                else:
                    self.realtableP[numb][1].text = str(1)
                self.realtableP[numb][2].text = str(0)
                self.realtableP[numb][3].text = str(1)
                self.realtableP[numb][4].text = str(20)
                self.realtableP[numb][5].text = str('X')
                self.fixtable[numb][0].active = True
                self.fixtable[numb][3].active = True
                self.fixtable[numb][4].active = True
                self.lboundstable[numb][0].disabled = True
                self.rboundstable[numb][0].disabled = True
                self.lboundstable[numb][3].disabled = True
                self.fixtable[numb][0].disabled = True
                self.fixtable[numb][3].disabled = True
                self.fixtable[numb][4].disabled = True
                self.lboundstable[numb][0].text = '1'
                # Mo = self.realtableP[numb-1][0].text
                self.rboundstable[numb][0].text = str(mod_len_def_M(self.realtableP[numb-1][0].text)-1)
            if text == 'Corr':
                self.realtableP[numb][2].size_hint = (len(self.realtableP[numb])-2, 1)
                self.realtableP[numb][2].halign = "left"
                for j in range (3, len(self.realtableP[numb])):
                    self.realtableP[numb][j].pos_hint = {'right': len(self.realtableP[numb])-j}
                    self.fixtable[numb][j-1].color = (0,0,0,0)
                    self.nametable[numb][j-1].disabled = True
                self.fixtable[numb][1].color = (0, 0, 0, 0)
                self.nametable[numb][1].disabled = True

                self.nametable[numb][0].text = 'par'
                self.nametable[numb][1].text = 'Dependency function'

                self.nametable[numb][1].text_size[0] = self.nametable[numb][4].size[0] * 8
                self.nametable[numb][1].halign = "right"
                # self.nametable[numb][4].size = self.nametable[numb][4].texture_size
                try:
                    self.realtableP[numb][1].text = str(int(self.realtableP[numb-1][1].text) + 1)
                except:
                    self.realtableP[numb][1].text = 'None'
                self.realtableP[numb][2].text = str('X')
                self.fixtable[numb][0].active = True
                self.fixtable[numb][1].active = True
                self.lboundstable[numb][0].disabled = True
                self.rboundstable[numb][0].disabled = True
                self.fixtable[numb][0].disabled = True
                self.fixtable[numb][1].disabled = True
                self.lboundstable[numb][0].text = '1'
                try:
                    Mo = self.realtableP[numb-2][0].text
                except:
                    Mo = str('XXX')
                self.rboundstable[numb][0].text = str(mod_len_def_M(Mo)-1)

            if text == 'Insert' and self.realtableP[numb][0].text != 'Insert':
                for k in range(3, len(self.realtableP)-numb+1):
                    self.realtableP[-k+1][0].text = 'None'
                    self.select(self.realtableP[-k+1][0], self.realtableP[-k][0].text, -k+1, _instance)
                    self.fix_memory_table[-k+1] = self.fix_memory_table[-k]
                    self.fix_table_ch[-k+1] = self.fix_table_ch[-k]
                    self.startlabel2[-k+1].text = self.startlabel2[-k].text
                    self.startlabel2[-k+1].color = self.startlabel2[-k].color
                    # self.realtableP[-k+1][0].text = self.realtableP[-k][0].text
                    for kk in range(0, len(self.realtableP[-k])-1):
                        # print(k, kk)
                        self.realtableP[-k+1][kk+1].text = self.realtableP[-k][kk+1].text
                        self.lboundstable[-k+1][kk].text = self.lboundstable[-k][kk].text
                        self.rboundstable[-k+1][kk].text = self.rboundstable[-k][kk].text
                        self.nametable[-k+1][kk].text = self.nametable[-k][kk].text
                        self.fixtable[-k+1][kk].active = self.fixtable[-k][kk].active
                for i in range(0, len(self.lboundstable[numb])):
                    self.lboundstable[numb][i].text = ''
                    self.rboundstable[numb][i].text = ''
                    self.nametable[numb][i].text = ''
                    self.fixtable[numb][i].active = False
                for i in range(1, len(self.lboundstable[numb]) + 1):
                    self.realtableP[numb][i].text = ''
            if self.realtableP[numb][0].text == 'Insert' and text == 'Insert':
                self.select(self.realtableP[numb+1][0], 'Insert', numb+1, _instance)

            if text == 'Delete':
                Model_del_name = self.realtableP[numb][0].text
                Mo = mod_len_def(Model_del_name)

                Parameter_counter = NBA
                for k in range(1, numb):
                    Parameter_counter += mod_len_def(self.realtableP[k][0].text)
                for k in range(0, len(self.realtableP)-1):
                    for kk in range(1, len(self.realtableP[k])):
                        kkk = 0
                        for kkkt in range(2, (len(str(self.realtableP[k][kk].text))+1)):
                            if str(self.realtableP[k][kk].text)[-kkkt + kkk] == '=' and str(self.realtableP[k][kk].text)[-kkkt+1+kkk] == '[':
                                for kkkk in range(-kkkt+2+kkk, 0):
                                    if str(self.realtableP[k][kk].text)[kkkk] == ',':
                                        try:
                                            old_number = int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk])
                                        except:
                                            old_number = -1
                                        if old_number >= (Parameter_counter + Mo):
                                            new_number = str(int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk]) - Mo)
                                            self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                            kkk += len(str(old_number)) - len(str(new_number))
                                        if old_number >= Parameter_counter and old_number < Parameter_counter + Mo:
                                            new_number = str('?')
                                            self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                            kkk += len(str(old_number)) - 1
                                        break
                            if str(self.realtableP[k][kk].text)[-kkkt + kkk] == 'p' and str(self.realtableP[k][kk].text)[-kkkt+1+kkk] == '[':
                                for kkkk in range(-kkkt+2+kkk, 0):
                                    if str(self.realtableP[k][kk].text)[kkkk] == ']':
                                        try:
                                            old_number = int(str(self.realtableP[k][kk].text)[-kkkt + 2 + kkk:kkkk])
                                        except:
                                            old_number = -1
                                        if old_number >= (Parameter_counter + Mo):
                                            new_number = str(int(str(self.realtableP[k][kk].text)[-kkkt+2+kkk:kkkk]) - Mo)
                                            self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                            kkk += len(str(old_number)) - len(str(new_number))
                                        if old_number >= Parameter_counter and old_number < Parameter_counter + Mo:
                                            new_number = str('?')
                                            self.realtableP[k][kk].text = str(self.realtableP[k][kk].text)[:-kkkt+2+kkk] + new_number + str(self.realtableP[k][kk].text)[kkkk:]
                                            kkk += len(str(old_number)) - 1
                                        break
                for k in range(numb, len(self.realtableP)-2):
                    self.realtableP[k][0].text = 'None'
                    self.realtableP[k][0].background_color = [1, 1, 1, 1]
                    if self.realtableP[k][0] != 'Insert' and self.realtableP[k + 1][0].text != 'Insert':
                        self.select(self.realtableP[k][0], self.realtableP[k + 1][0].text, k, _instance)
                    else:
                        self.realtableP[k][0].text = 'Insert'
                    # self.realtableP[k][0].text = self.realtableP[k + 1][0].text
                    self.fix_memory_table[k] = self.fix_memory_table[k+1]
                    self.fix_table_ch[k] = self.fix_table_ch[k+1]
                    self.startlabel2[k].text = self.startlabel2[k + 1].text
                    self.startlabel2[k].color = self.startlabel2[k + 1].color
                    for kk in range(0, len(self.realtableP[k])-1):
                        # print(k, kk)
                        self.realtableP[k][kk+1].text = self.realtableP[k+1][kk+1].text
                        self.lboundstable[k][kk].text = self.lboundstable[k+1][kk].text
                        self.rboundstable[k][kk].text = self.rboundstable[k+1][kk].text
                        self.nametable[k][kk].text = self.nametable[k+1][kk].text
                        self.fixtable[k][kk].active = self.fixtable[k+1][kk].active
                for i in range(0, len(self.lboundstable[len(self.realtableP)-2])):
                    self.lboundstable[len(self.realtableP)-2][i].text = ''
                    self.rboundstable[len(self.realtableP)-2][i].text = ''
                    self.nametable[len(self.realtableP)-2][i].text = ''
                    self.fixtable[len(self.realtableP)-2][i].active = False
                for i in range(1, len(self.lboundstable[len(self.realtableP)-2]) + 1):
                    self.realtableP[len(self.realtableP)-2][i].text = ''
                self.realtableP[len(self.realtableP) - 2][0].text = 'None'

            else:
                drop_button.text = text
                if text == 'Insert':
                    drop_button.background_color = [255, 0, 0, 1]


        def INTERUPT(self, _instance):
            global pool
            pool.close()
            pool.join()
            global RL, initial
            pool = mp.Pool((mp.cpu_count())*(mp.cpu_count()<=4) + (mp.cpu_count()-1)*(mp.cpu_count()>4))
            initial = 0
            RL = 0
            self.play_btn.disabled = False
            self.INS_btn.disabled = False
            self.INS_btn2.disabled = False
            self.show_btn.disabled = False
            self.showM_btn.disabled = False
            self.save_path.readonly = False
            self.cal_btn.disabled = False

        def chB_a(self, i):
            try:
                if i == 1 and self.fitway[1].active == True and self.switch.active == True:
                    self.L0.background_color = [1, 1, 1, 0.5]
                    if self.fitway[0].active == True:
                        self.JN0.text = str(int(int(self.JN0.text) / 2))
                    if self.fitway[2].active == True:
                        self.JN0.text = str(int(int(self.memoryJN[0]) / (1 + int(self.memoryJN[1]==0))))
                    if self.realtableP[0][5].text == '=[0,0.67]':
                        self.realtableP[0][5].text = '0'
                        self.background_calc(self)
                elif i == 0 and self.fitway[0].active == True and self.switch.active == True:
                    self.L0.background_color = [1, 1, 1, 1]
                    if self.fitway[1].active == True:
                        self.JN0.text = str(int(int(self.JN0.text) * 2))
                    if self.fitway[2].active == True:
                        self.JN0.text = str(int(self.memoryJN[0]) * (1 + int(self.memoryJN[1]==1)))
                    if self.realtableP[0][5].text == '0' or self.realtableP[0][5].text == '0.0':
                        self.realtableP[0][5].text = '=[0,0.67]'
                        self.background_calc(self)
                elif i == 2 and self.fitway[2].active == True and self.switch.active == True:
                    self.L0.background_color = [1, 1, 1, 0.5]
                    # if self.fitway[1].active == True:
                        # self.JN0.text = str(int(int(self.JN0.text) * 2))
                    if self.fitway[0].active == True or self.fitway[1].active == True:
                        self.memoryJN = [self.JN0.text, int(self.fitway[1].active == True)]
                        self.JN0.text = str(2048)
                        print(self.memoryJN)
                    if self.realtableP[0][5].text == '=[0,0.67]':
                        self.realtableP[0][5].text = '0'
                        self.background_calc(self)
            except:
                pass
            self.fitway[0].active = False
            self.fitway[1].active = False
            self.fitway[2].active = False
            self.fitway[i].active = True


        def chB_b(self, i):
            self.seq_fit[0].active = False
            self.seq_fit[1].active = False
            self.seq_fit[i].active = True

        def on_request_close(self, *args):
            global pool
            PhysicsApp().stop()
            pool.close()
            pool.join()
            print('pool is closed')
            os.kill(os.getpid(), signal.SIGTERM)
            exit()
            return True

        def Calibration(self, *args):
            global TR
            # self.play_btn.background_color = [0, 0.5, 0, 1]
            # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
            # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
            # self.cal_btn.background_color = [0.5, 0.5, 0.5, 0.5]
            # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
            # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
            TR = threading.Thread(target=partial(self.Calibration2))
            TR.name = 'TTT'
            TR.daemon = True
            TR.start()

        def Calibration2(self):
                # self.play_btn.background_color = [0, 0.5, 0, 1]
                # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
                # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
                # self.play_btn.disabled = True
                # self.INS_btn.disabled = True
                # self.INS_btn2.disabled = True
                # self.show_btn.disabled = True
                # self.showM_btn.disabled = True
                # self.save_path.readonly = True
                # self.cal_btn.disabled = True
                global RL, RA, Rlog, RlogCol
            # try:
                raw_path = self.process_path.text[2:-2]
                self.path_list = []
                v = 0
                for i in range(0, len(raw_path) - 2):
                    if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                        vv = i
                        self.path_list.append(str(raw_path)[v:vv])
                        v = i + 4
                self.path_list.append(str(raw_path)[v:])

                file = os.path.abspath(self.path_list[0])
                RL = 1
                if file[-4:] == '.mca' or file[-5:] == '.cmca' or file[-4:] == '.ws5' or file[-4:] == '.w98' or file[-4:] == '.moe' or file[-3:] == '.m1' or file[-4:].casefold() == '.mcs':
                    global x0, MulCo, initial
                    if initial == 0:
                        Tini = threading.Thread(target=partial(initialization, self))
                        Tini.name = 'T_ini'
                        Tini.daemon = True
                        Tini.start()
                        Tini.join()
                    JN = int(self.JN0.text)
                    VVV = (self.fitway[0].active == True) + 3*(self.fitway[1].active == True)
                    if VVV == 1:
                        INS = float(self.L0.text)
                    if VVV == 3:
                        if platform.system() == 'Windows':
                            realpath = str(self.dir_path) + str('\\\\INSexp.txt')
                        else:
                            realpath = str(self.dir_path) + str('/INSexp.txt')
                        INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)

                    RL = 1
                    # A, B, C = cal.Calibration(self.dir_path, file, pool) # FOR OLD calibration
                    A, B, C = cal.Calibration(self.dir_path, file, pool, VVV, INS, JN, x0, MulCo, int(self.Vel_start.active))

                    fig = plt.figure(figsize=(2942 / 300, 4.5), dpi=300)
                    # fig = plt.figure(figsize=(2942/300, 4.5), dpi=300)
                    ax = fig.add_subplot(111)
                    plt.xlim(min(A), max(A))
                    plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
                    plt.plot(A, B, linestyle='None', marker='x', color='m')
                    plt.plot(A, C, color='r')
                    plt.plot(A, B - C + min(B) - max(B - C), color='lime')
                    plt.text(0, -0.1, os.path.abspath(self.path_list[0]), horizontalalignment='left', verticalalignment='center', color='m', transform=ax.transAxes)
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    plt.ylabel('Transmission, counts')
                    plt.xlabel('Velocity, mm/s')
                    # global p
                    # if p[2] == 0 and p[6] == 0:
                    #     ymin, ymax = ax1.get_ylim()
                    #     ax2 = ax1.twinx()
                    #     ax2.set_ylim((ymin / p[0], ymax / p[0]))
                    if platform.system() == 'Windows':
                        realpath = str(self.dir_path) + str('\\\\result.png')
                    else:
                        realpath = str(self.dir_path) + str('/result.png')
                    fig.savefig(realpath, bbox_inches='tight')
                    # plt.close()
                    plt.cla()
                    plt.clf()
                    plt.close('all')
                    plt.close(fig)
                    gc.collect()

                    RA = True
                    # self.left.canvas.ask_update()
                    # for i in range(0, len(self.buttons)):
                    #     self.buttons[i].canvas.ask_update()
                    # for i in range(0, len(self.dd)):
                    #     self.dd[i].canvas.ask_update()

                    self.SP_DI.text = 'Distribution'
                    RL = 4
                    self.points_match = True
                    self.check_points_match = False
                    # self.log.text = "Calibration.dat created and will be used for *.mca"
                    self.cal_path = str(self.dir_path) + str('\\\\Calibration.dat')*(platform.system() == 'Windows') + str('/Calibration.dat')*(platform.system() != 'Windows')
                    # self.cal_cho_text.text = os.path.basename(self.cal_path)


                elif file[-4:] == '.dat':
                    RL = 3
                    RlogCol = [255, 0, 0, 1]
                    Rlog = "Calibration should be done with RAW data"

                else:
                    RL = 6
            # except:
            #     RL = 3
            #     # time.sleep(1)
            #     self.log.background_color = [255, 0, 0, 1]
            #     self.log.text = "Something goes wrong"

                #
                # self.save_path.readonly = False
                # self.play_btn.disabled = False
                # self.INS_btn.disabled = False
                # self.INS_btn2.disabled = False
                # self.show_btn.disabled = False
                # self.showM_btn.disabled = False
                # self.cal_btn.disabled = False

        def Calibration_path(self, *args):
            raw0_path = filechooser.open_file(path=self.workfolder, title="Choose file with correct velocity scale", filters=[("dat", "txt", "*.dat", "*.txt")], multiple=False)
            if raw0_path != [] and raw0_path != None:
                self.cal_path = str(raw0_path)[2:-2]
                # self.cal_cho_text.text = os.path.basename(self.cal_path)
                self.log.background_color = [0, 255, 0, 1]
                self.log.text = "New velocity scale will be applied for RAW files"
                if self.points_match == False:
                    self.show_pressed(self)
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Selection was canceled or path is too long"

        def velocityscale(self, *args):
            TR = threading.Thread(target=partial(self.velocityscale2))
            TR.name = 'TTT'
            TR.daemon = True
            TR.start()

        def velocityscale2(self):
            raw_path = self.process_path.text[2:-2]
            # self.path_list = []
            # v = 0
            # for i in range(0, len(raw_path) - 2):
            #     if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
            #         vv = i
            #         self.path_list.append(str(raw_path)[v:vv])
            #         v = i + 4
            # self.path_list.append(str(raw_path)[v:])

            if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/' or str(raw_path)[-1] == '.':
                try:
                    self.path_list = []
                    for files in os.listdir(str(raw_path)):
                        if files.endswith(".dat") or files.endswith(".mca"):
                            self.path_list.append(os.path.join(str(raw_path), files))
                    for i in range(0, len(self.path_list)):
                        if os.path.exists(self.path_list[i]) == False:
                            self.log.background_color = [255, 255, 0, 1]
                            self.log.text = "At least one path do not exist"
                        else:
                            raw_path = self.path_list[0]
                except:
                    check = False
                    self.log.background_color = [255, 255, 0, 1]
                    self.log.text = "Directory do not exist"


            try:
            # if True:
                if self.switch.active == True:
                    A_list = []
                    # if platform.system() == 'Windows':
                    #     rpath = str(self.dir_path) + str('\\\\Calibration.dat')
                    # else:
                    #     rpath = str(self.dir_path) + str('/Calibration.dat')
                    rpath = self.cal_path
                    with open(rpath, 'r') as catalog:
                        lines = (line.rstrip() for line in catalog)
                        lines = (line for line in lines if line)  # skipping white lines
                        for line in lines:
                            column = line.split()
                            if not line.startswith('#'):  # skipping column labels
                                x = float(column[0])
                                A_list.append(x)
                    A = np.array(A_list)

                    cal_type = open(rpath, 'r')
                    try:
                        cal_info = (cal_type.readline()).split()[1:]
                        cal_method = cal_info[0]
                        n1 = int(cal_info[1])
                        n2 = int(cal_info[2])
                    except:
                        cal_method = str('sin')
                        n1 = 0
                        n2 = int(len(np.array(A_list)))*2 - 1
                        print('No information in first line of calibration')
                    cal_type.close()

                global RL, Rlog, RlogCol
                RL = 1
                check_dat = 0
                check_dat2 = 0

                for kk in range(0, len(self.path_list)):
                    try:
                    # if True:
                        file = os.path.abspath(self.path_list[kk])
                        if file[-4:] == '.ws5' or file[-4:] == '.w98' or file[-4:] == '.moe' or file[-3:] == '.m1' or file[-4:].casefold() == '.mcs':
                            Y = []
                            Y.append([])
                            A, Y[0] = read_spectrum(self, file)
                        if file[-4:] == '.mca' or file[-5:] == '.cmca':
                            B_list = []
                            LS = len(open(file, 'r').readlines())
                            with open(file, 'r') as fi:
                                id = []
                                Tend = []
                                Tlon = []
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

                                        try:
                                            if ln.startswith("#D"):
                                                Tend.append(str(ln).split(' ')[4])
                                            if ln.startswith("#T"):
                                                Tlon.append(str(ln).split(' ')[1])
                                        except:
                                            print('! could not read time from *.mca file !')

                                    n += 1

                            if len(Tend) > len(Tlon):
                                Tend.pop(0)
                            ti_end_sec = []
                            ti_st_sec = []
                            Tst = []
                            temp_time = [0,0,0,0,0,0]
                            if len(Tend) > 1:
                                if str(Tend[0]).count(':') == 2:
                                    day_check = 0
                                    day_ini_check = 0
                                    ini_start = int(int(Tend[0][0])*36000 + int(Tend[0][1])*3600 + int(Tend[0][3])*600 + int(Tend[0][4])*60 + int(Tend[0][6])*10 + int(Tend[0][7])) - int(Tlon[0])
                                    for i in range(0, len(Tlon)):
                                        if int(Tend[i][0]) == 0 and int(Tend[i][1]) == 0:
                                            if day_ini_check == 1:
                                                day_check += 1
                                                day_ini_check = 0
                                        else:
                                            day_ini_check = 1
                                        ti_end_sec.append(int((int(Tend[i][0]) + 2*day_check)*36000 + (int(Tend[i][1]) + 4*day_check)*3600 + int(Tend[i][3])*600 + int(Tend[i][4])*60 + int(Tend[i][6])*10 + int(Tend[i][7])))
                                        ti_st_sec.append(ti_end_sec[i] - int(Tlon[i]))
                                        temp_time[0] = ti_st_sec[i] // 36000
                                        temp_time[1] = ti_st_sec[i] % 36000 // 3600
                                        temp_time[2] = ti_st_sec[i] % 36000 % 3600 // 600
                                        temp_time[3] = ti_st_sec[i] % 36000 % 3600 % 600 // 60
                                        temp_time[4] = ti_st_sec[i] % 36000 % 3600 % 600 % 60 // 10
                                        temp_time[5] = ti_st_sec[i] % 36000 % 3600 % 600 % 60 % 10
                                        day_counter = str(int(str(temp_time[0])+str(temp_time[1]))%24)
                                        Tst.append(str('0')*(len(day_counter)==1) + day_counter \
                                                   +str(':')+str(temp_time[2])+str(temp_time[3])+str(':')+str(temp_time[4])+str(temp_time[5]))
                                        ti_end_sec[i] = ti_end_sec[i] - ini_start
                                        ti_st_sec[i] = ti_st_sec[i] - ini_start

                            if self.switch.active == True:
                                id = np.array(id, dtype=float)
                                # id[j] = id[j][::-1]
                                # Y = np.array([[float(0)] * int(len(id[0]) / 2)] * len(id))
                                Y = []
                                for j in range(0, len(id)):  # len(id)
                                    # for i in range(0, int(len(id[0]) / 2)):
                                    #     Y[j][i] = id[j][i] + id[j][int(len(id[0])) - 1 - i]
                                    if file[-5:] == '.cmca':
                                        if (id[j][-1] == 0 and id[j][0] != 0) or (id[j][0] == 0 and id[j][-1] != 0):
                                            id_half = (id[j][-1] + id[j][0]) / 2
                                            id[j][-1] = id_half
                                            id[j][0] = id_half
                                    if cal_method == str('sin'):
                                        spc_1h = id[j][:int(n1 / 2)] + id[j][n1 - int(n1 / 2):n1][::-1]
                                        spc_2h = id[j][n2+1:n2 + int((len(id[j]) - 1 - n2) / 2)+1][::-1] + id[j][len(id[j]) - int((len(id[j]) - 1 - n2) / 2):len(id[j])]
                                        spc_3h = id[j][n1:n1 + int((n2 - n1 + 1) / 2)] + id[j][n2 - int((n2 - n1 + 1) / 2) + 1:n2 + 1][::-1]
                                        Y.append(np.concatenate((np.concatenate((spc_1h, spc_2h)), spc_3h)))
                                    if cal_method == str('lin'):
                                        # Y.append(np.array([float(0)] * int((n2 - n1 + 1) / 2)))
                                        # for i in range(0, int(len(id[0]) / 2) - n1):
                                        #     Y[j][i] = id[j][n1 + i] + id[j][n2 - i]
                                        ### new correction
                                        n_sh = 2 * (n1 + (int(len(id[-1])) - 1 - n2))
                                        Y.append(np.array([float(0)] * int((len(id[-1]) - n_sh) / 2)))
                                        for i in range(0, int((len(id[0]) - n_sh) / 2)):
                                            Y[j][i] = id[j][n1 + i] + id[j][n2 - i]

                        if self.switch.active == False: # NFS version
                            NFS_cal = np.genfromtxt(str(self.dir_path) +
                                                    str('\\\\NFS.txt') * (platform.system() == 'Windows') +
                                                    str('/NFS.txt') * (platform.system() != 'Windows'),
                                                    delimiter='\t', skip_footer=0)
                            A_n = []
                            for i in range(0, len(id)):
                                A_n.append([])
                                for j in range(0, len(id[i])):
                                    A_n[-1].append(NFS_cal[0] + NFS_cal[1] * j + NFS_cal[2] * j ** 2)

                            A = copy.deepcopy(A_n)
                            id_n = copy.deepcopy(id)
                            for i in range(0, len(A_n)):
                                for j in range(0, len(A_n[i])):
                                    if A_n[i][len(A_n[i]) - 1 - j] < NFS_cal[3]: # cut left part of spc - below fixed time (last value in "NFS.txt" in ns)
                                        # A[i].delete(A[i], len(A_n[i]) - 1 - j)
                                        # id_n[i] = np.delete(id_n[i], len(A_n[i]) - 1 - j)
                                        A[i].pop(len(A_n[i]) - 1 - j)
                                        id_n[i].pop(len(A_n[i]) - 1 - j)
                                        # print(len(A_n[i]), len(A_n[i]) - 1 - j)
                            if len(A) == 1:
                                A = A[0]
                            Y = id_n.copy()

                        if len(Y) == 1:
                            if os.path.exists(os.path.dirname(self.save_path.text))==False:
                                os.makedirs(os.path.dirname(self.save_path.text))
                            rpath = os.path.dirname(self.save_path.text) + OSslesh + os.path.splitext(os.path.basename(file))[0] + str('.dat') # str(file[:-4]) + str('.dat')
                            f = open(rpath, "w")
                            for i in range(0, int(len(A))):
                                f.write(str(A[i]) + '\t' + str(Y[0][i]) + '\n')
                            f.write('\n')
                            f.close()
                        else:
                            if os.path.exists(os.path.dirname(self.save_path.text))==False:
                                os.makedirs(os.path.dirname(self.save_path.text))
                            for j in range(0, len(Y)):
                                rpath = os.path.dirname(self.save_path.text) + OSslesh + os.path.splitext(os.path.basename(file))[0]\
                                        + str('_') + str('0')*(len(str(int(len(Y)))) - len(str(int(j)))) + str(j) +  str('.dat')
                                f = open(rpath, "w")
                                if self.switch.active == True:
                                    for i in range(0, int(len(A))):
                                        f.write(str(A[i]) + '\t' + str(Y[j][i]) + '\n')
                                elif self.switch.active == False:
                                    for i in range(0, int(len(A[j]))):
                                        f.write(str(A[j][i]) + '\t' + str(Y[j][i]) + '\n')
                                f.write('\n')
                                f.close()
                            rpath = os.path.dirname(self.save_path.text) + OSslesh + str('Time_table_') + os.path.splitext(os.path.basename(file))[0] + str('.txt')
                            f = open(rpath, "w")
                            f.write(str('#spc_num\ttime_start\ttime_end\tseconds_start\tseconds_end\n'))
                            for i in range(0, len(Tst)):
                                f.write(str('0')*(len(str(int(len(Y)))) - len(str(int(i)))) + str(i) + str('\t'))
                                f.write(str(Tst[i]) + str('\t'))
                                f.write(str(Tend[i]) + str('\t'))
                                f.write(str(ti_st_sec[i]) + str('\t'))
                                f.write(str(ti_end_sec[i]) + str('\n'))
                            f.write('\n')
                            f.close()

                        check_dat2 = 1
                    except:
                        check_dat = 1
                        # self.log.background_color = [255, 0, 0, 1]
                        # self.log.text = "Something goes wrong"
                RL = 3
                time.sleep(1)
                if check_dat == 0 and check_dat2 == 1:
                    RlogCol = [0, 255, 0, 1]
                    Rlog = "DAT file(s) were created"
                elif check_dat == 1 and check_dat2 == 1:
                    RlogCol = [255, 255, 0, 1]
                    Rlog = "DAT file(s) were created. But not all... Please check RAW files"
                elif check_dat == 1 and check_dat2 == 0:
                    RlogCol = [255, 0, 0, 1]
                    Rlog = "Something was wrong. Please check RAW file(s)"


            except:
                RL = 3
                time.sleep(1)
                RlogCol = [255, 0, 0, 1]
                Rlog = "Please do calibration first"

        def clean_model(self, _instance):
            if self.waitingdoubleclick == 1:
                print('double clicked clean - start cleaning the model')
                first_not_none = 1
                for i in range(2, len(self.realtableP) - 1):
                    if self.realtableP[len(self.realtableP) - i][0].text != 'None':
                        first_not_none = len(self.realtableP) - i
                        break
                for i in range(0, first_not_none):
                    self.select(self.realtableP[first_not_none - i][0], 'Delete', first_not_none - i, 0)
                print('model cleaned')
            # Clock.schedule_once(partial(self.clean_wait), 0)
            self.waitingdoubleclick = 1
            Clock.schedule_once(partial(self.clean_zero), 0.2)
            return True

        # def clean_wait(self, _instance):
        #     self.waitingdoubleclick = 1
        #     Clock.schedule_once(partial(self.clean_zero), 0.2)

        def clean_zero(self, _instance):
            self.waitingdoubleclick = 0


        def choose_workfolder(self, _instance):
            raw0_path = filechooser.choose_dir(path=self.workfolder, title="Choose workfolder...")
            print(raw0_path)
            if raw0_path != [] and raw0_path != None:
                self.workfolder = raw0_path[0] + OSslesh
                self.workfolder_check = 1

        def choose_file(self, _instance):
            # global fp, Rpa
            # self.popup = Popup(title='Select a Spectrum',
            #                    content=MyFileChooser(),
            #                    size_hint=(None, None), size=(1000, 1000))
            # self.popup.open()
            # raw_path = filechooser.open_file(title="Pick a spectrum...", dirselect=True)
            raw0_path = filechooser.open_file(path=self.workfolder, title="Pick a spectrum...", filters=[("dat", "*.dat", '*.spc', "*.exp"), ("RAW", "*.mca", "*.cmca", "*.ws5","*.moe", '*.w98', '*m1', '*.mcs'), ("dat, RAW", "*.dat", "*.mca", "*.cmca", "*.ws5","*.moe", '*.w98', '*.spc', '*m1', '*.mcs'), ("txt", "*.txt"), ("All", "*.*")], multiple=True)
            time.sleep(0.1) # preventing click on button after choose


            if raw0_path != [] and raw0_path != None :
                # Linux do not use last folder but Windows do. So for Linux have problem - workfolder
                if self.workfolder_check == 0 and platform.system() != 'Windows':
                    self.workfolder = os.path.dirname(raw0_path[0])
                    print('workfoler will be: ', self.workfolder)


                raw_path = str(raw0_path)[2:-2]
                print(raw0_path)
                self.path_list = raw0_path
                if len(self.path_list) > 1:
                    for i in range(0, len(str(self.path_list[0]))):
                        j = len(self.path_list[0])-1-i
                        if str(self.path_list[0])[j] == '/':
                            sys = 'L'
                            break
                        if str(self.path_list[0])[j] == '\\':
                            sys = 'W'
                            break
                        self.path_dir = str(self.path_list[0])[:j]
                    if sys == 'W':
                        self.path_dir = self.path_dir + str('result\\result')
                    else:
                        self.path_dir = self.path_dir + str('result/result')
                if len(self.path_list) == 1:
                    self.path_dir = str(self.path_list[0])
                if len(self.path_list) == 0:
                    self.path_dir = self.save_path.text
                self.process_path.text = str(self.path_list)
                self.save_path.text = self.path_dir
                self.show_pressed(self)
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Selection was canceled or path is too long"

        def loadmod_pressed(self, _instance):
            raw0_path = filechooser.open_file(path=self.workfolder, title="Pick a model...", filters=[("mdl", "*.mdl"), ("txt", "*.txt")], multiple=True)
            if raw0_path != [] and raw0_path != None :
                # try:
                    raw_path = str(raw0_path)[2:-2]
                    print(raw0_path)
                    M_list = []
                    file1 = open(raw_path, 'r', encoding="utf-8")
                    Lines = file1.readlines()
                    counter = 0
                    for line in Lines:
                        M_list.append(line.strip().split("\t"))
                        if M_list[counter][0] == str('False'):
                            M_list[counter].insert(0, str(''))
                            M_list[counter].insert(0, str(''))
                            M_list[counter].insert(0, str(''))
                            M_list[counter].insert(0, str(''))
                        counter += 1
                    first_not_none = 0
                    for i in range(2, len(self.realtableP)-1):
                        if self.realtableP[len(self.realtableP)-i][0].text != 'None':
                            first_not_none = len(self.realtableP)-i
                            break
                    for i in range(0, first_not_none):
                        self.select(self.realtableP[first_not_none-i][0], 'Delete', first_not_none-i, 0)
                    for i in range(1, min(len(self.realtableP) - 2, len(M_list) -1)):
                        self.select(self.realtableP[i][0],  M_list[0][i], i, 0)
                        # self.realtableP[i][0].text = M_list[0][i]
                    # for i in range(0, len(self.realtableP) - min(len(self.realtableP), len(M_list))):
                    #     self.select(self.realtableP[len(self.realtableP)-2-i][0], 'Delete', len(self.realtableP)-2-i, 0)
                    for k in range(0, len(M_list)-1):
                        for i in range(0, int(len(M_list[k+1])/5)):
                            self.realtableP[k][i + 1].text = M_list[k+1][i*5]
                            self.lboundstable[k][i].text = M_list[k+1][i*5+1]
                            self.rboundstable[k][i].text = M_list[k+1][i*5+2]
                            # self.nametable[k][i].text = M_list[k+1][i*5+3]
                            if M_list[k+1][i*5+4] == str('True'):
                                self.fixtable[k][i].active = True
                            else:
                                self.fixtable[k][i].active = False
                    self.log.text = "Model is ready"
                    self.log.background_color = [0, 255, 0, 1]
                # except:
                #     self.log.background_color = [255, 0, 0, 1]
                #     self.log.text = "Could not open model. Something went wrong."
                    if self.realtableP[0][NBA-1].text == '':
                        self.realtableP[0][7].text = self.realtableP[0][6].text
                        self.realtableP[0][6].text = self.realtableP[0][5].text
                        self.realtableP[0][5].text = self.realtableP[0][4].text
                        self.realtableP[0][4].text = '0'
                        self.realtableP[0][8].text = '0'
                        self.lboundstable[0][6].text = self.lboundstable[0][5].text
                        self.lboundstable[0][5].text = self.lboundstable[0][4].text
                        self.lboundstable[0][4].text = self.lboundstable[0][3].text
                        self.rboundstable[0][6].text = self.rboundstable[0][5].text
                        self.rboundstable[0][5].text = self.rboundstable[0][4].text
                        self.rboundstable[0][4].text = self.rboundstable[0][3].text
                        self.lboundstable[0][3].text = ''
                        self.lboundstable[0][7].text = ''
                        self.rboundstable[0][3].text = ''
                        self.rboundstable[0][7].text = ''
                        self.fixtable[0][6].active = self.fixtable[0][5].active
                        self.fixtable[0][5].active = self.fixtable[0][4].active
                        self.fixtable[0][4].active = self.fixtable[0][3].active
                        self.fixtable[0][3].active = True
                        self.fixtable[0][7].active = True
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Selection was canceled or path is too long"


            # numb = 0
            # for i in range(0, len(self.lboundstable[numb])):
            #     self.lboundstable[numb][i].text = ''
            #     self.rboundstable[numb][i].text = ''
            #     self.nametable[numb][i].text = ''
            #     self.fixtable[numb][i].active = False
            # for i in range(1, len(self.lboundstable[numb])+1):
            #     self.realtableP[numb][i].text = ''

        # def print(self, text):
        #     self.log.text = self.log.text + "\n" + text
        def showM_pressed(self, *args):
            global image, params, path, initial
            print(self.process_path.text)

            params = self.params
            path = self.process_path.text

            if initial == 0:
                Tini = threading.Thread(target=partial(initialization, self))
                Tini.name = 'T_ini'
                Tini.daemon = True
                Tini.start()
                Tini.join()

            raw_path = self.process_path.text[2:-2]
            self.path_list = []
            v=0
            for i in range(0, len(raw_path) - 2):
                if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                    vv = i
                    self.path_list.append(str(raw_path)[v:vv])
                    v = i + 4
            self.path_list.append(str(raw_path)[v:])

            if os.path.exists(self.path_list[0]) or self.process_path.text == 'tango':
                # self.play_btn.background_color = [0, 0.5, 0, 1]
                # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.cal_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
                # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
                # Clock.schedule_once(partial(work3, self), 0.3)
                # global RA, RL, TR
                # RL = 1
                TR = threading.Thread(target=partial(work4, self))
                TR.name = 'TTT'
                TR.daemon = True
                TR.start()
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Sorry I could not show model without spectrum"

        def show_pressed(self, _instance):
            # print(str(self.dir_path))
            self.check_points_match == True
            # if True:
            try:
                global RL
                RL = 1
                raw_path = self.process_path.text[2:-2]
                self.path_list = []
                v=0


                if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/' or str(raw_path)[-1] == '.':
                    try:
                        path_tmp_local = []
                        for files in os.listdir(str(raw_path)):
                            if files.endswith(".dat") or files.endswith(".mca"):
                                path_tmp_local.append(os.path.join(str(raw_path), files))
                        check = True
                        for i in range(0, len(path_tmp_local)):
                            if os.path.exists(path_tmp_local[i]) == False:
                                self.log.background_color = [255, 255, 0, 1]
                                self.log.text = "At least one path do not exist"
                                check = False
                            if check == True:
                                self.path_list = path_tmp_local
                    except:
                        check = False
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "Directory do not exist"
                else:
                    for i in range(0, len(raw_path) - 2):
                        if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                            vv = i
                            self.path_list.append(str(raw_path)[v:vv])
                            v = i + 4
                    self.path_list.append(str(raw_path)[v:])
                    check = True

                # if self.path_list[0][-1] != '\\' and self.path_list[0][-1] != '/':
                # print(os.path.abspath(self.path_list[0]))
                A, B = [], []
                Amin = 10**8
                Amax = -10**8
                for i in range(0, len(self.path_list)):
                    file = os.path.abspath(self.path_list[i])
                    # print(i, file)
                    # if (raw_path[-1] == '\\' or raw_path[-1] == '/') and file[-1] != '.' and file[-1] != '/' and file[-1] != '\\':
                    #     file += str('\\') * (platform.system() == 'Windows') + str('/') * (platform.system() != 'Windows')
                    Atmp, Btmp = read_spectrum(self, file)
                    if max(Atmp) > Amax:
                        Amax = max(Atmp)
                    if min(Atmp) < Amin:
                        Amin = min(Atmp)
                    A.append(Atmp)
                    B.append(Btmp)
                # print(Amin, Amax)

                self.points_match = True
                ct = ['m', 'blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato']
                fig, ax1 = plt.subplots(figsize=(2942/300, 4.5), dpi=300)
                # fig = plt.figure(figsize=(2942 / 300, 4.5), dpi=300)
                plt.xlim(Amin, Amax)
                plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.ylabel('Transmission, counts')
                plt.xlabel('Velocity, mm/s')
                # global p
                # if p[2] == 0 and p[6] == 0:
                # ymin, ymax = ax1.get_ylim()
                # ax2 = ax1.twinx()
                # ax2.set_ylim((ymin/p[0], ymax/p[0]))

                if platform.system() == 'Windows':
                    realpath = str(self.dir_path) + str('\\\\result.png')
                else:
                    realpath = str(self.dir_path) + str('/result.png')
                print(self.path_list)

                for i in range(0, len(self.path_list)):
                    if len(A[i]) == len(B[i]) and len(A[i]) != 0 and len(A[i]) != 1:
                        # if self.switch.active == False:
                        #     plt.yscale("log")
                        #     plt.xlabel('Time, ns')
                        #     N0 = max(B) * np.exp(A[int(np.where(B == max(B))[0][0])] / 141 * 1.5)
                        # else:
                        #     if (B[0]+B[1]+B[2]+B[3]+B[4]+B[-1]+B[-2]+B[-3]+B[-4]+B[-5])/10 < (max(B) - 3.5*np.sqrt(max(B))):
                        #         N0 = max(B) - 3 * np.sqrt(max(B))
                        #     else:
                        #         N0 = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10
                        #
                        #     # self.realtableP[0][5].text = '0'
                        #
                        # self.realtableP[0][1].text = "%.0f" % (N0 - float(self.realtableP[0][5].text))

                        if len(self.path_list) > 1:

                            self.background_calc(self, POS=i, list=1)
                            if self.realtableP[0][5].text[0] != '=':
                                NonRes = float(self.realtableP[0][5].text)
                            else:
                                pNM = (self.realtableP[0][5].text[2:-1]).split(',')
                                pN, pM = int(pNM[0]), float(pNM[1])
                                global p
                                read_model(self)
                                NonRes = p[pN]*pM
                            BCG = float(self.realtableP[0][1].text) + NonRes
                            B[i] = B[i]/BCG
                            if self.switch.active == True:
                                plt.ylim(0, 1.05)



                        # self.points_match = True
                        plt.plot(A[i], B[i], linestyle='None', marker='x', color=ct[int(i%len(ct))])
                        # print(A[i])
                        # print(B[i])
                    else:
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "Number of points in spectrum do not coincide with calibration"
                        self.points_match = False

                if len(self.path_list) > 1:
                    plt.ylabel('Transmission - normalized')
                    Bmin = min(B[0])
                    Bmax = max(B[0])
                    for i in range(1, len(self.path_list)):
                        if min(B[i]) < Bmin:
                            Bmin = min(B[i])
                        if max(B[i]) > Bmax:
                            Bmax = max(B[i])
                    if self.switch.active == True:
                        plt.ylim(Bmin*0.97, Bmax*1.03)
                elif self.switch.active == True:
                    ymin, ymax = ax1.get_ylim()
                    ax2 = ax1.twinx()
                    BCG = max(B[i]) - 2*np.sqrt(max(B[i]))
                    BCG = BCG + (BCG == 0)
                    ax2.set_ylim((ymin / BCG, ymax / BCG))

                fig.savefig(realpath, bbox_inches='tight')
                # plt.show()
                # plt.close()
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
                gc.collect()

                self.background_calc(self)

                self.left.canvas.ask_update()
                for i in range(0, len(self.buttons)):
                    self.buttons[i].canvas.ask_update()
                for i in range(0, len(self.dd)):
                    self.dd[i].canvas.ask_update()
                global RA
                RA = True
                self.SP_DI.text = 'Distribution'
                if self.points_match == True:
                    self.log.background_color = [255, 255, 255, 1]
                    self.log.text = "Spectrum"

                for i in range(0, len(self.path_list)):
                    if len(A[i]) == 0 and len(A[i]) == 1:
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "Oops this one do not exist. No, I would not creat it."

                RL = 6
            except FileNotFoundError:
                RL = 3
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "FileNotFoundError, Oops this one do not exist. No, I would not creat it."
            except ValueError:
                RL = 3
                self.points_match = False
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "ValueError, Oops number of points do not match or file have problems."
            except UnboundLocalError:
                RL = 3
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "UnboundLocalError, something is strange, please check your file."
            return True

        def background_calc(self, _instance, POS = 0, numb = 0, list = 0):
            try:
                if list == 0:
                    raw_path = self.process_path.text[2:-2]
                    self.path_list = []
                    v=0
                    if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/' or str(raw_path)[-1] == '.':
                        try:
                            path_tmp_local = []
                            for files in os.listdir(str(raw_path)):
                                if files.endswith(".dat"):
                                    path_tmp_local.append(os.path.join(str(raw_path), files))
                            check = True
                            for i in range(0, len(path_tmp_local)):
                                if os.path.exists(path_tmp_local[i]) == False:
                                    self.log.background_color = [255, 255, 0, 1]
                                    self.log.text = "At least one path do not exist"
                                    check = False
                                if check == True:
                                    self.path_list = path_tmp_local
                        except:
                            check = False
                            self.log.background_color = [255, 255, 0, 1]
                            self.log.text = "Directory do not exist"
                    else:
                        for i in range(0, len(raw_path) - 2):
                            if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                                vv = i
                                self.path_list.append(str(raw_path)[v:vv])
                                v = i + 4
                        self.path_list.append(str(raw_path)[v:])
                file = os.path.abspath(self.path_list[POS])

                # print(file)
                A, B = read_spectrum(self, file)

                if self.switch.active == False:
                    plt.yscale("log")
                    plt.xlabel('Time, ns')
                    N0 = max(B) * np.exp(A[int(np.where(B == max(B))[0][0])] / 141 * 1.5)
                else:
                    if (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10 < (
                            max(B) - 3.5 * np.sqrt(max(B))):
                        N0 = max(B) - 3 * np.sqrt(max(B))
                    else:
                        N0 = (B[0] + B[1] + B[2] + B[3] + B[4] + B[-1] + B[-2] + B[-3] + B[-4] + B[-5]) / 10
                    N0 += int(N0)==0
                    try:
                        if self.realtableP[numb][5].text[0] != '=':
                            N0 = N0 - float(self.realtableP[numb][5].text)
                        else:
                            pNM = (self.realtableP[numb][5].text[2:-1]).split(',')
                            pN, pM = int(pNM[0]), float(pNM[1])
                            global p
                            read_model(self)
                            NonRes = p[pN]*pM
                            N0tmp = N0 - NonRes
                            self.realtableP[numb][1].text = "%.0f" % N0tmp
                            read_model(self)
                            N0 = N0/(1+pM) * (NonRes != p[pN]*pM)\
                               + N0tmp     * (NonRes == p[pN]*pM)


                    except:
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "Unexpected error while calculating background."
                print('baseline', N0)
                self.realtableP[numb][1].text = "%.0f" % N0
            except FileNotFoundError:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Oops file do not exist."
            except ValueError:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "ValueError: Oops could not calculate baseline."
            except IndexError:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "IndexError: Oops could not calculate baseline."

        def instrumental_pressed(self, ref, mode, *args):
            if mode == 1 and self.realtableP[1][0].text == 'None':
                self.log.background_color = [255, 0, 0, 1]
                self.log.text = "Specify model to restore instrumental function"
            elif self.switch.active == True and self.fitway[0].active == True and mode == 0:
                self.log.background_color = [255, 0, 0, 1]
                self.log.text = "This will not work..."
            else:
                global initial
                if initial == 0:
                    Tini = threading.Thread(target=partial(initialization, self))
                    Tini.name = 'T_ini'
                    Tini.daemon = True
                    Tini.start()
                    Tini.join()
                initial = 0
                file = os.path.abspath(self.path_list[0])
                if os.path.exists(file) == True:
                    TR = threading.Thread(target=partial(instrumental, self, ref, mode))
                    TR.name = 'TTT'
                    TR.daemon = True
                    TR.start()
            return True

        def instrumental_refine_pressed(self, *args):
            global initial
            if initial == 0:
                # pool = mp.Pool(1)
                Tini = threading.Thread(target=partial(initialization, self))
                Tini.name = 'T_ini'
                Tini.daemon = True
                Tini.start()
                Tini.join()
            initial = 0
            # Clock.schedule_once(partial(dummy_ins, self, 1), 0.3)
            ref = 1
            file = os.path.abspath(self.path_list[0])
            if os.path.exists(file) == True:
                TR = threading.Thread(target=partial(instrumental, self, ref))
                TR.name = 'TTT'
                TR.daemon = True
                TR.start()
            return True

        def take_result(self, _instance):
            # global Distri
            # global Cor
            # Di = int(0)
            # Co = int(0)
            if self.realtable[0][0].text != '':
                for i in range(0, len(self.realtableP) - 1):
                    if self.realtable[i*2][0].text != '':
                        # if self.realtableP[i][0].text == self.realtable[i * 2][0].text:
                        #     for j in range(1, len(self.realtableP[0])):
                        #         l = ''
                        #         for t in self.realtable[i*2][j].text.split():
                        #             try:
                        #                 l = float(t)
                        #             except ValueError:
                        #                 pass
                        #         try:
                        #             if self.realtableP[i][j].text[0:2] != '=[' and self.realtableP[i][j].text[-1] != ']':
                        #                 self.realtableP[i][j].text = str(l)
                        #         except:
                        #             self.realtableP[i][j].text = str(l)
                        # else:
                            if i != 0:
                                if self.realtableP[i][0].text != self.realtable[i * 2][0].text:
                                    self.select(self.realtableP[i][0], self.realtable[i * 2][0].text, i, 0)
                            for j in range(1, len(self.realtableP[0])):
                                l = ''
                                for t in self.realtable[i*2][j].text.split(' '):
                                    try:
                                        l = float(t)
                                    except ValueError:
                                        pass
                                if (self.realtable[i * 2][0].text == "Distr" and j == 5)\
                                        or (self.realtable[i * 2][0].text == "Corr" and j == 2)\
                                        or (self.realtable[i * 2][0].text == "Expression" and j == 1):
                                    print(i, j)
                                    l = str(self.realtable[i * 2][j].text.split(' =')[0])
                                    print(l)
                                    # Di += 1
                                # if (self.realtable[i * 2][0].text == "Corr" and j == 2):
                                #     print(i, j)
                                #     l = str(self.realtable[i * 2][j].text)
                                #     print(l)
                                #     # Co += 1
                                if self.realtable[i * 2 + 1][j].text == "± nan":
                                    self.fixtable[i][j-1].active = True
                                # if self.realtable[i * 2 + 1][j].text != "± nan" and self.realtable[i * 2 + 1][j].text[:2] != "± ":
                                # else:
                                #     self.fixtable[i][j-1].active = False

                                try:
                                    if (self.realtableP[i][j].text[0:2] == '=[' and self.realtableP[i][j].text[-1] == ']')==False:
                                        if (self.realtable[i * 2][0].text == "Distr" and (j == 1 or j == 4)) \
                                                   or (self.realtable[i * 2][0].text == "Corr" and j == 1): # \or (self.realtable[i * 2][0].text == "Expression" and j == 0)
                                            self.realtableP[i][j].text = str(int(l))
                                        else:
                                            self.realtableP[i][j].text = str(l)
                                except:
                                    self.realtableP[i][j].text = str(l)

                    else:
                        self.select(self.realtableP[i][0], str('None'), i, 0)
                        # self.realtableP[i][0].text = 'None'
                        # for j in range(1, len(self.realtableP[0])):
                        #     self.realtableP[i][j].text = ''

        # def online_play(self, _instance, touch):
        #     a = 1
        def play_pressed(self, *args):
          # if touch.button == 'right':
          #       self.log.text = "Online"
          # if touch.button == 'left':
            global image, params, path, initial
            self.log.text = "Start"
            print(self.process_path.text)

            params = self.params
            path = self.process_path.text
            # image = self.image


            if initial == 0:
                # pool = mp.Pool(1)
                Tini = threading.Thread(target=partial(initialization, self))
                Tini.name = 'T_ini'
                Tini.daemon = True
                Tini.start()
                Tini.join()

                # async_start = pool.apply_async(initialization())
                # pool.close()
                # pool.join


            # Clock.schedule_once(partial(work, image), 0.5)
            # Clock.schedule_once(partial(ImgUpdate, self.image), 5)
            # if os.path.exists(self.process_path.text):

            # Clock.schedule_once(partial(work, self), 0.3)
            global RA, RL, TR
            # RL = 1
            RPM = numro - 1
            for i in range(1, len(self.realtableP)):
                if self.realtableP[len(self.realtableP) - 1 - i][0].text != 'None':
                    RPM = len(self.realtableP) - 1 - i
                    break
            for i in range(1, RPM):
                if self.realtableP[RPM - i][0].text == 'None':
                    self.select(self.realtableP[RPM - i][0], 'Delete', RPM - i, 0)

            TR = threading.Thread(target=partial(work2, self))
            TR.name = 'TTT'
            TR.daemon = True
            TR.start()

            # else:
            #     self.log.background_color = [255, 255, 0, 1]
            #     self.log.text = "There is nothing to fit on this way"

        def save_way_chooser(self, CH, *args):
            self.popa_check = CH
            self.popupWindow.dismiss()
        def save_continue(self, *args):
            if self.popa_check == 4:
                self.save_path.text = self.newfilename
                self.popa_check = 3
            if self.popa_check == 5:
                self.save_path.text = self.newfilename2
                self.popa_check = 3
            if self.popa_check == 0:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Saving was canceled"
            else:
                file = self.save_path.text + '_param.txt'
                if self.popa_check == 1:
                    try:
                        with open(file, encoding="utf-8") as f:
                            first_line = f.readline().rstrip()
                        first_line = re.split(r'\t+', first_line)
                        print(first_line)
                    except:
                        first_line = []
                    f = open(file, "a", encoding="utf-8") #write to file output values
                    # self.log.background_color = [0, 255, 0, 1]
                    # self.log.text = "*_param.txt was prolonged, others were overwritten"
                elif self.popa_check == 2:
                    f = open(file, "w", encoding="utf-8") #write to file output values
                    # self.log.background_color = [0, 255, 0, 1]
                    # self.log.text = "Results were overwritten"
                    first_line = []
                elif self.popa_check == 3:
                    try:
                        f = open(self.save_path.text + '_param.txt', "a", encoding="utf-8")
                        f.close
                    except FileNotFoundError:
                        v = 0
                        for i in range(0, len(str(self.save_path.text))):
                            if str(self.save_path.text)[-1 - i] == '\\' or str(self.save_path.text)[-1 - i] == '/':
                                save_dir_path = str(self.save_path.text)[:-1 - i]
                                break
                        os.makedirs(save_dir_path)
                    file = self.save_path.text + '_param.txt'
                    f = open(file, "w", encoding="utf-8")  # write to file output values
                    first_line = []
                    # self.log.background_color = [0, 255, 0, 1]
                    # self.log.text = "Results were saved"

                name = []
                model_values = []
                v = 0
                vv = 0
                for i in range(0, len(self.realtable)):
                    for j in range(0, len(self.realtable[i])):
                        if len(str(self.realtable[i][j].text)) != 0:
                            if str(self.realtable[i][j].text)[0] == '±':
                                model_values.append(str(self.realtable[i][j].text)[2:])
                                name.append('Δ' + str(name[v]))
                                v += 1
                                vv += 1
                            else:
                                tmp = str(self.realtable[i][j].text).split(' ')
                                if self.realtable[i][0].text == 'Distr' and j == 5:
                                    model_values.append(self.realtable[i][j].text)
                                    name.append('PDF')
                                    v += 1
                                elif self.realtable[i][0].text == 'Corr' and j == 2:
                                    model_values.append(self.realtable[i][j].text)
                                    name.append('CoF')
                                    v += 1
                                elif self.realtable[i][0].text == 'Expression' and j == 1:
                                    model_values.append(tmp[-1])
                                    name.append(str(' '.join(tmp[0:-2])))
                                elif len(tmp) == 3:
                                    model_values.append(str(tmp[2]))
                                    name.append(str(tmp[0]))

                                elif len(tmp) == 1:
                                    model_values.append(str(self.realtable[i][j].text))
                                    name.append(str('subspc'))
                                    v += 1
                                    tmp_next = str(self.realtable[i + 1][j].text).split(' ')
                                    if len(tmp_next) == 4:
                                        model_values.append(str(tmp_next[0]))
                                        name.append(str('I%'))
                                        model_values.append(str(tmp_next[2]))
                                        name.append(str('ΔI%'))
                                        v += 2
                                    elif tmp_next[0] == 'Be' or tmp_next[0] == 'KB':
                                        model_values.append(str(tmp_next[0]))
                                        name.append(str('I%'))
                                        v += 1

                    v += vv
                    vv = 0

                name[1] = u'N\u2092'
                name[2] = u'X\u2092'
                name[3] = 'coef²'
                name[4] = u'Nnr'
                name[5] = u'Xnr'
                name[6] = 'coef²nr'
                name[7] = u'ΔN\u2092'
                name[8] = u'ΔX\u2092'
                name[9] = 'Δcoef²'
                name[7] = u'ΔNnr'
                name[8] = u'ΔXnr'
                name[9] = 'Δcoef²nr'
                name.append(str('χ²'))
                name.insert(0, str('#File'))

                if first_line != name:
                    for i in range (0, len(name)):
                        f.write(name[i] + '\t')
                    f.write('\n')
                f.write(os.path.basename(self.process_path.text[2:-2]) + str('\t'))
                for i in range (0, len(name)-2):
                    f.write(model_values[i] + '\t')
                global hi2
                f.write(str(hi2))
                f.write('\n')
                print(name)
                f.close()
                if platform.system() == 'Windows':
                    realpath = str(self.dir_path) + str('\\\\result.txt')
                else:
                    realpath = str(self.dir_path) + str('/result.txt')
                file = self.save_path.text + '_graf.txt'
                try:
                    shutil.copyfile(realpath, file)
                except shutil.SameFileError:
                    pass

                if platform.system() == 'Windows':
                    realpath = str(self.dir_path) + str('\\\\result.png')
                else:
                    realpath = str(self.dir_path) + str('/result.png')
                # file = self.save_path.text + '.png'
                # try:
                #     shutil.copyfile(realpath, file)
                # except shutil.SameFileError:
                #     pass
                im1 = matplotlib.image.imread(os.path.abspath(realpath))

                if platform.system() == 'Windows':
                    realpath = str(self.dir_path) + str('\\\\result_table.png')
                else:
                    realpath = str(self.dir_path) + str('/result_table.png')
                # file = self.save_path.text + '_table.png'
                # try:
                #     shutil.copyfile(realpath, file)
                # except shutil.SameFileError:
                #     pass
                im2 = matplotlib.image.imread(os.path.abspath(realpath))
                im1.astype(np.uint8)
                im2.astype(np.uint8)
                print(len(im2), len(im2[0]))
                color_array = [0,0,0,1] * (self.BGcolor == 'k') + [1,1,1,1] * (self.BGcolor == 'w')
                if len(im2[0]) > len(im1[0]): # np.zeros((len(im1), len(im2[0]) - len(im1[0]), 4), np.uint8)
                    im1 = np.concatenate((im1, np.array([[color_array]*(len(im2[0]) - len(im1[0]))]*len(im1), np.uint8)), axis=1)
                if len(im2[0]) < len(im1[0]): # np.zeros((len(im2), len(im1[0]) - len(im2[0]), 4), np.uint8)
                    im2 = np.concatenate((im2, np.array([[color_array]*(len(im1[0]) - len(im2[0]))]*len(im2), np.uint8)), axis=1)
                combo_image = np.concatenate((im1, im2), axis=0)

                matplotlib.image.imsave(os.path.abspath(self.save_path.text + '_combo.png'), combo_image)

                if platform.system() == 'Windows':
                    realpath = str(self.dir_path) + str('\\\\result.svg')
                else:
                    realpath = str(self.dir_path) + str('/result.svg')
                file = self.save_path.text + '.svg'
                try:
                    shutil.copyfile(realpath, file)
                except shutil.SameFileError:
                    pass
                if self.popa_check == 1:
                    self.log.background_color = [0, 255, 0, 1]
                    self.log.text = "*_param.txt was prolonged, others were overwritten"
                elif self.popa_check == 2:
                    self.log.background_color = [0, 255, 0, 1]
                    self.log.text = "Results were overwritten"
                elif self.popa_check == 3:
                    self.log.background_color = [0, 255, 0, 1]
                    self.log.text = "Results were saved"

        def save_pressed(self, _instance):
            if self.realtable[0][0].text != '':
                if os.path.exists(self.save_path.text+ '_param.txt') == True:
                    self.popa_check = 0
                    self.popupWindow = Popup(title="File already exist", size_hint=(None, None), size=(300, 450))
                    self.popupWindow.bind(on_dismiss=self.save_continue)
                    layout = BoxLayout(orientation="vertical", spacing=5)
                    btn1 = Button(text="Continue parameter file\nothers will be overwriten", color = [0, 0, 0, 1], background_color=[0, 255, 0, 1], font_size='21sp')
                    btn2 = Button(text="Overwrite all result files", color=[0, 0, 0, 1], background_color=[255, 0, 0, 1], font_size='21sp')
                    btn3 = Button(text="Do nothing", color=[0, 0, 0, 1], background_color=[0, 255, 255, 1], font_size='21sp')
                    btn1.on_press = partial(self.save_way_chooser, 1)
                    btn2.on_press = partial(self.save_way_chooser, 2)
                    btn3.on_press = partial(self.save_way_chooser, 0)
                    layout.add_widget(btn1)
                    layout.add_widget(btn2)
                    layout.add_widget(btn3)
                    self.popupWindow.content = layout
                    self.popupWindow.open()

                    self.path_checker(whatissaved=1)

                    if self.newfilename != '':
                        btn4 = Button(text=str("Change name to:\n" + os.path.basename(self.newfilename)),
                                      color=[0, 0, 0, 1], background_color=[3, 2, 0, 1], font_size='20sp')
                        btn4.on_press = partial(self.save_way_chooser, 4)
                        layout.add_widget(btn4)

                    if self.newfilename2 != self.newfilename:
                        btn5 = Button(text=str("Change name to:\n" + os.path.basename(self.newfilename2)), color=[0, 0, 0, 1], background_color=[3, 2, 0, 1], font_size='20sp')
                        btn5.on_press = partial(self.save_way_chooser, 5)
                        layout.add_widget(btn5)

                    self.popupWindow.content = layout
                    self.popupWindow.open()
                else:
                    self.popa_check = 3
                    self.save_continue()

            else:
                self.log.background_color = [255, 0, 0, 0.9]
                self.log.text = "Void should not be saved"

        def save_as_pressed(self, _instance):

            raw0_path = filechooser.save_file(path=self.workfolder, title="Where to save the result files?", multiple=False)
            if raw0_path != [] and raw0_path != None:
                self.save_path.text = str(raw0_path)[2:-2]
                self.save_pressed(self)
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Saving was canceled or path is too long"
        #
        # def savemod_pressed_old(self, _instance):
        #
        #     if os.path.exists(self.save_path.text) == False:
        #         try:
        #             f = open(self.save_path.text + '_model.mdl', "a", encoding="utf-8")
        #             f.close
        #         except FileNotFoundError:
        #             v = 0
        #             for i in range(0, len(str(self.save_path.text))):
        #                 if str(self.save_path.text)[-1 - i] == '\\' or str(self.save_path.text)[-1 - i] == '/':
        #                     save_dir_path = str(self.save_path.text)[:-1 - i]
        #                     break
        #             os.makedirs(save_dir_path)
        #     file = self.save_path.text + '_model.mdl'
        #     f = open(file, "w", encoding="utf-8") #write to file output values
        #
        #     # f.write('#')
        #     for i in range(0, len(self.realtableP)-1):
        #         f.write(self.realtableP[i][0].text + '\t')
        #     f.write('\n')
        #     for k in range(0, len(self.realtableP)-1):
        #         # for i in range(0, len(self.realtableP[0])):
        #         #     f.write(self.realtableP[k][i].text + '\t')
        #         #     print(i)
        #         # f.write('\n')
        #         # for i in range(0, len(self.lboundstable[0])):
        #         #     f.write(self.lboundstable[k][i].text + '\t' + self.rboundstable[k][i].text + '\t'\
        #         #          +  self.nametable[k][i].text + '\t' +  str(self.fixtable[k][i].active) + '\t')
        #         # f.write('\n')
        #         for i in range(0, len(self.realtableP[0])-1):
        #             f.write(self.realtableP[k][i+1].text + '\t' + self.lboundstable[k][i].text + '\t' + self.rboundstable[k][i].text + '\t' \
        #                     + self.nametable[k][i].text + '\t' + str(self.fixtable[k][i].active) + '\t' )
        #         f.write('\n')
        #     f.close()

        def savemod_continue(self, *args):

            if self.popa_check == 4:
                self.save_path.text = self.newfilename
                self.popa_check = 3
            if self.popa_check == 5:
                self.save_path.text = self.newfilename2
                self.popa_check = 3
            # if os.path.exists(self.save_path.text) == False:
            if self.popa_check == 3:
                try:
                    f = open(self.save_path.text + '_model.mdl', "a", encoding="utf-8")
                    f.close
                except FileNotFoundError:
                    v = 0
                    for i in range(0, len(str(self.save_path.text))):
                        if str(self.save_path.text)[-1 - i] == '\\' or str(self.save_path.text)[-1 - i] == '/':
                            save_dir_path = str(self.save_path.text)[:-1 - i]
                            break
                    os.makedirs(save_dir_path)

            if self.popa_check == 3 or self.popa_check == 2:
                file = self.save_path.text + '_model.mdl'
                f = open(file, "w", encoding="utf-8")  # write to file output values

                for i in range(0, len(self.realtableP) - 1):
                    f.write(self.realtableP[i][0].text + '\t')
                f.write('\n')
                for k in range(0, len(self.realtableP) - 1):
                    for i in range(0, len(self.realtableP[0]) - 1):
                        f.write(self.realtableP[k][i + 1].text + '\t' + self.lboundstable[k][i].text + '\t' +
                                self.rboundstable[k][i].text + '\t' \
                                + self.nametable[k][i].text + '\t' + str(self.fixtable[k][i].active) + '\t')
                    f.write('\n')
                f.close()
                self.log.background_color = [0, 255, 0, 1]
                self.log.text = "Model was saved"

            if self.popa_check == 0:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Saving was skipped"


        def savemod_pressed(self, _instance):
            if os.path.exists(self.save_path.text + '_model.mdl') == True:
                self.popa_check = 0
                self.popupWindow = Popup(title="File already exist", size_hint=(None, None), size=(350, 300))
                self.popupWindow.bind(on_dismiss=self.savemod_continue)
                layout = BoxLayout(orientation="vertical", spacing=4)
                btn2 = Button(text="Overwrite", color=[0, 0, 0, 1], background_color=[255, 0, 0, 1], font_size='21sp')
                btn3 = Button(text="Do nothing", color=[0, 0, 0, 1], background_color=[0, 255, 255, 1], font_size='21sp')
                btn2.on_press = partial(self.save_way_chooser, 2)
                btn3.on_press = partial(self.save_way_chooser, 0)
                layout.add_widget(btn2)
                layout.add_widget(btn3)

                self.path_checker(whatissaved=0)

                if self.newfilename != '':
                    btn4 = Button(text=str("Change name to:\n" + os.path.basename(self.newfilename)), color=[0, 0, 0, 1], background_color=[3, 2, 0, 1], font_size='20sp')
                    btn4.on_press = partial(self.save_way_chooser, 4)
                    layout.add_widget(btn4)

                if self.newfilename2 != self.newfilename:
                    btn5 = Button(text=str("Change name to:\n" + os.path.basename(self.newfilename2)), color=[0, 0, 0, 1], background_color=[3, 2, 0, 1], font_size='20sp')
                    btn5.on_press = partial(self.save_way_chooser, 5)
                    layout.add_widget(btn5)

                self.popupWindow.content = layout
                self.popupWindow.open()




                # self.save_path.text
                # self.path_list



            else:
                self.popa_check = 3
                self.savemod_continue()

        def savemod_as_pressed(self, _instance):

            raw0_path = filechooser.save_file(path=self.workfolder, title="Where to save the model?", multiple=False)
            if raw0_path != [] and raw0_path != None:
                self.save_path.text = str(raw0_path)[2:-2]
                self.savemod_pressed(self)
            else:
                self.log.background_color = [255, 255, 0, 1]
                self.log.text = "Saving was canceled or path is too long"

        def path_checker(self, whatissaved):
            additional_text = str('_model.mdl')*(whatissaved==0) + str('_param.txt')*(whatissaved==1)
            openfilename = (os.path.basename(self.path_list[0]).split('.')[0])
            # openfilename = os.path.basename(self.path_list[0])
            newnumber = str('')
            if len(openfilename) == 3:
                try:
                    newnumber = str(int(openfilename))
                    while len(newnumber) < 3:
                        newnumber = str('0') + newnumber
                except:
                    pass

            olldnumber = str('')
            st_num = -1
            check = 0
            savefilename = (os.path.basename(self.save_path.text))
            for i in range(0, len(savefilename) - 2):
                if check == 0:
                    try:
                        oldnumber = str(int(savefilename[i:i + 3]))
                        st_num = i
                        check = 1
                    except:
                        pass

            self.newfilename = ''
            if st_num != -1 and newnumber != str(''):
                self.newfilename = savefilename[0:st_num] + newnumber + savefilename[st_num + 3:]
                while os.path.exists(self.newfilename + additional_text) == True:
                    if self.newfilename[-2] == '_':
                        try:
                            self.newfilename = self.newfilename[:-1] + str(int(self.newfilename[-1]) + 1)
                        except:
                            self.newfilename = self.newfilename + '_2'
                            break
                    else:
                        self.newfilename = self.newfilename + '_2'
                self.newfilename = self.save_path.text[:-len(os.path.basename(self.save_path.text))] + self.newfilename

            self.newfilename2 = self.save_path.text
            while os.path.exists(self.newfilename2 + additional_text) == True:
                if self.newfilename2[-2] == '_':
                    try:
                        self.newfilename2 = self.newfilename2[:-1] + str(int(self.newfilename2[-1]) + 1)
                    except:
                        self.newfilename2 = self.newfilename2 + '_2'
                        break
                else:
                    self.newfilename2 = self.newfilename2 + '_2'



        def spc_changes(self, text, *args):
            if initial == 0:
                Tini = threading.Thread(target=partial(initialization, self))
                Tini.name = 'T_ini'
                Tini.daemon = True
                Tini.start()
                Tini.join()
            if text == 'Sum all\nspectra':
                self.sum_spectra(self)
            if text == 'Substract\nmodel from\nspectrum':
                self.substruct_model_from_spectrum(self)
            if text == "Half points":
                self.sum_points(self)


        def sum_spectra(self, _instance):
            v = 0
            vv = 0
            raw_path = self.process_path.text[2:-2]
            print(raw_path)
            print(str(raw_path)[-1])
            self.path_list = []
            check_sum = 0

            if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/':
                try:
                    for file in os.listdir(str(raw_path)):
                        # if file.endswith(".txt"):
                        #     self.path_list.append(os.path.join(str(raw_path), file))
                        if file.endswith(".dat"):
                            self.path_list.append(os.path.join(str(raw_path), file))
                    for i in range(0, len(self.path_list)):
                        if os.path.exists(self.path_list[i]) == False:
                            check = False
                            self.log.background_color = [255, 255, 0, 1]
                            self.log.text = "At least one path do not exist"
                            check_sum = 1
                    # if ("." in str(self.save_path.text)) == True:
                    #     check = False
                    #     self.log.background_color = [255, 255, 0, 1]
                    #     self.log.text = "Directory path could not contain dots"
                    dir_ch = 0
                    raw_path_ch = (raw_path.replace("\\\\", "\\"))[:-1]
                    dir_path_ch = os.path.dirname(str(self.save_path.text))
                    print(raw_path_ch)
                    print(dir_path_ch)
                    # if raw_path_ch == dir_path_ch:
                    #     dir_ch = 1

                    # if (str(self.save_path.text)[-1] != '\\' or str(self.save_path.text)[-1] != '/') and dir_ch != 0:
                    #     self.save_path.text = str(self.save_path.text) + str('\\') * (
                    #                 platform.system() == 'Windows') + str('/') * (platform.system() != 'Windows')
                    #
                    # if raw_path_ch == self.save_path.text:
                    #     self.save_path.text = self.save_path.text + str('result') \
                    #                           + str('\\') * (platform.system() == 'Windows') + str('/') * (
                    #                                       platform.system() != 'Windows')



                except:
                    check = False
                    self.log.background_color = [255, 255, 0, 1]
                    self.log.text = "Directory do not exist"
                    check_sum = 1
            else:
                for i in range(0, len(raw_path) - 2):
                    if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                        vv = i
                        self.path_list.append(str(raw_path)[v:vv])
                        v = i + 4
                self.path_list.append(str(raw_path)[v:])
                check = True
                for i in range(0, len(self.path_list)):
                    if os.path.exists(self.path_list[i]) == False:
                        check = False
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "At least one path do not exist"
                        check_sum = 1

            if check_sum == 0:
                A_sum, B_sum = read_spectrum(self, os.path.abspath(self.path_list[0]))
                for l_p in range(1, len(self.path_list)):
                    file = os.path.abspath(self.path_list[l_p])
                    A, B = read_spectrum(self, file)
                    try:
                        B_sum += B
                    except:
                        self.log.background_color = [255, 0, 0, 1]
                        self.log.text = "Could not sum, please check files lengths"
                        check_sum = 1


            if check_sum == 0:
                realpath = str(self.dir_path) + str('\\\\SUM.txt')*(platform.system() == 'Windows') + str('/SUM.txt')*(platform.system() != 'Windows')
                f = open(realpath, "w")
                for i in range(0, len(A_sum)):
                    f.write(str(A_sum[i]) + '\t' + str(B_sum[i]) + '\n')
                f.close()

                raw0_path = filechooser.save_file(path=self.workfolder, title="Where to save the sum?", multiple=False)
                if raw0_path != [] and raw0_path != None:
                    raw0_path = raw0_path[0] + str('.dat')

                    try:
                        shutil.copyfile(realpath, raw0_path)
                    except shutil.SameFileError:
                        pass

                    self.process_path.text = str('[\'') + raw0_path + str('\']')
                    self.show_pressed(self)
                else:
                    self.log.background_color = [255, 255, 0, 1]
                    self.log.text = "Sum was canceled or path is too long"

        def substruct_model_from_spectrum(self, _instance):
            raw_path = self.process_path.text[2:-2]
            if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/':
                self.log.background_color = [255, 0, 0, 1]
                self.log.text = "Please select ONE spectrum for this operation"
            else:
                v = 0
                vv = 0
                self.path_list = []
                for i in range(0, len(raw_path) - 2):
                    if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                        vv = i
                        self.path_list.append(str(raw_path)[v:vv])
                        v = i + 4
                self.path_list.append(str(raw_path)[v:])
                if len(self.path_list) != 1:
                    self.log.background_color = [255, 0, 0, 1]
                    self.log.text = "Please select ONE spectrum for this operation"
                else:
                    if os.path.exists(self.path_list[0]) == False:
                        self.log.background_color = [255, 0, 0, 1]
                        self.log.text = "File do not exist"
                    else:
                        A, B = read_spectrum(self, os.path.abspath(self.path_list[0]))
                        global val
                        global model
                        global p

                        con1, con2, con3, Distri, Cor, Expr, NExpr, fix_tmp, DistriN = read_model(self)

                        for i in range(0, len(NExpr)):
                            p[NExpr[i]] = eval(Expr[i])
                        for i in range(0, len(con1)):
                            p[int(con1[i])] = p[int(con2[i])] * con3[i]

                        confu = np.array([con1, con2, con3])

                        global p0, er

                        JN = int(self.JN0.text)

                        if self.switch.active == True:
                            if self.fitway[0].active == True:
                                INS = float(self.L0.text)
                                pNorm = np.array([float(0)] * NBA)
                                pNorm[0] = 1
                                Norm = \
                                m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, 0.0, MulCoCMS, INS, [0], [0], Met=1)[0]
                                print('Normalization integral equal to', Norm)
                                def func(x, p):
                                    return m5.TI(x, p, model, JN, pool, 0.0, MulCoCMS, INS, Distri, Cor, Met=1, Norm=Norm)
                            if self.fitway[1].active == True:
                                if platform.system() == 'Windows':
                                    realpath = str(self.dir_path) + str('\\\\INSexp.txt')
                                else:
                                    realpath = str(self.dir_path) + str('/INSexp.txt')
                                INS = np.genfromtxt(realpath, delimiter=' ', skip_footer=0)
                                pNorm = np.array([float(0)] * NBA)
                                pNorm[0] = 1
                                Norm = m5.TI(np.array([float(1000)]), pNorm, [], JN, pool, x0, MulCo, INS, [0], [0])[0]
                                print('Normalization integral equal to', Norm)
                                def func(x, p):
                                    return m5.TI(x, p, model, JN, pool, x0, MulCo, INS, Distri, Cor, Norm=Norm)

                            if self.realtableP[0][1].text[0] != '=':
                                Res = float(self.realtableP[0][1].text)
                            else:
                                pNM = (self.realtableP[0][1].text[2:-1]).split(',')
                                pN, pM = int(pNM[0]), float(pNM[1])
                                Res = p[pN] * pM

                            if self.realtableP[0][1].text[0] != '=':
                                NoRes = float(self.realtableP[0][5].text)
                            else:
                                pNM = (self.realtableP[0][5].text[2:-1]).split(',')
                                pN, pM = int(pNM[0]), float(pNM[1])
                                NoRes = p[pN] * pM

                            print(model)
                            print(p)
                            SPC_f = func(A, p)
                            B = B - SPC_f + Res + NoRes

                            realpath = str(self.dir_path) + str('\\\\SUM.txt') * (platform.system() == 'Windows')\
                                       + str('/SUB.txt') * (platform.system() != 'Windows')
                            f = open(realpath, "w")
                            for i in range(0, len(A)):
                                f.write(str(A[i]) + '\t' + str(B[i]) + '\n')
                            f.close()

                            raw0_path = filechooser.save_file(path=self.workfolder, title="Where to save the difference?", multiple=False)
                            if raw0_path != [] and raw0_path != None:
                                raw0_path = raw0_path[0] + str('.dat')

                                try:
                                    shutil.copyfile(realpath, raw0_path)
                                except shutil.SameFileError:
                                    pass

                                self.process_path.text = str('[\'') + raw0_path + str('\']')
                                self.show_pressed(self)
                            else:
                                self.log.background_color = [255, 255, 0, 1]
                                self.log.text = "Difference was canceled or path is too long"


                        if self.switch.active == False:
                            VVV = 4
                            self.log.background_color = [255, 0, 0, 1]
                            self.log.text = "It is unacceptable for NFS!"

        def sum_points(self, _instance):
            raw_path = self.process_path.text[2:-2]
            if str(raw_path)[-1] == '\\' or str(raw_path)[-1] == '/':
                self.log.background_color = [255, 0, 0, 1]
                self.log.text = "Please select ONE spectrum for this operation"
            else:
                v = 0
                vv = 0
                self.path_list = []
                for i in range(0, len(raw_path) - 2):
                    if str(raw_path)[i] == '\'' and str(raw_path)[i + 1] == ',' and str(raw_path)[i + 2] == ' ':
                        vv = i
                        self.path_list.append(str(raw_path)[v:vv])
                        v = i + 4
                self.path_list.append(str(raw_path)[v:])
                if len(self.path_list) != 1:
                    self.log.background_color = [255, 0, 0, 1]
                    self.log.text = "Please select ONE spectrum for this operation"
                else:
                    A, B = read_spectrum(self, os.path.abspath(self.path_list[0]))
                    A_sum = np.array([float(0)]*int(len(A)/2))
                    B_sum = np.array([float(0)]*int(len(B)/2))
                    for i in range(0, int(len(A)/2)):
                        A_sum[i] = (A[i*2] + A[i*2+1])/2
                        B_sum[i] = B[i*2] + B[i*2+1]

                    realpath = str(self.dir_path) + str('\\\\tmp.txt') * (platform.system() == 'Windows')\
                               + str('/tmp.txt') * (platform.system() != 'Windows')
                    f = open(realpath, "w")
                    for i in range(0, len(A_sum)):
                        f.write(str(A_sum[i]) + '\t' + str(B_sum[i]) + '\n')
                    f.close()

                    raw0_path = filechooser.save_file(path=self.workfolder, title="Where to save the sum?", multiple=False)
                    if raw0_path != [] and raw0_path != None:
                        raw0_path = raw0_path[0] + str('.dat')
                        try:
                            shutil.copyfile(realpath, raw0_path)
                        except shutil.SameFileError:
                            pass

                        self.process_path.text = str('[\'') + raw0_path + str('\']')
                        self.show_pressed(self)
                    else:
                        self.log.background_color = [255, 255, 0, 1]
                        self.log.text = "Sum was canceled or path is too long"


        def replot_result(self, numb, _instance):
            # self.X_axis = []
            # self.Y_axis = []
            # self.Z_order = []
            # self.Color_order = []
            # self.SPC_plot = []
            # self.SPC_numb = []
            # self.Model_full_plot = []
            # self.Baseline_plot = []
            # self.Integral_line_plot = []
            color_check = 0
            if numb == -1:
                color_check = 1
                numb = 2

            check_color = 0
            for i in range(0, len(self.realtable)-2):
                if self.realtable[i][0].color != self.realtable[i+1][0].color:
                    check_color = 1
                    break

            if self.realtable[numb][0].text != '' and self.realtable[numb][0].text != 'baseline' and self.realtable[numb][0].text != 'Nbaseline' \
                    and self.realtable[numb][0].text != 'Expression' and self.realtable[numb][0].text != 'Variables'\
                    and self.switch.active == True and check_color == 1:
                try:
                    number_of_spectra = len(self.Z_order)
                    Dim2_tmp = len(self.Z_order[0])
                    print('z_oreder is fine')
                except:
                    self.X_axis = [self.X_axis]
                    self.Y_axis = [self.Y_axis]
                    self.Z_order = [self.Z_order]
                    self.Color_order = [self.Color_order]
                    self.SPC_plot = [self.SPC_plot]
                    self.SPC_numb = [self.SPC_numb]
                    self.Model_full_plot = [self.Model_full_plot]
                    self.Baseline_plot = [self.Baseline_plot]
                    self.Integral_line_plot = [self.Integral_line_plot]
                    number_of_spectra = 1
                    print('z_oreder is bad')

                print(self.Z_order)

                numb = int(numb/2)
                for i in range(1, numb+1):
                    numb -= (self.realtable[i * 2][0].text == 'Distr') + (self.realtable[i * 2][0].text == 'Corr') + (self.realtable[i * 2][0].text == 'Nbaseline')
                numb -= 1 # minus baseline
                print('numb', numb)
                dim1_counter = 0
                dim2_counter = 0
                dim3_counter = 0
                for i in range(0, number_of_spectra):
                    if dim2_counter == numb:
                        break
                    else:
                        for j in range(0, len(self.Z_order[i])):
                            if dim2_counter+j == numb:
                                dim2_counter += j
                                dim3_counter += j
                                break
                        if dim2_counter != numb:
                            dim2_counter += len(self.Z_order[i])
                            dim1_counter += 1
                            dim3_counter = 0
                print('dim_counters', dim1_counter, dim2_counter, dim3_counter)
                if color_check == 0:
                    self.Z_order[dim1_counter][dim3_counter] = max(self.Z_order[dim1_counter]) + 1

                global RL
                RL = 1
                # self.play_btn.background_color = [0, 0.5, 0, 1]
                # self.showM_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.show_btn.background_color = [0.5, 0.5, 0.5, 0.5]
                # self.INS_btn.background_color = [0.2, 0.2, 0.2, 1]
                # self.INS_btn2.background_color = [0.2, 0.2, 0.2, 1]
                # self.play_btn.disabled = True
                # self.INS_btn.disabled = True
                # self.INS_btn2.disabled = True
                # self.show_btn.disabled = True
                # self.showM_btn.disabled = True
                # self.cal_btn.disabled = True
                # self.switch.disabled = True

                try:
                    fig, ax1 = plt.subplots(figsize=(2942 / 300 * number_of_spectra, 4.5), dpi=300)
                    # fig = plt.figure(figsize=(2942/300*number_of_spectra, 4.5), dpi=300)

                    for NumSpc in range(0, number_of_spectra):
                        ax = plt.subplot(1, number_of_spectra, NumSpc + 1)

                        skip_step = 0
                        for i in range(0, len(self.Y_axis[NumSpc])):
                            plt.plot(self.X_axis[NumSpc], self.Y_axis[NumSpc][i], color=self.Color_order[NumSpc][i], zorder=self.Z_order[NumSpc][i])

                            plt.fill_between(self.X_axis[NumSpc], self.Baseline_plot[NumSpc], self.Y_axis[NumSpc][i], facecolor=self.Color_order[NumSpc][i], zorder=self.Z_order[NumSpc][i])

                            if len(self.FS_pos[i][0]) != 0:
                                minpos = self.FS_pos[i][0][0]
                                maxpos = self.FS_pos[i][0][0]
                                H_step = (max(self.SPC_plot[NumSpc]) - min(self.SPC_plot[NumSpc])) * 0.04
                                for j in range(0, len(self.FS_pos[i][0])):
                                    if self.FS_pos[i][0][j] < minpos:
                                        minpos = self.FS_pos[i][0][j]
                                    if self.FS_pos[i][0][j] > maxpos:
                                        maxpos = self.FS_pos[i][0][j]
                                plt.plot([minpos, maxpos],
                                         [max(self.SPC_plot[NumSpc]) + H_step * (1 + (i - skip_step) * 2),
                                          max(self.SPC_plot[NumSpc]) + H_step * (1 + (i - skip_step) * 2)],
                                         color=self.Color_order[NumSpc][i],
                                         zorder=self.Z_order[NumSpc][i])
                                for j in range(0, len(self.FS_pos[i][0])):
                                    plt.plot([self.FS_pos[i][0][j], self.FS_pos[i][0][j]],
                                             [max(self.SPC_plot[NumSpc]) + H_step * ((i - skip_step) * 2),
                                              max(self.SPC_plot[NumSpc]) + H_step * (1 + (i - skip_step) * 2)],
                                             color=self.Color_order[NumSpc][i],
                                             zorder=self.Z_order[NumSpc][i])
                            else:
                                skip_step += 1



                        plt.plot(self.X_axis[NumSpc], self.SPC_plot[NumSpc] - self.Model_full_plot[NumSpc] + min(self.SPC_plot[NumSpc]) - max(self.SPC_plot[NumSpc] - self.Model_full_plot[NumSpc]), color='lime')
                        plt.plot(self.X_axis[NumSpc], self.SPC_plot[NumSpc] - self.SPC_plot[NumSpc] + min(self.SPC_plot[NumSpc]) - max(self.SPC_plot[NumSpc] - self.Model_full_plot[NumSpc]), linestyle='--', color=self.gridcolor)

                        plt.xlim(min(self.X_axis[NumSpc]), max(self.X_axis[NumSpc]))
                        plt.grid(color=self.gridcolor, linestyle=(0, (1, 10)), linewidth=1)
                        plt.plot(self.X_axis[NumSpc], self.Model_full_plot[NumSpc], color='r', zorder=max(self.Z_order[NumSpc])+1)
                        plt.plot(self.X_axis[NumSpc], self.SPC_plot[NumSpc], linestyle='None', marker='x', color='m', zorder=max(self.Z_order[NumSpc])+2)
                        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        plt.ylabel(str('Transmission, counts')*(NumSpc==0))
                        plt.xlabel('Velocity, mm/s')


                        plt.text(0, -0.1, self.SPC_numb[NumSpc], horizontalalignment='left', verticalalignment='center', color='m', transform=ax.transAxes)
                        plt.plot(self.X_axis[NumSpc], self.Integral_line_plot[NumSpc], color='cyan')
                        if NumSpc == int(number_of_spectra / 2):
                            plt.title('χ² = %.3f' % hi2, y=1, color='r')
                        if max(self.Baseline_plot[NumSpc]) == min(self.Baseline_plot[NumSpc]):
                            ymin, ymax = ax.get_ylim()
                            ax2 = ax.twinx()
                            ax2.set_ylim((ymin / min(self.Baseline_plot[NumSpc]),
                                          ymax / min(self.Baseline_plot[NumSpc])))

                    if platform.system() == 'Windows':
                        realpath = str(self.dir_path) + str('\\\\result.svg')
                    else:
                        realpath = str(self.dir_path) + str('/result.svg')
                    fig.savefig(realpath, bbox_inches='tight')
                    if platform.system() == 'Windows':
                        realpath = str(self.dir_path) + str('\\\\result.png')
                    else:
                        realpath = str(self.dir_path) + str('/result.png')
                    fig.savefig(realpath, bbox_inches='tight')
                    imgpath = realpath
                    # plt.close()
                    plt.cla()
                    plt.clf()
                    plt.close('all')
                    plt.close(fig)
                    gc.collect()

                    global RA
                    RA = True


                except:
                    self.log.text = 'Problem with creating a picture. If it is open please close it.'
                    self.log.background_color = [255, 0, 0, 1]

                RL = 6
                # self.play_btn.disabled = False
                # self.INS_btn.disabled = False
                # self.INS_btn2.disabled = False
                # self.show_btn.disabled = False
                # self.showM_btn.disabled = False
                # self.cal_btn.disabled = False
                # self.switch.disabled = False

        def change_image(self, _instance):

            if self.SP_DI.text == 'Distribution':
                self.SP_DI.text = 'Spectrum'
            else:
                self.SP_DI.text = 'Distribution'
            global RA
            RA = True

        def show_par_number(self, i,j):
            global old_par_name
            N = 0

            for k in range(0, len(self.realtableP)-1):
                for kk in range(0, len(self.realtableP[0]) - 1):
                    NN = N + kk
                    if self.nametable[k][kk].text == 'p[%.i]' % NN:
                        self.nametable[k][kk].text = old_par_name
                # C = self.realtableP[k][0].text
                N += int(NBA*(k==0)) + mod_len_def(self.realtableP[k][0].text)



            if self.nametable[i][j].text != '':
                N = 0
                for k in range(0, i):
                    # C = self.realtableP[k][0].text
                    N += int(NBA*(k==0)) + mod_len_def(self.realtableP[k][0].text)
                N += j
                old_par_name = self.nametable[i][j].text
                self.nametable[i][j].text = 'p[%.i]' % N
        def hide_par_number(self, i,j):
            global old_par_name
            if self.nametable[i][j].text != '':
                self.nametable[i][j].text = old_par_name

        def change_visiual(self, _instance):
            if self.dark_light_mode.text == 'Light spc':
                self.BGcolor = 'w'
                plt.rcParams['axes.facecolor'] = '(1, 1, 1)'
                plt.rcParams['figure.facecolor'] = '(1, 1, 1)'
                plt.rcParams['axes.labelcolor'] = 'k'
                plt.rcParams['axes.edgecolor'] = 'k'
                plt.rcParams['xtick.color'] = 'k'
                plt.rcParams['ytick.color'] = 'k'
                self.gridcolor = 'k'
                self.dark_light_mode.text = 'Dark spc'
                self.dark_light_mode.background_color = [0.5, 0.5, 0.5, 0.5]

            elif self.dark_light_mode.text == 'Dark spc':
                self.BGcolor = 'k'
                plt.rcParams['axes.facecolor'] = '(0, 0, 0)'
                plt.rcParams['figure.facecolor'] = '(0, 0, 0)'
                plt.rcParams['axes.labelcolor'] = 'w'
                plt.rcParams['axes.edgecolor'] = 'w'
                plt.rcParams['xtick.color'] = 'w'
                plt.rcParams['ytick.color'] = 'w'
                self.gridcolor = 'w'
                self.dark_light_mode.text = 'Light spc'
                self.dark_light_mode.background_color = [1.5, 1.5, 1.5, 1.5]

            self.replot_result(-1, 0)

        def fix_all(self, row_number):
            if self.fix_table_ch[row_number] == 0:
                self.fix_table_ch[row_number] = 1
                for i in range(0, numco):
                    self.fix_memory_table[row_number][i] = self.fixtable[row_number][i].active
                    self.fixtable[row_number][i].active = True
                self.startlabel2[row_number].text = 'unfix model'
                self.startlabel2[row_number].color = 'tomato'
            elif self.fix_table_ch[row_number] == 1:
                self.fix_table_ch[row_number] = 0
                for i in range(0, numco):
                    self.fixtable[row_number][i].active = self.fix_memory_table[row_number][i]
                self.startlabel2[row_number].text = 'fix model'
                self.startlabel2[row_number].color = 'cyan'


        def on_quit(self, *args, source=0):
            global pool
            PhysicsApp().stop()
            pool.close()
            pool.join()
            print('pool is closed')
            os.kill(os.getpid(), signal.SIGTERM)
            exit()
            return True

        #dissable F1 button
        def open_settings(self, *largs):
            pass



        # def on_request_close(self):
        #     global pool
        #     PhysicsApp().stop()
        #     pool.close()
        #     pool.join()
        #     print('pool is closed')
        #     os.kill(os.getpid(), signal.SIGTERM)
        #     exit()
        #     return True


    # error_log_file = os.getcwd() + ("\\error_log.txt") * (platform.system() == 'Windows')\
    #                              + ("/error_log.txt")  * (platform.system() != 'Windows')
    #
    # sys.stdout = open(error_log_file, 'w')
    # sys.stdout.close()

    PhysicsApp().run()


