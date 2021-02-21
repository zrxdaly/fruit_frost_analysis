#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:10:01 2020

OBJ: get time series of temp from vrlab


@author: dai
"""
#%%
import numpy as np
import glob as glob

#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/ROTA_LD/Full/"
Goal_dir = sorted(glob.glob(dir1 + 'R*/'))

R060 = Goal_dir[0]
R080 = Goal_dir[1]
R096 = Goal_dir[2]
R120 = Goal_dir[3]
R160 = Goal_dir[4]
R192 = Goal_dir[5]
R240 = Goal_dir[6]
R320 = Goal_dir[7]
R480 = Goal_dir[8]

# calculation of the max warming and reference warming
G = 9.81
T_ref = 273
KCR = 273.15
INV = 0.3
T_hub = G/T_ref*INV*10.5

def K2C(T_hub):
    # from buoyancy to celsius unit
    return(T_hub*T_ref/G + T_ref - KCR)

TC_hub = K2C(T_hub)

Dt = 1    # s  the time interval
T0 = 60     # s  time length for no operation
T1 = 1020    # s  time length for fan operation

#%% 
def VAT(filedir, Tstart, Tend):
    # 1234 represent different case
    FR_05 = sorted(glob.glob(filedir + 't=*y=005'))
    FR_10 = sorted(glob.glob(filedir + 't=*y=010'))
    FR_15 = sorted(glob.glob(filedir + 't=*y=015'))
    FR_20 = sorted(glob.glob(filedir + 't=*y=020'))
    FR_25 = sorted(glob.glob(filedir + 't=*y=025'))
    FR_30 = sorted(glob.glob(filedir + 't=*y=030'))
    FR_35 = sorted(glob.glob(filedir + 't=*y=035'))
    FR_40 = sorted(glob.glob(filedir + 't=*y=040'))
    FR_45 = sorted(glob.glob(filedir + 't=*y=045'))
    FR_50 = sorted(glob.glob(filedir + 't=*y=050'))

    FR_series_05 = []
    FR_series_10 = []
    FR_series_15 = []
    FR_series_20 = []
    FR_series_25 = []
    FR_series_30 = []
    FR_series_35 = []
    FR_series_40 = []
    FR_series_45 = []
    FR_series_50 = []
    
    FR_series_05_append = FR_series_05.append
    FR_series_10_append = FR_series_10.append
    FR_series_15_append = FR_series_15.append
    FR_series_20_append = FR_series_20.append
    FR_series_25_append = FR_series_25.append
    FR_series_30_append = FR_series_30.append
    FR_series_35_append = FR_series_35.append
    FR_series_40_append = FR_series_40.append
    FR_series_45_append = FR_series_45.append
    FR_series_50_append = FR_series_50.append


    for i in np.arange(Tstart, Tend):
        FR_Bu05N = np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = np.loadtxt(FR_25[i], dtype='f',skiprows=2)
        FR_Bu30N = np.loadtxt(FR_30[i], dtype='f',skiprows=2)
        FR_Bu35N = np.loadtxt(FR_35[i], dtype='f',skiprows=2)
        FR_Bu40N = np.loadtxt(FR_40[i], dtype='f',skiprows=2)
        FR_Bu45N = np.loadtxt(FR_45[i], dtype='f',skiprows=2)
        FR_Bu50N = np.loadtxt(FR_50[i], dtype='f',skiprows=2)
        

        FR_series_05_append(K2C(FR_Bu05N))
        FR_series_10_append(K2C(FR_Bu10N))
        FR_series_15_append(K2C(FR_Bu15N))
        FR_series_20_append(K2C(FR_Bu20N))
        FR_series_25_append(K2C(FR_Bu25N))
        FR_series_30_append(K2C(FR_Bu30N))
        FR_series_35_append(K2C(FR_Bu35N))
        FR_series_40_append(K2C(FR_Bu40N))
        FR_series_45_append(K2C(FR_Bu45N))
        FR_series_50_append(K2C(FR_Bu50N))


    FR_VTH = [FR_series_05, FR_series_10, FR_series_15, 
               FR_series_20, FR_series_25, FR_series_30, 
               FR_series_35, FR_series_40, FR_series_45,
               FR_series_50]
    return(FR_VTH)

#%% get the time averaged data during operation
#DO_R060 = VAT(R060, 0, T1)
#DO_R080 = VAT(R080, 0, T1)
#DO_R096 = VAT(R096, 0, T1)
#DO_R120 = VAT(R120, 0, T1)
#DO_R160 = VAT(R160, 0, T1)
DO_R192 = VAT(R192, 0, T1)
DO_R240 = VAT(R240, 0, T1)
#DO_R320 = VAT(R320, 0, T1)
#DO_R480 = VAT(R480, 0, T1)

#np.save("Full/TS_R060.npy", DO_R060)
#np.save("Full/TS_R080.npy", DO_R080)
#np.save("Full/TS_R096.npy", DO_R096)
#np.save("Full/TS_R120.npy", DO_R120)
#np.save("Full/TS_R160.npy", DO_R160)
np.save("Full/TS_R192.npy", DO_R192)
np.save("Full/TS_R240.npy", DO_R240)
#np.save("Full/TS_R320.npy", DO_R320)
#np.save("Full/TS_R480.npy", DO_R480)

