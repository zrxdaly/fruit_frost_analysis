#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:10:01 2020

OBJ: using the function to get data from vrlab, then analyse the data offline

@author: dai
"""
#%%
# import matplotlib
# matplotlib.use("Agg")

import numpy as np
import glob as glob

         
#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/ROTA_LD/Full/"
# dir1 = '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/'
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
    return(T_hub*T_ref/G + T_ref - KCR)

TC_hub = K2C(T_hub)

Dt = 1    # s  the time interval
T0 = 60     # s  time length for no operation
T1 = 1020    # s  time length for fan operation

#%% get the reference state of 3D data at each height
def D2_N(filedir, Tstart, Tend):
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

    FR_Bu05N = np.loadtxt(FR_05[Tstart], dtype='f',skiprows=2)
    FR_Bu10N = np.loadtxt(FR_10[Tstart], dtype='f',skiprows=2)
    FR_Bu15N = np.loadtxt(FR_15[Tstart], dtype='f',skiprows=2)
    FR_Bu20N = np.loadtxt(FR_20[Tstart], dtype='f',skiprows=2)
    FR_Bu25N = np.loadtxt(FR_25[Tstart], dtype='f',skiprows=2)
    FR_Bu30N = np.loadtxt(FR_30[Tstart], dtype='f',skiprows=2)
    FR_Bu35N = np.loadtxt(FR_35[Tstart], dtype='f',skiprows=2)
    FR_Bu40N = np.loadtxt(FR_40[Tstart], dtype='f',skiprows=2)
    FR_Bu45N = np.loadtxt(FR_45[Tstart], dtype='f',skiprows=2)
    FR_Bu50N = np.loadtxt(FR_50[Tstart], dtype='f',skiprows=2)

    for i in np.arange(Tstart+1, Tend):
        FR_Bu05N = FR_Bu05N + np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = FR_Bu10N + np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = FR_Bu15N + np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = FR_Bu20N + np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = FR_Bu25N + np.loadtxt(FR_25[i], dtype='f',skiprows=2)
        FR_Bu30N = FR_Bu30N + np.loadtxt(FR_30[i], dtype='f',skiprows=2)
        FR_Bu35N = FR_Bu35N + np.loadtxt(FR_35[i], dtype='f',skiprows=2)
        FR_Bu40N = FR_Bu40N + np.loadtxt(FR_40[i], dtype='f',skiprows=2)
        FR_Bu45N = FR_Bu45N + np.loadtxt(FR_45[i], dtype='f',skiprows=2)
        FR_Bu50N = FR_Bu50N + np.loadtxt(FR_50[i], dtype='f',skiprows=2)

    FR_Bu05N_TA = K2C(FR_Bu05N/(Tend - Tstart))
    FR_Bu10N_TA = K2C(FR_Bu10N/(Tend - Tstart))
    FR_Bu15N_TA = K2C(FR_Bu15N/(Tend - Tstart))
    FR_Bu20N_TA = K2C(FR_Bu20N/(Tend - Tstart))
    FR_Bu25N_TA = K2C(FR_Bu25N/(Tend - Tstart))
    FR_Bu30N_TA = K2C(FR_Bu30N/(Tend - Tstart))
    FR_Bu35N_TA = K2C(FR_Bu35N/(Tend - Tstart))
    FR_Bu40N_TA = K2C(FR_Bu40N/(Tend - Tstart))
    FR_Bu45N_TA = K2C(FR_Bu45N/(Tend - Tstart))
    FR_Bu50N_TA = K2C(FR_Bu50N/(Tend - Tstart))

    D12 = np.array([FR_Bu05N_TA,FR_Bu10N_TA,FR_Bu15N_TA,FR_Bu20N_TA,FR_Bu25N_TA,
                FR_Bu30N_TA,FR_Bu35N_TA,FR_Bu40N_TA,FR_Bu45N_TA,FR_Bu50N_TA])
    
    return(D12)
# reference state of the buoyancy data (the temp differences over height)
DR_R060 = D2_N(R060, 0, T0)
DR_R080 = D2_N(R080, 0, T0)
DR_R096 = D2_N(R096, 0, T0)
DR_R120 = D2_N(R120, 0, T0)
DR_R160 = D2_N(R160, 0, T0)
DR_R192 = D2_N(R192, 0, T0)
DR_R240 = D2_N(R240, 0, T0)
DR_R320 = D2_N(R320, 0, T0)
DR_R480 = D2_N(R480, 0, T0)

#%% get the time averaged data during operation
DO_R060 = D2_N(R060, T0, T1)
DO_R080 = D2_N(R080, T0, T1)
DO_R096 = D2_N(R096, T0, T1)
DO_R120 = D2_N(R120, T0, T1)
DO_R160 = D2_N(R160, T0, T1)
DO_R192 = D2_N(R192, T0, T1)
DO_R240 = D2_N(R240, T0, T1)
DO_R320 = D2_N(R320, T0, T1)
DO_R480 = D2_N(R480, T0, T1)
# operational aeverage data
OAEF_R060 = (DO_R060-DR_R060)/(TC_hub - DR_R060)*100 
OAEF_R080 = (DO_R080-DR_R080)/(TC_hub - DR_R080)*100
OAEF_R096 = (DO_R096-DR_R096)/(TC_hub - DR_R096)*100
OAEF_R120 = (DO_R120-DR_R120)/(TC_hub - DR_R120)*100
OAEF_R160 = (DO_R160-DR_R160)/(TC_hub - DR_R160)*100
OAEF_R192 = (DO_R192-DR_R192)/(TC_hub - DR_R192)*100
OAEF_R240 = (DO_R240-DR_R240)/(TC_hub - DR_R240)*100
OAEF_R320 = (DO_R320-DR_R320)/(TC_hub - DR_R320)*100
OAEF_R480 = (DO_R480-DR_R480)/(TC_hub - DR_R480)*100  

np.save("Full_DS/OAEF_R060.npy", OAEF_R060)
np.save("Full_DS/OAEF_R080.npy", OAEF_R080)
np.save("Full_DS/OAEF_R096.npy", OAEF_R096)
np.save("Full_DS/OAEF_R120.npy", OAEF_R120)
np.save("Full_DS/OAEF_R160.npy", OAEF_R160)
np.save("Full_DS/OAEF_R192.npy", OAEF_R192)
np.save("Full_DS/OAEF_R240.npy", OAEF_R240)
np.save("Full_DS/OAEF_R320.npy", OAEF_R320)
np.save("Full_DS/OAEF_R480.npy", OAEF_R480)

