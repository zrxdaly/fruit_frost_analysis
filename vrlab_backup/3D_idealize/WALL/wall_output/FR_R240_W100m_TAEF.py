
import numpy as np
import glob as glob

         
#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/WALL/WALL_100m_visual/"
Goal_dir = dir1+"resultslice/buo/"

R060 = Goal_dir


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
T0 = 15     # s  time length for no operation
T1 = 255    # s  time length for fan operation

#%% get the reference state of 3D data at each height
def D2_N(filedir, Tstart, Tend):
    # 1234 represent different case
    FR_05 = sorted(glob.glob(filedir + 't=*y=001'))
    FR_10 = sorted(glob.glob(filedir + 't=*y=002'))
    FR_15 = sorted(glob.glob(filedir + 't=*y=003'))
    FR_20 = sorted(glob.glob(filedir + 't=*y=004'))
    FR_25 = sorted(glob.glob(filedir + 't=*y=005'))

    FR_Bu05N = np.loadtxt(FR_05[Tstart], dtype='f',skiprows=2)
    FR_Bu10N = np.loadtxt(FR_10[Tstart], dtype='f',skiprows=2)
    FR_Bu15N = np.loadtxt(FR_15[Tstart], dtype='f',skiprows=2)
    FR_Bu20N = np.loadtxt(FR_20[Tstart], dtype='f',skiprows=2)
    FR_Bu25N = np.loadtxt(FR_25[Tstart], dtype='f',skiprows=2)

    for i in np.arange(Tstart+1, Tend):
        FR_Bu05N = FR_Bu05N + np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = FR_Bu10N + np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = FR_Bu15N + np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = FR_Bu20N + np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = FR_Bu25N + np.loadtxt(FR_25[i], dtype='f',skiprows=2)

    FR_Bu05N_TA = K2C(FR_Bu05N/(Tend - Tstart))
    FR_Bu10N_TA = K2C(FR_Bu10N/(Tend - Tstart))
    FR_Bu15N_TA = K2C(FR_Bu15N/(Tend - Tstart))
    FR_Bu20N_TA = K2C(FR_Bu20N/(Tend - Tstart))
    FR_Bu25N_TA = K2C(FR_Bu25N/(Tend - Tstart))

    D12 = np.array([FR_Bu05N_TA,FR_Bu10N_TA,FR_Bu15N_TA,FR_Bu20N_TA,FR_Bu25N_TA])
    
    return(D12)
# reference state of the buoyancy data (the temp differences over height)
DR_R060 = D2_N(R060, 0, T0)


#%% get the time averaged data during operation
DO_R060 = D2_N(R060, T0, T1)

# operational aeverage data
OAEF_R060 = (DO_R060-DR_R060)/(TC_hub - DR_R060)*100 

np.save("FR_R240_W100m_TAEF.npy", OAEF_R060)

