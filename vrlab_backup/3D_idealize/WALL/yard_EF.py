
import numpy as np
import glob as glob

         
#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/WALL/"
Goal_dir = dir1+"yard/"

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


#%% get the time averaged data during operation
DO_R060 = D2_N(R060, T0, T1)

# operational aeverage data
OAEF_R060 = (DO_R060-DR_R060)/(TC_hub - DR_R060)*100 

np.save("WALL/yard_EF_R160.npy", OAEF_R060)

