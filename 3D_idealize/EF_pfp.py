import numpy as np
import glob as glob


#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/paraffin/test/FAC_U/"
Goal_dir = sorted(glob.glob(dir1 + 'U*/buo/'))

Dt = 1    # s  the time interval
T0 = 0     # s  time length for no operation
T1 = 600    # s  time length for fan operation

#%% get the reference state of 3D data at each height
def D2_N(filedirU, filedirV, filedirW, Tstart, Tend):
    # 1234 represent different case
    FR_01_U = sorted(glob.glob(filedirU + 't=*y=001'))
    FR_02_U = sorted(glob.glob(filedirU + 't=*y=002'))
    FR_03_U = sorted(glob.glob(filedirU + 't=*y=003'))
    FR_04_U = sorted(glob.glob(filedirU + 't=*y=004'))
    FR_05_U = sorted(glob.glob(filedirU + 't=*y=005'))

    for i in np.arange(Tstart, Tend):

        FR_Bu01N_U = np.loadtxt(FR_01_U[i], dtype='f',skiprows=2)
        FR_Bu02N_U = np.loadtxt(FR_02_U[i], dtype='f',skiprows=2)
        FR_Bu03N_U = np.loadtxt(FR_03_U[i], dtype='f',skiprows=2)
        FR_Bu04N_U = np.loadtxt(FR_04_U[i], dtype='f',skiprows=2)
        FR_Bu05N_U = np.loadtxt(FR_05_U[i], dtype='f',skiprows=2)

    FR_Bu05N_TA = FR_Bu01N_U/(Tend - Tstart)
    FR_Bu10N_TA = FR_Bu02N_U/(Tend - Tstart)
    FR_Bu15N_TA = FR_Bu03N_U/(Tend - Tstart)
    FR_Bu20N_TA = FR_Bu04N_U/(Tend - Tstart)
    FR_Bu25N_TA = FR_Bu05N_U/(Tend - Tstart)

    D12 = np.array([FR_Bu05N_TA,FR_Bu10N_TA,FR_Bu15N_TA,FR_Bu20N_TA,FR_Bu25N_TA])

    return(D12)
# reference state of the buoyancy data (the temp differences over height)
# DR_R060 = D2_N(Goal_dir[1], 0, T0)

#%% get the time averaged data during operation
U00_dis = D2_N(Goal_dir[0], T0, T1)
np.save("U00_dis.npy", U00_dis)

U01_dis = D2_N(Goal_dir[1], T0, T1)
np.save("U01_dis.npy", U00_dis)

U02_dis = D2_N(Goal_dir[2], T0, T1)
np.save("U02_dis.npy", U00_dis)

U03_dis = D2_N(Goal_dir[3], T0, T1)
np.save("U03_dis.npy", U00_dis)