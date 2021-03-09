import numpy as np
import glob as glob


#%%
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/WALL/WALL_100m_visual/resultslice/"
# dir1 = "/home/dai/software/navier-stoke/resultslice/"
Goal_dir = sorted(glob.glob(dir1 + '*/'))

# U_file = sorted(glob.glob(Goal_dir[1] + '*'))
# U_file = np.reshape(U_file, (286, 30))

# V_file = sorted(glob.glob(Goal_dir[2] + '*'))
# V_file = np.reshape(V_file, (286, 30))

# W_file = sorted(glob.glob(Goal_dir[3] + '*'))
# W_file = np.reshape(W_file, (286, 30))

Dt = 1    # s  the time interval
T0 = 15     # s  time length for no operation
T1 = 255    # s  time length for fan operation

#%% get the reference state of 3D data at each height
def D2_N(filedirU, filedirV, filedirW, Tstart, Tend):
    # 1234 represent different case
    FR_05_U = sorted(glob.glob(filedirU + 't=*y=003'))
    FR_10_U = sorted(glob.glob(filedirU + 't=*y=006'))
    FR_15_U = sorted(glob.glob(filedirU + 't=*y=009'))
    FR_20_U = sorted(glob.glob(filedirU + 't=*y=012'))
    FR_25_U = sorted(glob.glob(filedirU + 't=*y=015'))
    FR_30_U = sorted(glob.glob(filedirU + 't=*y=018'))
    FR_35_U = sorted(glob.glob(filedirU + 't=*y=021'))
    FR_40_U = sorted(glob.glob(filedirU + 't=*y=024'))
    FR_45_U = sorted(glob.glob(filedirU + 't=*y=027'))
    FR_50_U = sorted(glob.glob(filedirU + 't=*y=030'))

    FR_05_V = sorted(glob.glob(filedirV + 't=*y=003'))
    FR_10_V = sorted(glob.glob(filedirV + 't=*y=006'))
    FR_15_V = sorted(glob.glob(filedirV + 't=*y=009'))
    FR_20_V = sorted(glob.glob(filedirV + 't=*y=012'))
    FR_25_V = sorted(glob.glob(filedirV + 't=*y=015'))
    FR_30_V = sorted(glob.glob(filedirV + 't=*y=018'))
    FR_35_V = sorted(glob.glob(filedirV + 't=*y=021'))
    FR_40_V = sorted(glob.glob(filedirV + 't=*y=024'))
    FR_45_V = sorted(glob.glob(filedirV + 't=*y=027'))
    FR_50_V = sorted(glob.glob(filedirV + 't=*y=030'))

    FR_05_W = sorted(glob.glob(filedirW + 't=*y=003'))
    FR_10_W = sorted(glob.glob(filedirW + 't=*y=006'))
    FR_15_W = sorted(glob.glob(filedirW + 't=*y=009'))
    FR_20_W = sorted(glob.glob(filedirW + 't=*y=012'))
    FR_25_W = sorted(glob.glob(filedirW + 't=*y=015'))
    FR_30_W = sorted(glob.glob(filedirW + 't=*y=018'))
    FR_35_W = sorted(glob.glob(filedirW + 't=*y=021'))
    FR_40_W = sorted(glob.glob(filedirW + 't=*y=024'))
    FR_45_W = sorted(glob.glob(filedirW + 't=*y=027'))
    FR_50_W = sorted(glob.glob(filedirW + 't=*y=030'))

    FR_Bu05N_TKE = 0
    FR_Bu10N_TKE = 0
    FR_Bu15N_TKE = 0
    FR_Bu20N_TKE = 0
    FR_Bu25N_TKE = 0
    FR_Bu30N_TKE = 0
    FR_Bu35N_TKE = 0
    FR_Bu40N_TKE = 0
    FR_Bu45N_TKE = 0
    FR_Bu50N_TKE = 0

    for i in np.arange(Tstart, Tend):

        FR_Bu05N_U = np.loadtxt(FR_05_U[i], dtype='f',skiprows=2)
        FR_Bu10N_U = np.loadtxt(FR_10_U[i], dtype='f',skiprows=2)
        FR_Bu15N_U = np.loadtxt(FR_15_U[i], dtype='f',skiprows=2)
        FR_Bu20N_U = np.loadtxt(FR_20_U[i], dtype='f',skiprows=2)
        FR_Bu25N_U = np.loadtxt(FR_25_U[i], dtype='f',skiprows=2)
        FR_Bu30N_U = np.loadtxt(FR_30_U[i], dtype='f',skiprows=2)
        FR_Bu35N_U = np.loadtxt(FR_35_U[i], dtype='f',skiprows=2)
        FR_Bu40N_U = np.loadtxt(FR_40_U[i], dtype='f',skiprows=2)
        FR_Bu45N_U = np.loadtxt(FR_45_U[i], dtype='f',skiprows=2)
        FR_Bu50N_U = np.loadtxt(FR_50_U[i], dtype='f',skiprows=2)

        FR_Bu05N_V = np.loadtxt(FR_05_V[i], dtype='f',skiprows=2)
        FR_Bu10N_V = np.loadtxt(FR_10_V[i], dtype='f',skiprows=2)
        FR_Bu15N_V = np.loadtxt(FR_15_V[i], dtype='f',skiprows=2)
        FR_Bu20N_V = np.loadtxt(FR_20_V[i], dtype='f',skiprows=2)
        FR_Bu25N_V = np.loadtxt(FR_25_V[i], dtype='f',skiprows=2)
        FR_Bu30N_V = np.loadtxt(FR_30_V[i], dtype='f',skiprows=2)
        FR_Bu35N_V = np.loadtxt(FR_35_V[i], dtype='f',skiprows=2)
        FR_Bu40N_V = np.loadtxt(FR_40_V[i], dtype='f',skiprows=2)
        FR_Bu45N_V = np.loadtxt(FR_45_V[i], dtype='f',skiprows=2)
        FR_Bu50N_V = np.loadtxt(FR_50_V[i], dtype='f',skiprows=2)

        FR_Bu05N_W = np.loadtxt(FR_05_W[i], dtype='f',skiprows=2)
        FR_Bu10N_W = np.loadtxt(FR_10_W[i], dtype='f',skiprows=2)
        FR_Bu15N_W = np.loadtxt(FR_15_W[i], dtype='f',skiprows=2)
        FR_Bu20N_W = np.loadtxt(FR_20_W[i], dtype='f',skiprows=2)
        FR_Bu25N_W = np.loadtxt(FR_25_W[i], dtype='f',skiprows=2)
        FR_Bu30N_W = np.loadtxt(FR_30_W[i], dtype='f',skiprows=2)
        FR_Bu35N_W = np.loadtxt(FR_35_W[i], dtype='f',skiprows=2)
        FR_Bu40N_W = np.loadtxt(FR_40_W[i], dtype='f',skiprows=2)
        FR_Bu45N_W = np.loadtxt(FR_45_W[i], dtype='f',skiprows=2)
        FR_Bu50N_W = np.loadtxt(FR_50_W[i], dtype='f',skiprows=2)

        FR_Bu05N_TKE = FR_Bu05N_TKE + (FR_Bu05N_U**2 + FR_Bu05N_V**2 + FR_Bu05N_W**2)*0.5
        FR_Bu10N_TKE = FR_Bu10N_TKE + (FR_Bu10N_U**2 + FR_Bu10N_V**2 + FR_Bu10N_W**2)*0.5
        FR_Bu15N_TKE = FR_Bu15N_TKE + (FR_Bu15N_U**2 + FR_Bu15N_V**2 + FR_Bu15N_W**2)*0.5
        FR_Bu20N_TKE = FR_Bu20N_TKE + (FR_Bu20N_U**2 + FR_Bu20N_V**2 + FR_Bu20N_W**2)*0.5
        FR_Bu25N_TKE = FR_Bu25N_TKE + (FR_Bu25N_U**2 + FR_Bu25N_V**2 + FR_Bu25N_W**2)*0.5
        FR_Bu30N_TKE = FR_Bu30N_TKE + (FR_Bu30N_U**2 + FR_Bu30N_V**2 + FR_Bu30N_W**2)*0.5
        FR_Bu35N_TKE = FR_Bu35N_TKE + (FR_Bu35N_U**2 + FR_Bu35N_V**2 + FR_Bu35N_W**2)*0.5
        FR_Bu40N_TKE = FR_Bu40N_TKE + (FR_Bu40N_U**2 + FR_Bu40N_V**2 + FR_Bu40N_W**2)*0.5
        FR_Bu45N_TKE = FR_Bu45N_TKE + (FR_Bu45N_U**2 + FR_Bu45N_V**2 + FR_Bu45N_W**2)*0.5
        FR_Bu50N_TKE = FR_Bu50N_TKE + (FR_Bu50N_U**2 + FR_Bu50N_V**2 + FR_Bu50N_W**2)*0.5


    FR_Bu05N_TA = FR_Bu05N_TKE/(Tend - Tstart)
    FR_Bu10N_TA = FR_Bu10N_TKE/(Tend - Tstart)
    FR_Bu15N_TA = FR_Bu15N_TKE/(Tend - Tstart)
    FR_Bu20N_TA = FR_Bu20N_TKE/(Tend - Tstart)
    FR_Bu25N_TA = FR_Bu25N_TKE/(Tend - Tstart)
    FR_Bu30N_TA = FR_Bu30N_TKE/(Tend - Tstart)
    FR_Bu35N_TA = FR_Bu35N_TKE/(Tend - Tstart)
    FR_Bu40N_TA = FR_Bu40N_TKE/(Tend - Tstart)
    FR_Bu45N_TA = FR_Bu45N_TKE/(Tend - Tstart)
    FR_Bu50N_TA = FR_Bu50N_TKE/(Tend - Tstart)

    D12 = np.array([FR_Bu05N_TA,FR_Bu10N_TA,FR_Bu15N_TA,FR_Bu20N_TA,FR_Bu25N_TA,
                FR_Bu30N_TA,FR_Bu35N_TA,FR_Bu40N_TA,FR_Bu45N_TA,FR_Bu50N_TA])

    return(D12)
# reference state of the buoyancy data (the temp differences over height)
# DR_R060 = D2_N(Goal_dir[1], 0, T0)

#%% get the time averaged data during operation
# U_dis = D2_N(Goal_dir[1], T0, T1)
# V_dis = D2_N(Goal_dir[2], T0, T1)
# W_dis = D2_N(Goal_dir[3], T0, T1)
TKE_dis = D2_N(Goal_dir[1], Goal_dir[2], Goal_dir[3])

# np.save("UVW_output/U_dis.npy", U_dis)
# np.save("UVW_output/V_dis.npy", V_dis)
# np.save("UVW_output/W_dis.npy", W_dis)

