#%% get the data from vrlab  quiver animation
import numpy as np
import glob as glob
dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/WALL/WALL_100m_visual/resultslice/"
# dir1 = "/home/dai/software/navier-stoke/resultslice/"
Goal_dir = sorted(glob.glob(dir1 + '*/'))

# buo_file = sorted(glob.glob(Goal_dir[0] + '*'))
# buo_file = np.reshape(buo_file, (286, 30))

U_file = sorted(glob.glob(Goal_dir[1] + '*'))
U_file = np.reshape(U_file, (286, 30))

V_file = sorted(glob.glob(Goal_dir[2] + '*'))
V_file = np.reshape(V_file, (286, 30))

W_file = sorted(glob.glob(Goal_dir[3] + '*'))
W_file = np.reshape(W_file, (286, 30))

H = 2

def extractdata(buo_file):
    outj = np.loadtxt(buo_file[0,H], dtype='f',skiprows=2)[190,201]

    for i in range(1,286):
        xfield = np.loadtxt(buo_file[i,H], dtype='f',skiprows=2)[190.201]
        outj = np.append((outj, xfield))
    return(outj)

u_tyxz = extractdata(U_file)
v_tyxz = extractdata(V_file)
w_tyxz = extractdata(W_file)

np.save("u_tH2.npy", u_tyxz)
np.save("v_th2.npy", v_tyxz)
np.save("w_tH2.npy", w_tyxz)