#%%
import numpy as np
import glob as glob
from Goal_func import Slice3_AV, Slice3_PEF, plotRC
import matplotlib.pyplot as plt

#%% Wall plot
Wall_dir = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/distance/"

EF_50m_file = sorted(glob.glob(Wall_dir + "50m/" + '*_dis.npy'))
EF_100m_file = sorted(glob.glob(Wall_dir + "100m/" + '*_dis.npy'))

UVWT_50m_file = sorted(glob.glob(Wall_dir + "50m/" + '*_tyxz.npy'))
#%%
TKE_50m = np.load(EF_50m_file[0])
U_50m = np.load(EF_50m_file[1])
V_50m = np.load(EF_50m_file[2])
W_50m = np.load(EF_50m_file[3])

TKE_100m = np.load(EF_100m_file[0])
U_100m = np.load(EF_100m_file[1])
V_100m = np.load(EF_100m_file[2])
W_100m = np.load(EF_100m_file[3])

UT_50m = np.load(UVWT_50m_file[0])
VT_50m = np.load(UVWT_50m_file[1])
WT_50m = np.load(UVWT_50m_file[2])

TKE_T_50m = (UT_50m**2 + VT_50m**2 + WT_50m **2)*0.5
#%% projection plot

# Slice3_AV(U_50m, 'U_50m')
Slice3_AV(TKE_50m, 'TKE_50m')


#%%
Slice3_PEF(TKE_50m, 'TKE_50m')
# Slice3_PEF(U_100m, 'U_100m')

#%%

Slice3_AV(TKE_T_50m[2,], 'TKE_T_50m')

Slice3_PEF(TKE_T_50m[2,], 'TKE_CT_50m')


#%%








#%%
Slice3_PEF(FR_240_100, 'FR_240_100')
Slice3_PEF(FR_240_200, 'FR_240_200')
Slice3_PEF(FR_240_50, 'FR_240_50')

# %%
Slice3_AV(RR_160_100, 'RR_160_100')
Slice3_AV(RR_160_50, 'RR_160_50')
Slice3_AV(RR_240_100, 'RR_240_100')

Slice3_PEF(RR_160_100, 'RR_160_100')
Slice3_PEF(RR_160_50, 'RR_160_50')
Slice3_PEF(RR_240_100, 'RR_240_100')
# %%
FR_160_200 = np.load(FRR160_EF_file[0])
FR_160_50 = np.load(FRR160_EF_file[1])

Slice3_AV(FR_160_200, 'FR_160_200')
Slice3_AV(FR_160_50, 'FR_160_50')

Slice3_PEF(FR_160_200, 'FR_160_200')
Slice3_PEF(FR_160_50, 'FR_160_50')
# %%
