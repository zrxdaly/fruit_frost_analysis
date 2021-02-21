#%%
import numpy as np
import glob as glob
from Goal_func import Slice3_AV, Slice3_PEF, plotRC
import matplotlib.pyplot as plt

#%% Wall plot
Wall_dir = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/"

FR_EF_file = sorted(glob.glob(Wall_dir + 'FR_R240*.npy'))
FRR160_EF_file = sorted(glob.glob(Wall_dir + 'FR_R160*.npy'))
RR_EF_file = sorted(glob.glob(Wall_dir + 'RR*.npy'))

FR_240_100 = np.load(FR_EF_file[0])
FR_240_200 = np.load(FR_EF_file[1])
FR_240_50 = np.load(FR_EF_file[2])

FR_160_200 = np.load(FRR160_EF_file[0])
FR_160_50 = np.load(FRR160_EF_file[1])

RR_160_100 = np.load(RR_EF_file[0])
RR_160_50 = np.load(RR_EF_file[1])
RR_240_100 = np.load(RR_EF_file[2])
#%% projection plot

Slice3_AV(FR_240_100, 'FR_240_100')
Slice3_AV(FR_240_200, 'FR_240_200')
Slice3_AV(FR_240_50, 'FR_240_50')

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
