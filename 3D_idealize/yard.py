#%%
import numpy as np
import glob as glob
from Goal_func import Slice3_AV, Slice3_PEF, plotRC
import matplotlib.pyplot as plt

#%% Wall plot
Wall_dir = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/"
wall_EF = sorted(glob.glob(Wall_dir + 'yard*.npy'))

FR_cEF_R160 = np.load(wall_EF[0])

#%% projection plot
Slice3_AV(FR_cEF_R160, 'yard_R240')

#%%
Slice3_PEF(FR_cEF_R160, 'yard_R240')

# %%
