#%%
import numpy as np
import glob as glob
from Goal_func import Slice3_AV, Slice3_PEF, plotRC
import matplotlib.pyplot as plt

G = 9.81
T_ref = 273
KCR = 273.15
INV = 0.3
# T_hub = G/T_ref*INV*10.5

def K2C(T_hub):
    return(T_hub*T_ref/G)
    # return(T_hub*T_ref/G + T_ref - KCR)

#%% Wall plot
pfp_dir = "/home/dai/Desktop/paraffin/data/"

FAC_U = sorted(glob.glob(pfp_dir + "FAC_U/" + '*_dis.npy'))
FAC_U_ref = sorted(glob.glob(pfp_dir + "FAC_U/" + '*_ref.npy'))

FAC_INV = sorted(glob.glob(pfp_dir + "FAC_INV/" + '*_dis.npy'))
FAC_INV_ref = sorted(glob.glob(pfp_dir + "FAC_INV/" + '*_ref.npy'))
#%%
U00 = np.load(FAC_U[0])
U00_ref = np.load(FAC_U_ref[0])
EF_U00 = K2C(U00 - U00_ref)
Slice3_AV(EF_U00, "U00")

U01 = np.load(FAC_U[1])
U01_ref = np.load(FAC_U_ref[1])
EF_U01 = K2C(U01 - U01_ref)
Slice3_AV(EF_U01, "U01")

U02 = np.load(FAC_U[2])
U02_ref = np.load(FAC_U_ref[2])
EF_U02 = K2C(U02 - U02_ref)
Slice3_AV(EF_U02, "U02")

U03 = np.load(FAC_U[3])
U03_ref = np.load(FAC_U_ref[3])
EF_U03 = K2C(U03 - U03_ref)
Slice3_AV(EF_U03, "U03")

#%%
IN4 = np.load(FAC_INV[0])
IN4_ref = np.load(FAC_INV_ref[0])
EF_IN4 = K2C(IN4 - IN4_ref)
Slice3_AV(EF_IN4, "INV4")

IN5 = np.load(FAC_INV[1])
IN5_ref = np.load(FAC_INV_ref[1])
EF_IN5 = K2C(IN5 - IN5_ref)
Slice3_AV(EF_IN5, "INV5")

# %%
