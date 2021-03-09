#%%
import numpy as np
import glob
import matplotlib.pyplot as plt
# %matplotlib widget
# import ipywidgets as widgets

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='20')
plt.rc('ytick', labelsize='20')
plt.rc('text', usetex=True)
lw = 2
CLR = ["#d7191c","#fdae61","#ffffbf","#abdda4",
       "#2b83ba","#b2abd2","#8073ac","#542788",
       "#000000"]   
       
#%%

dir = "/home/dai/Desktop/trans/3D_idealize/T_timeseries/"
TS_file = sorted(glob.glob(dir + "Full/" + 'T*.npy'))
TR240_Series = np.load(TS_file[6])

UVW_file = sorted(glob.glob(dir + 'UVW_wall/' + '*H2.npy'))
U_2m_380m = np.load(UVW_file[0])
UU_2m = np.zeros(1020)
for i in range(1020):
    UU_2m[i] = U_2m_380m[int(i/4)]

T_2m_380m = TR240_Series[3,:,6] + 273.15

#%% the temperature series at 2m height at 380 meter

cs = 502      # jkg-1K-1
rho = 800     # kg m-3

ka = 0.0255   # Wm-1K-1
va = 1.5e-5   # m2s-1
Pr = 0.72
C = 0.683
m = 0.466
n = 1/3

emi = 0.5
emi_surf = 0.8
sig = 5.67e-8
T_surf = 273.15

r = 6.7e-4
ds = 2 * r
#%%
## assume the surface temperature is 0 degree (273.15K)
# def tree_model(T_2m_380m, UU_2m):

T_object = np.zeros_like(T_2m_380m)
T_object[0] = T_2m_380m[0]

T_object_doubleU = np.zeros_like(T_2m_380m)
T_object_doubleU[0] = T_2m_380m[0]

for i in range(1,1020):
    # calculation of h
    h = C * ds ** (m-1) * Pr ** n * ka * va ** (-m) * UU_2m[i] ** m
    h_u2 = C * ds ** (m-1) * Pr ** n * ka * va ** (-m) * (UU_2m[i]*2) ** m
    
    # longwave radiation
    EN_lw = emi_surf * sig * T_surf ** 4 * emi
    # longwave emit
    EN_lw_out = emi * sig * T_object[i-1] ** 4 

    T_object[i] = T_object[i-1] + (EN_lw - EN_lw_out - h * (T_object[i-1] - T_2m_380m[i-1])) * 2 / (cs * rho * r)
    T_object_doubleU[i] = T_object_doubleU[i-1] + (EN_lw - EN_lw_out - h_u2 * (T_object_doubleU[i-1] - T_2m_380m[i-1])) * 2 / (cs * rho * r)


# %%
T = np.arange(0, 1020, 1)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, T_2m_380m, "-", label="Temp",linewidth = 1,color = CLR[8])  
h2 = plt.plot(T, T_object, label="model DTS",color = CLR[1]) 
h2 = plt.plot(T, T_object_doubleU, label="model DTS (double U)",color = CLR[4]) 
# ax.set_ylim(0, 4.5)
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel("temperature [Kevin]",fontsize=18)
# plt.title("%s "%txt,fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
# plt.savefig("TS%s.pdf"%txt)
# %%
