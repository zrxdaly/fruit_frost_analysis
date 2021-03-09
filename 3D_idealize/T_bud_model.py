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

cs_dts = 502      # j kg-1 K-1
cs_bud = 1700
cp_air = 1000   #j kg-1 K-1

rho_dts = 800     # kg m-3
rho_air = 1.225   # kg m-3
rho_bud = 700

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

r = 1.6e-3
r_bud = 10e-3
ds = 2 * r
dl = 0.05   # m width of leave
#%%
## assume the surface temperature is 0 degree (273.15K)
# def tree_model(T_2m_380m, UU_2m):

T_DTS = np.zeros_like(T_2m_380m)
T_DTS[0] = T_2m_380m[0]

T_DTS_doubleU = np.zeros_like(T_2m_380m)
T_DTS_doubleU[0] = T_2m_380m[0]

T_bud = np.zeros_like(T_2m_380m)
T_bud[0] = T_2m_380m[0]

T_bud_u2 = np.zeros_like(T_2m_380m)
T_bud_u2[0] = T_2m_380m[0]

T_l = np.zeros_like(T_2m_380m)
T_l[0] = T_2m_380m[0]

T_l_u2 = np.zeros_like(T_2m_380m)
T_l_u2[0] = T_2m_380m[0]

for i in range(1,1020):
    # calculation of h
    h = C * ds ** (m-1) * Pr ** n * ka * va ** (-m) * UU_2m[i] ** m
    h_bud = C * (r_bud * 2) ** (m-1) * Pr ** n * ka * va ** (-m) * UU_2m[i] ** m
    h_bud_u2 = C * (r_bud * 2) ** (m-1) * Pr ** n * ka * va ** (-m) * (UU_2m[i]*2) ** m
    h_u2 = C * ds ** (m-1) * Pr ** n * ka * va ** (-m) * (UU_2m[i]*2) ** m
    h_l = 2 * 0.5 * dl**(-0.5) * Pr ** n * ka * va ** (-m) * (UU_2m[i]) ** m
    h_l_u2 = 2 * 0.5 * dl**(-0.5) * Pr ** n * ka * va ** (-m) * (UU_2m[i]*2) ** m
    # longwave radiation
    EN_lw = emi_surf * sig * T_surf ** 4 * emi
    # longwave emit
    EN_lw_out_DTS = emi * sig * T_DTS[i-1] ** 4 
    EN_lw_out_bud = emi * sig * T_bud[i-1] ** 4 
    EN_lw_out_bud_u2 = emi * sig * T_bud_u2[i-1] ** 4 
    EN_lw_out_DTS_U2 = emi * sig * T_DTS_doubleU[i-1] ** 4 
    EN_lw_out_l = emi * sig * T_l[i-1] ** 4 
    EN_lw_out_l_u2 = emi * sig * T_l_u2[i-1] ** 4 


    T_DTS[i] = T_DTS[i-1] + (EN_lw - EN_lw_out_DTS - h * (T_DTS[i-1] - T_2m_380m[i-1])) * 2 / (cs_dts * rho_dts * r)
    T_DTS_doubleU[i] = T_DTS_doubleU[i-1] + (EN_lw - EN_lw_out_DTS_U2 - h_u2 * (T_DTS_doubleU[i-1] - T_2m_380m[i-1])) * 2 / (cs_dts * rho_dts * r)
    T_bud[i] = T_bud[i-1] + (EN_lw - EN_lw_out_bud - h_bud * (T_bud[i-1] - T_2m_380m[i-1])) * 2 / (cs_bud * rho_bud * r_bud)
    T_bud_u2[i] = T_bud_u2[i-1] + (EN_lw - EN_lw_out_bud_u2 - h_bud_u2 * (T_bud_u2[i-1] - T_2m_380m[i-1])) * 2 / (cs_bud * rho_bud * r_bud)
    T_l[i] = T_l[i-1] + (EN_lw - EN_lw_out_l - h_l * (T_l[i-1] - T_2m_380m[i-1])) / (cs_bud * rho_bud * 0.002)
    T_l_u2[i] = T_l_u2[i-1] + (EN_lw - EN_lw_out_l_u2 - h_l_u2 * (T_l_u2[i-1] - T_2m_380m[i-1])) / (cs_bud * rho_bud * 0.002)
    

# %%
T = np.arange(0, 1020, 1)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, T_2m_380m - T_surf, "-", label="T air",linewidth = 1,color = CLR[8])  
h2 = plt.plot(T, T_DTS_doubleU - T_surf, label="DTS u2",color = CLR[1]) 
h2 = plt.plot(T, T_bud_u2 - T_surf, label="bud u2",color = CLR[4])
h2 = plt.plot(T, T_l_u2 - T_surf, label="leave u2",color = CLR[0]) 
# h2 = plt.plot(T, T_DTS - T_surf, label="DTS",color = CLR[1]) 
# h2 = plt.plot(T, T_bud - T_surf, label="bud",color = CLR[4]) 
# h2 = plt.plot(T, T_l - T_surf, label="leave",color = CLR[0])
ax.set_ylim(-4, 4)
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel("temperature [degree]",fontsize=18)
# plt.title("%s "%txt,fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18)
plt.tight_layout()
plt.savefig("model_bud_u2.pdf")

# %%
