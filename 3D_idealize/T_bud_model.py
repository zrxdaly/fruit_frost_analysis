#%%
import numpy as np
import glob
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol

#%%
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
cs_bud = 1700     # j kg-1 K-1
cs_l   = 1700     # j kg-1 K-1
cp_air = 1000     # j kg-1 K-1

rho_dts = 800     # kg m-3
rho_bud = 700     # kg m-3
rho_l   = 700     # kg m-3
rho_air = 1.225   # kg m-3

r_dts = 1.6e-3
r_bud = 5e-3
d_l = 0.04   # m width of leave
t_l = 0.001


def TGP(cs, rho, r):
    return(cs * rho * (r/2))

TGP_dts = TGP(cs_dts, rho_dts, r_dts)
TGP_bud = TGP(cs_bud, rho_bud, r_bud)
TGP_l   = TGP(cs_l  , rho_l  , t_l  )

Ka = 0.0255   # Wm-1K-1
vis = 1.5e-5   # m2s-1
Pr = 0.72
C = 0.683
m = 0.466
n = 1/3
g = 9.81


emi_dts = 0.5          # leave emissivity 
emi_bud = 0.96       # leave emissivity from (Vincent P. Gutschick 2016)
emi_l =   0.96
emi_surf = 0.95     # surface emissivity
sig = 5.67e-8
T_surf = 273.15

#%%
# the convection heat exchange coefficient
# from Cengel and Ghajar 2014 / Zukauskas 1972 / Ramshorst 2020
## forced convection 
# def h_coef_CG(U, d):  # U: velocity   d: leave width or diameter of dts
#     Re = U * d / vis
#     Nu = C * Re ** m * Pr ** n
#     h = Ka * Nu / d
#     return(h)

# def T_model(T_2m_380m, UU_2m, d, TGP_dts, emi_dts):
#     EN_lw = emi_surf * sig * T_surf ** 4 * emi_dts

#     T_DTS = np.zeros_like(T_2m_380m)
#     T_DTS[0] = T_2m_380m[0]
#     x = Symbol("x")
#     LW_emit0 = emi_dts * sig * x ** 4 
#     T_DTS[0] = solve(LW_emit0 + h_coef_CG(UU_2m[0], d) * (x - T_2m_380m[0]) - EN_lw, x)[1]

#     for i in range(1,1020):
#         LW_emit = emi_dts * sig * T_DTS[i-1] ** 4 
#         T_DTS[i] = T_DTS[i-1] + (EN_lw - LW_emit - h_coef_CG(UU_2m[i-1], d) * (T_DTS[i-1] - T_2m_380m[i-1])) / TGP_dts

#     return(T_DTS)

def h_coef_CG(U, d):  # U: velocity   d: leave width or diameter of dts
    Re = U * d / vis
    Nu = C * Re ** m * Pr ** n
    h = Ka * Nu / d
    return(h)

## False storage gives equillibrium state of the equations
def T_model(T_2m_380m, UU_2m, d, TGP_dts, emi_dts, storage):
    EN_lw = emi_surf * sig * T_surf ** 4 * emi_dts

    T_DTS = np.zeros_like(T_2m_380m)
    T_DTS[0] = T_2m_380m[0]
    x = Symbol("x")
    LW_emit0 = emi_dts * sig * x ** 4 
    T_DTS[0] = solve(LW_emit0 + h_coef_CG(UU_2m[0], d) * (x - T_2m_380m[0]) - EN_lw, x)[1]

    for i in range(1,1020):
        if storage:
            LW_emit = emi_dts * sig * T_DTS[i-1] ** 4 
            T_DTS[i] = T_DTS[i-1] + (EN_lw - LW_emit - h_coef_CG(UU_2m[i-1], d) * (T_DTS[i-1] - T_2m_380m[i-1])) / TGP_dts
        else:
            x = Symbol("x")
            LW_emit0 = emi_dts * sig * x ** 4 
            T_DTS[i] = solve(LW_emit0 + h_coef_CG(UU_2m[i], d) * (x - T_2m_380m[i]) - EN_lw, x)[1]

    return(T_DTS)

T_DTS = T_model(T_2m_380m, UU_2m, 2 * r_dts, TGP_dts, emi_dts, False)
T_bud = T_model(T_2m_380m, UU_2m, 2 * r_bud, TGP_bud, emi_bud, False)
T_l = T_model(T_2m_380m, UU_2m, d_l, TGP_l, emi_l, False)

T_DTS_u2 = T_model(T_2m_380m, UU_2m * 2, 2 * r_dts, TGP_dts, emi_dts, False)
T_bud_u2 = T_model(T_2m_380m, UU_2m * 2, 2 * r_bud, TGP_bud, emi_bud, False)
T_l_u2 = T_model(T_2m_380m, UU_2m * 2, d_l, TGP_l, emi_l, False)



# #%% R. Leuning 1988 model for h
# def h_coef_RL(U, d, Ta, Tl):  # U: velocity   d: leave width or diameter of dts
#     Re = U * d / vis
#     Gr = 1/Ta * g * d**3 * (Ta - Tl) / vis**2
#     if Gr/Re**2 > 16:
#         an = 1.34  # DIXON, MICHAEL GRACE, JOHN 1983
#         b = 0.171
#         Nu = an * (Gr * Pr) ** b
#     # elif Gr/Re**2 < 0.1:
#     else:
#         Nu = C * Re ** 0.5 * Pr ** (1/3)
#     h = Ka * Nu / d
#     return(h)

# def T_model_RL(T_2m_380m, UU_2m, d, TGP_dts, emi_dts):
#     EN_lw = emi_surf * sig * T_surf ** 4 * emi_dts

#     T_DTS = np.zeros_like(T_2m_380m)
#     T_DTS[0] = T_2m_380m[0]
#     # x = Symbol("x")
#     # LW_emit0 = emi_dts * sig * x ** 4 
#     # T_DTS[0] = solve(LW_emit0 + h_coef_RL(UU_2m[0], d, T_2m_380m[0], x) * (x - T_2m_380m[0]) - EN_lw, x)[1]

#     for i in range(1,1020):
#         LW_emit = emi_dts * sig * T_DTS[i-1] ** 4 
#         T_DTS[i] = T_DTS[i-1] + (EN_lw - LW_emit - h_coef_RL(UU_2m[i-1], d, T_2m_380m[i-1], T_DTS[i-1]) * (T_DTS[i-1] - T_2m_380m[i-1])) / TGP_dts

#     return(T_DTS)

# # T_DTS_RL = T_model_RL(T_2m_380m, UU_2m, 2 * r_dts, TGP_dts, emi_dts)
# # T_bud_RL = T_model_RL(T_2m_380m, UU_2m, 2 * r_bud, TGP_bud, emi_bud)
# T_l_RL = T_model_RL(T_2m_380m, UU_2m, d_l, TGP_l, emi_l)

# #%% D.N. Jordan et al 1994 model for h
# def h_coef_DNJ(U, d, Ta, Tl):  # U: velocity   d: leave width or diameter of dts
#     Re = U * d / vis
#     Gr = 1/Ta * g * d**3 * (Ta - Tl) / vis**2
#     Nu_for = 0.64 * Re ** 0.44 * Pr ** 0.33
#     Nu_fre = 0.4 * (Gr * Pr) ** 0.25
#     Nu = (Nu_for ** 3.55 + Nu_fre ** 3.55)**0.28
#     h = Ka * Nu / d
#     return(h)

# def T_model_DNJ(T_2m_380m, UU_2m, d, TGP_dts, emi_dts):
#     EN_lw = emi_surf * sig * T_surf ** 4 * emi_dts

#     T_DTS = np.zeros_like(T_2m_380m)
#     T_DTS[0] = T_2m_380m[0]
#     # x = Symbol("x")
#     # LW_emit0 = emi_dts * sig * x ** 4 
#     # T_DTS[0] = solve(LW_emit0 + h_coef_DNJ(UU_2m[0], d, T_2m_380m[0], x) * (x - T_2m_380m[0]) - EN_lw, x)[1]

#     for i in range(1,1020):
#         LW_emit = emi_dts * sig * T_DTS[i-1] ** 4 
#         T_DTS[i] = T_DTS[i-1] + (EN_lw - LW_emit - h_coef_DNJ(UU_2m[i-1], d, T_2m_380m[i-1], T_DTS[i-1]) * (T_DTS[i-1] - T_2m_380m[i-1])) / TGP_dts

#     return(T_DTS)

# T_l_DNJ = T_model_DNJ(T_2m_380m, UU_2m, d_l, TGP_l, emi_l)

#%% analytical model for delta_T
def ANA_T(T_2m_380m, UU_2m, d, TGP_dts, emi_dts):
    EN_lw = emi_surf * sig * T_surf ** 4 * emi_dts

    Delta_T = -(EN_lw - emi_dts * sig * T_2m_380m**4)/(4 * emi_dts * sig * T_2m_380m **3 + h_coef_CG(UU_2m[0], d))
    return(Delta_T)

# T_DTS_ANA = ANA_T(T_2m_380m, UU_2m, 2 * r_dts, TGP_dts, emi_dts)
# T_bud_ANA = ANA_T(T_2m_380m, UU_2m, 2 * r_bud, TGP_bud, emi_bud)
# T_l_ANA = ANA_T(T_2m_380m, UU_2m, d_l, TGP_l, emi_l)

T_2m = T_surf * np.ones_like(T_2m_380m)
UU = np.linspace(0,5,1020)
T_DTS_ANA = ANA_T(T_2m, UU, 2 * r_dts, TGP_dts, emi_dts)
T_bud_ANA = ANA_T(T_2m, UU, 2 * r_bud, TGP_bud, emi_bud)
T_l_ANA = ANA_T(T_2m, UU, d_l, TGP_l, emi_l)


# %%
T = np.arange(0, 1020, 1)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, T_2m_380m - T_surf, "-", label="T air",linewidth = 1,color = CLR[8])  
# h2 = plt.plot(T, T_DTS_u2 - T_surf, label="DTS u2",color = CLR[0]) 
# h2 = plt.plot(T, T_bud_u2 - T_surf, label="bud u2",color = CLR[4])
# h2 = plt.plot(T, T_l_u2 - T_surf, label="leave u2",color = CLR[1]) 
h2 = plt.plot(T, T_DTS - T_surf, label="DTS",color = CLR[0]) 
h2 = plt.plot(T, T_bud - T_surf, label="bud",color = CLR[4]) 
h2 = plt.plot(T, T_l - T_surf, label="leave",color = CLR[1])

# h2 = plt.plot(T, T_l_RL - T_surf, label="RL M",color = CLR[7])
# h2 = plt.plot(T, T_l_DNJ - T_surf, label="RL M",color = CLR[3])
ax.set_ylim(-4, 4)
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel("temperature [degree]",fontsize=18)
# plt.title("double wind speed",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18)
plt.tight_layout()
# plt.savefig("No_storage_E95_W1.pdf")

# %% plot the corelation between the wind and temperature difference
T = np.arange(0, 1020, 1)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
# h1 = plt.plot(T_2m_380m - T_DTS, UU_2m, '.', label='$T_{a} - T_{dts}$')  
# h1 = plt.plot(T_2m_380m - T_bud, UU_2m, '.', label='$T_{a} - T_{bud}$')
# h1 = plt.plot(T_2m_380m - T_l, UU_2m, '.', label='$T_{a} - T_{l}$')

h1 = plt.plot(UU_2m,T_2m_380m - T_DTS, 'r.', label='$T_{a} - T_{dts}$')  
h1 = plt.plot(UU_2m,T_2m_380m - T_bud, 'y.', label='$T_{a} - T_{bud}$')
h1 = plt.plot(UU_2m,T_2m_380m - T_l, 'b.', label='$T_{a} - T_{l}$')

h1 = plt.plot(UU, T_DTS_ANA,'r', label='ANA DTS')  
h1 = plt.plot(UU, T_bud_ANA,'y', label='ANA bud')
h1 = plt.plot(UU, T_l_ANA,'b', label='ANA leaf')


# h1 = plt.plot(T_2m_380m - T_DTS_u2, UU_2m * 2, '.', label='$T_{a} - T_{dts}$')  
# h1 = plt.plot(T_2m_380m - T_bud_u2, UU_2m * 2, '.', label='$T_{a} - T_{bud}$')
# h1 = plt.plot(T_2m_380m - T_l_u2, UU_2m * 2, '.', label='$T_{a} - T_{l}$')
ax.set_ylim(0, 4)
ax.set_xlim(0, 0.28)
ax.set_ylabel('$T_{diff}$',fontsize=18)
ax.set_xlabel("wind speed",fontsize=18)
# plt.title("double wind speed",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18)
plt.tight_layout()
# plt.savefig("W1_E95.pdf")

# %%
