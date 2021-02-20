#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:42:23 2020

OBJ: this file load the simulation data from vrlab and visualize the results

Time variant results

@author: dai
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import glob as glob

# plt.rc('font', family='serif')
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('xtick', labelsize='15')
plt.rc('ytick', labelsize='15')
plt.rc('text', usetex=True)

lw = 2
CLR = ["#b35806","#e08214","#fdb863","#fee0b6",
       "#d8daeb","#b2abd2","#8073ac","#542788",
       "#000000"]   


# load the results from vrlab
# old run
# dir_ROTA= '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/vrlab_/VA_RO/'
# VA_Full = sorted(glob.glob(dir_ROTA + "Full/*.npy"))
# VA_RR = sorted(glob.glob(dir_ROTA + "RR/*.npy"))
# re run
dir_ROTA= '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/vrlab_rerun/'
Goal_dir = sorted(glob.glob(dir_ROTA + '*R/'))

FR_dir = sorted(glob.glob(Goal_dir[0]+"DO*.npy"))
HR_dir = sorted(glob.glob(Goal_dir[1]+"DO*.npy"))
RR_dir = sorted(glob.glob(Goal_dir[2]+"DO*.npy"))

Dt = 1    # s  the time interval
T0 = 60     # s  time length for no operation
T1 = 1020    # s  time length for fan operation

#%% full domain for full rotation
# PVA_R060 = np.load(VA_Full[0])
L0 = 800
L = L0/2+1
def DOM_KM3(VA_Full, num):
    L = 351
    PVA_R060 = np.load(VA_Full[num])
    PV_R060 = np.sum(PVA_R060, axis=1)
    WV_R060 = PV_R060[1]/(L**2*10)*L0*L0*5
    KoM3_R060 = PV_R060[0]/WV_R060
    KaM3_R060 = PV_R060[0]*WV_R060
    return(KoM3_R060,KaM3_R060)

#%%


#%%
KoM3_F_R060, KaM3_F_R060 = DOM_KM3(FR_dir, 0)
KoM3_F_R080, KaM3_F_R080 = DOM_KM3(FR_dir, 1)
KoM3_F_R096, KaM3_F_R096 = DOM_KM3(FR_dir, 2)
KoM3_F_R120, KaM3_F_R120 = DOM_KM3(FR_dir, 3)
KoM3_F_R160, KaM3_F_R160 = DOM_KM3(FR_dir, 4)
KoM3_F_R192, KaM3_F_R192 = DOM_KM3(FR_dir, 5)
KoM3_F_R240, KaM3_F_R240 = DOM_KM3(FR_dir, 6)
KoM3_F_R320, KaM3_F_R320 = DOM_KM3(FR_dir, 7)
KoM3_F_R480, KaM3_F_R480 = DOM_KM3(FR_dir, 8)

KoM3_A_R060, KaM3_A_R060 = DOM_KM3(HR_dir, 0)
KoM3_A_R080, KaM3_A_R080 = DOM_KM3(HR_dir, 1)
KoM3_A_R096, KaM3_A_R096 = DOM_KM3(HR_dir, 2)
KoM3_A_R120, KaM3_A_R120 = DOM_KM3(HR_dir, 3)
KoM3_A_R160, KaM3_A_R160 = DOM_KM3(HR_dir, 4)
KoM3_A_R192, KaM3_A_R192 = DOM_KM3(HR_dir, 5)
KoM3_A_R240, KaM3_A_R240 = DOM_KM3(HR_dir, 6)
KoM3_A_R320, KaM3_A_R320 = DOM_KM3(HR_dir, 7)
KoM3_A_R480, KaM3_A_R480 = DOM_KM3(HR_dir, 8)

KoM3_R_R060, KaM3_R_R060 = DOM_KM3(RR_dir, 0)
KoM3_R_R080, KaM3_R_R080 = DOM_KM3(RR_dir, 1)
KoM3_R_R096, KaM3_R_R096 = DOM_KM3(RR_dir, 2)
KoM3_R_R120, KaM3_R_R120 = DOM_KM3(RR_dir, 3)
KoM3_R_R160, KaM3_R_R160 = DOM_KM3(RR_dir, 4)
KoM3_R_R192, KaM3_R_R192 = DOM_KM3(RR_dir, 5)
KoM3_R_R240, KaM3_R_R240 = DOM_KM3(RR_dir, 6)
KoM3_R_R320, KaM3_R_R320 = DOM_KM3(RR_dir, 7)
KoM3_R_R480, KaM3_R_R480 = DOM_KM3(RR_dir, 8)

#%% FF
T = np.arange(T0,T1, Dt)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KoM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KoM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KoM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KoM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KoM3_F_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KoM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KoM3_F_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KoM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KoM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.15])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, full rotation",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/FFAoT_Domain.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KaM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KaM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KaM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KaM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KaM3_F_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KaM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KaM3_F_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KaM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KaM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,1.5e12])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K*m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, full rotation",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/FFAaT_Domain.pdf")

#%% RR
fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KoM3_R_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KoM3_R_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KoM3_R_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KoM3_R_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KoM3_R_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KoM3_R_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KoM3_R_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KoM3_R_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KoM3_R_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.15])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180R",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/RRAoT_Domain.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KaM3_R_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KaM3_R_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KaM3_R_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KaM3_R_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KaM3_R_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KaM3_R_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KaM3_R_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KaM3_R_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KaM3_R_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,1.5e12])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K*m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180R",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/RRAaT_Domain.pdf")

#%% AA
fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KoM3_A_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KoM3_A_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KoM3_A_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KoM3_A_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KoM3_A_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KoM3_A_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KoM3_A_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KoM3_A_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KoM3_A_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.15])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180A",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/AAAoT_Domain.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, KaM3_A_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, KaM3_A_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, KaM3_A_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, KaM3_A_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, KaM3_A_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, KaM3_A_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, KaM3_A_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, KaM3_A_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, KaM3_A_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,1.5e12])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K*m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180A",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("vr_rerun/AAAaT_Domain.pdf")


#%%
Wall_dir = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/"
wall_EF = sorted(glob.glob(Wall_dir + 'WALL*'))

# WALL_F_R160 = np.load(wall_EF[0])
# WALL_F_R240 = np.load(wall_EF[1])
# WALL_R_R160 = np.load(wall_EF[2])
# WALL_R_R240 = np.load(wall_EF[3])

WoM3_F_R160, WaM3_F_R160 = DOM_KM3(wall_EF, 0)
WoM3_F_R240, WaM3_F_R240 = DOM_KM3(wall_EF, 1)
WoM3_R_R160, WaM3_R_R160 = DOM_KM3(wall_EF, 2)
WoM3_R_R240, WaM3_R_R240 = DOM_KM3(wall_EF, 3)


T = np.arange(T0,T1, Dt)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
# h1 = plt.plot(T, KoM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
# h2 = plt.plot(T, KoM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
# h3 = plt.plot(T, KoM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
# h4 = plt.plot(T, KoM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, WoM3_F_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
# h6 = plt.plot(T, KoM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, WoM3_F_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
# h8 = plt.plot(T, KoM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
# h9 = plt.plot(T, KoM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.15])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, full rotation with wall",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
# plt.savefig("vr_rerun/FFAoT_Domain.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
# h1 = plt.plot(T, KaM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
# h2 = plt.plot(T, KaM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
# h3 = plt.plot(T, KaM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
# h4 = plt.plot(T, KaM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, WaM3_F_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
# h6 = plt.plot(T, KaM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, WaM3_F_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
# h8 = plt.plot(T, KaM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
# h9 = plt.plot(T, KaM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,1.5e12])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K*m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, full rotation with wall",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
# plt.savefig("vr_rerun/FFAaT_Domain.pdf")

#%%
fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
# h1 = plt.plot(T, KoM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
# h2 = plt.plot(T, KoM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
# h3 = plt.plot(T, KoM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
# h4 = plt.plot(T, KoM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, WoM3_R_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
# h6 = plt.plot(T, KoM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, WoM3_R_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
# h8 = plt.plot(T, KoM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
# h9 = plt.plot(T, KoM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.15])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180R with wall",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
# plt.savefig("vr_rerun/FFAoT_Domain.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
# h1 = plt.plot(T, KaM3_F_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
# h2 = plt.plot(T, KaM3_F_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
# h3 = plt.plot(T, KaM3_F_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
# h4 = plt.plot(T, KaM3_F_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, WaM3_R_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
# h6 = plt.plot(T, KaM3_F_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, WaM3_R_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
# h8 = plt.plot(T, KaM3_F_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
# h9 = plt.plot(T, KaM3_F_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,1.5e12])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K*m^{3}$]",fontsize=18)
# ax.set_yscale('log')
plt.title("5m domain, 180R with wall",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
# plt.savefig("vr_rerun/FFAaT_Domain.pdf")








#%% height variation comparison
#%%
def H_KM3(VA_Full, num, height):
    # VA_Full is the file; num is the Rotation period index, height is the index of different height
    PVA_R060 = np.load(VA_Full[num])
    WV_R060 = PVA_R060[1,height]/(L**2)*700*700
    KM3H_R060 = PVA_R060[0,height]/WV_R060
    return(KM3H_R060)

# get the data from 1m 
FKM2_H1_R060 = H_KM3(VA_Full, 0, 1)
FKM2_H1_R080 = H_KM3(VA_Full, 1, 1)
FKM2_H1_R096 = H_KM3(VA_Full, 2, 1)
FKM2_H1_R120 = H_KM3(VA_Full, 3, 1)
FKM2_H1_R160 = H_KM3(VA_Full, 4, 1)
FKM2_H1_R192 = H_KM3(VA_Full, 5, 1)
FKM2_H1_R240 = H_KM3(VA_Full, 6, 1)
FKM2_H1_R320 = H_KM3(VA_Full, 7, 1)
FKM2_H1_R480 = H_KM3(VA_Full, 8, 1)

# get the data from 1m 
FKM2_H3_R060 = H_KM3(VA_Full, 0, 5)
FKM2_H3_R080 = H_KM3(VA_Full, 1, 5)
FKM2_H3_R096 = H_KM3(VA_Full, 2, 5)
FKM2_H3_R120 = H_KM3(VA_Full, 3, 5)
FKM2_H3_R160 = H_KM3(VA_Full, 4, 5)
FKM2_H3_R192 = H_KM3(VA_Full, 5, 5)
FKM2_H3_R240 = H_KM3(VA_Full, 6, 5)
FKM2_H3_R320 = H_KM3(VA_Full, 7, 5)
FKM2_H3_R480 = H_KM3(VA_Full, 8, 5)

# get the data from 3m 
RKM2_H1_R060 = H_KM3(VA_RR, 0, 1)
RKM2_H1_R080 = H_KM3(VA_RR, 1, 1)
RKM2_H1_R096 = H_KM3(VA_RR, 2, 1)
RKM2_H1_R120 = H_KM3(VA_RR, 3, 1)
RKM2_H1_R160 = H_KM3(VA_RR, 4, 1)
RKM2_H1_R192 = H_KM3(VA_RR, 5, 1)
RKM2_H1_R240 = H_KM3(VA_RR, 6, 1)
RKM2_H1_R320 = H_KM3(VA_RR, 7, 1)
RKM2_H1_R480 = H_KM3(VA_RR, 8, 1)

RKM2_H3_R060 = H_KM3(VA_RR, 0, 5)
RKM2_H3_R080 = H_KM3(VA_RR, 1, 5)
RKM2_H3_R096 = H_KM3(VA_RR, 2, 5)
RKM2_H3_R120 = H_KM3(VA_RR, 3, 5)
RKM2_H3_R160 = H_KM3(VA_RR, 4, 5)
RKM2_H3_R192 = H_KM3(VA_RR, 5, 5)
RKM2_H3_R240 = H_KM3(VA_RR, 6, 5)
RKM2_H3_R320 = H_KM3(VA_RR, 7, 5)
RKM2_H3_R480 = H_KM3(VA_RR, 8, 5)

#%%
fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, FKM2_H1_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, FKM2_H1_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, FKM2_H1_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, FKM2_H1_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, FKM2_H1_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, FKM2_H1_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, FKM2_H1_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, FKM2_H1_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, FKM2_H1_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.125])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{2}$]",fontsize=18)
plt.title("1m height, Full rotation",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 3)
plt.tight_layout()
plt.savefig("F_H1.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, FKM2_H3_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, FKM2_H3_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, FKM2_H3_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, FKM2_H3_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, FKM2_H3_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, FKM2_H3_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, FKM2_H3_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, FKM2_H3_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, FKM2_H3_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.125])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{2}$]",fontsize=18)
plt.title("3m height, Full rotation",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 3)
plt.tight_layout()
plt.savefig("F_H3.pdf")

#%%
fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, RKM2_H1_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, RKM2_H1_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, RKM2_H1_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, RKM2_H1_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, RKM2_H1_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, RKM2_H1_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, RKM2_H1_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, RKM2_H1_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, RKM2_H1_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.125])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{2}$]",fontsize=18)
plt.title("1m height, 180R rotation",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 3)
plt.tight_layout()
plt.savefig("R_H1.pdf")

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, RKM2_H3_R060, "-", label="R060",linewidth = lw,color = CLR[0])  
h2 = plt.plot(T, RKM2_H3_R080, "-", label="R080",linewidth = lw,color = CLR[1]) 
h3 = plt.plot(T, RKM2_H3_R096, "-", label="R096",linewidth = lw,color = CLR[2]) 
h4 = plt.plot(T, RKM2_H3_R120, "-", label="R120",linewidth = lw,color = CLR[3])  
h5 = plt.plot(T, RKM2_H3_R160, "-", label="R160",linewidth = lw,color = CLR[4])  
h6 = plt.plot(T, RKM2_H3_R192, "-", label="R192",linewidth = lw,color = CLR[5])  
h7 = plt.plot(T, RKM2_H3_R240, "-", label="R240",linewidth = lw,color = CLR[6])  
h8 = plt.plot(T, RKM2_H3_R320, "-", label="R320",linewidth = lw,color = CLR[7])  
h9 = plt.plot(T, RKM2_H3_R480, "-", label="R480",linewidth = lw,color = CLR[8])  
# ax.set_xlim([230,275])
ax.set_ylim([0,0.125])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"Warming [$K/m^{2}$]",fontsize=18)
plt.title("3m height, 180R rotation",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 3)
plt.tight_layout()
plt.savefig("R_H3.pdf")










#%% the evaluation at different height
# for full rotation along
def ERHplot(FR_dir):
    if FR_dir == dir_ROTA + "full_ROTA/":
        name = "FF"
    elif FR_dir == dir_ROTA + "RR_ROTA/":
        name = "FR1"
    else:
        name = "FR2"
        
    FR_file = sorted(glob.glob(FR_dir + "*.npy"))
    EF_H_R060 = np.load(FR_file[1])
    EF_H_R080 = np.load(FR_file[2])
    EF_H_R096 = np.load(FR_file[3])
    EF_H_R120 = np.load(FR_file[4])
    EF_H_R160 = np.load(FR_file[5])
    EF_H_R192 = np.load(FR_file[6])
    EF_H_R240 = np.load(FR_file[7])
    EF_H_R320 = np.load(FR_file[8])
    EF_H_R480 = np.load(FR_file[9])
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R060[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R060[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R060[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R060[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R060",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R060.pdf" %name)

    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R080[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R080[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R080[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R080[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R080[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R080",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R080.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R096[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R096[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R096[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R096[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R096[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R096",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R096.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R120[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R120[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R120[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R120[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R120[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R120",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R120.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R160[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R160[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R160[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R160[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R160[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R160",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R160.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R192[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R192[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R192[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R192[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R192[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R192",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R192.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R240[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R240[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R240[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R240[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R240[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R240",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R240.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R320[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R320[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R320[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R320[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R320[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R320",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R320.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R480[0], "-", label="0.75m",linewidth = lw,color = CLR[5])  
    h2 = plt.plot(T, EF_H_R480[1], "-", label="1.75m",linewidth = lw,color = CLR[6]) 
    h3 = plt.plot(T, EF_H_R480[2], "-", label="2.75m",linewidth = lw,color = CLR[7]) 
    h4 = plt.plot(T, EF_H_R480[3], "-", label="3.75m",linewidth = lw,color = CLR[8])  
    h5 = plt.plot(T, EF_H_R480[4], "-", label="4.75m",linewidth = lw,color = CLR[3])  
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, R480",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_R480.pdf" %name)
#%% different height comparsion
def ERRlot(FR_dir):
    if FR_dir == dir_ROTA + "full_ROTA/":
        name = "FF"
    elif FR_dir == dir_ROTA + "RR_ROTA/":
        name = "FR1"
    else:
        name = "FR2"
        
    FR_file = sorted(glob.glob(FR_dir + "*.npy"))
    EF_H_R060 = np.load(FR_file[1])
    EF_H_R080 = np.load(FR_file[2])
    EF_H_R096 = np.load(FR_file[3])
    EF_H_R120 = np.load(FR_file[4])
    EF_H_R160 = np.load(FR_file[5])
    EF_H_R192 = np.load(FR_file[6])
    EF_H_R240 = np.load(FR_file[7])
    EF_H_R320 = np.load(FR_file[8])
    EF_H_R480 = np.load(FR_file[9])
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[0], "-", label="R060",linewidth = lw,color = CLR[0])  
    h2 = plt.plot(T, EF_H_R080[0], "-", label="R080",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, EF_H_R096[0], "-", label="R096",linewidth = lw,color = CLR[2]) 
    h4 = plt.plot(T, EF_H_R120[0], "-", label="R120",linewidth = lw,color = CLR[3])  
    h5 = plt.plot(T, EF_H_R160[0], "-", label="R160",linewidth = lw,color = CLR[4])
    h6 = plt.plot(T, EF_H_R192[0], "-", label="R192",linewidth = lw,color = CLR[5])  
    h7 = plt.plot(T, EF_H_R240[0], "-", label="R240",linewidth = lw,color = CLR[6]) 
    h8 = plt.plot(T, EF_H_R320[0], "-", label="R320",linewidth = lw,color = CLR[7])  
    h9 = plt.plot(T, EF_H_R480[0], "-", label="R480",linewidth = lw,color = CLR[8])     
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, H 0.75m",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_H075.pdf" %name)

    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[1], "-", label="R060",linewidth = lw,color = CLR[0])  
    h2 = plt.plot(T, EF_H_R080[1], "-", label="R080",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, EF_H_R096[1], "-", label="R096",linewidth = lw,color = CLR[2]) 
    h4 = plt.plot(T, EF_H_R120[1], "-", label="R120",linewidth = lw,color = CLR[3])  
    h5 = plt.plot(T, EF_H_R160[1], "-", label="R160",linewidth = lw,color = CLR[4])
    h6 = plt.plot(T, EF_H_R192[1], "-", label="R192",linewidth = lw,color = CLR[5])  
    h7 = plt.plot(T, EF_H_R240[1], "-", label="R240",linewidth = lw,color = CLR[6]) 
    h8 = plt.plot(T, EF_H_R320[1], "-", label="R320",linewidth = lw,color = CLR[7])  
    h9 = plt.plot(T, EF_H_R480[1], "-", label="R480",linewidth = lw,color = CLR[8])     
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, H 1.75m",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_H175.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[2], "-", label="R060",linewidth = lw,color = CLR[0])  
    h2 = plt.plot(T, EF_H_R080[2], "-", label="R080",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, EF_H_R096[2], "-", label="R096",linewidth = lw,color = CLR[2]) 
    h4 = plt.plot(T, EF_H_R120[2], "-", label="R120",linewidth = lw,color = CLR[3])  
    h5 = plt.plot(T, EF_H_R160[2], "-", label="R160",linewidth = lw,color = CLR[4])
    h6 = plt.plot(T, EF_H_R192[2], "-", label="R192",linewidth = lw,color = CLR[5])  
    h7 = plt.plot(T, EF_H_R240[2], "-", label="R240",linewidth = lw,color = CLR[6]) 
    h8 = plt.plot(T, EF_H_R320[2], "-", label="R320",linewidth = lw,color = CLR[7])  
    h9 = plt.plot(T, EF_H_R480[2], "-", label="R480",linewidth = lw,color = CLR[8])     
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, H 2.75m",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_H275.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[3], "-", label="R060",linewidth = lw,color = CLR[0])  
    h2 = plt.plot(T, EF_H_R080[3], "-", label="R080",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, EF_H_R096[3], "-", label="R096",linewidth = lw,color = CLR[2]) 
    h4 = plt.plot(T, EF_H_R120[3], "-", label="R120",linewidth = lw,color = CLR[3])  
    h5 = plt.plot(T, EF_H_R160[3], "-", label="R160",linewidth = lw,color = CLR[4])
    h6 = plt.plot(T, EF_H_R192[3], "-", label="R192",linewidth = lw,color = CLR[5])  
    h7 = plt.plot(T, EF_H_R240[3], "-", label="R240",linewidth = lw,color = CLR[6]) 
    h8 = plt.plot(T, EF_H_R320[3], "-", label="R320",linewidth = lw,color = CLR[7])  
    h9 = plt.plot(T, EF_H_R480[3], "-", label="R480",linewidth = lw,color = CLR[8])     
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, H 3.75m",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_H375.pdf" %name)
    
    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(T, EF_H_R060[4], "-", label="R060",linewidth = lw,color = CLR[0])  
    h2 = plt.plot(T, EF_H_R080[4], "-", label="R080",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, EF_H_R096[4], "-", label="R096",linewidth = lw,color = CLR[2]) 
    h4 = plt.plot(T, EF_H_R120[4], "-", label="R120",linewidth = lw,color = CLR[4])  
    h5 = plt.plot(T, EF_H_R160[4], "-", label="R160",linewidth = lw,color = CLR[4])
    h6 = plt.plot(T, EF_H_R192[4], "-", label="R192",linewidth = lw,color = CLR[5])  
    h7 = plt.plot(T, EF_H_R240[4], "-", label="R240",linewidth = lw,color = CLR[6]) 
    h8 = plt.plot(T, EF_H_R320[4], "-", label="R320",linewidth = lw,color = CLR[7])  
    h9 = plt.plot(T, EF_H_R480[4], "-", label="R480",linewidth = lw,color = CLR[8])     
    ax.set_ylim([0,0.12])
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
    plt.title("height, time serie, H 4.75m",fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("%s_H475.pdf" %name)    
#%% 
ERHplot(FR_dir)
ERHplot(RR2_dir)
ERHplot(RR1_dir)

#%%
ERRlot(FR_dir)
ERRlot(RR2_dir)
ERRlot(RR1_dir)


    
    
    
    












