#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:10:01 2020

OBJ: compare the buoyancy field between the no operation and different operation

@author: dai
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import glob as glob

# plt.rc('font', family='serif')
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('xtick', labelsize='15')
plt.rc('ytick', labelsize='15')
plt.rc('text', usetex=True)
lw = 2
Color_R2 = "#000000"         
Color_R3 = '#fcf18f'          
Color_R4 = '#377eb8'         
Color_R5 = '#008000'   
Color_R6 = '#084594'             # 6.25m
Color_R7 = '#ff7f00'             # 8m
Color_R8 = '#808080'             # 10m     
#%%
# dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/ROTA/fan_goal"
dir1 = '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/'
Goal_dir = sorted(glob.glob(dir1 + 'R*/'))

R140 = Goal_dir[0] 
R180 = Goal_dir[1]  
R240 = Goal_dir[2] 
R360 = Goal_dir[3] 

# calculation of the max warming and reference warming
# in 2D
G = 9.81
T_ref = 273
KCR = 273.15
INV = 0.3
T_hub = G/T_ref*INV*10.5

def K2C(T_hub):
    return(T_hub*T_ref/G + T_ref - KCR)

TC_hub = K2C(T_hub)

Dt = 1    # s  the time interval
T0 = 60     # s  time length for no operation
T1 = 1020    # s  time length for fan operation

#%% average the data at different height over time
def D2_N(filedir, Tstart, Tend):
    # 1234 represent different case
    FR_05 = sorted(glob.glob(filedir + 't=*y=005'))
    FR_10 = sorted(glob.glob(filedir + 't=*y=010'))
    FR_15 = sorted(glob.glob(filedir + 't=*y=015'))
    FR_20 = sorted(glob.glob(filedir + 't=*y=020'))
    FR_25 = sorted(glob.glob(filedir + 't=*y=025'))
    FR_30 = sorted(glob.glob(filedir + 't=*y=030'))
    FR_35 = sorted(glob.glob(filedir + 't=*y=035'))
    FR_40 = sorted(glob.glob(filedir + 't=*y=040'))
    FR_45 = sorted(glob.glob(filedir + 't=*y=045'))
    FR_50 = sorted(glob.glob(filedir + 't=*y=050'))
    FR_series_05 = []
    FR_series_10 = []
    FR_series_15 = []
    FR_series_20 = []
    FR_series_25 = []
    FR_series_30 = []
    FR_series_35 = []
    FR_series_40 = []
    FR_series_45 = []
    FR_series_50 = []
    
    FR_series_05_append = FR_series_05.append
    FR_series_10_append = FR_series_10.append
    FR_series_15_append = FR_series_15.append
    FR_series_20_append = FR_series_20.append
    FR_series_25_append = FR_series_25.append
    FR_series_30_append = FR_series_30.append
    FR_series_35_append = FR_series_35.append
    FR_series_40_append = FR_series_40.append
    FR_series_45_append = FR_series_45.append
    FR_series_50_append = FR_series_50.append

    SUM_FR_N_H = []
    for i in np.arange(Tstart, Tend):
        FR_Bu05N = np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = np.loadtxt(FR_25[i], dtype='f',skiprows=2)
        FR_Bu30N = np.loadtxt(FR_30[i], dtype='f',skiprows=2)
        FR_Bu35N = np.loadtxt(FR_35[i], dtype='f',skiprows=2)
        FR_Bu40N = np.loadtxt(FR_40[i], dtype='f',skiprows=2)
        FR_Bu45N = np.loadtxt(FR_45[i], dtype='f',skiprows=2)
        FR_Bu50N = np.loadtxt(FR_50[i], dtype='f',skiprows=2)

        FR_series_05_append(K2C(FR_Bu05N.mean()))
        FR_series_10_append(K2C(FR_Bu10N.mean()))
        FR_series_15_append(K2C(FR_Bu15N.mean()))
        FR_series_20_append(K2C(FR_Bu20N.mean()))
        FR_series_25_append(K2C(FR_Bu25N.mean()))
        FR_series_30_append(K2C(FR_Bu30N.mean()))
        FR_series_35_append(K2C(FR_Bu35N.mean()))
        FR_series_40_append(K2C(FR_Bu40N.mean()))
        FR_series_45_append(K2C(FR_Bu45N.mean()))
        FR_series_50_append(K2C(FR_Bu50N.mean()))
    
    SUM_FR_N_H = [np.average(FR_series_05),
                  np.average(FR_series_10),
                  np.average(FR_series_15),
                  np.average(FR_series_20),
                  np.average(FR_series_25),
                  np.average(FR_series_30),
                  np.average(FR_series_35),
                  np.average(FR_series_40),
                  np.average(FR_series_45),
                  np.average(FR_series_50)]
    
    return(SUM_FR_N_H)

SUM_R140_N_H = D2_N(R140, 0, T0)
SUM_R140_N = np.mean(SUM_R140_N_H)

SUM_R180_N_H = D2_N(R180, 0, T0)
SUM_R180_N = np.mean(SUM_R180_N_H)

SUM_R240_N_H = D2_N(R240, 0, T0)
SUM_R240_N = np.mean(SUM_R240_N_H)

SUM_R360_N_H = D2_N(R360, 0, T0)
SUM_R360_N = np.mean(SUM_R360_N_H)

#%%
def D2(filedir, Tstart, Tend, SUM_FR_N_H):
    # 1234 represent different case
    FR_05 = sorted(glob.glob(filedir + 't=*y=005'))
    FR_10 = sorted(glob.glob(filedir + 't=*y=010'))
    FR_15 = sorted(glob.glob(filedir + 't=*y=015'))
    FR_20 = sorted(glob.glob(filedir + 't=*y=020'))
    FR_25 = sorted(glob.glob(filedir + 't=*y=025'))
    FR_30 = sorted(glob.glob(filedir + 't=*y=030'))
    FR_35 = sorted(glob.glob(filedir + 't=*y=035'))
    FR_40 = sorted(glob.glob(filedir + 't=*y=040'))
    FR_45 = sorted(glob.glob(filedir + 't=*y=045'))
    FR_50 = sorted(glob.glob(filedir + 't=*y=050'))
    FR_series_05 = []
    FR_series_10 = []
    FR_series_15 = []
    FR_series_20 = []
    FR_series_25 = []
    FR_series_30 = []
    FR_series_35 = []
    FR_series_40 = []
    FR_series_45 = []
    FR_series_50 = []

    FR_series_05_append = FR_series_05.append
    FR_series_10_append = FR_series_10.append
    FR_series_15_append = FR_series_15.append
    FR_series_20_append = FR_series_20.append
    FR_series_25_append = FR_series_25.append
    FR_series_30_append = FR_series_30.append
    FR_series_35_append = FR_series_35.append
    FR_series_40_append = FR_series_40.append
    FR_series_45_append = FR_series_45.append
    FR_series_50_append = FR_series_50.append

    D22 = []
    for i in np.arange(Tstart, Tend):
        FR_Bu05N = np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = np.loadtxt(FR_25[i], dtype='f',skiprows=2)
        FR_Bu30N = np.loadtxt(FR_30[i], dtype='f',skiprows=2)
        FR_Bu35N = np.loadtxt(FR_35[i], dtype='f',skiprows=2)
        FR_Bu40N = np.loadtxt(FR_40[i], dtype='f',skiprows=2)
        FR_Bu45N = np.loadtxt(FR_45[i], dtype='f',skiprows=2)
        FR_Bu50N = np.loadtxt(FR_50[i], dtype='f',skiprows=2)
        # substraction
        FR_series_05_append(K2C(FR_Bu05N.mean()) - SUM_FR_N_H[0])
        FR_series_10_append(K2C(FR_Bu10N.mean()) - SUM_FR_N_H[1])
        FR_series_15_append(K2C(FR_Bu15N.mean()) - SUM_FR_N_H[2])
        FR_series_20_append(K2C(FR_Bu20N.mean()) - SUM_FR_N_H[3])
        FR_series_25_append(K2C(FR_Bu25N.mean()) - SUM_FR_N_H[4])
        FR_series_30_append(K2C(FR_Bu30N.mean()) - SUM_FR_N_H[5])
        FR_series_35_append(K2C(FR_Bu35N.mean()) - SUM_FR_N_H[6])
        FR_series_40_append(K2C(FR_Bu40N.mean()) - SUM_FR_N_H[7])
        FR_series_45_append(K2C(FR_Bu45N.mean()) - SUM_FR_N_H[8])
        FR_series_50_append(K2C(FR_Bu50N.mean()) - SUM_FR_N_H[9])
    
    D22 = np.array([FR_series_05, FR_series_10, FR_series_15, FR_series_20,
          FR_series_25, FR_series_30, FR_series_35, FR_series_40,
          FR_series_45, FR_series_50])
    
    return(D22)

#%%
R140_2D = D2(R140, T0, T1, SUM_R140_N_H)
R180_2D = D2(R180, T0, T1, SUM_R180_N_H)
R240_2D = D2(R240, T0, T1, SUM_R240_N_H)
R360_2D = D2(R360, T0, T1, SUM_R360_N_H)

#%% plot the statistics

R140_DS = R140_2D.mean(axis = 0)
R180_DS = R180_2D.mean(axis = 0)
R240_DS = R240_2D.mean(axis = 0)
R360_DS = R360_2D.mean(axis = 0)


#%% the efficiency then can be expressed as 
EF_R140 = R140_DS/(TC_hub - SUM_R140_N)
EF_R180 = R180_DS/(TC_hub - SUM_R180_N)
EF_R240 = R240_DS/(TC_hub - SUM_R240_N)
EF_R360 = R360_DS/(TC_hub - SUM_R360_N)

#%%
T = np.arange(T0,T1, Dt)

fig1 = plt.figure(figsize=(6.4, 4.8))
ax = fig1.add_subplot(1,1,1)
h1 = plt.plot(T, EF_R140, "-", label="R140",linewidth = lw,color = Color_R2)  
# h2 = plt.plot(T, EF_R160, "-", label="R160",linewidth = lw,color = Color_R3) 
h3 = plt.plot(T, EF_R180, "-", label="R180",linewidth = lw,color = Color_R4) 
h4 = plt.plot(T, EF_R240, "-", label="R240",linewidth = lw,color = Color_R5)  
# h5 = plt.plot(T, EF_R288, "-", label="R288",linewidth = lw,color = Color_R5)  
h6 = plt.plot(T, EF_R360, "-", label="R360",linewidth = lw,color = Color_R5)  
# ax.set_xlim([230,275])
# ax.set_ylim([0,300])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("whole domain, time serie comparision",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
plt.tight_layout()
plt.savefig("RR_Domain.pdf")

#%% plot the height statistics
def list2Series(FR_2D,SUM_FR_N_H):
    FR_2D = np.array(FR_2D)
    FR_H1 = (FR_2D[0]+FR_2D[1])/2     
    FR_H2 = (FR_2D[2]+FR_2D[3])/2
    FR_H3 = (FR_2D[4]+FR_2D[5])/2
    FR_H4 = (FR_2D[6]+FR_2D[7])/2
    FR_H5 = (FR_2D[8]+FR_2D[9])/2
    
    FR_NH1 = (SUM_FR_N_H[0]+SUM_FR_N_H[1])/2     
    FR_NH2 = (SUM_FR_N_H[2]+SUM_FR_N_H[3])/2
    FR_NH3 = (SUM_FR_N_H[4]+SUM_FR_N_H[5])/2
    FR_NH4 = (SUM_FR_N_H[6]+SUM_FR_N_H[7])/2
    FR_NH5 = (SUM_FR_N_H[8]+SUM_FR_N_H[9])/2
    
    EF_FR_H1 = FR_H1/(TC_hub - FR_NH1)
    EF_FR_H2 = FR_H2/(TC_hub - FR_NH2)
    EF_FR_H3 = FR_H3/(TC_hub - FR_NH3)
    EF_FR_H4 = FR_H4/(TC_hub - FR_NH4)
    EF_FR_H5 = FR_H5/(TC_hub - FR_NH5)
    return(EF_FR_H1,EF_FR_H2,EF_FR_H3,EF_FR_H4,EF_FR_H5)

R140_H1,R140_H2,R140_H3,R140_H4,R140_H5 = list2Series(R140_2D,SUM_R140_N_H)
# R160_H1,R160_H2,R160_H3,R160_H4,R160_H5 = list2Series(R160_2D,SUM_R160_N_H)
R180_H1,R180_H2,R180_H3,R180_H4,R180_H5 = list2Series(R180_2D,SUM_R180_N_H)
R240_H1,R240_H2,R240_H3,R240_H4,R240_H5 = list2Series(R240_2D,SUM_R240_N_H)
# R288_H1,R288_H2,R288_H3,R288_H4,R288_H5 = list2Series(R288_2D,SUM_R288_N_H)
R360_H1,R360_H2,R360_H3,R360_H4,R360_H5 = list2Series(R360_2D,SUM_R360_N_H)

#%%
fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
h2 = plt.plot(T, R140_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
h3 = plt.plot(T, R140_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
h4 = plt.plot(T, R140_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
h5 = plt.plot(T, R140_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, R140",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RR_140.pdf")


# fig2 = plt.figure(figsize=(6.4, 4.8))
# ax = fig2.add_subplot(1,1,1)
# h1 = plt.plot(T, R160_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
# h2 = plt.plot(T, R160_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
# h3 = plt.plot(T, R160_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
# h4 = plt.plot(T, R160_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
# h5 = plt.plot(T, R160_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
# # ax.set_ylim([0,0.0013])
# ax.set_xlabel('Time [s]',fontsize=18)
# ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
# plt.title("height, time serie, R160",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
# plt.tight_layout()
# plt.savefig("RR_160.pdf")


fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R180_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
h2 = plt.plot(T, R180_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
h3 = plt.plot(T, R180_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
h4 = plt.plot(T, R180_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
h5 = plt.plot(T, R180_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, R180 ",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RR_180.pdf")


fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R240_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
h2 = plt.plot(T, R240_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
h3 = plt.plot(T, R240_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
h4 = plt.plot(T, R240_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
h5 = plt.plot(T, R240_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, R240",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RR_240.pdf")

# fig2 = plt.figure(figsize=(6.4, 4.8))
# ax = fig2.add_subplot(1,1,1)
# h1 = plt.plot(T, R288_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
# h2 = plt.plot(T, R288_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
# h3 = plt.plot(T, R288_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
# h4 = plt.plot(T, R288_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
# h5 = plt.plot(T, R288_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
# # ax.set_ylim([0,0.0013])
# ax.set_xlabel('Time [s]',fontsize=18)
# ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
# plt.title("height, time serie, R288",fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
# plt.tight_layout()
# plt.savefig("RR_288.pdf")

fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R360_H1, "-", label="0.75m",linewidth = lw,color = Color_R2)  
h2 = plt.plot(T, R360_H2, "-", label="1.75m",linewidth = lw,color = Color_R3) 
h3 = plt.plot(T, R360_H3, "-", label="2.75m",linewidth = lw,color = Color_R4) 
h4 = plt.plot(T, R360_H4, "-", label="3.75m",linewidth = lw,color = Color_R5)  
h5 = plt.plot(T, R360_H5, "-", label="4.75m",linewidth = lw,color = Color_R6)  
# ax.set_ylim([0,0.0013])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, R360",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RR_360.pdf")

#%%

fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H1, "-", label="R140",linewidth = lw,color = Color_R2)
# h2 = plt.plot(T, R160_H1, "-", label="R160",linewidth = lw,color = Color_R3)
h3 = plt.plot(T, R180_H1, "-", label="R180",linewidth = lw,color = Color_R4)
h4 = plt.plot(T, R240_H1, "-", label="R240",linewidth = lw,color = Color_R5)
# h4 = plt.plot(T, R288_H1, "-", label="R288",linewidth = lw,color = Color_R6)
h4 = plt.plot(T, R360_H1, "-", label="R360",linewidth = lw,color = Color_R7)
# ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, height 0.75m",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RO_H075.pdf")


fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H2, "-", label="R140",linewidth = lw,color = Color_R2)
# h2 = plt.plot(T, R160_H2, "-", label="R160",linewidth = lw,color = Color_R3)
h3 = plt.plot(T, R180_H2, "-", label="R180",linewidth = lw,color = Color_R4)
h4 = plt.plot(T, R240_H2, "-", label="R240",linewidth = lw,color = Color_R5)
# h4 = plt.plot(T, R288_H2, "-", label="R288",linewidth = lw,color = Color_R6)
h4 = plt.plot(T, R360_H2, "-", label="R360",linewidth = lw,color = Color_R7) 
ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, height 1.75m",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RO_H175.pdf")


fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H3, "-", label="R140",linewidth = lw,color = Color_R2)
# h2 = plt.plot(T, R160_H3, "-", label="R160",linewidth = lw,color = Color_R3)
h3 = plt.plot(T, R180_H3, "-", label="R180",linewidth = lw,color = Color_R4)
h4 = plt.plot(T, R240_H3, "-", label="R240",linewidth = lw,color = Color_R5)
# h4 = plt.plot(T, R288_H3, "-", label="R288",linewidth = lw,color = Color_R6)
h4 = plt.plot(T, R360_H3, "-", label="R360",linewidth = lw,color = Color_R7)   
# ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, height 2.75m ",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RO_H275.pdf")


fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H4, "-", label="R140",linewidth = lw,color = Color_R2)
# h2 = plt.plot(T, R160_H4, "-", label="R160",linewidth = lw,color = Color_R3)
h3 = plt.plot(T, R180_H4, "-", label="R180",linewidth = lw,color = Color_R4)
h4 = plt.plot(T, R240_H4, "-", label="R240",linewidth = lw,color = Color_R5)
# h4 = plt.plot(T, R288_H4, "-", label="R288",linewidth = lw,color = Color_R6)
h4 = plt.plot(T, R360_H4, "-", label="R360",linewidth = lw,color = Color_R7)
# ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, height 3.75m",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RO_H375.pdf")

fig2 = plt.figure(figsize=(6.4, 4.8))
ax = fig2.add_subplot(1,1,1)
h1 = plt.plot(T, R140_H5, "-", label="R140",linewidth = lw,color = Color_R2)
# h2 = plt.plot(T, R160_H5, "-", label="R160",linewidth = lw,color = Color_R3)
h3 = plt.plot(T, R180_H5, "-", label="R180",linewidth = lw,color = Color_R4)
h4 = plt.plot(T, R240_H5, "-", label="R240",linewidth = lw,color = Color_R5)
# h4 = plt.plot(T, R288_H5, "-", label="R288",linewidth = lw,color = Color_R6)
h4 = plt.plot(T, R360_H5, "-", label="R360",linewidth = lw,color = Color_R7)
# ax.set_ylim([0,0.08])
ax.set_xlabel('Time [s]',fontsize=18)
ax.set_ylabel(r"$\Delta T/\Delta T_{max}$",fontsize=18)
plt.title("height, time serie, height 4.75m",fontsize=18)
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.tight_layout()
plt.savefig("RO_H475.pdf")












