#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:10:01 2020

OBJ: explore the chance of guassian distrbution in warming effects

@author: dai
"""
#%%
# import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import glob as glob

from matplotlib.ticker import NullFormatter

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# plt.rc('font', family='serif')
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('xtick', labelsize='15')
plt.rc('ytick', labelsize='15')
# plt.rc('text', usetex=True)
lw = 2
Color_R2 = '#000000'
Color_R3 = '#fcf18f'
Color_R4 = '#377eb8'
Color_R5 = '#008000'
Color_R6 = '#084594'
Color_R7 = '#ff7f00'
Color_R8 = '#808080'
#%%
# dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/ROTA/fan_goal"
dir1 = '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/'
Goal_dir = sorted(glob.glob(dir1 + 'R*/'))

R140 = Goal_dir[0] 
# calculation of the max warming and reference warming
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


#%% get the reference state of 3D data at each height
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

    FR_Bu05N = np.loadtxt(FR_05[Tstart], dtype='f',skiprows=2)
    FR_Bu10N = np.loadtxt(FR_10[Tstart], dtype='f',skiprows=2)
    FR_Bu15N = np.loadtxt(FR_15[Tstart], dtype='f',skiprows=2)
    FR_Bu20N = np.loadtxt(FR_20[Tstart], dtype='f',skiprows=2)
    FR_Bu25N = np.loadtxt(FR_25[Tstart], dtype='f',skiprows=2)
    FR_Bu30N = np.loadtxt(FR_30[Tstart], dtype='f',skiprows=2)
    FR_Bu35N = np.loadtxt(FR_35[Tstart], dtype='f',skiprows=2)
    FR_Bu40N = np.loadtxt(FR_40[Tstart], dtype='f',skiprows=2)
    FR_Bu45N = np.loadtxt(FR_45[Tstart], dtype='f',skiprows=2)
    FR_Bu50N = np.loadtxt(FR_50[Tstart], dtype='f',skiprows=2)

    for i in np.arange(Tstart+1, Tend):
        FR_Bu05N = FR_Bu05N + np.loadtxt(FR_05[i], dtype='f',skiprows=2)
        FR_Bu10N = FR_Bu10N + np.loadtxt(FR_10[i], dtype='f',skiprows=2)
        FR_Bu15N = FR_Bu15N + np.loadtxt(FR_15[i], dtype='f',skiprows=2)
        FR_Bu20N = FR_Bu20N + np.loadtxt(FR_20[i], dtype='f',skiprows=2)
        FR_Bu25N = FR_Bu25N + np.loadtxt(FR_25[i], dtype='f',skiprows=2)
        FR_Bu30N = FR_Bu30N + np.loadtxt(FR_30[i], dtype='f',skiprows=2)
        FR_Bu35N = FR_Bu35N + np.loadtxt(FR_35[i], dtype='f',skiprows=2)
        FR_Bu40N = FR_Bu40N + np.loadtxt(FR_40[i], dtype='f',skiprows=2)
        FR_Bu45N = FR_Bu45N + np.loadtxt(FR_45[i], dtype='f',skiprows=2)
        FR_Bu50N = FR_Bu50N + np.loadtxt(FR_50[i], dtype='f',skiprows=2)

    FR_Bu05N_TA = K2C(FR_Bu05N/(Tend - Tstart))
    FR_Bu10N_TA = K2C(FR_Bu10N/(Tend - Tstart))
    FR_Bu15N_TA = K2C(FR_Bu15N/(Tend - Tstart))
    FR_Bu20N_TA = K2C(FR_Bu20N/(Tend - Tstart))
    FR_Bu25N_TA = K2C(FR_Bu25N/(Tend - Tstart))
    FR_Bu30N_TA = K2C(FR_Bu30N/(Tend - Tstart))
    FR_Bu35N_TA = K2C(FR_Bu35N/(Tend - Tstart))
    FR_Bu40N_TA = K2C(FR_Bu40N/(Tend - Tstart))
    FR_Bu45N_TA = K2C(FR_Bu45N/(Tend - Tstart))
    FR_Bu50N_TA = K2C(FR_Bu50N/(Tend - Tstart))

    D12 = np.array([FR_Bu05N_TA,FR_Bu10N_TA,FR_Bu15N_TA,FR_Bu20N_TA,FR_Bu25N_TA,
                FR_Bu30N_TA,FR_Bu35N_TA,FR_Bu40N_TA,FR_Bu45N_TA,FR_Bu50N_TA])
    
    return(D12)
# reference state of the buoyancy data (the temp differences over height)
DR_R140 = D2_N(R140, 0, T0)

#%% instantaneous exploration:
# choose a time when the fan is working
def IN_data(filedir, T_fan):
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

    FR_Bu05N = K2C(np.loadtxt(FR_05[T_fan], dtype='f',skiprows=2)) - DR_R140[0]
    FR_Bu10N = K2C(np.loadtxt(FR_10[T_fan], dtype='f',skiprows=2)) - DR_R140[1]
    FR_Bu15N = K2C(np.loadtxt(FR_15[T_fan], dtype='f',skiprows=2)) - DR_R140[2]
    FR_Bu20N = K2C(np.loadtxt(FR_20[T_fan], dtype='f',skiprows=2)) - DR_R140[3]
    FR_Bu25N = K2C(np.loadtxt(FR_25[T_fan], dtype='f',skiprows=2)) - DR_R140[4]
    FR_Bu30N = K2C(np.loadtxt(FR_30[T_fan], dtype='f',skiprows=2)) - DR_R140[5]
    FR_Bu35N = K2C(np.loadtxt(FR_35[T_fan], dtype='f',skiprows=2)) - DR_R140[6]
    FR_Bu40N = K2C(np.loadtxt(FR_40[T_fan], dtype='f',skiprows=2)) - DR_R140[7]
    FR_Bu45N = K2C(np.loadtxt(FR_45[T_fan], dtype='f',skiprows=2)) - DR_R140[8]
    FR_Bu50N = K2C(np.loadtxt(FR_50[T_fan], dtype='f',skiprows=2)) - DR_R140[9]

    IN_R140 = np.array([FR_Bu05N,FR_Bu10N,FR_Bu15N,FR_Bu20N,
                    FR_Bu25N,FR_Bu30N,FR_Bu35N,FR_Bu40N,FR_Bu45N,FR_Bu50N])
    return(IN_R140)
T_fan = 750
IN_R140 = IN_data(R140, T_fan)   # get the instantaneous data
IN_10_R140 = IN_R140[1]          # 1 meter heght data
INEF_10_R140 = IN_10_R140/(TC_hub - DR_R140[1])*100    # calculate the efficiency

#%% plot the 2D distribution data
X1 = np.arange(0,701,2)
Y1 = np.arange(0,701,2)
X2, Y2 = np.meshgrid(X1, Y1)

def plt_plot_bivariate_normal_pdf(x, y, z):
  fig = plt.figure(figsize=(12, 6))
  ax = fig.gca(projection='3d')
  suf = ax.plot_surface(x, y, z, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=True)
  ax.bar(X1, z[:,175], zs=0, zdir='x', color='y', alpha=0.8)
  ax.bar(Y1, z[200,:], zs=700, zdir='y', color='k', alpha=0.8)
  ax.set_zlim(0, 75)
  ax.set_ylim(0, 700)
  ax.set_xlim(0, 700)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('Efficiency [%]');
  fig.colorbar(suf, shrink=0.5, aspect=7)
  plt.show()

# plt_plot_bivariate_normal_pdf(X2,Y2,INEF_10_R140)

#%% get the time averaged data during operation
# DO_R140 = D2_N(R140, T0, T1)            # operation data
# operational aeverage data
OAEF_10_R140 = (DO_R140[1]-DR_R140[1])/(TC_hub - DR_R140[1])*100    # calculate the efficiency
plt_plot_bivariate_normal_pdf(X2,Y2,OAEF_10_R140)
#%%
fig, ax = plt.subplots()
for i in np.arange(180, 211, 5):
    plt.plot(Y1, OAEF_10_R140[i,:], color = "r", lw = (i-180)/10, label = "y = %s m" %(i*2))
ax.set_ylim(0, 45)
ax.set_xlim(0, 700)
ax.set_xlabel('x')
ax.set_ylabel('EF [\%]')
legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
plt.show()

# %%
# the random data

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.contourf(X1, Y1, OAEF_10_R140, cmap=cm.coolwarm)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(X1)), np.max(np.fabs(Y1))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((0, lim))
axScatter.set_ylim((0, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)
pos = np.arange(0,351,10)
data1 = [OAEF_10_R140[:,std] for std in pos]
data2 = [OAEF_10_R140[std,:] for std in pos]
axHistx.violinplot(data1, np.array(pos)*2, points=800, widths=2, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
axHisty.violinplot(data2, np.array(pos)*2, points=800, widths=2, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5, vert=False)
# axHistx.boxplot(data1)
# axHisty.boxplot(data2, vert=False)

# axes[0, 2].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
#                       showextrema=True, showmedians=True, bw_method=0.5)
# axes[0, 2].set_title('Custom violinplot 3', fontsize=fs)

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()
# %% get space averaged data -- and do 3D plot
OAEF_R140 = (DO_R140-DR_R140)/(TC_hub - DR_R140)*100
XY_data = np.mean(OAEF_R140, axis=0)    # XY data
ZY_data = np.mean(OAEF_R140, axis=1)    # ZY data
ZX_data = np.mean(OAEF_R140, axis=2)    # ZX data

X1 = np.arange(0,701,2)
Y1 = np.arange(0,701,2)
Z1 = np.arange(0.5,5.1,0.5)

XY_X, XY_Y = np.meshgrid(X1, Y1)
ZY_Y, ZY_Z = np.meshgrid(Y1, Z1)
ZX_X, ZX_Z = np.meshgrid(X1, Z1)

fig = plt.figure(figsize=(11, 10),constrained_layout=True)
widths = [8, 2]
heights = [8, 2]
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)
ax1 = fig.add_subplot(spec[0, 0])
cs = ax1.contourf(XY_X, XY_Y, XY_data, cmap=cm.coolwarm)
ax2 = fig.add_subplot(spec[1, 0])
ax2.contourf(ZY_Y, ZY_Z, ZY_data, cmap=cm.coolwarm)
ax3 = fig.add_subplot(spec[0, 1])
ax3.contourf(ZX_Z, ZX_X, ZX_data, cmap=cm.coolwarm)
fig.colorbar(cs, ax=ax3, shrink=0.9)

# %%
