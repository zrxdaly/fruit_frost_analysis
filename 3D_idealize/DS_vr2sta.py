#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:10:01 2020

OBJ: load the data from vrlab and do statistical analysis for:
            warming effect from different operations and different rotation period

@author: dai
"""
# %% import module

import matplotlib.pyplot as plt
import numpy as np
import glob as glob

from matplotlib.ticker import NullFormatter

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm 
from scipy import optimize

import matplotlib.ticker as mtick

def ticks(y, pos):
    return r'$e^{:.0f}$'.format(np.log(y))

#%%
nullfmt = NullFormatter()

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='20')
plt.rc('ytick', labelsize='20')
plt.rc('text', usetex=True)
lw = 2
Color_R2 = '#000000'
Color_R3 = '#fcf18f'
Color_R4 = '#377eb8'
Color_R5 = '#008000'
Color_R6 = '#084594'
Color_R7 = '#ff7f00'
Color_R8 = '#808080'
# %% comparison of different runs
# run = "run" 
run = "rerun"
if run == "run":
    dir1 = '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/vrlab_/'
    Goal_dir = sorted(glob.glob(dir1 + '*ROTA/'))
    L0 = 700
elif run == "rerun":
    dir1 = '/home/dai/Desktop/trans/3D_idealize/PARA/ROTA/vrlab_rerun/'
    Goal_dir = sorted(glob.glob(dir1 + '*R/'))
    L0 = 800

FR_dir = sorted(glob.glob(Goal_dir[0]+"OAEF*.npy"))
HR_dir = sorted(glob.glob(Goal_dir[1]+"OAEF*.npy"))
RR_dir = sorted(glob.glob(Goal_dir[2]+"OAEF*.npy"))

RREF_R060 = np.load(RR_dir[0])
RREF_R080 = np.load(RR_dir[1])
RREF_R096 = np.load(RR_dir[2])
RREF_R120 = np.load(RR_dir[3])
RREF_R160 = np.load(RR_dir[4])
RREF_R192 = np.load(RR_dir[5])
RREF_R240 = np.load(RR_dir[6])
RREF_R320 = np.load(RR_dir[7])
RREF_R480 = np.load(RR_dir[8])

ALL_RREF = np.array([RREF_R060,RREF_R080,RREF_R096,
                     RREF_R120,RREF_R160,RREF_R192,
                     RREF_R240,RREF_R320,RREF_R480])

FREF_R060 = np.load(FR_dir[0])
FREF_R080 = np.load(FR_dir[1])
FREF_R096 = np.load(FR_dir[2])
FREF_R120 = np.load(FR_dir[3])
FREF_R160 = np.load(FR_dir[4])
FREF_R192 = np.load(FR_dir[5])
FREF_R240 = np.load(FR_dir[6])
FREF_R320 = np.load(FR_dir[7])
FREF_R480 = np.load(FR_dir[8])

HREF_R060 = np.load(HR_dir[0])
HREF_R080 = np.load(HR_dir[1])
HREF_R096 = np.load(HR_dir[2])
HREF_R120 = np.load(HR_dir[3])
HREF_R160 = np.load(HR_dir[4])
HREF_R192 = np.load(HR_dir[5])
HREF_R240 = np.load(HR_dir[6])
HREF_R320 = np.load(HR_dir[7])
HREF_R480 = np.load(HR_dir[8])

Y1 = np.arange(0,L0+1,2)

# %% first do statistics for one operation and one period

# %% 3D projection plot the the EF with extraction of positive or not
def Slice3_AV(OAEF_R140, txt):
    XY_data = np.mean(OAEF_R140, axis=0)    # XY data
    ZY_data = np.mean(OAEF_R140, axis=1)    # ZY data
    ZX_data = np.mean(OAEF_R140, axis=2)    # ZX data

    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)
    Z1 = np.arange(0.5, 5.1, 0.5)

    XY_X, XY_Y = np.meshgrid(X1, Y1)
    ZY_Y, ZY_Z = np.meshgrid(Y1, Z1)
    ZX_X, ZX_Z = np.meshgrid(X1, Z1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    widths = [8, 2]
    heights = [8, 2]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                            height_ratios=heights)
    ax1 = fig.add_subplot(spec[0, 0])
    cs = ax1.contourf(XY_X, XY_Y, XY_data, cmap=cm.coolwarm, vmin=0, vmax=50)
    ax1.set_ylabel("y [m]", fontsize=18)
    ax1.set_title('%s operation' %txt)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.contourf(ZX_Z, ZX_X, ZX_data, cmap=cm.coolwarm, vmin=0, vmax=50)
    ax2.set_xlabel("z [m]", fontsize=18)

    ax3 = fig.add_subplot(spec[1, 0])
    ax3.contourf(ZY_Y, ZY_Z, ZY_data, cmap=cm.coolwarm, vmin=0, vmax=50)
    ax3.set_xlabel("x [m]", fontsize=18)
    ax3.set_ylabel("z [m]", fontsize=18)

    fig.colorbar(cs, ax=[ax1, ax2, ax3], shrink=0.9)
    ax1.xaxis.set_major_formatter(nullfmt)
    ax2.yaxis.set_major_formatter(nullfmt)
    plt.savefig("vr_rerun/WALL_%s_dis.pdf"%txt)
    # plt.tight_layout()

# %%
Slice3_AV(RREF_R160, 'RR')
Slice3_AV(FREF_R160, 'FR')
Slice3_AV(HREF_R160, 'AR')

#%% Slice with extraction of positive EF
def Slice3_PEF(OAEF_R140, txt):
    OAEF_R140[OAEF_R140<1] = np.nan
    XY_data = np.nanmean(OAEF_R140, axis=0)    # XY data
    ZY_data = np.nanmean(OAEF_R140, axis=1)    # ZY data
    ZX_data = np.nanmean(OAEF_R140, axis=2)    # ZX data

    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)
    Z1 = np.arange(0.5, 5.1, 0.5)

    XY_X, XY_Y = np.meshgrid(X1, Y1)
    ZY_Y, ZY_Z = np.meshgrid(Y1, Z1)
    ZX_X, ZX_Z = np.meshgrid(X1, Z1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    widths = [8, 2]
    heights = [8, 2]
    levels = np.arange(0,81,10)
    cmap_p = cm.coolwarm
    # cmap_p = cm.cool_r
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                            height_ratios=heights)
    ax1 = fig.add_subplot(spec[0, 0])
    cs = ax1.contourf(XY_X, XY_Y, XY_data, cmap=cmap_p, levels = levels)
    ax1.set_ylabel("y [m]", fontsize=24)
    ax1.set_title('%s operation' %txt, fontsize=24)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.contourf(ZX_Z, ZX_X, ZX_data, cmap=cmap_p, levels = levels)
    ax2.set_xlabel("z [m]", fontsize=24)

    ax3 = fig.add_subplot(spec[1, 0])
    ax3.contourf(ZY_Y, ZY_Z, ZY_data, cmap=cmap_p, levels = levels)
    ax3.set_xlabel("x [m]", fontsize=24)
    ax3.set_ylabel("z [m]", fontsize=24)
    # plt.colorbar()
    fig.colorbar(cs, ax=[ax1, ax2, ax3], shrink=0.9)
    ax1.xaxis.set_major_formatter(nullfmt)
    ax2.yaxis.set_major_formatter(nullfmt)
    plt.savefig("vr_rerun/WALL_%s_EF.pdf"%txt)

# %%
Slice3_PEF(RREF_R160, 'RR')
Slice3_PEF(FREF_R160, 'FR')
Slice3_PEF(HREF_R160, 'AR')







#%% height comparison 
def SliceXY_H(OAEF_R140, txt, P):
    if P == "P":
        OAEF_R140[OAEF_R140<1] = np.nan
    
    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)

    XY_X, XY_Y = np.meshgrid(X1, Y1)
    levels = np.arange(0,81,10) 
    fig, ax = plt.subplots(4, 3, figsize = (12,12))
    xmin, xmax, ymin, ymax = 300, 500, 100, 300
    for i in np.arange(np.shape(OAEF_R140)[0]-1):
        ax[i//3,i%3].contourf(XY_X, XY_Y, OAEF_R140[i], cmap=cm.coolwarm, levels = levels)
        ax[i//3,i%3].xaxis.set_major_formatter(nullfmt)
        ax[i//3,i%3].set_xlim([xmin, xmax])
        ax[i//3,i%3].set_ylim([ymin, ymax])
    ax[3,0].contourf(XY_X, XY_Y, OAEF_R140[9], cmap=cm.coolwarm, levels = levels)
    # ax[3,0].set_xlim([xmin, xmax])
    # ax[3,0].set_ylim([ymin, ymax])
    plt.tight_layout()

# SliceXY_H(RREF_R060, "RR", "P")
# SliceXY_H(RREF_R080, "RR", "P")
# SliceXY_H(RREF_R096, "RR", "P")
# SliceXY_H(RREF_R120, "RR", "P")
# SliceXY_H(RREF_R160, "RR", "P")
# SliceXY_H(RREF_R192, "RR", "P")
# SliceXY_H(RREF_R240, "RR", "P")
# SliceXY_H(RREF_R320, "RR", "P")
# SliceXY_H(RREF_R480, "RR", "P")

#%% rotation period comparison 
def SliceXY_tau(ALL_RREF, H, txt):
    ALL_RREF[ALL_RREF<1] = np.nan
    
    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)

    XY_X, XY_Y = np.meshgrid(X1, Y1)
    levels = np.arange(0,81,10) 
    fig, ax = plt.subplots(3, 3, figsize = (12,10), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    xmin, xmax, ymin, ymax = 300, 500, 100, 300
    for i in np.arange(np.shape(ALL_RREF)[0]):
        ax[i//3,i%3].contourf(XY_X, XY_Y, ALL_RREF[i,H,:], cmap=cm.coolwarm, levels = levels)

    # for axi in ax.flat:
    #     axi.xaxis.set_major_locator(plt.MaxNLocator(2))
    #     axi.yaxis.set_major_locator(plt.MaxNLocator(2))
        # axi.set_xlim([xmin, xmax])
        # axi.set_ylim([ymin, ymax])
    plt.tight_layout()
    plt.savefig("vr_rerun/Tau_%s_%sm.pdf"%(txt,H/2+0.5))

SliceXY_tau(ALL_RREF, 0, "RR")
SliceXY_tau(ALL_RREF, 3, "RR")
SliceXY_tau(ALL_RREF, 7, "RR")
SliceXY_tau(ALL_RREF, 9, "RR")

#%%
















#%% plot the warming volume 
# as factor of different warming intensity and rotation period 

# def EF2Vo(RREF_R060, LIM):
#     # get a number A * 700 * 700 volume
#     return(np.shape(RREF_R060[RREF_R060>LIM])[0]/(10*351*351)*5)

def EF2Vo(RREF_R060, LIM):
    # get a number A * A * 5: this way we get the horizontal coverage area
    size = L0/2+1
    return((np.shape(RREF_R060[RREF_R060>LIM])[0]/(10*size*size))**(0.5)*L0)

# 5 is the lower limit of the warming efficiency
# LIM = 5
def LIM2L(LIM):
    VoL_RR = [EF2Vo(RREF_R060,LIM), EF2Vo(RREF_R080,LIM), EF2Vo(RREF_R096,LIM),
            EF2Vo(RREF_R120,LIM), EF2Vo(RREF_R160,LIM), EF2Vo(RREF_R192,LIM),
            EF2Vo(RREF_R240,LIM), EF2Vo(RREF_R320,LIM), EF2Vo(RREF_R480,LIM)]
    VoL_FR = [EF2Vo(FREF_R060,LIM), EF2Vo(FREF_R080,LIM), EF2Vo(FREF_R096,LIM),
            EF2Vo(FREF_R120,LIM), EF2Vo(FREF_R160,LIM), EF2Vo(FREF_R192,LIM),
            EF2Vo(FREF_R240,LIM), EF2Vo(FREF_R320,LIM), EF2Vo(FREF_R480,LIM)]
    VoL_HR = [EF2Vo(HREF_R060,LIM), EF2Vo(HREF_R080,LIM), EF2Vo(HREF_R096,LIM),
            EF2Vo(HREF_R120,LIM), EF2Vo(HREF_R160,LIM), EF2Vo(HREF_R192,LIM),
            EF2Vo(HREF_R240,LIM), EF2Vo(HREF_R320,LIM), EF2Vo(HREF_R480,LIM)]
    return(VoL_RR, VoL_FR, VoL_HR)

# VoL_RR_1, VoL_FR_1, VoL_HR_1 = LIM2L(5)
#%% plot warming area for different limits
def plotRC(num, txt):
    ROTA_P = [60, 80, 96, 120, 160, 192, 240, 320, 480]
    ii = 0
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = fig.add_subplot(1,1,1)
    arry = np.append(np.arange(1,10,2), np.arange(11,61,10))
    for i in arry:
        ii = ii + 1
        plt.plot(ROTA_P, LIM2L(i)[num], "-*", label=str(i)+'%', linewidth = ii/2) 
        # np.save("PARA/ROTA/%s_%s.npy" %(txt, i),LIM2L(i)[num])  # uncomment this one to save the volume data
    ax.set_xlabel('Rotation period [s]',fontsize=18)
    # ax.set_ylabel("Coverage Volume [**2*5m3]",fontsize=18)
    ax.set_ylim([0,650])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 20)
    # ,fontsize = 20
    plt.title("%s operation"%txt,fontsize=18)
    # plt.tight_layout()
    # plt.savefig("vr_rerun/%sRC.pdf"%txt)
#%%
plotRC(0, 'RR')
#%%
plotRC(1, 'FR')
#%%
plotRC(2, 'HR')
#%% to get some statistics over the distribution of the statistics






























#%%





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
  ax.bar(Y1, z[189,:], zs=700, zdir='y', color='k', alpha=0.8)
  ax.set_zlim(0, 75)
  ax.set_ylim(0, 700)
  ax.set_xlim(0, 700)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('Efficiency [%]');
  fig.colorbar(suf, shrink=0.5, aspect=7)
  plt.savefig('RR_R160_1m.pdf')

RREF_10_R160 = RREF_R160[1,:]
plt_plot_bivariate_normal_pdf(X2,Y2,RREF_10_R160)

#%% 




# %% this is the plotting function for just one height
X1 = np.arange(0, 701, 2)
Y1 = np.arange(0, 701, 2)
X2, Y2 = np.meshgrid(X1, Y1)


def plt_plot_bivariate_normal_pdf(x, y, z):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')
    suf = ax.plot_surface(x, y, z,
                          cmap=cm.coolwarm,
                          linewidth=0,
                          antialiased=True)
    ax.bar(X1, z[:, 175], zs=0, zdir='x', color='y', alpha=0.8)
    ax.bar(Y1, z[189, :], zs=700, zdir='y', color='k', alpha=0.8)
    ax.set_zlim(0, 75)
    ax.set_ylim(0, 700)
    ax.set_xlim(0, 700)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Efficiency [%]')
    fig.colorbar(suf, shrink=0.5, aspect=7)
    plt.show()

# plt_plot_bivariate_normal_pdf(X2,Y2,INEF_10_R140)

# %% get the time averaged data during operation


# %%
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
pos = np.arange(0, 351, 10)
data1 = [OAEF_10_R140[:, std] for std in pos]
data2 = [OAEF_10_R140[std, :] for std in pos]
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
