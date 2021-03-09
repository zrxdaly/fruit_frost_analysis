# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Feb 01 2021
#
# OBJ: This function file provides analysis of goal function for quantifying the efficiency of ventilator
# @author: Dai
#
import matplotlib.pyplot as plt
import numpy as np
import glob as glob

from matplotlib.ticker import NullFormatter

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.ticker as mtick

nullfmt = NullFormatter()

#%% the output also require same amount of typing 
# def plot_P():
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='20')
plt.rc('ytick', labelsize='20')
# plt.rc('text', usetex=True)
#     lw = 2
#     Color_R2 = '#000000'
#     Color_R3 = '#fcf18f'
#     Color_R4 = '#377eb8'
#     Color_R5 = '#008000'
#     Color_R6 = '#084594'
#     Color_R7 = '#ff7f00'
#     Color_R8 = '#808080'

# # import file function (simulation type)
# def im_file(run):









# function to show the legend of contour with unit of e
def ticks(y, pos):
    return r'$e^{:.0f}$'.format(np.log(y))

# 3D projection of averaged EF
def Slice3_AV(OAEF_R140, txt):
    L0 = 800
    XY_data = np.mean(OAEF_R140, axis=0)    # XY data
    ZY_data = np.mean(OAEF_R140, axis=1)    # ZY data
    ZX_data = np.mean(OAEF_R140, axis=2)    # ZX data
    zlen = ZY_data.shape[0]
    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)
    if zlen == 10:
        Z1 = np.arange(0.5, 5.1, 0.5)
    else:
        Z1 = np.arange(1, 5.1, 1)

    XY_X, XY_Y = np.meshgrid(X1, Y1)
    ZY_Y, ZY_Z = np.meshgrid(Y1, Z1)
    ZX_X, ZX_Z = np.meshgrid(X1, Z1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    widths = [8, 2]
    heights = [8, 2]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                            height_ratios=heights)
    ax1 = fig.add_subplot(spec[0, 0])
    cs = ax1.contourf(XY_X, XY_Y, XY_data, cmap=cm.coolwarm)
    ax1.set_ylabel("y [m]", fontsize=18)
    ax1.set_title('%s operation' %txt)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.contourf(ZX_Z, ZX_X, ZX_data, cmap=cm.coolwarm)
    ax2.set_xlabel("z [m]", fontsize=18)

    ax3 = fig.add_subplot(spec[1, 0])
    ax3.contourf(ZY_Y, ZY_Z, ZY_data, cmap=cm.coolwarm)
    ax3.set_xlabel("x [m]", fontsize=18)
    ax3.set_ylabel("z [m]", fontsize=18)

    fig.colorbar(cs, ax=[ax1, ax2, ax3], shrink=0.9)
    ax1.xaxis.set_major_formatter(nullfmt)
    ax2.yaxis.set_major_formatter(nullfmt)
    plt.savefig("UVW_output/WALL_%s_dis.pdf"%txt)
    # plt.tight_layout()

# 3D projection of averaged EF with clipping of negative value
def Slice3_PEF(OAEF_R140, txt):
    L0 = 800
    OAEF_R140[OAEF_R140>6] = np.nan
    XY_data = np.nanmean(OAEF_R140, axis=0)    # XY data
    ZY_data = np.nanmean(OAEF_R140, axis=1)    # ZY data
    ZX_data = np.nanmean(OAEF_R140, axis=2)    # ZX data
    zlen = ZY_data.shape[0]
    X1 = np.arange(0, L0+1, 2)
    Y1 = np.arange(0, L0+1, 2)
    if zlen == 10:
        Z1 = np.arange(0.5, 5.1, 0.5)
    else:
        Z1 = np.arange(1, 5.1, 1)
    XY_X, XY_Y = np.meshgrid(X1, Y1)
    ZY_Y, ZY_Z = np.meshgrid(Y1, Z1)
    ZX_X, ZX_Z = np.meshgrid(X1, Z1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    widths = [8, 2]
    heights = [8, 2]
    levels = np.arange(0,1.5,0.02)
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
    plt.savefig("UVW_output/WALL_%s_PEF.pdf"%txt)


#%% the function used to calcuate the coverage volume over space
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

#%% plot warming area for different limits
def plotRC(num, txt):
    ROTA_P = [60, 80, 96, 120, 160, 192, 240, 320, 480]
    ii = 0
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = fig.add_subplot(1,1,1)
    # various scale of limits: dense in lower limits and sparse in high limits
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
    plt.title("%s operation"%txt,fontsize=18)
    # plt.tight_layout()
    # plt.savefig("vr_rerun/%sRC.pdf"%txt)