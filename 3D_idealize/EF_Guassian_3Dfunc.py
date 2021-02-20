# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Feb 01 2021
#
# OBJ: This file explore the possiblity of 3D function of 
# warming efficiency for different operation parameters
# @author: Dai
#
#%% import file
import matplotlib.pyplot as plt
import numpy as np
import glob as glob

from matplotlib.ticker import NullFormatter

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm 
from scipy import optimize
from scipy.special import gamma as gammafun

from scipy.stats import gamma as gammafit

### ticker modification
import matplotlib.ticker as mtick

from operator import itemgetter
from itertools import groupby

def ticks(y, pos):
    return r'$e^{:.0f}$'.format(np.log(y))

def dxtick(ax):
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

#%%
nullfmt = NullFormatter()
# ax2.yaxis.set_major_formatter(nullfmt, mtick.FuncFormatter(ticks))  
# 1. x, y axis without ticker
# 2. with mtick, the log scale can be set


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

#%% get the factor of z in to the equation
Y1 = np.arange(0,L0+1,2)
def gaussian(x, amplitude, mean, stddev):
    return(amplitude * np.exp(-((x - mean) / (4) / stddev)**2))

def norm_data(RREF_R160,H_index):
    popt, _ = optimize.curve_fit(gaussian, Y1, RREF_R160[H_index,100])
    for i in np.arange(101, int(L0/2)+1, 1):
        # 1 meter fitting
        popt1, _ = optimize.curve_fit(gaussian, Y1, RREF_R160[H_index,i])
        popt = np.vstack((popt,popt1))
    return(popt)

#%%
def GUA3D_EF(RREF_R160,H_index):
    
    popt_R160 = norm_data(RREF_R160, H_index)
    #%% plot the statistic of std
    # popt_R160 (amplitude, mean, std)
    Y_range = np.arange(200, 801, 2)
    fig, ax = plt.subplots(3, 1, figsize = (6,12))
    ax[0].plot(Y_range, popt_R160[:,0])
    ax[0].set_xlabel('distance (y)', fontsize=18)
    ax[0].set_ylabel('EF [\%]', fontsize=18)

    ax[1].plot(Y_range, popt_R160[:,1])
    ax[1].set_xlabel('distance (y)', fontsize=18)
    ax[1].set_ylabel('mean', fontsize=18)

    ax[2].plot(Y_range, popt_R160[:,2])
    ax[2].set_xlabel('distance (y)', fontsize=18)
    ax[2].set_ylabel('std', fontsize=18)
    strr = (H_index+1)/2
    plt.suptitle("height at %sm"%strr, fontsize=18)
    plt.tight_layout()

#%%
GUA3D_EF(RREF_R160, 0)
GUA3D_EF(RREF_R160, 1)
GUA3D_EF(RREF_R160, 2)
GUA3D_EF(RREF_R160, 3)
GUA3D_EF(RREF_R160, 4)
GUA3D_EF(RREF_R160, 5)
GUA3D_EF(RREF_R160, 6)
GUA3D_EF(RREF_R160, 7)
GUA3D_EF(RREF_R160, 8)
GUA3D_EF(RREF_R160, 9)


#%% put all the plot together
def GUA3D_ALL(RREF_R160, txt1):
    fig, ax = plt.subplots(3, 1, figsize = (6,12))
    Y_range = np.arange(200, L0+1, 2)
    for h in np.arange(0,10,1):
        S_Scale = np.e
        popt_R160 = norm_data(RREF_R160, h)
        ax[0].plot(Y_range, popt_R160[:,0])
        # ax[0].loglog(Y_range, popt_R160[:,0], basex=S_Scale, basey=S_Scale)
        # ax[0].set_xscale('log', basex=S_Scale)
        # ax[0].set_yscale('log', basey=S_Scale)
        ax[0].set_ylabel('A [\%]', fontsize=18)
        ax[0].xaxis.set_major_formatter(nullfmt)
        # ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(ticks))

        ax[1].plot(Y_range, popt_R160[:,1])
        # ax[1].set_xlabel('distance (y)', fontsize=18)
        ax[1].set_ylabel('B', fontsize=18)
        ax[1].xaxis.set_major_formatter(nullfmt)

        ax[2].plot(Y_range, np.abs(popt_R160[:,2]))
        ax[2].set_xlabel('distance (y)', fontsize=18)
        ax[2].set_ylabel('abs(C)', fontsize=18)
    plt.suptitle("%s"%txt1, fontsize=18)
    plt.tight_layout()
    plt.savefig("vr_rerun/%s_%s.pdf"%(txt1,run))

# GUA3D_ALL(RREF_R060, "R060")
# GUA3D_ALL(RREF_R080, "R080")
# GUA3D_ALL(RREF_R096, "R096")
# GUA3D_ALL(RREF_R120, "R120")
# GUA3D_ALL(RREF_R160, "R160")
# GUA3D_ALL(RREF_R192, "R192")
# GUA3D_ALL(RREF_R240, "R240")
# GUA3D_ALL(RREF_R320, "R320")
# GUA3D_ALL(RREF_R480, "R480")

#%%% explore the factor of B

#%% 2D plot of the mean 
xx = np.arange(200, 801, 2)
yy = np.arange(0.5, 5.1, 0.5)
XX, YY = np.meshgrid(xx, yy)

cmap_p = cm.coolwarm

def GUA3D2Dplot_ALL(RREF_R160, txt1):
    Y_range = np.arange(200, L0+1, 2)
    mag_2D = []
    mean_2D = []
    std_2D = []
    for h in np.arange(0,10,1):
        popt_R160 = norm_data(RREF_R160, h)
        mag_2D.append(popt_R160[:,0])
        mean_2D.append(popt_R160[:,1])
        std_2D.append(popt_R160[:,2])
    # fig, ax = plt.subplots(3, 1, figsize = (6,9))
    # cs1 = ax[0].contourf(XX, YY, mag_2D, cmap=cmap_p)
    # ax[0].xaxis.set_major_formatter(nullfmt)
    # cbar1 = fig.colorbar(cs1, ax=ax[0], shrink=0.9)
    # cbar1.ax.set_title("A")
    
    # cs2 = ax[1].contourf(XX, YY, mean_2D, cmap=cmap_p)
    # ax[1].xaxis.set_major_formatter(nullfmt)
    # cbar2 = fig.colorbar(cs2, ax=ax[1], shrink=0.9)
    # cbar2.ax.set_title("B")

    # cs3 = ax[2].contourf(XX, YY, np.abs(std_2D), cmap=cmap_p)
    # cbar3 = fig.colorbar(cs3, ax=ax[2], shrink=0.9)
    # cbar3.ax.set_title("abs(C)")
    # plt.tight_layout()

    # plt.savefig("vr_rerun/2D%s_%s.pdf"%(txt1,run))
    return(mag_2D, mean_2D, np.abs(std_2D))

#%%
A_R060, B_R060, C_R060 = GUA3D2Dplot_ALL(RREF_R060, "R060")
A_R080, B_R080, C_R080 = GUA3D2Dplot_ALL(RREF_R080, "R080")
A_R096, B_R096, C_R096 = GUA3D2Dplot_ALL(RREF_R096, "R096")
A_R120, B_R120, C_R120 = GUA3D2Dplot_ALL(RREF_R120, "R120")
A_R160, B_R160, C_R160 = GUA3D2Dplot_ALL(RREF_R160, "R160")
A_R192, B_R192, C_R192 = GUA3D2Dplot_ALL(RREF_R192, "R192")
A_R240, B_R240, C_R240 = GUA3D2Dplot_ALL(RREF_R240, "R240")
A_R320, B_R320, C_R320 = GUA3D2Dplot_ALL(RREF_R320, "R320")
A_R480, B_R480, C_R480 = GUA3D2Dplot_ALL(RREF_R480, "R480")

#%%
def cut_range(B_R060, H):
    B_05 = B_R060[H]
    index_P = [idx for idx, num in enumerate(B_05) if num > 300]
    ranges = []
    len0 = 0
    if len(index_P) == 0:
        return(np.array([0,0]))
    else:
        for k, g in groupby(enumerate(index_P), lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            if len(group) > len0:
                len0, ranges = len(group), np.array([group[0],group[-1]])
        return(ranges*2+200)

cut_range(B_R060, 2)
    # MAX_DIFF = 300
    # diff_B_05 = np.diff(B_05)
    # index_P = [idx for idx, num in enumerate(B_05) if num > 0]
    # # index_min = [idx for idx, num in enumerate(diff_B_05) if num < -MAX_DIFF]
    # if len(index_max) == 0:
    #     init_y = 200
    # else:
    #     init_y = min(index_max)*2 + 200
    # if len(index_min) == 0:
    #     end_y = 800
    # else:
    #     end_y = max(index_min)*2 + 200
    # if end_y < 600:
    #     end_y = 800
    # return(init_y, end_y)
#%%
def plot_diff(B_R060, H):
    diff_B = np.diff(B_R060[H])
    plt.plot(diff_B)

def rangeplot(H):
    init_R060, end_R060 = cut_range(B_R060, H)
    init_R080, end_R080 = cut_range(B_R080, H)
    init_R096, end_R096 = cut_range(B_R096, H)
    init_R120, end_R120 = cut_range(B_R120, H)
    init_R160, end_R160 = cut_range(B_R160, H)
    init_R192, end_R192 = cut_range(B_R192, H)
    init_R240, end_R240 = cut_range(B_R240, H)
    init_R320, end_R320 = cut_range(B_R320, H)
    init_R480, end_R480 = cut_range(B_R480, H)
        
    list_init = [init_R060, init_R080, init_R096,
                init_R120, init_R160, init_R192,
                init_R240, init_R320, init_R480]

    list_end = [end_R060, end_R080, end_R096,
                end_R120, end_R160, end_R192,
                end_R240, end_R320, end_R480]

    ROTA_P = [60, 80, 96, 120, 160, 192, 240, 320, 480]

    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax = fig2.add_subplot(1,1,1)
    h1 = plt.plot(list_init, ROTA_P, "*-", ms = lw*6, linewidth = lw,color = Color_R2)  
    h2 = plt.plot(list_end, ROTA_P, "*-", ms = lw*6, linewidth = lw,color = Color_R2)  
    ax.set_xlim([-5,820])
    ax.set_ylabel('Rotation [s]',fontsize=18)
    ax.set_xlabel('y distance [m]',fontsize=18)
    plt.title("height %sm"%(H/2+0.5),fontsize=18)
    # legend = plt.legend(loc='upper left', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("range_%sm.pdf"%H)

# rangeplot(0)
# rangeplot(1)
# rangeplot(2)
# rangeplot(3)
# rangeplot(4)
# rangeplot(5)
# rangeplot(6)
# rangeplot(7)
# rangeplot(8)
# rangeplot(9)

#%%
def rangeplot_RP(B_R060, txt):
    init_H0, end_H0 = cut_range(B_R060, 0)
    init_H1, end_H1 = cut_range(B_R060, 1)
    init_H2, end_H2 = cut_range(B_R060, 2)
    init_H3, end_H3 = cut_range(B_R060, 3)
    init_H4, end_H4 = cut_range(B_R060, 4)
    init_H5, end_H5 = cut_range(B_R060, 5)
    init_H6, end_H6 = cut_range(B_R060, 6)
    init_H7, end_H7 = cut_range(B_R060, 7)
    init_H8, end_H8 = cut_range(B_R060, 8)
    init_H9, end_H9 = cut_range(B_R060, 9)
        
    list_init = [init_H0,
                 init_H1,
                 init_H2,
                 init_H3,
                 init_H4,
                 init_H5,
                 init_H6,
                 init_H7,
                 init_H8,
                 init_H9,]

    list_end = [end_H0,
                end_H1,
                end_H2,
                end_H3,
                end_H4,
                end_H5,
                end_H6,
                end_H7,
                end_H8,
                end_H9,]

    HEI = np.arange(0.5,5.1,0.5)
    # ROTA_P = [60, 80, 96, 120, 160, 192, 240, 320, 480]

    # fig2 = plt.figure(figsize=(6.4, 4.8))
    # ax = fig2.add_subplot(1,1,1)
    # h1 = plt.plot(list_init, HEI, "*-", ms = lw*6, linewidth = lw,color = Color_R2)  
    # h2 = plt.plot(list_end, HEI, "*-", ms = lw*6, linewidth = lw,color = Color_R2)  
    # ax.set_xlim([-5,820])
    # ax.set_ylabel('Height [m]',fontsize=18)
    # ax.set_xlabel('y distance [m]',fontsize=18)
    # plt.title("%s"%txt,fontsize=18)
    # # legend = plt.legend(loc='upper left', frameon=False,fontsize = 18,ncol = 2)
    # plt.tight_layout()
    # plt.savefig("range_%s_RP.pdf"%txt)

    LI_R060 = np.array([HEI,list_init])
    LE_R060 = np.array([HEI,list_end])

    LI_R060 = LI_R060[:,LI_R060[1]>0]
    LE_R060 = LE_R060[:,LE_R060[1]>0]

    return(LI_R060, LE_R060)


# LI_R060, LE_R060 = rangeplot_RP(B_R060, "R060")
# LI_R080, LE_R080 = rangeplot_RP(B_R080, "R080")
# LI_R096, LE_R096 = rangeplot_RP(B_R096, "R096")
# LI_R120, LE_R120 = rangeplot_RP(B_R120, "R120")
# LI_R160, LE_R160 = rangeplot_RP(B_R160, "R160")
# LI_R192, LE_R192 = rangeplot_RP(B_R192, "R192")
# LI_R240, LE_R240 = rangeplot_RP(B_R240, "R240")
# LI_R320, LE_R320 = rangeplot_RP(B_R320, "R320")
# LI_R480, LE_R480 = rangeplot_RP(B_R480, "R480")
#%% curve fitting 
def linearline(x, a, b):
    return(a * x + b)

def fitting_line(LI_R060, LE_R060, txt):
    fit_expoI, pcovI = optimize.curve_fit(linearline, LI_R060[0], LI_R060[1])
    fit_expoE, pcovE = optimize.curve_fit(linearline, LE_R060[0], LE_R060[1])

    fig, ax = plt.subplots()
    plt.plot(LI_R060[0], LI_R060[1], "o", label = "I data", color = Color_R2)
    plt.plot(LE_R060[0], LE_R060[1], "o", label = "E data", color = Color_R4)
    plt.plot(LI_R060[0], linearline(LI_R060[0], *fit_expoI), label = "I fit", linewidth = lw, color = Color_R2)
    plt.plot(LE_R060[0], linearline(LE_R060[0], *fit_expoE), label = "E fit", linewidth = lw, color = Color_R4)
    ax.set_xlabel('z height [m]', fontsize=18)
    ax.set_ylabel('distance [m]', fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
    plt.title("%s"%txt,fontsize=18)
    plt.tight_layout()
    plt.savefig("fitrange_%s.pdf"%txt)
    print(*fit_expoI, *fit_expoE)
    return([*fit_expoI,*fit_expoE])

PARA_R060 = fitting_line(LI_R060, LE_R060, "R060")
PARA_R080 = fitting_line(LI_R080, LE_R080, "R080")
PARA_R096 = fitting_line(LI_R096, LE_R096, "R096")
PARA_R120 = fitting_line(LI_R120, LE_R120, "R120")
PARA_R160 = fitting_line(LI_R160, LE_R160, "R160")
PARA_R192 = fitting_line(LI_R192, LE_R192, "R192")
PARA_R240 = fitting_line(LI_R240, LE_R240, "R240")

#%%

ROTA_P = np.array([60, 80, 96, 120, 160, 192, 240])

def expoen(x, a, b, c):
    # return(b * np.exp(a * x) + c)
    return(a * x ** 2 + b * x + c) 

def fit_ABC(num, txt):
    PARA_I_A = [PARA_R060[num],
                PARA_R080[num],
                PARA_R096[num],
                PARA_R120[num],
                PARA_R160[num],
                PARA_R192[num],
                PARA_R240[num]]
    fit_expo, pcov = optimize.curve_fit(expoen, ROTA_P, PARA_I_A)
    fig, ax = plt.subplots()
    plt.plot(ROTA_P, PARA_I_A, "o", label = "data", color = Color_R2)
    ax.set_xlabel('rotation period [s]', fontsize=18)
    ax.set_ylabel('%s'%txt, fontsize=18)
    plt.plot(ROTA_P, expoen(ROTA_P, *fit_expo), label = "fit", linewidth = lw, color = Color_R2)
    plt.tight_layout()
    plt.savefig("fit%s.pdf"%txt)
    return([*fit_expo])
PA_IA = fit_ABC(0, "init[A]")
PA_IB = fit_ABC(1, "init[B]")
PA_EA = fit_ABC(2, "end[A]")
PA_EB = fit_ABC(3, "end[B]")


#%% find function of A
# A_R060
# A_R080
# A_R096
# A_R120
# A_R160
# A_R192
# A_R240
# A_R320
# A_R480
fit_alpha, fit_loc, fit_beta=gammafit.fit(A_R060[0]*0.01)


#%%
def gamma1(x, alp, beta, b, c):
    gma = gammafun(alp)
    # the pdf gamma function
    return(beta ** alp * (x) ** (alp - 1) * (np.exp(- beta * (x) + b) + c)/gma)

# def gamma1(x, theta):
#     # gma = gammafun(alp)
#     # the pdf gamma function
#     return(- theta ** x / (x * np.log(1- theta))) 

Y_range = np.arange(1,301.1)
Y_range = Y_range / 301 * 20

def ff_gam(num):
        
    data_A = np.array([Y_range, A_R060[num]*0.01])
    data_A = data_A[:,data_A[1,:]>0]

    fit_gam, pcov_gma = optimize.curve_fit(gamma1, data_A[0], data_A[1])

    fig, ax = plt.subplots()
    plt.plot(data_A[0], data_A[1], "+", label = "data", color = Color_R2)
    plt.plot(data_A[0], gamma1(data_A[0], *fit_gam), label = "gamma fit", linewidth = lw, color = Color_R2)
    ax.set_xlabel('y distance [m]', fontsize=18)
    ax.set_ylabel('A', fontsize=18)
    # ax.set_ylim(0,0.1)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
    # plt.title("%s"%txt,fontsize=18)
    plt.tight_layout()
    # plt.savefig("fitrange_%s.pdf"%txt)
    return([*fit_gam])
ff_gam(9)
#%%














#%% form up the distribution parameter relationship
#   plot the distribution of the data at different y slice
#  test plot for explanation
# Y1 = np.arange(0,L0+1,2)
# fig, ax = plt.subplots()
# for i in np.arange(180, 211, 5):
#     j = (i-180)
#     plt.plot(Y1, RREF_R160[1,i]+j, color = "r", lw = (i-179)/8, label = "y = %s m" %(i*2))
# ax.set_ylim(0, 100)
# ax.set_xlim(0, L0)
# ax.set_xlabel('x')
# ax.set_ylabel('EF [\%]')
# plt.legend(loc='best', frameon=False,ncol = 1, bbox_to_anchor=(1.05, 1))
# # plt.savefig("illustration.pdf")

#%%
# make a plot as an example
# def gaussian(x, amplitude, mean, stddev):
#     return(amplitude * np.exp(-((x - mean) / (4) / stddev)**2))

# popt, _ = optimize.curve_fit(gaussian, Y1, RREF_R160[1,150])
# fig, ax = plt.subplots()
# plt.plot(Y1, RREF_R160[1,150],label = "data")
# plt.plot(Y1, gaussian(Y1, *popt),label = "fit")
# ax.set_xlabel('x direction at y = 300m', fontsize=18)
# ax.set_ylabel('EF [\%]', fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
# plt.tight_layout()
# plt.savefig("fit300m.pdf")
#%% get the distribution data 
# def norm_data(RREF_R160):
#     popt, _ = optimize.curve_fit(gaussian, Y1, RREF_R160[1,100])
#     for i in np.arange(105, 401, 5):
#         # 1 meter fitting
#         popt1, _ = optimize.curve_fit(gaussian, Y1, RREF_R160[1,i])
#         popt = np.vstack((popt,popt1))

#     return(popt)
# popt_R160 = norm_data(RREF_R160)

# #%% plot the statistic of std
# # popt_R160 (amplitude, mean, std)
# Y_range = np.arange(200, 801, 10)
# fig, ax = plt.subplots()
# plt.plot(Y_range, popt_R160[:,0])
# ax.set_xlabel('distance (y)', fontsize=18)
# ax.set_ylabel('EF [\%]', fontsize=18)
# # legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
# plt.tight_layout()
# plt.savefig("amp_y_dis.pdf")
#%%
# amp_fit = popt_R160[:,0]
# de_index = np.arange(0,8,1)
# amp_fit = np.delete(amp_fit,de_index)

# XX = np.arange(280, 801, 10)
# def expon(x, a, b, c):
#     return(a * b ** (c * x))

# fit_expo, pcov = optimize.curve_fit(expon, XX, amp_fit)
# fig, ax = plt.subplots()
# plt.plot(XX, amp_fit,label = "data")
# ax.set_xscale('log', basex=10)
# ax.set_yscale('log', basey=10)
# # plt.plot(XX, expon(XX, *fit_expo),label = "fit")
# ax.set_xlabel('distance (y)', fontsize=18)
# ax.set_ylabel('EF [\%]', fontsize=18)
# legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
# plt.tight_layout()
# plt.savefig("fit_amp.pdf")
#%%
# fig, ax = plt.subplots()
# plt.plot(Y_range, popt_R160[:,1])
# ax.set_xlabel('distance (y)', fontsize=18)
# ax.set_ylabel('mean', fontsize=18)
# # legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
# plt.tight_layout()
# plt.savefig("mean_y_dis.pdf")

#%%
# fig, ax = plt.subplots()
# plt.plot(Y_range, popt_R160[:,2])
# ax.set_xlabel('distance (y)', fontsize=18)
# ax.set_ylabel('std', fontsize=18)
# # legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 1)
# plt.tight_layout()
# # plt.savefig("std_y_dis.pdf")