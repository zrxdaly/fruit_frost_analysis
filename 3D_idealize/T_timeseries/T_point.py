#%%%
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
#%% get the data from vrlab
# dir1 = "/home/yi/transfer/basilisk/code/3D_idealize/T_time_series/Full/"
# TS_file = sorted(glob.glob(dir1 + 'TS_*.npy'))

# TS_R060 = np.load(TS_file[0])
# TS_R080 = np.load(TS_file[1])
# TS_R096 = np.load(TS_file[2])
# TS_R120 = np.load(TS_file[3])
# TS_R160 = np.load(TS_file[4])
# TS_R192 = np.load(TS_file[5])
# TS_R240 = np.load(TS_file[6])
# TS_R320 = np.load(TS_file[7])
# TS_R480 = np.load(TS_file[8])

# x_index = np.append(np.arange(200, 501, 30), [600, 700])

# TR060_Series = TS_R060[:,:,x_index,201]
# TR080_Series = TS_R080[:,:,x_index,201]
# TR096_Series = TS_R096[:,:,x_index,201]
# TR120_Series = TS_R120[:,:,x_index,201]
# TR160_Series = TS_R160[:,:,x_index,201]
# TR192_Series = TS_R192[:,:,x_index,201]
# TR240_Series = TS_R240[:,:,x_index,201]
# TR320_Series = TS_R320[:,:,x_index,201]
# TR480_Series = TS_R480[:,:,x_index,201]

# np.save("Full/TR060_Series.npy", TR060_Series)
# np.save("Full/TR080_Series.npy", TR080_Series)
# np.save("Full/TR096_Series.npy", TR096_Series)
# np.save("Full/TR120_Series.npy", TR120_Series)
# np.save("Full/TR160_Series.npy", TR160_Series)
# np.save("Full/TR192_Series.npy", TR192_Series)
# np.save("Full/TR240_Series.npy", TR240_Series)
# np.save("Full/TR320_Series.npy", TR320_Series)
# np.save("Full/TR480_Series.npy", TR480_Series)

#%% get the data and analysis
dir = "/home/dai/Desktop/trans/3D_idealize/T_timeseries/Full/"
TS_file = sorted(glob.glob(dir + 'T*.npy'))

TR060_Series = np.load(TS_file[0])
TR080_Series = np.load(TS_file[1])
TR096_Series = np.load(TS_file[2])
TR120_Series = np.load(TS_file[3])
TR160_Series = np.load(TS_file[4])
TR192_Series = np.load(TS_file[5])
TR240_Series = np.load(TS_file[6])
TR320_Series = np.load(TS_file[7])
TR480_Series = np.load(TS_file[8])

# # %%
# def PARA(Rotation, distance, height):
#     T = np.arange(0, 1020, 1)
#     RO = [60, 80, 96, 120, 160, 192, 240, 320, 480]
#     index_RO = RO.index(Rotation)
#     TR060_Series = np.load(TS_file[index_RO])
#     ## TR_series is three dimensional data [10, 1020, 13] [height, time, point in space]
#     HE = np.arange(0.5, 5.5, 0.5)
#     index_HE = np.where(HE == height)[0][0]

#     X_distance = np.append(np.arange(200, 501, 30), [600, 700])
#     index_X = np.where(X_distance == distance)[0][0]

#     fig, ax = plt.subplots()
#     plt.plot(T, TR060_Series[index_HE, :, index_X])
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Temperature [celsius]')
#     plt.show()
     
# #%%
# _ = widgets.interact(
#     PARA, 
#     Rotation=[60, 80, 96, 120, 160, 192, 240, 320, 480], 
#     distance=np.append(np.arange(200, 501, 30), [600, 700]),
#     height=np.arange(0.5, 5.5, 0.5)
# )

#%%
T = np.arange(0, 1020, 1)

def Tplot(TR240_Series, X_distance, txt):
    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax = fig1.add_subplot(1,1,1)
    h1 = plt.plot(T, TR240_Series[0, :,X_distance], "-", label="0.5m",linewidth = lw,color = CLR[8])  
    h2 = plt.plot(T, TR240_Series[2, :,X_distance], "-", label="1.5m",linewidth = lw,color = CLR[1]) 
    h3 = plt.plot(T, TR240_Series[4, :,X_distance], "-", label="2.5m",linewidth = lw,color = CLR[0]) 
    h4 = plt.plot(T, TR240_Series[6, :,X_distance], "-", label="3.5m",linewidth = lw,color = CLR[3])  
    h5 = plt.plot(T, TR240_Series[8, :,X_distance], "-", label="4.5m",linewidth = lw,color = CLR[4])   
    ax.set_ylim(0, 4.5)
    ax.set_xlabel('Time [s]',fontsize=18)
    ax.set_ylabel("temperature [celsius]",fontsize=18)
    plt.title("%s "%txt,fontsize=18)
    legend = plt.legend(loc='best', frameon=False,fontsize = 18,ncol = 2)
    plt.tight_layout()
    plt.savefig("TS%s.pdf"%txt)


# %% different rotation period 300 -350m
Tplot(TR060_Series, 5, " R060 Y350m")
Tplot(TR080_Series, 5, " R080 Y350m")
Tplot(TR096_Series, 5, " R096 Y350m")
Tplot(TR120_Series, 5, " R120 Y350m")
Tplot(TR160_Series, 5, " R160 Y350m")
Tplot(TR192_Series, 5, " R192 Y350m")
Tplot(TR240_Series, 5, " R240 Y350m")
Tplot(TR320_Series, 5, " R320 Y350m")
Tplot(TR480_Series, 5, " R480 Y350m")

#%%

Tplot(TR240_Series, 1, " R240 Y230m")
Tplot(TR240_Series, 4, " R240 Y320m")
Tplot(TR240_Series, 5, " R240 Y350m")
Tplot(TR240_Series, 6, " R240 Y380m")
Tplot(TR240_Series, 8, " R240 Y440m")
Tplot(TR240_Series, 11, " R240 Y600m")
# %%
