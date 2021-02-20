#%%
import numpy as np
import glob as glob
from Goal_func import Slice3_AV, Slice3_PEF, plotRC
import matplotlib.pyplot as plt

#%% Wall plot
Wall_dir = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/"
wall_EF = sorted(glob.glob(Wall_dir + '*R_c*.npy'))

FR_cEF_R160 = np.load(wall_EF[0])
RR_cEF_R160 = np.load(wall_EF[2])

#%% projection plot
Slice3_AV(FR_cEF_R160, 'WF_R160')
Slice3_AV(RR_cEF_R160, 'WR_R160')

#%%
Slice3_PEF(FR_cEF_R160, 'WF_R160')
Slice3_PEF(RR_cEF_R160, 'WR_R160')
#%%
# def EF2Vo(RREF_R060, LIM):
#     # get a number A * A * 5: this way we get the horizontal coverage area
#     size = L0/2+1
#     return((np.shape(RREF_R060[RREF_R060>LIM])[0]/(10*size*size))**(0.5)*L0)


# def LIM2L2(LIM):
#     VoL_RR = [EF2Vo(EF_R_R160,LIM), EF2Vo(EF_R_R240,LIM)]
#     VoL_FR = [EF2Vo(EF_F_R160,LIM), EF2Vo(EF_F_R240,LIM)]
#     return(VoL_RR, VoL_FR)

# # VoL_RR_1, VoL_FR_1, VoL_HR_1 = LIM2L(5)
# #%% plot warming area for different limits
# def plotRC22(num, txt):
#     ROTA_P = [160, 240]
#     ii = 0
#     fig = plt.figure(figsize=(7.4, 4.8))
#     ax = fig.add_subplot(1,1,1)
#     arry = np.append(np.arange(1,10,2), np.arange(11,61,10))
#     for i in arry:
#         ii = ii + 1
#         plt.plot(ROTA_P, LIM2L2(i)[num], "-*", label=str(i)+'%', linewidth = ii/2) 
#         # np.save("PARA/ROTA/%s_%s.npy" %(txt, i),LIM2L(i)[num])  # uncomment this one to save the volume data
#     ax.set_xlabel('Rotation period [s]',fontsize=18)
#     # ax.set_ylabel("Coverage Volume [**2*5m3]",fontsize=18)
#     ax.set_ylim([0,650])
#     ax.set_xlim([60,480])
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 20)
#     # ,fontsize = 20
#     plt.title("%s operation"%txt,fontsize=18)
#     plt.tight_layout()
#     plt.savefig("vr_rerun/%sRC.pdf"%txt)
# #%%
# plotRC22(0,"WF_R160")
# plotRC22(1,"WR_R160")

# %%
