#%% get vertical slice from vrlab and load the data offline 
# plot the data in animation
# 3D data from ventilator model or other settings

import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation


#%%
# dir1 = "/net/labdata/yi/basilisk/Experiment/3D_idealize/PARA/WALL/WALL_100m_visual/resultslice/"
# dir1 = "/home/dai/Desktop/trans/3D_idealize/PARA/WALL/distance/"
dir1 = "/home/dai/Desktop/trans/3D_idealize/fan_model/"


## this part is for extracting the data from vrlab
# buo_file = sorted(glob.glob(Goal_dir[0] + '*'))
# buo_file = np.reshape(buo_file, (286, 30))

# U_file = sorted(glob.glob(Goal_dir[1] + '*'))
# U_file = np.reshape(U_file, (286, 30))

# W_file = sorted(glob.glob(Goal_dir[3] + '*'))
# W_file = np.reshape(W_file, (286, 30))
# # print(len(buo_file))
# # print(BUO_file[0,:])

# def extractdata(buo_file):
#     outj = np.empty(401)
#     outij = np.empty((31, 401))
#     for i in range(286):
#         for j in range(30):
#             xfield = np.loadtxt(buo_file[i,j], dtype='f',skiprows=2)[:,201]
#             outj = np.vstack((outj, xfield))
#         outij = np.dstack((outij, outj))
#     outij = np.delete(outij,[0],axis = 0)
#     outij = np.delete(outij,[0],axis = 1)
#     return(outj)

# b_tyx = extractdata(buo_file)
# u_tyx = extractdata(U_file)
# w_tyx = extractdata(W_file)

#%% load the data and define the dimension 
Goal_dir = sorted(glob.glob(dir1 + '*'))
Vlice_file = sorted(glob.glob(dir1 + "*tyx.npy"))

b_tyx = np.load(Vlice_file[0])
u_tyx = np.load(Vlice_file[1])
v_tyx = np.load(Vlice_file[2])
w_tyx = np.load(Vlice_file[3])

x = np.arange(0,801,2)
y = np.arange(1,b_tyx.shape[1]+1,1)

#%% quiver function -- nice visualization but didn't reveal anything
# def quiver2D(p_store, u_store, v_store, case_txt, x, y, nt, dt, fps):
#     X, Y = np.meshgrid(x, y) 
#     fig, ax = plt.subplots(figsize=(12, 2), dpi=300)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     con = ax.contourf(X, Y, p_store[0,:,:], cmap=cm.viridis)
#     fig.colorbar(con)
#     x_dim = 20
#     y_dim = 2
#     qax = ax.quiver(x[::x_dim], y[::y_dim], u_store[0,::y_dim,::x_dim], v_store[0,::y_dim,::x_dim])
#     fig.tight_layout()

#     def animate(i):
#         con = ax.contourf(X, Y, p_store[i,:,:], cmap=cm.viridis)
#         qax = ax.quiver(x[::x_dim], y[::y_dim], u_store[i,::y_dim,::x_dim], v_store[i,::y_dim,::x_dim])
#         plt.title(r'$%s$ t = %.5f sec' % (case_txt, i * dt))

#     anim = FuncAnimation(fig, animate, frames=np.linspace(0, len(p_store[0,...]), nt).astype(int), interval=1)
#     anim.save('%s.mp4'%case_txt, dpi=300, fps=10, extra_args=['-vcodec', 'libx264'])

# quiver2D(b_tyx, u_tyx, w_tyx, "wall full R", x, y, 285, 4, int(len(b_tyx[0,...])/100))


# %%
def Fun2D(u_store, case_txt, x, y, nt, dt, fps):
    X, Y = np.meshgrid(x, y) 
    fig = plt.figure(figsize=(12, 2), dpi=300)
    ax = plt.axes(xlabel='x', ylabel='y')
    levels = np.linspace(0.0, 1, 10)
    # levels = np.linspace(-0.3e-14, 0.3e-14, 7)
    con = plt.contourf(X, Y, u_store[0,:,:], levels = levels, cmap=cm.coolwarm)
    plt.xlim((0,400))
    plt.colorbar()
    plt.tight_layout()

    def animate(i):
    # global con
        # for c in con.collections:
        #     c.remove() 
        con = plt.contourf(X, Y, u_store[i,:,:], levels = levels, cmap=cm.coolwarm)
        plt.title(r'$%s$ t = %.5f sec' % (case_txt, i * dt))
        return(con)

    anim = FuncAnimation(fig, animate, frames=np.linspace(0, u_store.shape[0]-1, nt).astype(int), interval=1)
    anim.save('%s.mp4'%case_txt, dpi=300, fps=10, extra_args=['-vcodec', 'libx264'])
    plt.show()

#%%
Fun2D(u_tyx, "u_slice", x, y, 200, 8.1, 10)

#%%
Fun2D(w_tyx, "w_slice", x, y, 285, 4, 10)

#%%
Fun2D(b_tyx, "b_slice_fan_m", x, y, 200, 8.1, 10)

#%%
Fun2D((u_tyx**2 + w_tyx**2)**0.5, "TKE", x, y, 285, 4, 10)

# %%
X, Y = np.meshgrid(x, y) 
# levels = np.linspace(0.0, 0.6, 7)
fig = plt.figure(figsize=(10, 2), dpi=300)
plt.contourf(X, Y, np.mean(u_tyx, axis=0), cmap=cm.coolwarm)
plt.xlim((100,300))
plt.colorbar()

# %%
