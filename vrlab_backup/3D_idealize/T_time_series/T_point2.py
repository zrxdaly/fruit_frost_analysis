import numpy as np
import glob

dir1 = "/home/yi/transfer/basilisk/code/3D_idealize/T_time_series/Full/"
TS_file = sorted(glob.glob(dir1 + 'TS_*.npy'))

#TS_R060 = np.load(TS_file[0])
#TS_R080 = np.load(TS_file[1])
TS_R096 = np.load(TS_file[2])
TS_R120 = np.load(TS_file[3])
#TS_R160 = np.load(TS_file[4])
#TS_R192 = np.load(TS_file[5])
#TS_R240 = np.load(TS_file[6])
#TS_R320 = np.load(TS_file[7])
#TS_R480 = np.load(TS_file[8])

x_index = np.append(np.arange(100, 251, 15), [300, 350])

#TR060_Series = TS_R060[:,:,x_index,201]
#TR080_Series = TS_R080[:,:,x_index,201]
TR096_Series = TS_R096[:,:,x_index,201]
TR120_Series = TS_R120[:,:,x_index,201]
#TR160_Series = TS_R160[:,:,x_index,201]
#TR192_Series = TS_R192[:,:,x_index,201]
#TR240_Series = TS_R240[:,:,x_index,201]
#TR320_Series = TS_R320[:,:,x_index,201]
#TR480_Series = TS_R480[:,:,x_index,201]

#np.save("Full/TR060_Series.npy", TR060_Series)
#np.save("Full/TR080_Series.npy", TR080_Series)
np.save("Full/TR096_Series.npy", TR096_Series)
np.save("Full/TR120_Series.npy", TR120_Series)
#np.save("Full/TR160_Series.npy", TR160_Series)
#np.save("Full/TR192_Series.npy", TR192_Series)
#np.save("Full/TR240_Series.npy", TR240_Series)
#np.save("Full/TR320_Series.npy", TR320_Series)
#np.save("Full/TR480_Series.npy", TR480_Series)
