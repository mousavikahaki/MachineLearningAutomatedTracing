#
# Created on 8/6/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
import scipy.io as sio
import matlab.engine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import sparse

eng = matlab.engine.start_matlab()

def Z_Projection(IM):
    IM_Max = np.zeros((len(IM),len(IM[0])))
    for i in range(len(IM)):
        for j in range(len(IM[0])):
            IM_Max[i,j] = np.amax(IM[i,j,:])
    return IM_Max

G = sio.loadmat('E:/Traces/L1/1_L6_AS.mat')
IM = G['IM']
AM_G = G['AM']
r_G = G['r']
R_G = G['R']

# Use evalc to load the .mat file so that it is kept in the MATLAB Engine workspace.
# This stops the API from attempting to convert the variables Python types on load.
eng.evalc("s = load('E:/Traces/L1/1_L6_AS.mat');")
# Now we can grab the data, as needed, from the MATLAB Engine workspace and pull it into Python.
# At this point the API will need to convert the data to Python types.
# AM_G1 = eng.eval("s.AM")

# With unsupported types, we can convert the variables to supported types while they are in
# the MATLAB Engine workspace, and then pull the supported representation into Python.
eng.evalc("PlotAM_1(s.AM, s.r)")
# myDateStrings = eng.eval("cellfun(@datestr, dt, 'UniformOutput', false)")
# print (myDateStrings)

IM_Max = Z_Projection(IM)

#img=mpimg.imread('your_image.png')
imgplot = plt.imshow(IM_Max)
plt.show()
eng.evalc("hold on")
eng.PlotAM_1(eng.eval("s.AM"), r_G)


# IM_Max = Z_Projection(IM)
#
# #img=mpimg.imread('your_image.png')
# imgplot = plt.imshow(IM_Max)
# plt.show()
#
#
#
#
# var = sio.loadmat('AMG.mat')
# AM = var['AMG']
# AM_tmp = AM
# AM_BP = np.zeros((len(AM),len(AM)))
# maxvalue = []
# BP = []
# for i in range(len(AM)):
#     maxvalue = np.count_nonzero(AM[i,:])
#     if maxvalue > 2:
#         BP.append(i)
#         AM_BP[i,:] = AM[i,:]
#         AM_BP[:, i] = AM[:, i]
#
#
#
# AM_BP = np.asarray(AM_BP)
# var['AM_BP'] = AM_BP
# sio.savemat('test.mat',var)



# # Removing branch points
# for i in range(len(AM)):
#     maxvalue = np.count_nonzero(AM[i,:])
#     if maxvalue > 2:
#         BP.append(i)
#         AM_tmp[i, :] = 0
#         AM_tmp[:, i] = 0
# AM_tmp = np.asarray(AM_tmp)
# var['AM_tmp'] = AM_tmp
# sio.savemat('test.mat',var)




