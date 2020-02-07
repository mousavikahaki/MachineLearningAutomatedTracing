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
from numpy import *

eng = matlab.engine.start_matlab()

def Z_Projection(IM):
    IM_Max = np.zeros((len(IM),len(IM[0])))
    for i in range(len(IM)):
        for j in range(len(IM[0])):
            IM_Max[i,j] = np.amax(IM[i,j,:])
    return IM_Max

## Load Data
G = sio.loadmat('E:/AutomatedTracing/Data/Traces/L1/1_L6_AS.mat')
IM = G['IM']
AM_G = G['AM']
r_G = G['r']
R_G = G['R']
# imgplot = plt.imshow(IM_Max)
# plt.show()



AM_G = G['AM']
AM_tmp = AM_G
AM_BP = np.zeros((AM_G.shape))
maxvalue = []
BP = []

AM_G_A = AM_G.toarray()
for i in range (AM_G.shape[1]):
    maxvalue = np.count_nonzero(AM_G_A[i,:])
    if maxvalue > 2:
        BP.append(i)
        AM_BP[i,:] = AM_G_A[i,:]
        AM_BP[:, i] = AM_G_A[:, i]



var = {}
AM_BP = np.asarray(AM_BP)
var['AM_BP'] = sparse.csr_matrix(AM_BP)
var['r'] = r_G
var['R'] = R_G
var['IM'] = IM

AM_BP_sparse = sparse.csr_matrix(AM_BP)

AM_BPBKP = AM_BP

AM_BP.tolist()


# imgplot = plt.imshow(Z_Projection(IM))
# plt.show()

IM_Max = Z_Projection(IM)/255
eng.imshow(matlab.double(IM_Max.tolist()))
eng.hold


temp1 = matlab.double(AM_BPBKP.tolist())
temp2 = matlab.double(r_G.tolist())
pltt = eng.PlotAM_1(temp1,temp2)

plt.show()


# sio.savemat('temp.mat',var)

## Plot Trace and Image
# eng.evalc("s = load('temp.mat');figure;imshow(max(s.IM,[],3));hold on;PlotAM_1(s.AM_BP, s.r)")
# eng.evalc("s1 = load('E:/Traces/L1/1_L6_AS.mat');figure;imshow(max(s1.IM,[],3));hold on;PlotAM_1(s1.AM, s.r)")


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




