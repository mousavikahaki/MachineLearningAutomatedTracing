#
# Created on 8/8/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import Functions as Func
import csv

dataPath = 'E:/Traces/'
savePath = 'E:/AutomatedTraceResults/DataForConnectingTraining/Data_For_Confident_Level/'
data_name = 'L1'
users = ['AS','JC','RG']
N_images = {'L1':6, 'OP':9}
pad = 4
trainData = []
labels = []
trainCount = 0
for i in range(N_images[data_name]):
    for j in range(len(users)):
        path = dataPath+data_name+'/'+str(i+1)+'_L6_'+users[j]+'.mat'
        print(path)
        IM, AM, r, R = Func.loadIMTrace(path)
        # AMB = Func.findBranchPoints(AM)
        padIM = np.pad(IM,pad,'constant')
        AM_A = AM.toarray()
        for k in range(len(r)):
            print(k)
            maxvalue = np.count_nonzero(AM_A[k, :])
            loc = r[k, :].round().astype(np.int64)+pad
            subIM = padIM[loc[0] - pad:loc[0] + pad, loc[1] - pad:loc[1] + pad, loc[2] - pad:loc[2] + pad]
            # Func.show3DImage(subIM)
            print(subIM.shape)
            # if is a branch
            if maxvalue > 2:
                label = 1
                # check if the size of the data is correct, add to training data
                if subIM.shape == (pad*2,pad*2,pad*2):
                    for s in range(40):
                        trainData.append(subIM.flatten())
                        labels.append(label)
            else:
                label = 0
                # check if the size of the data is correct, add to training data
                if subIM.shape == (pad*2,pad*2,pad*2):
                    trainData.append(subIM.flatten())
                    labels.append(label)
                # trainData.append(np.insert(subIM.flatten(),512,values=label))
            # pointIsBrach(AMB)


var = {}
var['input'] = trainData
var['target'] = labels
sio.savemat('E:/AutomatedTraceResults/DataForConnectingTraining/Data_For_Confident_Level/data.mat', var)




# Datapath = 'E:/Traces/L1/1_L6_AS.mat'
#
# IM, AM, r, R = Func.loadIMTrace(Datapath)

# Func.show3DImage(IM)

# AMB = Func.findBranchPoints(AM)
# eng = Func.PlotAM(AM,r,IM)
# eng = Func.PlotAM(AMB,r,IM)

print('Done')