#
# Created on 8/20/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
import scipy.io as sio
from keras.models import load_model
import matlab.engine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import sparse
import AT_Classes as Classes
import scipy.ndimage as ndimage


model_path = 'E:\AutomatedTracing\Data\Models\Connectome\Mine_Connecting_July29\ShallowCluster1000epc\model'
model = load_model(model_path)
eng = matlab.engine.start_matlab()

users = ['AS','RG','JC']
mesh = 5
mesh_z = 3
b_thresh = 150 / 255

ppm = 1
dd = 15
c = 1

d_thresh = 15

netType = 1

# def getTestingData(self, isSampleTesting):
#     return DataProcessor.__matlabEngine.getTestingData(isSampleTesting, nargout=3)

print('here')
for imnum in range(1):
    for usernum in range(1):

        ImpathFiltered = 'E:/AutomatedTracing/Data/Traces/enhancedIM_NewGenerated/output' + str(imnum + 1) + '_1881000.mat'
        r_A = eng.Find_Seeds(ImpathFiltered, mesh, mesh_z, b_thresh)
        print('Features Size = '+str(r_A.size))

        ImpathOriginal = 'E:/AutomatedTracing/Data/Traces/L1/'+str(imnum+1)+'_L6_AS.mat'
        # AM_Astar, N, AM_C_path = eng.IM2TraceAstar(ImpathOriginal, r_A, c, dd)
        AMAll = eng.IM2TraceAstar(ImpathOriginal, r_A, c, dd)

        # AM_Astar = AMAll['AM_cost']
        # N = AMAll['AM_dist']
        # AM_C_path = AMAll['AM_C_path']
        # AMAll = []
        # AM_Astar, N, AM_C_path = getTestingData(AMAll)

        AMc = eng.ApplyAI(ImpathOriginal, r_A, d_thresh, netType, AMAll['AM_dist'],AMAll['AM_C_path'])

        AM_A = eng.MinimumSpanningTree(AMc)
        AM_A = eng.LabelTreesAM(AM_A)

        IM, AM_G, r_G, R_G = Classes.Trace.loadTrace('E:/AutomatedTracing/Data/Traces/L1/1_L6_AS.mat')
        AM_A = sparse.csr_matrix(AM_A)
        T1 = Classes.Trace(AM_A,np.array(r_A),IM)
        eng = T1.plt()





        # AMAll = []
        # AM_net = AM_netD['AM']
        # A = AM_net[29]
        # AM_netD = []

        # IM, AM_G, r_G, R_G = Classes.Trace.loadTrace('E:/AutomatedTracing/Data/Traces/L1/'+str(imnum+1)+'_L6_'+users[usernum]+'.mat')

        # tmp = sio.loadmat('E:/AutomatedTracing/Data/Traces/L1/'+str(imnum+1)+'_L6_AS.mat')
        # IMo = tmp['Original'].astype(np.double)
        # # tmp = sio.loadmat('E:/AutomatedTracing/Data/Traces/enhancedIM_NewGenerated/output'+str(imnum+1)+'_1881000.mat')
        # # IMf = tmp['IM'].astype(np.double)
        #
        # IMo *= 255 / IMo.max()
        # IMo = ndimage.gaussian_filter(IMo, 3)
        # # IMf *= 255 / IMf.max()
        # print(IMo.shape)



        # Alternatively load image in Find_Seeds.m
        # matlab.double(tuple(IMo))
        # data_list = IMo.tolist()
        # r_A = eng.Find_Seeds([data_list], mesh, mesh_z, b_thresh)

        # print(len(r_A))

print('done')







## Load Data
# IM, AM_G, r_G, R_G = Classes.Trace.loadTrace('E:/AutomatedTracing/Data/Traces/L1/1_L6_AS.mat')



# # Show Image
# IM1 = Classes.IM3D(IM)
# IM1.plt()
#
# # Show original Trace
# T1 = Classes.Trace(AM_G,r_G,R_G,IM)
# eng = T1.plt()
#
#
#
# # Get branches
# AM_BP = T1.GetBranch()
# T2 = Classes.Trace(AM_BP,r_G,R_G,IM)
# eng = T2.plt()
#
#
# # Remove Branches
# AM_No_Branch = T1.removeBranches()
# T3 = Classes.Trace(AM_No_Branch,r_G,R_G,IM)
# eng = T3.plt()







