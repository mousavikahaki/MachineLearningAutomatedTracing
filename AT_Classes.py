#
# Created on 8/13/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
from scipy import sparse
import scipy.io as sio
import matlab.engine
import matplotlib.pyplot as plt

class IM3D:

    def __init__(self, IM, IM_Max = None):
        self.IM = IM
        self.IM_Max = IM_Max

    def Z_Projection(self):
        IM_Max = np.zeros((len(self.IM), len(self.IM[0])))
        for i in range(len(self.IM)):
            for j in range(len(self.IM[0])):
                IM_Max[i, j] = np.amax(self.IM[i, j, :])
        return IM_Max

    def plt(self):
        IM_Max = self.Z_Projection()
        plt.imshow(IM_Max)
        plt.show()
        return plt


class Trace:

    def __init__(self, AM, r, IM):
        self.AM = AM
        self.r = r
        # self.R = R
        self.IM = IM
        self.var = {}

    def plt(self):
        print(self.AM.shape)
        self.var['AM_BP'] = np.asarray(self.AM)
        self.var['r'] = self.r
        # self.var['R'] = self.R
        self.var['IM'] = self.IM
        sio.savemat('temp.mat', self.var)
        ## Plot Trace and Image
        eng = matlab.engine.start_matlab()
        eng.evalc("s = load('temp.mat');figure;imshow(max(s.IM,[],3));hold on;PlotAM_1(s.AM_BP{1}, s.r)")
        return eng

    def GetBranch(self):
        AM_BP = np.zeros((self.AM.shape))
        # maxvalue = []
        BP = []
        AM_G_A = self.AM.toarray()
        for i in range(self.AM.shape[1]):
            maxvalue = np.count_nonzero(AM_G_A[i, :])
            if maxvalue > 2:
                BP.append(i)
                AM_BP[i, :] = AM_G_A[i, :]
                AM_BP[:, i] = AM_G_A[:, i]

        return sparse.csr_matrix(AM_BP)

    def removeBranches(self):
        # BP = []
        AM_rem_br = self.AM.toarray()
        for i in range(len(AM_rem_br)):
            maxvalue = np.count_nonzero(AM_rem_br[i,:])
            if maxvalue > 2:
                # BP.append(i)
                AM_rem_br[i, :] = 0
                AM_rem_br[:, i] = 0
        AM_rem_br = np.asarray(AM_rem_br)
        return sparse.csr_matrix(AM_rem_br)

    @classmethod
    def loadTrace(self,path):
        G = sio.loadmat(path)
        IM = G['IM']
        AM = G['AM']
        r = G['r']
        R = G['R']
        return IM,AM,r,R



class cl_scenario:
    k = 1 # instant variable
    emptyElementValue = 0.5#np.inf
    def __init__(self, maxNumPoints, scenariosShape,scenario):
        self.maxNumPoints = maxNumPoints
        self.scenariosShape = scenariosShape
        self.scenario = scenario

    # Regular method # automatically take the instance (i.e. S1,S2) as the first input
    def getUpperArr(self):
        inputSize = int((self.maxNumPoints * (self.maxNumPoints - 1))/2)
        upperTriangle = self.scenario[np.triu_indices(self.scenariosShape, k=1)]
        # print(upperTriangle.shape)
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(upperTriangle)] = upperTriangle
        return scenario_arr

    def getWholeArr(self):
        inputSize = int(self.maxNumPoints * self.maxNumPoints)
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()
        return scenario_arr

    # # class method # To work with CLASS information # NOT automatically take the instance (i.e. S1,S2) as the first input
    # @classmethod    # <----- Decorator
    # def classmethod(cls, amount): # clc is class variable
    #     cls.raise_amt = amount
    #
    # @staticmethod
    # def staticmethod():
    #     return 'static method called'
