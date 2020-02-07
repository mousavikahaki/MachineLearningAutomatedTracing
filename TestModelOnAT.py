#
# Created on 9/17/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

# https://www.datacamp.com/community/tutorials/deep-learning-python


# for Merging Model: https://datascience.stackexchange.com/questions/26103/merging-two-different-models-in-keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from keras.constraints import maxnorm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Concatenate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import datetime
import scipy.io as sio
import AT_Classes as Classes
from imblearn.over_sampling import SMOTE



x = datetime.datetime.today()
nowTimeDate = x.strftime("%b_%d_%Y_%H_%M")

ImagetoTest = 1
epoch  = 10
batch_size = 80
verbose=1 #verbose=1 will show you an animated progress bar
doSMOTE = False #do replicate data using SMOTE method

# PltNAme = 'ForIM'+str(ImagetoTest)+'_combined'+str(epoch)+'_batch_size='+str(batch_size)+'_SMOTE='+str(doSMOTE)+'_'+nowTimeDate


#### Read Data
# ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new.mat')
# ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_IM1_to_IM5.mat')
ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\AT_All_scenarios_label_14Features_Image_Trace_new_withIMnum.mat')

IMnums = ScenariosData['IMnum']
# IMnum = IMnums[0,IMnums.shape[1]-1]
# IMnums.shape

Features = ScenariosData['Features']
# Feature = Features[0,Features.shape[1]-1]
# Feature.shape

IMs = ScenariosData['IMs']
# IMtmp = IMs[0,IMs.shape[1]-1]
# IMtmp.shape

Scenarios = ScenariosData['Scenarios']
# Scenarios.shape
# Scenarios[0,Scenarios.shape[1]-1]


maxNumPoints = 12

ScenariosTest = []
IMsTest = []
FeatureTest = []

UseUpper = False
numScenarios = Scenarios.shape
counter = 0


for i in range(numScenarios[1]):

    scenario = Scenarios[0,i]
    IM = IMs[0,i]
    Feature = Features[0, i]
    # print(scenarios.shape)
    # if scenario.any():
    # scenarios.shape[2]
    S = Classes.cl_scenario(maxNumPoints, scenario.shape[0],scenario)
    if UseUpper:
        scenario_arr = S.getUpperArr()
    else:
        scenario_arr = S.getWholeArr()

    if IMnums[0, i] == ImagetoTest:
        ScenariosTest.append(scenario_arr)
        IMsTest.append(IM)
        FeatureTest.append(Feature)





ScenariosTest = np.asarray(ScenariosTest, dtype=np.float)
IMsTest = np.asarray(IMsTest, dtype=np.float)
FeatureTest = np.asarray(FeatureTest, dtype=np.float)
FeatureTest = FeatureTest[:,:,0]
IMsTest1 = np.reshape(IMsTest, [IMsTest.shape[0],np.product(IMsTest[0,:,:,:].shape)])

XIMs_test = IMsTest1
XFeature_test = FeatureTest
XScenarios_test = ScenariosTest

# # load model

model = load_model('E:/AutomatedTracing/Data/Models/ScenarioConnectome/Shuffled_IM_'+str(ImagetoTest)+'_out.h5')
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
# # summarize model.
# model.summary()

#######                Predict Values
y_pred = model.predict([XIMs_test,XFeature_test,XScenarios_test])
y_pred = y_pred[:,0]
y_pred.shape

ClusterStr = sio.loadmat('E:/AutomatedTracing/Data/Traces/L1/'+str(ImagetoTest)+'_L6_AS_AT_withALLClusters1.mat')
ClusterStr1 = ClusterStr['ClustersStr']



y_pred_Final = np.zeros(shape=(ClusterStr1.shape[1],ClusterStr1[0,ClusterStr1.shape[1]-1]['cost_components'].shape[1]))#np.zeros(shape=(54,11715))

counter = 0
for i in range(ClusterStr1.shape[1]):#54
    for s in range(ClusterStr1[0,i]['cost_components'].shape[1]):
        y_pred_Final[i,s] = y_pred[counter]
        counter = counter + 1

sio.savemat('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/AT_Shuffled_Matrix_Predict_IM_'+str(ImagetoTest)+'_out.mat',{"y_pred":y_pred_Final})

