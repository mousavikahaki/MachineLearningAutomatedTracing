#
# Created on 12/10/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import datetime
import scipy.io as sio
import h5py
from keras.models import Sequential, load_model

import AT_Classes as Classes



#### Read Data
# ScenariosData = sio.loadmat('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/scenariosData6_L6_AS.mat')
with h5py.File('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/scenariosData6_L6_AS.mat', 'r') as ScenariosData:
    print(list(ScenariosData.keys()))
    ClusterIM_All = np.array(ScenariosData['ClusterIM_All'])
    scenario_ext_All = np.array(ScenariosData['scenario_ext_All'])
    features_All = np.array(ScenariosData['features_All'])

ClusterIM_All.shape
scenario_ext_All.shape
features_All.shape

# modelPath = 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined50_batch_size=80_SMOTE=False_Nov_06_2019_09_38.h5'
# modelPath = 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/AllforFinal_Combined30_batch_size=80_SMOTE=False_Dec_11_2019_10_29.h5'#IM1-5 all data final

# modelPath = 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined30_batch_size=80_SMOTE=False_Dec_16_2019_10_58.h5'#IM1-5 all data final

modelPath = 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/IM1to5_Combined30_batch_size=80_SMOTE=False_Dec_17_2019_10_52.h5'#IM1-5 all data final


model = load_model(modelPath)
y_pred = np.zeros(shape=(54,11715))
for s in range(11715):#11715
    for i in range(54):#54
        IM = np.reshape(ClusterIM_All[:,s,i], (-1, 2197))
        scenario = np.reshape(scenario_ext_All[:, s, i], (-1, 144))
        features = np.reshape(features_All[:, s, i], (-1, 14))
        input=[IM, features, scenario]
        print(model.predict(input))
        y_pred[i,s] = np.array(model.predict(input))

sio.savemat('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/AllforFinal_Prediction6_L6_AS_NEW_1-5.mat',{"y_pred":y_pred})





#
#
# y_pred = model.predict([XIMs_test,XFeature_test,XScenarios_test])

# Features = ScenariosData['Features']
# Feature = Features[0,104980]
# Feature.shape



