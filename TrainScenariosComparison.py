#
# Created on 2/4/2020
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

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

ImagetoTest = 6
epoch  = 10
batch_size = 80
verbose=1 #verbose=1 will show you an animated progress bar
doSMOTE = False #do replicate data using SMOTE method

# PltNAme = 'ForIM'+str(ImagetoTest)+'_combined'+str(epoch)+'_batch_size='+str(batch_size)+'_SMOTE='+str(doSMOTE)+'_'+nowTimeDate

PltNAme = 'AT_Image_Shuffled_Ext_'+str(ImagetoTest)

#### Read Data
# ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new.mat')
# ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_IM1_to_IM5.mat')
ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_withIMnum.mat')

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

Labels = ScenariosData['Labels']
# Labels[0,Labels.shape[1]-1]




maxNumPoints = 12
ScenariosTrain = []
IMsTrain = []
FeatureTrain = []
LabelsTrain = []
ScenariosTest = []
IMsTest = []
FeatureTest = []
LabelsTest = []

UseUpper = False
numScenarios = Scenarios.shape
counter = 0


for i in range(numScenarios[1]):

    scenario = Scenarios[0,i]
    IM = IMs[0,i]
    Feature = Features[0, i]
    Label = Labels[0, i]
    # print(scenarios.shape)
    # if scenario.any():
    # scenarios.shape[2]
    S = Classes.cl_scenario(maxNumPoints, scenario.shape[0],scenario)
    if UseUpper:
        scenario_arr = S.getUpperArr()
    else:
        scenario_arr = S.getWholeArr()

    if IMnums[0, i] != ImagetoTest:
        ScenariosTrain.append(scenario_arr)
        IMsTrain.append(IM)
        FeatureTrain.append(Feature)
        LabelsTrain.append(Label)
    else:
        ScenariosTest.append(scenario_arr)
        IMsTest.append(IM)
        FeatureTest.append(Feature)
        LabelsTest.append(Label)



ScenariosTrain = np.asarray(ScenariosTrain, dtype=np.float)
IMsTrain = np.asarray(IMsTrain, dtype=np.float)
FeatureTrain = np.asarray(FeatureTrain, dtype=np.float)
FeatureTrain = FeatureTrain[:,:,0]
LabelsTrain = np.asarray(LabelsTrain, dtype=np.float)
LabelsTrain = LabelsTrain[:,0]
LabelsTrain = LabelsTrain[:,0]
IMsTrain1 = np.reshape(IMsTrain, [IMsTrain.shape[0],np.product(IMsTrain[0,:,:,:].shape)])

ScenariosTest = np.asarray(ScenariosTest, dtype=np.float)
IMsTest = np.asarray(IMsTest, dtype=np.float)
FeatureTest = np.asarray(FeatureTest, dtype=np.float)
FeatureTest = FeatureTest[:,:,0]
LabelsTest = np.asarray(LabelsTest, dtype=np.float)
LabelsTest = LabelsTest[:,0]
LabelsTest = LabelsTest[:,0]
IMsTest1 = np.reshape(IMsTest, [IMsTest.shape[0],np.product(IMsTest[0,:,:,:].shape)])

# Sbhuffle Data
indices = np.arange(len(ScenariosTrain))
np.random.shuffle(indices)
ScenariosTrain = ScenariosTrain[indices]
IMsTrain = IMsTrain[indices]
FeatureTrain = FeatureTrain[indices]
LabelsTrain = LabelsTrain[indices]

# indices = np.arange(len(ScenariosTest))
# np.random.shuffle(indices)
# ScenariosTest = ScenariosTest[indices]
# IMsTest = IMsTest[indices]
# FeatureTest = FeatureTest[indices]
# LabelsTest = LabelsTest[indices]



XIMs_train = IMsTrain1
XFeature_train = FeatureTrain
XScenarios_train = ScenariosTrain
yIMs_train = LabelsTrain

XIMs_test = IMsTest1
XFeature_test = FeatureTest
XScenarios_test = ScenariosTest
yIMs_test = LabelsTest
yFeature_test = LabelsTest
yScenarios_test = LabelsTest


# Balancing Data by SMOTE method
# np.count_nonzero(y_train == 1)

# smt = SMOTE()
# XIMs_train, yIMs_train1 = smt.fit_sample(XIMs_train, yIMs_train)
# smt = SMOTE()
# XFeature_train, yIMs_train1 = smt.fit_sample(XFeature_train, yIMs_train)
# smt = SMOTE()
# XScenarios_train, yIMs_train = smt.fit_sample(XScenarios_train, yIMs_train)



X_train = np.concatenate((XIMs_train,XFeature_train,XScenarios_train), axis=1)
Y_train = yIMs_train
X_test = np.concatenate((XIMs_test,XFeature_test,XScenarios_test), axis=1)
Y_test = yScenarios_test



#-------------------------------------------------------------------- Data Splicing
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)


#------------------------------------------------------------Data Normalization or scaling data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

#------------------------------------------------------------Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
print('---------------------------Logistic Regression------------------------------')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train,Y_train)))
print('Accuracy of Logistic regression classifier on Test set: {:.2f}'.format(logreg.score(X_test,Y_test)))
predict = logreg.predict(X_test)
print(confusion_matrix(Y_test,predict))
print(classification_report(Y_test,predict))
cm = confusion_matrix(Y_test,predict)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print('TP:'+str(TP)+'  FP:'+str(FP)+'  FN:'+str(FN)+'  TN:'+str(TN))
#------------------------------------------------------------Decision Tree (Good obn Training but failed for Test)
from sklearn.tree import DecisionTreeClassifier
clf =  DecisionTreeClassifier().fit(X_train,Y_train)
print('---------------------------Decision Tree------------------------------')
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train,Y_train)))
print('Accuracy of Decision Tree classifier on Test set: {:.2f}'.format(clf.score(X_test,Y_test)))
predict = clf.predict(X_test)
print(confusion_matrix(Y_test,predict))
print(classification_report(Y_test,predict))
cm = confusion_matrix(Y_test,predict)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print('TP:'+str(TP)+'  FP:'+str(FP)+'  FN:'+str(FN)+'  TN:'+str(TN))
#------------------------------------------------------------KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
print('---------------------------KNN------------------------------')
print('Accuracy of KNN classifier on training set: {:.2f}'.format(knn.score(X_train,Y_train)))
print('Accuracy of KNN classifier on Test set: {:.2f}'.format(knn.score(X_test,Y_test)))
predict = knn.predict(X_test)
print(confusion_matrix(Y_test,predict))
print(classification_report(Y_test,predict))
cm = confusion_matrix(Y_test,predict)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print('TP:'+str(TP)+'  FP:'+str(FP)+'  FN:'+str(FN)+'  TN:'+str(TN))
#------------------------------------------------------------ Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
print('---------------------------Naive Bayes------------------------------')
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'.format(gnb.score(X_train,Y_train)))
print('Accuracy of Naive Bayes classifier on Test set: {:.2f}'.format(gnb.score(X_test,Y_test)))
predict = gnb.predict(X_test)
print(confusion_matrix(Y_test,predict))
print(classification_report(Y_test,predict))

cm = confusion_matrix(Y_test,predict)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print('TP:'+str(TP)+'  FP:'+str(FP)+'  FN:'+str(FN)+'  TN:'+str(TN))

#------------------------------------------------------------ SVM (Good for Large Dataset)
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,Y_train)
print('---------------------------SVM------------------------------')
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train,Y_train)))
print('Accuracy of SVM classifier on Test set: {:.2f}'.format(svm.score(X_test,Y_test)))
predict = svm.predict(X_test)
print(classification_report(Y_test,predict))
print(confusion_matrix(Y_test,predict))

cm = confusion_matrix(Y_test,predict)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print('TP:'+str(TP)+'  FP:'+str(FP)+'  FN:'+str(FN)+'  TN:'+str(TN))

# perf_measure(Y_test, predict)







