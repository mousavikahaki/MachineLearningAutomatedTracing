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
from collections import Counter



x = datetime.datetime.today()
nowTimeDate = x.strftime("%b_%d_%Y_%H_%M")

ImagetoTest = 1
epoch  = 10
batch_size = 80
verbose=1 #verbose=1 will show you an animated progress bar
doSMOTE = False #do replicate data using SMOTE method

# PltNAme = 'ForIM'+str(ImagetoTest)+'_combined'+str(epoch)+'_batch_size='+str(batch_size)+'_SMOTE='+str(doSMOTE)+'_'+nowTimeDate

PltNAme = 'AT_Image_Shuffled_Ext_'+str(ImagetoTest)+'n'

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


# XIMs_train, XIMs_test, XFeature_train, XFeature_test, XScenarios_train, XScenarios_test,\
# yIMs_train, yIMs_test,yFeature_train, yFeature_test,yScenarios_train, yScenarios_test  \
#     = train_test_split(IMsTrain1,FeatureTrain,ScenariosTrain, LabelsTrain,LabelsTrain,LabelsTrain, test_size=0.2)


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


z_train = Counter(yIMs_train)
sns.countplot(yIMs_train)

# Balancing Data by SMOTE method
# np.count_nonzero(y_train == 1)

smt = SMOTE()
XIMs_train, yIMs_train1 = smt.fit_sample(XIMs_train, yIMs_train)
smt = SMOTE()
XFeature_train, yIMs_train1 = smt.fit_sample(XFeature_train, yIMs_train)
smt = SMOTE()
XScenarios_train, yIMs_train = smt.fit_sample(XScenarios_train, yIMs_train)

print(XIMs_train.shape)
print(XFeature_train.shape)
print(XScenarios_train.shape)
print(yIMs_train.shape)


z_train = Counter(yIMs_train)
sns.countplot(yIMs_train)
# to ignore image data
# XIMs_train = np.zeros(XIMs_train.shape)
# XIMs_test = np.zeros(XIMs_test.shape)

# XFeature_train = np.zeros(XFeature_train.shape)
# XFeature_test = np.zeros(XFeature_test.shape)
#
# XScenarios_train = np.zeros(XScenarios_train.shape)
# XScenarios_test = np.zeros(XScenarios_test.shape)




from keras import Sequential, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, concatenate
import numpy as np


import keras

input1 = keras.layers.Input(shape=(2197,))
x1 = keras.layers.Dense(512, activation='relu')(input1)
x2 = keras.layers.Dense(64, activation='relu')(x1)
x3 = keras.layers.Dense(16, activation='relu')(x2)
x4 = keras.layers.Dense(8, activation='relu')(x3)


input2 = keras.layers.Input(shape=(14,))
xx1 = keras.layers.Dense(14, activation='relu')(input2)


input3 = keras.layers.Input(shape=(144,))
input3_1 = keras.layers.Dense(64, activation='relu')(input3)
xxx1 = keras.layers.Dense(32, activation='relu')(input3_1)
xxx2 = keras.layers.Dense(16, activation='relu')(xxx1)
xxx3 = keras.layers.Dense(8, activation='relu')(xxx2)

# added = keras.layers.add([x4, xx1, xxx3])

added = keras.layers.concatenate([x4, xx1, xxx3])

out = keras.layers.Dense(4, activation='relu')(added)
out1 = keras.layers.Dense(1, activation='sigmoid')(out)
model = keras.models.Model(inputs=[input1, input2, input3], outputs=out1)


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

tensorboard = TensorBoard(log_dir="E:/AutomatedTracing/AutomatedTracing/Python/logs/"+PltNAme)
# tensorboard --logdir=E:\AutomatedTracing\AutomatedTracing\Python\logs


# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
history = model.fit([XIMs_train,XFeature_train,XScenarios_train],
                    yIMs_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=True,
                    callbacks=[tensorboard])





#######                Save Model
# model.save('E:/AutomatedTracing/Data/Models/ScenarioConnectome/Shuffled_IM_'+str(ImagetoTest)+'_out.h5')



# # load model
# model = load_model('E:/AutomatedTracing/Data/Models/ScenarioConnectome/Main_Signoid_epch80_batch_size=400_SMOTE=False_Oct_04_2019_09_35.h5')
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
# # summarize model.
# model.summary()

#######                Predict Values
y_pred = model.predict([XIMs_test,XFeature_test,XScenarios_test])
y_pred = y_pred[:,0]
y_pred.shape

ClusterStr = sio.loadmat('E:/AutomatedTracing/Data/Traces/L1_org/'+str(ImagetoTest)+'_L6_AS_withALLClusters1.mat')
ClusterStr1 = ClusterStr['ClustersStr']



y_pred_Final = np.zeros(shape=(ClusterStr1.shape[1],ClusterStr1[0,ClusterStr1.shape[1]-1]['cost_components'].shape[1]))#np.zeros(shape=(54,11715))
y_origina_Final = np.zeros(shape=(ClusterStr1.shape[1],ClusterStr1[0,ClusterStr1.shape[1]-1]['cost_components'].shape[1]))#np.zeros(shape=(54,11715))
counter = 0
for i in range(ClusterStr1.shape[1]):#54
    for s in range(ClusterStr1[0,i]['cost_components'].shape[1]):
        y_pred_Final[i,s] = y_pred[counter]
        y_origina_Final[i,s] = LabelsTest[counter]
        counter = counter + 1

# sio.savemat('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Shuffled_Matrix_Predict_IM_'+str(ImagetoTest)+'_out.mat',{"y_pred":y_pred_Final})
corrctnum = 0
incorrectnum = 0
# for i in range(y_pred_Final.shape[0]):
#     Pr = max(y_pred_Final[i, :])
#     Or = max(y_origina_Final[i,:])
#     if round(Pr) == round(Or):
#         corrctnum = corrctnum +1
#     else:
#         incorrectnum = incorrectnum + 1

for i in range(y_pred_Final.shape[0]):
    if np.argmax(y_pred_Final[i, :])==np.argmax(y_origina_Final[i, :]):
        corrctnum = corrctnum +1
    else:
        incorrectnum = incorrectnum + 1


print("Incorrect Scenario Connections: ",incorrectnum)
print("Total Scenarios: ",corrctnum+incorrectnum)
# print("Incorrect Scenario Connections: ",incorrectnum)

TP = corrctnum
FN = incorrectnum


# ################################## Plot Confusion Matrix - ROC ###############################################################
#
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score,roc_curve,auc
# import itertools
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         #print("Normalized confusion matrix")
#     else:
#         1#print('Confusion matrix, without normalization')
#
#     #print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
# class_names = [0,1]
# # TO DO: fix the rounding, got different result with the actual test
# cnf_matrix_tra = confusion_matrix(LabelsTest.round(), y_pred.round())
# plt.figure()
# plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Test Confusion matrix')
# plt.show()
#
# x_pred = model.predict([XIMs_train,XFeature_train,XScenarios_train])
# x_pred = x_pred[:,0]
# cnf_matrix_tra = confusion_matrix(yIMs_train.round(), x_pred.round())
# plt.figure()
# plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Train Confusion matrix')
# plt.show()
#
#
#
# fpr, tpr, thresholds = roc_curve(LabelsTest.round(), y_pred.round())
#
# roc_auc = auc(fpr,tpr)
#
# # Plot ROC
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.0])
# plt.ylim([-0.1,1.01])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
#
#
# ########################################################################################################################



# y_pred_pro = model.predict_proba([XIMs_test,XFeature_test,XScenarios_test])

# to compare predict and test
# y_pred_round = y_pred>0.04

# plt.figure()
# plt.plot(y_pred[:100].round())
# plt.show()
# plt.figure()
# plt.plot(LabelsTest[:100].round())
# plt.show()
# y_pred[:15].round()
# LabelsTest[:15].round()
# y_pred_pro[:5]
# y_test[:5]




###############                            Evaluate Model
# score = model.evaluate(X_test, y_test,verbose=1)
# print(score)

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

TP, FP, TN, FN = perf_measure(LabelsTest.round(), y_pred.round())
# plt.figure()
# plt.plot(LabelsTest)
# plt.show()
# plt.figure()
# plt.plot(y_pred)
# plt.show()
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("Sensitivity: ",TPR)
# Specificity or true negative rate
TNR = TN/(TN+FP)
print("Specificity: ",TNR)
# Precision or positive predictive value
PPV = TP/(TP+FP)
print("Positive predictive value: ",PPV)
# Negative predictive value
NPV = TN/(TN+FN)
print("Negative predictive value: ",NPV)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("Fall out or false positive rate: ",FPR)
# False negative rate
FNR = FN/(TP+FN)
print("False negative rate: ",FNR)
# False discovery rate
FDR = FP/(TP+FP)
print("False discovery rate: ",FDR)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall accuracy: ",ACC)

# print("Correct Scenario Connections: ",TP)
# print("Incorrect Scenario Connections: ",FP)


# ############### F1 score = 2TP / (TP+TN+FP+FN)
# f1_score = f1_score(LabelsTest.round(), y_pred.round())
# print("F1 Score: ",f1_score)
# ###############  Confusion matrix
# conf = confusion_matrix(LabelsTest.round(), y_pred.round())
# print("Confusion Matrix: ", conf)
# ###############  Precision = TP/(TP+FP)
# precision = precision_score(LabelsTest.round(), y_pred.round()) #  average=Nonefor precision from each class
# print("Precision: ",precision)
# ############### Recall TP / (TP+FN) # Sensitivity, hit rate, recall, or true positive rate
# recall = recall_score(LabelsTest.round(), y_pred.round())
# print("Recall: ",recall)
#
# ############### Cohen's kappa =
# cohen_kappa_score = cohen_kappa_score(LabelsTest.round(), y_pred.round())
# print("Cohen_Kappa Score: ",cohen_kappa_score)
# ############### Accuracy = (TPR + TNR) / Total
# accuracy= accuracy_score(LabelsTest.round(), y_pred.round())
# print("Accuracy: ",accuracy)
# ############### Balanced Accuracy = (TPR + TNR) / 2
# balanced_accuracy= balanced_accuracy_score(LabelsTest.round(), y_pred.round())
# print("Balanced Accuracy: ",balanced_accuracy)



print("Done!")

# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# weights = model.get_weights()
# plt.matshow(weights[1:2], cmap='viridis')
# plt.matshow(weights[3:4], cmap='viridis')
# weights[1]
##### Go Trhough the list
# for i in range(0, len(C1)):
#     print(C1.Dist[i])

## Run Tensorboard
# tensorboard --logdir=E:\AutomatedTracing\AutomatedTracing\Python\logs










#
# model1 = Sequential()
# # model1.add(Embedding(20,10, trainable=True))
# # model1.add(GlobalAveragePooling1D())
# # model1.add(Dense(1, activation='sigmoid'))
# model1.add(Dense(2197, activation='relu', input_shape=(2197,)))
# # model1.add(Dense(1024, activation='relu'))
# model1.add(Dense(512, activation='relu'))
# # model1.add(Dense(256, activation='relu'))
# # model1.add(Dense(128, activation='relu'))
# model1.add(Dense(64, activation='relu'))
# # model1.add(Dense(32, activation='relu'))
# model1.add(Dense(16, activation='relu'))
# model1.add(Dense(8, activation='relu'))
# # model1.add(Dense(4, activation='relu'))
# model1.add(Dense(4, activation='sigmoid'))
#
# # model1.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# #
# # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# #
# # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # history = model1.fit(IMsTrain1,
# #                     LabelsTrain,
# #                     batch_size=batch_size,
# #                     epochs=epoch,
# #                     # validation_split=0.2,
# #                     verbose=True,
# #                     callbacks=[tensorboard])
#
#
#
#
#
#
# model2 = Sequential()
# # model2.add(Embedding(20,10, trainable=True))
# # model2.add(GlobalAveragePooling1D())
# # model2.add(Dense(1, activation='sigmoid'))
# model2.add(Dense(14, activation='relu', input_shape=(14,)))
# model2.add(Dense(8, activation='relu'))
# model2.add(Dense(4, activation='relu'))
# # model2.add(Dense(1, activation='sigmoid'))
#
# # model2.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# #
# # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# #
# # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # history = model2.fit(FeatureTrain,
# #                     LabelsTrain,
# #                     batch_size=batch_size,
# #                     epochs=epoch,
# #                     # validation_split=0.2,
# #                     verbose=True,
# #                     callbacks=[tensorboard])
#
#
# model3 = Sequential()
# # model2.add(Embedding(20,10, trainable=True))
# # model2.add(GlobalAveragePooling1D())
# # model2.add(Dense(1, activation='sigmoid'))
# model3.add(Dense(66, activation='relu', input_shape=(66,)))
# model3.add(Dense(32, activation='relu'))
# model3.add(Dense(16, activation='relu'))
# model3.add(Dense(4, activation='sigmoid'))
#
# # model3.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# #
# # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# #
# # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # history = model3.fit(ScenariosTrain,
# #                     LabelsTrain,
# #                     batch_size=batch_size,
# #                     epochs=epoch,
# #                     # validation_split=0.2,
# #                     verbose=True,
# #                     callbacks=[tensorboard])
#
# model_concat = concatenate([model1.output,model2.output, model3.output], axis=-1)
# model_concat = Dense(1, activation='softmax')(model_concat)
# model = Model(inputs=[model1.input,model2.input, model3.input], outputs=model_concat)
#
# # model.compile(loss='binary_crossentropy', optimizer='adam')
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               # metrics=['accuracy']
#               metrics=['accuracy']
#               )
#
# tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
#
# # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# history = model.fit([IMsTrain1,FeatureTrain,ScenariosTrain],
#                     LabelsTrain,
#                     epochs=epoch,
#                     # validation_split=0.2,
#                     verbose=True,
#                     callbacks=[tensorboard])
#




#######                Save Model
# model.save('E:/AutomatedTracing/Data/Models/ScenarioConnectome/'+PltNAme+'.h5')



#
# from keras.layers import Concatenate, Input, Dense
# # First model - Image
# first_input = Input((2197, ))
# first_dense = Dense(128)(first_input)
#
# # Second model - Graph
# second_input = Input((66, ))
# second_dense = Dense(64)(second_input)
#
# # Third model - Features
# third_input = Input((14,))
# third_dense = Dense(32)(third_input)
#
# # Concatenate both
# merged = Concatenate()([first_dense, second_dense,third_dense])
# output_layer = Dense(1)(merged)
#
# model = Model(inputs=[first_input, second_input,third_input], outputs=output_layer)
# # model.compile(optimizer='sgd', loss='mse')
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               # metrics=['accuracy']
#               metrics=['accuracy']
#               )
# history = model.fit([IMsTrain1, ScenariosTrain, FeatureTrain], LabelsTrain, epochs=epoch, verbose=1)


# # Initialize the constructor
# model = Sequential()  # comes from import: from keras.models import Sequential
# # Add an input layer
# model.add(Dense(14, activation='relu', input_shape=(14,)))
# # Add Batch Normalization
# # model.add(BatchNormalization(axis=1))
# # Add one hidden layer
# model.add(Dense(8, activation='relu'))
# # Add Batch Normalization
# # model.add(BatchNormalization(axis=1))
# # Add one hidden layer
# model.add(Dense(4, activation='relu'))
# # Add Batch Normalization
# # model.add(BatchNormalization(axis=1))
# # Add an output layer
# model.add(Dense(1, activation='sigmoid'))
# # Model output shape
# model.output_shape
#
#
# # compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               # metrics=['accuracy']
#               metrics=['accuracy']
#               )
#
#
# # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# history = model.fit(FeatureTrain, LabelsTrain,
#                     epochs=epoch,
#                     # batch_size=len(X_train),
#                     batch_size=batch_size,
#                     verbose=verbose,
#                     shuffle=True,
#                     validation_split=0.2)


# Plot training & validation accuracy values
