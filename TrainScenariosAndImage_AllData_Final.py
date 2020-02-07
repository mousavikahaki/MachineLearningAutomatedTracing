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

epoch  = 30
batch_size = 80
verbose=1 #verbose=1 will show you an animated progress bar
doSMOTE = False #do replicate data using SMOTE method

PltNAme = 'Combined'+str(epoch)+'_batch_size='+str(batch_size)+'_SMOTE='+str(doSMOTE)+'_'+nowTimeDate

#### Read Data
ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_IM1_to_IM5.mat')

Features = ScenariosData['Features']
Feature = Features[0,Features.shape[1]]
Feature.shape

IMs = ScenariosData['IMs']
IMtmp = IMs[0,IMs.shape[1]]
IMtmp.shape
Scenarios = ScenariosData['Scenarios']
Scenarios.shape
Scenarios[0,Scenarios.shape[1]]
Labels = ScenariosData['Labels']
Labels[0,Labels.shape[1]]

maxNumPoints = 12
ScenariosTrain = []
IMsTrain = []
FeatureTrain = []
LabelsTrain = []

UseUpper = False
numScenarios = Scenarios.shape
counter = 0
# numBranches[1]
for i in range(numScenarios[1]):
# for i in range(10):
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

    ScenariosTrain.append(scenario_arr)
    IMsTrain.append(IM)
    FeatureTrain.append(Feature)
    LabelsTrain.append(Label)

# Data splicing
ScenariosTrain = np.asarray(ScenariosTrain, dtype=np.float)
print(ScenariosTrain.shape)
IMsTrain = np.asarray(IMsTrain, dtype=np.float)
print(IMsTrain.shape)
FeatureTrain = np.asarray(FeatureTrain, dtype=np.float)
FeatureTrain = FeatureTrain[:,:,0]
print(FeatureTrain.shape)
LabelsTrain = np.asarray(LabelsTrain, dtype=np.float)
LabelsTrain = LabelsTrain[:,0]
LabelsTrain = LabelsTrain[:,0]
print(LabelsTrain.shape)

# branch1 = Sequential()
# branch1.add(Dense(ScenariosTrain.shape[1], input_shape=(ScenariosTrain.shape[1],), init='normal', activation='relu'))
# branch1.add(BatchNormalization())
#
# branch2 = Sequential()
# branch2.add(Dense(FeatureTrain.shape[1], input_shape=(FeatureTrain.shape[1],), init='normal', activation='relu'))
# branch2.add(BatchNormalization())
# branch2.add(Dense(FeatureTrain.shape[1], init='normal', activation='relu', W_constraint=maxnorm(5)))
# branch2.add(BatchNormalization())
# branch2.add(Dense(FeatureTrain.shape[1], init='normal', activation='relu', W_constraint=maxnorm(5)))
# branch2.add(BatchNormalization())
#
# model = Sequential()
# model.add(Concatenate([branch1, branch2]))
# model.add(Dense(1, init='normal', activation='sigmoid'))
# sgd = SGD(lr=0.1, momentum=0.9, decay=0, nesterov=False)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# # seed(2017)
# model.fit([ScenariosTrain, FeatureTrain], LabelsTrain, batch_size=2000, epochs=100, verbose=1)
ScenariosTrain.shape
FeatureTrain.shape
LabelsTrain.shape
IMsTrain.shape
IMsTrain1 = np.reshape(IMsTrain, [IMsTrain.shape[0],np.product(IMsTrain[0,:,:,:].shape)])
IMsTrain1.shape
from keras.models import Model

# Split the data up in train and test sets
# XIMs_train, XIMs_test, yIMs_train, yIMs_test = train_test_split(IMsTrain1, LabelsTrain, test_size=0.20)
# XFeature_train, XFeature_test, yFeature_train, yFeature_test = train_test_split(FeatureTrain, LabelsTrain, test_size=0.20)
# XScenarios_train, XScenarios_test, yScenarios_train, yScenarios_test = train_test_split(ScenariosTrain, LabelsTrain, test_size=0.20)

XIMs_train, XIMs_test, XFeature_train, XFeature_test, XScenarios_train, XScenarios_test, yIMs_train, yIMs_test,yFeature_train, yFeature_test,yScenarios_train, yScenarios_test  = train_test_split(IMsTrain1,FeatureTrain,ScenariosTrain, LabelsTrain,LabelsTrain,LabelsTrain, test_size=0.20)


from keras import Sequential, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, concatenate
import numpy as np


import keras
from keras.layers import Conv3D, MaxPooling2D

input1 = keras.layers.Input(shape=(2197,))
x1 = keras.layers.Dense(512, activation='relu')(input1)
x2 = keras.layers.Dense(64, activation='relu')(x1)
x3 = keras.layers.Dense(16, activation='relu')(x2)
x4 = keras.layers.Dense(8, activation='relu')(x3)


input2 = keras.layers.Input(shape=(14,))
xx1 = keras.layers.Dense(8, activation='relu')(input2)

input3 = keras.layers.Input(shape=(144,))
input3_1 = keras.layers.Dense(64, activation='relu')(input3)
xxx1 = keras.layers.Dense(32, activation='relu')(input3_1)
xxx2 = keras.layers.Dense(16, activation='relu')(xxx1)
xxx3 = keras.layers.Dense(8, activation='relu')(xxx2)

added = keras.layers.add([x4, xx1, xxx3])

out = keras.layers.Dense(4, activation='relu')(added)
out1 = keras.layers.Dense(1, activation='sigmoid')(out)
model = keras.models.Model(inputs=[input1, input2, input3], outputs=out1)


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

tensorboard = TensorBoard(log_dir="logs/"+PltNAme)

# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# history = model.fit([XIMs_train,XFeature_train,XScenarios_train],#use separate train and test
#                    yIMs_train,\
history = model.fit([IMsTrain1,FeatureTrain,ScenariosTrain],#use all data
                    LabelsTrain,
                    epochs=epoch,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=True,
                    callbacks=[tensorboard])

#######                Save Model
model.save('E:/AutomatedTracing/Data/Models/ScenarioConnectome/AllforFinal_'+PltNAme+'.h5')

# # 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined50_batch_size=80_SMOTE=False_Nov_12_2019_10_17.h5'
# # Sensitivity:  0.9322033898305084
# # Specificity:  0.9999044798930175
# # Positive predictive value:  0.9649122807017544
# # Negative predictive value:  0.9998089780324737
# # Fall out or false positive rate:  9.552010698251981e-05
# # False negative rate:  0.06779661016949153
# # False discovery rate:  0.03508771929824561
# # Overall accuracy:  0.9997142448921275
# # F1 Score:  0.9482758620689654
# # Confusion Matrix:  [[20936     2]
# #  [    4    55]]
# # Precision:  0.9649122807017544
# # Recall:  0.9322033898305084
# # Cohen_Kappa Score:  0.9481326314593364
# # Accuracy:  0.9997142448921275
# # Balanced Accuracy:  0.9660539348617629
# # Correct Scenario Connections:  55
# # Incorrect Scenario Connections:  4
#
# # 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined50_batch_size=80_SMOTE=False_Nov_04_2019_12_55.h5'
# # Precision:  0.9253731343283582
# # Recall:  0.9393939393939394
# # Cohen_Kappa Score:  0.9321158420077614
# # Accuracy:  0.9995713673381912
# # Balanced Accuracy:  0.9695775296319944
# # Correct Scenario Connections:  62
# # Incorrect Scenario Connections:  4
#
# # 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined30_batch_size=100_SMOTE=False_Nov_04_2019_12_42.h5'
# # Precision:  1.0
# # Recall:  0.9066666666666666
# # Cohen_Kappa Score:  0.9508820931629133
# # Accuracy:  0.9996666190408153
# # Balanced Accuracy:  0.9533333333333334
# # Correct Scenario Connections:  68
# # Incorrect Scenario Connections:  7
#
#
# # Precision:  0.9473684210526315
# # Recall:  0.9
# # Cohen_Kappa Score:  0.9228621496614634
# # Accuracy:  0.9995713673381912
# # Balanced Accuracy:  0.9499283564980656
# # Correct Scenario Connections:  54
# # Incorrect Scenario Connections:  6
# # 'E:/AutomatedTracing/Data/Models/ScenarioConnectome/Combined100_batch_size=400_SMOTE=False_Nov_04_2019_11_14.h5'
#
# # # load model
# # model = load_model('E:/AutomatedTracing/Data/Models/ScenarioConnectome/Main_Signoid_epch80_batch_size=400_SMOTE=False_Oct_04_2019_09_35.h5')
# # from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
# # # summarize model.
# # model.summary()
#
# #######                Predict Values
# y_pred = model.predict([XIMs_test,XFeature_test,XScenarios_test])
# # y_pred_pro = model.predict_proba([XIMs_test,XFeature_test,XScenarios_test])
#
# # # to compare predict and test
# # y_pred_round = y_pred>0.04
# #
# # plt.figure()
# # plt.plot(y_pred[:10])
# # plt.show()
# # plt.figure()
# # plt.plot(y_pred_round[:10])
# # plt.show()
# # y_pred[:5]
# # y_pred_pro[:5]
# # y_test[:5]
# ###############                            Evaluate Model
# # score = model.evaluate(X_test, y_test,verbose=1)
# # print(score)
#
# def perf_measure(y_actual, y_pred):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#
#     for i in range(len(y_pred)):
#         if y_actual[i]==y_pred[i]==1:
#            TP += 1
#         if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
#            FP += 1
#         if y_actual[i]==y_pred[i]==0:
#            TN += 1
#         if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
#            FN += 1
#
#     return(TP, FP, TN, FN)
#
# TP, FP, TN, FN = perf_measure(yScenarios_test.round(), y_pred.round())
# plt.figure()
# plt.plot(yScenarios_test)
# plt.plot(y_pred)
# plt.show()
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# print("Sensitivity: ",TPR)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# print("Specificity: ",TNR)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# print("Positive predictive value: ",PPV)
# # Negative predictive value
# NPV = TN/(TN+FN)
# print("Negative predictive value: ",NPV)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# print("Fall out or false positive rate: ",FPR)
# # False negative rate
# FNR = FN/(TP+FN)
# print("False negative rate: ",FNR)
# # False discovery rate
# FDR = FP/(TP+FP)
# print("False discovery rate: ",FDR)
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# print("Overall accuracy: ",ACC)
#
#
#
#
# ############### F1 score = 2TP / (TP+TN+FP+FN)
# f1_score = f1_score(yIMs_test.round(), y_pred.round())
# print("F1 Score: ",f1_score)
# ###############  Confusion matrix
# conf = confusion_matrix(yIMs_test.round(), y_pred.round())
# print("Confusion Matrix: ", conf)
# ###############  Precision = TP/(TP+FP)
# precision = precision_score(yIMs_test.round(), y_pred.round()) #  average=Nonefor precision from each class
# print("Precision: ",precision)
# ############### Recall TP / (TP+FN) # Sensitivity, hit rate, recall, or true positive rate
# recall = recall_score(yIMs_test.round(), y_pred.round())
# print("Recall: ",recall)
#
# ############### Cohen's kappa =
# cohen_kappa_score = cohen_kappa_score(yIMs_test.round(), y_pred.round())
# print("Cohen_Kappa Score: ",cohen_kappa_score)
# ############### Accuracy = (TPR + TNR) / Total
# accuracy= accuracy_score(yIMs_test.round(), y_pred.round())
# print("Accuracy: ",accuracy)
# ############### Balanced Accuracy = (TPR + TNR) / 2
# balanced_accuracy= balanced_accuracy_score(yIMs_test.round(), y_pred.round())
# print("Balanced Accuracy: ",balanced_accuracy)
#
# print("Correct Scenario Connections: ",TP)
# print("Incorrect Scenario Connections: ",FN)
#
# print("Done!")
#
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
#
# # weights = model.get_weights()
# # plt.matshow(weights[1:2], cmap='viridis')
# # plt.matshow(weights[3:4], cmap='viridis')
# # weights[1]
# ##### Go Trhough the list
# # for i in range(0, len(C1)):
# #     print(C1.Dist[i])
#
# ## Run Tensorboard
# # tensorboard --logdir=E:\AutomatedTracing\AutomatedTracing\Python\logs
#
#
#
#
#
#
#
#
#
#
# #
# # model1 = Sequential()
# # # model1.add(Embedding(20,10, trainable=True))
# # # model1.add(GlobalAveragePooling1D())
# # # model1.add(Dense(1, activation='sigmoid'))
# # model1.add(Dense(2197, activation='relu', input_shape=(2197,)))
# # # model1.add(Dense(1024, activation='relu'))
# # model1.add(Dense(512, activation='relu'))
# # # model1.add(Dense(256, activation='relu'))
# # # model1.add(Dense(128, activation='relu'))
# # model1.add(Dense(64, activation='relu'))
# # # model1.add(Dense(32, activation='relu'))
# # model1.add(Dense(16, activation='relu'))
# # model1.add(Dense(8, activation='relu'))
# # # model1.add(Dense(4, activation='relu'))
# # model1.add(Dense(4, activation='sigmoid'))
# #
# # # model1.compile(loss='binary_crossentropy',
# # #               optimizer='adam',
# # #               # metrics=['accuracy']
# # #               metrics=['accuracy']
# # #               )
# # #
# # # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# # #
# # # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # # history = model1.fit(IMsTrain1,
# # #                     LabelsTrain,
# # #                     batch_size=batch_size,
# # #                     epochs=epoch,
# # #                     # validation_split=0.2,
# # #                     verbose=True,
# # #                     callbacks=[tensorboard])
# #
# #
# #
# #
# #
# #
# # model2 = Sequential()
# # # model2.add(Embedding(20,10, trainable=True))
# # # model2.add(GlobalAveragePooling1D())
# # # model2.add(Dense(1, activation='sigmoid'))
# # model2.add(Dense(14, activation='relu', input_shape=(14,)))
# # model2.add(Dense(8, activation='relu'))
# # model2.add(Dense(4, activation='relu'))
# # # model2.add(Dense(1, activation='sigmoid'))
# #
# # # model2.compile(loss='binary_crossentropy',
# # #               optimizer='adam',
# # #               # metrics=['accuracy']
# # #               metrics=['accuracy']
# # #               )
# # #
# # # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# # #
# # # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # # history = model2.fit(FeatureTrain,
# # #                     LabelsTrain,
# # #                     batch_size=batch_size,
# # #                     epochs=epoch,
# # #                     # validation_split=0.2,
# # #                     verbose=True,
# # #                     callbacks=[tensorboard])
# #
# #
# # model3 = Sequential()
# # # model2.add(Embedding(20,10, trainable=True))
# # # model2.add(GlobalAveragePooling1D())
# # # model2.add(Dense(1, activation='sigmoid'))
# # model3.add(Dense(66, activation='relu', input_shape=(66,)))
# # model3.add(Dense(32, activation='relu'))
# # model3.add(Dense(16, activation='relu'))
# # model3.add(Dense(4, activation='sigmoid'))
# #
# # # model3.compile(loss='binary_crossentropy',
# # #               optimizer='adam',
# # #               # metrics=['accuracy']
# # #               metrics=['accuracy']
# # #               )
# # #
# # # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# # #
# # # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # # history = model3.fit(ScenariosTrain,
# # #                     LabelsTrain,
# # #                     batch_size=batch_size,
# # #                     epochs=epoch,
# # #                     # validation_split=0.2,
# # #                     verbose=True,
# # #                     callbacks=[tensorboard])
# #
# # model_concat = concatenate([model1.output,model2.output, model3.output], axis=-1)
# # model_concat = Dense(1, activation='softmax')(model_concat)
# # model = Model(inputs=[model1.input,model2.input, model3.input], outputs=model_concat)
# #
# # # model.compile(loss='binary_crossentropy', optimizer='adam')
# #
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# #
# # tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# #
# # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # history = model.fit([IMsTrain1,FeatureTrain,ScenariosTrain],
# #                     LabelsTrain,
# #                     epochs=epoch,
# #                     # validation_split=0.2,
# #                     verbose=True,
# #                     callbacks=[tensorboard])
# #
#
#
#
#
# #######                Save Model
# # model.save('E:/AutomatedTracing/Data/Models/ScenarioConnectome/'+PltNAme+'.h5')
#
#
#
# #
# # from keras.layers import Concatenate, Input, Dense
# # # First model - Image
# # first_input = Input((2197, ))
# # first_dense = Dense(128)(first_input)
# #
# # # Second model - Graph
# # second_input = Input((66, ))
# # second_dense = Dense(64)(second_input)
# #
# # # Third model - Features
# # third_input = Input((14,))
# # third_dense = Dense(32)(third_input)
# #
# # # Concatenate both
# # merged = Concatenate()([first_dense, second_dense,third_dense])
# # output_layer = Dense(1)(merged)
# #
# # model = Model(inputs=[first_input, second_input,third_input], outputs=output_layer)
# # # model.compile(optimizer='sgd', loss='mse')
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# # history = model.fit([IMsTrain1, ScenariosTrain, FeatureTrain], LabelsTrain, epochs=epoch, verbose=1)
#
#
# # # Initialize the constructor
# # model = Sequential()  # comes from import: from keras.models import Sequential
# # # Add an input layer
# # model.add(Dense(14, activation='relu', input_shape=(14,)))
# # # Add Batch Normalization
# # # model.add(BatchNormalization(axis=1))
# # # Add one hidden layer
# # model.add(Dense(8, activation='relu'))
# # # Add Batch Normalization
# # # model.add(BatchNormalization(axis=1))
# # # Add one hidden layer
# # model.add(Dense(4, activation='relu'))
# # # Add Batch Normalization
# # # model.add(BatchNormalization(axis=1))
# # # Add an output layer
# # model.add(Dense(1, activation='sigmoid'))
# # # Model output shape
# # model.output_shape
# #
# #
# # # compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               # metrics=['accuracy']
# #               metrics=['accuracy']
# #               )
# #
# #
# # # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
# # history = model.fit(FeatureTrain, LabelsTrain,
# #                     epochs=epoch,
# #                     # batch_size=len(X_train),
# #                     batch_size=batch_size,
# #                     verbose=verbose,
# #                     shuffle=True,
# #                     validation_split=0.2)
#
#
# # Plot training & validation accuracy values
