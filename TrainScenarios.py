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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
from keras.callbacks import TensorBoard
import datetime
import scipy.io as sio
from imblearn.over_sampling import SMOTE

x = datetime.datetime.today()
nowTimeDate = x.strftime("%b_%d_%Y_%H_%M")

epoch  = 80
batch_size = 400
verbose=1 #verbose=1 will show you an animated progress bar
doSMOTE = False #do replicate data using SMOTE method

PltNAme = 'TEMP_NoBatch_NoScale_Signoid_epch'+str(epoch)+'_batch_size='+str(batch_size)+'_SMOTE='+str(doSMOTE)+'_'+nowTimeDate

#### Read Data
# white = pd.read_csv("C:/Users/Seyed/Dropbox/PhytonLearning/winequality-white.csv",sep=";")
C1 = pd.read_csv("C1.csv",sep=";")
C2 = pd.read_csv("C2.csv",sep=";")

C1.shape
C2.shape
# filePath = 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features.mat'
# Data = sio.loadmat(filePath)
# C1 = Data['C1']
# C2 = Data['C2']
# TData = Data['TData']

##### Some Data info
print(C1.head()) # First Rows - provide you with a quick way of inspecting your data
print(C1.tail()) # Last Rows - provide you with a quick way of inspecting your data
print(C1.sample(5)) # 5 Sample Data
print(C1.describe()) # summary statistics about your data such as min max STD count mean
print(pd.isnull(C1)) # if Null show True
print(C1.info()) # info of each column
print(C2.info())

# # ####### MatPlot Histogram
# fig, ax = plt.subplots(1, 2)
# ax[0].hist(C1.Dist,10,facecolor='green', alpha=0.5,label='C1')
# ax[1].hist(C2.Dist,10,facecolor='C2', alpha=0.5,label='C2')
# #fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
# ax[0].set_ylim([0, 1000])
# ax[0].set_xlabel("Dist in % Vol")
# ax[0].set_ylabel("Frequency")
# ax[1].set_xlabel("Dist in % Vol")
# ax[1].set_ylabel("Frequency")
# #ax[0].legend(loc='best')
# #ax[1].legend(loc='best')
# fig.suptitle("Distribution of Dist in % Vol")
# plt.show()

# # ######### Histogram values
# print(np.histogram(C2.Dist, bins=[7,8,9,10,11,12,13,14,15]))
# print(np.histogram(C1.Dist, bins=[7,8,9,10,11,12,13,14,15]))

# # ######## MatPlot Scater - Relation between two variables
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].scatter(C2['Icv'], C2["Plan"], color="C2")
# ax[1].scatter(C1['Icv'], C1['Rcv'], color="C1", edgecolors="black", lw=0.5)
# ax[0].set_title("C2 Scenario")
# ax[1].set_title("C1 Scenario")
# ax[0].set_xlabel("Icv")
# ax[1].set_xlabel("Icv")
# ax[0].set_ylabel("Plan")
# ax[1].set_ylabel("Plan")
# ax[0].set_xlim([0,10])
# ax[1].set_xlim([0,10])
# ax[0].set_ylim([0,2.5])
# ax[1].set_ylim([0,2.5])
# fig.subplots_adjust(wspace=0.5)
# fig.suptitle("Scenario Icv by Amount of Plan")
# plt.show()

# # ########## Create show scater of two features
# np.random.seed(570)
# C2labels = np.unique(C2['Icv'])
# C1labels = np.unique(C1['Icv'])
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# C2colors = np.random.rand(6, 4)
# C1colors = np.append(C2colors, np.random.rand(1, 4), axis=0)
# for i in range(len(C2colors)):
#     C2y = C2['Dist'][C2.Icv == C2labels[i]] # Alcohl value which the Icv is equal to C2labels[i], for example is 3
#     C2x = C2['Cos2'][C2.Icv == C2labels[i]]
#     ax[0].scatter(C2x, C2y, c=C2colors[i])
# for i in range(len(C1colors)):
#     C1y = C1['Dist'][C1.Icv == C1labels[i]]
#     C1x = C1['Cos2'][C1.Icv == C1labels[i]]
#     ax[1].scatter(C1x, C1y, c=C1colors[i])
# ax[0].set_title("C2 Scenario")
# ax[1].set_title("C1 Scenario")
# ax[0].set_xlim([0, 1.7])
# ax[1].set_xlim([0, 1.7])
# ax[0].set_ylim([5, 15.5])
# ax[1].set_ylim([5, 15.5])
# ax[0].set_xlabel("Cos2")
# ax[0].set_ylabel("Dist")
# ax[1].set_xlabel("Cos2")
# ax[1].set_ylabel("Dist")
# # ax[0].legend(C2labels, loc='best', bbox_to_anchor=(1.3, 1))
# ax[1].legend(C1labels, loc='best', bbox_to_anchor=(1.3, 1))
# # fig.suptitle("Dist - Cos2")
# fig.subplots_adjust(top=0.85, wspace=0.7)
# plt.show()

#### Add Labels and append two datasets and make one dataset with labels
C2['type'] = 0 # Not Connect
C1['type'] = 1 # Connect
Scenarios = C2.append(C1,ignore_index=True)
Scenarios.shape
# ##### Correlation to show which data are more related, especialy their relation to the type (class = C2 or C1)
# corr = Scenarios.corr()
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# # sns.plt.show()
# # from sklearn.preprocessing import Normalizer
# # normalizeddf_train = Normalizer().fit_transform(Scenarios)
# # normalizeddf_train_ = pd.DataFrame(normalizeddf_train).corr(method='pearson')
# # corr = normalizeddf_train_.corr()
# # sns.heatmap(corr,
# #             xticklabels=corr.columns.values,
# #             yticklabels=corr.columns.values)
# # # sns.plt.show()



######################                   Generate Input, Labels, and Train and Test Data
# Specify the data only (without Labels)
X=Scenarios.ix[:,0:14]
X.shape
# Specify the Target labels and flatten the array (Labels)
y=np.ravel(Scenarios.type)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Balancing Data by SMOTE method
# np.count_nonzero(y_train == 1)
if doSMOTE:
    smt = SMOTE()
    X_train, y_train = smt.fit_sample(X_train, y_train)
    X_train.shape

# # Split the training data up in train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

################                                      Normalize Data
# # # Define the scaler
# scaler = StandardScaler().fit(X_train)
# # to convert one row
# # xtrain = X_train.values.tolist()
# # xtrain = numpy_matrix = X_train.as_matrix()
# # xtrain1 = xtrain[:1]
# # X_train1 = scaler.transform(xtrain1)
#
# # Scale the train set (Values between -1 and 1)
# X_train = scaler.transform(X_train)
# # Scale the test set
# X_test = scaler.transform(X_test)

################################### Create the model
# Initialize the constructor
model = Sequential()  # comes from import: from keras.models import Sequential
# Add an input layer
model.add(Dense(14, activation='relu', input_shape=(14,)))
# Add Batch Normalization
# model.add(BatchNormalization(axis=1))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add Batch Normalization
# model.add(BatchNormalization(axis=1))
# Add one hidden layer
model.add(Dense(4, activation='relu'))
# Add Batch Normalization
# model.add(BatchNormalization(axis=1))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))
# Model output shape
model.output_shape
# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors
model.get_weights()

########## Compile and fit the Model

# ########## ----------------------------------------- Define Custem Metric in Keras
from keras import backend
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
# model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# ########## -----------------------------------------

# compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              # metrics=['accuracy']
              metrics=['accuracy']
              )

tensorboard = TensorBoard(log_dir="logs/"+PltNAme)

# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
history = model.fit(X_train, y_train,
                    epochs=epoch,
                    # batch_size=len(X_train),
                    batch_size=batch_size,
                    verbose=verbose,
                    shuffle=True,
                    validation_split=0.2,
                    # validation_data=(X_val, y_val),
                    callbacks=[tensorboard])

#######                Save Model




# # load model
# model = load_model('E:/AutomatedTracing/Data/Models/ScenarioConnectome/Main_Signoid_epch80_batch_size=400_SMOTE=False_Oct_04_2019_09_35.h5')
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
# # summarize model.
# model.summary()

#######                Predict Values
y_pred = model.predict(X_test)
y_pred_pro = model.predict_proba(X_test)

# # to compare predict and test
# y_pred_round = y_pred>0.04
#
# plt.figure()
# plt.plot(y_pred[:10])
# plt.show()
# plt.figure()
# plt.plot(y_pred_round[:10])
# plt.show()
# y_pred[:5]
# y_pred_pro[:5]
# y_test[:5]
###############                            Evaluate Model
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

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

TP, FP, TN, FN = perf_measure(y_test.round(), y_pred.round())
plt.figure()
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
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




############### F1 score = 2TP / (TP+TN+FP+FN)
f1_score = f1_score(y_test.round(), y_pred.round())
print("F1 Score: ",f1_score)
###############  Confusion matrix
conf = confusion_matrix(y_test.round(), y_pred.round())
print("Confusion Matrix: ", conf)
###############  Precision = TP/(TP+FP)
precision = precision_score(y_test.round(), y_pred.round()) #  average=Nonefor precision from each class
print("Precision: ",precision)
############### Recall TP / (TP+FN) # Sensitivity, hit rate, recall, or true positive rate
recall = recall_score(y_test.round(), y_pred.round())
print("Recall: ",recall)

############### Cohen's kappa =
cohen_kappa_score = cohen_kappa_score(y_test.round(), y_pred.round())
print("Cohen_Kappa Score: ",cohen_kappa_score)
############### Accuracy = (TPR + TNR) / Total
accuracy= accuracy_score(y_test.round(), y_pred.round())
print("Accuracy: ",accuracy)
############### Balanced Accuracy = (TPR + TNR) / 2
balanced_accuracy= balanced_accuracy_score(y_test.round(), y_pred.round())
print("Balanced Accuracy: ",balanced_accuracy)

print("Correct Scenario Connections: ",TP)
print("Incorrect Scenario Connections: ",FN)

print("Done!")

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# weights = model.get_weights()
# plt.matshow(weights[1:2], cmap='viridis')
# plt.matshow(weights[3:4], cmap='viridis')
# weights[1]
##### Go Trhough the list
# for i in range(0, len(C1)):
#     print(C1.Dist[i])

## Run Tensorboard
# tensorboard --logdir=E:\AutomatedTracing\AutomatedTracing\Python\logs