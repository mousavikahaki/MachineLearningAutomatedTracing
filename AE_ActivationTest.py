#
# Created on 1/10/2020
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import pylab as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf


threeD = False
################## Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

featurecolumns = [ 'pix'+str(i) for i in range(x_train.shape[1]) ]
df = pd.DataFrame(x_train,columns=featurecolumns)
df['y'] = y_train
df['label'] = df['y'].apply(lambda i: str(i))
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = 60000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[featurecolumns].values

################## PCA
# pca = PCA(n_components=3)
# PCA_results = pca.fit_transform(data_subset)
# df_subset['pca1'] = PCA_results[:,0]
# df_subset['pca2'] = PCA_results[:,1]
# df_subset['pca3'] = PCA_results[:,2]



# Rpca = np.dot(PCA_results[:,:2], V[:2,:]) + mu    # reconstruction
# err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
# print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)))

################## t-Distributed Stochastic Neighbor Embedding (t-SNE)
# if threeD:
#     ncom = 3
# else:
#     ncom = 2
# t_sne = TSNE(n_components=ncom, verbose=1, perplexity=40, n_iter=300)
# tsne_results = t_sne.fit_transform(data_subset)
#
# df_subset['tsne-1'] = tsne_results[:,0]
# df_subset['tsne-2'] = tsne_results[:,1]
# if threeD:
#     df_subset['tsne-3'] = tsne_results[:,2]



# ################################### elu
# m = Sequential()
# m.add(Dense(512,  activation='elu', input_shape=(784,)))
# m.add(Dense(128,  activation='elu'))
# if threeD:
#     m.add(Dense(3,    activation='linear', name="encoder"))
# else:
#     m.add(Dense(2, activation='linear', name="encoder"))
# m.add(Dense(128,  activation='elu'))
# m.add(Dense(512,  activation='elu'))
# m.add(Dense(784,  activation='sigmoid'))
# m.compile(loss='mean_squared_error', optimizer = Adam())
# PltNAme = 'AE_elu'
# tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# history = m.fit(data_subset, data_subset, batch_size=64, epochs=10, verbose=1,
#                 validation_data=(x_test, x_test),
#                     callbacks=[tensorboard])
# encoder = Model(m.input, m.get_layer('encoder').output)
# # encoder representation
# Encoder_results = encoder.predict(data_subset)
# # reconstruction of original data
# Renc = m.predict(data_subset)
#
# df_subset['AE1'] = Encoder_results[:,0]
# df_subset['AE2'] = Encoder_results[:,1]
# if threeD:
#     df_subset['AE3'] = Encoder_results[:,2]
#
# ax1 = plt.subplot(2, 2, 1)
# sns.scatterplot(
#         x="AE1", y="AE2",
#         hue="y",
#         palette=sns.color_palette("hls", 10),
#         data=df_subset,
#         legend="full",
#         alpha=0.3,
#         ax=ax1
#     )
# ax1.set_title("Autoencoder")
#
#
# ################################### ReLU
#
# def SigReLU(x):
#     return (K.sigmoid(x) * 5) - 1
#
#
# m = Sequential()
# m.add(Dense(512,  activation=SigReLU, input_shape=(784,)))
# m.add(Dense(128,  activation=SigReLU))
# if threeD:
#     m.add(Dense(3,    activation='linear', name="encoder"))
# else:
#     m.add(Dense(2, activation='linear', name="encoder"))
# m.add(Dense(128,  activation=SigReLU))
# m.add(Dense(512,  activation=SigReLU))
# m.add(Dense(784,  activation='sigmoid'))
# m.compile(loss='mean_squared_error', optimizer = Adam())
# PltNAme = 'AE_ReLU'
# tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# history = m.fit(data_subset, data_subset, batch_size=64, epochs=10, verbose=1,
#                 validation_data=(x_test, x_test),
#                     callbacks=[tensorboard])
# encoder = Model(m.input, m.get_layer('encoder').output)
# # encoder representation
# Encoder_results = encoder.predict(data_subset)
# # reconstruction of original data
# Renc = m.predict(data_subset)
#
# df_subset['AE4'] = Encoder_results[:,0]
# df_subset['AE5'] = Encoder_results[:,1]
# if threeD:
#     df_subset['AE6'] = Encoder_results[:,2]
#
# ax1 = plt.subplot(2, 2, 3)
# sns.scatterplot(
#         x="AE4", y="AE5",
#         hue="y",
#         palette=sns.color_palette("hls", 10),
#         data=df_subset,
#         legend="full",
#         alpha=0.3,
#         ax=ax1
#     )
# ax1.set_title("Autoencoder")
#
# ################################### ReLU
# from keras import backend as K
# func = 'swish'
#
# def swish(x):
#     beta = 1
#     alpha = 1
#     return (K.sigmoid(beta * x) * alpha *x)
#
#
# m = Sequential()
# m.add(Dense(512,  activation=swish, input_shape=(784,)))
# m.add(Dense(128,  activation=swish))
# if threeD:
#     m.add(Dense(3,    activation='linear', name="encoder"))
# else:
#     m.add(Dense(2, activation='linear', name="encoder"))
# m.add(Dense(128,  activation=swish))
# m.add(Dense(512,  activation=swish))
# m.add(Dense(784,  activation='sigmoid'))
# m.compile(loss='mean_squared_error', optimizer = Adam())
# PltNAme = 'AE_'+func
# tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
# history = m.fit(data_subset, data_subset, batch_size=64, epochs=10, verbose=1,
#                 validation_data=(x_test, x_test),
#                     callbacks=[tensorboard])
# encoder = Model(m.input, m.get_layer('encoder').output)
# # encoder representation
# Encoder_results = encoder.predict(data_subset)
# # reconstruction of original data
# Renc = m.predict(data_subset)
#
# df_subset['AE4'] = Encoder_results[:,0]
# df_subset['AE5'] = Encoder_results[:,1]
# if threeD:
#     df_subset['AE6'] = Encoder_results[:,2]
#
# ax1 = plt.subplot(2, 2, 2)
# sns.scatterplot(
#         x="AE4", y="AE5",
#         hue="y",
#         palette=sns.color_palette("hls", 10),
#         data=df_subset,
#         legend="full",
#         alpha=0.3,
#         ax=ax1
#     )
# ax1.set_title("Autoencoder")

##j################################# ReLU
import math
def Sofa(x):
    lamb = 0.98
    return (K.sigmoid(x) * K.tanh(x))
    # return 5 * (K.sigmoid(x) * (K.tanh(x)))
    # return (K.sigmoid(x) * (1 / K.tanh(x)))

    # return (K.sigmoid(K.pow(x,-1)) * x)
    # return x * (K.pow(1 + K.exp(-x), -1))
    # return tf.cond(x > 0, lambda: (K.sigmoid(-x) * 5) - 1, lambda: x/(1+K.sigmoid(-x)))

    # if tf.cond(x>0):
    #     return (K.sigmoid(-x) * 5) - 1
    # else:
    #     return x/(1+K.sigmoid(-x))
    # return 0.0001*(x / (1 + K.exp(-x)))
    # return x * (K.pow(1 + K.exp(-x),-1))


# def ReLU(x):
#     return 1. - x * x
#
# def dReLU(x):
#     return 1. * (x > 0)

m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=(784,)))
m.add(Dense(128,  activation='elu'))
if threeD:
    m.add(Dense(3,    activation='linear', name="encoder"))
else:
    m.add(Dense(2, activation='linear', name="encoder"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
PltNAme = 'AE__elu'
tensorboard = TensorBoard(log_dir="logs/"+PltNAme)
history = m.fit(data_subset, data_subset, batch_size=64, epochs=10000, verbose=1,
                validation_data=(x_test, x_test),
                    callbacks=[tensorboard])
encoder = Model(m.input, m.get_layer('encoder').output)
# encoder representation
Encoder_results = encoder.predict(data_subset)
# reconstruction of original data
Renc = m.predict(data_subset)

df_subset['AE1'] = Encoder_results[:,0]
df_subset['AE2'] = Encoder_results[:,1]
if threeD:
    df_subset['AE3'] = Encoder_results[:,2]

# ax1 = plt.subplot(2, 2, 1)
sns.scatterplot(
        x="AE1", y="AE2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
# ax1.set_title("Autoencoder")



# m.save('E:\AutomatedTracing\Data\AutoEncoderResults\model_K.sigmoid(x)_K.tanh(x)_epoch_10000.h5')










# ################## Reconstructions
# # plt.figure(figsize=(9,3))
# # toPlot = (x_train, Rpca, Renc)
# # for i in range(10):
# #     for j in range(3):
# #         ax = plt.subplot(3, 10, 10*j+i+1)
# #         plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest",
# #                    vmin=0, vmax=1)
# #         plt.gray()
# #         ax.get_xaxis().set_visible(False)
# #         ax.get_yaxis().set_visible(False)
# #
# # plt.tight_layout()