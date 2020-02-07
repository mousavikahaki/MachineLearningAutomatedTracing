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
pca = PCA(n_components=3)
PCA_results = pca.fit_transform(data_subset)
df_subset['pca1'] = PCA_results[:,0]
df_subset['pca2'] = PCA_results[:,1]
df_subset['pca3'] = PCA_results[:,2]



# Rpca = np.dot(PCA_results[:,:2], V[:2,:]) + mu    # reconstruction
# err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
# print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)))

################## t-Distributed Stochastic Neighbor Embedding (t-SNE)
if threeD:
    ncom = 3
else:
    ncom = 2
t_sne = TSNE(n_components=ncom, verbose=1, perplexity=40, n_iter=300)
tsne_results = t_sne.fit_transform(data_subset)

df_subset['tsne-1'] = tsne_results[:,0]
df_subset['tsne-2'] = tsne_results[:,1]
if threeD:
    df_subset['tsne-3'] = tsne_results[:,2]

################## Training Auto Encoder
import numpy
import sys
def dtanh(x):
    return 1. - x * 3.14159265359 * x

def ReLU(x):
    return 1. - x * x

def dReLU(x):
    return 1. * (x > 0)

m = Sequential()
m.add(Dense(512,  activation=ReLU, input_shape=(784,)))
m.add(Dense(128,  activation=ReLU))
if threeD:
    m.add(Dense(3,    activation='linear', name="encoder"))
else:
    m.add(Dense(2, activation='linear', name="encoder"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(data_subset, data_subset, batch_size=64, epochs=10, verbose=1,
                validation_data=(x_test, x_test))
encoder = Model(m.input, m.get_layer('encoder').output)
# encoder representation
Encoder_results = encoder.predict(data_subset)
# reconstruction of original data
Renc = m.predict(data_subset)

df_subset['AE1'] = Encoder_results[:,0]
df_subset['AE2'] = Encoder_results[:,1]
if threeD:
    df_subset['AE3'] = Encoder_results[:,2]

ax1 = plt.subplot(2, 2, 1)
sns.scatterplot(
        x="AE1", y="AE2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
ax1.set_title("Autoencoder")

################## Plotting
if threeD:
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset.loc[rndperm, :]["AE1"],
        ys=df_subset.loc[rndperm, :]["AE2"],
        zs=df_subset.loc[rndperm, :]["AE3"],
        c=df.loc[rndperm, :]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('AE1')
    ax.set_ylabel('AE2')
    ax.set_zlabel('AE3')
    plt.show()

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset.loc[rndperm, :]["pca1"],
        ys=df_subset.loc[rndperm, :]["pca2"],
        zs=df_subset.loc[rndperm, :]["pca3"],
        c=df.loc[rndperm, :]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_zlabel('pca3')
    plt.show()

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset.loc[rndperm, :]["pca1"],
        ys=df_subset.loc[rndperm, :]["pca2"],
        zs=df_subset.loc[rndperm, :]["pca3"],
        c=df.loc[rndperm, :]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_zlabel('pca3')
    plt.show()

    # ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    # ax.scatter(
    #     xs=df_subset.loc[rndperm, :]["tsne-1"],
    #     ys=df_subset.loc[rndperm, :]["tsne-2"],
    #     zs=df_subset.loc[rndperm, :]["tsne-3"],
    #     c=df.loc[rndperm, :]["y"],
    #     cmap='tab10'
    # )
    # ax.set_xlabel('tsne-1')
    # ax.set_ylabel('tsne-2')
    # ax.set_zlabel('tsne-3')
    # plt.show()


else:

    plt.figure(figsize=(16, 20))

    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(
        x="AE1", y="AE2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax1.set_title("Autoencoder")

    ax2 = plt.subplot(2, 2, 2)
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax = ax2
    )
    ax2.set_title("PCA")

    # ax3 = plt.subplot(3, 1, 3)
    # sns.scatterplot(
    #     x="tsne-1", y="tsne-2",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=0.3,
    #     ax = ax3
    # )
    # ax3.set_title("TSNE")
    plt.tight_layout()



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