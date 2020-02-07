#
# Created on 1/10/2020
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

import numpy as np
import seaborn as sns; sns.set()
import pylab as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import scipy.io as sio

threeD = False
################## Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

ScenariosData = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_withIMnum.mat')
Scenarios = ScenariosData['Scenarios']
numScenarios = Scenarios.shape
# Scenarios[0,Scenarios.shape[1]-1]
Labels = ScenariosData['Labels']
# Labels[0,Labels.shape[1]-1]



################## PCA
mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
PCA_results = np.dot(x_train - mu, V.transpose())

Rpca = np.dot(PCA_results[:,:2], V[:2,:]) + mu    # reconstruction
err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)))

################## t-Distributed Stochastic Neighbor Embedding (t-SNE)
if threeD:
    ncom = 3
else:
    ncom = 2
t_sne = TSNE(n_components=ncom, verbose=1, perplexity=40, n_iter=300)
tsne_results = t_sne.fit_transform(x_train)

################## Training Auto Encoder
m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=(784,)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(64,  activation='elu'))
m.add(Dense(32,  activation='elu'))
if threeD:
    m.add(Dense(3,    activation='linear', name="encoder"))
else:
    m.add(Dense(2, activation='linear', name="encoder"))
m.add(Dense(32,  activation='elu'))
m.add(Dense(64,  activation='elu'))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(x_train, x_train, batch_size=128, epochs=50, verbose=1,
                validation_data=(x_test, x_test))
encoder = Model(m.input, m.get_layer('encoder').output)
# encoder representation
Encoder_results = encoder.predict(x_train)
# reconstruction of original data
Renc = m.predict(x_train)

################## Plotting PCA projection side-by-side with the bottleneck representation
if threeD:
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(PCA_results[:6000, 0], PCA_results[:6000, 1], PCA_results[:6000, 2],c=y_train[:6000])
    plt.title('PCA')
    pyplot.show()

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(Encoder_results[:6000, 0], Encoder_results[:6000, 1], Encoder_results[:6000, 2], c=y_train[:6000])
    plt.title('Autoencoder')
    pyplot.show()

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(tsne_results[:6000, 0], tsne_results[:6000, 1], tsne_results[:6000, 2], c=y_train[:6000])
    plt.title('t-SNE')
    pyplot.show()
else:
    # PCA_results, Encoder_results, tsne_results
    plt.figure(figsize=(8,4))
    plt.subplot(221)
    plt.title('PCA')
    plt.scatter(PCA_results[:6000,0], PCA_results[:6000,1], c=y_train[:6000], s=8)
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    plt.subplot(222)
    plt.title('Autoencoder')
    plt.scatter(Encoder_results[:6000,0], Encoder_results[:6000,1], c=y_train[:6000], s=8)
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])
    plt.tight_layout()

    plt.subplot(223)
    plt.title('t-SNE')
    plt.scatter(tsne_results[:6000, 0], tsne_results[:6000, 1], c=y_train[:6000], s=8)
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])
    plt.tight_layout()

################## Reconstructions
# plt.figure(figsize=(9,3))
# toPlot = (x_train, Rpca, Renc)
# for i in range(10):
#     for j in range(3):
#         ax = plt.subplot(3, 10, 10*j+i+1)
#         plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest",
#                    vmin=0, vmax=1)
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
# plt.tight_layout()