import numpy as np
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.callbacks import TensorBoard

maxNumPoints = 12
inputSize  = int((maxNumPoints * (maxNumPoints - 1))/2)
scenario_arr=np.empty(inputSize)
scenario_arr[:] = -1
train = []

G = sio.loadmat('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\scenarios.mat')
MergerAM = G['MergerAM']
numBranches = MergerAM.shape
for i in range(numBranches[1]):
    scenarios = MergerAM[0,i]
    if scenarios.any():
        for j in range(scenarios.shape[2]):
            scenario =  scenarios[:,:,j]
            upperTriangle = scenario[np.triu_indices(scenarios.shape[0], k = 1)]
            scenario_arr[0:len(upperTriangle)] = upperTriangle
            train.append(scenario_arr)

x_train, x_test = train_test_split(train, test_size=0.33, random_state=42)

x_train = np.asarray(x_train, dtype=np.uint8)
x_test = np.asarray(x_test, dtype=np.uint8)


## ------------------------------------    Convolutional autoencoder
imSize = 8
x_train1 = []
x_test1 = []
for i in range(len(x_train)):
    x_train1.append(x_train[i,:64].reshape(imSize,imSize,1))
for i in range(len(x_test)):
    x_test1.append(x_test[i, :64].reshape(imSize, imSize,1))

x_train1 = np.asarray(x_train1, dtype=np.uint8)
x_test1 = np.asarray(x_test1, dtype=np.uint8)
print(x_train1.shape)
print(x_test1.shape)
input_img = Input(shape=(imSize, imSize, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# First, let's open up a terminal and start a TensorBoard server that will read logs stored at /tmp/autoencoder.
#
#     tensorboard --logdir=/tmp/autoencoder

autoencoder.fit(x_train1, x_test1,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(imSize, imSize))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(imSize, imSize))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(n):
#     ax = plt.subplot(1, n, i)
#     plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


# ## ------------------------------------    Deep autoencoder
#
# input_img = Input(shape=(inputSize,))
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(32, activation='relu')(encoded)
#
# decoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(128, activation='relu')(decoded)
# decoded = Dense(784, activation='sigmoid')(decoded)
#
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# autoencoder.fit(x_train, x_train,
#                 epochs=100,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))



## ------------------------------------    simplest autoencoder

# # this is the size of our encoded representations
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# # this is our input placeholder
# input_img = Input(shape=(inputSize,))
# # "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(inputSize, activation='sigmoid')(encoded)
# # this model maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
# # this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)
# # create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# # retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))




# # encode and decode some digits
# # note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
# weights1 = autoencoder.get_weights()
# layers1 = autoencoder.get_config()
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i,:64].reshape(8, 8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i,:64].reshape(8, 8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()