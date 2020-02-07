# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:54:09 2019

@author: Shih-Luen Wang
"""
import keras
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Nadam
from keras.layers import (Concatenate, Conv3D, Dropout, Input,
                          Dense, MaxPooling3D, UpSampling3D, Activation,Reshape, Lambda,
                          Permute)
from keras.backend import reshape, squeeze
from keras.models import Model, load_model
import numpy as np
import scipy.io as sio
from scipy.ndimage.interpolation import rotate
from sys import getsizeof
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn import preprocessing
import itertools
import time
import os

######################################################

def max_proj(x):
    y = np.zeros((len(x),len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            y[i,j] = np.amax(x[i,j,:])
    return y

def PlotIM_r(IM,AM,r):
    plt.figure(100)
    plt.imshow(max_proj(IM),cmap='gray')
    plt.scatter(r[:,1],r[:,0],s=1,c='r')
    plt.show()
    
def PlotIM_seeds(IM,AM,r,y):
    plt.imshow(max_proj(IM),cmap='gray')
    plt.scatter(r[y==0,1],r[y==0,0],s=8,c='yellow',marker='*')
    plt.scatter(r[y==1,1],r[y==1,0],s=5,c='r',marker='*')
    plt.scatter(r[y==2,1],r[y==2,0],s=8,c='b',marker='*')
    plt.show()

def plot_seedpairs(a,px,py,pz):
    m = a[:(2*px)*(2*py)*(2*pz)].reshape((2*px), (2*py), (2*pz), order='F')
    mm = a[(2*px)*(2*py)*(2*pz):2*(2*px)*(2*py)*(2*pz)].reshape((2*px), (2*py), (2*pz), order='F')
    m2 = max_proj(m)
    mm2 = max_proj(mm)
    m3 = np.zeros([m2.shape[0],m2.shape[1],3])
    m3[:,:,0] = m2
    m3[:,:,1] = mm2
    plt.figure()
    plt.imshow(m3)
    plt.show()
    
def get_seedpairs(data_name,imn,px,py,pz,d_thresh):
    data_dir = '../../data/'+data_name+'/'
    file_name = str(imn + 1) + '_L6_AS_opt10'
    file_path = data_dir + file_name
    
    stru = sio.loadmat(file_path)
    IM = stru['Original']
    IM = (IM/255)
    AM = stru['AM']
    r = stru['r'] - [1,1,1]
    
    train_x = np.zeros([1,2*(2*px)*(2*py)*(2*pz)+2])
    for i in range(AM.shape[0]):
        [xi,yi,zi] = np.round(r[i,:]).astype(int)
        for j in range(i):
            [xj,yj,zj] = np.round(r[j,:]).astype(int)
            if (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2 < d_thresh**2:
                xc = (xi + xj) // 2
                yc = (yi + yj) // 2
                zc = (zi + zj) // 2
                if xc > px and xc < IM.shape[0] - px and yc > py and yc < IM.shape[1] - py and zc > pz and zc < IM.shape[2] - pz:
                    temp2 = np.zeros([(2*px),(2*py),(2*pz)])
                    temp2[px-xc+xi,py-yc+yi,pz-zc+zi] = 1
                    temp2[px-xc+xj,py-yc+yj,pz-zc+zj] = 1
                    temp = np.append(IM[xc-px:xc+px,yc-py:yc+py,zc-pz:zc+pz].flatten('F'),temp2.flatten('F'))
                    temp = np.insert(temp,temp.shape,AM[i,j]>0)
                    temp = np.insert(temp,temp.shape,((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)**(1/2))
                    temp = np.reshape(temp,(-1,temp.shape[0]))
                    train_x = np.concatenate((train_x,temp),axis=0)
    data = train_x[1:,:]
    return data

def get_train_val_test(source,opt,d_thresh,imn_test,val_split,px,py,pz):
    start_time = time.time() 
    M_test = np.empty([1,2*px*2*py*2*pz*2+4])
    if source == 'mat':    
        j = 1
        while os.path.isfile(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython'+'/'+'stack'+str(imn_test+1)+'/'+'stack'+str(imn_test+1)+'_'+str(j)+'.mat'):
            model_var = sio.loadmat(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython'+'/'+'stack'+str(imn_test+1)+'/'+'stack'+str(imn_test+1)+'_'+str(j)+'.mat')            
            data = model_var['data']
            M_test = np.concatenate((M_test,data),axis=0)
            j = j + 1
        M_test = M_test[1:,:]
    print(M_test.shape)
    print(np.mean(M_test[:,-2]))
    print(np.sum(M_test[:,-2]))
    
    
    M_ = np.empty([1,M_test.shape[1]])
    for imn in range(6):
        if imn != imn_test:
            if source == 'mat':    
                j = 1
                while os.path.isfile(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython'+'/'+'stack'+str(imn+1)+'/'+'stack'+str(imn+1)+'_'+str(j)+'.mat'):
                    model_var = sio.loadmat(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython'+'/'+'stack'+str(imn+1)+'/'+'stack'+str(imn+1)+'_'+str(j)+'.mat')
                    j = j + 1
                    data = model_var['data']
                    M_ = np.concatenate((M_,data),axis=0)
            print(M_.shape)
            print(np.mean(M_[:,-2]))
            print(np.sum(M_[:,-2]))  
            
    M_ = M_[1:,:]
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print(M_.shape)
    print(M_test.shape)
    
    ind = np.arange(M_.shape[0])
    np.random.shuffle(ind) 
    val_ind = ind[:int(M_.shape[0]*val_split)]
    train_ind = np.delete(ind,val_ind)
    
    M_train = M_[train_ind,:] 
    M_val = M_[val_ind,:]
    
    return M_train,M_val,M_test

def get_train_val_test_sm(source,opt,d_thresh,imn_test,val_split,px,py,pz):
    start_time = time.time() 
    M_test = np.empty([1,2*px*2*py*2*pz*2+4])
    if source == 'mat':    
        j = 1
        while os.path.isfile(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython_Smoothed'+'/'+'stack'+str(imn_test+1)+'/'+'stack'+str(imn_test+1)+'_'+str(j)+'.mat'):
            model_var = sio.loadmat(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython_Smoothed'+'/'+'stack'+str(imn_test+1)+'/'+'stack'+str(imn_test+1)+'_'+str(j)+'.mat')            
            data = model_var['data']
            M_test = np.concatenate((M_test,data),axis=0)
            j = j + 1
        M_test = M_test[1:,:]
    print(M_test.shape)
    print(np.mean(M_test[:,-2]))
    print(np.sum(M_test[:,-2]))
    
    
    M_ = np.empty([1,M_test.shape[1]])
    for imn in range(6):
        if imn != imn_test:
            if source == 'mat':    
                j = 1
                while os.path.isfile(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython_Smoothed'+'/'+'stack'+str(imn+1)+'/'+'stack'+str(imn+1)+'_'+str(j)+'.mat'):
                    model_var = sio.loadmat(data_dir+'Pad_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_forPython_Smoothed'+'/'+'stack'+str(imn+1)+'/'+'stack'+str(imn+1)+'_'+str(j)+'.mat')
                    j = j + 1
                    data = model_var['data']
                    M_ = np.concatenate((M_,data),axis=0)
            print(M_.shape)
            print(np.mean(M_[:,-2]))
            print(np.sum(M_[:,-2]))  
            
    M_ = M_[1:,:]
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print(M_.shape)
    print(M_test.shape)
    
    ind = np.arange(M_.shape[0])
    np.random.shuffle(ind) 
    val_ind = ind[:int(M_.shape[0]*val_split)]
    train_ind = np.delete(ind,val_ind)
    
    M_train = M_[train_ind,:] 
    M_val = M_[val_ind,:]
    
    return M_train,M_val,M_test

def get_ShallowNN(optimizer):
    model = Sequential()
    model.add(Dense(10, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def get_MultiDenseNN(optimizer):
    model = Sequential()
#    model.add(Dense(100, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), activation='relu'))
#    model.add(Dense(100, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), activation='relu'))
#    model.add(Dense(100, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), activation='relu'))
    model.add(Dense(100, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def get_encodedMultiDenseNN(optimizer):
    load_folder = '../../data/training_result/'+'Autoencoder_DenseNN2_SGD_Pad5_600epoch_aug8__opt5_d15_px8_py8_pz8_lr0.001_batchsize100'
    encoder = load_model(load_folder+'/encoder',custom_objects={'squeeze':squeeze})
    encoder.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    input_im = Input(shape=(2*px*2*py*2*pz*2, ), name='input_im')
    encoded_input = encoder(input_im)
    latent_1 = Dense(100, activation='relu')(encoded_input)
#    latent_2 = Dense(100, activation='relu')(latent_1)
#    latent_3 = Dense(100, activation='relu')(latent_2)
    output_im = Dense(1, activation='sigmoid')(latent_1)

    gan = Model(inputs=input_im, outputs=output_im)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def get_3DUNet(optimizer):
    
    conv_properties = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
       
    input_im = Input(shape=(2*(2*px)*(2*py)*(2*pz), ), name='input_im')
#    input_im = Reshape((None,(2*px),(2*py),(2*pz),2), input_shape=(None,2*(2*px)*(2*py)*(2*pz), 1))
#    reshaped_im = reshape(input_im,(-1,(2*px),(2*py),(2*pz),2))
    reshaped_im = Reshape((2,(2*pz),(2*py),(2*px)))(input_im)
    reshaped_im = Permute((4,3,2,1))(reshaped_im)
    conv1 = Conv3D(8, (3, 3, 3), **conv_properties)(reshaped_im)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    conv2 = Conv3D(8, (3, 3, 3), **conv_properties)(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    conv3 = Conv3D(4, (3, 3, 1), **conv_properties)(pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    conv4 = Conv3D(4, (3, 3, 1), **conv_properties)(pool3)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)
    
    conv5 = Conv3D(4, (3, 3, 1), **conv_properties)(pool4)
    drop5 = Dropout(0.2)(conv5)
    
    up6 = UpSampling3D(size=(2, 2, 1))(drop5)
    up6 = Conv3D(4, (2, 2, 1), **conv_properties)(up6)
    cat6 = Concatenate(axis=-1)([drop4, up6])
    conv6 = Conv3D(4, (3, 3, 1), **conv_properties)(cat6)
    
    up7 = UpSampling3D(size=(2, 2, 1))(conv6)
    up7 = Conv3D(4, (2, 2, 1), **conv_properties)(up7)
    cat7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv3D(4, (3, 3, 1), **conv_properties)(cat7)
    
    up8 = UpSampling3D(size=(2, 2, 1))(conv7)
    up8 = Conv3D(8, (2, 2, 1), **conv_properties)(up8)
    cat8 = Concatenate(axis=-1)([conv2, up8])  
    conv8 = Conv3D(8, 3, **conv_properties)(cat8)
    
    up9 = UpSampling3D(size=(2, 2, 1))(conv8)
    up9 = Conv3D(8, (2, 2, 1), **conv_properties)(up9)
    cat9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv3D(8, (3, 3, 3), **conv_properties)(cat9)
   
    drop9 = Dropout(0.5)(conv9) 
    output_im = Conv3D(1, (2*px, 2*py, 2*pz), padding='valid', activation='sigmoid', name='output_im')(drop9)
    output_im = Lambda(lambda x: squeeze(x,3))(output_im)
    output_im = Lambda(lambda x: squeeze(x,2))(output_im)
    output_im = Lambda(lambda x: squeeze(x,1))(output_im)
    
    model = Model(inputs=[input_im], outputs=[output_im])
    
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics=['accuracy'])

    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def rotflip8(x_train, y_train):
    IM = np.reshape(x_train,(x_train.shape[0],2*px,2*py,2*pz,2),order='F')
    Output = x_train
    for i in range(1,8):
        temp = IM
        if i > 3:
            temp = np.flip(temp, 1)
        temp = np.rot90(temp, i % 4 , (1,2))
        temp = np.reshape(temp, (temp.shape[0],2*px*2*py*2*pz*2), order = 'F')
        Output = np.concatenate((Output,temp),axis=0)
    
    label = y_train
    label = np.tile(label,8)


#    plt.figure()
#    subimage = np.zeros([2*px,2*py,3])
#    subimage[:,:,0] = max_proj(IM[0,:,:,:,0]) 
#    subimage[:,:,1] = max_proj(IM[0,:,:,:,1]) 
#    plt.imshow(subimage)
#    plt.show() 
#    print(y_train[0])
#    
#    for i in range(1,8):
#        plt.figure()
#        subimage = np.zeros([2*px,2*py,3])
#        X = np.reshape(Output,(Output.shape[0],2*px,2*py,2*pz,2),order='F')
#        subimage[:,:,0] = max_proj(X[i*x_train.shape[0],:,:,:,0]) 
#        subimage[:,:,1] = max_proj(X[i*x_train.shape[0],:,:,:,1]) 
#        plt.imshow(subimage)
#        plt.show()
#        print(label[i*x_train.shape[0]])                    
    return Output, label

def augmentation(x_train, y_train, r):
    
    IM = np.reshape(x_train,(x_train.shape[0],2*px,2*py,2*pz,2),order='F')
    Output = x_train
    for i in range(1,2*r):
        temp = IM
        if i > r - 1:
            temp = np.flip(temp, 1)
        temp = rotate(temp, i*360/r, axes=(1,2),reshape=False)
        temp = np.reshape(temp, (temp.shape[0],2*px*2*py*2*pz*2), order = 'F')
        Output = np.concatenate((Output,temp),axis=0)

    label = y_train
    label = np.tile(label,2*r)
    
#    plt.figure()
#    subimage = np.zeros([2*px,2*py,3])
#    subimage[:,:,0] = max_proj(IM[0,:,:,:,0]) 
#    subimage[:,:,1] = max_proj(IM[0,:,:,:,1]) 
#    plt.imshow(subimage)
#    plt.show() 
#    print(y_train[0])
#    
#    for i in range(1,2*r):
#        plt.figure()
#        subimage = np.zeros([2*px,2*py,3])
#        X = np.reshape(Output,(Output.shape[0],2*px,2*py,2*pz,2),order='F')
#        subimage[:,:,0] = max_proj(X[i*x_train.shape[0],:,:,:,0]) 
#        subimage[:,:,1] = max_proj(X[i*x_train.shape[0],:,:,:,1]) 
#        plt.imshow(subimage)
#        plt.show()
#        print(label[i*x_train.shape[0]])  
    
    return Output, label

############################# main

########################## parameters


data_dir = 'L1/'
imn_test = 4
d_thresh = 15
opt = 5
val_split = 0.2
epoch = 1000
learning_rate = 0.001
folder_name = 'test'
px = 8
py = 8
pz = 8
batch_size = 200
r = 8

############################################################################

#M_train,M_val,M_test = get_train_val_test_onall('mat',opt,d_thresh,val_split,px,py,pz)
#M_train,M_val,M_test = get_train_val_test('mat',opt,d_thresh,imn_test,val_split,px,py,pz)
M_train,M_val,M_test = get_train_val_test_sm('mat',opt,d_thresh,imn_test,val_split,px,py,pz)


x_train = M_train[:,:-4]
y_train = M_train[:,-4]
#x_train, y_train = rotflip8(x_train, y_train)
x_train, y_train = augmentation(x_train, y_train,r)
x_val = M_val[:,:-4]
y_val = M_val[:,-4]
x_test = M_test[:,:-4]
y_test = M_test[:,-4]
dist_test = M_test[:,-3]

x_val = x_test
y_val = y_test

#x_train[:,:2*px*2*py*2*pz] = preprocessing.scale(x_train[:,:2*px*2*py*2*pz])
#x_val[:,:2*px*2*py*2*pz] = preprocessing.scale(x_val[:,:2*px*2*py*2*pz])
#x_test[:,:2*px*2*py*2*pz] = preprocessing.scale(x_test[:,:2*px*2*py*2*pz])
    
model = get_ShallowNN(SGD(lr=learning_rate))
#model = get_MultiDenseNN(SGD(lr=learning_rate))
#model = get_encodedMultiDenseNN(SGD(lr=learning_rate))
#model = get_MultiDenseNN(Adam(lr=learning_rate))
#model = get_3DUNet(Adam(lr=learning_rate))
#model = get_3DUNet(SGD(lr=learning_rate))

start_time = time.time()

history = model.fit(x_train, y_train,
          epochs=epoch,
          batch_size=batch_size,
          validation_data=(x_val,y_val))

training_time = time.time() - start_time

start_time = time.time()

result = model.predict(x_test)

test_time = time.time() - start_time

start_time = time.time()

#save_folder = '../../data/training_result/'+'3DUNet_SGD_'+'Pad'+str(imn_test+1)+'_'+str(epoch)+'epoch_'+'aug8_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_lr'+str(learning_rate)+'_batchsize'+str(batch_size)
#save_folder = '../../data/training_result/'+'VT_'+'EmNN1_'+'SGD_'+'f2r'+str(r)+'_'+'Pad'+str(imn_test+1)+'_'+str(epoch)+'epoch_'+'aug8_'+'_opt'+str(opt)+'_d'+str(d_thresh)+'_px'+str(px)+'_py'+str(py)+'_pz'+str(pz)+'_lr'+str(learning_rate)+'_batchsize'+str(batch_size)+'Sm'
save_folder = 'Shallow_test2'
createFolder(save_folder)
model.save(save_folder+'/model')
np.savez(save_folder+'/result.npz', result = result)

result_train = model.predict(x_train)
np.savez(save_folder+'/result_train.npz', result_train = result_train, y_train = y_train)

result_val = model.predict(x_val)
np.savez(save_folder+'/result_val.npz', result_val = result_val, y_val = y_val)

np.savez(save_folder+'/dist_test.npz', dist_test = dist_test)

history_train_loss = history.history['loss']
history_val_loss = history.history['val_loss']

np.savez(save_folder+'/history.npz', history_train_loss = history_train_loss, history_val_loss = history_val_loss)

saving_time = time.time() - start_time

np.savez(save_folder+'/running_time.npz', training_time = training_time, test_time = test_time, saving_time = saving_time)


