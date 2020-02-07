#
# Created on 8/8/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#


import numpy as np
from skimage import io
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import sparse


def loadIMTrace(path):
    G = sio.loadmat(path)
    IM = G['IM']
    AM = G['AM']
    r = G['r']
    R = G['R']
    return IM, AM, r, R

def PlotAM(AM,r,IM):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    var = {}
    AM_BP = np.asarray(AM)
    var['AM_BP'] = sparse.csr_matrix(AM_BP)
    var['r'] = r
    var['IM'] = IM
    sio.savemat('temp.mat', var)
    ## Plot Trace and Image
    eng.evalc("s = load('temp.mat');figure;imshow(max(s.IM,[],3));hold on;PlotAM_1(s.AM_BP, s.r)")
    return eng

def findBranchPoints(AM):
    AM_tmp = AM
    AM_BP = np.zeros((AM.shape))
    maxvalue = []
    BPidx = []
    AM_A = AM.toarray()
    for i in range(AM.shape[1]):
        maxvalue = np.count_nonzero(AM_A[i, :])
        if maxvalue > 2:
            BPidx.append(i)
            AM_BP[i, :] = AM_A[i, :]
            AM_BP[:, i] = AM_A[:, i]
    return AM_BP

def Z_Projection(IM):
    IM_Max = np.zeros((len(IM),len(IM[0])))
    for i in range(len(IM)):
        for j in range(len(IM[0])):
            IM_Max[i,j] = np.amax(IM[i,j,:])
    return IM_Max

def show3DImage(IM):
    IMmax = Z_Projection(IM)
    plt.imshow(IMmax,[0,50])
    plt.show()


def get_data(IM_path, label_path):
    IM = io.imread(IM_path).astype(float)
    IM = np.einsum('kij->ijk', IM)
    IM = (IM / 255)
    label = io.imread(label_path).astype(float)
    label = np.einsum('kij->ijk', label)
    label = ((label == 255) * 0.5 + (label != 0) * 0.5)
    return IM, label





def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_ind_IMlist(data_dir, N_images, out_x, out_y, out_z, pad_x, pad_y, pad_z):
    IM_name_list = []
    label_name_list = []
    for i in range(N_images):
        IM_name_list.append('image_' + str(i + 1) + '.tif')
        label_name_list.append('label_' + str(i + 1) + '.tif')
    Index_matrix = np.array([], dtype=int).reshape(0, 4)
    train_IM_list = IM_name_list
    train_label_list = label_name_list
    for i in range(len(IM_name_list)):
        phantom_IM, phantom_label = get_data(data_dir + IM_name_list[i], data_dir + label_name_list[i])
        x_N = (phantom_IM.shape[0] - out_x + 1)
        y_N = (phantom_IM.shape[1] - out_y + 1)
        z_N = (phantom_IM.shape[2] - out_z + 1)
        temp_Index_matrix = np.zeros([x_N * y_N * z_N, 4], dtype=int)
        temp_Index_matrix[:, 0] = i
        temp_Index_matrix[:, 1] = np.repeat(range(x_N), y_N * z_N)
        temp_Index_matrix[:, 2] = np.tile(np.repeat(range(y_N), z_N), x_N)
        temp_Index_matrix[:, 3] = np.tile(range(z_N), x_N * y_N)
        Index_matrix = np.concatenate((Index_matrix, temp_Index_matrix), axis=0)
        phantom_IM = np.pad(phantom_IM, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant', constant_values=0)
        phantom_label = np.pad(phantom_label, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant',
                               constant_values=0)
        train_IM_list[i] = phantom_IM
        train_label_list[i] = phantom_label
    return Index_matrix, train_IM_list, train_label_list


def get_validation_set(data_dir, valid_IM_name, valid_label_name):
    valid_IM, valid_label = get_data(data_dir + valid_IM_name, data_dir + valid_label_name)
    valid_IM = valid_IM[356:484, 208:336, 21:37]
    valid_label = valid_label[356:484, 208:336, 21:37]

    return valid_IM, valid_label