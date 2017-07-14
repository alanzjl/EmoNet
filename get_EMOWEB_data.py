#!/usr/bin/python
#########################################################################
# File Name: get_EMOWEB_data.py
# Description: 
# Author: Jialiang Zhao
# Mail: alanzjl@163.com
# Created_Time: 2017-05-15 21:51:25
# Last modified: 2017-05-15 21:51:1494856285
#########################################################################
import numpy as np

def prepocess_data(X, with_context = True):
    num_data = []
    for i in X:
        num_s = []
        for cha in i:                   # 3 ascii codes make up one utf 8 code
            num_s.append(int(cha))  # ascii code for every character
        num_data.append(num_s)

    X_res = np.zeros([len(num_data), 144, 3], dtype=float)

    for cnt in range(len(num_data)):
        data_cur = num_data[cnt]
        xe = np.zeros([3*144], dtype=float)
        tmp = 0
        for i in data_cur:              # current dialog
            xe[tmp] = i
            tmp += 1
            if tmp >= 144*3: break
        
        xe.reshape((3,144))
        X_res[cnt] = xe.reshape(144,3).copy()
    return X_res

def prepocess_label(ys):
    y = np.asarray(ys)
    for i in range(len(ys)):
        if y[i] == -1:
            y[i] = 4
    return y

def trans_back(X):
    str_ori = []
    this_str = X.reshape((-1))
    no_zero = this_str[this_str!=0]
    for i in no_zero:
        str_ori.append(chr(int(i)))
    return ''.join(str_ori)

def get_EMOWEB_data(num_training=4800, num_validation=200, num_test=10, with_context=True, pure=False):
    data_dir = ''
    if not pure:
        data_dir = 'datasets/all.npy'
    else:
        data_dir = 'datasets/all-pure.npy'
    data_all = np.load(data_dir)

    xs = []
    ys = []

    #np.random.shuffle(data_all)

    for data in data_all:
        xs.append(data[1])
        ys.append(int(data[2]))

    y = prepocess_label(ys)
    X = prepocess_data(xs, with_context)

    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training + num_validation,num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


