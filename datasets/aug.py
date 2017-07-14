#!/usr/bin/python
#########################################################################
# File Name: aug.py
# Description: 
# Author: Jialiang Zhao
# Mail: alanzjl@163.com
# Created_Time: 2017-05-28 21:30:22
# Last modified: 2017-05-28 21:30:1495978222
#########################################################################

import numpy as np
import sys

name = sys.argv[1]
value = int(sys.argv[2])
data = np.load('all.npy')
if value == 1: # positive
    posdata = open(name)
    lines = posdata.readlines()

    num = 0
    newdata = []
    print (data.shape)

    for line in lines:
        line = line.strip('\n')
        if len(line) < 3*20:
            newdata.append([0, line, 1])
            num += 1
    data = np.concatenate((np.asarray(newdata), data), axis = 0)
    print ('new pos: %d'%num)
    print (data.shape[0])

    posdata.close()
elif value==2: #neg

    negdata = open(name)
    lines = negdata.readlines()

    num = 0
    newdata = []
    print (data.shape)

    for line in lines:
        line = line.strip('\n')
        if len(line) < 3*20:
            newdata.append([0, line, 3])
            num += 1
    data = np.concatenate((np.asarray(newdata), data), axis = 0)
    print ('new neg: %d'%num)
    print (data.shape[0])

    negdata.close()

np.save('all.npy', data)

