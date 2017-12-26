#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : loadData.py
# @Author: Bin Zhao
# @Date  : 17-12-20
# @Desc  :
import os
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
def loadMnist(batchSize,isTraining=True,isNorm=True):
    path=os.path.join('data','mnist')
    if isTraining:
        trainDataFile=os.path.join(path,'train-images-idx3-ubyte')
        trainDataLabel=os.path.join(path,'train-labels-idx1-ubyte')

        fdTrain=open(trainDataFile)
        loadedX=np.fromfile(fdTrain,dtype=np.uint8)
        numTtl=60000
        X=loadedX[16:].reshape((numTtl,28,28,1)).astype(np.float32)
        fdTrain.close()

        fdLabel=open(trainDataLabel)
        loadedY=np.fromfile(fdLabel,dtype=np.int8)
        Y=loadedY[8:].reshape((numTtl)).astype(np.int32)
        fdLabel.close()

        numTra=55000
        traX=X[:numTra]
        traY=Y[:numTra]
        numBtcTra=numTra/batchSize

        # for i in range(9):
        #     imageTemp=X[i,:,:,0]
        #     print(imageTemp.shape)
        #     print(Y[i])
        #     plt.imshow(imageTemp)
        #     plt.show()
        valX=X[numTra:,]
        valY=Y[numTra:]
        numBtcVal = (numTtl - numTra) / batchSize
        if isNorm:
            traX=traX/255
            valX=valX/255
        return traX,traY,numBtcTra,valX,valY,numBtcVal
    else:
        dataFile = os.path.join(path, 't10k-images-idx3-ubyte')
        dataLabel = os.path.join(path, 't10k-labels-idx1-ubyte')

        numTtl = 10000
        fdX=open(dataFile)
        loadedX=np.fromfile(fdX,dtype=np.int8)
        X=loadedX[16:].reshape((numTtl,28,28,1)).astype(np.float32)
        fdX.close()

        fdLabel=open(dataLabel)
        loadedY=np.fromfile(fdLabel,dtype=np.int8)
        Y=loadedY[8:].reshape((numTtl,28,28,1)).astype(np.int32)
        fdLabel.close()

        numBtc=numTtl/batchSize
        if isNorm:
            X=X/255
        return X,Y,numBtc


def get_batch_data(dataSet,batchSize,numThreads):
    if dataSet=="MNIST":
        X,Y,numBatch,valX,valY,numValBatch=loadMnist(batchSize)
    dataQueue=tf.train.slice_input_producer([X,Y])
    traX,traY=tf.train.shuffle_batch(dataQueue,batch_size=batchSize,capacity=batchSize*256,
                                     min_after_dequeue=batchSize*128,
                                     num_threads=numThreads)
    return traX,traY,numBatch


