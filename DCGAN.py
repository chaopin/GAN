#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : DCGAN.py
# @Author: Bin Zhao
# @Date  : 17-12-20
# @Desc  :
#test for github
import tensorflow as tf
import loadData
import numpy as np
import scipy.misc
import os
import time
class DCGAN:
    def __init__(self,X,Y,batchNum):
        cal = time.strftime('%Y%m%d-%H%m', time.localtime(time.time()))
        self.path=os.path.join("sample",cal)
        if os.path.exists(self.path) is False:
            os.mkdir(self.path)
        print("path is :",self.path)
        self.zDim = 100
        self.X=X
        self.Y=tf.one_hot(Y,depth=10,dtype=tf.float32)
        self.batchNum=batchNum
        self.Z=tf.placeholder(tf.float32,[X.shape[0],self.zDim])

        self.samplerNum=5
        self.samZ = tf.placeholder(tf.float32, [self.samplerNum, self.zDim])
        self.samY=tf.placeholder(tf.int32,[self.samplerNum])
        self.samYHot=tf.one_hot(self.samY,depth=10,dtype=tf.float32)

        self.leakyAlpha=0.2
        self.Channel=1
        self.stddev=0.02
        # the generator
        self.G = self.generator(self.Z, self.Y)
        # the discriminator
        self.DG, self.DGLogits = self.discriminator(self.G, self.Y)
        self.D, self.DLogits = self.discriminator(self.X, self.Y,reuse=True)
        allTraVar = tf.trainable_variables()
        # seperate the trainable variables
        self.genVar = [var for var in allTraVar if "gene" in var.name]
        self.discVar = [var for var in allTraVar if "disc" in var.name]

        # the discrimative model
        self.DRealLoss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D),
                                                           logits=self.DLogits)
        self.DFakeLoss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.DG),
                                                               logits=self.DGLogits)
        print("self.DG.shape",self.DG.shape)
        self.DLose=tf.reduce_sum(self.DRealLoss)+tf.reduce_sum(self.DFakeLoss)
        self.dTrain = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.5).minimize(self.DLose, var_list=self.discVar)
        # the generative model
        self.GLosetemp = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.DG),
                                                                 logits=self.DGLogits)
        self.GLose = tf.reduce_sum(self.GLosetemp)
        self.gTrain = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(loss=self.GLose, var_list=self.genVar)

        self.epoch=100
        self.writer = tf.summary.FileWriter('log')
        self.writer.add_graph(tf.get_default_graph())
        self.GSum=tf.summary.scalar("GLOSE",self.GLose)
        self.DSum = tf.summary.scalar("DLOSE",self.DLose)


        self.samp=self.sampler(self.samZ,self.samYHot)
        self.train()

        self.writer.close()

    def generator(self,z,y=None):
        with tf.variable_scope("gene") as scope:
            zConcat=tf.concat([z,y],axis=1)
            shapeZ=zConcat.get_shape().as_list()
            yShape=y.get_shape().as_list()
            labelReshape=tf.reshape(y,shape=[yShape[0],1,1,yShape[1]])
            with tf.variable_scope("proj"):
                proShape=1024
                weights=tf.get_variable("projW",shape=[shapeZ[1],proShape],
                                        initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias=tf.get_variable("projB",shape=[proShape],
                                     initializer=tf.zeros_initializer())
                proOut=tf.matmul(zConcat,weights)+bias
                proOut=tf.layers.batch_normalization(proOut,training=True)
                proOut=tf.nn.relu(proOut)

                print("proOut.shape", proOut.shape)
            with tf.variable_scope("reshape"):
                resConcat=tf.concat([proOut,y],axis=1)
                resconShape=resConcat.shape
                resShape=7*7*128
                weights=tf.get_variable("projW",shape=[resconShape[1],resShape],
                                        initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias=tf.get_variable("projB",shape=resShape,
                                     initializer=tf.zeros_initializer())
                resOut=tf.matmul(resConcat,weights)+bias
                resOut=tf.reshape(resOut,shape=[-1,7,7,128])
                # BN
                resOut=tf.layers.batch_normalization(resOut,training=True)

                labelTilereshape=tf.tile(labelReshape,[1,7,7,1])
                resconOut=tf.concat([resOut,labelTilereshape],axis=3)
                print("resconOut.shape", resconOut.shape)

            #transpose convolution
            with tf.variable_scope("dconv1"):
                filtNum=128
                filtShape=5
                sizeWin=14
                resconOutShape=resconOut.get_shape().as_list()
                filtDconv=tf.get_variable("dconv1F",shape=[filtShape,filtShape,filtNum,resconOutShape[-1]],
                                          initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                shapeDconv=[resconOutShape[0],sizeWin,sizeWin,filtNum]

                strideDconv=[1,2,2,1]
                dconv1Out=tf.nn.conv2d_transpose(resconOut,filtDconv,shapeDconv,strideDconv,padding="SAME")

                # dconv1Out = tf.nn.conv2d_transpose(resconOut, filtDconv, sizeWin, strideDconv)
                dconv1Out=tf.layers.batch_normalization(dconv1Out,training=True)
                dconv1Out=tf.nn.relu(dconv1Out)
                print(dconv1Out.get_shape().as_list())

                # concat the data
                deconvShape=dconv1Out.get_shape().as_list()
                labelTiledconv1=tf.tile(labelReshape,[1,deconvShape[1],deconvShape[2],1])
                dconv1conOut=tf.concat([dconv1Out,labelTiledconv1],axis=3)
                dconv1Shape=dconv1conOut.get_shape().as_list()
                print("dconv1Out.shape",dconv1conOut.shape)

            with tf.variable_scope("dconv2"):
                filtNum = self.Channel
                filtShape = 5
                imgShape = 28
                filtDconv = tf.get_variable("dconv2F", shape=[filtShape, filtShape, filtNum, dconv1Shape[3]],
                                            initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                shapeDconv = [dconv1Shape[0], imgShape, imgShape, filtNum]
                strideDconv = [1, 2, 2, 1]
                dconv2Out = tf.nn.conv2d_transpose(dconv1conOut, filtDconv, shapeDconv, strideDconv)

                print("dconv2Out.shape", dconv2Out.shape)
                # imgOut=tf.nn.tanh(dconv2Out)
                imgOut=tf.nn.sigmoid(dconv2Out)
            return imgOut

    def lrelu(self,x):
        return tf.maximum(self.leakyAlpha*x,x);

    def conv2d(self,image,outChannel,kernalShape,stride):
        w=tf.get_variable("weights",[kernalShape,
                                     kernalShape,image.get_shape()[-1],outChannel],
                          initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        b=tf.get_variable("basis",[outChannel],
                          initializer=tf.zeros_initializer())
        Stride=[1,stride,stride,1]
        out=tf.nn.conv2d(image,w,Stride,padding='SAME')+b
        return out

    def discriminator(self,image,y=None,reuse=False):
        print("discriminator")
        with tf.variable_scope("disc") as scope:
            if reuse:
                scope.reuse_variables()
            imageShape=image.get_shape().as_list()

            yShape = y.get_shape().as_list()
            yReshape = tf.reshape(y, shape=[yShape[0], 1, 1, yShape[1]])
            yTile = tf.tile(yReshape, [1, imageShape[1], imageShape[2], 1])
            imgCom = tf.concat([image, yTile], axis=3)
            print("imgCom.zise",imgCom.shape)
            with tf.variable_scope("conv1Dis"):
                imgComShape=imgCom.get_shape().as_list()
                imgConv1=self.conv2d(image,imgComShape[3],5,2)
                imgConv1=self.lrelu(imgConv1)

                imgConv1Shape = imgConv1.get_shape().as_list()
                yTileConv1=tf.tile(yReshape, [1, imgConv1Shape[1], imgConv1Shape[2], 1])
                imgConv1Con=tf.concat([imgConv1,yTileConv1],axis=3)
                print("imgConv1Con.get_shape() :",imgConv1Con.get_shape())
            with tf.variable_scope("conv2Dis"):
                outConv2Channel=yShape[-1]+64

                imgConv2 = self.conv2d(imgConv1Con, outConv2Channel, 5, 2)
                imgConv2=tf.layers.batch_normalization(imgConv2,training=True)
                imgConv2 = self.lrelu(imgConv2)

                imgConv2=tf.reshape(imgConv2,[imageShape[0],-1])
                imgConv2Con=tf.concat([imgConv2,y],axis=1)
                print("imgConv2.shape() ",imgConv2.get_shape())
                print("imgConv2Con.shape() ", imgConv2Con.get_shape())
            with tf.variable_scope("deshape"):
                imgConv2ConShape=imgConv2Con.get_shape().as_list()
                deShape=1024
                weights = tf.get_variable("deshapeW",
                                          shape=[imgConv2ConShape[1], deShape],
                                          initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias = tf.get_variable("deshapeB", shape=[deShape],
                                       initializer=tf.zeros_initializer())
                deShapeOut=tf.matmul(imgConv2Con,weights)+bias
                deShapeOut=tf.layers.batch_normalization(deShapeOut,training=True)
                deShapeOut=self.lrelu(deShapeOut)
                deShapeOutCon=tf.concat([deShapeOut,y],axis=1)
                print("deShapeOutCon.shape() ", deShapeOutCon.get_shape())

            with tf.variable_scope("deproj"):
                deprojShape=1
                shapeDeout=deShapeOutCon.get_shape().as_list()
                weights = tf.get_variable("projW",
                                          shape=[shapeDeout[1], deprojShape],
                                          initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias = tf.get_variable("projB", shape=[deprojShape],
                                       initializer=tf.zeros_initializer())
                deprojOut = tf.matmul(deShapeOutCon, weights) + bias
                deprojOutSig=tf.nn.sigmoid(deprojOut)
                print("deprojOut.shape() ", deprojOut.get_shape())
        return deprojOut,deprojOutSig

    def sampler(self,z,y):
        with tf.variable_scope("gene") as scope:
            scope.reuse_variables()
            zConcat=tf.concat([z,y],axis=1)
            shapeZ=zConcat.get_shape().as_list()
            yShape=y.get_shape().as_list()
            labelReshape=tf.reshape(y,shape=[yShape[0],1,1,yShape[1]])
            with tf.variable_scope("proj"):
                proShape=1024
                weights=tf.get_variable("projW",shape=[shapeZ[1],proShape],
                                        initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias=tf.get_variable("projB",shape=[proShape],
                                     initializer=tf.zeros_initializer())
                proOut=tf.matmul(zConcat,weights)+bias
                proOut=tf.layers.batch_normalization(proOut,training=True)
                # leaky relu:
                proOut=tf.nn.relu(proOut)

                print("proOut.shape", proOut.shape)
            with tf.variable_scope("reshape"):
                resConcat=tf.concat([proOut,y],axis=1)
                resconShape=resConcat.shape
                resShape=7*7*128
                weights=tf.get_variable("projW",shape=[resconShape[1],resShape],
                                        initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                bias=tf.get_variable("projB",shape=resShape,
                                     initializer=tf.zeros_initializer())
                resOut=tf.matmul(resConcat,weights)+bias
                resOut=tf.reshape(resOut,shape=[-1,7,7,128])
                # BN
                resOut=tf.layers.batch_normalization(resOut,training=True)

                labelTilereshape=tf.tile(labelReshape,[1,7,7,1])
                resconOut=tf.concat([resOut,labelTilereshape],axis=3)
                print("resconOut.shape", resconOut.shape)

            #transpose convolution
            with tf.variable_scope("dconv1"):
                filtNum=128
                filtShape=5
                sizeWin=14
                resconOutShape=resconOut.get_shape().as_list()
                filtDconv=tf.get_variable("dconv1F",shape=[filtShape,filtShape,filtNum,resconOutShape[-1]],
                                          initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                shapeDconv=[resconOutShape[0],sizeWin,sizeWin,filtNum]

                strideDconv=[1,2,2,1]
                dconv1Out=tf.nn.conv2d_transpose(resconOut,filtDconv,shapeDconv,strideDconv,padding="SAME")

                # dconv1Out = tf.nn.conv2d_transpose(resconOut, filtDconv, sizeWin, strideDconv)
                dconv1Out=tf.layers.batch_normalization(dconv1Out,training=True)
                dconv1Out=tf.nn.relu(dconv1Out)
                print(dconv1Out.get_shape().as_list())

                # concat the data
                deconvShape=dconv1Out.get_shape().as_list()
                labelTiledconv1=tf.tile(labelReshape,[1,deconvShape[1],deconvShape[2],1])
                dconv1conOut=tf.concat([dconv1Out,labelTiledconv1],axis=3)
                dconv1Shape=dconv1conOut.get_shape().as_list()
                print("dconv1Out.shape",dconv1conOut.shape)

            with tf.variable_scope("dconv2"):
                filtNum = self.Channel
                filtShape = 5
                imgShape = 28
                filtDconv = tf.get_variable("dconv2F", shape=[filtShape, filtShape, filtNum, dconv1Shape[3]],
                                            initializer=tf.truncated_normal_initializer(stddev=self.stddev))
                shapeDconv = [dconv1Shape[0], imgShape, imgShape, filtNum]
                strideDconv = [1, 2, 2, 1]
                dconv2Out = tf.nn.conv2d_transpose(dconv1conOut, filtDconv, shapeDconv, strideDconv)

                print("dconv2Out.shape", dconv2Out.shape)
                imgOut=tf.nn.sigmoid(dconv2Out)
                # imgOut = tf.nn.tanh(dconv2Out)
                print("imgOut.shape", dconv2Out.shape)
            # imgOut=tf.nn.sigmoid(dconv2Out)
        # return dconv2Out
            return imgOut
    def visual(self,sess,epoch,batch):
        print("visualing")
        batchZ = np.random.uniform(-1, 1,self.samZ.get_shape().as_list()).astype(np.float32)
        batchY=np.random.randint(0,10,self.samY.get_shape().as_list()).astype(np.int32)

        images=sess.run(self.samp,feed_dict={self.samZ:batchZ,self.samY:batchY})
        images=np.squeeze(images,axis=3)
        for img in range(images.shape[0]):
            path='%d_%d_%d_%d.png'%(epoch,batch,img,batchY[img])
            path=os.path.join(self.path,path)
            scipy.misc.imsave(path,images[img])
        print("visualing end")


    def train(self):
        config=tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.operation_timeout_in_ms=50000
        merged = tf.summary.merge_all()
        with tf.Session(config=config) as sess:
            # sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(self.epoch):
                print(epoch,"epoch")
                for indBatch in range(int(self.batchNum)):

                    if indBatch%100==1:
                        print(indBatch, "indBatch")
                        self.visual(sess,epoch,indBatch)
                    batchZ = np.random.uniform(-1, 1,
                                               self.Z.get_shape().as_list()).astype(np.float32)
                    _,Dsum=sess.run([self.dTrain,self.DSum], feed_dict={self.Z: batchZ})
                    sess.run(self.gTrain, feed_dict={self.Z: batchZ})
                    _,Gsum=sess.run([self.gTrain,self.GSum], feed_dict={self.Z: batchZ})

                    self.writer.add_summary(Gsum,indBatch+epoch*self.batchNum)
                    self.writer.add_summary(Dsum,indBatch+epoch*self.batchNum)
            coord.request_stop()
            coord.join(threads)


X,Y,numBatch=loadData.get_batch_data('MNIST',100,4)
print("numBatch ",numBatch)

temp = DCGAN(X, Y,numBatch)
    # temp.generator(z, y)
    # temp.train()
    # temp.discriminator(temp.generator(z,y),y)
