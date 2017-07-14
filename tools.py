#!/usr/bin/python
#########################################################################
# File Name: get_EMOWEB_data.py
# Description: 
# Author: Jialiang Zhao
# Mail: alanzjl@163.com
# Created_Time: 2017-05-15 21:51:25
# Last modified: 2017-05-15 21:51:1494856285
#########################################################################

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def run_model(session, predict, loss_val, Xd, yd, drop1, drop2,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         keep_prob1: drop1,
                         keep_prob2: drop2,
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

def EmoNet(X,y,keep_prob1,keep_prob2,is_training):
    
    # setup variables
    
    with tf.name_scope("InputFlat") as name_scope:
        inflat = tf.reshape(X,[-1,144*3]) # 144*3
    
    with tf.name_scope("Affine-1") as name_scope:
        Wauger1 = tf.get_variable("Wauger1", regularizer=tf.contrib.layers.l2_regularizer(1.0), 
                                  shape=[144*3, 32*32*3])
        bauger1 = tf.get_variable("bauger1", shape=[1024*3])
        auger1 = tf.matmul(inflat,Wauger1) + bauger1
    
    with tf.name_scope("Augment-1") as name_scope:
        # Augment data from 144*3 to 32*32*3
        aug1 = tf.reshape(auger1,[-1,32,32,3])
        
    with tf.name_scope("ConvRELU1") as name_scope:  
        # Conv 32, 5x5 stride 1
        # before: ? x 32 x 32 x 3
        # after: ? x 28 x 28 x 32
        Wconv1_1 = tf.get_variable("Wconv1_1", shape=[5, 5, 3, 32])
        bconv1_1 = tf.get_variable("bconv1_1", shape=[32])
        conv1_1 = tf.nn.conv2d(aug1, Wconv1_1, strides=[1,1,1,1], padding='VALID') + bconv1_1
        # RELU
        relu1_1 = tf.nn.relu(conv1_1)
        
    with tf.name_scope("Maxpool1") as name_scope:  
        # Maxpool 2x2, stride 1
        # before: ? x 28 x 28 x 32
        # after: ? x 28 x 28 x 32
        maxpool1 = tf.nn.max_pool(relu1_1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        
    with tf.name_scope("ConvRELU2") as name_scope:  
        # Conv 64, 5x5 stride 1
        # before: ? x 28 x 28 x 32
        # after: ? x 24 x 24 x 64
        Wconv2_1 = tf.get_variable("Wconv2_1", shape=[5, 5, 32, 64])
        bconv2_1 = tf.get_variable("bconv2_1", shape=[64])
        conv2_1 = tf.nn.conv2d(maxpool1, Wconv2_1, strides=[1,1,1,1], padding='VALID') + bconv2_1
        # RELU
        relu2_1 = tf.nn.relu(conv2_1)
        # Conv 64, 5x5 stride 1
        # before: ? x 24 x 24 x 64
        # after: ? x 20 x 20 x 64
        Wconv2_2 = tf.get_variable("Wconv2_2", shape=[5, 5, 64, 64])
        bconv2_2 = tf.get_variable("bconv2_2", shape=[64])
        conv2_2 = tf.nn.conv2d(relu2_1, Wconv2_2, strides=[1,1,1,1], padding='VALID') + bconv2_2
        # RELU
        relu2_2 = tf.nn.relu(conv2_2)
        
    with tf.name_scope("Maxpool2") as name_scope:  
        # Maxpool 2x2, stride 1
        # before: ? x 20 x 20 x 64
        # after: ? x 20 x 20 x 64
        maxpool2 = tf.nn.max_pool(relu2_2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        
    with tf.name_scope("ConvRELU3") as name_scope:  
        # Conv 128, 5x5 stride 1
        # before: ? x 20 x 20 x 64
        # after: ? x 16 x 16 x 128
        Wconv3_1 = tf.get_variable("Wconv3_1", shape=[5, 5, 64, 128])
        bconv3_1 = tf.get_variable("bconv3_1", shape=[128])
        conv3_1 = tf.nn.conv2d(maxpool2, Wconv3_1, strides=[1,1,1,1], padding='VALID') + bconv3_1
        # RELU
        relu3_1 = tf.nn.relu(conv3_1)
        
    with tf.name_scope("Maxpool3") as name_scope:  
        # Maxpool 2x2, stride 1
        # before: ? x 16 x 16 x 128
        # after: ? x 16 x 16 x 128
        maxpool3 = tf.nn.max_pool(relu3_1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        
    with tf.name_scope("ConvRELU4") as name_scope:  
        # Conv 246, 5x5 stride 1
        # before: ? x 16 x 16 x 128
        # after: ? x 12 x 12 x 256
        Wconv4_1 = tf.get_variable("Wconv4_1", shape=[5, 5, 128, 256])
        bconv4_1 = tf.get_variable("bconv4_1", shape=[256])
        conv4_1 = tf.nn.conv2d(maxpool3, Wconv4_1, strides=[1,1,1,1], padding='VALID') + bconv4_1
        # RELU
        relu4_1 = tf.nn.relu(conv4_1)
        
    with tf.name_scope("Maxpool4") as name_scope:  
        # Maxpool 2x2, stride 2, VALID padding
        # before: ? x 12 x 12 x 256
        # after: ? x 6 x 6 x 256
        maxpool4 = tf.nn.max_pool(relu4_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    with tf.name_scope("Flat") as name_scope:
        flat = tf.reshape(maxpool4,[-1,9216]) # 6*6*256
        
    with tf.name_scope("AffineRELUDrop1") as name_scope:
        W1 = tf.get_variable("W1", regularizer=tf.contrib.layers.l2_regularizer(1.0), 
                             shape=[9216, 1024])
        b1 = tf.get_variable("b1", shape=[1024])
        affine1 = tf.matmul(flat,W1) + b1
        
        relu1 = tf.nn.relu(affine1)

        drop1 = tf.nn.dropout(relu1, keep_prob1)
        
    with tf.name_scope("AffineRELUDrop2") as name_scope:
        W2 = tf.get_variable("W2", regularizer=tf.contrib.layers.l2_regularizer(1.0), 
                             shape=[1024, 1024])
        b2 = tf.get_variable("b2", shape=[1024])
        affine2 = tf.matmul(drop1,W2) + b2
        
        relu2 = tf.nn.relu(affine2)
        
        drop2 = tf.nn.dropout(relu2, keep_prob2)
    
    with tf.name_scope("Affine3") as name_scope:
        W3 = tf.get_variable("W3", regularizer=tf.contrib.layers.l2_regularizer(1.0), 
                             shape=[1024, 5])
        b3 = tf.get_variable("b3", shape=[5])
        affine3 = tf.matmul(drop2,W3) + b3
    
    y_out = affine3
    
    return y_out

