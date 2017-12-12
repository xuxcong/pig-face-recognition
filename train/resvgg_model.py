from __future__ import print_function
import tensorflow as tf
import numpy as np
#from scipy.misc import imread, imresize
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle
import pickle 
from tflearn.data_utils import image_preloader
import h5py
import math
import random
import time
import csv

class resvgg:
    def __init__(self, imgs,keep_prob, weights=None, sess=None, res = False, finetune = True):
        self.finetune = finetune;           ## only to train the last fc layer
        self.keep_prob = keep_prob;
        self.imgs = imgs
        self.res = res;                     ## use the resnet structure
        self.last_layer_parameters = []     ## Parameters in this list will be optimized when only last layer is being trained 
        self.parameters = []                ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()                   ## Create Convolutional layers
        self.fc_layers()                    ## Create Fully connected layer
        self.weight_file = weights    
            


    def convlayers(self):
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
            print('Adding Data Augmentation')


        # conv1_1
        with tf.variable_scope("conv1_1"):
            weights = tf.get_variable("W", [3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv1_2
        with tf.variable_scope("conv1_2"):
            weights = tf.get_variable("W", [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv1_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            combine = conv + biases;
            if(self.res):
                combine = combine + self.conv1_1;
            self.conv1_2 = tf.nn.relu( combine )
            self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope("conv2_1"):
            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]



        # conv2_2
        with tf.variable_scope("conv2_2"):
            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv2_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            combine = conv + biases;
            if(self.res):
                combine = combine + self.conv2_1;
            self.conv2_2 = tf.nn.relu(combine)
            self.parameters += [weights, biases]


        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope("conv3_1"):
            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv3_2
        with tf.variable_scope("conv3_2"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv3_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv3_3
        with tf.variable_scope("conv3_3"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv3_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            combine = conv + biases;
            if(self.res):
                combine = combine + self.conv3_1;
            self.conv3_3 = tf.nn.relu(combine)
            self.parameters += [weights, biases]


        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope("conv4_1"):
            weights = tf.get_variable("W", [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.pool3, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_2
        with tf.variable_scope("conv4_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv4_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_3
        with tf.variable_scope("conv4_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv4_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            combine = conv + biases;
            if(self.res):
                combine = combine + self.conv4_1;
            self.conv4_3 = tf.nn.relu(combine)
            self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope("conv5_1"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.pool4, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv5_2
        with tf.variable_scope("conv5_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv5_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
            

        # conv5_3
        with tf.variable_scope("conv5_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=self.finetune)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=self.finetune)
            conv = tf.nn.conv2d(self.conv5_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            combine = conv + biases;
            if(self.res):
                combine = combine + self.conv5_1;
            self.conv5_3 = tf.nn.relu(conv + biases + self.conv5_1)
            self.parameters += [weights, biases]
            self.special_parameters = [weights,biases]


        self.z_l2 = self.get_bilinear_fc(self.conv5_3, self.conv5_3)
        # print('conv5_3  ',self.conv5_3.get_shape())
        # print('self.conv5_1  ',self.conv5_1.get_shape())
        self.z_l3 = self.get_bilinear_fc(self.conv5_3, self.conv5_1)
        # print('self.conv5_1  ',self.conv5_1.get_shape())
        pool4_1 = tf.nn.max_pool(self.conv4_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        self.z_l4 = self.get_bilinear_fc(self.conv5_3, pool4_1)

        self.final_z = tf.concat([self.z_l2 ,self.z_l3, self.z_l4],1)
        print(self.final_z.get_shape())

    def get_bilinear_fc(self,conv1, conv2):
        conv1 = tf.transpose(conv1, perm=[0,3,1,2])       
        conv1 = tf.reshape(conv1,[-1,512,784])            
        conv2 = tf.transpose(conv2, perm=[0,3,1,2])       
        conv2 = tf.reshape(conv2,[-1,512,784])                                                                          
        conv2 = tf.transpose(conv2, perm=[0,2,1])            
        phi_I = tf.matmul(conv1, conv2)                 
        phi_I = tf.reshape(phi_I,[-1,512*512])                
        phi_I = tf.divide(phi_I,784.0)  
        y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))       
        z = tf.nn.l2_normalize(y_ssqrt, dim=1)     
        print('Shape of z', z.get_shape())
        return z



    def fc_layers(self):


        with tf.variable_scope('fc-new') as scope:
            fc3w = tf.get_variable('W', [786432, 30], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            #fc3b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), name='biases', trainable=True)
            fc3b = tf.get_variable("b", [30], initializer=tf.constant_initializer(0.1), trainable=True)
            fc = tf.nn.bias_add(tf.matmul(self.final_z, fc3w), fc3b)
            self.fc3l = tf.nn.dropout(fc, self.keep_prob)
            self.last_layer_parameters += [fc3w, fc3b]
            self.parameters += [fc3w, fc3b]

    def load_initial_weights(self, session):
        weights_dict = np.load(self.weight_file, encoding = 'bytes')
        vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
        
        for op_name in vgg_layers:
            with tf.variable_scope(op_name, reuse = True):
                
              # Loop over list of weights/biases and assign them to their corresponding tf variable
                # Biases
              
              var = tf.get_variable('b', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_b']))
                  
            # Weights
              var = tf.get_variable('W', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_W']))




    def load_own_weight(self,session, filename):
        i = 0;
        weights_dict = np.load(filename, encoding = 'bytes')
        '''Loop over all layer names stored in the weights dict
           Load only conv-layers. Skip fc-layers in VGG16'''
        vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
        
        for op_name in vgg_layers:
            with tf.variable_scope(op_name, reuse = True):
              # Weights
              var = tf.get_variable('W', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict['arr_0' ][i]))

              var = tf.get_variable('b', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict['arr_0' ][i+1]))
              i = i + 2;



        with tf.variable_scope('fc-new', reuse = True):
            '''
            Load fc-layer weights trained in the first step. 
            Use file .py to train last layer
            '''
            print('Last layer weights: last_layers_epoch_best.npz')
            var = tf.get_variable('W', trainable = True)
            print('Adding weights to',var.name)
            session.run(var.assign(weights_dict['arr_0' ][i]))
            var = tf.get_variable('b', trainable = True)
            print('Adding weights to',var.name)
            session.run(var.assign(weights_dict['arr_0'][i+1]))
            i = i + 2;