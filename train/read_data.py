from __future__ import print_function
import numpy as np
import tflearn
import os
from tflearn.data_utils import shuffle
import pickle 
import h5py
import math
import random
import time
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
import csv
from keras.preprocessing import image
import PIL
#rotate the image 
def rotate(x, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotate_limit=(-90, 90)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

# change brightness  Color  contrast  sharpness
def random_brightness(img, delta):
    img = PIL.Image.fromarray(np.uint8(img))
    enh_bri = ImageEnhance.Brightness(img)  
    brightness = np.random.randint(8,14) / 10.0;
    image_brightened = enh_bri.enhance(brightness);

    #color
    enh_col = ImageEnhance.Color(image_brightened)  
    color = np.random.randint(8,14) / 10.0;
    image_colored = enh_col.enhance(color)  

    enh_con = ImageEnhance.Contrast(image_colored)  
    contrast =np.random.randint(8,14) / 10.0;
    image_contrasted = enh_con.enhance(contrast) 

    enh_sha = ImageEnhance.Sharpness(image_contrasted)  
    sharpness = contrast =np.random.randint(8,15) / 10.0;
    image_sharped = enh_sha.enhance(sharpness) 

    return np.asarray(image_sharped)


def self_random_crop(image_batch):
    result = []
    for n in range(image_batch.shape[0]):
        newimg = random_brightness(image_batch[n], 0.8);
        newimg = rotate(newimg);
        start_x = random.randint(0,39)
        start_y = random.randint(0,39)
        newimg = newimg[start_y:start_y+448,start_x:start_x+448,:];
        result.append(newimg)
    return result


def move_zero_label(y, len1, len2):
    y_result = [];
    for i in range(len1):
        ytem = y[i];
        y_result.append(ytem[1:len2]);
    return np.array(y_result);

class data_reader:
    def __init__(self, datasets, numclass, batchsize, shuffle = True):
    	self.shuffle = shuffle;
        self.num_class = numclass;
        self.dataset = datasets;
        self.file_num = len(datasets);
        self.now_read_file_pos = 0;
        self.batch_size = batchsize;
        self.data = None;
        self.X_data = None;
        self.Y_data = None;
        self.datanum = 0;
        self.batch_num = 0;
        self.tem_batch_pos = 0;
        self.total_datanum = 0;
        self.nextfile();

    def new_iterator(self):
        self.now_read_file_pos = 0;
        self.total_datanum = 0;
        self.nextfile();

    def nextfile(self):
        if(self.now_read_file_pos + 1 <= self.file_num):
            self.data = h5py.File(self.dataset[self.now_read_file_pos], 'r')
            self.X_data = self.data['X'];
            self.Y_data = self.data['Y'];
            self.datanum = self.X_data.shape[0];
            self.total_datanum = self.total_datanum + self.datanum;
            self.Y_data = move_zero_label(self.Y_data, self.Y_data.shape[0], self.Y_data.shape[1]);
            if(self.shuffle):
            	self.X_data, self.Y_data = shuffle(self.X_data, self.Y_data)
            self.batch_num = int(self.datanum / self.batch_size);
            self.tem_batch_pos = 0;
            print('Read data file: ' + self.dataset[self.now_read_file_pos]);
            self.now_read_file_pos = self.now_read_file_pos + 1;
            return True;
        else:
            return False;


    def next_batch(self, process = False):
        batch_xs = self.X_data[self.tem_batch_pos * self.batch_size: (self.tem_batch_pos + 1) * self.batch_size]
        batch_ys = self.Y_data[self.tem_batch_pos * self.batch_size: (self.tem_batch_pos + 1) * self.batch_size]
        if(process):
            batch_xs = self_random_crop(batch_xs);
        self.tem_batch_pos = self.tem_batch_pos + 1;
        return batch_xs , batch_ys 


    def have_next(self):
        if(self.tem_batch_pos < self.batch_num):
            return True;
        if(self.tem_batch_pos >= self.batch_num):
            if(self.nextfile() == False):
                return False;
            else:
                return self.have_next(); 

