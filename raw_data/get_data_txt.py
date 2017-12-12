import sys
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import shutil


if __name__ == '__main__':

    
	work_file = os.getcwd();
	
        data_directory = '/home/smie/zhengjx';
        filenum = 50;
        train_file = [];
        validation_file = [];
        for i in range(0,50):
	    train = open('txt/train_data' + str(i) + '.txt','w')
	    validation = open('txt/validation_data' + str(i) +'.txt','w');
	    train_file.append(train);
            validation_file.append(validation);

	data_path = os.path.join(data_directory,'ROI');
        train_num = 0;
        validation_num = 0;
	all_num = 0;
	for i in range(1,31):
		nowfile = os.path.join(data_path, str(i)); 
		files = os.listdir(nowfile)
		for filename in files:
			newname = str(all_num) + '.jpg';
			while(os.path.exists(os.path.join(nowfile,newname)) == True):
				newname = str(i) + newname;
			os.rename(os.path.join(nowfile,filename),os.path.join(nowfile,newname))
			filepath = os.path.join(nowfile,newname);
			label = str(i);
			all_num = all_num + 1;
			if(all_num % 5 == 0):
				validation_file[validation_num % filenum].write(filepath + ' ' + label + '\n');
                                validation_num = validation_num + 1;
			else:
				train_file[train_num % filenum].write(filepath + ' ' + label + '\n');
                                train_num = train_num + 1;
	'''
	test_file = open("testresult.txt",'w');
	data_path = os.path.join(work_file,'testresult');
	for i in range(1,31):
		nowfile = os.path.join(data_path, str(i)); 
		files = os.listdir(nowfile)
		for filename in files:
			filepath = os.path.join(nowfile,filename);
			label = str(i);
			test_file.write(filepath + ' ' + label + '\n');
        '''
	'''
	test_file = open("anstest_data.txt",'w');
	data_path = os.path.join(work_file,'process_test');
	for i in ['test_A']:
		nowfile = os.path.join(data_path, str(i)); 
		files = os.listdir(nowfile)
		for filename in files:
			filepath = os.path.join(nowfile,filename);
			label = str(1);
			test_file.write(filepath + ' ' + label + '\n');
	'''
