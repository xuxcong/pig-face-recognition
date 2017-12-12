import sys
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import shutil



#reshape the images to a square
def process_img(raw_path, result_path):
  img4 = Image.open(raw_path)
  longer_side = max(img4.size)
  horizontal_padding = (longer_side - img4.size[0]) / 2
  vertical_padding = (longer_side - img4.size[1]) / 2
  img5 = img4.crop(
      (
          -horizontal_padding,
          -vertical_padding,
          img4.size[0] + horizontal_padding,
          img4.size[1] + vertical_padding
      )
  )
  img4.close();
  img5 = img5.resize((512,512))
  img5.save(result_path)


  



if __name__ == '__main__':
    #This file reshape the images into specific size, and padding to a square
    '''
    train_path = "/home/smie/zhengjx/face_recognize/raw_train/"
    for i in range(1,31):
      data_path = train_path + str(i);
      result_file = data_path + '/';
      files = os.listdir(data_path)
      for img_name in files:
          img_path = data_path + '/' + img_name;
          result_img_path = result_file + '/' + img_name;
          process_img(img_path, result_img_path)

    '''

    test_path = "/home/smie/zhengjx/face_recognize/test_B/"
    result_path = 'process_testB/'
    for i in ['test_B']:
        data_path = test_path;
        result_file = result_path;
        if(os.path.exists(result_file) == True):        
            shutil.rmtree(result_file);
            time.sleep(1)
        os.mkdir(result_file);
        files = os.listdir(data_path)
        for img_name in files:
              img_path = data_path + '/' + img_name;
              result_img_path = result_file + '/' + img_name;
              process_img(img_path, result_img_path);

