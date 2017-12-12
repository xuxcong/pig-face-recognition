from tflearn.data_utils import build_hdf5_image_dataset
import h5py


path = '/home/smie/zhengjx/Res_Bilinear_cnns/raw_data/txt/' 
filenum = 50;
filename = 'train_data'
files = [];
result = [];
for i in range(0, filenum):
    files.append(path + filename + str(i) + '.txt');
    result.append(filename + str(i) + '.h5')
    build_hdf5_image_dataset(files[i], image_shape=(488, 488), mode='file', output_path=result[i], categorical_labels=True, normalize=False)
    print('Finish dataset ' + result[i]);
