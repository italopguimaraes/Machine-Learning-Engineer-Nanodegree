from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.nasnet import preprocess_input
from keras.preprocessing import image                  
from tqdm import tqdm
import tensorflow as tf
import nasnet
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 2)
    return files, targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(331,331))
    # convert PIL.Image.Image type to 3D tensor with shape (331,331,3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1,450,450,3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


train_files_1, train_targets_1 = load_dataset('data_1/train')
valid_files_1, valid_targets_1 = load_dataset('data_1/valid')
test_files_1, test_targets_1 = load_dataset('data_1/test')

train_files_2, train_targets_2 = load_dataset('data_2/train')
valid_files_2, valid_targets_2 = load_dataset('data_2/valid')
test_files_2, test_targets_2 = load_dataset('data_2/test')

train_tensors_1 = (paths_to_tensor(train_files_1).astype('float32')-127.5)/255
valid_tensors_1 = (paths_to_tensor(valid_files_1).astype('float32')-127.5)/255
test_tensors_1 = (paths_to_tensor(test_files_1).astype('float32')-127.5)/255

train_tensors_2 = (paths_to_tensor(train_files_2).astype('float32')-127.5)/255
valid_tensors_2 = (paths_to_tensor(valid_files_2).astype('float32')-127.5)/255
test_tensors_2 = (paths_to_tensor(test_files_2).astype('float32')-127.5)/255

#NASNet_Large_base = nasnet.large(weights='imagenet', include_top=False, input_shape=train_tensors.shape[1:])
NASNet_Large_base=tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=train_tensors_1.shape[1:])

bottleneck_feature_train_1=NASNet_Large_base.predict(train_tensors_1)
bottleneck_feature_test_1=NASNet_Large_base.predict(test_tensors_1) 
bottleneck_feature_valid_1=NASNet_Large_base.predict(valid_tensors_1)

bottleneck_feature_train_2=NASNet_Large_base.predict(train_tensors_2)
bottleneck_feature_test_2=NASNet_Large_base.predict(test_tensors_2) 
bottleneck_feature_valid_2=NASNet_Large_base.predict(valid_tensors_2)

np.savez_compressed('bottleneck_features/NASNetLarge_1', train=bottleneck_feature_train_1, test=bottleneck_feature_test_1,valid=bottleneck_feature_valid_1)
np.savez_compressed('bottleneck_features/NASNetLarge_2', train=bottleneck_feature_train_2, test=bottleneck_feature_test_2,valid=bottleneck_feature_valid_2)
