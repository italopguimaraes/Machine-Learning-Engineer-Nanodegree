from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.nasnet import preprocess_input
from keras.preprocessing import image                  
from tqdm import tqdm
import tensorflow as tf
import nasnet
import time
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

inicio = time.time()

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 3)
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


train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')


train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


#NASNet_Large_base = nasnet.large(weights='imagenet', include_top=False, input_shape=train_tensors.shape[1:])
NASNet_Large_base=tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=train_tensors.shape[1:])

"""
print('antes:',len(NASNet_Large_base.layers))
for layer in NASNet_Large_base.layers[-88:]:
    NASNet_Large_base.layers.pop()
print('depois:',len(NASNet_Large_base.layers))
"""

bottleneck_feature_train=NASNet_Large_base.predict(train_tensors)
bottleneck_feature_test=NASNet_Large_base.predict(test_tensors) 
bottleneck_feature_valid=NASNet_Large_base.predict(valid_tensors)
np.savez_compressed('bottleneck_features/NASNetLarge', train=bottleneck_feature_train, test=bottleneck_feature_test,valid=bottleneck_feature_valid)
np.savez_compressed('tensors/tensors', train=train_tensors, test=test_tensors, valid=valid_tensors)
np.savez_compressed('tensors/targets', train=train_targets, test=test_targets, valid=valid_targets)
fim = time.time()



