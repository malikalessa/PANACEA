import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import Configurations
import Preprocessing
from Configurations import *
import pandas as pd
import numpy as np
my_seed = 123
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
from sklearn.decomposition import PCA

tf.random.set_seed(my_seed)
from Global_Config import *
from Preprocessing import *
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



dataset_choice = int(input('Enter 1 : NSL-KDD, 2 : UNSW , 3 : CICIDS, 4: CICMalDroid2020 '))
epsilon = float(input('Enter Epsilon Value : '))

gc.epsilon = epsilon

if dataset_choice == 1:
    train_scaler, test_scaler, y_train,y_test = Preprocessing.read_datasets_NSL_KDD()
elif dataset_choice == 2:
    train_scaler, test_scaler, y_train,y_test = Preprocessing.read_Dataset_UNSW()
elif dataset_choice == 3 :
    train_scaler, test_scaler, y_train, y_test = read_datasets_CICIDS()
elif dataset_choice == 4 :
    train_scaler, test_scaler, y_train, y_test = Preprocessing.read_datasets_CICMalDroid2020()


path_CSV = '../Ensemble_Baseline/'

print('train_datast shape : ', train_scaler.shape)
print('test_datast shape : ', test_scaler.shape)
print('y_train shape : ', y_train.shape)
print('y_test shape : ', y_test.shape)

print('np.unque ' ,np.unique(y_train))
gc.n_class = len(np.unique(y_train))
print('n_class ' , gc.n_class)

print('n_class test ' , np.unique(y_test))


config_No = 1
exp1 = Configurations.DNN_Classifier_AdvSample_OriginalData(train_scaler,y_train,test_scaler,y_test,
                                                  path_CSV,n_class, config_No)

