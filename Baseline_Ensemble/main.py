import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import Configurations
from Configurations import *
import pandas as pd
import numpy as np
my_seed = 123
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)
from Global_Config import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def read_datasets_CICMalDroid2020():

    train_dataset = pd.read_csv('./train.csv')
    test_dataset = pd.read_csv('./test.csv')
    y_train = train_dataset['Class']
    y_test = test_dataset['Class']

    print('y_train labels : ', y_train.value_counts())
    print('y_test labels : ', y_test.value_counts())

    try:
        train_dataset.drop(['Class'], axis=1, inplace=True)
        test_dataset.drop(['Class'], axis=1, inplace=True)
    except IOError:
        print('IOERROR')
    return train_dataset,test_dataset,y_train,y_test


def read_datasets_NSL_KDD():

    train_dataset = pd.read_csv('E:/Processed_Dataset/NSL-KDD/train_numeric.csv')
    test_dataset = pd.read_csv('E:/Processed_Dataset/NSL-KDD/test_numeric.csv')
    y_train = train_dataset['attack']
    y_test = test_dataset['attack']

    try:
        train_dataset.drop(['attack'], axis=1, inplace=True)
        test_dataset.drop(['attack'], axis=1, inplace=True)
    except IOError:
        print('IOERROR')
    return train_dataset,test_dataset,y_train,y_test


def read_Dataset_UNSW():
    print('UNSW')
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    y_train = train['attack']
    y_test = test['attack']

    try:
      train.drop(['attack'], axis=1, inplace=True)
      test.drop(['attack'], axis=1, inplace=True)
    except IOError:
       print('IOERROR')

    train_scaler = np.asarray(train)
    test_scaler = np.asarray(test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return  train_scaler, test_scaler, y_train,y_test


def read_datasets_CICIDS():

    train_dataset = pd.read_csv('./train.csv')
    test_dataset = pd.read_csv('./test.csv')
    y_train = train_dataset['attack']
    y_test = test_dataset['attack']

    try:
        train_dataset.drop(['attack'], axis=1, inplace=True)
        test_dataset.drop(['attack'], axis=1, inplace=True)
    except IOError:
        print('IOERROR')
    return train_dataset,test_dataset,y_train,y_test






dataset_choice = int(input('Enter 1 : NSL-KDD, 2 : UNSW , 3 : CICIDS, 4: CICMalDroid2020 '))
epsilon = float(input('Enter Epsilon Value : '))

gc.epsilon = epsilon

if dataset_choice == 1:
    train_scaler, test_scaler, y_train,y_test = read_datasets_NSL_KDD()
elif dataset_choice == 2:
    train_scaler, test_scaler, y_train,y_test = read_Dataset_UNSW()
elif dataset_choice == 3 :
    train_scaler, test_scaler, y_train, y_test = read_datasets_CICIDS()
elif dataset_choice == 4 :
    train_scaler, test_scaler, y_train, y_test = read_datasets_CICMalDroid2020()


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

