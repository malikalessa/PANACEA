import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import Global_Config as gc

################ NSL-KDD Dataset ####################33
def read_NSL_KDD_Dataset():

    path = gc.NSL_KDD_Original_Dataset
    train_dataset  =  pd.read_csv(path +'train_numeric.csv')
    test_dataset = pd.read_csv( path + 'test_numeric.csv')
    y_train = train_dataset['attack']
    y_test = test_dataset['attack']
    try:
        train_dataset.drop(['attack'], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    try:
        test_dataset.drop(['attack'] , axis =1 , inplace= True)
    except IOError:
        print('IOERROR')

    print('Train Dataset shape : ', train_dataset.shape)
    print('Test Dataset shape : ', test_dataset.shape)
    print('YTrain Dataset shape : ', y_train.shape)
    print('Ytest Dataset shape : ', y_test.shape)

    return train_dataset, test_dataset, y_train,y_test


def read_T_A_Datasets_NSL(train_dataset,y_train,dataset_no):

    path = gc.NSL_KDD_datset
    train_adv_samples  =  pd.read_csv(path +'T_A_Dataset_NSL_KDD'+str(dataset_no)+'.csv')
    y_train_adv = train_adv_samples['attack']

    try:
        train_adv_samples.drop(['attack'], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    print('Train Adv Dataset shape : ', train_adv_samples.shape)
    print('YTrain Adv Dataset shape : ', y_train_adv.shape)

    train_dataset = train_dataset.append(train_adv_samples)
    y_train = y_train.append(y_train_adv)

    print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
    print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)


    return train_dataset, y_train

############ UNSW Dataset ##########################
def read_UNSW_Dataset():

    path = gc.UNSW_Original_Dataset

    train_dataset = pd.read_csv(path + 'train.csv')
    test_dataset = pd.read_csv(path + 'test.csv')

    y_train = train_dataset['attack']
    y_test = test_dataset['attack']

    try:
        train_dataset.drop(['attack'], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    try:
        test_dataset.drop(['attack'] , axis =1 , inplace= True)
    except IOError:
        print('IOERROR')

    print('Train Dataset shape : ', train_dataset.shape)
    print('Test Dataset shape : ', test_dataset.shape)
    print('YTrain Dataset shape : ', y_train.shape)
    print('Ytest Dataset shape : ', y_test.shape)

    return train_dataset, test_dataset, y_train,y_test

def read_T_A_Datasets_UNSW(train_dataset,y_train,dataset_no):

    path = gc.UNSW_Datasets

    train_adv_samples  =  pd.read_csv(path +'T_A_Dataset_UNSW'+str(dataset_no)+'.csv')
    y_train_adv = train_adv_samples['attack']
    print(train_adv_samples.columns)

    try:
        train_adv_samples.drop(['attack'], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    print('Adv Columns : ', train_adv_samples.columns)
    print('Train Adv Dataset shape : ', train_adv_samples.shape)
    print('YTrain Adv Dataset shape : ', y_train_adv.shape)

    train_dataset = train_dataset.append(train_adv_samples)
    y_train = y_train.append(y_train_adv)

    print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
    print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)



    return train_dataset, y_train


#################### CICDDOS2019 Dataset #########################333

def read_CICIDS_Dataset():

    path = './'

    train_dataset = pd.read_csv(path +'train.csv' )
    test_dataset = pd.read_csv( path + 'test.csv')

    y_train = train_dataset[ 'cc']
    y_test = test_dataset[ 'cc']
    try:
        train_dataset.drop([  ], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    try:
        test_dataset.drop([  ] , axis =1 , inplace= True)
    except IOError:
        print('IOERROR')

    print('Train Dataset shape : ', train_dataset.shape)
    print('Test Dataset shape : ', test_dataset.shape)
    print('YTrain Dataset shape : ', y_train.shape)
    print('Ytest Dataset shape : ', y_test.shape)

    print('class in y_train : ', y_train.value_counts())
    print('classes in y_test : ', y_test.value_counts())
    
    return train_dataset, test_dataset, y_train,y_test


def read_T_A_Datasets_CICIDS(train_dataset,y_train,dataset_no):

    path = gc.CIC_IDS_Datasets

    train_adv_samples  =  pd.read_csv(path +'T_A_Dataset_CICIDS'+str(dataset_no)+'.csv')
    y_train_adv = train_adv_samples[ 'cc'  ]
    print(train_adv_samples.columns)

    try:
        train_adv_samples.drop([ 'cc'   ], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    print('Adv Columns : ', train_adv_samples.columns)
    print('Train Adv Dataset shape : ', train_adv_samples.shape)
    print('YTrain Adv Dataset shape : ', y_train_adv.shape)

    train_dataset = train_dataset.append(train_adv_samples)
    y_train = y_train.append(y_train_adv)

    print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
    print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)

    train_dataset,y_train = shuffle(train_dataset,y_train, random_state=42)

    return train_dataset, y_train



########################  CIC_MaldDroid2020 ###########3

def read_CIC_MalDroid2020_Dataset():
    path = gc.Maldroid20_Original_Dataset

    train_dataset = pd.read_csv(path + 'train_scaler_70%.csv')
    test_dataset = pd.read_csv(path + 'test_scaler_30%.csv')

    y_train = train_dataset['Class']
    y_test = test_dataset['Class']
    try:
        train_dataset.drop(['Class'], axis=1, inplace=True)
    except IOError:
        print('IOERROR')
    try:
        test_dataset.drop(['Class'], axis=1, inplace=True)
    except IOError:
        print('IOERROR')

    print('Train Dataset shape : ', train_dataset.shape)
    print('Test Dataset shape : ', test_dataset.shape)
    print('YTrain Dataset shape : ', y_train.shape)
    print('Ytest Dataset shape : ', y_test.shape)

    print('class in y_train : ', y_train.value_counts())
    print('classes in y_test : ', y_test.value_counts())

    return train_dataset, test_dataset, y_train, y_test



def read_T_A_Datasets_CIC_MalDroid2020(train_dataset,y_train,dataset_no):

    path = gc.Maldroid20_Datasets

    train_adv_samples  =  pd.read_csv(path +'T_A_Dataset_MalDroid'+str(dataset_no)+'.csv')
    y_train_adv = train_adv_samples['Class']
    print(train_adv_samples.columns)

    try:
        train_adv_samples.drop(['Class'], axis = 1, inplace=True)
    except IOError:
        print('IOERROR')
    print('Adv Columns : ', train_adv_samples.columns)
    print('Train Adv Dataset shape : ', train_adv_samples.shape)
    print('YTrain Adv Dataset shape : ', y_train_adv.shape)

    train_dataset = train_dataset.append(train_adv_samples)
    y_train = y_train.append(y_train_adv)

    print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
    print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)

    train_dataset,y_train = shuffle(train_dataset,y_train, random_state=42)

    return train_dataset, y_train