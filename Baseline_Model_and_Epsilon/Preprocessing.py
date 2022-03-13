import pandas as pd
import numpy as np
import Global_Config as gc

def read_datasets_CICMalDroid2020():

    train_dataset = pd.read_csv(gc.Maldroid20_Original_Dataset + 'train_scaler_70%.csv')
    test_dataset = pd.read_csv(gc.Maldroid20_Original_Dataset + 'test_scaler_30%.csv')
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

    train_dataset = pd.read_csv(gc.NSL_KDD_Original_Dataset+'train_numeric.csv')
    test_dataset = pd.read_csv(gc.NSL_KDD_Original_Dataset+'test_numeric.csv')
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
    train_scaler = pd.read_csv(gc.UNSW_Original_Dataset+'train_numeric.csv')
    test_scaler = pd.read_csv(gc.UNSW_Original_Dataset+'test_numeric.csv')
    y_train = train_scaler['attack']
    y_test = test_scaler['attack']

    try:
      train_scaler.drop(['attack'], axis=1, inplace=True)
      test_scaler.drop(['attack'], axis=1, inplace=True)
    except IOError:
       print('IOERROR')

    #train_scaler = np.asarray(train)
    #test_scaler = np.asarray(test)

       ############### PCA for UNSW ##########################################################

    pca = PCA(n_components=0.97, svd_solver='full', random_state=42)
    train_scaler = pca.fit_transform(train_scaler)
    print('shape ha ', train_scaler.shape)
    test_scaler = pca.transform(test_scaler)
    print(pca.explained_variance_ratio_)
    dataTrain = pd.DataFrame(train_scaler).add_prefix('feature_')
    cls = 'attack'
    dataTrain[cls] = y_train
    dataTrain.to_csv('PCATrain.csv', index=False)
    dataTest = pd.DataFrame(test_scaler).add_prefix('feature_')
    dataTest[cls] = y_test
    dataTest.to_csv('PCATest.csv', index=False)

    #y_train = np.asarray(y_train)
    #y_test = np.asarray(y_test)
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
