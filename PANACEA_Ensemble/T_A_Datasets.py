
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import randn, randint
from keras.models import load_model
import numpy as np
import tensorflow as tf
import numpy as np
import os
from art.estimators.classification import TensorFlowV2Classifier
import time as time
from art.attacks.evasion import FastGradientMethod
from sklearn.preprocessing import MinMaxScaler

import Global_Config as gc


def Adversarial_Samples(train_dataset,y_train, epsilon,columns, dataset_path):
    ####### Create FastGradientMethod ##########

    train_dataset = np.asarray(train_dataset)
    model = gc.baseline_model
    classifier = TensorFlowV2Classifier(model, nb_classes=len(np.unique(y_train)), input_shape=(1, train_dataset.shape[1]),
                                        loss_object=tf.keras.losses.CategoricalCrossentropy())

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    adversarial_samples = attack.generate(x=train_dataset)

    adversarial_original_samples = np.append(train_dataset, adversarial_samples, axis=0)
    y_label_adversarial = np.append(y_train, y_train, axis=0)

    dataset_path = dataset_path + 'Adversarial+Original.csv'
    csv_adversarial_original_samples = pd.DataFrame(adversarial_original_samples, columns=columns)
    y_label_adversarial = pd.DataFrame(y_label_adversarial, columns=['Class'])
    print('label : ', y_label_adversarial.shape)
    # to save the adversarial samples with original training dataset
    csv_adversarial_original_samples = pd.concat([csv_adversarial_original_samples, y_label_adversarial], axis=1)
    csv_adversarial_original_samples.to_csv(path_or_buf=dataset_path, index=False)


    return   adversarial_samples


def Add_Adversarial_Samples(train_dataset,y_train, No_datasets_to_generate, n_samples_percentage,epsilon
                                             , dataset_path, dataset):

    columns = train_dataset.columns
    train_dataset = np.asarray(train_dataset)
    y_train = np.asarray(y_train)
    adversarial_samples = Adversarial_Samples(train_dataset, y_train, epsilon,columns, dataset_path)
    n_samples = int(train_dataset.shape[0] * n_samples_percentage)

    for i in range (No_datasets_to_generate):

    ######## Adding Adversarial Samples with Original Samples in the training dataset

        new_train_adv, _, y_new, _ = train_test_split(adversarial_samples, y_train, train_size=n_samples,
                                                      shuffle=True, stratify=y_train)
        new_train_adv = pd.DataFrame(new_train_adv, columns=columns)

        if dataset == 1 or 2 :
             y_new = pd.DataFrame(y_new, columns=['attack'])
        elif dataset == 3:
             y_new = pd.DataFrame(y_new, columns=[' Label'])
        elif dataset == 4:
              y_new = pd.DataFrame(y_new, columns=['Class'])
        print('after appending adv samples to the original dataset : ', new_train_adv.shape)
        print('after appending adv labels to the original labels : ', y_new.shape)

        print('y_new : ', y_new.value_counts())

        new_train = pd.concat([new_train_adv, y_new], axis=1)
        print('New train Dataset shape : ', new_train.shape)

        if dataset == 1:
            dataset_path_1 = dataset_path +'T_A_Dataset_NSL_KDD'+str(i)+'.csv'
            new_train.to_csv(path_or_buf=dataset_path_1, index=False)
        elif dataset == 2:
            dataset_path_2 = dataset_path + 'T_A_Dataset_UNSW' + str(i) + '.csv'
            new_train.to_csv(path_or_buf=dataset_path_2, index=False)
        elif dataset == 3 :
            dataset_path_3 = dataset_path + 'T_A_Dataset_CICIDS' + str(i) + '.csv'
            new_train.to_csv(path_or_buf=dataset_path_3, index=False)
        elif dataset == 4:
            dataset_path_4 = dataset_path + 'T_A_Dataset_Maldroid' + str(i) + '.csv'
            new_train.to_csv(path_or_buf=dataset_path_4, index=False)



