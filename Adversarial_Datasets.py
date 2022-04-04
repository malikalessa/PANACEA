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

import Global_Config as gc

class Adversarial_Datasets():

    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig

    def Adversarial_Samples(self,train_dataset,y_train):
        ####### Create FastGradientMethod ##########

        train_dataset = np.asarray(train_dataset)
        model = load_model(self.ds.get('pathModels')+self.ds.get('baseline_model'))

        classifier = TensorFlowV2Classifier(model, nb_classes=int(self.ds.get('n_class')),
                                            input_shape=(1, train_dataset.shape[1]),
                                            loss_object=tf.keras.losses.CategoricalCrossentropy())

        epsilon = float(self.ds.get('epsilon'))
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)

        adversarial_samples = attack.generate(x = train_dataset)

        return adversarial_samples

    def add_Adversarial_Samples(self,train_dataset,y_train):

        dataset_path = self.ds.get('Adv_dataset')
        No_datasets_to_generate = int(self.config.get('NUMBER_OF_MODELS'))
        # here the code to load the candidate models
        columns = train_dataset.columns
        train_dataset = np.asarray(train_dataset)
        y_train = np.asarray(y_train)
        adversarial_samples = self.Adversarial_Samples(train_dataset, y_train)
        n_samples = int(train_dataset.shape[0] * float(self.config.get('sigma')))

        for i in range(No_datasets_to_generate):
            ######## Adding Adversarial Samples with Original Samples in the training dataset
            new_train_adv, _, y_new, _ = train_test_split(adversarial_samples, y_train, train_size=n_samples,
                                                          shuffle=True, stratify=y_train)
            new_train_adv = pd.DataFrame(new_train_adv, columns=columns)

            y_new = pd.DataFrame(y_new, columns=[self.ds.get('label')])
            print('after appending adv samples to the original dataset : ', new_train_adv.shape)
            print('after appending adv labels to the original labels : ', y_new.shape)

            print('y_new : ', y_new.value_counts())

            new_train = pd.concat([new_train_adv, y_new], axis=1)
            print('New train Dataset shape : ', new_train.shape)

            dataset_path_1 = dataset_path + 'T_A_Dataset_'+self.ds.get('Dataset_name') + str(i) + '.csv'
            new_train.to_csv(path_or_buf=dataset_path_1, index=False)

############# To read the adversarial datasets that is concatenated to the original dataset



    def read_T_A_Datasets(self,train_dataset,y_train,dataset_no):

        path = self.ds.get('Adv_dataset')

        train_adv_samples = pd.read_csv(path + 'T_A_Dataset_'+self.ds.get('Dataset_name') + str(dataset_no) + '.csv')
        cls = self.ds.get('label')
        y_train_adv = train_adv_samples[cls]

        try:
            train_adv_samples.drop([cls], axis= 1, inplace=True)
        except IOError:
            print('IOERROR')
        print('Train Adv Dataset shape : ', train_adv_samples.shape)
        print('YTrain Adv Dataset shape : ', y_train_adv.shape)

        train_dataset = train_dataset.append(train_adv_samples)
        y_train = y_train.append(y_train_adv)

        print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
        print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)

        return train_dataset, y_train
