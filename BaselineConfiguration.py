import time
import os
from keras.models import load_model

import Baseline_HyperModel
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from Adversarial_Datasets import *
import Global_Config as gc
from report import *


class BaselineConfiguration():

    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig


    def train_baseline(self,x_train,x_test, y_train,y_test, n_class):
        ##Hyperopt on T1 to learn  DNN####
        path=self.ds.get('pathModels')
        gc.n_class = int(n_class)
        tic = time.time()
        report_name = 'Hyperopt_Config2Baseline.txt'
        report_name = path + report_name
        try:
            os.remove(report_name)
        except:
            print('')
        config_No = 2

       #model_hyperopt, time1,score = Baseline_HyperModel.hypersearch(x_train,y_train,x_test,y_test,path,config_No)
        #model_hyperopt.save(path + 'Baseline_model_NSL_KDD.h5')
        model_hyperopt = load_model(path + 'Baseline_model_NSL_KDD.h5')
        gc.baseline_model = model_hyperopt
        model_hyperopt.summary()


        Y_predicted = np.argmax(model_hyperopt.predict(x_test), axis=1)

        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)
        print('Accuracy : ', Accuracy)
        toc = time.time()
        tictoc = (toc - tic) / 60
        name = 'Hyperopt_Configuration2'

        report(report_name, Accuracy, Confusion_matrix, Classification_report, name)

        self.DNN_Classifier_on_Adv_Samples(x_train,y_train,x_test,y_test,model_hyperopt)


    def DNN_Classifier_on_Adv_Samples(self, train, y_train, x_test, y_test, model):
        #here the code to create adversarial sample (conf 4)

        path = self.ds.get('pathModels')

        epsilon = float(self.ds.get('epsilon'))
        adversarial_samples = Adversarial_Datasets.Adversarial_Samples(self,train, y_train)

        adversarial_original_samples = np.append(train, adversarial_samples, axis=0)
        y_label_adversarial = np.append(y_train, y_train, axis=0)

        ##### Training on Original Dataset(A) Predictions on Adversarial Dataset
        Y_predicted = np.argmax(model.predict(adversarial_samples), axis=1)
        Confusion_matrix = confusion_matrix(y_train, Y_predicted)
        Classification_report = classification_report(y_train, Y_predicted)
        Accuracy = accuracy_score(y_train, Y_predicted)
        print('Accuracy : ', Accuracy)

        report_name = path + 'Config_4.txt'
        name = 'Make Predictions Based on Adversarial Samples_Config 4'
        try:
            os.remove(report_name)
        except:
            print('')

        report(report_name, Accuracy, Confusion_matrix, Classification_report, name)

        ######## Configuration 6 , training using the adversarial samples+ original samples
        config_No = 6

        model_config_6, time1, score = Baseline_HyperModel.hypersearch(adversarial_original_samples, y_label_adversarial
                                                                       , x_test, y_test, path,config_No)

        model_config_6.save(path+'Model_config_6.h5')
        Y_predicted = np.argmax(model_config_6.predict(x_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)
        print('Accuracy : ', Accuracy)

        name = 'Config _ 6'
        report_name = path + 'Config_6.txt'
        try:
            os.remove(report_name)
        except:
            print('')

        report(report_name, Accuracy, Confusion_matrix, Classification_report, name)






