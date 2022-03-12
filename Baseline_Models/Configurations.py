from sklearn.metrics import confusion_matrix, accuracy_score
import os
import Create_Adv_Samples
import New_HyperModel
from report import *
import time as time
from New_HyperModel import *
from Global_Config import *
from keras.models import load_model
import numpy as np




def hyperopt(train_scaler, y_train, test_scaler, y_test, path, n_class, config_No):
    ##Hyperopt on T1 to learn  DNN####
    ########################### Config 2 #########################3
    tic = time.time()
    report_name = 'Hyperopt_Config2.txt'
    report_name = path + report_name
    try:
        os.remove(report_name)
    except:
        print('')
    config_No = 2

    model_hyperopt, time1,score = New_HyperModel.hypersearch(train_scaler,y_train,test_scaler,y_test,path,config_No)
    #model_hyperopt.save(path+'UNSW_Label_Encoder.h5')
    #model_hyperopt = load_model(path+'ConfigNo2_NN.h5')

    Y_predicted = np.argmax(model_hyperopt.predict(test_scaler), axis=1)
    Confusion_matrix = confusion_matrix(y_test, Y_predicted)
    Classification_report = classification_report(y_test, Y_predicted)
    Accuracy = accuracy_score(y_test, Y_predicted)
    print('Accuracy : ', Accuracy)
    toc = time.time()
    tictoc = (toc - tic) / 60
    name = 'Hyperopt_Configuration2'

    report(report_name, Accuracy, Confusion_matrix, Classification_report, tictoc, model_hyperopt, name)
    return model_hyperopt


def DNN_Classifier_on_Adv_Samples(train_scaler, y_train, test_scaler, y_test, path,n_class,config_No):

    ################# Config 4 #########################3
    model = hyperopt(train_scaler, y_train, test_scaler, y_test, path, n_class, config_No)
    #### Generate Adversarial Samples
    adversarial_original_samples, y_label_adversarial, adversarial_samples = Create_Adv_Samples.Adv_Samples(model,
                                                train_scaler, y_train, test_scaler,y_test, path, config_No)
    print('Adv Samples.shape', adversarial_samples.shape)
    tic = time.time()

    ##### Training on Original Dataset(A) Predictions on Adversarial Dataset
    Y_predicted = np.argmax(model.predict(adversarial_samples), axis=1)
    Confusion_matrix = confusion_matrix(y_train, Y_predicted)
    Classification_report = classification_report(y_train, Y_predicted)
    Accuracy = accuracy_score(y_train, Y_predicted)
    print('Accuracy : ', Accuracy)
    toc = time.time()
    tictoc = (toc - tic) / 60
    report_name = 'Adv_Prediction' + str(config_No)
    report_name = path + report_name + 'Config_4.txt'
    name = 'Make Predictions Based on Adversarial Samples_Config4'
    try:
        os.remove(report_name)
    except:
        print('')

    report(report_name, Accuracy, Confusion_matrix, Classification_report, tictoc, model, name)

    return adversarial_original_samples, y_label_adversarial, adversarial_samples


####################Config 6  #############################


def DNN_Classifier_AdvSample_OriginalData(train_scaler, y_train, test_scaler, y_test, path, n_class, config_No):
    tic = time.time()
    ### Calling function to predict Adversarial Samples only
    adversarial_original_samples, y_label_adversarial, adversarial_samples = DNN_Classifier_on_Adv_Samples(train_scaler,
                                                            y_train, test_scaler, y_test, path,n_class,config_No)

    print('Adv.shape', adversarial_original_samples.shape)
    print('Adv label ', y_label_adversarial.shape)

    config_No = 6
    model, time1, best_score = New_HyperModel.hypersearch(adversarial_original_samples,y_label_adversarial, test_scaler,
                                                          y_test, path, config_No)

    Y_predicted = np.argmax(model.predict(test_scaler), axis=1)
    Confusion_matrix = confusion_matrix(y_test, Y_predicted)
    Classification_report = classification_report(y_test, Y_predicted)
    Accuracy = accuracy_score(y_test, Y_predicted)
    print('Accuracy : ', Accuracy)

    toc = time.time()
    tictoc = (toc - tic) / 60
    report_name = 'AdvSample_OriginalData' + str(config_No) +'Config_6.txt'
    report_name = path + report_name
    try:
        os.remove(report_name)
    except:
        print('')
    name = 'AdvSample_OriginalData'
    report(report_name, Accuracy, Confusion_matrix, Classification_report, tictoc, model, name)
