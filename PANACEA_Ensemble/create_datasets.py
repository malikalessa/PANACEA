import pandas as pd
from keras.models import load_model

import Preprocessing
import T_A_Datasets
from T_A_Datasets import *
from Preprocessing import *
import Global_Config as gc



def Dataset():

    dataset = int(input('Enter 1 for NSL_KDD Dataset, 2 for UNSW Dataset, 3 for CICDDOS2019 , 4 for CIC_MalDroid2020: '))
    No_datasets_to_generate =  int (input('Enter Number of Datasets to be Generated : '))
    epsilon = float(input('Enter Epsilon Value : '))
    n_samples_percentage = float(input('Enter the Percentage Number of Adversarial Samples to be added :  '))


    if dataset == 1 :
        train_dataset, test_dataset, y_train, y_test = Preprocessing.read_NSL_KDD_Dataset()
        dataset_path = gc.NSL_KDD_datset
        gc.baseline_model = load_model('./NSL_KDD_baseline_model.h5')
        T_A_Datasets.Add_Adversarial_Samples(train_dataset, y_train, No_datasets_to_generate,
                                                     n_samples_percentage,
                                                     epsilon, dataset_path, dataset)  ### Calling NSL_KDD Adversarial Add Samples

    if dataset == 2 :

        train_dataset, test_dataset, y_train, y_test = Preprocessing.read_UNSW_Dataset()
        #dataset_path = gc
        gc.baseline_model = load_model('./UNSW_baseline_model.h5')

        T_A_Datasets_UNSW.Add_Adversarial_Samples_UNSW(train_dataset,y_train,No_datasets_to_generate,
                                                   n_samples_percentage, epsilon, dataset_path, dataset)  #### Calling UNSW Adversarial

    if dataset == 3 :

        train_dataset, test_dataset, y_train, y_test = Preprocessing.read_CICIDS_Dataset()
        #dataset_path = gc.
        gc.baseline_model = load_model('./CICIDS_baseline_model.h5')

        T_A_Datasets_CICDDOS.Add_Adversarial_Samples_CICDDS(train_dataset, y_train, No_datasets_to_generate,
                                                            n_samples_percentage, epsilon, dataset_path, dataset)

    if dataset == 4:
        train_dataset, test_dataset, y_train, y_test = Preprocessing.read_CIC_MalDroid2020_Dataset()
        dataset_path = gc.Maldroid20_Datasets
        gc.baseline_model = load_model('./Maldroid20_baseline_model.h5')

        T_A_Datasets.Add_Adversarial_Samples(train_dataset, y_train, No_datasets_to_generate,
                                                            n_samples_percentage, epsilon, dataset_path, dataset)
