

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import CICIDS_Configurations
import CIC_MalDroid2020_Configurations
import Model_Ensembling_T_A
import NSL_KDD_Configurations
import UNSW_Configurations
import create_datasets
import Create_Model
import New_Hypermodel
import Global_Config as gc
import Preprocessing
from Model_Ensembling_T_A import *
from create_datasets import *
from NSL_KDD_Configurations import *
from CICIDS_Configurations import *
from CIC_MalDroid2020_Configurations import *
import tensorflow as tf

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
testPath = './Ensembled_for_Server/'

## Initialize Datasets

datset_no = 0

# Choice 1 is to create a new datasets that contain original samples + adversarial samples
program_choice = int(input('1 for create datasets, 2 for create Models : '))

# Choice 2 to create models
if program_choice == 1 :
    create_datasets.Dataset()


if program_choice == 2:
    choice = int(input('Enter 1 for NSL-KDD dataset, 2 for UNSW Dataset, 3 for CICDDOS2019 :  , 4 for CIC_MaDroid2020 : '))
    n_models = int(input('Enter Number of Models  : '))

    gc.n_models = n_models
    config_No = 1

    if choice == 1:  # NSL_KDD

        print('\nThe following functions will be implemented using NSL_KDD Dataset\n')
        print('\n 1 for creating new Models : ')
        print('\n 2 for Model Ensembling using Sequential : ')
        print('\n 3 for Individual Predictions for Models : ')
        print('\n 4 NSL_KDD Medoids : ')
        print('\n 5 for NSL_KDD Voting : ')

        func = int(input('Enter Function Number : '))
        if func == 1 :
            NSL_KDD_Configurations.create_models_NSL_KDD()
            print('')
        elif func == 2 :
            NSL_KDD_Configurations.Model_Ensembling_Train_Sequential()
        elif func == 3 :
            NSL_KDD_Configurations.Individual_Models_Predictions()
        elif func == 4 :
            NSL_KDD_Configurations.Model_Ensembling_Train_Adversarial_NSL_KDD_Medoids()
        elif func == 5:
            NSL_KDD_Configurations.create_ensemble_voting()

    if choice == 2: # UNSW-NB 15 Dataset

        print('\nThe following functions will be implemented using UNSW Dataset\n')
        print('\n 1 for creating new Models : ')
        print('\n 2 for Model Ensembling using Sequential : ')
        print('\n 3 for Individual Predictions for Models : ')
        print('\n 4 for UNSW Medoids')
        print('\n 5 for Voting : ')

        func = int(input('Enter Function Number : '))
        if func == 1:
            UNSW_Configurations.create_models_UNSW()
        elif func == 2:
            UNSW_Configurations.Model_Ensembling_Train_Sequential()
        elif func == 3:
            UNSW_Configurations.Individual_Models_Predictions()
        elif func == 4 :
            UNSW_Configurations.Model_Ensembling_Train_Adversarial_UNSW_Medoids()
        elif func == 5 :
            UNSW_Configurations.create_ensemble_voting()

    if choice == 3:

        print('\nThe following functions will be implemented using CICIDS Dataset\n')
        print('\n 1 for creating new Models : ')
        print('\n 2 for Model Ensembling using Sequential : ')
        print('\n 3 for Individual Predictions for Models : ')
        print('\n 4 CICIDS Medoids : ')
        print('\n 5 for CICIDS Voting : ')

        func = int(input('Enter Function Number : '))
        if func == 1:
            CICIDS_Configurations.create_models_CICIDS()
            print('')
        elif func == 2:
            CICIDS_Configurations.Model_Ensembling_Train_Sequential()
        elif func == 3:
            CICIDS_Configurations.Individual_Models_Predictions()
        elif func == 4:
            CICIDS_Configurations.Model_Ensembling_Train_Adversarial_CICIDS_Medoids()
        elif func == 5:
            CICIDS_Configurations.create_ensemble_voting()

    if choice == 4:

        print('\nThe following functions will be implemented using CIC_MalDroid2020 Dataset\n')
        print('\n 1 for creating new Models : ')
        print('\n 2 for Model Ensembling using Sequential : ')
        print('\n 3 for Individual Predictions for Models : ')
        print('\n 4 CIC_MalDroid2020 Medoids : ')
        print('\n 5 for CIC_MalDroid2020 Voting : ')

        func = int(input('Enter Function Number : '))
        if func == 1:
            CIC_MalDroid2020_Configurations.create_models_CIC_MalDroid2020()
            print('')
        elif func == 2:
            CIC_MalDroid2020_Configurations.Model_Ensembling_Train_Sequential()
        elif func == 3:
           CIC_MalDroid2020_Configurations.Individual_Models_Predictions()

        elif func == 4:
            CIC_MalDroid2020_Configurations.Model_Ensembling_Train_Adversarial_CIC_MalDroid2020_Medoids()
        elif func == 5:
            CIC_MalDroid2020_Configurations.create_ensemble_voting()
