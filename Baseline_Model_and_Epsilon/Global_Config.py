
import numpy as np

best_score = np.inf
best_accuracy=0
best_scoreTest=0
best_model = None
best_model_test=None
best_numparameters = 0
best_score2=0
best_time =0
savedScore=[]
savedTrain=[]
n_class=None
test_path=None

train_X= None
train_Y= None
test_X=None
test_Y =None
train_R =None

DNN_Model = None
DNN_Model_Batch_Size = 0

epsilon = 0

# Paths for Original Datasets

NSL_KDD_Original_Dataset = 'E:/Processed_Dataset/NSL-KDD/' # To read Original Dataset

Maldroid20_Original_Dataset =  'E:/Processed_Dataset/CICMalDroid2020/'   # To read Original Dataset


UNSW_Original_Dataset = 'E:/Processed_Dataset/UNSW-NB15/' # To read original Dataset
