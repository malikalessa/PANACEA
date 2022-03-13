
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


n_models = 0
members = None

New_XTraining = None
YTraining = None
New_XValidation = None
YValidation = None

counter = 0
best_model_ensemble = None

###### To set the path

NSL_KDD_Original_Dataset = 'E:/Processed_Dataset/NSL-KDD/' # To read Original Dataset
NSL_KDD_datset = './NSL_KDD/Datasets/' # To save and read Adversarial Samples
NSL_KDD_Models = './NSL_KDD/Models/'  # To save and read Baseline Models

Maldroid20_Original_Dataset =  'E:/Processed_Dataset/CICMalDroid2020/'   # To read Original Dataset
Maldroid20_Datasets = './Maldroid20/Datasets/' # To save and read Adversarial Samples
Maldroid_20_Models = './Maldroid20/Models/'  # To save and read Baseline Models


UNSW_Original_Dataset = 'E:/Processed_Dataset/UNSW-NB15/' # To read original Dataset
UNSW_Datasets = './UNSW/Datasets/'  # To save and read Adversarial Samples
UNSW_Models = './UNSW/Models/'  # To save and read  baseline Models


CIC_IDS_Original_Dataset ='E:/Processed_Dataset/CIC_IDS/'  # To read Original Dataset
CIC_IDS_Datasets = './CICIDS/Datasets/' # To save and read Adversarial Samples
CIC_IDS_Models = './CICIDS/Models/'  # To save and read baseline Models




baseline_model = None
